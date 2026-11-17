#!/usr/bin/env python3
"""
Pipeline Polars + DuckDB d'analyse croisée Matin (HRV, RHR, Sommeil) /
Entraînement (TSS, Durée) / Soir (11 champs Bilan Vespéral) sur 60 jours.
"""
from __future__ import annotations

import datetime
import inspect
import json
import pathlib
import sys
from typing import Any

import duckdb
import polars as pl
import requests

BASE_DIR = pathlib.Path(r"C:\Dev\bilan-soir")
CREDS_PATH = BASE_DIR / "data" / "icu_credentials.json"
DUCKDB_PATH = BASE_DIR / "data" / "telemetry.duckdb"

EVENING_CODES: list[str] = [
    "EveningSrpe",
    "MuscularStress",
    "PhysicalCap",
    "CogHours",
    "MentalDemand",
    "CogLoadUA",
    "FrustrationTilt",
    "OverallStress",
    "EmotionalBalance",
    "FuelingRecov",
    "SleepHygiene",
]


def load_credentials() -> tuple[str, str]:
    if not CREDS_PATH.exists():
        raise FileNotFoundError(f"Credentials file not found: {CREDS_PATH}")
    data = json.loads(CREDS_PATH.read_text(encoding="utf-8"))
    return str(data["athlete_id"]).strip(), str(data["api_key"]).strip()


def _rolling_mean(expr: pl.Expr, window_size: int, min_s: int = 3) -> pl.Expr:
    sig = inspect.signature(pl.Expr.rolling_mean)
    if "min_samples" in sig.parameters:
        return expr.rolling_mean(window_size=window_size, min_samples=min_s)
    return expr.rolling_mean(window_size=window_size, min_periods=min_s)


def _rolling_std(expr: pl.Expr, window_size: int, min_s: int = 3) -> pl.Expr:
    sig = inspect.signature(pl.Expr.rolling_std)
    if "min_samples" in sig.parameters:
        return expr.rolling_std(window_size=window_size, min_samples=min_s)
    return expr.rolling_std(window_size=window_size, min_periods=min_s)


def _to_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _to_int(val: Any) -> int | None:
    if val is None or val == "":
        return None
    try:
        return int(round(float(val)))
    except (ValueError, TypeError):
        return None


def main() -> int:
    athlete_id, api_key = load_credentials()
    session = requests.Session()
    session.auth = ("API_KEY", api_key)
    session.headers.update(
        {
            "User-Agent": "BilanSoirPWA/1.0",
            "Accept": "application/json",
        }
    )

    today = datetime.date.today()
    oldest_date = today - datetime.timedelta(days=59)
    oldest_str = oldest_date.isoformat()
    newest_str = today.isoformat()

    print(f"=== Pipeline Polars + DuckDB : {oldest_str} -> {newest_str} (60 jours) ===")

    wellness_url = f"https://intervals.icu/api/v1/athlete/{athlete_id}/wellness?oldest={oldest_str}&newest={newest_str}"
    activities_url = f"https://intervals.icu/api/v1/athlete/{athlete_id}/activities?oldest={oldest_str}&newest={newest_str}"

    print(f"[1/4] Téléchargement Wellness & Activities depuis Intervals.icu...")
    w_resp = session.get(wellness_url, timeout=30)
    w_resp.raise_for_status()
    wellness_raw: list[dict[str, Any]] = w_resp.json()

    a_resp = session.get(activities_url, timeout=30)
    a_resp.raise_for_status()
    activities_raw: list[dict[str, Any]] = a_resp.json()

    print(f"      -> {len(wellness_raw)} entrées wellness reçues.")
    print(f"      -> {len(activities_raw)} activités reçues.")

    print("[2/4] Agrégation Polars des activités et jointure Wellness...")
    date_spine = pl.DataFrame(
        {
            "date": [
                (oldest_date + datetime.timedelta(days=i)).isoformat()
                for i in range(60)
            ]
        }
    )

    act_rows: list[dict[str, Any]] = []
    for act in activities_raw:
        if not isinstance(act, dict):
            continue
        start_local = str(act.get("start_date_local") or act.get("start_date") or "")
        if len(start_local) < 10:
            continue
        act_rows.append(
            {
                "date": start_local[:10],
                "icu_training_load": _to_float(act.get("icu_training_load")) or 0.0,
                "moving_time": _to_int(act.get("moving_time")) or 0,
            }
        )

    if act_rows:
        act_df = (
            pl.DataFrame(act_rows)
            .group_by("date")
            .agg(
                pl.col("icu_training_load").cast(pl.Float64).sum().alias("daily_tss"),
                pl.col("moving_time").cast(pl.Int64).sum().alias("daily_duration_secs"),
                pl.len().cast(pl.Int64).alias("activity_count"),
            )
        )
    else:
        act_df = pl.DataFrame(
            schema={
                "date": pl.String,
                "daily_tss": pl.Float64,
                "daily_duration_secs": pl.Int64,
                "activity_count": pl.Int64,
            }
        )

    well_rows: list[dict[str, Any]] = []
    for w in wellness_raw:
        if not isinstance(w, dict):
            continue
        d_str = str(w.get("id") or "")
        if len(d_str) < 10:
            continue
        row: dict[str, Any] = {
            "date": d_str[:10],
            "hrv": _to_float(w.get("hrv")),
            "restingHR": _to_float(w.get("restingHR")),
            "sleepSecs": _to_int(w.get("sleepSecs")),
            "soreness": _to_int(w.get("soreness")),
            "fatigue": _to_int(w.get("fatigue")),
            "stress": _to_int(w.get("stress")),
            "ctl": _to_float(w.get("ctl")),
            "atl": _to_float(w.get("atl")),
            "rampRate": _to_float(w.get("rampRate")),
        }
        for code in EVENING_CODES:
            val = w.get(code)
            if val is None and isinstance(w.get("custom"), dict):
                val = w["custom"].get(code)
            row[code] = _to_float(val)
        well_rows.append(row)

    well_schema: dict[str, pl.DataType] = {
        "date": pl.String,
        "hrv": pl.Float64,
        "restingHR": pl.Float64,
        "sleepSecs": pl.Int64,
        "soreness": pl.Int64,
        "fatigue": pl.Int64,
        "stress": pl.Int64,
        "ctl": pl.Float64,
        "atl": pl.Float64,
        "rampRate": pl.Float64,
        **{code: pl.Float64 for code in EVENING_CODES},
    }

    if well_rows:
        well_df = pl.DataFrame(well_rows, schema=well_schema)
    else:
        well_df = pl.DataFrame(schema=well_schema)

    df = (
        date_spine.join(well_df, on="date", how="left")
        .join(act_df, on="date", how="left")
        .sort("date")
        .with_columns(
            pl.col("daily_tss").fill_null(0.0),
            pl.col("daily_duration_secs").fill_null(0),
            pl.col("activity_count").fill_null(0),
            (pl.col("sleepSecs").cast(pl.Float64) / 3600.0).round(2).alias("sleep_hours"),
        )
    )

    print("[3/4] Calcul des métriques glissantes 28j (HRV / RHR Z-Score) et ratio sRPE/TSS...")
    df = df.with_columns(
        _rolling_mean(pl.col("hrv"), 28, 3).alias("hrv_mean_28d"),
        _rolling_std(pl.col("hrv"), 28, 3).alias("hrv_std_28d"),
        _rolling_mean(pl.col("restingHR"), 28, 3).alias("rhr_mean_28d"),
        _rolling_std(pl.col("restingHR"), 28, 3).alias("rhr_std_28d"),
        pl.when((pl.col("daily_tss") > 0) & pl.col("EveningSrpe").is_not_null())
        .then(pl.col("EveningSrpe") / (pl.col("daily_tss") / 100.0))
        .otherwise(None)
        .alias("srpe_tss_ratio"),
        pl.when((pl.col("daily_tss") > 0) & pl.col("EveningSrpe").is_not_null())
        .then(pl.col("EveningSrpe") / pl.col("daily_tss"))
        .otherwise(None)
        .alias("srpe_per_tss"),
    ).with_columns(
        pl.when((pl.col("hrv_std_28d") > 0) & pl.col("hrv").is_not_null())
        .then((pl.col("hrv") - pl.col("hrv_mean_28d")) / pl.col("hrv_std_28d"))
        .otherwise(None)
        .alias("hrv_z_score_28d"),
        pl.when((pl.col("rhr_std_28d") > 0) & pl.col("restingHR").is_not_null())
        .then((pl.col("restingHR") - pl.col("rhr_mean_28d")) / pl.col("rhr_std_28d"))
        .otherwise(None)
        .alias("rhr_z_score_28d"),
    )

    print(f"[4/4] Enregistrement dans DuckDB : {DUCKDB_PATH} (table: daily_telemetry)...")
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_parquet = DUCKDB_PATH.with_suffix(".parquet")
    df.write_parquet(tmp_parquet)
    try:
        with duckdb.connect(str(DUCKDB_PATH)) as con:
            con.execute(
                "CREATE OR REPLACE TABLE daily_telemetry AS SELECT * FROM read_parquet(?)",
                [str(tmp_parquet)],
            )
            row_count = con.execute("SELECT COUNT(*) FROM daily_telemetry").fetchone()[0]
            evening_count = con.execute("SELECT COUNT(*) FROM daily_telemetry WHERE EveningSrpe IS NOT NULL").fetchone()[0]
    finally:
        tmp_parquet.unlink(missing_ok=True)

    print(f"\n=== DIAGNOSTIC COMPLET (daily_telemetry : {row_count} lignes, {evening_count} avec Bilan Soir) ===")
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(160)
    pl.Config.set_tbl_rows(15)

    summary_cols = [
        "date",
        "hrv",
        "hrv_mean_28d",
        "hrv_z_score_28d",
        "restingHR",
        "sleep_hours",
        "daily_tss",
        "activity_count",
        "EveningSrpe",
        "srpe_tss_ratio",
        "MuscularStress",
        "PhysicalCap",
        "CogLoadUA",
        "OverallStress",
    ]
    print("\n--- Aperçu des 10 derniers jours ---")
    print(
        df.tail(10).select(
            [
                pl.col("date"),
                pl.col("hrv").round(1),
                pl.col("hrv_mean_28d").round(1),
                pl.col("hrv_z_score_28d").round(2),
                pl.col("restingHR").round(1),
                pl.col("sleep_hours"),
                pl.col("daily_tss").round(1),
                pl.col("activity_count"),
                pl.col("EveningSrpe"),
                pl.col("srpe_tss_ratio").round(2),
                pl.col("PhysicalCap"),
                pl.col("CogLoadUA"),
            ]
        )
    )

    stats = df.select(
        pl.len().alias("days_total"),
        pl.col("hrv").drop_nulls().len().alias("hrv_days"),
        pl.col("hrv").mean().round(2).alias("hrv_avg"),
        pl.col("hrv_z_score_28d").min().round(2).alias("hrv_z_min"),
        pl.col("hrv_z_score_28d").max().round(2).alias("hrv_z_max"),
        pl.col("restingHR").mean().round(1).alias("rhr_avg"),
        pl.col("daily_tss").sum().round(1).alias("total_tss_60d"),
        pl.col("daily_tss").mean().round(1).alias("avg_daily_tss"),
        pl.col("activity_count").sum().alias("total_activities"),
        pl.col("EveningSrpe").drop_nulls().len().alias("evening_logged_days"),
    )
    print("\n--- Statistiques Globales 60 Jours ---")
    print(stats)

    evening_rows = df.filter(pl.col("EveningSrpe").is_not_null()).select(
        ["date", "daily_tss", "srpe_tss_ratio", *EVENING_CODES]
    )
    print("\n--- Journées avec champs vespéraux renseignés ---")
    print(evening_rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())