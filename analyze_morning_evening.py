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
import unicodedata
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

# (nom_colonne, libellé_custom_item, code_fallback, type_polars)
MORNING_CUSTOM_SPECS: list[tuple[str, str, str, pl.DataType]] = [
    ("handgrip_left", "Handgrip Gauche", "HandgripGauche", pl.Float64),
    ("handgrip_right", "Handgrip Droit", "HandgripDroit", pl.Float64),
    ("pvt_median", "PVT Médian", "PvtMedian", pl.Float64),
    ("pvt_lapses", "PVT Lapses", "PvtLapses", pl.Int64),
    ("stroop_ies", "Stroop IES", "StroopIES", pl.Float64),
    ("cognitive_z", "Cognitive Z", "CognitiveZ", pl.Float64),
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


def _normalize_label(text: str) -> str:
    """Normalise un libellé (sans accents, minuscules, espaces condensés)."""
    decomposed = unicodedata.normalize("NFKD", text)
    no_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(no_accents.lower().split())


def _resolve_morning_custom_codes(
    session: requests.Session,
    athlete_id: str,
) -> dict[str, str]:
    """Résout dynamiquement les clés JSON custom-item vers les 6 colonnes matinales."""
    resolved: dict[str, str] = {
        col: fallback for col, _, fallback, _ in MORNING_CUSTOM_SPECS
    }
    target_by_norm = {
        _normalize_label(label): col for col, label, _, _ in MORNING_CUSTOM_SPECS
    }

    url = f"https://intervals.icu/api/v1/athlete/{athlete_id}/custom-item"
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        items = resp.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"      [WARN] Impossible de lire custom-item ({exc}) -> fallback codes par défaut.")
        return resolved

    if not isinstance(items, list):
        return resolved

    for item in items:
        if not isinstance(item, dict):
            continue
        content = item.get("content") if isinstance(item.get("content"), dict) else {}
        raw_name = item.get("name") or content.get("name")
        raw_code = content.get("code") or item.get("code")
        if not raw_name or not raw_code:
            continue
        norm_name = _normalize_label(str(raw_name))
        if norm_name in target_by_norm:
            resolved[target_by_norm[norm_name]] = str(raw_code)

    return resolved


def _extract_wellness_custom(w: dict[str, Any], code: str) -> Any:
    """Extrait un champ custom depuis la racine wellness ou son sous-dict custom."""
    val = w.get(code)
    if val is None and isinstance(w.get("custom"), dict):
        val = w["custom"].get(code)
    return val


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

    print("[1/4] Téléchargement Custom Items, Wellness & Activities depuis Intervals.icu...")
    morning_codes = _resolve_morning_custom_codes(session, athlete_id)
    print(f"      -> Clés matinales résolues : {morning_codes}")

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
            row[code] = _to_float(_extract_wellness_custom(w, code))
        for col, _, _, dtype in MORNING_CUSTOM_SPECS:
            raw_val = _extract_wellness_custom(w, morning_codes[col])
            row[col] = _to_int(raw_val) if dtype == pl.Int64 else _to_float(raw_val)
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
        **{col: dtype for col, _, _, dtype in MORNING_CUSTOM_SPECS},
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
            pl.when(
                pl.col("handgrip_left").is_not_null()
                & pl.col("handgrip_right").is_not_null()
            )
            .then((pl.col("handgrip_left") + pl.col("handgrip_right")) / 2.0)
            .otherwise(pl.coalesce(pl.col("handgrip_left"), pl.col("handgrip_right")))
            .alias("handgrip_mean"),
            pl.when(
                pl.col("handgrip_left").is_not_null()
                & pl.col("handgrip_right").is_not_null()
                & (pl.max_horizontal("handgrip_left", "handgrip_right") > 0)
            )
            .then(
                (pl.col("handgrip_left") - pl.col("handgrip_right")).abs()
                / pl.max_horizontal("handgrip_left", "handgrip_right")
                * 100.0
            )
            .otherwise(None)
            .alias("handgrip_asym_pct"),
        )
    )

    print(
        "[3/4] Calcul des métriques glissantes 28j (HRV / RHR / CogLoad / Handgrip / PVT / Stroop Z-Score) et ratio sRPE/TSS..."
    )
    df = df.with_columns(
        _rolling_mean(pl.col("hrv"), 28, 3).alias("hrv_mean_28d"),
        _rolling_std(pl.col("hrv"), 28, 3).alias("hrv_std_28d"),
        _rolling_mean(pl.col("restingHR"), 28, 3).alias("rhr_mean_28d"),
        _rolling_std(pl.col("restingHR"), 28, 3).alias("rhr_std_28d"),
        _rolling_mean(pl.col("CogLoadUA"), 28, 2).alias("cog_load_mean_28d"),
        _rolling_std(pl.col("CogLoadUA"), 28, 2).alias("cog_load_std_28d"),
        _rolling_mean(pl.col("handgrip_mean"), 28, 5).alias("handgrip_mean_28d"),
        _rolling_std(pl.col("handgrip_mean"), 28, 5).alias("handgrip_std_28d"),
        _rolling_mean(pl.col("pvt_median"), 28, 5).alias("pvt_mean_28d"),
        _rolling_std(pl.col("pvt_median"), 28, 5).alias("pvt_std_28d"),
        _rolling_mean(pl.col("stroop_ies"), 28, 5).alias("stroop_ies_mean_28d"),
        _rolling_std(pl.col("stroop_ies"), 28, 5).alias("stroop_ies_std_28d"),
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
        pl.when((pl.col("cog_load_std_28d") > 0) & pl.col("CogLoadUA").is_not_null())
        .then((pl.col("CogLoadUA") - pl.col("cog_load_mean_28d")) / pl.col("cog_load_std_28d"))
        .otherwise(None)
        .alias("cog_load_z_score_28d"),
        pl.when((pl.col("handgrip_std_28d") > 0) & pl.col("handgrip_mean").is_not_null())
        .then((pl.col("handgrip_mean") - pl.col("handgrip_mean_28d")) / pl.col("handgrip_std_28d"))
        .otherwise(None)
        .alias("handgrip_z_28d"),
        pl.when((pl.col("pvt_std_28d") > 0) & pl.col("pvt_median").is_not_null())
        .then((pl.col("pvt_median") - pl.col("pvt_mean_28d")) / pl.col("pvt_std_28d"))
        .otherwise(None)
        .alias("pvt_z_28d"),
        pl.when((pl.col("stroop_ies_std_28d") > 0) & pl.col("stroop_ies").is_not_null())
        .then((pl.col("stroop_ies") - pl.col("stroop_ies_mean_28d")) / pl.col("stroop_ies_std_28d"))
        .otherwise(None)
        .alias("stroop_ies_z_28d"),
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
        "rhr_z_score_28d",
        "sleep_hours",
        "daily_tss",
        "activity_count",
        "EveningSrpe",
        "srpe_tss_ratio",
        "MuscularStress",
        "PhysicalCap",
        "CogLoadUA",
        "cog_load_z_score_28d",
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
                pl.col("rhr_z_score_28d").round(2),
                pl.col("sleep_hours"),
                pl.col("daily_tss").round(1),
                pl.col("activity_count"),
                pl.col("EveningSrpe"),
                pl.col("srpe_tss_ratio").round(2),
                pl.col("PhysicalCap"),
                pl.col("CogLoadUA"),
                pl.col("cog_load_z_score_28d").round(2),
            ]
        )
    )

    print("\n--- Aperçu Matinal Neuro-Musculaire & Cognitif (10 derniers jours) ---")
    print(
        df.tail(10).select(
            [
                pl.col("date"),
                pl.col("handgrip_left").round(1),
                pl.col("handgrip_right").round(1),
                pl.col("handgrip_mean").round(1),
                pl.col("handgrip_asym_pct").round(1),
                pl.col("handgrip_z_28d").round(2),
                pl.col("pvt_median").round(1),
                pl.col("pvt_lapses"),
                pl.col("pvt_z_28d").round(2),
                pl.col("stroop_ies").round(1),
                pl.col("stroop_ies_z_28d").round(2),
                pl.col("cognitive_z").round(2),
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
        [
            "date",
            "daily_tss",
            "srpe_tss_ratio",
            *EVENING_CODES,
            pl.col("cog_load_z_score_28d").round(2),
        ]
    )
    print("\n--- Journées avec champs vespéraux renseignés ---")
    print(evening_rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())