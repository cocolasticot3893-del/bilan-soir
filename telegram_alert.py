#!/usr/bin/env python3
"""Pipeline de notification matinale Telegram à partir de telemetry.duckdb."""
from __future__ import annotations

import datetime
import html
import json
import math
import pathlib
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal

import duckdb

BASE_DIR = pathlib.Path(r"C:\Dev\bilan-soir")
ANALYZE_SCRIPT = BASE_DIR / "analyze_morning_evening.py"
DUCKDB_PATH = BASE_DIR / "data" / "telemetry.duckdb"
TELEGRAM_CREDS_PATH = BASE_DIR / "data" / "telegram_credentials.json"

TELEGRAM_API_BASE = "https://api.telegram.org"
HTTP_TIMEOUT_SECS = 10
MAX_ATTEMPTS = 3
BASE_BACKOFF_SECS = 1.0

Severity = Literal["CRITICAL", "WARNING", "NOMINAL"]

REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "hrv",
    "hrv_mean_28d",
    "hrv_std_28d",
    "hrv_z_score_28d",
    "restingHR",
    "rhr_mean_28d",
    "rhr_std_28d",
    "handgrip_left",
    "handgrip_right",
    "handgrip_mean",
    "handgrip_asym_pct",
    "handgrip_z_28d",
    "pvt_median",
    "pvt_lapses",
    "pvt_z_28d",
    "stroop_ies",
    "stroop_ies_z_28d",
    "cognitive_z",
    "EveningSrpe",
    "MuscularStress",
    "PhysicalCap",
    "CogLoadUA",
    "FrustrationTilt",
    "FuelingRecov",
    "SleepHygiene",
    "srpe_tss_ratio",
)

_TOKEN_URL_PATTERN = re.compile(r"bot[0-9]+:[A-Za-z0-9_-]+")


@dataclass(frozen=True, slots=True)
class TelegramCredentials:
    """Identifiants Telegram chargés depuis le fichier local sécurisé."""

    bot_token: str
    chat_id_info: str
    chat_id_error: str


@dataclass(frozen=True, slots=True)
class Alert:
    """Alerte physiologique ou cognitive détectée par les règles métier."""

    code: str
    severity: Literal["CRITICAL", "WARNING"]
    title: str
    reasons: tuple[str, ...]
    action: str


@dataclass(frozen=True, slots=True)
class TelemetryContext:
    """Snapshot biométrique du jour, de la veille et baseline 28 jours."""

    today_str: str
    yesterday_str: str
    today_row: dict[str, float | None]
    yesterday_row: dict[str, float | None]
    srpe_tss_ratio_mean_28d: float | None


def _sanitize_secret(text: str, bot_token: str) -> str:
    """Masque systématiquement le token Telegram dans tout message ou URL."""
    cleaned = text.replace(bot_token, "***REDACTED***") if bot_token else text
    return _TOKEN_URL_PATTERN.sub("bot***REDACTED***", cleaned)


def _to_clean_float(value: Any) -> float | None:
    """Convertit une valeur SQL en float valide, ou None si nulle/NaN/infinie."""
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def _fmt(value: float | None, decimals: int = 1, suffix: str = "") -> str:
    """Formate un float pour l'affichage HTML avec repli 'N/D'."""
    if value is None:
        return "N/D"
    return f"{value:.{decimals}f}{suffix}"


def _fmt_int_like(value: float | None) -> str:
    """Formate une métrique discrète entière si exacte, sinon à 1 décimale."""
    if value is None:
        return "N/D"
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}"


def load_telegram_credentials(
    creds_path: pathlib.Path = TELEGRAM_CREDS_PATH,
) -> TelegramCredentials:
    """Charge les identifiants Telegram sans exposer leur contenu."""
    if not creds_path.exists():
        raise FileNotFoundError(f"Fichier d'identifiants introuvable : {creds_path}")
    raw = json.loads(creds_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Format JSON invalide dans le fichier d'identifiants Telegram.")

    missing = [
        key for key in ("bot_token", "chat_id_info", "chat_id_error") if not raw.get(key)
    ]
    if missing:
        raise KeyError(f"Clés manquantes dans les identifiants Telegram : {', '.join(missing)}")

    return TelegramCredentials(
        bot_token=str(raw["bot_token"]).strip(),
        chat_id_info=str(raw["chat_id_info"]).strip(),
        chat_id_error=str(raw["chat_id_error"]).strip(),
    )


def refresh_telemetry(script_path: pathlib.Path = ANALYZE_SCRIPT) -> None:
    """Exécute le pipeline amont pour rafraîchir telemetry.duckdb."""
    subprocess.run([sys.executable, str(script_path)], check=True)


def _fetch_row_by_date(
    con: duckdb.DuckDBPyConnection,
    available_cols: set[str],
    target_date: str,
) -> dict[str, float | None]:
    """Lit une ligne de daily_telemetry en ne requêtant que les colonnes existantes."""
    select_cols = [col for col in REQUIRED_COLUMNS if col != "date" and col in available_cols]
    row_dict: dict[str, float | None] = {
        col: None for col in REQUIRED_COLUMNS if col != "date"
    }
    if not select_cols:
        return row_dict

    quoted_cols = ", ".join(f'"{col}"' for col in select_cols)
    query = f"SELECT {quoted_cols} FROM daily_telemetry WHERE date = ? LIMIT 1"
    result = con.execute(query, [target_date]).fetchone()
    if result is None:
        return row_dict

    for col, val in zip(select_cols, result, strict=True):
        row_dict[col] = _to_clean_float(val)
    return row_dict


def load_telemetry_context(
    db_path: pathlib.Path = DUCKDB_PATH,
    today: datetime.date | None = None,
) -> TelemetryContext:
    """Inspecte daily_telemetry et extrait le contexte matin/veille/28j."""
    if not db_path.exists():
        raise FileNotFoundError(f"Base DuckDB introuvable : {db_path}")

    ref_today = today or datetime.date.today()
    ref_yesterday = ref_today - datetime.timedelta(days=1)
    start_28d = ref_yesterday - datetime.timedelta(days=27)

    today_str = ref_today.isoformat()
    yesterday_str = ref_yesterday.isoformat()

    with duckdb.connect(str(db_path), read_only=True) as con:
        pragma_rows = con.execute("PRAGMA table_info('daily_telemetry')").fetchall()
        available_cols = {str(row[1]) for row in pragma_rows}
        if "date" not in available_cols:
            raise RuntimeError("La table 'daily_telemetry' ne contient pas la colonne 'date'.")

        today_row = _fetch_row_by_date(con, available_cols, today_str)
        yesterday_row = _fetch_row_by_date(con, available_cols, yesterday_str)

        srpe_tss_ratio_mean_28d: float | None = None
        if "srpe_tss_ratio" in available_cols:
            avg_row = con.execute(
                """
                SELECT AVG(srpe_tss_ratio)
                FROM daily_telemetry
                WHERE date >= ? AND date <= ? AND srpe_tss_ratio IS NOT NULL
                """,
                [start_28d.isoformat(), yesterday_str],
            ).fetchone()
            if avg_row is not None:
                srpe_tss_ratio_mean_28d = _to_clean_float(avg_row[0])

    return TelemetryContext(
        today_str=today_str,
        yesterday_str=yesterday_str,
        today_row=today_row,
        yesterday_row=yesterday_row,
        srpe_tss_ratio_mean_28d=srpe_tss_ratio_mean_28d,
    )


def evaluate_alerts(ctx: TelemetryContext) -> list[Alert]:
    """Applique les règles croisées Matin (J) + Soir (J-1) avec gardes stricts."""
    alerts: list[Alert] = []
    t = ctx.today_row
    y = ctx.yesterday_row

    hrv_z = t.get("hrv_z_score_28d")
    rhr_today = t.get("restingHR")
    rhr_mean_28d = t.get("rhr_mean_28d")
    handgrip_z = t.get("handgrip_z_28d")
    handgrip_asym = t.get("handgrip_asym_pct")
    muscular_y = y.get("MuscularStress")
    physical_cap_y = y.get("PhysicalCap")

    # Règle 1 : CRITICAL (SNC / SNA / Neuromusculaire)
    r1_reasons: list[str] = []
    if hrv_z is not None and hrv_z <= -1.5:
        r1_reasons.append(f"HRV Z-score 28j = {hrv_z:.2f} (<= -1.50)")
    if rhr_today is not None and rhr_mean_28d is not None:
        rhr_delta = rhr_today - rhr_mean_28d
        if rhr_delta >= 5.0:
            r1_reasons.append(
                f"RHR matin = {rhr_today:.1f} bpm (+{rhr_delta:.1f} vs moy. 28j {rhr_mean_28d:.1f})"
            )
    if handgrip_z is not None and handgrip_z <= -1.5:
        r1_reasons.append(f"Handgrip Z-score 28j = {handgrip_z:.2f} (<= -1.50)")
    if handgrip_asym is not None and handgrip_asym >= 12.0:
        r1_reasons.append(f"Asymétrie Handgrip = {handgrip_asym:.1f}% (>= 12.0%)")
    if (
        muscular_y is not None
        and physical_cap_y is not None
        and muscular_y >= 5.0
        and physical_cap_y <= 1.0
    ):
        r1_reasons.append(
            f"MuscularStress veille = {_fmt_int_like(muscular_y)} (>= 5) & PhysicalCap = {_fmt_int_like(physical_cap_y)} (<= 1)"
        )

    if r1_reasons:
        alerts.append(
            Alert(
                code="SNC_SNA_CRITICAL",
                severity="CRITICAL",
                title="ALERTE CRITIQUE SNC/SNA",
                reasons=tuple(r1_reasons),
                action="Annuler tout travail > SV1, basculer sur récupération passive/active légère et couper le volume Expresso.",
            )
        )

    # Règle 2 : WARNING — CORTEX / POKER
    pvt_z = t.get("pvt_z_28d")
    pvt_lapses = t.get("pvt_lapses")
    stroop_z = t.get("stroop_ies_z_28d")
    cognitive_z = t.get("cognitive_z")
    cog_load_y = y.get("CogLoadUA")
    frust_y = y.get("FrustrationTilt")
    sleep_hyg_y = y.get("SleepHygiene")

    r2_reasons: list[str] = []
    if pvt_z is not None and pvt_z >= 1.2:
        r2_reasons.append(f"PVT Z-score 28j = {pvt_z:.2f} (>= 1.20)")
    if pvt_lapses is not None and pvt_lapses >= 2.0:
        r2_reasons.append(f"PVT Lapses = {_fmt_int_like(pvt_lapses)} (>= 2)")
    if stroop_z is not None and stroop_z >= 1.2:
        r2_reasons.append(f"Stroop IES Z-score 28j = {stroop_z:.2f} (>= 1.20)")
    if cognitive_z is not None and cognitive_z <= -1.0:
        r2_reasons.append(f"Cognitive Z = {cognitive_z:.2f} (<= -1.00)")
    if cog_load_y is not None and cog_load_y >= 30.0:
        r2_reasons.append(f"CogLoadUA veille = {cog_load_y:.1f} UA (>= 30.0)")
    if frust_y is not None and frust_y >= 6.0:
        r2_reasons.append(f"FrustrationTilt veille = {_fmt_int_like(frust_y)} (>= 6)")
    if (
        sleep_hyg_y is not None
        and hrv_z is not None
        and sleep_hyg_y == 0.0
        and hrv_z < -0.5
    ):
        r2_reasons.append(
            f"SleepHygiene veille = 0 & HRV Z-score 28j = {hrv_z:.2f} (< -0.50)"
        )

    if r2_reasons:
        alerts.append(
            Alert(
                code="CORTICAL_SATURATION",
                severity="WARNING",
                title="WARNING — CORTEX / POKER",
                reasons=tuple(r2_reasons),
                action="Restreindre ou annuler le grind Expresso (risque de perte d'EV par défaut d'inhibition préfrontale), privilégier la review solver/GTO.",
            )
        )

    # Règle 3 : WARNING — MÉTABOLIQUE / GLYCOGÈNE
    fuel_y = y.get("FuelingRecov")
    srpe_ratio_y = y.get("srpe_tss_ratio")
    srpe_ratio_mean = ctx.srpe_tss_ratio_mean_28d

    r3_reasons: list[str] = []
    if (
        fuel_y is not None
        and physical_cap_y is not None
        and fuel_y == 1.0
        and physical_cap_y <= 2.0
    ):
        r3_reasons.append(
            f"FuelingRecov veille = 1 & PhysicalCap = {_fmt_int_like(physical_cap_y)} (<= 2)"
        )
    if (
        srpe_ratio_y is not None
        and srpe_ratio_mean is not None
        and srpe_ratio_mean > 0.0
        and srpe_ratio_y > 1.25 * srpe_ratio_mean
    ):
        r3_reasons.append(
            f"sRPE/TSS veille = {srpe_ratio_y:.2f} (> 1.25 × moy. 28j {srpe_ratio_mean:.2f})"
        )

    if r3_reasons:
        alerts.append(
            Alert(
                code="GLYCOGEN_INTERNAL_COST",
                severity="WARNING",
                title="WARNING — MÉTABOLIQUE / GLYCOGÈNE",
                reasons=tuple(r3_reasons),
                action="Recharge glucidique prioritaire (8-10 g/kg/j) avant toute intensité SV2 / VO2max.",
            )
        )

    return alerts


def _format_biometric_context_lines(ctx: TelemetryContext) -> list[str]:
    """Construit les lignes HTML du tableau de bord biométrique matin + veille."""
    t = ctx.today_row
    y = ctx.yesterday_row

    hrv_str = _fmt(t.get("hrv"), 1, " ms")
    hrv_z_str = _fmt(t.get("hrv_z_score_28d"), 2)
    hrv_mean_str = _fmt(t.get("hrv_mean_28d"), 1, " ms")

    rhr_today = t.get("restingHR")
    rhr_mean = t.get("rhr_mean_28d")
    rhr_delta_str = ""
    if rhr_today is not None and rhr_mean is not None:
        delta = rhr_today - rhr_mean
        rhr_delta_str = f" (Δ {delta:+.1f} vs {rhr_mean:.1f})"

    hg_l = _fmt(t.get("handgrip_left"), 1)
    hg_r = _fmt(t.get("handgrip_right"), 1)
    hg_mean = _fmt(t.get("handgrip_mean"), 1, " kg")
    hg_asym = _fmt(t.get("handgrip_asym_pct"), 1, "%")
    hg_z = _fmt(t.get("handgrip_z_28d"), 2)

    pvt_med = _fmt(t.get("pvt_median"), 0, " ms")
    pvt_lap = _fmt_int_like(t.get("pvt_lapses"))
    pvt_z = _fmt(t.get("pvt_z_28d"), 2)

    stroop_ies = _fmt(t.get("stroop_ies"), 1)
    stroop_z = _fmt(t.get("stroop_ies_z_28d"), 2)
    cog_z = _fmt(t.get("cognitive_z"), 2)

    srpe_ratio_str = _fmt(y.get("srpe_tss_ratio"), 2)
    srpe_mean_str = _fmt(ctx.srpe_tss_ratio_mean_28d, 2)

    return [
        f"<b>Matin ({html.escape(ctx.today_str)})</b>",
        f"• HRV : <code>{hrv_str}</code> | Z-28j : <code>{hrv_z_str}</code> (moy. 28j : {hrv_mean_str})",
        f"• RHR : <code>{_fmt(rhr_today, 1, ' bpm')}</code>{html.escape(rhr_delta_str)}",
        (
            f"• Handgrip G/D : <code>{hg_l}</code> / <code>{hg_r}</code> kg "
            f"(moy. {hg_mean}, asym. <code>{hg_asym}</code>, Z-28j : <code>{hg_z}</code>)"
        ),
        f"• PVT : <code>{pvt_med}</code> | Lapses : <code>{pvt_lap}</code> | Z-28j : <code>{pvt_z}</code>",
        (
            f"• Stroop IES : <code>{stroop_ies}</code> (Z-28j : <code>{stroop_z}</code>) | "
            f"Cognitive Z : <code>{cog_z}</code>"
        ),
        "",
        f"<b>Bilan vespéral veille ({html.escape(ctx.yesterday_str)})</b>",
        (
            f"• EveningSrpe : <code>{_fmt_int_like(y.get('EveningSrpe'))}</code> | "
            f"PhysicalCap : <code>{_fmt_int_like(y.get('PhysicalCap'))}</code> | "
            f"MuscularStress : <code>{_fmt_int_like(y.get('MuscularStress'))}</code>"
        ),
        (
            f"• CogLoadUA : <code>{_fmt(y.get('CogLoadUA'), 1)}</code> | "
            f"FrustrationTilt : <code>{_fmt_int_like(y.get('FrustrationTilt'))}</code>"
        ),
        (
            f"• FuelingRecov : <code>{_fmt_int_like(y.get('FuelingRecov'))}</code> | "
            f"SleepHygiene : <code>{_fmt_int_like(y.get('SleepHygiene'))}</code> | "
            f"sRPE/TSS : <code>{srpe_ratio_str}</code> (moy. 28j : {srpe_mean_str})"
        ),
    ]


def build_html_message(ctx: TelemetryContext, alerts: list[Alert]) -> tuple[Severity, str]:
    """Assemble le message Telegram HTML et détermine la sévérité globale."""
    context_lines = _format_biometric_context_lines(ctx)

    if not alerts:
        lines = [
            f"🟢 <b>STATUT NOMINAL — Tableau de bord matinal ({html.escape(ctx.today_str)})</b>",
            "",
            *context_lines,
            "",
            "<b>Décision :</b> Feu vert entraînement (intensité selon plan) et poker (grind Expresso).",
        ]
        return "NOMINAL", "\n".join(lines)

    has_critical = any(alert.severity == "CRITICAL" for alert in alerts)
    global_severity: Severity = "CRITICAL" if has_critical else "WARNING"
    header_icon = "🔴" if has_critical else "🟠"
    header_label = "ALERTE CRITIQUE" if has_critical else "VIGILANCE"

    lines = [
        f"{header_icon} <b>{header_label} — Bilan Matinal ({html.escape(ctx.today_str)})</b>",
        "",
        *context_lines,
        "",
        "<b>Alertes détectées :</b>",
    ]
    for alert in alerts:
        badge = "🔴" if alert.severity == "CRITICAL" else "🟠"
        lines.append(f"\n{badge} <b>{html.escape(alert.title)}</b>")
        for reason in alert.reasons:
            lines.append(f"  - {html.escape(reason)}")
        lines.append(f"  → <i>Action :</i> {html.escape(alert.action)}")

    return global_severity, "\n".join(lines)


def send_telegram_message(
    creds: TelegramCredentials,
    chat_id: str,
    html_text: str,
    *,
    max_attempts: int = MAX_ATTEMPTS,
    timeout_secs: int = HTTP_TIMEOUT_SECS,
) -> dict[str, Any]:
    """Envoie un message HTML via l'API Bot Telegram avec retry exponentiel."""
    url = f"{TELEGRAM_API_BASE}/bot{creds.bot_token}/sendMessage"
    payload = json.dumps(
        {
            "chat_id": chat_id,
            "text": html_text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
        ensure_ascii=False,
    ).encode("utf-8")

    headers = {
        "User-Agent": "BilanSoirAlert/1.0",
        "Content-Type": "application/json",
    }

    last_error = "Erreur inconnue"
    for attempt in range(max_attempts):
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        retry_delay: float | None = None

        try:
            with urllib.request.urlopen(req, timeout=timeout_secs) as resp:
                raw_body = resp.read().decode("utf-8", errors="replace")
            parsed = json.loads(raw_body)
            if not isinstance(parsed, dict) or parsed.get("ok") is not True:
                safe_body = _sanitize_secret(raw_body, creds.bot_token)
                raise RuntimeError(f"Réponse Telegram invalide (ok != true) : {safe_body}")

            result = parsed.get("result")
            message_id = result.get("message_id") if isinstance(result, dict) else None
            return {
                "chat_id": chat_id,
                "ok": True,
                "message_id": message_id,
            }

        except urllib.error.HTTPError as exc:
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except OSError:
                err_body = ""
            safe_msg = _sanitize_secret(
                f"HTTP {exc.code} {exc.reason} {err_body}".strip(), creds.bot_token
            )
            last_error = safe_msg
            is_retryable = exc.code == 429 or 500 <= exc.code < 600
            if not is_retryable or attempt >= max_attempts - 1:
                raise RuntimeError(
                    f"Échec envoi Telegram vers chat_id={chat_id} : {safe_msg}"
                ) from None

            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            if exc.code == 429 and retry_after and retry_after.isdigit():
                retry_delay = float(retry_after) + random.uniform(0.1, 0.5)
            else:
                retry_delay = BASE_BACKOFF_SECS * (2**attempt) + random.uniform(0.1, 0.6)

        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            safe_msg = _sanitize_secret(str(exc), creds.bot_token)
            last_error = safe_msg
            if attempt >= max_attempts - 1:
                raise RuntimeError(
                    f"Erreur réseau Telegram vers chat_id={chat_id} : {safe_msg}"
                ) from None
            retry_delay = BASE_BACKOFF_SECS * (2**attempt) + random.uniform(0.1, 0.6)

        except json.JSONDecodeError as exc:
            safe_msg = _sanitize_secret(str(exc), creds.bot_token)
            raise RuntimeError(
                f"Réponse JSON Telegram illisible pour chat_id={chat_id} : {safe_msg}"
            ) from None

        if retry_delay is not None:
            time.sleep(retry_delay)

    raise RuntimeError(
        f"Échec envoi Telegram vers chat_id={chat_id} après {max_attempts} tentatives : {last_error}"
    )


def resolve_target_chats(
    creds: TelegramCredentials,
    severity: Severity,
) -> list[str]:
    """Détermine la liste ordonnée des chat_id destinataires selon la sévérité."""
    targets = [creds.chat_id_info]
    if severity == "CRITICAL" and creds.chat_id_error not in targets:
        targets.append(creds.chat_id_error)
    return targets


def main() -> int:
    """Point d'entrée : rafraîchissement, détection des règles et notification Telegram."""
    creds = load_telegram_credentials()
    refresh_telemetry()
    ctx = load_telemetry_context()
    alerts = evaluate_alerts(ctx)
    severity, html_message = build_html_message(ctx, alerts)
    target_chats = resolve_target_chats(creds, severity)

    for chat_id in target_chats:
        ack = send_telegram_message(creds, chat_id, html_message)
        print(json.dumps(ack, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
