#!/usr/bin/env python3
"""
Idempotent provisioning and verification of the 11 Evening Wellness custom INPUT_FIELDs
on Intervals.icu for the Bilan Vespéral PWA.
"""
from __future__ import annotations

import copy
import datetime
import json
import pathlib
import sys
import unicodedata
from typing import Any

import requests

BASE_DIR = pathlib.Path(r"C:\Dev\bilan-soir")
CREDS_PATH = BASE_DIR / "data" / "icu_credentials.json"

EVENING_FIELDS: list[dict[str, Any]] = [
    {"code": "EveningSrpe", "name": "Soir: Effort (sRPE)", "units": "/10", "max": 10.0, "fmt": ".0f", "description": "Effort perçu global de la séance ou journée sportive (0-10)"},
    {"code": "MuscularStress", "name": "Soir: Contrainte Musculaire", "units": "/6", "max": 6.0, "fmt": ".0f", "description": "Tension / douleur musculaire vespérale (0-6)"},
    {"code": "PhysicalCap", "name": "Soir: Capacite Physique", "units": "/6", "max": 6.0, "fmt": ".0f", "description": "Capacité physique résiduelle perçue le soir (0-6)"},
    {"code": "CogHours", "name": "Soir: Volume Cognitif (h)", "units": "h", "max": 16.0, "fmt": ".1f", "description": "Volume de travail cognitif intense en heures"},
    {"code": "MentalDemand", "name": "Soir: Exigence Mentale", "units": "/10", "max": 10.0, "fmt": ".0f", "description": "Densité et exigence mentale de la journée (0-10)"},
    {"code": "CogLoadUA", "name": "Soir: Charge Cognitive (UA)", "units": "UA", "max": 160.0, "fmt": ".1f", "description": "Charge cognitive calculée (CogHours x MentalDemand)"},
    {"code": "FrustrationTilt", "name": "Soir: Frustration & Tilt", "units": "/10", "max": 10.0, "fmt": ".0f", "description": "Niveau d'irritation, frustration ou tilt en fin de journée (0-10)"},
    {"code": "OverallStress", "name": "Soir: Stress General", "units": "/6", "max": 6.0, "fmt": ".0f", "description": "Stress global extra-sportif et nerveux (0-6)"},
    {"code": "EmotionalBalance", "name": "Soir: Equilibre Emotionnel", "units": "/6", "max": 6.0, "fmt": ".0f", "description": "Stabilité de l'humeur et équilibre émotionnel (0-6)"},
    {"code": "FuelingRecov", "name": "Soir: Nutrition & Recup", "units": "/3", "max": 3.0, "fmt": ".0f", "description": "Qualité de l'apport énergétique et hydratation (0-3)"},
    {"code": "SleepHygiene", "name": "Soir: Hygiene Pre-Sommeil", "units": "bool", "max": 1.0, "fmt": ".0f", "description": "Hygiène pré-sommeil (1 = Aucun perturbateur, 0 = Perturbateur présent)"},
]

TEST_VALUES: dict[str, int | float] = {
    "EveningSrpe": 5,
    "MuscularStress": 2,
    "PhysicalCap": 4,
    "CogHours": 6.5,
    "MentalDemand": 6,
    "CogLoadUA": 39.0,
    "FrustrationTilt": 2,
    "OverallStress": 2,
    "EmotionalBalance": 4,
    "FuelingRecov": 2,
    "SleepHygiene": 1,
}


EVENING_CODES: set[str] = {spec["code"] for spec in EVENING_FIELDS}
SERVER_DROP_KEYS: tuple[str, ...] = ("id", "updated", "athlete_id", "usage_count", "from_athlete", "from_id")
TARGET_TEMPLATE_NAMES: set[str] = {"handgrip gauche", "pvt median"}


def load_credentials() -> tuple[str, str]:
    if not CREDS_PATH.exists():
        raise FileNotFoundError(f"Credentials file not found: {CREDS_PATH}")
    data = json.loads(CREDS_PATH.read_text(encoding="utf-8"))
    return str(data["athlete_id"]).strip(), str(data["api_key"]).strip()


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    decomposed = unicodedata.normalize("NFKD", value)
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return without_accents.casefold().strip()


def extract_code(item: dict[str, Any]) -> str | None:
    top_code = item.get("code")
    if top_code:
        return str(top_code)
    content = item.get("content")
    if isinstance(content, dict) and content.get("code"):
        return str(content["code"])
    return None


def extract_name(item: dict[str, Any]) -> str | None:
    top_name = item.get("name")
    if isinstance(top_name, str) and top_name.strip():
        return top_name
    content = item.get("content")
    if isinstance(content, dict) and isinstance(content.get("name"), str) and content["name"].strip():
        return content["name"]
    return None


def extract_units(item: dict[str, Any]) -> Any:
    for key in ("units", "suffix"):
        if item.get(key) is not None:
            return item.get(key)
    content = item.get("content")
    if isinstance(content, dict):
        for key in ("units", "suffix"):
            if content.get(key) is not None:
                return content.get(key)
    return None


def redact_item(item: dict[str, Any]) -> dict[str, Any]:
    masked = copy.deepcopy(item)
    if "athlete_id" in masked:
        masked["athlete_id"] = "<REDACTED>"
    if isinstance(masked.get("content"), dict) and "athlete_id" in masked["content"]:
        masked["content"]["athlete_id"] = "<REDACTED>"
    return masked


def is_target_for_deletion(item: dict[str, Any]) -> bool:
    code = extract_code(item)
    if code in EVENING_CODES:
        return True
    item_id = item.get("id")
    try:
        numeric_id = int(item_id) if item_id is not None else None
    except (TypeError, ValueError):
        numeric_id = None
    return numeric_id is not None and 1330713 <= numeric_id <= 1330723


def find_template_item(items: list[dict[str, Any]], excluded_ids: set[Any]) -> dict[str, Any]:
    available_summary: list[str] = []
    for item in items:
        if item.get("id") in excluded_ids:
            continue
        top_name = item.get("name")
        content = item.get("content")
        content_name = content.get("name") if isinstance(content, dict) else None
        candidates = [normalize_text(top_name), normalize_text(content_name)]
        if any(candidate in TARGET_TEMPLATE_NAMES for candidate in candidates if candidate):
            return item
        available_summary.append(
            f"{extract_name(item) or '<sans nom>'} (type={item.get('type')!r}, code={extract_code(item)!r})"
        )
    raise RuntimeError(
        "Aucun item modèle 'Handgrip Gauche' ou 'PVT Médian' trouvé (hors items supprimés). "
        f"Items disponibles: {', '.join(available_summary) if available_summary else 'aucun'}"
    )


def build_payload(field_spec: dict[str, Any], template_item: dict[str, Any]) -> dict[str, Any]:
    cloned = copy.deepcopy(template_item)
    for drop_key in SERVER_DROP_KEYS:
        cloned.pop(drop_key, None)

    cloned["name"] = field_spec["name"]
    cloned["description"] = field_spec["description"]

    if "code" in template_item:
        cloned["code"] = field_spec["code"]
    if "units" in template_item:
        cloned["units"] = field_spec["units"]
    if "suffix" in template_item:
        cloned["suffix"] = field_spec["units"]
    if "min" in template_item:
        cloned["min"] = 0.0
    if "max" in template_item:
        cloned["max"] = float(field_spec["max"])
    if "number_format" in template_item:
        cloned["number_format"] = field_spec["fmt"]

    content = cloned.get("content")
    if isinstance(content, dict):
        if "name" in content:
            content["name"] = field_spec["name"]
        if "description" in content:
            content["description"] = field_spec["description"]
        if "code" in content or "code" not in template_item:
            content["code"] = field_spec["code"]
        if "units" in content:
            content["units"] = field_spec["units"]
        if "suffix" in content:
            content["suffix"] = field_spec["units"]
        content["min"] = 0.0
        content["max"] = float(field_spec["max"])
        content["number_format"] = field_spec["fmt"]

    return cloned


def main() -> int:
    athlete_id, api_key = load_credentials()
    session = requests.Session()
    session.auth = ("API_KEY", api_key)
    session.headers.update(
        {
            "User-Agent": "BilanSoirPWA/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
    )

    custom_item_url = f"https://intervals.icu/api/v1/athlete/{athlete_id}/custom-item"
    print("[1/5] GET /api/v1/athlete/<athlete_id>/custom-item")
    resp = session.get(custom_item_url, timeout=20)
    resp.raise_for_status()
    items: list[dict[str, Any]] = resp.json()
    print(f"      -> {len(items)} custom items récupérés.")

    deleted_ids: set[Any] = set()
    print("\n[2/5] Suppression des items vespéraux existants / IDs 1330713..1330723...")
    for item in items:
        if is_target_for_deletion(item):
            item_id = item.get("id")
            item_type = item.get("type")
            item_code = extract_code(item)
            del_resp = session.delete(f"{custom_item_url}/{item_id}", timeout=20)
            del_resp.raise_for_status()
            deleted_ids.add(item_id)
            print(f"  [SUPPRIMÉ] id={item_id} | type={item_type!r} | code={item_code!r}")
    if not deleted_ids:
        print("      -> Aucun item à supprimer.")

    print("\n[3/5] Recherche de l'item modèle ('Handgrip Gauche' ou 'PVT Médian')...")
    template_item = find_template_item(items, deleted_ids)
    print("=== Structure JSON exacte de l'item modèle (athlete_id masqué) ===")
    print(json.dumps(redact_item(template_item), indent=2, ensure_ascii=False))

    print("\n[4/5] Clonage et création des 11 champs Wellness...")
    created_count = 0
    for spec in EVENING_FIELDS:
        payload = build_payload(spec, template_item)
        post_resp = session.post(custom_item_url, json=payload, timeout=20)
        post_resp.raise_for_status()
        created_item = post_resp.json()
        print(
            f"  [CRÉÉ] id={created_item.get('id')} | code={extract_code(created_item)!r} "
            f"| type={created_item.get('type')!r} | unit={extract_units(created_item)!r}"
        )
        created_count += 1
    print(f"      -> {created_count} champs créés.")

    today_str = datetime.date.today().isoformat()
    wellness_url = f"https://intervals.icu/api/v1/athlete/{athlete_id}/wellness/{today_str}"
    print(f"\n[5/5] Test PUT puis GET sur wellness/{today_str}")

    get_before = session.get(wellness_url, timeout=20)
    before_data: dict[str, Any] = get_before.json() if get_before.status_code == 200 else {}
    before_custom = before_data.get("custom") if isinstance(before_data.get("custom"), dict) else {}

    put_payload: dict[str, Any] = {}
    for code, default_val in TEST_VALUES.items():
        existing_val = before_data.get(code)
        if existing_val is None:
            existing_val = before_custom.get(code)
        put_payload[code] = existing_val if existing_val is not None else default_val

    put_resp = session.put(wellness_url, json=put_payload, timeout=20)
    print(f"      -> PUT status: {put_resp.status_code}")
    put_resp.raise_for_status()

    get_after = session.get(wellness_url, timeout=20)
    print(f"      -> GET status: {get_after.status_code}")
    get_after.raise_for_status()
    after_data: dict[str, Any] = get_after.json()
    after_custom = after_data.get("custom") if isinstance(after_data.get("custom"), dict) else {}

    all_ok = True
    print(f"=== Valeurs relues sur {today_str} ===")
    for spec in EVENING_FIELDS:
        code = spec["code"]
        expected = put_payload[code]
        actual_root = after_data.get(code)
        actual_custom = after_custom.get(code)
        actual = actual_root if actual_root is not None else actual_custom
        source = "root" if actual_root is not None else ("custom" if actual_custom is not None else "missing")
        match = actual is not None and float(actual) == float(expected)
        status_str = f"OK:{source}" if match else f"MISMATCH:{source}"
        if not match:
            all_ok = False
        print(f"  - {code:18s} : attendu={expected!r:6} | relu={actual!r:6} [{status_str}]")

    if not all_ok:
        print("ERREUR: Certains champs n'ont pas été relus correctement.", file=sys.stderr)
        return 1

    print("\nSUCCÈS: Les 11 champs custom Wellness sont actifs et validés par GET après PUT.")
    return 0


if __name__ == "__main__":
    sys.exit(main())