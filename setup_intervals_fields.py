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


def load_credentials() -> tuple[str, str]:
    if not CREDS_PATH.exists():
        raise FileNotFoundError(f"Credentials file not found: {CREDS_PATH}")
    data = json.loads(CREDS_PATH.read_text(encoding="utf-8"))
    return str(data["athlete_id"]).strip(), str(data["api_key"]).strip()


def extract_code(item: dict[str, Any]) -> str | None:
    top_code = item.get("code")
    if top_code:
        return str(top_code)
    content = item.get("content")
    if isinstance(content, dict) and content.get("code"):
        return str(content["code"])
    return None


def build_payload(field_spec: dict[str, Any], template_item: dict[str, Any] | None) -> dict[str, Any]:
    if template_item is not None:
        cloned = copy.deepcopy(template_item)
        for drop_key in ("id", "updated", "athlete_id", "usage_count", "from_athlete", "from_id"):
            cloned.pop(drop_key, None)
        cloned["name"] = field_spec["name"]
        cloned["description"] = field_spec["description"]
        cloned["type"] = "INPUT_FIELD"
        content = copy.deepcopy(cloned.get("content")) if isinstance(cloned.get("content"), dict) else {}
        content["code"] = field_spec["code"]
        content["name"] = field_spec["name"]
        content["type"] = "numeric"
        content["min"] = 0.0
        content["max"] = float(field_spec["max"])
        content["units"] = field_spec["units"]
        content["suffix"] = field_spec["units"]
        content["number_format"] = field_spec["fmt"]
        cloned["content"] = content
        return cloned

    return {
        "type": "INPUT_FIELD",
        "visibility": "PRIVATE",
        "name": field_spec["name"],
        "description": field_spec["description"],
        "content": {
            "code": field_spec["code"],
            "name": field_spec["name"],
            "type": "numeric",
            "min": 0.0,
            "max": float(field_spec["max"]),
            "units": field_spec["units"],
            "suffix": field_spec["units"],
            "number_format": field_spec["fmt"],
            "gauge": True,
            "text_align": "center",
            "text_wrap": "no",
            "processes_fit_messages": False,
        },
    }


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
    print(f"[1/4] GET {custom_item_url}")
    resp = session.get(custom_item_url, timeout=20)
    resp.raise_for_status()
    items: list[dict[str, Any]] = resp.json()
    print(f"      -> {len(items)} custom items récupérés.")

    input_fields = [item for item in items if item.get("type") == "INPUT_FIELD"]
    print(f"      -> {len(input_fields)} items de type INPUT_FIELD trouvés.")

    template_item: dict[str, Any] | None = input_fields[0] if input_fields else None
    if template_item is not None:
        print("\n=== Structure JSON exacte d'un INPUT_FIELD existant ===")
        print(json.dumps(template_item, indent=2, ensure_ascii=False))

    existing_by_code: dict[str, dict[str, Any]] = {}
    for item in items:
        c = extract_code(item)
        if c:
            existing_by_code[c] = item

    print("\n[2/4] Provisionnement idempotent des 11 champs Wellness (INPUT_FIELD)...")
    created_count = 0
    updated_count = 0
    existing_count = 0

    for spec in EVENING_FIELDS:
        code = spec["code"]
        payload = build_payload(spec, template_item)
        if code in existing_by_code:
            existing_item = existing_by_code[code]
            item_id = existing_item.get("id")
            cur_content = existing_item.get("content") or {}
            if cur_content.get("suffix") != spec["units"] or cur_content.get("units") != spec["units"]:
                put_item_url = f"{custom_item_url}/{item_id}"
                up_resp = session.put(put_item_url, json=payload, timeout=20)
                up_resp.raise_for_status()
                print(f"  [MIS À JOUR] {code:18s} (id={item_id}, name={spec['name']!r}, units={spec['units']!r})")
                updated_count += 1
            else:
                print(f"  [OK - EXISTANT] {code:18s} (id={item_id}, name={spec['name']!r}, units={cur_content.get('units')!r})")
                existing_count += 1
            continue

        post_resp = session.post(custom_item_url, json=payload, timeout=20)
        post_resp.raise_for_status()
        created_item = post_resp.json()
        c_content = created_item.get("content") or {}
        print(f"  [CRÉÉ]          {code:18s} (id={created_item.get('id')}, name={created_item.get('name')!r}, units={c_content.get('units')!r})")
        created_count += 1

    print(f"      -> Bilan: {created_count} créés, {updated_count} mis à jour, {existing_count} déjà conformes.")

    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    wellness_url = f"https://intervals.icu/api/v1/athlete/{athlete_id}/wellness/{yesterday}"
    print(f"\n[3/4] Test PUT sur {wellness_url}")

    get_before = session.get(wellness_url, timeout=20)
    before_data: dict[str, Any] = get_before.json() if get_before.status_code == 200 else {}

    put_payload: dict[str, Any] = {}
    for code, default_val in TEST_VALUES.items():
        existing_val = before_data.get(code)
        put_payload[code] = existing_val if existing_val is not None else default_val

    put_resp = session.put(wellness_url, json=put_payload, timeout=20)
    print(f"      -> PUT status: {put_resp.status_code}")
    put_resp.raise_for_status()

    print(f"\n[4/4] Vérification GET sur {wellness_url}")
    get_after = session.get(wellness_url, timeout=20)
    get_after.raise_for_status()
    after_data: dict[str, Any] = get_after.json()

    all_ok = True
    print(f"=== Valeurs relues sur {yesterday} ===")
    for spec in EVENING_FIELDS:
        code = spec["code"]
        expected = put_payload[code]
        actual = after_data.get(code)
        match = actual is not None and float(actual) == float(expected)
        status_str = "OK" if match else "MISMATCH"
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