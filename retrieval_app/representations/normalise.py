from __future__ import annotations

import re


ALIASES = {
    "bed": "bedroom", "bedroom": "bedroom", "master": "bedroom",
    "bath": "bathroom", "bathroom": "bathroom", "toilet": "bathroom",
    "living": "livingroom", "livingroom": "livingroom", "living room": "livingroom",
    "dining": "diningroom", "dining room": "diningroom",
    "kitchen": "kitchen", "entry": "entry", "entrance": "entry",
    "corridor": "corridor", "hall": "corridor", "balcony": "balcony",
    "storage": "storage", "garage": "garage", "outdoor": "outdoor",
}


def room_type(value: str) -> str:
    cleaned = re.sub(r"[_-]+", " ", value).strip().lower()
    cleaned = re.sub(r"\d+$", "", cleaned).strip()
    return ALIASES.get(cleaned, cleaned.replace(" ", ""))
