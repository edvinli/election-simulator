"""Configuration, constants, and official constituency metadata for Swedish Riksdag mandate allocation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "mandates"
DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "mandates"

TOTAL_RIKSDAG_SEATS: int = 349
TOTAL_FIXED_SEATS: int = 310
TOTAL_ADJUSTMENT_SEATS: int = 39

NATIONAL_THRESHOLD_PCT: float = 4.0
CONSTITUENCY_THRESHOLD_PCT: float = 12.0

DEFAULT_INELIGIBLE_PARTIES: tuple[str, ...] = (
    "REST",
    "OTHER_INELIGIBLE",
    "OTHER",
    "ÖVR",
    "OG",
    "BLANK",
)

OFFICIAL_CONSTITUENCY_CODES: tuple[str, ...] = tuple(f"{i:02d}" for i in range(1, 30))

# Official Valmyndigheten constituency codes and standard names
OFFICIAL_CONSTITUENCIES: dict[str, str] = {
    "01": "Stockholms kommun",
    "02": "Stockholms län",
    "03": "Uppsala län",
    "04": "Södermanlands län",
    "05": "Östergötlands län",
    "06": "Jönköpings län",
    "07": "Kronobergs län",
    "08": "Kalmar län",
    "09": "Gotlands län",
    "10": "Blekinge län",
    "11": "Malmö kommun",
    "12": "Skåne läns västra",
    "13": "Skåne läns södra",
    "14": "Skåne läns norra och östra",
    "15": "Hallands län",
    "16": "Göteborgs kommun",
    "17": "Västra Götalands läns västra",
    "18": "Västra Götalands läns norra",
    "19": "Västra Götalands läns södra",
    "20": "Västra Götalands läns östra",
    "21": "Värmlands län",
    "22": "Örebro län",
    "23": "Västmanlands län",
    "24": "Dalarnas län",
    "25": "Gävleborgs län",
    "26": "Västernorrlands län",
    "27": "Jämtlands län",
    "28": "Västerbottens län",
    "29": "Norrbottens län",
}

# 2026 official fixed seats decided by Valmyndigheten
FIXED_SEATS_2026: dict[str, int] = {
    "01": 29,  # Stockholms kommun
    "02": 41,  # Stockholms län (+1 vs 2022)
    "03": 12,  # Uppsala län
    "04": 9,   # Södermanlands län
    "05": 14,  # Östergötlands län
    "06": 11,  # Jönköpings län
    "07": 6,   # Kronobergs län
    "08": 7,   # Kalmar län (-1 vs 2022)
    "09": 2,   # Gotlands län
    "10": 5,   # Blekinge län
    "11": 10,  # Malmö kommun
    "12": 9,   # Skåne läns västra
    "13": 12,  # Skåne läns södra
    "14": 10,  # Skåne läns norra och östra
    "15": 10,  # Hallands län
    "16": 18,  # Göteborgs kommun (+1 vs 2022)
    "17": 11,  # Västra Götalands läns västra
    "18": 8,   # Västra Götalands läns norra
    "19": 7,   # Västra Götalands läns södra
    "20": 8,   # Västra Götalands läns östra
    "21": 9,   # Värmlands län
    "22": 9,   # Örebro län
    "23": 8,   # Västmanlands län
    "24": 9,   # Dalarnas län
    "25": 9,   # Gävleborgs län
    "26": 7,   # Västernorrlands län (-1 vs 2022)
    "27": 4,   # Jämtlands län
    "28": 8,   # Västerbottens län
    "29": 8,   # Norrbottens län
}

# 2022 official fixed seats
FIXED_SEATS_2022: dict[str, int] = {
    "01": 29,
    "02": 40,
    "03": 12,
    "04": 9,
    "05": 14,
    "06": 11,
    "07": 6,
    "08": 8,
    "09": 2,
    "10": 5,
    "11": 10,
    "12": 9,
    "13": 12,
    "14": 10,
    "15": 10,
    "16": 17,
    "17": 11,
    "18": 8,
    "19": 7,
    "20": 8,
    "21": 9,
    "22": 9,
    "23": 8,
    "24": 9,
    "25": 9,
    "26": 8,
    "27": 4,
    "28": 8,
    "29": 8,
}

# 2018 official fixed seats
FIXED_SEATS_2018: dict[str, int] = {
    "01": 29,
    "02": 39,
    "03": 11,
    "04": 9,
    "05": 14,
    "06": 11,
    "07": 6,
    "08": 8,
    "09": 2,
    "10": 5,
    "11": 10,
    "12": 9,
    "13": 12,
    "14": 10,
    "15": 10,
    "16": 17,
    "17": 11,
    "18": 8,
    "19": 7,
    "20": 9,
    "21": 9,
    "22": 9,
    "23": 8,
    "24": 9,
    "25": 9,
    "26": 8,
    "27": 4,
    "28": 9,
    "29": 8,
}

CANONICAL_PARTIES: tuple[str, ...] = (
    "M",
    "L",
    "C",
    "KD",
    "S",
    "V",
    "MP",
    "SD",
    "FI",
    "OTHER",
)

PARTY_NAME_MAPPINGS: Mapping[str, str] = {
    "M": "M",
    "MODERATERNA": "M",
    "MODERATA SAMLINGSPARTIET": "M",
    "L": "L",
    "LIBERALERNA": "L",
    "LIBERALERNA (TIDIGARE FOLKPARTIET)": "L",
    "FP": "L",
    "FOLKPARTIET": "L",
    "FOLKPARTIET LIBERALERNA": "L",
    "C": "C",
    "CENTERPARTIET": "C",
    "KD": "KD",
    "KRISTDEMOKRATERNA": "KD",
    "KRISTDEMOKRATISKA SAMHÄLLSPARTIET": "KD",
    "KDS": "KD",
    "S": "S",
    "SOCIALDEMOKRATERNA": "S",
    "ARBETAREPARTIET-SOCIALDEMOKRATERNA": "S",
    "SVERIGES SOCIALDEMOKRATISKA ARBETAREPARTI": "S",
    "V": "V",
    "VÄNSTERPARTIET": "V",
    "VÄNSTERPARTIET KOMMUNISTERNA": "V",
    "VPK": "V",
    "MP": "MP",
    "MILJÖPARTIET": "MP",
    "MILJÖPARTIET DE GRÖNA": "MP",
    "SD": "SD",
    "SVERIGEDEMOKRATERNA": "SD",
    "FI": "FI",
    "F!": "FI",
    "FEMINISTISKT INITIATIV": "FI",
    "ÖVR": "OTHER",
    "ÖVRIGA": "OTHER",
    "ÖVRIGA ANMÄLDA PARTIER": "OTHER",
}


def normalize_party_code(name_or_code: str) -> str:
    cleaned = name_or_code.strip().upper()
    if cleaned in PARTY_NAME_MAPPINGS:
        return PARTY_NAME_MAPPINGS[cleaned]
    simplified = cleaned.replace("-", " ").replace("!", "").strip()
    if simplified in PARTY_NAME_MAPPINGS:
        return PARTY_NAME_MAPPINGS[simplified]
    return "OTHER"
