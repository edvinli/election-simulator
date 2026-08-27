"""Configuration and constants for Geographic Projection v1."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

DEFAULT_RAW_GEOGRAPHY_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "geography"
DEFAULT_PROCESSED_GEOGRAPHY_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "geography"

MODEL_PARTIES_9: tuple[str, ...] = (
    "M",
    "L",
    "C",
    "KD",
    "S",
    "V",
    "MP",
    "SD",
    "REST",
)

# Ineligible category label used when passing projected votes to MandateAllocator
REST_MANDATE_LABEL: str = "OTHER_INELIGIBLE"

OFFICIAL_CONSTITUENCY_CODES: tuple[str, ...] = tuple(f"{i:02d}" for i in range(1, 30))

CONSTITUENCY_NAME_TO_CODE: Mapping[str, str] = {
    "Stockholms kommun": "01",
    "Stockholms län": "02",
    "Uppsala län": "03",
    "Södermanlands län": "04",
    "Östergötlands län": "05",
    "Jönköpings län": "06",
    "Kronobergs län": "07",
    "Kalmar län": "08",
    "Gotlands län": "09",
    "Blekinge län": "10",
    "Malmö kommun": "11",
    "Skåne läns västra": "12",
    "Skåne läns södra": "13",
    "Skåne läns norra och östra": "14",
    "Hallands län": "15",
    "Göteborgs kommun": "16",
    "Västra Götalands läns västra": "17",
    "Västra Götalands läns norra": "18",
    "Västra Götalands läns södra": "19",
    "Västra Götalands läns östra": "20",
    "Värmlands län": "21",
    "Örebro län": "22",
    "Västmanlands län": "23",
    "Dalarnas län": "24",
    "Gävleborgs län": "25",
    "Västernorrlands län": "26",
    "Jämtlands län": "27",
    "Västerbottens län": "28",
    "Norrbottens län": "29",
}

CODE_TO_CONSTITUENCY_NAME: Mapping[str, str] = {
    code: name for name, code in CONSTITUENCY_NAME_TO_CODE.items()
}
