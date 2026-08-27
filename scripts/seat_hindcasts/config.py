"""Configuration and constants for historical probabilistic Riksdag seat hindcasts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

DEFAULT_HORIZONS: tuple[int, ...] = (112, 84, 56, 28, 14, 7)

# Evaluation target elections and their strictly chronologically lagged geographic baselines
EVALUATION_ELECTIONS: dict[str, dict[str, Any]] = {
    "2018": {
        "election_date": date(2018, 9, 9),
        "geography_baseline_year": 2014,
        "actual_seats": {
            "M": 70,
            "L": 20,
            "C": 31,
            "KD": 22,
            "S": 100,
            "V": 28,
            "MP": 16,
            "SD": 62,
        },
        "actual_shares": {
            "M": 19.84,
            "L": 5.49,
            "C": 8.61,
            "KD": 6.32,
            "S": 28.26,
            "V": 8.00,
            "MP": 4.41,
            "SD": 17.53,
            "REST": 1.53,
        },
    },
    "2022": {
        "election_date": date(2022, 9, 11),
        "geography_baseline_year": 2018,
        "actual_seats": {
            "M": 68,
            "L": 16,
            "C": 24,
            "KD": 19,
            "S": 107,
            "V": 24,
            "MP": 18,
            "SD": 73,
        },
        "actual_shares": {
            "M": 19.10,
            "L": 4.61,
            "C": 6.71,
            "KD": 5.34,
            "S": 30.33,
            "V": 6.75,
            "MP": 5.08,
            "SD": 20.54,
            "REST": 1.54,
        },
    },
}

PARLIAMENTARY_PARTIES_8: tuple[str, ...] = (
    "M",
    "L",
    "C",
    "KD",
    "S",
    "V",
    "MP",
    "SD",
)

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

DEFAULT_SAMPLES: int = 5_000
DEFAULT_SEED: int = 12345

DEFAULT_OUTPUT_DIR: Path = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "seat_hindcasts"
)
