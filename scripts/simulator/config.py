"""Configuration, constants, and paths for Swedish Riksdag ElectionSimulator v1."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

DEFAULT_ELECTION_DATE: str = "2026-09-13"
DEFAULT_SIMULATION_SAMPLES: int = 100_000
DEFAULT_SIMULATION_SEED: int = 12345
DEFAULT_GEOGRAPHY_BASELINE_YEAR: int = 2022
DEFAULT_MAJORITY_THRESHOLD: int = 175

MODEL_VERSION: str = "1.0.0-rc1"
RELEASE_TAG: str = "election-simulator-v1.0-rc1"

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

DEFAULT_SIMULATIONS_DIR: Path = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "simulations"
)
