"""Configuration and constants for Residual Robustness and Election Layer v1."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Sequence

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "election_layer"
DEFAULT_RESIDUALS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "election_residuals"
DEFAULT_POLLS_FILE = Path(__file__).resolve().parents[2] / "data" / "processed" / "pollofpolls" / "swedishpolls_individual_polls.csv"
DEFAULT_ELECTIONS_FILE = Path(__file__).resolve().parents[2] / "data" / "processed" / "elections" / "riksdag_election_results.csv"

EVALUATION_ELECTIONS: tuple[date, ...] = (
    date(2018, 9, 9),
    date(2022, 9, 11),
)

ALL_HISTORICAL_ELECTIONS: tuple[date, ...] = (
    date(2002, 9, 15),
    date(2006, 9, 17),
    date(2010, 9, 19),
    date(2014, 9, 14),
    date(2018, 9, 9),
    date(2022, 9, 11),
)

ROBUSTNESS_WINDOWS: tuple[int, ...] = (7, 14, 21)
CANONICAL_WINDOW_DAYS: int = 14

ELECTION_LAYER_VARIANTS: tuple[str, ...] = (
    "base",
    "bias_only",
    "noise_only",
    "bias_plus_noise",
)

DEFAULT_HORIZONS: tuple[int, ...] = (112, 84, 56, 28, 14, 7)
DEFAULT_SAMPLES: int = 5_000
DEFAULT_SEED: int = 12345
