"""Configuration and constants for Election Result Layer v2."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Sequence

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "election_layer_v2"
DEFAULT_RESIDUALS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "election_residuals"
DEFAULT_POLLS_FILE = Path(__file__).resolve().parents[2] / "data" / "processed" / "pollofpolls" / "swedishpolls_individual_polls.csv"
DEFAULT_ELECTIONS_FILE = Path(__file__).resolve().parents[2] / "data" / "processed" / "elections" / "riksdag_election_results.csv"

EVALUATION_ELECTIONS: tuple[date, ...] = (
    date(2018, 9, 9),
    date(2022, 9, 11),
)

FORWARD_EVALUATION_ELECTIONS: tuple[date, ...] = (
    date(2010, 9, 19),
    date(2014, 9, 14),
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

CANONICAL_WINDOW_DAYS: int = 14
MIN_SHARE_PCT: float = 0.01  # Epsilon lower bound in percentage points (0.01%)

ELECTION_LAYER_V2_VARIANTS: tuple[str, ...] = (
    "base",
    "pp_bias_only",
    "pp_noise_only",
    "pp_bias_plus_noise",
)

DEFAULT_HORIZONS: tuple[int, ...] = (112, 84, 56, 28, 14, 7)
DEFAULT_SAMPLES: int = 5_000
DEFAULT_SEED: int = 12345
