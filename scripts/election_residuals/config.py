"""Configuration and constants for Historical Poll-to-Election Residual Study."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Sequence


DEFAULT_POLLS_FILE = Path(__file__).resolve().parents[2] / "data" / "processed" / "pollofpolls" / "swedishpolls_individual_polls.csv"
DEFAULT_ELECTIONS_FILE = Path(__file__).resolve().parents[2] / "data" / "processed" / "elections" / "riksdag_election_results.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "election_residuals"

EVALUATION_ELECTIONS: tuple[date, ...] = (
    date(2002, 9, 15),
    date(2006, 9, 17),
    date(2010, 9, 19),
    date(2014, 9, 14),
    date(2018, 9, 9),
    date(2022, 9, 11),
)

PARLIAMENTARY_PARTIES: tuple[str, ...] = ("M", "L", "C", "KD", "S", "V", "MP", "SD")
ALL_CATEGORIES: tuple[str, ...] = ("M", "L", "C", "KD", "S", "V", "MP", "SD", "REST")

LOOKBACK_WINDOW_DAYS: int = 14
SAMPLE_SIZE_BENCHMARK: float = 1000.0
WEIGHT_MIN: float = 0.7
WEIGHT_MAX: float = 1.5
THRESHOLD_PCT: float = 4.0
THRESHOLD_MARGIN_PCT: float = 1.5  # near_threshold if abs(val - 4.0) <= 1.5
