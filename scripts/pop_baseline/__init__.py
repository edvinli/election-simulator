"""Faithful, opt-in Poll of Polls simulation baseline.

The package deliberately starts from the repository's stored Poll of Polls
point estimate.  It does not aggregate individual polls and it never changes
the frozen ElectionSimulator Candidate A.
"""

from .config import BASELINE_VERSION, DEFAULT_STEP_WINDOWS, PARTY_ORDER
from .model import BaselineForecast, PoPBaselineConfig, simulate_baseline

__all__ = [
    "BASELINE_VERSION",
    "DEFAULT_STEP_WINDOWS",
    "PARTY_ORDER",
    "BaselineForecast",
    "PoPBaselineConfig",
    "simulate_baseline",
]
