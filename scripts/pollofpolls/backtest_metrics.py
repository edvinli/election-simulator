"""Scoring rules, point errors, calibration intervals, and fast CRPS for probabilistic forecasts."""

from __future__ import annotations

import math
from typing import Any, Sequence
import numpy as np


def precompute_crps_sample_term(sorted_samples: np.ndarray | Sequence[float]) -> float:
    """Precompute the second, sample-only term of the empirical CRPS.

    For sorted samples x_{(0)} <= x_{(1)} <= ... <= x_{(n-1)}:
        term2 = (1 / n^2) * sum_{i=0}^{n-1} (2i + 1 - n) * x_{(i)}
    which is mathematically equal to:
        (1 / (2 * n^2)) * sum_{i=0}^{n-1} sum_{j=0}^{n-1} |x_i - x_j|

    This term depends only on the forecast distribution and is independent of the actual target y.
    """
    arr = np.asarray(sorted_samples, dtype=float)
    n = arr.shape[0]
    if n < 1:
        raise ValueError("Cannot compute CRPS on empty sample array")
    if n == 1:
        return 0.0

    # Linear weights: (2*i + 1 - n) for i in 0..n-1
    i_indices = np.arange(n, dtype=float)
    weights = (2.0 * i_indices + 1.0 - n) / (n * n)
    return float(np.dot(weights, arr))


def calculate_crps(
    samples: np.ndarray | Sequence[float],
    actual: float,
    precomputed_sample_term: float | None = None,
) -> float:
    """Calculate the exact continuous ranked probability score (CRPS) in O(n) given sorted samples.

    Parameters:
        samples: Array of Monte Carlo samples from the forecast distribution.
        actual: Observed scalar realization y.
        precomputed_sample_term: Optional precomputed sample-dispersion term.

    Returns:
        Empirical CRPS scalar value.
    """
    arr = np.asarray(samples, dtype=float)
    n = arr.shape[0]
    if n < 1:
        raise ValueError("Cannot compute CRPS on empty sample array")

    term1 = float(np.mean(np.abs(arr - actual)))

    if precomputed_sample_term is not None:
        term2 = precomputed_sample_term
    else:
        sorted_arr = np.sort(arr)
        term2 = precompute_crps_sample_term(sorted_arr)

    return term1 - term2


def calculate_point_error(point_forecast: float, actual: float) -> dict[str, float]:
    """Calculate directional error, absolute error, and squared error."""
    err = point_forecast - actual
    return {
        "error": err,
        "absolute_error": abs(err),
        "squared_error": err ** 2,
    }


def calculate_interval_metrics(
    quantiles: dict[float, float],
    actual: float,
) -> dict[str, Any]:
    """Calculate central 50%, 80%, and 90% interval coverage indicators and interval widths.

    Intervals:
        50% central interval: [P25, P75]
        80% central interval: [P10, P90]
        90% central interval: [P05, P95]
    """
    p05 = quantiles[0.05]
    p10 = quantiles[0.10]
    p25 = quantiles[0.25]
    p75 = quantiles[0.75]
    p90 = quantiles[0.90]
    p95 = quantiles[0.95]

    cov50 = 1 if (p25 <= actual <= p75) else 0
    cov80 = 1 if (p10 <= actual <= p90) else 0
    cov90 = 1 if (p05 <= actual <= p95) else 0

    return {
        "p05": p05,
        "p10": p10,
        "p25": p25,
        "p50": quantiles[0.50],
        "p75": p75,
        "p90": p90,
        "p95": p95,
        "interval50_contains_actual": cov50,
        "interval80_contains_actual": cov80,
        "interval90_contains_actual": cov90,
        "width_50": p75 - p25,
        "width_80": p90 - p10,
        "width_90": p95 - p05,
    }
