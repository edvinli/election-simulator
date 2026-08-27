"""Historical CLR transition construction, structural leakage filtering, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math
from typing import Any, Sequence
import numpy as np

from .clr import composition_to_clr
from .state import subtract_calendar_years
from .state_config import ALL_CATEGORIES


MIN_TRANSITIONS: int = 30
RECENCY_HALF_LIFE_DAYS: float = 730.0


@dataclass(frozen=True)
class HistoricalTransition:
    """Represents a joint 9-party CLR transition between two exact historical Poll of Polls dates."""

    start_date: date
    end_date: date
    horizon_days: int
    clr_transition: np.ndarray  # 9-dimensional CLR delta: CLR(PoP_{s+h}) - CLR(PoP_s)
    start_floored: bool = False
    end_floored: bool = False


def build_all_historical_transitions(
    timeseries_data: Sequence[dict[str, Any]],
    horizons: Sequence[int] = (7, 14, 28, 56, 84, 112),
) -> dict[int, list[HistoricalTransition]]:
    """Construct all exact historical CLR transitions across requested horizons.

    Transitions are constructed from exact pairs (s, s+h) in the continuous timeseries.
    """
    ts_by_date = {row["date"]: row for row in timeseries_data}
    sorted_dates = sorted(ts_by_date.keys())

    transitions_by_horizon: dict[int, list[HistoricalTransition]] = {h: [] for h in horizons}

    for start_d in sorted_dates:
        start_row = ts_by_date[start_d]
        start_clr, start_fl = composition_to_clr(start_row["composition"])

        for h in horizons:
            end_d = start_d + timedelta(days=h)
            if end_d in ts_by_date:
                end_row = ts_by_date[end_d]
                end_clr, end_fl = composition_to_clr(end_row["composition"])
                delta_clr = end_clr - start_clr

                transitions_by_horizon[h].append(
                    HistoricalTransition(
                        start_date=start_d,
                        end_date=end_d,
                        horizon_days=h,
                        clr_transition=delta_clr,
                        start_floored=start_fl,
                        end_floored=end_fl,
                    )
                )

    # Sort each horizon pool by end_date for fast filtering
    for h in horizons:
        transitions_by_horizon[h].sort(key=lambda t: t.end_date)

    return transitions_by_horizon


def filter_transitions_as_of(
    transitions: Sequence[HistoricalTransition],
    origin_date: date,
    lookback_years: int | None = None,
) -> tuple[HistoricalTransition, ...]:
    """Strict structural leakage boundary: transitions must satisfy transition_end <= origin_date.

    If lookback_years is specified, additionally enforces:
        origin_date - lookback_years <= transition_end <= origin_date.
    """
    if lookback_years is None:
        return tuple(t for t in transitions if t.end_date <= origin_date)
    else:
        window_start = subtract_calendar_years(origin_date, lookback_years)
        return tuple(t for t in transitions if window_start <= t.end_date <= origin_date)


def compute_recency_weights(
    transitions: Sequence[HistoricalTransition],
    origin_date: date,
    half_life_days: float = RECENCY_HALF_LIFE_DAYS,
) -> tuple[np.ndarray, float, float]:
    """Compute exponential recency weights, uncapped Kish effective count, and weighted mean age.

    Parameters:
        transitions: Sequence of eligible historical transitions (transition_end <= origin_date).
        origin_date: Current forecast origin date.
        half_life_days: Exponential decay half-life in days (default: 730 days).

    Returns:
        tuple of (normalized_probabilities, kish_effective_count, weighted_mean_age_days)
    """
    if not transitions:
        return np.array([], dtype=float), 0.0, 0.0

    ages = np.array([(origin_date - t.end_date).days for t in transitions], dtype=float)
    decay_rate = math.log(2.0) / half_life_days
    raw_weights = np.exp(-decay_rate * ages)

    sum_w = float(np.sum(raw_weights))
    if sum_w <= 0.0:
        raise ValueError("Sum of recency weights must be positive")

    sum_w_sq = float(np.sum(raw_weights ** 2))
    kish_eff = (sum_w ** 2) / sum_w_sq
    weighted_mean_age = float(np.sum(raw_weights * ages) / sum_w)
    probabilities = raw_weights / sum_w

    return probabilities, kish_eff, weighted_mean_age


def summarize_transition_pool(
    transitions: Sequence[HistoricalTransition],
    origin_date: date | None = None,
    categories: Sequence[str] = ALL_CATEGORIES,
) -> dict[str, Any]:
    """Compute structural diagnostics for an eligible historical transition pool."""
    if not transitions:
        return {
            "count": 0,
            "unique_start_dates_count": 0,
            "earliest_start": None,
            "latest_end": None,
            "earliest_transition_end": None,
            "latest_transition_end": None,
            "floored_fraction": 0.0,
            "mean_clr_shift": {cat: 0.0 for cat in categories},
            "weighted_mean_age_days": 0.0,
            "kish_effective_transition_count": 0.0,
        }

    count = len(transitions)
    unique_starts = len({t.start_date for t in transitions})
    earliest_start = min(t.start_date for t in transitions).isoformat()
    latest_end = max(t.end_date for t in transitions).isoformat()
    earliest_end = min(t.end_date for t in transitions).isoformat()
    floored_count = sum(1 for t in transitions if t.start_floored or t.end_floored)
    floored_fraction = floored_count / count

    matrix = np.array([t.clr_transition for t in transitions], dtype=float)
    mean_shifts = np.mean(matrix, axis=0)

    diag: dict[str, Any] = {
        "count": count,
        "unique_start_dates_count": unique_starts,
        "earliest_start": earliest_start,
        "latest_end": latest_end,
        "earliest_transition_end": earliest_end,
        "latest_transition_end": latest_end,
        "floored_fraction": round(floored_fraction, 6),
        "mean_clr_shift": {cat: round(float(mean_shifts[i]), 6) for i, cat in enumerate(categories)},
    }

    if origin_date is not None:
        _, kish_eff, w_age = compute_recency_weights(transitions, origin_date)
        diag["weighted_mean_age_days"] = round(w_age, 2)
        diag["kish_effective_transition_count"] = round(kish_eff, 2)

    return diag
