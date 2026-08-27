"""Leakage-safe PoP transition pool construction and CLR state representation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np

from scripts.pollofpolls.clr import composition_to_clr
from scripts.pollofpolls.state import load_timeseries_dataset
from scripts.pop_state_diagnostics.config import (
    ALL_CATEGORIES_9,
    DEFAULT_HORIZONS,
    POP_TIMESERIES_FILE,
)


@dataclass(frozen=True)
class DailyState:
    """Canonical daily 9-part Poll of Polls state."""

    observation_date: date
    composition: Dict[str, float]  # Percentages summing to 100.0
    clr: np.ndarray  # 9-dimensional CLR coordinate vector
    vote_shares: np.ndarray  # 9-dimensional percentage vector in ALL_CATEGORIES_9 order


@dataclass(frozen=True)
class StateTransition:
    """Exact-horizon joint 9-party CLR transition with start and end states."""

    start_date: date
    end_date: date
    horizon_days: int
    start_composition: Dict[str, float]
    end_composition: Dict[str, float]
    start_clr: np.ndarray
    end_clr: np.ndarray
    start_vote_shares: np.ndarray
    end_vote_shares: np.ndarray
    clr_transition: np.ndarray  # end_clr - start_clr


def load_canonical_pop_series(
    filepath: Path = POP_TIMESERIES_FILE,
) -> Tuple[List[DailyState], Dict[date, DailyState]]:
    """Load daily PoP dataset and compute CLR representations for each date."""
    raw_ts = load_timeseries_dataset(filepath)
    daily_states: List[DailyState] = []
    by_date: Dict[date, DailyState] = {}

    for row in raw_ts:
        d = row["date"]
        comp = row["composition"]
        clr_vec, _ = composition_to_clr(comp)
        shares_vec = np.array([comp[p] for p in ALL_CATEGORIES_9], dtype=float)

        st = DailyState(
            observation_date=d,
            composition=comp,
            clr=clr_vec,
            vote_shares=shares_vec,
        )
        daily_states.append(st)
        by_date[d] = st

    return daily_states, by_date


def build_all_exact_transitions(
    daily_states: Sequence[DailyState],
    by_date: Dict[date, DailyState],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> Dict[int, List[StateTransition]]:
    """Build all exact (t, t+h) transitions present in the continuous PoP daily series."""
    transitions_by_horizon: Dict[int, List[StateTransition]] = {h: [] for h in horizons}

    for st in daily_states:
        start_d = st.observation_date
        for h in horizons:
            end_d = start_d + timedelta(days=h)
            if end_d in by_date:
                end_st = by_date[end_d]
                delta_clr = end_st.clr - st.clr

                tr = StateTransition(
                    start_date=start_d,
                    end_date=end_d,
                    horizon_days=h,
                    start_composition=st.composition,
                    end_composition=end_st.composition,
                    start_clr=st.clr,
                    end_clr=end_st.clr,
                    start_vote_shares=st.vote_shares,
                    end_vote_shares=end_st.vote_shares,
                    clr_transition=delta_clr,
                )
                transitions_by_horizon[h].append(tr)

    return transitions_by_horizon


def get_leakage_safe_candidate_pool(
    origin_date: date,
    horizon_days: int,
    transitions_by_horizon: Dict[int, List[StateTransition]],
) -> List[StateTransition]:
    """Retrieve historical transitions whose realization end_date is strictly <= origin_date."""
    all_h_transitions = transitions_by_horizon.get(horizon_days, [])
    # Strict anti-leakage filter: candidate transition end_date <= origin_date
    eligible = [tr for tr in all_h_transitions if tr.end_date <= origin_date]
    return eligible


def compute_historical_clr_stds_as_of(
    daily_states: Sequence[DailyState],
    as_of_date: date,
) -> np.ndarray:
    """Compute standard deviation of each CLR coordinate using only dates <= as_of_date."""
    historical_clrs = [st.clr for st in daily_states if st.observation_date <= as_of_date]
    if len(historical_clrs) < 2:
        return np.ones(len(ALL_CATEGORIES_9), dtype=float)

    mat = np.array(historical_clrs, dtype=float)
    stds = np.std(mat, axis=0, ddof=1)
    # Avoid zero division
    stds = np.where(stds < 1e-4, 1.0, stds)
    return stds
