"""State similarity metrics, nearest-neighbor ranking, and neighbor diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Tuple
import numpy as np

from scripts.pop_state_diagnostics.transitions import DailyState, StateTransition


@dataclass(frozen=True)
class CandidateNeighborRecord:
    """Detailed audit record of a candidate historical transition relative to an origin."""

    origin_date: date
    horizon_days: int
    candidate_start: date
    candidate_end: date
    distance: float
    distance_rank: int
    recency_rank: int
    age_days: int  # (origin_date - candidate_end).days
    pool_size: int
    is_top_50_nn: bool
    is_top_50_recent: bool


def compute_standardized_clr_distance(
    origin_clr: np.ndarray,
    candidate_clr: np.ndarray,
    clr_stds: np.ndarray,
) -> float:
    """Compute standardized Euclidean distance in 9-dimensional CLR space."""
    scaled_diff = (origin_clr - candidate_clr) / clr_stds
    return float(np.sqrt(np.sum(scaled_diff**2)))


def rank_candidate_transitions(
    origin_state: DailyState,
    candidate_pool: List[StateTransition],
    clr_stds: np.ndarray,
) -> Tuple[List[Tuple[StateTransition, float]], List[StateTransition], List[CandidateNeighborRecord]]:
    """Rank candidate transitions by state distance and by recency.

    Returns:
        - List of (transition, distance) sorted ascending by distance (nearest first).
        - List of transitions sorted descending by candidate_end (most recent first).
        - List of audit records for top neighbors.
    """
    origin_d = origin_state.observation_date
    h = candidate_pool[0].horizon_days if candidate_pool else 0
    n_pool = len(candidate_pool)

    # 1. Compute distances
    transitions_with_dist: List[Tuple[StateTransition, float, int]] = []
    for tr in candidate_pool:
        dist = compute_standardized_clr_distance(origin_state.clr, tr.start_clr, clr_stds)
        age = (origin_d - tr.end_date).days
        transitions_with_dist.append((tr, dist, age))

    # 2. Sort by distance (ascending)
    by_distance = sorted(transitions_with_dist, key=lambda x: (x[1], x[2]))

    # 3. Sort by recency (descending end_date, so smallest age first)
    by_recency = sorted(transitions_with_dist, key=lambda x: (x[2], x[1]))

    # Map transitions to ranks
    dist_ranks: Dict[Tuple[date, date], int] = {
        (tr.start_date, tr.end_date): rank + 1 for rank, (tr, _, _) in enumerate(by_distance)
    }
    recency_ranks: Dict[Tuple[date, date], int] = {
        (tr.start_date, tr.end_date): rank + 1 for rank, (tr, _, _) in enumerate(by_recency)
    }

    # Build audit records for top 50 nearest and top 50 most recent
    audit_records: List[CandidateNeighborRecord] = []
    for rank, (tr, dist, age) in enumerate(by_distance[:50]):
        d_rank = rank + 1
        r_rank = recency_ranks[(tr.start_date, tr.end_date)]
        rec = CandidateNeighborRecord(
            origin_date=origin_d,
            horizon_days=h,
            candidate_start=tr.start_date,
            candidate_end=tr.end_date,
            distance=round(dist, 5),
            distance_rank=d_rank,
            recency_rank=r_rank,
            age_days=age,
            pool_size=n_pool,
            is_top_50_nn=True,
            is_top_50_recent=(r_rank <= 50),
        )
        audit_records.append(rec)

    sorted_nn_transitions = [(tr, dist) for (tr, dist, _) in by_distance]
    sorted_recent_transitions = [tr for (tr, _, _) in by_recency]

    return sorted_nn_transitions, sorted_recent_transitions, audit_records
