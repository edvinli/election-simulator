"""Hindcast models and shared Monte Carlo dynamics sampling for Election Hindcast v1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
from typing import Any, Sequence
import numpy as np

from scripts.pollofpolls.clr import clr_to_composition_matrix, composition_to_clr
from scripts.pollofpolls.state import OpinionState
from scripts.pollofpolls.state_config import ALL_CATEGORIES
from scripts.pollofpolls.transitions import HistoricalTransition, MIN_TRANSITIONS


def derive_shared_dynamics_seed(base_seed: int, origin_date: date, horizon_days: int) -> int:
    """Derive deterministic SHA-256 seed for shared dynamics transition sampling."""
    token = f"{base_seed}:{origin_date.isoformat()}:{horizon_days}:shared_dynamics".encode("utf-8")
    digest = hashlib.sha256(token).hexdigest()
    return int(digest[:8], 16) % 2_147_483_647


def derive_opinion_state_seed(base_seed: int, origin_date: date) -> int:
    """Derive deterministic SHA-256 seed for OpinionState sampling."""
    token = f"{base_seed}:{origin_date.isoformat()}:opinion_state".encode("utf-8")
    digest = hashlib.sha256(token).hexdigest()
    return int(digest[:8], 16) % 2_147_483_647


def sample_shared_symmetric_dynamics(
    eligible_transitions: Sequence[HistoricalTransition],
    samples_count: int,
    seed: int,
) -> np.ndarray:
    """Sample joint symmetric CLR transition vectors S^(i) * Delta_h^(i).

    Parameters:
        eligible_transitions: Transitions satisfying transition_end <= origin_date.
        samples_count: Number of Monte Carlo draws (default: 5000).
        seed: Deterministic integer seed.

    Returns:
        NumPy array of shape (samples_count, 9) representing symmetric CLR deltas.
    """
    if len(eligible_transitions) < MIN_TRANSITIONS:
        raise ValueError(
            f"Insufficient historical transitions ({len(eligible_transitions)} < {MIN_TRANSITIONS})"
        )

    delta_matrix = np.array([t.clr_transition for t in eligible_transitions], dtype=float)
    rng = np.random.default_rng(seed)

    # 1. Sample transition indices with replacement
    sampled_indices = rng.integers(0, len(eligible_transitions), size=samples_count)
    sampled_deltas = delta_matrix[sampled_indices]

    # 2. Sample independent random signs +1 / -1
    signs = rng.choice([-1.0, 1.0], size=(samples_count, 1))

    return signs * sampled_deltas


def hindcast_point_persistence(
    origin_pop: dict[str, float],
    samples_count: int = 5000,
    categories: Sequence[str] = ALL_CATEGORIES,
) -> np.ndarray:
    """Generate deterministic point persistence predictive matrix (theta_E = PoP_t)."""
    base_row = np.array([origin_pop[cat] for cat in categories], dtype=float)
    return np.tile(base_row, (samples_count, 1))


def hindcast_dynamics_only(
    origin_pop: dict[str, float],
    symmetric_deltas: np.ndarray,
    categories: Sequence[str] = ALL_CATEGORIES,
) -> np.ndarray:
    """Generate dynamics-only predictive matrix: CLR(theta_E) = CLR(PoP_t) + S * Delta_h."""
    origin_clr, _ = composition_to_clr(origin_pop, categories=categories)
    sampled_clr = origin_clr + symmetric_deltas
    return clr_to_composition_matrix(sampled_clr)


def hindcast_state_plus_dynamics(
    opinion_state: OpinionState,
    symmetric_deltas: np.ndarray,
    state_seed: int,
    samples_count: int = 5000,
    categories: Sequence[str] = ALL_CATEGORIES,
) -> np.ndarray:
    """Generate state + dynamics predictive matrix: CLR(theta_E) = CLR(theta_t) + S * Delta_h."""
    # 1. Draw samples from OpinionState v1.1
    state_samples = opinion_state.sample(n=samples_count, seed=state_seed)

    # 2. Convert state samples to CLR space
    state_matrix = np.array([[s[cat] for cat in categories] for s in state_samples], dtype=float)
    log_vals = np.log(state_matrix)
    mean_logs = np.mean(log_vals, axis=1, keepdims=True)
    state_clr = log_vals - mean_logs

    # 3. Add shared symmetric dynamics in CLR space
    forecast_clr = state_clr + symmetric_deltas

    # 4. Inverse CLR back to composition space
    return clr_to_composition_matrix(forecast_clr)

