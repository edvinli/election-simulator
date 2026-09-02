"""Projection-only simulator using the frozen production scientific components.

The production simulator intentionally remains untouched.  This module composes
the same OpinionState, Dynamics v2, adopted ElectionNoise, geography,
integerization, and mandate-allocation functions while allowing the Dynamics
horizon to be supplied explicitly.  It exists only for the chart's conditional
future fan.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals
from scripts.election_layer_v2.transfer import apply_batch_simplex_transfer
from scripts.geography.config import DEFAULT_PROCESSED_GEOGRAPHY_DIR, OFFICIAL_CONSTITUENCY_CODES
from scripts.geography.integerization import biproportional_controlled_rounding
from scripts.geography.projection import _get_cached_geography_structures
from scripts.hindcasts.models import (
    derive_opinion_state_seed,
    derive_shared_dynamics_seed,
    sample_shared_symmetric_dynamics,
)
from scripts.mandates.config import FIXED_SEATS_2018, FIXED_SEATS_2022, FIXED_SEATS_2026
from scripts.pollofpolls.clr import clr_to_composition_matrix
from scripts.pollofpolls.state import OpinionState, estimate_opinion, load_timeseries_dataset
from scripts.pollofpolls.state_config import ALL_CATEGORIES
from scripts.pollofpolls.transitions import build_all_historical_transitions, filter_transitions_as_of
from scripts.simulator.config import MODEL_PARTIES_9, PARLIAMENTARY_PARTIES_8
from scripts.simulator.engine import _apportion_constituency_units_of_25, _apportion_national_party_integers
from scripts.simulator.fast_allocator import dispatch_production_allocation
from scripts.vote_share_calibration.config import DEFAULT_ELECTIONS_FILE, DEFAULT_POLLS_FILE, MIN_SHARE_PCT
from scripts.vote_share_calibration.election_noise_b import (
    derive_election_noise_b_seed,
    draw_election_noise_b,
    fit_election_noise_b,
)


@dataclass(frozen=True)
class ProjectionSimulationResult:
    """Only the joint matrices needed by the history publication."""

    summary: Any
    vote_shares_matrix: np.ndarray
    seats_matrix: np.ndarray
    diagnostics: dict[str, Any]


def _coerce_date(value: str | date, *, name: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date") from exc


def _sample_national_shares(
    *,
    opinion_state: OpinionState,
    election_date: date,
    dynamics_horizon_days: int,
    samples: int,
    seed: int,
    data_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run production state/dynamics/noise components with an explicit horizon."""

    if dynamics_horizon_days < 0:
        raise ValueError("dynamics_horizon_days cannot be negative")
    as_of_date = opinion_state.as_of
    state_seed = derive_opinion_state_seed(base_seed=seed, origin_date=as_of_date)
    state_samples = opinion_state.sample(n=samples, seed=state_seed)
    state_matrix = np.array(
        [[sample[category] for category in ALL_CATEGORIES] for sample in state_samples],
        dtype=np.float64,
    )
    state_fractions = state_matrix / np.sum(state_matrix, axis=1, keepdims=True)
    log_state = np.log(state_fractions)
    state_clr = log_state - np.mean(log_state, axis=1, keepdims=True)

    if dynamics_horizon_days == 0:
        sym_deltas = np.zeros_like(state_clr)
        eval_h = 0
        eligible_count = 0
    else:
        eval_h = min(dynamics_horizon_days, 112)
        timeseries = load_timeseries_dataset(data_dir / "pollofpolls" / "pollofpolls_timeseries.csv")
        all_transitions = build_all_historical_transitions(timeseries, horizons=[eval_h])
        eligible = filter_transitions_as_of(all_transitions[eval_h], as_of_date)
        if len(eligible) < 30:
            for fallback_h in (28, 14, 7):
                fallback = build_all_historical_transitions(timeseries, horizons=[fallback_h])
                eligible = filter_transitions_as_of(fallback[fallback_h], as_of_date)
                if len(eligible) >= 30:
                    eval_h = fallback_h
                    break
        dyn_seed = derive_shared_dynamics_seed(
            base_seed=seed,
            origin_date=as_of_date,
            horizon_days=dynamics_horizon_days,
        )
        sym_deltas = sample_shared_symmetric_dynamics(
            eligible_transitions=eligible,
            samples_count=samples,
            seed=dyn_seed,
        )
        eligible_count = len(eligible)

    base_comp = clr_to_composition_matrix(state_clr + sym_deltas)
    training_pool = load_chronological_pp_residuals(
        target_election_year=election_date.year,
        polls_file=data_dir / "pollofpolls" / "swedishpolls_individual_polls.csv",
        elections_file=data_dir / "elections" / "riksdag_election_results.csv",
    )
    fit = fit_election_noise_b(training_pool.centered_residuals_matrix)
    noise_seed = derive_election_noise_b_seed(seed, as_of_date, dynamics_horizon_days)
    residuals = draw_election_noise_b(fit, samples, np.random.default_rng(noise_seed))
    national, lambdas = apply_batch_simplex_transfer(base_comp, residuals, eps=MIN_SHARE_PCT)
    national = national / np.sum(national, axis=1, keepdims=True)
    return national, {
        "state_cutoff_date": as_of_date.isoformat(),
        "dynamics_horizon_days": dynamics_horizon_days,
        "dynamics_eval_horizon": eval_h,
        "eligible_transitions_count": eligible_count,
        "election_noise_model": "pp_lw_gaussian",
        "election_noise_seed": noise_seed,
        "mean_lambda": float(np.mean(lambdas)),
    }


def _allocate_seats(
    national: np.ndarray,
    *,
    election_date: date,
    processed_geo_dir: Path,
    baseline_year: int,
    total_national_votes: int,
) -> np.ndarray:
    """Apply the production geography, exact rounding and mandate dispatcher."""

    target_year = election_date.year
    base, row_targets = _get_cached_geography_structures(
        baseline_year=baseline_year,
        target_year=target_year,
        mode="chronological",
        processed_dir_str=str(processed_geo_dir),
    )
    base = base.copy()
    row_int = _apportion_constituency_units_of_25(row_targets.copy(), total_national_votes)
    row_col = row_int[:, np.newaxis].astype(np.float64)
    if target_year == 2018:
        fixed = FIXED_SEATS_2018
    elif target_year == 2022:
        fixed = FIXED_SEATS_2022
    else:
        fixed = FIXED_SEATS_2026
    fixed_arr = np.array([fixed[code] for code in OFFICIAL_CONSTITUENCY_CODES], dtype=np.int64)

    seats = np.zeros((national.shape[0], len(PARLIAMENTARY_PARTIES_8)), dtype=np.int64)
    buffer = np.empty((29, len(MODEL_PARTIES_9)), dtype=np.float64)
    for sample_index in range(national.shape[0]):
        column_int = _apportion_national_party_integers(national[sample_index], total_national_votes)
        column_row = column_int[np.newaxis, :].astype(np.float64)
        np.copyto(buffer, base)
        for _ in range(8):
            buffer *= row_col / np.maximum(np.sum(buffer, axis=1, keepdims=True), 1e-12)
            buffer *= column_row / np.maximum(np.sum(buffer, axis=0, keepdims=True), 1e-12)
        rounded = biproportional_controlled_rounding(
            buffer,
            row_int,
            column_int,
            solver="auto",
        ).rounded_matrix
        allocation = dispatch_production_allocation(rounded, fixed_seats_arr=fixed_arr)
        for party_index, party in enumerate(PARLIAMENTARY_PARTIES_8):
            seats[sample_index, party_index] = allocation.seats_by_party[party]
    if not np.all(np.sum(seats, axis=1) == 349):
        raise RuntimeError("projection mandate allocation violated the 349-seat invariant")
    return seats


def simulate_conditional_projection(
    *,
    as_of: str | date,
    election_date: str | date,
    dynamics_horizon_days: int,
    samples: int,
    seed: int,
    data_dir: Path | str | None = None,
    opinion_state: OpinionState | None = None,
    baseline_year: int = 2022,
    total_national_votes: int = 6_500_000,
) -> ProjectionSimulationResult:
    """Simulate one projection point with a frozen state and explicit horizon."""

    origin = _coerce_date(as_of, name="as_of")
    election = _coerce_date(election_date, name="election_date")
    if dynamics_horizon_days < 0 or dynamics_horizon_days > (election - origin).days:
        raise ValueError("dynamics_horizon_days must be between zero and the natural remaining horizon")
    root = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parents[2] / "data" / "processed"
    state = opinion_state or estimate_opinion(as_of=origin, data_dir=root / "pollofpolls")
    if state.as_of != origin:
        raise ValueError("OpinionState cutoff differs from requested frozen as_of")
    national, diagnostics = _sample_national_shares(
        opinion_state=state,
        election_date=election,
        dynamics_horizon_days=dynamics_horizon_days,
        samples=samples,
        seed=seed,
        data_dir=root,
    )
    geo_dir = root / "geography" if data_dir is not None else DEFAULT_PROCESSED_GEOGRAPHY_DIR
    seats = _allocate_seats(
        national,
        election_date=election,
        processed_geo_dir=geo_dir,
        baseline_year=baseline_year,
        total_national_votes=total_national_votes,
    )
    return ProjectionSimulationResult(
        summary=type("ProjectionSummary", (), {"as_of": origin.isoformat()})(),
        vote_shares_matrix=national * 100.0,
        seats_matrix=seats,
        diagnostics=diagnostics,
    )


__all__ = ["ProjectionSimulationResult", "simulate_conditional_projection"]
