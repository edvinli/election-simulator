"""Canonical national vote-share simulation engine combining OpinionState, Dynamics, and ElectionNoise."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
import numpy as np

from scripts.election_layer_v2.residuals_pool import (
    ChronologicalPPResidualsPool,
    load_chronological_pp_residuals,
)
from scripts.election_layer_v2.transfer import apply_batch_simplex_transfer
from scripts.hindcasts.models import (
    derive_opinion_state_seed,
    derive_shared_dynamics_seed,
    hindcast_state_plus_dynamics,
    sample_shared_symmetric_dynamics,
)
from scripts.pollofpolls.clr import clr_to_composition_matrix
from scripts.pollofpolls.state import OpinionState, estimate_opinion, load_timeseries_dataset
from scripts.pollofpolls.state_config import ALL_CATEGORIES
from scripts.pollofpolls.transitions import (
    HistoricalTransition,
    build_all_historical_transitions,
    filter_transitions_as_of,
)

from .config import DEFAULT_ELECTIONS_FILE, DEFAULT_POLLS_FILE, MIN_SHARE_PCT
from .election_noise_b import (
    LEGACY_MODEL_ID,
    MODEL_ID as ADOPTED_NOISE_MODEL,
    derive_election_noise_b_seed,
    draw_election_noise_b,
    fit_election_noise_b,
)
from .models import apply_vote_share_models, derive_vote_share_layer_seeds


@dataclass(frozen=True)
class NationalVoteShareSampleResult:
    """Output from canonical national vote-share simulation."""

    as_of: date
    election_date: date
    horizon_days: int
    samples: int
    seed: int
    opinion_state_draws: np.ndarray  # shape (N, 9), ALR or direct composition fractions
    dynamics_deltas: np.ndarray       # shape (N, 9), symmetric CLR transition vectors
    base_comp_matrix: np.ndarray      # shape (N, 9), state_plus_dynamics compositions
    nat_shares_matrix: np.ndarray     # shape (N, 9), final pp_centered_noise vote shares summing to 1.0
    lambdas: np.ndarray               # shape (N,), transfer scaling attenuation factors
    training_years: tuple[int, ...]
    diagnostics: dict[str, Any]


def generate_national_vote_shares(
    as_of: str | date | None = None,
    election_date: str | date = "2026-09-13",
    samples: int = 100_000,
    seed: int = 12345,
    data_dir: Path | str | None = None,
    polls_file: Path | str | None = None,
    elections_file: Path | str | None = None,
    noise_model: str = ADOPTED_NOISE_MODEL,
) -> NationalVoteShareSampleResult:
    """Canonical sampling function for Swedish national party vote shares.

    Pipeline:
        1. OpinionState v1.1: Draw ALR uncertainty samples.
        2. Dynamics v2 (symmetric_all_history): Sample joint CLR transition vectors S * Delta_h.
           (Strictly NO sqrt(h) scaling).
        3. ElectionNoise: Apply bounded simplex-safe transfer of a joint zero-sum
           residual draw from the chronological training pool.

    ElectionNoise law selection (``noise_model``):
        ``pp_lw_gaussian``   - DEFAULT. The regularized joint Gaussian law selected by
                               the preregistered historical evaluation (ADOPT_B). See
                               ``docs/election_noise_adopted_model.md``.
        ``pp_centered_noise`` - the superseded empirical discrete bootstrap, retained
                               unmodified so archived forecasts stay reproducible.

    Only the ElectionNoise draw differs between the two: OpinionState, Dynamics, the
    seed derivation and ``base_comp_matrix`` are identical, so the two laws are
    exactly paired at any given ``(as_of, election_date, samples, seed)``.

    Parameters:
        as_of: Polling observation cutoff date (None => latest available in timeseries).
        election_date: Target election date.
        samples: Number of Monte Carlo draws N.
        seed: Base random seed for deterministic sub-seed derivation.
        noise_model: ElectionNoise law; defaults to the adopted ``pp_lw_gaussian``.
        data_dir: Base directory containing processed data.
        polls_file: Path to individual_polls.csv.
        elections_file: Path to riksdag_election_results.csv.

    Returns:
        NationalVoteShareSampleResult containing national share matrix and latent components.
    """
    elec_date = date.fromisoformat(str(election_date)) if isinstance(election_date, str) else election_date
    root_data = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "data" / "processed"
    p_file = Path(polls_file) if polls_file else root_data / "pollofpolls" / "swedishpolls_individual_polls.csv"
    e_file = Path(elections_file) if elections_file else root_data / "elections" / "riksdag_election_results.csv"
    ts_file = root_data / "pollofpolls" / "pollofpolls_timeseries.csv"

    # 1. Fit OpinionState v1.1
    opinion_state = estimate_opinion(as_of=as_of, data_dir=root_data / "pollofpolls")
    as_of_date = opinion_state.as_of
    horizon_days = max(1, (elec_date - as_of_date).days)

    # 2. Deterministic Sub-Seeds
    state_seed = derive_opinion_state_seed(base_seed=seed, origin_date=as_of_date)
    dyn_seed = derive_shared_dynamics_seed(base_seed=seed, origin_date=as_of_date, horizon_days=horizon_days)
    idx_seed, sign_seed = derive_vote_share_layer_seeds(base_seed=seed, origin_date=as_of_date, horizon_days=horizon_days)

    # 3. OpinionState draws
    state_samples = opinion_state.sample(n=samples, seed=state_seed)
    state_matrix = np.array([[s[c] for c in ALL_CATEGORIES] for s in state_samples], dtype=np.float64)
    # Convert percentages [0, 100] to fractions [0, 1]
    state_fractions = state_matrix / np.sum(state_matrix, axis=1, keepdims=True)
    log_state = np.log(state_fractions)
    state_clr = log_state - np.mean(log_state, axis=1, keepdims=True)

    # 4. Dynamics v2 (symmetric_all_history)
    timeseries_data = load_timeseries_dataset(ts_file)
    eval_h = min(horizon_days, 112) if horizon_days > 112 else horizon_days
    all_trans = build_all_historical_transitions(timeseries_data, horizons=[eval_h])
    eligible_trans = filter_transitions_as_of(all_trans[eval_h], as_of_date)

    if len(eligible_trans) < 30:
        for fallback_h in [28, 14, 7]:
            fb_trans = build_all_historical_transitions(timeseries_data, horizons=[fallback_h])
            eligible_trans = filter_transitions_as_of(fb_trans[fallback_h], as_of_date)
            if len(eligible_trans) >= 30:
                eval_h = fallback_h
                break

    # Sample joint symmetric CLR dynamics with NO sqrt(h) scaling
    sym_deltas = sample_shared_symmetric_dynamics(
        eligible_transitions=eligible_trans,
        samples_count=samples,
        seed=dyn_seed,
    )

    base_clr_matrix = state_clr + sym_deltas
    base_comp_matrix = clr_to_composition_matrix(base_clr_matrix)

    # 5. ElectionNoise
    training_pool = load_chronological_pp_residuals(
        target_election_year=elec_date.year,
        polls_file=p_file,
        elections_file=e_file,
    )
    if noise_model == LEGACY_MODEL_ID:
        model_draws = apply_vote_share_models(
            base_comp_matrix=base_comp_matrix,
            training_pool=training_pool,
            samples_count=samples,
            index_seed=idx_seed,
            sign_seed=sign_seed,
            eps=MIN_SHARE_PCT,
        )
        nat_shares_matrix, lambdas = model_draws["pp_centered_noise"]
        noise_diagnostics: dict[str, Any] = {"election_noise_model": LEGACY_MODEL_ID}
    elif noise_model == ADOPTED_NOISE_MODEL:
        fit = fit_election_noise_b(training_pool.centered_residuals_matrix)
        noise_seed = derive_election_noise_b_seed(seed, as_of_date, horizon_days)
        residuals = draw_election_noise_b(fit, samples, np.random.default_rng(noise_seed))
        nat_shares_matrix, lambdas = apply_batch_simplex_transfer(
            base_comp_matrix, residuals, eps=MIN_SHARE_PCT
        )
        noise_diagnostics = {
            "election_noise_model": ADOPTED_NOISE_MODEL,
            "election_noise_k": fit.k,
            "election_noise_delta": fit.delta,
            "election_noise_tau_sq": fit.tau_sq,
            "election_noise_bessel_factor": fit.bessel_factor,
            "election_noise_seed": noise_seed,
            "election_noise_tunable_parameters": 0,
        }
    else:
        raise ValueError(
            f"unknown noise_model {noise_model!r}; expected "
            f"{ADOPTED_NOISE_MODEL!r} (adopted) or {LEGACY_MODEL_ID!r} (historical)"
        )
    # Re-normalize to strictly exact simplex
    nat_shares_matrix = nat_shares_matrix / np.sum(nat_shares_matrix, axis=1, keepdims=True)

    return NationalVoteShareSampleResult(
        as_of=as_of_date,
        election_date=elec_date,
        horizon_days=horizon_days,
        samples=samples,
        seed=seed,
        opinion_state_draws=state_fractions,
        dynamics_deltas=sym_deltas,
        base_comp_matrix=base_comp_matrix,
        nat_shares_matrix=nat_shares_matrix,
        lambdas=lambdas,
        training_years=training_pool.training_years,
        diagnostics={
            "dynamics_eval_horizon": eval_h,
            "eligible_transitions_count": len(eligible_trans),
            "mean_lambda": float(np.mean(lambdas)),
            "min_lambda": float(np.min(lambdas)),
            **noise_diagnostics,
        },
    )
