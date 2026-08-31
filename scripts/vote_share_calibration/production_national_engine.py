"""Production national vote-share sampler under the adopted ElectionNoise law.

Only the ElectionNoise layer differs from the legacy path. This module does not
re-implement OpinionState, Dynamics or the seed derivation: it calls the unchanged
``generate_national_vote_shares`` and reuses the ``base_comp_matrix`` that call
produced, so the state-plus-dynamics composition entering ElectionNoise is
**literally the same array** the legacy law received for the same
``(as_of, election_date, samples, seed)``. Pairing is therefore exact by
construction rather than by convention, and "the only scientific change is
ElectionNoise CONTROL -> B" is a property of the code, not a claim.

Nothing in the evaluator freeze or the challenger freeze is modified; this module is
additive, and both freezes still verify.

The legacy law remains fully available: the same call returns its draws too, so
archived RC1 forecasts stay reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals
from scripts.election_layer_v2.transfer import apply_batch_simplex_transfer

from .config import MIN_SHARE_PCT
from .election_noise_b import (
    LEGACY_MODEL_ID,
    MODEL_ID,
    ElectionNoiseBFit,
    derive_election_noise_b_seed,
    draw_election_noise_b,
    fit_election_noise_b,
)
from .national_engine import NationalVoteShareSampleResult, generate_national_vote_shares


@dataclass(frozen=True)
class ElectionNoiseBDetail:
    """Side-channel detail for the adopted law; the forecast itself is unaffected."""

    residuals_pp: np.ndarray
    fit: ElectionNoiseBFit
    election_noise_seed: int


def generate_production_vote_shares(
    as_of: str | date | None = None,
    election_date: str | date = "2026-09-13",
    samples: int = 100_000,
    seed: int = 12345,
    model_id: str = MODEL_ID,
    data_dir: Path | str | None = None,
    polls_file: Path | str | None = None,
    elections_file: Path | str | None = None,
) -> tuple[NationalVoteShareSampleResult, ElectionNoiseBDetail | None]:
    """Sample national vote shares under ``model_id``.

    Returns the **native** ``NationalVoteShareSampleResult`` so the value is a
    drop-in replacement everywhere the legacy result is consumed, plus optional
    ElectionNoise detail for diagnostics.

    ``model_id`` is either the adopted ``pp_lw_gaussian`` or the legacy
    ``pp_centered_noise``; both consume the identical upstream draw.
    """
    legacy = generate_national_vote_shares(
        as_of=as_of, election_date=election_date, samples=samples, seed=seed,
        data_dir=data_dir, polls_file=polls_file, elections_file=elections_file,
    )

    if model_id == LEGACY_MODEL_ID:
        return legacy, None
    if model_id != MODEL_ID:
        raise ValueError(f"unknown ElectionNoise model_id {model_id!r}")

    root_data = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "data" / "processed"
    p_file = Path(polls_file) if polls_file else root_data / "pollofpolls" / "swedishpolls_individual_polls.csv"
    e_file = Path(elections_file) if elections_file else root_data / "elections" / "riksdag_election_results.csv"

    pool = load_chronological_pp_residuals(
        target_election_year=legacy.election_date.year,
        polls_file=p_file, elections_file=e_file,
    )
    if tuple(pool.training_years) != tuple(legacy.training_years):
        raise RuntimeError("training pool differs from the legacy path; inputs are not paired")
    if any(int(y) >= legacy.election_date.year for y in pool.training_years):
        raise RuntimeError("future residual year in the training pool")

    fit = fit_election_noise_b(pool.centered_residuals_matrix)
    sub_seed = derive_election_noise_b_seed(seed, legacy.as_of, legacy.horizon_days)
    residuals = draw_election_noise_b(fit, samples, np.random.default_rng(sub_seed))

    shares, lambdas = apply_batch_simplex_transfer(
        legacy.base_comp_matrix, residuals, eps=MIN_SHARE_PCT
    )
    # Identical exact-simplex renormalisation to the legacy path.
    shares = shares / np.sum(shares, axis=1, keepdims=True)

    diagnostics = dict(legacy.diagnostics)
    diagnostics.update({
        "election_noise_model": MODEL_ID,
        "election_noise_k": fit.k,
        "election_noise_delta": fit.delta,
        "election_noise_tau_sq": fit.tau_sq,
        "election_noise_bessel_factor": fit.bessel_factor,
        "election_noise_tunable_parameters": 0,
    })
    result = NationalVoteShareSampleResult(
        as_of=legacy.as_of,
        election_date=legacy.election_date,
        horizon_days=legacy.horizon_days,
        samples=legacy.samples,
        seed=legacy.seed,
        opinion_state_draws=legacy.opinion_state_draws,
        dynamics_deltas=legacy.dynamics_deltas,
        base_comp_matrix=legacy.base_comp_matrix,   # identical array; pairing is exact
        nat_shares_matrix=shares,
        lambdas=lambdas,
        training_years=tuple(int(y) for y in pool.training_years),
        diagnostics=diagnostics,
    )
    detail = ElectionNoiseBDetail(residuals_pp=residuals, fit=fit, election_noise_seed=sub_seed)
    return result, detail
