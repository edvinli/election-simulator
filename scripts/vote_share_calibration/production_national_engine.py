"""Production national vote-share sampler under the adopted ElectionNoise law.

Since the production default was flipped to the adopted law, ``generate_national_vote_shares``
itself dispatches on ``noise_model``. This module is now a thin convenience wrapper
that selects a law and additionally returns the fitted covariance detail for
diagnostics, which the national engine does not carry in its result type.

It deliberately does not re-implement the law: a second implementation of the same
dispatch is exactly the divergence risk the promotion exists to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals

from .election_noise_b import (
    LEGACY_MODEL_ID,
    MODEL_ID,
    ElectionNoiseBFit,
    derive_election_noise_b_seed,
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
    """Sample national vote shares under ``model_id``, plus optional fit detail.

    The forecast itself comes from the unchanged production national engine; the
    detail is recomputed deterministically from the same pool and sub-seed purely
    for diagnostics.
    """
    if model_id not in (MODEL_ID, LEGACY_MODEL_ID):
        raise ValueError(f"unknown ElectionNoise model_id {model_id!r}")

    result = generate_national_vote_shares(
        as_of=as_of, election_date=election_date, samples=samples, seed=seed,
        data_dir=data_dir, polls_file=polls_file, elections_file=elections_file,
        noise_model=model_id,
    )
    if model_id == LEGACY_MODEL_ID:
        return result, None

    root_data = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "data" / "processed"
    p_file = Path(polls_file) if polls_file else root_data / "pollofpolls" / "swedishpolls_individual_polls.csv"
    e_file = Path(elections_file) if elections_file else root_data / "elections" / "riksdag_election_results.csv"
    pool = load_chronological_pp_residuals(
        target_election_year=result.election_date.year,
        polls_file=p_file, elections_file=e_file,
    )
    if tuple(pool.training_years) != tuple(result.training_years):
        raise RuntimeError("training pool differs from the forecast path; inputs are not paired")
    if any(int(y) >= result.election_date.year for y in pool.training_years):
        raise RuntimeError("future residual year in the training pool")

    fit = fit_election_noise_b(pool.centered_residuals_matrix)
    sub_seed = derive_election_noise_b_seed(seed, result.as_of, result.horizon_days)
    residuals = np.random.default_rng(sub_seed).standard_normal(
        (samples, 9)) @ _factor(fit)
    detail = ElectionNoiseBDetail(residuals_pp=residuals, fit=fit,
                                  election_noise_seed=sub_seed)
    return result, detail


def _factor(fit: ElectionNoiseBFit) -> np.ndarray:
    from .election_noise_b import symmetric_factor
    return symmetric_factor(fit.sigma_tilde).T
