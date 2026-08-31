"""Challenger draw paths - ElectionNoise replaced, everything downstream unchanged.

Both challengers hand percentage-point residual draws to the **unmodified**
production ``apply_batch_simplex_transfer`` and then, on the seat path, to the
frozen Tier 3-ISO map:

    consensus -> ElectionNoise (A or B) -> apply_batch_simplex_transfer
              -> chronological geography -> historically correct mandate allocator

Nothing in the transfer is touched: the ε = 0.01 pp floor, the λ rule, donor
attenuation and the simplex constraints are the production ones. λ is descriptive
only - it is not a tuning parameter and has no adoption gate.

The frozen evaluator's ``isolated.consensus_vector`` and ``isolated.votes_to_seats``
are *imported and called*, never modified, so geography mode, law dispatch and the
349-seat invariant are exactly CONTROL's.

This module produces draws. It computes **no** score: no energy score, no CRPS, no
Brier, no calibration, and it never loads a certified election result.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.election_noise_v2.control_baseline_amendment2.harness2.isolated import (
    TIER3_ISO_TARGETS,
    consensus_vector,
    votes_to_seats,
)
from scripts.election_layer_v2.config import CANONICAL_WINDOW_DAYS, MIN_SHARE_PCT
from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals
from scripts.election_layer_v2.transfer import apply_batch_simplex_transfer

from .challenger_a import draw_challenger_a, fit_challenger_a
from .challenger_b import draw_challenger_b, fit_challenger_b
from .rng import A_INDEX, A_KERNEL, B_NORMAL, challenger_rng

MODEL_A = "CHALLENGER_A_smoothed_bootstrap"
MODEL_B = "CHALLENGER_B_lw_gaussian"


@dataclass(frozen=True)
class ChallengerDraws:
    model: str
    votes_pct: np.ndarray            # (N, 9) pp, rows sum to 100
    lambdas: np.ndarray              # (N,)
    residuals_pp: np.ndarray         # (N, 9) the raw ElectionNoise draw
    training_years: tuple[int, ...]
    consensus_pct: np.ndarray
    h: float | None = None           # Challenger A only
    atom_index: np.ndarray | None = None  # Challenger A only
    seats: np.ndarray | None = None  # (N, 8) integer, rows sum to 349


def challenger_residual_draws(
    model: str,
    target_year: int,
    seed: int,
    n: int,
    origin_date: date,
    horizon_days: int = CANONICAL_WINDOW_DAYS,
    h: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None, tuple[int, ...]]:
    """Draw ``n`` residual vectors from a challenger on the chronological pool."""
    pool = load_chronological_pp_residuals(target_election_year=target_year)
    if any(int(y) >= target_year for y in pool.training_years):
        raise RuntimeError(f"future residual year in the {target_year} training pool")
    centered = pool.centered_residuals_matrix
    years = tuple(int(y) for y in pool.training_years)

    if model == MODEL_A:
        if h is None:
            raise ValueError(
                "Challenger A requires an explicit h from LOEO-FIT on this pool; "
                "it is never defaulted."
            )
        fit = fit_challenger_a(centered, h)
        idx_rng = challenger_rng(seed, origin_date, horizon_days, A_INDEX)
        ker_rng = challenger_rng(seed, origin_date, horizon_days, A_KERNEL)
        r, idx = draw_challenger_a(fit, n, idx_rng, ker_rng)
        return r, idx, years

    if model == MODEL_B:
        if h is not None:
            raise ValueError("Challenger B has no tunable hyperparameter; h must be None")
        fit = fit_challenger_b(centered)
        rng = challenger_rng(seed, origin_date, horizon_days, B_NORMAL)
        return draw_challenger_b(fit, n, rng), None, years

    raise ValueError(f"unknown challenger model {model!r}")


def challenger_iso_draws(
    model: str, target_year: int, seed: int, n: int, h: float | None = None,
    with_seats: bool = True,
) -> ChallengerDraws:
    """A or B on the Tier 3-ISO isolated seat path, for one (election, seed)."""
    ed = TIER3_ISO_TARGETS[target_year]["election_date"]
    base = consensus_vector(ed)
    r, idx, years = challenger_residual_draws(
        model, target_year, seed, n, origin_date=ed,
        horizon_days=CANONICAL_WINDOW_DAYS, h=h,
    )
    votes, lam = apply_batch_simplex_transfer(np.tile(base, (n, 1)), r, eps=MIN_SHARE_PCT)
    seats = votes_to_seats(votes, target_year) if with_seats else None
    return ChallengerDraws(
        model=model, votes_pct=votes, lambdas=lam, residuals_pp=r,
        training_years=years, consensus_pct=base, h=h, atom_index=idx, seats=seats,
    )


def challenger_tier1_draws(
    model: str, target_year: int, seed: int, n: int, h: float | None = None
) -> ChallengerDraws:
    """A or B at Tier 1 (vote level only); identical origin convention to CONTROL."""
    return challenger_iso_draws(model, target_year, seed, n, h=h, with_seats=False)
