"""CONTROL draw laws and the frozen deterministic downstream, per tier.

CONTROL is the unmodified production ElectionNoise (``pp_centered_noise``): a
uniform draw over the K centered chronological residual atoms, applied through the
unchanged bounded simplex transfer. No smoothing, no reweighting, no exclusion.

Tier 1 uses the deterministic 14-day polling consensus as its base, so it has no
upstream randomness at all. Tier 2/3 call the production national engine and the
production simulator, so their upstream draws are the frozen OpinionState and
Dynamics streams.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.election_layer_v2.config import CANONICAL_WINDOW_DAYS
from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals
from scripts.election_layer_v2.transfer import (
    apply_batch_simplex_transfer,
    apply_simplex_transfer,
)
from scripts.election_residuals.config import ALL_CATEGORIES, DEFAULT_POLLS_FILE
from scripts.election_residuals.consensus import build_election_polling_consensus
from scripts.mandates.law import MandateLaw, mandate_law_for_election_year
from scripts.simulator.engine import simulate_election
from scripts.vote_share_calibration.config import MIN_SHARE_PCT
from scripts.vote_share_calibration.national_engine import generate_national_vote_shares

from .rng import control_residual_indices, stream_seeds, tier1_origin

_POLLS_CACHE: pd.DataFrame | None = None


def _polls() -> pd.DataFrame:
    global _POLLS_CACHE
    if _POLLS_CACHE is None:
        _POLLS_CACHE = pd.read_csv(DEFAULT_POLLS_FILE)
    return _POLLS_CACHE


@dataclass(frozen=True)
class VoteDraws:
    """One (case, seed) block of CONTROL vote draws, in percentage points."""

    votes_pct: np.ndarray        # (N, 9), rows sum to 100
    lambdas: np.ndarray          # (N,)
    base_comp_pct: np.ndarray    # (N, 9) or (1, 9) — the pre-ElectionNoise composition
    residual_index: np.ndarray   # (N,) index into the K atoms
    training_years: tuple[int, ...]


def tier1_support(election_date: date, target_year: int) -> np.ndarray:
    """The exact K-atom support of CONTROL's Tier-1 predictive law, in pp.

    Identical construction to ``scripts/election_layer_v2/forward_eval.py``'s
    ``pp_noise_only`` variant, which is the frozen artifact this anchors against.
    """
    consensus = build_election_polling_consensus(
        election_date, _polls(), window_days=CANONICAL_WINDOW_DAYS
    )
    base = np.array([consensus.consensus_composition[c] for c in ALL_CATEGORIES], dtype=float)
    pool = load_chronological_pp_residuals(target_election_year=target_year)
    pts = [apply_simplex_transfer(base, pool.centered_residuals_matrix[i])[0] for i in range(len(pool.training_years))]
    return np.array(pts, dtype=float)


def tier1_control_draws(election_date: date, target_year: int, seed: int, n: int) -> VoteDraws:
    """CONTROL at Tier 1: uniform atom draw applied to the deterministic consensus."""
    consensus = build_election_polling_consensus(
        election_date, _polls(), window_days=CANONICAL_WINDOW_DAYS
    )
    base = np.array([consensus.consensus_composition[c] for c in ALL_CATEGORIES], dtype=float)
    pool = load_chronological_pp_residuals(target_election_year=target_year)
    k = len(pool.training_years)

    origin, horizon = tier1_origin(election_date)
    ss = stream_seeds(seed, origin, horizon)
    idx = control_residual_indices(ss.election_noise_index_seed, k, n)

    base_matrix = np.tile(base, (n, 1))
    votes, lambdas = apply_batch_simplex_transfer(
        base_matrix, pool.centered_residuals_matrix[idx], eps=MIN_SHARE_PCT
    )
    return VoteDraws(
        votes_pct=votes,
        lambdas=lambdas,
        base_comp_pct=base[None, :],
        residual_index=idx,
        training_years=tuple(int(y) for y in pool.training_years),
    )


def assert_law_dispatch(target_year: int, engine: str) -> MandateLaw:
    """Hard guard: refuse to score a case through an engine that applies the wrong law.

    ``scripts.simulator.engine.simulate_election`` always allocates under the
    current law (POST_2018). A target whose statutory law is PRE_2018 therefore
    must never be scored through it; it needs the historical path built in
    Part 2B (``allocate_riksdag_seats(..., law=PRE_2018, first_divisor=7/5)``).
    This function fails loudly rather than silently producing a legally wrong
    seat vector.
    """
    cfg = mandate_law_for_election_year(target_year)
    if engine == "production_simulate_election" and cfg.law is not MandateLaw.POST_2018:
        raise RuntimeError(
            f"LAW DISPATCH VIOLATION: target {target_year} is governed by {cfg.law.value} "
            f"(first divisor {cfg.first_divisor}), but the production engine applies "
            f"POST_2018. Route this case through the Part-2B historical allocator instead."
        )
    return cfg.law


def tier23_control_draws(
    as_of: date,
    election_date: date,
    seed: int,
    n: int,
    baseline_year: int,
    target_year: int,
) -> tuple[VoteDraws, np.ndarray]:
    """CONTROL at Tier 2/3: the production pipeline, unchanged.

    Returns the vote draws and the (N, 8) seat matrix. The seat matrix comes from
    ``simulate_election``, i.e. the identical geography -> integerisation ->
    exact allocator path the production forecast uses.
    """
    assert_law_dispatch(target_year, engine="production_simulate_election")

    nat = generate_national_vote_shares(
        as_of=as_of, election_date=election_date, samples=n, seed=seed
    )
    sim = simulate_election(
        as_of=as_of,
        election_date=election_date,
        samples=n,
        seed=seed,
        baseline_year=baseline_year,
        geography_mode="chronological",
    )
    votes_pct = sim.vote_shares_matrix  # (N, 9) in pp, already normalised

    # Internal consistency: the national engine called twice with identical
    # arguments must be bit-identical, so the two vote matrices must agree.
    engine_votes = nat.nat_shares_matrix * 100.0
    if not np.allclose(engine_votes, votes_pct, rtol=0, atol=1e-9):
        raise RuntimeError(
            "national engine and simulator disagree on the vote matrix; "
            "determinism assumption violated"
        )

    origin_horizon = max(1, (election_date - as_of).days)
    ss = stream_seeds(seed, as_of, origin_horizon)
    pool = load_chronological_pp_residuals(target_election_year=election_date.year)
    idx = control_residual_indices(
        ss.election_noise_index_seed, len(pool.training_years), n
    )

    draws = VoteDraws(
        votes_pct=votes_pct,
        lambdas=nat.lambdas,
        base_comp_pct=nat.base_comp_matrix,
        residual_index=idx,
        training_years=tuple(int(y) for y in pool.training_years),
    )
    return draws, sim.seats_matrix


def paired_base_composition(as_of: date, election_date: date, seed: int, n: int) -> np.ndarray:
    """The pre-ElectionNoise composition a challenger must receive unchanged.

    Exposed so a later Challenger A/B implementation consumes exactly the same
    upstream draws as CONTROL for the same (case, horizon, seed).
    """
    nat = generate_national_vote_shares(
        as_of=as_of, election_date=election_date, samples=n, seed=seed
    )
    return nat.base_comp_matrix
