"""Tier 3-ISO: the Amendment-2 isolated ElectionNoise seat path.

    final 14-day publication-safe polling consensus
      -> CONTROL pp_centered_noise            (uniform over K centered atoms)
      -> unchanged bounded simplex transfer    (λ rule, ε = 0.01 pp)
      -> chronological deterministic geography (oracle mode FORBIDDEN)
      -> historically correct mandate law
      -> joint per-draw 8-party seat vector

Every non-ElectionNoise input on this path is **deterministic**: the consensus is a
fixed function of the archived polls, and geography, integerisation and the
allocator are deterministic maps. There are therefore no upstream random draws to
pair, which makes CONTROL/A/B pairing exact by construction (see
``PAIRED_RANDOMNESS`` in ``freeze.py``).

Because the vote → seat map is deterministic, identical vote vectors are memoised.
For CONTROL this collapses 20 000 draws onto ``K`` distinct evaluations and is
bit-identical to evaluating each draw separately — asserted by
``verify_memoisation_is_exact``. A continuous challenger will simply miss the cache
on every draw, so the runner does not depend on the collapse structurally.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from scripts.election_layer_v2.config import CANONICAL_WINDOW_DAYS, MIN_SHARE_PCT
from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals
from scripts.election_layer_v2.transfer import apply_batch_simplex_transfer
from scripts.election_residuals.config import ALL_CATEGORIES, DEFAULT_POLLS_FILE
from scripts.election_residuals.consensus import build_election_polling_consensus
from scripts.geography.projection import project_constituency_votes
from scripts.mandates.allocator import allocate_riksdag_seats
from scripts.mandates.law import MandateLaw, mandate_law_for_election_year
from scripts.simulator.config import MODEL_PARTIES_9, PARLIAMENTARY_PARTIES_8
from scripts.vote_share_calibration.models import derive_vote_share_layer_seeds

PART2B = REPO_ROOT / "diagnostics/election_noise_v2/historical_seat_extension/processed"
RESEARCH_GEO = PART2B / "research_geography"

#: Amendment 2 §E.2a. One case per election; no horizon dimension.
TIER3_ISO_TARGETS: dict[int, dict] = {
    2014: {"election_date": date(2014, 9, 14), "geography_baseline_year": 2010},
    2018: {"election_date": date(2018, 9, 9), "geography_baseline_year": 2014},
    2022: {"election_date": date(2022, 9, 11), "geography_baseline_year": 2018},
}

#: Amendment 2 forbids oracle mode on this path.
GEOGRAPHY_MODE = "chronological"
FORBIDDEN_GEOGRAPHY_MODES = ("oracle",)

_POLLS: pd.DataFrame | None = None


def polls() -> pd.DataFrame:
    global _POLLS
    if _POLLS is None:
        _POLLS = pd.read_csv(DEFAULT_POLLS_FILE)
    return _POLLS


def fixed_seats(year: int) -> dict[str, int]:
    payload = json.loads((PART2B / "fixed_seats_by_year.json").read_text())
    return {k: int(v) for k, v in payload["fixed_seats_by_year"][str(year)].items()}


def certified_seats(year: int) -> dict[str, int]:
    if year == 2014:
        df = pd.read_csv(PART2B / "certified_mandates_2010_2014.csv")
    else:
        df = pd.read_csv(REPO_ROOT / "data/processed/mandates/historical_certified_mandates.csv")
    sub = df[df["election_year"] == year]
    return {p: int(sub[sub["party"] == p]["total_seats"].sum()) for p in PARLIAMENTARY_PARTIES_8}


def consensus_vector(election_date: date) -> np.ndarray:
    """The frozen publication-safe 14-day consensus, in pp over the 9 categories."""
    cons = build_election_polling_consensus(
        election_date, polls(), window_days=CANONICAL_WINDOW_DAYS
    )
    for p in cons.contributing_polls:
        if p.publication_date > election_date or p.interview_end > election_date:
            raise RuntimeError(
                f"consensus admitted a poll unavailable at the origin: {p.pollster} "
                f"pub={p.publication_date} end={p.interview_end} > {election_date}"
            )
    return np.array([cons.consensus_composition[c] for c in ALL_CATEGORIES], dtype=float)


def assert_geography_mode(mode: str) -> None:
    if mode in FORBIDDEN_GEOGRAPHY_MODES:
        raise RuntimeError(
            f"GEOGRAPHY MODE VIOLATION: '{mode}' is forbidden on the Tier 3-ISO path "
            "(Amendment 2 §E.2a). Its row margins come from the target election's "
            "realized constituency valid votes."
        )
    if mode != GEOGRAPHY_MODE:
        raise RuntimeError(f"Tier 3-ISO requires mode '{GEOGRAPHY_MODE}', got '{mode}'")


def votes_to_seats(
    votes_pct: np.ndarray, target_year: int, mode: str = GEOGRAPHY_MODE
) -> np.ndarray:
    """Deterministic map: (N, 9) pp vote matrix -> (N, 8) integer seat matrix.

    Identical vote rows are memoised; the map is deterministic, so this is exact.
    """
    assert_geography_mode(mode)
    spec = TIER3_ISO_TARGETS[target_year]
    cfg = mandate_law_for_election_year(target_year)
    fixed = fixed_seats(target_year)

    cache: dict[bytes, list[int]] = {}
    out = np.empty((votes_pct.shape[0], 8), dtype=np.int64)
    for i in range(votes_pct.shape[0]):
        row = votes_pct[i]
        key = row.tobytes()
        hit = cache.get(key)
        if hit is None:
            shares = {p: float(row[j] / 100.0) for j, p in enumerate(MODEL_PARTIES_9)}
            proj = project_constituency_votes(
                national_vote_shares=shares,
                baseline_year=spec["geography_baseline_year"],
                target_year=target_year,
                mode=mode,
                total_national_votes=None,  # scale comes from the baseline election
                processed_dir=RESEARCH_GEO,
            )
            alloc = allocate_riksdag_seats(
                proj.to_allocator_input(),
                fixed,
                first_divisor=cfg.first_divisor,
                law=cfg.law,
                scenario_id=f"tier3iso_{target_year}",
            )
            if alloc.law != cfg.law.value:
                raise RuntimeError(f"law dispatch mismatch for {target_year}")
            hit = [alloc.final_seats_by_party.get(p, 0) for p in PARLIAMENTARY_PARTIES_8]
            cache[key] = hit
        out[i] = hit
    if not np.all(out.sum(axis=1) == 349):
        raise RuntimeError("seat total invariant violated on the Tier 3-ISO path")
    return out


@dataclass(frozen=True)
class IsoDraws:
    votes_pct: np.ndarray     # (N, 9) pp, rows sum to 100
    seats: np.ndarray         # (N, 8) integer, rows sum to 349
    lambdas: np.ndarray       # (N,)
    residual_index: np.ndarray
    training_years: tuple[int, ...]
    consensus_pct: np.ndarray
    index_seed: int


def control_iso_draws(target_year: int, seed: int, n: int) -> IsoDraws:
    """CONTROL on the Tier 3-ISO path for one (election, seed)."""
    spec = TIER3_ISO_TARGETS[target_year]
    ed = spec["election_date"]
    base = consensus_vector(ed)
    pool = load_chronological_pp_residuals(target_election_year=target_year)
    k = len(pool.training_years)
    if any(int(y) >= target_year for y in pool.training_years):
        raise RuntimeError(f"future residual year in the {target_year} training pool")

    # Frozen Tier-1/ISO origin convention: origin = election date, horizon = 14.
    idx_seed, _ = derive_vote_share_layer_seeds(
        base_seed=seed, origin_date=ed, horizon_days=CANONICAL_WINDOW_DAYS
    )
    idx = np.random.default_rng(idx_seed).integers(0, k, size=n)
    votes, lam = apply_batch_simplex_transfer(
        np.tile(base, (n, 1)), pool.centered_residuals_matrix[idx], eps=MIN_SHARE_PCT
    )
    seats = votes_to_seats(votes, target_year)
    return IsoDraws(
        votes_pct=votes,
        seats=seats,
        lambdas=lam,
        residual_index=idx,
        training_years=tuple(int(y) for y in pool.training_years),
        consensus_pct=base,
        index_seed=int(idx_seed),
    )


def verify_memoisation_is_exact(target_year: int, n: int = 40) -> dict:
    """Assert the memoised map equals a per-draw evaluation, row for row."""
    d = control_iso_draws(target_year, seed=12345, n=n)
    direct = np.empty_like(d.seats)
    for i in range(n):
        direct[i] = votes_to_seats(d.votes_pct[i : i + 1], target_year)[0]
    identical = bool(np.array_equal(direct, d.seats))
    return {
        "target_year": target_year,
        "rows_checked": n,
        "memoised_equals_per_draw": identical,
        "distinct_vote_rows": int(np.unique(np.round(d.votes_pct, 12), axis=0).shape[0]),
        "distinct_seat_rows": int(np.unique(d.seats, axis=0).shape[0]),
    }
