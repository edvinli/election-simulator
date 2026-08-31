"""Exact finite-support CONTROL oracle for the Tier 3-ISO path.

CONTROL's law on this path is uniform over the ``K`` centered historical residual
atoms, and λ ≡ 1 at all three targets, so each atom maps *deterministically*
through consensus → residual transfer → geography → allocator. The predictive
distribution is therefore an exact ``K``-point law and every quantity below is
computed analytically rather than sampled.

This is a **validation artifact**. The preregistered baseline remains the frozen
five-seed × 20 000-draw Monte Carlo run; the oracle exists to prove that run has no
systematic error.

Two definitional notes carried from Part 3:

* The energy score of a uniform ``K``-atom law is
  ``(1/K) Σ‖s_m − y‖ − ½ (1/K²) Σ_{m,l}‖s_m − s_l‖`` — the ``1/K²`` normalisation,
  which includes the ``m = l`` pairs. This is what a Monte Carlo
  ``compute_energy_score`` converges to. ``compute_discrete_energy_score``
  normalises by ``K(K−1)`` and is deliberately **not** used.
* Quantiles of a discrete law are step functions of the empirical weights, so
  interval coverage is not a continuous functional of the atom probabilities and
  cannot be expected to converge smoothly. It is reported and flagged, not compared
  numerically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.election_layer_v2.config import CANONICAL_WINDOW_DAYS, MIN_SHARE_PCT
from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals
from scripts.election_layer_v2.transfer import apply_simplex_transfer
from scripts.election_residuals.config import ALL_CATEGORIES
from scripts.simulator.config import PARLIAMENTARY_PARTIES_8

from .isolated import (
    TIER3_ISO_TARGETS,
    certified_seats,
    consensus_vector,
    votes_to_seats,
)

MASKS: tuple[int, ...] = tuple(range(1, 255))
MAJORITY = 175


def mask_columns(mask: int) -> list[int]:
    return [i for i in range(8) if mask >> i & 1]


@dataclass(frozen=True)
class ExactSupport:
    target_year: int
    k: int
    residual_years: tuple[int, ...]
    atom_probability: float          # 1/K
    votes_pct: np.ndarray            # (K, 9)
    seats: np.ndarray                # (K, 8)
    lambdas: np.ndarray              # (K,)
    consensus_pct: np.ndarray        # (9,)
    truth_seats: np.ndarray          # (8,)


def build_exact_support(target_year: int) -> ExactSupport:
    """Enumerate every residual atom exactly once and push it through the path."""
    ed = TIER3_ISO_TARGETS[target_year]["election_date"]
    base = consensus_vector(ed)
    pool = load_chronological_pp_residuals(target_election_year=target_year)
    k = len(pool.training_years)

    votes = np.empty((k, 9), dtype=float)
    lams = np.empty(k, dtype=float)
    for m in range(k):
        x, lam = apply_simplex_transfer(
            base, pool.centered_residuals_matrix[m], eps=MIN_SHARE_PCT
        )
        votes[m] = x
        lams[m] = lam
    seats = votes_to_seats(votes, target_year)
    truth = np.array([certified_seats(target_year)[p] for p in PARLIAMENTARY_PARTIES_8], dtype=np.int64)
    return ExactSupport(
        target_year=target_year,
        k=k,
        residual_years=tuple(int(y) for y in pool.training_years),
        atom_probability=1.0 / k,
        votes_pct=votes,
        seats=seats,
        lambdas=lams,
        consensus_pct=base,
        truth_seats=truth,
    )


def exact_energy_score(support: np.ndarray, truth: np.ndarray) -> float:
    """ES of a uniform K-atom law, with the 1/K^2 dispersion normalisation."""
    k = support.shape[0]
    t1 = float(np.mean(np.linalg.norm(support - truth[None, :], axis=1)))
    d = np.linalg.norm(support[:, None, :] - support[None, :, :], axis=2)
    return t1 - 0.5 * float(d.sum()) / (k * k)


def exact_crps_1d(atoms: np.ndarray, y: float) -> float:
    """CRPS of a uniform K-atom scalar law: E|X-y| - 0.5 E|X-X'| with 1/K^2."""
    k = atoms.shape[0]
    t1 = float(np.mean(np.abs(atoms - y)))
    t2 = float(np.abs(atoms[:, None] - atoms[None, :]).sum()) / (k * k)
    return t1 - 0.5 * t2


def exact_oracle(target_year: int) -> dict:
    s = build_exact_support(target_year)
    k, p = s.k, s.atom_probability
    truth_v = None  # vote truth attached by the caller where needed

    # --- exact seat quantities -------------------------------------------------
    mean_seats = {
        party: float(np.mean(s.seats[:, j])) for j, party in enumerate(PARLIAMENTARY_PARTIES_8)
    }
    seat_support: dict[str, dict[str, float]] = {}
    for j, party in enumerate(PARLIAMENTARY_PARTIES_8):
        vals, counts = np.unique(s.seats[:, j], return_counts=True)
        seat_support[party] = {str(int(v)): float(c) / k for v, c in zip(vals, counts)}

    # --- exact coalition quantities -------------------------------------------
    per_mask: dict[str, dict] = {}
    briers = np.empty(len(MASKS), dtype=float)
    for i, m in enumerate(MASKS):
        cols = mask_columns(m)
        atom_sums = s.seats[:, cols].sum(axis=1)
        hits = int(np.count_nonzero(atom_sums >= MAJORITY))
        prob = hits / k
        y = 1.0 if int(s.truth_seats[cols].sum()) >= MAJORITY else 0.0
        b = (prob - y) ** 2
        briers[i] = b
        per_mask[str(m)] = {
            "parties": "+".join(PARLIAMENTARY_PARTIES_8[c] for c in cols),
            "atom_coalition_seats": [int(x) for x in atom_sums],
            "atoms_at_or_above_175": hits,
            "exact_probability": prob,
            "exact_probability_numerator_over_k": f"{hits}/{k}",
            "certified_coalition_seats": int(s.truth_seats[cols].sum()),
            "certified_indicator": y,
            "exact_brier": b,
        }
    probs = np.array([per_mask[str(m)]["exact_probability"] for m in MASKS])
    multiples_ok = bool(np.all(np.abs(probs * k - np.round(probs * k)) < 1e-12))

    return {
        "target_year": target_year,
        "k": k,
        "residual_years": list(s.residual_years),
        "atom_probability": p,
        "lambda_per_atom": [float(x) for x in s.lambdas],
        "lambda_identically_one": bool(np.all(s.lambdas == 1.0)),
        "consensus_pct": {c: float(v) for c, v in zip(ALL_CATEGORIES, s.consensus_pct)},
        "exact_mean_vote_pct": {
            c: float(np.mean(s.votes_pct[:, j])) for j, c in enumerate(ALL_CATEGORIES)
        },
        "exact_vote_support": [
            {
                "residual_year": s.residual_years[m],
                "probability": p,
                "vote_pct": {c: float(s.votes_pct[m, j]) for j, c in enumerate(ALL_CATEGORIES)},
                "seats": {
                    party: int(s.seats[m, j]) for j, party in enumerate(PARLIAMENTARY_PARTIES_8)
                },
                "seat_total": int(s.seats[m].sum()),
            }
            for m in range(k)
        ],
        "exact_mean_seats": mean_seats,
        "exact_seat_support": seat_support,
        "exact_seat_energy_score": exact_energy_score(
            s.seats.astype(float), s.truth_seats.astype(float)
        ),
        "exact_seat_crps_per_party": {
            party: exact_crps_1d(s.seats[:, j].astype(float), float(s.truth_seats[j]))
            for j, party in enumerate(PARLIAMENTARY_PARTIES_8)
        },
        "exact_coalition_brier_mean_over_masks": float(np.mean(briers)),
        "masks_evaluated": len(MASKS),
        "effective_distinct_events": 127,
        "all_coalition_probabilities_are_multiples_of_1_over_k": multiples_ok,
        "probability_mass_sums_to_one": abs(k * p - 1.0) < 1e-15,
        "per_mask": per_mask,
        "truth_seats": {
            party: int(s.truth_seats[j]) for j, party in enumerate(PARLIAMENTARY_PARTIES_8)
        },
        "truth_sums_to_349": int(s.truth_seats.sum()) == 349,
    }
