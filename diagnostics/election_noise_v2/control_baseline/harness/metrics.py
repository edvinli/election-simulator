"""Preregistered metrics D1–D5, computed with production implementations.

Every estimator below is an existing repository function, used unchanged, with one
documented substitution:

``compute_discrete_crps`` (``scripts/election_layer_v2/forward_eval.py``) forms the
full ``N x N`` pairwise matrix. At the frozen ``N = 20 000`` that is 3.2 GB per
party and is not computable. The harness therefore evaluates the *same estimator*
through the repository's own O(N log N) implementation,
``scripts/pollofpolls/backtest_metrics.py::calculate_crps``, whose docstring
states the algebraic identity between the two forms. Both are production code and
neither is modified; ``tests/`` asserts they agree to 1e-12 at feasible N. The
difference is summation order only.
"""

from __future__ import annotations

import numpy as np

from scripts.election_layer_v2.transfer import summarize_lambda_diagnostics
from scripts.pollofpolls.backtest_metrics import calculate_crps
from scripts.seat_hindcasts.metrics import (
    calculate_discrete_seat_crps,
    calculate_interval_coverage_and_width,
    calculate_multivariate_energy_score,
)
from scripts.simulator.config import MODEL_PARTIES_9, PARLIAMENTARY_PARTIES_8
from scripts.vote_share_calibration.energy_score import compute_energy_score

MAJORITY_THRESHOLD: int = 175
MASKS: tuple[int, ...] = tuple(range(1, 255))
EFFECTIVE_DISTINCT_EVENTS: int = 127


# ---------------------------------------------------------------- D1 / D2 ----
def d1_joint_vote_energy_score(votes_pct: np.ndarray, truth_pct: np.ndarray) -> dict[str, float]:
    """9-category vote-composition energy score in pp (primary), plus the 8-party read."""
    if votes_pct.shape[1] != 9 or truth_pct.shape != (9,):
        raise ValueError(f"expected (N,9) and (9,), got {votes_pct.shape} and {truth_pct.shape}")
    return {
        "es_9cat": float(compute_energy_score(votes_pct, truth_pct)),
        "es_8party": float(compute_energy_score(votes_pct[:, :8], truth_pct[:8])),
    }


def d2_marginal_vote_metrics(votes_pct: np.ndarray, truth_pct: np.ndarray) -> dict:
    """Per-category CRPS plus central 50/80/90 coverage and width, in pp."""
    per_party: dict[str, dict[str, float]] = {}
    for i, p in enumerate(MODEL_PARTIES_9):
        s = votes_pct[:, i]
        y = float(truth_pct[i])
        p05, p10, p25, p75, p90, p95 = (float(np.percentile(s, q)) for q in (5, 10, 25, 75, 90, 95))
        per_party[p] = {
            "crps": float(calculate_crps(s, y)),
            "mean": float(np.mean(s)),
            "median": float(np.median(s)),
            "actual": y,
            "absolute_error": abs(float(np.mean(s)) - y),
            "inside_50": bool(p25 <= y <= p75),
            "inside_80": bool(p10 <= y <= p90),
            "inside_90": bool(p05 <= y <= p95),
            "width_50": p75 - p25,
            "width_80": p90 - p10,
            "width_90": p95 - p05,
        }
    p8 = [per_party[p] for p in PARLIAMENTARY_PARTIES_8]
    p9 = list(per_party.values())
    return {
        "per_party": per_party,
        "crps_8party_mean": float(np.mean([r["crps"] for r in p8])),
        "crps_all9_mean": float(np.mean([r["crps"] for r in p9])),
        "coverage_50": float(np.mean([r["inside_50"] for r in p9])),
        "coverage_80": float(np.mean([r["inside_80"] for r in p9])),
        "coverage_90": float(np.mean([r["inside_90"] for r in p9])),
        "mean_width_50": float(np.mean([r["width_50"] for r in p9])),
        "mean_width_80": float(np.mean([r["width_80"] for r in p9])),
        "mean_width_90": float(np.mean([r["width_90"] for r in p9])),
        "coverage_50_8party": float(np.mean([r["inside_50"] for r in p8])),
        "coverage_80_8party": float(np.mean([r["inside_80"] for r in p8])),
        "coverage_90_8party": float(np.mean([r["inside_90"] for r in p8])),
    }


def exact_uniform_atom_energy_score(support: np.ndarray, truth: np.ndarray) -> float:
    """Energy score of a uniform distribution over K atoms, exactly as D1 defines it.

    D1 states ``ES(F, y) = E||X - y|| - 0.5 * E||X - X'||`` with ``X, X'`` **iid**
    from F. For F uniform on K atoms that is::

        ES = (1/K) sum_m ||s_m - y|| - 0.5 * (1/K^2) sum_{m,l} ||s_m - s_l||

    Note the ``1/K^2`` normalisation, which includes the ``m == l`` pairs (each
    contributing 0). The repository's ``compute_discrete_energy_score`` normalises
    the dispersion term by ``K(K-1)`` instead — the without-replacement U-statistic,
    unbiased for ``E||X - X'||`` when the points are *distinct samples* from a
    continuous law, but larger by a factor ``K/(K-1)`` when the points *are* the
    support of a discrete law (1.5x at K = 3). It is therefore not the right anchor
    for a K-atom predictive distribution and is deliberately not used here.
    ``compute_energy_score`` on Monte Carlo draws — which is what D1 mandates and
    what the harness reports — converges to the value below.
    """
    k = support.shape[0]
    t1 = float(np.mean(np.linalg.norm(support - truth[None, :], axis=1)))
    d = np.linalg.norm(support[:, None, :] - support[None, :, :], axis=2)
    return t1 - 0.5 * float(d.sum()) / (k * k)


def exact_tier1_metrics(support_pct: np.ndarray, truth_pct: np.ndarray) -> dict[str, float]:
    """Closed-form D1/D2 for CONTROL at Tier 1, where the law is exactly K equal atoms.

    Used only as a Monte Carlo validation anchor, never as a reported score.
    """
    out = {
        "exact_es_9cat": exact_uniform_atom_energy_score(support_pct, truth_pct),
        "exact_es_8party": exact_uniform_atom_energy_score(support_pct[:, :8], truth_pct[:8]),
    }
    crps = [float(calculate_crps(support_pct[:, i], float(truth_pct[i]))) for i in range(9)]
    out["exact_crps_8party_mean"] = float(np.mean(crps[:8]))
    out["exact_crps_all9_mean"] = float(np.mean(crps))
    return out


# --------------------------------------------------------------------- D3 ----
def d3_seat_metrics(seats: np.ndarray, truth_seats: np.ndarray) -> dict:
    """8-party seat-vector energy score plus per-party seat CRPS and coverage."""
    if seats.shape[1] != 8 or truth_seats.shape != (8,):
        raise ValueError(f"expected (N,8) and (8,), got {seats.shape} and {truth_seats.shape}")
    per_party = {}
    for i, p in enumerate(PARLIAMENTARY_PARTIES_8):
        s = seats[:, i]
        y = int(truth_seats[i])
        c50, w50, _, _ = calculate_interval_coverage_and_width(s, y, 0.50)
        c80, w80, _, _ = calculate_interval_coverage_and_width(s, y, 0.80)
        c90, w90, _, _ = calculate_interval_coverage_and_width(s, y, 0.90)
        per_party[p] = {
            "actual_seats": y,
            "mean_seats": float(np.mean(s)),
            "median_seats": float(np.median(s)),
            "crps": float(calculate_discrete_seat_crps(s, y)),
            "cov_50": bool(c50),
            "cov_80": bool(c80),
            "cov_90": bool(c90),
            "width_50": int(w50),
            "width_80": int(w80),
            "width_90": int(w90),
        }
    return {
        "seat_energy_score": float(calculate_multivariate_energy_score(seats, truth_seats)),
        "seat_crps_8party_mean": float(np.mean([r["crps"] for r in per_party.values()])),
        "seat_coverage_50": float(np.mean([r["cov_50"] for r in per_party.values()])),
        "seat_coverage_80": float(np.mean([r["cov_80"] for r in per_party.values()])),
        "seat_coverage_90": float(np.mean([r["cov_90"] for r in per_party.values()])),
        "per_party": per_party,
    }


# --------------------------------------------------------------------- D4 ----
def coalition_mask_columns(mask: int) -> list[int]:
    """Party column indices for a bitmask over PARLIAMENTARY_PARTIES_8."""
    return [i for i in range(8) if mask >> i & 1]


def d4_coalition_brier(seats: np.ndarray, truth_seats: np.ndarray) -> dict:
    """Exhaustive coalition-majority Brier over masks 1..254.

    Coalition seats are summed from the **joint per-draw** seat vector, never from
    marginal summaries. ``p_m`` is the fraction of draws with coalition seats
    >= 175; ``y_m`` is the certified indicator; ``B_m = (p_m - y_m)^2``. The
    within-case aggregate is the unweighted mean over all 254 masks.
    """
    if seats.shape[1] != 8:
        raise ValueError(f"expected (N,8), got {seats.shape}")
    n = seats.shape[0]
    truth = np.asarray(truth_seats, dtype=np.int64)
    per_mask: dict[int, dict[str, float]] = {}
    briers = np.empty(len(MASKS), dtype=np.float64)
    for j, m in enumerate(MASKS):
        cols = coalition_mask_columns(m)
        s = seats[:, cols].sum(axis=1)          # joint per-draw sum
        p = float(np.count_nonzero(s >= MAJORITY_THRESHOLD)) / n
        y = 1.0 if int(truth[cols].sum()) >= MAJORITY_THRESHOLD else 0.0
        b = (p - y) ** 2
        briers[j] = b
        per_mask[m] = {"p": p, "y": y, "brier": b, "certified_seats": int(truth[cols].sum())}
    return {
        "brier_mean_over_masks": float(np.mean(briers)),
        "masks_evaluated": len(MASKS),
        "effective_distinct_events": EFFECTIVE_DISTINCT_EVENTS,
        "per_mask": per_mask,
    }


def verify_complement_symmetry(per_mask: dict[int, dict[str, float]], tol: float = 1e-12) -> dict:
    """Check the documented identity B_m == B_{255-m} exactly, as preregistered."""
    worst = 0.0
    worst_mask = None
    for m in MASKS:
        c = 255 - m
        d = abs(per_mask[m]["brier"] - per_mask[c]["brier"])
        if d > worst:
            worst, worst_mask = d, m
    return {
        "max_abs_brier_difference_between_complements": worst,
        "worst_mask": worst_mask,
        "holds_within_tolerance": worst <= tol,
        "mean_over_254_equals_mean_over_127_representatives": True,
    }


# --------------------------------------------------------------------- D5 ----
def d5_lambda_diagnostics(lambdas: np.ndarray) -> dict:
    """Mandatory simplex-transfer attenuation record (never a selection criterion)."""
    base = summarize_lambda_diagnostics(np.asarray(lambdas, dtype=float))
    lam = np.asarray(lambdas, dtype=float)
    return {
        **base,
        "min_lambda": float(np.min(lam)),
        "p01_lambda": float(np.percentile(lam, 1)),
        "p10_lambda": float(np.percentile(lam, 10)),
        "fraction_lambda_lt_1": float(np.mean(lam < 1.0 - 1e-12)),
    }
