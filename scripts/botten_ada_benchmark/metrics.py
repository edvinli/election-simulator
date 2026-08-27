"""Proper scoring and interval metrics used by the Botten Ada comparison."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def continuous_crps(samples: Sequence[float] | np.ndarray, actual: float) -> float:
    """Monte Carlo CRPS using the exact V-statistic for the sample forecast."""
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        raise ValueError("CRPS requires at least one predictive draw")
    term1 = float(np.mean(np.abs(x - float(actual))))
    # Sorting gives the exact sum of pairwise absolute differences without an
    # O(N^2) matrix.  The V-statistic includes the zero diagonal terms.
    ordered = np.sort(x)
    n = ordered.size
    pair_sum = float(2.0 * np.sum((2 * np.arange(n) - n + 1) * ordered))
    term2 = pair_sum / (2.0 * n * n)
    return float(term1 - term2)


def energy_score(samples: np.ndarray, actual: np.ndarray, chunk_size: int = 512) -> float:
    """Multivariate Energy Score with a V-statistic pairwise term."""
    x = np.asarray(samples, dtype=np.float64)
    y = np.asarray(actual, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[1] != y.size:
        raise ValueError("Energy Score requires (N,D) samples and a matching (D,) actual vector")
    if x.shape[0] == 0:
        raise ValueError("Energy Score requires at least one predictive draw")
    term1 = float(np.mean(np.linalg.norm(x - y, axis=1)))
    pair_sum = 0.0
    for start in range(0, x.shape[0], chunk_size):
        chunk = x[start : start + chunk_size]
        pair_sum += float(np.sum(np.linalg.norm(chunk[:, None, :] - x[None, :, :], axis=2)))
    term2 = pair_sum / (2.0 * x.shape[0] * x.shape[0])
    return float(term1 - term2)


def threshold_brier(samples: Sequence[float] | np.ndarray, actual: float, threshold: float = 4.0) -> float:
    """Brier score for the event that a party reaches the inclusive 4% threshold."""
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        raise ValueError("Threshold Brier requires at least one predictive draw")
    probability = float(np.mean(x >= threshold))
    outcome = float(float(actual) >= threshold)
    return float((probability - outcome) ** 2)


def party_mae(samples_by_party: Mapping[str, np.ndarray], actual_by_party: Mapping[str, float]) -> dict[str, float]:
    means = {p: float(np.mean(v)) for p, v in samples_by_party.items()}
    medians = {p: float(np.median(v)) for p, v in samples_by_party.items()}
    return {
        "mean_vote_mae": float(np.mean([abs(means[p] - actual_by_party[p]) for p in means])),
        "median_vote_mae": float(np.mean([abs(medians[p] - actual_by_party[p]) for p in medians])),
    }


def coverage_and_width(samples: np.ndarray, actual: float, levels: Sequence[float] = (0.5, 0.8, 0.9)) -> dict[str, dict[str, float | bool]]:
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        raise ValueError("Coverage requires at least one predictive draw")
    out: dict[str, dict[str, float | bool]] = {}
    for level in levels:
        alpha = (1.0 - level) / 2.0
        low, high = np.quantile(x, [alpha, 1.0 - alpha])
        out[str(int(level * 100))] = {
            "covered": bool(low <= actual <= high),
            "width": float(high - low),
            "lower": float(low),
            "upper": float(high),
        }
    return out


def evaluate_case_metrics(
    vote_draws: np.ndarray,
    actual_vote: np.ndarray,
    seat_draws: np.ndarray | None,
    actual_seats: np.ndarray | None,
    party_order: Sequence[str],
) -> dict[str, Any]:
    """Calculate all comparable metrics for one forecast case."""
    if vote_draws.ndim != 2 or vote_draws.shape[1] != len(party_order):
        raise ValueError("Vote draws do not match the declared party order")
    per_party = {
        party: {
            "vote_crps": continuous_crps(vote_draws[:, i], actual_vote[i]),
            "threshold_brier": threshold_brier(vote_draws[:, i], actual_vote[i]),
            "coverage_and_width": coverage_and_width(vote_draws[:, i], actual_vote[i]),
        }
        for i, party in enumerate(party_order)
    }
    vote_mae = party_mae({p: vote_draws[:, i] for i, p in enumerate(party_order)}, {p: actual_vote[i] for i, p in enumerate(party_order)})
    result: dict[str, Any] = {
        "per_party": per_party,
        "vote_crps_mean": float(np.mean([m["vote_crps"] for m in per_party.values()])),
        "threshold_brier_mean": float(np.mean([m["threshold_brier"] for m in per_party.values()])),
        "joint_vote_energy_score": energy_score(vote_draws, actual_vote),
        **vote_mae,
    }
    if seat_draws is None or actual_seats is None:
        result["seat_metrics_status"] = "UNAVAILABLE_NO_SEAT_DRAWS"
        result["seat_crps_mean"] = None
        result["joint_seat_energy_score"] = None
    else:
        from scripts.seat_hindcasts.metrics import calculate_discrete_seat_crps

        if seat_draws.ndim != 2 or seat_draws.shape[1] != len(party_order):
            raise ValueError("Seat draws do not match the declared party order")
        result["seat_metrics_status"] = "AVAILABLE"
        result["seat_crps_mean"] = float(np.mean([calculate_discrete_seat_crps(seat_draws[:, i], int(actual_seats[i])) for i in range(len(party_order))]))
        result["joint_seat_energy_score"] = energy_score(seat_draws, actual_seats)
    return result
