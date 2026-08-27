"""Proper scoring rules for the matched PoPBaseline/Candidate-A benchmark."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def continuous_crps(samples: Sequence[float] | np.ndarray, actual: float) -> float:
    """Empirical CRPS using the exact V-statistic pairwise term."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("CRPS requires a non-empty one-dimensional sample array")
    ordered = np.sort(values)
    n = ordered.size
    term1 = float(np.mean(np.abs(ordered - float(actual))))
    pair_term = float(np.sum((2.0 * np.arange(n) - n + 1.0) * ordered) / (n * n))
    return term1 - pair_term


def energy_score(samples: np.ndarray, actual: np.ndarray, *, chunk_size: int = 512) -> float:
    """Multivariate Energy Score with a chunked V-statistic calculation."""
    values = np.asarray(samples, dtype=np.float64)
    target = np.asarray(actual, dtype=np.float64)
    if values.ndim != 2 or target.ndim != 1 or values.shape[1] != target.size or values.shape[0] == 0:
        raise ValueError("Energy Score requires non-empty (N,D) samples and matching (D,) target")
    term1 = float(np.mean(np.linalg.norm(values - target, axis=1)))
    pair_sum = 0.0
    for start in range(0, values.shape[0], chunk_size):
        chunk = values[start : start + chunk_size]
        pair_sum += float(np.sum(np.linalg.norm(chunk[:, None, :] - values[None, :, :], axis=2)))
    term2 = pair_sum / (2.0 * values.shape[0] * values.shape[0])
    return term1 - term2


def threshold_brier(samples: Sequence[float] | np.ndarray, actual: float, threshold: float = 4.0) -> float:
    """Brier score for inclusive ``share >= threshold``."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Threshold Brier requires a non-empty one-dimensional sample array")
    probability = float(np.mean(values >= threshold))
    outcome = float(float(actual) >= threshold)
    return float((probability - outcome) ** 2)


def threshold_probability(
    samples: Sequence[float] | np.ndarray,
    threshold: float = 4.0,
) -> float:
    """Return the empirical probability of an inclusive threshold event."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Threshold probability requires a non-empty one-dimensional sample array")
    if not np.isfinite(values).all():
        raise ValueError("Threshold probability samples must be finite")
    return float(np.mean(values >= float(threshold)))


def _central_interval(values: np.ndarray, level: float) -> tuple[float, float, bool, float]:
    alpha = (1.0 - level) / 2.0
    lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
    return float(lower), float(upper), False, float(upper - lower)


def score_vote_draws(
    samples: np.ndarray,
    actual: np.ndarray,
    party_order: Sequence[str],
    *,
    threshold_parties: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute the benchmark's per-party and joint vote metrics."""
    values = np.asarray(samples, dtype=np.float64)
    target = np.asarray(actual, dtype=np.float64)
    parties = tuple(party_order)
    if values.ndim != 2 or values.shape[1] != len(parties) or target.shape != (len(parties),):
        raise ValueError("Vote samples/actual do not match party_order")
    threshold_set = set(threshold_parties or parties)
    per_party: dict[str, dict[str, Any]] = {}
    means: list[float] = []
    medians: list[float] = []
    for index, party in enumerate(parties):
        draws = values[:, index]
        mean = float(np.mean(draws))
        median = float(np.median(draws))
        means.append(mean)
        medians.append(median)
        intervals: dict[str, dict[str, float | bool]] = {}
        for level in (0.50, 0.80, 0.90):
            low, high, _, width = _central_interval(draws, level)
            intervals[str(int(level * 100))] = {
                "lower": low,
                "upper": high,
                "covered": bool(low <= target[index] <= high),
                "width": width,
            }
        per_party[party] = {
            "crps": continuous_crps(draws, target[index]),
            "threshold_brier": threshold_brier(draws, target[index]) if party in threshold_set else None,
            "threshold_probability": threshold_probability(draws) if party in threshold_set else None,
            "threshold_outcome": bool(target[index] >= 4.0) if party in threshold_set else None,
            "mean": mean,
            "median": median,
            "absolute_error_mean": abs(mean - target[index]),
            "absolute_error_median": abs(median - target[index]),
            "coverage_and_width": intervals,
        }

    def _mean_metric(name: str, selected: Sequence[str]) -> float:
        vals = [per_party[p][name] for p in selected if per_party[p][name] is not None]
        return float(np.mean(vals)) if vals else float("nan")

    parl = tuple(p for p in parties if p != "REST")
    return {
        "per_party": per_party,
        "vote_crps_mean_8parties": _mean_metric("crps", parl),
        "vote_crps_mean_9parties": _mean_metric("crps", parties),
        "threshold_brier_mean_8parties": _mean_metric("threshold_brier", parl),
        "joint_vote_energy_score_9parties": energy_score(values, target),
        "mean_vote_mae_8parties": float(np.mean([abs(means[i] - target[i]) for i in range(len(parl))])),
        "median_vote_mae_8parties": float(np.mean([abs(medians[i] - target[i]) for i in range(len(parl))])),
        "mean_vote_mae_9parties": float(np.mean(np.abs(np.asarray(means) - target))),
        "median_vote_mae_9parties": float(np.mean(np.abs(np.asarray(medians) - target))),
        "coverage_and_width": {
            level: {
                "coverage_rate_8parties": float(np.mean([per_party[p]["coverage_and_width"][level]["covered"] for p in parl])),
                "mean_width_8parties": float(np.mean([per_party[p]["coverage_and_width"][level]["width"] for p in parl])),
                "coverage_rate_9parties": float(np.mean([per_party[p]["coverage_and_width"][level]["covered"] for p in parties])),
                "mean_width_9parties": float(np.mean([per_party[p]["coverage_and_width"][level]["width"] for p in parties])),
            }
            for level in ("50", "80", "90")
        },
    }


def aggregate_case_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Macro-average scalar and interval metrics over scored case rows."""
    if not rows:
        return None
    scalar_names = (
        "vote_crps_mean_8parties",
        "vote_crps_mean_9parties",
        "threshold_brier_mean_8parties",
        "joint_vote_energy_score_9parties",
        "mean_vote_mae_8parties",
        "median_vote_mae_8parties",
        "mean_vote_mae_9parties",
        "median_vote_mae_9parties",
    )
    out: dict[str, Any] = {"scored_case_count": len(rows)}
    for name in scalar_names:
        vals = [float(row[name]) for row in rows if row.get(name) is not None and np.isfinite(float(row[name]))]
        out[name] = float(np.mean(vals)) if vals else None
    out["coverage_and_width"] = {
        level: {
            metric: float(np.mean([float(row["coverage_and_width"][level][metric]) for row in rows]))
            for metric in (
                "coverage_rate_8parties",
                "mean_width_8parties",
                "coverage_rate_9parties",
                "mean_width_9parties",
            )
        }
        for level in ("50", "80", "90")
    }
    return out
