"""Threshold-probability diagnostics for matched retrospective forecasts.

The simulator reports a probability of clearing the inclusive national 4%
threshold for each parliamentary party.  This module keeps that probability
diagnostic separate from model fitting: it only expands already-scored case
rows into auditable observations and fixed reliability summaries.

All functions are deterministic and use predeclared probability bins.  They
accept the in-memory case schema emitted by ``scripts.pop_baseline.benchmark``
and deliberately fail when a scored row does not contain both models or a
party-level threshold probability.  Missing data must therefore remain an
explicit skipped case rather than being silently imputed.
"""

from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


# Fixed bins are broad enough to remain interpretable with only a few
# elections.  The rightmost interval is closed at 1.0 by ``probability_bin``.
DEFAULT_PROBABILITY_BINS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def probability_bin(probability: float, bins: Sequence[float] = DEFAULT_PROBABILITY_BINS) -> str:
    """Return a fixed half-open probability bin label.

    ``1.0`` belongs to the final interval.  Values outside [0, 1] are
    rejected because a probability outside that range indicates a malformed
    forecast rather than a calibration observation.
    """

    p = float(probability)
    edges = tuple(float(x) for x in bins)
    if not np.isfinite(p) or p < 0.0 or p > 1.0:
        raise ValueError(f"threshold probability must be in [0, 1], got {p!r}")
    if len(edges) < 2 or any(not np.isfinite(x) for x in edges):
        raise ValueError("probability bins must contain at least two finite edges")
    if any(b <= a for a, b in zip(edges, edges[1:])) or edges[0] < 0.0 or edges[-1] > 1.0:
        raise ValueError("probability bins must be strictly increasing within [0, 1]")
    index = int(np.searchsorted(edges, p, side="right") - 1)
    index = min(max(index, 0), len(edges) - 2)
    return f"[{edges[index]:g},{edges[index + 1]:g}{']' if index == len(edges) - 2 else ')'}"


def _wilson_interval(successes: int, count: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if count <= 0:
        return None, None
    p = successes / count
    denominator = 1.0 + z * z / count
    centre = (p + z * z / (2.0 * count)) / denominator
    margin = z * sqrt((p * (1.0 - p) + z * z / (4.0 * count)) / count) / denominator
    return float(max(0.0, centre - margin)), float(min(1.0, centre + margin))


def _threshold_probability_from_metrics(
    party_metrics: Mapping[str, Any],
    *,
    actual: float,
    threshold: float,
) -> float:
    """Read a stored probability, with a backwards-compatible Brier fallback."""

    probability = party_metrics.get("threshold_probability")
    if probability is not None:
        return float(probability)

    # Reports created before threshold_probability was persisted still contain
    # a Brier score.  Reconstructing p from Brier is exact given the binary
    # outcome; this fallback is only for old evidence and is marked below.
    brier = party_metrics.get("threshold_brier")
    if brier is None:
        raise KeyError("party metrics must contain threshold_probability or threshold_brier")
    outcome = float(float(actual) >= threshold)
    distance = sqrt(max(0.0, float(brier)))
    return outcome - distance if outcome else distance


def build_threshold_brier_breakdown(
    cases: Iterable[Mapping[str, Any]],
    *,
    threshold: float = 4.0,
    threshold_parties: Sequence[str] | None = None,
    probability_bins: Sequence[float] = DEFAULT_PROBABILITY_BINS,
) -> list[dict[str, Any]]:
    """Expand scored cases into one auditable row per model/party.

    Each row contains the requested dimensions: election, horizon, party,
    forecast probability, and actual binary outcome.  ``brier`` is retained
    as a row-level proper score.  Rows from skipped cases are not fabricated;
    skipped counts are handled by the benchmark report.
    """

    rows: list[dict[str, Any]] = []
    for case in cases:
        if case.get("status") != "SCORED":
            continue
        actual_map = case.get("actual_vote_share_pct")
        if not isinstance(actual_map, Mapping):
            raise KeyError("scored case is missing actual_vote_share_pct")
        models = case.get("models")
        if not isinstance(models, Mapping):
            raise KeyError("scored case is missing models")
        for model_id, metrics in models.items():
            per_party = metrics.get("per_party", {})
            if threshold_parties is None:
                # REST is intentionally not a threshold party in the
                # canonical eight-party metric.  Older reports have an
                # explicit ``None`` for its threshold Brier, so infer the
                # eligible party set from that field rather than fabricating
                # a REST event.
                parties = tuple(
                    party
                    for party, party_metrics in per_party.items()
                    if party_metrics.get("threshold_probability") is not None
                    or party_metrics.get("threshold_brier") is not None
                )
            else:
                parties = tuple(threshold_parties)
            for party in parties:
                if party not in per_party or party not in actual_map:
                    raise KeyError(f"scored case/model is missing threshold party {party!r}")
                actual = float(actual_map[party])
                party_metrics = per_party[party]
                probability = _threshold_probability_from_metrics(
                    party_metrics, actual=actual, threshold=threshold
                )
                outcome = bool(actual >= threshold)
                brier = float((probability - float(outcome)) ** 2)
                stored_brier = party_metrics.get("threshold_brier")
                rows.append({
                    "evaluation": case.get("evaluation"),
                    "origin_date": case.get("origin_date"),
                    "target_date": case.get("target_date"),
                    "election_year": int(str(case["target_date"])[:4]),
                    "horizon_days": int(case["horizon_days"]),
                    "model": str(model_id),
                    "party": str(party),
                    "threshold_pct": float(threshold),
                    "forecast_probability": probability,
                    "probability_bin": probability_bin(probability, probability_bins),
                    "actual_vote_share_pct": actual,
                    "actual_above_threshold": outcome,
                    "brier": brier,
                    "stored_brier": float(stored_brier) if stored_brier is not None else None,
                    "probability_reconstructed_from_brier": stored_brier is not None and party_metrics.get("threshold_probability") is None,
                })
    return rows


def _group_summary(group: Sequence[Mapping[str, Any]], *, grouping: str, value: Any) -> dict[str, Any]:
    probabilities = np.asarray([float(row["forecast_probability"]) for row in group], dtype=float)
    outcomes = np.asarray([float(row["actual_above_threshold"]) for row in group], dtype=float)
    briers = np.asarray([float(row["brier"]) for row in group], dtype=float)
    successes = int(np.sum(outcomes))
    lo, hi = _wilson_interval(successes, len(group))
    return {
        "grouping": grouping,
        "group": value,
        "model": str(group[0]["model"]),
        "observation_count": len(group),
        "mean_forecast_probability": float(np.mean(probabilities)),
        "observed_rate": float(np.mean(outcomes)),
        "observed_successes": successes,
        "observed_rate_wilson95_low": lo,
        "observed_rate_wilson95_high": hi,
        "mean_brier": float(np.mean(briers)),
    }


def summarize_threshold_reliability(
    rows: Iterable[Mapping[str, Any]],
    *,
    groupings: Sequence[str] = ("model", "probability_bin", "party", "election_year", "horizon_days", "actual_above_threshold"),
) -> list[dict[str, Any]]:
    """Build fixed, descriptive calibration groupings from breakdown rows.

    This function intentionally performs no bin merging, smoothing, fitting,
    or model selection.  Sparse groups remain sparse and are reported as such.
    """

    materialized = list(rows)
    output: list[dict[str, Any]] = []
    for grouping in groupings:
        grouped: dict[tuple[Any, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in materialized:
            if grouping == "model":
                key = ("all", str(row["model"]))
            else:
                key = (row.get(grouping), str(row["model"]))
            grouped[key].append(row)
        for (value, _model), group in sorted(grouped.items(), key=lambda item: (str(item[0][0]), item[0][1])):
            summary = _group_summary(group, grouping=grouping, value=value)
            if grouping == "model":
                summary["group"] = "all"
            output.append(summary)
    return output


def summarize_threshold_by_dimensions(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Return the standard election/horizon/party/probability/outcome tables."""

    materialized = list(rows)
    return {
        "by_election": summarize_threshold_reliability(materialized, groupings=("model", "election_year")),
        "by_horizon": summarize_threshold_reliability(materialized, groupings=("model", "horizon_days")),
        "by_party": summarize_threshold_reliability(materialized, groupings=("model", "party")),
        "by_probability_bin": summarize_threshold_reliability(materialized, groupings=("model", "probability_bin")),
        "by_outcome": summarize_threshold_reliability(materialized, groupings=("model", "actual_above_threshold")),
    }


__all__ = [
    "DEFAULT_PROBABILITY_BINS",
    "build_threshold_brier_breakdown",
    "probability_bin",
    "summarize_threshold_by_dimensions",
    "summarize_threshold_reliability",
]
