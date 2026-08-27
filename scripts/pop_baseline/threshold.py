"""Pre-declared support-voting/threshold diagnostic for historical elections."""

from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scripts.election_residuals.consensus import build_election_polling_consensus
from scripts.election_residuals.config import EVALUATION_ELECTIONS
from scripts.elections.load import load_election_targets_for_forecasting
from scripts.pollofpolls.state_config import PARTIES


THRESHOLD_BANDS: tuple[tuple[str, float | None, float | None], ...] = (
    ("<3", None, 3.0),
    ("3-3.5", 3.0, 3.5),
    ("3.5-4", 3.5, 4.0),
    ("4-4.5", 4.0, 4.5),
    ("4.5-5", 4.5, 5.0),
    (">5", 5.0, None),
)
# The final-poll residual study has exact consensus provenance for all six
# available elections.  This diagnostic deliberately uses all six; the
# higher-horizon PoP benchmark remains restricted to exact stored origins.
DEFAULT_ELECTIONS: tuple[date, ...] = tuple(EVALUATION_ELECTIONS)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_threshold_band(value: float) -> str:
    """Classify a final-poll support value under fixed half-open boundaries."""
    x = float(value)
    for name, lower, upper in THRESHOLD_BANDS:
        if (lower is None or x >= lower) and (upper is None or x < upper):
            return name
    raise ValueError(f"Cannot classify support value {x}")


def _wilson_interval(successes: int, count: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if count <= 0:
        return (None, None)
    p = successes / count
    denominator = 1.0 + z * z / count
    centre = (p + z * z / (2.0 * count)) / denominator
    margin = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * count)) / count) / denominator
    return float(max(0.0, centre - margin)), float(min(1.0, centre + margin))


def run_threshold_support_diagnostic(
    *,
    data_dir: Path | str | None = None,
    elections: Sequence[date] = DEFAULT_ELECTIONS,
    min_observations_per_band: int = 8,
    min_independent_elections: int = 4,
    window_days: int = 14,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Quantify threshold-adjacent residuals without fitting flexible effects."""
    base = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "data" / "processed" / "pollofpolls"
    polls = pd.read_csv(base / "swedishpolls_individual_polls.csv")
    targets = load_election_targets_for_forecasting(base.parent / "elections" / "riksdag_election_results.csv")
    observations: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for election_date in elections:
        try:
            consensus = build_election_polling_consensus(election_date, polls, window_days=window_days)
            target = targets[election_date]
        except (KeyError, ValueError) as exc:
            missing.append({"election_date": election_date.isoformat(), "reason": str(exc)})
            continue
        for party in PARTIES:
            poll_value = float(consensus.consensus_composition[party])
            actual_value = float(target[party])
            observations.append({
                "election_date": election_date.isoformat(),
                "election_year": election_date.year,
                "party": party,
                "poll_consensus": poll_value,
                "actual": actual_value,
                "residual_actual_minus_poll": actual_value - poll_value,
                "band": classify_threshold_band(poll_value),
                "actual_above_4pct": bool(actual_value >= 4.0),
            })

    by_band_party: list[dict[str, Any]] = []
    for party in PARTIES:
        for band, _, _ in THRESHOLD_BANDS:
            subset = [row for row in observations if row["party"] == party and row["band"] == band]
            residuals = np.asarray([row["residual_actual_minus_poll"] for row in subset], dtype=float)
            outcomes = [bool(row["actual_above_4pct"]) for row in subset]
            successes = sum(outcomes)
            lo, hi = _wilson_interval(successes, len(outcomes))
            by_band_party.append({
                "party": party,
                "band": band,
                "observation_count": len(subset),
                "independent_election_count": len({row["election_year"] for row in subset}),
                "mean_residual_pp": float(np.mean(residuals)) if len(residuals) else None,
                "sd_residual_pp": float(np.std(residuals, ddof=1)) if len(residuals) > 1 else None,
                "threshold_successes": successes,
                "threshold_probability_empirical": float(successes / len(outcomes)) if outcomes else None,
                "threshold_probability_wilson95_low": lo,
                "threshold_probability_wilson95_high": hi,
                "sufficient_support": (
                    len(subset) >= min_observations_per_band
                    and len({row["election_year"] for row in subset}) >= min_independent_elections
                ),
            })

    sufficient_rows = [row for row in by_band_party if row["sufficient_support"]]
    diagnostic_status = "SUFFICIENT_SUPPORT" if sufficient_rows else "INSUFFICIENT_HISTORICAL_SUPPORT"
    focus_rows = [row for row in observations if 3.0 <= float(row["poll_consensus"]) <= 5.0]
    focus_passes = sum(bool(row["actual_above_4pct"]) for row in focus_rows)
    focus_failures = len(focus_rows) - focus_passes
    all_passes = sum(bool(row["actual_above_4pct"]) for row in observations)
    outcome_breakdown = [
        {
            "actual_outcome": "PASS_4PCT" if outcome else "FAIL_4PCT",
            "observation_count": sum(bool(row["actual_above_4pct"]) == outcome for row in observations),
            "mean_poll_consensus_pct": (
                float(np.mean([row["poll_consensus"] for row in observations if bool(row["actual_above_4pct"]) == outcome]))
                if any(bool(row["actual_above_4pct"]) == outcome for row in observations)
                else None
            ),
        }
        for outcome in (True, False)
    ]
    threshold_data_files = {
        "individual_polls": base / "swedishpolls_individual_polls.csv",
        "election_results": base.parent / "elections" / "riksdag_election_results.csv",
    }
    report = {
        "schema_version": "1.0",
        "diagnostic": "threshold_support_voting",
        "status": diagnostic_status,
        "decision": "ELIGIBLE_FOR_LEAVE_ONE_ELECTION_OUT_EXPERIMENT" if sufficient_rows else "KEEP_RC1",
        "evidence_type": "retrospective_diagnostic_not_holdout",
        "probabilistic_evaluation": {
            "status": "NOT_RUN",
            "decision": "NOT_A_PROBABILISTIC_BRIER_BENCHMARK",
            "reason": (
                "This six-election table evaluates deterministic final-poll consensus residuals and binary outcomes. "
                "It does not contain simulated A/B threshold probabilities. Exact matched probabilistic origins for "
                "2002-2014 are unavailable in the stored PoP timeseries; no probabilities are fabricated."
            ),
            "probabilistic_cases_available": 0,
        },
        "elections_requested": [e.isoformat() for e in elections],
        "elections_observed": sorted({row["election_year"] for row in observations}),
        "window_days": window_days,
        "bands": [{"name": name, "lower_inclusive": lower, "upper_exclusive": upper} for name, lower, upper in THRESHOLD_BANDS],
        "minimum_support_rule": {
            "observations_per_band": min_observations_per_band,
            "independent_elections_per_band": min_independent_elections,
        },
        "by_party_band": by_band_party,
        "missing_elections": missing,
        "observations": observations,
        "outcome_breakdown": outcome_breakdown,
        "focus_3_to_5_pct": {
            "lower_inclusive": 3.0,
            "upper_inclusive": 5.0,
            "observation_count": len(focus_rows),
            "pass_count": focus_passes,
            "fail_count": focus_failures,
            "pass_rate": float(focus_passes / len(focus_rows)) if focus_rows else None,
            "failure_cases": [row for row in focus_rows if not row["actual_above_4pct"]],
            "interpretation": (
                "No 3-5% final-poll failure is observed in the available six-election data. "
                "This absence is reported, not replaced with a synthetic failure."
                if focus_failures == 0
                else "Both pass and fail outcomes are observed in the 3-5% final-poll band."
            ),
        },
        "all_observation_outcomes": {
            "observation_count": len(observations),
            "pass_count": all_passes,
            "fail_count": len(observations) - all_passes,
        },
        "data_provenance": {
            "consensus_builder": "scripts.election_residuals.consensus.build_election_polling_consensus",
            "window_days": int(window_days),
            "files": {
                name: {"path": str(path), "sha256": _sha256_file(path)}
                for name, path in threshold_data_files.items()
                if path.is_file()
            },
            "target_loader": "scripts.elections.load.load_election_targets_for_forecasting",
        },
        "interpretation": "No threshold-conditioned layer is adopted by this diagnostic. A leave-one-election-out experiment is eligible only when the predeclared support rule is met.",
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    return report
