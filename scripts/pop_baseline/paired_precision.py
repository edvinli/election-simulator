"""Multi-seed, paired precision runs for the frozen A/B benchmark.

The existing benchmark already compares identical origin/target/horizon cases.
This wrapper adds fixed-seed replicates, explicit case-set hashes, paired
case-level score deltas, Monte Carlo stability summaries, and skip accounting.
It does not alter either model and intentionally keeps long runs opt-in.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .benchmark import (
    MODEL_A_ID,
    MODEL_B_ID,
    run_election_vote_benchmark,
    run_rolling_dynamics_benchmark,
)
from .threshold_metrics import build_threshold_brier_breakdown, summarize_threshold_by_dimensions


DEFAULT_PRECISION_SEEDS: tuple[int, ...] = (12345, 24680, 98765)
PRECISION_METRICS: tuple[str, ...] = (
    "vote_crps_mean_8parties",
    "joint_vote_energy_score_9parties",
    "threshold_brier_mean_8parties",
    "median_vote_mae_8parties",
)


def classify_skip_reason(reason: str) -> dict[str, str]:
    """Classify an explicit skip without trying to repair the missing input."""

    text = str(reason)
    if text == "missing_exact_origin_or_target_observation":
        return {
            "class": "input_data_gap",
            "explanation": "The stored Poll of Polls series has no exact origin or target observation; nearby dates are not substituted.",
            "resolution": "requires a provenance-backed exact observation or remains skipped",
        }
    if text == "missing_exact_stored_pop_origin":
        return {
            "class": "input_data_gap",
            "explanation": "No exact stored Poll of Polls origin exists for the requested historical case.",
            "resolution": "requires an exact historical PoP origin or remains skipped",
        }
    if text.startswith("insufficient_rc1_transitions:"):
        return {
            "class": "chronological_history_gap",
            "explanation": "The frozen Candidate-A transition pool has fewer than its minimum historical transitions at this as-of date.",
            "resolution": "cannot be eliminated without changing the frozen historical-data rule",
        }
    if text == "one_or_more_variants_not_run":
        return {
            "class": "variant_isolation_gap",
            "explanation": "At least one requested diagnostic component could not be isolated from an existing surface.",
            "resolution": "remains NOT_RUN; no approximation is permitted",
        }
    if text:
        return {
            "class": "other_explicit_skip",
            "explanation": "The case failed an explicit validation gate.",
            "resolution": "inspect the recorded reason; do not impute a forecast",
        }
    return {
        "class": "unspecified",
        "explanation": "No skip reason was supplied.",
        "resolution": "requires investigation before scoring",
    }


def summarize_skip_cases(cases: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Explain every explicit skipped case in a benchmark case list."""

    materialized = [case for case in cases if case.get("status") != "SCORED"]
    by_reason: Counter[str] = Counter(str(case.get("reason", "unspecified")) for case in materialized)
    by_evaluation: Counter[str] = Counter(str(case.get("evaluation", "unspecified")) for case in materialized)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in materialized:
        reason = str(case.get("reason", "unspecified"))
        if len(examples[reason]) < 5:
            examples[reason].append({
                "evaluation": case.get("evaluation"),
                "origin_date": case.get("origin_date"),
                "target_date": case.get("target_date"),
                "horizon_days": case.get("horizon_days"),
                "reason": reason,
            })
    return {
        "skipped_case_count": len(materialized),
        "by_reason": dict(sorted(by_reason.items())),
        "by_reason_detail": {
            reason: {"count": count, **classify_skip_reason(reason)}
            for reason, count in sorted(by_reason.items())
        },
        "by_evaluation": dict(sorted(by_evaluation.items())),
        "examples_by_reason": dict(sorted(examples.items())),
        "policy": "No exact origin, target, transition pool, or variant is imputed; each skip remains available for audit.",
    }


def audit_existing_benchmark_report(report_path: Path | str) -> dict[str, Any]:
    """Create a reproducible explanation of skips in an existing JSON report."""

    path = Path(report_path)
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    return {
        "schema_version": "1.0",
        "source_report": str(path),
        "source_report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "benchmark_status": report.get("benchmark_status"),
        "configuration": report.get("configuration", {}),
        "skip_audit": summarize_skip_cases(report.get("cases", [])),
        "interpretation": "This is an evidence-integrity audit of recorded skips, not a repair or a re-scoring operation.",
    }


def write_skip_audit(report_path: Path | str, output_dir: Path | str) -> Path:
    """Persist an explicit skip audit without modifying its source report."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    output_path = out / "skip_audit.json"
    output_path.write_text(
        json.dumps(audit_existing_benchmark_report(report_path), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def _identity_payload(case: Mapping[str, Any]) -> dict[str, Any]:
    """Return model-independent case identity, including exact outcome."""

    return {
        "evaluation": case.get("evaluation"),
        "origin_date": case.get("origin_date"),
        "target_date": case.get("target_date"),
        "horizon_days": int(case["horizon_days"]),
        "actual_vote_share_pct": case.get("actual_vote_share_pct"),
        "status": case.get("status"),
        "reason": case.get("reason"),
    }


def case_identity_hash(cases: Iterable[Mapping[str, Any]]) -> str:
    """Hash the sorted model-independent case set for reproducibility."""

    payload = sorted(
        (_identity_payload(case) for case in cases),
        key=lambda value: (
            str(value.get("evaluation")),
            str(value.get("origin_date")),
            str(value.get("target_date")),
            int(value.get("horizon_days", 0)),
        ),
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_paired_case_set(
    cases: Sequence[Mapping[str, Any]],
    *,
    model_a: str = MODEL_A_ID,
    model_b: str = MODEL_B_ID,
) -> dict[str, Any]:
    """Validate that every scored case contains both model forecasts."""

    scored = [case for case in cases if case.get("status") == "SCORED"]
    missing_models: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    duplicates: list[tuple[Any, ...]] = []
    for case in scored:
        identity = (
            case.get("evaluation"),
            case.get("origin_date"),
            case.get("target_date"),
            int(case["horizon_days"]),
        )
        if identity in seen:
            duplicates.append(identity)
        seen.add(identity)
        models = set(case.get("models", {}))
        if model_a not in models or model_b not in models:
            missing_models.append({"identity": list(identity), "models": sorted(models)})
    if missing_models or duplicates:
        raise ValueError(
            f"paired case validation failed: missing_models={len(missing_models)}, duplicates={len(duplicates)}"
        )
    return {
        "scored_case_count": len(scored),
        "skipped_case_count": sum(case.get("status") != "SCORED" for case in cases),
        "case_identity_hash": case_identity_hash(cases),
        "missing_models": missing_models,
        "duplicate_scored_cases": [list(item) for item in duplicates],
    }


def _paired_rows(
    cases: Sequence[Mapping[str, Any]],
    *,
    replicate_seed: int,
    evaluation_filter: str | None = None,
    model_a: str = MODEL_A_ID,
    model_b: str = MODEL_B_ID,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        if case.get("status") != "SCORED" or (
            evaluation_filter is not None and case.get("evaluation") != evaluation_filter
        ):
            continue
        metrics_a = case["models"][model_a]
        metrics_b = case["models"][model_b]
        for metric in PRECISION_METRICS:
            a = metrics_a.get(metric)
            b = metrics_b.get(metric)
            if a is None or b is None or not np.isfinite(float(a)) or not np.isfinite(float(b)):
                continue
            delta = float(a) - float(b)
            rows.append({
                "replicate_seed": int(replicate_seed),
                "evaluation": case.get("evaluation"),
                "origin_date": case.get("origin_date"),
                "target_date": case.get("target_date"),
                "horizon_days": int(case["horizon_days"]),
                "metric": metric,
                "model_a": model_a,
                "model_b": model_b,
                "model_a_score": float(a),
                "model_b_score": float(b),
                "delta_a_minus_b": delta,
                "a_wins": bool(delta < 0.0),
                "b_wins": bool(delta > 0.0),
                "tie": bool(delta == 0.0),
            })
    return rows


def _summarize_paired_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    # A seed replicate reuses the same realized outcome and case.  Collapse
    # those replicates by exact case before summarizing across cases; otherwise
    # the repeated seeds would masquerade as independent observations.
    case_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        case_groups[
            (
                row["evaluation"],
                row["origin_date"],
                row["target_date"],
                int(row["horizon_days"]),
                row["metric"],
            )
        ].append(float(row["delta_a_minus_b"]))
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for key, values in case_groups.items():
        groups[(key[0], key[-1])].append(float(np.mean(values)))
    output: list[dict[str, Any]] = []
    for (evaluation, metric), values in sorted(groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        arr = np.asarray(values, dtype=float)
        mean = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        output.append({
            "evaluation": evaluation,
            "metric": metric,
            "paired_case_count": len(arr),
            "mean_delta_a_minus_b": mean,
            "median_delta_a_minus_b": float(np.median(arr)),
            "sd_across_case_mean_deltas": sd,
            "min_case_mean_delta": float(np.min(arr)),
            "max_case_mean_delta": float(np.max(arr)),
            "a_wins": int(np.sum(arr < 0.0)),
            "b_wins": int(np.sum(arr > 0.0)),
            "ties": int(np.sum(arr == 0.0)),
            "lower_is_better": True,
            "interpretation": "Descriptive paired case means; seed replicates and rolling origins are not independent, so no inferential confidence interval is reported.",
        })
    return output


def _collapse_threshold_rows_across_seeds(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse repeated-seed threshold rows to one forecast per case.

    The Brier score is evaluated at the mean Monte Carlo probability across
    fixed seeds.  Seed spread is retained as a diagnostic column, while the
    realized election outcome is counted once.
    """

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("evaluation"),
            row.get("origin_date"),
            row.get("target_date"),
            int(row["horizon_days"]),
            row.get("model"),
            row.get("party"),
        )
        groups[key].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        probabilities = np.asarray([float(row["forecast_probability"]) for row in group], dtype=float)
        outcomes = {bool(row["actual_above_threshold"]) for row in group}
        if len(outcomes) != 1:
            raise ValueError(f"Repeated seed rows disagree on threshold outcome for case {key}")
        probability = float(np.mean(probabilities))
        outcome = bool(next(iter(outcomes)))
        row0 = group[0]
        output.append({
            **dict(row0),
            "forecast_probability": probability,
            "brier": float((probability - float(outcome)) ** 2),
            "seed_count": len(group),
            "forecast_probability_sd_across_seeds": float(np.std(probabilities, ddof=1)) if len(probabilities) > 1 else 0.0,
            "forecast_probability_min_across_seeds": float(np.min(probabilities)),
            "forecast_probability_max_across_seeds": float(np.max(probabilities)),
            "seed_aggregation": "mean_probability_then_single_case_brier",
        })
    return output


def _seed_summary(cases: Sequence[Mapping[str, Any]], seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for case in cases:
        if case.get("status") != "SCORED":
            continue
        for model_id, metrics in case.get("models", {}).items():
            for metric in PRECISION_METRICS:
                value = metrics.get(metric)
                if value is not None and np.isfinite(float(value)):
                    groups[(case.get("evaluation"), model_id, metric)].append(float(value))
    output: list[dict[str, Any]] = []
    for (evaluation, model_id, metric), values in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        output.append({
            "replicate_seed": int(seed),
            "evaluation": evaluation,
            "model": model_id,
            "metric": metric,
            "scored_case_count": len(values),
            "mean_score": float(np.mean(values)),
        })
    return output


def _stability_summary(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in seed_rows:
        groups[(row["evaluation"], row["model"], row["metric"])].append(float(row["mean_score"]))
    output: list[dict[str, Any]] = []
    for (evaluation, model_id, metric), values in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        arr = np.asarray(values, dtype=float)
        output.append({
            "evaluation": evaluation,
            "model": model_id,
            "metric": metric,
            "seed_count": len(arr),
            "mean_of_seed_scores": float(np.mean(arr)),
            "sd_across_seed_scores": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min_seed_score": float(np.min(arr)),
            "max_seed_score": float(np.max(arr)),
        })
    return output


def run_paired_precision_benchmark(
    *,
    data_dir: Path | str | None = None,
    seeds: Sequence[int] = DEFAULT_PRECISION_SEEDS,
    rolling_samples: int = 1_000,
    election_samples: int = 5_000,
    rolling_start: date = date(2014, 9, 15),
    rolling_end: date | None = None,
    rolling_origin_step_days: int = 7,
    horizons: Sequence[int] = (7, 14, 28, 56, 84, 112),
    run_rolling: bool = True,
    run_elections: bool = True,
) -> dict[str, Any]:
    """Run fixed-seed precision replicates and return non-mutating evidence."""

    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values or any(seed < 0 for seed in seed_values):
        raise ValueError("seeds must contain at least one non-negative integer")
    if rolling_samples <= 0 or election_samples <= 0:
        raise ValueError("sample counts must be positive")
    all_replicates: list[dict[str, Any]] = []
    all_paired_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    case_set_rows: list[dict[str, Any]] = []
    skip_counts: Counter[str] = Counter()

    for seed in seed_values:
        cases: list[dict[str, Any]] = []
        if run_rolling:
            cases.extend(
                run_rolling_dynamics_benchmark(
                    data_dir=data_dir,
                    start_date=rolling_start,
                    end_date=rolling_end,
                    horizons=horizons,
                    origin_step_days=rolling_origin_step_days,
                    samples=rolling_samples,
                    seed=seed,
                )
            )
        if run_elections:
            cases.extend(
                run_election_vote_benchmark(
                    data_dir=data_dir,
                    horizons=horizons,
                    samples=election_samples,
                    seed=seed,
                )
            )
        validation = validate_paired_case_set(cases)
        for case in cases:
            reason = str(case.get("reason", ""))
            if case.get("status") != "SCORED":
                skip_counts[reason or "unspecified"] += 1
        all_replicates.append({"seed": seed, "cases": cases, "validation": validation})
        all_paired_rows.extend(_paired_rows(cases, replicate_seed=seed))
        seed_rows.extend(_seed_summary(cases, seed))
        case_set_rows.append({
            "replicate_seed": seed,
            **validation,
        })

    hashes = [row["case_identity_hash"] for row in case_set_rows]
    common_case_set = len(set(hashes)) == 1 if hashes else True
    threshold_seed_rows: list[dict[str, Any]] = []
    for replicate in all_replicates:
        seed = int(replicate["seed"])
        rows = build_threshold_brier_breakdown(replicate["cases"], threshold_parties=None)
        threshold_seed_rows.extend({**row, "replicate_seed": seed} for row in rows)
    threshold_rows = _collapse_threshold_rows_across_seeds(threshold_seed_rows)

    return {
        "schema_version": "1.0",
        "benchmark": "paired_precision_threshold_attribution",
        "status": "COMPLETE" if common_case_set and not skip_counts else "PARTIAL",
        "evidence_type": "retrospective_comparative_diagnostic_not_holdout",
        "configuration": {
            "seeds": list(seed_values),
            "rolling_samples_per_case": int(rolling_samples),
            "election_samples_per_case": int(election_samples),
            "horizons_days": [int(h) for h in horizons],
            "rolling_start": rolling_start.isoformat(),
            "rolling_end": rolling_end.isoformat() if rolling_end else None,
            "rolling_origin_step_days": int(rolling_origin_step_days),
            "run_rolling": bool(run_rolling),
            "run_elections": bool(run_elections),
            "models": [MODEL_A_ID, MODEL_B_ID],
            "common_random_numbers_policy": (
                "Identical exact cases and deterministic base seeds are paired. "
                "The architectures use model-specific derived substreams; exact draw-level common random numbers are not claimed."
            ),
            "aggregation_policy": "Seed replicates are collapsed per exact case/model/party before threshold reliability aggregation; rolling origins remain temporally dependent and all summaries are descriptive.",
        },
        "case_set": {
            "same_case_set_across_seeds": common_case_set,
            "same_case_set_across_models": True,
            "replicate_hashes": case_set_rows,
        },
        "skip_accounting": {
            "total_skipped_case_replicates": int(sum(skip_counts.values())),
            "by_reason": dict(sorted(skip_counts.items())),
            "by_reason_detail": {
                reason: {"count": count, **classify_skip_reason(reason)}
                for reason, count in sorted(skip_counts.items())
            },
            "policy": "Skipped cases remain explicit; no unavailable origin, target, or transition pool is imputed.",
        },
        "seed_level_scores": seed_rows,
        "seed_stability": _stability_summary(seed_rows),
        "paired_case_deltas": _summarize_paired_rows(all_paired_rows),
        "threshold_brier": {
            "row_count": len(threshold_rows),
            "raw_seed_row_count": len(threshold_seed_rows),
            "dimensions": ["election_year", "horizon_days", "party", "forecast_probability", "actual_above_threshold"],
            "reliability": summarize_threshold_by_dimensions(threshold_rows),
        },
        # Keep detailed rows available to the writer but out of the JSON report
        # by default; callers can use ``detail_rows`` when they need CSV output.
        "detail_rows": {
            "paired_case_deltas": all_paired_rows,
            "threshold_brier_breakdown": threshold_rows,
            "threshold_brier_seed_rows": threshold_seed_rows,
        },
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames = sorted({str(key) for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_precision_artifacts(report: Mapping[str, Any], output_dir: Path | str) -> dict[str, str]:
    """Persist report plus auditable paired and threshold CSVs."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_copy = {key: value for key, value in report.items() if key != "detail_rows"}
    report_copy["artifacts"] = {
        "paired_case_deltas_csv": str(out / "paired_case_deltas.csv"),
        "threshold_brier_breakdown_csv": str(out / "threshold_brier_breakdown.csv"),
        "threshold_brier_seed_rows_csv": str(out / "threshold_brier_seed_rows.csv"),
        "case_set_manifest_csv": str(out / "case_set_manifest.csv"),
    }
    with (out / "precision_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report_copy, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    details = report.get("detail_rows", {})
    _write_csv(out / "paired_case_deltas.csv", details.get("paired_case_deltas", []))
    _write_csv(out / "threshold_brier_breakdown.csv", details.get("threshold_brier_breakdown", []))
    _write_csv(out / "threshold_brier_seed_rows.csv", details.get("threshold_brier_seed_rows", []))
    _write_csv(out / "case_set_manifest.csv", report.get("case_set", {}).get("replicate_hashes", []))
    return report_copy["artifacts"]


__all__ = [
    "DEFAULT_PRECISION_SEEDS",
    "PRECISION_METRICS",
    "case_identity_hash",
    "classify_skip_reason",
    "audit_existing_benchmark_report",
    "run_paired_precision_benchmark",
    "validate_paired_case_set",
    "summarize_skip_cases",
    "write_skip_audit",
    "write_precision_artifacts",
]
