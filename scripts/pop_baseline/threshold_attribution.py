"""One-cycle threshold-loss attribution report for frozen Candidate A.

This is intentionally a diagnostic runner.  It compares the six declared
variants without tuning and applies a predeclared adoption gate.  A mixed or
underpowered result produces ``STOP_KEEP_RC1``; this module never edits the
production simulator or PoPBaseline.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .metrics import aggregate_case_metrics
from .paired_precision import case_identity_hash
from .threshold import DEFAULT_ELECTIONS, run_threshold_support_diagnostic
from .threshold_metrics import build_threshold_brier_breakdown, summarize_threshold_by_dimensions
from .variants import (
    VARIANT_A,
    VARIANT_B,
    VARIANT_C,
    VARIANT_D,
    VARIANT_E,
    VARIANT_F,
    VARIANT_ORDER,
    run_variant_election_benchmark,
    variant_contract,
)


# These are policy gates, not fitted parameters.  They are recorded in every
# report so an attribution result cannot be selected after inspecting scores.
MIN_THRESHOLD_BRIER_IMPROVEMENT = 0.005
MIN_THRESHOLD_WIN_RATE = 0.75
MAX_CRPS_DEGRADATION = 0.01
MAX_ENERGY_DEGRADATION = 0.02


# These pairings are the causal comparisons that can be made from the frozen
# surfaces.  Every delta below is candidate minus reference, so a negative
# value means that the candidate has the lower (better) loss.  The support
# comparison intentionally uses A/F rather than the RC1 reference B: it asks
# whether PoPBaseline's own support transfer explains its threshold result.
COMPONENT_PAIR_SPECS: tuple[dict[str, str], ...] = (
    {
        "component": "support_transfer",
        "delta_label": "F_minus_A",
        "candidate_variant": VARIANT_F,
        "reference_variant": VARIANT_A,
        "interpretation": "Positive F-minus-A threshold loss means disabling PoPBaseline support transfer worsened the threshold score; this is attribution evidence, not a tactical-voting adoption recommendation.",
    },
    {
        "component": "opinion_state_uncertainty",
        "delta_label": "C_minus_B",
        "candidate_variant": VARIANT_C,
        "reference_variant": VARIANT_B,
        "interpretation": "C-minus-B isolates removal of OpinionState sampling uncertainty while preserving the deterministic center and shared dynamics/residual draws.",
    },
    {
        "component": "pp_centered_noise",
        "delta_label": "D_minus_B",
        "candidate_variant": VARIANT_D,
        "reference_variant": VARIANT_B,
        "interpretation": "D-minus-B isolates removal of the frozen pp_centered_noise election residual layer.",
    },
    {
        "component": "dynamics",
        "delta_label": "E_minus_B",
        "candidate_variant": VARIANT_E,
        "reference_variant": VARIANT_B,
        "interpretation": "E-minus-B isolates replacement of RC1 dynamics with PoP-style dynamics while retaining the RC1 state and residual surfaces.",
    },
)

COMPONENT_METRICS: tuple[tuple[str, str], ...] = (
    ("threshold_brier", "threshold_brier_mean_8parties"),
    ("crps", "vote_crps_mean_8parties"),
    ("energy_score", "joint_vote_energy_score_9parties"),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _variant_score_table(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant_id in VARIANT_ORDER:
        scored = [
            case["models"][variant_id]
            for case in cases
            if case.get("status") == "SCORED" and variant_id in case.get("models", {})
        ]
        aggregate = aggregate_case_metrics(scored)
        rows.append({
            "variant": variant_id,
            "scored_case_count": len(scored),
            **(aggregate or {}),
        })
    return rows


def _variant_case_deltas(cases: Sequence[Mapping[str, Any]], variant_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        if case.get("status") != "SCORED":
            continue
        if VARIANT_B not in case.get("models", {}) or variant_id not in case.get("models", {}):
            continue
        reference = case["models"][VARIANT_B]
        candidate = case["models"][variant_id]
        rows.append({
            "variant": variant_id,
            "evaluation": case.get("evaluation"),
            "origin_date": case.get("origin_date"),
            "target_date": case.get("target_date"),
            "horizon_days": int(case["horizon_days"]),
            "threshold_brier_delta_candidate_minus_rc1": float(
                candidate["threshold_brier_mean_8parties"] - reference["threshold_brier_mean_8parties"]
            ),
            "threshold_brier_improvement_rc1_minus_candidate": float(
                reference["threshold_brier_mean_8parties"] - candidate["threshold_brier_mean_8parties"]
            ),
            "crps_delta_candidate_minus_rc1": float(
                candidate["vote_crps_mean_8parties"] - reference["vote_crps_mean_8parties"]
            ),
            "energy_delta_candidate_minus_rc1": float(
                candidate["joint_vote_energy_score_9parties"] - reference["joint_vote_energy_score_9parties"]
            ),
        })
    return rows


def _component_pair_case_deltas(
    cases: Sequence[Mapping[str, Any]],
    *,
    component: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Return paired per-case deltas for one declared component comparison."""

    candidate_id = component["candidate_variant"]
    reference_id = component["reference_variant"]
    rows: list[dict[str, Any]] = []
    for case in cases:
        if case.get("status") != "SCORED":
            continue
        models = case.get("models", {})
        if candidate_id not in models or reference_id not in models:
            continue
        candidate = models[candidate_id]
        reference = models[reference_id]
        for metric_name, field_name in COMPONENT_METRICS:
            candidate_score = candidate.get(field_name)
            reference_score = reference.get(field_name)
            if candidate_score is None or reference_score is None:
                continue
            candidate_value = float(candidate_score)
            reference_value = float(reference_score)
            if not np.isfinite(candidate_value) or not np.isfinite(reference_value):
                continue
            delta = candidate_value - reference_value
            rows.append({
                "component": component["component"],
                "delta_label": component["delta_label"],
                "candidate_variant": candidate_id,
                "reference_variant": reference_id,
                "evaluation": case.get("evaluation"),
                "origin_date": case.get("origin_date"),
                "target_date": case.get("target_date"),
                "horizon_days": int(case["horizon_days"]),
                "metric": metric_name,
                "candidate_score": candidate_value,
                "reference_score": reference_value,
                "delta_candidate_minus_reference": delta,
                "candidate_wins": bool(delta < 0.0),
                "reference_wins": bool(delta > 0.0),
                "tie": bool(delta == 0.0),
            })
    return rows


def build_component_attribution(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize the predeclared A/F and B/C/D/E component pairings.

    The output is deliberately descriptive.  It reports paired case means,
    spread, and win rates for threshold Brier, vote CRPS, and joint Energy
    Score.  It does not fit a model, select a tactical rule, or treat cases as
    independent inferential observations.
    """

    output: list[dict[str, Any]] = []
    for component in COMPONENT_PAIR_SPECS:
        case_rows = _component_pair_case_deltas(cases, component=component)
        row: dict[str, Any] = {
            **component,
            "delta_convention": "candidate_score_minus_reference_score; lower is better; negative means candidate wins",
            "paired_case_count": len({
                (
                    item["evaluation"],
                    item["origin_date"],
                    item["target_date"],
                    item["horizon_days"],
                )
                for item in case_rows
            }),
            "interpretation": component["interpretation"],
            "aggregation_note": "Descriptive paired case means; no inferential confidence interval is reported because these are retrospective cases and repeated simulation seeds are not independent.",
        }
        for metric_name, _field_name in COMPONENT_METRICS:
            metric_rows = [item for item in case_rows if item["metric"] == metric_name]
            deltas = np.asarray(
                [float(item["delta_candidate_minus_reference"]) for item in metric_rows],
                dtype=float,
            )
            prefix = metric_name
            if not metric_rows:
                row.update({
                    f"{prefix}_delta_candidate_minus_reference": None,
                    f"{prefix}_case_win_rate_candidate": None,
                    f"{prefix}_candidate_wins": 0,
                    f"{prefix}_reference_wins": 0,
                    f"{prefix}_ties": 0,
                })
                continue
            row.update({
                f"{prefix}_delta_candidate_minus_reference": float(np.mean(deltas)),
                f"{prefix}_median_delta_candidate_minus_reference": float(np.median(deltas)),
                f"{prefix}_sd_across_case_deltas": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                f"{prefix}_min_case_delta": float(np.min(deltas)),
                f"{prefix}_max_case_delta": float(np.max(deltas)),
                f"{prefix}_case_win_rate_candidate": float(np.mean(deltas < 0.0)),
                f"{prefix}_candidate_wins": int(np.sum(deltas < 0.0)),
                f"{prefix}_reference_wins": int(np.sum(deltas > 0.0)),
                f"{prefix}_ties": int(np.sum(deltas == 0.0)),
            })
        output.append(row)
    return output


def _gate_variant(deltas: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not deltas:
        return {
            "status": "NOT_RUN",
            "reason": "No paired scored cases for this variant",
            "passes": False,
        }
    threshold_improvements = np.asarray(
        [float(row["threshold_brier_improvement_rc1_minus_candidate"]) for row in deltas], dtype=float
    )
    crps_deltas = np.asarray([float(row["crps_delta_candidate_minus_rc1"]) for row in deltas], dtype=float)
    energy_deltas = np.asarray([float(row["energy_delta_candidate_minus_rc1"]) for row in deltas], dtype=float)
    target_wins = threshold_improvements >= MIN_THRESHOLD_BRIER_IMPROVEMENT
    target_mean = float(np.mean(threshold_improvements))
    target_win_rate = float(np.mean(target_wins))
    mean_crps = float(np.mean(crps_deltas))
    mean_energy = float(np.mean(energy_deltas))
    passes = bool(
        target_mean >= MIN_THRESHOLD_BRIER_IMPROVEMENT
        and target_win_rate >= MIN_THRESHOLD_WIN_RATE
        and mean_crps <= MAX_CRPS_DEGRADATION
        and mean_energy <= MAX_ENERGY_DEGRADATION
    )
    return {
        "status": "PASS" if passes else "FAIL",
        "passes": passes,
        "paired_case_count": len(deltas),
        "mean_threshold_brier_improvement": target_mean,
        "threshold_brier_win_rate": target_win_rate,
        "mean_crps_delta_candidate_minus_rc1": mean_crps,
        "mean_energy_delta_candidate_minus_rc1": mean_energy,
        "threshold_improvement_minimum": MIN_THRESHOLD_BRIER_IMPROVEMENT,
        "threshold_win_rate_minimum": MIN_THRESHOLD_WIN_RATE,
        "max_allowed_mean_crps_degradation": MAX_CRPS_DEGRADATION,
        "max_allowed_mean_energy_degradation": MAX_ENERGY_DEGRADATION,
    }


def apply_attribution_gate(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the fixed target/protection gate to C–F, without adoption."""

    deltas_by_variant = {
        variant_id: _variant_case_deltas(cases, variant_id)
        for variant_id in (VARIANT_C, VARIANT_D, VARIANT_E, VARIANT_F)
    }
    gates = {variant_id: _gate_variant(rows) for variant_id, rows in deltas_by_variant.items()}
    passing = [variant_id for variant_id, gate in gates.items() if gate.get("passes")]
    if len(passing) == 1:
        decision = "DIAGNOSTIC_CANDIDATE_SELECTED"
        selected = passing[0]
        rationale = "Exactly one declared component variant satisfies all predeclared gates; it remains diagnostic and is not applied to RC1 by this report."
    elif len(passing) > 1:
        decision = "NO_CLEAR_ATTRIBUTION_STOP_KEEP_RC1"
        selected = None
        rationale = "More than one component satisfies the gate; attribution is not unique, so RC1 is retained."
    else:
        decision = "NO_CLEAR_ATTRIBUTION_STOP_KEEP_RC1"
        selected = None
        rationale = "No component variant satisfies the predeclared target and protection gates; RC1 is retained."
    return {
        "decision": decision,
        "selected_variant": selected,
        "production_action": "KEEP_RC1",
        "automatic_adoption": False,
        "gate_policy": {
            "target_metric": "threshold_brier_mean_8parties",
            "target_improvement_is_rc1_minus_candidate": True,
            "minimum_mean_improvement": MIN_THRESHOLD_BRIER_IMPROVEMENT,
            "minimum_case_win_rate": MIN_THRESHOLD_WIN_RATE,
            "maximum_mean_crps_degradation": MAX_CRPS_DEGRADATION,
            "maximum_mean_energy_degradation": MAX_ENERGY_DEGRADATION,
            "candidate_variants": [VARIANT_C, VARIANT_D, VARIANT_E, VARIANT_F],
        },
        "by_variant": gates,
        "rationale": rationale,
        "evidence_type": "retrospective_component_diagnostic_not_holdout",
    }


def summarize_variant_cases(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return compact per-variant and per-election/horizon diagnostics."""

    by_variant = _variant_score_table(cases)
    by_case: list[dict[str, Any]] = []
    for case in cases:
        if case.get("status") != "SCORED":
            continue
        for variant_id, metrics in case.get("models", {}).items():
            by_case.append({
                "variant": variant_id,
                "evaluation": case.get("evaluation"),
                "target_date": case.get("target_date"),
                "horizon_days": int(case["horizon_days"]),
                "threshold_brier_mean_8parties": metrics.get("threshold_brier_mean_8parties"),
                "vote_crps_mean_8parties": metrics.get("vote_crps_mean_8parties"),
                "joint_vote_energy_score_9parties": metrics.get("joint_vote_energy_score_9parties"),
                "median_vote_mae_8parties": metrics.get("median_vote_mae_8parties"),
            })
    return {
        "by_variant": by_variant,
        "by_case": by_case,
    }


def run_threshold_attribution(
    *,
    processed_root: Path | str | None = None,
    elections: Sequence[date] = (date(2018, 9, 9), date(2022, 9, 11)),
    horizons: Sequence[int] = (7, 14, 28, 56, 84, 112),
    samples: int = 256,
    seed: int = 12345,
    final_poll_elections: Sequence[date] = DEFAULT_ELECTIONS,
) -> dict[str, Any]:
    """Run the final six-variant threshold attribution cycle."""

    cases = run_variant_election_benchmark(
        processed_root=processed_root,
        elections=elections,
        horizons=horizons,
        samples=samples,
        seed=seed,
    )
    scored = [case for case in cases if case.get("status") == "SCORED"]
    skipped = [case for case in cases if case.get("status") != "SCORED"]
    threshold_rows = build_threshold_brier_breakdown(
        scored,
        threshold_parties=None,
    )
    final_poll = run_threshold_support_diagnostic(
        data_dir=(Path(processed_root) / "pollofpolls") if processed_root else None,
        elections=final_poll_elections,
    )
    skip_counts: dict[str, int] = defaultdict(int)
    for case in skipped:
        skip_counts[str(case.get("reason", "unspecified"))] += 1
    report = {
        "schema_version": "1.0",
        "diagnostic": "threshold_loss_attribution",
        "status": "COMPLETE" if not skipped else "PARTIAL",
        "evidence_type": "retrospective_component_diagnostic_not_holdout",
        "configuration": {
            "elections": [e.isoformat() for e in elections],
            "horizons_days": [int(h) for h in horizons],
            "samples_per_case": int(samples),
            "seed": int(seed),
            "threshold_pct": 4.0,
            "common_case_policy": "Every scored A-F case has one exact origin, target, horizon, outcome, and sample count; unavailable cases are explicit skips.",
        },
        "variant_contract": variant_contract(),
        "case_set": {
            "total_cases": len(cases),
            "scored_cases": len(scored),
            "skipped_cases": len(skipped),
            "identity_hash": case_identity_hash(cases),
        },
        "skip_accounting": {
            "by_reason": dict(sorted(skip_counts.items())),
            "policy": "No exact origin, target, or unisolatable variant is imputed.",
        },
        "variant_scores": summarize_variant_cases(cases),
        "threshold_brier": {
            "row_count": len(threshold_rows),
            "dimensions": ["election_year", "horizon_days", "party", "forecast_probability", "actual_above_threshold"],
            "reliability": summarize_threshold_by_dimensions(threshold_rows),
        },
        "component_attribution": {
            "status": "DESCRIPTIVE_RETROSPECTIVE_DIAGNOSTIC",
            "by_component": build_component_attribution(cases),
            "interpretation": "The A/F support pairing is the reference comparison for PoPBaseline's support transfer; C/B, D/B, and E/B isolate the named RC1 layers. Positive candidate-minus-reference loss means the candidate/removal is worse. These pairings do not justify a tactical rule or any production change.",
        },
        "attribution_gate": apply_attribution_gate(cases),
        "final_poll_threshold_evidence": final_poll,
        "interpretation": {
            "claim_policy": "Diagnostic evidence only; no superiority claim and no production model change.",
            "retrospective_warning": "2018/2022 variant cases are retrospective and share historical model-development evidence.",
        },
        "detail_rows": {
            "cases": cases,
            "threshold_brier_breakdown": threshold_rows,
            "variant_deltas": {
                variant_id: _variant_case_deltas(cases, variant_id)
                for variant_id in (VARIANT_C, VARIANT_D, VARIANT_E, VARIANT_F)
            },
            "component_attribution_case_deltas": {
                component["component"]: _component_pair_case_deltas(cases, component=component)
                for component in COMPONENT_PAIR_SPECS
            },
        },
    }
    return report


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted({str(key) for row in materialized for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def write_threshold_attribution_artifacts(report: Mapping[str, Any], output_dir: Path | str) -> dict[str, str]:
    """Write JSON/CSV evidence without touching the publication contract."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    details = report.get("detail_rows", {})
    paths = {
        "report": str(out / "threshold_attribution_report.json"),
        "threshold_brier_breakdown": str(out / "threshold_brier_breakdown.csv"),
        "variant_case_scores": str(out / "variant_case_scores.csv"),
        "variant_deltas": str(out / "variant_deltas.csv"),
        "component_attribution": str(out / "component_attribution.csv"),
    }
    report_copy = {key: value for key, value in report.items() if key != "detail_rows"}
    report_copy["artifacts"] = paths
    with (out / "threshold_attribution_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report_copy, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    _write_csv(out / "threshold_brier_breakdown.csv", details.get("threshold_brier_breakdown", []))
    _write_csv(out / "variant_case_scores.csv", report.get("variant_scores", {}).get("by_case", []))
    delta_rows: list[dict[str, Any]] = []
    for rows in details.get("variant_deltas", {}).values():
        delta_rows.extend(rows)
    _write_csv(out / "variant_deltas.csv", delta_rows)
    _write_csv(
        out / "component_attribution.csv",
        report.get("component_attribution", {}).get("by_component", []),
    )
    return paths


__all__ = [
    "apply_attribution_gate",
    "build_component_attribution",
    "run_threshold_attribution",
    "summarize_variant_cases",
    "write_threshold_attribution_artifacts",
]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for an opt-in, diagnostic-only A–F threshold run."""

    parser = argparse.ArgumentParser(description="Run frozen A-F threshold-loss attribution diagnostics")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None, help="Processed data root (default: repository data/processed)")
    args = parser.parse_args(argv)
    report = run_threshold_attribution(
        processed_root=args.data_root,
        samples=args.samples,
        seed=args.seed,
    )
    if args.output is not None:
        paths = write_threshold_attribution_artifacts(report, args.output)
        print(json.dumps({"status": report["status"], "artifacts": paths}, indent=2))
    else:
        print(json.dumps({
            "status": report["status"],
            "case_set": report["case_set"],
            "attribution_gate": report["attribution_gate"],
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
