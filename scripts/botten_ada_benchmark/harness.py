"""Run the comparative benchmark without silently modifying Candidate A."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .adapters import ForecastBundle, load_bundle, unavailable_botten_ada_status
from .config import BOTTEN_ADA_SOURCE, HISTORICAL_ELECTION_DATES, HISTORICAL_HORIZONS, PARTY_ORDER, PIVOT_RULE
from .metrics import evaluate_case_metrics
from scripts.seat_hindcasts.config import EVALUATION_ELECTIONS


def _common_cases(a: ForecastBundle, b: ForecastBundle) -> list[tuple[Any, Any]]:
    b_by_key = {case.key: case for case in b.cases}
    pairs = [(case, b_by_key[case.key]) for case in a.cases if case.key in b_by_key]
    return pairs


def _validate_common_case(a: Any, b: Any) -> None:
    if a.election_date != b.election_date or a.as_of != b.as_of or a.horizon_days != b.horizon_days:
        raise ValueError("Candidates must use identical election dates, as-of cutoffs, and horizons")
    if a.vote_draws.shape[0] != b.vote_draws.shape[0]:
        raise ValueError("Candidates must use the same number of draws at each cutoff")


def _case_actual(a: Any, b: Any) -> tuple[Any, Any]:
    vote = a.actual_vote if a.actual_vote is not None else b.actual_vote
    seats = a.actual_seats if a.actual_seats is not None else b.actual_seats
    if vote is None:
        year = int(a.election_date[:4])
        target = EVALUATION_ELECTIONS.get(str(year))
        if target is not None:
            vote = [target["actual_shares"][party] for party in PARTY_ORDER]
            seats = [target["actual_seats"][party] for party in PARTY_ORDER]
    if vote is None:
        return None, None
    if seats is None and (a.seat_draws is not None or b.seat_draws is not None):
        # Seat metrics are unavailable unless the realized seat vector is
        # supplied; never treat a missing target as zero seats.
        seats = None
    return vote, seats


def _score_pair(a_case: Any, b_case: Any) -> dict[str, Any]:
    _validate_common_case(a_case, b_case)
    actual_vote, actual_seats = _case_actual(a_case, b_case)
    if actual_vote is None:
        return {
            "election_date": a_case.election_date,
            "as_of": a_case.as_of,
            "horizon_days": a_case.horizon_days,
            "status": "UNSCORED_NO_REALIZED_OUTCOME",
            "evidence_type": "prospective_unscored",
        }
    a_metrics = evaluate_case_metrics(a_case.vote_draws, actual_vote, a_case.seat_draws, actual_seats, PARTY_ORDER)
    b_metrics = evaluate_case_metrics(b_case.vote_draws, actual_vote, b_case.seat_draws, actual_seats, PARTY_ORDER)
    return {"election_date": a_case.election_date, "as_of": a_case.as_of, "horizon_days": a_case.horizon_days, "candidate_a": a_metrics, "candidate_b": b_metrics, "evidence_type": "comparative_retrospective" if int(a_case.election_date[:4]) in HISTORICAL_ELECTION_DATES else "prospective_unscored"}


def _pivot_decision(cases: list[dict[str, Any]]) -> dict[str, Any]:
    late = [
        c for c in cases
        if c.get("status") is None
        and c["horizon_days"] in PIVOT_RULE["priority_horizons_days"]
        and "candidate_a" in c and "candidate_b" in c
    ]
    if not late:
        return {"status": "NOT_ASSESSED_NO_SCORABLE_LATE_CASES", "rule": PIVOT_RULE}
    wins = sum(c["candidate_b"]["vote_crps_mean"] + PIVOT_RULE["minimum_score_improvement"] < c["candidate_a"]["vote_crps_mean"] for c in late)
    threshold_wins = sum(c["candidate_b"]["threshold_brier_mean"] + PIVOT_RULE["minimum_threshold_brier_improvement"] < c["candidate_a"]["threshold_brier_mean"] for c in late)
    if wins >= PIVOT_RULE["minimum_late_horizon_wins"] and threshold_wins >= PIVOT_RULE["minimum_late_horizon_wins"]:
        status = "TARGETED_LAYER_INVESTIGATION_ELIGIBLE"
    else:
        status = "KEEP_CANDIDATE_A_UNCHANGED"
    return {"status": status, "late_cases": len(late), "candidate_b_vote_crps_wins": wins, "candidate_b_threshold_brier_wins": threshold_wins, "rule": PIVOT_RULE}


def _aggregate_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Macro-average comparable metrics over scored cases, preserving missingness."""
    scored = [case for case in cases if "candidate_a" in case and "candidate_b" in case]
    result: dict[str, Any] = {"scored_case_count": len(scored)}
    for candidate in ("candidate_a", "candidate_b"):
        if not scored:
            result[candidate] = None
            continue
        names = ("vote_crps_mean", "joint_vote_energy_score", "threshold_brier_mean", "mean_vote_mae", "median_vote_mae", "seat_crps_mean", "joint_seat_energy_score")
        result[candidate] = {
            name: (None if not any(case[candidate].get(name) is not None for case in scored) else float(sum(case[candidate][name] for case in scored if case[candidate].get(name) is not None) / sum(case[candidate].get(name) is not None for case in scored)))
            for name in names
        }
        result[candidate]["coverage_and_width"] = {
            level: {
                "coverage_rate": float(sum(case[candidate]["per_party"][party]["coverage_and_width"][level]["covered"] for case in scored for party in PARTY_ORDER) / (len(scored) * len(PARTY_ORDER))),
                "mean_width": float(sum(case[candidate]["per_party"][party]["coverage_and_width"][level]["width"] for case in scored for party in PARTY_ORDER) / (len(scored) * len(PARTY_ORDER))),
            }
            for level in ("50", "80", "90")
        }
    return result


def run_benchmark(candidate_a_path: Path | str, candidate_b_path: Path | str | None = None, *, output_path: Path | str | None = None) -> dict[str, Any]:
    """Compare two standardized bundles or return an honest NOT_RUN report."""
    a = load_bundle(candidate_a_path, expected_candidate="A")
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_status": "NOT_RUN",
        "comparison_type": "comparative_retrospective_2018_2022_plus_prospective_2026",
        "party_order": list(PARTY_ORDER),
        "candidate_a": {"model_name": a.model_name, "model_version": a.model_version, "source": a.source, "file_sha256": a.source_file_sha256},
        "candidate_b": unavailable_botten_ada_status() if candidate_b_path is None else None,
        "pinned_botten_ada_source": BOTTEN_ADA_SOURCE,
        "pivot_rule": PIVOT_RULE,
        "metric_contract": {
            "vote_crps": "mean absolute error minus one-half mean pairwise absolute difference (V-statistic), in percentage points",
            "joint_energy_score": "E||X-y|| minus one-half E||X-X'|| using the V-statistic, party order fixed above",
            "threshold_brier": "(P(vote share >= 4.0%) - I(actual share >= 4.0%))^2, inclusive event",
            "sample_weighting": "equal weight per forecast case; per-party means are macro-averages across the eight parties",
            "coverage": "central empirical intervals at 50%, 80%, and 90%; width is upper minus lower",
            "seat_draw_policy": "seat metrics are unavailable when seat draws or realized seats are missing; no point forecast is converted into draws",
        },
        "cases": [],
        "aggregate": None,
    }
    if candidate_b_path is not None:
        b = load_bundle(candidate_b_path, expected_candidate="B")
        pairs = _common_cases(a, b)
        report["candidate_b"] = {"model_name": b.model_name, "model_version": b.model_version, "source": b.source, "file_sha256": b.source_file_sha256}
        report["cases"] = [_score_pair(a_case, b_case) for a_case, b_case in pairs]
        report["benchmark_status"] = "COMPLETE" if report["cases"] else "UNAVAILABLE_NO_COMMON_CASES"
        report["aggregate"] = _aggregate_metrics(report["cases"])
        report["pivot_decision"] = _pivot_decision(report["cases"])
        report["common_case_count"] = len(report["cases"])
    else:
        report["pivot_decision"] = {"status": "NOT_ASSESSED_EXTERNAL_MODEL_DATA_MISSING", "rule": PIVOT_RULE}
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare frozen ElectionSimulator Candidate A with an external Botten Ada bundle")
    parser.add_argument("--candidate-a", required=True, type=Path, help="Standardized Candidate A bundle JSON")
    parser.add_argument("--candidate-b", type=Path, default=None, help="Independently generated Botten Ada bundle JSON")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run_benchmark(args.candidate_a, args.candidate_b, output_path=args.output)
    print(json.dumps({"benchmark_status": report["benchmark_status"], "common_case_count": report.get("common_case_count", 0), "pivot_status": report["pivot_decision"]["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
