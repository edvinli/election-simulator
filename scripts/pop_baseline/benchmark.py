"""Matched-information benchmark for PoPBaseline v1 versus frozen Candidate A.

The harness deliberately keeps the evaluation layers separate:

* rolling future-PoP dynamics compares only the two opinion-movement models;
* election hindcasts compares national vote distributions from the same stored
  origin dates;
* final-poll diagnostics reports residual evidence and missing components
  rather than manufacturing a forecast where the source series cannot support
  one.

No Candidate-A parameter or artifact is written by this module.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.election_residuals.consensus import build_election_polling_consensus
from scripts.elections.load import load_election_targets_for_forecasting
from scripts.hindcasts.models import hindcast_dynamics_only, sample_shared_symmetric_dynamics
from scripts.pollofpolls.clr import composition_to_clr
from scripts.pollofpolls.state import load_timeseries_dataset
from scripts.pollofpolls.state_config import ALL_CATEGORIES, PARTIES
from scripts.pollofpolls.transitions import build_all_historical_transitions, filter_transitions_as_of
from scripts.election_residuals.config import EVALUATION_ELECTIONS

from .config import BASELINE_VERSION, DEFAULT_CONFIG, PARTY_ORDER, PoPBaselineConfig
from .metrics import aggregate_case_metrics, score_vote_draws
from .diagnostics import run_candidate_a_variance_diagnostic
from .model import derive_baseline_seed, simulate_baseline
from .threshold import run_threshold_support_diagnostic


DEFAULT_HORIZONS: tuple[int, ...] = (7, 14, 28, 56, 84, 112)
DEFAULT_ELECTIONS: tuple[date, ...] = (date(2018, 9, 9), date(2022, 9, 11))
DEFAULT_SAMPLES = 1_000
DEFAULT_ORIGIN_STEP_DAYS = 7
MODEL_A_ID = "election_simulator_v1_rc1_dynamics"
MODEL_B_ID = "pop_baseline_v1"
CASE_SCALAR_METRICS: tuple[str, ...] = (
    "vote_crps_mean_8parties",
    "vote_crps_mean_9parties",
    "threshold_brier_mean_8parties",
    "joint_vote_energy_score_9parties",
    "mean_vote_mae_8parties",
    "median_vote_mae_8parties",
    "mean_vote_mae_9parties",
    "median_vote_mae_9parties",
)
COVERAGE_METRICS: tuple[str, ...] = (
    "coverage_rate_8parties",
    "mean_width_8parties",
    "coverage_rate_9parties",
    "mean_width_9parties",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_metadata(repo_dir: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True, capture_output=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo_dir,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        tag_result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match", "HEAD"],
            cwd=repo_dir,
            text=True,
            capture_output=True,
            check=False,
        )
        implementation_paths = {
            "baseline_config": repo_dir / "scripts" / "pop_baseline" / "config.py",
            "baseline_model": repo_dir / "scripts" / "pop_baseline" / "model.py",
            "benchmark_harness": repo_dir / "scripts" / "pop_baseline" / "benchmark.py",
            "benchmark_metrics": repo_dir / "scripts" / "pop_baseline" / "metrics.py",
            "threshold_diagnostic": repo_dir / "scripts" / "pop_baseline" / "threshold.py",
            "candidate_a_national_engine": repo_dir / "scripts" / "vote_share_calibration" / "national_engine.py",
        }
        return {
            "source_git_commit": commit,
            "source_worktree_clean": not bool(status),
            "source_exact_tag": tag_result.stdout.strip() or None,
            "implementation_sha256": {
                name: _sha256_file(path) for name, path in implementation_paths.items() if path.is_file()
            },
        }
    except (OSError, subprocess.SubprocessError):
        return {"source_git_commit": "unknown", "source_worktree_clean": False}


def _row_to_vector(row: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([float(row["composition"][party]) for party in PARTY_ORDER], dtype=np.float64)


def _case_record(
    *,
    evaluation: str,
    origin_date: date,
    target_date: date,
    horizon_days: int,
    actual: np.ndarray,
    forecasts: Mapping[str, np.ndarray],
    samples: int,
    seed: int,
    source_kind: str,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model_id, draws in forecasts.items():
        metrics = score_vote_draws(draws, actual, PARTY_ORDER, threshold_parties=PARTIES)
        models[model_id] = metrics
    return {
        "evaluation": evaluation,
        "status": "SCORED",
        "origin_date": origin_date.isoformat(),
        "target_date": target_date.isoformat(),
        "horizon_days": int(horizon_days),
        "actual_vote_share_pct": {party: float(actual[i]) for i, party in enumerate(PARTY_ORDER)},
        "samples": int(samples),
        "seed": int(seed),
        "source_kind": source_kind,
        "models": models,
    }


def _skipped_record(
    *, evaluation: str, origin_date: date, target_date: date, horizon_days: int, reason: str
) -> dict[str, Any]:
    return {
        "evaluation": evaluation,
        "status": "SKIPPED",
        "origin_date": origin_date.isoformat(),
        "target_date": target_date.isoformat(),
        "horizon_days": int(horizon_days),
        "reason": reason,
    }


def run_rolling_dynamics_benchmark(
    *,
    data_dir: Path | str | None = None,
    start_date: date = date(2014, 9, 15),
    end_date: date | None = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    origin_step_days: int = DEFAULT_ORIGIN_STEP_DAYS,
    samples: int = DEFAULT_SAMPLES,
    seed: int = 12345,
    baseline_config: PoPBaselineConfig | None = None,
) -> list[dict[str, Any]]:
    """Compare baseline and RC1 dynamics against future stored PoP rows."""
    if origin_step_days <= 0:
        raise ValueError("origin_step_days must be positive")
    if samples <= 0:
        raise ValueError("samples must be positive")
    base = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "data" / "processed" / "pollofpolls"
    rows = load_timeseries_dataset(base / "pollofpolls_timeseries.csv")
    by_date = {row["date"]: row for row in rows}
    max_date = end_date or rows[-1]["date"]
    if start_date > max_date:
        raise ValueError("start_date cannot be after end_date")
    horizons = tuple(int(h) for h in horizons)
    all_transitions = build_all_historical_transitions(rows, horizons=horizons)
    config = baseline_config or PoPBaselineConfig(apply_support_voting=False)
    baseline_transitions = build_all_historical_transitions(rows, horizons=config.step_windows)
    output: list[dict[str, Any]] = []

    origin = start_date
    while origin <= max_date:
        for horizon in horizons:
            target = origin + timedelta(days=horizon)
            if target > max_date or target not in by_date or origin not in by_date:
                output.append(_skipped_record(
                    evaluation="rolling_future_pop",
                    origin_date=origin,
                    target_date=target,
                    horizon_days=horizon,
                    reason="missing_exact_origin_or_target_observation",
                ))
                continue
            eligible = filter_transitions_as_of(all_transitions[horizon], origin)
            if len(eligible) < 30:
                output.append(_skipped_record(
                    evaluation="rolling_future_pop",
                    origin_date=origin,
                    target_date=target,
                    horizon_days=horizon,
                    reason=f"insufficient_rc1_transitions:{len(eligible)}<30",
                ))
                continue
            origin_pop = by_date[origin]["composition"]
            base_forecast = simulate_baseline(
                origin_date=origin,
                horizon_days=horizon,
                samples_count=samples,
                seed=derive_baseline_seed(seed, origin, horizon, "rolling"),
                origin_pop=origin_pop,
                data_dir=base,
                config=config,
                _timeseries_data=rows,
                _transitions_by_window=baseline_transitions,
            )
            rc1_seed = derive_baseline_seed(seed, origin, horizon, "candidate-a-dynamics")
            rc1_delta = sample_shared_symmetric_dynamics(eligible, samples, rc1_seed)
            rc1_forecast = hindcast_dynamics_only(origin_pop, rc1_delta, categories=ALL_CATEGORIES)
            output.append(_case_record(
                evaluation="rolling_future_pop",
                origin_date=origin,
                target_date=target,
                horizon_days=horizon,
                actual=_row_to_vector(by_date[target]),
                forecasts={MODEL_B_ID: base_forecast.samples_matrix, MODEL_A_ID: rc1_forecast},
                samples=samples,
                seed=seed,
                source_kind="stored_pollofpolls_timeseries",
            ))
        origin += timedelta(days=origin_step_days)
    return output


def run_election_vote_benchmark(
    *,
    data_dir: Path | str | None = None,
    elections: Sequence[date] = DEFAULT_ELECTIONS,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    samples: int = DEFAULT_SAMPLES,
    seed: int = 12345,
    baseline_config: PoPBaselineConfig | None = None,
) -> list[dict[str, Any]]:
    """Compare national vote distributions for the 2018/2022 election targets."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    base = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "data" / "processed" / "pollofpolls"
    processed_root = base.parent
    rows = load_timeseries_dataset(base / "pollofpolls_timeseries.csv")
    by_date = {row["date"]: row for row in rows}
    targets = load_election_targets_for_forecasting(processed_root / "elections" / "riksdag_election_results.csv")
    config = baseline_config or PoPBaselineConfig(apply_support_voting=True)
    baseline_transitions = build_all_historical_transitions(rows, horizons=config.step_windows)
    output: list[dict[str, Any]] = []
    for election_date in elections:
        if election_date not in targets:
            raise KeyError(f"No official target for election {election_date}")
        actual = np.asarray([targets[election_date][party] for party in PARTY_ORDER], dtype=np.float64)
        for horizon in sorted((int(h) for h in horizons), reverse=True):
            origin = election_date - timedelta(days=horizon)
            if origin not in by_date:
                output.append(_skipped_record(
                    evaluation="election_vote_hindcast",
                    origin_date=origin,
                    target_date=election_date,
                    horizon_days=horizon,
                    reason="missing_exact_stored_pop_origin",
                ))
                continue
            baseline = simulate_baseline(
                origin_date=origin,
                horizon_days=horizon,
                samples_count=samples,
                seed=derive_baseline_seed(seed, origin, horizon, "election-baseline"),
                origin_pop=by_date[origin]["composition"],
                data_dir=base,
                config=config,
                _timeseries_data=rows,
                _transitions_by_window=baseline_transitions,
            )
            # Candidate A is called as an imported frozen engine.  It is not
            # edited or reconfigured here; only its national vote draws are
            # used because the matched comparison is a vote benchmark.
            from scripts.vote_share_calibration.national_engine import generate_national_vote_shares

            rc1_result = generate_national_vote_shares(
                as_of=origin,
                election_date=election_date,
                samples=samples,
                seed=derive_baseline_seed(seed, origin, horizon, "candidate-a-election"),
                data_dir=processed_root,
            )
            output.append(_case_record(
                evaluation="election_vote_hindcast",
                origin_date=origin,
                target_date=election_date,
                horizon_days=horizon,
                actual=actual,
                forecasts={MODEL_B_ID: baseline.samples_matrix, MODEL_A_ID: rc1_result.nat_shares_matrix * 100.0},
                samples=samples,
                seed=seed,
                source_kind="official_processed_election_results",
            ))
    return output


def run_final_poll_experiment(
    *,
    data_dir: Path | str | None = None,
    elections: Sequence[date] = EVALUATION_ELECTIONS,
) -> dict[str, Any]:
    """Report final-poll residual evidence and exact missing-origin limitations."""
    base = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "data" / "processed" / "pollofpolls"
    rows = load_timeseries_dataset(base / "pollofpolls_timeseries.csv")
    by_date = {row["date"]: row for row in rows}
    polls = pd.read_csv(base / "swedishpolls_individual_polls.csv")
    targets = load_election_targets_for_forecasting(base.parent / "elections" / "riksdag_election_results.csv")
    records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for election_date in elections:
        try:
            consensus = build_election_polling_consensus(election_date, polls)
            target = targets[election_date]
        except (KeyError, ValueError) as exc:
            missing.append({"election_date": election_date.isoformat(), "reason": str(exc)})
            continue
        residual = {party: float(target[party] - consensus.consensus_composition[party]) for party in PARTY_ORDER}
        # A final-poll experiment needs a point origin in the stored PoP series.
        # The dataset starts after the 2014 election, so this remains a clear
        # partial diagnostic rather than an invented candidate comparison.
        origin_candidates = [election_date - timedelta(days=d) for d in range(0, 15)]
        origin = next((d for d in origin_candidates if d in by_date), None)
        records.append({
            "election_date": election_date.isoformat(),
            "origin_available_within_14_days": origin is not None,
            "origin_date": origin.isoformat() if origin else None,
            "poll_consensus": consensus.consensus_composition,
            "actual": target,
            "residual_actual_minus_poll": residual,
            "poll_count": consensus.total_eligible_polls_in_window,
            "retained_pollsters": consensus.retained_pollsters_count,
            "status": "DIAGNOSTIC_ONLY_NO_MATCHED_DRAWS" if origin is None else "DIAGNOSTIC_ONLY",
        })
    status = "COMPLETE_DIAGNOSTIC_ONLY" if records and not missing else ("PARTIAL" if records else "NOT_RUN")
    return {
        "status": status,
        "description": "Final-poll residual diagnostic; no baseline or Candidate-A draws are fabricated when exact PoP origins are unavailable.",
        "probabilistic_evaluation": {
            "status": "NOT_RUN",
            "decision": "NOT_A_PROBABILISTIC_BRIER_BENCHMARK",
            "reason": "This component records deterministic final-poll consensus residuals only. It does not generate A/B threshold probabilities; older exact PoP origins and Candidate-A chronological pools are unavailable for a matched 2002-2014 probability run.",
            "probabilistic_cases_available": 0,
        },
        "records": records,
        "missing": missing,
        "evidence_type": "retrospective_diagnostic",
    }


def _aggregate_by_model(cases: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    model_rows: dict[str, list[Mapping[str, Any]]] = {}
    for case in cases:
        if case.get("status") != "SCORED":
            continue
        for model_id, metrics in case.get("models", {}).items():
            model_rows.setdefault(model_id, []).append(metrics)
    return {model: aggregate_case_metrics(rows) for model, rows in model_rows.items()}


def _aggregate_by_horizon(cases: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    horizons = sorted({int(c["horizon_days"]) for c in cases if c.get("status") == "SCORED"})
    for horizon in horizons:
        subset = [c for c in cases if c.get("status") == "SCORED" and int(c["horizon_days"]) == horizon]
        by_model = _aggregate_by_model(subset)
        output.append({"horizon_days": horizon, "models": by_model})
    return output


def _compact_case(case: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the JSON report concise; detailed party rows live in CSV evidence."""

    compact = {key: value for key, value in case.items() if key != "models"}
    compact_models: dict[str, Any] = {}
    for model_id, metrics in case.get("models", {}).items():
        compact_models[model_id] = {
            name: metrics.get(name) for name in CASE_SCALAR_METRICS
        }
    compact["models"] = compact_models
    return compact


def _flatten_case_metrics(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        for model_id, metrics in case.get("models", {}).items():
            row: dict[str, Any] = {
                "evaluation": case["evaluation"],
                "status": case["status"],
                "origin_date": case["origin_date"],
                "target_date": case["target_date"],
                "horizon_days": case["horizon_days"],
                "model": model_id,
            }
            row.update({name: metrics.get(name) for name in CASE_SCALAR_METRICS})
            for level, coverage in metrics.get("coverage_and_width", {}).items():
                for metric in COVERAGE_METRICS:
                    row[f"{level}_{metric}"] = coverage.get(metric)
            rows.append(row)
    return rows


def _flatten_party_metrics(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        for model_id, metrics in case.get("models", {}).items():
            for party, party_metrics in metrics.get("per_party", {}).items():
                row: dict[str, Any] = {
                    "evaluation": case["evaluation"],
                    "status": case["status"],
                    "origin_date": case["origin_date"],
                    "target_date": case["target_date"],
                    "horizon_days": case["horizon_days"],
                    "model": model_id,
                    "party": party,
                    "crps": party_metrics.get("crps"),
                    "threshold_probability": party_metrics.get("threshold_probability"),
                    "threshold_outcome": party_metrics.get("threshold_outcome"),
                    "threshold_brier": party_metrics.get("threshold_brier"),
                    "mean": party_metrics.get("mean"),
                    "median": party_metrics.get("median"),
                    "absolute_error_mean": party_metrics.get("absolute_error_mean"),
                    "absolute_error_median": party_metrics.get("absolute_error_median"),
                }
                for level, coverage in party_metrics.get("coverage_and_width", {}).items():
                    row[f"{level}_covered"] = coverage.get("covered")
                    row[f"{level}_width"] = coverage.get("width")
                rows.append(row)
    return rows


def _comparison_decision(aggregate_by_evaluation: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize a conservative model decision without selecting by one score."""

    score_names = (
        "vote_crps_mean_8parties",
        "joint_vote_energy_score_9parties",
        "threshold_brier_mean_8parties",
        "median_vote_mae_8parties",
    )
    comparisons: dict[str, Any] = {}
    for evaluation, by_model in aggregate_by_evaluation.items():
        candidate = by_model.get(MODEL_A_ID)
        baseline = by_model.get(MODEL_B_ID)
        if not candidate or not baseline:
            comparisons[evaluation] = {"status": "NOT_COMPARABLE"}
            continue
        rows: dict[str, str] = {}
        for metric in score_names:
            a = candidate.get(metric)
            b = baseline.get(metric)
            if a is None or b is None:
                rows[metric] = "UNAVAILABLE"
            elif float(a) < float(b):
                rows[metric] = MODEL_A_ID
            elif float(b) < float(a):
                rows[metric] = MODEL_B_ID
            else:
                rows[metric] = "TIE"
        comparisons[evaluation] = rows
    return {
        "status": "NO_CLEAR_UNIVERSAL_WINNER",
        "candidate_a_action": "KEEP_RC1",
        "automatic_adoption": False,
        "lower_is_better": list(score_names),
        "metric_winners": comparisons,
        "rationale": "The benchmark is retrospective comparative evidence. Candidate A is retained because score winners differ by evaluation/metric; no experimental layer is adopted from a mixed result.",
    }


def run_benchmark(
    *,
    data_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    samples: int = DEFAULT_SAMPLES,
    seed: int = 12345,
    origin_step_days: int = DEFAULT_ORIGIN_STEP_DAYS,
    start_date: date = date(2014, 9, 15),
    end_date: date | None = None,
    run_rolling: bool = True,
    run_elections: bool = True,
    run_final_poll: bool = True,
) -> dict[str, Any]:
    """Run available benchmark components and persist case-level evidence."""
    base = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "data" / "processed" / "pollofpolls"
    repo_dir = base.parents[2]
    config = PoPBaselineConfig()
    rolling_cases = run_rolling_dynamics_benchmark(
        data_dir=base,
        start_date=start_date,
        end_date=end_date,
        horizons=horizons,
        origin_step_days=origin_step_days,
        samples=samples,
        seed=seed,
        baseline_config=PoPBaselineConfig(apply_support_voting=False),
    ) if run_rolling else []
    election_cases = run_election_vote_benchmark(
        data_dir=base,
        horizons=horizons,
        samples=samples,
        seed=seed,
        baseline_config=config,
    ) if run_elections else []
    final_poll = run_final_poll_experiment(data_dir=base) if run_final_poll else {"status": "NOT_RUN"}
    threshold_diagnostic = run_threshold_support_diagnostic(data_dir=base) if run_elections else {"status": "NOT_RUN", "decision": "KEEP_RC1"}
    all_cases = rolling_cases + election_cases
    skipped_reason_counts = dict(sorted(Counter(
        str(case.get("reason", "unspecified"))
        for case in all_cases
        if case.get("status") != "SCORED"
    ).items()))
    if run_rolling or run_elections:
        try:
            underdispersion = run_candidate_a_variance_diagnostic(
                processed_root=base.parent,
                samples=samples,
                seed=seed,
                coverage_rows=all_cases,
            )
        except Exception as exc:  # noqa: BLE001 - preserve an explicit NOT_RUN report
            underdispersion = {
                "status": "NOT_RUN",
                "decision": "KEEP_RC1",
                "reason": f"variance diagnostic failed without modifying Candidate A: {type(exc).__name__}: {exc}",
            }
    else:
        underdispersion = {"status": "NOT_RUN", "decision": "KEEP_RC1", "reason": "benchmark components disabled"}

    source_files = {
        "timeseries": base / "pollofpolls_timeseries.csv",
        "individual_polls": base / "swedishpolls_individual_polls.csv",
        "election_results": base.parent / "elections" / "riksdag_election_results.csv",
    }
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "benchmark_version": "PoPBaseline-head_to_head-v1.0",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "benchmark_status": "COMPLETE" if all(c.get("status") == "SCORED" for c in all_cases) and final_poll.get("status") != "PARTIAL" else "PARTIAL",
        "evidence_type": "retrospective_comparative_2014_2026_and_2018_2022",
        "party_order": list(PARTY_ORDER),
        "threshold_parties": list(PARTIES),
        "models": {
            MODEL_B_ID: {"model_version": BASELINE_VERSION, "support_voting": config.apply_support_voting},
            MODEL_A_ID: {"model_version": "ElectionSimulator v1.0-rc1", "frozen": True},
        },
        "configuration": {
            "horizons_days": [int(h) for h in horizons],
            "samples": int(samples),
            "seed": int(seed),
            "origin_step_days": int(origin_step_days),
            "rolling_start_date": start_date.isoformat(),
            "rolling_end_date": end_date.isoformat() if end_date else None,
            "baseline_step_windows": list(config.step_windows),
            "baseline_partial_step_policy": config.partial_step_policy,
        },
        "source_files": {name: {"path": str(path), "sha256": _sha256_file(path)} for name, path in source_files.items()},
        "source": _source_metadata(repo_dir),
        # Keep full per-party metrics in the case-level CSV artifacts below;
        # embedding them in JSON made a full rolling report unnecessarily
        # large and difficult to review.
        "cases": [_compact_case(case) for case in all_cases],
        "case_level_artifacts": {
            "detail_policy": "Case aggregate metrics and all per-party metrics are persisted in CSV files when output_dir is provided.",
            "party_metrics_include": ["CRPS", "threshold Brier", "mean/median absolute error", "50/80/90 coverage and width"],
        },
        "aggregate_by_evaluation": {
            "rolling_future_pop": _aggregate_by_model(rolling_cases),
            "election_vote_hindcast": _aggregate_by_model(election_cases),
        },
        "aggregate_by_horizon": _aggregate_by_horizon(all_cases),
        "final_poll_experiment": final_poll,
        "threshold_support_diagnostic": threshold_diagnostic,
        "underdispersion_diagnostic": underdispersion,
        "comparison_decision": _comparison_decision({
            "rolling_future_pop": _aggregate_by_model(rolling_cases),
            "election_vote_hindcast": _aggregate_by_model(election_cases),
        }),
        "interpretation": {
            "no_automatic_winner_claim": True,
            "retrospective_warning": "2018/2022 cases are retrospective comparative evidence, not independent holdout validation.",
            "seat_metrics_status": "UNAVAILABLE_NO_BASELINE_SEAT_DRAWS",
            "missing_or_skipped_cases": sum(c.get("status") != "SCORED" for c in all_cases),
            "skipped_cases_by_reason": skipped_reason_counts,
            "comparison_policy": "Models receive the same exact stored origin, target, horizon, sample count, and outcome; cases lacking an exact origin/target or the Candidate-A minimum historical transition pool are skipped rather than imputed.",
        },
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "benchmark_report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        pd.DataFrame(_flatten_case_metrics(all_cases)).to_csv(out / "benchmark_case_metrics.csv", index=False)
        pd.DataFrame(_flatten_party_metrics(all_cases)).to_csv(out / "benchmark_party_metrics.csv", index=False)
        report["case_level_artifacts"].update({
            "case_metrics_csv": str(out / "benchmark_case_metrics.csv"),
            "party_metrics_csv": str(out / "benchmark_party_metrics.csv"),
        })
        # The report itself was already serialized above in the previous
        # implementation.  Rewrite it after adding the artifact paths so the
        # manifest is self-describing.
        with (out / "benchmark_report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run matched PoPBaseline v1 versus frozen ElectionSimulator RC1 benchmark")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/pop_baseline_benchmark"))
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--origin-step", type=int, default=DEFAULT_ORIGIN_STEP_DAYS)
    parser.add_argument("--start", default="2014-09-15")
    parser.add_argument("--end", default=None)
    parser.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS))
    parser.add_argument("--skip-rolling", action="store_true")
    parser.add_argument("--skip-elections", action="store_true")
    parser.add_argument("--skip-final-poll", action="store_true")
    args = parser.parse_args(argv)
    horizons = tuple(int(value.strip()) for value in args.horizons.split(",") if value.strip())
    report = run_benchmark(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        horizons=horizons,
        samples=args.samples,
        seed=args.seed,
        origin_step_days=args.origin_step,
        start_date=date.fromisoformat(args.start),
        end_date=date.fromisoformat(args.end) if args.end else None,
        run_rolling=not args.skip_rolling,
        run_elections=not args.skip_elections,
        run_final_poll=not args.skip_final_poll,
    )
    print(json.dumps({
        "benchmark_status": report["benchmark_status"],
        "case_count": len(report["cases"]),
        "scored_case_count": sum(c.get("status") == "SCORED" for c in report["cases"]),
        "final_poll_status": report["final_poll_experiment"]["status"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
