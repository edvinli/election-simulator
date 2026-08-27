"""Non-adaptive uncertainty attribution diagnostics for Candidate A comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.pollofpolls.state_config import ALL_CATEGORIES


def _trace_covariance(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        return 0.0
    return float(np.trace(np.cov(values, rowvar=False, ddof=1)))


def _to_fraction_matrix(matrix: np.ndarray, name: str) -> tuple[np.ndarray, str]:
    """Normalize a composition matrix while making its input units explicit."""

    values = np.asarray(matrix, dtype=np.float64)
    totals = np.sum(values, axis=1)
    if not np.isfinite(values).all() or not np.all(totals > 0.0):
        raise ValueError(f"{name} contains non-finite values or non-positive row totals")
    if np.allclose(totals, 1.0, atol=1e-6):
        return values / totals[:, None], "fractions"
    if np.allclose(totals, 100.0, atol=1e-4):
        return values / totals[:, None], "percentages"
    raise ValueError(f"{name} rows must sum to either 1 or 100; observed {float(np.median(totals)):g}")


def attribute_national_variance(
    *,
    opinion_state_draws: np.ndarray,
    base_comp_matrix: np.ndarray,
    nat_shares_matrix: np.ndarray,
    party_order: Sequence[str] = ALL_CATEGORIES,
) -> dict[str, Any]:
    """Report sequential variance increments for available Candidate-A layers.

    This is intentionally diagnostic rather than a model-fitting routine.  The
    increments are order-dependent (state -> dynamics -> election residual),
    so the report labels them as such and never uses them to alter RC1.
    """
    state = np.asarray(opinion_state_draws, dtype=np.float64)
    state_plus_dynamics = np.asarray(base_comp_matrix, dtype=np.float64)
    final = np.asarray(nat_shares_matrix, dtype=np.float64)
    expected_shape = (len(party_order),)
    if any(arr.ndim != 2 or arr.shape[1:] != expected_shape for arr in (state, state_plus_dynamics, final)):
        raise ValueError(f"All component matrices must have shape (N, {len(party_order)})")
    if not (state.shape[0] == state_plus_dynamics.shape[0] == final.shape[0]):
        raise ValueError("Component matrices must have the same sample count")

    # NationalVoteShareSampleResult historically exposes ``base_comp_matrix``
    # in percentages while its state/final surfaces are fractions.  Normalize
    # each surface before attribution so a unit mismatch cannot manufacture a
    # negative residual increment.
    state_fraction, state_units = _to_fraction_matrix(state, "opinion_state_draws")
    dynamic_fraction, dynamic_units = _to_fraction_matrix(state_plus_dynamics, "base_comp_matrix")
    final_fraction, final_units = _to_fraction_matrix(final, "nat_shares_matrix")
    state_pp = state_fraction * 100.0
    dynamic_pp = dynamic_fraction * 100.0
    final_pp = final_fraction * 100.0
    state_trace = _trace_covariance(state_pp)
    dynamic_trace = _trace_covariance(dynamic_pp)
    final_trace = _trace_covariance(final_pp)
    return {
        "schema_version": "1.0",
        "party_order": list(party_order),
        "sample_count": int(state.shape[0]),
        "units": "percentage_point_variance_trace",
        "input_units": {
            "opinion_state_draws": state_units,
            "base_comp_matrix": dynamic_units,
            "nat_shares_matrix": final_units,
        },
        "layers": {
            "opinion_state": {"trace": state_trace, "increment_from_previous": state_trace},
            "future_dynamics": {"trace_after_layer": dynamic_trace, "increment_from_previous": dynamic_trace - state_trace},
            "election_residual": {"trace_after_layer": final_trace, "increment_from_previous": final_trace - dynamic_trace},
        },
        "total_final_trace": final_trace,
        "attribution_warning": "Sequential increments are diagnostic and order-dependent; they are not orthogonal variance components and do not tune Candidate A.",
        "threshold_discontinuity": "Not identified by national vote matrices alone; requires a controlled seat counterfactual with the frozen allocator.",
    }


def compare_coverage_rows(
    case_rows: Sequence[Mapping[str, Any]],
    *,
    model_ids: Sequence[str],
) -> dict[str, Any]:
    """Aggregate stored benchmark coverage rows without fitting calibration."""
    result: dict[str, Any] = {}
    scored = [row for row in case_rows if row.get("status") == "SCORED"]
    for model_id in model_ids:
        metrics = [row["models"][model_id] for row in scored if model_id in row.get("models", {})]
        result[model_id] = {
            "scored_cases": len(metrics),
            "coverage_and_width": {
                level: {
                    key: float(np.mean([m["coverage_and_width"][level][key] for m in metrics])) if metrics else None
                    for key in ("coverage_rate_8parties", "mean_width_8parties", "coverage_rate_9parties", "mean_width_9parties")
                }
                for level in ("50", "80", "90")
            },
        }
    return result


def run_candidate_a_variance_diagnostic(
    *,
    processed_root: Path | str,
    as_of: str | None = None,
    election_date: str = "2026-09-13",
    samples: int = 256,
    seed: int = 12345,
    coverage_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Run a controlled, non-adaptive variance attribution for frozen RC1.

    The national engine exposes the three composition surfaces needed for the
    diagnostic.  This helper deliberately returns evidence only: it never
    changes a simulator setting, writes an artifact, or proposes a tuned
    replacement.  The threshold discontinuity remains unidentifiable from
    national draws alone and is reported as such.
    """

    from scripts.vote_share_calibration.national_engine import generate_national_vote_shares

    result = generate_national_vote_shares(
        as_of=as_of,
        election_date=election_date,
        samples=samples,
        seed=seed,
        data_dir=processed_root,
    )
    attribution = attribute_national_variance(
        opinion_state_draws=result.opinion_state_draws,
        base_comp_matrix=result.base_comp_matrix,
        nat_shares_matrix=result.nat_shares_matrix,
        party_order=ALL_CATEGORIES,
    )
    attribution["status"] = "DIAGNOSTIC_ONLY"
    attribution["as_of"] = result.as_of.isoformat()
    attribution["election_date"] = result.election_date.isoformat()
    attribution["candidate_a_diagnostics"] = result.diagnostics
    attribution["coverage_comparison"] = compare_coverage_rows(
        coverage_rows,
        model_ids=("pop_baseline_v1", "election_simulator_v1_rc1_dynamics"),
    )
    attribution["election_noise_scale_variant"] = {
        "status": "NOT_RUN",
        "decision": "KEEP_RC1",
        "reason": "Coverage attribution is diagnostic only; no predeclared leave-one-election-out scale experiment is warranted from two elections and this run does not alter the frozen candidate.",
    }
    return attribution
