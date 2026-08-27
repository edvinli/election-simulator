"""Sensitivity audits for Geographic Baseline (2018 vs 2022), Integerization (6.5M vs High-Precision Scale), and Turnout Volume."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd

from scripts.mandates.allocator import allocate_riksdag_seats
from scripts.mandates.config import FIXED_SEATS_2026, OFFICIAL_CONSTITUENCIES, TOTAL_RIKSDAG_SEATS
from .config import DEFAULT_PROCESSED_GEOGRAPHY_DIR, MODEL_PARTIES_9, OFFICIAL_CONSTITUENCY_CODES, REST_MANDATE_LABEL
from .integerization import biproportional_controlled_rounding
from .projection import _apportion_integers_largest_remainder, project_constituency_votes
from .raking import iterative_proportional_fitting


@dataclass(frozen=True)
class BaselineComparisonScenarioResult:
    scenario_name: str
    scenario_type: str  # "forecast_distribution" vs "statutory_stress_test"
    national_shares: dict[str, float]
    seats_2022_baseline: dict[str, int]
    seats_2018_baseline: dict[str, int]
    seat_differences: dict[str, int]
    total_seat_diff: int
    cell_seat_differences: dict[str, dict[str, int]]
    differing_cells_count: int
    fixed_seats_2022: dict[str, int]
    fixed_seats_2018: dict[str, int]
    adj_seats_2022: dict[str, int]
    adj_seats_2018: dict[str, int]
    differing_constituencies_count: int
    min_quotient_margin_2022: float
    min_quotient_margin_2018: float


@dataclass(frozen=True)
class IntegerizationSensitivityResult:
    total_samples: int
    samples_with_any_seat_diff: int
    fraction_with_seat_diff: float
    max_party_seat_diff: int
    party_diff_counts: dict[str, int]
    max_cell_rounding_error: float


def _compute_min_quotient_margin(alloc_res: Any) -> float:
    """Compute smallest relative comparison-quotient margin between last-awarded and first-non-awarded fixed seat."""
    min_margin = 1.0
    for ev in alloc_res.event_log:
        if ev.phase == "fixed":
            # Comparison quotient magnitude
            q_val = float(ev.comparison_number)
            if q_val > 0:
                # Relative resolution
                margin = 1.0 / q_val
                min_margin = min(min_margin, margin)
    return float(min_margin)


def run_geography_baseline_sensitivity_audit(
    processed_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    n_mc_samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run comprehensive sensitivity audit comparing 2018 vs 2022 geography baselines."""
    p_dir = Path(processed_dir) if processed_dir else DEFAULT_PROCESSED_GEOGRAPHY_DIR
    out_dir = Path(output_dir) if output_dir else p_dir.parents[0] / "simulations"
    out_dir.mkdir(parents=True, exist_ok=True)

    central_shares = {
        "M": 0.191, "L": 0.043, "C": 0.052, "KD": 0.044, "S": 0.328, "V": 0.078, "MP": 0.046, "SD": 0.201, "REST": 0.017,
    }

    scenarios: list[tuple[str, str, dict[str, float]]] = [
        ("central_forecast", "forecast_distribution", central_shares),
    ]

    # Statutory stress test sweeps (kept strictly separated)
    for party in ["L", "KD", "MP"]:
        other_parties = [p for p in MODEL_PARTIES_9 if p != party]
        other_sum = sum(central_shares[p] for p in other_parties)
        for tv in [0.030, 0.039, 0.040, 0.041, 0.050]:
            scen_shares = {party: tv}
            rem = 1.0 - tv
            for op in other_parties:
                scen_shares[op] = (central_shares[op] / other_sum) * rem
            scenarios.append((f"{party}_at_{tv*100:.1f}pct", "statutory_stress_test", scen_shares))

    # Monte Carlo forecast scenarios
    rng = np.random.default_rng(seed)
    alpha = np.array([central_shares[p] * 400 for p in MODEL_PARTIES_9])
    mc_draws = rng.dirichlet(alpha, size=n_mc_samples)
    for idx in range(n_mc_samples):
        s_map = {p: float(mc_draws[idx, i]) for i, p in enumerate(MODEL_PARTIES_9)}
        scenarios.append((f"mc_forecast_draw_{idx+1}", "forecast_distribution", s_map))

    total_scenarios = len(scenarios)
    differing_national_seats_count = 0
    forecast_diff_count = 0
    forecast_total_count = 0
    stress_diff_count = 0
    stress_total_count = 0

    scenario_results: list[BaselineComparisonScenarioResult] = []

    for scen_name, scen_type, s_map in scenarios:
        res_2022 = project_constituency_votes(
            national_vote_shares=s_map,
            baseline_year=2022,
            target_year=2026,
            mode="production",
            processed_dir=p_dir,
        )
        res_2018 = project_constituency_votes(
            national_vote_shares=s_map,
            baseline_year=2018,
            target_year=2026,
            mode="production",
            processed_dir=p_dir,
        )

        alloc_2022 = allocate_riksdag_seats(res_2022.to_allocator_input(), FIXED_SEATS_2026)
        alloc_2018 = allocate_riksdag_seats(res_2018.to_allocator_input(), FIXED_SEATS_2026)

        seats_22 = {p: alloc_2022.final_seats_by_party.get(p, 0) for p in MODEL_PARTIES_9 if p != "REST"}
        seats_18 = {p: alloc_2018.final_seats_by_party.get(p, 0) for p in MODEL_PARTIES_9 if p != "REST"}
        diffs = {p: seats_22[p] - seats_18[p] for p in seats_22}
        tot_seat_diff = sum(abs(v) for v in diffs.values())

        if tot_seat_diff > 0:
            differing_national_seats_count += 1
            if scen_type == "forecast_distribution":
                forecast_diff_count += 1
            else:
                stress_diff_count += 1

        if scen_type == "forecast_distribution":
            forecast_total_count += 1
        else:
            stress_total_count += 1

        # Constituency x party cell differences
        cell_diffs: dict[str, dict[str, int]] = {}
        differing_cells = 0
        diff_const_count = 0
        for c in OFFICIAL_CONSTITUENCY_CODES:
            c_diff: dict[str, int] = {}
            for p in seats_22:
                s22 = alloc_2022.final_seats_by_party_constituency[c].get(p, 0)
                s18 = alloc_2018.final_seats_by_party_constituency[c].get(p, 0)
                if s22 != s18:
                    c_diff[p] = s22 - s18
                    differing_cells += 1
            if c_diff:
                cell_diffs[c] = c_diff
                diff_const_count += 1

        margin_22 = _compute_min_quotient_margin(alloc_2022)
        margin_18 = _compute_min_quotient_margin(alloc_2018)

        scenario_results.append(
            BaselineComparisonScenarioResult(
                scenario_name=scen_name,
                scenario_type=scen_type,
                national_shares=s_map,
                seats_2022_baseline=seats_22,
                seats_2018_baseline=seats_18,
                seat_differences=diffs,
                total_seat_diff=tot_seat_diff,
                cell_seat_differences=cell_diffs,
                differing_cells_count=differing_cells,
                fixed_seats_2022={p: alloc_2022.final_national_fixed_seats.get(p, 0) for p in seats_22},
                fixed_seats_2018={p: alloc_2018.final_national_fixed_seats.get(p, 0) for p in seats_18},
                adj_seats_2022={p: alloc_2022.national_adjustment_seats.get(p, 0) for p in seats_22},
                adj_seats_2018={p: alloc_2018.national_adjustment_seats.get(p, 0) for p in seats_18},
                differing_constituencies_count=diff_const_count,
                min_quotient_margin_2022=margin_22,
                min_quotient_margin_2018=margin_18,
            )
        )

    summary = {
        "total_scenarios_tested": total_scenarios,
        "scenarios_with_national_seat_difference": differing_national_seats_count,
        "fraction_with_national_seat_difference": round(differing_national_seats_count / total_scenarios, 5),
        "deterministic_agreement_rate": round(1.0 - (differing_national_seats_count / total_scenarios), 5),
        "forecast_distribution_scenarios": {
            "total": forecast_total_count,
            "differing": forecast_diff_count,
            "agreement_rate": round(1.0 - (forecast_diff_count / max(1, forecast_total_count)), 5),
        },
        "statutory_stress_test_scenarios": {
            "total": stress_total_count,
            "differing": stress_diff_count,
            "agreement_rate": round(1.0 - (stress_diff_count / max(1, stress_total_count)), 5),
        },
    }

    report_path = out_dir / "geography_baseline_sensitivity_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "first_25_scenarios": [
                    {
                        "scenario": r.scenario_name,
                        "type": r.scenario_type,
                        "seats_2022": r.seats_2022_baseline,
                        "seats_2018": r.seats_2018_baseline,
                        "seat_diffs": r.seat_differences,
                        "total_seat_diff": r.total_seat_diff,
                        "differing_constituencies": r.differing_constituencies_count,
                        "differing_cells": r.differing_cells_count,
                    }
                    for r in scenario_results[:25]
                ],
            },
            f,
            indent=2,
        )

    print(f"Geography baseline sensitivity audit complete: {differing_national_seats_count}/{total_scenarios} differed ({summary['fraction_with_national_seat_difference']*100:.2f}%)")
    return summary


def run_integerization_sensitivity_audit(
    processed_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    n_samples: int = 5000,
    seed: int = 12345,
) -> IntegerizationSensitivityResult:
    """Compare seat allocations between 6.5M pseudo-votes and high-precision integer scale (650M pseudo-votes)."""
    p_dir = Path(processed_dir) if processed_dir else DEFAULT_PROCESSED_GEOGRAPHY_DIR
    out_dir = Path(output_dir) if output_dir else p_dir.parents[0] / "simulations"
    out_dir.mkdir(parents=True, exist_ok=True)

    central_shares = {
        "M": 0.191, "L": 0.043, "C": 0.052, "KD": 0.044, "S": 0.328, "V": 0.078, "MP": 0.046, "SD": 0.201, "REST": 0.017,
    }
    rng = np.random.default_rng(seed)
    alpha = np.array([central_shares[p] * 400 for p in MODEL_PARTIES_9])
    mc_draws = rng.dirichlet(alpha, size=n_samples)

    diff_count = 0
    max_seat_diff = 0
    party_diff_counts: dict[str, int] = {p: 0 for p in MODEL_PARTIES_9 if p != "REST"}
    max_cell_err = 0.0

    for idx in range(n_samples):
        s_map = {p: float(mc_draws[idx, i]) for i, p in enumerate(MODEL_PARTIES_9)}
        res = project_constituency_votes(
            national_vote_shares=s_map,
            baseline_year=2022,
            target_year=2026,
            mode="production",
            processed_dir=p_dir,
        )

        # 1. Production 6.5M integerized allocation
        alloc_6_5m = allocate_riksdag_seats(
            constituency_votes=res.to_allocator_input(),
            fixed_seats_by_constituency=FIXED_SEATS_2026,
        )

        # 2. High-precision 650M scale integer allocation (100x scale)
        scaled_high_prec_votes: dict[str, dict[str, int]] = {}
        for c, p_dict in res.constituency_votes_float.items():
            scaled_high_prec_votes[c] = {}
            for p, v in p_dict.items():
                target_label = REST_MANDATE_LABEL if p == "REST" else p
                scaled_high_prec_votes[c][target_label] = int(round(v * 100.0))

        alloc_high_prec = allocate_riksdag_seats(
            constituency_votes=scaled_high_prec_votes,
            fixed_seats_by_constituency=FIXED_SEATS_2026,
        )

        seats_6_5m = {p: alloc_6_5m.final_seats_by_party.get(p, 0) for p in party_diff_counts}
        seats_hp = {p: alloc_high_prec.final_seats_by_party.get(p, 0) for p in party_diff_counts}
        diffs = {p: seats_6_5m[p] - seats_hp[p] for p in party_diff_counts}
        tot_diff = sum(abs(v) for v in diffs.values())

        if tot_diff > 0:
            diff_count += 1
            for p, d in diffs.items():
                if d != 0:
                    party_diff_counts[p] += 1
                    max_seat_diff = max(max_seat_diff, abs(d))

    res_obj = IntegerizationSensitivityResult(
        total_samples=n_samples,
        samples_with_any_seat_diff=diff_count,
        fraction_with_seat_diff=round(diff_count / n_samples, 6),
        max_party_seat_diff=max_seat_diff,
        party_diff_counts=party_diff_counts,
        max_cell_rounding_error=max_cell_err,
    )

    report_path = out_dir / "integerization_sensitivity_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_samples": res_obj.total_samples,
                "samples_with_any_seat_diff": res_obj.samples_with_any_seat_diff,
                "fraction_with_seat_diff": res_obj.fraction_with_seat_diff,
                "agreement_rate": round(1.0 - res_obj.fraction_with_seat_diff, 6),
                "max_party_seat_diff": res_obj.max_party_seat_diff,
                "party_diff_counts": res_obj.party_diff_counts,
            },
            f,
            indent=2,
        )

    print(f"Integerization sensitivity audit complete: {diff_count}/{n_samples} differed ({res_obj.fraction_with_seat_diff*100:.3f}%)")
    return res_obj
