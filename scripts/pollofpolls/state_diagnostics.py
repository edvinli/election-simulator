"""Comprehensive diagnostic suite and audit tool for Opinion State Estimator v1.

Provides deep-dive inspection of:
1. Reconstructed poll integrity and validation.
2. Percentage-point residuals before and after house-effect adjustment.
3. Top extreme polls by L2 disagreement.
4. REST volatility and Pearson correlation with ALR coordinates.
5. Raw vs Shrunk ALR covariance and correlation matrices.
6. Reference-category sensitivity (REST vs S vs M) with/without shrinkage.
7. Step-by-step variance attribution (A: Raw, B: HE-adj, C: Shrinkage, D: n_eff).
8. Binomial sampling error comparison against empirical residuals.
9. Historical time-window stability.
10. Pollster house effects in ALR vs Percentage-Point space.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Sequence

from .normalize import parse_date
from .state import (
    OpinionState,
    ReconstructedPoll,
    calculate_poll_reference_date,
    estimate_opinion,
    load_individual_polls_dataset,
    load_timeseries_dataset,
    subtract_calendar_years,
)
from .state_config import (
    ALL_CATEGORIES,
    COVARIANCE_DIAGONAL_SHRINKAGE,
    COVARIANCE_LOOKBACK_YEARS,
    MAX_EFFECTIVE_POLLS,
    MAX_ESTIMATE_MATCH_LAG_DAYS,
    MAX_SAMPLE_WEIGHT,
    MIN_POLLS_FOR_HOUSE_EFFECT,
    MIN_RESIDUAL_POLLS,
    MIN_SAMPLE_WEIGHT,
    MIN_SHARE_PCT,
    PARTIES,
    RECENCY_HALF_LIFE_DAYS,
    RECENT_POLL_LOOKBACK_DAYS,
    REFERENCE_CATEGORY,
    SAMPLE_SIZE_BENCHMARK,
)
from .state_math import (
    alr_to_composition,
    apply_covariance_shrinkage,
    calculate_percentile,
    calculate_sample_covariance,
    calculate_sample_mean,
    cholesky_decomposition_with_jitter,
    composition_to_alr,
    sample_multivariate_normal,
    summarize_samples,
)


def calculate_pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Calculate Pearson correlation coefficient between two sequences."""
    n = len(x)
    if n != len(y) or n < 2:
        raise ValueError(f"Sequences must be equal length >= 2, got {len(x)} and {len(y)}")
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    den = math.sqrt(sum((x[i] - mx) ** 2 for i in range(n)) * sum((y[i] - my) ** 2 for i in range(n)))
    if den == 0.0:
        return 0.0
    return num / den


def covariance_to_correlation(cov: Sequence[Sequence[float]]) -> list[list[float]]:
    """Convert a covariance matrix to a correlation matrix."""
    dim = len(cov)
    corr = [[0.0] * dim for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            diag_prod = cov[i][i] * cov[j][j]
            corr[i][j] = cov[i][j] / math.sqrt(diag_prod) if diag_prod > 0 else 0.0
    return corr


def calculate_distribution_stats(values: Sequence[float]) -> dict[str, float]:
    """Calculate comprehensive distribution statistics for a sequence of values."""
    n = len(values)
    if n == 0:
        raise ValueError("Cannot calculate statistics for empty sequence")
    sorted_v = sorted(values)
    mean_v = sum(values) / n
    var_v = sum((x - mean_v) ** 2 for x in values) / (n - 1 if n > 1 else 1)
    sd_v = math.sqrt(var_v)
    med_v = calculate_percentile(sorted_v, 0.50)
    abs_dev = sorted(abs(x - med_v) for x in values)
    mad_v = calculate_percentile(abs_dev, 0.50)
    return {
        "N": float(n),
        "mean": mean_v,
        "std_dev": sd_v,
        "median": med_v,
        "mad": mad_v,
        "p01": calculate_percentile(sorted_v, 0.01),
        "p05": calculate_percentile(sorted_v, 0.05),
        "p25": calculate_percentile(sorted_v, 0.25),
        "p75": calculate_percentile(sorted_v, 0.75),
        "p95": calculate_percentile(sorted_v, 0.95),
        "p99": calculate_percentile(sorted_v, 0.99),
        "min": min(values),
        "max": max(values),
    }


def run_reference_sensitivity_audit(
    residual_polls: Sequence[dict[str, Any]],
    timeseries_data: Sequence[dict[str, Any]],
    as_of: date,
    n_eff_used: float,
    seed: int = 12345,
    samples_count: int = 10_000,
) -> dict[str, Any]:
    """Run reference-category sensitivity diagnostic with shrinkage = 0.20 and shrinkage = 0.00."""
    central_row = [row for row in timeseries_data if row["date"] <= as_of][-1]
    results: dict[str, Any] = {}

    for shrinkage in (0.20, 0.00):
        shrink_key = f"shrinkage_{int(shrinkage * 100):02d}"
        results[shrink_key] = {}
        for ref_cat in (REFERENCE_CATEGORY, "S", "M"):
            other_cats = [c for c in ALL_CATEGORIES if c != ref_cat]

            def to_alr_ref(comp: dict[str, float]) -> list[float]:
                vals = {k: max(comp[k], MIN_SHARE_PCT) for k in ALL_CATEGORIES}
                tot = sum(vals.values())
                norm = {k: vals[k] * 100.0 / tot for k in ALL_CATEGORIES}
                ref_val = norm[ref_cat]
                return [math.log(norm[k] / ref_val) for k in other_cats]

            def from_alr_ref(alr_vec: Sequence[float]) -> dict[str, float]:
                shift = max(max(alr_vec), 0.0)
                exp_others = [math.exp(z - shift) for z in alr_vec]
                exp_ref = math.exp(0.0 - shift)
                tot_exp = sum(exp_others) + exp_ref
                res = {
                    other_cats[i]: 100.0 * exp_others[i] / tot_exp
                    for i in range(len(other_cats))
                }
                res[ref_cat] = 100.0 * exp_ref / tot_exp
                return res

            residuals = []
            for r in residual_polls:
                p_alr = to_alr_ref(r["poll"].composition)
                m_alr = to_alr_ref(r["matched"]["composition"])
                res = [p_alr[i] - m_alr[i] for i in range(len(other_cats))]
                residuals.append({"pollster": r["poll"].pollster, "res": res})

            pollster_res: dict[str, list[list[float]]] = {}
            for r in residuals:
                pollster_res.setdefault(r["pollster"], []).append(r["res"])

            he = {}
            for p_name, r_list in pollster_res.items():
                if len(r_list) >= MIN_POLLS_FOR_HOUSE_EFFECT:
                    he[p_name] = calculate_sample_mean(r_list)
                else:
                    he[p_name] = [0.0] * len(other_cats)

            adj_residuals = [
                [r["res"][i] - he[r["pollster"]][i] for i in range(len(other_cats))]
                for r in residuals
            ]

            cov_raw = calculate_sample_covariance(adj_residuals)
            cov_used = apply_covariance_shrinkage(cov_raw, shrinkage) if shrinkage > 0 else cov_raw
            state_cov = [
                [cov_used[i][j] / n_eff_used for j in range(len(other_cats))]
                for i in range(len(other_cats))
            ]

            L, jitter = cholesky_decomposition_with_jitter(state_cov)
            mean_alr = to_alr_ref(central_row["composition"])

            rng = random.Random(seed)
            draws = []
            for _ in range(samples_count):
                z = sample_multivariate_normal(mean_alr, L, rng)
                draws.append(from_alr_ref(z))

            results[shrink_key][ref_cat] = summarize_samples(draws)

    return results


def run_full_audit(
    as_of: str | date | None = None,
    data_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Execute complete Opinion State Estimator v1 audit across all 15 diagnostic categories."""
    base_path = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "data" / "processed" / "pollofpolls"
    ts_file = base_path / "pollofpolls_timeseries.csv"
    ind_file = base_path / "individual_polls.csv"

    timeseries_data = load_timeseries_dataset(ts_file)
    individual_polls, load_diagnostics = load_individual_polls_dataset(ind_file)

    max_ts_date = timeseries_data[-1]["date"]
    target_as_of = max_ts_date if as_of is None else (parse_date(as_of) if isinstance(as_of, str) else as_of)

    # 1. State production run
    prod_state = estimate_opinion(as_of=target_as_of, data_dir=base_path)

    # 2. Extract active 4-year residual polls
    window_start = subtract_calendar_years(target_as_of, COVARIANCE_LOOKBACK_YEARS)
    active_residual_matches: list[dict[str, Any]] = []

    for poll in individual_polls:
        if poll.publication_date is None or poll.publication_date >= target_as_of:
            continue
        if poll.interview_end is not None and poll.interview_end > target_as_of:
            continue
        if poll.interview_end is None or poll.reference_date is None:
            continue
        candidates = [row for row in timeseries_data if row["date"] <= poll.reference_date]
        if not candidates:
            continue
        matched = candidates[-1]
        lag = (poll.reference_date - matched["date"]).days
        if lag > MAX_ESTIMATE_MATCH_LAG_DAYS:
            continue
        if poll.publication_date >= window_start:
            active_residual_matches.append({"poll": poll, "matched": matched, "lag": lag})

    n_residuals = len(active_residual_matches)

    # Reconstruction validation
    unique_ids = set(r["poll"].poll_id for r in active_residual_matches)
    spans = [(r["poll"].pollster, r["poll"].interview_start, r["poll"].interview_end) for r in active_residual_matches]
    duplicate_spans = [span for span in spans if spans.count(span) > 1]
    rest_vals = [r["poll"].rest_value for r in active_residual_matches]

    reconstruction_report = {
        "residual_polls_count": n_residuals,
        "unique_poll_ids_count": len(unique_ids),
        "duplicate_poll_ids": n_residuals - len(unique_ids),
        "duplicate_fieldwork_spans": len(duplicate_spans),
        "zero_rest_polls_count": sum(1 for v in rest_vals if v == 0.0),
        "min_rest": min(rest_vals),
        "max_rest": max(rest_vals),
        "mean_rest": sum(rest_vals) / n_residuals,
    }

    # 3. PP Residuals (Raw vs House-Effect Adjusted)
    pp_residuals_raw: dict[str, list[float]] = {cat: [] for cat in ALL_CATEGORIES}
    for r in active_residual_matches:
        p = r["poll"]
        m = r["matched"]
        for party in PARTIES:
            pp_residuals_raw[party].append(p.party_values[party] - m[party])
        pp_residuals_raw[REFERENCE_CATEGORY].append(p.rest_value - m[REFERENCE_CATEGORY])

    pp_stats_raw = {cat: calculate_distribution_stats(pp_residuals_raw[cat]) for cat in ALL_CATEGORIES}

    pollster_groups: dict[str, list[int]] = {}
    for idx, r in enumerate(active_residual_matches):
        pollster_groups.setdefault(r["poll"].pollster, []).append(idx)

    pp_house_effects: dict[str, dict[str, float]] = {}
    for pollster, idxs in pollster_groups.items():
        if len(idxs) >= MIN_POLLS_FOR_HOUSE_EFFECT:
            pp_house_effects[pollster] = {
                cat: sum(pp_residuals_raw[cat][i] for i in idxs) / len(idxs)
                for cat in ALL_CATEGORIES
            }
        else:
            pp_house_effects[pollster] = {cat: 0.0 for cat in ALL_CATEGORIES}

    pp_residuals_adj: dict[str, list[float]] = {cat: [] for cat in ALL_CATEGORIES}
    for idx, r in enumerate(active_residual_matches):
        p_name = r["poll"].pollster
        for cat in ALL_CATEGORIES:
            pp_residuals_adj[cat].append(pp_residuals_raw[cat][idx] - pp_house_effects[p_name][cat])

    pp_stats_adj = {cat: calculate_distribution_stats(pp_residuals_adj[cat]) for cat in ALL_CATEGORIES}

    # 4. Top 20 Extreme Polls by L2 disagreement
    extreme_ranked = []
    for r in active_residual_matches:
        p = r["poll"]
        m = r["matched"]
        diffs = {party: round(p.party_values[party] - m[party], 2) for party in PARTIES}
        rest_diff = round(p.rest_value - m[REFERENCE_CATEGORY], 2)
        l2_err = math.sqrt(sum(d ** 2 for d in diffs.values()))
        extreme_ranked.append({
            "poll_id": p.poll_id,
            "pollster": p.pollster,
            "interview_start": p.interview_start.isoformat() if p.interview_start else None,
            "interview_end": p.interview_end.isoformat() if p.interview_end else None,
            "publication_date": p.publication_date.isoformat() if p.publication_date else None,
            "matched_pop_date": m["date"].isoformat(),
            "party_values": p.party_values,
            "pop_values": {party: m[party] for party in PARTIES},
            "pp_residuals": diffs,
            "poll_rest": round(p.rest_value, 2),
            "pop_rest": round(m[REFERENCE_CATEGORY], 2),
            "rest_residual": rest_diff,
            "l2_magnitude": round(l2_err, 4),
        })

    extreme_ranked.sort(key=lambda x: x["l2_magnitude"], reverse=True)
    top_20_extreme = extreme_ranked[:20]

    # 5. REST Volatility & ALR Correlation
    poll_rests = [r["poll"].rest_value for r in active_residual_matches]
    pop_rests = [r["matched"][REFERENCE_CATEGORY] for r in active_residual_matches]
    diff_rests = [poll_rests[i] - pop_rests[i] for i in range(n_residuals)]

    rest_distribution = {
        "poll_rest": calculate_distribution_stats(poll_rests),
        "pop_rest": calculate_distribution_stats(pop_rests),
        "diff_rest": calculate_distribution_stats(diff_rests),
    }

    alr_residuals_raw = [
        [r["poll"].alr_vector[i] - r["matched"]["alr"][i] for i in range(len(PARTIES))]
        for r in active_residual_matches
    ]

    rest_alr_correlations = {
        party: calculate_pearson_correlation(diff_rests, [res[i] for res in alr_residuals_raw])
        for i, party in enumerate(PARTIES)
    }

    # 6. ALR Covariance & Correlation Structure
    he_alr = prod_state.house_effects_alr
    alr_residuals_adj = [
        [alr_residuals_raw[idx][j] - he_alr[r["poll"].pollster][j] for j in range(len(PARTIES))]
        for idx, r in enumerate(active_residual_matches)
    ]
    cov_alr_raw = calculate_sample_covariance(alr_residuals_adj)
    cov_alr_shrunk = apply_covariance_shrinkage(cov_alr_raw, COVARIANCE_DIAGONAL_SHRINKAGE)

    corr_raw = covariance_to_correlation(cov_alr_raw)
    corr_shrunk = covariance_to_correlation(cov_alr_shrunk)

    off_raw = [corr_raw[i][j] for i in range(8) for j in range(i + 1, 8)]
    off_shrunk = [corr_shrunk[i][j] for i in range(8) for j in range(i + 1, 8)]

    alr_correlation_summary = {
        "raw_off_diagonal_mean": sum(off_raw) / len(off_raw),
        "raw_off_diagonal_min": min(off_raw),
        "raw_off_diagonal_max": max(off_raw),
        "shrunk_off_diagonal_mean": sum(off_shrunk) / len(off_shrunk),
        "shrunk_off_diagonal_min": min(off_shrunk),
        "shrunk_off_diagonal_max": max(off_shrunk),
        "raw_correlation_matrix": [[round(v, 4) for v in row] for row in corr_raw],
        "shrunk_correlation_matrix": [[round(v, 4) for v in row] for row in corr_shrunk],
    }

    # 7. Reference Category Sensitivity
    ref_sensitivity = run_reference_sensitivity_audit(
        active_residual_matches, timeseries_data, target_as_of, prod_state.effective_poll_count
    )

    # 8. Step Breakdown
    def sample_cov_variant(C: list[list[float]], n_eff: float) -> dict[str, dict[str, float]]:
        sc = [[C[i][j] / n_eff for j in range(8)] for i in range(8)]
        L, _ = cholesky_decomposition_with_jitter(sc)
        rng = random.Random(12345)
        draws = []
        for _ in range(10_000):
            z = sample_multivariate_normal(prod_state.mean_alr, L, rng)
            draws.append(alr_to_composition(z))
        return summarize_samples(draws)

    step_a = sample_cov_variant(calculate_sample_covariance(alr_residuals_raw), 1.0)
    step_b = sample_cov_variant(cov_alr_raw, 1.0)
    step_c = sample_cov_variant(cov_alr_shrunk, 1.0)
    step_d = sample_cov_variant(cov_alr_shrunk, prod_state.effective_poll_count)
    step_unshrunk = sample_cov_variant(cov_alr_raw, prod_state.effective_poll_count)

    modeling_step_breakdown = {
        "A_raw_neff1": {cat: round(step_a[cat]["std_dev"], 3) for cat in ALL_CATEGORIES},
        "B_he_adj_neff1": {cat: round(step_b[cat]["std_dev"], 3) for cat in ALL_CATEGORIES},
        "C_he_adj_shrunk_neff1": {cat: round(step_c[cat]["std_dev"], 3) for cat in ALL_CATEGORIES},
        "D_production_neff": {cat: round(step_d[cat]["std_dev"], 3) for cat in ALL_CATEGORIES},
        "Unshrunk_with_production_neff": {cat: round(step_unshrunk[cat]["std_dev"], 3) for cat in ALL_CATEGORIES},
    }

    # 9. Sampling Error Sanity Check
    sample_sizes = [
        r["poll"].sample_size for r in active_residual_matches if r["poll"].sample_size and r["poll"].sample_size > 0
    ]
    median_sample_size = calculate_percentile(sorted(sample_sizes), 0.50) if sample_sizes else 1000.0

    sampling_error_ratios = {}
    for party in PARTIES:
        p_pct = prod_state.mean_pct[party]
        p_prop = p_pct / 100.0
        se_list = [100.0 * math.sqrt(p_prop * (1.0 - p_prop) / n) for n in sample_sizes]
        med_se = calculate_percentile(sorted(se_list), 0.50)
        raw_sd = pp_stats_raw[party]["std_dev"]
        adj_sd = pp_stats_adj[party]["std_dev"]
        sampling_error_ratios[party] = {
            "point_estimate": p_pct,
            "median_binomial_se": round(med_se, 3),
            "empirical_raw_residual_sd": round(raw_sd, 3),
            "ratio_raw_to_median_se": round(raw_sd / med_se, 2),
            "empirical_adj_residual_sd": round(adj_sd, 3),
            "ratio_adj_to_median_se": round(adj_sd / med_se, 2),
        }

    # 10. Time Window Stability
    all_historical_matches = []
    for poll in individual_polls:
        if poll.publication_date and poll.interview_end and poll.reference_date:
            candidates = [row for row in timeseries_data if row["date"] <= poll.reference_date]
            if candidates and (poll.reference_date - candidates[-1]["date"]).days <= MAX_ESTIMATE_MATCH_LAG_DAYS:
                all_historical_matches.append({"poll": poll, "matched": candidates[-1], "pub": poll.publication_date})

    time_periods = (
        ("2018-01-01", "2019-12-31"),
        ("2020-01-01", "2021-12-31"),
        ("2022-01-01", "2023-12-31"),
        ("2024-01-01", target_as_of.isoformat()),
    )
    time_stability: dict[str, dict[str, Any]] = {}
    for start_str, end_str in time_periods:
        period_key = f"{start_str}..{end_str}"
        d_s = parse_date(start_str)
        d_e = parse_date(end_str)
        in_p = [r for r in all_historical_matches if d_s <= r["pub"] <= d_e]
        time_stability[period_key] = {"count": len(in_p), "sd_by_party": {}}
        if in_p:
            for cat in ALL_CATEGORIES:
                diffs = [
                    (r["poll"].party_values[cat] - r["matched"][cat])
                    if cat != REFERENCE_CATEGORY
                    else (r["poll"].rest_value - r["matched"][REFERENCE_CATEGORY])
                    for r in in_p
                ]
                m_diff = sum(diffs) / len(diffs)
                sd_diff = math.sqrt(sum((x - m_diff) ** 2 for x in diffs) / (len(diffs) - 1))
                time_stability[period_key]["sd_by_party"][cat] = round(sd_diff, 3)

    # 11. Pollster House Effects ALR vs PP
    pollster_audit: dict[str, Any] = {}
    for pollster, r_list in sorted(pollster_groups.items()):
        if len(r_list) < MIN_POLLS_FOR_HOUSE_EFFECT:
            continue
        alr_vec = prod_state.house_effects_alr[pollster]
        pp_dict = pp_house_effects[pollster]
        pollster_audit[pollster] = {
            "poll_count": len(r_list),
            "alr_house_effect": {party: round(alr_vec[i], 4) for i, party in enumerate(PARTIES)},
            "pp_mean_residual": {cat: round(pp_dict[cat], 3) for cat in ALL_CATEGORIES},
        }

    return {
        "as_of": target_as_of.isoformat(),
        "reconstruction_report": reconstruction_report,
        "pp_stats_raw": pp_stats_raw,
        "pp_stats_adjusted": pp_stats_adj,
        "top_20_extreme_polls": top_20_extreme,
        "rest_distribution": rest_distribution,
        "rest_alr_correlations": rest_alr_correlations,
        "alr_correlation_summary": alr_correlation_summary,
        "reference_sensitivity": ref_sensitivity,
        "modeling_step_breakdown": modeling_step_breakdown,
        "sampling_error_ratios": sampling_error_ratios,
        "time_stability": time_stability,
        "pollster_audit": pollster_audit,
    }


def main(args_list: Sequence[str] | None = None) -> int:
    """CLI entry point for opinion state audit."""
    parser = argparse.ArgumentParser(description="Run statistical audit on Opinion State Estimator v1.")
    parser.add_argument("--as-of", dest="as_of", default=None, help="Target as-of date (YYYY-MM-DD).")
    parser.add_argument("--json", dest="json_out", action="store_true", help="Print JSON report.")
    parser.add_argument("--data-dir", dest="data_dir", default=None, help="Custom data directory path.")
    args = parser.parse_args(args_list)

    try:
        report = run_full_audit(as_of=args.as_of, data_dir=args.data_dir)
        if args.json_out:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print("=================================================================")
            print(f"        OPINION STATE ESTIMATOR v1 STATISTICAL AUDIT ({report['as_of']})")
            print("=================================================================")
            rec = report["reconstruction_report"]
            print(f"Residual Polls: {rec['residual_polls_count']} (Unique IDs: {rec['unique_poll_ids_count']}, Duplicate spans: {rec['duplicate_fieldwork_spans']})")
            print(f"REST Range: {rec['min_rest']:.2f}% to {rec['max_rest']:.2f}% (Mean: {rec['mean_rest']:.2f}%, Zeros: {rec['zero_rest_polls_count']})")
            print("-----------------------------------------------------------------")
            print("Percentage-Point Residuals vs PoP:")
            print("Party  | Raw SD | Raw Med | HE-adj SD | HE-adj Med | Median SRS SE | Emp/SE Ratio")
            print("-------+--------+---------+-----------+------------+---------------+-------------")
            for p in PARTIES:
                raw_sd = report["pp_stats_raw"][p]["std_dev"]
                raw_med = report["pp_stats_raw"][p]["median"]
                adj_sd = report["pp_stats_adjusted"][p]["std_dev"]
                adj_med = report["pp_stats_adjusted"][p]["median"]
                se_info = report["sampling_error_ratios"][p]
                print(f"{p:<6} | {raw_sd:>6.2f} | {raw_med:>7.2f} | {adj_sd:>9.2f} | {adj_med:>10.2f} | {se_info['median_binomial_se']:>13.2f} | {se_info['ratio_adj_to_median_se']:>11.2f}")
            rest_raw = report["pp_stats_raw"][REFERENCE_CATEGORY]["std_dev"]
            rest_adj = report["pp_stats_adjusted"][REFERENCE_CATEGORY]["std_dev"]
            print(f"{REFERENCE_CATEGORY:<6} | {rest_raw:>6.2f} | {report['pp_stats_raw'][REFERENCE_CATEGORY]['median']:>7.2f} | {rest_adj:>9.2f} | {report['pp_stats_adjusted'][REFERENCE_CATEGORY]['median']:>10.2f} |           N/A |         N/A")
            print("-----------------------------------------------------------------")
            cor_sum = report["alr_correlation_summary"]
            print(f"ALR Raw Off-Diagonal Correlation:   Mean = {cor_sum['raw_off_diagonal_mean']:.4f} (Min={cor_sum['raw_off_diagonal_min']:.4f}, Max={cor_sum['raw_off_diagonal_max']:.4f})")
            print(f"ALR Shrunk Off-Diagonal Correlation: Mean = {cor_sum['shrunk_off_diagonal_mean']:.4f} (Min={cor_sum['shrunk_off_diagonal_min']:.4f}, Max={cor_sum['shrunk_off_diagonal_max']:.4f})")
            print("-----------------------------------------------------------------")
            print("Reference-Category Sensitivity (State SD in Percentage Points):")
            print("Party  | Ref: REST (20% Shrink) | Ref: S (20% Shrink) | Ref: REST (No Shrink) | Ref: S (No Shrink)")
            print("-------+------------------------+---------------------+-----------------------+-------------------")
            s20 = report["reference_sensitivity"]["shrinkage_20"]
            s00 = report["reference_sensitivity"]["shrinkage_00"]
            for cat in ALL_CATEGORIES:
                r20 = s20[REFERENCE_CATEGORY][cat]["std_dev"]
                s_s20 = s20["S"][cat]["std_dev"]
                r00 = s00[REFERENCE_CATEGORY][cat]["std_dev"]
                s_s00 = s00["S"][cat]["std_dev"]
                print(f"{cat:<6} | {r20:>22.2f} | {s_s20:>19.2f} | {r00:>21.2f} | {s_s00:>17.2f}")
            print("=================================================================")
        return 0
    except Exception as err:
        sys.stderr.write(f"Audit error: {err}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
