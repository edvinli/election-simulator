"""Analysis of the instrumented 100k production run: coalition-seat mixture diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.election_residuals.config import ALL_CATEGORIES
from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals

HERE = Path(__file__).resolve().parent
RUNS = HERE / "_runs"
OUT = HERE
PLOTS = OUT / "plots"
PUBLISHED = REPO_ROOT / "files" / "election-simulator" / "versions" / "20260828T201250Z-1da59168"
MAJORITY = 175

COALITIONS = {
    "C+S+MP": ("C", "S", "MP"),
    "S+V+MP": ("S", "V", "MP"),
}
MASKS = {"C+S+MP": 84, "S+V+MP": 112}


def coalition_seats(npz, parties):
    p8 = [str(x) for x in npz["parties_8"]]
    cols = [p8.index(p) for p in parties]
    return npz["seats_matrix"][:, cols].sum(axis=1).astype(np.int64)


def describe(x: np.ndarray) -> dict:
    return {
        "n": int(x.size),
        "mean_seats": float(np.mean(x)),
        "median_seats": float(np.median(x)),
        "p05": float(np.percentile(x, 5)),
        "p10": float(np.percentile(x, 10)),
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "min_seats": int(np.min(x)),
        "max_seats": int(np.max(x)),
        "count_ge_175": int(np.sum(x >= MAJORITY)),
        "p_ge_175": float(np.mean(x >= MAJORITY)),
    }


def published_histogram(mask: int):
    g = json.loads((PUBLISHED / "groups.json").read_text())
    entry = g["coalition_builder"]["coalitions"][str(mask)]
    h = entry["seat_histogram"]
    counts = np.array(h["counts"], dtype=np.int64)
    seats = np.arange(h["min_seats"], h["min_seats"] + counts.size)
    return entry, seats, counts


def hist_from_draws(x: np.ndarray):
    lo, hi = int(x.min()), int(x.max())
    counts = np.bincount(x - lo, minlength=hi - lo + 1)
    return np.arange(lo, hi + 1), counts


def main() -> int:
    PLOTS.mkdir(parents=True, exist_ok=True)
    prod = np.load(RUNS / "production.npz", allow_pickle=False)
    pre = np.load(RUNS / "prenoise.npz", allow_pickle=False)

    n = int(prod["samples"])
    years = [int(y) for y in prod["training_years"]]
    ryear = prod["residual_year"]
    ridx = prod["residual_index"].astype(np.int64)

    report: dict = {
        "as_of": str(prod["as_of"]),
        "election_date": str(prod["election_date"]),
        "samples": n,
        "seed": int(prod["seed"]),
        "residual_pool_years": years,
        "index_seed": int(prod["index_seed"]),
        "sign_seed": int(prod["sign_seed"]),
    }

    # ------------------------------------------------------------------
    # 1. Reproduction verification against the published contract
    # ------------------------------------------------------------------
    repro = {}
    for label, parties in COALITIONS.items():
        seats = coalition_seats(prod, parties)
        entry, pub_seats, pub_counts = published_histogram(MASKS[label])
        my_seats, my_counts = hist_from_draws(seats)
        # Align on the union of supports.
        lo = min(int(pub_seats[0]), int(my_seats[0]))
        hi = max(int(pub_seats[-1]), int(my_seats[-1]))
        pub_full = np.zeros(hi - lo + 1, dtype=np.int64)
        my_full = np.zeros(hi - lo + 1, dtype=np.int64)
        pub_full[pub_seats - lo] = pub_counts
        my_full[my_seats - lo] = my_counts
        repro[label] = {
            "published_mask": MASKS[label],
            "published_parties": entry["parties"],
            "histogram_exact_match": bool(np.array_equal(pub_full, my_full)),
            "max_abs_bin_difference": int(np.max(np.abs(pub_full - my_full))),
            "published_mean": entry["mean_seats"],
            "reproduced_mean": float(np.mean(seats)),
            "published_median": entry["median_seats"],
            "reproduced_median": int(np.median(seats)),
            "published_prob_majority": entry["prob_majority"],
            "reproduced_prob_majority": float(np.mean(seats >= MAJORITY)),
            "published_count_ge_175": int(pub_counts[pub_seats >= MAJORITY].sum()),
            "reproduced_count_ge_175": int(np.sum(seats >= MAJORITY)),
        }
    # Full 256-coalition check plus per-party marginals.
    g = json.loads((PUBLISHED / "groups.json").read_text())
    cb = g["coalition_builder"]
    order = list(cb["party_order"])
    p8 = [str(x) for x in prod["parties_8"]]
    assert order == p8, (order, p8)
    all_match = True
    mismatched = []
    sm = prod["seats_matrix"].astype(np.int64)
    for key, entry in cb["coalitions"].items():
        mask = int(key)
        cols = [i for i in range(8) if mask >> i & 1]
        s = sm[:, cols].sum(axis=1) if cols else np.zeros(n, dtype=np.int64)
        h = entry["seat_histogram"]
        pc = np.array(h["counts"], dtype=np.int64)
        ps = np.arange(h["min_seats"], h["min_seats"] + pc.size)
        ms, mc = hist_from_draws(s)
        lo = min(int(ps[0]), int(ms[0]))
        hi = max(int(ps[-1]), int(ms[-1]))
        a = np.zeros(hi - lo + 1, dtype=np.int64)
        b = np.zeros(hi - lo + 1, dtype=np.int64)
        a[ps - lo] = pc
        b[ms - lo] = mc
        if not np.array_equal(a, b):
            all_match = False
            mismatched.append(mask)
    repro["all_256_coalition_histograms_exact_match"] = all_match
    repro["mismatched_masks"] = mismatched
    report["reproduction"] = repro

    # ------------------------------------------------------------------
    # 2. Conditional decomposition by residual election year
    # ------------------------------------------------------------------
    rows = []
    cond_report = {}
    for label, parties in COALITIONS.items():
        seats = coalition_seats(prod, parties)
        overall = describe(seats)
        total_maj = overall["count_ge_175"]
        rows.append(
            {
                "coalition": label,
                "residual_election_year": "ALL",
                "p_residual_year": 1.0,
                **overall,
                "share_of_coalition_majority_draws": 1.0 if total_maj else 0.0,
                "unconditional_majority_contribution": overall["p_ge_175"],
            }
        )
        per_year = {}
        for y in years:
            m = ryear == y
            d = describe(seats[m])
            share = (d["count_ge_175"] / total_maj) if total_maj else float("nan")
            rows.append(
                {
                    "coalition": label,
                    "residual_election_year": y,
                    "p_residual_year": float(m.mean()),
                    **d,
                    "share_of_coalition_majority_draws": share,
                    "unconditional_majority_contribution": float(m.mean()) * d["p_ge_175"],
                }
            )
            per_year[y] = {**d, "p_residual_year": float(m.mean()), "share_of_majority": share}
        cond_report[label] = {"overall": overall, "by_year": per_year}
    report["conditional"] = cond_report

    fields = list(rows[0].keys())
    with open(OUT / "conditional_by_residual_year.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------------------------
    # 3. Pre-ElectionNoise vs final
    # ------------------------------------------------------------------
    prenoise = {}
    for label, parties in COALITIONS.items():
        s_pre = coalition_seats(pre, parties)
        s_fin = coalition_seats(prod, parties)
        prenoise[label] = {"pre": describe(s_pre), "final": describe(s_fin)}
    report["prenoise_vs_final"] = prenoise

    # ------------------------------------------------------------------
    # 4. Residual shock table (step 6)
    # ------------------------------------------------------------------
    pool = load_chronological_pp_residuals(target_election_year=2026)
    cats = list(ALL_CATEGORIES)
    shock_rows = []
    for i, y in enumerate(pool.training_years):
        raw = pool.residuals_matrix[i]
        cen = pool.centered_residuals_matrix[i]
        row = {"residual_election_year": y}
        for p in ("C", "S", "V", "MP"):
            j = cats.index(p)
            row[f"raw_{p}_pp"] = float(raw[j])
            row[f"centered_{p}_pp"] = float(cen[j])
        for p in ("M", "L", "KD", "SD", "REST"):
            j = cats.index(p)
            row[f"centered_{p}_pp"] = float(cen[j])
        row["centered_C+S+MP_pp"] = float(sum(cen[cats.index(p)] for p in ("C", "S", "MP")))
        row["centered_S+V+MP_pp"] = float(sum(cen[cats.index(p)] for p in ("S", "V", "MP")))
        row["raw_C+S+MP_pp"] = float(sum(raw[cats.index(p)] for p in ("C", "S", "MP")))
        row["raw_S+V+MP_pp"] = float(sum(raw[cats.index(p)] for p in ("S", "V", "MP")))
        # mean applied lambda for draws that used this year
        m = ridx == i
        shock_rows.append(row)
    mb = {f"mean_bias_{p}_pp": float(pool.mean_bias_pp[cats.index(p)]) for p in cats}
    report["mean_bias_pp"] = mb
    with open(OUT / "coalition_residual_shocks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(shock_rows[0].keys()))
        w.writeheader()
        w.writerows(shock_rows)
    report["residual_shocks"] = shock_rows

    # ------------------------------------------------------------------
    # 5. Sensitivity: reweighting diagnostic + true leave-one-out runs
    # ------------------------------------------------------------------
    sens_rows = []
    sens_report = {"reweighting": {}, "leave_one_out": {}}
    for label, parties in COALITIONS.items():
        seats = coalition_seats(prod, parties)
        base = describe(seats)
        sens_rows.append(
            {
                "coalition": label,
                "diagnostic": "baseline_production",
                "removed_year": "",
                "n": base["n"],
                "median_seats": base["median_seats"],
                "p05": base["p05"],
                "p95": base["p95"],
                "mean_seats": base["mean_seats"],
                "p_ge_175": base["p_ge_175"],
                "delta_p_ge_175_pp": 0.0,
                "relative_change_pct": 0.0,
            }
        )
        for y in years:
            keep = ryear != y
            d = describe(seats[keep])
            sens_rows.append(
                {
                    "coalition": label,
                    "diagnostic": "conditional_reweighting_diagnostic",
                    "removed_year": y,
                    "n": d["n"],
                    "median_seats": d["median_seats"],
                    "p05": d["p05"],
                    "p95": d["p95"],
                    "mean_seats": d["mean_seats"],
                    "p_ge_175": d["p_ge_175"],
                    "delta_p_ge_175_pp": 100.0 * (d["p_ge_175"] - base["p_ge_175"]),
                    "relative_change_pct": 100.0 * (d["p_ge_175"] / base["p_ge_175"] - 1.0),
                }
            )
            sens_report["reweighting"].setdefault(label, {})[y] = d

        for y in years:
            f = RUNS / f"loo_{y}.npz"
            if not f.exists():
                continue
            loo = np.load(f, allow_pickle=False)
            s = coalition_seats(loo, parties)
            d = describe(s)
            sens_rows.append(
                {
                    "coalition": label,
                    "diagnostic": "leave_one_election_out_rerun",
                    "removed_year": y,
                    "n": d["n"],
                    "median_seats": d["median_seats"],
                    "p05": d["p05"],
                    "p95": d["p95"],
                    "mean_seats": d["mean_seats"],
                    "p_ge_175": d["p_ge_175"],
                    "delta_p_ge_175_pp": 100.0 * (d["p_ge_175"] - base["p_ge_175"]),
                    "relative_change_pct": 100.0 * (d["p_ge_175"] / base["p_ge_175"] - 1.0),
                }
            )
            sens_report["leave_one_out"].setdefault(label, {})[y] = d
    with open(OUT / "reweighting_sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sens_rows[0].keys()))
        w.writeheader()
        w.writerows(sens_rows)
    report["sensitivity"] = sens_report

    # ------------------------------------------------------------------
    # 6. Implementation-bug audit
    # ------------------------------------------------------------------
    votes = prod["vote_shares_pct"]
    p9 = [str(x) for x in prod["parties_9"]]
    counts = np.bincount(ridx, minlength=len(years))
    expected = n / len(years)
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    audit = {
        "party_order_matches_published_bitmask_order": order == p8,
        "coalition_column_indices": {
            label: [p8.index(p) for p in parties] for label, parties in COALITIONS.items()
        },
        "joint_vs_marginal_median_check": {
            label: {
                "joint_median": float(np.median(coalition_seats(prod, parties))),
                "sum_of_marginal_medians": float(
                    sum(np.median(prod["seats_matrix"][:, p8.index(p)]) for p in parties)
                ),
            }
            for label, parties in COALITIONS.items()
        },
        "residual_year_counts": {int(y): int(c) for y, c in zip(years, counts)},
        "residual_year_uniformity_chi2_df5": chi2,
        "residual_pool_has_duplicate_years": len(set(years)) != len(years),
        "residual_pool_has_duplicate_vectors": bool(
            np.unique(np.round(pool.residuals_matrix, 9), axis=0).shape[0] != len(years)
        ),
        "centered_residuals_column_mean_max_abs": float(
            np.max(np.abs(pool.centered_residuals_matrix.mean(axis=0)))
        ),
        "centered_residuals_row_sum_max_abs": float(
            np.max(np.abs(pool.centered_residuals_matrix.sum(axis=1)))
        ),
        "seat_total_always_349": bool(np.all(prod["seats_matrix"].sum(axis=1) == 349)),
        "vote_row_sum_max_dev_from_100": float(np.max(np.abs(votes.sum(axis=1) - 100.0))),
    }
    # Threshold proximity for every party
    thr = {}
    for i, p in enumerate(p9):
        v = votes[:, i]
        thr[p] = {
            "mean_pct": float(np.mean(v)),
            "min_pct": float(np.min(v)),
            "max_pct": float(np.max(v)),
            "p_above_4pct": float(np.mean(v >= 4.0)),
            "p_within_1pp_of_4": float(np.mean(np.abs(v - 4.0) <= 1.0)),
        }
    audit["threshold_proximity"] = thr
    # Lambda clipping behaviour: recompute lambdas is unnecessary; instead check the
    # simplex floor was ever binding by looking for shares pinned at the eps floor.
    audit["draws_with_any_party_at_or_below_0_02pct"] = int(
        np.sum(np.any(votes[:, :8] <= 0.02, axis=1))
    )
    # Simplex-transfer attenuation: recompute lambda with the production function.
    from scripts.election_layer_v2.transfer import compute_simplex_transfer_scale

    base = prod["base_comp_matrix"]
    res_used = pool.centered_residuals_matrix[ridx]
    lam = np.array(
        [compute_simplex_transfer_scale(base[i], res_used[i], eps=0.01) for i in range(n)]
    )
    lam_by_year = {}
    for j, y in enumerate(years):
        m = ryear == y
        neg = np.where(pool.centered_residuals_matrix[j] < -1e-12)[0]
        binding = {}
        b = m & (lam < 1 - 1e-12)
        if b.sum():
            ratios = (base[b][:, neg] - 0.01) / (-pool.centered_residuals_matrix[j][neg])
            w = np.argmin(ratios, axis=1)
            for u, c in zip(*np.unique(w, return_counts=True)):
                binding[cats[neg[u]]] = int(c)
        lam_by_year[int(y)] = {
            "n": int(m.sum()),
            "mean_lambda": float(lam[m].mean()),
            "min_lambda": float(lam[m].min()),
            "fraction_lambda_lt_1": float((lam[m] < 1 - 1e-12).mean()),
            "binding_donor_counts": binding,
        }
    audit["simplex_transfer_attenuation"] = {
        "mean_lambda": float(lam.mean()),
        "min_lambda": float(lam.min()),
        "fraction_lambda_lt_1": float((lam < 1 - 1e-12).mean()),
        "fraction_lambda_lt_0_99": float((lam < 0.99).mean()),
        "by_year": lam_by_year,
    }
    # Residual index / seat row alignment: the realised mean vote shift within each
    # residual-year stratum must equal that year's centered residual vector.
    align = {}
    for j, y in enumerate(years):
        m = ryear == y
        realised = (votes[m] - base[m]).mean(axis=0)
        align[int(y)] = float(np.max(np.abs(realised - pool.centered_residuals_matrix[j])))
    audit["residual_index_seat_row_alignment_max_abs_pp"] = align

    # Bimodality descriptors
    for label, parties in COALITIONS.items():
        s = coalition_seats(prod, parties)
        s_seats, s_counts = hist_from_draws(s)
        dens = s_counts / s_counts.sum()
        # local maxima with a 3-seat smoothing window
        k = np.ones(3) / 3.0
        sm_dens = np.convolve(dens, k, mode="same")
        peaks = [
            int(s_seats[i])
            for i in range(1, len(sm_dens) - 1)
            if sm_dens[i] > sm_dens[i - 1] and sm_dens[i] >= sm_dens[i + 1] and sm_dens[i] > 0.002
        ]
        audit.setdefault("smoothed_local_maxima", {})[label] = peaks
    report["bug_audit"] = audit

    # Party-level conditional seat means, useful for the write-up
    party_cond = {}
    for p in ("C", "S", "V", "MP", "M", "SD", "KD", "L"):
        col = p8.index(p)
        party_cond[p] = {
            "overall_mean_seats": float(np.mean(sm[:, col])),
            "by_year": {int(y): float(np.mean(sm[ryear == y, col])) for y in years},
            "overall_mean_vote_pct": float(np.mean(votes[:, p9.index(p)])),
            "by_year_vote_pct": {
                int(y): float(np.mean(votes[ryear == y, p9.index(p)])) for y in years
            },
        }
    report["party_conditional"] = party_cond

    # Variance decomposition: between-residual-year vs within-residual-year
    var_dec = {}
    for label, parties in COALITIONS.items():
        s = coalition_seats(prod, parties).astype(float)
        s_pre = coalition_seats(pre, parties).astype(float)
        group_means = np.array([s[ryear == y].mean() for y in years])
        group_vars = np.array([s[ryear == y].var() for y in years])
        w = np.array([float((ryear == y).mean()) for y in years])
        between = float(np.sum(w * (group_means - s.mean()) ** 2))
        within = float(np.sum(w * group_vars))
        var_dec[label] = {
            "total_variance_final": float(s.var()),
            "between_year_variance": between,
            "within_year_variance": within,
            "between_share": between / float(s.var()),
            "sd_final": float(s.std()),
            "sd_prenoise": float(s_pre.std()),
            "conditional_group_means": {int(y): float(m) for y, m in zip(years, group_means)},
            "conditional_group_sds": {int(y): float(np.sqrt(v)) for y, v in zip(years, group_vars)},
        }
    report["variance_decomposition"] = var_dec

    # Pre-noise conditional (should be independent of residual year by construction)
    pre_by_year = {}
    for label, parties in COALITIONS.items():
        s = coalition_seats(pre, parties)
        pre_by_year[label] = {int(y): describe(s[ryear == y]) for y in years}
    report["prenoise_by_year"] = pre_by_year

    # First-order vote->seat map: does simple pp arithmetic already explain the
    # branch locations, or do geography / the allocator add the separation?
    base_pre = prod["base_comp_matrix"]
    elig_parties = ["M", "C", "KD", "S", "V", "MP", "SD"]  # >=4% in essentially every draw
    E = float(np.mean((base_pre[:, :8] * (base_pre[:, :8] >= 4.0)).sum(axis=1)))
    amp = {}
    pre_sm = pre["seats_matrix"].astype(np.int64)
    for label, parties in COALITIONS.items():
        cols = [p8.index(p) for p in parties]
        c_share = float(np.mean(base_pre[:, [cats.index(p) for p in parties]].sum(axis=1)))
        pre_mean = float(pre_sm[:, cols].mean(axis=0).sum())
        per_year = {}
        for j, y in enumerate(years):
            r = pool.centered_residuals_matrix[j]
            shock = float(sum(r[cats.index(p)] for p in parties))
            dE = float(sum(r[cats.index(p)] for p in elig_parties))
            linear = 349.0 * (shock / E - c_share * dE / E**2)
            realised = float(sm[ryear == y][:, cols].sum(axis=1).mean()) - pre_mean
            per_year[int(y)] = {
                "coalition_shock_pp": shock,
                "eligible_mass_change_pp": dE,
                "first_order_predicted_seat_shift": linear,
                "realised_seat_shift": realised,
                "realised_over_predicted": realised / linear if linear else None,
            }
        amp[label] = {
            "base_coalition_vote_share_pp": c_share,
            "mean_eligible_vote_share_pp": E,
            "prenoise_mean_seats": pre_mean,
            "by_year": per_year,
        }
    report["vote_to_seat_first_order_map"] = amp

    (OUT / "diagnostic_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report["reproduction"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
