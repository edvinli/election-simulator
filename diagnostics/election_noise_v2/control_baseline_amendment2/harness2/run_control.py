"""Recertify the CONTROL baseline under Amendment 2. CONTROL only.

Runs the two gate tiers at the frozen design (5 seeds × 20 000 draws):

* **Tier 1** — vote level, reusing the Part-3 certified code path unchanged, so the
  result can be compared byte-for-byte against the Part-3 baseline.
* **Tier 3-ISO** — the Amendment-2 isolated seat/coalition path.

Then builds the exact finite-support CONTROL oracle and compares the Monte Carlo
run against it.

No challenger is implemented or scored, no adoption gate is evaluated, and no 2026
forecast is produced. The Part-3 full-pipeline outputs are read for comparison and
never written.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import date
import json
from pathlib import Path
import statistics as st
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.election_noise_v2.control_baseline.harness import metrics as M
from diagnostics.election_noise_v2.control_baseline.harness.pipeline import tier1_control_draws
from diagnostics.election_noise_v2.control_baseline.harness.rng import DRAWS_PER_SEED, FROZEN_SEEDS
from scripts.election_residuals.config import ALL_CATEGORIES
from scripts.simulator.config import PARLIAMENTARY_PARTIES_8

from .exact_oracle import MASKS, exact_oracle, mask_columns
from .isolated import TIER3_ISO_TARGETS, control_iso_draws, verify_memoisation_is_exact
from .manifest import build_manifest, validate_manifest

OUT = Path(__file__).resolve().parents[1]
MASK_DIR = OUT / "mask_level"
PART3 = REPO_ROOT / "diagnostics/election_noise_v2/control_baseline"
MODEL_ID = "CONTROL_pp_centered_noise"

TIER1_KEYS = ["es_9cat", "es_8party", "crps_8party_mean", "crps_all9_mean",
              "coverage_50", "coverage_80", "coverage_90",
              "mean_width_50", "mean_width_80", "mean_width_90", "mean_lambda"]
ISO_KEYS = ["seat_energy_score", "seat_crps_8party_mean",
            "seat_coverage_50", "seat_coverage_80", "seat_coverage_90",
            "coalition_brier_mean_over_masks", "mean_lambda"]


def _truth_vote(case: dict) -> np.ndarray:
    return np.array([case["truth_vote_pct"][c] for c in ALL_CATEGORIES], dtype=float)


def _truth_seats(case: dict) -> np.ndarray:
    return np.array([case["truth_seats"][p] for p in PARLIAMENTARY_PARTIES_8], dtype=np.int64)


def run_job(job: dict) -> dict:
    tier, case, seed, n = job["tier"], job["case"], job["seed"], job["draws"]
    t0 = time.perf_counter()
    year = case["target_year"]
    ed = date.fromisoformat(case["election_date"])
    row = {
        "model": MODEL_ID, "tier": tier, "target_year": year,
        "horizon_days": case["horizon_days"], "seed": seed, "draws": n,
        "k_outer": case["k_outer"],
        "training_residual_years": "|".join(str(y) for y in case["training_residual_years"]),
    }
    mask_rows: list[dict] = []

    if tier == "tier1":
        d = tier1_control_draws(ed, year, seed, n)
        tv = _truth_vote(case)
        row.update(M.d1_joint_vote_energy_score(d.votes_pct, tv))
        d2 = M.d2_marginal_vote_metrics(d.votes_pct, tv)
        row.update({k: v for k, v in d2.items() if k != "per_party"})
        row.update(M.d5_lambda_diagnostics(d.lambdas))
    else:
        row["mandate_law"] = case["mandate_law"]
        row["first_divisor"] = case["first_divisor"]
        row["geography_mode"] = case["geography_mode"]
        row["geography_baseline_year"] = case["geography_baseline_year"]
        d = control_iso_draws(year, seed, n)
        ts = _truth_seats(case)
        d3 = M.d3_seat_metrics(d.seats, ts)
        row.update({k: v for k, v in d3.items() if k != "per_party"})
        d4 = M.d4_coalition_brier(d.seats, ts)
        row["coalition_brier_mean_over_masks"] = d4["brier_mean_over_masks"]
        sym = M.verify_complement_symmetry(d4["per_mask"])
        row["complement_symmetry_max_abs_diff"] = sym["max_abs_brier_difference_between_complements"]
        row["complement_symmetry_holds"] = sym["holds_within_tolerance"]
        row["seat_total_always_349"] = bool(np.all(d.seats.sum(axis=1) == 349))
        row["distinct_vote_rows"] = int(np.unique(np.round(d.votes_pct, 12), axis=0).shape[0])
        row["distinct_seat_rows"] = int(np.unique(d.seats, axis=0).shape[0])
        row.update(M.d5_lambda_diagnostics(d.lambdas))
        # atom frequencies, for the MC-vs-exact comparison
        cnt = np.bincount(d.residual_index, minlength=case["k_outer"])
        row["atom_index_counts"] = "|".join(str(int(x)) for x in cnt)
        row["mc_mean_seats"] = "|".join(f"{v:.6f}" for v in d.seats.mean(axis=0))
        row["mc_mean_vote_pct"] = "|".join(f"{v:.8f}" for v in d.votes_pct.mean(axis=0))
        for m, r in d4["per_mask"].items():
            mask_rows.append({
                "tier": tier, "target_year": year, "seed": seed, "mask": m,
                "parties": "+".join(PARLIAMENTARY_PARTIES_8[i] for i in mask_columns(m)),
                "p_majority": r["p"], "certified_indicator": int(r["y"]),
                "certified_coalition_seats": r["certified_seats"], "brier": r["brier"],
            })

    row["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
    return {"row": row, "mask_rows": mask_rows}


def seed_agg(sub: list[dict], key: str) -> dict:
    vals = [r[key] for r in sub if key in r and r[key] is not None]
    if not vals:
        return {}
    return {"mean": float(np.mean(vals)),
            "sd": float(st.stdev(vals)) if len(vals) > 1 else 0.0,
            "n_seeds": len(vals),
            "values": {int(r["seed"]): r[key] for r in sub if key in r}}


def aggregate(rows: list[dict], manifest: dict) -> dict:
    out = {"model": MODEL_ID, "tiers": {}}
    for tier, keys, years in (
        ("tier1", TIER1_KEYS, manifest["counts"]["tier1_elections"]),
        ("tier3_iso", ISO_KEYS, manifest["counts"]["tier3_iso_elections"]),
    ):
        sub = [r for r in rows if r["tier"] == tier]
        t = {"by_election": {}, "headline": {}}
        for y in years:
            cs = [r for r in sub if r["target_year"] == y]
            t["by_election"][str(y)] = {k: seed_agg(cs, k) for k in keys}
        for k in keys:
            per_el = [t["by_election"][str(y)][k]["mean"] for y in years]
            t["headline"][k] = {
                "mean_over_elections": float(np.mean(per_el)),
                "per_election": {str(y): t["by_election"][str(y)][k]["mean"] for y in years},
                "n_elections": len(per_el),
            }
        t["aggregation_note"] = (
            "One case per election, so the horizon step of the frozen D4 aggregation is the "
            "identity: mask mean -> five-seed mean -> unweighted mean over elections."
        )
        out["tiers"][tier] = t
    return out


def verify_tier1_unchanged(rows: list[dict]) -> dict:
    """Tier 1 must reproduce the Part-3 baseline exactly; verified, not assumed."""
    ref = {}
    with open(PART3 / "control_scores_by_case_seed.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["tier"] == "tier1":
                ref[(int(r["target_year"]), int(r["seed"]))] = r
    checked, mismatches = 0, []
    for r in rows:
        if r["tier"] != "tier1":
            continue
        key = (r["target_year"], r["seed"])
        if key not in ref:
            mismatches.append({"case": key, "issue": "absent from the Part-3 baseline"})
            continue
        for k in TIER1_KEYS:
            a, b = float(r[k]), float(ref[key][k])
            checked += 1
            if a != b:
                mismatches.append({"case": key, "metric": k, "part3": b, "amendment2": a,
                                   "abs_diff": abs(a - b)})
    return {
        "reference": "diagnostics/election_noise_v2/control_baseline/control_scores_by_case_seed.csv",
        "cases_compared": len({(r['target_year'], r['seed']) for r in rows if r['tier'] == 'tier1'}),
        "metric_values_compared": checked,
        "mismatches": mismatches,
        "tier1_bit_identical_to_part3": len(mismatches) == 0,
    }


def compare_mc_to_exact(rows: list[dict], oracles: dict) -> dict:
    """MC vs exact finite-support oracle, per election."""
    per_mask_mc: dict[tuple[int, int], dict[int, float]] = {}
    with open(MASK_DIR / "coalition_brier_by_mask.csv", newline="") as f:
        for r in csv.DictReader(f):
            per_mask_mc.setdefault((int(r["target_year"]), int(r["seed"])), {})[
                int(r["mask"])] = float(r["p_majority"])

    out = {"note": (
        "CONTROL's law on this path is exactly K atoms, so the exact oracle is the "
        "limit of the Monte Carlo run. Residuals below are pure sampling error in the "
        "atom frequencies; the expected scale is sqrt(p(1-p)/N) ~ 0.0033 at p=1/3, N=20000."
    ), "by_election": {}}
    blockers = []
    for year, orc in oracles.items():
        y = int(year)
        k = orc["k"]
        sub = [r for r in rows if r["tier"] == "tier3_iso" and r["target_year"] == y]
        exact_p = np.array([orc["per_mask"][str(m)]["exact_probability"] for m in MASKS])

        per_seed = []
        for r in sub:
            mc_p = np.array([per_mask_mc[(y, r["seed"])][m] for m in MASKS])
            mc_seats = np.array([float(x) for x in r["mc_mean_seats"].split("|")])
            mc_votes = np.array([float(x) for x in r["mc_mean_vote_pct"].split("|")])
            ex_seats = np.array([orc["exact_mean_seats"][p] for p in PARLIAMENTARY_PARTIES_8])
            ex_votes = np.array([orc["exact_mean_vote_pct"][c] for c in ALL_CATEGORIES])
            counts = np.array([int(x) for x in r["atom_index_counts"].split("|")], dtype=float)
            per_seed.append({
                "seed": r["seed"],
                "max_abs_coalition_probability_error": float(np.max(np.abs(mc_p - exact_p))),
                "mean_abs_coalition_probability_error": float(np.mean(np.abs(mc_p - exact_p))),
                "coalition_brier_error": float(r["coalition_brier_mean_over_masks"]
                                               - orc["exact_coalition_brier_mean_over_masks"]),
                "max_abs_seat_mean_error": float(np.max(np.abs(mc_seats - ex_seats))),
                "max_abs_vote_mean_error_pp": float(np.max(np.abs(mc_votes - ex_votes))),
                "seat_energy_score_error": float(r["seat_energy_score"]
                                                 - orc["exact_seat_energy_score"]),
                "max_abs_atom_frequency_error": float(np.max(np.abs(counts / counts.sum() - 1.0 / k))),
            })
        five = {
            "coalition_brier_five_seed_mean": float(np.mean([r["coalition_brier_mean_over_masks"] for r in sub])),
            "exact_coalition_brier": orc["exact_coalition_brier_mean_over_masks"],
            "seat_es_five_seed_mean": float(np.mean([r["seat_energy_score"] for r in sub])),
            "exact_seat_es": orc["exact_seat_energy_score"],
        }
        five["coalition_brier_five_seed_error"] = five["coalition_brier_five_seed_mean"] - five["exact_coalition_brier"]
        five["seat_es_five_seed_error"] = five["seat_es_five_seed_mean"] - five["exact_seat_es"]
        five["coalition_brier_relative_error_pct"] = (
            100.0 * five["coalition_brier_five_seed_error"] / five["exact_coalition_brier"]
            if five["exact_coalition_brier"] else 0.0)
        five["seat_es_relative_error_pct"] = 100.0 * five["seat_es_five_seed_error"] / five["exact_seat_es"]

        # Sampling-error tolerance: 5 sigma on an atom frequency, N = 20000.
        n = sub[0]["draws"]
        sigma = float(np.sqrt((1.0 / k) * (1 - 1.0 / k) / n))
        tol = 5.0 * sigma
        worst_p = max(r["max_abs_coalition_probability_error"] for r in per_seed)
        consistent = worst_p <= tol
        if not consistent:
            blockers.append(f"{year}: max coalition probability error {worst_p:.5f} exceeds 5-sigma {tol:.5f}")
        out["by_election"][str(year)] = {
            "k": k,
            "atom_frequency_sigma": sigma,
            "five_sigma_tolerance": tol,
            "worst_max_abs_coalition_probability_error": worst_p,
            "consistent_with_sampling_error": consistent,
            "five_seed": five,
            "per_seed": per_seed,
            "definitional_differences": [
                "Interval coverage: quantiles of a discrete K-atom law are step functions of "
                "the empirical atom weights, so coverage is not a continuous functional of the "
                "atom probabilities and is reported rather than numerically compared.",
                "Energy score: the exact value uses the 1/K^2 dispersion normalisation, which "
                "is the limit of compute_energy_score on Monte Carlo draws. "
                "compute_discrete_energy_score normalises by K(K-1) and is deliberately unused.",
            ],
        }
    out["blockers"] = blockers
    out["all_consistent_with_sampling_error"] = not blockers
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--draws", type=int, default=DRAWS_PER_SEED)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    manifest = build_manifest()
    probs = validate_manifest(manifest)
    if probs:
        raise SystemExit("MANIFEST VALIDATION FAILED:\n" + "\n".join(f"  - {p}" for p in probs))

    OUT.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    (OUT / "evaluation_case_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    seeds, draws = manifest["monte_carlo"]["seeds"], args.draws
    if args.quick:
        seeds, draws = seeds[:2], 1000

    jobs = [{"tier": t, "case": c, "seed": s, "draws": draws}
            for t in ("tier1", "tier3_iso") for c in manifest["cases"][t] for s in seeds]
    print(f"{len(jobs)} jobs ({draws} draws x {len(seeds)} seeds), {args.workers} workers", flush=True)

    t0 = time.perf_counter()
    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, r in enumerate(ex.map(run_job, jobs), 1):
                results.append(r)
                if i % 5 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} ({time.perf_counter()-t0:.0f}s)", flush=True)
    else:
        for j in jobs:
            results.append(run_job(j))

    rows = [r["row"] for r in results]
    mask_rows = [m for r in results for m in r["mask_rows"]]

    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(OUT / "control_scores_by_case_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader(); w.writerows(rows)
    with open(MASK_DIR / "coalition_brier_by_mask.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mask_rows[0].keys()))
        w.writeheader(); w.writerows(mask_rows)

    # ---- exact oracle ----
    oracles = {str(y): exact_oracle(y) for y in TIER3_ISO_TARGETS}
    memo = {str(y): verify_memoisation_is_exact(y) for y in TIER3_ISO_TARGETS}
    (OUT / "exact_control_oracle.json").write_text(
        json.dumps({"status": "EXACT FINITE-SUPPORT CONTROL ORACLE - validation artifact, "
                              "not a replacement for the preregistered five-seed Monte Carlo baseline",
                    "memoisation_exactness_check": memo,
                    "by_election": oracles}, indent=2) + "\n")

    sup_rows = []
    for y, orc in oracles.items():
        for atom in orc["exact_vote_support"]:
            rec = {"target_year": int(y), "k": orc["k"],
                   "residual_year": atom["residual_year"], "probability": atom["probability"]}
            for c in ALL_CATEGORIES:
                rec[f"vote_{c}_pct"] = atom["vote_pct"][c]
            for p in PARLIAMENTARY_PARTIES_8:
                rec[f"seats_{p}"] = atom["seats"][p]
            rec["seat_total"] = atom["seat_total"]
            for m in (15, 84, 112, 120, 135):
                cols = mask_columns(m)
                rec[f"mask{m}_{'+'.join(PARLIAMENTARY_PARTIES_8[i] for i in cols)}_seats"] = sum(
                    atom["seats"][PARLIAMENTARY_PARTIES_8[i]] for i in cols)
            sup_rows.append(rec)
    with open(OUT / "exact_control_support.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sup_rows[0].keys()))
        w.writeheader(); w.writerows(sup_rows)

    # ---- comparisons and aggregation ----
    mc_vs_exact = compare_mc_to_exact(rows, oracles)
    (OUT / "monte_carlo_vs_exact.json").write_text(json.dumps(mc_vs_exact, indent=2) + "\n")

    t1_check = verify_tier1_unchanged(rows)
    agg = aggregate(rows, manifest)
    summary = {
        "status": "AMENDMENT-2 CONTROL BASELINE - research only; no challenger implemented or scored",
        "model": MODEL_ID,
        "preregistration": manifest["preregistration"],
        "monte_carlo": {"seeds": seeds, "draws_per_seed": draws},
        "counts": manifest["counts"],
        "tier1_unchanged_vs_part3": t1_check,
        "exact_oracle_agreement": {
            "all_consistent_with_sampling_error": mc_vs_exact["all_consistent_with_sampling_error"],
            "blockers": mc_vs_exact["blockers"],
        },
        "aggregation": agg,
        "brier_interpretation": (
            "CONTROL's coalition probabilities are structurally coarse: its law on this path "
            "has only K = 3/4/5 atoms, so p_m is confined to multiples of 1/K. A continuous "
            "challenger may therefore clear the >=2% aggregate Brier improvement threshold "
            "relatively easily. The threshold is NOT changed and no gate is added; the decision "
            "must rest on the complete frozen gate - Tier-1 primary joint vote improvement, "
            "marginal non-inferiority, seat-vector non-inferiority, and the election-level and "
            "coalition-Brier robustness conditions."
        ),
    }
    (OUT / "control_scores_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    el_rows = []
    for tier, t in agg["tiers"].items():
        for y, vals in t["by_election"].items():
            rec = {"model": MODEL_ID, "tier": tier, "target_year": int(y)}
            for k, v in vals.items():
                rec[k] = v["mean"] if isinstance(v, dict) and "mean" in v else v
            el_rows.append(rec)
    ef = []
    for r in el_rows:
        for k in r:
            if k not in ef:
                ef.append(k)
    with open(OUT / "control_scores_by_election.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ef, restval=""); w.writeheader(); w.writerows(el_rows)

    cb = []
    for y, vals in agg["tiers"]["tier3_iso"]["by_election"].items():
        cb.append({"model": MODEL_ID, "target_year": int(y),
                   "coalition_brier_election_aggregate": vals["coalition_brier_mean_over_masks"]["mean"],
                   "five_seed_sd": vals["coalition_brier_mean_over_masks"]["sd"],
                   "exact_coalition_brier": oracles[y]["exact_coalition_brier_mean_over_masks"],
                   "k_atoms": oracles[y]["k"], "masks": 254, "effective_distinct_events": 127})
    cb.append({"model": MODEL_ID, "target_year": "HEADLINE_mean_over_elections",
               "coalition_brier_election_aggregate":
                   agg["tiers"]["tier3_iso"]["headline"]["coalition_brier_mean_over_masks"]["mean_over_elections"],
               "five_seed_sd": "",
               "exact_coalition_brier": float(np.mean([o["exact_coalition_brier_mean_over_masks"] for o in oracles.values()])),
               "k_atoms": "", "masks": 254, "effective_distinct_events": 127})
    with open(OUT / "coalition_brier_by_election.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cb[0].keys())); w.writeheader(); w.writerows(cb)

    lam_fields = ["tier", "target_year", "seed", "mean_lambda", "min_lambda", "p01_lambda",
                  "p05_lambda", "p10_lambda", "fraction_lambda_lt_1", "fraction_lambda_lt_0_99",
                  "fraction_lambda_lt_0_90", "fraction_lambda_lt_0_75"]
    with open(OUT / "lambda_diagnostics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=lam_fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    print(f"\ndone in {time.perf_counter()-t0:.0f}s")
    print("tier1 bit-identical to Part 3:", t1_check["tier1_bit_identical_to_part3"])
    print("MC consistent with exact oracle:", mc_vs_exact["all_consistent_with_sampling_error"])
    print(json.dumps({
        "tier1_es_9cat_headline": agg["tiers"]["tier1"]["headline"]["es_9cat"]["mean_over_elections"],
        "tier3_iso_seat_es_headline": agg["tiers"]["tier3_iso"]["headline"]["seat_energy_score"]["mean_over_elections"],
        "tier3_iso_coalition_brier_headline": agg["tiers"]["tier3_iso"]["headline"]["coalition_brier_mean_over_masks"]["mean_over_elections"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
