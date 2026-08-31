"""Run the frozen preregistered evaluation for CONTROL only, and write the outputs.

CONTROL is the unmodified production ElectionNoise (``pp_centered_noise``). No
challenger is implemented, no adoption gate is evaluated, and no 2026 forecast is
produced.

Usage::

    uv run python -m diagnostics.election_noise_v2.control_baseline.harness.run_control
    ... --workers 10 [--draws 20000] [--quick]

``--draws``/``--quick`` exist only for smoke testing; the reported baseline is
always produced at the frozen N = 20 000 x 5 seeds.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import date
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.election_noise_v2.control_baseline.harness import metrics as M
from diagnostics.election_noise_v2.control_baseline.harness.manifest import (
    build_manifest,
    validate_manifest,
)
from diagnostics.election_noise_v2.control_baseline.harness.pipeline import (
    tier1_control_draws,
    tier1_support,
    tier23_control_draws,
)
from diagnostics.election_noise_v2.control_baseline.harness.rng import DRAWS_PER_SEED
from scripts.election_residuals.config import ALL_CATEGORIES
from scripts.simulator.config import PARLIAMENTARY_PARTIES_8

OUT = Path(__file__).resolve().parents[1]
MASK_DIR = OUT / "mask_level"
MODEL_ID = "CONTROL_pp_centered_noise"


def _truth_vote_vec(case: dict) -> np.ndarray:
    return np.array([case["truth_vote_pct"][c] for c in ALL_CATEGORIES], dtype=float)


def _truth_seat_vec(case: dict) -> np.ndarray:
    return np.array([case["truth_seats"][p] for p in PARLIAMENTARY_PARTIES_8], dtype=np.int64)


def run_job(job: dict) -> dict:
    """One (case, seed) evaluation block.

    Tier 2 and Tier 3 are, by the preregistration, *the same cases*. They are
    therefore evaluated from a single simulation per (case, seed) and emitted as
    two rows, so the frozen pipeline is never run twice on identical inputs.
    """
    kind, case, seed, n = job["kind"], job["case"], job["seed"], job["draws"]
    t0 = time.perf_counter()
    ed = date.fromisoformat(case["election_date"])
    truth_v = _truth_vote_vec(case)

    def shell(tier: str) -> dict:
        return {
            "model": MODEL_ID,
            "tier": tier,
            "target_year": case["target_year"],
            "horizon_days": case["horizon_days"],
            "seed": seed,
            "draws": n,
            "k_outer": case["k_outer"],
            "training_residual_years": "|".join(str(y) for y in case["training_residual_years"]),
        }

    rows: list[dict] = []
    mask_rows: list[dict] = []

    if kind == "tier1":
        draws = tier1_control_draws(ed, case["target_year"], seed, n)
        row = shell("tier1")
        row.update(M.d1_joint_vote_energy_score(draws.votes_pct, truth_v))
        d2 = M.d2_marginal_vote_metrics(draws.votes_pct, truth_v)
        row.update({k: v for k, v in d2.items() if k != "per_party"})
        row.update(M.d5_lambda_diagnostics(draws.lambdas))
        row["atom_index_counts"] = "|".join(
            str(int(c)) for c in np.bincount(draws.residual_index, minlength=case["k_outer"])
        )
        rows.append(row)
    else:
        draws, seats = tier23_control_draws(
            as_of=date.fromisoformat(case["as_of"]),
            election_date=ed,
            seed=seed,
            n=n,
            baseline_year=case["geography_baseline_year"],
            target_year=case["target_year"],
        )
        truth_s = _truth_seat_vec(case)

        r2 = shell("tier2")
        r2["mandate_law"] = case["mandate_law"]
        r2["first_divisor"] = case["first_divisor"]
        r2.update(M.d1_joint_vote_energy_score(draws.votes_pct, truth_v))
        d2 = M.d2_marginal_vote_metrics(draws.votes_pct, truth_v)
        r2.update({k: v for k, v in d2.items() if k != "per_party"})
        r2.update(M.d5_lambda_diagnostics(draws.lambdas))
        rows.append(r2)

        r3 = shell("tier3")
        r3["mandate_law"] = case["mandate_law"]
        r3["first_divisor"] = case["first_divisor"]
        d3 = M.d3_seat_metrics(seats, truth_s)
        r3.update({k: v for k, v in d3.items() if k != "per_party"})
        d4 = M.d4_coalition_brier(seats, truth_s)
        r3["coalition_brier_mean_over_masks"] = d4["brier_mean_over_masks"]
        sym = M.verify_complement_symmetry(d4["per_mask"])
        r3["complement_symmetry_max_abs_diff"] = sym["max_abs_brier_difference_between_complements"]
        r3["complement_symmetry_holds"] = sym["holds_within_tolerance"]
        r3["seat_total_always_349"] = bool(np.all(seats.sum(axis=1) == 349))
        r3.update(M.d5_lambda_diagnostics(draws.lambdas))
        rows.append(r3)

        for m, d in d4["per_mask"].items():
            mask_rows.append(
                {
                    "tier": "tier3",
                    "target_year": case["target_year"],
                    "horizon_days": case["horizon_days"],
                    "seed": seed,
                    "mask": m,
                    "parties": "+".join(PARLIAMENTARY_PARTIES_8[i] for i in M.coalition_mask_columns(m)),
                    "p_majority": d["p"],
                    "certified_indicator": int(d["y"]),
                    "certified_coalition_seats": d["certified_seats"],
                    "brier": d["brier"],
                }
            )

    el = round(time.perf_counter() - t0, 2)
    for r in rows:
        r["elapsed_seconds"] = el
    return {"rows": rows, "mask_rows": mask_rows}


def aggregate(rows: list[dict], manifest: dict) -> dict:
    """Preregistered aggregation: seeds -> case, cases(horizons) -> election, elections -> headline."""
    import statistics as st

    def by(tier: str) -> list[dict]:
        return [r for r in rows if r["tier"] == tier]

    def seed_agg(subset: list[dict], key: str) -> dict:
        vals = [r[key] for r in subset if key in r and r[key] is not None]
        if not vals:
            return {}
        return {
            "mean": float(np.mean(vals)),
            "sd": float(st.stdev(vals)) if len(vals) > 1 else 0.0,
            "n_seeds": len(vals),
            "values": {int(r["seed"]): r[key] for r in subset if key in r},
        }

    tier1_keys = [
        "es_9cat", "es_8party", "crps_8party_mean", "crps_all9_mean",
        "coverage_50", "coverage_80", "coverage_90",
        "mean_width_50", "mean_width_80", "mean_width_90", "mean_lambda",
    ]
    tier23_vote_keys = tier1_keys
    tier3_keys = [
        "seat_energy_score", "seat_crps_8party_mean",
        "seat_coverage_50", "seat_coverage_80", "seat_coverage_90",
        "coalition_brier_mean_over_masks",
    ]

    out: dict = {"model": MODEL_ID, "tiers": {}}

    # ---- Tier 1: one case per election ----
    t1: dict = {"by_election": {}, "headline": {}}
    for year in manifest["counts"]["tier1_elections"]:
        sub = [r for r in by("tier1") if r["target_year"] == year]
        t1["by_election"][str(year)] = {k: seed_agg(sub, k) for k in tier1_keys}
    for k in tier1_keys:
        per_el = [t1["by_election"][str(y)][k]["mean"] for y in manifest["counts"]["tier1_elections"]]
        if not per_el:
            continue
        t1["headline"][k] = {
            "mean_over_elections": float(np.mean(per_el)),
            "per_election": {str(y): t1["by_election"][str(y)][k]["mean"] for y in manifest["counts"]["tier1_elections"]},
            "n_elections": len(per_el),
        }
    out["tiers"]["tier1"] = t1

    # ---- Tier 2 / Tier 3: election x horizon ----
    for tier, keys in (("tier2", tier23_vote_keys), ("tier3", tier3_keys)):
        t: dict = {"by_case": {}, "by_election": {}, "headline": {}, "short_horizon_guard": {}}
        subset = by(tier)
        years = manifest["counts"]["tier23_elections"]
        horizons = sorted({r["horizon_days"] for r in subset}, reverse=True)
        for year in years:
            for h in horizons:
                cs = [r for r in subset if r["target_year"] == year and r["horizon_days"] == h]
                t["by_case"][f"{year}_h{h}"] = {k: seed_agg(cs, k) for k in keys}
            # per election: mean over horizons of the five-seed means
            t["by_election"][str(year)] = {
                k: float(np.mean([t["by_case"][f"{year}_h{h}"][k]["mean"] for h in horizons]))
                for k in keys
            }
        for k in keys:
            per_el = [t["by_election"][str(y)][k] for y in years]
            t["headline"][k] = {
                "mean_over_elections": float(np.mean(per_el)),
                "per_election": {str(y): t["by_election"][str(y)][k] for y in years},
                "n_elections": len(per_el),
            }
        # G4b short-horizon operational guard: horizons 14 and 28 only
        sh = [r for r in subset if r["horizon_days"] in (14, 28)]
        t["short_horizon_guard"] = {
            "horizons": [28, 14],
            "n_cases": len({(r["target_year"], r["horizon_days"]) for r in sh}),
            **{k: seed_agg(sh, k) for k in keys},
        }
        out["tiers"][tier] = t

    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    # Each worker peaks at ~0.7 GB while accumulating the energy-score pairwise
    # chunks at N = 20 000, so a modest worker count is faster than a large one.
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--draws", type=int, default=DRAWS_PER_SEED)
    ap.add_argument("--quick", action="store_true", help="smoke test: 2 seeds, 1000 draws")
    args = ap.parse_args(argv)

    manifest = build_manifest()
    problems = validate_manifest(manifest)
    if problems:
        raise SystemExit("MANIFEST VALIDATION FAILED:\n" + "\n".join(f"  - {p}" for p in problems))

    OUT.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    (OUT / "evaluation_case_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )

    seeds = manifest["monte_carlo"]["seeds"]
    draws = args.draws
    if args.quick:
        seeds, draws = seeds[:2], 1000

    jobs: list[dict] = []
    for case in manifest["cases"]["tier1"]:
        for s in seeds:
            jobs.append({"kind": "tier1", "case": case, "seed": s, "draws": draws})
    # Tier 2 and Tier 3 share cases; one simulation serves both.
    for case in manifest["cases"]["tier2"]:
        for s in seeds:
            jobs.append({"kind": "tier23", "case": case, "seed": s, "draws": draws})
    print(f"{len(jobs)} jobs ({draws} draws x {len(seeds)} seeds), {args.workers} workers", flush=True)

    t0 = time.perf_counter()
    results: list[dict] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, res in enumerate(ex.map(run_job, jobs), 1):
                results.append(res)
                if i % 5 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            results.append(run_job(j))
            print(f"  {i}/{len(jobs)}")

    rows = [row for r in results for row in r["rows"]]
    mask_rows = [m for r in results for m in r["mask_rows"]]

    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(OUT / "control_scores_by_case_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    if mask_rows:
        with open(MASK_DIR / "coalition_brier_by_mask.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(mask_rows[0].keys()))
            w.writeheader()
            w.writerows(mask_rows)

    agg = aggregate(rows, manifest)
    summary = {
        "status": "CONTROL BASELINE - research only; no challenger implemented or scored",
        "model": MODEL_ID,
        "monte_carlo": {"seeds": seeds, "draws_per_seed": draws},
        "counts": manifest["counts"],
        "aggregation": agg,
    }
    (OUT / "control_scores_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # by-election CSV
    el_rows = []
    for tier, t in agg["tiers"].items():
        for year, vals in t["by_election"].items():
            rec = {"model": MODEL_ID, "tier": tier, "target_year": int(year)}
            for k, v in vals.items():
                rec[k] = v["mean"] if isinstance(v, dict) else v
            el_rows.append(rec)
    ef: list[str] = []
    for r in el_rows:
        for k in r:
            if k not in ef:
                ef.append(k)
    with open(OUT / "control_scores_by_election.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ef, restval="")
        w.writeheader()
        w.writerows(el_rows)

    # coalition Brier by election
    cb = []
    for year, vals in agg["tiers"]["tier3"]["by_election"].items():
        cb.append(
            {
                "model": MODEL_ID,
                "target_year": int(year),
                "coalition_brier_election_aggregate": vals["coalition_brier_mean_over_masks"],
                "masks_per_case": 254,
                "effective_distinct_events": 127,
                "horizons_averaged": 6,
                "n_seeds": len(seeds),
            }
        )
    cb.append(
        {
            "model": MODEL_ID,
            "target_year": "HEADLINE_mean_over_elections",
            "coalition_brier_election_aggregate": agg["tiers"]["tier3"]["headline"][
                "coalition_brier_mean_over_masks"
            ]["mean_over_elections"],
            "masks_per_case": 254,
            "effective_distinct_events": 127,
            "horizons_averaged": 6,
            "n_seeds": len(seeds),
        }
    )
    with open(OUT / "coalition_brier_by_election.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cb[0].keys()))
        w.writeheader()
        w.writerows(cb)

    # Monte Carlo stability
    stab = []
    for tier, t in agg["tiers"].items():
        container = t["by_election"] if tier == "tier1" else t["by_case"]
        for label, vals in (t["by_case"] if tier != "tier1" else t["by_election"]).items():
            for k, v in vals.items():
                if isinstance(v, dict) and "sd" in v:
                    stab.append(
                        {
                            "tier": tier,
                            "case": label,
                            "metric": k,
                            "five_seed_mean": v["mean"],
                            "five_seed_sd": v["sd"],
                            "relative_sd_pct": (100.0 * v["sd"] / v["mean"]) if v["mean"] else 0.0,
                            "n_seeds": v["n_seeds"],
                        }
                    )
    with open(OUT / "monte_carlo_stability.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stab[0].keys()))
        w.writeheader()
        w.writerows(stab)

    # lambda diagnostics
    lam_fields = [
        "tier", "target_year", "horizon_days", "seed", "mean_lambda", "min_lambda",
        "p01_lambda", "p05_lambda", "p10_lambda", "fraction_lambda_lt_1",
        "fraction_lambda_lt_0_99", "fraction_lambda_lt_0_90", "fraction_lambda_lt_0_75",
    ]
    with open(OUT / "lambda_diagnostics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=lam_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\ndone in {time.perf_counter()-t0:.0f}s -> {OUT}")
    print(json.dumps(
        {
            "tier1_es_9cat_headline": agg["tiers"]["tier1"]["headline"]["es_9cat"]["mean_over_elections"],
            "tier2_es_9cat_headline": agg["tiers"]["tier2"]["headline"]["es_9cat"]["mean_over_elections"],
            "tier3_seat_es_headline": agg["tiers"]["tier3"]["headline"]["seat_energy_score"]["mean_over_elections"],
            "tier3_coalition_brier_headline": agg["tiers"]["tier3"]["headline"]["coalition_brier_mean_over_masks"]["mean_over_elections"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
