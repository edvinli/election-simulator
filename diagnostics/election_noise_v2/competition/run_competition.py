"""Execute the frozen CONTROL-vs-A-vs-B competition and apply the frozen gates.

Order is deliberate and enforced: the run manifest is written to disk BEFORE any
target-election score is computed, so the configuration cannot be revised once
results are visible.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics as st
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.election_noise_v2.control_baseline.harness.rng import (
    DRAWS_PER_SEED,
    FROZEN_SEEDS,
)
from diagnostics.election_noise_v2.control_baseline_amendment2.harness2.manifest import (
    AMENDMENT2,
    build_manifest,
    validate_manifest,
)
from diagnostics.election_noise_v2.competition import gates as G
from diagnostics.election_noise_v2.competition.runner import (
    CONTROL,
    FROZEN_H,
    MODELS,
    run_all,
    run_job,
)
from diagnostics.election_noise_v2.challengers.draws import MODEL_A, MODEL_B

OUT = Path(__file__).resolve().parent
MASK_DIR = OUT / "mask_level"
A2 = REPO_ROOT / "diagnostics/election_noise_v2/control_baseline_amendment2"
CH = REPO_ROOT / "diagnostics/election_noise_v2/challengers"

SHORT = {CONTROL: "CONTROL", MODEL_A: "A", MODEL_B: "B"}

TIER1_KEYS = ["es_9cat", "es_8party", "crps_8party_mean", "crps_all9_mean",
              "coverage_50", "coverage_80", "coverage_90",
              "mean_width_50", "mean_width_80", "mean_width_90", "mean_lambda"]
ISO_KEYS = ["seat_energy_score", "seat_crps_8party_mean",
            "seat_coverage_50", "seat_coverage_80", "seat_coverage_90",
            "coalition_brier_mean_over_masks", "mean_lambda"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*a: str) -> str:
    return subprocess.check_output(["git", *a], cwd=REPO_ROOT).decode().strip()


def write_manifest(cases: dict, seeds: list[int], draws: int) -> dict:
    m = {
        "artifact": "ELECTIONNOISE V2 COMPETITION RUN MANIFEST",
        "status": "written BEFORE any target-election score was computed; immutable thereafter",
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": {"head": git("rev-parse", "HEAD"),
                "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "worktree_clean": git("status", "--porcelain") == ""},
        "authoritative_commits": {
            "evaluator_refreeze_base": "a5b8c7a234acf60cac71ef1ab1439343fae88639",
            "challenger_implementation": "0facfe6cf542e650bcdf7fdc2d01a6795366c045",
            "challenger_freeze": "1450e6f301a98d5d6e4af1357113435534b0e7a9",
        },
        "freeze_hashes": {
            "challenger_implementation_freeze": sha(CH / "challenger_implementation_freeze.json"),
            "challenger_implementation_freeze_expected":
                "2454ac15309361443656fe1d00abd5cb655d5a8efc8ddaded9e8c7164d8c1c22",
            "evaluator_freeze": sha(A2 / "evaluator_freeze.json"),
            "evaluation_case_manifest": sha(A2 / "evaluation_case_manifest.json"),
            "control_scores_summary": sha(A2 / "control_scores_summary.json"),
            "exact_control_oracle": sha(A2 / "exact_control_oracle.json"),
            "bandwidth_selection": sha(CH / "bandwidth_selection.json"),
            "apply_batch_simplex_transfer": sha(REPO_ROOT / "scripts/election_layer_v2/transfer.py"),
        },
        "preregistration": AMENDMENT2,
        "models": {"CONTROL": CONTROL, "A": MODEL_A, "B": MODEL_B},
        "targets": [2014, 2018, 2022],
        "tier1_cases": [{"target_year": c["target_year"], "election_date": c["election_date"],
                         "horizon_days": c["horizon_days"], "k_outer": c["k_outer"],
                         "training_residual_years": c["training_residual_years"]}
                        for c in cases["tier1"]],
        "tier3_iso_cases": [{"target_year": c["target_year"], "election_date": c["election_date"],
                             "mandate_law": c["mandate_law"], "first_divisor": c["first_divisor"],
                             "geography_mode": c["geography_mode"],
                             "geography_baseline_year": c["geography_baseline_year"],
                             "k_outer": c["k_outer"],
                             "training_residual_years": c["training_residual_years"]}
                            for c in cases["tier3_iso"]],
        "N_T1": 3, "N_seat": 3,
        "seeds": list(seeds), "draws_per_seed": draws,
        "challenger_a_bandwidths": {str(k): v for k, v in FROZEN_H.items()},
        "challenger_b_hyperparameters": 0,
        "mandate_law": {"2014": "PRE_2018", "2018": "POST_2018", "2022": "POST_2018"},
        "geography": {"mode": "chronological", "oracle": "forbidden"},
        "coalition_masks": {"range": "1..254", "majority_threshold": 175,
                            "effective_distinct_events": 127},
        "gate_thresholds": {
            "improves_pct": G.IMPROVE_PCT, "non_inferiority_pct": G.NONINFERIOR_PCT,
            "coverage_pp": G.COVERAGE_PP, "loo_improve_pct": G.LOO_IMPROVE_PCT,
            "brier_loo_degrade_pct": G.BRIER_LOO_DEGRADE_PCT,
        },
        "excluded_from_adoption": ("full-pipeline Tier 2 and Tier 3 remain retrospective "
                                   "diagnostics only (Amendment 2)"),
        "forecast_2026": "not run; not an adoption input",
    }
    (OUT / "competition_manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    return m


def agg(rows, model, tier, keys, years):
    sub = [r for r in rows if r["model"] == model and r["tier"] == tier]
    by_el, headline = {}, {}
    for y in years:
        cs = [r for r in sub if r["target_year"] == y]
        by_el[y] = {}
        for k in keys:
            vals = [r[k] for r in cs if k in r and r[k] is not None]
            if not vals:
                continue
            by_el[y][k] = {"mean": float(np.mean(vals)),
                           "sd": float(st.stdev(vals)) if len(vals) > 1 else 0.0,
                           "n_seeds": len(vals),
                           "values": {int(r["seed"]): r[k] for r in cs if k in r}}
    for k in keys:
        vals = [by_el[y][k]["mean"] for y in years if k in by_el[y]]
        if vals:
            headline[k] = {"mean_over_elections": float(np.mean(vals))}
    return {"by_election": by_el, "headline": headline}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--draws", type=int, default=DRAWS_PER_SEED)
    args = ap.parse_args(argv)

    base = build_manifest()
    probs = validate_manifest(base)
    if probs:
        raise SystemExit("EVALUATOR MANIFEST VALIDATION FAILED:\n" + "\n".join(probs))

    OUT.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    seeds = list(base["monte_carlo"]["seeds"])
    # ---- manifest FIRST, before any score exists ----
    write_manifest(base["cases"], seeds, args.draws)
    print("manifest pinned; scoring starts now", flush=True)

    rows, mask_rows = run_all(base, args.draws, seeds, workers=args.workers)

    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(OUT / "scores_by_model_case_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval=""); w.writeheader(); w.writerows(rows)
    with open(MASK_DIR / "coalition_probabilities.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mask_rows[0].keys())); w.writeheader()
        w.writerows(mask_rows)

    years = [2014, 2018, 2022]
    A = {SHORT[m]: {"tier1": agg(rows, m, "tier1", TIER1_KEYS, years),
                    "tier3_iso": agg(rows, m, "tier3_iso", ISO_KEYS, years)} for m in MODELS}

    el_rows = []
    for short, tiers in A.items():
        for tier, t in tiers.items():
            for y in years:
                rec = {"model": short, "tier": tier, "target_year": y}
                for k, v in t["by_election"][y].items():
                    rec[k] = v["mean"]; rec[f"{k}_sd"] = v["sd"]
                el_rows.append(rec)
    ef = []
    for r in el_rows:
        for k in r:
            if k not in ef:
                ef.append(k)
    with open(OUT / "scores_by_model_election.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ef, restval=""); w.writeheader(); w.writerows(el_rows)

    # ---------------- gates ----------------
    c1 = A["CONTROL"]["tier1"]; c3 = A["CONTROL"]["tier3_iso"]
    gate_rows, gate_detail, passes, t1es = [], {}, {}, {}

    for short in ("A", "B"):
        h1 = A[short]["tier1"]; h3 = A[short]["tier3_iso"]
        ctrl_h = {k: c1["headline"][k]["mean_over_elections"] for k in TIER1_KEYS}
        chal_h = {k: h1["headline"][k]["mean_over_elections"] for k in TIER1_KEYS}
        t1es[short] = chal_h["es_9cat"]

        rs = [G.g1_tier1_improvement(ctrl_h["es_9cat"], chal_h["es_9cat"])]
        rs += G.g3_noninferiority(ctrl_h, chal_h)
        rs.append(G.g4_seat_noninferiority(
            c3["headline"]["seat_energy_score"]["mean_over_elections"],
            h3["headline"]["seat_energy_score"]["mean_over_elections"]))
        rs.append(G.g2_coalition_improvement(
            c3["headline"]["coalition_brier_mean_over_masks"]["mean_over_elections"],
            h3["headline"]["coalition_brier_mean_over_masks"]["mean_over_elections"]))

        t1_ctrl_el = {y: c1["by_election"][y]["es_9cat"]["mean"] for y in years}
        t1_chal_el = {y: h1["by_election"][y]["es_9cat"]["mean"] for y in years}
        r5, d5 = G.g5_tier1_robustness(t1_ctrl_el, t1_chal_el)
        rs += r5

        cb_ctrl_el = {y: c3["by_election"][y]["coalition_brier_mean_over_masks"]["mean"]
                      for y in years}
        cb_chal_el = {y: h3["by_election"][y]["coalition_brier_mean_over_masks"]["mean"]
                      for y in years}
        r5b, d5b = G.g5_coalition_robustness(cb_ctrl_el, cb_chal_el)
        rs += r5b

        for r in rs:
            r["model"] = short
        gate_rows += rs
        gate_detail[short] = {"tier1_robustness": d5, "coalition_robustness": d5b,
                              "tier1_by_election": {"control": t1_ctrl_el, "challenger": t1_chal_el},
                              "coalition_by_election": {"control": cb_ctrl_el,
                                                        "challenger": cb_chal_el}}
        passes[short] = all(r["result"] == "PASS" for r in rs)

    with open(OUT / "tier1_leave_one_out.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dropped_election", "kept", "control_es_9cat",
                    "challenger_es_9cat", "relative_improvement_pct"])
        for short in ("A", "B"):
            for k, v in gate_detail[short]["tier1_robustness"]["leave_one_out"].items():
                w.writerow([short, k, "+".join(str(x) for x in v["kept"]),
                            v["control"], v["challenger"], v["relative_improvement_pct"]])

    with open(OUT / "coalition_leave_one_out.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dropped_election", "kept", "control_brier",
                    "challenger_brier", "relative_improvement_pct"])
        for short in ("A", "B"):
            for k, v in gate_detail[short]["coalition_robustness"]["leave_one_out"].items():
                w.writerow([short, k, "+".join(str(x) for x in v["kept"]),
                            v["control"], v["challenger"], v["relative_improvement_pct"]])

    with open(OUT / "coalition_brier_by_model_election.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "target_year", "coalition_brier", "five_seed_sd"])
        for short in ("CONTROL", "A", "B"):
            t = A[short]["tier3_iso"]
            for y in years:
                v = t["by_election"][y]["coalition_brier_mean_over_masks"]
                w.writerow([short, y, v["mean"], v["sd"]])
            w.writerow([short, "HEADLINE",
                        t["headline"]["coalition_brier_mean_over_masks"]["mean_over_elections"], ""])

    with open(OUT / "coverage_gate.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "level", "nominal", "coverage", "abs_deviation_pp",
                    "control_abs_deviation_pp", "increase_pp", "tolerance_pp", "result"])
        for short in ("CONTROL", "A", "B"):
            for lvl, nom in G.NOMINAL.items():
                cov = A[short]["tier1"]["headline"][f"coverage_{lvl}"]["mean_over_elections"]
                cc = A["CONTROL"]["tier1"]["headline"][f"coverage_{lvl}"]["mean_over_elections"]
                dev = abs(cov - nom) * 100.0
                cdev = abs(cc - nom) * 100.0
                inc = dev - cdev
                res = "-" if short == "CONTROL" else ("PASS" if inc <= G.COVERAGE_PP else "FAIL")
                w.writerow([short, lvl, nom, cov, dev, cdev, inc, G.COVERAGE_PP, res])

    gf = ["model", "gate", "metric", "control", "challenger", "absolute_difference",
          "relative_difference_pct", "required_threshold", "result", "artifact", "detail"]
    with open(OUT / "gate_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=gf, extrasaction="ignore")
        w.writeheader(); w.writerows(gate_rows)
    (OUT / "gate_table.json").write_text(json.dumps(
        {"rows": gate_rows, "detail": gate_detail,
         "all_gates_pass": passes}, indent=2, default=str) + "\n")

    (OUT / "scores_summary.json").write_text(json.dumps({
        "status": "frozen competition result; gates applied literally",
        "seeds": seeds, "draws_per_seed": args.draws,
        "models": {k: {"tier1": v["tier1"], "tier3_iso": v["tier3_iso"]} for k, v in A.items()},
    }, indent=2, default=str) + "\n")

    decision = G.decide(passes, t1es)
    print("\n=== GATE SUMMARY ===")
    for short in ("A", "B"):
        fails = [r["gate"] + ":" + r["metric"] for r in gate_rows
                 if r["model"] == short and r["result"] == "FAIL"]
        print(f"  {short}: {'PASS' if passes[short] else 'FAIL'}"
              + ("" if passes[short] else f"  failing -> {len(fails)}"))
    print("DECISION:", decision["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
