"""Post-scoring integrity audit.

This audit may invalidate a run if implementation integrity failed. It may NOT be
used to retune a challenger because its score is disappointing; it inspects
structural validity only and never touches any model definition.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.election_noise_v2.control_baseline_amendment2.harness2 import freeze as ev_freeze
from diagnostics.election_noise_v2.challengers import freeze_challengers as ch_freeze
from diagnostics.election_noise_v2.competition.runner import CONTROL, FROZEN_H, MODELS, run_job

OUT = Path(__file__).resolve().parent
A2 = REPO_ROOT / "diagnostics/election_noise_v2/control_baseline_amendment2"

SEEDS = [12345, 24680, 98765, 54321, 13579]
YEARS = [2014, 2018, 2022]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit(manifest: dict) -> dict:
    rows = list(csv.DictReader((OUT / "scores_by_model_case_seed.csv").open(newline="")))
    masks = list(csv.DictReader((OUT / "mask_level/coalition_probabilities.csv").open(newline="")))
    problems: list[str] = []
    checks: dict = {}

    # ---- case / run counts ----
    cells = {}
    for r in rows:
        cells.setdefault(r["model"], set()).add((r["tier"], int(r["target_year"]), int(r["seed"])))
    checks["run_count"] = len(rows)
    checks["models"] = sorted(cells)
    expected = {(t, y, s) for t in ("tier1", "tier3_iso") for y in YEARS for s in SEEDS}
    for m, got in cells.items():
        if got != expected:
            problems.append(f"{m}: case/seed set differs from the frozen design")
    if len(rows) != 3 * 2 * 3 * 5:
        problems.append(f"expected 90 rows, got {len(rows)}")
    checks["identical_case_sets_across_models"] = all(v == expected for v in cells.values())

    # ---- numeric validity ----
    bad_num, nonfinite = [], []
    numeric_keys = ["es_9cat", "es_8party", "crps_8party_mean", "seat_energy_score",
                    "coalition_brier_mean_over_masks", "mean_lambda"]
    for r in rows:
        for k in numeric_keys:
            v = r.get(k, "")
            if v == "":
                continue
            f = float(v)
            if not math.isfinite(f):
                nonfinite.append(f"{r['model']}/{r['tier']}/{r['target_year']}/{r['seed']}:{k}")
        if r["tier"] == "tier3_iso":
            if r["seat_total_always_349"] != "True":
                bad_num.append(f"{r['model']} {r['target_year']} seed {r['seed']}: seat total != 349")
            if float(r["min_vote_pct"]) < 0:
                bad_num.append(f"{r['model']} {r['target_year']}: negative vote share")
            if float(r["max_abs_sum_deviation"]) > 1e-9:
                bad_num.append(f"{r['model']} {r['target_year']}: composition does not sum to 100")
            if r["any_nonfinite"] != "False":
                bad_num.append(f"{r['model']} {r['target_year']}: non-finite vote value")
        if r["min_lambda_ok"] != "True":
            bad_num.append(f"{r['model']} {r['tier']} {r['target_year']}: lambda outside [0,1]")
    problems += bad_num
    if nonfinite:
        problems.append(f"non-finite metric values: {nonfinite[:5]}")
    checks["nan_or_inf"] = len(nonfinite)
    checks["seat_total_always_349"] = not any("seat total" in p for p in bad_num)
    checks["compositions_valid"] = not any("composition" in p or "negative" in p for p in bad_num)
    checks["lambda_in_unit_interval"] = not any("lambda" in p for p in bad_num)

    # ---- law dispatch / geography ----
    law = {"2014": "PRE_2018", "2018": "POST_2018", "2022": "POST_2018"}
    geo_bad, law_bad = [], []
    for r in rows:
        if r["tier"] != "tier3_iso":
            continue
        if r["geography_mode"] != "chronological":
            geo_bad.append(f"{r['model']} {r['target_year']}: {r['geography_mode']}")
        if r["mandate_law"] != law[r["target_year"]]:
            law_bad.append(f"{r['model']} {r['target_year']}: {r['mandate_law']}")
    problems += [f"geography mode violation: {g}" for g in geo_bad]
    problems += [f"law dispatch violation: {l}" for l in law_bad]
    checks["geography_chronological_only"] = not geo_bad
    checks["oracle_geography_used"] = bool(geo_bad)
    checks["law_dispatch_correct"] = not law_bad

    # ---- residual leakage ----
    leak = []
    for r in rows:
        yrs = [int(y) for y in r["training_residual_years"].split("|")]
        if any(y >= int(r["target_year"]) for y in yrs):
            leak.append(f"{r['model']} {r['target_year']}: {yrs}")
        if 2026 in yrs:
            leak.append(f"{r['model']} {r['target_year']}: 2026 in training pool")
    problems += [f"future residual leakage: {x}" for x in leak]
    checks["no_future_residual_leakage"] = not leak

    # ---- Challenger A bandwidth ----
    h_bad = []
    for r in rows:
        if r["model"].startswith("CHALLENGER_A"):
            if float(r["h"]) != FROZEN_H[int(r["target_year"])]:
                h_bad.append(f"{r['target_year']}: h={r['h']}")
        elif r["h"] not in ("", None):
            h_bad.append(f"{r['model']}: unexpected h={r['h']}")
    problems += [f"wrong bandwidth: {x}" for x in h_bad]
    checks["challenger_a_bandwidth_pinned_075"] = not h_bad

    # ---- coalition masks ----
    per_cell = {}
    for m in masks:
        per_cell.setdefault((m["model"], int(m["target_year"]), int(m["seed"])), set()).add(
            int(m["mask"]))
    mask_bad = [k for k, v in per_cell.items() if v != set(range(1, 255))]
    if mask_bad:
        problems.append(f"mask set is not 1..254 in {len(mask_bad)} cells")
    checks["masks_1_to_254_everywhere"] = not mask_bad
    checks["mask_rows"] = len(masks)
    probs_out = [m for m in masks if not (0.0 <= float(m["p_majority"]) <= 1.0)]
    if probs_out:
        problems.append(f"{len(probs_out)} coalition probabilities outside [0,1]")
    checks["coalition_probabilities_in_unit_interval"] = not probs_out
    sym_bad = [r for r in rows if r["tier"] == "tier3_iso"
               and r["complement_symmetry_holds"] != "True"]
    if sym_bad:
        problems.append(f"complement symmetry violated in {len(sym_bad)} cells")
    checks["complement_symmetry_holds"] = not sym_bad

    # ---- determinism: re-run fixed cells and require bit-identical output ----
    det = []
    base = json.loads((A2 / "evaluation_case_manifest.json").read_text())
    for model in MODELS:
        for tier in ("tier1", "tier3_iso"):
            case = [c for c in base["cases"][tier] if c["target_year"] == 2018][0]
            again = run_job({"model": model, "tier": tier, "case": case,
                             "seed": 12345, "draws": 20000})["row"]
            orig = [r for r in rows if r["model"] == model and r["tier"] == tier
                    and int(r["target_year"]) == 2018 and int(r["seed"]) == 12345][0]
            # Compare only the fields this tier actually produces. The CSV is written
            # with restval="" across the union of both tiers' columns, so a tier-1 row
            # carries empty strings for every seat field and vice versa; those are
            # absent (None) in a fresh row and would otherwise register as spurious
            # mismatches. This is a comparison fix in the audit only - no model, gate,
            # threshold or score is involved.
            keys = [k for k in again if k not in ("elapsed_seconds",) and again[k] is not None]
            mismatch = [k for k in keys if str(again[k]) != orig.get(k, "")]
            det.append({"model": model, "tier": tier, "target_year": 2018, "seed": 12345,
                        "fields_compared": len(keys), "mismatched_fields": mismatch,
                        "note": "cross-tier empty columns excluded; see audit.py",
                        "bit_identical": not mismatch})
            if mismatch:
                problems.append(f"nondeterminism: {model}/{tier} fields {mismatch[:5]}")
    checks["determinism"] = det
    checks["all_reruns_bit_identical"] = all(d["bit_identical"] for d in det)

    # ---- CONTROL reproduces its certified baseline ----
    cert = {}
    with (A2 / "control_scores_by_case_seed.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            cert[(r["tier"], int(r["target_year"]), int(r["seed"]))] = r
    ctrl_mismatch = []
    compare = ["es_9cat", "es_8party", "crps_8party_mean", "crps_all9_mean",
               "seat_energy_score", "seat_crps_8party_mean",
               "coalition_brier_mean_over_masks", "mean_lambda"]
    n_cmp = 0
    for r in rows:
        if r["model"] != CONTROL:
            continue
        ref = cert.get((r["tier"], int(r["target_year"]), int(r["seed"])))
        if ref is None:
            ctrl_mismatch.append(f"missing baseline cell {r['tier']} {r['target_year']}")
            continue
        for k in compare:
            if k in r and r[k] != "" and k in ref and ref[k] != "":
                n_cmp += 1
                if float(r[k]) != float(ref[k]):
                    ctrl_mismatch.append(
                        f"{r['tier']} {r['target_year']} seed {r['seed']} {k}: "
                        f"{r[k]} vs certified {ref[k]}")
    problems += [f"CONTROL baseline mismatch: {m}" for m in ctrl_mismatch[:10]]
    checks["control_reproduces_certified_baseline"] = not ctrl_mismatch
    checks["control_metric_values_compared"] = n_cmp

    # ---- frozen-file drift ----
    ev = ev_freeze.verify()
    ch = ch_freeze.verify()
    checks["evaluator_freeze"] = {"checks": ev["checks"], "drift": ev["drift"],
                                  "unchanged": ev["evaluator_unchanged"]}
    checks["challenger_freeze"] = {"checks": ch["checks"], "drift": ch["drift"],
                                   "unchanged": ch["challengers_unchanged"]}
    if ev["drift"]:
        problems.append("evaluator freeze drifted during scoring")
    if ch["drift"]:
        problems.append("challenger freeze drifted during scoring")

    # ---- competition code identity (this runner is new, so record it) ----
    checks["competition_code_sha256"] = {
        f.name: _sha(f) for f in sorted(OUT.glob("*.py"))
    }

    return {
        "artifact": "PART 5 SCORE AUDIT",
        "purpose": ("Structural integrity of the competition run. May invalidate the run; "
                    "may NOT be used to retune a challenger."),
        "audited_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checks": checks,
        "problems": problems,
        "run_valid": not problems,
    }
