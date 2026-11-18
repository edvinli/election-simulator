"""T3 of the frozen calibration-2030 preregistration: the election-noise law.

This implements §3 of ``docs/calibration_2030_preregistration.md`` exactly.
It changes no production code. Each case scores six arms
(bias ``β`` ∈ {0, 0.5, 1} × {Gaussian, Student-t with ν = 5 and matched
covariance}) against the corrected residuals of amendments 001 and 002.

Run with ``uv run python -m diagnostics.calibration_2030.t3_noise``. It writes
``t3_results.json`` beside this file.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from diagnostics.election_noise_v2.control_baseline.harness.metrics import (
    d1_joint_vote_energy_score,
    d2_marginal_vote_metrics,
)
from scripts.election_layer_v2.transfer import apply_batch_simplex_transfer
from scripts.election_residuals.config import ALL_CATEGORIES, DEFAULT_ELECTIONS_FILE, DEFAULT_POLLS_FILE
from scripts.election_residuals.eve_consensus import build_eve_polling_consensus
from scripts.elections.load import load_election_targets_for_forecasting
from scripts.vote_share_calibration.election_noise_b import fit_election_noise_b, symmetric_factor

from .common import ROOT, verify_and_describe

OUTPUT = Path(__file__).resolve().parent / "t3_results.json"
MANIFEST_2026 = ROOT / "data" / "raw" / "elections" / "val2026" / "official_result_manifest.json"
YEARS = (2002, 2006, 2010, 2014, 2018, 2022, 2026)
ELECTION_DAYS = {2002: date(2002, 9, 15), 2006: date(2006, 9, 17), 2010: date(2010, 9, 19), 2014: date(2014, 9, 14),
                 2018: date(2018, 9, 9), 2022: date(2022, 9, 11), 2026: date(2026, 9, 13)}
N_DRAWS = 20_000
NU = 5
EPS_PP = 0.01
BETAS = (0.0, 0.5, 1.0)
TAILS = ("gaussian", "t5")
ARMS = tuple((tail, beta) for tail in TAILS for beta in BETAS)
CONTROL = ("gaussian", 0.0)
COMPLEXITY = {("gaussian", 0.0): 0, ("gaussian", 0.5): 1, ("gaussian", 1.0): 1, ("t5", 0.0): 1, ("t5", 0.5): 2, ("t5", 1.0): 2}
SEED_SUFFIXES = ("", ":1", ":2", ":3", ":4")
PRE_2026 = (2002, 2006, 2010, 2014, 2018, 2022)


def arm_id(arm: tuple[str, float]) -> str:
    return f"{arm[0]}_beta{arm[1]:g}"


def zero_sum_clean(v: np.ndarray) -> np.ndarray:
    s = float(np.sum(v))
    if abs(s) > 0.05:
        raise ValueError(f"residual sum {s:.6f} deviates from zero")
    return v - s / len(v) if abs(s) > 1e-12 else v


def corrected_inputs() -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Corrected consensus, truth and residual (all nine categories, pp) per election."""

    polls = pd.read_csv(DEFAULT_POLLS_FILE)
    targets = load_election_targets_for_forecasting(DEFAULT_ELECTIONS_FILE)
    manifest = json.loads(MANIFEST_2026.read_text(encoding="utf-8"))
    consensus, truth, residual = {}, {}, {}
    for year in YEARS:
        e = ELECTION_DAYS[year]
        if year == 2026:
            t = {p: float(v["vote_share_percentage_points"]) for p, v in manifest["parties"].items()}
            t["REST"] = 100.0 - sum(t.values())
        else:
            t = targets[e]
        c = build_eve_polling_consensus(e, polls, window_days=14).consensus_composition
        consensus[year] = np.array([c[k] for k in ALL_CATEGORIES], dtype=float)
        truth[year] = np.array([t[k] for k in ALL_CATEGORIES], dtype=float)
        residual[year] = zero_sum_clean(truth[year] - consensus[year])
    return consensus, truth, residual


def pool_law(pool: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Zero-sum pool mean bias and the symmetric factor of the Ledoit-Wolf covariance."""

    mean_bias = np.mean(pool, axis=0)
    s = float(np.sum(mean_bias))
    if abs(s) > 1e-12:
        mean_bias = mean_bias - s / len(mean_bias)
    fit = fit_election_noise_b(pool - mean_bias)
    factor = symmetric_factor(fit.sigma_tilde)
    return mean_bias, factor, {"k": int(fit.k), "delta": float(fit.delta), "tau_sq": float(fit.tau_sq),
                               "sd_pp": [float(x) for x in np.sqrt(np.diag(fit.sigma_tilde))]}


def case_seed(year: int, design: str, suffix: str) -> int:
    return int(hashlib.sha256(f"calibration-2030:T3:{year}:{design}{suffix}".encode()).hexdigest()[:8], 16)


def score_case(job: dict) -> dict:
    """All six arms of one (case, seed): shared Z and V, engine transfer, frozen metrics."""

    consensus, truth, pool = np.array(job["consensus"]), np.array(job["truth"]), np.array(job["pool"])
    mean_bias, factor, law = pool_law(pool)
    rng = np.random.default_rng(job["seed"])
    z = rng.standard_normal((N_DRAWS, len(ALL_CATEGORIES)))
    v = rng.chisquare(NU, N_DRAWS)
    gaussian = z @ factor.T
    student = math.sqrt((NU - 2) / NU) * gaussian / np.sqrt(v / NU)[:, None]
    base = np.tile(consensus, (N_DRAWS, 1))
    arms = {}
    for tail, beta in ARMS:
        eps = gaussian if tail == "gaussian" else student
        draws, lambdas = apply_batch_simplex_transfer(base, beta * mean_bias + eps, eps=EPS_PP)
        d1 = d1_joint_vote_energy_score(draws, truth)
        d2 = d2_marginal_vote_metrics(draws, truth)
        arms[arm_id((tail, beta))] = {
            "es_9cat": d1["es_9cat"], "es_8party": d1["es_8party"],
            "crps_8party_mean": d2["crps_8party_mean"], "crps_all9_mean": d2["crps_all9_mean"],
            "coverage_50": d2["coverage_50"], "coverage_90": d2["coverage_90"],
            "mean_width_90": d2["mean_width_90"], "mean_lambda": float(np.mean(lambdas)),
        }
    return {"design": job["design"], "target": job["target"], "suffix": job["suffix"], "seed": job["seed"],
            "pool_years": job["pool_years"], "law": law, "mean_bias_pp": [float(x) for x in mean_bias], "arms": arms}


def jobs(consensus, truth, residual) -> list[dict]:
    designs = [("rolling", t, [y for y in YEARS if y < t]) for t in (2014, 2018, 2022)]
    designs += [("loo", t, [y for y in PRE_2026 if y != t]) for t in PRE_2026]
    designs += [("heldout", 2026, list(PRE_2026))]
    out = []
    for design, target, pool_years in designs:
        for suffix in SEED_SUFFIXES:
            out.append({
                "design": design, "target": target, "suffix": suffix, "seed": case_seed(target, design, suffix),
                "pool_years": pool_years, "pool": [residual[y].tolist() for y in pool_years],
                "consensus": consensus[target].tolist(), "truth": truth[target].tolist(),
            })
    return out


def evaluate(results: list[dict]) -> dict:
    """Gates T3-G1..G3 per seed, the Monte Carlo fragility rule, selection and the 2026 flag."""

    def es(design, target, suffix, arm):
        (r,) = [x for x in results if x["design"] == design and x["target"] == target and x["suffix"] == suffix]
        return r["arms"][arm]["es_9cat"]

    ctrl = arm_id(CONTROL)
    rolling, loo = (2014, 2018, 2022), PRE_2026
    per_seed = {}
    for suffix in SEED_SUFFIXES:
        rows = {}
        for arm in ARMS:
            a = arm_id(arm)
            roll = {t: es("rolling", t, suffix, a) for t in rolling}
            roll_c = {t: es("rolling", t, suffix, ctrl) for t in rolling}
            loo_mean = float(np.mean([es("loo", t, suffix, a) for t in loo]))
            loo_c = float(np.mean([es("loo", t, suffix, ctrl) for t in loo]))
            rows[a] = {
                "rolling_es": roll, "rolling_mean_es": float(np.mean(list(roll.values()))),
                "loo_mean_es": loo_mean, "heldout_2026_es": es("heldout", 2026, suffix, a),
                "T3-G1": float(np.mean(list(roll.values()))) < float(np.mean(list(roll_c.values()))),
                "T3-G2": all(roll[t] <= 1.10 * roll_c[t] for t in rolling),
                "T3-G3": loo_mean <= 1.02 * loo_c,
            }
        per_seed[suffix or "primary"] = rows

    gates = {}
    for arm in ARMS:
        a = arm_id(arm)
        if arm == CONTROL:
            continue
        gates[a] = {}
        for g in ("T3-G1", "T3-G2", "T3-G3"):
            outcomes = [per_seed[s][a][g] for s in per_seed]
            gates[a][g] = "pass" if all(outcomes) else ("fail" if not any(outcomes) else "fail (Monte-Carlo-fragile)")
        gates[a]["passes"] = all(v == "pass" for k, v in gates[a].items() if k.startswith("T3-"))

    primary = per_seed["primary"]
    passing = [arm for arm in ARMS if arm != CONTROL and gates[arm_id(arm)]["passes"]]
    if passing:
        best = min(passing, key=lambda arm: primary[arm_id(arm)]["rolling_mean_es"])
        best_es = primary[arm_id(best)]["rolling_mean_es"]
        near = [arm for arm in passing if primary[arm_id(arm)]["rolling_mean_es"] <= 1.01 * best_es]
        selected = min(near, key=lambda arm: (COMPLEXITY[arm], arm[1], TAILS.index(arm[0])))
    else:
        selected = CONTROL
    sel, ctrl_2026 = arm_id(selected), primary[ctrl]["heldout_2026_es"]
    return {
        "per_seed": per_seed, "gates": gates, "passing_arms": [arm_id(a) for a in passing],
        "selected": sel, "selected_is_control": selected == CONTROL,
        "selection_reading": "the control is not an arm that 'passes', so rule 2's 'less complex arm' ranges over passing non-control arms",
        "flag_2026": {"selected_es": primary[sel]["heldout_2026_es"], "control_es": ctrl_2026,
                      "ratio": primary[sel]["heldout_2026_es"] / ctrl_2026,
                      "flagged": primary[sel]["heldout_2026_es"] > 1.15 * ctrl_2026},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    provenance = verify_and_describe()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    consensus, truth, residual = corrected_inputs()
    work = jobs(consensus, truth, residual)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(score_case, work))
    selected_arm = evaluate(results)
    sel_tail, sel_beta = next(a for a in ARMS if arm_id(a) == selected_arm["selected"])
    final_bias, _, final_law = pool_law(np.array([residual[y] for y in YEARS]))
    record = {
        "tier": "T3", "preregistration": "docs/calibration_2030_preregistration.md",
        "started_utc": started, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": provenance, "categories": list(ALL_CATEGORIES), "draws": N_DRAWS,
        "corrected_residuals_pp": {str(y): residual[y].tolist() for y in YEARS},
        "corrected_consensus_pct": {str(y): consensus[y].tolist() for y in YEARS},
        "cases": results, "decision": selected_arm,
        "final_refit_2002_2026": {"specification": {"tails": sel_tail, "beta": sel_beta}, "law": final_law,
                                  "mean_bias_pp": final_bias.tolist(), "mean_applied_pp": (sel_beta * final_bias).tolist()},
    }
    OUTPUT.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: selected_arm[k] for k in ("gates", "selected", "flag_2026")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
