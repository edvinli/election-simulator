"""T3-rev of the frozen calibration-2030 final round: the hierarchical, size-scaled noise law.

This implements §4 of ``docs/calibration_2030_final_round_preregistration.md``
(frozen, body SHA-256 5d070b4b...), under its §1 rules. It changes no
production code. The design mirrors T3 (``t3_noise.py``): each case scores the
control ``N0`` (the interim law of calibration amendment 001) and the three
hierarchical arms ``NH0``, ``NH0.5`` and ``NH1`` on the corrected election-eve
residuals, with shared standard normals, production's simplex transfer and the
frozen metric module. The laws are in ``noise_laws.py``.

Run with ``uv run python -m diagnostics.calibration_2030.t3rev_noise``. It writes
``t3rev_results.json`` beside this file. ``--smoke`` runs N = 400 draws with the
primary seed only, writes under /tmp and skips the clean-checkout check; its
numbers have no standing.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from diagnostics.election_noise_v2.control_baseline.harness.metrics import (
    d1_joint_vote_energy_score,
    d2_marginal_vote_metrics,
)
from scripts.election_layer_v2.transfer import apply_batch_simplex_transfer
from scripts.election_residuals.config import ALL_CATEGORIES
from scripts.poll_aggregate.replay import PARTIES, THRESHOLD_PCT

from . import noise_laws
from .common import verify_final_round
from .t3_noise import corrected_inputs

OUTPUT = Path(__file__).resolve().parent / "t3rev_results.json"
SMOKE_OUTPUT = Path("/tmp/t3rev_results_smoke.json")
T3_RESULTS = Path(__file__).resolve().parent / "t3_results.json"
PREREG = "docs/calibration_2030_final_round_preregistration.md"
YEARS = (2002, 2006, 2010, 2014, 2018, 2022, 2026)
PRE_2026 = (2002, 2006, 2010, 2014, 2018, 2022)
ROLLING = (2014, 2018, 2022)
N_DRAWS = 20_000
SMOKE_DRAWS = 400
EPS_PP = 0.01
CONTROL = "N0"
ARMS = ("N0", "NH0", "NH0.5", "NH1")
NH_ARMS = ("NH0", "NH0.5", "NH1")
SIMPLICITY = {"NH0": 0, "NH0.5": 1, "NH1": 2}
SEED_SUFFIXES = ("", ":1", ":2", ":3", ":4")
GATES = ("T3r-G1", "T3r-G2", "T3r-G3", "T3r-G4")

if tuple(PARTIES) != tuple(ALL_CATEGORIES[:8]):
    raise RuntimeError("replay.py PARTIES order differs from the nine-category order")


def case_seed(year: int, design: str, suffix: str) -> int:
    return int(hashlib.sha256(f"calibration-2030:T3rev:{year}:{design}{suffix}".encode()).hexdigest()[:8], 16)


def designs() -> list[tuple[str, int, list[int]]]:
    """§4.1 cases: rolling 2014/2018/2022, leave-one-out over 2002-2022, held-out 2026."""

    out = [("rolling", t, [y for y in YEARS if y < t]) for t in ROLLING]
    out += [("loo", t, [y for y in PRE_2026 if y != t]) for t in PRE_2026]
    out += [("heldout", 2026, list(PRE_2026))]
    return out


def brier_threshold(draws: np.ndarray, truth: np.ndarray) -> float:
    """Protocol v4 §6, computed as ``scripts/poll_aggregate/replay.py`` computes it."""

    p_threshold = (draws[:, : len(PARTIES)] >= THRESHOLD_PCT).mean(axis=0)
    y_threshold = (truth[: len(PARTIES)] >= THRESHOLD_PCT).astype(float)
    return float(np.mean((p_threshold - y_threshold) ** 2))


def _pool_from_job(job: dict) -> dict:
    return {"residuals": np.array(job["pool"]["residuals"], dtype=float),
            "consensus": np.array(job["pool"]["consensus"], dtype=float),
            "unreported": np.array(job["pool"]["unreported"], dtype=bool),
            "years": list(job["pool"]["years"])}


def score_task(job: dict) -> dict:
    """One arm of one (case, seed): shared Z, the arm's law, engine transfer, frozen metrics."""

    started = time.perf_counter()
    law = noise_laws.fit_law(job["arm"], _pool_from_job(job))
    consensus, truth = np.array(job["consensus"], dtype=float), np.array(job["truth"], dtype=float)
    z = np.random.default_rng(job["seed"]).standard_normal((job["draws"], len(ALL_CATEGORIES)))
    # PREREG-AMBIGUITY: §4.3's unreported-category rule is stated "for each pool election e". Reading:
    # it applies to pool rows only. As a target (the 2002 leave-one-out fold), SD's 0.00 pp consensus
    # is only floored at 1.0 pp for c_T^γ; the transfer base is the unfloored consensus (§4.1).
    base = np.tile(consensus, (job["draws"], 1))
    residuals = law.draw(z, base)
    draws, lambdas = apply_batch_simplex_transfer(base, residuals, eps=EPS_PP)
    d1 = d1_joint_vote_energy_score(draws, truth)
    d2 = d2_marginal_vote_metrics(draws, truth)
    return {
        "design": job["design"], "target": job["target"], "suffix": job["suffix"], "seed": job["seed"],
        "arm": job["arm"],
        "metrics": {
            "es_9cat": d1["es_9cat"], "es_8party": d1["es_8party"],
            "crps_8party_mean": d2["crps_8party_mean"], "crps_all9_mean": d2["crps_all9_mean"],
            "coverage_50": d2["coverage_50"], "coverage_90": d2["coverage_90"],
            "mean_width_90": d2["mean_width_90"], "brier_threshold": brier_threshold(draws, truth),
            "mean_lambda": float(np.mean(lambdas)),
        },
        "seconds": time.perf_counter() - started,
    }


def build_jobs(pools: dict, consensus: dict, truth: dict, draws: int, suffixes) -> list[dict]:
    out = []
    for design, target, pool_years in designs():
        pool = pools[(design, target)]
        for suffix in suffixes:
            for arm in ARMS:
                out.append({
                    "design": design, "target": target, "suffix": suffix, "arm": arm, "draws": draws,
                    "seed": case_seed(target, design, suffix),
                    "pool": {"residuals": pool["residuals"].tolist(), "consensus": pool["consensus"].tolist(),
                             "unreported": pool["unreported"].tolist(), "years": pool["years"]},
                    "consensus": consensus[target].tolist(), "truth": truth[target].tolist(),
                })
    return out


def evaluate(results: list[dict], suffixes) -> dict:
    """Gates T3r-G1..G4 per seed, the Monte Carlo fragility rule and §4.7 selection."""

    index = {(r["design"], r["target"], r["suffix"], r["arm"]): r["metrics"] for r in results}

    def m(design, target, suffix, arm, key="es_9cat"):
        return index[(design, target, suffix, arm)][key]

    per_seed = {}
    for suffix in suffixes:
        rows = {}
        for arm in ARMS:
            roll = {t: m("rolling", t, suffix, arm) for t in ROLLING}
            # PREREG-AMBIGUITY: §4.6 G4 says "pooled rolling Brier_threshold" without a pooling rule.
            # Reading: the unweighted mean over the three rolling cases (parent §4.3, final round §6);
            # each case averages eight parties, so this equals the mean over all 24 party terms.
            brier = float(np.mean([m("rolling", t, suffix, arm, "brier_threshold") for t in ROLLING]))
            rows[arm] = {
                "rolling_es": {str(t): v for t, v in roll.items()},
                "rolling_mean_es": float(np.mean(list(roll.values()))),
                "loo_mean_es": float(np.mean([m("loo", t, suffix, arm) for t in PRE_2026])),
                "rolling_pooled_brier_threshold": brier,
                "heldout_2026": dict(index[("heldout", 2026, suffix, arm)]),
            }
        c = rows[CONTROL]
        for arm in NH_ARMS:
            r = rows[arm]
            r["T3r-G1"] = r["rolling_mean_es"] < c["rolling_mean_es"]
            r["T3r-G2"] = all(r["rolling_es"][str(t)] <= 1.10 * c["rolling_es"][str(t)] for t in ROLLING)
            r["T3r-G3"] = r["loo_mean_es"] <= 1.02 * c["loo_mean_es"]
            r["T3r-G4"] = r["rolling_pooled_brier_threshold"] <= c["rolling_pooled_brier_threshold"] + 0.005
        per_seed[suffix or "primary"] = rows

    gates = {}
    for arm in NH_ARMS:
        gates[arm] = {}
        for g in GATES:
            outcomes = [per_seed[s][arm][g] for s in per_seed]
            gates[arm][g] = "pass" if all(outcomes) else ("fail" if not any(outcomes) else "fail (Monte-Carlo-fragile)")
        gates[arm]["passes"] = all(gates[arm][g] == "pass" for g in GATES)

    primary = per_seed["primary"]
    passing = [arm for arm in NH_ARMS if gates[arm]["passes"]]
    if passing:
        # PREREG-AMBIGUITY: §4.7 does not say which seed's rolling mean ES ranks the passing arms.
        # Reading (as T3): the primary seed; the four extra seeds serve only the fragility rule.
        best = min(passing, key=lambda arm: primary[arm]["rolling_mean_es"])
        best_es = primary[best]["rolling_mean_es"]
        near = [arm for arm in passing if primary[arm]["rolling_mean_es"] <= 1.01 * best_es]
        selected = min(near, key=lambda arm: SIMPLICITY[arm])
    else:
        selected = CONTROL
    return {
        "per_seed": per_seed, "gates": gates, "passing_arms": passing, "selected": selected,
        "selected_is_control": selected == CONTROL,
        "selection_reading": "the control N0 is not an arm that 'passes'; rule 2's simpler arm ranges over passing NH arms; ranking uses the primary seed",
        "seeds_evaluated": [s or "primary" for s in suffixes],
    }


def check_against_t3(pool: dict) -> dict:
    """§4.1: the corrected residuals and consensus, recomputed, must equal ``t3_results.json``."""

    t3 = json.loads(T3_RESULTS.read_text(encoding="utf-8"))
    for i, year in enumerate(pool["years"]):
        if not (np.array_equal(pool["residuals"][i], np.array(t3["corrected_residuals_pp"][str(year)], dtype=float))
                and np.array_equal(pool["consensus"][i], np.array(t3["corrected_consensus_pct"][str(year)], dtype=float))):
            raise RuntimeError(f"corrected residuals or consensus for {year} differ from t3_results.json")
    return {"t3_results_sha256": hashlib.sha256(T3_RESULTS.read_bytes()).hexdigest(), "equal": True, "exact": True}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="N = 400, primary seed only, output under /tmp")
    args = parser.parse_args(argv)
    provenance = verify_final_round(allow_dirty=args.smoke)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    wall = time.perf_counter()
    draws = SMOKE_DRAWS if args.smoke else N_DRAWS
    suffixes = ("",) if args.smoke else SEED_SUFFIXES
    out_path = args.out or (SMOKE_OUTPUT if args.smoke else OUTPUT)

    full = noise_laws.corrected_pool(YEARS)
    t3_check = check_against_t3(full)
    consensus, truth, residual = corrected_inputs()
    for i, year in enumerate(YEARS):
        if not (np.array_equal(full["residuals"][i], residual[year]) and np.array_equal(full["consensus"][i], consensus[year])):
            raise RuntimeError(f"noise_laws.corrected_pool differs from t3_noise.corrected_inputs for {year}")
    pools = {(design, target): noise_laws.corrected_pool(pool_years) for design, target, pool_years in designs()}

    laws = {}
    for (design, target), pool in pools.items():
        laws[f"{design}:{target}"] = {"pool_years": pool["years"],
                                      "laws": {arm: noise_laws.fit_law(arm, pool).summary() for arm in ARMS}}

    work = build_jobs(pools, consensus, truth, draws, suffixes)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(score_task, work, chunksize=1))
    decision = evaluate(results, suffixes)
    final_pool = noise_laws.corrected_pool(YEARS)
    final = noise_laws.fit_law(decision["selected"], final_pool).summary()
    record = {
        "tier": "T3-rev", "preregistration": PREREG, "smoke": bool(args.smoke),
        "started_utc": started, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": time.perf_counter() - wall, "task_seconds_total": float(sum(r["seconds"] for r in results)),
        "workers": args.workers, "provenance": provenance, "categories": list(ALL_CATEGORIES), "draws": draws,
        "eps_pp": EPS_PP, "seed_suffixes": list(suffixes),
        "seeds": {f"{d}:{t}{s}": case_seed(t, d, s) for d, t, _ in designs() for s in suffixes},
        "corrected_residuals_pp": {str(y): full["residuals"][i].tolist() for i, y in enumerate(YEARS)},
        "corrected_consensus_pct": {str(y): full["consensus"][i].tolist() for i, y in enumerate(YEARS)},
        "unreported_cells": [[int(YEARS[i]), ALL_CATEGORIES[j]] for i, j in zip(*np.nonzero(full["unreported"]))],
        "t3_recomputation_check": t3_check,
        "case_laws": laws, "cases": results, "decision": decision,
        "final_refit_2002_2026": {"law": decision["selected"], "pool_years": list(YEARS), "summary": final},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_path), "smoke": bool(args.smoke), "wall_seconds": record["wall_seconds"],
                      "task_seconds_total": record["task_seconds_total"],
                      "kappa_beta": {case: {arm: [v["laws"][arm]["kappa_hat"], v["laws"][arm]["beta"]]
                                            for arm in NH_ARMS} for case, v in laws.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
