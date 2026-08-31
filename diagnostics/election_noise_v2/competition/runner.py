"""Run the frozen competition. Execution only - no modelling choice is made here.

Every metric is computed by the **frozen** Part-3 metric module, called through the
same code path CONTROL was certified on. The only thing that varies across models is
the ElectionNoise draw; the consensus, transfer, geography, allocator, truth,
seeds, N, case set and metric implementations are identical by construction.

CONTROL is re-run rather than read from the baseline, so its reproduction of the
certified numbers is an integrity check on this runner.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import date
import time
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.election_noise_v2.control_baseline.harness import metrics as M
from diagnostics.election_noise_v2.control_baseline.harness.pipeline import tier1_control_draws
from diagnostics.election_noise_v2.control_baseline_amendment2.harness2.exact_oracle import (
    mask_columns,
)
from diagnostics.election_noise_v2.control_baseline_amendment2.harness2.isolated import (
    control_iso_draws,
)
from diagnostics.election_noise_v2.challengers.draws import (
    MODEL_A,
    MODEL_B,
    challenger_iso_draws,
    challenger_tier1_draws,
)
from scripts.simulator.config import PARLIAMENTARY_PARTIES_8

CONTROL = "CONTROL_pp_centered_noise"
MODELS = (CONTROL, MODEL_A, MODEL_B)

#: Challenger A bandwidths, frozen in the challenger implementation freeze.
FROZEN_H = {2014: 0.75, 2018: 0.75, 2022: 0.75}


def _truth_vote(case: dict) -> np.ndarray:
    from scripts.election_residuals.config import ALL_CATEGORIES
    return np.array([case["truth_vote_pct"][c] for c in ALL_CATEGORIES], dtype=float)


def _truth_seats(case: dict) -> np.ndarray:
    return np.array([case["truth_seats"][p] for p in PARLIAMENTARY_PARTIES_8], dtype=np.int64)


def _draws(model: str, tier: str, year: int, ed: date, seed: int, n: int):
    """Model dispatch. This is the ONLY thing that differs between models."""
    if tier == "tier1":
        if model == CONTROL:
            d = tier1_control_draws(ed, year, seed, n)
            return d.votes_pct, d.lambdas, None, d.residual_index
        h = FROZEN_H[year] if model == MODEL_A else None
        d = challenger_tier1_draws(model, year, seed, n, h=h)
        return d.votes_pct, d.lambdas, None, d.atom_index
    if model == CONTROL:
        d = control_iso_draws(year, seed, n)
        return d.votes_pct, d.lambdas, d.seats, d.residual_index
    h = FROZEN_H[year] if model == MODEL_A else None
    d = challenger_iso_draws(model, year, seed, n, h=h)
    return d.votes_pct, d.lambdas, d.seats, d.atom_index


def run_job(job: dict) -> dict:
    """One (model, tier, election, seed) cell. Mirrors the frozen CONTROL job."""
    model, tier, case, seed, n = job["model"], job["tier"], job["case"], job["seed"], job["draws"]
    t0 = time.perf_counter()
    year = case["target_year"]
    ed = date.fromisoformat(case["election_date"])
    row = {
        "model": model, "tier": tier, "target_year": year,
        "horizon_days": case["horizon_days"], "seed": seed, "draws": n,
        "k_outer": case["k_outer"],
        "training_residual_years": "|".join(str(y) for y in case["training_residual_years"]),
        "h": FROZEN_H[year] if model == MODEL_A else "",
    }
    mask_rows: list[dict] = []
    votes, lambdas, seats, index = _draws(model, tier, year, ed, seed, n)

    if tier == "tier1":
        tv = _truth_vote(case)
        row.update(M.d1_joint_vote_energy_score(votes, tv))
        d2 = M.d2_marginal_vote_metrics(votes, tv)
        row.update({k: v for k, v in d2.items() if k != "per_party"})
        row.update(M.d5_lambda_diagnostics(lambdas))
        for p, r in d2["per_party"].items():
            row[f"crps_{p}"] = r["crps"]
    else:
        row["mandate_law"] = case["mandate_law"]
        row["first_divisor"] = case["first_divisor"]
        row["geography_mode"] = case["geography_mode"]
        row["geography_baseline_year"] = case["geography_baseline_year"]
        ts = _truth_seats(case)
        d3 = M.d3_seat_metrics(seats, ts)
        row.update({k: v for k, v in d3.items() if k != "per_party"})
        for p, r in d3["per_party"].items():
            row[f"seat_crps_{p}"] = r["crps"]
        d4 = M.d4_coalition_brier(seats, ts)
        row["coalition_brier_mean_over_masks"] = d4["brier_mean_over_masks"]
        sym = M.verify_complement_symmetry(d4["per_mask"])
        row["complement_symmetry_max_abs_diff"] = sym["max_abs_brier_difference_between_complements"]
        row["complement_symmetry_holds"] = sym["holds_within_tolerance"]
        row["seat_total_always_349"] = bool(np.all(seats.sum(axis=1) == 349))
        row["min_vote_pct"] = float(votes.min())
        row["max_abs_sum_deviation"] = float(np.abs(votes.sum(axis=1) - 100.0).max())
        row["any_nonfinite"] = bool(not np.all(np.isfinite(votes)))
        row["distinct_vote_rows"] = int(np.unique(np.round(votes, 12), axis=0).shape[0])
        row["distinct_seat_rows"] = int(np.unique(seats, axis=0).shape[0])
        row.update(M.d5_lambda_diagnostics(lambdas))
        row["mc_mean_seats"] = "|".join(f"{v:.6f}" for v in seats.mean(axis=0))
        row["mc_mean_vote_pct"] = "|".join(f"{v:.8f}" for v in votes.mean(axis=0))
        if index is not None:
            cnt = np.bincount(np.asarray(index), minlength=case["k_outer"])
            row["atom_index_counts"] = "|".join(str(int(x)) for x in cnt)
        for m, r in d4["per_mask"].items():
            mask_rows.append({
                "model": model, "tier": tier, "target_year": year, "seed": seed, "mask": m,
                "parties": "+".join(PARLIAMENTARY_PARTIES_8[i] for i in mask_columns(m)),
                "p_majority": r["p"], "certified_indicator": int(r["y"]),
                "certified_coalition_seats": r["certified_seats"], "brier": r["brier"],
            })

    row["min_lambda_ok"] = bool(np.all((lambdas >= 0.0) & (lambdas <= 1.0)))
    row["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
    return {"row": row, "mask_rows": mask_rows}


def run_all(manifest: dict, draws: int, seeds: list[int], workers: int = 6):
    jobs = [
        {"model": m, "tier": t, "case": c, "seed": s, "draws": draws}
        for m in MODELS
        for t in ("tier1", "tier3_iso")
        for c in manifest["cases"][t]
        for s in seeds
    ]
    print(f"{len(jobs)} jobs ({draws} draws x {len(seeds)} seeds x {len(MODELS)} models)",
          flush=True)
    out, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(run_job, jobs), 1):
            out.append(r)
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} ({time.perf_counter()-t0:.0f}s)", flush=True)
    return [r["row"] for r in out], [m for r in out for m in r["mask_rows"]]
