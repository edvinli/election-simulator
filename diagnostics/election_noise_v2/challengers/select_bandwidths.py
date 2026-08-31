"""Run LOEO-FIT on the production chronological pools and record h* per target.

This is **model fitting, not scoring**. Everything consumed here lives strictly
inside each target's own training pool, which the production
``load_chronological_pp_residuals`` restricts to elections earlier than the target.
No target-election outcome is read, no CONTROL comparison is made, and no energy
score, CRPS or Brier against any certified result is computed. The only energy
scores computed are held-out *residual* scores inside the training pool, which is
exactly what the preregistered LOEO-FIT rule specifies.

Running it here rather than at scoring time pins ``h*`` in the implementation
freeze, so the bandwidth cannot drift between freezing and scoring. The values are
fully reproducible from the frozen code and the frozen seeds.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.election_noise_v2.control_baseline.harness.rng import (
    DRAWS_PER_SEED,
    FROZEN_SEEDS,
)
from diagnostics.election_noise_v2.control_baseline_amendment2.harness2.isolated import (
    TIER3_ISO_TARGETS,
)
from scripts.election_layer_v2.config import CANONICAL_WINDOW_DAYS
from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals

from .challenger_a import FROZEN_H_GRID
from .loeo import loeo_select_bandwidth, select_smallest_on_tie

OUT = Path(__file__).resolve().parent / "bandwidth_selection.json"


def _job(args: tuple[int, float]) -> dict:
    """One (target, h) cell: all folds x all five seeds."""
    target, h = args
    ed = TIER3_ISO_TARGETS[target]["election_date"]
    pool = load_chronological_pp_residuals(target_election_year=target)
    res = loeo_select_bandwidth(
        pool.residuals_matrix, ed, CANONICAL_WINDOW_DAYS,
        seeds=FROZEN_SEEDS, draws=DRAWS_PER_SEED, h_grid=(h,),
        training_years=tuple(int(y) for y in pool.training_years),
    )
    return {"target": target, "h": h, "score": res.scores[h],
            "per_seed": res.per_seed_scores[h], "folds": res.fold_scores[h],
            "k_outer": res.k_outer,
            "training_years": [int(y) for y in pool.training_years]}


def main(workers: int = 6) -> int:
    jobs = [(t, h) for t in sorted(TIER3_ISO_TARGETS) for h in FROZEN_H_GRID]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        cells = list(ex.map(_job, jobs))

    by_target: dict[int, dict] = {}
    for c in cells:
        by_target.setdefault(c["target"], {"cells": {}})["cells"][c["h"]] = c

    out = {
        "artifact": "CHALLENGER A LOEO-FIT BANDWIDTH SELECTION",
        "status": ("model fitting inside the training pools only - NOT a score. No target "
                   "election outcome was read; no CONTROL comparison was made."),
        "rule": ("score(h) = (1/K_outer) sum_j ES(F^A(h, P\\{j}), r_j - rbar_{P\\{j}}); "
                 "h* = argmin, exact ties resolved toward the smallest h"),
        "seed_convention": ("five-seed mean per preregistration D0, so h* is a property of "
                            "the training pool and does not vary with the evaluation seed"),
        "h_grid": list(FROZEN_H_GRID),
        "seeds": list(FROZEN_SEEDS),
        "draws_per_seed": DRAWS_PER_SEED,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "by_target": {},
    }
    for target in sorted(by_target):
        cells_t = by_target[target]["cells"]
        scores = {h: cells_t[h]["score"] for h in FROZEN_H_GRID}
        h_star, tied = select_smallest_on_tie(scores, FROZEN_H_GRID)
        any_cell = cells_t[FROZEN_H_GRID[0]]
        out["by_target"][str(target)] = {
            "k_outer": any_cell["k_outer"],
            "training_residual_years": any_cell["training_years"],
            "scores_five_seed_mean": {str(h): scores[h] for h in FROZEN_H_GRID},
            "per_seed_scores": {str(h): cells_t[h]["per_seed"] for h in FROZEN_H_GRID},
            "fold_scores": {str(h): cells_t[h]["folds"] for h in FROZEN_H_GRID},
            "h_star": h_star,
            "exact_tie_encountered": tied,
        }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    for t, v in out["by_target"].items():
        print(f"  {t}: K_outer={v['k_outer']} h*={v['h_star']} "
              f"scores={ {k: round(x, 6) for k, x in v['scores_five_seed_mean'].items()} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
