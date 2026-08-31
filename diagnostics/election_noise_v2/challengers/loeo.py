"""LOEO-FIT - Challenger A's bandwidth selection (preregistration §C, §E.4).

``h`` is selected **separately inside each OUTER chronological training pool** ``P``,
using only information already inside that pool::

    for each h in H:
        score(h) = (1/K_outer) Σ_{j∈P}  ES( F^A(h, P\\{j}),  r_j − r̄_{P\\{j}} )
    h* = argmin_h score(h);   ties broken toward the SMALLEST h

``F^A(h, P\\{j})`` is Challenger A fitted on the inner pool ``P\\{j}``
(``K_inner = K_outer − 1``) and re-centered by the **production** centering
algorithm on that inner pool; the held-out target is the held-out residual expressed
in that same centering. ``ES`` is the §D1 energy score, the unchanged production
``compute_energy_score``, evaluated in 9-category residual space.

Leakage properties, all tested
------------------------------
* The outer pool is built by the production ``load_chronological_pp_residuals``,
  which admits only elections strictly earlier than the target, so **no outer target
  residual and no future residual can enter tuning**.
* Every quantity in the loop is a function of the inner pool and the single held-out
  residual. **No election result outside the held-out residual calculation affects
  ``h``**, and no target-election outcome is read at any point.
* No 2026 information enters: the pools are historical and chronological.

Binding sample-size rules (§E.4, Amendment 1)
---------------------------------------------
* an outer comparative target is eligible only when ``K_outer >= 3``;
* inner folds at ``K_inner = 2`` are explicitly allowed;
* ``K_inner = 1`` is prohibited and fails loudly.

Seeding
-------
``ES`` is estimated from ``DRAWS_PER_SEED`` draws for each of the five frozen seeds
under the reserved ``election_noise_v2_a_loeo`` token. Per §D0, a score estimated
across the five seeds is reported and used as the **five-seed mean**, so ``score(h)``
- and therefore ``h*`` - is a deterministic property of the training pool alone and
does not vary with whichever evaluation seed is running. Selecting a different ``h``
per seed would make the challenger five different models rather than one, which §G7
("Challenger A's ``h*`` is produced solely by LOEO-FIT inside training pools",
singular) does not permit. This reading is recorded in the implementation freeze.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
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
from scripts.vote_share_calibration.energy_score import compute_energy_score

from .challenger_a import FROZEN_H_GRID, draw_challenger_a, fit_challenger_a
from .rng import A_LOEO, challenger_rng

#: Frozen minimum outer pool size for an eligible comparative target (§E.4).
MIN_K_OUTER: int = 3
#: Frozen minimum inner pool size inside LOEO-FIT. K_inner = 1 is prohibited.
MIN_K_INNER: int = 2

#: Role indices, so the index and kernel sub-streams of one fold never collide.
_ROLE_INDEX, _ROLE_KERNEL = 0, 1


class OuterPoolTooSmall(ValueError):
    """Raised when K_outer < 3. There is no K_outer = 2 fallback."""


class InnerPoolTooSmall(ValueError):
    """Raised when a fold would run at K_inner = 1, which is prohibited."""


def production_centering(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The production centering algorithm, applied to an arbitrary pool.

    Mirrors ``load_chronological_pp_residuals`` exactly: the mean bias is the column
    mean, zero-sum cleaned when its own sum exceeds 1e-12, and the centered pool is
    the raw pool minus that bias.
    """
    r = np.asarray(raw, dtype=float)
    mean_bias = np.mean(r, axis=0)
    mb_sum = float(np.sum(mean_bias))
    if abs(mb_sum) > 1e-12:
        mean_bias = mean_bias - (mb_sum / len(mean_bias))
    return mean_bias, r - mean_bias


@dataclass(frozen=True)
class LoeoResult:
    """Outcome of LOEO-FIT on one outer pool."""

    k_outer: int
    h_grid: tuple[float, ...]
    scores: dict[float, float]                 # five-seed mean held-out ES per h
    per_seed_scores: dict[float, list[float]]  # per-seed value, all five retained
    fold_scores: dict[float, list[float]]      # per-fold five-seed mean, per h
    h_star: float
    tie_broken: bool
    training_years: tuple[int, ...] = field(default=())

    @property
    def selection_rule(self) -> str:
        return "argmin over the frozen grid; exact ties resolved toward the smallest h"


def select_smallest_on_tie(scores: dict[float, float], h_grid=FROZEN_H_GRID) -> tuple[float, bool]:
    """Frozen tie rule: on an exact tie choose the SMALLEST h.

    The grid is walked in ascending order and the incumbent is replaced only on a
    strict improvement, so an exact tie always leaves the smaller h in place.
    """
    ordered = sorted(h_grid)
    best_h = ordered[0]
    best = scores[best_h]
    for h in ordered[1:]:
        if scores[h] < best:
            best_h, best = h, scores[h]
    tied = sum(1 for h in ordered if scores[h] == best) > 1
    return float(best_h), bool(tied)


def loeo_select_bandwidth(
    raw_residuals: np.ndarray,
    origin_date: date,
    horizon_days: int,
    *,
    seeds: tuple[int, ...] = FROZEN_SEEDS,
    draws: int = DRAWS_PER_SEED,
    h_grid: tuple[float, ...] = FROZEN_H_GRID,
    training_years: tuple[int, ...] = (),
) -> LoeoResult:
    """Run LOEO-FIT on one outer training pool of RAW (uncentered) residuals."""
    raw = np.asarray(raw_residuals, dtype=float)
    k_outer = raw.shape[0]
    if k_outer < MIN_K_OUTER:
        raise OuterPoolTooSmall(
            f"K_outer = {k_outer} < {MIN_K_OUTER}: the target is excluded outright. "
            "There is no K_outer = 2 fallback (preregistration §E.2, §E.4)."
        )

    scores: dict[float, float] = {}
    per_seed: dict[float, list[float]] = {}
    folds: dict[float, list[float]] = {}

    for hi, h in enumerate(h_grid):
        fold_means: list[float] = []
        seed_totals = [0.0] * len(seeds)
        for j in range(k_outer):
            inner_raw = np.delete(raw, j, axis=0)
            if inner_raw.shape[0] < MIN_K_INNER:
                raise InnerPoolTooSmall(
                    f"fold {j} would run at K_inner = {inner_raw.shape[0]}; "
                    "K_inner = 1 is prohibited (preregistration §C, §E.4)."
                )
            mean_bias, inner_centered = production_centering(inner_raw)
            held_out = raw[j] - mean_bias          # held-out residual in the inner centering
            fit = fit_challenger_a(inner_centered, h)

            seed_es: list[float] = []
            for si, seed in enumerate(seeds):
                idx_rng = challenger_rng(seed, origin_date, horizon_days, A_LOEO,
                                         spawn_key=(hi, j, si, _ROLE_INDEX))
                ker_rng = challenger_rng(seed, origin_date, horizon_days, A_LOEO,
                                         spawn_key=(hi, j, si, _ROLE_KERNEL))
                r, _ = draw_challenger_a(fit, draws, idx_rng, ker_rng)
                es = compute_energy_score(r, held_out)
                seed_es.append(float(es))
                seed_totals[si] += float(es)
            fold_means.append(float(np.mean(seed_es)))

        folds[h] = fold_means
        per_seed[h] = [t / k_outer for t in seed_totals]
        scores[h] = float(np.mean(fold_means))

    h_star, tied = select_smallest_on_tie(scores, h_grid)
    return LoeoResult(
        k_outer=k_outer, h_grid=tuple(h_grid), scores=scores, per_seed_scores=per_seed,
        fold_scores=folds, h_star=h_star, tie_broken=tied,
        training_years=tuple(int(y) for y in training_years),
    )
