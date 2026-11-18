"""T2 of the frozen final-round preregistration: state uncertainty and dynamics (poll-prediction test).

Implements §3 of ``docs/calibration_2030_final_round_preregistration.md``
(frozen, body SHA-256 5d070b4b...). The arms live in ``t2_arms``; this
harness covers the origin grid and first scored origin (§3.1), targets
(§3.2), seeds and draw order (§3.6), metrics (§3.7), gates (§3.8), selection
(§3.9) and the reported-only items (§3.10).

Run the official scoring from a clean, committed tree::

    uv run python -m diagnostics.calibration_2030.t2_dynamics [--workers N] [--out PATH]

``--smoke`` scores only a handful of origins and horizons with N = 200,
writes to /tmp and skips the guard's clean-checkout check.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from scripts.pollofpolls.backtest_metrics import calculate_crps, calculate_interval_metrics
from scripts.poll_aggregate.config import PARTIES

from . import t2_arms as A
from .common import verify_final_round
from .v03 import AGGREGATE_VERSION, self_check

OUTPUT = Path(__file__).resolve().parent / "t2_results.json"
SMOKE_OUTPUT = Path("/tmp/calibration_2030_t2_smoke.json")
DECISION_BEFORE = date(2026, 9, 13)
GROUPS = {"short": (7, 28), "medium": (91, 182, 365), "long": (730, 1095, 1460)}
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
BOOTSTRAP_RESAMPLES = 10_000
SIMPLICITY = ("Ua", "Ub1", "Ub2", "Uc")
SMOKE_N = 200
SMOKE_HORIZONS = (7, 91, 730)
SMOKE_ORIGIN_INDICES = (0, 100, 200, 300, 380)

_CTX: dict | None = None


def _pool(workers: int) -> concurrent.futures.ProcessPoolExecutor:
    # Workers are forked after _CTX holds everything computed so far.
    return concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork"))


def _map(workers: int, fn, items: list) -> list:
    if workers <= 1:
        return [fn(x) for x in items]
    with _pool(workers) as pool:
        return list(pool.map(fn, items, chunksize=1))


def bootstrap_seed() -> int:
    # PREREG-AMBIGUITY: §3.6 says the block bootstrap seed is "derived by the
    # same scheme" without naming the token. Reading implemented: one seed,
    # int(sha256("calibration-2030:T2:bootstrap")[:8], 16), whose single
    # generator draws the cycle resamples shared by every U arm.
    return int(hashlib.sha256(b"calibration-2030:T2:bootstrap").hexdigest()[:8], 16)


def target_exists(ctx: dict, origin: date, h: int) -> bool:
    d = origin + timedelta(days=h)
    return d + timedelta(days=A.WINDOW_HALF) <= A.POLL_CUTOFF and A.poll_target(ctx, d) is not None


# --------------------------------------------------------------------------- phase A: per-origin definability

def _phase_a(origin: date) -> tuple:
    ctx = _CTX
    try:
        state = A.d0_opinion_state(ctx, origin)
    except ValueError as exc:
        state = exc
    return origin, state, A.omega_at(ctx, origin), A.theta_details(ctx, origin)


# --------------------------------------------------------------------------- phase B: k = 1 predictions

def _phase_b(origin: date) -> list:
    ctx = _CTX
    out = []
    for h in A.K_HORIZONS:
        if target_exists(ctx, origin, h):
            out.append(((origin, h, ctx["n_draws"]), A.k1_cell(ctx, origin, h)))
    return out


# --------------------------------------------------------------------------- phase C: scored cells

def score_draws(pred: np.ndarray, y: np.ndarray) -> dict:
    crps, cov50, cov90 = [], [], []
    for p in range(len(PARTIES)):
        column = pred[:, p]
        crps.append(float(calculate_crps(column, float(y[p]))))
        q = np.quantile(column, QUANTILES, method="linear")
        iv = calculate_interval_metrics(dict(zip(QUANTILES, (float(v) for v in q))), float(y[p]))
        cov50.append(int(iv["interval50_contains_actual"]))
        cov90.append(int(iv["interval90_contains_actual"]))
    return {"crps": crps, "crps_mean": float(np.mean(crps)), "cov50": cov50, "cov90": cov90,
            "mean": pred.mean(axis=0).tolist(), "sd": pred.std(axis=0).tolist()}


def score_cell(ctx: dict, origin: date, h: int) -> dict:
    target = origin + timedelta(days=h)
    y = A.poll_target(ctx, target)
    n = ctx["n_draws"]
    z_state, z_dyn, z_poll, rng = A.cell_normals(origin, h, n)
    poll = A.poll_term(ctx, origin, z_poll)
    seed = A.t2_seed(origin, h)
    arms, info = {}, {}
    for arm in A.ARMS:
        if arm == "D0":
            draws = A.state_dynamics_draws(arm, ctx, origin, target, z_state=z_state, z_dyn=z_dyn,
                                           boot_rng=None, d0_base_seed=seed)
            info["D0_eval_horizon"] = A.d0_eligible(ctx, origin, h)[1]
        else:
            k = A.inflation_k(arm, ctx, origin)[0]
            draws = A.state_dynamics_draws(arm, ctx, origin, target, z_state=z_state, z_dyn=z_dyn,
                                           boot_rng=rng if arm == "Uc" else None, d0_base_seed=None, k=k)
            info[f"{arm}_k"] = k
        arms[arm] = score_draws(draws[:, :len(PARTIES)] + poll, y)
    info["Uc_horizon_used"] = A.uc_pool(ctx, origin, h)[1]
    return {"origin": origin.isoformat(), "h": h, "target_date": target.isoformat(), "seed": seed,
            "decision": target < DECISION_BEFORE, "cycle": cycle_of(ctx, origin), "target": y.tolist(),
            "info": info, "arms": arms}


def _phase_c(task: tuple) -> list:
    origin, horizons = task
    return [score_cell(_CTX, origin, h) for h in horizons]


# --------------------------------------------------------------------------- aggregation, gates, selection

def cycle_of(ctx: dict, origin: date) -> int:
    # PREREG-AMBIGUITY: §3.10 bootstraps "by election cycle" without saying
    # whether a cell belongs to its origin's or its target's cycle. Reading
    # implemented: the cycle of the origin (the last election day on or before
    # it), i.e. the information set the forecast was made from.
    return max(e for e in ctx["election_days"] if e <= origin).year


def group_values(cells: list[dict]) -> dict:
    out = {}
    for name, hs in GROUPS.items():
        sel = [c for c in cells if c["h"] in hs]
        out[name] = {"cells": len(sel)}
        for arm in A.ARMS:
            if not sel:
                out[name][arm] = None
                continue
            out[name][arm] = {
                "crps": float(np.mean([c["arms"][arm]["crps_mean"] for c in sel])),
                "coverage50": float(np.mean([v for c in sel for v in c["arms"][arm]["cov50"]])),
                "coverage90": float(np.mean([v for c in sel for v in c["arms"][arm]["cov90"]])),
            }
    return out


def by_horizon(cells: list[dict]) -> dict:
    out = {}
    for h in A.HORIZONS:
        sel = [c for c in cells if c["h"] == h]
        if sel:
            out[str(h)] = {"cells": len(sel), **{arm: {
                "crps": float(np.mean([c["arms"][arm]["crps_mean"] for c in sel])),
                "coverage50": float(np.mean([v for c in sel for v in c["arms"][arm]["cov50"]])),
                "coverage90": float(np.mean([v for c in sel for v in c["arms"][arm]["cov90"]])),
            } for arm in A.ARMS}}
    return out


def gates(groups: dict) -> dict:
    out = {}
    for arm in A.U_ARMS:
        long_, short, medium = groups["long"], groups["short"], groups["medium"]
        if any(g[arm] is None or g["D0"] is None for g in (long_, short, medium)):
            out[arm] = {"T2-G1": None, "T2-G2": None, "T2-G3": None, "passes": False, "note": "a group has no cells"}
            continue
        g1 = long_[arm]["crps"] < long_["D0"]["crps"]
        g2 = short[arm]["crps"] <= 1.05 * short["D0"]["crps"]
        g3 = all(0.80 <= g[arm]["coverage90"] <= 0.97 for g in (medium, long_))
        out[arm] = {"T2-G1": bool(g1), "T2-G2": bool(g2), "T2-G3": bool(g3), "passes": bool(g1 and g2 and g3),
                    "long_crps": long_[arm]["crps"], "long_crps_D0": long_["D0"]["crps"],
                    "short_crps": short[arm]["crps"], "short_crps_D0": short["D0"]["crps"],
                    "medium_coverage90": medium[arm]["coverage90"], "long_coverage90": long_[arm]["coverage90"]}
    return out


def select(gate_results: dict, groups: dict) -> dict:
    passing = [a for a in SIMPLICITY if gate_results[a]["passes"]]
    if not passing:
        return {"passing": [], "best": None, "selected": "D0", "rule": "no arm passes: CAND keeps D0 (§3.9.3)"}
    long_crps = {a: groups["long"][a]["crps"] for a in passing}
    best = min(passing, key=lambda a: long_crps[a])
    selected = best
    for a in SIMPLICITY:  # simplest first
        if a in passing and SIMPLICITY.index(a) < SIMPLICITY.index(best) and long_crps[a] <= 1.02 * long_crps[best]:
            selected = a
            break
    return {"passing": passing, "best": best, "selected": selected, "long_crps": long_crps,
            "rule": "lowest long-group CRPS; a simpler passing arm within 2 % preferred (§3.9)"}


def block_bootstrap(cells: list[dict]) -> dict:
    """§3.10: 10,000 resamples of election cycles of each U arm's long-group CRPS difference against D0."""

    sel = [c for c in cells if c["h"] in GROUPS["long"]]
    cycles = sorted({c["cycle"] for c in sel})
    if not cycles:
        return {"cycles": [], "note": "no long-group decision cells"}
    rng = np.random.default_rng(bootstrap_seed())
    draws = rng.integers(0, len(cycles), size=(BOOTSTRAP_RESAMPLES, len(cycles)))
    counts = np.array([sum(1 for c in sel if c["cycle"] == y) for y in cycles], dtype=float)
    out = {"seed": bootstrap_seed(), "resamples": BOOTSTRAP_RESAMPLES, "cycles": cycles,
           "cells_per_cycle": counts.astype(int).tolist(), "unit": "origin's election cycle", "arms": {}}
    for arm in A.U_ARMS:
        sums = np.array([sum(c["arms"][arm]["crps_mean"] - c["arms"]["D0"]["crps_mean"] for c in sel if c["cycle"] == y)
                         for y in cycles])
        stat = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        out["arms"][arm] = {"observed": float(sums.sum() / counts.sum()), "bootstrap_mean": float(stat.mean()),
                            "p025": float(np.quantile(stat, 0.025)), "p975": float(np.quantile(stat, 0.975)),
                            "share_negative": float(np.mean(stat < 0.0))}
    return out


# --------------------------------------------------------------------------- main

def main(argv=None) -> int:
    global _CTX
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    out_path = args.out or (SMOKE_OUTPUT if args.smoke else OUTPUT)
    if args.smoke and not str(out_path.resolve()).startswith("/tmp"):
        parser.error("--smoke output must be under /tmp")

    provenance = verify_final_round(allow_dirty=args.smoke)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    timings = {}
    t0 = time.time()
    ctx = A.t2_context(n_draws=SMOKE_N if args.smoke else A.N_DRAWS)
    check = self_check(ctx["rows"])
    if not check["ok"]:
        raise RuntimeError(f"v0.3 self-check failed: {check['abs_diff']}")
    # Shared tables, built once before forking.
    for h in (7, 14, 28, 91, 112):
        A._d0_transitions(ctx, h)
    for h in A.HORIZONS:
        A._uc_table(ctx, h)
    A._variogram_tables(ctx)
    A._centre_latest_publication(ctx)
    timings["context_seconds"] = round(time.time() - t0, 1)

    grid = A.grid_origins()
    _CTX = ctx

    # Phase A: OpinionState, Ω and θ at every grid origin.
    t0 = time.time()
    last_needed = grid[-1]
    if args.smoke:
        # k at a smoke origin needs every earlier scored origin, so phases A and B
        # run on the whole grid up to the last smoke origin.
        last_needed = grid[max(SMOKE_ORIGIN_INDICES)]
    phase_a_origins = [o for o in grid if o <= last_needed]
    for origin, state, omega, theta in _map(args.workers, _phase_a, phase_a_origins):
        ctx["cache"]["d0_state"][origin] = state
        ctx["cache"]["omega"][origin] = omega
        ctx["cache"]["theta"][origin] = theta
    first = A.first_scored_origin(ctx)
    ctx["scored_origins"] = [o for o in phase_a_origins if o >= first and A.origin_defined(ctx, o)["all"]]
    scored = ctx["scored_origins"]
    timings["phase_a_seconds"] = round(time.time() - t0, 1)

    # Phase B: k = 1 predictions at every scored origin (h ∈ {7, 28}).
    t0 = time.time()
    _CTX = ctx
    for chunk in _map(args.workers, _phase_b, scored):
        for key, value in chunk:
            ctx["cache"]["k1"][key] = value
    k_over_time = {arm: {} for arm in A.U_ARMS}
    for origin in scored:
        for arm in A.U_ARMS:
            k, n_eligible = A.inflation_k(arm, ctx, origin)
            k_over_time[arm][origin.isoformat()] = {"k": k, "eligible_targets": n_eligible}
    timings["phase_b_seconds"] = round(time.time() - t0, 1)

    # Phase C: every scored (origin, h) cell with a target.
    t0 = time.time()
    if args.smoke:
        score_origins = [grid[i] for i in SMOKE_ORIGIN_INDICES if grid[i] in scored]
        if first not in score_origins:
            score_origins.insert(0, first)
        horizons = SMOKE_HORIZONS
    else:
        score_origins, horizons = scored, A.HORIZONS
    tasks = [(o, [h for h in horizons if target_exists(ctx, o, h)]) for o in score_origins]
    tasks = [t for t in tasks if t[1]]
    _CTX = ctx
    cells = [c for chunk in _map(args.workers, _phase_c, tasks) for c in chunk]
    cells.sort(key=lambda c: (c["origin"], c["h"]))
    timings["phase_c_seconds"] = round(time.time() - t0, 1)
    timings["phase_c_cells"] = len(cells)

    decision = [c for c in cells if c["decision"]]
    reported = [c for c in cells if not c["decision"]]
    groups = group_values(decision)
    gate_results = gates(groups)
    selection = select(gate_results, groups)

    unscored = {o.isoformat(): A.origin_defined(ctx, o) for o in phase_a_origins if o not in scored}
    record = {
        "tier": "T2", "smoke": bool(args.smoke),
        "preregistration": "docs/calibration_2030_final_round_preregistration.md",
        "started_utc": started, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": provenance,
        "aggregate": {"version": AGGREGATE_VERSION, "self_check_ok": check["ok"],
                      "self_check_abs_diff": check["abs_diff"], "timeseries_file_sha256": ctx["v03_file_sha256"],
                      "rows": len(ctx["rows"]), "first_row": ctx["row_days"][0].isoformat(),
                      "last_row": ctx["row_days"][-1].isoformat()},
        "settings": {"draws": ctx["n_draws"], "horizons": list(horizons), "grid_origins": len(grid),
                     "decision_targets_before": DECISION_BEFORE.isoformat(), "groups": GROUPS,
                     "target_polls_excluded_after_cutoff": ctx["target_polls_excluded_after_cutoff"]},
        "first_scored_origin": first.isoformat(),
        "scored_origins": len(scored),
        "unscored_grid_origins": unscored,
        "targets": {"all_grid": sum(target_exists(ctx, o, h) for o in grid for h in A.HORIZONS),
                    "decision_grid": sum(target_exists(ctx, o, h) and o + timedelta(days=h) < DECISION_BEFORE
                                         for o in grid for h in A.HORIZONS),
                    "scored_cells": len(cells), "scored_decision_cells": len(decision),
                    "scored_reported_cells": len(reported)},
        "k_over_time": k_over_time,
        "theta_over_time": {o.isoformat(): ctx["cache"]["theta"][o] for o in scored},
        "omega_centres": {o.isoformat(): ctx["cache"]["omega"][o][1] for o in scored},
        "d0_opinion_state": {o.isoformat(): {"residual_poll_count": ctx["cache"]["d0_state"][o].residual_poll_count,
                                             "effective_poll_count": ctx["cache"]["d0_state"][o].effective_poll_count}
                             for o in scored},
        "decision": {"groups": groups, "by_horizon": by_horizon(decision), "gates": gate_results,
                     "selection": selection},
        "reported_only": {"targets_on_or_after_2026_09_13": {"groups": group_values(reported),
                                                              "by_horizon": by_horizon(reported)},
                          "block_bootstrap_long_crps_difference_vs_D0": block_bootstrap(decision)},
        "timings": timings,
        "cells": cells,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_path), "first_scored_origin": record["first_scored_origin"],
                      "scored_origins": len(scored), "cells": len(cells), "timings": timings,
                      "gates": {a: {k: v for k, v in g.items() if k.startswith("T2") or k == "passes"}
                                for a, g in gate_results.items()},
                      "selected": selection["selected"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
