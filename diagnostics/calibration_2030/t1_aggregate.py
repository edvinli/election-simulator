"""T1 of the frozen calibration-2030 preregistration: post-election behaviour of the poll aggregate.

This implements §2 of ``docs/calibration_2030_preregistration.md`` exactly.
It changes no production code. ``scripts/poll_aggregate/`` and the v0.2
files are only read.

Arms:

* ``C0`` (v0.2, with production's stored fits).
* ``S``: after each non-anchor election observation, add
  ``s_shock × multinomial_shape(s)`` to ``Cov[s, s]``.
* ``H``: after each non-anchor election observation, add
  ``h_reset × reference_shape`` to every house block.
* ``SH``: both.

The new hyperparameters are fitted by production's Nelder–Mead procedure
plus the frozen extra start. Then come the out-of-sample prequential poll
log scores, gates T1-G1..G3 and selection. Last comes arm ``W`` (daily
rolling consensus weights) on top of the selected filter arm.

Run with ``uv run python -m diagnostics.calibration_2030.t1_aggregate``. It
writes ``t1_results.json`` beside this file.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from scripts.poll_aggregate.build import _CheckpointedRunner, _known, _start, consensus_weights_at
from scripts.poll_aggregate.config import (
    DEFAULT_2026_RESULT_MANIFEST,
    DEFAULT_ELECTION_RESULTS_FILE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POLLS_FILE,
    HISTORY_START_ELECTION,
    HYPERPARAMETER_BOUNDS,
    HYPERPARAMETER_NAMES,
    HYPERPARAMETER_START,
    METADATA_FILENAME,
    OPTIMIZER_MAX_ITERATIONS,
    OPTIMIZER_TOLERANCE,
    PARTIES,
    TIMESERIES_FILENAME,
)
from scripts.poll_aggregate.data import load_elections, load_polls
from scripts.poll_aggregate.model import K, Hyperparameters, SupportFilter, multinomial_shape

from .common import verify_and_describe

OUTPUT = Path(__file__).resolve().parent / "t1_results.json"
V02_DIR = DEFAULT_OUTPUT_DIR / "v0_2"
NEW_PARAMS = {"s_shock": (1e-6, 1e-1), "h_reset": (1e-8, 1e-2)}
NEW_STARTS = {n: math.sqrt(lo * hi) for n, (lo, hi) in NEW_PARAMS.items()}  # geometric midpoints
ARMS = {"C0": (), "S": ("s_shock",), "H": ("h_reset",), "SH": ("s_shock", "h_reset")}
WINDOW_DAYS = 120
DECISION = (2014, 2018, 2022)
MID_CYCLE_GATED = (2010, 2014, 2018, 2022)
CONSENSUS_WINDOW = 365
CAUSALITY_CUTOFF = date(2026, 10, 31)
M_READING_DAYS = (date(2026, 9, 29), date(2026, 10, 14), date(2026, 11, 18))


class ArmFilter(SupportFilter):
    """The production filter plus the arm's post-election variance injections, and optional scoring."""

    def __init__(self, houses, reference_shares, params, s_shock=0.0, h_reset=0.0, record=None):
        super().__init__(houses, reference_shares, params)
        self.s_shock, self.h_reset, self.record = s_shock, h_reset, record

    def update(self, state, obs):
        z = None
        if self.record is not None and obs.kind == "poll":
            # Standardized innovation, computed exactly as update() will see it:
            # predict and house drift applied first; update() then re-applies
            # both with a zero gap, so nothing is applied twice.
            self.predict(state, obs.obs_date)
            house = self.house_index[obs.house]
            self._drift_house(state, house)
            lo = K * (1 + house)
            s = state.mean[:K]
            p = state.cov
            var = np.diag(p[:K, :K] + p[lo:lo + K, lo:lo + K] + p[:K, lo:lo + K] + p[lo:lo + K, :K])
            noise = np.diag(self.params.design_effect * multinomial_shape(s) / obs.sample_size) + obs.rounding_pp ** 2 / 12.0
            z = (np.asarray(obs.shares) - (s + state.mean[lo:lo + K])) / np.sqrt(var + noise)
        before = state.loglik
        super().update(state, obs)
        if z is not None:
            self.record[obs.key] = (state.loglik - before, z)
        if obs.kind == "election" and obs.obs_date != HISTORY_START_ELECTION:
            if self.s_shock:
                state.cov[:K, :K] += self.s_shock * multinomial_shape(state.mean[:K])
            if self.h_reset:
                for i in range(len(self.house_index)):
                    lo = K * (1 + i)
                    state.cov[lo:lo + K, lo:lo + K] += self.h_reset * self.reference_shape


def model_for(arm, houses, reference_shares, theta, record=None):
    base = Hyperparameters(**{n: theta[n] for n in HYPERPARAMETER_NAMES})
    return ArmFilter(houses, reference_shares, base, theta.get("s_shock", 0.0), theta.get("h_reset", 0.0), record)


def load_inputs():
    elections = load_elections(DEFAULT_ELECTION_RESULTS_FILE, [DEFAULT_2026_RESULT_MANIFEST])
    polls = load_polls(DEFAULT_POLLS_FILE, [e.obs_date for e in elections]).polls
    everything = sorted([*polls, *elections], key=lambda o: o.sort_key())
    houses = sorted({o.house for o in polls})
    return elections, polls, everything, houses


def segments(elections, everything):
    """Fit date, reference election and known observations per segment, as production builds them."""

    out = []
    for fit_date in sorted(e.available for e in elections)[1:]:
        known = _known(everything, fit_date)
        reference = max((o for o in known if o.kind == "election"), key=lambda o: o.obs_date)
        out.append({"fit_date": fit_date, "reference": reference, "known": known})
    return out


def loglik(arm, houses, reference_shares, theta, observations) -> float:
    model = model_for(arm, houses, reference_shares, theta)
    return model.run(model.initial_state(*_start(observations)), observations).loglik


def fit_segment(arm, houses, seg, previous, c0_theta):
    """Production's Nelder–Mead procedure over the arm's parameters, plus the frozen extra start."""

    names = HYPERPARAMETER_NAMES + ARMS[arm]
    bounds = {**HYPERPARAMETER_BOUNDS, **NEW_PARAMS}
    log_bounds = [(math.log(bounds[n][0]), math.log(bounds[n][1])) for n in names]
    config_start = {**HYPERPARAMETER_START, **NEW_STARTS}
    starts = [[math.log(config_start[n]) for n in names]]
    if previous is not None:
        starts.append([math.log(previous[n]) for n in names])
    starts.append([math.log({**c0_theta, **NEW_STARTS}[n]) for n in names])
    start_day_shares = _start(seg["known"])
    evaluations = 0

    def negative(x):
        nonlocal evaluations
        evaluations += 1
        if any(not lo <= v <= hi for v, (lo, hi) in zip(x, log_bounds)):
            return 1e12
        theta = {n: math.exp(v) for n, v in zip(names, x)}
        model = model_for(arm, houses, seg["reference"].shares, theta)
        try:
            return -model.run(model.initial_state(*start_day_shares), seg["known"]).loglik
        except np.linalg.LinAlgError:
            return 1e12

    def search(x0):
        return minimize(negative, np.asarray(x0, dtype=float), method="Nelder-Mead",
                        options={"maxiter": OPTIMIZER_MAX_ITERATIONS, "xatol": OPTIMIZER_TOLERANCE, "fatol": OPTIMIZER_TOLERANCE})

    results = [search(x0) for x0 in starts]
    best = min(results, key=lambda r: r.fun)
    results.append(search(best.x))
    best = min(results, key=lambda r: r.fun)
    theta = {n: math.exp(v) for n, v in zip(names, best.x)}
    return theta, {"loglik": -float(best.fun), "converged": bool(best.success), "iterations": int(sum(r.nit for r in results)),
                   "evaluations": evaluations, "starts": len(starts)}


def build_rows(arm, houses, polls, everything, segs, thetas, end, weights_mode):
    """Daily readings exactly as ``build_aggregate`` makes them, with the arm's filter.

    ``weights_mode``: "frozen" (v0.2: weights fixed at the fit date) or
    "rolling" (arm W: recomputed on each day with v0.2's own window test).
    """

    change_dates = sorted({o.available for o in everything if o.available <= end} | {s["fit_date"] for s in segs})
    rows = {}
    for i, seg in enumerate(segs):
        fit_date = seg["fit_date"]
        if fit_date > end:
            break
        seg_end = min(segs[i + 1]["fit_date"] - timedelta(days=1), end) if i + 1 < len(segs) else end
        frozen = consensus_weights_at(polls, fit_date, CONSENSUS_WINDOW)
        model = model_for(arm, houses, seg["reference"].shares, thetas[i])
        runner = _CheckpointedRunner(model, *_start(seg["known"]))
        dates = [d for d in change_dates if fit_date <= d <= seg_end]
        for c, info_date in enumerate(dates):
            state = runner.run(_known([o for o in everything if o.available <= end], info_date))
            until = dates[c + 1] - timedelta(days=1) if c + 1 < len(dates) else seg_end
            day = info_date
            while day <= until:
                weights = frozen if weights_mode == "frozen" else consensus_weights_at(polls, day, CONSENSUS_WINDOW)
                mean, _ = model.consensus_reading_at(state, day, weights)
                rows[day] = (fit_date, mean.copy())
                day += timedelta(days=1)
    return rows


def score_polls(arm, houses, everything, segs, thetas, election_days):
    """Out-of-sample prequential log score per poll, by segment and window (frozen §2.2)."""

    scored = []
    for i, seg in enumerate(segs):
        record = {}
        model = model_for(arm, houses, seg["reference"].shares, thetas[i], record)
        model.run(model.initial_state(*_start(everything)), everything)
        e_day = election_days[i + 1]  # segment i is fitted on election i+1's result (2006 is the anchor)
        next_day = election_days[i + 2] if i + 2 < len(election_days) else date.max
        for o in everything:
            if o.kind != "poll" or not (e_day <= o.obs_date < next_day) or o.available <= seg["fit_date"]:
                continue
            ll, z = record[o.key]
            scored.append({"key": o.key, "segment": e_day.year, "window": "W" if (o.obs_date - e_day).days < WINDOW_DAYS else "M",
                           "midpoint": o.obs_date.isoformat(), "published": o.available.isoformat(), "house": o.house,
                           "loglik": ll, "z": [float(v) for v in z]})
    return scored


def run_arm(arm: str) -> dict:
    t0 = time.time()
    elections, polls, everything, houses = load_inputs()
    segs = segments(elections, everything)
    stored = json.loads((V02_DIR / METADATA_FILENAME).read_text(encoding="utf-8"))["segments"]
    if [s["hyperparameters_fit_date"] for s in stored] != [s["fit_date"].isoformat() for s in segs]:
        raise RuntimeError("stored v0.2 segments do not match the reconstructed fit dates")
    c0 = [dict(s["hyperparameters_exact"]) for s in stored]
    fits, thetas = [], []
    for i, seg in enumerate(segs):
        c0_ll = loglik("C0", houses, seg["reference"].shares, c0[i], seg["known"])
        if arm == "C0":
            theta, info = c0[i], {"loglik": c0_ll, "stored": True}
        else:
            theta, info = fit_segment(arm, houses, seg, thetas[-1] if thetas else None, c0[i])
        thetas.append(theta)
        fits.append({"fit_date": seg["fit_date"].isoformat(), "theta": theta, **info, "c0_loglik": c0_ll,
                     "T1-G3_nesting_ok": info["loglik"] >= c0_ll - 0.5})
    election_days = sorted(e.obs_date for e in elections)
    scored = score_polls(arm, houses, everything, segs, thetas, election_days)
    end = max(o.available for o in everything)
    rows = build_rows(arm, houses, polls, everything, segs, thetas, end, "frozen")
    truncated = build_rows(arm, houses, polls, everything, segs, thetas, CAUSALITY_CUTOFF, "frozen")
    causal = all(np.array_equal(rows[d][1], truncated[d][1]) for d in truncated)
    simplex = all(float(m.min()) > 0.0 and float(m.sum()) < 100.0 for _, m in rows.values())
    return {
        "arm": arm, "fits": fits, "thetas": thetas, "scored_polls": scored,
        "rows": {d.isoformat(): [f.isoformat(), m.tolist()] for d, (f, m) in rows.items()},
        "integrity": {"causality_rows_checked": len(truncated), "causality_ok": causal, "simplex_ok": simplex,
                      "nesting_ok": all(f["T1-G3_nesting_ok"] for f in fits)},
        "seconds": round(time.time() - t0, 1),
    }


def window_sums(scored):
    out = {}
    for p in scored:
        key = (p["segment"], p["window"])
        out.setdefault(key, []).append(p["loglik"])
    return {f"{s}{w}": {"n": len(v), "sum": float(np.sum(v)), "mean": float(np.mean(v))} for (s, w), v in sorted(out.items())}


def innovation_summary(scored):
    out = {}
    for p in scored:
        out.setdefault(f"{p['segment']}{p['window']}", []).append(p["z"])
    return {k: {"n": len(v), "mean": dict(zip(PARTIES, np.round(np.mean(v, axis=0), 3).tolist())),
                "sd": dict(zip(PARTIES, np.round(np.std(v, axis=0), 3).tolist()))} for k, v in sorted(out.items())}


def gates(arms: dict) -> dict:
    c0 = window_sums(arms["C0"]["scored_polls"])
    out = {}
    for arm in ("S", "H", "SH"):
        w = window_sums(arms[arm]["scored_polls"])
        for k in c0:
            if w[k]["n"] != c0[k]["n"]:
                raise RuntimeError(f"{arm} scored a different poll set in {k}")
        gains = {y: w[f"{y}W"]["sum"] - c0[f"{y}W"]["sum"] for y in DECISION}
        g1 = sum(g > 0 for g in gains.values()) >= 2 and sum(gains.values()) > 0
        mid = {y: w[f"{y}M"]["mean"] - c0[f"{y}M"]["mean"] for y in MID_CYCLE_GATED}
        g2 = all(d >= -0.01 for d in mid.values())
        integ = arms[arm]["integrity"]
        g3 = integ["nesting_ok"] and integ["simplex_ok"] and integ["causality_ok"]
        out[arm] = {"post_election_gain_nats": gains, "post_election_gain_total": float(sum(gains.values())),
                    "mid_cycle_mean_delta_nats": mid, "T1-G1": g1, "T1-G2": g2, "T1-G3": g3, "passes": g1 and g2 and g3}
    passing = [a for a in out if out[a]["passes"]]
    singles = [a for a in ("S", "H") if a in passing]
    total = {a: out[a]["post_election_gain_total"] for a in out}
    if "SH" in passing and (not singles or total["SH"] >= max(total[a] for a in singles) + 2.0):
        selected = "SH"
    elif singles:
        selected = max(singles, key=lambda a: (total[a], a == "S"))
    else:
        selected = "C0"
    return {"arms": out, "passing": passing, "selected_filter_arm": selected,
            "selection_reading": "'the better single arm' ranges over passing single arms; with SH passing and no single arm passing, SH is selected"}


def rolling_weights_gate(arms, selected, rolling_rows, polls) -> dict:
    """T1-GW: daily rolling weights against frozen weights, on the selected filter arm."""

    frozen_rows = {date.fromisoformat(d): (date.fromisoformat(f), np.array(m)) for d, (f, m) in arms[selected]["rows"].items()}
    fit_year = {}
    for d, (f, _) in frozen_rows.items():
        fit_year[d] = f.year  # fit dates fall a week after their election, in the same year
    per = {"frozen": {}, "rolling": {}}
    by_pub = sorted(polls, key=lambda o: o.available)
    for d in sorted(frozen_rows):
        upcoming = [o for o in by_pub if d < o.available <= d + timedelta(days=30)]
        if len(upcoming) < 3:
            continue
        n = np.array([o.sample_size for o in upcoming])
        target = (n[:, None] * np.array([o.shares for o in upcoming])).sum(axis=0) / n.sum()
        for mode, mean in (("frozen", frozen_rows[d][1]), ("rolling", rolling_rows[d][1])):
            per[mode].setdefault(fit_year[d], []).append(float(np.mean((mean - target) ** 2)))
    seg = {y: {m: float(np.mean(per[m][y])) for m in per} | {"days": len(per["frozen"][y])} for y in sorted(per["frozen"])}
    gated = [y for y in MID_CYCLE_GATED if y in seg]
    pooled = {m: float(np.mean([v for y in gated for v in per[m][y]])) for m in per}
    wins = sum(seg[y]["rolling"] < seg[y]["frozen"] for y in gated)
    return {"segments": seg, "pooled_2010_2022": pooled, "rolling_better_segments": wins,
            "T1-GW": wins >= 3 and pooled["rolling"] < pooled["frozen"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    provenance = verify_and_describe()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(ARMS)) as pool:
        arms = dict(zip(ARMS, pool.map(run_arm, ARMS)))

    # Harness self-check: C0 must reproduce the pinned v0.2 series to its 2-decimal rounding.
    with (V02_DIR / TIMESERIES_FILENAME).open(encoding="utf-8") as fh:
        pinned = {r["date"]: [float(r[p]) for p in PARTIES] for r in csv.DictReader(fh)}
    c0_rows = arms["C0"]["rows"]
    mismatches = [d for d, v in pinned.items() if d not in c0_rows or
                  any(abs(round(x, 2) - y) > 0.0051 for x, y in zip(c0_rows[d][1], v))]

    decision = gates(arms)
    selected = decision["selected_filter_arm"]
    elections, polls, everything, houses = load_inputs()
    segs = segments(elections, everything)
    rolling_rows = build_rows(selected, houses, polls, everything, segs, arms[selected]["thetas"],
                              max(o.available for o in everything), "rolling")
    decision["rolling_weights"] = rolling_weights_gate(arms, selected, rolling_rows, polls)
    m_index = PARTIES.index("M")
    record = {
        "tier": "T1", "preregistration": "docs/calibration_2030_preregistration.md",
        "started_utc": started, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": provenance,
        "harness_self_check": {"c0_reproduces_pinned_v0_2": not mismatches, "pinned_rows": len(pinned), "mismatched_rows": mismatches[:20]},
        "window_definition": "W_e: 0 <= (midpoint - election day).days < 120",
        "decision": decision,
        "arms": {
            a: {"fits": r["fits"], "integrity": r["integrity"], "seconds": r["seconds"],
                "window_scores": window_sums(r["scored_polls"]), "innovations": innovation_summary(r["scored_polls"]),
                "m_reading_2026": {d.isoformat(): r["rows"][d.isoformat()][1][m_index] for d in M_READING_DAYS},
                "m_reading_2026_rolling_weights": ({d.isoformat(): float(rolling_rows[d][1][m_index]) for d in M_READING_DAYS}
                                                   if a == selected else None)}
            for a, r in arms.items()
        },
    }
    OUTPUT.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"self_check": record["harness_self_check"]["c0_reproduces_pinned_v0_2"],
                      "gates": {a: {k: v for k, v in g.items() if k.startswith("T1") or k == "passes"} for a, g in decision["arms"].items()},
                      "selected": selected, "T1-GW": decision["rolling_weights"]["T1-GW"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
