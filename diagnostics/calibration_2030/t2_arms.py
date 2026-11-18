"""T2 arms of the frozen final-round preregistration (state uncertainty and dynamics).

``docs/calibration_2030_final_round_preregistration.md`` §3 defines five arms
that each turn the v0.3 aggregate at a forecast origin into draws of the
eight named-party shares at a target date D:

* ``D0``: production OpinionState + production symmetric CLR bootstrap, run on
  the v0.3 rows written as a production timeseries file;
* ``Ua``: the v0.3 filter's consensus-reading covariance × k², plus the
  filter's own random-walk forecast and the election-day shocks;
* ``Ub1``: as ``Ua`` in the state, OU dynamics anchored at the reading;
* ``Ub2``: as ``Ub1`` but with the mean pulled toward the last election;
* ``Uc``: as ``Ua`` in the state, symmetric CLR bootstrap of v0.3
  transitions at horizon h exactly.

This module is a library: other harnesses (T4, T5) import it. It reads
production code and never modifies it. Every per-origin intermediate is cached
inside ``ctx``; each cache entry is a deterministic recomputation.

The common poll-average term Ω (§3.3) is *not* added here:
``state_dynamics_draws`` returns state + dynamics after the §3.4 simplex map.
"""

from __future__ import annotations

import bisect
import hashlib
import math
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from scripts.hindcasts.models import (
    derive_opinion_state_seed,
    derive_shared_dynamics_seed,
    sample_shared_symmetric_dynamics,
)
from scripts.poll_aggregate.build import _known, _start, consensus_weights_at
from scripts.poll_aggregate.config import PARTIES
from scripts.poll_aggregate.model import K, multinomial_shape
from scripts.pollofpolls.clr import clr_to_composition_matrix
from scripts.pollofpolls.state import estimate_opinion, load_timeseries_dataset
from scripts.pollofpolls.state_config import ALL_CATEGORIES
from scripts.pollofpolls.transitions import (
    MIN_TRANSITIONS,
    build_all_historical_transitions,
    filter_transitions_as_of,
)

from .common import ROOT, sha256
from .t1_aggregate import model_for
from .v03 import v03_context, v03_rows, write_timeseries

ARMS = ("D0", "Ua", "Ub1", "Ub2", "Uc")
U_ARMS = ("Ua", "Ub1", "Ub2", "Uc")
COLUMNS = (*PARTIES, "REST")

HORIZONS = (7, 28, 91, 182, 365, 730, 1095, 1460)
K_HORIZONS = (7, 28)
N_DRAWS = 4000
GRID_FIRST = date(2010, 10, 3)
GRID_STEP = 14
POLL_CUTOFF = date(2026, 11, 18)
ROWS_FIRST = date(2010, 9, 26)
WINDOW_HALF = 7
MIN_TARGET_POLLS = 3
MIN_OMEGA_CENTRES = 30
VARIOGRAM_LAGS = (28, 91, 182, 365, 730)
MIN_VARIOGRAM_LAGS = 3
THETA_BOUNDS = (1e-5, 1e-1)
K_BOUNDS = (0.5, 3.0)
MIN_K_TARGETS = 20
CLIP_PP = 0.01
CONSENSUS_WINDOW = 365
D0_CAP = 112
D0_FALLBACKS = (28, 14, 7)
PRODUCTION_POLLS_FILE = ROOT / "data" / "processed" / "poll_aggregate" / "v0_2" / "individual_polls.csv"


# --------------------------------------------------------------------------- seeds and small pure helpers

def t2_seed(origin: date, h: int) -> int:
    """§3.6: int(sha256("calibration-2030:T2:<origin ISO>:<h>")[:8], 16)."""

    return int(hashlib.sha256(f"calibration-2030:T2:{origin.isoformat()}:{h}".encode("utf-8")).hexdigest()[:8], 16)


def grid_origins() -> list[date]:
    """§3.1: every 14th day from 2010-10-03 through 2026-11-18 (421 origins)."""

    out, day = [], GRID_FIRST
    while day <= POLL_CUTOFF:
        out.append(day)
        day += timedelta(days=GRID_STEP)
    return out


def simplex_map(eight: np.ndarray) -> np.ndarray:
    """§3.4: complete with REST = 100 − Σ eight, clip each of nine at 0.01 pp, renormalize to 100."""

    eight = np.atleast_2d(np.asarray(eight, dtype=float))
    nine = np.concatenate([eight, 100.0 - eight.sum(axis=1, keepdims=True)], axis=1)
    nine = np.maximum(nine, CLIP_PP)
    return 100.0 * nine / nine.sum(axis=1, keepdims=True)


def _factor(cov: np.ndarray) -> np.ndarray:
    if not np.any(cov):
        return np.zeros_like(cov)
    return np.linalg.cholesky(0.5 * (cov + cov.T))


def ou_variance_factor(theta: float, days: int) -> float:
    """(1 − e^{−θ·days})/θ: multiply by σ² for the OU dynamics variance (§3.4)."""

    return -math.expm1(-theta * days) / theta


def fit_theta(lags, gamma_hat, sigma2: float) -> float:
    """§3.5: θ minimizing Σ (log γ̂(h) − log((σ²/θ)(1 − e^{−θh})))² on [1e-5, 1e-1].

    The minimization is over log θ: a dense grid locates the basin, then a
    bounded Brent search refines it inside the bracketing grid cells. The
    procedure is deterministic.
    """

    lags = np.asarray(lags, dtype=float)
    log_gamma = np.log(np.asarray(gamma_hat, dtype=float))

    def objective(log_theta: float) -> float:
        theta = math.exp(log_theta)
        model = sigma2 * (-np.expm1(-theta * lags)) / theta
        return float(np.sum((log_gamma - np.log(model)) ** 2))

    lo, hi = math.log(THETA_BOUNDS[0]), math.log(THETA_BOUNDS[1])
    grid = np.linspace(lo, hi, 2001)
    values = [objective(x) for x in grid]
    best = int(np.argmin(values))
    a, b = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
    res = minimize_scalar(objective, bounds=(a, b), method="bounded", options={"xatol": 1e-10})
    x = float(res.x) if res.fun <= values[best] else float(grid[best])
    return float(min(max(math.exp(x), THETA_BOUNDS[0]), THETA_BOUNDS[1]))


def solve_k(e2: np.ndarray, s: np.ndarray, d: np.ndarray, om: np.ndarray) -> float:
    """§3.5: k on [0.5, 3] solving mean_t (1/8) Σ_p e²/(k² S + D + Ω) = 1; no root → nearer bound.

    Arrays are (targets, 8). The left side decreases in k, so "no root" means
    it stays above 1 (k = 3) or below 1 (k = 0.5) over the whole interval.
    """

    e2, s, d, om = (np.asarray(x, dtype=float) for x in (e2, s, d, om))

    def f(k: float) -> float:
        return float(np.mean(np.mean(e2 / (k * k * s + d + om), axis=1))) - 1.0

    lo, hi = K_BOUNDS
    f_lo, f_hi = f(lo), f(hi)
    if f_lo <= 0.0:
        return lo
    if f_hi >= 0.0:
        return hi
    return float(brentq(f, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200))


# --------------------------------------------------------------------------- context

def t2_context(n_draws: int = N_DRAWS, v03_dir: Path | None = None) -> dict:
    """v0.3 context, rows, the v0.3 timeseries file, target polls, and empty caches."""

    ctx = v03_context()
    rows = v03_rows(ctx)
    ctx["rows"] = rows
    ctx["row_days"] = sorted(rows)
    ctx["row_array"] = {d: np.asarray(m, dtype=float) for d, (_, m) in rows.items()}
    directory = Path(v03_dir) if v03_dir else Path(tempfile.mkdtemp(prefix="calibration2030_t2_v03_"))
    ctx["v03_file"] = write_timeseries(rows, directory / "swedishpolls_aggregate_timeseries.csv")
    ctx["v03_file_sha256"] = sha256(ctx["v03_file"])
    ctx["polls_file"] = PRODUCTION_POLLS_FILE
    ctx["election_days"] = sorted(e.obs_date for e in ctx["elections"])
    ctx["n_draws"] = int(n_draws)

    # §3.2 target polls: load_polls observations, none published after the poll cutoff.
    tp = [o for o in ctx["polls"] if o.available <= POLL_CUTOFF]
    ctx["target_polls_excluded_after_cutoff"] = len(ctx["polls"]) - len(tp)
    ctx["tp_obs"] = np.array([o.obs_date.toordinal() for o in tp], dtype=np.int64)
    ctx["tp_avail"] = np.array([o.available.toordinal() for o in tp], dtype=np.int64)
    ctx["tp_n"] = np.array([float(o.sample_size) for o in tp])
    ctx["tp_shares"] = np.array([o.shares for o in tp], dtype=float)

    ctx["cache"] = {"state": {}, "omega": {}, "theta": {}, "d0_state": {}, "k1": {}, "k": {},
                    "target": {}, "window": {}}
    ctx["first_scored_origin"] = None
    return ctx


# --------------------------------------------------------------------------- §3.2 targets

def _window_mask(ctx: dict, target: date) -> np.ndarray:
    key = target
    mask = ctx["cache"]["window"].get(key)
    if mask is None:
        mask = np.abs(ctx["tp_obs"] - target.toordinal()) <= WINDOW_HALF
        ctx["cache"]["window"][key] = mask
    return mask


def poll_target(ctx: dict, target: date, known_by: date | None = None) -> np.ndarray | None:
    """§3.2: effective-sample-weighted mean of the polls with midpoint in [D−7, D+7]; ≥ 3 polls.

    With ``known_by``, only polls published on or before that day are used.
    """

    key = (target, known_by)
    cache = ctx["cache"]["target"]
    if key in cache:
        return cache[key]
    mask = _window_mask(ctx, target)
    if known_by is not None:
        mask = mask & (ctx["tp_avail"] <= known_by.toordinal())
    if int(mask.sum()) < MIN_TARGET_POLLS:
        value = None
    else:
        n = ctx["tp_n"][mask]
        value = (n[:, None] * ctx["tp_shares"][mask]).sum(axis=0) / n.sum()
    cache[key] = value
    return value


def target_fully_known(ctx: dict, target: date, by: date) -> bool:
    """§3.5 eligibility: the target exists and every poll in its window was published on or before ``by``."""

    if poll_target(ctx, target) is None:
        return False
    mask = _window_mask(ctx, target)
    return bool(np.all(ctx["tp_avail"][mask] <= by.toordinal()))


# --------------------------------------------------------------------------- filter state at an origin

def _segment_index(ctx: dict, origin: date) -> int:
    fits = [s["fit_date"] for s in ctx["segments"]]
    i = bisect.bisect_right(fits, origin) - 1
    if i < 0:
        raise ValueError(f"no v0.3 segment covers {origin}")
    return i


def origin_state(ctx: dict, origin: date) -> dict:
    """The v0.3 filter on the origin over the observations known that day, with the segment's SH fit."""

    cache = ctx["cache"]["state"]
    if origin in cache:
        return cache[origin]
    i = _segment_index(ctx, origin)
    seg, theta = ctx["segments"][i], ctx["thetas"][i]
    model = model_for("SH", ctx["houses"], seg["reference"].shares, theta)
    state = model.run(model.initial_state(*_start(seg["known"])), _known(ctx["everything"], origin))
    weights = consensus_weights_at(ctx["polls"], origin, CONSENSUS_WINDOW)
    mean, cov = model.consensus_reading_at(state, origin, weights)
    if origin in ctx["row_array"] and not np.array_equal(mean, ctx["row_array"][origin]):
        raise RuntimeError(f"filter reading at {origin} does not reproduce the v0.3 row")
    total = sum(weights.values())
    house_shape = sum((w / total) ** 2 for w in weights.values()) * model.reference_shape
    elections = [e for e in ctx["elections"] if e.available <= origin]
    anchor = np.asarray(max(elections, key=lambda e: e.obs_date).shares, dtype=float)
    out = {"segment": i, "theta": theta, "model": model, "state": state, "weights": weights,
           "mean": mean, "cov": cov, "latent": state.mean[:K].copy(), "house_shape": house_shape,
           "anchor": anchor, "chol_state": _factor(cov)}
    cache[origin] = out
    return out


def election_shock_cov(ctx: dict, origin: date, target: date) -> np.ndarray:
    """§3.4 shocks for every election day E with origin < E < D (open interval)."""

    # Election days are the dates of the pinned results file plus the 2026
    # manifest (2006–2026). A caller whose open interval contains a later,
    # scheduled election (e.g. a T5 origin with a target after 2030-09) would
    # need that date added to ctx["election_days"].
    st = origin_state(ctx, origin)
    count = sum(1 for e in ctx["election_days"] if origin < e < target)
    if not count:
        return np.zeros((K, K))
    per = st["theta"]["s_shock"] * multinomial_shape(st["latent"]) + st["theta"]["h_reset"] * st["house_shape"]
    return count * per


def ua_dynamics_cov(ctx: dict, origin: date, target: date) -> np.ndarray:
    """What consensus_reading_at(state, D, w) adds to the covariance at the origin, plus shocks."""

    st = origin_state(ctx, origin)
    _, cov_d = st["model"].consensus_reading_at(st["state"], target, st["weights"])
    return (cov_d - st["cov"]) + election_shock_cov(ctx, origin, target)


def ub_dynamics_cov(ctx: dict, origin: date, target: date, theta: float) -> np.ndarray:
    """(σ²/θ)(1 − e^{−θ·days}) × multinomial_shape(s), no house drift, plus shocks."""

    st = origin_state(ctx, origin)
    days = (target - origin).days
    sigma2 = st["theta"]["process"]
    return (sigma2 * ou_variance_factor(theta, days) * multinomial_shape(st["latent"])
            + election_shock_cov(ctx, origin, target))


# --------------------------------------------------------------------------- §3.3 Ω

def _centre_latest_publication(ctx: dict) -> dict:
    """Per daily centre from 2010-09-26 with any window poll: the latest publication (ordinal) in its window."""

    if "centre_table" in ctx:
        return ctx["centre_table"]
    days = [d for d in ctx["row_days"] if d >= ROWS_FIRST]
    table = {}
    for c in days:
        mask = _window_mask(ctx, c)
        if int(mask.sum()) == 0:
            continue
        table[c] = int(ctx["tp_avail"][mask].max())
    ctx["centre_table"] = table
    return table


def omega_at(ctx: dict, origin: date) -> tuple[np.ndarray, int] | None:
    """§3.3: sample covariance (8×8, pp) of window poll average − v0.3 reading over daily centres.

    Centres: every day c ≥ 2010-09-26 with c + 7 ≤ origin meeting the 3-poll
    minimum with polls published on or before the origin. At least 30.
    """

    # PREREG-AMBIGUITY: "the v0.3 reading at c, taken from c's causal row" does
    # not say rounded (file) or unrounded. Reading implemented: the unrounded
    # row, because Ω is the common term of the U arms, which §2 gives the
    # unrounded state. Ω's square-root factor for the draws is its Cholesky
    # factor (the spec names none; the law N(0, Ω) is the same).
    cache = ctx["cache"]["omega"]
    if origin in cache:
        return cache[origin]
    table = _centre_latest_publication(ctx)
    last = origin - timedelta(days=WINDOW_HALF)
    diffs = []
    for c in ctx["row_days"]:
        if c > last:
            break
        if c < ROWS_FIRST or c not in table:
            continue
        avg = poll_target(ctx, c) if table[c] <= origin.toordinal() else poll_target(ctx, c, known_by=origin)
        if avg is None:
            continue
        diffs.append(avg - ctx["row_array"][c])
    if len(diffs) < MIN_OMEGA_CENTRES:
        out = None
    else:
        out = (np.cov(np.asarray(diffs), rowvar=False, ddof=1), len(diffs))
    cache[origin] = out
    return out


def _omega_factor(ctx: dict, origin: date) -> np.ndarray:
    key = ("chol", origin)
    cache = ctx["cache"]["omega"]
    if key not in cache:
        om = omega_at(ctx, origin)
        cache[key] = None if om is None else np.linalg.cholesky(om[0])
    return cache[key]


def poll_term(ctx: dict, origin: date, z_poll: np.ndarray) -> np.ndarray:
    """The common §3.3 draw N(0, Ω_origin) from shared standard normals (n × 8)."""

    factor = _omega_factor(ctx, origin)
    if factor is None:
        raise ValueError(f"Ω undefined at {origin}")
    return np.asarray(z_poll) @ factor.T


# --------------------------------------------------------------------------- §3.5 θ

def _variogram_tables(ctx: dict) -> dict:
    """Per lag: later dates (ordinal, sorted) and per-pair party-averaged normalized squared increments."""

    if "variogram" in ctx:
        return ctx["variogram"]
    rows = ctx["row_array"]
    days = [d for d in ctx["row_days"] if d >= ROWS_FIRST]
    tables = {}
    for lag in VARIOGRAM_LAGS:
        later, values = [], []
        for t in days:
            u = t + timedelta(days=lag)
            if u not in rows:
                continue
            x_t = rows[t]
            inc = (rows[u] - x_t) ** 2 / np.diag(multinomial_shape(x_t))
            later.append(u.toordinal())
            values.append(float(np.mean(inc)))
        order = np.argsort(later, kind="stable")
        later_arr = np.asarray(later, dtype=np.int64)[order]
        values_arr = np.asarray(values, dtype=float)[order]
        tables[lag] = (later_arr, np.concatenate([[0.0], np.cumsum(values_arr)]))
    ctx["variogram"] = tables
    return tables


def theta_details(ctx: dict, origin: date) -> dict:
    cache = ctx["cache"]["theta"]
    if origin in cache:
        return cache[origin]
    tables = _variogram_tables(ctx)
    lags, gammas, pairs = [], [], []
    for lag in VARIOGRAM_LAGS:
        later, csum = tables[lag]
        n = int(np.searchsorted(later, origin.toordinal(), side="right"))
        if n == 0:
            continue
        lags.append(lag)
        gammas.append(float(csum[n] / n))
        pairs.append(n)
    sigma2 = ctx["thetas"][_segment_index(ctx, origin)]["process"]
    theta = fit_theta(lags, gammas, sigma2) if len(lags) >= MIN_VARIOGRAM_LAGS else None
    out = {"theta": theta, "lags": lags, "gamma_hat": gammas, "pairs": pairs, "sigma2": sigma2}
    cache[origin] = out
    return out


def theta_at(ctx: dict, origin: date) -> float | None:
    """§3.5: the scalar OU decay θ fitted at the origin; None if fewer than 3 lags have pairs."""

    return theta_details(ctx, origin)["theta"]


# --------------------------------------------------------------------------- transitions (Uc and D0)

def _uc_rows_data(ctx: dict) -> list[dict]:
    if "uc_rows_data" not in ctx:
        data = []
        for d in ctx["row_days"]:
            m = ctx["row_array"][d]
            comp = {p: float(v) for p, v in zip(PARTIES, m)}
            comp["REST"] = max(100.0 - float(m.sum()), 0.0)
            data.append({"date": d, "composition": comp})
        ctx["uc_rows_data"] = data
    return ctx["uc_rows_data"]


def _uc_table(ctx: dict, h: int) -> tuple[np.ndarray, np.ndarray]:
    """End dates (ordinal, ascending) and CLR deltas of the unrounded v0.3 transitions at h."""

    store = ctx.setdefault("uc_transitions", {})
    if h not in store:
        ts = build_all_historical_transitions(_uc_rows_data(ctx), horizons=[h])[h]
        store[h] = (np.array([t.end_date.toordinal() for t in ts], dtype=np.int64),
                    np.array([t.clr_transition for t in ts], dtype=float).reshape(-1, len(ALL_CATEGORIES)))
    return store[h]


def uc_pool(ctx: dict, origin: date, h: int) -> tuple[np.ndarray, int, float] | None:
    """Eligible Uc deltas (end_date ≤ origin) at h exactly; with fewer than 30, the largest grid
    h′ < h with at least 30, deltas scaled by √(h/h′). Returns (deltas, h used, scale) or None."""

    # PREREG-AMBIGUITY: §3.4 says Uc bootstraps "v0.3 transitions"; §2 says the
    # U arms use the unrounded state and only D0/FB/T4 read the rounded file.
    # Reading implemented: Uc's transitions come from the unrounded v0.3 rows
    # (REST = 100 − Σ eight, clamped at 0 as load_timeseries_dataset does),
    # built by production's build_all_historical_transitions.
    if h <= 0:
        return None
    for hp in [h] + sorted((x for x in HORIZONS if x < h), reverse=True):
        ends, deltas = _uc_table(ctx, hp)
        n = int(np.searchsorted(ends, origin.toordinal(), side="right"))
        if n >= MIN_TRANSITIONS:
            return deltas[:n], hp, math.sqrt(h / hp)
    return None


def _d0_timeseries(ctx: dict) -> list:
    if "d0_timeseries" not in ctx:
        ctx["d0_timeseries"] = load_timeseries_dataset(ctx["v03_file"])
    return ctx["d0_timeseries"]


def _d0_transitions(ctx: dict, h: int) -> list:
    store = ctx.setdefault("d0_transitions", {})
    if h not in store:
        store[h] = build_all_historical_transitions(_d0_timeseries(ctx), horizons=[h])[h]
    return store[h]


def d0_eligible(ctx: dict, origin: date, horizon_days: int) -> tuple[tuple, int]:
    """Production's rule in national_engine: eval horizon min(h, 112), fallback 28, 14, 7 below 30."""

    eval_h = min(horizon_days, D0_CAP) if horizon_days > D0_CAP else horizon_days
    eligible = filter_transitions_as_of(_d0_transitions(ctx, eval_h), origin)
    if len(eligible) < MIN_TRANSITIONS:
        for fallback_h in D0_FALLBACKS:
            fb = filter_transitions_as_of(_d0_transitions(ctx, fallback_h), origin)
            eligible = fb
            if len(fb) >= MIN_TRANSITIONS:
                eval_h = fallback_h
                break
    return eligible, eval_h


def d0_opinion_state(ctx: dict, origin: date):
    """estimate_opinion on the v0.3 file and the production polls file; an exception is cached and re-raised."""

    cache = ctx["cache"]["d0_state"]
    if origin not in cache:
        try:
            cache[origin] = estimate_opinion(as_of=origin, timeseries_file=ctx["v03_file"],
                                             polls_file=ctx["polls_file"])
        except ValueError as exc:
            cache[origin] = exc
    value = cache[origin]
    if isinstance(value, Exception):
        raise value
    return value


def _d0_draws(ctx: dict, origin: date, target: date, n: int, base_seed: int) -> np.ndarray:
    """national_engine.generate_national_vote_shares steps 1–4 on the v0.3 inputs, then the §3.4 map."""

    opinion_state = d0_opinion_state(ctx, origin)
    as_of_date = opinion_state.as_of
    horizon_days = max(1, (target - as_of_date).days)
    state_seed = derive_opinion_state_seed(base_seed=base_seed, origin_date=as_of_date)
    dyn_seed = derive_shared_dynamics_seed(base_seed=base_seed, origin_date=as_of_date, horizon_days=horizon_days)
    state_samples = opinion_state.sample(n=n, seed=state_seed)
    state_matrix = np.array([[s[c] for c in ALL_CATEGORIES] for s in state_samples], dtype=np.float64)
    state_fractions = state_matrix / np.sum(state_matrix, axis=1, keepdims=True)
    log_state = np.log(state_fractions)
    state_clr = log_state - np.mean(log_state, axis=1, keepdims=True)
    eligible, _ = d0_eligible(ctx, as_of_date, horizon_days)
    sym_deltas = sample_shared_symmetric_dynamics(eligible_transitions=eligible, samples_count=n, seed=dyn_seed)
    base_comp = clr_to_composition_matrix(state_clr + sym_deltas)
    return simplex_map(base_comp[:, :K])


# --------------------------------------------------------------------------- definability and first scored origin

def origin_defined(ctx: dict, origin: date) -> dict:
    """Which §3.1 quantities exist at the origin (horizon-independent: 7 days is every fallback's floor)."""

    out = {}
    try:
        d0_opinion_state(ctx, origin)
        out["opinion_state"] = True
    except ValueError:
        out["opinion_state"] = False
    out["d0_transitions"] = len(filter_transitions_as_of(_d0_transitions(ctx, 7), origin)) >= MIN_TRANSITIONS
    out["uc_transitions"] = uc_pool(ctx, origin, 7) is not None
    out["omega"] = omega_at(ctx, origin) is not None
    out["theta"] = theta_at(ctx, origin) is not None
    out["all"] = all(out.values())
    return out


def first_scored_origin(ctx: dict) -> date:
    if ctx.get("first_scored_origin") is None:
        for origin in grid_origins():
            if origin_defined(ctx, origin)["all"]:
                ctx["first_scored_origin"] = origin
                break
        else:
            raise RuntimeError("no grid origin has every T2 quantity defined")
    return ctx["first_scored_origin"]


def scored_origins(ctx: dict) -> list[date]:
    """Grid origins on or after the first scored origin at which everything is defined."""

    if "scored_origins" not in ctx:
        first = first_scored_origin(ctx)
        ctx["scored_origins"] = [o for o in grid_origins() if o >= first and origin_defined(ctx, o)["all"]]
    return ctx["scored_origins"]


# --------------------------------------------------------------------------- draws

def state_dynamics_draws(arm: str, ctx: dict, origin: date, target: date, *, z_state: np.ndarray,
                         z_dyn: np.ndarray, boot_rng: np.random.Generator | None,
                         d0_base_seed: int | None, k: float | None = None) -> np.ndarray:
    """(n, 9) compositions (pp) after the §3.4 map: columns M, L, C, KD, S, V, MP, SD, REST. No Ω."""

    n = len(z_state)
    if target < origin:
        raise ValueError("target precedes origin")
    if arm == "D0":
        if d0_base_seed is None:
            raise ValueError("D0 needs d0_base_seed")
        return _d0_draws(ctx, origin, target, n, d0_base_seed)
    if arm not in U_ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if k is None:
        k = inflation_k(arm, ctx, origin)[0]
    st = origin_state(ctx, origin)
    z_state = np.asarray(z_state, dtype=float)
    state = st["mean"] + k * (z_state @ st["chol_state"].T)
    days = (target - origin).days
    if arm == "Uc":
        if boot_rng is None:
            raise ValueError("Uc needs boot_rng")
        pool = uc_pool(ctx, origin, days)
        if pool is None:
            raise ValueError(f"Uc has fewer than {MIN_TRANSITIONS} transitions at {origin}")
        deltas, _, scale = pool
        mapped = simplex_map(state)
        log_state = np.log(mapped)
        clr = log_state - log_state.mean(axis=1, keepdims=True)
        # Production's symmetric bootstrap order: indices, then signs.
        idx = boot_rng.integers(0, len(deltas), size=n)
        signs = boot_rng.choice([-1.0, 1.0], size=(n, 1))
        # PREREG-AMBIGUITY: §3.4 maps Uc's state draw, adds the CLR delta and
        # "converts back to a composition"; it does not say to re-apply the map
        # afterwards. Reading implemented: the inverse CLR is the final step.
        return clr_to_composition_matrix(clr + signs * deltas[idx] * scale)
    z_dyn = np.asarray(z_dyn, dtype=float)
    if arm == "Ua":
        dyn = z_dyn @ _factor(ua_dynamics_cov(ctx, origin, target)).T
        return simplex_map(state + dyn)
    theta = theta_at(ctx, origin)
    if theta is None:
        raise ValueError(f"θ undefined at {origin}")
    dyn = z_dyn @ _factor(ub_dynamics_cov(ctx, origin, target, theta)).T
    if arm == "Ub1":
        return simplex_map(state + dyn)
    # PREREG-AMBIGUITY: Ub2's mean is "A + (x − A)e^{−θ·days}" while its state
    # uncertainty is "as Ua". Reading implemented: x is the v0.3 reading at the
    # origin (the arm's mean is replaced), and the k-scaled state deviation is
    # added undamped, as in Ua; the OU decay is applied to the mean only.
    a = st["anchor"]
    shift = (a + (st["mean"] - a) * math.exp(-theta * days)) - st["mean"]
    return simplex_map(state + shift + dyn)


# --------------------------------------------------------------------------- §3.5 k

def cell_normals(origin: date, h: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.random.Generator]:
    """§3.6 draw order: one default_rng(seed) draws Z_state, Z_dyn, Z_poll (n × 8); then Uc's bootstrap."""

    rng = np.random.default_rng(t2_seed(origin, h))
    z_state = rng.standard_normal((n, K))
    z_dyn = rng.standard_normal((n, K))
    z_poll = rng.standard_normal((n, K))
    return z_state, z_dyn, z_poll, rng


def k1_cell(ctx: dict, origin: date, h: int) -> dict:
    """k = 1 predictive draws (state + dynamics + Ω) of every U arm at (origin, h): mean, variance, S, Ω."""

    n = ctx["n_draws"]
    key = (origin, h, n)
    cache = ctx["cache"]["k1"]
    if key in cache:
        return cache[key]
    target = origin + timedelta(days=h)
    z_state, z_dyn, z_poll, rng = cell_normals(origin, h, n)
    poll = poll_term(ctx, origin, z_poll)
    out = {"S": np.diag(origin_state(ctx, origin)["cov"]).copy(), "Omega": np.diag(omega_at(ctx, origin)[0]).copy()}
    for arm in U_ARMS:
        draws = state_dynamics_draws(arm, ctx, origin, target, z_state=z_state, z_dyn=z_dyn,
                                     boot_rng=rng if arm == "Uc" else None, d0_base_seed=None, k=1.0)
        pred = draws[:, :K] + poll
        # PREREG-AMBIGUITY: "k = 1 predictive draws" — §3.3 adds Ω to every
        # arm's predictive draws, so the Ω draw (Z_poll of the same cell) is
        # included, consistent with D_pp = Var − S − Ω. Var_p is the population
        # variance of the n draws (ddof = 0; the spec names no divisor).
        out[arm] = (pred.mean(axis=0), pred.var(axis=0))
    cache[key] = out
    return out


def k_eligible_cells(ctx: dict, origin: date) -> list[tuple[date, int]]:
    """Past (origin′, h ∈ {7, 28}) targets with D′ + 7 ≤ origin and every window poll published by the origin."""

    # PREREG-AMBIGUITY: §3.5 says "past (origin′, h) targets". Reading
    # implemented: origin′ ranges over the scored T2 grid origins (on or after
    # the first scored origin), where every k = 1 term (S, Ω, θ, the Uc pool)
    # is defined; earlier, unscored grid dates contribute no k targets.
    out = []
    for o in scored_origins(ctx):
        if o >= origin:
            break
        for h in K_HORIZONS:
            d = o + timedelta(days=h)
            if d + timedelta(days=WINDOW_HALF) <= origin and target_fully_known(ctx, d, origin):
                out.append((o, h))
    return out


def inflation_k(arm: str, ctx: dict, origin: date) -> tuple[float, int]:
    """§3.5: (k, number of eligible targets). k = 1 until 20 eligible targets exist."""

    if arm not in U_ARMS:
        raise ValueError(f"k is defined for {U_ARMS}, not {arm!r}")
    key = (arm, origin, ctx["n_draws"])
    cache = ctx["cache"]["k"]
    if key in cache:
        return cache[key]
    cells = k_eligible_cells(ctx, origin)
    if len(cells) < MIN_K_TARGETS:
        out = (1.0, len(cells))
    else:
        e2, s, d, om = [], [], [], []
        for o, h in cells:
            cell = k1_cell(ctx, o, h)
            mean, var = cell[arm]
            e2.append((poll_target(ctx, o + timedelta(days=h)) - mean) ** 2)
            s.append(cell["S"])
            om.append(cell["Omega"])
            d.append(np.maximum(0.0, var - cell["S"] - cell["Omega"]))
        out = (solve_k(np.array(e2), np.array(s), np.array(d), np.array(om)), len(cells))
    cache[key] = out
    return out
