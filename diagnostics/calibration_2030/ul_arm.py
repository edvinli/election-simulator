"""T5 amendment 001: the shadow arm ``Ul`` (log-ratio drift).

``Ul`` takes ``Ua``'s pre-map draw ``y = m + k·state + dyn`` (built exactly as
``t2_arms.state_dynamics_draws`` builds ``Ua``'s, with the same ``z_state`` and
``z_dyn``) and applies its total deviation multiplicatively:

    ε = y − m,  ε_REST = −Σ ε,  π = (m, 100 − Σ m),
    x_i = π_i · exp(ε_i / max(π_i, 0.5)),  renormalized to 100.

``Ul`` has its own k by the final round's §3.5 procedure (``t2_arms.solve_k``
on its own k = 1 predictive draws). It consumes no random numbers of its own.
This module reads ``t2_arms`` and never edits it.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from . import t2_arms as A

ARM = "Ul"
FLOOR_PP = 0.5


def log_ratio_map(y8: np.ndarray, m8: np.ndarray) -> np.ndarray:
    """(n, 9) compositions (pp): Ul's multiplicative map of Ua's pre-map draws ``y8`` around reading ``m8``."""

    y8 = np.atleast_2d(np.asarray(y8, dtype=float))
    m8 = np.asarray(m8, dtype=float)
    eps = y8 - m8
    eps9 = np.concatenate([eps, -eps.sum(axis=1, keepdims=True)], axis=1)
    pi = np.concatenate([m8, [100.0 - m8.sum()]])
    x = pi * np.exp(eps9 / np.maximum(pi, FLOOR_PP))
    return 100.0 * x / x.sum(axis=1, keepdims=True)


def ua_premap(ctx: dict, origin: date, target: date, *, z_state: np.ndarray, z_dyn: np.ndarray,
              k: float) -> tuple[np.ndarray, np.ndarray]:
    """(Ua's pre-map draws y (n, 8), the v0.3 reading m (8,)), exactly as ``state_dynamics_draws('Ua')``."""

    if target < origin:
        raise ValueError("target precedes origin")
    st = A.origin_state(ctx, origin)
    state = st["mean"] + k * (np.asarray(z_state, dtype=float) @ st["chol_state"].T)
    dyn = np.asarray(z_dyn, dtype=float) @ A._factor(A.ua_dynamics_cov(ctx, origin, target)).T
    return state + dyn, st["mean"]


def ul_draws(ctx: dict, origin: date, target: date, *, z_state: np.ndarray, z_dyn: np.ndarray,
             k: float) -> np.ndarray:
    """(n, 9) Ul compositions (pp): columns M, L, C, KD, S, V, MP, SD, REST. No Ω."""

    y, m = ua_premap(ctx, origin, target, z_state=z_state, z_dyn=z_dyn, k=k)
    return log_ratio_map(y, m)


def ul_k1_cell(ctx: dict, origin: date, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Ul's k = 1 predictive (state + dynamics + Ω) mean and variance at (origin, h), as ``A.k1_cell``."""

    n = ctx["n_draws"]
    cache = ctx["cache"].setdefault("k1_ul", {})
    key = (origin, h, n)
    if key in cache:
        return cache[key]
    target = origin + timedelta(days=h)
    z_state, z_dyn, z_poll, _ = A.cell_normals(origin, h, n)
    poll = A.poll_term(ctx, origin, z_poll)
    pred = ul_draws(ctx, origin, target, z_state=z_state, z_dyn=z_dyn, k=1.0)[:, :A.K] + poll
    out = (pred.mean(axis=0), pred.var(axis=0))
    cache[key] = out
    return out


def ul_inflation_k(ctx: dict, origin: date) -> tuple[float, int]:
    """§3.5 for Ul: (k, number of eligible targets); k = 1 until 20 eligible targets exist."""

    cells = A.k_eligible_cells(ctx, origin)
    if len(cells) < A.MIN_K_TARGETS:
        return 1.0, len(cells)
    e2, s, d, om = [], [], [], []
    for o, h in cells:
        mean, var = ul_k1_cell(ctx, o, h)
        # S and Ω exactly as A.k1_cell records them (reused from its cache when filled).
        cell = A.k1_cell(ctx, o, h)
        S, omega = cell["S"], cell["Omega"]
        e2.append((A.poll_target(ctx, o + timedelta(days=h)) - mean) ** 2)
        s.append(S)
        om.append(omega)
        d.append(np.maximum(0.0, var - S - omega))
    return A.solve_k(np.array(e2), np.array(s), np.array(d), np.array(om)), len(cells)
