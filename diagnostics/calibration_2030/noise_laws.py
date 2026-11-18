"""Election-noise laws of the calibration-2030 final round (T3-rev and T4).

This implements the laws of ``docs/calibration_2030_final_round_preregistration.md``
§4.2-§4.4 (frozen, body SHA-256 5d070b4b...). It changes no production code.

Laws (``LAWS``):

* ``LW0``: production ``pp_lw_gaussian``. ``r = Z Lᵀ`` with ``L`` the symmetric
  factor of ``Σ̃ = fit_election_noise_b(centred pool)``. No mean, no scaling.
  It reproduces production's draws byte for byte for the same ``Z`` and pool.
* ``N0``: the interim law of calibration amendment 001. ``r = 0.5 · r̄ + Z Lᵀ``,
  ``r̄`` the zero-sum-cleaned pool mean, ``L`` as for ``LW0``. γ = 0. It keeps
  every pool cell, SD 2002 included (§4.3, disclosure).
* ``NH0``, ``NH0.5``, ``NH1``: the hierarchical normal-normal law with size
  scaling γ ∈ {0, 0.5, 1} (§4.3-§4.4).

Every law returns zero-sum percentage-point residual draws *before* the engine's
simplex transfer (``apply_batch_simplex_transfer``), which the caller applies.

Pools are built either with the corrected election-eve consensus
(``scripts/election_residuals/eve_consensus.py``, residual amendments 001 and
002) or with the committed rule (``scripts/election_residuals/consensus.py``,
production 1.2.0-rc1, used by ``C0`` in T4).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import Sequence

import numpy as np
import pandas as pd

from scripts.election_residuals.config import (
    ALL_CATEGORIES,
    DEFAULT_ELECTIONS_FILE,
    DEFAULT_POLLS_FILE,
    PARLIAMENTARY_PARTIES,
)
from scripts.election_residuals.consensus import build_election_polling_consensus
from scripts.election_residuals.eve_consensus import build_eve_polling_consensus
from scripts.elections.load import load_election_targets_for_forecasting
from scripts.vote_share_calibration.election_noise_b import fit_election_noise_b, symmetric_factor

from .common import ROOT

LAWS = ("LW0", "N0", "NH0", "NH0.5", "NH1")
NH_GAMMA = {"NH0": 0.0, "NH0.5": 0.5, "NH1": 1.0}
N0_BETA = 0.5
FLOOR_PP = 1.0
WINDOW_DAYS = 14
N_CAT = len(ALL_CATEGORIES)

MANIFEST_2026 = ROOT / "data" / "raw" / "elections" / "val2026" / "official_result_manifest.json"
ELECTION_DAYS = {2002: date(2002, 9, 15), 2006: date(2006, 9, 17), 2010: date(2010, 9, 19), 2014: date(2014, 9, 14),
                 2018: date(2018, 9, 9), 2022: date(2022, 9, 11), 2026: date(2026, 9, 13)}


# ------------------------------------------------------------------ pools ----
def _zero_sum_clean(v: np.ndarray) -> np.ndarray:
    """The production residual cleaning (``residuals_pool.load_chronological_pp_residuals``)."""

    s = float(np.sum(v))
    if abs(s) > 0.05:
        raise ValueError(f"residual sum {s:.6f} deviates from zero")
    return v - s / len(v) if abs(s) > 1e-12 else v


@lru_cache(maxsize=1)
def _inputs() -> tuple[pd.DataFrame, dict]:
    polls = pd.read_csv(DEFAULT_POLLS_FILE)
    targets = load_election_targets_for_forecasting(DEFAULT_ELECTIONS_FILE)
    return polls, targets


def _truth(year: int) -> np.ndarray:
    if year == 2026:
        manifest = json.loads(MANIFEST_2026.read_text(encoding="utf-8"))
        t = {p: float(v["vote_share_percentage_points"]) for p, v in manifest["parties"].items()}
        t["REST"] = 100.0 - sum(t.values())
    else:
        t = _inputs()[1][ELECTION_DAYS[year]]
    return np.array([t[k] for k in ALL_CATEGORIES], dtype=float)


def _pool(years: Sequence[int], builder) -> dict:
    polls = _inputs()[0]
    residuals, consensus, unreported = [], [], []
    for year in years:
        cons = builder(ELECTION_DAYS[year], polls, window_days=WINDOW_DAYS)
        c = np.array([cons.consensus_composition[k] for k in ALL_CATEGORIES], dtype=float)
        reported = {p for poll in cons.contributing_polls for p in poll.party_support}
        # REST is the remainder and is never "unreported".
        unreported.append([k in PARLIAMENTARY_PARTIES and k not in reported for k in ALL_CATEGORIES])
        consensus.append(c)
        residuals.append(_zero_sum_clean(_truth(year) - c))
    return {
        "residuals": np.array(residuals, dtype=float).reshape(len(years), N_CAT),
        "consensus": np.array(consensus, dtype=float).reshape(len(years), N_CAT),
        "unreported": np.array(unreported, dtype=bool).reshape(len(years), N_CAT),
        "years": [int(y) for y in years],
    }


def corrected_pool(years: Sequence[int]) -> dict:
    """Corrected residuals (residual amendments 001 and 002), nine categories, pp."""

    return _pool(years, build_eve_polling_consensus)


def current_rule_pool(years: Sequence[int]) -> dict:
    """The same pool built with the committed production rule (``consensus.py``), for ``C0``."""

    return _pool(years, build_election_polling_consensus)


# ------------------------------------------------------------- size maps ----
def size_scale(c: np.ndarray, gamma: float) -> np.ndarray:
    """``max(c, 1.0 pp) ** γ`` elementwise (§4.3)."""

    return np.maximum(np.asarray(c, dtype=float), FLOOR_PP) ** gamma


def standardize(r: np.ndarray, c: np.ndarray, gamma: float, unreported: np.ndarray | None = None) -> np.ndarray:
    """§4.3 steps 1-4 for pool rows: ``u = r / c^γ``, unreported cells to 0, then zero-sum projection."""

    u = np.asarray(r, dtype=float) / size_scale(c, gamma)
    if unreported is not None:
        u = np.where(unreported, 0.0, u)
    return u - np.mean(u, axis=-1, keepdims=True)


def rescale(u: np.ndarray, c: np.ndarray, gamma: float) -> np.ndarray:
    """§4.3 target steps 2-3: ``r = ũ ⊙ c^γ``, then subtract ``(Σr) · c^γ / Σ c^γ`` (row-wise)."""

    s = size_scale(c, gamma)
    r = u * s
    return r - np.sum(r, axis=-1, keepdims=True) * s / np.sum(s, axis=-1, keepdims=True)


def _clean_mean(pool: np.ndarray) -> np.ndarray:
    """Pool mean, zero-sum cleaned exactly as production (``load_chronological_pp_residuals``)."""

    mean = np.mean(pool, axis=0)
    s = float(np.sum(mean))
    if abs(s) > 1e-12:
        mean = mean - (s / len(mean))
    return mean


# ------------------------------------------------------------------- laws ----
@dataclass(frozen=True)
class NoiseLaw:
    name: str
    gamma: float
    beta: float
    kappa_hat: float | None
    K: int
    mean: np.ndarray            # predictive mean of the (standardized) draw: 0, 0.5 r̄ or β ū
    scale: float                # √(1 + β/K) for NH, 1 otherwise
    factor_t: np.ndarray        # symmetric_factor(Σ).T
    sigma: np.ndarray           # Σ̃ (LW0, N0) or Σ̂ in u-space (NH)
    fit_info: dict = field(default_factory=dict)

    @property
    def hierarchical(self) -> bool:
        return self.name in NH_GAMMA

    def draw(self, z: np.ndarray, base: np.ndarray | None = None) -> np.ndarray:
        """Zero-sum pp residual draws for standard normals ``z`` (n, 9); ``base`` is c_T per draw."""

        z = np.asarray(z, dtype=float)
        if z.ndim != 2 or z.shape[1] != N_CAT:
            raise ValueError(f"z must be (n, {N_CAT}), got {z.shape}")
        eps = z @ self.factor_t
        if self.name == "LW0":
            return eps  # production exactly: no mean term, no scaling
        if self.name == "N0":
            return self.mean + eps  # 0.5 · r̄ + ε, as T3's gaussian β = 0.5 arm
        u = self.mean + self.scale * eps
        if self.gamma == 0.0:
            c = np.ones_like(u)  # c^0 = 1; base is ignored
        else:
            if base is None:
                raise ValueError(f"{self.name} (gamma={self.gamma}) needs a base composition")
            c = np.broadcast_to(np.asarray(base, dtype=float), u.shape)
        # PREREG-AMBIGUITY: §4.3 says "γ = 0 is the unscaled law" but lists the zero-sum
        # restoration as a step of every target map. Reading: NH arms always apply step 3; at
        # γ = 0 it subtracts the row mean, which only removes floating-point residue.
        return rescale(u, c, self.gamma)

    def summary(self) -> dict:
        return {
            "law": self.name,
            "kappa_hat": self.kappa_hat,
            "beta": self.beta,
            "K": self.K,
            "gamma": self.gamma,
            "space": "pp" if self.gamma == 0.0 else f"u = r / max(c, 1 pp)^{self.gamma:g}",
            "mean_pp": [float(x) for x in self.mean],
            "sd_pp": [float(x) for x in self.scale * np.sqrt(np.diag(self.sigma))],
            "variance_inflation": self.scale ** 2,
            **self.fit_info,
        }


def _lw(centred: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    fit = fit_election_noise_b(centred)
    info = {"lw_delta": float(fit.delta), "lw_tau_sq": float(fit.tau_sq)}
    return fit.sigma_tilde, symmetric_factor(fit.sigma_tilde).T, info


def fit_law(name: str, pool: dict) -> NoiseLaw:
    """Fit a law of ``LAWS`` on a pool from ``corrected_pool`` or ``current_rule_pool``."""

    if name not in LAWS:
        raise ValueError(f"unknown law {name!r}; expected one of {LAWS}")
    residuals = np.asarray(pool["residuals"], dtype=float)
    k = int(residuals.shape[0])
    if name in ("LW0", "N0"):
        r_bar = _clean_mean(residuals)
        sigma, factor_t, info = _lw(residuals - r_bar)
        if name == "LW0":
            return NoiseLaw(name, 0.0, 0.0, None, k, np.zeros(N_CAT), 1.0, factor_t, sigma, info)
        return NoiseLaw(name, 0.0, N0_BETA, None, k, N0_BETA * r_bar, 1.0, factor_t, sigma,
                        {**info, "pool_mean_pp": [float(x) for x in r_bar]})

    gamma = NH_GAMMA[name]
    u = standardize(residuals, pool["consensus"], gamma, np.asarray(pool["unreported"], dtype=bool))
    u_bar = _clean_mean(u)  # ū: pool mean, zero-sum cleaned
    # PREREG-AMBIGUITY: §4.4 names Σ̂ = fit_election_noise_b(u − ū) without saying which of its
    # outputs. Reading: the law's covariance, sigma_tilde (Bessel-corrected, as production and
    # N0 use it), both in κ̂'s trace and in the predictive.
    sigma, factor_t, info = _lw(u - u_bar)
    kappa_hat = max(0.0, float(np.trace(np.outer(u_bar, u_bar))) / float(np.trace(sigma)) - 1.0 / k)
    beta = k * kappa_hat / (k * kappa_hat + 1.0)
    return NoiseLaw(name, gamma, beta, kappa_hat, k, beta * u_bar, math.sqrt(1.0 + beta / k), factor_t, sigma,
                    {**info, "u_bar": [float(x) for x in u_bar],
                     "unreported_cells": [[int(pool["years"][i]), ALL_CATEGORIES[j]]
                                          for i, j in zip(*np.nonzero(np.asarray(pool["unreported"])))]})
