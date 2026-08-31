"""CHALLENGER A - variance-corrected smoothed empirical bootstrap (preregistration §C).

Law, for a centered training pool ``C`` (K x 9, percentage points, rows zero-sum)::

    S_P   = Cᵀ C / K                      (divisor K, maximum likelihood, NO Bessel)
    k     ~ Uniform({1, …, K})
    z_j   ~ iid N(0, 1),  j = 1 … K
    ε     = (1/√K) Σ_j z_j c_j            so  ε ~ N(0, S_P)
    R     = (c_k + h ε) / √(1 + h²)

Moments, exact:

* ``E[R] = 0`` because Σ_j c_j = 0, so the atom draw has mean zero and ε has mean zero.
* ``Cov(R) = (Cov(c_k) + h² Cov(ε)) / (1 + h²) = (S_P + h² S_P) / (1 + h²) = S_P``.

The ``√(1 + h²)`` denominator is therefore **binding**: without it the law would
inflate the pool covariance by ``1 + h²``. A dedicated test asserts that removing it
breaks the covariance identity.

The divisor ``K`` is binding too (§C, "Covariance convention"): it is what makes
``Cov(R) = S_P`` hold exactly and makes A nest CONTROL as ``h → 0``. ``h = 0`` is
deliberately excluded from the grid because CONTROL already *is* the unsmoothed
empirical model.

Support, disclosed in the preregistration and not a defect: ``ε ∈ span{c_1,…,c_K}``,
so A is continuous but singular - supported on a ``(K−1)``-dimensional affine
subspace of the 8-dimensional zero-sum hyperplane. It cannot produce an error
pattern outside the historical span. That is the deliberate scientific contrast
with Challenger B.

Exactly one free parameter, ``h``, chosen from the frozen grid by LOEO-FIT inside
the outer training pool (see ``loeo.py``). Nothing else here is tunable: there is no
h = 0, no interpolation, no continuous optimisation, no party-specific bandwidth, no
recency or residual-year weighting, no Student-t component and no covariance
shrinkage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Preregistered bandwidth grid (§C). Frozen: no value may be added or removed.
FROZEN_H_GRID: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)

#: Smallest pool Challenger A may be fitted on. K = 1 carries no covariance and no
#: meaningful smoothing; the preregistration prohibits it (K_inner = 1 prohibited).
MIN_POOL_SIZE: int = 2


class PoolTooSmall(ValueError):
    """Raised when a Challenger A fit is attempted at K < 2 (e.g. K_inner = 1)."""


def validate_bandwidth(h: float) -> float:
    """Accept only a value on the frozen grid, compared exactly."""
    for g in FROZEN_H_GRID:
        if h == g:
            return float(g)
    raise ValueError(
        f"h={h!r} is not on the frozen grid {list(FROZEN_H_GRID)}; the grid may not be "
        "extended, interpolated or continuously optimised (preregistration §C)"
    )


def pool_covariance(centered: np.ndarray) -> np.ndarray:
    """``S_P = Cᵀ C / K`` - divisor K, no Bessel correction (binding convention)."""
    c = np.asarray(centered, dtype=float)
    if c.ndim != 2:
        raise ValueError(f"centered residual pool must be 2-D, got shape {c.shape}")
    return c.T @ c / c.shape[0]


@dataclass(frozen=True)
class ChallengerAFit:
    """Challenger A fitted on one centered pool at one frozen bandwidth."""

    centered: np.ndarray      # (K, 9) centered residual pool
    h: float                  # bandwidth, on the frozen grid
    s_p: np.ndarray           # (9, 9) pool covariance, divisor K

    @property
    def k(self) -> int:
        return int(self.centered.shape[0])

    @property
    def theoretical_mean(self) -> np.ndarray:
        """``E[R] = 0`` exactly."""
        return np.zeros(self.centered.shape[1], dtype=float)

    @property
    def theoretical_covariance(self) -> np.ndarray:
        """``Cov(R) = S_P`` exactly, for every h on the grid."""
        return self.s_p


def fit_challenger_a(centered: np.ndarray, h: float) -> ChallengerAFit:
    """Fit A on a centered pool at a frozen bandwidth."""
    c = np.asarray(centered, dtype=float)
    if c.ndim != 2:
        raise ValueError(f"centered residual pool must be 2-D, got shape {c.shape}")
    if c.shape[0] < MIN_POOL_SIZE:
        raise PoolTooSmall(
            f"Challenger A requires K >= {MIN_POOL_SIZE}; got K = {c.shape[0]}. "
            "K_inner = 1 is prohibited by the preregistration (§C, §E.4)."
        )
    return ChallengerAFit(centered=c, h=validate_bandwidth(h), s_p=pool_covariance(c))


def draw_challenger_a(
    fit: ChallengerAFit,
    n: int,
    index_rng: np.random.Generator,
    kernel_rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n`` residual vectors from Challenger A.

    The two stochastic pieces come from separate generators so each is
    independently reproducible: ``index_rng`` supplies the discrete atom index and
    ``kernel_rng`` the Gaussian smoothing draw.

    Returns ``(R, atom_index)`` with ``R`` of shape ``(n, 9)``.
    """
    c, h, k = fit.centered, fit.h, fit.k
    idx = index_rng.integers(0, k, size=n)
    z = kernel_rng.standard_normal((n, k))
    epsilon = (z @ c) / np.sqrt(k)                    # ε ~ N(0, S_P)
    r = (c[idx] + h * epsilon) / np.sqrt(1.0 + h * h)  # variance correction, binding
    return r, idx
