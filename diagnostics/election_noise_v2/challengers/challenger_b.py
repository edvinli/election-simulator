"""CHALLENGER B - Ledoit-Wolf-regularized joint Gaussian residual model (§C).

For a centered training pool ``C`` (K x 9, percentage points, rows zero-sum), with
the normalized Frobenius norm ``‖A‖² := tr(A Aᵀ)/8`` (8 = rank of the zero-sum
subspace)::

    S_P   = Cᵀ C / K                          (divisor K, maximum likelihood)
    P₉    = I − 𝟙𝟙ᵀ/9                          (exact zero-sum projector)
    τ²    = tr(S_P) / 8
    T     = τ² P₉                             (isotropic target on the zero-sum subspace)
    d²    = ‖S_P − T‖²
    b̄²    = (1/K²) Σ_j ‖c_j c_jᵀ − S_P‖²
    b²    = min(b̄², d²)
    δ     = b² / d²      (δ := 1 if d² = 0)
    Σ_LW  = δ T + (1−δ) S_P
    Σ̃     = (K/(K−1)) Σ_LW                     (single Bessel correction, at the end)

    R ~ N(0, Σ̃)

Binding conventions, none of which may be silently changed:

* percentage-point space, 9 categories, ``ALL_CATEGORIES`` order;
* ``S_P`` uses divisor ``K`` and ``b̄²`` the ``1/K²`` prefactor, the convention
  Ledoit-Wolf (2004) is stated in;
* the Bessel correction ``K/(K−1)`` is applied **exactly once, at the very end**,
  to the already-shrunk ``Σ_LW`` - never inside ``τ²``, ``d²``, ``b̄²`` or ``δ``;
* Gaussian only. No Student-t, no added ridge, no empirical tail multiplier, no
  recency weighting, and **zero tunable hyperparameters** - ``δ`` is a closed-form
  function of the pool.

The ``1/8`` normalization cancels in ``δ = b²/d²``, so ``δ`` is invariant to that
choice; a test asserts this against the unnormalized Frobenius convention.

Structural consequences, asserted by tests: ``S_P 𝟙 = 0`` because every ``c_j`` is
zero-sum, hence ``T 𝟙 = 0`` and ``Σ̃ 𝟙 = 0``; the symmetric square root shares
eigenvectors with ``Σ̃`` so ``Σ̃^{1/2} 𝟙 = 0`` and **every draw is zero-sum almost
surely**; and ``δ ∈ [0, 1]`` since ``b² = min(b̄², d²) ≤ d²``.

Numerical policy - deliberately strict
--------------------------------------
``Σ̃`` is PSD by construction (a convex combination of the PSD matrices ``T`` and
``S_P``, times a positive scalar) and singular, since ``𝟙`` is exactly in its null
space. Cholesky is therefore unavailable and a symmetric eigendecomposition is
used. The model covariance is **never** altered to make factorization succeed:

* a materially negative eigenvalue raises :class:`NonPSDCovariance` rather than
  being clipped - the preregistration forbids silently modifying ``Σ̃``;
* only round-off-level eigenvalues, ``|λ| <= tol`` with ``tol`` a relative
  multiple of the spectral scale, are set to exactly ``0``. These are the
  structural zeros of the zero-sum null space, whose exact value is known to be 0
  analytically; this is floating-point cleanup, not shrinkage or regularization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Dimension of the composition space and the rank of its zero-sum subspace.
N_CATEGORIES: int = 9
ZERO_SUM_RANK: int = 8

#: Relative tolerance separating a structural zero eigenvalue from a real negative one.
EIGENVALUE_TOLERANCE: float = 1e-10

#: Smallest pool B may be fitted on: the Bessel factor K/(K-1) is undefined at K = 1.
MIN_POOL_SIZE: int = 2


class NonPSDCovariance(RuntimeError):
    """Raised when Sigma_tilde is materially indefinite.

    This is a hard stop by design. The preregistration forbids eigenvalue clipping
    or any other silent alteration of the model covariance, so an indefinite matrix
    is reported rather than repaired.
    """


def zero_sum_projector(d: int = N_CATEGORIES) -> np.ndarray:
    """``P₉ = I − 𝟙𝟙ᵀ/9`` - the exact zero-sum projector."""
    return np.eye(d) - np.ones((d, d)) / d


def _normalized_frobenius_sq(a: np.ndarray) -> float:
    """``‖A‖² = tr(A Aᵀ) / 8``, the preregistered normalization."""
    return float(np.trace(a @ a.T)) / ZERO_SUM_RANK


@dataclass(frozen=True)
class ChallengerBFit:
    """Challenger B fitted on one centered pool. No tunable hyperparameter exists."""

    centered: np.ndarray
    s_p: np.ndarray          # (9, 9) pool covariance, divisor K
    p9: np.ndarray           # (9, 9) zero-sum projector
    tau_sq: float
    target: np.ndarray       # T = tau^2 * P9
    d_sq: float
    bbar_sq: float
    b_sq: float
    delta: float
    sigma_lw: np.ndarray     # delta*T + (1-delta)*S_P, before Bessel
    sigma_tilde: np.ndarray  # (K/(K-1)) * sigma_lw

    @property
    def k(self) -> int:
        return int(self.centered.shape[0])

    @property
    def bessel_factor(self) -> float:
        return self.k / (self.k - 1)


def fit_challenger_b(centered: np.ndarray) -> ChallengerBFit:
    """Closed-form fit. Every quantity follows the preregistered formula exactly."""
    c = np.asarray(centered, dtype=float)
    if c.ndim != 2 or c.shape[1] != N_CATEGORIES:
        raise ValueError(f"centered pool must be (K, {N_CATEGORIES}), got {c.shape}")
    k = c.shape[0]
    if k < MIN_POOL_SIZE:
        raise ValueError(
            f"Challenger B requires K >= {MIN_POOL_SIZE}; the Bessel factor K/(K-1) "
            f"is undefined at K = {k}."
        )

    s_p = c.T @ c / k
    p9 = zero_sum_projector(N_CATEGORIES)
    tau_sq = float(np.trace(s_p)) / ZERO_SUM_RANK
    target = tau_sq * p9

    d_sq = _normalized_frobenius_sq(s_p - target)
    bbar_sq = sum(_normalized_frobenius_sq(np.outer(cj, cj) - s_p) for cj in c) / (k * k)
    b_sq = min(bbar_sq, d_sq)

    # Preregistered limit: delta := 1 if d^2 = 0. At d^2 = 0 the sample covariance
    # already equals the shrinkage target, so delta = 1 and delta = 0 give the same
    # matrix; delta = 1 is the value the frozen document fixes.
    delta = 1.0 if d_sq == 0.0 else b_sq / d_sq

    sigma_lw = delta * target + (1.0 - delta) * s_p
    sigma_tilde = (k / (k - 1)) * sigma_lw   # Bessel correction, applied once, at the end

    return ChallengerBFit(
        centered=c, s_p=s_p, p9=p9, tau_sq=tau_sq, target=target,
        d_sq=d_sq, bbar_sq=bbar_sq, b_sq=b_sq, delta=delta,
        sigma_lw=sigma_lw, sigma_tilde=sigma_tilde,
    )


def symmetric_factor(sigma: np.ndarray, tol: float = EIGENVALUE_TOLERANCE) -> np.ndarray:
    """Symmetric square-root factor ``L`` with ``L Lᵀ = Σ̃``, without altering ``Σ̃``.

    Raises :class:`NonPSDCovariance` on a materially negative eigenvalue instead of
    clipping it. Only round-off-level eigenvalues are set to exactly zero.
    """
    asym = float(np.max(np.abs(sigma - sigma.T)))
    scale = float(np.max(np.abs(sigma))) or 1.0
    if asym > tol * scale:
        raise NonPSDCovariance(f"covariance is not symmetric: max|Σ − Σᵀ| = {asym:.3e}")

    w, v = np.linalg.eigh(sigma)
    cut = tol * max(float(np.max(np.abs(w))), 1e-300)
    if float(np.min(w)) < -cut:
        raise NonPSDCovariance(
            f"Sigma_tilde has a materially negative eigenvalue {float(np.min(w)):.6e} "
            f"(tolerance {-cut:.3e}). The preregistration forbids eigenvalue clipping; "
            "this is reported rather than repaired."
        )
    w_clean = np.where(w > cut, w, 0.0)   # structural zeros of the zero-sum null space
    return v * np.sqrt(w_clean)


def draw_challenger_b(
    fit: ChallengerBFit, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw ``n`` residual vectors ``R ~ N(0, Σ̃)``. Rows are zero-sum a.s."""
    factor = symmetric_factor(fit.sigma_tilde)
    z = rng.standard_normal((n, N_CATEGORIES))
    return z @ factor.T


def zero_sum_rank(sigma: np.ndarray, tol: float = EIGENVALUE_TOLERANCE) -> int:
    """Number of eigenvalues materially above zero - 8 when delta > 0 and tau² > 0."""
    w = np.linalg.eigvalsh(sigma)
    return int(np.sum(w > tol * max(float(np.max(np.abs(w))), 1e-300)))
