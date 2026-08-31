"""Production ElectionNoise: the adopted regularized joint Gaussian residual law.

ADOPTION RECORD. The former production ElectionNoise (``pp_centered_noise``) drew
uniformly from a small pool of centered historical final-poll-to-election residual
vectors — a discrete law on K = 3/4/5 atoms. A preregistered historical comparison
over 2014, 2018 and 2022 selected this Ledoit-Wolf-regularized joint Gaussian law
("Challenger B") on proper scoring rules, not on any property of a prospective
forecast. See ``docs/election_noise_v2_preregistration.md`` and
``diagnostics/election_noise_v2/competition/RESULTS.md``.

This module is a **production re-implementation** of the frozen research
implementation in
``diagnostics/election_noise_v2/challengers/challenger_b.py``. It deliberately does
not import it: the two are independent expressions of the same frozen law, and
``tests/test_production_challenger_b.py`` asserts they agree bit-for-bit on the
covariance construction and on the generated draws. Neither the research
implementation nor any file in the evaluator or challenger freeze is modified.

The law, for a centered training pool ``C`` (K x 9, percentage points, rows
zero-sum)::

    S_P   = Cᵀ C / K                          divisor K, maximum likelihood
    P₉    = I − 𝟙𝟙ᵀ/9                          exact zero-sum projector
    τ²    = tr(S_P) / 8
    T     = τ² P₉
    d²    = ‖S_P − T‖²                        ‖A‖² := tr(A Aᵀ)/8
    b̄²    = (1/K²) Σ_j ‖c_j c_jᵀ − S_P‖²
    b²    = min(b̄², d²)
    δ     = b²/d²,  and δ := 1 if d² = 0
    Σ̃     = (K/(K−1))·[δT + (1−δ)S_P]          Bessel correction, once, at the end
    R     ~ N(0, Σ̃)

Zero tunable hyperparameters. Gaussian only — no Student-t, no added ridge, no tail
multiplier, no recency or residual-year weighting, and no 2026-specific adjustment.
``Σ̃𝟙 = 0``, so every draw is zero-sum almost surely and the downstream simplex
transfer is entered exactly as the discrete law entered it.

Numerical policy, unchanged from the frozen implementation: ``Σ̃`` is PSD by
construction and singular, so a symmetric eigendecomposition is used rather than
Cholesky. A materially negative eigenvalue raises rather than being clipped; only
round-off-level structural zeros are set to exactly zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib

import numpy as np

#: Model identifier for this law in production artifacts and diagnostics.
MODEL_ID = "pp_lw_gaussian"

#: The superseded production law, retained and still reproducible.
LEGACY_MODEL_ID = "pp_centered_noise"

#: Version identifier for a release carrying this law, following the existing
#: convention in ``scripts/simulator/config.py`` (semver + ``-rcN``, with the release
#: tag mirroring it). The ElectionNoise layer changed, which is a model change rather
#: than a fix, so the minor version advances and the candidate letter follows the
#: adopted challenger:
#:
#:     1.0.0-rc1 / candidate "A"  ->  1.1.0-rc1 / candidate "B"
#:
#: These constants are declarative here. They take effect in published artifacts only
#: when ``scripts/simulator/config.py::MODEL_VERSION`` is advanced as part of the
#: default flip, which also re-issues the evaluator and challenger freezes.
ADOPTED_MODEL_VERSION = "1.1.0-rc1"
ADOPTED_RELEASE_TAG = "election-simulator-v1.1-rc1"

#: The adopted challenger in the ELECTIONNOISE COMPETITION namespace.
#:
#: This is NOT the artifact's ``candidate`` field. That field lives in the
#: botten-ada benchmark / model-lineage namespace, where "Candidate A" is this
#: simulator as a whole and "Candidate B" would be a rival external model. The two
#: namespaces collide on the letter B by coincidence and must never be merged; see
#: ``docs/election_simulator_rc1.md`` for the benchmark meaning.
ADOPTED_ELECTION_NOISE_CANDIDATE = "B"

#: ElectionNoise law -> its name in the ElectionNoise competition namespace.
#:
#: The superseded law maps to "CONTROL", not "A". ElectionNoise Challenger A was the
#: variance-corrected smoothed empirical bootstrap, which was evaluated and not
#: adopted; it is a different model from the CONTROL empirical bootstrap, and
#: labelling the legacy law "A" would be false.
ELECTION_NOISE_CANDIDATE_BY_LAW = {
    MODEL_ID: ADOPTED_ELECTION_NOISE_CANDIDATE,
    LEGACY_MODEL_ID: "CONTROL",
}


def election_noise_candidate_for_law(law: str | None) -> str | None:
    """Competition-namespace name for an ElectionNoise law, or None if unknown."""
    if law is None:
        return None
    return ELECTION_NOISE_CANDIDATE_BY_LAW.get(law)

N_CATEGORIES = 9
ZERO_SUM_RANK = 8
EIGENVALUE_TOLERANCE = 1e-10
MIN_POOL_SIZE = 2

#: Reserved seed token for this law, fixed by the preregistration's RNG contract.
SEED_TOKEN = "election_noise_v2_b_normal"
_MODULUS = 2_147_483_647


class NonPSDCovariance(RuntimeError):
    """Raised when the covariance is materially indefinite. Never repaired silently."""


def derive_election_noise_b_seed(base_seed: int, origin_date: date, horizon_days: int) -> int:
    """Production sub-seed, using the unchanged SHA-256 token convention."""
    token = f"{base_seed}:{origin_date.isoformat()}:{horizon_days}:{SEED_TOKEN}".encode("utf-8")
    return int(hashlib.sha256(token).hexdigest()[:8], 16) % _MODULUS


def zero_sum_projector(d: int = N_CATEGORIES) -> np.ndarray:
    return np.eye(d) - np.ones((d, d)) / d


def _norm_sq(a: np.ndarray) -> float:
    return float(np.trace(a @ a.T)) / ZERO_SUM_RANK


@dataclass(frozen=True)
class ElectionNoiseBFit:
    centered: np.ndarray
    s_p: np.ndarray
    p9: np.ndarray
    tau_sq: float
    target: np.ndarray
    d_sq: float
    bbar_sq: float
    b_sq: float
    delta: float
    sigma_lw: np.ndarray
    sigma_tilde: np.ndarray

    @property
    def k(self) -> int:
        return int(self.centered.shape[0])

    @property
    def bessel_factor(self) -> float:
        return self.k / (self.k - 1)


def fit_election_noise_b(centered: np.ndarray) -> ElectionNoiseBFit:
    """Closed-form fit. No hyperparameter is accepted, because none exists."""
    c = np.asarray(centered, dtype=float)
    if c.ndim != 2 or c.shape[1] != N_CATEGORIES:
        raise ValueError(f"centered pool must be (K, {N_CATEGORIES}), got {c.shape}")
    k = c.shape[0]
    if k < MIN_POOL_SIZE:
        raise ValueError(f"requires K >= {MIN_POOL_SIZE}; the Bessel factor is undefined at K={k}")

    s_p = c.T @ c / k
    p9 = zero_sum_projector(N_CATEGORIES)
    tau_sq = float(np.trace(s_p)) / ZERO_SUM_RANK
    target = tau_sq * p9

    d_sq = _norm_sq(s_p - target)
    bbar_sq = sum(_norm_sq(np.outer(cj, cj) - s_p) for cj in c) / (k * k)
    b_sq = min(bbar_sq, d_sq)
    delta = 1.0 if d_sq == 0.0 else b_sq / d_sq

    sigma_lw = delta * target + (1.0 - delta) * s_p
    sigma_tilde = (k / (k - 1)) * sigma_lw

    return ElectionNoiseBFit(centered=c, s_p=s_p, p9=p9, tau_sq=tau_sq, target=target,
                             d_sq=d_sq, bbar_sq=bbar_sq, b_sq=b_sq, delta=delta,
                             sigma_lw=sigma_lw, sigma_tilde=sigma_tilde)


def symmetric_factor(sigma: np.ndarray, tol: float = EIGENVALUE_TOLERANCE) -> np.ndarray:
    """Symmetric factor ``L`` with ``L Lᵀ = Σ̃``, without altering ``Σ̃``."""
    asym = float(np.max(np.abs(sigma - sigma.T)))
    scale = float(np.max(np.abs(sigma))) or 1.0
    if asym > tol * scale:
        raise NonPSDCovariance(f"covariance is not symmetric: max|Σ − Σᵀ| = {asym:.3e}")
    w, v = np.linalg.eigh(sigma)
    cut = tol * max(float(np.max(np.abs(w))), 1e-300)
    if float(np.min(w)) < -cut:
        raise NonPSDCovariance(
            f"materially negative eigenvalue {float(np.min(w)):.6e}; eigenvalue clipping is "
            "forbidden, so this is reported rather than repaired")
    return v * np.sqrt(np.where(w > cut, w, 0.0))


def draw_election_noise_b(fit: ElectionNoiseBFit, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` percentage-point residual vectors ``R ~ N(0, Σ̃)``; rows zero-sum a.s."""
    return rng.standard_normal((n, N_CATEGORIES)) @ symmetric_factor(fit.sigma_tilde).T


def election_noise_b_residuals(
    centered: np.ndarray, n: int, base_seed: int, origin_date: date, horizon_days: int
) -> tuple[np.ndarray, ElectionNoiseBFit, int]:
    """Fit and draw in one deterministic step. Returns ``(R, fit, sub_seed)``."""
    fit = fit_election_noise_b(centered)
    sub = derive_election_noise_b_seed(base_seed, origin_date, horizon_days)
    return draw_election_noise_b(fit, n, np.random.default_rng(sub)), fit, sub
