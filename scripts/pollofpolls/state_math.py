"""Compositional and statistical mathematical operations for Opinion State Estimator v1.

Implements additive log-ratio (ALR) transformations, sample covariance estimation,
diagonal shrinkage, robust bounded-jitter Cholesky decomposition, and deterministic
Monte Carlo sampling using only the Python standard library.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

from .state_config import (
    ALL_CATEGORIES,
    CHOLESKY_JITTER_FACTORS,
    COVARIANCE_DIAGONAL_SHRINKAGE,
    FLOATING_POINT_TOLERANCE,
    MIN_SHARE_PCT,
    PARTIES,
    REFERENCE_CATEGORY,
)


def composition_to_alr(composition: dict[str, float]) -> list[float]:
    """Convert an 8-party + REST percentage composition to 8-dimensional ALR coordinates.

    Parameters:
        composition: Dictionary containing shares for all 8 parties and 'REST'.

    Returns:
        List of 8 log-ratio values relative to REST: log(P_party / P_REST).

    Raises:
        ValueError: If categories are missing, or if any share is materially negative.
    """
    for cat in ALL_CATEGORIES:
        if cat not in composition or composition[cat] is None:
            raise ValueError(f"Missing required composition category: {cat!r}")

    cleaned: list[float] = []
    for cat in ALL_CATEGORIES:
        val = float(composition[cat])
        if val < -FLOATING_POINT_TOLERANCE:
            raise ValueError(
                f"Materially negative support value for {cat}: {val} (< -{FLOATING_POINT_TOLERANCE})"
            )
        # Clamp tiny floating-point artifact to zero before min-share flooring
        if val < 0.0:
            val = 0.0
        # Floor tiny/zero share to MIN_SHARE_PCT
        if val < MIN_SHARE_PCT:
            val = MIN_SHARE_PCT
        cleaned.append(val)

    # Renormalize the 9 categories to sum to exactly 100.0%
    total = sum(cleaned)
    normalized = [(v / total) * 100.0 for v in cleaned]
    rest_val = normalized[-1]

    # Compute ALR coordinates: log(P_party / P_REST)
    return [math.log(normalized[i] / rest_val) for i in range(len(PARTIES))]


def alr_to_composition(alr_vector: Sequence[float]) -> dict[str, float]:
    """Convert 8-dimensional ALR coordinates to an 8-party + REST percentage composition.

    Uses a max-shifted softmax formulation for guaranteed numerical stability.

    Returns:
        Dictionary mapping M, L, C, KD, S, V, MP, SD, and REST to percentages summing to 100.
    """
    if len(alr_vector) != len(PARTIES):
        raise ValueError(
            f"Expected ALR vector of length {len(PARTIES)}, got {len(alr_vector)}"
        )

    # Reference category log-ratio is 0.0
    shift = max(max(alr_vector), 0.0)
    exp_parties = [math.exp(z - shift) for z in alr_vector]
    exp_rest = math.exp(0.0 - shift)

    total_exp = sum(exp_parties) + exp_rest
    composition: dict[str, float] = {
        party: (exp_val / total_exp) * 100.0
        for party, exp_val in zip(PARTIES, exp_parties)
    }
    composition[REFERENCE_CATEGORY] = (exp_rest / total_exp) * 100.0
    return composition


def calculate_sample_mean(vectors: Sequence[Sequence[float]]) -> list[float]:
    """Calculate the component-wise arithmetic mean of a sequence of vectors."""
    n = len(vectors)
    if n == 0:
        raise ValueError("Cannot calculate mean of empty vector sequence")
    dim = len(vectors[0])
    sums = [0.0] * dim
    for vec in vectors:
        if len(vec) != dim:
            raise ValueError("Vector dimension mismatch in sample mean calculation")
        for i in range(dim):
            sums[i] += vec[i]
    return [s / n for s in sums]


def calculate_sample_covariance(
    vectors: Sequence[Sequence[float]],
    means: Sequence[float] | None = None,
) -> list[list[float]]:
    """Calculate sample covariance matrix with Bessel's correction (N - 1 denominator)."""
    n = len(vectors)
    if n < 2:
        raise ValueError(
            f"Sample covariance requires at least 2 observations, got {n}"
        )
    dim = len(vectors[0])
    mu = means if means is not None else calculate_sample_mean(vectors)
    if len(mu) != dim:
        raise ValueError("Mean vector dimension does not match data dimension")

    cov = [[0.0] * dim for _ in range(dim)]
    for vec in vectors:
        diffs = [vec[i] - mu[i] for i in range(dim)]
        for i in range(dim):
            for j in range(i, dim):
                cov[i][j] += diffs[i] * diffs[j]

    denom = float(n - 1)
    for i in range(dim):
        for j in range(i, dim):
            cov[i][j] /= denom
            cov[j][i] = cov[i][j]
    return cov


def apply_covariance_shrinkage(
    cov_raw: Sequence[Sequence[float]],
    shrinkage: float = COVARIANCE_DIAGONAL_SHRINKAGE,
) -> list[list[float]]:
    """Apply fixed diagonal shrinkage: (1 - shrinkage) * cov + shrinkage * diag(cov)."""
    dim = len(cov_raw)
    shrunk = [[0.0] * dim for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            if i == j:
                shrunk[i][j] = float(cov_raw[i][i])
            else:
                shrunk[i][j] = (1.0 - shrinkage) * float(cov_raw[i][j])
    return shrunk


def cholesky_decomposition_with_jitter(
    matrix: Sequence[Sequence[float]],
    jitter_factors: Sequence[float] = CHOLESKY_JITTER_FACTORS,
) -> tuple[list[list[float]], float]:
    """Compute lower-triangular Cholesky factor L (such that L * L^T = A).

    Attempts decomposition without jitter first. If non-positive-definite due to numerical
    imprecision, searches over a bounded sequence of scale-relative diagonal jitters.

    Returns:
        tuple of (L_matrix, jitter_added).

    Raises:
        ValueError: If matrix is not symmetric, has non-positive diagonal entries,
                    or fails Cholesky decomposition after bounded jitter sequence.
    """
    dim = len(matrix)
    for i in range(dim):
        if len(matrix[i]) != dim:
            raise ValueError(f"Matrix is not square: row {i} has length {len(matrix[i])}")
        if matrix[i][i] <= 0.0:
            raise ValueError(f"Matrix has non-positive diagonal entry at index {i}: {matrix[i][i]}")
        for j in range(i + 1, dim):
            if abs(matrix[i][j] - matrix[j][i]) > 1e-6:
                raise ValueError(f"Matrix is asymmetric at ({i}, {j}): {matrix[i][j]} != {matrix[j][i]}")

    mean_diag = sum(matrix[i][i] for i in range(dim)) / dim
    candidate_jitters = [0.0] + [mean_diag * factor for factor in jitter_factors]

    for jitter in candidate_jitters:
        try:
            L = [[0.0] * dim for _ in range(dim)]
            for i in range(dim):
                for j in range(i + 1):
                    s = sum(L[i][k] * L[j][k] for k in range(j))
                    if i == j:
                        val = (matrix[i][i] + jitter) - s
                        if val <= 0.0:
                            raise ArithmeticError("Non-positive pivot in Cholesky")
                        L[i][j] = math.sqrt(val)
                    else:
                        L[i][j] = (matrix[i][j] - s) / L[j][j]
            return L, jitter
        except ArithmeticError:
            continue

    raise ValueError(
        "Matrix is not positive-definite after searching bounded diagonal jitter sequence"
    )


def sample_multivariate_normal(
    mean: Sequence[float],
    L_matrix: Sequence[Sequence[float]],
    rng: random.Random,
) -> list[float]:
    """Draw a single sample from N(mean, Sigma) given lower Cholesky factor L."""
    dim = len(mean)
    z = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    sample = [0.0] * dim
    for i in range(dim):
        sample[i] = mean[i] + sum(L_matrix[i][k] * z[k] for k in range(i + 1))
    return sample


def calculate_percentile(sorted_values: Sequence[float], p: float) -> float:
    """Calculate percentile using linear interpolation: index h = (n - 1) * p."""
    n = len(sorted_values)
    if n == 0:
        raise ValueError("Cannot calculate percentile of empty sequence")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Percentile p must be in [0, 1], got {p}")
    if n == 1:
        return float(sorted_values[0])

    h = (n - 1) * p
    i = int(math.floor(h))
    f = h - i
    if i >= n - 1:
        return float(sorted_values[-1])
    return (1.0 - f) * float(sorted_values[i]) + f * float(sorted_values[i + 1])


def summarize_samples(
    samples: Sequence[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Compute summary statistics (mean, SD, median, P05, P25, P75, P95) for all categories."""
    n = len(samples)
    if n == 0:
        raise ValueError("Cannot summarize empty sample sequence")

    summary: dict[str, dict[str, float]] = {}
    for cat in ALL_CATEGORIES:
        vals = sorted(sample[cat] for sample in samples)
        mean_val = sum(vals) / n
        variance = sum((x - mean_val) ** 2 for x in vals) / (n - 1 if n > 1 else 1)
        sd_val = math.sqrt(variance)

        summary[cat] = {
            "mean": mean_val,
            "std_dev": sd_val,
            "p05": calculate_percentile(vals, 0.05),
            "p25": calculate_percentile(vals, 0.25),
            "p50": calculate_percentile(vals, 0.50),
            "p75": calculate_percentile(vals, 0.75),
            "p95": calculate_percentile(vals, 0.95),
        }
    return summary
