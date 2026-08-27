"""Multivariate Energy Score computation for full 9-category vote composition forecasts."""

from __future__ import annotations

import numpy as np


def compute_energy_score(
    samples: np.ndarray,
    actual: np.ndarray,
    chunk_size: int = 500,
) -> float:
    """Compute multivariate Energy Score for continuous/Monte Carlo sample matrix.

    Formula:
        ES(F, y) = E||X - y||_2 - 0.5 * E||X - X'||_2
        E||X - y||_2 = (1 / N) * sum_{i=1}^N ||x^{(i)} - y||_2
        E||X - X'||_2 = (1 / (N * (N - 1))) * sum_{i != j} ||x^{(i)} - x^{(j)}||_2

    Parameters:
        samples: Array of shape (N, D) containing predictive composition draws.
        actual: Array of shape (D,) containing actual vote composition vector.
        chunk_size: Block size for vectorized memory-efficient pairwise distance accumulation.

    Returns:
        Multivariate Energy Score (float).
    """
    n, d = samples.shape
    if n == 0:
        return 0.0
    if n == 1:
        return float(np.linalg.norm(samples[0] - actual))

    # 1. First term: E||X - y||_2
    dist_to_actual = np.linalg.norm(samples - actual[None, :], axis=1)
    term1 = float(np.mean(dist_to_actual))

    # 2. Second term: 0.5 * E||X - X'||_2 (chunked pairwise U-statistic)
    total_pairwise_dist = 0.0
    for start_idx in range(0, n, chunk_size):
        end_idx = min(start_idx + chunk_size, n)
        chunk = samples[start_idx:end_idx]  # Shape (B, D)
        # Broadcasting: (B, 1, D) - (1, N, D) -> (B, N, D)
        diff = chunk[:, None, :] - samples[None, :, :]
        chunk_norms = np.linalg.norm(diff, axis=2)  # Shape (B, N)
        total_pairwise_dist += float(np.sum(chunk_norms))

    # Diagonal elements ||x^{(i)} - x^{(i)}||_2 are identically 0
    term2 = 0.5 * (total_pairwise_dist / (n * (n - 1)))
    return float(term1 - term2)


def compute_discrete_energy_score(
    support_points: np.ndarray,
    actual: np.ndarray,
) -> float:
    """Compute exact multivariate Energy Score for a finite discrete distribution with equal weights."""
    m, d = support_points.shape
    if m == 0:
        return 0.0
    if m == 1:
        return float(np.linalg.norm(support_points[0] - actual))

    # 1. First term: (1 / M) * sum ||s_m - y||_2
    term1 = float(np.mean(np.linalg.norm(support_points - actual[None, :], axis=1)))

    # 2. Second term: 0.5 / (M * (M - 1)) * sum_{m != l} ||s_m - s_l||_2
    diff = support_points[:, None, :] - support_points[None, :, :]
    pairwise_matrix = np.linalg.norm(diff, axis=2)
    total_pairwise = float(np.sum(pairwise_matrix))
    term2 = 0.5 * (total_pairwise / (m * (m - 1)))

    return float(term1 - term2)


def reference_energy_score_slow(
    samples: np.ndarray,
    actual: np.ndarray,
) -> float:
    """Slow reference double-loop implementation of Energy Score for unit test verification."""
    n = len(samples)
    if n == 0:
        return 0.0
    if n == 1:
        return float(np.linalg.norm(samples[0] - actual))

    # Term 1
    term1 = sum(float(np.linalg.norm(samples[i] - actual)) for i in range(n)) / n

    # Term 2
    pair_sum = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                pair_sum += float(np.linalg.norm(samples[i] - samples[j]))

    term2 = 0.5 * pair_sum / (n * (n - 1))
    return term1 - term2
