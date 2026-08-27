"""Probabilistic evaluation metrics for discrete seat distributions and multivariate seat vectors."""

from __future__ import annotations

import numpy as np


def calculate_discrete_seat_crps(samples: np.ndarray, actual: int | float) -> float:
    """Calculate exact discrete Continuous Ranked Probability Score (CRPS) on integer seat domain.

    For an integer-valued random variable S with support in [0, 349]:
        CRPS(F, y) = sum_{k=0}^{348} (F(k) - 1(y <= k))^2
    where F(k) = (1/N) * sum_{i=1}^N 1(s_i <= k).
    """
    s_arr = np.asarray(samples, dtype=np.int64)
    y = int(round(actual))
    n = len(s_arr)
    if n == 0:
        return 0.0

    # Evaluate CDF F(k) on support k in [0, 349]
    k_vals = np.arange(350, dtype=np.int64)
    # Using np.searchsorted on sorted samples for O(N log N) speed
    s_sorted = np.sort(s_arr)
    f_k = np.searchsorted(s_sorted, k_vals, side="right") / n
    h_k = (y <= k_vals).astype(np.float64)

    crps = np.sum((f_k - h_k) ** 2)
    return float(round(crps, 4))


def calculate_multivariate_energy_score(
    samples_matrix: np.ndarray,
    actual_vector: np.ndarray,
    chunk_size: int = 500,
) -> float:
    """Calculate exact multivariate Energy Score on 8-party seat vector.

    ES(F, y) = E[||S - y||_2] - 0.5 * E[||S - S'||_2]

    Parameters:
        samples_matrix: shape (N, 8) integer seat draws.
        actual_vector: shape (8,) actual seat vector.
        chunk_size: Block size for pairwise distance accumulation to avoid large memory footprint.

    Returns:
        Energy Score value.
    """
    S = np.asarray(samples_matrix, dtype=np.float64)
    y = np.asarray(actual_vector, dtype=np.float64)
    n = len(S)
    if n == 0:
        return 0.0

    # Term 1: E[||S - y||_2]
    term1 = float(np.mean(np.linalg.norm(S - y, axis=1)))

    # Term 2: 0.5 * E[||S - S'||_2] computed chunked
    total_dist = 0.0
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        diffs = S[start:end, np.newaxis, :] - S[np.newaxis, :, :]  # shape (chunk, N, 8)
        dists = np.linalg.norm(diffs, axis=-1)
        total_dist += float(np.sum(dists))

    term2 = total_dist / (2.0 * n * n)
    es = term1 - term2
    return float(round(es, 4))


def calculate_interval_coverage_and_width(
    samples: np.ndarray,
    actual: int | float,
    level: float = 0.80,
) -> tuple[bool, int, int, int]:
    """Calculate empirical interval coverage and width for specified confidence level (e.g. 0.50, 0.80, 0.90).

    Returns:
        tuple (is_covered, width, lower_quantile, upper_quantile)
    """
    s_arr = np.asarray(samples, dtype=np.int64)
    alpha = 1.0 - level
    q_low = int(np.quantile(s_arr, alpha / 2.0, method="nearest"))
    q_high = int(np.quantile(s_arr, 1.0 - alpha / 2.0, method="nearest"))

    y = int(round(actual))
    is_covered = bool(q_low <= y <= q_high)
    width = int(q_high - q_low)

    return is_covered, width, q_low, q_high


def calculate_empirical_percentile(samples: np.ndarray, actual: int | float) -> float:
    """Calculate mid-rank empirical percentile: P = 100 * (#(s < y) + 0.5 * #(s == y)) / n."""
    s_arr = np.asarray(samples, dtype=np.int64)
    y = int(round(actual))
    n = len(s_arr)
    if n == 0:
        return 0.0
    less_count = np.sum(s_arr < y)
    equal_count = np.sum(s_arr == y)
    return float(round(100.0 * (less_count + 0.5 * equal_count) / n, 2))
