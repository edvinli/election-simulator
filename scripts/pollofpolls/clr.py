"""Centered Log-Ratio (CLR) compositional transformations for opinion dynamics modeling."""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from .state_config import ALL_CATEGORIES, MIN_SHARE_PCT


def composition_to_clr(
    comp: dict[str, float] | Sequence[float],
    categories: Sequence[str] = ALL_CATEGORIES,
    min_share_pct: float = MIN_SHARE_PCT,
) -> tuple[np.ndarray, bool]:
    """Transform a percentage composition to Centered Log-Ratio (CLR) coordinates.

    Definition:
        clr_i(p) = ln(p_i / g(p)) = ln(p_i) - (1 / D) * sum_{j=1}^D ln(p_j)
    where g(p) is the geometric mean of the shares.

    Parameters:
        comp: Dict or Sequence of category support shares summing to 100%.
        categories: Ordered category names (default: 9 canonical Swedish categories).
        min_share_pct: Minimum percentage floor applied before logarithm.

    Returns:
        tuple (clr_vector, was_floored):
            clr_vector: 1D NumPy array of length D satisfying sum(clr_vector) == 0.
            was_floored: Boolean indicating if any component was floored and renormalized.
    """
    if isinstance(comp, dict):
        raw_vals = [float(comp[cat]) for cat in categories]
    else:
        raw_vals = [float(x) for x in comp]

    if len(raw_vals) != len(categories):
        raise ValueError(f"Expected {len(categories)} elements, got {len(raw_vals)}")

    was_floored = False
    arr = np.array(raw_vals, dtype=float)

    if np.any(arr < min_share_pct * 0.99):
        was_floored = True
        arr = np.maximum(arr, min_share_pct)
        # Renormalize to exact 100%
        arr = 100.0 * (arr / np.sum(arr))


    log_vals = np.log(arr)
    mean_log = np.mean(log_vals)
    clr_vec = log_vals - mean_log

    return clr_vec, was_floored


def clr_to_composition(
    clr_vec: np.ndarray | Sequence[float],
    categories: Sequence[str] = ALL_CATEGORIES,
) -> dict[str, float]:
    """Inverse Centered Log-Ratio transformation mapping CLR vector back to percentage simplex.

    Formula:
        p_i = 100 * exp(clr_i - m) / sum_{j=1}^D exp(clr_j - m)
    where m = max(clr) ensures numerical stability.

    Returns:
        Dict mapping category names to positive support percentages summing strictly to 100.0%.
    """
    arr = np.asarray(clr_vec, dtype=float)
    if arr.shape[0] != len(categories):
        raise ValueError(f"Expected {len(categories)} CLR coordinates, got {arr.shape[0]}")

    max_val = np.max(arr)
    exp_vals = np.exp(arr - max_val)
    sum_exp = np.sum(exp_vals)
    shares = 100.0 * (exp_vals / sum_exp)

    return {cat: float(shares[i]) for i, cat in enumerate(categories)}


def clr_to_composition_matrix(clr_matrix: np.ndarray) -> np.ndarray:
    """Vectorized inverse CLR mapping for 2D array of shape (N, D).

    Parameters:
        clr_matrix: 2D NumPy array of shape (N, D).

    Returns:
        2D NumPy array of shape (N, D) where each row sums strictly to 100.0%.
    """
    arr = np.asarray(clr_matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    max_vals = np.max(arr, axis=-1, keepdims=True)
    exp_vals = np.exp(arr - max_vals)
    sum_exp = np.sum(exp_vals, axis=-1, keepdims=True)
    return 100.0 * (exp_vals / sum_exp)
