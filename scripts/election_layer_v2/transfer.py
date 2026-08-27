"""Simplex-safe bounded percentage-point transfer scaling and diagnostics."""

from __future__ import annotations

from typing import Any
import numpy as np

from .config import MIN_SHARE_PCT


def compute_simplex_transfer_scale(
    base_comp: np.ndarray,
    residual_vec: np.ndarray,
    eps: float = MIN_SHARE_PCT,
) -> float:
    """Calculate maximum feasible transfer scale lambda in [0, 1] preserving simplex bounds >= eps.

    Formula:
        lambda = min(1.0, min_{r_p < 0} (x_p - eps) / (-r_p))
        If no component has r_p < 0, lambda = 1.0.
    """
    neg_mask = residual_vec < -1e-12
    if not np.any(neg_mask):
        return 1.0

    donor_bases = base_comp[neg_mask]
    donor_neg_res = -residual_vec[neg_mask]

    # Calculate individual donor exhaustion ratios
    ratios = (donor_bases - eps) / donor_neg_res
    min_ratio = float(np.min(ratios))

    lambda_val = min(1.0, max(0.0, min_ratio))
    return float(np.clip(lambda_val, 0.0, 1.0))


def apply_simplex_transfer(
    base_comp: np.ndarray,
    residual_vec: np.ndarray,
    eps: float = MIN_SHARE_PCT,
) -> tuple[np.ndarray, float]:
    """Apply bounded percentage-point transfer: x' = x + lambda * r.

    Returns:
        (x_prime, lambda_val)
    """
    # 1. Residual zero-sum validation
    res_sum = float(np.sum(residual_vec))
    if abs(res_sum) > 0.05:
        raise ValueError(f"Residual vector sum ({res_sum:.6f}) deviates materially from zero")

    # Clean tiny floating residue if needed
    cleaned_res = residual_vec.copy()
    if abs(res_sum) > 1e-12:
        cleaned_res = cleaned_res - (res_sum / len(cleaned_res))

    # 2. Compute lambda
    lam = compute_simplex_transfer_scale(base_comp, cleaned_res, eps=eps)

    # 3. Apply transfer
    x_prime = base_comp + lam * cleaned_res

    # 4. Clamp numerical floating-point noise
    x_prime = np.maximum(x_prime, eps)
    sum_prime = np.sum(x_prime)
    if abs(sum_prime - 100.0) > 1e-10:
        x_prime = x_prime * (100.0 / sum_prime)

    return x_prime, lam


def apply_batch_simplex_transfer(
    base_matrix: np.ndarray,
    residuals_matrix: np.ndarray,
    eps: float = MIN_SHARE_PCT,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply bounded percentage-point transfer row-by-row across N samples.

    Parameters:
        base_matrix: Array of shape (N, C) representing base sample compositions.
        residuals_matrix: Array of shape (N, C) or (C,) representing transfer vectors.

    Returns:
        (transferred_matrix, lambdas_array)
    """
    n, c = base_matrix.shape
    if residuals_matrix.ndim == 1:
        res_mat = np.tile(residuals_matrix, (n, 1))
    else:
        res_mat = residuals_matrix

    out_matrix = np.empty_like(base_matrix)
    lambdas = np.empty(n, dtype=float)

    for i in range(n):
        x_p, lam = apply_simplex_transfer(base_matrix[i], res_mat[i], eps=eps)
        out_matrix[i] = x_p
        lambdas[i] = lam

    return out_matrix, lambdas


def summarize_lambda_diagnostics(lambdas: np.ndarray) -> dict[str, float]:
    """Calculate standard lambda attenuation metrics."""
    if len(lambdas) == 0:
        return {
            "mean_lambda": 1.0,
            "p05_lambda": 1.0,
            "fraction_lambda_lt_0_99": 0.0,
            "fraction_lambda_lt_0_90": 0.0,
            "fraction_lambda_lt_0_75": 0.0,
        }

    return {
        "mean_lambda": round(float(np.mean(lambdas)), 4),
        "p05_lambda": round(float(np.percentile(lambdas, 5)), 4),
        "fraction_lambda_lt_0_99": round(float(np.mean(lambdas < 0.99)), 4),
        "fraction_lambda_lt_0_90": round(float(np.mean(lambdas < 0.90)), 4),
        "fraction_lambda_lt_0_75": round(float(np.mean(lambdas < 0.75)), 4),
    }
