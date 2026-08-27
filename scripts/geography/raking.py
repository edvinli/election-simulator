"""Deterministic Iterative Proportional Fitting (IPF / Raking) for geographical vote projection."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class IPFResult:
    """Results and convergence diagnostics from Iterative Proportional Fitting."""

    matrix: np.ndarray
    iterations: int
    max_row_error: float
    max_column_error: float
    converged: bool


def iterative_proportional_fitting(
    baseline_matrix: np.ndarray,
    target_row_sums: np.ndarray,
    target_col_sums: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-8,
    eps: float = 1e-12,
) -> IPFResult:
    """Fit a 2D matrix X to target row and column margins using deterministic biproportional scaling (IPF).

    Finds X_{c,p} = a_c * B_{c,p} * b_p such that:
        sum_p X_{c,p} = target_row_sums[c]  for all c
        sum_c X_{c,p} = target_col_sums[p]  for all p

    Parameters:
        baseline_matrix: 2D array of shape (C, P), non-negative base values.
        target_row_sums: 1D array of shape (C,), target constituency row totals.
        target_col_sums: 1D array of shape (P,), target national column totals.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance for absolute margin errors.
        eps: Minimum floor for zero-cells in baseline to ensure numerical stability.

    Returns:
        IPFResult containing the converged matrix and diagnostic metrics.
    """
    B = np.asarray(baseline_matrix, dtype=np.float64)
    R = np.asarray(target_row_sums, dtype=np.float64)
    C = np.asarray(target_col_sums, dtype=np.float64)

    if B.ndim != 2:
        raise ValueError(f"Baseline matrix must be 2D, got shape {B.shape}")
    if R.ndim != 1 or R.shape[0] != B.shape[0]:
        raise ValueError(f"target_row_sums length {R.shape} must match rows {B.shape[0]}")
    if C.ndim != 1 or C.shape[0] != B.shape[1]:
        raise ValueError(f"target_col_sums length {C.shape} must match columns {B.shape[1]}")

    tot_r = np.sum(R)
    tot_c = np.sum(C)
    if not np.isclose(tot_r, tot_c, rtol=1e-5, atol=1e-4):
        raise ValueError(f"Target row sum ({tot_r:.4f}) does not match column sum ({tot_c:.4f})")

    # Replace absolute zeros with eps if column target > 0 to allow scaling
    X = np.maximum(B, 0.0).copy()
    for p in range(X.shape[1]):
        if C[p] > 0 and np.all(X[:, p] == 0):
            X[:, p] = eps

    max_row_err = float("inf")
    max_col_err = float("inf")
    converged = False

    for iteration in range(1, max_iter + 1):
        # 1. Row scaling (match constituency margins)
        row_sums = np.sum(X, axis=1)
        row_scale = np.where(row_sums > 0, R / np.maximum(row_sums, 1e-15), 1.0)
        X = X * row_scale[:, np.newaxis]

        # 2. Column scaling (match national party margins)
        col_sums = np.sum(X, axis=0)
        col_scale = np.where(col_sums > 0, C / np.maximum(col_sums, 1e-15), 1.0)
        X = X * col_scale[np.newaxis, :]

        # 3. Check convergence
        cur_row_sums = np.sum(X, axis=1)
        cur_col_sums = np.sum(X, axis=0)
        max_row_err = float(np.max(np.abs(cur_row_sums - R)))
        max_col_err = float(np.max(np.abs(cur_col_sums - C)))

        if max_row_err < tol and max_col_err < tol:
            converged = True
            return IPFResult(
                matrix=X,
                iterations=iteration,
                max_row_error=max_row_err,
                max_column_error=max_col_err,
                converged=True,
            )

    return IPFResult(
        matrix=X,
        iterations=max_iter,
        max_row_error=max_row_err,
        max_column_error=max_col_err,
        converged=converged,
    )
