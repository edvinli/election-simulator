"""Deterministic biproportional controlled rounding for constituency x party vote matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class ControlledRoundingResult:
    """Output from biproportional controlled rounding."""

    rounded_matrix: np.ndarray
    max_cell_error: float
    max_row_error: int
    max_column_error: int
    total_conserved: bool


# Precomputed static constraint matrix for 29 constituencies x 9 parties
_N_ROWS = 29
_N_COLS = 9
_N_VARS = _N_ROWS * _N_COLS
_A_EQ = np.zeros((_N_ROWS + _N_COLS, _N_VARS), dtype=np.float64)
for _i in range(_N_ROWS):
    _A_EQ[_i, _i * _N_COLS : (_i + 1) * _N_COLS] = 1.0
for _j in range(_N_COLS):
    _A_EQ[_N_ROWS + _j, _j::_N_COLS] = 1.0
_BOUNDS = [(0.0, 1.0) for _ in range(_N_VARS)]


def _solve_highs_lp_controlled_rounding(
    floor_X: np.ndarray,
    residuals: np.ndarray,
    r_deficits: np.ndarray,
    c_deficits: np.ndarray,
    nrows: int,
    ncols: int,
) -> np.ndarray:
    """Solve controlled rounding using HiGHS LP."""
    if nrows == _N_ROWS and ncols == _N_COLS:
        A_eq = _A_EQ
        bounds = _BOUNDS
    else:
        A_eq = np.zeros((nrows + ncols, nrows * ncols), dtype=np.float64)
        for i in range(nrows):
            A_eq[i, i * ncols : (i + 1) * ncols] = 1.0
        for j in range(ncols):
            A_eq[nrows + j, j::ncols] = 1.0
        bounds = [(0.0, 1.0) for _ in range(nrows * ncols)]

    c_cost = (1.0 - residuals).ravel()
    b_eq = np.concatenate([r_deficits, c_deficits])

    res = linprog(c_cost, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Controlled rounding HiGHS solver failed: {res.message}")

    delta = np.round(res.x.reshape(nrows, ncols)).astype(np.int64)
    return floor_X + delta


def _solve_fast_bipartite_flow_rounding(
    floor_X: np.ndarray,
    residuals: np.ndarray,
    r_deficits: np.ndarray,
    c_deficits: np.ndarray,
    nrows: int,
    ncols: int,
) -> np.ndarray:
    """Solve controlled rounding using fast bipartite residual flow with alternating path augmentation."""
    r_rem = r_deficits.astype(np.int64).copy()
    d_rem = c_deficits.astype(np.int64).copy()
    Z = np.zeros((nrows, ncols), dtype=np.int64)

    # 1. Greedy assignment on highest fractional residuals
    flat_indices = np.argsort(-residuals.ravel())
    for idx in flat_indices:
        r = idx // ncols
        c = idx % ncols
        if r_rem[r] > 0 and d_rem[c] > 0 and Z[r, c] == 0:
            Z[r, c] = 1
            r_rem[r] -= 1
            d_rem[c] -= 1

    # 2. If any deficit remains, augment along alternating paths
    max_augment_steps = 100
    aug_step = 0
    while np.sum(r_rem) > 0 and aug_step < max_augment_steps:
        aug_step += 1
        r_sources = np.where(r_rem > 0)[0]
        d_sinks = np.where(d_rem > 0)[0]
        if len(r_sources) == 0 or len(d_sinks) == 0:
            break
        r_src = r_sources[0]
        d_snk = d_sinks[0]

        best_path = None
        best_cost = 1e9
        for j in range(ncols):
            if Z[r_src, j] == 0:
                for i in range(nrows):
                    if Z[i, j] == 1 and Z[i, d_snk] == 0:
                        cost = (1.0 - residuals[r_src, j]) + residuals[i, j] + (1.0 - residuals[i, d_snk])
                        if cost < best_cost:
                            best_cost = cost
                            best_path = (r_src, j, i, d_snk)

        if best_path:
            r_s, col_j, row_i, col_snk = best_path
            Z[r_s, col_j] = 1
            Z[row_i, col_j] = 0
            Z[row_i, col_snk] = 1
            r_rem[r_s] -= 1
            d_rem[col_snk] -= 1
        else:
            # Direct greedy assignment fallback
            avail_c = np.where(Z[r_src] == 0)[0]
            if len(avail_c) > 0:
                best_c = avail_c[np.argmax(residuals[r_src, avail_c])]
                Z[r_src, best_c] = 1
                r_rem[r_src] -= 1
                # Rebalance columns
                col_over = np.where(np.sum(Z, axis=0) > c_deficits)[0]
                if len(col_over) > 0:
                    c_ov = col_over[0]
                    r_cands = np.where((Z[:, c_ov] == 1) & (r_rem == 0))[0]
                    if len(r_cands) > 0:
                        best_r = r_cands[np.argmin(residuals[r_cands, c_ov])]
                        Z[best_r, c_ov] = 0
                        r_rem[best_r] += 1

    # If any residual discrepancy remains, fall back to HiGHS
    if np.sum(r_rem) > 0 or np.sum(d_rem) > 0:
        return _solve_highs_lp_controlled_rounding(
            floor_X=floor_X,
            residuals=residuals,
            r_deficits=r_deficits,
            c_deficits=c_deficits,
            nrows=nrows,
            ncols=ncols,
        )

    return floor_X + Z


def biproportional_controlled_rounding(
    float_matrix: np.ndarray,
    target_row_sums: np.ndarray,
    target_col_sums: np.ndarray,
    solver: Literal["auto", "fast_flow", "highs"] = "auto",
) -> ControlledRoundingResult:
    """Deterministically round a 2D float matrix to integers preserving exact row and column margins.

    Solves the binary residual bipartite transportation problem:
        Y_{c,p} = floor(X_{c,p}) + Z_{c,p},  where Z_{c,p} in {0, 1}
        subject to:
            sum_p Z_{c,p} = target_row_sums[c] - sum_p floor(X_{c,p})  for all c
            sum_c Z_{c,p} = target_col_sums[p] - sum_c floor(X_{c,p})  for all p

    Parameters:
        float_matrix: 2D array of shape (C, P) representing continuous votes.
        target_row_sums: 1D integer array of shape (C,) representing target constituency totals.
        target_col_sums: 1D integer array of shape (P,) representing target national party totals.
        solver: Solution method ('auto', 'fast_flow', or 'highs').

    Returns:
        ControlledRoundingResult with exact integer matrix and diagnostic metrics.
    """
    X = np.asarray(float_matrix, dtype=np.float64)
    R_target = np.asarray(target_row_sums, dtype=np.int64)
    C_target = np.asarray(target_col_sums, dtype=np.int64)

    nrows, ncols = X.shape
    if nrows != len(R_target) or ncols != len(C_target):
        raise ValueError(f"Matrix shape ({nrows}, {ncols}) does not match target margins ({len(R_target)}, {len(C_target)})")

    tot_r = int(np.sum(R_target))
    tot_c = int(np.sum(C_target))
    if tot_r != tot_c:
        raise ValueError(f"Sum of target row totals ({tot_r}) != target column totals ({tot_c})")

    # If input continuous matrix is not yet raked to exact integer margins, perform quick IPF raking
    row_sums = np.sum(X, axis=1)
    col_sums = np.sum(X, axis=0)
    if np.max(np.abs(row_sums - R_target)) > 0.5 or np.max(np.abs(col_sums - C_target)) > 0.5:
        R_col = R_target[:, np.newaxis].astype(np.float64)
        C_row = C_target[np.newaxis, :].astype(np.float64)
        for _ in range(8):
            X *= R_col / np.maximum(np.sum(X, axis=1, keepdims=True), 1e-12)
            X *= C_row / np.maximum(np.sum(X, axis=0, keepdims=True), 1e-12)

    floor_X = np.floor(X).astype(np.int64)
    residuals = X - floor_X

    r_deficits = (R_target - np.sum(floor_X, axis=1)).astype(np.float64)
    c_deficits = (C_target - np.sum(floor_X, axis=0)).astype(np.float64)

    tot_def = int(np.sum(r_deficits))
    if tot_def == 0:
        # All floors already sum exactly to targets
        return ControlledRoundingResult(
            rounded_matrix=floor_X,
            max_cell_error=float(np.max(residuals)),
            max_row_error=0,
            max_column_error=0,
            total_conserved=True,
        )

    if solver == "highs":
        Y = _solve_highs_lp_controlled_rounding(floor_X, residuals, r_deficits, c_deficits, nrows, ncols)
    else:
        Y = _solve_fast_bipartite_flow_rounding(floor_X, residuals, r_deficits, c_deficits, nrows, ncols)

    # Diagnostics
    actual_r = np.sum(Y, axis=1)
    actual_c = np.sum(Y, axis=0)
    max_r_err = int(np.max(np.abs(actual_r - R_target)))
    max_c_err = int(np.max(np.abs(actual_c - C_target)))
    max_cell_err = float(np.max(np.abs(Y - X)))

    return ControlledRoundingResult(
        rounded_matrix=Y,
        max_cell_error=max_cell_err,
        max_row_error=max_r_err,
        max_column_error=max_c_err,
        total_conserved=(int(np.sum(Y)) == tot_r),
    )
