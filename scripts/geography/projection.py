"""Deterministic national-to-constituency vote projection using raking / IPF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd

from .config import (
    DEFAULT_PROCESSED_GEOGRAPHY_DIR,
    MODEL_PARTIES_9,
    OFFICIAL_CONSTITUENCY_CODES,
    REST_MANDATE_LABEL,
)
from .integerization import biproportional_controlled_rounding
from .raking import IPFResult, iterative_proportional_fitting


@dataclass(frozen=True)
class ProjectionResult:
    """Output of a geographic vote projection run."""

    constituency_votes: dict[str, dict[str, int]]
    constituency_votes_float: dict[str, dict[str, float]]
    constituency_valid_votes: dict[str, int]
    national_votes: dict[str, int]
    national_vote_shares: dict[str, float]
    ipf_result: IPFResult
    mode: str  # "oracle" or "production"
    baseline_year: int
    target_year: int

    def to_allocator_input(self) -> dict[str, dict[str, int]]:
        """Return constituency votes dict with REST mapped to REST_MANDATE_LABEL for mandate allocation."""
        alloc_input: dict[str, dict[str, int]] = {}
        for c, p_map in self.constituency_votes.items():
            alloc_input[c] = {}
            for p, v in p_map.items():
                if p == "REST":
                    alloc_input[c][REST_MANDATE_LABEL] = v
                else:
                    alloc_input[c][p] = v
        return alloc_input


def _apportion_integers_largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    """Deterministically apportion a target integer total across float weights using Hamilton / largest remainder."""
    float_alloc = weights / np.sum(weights) * total
    int_alloc = np.floor(float_alloc).astype(np.int64)
    remainder = float_alloc - int_alloc
    deficit = total - int(np.sum(int_alloc))
    if deficit > 0:
        # Give +1 to elements with largest fractional remainders
        top_indices = np.argsort(-remainder)[:deficit]
        int_alloc[top_indices] += 1
    return int_alloc


from functools import lru_cache

@lru_cache(maxsize=32)
def _get_cached_geography_structures(
    baseline_year: int,
    target_year: int,
    mode: str,
    processed_dir_str: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Cache static baseline matrix B and target row vector R."""
    p_dir = Path(processed_dir_str)
    pv_df = pd.read_csv(p_dir / "constituency_party_votes_2014_2022.csv")
    el_df = pd.read_csv(p_dir / "constituency_electorates_2014_2026.csv")

    # 1. Build baseline matrix B (shape 29 x 9)
    sub_base = pv_df[pv_df["election_year"] == baseline_year]
    if sub_base.empty:
        raise ValueError(f"No baseline data available for election year {baseline_year}")

    B = np.zeros((len(OFFICIAL_CONSTITUENCY_CODES), len(MODEL_PARTIES_9)), dtype=np.float64)
    code_to_idx = {c: i for i, c in enumerate(OFFICIAL_CONSTITUENCY_CODES)}
    party_to_idx = {p: i for i, p in enumerate(MODEL_PARTIES_9)}

    for _, r in sub_base.iterrows():
        c_code = f"{int(r['constituency_code']):02d}"
        p_code = str(r["party"])
        if c_code in code_to_idx and p_code in party_to_idx:
            B[code_to_idx[c_code], party_to_idx[p_code]] = float(r["votes"])

    # 2. Determine target constituency row sums R_c
    if mode == "oracle":
        sub_el_target = el_df[el_df["election_year"] == target_year]
        if sub_el_target.empty or sub_el_target["valid_votes"].isnull().any():
            raise ValueError(f"Oracle mode requested but target election {target_year} has missing valid votes")
        R = np.zeros(len(OFFICIAL_CONSTITUENCY_CODES), dtype=np.float64)
        for _, r in sub_el_target.iterrows():
            c_code = f"{int(r['constituency_code']):02d}"
            if c_code in code_to_idx:
                R[code_to_idx[c_code]] = float(r["valid_votes"])
    elif mode in ("chronological", "production"):
        if target_year <= 2022:
            # Strictly chronological: row totals are derived entirely from baseline valid votes
            # Zero information from target election electorate or valid votes is accessed!
            R = np.sum(B, axis=1)
        else:
            # Forward 2026 production forecast: use decided 2026 electorate scaled by baseline turnout
            sub_el_target = el_df[el_df["election_year"] == target_year]
            sub_el_base = el_df[el_df["election_year"] == baseline_year]
            R = np.zeros(len(OFFICIAL_CONSTITUENCY_CODES), dtype=np.float64)
            for c_code in OFFICIAL_CONSTITUENCY_CODES:
                row_t = sub_el_target[sub_el_target["constituency_code"] == int(c_code)].iloc[0]
                row_b = sub_el_base[sub_el_base["constituency_code"] == int(c_code)].iloc[0]
                el_t = float(row_t["eligible_voters"])
                el_b = float(row_b["eligible_voters"])
                val_b = float(row_b["valid_votes"])
                rate_b = val_b / el_b if el_b > 0 else 0.85
                R[code_to_idx[c_code]] = el_t * rate_b
    else:
        raise ValueError(f"Unknown mode '{mode}'. Must be 'chronological', 'production', or 'oracle'")

    return B, R


def project_constituency_votes(
    national_vote_shares: Mapping[str, float],
    baseline_year: int,
    target_year: int,
    mode: str = "oracle",
    total_national_votes: int | None = None,
    processed_dir: Path | str | None = None,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> ProjectionResult:
    """Project national simulated vote shares onto 29 constituencies using deterministic IPF."""
    p_dir = Path(processed_dir) if processed_dir else DEFAULT_PROCESSED_GEOGRAPHY_DIR
    B_cached, R_cached = _get_cached_geography_structures(
        baseline_year=baseline_year,
        target_year=target_year,
        mode=mode,
        processed_dir_str=str(p_dir),
    )
    B = B_cached.copy()
    R = R_cached.copy()
    party_to_idx = {p: i for i, p in enumerate(MODEL_PARTIES_9)}

    total_valid = np.sum(R)
    if total_national_votes is not None and total_national_votes > 0:
        # Scale R proportionally to match explicitly supplied total_national_votes
        R = R * (total_national_votes / total_valid)
        total_valid = float(total_national_votes)

    # 3. Determine target column sums C_p
    # Normalize shares to sum exactly to 1.0
    shares_arr = np.array([float(national_vote_shares.get(p, 0.0)) for p in MODEL_PARTIES_9], dtype=np.float64)
    tot_s = np.sum(shares_arr)
    if tot_s <= 0:
        raise ValueError("National vote shares must sum to > 0")
    shares_arr = shares_arr / tot_s

    C = shares_arr * total_valid

    # 4. Run IPF Raking
    ipf_res = iterative_proportional_fitting(
        baseline_matrix=B,
        target_row_sums=R,
        target_col_sums=C,
        max_iter=max_iter,
        tol=tol,
    )

    if not ipf_res.converged:
        raise RuntimeError(
            f"IPF did not converge within {max_iter} iterations (max_row_err={ipf_res.max_row_error:.2e}, max_col_err={ipf_res.max_column_error:.2e})"
        )

    # 5. Format Output with Biproportional Controlled Rounding
    total_int_votes = int(round(total_valid))
    R_int = _apportion_integers_largest_remainder(R, total_int_votes)
    C_int = _apportion_integers_largest_remainder(C, total_int_votes)

    cr_res = biproportional_controlled_rounding(
        float_matrix=ipf_res.matrix,
        target_row_sums=R_int,
        target_col_sums=C_int,
    )
    int_matrix = cr_res.rounded_matrix

    const_votes_dict: dict[str, dict[str, int]] = {}
    const_votes_float_dict: dict[str, dict[str, float]] = {}
    const_valid_dict: dict[str, int] = {}

    for i, c_code in enumerate(OFFICIAL_CONSTITUENCY_CODES):
        const_votes_dict[c_code] = {}
        const_votes_float_dict[c_code] = {}
        for j, p_code in enumerate(MODEL_PARTIES_9):
            const_votes_dict[c_code][p_code] = int(int_matrix[i, j])
            const_votes_float_dict[c_code][p_code] = float(ipf_res.matrix[i, j])
        const_valid_dict[c_code] = sum(const_votes_dict[c_code].values())

    nat_votes_dict = {p: int(np.sum(int_matrix[:, party_to_idx[p]])) for p in MODEL_PARTIES_9}
    tot_int_votes = sum(nat_votes_dict.values())
    nat_shares_dict = {p: nat_votes_dict[p] / tot_int_votes for p in MODEL_PARTIES_9}

    return ProjectionResult(
        constituency_votes=const_votes_dict,
        constituency_votes_float=const_votes_float_dict,
        constituency_valid_votes=const_valid_dict,
        national_votes=nat_votes_dict,
        national_vote_shares=nat_shares_dict,
        ipf_result=ipf_res,
        mode=mode,
        baseline_year=baseline_year,
        target_year=target_year,
    )
