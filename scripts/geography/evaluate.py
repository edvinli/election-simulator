"""Historical evaluation of GeographicProjection v1 for 2014->2018 and 2018->2022."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from scripts.mandates.allocator import allocate_riksdag_seats
from scripts.mandates.config import (
    FIXED_SEATS_2018,
    FIXED_SEATS_2022,
    OFFICIAL_CONSTITUENCIES,
)
from .config import DEFAULT_PROCESSED_GEOGRAPHY_DIR, MODEL_PARTIES_9, OFFICIAL_CONSTITUENCY_CODES
from .projection import project_constituency_votes


@dataclass(frozen=True)
class HistoricalEvaluationMetrics:
    baseline_year: int
    target_year: int
    mode: str
    constituency_share_mae: float
    party_share_maes: dict[str, float]
    national_share_max_err: float
    constituency_valid_votes_max_err: float
    constituency_valid_votes_mape: float
    seat_differences: dict[str, int]
    total_seat_error: int
    projected_seats: dict[str, int]
    certified_seats: dict[str, int]
    ipf_iterations: int


def evaluate_projection_pair(
    baseline_year: int,
    target_year: int,
    mode: str = "oracle",
    processed_dir: Path | str | None = None,
) -> HistoricalEvaluationMetrics:
    """Evaluate geographic projection from baseline_year to target_year against certified election outcomes."""
    p_dir = Path(processed_dir) if processed_dir else DEFAULT_PROCESSED_GEOGRAPHY_DIR
    pv_df = pd.read_csv(p_dir / "constituency_party_votes_2014_2022.csv")
    el_df = pd.read_csv(p_dir / "constituency_electorates_2014_2026.csv")

    # 1. Get actual national vote shares in target election
    sub_target = pv_df[pv_df["election_year"] == target_year]
    nat_target_votes = {
        p: int(sub_target[sub_target["party"] == p]["votes"].sum()) for p in MODEL_PARTIES_9
    }
    tot_target_valid = sum(nat_target_votes.values())
    nat_target_shares = {p: nat_target_votes[p] / tot_target_valid for p in MODEL_PARTIES_9}

    # 2. Run Projection
    proj_res = project_constituency_votes(
        national_vote_shares=nat_target_shares,
        baseline_year=baseline_year,
        target_year=target_year,
        mode=mode,
        total_national_votes=tot_target_valid if mode == "oracle" else None,
        processed_dir=p_dir,
    )

    # 3. Compute Constituency Party-Share MAEs
    actual_shares_matrix = np.zeros((len(OFFICIAL_CONSTITUENCY_CODES), len(MODEL_PARTIES_9)))
    proj_shares_matrix = np.zeros((len(OFFICIAL_CONSTITUENCY_CODES), len(MODEL_PARTIES_9)))

    code_to_idx = {c: i for i, c in enumerate(OFFICIAL_CONSTITUENCY_CODES)}
    party_to_idx = {p: i for i, p in enumerate(MODEL_PARTIES_9)}

    for _, r in sub_target.iterrows():
        c_code = f"{int(r['constituency_code']):02d}"
        p_code = str(r["party"])
        actual_shares_matrix[code_to_idx[c_code], party_to_idx[p_code]] = float(r["party_share"])

    for c_code, p_map in proj_res.constituency_votes.items():
        c_val = proj_res.constituency_valid_votes[c_code]
        for p_code, v in p_map.items():
            proj_shares_matrix[code_to_idx[c_code], party_to_idx[p_code]] = v / c_val if c_val > 0 else 0.0

    abs_diffs = np.abs(proj_shares_matrix - actual_shares_matrix)
    overall_mae = float(np.mean(abs_diffs))
    party_maes = {p: float(np.mean(abs_diffs[:, party_to_idx[p]])) for p in MODEL_PARTIES_9}

    # National Share error
    nat_share_errs = [
        abs(proj_res.national_vote_shares[p] - nat_target_shares[p]) for p in MODEL_PARTIES_9
    ]
    max_nat_share_err = float(max(nat_share_errs))

    # Constituency Valid Votes error
    actual_const_valid = np.zeros(len(OFFICIAL_CONSTITUENCY_CODES))
    proj_const_valid = np.zeros(len(OFFICIAL_CONSTITUENCY_CODES))
    sub_el_t = el_df[el_df["election_year"] == target_year]
    for _, r in sub_el_t.iterrows():
        c_code = f"{int(r['constituency_code']):02d}"
        actual_const_valid[code_to_idx[c_code]] = float(r["valid_votes"])
        proj_const_valid[code_to_idx[c_code]] = float(proj_res.constituency_valid_votes[c_code])

    valid_diffs = np.abs(proj_const_valid - actual_const_valid)
    max_valid_err = float(np.max(valid_diffs))
    valid_mape = float(np.mean(valid_diffs / actual_const_valid)) * 100.0

    # 4. Run Mandate Allocator on projected constituencies
    fixed_cfg = FIXED_SEATS_2018 if target_year == 2018 else FIXED_SEATS_2022
    alloc_in = proj_res.to_allocator_input()
    alloc_res = allocate_riksdag_seats(
        constituency_votes=alloc_in,
        fixed_seats_by_constituency=fixed_cfg,
    )

    # 5. Compare with Certified Seats
    mandates_csv = p_dir.parents[0] / "mandates" / "historical_certified_mandates.csv"
    m_df = pd.read_csv(mandates_csv)
    sub_m = m_df[m_df["election_year"] == target_year]
    cert_seats = {
        p: int(sub_m[sub_m["party"] == p]["total_seats"].sum())
        for p in ["M", "L", "C", "KD", "S", "V", "MP", "SD"]
    }

    seat_diffs = {}
    for p in ["M", "L", "C", "KD", "S", "V", "MP", "SD"]:
        calc_s = alloc_res.final_seats_by_party.get(p, 0)
        cert_s = cert_seats.get(p, 0)
        seat_diffs[p] = calc_s - cert_s

    total_seat_err = sum(abs(v) for v in seat_diffs.values())

    return HistoricalEvaluationMetrics(
        baseline_year=baseline_year,
        target_year=target_year,
        mode=mode,
        constituency_share_mae=overall_mae,
        party_share_maes=party_maes,
        national_share_max_err=max_nat_share_err,
        constituency_valid_votes_max_err=max_valid_err,
        constituency_valid_votes_mape=valid_mape,
        seat_differences=seat_diffs,
        total_seat_error=total_seat_err,
        projected_seats={p: alloc_res.final_seats_by_party.get(p, 0) for p in ["M", "L", "C", "KD", "S", "V", "MP", "SD"]},
        certified_seats=cert_seats,
        ipf_iterations=proj_res.ipf_result.iterations,
    )


def run_all_historical_evaluations() -> dict[str, HistoricalEvaluationMetrics]:
    """Run historical evaluations for 2014->2018 and 2018->2022 in both oracle and production modes."""
    results: dict[str, HistoricalEvaluationMetrics] = {}

    configs = [
        ("2014_to_2018_oracle", 2014, 2018, "oracle"),
        ("2014_to_2018_production", 2014, 2018, "production"),
        ("2022_oracle", 2018, 2022, "oracle"),
        ("2022_production", 2018, 2022, "production"),
    ]

    for label, base_yr, tgt_yr, mode in configs:
        res = evaluate_projection_pair(base_yr, tgt_yr, mode=mode)
        results[label] = res

    return results
