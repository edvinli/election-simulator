"""Model wrappers for historical seat hindcasts: point baseline and full ElectionSimulator v1."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any
import numpy as np

from scripts.geography.config import (
    DEFAULT_PROCESSED_GEOGRAPHY_DIR,
    MODEL_PARTIES_9,
    OFFICIAL_CONSTITUENCY_CODES,
)
from scripts.geography.integerization import biproportional_controlled_rounding
from scripts.geography.projection import _get_cached_geography_structures
from scripts.mandates.config import FIXED_SEATS_2018, FIXED_SEATS_2022, FIXED_SEATS_2026
from scripts.pollofpolls.state import estimate_opinion
from scripts.pollofpolls.state_config import ALL_CATEGORIES
from scripts.simulator.config import PARLIAMENTARY_PARTIES_8
from scripts.simulator.engine import (
    SimulationResult,
    _apportion_constituency_units_of_25,
    _apportion_national_party_integers,
    simulate_election,
)
from scripts.simulator.fast_allocator import fast_allocate_seats_from_matrix


def evaluate_seat_point_baseline(
    as_of: date,
    election_date: date,
    baseline_year: int,
    total_national_votes: int = 6_500_000,
    geography_mode: str = "chronological",
    data_dir: Path | str | None = None,
) -> dict[str, int]:
    """Evaluate deterministic point-forecast baseline.

    Pipeline:
        1. Point polling consensus at origin date (theta_origin from OpinionState v1.1).
        2. Geographic Projection v1 (IPF raking from historical baseline B to target R).
        3. Exact-margin controlled rounding to integer votes.
        4. Exact Riksdag mandate allocation.
    """
    root_data = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "data" / "processed"
    target_year = election_date.year

    # 1. Opinion State consensus point forecast
    op_state = estimate_opinion(as_of=as_of, data_dir=root_data / "pollofpolls")
    # Take mean percentages [0, 100] and convert to fractions [0, 1]
    point_shares_9 = np.array([op_state.mean_pct.get(cat, op_state.rest_pct) for cat in ALL_CATEGORIES], dtype=np.float64) / 100.0
    point_shares_9 = point_shares_9 / np.sum(point_shares_9)

    # 2. Geographic structures
    p_geo_dir = root_data / "geography"
    B_cached, R_cached = _get_cached_geography_structures(
        baseline_year=baseline_year,
        target_year=target_year,
        mode=geography_mode,
        processed_dir_str=str(p_geo_dir),
    )
    B_base = B_cached.copy()
    R_base = R_cached.copy()

    R_int = _apportion_constituency_units_of_25(R_base, total_national_votes)
    R_col_vec = R_int[:, np.newaxis].astype(np.float64)
    C_int = _apportion_national_party_integers(point_shares_9, total_national_votes)
    C_row_vec = C_int[np.newaxis, :].astype(np.float64)

    # 3. IPF raking
    X = B_base.copy()
    for _ in range(8):
        row_sums = np.sum(X, axis=1, keepdims=True)
        X *= R_col_vec / np.maximum(row_sums, 1e-12)
        col_sums = np.sum(X, axis=0, keepdims=True)
        X *= C_row_vec / np.maximum(col_sums, 1e-12)

    # 4. Exact-margin integerization
    cr_res = biproportional_controlled_rounding(X, R_int, C_int, solver="auto")
    int_mat = cr_res.rounded_matrix

    # 5. Mandate allocation with official fixed seats
    if target_year == 2018:
        fixed_seats_dict = FIXED_SEATS_2018
    elif target_year == 2022:
        fixed_seats_dict = FIXED_SEATS_2022
    else:
        fixed_seats_dict = FIXED_SEATS_2026
    fixed_seats_arr = np.array([fixed_seats_dict[c] for c in OFFICIAL_CONSTITUENCY_CODES], dtype=np.int64)

    return fast_allocate_seats_from_matrix(int_mat, fixed_seats_arr=fixed_seats_arr)


def evaluate_election_simulator_v1(
    as_of: date,
    election_date: date,
    baseline_year: int,
    samples: int = 5_000,
    seed: int = 12345,
    geography_mode: str = "chronological",
    data_dir: Path | str | None = None,
) -> SimulationResult:
    """Evaluate frozen ElectionSimulator v1 historically."""
    root_data = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "data" / "processed"
    return simulate_election(
        as_of=as_of,
        election_date=election_date,
        samples=samples,
        seed=seed,
        baseline_year=baseline_year,
        geography_mode=geography_mode,
        processed_geo_dir=root_data / "geography",
    )
