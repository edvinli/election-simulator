"""Chronological percentage-point election residual pool extraction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd

from scripts.election_residuals.config import (
    ALL_CATEGORIES,
    DEFAULT_ELECTIONS_FILE,
    DEFAULT_POLLS_FILE,
)
from scripts.election_residuals.consensus import build_election_polling_consensus
from scripts.elections.load import load_election_targets_for_forecasting

from .config import (
    ALL_HISTORICAL_ELECTIONS,
    CANONICAL_WINDOW_DAYS,
)


@dataclass(frozen=True)
class ChronologicalPPResidualsPool:
    """Historical percentage-point residual training pool strictly prior to target election."""

    target_election_year: int
    training_years: tuple[int, ...]
    residuals_matrix: np.ndarray  # Shape (K, 9) in percentage points
    mean_bias_pp: np.ndarray      # Shape (9,) in percentage points
    centered_residuals_matrix: np.ndarray  # Shape (K, 9) in percentage points


def derive_election_layer_v2_seed(base_seed: int, origin_date: date, horizon_days: int) -> int:
    """Derive deterministic SHA-256 seed for paired election-layer residual sampling."""
    token = f"{base_seed}:{origin_date.isoformat()}:{horizon_days}:election_layer_v2".encode("utf-8")
    digest = hashlib.sha256(token).hexdigest()
    return int(digest[:8], 16) % 2_147_483_647


def load_chronological_pp_residuals(
    target_election_year: int,
    window_days: int = CANONICAL_WINDOW_DAYS,
    polls_file: Path | str | None = None,
    elections_file: Path | str | None = None,
) -> ChronologicalPPResidualsPool:
    """Extract historical 9-category percentage-point residuals strictly prior to target_election_year.

    Enforces strict chronological boundary:
        For 2010 -> exactly {2002, 2006} (K=2)
        For 2014 -> exactly {2002, 2006, 2010} (K=3)
        For 2018 -> exactly {2002, 2006, 2010, 2014} (K=4)
        For 2022 -> exactly {2002, 2006, 2010, 2014, 2018} (K=5)
    """
    p_file = Path(polls_file) if polls_file else DEFAULT_POLLS_FILE
    e_file = Path(elections_file) if elections_file else DEFAULT_ELECTIONS_FILE

    polls_df = pd.read_csv(p_file)
    election_targets = load_election_targets_for_forecasting(e_file)

    eligible_elections = [
        el for el in sorted(ALL_HISTORICAL_ELECTIONS) if el.year < target_election_year
    ]

    if not eligible_elections:
        raise ValueError(f"No historical elections available prior to {target_election_year}")

    residuals_list: list[np.ndarray] = []
    training_years: list[int] = []

    for el_date in eligible_elections:
        target_comp = election_targets[el_date]
        consensus = build_election_polling_consensus(el_date, polls_df, window_days=window_days)

        t_vec = np.array([target_comp[c] for c in ALL_CATEGORIES], dtype=float)
        c_vec = np.array([consensus.consensus_composition[c] for c in ALL_CATEGORIES], dtype=float)
        r_vec = t_vec - c_vec

        # Zero-sum validation
        res_sum = float(np.sum(r_vec))
        if abs(res_sum) > 0.05:
            raise ValueError(f"Election {el_date.year} residual sum ({res_sum:.6f}) deviates from zero")

        # Clean tiny floating precision residue
        if abs(res_sum) > 1e-12:
            r_vec = r_vec - (res_sum / len(r_vec))

        residuals_list.append(r_vec)
        training_years.append(el_date.year)

    residuals_mat = np.array(residuals_list, dtype=float)  # Shape (K, 9)
    mean_bias = np.mean(residuals_mat, axis=0)             # Shape (9,)

    # Mean bias zero-sum cleaning
    mb_sum = float(np.sum(mean_bias))
    if abs(mb_sum) > 1e-12:
        mean_bias = mean_bias - (mb_sum / len(mean_bias))

    centered_mat = residuals_mat - mean_bias               # Shape (K, 9)

    return ChronologicalPPResidualsPool(
        target_election_year=target_election_year,
        training_years=tuple(training_years),
        residuals_matrix=residuals_mat,
        mean_bias_pp=mean_bias,
        centered_residuals_matrix=centered_mat,
    )
