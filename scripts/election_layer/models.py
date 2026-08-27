"""Chronological election residual pool extraction and paired election-layer model variants."""

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
from scripts.pollofpolls.clr import clr_to_composition_matrix, composition_to_clr

from .config import (
    ALL_HISTORICAL_ELECTIONS,
    CANONICAL_WINDOW_DAYS,
    ELECTION_LAYER_VARIANTS,
)


@dataclass(frozen=True)
class ChronologicalTrainingResiduals:
    """Historical election residual training set available strictly prior to target election."""

    target_election_year: int
    training_years: tuple[int, ...]
    residuals_matrix: np.ndarray  # Shape (K, 9)
    mean_bias_clr: np.ndarray      # Shape (9,)
    centered_residuals_matrix: np.ndarray  # Shape (K, 9)


def derive_election_layer_seed(base_seed: int, origin_date: date, horizon_days: int) -> int:
    """Derive deterministic SHA-256 seed for paired election-layer residual sampling."""
    token = f"{base_seed}:{origin_date.isoformat()}:{horizon_days}:election_layer".encode("utf-8")
    digest = hashlib.sha256(token).hexdigest()
    return int(digest[:8], 16) % 2_147_483_647


def load_chronological_training_residuals(
    target_election_year: int,
    window_days: int = CANONICAL_WINDOW_DAYS,
    polls_file: Path | str | None = None,
    elections_file: Path | str | None = None,
) -> ChronologicalTrainingResiduals:
    """Extract historical 9-dimensional CLR residuals strictly for elections prior to target_election_year.

    Enforces strict chronological boundary:
        For 2018 -> exactly {2002, 2006, 2010, 2014}
        For 2022 -> exactly {2002, 2006, 2010, 2014, 2018}
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

        clr_target, _ = composition_to_clr(target_comp, categories=ALL_CATEGORIES)
        clr_consensus, _ = composition_to_clr(consensus.consensus_composition, categories=ALL_CATEGORIES)
        clr_res = clr_target - clr_consensus

        residuals_list.append(clr_res)
        training_years.append(el_date.year)

    residuals_mat = np.array(residuals_list, dtype=float)  # Shape (K, 9)
    mean_bias = np.mean(residuals_mat, axis=0)             # Shape (9,)
    centered_mat = residuals_mat - mean_bias               # Shape (K, 9)

    return ChronologicalTrainingResiduals(
        target_election_year=target_election_year,
        training_years=tuple(training_years),
        residuals_matrix=residuals_mat,
        mean_bias_clr=mean_bias,
        centered_residuals_matrix=centered_mat,
    )


def apply_election_layer_variants(
    base_clr_matrix: np.ndarray,
    training_pool: ChronologicalTrainingResiduals,
    samples_count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Generate predictive composition matrices across all 4 election layer variants.

    Parameters:
        base_clr_matrix: Paired CLR sample matrix of shape (samples_count, 9) from state_plus_dynamics.
        training_pool: ChronologicalTrainingResiduals containing historical residuals prior to target election.
        samples_count: Number of Monte Carlo draws (default: 5000).
        seed: Deterministic integer seed for election-layer residual index sampling.

    Returns:
        dict mapping variant ID ('base', 'bias_only', 'noise_only', 'bias_plus_noise')
        to composition matrix of shape (samples_count, 9) summing to 100.0%.
    """
    k = len(training_pool.training_years)
    if k == 0:
        raise ValueError("Cannot apply election layer with empty training pool")

    # 1. Base variant (no election residual)
    comp_base = clr_to_composition_matrix(base_clr_matrix)

    # 2. Bias only (fixed training mean CLR residual)
    clr_bias_only = base_clr_matrix + training_pool.mean_bias_clr
    comp_bias_only = clr_to_composition_matrix(clr_bias_only)

    # 3. Sample historical election indices with replacement
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(0, k, size=samples_count)

    # 4. Noise only (centered residuals: r_e - mean_r)
    sampled_centered_noise = training_pool.centered_residuals_matrix[sampled_indices]
    clr_noise_only = base_clr_matrix + sampled_centered_noise
    comp_noise_only = clr_to_composition_matrix(clr_noise_only)

    # 5. Bias plus noise (raw residuals: r_e)
    # Using the exact same sampled_indices so difference isolates mean_bias_clr!
    sampled_raw_residuals = training_pool.residuals_matrix[sampled_indices]
    clr_bias_plus_noise = base_clr_matrix + sampled_raw_residuals
    comp_bias_plus_noise = clr_to_composition_matrix(clr_bias_plus_noise)

    return {
        "base": comp_base,
        "bias_only": comp_bias_only,
        "noise_only": comp_noise_only,
        "bias_plus_noise": comp_bias_plus_noise,
    }
