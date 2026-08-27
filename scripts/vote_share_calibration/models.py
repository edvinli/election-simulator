"""Vote-share model definitions and paired Monte Carlo sampling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
from typing import Any
import numpy as np

from scripts.election_layer_v2.residuals_pool import ChronologicalPPResidualsPool
from scripts.election_layer_v2.transfer import apply_batch_simplex_transfer

from .config import MIN_SHARE_PCT


def derive_vote_share_layer_seeds(
    base_seed: int,
    origin_date: date,
    horizon_days: int,
) -> tuple[int, int]:
    """Derive deterministic SHA-256 seeds for paired residual index and sign draws.

    Returns:
        (index_seed, sign_seed)
    """
    token_idx = f"{base_seed}:{origin_date.isoformat()}:{horizon_days}:residual_index".encode("utf-8")
    digest_idx = hashlib.sha256(token_idx).hexdigest()
    idx_seed = int(digest_idx[:8], 16) % 2_147_483_647

    token_sign = f"{base_seed}:{origin_date.isoformat()}:{horizon_days}:sign_draw".encode("utf-8")
    digest_sign = hashlib.sha256(token_sign).hexdigest()
    sign_seed = int(digest_sign[:8], 16) % 2_147_483_647

    return idx_seed, sign_seed


def apply_vote_share_models(
    base_comp_matrix: np.ndarray,
    training_pool: ChronologicalPPResidualsPool,
    samples_count: int,
    index_seed: int,
    sign_seed: int,
    eps: float = MIN_SHARE_PCT,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Apply base, pp_centered_noise, and pp_symmetric_noise models with strict architectural pairing.

    Parameters:
        base_comp_matrix: Array of shape (N, 9) representing paired state_plus_dynamics compositions.
        training_pool: ChronologicalPPResidualsPool containing historical residuals prior to target election.
        samples_count: Number of Monte Carlo draws N.
        index_seed: Deterministic seed for sampling historical election index k in {0, ..., K-1}.
        sign_seed: Deterministic seed for independent sign draw S in {-1, +1}.
        eps: Minimum party share bound (default: 0.01%).

    Returns:
        dict mapping model ID to (transferred_matrix, lambdas_array).
    """
    k = len(training_pool.training_years)
    if k == 0:
        raise ValueError("Cannot apply vote-share layer with empty training pool")

    # 1. Base model
    comp_base = base_comp_matrix
    lambdas_base = np.ones(samples_count, dtype=float)

    # 2. Draw shared historical election indices for sample i
    rng_idx = np.random.default_rng(index_seed)
    sampled_indices = rng_idx.integers(0, k, size=samples_count)

    # 3. Draw independent signs S in {-1, +1} with equal probability
    rng_sign = np.random.default_rng(sign_seed)
    sampled_signs = rng_sign.choice([-1.0, 1.0], size=samples_count, p=[0.5, 0.5])

    # 4. pp_centered_noise: r_e - mean_r
    res_centered = training_pool.centered_residuals_matrix[sampled_indices]
    comp_centered, lambdas_centered = apply_batch_simplex_transfer(
        base_comp_matrix, res_centered, eps=eps
    )

    # 5. pp_symmetric_noise: S * r_e (shared index, independent sign)
    res_raw = training_pool.residuals_matrix[sampled_indices]
    res_symmetric = sampled_signs[:, None] * res_raw
    comp_symmetric, lambdas_symmetric = apply_batch_simplex_transfer(
        base_comp_matrix, res_symmetric, eps=eps
    )

    return {
        "base": (comp_base, lambdas_base),
        "pp_centered_noise": (comp_centered, lambdas_centered),
        "pp_symmetric_noise": (comp_symmetric, lambdas_symmetric),
    }
