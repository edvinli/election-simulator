"""Diagnostic runner for the ElectionNoise coalition-seat mixture investigation.

This module NEVER modifies production code. It calls the frozen production entry
point ``scripts.simulator.engine.simulate_election`` and captures already-computed
intermediates through passive wrappers that draw no randomness of their own.

Modes
-----
production
    Canonical production run, instrumented to retain ``base_comp_matrix``
    (post OpinionState + Dynamics, pre ElectionNoise) and the sampled
    residual-election index per draw.

prenoise
    Same OpinionState + Dynamics draws, but the national composition handed to
    geography + mandate allocation is the pre-ElectionNoise composition,
    normalised with exactly the production normalisation
    (``x / rowsum``).  Every downstream component is the untouched production code.

loo:<year>
    Leave-one-election-out ElectionNoise.  The historical election list consumed by
    the production loader is filtered, and the production loader itself performs the
    re-centering.  All other components, seeds and data are unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.simulator import engine as sim_engine
from scripts.simulator.config import (
    DEFAULT_ELECTION_DATE,
    DEFAULT_SIMULATION_SAMPLES,
    DEFAULT_SIMULATION_SEED,
    MODEL_PARTIES_9,
    PARLIAMENTARY_PARTIES_8,
)
from scripts.vote_share_calibration import national_engine as nat_engine
from scripts.vote_share_calibration.models import (
    apply_vote_share_models as _real_apply_vote_share_models,
    derive_vote_share_layer_seeds,
)
from scripts.election_layer_v2 import residuals_pool as res_pool_mod


CAPTURE: dict[str, Any] = {}


def _capturing_apply_vote_share_models(
    base_comp_matrix, training_pool, samples_count, index_seed, sign_seed, eps
):
    """Passive wrapper: records inputs, consumes no randomness, returns the real result."""
    CAPTURE["base_comp_matrix"] = np.array(base_comp_matrix, copy=True)
    CAPTURE["training_years"] = tuple(training_pool.training_years)
    CAPTURE["centered_residuals_matrix"] = np.array(
        training_pool.centered_residuals_matrix, copy=True
    )
    CAPTURE["residuals_matrix"] = np.array(training_pool.residuals_matrix, copy=True)
    CAPTURE["mean_bias_pp"] = np.array(training_pool.mean_bias_pp, copy=True)
    CAPTURE["index_seed"] = int(index_seed)
    CAPTURE["sign_seed"] = int(sign_seed)
    CAPTURE["samples_count"] = int(samples_count)
    return _real_apply_vote_share_models(
        base_comp_matrix=base_comp_matrix,
        training_pool=training_pool,
        samples_count=samples_count,
        index_seed=index_seed,
        sign_seed=sign_seed,
        eps=eps,
    )


def _install_capture() -> None:
    nat_engine.apply_vote_share_models = _capturing_apply_vote_share_models


def _install_prenoise_substitution() -> None:
    """Replace the national engine result's final shares with the pre-ElectionNoise shares."""
    real_generate = nat_engine.generate_national_vote_shares

    def patched(*args, **kwargs):
        res = real_generate(*args, **kwargs)
        base = res.base_comp_matrix
        pre = base / np.sum(base, axis=1, keepdims=True)
        CAPTURE["prenoise_shares"] = np.array(pre, copy=True)
        CAPTURE["production_shares"] = np.array(res.nat_shares_matrix, copy=True)
        return replace(res, nat_shares_matrix=pre)

    sim_engine.generate_national_vote_shares = patched


def _install_loo(drop_year: int) -> None:
    """Filter the historical election list the production loader iterates over."""
    original = res_pool_mod.ALL_HISTORICAL_ELECTIONS
    filtered = tuple(d for d in original if d.year != drop_year)
    if len(filtered) != len(original) - 1:
        raise ValueError(f"Leave-one-out year {drop_year} not found in {original}")
    res_pool_mod.ALL_HISTORICAL_ELECTIONS = filtered


def recompute_residual_indices(index_seed: int, k: int, samples: int) -> np.ndarray:
    """Reproduce production's residual-election index draw exactly.

    Production performs ``np.random.default_rng(index_seed).integers(0, k, size=N)``
    on a dedicated generator instance, so recomputing it outside the production call
    is bit-identical and cannot perturb any other stream.
    """
    return np.random.default_rng(index_seed).integers(0, k, size=samples)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, help="production | prenoise | loo:<year>")
    parser.add_argument("--as-of", default="2026-08-24")
    parser.add_argument("--election-date", default=DEFAULT_ELECTION_DATE)
    parser.add_argument("--samples", type=int, default=DEFAULT_SIMULATION_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SIMULATION_SEED)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    mode = args.mode
    _install_capture()

    drop_year: int | None = None
    if mode == "prenoise":
        _install_prenoise_substitution()
    elif mode.startswith("loo:"):
        drop_year = int(mode.split(":", 1)[1])
        _install_loo(drop_year)
    elif mode != "production":
        raise SystemExit(f"unknown mode {mode}")

    t0 = time.perf_counter()
    result = sim_engine.simulate_election(
        as_of=args.as_of,
        election_date=args.election_date,
        samples=args.samples,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - t0

    k = len(CAPTURE["training_years"])
    residual_idx = recompute_residual_indices(CAPTURE["index_seed"], k, args.samples)
    residual_year = np.array(
        [CAPTURE["training_years"][i] for i in residual_idx], dtype=np.int64
    )

    payload: dict[str, Any] = {
        "mode": np.array(mode),
        "as_of": np.array(result.summary.as_of),
        "election_date": np.array(result.summary.election_date),
        "samples": np.array(args.samples),
        "seed": np.array(args.seed),
        "seats_matrix": result.seats_matrix.astype(np.int16),
        "vote_shares_pct": result.vote_shares_matrix.astype(np.float64),
        "residual_index": residual_idx.astype(np.int8),
        "residual_year": residual_year,
        "training_years": np.array(CAPTURE["training_years"], dtype=np.int64),
        "centered_residuals_matrix": CAPTURE["centered_residuals_matrix"],
        "raw_residuals_matrix": CAPTURE["residuals_matrix"],
        "mean_bias_pp": CAPTURE["mean_bias_pp"],
        "index_seed": np.array(CAPTURE["index_seed"]),
        "sign_seed": np.array(CAPTURE["sign_seed"]),
        "base_comp_matrix": CAPTURE["base_comp_matrix"].astype(np.float64),
        "parties_8": np.array(PARLIAMENTARY_PARTIES_8),
        "parties_9": np.array(MODEL_PARTIES_9),
        "elapsed_seconds": np.array(elapsed),
    }
    if drop_year is not None:
        payload["drop_year"] = np.array(drop_year)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    print(f"[{mode}] samples={args.samples} elapsed={elapsed:.1f}s -> {out_path}")
    print(f"[{mode}] training_years={CAPTURE['training_years']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
