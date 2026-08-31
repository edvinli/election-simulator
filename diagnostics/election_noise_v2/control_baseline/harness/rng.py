"""Frozen paired-randomness scheme for the ElectionNoise v2 evaluation harness.

The harness must guarantee that for one ``(case, horizon, seed)`` triple,
CONTROL, Challenger A and Challenger B receive **identical draws from every
frozen upstream component** and differ only in the ElectionNoise draw law. This
module states that scheme once, so a later challenger implementation cannot
quietly change it.

Nothing here is new machinery: every derivation below is an existing production
function, reused. The scheme is a documentation and enforcement layer.

--------------------------------------------------------------------------------
Streams and their derivations
--------------------------------------------------------------------------------

All production sub-seeds come from the same SHA-256 token convention::

    token  = f"{base_seed}:{origin_date.isoformat()}:{horizon_days}:{label}"
    subseed = int(sha256(token).hexdigest()[:8], 16) % 2_147_483_647

===============  ==========================================  =======================
Stream           Production derivation                       Depends on noise model?
===============  ==========================================  =======================
OpinionState     ``derive_opinion_state_seed(seed, as_of)``   **No**
Dynamics         ``derive_shared_dynamics_seed(seed, as_of,   **No**
                 horizon_days)``
ElectionNoise    ``derive_vote_share_layer_seeds(seed,        **Yes** — this is the
  index/sign     as_of, horizon_days)``                       only stream under test
Geography        none — fully deterministic                    No
Integerization   none — fully deterministic                    No
Allocator        keyed deterministic lottery, seeded from      No
                 canonical legal state
===============  ==========================================  =======================

Because the OpinionState and Dynamics tokens contain **no model identifier**, two
models run at the same ``(as_of, horizon_days, base_seed)`` observe bit-identical
`base_comp_matrix` values. Pairing is therefore a property of the existing seed
derivation, not something the harness has to arrange. :func:`assert_paired_base`
turns that property into an executable check.

Challengers must draw their own randomness from the tokens the preregistration
reserves for them (``election_noise_v2_a_index``, ``election_noise_v2_a_kernel``,
``election_noise_v2_a_loeo``, ``election_noise_v2_b_normal``) and from **no
others**. In particular a challenger may not reuse or perturb the OpinionState or
Dynamics tokens, and may not consume from CONTROL's ``residual_index`` /
``sign_draw`` streams.

--------------------------------------------------------------------------------
Tier-1 origin convention
--------------------------------------------------------------------------------

Tier 1 has no ``as_of`` in the production sense: its base composition is the
deterministic 14-day pre-election polling consensus, with no upstream randomness
at all. Pairing is therefore automatic. To keep the seed derivation deterministic
and stated, Tier 1 uses::

    origin_date   = election_date
    horizon_days  = CANONICAL_WINDOW_DAYS (14)

This choice is arbitrary but frozen here; it affects only which sub-seed the
ElectionNoise index stream starts from, never the shape of the predictive law.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from scripts.election_layer_v2.config import CANONICAL_WINDOW_DAYS
from scripts.hindcasts.models import (
    derive_opinion_state_seed,
    derive_shared_dynamics_seed,
)
from scripts.vote_share_calibration.models import derive_vote_share_layer_seeds

#: The frozen seed set (preregistration §D0). Never reorder, extend or subset.
FROZEN_SEEDS: tuple[int, ...] = (12345, 24680, 98765, 54321, 13579)

#: Draws per seed, per case, per model (preregistration §D0).
DRAWS_PER_SEED: int = 20_000

#: Seed tokens reserved for challengers; CONTROL must never consume from these.
CHALLENGER_RESERVED_TOKENS: tuple[str, ...] = (
    "election_noise_v2_a_index",
    "election_noise_v2_a_kernel",
    "election_noise_v2_a_loeo",
    "election_noise_v2_b_normal",
)

#: Tokens CONTROL consumes (production, unchanged).
CONTROL_TOKENS: tuple[str, ...] = ("residual_index", "sign_draw")


@dataclass(frozen=True)
class StreamSeeds:
    """Every sub-seed for one (case, horizon, base seed), split by model dependence."""

    base_seed: int
    origin_date: date
    horizon_days: int
    opinion_state_seed: int
    dynamics_seed: int
    election_noise_index_seed: int
    election_noise_sign_seed: int

    @property
    def upstream(self) -> tuple[int, int]:
        """Sub-seeds that must be identical across CONTROL / A / B."""
        return (self.opinion_state_seed, self.dynamics_seed)


def stream_seeds(base_seed: int, origin_date: date, horizon_days: int) -> StreamSeeds:
    """Derive every production sub-seed for one evaluation draw block."""
    idx_seed, sign_seed = derive_vote_share_layer_seeds(
        base_seed=base_seed, origin_date=origin_date, horizon_days=horizon_days
    )
    return StreamSeeds(
        base_seed=base_seed,
        origin_date=origin_date,
        horizon_days=horizon_days,
        opinion_state_seed=derive_opinion_state_seed(base_seed=base_seed, origin_date=origin_date),
        dynamics_seed=derive_shared_dynamics_seed(
            base_seed=base_seed, origin_date=origin_date, horizon_days=horizon_days
        ),
        election_noise_index_seed=idx_seed,
        election_noise_sign_seed=sign_seed,
    )


def tier1_origin(election_date: date) -> tuple[date, int]:
    """Frozen Tier-1 origin convention (see module docstring)."""
    return election_date, CANONICAL_WINDOW_DAYS


def control_residual_indices(index_seed: int, k: int, n: int) -> np.ndarray:
    """CONTROL's residual-atom index draw, bit-identical to production.

    Production performs ``np.random.default_rng(index_seed).integers(0, k, size=n)``
    on a dedicated generator instance, so reproducing it here cannot perturb any
    other stream. This is the same reconstruction validated in Part 1.
    """
    return np.random.default_rng(index_seed).integers(0, k, size=n)


def assert_paired_base(base_a: np.ndarray, base_b: np.ndarray, label: str = "") -> None:
    """Fail loudly if two models' upstream draws are not bit-identical."""
    if base_a.shape != base_b.shape:
        raise AssertionError(f"paired base shape mismatch {label}: {base_a.shape} vs {base_b.shape}")
    if not np.array_equal(base_a, base_b):
        bad = int(np.sum(np.any(base_a != base_b, axis=1)))
        raise AssertionError(
            f"paired base draws differ {label}: {bad} of {base_a.shape[0]} rows are not identical"
        )
