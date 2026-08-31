"""Deterministic, model-specific RNG streams for the preregistered challengers.

The preregistration reserves exactly four seed tokens for challenger use
(``CHALLENGER_RESERVED_TOKENS`` in the frozen Part-3 harness). This module is the
only place a challenger stream may be created, and it refuses any label outside
that reserved list, so a challenger cannot silently consume CONTROL's
``residual_index`` / ``sign_draw`` streams or invent a token of its own.

Seed derivation is the unchanged production convention::

    token   = f"{base_seed}:{origin_date.isoformat()}:{horizon_days}:{label}"
    subseed = int(sha256(token).hexdigest()[:8], 16) % 2_147_483_647

Sub-streams within one token
----------------------------
LOEO-FIT needs an independent stream per (bandwidth, fold, seed, role) without
inventing new tokens. The reserved token supplies the ``SeedSequence`` entropy and
the coordinate tuple supplies its ``spawn_key``, which is a documented,
deterministic and collision-free derivation. No wall clock, PID, environment value
or unordered-set iteration enters any path here.

Common random numbers
---------------------
None are imposed. Challenger A's atom index has the same marginal law as
CONTROL's, so reusing CONTROL's stream would be a mathematically valid pairing,
but the preregistration reserves ``election_noise_v2_a_index`` as A's own stream
and that reservation is honoured. A's index and kernel streams, and B's Gaussian
stream, are mutually independent; no artificial coupling is introduced to make any
comparison look more stable.
"""

from __future__ import annotations

from datetime import date
import hashlib
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.election_noise_v2.control_baseline.harness.rng import (
    CHALLENGER_RESERVED_TOKENS,
    CONTROL_TOKENS,
)

#: Reserved labels, re-exported from the frozen harness so they cannot drift apart.
A_INDEX = "election_noise_v2_a_index"
A_KERNEL = "election_noise_v2_a_kernel"
A_LOEO = "election_noise_v2_a_loeo"
B_NORMAL = "election_noise_v2_b_normal"

_MODULUS = 2_147_483_647


class ForbiddenSeedToken(RuntimeError):
    """Raised when a challenger asks for a stream it is not permitted to consume."""


def challenger_subseed(base_seed: int, origin_date: date, horizon_days: int, label: str) -> int:
    """Production sub-seed derivation, restricted to reserved challenger tokens."""
    if label in CONTROL_TOKENS:
        raise ForbiddenSeedToken(
            f"'{label}' is a CONTROL stream; challengers must not consume it "
            f"(reserved challenger tokens: {list(CHALLENGER_RESERVED_TOKENS)})"
        )
    if label not in CHALLENGER_RESERVED_TOKENS:
        raise ForbiddenSeedToken(
            f"'{label}' is not a reserved challenger token; the preregistration permits "
            f"only {list(CHALLENGER_RESERVED_TOKENS)}"
        )
    token = f"{base_seed}:{origin_date.isoformat()}:{horizon_days}:{label}".encode("utf-8")
    return int(hashlib.sha256(token).hexdigest()[:8], 16) % _MODULUS


def challenger_rng(
    base_seed: int,
    origin_date: date,
    horizon_days: int,
    label: str,
    spawn_key: Sequence[int] = (),
) -> np.random.Generator:
    """A deterministic generator for one reserved challenger stream.

    ``spawn_key`` selects an independent sub-stream of the same token, used by
    LOEO-FIT to separate (bandwidth, fold, seed, role) without new tokens.
    """
    sub = challenger_subseed(base_seed, origin_date, horizon_days, label)
    return np.random.default_rng(np.random.SeedSequence(entropy=sub, spawn_key=tuple(spawn_key)))
