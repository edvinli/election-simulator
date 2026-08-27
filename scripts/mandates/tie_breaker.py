"""Tie-breaker protocol and deterministic lottery implementations according to Swedish electoral law."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import random
from typing import Any, Protocol, Sequence, TypeVar

T = TypeVar("T")


class TieBreaker(Protocol):
    """Protocol for resolving ties by lottery (lotten) under Vallagen 14 kap. 2 §."""

    def pick_winner(self, candidates: Sequence[T], context: dict[str, Any] | None = None) -> T:
        """Select a single winning candidate from a sequence of tied candidate identifiers."""
        ...


class DeterministicLotteryTieBreaker:
    """Deterministic, reproducible lottery tie-breaker using SHA-256 hashing over legal allocation state."""

    def __init__(self, seed: int = 12345) -> None:
        self.seed = seed

    def pick_winner(self, candidates: Sequence[T], context: dict[str, Any] | None = None) -> T:
        """Select a deterministic winning candidate based on legal state hash."""
        if not candidates:
            raise ValueError("Cannot break tie among empty candidate sequence")
        if len(candidates) == 1:
            return candidates[0]

        # Canonical context serialization
        ctx_parts: list[str] = [f"seed={self.seed}"]
        if context:
            for k in sorted(context.keys()):
                val = context[k]
                if isinstance(val, Fraction):
                    v_str = f"{val.numerator}/{val.denominator}"
                elif isinstance(val, (dict, list, tuple)):
                    v_str = json.dumps(val, sort_keys=True)
                else:
                    v_str = str(val)
                ctx_parts.append(f"{k}={v_str}")

        cand_strings = [str(c) for c in candidates]
        sorted_cands = sorted(cand_strings)
        ctx_parts.append(f"tied={','.join(sorted_cands)}")

        canonical_token = "|".join(ctx_parts).encode("utf-8")
        digest = hashlib.sha256(canonical_token).hexdigest()
        lottery_index = int(digest[:8], 16) % len(candidates)

        # Sort candidate objects using string representation for deterministic indexing
        sorted_candidate_objs = sorted(candidates, key=lambda c: str(c))
        return sorted_candidate_objs[lottery_index]


class SeededRandomTieBreaker:
    """Deterministic random-generator tie-breaker for randomized testing."""

    def __init__(self, seed: int = 12345) -> None:
        self.rng = random.Random(seed)

    def pick_winner(self, candidates: Sequence[T], context: dict[str, Any] | None = None) -> T:
        if not candidates:
            raise ValueError("Cannot break tie among empty candidate sequence")
        if len(candidates) == 1:
            return candidates[0]
        sorted_candidates = sorted(candidates, key=lambda c: str(c))
        return self.rng.choice(sorted_candidates)
