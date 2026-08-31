"""Run the full production simulation under a selectable ElectionNoise law.

The vote-to-seat half of the pipeline — geographic IPF projection, exact-margin
biproportional controlled rounding and the Sainte-Laguë mandate allocator — is used
**exactly as it stands** in ``scripts/simulator/engine.py``. None of it is
re-implemented here, because re-implementing it would be precisely the kind of
silent divergence the promotion is supposed to rule out.

``simulate_election`` resolves its national vote shares through the module-level
name ``generate_national_vote_shares`` in the engine's namespace. This runner
substitutes a drop-in replacement for that one name for the duration of a single
call, so the unmodified engine executes with the adopted law's vote shares and
everything downstream is bit-for-bit the production path.

Since the production default was flipped, ``simulate_election`` itself accepts
``noise_model`` and defaults to the adopted law. This runner remains as the
convenience entry point that also returns the national result and the fitted
covariance detail, which ``SimulationResult`` does not carry.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from scripts.vote_share_calibration.election_noise_b import LEGACY_MODEL_ID, MODEL_ID
from scripts.vote_share_calibration.production_national_engine import (
    generate_production_vote_shares,
)

from .engine import SimulationResult, simulate_election


def simulate_election_with_noise_model(
    model_id: str = MODEL_ID,
    *,
    as_of: str | date | None = None,
    election_date: str | date = "2026-09-13",
    samples: int = 100_000,
    seed: int = 12345,
    **engine_kwargs: Any,
) -> tuple[SimulationResult, Any, Any]:
    """Run the unmodified production engine under ``model_id``.

    Returns ``(SimulationResult, national_result, ElectionNoiseBDetail | None)``. The
    national result is captured for both laws so the shared latent state and the
    transfer attenuation lambda are available to diagnostics; ``SimulationResult``
    itself does not carry them.
    """
    if model_id not in (MODEL_ID, LEGACY_MODEL_ID):
        raise ValueError(f"unknown ElectionNoise model_id {model_id!r}")

    captured: dict[str, Any] = {}

    def _national(as_of, election_date, samples, seed, **kw):
        result, detail = generate_production_vote_shares(
            as_of=as_of, election_date=election_date, samples=samples, seed=seed,
            model_id=model_id,
        )
        captured["national"] = result
        captured["detail"] = detail
        return result

    from scripts.simulator import engine as _engine

    original = _engine.generate_national_vote_shares
    try:
        _engine.generate_national_vote_shares = _national
        result = simulate_election(as_of=as_of, election_date=election_date,
                                   samples=samples, seed=seed,
                                   noise_model=model_id, **engine_kwargs)
    finally:
        _engine.generate_national_vote_shares = original
    return result, captured["national"], captured.get("detail")
