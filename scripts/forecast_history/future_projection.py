"""Conditional forward projection for the coalition forecast history chart.

This module deliberately keeps hypothetical future chart points separate from
``history["series"]``.  Every projection point freezes the polling cutoff at
the latest certified production date and changes only the remaining Dynamics
v2 horizon.  Future polls are therefore never synthesized or admitted.

The projection uses the ordinary ElectionSimulator path, including the adopted
ElectionNoise law, geographic projection, controlled rounding, and statutory
mandate allocator.  It is a conditional visualization, not a forecast of
future polling observations.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from scripts.simulator.config import DEFAULT_SIMULATION_SEED
from scripts.simulator.engine import simulate_election

from .contract import (
    DEFAULT_COALITIONS,
    build_groups_from_matrices,
    deterministic_history_sha256,
    validate_history_contract,
)
from .generate import update_history_with_production_result as _update_history_with_production_result


DEFAULT_PROJECTION_SAMPLES = 10_000
PROJECTION_ASSUMPTION = "frozen_opinion_state_shrinking_dynamics_horizon"
PROJECTION_LEGEND_SV = "Framåtblickande projektion"
LATEST_FORECAST_LABEL_SV = "Senaste prognos"
ELECTION_DAY_LABEL_SV = "Valdag 13 sep"

_MONTHS_SV = (
    "jan",
    "feb",
    "mar",
    "apr",
    "maj",
    "jun",
    "jul",
    "aug",
    "sep",
    "okt",
    "nov",
    "dec",
)


def _coerce_date(value: str | date, *, name: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date") from exc


def _short_date_sv(value: date) -> str:
    return f"{value.day} {_MONTHS_SV[value.month - 1]}"


def projection_tooltip_sv(origin_date: str | date) -> str:
    """Return the explicit conditional-projection disclosure shown by consumers."""

    origin = _coerce_date(origin_date, name="origin_date")
    return (
        f"Framåtblickande projektion från opinionsläget {_short_date_sv(origin)}. "
        "Antar oförändrat underliggande opinionsläge; framtida mätningar är okända."
    )


def build_future_projection(
    *,
    origin_date: str | date,
    election_date: str | date,
    anchor_point: Mapping[str, Any],
    samples: int = DEFAULT_PROJECTION_SAMPLES,
    seed: int = DEFAULT_SIMULATION_SEED,
    data_dir: Path | str | None = None,
    coalitions: Mapping[str, Sequence[str]] = DEFAULT_COALITIONS,
    simulation_runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Build daily conditional points strictly after ``origin_date``.

    ``as_of`` is held fixed at ``origin_date`` for every call.  The explicit
    ``dynamics_horizon_days`` argument alone decreases from the natural
    remaining horizon to zero on election day.  The canonical simulator then
    applies ElectionNoise, geography and mandates exactly as usual.

    ``simulation_runner`` is injectable for focused contract tests.  Production
    callers leave it unset so the canonical :func:`simulate_election` is used.
    """

    origin = _coerce_date(origin_date, name="origin_date")
    election = _coerce_date(election_date, name="election_date")
    if origin > election:
        raise ValueError("origin_date cannot occur after election_date")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError("samples must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(anchor_point, Mapping):
        raise ValueError("anchor_point must be the current production history point")
    if str(anchor_point.get("date")) != origin.isoformat():
        raise ValueError("anchor_point date must equal origin_date")
    if anchor_point.get("provenance") != "current_production":
        raise ValueError("anchor_point must be the current_production point")
    if not isinstance(anchor_point.get("groups"), Mapping):
        raise ValueError("anchor_point must contain joint coalition groups")

    runner = simulation_runner or simulate_election
    coalition_config = {
        str(key): tuple(str(party) for party in members)
        for key, members in coalitions.items()
    }
    points: list[dict[str, Any]] = []
    days_to_election = (election - origin).days
    for offset in range(1, days_to_election + 1):
        projection_date = origin + timedelta(days=offset)
        remaining = (election - projection_date).days
        kwargs: dict[str, Any] = {
            "as_of": origin.isoformat(),
            "election_date": election.isoformat(),
            "samples": samples,
            "seed": seed,
            "dynamics_horizon_days": remaining,
        }
        if data_dir is not None:
            kwargs["data_dir"] = Path(data_dir)
        result = runner(**kwargs)
        result_as_of = getattr(getattr(result, "summary", None), "as_of", None)
        if result_as_of is not None and str(result_as_of) != origin.isoformat():
            raise ValueError("projection runner changed the frozen opinion-state cutoff")
        points.append(
            {
                "date": projection_date.isoformat(),
                "remaining_horizon_days": remaining,
                "samples": int(len(result.vote_shares_matrix)),
                "groups": build_groups_from_matrices(
                    result.vote_shares_matrix,
                    result.seats_matrix,
                    coalitions=coalition_config,
                ),
            }
        )

    return {
        "projection_type": "conditional_forward_projection",
        "assumption": PROJECTION_ASSUMPTION,
        "origin_date": origin.isoformat(),
        "state_cutoff_date": origin.isoformat(),
        "election_date": election.isoformat(),
        "future_measurements_known": False,
        "state_condition": "underlying_opinion_unchanged_from_origin",
        "dynamics_horizon_rule": "election_date_minus_projection_date",
        "election_noise": "canonical_adopted_law",
        "mandate_allocation": "canonical_production_path",
        "tooltip_sv": projection_tooltip_sv(origin),
        "anchor": {
            "date": origin.isoformat(),
            "samples": int(anchor_point["samples"]),
            "provenance": "current_production",
            "groups": anchor_point["groups"],
        },
        "series": points,
        "rendering": {
            "x_axis_max": election.isoformat(),
            "future_region": {
                "start": origin.isoformat(),
                "end": election.isoformat(),
                "background": "light_neutral",
            },
            "latest_forecast_label": LATEST_FORECAST_LABEL_SV,
            "election_day_label": ELECTION_DAY_LABEL_SV,
            "legend_label": PROJECTION_LEGEND_SV,
            "median_line": "dashed_lighter",
            "interval_bands": ["p25_p75", "p05_p95"],
            "units": ["vote", "seats"],
            "poll_observations_in_future": False,
            "poll_of_polls_observations_in_future": False,
            "connect_from_history_anchor": True,
        },
    }


def update_history_with_production_result(
    existing_payload: Mapping[str, Any],
    production_result: Any,
    *,
    projection_samples: int = DEFAULT_PROJECTION_SAMPLES,
    projection_data_dir: Path | str | None = None,
    projection_runner: Callable[..., Any] | None = None,
    **history_kwargs: Any,
) -> dict[str, Any]:
    """Roll in the certified point and attach a separate conditional fan.

    The underlying history updater remains the authority for historical and
    current points.  This wrapper adds only ``future_projection`` and then
    recomputes the history artifact's deterministic self-hash.
    """

    history = _update_history_with_production_result(
        existing_payload,
        production_result,
        **history_kwargs,
    )
    current_points = [
        point
        for point in history["series"]
        if point.get("provenance") == "current_production"
    ]
    if len(current_points) != 1:
        raise ValueError("history must contain exactly one current_production anchor")
    current = current_points[0]
    manifest = getattr(production_result, "manifest", None)
    manifest_map = manifest if isinstance(manifest, Mapping) else {}
    seed = manifest_map.get("base_seed", DEFAULT_SIMULATION_SEED)
    if not isinstance(seed, int) or isinstance(seed, bool):
        seed = DEFAULT_SIMULATION_SEED

    history["future_projection"] = build_future_projection(
        origin_date=current["date"],
        election_date=history["election_date"],
        anchor_point=current,
        samples=projection_samples,
        seed=seed,
        data_dir=projection_data_dir,
        coalitions=history["coalitions"],
        simulation_runner=projection_runner,
    )
    history["deterministic_content_sha256"] = deterministic_history_sha256(history)
    validate_history_contract(history)
    return history


__all__ = [
    "DEFAULT_PROJECTION_SAMPLES",
    "ELECTION_DAY_LABEL_SV",
    "LATEST_FORECAST_LABEL_SV",
    "PROJECTION_ASSUMPTION",
    "PROJECTION_LEGEND_SV",
    "build_future_projection",
    "projection_tooltip_sv",
    "update_history_with_production_result",
]
