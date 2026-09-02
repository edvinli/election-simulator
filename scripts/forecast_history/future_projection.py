"""Conditional shrinking-horizon fan — the *secondary* analytical future view.

This module deliberately keeps hypothetical future chart points separate from
``history["series"]``. Every projection point freezes the polling cutoff at the
latest certified production date and changes only the remaining Dynamics v2
horizon. Future polls are therefore never synthesized or admitted.

The projection reuses the frozen production scientific components, including
the adopted ElectionNoise law, geographic projection, controlled rounding, and
statutory mandate allocator. It is a conditional visualization, not a forecast
of future polling observations.

Since the coherent campaign-path model was introduced this fan is **no longer
the primary future prognosis**. It answers the narrower question "how much
uncertainty remains if the underlying opinion stays unchanged", and the
published object is explicitly demoted to
``role = "secondary_analytical_view"``. The headline future view lives in
``scripts.forecast_history.campaign_paths`` and its published contract in
``scripts.forecast_history.campaign_paths_contract``.
"""

from __future__ import annotations

from datetime import date, timedelta
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from scripts.simulator.config import DEFAULT_SIMULATION_SEED

from .contract import (
    DEFAULT_COALITIONS,
    QUANTILE_LEVELS,
    build_groups_from_matrices,
    deterministic_history_sha256,
    validate_history_contract,
)
from .generate import update_history_with_production_result as _update_history_with_production_result
from .projection_simulator import ELECTION_NOISE_RNG_POLICY, simulate_conditional_projection


DEFAULT_PROJECTION_SAMPLES = 10_000
PROJECTION_ASSUMPTION = "frozen_opinion_state_shrinking_dynamics_horizon"
PROJECTION_LEGEND_SV = "Framåtblickande projektion"
LATEST_FORECAST_LABEL_SV = "Senaste prognos"

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


def _validate_quantiles(value: Any, *, name: str, integer: bool, upper: float) -> None:
    expected = [key for key, _ in QUANTILE_LEVELS]
    if not isinstance(value, Mapping) or list(value) != expected:
        raise ValueError(f"{name} must contain p05, p25, p50, p75, p95 in order")
    numbers: list[float] = []
    for key in expected:
        current = value[key]
        if not isinstance(current, (int, float)) or isinstance(current, bool):
            raise ValueError(f"{name}.{key} must be numeric")
        if integer and not isinstance(current, int):
            raise ValueError(f"{name}.{key} must be an integer")
        number = float(current)
        if not math.isfinite(number) or number < 0 or number > upper:
            raise ValueError(f"{name}.{key} is outside the allowed range")
        numbers.append(number)
    if numbers != sorted(numbers):
        raise ValueError(f"{name} quantiles must be monotone")


def _validate_groups(
    groups: Any,
    *,
    coalitions: Mapping[str, Sequence[str]],
    name: str,
) -> None:
    if not isinstance(groups, Mapping) or list(groups) != list(coalitions):
        raise ValueError(f"{name} must cover the configured coalitions in order")
    for coalition in coalitions:
        group = groups[coalition]
        if not isinstance(group, Mapping) or list(group) != ["vote", "seats"]:
            raise ValueError(f"{name}.{coalition} must contain vote and seats")
        _validate_quantiles(
            group["vote"],
            name=f"{name}.{coalition}.vote",
            integer=False,
            upper=100.0,
        )
        _validate_quantiles(
            group["seats"],
            name=f"{name}.{coalition}.seats",
            integer=True,
            upper=349.0,
        )


def projection_tooltip_sv(origin_date: str | date) -> str:
    """Return the explicit conditional-projection disclosure shown by consumers."""

    origin = _coerce_date(origin_date, name="origin_date")
    return (
        f"Framåtblickande projektion från opinionsläget {_short_date_sv(origin)}. "
        "Antar oförändrat underliggande opinionsläge; framtida mätningar är okända."
    )


def election_day_label_sv(election_date: str | date) -> str:
    """Return the rendering label derived from the artifact's election date."""

    election = _coerce_date(election_date, name="election_date")
    return f"Valdag {_short_date_sv(election)}"


def validate_future_projection_contract(
    history: Mapping[str, Any],
    projection: Mapping[str, Any] | None = None,
) -> None:
    """Validate the additive conditional projection against its history anchor."""

    if not isinstance(history, Mapping):
        raise ValueError("history must be an object")
    value = projection if projection is not None else history.get("future_projection")
    if not isinstance(value, Mapping):
        raise ValueError("future_projection must be an object")
    election = _coerce_date(history.get("election_date"), name="history.election_date")
    origin = _coerce_date(value.get("origin_date"), name="future_projection.origin_date")
    projection_election = _coerce_date(
        value.get("election_date"), name="future_projection.election_date"
    )
    if projection_election != election:
        raise ValueError("future_projection.election_date must match history.election_date")
    if origin > election:
        raise ValueError("future_projection origin cannot occur after election day")
    if value.get("projection_type") != "conditional_forward_projection":
        raise ValueError("future_projection has an unknown projection_type")
    if value.get("assumption") != PROJECTION_ASSUMPTION:
        raise ValueError("future_projection has an unknown assumption")
    if value.get("state_cutoff_date") != origin.isoformat():
        raise ValueError("future_projection state cutoff must equal its origin")
    if value.get("future_measurements_known") is not False:
        raise ValueError("future_projection must explicitly state that future measurements are unknown")
    if value.get("state_condition") != "underlying_opinion_unchanged_from_origin":
        raise ValueError("future_projection must freeze the underlying opinion state")
    if value.get("dynamics_horizon_rule") != "election_date_minus_projection_date":
        raise ValueError("future_projection has an unknown dynamics horizon rule")
    if value.get("election_noise_rng_policy") != ELECTION_NOISE_RNG_POLICY:
        raise ValueError("future_projection has an unknown ElectionNoise RNG policy")
    if value.get("tooltip_sv") != projection_tooltip_sv(origin):
        raise ValueError("future_projection tooltip disclosure is missing or changed")

    current = [
        point
        for point in history.get("series", [])
        if isinstance(point, Mapping) and point.get("provenance") == "current_production"
    ]
    if len(current) != 1:
        raise ValueError("history must contain exactly one current_production point")
    current_point = current[0]
    if current_point.get("date") != origin.isoformat():
        raise ValueError("future_projection origin must equal the current production date")

    for index, poll in enumerate(history.get("polls", [])):
        if not isinstance(poll, Mapping):
            continue
        publication = _coerce_date(
            poll.get("publication_date"),
            name=f"history.polls[{index}].publication_date",
        )
        if publication > origin:
            raise ValueError(
                "history.polls contains an observation after future_projection.origin_date"
            )
    for index, observation in enumerate(history.get("poll_of_polls", [])):
        if not isinstance(observation, Mapping):
            continue
        observation_date = _coerce_date(
            observation.get("date"),
            name=f"history.poll_of_polls[{index}].date",
        )
        if observation_date > origin:
            raise ValueError(
                "history.poll_of_polls contains an observation after future_projection.origin_date"
            )
    anchor = value.get("anchor")
    if not isinstance(anchor, Mapping):
        raise ValueError("future_projection.anchor must be an object")
    if anchor.get("date") != origin.isoformat() or anchor.get("provenance") != "current_production":
        raise ValueError("future_projection anchor identity is invalid")
    if anchor.get("samples") != current_point.get("samples"):
        raise ValueError("future_projection anchor sample count must equal the official current point")
    if anchor.get("groups") != current_point.get("groups"):
        raise ValueError("future_projection anchor must exactly reproduce current joint coalition summaries")

    coalitions = history.get("coalitions")
    if not isinstance(coalitions, Mapping) or not coalitions:
        raise ValueError("history coalitions are required for future projection validation")
    _validate_groups(anchor.get("groups"), coalitions=coalitions, name="future_projection.anchor.groups")

    future_series = value.get("series")
    if not isinstance(future_series, list):
        raise ValueError("future_projection.series must be a list")
    expected_dates = [
        origin + timedelta(days=offset)
        for offset in range(1, (election - origin).days + 1)
    ]
    if len(future_series) != len(expected_dates):
        raise ValueError("future_projection must contain exactly one point per future calendar day")
    historical_dates = {
        str(point.get("date"))
        for point in history.get("series", [])
        if isinstance(point, Mapping)
    }
    for index, (point, expected_date) in enumerate(zip(future_series, expected_dates)):
        if not isinstance(point, Mapping):
            raise ValueError(f"future_projection.series[{index}] must be an object")
        if point.get("date") != expected_date.isoformat():
            raise ValueError("future_projection dates must be strictly daily and end on election day")
        if point.get("date") in historical_dates:
            raise ValueError("future_projection points must never be mixed into historical series")
        remaining = (election - expected_date).days
        if point.get("remaining_horizon_days") != remaining:
            raise ValueError("future_projection remaining horizon disagrees with projection date")
        samples = point.get("samples")
        if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
            raise ValueError("future_projection samples must be positive integers")
        _validate_groups(
            point.get("groups"),
            coalitions=coalitions,
            name=f"future_projection.series[{index}].groups",
        )
    if future_series and future_series[-1].get("remaining_horizon_days") != 0:
        raise ValueError("future_projection election-day point must have zero dynamics horizon")

    rendering = value.get("rendering")
    if not isinstance(rendering, Mapping):
        raise ValueError("future_projection.rendering must be an object")
    region = rendering.get("future_region")
    if (
        rendering.get("x_axis_max") != election.isoformat()
        or not isinstance(region, Mapping)
        or region.get("start") != origin.isoformat()
        or region.get("end") != election.isoformat()
    ):
        raise ValueError("future_projection rendering boundaries must span origin to election day")
    if rendering.get("latest_forecast_label") != LATEST_FORECAST_LABEL_SV:
        raise ValueError("future_projection latest forecast label changed")
    if rendering.get("election_day_label") != election_day_label_sv(election):
        raise ValueError("future_projection election-day label changed")
    if rendering.get("legend_label") != PROJECTION_LEGEND_SV:
        raise ValueError("future_projection legend label changed")
    if rendering.get("units") != ["vote", "seats"]:
        raise ValueError("future_projection must support both vote and seat views")
    if rendering.get("poll_observations_in_future") is not False:
        raise ValueError("future_projection must prohibit future poll observations")
    if rendering.get("poll_of_polls_observations_in_future") is not False:
        raise ValueError("future_projection must prohibit future Poll of Polls observations")


def build_future_projection(
    *,
    origin_date: str | date,
    election_date: str | date,
    anchor_point: Mapping[str, Any],
    samples: int = DEFAULT_PROJECTION_SAMPLES,
    seed: int = DEFAULT_SIMULATION_SEED,
    data_dir: Path | str | None = None,
    coalitions: Mapping[str, Sequence[str]] = DEFAULT_COALITIONS,
    projection_runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Build daily conditional points strictly after ``origin_date``.

    ``as_of`` is held fixed at ``origin_date`` for every call. The explicit
    ``dynamics_horizon_days`` argument alone decreases to zero on election day.
    Production callers use the projection-only simulator that composes the same
    scientific and mandate components without modifying the frozen production
    entrypoint.
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

    runner = projection_runner or simulate_conditional_projection
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
        if len(result.vote_shares_matrix) != samples or len(result.seats_matrix) != samples:
            raise ValueError("projection runner returned a different draw count than requested")
        points.append(
            {
                "date": projection_date.isoformat(),
                "remaining_horizon_days": remaining,
                "samples": samples,
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
        "election_noise_rng_policy": ELECTION_NOISE_RNG_POLICY,
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
            "election_day_label": election_day_label_sv(election),
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
    campaign_path_samples: int | None = None,
    campaign_path_simulator: Callable[..., Any] | None = None,
    **history_kwargs: Any,
) -> dict[str, Any]:
    """Roll in the certified point, the primary paths and the secondary fan.

    ``future_campaign_paths`` is the headline future view: coherent simulated
    opinion trajectories from the certified origin through election day, whose
    election-day endpoint is bitwise identical to the certified production
    draws.  ``future_projection`` is retained and explicitly demoted to the
    secondary "remaining uncertainty if opinion stays unchanged" view.
    """

    # Imported lazily: the campaign-path contract reuses this module's date and
    # label helpers, so a module-level import would be circular.
    from .campaign_paths_contract import (
        build_future_campaign_paths,
        mark_secondary_projection,
        validate_future_campaign_paths_contract,
        validate_secondary_projection_role,
    )

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

    history["future_projection"] = mark_secondary_projection(
        build_future_projection(
            origin_date=current["date"],
            election_date=history["election_date"],
            anchor_point=current,
            samples=projection_samples,
            seed=seed,
            data_dir=projection_data_dir,
            coalitions=history["coalitions"],
            projection_runner=projection_runner,
        )
    )
    validate_future_projection_contract(history)
    validate_secondary_projection_role(history["future_projection"])

    origin = _coerce_date(current["date"], name="current production date")
    election = _coerce_date(history["election_date"], name="history.election_date")
    if origin < election:
        history["future_campaign_paths"] = build_future_campaign_paths(
            origin_date=current["date"],
            election_date=history["election_date"],
            anchor_point=current,
            samples=campaign_path_samples,
            seed=seed,
            data_dir=projection_data_dir,
            coalitions=history["coalitions"],
            path_simulator=campaign_path_simulator,
        )
        validate_future_campaign_paths_contract(history)
    else:
        # On election day there is no remaining campaign to simulate.  Drop the
        # key rather than publishing an empty primary view.
        history.pop("future_campaign_paths", None)

    history["deterministic_content_sha256"] = deterministic_history_sha256(history)
    validate_history_contract(history)
    validate_future_projection_contract(history)
    validate_secondary_projection_role(history["future_projection"])
    if "future_campaign_paths" in history:
        validate_future_campaign_paths_contract(history)
    return history


__all__ = [
    "DEFAULT_PROJECTION_SAMPLES",
    "ELECTION_NOISE_RNG_POLICY",
    "LATEST_FORECAST_LABEL_SV",
    "PROJECTION_ASSUMPTION",
    "PROJECTION_LEGEND_SV",
    "build_future_projection",
    "election_day_label_sv",
    "projection_tooltip_sv",
    "update_history_with_production_result",
    "validate_future_projection_contract",
]
