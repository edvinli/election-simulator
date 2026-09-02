"""Published contract for the coherent forward campaign-path projection.

The history artifact gains one additive top-level object,
``future_campaign_paths``.  It is the **primary** future view: simulated
opinion trajectories from the certified origin through election day, plus the
emphasized election-day forecast distribution.

Two quantities live in that object and are deliberately kept apart:

``bands`` / ``paths``
    the *underlying opinion* share ``theta[t + d]``, i.e. the same quantity the
    Poll of Polls series measures in the historical region.  No ElectionNoise,
    no geography, no seats.
``election_day``
    the official election-day forecast distribution, copied verbatim from the
    certified ``current_production`` history point, therefore including
    ElectionNoise, the geographic projection, exact integerization and the
    statutory mandate allocation.

The pre-existing ``future_projection`` object stays in the payload as a
clearly labelled *secondary* analytical view.  See
``docs/future_campaign_paths.md``.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from scripts.simulator.config import DEFAULT_SIMULATION_SEED

from .campaign_paths import (
    BAND_LEGEND_SV,
    CAMPAIGN_PATH_ASSUMPTION,
    CAMPAIGN_PATH_MODEL_ID,
    CAMPAIGN_PATH_TYPE,
    DEFAULT_REPRESENTATIVE_PATHS,
    ELECTION_DAY_LABEL_SV,
    FUTURE_REGION_LABEL_SV,
    ORIGIN_BOUNDARY_LABEL_SV,
    PATH_LEGEND_SV,
    campaign_paths_tooltip_sv,
    election_day_tooltip_sv,
    simulate_campaign_paths,
)
from .contract import DEFAULT_COALITIONS, QUANTILE_LEVELS
from .future_projection import _coerce_date, election_day_label_sv


PRIMARY_ROLE = "primary_future_view"
SECONDARY_ROLE = "secondary_analytical_view"
ENDPOINT_PARITY_GUARANTEE = "bitwise_identical_to_production_election_day_draws"
PATH_QUANTITY = "underlying_opinion_share"
PATH_SELECTION_RULE = "evenly_spaced_draw_indices"
CONTINUES_FROM = "poll_of_polls_opinion_series"

#: The secondary fan answers a strictly narrower question than the primary
#: view, and the published copy has to say so in the reader's language.
SECONDARY_DESCRIPTION_SV = (
    "Kvarvarande osäkerhet om det underliggande opinionsläget står stilla. "
    "Detta är en sekundär analysvy, inte huvudprognosen för framtiden."
)

_PATH_VALUE_DECIMALS = 4
_QUANTILE_KEYS: tuple[str, ...] = tuple(name for name, _ in QUANTILE_LEVELS)


def _vote_quantiles(draws: np.ndarray) -> dict[str, float]:
    values = np.asarray(draws, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("cannot summarize an empty coalition distribution")
    if not np.isfinite(values).all():
        raise ValueError("coalition distribution contains non-finite values")
    summary: dict[str, float] = {}
    for name, level in QUANTILE_LEVELS:
        summary[name] = round(float(np.quantile(values, level)), 6)
    return summary


def build_future_campaign_paths(
    *,
    origin_date: str | date,
    election_date: str | date,
    anchor_point: Mapping[str, Any],
    samples: int | None = None,
    seed: int = DEFAULT_SIMULATION_SEED,
    data_dir: Path | str | None = None,
    coalitions: Mapping[str, Sequence[str]] = DEFAULT_COALITIONS,
    representative_paths: int = DEFAULT_REPRESENTATIVE_PATHS,
    path_simulator: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Build the primary forward campaign-path object.

    ``anchor_point`` is the certified ``current_production`` history point.  Its
    joint coalition summaries are reused **verbatim** for election day, so the
    published election-day vote and seat probabilities are the production ones
    by construction rather than by recomputation.

    The simulator re-derives its own ``d = n`` endpoint through the canonical
    ``generate_national_vote_shares`` engine and fails closed unless the two
    agree bitwise; this function publishes the outcome of that gate and
    refuses a non-zero difference.
    """

    origin = _coerce_date(origin_date, name="origin_date")
    election = _coerce_date(election_date, name="election_date")
    if origin >= election:
        raise ValueError("origin_date must be strictly before election_date")
    if not isinstance(anchor_point, Mapping):
        raise ValueError("anchor_point must be the certified current production point")
    if str(anchor_point.get("date")) != origin.isoformat():
        raise ValueError("anchor_point date must equal origin_date")
    if anchor_point.get("provenance") != "current_production":
        raise ValueError("anchor_point must be the current_production point")
    anchor_groups = anchor_point.get("groups")
    if not isinstance(anchor_groups, Mapping):
        raise ValueError("anchor_point must contain joint coalition groups")
    anchor_samples = anchor_point.get("samples")
    if not isinstance(anchor_samples, int) or isinstance(anchor_samples, bool) or anchor_samples <= 0:
        raise ValueError("anchor_point samples must be a positive integer")

    resolved_samples = anchor_samples if samples is None else samples
    if not isinstance(resolved_samples, int) or isinstance(resolved_samples, bool) or resolved_samples <= 0:
        raise ValueError("samples must be a positive integer")

    coalition_config = {
        str(key): tuple(str(party) for party in members) for key, members in coalitions.items()
    }
    if list(coalition_config) != list(anchor_groups):
        raise ValueError("anchor coalition groups must cover the configured coalitions in order")

    simulator = path_simulator or simulate_campaign_paths
    kwargs: dict[str, Any] = {
        "as_of": origin.isoformat(),
        "election_date": election.isoformat(),
        "samples": resolved_samples,
        "seed": seed,
        "coalitions": coalition_config,
        "representative_paths": representative_paths,
    }
    if data_dir is not None:
        kwargs["data_dir"] = Path(data_dir)
    simulation = simulator(**kwargs)

    path_days = (election - origin).days
    if simulation.path_days != path_days:
        raise ValueError("campaign-path simulation returned a different path length")
    if str(simulation.origin_date) != origin.isoformat():
        raise ValueError("campaign-path simulation changed the frozen opinion-state cutoff")
    if len(simulation.day_dates) != path_days + 1:
        raise ValueError("campaign-path simulation must cover the origin and every campaign day")
    if list(simulation.coalition_draws) != list(coalition_config):
        raise ValueError("campaign-path simulation returned unexpected coalitions")

    # ---- endpoint parity gate ------------------------------------------
    construction = dict(simulation.diagnostics)
    parity_verified = bool(construction.get("endpoint_parity_verified", False))
    parity_difference = construction.get("endpoint_parity_max_abs_difference_pp")
    if parity_verified:
        if parity_difference is None or float(parity_difference) != 0.0:
            raise ValueError(
                "campaign-path election-day draws are not bitwise identical to the "
                f"canonical production draws (max |difference| = {parity_difference})"
            )
        parity_difference = 0.0
    else:
        parity_difference = None

    # ---- predictive bands ----------------------------------------------
    bands: list[dict[str, Any]] = []
    for day_offset, day_date in enumerate(simulation.day_dates):
        bands.append(
            {
                "date": day_date.isoformat(),
                "path_day": day_offset,
                "groups": {
                    key: {"vote": _vote_quantiles(simulation.coalition_draws[key][day_offset])}
                    for key in coalition_config
                },
            }
        )

    # ---- representative individual trajectories -------------------------
    sample_indices = tuple(int(index) for index in simulation.representative_indices)
    if not sample_indices:
        raise ValueError("campaign-path simulation returned no representative trajectories")
    if any(index < 0 or index >= resolved_samples for index in sample_indices):
        raise ValueError("representative trajectory indices fall outside the draw matrix")
    path_series = [
        {
            "sample_index": index,
            "values": {
                key: [
                    round(float(simulation.coalition_draws[key][day_offset, index]), _PATH_VALUE_DECIMALS)
                    for day_offset in range(path_days + 1)
                ]
                for key in coalition_config
            },
        }
        for index in sample_indices
    ]

    return {
        "projection_type": CAMPAIGN_PATH_TYPE,
        "model_id": CAMPAIGN_PATH_MODEL_ID,
        "assumption": CAMPAIGN_PATH_ASSUMPTION,
        "role": PRIMARY_ROLE,
        "origin_date": origin.isoformat(),
        "state_cutoff_date": origin.isoformat(),
        "election_date": election.isoformat(),
        "path_days": path_days,
        "samples": resolved_samples,
        "quantity": PATH_QUANTITY,
        "future_measurements_known": False,
        "path_construction": {
            "space": "clr",
            "categories": 9,
            "formula": "CLR(theta[t+d]) = CLR(theta_t) + sign * (CLR(PoP[s+d]) - CLR(PoP[s]))",
            "sign_policy": "single_sign_per_whole_trajectory",
            "transition_pool": "all_history_leakage_safe",
            "leakage_rule": "trajectory_end_le_origin",
            "eligible_trajectories": int(construction["eligible_trajectories"]),
            "earliest_trajectory_start": str(construction["earliest_trajectory_start"]),
            "latest_trajectory_end": str(construction["latest_trajectory_end"]),
            "endpoint_horizon_days": int(construction["endpoint_horizon_days"]),
            "time_warp": str(construction["time_warp"]),
            "synthesized_future_polls": False,
            "daily_independent_random_walk": False,
            "directional_momentum": False,
        },
        "endpoint_parity": {
            "guarantee": ENDPOINT_PARITY_GUARANTEE,
            "verified": parity_verified,
            "max_abs_vote_share_difference_pp": parity_difference,
            "election_day_summaries_source": "certified_current_production_point",
            "shared_seeds": {
                "opinion_state": int(construction["opinion_state_seed"]),
                "shared_dynamics": int(construction["dynamics_seed"]),
                "election_noise": int(construction["election_noise_seed"]),
            },
        },
        "bands": bands,
        "paths": {
            "count": len(path_series),
            "selection": PATH_SELECTION_RULE,
            "sample_indices": list(sample_indices),
            "series": path_series,
        },
        "election_day": {
            "date": election.isoformat(),
            "samples": anchor_samples,
            "label_sv": ELECTION_DAY_LABEL_SV,
            "includes_election_noise": True,
            "includes_geography_and_mandates": True,
            "provenance": "current_production",
            # Deep-copied so the published election-day distribution cannot be
            # mutated through the history series (or vice versa) while still
            # being value-identical to the certified production point.
            "groups": deepcopy(dict(anchor_groups)),
            "tooltip_sv": election_day_tooltip_sv(election),
        },
        "tooltip_sv": campaign_paths_tooltip_sv(origin, election),
        "rendering": {
            "x_axis_max": election.isoformat(),
            "future_region": {
                "start": origin.isoformat(),
                "end": election.isoformat(),
                "background": "light_distinct",
                "label": FUTURE_REGION_LABEL_SV,
            },
            "origin_boundary_label": ORIGIN_BOUNDARY_LABEL_SV,
            "election_day_label": election_day_label_sv(election),
            "election_day_distribution_label": ELECTION_DAY_LABEL_SV,
            "path_legend_label": PATH_LEGEND_SV,
            "band_legend_label": BAND_LEGEND_SV,
            "interval_bands": ["p25_p75", "p05_p95"],
            "path_units": ["vote"],
            "election_day_units": ["vote", "seats"],
            "median_may_be_flat": True,
            "intermediate_seat_trajectory": False,
            "poll_observations_in_future": False,
            "poll_of_polls_observations_in_future": False,
            "continues_from": CONTINUES_FROM,
        },
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_vote_quantiles(value: Any, *, name: str) -> None:
    if not isinstance(value, Mapping) or list(value) != list(_QUANTILE_KEYS):
        raise ValueError(f"{name} must contain p05, p25, p50, p75, p95 in order")
    numbers: list[float] = []
    for key in _QUANTILE_KEYS:
        current = value[key]
        if not isinstance(current, (int, float)) or isinstance(current, bool):
            raise ValueError(f"{name}.{key} must be numeric")
        number = float(current)
        if not math.isfinite(number) or number < 0.0 or number > 100.0:
            raise ValueError(f"{name}.{key} is outside 0–100")
        numbers.append(number)
    if numbers != sorted(numbers):
        raise ValueError(f"{name} quantiles must be monotone")


def validate_future_campaign_paths_contract(
    history: Mapping[str, Any],
    campaign_paths: Mapping[str, Any] | None = None,
) -> None:
    """Fail closed on any structurally or scientifically invalid path object."""

    if not isinstance(history, Mapping):
        raise ValueError("history must be an object")
    value = campaign_paths if campaign_paths is not None else history.get("future_campaign_paths")
    if not isinstance(value, Mapping):
        raise ValueError("future_campaign_paths must be an object")

    election = _coerce_date(history.get("election_date"), name="history.election_date")
    origin = _coerce_date(value.get("origin_date"), name="future_campaign_paths.origin_date")
    path_election = _coerce_date(
        value.get("election_date"), name="future_campaign_paths.election_date"
    )
    if path_election != election:
        raise ValueError("future_campaign_paths.election_date must match history.election_date")
    if origin >= election:
        raise ValueError("future_campaign_paths origin must precede election day")
    if value.get("projection_type") != CAMPAIGN_PATH_TYPE:
        raise ValueError("future_campaign_paths has an unknown projection_type")
    if value.get("model_id") != CAMPAIGN_PATH_MODEL_ID:
        raise ValueError("future_campaign_paths has an unknown model_id")
    if value.get("assumption") != CAMPAIGN_PATH_ASSUMPTION:
        raise ValueError("future_campaign_paths has an unknown assumption")
    if value.get("role") != PRIMARY_ROLE:
        raise ValueError("future_campaign_paths must declare itself the primary future view")
    if value.get("state_cutoff_date") != origin.isoformat():
        raise ValueError("future_campaign_paths state cutoff must equal its origin")
    if value.get("future_measurements_known") is not False:
        raise ValueError("future_campaign_paths must state that future measurements are unknown")
    if value.get("quantity") != PATH_QUANTITY:
        raise ValueError("future_campaign_paths must publish the underlying opinion share")
    path_days = (election - origin).days
    if value.get("path_days") != path_days:
        raise ValueError("future_campaign_paths path_days disagrees with the calendar")
    samples = value.get("samples")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError("future_campaign_paths samples must be a positive integer")
    if value.get("tooltip_sv") != campaign_paths_tooltip_sv(origin, election):
        raise ValueError("future_campaign_paths disclosure is missing or changed")

    construction = value.get("path_construction")
    if not isinstance(construction, Mapping):
        raise ValueError("future_campaign_paths.path_construction must be an object")
    if construction.get("space") != "clr" or construction.get("categories") != 9:
        raise ValueError("campaign paths must be constructed jointly in nine-category CLR space")
    if construction.get("sign_policy") != "single_sign_per_whole_trajectory":
        raise ValueError("campaign paths must apply one sign to the whole trajectory")
    if construction.get("transition_pool") != "all_history_leakage_safe":
        raise ValueError("campaign paths must resample all-history leakage-safe trajectories")
    if construction.get("leakage_rule") != "trajectory_end_le_origin":
        raise ValueError("campaign paths must enforce the structural leakage boundary")
    for forbidden in ("synthesized_future_polls", "daily_independent_random_walk", "directional_momentum"):
        if construction.get(forbidden) is not False:
            raise ValueError(f"campaign paths must explicitly disclaim {forbidden}")
    eligible = construction.get("eligible_trajectories")
    if not isinstance(eligible, int) or isinstance(eligible, bool) or eligible < 30:
        raise ValueError("campaign paths need at least the 30-transition minimum pool")
    endpoint_horizon = construction.get("endpoint_horizon_days")
    if not isinstance(endpoint_horizon, int) or isinstance(endpoint_horizon, bool) or endpoint_horizon < 1:
        raise ValueError("campaign paths must publish a positive endpoint horizon")
    if construction.get("time_warp") not in {"identity", "monotone_stretch"}:
        raise ValueError("campaign paths declare an unknown time warp")
    if construction.get("time_warp") == "identity" and endpoint_horizon != path_days:
        raise ValueError("an identity time warp requires the endpoint horizon to equal path_days")
    latest_end = _coerce_date(
        construction.get("latest_trajectory_end"),
        name="future_campaign_paths.path_construction.latest_trajectory_end",
    )
    if latest_end > origin:
        raise ValueError("campaign paths used a trajectory ending after the origin")

    parity = value.get("endpoint_parity")
    if not isinstance(parity, Mapping):
        raise ValueError("future_campaign_paths.endpoint_parity must be an object")
    if parity.get("guarantee") != ENDPOINT_PARITY_GUARANTEE:
        raise ValueError("future_campaign_paths must guarantee bitwise endpoint parity")
    if parity.get("election_day_summaries_source") != "certified_current_production_point":
        raise ValueError("election-day summaries must come from the certified production point")
    difference = parity.get("max_abs_vote_share_difference_pp")
    if parity.get("verified") is True:
        if not isinstance(difference, (int, float)) or isinstance(difference, bool) or float(difference) != 0.0:
            raise ValueError("a verified endpoint parity check must report exactly zero difference")
    elif difference is not None:
        raise ValueError("an unverified endpoint parity check must not report a difference")
    shared = parity.get("shared_seeds")
    if not isinstance(shared, Mapping) or list(shared) != [
        "opinion_state",
        "shared_dynamics",
        "election_noise",
    ]:
        raise ValueError("endpoint parity must publish the three shared production sub-seeds")
    for key, seed_value in shared.items():
        if not isinstance(seed_value, int) or isinstance(seed_value, bool) or seed_value < 0:
            raise ValueError(f"endpoint parity seed {key} must be a non-negative integer")

    coalitions = history.get("coalitions")
    if not isinstance(coalitions, Mapping) or not coalitions:
        raise ValueError("history coalitions are required for campaign-path validation")

    current = [
        point
        for point in history.get("series", [])
        if isinstance(point, Mapping) and point.get("provenance") == "current_production"
    ]
    if len(current) != 1:
        raise ValueError("history must contain exactly one current_production point")
    current_point = current[0]
    if current_point.get("date") != origin.isoformat():
        raise ValueError("future_campaign_paths origin must equal the current production date")

    for index, poll in enumerate(history.get("polls", [])):
        if not isinstance(poll, Mapping):
            continue
        published = _coerce_date(
            poll.get("publication_date"), name=f"history.polls[{index}].publication_date"
        )
        if published > origin:
            raise ValueError("history.polls contains an observation after the campaign-path origin")
    for index, observation in enumerate(history.get("poll_of_polls", [])):
        if not isinstance(observation, Mapping):
            continue
        observed = _coerce_date(
            observation.get("date"), name=f"history.poll_of_polls[{index}].date"
        )
        if observed > origin:
            raise ValueError(
                "history.poll_of_polls contains an observation after the campaign-path origin"
            )

    bands = value.get("bands")
    if not isinstance(bands, list) or len(bands) != path_days + 1:
        raise ValueError("future_campaign_paths.bands must cover the origin and every campaign day")
    for index, band in enumerate(bands):
        expected_date = origin + timedelta(days=index)
        if not isinstance(band, Mapping):
            raise ValueError(f"future_campaign_paths.bands[{index}] must be an object")
        if band.get("date") != expected_date.isoformat() or band.get("path_day") != index:
            raise ValueError("campaign-path bands must be strictly daily from the origin")
        groups = band.get("groups")
        if not isinstance(groups, Mapping) or list(groups) != list(coalitions):
            raise ValueError(
                f"future_campaign_paths.bands[{index}].groups must cover the coalitions in order"
            )
        for coalition in coalitions:
            group = groups[coalition]
            if not isinstance(group, Mapping) or list(group) != ["vote"]:
                raise ValueError(
                    "campaign-path bands publish opinion vote shares only; seats belong to "
                    "the historical series and the election-day distribution"
                )
            _validate_vote_quantiles(
                group["vote"], name=f"future_campaign_paths.bands[{index}].groups.{coalition}.vote"
            )

    paths = value.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("future_campaign_paths.paths must be an object")
    if paths.get("selection") != PATH_SELECTION_RULE:
        raise ValueError("campaign-path trajectory selection rule changed")
    series = paths.get("series")
    indices = paths.get("sample_indices")
    if not isinstance(series, list) or not series:
        raise ValueError("future_campaign_paths.paths.series must be a non-empty list")
    if paths.get("count") != len(series):
        raise ValueError("future_campaign_paths.paths.count disagrees with the published series")
    if not isinstance(indices, list) or len(indices) != len(series):
        raise ValueError("future_campaign_paths.paths.sample_indices must match the series")
    if sorted(indices) != list(indices) or len(set(indices)) != len(indices):
        raise ValueError("campaign-path sample indices must be sorted and unique")
    for index, item in enumerate(series):
        if not isinstance(item, Mapping):
            raise ValueError(f"future_campaign_paths.paths.series[{index}] must be an object")
        sample_index = item.get("sample_index")
        if sample_index != indices[index]:
            raise ValueError("campaign-path trajectories must follow the published index order")
        if not isinstance(sample_index, int) or isinstance(sample_index, bool) or not (
            0 <= sample_index < samples
        ):
            raise ValueError("campaign-path sample_index falls outside the draw matrix")
        values = item.get("values")
        if not isinstance(values, Mapping) or list(values) != list(coalitions):
            raise ValueError("campaign-path trajectories must cover the coalitions in order")
        for coalition in coalitions:
            track = values[coalition]
            if not isinstance(track, list) or len(track) != path_days + 1:
                raise ValueError(
                    "each campaign-path trajectory needs one value for the origin and every day"
                )
            for point in track:
                if not isinstance(point, (int, float)) or isinstance(point, bool):
                    raise ValueError("campaign-path trajectory values must be numeric")
                number = float(point)
                if not math.isfinite(number) or number < 0.0 or number > 100.0:
                    raise ValueError("campaign-path trajectory values must be shares in 0–100")

    election_day = value.get("election_day")
    if not isinstance(election_day, Mapping):
        raise ValueError("future_campaign_paths.election_day must be an object")
    if election_day.get("date") != election.isoformat():
        raise ValueError("the emphasized distribution must sit exactly on election day")
    if election_day.get("includes_election_noise") is not True:
        raise ValueError("the election-day distribution must include ElectionNoise")
    if election_day.get("includes_geography_and_mandates") is not True:
        raise ValueError("the election-day distribution must include geography and mandates")
    if election_day.get("provenance") != "current_production":
        raise ValueError("the election-day distribution must be the certified production one")
    if election_day.get("label_sv") != ELECTION_DAY_LABEL_SV:
        raise ValueError("the election-day distribution label changed")
    if election_day.get("tooltip_sv") != election_day_tooltip_sv(election):
        raise ValueError("the election-day distribution disclosure changed")
    if election_day.get("samples") != current_point.get("samples"):
        raise ValueError("the election-day distribution must keep the certified draw count")
    if election_day.get("groups") != current_point.get("groups"):
        raise ValueError(
            "the election-day distribution must reproduce the certified production vote and "
            "seat summaries exactly"
        )

    rendering = value.get("rendering")
    if not isinstance(rendering, Mapping):
        raise ValueError("future_campaign_paths.rendering must be an object")
    region = rendering.get("future_region")
    if (
        rendering.get("x_axis_max") != election.isoformat()
        or not isinstance(region, Mapping)
        or region.get("start") != origin.isoformat()
        or region.get("end") != election.isoformat()
        or region.get("background") != "light_distinct"
        or region.get("label") != FUTURE_REGION_LABEL_SV
    ):
        raise ValueError("campaign-path rendering boundaries must span origin to election day")
    if rendering.get("origin_boundary_label") != ORIGIN_BOUNDARY_LABEL_SV:
        raise ValueError("campaign-path origin boundary label changed")
    if rendering.get("election_day_label") != election_day_label_sv(election):
        raise ValueError("campaign-path election-day label changed")
    if rendering.get("election_day_distribution_label") != ELECTION_DAY_LABEL_SV:
        raise ValueError("campaign-path election-day distribution label changed")
    if rendering.get("path_legend_label") != PATH_LEGEND_SV:
        raise ValueError("campaign-path legend label changed")
    if rendering.get("band_legend_label") != BAND_LEGEND_SV:
        raise ValueError("campaign-path band legend label changed")
    if rendering.get("interval_bands") != ["p25_p75", "p05_p95"]:
        raise ValueError("campaign paths must publish the 50% and 90% predictive bands")
    if rendering.get("path_units") != ["vote"]:
        raise ValueError("campaign-path trajectories are vote-share only")
    if rendering.get("election_day_units") != ["vote", "seats"]:
        raise ValueError("the election-day distribution must support vote and seats")
    if rendering.get("intermediate_seat_trajectory") is not False:
        raise ValueError("campaign paths must not imply a smooth future seat trajectory")
    if rendering.get("median_may_be_flat") is not True:
        raise ValueError("campaign paths must disclose that the median may remain flat")
    if rendering.get("poll_observations_in_future") is not False:
        raise ValueError("campaign paths must prohibit future poll observations")
    if rendering.get("poll_of_polls_observations_in_future") is not False:
        raise ValueError("campaign paths must prohibit future Poll of Polls observations")
    if rendering.get("continues_from") != CONTINUES_FROM:
        raise ValueError("campaign paths must declare that they continue the opinion series")


def mark_secondary_projection(projection: Mapping[str, Any]) -> dict[str, Any]:
    """Relabel the shrinking-horizon fan as the secondary analytical view."""

    if not isinstance(projection, Mapping):
        raise ValueError("projection must be an object")
    marked = dict(projection)
    marked["role"] = SECONDARY_ROLE
    marked["primary"] = False
    marked["description_sv"] = SECONDARY_DESCRIPTION_SV
    return marked


def validate_secondary_projection_role(projection: Mapping[str, Any]) -> None:
    """Fail closed unless the old fan is demoted and described as secondary."""

    if not isinstance(projection, Mapping):
        raise ValueError("future_projection must be an object")
    if projection.get("role") != SECONDARY_ROLE:
        raise ValueError("future_projection must declare the secondary analytical role")
    if projection.get("primary") is not False:
        raise ValueError("future_projection must not claim to be the primary future view")
    if projection.get("description_sv") != SECONDARY_DESCRIPTION_SV:
        raise ValueError("future_projection secondary description is missing or changed")


__all__ = [
    "CONTINUES_FROM",
    "ENDPOINT_PARITY_GUARANTEE",
    "PATH_QUANTITY",
    "PATH_SELECTION_RULE",
    "PRIMARY_ROLE",
    "SECONDARY_DESCRIPTION_SV",
    "SECONDARY_ROLE",
    "build_future_campaign_paths",
    "mark_secondary_projection",
    "validate_future_campaign_paths_contract",
    "validate_secondary_projection_role",
]
