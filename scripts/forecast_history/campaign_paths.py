"""Coherent forward campaign-path projection for the forecast history chart.

The headline future visualization answers a *forward opinion* question:

    Given the certified ``OpinionState`` at origin ``t``, what complete
    opinion trajectories from ``t`` to election day are consistent with how
    Swedish opinion has actually moved over comparable windows in the past?

Construction (all in centred-log-ratio space, ``D = 9`` categories):

1. Freeze the certified ``OpinionState`` at origin ``t``; draw ``CLR(theta_t)``
   exactly as production does (identical sub-seed, identical draws).
2. Let ``n = election_date - t``.
3. For every Monte Carlo draw sample **one complete historical Poll-of-Polls
   trajectory** starting at a historical date ``s``.  Only trajectories whose
   final observation satisfies ``s + endpoint_horizon <= t`` are eligible, so
   nothing observed after the origin can enter the pool.
4. Compute the whole-path displacement
   ``Delta_d = CLR(PoP[s + d]) - CLR(PoP[s])`` for ``d = 1 .. n``.
5. Apply **one** random sign ``S in {-1, +1}`` to the *entire* trajectory.
6. Construct ``CLR(theta[t + d]) = CLR(theta_t) + S * Delta_d``.
7. Because every step is a joint nine-category CLR vector, the multi-party
   composition and its empirical cross-party correlation structure survive
   intact and each day inverts back onto the simplex.
8. On election day the adopted ``ElectionNoise`` law, the geographic
   projection, exact integerization and the statutory mandate allocation are
   applied exactly as production applies them.

The model synthesizes **no** future polls, introduces **no** directional
momentum and **no** daily independent random walk: all randomness is the
transition index and the single whole-path sign, drawn from the frozen
production Dynamics v2 sub-seed.

Endpoint parity
---------------
The ``d = n`` endpoint is *bitwise* identical to the production Dynamics v2
draw, not merely distributionally similar.  See
``docs/future_campaign_paths.md`` for the full argument; the mechanism is:

* the trajectory pool **is** production's own eligible transition pool at the
  same endpoint horizon, in the same order, so index ``j`` denotes the same
  historical transition in both models;
* the transition indices and signs are drawn from the same generator, seeded
  with the same production sub-seed, consuming randomness in the same order;
* at ``d = n`` the path displacement reduces to the same floating-point
  subtraction ``CLR(PoP[s + n]) - CLR(PoP[s])`` that production stores in its
  transition pool, and ``_assert_endpoint_pool_parity`` checks that equality
  element by element at build time;
* the ``OpinionState`` and ``ElectionNoise`` sub-seeds are production's.

Every one of those links is asserted, not assumed, and each one fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals
from scripts.election_layer_v2.transfer import apply_batch_simplex_transfer
from scripts.hindcasts.models import derive_opinion_state_seed, derive_shared_dynamics_seed
from scripts.pollofpolls.clr import clr_to_composition_matrix, composition_to_clr
from scripts.pollofpolls.state import OpinionState, estimate_opinion, load_timeseries_dataset
from scripts.pollofpolls.state_config import ALL_CATEGORIES
from scripts.pollofpolls.transitions import (
    HistoricalTransition,
    MIN_TRANSITIONS,
    build_all_historical_transitions,
    filter_transitions_as_of,
)
from scripts.simulator.config import DEFAULT_SIMULATION_SEED
from scripts.vote_share_calibration.config import MIN_SHARE_PCT
from scripts.vote_share_calibration.election_noise_b import (
    MODEL_ID as ADOPTED_NOISE_MODEL,
    derive_election_noise_b_seed,
    draw_election_noise_b,
    fit_election_noise_b,
)

from .contract import (
    DEFAULT_COALITIONS,
    HISTORY_DYNAMICS_CAP_DAYS,
    QUANTILE_LEVELS,
    coalition_vote_draws,
)


CAMPAIGN_PATH_MODEL_ID = "coherent_campaign_paths_v1"
CAMPAIGN_PATH_TYPE = "coherent_campaign_paths"
CAMPAIGN_PATH_ASSUMPTION = "frozen_opinion_state_whole_path_sign_symmetric_historical_trajectory"

#: Production mirrors this fallback ladder when the exact-horizon pool is thin.
FALLBACK_HORIZONS: tuple[int, ...] = (28, 14, 7)

#: How many faint individual trajectories the chart draws.
DEFAULT_REPRESENTATIVE_PATHS = 24

#: Swedish presentation strings.  They are part of the published contract so a
#: consumer never invents its own description of the model.
FUTURE_REGION_LABEL_SV = "Möjliga opinionsbanor"
ELECTION_DAY_LABEL_SV = "Valdagsprognos"
ORIGIN_BOUNDARY_LABEL_SV = "I dag"
#: Path day 0 is the latent ``OpinionState`` at the origin. It is a *different
#: quantity* from the certified forecast point on the same calendar date, which
#: also carries campaign dynamics and ElectionNoise, so it gets its own label
#: and a consumer must never reuse the forecast marker as the fan's origin.
ORIGIN_STATE_LABEL_SV = "Opinionsläge i dag"
PATH_LEGEND_SV = "Simulerade opinionsbanor"
BAND_LEGEND_SV = "50 % och 90 % av opinionsbanorna"

_MONTHS_SV = (
    "jan", "feb", "mar", "apr", "maj", "jun",
    "jul", "aug", "sep", "okt", "nov", "dec",
)


def _coerce_date(value: str | date, *, name: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an ISO date") from exc


def _short_date_sv(value: date) -> str:
    return f"{value.day} {_MONTHS_SV[value.month - 1]}"


def campaign_paths_tooltip_sv(
    origin_date: str | date,
    election_date: str | date,
    time_warp: str = "identity",
    endpoint_horizon_days: int | None = None,
) -> str:
    """Return the published disclosure for the forward opinion-path region.

    The "same length" claim is only literally true under the identity day map.
    Above the 112-day Dynamics v2 cap (or on the fallback ladder) the sampled
    trajectory is shorter than the displayed period and is stretched over it,
    so the disclosure says that instead of overstating the construction.
    """

    origin = _coerce_date(origin_date, name="origin_date")
    election = _coerce_date(election_date, name="election_date")
    if time_warp == "identity":
        construction = "Varje bana är en hel historisk opinionsrörelse av samma längd"
    else:
        days = "" if endpoint_horizon_days is None else f" på {endpoint_horizon_days} dagar"
        construction = (
            f"Varje bana är en hel historisk opinionsrörelse{days}, "
            "tidsutsträckt över perioden"
        )
    return (
        f"Simulerade opinionsbanor från {_short_date_sv(origin)} till "
        f"{_short_date_sv(election)}. {construction}, med slumpmässigt tecken. "
        "Framtida mätningar är okända och simuleras inte."
    )


def origin_state_tooltip_sv(origin_date: str | date) -> str:
    """Return the published disclosure for the path origin at day 0."""

    origin = _coerce_date(origin_date, name="origin_date")
    return (
        f"Opinionsläget {_short_date_sv(origin)} enligt modellens skattning av "
        "dagens underliggande opinion. Detta är inte valdagsprognosen: den "
        "innehåller dessutom kampanjrörelse och valdagsavvikelse."
    )


def election_day_tooltip_sv(election_date: str | date) -> str:
    """Return the published disclosure for the emphasized election-day point."""

    election = _coerce_date(election_date, name="election_date")
    return (
        f"Valdagsprognos {_short_date_sv(election)}: samma officiella "
        "sannolikhetsfördelning som den publicerade prognosen, inklusive "
        "valdagsavvikelse, geografi och mandatfördelning."
    )


# ---------------------------------------------------------------------------
# Trajectory pool construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CampaignPathPool:
    """Aligned historical trajectory pool for one ``(origin, path_days)`` pair.

    ``delta_tensor[j, k]`` is the joint nine-category CLR displacement of
    trajectory ``j`` after ``trajectory_day_index[k]`` historical days, i.e.
    ``CLR(PoP[s_j + trajectory_day_index[k]]) - CLR(PoP[s_j])``.  Column
    ``k = path_days - 1`` is always the endpoint displacement and is asserted
    to equal production's stored transition vector exactly.
    """

    origin_date: date
    path_days: int
    endpoint_horizon_days: int
    time_warp: str
    start_dates: tuple[date, ...]
    trajectory_day_index: tuple[int, ...]
    delta_tensor: np.ndarray  # shape (M, path_days, 9)

    @property
    def size(self) -> int:
        return len(self.start_dates)


def resolve_endpoint_horizon(
    timeseries_data: Sequence[Mapping[str, Any]],
    origin_date: date,
    path_days: int,
) -> tuple[int, tuple[HistoricalTransition, ...]]:
    """Resolve production's Dynamics v2 endpoint horizon and eligible pool.

    This reproduces the canonical production selection exactly: cap the
    natural horizon at the 112-day empirical support, then walk the
    ``28 / 14 / 7`` fallback ladder when fewer than ``MIN_TRANSITIONS``
    leakage-safe transitions exist.
    """

    if path_days < 1:
        raise ValueError("path_days must be at least one day")
    horizon = min(max(1, path_days), HISTORY_DYNAMICS_CAP_DAYS)
    eligible = filter_transitions_as_of(
        build_all_historical_transitions(timeseries_data, horizons=[horizon])[horizon],
        origin_date,
    )
    if len(eligible) < MIN_TRANSITIONS:
        for fallback in FALLBACK_HORIZONS:
            candidate = filter_transitions_as_of(
                build_all_historical_transitions(timeseries_data, horizons=[fallback])[fallback],
                origin_date,
            )
            if len(candidate) >= MIN_TRANSITIONS:
                return fallback, candidate
        raise ValueError(
            "no leakage-safe Dynamics v2 transition pool reaches the minimum of "
            f"{MIN_TRANSITIONS} transitions at origin {origin_date.isoformat()}"
        )
    return horizon, eligible


def _trajectory_day_index(path_days: int, endpoint_horizon_days: int) -> tuple[tuple[int, ...], str]:
    """Map display day ``d = 1 .. path_days`` onto a historical trajectory day.

    When the remaining campaign is no longer than the empirically supported
    endpoint horizon the map is the identity, which is the only case that
    occurs for a Swedish general election inside the final 112 days.  Outside
    that window production itself already evaluates a shorter transition, so
    the path is a monotone stretch of that shorter historical trajectory whose
    final day is unchanged.  The stretch keeps endpoint parity exact while
    making the intermediate days deliberately smoother than a genuine
    ``path_days``-long historical movement.
    """

    if path_days == endpoint_horizon_days:
        return tuple(range(1, path_days + 1)), "identity"
    scale = endpoint_horizon_days / path_days
    mapped = [max(1, min(endpoint_horizon_days, int(round(day * scale)))) for day in range(1, path_days + 1)]
    mapped[-1] = endpoint_horizon_days
    for index in range(1, len(mapped)):
        mapped[index] = max(mapped[index], mapped[index - 1])
    return tuple(mapped), "monotone_stretch"


def _assert_endpoint_pool_parity(
    pool: Sequence[HistoricalTransition],
    delta_tensor: np.ndarray,
) -> None:
    """Fail closed unless the path endpoint equals production's transition."""

    production = np.asarray([transition.clr_transition for transition in pool], dtype=np.float64)
    endpoint = delta_tensor[:, -1, :]
    if production.shape != endpoint.shape or not np.array_equal(production, endpoint):
        raise ValueError(
            "campaign-path endpoint displacement is not bitwise identical to the "
            "production Dynamics v2 transition pool"
        )


def build_campaign_path_pool(
    timeseries_data: Sequence[Mapping[str, Any]],
    origin_date: date,
    path_days: int,
) -> CampaignPathPool:
    """Build the leakage-safe, production-aligned historical trajectory pool."""

    endpoint_horizon, pool = resolve_endpoint_horizon(timeseries_data, origin_date, path_days)
    day_index, warp = _trajectory_day_index(path_days, endpoint_horizon)

    rows_by_date = {row["date"]: row for row in timeseries_data}
    clr_cache: dict[date, np.ndarray] = {}

    def clr_at(observation_date: date) -> np.ndarray:
        cached = clr_cache.get(observation_date)
        if cached is None:
            row = rows_by_date.get(observation_date)
            if row is None:
                raise ValueError(
                    "the Poll of Polls series has a gap at "
                    f"{observation_date.isoformat()}; the campaign-path pool cannot "
                    "stay aligned with the production transition pool"
                )
            cached, _ = composition_to_clr(row["composition"])
            clr_cache[observation_date] = cached
        return cached

    start_dates = tuple(transition.start_date for transition in pool)
    tensor = np.empty((len(pool), path_days, len(ALL_CATEGORIES)), dtype=np.float64)
    for row, start in enumerate(start_dates):
        start_clr = clr_at(start)
        for column, trajectory_day in enumerate(day_index):
            tensor[row, column, :] = clr_at(start + timedelta(days=trajectory_day)) - start_clr

    _assert_endpoint_pool_parity(pool, tensor)

    for transition in pool:
        if transition.end_date > origin_date:
            raise ValueError("campaign-path pool contains a trajectory ending after the origin")

    return CampaignPathPool(
        origin_date=origin_date,
        path_days=path_days,
        endpoint_horizon_days=endpoint_horizon,
        time_warp=warp,
        start_dates=start_dates,
        trajectory_day_index=day_index,
        delta_tensor=tensor,
    )


def draw_trajectory_indices_and_signs(
    pool_size: int,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one trajectory index and one whole-path sign per Monte Carlo draw.

    The generator, the seed and the *order* of consumption reproduce
    ``scripts.hindcasts.models.sample_shared_symmetric_dynamics`` exactly, so
    ``signs * pool_deltas[indices]`` is bitwise equal to the production
    Dynamics v2 sample.  ``tests/test_campaign_paths.py`` locks that equality.
    """

    if pool_size < MIN_TRANSITIONS:
        raise ValueError(
            f"Insufficient historical transitions ({pool_size} < {MIN_TRANSITIONS})"
        )
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, pool_size, size=samples)
    signs = generator.choice([-1.0, 1.0], size=(samples, 1))
    return indices, signs


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CampaignPathSimulation:
    """Day-by-day opinion paths plus the bitwise-production election endpoint."""

    origin_date: date
    election_date: date
    path_days: int
    samples: int
    seed: int
    #: ``(path_days + 1, samples, 9)`` is never materialised.  ``day_dates[k]``
    #: describes ``coalition_draws[k]`` and ``composition_row_sums[k]``.
    day_dates: tuple[date, ...]
    coalition_draws: dict[str, np.ndarray]  # coalition -> (path_days + 1, samples)
    representative_indices: tuple[int, ...]
    endpoint_national_shares: np.ndarray    # (samples, 9) fractions, post ElectionNoise
    endpoint_opinion_composition: np.ndarray  # (samples, 9) percent, pre ElectionNoise
    diagnostics: dict[str, Any]


def simulate_campaign_paths(
    *,
    as_of: str | date,
    election_date: str | date,
    samples: int,
    seed: int = DEFAULT_SIMULATION_SEED,
    data_dir: Path | str | None = None,
    opinion_state: OpinionState | None = None,
    coalitions: Mapping[str, Sequence[str]] = DEFAULT_COALITIONS,
    representative_paths: int = DEFAULT_REPRESENTATIVE_PATHS,
    verify_endpoint_parity: bool = True,
) -> CampaignPathSimulation:
    """Simulate complete opinion paths from ``as_of`` through election day.

    With ``verify_endpoint_parity`` the ``d = n`` endpoint is re-derived through
    the canonical ``generate_national_vote_shares`` engine and compared bitwise.
    That comparison is the load-bearing scientific gate: it is independent of
    whoever produced the certified summaries, and it fails closed, so a wrong
    seed, a misaligned trajectory pool or a changed ElectionNoise horizon
    cannot reach publication.
    """

    origin = _coerce_date(as_of, name="as_of")
    election = _coerce_date(election_date, name="election_date")
    path_days = (election - origin).days
    if path_days < 1:
        raise ValueError("as_of must be strictly before election_date")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError("samples must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(representative_paths, int) or representative_paths <= 0:
        raise ValueError("representative_paths must be a positive integer")

    root = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parents[2] / "data" / "processed"
    state = opinion_state or estimate_opinion(as_of=origin, data_dir=root / "pollofpolls")
    if state.as_of != origin:
        raise ValueError("OpinionState cutoff differs from the requested frozen origin")

    # 1. Frozen OpinionState draws, byte-identical to production.
    state_seed = derive_opinion_state_seed(base_seed=seed, origin_date=origin)
    state_samples = state.sample(n=samples, seed=state_seed)
    state_matrix = np.array(
        [[draw[category] for category in ALL_CATEGORIES] for draw in state_samples],
        dtype=np.float64,
    )
    state_fractions = state_matrix / np.sum(state_matrix, axis=1, keepdims=True)
    log_state = np.log(state_fractions)
    state_clr = log_state - np.mean(log_state, axis=1, keepdims=True)

    # 2. Leakage-safe historical trajectory pool aligned with production.
    timeseries = load_timeseries_dataset(root / "pollofpolls" / "pollofpolls_timeseries.csv")
    pool = build_campaign_path_pool(timeseries, origin, path_days)

    # 3. Production's Dynamics v2 sub-seed and draw order.
    production_horizon = max(1, path_days)
    dynamics_seed = derive_shared_dynamics_seed(
        base_seed=seed,
        origin_date=origin,
        horizon_days=production_horizon,
    )
    indices, signs = draw_trajectory_indices_and_signs(pool.size, samples, dynamics_seed)

    coalition_config = {
        str(key): tuple(str(party) for party in members) for key, members in coalitions.items()
    }
    coalition_draws: dict[str, np.ndarray] = {
        key: np.empty((path_days + 1, samples), dtype=np.float64) for key in coalition_config
    }
    day_dates = tuple(origin + timedelta(days=offset) for offset in range(path_days + 1))

    max_row_sum_error = 0.0
    endpoint_opinion = np.empty((samples, len(ALL_CATEGORIES)), dtype=np.float64)
    for day_offset in range(path_days + 1):
        if day_offset == 0:
            day_clr = state_clr
        else:
            day_clr = state_clr + signs * pool.delta_tensor[:, day_offset - 1, :][indices]
        composition = clr_to_composition_matrix(day_clr)
        row_sums = np.sum(composition, axis=1)
        max_row_sum_error = max(max_row_sum_error, float(np.max(np.abs(row_sums - 100.0))))
        for key, members in coalition_config.items():
            coalition_draws[key][day_offset, :] = coalition_vote_draws(composition, members)
        if day_offset == path_days:
            endpoint_opinion = composition

    # 4. Election day: the adopted ElectionNoise law with production's seed.
    training_pool = load_chronological_pp_residuals(
        target_election_year=election.year,
        polls_file=root / "pollofpolls" / "swedishpolls_individual_polls.csv",
        elections_file=root / "elections" / "riksdag_election_results.csv",
    )
    noise_fit = fit_election_noise_b(training_pool.centered_residuals_matrix)
    noise_seed = derive_election_noise_b_seed(seed, origin, production_horizon)
    residuals = draw_election_noise_b(noise_fit, samples, np.random.default_rng(noise_seed))
    national, lambdas = apply_batch_simplex_transfer(endpoint_opinion, residuals, eps=MIN_SHARE_PCT)
    national = national / np.sum(national, axis=1, keepdims=True)

    count = min(representative_paths, samples)
    representative = tuple(
        sorted({int(value) for value in np.round(np.linspace(0, samples - 1, count)).astype(int)})
    )

    # Independent re-derivation of the election-day endpoint through the
    # canonical production engine.  Two separately written code paths sharing
    # the same frozen primitives must agree to the last bit.
    parity_verified = False
    parity_difference: float | None = None
    if verify_endpoint_parity:
        from scripts.vote_share_calibration.national_engine import generate_national_vote_shares

        reference = generate_national_vote_shares(
            as_of=origin,
            election_date=election,
            samples=samples,
            seed=seed,
            data_dir=root,
        )
        parity_difference = float(np.max(np.abs(reference.nat_shares_matrix - national)))
        if not np.array_equal(reference.nat_shares_matrix, national):
            raise ValueError(
                "campaign-path election-day draws are not bitwise identical to the "
                f"canonical production draws (max |difference| = {parity_difference})"
            )
        parity_verified = True

    diagnostics = {
        "model_id": CAMPAIGN_PATH_MODEL_ID,
        "state_cutoff_date": origin.isoformat(),
        "path_days": path_days,
        "endpoint_horizon_days": pool.endpoint_horizon_days,
        "time_warp": pool.time_warp,
        "eligible_trajectories": pool.size,
        "earliest_trajectory_start": pool.start_dates[0].isoformat(),
        "latest_trajectory_end": (
            pool.start_dates[-1] + timedelta(days=pool.endpoint_horizon_days)
        ).isoformat(),
        "opinion_state_seed": state_seed,
        "dynamics_seed": dynamics_seed,
        "dynamics_seed_horizon_days": production_horizon,
        "election_noise_model": ADOPTED_NOISE_MODEL,
        "election_noise_seed": noise_seed,
        "max_composition_sum_error_pp": max_row_sum_error,
        "mean_lambda": float(np.mean(lambdas)),
        "synthesized_future_polls": False,
        "daily_independent_random_walk": False,
        "directional_momentum": False,
        "endpoint_parity_verified": parity_verified,
        "endpoint_parity_max_abs_difference_pp": (
            None if parity_difference is None else parity_difference * 100.0
        ),
        "endpoint_parity_reference": "generate_national_vote_shares",
    }

    return CampaignPathSimulation(
        origin_date=origin,
        election_date=election,
        path_days=path_days,
        samples=samples,
        seed=seed,
        day_dates=day_dates,
        coalition_draws=coalition_draws,
        representative_indices=representative,
        endpoint_national_shares=national,
        endpoint_opinion_composition=endpoint_opinion,
        diagnostics=diagnostics,
    )


__all__ = [
    "BAND_LEGEND_SV",
    "ORIGIN_STATE_LABEL_SV",
    "CAMPAIGN_PATH_ASSUMPTION",
    "CAMPAIGN_PATH_MODEL_ID",
    "CAMPAIGN_PATH_TYPE",
    "CampaignPathPool",
    "CampaignPathSimulation",
    "DEFAULT_REPRESENTATIVE_PATHS",
    "ELECTION_DAY_LABEL_SV",
    "FALLBACK_HORIZONS",
    "FUTURE_REGION_LABEL_SV",
    "ORIGIN_BOUNDARY_LABEL_SV",
    "PATH_LEGEND_SV",
    "build_campaign_path_pool",
    "campaign_paths_tooltip_sv",
    "draw_trajectory_indices_and_signs",
    "election_day_tooltip_sv",
    "origin_state_tooltip_sv",
    "resolve_endpoint_horizon",
    "simulate_campaign_paths",
]
