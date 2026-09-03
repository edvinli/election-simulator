"""Per-party quantities for the historical forecast publication.

The chart in *Vägen till valdagen* was built for coalitions.  This module adds
the eight parliamentary parties as a second family of *definitions* on the same
time series, derived from the same joint simulation draws, so the website can
render either family through one renderer.

Two things make a party a different quantity from a one-party coalition, and
both are deliberate:

``vote share denominator``
    :func:`~scripts.forecast_history.contract.coalition_vote_draws` renormalizes
    over the eight parliamentary parties, so a coalition share answers "what
    fraction of the seats-eligible vote".  A **party** share is the ordinary
    national vote share over all nine model categories, ``REST`` included.  That
    is the number the certified publication reports in ``parties.json``, the
    number a poll prints, and the number the statutory 4 % threshold is defined
    on.  Publishing a renormalized party share would put the chart on a
    different scale from the forecast it illustrates and would silently move
    every party ~2 % away from the threshold line.

``seats``
    Identical in kind to the coalition case: one column of the same joint
    integer ``seats_matrix``.  Nothing is summed and nothing is re-allocated.

Everything here is additive.  No coalition contract changes, and every function
reads matrices that the caller already has; none of them simulates.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.simulator.config import MODEL_PARTIES_9

from .contract import (
    HISTORY_PARTY_ORDER,
    QUANTILE_LEVELS,
    _quantiles,
    _validate_iso_date,
    _validate_quantile_map,
)


#: Version of the party family of the history contract.  It is carried inside
#: the additive ``parties_view`` object rather than replacing the history
#: schema version, so a reader that knows nothing about parties is unaffected.
PARTY_VIEW_SCHEMA_VERSION = "1.0"
PARTY_VIEW_ROLE = "party_time_series"

#: The eight parliamentary parties are the only party definitions published.
#: ``REST`` is aggregate vote mass for modelled-ineligible parties: it cannot
#: reach the threshold, cannot hold seats, and is not a party a reader can
#: follow, so it is absent from this family exactly as it is absent from every
#: other threshold and seat surface.
PARTY_DEFINITION_ORDER: tuple[str, ...] = tuple(HISTORY_PARTY_ORDER)

#: The published denominator, stated in the artifact so a consumer cannot guess
#: wrong.  Nine categories, i.e. the whole modelled electorate.
PARTY_VOTE_DENOMINATOR = "all_nine_model_categories_including_rest"
PARTY_VOTE_DEFINITION = "national_vote_share"
PARTY_SEAT_DEFINITION = "statutory_mandate_allocation"

#: The statutory national threshold, on the same scale as the published party
#: vote share.  The website draws it only when the visible domain already
#: contains it; it is published here so the number is never hard-coded there.
NATIONAL_THRESHOLD_PCT = 4.0
THRESHOLD_LABEL_SV = "4 %-spärren"

PARTY_ELECTION_DAY_PARITY = "identical_to_certified_production_party_forecast"
PARTY_ELECTION_DAY_SOURCE = "certified_production_result_draw_matrices"

PARTY_PROVENANCE_NOTE_SV = (
    "Partiernas röstandelar är andelar av hela valmanskåren, med övriga partier "
    "i nämnaren. Det är samma definition som valdagsprognosen och som "
    "4 %-spärren, och därför en annan nämnare än koalitionernas andelar."
)

PARTY_NAMES_SV: dict[str, str] = {
    "M": "Moderaterna",
    "L": "Liberalerna",
    "C": "Centerpartiet",
    "KD": "Kristdemokraterna",
    "S": "Socialdemokraterna",
    "V": "Vänsterpartiet",
    "MP": "Miljöpartiet",
    "SD": "Sverigedemokraterna",
}

_QUANTILE_KEYS: tuple[str, ...] = tuple(name for name, _ in QUANTILE_LEVELS)


def _party_column(party: str) -> int:
    if party not in PARTY_DEFINITION_ORDER:
        raise ValueError(f"{party!r} is not one of the eight parliamentary parties")
    # The eight parliamentary parties occupy the first eight columns of the
    # nine-category model order, so one index addresses both matrices.
    return PARTY_DEFINITION_ORDER.index(party)


def party_vote_draws(
    vote_shares_matrix: Any,
    party: str,
    *,
    party_order: Sequence[str] = HISTORY_PARTY_ORDER,
) -> np.ndarray:
    """Return one party's national vote share for every joint vote draw.

    ``vote_shares_matrix`` must be the production **nine**-column matrix in
    percentage points.  An eight-column parliamentary matrix is rejected rather
    than silently accepted: without the ``REST`` column the rows do not sum to
    the whole electorate, so the values would be on the coalition denominator
    and would disagree with both ``parties.json`` and the 4 % threshold.
    """

    if tuple(party_order) != PARTY_DEFINITION_ORDER:
        raise ValueError(
            f"Party matrices must use the canonical party order {list(PARTY_DEFINITION_ORDER)}"
        )
    arr = np.asarray(vote_shares_matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] <= 0:
        raise ValueError("vote_shares_matrix must be a non-empty two-dimensional matrix")
    if arr.shape[1] != len(MODEL_PARTIES_9):
        raise ValueError(
            "A party vote share is defined on the nine-category model composition; "
            f"got {arr.shape[1]} columns, expected {len(MODEL_PARTIES_9)}"
        )
    if not np.isfinite(arr).all() or np.any(arr < 0):
        raise ValueError("vote_shares_matrix contains invalid vote shares")
    draws = arr[:, _party_column(party)]
    if np.any(draws > 100.0 + 1e-9):
        raise ValueError(f"Party {party} vote shares fall outside 0–100")
    return draws


def party_seat_draws(
    seats_matrix: Any,
    party: str,
    *,
    party_order: Sequence[str] = HISTORY_PARTY_ORDER,
) -> np.ndarray:
    """Return one party's seat count for every joint seat draw."""

    if tuple(party_order) != PARTY_DEFINITION_ORDER:
        raise ValueError(
            f"Party matrices must use the canonical party order {list(PARTY_DEFINITION_ORDER)}"
        )
    arr = np.asarray(seats_matrix)
    if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] != len(PARTY_DEFINITION_ORDER):
        raise ValueError(
            f"seats_matrix must have shape (N, {len(PARTY_DEFINITION_ORDER)}) with N > 0"
        )
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("seats_matrix must contain integer seat draws")
    if np.any(arr < 0) or np.any(arr > 349):
        raise ValueError("seats_matrix contains seats outside 0–349")
    if not np.all(np.sum(arr, axis=1, dtype=np.int64) == 349):
        raise ValueError("Every seat draw must contain exactly 349 seats")
    return arr[:, _party_column(party)]


def summarize_party_draws(vote_draws: Any, seat_draws: Any) -> dict[str, dict[str, int | float]]:
    """Return the compact vote and seat quantiles for one party."""

    votes = np.asarray(vote_draws, dtype=np.float64)
    seats = np.asarray(seat_draws)
    if votes.ndim != 1 or seats.ndim != 1 or votes.size != seats.size or votes.size <= 0:
        raise ValueError("Party vote and seat draws must be equally sized non-empty vectors")
    return {
        "vote": _quantiles(votes, integer=False),
        "seats": _quantiles(seats, integer=True),
    }


def build_parties_from_matrices(
    vote_shares_matrix: Any,
    seats_matrix: Any,
    *,
    party_order: Sequence[str] = HISTORY_PARTY_ORDER,
) -> dict[str, dict[str, dict[str, int | float]]]:
    """Build all eight party summaries directly from the same joint matrices.

    This is the *only* way a party point is produced from a simulation result.
    It reads the certified ``SimulationResult`` matrices, so the party
    quantiles at the certified production point are the production quantiles by
    construction rather than by a second, parallel computation.
    """

    votes = np.asarray(vote_shares_matrix, dtype=np.float64)
    seats = np.asarray(seats_matrix)
    if votes.shape[0] != seats.shape[0]:
        raise ValueError("Vote and seat matrices must contain the same number of draws")
    return {
        party: summarize_party_draws(
            party_vote_draws(votes, party, party_order=party_order),
            party_seat_draws(seats, party, party_order=party_order),
        )
        for party in PARTY_DEFINITION_ORDER
    }


def build_party_vote_quantiles(draws: Any) -> dict[str, float]:
    """Vote-only quantiles, for the campaign-path opinion bands."""

    values = np.asarray(draws, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("cannot summarize an empty party distribution")
    if not np.isfinite(values).all():
        raise ValueError("party distribution contains non-finite values")
    return {name: round(float(np.quantile(values, level)), 6) for name, level in QUANTILE_LEVELS}


def series_carries_parties(series: Any) -> bool:
    """Whether any point in ``series`` actually carries party summaries.

    The declaration follows the data. A resume run that regenerates nothing --
    every point served from the cache, none of them carrying a party block --
    must not announce a family it has no values for, so this is the single
    condition both payload builders gate on.
    """

    if not isinstance(series, Sequence) or isinstance(series, (str, bytes)):
        return False
    return any(
        isinstance(point, Mapping) and isinstance(point.get("parties"), Mapping) and point["parties"]
        for point in series
    )


def parties_view_metadata() -> dict[str, Any]:
    """The additive top-level object that declares the party family."""

    return {
        "schema_version": PARTY_VIEW_SCHEMA_VERSION,
        "role": PARTY_VIEW_ROLE,
        "party_order": list(PARTY_DEFINITION_ORDER),
        "party_names_sv": {party: PARTY_NAMES_SV[party] for party in PARTY_DEFINITION_ORDER},
        "vote_share_definition": PARTY_VOTE_DEFINITION,
        "vote_share_denominator": PARTY_VOTE_DENOMINATOR,
        "seat_definition": PARTY_SEAT_DEFINITION,
        "national_threshold_pct": NATIONAL_THRESHOLD_PCT,
        "threshold_label_sv": THRESHOLD_LABEL_SV,
        "rest_is_a_party": False,
        "election_day_parity": {
            "guarantee": PARTY_ELECTION_DAY_PARITY,
            "source": PARTY_ELECTION_DAY_SOURCE,
            "reconstructed_from_coalitions": False,
        },
        "intermediate_seat_trajectory": False,
        "provenance_note_sv": PARTY_PROVENANCE_NOTE_SV,
    }


# ---------------------------------------------------------------------------
# Archived prospective points
# ---------------------------------------------------------------------------


def party_point_from_archive_record(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Recover party quantiles from an immutable prospective archive snapshot.

    Unlike coalition intervals, party marginals *are* recoverable from an
    archived snapshot: the archive stores each party's own vote and seat
    quantiles, computed from the same draws by the same helper.  A snapshot
    that predates those fields returns ``None`` and the point simply carries no
    party block, which the consumer feature-detects.
    """

    if not isinstance(record, Mapping):
        return None
    source: Mapping[str, Any] = record
    point = record.get("forecast_point")
    if isinstance(point, Mapping) and "national_vote_distributions" in point:
        source = point
    votes = source.get("national_vote_distributions")
    seats = source.get("seat_distributions")
    if not isinstance(votes, Mapping) or not isinstance(seats, Mapping):
        return None
    parties: dict[str, Any] = {}
    for party in PARTY_DEFINITION_ORDER:
        vote_entry = votes.get(party)
        seat_entry = seats.get(party)
        if not isinstance(vote_entry, Mapping) or not isinstance(seat_entry, Mapping):
            return None
        vote_quantiles = vote_entry.get("quantiles")
        seat_quantiles = seat_entry.get("quantiles")
        if not isinstance(vote_quantiles, Mapping) or not isinstance(seat_quantiles, Mapping):
            return None
        if any(key not in vote_quantiles for key in _QUANTILE_KEYS):
            return None
        if any(key not in seat_quantiles for key in _QUANTILE_KEYS):
            return None
        parties[party] = {
            "vote": {key: round(float(vote_quantiles[key]), 6) for key in _QUANTILE_KEYS},
            # Archived seat quantiles are stored as floats by the shared
            # histogram helper; the published seat contract is integral.
            "seats": {key: int(float(seat_quantiles[key])) for key in _QUANTILE_KEYS},
        }
    return parties


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_party_summaries(value: Any, *, name: str) -> None:
    """Validate one ``parties`` block: all eight parties, vote and seats."""

    if not isinstance(value, Mapping) or list(value) != list(PARTY_DEFINITION_ORDER):
        raise ValueError(f"{name} must contain the eight parliamentary parties in order")
    for party in PARTY_DEFINITION_ORDER:
        entry = value[party]
        if not isinstance(entry, Mapping) or list(entry) != ["vote", "seats"]:
            raise ValueError(f"{name}.{party} must contain vote and seats")
        _validate_quantile_map(
            entry["vote"], name=f"{name}.{party}.vote", integer=False, lower=0.0, upper=100.0
        )
        _validate_quantile_map(
            entry["seats"], name=f"{name}.{party}.seats", integer=True, lower=0.0, upper=349.0
        )


def validate_party_vote_only(value: Any, *, name: str) -> None:
    """Validate a vote-only ``parties`` block, as used by the opinion bands.

    A seat quantile here would imply an intermediate future seat trajectory,
    which the model refuses to publish: the statutory allocator is defined on
    an election result, not on a poll average.
    """

    if not isinstance(value, Mapping) or list(value) != list(PARTY_DEFINITION_ORDER):
        raise ValueError(f"{name} must contain the eight parliamentary parties in order")
    for party in PARTY_DEFINITION_ORDER:
        entry = value[party]
        if not isinstance(entry, Mapping) or list(entry) != ["vote"]:
            raise ValueError(
                f"{name}.{party} must contain vote only; a seat quantile in an opinion "
                "band would imply an intermediate future mandate trajectory"
            )
        _validate_quantile_map(
            entry["vote"], name=f"{name}.{party}.vote", integer=False, lower=0.0, upper=100.0
        )


def validate_parties_view(payload: Mapping[str, Any]) -> None:
    """Validate the additive ``parties_view`` declaration and every party block.

    The party family is optional.  A history artifact without it is valid and
    is exactly the artifact the previous website consumed; this validator only
    has an opinion once ``parties_view`` is present, and then it is strict.
    """

    if "parties_view" not in payload:
        # Nothing declared: no point may carry a party block either, or the
        # consumer would find data it has been given no definition for.
        for index, point in enumerate(payload.get("series") or []):
            if isinstance(point, Mapping) and "parties" in point:
                raise ValueError(
                    f"series[{index}] carries party summaries without a parties_view declaration"
                )
        return

    view = payload["parties_view"]
    if not isinstance(view, Mapping):
        raise ValueError("parties_view must be an object")
    if view.get("schema_version") != PARTY_VIEW_SCHEMA_VERSION:
        raise ValueError(f"Unsupported parties_view schema: {view.get('schema_version')!r}")
    if view.get("role") != PARTY_VIEW_ROLE:
        raise ValueError("parties_view.role must be the party time-series role")
    if view.get("party_order") != list(PARTY_DEFINITION_ORDER):
        raise ValueError(f"parties_view.party_order must be {list(PARTY_DEFINITION_ORDER)}")
    if view.get("vote_share_denominator") != PARTY_VOTE_DENOMINATOR:
        raise ValueError(
            "parties_view.vote_share_denominator must state the nine-category denominator; "
            "a renormalized party share would disagree with parties.json and the 4 % threshold"
        )
    if view.get("vote_share_definition") != PARTY_VOTE_DEFINITION:
        raise ValueError("parties_view.vote_share_definition must be the national vote share")
    if view.get("seat_definition") != PARTY_SEAT_DEFINITION:
        raise ValueError("parties_view.seat_definition must be the statutory allocation")
    if view.get("rest_is_a_party") is not False:
        raise ValueError("REST is aggregate vote mass and is never published as a party")
    if view.get("intermediate_seat_trajectory") is not False:
        raise ValueError("parties_view must not declare an intermediate seat trajectory")
    threshold = view.get("national_threshold_pct")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or float(threshold) != NATIONAL_THRESHOLD_PCT:
        raise ValueError(f"parties_view.national_threshold_pct must be {NATIONAL_THRESHOLD_PCT}")
    parity = view.get("election_day_parity")
    if not isinstance(parity, Mapping):
        raise ValueError("parties_view.election_day_parity must be an object")
    if parity.get("guarantee") != PARTY_ELECTION_DAY_PARITY:
        raise ValueError("parties_view.election_day_parity.guarantee is not the published guarantee")
    if parity.get("source") != PARTY_ELECTION_DAY_SOURCE:
        raise ValueError("party election-day values must come from the certified draw matrices")
    if parity.get("reconstructed_from_coalitions") is not False:
        raise ValueError("party uncertainty is never reconstructed from coalition quantities")
    names = view.get("party_names_sv")
    if not isinstance(names, Mapping) or list(names) != list(PARTY_DEFINITION_ORDER):
        raise ValueError("parties_view.party_names_sv must name the eight parties in order")
    for party in PARTY_DEFINITION_ORDER:
        if not isinstance(names[party], str) or not names[party].strip():
            raise ValueError(f"parties_view.party_names_sv.{party} must be a non-empty label")
    note = view.get("provenance_note_sv")
    if not isinstance(note, str) or not note.strip():
        raise ValueError("parties_view.provenance_note_sv must be reader-facing Swedish copy")

    series = payload.get("series")
    if not isinstance(series, list) or not series:
        raise ValueError("series must be a non-empty list")
    covered = 0
    election_date = _validate_iso_date(payload["election_date"], name="election_date")
    for index, point in enumerate(series):
        if not isinstance(point, Mapping) or "parties" not in point:
            continue
        validate_party_summaries(point["parties"], name=f"series[{index}].parties")
        point_date = _validate_iso_date(point["date"], name=f"series[{index}].date")
        if point_date > election_date:
            raise ValueError(f"series[{index}] carries party summaries after election_date")
        covered += 1
    if not covered:
        raise ValueError("parties_view is declared but no series point carries party summaries")
    current = [
        point
        for point in series
        if isinstance(point, Mapping) and point.get("provenance") == "current_production"
    ]
    # The certified point is the one the reader compares against the published
    # forecast, so it is the one point that may never be missing.
    if current and "parties" not in current[0]:
        raise ValueError(
            "the certified current_production point must carry party summaries when "
            "parties_view is declared"
        )


def assert_election_day_party_parity(
    published: Mapping[str, Any],
    certified_party_rows: Sequence[Mapping[str, Any]],
    *,
    decimals: int = 3,
) -> None:
    """Fail closed unless the published party point *is* the certified forecast.

    ``certified_party_rows`` are the rows of the publication's ``parties.json``.
    Both sides derive from the same draws; the published history keeps six
    decimals while ``parties.json`` keeps three, so the vote comparison is made
    at the coarser published precision and the seat comparison is exact.
    """

    rows = {str(row.get("party")): row for row in certified_party_rows}
    for party in PARTY_DEFINITION_ORDER:
        row = rows.get(party)
        if row is None:
            raise ValueError(f"certified publication has no party row for {party}")
        entry = published.get(party)
        if not isinstance(entry, Mapping):
            raise ValueError(f"published history has no party block for {party}")
        for key, field in (
            ("p05", "vote_share_p05"),
            ("p25", "vote_share_p25"),
            ("p50", "vote_share_median"),
            ("p75", "vote_share_p75"),
            ("p95", "vote_share_p95"),
        ):
            published_value = round(float(entry["vote"][key]), decimals)
            certified_value = round(float(row[field]), decimals)
            if published_value != certified_value:
                raise ValueError(
                    f"{party} election-day vote {key} is {published_value}, but the certified "
                    f"forecast publishes {certified_value}"
                )
        for key, field in (
            ("p05", "seats_p05"),
            ("p25", "seats_p25"),
            ("p50", "seats_median"),
            ("p75", "seats_p75"),
            ("p95", "seats_p95"),
        ):
            published_seats = int(entry["seats"][key])
            certified_seats = int(row[field])
            if published_seats != certified_seats:
                raise ValueError(
                    f"{party} election-day seats {key} is {published_seats}, but the certified "
                    f"forecast publishes {certified_seats}"
                )


__all__ = [
    "NATIONAL_THRESHOLD_PCT",
    "PARTY_DEFINITION_ORDER",
    "PARTY_ELECTION_DAY_PARITY",
    "PARTY_ELECTION_DAY_SOURCE",
    "PARTY_NAMES_SV",
    "PARTY_PROVENANCE_NOTE_SV",
    "PARTY_SEAT_DEFINITION",
    "PARTY_VIEW_ROLE",
    "PARTY_VIEW_SCHEMA_VERSION",
    "PARTY_VOTE_DEFINITION",
    "PARTY_VOTE_DENOMINATOR",
    "THRESHOLD_LABEL_SV",
    "assert_election_day_party_parity",
    "build_parties_from_matrices",
    "build_party_vote_quantiles",
    "parties_view_metadata",
    "party_point_from_archive_record",
    "party_seat_draws",
    "party_vote_draws",
    "series_carries_parties",
    "summarize_party_draws",
    "validate_parties_view",
    "validate_party_summaries",
    "validate_party_vote_only",
]
