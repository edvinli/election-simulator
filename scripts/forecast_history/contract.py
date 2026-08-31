"""Schema and mathematical helpers for the historical forecast publication.

The normal forecast publication has a deliberately frozen contract.  This
module defines a separate, small contract for the time-series chart so that
the chart can evolve without changing the publication consumed by the rest of
the website.

The functions in this module are intentionally independent of the simulator
runner.  They accept already-produced joint draw matrices and derive every
coalition distribution from those same draws.  In particular, no party-level
quantiles are ever added together and the vote denominator never includes
``REST`` (or an invented ``Other`` category).
"""

from __future__ import annotations

from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.simulator.config import PARLIAMENTARY_PARTIES_8


HISTORY_SCHEMA_VERSION = "1.1"
HISTORY_DYNAMICS_CAP_DAYS = 112
HISTORY_PARTY_ORDER: tuple[str, ...] = tuple(PARLIAMENTARY_PARTIES_8)
QUANTILE_LEVELS: tuple[tuple[str, float], ...] = (
    ("p05", 0.05),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p95", 0.95),
)

# The names are stable public identifiers.  Their labels and colours belong
# to the website, while this mapping is the canonical party membership used
# by the offline publisher and by tests.
DEFAULT_COALITIONS: dict[str, tuple[str, ...]] = {
    "red_green_center": ("V", "MP", "S", "C"),
    "tido": ("L", "KD", "M", "SD"),
    "s_m": ("S", "M"),
    "v_s_mp": ("V", "S", "MP"),
    "s_mp_c": ("S", "MP", "C"),
    "c_kd_l_m": ("C", "KD", "L", "M"),
    "s_mp_c_kd": ("S", "MP", "C", "KD"),
}

PROVENANCE_VALUES: tuple[str, ...] = (
    "reconstructed_current_model",
    "prospective_archived",
    "current_production",
)


def _as_party_order(party_order: Sequence[str] | None) -> tuple[str, ...]:
    order = tuple(party_order or HISTORY_PARTY_ORDER)
    if order != HISTORY_PARTY_ORDER:
        raise ValueError(
            "Historical forecast matrices must use the canonical party order "
            f"{list(HISTORY_PARTY_ORDER)}"
        )
    return order


def _validate_matrix(matrix: Any, *, columns: int, name: str, integer: bool = False) -> np.ndarray:
    arr = np.asarray(matrix)
    if arr.ndim != 2 or arr.shape[1] != columns or arr.shape[0] <= 0:
        raise ValueError(f"{name} must have shape (N, {columns}) with N > 0")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    if integer:
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"{name} must contain integer seat draws")
        if np.any(arr < 0) or np.any(arr > 349):
            raise ValueError(f"{name} contains seats outside 0–349")
    return arr


def _party_indices(parties: Sequence[str], party_order: Sequence[str]) -> list[int]:
    order = _as_party_order(party_order)
    members = tuple(parties)
    if not members:
        raise ValueError("A coalition must contain at least one party")
    if len(set(members)) != len(members):
        raise ValueError(f"Coalition contains duplicate parties: {members!r}")
    unknown = [party for party in members if party not in order]
    if unknown:
        raise ValueError(f"Coalition contains unknown parties: {unknown!r}")
    return [order.index(party) for party in members]


def coalition_vote_draws(
    vote_shares_matrix: Any,
    parties: Sequence[str],
    *,
    party_order: Sequence[str] = HISTORY_PARTY_ORDER,
) -> np.ndarray:
    """Calculate one coalition vote share for every joint vote draw.

    ``vote_shares_matrix`` may be the production nine-column matrix (where
    ``REST`` is the final column) or the eight-column parliamentary matrix.
    The denominator is *always* the sum of the eight parliamentary columns.
    The result is in percentage points.
    """

    order = _as_party_order(party_order)
    arr = np.asarray(vote_shares_matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] not in {len(order), len(order) + 1} or arr.shape[0] <= 0:
        raise ValueError(
            "vote_shares_matrix must have shape (N, 8) or (N, 9) with N > 0"
        )
    if not np.isfinite(arr).all() or np.any(arr < 0):
        raise ValueError("vote_shares_matrix contains invalid vote shares")
    indices = _party_indices(parties, order)
    parliamentary = arr[:, : len(order)]
    denominator = np.sum(parliamentary, axis=1, dtype=np.float64)
    if np.any(denominator <= 0):
        raise ValueError("Vote-share denominator must be positive for every draw")
    draws = 100.0 * np.sum(parliamentary[:, indices], axis=1) / denominator
    if not np.isfinite(draws).all() or np.any(draws < -1e-9) or np.any(draws > 100.0 + 1e-9):
        raise ValueError("Coalition vote shares fall outside 0–100")
    return draws


def coalition_seat_draws(
    seats_matrix: Any,
    parties: Sequence[str],
    *,
    party_order: Sequence[str] = HISTORY_PARTY_ORDER,
) -> np.ndarray:
    """Calculate coalition seats by summing columns in the original joint draws."""

    order = _as_party_order(party_order)
    arr = _validate_matrix(
        seats_matrix,
        columns=len(order),
        name="seats_matrix",
        integer=True,
    )
    if not np.all(np.sum(arr, axis=1, dtype=np.int64) == 349):
        raise ValueError("Every seat draw must contain exactly 349 seats")
    indices = _party_indices(parties, order)
    return np.sum(arr[:, indices], axis=1, dtype=np.int64)


def _quantiles(values: Any, *, integer: bool) -> dict[str, int | float]:
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.size <= 0:
        raise ValueError("Cannot calculate quantiles for an empty distribution")
    if not np.isfinite(arr).all():
        raise ValueError("Distribution contains non-finite values")
    result: dict[str, int | float] = {}
    for name, level in QUANTILE_LEVELS:
        value = float(np.quantile(arr, level))
        # Existing seat summaries use int(np.percentile(...)); retain that
        # public convention while vote shares retain sub-percentage precision.
        result[name] = int(value) if integer else round(value, 6)
    return result


def summarize_coalition_draws(
    vote_draws: Any,
    seat_draws: Any,
) -> dict[str, dict[str, int | float]]:
    """Return the compact vote and seat quantiles for one coalition."""

    votes = np.asarray(vote_draws, dtype=np.float64)
    seats = np.asarray(seat_draws)
    if votes.ndim != 1 or seats.ndim != 1 or votes.size != seats.size or votes.size <= 0:
        raise ValueError("Coalition vote and seat draws must be equally sized non-empty vectors")
    if np.any(votes < -1e-9) or np.any(votes > 100.0 + 1e-9):
        raise ValueError("Coalition vote draws fall outside 0–100")
    if not np.issubdtype(seats.dtype, np.integer) or np.any(seats < 0) or np.any(seats > 349):
        raise ValueError("Coalition seat draws must be integer values in 0–349")
    return {
        "vote": _quantiles(votes, integer=False),
        "seats": _quantiles(seats, integer=True),
    }


def build_groups_from_matrices(
    vote_shares_matrix: Any,
    seats_matrix: Any,
    *,
    coalitions: Mapping[str, Sequence[str]] = DEFAULT_COALITIONS,
    party_order: Sequence[str] = HISTORY_PARTY_ORDER,
) -> dict[str, dict[str, dict[str, int | float]]]:
    """Build all coalition summaries directly from the same joint matrices."""

    order = _as_party_order(party_order)
    votes = np.asarray(vote_shares_matrix, dtype=np.float64)
    seats = _validate_matrix(
        seats_matrix,
        columns=len(order),
        name="seats_matrix",
        integer=True,
    )
    if votes.shape[0] != seats.shape[0]:
        raise ValueError("Vote and seat matrices must contain the same number of draws")
    result: dict[str, dict[str, dict[str, int | float]]] = {}
    for key, members in coalitions.items():
        vote_draws = coalition_vote_draws(votes, members, party_order=order)
        seat_draws = coalition_seat_draws(seats, members, party_order=order)
        result[str(key)] = summarize_coalition_draws(vote_draws, seat_draws)
    return result


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def deterministic_history_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a history payload while excluding its self-referential hash."""

    copy = dict(payload)
    copy.pop("deterministic_content_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(copy)).hexdigest()


def _validate_quantile_map(
    value: Any,
    *,
    name: str,
    integer: bool,
    lower: float,
    upper: float,
) -> None:
    if not isinstance(value, Mapping) or list(value) != [name for name, _ in QUANTILE_LEVELS]:
        raise ValueError(f"{name} must contain p05, p25, p50, p75, p95 in order")
    values: list[float] = []
    for key, _ in QUANTILE_LEVELS:
        current = value.get(key)
        if not _finite_number(current):
            raise ValueError(f"{name}.{key} must be a finite number")
        if integer and (not isinstance(current, int) or isinstance(current, bool)):
            raise ValueError(f"{name}.{key} must be an integer seat count")
        number = float(current)
        if number < lower or number > upper:
            raise ValueError(f"{name}.{key} is outside {lower}–{upper}")
        values.append(number)
    if values != sorted(values):
        raise ValueError(f"{name} quantiles must be monotone")


def _validate_iso_date(value: Any, *, name: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date string") from exc


def _validate_poll(value: Any, index: int, *, latest_date: date | None) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"polls[{index}] must be an object")
    required = (
        "poll_id",
        "company",
        "publication_date",
        "fieldwork_start",
        "fieldwork_end",
        "n",
        "parties",
    )
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"polls[{index}] is missing {missing}")
    if not isinstance(value["poll_id"], str) or not value["poll_id"]:
        raise ValueError(f"polls[{index}].poll_id must be a non-empty string")
    if not isinstance(value["company"], str) or not value["company"].strip():
        raise ValueError(f"polls[{index}].company must be a non-empty string")
    for key in ("publication_date", "fieldwork_start", "fieldwork_end"):
        current = value[key]
        if current is not None:
            parsed = _validate_iso_date(current, name=f"polls[{index}].{key}")
            if latest_date is not None and parsed > latest_date:
                # Poll observations may be newer than a stale forecast point;
                # they are allowed in the standalone poll layer.  Do not
                # reject them as this would silently discard source history.
                pass
    sample = value["n"]
    if sample is not None and (
        not isinstance(sample, int) or isinstance(sample, bool) or sample < 0
    ):
        raise ValueError(f"polls[{index}].n must be a non-negative integer or null")
    parties = value["parties"]
    if not isinstance(parties, Mapping) or list(parties) != list(HISTORY_PARTY_ORDER):
        raise ValueError(
            f"polls[{index}].parties must contain exactly the eight parliamentary parties"
        )
    for party in HISTORY_PARTY_ORDER:
        support = parties[party]
        if not _finite_number(support) or float(support) < 0 or float(support) > 100:
            raise ValueError(f"polls[{index}].parties.{party} must be between 0 and 100")


def validate_history_contract(payload: Mapping[str, Any]) -> None:
    """Validate the complete schema 1.0 history payload.

    Validation is deliberately strict about mathematical and provenance
    fields, while allowing additive metadata keys so future chart metadata can
    be added without invalidating old readers.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("Historical forecast payload must be a JSON object")
    required = (
        "schema_version",
        "election_date",
        "model_commit",
        "poll_source_sha256",
        "party_order",
        "coalitions",
        "series",
        "poll_of_polls",
        "polls",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Historical forecast payload is missing {missing}")
    if payload.get("schema_version") != HISTORY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported historical forecast schema: {payload.get('schema_version')!r}")
    election_date = _validate_iso_date(payload["election_date"], name="election_date")
    model_commit = payload["model_commit"]
    if not isinstance(model_commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", model_commit):
        raise ValueError("model_commit must be a resolvable 40- or 64-character hexadecimal commit")
    if not _valid_sha256(payload["poll_source_sha256"]):
        raise ValueError("poll_source_sha256 must be a 64-character hexadecimal SHA-256")
    if payload["party_order"] != list(HISTORY_PARTY_ORDER):
        raise ValueError(f"party_order must be {list(HISTORY_PARTY_ORDER)}")

    coalitions = payload["coalitions"]
    if not isinstance(coalitions, Mapping) or not coalitions:
        raise ValueError("coalitions must be a non-empty object")
    for key, members in coalitions.items():
        if not isinstance(key, str) or not key:
            raise ValueError("coalition identifiers must be non-empty strings")
        if not isinstance(members, list):
            raise ValueError(f"coalition {key} membership must be a list")
        _party_indices(members, HISTORY_PARTY_ORDER)

    series = payload["series"]
    if not isinstance(series, list) or not series:
        raise ValueError("series must be a non-empty list")
    prior_key: tuple[str, str] | None = None
    seen_point_keys: set[tuple[str, str]] = set()
    latest_series_date: date | None = None
    for index, point in enumerate(series):
        if not isinstance(point, Mapping):
            raise ValueError(f"series[{index}] must be an object")
        required_point = (
            "date",
            "samples",
            "horizon_days",
            "dynamics_horizon_days",
            "provenance",
            "groups",
        )
        missing_point = [key for key in required_point if key not in point]
        if missing_point:
            raise ValueError(f"series[{index}] is missing {missing_point}")
        point_date = _validate_iso_date(point["date"], name=f"series[{index}].date")
        if point_date > election_date:
            raise ValueError(f"series[{index}] occurs after election_date")
        samples = point["samples"]
        if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
            raise ValueError(f"series[{index}].samples must be a positive integer")
        horizon = point["horizon_days"]
        dynamics_horizon = point["dynamics_horizon_days"]
        if (
            not isinstance(horizon, int)
            or isinstance(horizon, bool)
            or horizon != max(0, (election_date - point_date).days)
        ):
            raise ValueError(f"series[{index}].horizon_days disagrees with date and election_date")
        if (
            not isinstance(dynamics_horizon, int)
            or isinstance(dynamics_horizon, bool)
            or dynamics_horizon < 0
            or dynamics_horizon > min(horizon, HISTORY_DYNAMICS_CAP_DAYS)
        ):
            raise ValueError(f"series[{index}].dynamics_horizon_days exceeds the 112-day cap")
        provenance = point["provenance"]
        if provenance not in PROVENANCE_VALUES:
            raise ValueError(f"series[{index}].provenance is not recognised")
        groups = point["groups"]
        if not isinstance(groups, Mapping) or list(groups) != list(coalitions):
            raise ValueError(f"series[{index}].groups must cover the configured coalitions in order")
        for coalition_key in coalitions:
            group = groups[coalition_key]
            if not isinstance(group, Mapping) or list(group) != ["vote", "seats"]:
                raise ValueError(f"series[{index}].groups.{coalition_key} must contain vote and seats")
            _validate_quantile_map(
                group["vote"],
                name=f"series[{index}].groups.{coalition_key}.vote",
                integer=False,
                lower=0.0,
                upper=100.0,
            )
            _validate_quantile_map(
                group["seats"],
                name=f"series[{index}].groups.{coalition_key}.seats",
                integer=True,
                lower=0.0,
                upper=349.0,
            )
        ordering_key = (point["date"], str(point["provenance"]))
        if ordering_key in seen_point_keys:
            raise ValueError("series contains duplicate (date, provenance) points")
        seen_point_keys.add(ordering_key)
        if prior_key is not None and ordering_key < prior_key:
            raise ValueError("series must be ordered by date and provenance")
        prior_key = ordering_key
        latest_series_date = point_date if latest_series_date is None else max(latest_series_date, point_date)

    poll_of_polls = payload["poll_of_polls"]
    if not isinstance(poll_of_polls, list) or not poll_of_polls:
        raise ValueError("poll_of_polls must be a non-empty list")
    seen_pop_dates: set[str] = set()
    prior_pop_date: str | None = None
    for index, item in enumerate(poll_of_polls):
        if not isinstance(item, Mapping):
            raise ValueError(f"poll_of_polls[{index}] must be an object")
        if "date" not in item or "parties" not in item:
            raise ValueError(f"poll_of_polls[{index}] must contain date and parties")
        pop_date = _validate_iso_date(item["date"], name=f"poll_of_polls[{index}].date")
        if pop_date > election_date:
            raise ValueError(f"poll_of_polls[{index}] occurs after election_date")
        iso_str = item["date"]
        if iso_str in seen_pop_dates:
            raise ValueError(f"poll_of_polls contains duplicate date {iso_str!r}")
        seen_pop_dates.add(iso_str)
        if prior_pop_date is not None and iso_str <= prior_pop_date:
            raise ValueError("poll_of_polls must be strictly sorted by date")
        prior_pop_date = iso_str

        parties = item["parties"]
        if not isinstance(parties, Mapping) or list(parties) != list(HISTORY_PARTY_ORDER):
            raise ValueError(
                f"poll_of_polls[{index}].parties must contain exactly the eight parliamentary parties in order"
            )
        denom = 0.0
        for party in HISTORY_PARTY_ORDER:
            val = parties[party]
            if not _finite_number(val) or float(val) < 0:
                raise ValueError(f"poll_of_polls[{index}].parties.{party} must be a non-negative finite number")
            denom += float(val)
        if denom <= 0:
            raise ValueError(f"poll_of_polls[{index}] eight-party denominator must be positive")

    polls = payload["polls"]
    if not isinstance(polls, list):
        raise ValueError("polls must be a list")
    poll_ids: set[str] = set()
    for index, poll in enumerate(polls):
        _validate_poll(poll, index, latest_date=latest_series_date)
        poll_id = poll["poll_id"]
        if poll_id in poll_ids:
            raise ValueError(f"polls contains duplicate poll_id {poll_id!r}")
        poll_ids.add(poll_id)

    digest = payload.get("deterministic_content_sha256")
    if digest is not None:
        if not _valid_sha256(digest):
            raise ValueError("deterministic_content_sha256 must be a 64-character hexadecimal SHA-256")
        if digest != deterministic_history_sha256(payload):
            raise ValueError("deterministic_content_sha256 does not match payload")


def write_history_json(path: Path | str, payload: Mapping[str, Any]) -> Path:
    """Validate and atomically write a compact JSON history artifact."""

    validate_history_contract(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    temporary.write_text(body, encoding="utf-8")
    temporary.replace(destination)
    return destination


__all__ = [
    "DEFAULT_COALITIONS",
    "HISTORY_DYNAMICS_CAP_DAYS",
    "HISTORY_PARTY_ORDER",
    "HISTORY_SCHEMA_VERSION",
    "PROVENANCE_VALUES",
    "QUANTILE_LEVELS",
    "build_groups_from_matrices",
    "coalition_seat_draws",
    "coalition_vote_draws",
    "deterministic_history_sha256",
    "summarize_coalition_draws",
    "validate_history_contract",
    "write_history_json",
]
