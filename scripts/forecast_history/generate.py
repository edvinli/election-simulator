"""Offline generation of the coalition forecast time-series artifact.

The website consumes the JSON produced here; it never calls the simulator.
Reconstructed points invoke the existing canonical simulator with an explicit
historical ``as_of`` date.  Coalition summaries are derived from the returned
joint vote and seat matrices by :mod:`scripts.forecast_history.contract`.

Provenance classification:
- ``reconstructed_current_model``: Historical backfill simulations using the
  current model and finalized historical Poll of Polls timeseries.
- ``prospective_archived``: Genuine prospective archived forecast runs. These
  are only substituted when the archive contains the required joint coalition
  draws / distributions needed to compute non-linear coalition quantiles correctly;
  marginal party quantiles are never summed to fabricate coalition intervals.
- ``current_production``: The official, full-sample production forecast point
  for the latest available date.

The default CLI schedule is deliberately sparse before the 112-day model
horizon (weekly) and daily inside that horizon.  It is therefore practical to
run as a publication step while avoiding a 100,000-draw simulation for every
calendar day.  The current production forecast remains entirely separate and
is never replaced by this reduced-sample history.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from scripts.pollofpolls.normalize import normalize_party, parse_date
from scripts.simulator.config import (
    DEFAULT_ELECTION_DATE,
    DEFAULT_SIMULATION_SEED,
)
from scripts.simulator.engine import simulate_election
from scripts.simulator.reproducibility import (
    compute_file_sha256,
    get_git_commit_hash,
    is_git_worktree_clean,
)

from .contract import (
    DEFAULT_COALITIONS,
    HISTORY_DYNAMICS_CAP_DAYS,
    HISTORY_PARTY_ORDER,
    HISTORY_SCHEMA_VERSION,
    build_groups_from_matrices,
    deterministic_history_sha256,
    validate_history_contract,
    write_history_json,
)
from .party_contract import (
    build_parties_from_matrices,
    parties_view_metadata,
    party_point_from_archive_record,
    series_carries_parties,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_ROOT = REPOSITORY_ROOT / "data" / "processed"
DEFAULT_POLL_FILE = DEFAULT_PROCESSED_ROOT / "pollofpolls" / "swedishpolls_individual_polls.csv"
DEFAULT_TIMESERIES_FILE = DEFAULT_PROCESSED_ROOT / "pollofpolls" / "pollofpolls_timeseries.csv"
DEFAULT_ARCHIVE_DIR = DEFAULT_PROCESSED_ROOT / "prospective_forecasts"
DEFAULT_HISTORY_OUTPUT = REPOSITORY_ROOT / "files" / "election-simulator" / "history" / "coalition-timeseries.json"
HISTORY_START_DATE = date(2022, 9, 18)
HISTORY_CAP_DATE = date(2026, 5, 24)
DEFAULT_HISTORY_SAMPLES = 10_000


def _coerce_date(value: str | date, *, name: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date") from exc


def _first(row: Mapping[str, Any], *names: str) -> Any:
    """Get a field from normalized or source-style spelling."""

    lowered = {str(key).strip().casefold(): value for key, value in row.items()}
    for name in names:
        if name.casefold() in lowered:
            return lowered[name.casefold()]
    return None


def _parse_optional_date(value: Any) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    raw = str(value).strip()
    try:
        return parse_date(raw).isoformat()
    except ValueError:
        try:
            return date.fromisoformat(raw).isoformat()
        except ValueError as exc:
            raise ValueError(f"Invalid poll date {raw!r}") from exc


def _parse_support(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip().replace(",", ".")
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not np.isfinite(parsed) or parsed < 0 or parsed > 100:
        return None
    return parsed


def _parse_sample_size(value: Any) -> int | None:
    if value is None or str(value).strip() in {"", "None", "null", "NA", "na"}:
        return None
    try:
        parsed = int(float(str(value).strip()))
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _poll_key(row: Mapping[str, Any]) -> str:
    raw = _first(row, "poll_id", "id", "poll")
    if raw is not None and str(raw).strip():
        return str(raw).strip()
    company = str(_first(row, "company", "pollster", "institute") or "").strip()
    publication = str(_first(row, "publication_date", "publication date", "date") or "").strip()
    start = str(_first(row, "fieldwork_start", "interview_start", "fieldwork start") or "").strip()
    end = str(_first(row, "fieldwork_end", "interview_end", "fieldwork end") or "").strip()
    identity = "|".join((company, publication, start, end))
    if not identity.replace("|", ""):
        return ""
    return "swp-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _new_poll(row: Mapping[str, Any], poll_id: str) -> dict[str, Any]:
    company = _first(row, "company", "pollster", "institute")
    if company is None or not str(company).strip():
        raise ValueError(f"Poll {poll_id!r} has no company/pollster")
    publication_date = _parse_optional_date(
        _first(row, "publication_date", "publication date", "published", "date")
    )
    fieldwork_start = _parse_optional_date(
        _first(row, "fieldwork_start", "interview_start", "fieldwork start", "start")
    )
    fieldwork_end = _parse_optional_date(
        _first(row, "fieldwork_end", "interview_end", "fieldwork end", "end")
    )
    return {
        "poll_id": poll_id,
        "company": str(company).strip(),
        "house": (
            str(_first(row, "house", "house_original") or "").strip() or None
        ),
        "publication_date": publication_date,
        "fieldwork_start": fieldwork_start,
        "fieldwork_end": fieldwork_end,
        "n": _parse_sample_size(_first(row, "n", "sample_size", "sample size")),
        "parties": {},
    }


def _is_uncertain(row: Mapping[str, Any], party: str | None) -> bool:
    if party is not None and party.strip().casefold() == "uncertain":
        return True
    status = str(_first(row, "support_status", "status") or "").casefold()
    return "uncertain" in status


def _add_long_poll_row(polls: dict[str, dict[str, Any]], row: Mapping[str, Any]) -> None:
    raw_party = _first(row, "party", "party_name")
    if raw_party is None or _is_uncertain(row, str(raw_party)):
        # SwedishPolls' Uncertain column is explicitly not party support.
        return
    party = normalize_party(str(raw_party))
    if party not in HISTORY_PARTY_ORDER:
        # This also excludes FI, other, and any future source category.  They
        # are not silently turned into an invented Other denominator.
        return
    value = _parse_support(_first(row, "support"))
    if value is None:
        # Never fall back to source_value: ambiguous/uncertain source values
        # must remain missing rather than becoming visible support.
        return
    poll_id = _poll_key(row)
    if not poll_id:
        return
    poll = polls.setdefault(poll_id, _new_poll(row, poll_id))
    previous = poll["parties"].get(party)
    if previous is None:
        poll["parties"][party] = value
    elif not np.isclose(float(previous), value, rtol=0.0, atol=1e-12):
        # A duplicate party row is malformed for one poll.  Dropping the
        # conflicting value makes the poll incomplete and therefore prevents
        # an unsafe coalition point from being published.
        poll["parties"].pop(party, None)


def _add_wide_poll_row(polls: dict[str, dict[str, Any]], row: Mapping[str, Any]) -> None:
    poll_id = _poll_key(row)
    if not poll_id:
        return
    poll = polls.setdefault(poll_id, _new_poll(row, poll_id))
    for party in HISTORY_PARTY_ORDER:
        raw = _first(row, party)
        value = _parse_support(raw)
        if value is not None:
            poll["parties"][party] = value


def filter_swedishpolls_as_of(
    polls: Iterable[Mapping[str, Any]],
    as_of: str | date,
) -> list[dict[str, Any]]:
    """Return only observations admissible to a historical publication date.

    This helper mirrors the canonical state's strict historical rule: a poll
    must have been published on or before ``as_of`` and its interview must
    have ended by then when an end date is available.  The chart artifact
    itself stores all visible observations once; callers generating a
    historical state can use this helper when they need a per-date view.
    """

    target = _coerce_date(as_of, name="as_of")
    filtered: list[dict[str, Any]] = []
    for poll in polls:
        publication = poll.get("publication_date")
        if publication is None:
            continue
        publication_day = _coerce_date(publication, name="poll publication_date")
        if publication_day > target:
            continue
        interview_end = poll.get("fieldwork_end")
        if interview_end is not None and _coerce_date(interview_end, name="poll fieldwork_end") > target:
            continue
        filtered.append(dict(poll))
    return filtered


def filter_swedishpolls_period(
    polls: Iterable[Mapping[str, Any]],
    start_date: str | date,
    end_date: str | date,
) -> list[dict[str, Any]]:
    """Keep dated poll observations inside the chart's visible time range."""

    start = _coerce_date(start_date, name="start_date")
    end = _coerce_date(end_date, name="end_date")
    if start > end:
        raise ValueError("start_date must not be after end_date")
    filtered: list[dict[str, Any]] = []
    for poll in polls:
        publication = poll.get("publication_date")
        if publication is None:
            # An undated source observation cannot be placed on the SVG axis.
            continue
        publication_day = _coerce_date(publication, name="poll publication_date")
        if start <= publication_day <= end:
            filtered.append(dict(poll))
    return filtered


def serialize_swedishpolls(
    source: Path | str | Iterable[Mapping[str, Any]],
    *,
    as_of: str | date | None = None,
) -> list[dict[str, Any]]:
    """Serialize complete SwedishPolls observations once per poll.

    The processed repository file is long-format.  A small wide-format reader
    is also accepted for deterministic fixture use.  Only the eight
    parliamentary party values are retained; ``Uncertain``, FI, and any
    ``Other``-like value are excluded from the output.
    """

    if isinstance(source, (str, Path)):
        path = Path(source)
        with path.open(newline="", encoding="utf-8") as handle:
            rows: list[Mapping[str, Any]] = list(csv.DictReader(handle))
    else:
        rows = list(source)
    polls: dict[str, dict[str, Any]] = {}
    for row in rows:
        has_party_column = _first(row, "party", "party_name") is not None
        if has_party_column:
            _add_long_poll_row(polls, row)
        else:
            _add_wide_poll_row(polls, row)

    complete: list[dict[str, Any]] = []
    for poll in polls.values():
        if set(poll["parties"]) != set(HISTORY_PARTY_ORDER):
            continue
        # Dict insertion order is part of the compact publication contract.
        poll["parties"] = {party: float(poll["parties"][party]) for party in HISTORY_PARTY_ORDER}
        complete.append(poll)
    complete.sort(
        key=lambda item: (
            item["publication_date"] or "",
            item["fieldwork_end"] or "",
            item["poll_id"],
        )
    )
    return filter_swedishpolls_as_of(complete, as_of) if as_of is not None else complete


def serialize_poll_of_polls_timeseries(
    path: Path | str,
    *,
    start_date: str | date,
    end_date: str | date,
) -> list[dict[str, Any]]:
    """Serialize daily eight-party Poll of Polls observations for the chart range."""

    start = _coerce_date(start_date, name="start_date")
    end = _coerce_date(end_date, name="end_date")
    if start > end:
        raise ValueError("start_date must not be after end_date")
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Poll of Polls timeseries CSV not found: {source_path}")
    records: list[dict[str, Any]] = []
    with source_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_date = row.get("date")
            if not raw_date:
                continue
            d = _coerce_date(raw_date, name="timeseries date")
            if start <= d <= end:
                parties: dict[str, float] = {}
                for party in HISTORY_PARTY_ORDER:
                    val = _parse_support(row.get(party))
                    if val is None:
                        raise ValueError(f"Missing or invalid {party} in PoP timeseries on {raw_date}")
                    parties[party] = val
                records.append({
                    "date": d.isoformat(),
                    "parties": parties,
                })
    records.sort(key=lambda item: item["date"])
    return records



def build_history_dates(
    *,
    start_date: str | date = HISTORY_START_DATE,
    latest_date: str | date,
    election_date: str | date = DEFAULT_ELECTION_DATE,
    dynamics_cap_date: str | date = HISTORY_CAP_DATE,
) -> list[date]:
    """Return weekly pre-cap and daily post-cap observation dates.

    The date immediately before the cap (2026-05-23 for the 2026 election) is
    always retained even when the weekly anchor does not land on it.  Dates
    are never generated after the requested latest source date or election.
    """

    start = _coerce_date(start_date, name="start_date")
    latest = _coerce_date(latest_date, name="latest_date")
    election = _coerce_date(election_date, name="election_date")
    cap = _coerce_date(dynamics_cap_date, name="dynamics_cap_date")
    if start > latest:
        return []
    if latest > election:
        latest = election
    if cap <= start:
        cap = start
    dates: list[date] = []
    weekly_end = min(latest, cap - timedelta(days=1), election)
    current = start
    while current <= weekly_end:
        dates.append(current)
        current += timedelta(days=7)
    if weekly_end >= start and dates and dates[-1] != weekly_end:
        dates.append(weekly_end)
    daily_start = max(start, cap)
    current = daily_start
    while current <= latest:
        dates.append(current)
        current += timedelta(days=1)
    return sorted(set(dates))


def _latest_timeseries_date(path: Path) -> date:
    if not path.is_file():
        raise FileNotFoundError(f"Poll of Polls timeseries CSV not found: {path}")
    latest: date | None = None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw = row.get("date")
            if raw:
                parsed = _coerce_date(raw, name="timeseries date")
                latest = parsed if latest is None else max(latest, parsed)
    if latest is None:
        raise ValueError(f"Poll of Polls timeseries has no dates: {path}")
    return latest


def _extract_matrix(result: Any, name: str) -> np.ndarray:
    matrix = getattr(result, name, None)
    if matrix is None:
        raise ValueError(f"Simulation result has no {name}")
    arr = np.asarray(matrix)
    if arr.ndim != 2 or arr.shape[0] <= 0:
        raise ValueError(f"Simulation result {name} must be a non-empty matrix")
    if not np.isfinite(arr).all():
        raise ValueError(f"Simulation result {name} contains non-finite values")
    return arr


def _point_from_result(
    result: Any,
    *,
    point_date: date,
    election_date: date,
    coalitions: Mapping[str, Sequence[str]],
    provenance: str = "reconstructed_current_model",
    publication_generation: str | None = None,
    deterministic_payload_sha256: str | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    votes = _extract_matrix(result, "vote_shares_matrix")
    seats = _extract_matrix(result, "seats_matrix")
    if votes.shape[0] != seats.shape[0]:
        raise ValueError("Simulation vote and seat matrices have different draw counts")
    groups = build_groups_from_matrices(
        votes,
        seats,
        coalitions=coalitions,
        party_order=HISTORY_PARTY_ORDER,
    )
    actual_horizon = max(0, (election_date - point_date).days)
    dynamics_horizon = min(actual_horizon, HISTORY_DYNAMICS_CAP_DAYS)
    manifest = getattr(result, "manifest", None)
    candidates: list[Any] = []
    if isinstance(manifest, Mapping):
        candidates.extend(
            [
                manifest.get("dynamics_horizon_days"),
                manifest.get("dynamics_eval_horizon"),
            ]
        )
        model_config = manifest.get("model_config")
        if isinstance(model_config, Mapping):
            candidates.extend(
                [
                    model_config.get("dynamics_horizon_days"),
                    model_config.get("dynamics_eval_horizon"),
                ]
            )
    diagnostics = getattr(result, "diagnostics", None)
    if isinstance(diagnostics, Mapping):
        candidates.extend(
            [diagnostics.get("dynamics_horizon_days"), diagnostics.get("dynamics_eval_horizon")]
        )
    candidates.extend(
        [
            getattr(result, "dynamics_horizon_days", None),
            getattr(result, "dynamics_eval_horizon", None),
        ]
    )
    # The canonical engine currently exposes the evaluation horizon only via
    # this metadata when a caller retains it.  Its frozen fallback is exactly
    # min(actual_horizon, 112), so that remains the truthful default here.
    for configured in candidates:
        if isinstance(configured, int) and not isinstance(configured, bool):
            dynamics_horizon = max(0, min(actual_horizon, HISTORY_DYNAMICS_CAP_DAYS, configured))
            break
    point: dict[str, Any] = {
        "date": point_date.isoformat(),
        "samples": int(votes.shape[0]),
        "horizon_days": actual_horizon,
        "dynamics_horizon_days": dynamics_horizon,
        "provenance": provenance,
        "groups": groups,
        # Additive party marginals from the identical joint draws.  Nothing is
        # simulated twice and nothing is reconstructed from the coalition
        # quantiles above; at the certified point these *are* the published
        # party forecast.
        "parties": build_parties_from_matrices(
            votes,
            seats,
            party_order=HISTORY_PARTY_ORDER,
        ),
    }
    if provenance != "reconstructed_current_model":
        manifest = manifest if isinstance(manifest, Mapping) else {}
        source_commit = manifest.get("source_git_commit", manifest.get("git_commit"))
        if isinstance(source_commit, str) and source_commit:
            point["source_git_commit"] = source_commit
        if publication_generation is not None:
            point["publication_generation"] = str(publication_generation)
        if deterministic_payload_sha256 is not None:
            point["deterministic_payload_sha256"] = str(deterministic_payload_sha256)
        if generated_at_utc is not None:
            point["generated_at_utc"] = str(generated_at_utc)
    return point


def _archive_point_from_record(
    record: Mapping[str, Any],
    *,
    election_date: date,
    coalitions: Mapping[str, Sequence[str]],
) -> dict[str, Any] | None:
    """Extract a rich archived point, rejecting marginal-only legacy data."""

    point = record.get("forecast_point")
    source: Mapping[str, Any] = point if isinstance(point, Mapping) else record
    raw_date = source.get("date", source.get("as_of", source.get("snapshot_date")))
    if raw_date is None:
        return None
    try:
        point_date = _coerce_date(raw_date, name="archived point date")
    except ValueError:
        return None
    if point_date > election_date:
        return None
    groups = source.get("groups", source.get("coalition_groups"))
    if not isinstance(groups, Mapping):
        # Existing prospective archive snapshots intentionally contain only
        # marginal party histograms and bloc means.  Those cannot recover
        # joint coalition quantiles and must not be represented as forecasts.
        return None
    normalized_groups: dict[str, Any] = {}
    for key in coalitions:
        group = groups.get(key)
        if not isinstance(group, Mapping):
            return None
        vote = group.get("vote", group.get("vote_quantiles"))
        seats = group.get("seats", group.get("seat_quantiles"))
        if not isinstance(vote, Mapping) or not isinstance(seats, Mapping):
            return None
        normalized_groups[key] = {"vote": dict(vote), "seats": dict(seats)}
    samples = source.get("samples", source.get("draws"))
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        return None
    actual_horizon = max(0, (election_date - point_date).days)
    dynamics_horizon = source.get("dynamics_horizon_days", source.get("dynamics_eval_horizon"))
    if not isinstance(dynamics_horizon, int) or isinstance(dynamics_horizon, bool):
        dynamics_horizon = min(actual_horizon, HISTORY_DYNAMICS_CAP_DAYS)
    archived_parties = party_point_from_archive_record(record)
    return {
        "date": point_date.isoformat(),
        "samples": samples,
        "horizon_days": int(source.get("horizon_days", actual_horizon)),
        "dynamics_horizon_days": dynamics_horizon,
        "provenance": "prospective_archived",
        "groups": normalized_groups,
        # Party marginals, unlike joint coalition intervals, *are* recoverable
        # from an archived snapshot.  A snapshot that predates them yields
        # None and the point simply carries no party block.
        **({"parties": archived_parties} if archived_parties else {}),
        "generated_at_utc": source.get("generated_at_utc", record.get("generated_at_utc")),
        "source_git_commit": source.get("source_git_commit", record.get("source_git_commit")),
        "publication_generation": source.get(
            "publication_generation",
            source.get("generation_id", record.get("generation_id")),
        ),
        "deterministic_payload_sha256": source.get(
            "deterministic_payload_sha256", record.get("deterministic_payload_sha256")
        ),
    }


def _load_archive_records(archive_dir: Path | str | None) -> list[Mapping[str, Any]]:
    if archive_dir is None:
        return []
    root = Path(archive_dir)
    if root.is_file():
        with root.open(encoding="utf-8") as handle:
            value = json.load(handle)
        return [value] if isinstance(value, Mapping) else []
    index_path = root / "index.json"
    if not index_path.is_file():
        return []
    with index_path.open(encoding="utf-8") as handle:
        index = json.load(handle)
    records: list[Mapping[str, Any]] = []
    for entry in index.get("snapshots", []) if isinstance(index, Mapping) else []:
        if not isinstance(entry, Mapping):
            continue
        relative = entry.get("path")
        if not isinstance(relative, str):
            continue
        path = root / relative
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, Mapping):
            records.append(value)
    return records


def _model_commit_from(result: Any, fallback: str) -> str:
    manifest = getattr(result, "manifest", None)
    if isinstance(manifest, Mapping):
        value = manifest.get("source_git_commit", manifest.get("git_commit"))
        if isinstance(value, str) and value.strip() and value != "unknown_git_commit":
            return value
    return fallback


def _result_as_of(result: Any) -> date | None:
    """Read the as-of date carried by a latest/official result, if present."""

    summary = getattr(result, "summary", None)
    candidates: list[Any] = []
    if summary is not None:
        candidates.append(getattr(summary, "as_of", None))
        if isinstance(summary, Mapping):
            candidates.append(summary.get("as_of"))
    manifest = getattr(result, "manifest", None)
    if isinstance(manifest, Mapping):
        candidates.append(manifest.get("as_of"))
    for value in candidates:
        if value is None:
            continue
        try:
            return _coerce_date(value, name="latest_result as_of")
        except ValueError:
            continue
    return None


def _worker_simulate_date(
    args: tuple[str, str, int, int],
) -> tuple[str, np.ndarray, np.ndarray, Any]:
    as_of, election_date, samples, seed = args
    result = simulate_election(
        as_of=as_of,
        election_date=election_date,
        samples=samples,
        seed=seed,
    )
    return (
        as_of,
        np.asarray(result.vote_shares_matrix),
        np.asarray(result.seats_matrix),
        getattr(result, "manifest", None),
    )


def build_history(
    *,
    election_date: str | date = DEFAULT_ELECTION_DATE,
    start_date: str | date = HISTORY_START_DATE,
    latest_date: str | date | None = None,
    dates: Sequence[str | date] | None = None,
    samples: int = DEFAULT_HISTORY_SAMPLES,
    seed: int = DEFAULT_SIMULATION_SEED,
    poll_file: Path | str = DEFAULT_POLL_FILE,
    timeseries_file: Path | str = DEFAULT_TIMESERIES_FILE,
    archive_dir: Path | str | None = DEFAULT_ARCHIVE_DIR,
    archived_points: Sequence[Mapping[str, Any]] | None = None,
    existing_payload: Mapping[str, Any] | None = None,
    latest_result: Any | None = None,
    production_latest_samples: int = 100_000,
    simulation_runner: Callable[..., Any] | None = None,
    coalitions: Mapping[str, Sequence[str]] = DEFAULT_COALITIONS,
    model_commit: str | None = None,
    generated_at_utc: str | None = None,
    source_worktree_clean: bool | None = None,
    production_metadata: Mapping[str, Any] | None = None,
    workers: int = 1,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Build and validate one schema 1.0 history artifact.

    ``simulation_runner`` is injectable for tests and for a publication host
    that has already run a certified simulation.  The default is the existing
    canonical ``simulate_election``; no model variant or seat allocator is
    introduced here.
    """

    if samples <= 0 or production_latest_samples <= 0:
        raise ValueError("samples must be positive")
    election = _coerce_date(election_date, name="election_date")
    poll_path = Path(poll_file)
    timeseries_path = Path(timeseries_file)
    if dates is None:
        latest = (
            _coerce_date(latest_date, name="latest_date")
            if latest_date is not None
            else _latest_timeseries_date(timeseries_path)
        )
        observation_dates = build_history_dates(
            start_date=start_date,
            latest_date=latest,
            election_date=election,
        )
    else:
        observation_dates = sorted({_coerce_date(item, name="date") for item in dates})
    latest_result_date = _result_as_of(latest_result) if latest_result is not None else None
    if latest_result is not None and latest_result_date is None:
        raise ValueError("latest_result must carry an as_of date in summary or manifest")
    if latest_result_date is not None and latest_result_date not in observation_dates:
        observation_dates.append(latest_result_date)
        observation_dates.sort()
    if not observation_dates:
        raise ValueError("No historical observation dates were selected")
    if any(point_date > election for point_date in observation_dates):
        raise ValueError("Historical observation dates cannot occur after election_date")

    coalition_config = {
        str(key): [str(party) for party in members]
        for key, members in coalitions.items()
    }
    # Contract-level validation of the configured membership happens when the
    # first point is validated; reject an empty configuration early.
    if not coalition_config:
        raise ValueError("At least one coalition must be configured")

    # An existing valid artifact is a resumable cache.  Its points are kept
    # byte-for-byte and only missing dates are simulated.  The source and
    # contract identity must match; silently combining different model/input
    # revisions would make the chart's provenance misleading.
    existing_points: dict[date, dict[str, Any]] = {}
    existing_model_commit: str | None = None
    if existing_payload is not None:
        validate_history_contract(existing_payload)
        if existing_payload["election_date"] != election.isoformat():
            raise ValueError("existing_payload uses a different election_date")
        if existing_payload["party_order"] != list(HISTORY_PARTY_ORDER):
            raise ValueError("existing_payload uses a different party_order")
        if existing_payload["coalitions"] != coalition_config:
            raise ValueError("existing_payload uses a different coalition configuration")
        existing_model_commit = str(existing_payload["model_commit"])
        for point in existing_payload["series"]:
            existing_points[date.fromisoformat(point["date"])] = dict(point)
        for point_date in existing_points:
            if point_date not in observation_dates:
                observation_dates.append(point_date)
        observation_dates.sort()

    all_polls = serialize_swedishpolls(poll_path)
    poll_hash = compute_file_sha256(poll_path)
    if not isinstance(poll_hash, str) or len(poll_hash) != 64:
        raise FileNotFoundError(f"SwedishPolls source file not found: {poll_path}")
    if existing_payload is not None and existing_payload["poll_source_sha256"] != poll_hash:
        raise ValueError("existing_payload was generated from a different SwedishPolls source")
    if existing_payload is not None and model_commit is not None and existing_payload["model_commit"] != model_commit:
        raise ValueError("existing_payload uses a different model_commit")
    records = list(archived_points or []) + _load_archive_records(archive_dir)
    archived_by_date: dict[date, dict[str, Any]] = {}
    skipped_archives = 0
    for record in records:
        normalized = _archive_point_from_record(
            record,
            election_date=election,
            coalitions=coalition_config,
        )
        if normalized is None:
            skipped_archives += 1
            continue
        point_date = date.fromisoformat(normalized["date"])
        previous = archived_by_date.get(point_date)
        previous_key = str(previous.get("generated_at_utc") or "") if previous else ""
        current_key = str(normalized.get("generated_at_utc") or "")
        if previous is None or current_key >= previous_key:
            archived_by_date[point_date] = normalized

    for point_date in archived_by_date:
        if point_date not in observation_dates:
            observation_dates.append(point_date)
    observation_dates.sort()
    if latest_result_date is not None and latest_result_date != max(observation_dates):
        raise ValueError("latest_result must correspond to the latest requested observation date")

    chart_start = min(observation_dates)
    poll_publication_dates = [
        date.fromisoformat(str(poll["publication_date"]))
        for poll in all_polls
        if poll.get("publication_date") is not None
    ]
    chart_end = min(election, max([max(observation_dates), *poll_publication_dates]))
    polls = filter_swedishpolls_period(all_polls, chart_start, chart_end)

    # Determine which observation dates require new simulations
    candidate_latest_dates = [*observation_dates, *archived_by_date]
    if existing_points:
        candidate_latest_dates.extend(existing_points)
    official_latest_date = latest_result_date or max(candidate_latest_dates)

    dates_to_simulate: list[tuple[date, int]] = []
    for point_date in observation_dates:
        if (
            point_date in existing_points
            and (
                point_date != official_latest_date
                or (
                    latest_result is None
                    and int(existing_points[point_date]["samples"]) == production_latest_samples
                )
            )
        ):
            continue
        if point_date == official_latest_date and latest_result is not None:
            continue
        archived = archived_by_date.get(point_date)
        if archived is not None and (
            point_date != official_latest_date or archived["samples"] == production_latest_samples
        ):
            continue
        requested_samples = production_latest_samples if point_date == official_latest_date else samples
        dates_to_simulate.append((point_date, requested_samples))

    computed_results: dict[date, tuple[np.ndarray, np.ndarray, Any]] = {}
    workers_count = max(1, int(workers)) if isinstance(workers, int) else 1
    if workers_count > 1 and simulation_runner is None and len(dates_to_simulate) > 1:
        import concurrent.futures
        tasks = [
            (point_date.isoformat(), election.isoformat(), req_samples, seed)
            for point_date, req_samples in dates_to_simulate
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers_count) as executor:
            future_to_date = {
                executor.submit(_worker_simulate_date, t): t[0]
                for t in tasks
            }
            completed = 0
            for future in concurrent.futures.as_completed(future_to_date):
                date_iso, votes, seats, manifest = future.result()
                p_date = date.fromisoformat(date_iso)
                computed_results[p_date] = (votes, seats, manifest)
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, len(dates_to_simulate), date_iso)

    runner = simulation_runner or simulate_election
    series: list[dict[str, Any]] = []
    model_commit_value = model_commit or existing_model_commit or get_git_commit_hash(REPOSITORY_ROOT)

    for point_date in observation_dates:
        if (
            point_date in existing_points
            and (
                point_date != official_latest_date
                or (
                    latest_result is None
                    and int(existing_points[point_date]["samples"]) == production_latest_samples
                )
            )
        ):
            series.append(existing_points[point_date])
            continue
        if point_date == official_latest_date and latest_result is not None:
            point = _point_from_result(
                latest_result,
                point_date=point_date,
                election_date=election,
                coalitions=coalition_config,
                provenance="current_production",
                publication_generation=(
                    production_metadata.get("publication_generation")
                    if production_metadata
                    else None
                ),
                deterministic_payload_sha256=(
                    production_metadata.get("deterministic_payload_sha256")
                    if production_metadata
                    else None
                ),
                generated_at_utc=(
                    production_metadata.get("generated_at_utc")
                    if production_metadata
                    else None
                ),
            )
            if point["samples"] != production_latest_samples:
                raise ValueError(
                    "latest_result draw count does not match production_latest_samples; "
                    "the current chart point must remain an official/exact joint artifact"
                )
            series.append(point)
            if not model_commit:
                model_commit_value = _model_commit_from(latest_result, model_commit_value)
            continue
        archived = archived_by_date.get(point_date)
        if archived is not None and (
            point_date != official_latest_date or archived["samples"] == production_latest_samples
        ):
            series.append(
                {
                    key: value
                    for key, value in archived.items()
                    if key
                    in {
                        "date",
                        "samples",
                        "horizon_days",
                        "dynamics_horizon_days",
                        "provenance",
                        "groups",
                        "source_git_commit",
                        "publication_generation",
                        "deterministic_payload_sha256",
                        "generated_at_utc",
                    }
                    and value is not None
                }
            )
            if not model_commit:
                model_commit_value = str(archived.get("source_git_commit") or model_commit_value)
            continue
        if point_date in computed_results:
            votes, seats, manifest = computed_results[point_date]
            point_provenance = (
                "current_production"
                if point_date == official_latest_date
                else "reconstructed_current_model"
            )
            point = _point_from_result(
                SimpleNamespace(vote_shares_matrix=votes, seats_matrix=seats, manifest=manifest),
                point_date=point_date,
                election_date=election,
                coalitions=coalition_config,
                provenance=point_provenance,
            )
            series.append(point)
            if not model_commit and isinstance(manifest, Mapping):
                val = manifest.get("source_git_commit", manifest.get("git_commit"))
                if isinstance(val, str) and val.strip() and val != "unknown_git_commit":
                    model_commit_value = val
            continue
        requested_samples = production_latest_samples if point_date == official_latest_date else samples
        result = runner(
            as_of=point_date.isoformat(),
            election_date=election.isoformat(),
            samples=requested_samples,
            seed=seed,
        )
        point_provenance = (
            "current_production"
            if point_date == official_latest_date
            else "reconstructed_current_model"
        )
        point = _point_from_result(
            result,
            point_date=point_date,
            election_date=election,
            coalitions=coalition_config,
            provenance=point_provenance,
        )
        if point["samples"] != requested_samples:
            raise ValueError(
                "simulation runner returned a draw count different from the requested count"
            )
        if point_date == official_latest_date and point["samples"] != production_latest_samples:
            raise ValueError(
                "latest simulation draw count does not match production_latest_samples"
            )
        series.append(point)
        if not model_commit:
            model_commit_value = _model_commit_from(result, model_commit_value)

    # Rich archived points outside the regular schedule are retained as
    # meaningful publication observations (for example, a prospective run
    # archived on a Thursday between weekly reconstructed points).
    for point_date, archived in archived_by_date.items():
        if point_date not in {date.fromisoformat(point["date"]) for point in series}:
            series.append(
                {
                    key: value
                    for key, value in archived.items()
                    if key
                    in {
                        "date",
                        "samples",
                        "horizon_days",
                        "dynamics_horizon_days",
                        "provenance",
                        "groups",
                        "source_git_commit",
                        "publication_generation",
                        "deterministic_payload_sha256",
                        "generated_at_utc",
                    }
                    and value is not None
                }
            )
    # Preserve points already present in a resumable artifact, including dates
    # from an earlier chunk that are not part of this invocation's requested
    # date window.
    for point_date, existing in existing_points.items():
        if point_date not in {date.fromisoformat(point["date"]) for point in series}:
            series.append(existing)
    series.sort(key=lambda point: (point["date"], point["provenance"]))

    poll_of_polls = serialize_poll_of_polls_timeseries(
        timeseries_path,
        start_date=chart_start,
        end_date=chart_end,
    )

    source_hashes = {
        "poll_source_sha256": poll_hash,
        "timeseries_source_sha256": compute_file_sha256(timeseries_path),
    }
    payload: dict[str, Any] = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "election_date": election.isoformat(),
        "model_commit": model_commit_value,
        "poll_source_sha256": poll_hash,
        "party_order": list(HISTORY_PARTY_ORDER),
        "coalitions": coalition_config,
        "series": series,
        "poll_of_polls": poll_of_polls,
        "polls": polls,
        "source_hashes": source_hashes,
        "model": {
            "name": "ElectionSimulator",
            "dynamics_model": "symmetric_all_history",
            "dynamics_horizon_cap_days": HISTORY_DYNAMICS_CAP_DAYS,
            "history_samples": samples,
            "seed": seed,
        },
        "schedule": {
            "start_date": _coerce_date(start_date, name="start_date").isoformat(),
            "weekly_until": (HISTORY_CAP_DATE - timedelta(days=1)).isoformat(),
            "daily_from": HISTORY_CAP_DATE.isoformat(),
            "observation_count": len(series),
        },
        "poll_date_range": {
            "start_date": chart_start.isoformat(),
            "end_date": chart_end.isoformat(),
        },
        "provenance_note": (
            "Historiska prognoser är rekonstruerade i efterhand med dagens modell och den slutliga historiska Poll of Polls-serien. "
            "Den senaste punkten visar den officiella aktuella valprognosen. "
            "Äkta prospektiva arkivpunkter ersätts inte där arkivet saknar de gemensamma koalitionsdragningar som krävs för att beräkna koalitionsintervall korrekt. "
            "Koalitionernas röstandelar beräknas över de åtta riksdagspartierna; "
            "enskilda partiers röstandelar redovisas över hela valmanskåren."
        ),
        "archive_diagnostics": {
            "rich_archived_points_used": len(archived_by_date),
            "legacy_or_incomplete_archives_skipped": skipped_archives,
        },
    }
    # Declared only when the series actually carries party summaries. Resuming
    # from an artifact generated before this contract regenerates nothing, so
    # such a run legitimately produces no party family at all.
    if series_carries_parties(series):
        payload["parties_view"] = parties_view_metadata()
    if existing_payload is not None:
        payload["resume_diagnostics"] = {
            "existing_points_reused": sum(
                1 for point in series if point["date"] in {item["date"] for item in existing_payload["series"]}
            ),
            "new_points_generated": sum(
                1 for point in series if point["date"] not in {item["date"] for item in existing_payload["series"]}
            ),
        }
    if generated_at_utc is not None:
        parsed = datetime.fromisoformat(generated_at_utc.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("generated_at_utc must include a timezone")
        payload["generated_at_utc"] = generated_at_utc
    payload["source_worktree_clean"] = (
        bool(source_worktree_clean)
        if source_worktree_clean is not None
        else is_git_worktree_clean(REPOSITORY_ROOT)
    )
    payload["deterministic_content_sha256"] = deterministic_history_sha256(payload)
    validate_history_contract(payload)
    return payload


def update_history_with_production_result(
    existing_payload: Mapping[str, Any],
    production_result: Any,
    *,
    poll_file: Path | str = DEFAULT_POLL_FILE,
    timeseries_file: Path | str = DEFAULT_TIMESERIES_FILE,
    archive_dir: Path | str | None = DEFAULT_ARCHIVE_DIR,
    election_date: str | date = DEFAULT_ELECTION_DATE,
    coalitions: Mapping[str, Sequence[str]] = DEFAULT_COALITIONS,
    publication_generation: str | None = None,
    deterministic_payload_sha256: str | None = None,
    generated_at_utc: str | None = None,
    model_commit: str | None = None,
    source_worktree_clean: bool | None = None,
) -> dict[str, Any]:
    """Roll one certified production result into the history artifact.

    This is the production-history boundary.  It deliberately accepts an
    already-computed ``SimulationResult`` and never invokes the simulator.
    Existing reconstructed points are copied as-is; the prior official point
    is relabelled ``prospective_archived`` on a later day, while a same-day
    rerun replaces the one point for that date.  Coalition summaries for the
    new point are derived from the result's original joint matrices.
    """

    validate_history_contract(existing_payload)
    election = _coerce_date(election_date, name="election_date")
    if existing_payload["election_date"] != election.isoformat():
        raise ValueError("existing_payload uses a different election_date")
    coalition_config = {
        str(key): [str(party) for party in members]
        for key, members in coalitions.items()
    }
    if existing_payload["coalitions"] != coalition_config:
        raise ValueError("existing_payload uses a different coalition configuration")

    current_date = _result_as_of(production_result)
    if current_date is None:
        raise ValueError("production_result must carry an as_of date")
    if current_date > election:
        raise ValueError("production_result as_of cannot occur after election_date")
    manifest = getattr(production_result, "manifest", None)
    manifest_map = manifest if isinstance(manifest, Mapping) else {}
    source_commit = model_commit or _model_commit_from(
        production_result, str(existing_payload["model_commit"])
    )
    if not isinstance(source_commit, str) or not re.fullmatch(
        r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", source_commit
    ):
        raise ValueError("production_result does not carry a resolvable source Git commit")

    # Locate archive metadata for an earlier official point.  Legacy current
    # points may predate the audit fields, but their corresponding immutable
    # archive entry can still supply the exact generation/provenance linkage.
    archive_metadata: dict[date, Mapping[str, Any]] = {}
    for record in _load_archive_records(archive_dir):
        raw_date = record.get("as_of", record.get("snapshot_date"))
        try:
            record_date = _coerce_date(raw_date, name="archive snapshot date")
        except (TypeError, ValueError):
            continue
        previous = archive_metadata.get(record_date)
        previous_key = str(previous.get("generated_at_utc") or "") if previous else ""
        current_key = str(record.get("generated_at_utc") or "")
        if previous is None or current_key >= previous_key:
            archive_metadata[record_date] = record

    series: list[dict[str, Any]] = []
    for original in existing_payload["series"]:
        point = dict(original)
        point_date = date.fromisoformat(str(point["date"]))
        if point_date == current_date:
            # The new official result is authoritative for this calendar day,
            # regardless of whether the prior point was reconstructed/current.
            continue
        if point.get("provenance") == "current_production":
            point["provenance"] = "prospective_archived"
            archive_record = archive_metadata.get(point_date)
            if archive_record is not None:
                for field in (
                    "source_git_commit",
                    "publication_generation",
                    "generation_id",
                    "deterministic_payload_sha256",
                    "generated_at_utc",
                ):
                    if field not in point and archive_record.get(field) is not None:
                        target = "publication_generation" if field == "generation_id" else field
                        point[target] = archive_record[field]
        series.append(point)

    new_point = _point_from_result(
        production_result,
        point_date=current_date,
        election_date=election,
        coalitions=coalition_config,
        provenance="current_production",
        publication_generation=publication_generation,
        deterministic_payload_sha256=deterministic_payload_sha256,
        generated_at_utc=generated_at_utc,
    )
    # Keep the object construction explicit above, then validate the audit
    # values against the result manifest instead of allowing a caller to put a
    # false commit on the chart point.
    new_point["source_git_commit"] = str(manifest_map.get("source_git_commit", source_commit))
    if not re.fullmatch(
        r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", new_point["source_git_commit"]
    ):
        raise ValueError("production_result source_git_commit is not a Git commit hash")
    series.append(new_point)
    series.sort(key=lambda point: (point["date"], str(point["provenance"])))
    if len({point["date"] for point in series}) != len(series):
        raise ValueError("production history update would create duplicate calendar dates")
    if sum(point.get("provenance") == "current_production" for point in series) != 1:
        raise ValueError("production history must contain exactly one current_production point")

    poll_path = Path(poll_file)
    timeseries_path = Path(timeseries_file)
    all_polls = serialize_swedishpolls(poll_path)
    poll_hash = compute_file_sha256(poll_path)
    timeseries_hash = compute_file_sha256(timeseries_path)
    if len(poll_hash) != 64 or len(timeseries_hash) != 64:
        raise FileNotFoundError("History inputs must be present and hashable")
    series_dates = [date.fromisoformat(str(point["date"])) for point in series]
    chart_start = min(series_dates)
    publication_dates = [
        date.fromisoformat(str(poll["publication_date"]))
        for poll in all_polls
        if poll.get("publication_date") is not None
    ]
    chart_end = min(election, max([*series_dates, *publication_dates]))
    polls = filter_swedishpolls_period(all_polls, chart_start, chart_end)
    poll_of_polls = serialize_poll_of_polls_timeseries(
        timeseries_path, start_date=chart_start, end_date=chart_end
    )

    payload: dict[str, Any] = {
        "schema_version": existing_payload["schema_version"],
        "election_date": election.isoformat(),
        "model_commit": source_commit,
        "poll_source_sha256": poll_hash,
        "party_order": list(HISTORY_PARTY_ORDER),
        "coalitions": coalition_config,
        "series": series,
        "poll_of_polls": poll_of_polls,
        "polls": polls,
        "source_hashes": {
            "poll_source_sha256": poll_hash,
            "timeseries_source_sha256": timeseries_hash,
        },
        "model": dict(existing_payload.get("model") or {}),
        "schedule": dict(existing_payload.get("schedule") or {}),
        "poll_date_range": {
            "start_date": chart_start.isoformat(),
            "end_date": chart_end.isoformat(),
        },
        "provenance_note": existing_payload.get("provenance_note"),
        "archive_diagnostics": dict(existing_payload.get("archive_diagnostics") or {}),
        "source_worktree_clean": (
            bool(source_worktree_clean)
            if source_worktree_clean is not None
            else bool(manifest_map.get("source_worktree_clean"))
        ),
    }
    payload["model"].update(
        {
            "name": "ElectionSimulator",
            "history_samples": payload["model"].get("history_samples", DEFAULT_HISTORY_SAMPLES),
            "seed": manifest_map.get("base_seed", payload["model"].get("seed")),
            "production_samples": int(getattr(getattr(production_result, "summary", None), "total_samples", 0)),
        }
    )
    payload["schedule"].update({"observation_count": len(series)})
    # The party family travels with the artifact. Reconstructed points carry
    # whatever the last full generation gave them; the certified point always
    # carries its own, derived from its own draws.
    if series_carries_parties(series):
        payload["parties_view"] = parties_view_metadata()
    if generated_at_utc is not None:
        payload["generated_at_utc"] = str(generated_at_utc)
    payload["deterministic_content_sha256"] = deterministic_history_sha256(payload)
    validate_history_contract(payload)
    return payload


def generate_history(**kwargs: Any) -> dict[str, Any]:
    """Backward-compatible descriptive alias for :func:`build_history`."""

    return build_history(**kwargs)


def generate_history_artifact(
    output: Path | str = DEFAULT_HISTORY_OUTPUT,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build and write a validated history artifact."""

    payload = build_history(**kwargs)
    write_history_json(output, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the offline coalition forecast history JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_HISTORY_OUTPUT)
    parser.add_argument("--poll-file", type=Path, default=DEFAULT_POLL_FILE)
    parser.add_argument("--timeseries-file", type=Path, default=DEFAULT_TIMESERIES_FILE)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Reuse points from an existing schema 1.0 artifact and generate only missing dates",
    )
    parser.add_argument(
        "--date",
        dest="selected_dates",
        action="append",
        default=None,
        help="Generate one explicit date (repeat for chunked/resumable runs)",
    )
    parser.add_argument("--start-date", default=HISTORY_START_DATE.isoformat())
    parser.add_argument("--latest-date", default=None)
    parser.add_argument("--election-date", default=DEFAULT_ELECTION_DATE)
    parser.add_argument("--samples", type=int, default=DEFAULT_HISTORY_SAMPLES)
    parser.add_argument("--production-latest-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SIMULATION_SEED)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, os.cpu_count() or 1)),
        help="Number of parallel worker processes to use",
    )
    args = parser.parse_args(argv)
    existing_payload = None
    if args.resume is not None:
        with args.resume.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, Mapping):
            parser.error("--resume must point to a JSON object")
        existing_payload = loaded

    import time
    start_time = time.time()

    def cli_progress(done: int, total: int, current_date: str) -> None:
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (total - done) / rate if rate > 0 else 0
        print(
            f"[{done}/{total}] {current_date} done | Elapsed: {elapsed:.0f}s | Est. remaining: {remaining:.0f}s",
            flush=True,
        )

    payload = generate_history_artifact(
        output=args.output,
        poll_file=args.poll_file,
        timeseries_file=args.timeseries_file,
        archive_dir=args.archive_dir,
        start_date=args.start_date,
        latest_date=args.latest_date,
        dates=args.selected_dates,
        existing_payload=existing_payload,
        election_date=args.election_date,
        samples=args.samples,
        production_latest_samples=args.production_latest_samples,
        seed=args.seed,
        workers=args.workers,
        progress_callback=cli_progress,
    )
    print(json.dumps({
        "output": str(args.output),
        "schema_version": payload["schema_version"],
        "series": len(payload["series"]),
        "polls": len(payload["polls"]),
        "deterministic_content_sha256": payload["deterministic_content_sha256"],
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_HISTORY_OUTPUT",
    "DEFAULT_HISTORY_SAMPLES",
    "DEFAULT_POLL_FILE",
    "DEFAULT_TIMESERIES_FILE",
    "HISTORY_CAP_DATE",
    "HISTORY_START_DATE",
    "build_history",
    "build_history_dates",
    "generate_history",
    "generate_history_artifact",
    "filter_swedishpolls_as_of",
    "filter_swedishpolls_period",
    "serialize_poll_of_polls_timeseries",
    "serialize_swedishpolls",
]
