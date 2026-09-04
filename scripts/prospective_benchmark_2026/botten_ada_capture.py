"""Capture and validate the official Botten Ada publication surfaces.

This module is intentionally separate from ``scripts.botten_ada_benchmark``.
The latter is a historical, externally supplied bundle harness.  The code in
this file is for the 2026 prospective campaign and has one important rule:
only quantities that Botten Ada actually publishes are extracted.  In
particular, the ``pop.rds`` download is *not* treated as a predictive draw
bundle merely because the data page says that it contains 1,000 posterior
draws.  A caller must provide independent semantic evidence and demonstrate
parity with the public election forecast before exact draws can be eligible
for scoring.

No network request is made at import time.  ``capture_botten_ada`` performs
requests only when explicitly called, and accepts an injected fetcher so all
parser and failure tests can use committed fixtures.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np


SCHEMA_VERSION = "1.0"
PARSER_VERSION = "2026-09-03.botten-ada-v1"
PARTY_ORDER: tuple[str, ...] = ("M", "L", "C", "KD", "S", "V", "MP", "SD")
# These are frozen in ``data/processed/prospective_benchmark_2026/protocol.json``.
# Vote-share quantiles are published as proportions (0.051 percentage points is
# 0.00051 on that scale), while threshold probabilities use a separate
# probability tolerance (0.51 percentage points is 0.0051).  Keeping the two
# tolerances separate prevents a rounding allowance for a probability event
# from accidentally making a vote-share parity check too permissive.
PARITY_VOTE_TOLERANCE_PERCENTAGE_POINTS = 0.051
PARITY_VOTE_TOLERANCE_PROPORTION = PARITY_VOTE_TOLERANCE_PERCENTAGE_POINTS / 100.0
PARITY_THRESHOLD_TOLERANCE_PROBABILITY = 0.0051
THRESHOLD_EVENTS: dict[str, str] = {
    "L": "is_L_above_4_pct",
    "C": "is_C_above_4_pct",
    "KD": "is_KD_above_4_pct",
    "MP": "is_MP_above_4_pct",
}
DECISION_SOURCE_KEYS: tuple[str, ...] = (
    "forecast",
    "threshold_L",
    "threshold_C",
    "threshold_KD",
    "threshold_MP",
)
DECISION_GENERATION_FIELDS: tuple[str, ...] = ("run", "model", "run_written")
ELECTION_DATE = "2026-09-13"

OFFICIAL_SITE_URL = "https://www.bottenada.se/"
OFFICIAL_DATA_URL = "https://www.bottenada.se/data"
OFFICIAL_FAQ_URL = "https://www.bottenada.se/faq"
API_BASE_URL = "https://ada-site-data.s3.eu-north-1.amazonaws.com"
RDS_URL = "https://ada-model-results.s3.eu-north-1.amazonaws.com/pop.rds"
ADA_REPOSITORY_URL = "https://github.com/MansMeg/ada_code"
ADA_REPOSITORY_COMMIT = "2dfe246b86c5cab517e4a0cb87fd57e5a9c62512"
ADA_REPOSITORY_COMMIT_URL = f"{ADA_REPOSITORY_URL}/commit/{ADA_REPOSITORY_COMMIT}"
ADA_CONFIG_URL = (
    "https://raw.githubusercontent.com/MansMeg/ada_code/"
    f"{ADA_REPOSITORY_COMMIT}/run_ada/ada_config.yml"
)
ADA_LICENSE = "MIT (ada_code); CC BY-NC-SA 4.0 (Botten Ada published data/code page)"
ADA_LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"

# Evidence URLs that may establish the meaning of an extracted RDS object.
# They are intentionally exact, pinned publication surfaces rather than an
# arbitrary URL supplied by a caller.  The bytes from one of these URLs must
# also be passed to ``verify_official_draws`` and hash-checked there.
SEMANTIC_EVIDENCE_URLS = frozenset(
    {
        OFFICIAL_SITE_URL,
        OFFICIAL_DATA_URL,
        OFFICIAL_FAQ_URL,
        ADA_REPOSITORY_COMMIT_URL,
        ADA_CONFIG_URL,
    }
)

PARTY_PAGE_URLS: dict[str, str] = {
    "V": f"{OFFICIAL_SITE_URL}parti/vansterpartiet",
    "S": f"{OFFICIAL_SITE_URL}parti/socialdemokraterna",
    "MP": f"{OFFICIAL_SITE_URL}parti/miljopartiet",
    "C": f"{OFFICIAL_SITE_URL}parti/centerpartiet",
    "L": f"{OFFICIAL_SITE_URL}parti/liberalerna",
    "M": f"{OFFICIAL_SITE_URL}parti/moderaterna",
    "KD": f"{OFFICIAL_SITE_URL}parti/kristdemokraterna",
    "SD": f"{OFFICIAL_SITE_URL}parti/sverigedemokraterna",
}

STATUS_COMPLETE = "COMPLETE"
STATUS_AVAILABLE = "AVAILABLE"
STATUS_SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
STATUS_PARSE_FAILED = "PARSE_FAILED"
STATUS_SOURCE_STALE = "SOURCE_STALE"
STATUS_PARITY_UNVERIFIED = "PARITY_UNVERIFIED"
STATUS_PARITY_VERIFIED = "VERIFIED"


class BottenAdaCaptureError(ValueError):
    """Base error for malformed official Botten Ada evidence."""


class BottenAdaParseError(BottenAdaCaptureError):
    """Raised when an official artifact is present but malformed."""


class BottenAdaDrawsNotVerified(BottenAdaCaptureError):
    """Raised when a caller attempts to score unverified RDS/posterior draws."""


class BottenAdaSourceError(RuntimeError):
    """Raised by the network fetcher for an unavailable or too-large source."""


@dataclass(frozen=True)
class SourceSpec:
    """One official source and its safe archive-relative raw filename."""

    key: str
    url: str
    raw_name: str
    kind: str
    head_only: bool = False


# ``latest_forecast`` is the publication surface for election-day forecasts;
# ``latest_pop`` is the separately served current poll/latent-state surface.
# Keeping both mode paths explicit prevents an accidentally stale or current
# poll value from being substituted for an election-day forecast.
DEFAULT_SOURCE_SPECS: dict[str, SourceSpec] = {
    "forecast": SourceSpec(
        "forecast",
        f"{API_BASE_URL}/latest_forecast/seats--all.json",
        "raw/latest_forecast_seats--all.json",
        "forecast_json",
    ),
    "latest_polls": SourceSpec(
        "latest_polls",
        f"{API_BASE_URL}/latest_forecast/latest_polls--all.json",
        "raw/latest_forecast_latest_polls--all.json",
        "latest_polls_json",
    ),
    "timeseries": SourceSpec(
        "timeseries",
        f"{API_BASE_URL}/latest_pop/timeseries.csv",
        "raw/latest_pop_timeseries.csv",
        "timeseries_csv",
    ),
    "threshold_L": SourceSpec(
        "threshold_L",
        f"{API_BASE_URL}/latest_forecast/question--{THRESHOLD_EVENTS['L']}.json",
        "raw/latest_forecast_question--is_L_above_4_pct.json",
        "threshold_json",
    ),
    "threshold_C": SourceSpec(
        "threshold_C",
        f"{API_BASE_URL}/latest_forecast/question--{THRESHOLD_EVENTS['C']}.json",
        "raw/latest_forecast_question--is_C_above_4_pct.json",
        "threshold_json",
    ),
    "threshold_KD": SourceSpec(
        "threshold_KD",
        f"{API_BASE_URL}/latest_forecast/question--{THRESHOLD_EVENTS['KD']}.json",
        "raw/latest_forecast_question--is_KD_above_4_pct.json",
        "threshold_json",
    ),
    "threshold_MP": SourceSpec(
        "threshold_MP",
        f"{API_BASE_URL}/latest_forecast/question--{THRESHOLD_EVENTS['MP']}.json",
        "raw/latest_forecast_question--is_MP_above_4_pct.json",
        "threshold_json",
    ),
    "homepage": SourceSpec(
        "homepage",
        OFFICIAL_SITE_URL,
        "raw/homepage.html",
        "homepage_html",
    ),
    # The RDS file is approximately 1.7 GB and is described by the publisher
    # as a model object containing posterior draws.  We record a HEAD response
    # by default, but never download or interpret it as predictive draws.
    "rds": SourceSpec("rds", RDS_URL, "raw/pop.rds", "rds", head_only=True),
}


@dataclass(frozen=True)
class SourceArtifact:
    """Fetched source bytes plus immutable transport evidence.

    ``body`` is ``None`` for an unavailable source and for a HEAD-only source.
    ``headers`` are retained as supplied by the HTTP client; provenance code
    reads them case-insensitively.
    """

    url: str
    body: bytes | None
    retrieved_at_utc: str
    status_code: int = 200
    headers: Mapping[str, str] = field(default_factory=dict)
    method: str = "GET"
    error: str | None = None


@dataclass(frozen=True)
class BottenAdaCapture:
    """JSON-safe normalized record and raw official bytes for one cutoff."""

    record: Mapping[str, Any]
    raw_files: Mapping[str, bytes]

    def jsonable(self) -> dict[str, Any]:
        """Return the normalized record (never embeds raw bytes)."""

        # A JSON round-trip catches accidental non-JSON values before a caller
        # passes the object to the append-only archive.  It also returns a
        # detached value, so a caller cannot mutate this capture through a
        # nested reference and accidentally change the evidence to be hashed.
        return json.loads(json.dumps(dict(self.record), ensure_ascii=False, allow_nan=False))

    def write(self, output_dir: Path | str) -> Path:
        """Write this model capture using exclusive, atomic file creation.

        The benchmark archive normally writes ``record`` itself through its
        own staging directory.  This helper is useful for standalone capture
        jobs and is deliberately refuse-on-existing: a failed or old capture
        can never be replaced by a retry.
        """

        destination = Path(output_dir)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing Botten Ada capture: {destination}")
        destination.mkdir(parents=True, exist_ok=False)
        try:
            _write_exclusive_json(destination / "forecast.json", self.record.get("forecast"))
            _write_exclusive_json(destination / "provenance.json", self.record.get("provenance", {}))
            for relative_name, content in sorted(self.raw_files.items()):
                relative = _safe_relative_name(relative_name)
                _write_exclusive_bytes(destination.joinpath(*relative.parts), content)
        except Exception:
            # Keep a failed directory as evidence of an interrupted standalone
            # write; callers must not mistake it for a completed capture.  The
            # benchmark append operation validates the staging directory before
            # installation, so it remains safe to retry there.
            raise
        return destination


Fetcher = Callable[..., SourceArtifact]


def _iso_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must include an explicit timezone")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str | datetime) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Timestamp must include an explicit timezone")
    return parsed


def _header(headers: Mapping[str, str], name: str) -> str | None:
    wanted = name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted:
            return str(value)
    return None


def _header_datetime(headers: Mapping[str, str], name: str) -> str | None:
    value = _header(headers, name)
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return _iso_utc(parsed)


def _declared_content_length(headers: Mapping[str, str], *, url: str) -> int | None:
    """Return a validated HTTP content length, or ``None`` when absent.

    A malformed response header is source evidence failure, not a parser
    crash.  In particular, do not let ``int()``'s ``ValueError`` escape from
    the network boundary: capture jobs must record an unavailable artifact and
    leave the append-only archive retryable.
    """

    value = _header(headers, "Content-Length")
    if value is None:
        return None
    try:
        length = int(value, 10)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BottenAdaSourceError(f"Invalid Content-Length header for {url}: {value!r}") from exc
    if length < 0:
        raise BottenAdaSourceError(f"Invalid negative Content-Length header for {url}: {value!r}")
    return length


def fetch_source(
    url: str,
    *,
    head_only: bool = False,
    timeout: float = 30.0,
    max_bytes: int = 25_000_000,
) -> SourceArtifact:
    """Fetch one source with bounded reads and transport metadata.

    The default bound intentionally excludes the current ``pop.rds`` body.
    Capturing its HEAD metadata is enough to establish that the publisher
    exposes a large, separately updated model object; downloading it requires
    an explicit custom fetcher and an independent semantic/parity review.
    """

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    method = "HEAD" if head_only else "GET"
    started = datetime.now(timezone.utc)
    request = Request(url, method=method, headers={"Accept": "*/*"})
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL is preregistered above
            headers = {str(k): str(v) for k, v in response.headers.items()}
            status_code = int(getattr(response, "status", 200))
            body: bytes | None
            if head_only:
                body = None
            else:
                declared_length = _declared_content_length(headers, url=url)
                if declared_length is not None and declared_length > max_bytes:
                    raise BottenAdaSourceError(
                        f"Refusing source larger than {max_bytes} bytes ({url}, {declared_length})"
                    )
                body = response.read(max_bytes + 1)
                if len(body) > max_bytes:
                    raise BottenAdaSourceError(f"Source exceeded {max_bytes} bytes: {url}")
            return SourceArtifact(
                url=url,
                body=body,
                retrieved_at_utc=_iso_utc(datetime.now(timezone.utc)),
                status_code=status_code,
                headers=headers,
                method=method,
            )
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, BottenAdaSourceError) as exc:
        # A failed source is represented explicitly by the capture layer.  We
        # retain a retrieval timestamp and never substitute another date's
        # response.
        return SourceArtifact(
            url=url,
            body=None,
            retrieved_at_utc=_iso_utc(started),
            status_code=int(getattr(exc, "code", 0) or 0),
            headers={},
            method=method,
            error=f"{type(exc).__name__}: {exc}",
        )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_relative_name(value: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise BottenAdaCaptureError(f"Unsafe raw evidence path: {value!r}")
    return relative


def _write_exclusive_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_exclusive_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _write_exclusive_bytes(path, payload)


def _json_object(body: bytes, *, source_name: str) -> dict[str, Any]:
    try:
        value = json.loads(body.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BottenAdaParseError(f"{source_name} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise BottenAdaParseError(f"{source_name} must contain a JSON object")
    return value


def _number(value: Any, *, field_name: str, lower: float | None = None, upper: float | None = None) -> float:
    if isinstance(value, bool):
        raise BottenAdaParseError(f"{field_name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BottenAdaParseError(f"{field_name} must be numeric") from exc
    if not math.isfinite(result):
        raise BottenAdaParseError(f"{field_name} must be finite")
    if lower is not None and result < lower:
        raise BottenAdaParseError(f"{field_name} is below {lower}")
    if upper is not None and result > upper:
        raise BottenAdaParseError(f"{field_name} is above {upper}")
    return result


def _quantiles(value: Mapping[str, Any], *, field_name: str, lower: float, upper: float) -> dict[str, float]:
    required = ("p5", "p50", "p95")
    if any(key not in value for key in required):
        raise BottenAdaParseError(f"{field_name} must provide p5, p50, and p95")
    result = {
        key: _number(value[key], field_name=f"{field_name}.{key}", lower=lower, upper=upper)
        for key in required
    }
    if not result["p5"] <= result["p50"] <= result["p95"]:
        raise BottenAdaParseError(f"{field_name} quantiles are not ordered")
    return result


def _metadata(value: Mapping[str, Any], *, source_name: str) -> dict[str, Any]:
    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        raise BottenAdaParseError(f"{source_name} lacks a metadata object")
    # Retain publisher strings verbatim.  ``run_written`` has no timezone in
    # the current API, so it must not be silently interpreted as UTC.
    return dict(metadata)


def _copy_party_forecast(value: Mapping[str, Any], *, source_name: str) -> dict[str, Any]:
    election = value.get("election")
    if not isinstance(election, dict):
        raise BottenAdaParseError(f"{source_name} lacks an election object")
    parties: dict[str, Any] = {}
    for party in PARTY_ORDER:
        entry = election.get(party)
        if not isinstance(entry, dict):
            raise BottenAdaParseError(f"{source_name} election forecast lacks party {party}")
        votes = entry.get("votes")
        if not isinstance(votes, dict):
            raise BottenAdaParseError(f"{source_name} election forecast lacks votes for {party}")
        normalized: dict[str, Any] = {
            "votes": _quantiles(votes, field_name=f"election.{party}.votes", lower=0.0, upper=1.0),
        }
        # Seats are useful secondary evidence but are not assumed to be
        # predictive draws.  Preserve them when the endpoint publishes them.
        if isinstance(entry.get("seats"), dict):
            normalized["seats"] = _quantiles(
                entry["seats"], field_name=f"election.{party}.seats", lower=0.0, upper=349.0
            )
        parties[party] = normalized
    return parties


def parse_forecast_json(body: bytes, *, source_name: str = "forecast.json") -> dict[str, Any]:
    """Parse ``latest_forecast/seats--all.json`` without inventing draws."""

    value = _json_object(body, source_name=source_name)
    metadata = _metadata(value, source_name=source_name)
    parties = _copy_party_forecast(value, source_name=source_name)
    election_day = metadata.get("election_day") or metadata.get("election_date")
    if election_day is not None and str(election_day) != ELECTION_DATE:
        raise BottenAdaParseError(f"{source_name} reports election day {election_day!r}, expected {ELECTION_DATE}")
    n_draws = metadata.get("n_draws")
    if n_draws is not None:
        if isinstance(n_draws, bool) or not isinstance(n_draws, (int, float)) or int(n_draws) != n_draws or n_draws <= 0:
            raise BottenAdaParseError(f"{source_name} metadata.n_draws must be a positive integer")
    return {
        "source_kind": "official_machine_readable_election_forecast",
        "vote_share_unit": "proportion_of_national_valid_votes",
        "party_order": list(PARTY_ORDER),
        "election_date": ELECTION_DATE,
        "metadata": metadata,
        "election": parties,
    }


def parse_threshold_json(
    body: bytes,
    *,
    expected_party: str | None = None,
    source_name: str = "threshold.json",
) -> dict[str, Any]:
    """Extract a publisher-supplied inclusive 4% election probability."""

    value = _json_object(body, source_name=source_name)
    metadata = _metadata(value, source_name=source_name)
    questions = value.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise BottenAdaParseError(f"{source_name} lacks a questions object")
    event_name = next(iter(questions)) if len(questions) == 1 else None
    if expected_party is not None:
        expected_event = THRESHOLD_EVENTS.get(expected_party)
        if event_name is None or event_name != expected_event:
            raise BottenAdaParseError(
                f"{source_name} event does not match expected {expected_event!r}: {list(questions)}"
            )
    if event_name is None:
        raise BottenAdaParseError(f"{source_name} must contain exactly one question event")
    question = questions.get(event_name)
    if not isinstance(question, dict):
        raise BottenAdaParseError(f"{source_name} question must be an object")
    parties = question.get("parties")
    if not isinstance(parties, list) or len(parties) != 1 or str(parties[0]) not in PARTY_ORDER:
        raise BottenAdaParseError(f"{source_name} must identify exactly one modeled party")
    party = str(parties[0])
    if expected_party is not None and party != expected_party:
        raise BottenAdaParseError(f"{source_name} identifies {party}, expected {expected_party}")
    election = question.get("election")
    if not isinstance(election, dict) or "prob" not in election:
        raise BottenAdaParseError(f"{source_name} lacks an election probability")
    probability = _number(election["prob"], field_name=f"{event_name}.election.prob", lower=0.0, upper=1.0)
    n_true = election.get("n_true")
    if n_true is not None:
        if isinstance(n_true, bool) or not isinstance(n_true, (int, float)) or int(n_true) != n_true or n_true < 0:
            raise BottenAdaParseError(f"{source_name} election.n_true must be a non-negative integer")
        if metadata.get("n_draws") is not None and n_true > int(metadata["n_draws"]):
            raise BottenAdaParseError(f"{source_name} election.n_true exceeds metadata.n_draws")
    return {
        "event": event_name,
        "party": party,
        "threshold": 4.0,
        "inclusive": True,
        "probability": probability,
        "n_true": None if n_true is None else int(n_true),
        "description": question.get("desc"),
        "metadata": metadata,
    }


def _date_string(value: Any, *, field_name: str) -> str:
    text = str(value)
    try:
        datetime.fromisoformat(text)
    except ValueError as exc:
        raise BottenAdaParseError(f"{field_name} must be an ISO date") from exc
    return text[:10]


def parse_latest_polls_json(body: bytes, *, source_name: str = "latest_polls.json") -> dict[str, Any]:
    """Parse official latest poll rows and retain source freshness metadata."""

    value = _json_object(body, source_name=source_name)
    metadata = _metadata(value, source_name=source_name)
    latest_polls = value.get("latest_polls")
    if not isinstance(latest_polls, list):
        raise BottenAdaParseError(f"{source_name}.latest_polls must be a list")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(latest_polls):
        if not isinstance(row, dict):
            raise BottenAdaParseError(f"{source_name} poll row {index} must be an object")
        for key in ("house", "publish_date"):
            if key not in row:
                raise BottenAdaParseError(f"{source_name} poll row {index} lacks {key}")
        normalized = dict(row)
        normalized["publish_date"] = _date_string(row["publish_date"], field_name=f"polls[{index}].publish_date")
        if row.get("start_date") is not None:
            normalized["start_date"] = _date_string(row["start_date"], field_name=f"polls[{index}].start_date")
        if row.get("end_date") is not None:
            normalized["end_date"] = _date_string(row["end_date"], field_name=f"polls[{index}].end_date")
        rows.append(normalized)
    latest_poll_date = max((row["publish_date"] for row in rows), default=None)
    return {
        "source_kind": "official_machine_readable_latest_polls",
        "metadata": metadata,
        "latest_poll_date": latest_poll_date,
        "latest_polls": rows,
    }


def _timeseries_row(row: Mapping[str, str], *, index: int, source_name: str) -> dict[str, Any]:
    if not row.get("date"):
        raise BottenAdaParseError(f"{source_name} row {index} lacks date")
    date_value = _date_string(row["date"], field_name=f"timeseries[{index}].date")
    parties: dict[str, dict[str, float]] = {}
    for party in PARTY_ORDER:
        values = {
            quantile: _number(
                row.get(f"{party}_{quantile}", ""),
                field_name=f"timeseries[{index}].{party}_{quantile}",
                lower=0.0,
                upper=1.0,
            )
            for quantile in ("p5", "p50", "p95")
        }
        if not values["p5"] <= values["p50"] <= values["p95"]:
            raise BottenAdaParseError(f"{source_name} row {index} quantiles are not ordered for {party}")
        parties[party] = values
    return {"date": date_value, "parties": parties}


def parse_timeseries_csv(body: bytes, *, source_name: str = "timeseries.csv") -> dict[str, Any]:
    """Parse official ``latest_pop/timeseries.csv`` as a latent-state series.

    The CSV is deliberately kept distinct from ``latest_forecast``.  Its
    election-date row is useful evidence, but it must not silently replace the
    election forecast JSON because the site serves these from different runs
    (and currently different model configurations).
    """

    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BottenAdaParseError(f"{source_name} is not UTF-8 CSV") from exc
    reader = csv.DictReader(text.splitlines())
    required = {"date"} | {f"{party}_{quantile}" for party in PARTY_ORDER for quantile in ("p5", "p50", "p95")}
    fieldnames = set(reader.fieldnames or ())
    if not required.issubset(fieldnames):
        missing = sorted(required - fieldnames)
        raise BottenAdaParseError(f"{source_name} lacks required columns: {missing}")
    rows: list[dict[str, Any]] = []
    previous_date: str | None = None
    for index, row in enumerate(reader):
        parsed = _timeseries_row(row, index=index, source_name=source_name)
        if previous_date is not None and parsed["date"] <= previous_date:
            raise BottenAdaParseError(f"{source_name} dates must be strictly increasing")
        previous_date = parsed["date"]
        rows.append(parsed)
    if not rows:
        raise BottenAdaParseError(f"{source_name} contains no data rows")
    election_row = next((row for row in rows if row["date"] == ELECTION_DATE), None)
    return {
        "source_kind": "official_machine_readable_latent_timeseries",
        "party_order": list(PARTY_ORDER),
        "row_count": len(rows),
        "first_date": rows[0]["date"],
        "last_date": rows[-1]["date"],
        "latest_row": rows[-1],
        "election_day_row": election_row,
    }


def parse_homepage_html(body: bytes, *, source_name: str = "homepage.html") -> dict[str, Any]:
    """Record only stable machine-readable hints from the HTML evidence.

    The homepage is retained verbatim in ``raw/homepage.html``.  We extract
    Nuxt's configured next-election date as a diagnostic, but intentionally do
    not regex-scrape rendered numbers: if the API endpoints are unavailable,
    the raw page is the evidence a human can audit under the fallback rule.
    """

    try:
        text = body.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise BottenAdaParseError(f"{source_name} is not UTF-8 HTML") from exc
    match = re.search(r"nextElectionDate\s*:\s*[\"']([^\"']+)[\"']", text)
    next_election_date = match.group(1) if match else None
    return {
        "source_kind": "official_homepage_raw_evidence",
        "next_election_date": next_election_date,
        "numeric_values_extracted": False,
        "reason": "Machine-readable API artifacts are authoritative; HTML is retained for audit/fallback only.",
    }


def _source_provenance(
    key: str,
    artifact: SourceArtifact,
    spec: SourceSpec,
    *,
    parsed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    body = artifact.body
    header_length = _header(artifact.headers, "Content-Length")
    byte_size: int | None = None
    if body is not None:
        byte_size = len(body)
    elif header_length is not None:
        try:
            byte_size = int(header_length)
        except ValueError:
            byte_size = None
    source_updates: dict[str, Any] = {}
    if parsed and isinstance(parsed.get("metadata"), dict):
        metadata = parsed["metadata"]
        for metadata_key in ("run_written", "created", "latest_date", "election_day", "run", "model"):
            if metadata_key in metadata:
                source_updates[metadata_key] = metadata[metadata_key]
    return {
        "artifact_key": key,
        "url": artifact.url,
        "retrieved_at_utc": artifact.retrieved_at_utc,
        "http_status": artifact.status_code,
        "method": artifact.method,
        "content_sha256": None if body is None else _sha256(body),
        "byte_size": byte_size,
        "etag": _header(artifact.headers, "ETag"),
        "last_modified": _header(artifact.headers, "Last-Modified"),
        "last_modified_utc": _header_datetime(artifact.headers, "Last-Modified"),
        "content_type": _header(artifact.headers, "Content-Type"),
        "raw_path": spec.raw_name if body is not None else None,
        "body_captured": body is not None,
        "source_reported_update": source_updates or None,
        "parsing_version": PARSER_VERSION,
        "error": artifact.error,
    }


def _source_error(artifact: SourceArtifact | None) -> str:
    if artifact is None:
        return "source artifact was not supplied"
    if artifact.error:
        return artifact.error
    return f"HTTP status {artifact.status_code} with no body"


def _coerce_artifact(value: SourceArtifact | Mapping[str, Any], *, key: str) -> SourceArtifact:
    if isinstance(value, SourceArtifact):
        body = value.body
        if isinstance(body, str):
            body = body.encode("utf-8")
        if body is not None and not isinstance(body, bytes):
            raise TypeError(f"Artifact {key}.body must be bytes")
        headers = value.headers
        if headers is None:
            headers = {}
        if not isinstance(headers, Mapping):
            raise TypeError(f"Artifact {key}.headers must be a mapping")
        try:
            status_code = int(value.status_code)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"Artifact {key}.status_code must be an integer") from exc
        return SourceArtifact(
            url=str(value.url),
            body=body,
            retrieved_at_utc=str(value.retrieved_at_utc),
            status_code=status_code,
            headers=dict(headers),
            method=str(value.method),
            error=None if value.error is None else str(value.error),
        )
    if not isinstance(value, Mapping):
        raise TypeError(f"Artifact {key} must be SourceArtifact or mapping")
    body = value.get("body")
    if isinstance(body, str):
        body = body.encode("utf-8")
    if body is not None and not isinstance(body, bytes):
        raise TypeError(f"Artifact {key}.body must be bytes")
    headers = value.get("headers", {})
    if headers is None:
        headers = {}
    if not isinstance(headers, Mapping):
        raise TypeError(f"Artifact {key}.headers must be a mapping")
    status_code = value.get("status_code", 200)
    try:
        status_code = int(status_code)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"Artifact {key}.status_code must be an integer") from exc
    return SourceArtifact(
        url=str(value.get("url", "")),
        body=body,
        retrieved_at_utc=str(value.get("retrieved_at_utc", "")),
        status_code=status_code,
        headers=dict(headers),
        method=str(value.get("method", "GET")),
        error=None if value.get("error") is None else str(value.get("error")),
    )


def _validate_public_artifact_identity(artifacts: Mapping[str, SourceArtifact]) -> None:
    """Require every supplied artifact to be the exact preregistered source.

    Parsing a JSON object with the expected shape is not enough to establish
    that it was published by Botten Ada.  A caller-controlled URL must never
    be able to enter the archive merely by returning plausible bytes.  The
    exact URL check is intentionally strict (including path and trailing
    slash); ``fetch_source`` records the requested URL, so normal HTTP
    redirects do not weaken this identity check.
    """

    unknown = sorted(set(artifacts) - set(DEFAULT_SOURCE_SPECS))
    if unknown:
        raise BottenAdaCaptureError(f"Unexpected Botten Ada source artifact keys: {unknown}")
    for key, artifact in artifacts.items():
        expected = DEFAULT_SOURCE_SPECS[key].url
        if artifact.url != expected:
            raise BottenAdaCaptureError(
                f"{key} artifact URL is not the preregistered Botten Ada source: "
                f"expected {expected!r}, got {artifact.url!r}"
            )


def _source_reported_updates(provenance: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        key: value.get("source_reported_update")
        for key, value in provenance.items()
        if value.get("source_reported_update") is not None
    }


def _decision_generation_identity(parsed: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Require every scoring-bearing latest_forecast object to be one Ada run."""

    missing_sources = [key for key in DECISION_SOURCE_KEYS if key not in parsed]
    if missing_sources:
        # Source/parse errors are classified by the ordinary bundle logic.
        return None, None
    identities: dict[str, dict[str, Any]] = {}
    for key in DECISION_SOURCE_KEYS:
        value = parsed[key]
        metadata = value.get("metadata") if isinstance(value, Mapping) else None
        if not isinstance(metadata, Mapping):
            return None, f"{key} has no metadata for Ada generation identity"
        missing_fields = [
            field
            for field in DECISION_GENERATION_FIELDS
            if metadata.get(field) is None or metadata.get(field) == ""
        ]
        if missing_fields:
            return None, f"{key} lacks Ada generation fields {missing_fields}"
        identities[key] = {field: metadata[field] for field in DECISION_GENERATION_FIELDS}
    expected = identities["forecast"]
    mismatches = {
        key: identity
        for key, identity in identities.items()
        if identity != expected
    }
    if mismatches:
        return None, (
            "decision-bearing latest_forecast artifacts do not share one "
            f"run/model/run_written identity: expected {expected!r}, mismatches {mismatches!r}"
        )
    return dict(expected), None


def parse_public_bundle(
    artifacts: Mapping[str, SourceArtifact | Mapping[str, Any]],
    *,
    expected_election_date: str = ELECTION_DATE,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Normalize official artifacts and return ``(record, raw_files)``.

    The returned record is JSON serializable.  ``raw_files`` maps safe
    archive-relative names (for example ``raw/latest_forecast_seats--all.json``)
    to the exact response bytes.  Missing or malformed optional sources are
    retained in provenance and status, never silently replaced by yesterday's
    values.
    """

    if expected_election_date != ELECTION_DATE:
        raise BottenAdaCaptureError(
            f"This prospective module is frozen for election date {ELECTION_DATE}, got {expected_election_date}"
        )
    try:
        normalized_artifacts: dict[str, SourceArtifact] = {
            key: _coerce_artifact(value, key=key) for key, value in artifacts.items()
        }
    except (TypeError, ValueError) as exc:
        raise BottenAdaCaptureError(f"Malformed Botten Ada source artifact: {exc}") from exc
    _validate_public_artifact_identity(normalized_artifacts)
    provenance: dict[str, Any] = {}
    raw_files: dict[str, bytes] = {}
    parsed: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for key, spec in DEFAULT_SOURCE_SPECS.items():
        artifact = normalized_artifacts.get(key)
        parsed_value: dict[str, Any] | None = None
        # RDS is deliberately a HEAD-only provenance record unless a caller
        # explicitly supplies bytes.  ``body is None`` therefore means the
        # expected safe capture mode, not a source outage.
        if key == "rds" and (artifact is None or (artifact.method.upper() == "HEAD" and artifact.body is None)):
            pass
        elif artifact is None or artifact.body is None or artifact.status_code < 200 or artifact.status_code >= 300:
            errors[key] = _source_error(artifact)
        elif key == "forecast":
            try:
                parsed_value = parse_forecast_json(artifact.body, source_name=key)
            except BottenAdaParseError as exc:
                errors[key] = str(exc)
        elif key.startswith("threshold_"):
            try:
                parsed_value = parse_threshold_json(
                    artifact.body,
                    expected_party=key.removeprefix("threshold_"),
                    source_name=key,
                )
            except BottenAdaParseError as exc:
                errors[key] = str(exc)
        elif key == "latest_polls":
            try:
                parsed_value = parse_latest_polls_json(artifact.body, source_name=key)
            except BottenAdaParseError as exc:
                errors[key] = str(exc)
        elif key == "timeseries":
            try:
                parsed_value = parse_timeseries_csv(artifact.body, source_name=key)
            except BottenAdaParseError as exc:
                errors[key] = str(exc)
        elif key == "homepage":
            try:
                parsed_value = parse_homepage_html(artifact.body, source_name=key)
            except BottenAdaParseError as exc:
                errors[key] = str(exc)
        # ``rds`` is deliberately never parsed.  Its bytes, if a caller
        # explicitly supplies them, are preserved and its semantics remain
        # PARITY_UNVERIFIED until an independent verification step succeeds.
        if parsed_value is not None:
            parsed[key] = parsed_value
        provenance[key] = _source_provenance(key, artifact or SourceArtifact(spec.url, None, ""), spec, parsed=parsed_value)
        if artifact is not None and artifact.body is not None and 200 <= artifact.status_code < 300:
            raw_files[spec.raw_name] = artifact.body

    decision_generation, generation_error = _decision_generation_identity(parsed)
    if generation_error is not None:
        errors["generation_identity"] = generation_error
    decision_bundle_eligible = generation_error is None
    # A mixed Ada generation is retained in the exact raw bytes and source
    # provenance, but none of its decision-bearing quantities may reach the
    # scorer as a synthetic bundle.
    forecast = parsed.get("forecast") if decision_bundle_eligible else None
    threshold_probabilities = {
        parsed_value["party"]: parsed_value
        for key, parsed_value in parsed.items()
        if decision_bundle_eligible
        and key.startswith("threshold_")
        and isinstance(parsed_value, dict)
    }
    latest_polls = parsed.get("latest_polls")
    timeseries = parsed.get("timeseries")

    if "forecast" in errors:
        forecast_artifact = normalized_artifacts.get("forecast")
        source_missing = (
            forecast_artifact is None
            or forecast_artifact.body is None
            or forecast_artifact.status_code < 200
            or forecast_artifact.status_code >= 300
        )
        status = STATUS_SOURCE_UNAVAILABLE if source_missing else STATUS_PARSE_FAILED
    elif "generation_identity" in errors:
        status = STATUS_PARSE_FAILED
    elif any(key in errors for key in normalized_artifacts if key != "rds"):
        # A partial API outage or malformed optional source remains visible in
        # the record; a caller may apply the preregistered metric fallback.
        status = STATUS_SOURCE_UNAVAILABLE if any(
            normalized_artifacts.get(key) is None
            or normalized_artifacts[key].body is None
            or normalized_artifacts[key].status_code < 200
            or normalized_artifacts[key].status_code >= 300
            for key in errors
        ) else STATUS_PARSE_FAILED
    else:
        # Source/publication availability and draw verification are separate
        # concepts. Exact-draw eligibility remains nested under capabilities
        # and provenance.draws.
        status = STATUS_AVAILABLE

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model": "botten_ada",
        "model_name": "Botten Ada",
        "status": status,
        "election_date": ELECTION_DATE,
        "party_order": list(PARTY_ORDER),
        "forecast": forecast,
        "threshold_probabilities_4pct": threshold_probabilities,
        "latest_polls": latest_polls,
        "timeseries": timeseries,
        # These are explicit comparison/freshness fields consumed by the
        # shared benchmark archive.  ``source_updated_at`` is the HTTP
        # Last-Modified value normalized to UTC; the publisher-reported
        # metadata is retained separately because run_written currently has
        # no timezone annotation.
        "source_updated_at": provenance.get("forecast", {}).get("last_modified_utc"),
        "source_updated_at_reported": provenance.get("forecast", {}).get("source_reported_update"),
        "latest_poll_date": None if latest_polls is None else latest_polls.get("latest_poll_date"),
        "errors": errors,
        "capabilities": {
            "verified_predictive_vote_draws": False,
            "verified_predictive_seat_draws": False,
            "published_vote_quantiles": forecast is not None,
            "published_central_predictions": forecast is not None,
            "published_threshold_probabilities": sorted(threshold_probabilities),
            "published_seat_quantiles": bool(
                forecast and all("seats" in forecast["election"][party] for party in PARTY_ORDER)
            ),
            "draw_policy": "Do not infer draws from quantiles or unverified RDS posterior samples.",
        },
        "provenance": {
            "source_urls": {
                "official_site": OFFICIAL_SITE_URL,
                "official_data_page": OFFICIAL_DATA_URL,
                "official_faq": OFFICIAL_FAQ_URL,
                "ada_repository": ADA_REPOSITORY_URL,
                "ada_repository_commit": ADA_REPOSITORY_COMMIT_URL,
                "ada_config_at_commit": ADA_CONFIG_URL,
                "rds": RDS_URL,
            },
            "party_pages": PARTY_PAGE_URLS,
            "machine_readable_notes": {
                "election_forecast": f"{API_BASE_URL}/latest_forecast/seats--all.json",
                "current_timeseries": f"{API_BASE_URL}/latest_pop/timeseries.csv",
                "threshold_events": {
                    party: f"{API_BASE_URL}/latest_forecast/question--{event}.json"
                    for party, event in THRESHOLD_EVENTS.items()
                },
                "party_page_policy": "Root latest_forecast JSON and latest_pop CSV are captured; party HTML pages are redundant evidence and are listed for audit/fallback.",
            },
            "repository": {
                "url": ADA_REPOSITORY_URL,
                "commit": ADA_REPOSITORY_COMMIT,
                "commit_url": ADA_REPOSITORY_COMMIT_URL,
                "license": "MIT",
            },
            "license": {
                "description": ADA_LICENSE,
                "url": ADA_LICENSE_URL,
                "attribution": "Botten Ada / Måns Magnusson and Jonas Wallin; site and publication by Newsworthy.",
            },
            "sources": provenance,
            "source_reported_updates": _source_reported_updates(provenance),
            "decision_generation_identity": decision_generation,
            "draws": {
                "status": STATUS_PARITY_UNVERIFIED,
                "eligible_for_probabilistic_scoring": False,
                "rds_url": RDS_URL,
                "rds_semantics": "The official data page describes pop.rds as containing posterior draws; election-day predictive semantics are not assumed.",
                "reason": "No verified parity evidence was supplied for exact election-day predictive draws.",
            },
            "parsing_version": PARSER_VERSION,
        },
        "raw_file_paths": sorted(raw_files),
    }
    return record, raw_files


def _apply_decision_cutoff(record: dict[str, Any], cutoff: datetime) -> None:
    """Remove scoring evidence not proved to have existed by the cutoff.

    S3 ``Last-Modified`` is the contemporaneous publication evidence for the
    mutable ``latest_forecast`` objects. Raw bytes and their hashes remain in
    provenance even when a source is rejected here.
    """

    provenance = record.get("provenance")
    sources = provenance.get("sources") if isinstance(provenance, Mapping) else None
    violations: dict[str, str] = {}
    observed: dict[str, str | None] = {}
    for key in DECISION_SOURCE_KEYS:
        source = sources.get(key) if isinstance(sources, Mapping) else None
        modified = source.get("last_modified_utc") if isinstance(source, Mapping) else None
        observed[key] = modified if isinstance(modified, str) else None
        if not isinstance(modified, str) or not modified:
            violations[key] = "missing or unparseable HTTP Last-Modified"
            continue
        try:
            modified_at = _parse_timestamp(modified)
        except ValueError:
            violations[key] = f"unparseable HTTP Last-Modified {modified!r}"
            continue
        if modified_at > cutoff:
            violations[key] = (
                f"HTTP Last-Modified {_iso_utc(modified_at)} is after benchmark cutoff "
                f"{_iso_utc(cutoff)}"
            )

    decision_cutoff = {
        "benchmark_cutoff": _iso_utc(cutoff),
        "required_sources": list(DECISION_SOURCE_KEYS),
        "last_modified_utc": observed,
        "eligible": not violations,
        "violations": violations,
    }
    if isinstance(provenance, dict):
        provenance["decision_cutoff"] = decision_cutoff
    if not violations:
        return

    # Preserve enough normalized identity/hash evidence to explain the
    # rejection, while ensuring the ordinary scorer sees no forecast values.
    record["rejected_decision_evidence"] = {
        "reason": "Decision-bearing Ada artifact was not proved available by the cutoff",
        "decision_generation_identity": (
            provenance.get("decision_generation_identity")
            if isinstance(provenance, Mapping)
            else None
        ),
        "source_content_sha256": {
            key: (sources.get(key) or {}).get("content_sha256")
            for key in DECISION_SOURCE_KEYS
        } if isinstance(sources, Mapping) else {},
    }
    errors = dict(record.get("errors", {}))
    errors["cutoff_eligibility"] = json.dumps(violations, sort_keys=True)
    record["errors"] = errors
    record["status"] = STATUS_SOURCE_UNAVAILABLE
    record["forecast"] = None
    record["threshold_probabilities_4pct"] = {}
    capabilities = record.get("capabilities")
    if isinstance(capabilities, dict):
        capabilities["published_vote_quantiles"] = False
        capabilities["published_central_predictions"] = False
        capabilities["published_threshold_probabilities"] = []
        capabilities["published_seat_quantiles"] = False


def _call_fetcher(fetcher: Fetcher, spec: SourceSpec) -> SourceArtifact:
    """Call modern or simple injected fetchers without hiding fetch errors."""

    try:
        value = fetcher(spec.url, head_only=spec.head_only)
    except TypeError as first_error:
        # Fixture tests often use ``lambda url: ...``.  Only retry when the
        # callable rejected the keyword; an artifact returned by the second
        # call is still validated by ``_coerce_artifact``.
        try:
            value = fetcher(spec.url)
        except TypeError:
            raise first_error
    return _coerce_artifact(value, key=spec.key)


def capture_botten_ada(
    benchmark_cutoff: str | datetime,
    *,
    fetcher: Fetcher = fetch_source,
    source_specs: Mapping[str, SourceSpec] | None = None,
    stale_before: str | datetime | None = None,
    output_dir: Path | str | None = None,
) -> BottenAdaCapture:
    """Capture all preregistered Botten Ada sources for one cutoff.

    Parameters
    ----------
    benchmark_cutoff:
        An aware datetime or aware ISO timestamp.  The value is recorded in
        UTC and Europe/Stockholm; schedule eligibility is decided by the
        benchmark's shared ``time_rules`` module, not by this source parser.
    fetcher:
        Callable accepting ``url`` and optional ``head_only`` and returning a
        :class:`SourceArtifact` (or equivalent mapping).  The default performs
        bounded HTTP GET/HEAD requests.
    source_specs:
        Optional fixture or test source map.  Any omitted key is not fetched;
        this permits deterministic outage tests without network access.
    stale_before:
        Optional explicit UTC/aware cutoff used by a caller's pre-registered
        freshness policy.  No freshness threshold is guessed here: when
        supplied, a forecast whose HTTP ``Last-Modified`` is older is marked
        ``SOURCE_STALE`` with evidence in ``errors``.
    output_dir:
        If supplied, write the model's normalized ``forecast.json``,
        ``provenance.json``, and exact raw bytes with refuse-on-existing
        semantics.  The returned object is always available to the caller.
    """

    parsed_cutoff = _parse_timestamp(benchmark_cutoff)
    cutoff_record = {
        "benchmark_cutoff": _iso_utc(parsed_cutoff),
        "benchmark_cutoff_europe_stockholm": parsed_cutoff.astimezone(ZoneInfo("Europe/Stockholm")).isoformat(),
    }
    specs = dict(source_specs or DEFAULT_SOURCE_SPECS)
    artifacts: dict[str, SourceArtifact] = {}
    for key, spec in specs.items():
        try:
            artifacts[key] = _call_fetcher(fetcher, spec)
        except (BottenAdaSourceError, OSError, URLError, HTTPError) as exc:
            artifacts[key] = SourceArtifact(
                url=spec.url,
                body=None,
                retrieved_at_utc=_iso_utc(datetime.now(timezone.utc)),
                status_code=int(getattr(exc, "code", 0) or 0),
                method="HEAD" if spec.head_only else "GET",
                error=f"{type(exc).__name__}: {exc}",
            )
    record, raw_files = parse_public_bundle(artifacts)
    record = dict(record)
    record["capture"] = cutoff_record
    record["provenance"] = dict(record["provenance"])
    record["provenance"]["capture_cutoff"] = cutoff_record
    _apply_decision_cutoff(record, parsed_cutoff)
    if stale_before is not None:
        stale_bound = _parse_timestamp(stale_before)
        updated_at = record.get("source_updated_at")
        stale = False
        if updated_at:
            try:
                stale = _parse_timestamp(str(updated_at)) < stale_bound
            except ValueError:
                # An unparseable validator is retained as provenance but cannot
                # support a freshness claim; fail closed as unverified rather
                # than silently calling the source fresh.
                stale = True
        else:
            stale = True
        if stale and record.get("status") not in {STATUS_SOURCE_UNAVAILABLE, STATUS_PARSE_FAILED}:
            record["status"] = STATUS_SOURCE_STALE
            errors = dict(record.get("errors", {}))
            errors["freshness"] = (
                f"forecast Last-Modified {updated_at!r} is older than explicit stale_before "
                f"{_iso_utc(stale_bound)}"
            )
            record["errors"] = errors
            record["provenance"]["freshness_policy"] = {
                "stale_before": _iso_utc(stale_bound),
                "status": STATUS_SOURCE_STALE,
            }
    capture = BottenAdaCapture(record=record, raw_files=raw_files)
    if output_dir is not None:
        capture.write(output_dir)
    return capture


def _unverified_parity(reason: str) -> dict[str, Any]:
    return {
        "status": STATUS_PARITY_UNVERIFIED,
        "eligible_for_probabilistic_scoring": False,
        "reason": reason,
    }


def _validate_frozen_tolerance(value: float, *, name: str, maximum: float) -> float:
    """Validate a parity tolerance without allowing the protocol to widen."""

    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    if parsed > maximum:
        raise ValueError(
            f"{name} cannot exceed the frozen protocol maximum {maximum}"
        )
    return parsed


def _coerce_vote_draws(vote_draws: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Validate and canonicalize an eight-party proportion matrix."""

    try:
        draws = np.asarray(vote_draws, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BottenAdaCaptureError("vote_draws must be numeric") from exc
    if draws.ndim != 2 or draws.shape[0] == 0 or draws.shape[1] != len(PARTY_ORDER):
        raise BottenAdaCaptureError(f"vote_draws must have shape (N, {len(PARTY_ORDER)})")
    if not np.all(np.isfinite(draws)) or np.any(draws < 0.0) or np.any(draws > 1.0):
        raise BottenAdaCaptureError("vote_draws must be finite proportions in [0, 1]")
    # The digest used by the verified-draw gate is independent of native byte
    # order, strides, and the caller's input container.  Party order and unit
    # are included in the digest descriptor so a column permutation cannot be
    # presented as the same extracted evidence.
    return np.ascontiguousarray(draws, dtype="<f8")


def _draw_matrix_digest(draws: np.ndarray) -> str:
    descriptor = json.dumps(
        {
            "dtype": draws.dtype.str,
            "shape": [int(value) for value in draws.shape],
            "party_order": list(PARTY_ORDER),
            "unit": "proportion_of_national_valid_votes",
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(descriptor)
    digest.update(b"\x00")
    digest.update(draws.tobytes(order="C"))
    return digest.hexdigest()


def draw_matrix_sha256(vote_draws: Sequence[Sequence[float]] | np.ndarray) -> str:
    """Return the canonical SHA-256 used in ``verify_official_draws``.

    Callers extracting a future official draw object should record this digest
    in ``draw_provenance['draws_sha256']``.  The digest covers dtype, shape,
    canonical party order, units, and bytes; it is not a hash of quantiles or
    of a caller-selected summary.
    """

    return _draw_matrix_digest(_coerce_vote_draws(vote_draws))


def _sha256_digest(value: Any, *, field_name: str) -> str | None:
    if not isinstance(value, str) or len(value) != 64:
        return None
    try:
        int(value, 16)
    except ValueError:
        return None
    return value.lower()


def _nonnegative_integer(value: Any, *, field_name: str, positive: bool = False) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        finite = math.isfinite(float(value))
        integer = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not finite or integer != value:
        return None
    if (positive and integer <= 0) or (not positive and integer < 0):
        return None
    return integer


def _bound_evidence_artifact(
    value: SourceArtifact | Mapping[str, Any] | None,
    *,
    key: str,
    expected_url: str,
    declared_sha256: Any,
    declared_byte_size: Any,
) -> tuple[SourceArtifact | None, str | None]:
    """Validate bytes supplied as cryptographic evidence for a draw gate."""

    digest = _sha256_digest(declared_sha256, field_name=f"{key}.sha256")
    if digest is None:
        return None, f"{key} requires a SHA-256 digest of the exact archived bytes"
    size = _nonnegative_integer(declared_byte_size, field_name=f"{key}.byte_size", positive=True)
    if size is None:
        return None, f"{key} requires a positive exact byte size"
    if value is None:
        return None, f"{key} bytes were not supplied for hash validation"
    try:
        artifact = _coerce_artifact(value, key=key)
    except (TypeError, ValueError, BottenAdaCaptureError) as exc:
        return None, f"{key} evidence artifact is malformed: {exc}"
    if artifact.url != expected_url:
        return None, f"{key} evidence URL is not the preregistered official source"
    if artifact.body is None:
        return None, f"{key} evidence artifact has no body bytes"
    if artifact.status_code < 200 or artifact.status_code >= 300 or artifact.error:
        return None, f"{key} evidence artifact is not a successful response"
    if artifact.method.upper() != "GET":
        return None, f"{key} evidence requires the exact GET body, not method {artifact.method!r}"
    actual_digest = _sha256(artifact.body)
    if actual_digest != digest:
        return None, f"{key} SHA-256 does not match the supplied official bytes"
    if len(artifact.body) != size:
        return None, f"{key} byte size does not match the supplied official bytes"
    return artifact, None


def parity_evaluate(
    vote_draws: Sequence[Sequence[float]] | np.ndarray,
    published_forecast: Mapping[str, Any],
    *,
    threshold_probabilities: Mapping[str, Any] | None = None,
    expected_n_draws: int | None = None,
    tolerance: float = PARITY_VOTE_TOLERANCE_PROPORTION,
    threshold_tolerance: float = PARITY_THRESHOLD_TOLERANCE_PROBABILITY,
) -> dict[str, Any]:
    """Check exact supplied draws against published p5/p50/p95 and thresholds.

    This function does not read RDS files or construct draws.  It only checks
    an already extracted matrix supplied by a caller who can document its
    origin.  Values in the official API are proportions, so the threshold is
    tested at ``0.04``.  The two defaults are frozen separately in the
    protocol: 0.051 percentage points for published vote-share quantiles and
    0.51 percentage points for published threshold probabilities.
    """

    tolerance = _validate_frozen_tolerance(
        tolerance,
        name="tolerance",
        maximum=PARITY_VOTE_TOLERANCE_PROPORTION,
    )
    threshold_tolerance = _validate_frozen_tolerance(
        threshold_tolerance,
        name="threshold_tolerance",
        maximum=PARITY_THRESHOLD_TOLERANCE_PROBABILITY,
    )
    draws = _coerce_vote_draws(vote_draws)
    if not isinstance(published_forecast, Mapping):
        raise BottenAdaCaptureError("published_forecast must be a mapping")
    if expected_n_draws is not None and draws.shape[0] != expected_n_draws:
        return _unverified_parity(
            f"draw count {draws.shape[0]} does not match publisher-declared count {expected_n_draws}"
        )
    election = published_forecast.get("election") if isinstance(published_forecast, Mapping) else None
    if not isinstance(election, Mapping):
        raise BottenAdaCaptureError("published_forecast lacks election values")
    quantile_checks: dict[str, Any] = {}
    max_error = 0.0
    for index, party in enumerate(PARTY_ORDER):
        entry = election.get(party)
        if not isinstance(entry, Mapping) or not isinstance(entry.get("votes"), Mapping):
            raise BottenAdaCaptureError(f"published_forecast lacks election votes for {party}")
        published = _quantiles(entry["votes"], field_name=f"published.{party}.votes", lower=0.0, upper=1.0)
        actual = np.quantile(draws[:, index], [0.05, 0.50, 0.95])
        errors = {
            key: abs(float(actual[position]) - published[key])
            for position, key in enumerate(("p5", "p50", "p95"))
        }
        max_error = max(max_error, *errors.values())
        quantile_checks[party] = {
            "draw_quantiles": {key: float(actual[position]) for position, key in enumerate(("p5", "p50", "p95"))},
            "published_quantiles": published,
            "absolute_error": errors,
            "within_tolerance": all(error <= tolerance for error in errors.values()),
        }
    threshold_checks: dict[str, Any] = {}
    if threshold_probabilities:
        for party, published_value in threshold_probabilities.items():
            if party not in PARTY_ORDER:
                continue
            if isinstance(published_value, Mapping):
                published_probability = _number(
                    published_value.get("probability"),
                    field_name=f"published_threshold.{party}",
                    lower=0.0,
                    upper=1.0,
                )
            else:
                published_probability = _number(
                    published_value,
                    field_name=f"published_threshold.{party}",
                    lower=0.0,
                    upper=1.0,
                )
            probability = float(np.mean(draws[:, PARTY_ORDER.index(party)] >= 0.04))
            error = abs(probability - published_probability)
            threshold_checks[party] = {
                "draw_probability": probability,
                "published_probability": published_probability,
                "absolute_error": error,
                "within_tolerance": error <= threshold_tolerance,
                "inclusive_threshold": 0.04,
            }
    verified = max_error <= tolerance and all(item["within_tolerance"] for item in threshold_checks.values())
    return {
        "status": STATUS_PARITY_VERIFIED if verified else STATUS_PARITY_UNVERIFIED,
        "eligible_for_probabilistic_scoring": bool(verified),
        "draw_count": int(draws.shape[0]),
        # ``tolerance`` is retained as a backwards-compatible alias for the
        # vote-share tolerance; new consumers should use the explicit fields.
        "tolerance": tolerance,
        "vote_tolerance": tolerance,
        "threshold_tolerance": threshold_tolerance,
        "max_quantile_absolute_error": max_error,
        "quantiles": quantile_checks,
        "thresholds": threshold_checks,
        "reason": None if verified else "Supplied draws do not reproduce all published election quantities within tolerance.",
    }


def verify_official_draws(
    vote_draws: Sequence[Sequence[float]] | np.ndarray,
    *,
    draw_provenance: Mapping[str, Any],
    published_forecast: Mapping[str, Any],
    threshold_probabilities: Mapping[str, Any] | None = None,
    tolerance: float = PARITY_VOTE_TOLERANCE_PROPORTION,
    threshold_tolerance: float = PARITY_THRESHOLD_TOLERANCE_PROBABILITY,
    source_artifact: SourceArtifact | Mapping[str, Any] | None = None,
    semantic_evidence_artifact: SourceArtifact | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Gate exact draws on explicit semantic evidence and parity.

    ``draw_provenance`` is deliberately verbose.  A bare RDS path, a declared
    sample count, or a posterior object is insufficient.  The caller must
    assert that the extracted matrix is election-day predictive draws and
    provide a human-auditable evidence reference before this function can
    return ``VERIFIED``.

    The cryptographic binding is part of this gate, rather than a convention
    left to callers.  A verified call must provide:

    * ``source_artifact``: the exact successful GET body for ``RDS_URL``;
    * ``draw_provenance.source_sha256`` and ``source_byte_size`` matching that
      body;
    * ``draw_provenance.draws_sha256`` matching :func:`draw_matrix_sha256`;
    * ``semantic_evidence_artifact`` from one of the pinned official evidence
      URLs, with matching ``semantic_evidence_sha256`` and byte size; and
    * non-empty ``extraction_method`` and ``extraction_version`` fields.

    This means a local matrix cannot become an official forecast by claiming
    an official URL and a free-form reference.  The RDS and semantic evidence
    bytes are passed in by the caller and are never fetched implicitly here.
    If either artifact is unavailable, the result is explicitly
    ``PARITY_UNVERIFIED``.
    """

    provenance = dict(draw_provenance) if isinstance(draw_provenance, Mapping) else {}

    def rejected(reason: str) -> dict[str, Any]:
        result = _unverified_parity(reason)
        result["draw_provenance"] = provenance
        return result

    if not isinstance(draw_provenance, Mapping):
        return rejected("draw_provenance must be a mapping")
    if draw_provenance.get("source_url") != RDS_URL:
        return rejected("Exact draws are not tied to the official Botten Ada RDS URL")
    if draw_provenance.get("draw_role") != "election_day_predictive_draws":
        return rejected(
            "RDS/posterior samples were supplied without an explicit election_day_predictive_draws role"
        )
    reference = draw_provenance.get("semantic_evidence_reference")
    if not isinstance(reference, str) or not reference.strip():
        return rejected("No auditable evidence reference establishes RDS election-day predictive semantics")
    for field_name in ("extraction_method", "extraction_version"):
        value = draw_provenance.get(field_name)
        if not isinstance(value, str) or not value.strip():
            return rejected(f"Verified draws require a non-empty {field_name}")

    draws = _coerce_vote_draws(vote_draws)
    declared_draws_digest = _sha256_digest(
        draw_provenance.get("draws_sha256"),
        field_name="draws_sha256",
    )
    if declared_draws_digest is None:
        return rejected("Verified draws require draws_sha256 for the exact canonical matrix")
    actual_draws_digest = _draw_matrix_digest(draws)
    if actual_draws_digest != declared_draws_digest:
        return rejected("draws_sha256 does not match the supplied vote-draw matrix")

    _, source_error = _bound_evidence_artifact(
        source_artifact,
        key="source_artifact",
        expected_url=RDS_URL,
        declared_sha256=draw_provenance.get("source_sha256"),
        declared_byte_size=draw_provenance.get("source_byte_size"),
    )
    if source_error is not None:
        return rejected(source_error)

    semantic_url = draw_provenance.get("semantic_evidence_url")
    if not isinstance(semantic_url, str) or semantic_url not in SEMANTIC_EVIDENCE_URLS:
        return rejected(
            "semantic_evidence_url must be one of the pinned official Botten Ada or repository sources"
        )
    _, semantic_error = _bound_evidence_artifact(
        semantic_evidence_artifact,
        key="semantic_evidence_artifact",
        expected_url=str(semantic_url),
        declared_sha256=draw_provenance.get("semantic_evidence_sha256"),
        declared_byte_size=draw_provenance.get("semantic_evidence_byte_size"),
    )
    if semantic_error is not None:
        return rejected(semantic_error)

    if not isinstance(published_forecast, Mapping):
        return rejected("published_forecast must be a mapping")
    metadata = published_forecast.get("metadata", {})
    expected_n_draws = metadata.get("n_draws") if isinstance(metadata, Mapping) else None
    if expected_n_draws is not None:
        expected_n_draws = _nonnegative_integer(
            expected_n_draws,
            field_name="published_forecast.metadata.n_draws",
            positive=True,
        )
        if expected_n_draws is None:
            return rejected("published_forecast.metadata.n_draws must be a positive integer")
    parity = parity_evaluate(
        draws,
        published_forecast,
        threshold_probabilities=threshold_probabilities,
        expected_n_draws=expected_n_draws,
        tolerance=tolerance,
        threshold_tolerance=threshold_tolerance,
    )
    parity["draw_provenance"] = provenance
    return parity


def require_verified_draws(parity: Mapping[str, Any]) -> None:
    """Fail closed when a caller tries to score unverified exact draws."""

    if parity.get("status") != STATUS_PARITY_VERIFIED or not parity.get("eligible_for_probabilistic_scoring"):
        raise BottenAdaDrawsNotVerified(str(parity.get("reason") or "Botten Ada draws are not verified"))


__all__ = [
    "ADA_LICENSE",
    "ADA_LICENSE_URL",
    "ADA_REPOSITORY_COMMIT",
    "ADA_REPOSITORY_COMMIT_URL",
    "ADA_REPOSITORY_URL",
    "ADA_CONFIG_URL",
    "API_BASE_URL",
    "BottenAdaCapture",
    "BottenAdaCaptureError",
    "BottenAdaDrawsNotVerified",
    "BottenAdaParseError",
    "BottenAdaSourceError",
    "DECISION_GENERATION_FIELDS",
    "DECISION_SOURCE_KEYS",
    "DEFAULT_SOURCE_SPECS",
    "ELECTION_DATE",
    "OFFICIAL_DATA_URL",
    "OFFICIAL_FAQ_URL",
    "OFFICIAL_SITE_URL",
    "PARITY_THRESHOLD_TOLERANCE_PROBABILITY",
    "PARITY_VOTE_TOLERANCE_PERCENTAGE_POINTS",
    "PARITY_VOTE_TOLERANCE_PROPORTION",
    "PARSER_VERSION",
    "PARTY_ORDER",
    "PARTY_PAGE_URLS",
    "RDS_URL",
    "SCHEMA_VERSION",
    "SEMANTIC_EVIDENCE_URLS",
    "STATUS_COMPLETE",
    "STATUS_AVAILABLE",
    "STATUS_PARITY_UNVERIFIED",
    "STATUS_PARITY_VERIFIED",
    "STATUS_PARSE_FAILED",
    "STATUS_SOURCE_STALE",
    "STATUS_SOURCE_UNAVAILABLE",
    "SourceArtifact",
    "SourceSpec",
    "THRESHOLD_EVENTS",
    "capture_botten_ada",
    "draw_matrix_sha256",
    "fetch_source",
    "parse_forecast_json",
    "parse_homepage_html",
    "parse_latest_polls_json",
    "parse_public_bundle",
    "parse_threshold_json",
    "parse_timeseries_csv",
    "parity_evaluate",
    "require_verified_draws",
    "verify_official_draws",
]
