"""Network acquisition with provenance and an upstream-host circuit breaker."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
import csv
import io
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import SOURCES, Source
from .normalize import (
    PARTY_ORDER,
    parse_homepage_polls,
    parse_party_chart_payload,
    parse_party_chart_pop_series,
    parse_reference_date,
    parse_timeseries_payload,
)


# Polite identification only.  This names the repository that operates the
# ingestion so a source maintainer can find us; it carries no credentials and
# affects nothing about what is requested or how a response is handled.
USER_AGENT = (
    "election-simulator-pollofpolls-ingestion/1.0 "
    "(+https://github.com/edvinli/election-simulator)"
)
USEFUL_HEADERS = {
    "content-type",
    "content-length",
    "date",
    "etag",
    "last-modified",
    "memento-datetime",
    "x-archive-orig-date",
    "x-archive-orig-content-type",
    "x-archive-orig-etag",
    "x-archive-orig-last-modified",
    "x-archive-src",
}


class AcquisitionError(RuntimeError):
    """Raised when neither the first-party source nor its preserved copy works."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = list(diagnostics or [])


REQUIRED_TIMESERIES_PARTIES = tuple(party for party in PARTY_ORDER if party != "other")
REQUIRED_SWEDISHPOLLS_COLUMNS = {
    "PublYearMonth",
    "Company",
    "PublDate",
    "collectPeriodFrom",
    "collectPeriodTo",
    "approxPeriod",
    "house",
}
REQUIRED_SWEDISHPOLLS_SOURCES_COLUMNS = {"Company", "PublYearMonth", "PublDate", "source"}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_url(url: str, timeout: float) -> tuple[bytes, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/csv,text/html;q=0.9,*/*;q=0.1",
            "Accept-Encoding": "identity",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
        headers = {
            key.lower(): value
            for key, value in response.headers.items()
            if key.lower() in USEFUL_HEADERS
        }
        return payload, {
            "http_status": response.status,
            "final_url": response.geturl(),
            "http_headers": headers,
        }


def _decode_for_validation(payload: bytes) -> str:
    """Decode a source payload without retaining or printing its contents."""

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("source payload is not decodable as UTF-8 or Latin-1")


def _response_endpoint(response: dict[str, Any], requested_url: str) -> tuple[str | None, str]:
    """Return only safe endpoint diagnostics, never response content."""

    final_url = str(response.get("final_url") or requested_url)
    parsed = urlparse(final_url)
    return parsed.hostname or parsed.netloc or None, parsed.path or "/"


def _safe_requested_url(url: str) -> str:
    """Remove URL credentials/query data before a URL enters probe output."""

    parsed = urlparse(url)
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    if not parsed.scheme or not host:
        return parsed.path or "<invalid-url>"
    return f"{parsed.scheme}://{host}{parsed.path or '/'}"


def _attempt_diagnostic(
    source: Source,
    *,
    method: str,
    requested_url: str,
    response: dict[str, Any] | None = None,
    payload: bytes | None = None,
    semantic: str = "FAIL",
    retained_previous: bool = False,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a payload-free record for one source attempt.

    The diagnostic deliberately records the final host/path and response
    shape, but never the body.  ``semantic`` is PASS only after the
    source-kind parser and required-column checks have succeeded.
    """

    response = response or {}
    raw_headers = response.get("http_headers") or {}
    headers = {str(key).lower(): value for key, value in raw_headers.items()}
    final_host, final_path = _response_endpoint(response, requested_url)
    byte_length = len(payload) if payload is not None else 0
    diagnostic: dict[str, Any] = {
        "source_key": source.key,
        "retrieval_method": method,
        "requested_url": _safe_requested_url(requested_url),
        "http_status": response.get("http_status"),
        "final_host": final_host,
        "final_path": final_path,
        "final_host_path": f"{final_host}{final_path}" if final_host else final_path,
        "content_type": headers.get("content-type"),
        "byte_length": byte_length,
        # Keep the shorter aliases convenient for shell/JSON consumers while
        # retaining an explicit name for the semantic gate.
        "bytes": byte_length,
        "semantic": semantic,
        "semantic_validation": semantic,
        "retained_previous": bool(retained_previous),
    }
    if error:
        # Parser errors contain a useful reason (for example, the missing
        # latest-polls table), but truncate defensively so a future parser
        # cannot put a response body into probe logs.
        diagnostic["error"] = str(error)[:240]
    return diagnostic


def _safe_validation_error(exc: BaseException) -> str:
    """Keep parser diagnostics useful without echoing source cell contents."""

    # ``parse_date`` and ``float`` include the offending value in their
    # standard exception text.  Strip quoted values before the message is
    # persisted or surfaced by the probe; diagnostics describe shape/gates,
    # never payload data.
    message = re.sub(r"(['\"])(.*?)\1", "<value>", str(exc), flags=re.DOTALL)
    return message[:240]


def _homepage_reference(response: dict[str, Any]) -> date:
    headers = response.get("http_headers") or {}
    raw = (
        headers.get("memento-datetime")
        or headers.get("x-archive-orig-date")
        or headers.get("date")
    )
    if raw:
        try:
            return parse_reference_date(str(raw))
        except (TypeError, ValueError):
            # A malformed response Date header must not make a valid body
            # unusable.  The body parser still supplies the decisive gate.
            pass
    return datetime.now(timezone.utc).date()


def _validate_source_payload(
    source: Source,
    payload: bytes,
    response: dict[str, Any] | None = None,
) -> None:
    """Semantically validate one fetched payload before it can be accepted.

    The status is checked when available as a basic transport gate, but a 2xx
    status is not sufficient: the parser/schema gates below distinguish a real
    source response from a block page or an HTML error document served with
    status 200.
    """

    if not payload.strip():
        raise ValueError("source payload is empty")

    status = (response or {}).get("http_status")
    if status is not None:
        try:
            status_code = int(status)
        except (TypeError, ValueError) as exc:
            raise ValueError("source response has an invalid HTTP status") from exc
        if not 200 <= status_code < 300:
            raise ValueError(f"source response has HTTP status {status_code}")

    if source.kind == "homepage":
        polls = parse_homepage_polls(payload, _homepage_reference(response or {}))
        if not polls:
            raise ValueError("homepage latest-polls table has no poll rows")
        return

    if source.kind == "timeseries":
        rows, source_labels = parse_timeseries_payload(payload)
        if not rows:
            raise ValueError("timeseries source has no observations")
        missing = sorted(set(REQUIRED_TIMESERIES_PARTIES) - set(source_labels))
        if missing:
            raise ValueError(
                "timeseries source is missing required party columns: "
                + ", ".join(missing)
            )
        return

    if source.kind == "party_chart":
        if source.party is None:
            raise ValueError("party chart source has no configured party")
        # Validate every displayed chart column, not just pofp, so a malformed
        # CSV cannot pass a narrow date/pofp probe and fail during normalize.
        chart = parse_party_chart_payload(payload, source.party)
        pop_series = parse_party_chart_pop_series(payload, source.party)
        if not chart or not pop_series:
            raise ValueError(f"party {source.party} chart CSV has no observations")
        return

    if source.kind == "supplementary_individual":
        reader = csv.DictReader(io.StringIO(_decode_for_validation(payload)))
        fields = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_SWEDISHPOLLS_COLUMNS - fields)
        if missing:
            raise ValueError(
                "SwedishPolls CSV is missing required columns: " + ", ".join(missing)
            )
        if next(reader, None) is None:
            raise ValueError("SwedishPolls CSV has no observations")
        return

    if source.kind == "supplementary_provenance":
        reader = csv.DictReader(io.StringIO(_decode_for_validation(payload)), delimiter=";")
        fields = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_SWEDISHPOLLS_SOURCES_COLUMNS - fields)
        if missing:
            raise ValueError(
                "SwedishPolls sources CSV is missing required columns: "
                + ", ".join(missing)
            )
        if next(reader, None) is None:
            raise ValueError("SwedishPolls sources CSV has no observations")
        return

    raise ValueError(f"no semantic validator configured for source kind {source.kind!r}")


def _response_from_error(exc: BaseException, requested_url: str) -> dict[str, Any]:
    """Extract safe HTTP diagnostics from a failed urllib request."""

    headers: dict[str, Any] = {}
    raw_headers = getattr(exc, "headers", None)
    if raw_headers is not None:
        try:
            headers = {
                key.lower(): value
                for key, value in raw_headers.items()
                if key.lower() in USEFUL_HEADERS
            }
        except AttributeError:
            headers = {}
    geturl = getattr(exc, "geturl", None)
    final_url = geturl() if callable(geturl) else requested_url
    return {
        "http_status": getattr(exc, "code", None),
        "final_url": final_url,
        "http_headers": headers,
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "sources": {}}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    temporary = staging / f"{path.name}.{os.getpid()}.tmp"
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _record(
    source: Source,
    payload: bytes,
    response: dict[str, Any],
    method: str,
    retrieval_url: str,
    old_record: dict[str, Any] | None,
) -> dict[str, Any]:
    digest = sha256_bytes(payload)
    retrieved_at = _timestamp()
    if old_record and old_record.get("sha256") == digest:
        # An unchanged source produces byte-for-byte stable processed outputs.
        retrieved_at = old_record.get("retrieved_at", retrieved_at)
    raw_headers = response.get("http_headers", {})
    headers = {str(key).lower(): value for key, value in raw_headers.items()}
    final_host, final_path = _response_endpoint(response, retrieval_url)
    return {
        "source_key": source.key,
        "source_url": source.url,
        "source_page_url": source.page_url,
        "retrieval_url": retrieval_url,
        "retrieval_method": method,
        "retrieved_at": retrieved_at,
        "upstream_capture_at": (
            headers.get("memento-datetime") or headers.get("x-archive-orig-date")
        ),
        "final_url": response.get("final_url"),
        "http_status": response.get("http_status"),
        "http_headers": headers,
        "final_host": final_host,
        "final_path": final_path,
        "raw_filename": source.raw_filename,
        "sha256": digest,
        "bytes": len(payload),
        "semantic_validation": "PASS",
    }


def acquire_all(
    raw_dir: Path,
    *,
    offline: bool = False,
    allow_archive_fallback: bool = True,
    timeout: float = 45.0,
) -> tuple[dict[str, Any], list[str]]:
    """Acquire configured sources, validating bodies before accepting them.

    A 455 response, TLS certificate error, or connection-level failure opens a
    circuit breaker for the Pollofpolls host.  A semantically invalid 200 is
    scoped to that source: it is rejected and sent through its configured
    fallback, while later source keys still receive their own live attempts.
    Every candidate body is parsed before ``_record`` or ``_atomic_write`` is
    reached, so a malformed response can never replace a verified raw payload.
    """

    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "retrieval_manifest.json"
    old_manifest = _load_manifest(manifest_path)
    old_sources = old_manifest.get("sources", {})
    records: dict[str, Any] = {}
    messages: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    source_outcomes: dict[str, str] = {}
    live_host_unavailable = False

    for index, source in enumerate(SOURCES):
        destination = raw_dir / source.raw_filename
        old_record = old_sources.get(source.key)

        if offline:
            if not destination.exists() or not old_record:
                raise AcquisitionError(
                    f"offline mode requires {destination} and a manifest record",
                    diagnostics=diagnostics,
                )
            previous_payload = destination.read_bytes()
            actual_hash = sha256_bytes(previous_payload)
            if actual_hash != old_record.get("sha256"):
                raise AcquisitionError(
                    f"raw file hash differs from manifest: {destination}",
                    diagnostics=diagnostics,
                )
            previous_response = {
                "http_status": old_record.get("http_status"),
                "final_url": old_record.get("final_url") or old_record.get("source_url"),
                "http_headers": old_record.get("http_headers") or {},
            }
            try:
                _validate_source_payload(source, previous_payload, previous_response)
            except Exception as exc:
                diagnostics.append(
                    _attempt_diagnostic(
                        source,
                        method="verified_local_snapshot",
                        requested_url=str(old_record.get("source_url") or source.url),
                        response=previous_response,
                        payload=previous_payload,
                        semantic="FAIL",
                        error=f"{type(exc).__name__}: {_safe_validation_error(exc)}",
                    )
                )
                raise AcquisitionError(
                    f"verified raw file failed semantic validation: {destination}",
                    diagnostics=diagnostics,
                ) from exc
            diagnostics.append(
                _attempt_diagnostic(
                    source,
                    method="verified_local_snapshot",
                    requested_url=str(old_record.get("source_url") or source.url),
                    response=previous_response,
                    payload=previous_payload,
                    semantic="PASS",
                    retained_previous=True,
                )
            )
            records[source.key] = old_record
            source_outcomes[source.key] = "verified_local_snapshot"
            continue

        attempts: list[tuple[str, str]] = []
        is_pollofpolls = urlparse(source.url).hostname == "pollofpolls.se"
        if not (live_host_unavailable and is_pollofpolls):
            attempts.append(
                ("first_party_http" if is_pollofpolls else "direct_repository_http", source.url)
            )
        if allow_archive_fallback and source.allow_archive_fallback:
            attempts.append(("wayback_preserved_first_party_response", source.archive_url))

        payload: bytes | None = None
        record: dict[str, Any] | None = None
        failures: list[str] = []
        source_attempt_diagnostics: list[dict[str, Any]] = []
        for method, url in attempts:
            candidate_payload: bytes | None = None
            response: dict[str, Any] = {}
            try:
                candidate_payload, response = _read_url(url, timeout)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                status = getattr(exc, "code", None)
                response = _response_from_error(exc, url)
                diagnostic = _attempt_diagnostic(
                    source,
                    method=method,
                    requested_url=url,
                    response=response,
                    semantic="FAIL",
                    error=f"{type(exc).__name__}: {status or 'request failed'}",
                )
                diagnostics.append(diagnostic)
                source_attempt_diagnostics.append(diagnostic)
                failures.append(f"{method}: {status or type(exc).__name__}")
                if method == "first_party_http" and is_pollofpolls:
                    live_host_unavailable = True
                    messages.append(
                        "first-party host unavailable; no further live requests "
                        f"this run ({status or type(exc).__name__})"
                    )
                continue

            try:
                _validate_source_payload(source, candidate_payload, response)
            except Exception as exc:
                diagnostic = _attempt_diagnostic(
                    source,
                    method=method,
                    requested_url=url,
                    response=response,
                    payload=candidate_payload,
                    semantic="FAIL",
                    error=f"{type(exc).__name__}: {_safe_validation_error(exc)}",
                )
                diagnostics.append(diagnostic)
                source_attempt_diagnostics.append(diagnostic)
                failures.append(
                    f"{method}: semantic validation failed ({type(exc).__name__}: "
                    f"{_safe_validation_error(exc)})"
                )
                if method == "first_party_http" and is_pollofpolls:
                    messages.append(
                        "first-party source response failed semantic validation; "
                        "trying its configured fallback"
                    )
                continue

            diagnostic = _attempt_diagnostic(
                source,
                method=method,
                requested_url=url,
                response=response,
                payload=candidate_payload,
                semantic="PASS",
            )
            diagnostics.append(diagnostic)
            source_attempt_diagnostics.append(diagnostic)
            payload = candidate_payload
            record = _record(source, payload, response, method, url, old_record)
            break

        if payload is None or record is None:
            retained = False
            if destination.exists() and old_record:
                previous_payload = destination.read_bytes()
                actual_hash = sha256_bytes(previous_payload)
                if actual_hash == old_record.get("sha256"):
                    previous_response = {
                        "http_status": old_record.get("http_status"),
                        "final_url": old_record.get("final_url") or old_record.get("source_url"),
                        "http_headers": old_record.get("http_headers") or {},
                    }
                    try:
                        _validate_source_payload(source, previous_payload, previous_response)
                    except Exception as exc:
                        diagnostic = _attempt_diagnostic(
                            source,
                            method="verified_previous_snapshot",
                            requested_url=str(old_record.get("source_url") or source.url),
                            response=previous_response,
                            payload=previous_payload,
                            semantic="FAIL",
                            error=f"{type(exc).__name__}: {_safe_validation_error(exc)}",
                        )
                        diagnostics.append(diagnostic)
                        raise AcquisitionError(
                            f"could not acquire {source.key}; previous raw file failed "
                            "semantic validation",
                            diagnostics=diagnostics,
                        ) from exc
                    retained = True
                    for diagnostic in source_attempt_diagnostics:
                        diagnostic["retained_previous"] = True
                    records[source.key] = old_record
                    source_outcomes[source.key] = "retained_verified_previous"
                    messages.append(
                        f"{source.key}: acquisition failed; retained verified raw file "
                        f"({'; '.join(failures)})"
                    )
            if retained:
                continue
            raise AcquisitionError(
                f"could not acquire {source.key}: {'; '.join(failures) or 'no method'}",
                diagnostics=diagnostics,
            )

        if not old_record or old_record.get("sha256") != record["sha256"]:
            # This is reached only after semantic validation passed.  A
            # malformed first-party body therefore cannot overwrite verified
            # raw content, even when it returned HTTP 200.
            _atomic_write(destination, payload)
            messages.append(
                f"{source.key}: saved {len(payload):,} bytes ({record['retrieval_method']})"
            )
            source_outcomes[source.key] = "saved"
        else:
            messages.append(f"{source.key}: unchanged ({record['sha256'][:12]})")
            source_outcomes[source.key] = "unchanged"
        records[source.key] = record

        # One request at a time; a modest delay is enough for these small assets.
        if index != len(SOURCES) - 1:
            time.sleep(0.75)

    manifest = {
        "schema_version": 1,
        "description": "Raw polling-source retrieval provenance; hashes cover exact payloads.",
        "sources": records,
        "source_outcomes": source_outcomes,
        "acquisition_diagnostics": diagnostics,
    }
    manifest_payload = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write(manifest_path, manifest_payload)
    return manifest, messages
