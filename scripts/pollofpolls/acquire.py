"""Network acquisition with provenance and an upstream-host circuit breaker."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import SOURCES, Source


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
    headers = response.get("http_headers", {})
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
        "raw_filename": source.raw_filename,
        "sha256": digest,
        "bytes": len(payload),
    }


def acquire_all(
    raw_dir: Path,
    *,
    offline: bool = False,
    allow_archive_fallback: bool = True,
    timeout: float = 45.0,
) -> tuple[dict[str, Any], list[str]]:
    """Acquire configured sources, stopping live host requests after one block.

    A 455 response, TLS certificate error, or other connection-level failure
    opens a circuit breaker for the Pollofpolls host.  This prevents ten nearly
    identical requests to a host that has already refused normal public access.
    """

    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "retrieval_manifest.json"
    old_manifest = _load_manifest(manifest_path)
    old_sources = old_manifest.get("sources", {})
    records: dict[str, Any] = {}
    messages: list[str] = []
    live_host_unavailable = False

    for index, source in enumerate(SOURCES):
        destination = raw_dir / source.raw_filename
        old_record = old_sources.get(source.key)

        if offline:
            if not destination.exists() or not old_record:
                raise AcquisitionError(
                    f"offline mode requires {destination} and a manifest record"
                )
            actual_hash = sha256_bytes(destination.read_bytes())
            if actual_hash != old_record.get("sha256"):
                raise AcquisitionError(
                    f"raw file hash differs from manifest: {destination}"
                )
            records[source.key] = old_record
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
        for method, url in attempts:
            try:
                payload, response = _read_url(url, timeout)
                record = _record(source, payload, response, method, url, old_record)
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                status = getattr(exc, "code", None)
                failures.append(f"{method}: {status or type(exc).__name__}")
                if method == "first_party_http" and is_pollofpolls:
                    live_host_unavailable = True
                    messages.append(
                        "first-party host unavailable; no further live requests "
                        f"this run ({status or type(exc).__name__})"
                    )
                continue

        if payload is None or record is None:
            if destination.exists() and old_record:
                actual_hash = sha256_bytes(destination.read_bytes())
                if actual_hash == old_record.get("sha256"):
                    records[source.key] = old_record
                    messages.append(
                        f"{source.key}: acquisition failed; retained verified raw file "
                        f"({'; '.join(failures)})"
                    )
                    continue
            raise AcquisitionError(
                f"could not acquire {source.key}: {'; '.join(failures) or 'no method'}"
            )

        if not old_record or old_record.get("sha256") != record["sha256"]:
            _atomic_write(destination, payload)
            messages.append(
                f"{source.key}: saved {len(payload):,} bytes ({record['retrieval_method']})"
            )
        else:
            messages.append(f"{source.key}: unchanged ({record['sha256'][:12]})")
        records[source.key] = record

        # One request at a time; a modest delay is enough for these small assets.
        if index != len(SOURCES) - 1:
            time.sleep(0.75)

    manifest = {
        "schema_version": 1,
        "description": "Raw polling-source retrieval provenance; hashes cover exact payloads.",
        "sources": records,
    }
    manifest_payload = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write(manifest_path, manifest_payload)
    return manifest, messages
