"""Strict loader for final certified Valmyndigheten result evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


PARTY_ORDER = ("M", "L", "C", "KD", "S", "V", "MP", "SD")
RESULT_SCHEMA_VERSION = "1.0"
ELECTION_DATE = "2026-09-13"
# This is the official Valmyndigheten result host used by the preregistered
# fixtures and protocol.  Requiring the exact host prevents a caller from
# satisfying the weaker ``https://`` check with an attacker-controlled URL.
OFFICIAL_RESULT_HOST = "resultat.val.se"


class OfficialResultError(ValueError):
    """Raised when result evidence is preliminary, incomplete, or inconsistent."""


@dataclass(frozen=True)
class OfficialResult:
    manifest_path: Path
    manifest_sha256: str
    raw_path: Path
    raw_sha256: str
    official_source_url: str
    retrieved_at_utc: str
    valid_national_votes: int
    vote_shares: dict[str, float]
    votes: dict[str, int]
    seats: dict[str, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_result(manifest_path: Path | str) -> OfficialResult:
    supplied_path = Path(manifest_path)
    if supplied_path.is_symlink():
        raise OfficialResultError("Official-result manifest must be a regular file, not a symlink")
    path = supplied_path.resolve()
    try:
        raw_manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OfficialResultError(f"Cannot read official-result manifest: {exc}") from exc
    if not isinstance(raw_manifest, dict) or raw_manifest.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise OfficialResultError("Unsupported official-result manifest schema")
    if raw_manifest.get("authority") != "Valmyndigheten":
        raise OfficialResultError("Final result authority must be Valmyndigheten")
    if raw_manifest.get("certification_status") != "FINAL_CERTIFIED":
        raise OfficialResultError("Preliminary or unverified election results cannot be scored")
    if raw_manifest.get("election_date") != ELECTION_DATE:
        raise OfficialResultError(
            f"Official result must identify the 2026 election date {ELECTION_DATE}"
        )
    source_url = raw_manifest.get("official_source_url")
    if not isinstance(source_url, str):
        raise OfficialResultError("An HTTPS Valmyndigheten source URL is required")
    parsed_source = urlparse(source_url)
    try:
        source_port = parsed_source.port
    except ValueError as exc:
        raise OfficialResultError("Official result URL has an invalid port") from exc
    if (
        parsed_source.scheme.lower() != "https"
        or parsed_source.hostname is None
        or parsed_source.hostname.lower() != OFFICIAL_RESULT_HOST
        or parsed_source.username is not None
        or parsed_source.password is not None
        or source_port not in {None, 443}
    ):
        raise OfficialResultError(
            f"Official result URL must use the exact Valmyndigheten host {OFFICIAL_RESULT_HOST}"
        )
    retrieved = raw_manifest.get("retrieved_at_utc")
    if not isinstance(retrieved, str):
        raise OfficialResultError("Result retrieval timestamp is required")
    try:
        timestamp = datetime.fromisoformat(retrieved.replace("Z", "+00:00"))
    except ValueError as exc:
        raise OfficialResultError("Result retrieval timestamp is invalid") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None or timestamp.utcoffset().total_seconds() != 0:
        raise OfficialResultError("Result retrieval timestamp must be UTC")
    raw_relative = raw_manifest.get("raw_path")
    if not isinstance(raw_relative, str):
        raise OfficialResultError("Result raw_path is required")
    candidate = Path(raw_relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise OfficialResultError("Result raw_path must be a safe path relative to the manifest")
    candidate_path = path.parent / candidate
    if candidate_path.is_symlink():
        raise OfficialResultError("Raw official result artifact must not be a symlink")
    raw_path = candidate_path.resolve()
    if raw_path.parent != path.parent and path.parent not in raw_path.parents:
        raise OfficialResultError("Result raw_path escapes the evidence directory")
    if not raw_path.is_file() or raw_path.is_symlink():
        raise OfficialResultError("Raw official result artifact is missing or unsafe")
    expected_raw_hash = raw_manifest.get("raw_sha256")
    actual_raw_hash = _sha256(raw_path)
    if expected_raw_hash != actual_raw_hash:
        raise OfficialResultError("Raw official result SHA-256 mismatch")
    denominator = raw_manifest.get("valid_national_votes")
    if not isinstance(denominator, int) or isinstance(denominator, bool) or denominator <= 0:
        raise OfficialResultError("Positive integer valid_national_votes is required")
    parties = raw_manifest.get("parties")
    if not isinstance(parties, dict) or not set(PARTY_ORDER).issubset(parties):
        raise OfficialResultError(f"Official result must contain at least {list(PARTY_ORDER)}")
    vote_shares: dict[str, float] = {}
    votes: dict[str, int] = {}
    seats: dict[str, int] = {}
    for party in PARTY_ORDER:
        row = parties.get(party)
        if not isinstance(row, dict):
            raise OfficialResultError(f"Official result for {party} must be an object")
        vote_count = row.get("votes")
        share = row.get("vote_share_percentage_points")
        seat_count = row.get("seats")
        if not isinstance(vote_count, int) or isinstance(vote_count, bool) or vote_count < 0:
            raise OfficialResultError(f"Official vote count for {party} is invalid")
        if not isinstance(share, (int, float)) or isinstance(share, bool):
            raise OfficialResultError(f"Official vote share for {party} is invalid")
        if not math.isfinite(float(share)) or not 0.0 <= float(share) <= 100.0:
            raise OfficialResultError(f"Official vote share for {party} is invalid")
        if not isinstance(seat_count, int) or isinstance(seat_count, bool) or seat_count < 0:
            raise OfficialResultError(f"Final seat count for {party} is invalid")
        exact_share = 100.0 * vote_count / denominator
        if abs(float(share) - exact_share) > 5e-7:
            raise OfficialResultError(
                f"{party} share is not on the declared national valid-vote denominator"
            )
        votes[party] = vote_count
        vote_shares[party] = float(share)
        seats[party] = seat_count
    if sum(votes.values()) > denominator:
        raise OfficialResultError("Eight-party votes exceed the national valid-vote denominator")
    # Deliberately do not require or force the eight selected parties to sum to
    # 100%; votes for other parties remain in the official denominator.
    return OfficialResult(
        manifest_path=path,
        manifest_sha256=_sha256(path),
        raw_path=raw_path,
        raw_sha256=actual_raw_hash,
        official_source_url=source_url,
        retrieved_at_utc=timestamp.isoformat().replace("+00:00", "Z"),
        valid_national_votes=denominator,
        vote_shares=vote_shares,
        votes=votes,
        seats=seats,
    )
