"""Mirror the certified election result into the website repository.

The website leads with the last certified Riksdag result once a publication
postdates it, and sets the current forecast against it. That file is built
here from the same strictly validated Valmyndigheten evidence the prospective
benchmark scores against, so the page and the benchmark cannot disagree about
what the result was.

Like :mod:`scripts.site_publisher.publisher` this performs no simulation and
no version-control operation. It is a lookup file outside the frozen
publication bundle: no forecast quantity is derived from it, and the page
ignores it unless it validates whole.
"""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
from typing import Any

from scripts.mandates.config import TOTAL_RIKSDAG_SEATS
from scripts.simulator.election_cycles import ordinary_election_date
from scripts.prospective_benchmark_2026.results import (
    PARTY_ORDER,
    OfficialResultError,
    load_official_result,
)

from .publisher import SITE_PUBLICATION_RELATIVE, SitePublishError, _write_json_atomic


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_MANIFEST = (
    REPO_ROOT / "data" / "raw" / "elections" / "val2026" / "official_result_manifest.json"
)

#: The website's own validation (``normalizeElectionResult``) pins these three
#: values. A change here is a change to a cross-repository contract.
SITE_RESULT_SCHEMA_VERSION = "1.0"
SITE_RESULT_ROLE = "official_election_result"
#: The same definition the history's party view declares: a share of all valid
#: national votes, other parties included in the denominator.
SITE_RESULT_VOTE_SHARE_DEFINITION = "national_vote_share"
SITE_RESULT_SHARE_DIGITS = 4


def site_result_relative(year: int) -> Path:
    return SITE_PUBLICATION_RELATIVE / "results" / f"{year}.json"


def _parse_turnout(notes: Any) -> float | None:
    """Valmyndigheten prints turnout as Swedish text, e.g. ``"84,89 %"``."""

    if not isinstance(notes, dict) or not isinstance(notes.get("turnout"), str):
        return None
    text = notes["turnout"].replace(" ", " ").replace("%", "").replace(",", ".").strip()
    try:
        value = float(text)
    except ValueError as exc:
        raise SitePublishError(f"Result turnout is not a percentage: {notes['turnout']!r}") from exc
    if not 0.0 < value <= 100.0:
        raise SitePublishError(f"Result turnout is out of range: {value}")
    return value


def build_site_election_result(manifest_path: Path | str = DEFAULT_RESULT_MANIFEST) -> dict[str, Any]:
    """The website's result payload, from certified Valmyndigheten evidence.

    Every number comes from :func:`load_official_result`, which checks the raw
    file's SHA-256, the exact authority host, FINAL_CERTIFIED status and that
    each share is on the national valid-vote denominator. The website adds one
    requirement of its own, checked here: the eight parties hold all 349 seats.
    """

    try:
        result = load_official_result(manifest_path)
    except OfficialResultError as exc:
        raise SitePublishError(f"Official result failed validation: {exc}") from exc
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    election_day = date.fromisoformat(manifest["election_date"])
    if election_day != ordinary_election_date(election_day.year):
        raise SitePublishError(
            f"{election_day} is not an ordinary Riksdag election day; "
            "the next election date cannot be derived")
    seat_total = sum(result.seats[party] for party in PARTY_ORDER)
    if seat_total != TOTAL_RIKSDAG_SEATS:
        raise SitePublishError(
            f"The eight parties hold {seat_total} seats, not {TOTAL_RIKSDAG_SEATS}")
    notes = manifest.get("notes") if isinstance(manifest.get("notes"), dict) else {}
    other_votes = result.valid_national_votes - sum(result.votes[party] for party in PARTY_ORDER)
    payload: dict[str, Any] = {
        "schema_version": SITE_RESULT_SCHEMA_VERSION,
        "role": SITE_RESULT_ROLE,
        "election_date": election_day.isoformat(),
        "next_election_date": ordinary_election_date(election_day.year + 4).isoformat(),
        "authority": manifest["authority"],
        "certification_status": manifest["certification_status"],
        "source_url": result.official_source_url,
        "source_title": notes.get("source_title"),
        "decision_protocol_url": notes.get("decision_protocol_url"),
        "retrieved_at_utc": result.retrieved_at_utc,
        "raw_sha256": result.raw_sha256,
        "vote_share_definition": SITE_RESULT_VOTE_SHARE_DEFINITION,
        "valid_national_votes": result.valid_national_votes,
        "turnout_pct": _parse_turnout(notes),
        "total_seats": TOTAL_RIKSDAG_SEATS,
        "party_order": list(PARTY_ORDER),
        "parties": {
            party: {
                "votes": result.votes[party],
                "vote_share_pct": round(result.vote_shares[party], SITE_RESULT_SHARE_DIGITS),
                "seats": result.seats[party],
            }
            for party in PARTY_ORDER
        },
        "other_parties": {
            "votes": other_votes,
            "vote_share_pct": round(
                100.0 * other_votes / result.valid_national_votes, SITE_RESULT_SHARE_DIGITS),
            "seats": 0,
        },
    }
    return payload


def sync_election_result_to_site(
    *,
    site_repo: Path | str,
    manifest_path: Path | str = DEFAULT_RESULT_MANIFEST,
) -> dict[str, Any]:
    """Build and atomically install ``results/<year>.json`` in the website.

    Equal bytes are left untouched, so a no-op sync creates no website commit.
    """

    site_root = Path(site_repo).resolve()
    if not site_root.is_dir():
        raise SitePublishError(f"--site-repo must be an existing directory: {site_root}")
    payload = build_site_election_result(manifest_path)
    year = int(payload["election_date"][:4])
    destination = site_root / site_result_relative(year)
    rendered = (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    changed = not destination.is_file() or destination.read_bytes() != rendered
    if changed:
        _write_json_atomic(destination, payload)
        if destination.read_bytes() != rendered:
            raise SitePublishError(f"Installed result does not match what was built: {destination}")
    return {
        "status": "SYNCED" if changed else "UNCHANGED",
        "destination": str(destination),
        "election_date": payload["election_date"],
        "raw_sha256": payload["raw_sha256"],
        "changed": changed,
        "committed": False,
        "pushed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Write the certified election result into a website repository. "
            "Runs no simulation and makes no Git commit."
        )
    )
    parser.add_argument(
        "--site-repo",
        type=Path,
        required=True,
        help="Path to the website repository checkout (required, never inferred)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_RESULT_MANIFEST,
        help="Official-result manifest to export",
    )
    args = parser.parse_args(argv)
    try:
        report = sync_election_result_to_site(site_repo=args.site_repo, manifest_path=args.manifest)
    except SitePublishError as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}, indent=2, ensure_ascii=False))
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
