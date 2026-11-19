"""The website layout of the riksdag outputs (``files/riksdag/`` on the site).

    uv run python -m scripts.riksdag_votes.site --site-dir /tmp/site

Layout, relative to the site directory::

    index.json                    role riksdag_votes_index
    <YYYY-ZZ>/votes.json          copy of data/processed/riksdag_votes/<YYYY-ZZ>/votes.json
    <YYYY-ZZ>/summary.json        copy of .../<YYYY-ZZ>/summary.json
    term-<YYYY-ZZ>/summary.json   copy of .../term-<YYYY-ZZ>/summary.json

The copies are byte-identical to the committed outputs. ``index.json`` lists
every session and term with the relative path, size and SHA-256 of each file,
its vote counts and latest vote date. Its ``generated_at_utc`` is the newest
``provenance.generated_at_utc`` among the listed files -- when the published
content last changed -- not the time of the export, so exporting unchanged
outputs again produces identical bytes and the website sees no change.

The export only writes inside the site directory and never deletes anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Sequence

from .constants import (
    DEFAULT_OUTPUT_DIR,
    INDEX_FILENAME,
    PROCESSED_DIR_RELATIVE,
    ROLE_INDEX,
    ROLE_SESSION_VOTES,
    ROLE_SUMMARY,
    SCHEMA_VERSION,
)

_SESSION_DIR = re.compile(r"^\d{4}-\d{2}$")
_TERM_DIR = re.compile(r"^term-\d{4}-\d{2}$")


class SiteExportError(ValueError):
    pass


def _dumps(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def _load(path: Path, role: str) -> tuple[bytes, dict]:
    data = path.read_bytes()
    payload = json.loads(data.decode("utf-8"))
    if payload.get("role") != role:
        raise SiteExportError(f"{path}: role {payload.get('role')!r}, expected {role!r}")
    return data, payload


def _file_record(relative: str, data: bytes) -> dict:
    return {"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _write_if_different(path: Path, data: bytes) -> int:
    if path.is_symlink():
        raise SiteExportError(f"refusing to write through a symlink: {path}")
    if not (path.exists() and path.read_bytes() == data):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    return len(data)


def _max_or_none(values) -> str | None:
    values = [value for value in values if value]
    return max(values) if values else None


def build_site(output_dir: Path) -> tuple[dict[str, bytes], dict]:
    """``({relative path: bytes}, index payload)`` for everything in ``output_dir``."""
    output_dir = Path(output_dir)
    files: dict[str, bytes] = {}
    sessions = []
    session_latest: dict[str, str | None] = {}
    session_counts: dict[str, int] = {}
    stamps = []
    for directory in sorted(p for p in output_dir.iterdir() if p.is_dir() and _SESSION_DIR.match(p.name)):
        votes_path, summary_path = directory / "votes.json", directory / "summary.json"
        if not (votes_path.is_file() and summary_path.is_file()):
            raise SiteExportError(f"{directory} lacks votes.json or summary.json")
        votes_bytes, votes = _load(votes_path, ROLE_SESSION_VOTES)
        summary_bytes, summary = _load(summary_path, ROLE_SUMMARY)
        rm = votes["rm"]
        if rm.replace("/", "-") != directory.name or summary.get("scope", {}).get("label") != rm:
            raise SiteExportError(f"{directory}: rm {rm!r} does not match the directory or its summary")
        dates = [vote.get("date") for vote in votes["votes"]]
        session_latest[rm] = _max_or_none(dates)
        session_counts[rm] = len(votes["votes"])
        generated = _max_or_none(
            [votes.get("provenance", {}).get("generated_at_utc"), summary.get("provenance", {}).get("generated_at_utc")]
        )
        stamps.append(generated)
        votes_rel, summary_rel = f"{directory.name}/votes.json", f"{directory.name}/summary.json"
        files[votes_rel], files[summary_rel] = votes_bytes, summary_bytes
        sessions.append(
            {
                "rm": rm,
                "slug": directory.name,
                "n_votes": len(votes["votes"]),
                "n_votes_analysis": summary["counts"]["votes_analysis"],
                "n_votes_titled": sum(vote.get("title") is not None for vote in votes["votes"]),
                "first_vote_date": min((d for d in dates if d), default=None),
                "latest_vote_date": session_latest[rm],
                "generated_at_utc": generated,
                "files": {"votes": _file_record(votes_rel, votes_bytes), "summary": _file_record(summary_rel, summary_bytes)},
            }
        )
    terms = []
    for directory in sorted(p for p in output_dir.iterdir() if p.is_dir() and _TERM_DIR.match(p.name)):
        summary_path = directory / "summary.json"
        if not summary_path.is_file():
            raise SiteExportError(f"{directory} lacks summary.json")
        summary_bytes, summary = _load(summary_path, ROLE_SUMMARY)
        scope = summary.get("scope", {})
        if f"term-{scope.get('label')}" != directory.name:
            raise SiteExportError(f"{directory}: scope label {scope.get('label')!r} does not match")
        generated = summary.get("provenance", {}).get("generated_at_utc")
        stamps.append(generated)
        summary_rel = f"{directory.name}/summary.json"
        files[summary_rel] = summary_bytes
        terms.append(
            {
                "label": scope["label"],
                "sessions": scope.get("sessions", []),
                "term_complete": scope.get("term_complete"),
                "n_votes": summary["counts"]["votes_all"],
                "n_votes_analysis": summary["counts"]["votes_analysis"],
                "latest_vote_date": _max_or_none(session_latest.get(rm) for rm in scope.get("sessions", [])),
                "generated_at_utc": generated,
                "files": {"summary": _file_record(summary_rel, summary_bytes)},
            }
        )
    if not sessions:
        raise SiteExportError(f"no session outputs in {output_dir}")
    index = {
        "schema_version": SCHEMA_VERSION,
        "role": ROLE_INDEX,
        "generated_at_utc": _max_or_none(stamps),
        "latest_vote_date": _max_or_none(session_latest.values()),
        "source": {"repository": "edvinli/election-simulator", "path": PROCESSED_DIR_RELATIVE},
        "paths": "relative to this index",
        "sessions": sessions,
        "terms": terms,
    }
    return files, index


def export_site(output_dir: Path, site_dir: Path) -> dict[str, int]:
    """Write the layout into ``site_dir``; returns {path: bytes}."""
    files, index = build_site(output_dir)
    site_dir = Path(site_dir)
    written = {}
    for relative, data in sorted(files.items()):
        path = site_dir / relative
        written[str(path)] = _write_if_different(path, data)
    path = site_dir / INDEX_FILENAME
    written[str(path)] = _write_if_different(path, _dumps(index).encode("utf-8"))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--site-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    for path, size in sorted(export_site(args.output_dir, args.site_dir).items()):
        print(f"{size:>10}  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
