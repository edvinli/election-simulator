"""Build the riksdag roll-call outputs.

    uv run python -m scripts.riksdag_votes.build --sessions 2022/23 2023/24 2024/25 2025/26 2026/27

Writes, under ``data/processed/riksdag_votes/``:

* ``<YYYY-ZZ>/votes.json`` and ``<YYYY-ZZ>/summary.json`` per session;
* ``term-<YYYY-ZZ>/summary.json`` per term, only when the run includes every
  session of that term that has begun (so a partial run never overwrites a
  term summary with a partial one);
* ``manifest.json`` -- URL, retrieval time (UTC) and SHA-256 of every raw file.

Raw downloads are cached in ``data/raw/riksdag_votes/`` (git-ignored).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Sequence

from .constants import (
    DEFAULT_CACHE_DIR,
    DEFAULT_OUTPUT_DIR,
    MANIFEST_FILENAME,
    REPO_ROOT,
    TERMS,
)
from .fetch import RawStore, session_code, session_slug
from .ingest import SessionData, load_session
from .summarize import build_summary, build_votes_payload

LOGGER = logging.getLogger("scripts.riksdag_votes.build")

DEFAULT_SESSIONS = ("2022/23", "2023/24", "2024/25", "2025/26", "2026/27")


def dumps(payload: dict) -> str:
    """Deterministic JSON: sorted keys, compact separators, UTF-8, trailing newline."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def write_json(path: Path, payload: dict) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = dumps(payload)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return len(text.encode("utf-8"))


def code_commit() -> dict:
    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--", "scripts/riksdag_votes")
    return {"code_commit": commit, "code_dirty": bool(status) if status is not None else None}


def session_has_begun(rm: str, today: date) -> bool:
    return today >= date(int(rm[:4]), 9, 1)


def terms_to_write(sessions: Sequence[str], today: date) -> dict[str, list[str]]:
    selected = {}
    for label, term_sessions in TERMS.items():
        begun = [rm for rm in term_sessions if session_has_begun(rm, today)]
        if begun and all(rm in sessions for rm in begun):
            selected[label] = begun
    return selected


def session_notes(session: SessionData) -> list[str]:
    notes = [f"{session.rm}: {note}" for note in session.notes]
    if session.excluded:
        notes.append(f"{session.rm}: {len(session.excluded)} bulk-archive vote(s) excluded; see votes.json excluded_votes")
    if session.unknown_committee_prefixes:
        notes.append(f"{session.rm}: unknown committee designations {session.unknown_committee_prefixes}")
    return notes


def build(
    sessions: Sequence[str],
    store: RawStore,
    output_dir: Path,
    *,
    generated_at_utc: str,
    code: dict,
    today: date,
) -> dict[str, int]:
    loaded: dict[str, SessionData] = {}
    for rm in sessions:
        session_code(rm)
        loaded[rm] = load_session(rm, store)

    written: dict[str, int] = {}

    def provenance(source_files) -> dict:
        return dict(code, generated_at_utc=generated_at_utc, sources=store.provenance(source_files))

    for rm, session in loaded.items():
        slug = session_slug(rm)
        prov = provenance(session.source_files)
        path = output_dir / slug / "votes.json"
        written[str(path)] = write_json(path, build_votes_payload(session, prov))
        notes = session_notes(session)
        summary = build_summary(
            session.votes,
            scope={"kind": "session", "label": rm, "sessions": [rm]},
            provenance=prov,
            notes=notes,
        )
        path = output_dir / slug / "summary.json"
        written[str(path)] = write_json(path, summary)

    for label, term_sessions in terms_to_write(list(loaded), today).items():
        votes = [vote for rm in term_sessions for vote in loaded[rm].votes]
        files = set().union(*(loaded[rm].source_files for rm in term_sessions))
        notes = [note for rm in term_sessions for note in session_notes(loaded[rm])]
        summary = build_summary(
            votes,
            scope={
                "kind": "term",
                "label": label,
                "sessions": term_sessions,
                "term_complete": len(term_sessions) == len(TERMS[label]),
            },
            provenance=provenance(files),
            notes=notes,
        )
        path = output_dir / f"term-{label}" / "summary.json"
        written[str(path)] = write_json(path, summary)

    store.write_manifest([session_code(rm) for rm in loaded])
    written[str(store.manifest_path)] = store.manifest_path.stat().st_size
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions", nargs="+", default=list(DEFAULT_SESSIONS), help="riksmöten, e.g. 2025/26")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--refresh", action="store_true", help="re-download every raw file")
    parser.add_argument("--offline", action="store_true", help="use only cached files attested by the manifest")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    store = RawStore(
        cache_dir=args.cache_dir,
        manifest_path=args.output_dir / MANIFEST_FILENAME,
        refresh=args.refresh,
        offline=args.offline,
    )
    now = datetime.now(timezone.utc).replace(microsecond=0)
    written = build(
        args.sessions,
        store,
        args.output_dir,
        generated_at_utc=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        code=code_commit(),
        today=now.date(),
    )
    for path, size in sorted(written.items()):
        print(f"{size:>10}  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
