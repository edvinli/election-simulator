"""Prospective shadow captures for replay protocol v3.

Each capture records, at one timestamp, the PoP value actually available and
the v0.2 value computable from the SwedishPolls data available then. Captures
are appended to a JSON-lines file and never rewritten. A failed or invalid
PoP fetch is recorded as missing: no value is filled in later from PoP's
revised history, and no capture is ever made for a past date.

    uv run python -m scripts.poll_aggregate.shadow capture

The v0.2 value is the consensus reading on the capture's Stockholm date from
the frozen 2026 segment (hyperparameters, reference election and consensus
weights read from the committed v0.2 metadata). That segment holds until the
2030 result is available, so the value equals what a full v0.2 build would
report for that date from the same polls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Sequence
from zoneinfo import ZoneInfo

from .config import (
    DEFAULT_2026_RESULT_MANIFEST,
    DEFAULT_ELECTION_RESULTS_FILE,
    METADATA_FILENAME,
    PARTIES,
    REPOSITORY_ROOT,
    SPEC_V02,
)
from .data import Observation, load_elections, load_polls
from .model import Hyperparameters, SupportFilter

SCHEMA_VERSION = "1.0"
PROTOCOL = "v3"
STOCKHOLM = ZoneInfo("Europe/Stockholm")
DEFAULT_CAPTURE_FILE = REPOSITORY_ROOT / "data" / "shadow" / "v3" / "captures.jsonl"
V02_METADATA = SPEC_V02.output_dir / METADATA_FILENAME
#: The frozen segment is valid until the next election; captures stop before it.
LAST_CAPTURE_DATE = date(2030, 9, 7)
FETCH_TIMEOUT_SECONDS = 60.0

Fetcher = Callable[[str], bytes]


def _default_fetch(url: str) -> bytes:
    from scripts.pollofpolls.acquire import _read_url

    payload, _ = _read_url(url, FETCH_TIMEOUT_SECONDS)
    return payload


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _error(exc: BaseException) -> str:
    # Class and a short message only; never a response body.
    return f"{type(exc).__name__}: {str(exc)[:200]}"


def pop_record(payload: bytes, day: date) -> dict:
    """The PoP value available on ``day``: its latest complete row dated on or before it."""

    from scripts.pollofpolls.normalize import parse_timeseries_payload

    rows, _ = parse_timeseries_payload(payload)
    complete = [r for r in rows if all(r.get(p) is not None for p in PARTIES) and r["date"] <= day.isoformat()]
    if not complete:
        raise ValueError(f"no complete PoP row on or before {day}")
    latest = max(complete, key=lambda r: r["date"])
    row_date = date.fromisoformat(latest["date"])
    return {
        "status": "ok",
        "payload_sha256": _sha256(payload),
        "row_date": row_date.isoformat(),
        "staleness_days": (day - row_date).days,
        "values": {p: float(latest[p]) for p in PARTIES},
    }


def swedishpolls_observations(polls_payload: bytes, sources_payload: bytes, elections) -> list[Observation]:
    """Clean the live SwedishPolls table with the aggregate's own rules."""

    from scripts.pollofpolls.normalize import parse_swedishpolls_payloads
    from scripts.pollofpolls.validate import SWEDISHPOLLS_FIELDS

    _, long_rows = parse_swedishpolls_payloads(polls_payload, sources_payload)
    with tempfile.TemporaryDirectory(prefix="shadow-v3-") as tmp:
        path = Path(tmp) / "swedishpolls_individual_polls.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SWEDISHPOLLS_FIELDS, extrasaction="ignore",
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(long_rows)
        return load_polls(path, [e.obs_date for e in elections]).polls


def frozen_v02_segment(metadata_file: Path = V02_METADATA) -> dict:
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata.get("aggregate_version") != SPEC_V02.version:
        raise ValueError(f"{metadata_file} is not {SPEC_V02.version}")
    segment = metadata["segments"][-1]
    return {
        "fit_date": date.fromisoformat(segment["hyperparameters_fit_date"]),
        "reference_election": segment["reference_election"],
        "params": Hyperparameters(**segment["hyperparameters"]),
        "weights": segment["consensus_weights"],
        "metadata_sha256": _sha256(metadata_file.read_bytes()),
    }


def v02_value(polls: Sequence[Observation], elections: Sequence[Observation], day: date,
              segment: dict) -> dict:
    """v0.2's consensus reading on ``day`` from observations available by then."""

    if not segment["fit_date"] <= day <= LAST_CAPTURE_DATE:
        raise ValueError(f"{day} is outside the frozen v0.2 segment starting {segment['fit_date']}")
    by_key = {e.key: e for e in elections}
    anchor = min(elections, key=lambda e: e.obs_date)
    houses = sorted({o.house for o in polls} | set(segment["weights"]))
    known = sorted((o for o in [*polls, *elections] if o.available <= day), key=Observation.sort_key)
    model = SupportFilter(houses, by_key[segment["reference_election"]].shares, segment["params"])
    state = model.run(model.initial_state(anchor.obs_date, anchor.shares), known)
    mean, cov = model.consensus_reading_at(state, day, segment["weights"])
    known_polls = [o for o in known if o.kind == "poll"]
    return {
        "status": "ok",
        "values": {p: round(float(v), 6) for p, v in zip(PARTIES, mean)},
        "sd": {p: round(float(v), 6) for p, v in zip(PARTIES, cov.diagonal() ** 0.5)},
        "polls_known": len(known_polls),
        "latest_poll_publication": max(o.available for o in known_polls).isoformat() if known_polls else None,
    }


def _code_commit() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPOSITORY_ROOT,
                            capture_output=True, text=True)
    return {"commit": head.stdout.strip() or None, "tracked_changes": bool(status.stdout.strip())}


def capture(now: datetime | None = None, fetch: Fetcher = _default_fetch) -> dict:
    """One capture record. Each source fails independently and is then missing."""

    from scripts.pollofpolls.config import SWEDISHPOLLS_CSV, SWEDISHPOLLS_SOURCES_CSV, TIMESERIES_CSV

    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    day = now.astimezone(STOCKHOLM).date()
    record: dict = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "captured_at_utc": now.isoformat(timespec="seconds"),
        "stockholm_date": day.isoformat(),
        "code": _code_commit(),
    }
    try:
        record["pop"] = {"url": TIMESERIES_CSV, **pop_record(fetch(TIMESERIES_CSV), day)}
    except Exception as exc:  # noqa: BLE001 - any failure is a missing value, never a fill
        record["pop"] = {"url": TIMESERIES_CSV, "status": "missing", "error": _error(exc)}
    try:
        segment = frozen_v02_segment()
        elections = load_elections(DEFAULT_ELECTION_RESULTS_FILE, [DEFAULT_2026_RESULT_MANIFEST])
        polls_payload = fetch(SWEDISHPOLLS_CSV)
        sources_payload = fetch(SWEDISHPOLLS_SOURCES_CSV)
        polls = swedishpolls_observations(polls_payload, sources_payload, elections)
        record["v02"] = {
            "aggregate_version": SPEC_V02.version,
            "metadata_sha256": segment["metadata_sha256"],
            "polls_payload_sha256": _sha256(polls_payload),
            "sources_payload_sha256": _sha256(sources_payload),
            **v02_value(polls, elections, day, segment),
        }
    except Exception as exc:  # noqa: BLE001
        record["v02"] = {"aggregate_version": SPEC_V02.version, "status": "missing", "error": _error(exc)}
    return record


def append_capture(record: dict, path: Path = DEFAULT_CAPTURE_FILE) -> None:
    """Append one line durably; existing lines are never touched."""

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["capture"])
    parser.add_argument("--output", type=Path, default=DEFAULT_CAPTURE_FILE)
    args = parser.parse_args(argv)
    record = capture()
    append_capture(record, args.output)
    print(json.dumps({"stockholm_date": record["stockholm_date"], "pop": record["pop"]["status"],
                      "v02": record["v02"]["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
