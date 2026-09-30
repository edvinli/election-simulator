"""A simple SwedishPolls-only poll-consensus baseline series.

For each day ``T``: take each pollster's latest poll (by fieldwork midpoint,
then publication) among cleaned SwedishPolls polls published by ``T`` whose
fieldwork midpoint lies in the 60 days up to ``T``, and average them weighted
by effective sample. No filter, no house effects, no election input. It is the
election-noise consensus rule (latest poll per pollster, sample-weighted)
applied daily, and serves as the benchmark v0.2 must beat or match.

    uv run python -m scripts.poll_aggregate.consensus_baseline
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

from .config import (
    DEFAULT_2026_RESULT_MANIFEST,
    DEFAULT_ELECTION_RESULTS_FILE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POLLS_FILE,
    OUTPUT_SHARE_DIGITS,
    PARTIES,
    SOURCE_URL,
    TIMESERIES_FILENAME,
)
from .build import TIMESERIES_FIELDS
from .data import Observation, load_elections, load_polls, sha256_file

BASELINE_VERSION = "SwedishPollsConsensusBaseline-v1"
WINDOW_DAYS = 60
FIRST_DATE = date(2010, 9, 26)  # the aggregates' first row
DEFAULT_BASELINE_DIR = DEFAULT_OUTPUT_DIR / "consensus_baseline"


def consensus_rows(polls: Sequence[Observation], first: date, last: date) -> list[dict]:
    ordered = sorted(polls, key=lambda o: (o.available, o.key))
    rows = []
    day = first
    while day <= last:
        latest: dict[str, Observation] = {}
        for o in ordered:
            if o.available > day:
                break
            if day - timedelta(days=WINDOW_DAYS) <= o.obs_date <= day:
                current = latest.get(o.house)
                if current is None or (o.obs_date, o.available, o.key) > (current.obs_date, current.available, current.key):
                    latest[o.house] = o
        if latest:
            weight = sum(o.sample_size for o in latest.values())
            shares = [sum(o.shares[i] * o.sample_size for o in latest.values()) / weight for i in range(len(PARTIES))]
            rows.append({
                "day": day, "shares": shares, "pollsters": len(latest),
                "information_date": max(o.available for o in latest.values()),
            })
        day += timedelta(days=1)
    return rows


def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMESERIES_FIELDS, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            record = {p: f"{round(v, OUTPUT_SHARE_DIGITS):.{OUTPUT_SHARE_DIGITS}f}" for p, v in zip(PARTIES, r["shares"])}
            record.update({
                "date": r["day"].isoformat(), "FI": "",
                "other": f"{round(100.0 - sum(r['shares']), OUTPUT_SHARE_DIGITS):.{OUTPUT_SHARE_DIGITS}f}",
                "source_extra_json": json.dumps({"aggregate_version": BASELINE_VERSION,
                                                 "information_date": r["information_date"].isoformat(),
                                                 "pollsters": r["pollsters"]},
                                                sort_keys=True, separators=(",", ":")),
                "source_url": SOURCE_URL, "retrieved_at": "",
            })
            writer.writerow(record)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--polls", type=Path, default=DEFAULT_POLLS_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    args = parser.parse_args(argv)
    elections = load_elections(DEFAULT_ELECTION_RESULTS_FILE, [DEFAULT_2026_RESULT_MANIFEST])
    polls = load_polls(args.polls, [e.obs_date for e in elections]).polls
    last = max(o.available for o in polls)
    rows = consensus_rows(polls, FIRST_DATE, last)
    write_rows(rows, args.output_dir / TIMESERIES_FILENAME)
    metadata = {
        "aggregate_version": BASELINE_VERSION, "window_days": WINDOW_DAYS,
        "polls": {"path": args.polls.name, "sha256": sha256_file(args.polls)},
        "rows": len(rows), "first_date": rows[0]["day"].isoformat(), "last_date": rows[-1]["day"].isoformat(),
        "min_pollsters": min(r["pollsters"] for r in rows),
        "days_below_three_pollsters": sum(r["pollsters"] < 3 for r in rows),
    }
    (args.output_dir / "consensus_baseline_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
