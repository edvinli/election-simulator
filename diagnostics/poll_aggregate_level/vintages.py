"""Genuinely available PoP values, reconstructed from committed polling snapshots.

The automation commits the processed polling folder only when its content
changes, so the last snapshot committed on or before a date is the PoP state
known on that date. Commit time is an upper bound on retrieval time; the
committed raw retrieval manifest records the retrieval itself.
"""

from __future__ import annotations

import csv
import io
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts.poll_aggregate.config import PARTIES, REPOSITORY_ROOT

TIMESERIES_PATH = "data/processed/pollofpolls/pollofpolls_timeseries.csv"
POLLING_DIR = "data/processed/pollofpolls"
STOCKHOLM = ZoneInfo("Europe/Stockholm")


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=True)
    return result.stdout


@dataclass(frozen=True)
class Snapshot:
    commit: str
    committed_at: datetime

    @property
    def known_on(self) -> date:
        """The Stockholm date by the end of which this snapshot was available."""
        return self.committed_at.astimezone(STOCKHOLM).date()


def snapshots(ref: str = "origin/main") -> list[Snapshot]:
    """Every committed state of the PoP timeseries on ``ref``, oldest first."""

    out = []
    for line in _git("log", "--format=%H %cI", ref, "--", TIMESERIES_PATH).splitlines():
        commit, stamp = line.split(" ", 1)
        out.append(Snapshot(commit, datetime.fromisoformat(stamp).astimezone(timezone.utc)))
    return sorted(out, key=lambda s: s.committed_at)


def snapshot_known_on(day: date, ref: str = "origin/main") -> Snapshot | None:
    """The latest snapshot committed by the end of ``day`` (Stockholm)."""

    known = [s for s in snapshots(ref) if s.known_on <= day]
    return known[-1] if known else None


def read_timeseries(commit: str) -> dict[date, dict[str, float]]:
    rows = {}
    for row in csv.DictReader(io.StringIO(_git("show", f"{commit}:{TIMESERIES_PATH}"))):
        if all(row.get(p) not in (None, "") for p in PARTIES):
            rows[date.fromisoformat(row["date"])] = {p: float(row[p]) for p in PARTIES}
    return rows


def materialize_polling_dir(commit: str, dest: Path) -> list[str]:
    """Write the snapshot's processed polling folder to ``dest``; returns file names."""

    dest.mkdir(parents=True, exist_ok=True)
    names = []
    for path in _git("ls-tree", "--name-only", f"{commit}:{POLLING_DIR}").splitlines():
        blob = subprocess.run(
            ["git", "show", f"{commit}:{POLLING_DIR}/{path}"], cwd=REPOSITORY_ROOT, capture_output=True, check=True
        ).stdout
        (dest / path).write_bytes(blob)
        names.append(path)
    return sorted(names)


def value_on_or_before(series: dict[date, dict[str, float]], day: date) -> tuple[date, dict[str, float]] | None:
    eligible = [d for d in series if d <= day]
    if not eligible:
        return None
    selected = max(eligible)
    return selected, series[selected]
