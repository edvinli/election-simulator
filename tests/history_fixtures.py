"""Frozen, self-consistent history fixtures for the publication tests.

The end-to-end publication tests used to roll their certified point into the
*committed* history artifact and the *committed* polling CSVs. Every daily
publication commit therefore moved the inputs underneath them, and the tests
failed for reasons that had nothing to do with the behaviour under test:

* a hardcoded forecast date fell behind the artifact's certified point, so the
  future projection spanned a day that already carried history (PR #17);
* a synthetic ``generated_at_utc`` fell behind the committed publication's
  generation id, so a generation-agreement guard refused the sync.

Both are the same defect: production data as a test fixture. This module
builds a history that is internally consistent at a frozen date, and truncates
the polling inputs to agree with it, so the publication tests depend on
nothing that a publication can change.

The frozen date is deliberately well before election day: the projection then
has a non-trivial horizon, and the reconstructed curve is continuous from the
dynamics cap, so the publication's curve backfill has no hole to simulate.

Not named ``test_*``, so unittest discovery does not collect it.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.forecast_history.contract import (
    DEFAULT_COALITIONS,
    HISTORY_PARTY_ORDER,
    HISTORY_SCHEMA_VERSION,
    build_groups_from_matrices,
    deterministic_history_sha256,
    validate_history_contract,
)
from scripts.forecast_history.generate import HISTORY_CAP_DATE
from scripts.forecast_history.party_contract import (
    build_parties_from_matrices,
    parties_view_metadata,
)

# Frozen inputs. These are constants, not derived from any committed artifact:
# that is the whole point of this module.
FROZEN_AS_OF = "2026-09-05"
FROZEN_ELECTION_DATE = "2026-09-13"
FROZEN_MODEL_COMMIT = "f" * 40
FROZEN_POLL_SHA256 = "e" * 64

# Generation ids sort lexicographically and begin with a UTC timestamp, so the
# frozen publication the site starts from is stamped before any timestamp the
# tests then publish with.
FROZEN_SITE_GENERATION = "20260905T000000Z-0000f1ed"

_VOTES = np.array(
    [
        [20.0, 5.0, 10.0, 5.0, 30.0, 10.0, 8.0, 12.0, 0.0],
        [18.0, 6.0, 11.0, 4.0, 32.0, 9.0, 7.0, 13.0, 0.0],
        [22.0, 4.0, 9.0, 6.0, 28.0, 11.0, 9.0, 11.0, 0.0],
        [19.0, 7.0, 12.0, 5.0, 31.0, 8.0, 6.0, 12.0, 0.0],
    ],
    dtype=float,
)
_SEATS = np.array(
    [
        [40, 20, 30, 20, 80, 50, 30, 79],
        [39, 21, 31, 19, 82, 48, 31, 78],
        [42, 18, 29, 22, 78, 52, 29, 79],
        [41, 19, 32, 21, 81, 49, 28, 78],
    ],
    dtype=np.int64,
)
_SAMPLES = int(_VOTES.shape[0])


def _point(day: date, election: date, *, provenance: str) -> dict[str, Any]:
    horizon = (election - day).days
    return {
        "date": day.isoformat(),
        "samples": _SAMPLES,
        "horizon_days": horizon,
        "dynamics_horizon_days": horizon,
        "provenance": provenance,
        "groups": build_groups_from_matrices(_VOTES, _SEATS),
        "parties": build_parties_from_matrices(_VOTES, _SEATS),
    }


def make_history_fixture(
    *,
    as_of: str = FROZEN_AS_OF,
    election_date: str = FROZEN_ELECTION_DATE,
    start: date = HISTORY_CAP_DATE,
    model_commit: str = FROZEN_MODEL_COMMIT,
    generation: str = FROZEN_SITE_GENERATION,
) -> dict[str, Any]:
    """A contract-valid history whose certified point sits on ``as_of``.

    The reconstructed curve runs daily from the dynamics cap so that
    ``missing_curve_dates`` finds no hole, and the Poll of Polls series covers
    exactly the same span so no observation can fall after the projection
    origin.
    """

    certified_day = date.fromisoformat(as_of)
    election = date.fromisoformat(election_date)
    if certified_day < start:
        raise ValueError("as_of must not precede the reconstructed curve start")

    series: list[dict[str, Any]] = []
    day = start
    while day < certified_day:
        series.append(_point(day, election, provenance="reconstructed_current_model"))
        day += timedelta(days=1)
    certified = _point(certified_day, election, provenance="current_production")
    certified.update(
        {
            "publication_generation": generation,
            "deterministic_payload_sha256": "d" * 64,
            "generated_at_utc": f"{as_of}T00:00:00+00:00",
            "source_git_commit": model_commit,
        }
    )
    series.append(certified)

    payload: dict[str, Any] = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "election_date": election_date,
        "model_commit": model_commit,
        "poll_source_sha256": FROZEN_POLL_SHA256,
        "party_order": list(HISTORY_PARTY_ORDER),
        "coalitions": {key: list(value) for key, value in DEFAULT_COALITIONS.items()},
        "series": series,
        "poll_of_polls": [
            {
                "date": (start + timedelta(days=offset)).isoformat(),
                "parties": {party: 12.5 for party in HISTORY_PARTY_ORDER},
            }
            for offset in range((certified_day - start).days + 1)
        ],
        "polls": [],
        "parties_view": parties_view_metadata(),
        "model": {
            "name": "ElectionSimulator",
            "history_samples": _SAMPLES,
            "seed": 12345,
            "production_samples": _SAMPLES,
        },
        "schedule": {"observation_count": len(series)},
        "poll_date_range": {"start_date": start.isoformat(), "end_date": as_of},
        "source_hashes": {
            "poll_source_sha256": FROZEN_POLL_SHA256,
            "timeseries_source_sha256": "c" * 64,
        },
        "archive_diagnostics": {},
        "source_worktree_clean": True,
        "generated_at_utc": f"{as_of}T00:00:00+00:00",
    }
    payload["deterministic_content_sha256"] = deterministic_history_sha256(payload)
    validate_history_contract(payload)
    return payload


def freeze_poll_inputs(pollofpolls_dir: Path, *, as_of: str = FROZEN_AS_OF) -> None:
    """Drop polling rows dated after ``as_of`` from a fixture's inputs.

    Truncation rather than synthesis: the real rows keep their real shape and
    values, so the tests that mutate one normalized support value or assert a
    real schema keep working. What it removes is the drift -- the file no
    longer changes when a later poll is published.
    """

    _truncate(pollofpolls_dir / "swedishpolls_individual_polls.csv", "publication_date", as_of)
    _truncate(pollofpolls_dir / "individual_polls.csv", "publication_date", as_of)
    _truncate(pollofpolls_dir / "pollofpolls_timeseries.csv", "date", as_of)


def freeze_archive_inputs(archive_dir: Path, *, as_of: str = FROZEN_AS_OF) -> None:
    """Drop archived generations published after ``as_of`` from a fixture copy.

    An archive cannot run ahead of the forecast being published: the newest
    snapshot in it *is* the generation under publication. The committed
    archive does run ahead of the frozen date, though, and ``build_history``
    derives "which date is the official one" from the newest date it can see
    anywhere -- the requested dates, the existing points and the archive. A
    fixture that publishes at the frozen date against the live archive
    therefore hands the reconstruction a later official date than the
    certified point, and the result carries two ``current_production`` points.

    Only ``index.json`` is pruned, because that is the only thing
    ``_load_archive_records`` reads; the generation directories are left alone
    so tests that address one by id still find it.
    """

    index_path = Path(archive_dir) / "index.json"
    if not index_path.is_file():
        return
    with index_path.open(encoding="utf-8") as handle:
        index = json.load(handle)
    snapshots = index.get("snapshots")
    if not isinstance(snapshots, list):
        return
    index["snapshots"] = [
        entry
        for entry in snapshots
        if not (
            isinstance(entry, dict)
            and str(entry.get("as_of") or entry.get("snapshot_date") or "") > as_of
        )
    ]
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, ensure_ascii=False, indent=2)


def _truncate(path: Path, column: str, as_of: str) -> None:
    if not path.is_file():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if column not in fields:
            return
        kept = [row for row in reader if not (row.get(column) or "") > as_of]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
