"""Poll and election observations for the SwedishPolls aggregate.

Every quantity attached to a poll is computed from information published no
later than that poll: imputed sample sizes use earlier polls only, and the
fieldwork overlap that discounts a rolling tracker is measured against
same-house polls published no later than it.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Iterable

from .config import (
    DEFAULT_SAMPLE_SIZE,
    HISTORICAL_RESULT_AVAILABILITY_LAG_DAYS,
    HISTORY_START_ELECTION,
    MAX_NAMED_PARTY_SUM,
    PARTIES,
)


class AggregateInputError(ValueError):
    """Raised when aggregate inputs are structurally unusable."""


@dataclass(frozen=True)
class Observation:
    """One reading of latent support on ``obs_date``, known from ``available``."""

    key: str
    kind: str  # "poll" or "election"
    obs_date: date
    available: date
    shares: tuple[float, ...]  # the eight named parties, percentage points
    house: str | None = None
    sample_size: float | None = None  # effective size after fieldwork overlap
    rounding_pp: float = 0.0

    def sort_key(self) -> tuple[date, date, str]:
        return (self.obs_date, self.available, self.key)


@dataclass
class PollDataset:
    polls: list[Observation]
    exclusions: dict[str, list[str]] = field(default_factory=dict)
    imputed_sample_size: list[str] = field(default_factory=list)
    approximate_fieldwork: list[str] = field(default_factory=list)
    overlap_discounted: list[str] = field(default_factory=list)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_date(value: str | None) -> date | None:
    if value is None or value.strip() == "":
        return None
    return date.fromisoformat(value.strip()[:10])


def _parse_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def _rounding_resolution(values: Iterable[float]) -> float:
    """Reporting resolution: 1 pp when every share is whole, else 0.1 pp."""

    return 1.0 if all(abs(v - round(v)) < 1e-9 for v in values) else 0.1


def election_dates_from(results_file: Path | str) -> list[date]:
    with Path(results_file).open(encoding="utf-8", newline="") as handle:
        return sorted({date.fromisoformat(row["election_date"]) for row in csv.DictReader(handle)})


def _wide_polls(path: Path) -> list[dict]:
    polls: dict[str, dict] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            poll = polls.setdefault(
                row["poll_id"],
                {
                    "poll_id": row["poll_id"],
                    "house": row["pollster"],
                    "publication": _parse_date(row["publication_date"]),
                    "start": _parse_date(row["interview_start"]),
                    "end": _parse_date(row["interview_end"]),
                    "approximate": row["collection_period_approximate"].strip().lower() == "true",
                    "sample_size": _parse_float(row["sample_size"]),
                    "shares": {},
                },
            )
            if row["party"] in PARTIES:
                poll["shares"][row["party"]] = _parse_float(row["support"])
    return list(polls.values())


def load_polls(path: Path | str, election_dates: Iterable[date]) -> PollDataset:
    """Eligible polls as observations, with every exclusion recorded by reason."""

    elections = sorted(election_dates)
    dataset = PollDataset(polls=[])
    reasons = dataset.exclusions

    def exclude(reason: str, poll_id: str) -> None:
        reasons.setdefault(reason, []).append(poll_id)

    candidates = []
    for poll in _wide_polls(Path(path)):
        pid = poll["poll_id"]
        if poll["publication"] is None or poll["start"] is None or poll["end"] is None:
            exclude("missing_dates", pid)
            continue
        if poll["end"] < poll["start"] or poll["end"] > poll["publication"]:
            exclude("inconsistent_dates", pid)
            continue
        if poll["end"] < HISTORY_START_ELECTION:
            exclude("before_history_start", pid)
            continue
        if any(poll["shares"].get(p) is None for p in PARTIES):
            exclude("party_not_reported", pid)
            continue
        if sum(poll["shares"][p] for p in PARTIES) > MAX_NAMED_PARTY_SUM:
            exclude("named_sum_above_100", pid)
            continue
        # Exit and election-day polls are superseded by the count itself.
        if any(poll["start"] <= e <= poll["end"] for e in elections):
            exclude("election_day_poll", pid)
            continue
        candidates.append(poll)

    # Chronological by availability, so each poll sees only its predecessors.
    candidates.sort(key=lambda p: (p["publication"], p["start"], p["end"], p["poll_id"]))

    seen_periods: dict[tuple[str, date, date], str] = {}
    house_fieldwork: dict[str, list[tuple[date, date]]] = {}
    reported_sizes: list[tuple[date, str, float]] = []

    for poll in candidates:
        pid, house = poll["poll_id"], poll["house"]
        period = (house, poll["start"], poll["end"])
        if period in seen_periods:
            exclude("duplicate_house_period", pid)
            continue
        seen_periods[period] = pid

        size = poll["sample_size"]
        if size is None or size <= 0:
            earlier_house = [n for d, h, n in reported_sizes if h == house and d < poll["publication"]]
            earlier_any = [n for d, _, n in reported_sizes if d < poll["publication"]]
            size = float(median(earlier_house or earlier_any or [DEFAULT_SAMPLE_SIZE]))
            dataset.imputed_sample_size.append(pid)
        else:
            reported_sizes.append((poll["publication"], house, size))

        days = (poll["end"] - poll["start"]).days + 1
        covered = set()
        for start, end in house_fieldwork.get(house, []):
            lo, hi = max(start, poll["start"]), min(end, poll["end"])
            covered.update(lo + timedelta(days=i) for i in range((hi - lo).days + 1))
        new_days = max(days - len(covered), 1)
        if new_days < days:
            dataset.overlap_discounted.append(pid)
        house_fieldwork.setdefault(house, []).append((poll["start"], poll["end"]))
        if poll["approximate"]:
            dataset.approximate_fieldwork.append(pid)

        values = [poll["shares"][p] for p in PARTIES]
        midpoint = poll["start"] + timedelta(days=(poll["end"] - poll["start"]).days // 2)
        dataset.polls.append(
            Observation(
                key=pid,
                kind="poll",
                obs_date=midpoint,
                available=poll["publication"],
                shares=tuple(values),
                house=house,
                sample_size=size * new_days / days,
                rounding_pp=_rounding_resolution(values),
            )
        )

    return dataset


def load_elections(
    results_file: Path | str,
    certified_manifests: Iterable[Path | str] = (),
) -> list[Observation]:
    """Election results with explicit availability dates.

    A certified manifest is dated by its retrieval; other elections use the
    conservative historical lag. The two sources must not describe the same
    election.
    """

    by_date: dict[date, dict[str, float]] = {}
    with Path(results_file).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["party"] in PARTIES:
                by_date.setdefault(date.fromisoformat(row["election_date"]), {})[row["party"]] = float(
                    row["vote_share"]
                )

    observations = []
    for day, shares in sorted(by_date.items()):
        if day < HISTORY_START_ELECTION:
            continue
        if set(shares) != set(PARTIES):
            raise AggregateInputError(f"Election {day} lacks a named party")
        observations.append(
            Observation(
                key=f"election-{day.isoformat()}",
                kind="election",
                obs_date=day,
                available=day + timedelta(days=HISTORICAL_RESULT_AVAILABILITY_LAG_DAYS),
                shares=tuple(shares[p] for p in PARTIES),
            )
        )

    for manifest_path in certified_manifests:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if manifest.get("certification_status") != "FINAL_CERTIFIED":
            raise AggregateInputError(f"{manifest_path} is not a final certified result")
        day = date.fromisoformat(manifest["election_date"])
        if day in by_date:
            raise AggregateInputError(f"Election {day} is supplied twice")
        retrieved = datetime.fromisoformat(manifest["retrieved_at_utc"].replace("Z", "+00:00")).date()
        parties = manifest["parties"]
        observations.append(
            Observation(
                key=f"election-{day.isoformat()}",
                kind="election",
                obs_date=day,
                available=retrieved,
                shares=tuple(float(parties[p]["vote_share_percentage_points"]) for p in PARTIES),
            )
        )

    observations.sort(key=Observation.sort_key)
    return observations
