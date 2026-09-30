"""Scoring rules for replay protocol v3 (prospective shadow evaluation).

These rules are frozen with ``docs/poll_aggregate_replay_protocol_v3.md``
before any target poll exists. They compare how well the real-time PoP value
and the v0.2 value, as captured each day, predict the polls fielded after that
day. This evaluates the opinion estimate only, not the election forecast.

    uv run python -m scripts.poll_aggregate.shadow_score --targets <processed SwedishPolls csv>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .config import DEFAULT_2026_RESULT_MANIFEST, DEFAULT_ELECTION_RESULTS_FILE, PARTIES
from .data import Observation, load_elections, load_polls, sha256_file
from .shadow import DEFAULT_CAPTURE_FILE

SOURCES: tuple[str, ...] = ("pop", "v02")
TARGET_MIN_OFFSET_DAYS = 7
TARGET_MAX_OFFSET_DAYS = 14
TARGET_PUBLICATION_DEADLINE_DAYS = 28
MIN_TARGET_POLLSTERS = 3
BOOTSTRAP_BLOCK_DAYS = 14
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260930
FIRST_ANALYSIS_MIN_WEEKS = 26
FIRST_ANALYSIS_MIN_PAIRED_DAYS = 100


def forecasts_by_day(captures: Iterable[dict]) -> dict[str, dict[date, np.ndarray]]:
    """Each source's value per Stockholm date: its latest successful capture that day.

    A source with no successful capture on a date is missing for that date;
    no value is ever carried from another date or another capture file.
    """

    best: dict[str, dict[date, tuple[datetime, np.ndarray]]] = {s: {} for s in SOURCES}
    for record in captures:
        day = date.fromisoformat(record["stockholm_date"])
        stamp = datetime.fromisoformat(record["captured_at_utc"])
        for source in SOURCES:
            entry = record.get(source) or {}
            if entry.get("status") != "ok":
                continue
            values = np.array([float(entry["values"][p]) for p in PARTIES])
            current = best[source].get(day)
            if current is None or stamp > current[0]:
                best[source][day] = (stamp, values)
    return {s: {d: v for d, (_, v) in days.items()} for s, days in best.items()}


@dataclass(frozen=True)
class Target:
    day: date
    shares: np.ndarray              # sample-weighted over pollsters
    equal_weight_shares: np.ndarray  # each pollster counts once
    pollsters: int
    polls: int


def target_for(day: date, polls: Sequence[Observation]) -> Target | None:
    """The polls fielded after ``day``, collapsed to one reading per pollster.

    Eligible: fieldwork starting after ``day`` (so unknown on ``day``), a
    fieldwork midpoint 7–14 days after it, and published within 28 days of
    it. A pollster's polls are averaged by effective sample (rolling trackers
    already count only new fieldwork days), so a pollster that publishes
    often is one reading, weighted by its sample. Fewer than three pollsters:
    no target.
    """

    lo, hi = day + timedelta(days=TARGET_MIN_OFFSET_DAYS), day + timedelta(days=TARGET_MAX_OFFSET_DAYS)
    deadline = day + timedelta(days=TARGET_PUBLICATION_DEADLINE_DAYS)
    by_house: dict[str, list[Observation]] = {}
    for o in polls:
        if o.fieldwork_start is None or o.fieldwork_start <= day:
            continue
        if lo <= o.obs_date <= hi and o.available <= deadline:
            by_house.setdefault(o.house, []).append(o)
    if len(by_house) < MIN_TARGET_POLLSTERS:
        return None
    readings, weights = [], []
    for house_polls in by_house.values():
        n = np.array([o.sample_size for o in house_polls])
        shares = np.array([o.shares for o in house_polls])
        readings.append((n[:, None] * shares).sum(axis=0) / n.sum())
        weights.append(n.sum())
    readings, weights = np.array(readings), np.array(weights)
    return Target(
        day,
        (weights[:, None] * readings).sum(axis=0) / weights.sum(),
        readings.mean(axis=0),
        len(by_house),
        sum(len(v) for v in by_house.values()),
    )


def block_bootstrap_ci(values: np.ndarray, block: int = BOOTSTRAP_BLOCK_DAYS,
                       resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """95 % moving-block bootstrap interval for the mean of a daily series."""

    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    block = min(block, n)
    rng = np.random.default_rng(seed)
    starts_available = n - block + 1
    blocks_needed = int(np.ceil(n / block))
    means = np.empty(resamples)
    for i in range(resamples):
        starts = rng.integers(0, starts_available, blocks_needed)
        sample = np.concatenate([values[s : s + block] for s in starts])[:n]
        means[i] = sample.mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def score(captures: Sequence[dict], polls: Sequence[Observation], scored_through: date) -> dict:
    """The protocol-v3 analysis. Days whose publication deadline is after
    ``scored_through`` are not yet scorable and are left out."""

    forecasts = forecasts_by_day(captures)
    capture_days = sorted({date.fromisoformat(r["stockholm_date"]) for r in captures})
    rows = []
    for day in capture_days:
        if day + timedelta(days=TARGET_PUBLICATION_DEADLINE_DAYS) > scored_through:
            continue
        target = target_for(day, polls)
        row = {"day": day.isoformat(), "target": target is not None,
               **{f"{s}_available": day in forecasts[s] for s in SOURCES}}
        if target is not None:
            row.update({"target_pollsters": target.pollsters, "target_polls": target.polls})
            for s in SOURCES:
                if day in forecasts[s]:
                    err = forecasts[s][day] - target.shares
                    row[f"{s}_error"] = err.tolist()
                    row[f"{s}_mae"] = float(np.abs(err).mean())
                    row[f"{s}_mae_equal_weight"] = float(np.abs(forecasts[s][day] - target.equal_weight_shares).mean())
        rows.append(row)

    paired = [r for r in rows if r["target"] and all(f"{s}_mae" in r for s in SOURCES)]
    result: dict = {
        "scored_through": scored_through.isoformat(),
        "capture_days": len(capture_days),
        "scorable_days": len(rows),
        "days_with_target": sum(r["target"] for r in rows),
        "paired_days": len(paired),
        "availability": {s: (sum(r[f"{s}_available"] for r in rows) / len(rows) if rows else None) for s in SOURCES},
        "first_analysis_ready": bool(
            capture_days
            and (scored_through - capture_days[0]).days >= 7 * FIRST_ANALYSIS_MIN_WEEKS
            and len(paired) >= FIRST_ANALYSIS_MIN_PAIRED_DAYS
        ),
        "rows": rows,
    }
    if not paired:
        return result
    diff = np.array([r["v02_mae"] - r["pop_mae"] for r in paired])
    lo, hi = block_bootstrap_ci(diff)
    result["primary"] = {
        "mean_mae": {s: float(np.mean([r[f"{s}_mae"] for r in paired])) for s in SOURCES},
        "mean_difference_v02_minus_pop": float(diff.mean()),
        "ci95": [lo, hi],
    }
    errors = {s: np.array([r[f"{s}_error"] for r in paired]) for s in SOURCES}
    offsets = {s: errors[s].mean(axis=0) for s in SOURCES}
    adjusted = {s: np.abs(errors[s] - offsets[s]).mean(axis=1) for s in SOURCES}
    adj_diff = adjusted["v02"] - adjusted["pop"]
    result["secondary"] = {
        "offset_adjusted_mean_mae": {s: float(adjusted[s].mean()) for s in SOURCES},
        "offset_adjusted_difference_v02_minus_pop": float(adj_diff.mean()),
        "offset_adjusted_ci95": list(block_bootstrap_ci(adj_diff)),
        "equal_weight_mean_mae": {s: float(np.mean([r[f"{s}_mae_equal_weight"] for r in paired])) for s in SOURCES},
        "mean_signed_error": {s: {p: float(v) for p, v in zip(PARTIES, offsets[s])} for s in SOURCES},
    }
    return result


def load_captures(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captures", type=Path, default=DEFAULT_CAPTURE_FILE)
    parser.add_argument("--targets", type=Path, required=True,
                        help="processed SwedishPolls long CSV retrieved at scoring time")
    parser.add_argument("--scored-through", type=date.fromisoformat, required=True,
                        help="the target snapshot's retrieval date")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    elections = load_elections(DEFAULT_ELECTION_RESULTS_FILE, [DEFAULT_2026_RESULT_MANIFEST])
    polls = load_polls(args.targets, [e.obs_date for e in elections]).polls
    result = score(load_captures(args.captures), polls, args.scored_through)
    result["inputs"] = {"captures_sha256": sha256_file(args.captures), "targets_sha256": sha256_file(args.targets)}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: result.get(k) for k in ("paired_days", "first_analysis_ready", "primary")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
