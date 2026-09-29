"""Look-ahead variants of the SwedishPolls aggregate. DIAGNOSTIC ONLY.

A look-ahead series with lag ``k`` reports, for each day ``T``, the estimate
of support on ``T`` from every observation published by ``T + k``, whatever
its fieldwork dates. That mimics a fieldwork-dated series backfilled as later
polls arrive, which is how the stored PoP series behaves
(docs/election_noise_v2_historical_pop_extension.md, section 4). It uses
information a forecaster at ``T`` did not have, so these series must never be
used as, or turned into, a production candidate. ``k = 0`` is exactly the
causal v0.1 series and serves as the reproduction check.

Two variants:

- ``smooth=True`` (the definition above) is an exact fixed-lag smoother:
  after filtering to ``T``, a frozen copy of the eight shares is appended to
  the state, and observations with fieldwork midpoints in ``(T, T + k]`` then
  update it through its covariance with the live state.
- ``smooth=False`` only adds observations with fieldwork midpoints on or
  before ``T`` that were published late, to isolate that part.

The v0.1 segment hyperparameters are reused unchanged (read from the v0.1
metadata): only the information set differs, so any difference against v0.1
is attributable to the extra polls alone.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np

from scripts.poll_aggregate.build import AggregateResult, AggregateRow, Segment, _CheckpointedRunner
from scripts.poll_aggregate.config import (
    DEFAULT_2026_RESULT_MANIFEST,
    DEFAULT_ELECTION_RESULTS_FILE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POLLS_FILE,
    METADATA_FILENAME,
)
from scripts.poll_aggregate.data import Observation, load_elections, load_polls
from scripts.poll_aggregate.model import K, FilterState, Hyperparameters, SupportFilter

DIAGNOSTIC_ONLY = True
LOOKAHEAD_DAYS: tuple[int, ...] = (3, 7, 14)


def v01_segments(metadata_file: Path = DEFAULT_OUTPUT_DIR / METADATA_FILENAME) -> list[dict]:
    """The v0.1 segments: fit date, reference election and frozen hyperparameters."""

    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    return [
        {
            "fit_date": date.fromisoformat(s["hyperparameters_fit_date"]),
            "reference_election": s["reference_election"],
            "params": Hyperparameters(**s["hyperparameters"]),
        }
        for s in metadata["segments"]
    ]


def load_v01_observations() -> tuple[list[Observation], list[Observation]]:
    elections = load_elections(DEFAULT_ELECTION_RESULTS_FILE, [DEFAULT_2026_RESULT_MANIFEST])
    polls = load_polls(DEFAULT_POLLS_FILE, [e.obs_date for e in elections]).polls
    return polls, elections


def _visible(observations: Sequence[Observation], day: date, lookahead: int) -> list[Observation]:
    horizon = day + timedelta(days=lookahead)
    return sorted(
        (o for o in observations if o.available <= horizon and o.obs_date <= day),
        key=Observation.sort_key,
    )


def _smoothed(model: SupportFilter, state: FilterState, later: Sequence[Observation], day: date):
    """Support on ``day`` given ``state`` (filtered to ``day``) and later observations.

    Appending a noiseless frozen copy of the shares makes the filter carry
    their posterior as later observations arrive: an exact fixed-lag smoother
    that reuses the filter's own update. The copy is never read by an
    observation and gets no process noise, because both only touch the
    original indices.
    """

    live = state.copy()
    model.predict(live, day)
    dim = live.mean.size
    mean = np.concatenate([live.mean, live.mean[:K]])
    cov = np.zeros((dim + K, dim + K))
    cov[:dim, :dim] = live.cov
    cov[:dim, dim:] = live.cov[:, :K]
    cov[dim:, :dim] = live.cov[:K, :]
    cov[dim:, dim:] = live.cov[:K, :K]
    augmented = FilterState(live.day, mean, cov, live.loglik, live.n_obs, dict(live.house_last_day))
    for obs in later:
        model.update(augmented, obs)
    return augmented.mean[dim:].copy(), augmented.cov[dim:, dim:].copy()


def _later(observations: Sequence[Observation], day: date, lookahead: int) -> list[Observation]:
    horizon = day + timedelta(days=lookahead)
    return sorted(
        (o for o in observations if o.available <= horizon and day < o.obs_date <= horizon),
        key=Observation.sort_key,
    )


def build_lookahead(
    polls: Sequence[Observation],
    elections: Sequence[Observation],
    segments: Sequence[dict],
    lookahead: int,
    end: date,
    smooth: bool = True,
) -> AggregateResult:
    """The daily series with ``lookahead`` days of extra publication, v0.1 parameters."""

    everything = sorted([*polls, *elections], key=Observation.sort_key)
    houses = sorted({o.house for o in polls})
    by_key = {e.key: e for e in elections}
    start = min(elections, key=lambda e: e.obs_date)
    rows: list[AggregateRow] = []
    out_segments: list[Segment] = []
    for index, segment in enumerate(segments):
        first = segment["fit_date"]
        if first > end:
            break
        last = segments[index + 1]["fit_date"] - timedelta(days=1) if index + 1 < len(segments) else end
        last = min(last, end)
        reference = by_key[segment["reference_election"]]
        model = SupportFilter(houses, reference.shares, segment["params"])
        runner = _CheckpointedRunner(model, start.obs_date, start.shares)
        out_segments.append(Segment(first, segment["params"], float("nan"), 0, reference.key, 0, True))
        previous_keys: tuple[str, ...] | None = None
        state = None
        day = first
        while day <= last:
            visible = _visible(everything, day, lookahead)
            keys = tuple(o.key for o in visible)
            if keys != previous_keys:
                state = runner.run(visible)
                previous_keys = keys
                polls_known = [o for o in visible if o.kind == "poll"]
                information = max((o.available for o in visible), default=first)
                latest = max((o.available for o in polls_known), default=None)
            later = _later(everything, day, lookahead) if smooth else []
            if later:
                mean, cov = _smoothed(model, state, later, day)
            else:
                mean, cov = model.support_at(state, day)
            rows.append(
                AggregateRow(day, mean, np.sqrt(np.diag(cov)), information, first, len(polls_known), latest)
            )
            day += timedelta(days=1)
    return AggregateResult(rows, out_segments, houses, {})
