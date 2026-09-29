"""Replay the aggregate day by day, refitting only on information then available.

Hyperparameters are refitted when each election result becomes available,
by maximum likelihood over the observations known on that day, and are then
frozen until the next result. Each output row is the filtered estimate from
the observations available by its date, so the series is append-only by
construction: rebuilding with later data leaves earlier rows unchanged.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize

from .config import (
    AGGREGATE_VERSION,
    CHECKPOINT_WINDOW,
    HYPERPARAMETER_BOUNDS,
    HYPERPARAMETER_NAMES,
    HYPERPARAMETER_START,
    OPTIMIZER_MAX_ITERATIONS,
    OPTIMIZER_TOLERANCE,
    OUTPUT_SHARE_DIGITS,
    PARTIES,
    SOURCE_URL,
    SPEC_V01,
    AggregateSpec,
)
from .data import AggregateInputError, Observation
from .model import FilterState, Hyperparameters, SupportFilter

TIMESERIES_FIELDS = (
    "date", "M", "L", "C", "KD", "S", "V", "MP", "SD", "FI", "other",
    "source_extra_json", "source_url", "retrieved_at",
)


@dataclass
class Segment:
    fit_date: date
    params: Hyperparameters
    loglik: float
    n_obs: int
    reference_election: str
    optimizer_iterations: int
    optimizer_converged: bool
    consensus_weights: dict[str, float] | None = None


@dataclass
class AggregateRow:
    day: date
    mean: np.ndarray
    sd: np.ndarray
    information_date: date
    fit_date: date
    polls_known: int
    latest_poll_publication: date | None


@dataclass
class AggregateResult:
    rows: list[AggregateRow]
    segments: list[Segment]
    houses: list[str]
    stats: dict = field(default_factory=dict)


def _known(observations: Sequence[Observation], as_of: date) -> list[Observation]:
    return sorted((o for o in observations if o.available <= as_of), key=Observation.sort_key)


def _start(observations: Sequence[Observation]) -> tuple[date, tuple[float, ...]]:
    first = observations[0]
    if first.kind != "election":
        raise AggregateInputError("The first observation must be the anchor election")
    return first.obs_date, first.shares


def fit_hyperparameters(
    observations: Sequence[Observation],
    houses: Sequence[str],
    reference_shares: Sequence[float],
    initial: Hyperparameters | None = None,
    max_iterations: int = OPTIMIZER_MAX_ITERATIONS,
) -> tuple[Hyperparameters, float, int, bool]:
    """Maximum-likelihood hyperparameters within the configured bounds.

    Nelder-Mead finds different local optima from different starts, so it
    runs from the configured start and from ``initial`` (the previous
    segment's fit), then restarts once from the best point; the best
    likelihood wins.
    """

    start_day, start_shares = _start(observations)
    bounds = [tuple(math.log(v) for v in HYPERPARAMETER_BOUNDS[n]) for n in HYPERPARAMETER_NAMES]
    starts = [Hyperparameters(**HYPERPARAMETER_START).to_log()]
    if initial is not None:
        starts.append(initial.to_log())

    def negative_loglik(x: np.ndarray) -> float:
        if any(not lo <= v <= hi for v, (lo, hi) in zip(x, bounds)):
            return 1e12
        model = SupportFilter(houses, reference_shares, Hyperparameters.from_log(x))
        try:
            state = model.run(model.initial_state(start_day, start_shares), observations)
        except np.linalg.LinAlgError:
            return 1e12
        return -state.loglik

    def search(x0: Sequence[float]):
        return minimize(
            negative_loglik,
            np.asarray(x0, dtype=float),
            method="Nelder-Mead",
            options={"maxiter": max_iterations, "xatol": OPTIMIZER_TOLERANCE, "fatol": OPTIMIZER_TOLERANCE},
        )

    results = [search(x0) for x0 in starts]
    best = min(results, key=lambda r: r.fun)
    results.append(search(best.x))
    best = min(results, key=lambda r: r.fun)
    iterations = sum(int(r.nit) for r in results)
    return Hyperparameters.from_log(best.x), -float(best.fun), iterations, bool(best.success)


def consensus_weights_at(polls: Sequence[Observation], fit_date: date, window_days: int) -> dict[str, float]:
    """Each pollster's share of effective sample among polls known at ``fit_date``
    whose fieldwork midpoint lies in the trailing window. Fixed for the segment."""

    total: dict[str, float] = {}
    for o in polls:
        if o.available <= fit_date and (fit_date - o.obs_date).days < window_days:
            total[o.house] = total.get(o.house, 0.0) + float(o.sample_size)
    grand = sum(total.values())
    if grand <= 0:
        raise AggregateInputError(f"No polls in the {window_days}-day consensus window before {fit_date}")
    return {h: v / grand for h, v in sorted(total.items())}


class _CheckpointedRunner:
    """Re-runs the filter per information set, resuming from the longest shared prefix."""

    def __init__(self, model: SupportFilter, start_day: date, start_shares: Sequence[float]) -> None:
        self.model = model
        self.start_day = start_day
        self.start_shares = start_shares
        self.keys: list[str] = []
        self.checkpoints: dict[int, FilterState] = {}

    def run(self, observations: Sequence[Observation]) -> FilterState:
        keys = [o.key for o in observations]
        shared = 0
        for old, new in zip(self.keys, keys):
            if old != new:
                break
            shared += 1
        resume = max((i for i in self.checkpoints if i < shared), default=None)
        if resume is None:
            state = self.model.initial_state(self.start_day, self.start_shares)
            first = 0
        else:
            state = self.checkpoints[resume].copy()
            first = resume + 1
        self.checkpoints = {i: s for i, s in self.checkpoints.items() if i <= (resume if resume is not None else -1)}
        for i in range(first, len(observations)):
            self.model.update(state, observations[i])
            if i >= len(observations) - CHECKPOINT_WINDOW:
                self.checkpoints[i] = state.copy()
        self.keys = keys
        return state


def build_aggregate(
    polls: Sequence[Observation],
    elections: Sequence[Observation],
    fit: Callable[..., tuple[Hyperparameters, float, int, bool]] = fit_hyperparameters,
    end: date | None = None,
    spec: AggregateSpec = SPEC_V01,
) -> AggregateResult:
    """The daily causal series from the second anchor election's result onward.

    The spec only chooses the reported level (latent support, or the
    consensus reading); the filter and its fit are the same for every spec.
    """

    everything = sorted([*polls, *elections], key=Observation.sort_key)
    houses = sorted({o.house for o in polls})
    refits = sorted(e.available for e in elections)[1:]
    if not refits:
        raise AggregateInputError("At least two elections are needed to fit before the first output row")
    last = end or max(o.available for o in everything)
    change_dates = sorted({o.available for o in everything} | set(refits))

    rows: list[AggregateRow] = []
    segments: list[Segment] = []
    previous_params: Hyperparameters | None = None
    for s_index, fit_date in enumerate(refits):
        if fit_date > last:
            break
        segment_end = refits[s_index + 1] - timedelta(days=1) if s_index + 1 < len(refits) else last
        segment_end = min(segment_end, last)
        known = _known(everything, fit_date)
        reference = max((o for o in known if o.kind == "election"), key=lambda o: o.obs_date)
        params, loglik, iterations, converged = fit(known, houses, reference.shares, previous_params)
        previous_params = params
        weights = (
            consensus_weights_at(polls, fit_date, spec.consensus_window_days)
            if spec.consensus_window_days is not None else None
        )
        segments.append(
            Segment(fit_date, params, loglik, len(known), reference.key, iterations, converged, weights)
        )

        model = SupportFilter(houses, reference.shares, params)
        start_day, start_shares = _start(known)
        runner = _CheckpointedRunner(model, start_day, start_shares)
        dates = [d for d in change_dates if fit_date <= d <= segment_end]
        for c_index, info_date in enumerate(dates):
            info = _known(everything, info_date)
            state = runner.run(info)
            polls_known = [o for o in info if o.kind == "poll"]
            latest_publication = max((o.available for o in polls_known), default=None)
            until = dates[c_index + 1] - timedelta(days=1) if c_index + 1 < len(dates) else segment_end
            day = info_date
            while day <= until:
                if weights is None:
                    mean, cov = model.support_at(state, day)
                else:
                    mean, cov = model.consensus_reading_at(state, day, weights)
                rows.append(
                    AggregateRow(
                        day, mean, np.sqrt(np.diag(cov)), info_date, fit_date, len(polls_known), latest_publication
                    )
                )
                day += timedelta(days=1)

    stats = {
        "min_named_share": float(min(r.mean.min() for r in rows)) if rows else None,
        "min_other_share": float(min(100.0 - r.mean.sum() for r in rows)) if rows else None,
    }
    if rows and (stats["min_named_share"] <= 0.0 or stats["min_other_share"] < 0.0):
        raise AggregateInputError(f"Aggregate left the simplex: {stats}")
    return AggregateResult(rows, segments, houses, stats)


def _round(value: float) -> float:
    return round(float(value), OUTPUT_SHARE_DIGITS)


def write_timeseries(result: AggregateResult, path: Path, version: str = AGGREGATE_VERSION) -> None:
    """Rows in pollofpolls_timeseries.csv's columns; other parties go in ``other``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMESERIES_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in result.rows:
            record = {p: f"{_round(v):.{OUTPUT_SHARE_DIGITS}f}" for p, v in zip(PARTIES, row.mean)}
            record["FI"] = ""
            record["other"] = f"{_round(100.0 - row.mean.sum()):.{OUTPUT_SHARE_DIGITS}f}"
            record["source_extra_json"] = json.dumps(
                {
                    "aggregate_version": version,
                    "sd": {p: _round(v) for p, v in zip(PARTIES, row.sd)},
                    "information_date": row.information_date.isoformat(),
                    "hyperparameters_fit_date": row.fit_date.isoformat(),
                    "polls_known": row.polls_known,
                    "latest_poll_publication": (
                        row.latest_poll_publication.isoformat() if row.latest_poll_publication else None
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            record["source_url"] = SOURCE_URL
            record["retrieved_at"] = ""
            record["date"] = row.day.isoformat()
            writer.writerow(record)
    tmp.replace(path)
