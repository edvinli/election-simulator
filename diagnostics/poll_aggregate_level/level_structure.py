"""Where the aggregate's level sits, measured without any election outcome.

Everything here compares series with polls and with each other, never with
results, so it can inform a model choice without outcome-driven selection:

- each pollster's mean deviation from v0.1 and from stored PoP;
- each series against a daily sample-weighted pollster consensus (the latest
  poll per pollster with fieldwork in the trailing 30 days, weighted by
  effective sample; the election-noise consensus rule with a daily window);
- each series against stored PoP.

    uv run python -m diagnostics.poll_aggregate_level.level_structure
"""

from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from scripts.poll_aggregate.config import DEFAULT_OUTPUT_DIR, PARTIES, SPEC_V02, TIMESERIES_FILENAME

from .lookahead import load_v01_observations
from .run import RESULTS_DIR, STORED_POP, _read_series

CONSENSUS_WINDOW_DAYS = 30
CONSENSUS_STEP_DAYS = 7
MIN_POLLSTERS = 3


def daily_consensus(polls, days) -> dict[date, np.ndarray]:
    ordered = sorted(polls, key=lambda o: (o.available, o.key))
    out = {}
    for day in days:
        latest = {}
        for o in ordered:
            if o.available <= day and o.obs_date >= day - timedelta(days=CONSENSUS_WINDOW_DAYS):
                latest[o.house] = o
        if len(latest) >= MIN_POLLSTERS:
            w = np.array([o.sample_size for o in latest.values()])
            s = np.array([o.shares for o in latest.values()])
            out[day] = (w[:, None] * s).sum(axis=0) / w.sum()
    return out


def _gap(a: dict[date, np.ndarray], b: dict[date, np.ndarray]) -> dict:
    common = sorted(set(a) & set(b))
    diff = np.array([a[d] - b[d] for d in common])
    return {
        "days": len(common),
        "mean_signed": {p: round(float(v), 3) for p, v in zip(PARTIES, diff.mean(axis=0))},
        "mean_abs": round(float(np.abs(diff).mean()), 3),
    }


def main() -> int:
    polls, _ = load_v01_observations()
    series = {
        "v0.1": _read_series(DEFAULT_OUTPUT_DIR / TIMESERIES_FILENAME),
        "v0.2": _read_series(SPEC_V02.output_dir / TIMESERIES_FILENAME),
        "stored_pop": _read_series(STORED_POP),
    }
    start = min(series["v0.1"])
    days = [start + timedelta(days=i) for i in range(0, (max(series["v0.1"]) - start).days + 1, CONSENSUS_STEP_DAYS)]
    consensus = daily_consensus(polls, days)

    pollsters: dict[str, dict[str, list[float]]] = {}
    for o in polls:
        for name in ("v0.1", "stored_pop"):
            ref = series[name].get(o.obs_date)
            if ref is not None:
                bucket = pollsters.setdefault(o.house, {"n": [], "v0.1": [], "stored_pop": []})
                bucket[name].append(o.shares[PARTIES.index("SD")] - ref[PARTIES.index("SD")])
        if o.obs_date in series["v0.1"]:
            pollsters[o.house]["n"].append(o.sample_size)
    pollster_rows = [
        {"pollster": h, "polls": len(v["n"]), "effective_sample_share": 0.0,
         "sd_minus_v01": round(float(np.mean(v["v0.1"])), 3) if v["v0.1"] else None,
         "sd_minus_stored_pop": round(float(np.mean(v["stored_pop"])), 3) if v["stored_pop"] else None}
        for h, v in sorted(pollsters.items())
    ]
    total = sum(sum(v["n"]) for v in pollsters.values())
    for row in pollster_rows:
        row["effective_sample_share"] = round(sum(pollsters[row["pollster"]]["n"]) / total, 4)

    summary = {
        "outcome_free": True,
        "consensus_rule": {"window_days": CONSENSUS_WINDOW_DAYS, "step_days": CONSENSUS_STEP_DAYS,
                           "min_pollsters": MIN_POLLSTERS},
        "minus_weighted_consensus": {name: _gap(values, consensus) for name, values in series.items()},
        "minus_stored_pop": {name: _gap(values, series["stored_pop"]) for name, values in series.items()
                             if name != "stored_pop"},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "pollster_sd_deviation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pollster_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(pollster_rows)
    (RESULTS_DIR / "level_structure.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                                      encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
