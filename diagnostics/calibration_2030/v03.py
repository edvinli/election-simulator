"""SwedishPollsAggregate-v0.3: T1's SH filter with rolling consensus weights W (final round §2).

Hyperparameters are not refitted. Each segment uses SH's recorded fit in
``t1_results.json``. The readings are exactly those of the T1 harness's
``build_rows(..., weights_mode="rolling")``. The self-check reproduces the
SH M readings that T1 recorded with rolling weights.
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

from scripts.poll_aggregate.config import PARTIES

from .t1_aggregate import build_rows, load_inputs, segments

AGGREGATE_VERSION = "SwedishPollsAggregate-v0.3"
T1_RESULTS = Path(__file__).resolve().parent / "t1_results.json"
TIMESERIES_FIELDS = ("date", "M", "L", "C", "KD", "S", "V", "MP", "SD", "FI", "other",
                     "source_extra_json", "source_url", "retrieved_at")


def sh_thetas() -> list[dict]:
    fits = json.loads(T1_RESULTS.read_text(encoding="utf-8"))["arms"]["SH"]["fits"]
    return [dict(f["theta"]) for f in fits]


def v03_context():
    """Inputs, segments and SH thetas, loaded once for callers that need the filter itself."""

    elections, polls, everything, houses = load_inputs()
    segs = segments(elections, everything)
    thetas = sh_thetas()
    if len(thetas) != len(segs):
        raise RuntimeError("SH fits do not match the reconstructed segments")
    return {"elections": elections, "polls": polls, "everything": everything, "houses": houses,
            "segments": segs, "thetas": thetas}


def v03_rows(ctx: dict | None = None, end: date | None = None) -> dict[date, tuple[date, list[float]]]:
    """Daily v0.3 readings from 2010-09-26: {day: (fit_date, eight shares in pp)}, unrounded."""

    ctx = ctx or v03_context()
    end = end or max(o.available for o in ctx["everything"])
    rows = build_rows("SH", ctx["houses"], ctx["polls"], ctx["everything"], ctx["segments"], ctx["thetas"], end, "rolling")
    return {d: (f, [float(x) for x in m]) for d, (f, m) in rows.items()}


def write_timeseries(rows: dict[date, tuple[date, list[float]]], path: Path) -> Path:
    """Rows in production's timeseries schema with 2-decimal rounding (as ``write_timeseries`` does)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMESERIES_FIELDS, lineterminator="\n")
        writer.writeheader()
        for day in sorted(rows):
            fit_date, mean = rows[day]
            record = {p: f"{round(v, 2):.2f}" for p, v in zip(PARTIES, mean)}
            record.update({
                "date": day.isoformat(), "FI": "", "other": f"{round(100.0 - sum(mean), 2):.2f}",
                "source_extra_json": json.dumps({"aggregate_version": AGGREGATE_VERSION,
                                                 "hyperparameters_fit_date": fit_date.isoformat()},
                                                sort_keys=True, separators=(",", ":")),
                "source_url": "", "retrieved_at": "",
            })
            writer.writerow(record)
    return path


def self_check(rows: dict[date, tuple[date, list[float]]]) -> dict:
    """The regenerated v0.3 must reproduce T1's recorded SH M readings with rolling weights."""

    recorded = json.loads(T1_RESULTS.read_text(encoding="utf-8"))["arms"]["SH"]["m_reading_2026_rolling_weights"]
    m = PARTIES.index("M")
    diffs = {d: abs(rows[date.fromisoformat(d)][1][m] - v) for d, v in recorded.items()}
    return {"ok": all(x < 1e-9 for x in diffs.values()), "abs_diff": diffs}
