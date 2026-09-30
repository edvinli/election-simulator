"""Level investigation for the SwedishPolls aggregate. DIAGNOSTIC ONLY.

Four measurements on the replay v1 cases, none of which is a gate:

1. Look-ahead: the aggregate rebuilt with 3, 7 and 14 days of extra
   publication (``lookahead.py``), scored as full forecasts, to see how much
   of the gap to the stored PoP series such hindsight buys.
2. Real-time PoP: the production forecast for the 2026 14- and 7-day cases
   rerun on the PoP snapshot actually committed by each date (``vintages.py``),
   plus the revision of every committed snapshot against the stored series.
3. Forecast errors by party and horizon, for every arm.
4. The aggregate-minus-PoP gap by party and horizon.

    uv run python -m diagnostics.poll_aggregate_level.run

Writes ``diagnostics/poll_aggregate_level/results/``. Nothing under ``data/``
is modified; the replay v1 outputs are read, not rewritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from scripts.poll_aggregate.build import write_timeseries
from scripts.poll_aggregate.config import DEFAULT_OUTPUT_DIR, PARTIES, REPOSITORY_ROOT, TIMESERIES_FILENAME
from scripts.poll_aggregate.data import sha256_file
from scripts.poll_aggregate import replay as R

from .lookahead import LOOKAHEAD_DAYS, build_lookahead, load_v01_observations, v01_segments
from .vintages import materialize_polling_dir, read_timeseries, snapshot_known_on, snapshots, value_on_or_before

RESULTS_DIR = Path(__file__).resolve().parent / "results"
V1_CASES = DEFAULT_OUTPUT_DIR / "replay_v1" / "cases_scored.csv"
STORED_POP = R.PROCESSED_ROOT / "pollofpolls" / "pollofpolls_timeseries.csv"
REALTIME_CASES = ((date(2026, 9, 13), 14), (date(2026, 9, 13), 7))
SERIES_END = date(2026, 9, 28)
CATEGORIES = (*PARTIES, "REST")


def lookahead_arm(k: int) -> str:
    return f"DIAGNOSTIC_LOOKAHEAD_{k}D"


REALTIME_ARM = "DIAGNOSTIC_REALTIME_POP"


def _link_production(root: Path, replace: dict[str, Path | None]) -> None:
    """A processed folder that links production except the named polling files."""

    (root / "pollofpolls").mkdir(parents=True)
    for child in R.PROCESSED_ROOT.iterdir():
        if child.name != "pollofpolls":
            os.symlink(child, root / child.name)
    for child in (R.PROCESSED_ROOT / "pollofpolls").iterdir():
        if child.name not in replace:
            os.symlink(child, root / "pollofpolls" / child.name)
    for name, source in replace.items():
        if source is not None:
            os.symlink(source, root / "pollofpolls" / name)


def _read_series(path: Path) -> dict[date, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            date.fromisoformat(r["date"]): np.array([float(r[p]) for p in PARTIES])
            for r in csv.DictReader(handle)
            if all(r.get(p) not in (None, "") for p in PARTIES)
        }


def _series_on_or_before(series: dict[date, np.ndarray], day: date) -> np.ndarray:
    return series[max(d for d in series if d <= day)]


def _cases() -> list[tuple[date, int]]:
    return [(e, h) for e in R.ELECTIONS for h in R.HORIZONS]


def center_errors(series: dict[str, dict[date, np.ndarray]], truth: dict[int, dict]) -> list[dict]:
    """Each series' estimate on each case's as_of, minus the result, per party."""

    rows = []
    for election, horizon in _cases():
        as_of = election - timedelta(days=horizon)
        actual = np.array([truth[election.year]["vote"][p] for p in PARTIES])
        for name, values in series.items():
            error = _series_on_or_before(values, as_of) - actual
            rows.append({"series": name, "election": election.isoformat(), "horizon_days": horizon,
                         **{p: round(float(e), 4) for p, e in zip(PARTIES, error)},
                         "mae": round(float(np.abs(error).mean()), 4)})
    return rows


def vintage_revisions() -> list[dict]:
    """For each committed snapshot, its latest value against the stored series' value for that date."""

    stored = _read_series(STORED_POP)
    rows = []
    for snap in snapshots():
        series = read_timeseries(snap.commit)
        latest = max(series)
        if latest not in stored:
            continue
        revision = stored[latest] - np.array([series[latest][p] for p in PARTIES])
        rows.append({"commit": snap.commit, "known_on": snap.known_on.isoformat(),
                     "latest_date": latest.isoformat(), "staleness_days": (snap.known_on - latest).days,
                     **{p: round(float(v), 2) for p, v in zip(PARTIES, revision)},
                     "mean_abs_revision": round(float(np.abs(revision).mean()), 3),
                     "max_abs_revision": round(float(np.abs(revision).max()), 3)})
    return rows


def forecast_errors(rows: list[dict], truth: dict[int, dict]) -> list[dict]:
    """Forecast-mean minus result per party for each scored case row."""

    out = []
    for r in rows:
        means = [float(v) for v in str(r["mean_vote_pct"]).split("|")][: len(PARTIES)]
        actual = [truth[int(r["election"][:4])]["vote"][p] for p in PARTIES]
        out.append({"arm": r["arm"], "election": r["election"], "horizon_days": int(r["horizon_days"]),
                    **{p: round(m - a, 4) for p, m, a in zip(PARTIES, means, actual)}})
    return out


def summarize_by(rows: list[dict], key: str, group: tuple[str, ...]) -> list[dict]:
    """Mean signed and absolute per-party values of ``rows`` grouped by ``group``."""

    buckets: dict[tuple, list[dict]] = {}
    for r in rows:
        buckets.setdefault(tuple(r[g] for g in group), []).append(r)
    out = []
    for k, items in sorted(buckets.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        record = dict(zip(group, k))
        for p in PARTIES:
            vals = np.array([float(i[p]) for i in items])
            record[f"{p}_signed"] = round(float(vals.mean()), 3)
            record[f"{p}_abs"] = round(float(np.abs(vals).mean()), 3)
        record["mean_abs_8"] = round(float(np.mean([abs(float(i[p])) for i in items for p in PARTIES])), 3)
        record["n"] = len(items)
        record["_kind"] = key
        out.append(record)
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _pooled(rows: list[dict], arm: str, metric: str, cases: set[tuple[str, int]] | None = None) -> float:
    vals = [float(r[metric]) for r in rows if r["arm"] == arm
            and (cases is None or (r["election"], int(r["horizon_days"])) in cases)]
    return round(float(np.mean(vals)), 4)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--skip-forecasts", action="store_true", help="centre and vintage diagnostics only")
    args = parser.parse_args(argv)

    truth = R.load_truth()
    polls, elections = load_v01_observations()
    segments = v01_segments()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    v1_rows = list(csv.DictReader(V1_CASES.open(encoding="utf-8", newline="")))

    with tempfile.TemporaryDirectory(prefix="poll-aggregate-level-") as tmp_name:
        tmp = Path(tmp_name)
        series_files: dict[str, Path] = {"v0.1": DEFAULT_OUTPUT_DIR / TIMESERIES_FILENAME, "stored_pop": STORED_POP}
        for k in LOOKAHEAD_DAYS:
            path = tmp / f"lookahead_{k}d.csv"
            write_timeseries(build_lookahead(polls, elections, segments, k, SERIES_END), path)
            series_files[f"lookahead_{k}d"] = path
            filtered = tmp / f"lookahead_{k}d_filter_only.csv"
            write_timeseries(build_lookahead(polls, elections, segments, k, SERIES_END, smooth=False), filtered)
            series_files[f"lookahead_{k}d_filter_only"] = filtered
        series = {name: _read_series(path) for name, path in series_files.items()}

        centre = center_errors(series, truth)
        _write_csv(RESULTS_DIR / "center_errors.csv", centre)
        revisions = vintage_revisions()
        _write_csv(RESULTS_DIR / "pop_vintage_revisions.csv", revisions)

        # Aggregate-minus-PoP gap on case dates and over all overlapping days.
        gap_rows = []
        for election, horizon in _cases():
            as_of = election - timedelta(days=horizon)
            gap = _series_on_or_before(series["v0.1"], as_of) - _series_on_or_before(series["stored_pop"], as_of)
            gap_rows.append({"election": election.isoformat(), "horizon_days": horizon,
                             **{p: round(float(g), 4) for p, g in zip(PARTIES, gap)}})
        overlap = sorted(set(series["v0.1"]) & set(series["stored_pop"]))
        daily_gap = np.array([series["v0.1"][d] - series["stored_pop"][d] for d in overlap])

        realtime_rows = []
        for election, horizon in REALTIME_CASES:
            as_of = election - timedelta(days=horizon)
            snap = snapshot_known_on(as_of)
            selected_date, values = value_on_or_before(read_timeseries(snap.commit), as_of)
            realtime_rows.append({"election": election.isoformat(), "horizon_days": horizon,
                                  "as_of": as_of.isoformat(), "snapshot_commit": snap.commit,
                                  "snapshot_known_on": snap.known_on.isoformat(),
                                  "selected_estimate_date": selected_date.isoformat(),
                                  **{p: values[p] for p in PARTIES}})

        forecast_rows: list[dict] = []
        if not args.skip_forecasts:
            jobs = []
            for k in LOOKAHEAD_DAYS:
                root = tmp / lookahead_arm(k)
                polls_file = tmp / f"individual_polls_{k}.csv"
                R.write_swedishpolls_individual_polls(polls_file)
                _link_production(root, {"pollofpolls_timeseries.csv": series_files[f"lookahead_{k}d"],
                                        "individual_polls.csv": polls_file})
                for election, horizon in _cases():
                    jobs.append((lookahead_arm(k), root, election, horizon, str(series_files[f"lookahead_{k}d"])))
            for election, horizon in REALTIME_CASES:
                as_of = election - timedelta(days=horizon)
                snap = snapshot_known_on(as_of)
                root = tmp / f"{REALTIME_ARM}_{horizon}"
                vintage_dir = tmp / f"vintage_{snap.commit[:12]}"
                names = materialize_polling_dir(snap.commit, vintage_dir)
                # Every polling file comes from the snapshot; nothing is linked from production.
                _link_production(root, {n: vintage_dir / n for n in names} | {
                    c.name: None for c in (R.PROCESSED_ROOT / "pollofpolls").iterdir() if c.name not in names})
                jobs.append((REALTIME_ARM, root, election, horizon, ""))
            case_jobs = []
            for arm, root, election, horizon, aggregate_file in jobs:
                record = truth[election.year]
                case_jobs.append({
                    "arm": arm, "election": election.isoformat(), "horizon": horizon, "data_root": str(root),
                    "aggregate_file": aggregate_file, "gated": False,
                    "truth": {"vote": [record["vote"][c] for c in CATEGORIES],
                              "seats": [record["seats"][p] for p in PARTIES]},
                })
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                forecast_rows = list(pool.map(R.run_case, case_jobs))
            _write_csv(RESULTS_DIR / "forecast_cases.csv", forecast_rows)

        series_hashes = {name: sha256_file(path) for name, path in series_files.items()}

    all_rows = [r for r in v1_rows if r["arm"] != R.DIAGNOSTIC] + forecast_rows
    errors = forecast_errors(all_rows, truth)
    _write_csv(RESULTS_DIR / "forecast_errors_by_case.csv", errors)
    party_tables = (summarize_by(errors, "forecast_by_arm_horizon", ("arm", "horizon_days"))
                    + summarize_by(errors, "forecast_by_arm_election", ("arm", "election"))
                    + summarize_by(errors, "forecast_by_arm", ("arm",))
                    + summarize_by(centre, "center_by_series_horizon", ("series", "horizon_days"))
                    + summarize_by(centre, "center_by_series", ("series",))
                    + summarize_by(gap_rows, "gap_v01_minus_stored_pop_by_horizon", ("horizon_days",))
                    + summarize_by(gap_rows, "gap_v01_minus_stored_pop_by_election", ("election",)))
    _write_csv(RESULTS_DIR / "party_breakdowns.csv", party_tables)

    arms = [R.CONTROL, R.CANDIDATE] + [lookahead_arm(k) for k in LOOKAHEAD_DAYS]
    short = {(e.isoformat(), h) for e in R.ELECTIONS for h in (7, 14)}
    long = {(e.isoformat(), h) for e in R.ELECTIONS for h in (84, 112)}
    rt_cases = {(e.isoformat(), h) for e, h in REALTIME_CASES}
    scored = [r for r in all_rows if r.get("es_9cat") not in (None, "")]
    summary = {
        "diagnostic_only": True,
        "head_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
                                      capture_output=True, text=True).stdout.strip(),
        "v1_cases_sha256": sha256_file(V1_CASES),
        "series_sha256": series_hashes,
        "draws": R.DRAWS, "seed": R.SEED,
        "es_9cat_pooled": {a: _pooled(scored, a, "es_9cat") for a in arms if any(r["arm"] == a for r in scored)},
        "es_9cat_short_7_14": {a: _pooled(scored, a, "es_9cat", short) for a in arms
                               if any(r["arm"] == a for r in scored)},
        "es_9cat_long_84_112": {a: _pooled(scored, a, "es_9cat", long) for a in arms
                                if any(r["arm"] == a for r in scored)},
        "coverage_90_pooled": {a: _pooled(scored, a, "coverage_90") for a in arms
                               if any(r["arm"] == a for r in scored)},
        "realtime_2026": {
            "cases": realtime_rows,
            "es_9cat": {a: _pooled(scored, a, "es_9cat", rt_cases)
                        for a in [R.CONTROL, R.CANDIDATE, REALTIME_ARM] + [lookahead_arm(k) for k in LOOKAHEAD_DAYS]
                        if any(r["arm"] == a and (r["election"], int(r["horizon_days"])) in rt_cases for r in scored)},
        },
        "center_mae_pooled": {name: round(float(np.mean([r["mae"] for r in centre if r["series"] == name])), 4)
                              for name in series_files},
        "daily_gap_v01_minus_stored_pop": {
            "days": len(overlap),
            "mean_signed": {p: round(float(v), 3) for p, v in zip(PARTIES, daily_gap.mean(axis=0))},
            "mean_abs": {p: round(float(v), 3) for p, v in zip(PARTIES, np.abs(daily_gap).mean(axis=0))},
        },
        "pop_vintage_revision_mean_abs": round(float(np.mean([r["mean_abs_revision"] for r in revisions])), 3)
        if revisions else None,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("es_9cat_pooled", "es_9cat_short_7_14", "center_mae_pooled")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
