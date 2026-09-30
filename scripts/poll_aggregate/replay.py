"""Replay the full forecast on PoP and on the aggregate, per the replay protocol.

``docs/poll_aggregate_replay_protocol.md`` fixes the arms, cases, metrics and
gates. Scoring against election outcomes is refused unless that document's
SHA-256 equals ``AGREED_PROTOCOL_SHA256``, which is set only once the protocol
has been agreed. Without ``--score`` the harness runs every case and records
integrity checks and forecast means, but never reads an outcome.

    uv run python -m scripts.poll_aggregate.replay            # dry run
    uv run python -m scripts.poll_aggregate.replay --score    # after agreement
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import (
    DEFAULT_2026_RESULT_MANIFEST,
    DEFAULT_ELECTION_RESULTS_FILE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POLLS_FILE,
    PARTIES,
    REPOSITORY_ROOT,
    TIMESERIES_FILENAME,
)
from .data import load_elections, load_polls, sha256_file
from .outputs import INDIVIDUAL_POLL_FIELDS, write_individual_polls

PROTOCOL_PATH = REPOSITORY_ROOT / "docs" / "poll_aggregate_replay_protocol.md"
#: Set to the protocol's SHA-256 when it is agreed; ``None`` refuses scoring.
AGREED_PROTOCOL_SHA256: str | None = "3a34221883158a76b2b1021605c2cb1f636a44bba26b06063b2e727c6dc1be5c"

#: The fixed baseline (the committed step-1 aggregate). Scored inputs and model
#: code must be byte-identical to it; only this harness may differ.
BASELINE_COMMIT = "fd405558ce95388d19da1b04077c19639bc61256"
BASELINE_PATHS: tuple[str, ...] = ("data", "scripts", "diagnostics", "pyproject.toml", "uv.lock")
HARNESS_PATH = "scripts/poll_aggregate/replay.py"

ELECTIONS: tuple[date, ...] = (date(2018, 9, 9), date(2022, 9, 11), date(2026, 9, 13))
HORIZONS: tuple[int, ...] = (112, 84, 56, 28, 14, 7)
DRAWS: int = 20_000
SEED: int = 20260929

CONTROL = "CONTROL_POP"
CANDIDATE = "CANDIDATE_SWEDISHPOLLS"
DIAGNOSTIC = "DIAGNOSTIC_TS_ONLY"
BASELINE = "CONTROL_SWEDISHPOLLS_CONSENSUS"
ARMS: tuple[str, ...] = (CONTROL, CANDIDATE, DIAGNOSTIC)

PROCESSED_ROOT = REPOSITORY_ROOT / "data" / "processed"
MANDATES_FILE = PROCESSED_ROOT / "mandates" / "historical_certified_mandates.csv"
DEFAULT_REPLAY_DIR = DEFAULT_OUTPUT_DIR / "replay_v1"
THRESHOLD_PCT = 4.0


@dataclass(frozen=True)
class ReplayConfig:
    """One frozen replay: its protocol, agreement, baseline and aggregate.

    Every replay shares the cases, draws, seed, metrics and gates, fixed by
    each protocol to the v1 values; the arms and the control can differ.
    """

    name: str
    protocol_path: Path
    agreed_sha256: str | None
    baseline_commit: str | None
    aggregate_file: Path
    output_dir: Path
    arms: tuple[str, ...] = ARMS
    control: str = CONTROL
    #: For a series control (not PoP): the timeseries its arm reads, with the
    #: same SwedishPolls individual polls as the candidate.
    control_timeseries: Path | None = None


REPLAY_V1 = ReplayConfig(
    "v1", PROTOCOL_PATH, AGREED_PROTOCOL_SHA256, BASELINE_COMMIT,
    DEFAULT_OUTPUT_DIR / TIMESERIES_FILENAME, DEFAULT_REPLAY_DIR,
)
#: v2 scores SwedishPollsAggregate-v0.2. Its agreement hash and baseline commit
#: are set together, in their own commit, after the v2 protocol is committed.
REPLAY_V2 = ReplayConfig(
    "v2", REPOSITORY_ROOT / "docs" / "poll_aggregate_replay_protocol_v2.md",
    "fe6c52299b9bab10c1c0972e0027e7842cd69096251325959397b3c52ce0a7f9",
    "7153bcbef80518a3c3e32f76ca35ec6034a69557",
    DEFAULT_OUTPUT_DIR / "v0_2" / TIMESERIES_FILENAME, DEFAULT_OUTPUT_DIR / "replay_v2",
)
#: v4 benchmarks v0.2 against the SwedishPolls-only consensus baseline, not
#: against backfilled PoP. Agreement and baseline are set in their own commit.
REPLAY_V4 = ReplayConfig(
    "v4", REPOSITORY_ROOT / "docs" / "poll_aggregate_replay_protocol_v4.md",
    "daaac76c3765f869d3e4b5f35bbbca6c76670a875b0d2b42c0448ef965a76c64",
    "286b6a274cb31e188f557d3592d88207f46b39f8",
    DEFAULT_OUTPUT_DIR / "v0_2" / TIMESERIES_FILENAME, DEFAULT_OUTPUT_DIR / "replay_v4",
    arms=(BASELINE, CANDIDATE), control=BASELINE,
    control_timeseries=DEFAULT_OUTPUT_DIR / "consensus_baseline" / TIMESERIES_FILENAME,
)
REPLAYS: dict[str, ReplayConfig] = {"v1": REPLAY_V1, "v2": REPLAY_V2, "v4": REPLAY_V4}



def write_swedishpolls_individual_polls(dest: Path, polls_file: Path = DEFAULT_POLLS_FILE) -> int:
    """SwedishPolls polls that pass the aggregate's cleaning, in individual_polls.csv's schema."""

    elections = load_elections(DEFAULT_ELECTION_RESULTS_FILE, [DEFAULT_2026_RESULT_MANIFEST])
    eligible = {o.key for o in load_polls(polls_file, [e.obs_date for e in elections]).polls}
    return write_individual_polls(polls_file, eligible, dest)


def build_data_root(arm: str, parent: Path, aggregate_file: Path, control_timeseries: Path | None = None) -> Path:
    """A processed-data folder identical to production except the arm's two files.

    With ``control_timeseries``, the control arm reads that series and the
    candidate's SwedishPolls individual polls, so the two arms differ only in
    their timeseries.
    """

    root = parent / arm
    (root / "pollofpolls").mkdir(parents=True)
    for child in PROCESSED_ROOT.iterdir():
        if child.name != "pollofpolls":
            os.symlink(child, root / child.name)
    replaced = {
        CONTROL: set(),
        CANDIDATE: {"pollofpolls_timeseries.csv", "individual_polls.csv"},
        DIAGNOSTIC: {"pollofpolls_timeseries.csv"},
        BASELINE: {"pollofpolls_timeseries.csv", "individual_polls.csv"},
    }[arm]
    series = control_timeseries if arm == BASELINE else aggregate_file
    if arm == BASELINE and control_timeseries is None:
        raise ValueError("the consensus baseline arm needs its timeseries")
    for child in (PROCESSED_ROOT / "pollofpolls").iterdir():
        if child.name not in replaced:
            os.symlink(child, root / "pollofpolls" / child.name)
    if "pollofpolls_timeseries.csv" in replaced:
        os.symlink(series, root / "pollofpolls" / "pollofpolls_timeseries.csv")
    if "individual_polls.csv" in replaced:
        write_swedishpolls_individual_polls(root / "pollofpolls" / "individual_polls.csv")
    return root


def load_truth() -> dict[int, dict]:
    """Nine-category vote shares (pp) and eight-party national seats per election."""

    truth: dict[int, dict] = {}
    with DEFAULT_ELECTION_RESULTS_FILE.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            year = int(row["election_year"])
            if row["party"] in PARTIES:
                truth.setdefault(year, {"vote": {}, "seats": {p: 0 for p in PARTIES}})["vote"][row["party"]] = float(
                    row["vote_share"]
                )
    with MANDATES_FILE.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            year = int(row["election_year"])
            if year in truth and row["party"] in PARTIES:
                truth[year]["seats"][row["party"]] += int(row["total_seats"])
    manifest = json.loads(DEFAULT_2026_RESULT_MANIFEST.read_text(encoding="utf-8"))
    truth[2026] = {
        "vote": {p: float(manifest["parties"][p]["vote_share_percentage_points"]) for p in PARTIES},
        "seats": {p: int(manifest["parties"][p]["seats"]) for p in PARTIES},
    }
    for year, record in truth.items():
        record["vote"]["REST"] = 100.0 - sum(record["vote"][p] for p in PARTIES)
    return truth


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPOSITORY_ROOT, capture_output=True, text=True)


def verify_baseline(config: ReplayConfig = REPLAY_V1) -> dict:
    """Refuse to score unless the checkout is clean and inputs match the baseline.

    Clean means no modified, staged or untracked file. Matching means no
    difference between the baseline commit and HEAD under the model-input and
    code paths, except this harness; with a clean checkout, the files the
    harness reads are then exactly the baseline's.
    """

    if config.baseline_commit is None:
        raise RuntimeError(f"Replay {config.name} has no baseline commit recorded")
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status.returncode != 0:
        raise RuntimeError(f"git status failed: {status.stderr.strip()}")
    if status.stdout.strip():
        raise RuntimeError(f"Scoring needs a clean checkout; found:\n{status.stdout}")
    diff = _git("diff", "--name-only", config.baseline_commit, "HEAD", "--", *BASELINE_PATHS,
                f":(exclude){HARNESS_PATH}")
    if diff.returncode != 0:
        raise RuntimeError(f"git diff against the baseline failed: {diff.stderr.strip()}")
    if diff.stdout.strip():
        raise RuntimeError(f"Inputs or model code differ from baseline {config.baseline_commit}:\n{diff.stdout}")
    return {
        "replay": config.name,
        "baseline_commit": config.baseline_commit,
        "head_commit": _git("rev-parse", "HEAD").stdout.strip(),
        "clean_checkout": True,
        "inputs_match_baseline": True,
        "checked_paths": list(BASELINE_PATHS),
    }


def _aggregate_row(aggregate_file: Path, day: str) -> dict | None:
    with aggregate_file.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["date"] == day:
                return row
    return None


def selected_estimate(job: dict, as_of: date) -> dict:
    """The timeseries row OpinionState actually selects, as the engine calls it.

    For arms reading the aggregate, the selected values must equal the
    aggregate row of that date, and that row's information date must not be
    after ``as_of``.
    """

    from scripts.pollofpolls.state import estimate_opinion

    state = estimate_opinion(as_of=as_of, data_dir=Path(job["data_root"]) / "pollofpolls")
    record = {
        "selected_estimate_date": state.estimate_date.isoformat(),
        "selected_estimate_pct": "|".join(f"{state.mean_pct[p]:.4f}" for p in PARTIES),
        "aggregate_row_matches": "",
        "aggregate_information_date": "",
    }
    if job["aggregate_file"]:
        row = _aggregate_row(Path(job["aggregate_file"]), record["selected_estimate_date"])
        matches = row is not None and all(
            abs(float(row[p]) - state.mean_pct[p]) <= 1e-6 for p in PARTIES
        )
        record["aggregate_row_matches"] = bool(matches)
        if row is not None:
            record["aggregate_information_date"] = json.loads(row["source_extra_json"])["information_date"]
    return record


def case_dates_ok(row: dict) -> bool:
    """G6's date condition: the selected estimate is dated on ``as_of`` and, for
    arms reading the aggregate, is that row with information known by then."""

    if row["selected_estimate_date"] != row["as_of"]:
        return False
    if row["aggregate_row_matches"] == "":  # the arm reads production PoP, not a series file
        return True
    return row["aggregate_row_matches"] is True and row["aggregate_information_date"] <= row["as_of"]


def run_case(job: dict) -> dict:
    """One (arm, election, horizon) forecast; scores it only when asked to."""

    from scripts.simulator.engine import simulate_election

    election = date.fromisoformat(job["election"])
    as_of = election - timedelta(days=job["horizon"])
    result = simulate_election(
        as_of=as_of, election_date=election, samples=DRAWS, seed=SEED, data_dir=Path(job["data_root"])
    )
    votes = np.asarray(result.vote_shares_matrix, dtype=float)
    seats = np.asarray(result.seats_matrix, dtype=np.int64)
    row = {
        "arm": job["arm"],
        "election": job["election"],
        "horizon_days": job["horizon"],
        "as_of": as_of.isoformat(),
        "opinion_as_of": str(result.summary.as_of),
        **selected_estimate(job, as_of),
        "gated": job["gated"],
        "seat_total_always_349": bool(np.all(seats.sum(axis=1) == 349)),
        "all_finite": bool(np.all(np.isfinite(votes))),
        "mean_vote_pct": "|".join(f"{v:.4f}" for v in votes.mean(axis=0)),
        "mean_seats": "|".join(f"{v:.3f}" for v in seats.mean(axis=0)),
    }
    if job["truth"] is not None:
        from diagnostics.election_noise_v2.control_baseline.harness import metrics as M

        truth_vote = np.asarray(job["truth"]["vote"], dtype=float)
        truth_seats = np.asarray(job["truth"]["seats"], dtype=np.int64)
        row.update(M.d1_joint_vote_energy_score(votes, truth_vote))
        marginal = M.d2_marginal_vote_metrics(votes, truth_vote)
        row.update({k: v for k, v in marginal.items() if k != "per_party"})
        seat = M.d3_seat_metrics(seats, truth_seats)
        row.update({f"seat_{k}" if not k.startswith("seat") else k: v
                    for k, v in seat.items() if k != "per_party"})
        row["brier_coalition"] = M.d4_coalition_brier(seats, truth_seats)["brier_mean_over_masks"]
        p_threshold = (votes[:, : len(PARTIES)] >= THRESHOLD_PCT).mean(axis=0)
        y_threshold = (truth_vote[: len(PARTIES)] >= THRESHOLD_PCT).astype(float)
        row["brier_threshold"] = float(np.mean((p_threshold - y_threshold) ** 2))
    return row


def _jobs(roots: dict[str, Path], truth: dict[int, dict] | None, aggregate_file: Path,
          config: "ReplayConfig | None" = None) -> list[dict]:
    config = config or REPLAY_V1
    categories = (*PARTIES, "REST")
    jobs = []
    for arm in config.arms:
        for election in ELECTIONS:
            for horizon in HORIZONS:
                case_truth = None
                if truth is not None:
                    record = truth[election.year]
                    case_truth = {
                        "vote": [record["vote"][c] for c in categories],
                        "seats": [record["seats"][p] for p in PARTIES],
                    }
                jobs.append(
                    {
                        "arm": arm,
                        "election": election.isoformat(),
                        "horizon": horizon,
                        "data_root": str(roots[arm]),
                        "aggregate_file": (
                            "" if arm == CONTROL
                            else str(config.control_timeseries) if arm == BASELINE
                            else str(aggregate_file)
                        ),
                        "gated": arm != DIAGNOSTIC,
                        "truth": case_truth,
                    }
                )
    return jobs


def _pooled(rows: list[dict], arm: str, metric: str, election: str | None = None) -> float:
    values = [
        float(r[metric]) for r in rows
        if r["arm"] == arm and r["gated"] and (election is None or r["election"] == election)
    ]
    if not values:
        raise ValueError(f"no gated rows for {arm} {metric} {election}")
    return float(np.mean(values))


def evaluate_gates(rows: list[dict], control: str = CONTROL) -> dict:
    """The protocol's section-7 gates on scored case rows."""

    def pooled(arm: str, metric: str, election: str | None = None) -> float:
        return _pooled(rows, arm, metric, election)

    c, s = control, CANDIDATE
    gates: dict[str, dict] = {}
    es_c, es_s = pooled(c, "es_9cat"), pooled(s, "es_9cat")
    gates["G1_accuracy"] = {"candidate": es_s, "control": es_c, "pass": es_s <= 1.05 * es_c}
    per_election = {}
    for election in ELECTIONS:
        e = election.isoformat()
        ec, es = pooled(c, "es_9cat", e), pooled(s, "es_9cat", e)
        per_election[e] = {"candidate": es, "control": ec, "pass": es <= 1.15 * ec}
    gates["G2_per_election"] = {"elections": per_election, "pass": all(v["pass"] for v in per_election.values())}
    seat_c, seat_s = pooled(c, "seat_energy_score"), pooled(s, "seat_energy_score")
    coal_c, coal_s = pooled(c, "brier_coalition"), pooled(s, "brier_coalition")
    gates["G3_seats"] = {
        "seat_energy_score": {"candidate": seat_s, "control": seat_c},
        "brier_coalition": {"candidate": coal_s, "control": coal_c},
        "pass": seat_s <= 1.05 * seat_c and coal_s <= coal_c + 0.01,
    }
    thr_c, thr_s = pooled(c, "brier_threshold"), pooled(s, "brier_threshold")
    gates["G4_thresholds"] = {"candidate": thr_s, "control": thr_c, "pass": thr_s <= thr_c + 0.01}
    cov = {k: (pooled(s, k), pooled(c, k)) for k in ("coverage_90", "coverage_50")}
    gates["G5_calibration"] = {
        "coverage_90": {"candidate": cov["coverage_90"][0], "control": cov["coverage_90"][1]},
        "coverage_50": {"candidate": cov["coverage_50"][0], "control": cov["coverage_50"][1]},
        "pass": (
            cov["coverage_90"][0] >= 0.80
            and cov["coverage_90"][0] >= cov["coverage_90"][1] - 0.05
            and cov["coverage_50"][0] >= 0.35
            and cov["coverage_50"][0] >= cov["coverage_50"][1] - 0.10
        ),
    }
    expected = len(ELECTIONS) * len(HORIZONS)
    gated = [r for r in rows if r["gated"]]
    complete = all(sum(1 for r in gated if r["arm"] == arm) == expected for arm in (c, s))
    integrity = all(
        r["seat_total_always_349"] and r["all_finite"] and case_dates_ok(r) for r in gated
    )
    gates["G6_integrity"] = {"complete": complete, "valid_draws_and_dates": integrity,
                             "pass": complete and integrity}
    return {"gates": gates, "decision": "PASS" if all(g["pass"] for g in gates.values()) else "FAIL"}


def protocol_is_agreed(config: ReplayConfig = REPLAY_V1) -> bool:
    return (
        config.agreed_sha256 is not None
        and config.protocol_path.exists()
        and sha256_file(config.protocol_path) == config.agreed_sha256
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score", action="store_true", help="score against outcomes (agreed protocol only)")
    parser.add_argument("--replay", choices=sorted(REPLAYS), default="v1")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)

    config = REPLAYS[args.replay]
    args.output_dir = args.output_dir or config.output_dir
    if args.score and not protocol_is_agreed(config):
        parser.error(f"replay {config.name}'s protocol is not agreed (agreement hash unset or mismatched)")
    baseline = verify_baseline(config) if args.score else None
    truth = load_truth() if args.score else None
    aggregate_file = config.aggregate_file

    with tempfile.TemporaryDirectory(prefix="poll-aggregate-replay-") as tmp:
        roots = {arm: build_data_root(arm, Path(tmp), aggregate_file, config.control_timeseries)
                 for arm in config.arms}
        jobs = _jobs(roots, truth, aggregate_file, config)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(run_case, jobs))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = "cases_scored.csv" if args.score else "cases_dry_run.csv"
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (args.output_dir / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "replay": config.name,
        "protocol_sha256": sha256_file(config.protocol_path) if config.protocol_path.exists() else None,
        "protocol_agreed": protocol_is_agreed(config),
        "scored": args.score,
        "aggregate_sha256": sha256_file(aggregate_file),
        "draws": DRAWS,
        "seed": SEED,
        "cases": len(rows),
        "baseline": baseline,
    }
    (args.output_dir / name.replace(".csv", "_provenance.json")).write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance))
    if args.score:
        decision = evaluate_gates(rows, config.control)
        (args.output_dir / "decision.json").write_text(
            json.dumps({**decision, "provenance": provenance}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"decision": decision["decision"],
                          "gates": {k: v["pass"] for k, v in decision["gates"].items()}}))
    else:
        failures = [r for r in rows if not (r["seat_total_always_349"] and r["all_finite"]
                                            and case_dates_ok(r))]
        print(json.dumps({"dry_run_integrity_failures": len(failures)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
