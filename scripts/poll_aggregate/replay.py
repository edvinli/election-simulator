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
import tempfile
from concurrent.futures import ProcessPoolExecutor
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

PROTOCOL_PATH = REPOSITORY_ROOT / "docs" / "poll_aggregate_replay_protocol.md"
#: Set to the protocol's SHA-256 when it is agreed; ``None`` refuses scoring.
AGREED_PROTOCOL_SHA256: str | None = None

ELECTIONS: tuple[date, ...] = (date(2018, 9, 9), date(2022, 9, 11), date(2026, 9, 13))
HORIZONS: tuple[int, ...] = (112, 84, 56, 28, 14, 7)
DRAWS: int = 20_000
SEED: int = 20260929

CONTROL = "CONTROL_POP"
CANDIDATE = "CANDIDATE_SWEDISHPOLLS"
DIAGNOSTIC = "DIAGNOSTIC_TS_ONLY"
ARMS: tuple[str, ...] = (CONTROL, CANDIDATE, DIAGNOSTIC)

PROCESSED_ROOT = REPOSITORY_ROOT / "data" / "processed"
MANDATES_FILE = PROCESSED_ROOT / "mandates" / "historical_certified_mandates.csv"
DEFAULT_REPLAY_DIR = DEFAULT_OUTPUT_DIR / "replay_v1"
THRESHOLD_PCT = 4.0

INDIVIDUAL_POLL_FIELDS = (
    "poll_id", "pollster", "pollster_original", "interview_start", "interview_end",
    "publication_date", "party", "support", "source_value", "support_status", "sample_size",
    "poll_method", "source_url", "retrieved_at", "metadata_source_url", "metadata_retrieved_at",
    "metadata_match_status", "metadata_row_source_references_json",
)


def write_swedishpolls_individual_polls(dest: Path, polls_file: Path = DEFAULT_POLLS_FILE) -> int:
    """SwedishPolls polls that pass the aggregate's cleaning, in individual_polls.csv's schema.

    Sample sizes are the reported ones (blank when unreported); the
    aggregate's overlap discount and imputation are its own concerns.
    """

    elections = load_elections(DEFAULT_ELECTION_RESULTS_FILE, [DEFAULT_2026_RESULT_MANIFEST])
    eligible = {o.key for o in load_polls(polls_file, [e.obs_date for e in elections]).polls}
    written = set()
    with polls_file.open(encoding="utf-8", newline="") as src, dest.open(
        "w", encoding="utf-8", newline=""
    ) as out:
        writer = csv.DictWriter(out, fieldnames=INDIVIDUAL_POLL_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in csv.DictReader(src):
            if row["poll_id"] not in eligible or row["party"] not in PARTIES:
                continue
            written.add(row["poll_id"])
            writer.writerow(
                {
                    "poll_id": row["poll_id"],
                    "pollster": row["pollster"],
                    "pollster_original": row["pollster_original"],
                    "interview_start": row["interview_start"],
                    "interview_end": row["interview_end"],
                    "publication_date": row["publication_date"],
                    "party": row["party"],
                    "support": row["support"],
                    "source_value": row["source_value"],
                    "support_status": row["support_status"],
                    "sample_size": row["sample_size"],
                    "source_url": row["dataset_source_url"],
                    "metadata_match_status": "swedishpolls",
                    "metadata_row_source_references_json": "[]",
                }
            )
    return len(written)


def build_data_root(arm: str, parent: Path, aggregate_file: Path) -> Path:
    """A processed-data folder identical to production except the arm's two files."""

    root = parent / arm
    (root / "pollofpolls").mkdir(parents=True)
    for child in PROCESSED_ROOT.iterdir():
        if child.name != "pollofpolls":
            os.symlink(child, root / child.name)
    replaced = {
        CONTROL: set(),
        CANDIDATE: {"pollofpolls_timeseries.csv", "individual_polls.csv"},
        DIAGNOSTIC: {"pollofpolls_timeseries.csv"},
    }[arm]
    for child in (PROCESSED_ROOT / "pollofpolls").iterdir():
        if child.name not in replaced:
            os.symlink(child, root / "pollofpolls" / child.name)
    if "pollofpolls_timeseries.csv" in replaced:
        os.symlink(aggregate_file, root / "pollofpolls" / "pollofpolls_timeseries.csv")
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


def _jobs(roots: dict[str, Path], truth: dict[int, dict] | None) -> list[dict]:
    categories = (*PARTIES, "REST")
    jobs = []
    for arm in ARMS:
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


def evaluate_gates(rows: list[dict]) -> dict:
    """The protocol's section-7 gates on scored case rows."""

    def pooled(arm: str, metric: str, election: str | None = None) -> float:
        return _pooled(rows, arm, metric, election)

    c, s = CONTROL, CANDIDATE
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
        r["seat_total_always_349"] and r["all_finite"] and r["opinion_as_of"] == r["as_of"] for r in gated
    )
    gates["G6_integrity"] = {"complete": complete, "valid_draws_and_dates": integrity,
                             "pass": complete and integrity}
    return {"gates": gates, "decision": "PASS" if all(g["pass"] for g in gates.values()) else "FAIL"}


def protocol_is_agreed() -> bool:
    return AGREED_PROTOCOL_SHA256 is not None and sha256_file(PROTOCOL_PATH) == AGREED_PROTOCOL_SHA256


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score", action="store_true", help="score against outcomes (agreed protocol only)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)

    if args.score and not protocol_is_agreed():
        parser.error("the replay protocol is not agreed (AGREED_PROTOCOL_SHA256 unset or mismatched)")
    truth = load_truth() if args.score else None
    aggregate_file = DEFAULT_OUTPUT_DIR / TIMESERIES_FILENAME

    with tempfile.TemporaryDirectory(prefix="poll-aggregate-replay-") as tmp:
        roots = {arm: build_data_root(arm, Path(tmp), aggregate_file) for arm in ARMS}
        jobs = _jobs(roots, truth)
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
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "protocol_agreed": protocol_is_agreed(),
        "scored": args.score,
        "aggregate_sha256": sha256_file(aggregate_file),
        "draws": DRAWS,
        "seed": SEED,
        "cases": len(rows),
    }
    (args.output_dir / name.replace(".csv", "_provenance.json")).write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance))
    if args.score:
        decision = evaluate_gates(rows)
        (args.output_dir / "decision.json").write_text(
            json.dumps({**decision, "provenance": provenance}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"decision": decision["decision"],
                          "gates": {k: v["pass"] for k, v in decision["gates"].items()}}))
    else:
        failures = [r for r in rows if not (r["seat_total_always_349"] and r["all_finite"]
                                            and r["opinion_as_of"] == r["as_of"])]
        print(json.dumps({"dry_run_integrity_failures": len(failures)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
