"""Offline-first, fail-safe orchestration for a production forecast.

This module is deliberately a thin boundary around already-tested data and
simulation components.  It does not fetch data itself.  The acquisition
pipeline may be run separately when a source refresh is explicitly approved;
this command only consumes the resulting checked-in/processed snapshot.

The mutation order is:

1. validate existing processed inputs (no network and no writes);
2. run the frozen ElectionSimulator Candidate A;
3. validate simulation invariants;
4. append an immutable prospective archive snapshot;
5. publish a complete immutable static version behind one atomic pointer.

If a later stage fails, the previous static publication remains untouched.
An archive append is intentionally not rolled back: the archive is append-only
and must never be rewritten to make a failed publication look successful.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

import numpy as np

from scripts.elections.load import load_election_targets_for_forecasting
from scripts.pollofpolls.normalize import parse_percentage
from scripts.pollofpolls.validate import validation_report
from scripts.prospective_archive.archive import SnapshotCollisionError, write_snapshot
from scripts.simulator.config import MODEL_PARTIES_9, PARLIAMENTARY_PARTIES_8
from scripts.simulator.engine import SimulationResult, simulate_election
from scripts.simulator.pipeline import build_canonical_summary_dict
from scripts.static_exporter import export_static_data
from scripts.simulator.reproducibility import compute_file_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_ROOT = REPOSITORY_ROOT / "data" / "processed"
DEFAULT_ARCHIVE_DIR = DEFAULT_PROCESSED_ROOT / "prospective_forecasts"
DEFAULT_PUBLICATION_DIR = REPOSITORY_ROOT / "files" / "election-simulator"

TIMESERIES_FIELDS = ("date", "M", "L", "C", "KD", "S", "V", "MP", "SD", "FI", "other")
REQUIRED_INPUTS = {
    "poll_timeseries": Path("pollofpolls") / "pollofpolls_timeseries.csv",
    "election_results": Path("elections") / "riksdag_election_results.csv",
    "mandates": Path("mandates") / "historical_certified_mandates.csv",
    "geography_votes": Path("geography") / "constituency_party_votes_2014_2022.csv",
    "geography_electorates": Path("geography") / "constituency_electorates_2014_2026.csv",
    "constituencies_2026": Path("mandates") / "constituencies_2026.csv",
}


class PipelineInputError(ValueError):
    """Raised when a source snapshot is absent or fails validation."""


@dataclass
class PipelineRun:
    """JSON-friendly record of every orchestration stage."""

    status: str
    stages: list[dict[str, Any]] = field(default_factory=list)
    input_manifest: dict[str, Any] | None = None
    simulation_validation: dict[str, Any] | None = None
    snapshot: dict[str, Any] | None = None
    publication_manifest: dict[str, Any] | None = None
    error: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "stages": self.stages,
            "input_manifest": self.input_manifest,
            "simulation_validation": self.simulation_validation,
            "snapshot": self.snapshot,
            "publication_manifest": self.publication_manifest,
            "error": self.error,
        }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _coerce_timeseries(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        for party in TIMESERIES_FIELDS[1:]:
            item[party] = parse_percentage(row.get(party))
        converted.append(item)
    return converted


def _coerce_individual(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    numeric_fields = {"support", "source_value"}
    integer_fields = {"sample_size"}
    for row in rows:
        item: dict[str, Any] = dict(row)
        for field_name in numeric_fields:
            item[field_name] = parse_percentage(row.get(field_name))
        for field_name in integer_fields:
            raw = row.get(field_name)
            item[field_name] = int(raw) if raw not in (None, "", "None") else None
        converted.append(item)
    return converted


def _coerce_supplementary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        item["support"] = parse_percentage(row.get("support"))
        item["source_value"] = parse_percentage(row.get("source_value"))
        raw = row.get("sample_size")
        item["sample_size"] = int(raw) if raw not in (None, "", "None") else None
        converted.append(item)
    return converted


def validate_existing_inputs(
    processed_root: Path | str = DEFAULT_PROCESSED_ROOT,
    *,
    include_supplementary: bool = True,
) -> dict[str, Any]:
    """Validate the saved source snapshot without fetching or writing.

    Warnings from the project validators are retained in the returned report,
    but only validation errors fail the pipeline.  Every required file is
    hashed so the simulator manifest and publication can identify the exact
    input snapshot used.
    """

    root = Path(processed_root).resolve()
    resolved = {name: root / relative for name, relative in REQUIRED_INPUTS.items()}
    missing = [name for name, path in resolved.items() if not path.is_file()]
    if missing:
        raise PipelineInputError(
            "Missing required processed inputs: "
            + ", ".join(f"{name} ({resolved[name]})" for name in missing)
        )

    poll_dir = root / "pollofpolls"
    timeseries_path = resolved["poll_timeseries"]
    individual_path = poll_dir / "individual_polls.csv"
    supplementary_path = poll_dir / "swedishpolls_individual_polls.csv"
    if not individual_path.is_file():
        raise PipelineInputError(f"Missing normalized individual polls: {individual_path}")
    if include_supplementary and not supplementary_path.is_file():
        raise PipelineInputError(f"Missing normalized supplementary polls: {supplementary_path}")

    timeseries = _coerce_timeseries(_read_csv(timeseries_path))
    individual = _coerce_individual(_read_csv(individual_path))
    supplementary = (
        _coerce_supplementary(_read_csv(supplementary_path)) if include_supplementary else None
    )
    report = validation_report(timeseries, individual, supplementary)
    if not report["valid"]:
        raise PipelineInputError(
            f"Processed poll inputs failed validation with {report['error_count']} error(s)"
        )

    # Load and validate the canonical election file through the same loader
    # used by hindcasts.  This catches malformed integer totals, unknown party
    # labels, and invalid dates before any simulation is started.
    election_targets = load_election_targets_for_forecasting(resolved["election_results"])
    if not election_targets:
        raise PipelineInputError("Canonical election results contain no election targets")

    hashes = {name: compute_file_sha256(path) for name, path in resolved.items()}
    hashes["individual_polls"] = compute_file_sha256(individual_path)
    if supplementary_path.exists():
        hashes["supplementary_polls"] = compute_file_sha256(supplementary_path)
    return {
        "status": "OFFLINE_VALIDATED",
        "network_access": "none",
        "processed_root": str(root),
        "required_inputs": {name: str(path) for name, path in resolved.items()},
        "input_sha256": hashes,
        "poll_validation": report,
        "election_target_dates": sorted(str(day) for day in election_targets),
        "warning_policy": "warnings are retained; validation errors fail closed",
    }


def validate_simulation_result(result: SimulationResult) -> dict[str, Any]:
    """Validate hard output invariants before archive or publication writes."""

    if not isinstance(result, SimulationResult):
        raise ValueError("Simulation runner did not return a SimulationResult")
    votes = np.asarray(result.vote_shares_matrix)
    seats = np.asarray(result.seats_matrix)
    flags = np.asarray(result.threshold_flags)
    if votes.ndim != 2 or votes.shape[1] != len(MODEL_PARTIES_9):
        raise ValueError(f"Vote matrix must have shape (N, {len(MODEL_PARTIES_9)})")
    if seats.ndim != 2 or seats.shape != (votes.shape[0], len(PARLIAMENTARY_PARTIES_8)):
        raise ValueError("Seat matrix shape does not match vote samples and party order")
    if flags.shape != (votes.shape[0], len(PARLIAMENTARY_PARTIES_8)):
        raise ValueError("Threshold flag shape does not match vote samples")
    if not np.isfinite(votes).all() or not np.isfinite(seats).all():
        raise ValueError("Simulation output contains non-finite values")
    if np.any(votes < -1e-10):
        raise ValueError("Simulation output contains negative vote shares")
    vote_totals = votes.sum(axis=1)
    max_vote_sum_error = float(np.max(np.abs(vote_totals - 100.0)))
    if max_vote_sum_error > 1e-7:
        raise ValueError(f"Vote-share total invariant failed (max error {max_vote_sum_error:g})")
    if not np.issubdtype(seats.dtype, np.integer) or np.any(seats < 0):
        raise ValueError("Seat output must be non-negative integers")
    seat_totals = seats.sum(axis=1)
    if not np.all(seat_totals == 349):
        raise ValueError("Seat total invariant failed: every sample must contain 349 seats")
    if result.summary.total_samples != votes.shape[0]:
        raise ValueError("Summary sample count does not match simulation arrays")
    # REST is the ninth vote category and is deliberately absent from the seat
    # matrix.  The summary's zero seat surface is the authoritative check.
    if result.summary.parties["REST"].seats_mean != 0.0:
        raise ValueError("REST must remain an aggregate ineligible category")
    return {
        "status": "VALID",
        "samples": int(votes.shape[0]),
        "vote_party_order": list(MODEL_PARTIES_9),
        "seat_party_order": list(PARLIAMENTARY_PARTIES_8),
        "max_vote_sum_error": max_vote_sum_error,
        "seat_total_min": int(np.min(seat_totals)),
        "seat_total_max": int(np.max(seat_totals)),
        "rest_seat_surface": "zero_and_absent_from_allocator_contract",
    }


def _temporary_canonical_artifacts(result: SimulationResult, directory: Path) -> tuple[Path, Path]:
    """Materialize the exact run payload for the archive's hash linkage."""

    summary = build_canonical_summary_dict(result)
    canonical = directory / "simulation_summary.json"
    sidecar = directory / "deterministic_payload.sha256"
    canonical.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    sidecar.write_text(summary["deterministic_payload_sha256"] + "\n", encoding="utf-8")
    return canonical, sidecar


def _load_prior_snapshot(archive_dir: Path | str, current_as_of: str) -> dict[str, Any] | None:
    """Load the latest immutable snapshot strictly before the current as-of date."""

    root = Path(archive_dir)
    index_path = root / "index.json"
    if not index_path.exists():
        return None
    with index_path.open(encoding="utf-8") as handle:
        index = json.load(handle)
    entries = index.get("snapshots", [])
    prior = [entry for entry in entries if str(entry.get("snapshot_date", "")) < str(current_as_of)]
    if not prior:
        return None
    prior_entry = max(prior, key=lambda entry: str(entry["snapshot_date"]))
    relative = prior_entry.get("path")
    if not relative:
        raise PipelineInputError("Archive index prior entry has no snapshot path")
    snapshot_path = root / str(relative)
    if not snapshot_path.is_file():
        raise PipelineInputError(f"Archive index points to missing prior snapshot: {snapshot_path}")
    with snapshot_path.open(encoding="utf-8") as handle:
        snapshot = json.load(handle)
    if not isinstance(snapshot, dict):
        raise PipelineInputError(f"Prior snapshot is not a JSON object: {snapshot_path}")
    return snapshot


def _next_stage_name(run: PipelineRun, *, append_archive: bool, export_publication: bool) -> str:
    """Name the first gate that has not completed for failure reporting."""

    completed = {stage["name"] for stage in run.stages}
    ordered = ["input_validation", "frozen_simulation"]
    # Both certified outputs (the static publication and the prospective
    # archive) carry source provenance.  Keep the failure stage aligned with
    # the gate below when an archive-only run is requested.
    if append_archive or export_publication:
        ordered.append("source_certification")
    if append_archive:
        ordered.append("prospective_archive_append")
    if export_publication:
        ordered.append("static_publication")
    return next((name for name in ordered if name not in completed), "pipeline")


def run_publication_pipeline(
    *,
    as_of: str | None = None,
    election_date: str = "2026-09-13",
    samples: int = 100_000,
    seed: int = 12_345,
    baseline_year: int = 2022,
    processed_root: Path | str = DEFAULT_PROCESSED_ROOT,
    archive_dir: Path | str = DEFAULT_ARCHIVE_DIR,
    publication_dir: Path | str = DEFAULT_PUBLICATION_DIR,
    generated_at_utc: str | None = None,
    append_archive: bool = True,
    export_publication: bool = True,
    include_supplementary: bool = True,
    simulation_runner: Callable[..., SimulationResult] | None = None,
) -> PipelineRun:
    """Run the offline-first production contract and return stage evidence."""

    run = PipelineRun(status="FAILED")
    generated = generated_at_utc or datetime.now(timezone.utc).isoformat()
    runner = simulation_runner or simulate_election
    try:
        selected_processed_root = Path(processed_root).resolve()
        canonical_processed_root = DEFAULT_PROCESSED_ROOT.resolve()
        if selected_processed_root != canonical_processed_root:
            raise PipelineInputError(
                "Custom processed_root is not supported by the frozen simulator; "
                f"use {canonical_processed_root} so validated inputs and simulation inputs cannot diverge"
            )
        run.input_manifest = validate_existing_inputs(
            selected_processed_root,
            include_supplementary=include_supplementary,
        )
        run.stages.append({"name": "input_validation", "status": "PASS", "detail": run.input_manifest["status"]})

        result = runner(
            as_of=as_of,
            election_date=election_date,
            samples=samples,
            seed=seed,
            baseline_year=baseline_year,
        )
        run.simulation_validation = validate_simulation_result(result)
        run.stages.append({"name": "frozen_simulation", "status": "PASS", "detail": run.simulation_validation})

        requires_source_certification = append_archive or export_publication
        if requires_source_certification and result.manifest.get("source_worktree_clean") is not True:
            raise PipelineInputError(
                "Certified archive/publication requires source_worktree_clean to be the boolean true"
            )
        if requires_source_certification:
            run.stages.append({"name": "source_certification", "status": "PASS"})

        prior_snapshot = _load_prior_snapshot(archive_dir, result.summary.as_of)

        if append_archive:
            with tempfile.TemporaryDirectory(prefix="election-simulator-canonical-") as temp_name:
                canonical, sidecar = _temporary_canonical_artifacts(result, Path(temp_name))
                snapshot_path, index_path, snapshot = write_snapshot(
                    result,
                    archive_dir=archive_dir,
                    generated_at_utc=generated,
                    canonical_artifact_path=canonical,
                    canonical_payload_hash_path=sidecar,
                )
            run.snapshot = {
                "snapshot_path": str(snapshot_path),
                "index_path": str(index_path),
                "snapshot_id": snapshot["snapshot_id"],
                "deterministic_payload_sha256": snapshot["deterministic_payload_sha256"],
                "prior_snapshot_id": prior_snapshot.get("snapshot_id") if prior_snapshot else None,
            }
            run.stages.append({"name": "prospective_archive_append", "status": "PASS", "detail": run.snapshot})
        else:
            run.stages.append({"name": "prospective_archive_append", "status": "SKIPPED"})

        if export_publication:
            manifest = export_static_data(
                result,
                output_dir=publication_dir,
                generated_at_utc=generated,
                calibration_dir=Path(processed_root),
                prior_snapshot=prior_snapshot,
            )
            run.publication_manifest = manifest
            run.stages.append({"name": "static_publication", "status": "PASS", "detail": manifest})
        else:
            run.stages.append({"name": "static_publication", "status": "SKIPPED"})
        run.status = "PUBLISHED" if export_publication else "SIMULATED"
    except SnapshotCollisionError as exc:
        # Collision is a normal fail-closed outcome for an append-only daily
        # archive.  Do not attempt an export using a snapshot that was not
        # appended; the previous publication remains untouched.
        run.status = "COLLISION"
        run.stages.append({"name": "prospective_archive_append", "status": "FAIL_COLLISION"})
        run.error = {"type": type(exc).__name__, "message": str(exc)}
    except Exception as exc:  # noqa: BLE001 - return a machine-readable stage failure
        run.status = "FAILED"
        run.stages.append({
            "name": _next_stage_name(
                run,
                append_archive=append_archive,
                export_publication=export_publication,
            ),
            "status": "FAIL",
        })
        run.error = {"type": type(exc).__name__, "message": str(exc)}
    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the offline-first ElectionSimulator publication pipeline"
    )
    parser.add_argument("--as-of", default=None, help="Poll cutoff date (YYYY-MM-DD)")
    parser.add_argument("--election-date", default="2026-09-13")
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=12_345)
    parser.add_argument("--baseline-year", type=int, default=2022)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--publication-dir", type=Path, default=DEFAULT_PUBLICATION_DIR)
    parser.add_argument("--generated-at-utc", default=None)
    parser.add_argument("--no-archive", action="store_true", help="Do not append a prospective snapshot")
    parser.add_argument("--no-publication", action="store_true", help="Run/validate but do not export static JSON")
    parser.add_argument("--no-supplementary", action="store_true", help="Skip optional SwedishPolls validation")
    parser.add_argument(
        "--live-fetch",
        action="store_true",
        help="Rejected: source acquisition is a separately approved operation",
    )
    args = parser.parse_args(argv)
    if args.live_fetch:
        parser.error("Live fetching is intentionally unavailable in this fail-safe command; refresh inputs separately")
    run = run_publication_pipeline(
        as_of=args.as_of,
        election_date=args.election_date,
        samples=args.samples,
        seed=args.seed,
        baseline_year=args.baseline_year,
        processed_root=args.processed_root,
        archive_dir=args.archive_dir,
        publication_dir=args.publication_dir,
        generated_at_utc=args.generated_at_utc,
        append_archive=not args.no_archive,
        export_publication=not args.no_publication,
        include_supplementary=not args.no_supplementary,
    )
    print(json.dumps(run.to_dict(), indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if run.status in {"PUBLISHED", "SIMULATED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
