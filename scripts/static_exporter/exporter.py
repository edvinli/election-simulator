"""Export compact, validated static forecast JSON atomically.

The exporter is deliberately downstream of the frozen Python simulator.  It
does not run Monte Carlo in a browser and it never writes into the immutable
prospective archive.  A complete publication is assembled in a sibling
temporary directory, validated, and published as an immutable version behind
one atomically replaced current.json pointer.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

from scripts.simulator.config import MODEL_PARTIES_9, PARLIAMENTARY_PARTIES_8
from scripts.simulator.pipeline import build_canonical_summary_dict
from scripts.simulator.reproducibility import (
    GENERATION_ID_PATTERN,
    SOURCE_REPOSITORY,
    build_generation_id,
    compute_file_sha256,
    require_certified_source_provenance,
    resolve_source_repository,
)


# 1.0 is the pre-extraction publication schema.  1.1 adds ``source_repository``
# to metadata and manifests.  Validators accept both so historical 1.0
# publications stay readable and are never rewritten; only 1.1 is written.
PUBLICATION_SCHEMA_VERSION = "1.1"
SUPPORTED_PUBLICATION_SCHEMA_VERSIONS: tuple[str, ...] = ("1.0", "1.1")
PUBLICATION_FILES: tuple[str, ...] = (
    "forecast.json",
    "parties.json",
    "seats.json",
    "groups.json",
    "calibration.json",
    "metadata.json",
)

# The publication is a public artifact, so it must never carry the local
# filesystem layout of whichever machine generated it.  Calibration sources are
# addressed by their stable logical name under the processed data root instead.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOGICAL_CALIBRATION_ROOT = "data/processed"
CALIBRATION_SOURCE_RELATIVE_PARTS: dict[str, tuple[str, ...]] = {
    "seat_hindcast": ("seat_hindcasts", "seat_hindcast_summary.json"),
    "vote_share_hindcast": ("vote_share_calibration", "vote_share_summary_2018_2022.json"),
    "pop_head_to_head": ("pop_baseline_benchmark", "benchmark_report.json"),
}


def _public_source_path(path: Path, relative_parts: Sequence[str]) -> str:
    """Return a stable POSIX path safe to publish for a calibration artifact.

    A file inside the repository is serialised repo-relative.  Anything else --
    a temporary directory in tests, a mounted artifact store -- falls back to
    the logical ``data/processed`` name of that artifact.  Neither form can
    contain a local absolute path, and both are byte-identical across machines
    so the deterministic publication hash stays reproducible.
    """

    try:
        return path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return "/".join((LOGICAL_CALIBRATION_ROOT, *relative_parts))


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strip_runtime_timestamps(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_runtime_timestamps(item)
            for key, item in value.items()
            if key not in {"generated_at_utc", "published_at_utc", "updated_at_utc"}
        }
    if isinstance(value, list):
        return [_strip_runtime_timestamps(item) for item in value]
    return value


def _histogram(values: np.ndarray, *, lower: float, upper: float, width: float) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Cannot export an empty distribution")
    edges = np.arange(lower, upper + width, width, dtype=np.float64)
    if edges[-1] < upper:
        edges = np.append(edges, upper)
    counts, actual_edges = np.histogram(arr, bins=edges)
    return {
        "bin_edges": [round(float(x), 6) for x in actual_edges],
        "counts": [int(x) for x in counts],
        "total": int(arr.size),
        "quantiles": {
            "p05": round(float(np.quantile(arr, 0.05)), 6),
            "p25": round(float(np.quantile(arr, 0.25)), 6),
            "p50": round(float(np.quantile(arr, 0.50)), 6),
            "p75": round(float(np.quantile(arr, 0.75)), 6),
            "p95": round(float(np.quantile(arr, 0.95)), 6),
        },
    }


def _representative_seat_allocation(result: Any) -> dict[str, Any]:
    """Select one coherent simulated parliament for the public display.

    Marginal party medians are useful summaries, but they need not form a
    legal parliament when added together.  The representative display is the
    simulated row with minimum squared distance to the vector of marginal
    medians.  ``argmin`` is deterministic and therefore stable for a fixed
    simulation payload; every row has already passed the 349-seat invariant in
    the production pipeline.
    """

    seats = np.asarray(result.seats_matrix)
    if seats.ndim != 2 or seats.shape[1] != len(PARLIAMENTARY_PARTIES_8) or seats.shape[0] == 0:
        raise ValueError("Cannot select a representative allocation from an invalid seat matrix")
    if not np.issubdtype(seats.dtype, np.integer) or np.any(seats < 0):
        raise ValueError("Cannot select a representative allocation from invalid seat values")
    seat_totals = np.sum(seats, axis=1)
    if not np.all(seat_totals == 349):
        raise ValueError("Cannot select a representative allocation before the 349-seat invariant holds")
    marginal_medians = np.median(seats, axis=0)
    distances = np.sum((seats.astype(np.float64) - marginal_medians) ** 2, axis=1)
    draw_index = int(np.argmin(distances))
    allocation = {
        party: int(seats[draw_index, index])
        for index, party in enumerate(PARLIAMENTARY_PARTIES_8)
    }
    total_seats = int(sum(allocation.values()))
    if total_seats != 349:
        raise ValueError("Representative allocation does not contain exactly 349 seats")
    return {
        "method": "joint_simulation_draw_closest_to_marginal_medians",
        "draw_index": draw_index,
        "seats": allocation,
        "total_seats": total_seats,
    }


def _build_contracts(
    result: Any,
    *,
    generated_at_utc: str,
    calibration_dir: Path | None,
    prior_snapshot: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    summary = build_canonical_summary_dict(result)
    manifest = dict(result.manifest)
    if manifest.get("source_worktree_clean") is not True:
        raise ValueError(
            "Certified static publication requires source_worktree_clean to be the boolean true"
        )
    deterministic_payload_sha256 = summary["deterministic_payload_sha256"]
    # ``build_canonical_summary_dict`` is intentionally compact for the
    # archive.  The publication contract additionally exposes the inner
    # quantiles needed to display all three predictive intervals.
    parties = {party: dict(value) for party, value in summary["parties"].items()}
    for party in MODEL_PARTIES_9:
        party_summary = result.summary.parties[party]
        parties[party].update({
            "vote_share_p10": round(float(party_summary.vote_share_p10 * 100.0), 3),
            "vote_share_p25": round(float(party_summary.vote_share_p25 * 100.0), 3),
            "vote_share_p75": round(float(party_summary.vote_share_p75 * 100.0), 3),
            "vote_share_p90": round(float(party_summary.vote_share_p90 * 100.0), 3),
            "seats_p10": int(party_summary.seats_p10),
            "seats_p25": int(party_summary.seats_p25),
            "seats_p75": int(party_summary.seats_p75),
            "seats_p90": int(party_summary.seats_p90),
        })
    for party in PARLIAMENTARY_PARTIES_8:
        if party not in parties:
            raise ValueError(f"Canonical result is missing party {party}")
    if "REST" not in parties:
        raise ValueError("Canonical result is missing aggregate REST")

    party_rows: list[dict[str, Any]] = []
    for party in MODEL_PARTIES_9:
        row = dict(parties[party])
        row["party"] = party
        row["eligible_for_national_threshold"] = party in PARLIAMENTARY_PARTIES_8
        row["threshold_probability_defined"] = party in PARLIAMENTARY_PARTIES_8
        party_rows.append(row)

    vote_distributions = {
        party: _histogram(result.vote_shares_matrix[:, index], lower=0.0, upper=100.0, width=0.25)
        for index, party in enumerate(MODEL_PARTIES_9)
    }
    seat_distributions = {
        party: _histogram(result.seats_matrix[:, index], lower=0.0, upper=349.0, width=1.0)
        for index, party in enumerate(PARLIAMENTARY_PARTIES_8)
    }
    representative_allocation = _representative_seat_allocation(result)
    change_since_prior: dict[str, Any]
    if prior_snapshot is None:
        change_since_prior = {
            "status": "NOT_AVAILABLE_NO_PRIOR_SNAPSHOT",
            "prior_as_of": None,
            "prior_snapshot_id": None,
            "prior_deterministic_payload_sha256": None,
            "vote_share_median_change_pp": {},
            "seat_median_change": {},
        }
    else:
        prior_vote_summary = prior_snapshot.get("national_vote_summary", {})
        prior_seat_summary = prior_snapshot.get("seat_summary", {})
        vote_changes: dict[str, float] = {}
        for party in MODEL_PARTIES_9:
            current_value = parties[party].get("vote_share_median")
            prior_value = prior_vote_summary.get(party, {}).get("vote_share_median")
            if current_value is not None and prior_value is not None:
                vote_changes[party] = round(float(current_value) - float(prior_value), 3)
        seat_changes: dict[str, int] = {}
        for party in PARLIAMENTARY_PARTIES_8:
            current_value = parties[party].get("seats_median")
            prior_value = prior_seat_summary.get(party, {}).get("median")
            if current_value is not None and prior_value is not None:
                seat_changes[party] = int(current_value) - int(prior_value)
        change_since_prior = {
            "status": "AVAILABLE",
            "prior_as_of": prior_snapshot.get("as_of"),
            "prior_snapshot_id": prior_snapshot.get("snapshot_id"),
            "prior_deterministic_payload_sha256": prior_snapshot.get("deterministic_payload_sha256"),
            "vote_share_median_change_pp": vote_changes,
            "seat_median_change": seat_changes,
        }
    groups = {
        "tido": result.summarize_group(["M", "SD", "KD", "L"]).__dict__,
        "red_green_center": result.summarize_group(["S", "V", "MP", "C"]).__dict__,
    }
    for row in groups.values():
        row["parties"] = list(row["parties"])

    calibration: dict[str, Any] = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "status": "AVAILABLE_IF_ARTIFACTS_EXIST",
        "evidence_type": "retrospective_historical_not_holdout",
        "source_files": {},
    }
    if calibration_dir is not None:
        processed_root = calibration_dir
        calibration_paths = {
            key: processed_root.joinpath(*relative_parts)
            for key, relative_parts in CALIBRATION_SOURCE_RELATIVE_PARTS.items()
        }
        for key, path in calibration_paths.items():
            path = path.resolve()
            if path.exists():
                with path.open(encoding="utf-8") as handle:
                    source_value = json.load(handle)
                # Keep the publication compact: retain status/aggregates, not
                # per-draw or per-case evidence in the website contract.
                if key == "pop_head_to_head":
                    source_value = {
                        "benchmark_status": source_value.get("benchmark_status"),
                        "aggregate_by_evaluation": source_value.get("aggregate_by_evaluation"),
                        "aggregate_by_horizon": source_value.get("aggregate_by_horizon"),
                        "threshold_support_diagnostic": source_value.get("threshold_support_diagnostic"),
                        "underdispersion_diagnostic": source_value.get("underdispersion_diagnostic"),
                        "comparison_decision": source_value.get("comparison_decision"),
                        "evidence_type": source_value.get("evidence_type"),
                    }
                else:
                    source_value = {
                        "evidence_type": calibration["evidence_type"],
                        "summary": source_value.get("summary", source_value.get("by_model_overall", source_value)),
                    }
                calibration["source_files"][key] = {
                    "path": _public_source_path(path, CALIBRATION_SOURCE_RELATIVE_PARTS[key]),
                    "sha256": compute_file_sha256(path),
                    "summary": source_value,
                }
    if not calibration["source_files"]:
        calibration["status"] = "NOT_AVAILABLE"
        calibration["reason"] = "No validation summary artifacts were found"

    forecast = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "as_of": summary["as_of"],
        "election_date": summary["election_date"],
        "model": {"name": "ElectionSimulator", "version": manifest.get("model_version"), "candidate": "A"},
        "total_samples": int(summary["total_samples"]),
        "parties": parties,
        "threshold_probabilities_4pct": {
            party: parties[party]["prob_above_4pct"] for party in PARLIAMENTARY_PARTIES_8
        },
        "groups": summary["blocs"],
        "change_since_prior": change_since_prior,
        "deterministic_payload_sha256": deterministic_payload_sha256,
        "rest_semantics": "REST is aggregate vote mass for modeled-as-ineligible parties; it cannot independently qualify or receive seats.",
    }
    party_contract = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "party_order": list(MODEL_PARTIES_9),
        "parties": party_rows,
        "rest_semantics": forecast["rest_semantics"],
        "deterministic_payload_sha256": deterministic_payload_sha256,
    }
    seat_contract = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "party_order": list(PARLIAMENTARY_PARTIES_8),
        "total_seats": 349,
        "seat_distributions": seat_distributions,
        "seat_summary": {
            party: {
                "mean": parties[party]["seats_mean"],
                "median": parties[party]["seats_median"],
                "p05": parties[party]["seats_p05"],
                "p95": parties[party]["seats_p95"],
            }
            for party in PARLIAMENTARY_PARTIES_8
        },
        "representative_allocation": representative_allocation,
        "rest_semantics": forecast["rest_semantics"],
        "deterministic_payload_sha256": deterministic_payload_sha256,
    }
    group_contract = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "majority_threshold": 175,
        "groups": groups,
        "note": "Groups are configurable summaries over parliamentary-party seat draws; REST is never included as an eligible party.",
        "deterministic_payload_sha256": deterministic_payload_sha256,
    }
    calibration["deterministic_payload_sha256"] = deterministic_payload_sha256
    metadata = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc,
        "as_of": summary["as_of"],
        "election_date": summary["election_date"],
        "model": {"name": "ElectionSimulator", "version": manifest.get("model_version"), "candidate": "A"},
        "source_repository": SOURCE_REPOSITORY,
        "source_git_commit": manifest.get("source_git_commit", manifest.get("git_commit")),
        "source_worktree_clean": manifest.get("source_worktree_clean"),
        "input_hashes": {
            key: manifest.get(key)
            for key in ("poll_data_hash", "election_data_hash", "mandate_data_hash", "geography_data_hash", "model_config_hash")
        },
        "deterministic_payload_sha256": deterministic_payload_sha256,
        "rest_semantics": forecast["rest_semantics"],
        "interval_semantics": "Central empirical predictive intervals at 50%, 80%, and 90%; they are not confidence intervals.",
        "validation_note": "Historical validation is retrospective and not independent holdout validation.",
    }
    return {
        "forecast.json": forecast,
        "parties.json": party_contract,
        "seats.json": seat_contract,
        "groups.json": group_contract,
        "calibration.json": calibration,
        "metadata.json": metadata,
    }


def validate_publication_contract(contracts: Mapping[str, Mapping[str, Any]]) -> None:
    """Validate all compact publication contracts before any directory swap."""
    if set(contracts) != set(PUBLICATION_FILES):
        raise ValueError(f"Publication files must be exactly {list(PUBLICATION_FILES)}")
    for name, value in contracts.items():
        if value.get("schema_version") not in SUPPORTED_PUBLICATION_SCHEMA_VERSIONS:
            raise ValueError(f"{name} has unsupported schema version")
    forecast = contracts["forecast.json"]
    parties = contracts["parties.json"]
    seats = contracts["seats.json"]
    groups = contracts["groups.json"]
    if parties["party_order"] != list(MODEL_PARTIES_9):
        raise ValueError("parties.json has incorrect canonical party order")
    if seats["party_order"] != list(PARLIAMENTARY_PARTIES_8) or seats["total_seats"] != 349:
        raise ValueError("seats.json has incorrect party order or seat total")
    if "REST" not in parties["party_order"] or "REST" in seats["party_order"]:
        raise ValueError("REST must be present in vote contracts and absent from seat contracts")
    rest_rows = [row for row in parties["parties"] if row["party"] == "REST"]
    if len(rest_rows) != 1 or rest_rows[0]["eligible_for_national_threshold"]:
        raise ValueError("REST must be explicitly marked ineligible")
    if forecast["threshold_probabilities_4pct"].keys() != set(PARLIAMENTARY_PARTIES_8):
        raise ValueError("Threshold probabilities must cover exactly the eight parliamentary parties")
    if groups.get("majority_threshold") != 175:
        raise ValueError("Group majority threshold must be 175")
    metadata = contracts["metadata.json"]
    if not metadata.get("deterministic_payload_sha256"):
        raise ValueError("metadata.json must link the deterministic simulation payload")
    # An unresolvable Git commit and a dirty worktree are both hard
    # certification failures; neither may reach a published artifact.
    require_certified_source_provenance(metadata)
    # Missing on historical 1.0 artifacts, where it means the original
    # repository; required on everything this exporter writes.
    if metadata.get("schema_version") != "1.0" and metadata.get("source_repository") != SOURCE_REPOSITORY:
        raise ValueError(f"metadata.json must record source_repository={SOURCE_REPOSITORY!r}")
    payload_hash = metadata["deterministic_payload_sha256"]
    for name, contract in contracts.items():
        if contract.get("deterministic_payload_sha256") != payload_hash:
            raise ValueError(f"{name} does not link the common deterministic simulation payload")
    representative = seats.get("representative_allocation")
    if not isinstance(representative, dict):
        raise ValueError("seats.json must include a representative joint allocation")
    representative_seats = representative.get("seats")
    if not isinstance(representative_seats, dict):
        raise ValueError("Representative allocation must contain a seat mapping")
    if set(representative_seats) != set(PARLIAMENTARY_PARTIES_8):
        raise ValueError("Representative allocation must cover exactly the eight parliamentary parties")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in representative_seats.values()):
        raise ValueError("Representative allocation seats must be non-negative integers")
    if sum(representative_seats.values()) != 349 or representative.get("total_seats") != 349:
        raise ValueError("Representative allocation must contain exactly 349 seats")


def _validate_publication_version(
    root: Path,
    *,
    expected_generation: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate one immutable version directory and its manifest hashes."""

    contracts: dict[str, dict[str, Any]] = {}
    for filename in PUBLICATION_FILES:
        path = root / filename
        # Every published version file must be a real file.  A symlink would
        # not survive static hosting and would break the immutability promise.
        if path.is_symlink():
            raise ValueError(f"Published contract must be a real file, not a symlink: {path}")
        if not path.is_file():
            raise ValueError(f"Published contract is missing {path}")
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"Published contract is not a JSON object: {path}")
        contracts[filename] = value
    validate_publication_contract(contracts)
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError(f"Published manifest must be a real file, not a symlink: {manifest_path}")
    if not manifest_path.is_file():
        raise ValueError(f"Published manifest is missing: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") not in SUPPORTED_PUBLICATION_SCHEMA_VERSIONS:
        raise ValueError("Published manifest has unsupported schema version")
    if manifest.get("schema_version") != contracts["metadata.json"].get("schema_version"):
        raise ValueError("Published manifest and metadata disagree on schema version")
    # Historical 1.0 artifacts legitimately carry no source_repository and are
    # read as belonging to the original repository; they are never rewritten.
    if resolve_source_repository(manifest.get("source_repository")) != resolve_source_repository(
        contracts["metadata.json"].get("source_repository")
    ):
        raise ValueError("Published manifest and metadata disagree on source repository")
    if manifest.get("publication_state") != "COMPLETE":
        raise ValueError("Published manifest is not marked COMPLETE")
    generation = manifest.get("publication_generation")
    if not isinstance(generation, str) or not generation:
        raise ValueError("Published manifest has no immutable publication generation")
    if root.name != generation:
        raise ValueError("Publication version path does not match the manifest generation")
    if expected_generation is not None and generation != expected_generation:
        raise ValueError("Current pointer does not match the manifest generation")
    actual_manifest_sha256 = compute_file_sha256(manifest_path)
    if expected_manifest_sha256 is not None and actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError("Current pointer manifest hash does not match the active version")
    if manifest.get("source_worktree_clean") is not True:
        raise ValueError("Published manifest is not certified from a clean source worktree")
    expected_file_hashes = {
        filename: compute_file_sha256(root / filename) for filename in PUBLICATION_FILES
    }
    if manifest.get("publication_files") != expected_file_hashes:
        raise ValueError("Published file hashes do not match manifest")
    expected_deterministic_hashes = {
        filename: _sha256_bytes(_canonical_bytes(_strip_runtime_timestamps(contracts[filename])))
        for filename in PUBLICATION_FILES
    }
    if manifest.get("deterministic_content_hashes") != expected_deterministic_hashes:
        raise ValueError("Published deterministic content hashes do not match manifest")
    expected_manifest_hash = _sha256_bytes(_canonical_bytes(expected_deterministic_hashes))
    if manifest.get("deterministic_content_sha256") != expected_manifest_hash:
        raise ValueError("Published deterministic manifest hash does not match content")
    if manifest.get("deterministic_payload_sha256") != contracts["metadata.json"].get("deterministic_payload_sha256"):
        raise ValueError("Published payload hash is not linked through metadata")
    return manifest


def validate_publication_version(
    version_dir: Path | str,
    *,
    expected_generation: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate one immutable version directory in isolation.

    Public entry point for consumers that address a version directly rather
    than through ``current.json`` — notably the cross-repository site
    publisher, which validates the source and the copied destination
    independently.
    """

    return _validate_publication_version(
        Path(version_dir),
        expected_generation=expected_generation,
        expected_manifest_sha256=expected_manifest_sha256,
    )


def validate_published_directory(output_dir: Path | str) -> dict[str, Any]:
    """Validate the active immutable version addressed by ``current.json``."""

    root = Path(output_dir)
    pointer_path = root / "current.json"
    if not pointer_path.is_file():
        if os.path.lexists(pointer_path):
            raise ValueError("Current publication pointer is not a regular readable file")
        # Backward-compatible read path for an already materialized version.
        # New canonical publications always contain current.json at their
        # stable root, while callers may validate a version directly.
        return _validate_publication_version(root)
    with pointer_path.open(encoding="utf-8") as handle:
        pointer = json.load(handle)
    if not isinstance(pointer, dict):
        raise ValueError("Current publication pointer is not a JSON object")
    if pointer.get("schema_version") not in SUPPORTED_PUBLICATION_SCHEMA_VERSIONS:
        raise ValueError("Current publication pointer has unsupported schema version")
    if pointer.get("publication_state") != "COMPLETE":
        raise ValueError("Current publication pointer is not marked COMPLETE")
    generation = pointer.get("publication_generation")
    relative_path = pointer.get("path")
    manifest_sha256 = pointer.get("manifest_sha256")
    if not isinstance(generation, str) or not generation:
        raise ValueError("Current publication pointer has no generation")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("Current publication pointer has no version path")
    if not isinstance(manifest_sha256, str) or not manifest_sha256:
        raise ValueError("Current publication pointer has no manifest hash")
    relative = Path(relative_path)
    if relative.is_absolute() or relative.parts[:1] != ("versions",) or len(relative.parts) != 2:
        raise ValueError("Current publication pointer path must address one direct versions child")
    versions_root = (root / "versions").resolve()
    version_path = (root / relative).resolve()
    if version_path.parent != versions_root:
        raise ValueError("Current publication pointer escapes the immutable versions directory")
    return _validate_publication_version(
        version_path,
        expected_generation=generation,
        expected_manifest_sha256=manifest_sha256,
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    """Durably flush a directory entry on filesystems that support it."""

    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_replace_pointer(pointer: Path, output: Path) -> None:
    """Replace the consumer pointer in one filesystem operation."""

    os.replace(pointer, output)


def _write_pointer(output: Path, *, generation: str, version_path: Path) -> None:
    """Atomically commit the active immutable version through ``current.json``."""

    pointer_payload = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "publication_state": "COMPLETE",
        "publication_generation": generation,
        "path": str(version_path.relative_to(output)).replace(os.sep, "/"),
        "manifest_sha256": compute_file_sha256(version_path / "manifest.json"),
    }
    pointer = output / f".current-{uuid.uuid4().hex}.tmp"
    _write_json(pointer, pointer_payload)
    try:
        _atomic_replace_pointer(pointer, output / "current.json")
        _fsync_directory(output)
    except Exception:
        if os.path.lexists(pointer):
            os.unlink(pointer)
        raise


def _swap_directory(staging: Path, output: Path, *, generation: str | None = None) -> None:
    """Publish an immutable version by atomically replacing ``current.json``.

    ``output`` is a normal, stable directory so static hosting works without
    special configuration.  Each complete staged payload is moved into
    ``output/versions/<generation>``.  The only authoritative switch is the
    atomic replacement of ``output/current.json``, which is always written
    last; a crash before that rename leaves the previous pointer and version
    loadable, while a crash after it exposes only the complete new version.

    No flat top-level aliases are written.  The pointer plus the immutable
    version directory is the entire canonical contract; any flat files already
    present in ``output`` are legacy artifacts and are left untouched.
    """

    output.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output) and output.is_symlink():
        raise ValueError(
            f"Publication path {output} is a symlink; migrate it to a normal directory with current.json"
        )
    output.mkdir(parents=True, exist_ok=True)
    versions = output / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    if not generation:
        raise ValueError("An immutable publication version requires a generation id")
    generation_name = generation
    version_path = versions / generation_name
    # Never overwrite an existing immutable version.  UUID collisions are
    # unlikely in normal operation, but a deterministic test seed, retry, or
    # caller-supplied generation must fail closed rather than corrupting a
    # version that an older current.json may still address.
    if os.path.lexists(version_path):
        raise FileExistsError(f"Immutable publication version already exists: {version_path}")
    os.replace(staging, version_path)
    _fsync_directory(versions)
    # The version is complete and immutable on disk before the pointer moves.
    # current.json is always the last write of a publication.
    _write_pointer(output, generation=generation_name, version_path=version_path)


def export_static_data(
    result: Any,
    *,
    output_dir: Path | str,
    generated_at_utc: str | None = None,
    calibration_dir: Path | str | None = None,
    prior_snapshot: Mapping[str, Any] | None = None,
    generation_id: str | None = None,
) -> dict[str, Any]:
    """Validate and atomically publish one immutable version behind a pointer.

    ``generation_id`` is the canonical generation shared with the prospective
    archive snapshot this publication was built from.  When omitted it is
    derived from the generation timestamp and the deterministic payload hash,
    so a publication always has a sortable, content-linked identity.
    """
    generated = generated_at_utc or datetime.now(timezone.utc).isoformat()
    parsed = datetime.fromisoformat(generated.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("generated_at_utc must include a timezone")
    contracts = _build_contracts(
        result,
        generated_at_utc=generated,
        calibration_dir=Path(calibration_dir) if calibration_dir else None,
        prior_snapshot=prior_snapshot,
    )
    validate_publication_contract(contracts)
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        for filename in PUBLICATION_FILES:
            _write_json(staging / filename, contracts[filename])
        file_hashes = {
            filename: compute_file_sha256(staging / filename)
            for filename in PUBLICATION_FILES
        }
        deterministic_file_hashes = {
            filename: _sha256_bytes(_canonical_bytes(_strip_runtime_timestamps(contracts[filename])))
            for filename in PUBLICATION_FILES
        }
        deterministic_manifest_hash = _sha256_bytes(_canonical_bytes(deterministic_file_hashes))
        if generation_id is None:
            generation = build_generation_id(
                generated, contracts["metadata.json"]["deterministic_payload_sha256"]
            )
        else:
            generation = str(generation_id)
            if not GENERATION_ID_PATTERN.fullmatch(generation):
                raise ValueError(f"Publication generation id is not web-safe: {generation!r}")
        manifest = {
            "schema_version": PUBLICATION_SCHEMA_VERSION,
            "publication_state": "COMPLETE",
            "publication_generation": generation,
            "generated_at_utc": generated,
            "publication_files": file_hashes,
            "deterministic_content_hashes": deterministic_file_hashes,
            "deterministic_content_sha256": deterministic_manifest_hash,
            "model_version": contracts["metadata.json"]["model"]["version"],
            "source_repository": contracts["metadata.json"]["source_repository"],
            "source_git_commit": contracts["metadata.json"]["source_git_commit"],
            "source_worktree_clean": contracts["metadata.json"]["source_worktree_clean"],
            "deterministic_payload_sha256": contracts["metadata.json"]["deterministic_payload_sha256"],
        }
        _write_json(staging / "manifest.json", manifest)
        _swap_directory(staging, output, generation=generation)
        staging = None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return manifest
