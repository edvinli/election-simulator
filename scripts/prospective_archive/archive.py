"""Write immutable, compact prospective ElectionSimulator forecast snapshots.

The archive deliberately stores summaries and histograms rather than the full
Monte Carlo arrays.  The canonical deterministic payload hash and the source
and input hashes make a snapshot independently traceable to the full forecast
artifact used to create it.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from scripts.simulator.config import (
    DEFAULT_ELECTION_DATE,
    DEFAULT_SIMULATION_SEED,
    DEFAULT_SIMULATION_SAMPLES,
    DEFAULT_SIMULATIONS_DIR,
    MODEL_PARTIES_9,
    BENCHMARK_LINEAGE_CANDIDATE,
    MODEL_VERSION,
    PARLIAMENTARY_PARTIES_8,
)
from scripts.simulator.engine import SimulationResult, simulate_election
from scripts.simulator.pipeline import build_canonical_summary_dict
from scripts.simulator.reproducibility import (
    build_generation_id,
    compute_file_sha256,
    compute_dict_sha256,
    require_resolvable_source_commit,
)
from scripts.forecast_history.contract import build_groups_from_matrices


from scripts.vote_share_calibration.election_noise_b import (
    election_noise_candidate_for_law,
)
# 1.0 snapshots are keyed one-per-calendar-day at ``<snapshot_date>/``.  1.1
# snapshots carry a sortable ``generation_id`` and live at
# ``<generation_id>/``, which allows several immutable forecasts per day.
# Both are readable; only 1.1 is written.
ARCHIVE_SCHEMA_VERSION = "1.2"
SUPPORTED_ARCHIVE_SCHEMA_VERSIONS: tuple[str, ...] = ("1.0", "1.1", "1.2")

#: Archive schema 1.2 adds ``election_noise_law`` and ``election_noise_candidate``
#: to the snapshot ``model`` block, following the same additive convention as 1.1
#: (which added ``generation_id``). Existing snapshots stay valid and are never
#: rewritten. ``model.candidate`` remains the botten-ada benchmark lineage label.
DEFAULT_ARCHIVE_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "prospective_forecasts"
DEFAULT_CANONICAL_FORECAST = DEFAULT_SIMULATIONS_DIR / "simulation_summary_N100000_seed12345.json"
DEFAULT_CANONICAL_HASH = DEFAULT_SIMULATIONS_DIR / "deterministic_payload.sha256"


class SnapshotCollisionError(FileExistsError):
    """Raised when an archive operation would overwrite or duplicate a snapshot."""


def _canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write JSON atomically, refusing to replace an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SnapshotCollisionError(f"Refusing to overwrite existing archive file: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # The destination was checked before writing.  Do not use this helper
        # for the append-only index, where replacing the existing index is
        # intentional; that operation has its own helper below.
        os.link(temporary, path)
        temporary.unlink(missing_ok=True)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_replace_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically replace an existing JSON index after validation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _histogram(values: np.ndarray, *, lower: float, upper: float, width: float) -> dict[str, Any]:
    """Return deterministic compact histogram plus quantiles for one variable."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot archive an empty distribution")
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


def _party_distribution_payload(result: SimulationResult) -> tuple[dict[str, Any], dict[str, Any]]:
    vote_distributions: dict[str, Any] = {}
    for idx, party in enumerate(MODEL_PARTIES_9):
        vote_distributions[party] = _histogram(
            result.vote_shares_matrix[:, idx], lower=0.0, upper=100.0, width=0.25
        )

    seat_distributions: dict[str, Any] = {}
    for idx, party in enumerate(PARLIAMENTARY_PARTIES_8):
        seat_distributions[party] = _histogram(
            result.seats_matrix[:, idx], lower=0.0, upper=349.0, width=1.0
        )
    return vote_distributions, seat_distributions


def _summary_from_result(result: SimulationResult) -> dict[str, Any]:
    summary = build_canonical_summary_dict(result)
    return {
        "parties": summary["parties"],
        "blocs": summary["blocs"],
        "total_samples": summary["total_samples"],
    }


def _archive_identity(manifest: Mapping[str, Any], payload_hash: str) -> str:
    """Stable identity for one forecast information set and model run."""
    fields = {
        "as_of": manifest.get("as_of"),
        "election_date": manifest.get("election_date"),
        "model_version": manifest.get("model_version"),
        "base_seed": manifest.get("base_seed"),
        "poll_data_hash": manifest.get("poll_data_hash"),
        "source_git_commit": manifest.get("source_git_commit", manifest.get("git_commit")),
        "deterministic_payload_sha256": payload_hash,
    }
    return _canonical_json_hash(fields)


def _validate_rest_semantics(summary: Mapping[str, Any]) -> None:
    """Ensure REST is aggregate residual mass and cannot qualify independently."""
    parties = summary.get("parties", {})
    rest = parties.get("REST")
    if rest is None:
        raise ValueError("Canonical summary must contain aggregate REST")
    # REST is a residual aggregate, so its numerical residual share can
    # occasionally exceed 4%; that does not make REST an eligible party.  The
    # eligibility/seat surfaces in the archive intentionally omit it.
    if any(rest.get(field, 0) for field in ("seats_mean", "seats_median", "prob_any_seats")):
        raise ValueError("REST must remain an aggregate ineligible category and cannot receive seats")


def build_snapshot(
    result: SimulationResult,
    *,
    generated_at_utc: str | None = None,
    canonical_artifact_path: Path | str | None = None,
    canonical_payload_hash_path: Path | str | None = None,
    allow_duplicate_payload: bool = False,
    duplicate_payload_sequence: int = 0,
) -> dict[str, Any]:
    """Build a versioned compact snapshot from an already-run simulation."""
    summary = _summary_from_result(result)
    _validate_rest_semantics(summary)
    manifest = dict(result.manifest)
    payload_hash = build_canonical_summary_dict(result)["deterministic_payload_sha256"]
    generated = generated_at_utc or datetime.now(timezone.utc).isoformat()
    parsed = datetime.fromisoformat(generated.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("generated_at_utc must include a timezone")

    canonical_path = Path(canonical_artifact_path) if canonical_artifact_path else DEFAULT_CANONICAL_FORECAST
    hash_path = Path(canonical_payload_hash_path) if canonical_payload_hash_path else DEFAULT_CANONICAL_HASH
    if not canonical_path.exists():
        raise FileNotFoundError(f"Canonical forecast artifact is required for an archive snapshot: {canonical_path}")
    if not hash_path.exists():
        raise FileNotFoundError(f"Canonical deterministic payload sidecar is required: {hash_path}")
    canonical_hash = compute_file_sha256(canonical_path)
    sidecar_value = hash_path.read_text(encoding="utf-8").strip()
    if sidecar_value != payload_hash:
        raise ValueError("Simulation payload does not match canonical deterministic payload sidecar")

    # Writing an unresolvable commit sentinel into an immutable archive entry
    # would permanently destroy the artifact's traceability, so fail closed.
    require_resolvable_source_commit(manifest)

    vote_dists, seat_dists = _party_distribution_payload(result)
    information_set_identity = _archive_identity(manifest, payload_hash)
    # A scheduled daily publication or an explicitly forced rerun may produce
    # bit-identical draws (the seed and inputs are intentionally frozen).  The
    # default archive API still refuses duplicate identities, while the
    # production boundary opts in to retaining that immutable generation by
    # salting only the archive identity with its publication timestamp.  The
    # deterministic payload itself remains unchanged and is still linked in
    # every artifact.
    if duplicate_payload_sequence < 0:
        raise ValueError("duplicate_payload_sequence must be non-negative")
    identity = (
        _canonical_json_hash(
            {
                "information_set_identity": information_set_identity,
                "generated_at_utc": generated,
                "duplicate_payload_sequence": int(duplicate_payload_sequence),
            }
        )
        if allow_duplicate_payload
        else information_set_identity
    )
    snapshot = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "snapshot_id": identity,
        "information_set_id": information_set_identity,
        "duplicate_payload_allowed": bool(allow_duplicate_payload),
        # One canonical generation id, shared verbatim with the static
        # publication version directory built from this snapshot.
        "generation_id": build_generation_id(generated, identity),
        "snapshot_date": str(manifest["as_of"]),
        "generated_at_utc": generated,
        "as_of": str(manifest["as_of"]),
        "election_date": str(manifest["election_date"]),
        # ``candidate`` is the botten-ada benchmark / model-lineage label, NOT the
        # ElectionNoise challenger. The ElectionNoise identity is the two explicit
        # fields below, sourced from the run manifest so the snapshot reports the law
        # that actually ran rather than a hard-coded assumption.
        "model": {
            "name": "ElectionSimulator",
            "version": manifest.get("model_version", MODEL_VERSION),
            "candidate": BENCHMARK_LINEAGE_CANDIDATE,
            "candidate_namespace": (
                "botten_ada_benchmark_model_lineage; not the ElectionNoise challenger"
            ),
            "election_noise_law": (manifest.get("model_config") or {}).get("noise_model"),
            "election_noise_candidate": election_noise_candidate_for_law(
                (manifest.get("model_config") or {}).get("noise_model")
            ),
        },
        "seed": int(manifest["base_seed"]),
        "samples": int(manifest["samples"]),
        "source_git_commit": manifest.get("source_git_commit", manifest.get("git_commit")),
        "source_worktree_clean": bool(manifest.get("source_worktree_clean")),
        "hashes": {
            "poll_data_sha256": manifest.get("poll_data_hash"),
            "election_data_sha256": manifest.get("election_data_hash"),
            "mandate_data_sha256": manifest.get("mandate_data_hash"),
            "geography_data_sha256": manifest.get("geography_data_hash"),
            "model_config_sha256": manifest.get("model_config_hash"),
            "deterministic_payload_sha256": payload_hash,
            "canonical_artifact_sha256": canonical_hash,
        },
        "input_config": {
            "model_config": manifest.get("model_config", {}),
            "geography_baseline_year": manifest.get("model_config", {}).get("geography_baseline_year"),
            "total_national_votes": manifest.get("model_config", {}).get("total_national_votes"),
        },
        "national_vote_distributions": vote_dists,
        "national_vote_summary": summary["parties"],
        "threshold_probabilities_4pct": {
            party: summary["parties"][party]["prob_above_4pct"]
            for party in PARLIAMENTARY_PARTIES_8
        },
        "seat_distributions": seat_dists,
        "seat_summary": {
            party: {
                "mean": summary["parties"][party]["seats_mean"],
                "median": summary["parties"][party]["seats_median"],
                "p05": summary["parties"][party]["seats_p05"],
                "p95": summary["parties"][party]["seats_p95"],
            }
            for party in PARLIAMENTARY_PARTIES_8
        },
        "group_probabilities": summary["blocs"],
        # Preserve compact coalition summaries computed from the original
        # draw matrices.  This is the only safe input for a prospective chart
        # point; marginal party quantiles cannot recover coalition intervals.
        "groups": build_groups_from_matrices(
            result.vote_shares_matrix,
            result.seats_matrix,
        ),
        "deterministic_payload_sha256": payload_hash,
        "canonical_artifact_sha256": canonical_hash,
        "canonical_artifact_path": str(canonical_path),
        "rest_semantics": "REST is aggregate vote mass for modeled-as-ineligible parties; it cannot independently qualify or receive seats.",
    }
    # Ensure identity does not silently change when the schema evolves.
    snapshot["snapshot_id"] = identity
    return snapshot


def _load_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": ARCHIVE_SCHEMA_VERSION, "archive": "ElectionSimulator prospective forecasts", "snapshots": []}
    index = _read_json(path)
    if index.get("schema_version") not in SUPPORTED_ARCHIVE_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported archive index schema: {index.get('schema_version')}")
    if not isinstance(index.get("snapshots"), list):
        raise ValueError("Archive index snapshots must be a list")
    return index


def _validate_index(index: Mapping[str, Any]) -> None:
    """Validate append-only index integrity.

    Snapshot identity, generation id, and stored path must each be unique.
    Deterministic payload hashes are unique for ordinary entries; a production
    entry may explicitly mark a retained duplicate payload.  ``snapshot_date``
    is deliberately *not* unique: the archive supports several immutable
    forecasts on the same calendar day.
    """

    seen_id: set[str] = set()
    seen_hash: set[str] = set()
    seen_generation: set[str] = set()
    seen_path: set[str] = set()
    for row in index.get("snapshots", []):
        if not isinstance(row, Mapping):
            raise ValueError("Archive index entries must be objects")
        identity = str(row.get("snapshot_id", ""))
        payload_hash = str(row.get("deterministic_payload_sha256", ""))
        snapshot_date = str(row.get("snapshot_date", ""))
        stored_path = str(row.get("path", ""))
        if not identity or identity in seen_id:
            raise ValueError("Archive index contains duplicate or missing snapshot identity")
        if not payload_hash:
            raise ValueError("Archive index contains duplicate or missing payload hash")
        # Legacy/default entries remain unique.  A production entry may carry
        # an explicit opt-in marker because an unchanged seeded simulation is
        # still a real immutable publication generation (daily/force runs).
        if payload_hash in seen_hash and row.get("duplicate_payload_allowed") is not True:
            raise ValueError("Archive index contains duplicate or missing payload hash")
        if not snapshot_date:
            raise ValueError("Archive index contains a missing snapshot date")
        if not stored_path or stored_path in seen_path:
            raise ValueError("Archive index contains duplicate or missing snapshot path")
        # Pre-extraction 1.0 entries carry no generation id.  They remain
        # valid and are never rewritten; only new entries are constrained.
        generation = row.get("generation_id")
        if generation is not None:
            generation = str(generation)
            if not generation or generation in seen_generation:
                raise ValueError("Archive index contains duplicate or missing generation id")
            seen_generation.add(generation)
        seen_id.add(identity)
        seen_hash.add(payload_hash)
        seen_path.add(stored_path)


def write_snapshot(
    result: SimulationResult,
    *,
    archive_dir: Path | str = DEFAULT_ARCHIVE_DIR,
    generated_at_utc: str | None = None,
    canonical_artifact_path: Path | str | None = None,
    canonical_payload_hash_path: Path | str | None = None,
    allow_duplicate_payload: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    """Atomically write one snapshot and append its validated manifest entry."""
    root = Path(archive_dir)
    index_path = root / "index.json"
    index = _load_index(index_path)
    _validate_index(index)
    duplicate_sequence = 0
    if allow_duplicate_payload:
        # The sequence is derived solely from the append-only index, so even
        # same-second manual retries receive a distinct deterministic identity
        # and generation while generation/path collisions remain impossible to
        # overwrite silently.
        duplicate_sequence = sum(
            1
            for row in index["snapshots"]
            if row.get("deterministic_payload_sha256")
            == build_canonical_summary_dict(result)["deterministic_payload_sha256"]
        )
    snapshot = build_snapshot(
        result,
        generated_at_utc=generated_at_utc,
        canonical_artifact_path=canonical_artifact_path,
        canonical_payload_hash_path=canonical_payload_hash_path,
        allow_duplicate_payload=allow_duplicate_payload,
        duplicate_payload_sequence=duplicate_sequence,
    )
    existing_ids = {row["snapshot_id"] for row in index["snapshots"]}
    existing_hashes = {row["deterministic_payload_sha256"] for row in index["snapshots"]}
    existing_generations = {
        row["generation_id"] for row in index["snapshots"] if row.get("generation_id")
    }
    if snapshot["snapshot_id"] in existing_ids:
        raise SnapshotCollisionError("A forecast with this information-set identity is already archived")
    if (
        snapshot["deterministic_payload_sha256"] in existing_hashes
        and not allow_duplicate_payload
    ):
        raise SnapshotCollisionError("A forecast with this deterministic payload is already archived")
    if snapshot["generation_id"] in existing_generations:
        raise SnapshotCollisionError("A forecast with this generation id is already archived")
    # Several immutable forecasts may share a calendar day; the sortable
    # generation id, not the as-of date, is what must stay unique.

    snapshot_path = root / str(snapshot["generation_id"]) / "snapshot.json"
    _atomic_write_json(snapshot_path, snapshot)
    index_entry = {
        "snapshot_id": snapshot["snapshot_id"],
        "generation_id": snapshot["generation_id"],
        "snapshot_date": snapshot["snapshot_date"],
        "as_of": snapshot["as_of"],
        "election_date": snapshot["election_date"],
        "generated_at_utc": snapshot["generated_at_utc"],
        "source_git_commit": snapshot["source_git_commit"],
        "model_version": snapshot["model"]["version"],
        "seed": snapshot["seed"],
        "deterministic_payload_sha256": snapshot["deterministic_payload_sha256"],
        "duplicate_payload_allowed": bool(allow_duplicate_payload),
        "canonical_artifact_sha256": snapshot["canonical_artifact_sha256"],
        "snapshot_file_sha256": compute_file_sha256(snapshot_path),
        "path": str(snapshot_path.relative_to(root)),
    }
    index["snapshots"].append(index_entry)
    # The index header declares the newest schema it contains.  Every existing
    # entry is carried over byte-for-byte; only the header and the appended
    # entry are new.
    index["schema_version"] = ARCHIVE_SCHEMA_VERSION
    index["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _validate_index(index)
    try:
        _atomic_replace_json(index_path, index)
    except SnapshotCollisionError:
        # A concurrent index update must not leave a snapshot that is absent
        # from the index; the snapshot remains immutable and the operation is
        # reported as failed for manual reconciliation.
        raise
    return snapshot_path, index_path, snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Archive one immutable ElectionSimulator prospective forecast")
    parser.add_argument("--as-of", default=None, help="Poll cutoff date (YYYY-MM-DD); defaults to configured latest")
    parser.add_argument("--election-date", default=DEFAULT_ELECTION_DATE)
    parser.add_argument("--samples", type=int, default=DEFAULT_SIMULATION_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SIMULATION_SEED)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--canonical-artifact", type=Path, default=DEFAULT_CANONICAL_FORECAST)
    parser.add_argument("--canonical-payload-hash", type=Path, default=DEFAULT_CANONICAL_HASH)
    parser.add_argument("--generated-at-utc", default=None, help="Fixed ISO timestamp for deterministic tests/reproduction")
    args = parser.parse_args(argv)

    result = simulate_election(
        as_of=args.as_of,
        election_date=args.election_date,
        samples=args.samples,
        seed=args.seed,
    )
    snapshot_path, index_path, snapshot = write_snapshot(
        result,
        archive_dir=args.archive_dir,
        generated_at_utc=args.generated_at_utc,
        canonical_artifact_path=args.canonical_artifact,
        canonical_payload_hash_path=args.canonical_payload_hash,
    )
    print(json.dumps({
        "snapshot": str(snapshot_path),
        "index": str(index_path),
        "snapshot_id": snapshot["snapshot_id"],
        "generation_id": snapshot["generation_id"],
        "payload_sha256": snapshot["deterministic_payload_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
