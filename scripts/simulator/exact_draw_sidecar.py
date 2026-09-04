"""Certified ElectionSimulator draw sidecars and cutoff selection.

The production simulator keeps its complete joint draw matrices in the
``SimulationResult`` returned by :func:`scripts.simulator.engine.simulate_election`.
The normal prospective archive deliberately stores compact summaries instead.
This module is the narrow bridge for the 2026 prospective benchmark: a caller
can persist those exact matrices immediately after the certified production
run, and a later collector can select an already committed generation without
re-running (or reinterpreting) the simulator.

Nothing in this module creates draws from quantiles.  A sidecar is accepted
only when its arrays, metadata, generation id, source provenance and
deterministic payload hash all agree.  Selection also proves that the first
commit containing the immutable archive snapshot was itself made no later than
the requested wall-clock cutoff.  This prevents a later artifact commit from
being mistaken for evidence that was available prospectively.

The sidecar format is intentionally independent of the compact
``prospective_forecasts`` schema.  Existing snapshots are never rewritten;
future benchmark captures may call :func:`write_exact_draw_sidecar` with the
same ``SimulationResult`` used by the production publication pipeline.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any, Mapping
import zipfile

import numpy as np

from scripts.prospective_archive.archive import _validate_index
from scripts.simulator.config import MODEL_PARTIES_9, PARLIAMENTARY_PARTIES_8
from scripts.simulator.pipeline import build_canonical_summary_dict
from scripts.simulator.reproducibility import compute_file_sha256, is_git_worktree_clean


SIDECAR_SCHEMA_VERSION = "1.0"
SIDECAR_DRAWS_FILENAME = "draws.npz"
SIDECAR_METADATA_FILENAME = "draws.json"
_PAYLOAD_HASH_LENGTH = 64
REPLAY_DEPENDENCY_PATHS: tuple[str, ...] = ("pyproject.toml", "uv.lock")
REPLAY_EQUIVALENCE_PATHS: tuple[str, ...] = (
    "scripts",
    *REPLAY_DEPENDENCY_PATHS,
    "data/processed/pollofpolls",
    "data/processed/elections",
    "data/processed/mandates",
    "data/processed/geography",
)


class ExactDrawSidecarError(ValueError):
    """Raised when exact-draw evidence fails closed validation."""


def _parse_utc(value: Any, *, field: str) -> datetime:
    """Parse an ISO-8601 timestamp and normalize it to UTC."""

    if not isinstance(value, str) or not value.strip():
        raise ExactDrawSidecarError(f"{field} must be a non-empty ISO-8601 timestamp")
    raw = value.strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ExactDrawSidecarError(f"{field} is not a valid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ExactDrawSidecarError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _array_digest(value: np.ndarray) -> str:
    """Hash one array with dtype and shape included in the digest."""

    array = np.asarray(value)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    descriptor = {
        "dtype": array.dtype.str,
        "shape": [int(item) for item in array.shape],
    }
    digest = hashlib.sha256()
    digest.update(_canonical_bytes(descriptor))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _require_hash(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != _PAYLOAD_HASH_LENGTH:
        raise ExactDrawSidecarError(f"{field} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ExactDrawSidecarError(f"{field} must be a SHA-256 hex digest") from exc
    return value


def _regular_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ExactDrawSidecarError(f"{label} must be a regular file: {path}")


def _validated_matrices(result: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate the exact matrices at the production boundary."""

    try:
        votes = np.asarray(result.vote_shares_matrix)
        seats = np.asarray(result.seats_matrix)
    except AttributeError as exc:
        raise ExactDrawSidecarError("SimulationResult must expose vote_shares_matrix and seats_matrix") from exc

    if votes.ndim != 2 or votes.shape[1] != len(MODEL_PARTIES_9) or votes.shape[0] == 0:
        raise ExactDrawSidecarError(
            f"vote_shares_matrix must have shape (N, {len(MODEL_PARTIES_9)})"
        )
    if seats.ndim != 2 or seats.shape != (votes.shape[0], len(PARLIAMENTARY_PARTIES_8)):
        raise ExactDrawSidecarError(
            f"seats_matrix must have shape ({votes.shape[0]}, {len(PARLIAMENTARY_PARTIES_8)})"
        )
    if votes.dtype != np.dtype("<f8"):
        # The simulator contract is float64.  Refusing another dtype prevents
        # a caller from silently changing binary draw evidence at this boundary.
        raise ExactDrawSidecarError(f"vote_shares_matrix must be float64, got {votes.dtype}")
    if seats.dtype != np.dtype("<i8"):
        raise ExactDrawSidecarError(f"seats_matrix must be int64, got {seats.dtype}")
    if not np.isfinite(votes).all() or not np.isfinite(seats).all():
        raise ExactDrawSidecarError("exact draw matrices contain non-finite values")
    if np.any(votes < -1e-10):
        raise ExactDrawSidecarError("exact vote-share draws contain negative values")
    if np.max(np.abs(votes.sum(axis=1) - 100.0)) > 1e-7:
        raise ExactDrawSidecarError("exact vote-share draws do not sum to 100 percentage points")
    if np.any(seats < 0) or np.any(seats.sum(axis=1) != 349):
        raise ExactDrawSidecarError("exact seat draws violate the 349-seat invariant")
    return np.ascontiguousarray(votes, dtype="<f8"), np.ascontiguousarray(seats, dtype="<i8")


def _manifest_for_sidecar(result: Any) -> Mapping[str, Any]:
    manifest = getattr(result, "manifest", None)
    if not isinstance(manifest, Mapping):
        raise ExactDrawSidecarError("SimulationResult manifest is required for exact-draw provenance")
    source_commit = manifest.get("source_git_commit", manifest.get("git_commit"))
    if not isinstance(source_commit, str) or not source_commit or source_commit == "unknown_git_commit":
        raise ExactDrawSidecarError("exact-draw evidence requires a resolvable source_git_commit")
    if manifest.get("source_worktree_clean") is not True:
        raise ExactDrawSidecarError("exact-draw evidence requires source_worktree_clean=true")
    return manifest


def _validate_generation_link(
    *,
    result: Any,
    generation_id: str,
    certified_snapshot: Mapping[str, Any] | None,
    payload_hash: str,
) -> None:
    """Ensure the sidecar is linked to the certified publication generation."""

    if not isinstance(generation_id, str) or not generation_id or "/" in generation_id or "\\" in generation_id:
        raise ExactDrawSidecarError("generation_id must be a non-empty path-safe string")
    manifest = _manifest_for_sidecar(result)
    if certified_snapshot is None:
        return

    snapshot_generation = certified_snapshot.get("generation_id")
    if snapshot_generation != generation_id:
        raise ExactDrawSidecarError(
            f"certified generation mismatch: expected {generation_id!r}, got {snapshot_generation!r}"
        )
    snapshot_hashes = certified_snapshot.get("hashes")
    if snapshot_hashes is not None and not isinstance(snapshot_hashes, Mapping):
        raise ExactDrawSidecarError("certified snapshot hashes must be an object")
    snapshot_payload = certified_snapshot.get(
        "deterministic_payload_sha256",
        (snapshot_hashes or {}).get("deterministic_payload_sha256"),
    )
    if snapshot_payload != payload_hash:
        raise ExactDrawSidecarError("certified snapshot does not match the SimulationResult payload hash")

    for field in ("as_of", "election_date"):
        result_value = manifest.get(field)
        snapshot_value = certified_snapshot.get(field)
        if snapshot_value is not None and str(result_value) != str(snapshot_value):
            raise ExactDrawSidecarError(f"certified snapshot {field} disagrees with SimulationResult")
    snapshot_commit = certified_snapshot.get("source_git_commit")
    if snapshot_commit is not None and snapshot_commit != manifest.get("source_git_commit", manifest.get("git_commit")):
        raise ExactDrawSidecarError("certified snapshot source_git_commit disagrees with SimulationResult")
    snapshot_samples = certified_snapshot.get("samples")
    if snapshot_samples is not None and int(snapshot_samples) != int(np.asarray(result.vote_shares_matrix).shape[0]):
        raise ExactDrawSidecarError("certified snapshot sample count disagrees with SimulationResult")


def _sidecar_arrays(result: Any) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any], str]:
    votes, seats = _validated_matrices(result)
    manifest = _manifest_for_sidecar(result)
    # Recompute the canonical payload hash from the exact result that is being
    # archived.  A caller cannot supply an unrelated hash as provenance.
    canonical = build_canonical_summary_dict(result)
    payload_hash = _require_hash(
        canonical.get("deterministic_payload_sha256"),
        field="deterministic_payload_sha256",
    )
    if int(manifest.get("samples", votes.shape[0])) != votes.shape[0]:
        raise ExactDrawSidecarError("SimulationResult manifest sample count disagrees with its matrices")
    return votes, seats, manifest, payload_hash


def build_exact_draw_metadata(
    result: Any,
    *,
    generation_id: str,
    certified_snapshot: Mapping[str, Any] | None = None,
    draws_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Build JSON metadata for a sidecar from one already-run result.

    ``certified_snapshot`` is the compact archive snapshot (or the equivalent
    pipeline record) for the same generation.  If supplied, all available
    identity fields are cross-checked before metadata is returned.
    """

    votes, seats, manifest, payload_hash = _sidecar_arrays(result)
    _validate_generation_link(
        result=result,
        generation_id=generation_id,
        certified_snapshot=certified_snapshot,
        payload_hash=payload_hash,
    )
    source_commit = manifest.get("source_git_commit", manifest.get("git_commit"))
    metadata: dict[str, Any] = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "artifact": "ElectionSimulator exact prospective predictive draws",
        "generation_id": generation_id,
        "snapshot_id": None if certified_snapshot is None else certified_snapshot.get("snapshot_id"),
        "as_of": str(manifest.get("as_of")),
        "election_date": str(manifest.get("election_date")),
        "generated_at_utc": None if certified_snapshot is None else certified_snapshot.get("generated_at_utc"),
        "model": {
            "name": "ElectionSimulator",
            "version": manifest.get("model_version"),
        },
        "source_git_commit": source_commit,
        "source_worktree_clean": True,
        "seed": int(manifest.get("base_seed")),
        "samples": int(votes.shape[0]),
        "vote_share_unit": "percentage_points",
        "vote_party_order": list(MODEL_PARTIES_9),
        "seat_party_order": list(PARLIAMENTARY_PARTIES_8),
        "deterministic_payload_sha256": payload_hash,
        "arrays": {
            "vote_shares_pct": {
                "dtype": votes.dtype.str,
                "shape": [int(item) for item in votes.shape],
                "sha256": _array_digest(votes),
            },
            "seats": {
                "dtype": seats.dtype.str,
                "shape": [int(item) for item in seats.shape],
                "sha256": _array_digest(seats),
            },
        },
        "draws_file_sha256": None if draws_file_sha256 is None else _require_hash(draws_file_sha256, field="draws_file_sha256"),
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy_version": np.__version__,
        },
        "draw_semantics": (
            "Exact joint matrices emitted by the certified production SimulationResult; "
            "no draws are reconstructed from published quantiles."
        ),
    }
    return metadata


def _exclusive_write_bytes(path: Path, content: bytes) -> None:
    """Atomically create one file, refusing to replace an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"Refusing to overwrite immutable exact-draw evidence: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-linking a fully written temporary file gives create-without-
        # replace semantics even if another process races this writer.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _npz_bytes(votes: np.ndarray, seats: np.ndarray) -> bytes:
    """Encode a deterministic NPZ (``np.savez_compressed`` stores wall time)."""

    def npy_bytes(array: np.ndarray) -> bytes:
        stream = io.BytesIO()
        np.save(stream, array, allow_pickle=False)
        return stream.getvalue()

    stream = io.BytesIO()
    # ``np.savez_compressed`` uses ZIP_DEFLATED but stamps each member with the
    # current local time.  Fixed timestamps make a retry byte-identical, which
    # is useful for immutable/idempotent capture commits and avoids a sidecar
    # hash changing merely because a run was retried.
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, array in (("vote_shares_pct", votes), ("seats", seats)):
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 3
            archive.writestr(member, npy_bytes(array))
    return stream.getvalue()


def build_exact_draw_sidecar_files(
    result: Any,
    *,
    generation_id: str,
    certified_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, bytes]:
    """Return immutable sidecar file bytes from the exact production result.

    This is the preferred integration seam for the 2026 capture orchestrator:
    pass the returned ``draws.npz`` and ``draws.json`` bytes to its atomic
    capture materializer.  The arrays are produced from the same
    ``SimulationResult`` that the certified publication consumed, so no second
    simulation or deterministic replay is involved.
    """

    votes, seats, _manifest, payload_hash = _sidecar_arrays(result)
    _validate_generation_link(
        result=result,
        generation_id=generation_id,
        certified_snapshot=certified_snapshot,
        payload_hash=payload_hash,
    )
    npz_content = _npz_bytes(votes, seats)
    metadata = build_exact_draw_metadata(
        result,
        generation_id=generation_id,
        certified_snapshot=certified_snapshot,
        draws_file_sha256=hashlib.sha256(npz_content).hexdigest(),
    )
    metadata_content = (
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    return {
        SIDECAR_DRAWS_FILENAME: npz_content,
        SIDECAR_METADATA_FILENAME: metadata_content,
    }


def replay_certified_generation(
    certified_snapshot: Mapping[str, Any],
    repo_root: Path | str,
) -> tuple[Any, dict[str, Any]]:
    """Reproduce a certified generation and prove payload/summaries agree.

    This is a deliberately conservative fallback for a production workflow
    whose separate simulator job did not retain its in-memory result.  It runs
    the ordinary ``simulate_election`` entrypoint against the local checkout;
    before doing so it proves that the checkout is clean and that no tracked
    ``scripts`` file, locked dependency contract, or model-input file differs
    from the generation's source commit. The replay is accepted only when the
    deterministic payload hash, input hashes, manifest identity, and every
    published compact summary agree.

    The returned result is the exact result used to build a sidecar.  The
    evidence mapping is JSON-serializable and records the equivalence checks.
    No replay is attempted when those checks cannot be established.
    """

    if not isinstance(certified_snapshot, Mapping):
        raise ExactDrawSidecarError("certified snapshot must be a mapping")
    source_commit = certified_snapshot.get("source_git_commit")
    if not isinstance(source_commit, str) or not source_commit:
        raise ExactDrawSidecarError("certified snapshot has no source_git_commit")
    generation_id = certified_snapshot.get("generation_id")
    if not isinstance(generation_id, str) or not generation_id:
        raise ExactDrawSidecarError("certified snapshot has no generation_id")
    payload_hash = _require_hash(
        certified_snapshot.get("deterministic_payload_sha256"),
        field="certified_snapshot.deterministic_payload_sha256",
    )
    repo = Path(repo_root).resolve()
    head = _git(repo, ["rev-parse", "HEAD"]).stdout.strip()
    if not head or not _commit_exists(repo, source_commit):
        raise ExactDrawSidecarError("certified source commit is not resolvable in the replay checkout")
    if not _is_ancestor(repo, source_commit, head):
        raise ExactDrawSidecarError("certified source commit is not an ancestor of the replay checkout")
    if not is_git_worktree_clean(repo):
        raise ExactDrawSidecarError("replay checkout must be clean before exact-draw reproduction")

    # A later artifact-only commit is harmless, but any tracked code, locked
    # dependency contract, or model-input change makes the current checkout
    # an unproved substitute for the certified environment.
    diff = _git(
        repo,
        ["diff", "--quiet", f"{source_commit}..{head}", "--", *REPLAY_EQUIVALENCE_PATHS],
        check=False,
    )
    if diff.returncode != 0:
        raise ExactDrawSidecarError(
            "replay checkout differs from the certified source in code or model inputs "
            "or dependency contract"
        )
    dependency_hashes: dict[str, str] = {}
    for relative in REPLAY_DEPENDENCY_PATHS:
        dependency = repo / relative
        _regular_file(dependency, label=f"replay dependency contract {relative}")
        dependency_hashes[relative] = compute_file_sha256(dependency)

    from scripts.simulator.engine import simulate_election

    try:
        samples = int(certified_snapshot["samples"])
        seed = int(certified_snapshot["seed"])
        as_of = str(certified_snapshot["as_of"])
        election_date = str(certified_snapshot["election_date"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExactDrawSidecarError("certified snapshot lacks replay configuration") from exc
    result = simulate_election(
        as_of=as_of,
        election_date=election_date,
        samples=samples,
        seed=seed,
        repo_dir=repo,
    )
    manifest = result.manifest
    snapshot_model = certified_snapshot.get("model")
    if not isinstance(snapshot_model, Mapping):
        raise ExactDrawSidecarError("certified snapshot model must be an object")
    expected_manifest = {
        "as_of": as_of,
        "election_date": election_date,
        "samples": samples,
        "base_seed": seed,
        "model_version": snapshot_model.get("version"),
    }
    for field, expected in expected_manifest.items():
        if expected is not None and manifest.get(field) != expected:
            raise ExactDrawSidecarError(f"replay manifest {field} disagrees with certified snapshot")
    snapshot_hashes = certified_snapshot.get("hashes")
    if not isinstance(snapshot_hashes, Mapping):
        raise ExactDrawSidecarError("certified snapshot lacks input hashes required for replay")
    manifest_hash_fields = {
        "poll_data_sha256": "poll_data_hash",
        "election_data_sha256": "election_data_hash",
        "mandate_data_sha256": "mandate_data_hash",
        "geography_data_sha256": "geography_data_hash",
        "model_config_sha256": "model_config_hash",
    }
    for snapshot_field, manifest_field in manifest_hash_fields.items():
        if snapshot_hashes.get(snapshot_field) != manifest.get(manifest_field):
            raise ExactDrawSidecarError(f"replay {manifest_field} disagrees with certified snapshot")
    snapshot_input_config = certified_snapshot.get("input_config")
    if snapshot_input_config is not None and not isinstance(snapshot_input_config, Mapping):
        raise ExactDrawSidecarError("certified snapshot input_config must be an object")
    snapshot_config = (snapshot_input_config or {}).get("model_config")
    if snapshot_config is not None and snapshot_config != manifest.get("model_config"):
        raise ExactDrawSidecarError("replay model_config disagrees with certified snapshot")

    summary = build_canonical_summary_dict(result)
    if summary.get("deterministic_payload_sha256") != payload_hash:
        raise ExactDrawSidecarError("replay deterministic payload hash does not match certified snapshot")
    if certified_snapshot.get("national_vote_summary") != summary.get("parties"):
        raise ExactDrawSidecarError("replay national vote summaries do not match certified snapshot")
    if certified_snapshot.get("group_probabilities") != summary.get("blocs"):
        raise ExactDrawSidecarError("replay group summaries do not match certified snapshot")
    expected_thresholds = {
        party: summary["parties"][party]["prob_above_4pct"]
        for party in PARLIAMENTARY_PARTIES_8
    }
    if certified_snapshot.get("threshold_probabilities_4pct") != expected_thresholds:
        raise ExactDrawSidecarError("replay threshold probabilities do not match certified snapshot")
    expected_seats = {
        party: {
            "mean": summary["parties"][party]["seats_mean"],
            "median": summary["parties"][party]["seats_median"],
            "p05": summary["parties"][party]["seats_p05"],
            "p95": summary["parties"][party]["seats_p95"],
        }
        for party in PARLIAMENTARY_PARTIES_8
    }
    if certified_snapshot.get("seat_summary") != expected_seats:
        raise ExactDrawSidecarError("replay seat summaries do not match certified snapshot")

    # The production source commit is the certified identity even when HEAD
    # contains a later artifact-only commit.  This override is made only after
    # all source/input equivalence and payload checks above have passed.
    manifest["source_git_commit"] = source_commit
    manifest["git_commit"] = source_commit
    if manifest.get("source_worktree_clean") is not True:
        raise ExactDrawSidecarError("replay result was not certified from a clean source worktree")
    evidence = {
        "status": "REPLAY_VERIFIED",
        "generation_id": generation_id,
        "certified_source_git_commit": source_commit,
        "replay_checkout_head": head,
        "source_equivalence_basis": (
            "git diff --quiet source_commit..HEAD over scripts, pyproject.toml, uv.lock, "
            "and all model-relevant processed input directories"
        ),
        "dependency_contract_sha256": dependency_hashes,
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy_version": np.__version__,
        },
        "deterministic_payload_sha256": payload_hash,
        "published_summary_parity": True,
        "input_hash_parity": True,
    }
    return result, evidence


def write_exact_draw_sidecar(
    result: Any,
    output_dir: Path | str,
    *,
    generation_id: str,
    certified_snapshot: Mapping[str, Any] | None = None,
    draws_filename: str = SIDECAR_DRAWS_FILENAME,
    metadata_filename: str = SIDECAR_METADATA_FILENAME,
) -> dict[str, Any]:
    """Persist exact joint draw matrices and provenance immutably.

    The two files are created with exclusive atomic writes.  If both already
    exist and validate against the requested generation/result, they are
    returned as an idempotent retry; neither file is ever replaced.  A single
    pre-existing file is treated as an incomplete write and fails closed.
    """

    if Path(draws_filename).name != draws_filename or Path(metadata_filename).name != metadata_filename:
        raise ExactDrawSidecarError("sidecar filenames must be direct children of output_dir")
    files = build_exact_draw_sidecar_files(
        result,
        generation_id=generation_id,
        certified_snapshot=certified_snapshot,
    )
    root = Path(output_dir)
    draws_path = root / draws_filename
    metadata_path = root / metadata_filename
    npz_content = files[SIDECAR_DRAWS_FILENAME]
    npz_hash = hashlib.sha256(npz_content).hexdigest()
    metadata_content = files[SIDECAR_METADATA_FILENAME]

    draws_exists = os.path.lexists(draws_path)
    metadata_exists = os.path.lexists(metadata_path)
    if draws_exists or metadata_exists:
        if not (draws_exists and metadata_exists):
            raise FileExistsError("partial exact-draw sidecar exists; refusing to complete or replace it")
        # Safe retry: require byte-for-byte draw evidence and semantic metadata
        # equality.  This does not make mutable evidence acceptable; it merely
        # makes a retry after a successful durable write idempotent.
        _regular_file(draws_path, label="draw sidecar")
        _regular_file(metadata_path, label="draw metadata")
        if draws_path.read_bytes() != npz_content or metadata_path.read_bytes() != metadata_content:
            raise FileExistsError("existing exact-draw sidecar differs; refusing to overwrite immutable evidence")
        loaded = load_verified_draw_sidecar(draws_path, metadata_path)
        return {
            "status": "ALREADY_PRESENT_VERIFIED",
            "draws_path": str(draws_path),
            "metadata_path": str(metadata_path),
            "draws_file_sha256": npz_hash,
            "metadata": loaded["metadata"],
        }

    # If the metadata write fails after the draw file is durable, the caller
    # receives a hard failure and the unindexed partial sidecar remains for
    # diagnosis.  A benchmark capture must index only the verified pair.
    _exclusive_write_bytes(draws_path, npz_content)
    try:
        _exclusive_write_bytes(metadata_path, metadata_content)
    except Exception:
        raise
    loaded = load_verified_draw_sidecar(draws_path, metadata_path)
    return {
        "status": "WRITTEN_VERIFIED",
        "draws_path": str(draws_path),
        "metadata_path": str(metadata_path),
        "draws_file_sha256": npz_hash,
        "metadata": loaded["metadata"],
    }


def load_verified_draw_sidecar(
    draws_path: Path | str,
    metadata_path: Path | str | None = None,
    *,
    expected_generation_id: str | None = None,
    expected_payload_hash: str | None = None,
    include_arrays: bool = True,
) -> dict[str, Any]:
    """Load and validate an exact-draw sidecar without enabling pickle."""

    draws = Path(draws_path)
    metadata = Path(metadata_path) if metadata_path is not None else draws.with_name(SIDECAR_METADATA_FILENAME)
    _regular_file(draws, label="draw sidecar")
    _regular_file(metadata, label="draw metadata")
    try:
        with metadata.open(encoding="utf-8") as handle:
            description = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ExactDrawSidecarError(f"cannot read exact-draw metadata: {metadata}") from exc
    if not isinstance(description, Mapping) or description.get("schema_version") != SIDECAR_SCHEMA_VERSION:
        raise ExactDrawSidecarError("unsupported exact-draw sidecar schema")
    generation_id = description.get("generation_id")
    if (
        not isinstance(generation_id, str)
        or not generation_id
        or Path(generation_id).name != generation_id
        or generation_id in {".", ".."}
    ):
        raise ExactDrawSidecarError("exact-draw sidecar generation_id is invalid")
    if expected_generation_id is not None and generation_id != expected_generation_id:
        raise ExactDrawSidecarError("exact-draw sidecar generation_id does not match selected generation")
    source_commit = description.get("source_git_commit")
    if (
        not isinstance(source_commit, str)
        or not source_commit
        or source_commit == "unknown_git_commit"
    ):
        raise ExactDrawSidecarError("exact-draw metadata requires a resolvable source_git_commit")
    if description.get("source_worktree_clean") is not True:
        raise ExactDrawSidecarError("exact-draw metadata requires source_worktree_clean=true")
    model = description.get("model")
    if not isinstance(model, Mapping) or model.get("name") != "ElectionSimulator":
        raise ExactDrawSidecarError("exact-draw metadata model is not ElectionSimulator")
    payload_hash = _require_hash(description.get("deterministic_payload_sha256"), field="deterministic_payload_sha256")
    if expected_payload_hash is not None and payload_hash != expected_payload_hash:
        raise ExactDrawSidecarError("exact-draw sidecar payload hash does not match selected generation")
    if description.get("vote_party_order") != list(MODEL_PARTIES_9):
        raise ExactDrawSidecarError("exact-draw sidecar vote party order is not canonical")
    if description.get("seat_party_order") != list(PARLIAMENTARY_PARTIES_8):
        raise ExactDrawSidecarError("exact-draw sidecar seat party order is not canonical")
    if description.get("vote_share_unit") != "percentage_points":
        raise ExactDrawSidecarError("exact-draw sidecar vote unit must be percentage_points")
    actual_file_hash = compute_file_sha256(draws)
    if actual_file_hash != description.get("draws_file_sha256"):
        raise ExactDrawSidecarError("exact-draw sidecar file hash does not match metadata")
    arrays_description = description.get("arrays")
    if not isinstance(arrays_description, Mapping):
        raise ExactDrawSidecarError("exact-draw metadata is missing array descriptors")
    samples = description.get("samples")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
        raise ExactDrawSidecarError("exact-draw metadata samples must be an integer >= 2")

    try:
        loaded = np.load(draws, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - normalize malformed evidence
        raise ExactDrawSidecarError(f"cannot read exact-draw NPZ: {draws}") from exc
    try:
        if set(loaded.files) != {"vote_shares_pct", "seats"}:
            raise ExactDrawSidecarError("exact-draw NPZ has unexpected array keys")
        votes = np.asarray(loaded["vote_shares_pct"])
        seats = np.asarray(loaded["seats"])
    finally:
        loaded.close()
    votes, seats = _validated_matrices(type("Draws", (), {"vote_shares_matrix": votes, "seats_matrix": seats})())
    for key, array in (("vote_shares_pct", votes), ("seats", seats)):
        descriptor = arrays_description.get(key)
        if not isinstance(descriptor, Mapping):
            raise ExactDrawSidecarError(f"missing descriptor for {key}")
        if descriptor.get("dtype") != array.dtype.str:
            raise ExactDrawSidecarError(f"{key} dtype disagrees with metadata")
        if descriptor.get("shape") != [int(item) for item in array.shape]:
            raise ExactDrawSidecarError(f"{key} shape disagrees with metadata")
        if descriptor.get("sha256") != _array_digest(array):
            raise ExactDrawSidecarError(f"{key} array hash disagrees with metadata")
    if samples != int(votes.shape[0]):
        raise ExactDrawSidecarError("exact-draw sample count disagrees with matrix shape")
    result: dict[str, Any] = {
        "metadata": dict(description),
        "draws_path": str(draws),
        "metadata_path": str(metadata),
    }
    if include_arrays:
        result["vote_shares_pct"] = votes
        result["seats"] = seats
    return result


def _git(repo_root: Path, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=check,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExactDrawSidecarError(f"git command failed in {repo_root}: {' '.join(args[:2])}") from exc


def _git_bytes(repo_root: Path, args: list[str]) -> subprocess.CompletedProcess[bytes]:
    """Run Git for an exact blob comparison without text decoding."""

    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise ExactDrawSidecarError(
            f"git command failed in {repo_root}: {' '.join(args[:2])}"
        ) from exc


def _commit_exists(repo_root: Path, commit: str) -> bool:
    result = _git(repo_root, ["cat-file", "-e", f"{commit}^{{commit}}"], check=False)
    return result.returncode == 0


def _history_for_path(repo_root: Path, relative_path: str) -> list[tuple[str, datetime]]:
    """Return commits touching one path in chronological order."""

    result = _git(
        repo_root,
        ["log", "HEAD", "--reverse", "--format=%H%x00%cI", "--", relative_path],
        check=False,
    )
    if result.returncode != 0:
        return []
    history: list[tuple[str, datetime]] = []
    for line in result.stdout.splitlines():
        commit, separator, timestamp = line.partition("\x00")
        if not separator:
            continue
        try:
            parsed = _parse_utc(timestamp, field="archive commit timestamp")
        except ExactDrawSidecarError:
            continue
        history.append((commit, parsed))
    return history


def _first_archive_commit(
    repo_root: Path,
    relative_path: str,
    *,
    expected_sha256: str,
) -> tuple[str, datetime] | None:
    """Find the first commit containing the exact current archive snapshot.

    Looking only for the commit that first added the path is insufficient if a
    later commit rewrote that path.  The content check makes the cutoff test
    about the immutable bytes selected today, not merely an old directory name.
    """

    _require_hash(expected_sha256, field="snapshot_file_sha256")
    for commit, committed_at in _history_for_path(repo_root, relative_path):
        blob = _git_bytes(repo_root, ["show", f"{commit}:{relative_path}"])
        if blob.returncode == 0 and hashlib.sha256(blob.stdout).hexdigest() == expected_sha256:
            return commit, committed_at
    return None


def _commit_timestamp(repo_root: Path, commit: str) -> datetime | None:
    """Return a commit's committer timestamp when it is available."""

    result = _git(repo_root, ["show", "-s", "--format=%cI", commit], check=False)
    if result.returncode != 0:
        return None
    try:
        return _parse_utc(result.stdout.strip(), field="source commit timestamp")
    except ExactDrawSidecarError:
        return None


def _first_index_commit(
    repo_root: Path,
    relative_index_path: str,
    *,
    generation_id: str,
    snapshot_path: str,
    snapshot_file_sha256: str,
) -> tuple[str, datetime] | None:
    """Find the first commit whose index blob contains the exact generation row."""

    _require_hash(snapshot_file_sha256, field="snapshot_file_sha256")
    for commit, committed_at in _history_for_path(repo_root, relative_index_path):
        blob = _git_bytes(repo_root, ["show", f"{commit}:{relative_index_path}"])
        if blob.returncode != 0:
            continue
        try:
            index = json.loads(blob.stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(index, Mapping) or not isinstance(index.get("snapshots"), list):
            continue
        for row in index["snapshots"]:
            if not isinstance(row, Mapping):
                continue
            if (
                row.get("generation_id") == generation_id
                and row.get("path") == snapshot_path
                and row.get("snapshot_file_sha256") == snapshot_file_sha256
            ):
                return commit, committed_at
    return None


def _is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    result = _git(repo_root, ["merge-base", "--is-ancestor", ancestor, descendant], check=False)
    return result.returncode == 0


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    _regular_file(path, label=label)
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ExactDrawSidecarError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ExactDrawSidecarError(f"{label} must be a JSON object: {path}")
    return value


def _sidecar_summary_parity(
    *,
    loaded: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> None:
    """Require exact-draw arrays to reproduce the certified compact summaries.

    Internal sidecar hashes prove that the sidecar is self-consistent, but a
    malicious or accidental rewrite could update those hashes together.  The
    selected compact snapshot is an independent commitment, so a sidecar is
    only considered verified after its vote/seat arrays reproduce every
    summary that those arrays can determine.  The local 12% exception flag is
    not present in this sidecar format; a non-zero certified value therefore
    fails closed instead of being silently treated as zero.
    """

    metadata = loaded.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ExactDrawSidecarError("exact-draw sidecar metadata is not an object")
    snapshot_model = snapshot.get("model")
    metadata_model = metadata.get("model")
    if not isinstance(snapshot_model, Mapping) or not isinstance(metadata_model, Mapping):
        raise ExactDrawSidecarError("exact-draw sidecar and snapshot models must be objects")
    metadata_identity = {
        "as_of": snapshot.get("as_of"),
        "election_date": snapshot.get("election_date"),
        "source_git_commit": snapshot.get("source_git_commit"),
        "samples": snapshot.get("samples"),
        "seed": snapshot.get("seed"),
        "model_version": snapshot_model.get("version"),
    }
    sidecar_identity = {
        "as_of": metadata.get("as_of"),
        "election_date": metadata.get("election_date"),
        "source_git_commit": metadata.get("source_git_commit"),
        "samples": metadata.get("samples"),
        "seed": metadata.get("seed"),
        "model_version": metadata_model.get("version"),
    }
    for field, expected in metadata_identity.items():
        # Legacy compact snapshots may omit seed.  Future certified snapshots
        # include it, and when present it must agree with sidecar metadata.
        if expected is not None and sidecar_identity.get(field) != expected:
            raise ExactDrawSidecarError(f"exact-draw metadata disagrees with certified snapshot for {field}")
    if metadata.get("generated_at_utc") is not None and snapshot.get("generated_at_utc") is not None:
        if metadata.get("generated_at_utc") != snapshot.get("generated_at_utc"):
            raise ExactDrawSidecarError(
                "exact-draw metadata disagrees with certified snapshot for generated_at_utc"
            )

    try:
        votes = np.asarray(loaded["vote_shares_pct"])
        seats = np.asarray(loaded["seats"])
    except (KeyError, TypeError) as exc:
        raise ExactDrawSidecarError("verified sidecar arrays are required for summary parity") from exc
    votes, seats = _validated_matrices(
        type("Draws", (), {"vote_shares_matrix": votes, "seats_matrix": seats})()
    )

    # ``compute_simulation_summary`` consumes fractions while the persisted
    # exact sidecar deliberately uses percentage points.
    from types import SimpleNamespace

    from scripts.simulator.summary import compute_simulation_summary

    summary_manifest = {
        "base_seed": int(metadata.get("seed", snapshot.get("seed", 0) or 0)),
    }
    summary_object, group_helper = compute_simulation_summary(
        as_of=str(snapshot.get("as_of")),
        election_date=str(snapshot.get("election_date")),
        vote_shares_matrix=votes / 100.0,
        seats_matrix=seats,
        manifest=summary_manifest,
    )
    summary_result = SimpleNamespace(
        summary=summary_object,
        manifest=summary_manifest,
        vote_shares_matrix=votes,
        seats_matrix=seats,
        quantization_audit=None,
        summarize_group=group_helper.summarize_group,
    )
    canonical = build_canonical_summary_dict(summary_result)

    expected_parties = snapshot.get("national_vote_summary")
    actual_parties = canonical.get("parties")
    if not isinstance(expected_parties, Mapping) or not isinstance(actual_parties, Mapping):
        raise ExactDrawSidecarError("certified snapshot lacks national vote summaries for sidecar parity")
    non_derivable_local_field = "prob_local_12pct_exception_sub_4pct"
    for party in (*PARLIAMENTARY_PARTIES_8, "REST"):
        expected = expected_parties.get(party)
        actual = actual_parties.get(party)
        if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
            raise ExactDrawSidecarError(f"certified snapshot lacks summary parity data for {party}")
        for field, expected_value in expected.items():
            if field == non_derivable_local_field:
                if expected_value != 0:
                    raise ExactDrawSidecarError(
                        "certified local 12% exception probability cannot be verified from vote/seat sidecar arrays"
                    )
                continue
            if actual.get(field) != expected_value:
                raise ExactDrawSidecarError(
                    f"exact-draw sidecar summary parity failed for {party}.{field}"
                )

    expected_groups = snapshot.get("group_probabilities")
    if not isinstance(expected_groups, Mapping) or expected_groups != canonical.get("blocs"):
        raise ExactDrawSidecarError("exact-draw sidecar coalition summary parity failed")

    expected_thresholds = snapshot.get("threshold_probabilities_4pct")
    actual_thresholds = {
        party: canonical["parties"][party]["prob_above_4pct"]
        for party in PARLIAMENTARY_PARTIES_8
    }
    if not isinstance(expected_thresholds, Mapping) or dict(expected_thresholds) != actual_thresholds:
        raise ExactDrawSidecarError("exact-draw sidecar threshold summary parity failed")

    expected_seats = snapshot.get("seat_summary")
    actual_seats = {
        party: {
            "mean": canonical["parties"][party]["seats_mean"],
            "median": canonical["parties"][party]["seats_median"],
            "p05": canonical["parties"][party]["seats_p05"],
            "p95": canonical["parties"][party]["seats_p95"],
        }
        for party in PARLIAMENTARY_PARTIES_8
    }
    if not isinstance(expected_seats, Mapping) or dict(expected_seats) != actual_seats:
        raise ExactDrawSidecarError("exact-draw sidecar seat summary parity failed")


def validate_exact_draw_sidecar(
    draws_path: Path | str,
    metadata_path: Path | str,
    *,
    certified_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an exact sidecar and all summaries against its snapshot."""

    generation_id = certified_snapshot.get("generation_id")
    payload_hash = certified_snapshot.get("deterministic_payload_sha256")
    if not isinstance(generation_id, str) or not generation_id:
        raise ExactDrawSidecarError("certified snapshot has no generation_id")
    loaded = load_verified_draw_sidecar(
        draws_path,
        metadata_path,
        expected_generation_id=generation_id,
        expected_payload_hash=_require_hash(
            payload_hash,
            field="certified_snapshot.deterministic_payload_sha256",
        ),
        include_arrays=True,
    )
    _sidecar_summary_parity(loaded=loaded, snapshot=certified_snapshot)
    return loaded


def _validate_archive_entry(
    *,
    archive_root: Path,
    repo_root: Path,
    entry: Mapping[str, Any],
    cutoff: datetime,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    generation_id = entry.get("generation_id")
    if not isinstance(generation_id, str) or not generation_id:
        return None  # pre-1.1 snapshots have no generation identity
    relative = entry.get("path")
    if not isinstance(relative, str) or not relative:
        raise ExactDrawSidecarError("archive entry has no snapshot path")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ExactDrawSidecarError("archive entry path escapes archive root")
    if len(relative_path.parts) != 2 or relative_path.parts[0] != generation_id or relative_path.name != "snapshot.json":
        raise ExactDrawSidecarError(
            "archive entry path must be <generation_id>/snapshot.json"
        )
    snapshot_path = (archive_root / relative_path).resolve()
    if snapshot_path.parent.parent != archive_root.resolve():
        raise ExactDrawSidecarError("archive entry path must remain below archive root")
    snapshot = _load_json(snapshot_path, label="archive snapshot")
    snapshot_file_sha256 = _require_hash(
        entry.get("snapshot_file_sha256"),
        field="snapshot_file_sha256",
    )
    if compute_file_sha256(snapshot_path) != snapshot_file_sha256:
        raise ExactDrawSidecarError(f"archive snapshot hash mismatch: {snapshot_path}")
    if snapshot.get("snapshot_id") != entry.get("snapshot_id"):
        raise ExactDrawSidecarError("archive snapshot identity does not match index")
    payload = snapshot.get("deterministic_payload_sha256")
    if payload != entry.get("deterministic_payload_sha256"):
        raise ExactDrawSidecarError("archive snapshot payload hash does not match index")
    _require_hash(payload, field="deterministic_payload_sha256")
    if snapshot.get("generation_id") != generation_id:
        raise ExactDrawSidecarError("archive snapshot generation_id does not match index")
    model = snapshot.get("model")
    if not isinstance(model, Mapping):
        raise ExactDrawSidecarError("archive snapshot model must be an object")
    if model.get("name") != "ElectionSimulator":
        return None
    index_snapshot_fields = {
        "snapshot_date": snapshot.get("snapshot_date"),
        "as_of": snapshot.get("as_of"),
        "election_date": snapshot.get("election_date"),
        "generated_at_utc": snapshot.get("generated_at_utc"),
        "source_git_commit": snapshot.get("source_git_commit"),
        "seed": snapshot.get("seed"),
        "deterministic_payload_sha256": payload,
    }
    for field, snapshot_value in index_snapshot_fields.items():
        # A few legacy test/production snapshots did not carry every field
        # duplicated by the index (notably ``seed``).  Compare only values
        # that are actually present in the snapshot; the index remains the
        # selector, while the snapshot is the immutable forecast evidence.
        if field in entry and field in snapshot and entry.get(field) != snapshot_value:
            raise ExactDrawSidecarError(f"archive index disagrees with snapshot for {field}")
    if "model_version" in entry and entry.get("model_version") != model.get("version"):
        raise ExactDrawSidecarError("archive index disagrees with snapshot for model_version")
    if snapshot.get("source_worktree_clean") is not True:
        return None
    generated = _parse_utc(snapshot.get("generated_at_utc"), field="generated_at_utc")
    if generated > cutoff:
        return None
    source_commit = snapshot.get("source_git_commit")
    if not isinstance(source_commit, str) or not _commit_exists(repo_root, source_commit):
        return None
    try:
        archive_relative = snapshot_path.relative_to(repo_root).as_posix()
    except ValueError:
        # A copied archive has no verifiable first-containing commit.  This is
        # deliberately ineligible rather than silently trusting filesystem
        # mtime or a caller-provided timestamp.
        return None
    first = _first_archive_commit(
        repo_root,
        archive_relative,
        expected_sha256=snapshot_file_sha256,
    )
    if first is None:
        return None
    first_commit, first_at = first
    if first_at > cutoff:
        return None
    index_path = (archive_root / "index.json").resolve()
    try:
        index_relative = index_path.relative_to(repo_root).as_posix()
    except ValueError:
        return None
    first_index = _first_index_commit(
        repo_root,
        index_relative,
        generation_id=generation_id,
        snapshot_path=relative_path.as_posix(),
        snapshot_file_sha256=snapshot_file_sha256,
    )
    if first_index is None:
        return None
    first_index_commit, first_index_at = first_index
    if first_index_at > cutoff:
        return None
    if not _is_ancestor(repo_root, source_commit, first_commit):
        return None
    if not _is_ancestor(repo_root, source_commit, first_index_commit):
        return None
    source_commit_at = _commit_timestamp(repo_root, source_commit)
    if source_commit_at is None or source_commit_at > cutoff:
        return None
    provenance = {
        "selected_at_cutoff_utc": cutoff.isoformat().replace("+00:00", "Z"),
        "generated_at_utc": generated.isoformat().replace("+00:00", "Z"),
        "first_archive_commit": first_commit,
        "first_archive_commit_at_utc": first_at.isoformat().replace("+00:00", "Z"),
        "first_index_commit": first_index_commit,
        "first_index_commit_at_utc": first_index_at.isoformat().replace("+00:00", "Z"),
        "source_git_commit": source_commit,
        "source_commit_at_utc": source_commit_at.isoformat().replace("+00:00", "Z"),
        "source_commit_resolved": True,
        "source_worktree_clean": True,
        "selection_rule": (
            "latest certified generation whose exact snapshot blob, matching index row, "
            "and certified source commit all existed by cutoff UTC"
        ),
        "archive_snapshot_path": archive_relative,
        "archive_index_entry": dict(entry),
    }
    return snapshot, provenance


def _discover_verified_sidecar(
    *,
    archive_root: Path,
    repo_root: Path,
    cutoff: datetime,
    snapshot: Mapping[str, Any],
    provenance: Mapping[str, Any],
    include_draws: bool,
    include_sidecar_bytes: bool,
) -> dict[str, Any]:
    generation_dir = archive_root / str(snapshot["generation_id"])
    candidates = (
        (generation_dir / SIDECAR_DRAWS_FILENAME, generation_dir / SIDECAR_METADATA_FILENAME),
        (generation_dir / "exact_draws.npz", generation_dir / "exact_draws.json"),
    )
    existing = [(draws, metadata) for draws, metadata in candidates if os.path.lexists(draws) or os.path.lexists(metadata)]
    if not existing:
        return {
            "status": "UNAVAILABLE_NO_VERIFIED_DRAWS",
            "reason": "No exact-draw sidecar is present for the selected certified generation; compact summaries are not draws.",
        }
    draws_path, metadata_path = existing[0]
    try:
        loaded = validate_exact_draw_sidecar(
            draws_path,
            metadata_path,
            certified_snapshot=snapshot,
        )
        try:
            draws_relative = draws_path.relative_to(repo_root).as_posix()
            metadata_relative = metadata_path.relative_to(repo_root).as_posix()
        except ValueError as exc:
            raise ExactDrawSidecarError(
                "exact-draw sidecar is outside the verifiable repository checkout"
            ) from exc
        draws_first = _first_archive_commit(
            repo_root,
            draws_relative,
            expected_sha256=compute_file_sha256(draws_path),
        )
        metadata_first = _first_archive_commit(
            repo_root,
            metadata_relative,
            expected_sha256=compute_file_sha256(metadata_path),
        )
        snapshot_first = provenance.get("first_archive_commit")
        if draws_first is None or metadata_first is None:
            raise ExactDrawSidecarError("exact-draw sidecar has no verifiable first-containing commit")
        if draws_first[0] != snapshot_first or metadata_first[0] != snapshot_first:
            raise ExactDrawSidecarError(
                "exact-draw sidecar was not committed atomically with the certified snapshot"
            )
        if draws_first[1] > cutoff or metadata_first[1] > cutoff:
            raise ExactDrawSidecarError("exact-draw sidecar first-containing commit is after cutoff")
    except ExactDrawSidecarError as exc:
        return {
            "status": "UNVERIFIED_SIDECAR",
            "reason": str(exc),
            "draws_path": str(draws_path.relative_to(archive_root)),
            "metadata_path": str(metadata_path.relative_to(archive_root)),
        }
    sidecar: dict[str, Any] = {
        "status": "VERIFIED",
        "draws_path": str(draws_path.relative_to(archive_root)),
        "metadata_path": str(metadata_path.relative_to(archive_root)),
        "draws_file_sha256": loaded["metadata"]["draws_file_sha256"],
        "metadata": loaded["metadata"],
        "first_archive_commit": draws_first[0],
        "first_archive_commit_at_utc": draws_first[1].isoformat().replace("+00:00", "Z"),
    }
    if include_draws:
        sidecar["vote_shares_pct"] = loaded["vote_shares_pct"].tolist()
        sidecar["seats"] = loaded["seats"].tolist()
    if include_sidecar_bytes:
        sidecar["draws_bytes_base64"] = base64.b64encode(draws_path.read_bytes()).decode("ascii")
        sidecar["metadata_bytes_base64"] = base64.b64encode(metadata_path.read_bytes()).decode("ascii")
    return sidecar


def collect_latest_certified_generation(
    archive_root: Path | str,
    cutoff_utc: datetime | str,
    repo_root: Path | str,
    *,
    include_draws: bool = False,
    include_sidecar_bytes: bool = False,
    reproduce_missing_draws: bool = False,
) -> dict[str, Any]:
    """Select one certified ES generation available at a cutoff.

    The returned object is JSON-serializable by default.  ``forecast`` is the
    immutable compact snapshot; ``provenance`` records why it was eligible;
    ``exact_draws`` reports a verified sidecar path when one exists.  Set
    ``include_draws`` or ``include_sidecar_bytes`` only for an explicit
    exchange boundary—the default avoids copying 100,000 rows into a status
    object.  ``reproduce_missing_draws`` enables the conservative local replay
    fallback described by :func:`replay_certified_generation`; it is disabled
    by default because capture workflows should make that expensive action
    explicit.

    Selection is from current ``HEAD`` history only.  An uncommitted snapshot,
    or one whose first archive commit was made after ``cutoff_utc``, is not a
    prospective generation and is excluded.  No later generation is selected
    merely because it has a more favorable result.
    """

    try:
        cutoff = _parse_utc(
            cutoff_utc.isoformat() if isinstance(cutoff_utc, datetime) else str(cutoff_utc),
            field="cutoff_utc",
        )
    except AttributeError as exc:
        raise ExactDrawSidecarError("cutoff_utc must be an aware datetime or ISO-8601 string") from exc
    archive = Path(archive_root).resolve()
    repo = Path(repo_root).resolve()
    index_path = archive / "index.json"
    if not index_path.is_file() or index_path.is_symlink():
        return {
            "status": "ARCHIVE_UNAVAILABLE",
            "forecast": None,
            "provenance": {"archive_root": str(archive), "cutoff_utc": cutoff.isoformat().replace("+00:00", "Z")},
            "diagnostics": [f"archive index is missing: {index_path}"],
        }
    try:
        index = _load_json(index_path, label="archive index")
        _validate_index(index)
    except (ExactDrawSidecarError, ValueError) as exc:
        return {
            "status": "ARCHIVE_INVALID",
            "forecast": None,
            "provenance": {"archive_root": str(archive), "cutoff_utc": cutoff.isoformat().replace("+00:00", "Z")},
            "diagnostics": [str(exc)],
        }

    eligible: list[tuple[datetime, str, dict[str, Any], dict[str, Any]]] = []
    diagnostics: list[str] = []
    for raw_entry in index.get("snapshots", []):
        try:
            selected = _validate_archive_entry(
                archive_root=archive,
                repo_root=repo,
                entry=raw_entry,
                cutoff=cutoff,
            )
        except ExactDrawSidecarError as exc:
            diagnostics.append(str(exc))
            continue
        if selected is None:
            continue
        snapshot, provenance = selected
        generated = _parse_utc(snapshot["generated_at_utc"], field="generated_at_utc")
        eligible.append((generated, str(snapshot["generation_id"]), snapshot, provenance))
    if not eligible:
        return {
            "status": "NO_CERTIFIED_GENERATION",
            "forecast": None,
            "provenance": {
                "archive_root": str(archive),
                "cutoff_utc": cutoff.isoformat().replace("+00:00", "Z"),
                "selection_rule": "latest certified generation with generated_at_utc and first archive commit <= cutoff UTC",
            },
            "diagnostics": diagnostics,
        }

    _generated, _generation, snapshot, provenance = max(eligible, key=lambda row: (row[0], row[1]))
    exact_draws = _discover_verified_sidecar(
        archive_root=archive,
        repo_root=repo,
        cutoff=cutoff,
        snapshot=snapshot,
        provenance=provenance,
        include_draws=include_draws,
        include_sidecar_bytes=include_sidecar_bytes,
    )
    if (
        reproduce_missing_draws
        and exact_draws["status"] == "UNAVAILABLE_NO_VERIFIED_DRAWS"
    ):
        try:
            replay, replay_evidence = replay_certified_generation(snapshot, repo)
            files = build_exact_draw_sidecar_files(
                replay,
                generation_id=str(snapshot["generation_id"]),
                certified_snapshot=snapshot,
            )
            metadata = json.loads(files[SIDECAR_METADATA_FILENAME].decode("utf-8"))
            exact_draws = {
                "status": "REPLAY_VERIFIED",
                "metadata": metadata,
                "replay_evidence": replay_evidence,
            }
            if include_draws:
                exact_draws["vote_shares_pct"] = replay.vote_shares_matrix.tolist()
                exact_draws["seats"] = replay.seats_matrix.tolist()
            if include_sidecar_bytes:
                exact_draws["draws_bytes_base64"] = base64.b64encode(
                    files[SIDECAR_DRAWS_FILENAME]
                ).decode("ascii")
                exact_draws["metadata_bytes_base64"] = base64.b64encode(
                    files[SIDECAR_METADATA_FILENAME]
                ).decode("ascii")
        except ExactDrawSidecarError as exc:
            exact_draws = {
                "status": "REPLAY_UNVERIFIED",
                "reason": str(exc),
            }
    status = (
        "FOUND_WITH_VERIFIED_DRAWS"
        if exact_draws["status"] == "VERIFIED"
        or exact_draws["status"] == "REPLAY_VERIFIED"
        else "FOUND_NO_VERIFIED_DRAWS"
        if exact_draws["status"] == "UNAVAILABLE_NO_VERIFIED_DRAWS"
        else "FOUND_WITH_UNVERIFIED_SIDECAR"
    )
    provenance = dict(provenance)
    provenance["selection_cutoff_utc"] = cutoff.isoformat().replace("+00:00", "Z")
    return {
        "status": status,
        "forecast": snapshot,
        "provenance": provenance,
        "exact_draws": exact_draws,
        "diagnostics": diagnostics,
    }


__all__ = [
    "ExactDrawSidecarError",
    "SIDECAR_DRAWS_FILENAME",
    "SIDECAR_METADATA_FILENAME",
    "build_exact_draw_metadata",
    "build_exact_draw_sidecar_files",
    "collect_latest_certified_generation",
    "load_verified_draw_sidecar",
    "replay_certified_generation",
    "validate_exact_draw_sidecar",
    "write_exact_draw_sidecar",
]
