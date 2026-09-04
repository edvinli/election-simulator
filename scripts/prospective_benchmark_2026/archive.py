"""Append-only, content-addressed archive for the 2026 benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Mapping

try:  # pragma: no cover - the production runner is Linux; keep imports portable.
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback remains fail-closed per process.
    fcntl = None  # type: ignore[assignment]

from .time_rules import CaptureTimeError, capture_id_for_date, classify_capture_time


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE_ROOT = REPOSITORY_ROOT / "data" / "processed" / "prospective_benchmark_2026"
INDEX_SCHEMA_VERSION = "1.0"
CAPTURE_SCHEMA_VERSION = "1.0"
AMENDMENT_SCHEMA_VERSION = "1.0"
MODEL_NAMES = ("election_simulator", "botten_ada")
MODEL_STATUSES = {
    "AVAILABLE",
    "SOURCE_UNAVAILABLE",
    "PARSE_FAILED",
    "SOURCE_STALE",
    "PARITY_UNVERIFIED",
    # These statuses are retained for a future independently verified
    # publication object.  They are never inferred by this archive layer.
    "VERIFIED",
    "REPLAY_VERIFIED",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TIMING_FIELDS = (
    "scheduled_date",
    "benchmark_cutoff",
    "benchmark_cutoff_europe_stockholm",
    "retrieved_at_utc",
    "retrieved_at_europe_stockholm",
    "timing_status",
    "timing_eligible",
)
_AMENDMENT_REF_FIELDS = (
    "amendment_number",
    "path",
    "sha256",
    "primary_scoring_effect",
)


class ArchiveValidationError(ValueError):
    """Raised when archive evidence is incomplete or cryptographically inconsistent."""


class CaptureCollisionError(FileExistsError):
    """Raised when a scheduled slot or immutable capture already exists."""


@dataclass(frozen=True)
class ModelCapture:
    status: str
    forecast: Mapping[str, Any]
    provenance: Mapping[str, Any]
    files: Mapping[str, bytes] | None = None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveValidationError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArchiveValidationError(f"Expected JSON object: {path}")
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ArchiveValidationError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _regular_file(path: Path, *, label: str) -> None:
    if os.path.islink(path) or not path.is_file():
        raise ArchiveValidationError(f"{label} must be a regular file: {path}")


def _fsync_directory(path: Path) -> None:
    """Flush directory metadata where the host filesystem supports it."""

    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        # Some platforms/filesystems reject fsync on directories.  The file
        # itself was already flushed; validation remains the final gate.
        pass
    finally:
        os.close(fd)


@contextmanager
def _archive_lock(root: Path):
    """Serialize appenders without leaving lock artifacts in the repository.

    GitHub Actions also uses a workflow-level concurrency group, but a local
    retry and an action retry can still overlap.  The lock lives in the host
    temporary directory so a successful run changes only the benchmark paths.
    ``flock`` is advisory and released automatically if a process dies.
    """

    key = hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()[:32]
    lock_path = Path(tempfile.gettempdir()) / f"prospective-benchmark-2026-{key}.lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _read_sidecar_hash(path: Path, *, expected_name: str) -> str:
    _regular_file(path, label="Amendment SHA-256 sidecar")
    try:
        tokens = path.read_text(encoding="utf-8").split()
    except (OSError, UnicodeDecodeError) as exc:
        raise ArchiveValidationError(f"Cannot read amendment SHA-256 sidecar {path}: {exc}") from exc
    if not tokens:
        raise ArchiveValidationError(f"Empty amendment SHA-256 sidecar: {path}")
    digest = _require_sha256(tokens[0], field=f"{path.name} digest")
    if len(tokens) > 1 and tokens[1] != expected_name:
        raise ArchiveValidationError(f"Amendment SHA-256 sidecar names the wrong file: {path}")
    return digest


def _parse_utc_timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ArchiveValidationError(f"{field} must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ArchiveValidationError(f"{field} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ArchiveValidationError(f"{field} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _load_amendment_refs(root: Path, *, protocol_hash: str) -> list[dict[str, Any]]:
    """Validate immutable amendment artifacts and return canonical references."""

    directory = root / "amendments"
    if not os.path.lexists(directory):
        return []
    if directory.is_symlink() or not directory.is_dir():
        raise ArchiveValidationError(f"Amendment directory is missing or unsafe: {directory}")

    entries = list(directory.iterdir())
    for entry in entries:
        if entry.is_symlink() or entry.is_dir() or not entry.is_file():
            raise ArchiveValidationError(f"Unexpected amendment path: {entry}")
        if entry.suffix not in {".json", ".sha256"}:
            raise ArchiveValidationError(f"Unexpected amendment artifact: {entry.name}")
    json_paths = sorted((entry for entry in entries if entry.suffix == ".json"), key=lambda p: p.name)
    json_stems = {entry.stem for entry in entries if entry.suffix == ".json"}
    sidecar_stems = {entry.stem for entry in entries if entry.suffix == ".sha256"}
    if json_stems != sidecar_stems:
        raise ArchiveValidationError("Every amendment JSON must have exactly one SHA-256 sidecar")
    refs: list[dict[str, Any]] = []
    numbers: list[int] = []
    for json_path in json_paths:
        sidecar = json_path.with_suffix(".sha256")
        if not sidecar.is_file() or sidecar.is_symlink():
            raise ArchiveValidationError(f"Amendment SHA-256 sidecar is missing: {sidecar}")
        digest = sha256_file(json_path)
        sidecar_digest = _read_sidecar_hash(sidecar, expected_name=json_path.name)
        if digest != sidecar_digest:
            raise ArchiveValidationError(f"Amendment SHA-256 mismatch: {json_path.name}")
        amendment = _read_json(json_path)
        if amendment.get("schema_version") != AMENDMENT_SCHEMA_VERSION:
            raise ArchiveValidationError(f"Unsupported amendment schema: {json_path.name}")
        number = amendment.get("amendment_number")
        if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
            raise ArchiveValidationError(f"Amendment number is invalid: {json_path.name}")
        prefix = json_path.stem.split("-", 1)[0]
        if not prefix.isdigit() or int(prefix) != number:
            raise ArchiveValidationError(
                f"Amendment filename must begin with its numeric amendment number: {json_path.name}"
            )
        amendment_id = amendment.get("amendment_id")
        if not isinstance(amendment_id, str) or not amendment_id or "/" in amendment_id or "\\" in amendment_id:
            raise ArchiveValidationError(f"Amendment ID is invalid: {json_path.name}")
        _parse_utc_timestamp(amendment.get("created_at_utc"), field=f"{json_path.name}.created_at_utc")
        if amendment.get("original_protocol_sha256") != protocol_hash:
            raise ArchiveValidationError(f"Amendment references another protocol: {json_path.name}")
        if amendment.get("immutable") is not True:
            raise ArchiveValidationError(f"Amendment is not marked immutable: {json_path.name}")
        for field in ("reason", "primary_scoring_effect"):
            if not isinstance(amendment.get(field), str) or not amendment[field].strip():
                raise ArchiveValidationError(f"Amendment {field} is required: {json_path.name}")
        numbers.append(number)
        refs.append({
            "amendment_number": number,
            "path": f"amendments/{json_path.name}",
            "sha256": digest,
            "primary_scoring_effect": amendment["primary_scoring_effect"],
        })

    expected_numbers = list(range(1, len(numbers) + 1))
    if sorted(numbers) != expected_numbers:
        raise ArchiveValidationError(
            f"Amendment numbers must be sequential starting at 1; got {sorted(numbers)}"
        )
    return sorted(refs, key=lambda ref: int(ref["amendment_number"]))


def _validate_index_amendments(index: Mapping[str, Any], active: list[dict[str, Any]]) -> None:
    value = index.get("amendments")
    if value is None:
        if active:
            raise ArchiveValidationError("Archive index omits the immutable amendment catalog")
        return
    if not isinstance(value, list) or value != active:
        raise ArchiveValidationError("Archive index amendment catalog does not match immutable artifacts")


def _validate_capture_amendments(capture_dir: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = manifest.get("amendments", [])
    if not isinstance(value, list):
        raise ArchiveValidationError("Capture amendments must be a list")
    protocol_hash = str(manifest.get("protocol_sha256", ""))
    active = _load_amendment_refs(capture_dir.parent.parent, protocol_hash=protocol_hash)
    active_by_path = {str(item["path"]): item for item in active}
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != set(_AMENDMENT_REF_FIELDS):
            raise ArchiveValidationError("Capture amendment reference has an invalid shape")
        path = item.get("path")
        if not isinstance(path, str) or path in seen or path not in active_by_path:
            raise ArchiveValidationError("Capture references an unknown or duplicate amendment")
        expected = active_by_path[path]
        if item != expected:
            raise ArchiveValidationError("Capture amendment reference does not match its immutable artifact")
        seen.add(path)
        normalized.append(dict(item))
    return normalized


def _validate_capture_id(capture_id: Any) -> str:
    if not isinstance(capture_id, str) or not capture_id:
        raise ArchiveValidationError("capture_id must be a non-empty path-safe string")
    if capture_id in {".", ".."} or "/" in capture_id or "\\" in capture_id:
        raise ArchiveValidationError(f"Unsafe capture_id: {capture_id!r}")
    path = PurePosixPath(capture_id)
    if path.name != capture_id or any(part in {"", ".", ".."} for part in path.parts):
        raise ArchiveValidationError(f"Unsafe capture_id: {capture_id!r}")
    return capture_id


def _validate_timing_for_append(timing: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(timing, Mapping):
        raise ArchiveValidationError("Capture timing must be an object")
    missing = [field for field in _TIMING_FIELDS if field not in timing]
    if missing:
        raise ArchiveValidationError(f"Capture timing lacks required fields: {missing}")
    try:
        expected = classify_capture_time(
            str(timing["scheduled_date"]),
            str(timing["retrieved_at_utc"]),
            durable=True,
        ).to_dict()
    except (CaptureTimeError, TypeError, ValueError) as exc:
        raise ArchiveValidationError(f"Invalid or retroactive capture timing: {exc}") from exc
    for field in _TIMING_FIELDS:
        if timing.get(field) != expected[field]:
            raise ArchiveValidationError(
                f"Capture timing field {field!r} does not match explicit Stockholm rules"
            )
    return expected


def protocol_sha256(root: Path = DEFAULT_ARCHIVE_ROOT) -> str:
    root = _resolve_archive_root(root)
    protocol = root / "protocol.json"
    sidecar = root / "protocol.sha256"
    if protocol.is_symlink() or sidecar.is_symlink() or not protocol.is_file() or not sidecar.is_file():
        raise ArchiveValidationError("Frozen protocol and SHA-256 sidecar are required")
    try:
        tokens = sidecar.read_text(encoding="utf-8").split()
    except (OSError, UnicodeDecodeError) as exc:
        raise ArchiveValidationError(f"Cannot read protocol SHA-256 sidecar: {exc}") from exc
    if not tokens:
        raise ArchiveValidationError("Frozen protocol SHA-256 sidecar is empty")
    expected = tokens[0]
    _require_sha256(expected, field="protocol.sha256")
    if len(tokens) > 1 and tokens[1] != protocol.name:
        raise ArchiveValidationError("Frozen protocol SHA-256 sidecar names the wrong file")
    actual = sha256_file(protocol)
    if expected != actual:
        raise ArchiveValidationError("Frozen protocol SHA-256 does not match protocol.sha256")
    return actual


def _safe_relative_path(value: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ArchiveValidationError(f"Unsafe relative archive path: {value!r}")
    return relative


def _resolve_archive_root(root: Path | str) -> Path:
    """Resolve the archive root without following an archive-root symlink."""

    raw_root = Path(root)
    if raw_root.is_symlink():
        raise ArchiveValidationError(f"Archive root is an unsafe symlink: {raw_root}")
    return raw_root.resolve()


def _atomic_replace(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _entry_hash(entry: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in entry.items() if key != "entry_sha256"}
    return sha256_bytes(canonical_json_bytes(unsigned))


def _load_index(root: Path) -> dict[str, Any]:
    index_path = root / "index.json"
    if index_path.is_symlink() or not index_path.is_file():
        raise ArchiveValidationError(f"Archive index is unsafe: {index_path}")
    index = _read_json(index_path)
    if index.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise ArchiveValidationError(f"Unsupported index schema: {index.get('schema_version')}")
    if not isinstance(index.get("captures"), list):
        raise ArchiveValidationError("Archive index captures must be a list")
    actual_protocol_hash = protocol_sha256(root)
    if index.get("protocol_sha256") != actual_protocol_hash:
        raise ArchiveValidationError("Index protocol SHA-256 does not match the frozen protocol")
    _validate_index_amendments(index, _load_amendment_refs(root, protocol_hash=actual_protocol_hash))
    for row in index["captures"]:
        if not isinstance(row, dict):
            raise ArchiveValidationError("Archive index capture rows must be objects")
    return index


def _materialize_capture(
    staging: Path,
    *,
    capture_id: str,
    timing: Mapping[str, Any],
    models: Mapping[str, ModelCapture],
    frozen_protocol_sha256: str,
    amendments: list[dict[str, Any]],
) -> dict[str, Any]:
    if set(models) != set(MODEL_NAMES):
        raise ArchiveValidationError(f"Capture must contain exactly {list(MODEL_NAMES)}")
    file_records: list[dict[str, Any]] = []
    model_statuses: dict[str, str] = {}
    for model_name in MODEL_NAMES:
        model = models[model_name]
        if not isinstance(model.status, str) or model.status not in MODEL_STATUSES:
            raise ArchiveValidationError(f"Unknown status for {model_name}: {model.status!r}")
        if not isinstance(model.forecast, Mapping) or not isinstance(model.provenance, Mapping):
            raise ArchiveValidationError(f"{model_name} forecast and provenance must be JSON objects")
        model_statuses[model_name] = model.status
        payloads: dict[str, bytes] = {
            f"{model_name}/forecast.json": canonical_json_bytes(dict(model.forecast)),
            f"{model_name}/provenance.json": canonical_json_bytes(dict(model.provenance)),
        }
        for relative_name, content in (model.files or {}).items():
            nested = _safe_relative_path(relative_name)
            if nested.parts[0] in MODEL_NAMES:
                raise ArchiveValidationError("Model extra-file paths must be relative to that model directory")
            archive_name = str(PurePosixPath(model_name) / nested)
            if archive_name in payloads or archive_name.endswith("/manifest.json"):
                raise ArchiveValidationError(f"Duplicate or reserved capture file: {archive_name}")
            if not isinstance(content, bytes):
                raise ArchiveValidationError(f"Extra capture file must be bytes: {archive_name}")
            payloads[archive_name] = content
        for relative_name, content in sorted(payloads.items()):
            destination = staging.joinpath(*PurePosixPath(relative_name).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            file_records.append({
                "path": relative_name,
                "sha256": sha256_bytes(content),
                "byte_size": len(content),
            })
    statuses = set(model_statuses.values())
    overall_status = "COMPLETE" if statuses == {"AVAILABLE"} else "PARTIAL_OR_FAILED"
    manifest = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "capture_id": capture_id,
        "protocol_sha256": frozen_protocol_sha256,
        "amendments": [dict(item) for item in amendments],
        **dict(timing),
        "capture_status": overall_status,
        "model_statuses": model_statuses,
        "files": file_records,
    }
    (staging / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return manifest


def _validate_capture_directory(
    capture_dir: Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_protocol_sha256: str,
    expected_capture_id: str | None = None,
) -> dict[str, Any]:
    if not capture_dir.is_dir() or capture_dir.is_symlink():
        raise ArchiveValidationError(f"Capture directory missing or unsafe: {capture_dir}")
    if any(path.is_symlink() for path in capture_dir.rglob("*")):
        raise ArchiveValidationError(f"Capture contains an unsafe symlink: {capture_dir}")
    manifest_path = capture_dir / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ArchiveValidationError(f"Capture manifest missing or unsafe: {manifest_path}")
    actual_manifest_hash = sha256_file(manifest_path)
    if expected_manifest_sha256 is not None:
        expected_manifest_sha256 = _require_sha256(
            expected_manifest_sha256,
            field=f"{capture_dir.name}/manifest.json sha256",
        )
        if actual_manifest_hash != expected_manifest_sha256:
            raise ArchiveValidationError(f"Capture manifest hash mismatch: {capture_dir.name}")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != CAPTURE_SCHEMA_VERSION:
        raise ArchiveValidationError(f"Unsupported capture schema: {manifest.get('schema_version')}")
    canonical_capture_id = expected_capture_id or capture_dir.name
    _validate_capture_id(canonical_capture_id)
    if manifest.get("capture_id") != canonical_capture_id:
        raise ArchiveValidationError("Capture ID does not match its directory")
    if manifest.get("protocol_sha256") != expected_protocol_sha256:
        raise ArchiveValidationError("Capture does not reference the frozen protocol")
    _validate_timing_for_append(manifest)
    model_statuses = manifest.get("model_statuses")
    if not isinstance(model_statuses, Mapping) or set(model_statuses) != set(MODEL_NAMES):
        raise ArchiveValidationError("Capture model status set is incomplete")
    if any(status not in MODEL_STATUSES for status in model_statuses.values()):
        raise ArchiveValidationError("Capture contains an unknown model status")
    _validate_capture_amendments(capture_dir, manifest)
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ArchiveValidationError("Capture manifest has no file inventory")
    declared: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ArchiveValidationError("Capture file record must be an object")
        relative = _safe_relative_path(str(record.get("path", "")))
        if relative.parts[0] not in MODEL_NAMES or str(relative) in declared:
            raise ArchiveValidationError(f"Invalid or duplicate captured file: {relative}")
        declared.add(str(relative))
        path = capture_dir.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise ArchiveValidationError(f"Captured file missing or unsafe: {relative}")
        byte_size = record.get("byte_size")
        digest = record.get("sha256")
        if not isinstance(byte_size, int) or isinstance(byte_size, bool) or byte_size < 0:
            raise ArchiveValidationError(f"Captured file byte_size is invalid: {relative}")
        _require_sha256(digest, field=f"captured file {relative} sha256")
        if path.stat().st_size != byte_size or sha256_file(path) != digest:
            raise ArchiveValidationError(f"Captured file integrity failure: {relative}")
    required = {
        f"{name}/forecast.json" for name in MODEL_NAMES
    } | {f"{name}/provenance.json" for name in MODEL_NAMES}
    if not required.issubset(declared):
        raise ArchiveValidationError("Capture lacks required forecast/provenance files")
    actual_files = {
        path.relative_to(capture_dir).as_posix()
        for path in capture_dir.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual_files != declared:
        raise ArchiveValidationError("Capture contains undeclared or missing evidence files")
    return manifest


def append_capture(
    *,
    root: Path | str,
    capture_id: str,
    timing: Mapping[str, Any],
    models: Mapping[str, ModelCapture],
) -> tuple[Path, dict[str, Any]]:
    """Atomically install one capture, then append its hash-chained index row.

    A crash after directory installation but before index replacement leaves an
    auditable orphan. Re-running this function indexes that exact validated
    orphan; it never fetches replacement evidence into the occupied directory.
    """

    archive_root = _resolve_archive_root(root)
    with _archive_lock(archive_root):
        # All checks and both immutable installs happen under one lock.  This
        # prevents two retries from deriving the same previous-entry hash and
        # one replacing the other's newly appended index row.
        index = _load_index(archive_root)
        frozen_hash = protocol_sha256(archive_root)
        # Validate every already-indexed row before installing a new immutable
        # directory.  A valid complete orphan is the one intentional exception:
        # it can remain after a crash between directory install and index
        # replacement and is safely recovered below without replacement.
        validate_archive(archive_root, allow_unindexed_orphans=True)
        expected_timing = _validate_timing_for_append(timing)
        scheduled_date = expected_timing["scheduled_date"]
        captures = index["captures"]

        # A same-date collision is checked before capture-id validation so a
        # second producer attempting a different id receives an explicit slot
        # collision instead of accidentally creating a second directory.
        if any(row.get("scheduled_date") == scheduled_date for row in captures):
            raise CaptureCollisionError(f"Scheduled slot is already indexed: {scheduled_date}")
        if any(row.get("capture_id") == capture_id for row in captures):
            raise CaptureCollisionError(f"Capture ID is already indexed: {capture_id}")
        _validate_capture_id(capture_id)
        expected_capture_id = capture_id_for_date(scheduled_date)
        if capture_id != expected_capture_id:
            raise ArchiveValidationError(
                f"capture_id must be the UTC cutoff id {expected_capture_id!r} for {scheduled_date}"
            )

        captures_root = archive_root / "captures"
        if os.path.lexists(captures_root) and (captures_root.is_symlink() or not captures_root.is_dir()):
            raise ArchiveValidationError(f"Capture root is missing or unsafe: {captures_root}")
        captures_root.mkdir(parents=True, exist_ok=True)
        hidden_paths = [path.name for path in captures_root.iterdir() if path.name.startswith(".")]
        if hidden_paths:
            raise ArchiveValidationError(
                f"Unfinished staging paths remain in captures root: {hidden_paths}"
            )
        active_amendments = _load_amendment_refs(archive_root, protocol_hash=frozen_hash)
        destination = captures_root / capture_id
        indexed_ids = {str(row["capture_id"]) for row in captures}
        # A complete orphan can only be recovered for the exact requested
        # slot.  Leaving another orphan in place would make the post-append
        # archive invalid and could hide a missed scheduled date.
        for sibling in captures_root.iterdir():
            if sibling.name.startswith(".") or sibling.name == capture_id or sibling.name in indexed_ids:
                continue
            if sibling.is_symlink() or not sibling.is_dir():
                raise ArchiveValidationError(f"Unexpected capture path: {sibling}")
            sibling_manifest = _validate_capture_directory(
                sibling,
                expected_protocol_sha256=frozen_hash,
            )
            if sibling_manifest.get("scheduled_date") == scheduled_date:
                raise CaptureCollisionError(
                    f"An unindexed capture already claims scheduled slot {scheduled_date}"
                )
            raise ArchiveValidationError(
                f"An unindexed capture must be recovered before appending another slot: {sibling.name}"
            )
        if os.path.lexists(destination):
            if destination.is_symlink():
                raise ArchiveValidationError(f"Existing capture path is unsafe: {destination}")
            manifest = _validate_capture_directory(
                destination,
                expected_protocol_sha256=frozen_hash,
                expected_capture_id=capture_id,
            )
            if manifest.get("scheduled_date") != scheduled_date:
                raise CaptureCollisionError("Existing unindexed capture claims a different scheduled slot")
            for field in _TIMING_FIELDS:
                if manifest.get(field) != expected_timing[field]:
                    raise CaptureCollisionError(
                        "Existing unindexed capture has different immutable timing evidence"
                    )
        else:
            temporary = Path(tempfile.mkdtemp(prefix=f".{capture_id}.", suffix=".staging", dir=captures_root))
            installed = False
            try:
                _materialize_capture(
                    temporary,
                    capture_id=capture_id,
                    timing=expected_timing,
                    models=models,
                    frozen_protocol_sha256=frozen_hash,
                    amendments=active_amendments,
                )
                manifest = _validate_capture_directory(
                    temporary,
                    expected_protocol_sha256=frozen_hash,
                    expected_capture_id=capture_id,
                )
                _fsync_directory(temporary)
                # ``destination`` was checked while holding the archive lock;
                # rename is atomic within the same filesystem and leaves no
                # partly populated directory visible to readers.
                os.rename(temporary, destination)
                _fsync_directory(captures_root)
                installed = True
            finally:
                if not installed and temporary.exists():
                    for path in sorted(temporary.rglob("*"), reverse=True):
                        if path.is_file() or path.is_symlink():
                            path.unlink()
                        elif path.is_dir():
                            path.rmdir()
                    temporary.rmdir()

        manifest_hash = sha256_file(destination / "manifest.json")
        previous_hash = captures[-1]["entry_sha256"] if captures else None
        row: dict[str, Any] = {
            "sequence": len(captures) + 1,
            "capture_id": capture_id,
            "scheduled_date": scheduled_date,
            "timing_status": manifest["timing_status"],
            "timing_eligible": manifest["timing_eligible"],
            "capture_status": manifest["capture_status"],
            "amendments": [dict(item) for item in manifest.get("amendments", [])],
            "path": f"captures/{capture_id}/manifest.json",
            "manifest_sha256": manifest_hash,
            "protocol_sha256": frozen_hash,
            "previous_entry_sha256": previous_hash,
        }
        row["entry_sha256"] = _entry_hash(row)
        new_index = dict(index)
        new_index["amendments"] = active_amendments
        new_index["captures"] = [*captures, row]
        _atomic_replace(archive_root / "index.json", canonical_json_bytes(new_index))
        validate_archive(archive_root)
        return destination, row


def validate_archive(
    root: Path | str = DEFAULT_ARCHIVE_ROOT,
    *,
    allow_unindexed_orphans: bool = False,
) -> dict[str, Any]:
    archive_root = _resolve_archive_root(root)
    index = _load_index(archive_root)
    frozen_hash = protocol_sha256(archive_root)
    active_amendments = _load_amendment_refs(archive_root, protocol_hash=frozen_hash)
    seen_ids: set[str] = set()
    seen_dates: set[str] = set()
    previous_hash: str | None = None
    indexed_dirs: set[str] = set()
    for expected_sequence, row in enumerate(index["captures"], start=1):
        if not isinstance(row, dict) or row.get("sequence") != expected_sequence:
            raise ArchiveValidationError("Capture index sequence is not contiguous")
        if row.get("entry_sha256") != _entry_hash(row):
            raise ArchiveValidationError("Capture index chain entry hash mismatch")
        if row.get("previous_entry_sha256") != previous_hash:
            raise ArchiveValidationError("Capture index previous-entry chain is broken")
        if row.get("protocol_sha256") != frozen_hash:
            raise ArchiveValidationError("Capture index row references another protocol")
        capture_id = str(row.get("capture_id", ""))
        scheduled_date = str(row.get("scheduled_date", ""))
        _validate_capture_id(capture_id)
        try:
            expected_capture_id = capture_id_for_date(scheduled_date)
        except (CaptureTimeError, TypeError, ValueError) as exc:
            raise ArchiveValidationError(f"Invalid scheduled date in capture index: {scheduled_date!r}") from exc
        if capture_id != expected_capture_id:
            raise ArchiveValidationError("Capture index id does not match its scheduled UTC cutoff")
        if capture_id in seen_ids or scheduled_date in seen_dates:
            raise ArchiveValidationError("Duplicate capture ID or scheduled slot")
        seen_ids.add(capture_id)
        seen_dates.add(scheduled_date)
        relative = _safe_relative_path(str(row.get("path", "")))
        if relative != PurePosixPath("captures") / capture_id / "manifest.json":
            raise ArchiveValidationError("Index path is not canonical")
        capture_dir = archive_root / "captures" / capture_id
        manifest = _validate_capture_directory(
            capture_dir,
            expected_manifest_sha256=str(row.get("manifest_sha256", "")),
            expected_protocol_sha256=frozen_hash,
        )
        for field in ("scheduled_date", "timing_status", "timing_eligible", "capture_status"):
            if manifest.get(field) != row.get(field):
                raise ArchiveValidationError(f"Index/manifest mismatch for {field}")
        if row.get("amendments", []) != manifest.get("amendments", []):
            raise ArchiveValidationError("Index/manifest amendment references differ")
        indexed_dirs.add(capture_id)
        previous_hash = row["entry_sha256"]
    captures_root = archive_root / "captures"
    actual_dirs = set()
    orphan_dirs: set[str] = set()
    if os.path.lexists(captures_root):
        if captures_root.is_symlink() or not captures_root.is_dir():
            raise ArchiveValidationError(f"Capture root is missing or unsafe: {captures_root}")
        hidden_paths = [path.name for path in captures_root.iterdir() if path.name.startswith(".")]
        if hidden_paths:
            raise ArchiveValidationError(f"Unfinished staging paths remain in captures root: {hidden_paths}")
        actual_dirs = {path.name for path in captures_root.iterdir() if path.is_dir()}
        unexpected_files = [path.name for path in captures_root.iterdir() if path.is_file() or path.is_symlink()]
        unexpected_paths = [path.name for path in captures_root.iterdir() if not path.is_dir() and not path.is_file()]
        if unexpected_files or unexpected_paths:
            unexpected = unexpected_files + unexpected_paths
            raise ArchiveValidationError(f"Unexpected paths in captures root: {unexpected}")
    if actual_dirs != indexed_dirs:
        orphan_dirs = actual_dirs - indexed_dirs
        missing_dirs = indexed_dirs - actual_dirs
        if not allow_unindexed_orphans or missing_dirs:
            raise ArchiveValidationError(
                f"Indexed capture directories differ from disk: indexed={sorted(indexed_dirs)}, actual={sorted(actual_dirs)}"
            )
        # A crash may leave a complete, validated directory before the index
        # replacement.  It is safe to report it to an append retry, but never
        # silently treat an arbitrary directory as evidence.
        for orphan_id in sorted(orphan_dirs):
            _validate_capture_directory(
                captures_root / orphan_id,
                expected_protocol_sha256=frozen_hash,
                expected_capture_id=orphan_id,
            )
    return {
        "status": "VALID",
        "protocol_sha256": frozen_hash,
        "active_amendments": active_amendments,
        "capture_count": len(index["captures"]),
        "unindexed_orphans": sorted(orphan_dirs),
        "last_entry_sha256": previous_hash,
    }
