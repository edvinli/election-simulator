"""Mirror one certified publication generation into the website repository.

This module is deliberately separate from ``scripts.publication_pipeline``.
It performs no simulation, no statistical work, and no version control
operation.  It copies an already-certified immutable generation across the
repository boundary and then, only after the copied destination has itself
passed full validation, writes the consumer pointer.

The mutation order is always:

1. validate the certified source generation;
2. refuse to touch an existing destination generation;
3. stage and atomically install the seven real destination files;
4. validate the destination generation independently;
5. write ``current.json`` last.

A failure at any step leaves the website's previous pointer and previous
versions exactly as they were.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import uuid

from scripts.forecast_history.contract import validate_history_contract
from scripts.simulator.reproducibility import GENERATION_ID_PATTERN, compute_file_sha256
from scripts.static_exporter.exporter import (
    PUBLICATION_FILES,
    SUPPORTED_PUBLICATION_SCHEMA_VERSIONS,
    validate_publication_version,
)


SITE_PUBLICATION_RELATIVE = Path("files") / "election-simulator"
SITE_HISTORY_RELATIVE = SITE_PUBLICATION_RELATIVE / "history" / "coalition-timeseries.json"

# The seven real files that constitute one immutable published generation.
GENERATION_FILES: tuple[str, ...] = (*PUBLICATION_FILES, "manifest.json")


class SitePublishError(RuntimeError):
    """Raised when a cross-repository publication cannot proceed safely."""


def _read_pointer(publication_dir: Path) -> dict[str, Any]:
    pointer_path = publication_dir / "current.json"
    if not pointer_path.is_file():
        raise SitePublishError(f"Source publication has no current.json pointer: {pointer_path}")
    with pointer_path.open(encoding="utf-8") as handle:
        pointer = json.load(handle)
    if not isinstance(pointer, dict):
        raise SitePublishError("Source publication pointer is not a JSON object")
    return pointer


def _resolve_generation(publication_dir: Path, generation: str | None) -> str:
    if generation is None:
        pointer = _read_pointer(publication_dir)
        generation = pointer.get("publication_generation")
    if not isinstance(generation, str) or not generation:
        raise SitePublishError("No publication generation was given and none could be resolved")
    if not GENERATION_ID_PATTERN.fullmatch(generation):
        raise SitePublishError(f"Publication generation is not web-safe: {generation!r}")
    return generation


def _write_json_atomic(path: Path, value: Any) -> None:
    """Write JSON durably, replacing the destination in one operation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if os.path.lexists(temporary):
            os.unlink(temporary)
        raise


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_generation_to_site(
    *,
    site_repo: Path | str,
    source_publication_dir: Path | str,
    generation: str | None = None,
    update_pointer: bool = True,
    allow_existing: bool = False,
) -> dict[str, Any]:
    """Copy one certified generation into a website repository.

    ``generation`` defaults to the generation the source pointer addresses.
    Nothing is ever committed or pushed; the caller reviews and commits the
    resulting working-tree change themselves.
    """

    source_root = Path(source_publication_dir).resolve()
    site_root = Path(site_repo).resolve()
    if not site_root.is_dir():
        raise SitePublishError(f"--site-repo must be an existing directory: {site_root}")
    if site_root == source_root or source_root.is_relative_to(site_root):
        raise SitePublishError(
            "Refusing to publish a repository into itself; --site-repo must be the website repository"
        )

    generation_name = _resolve_generation(source_root, generation)
    source_version = source_root / "versions" / generation_name
    if not source_version.is_dir():
        raise SitePublishError(f"Source generation does not exist: {source_version}")

    # The source must already be a fully certified, hash-consistent version.
    # This publisher never repairs, regenerates, or re-certifies a payload.
    source_manifest = validate_publication_version(source_version, expected_generation=generation_name)

    destination_publication = site_root / SITE_PUBLICATION_RELATIVE
    destination_versions = destination_publication / "versions"
    destination_version = destination_versions / generation_name
    existing_destination = os.path.lexists(destination_version)
    if existing_destination and not allow_existing:
        raise SitePublishError(
            f"Refusing to overwrite an existing published generation: {destination_version}"
        )

    if existing_destination:
        if not destination_version.is_dir():
            raise SitePublishError(f"Existing published generation is not a directory: {destination_version}")
        try:
            existing_manifest = validate_publication_version(
                destination_version,
                expected_generation=generation_name,
            )
        except (OSError, ValueError) as exc:
            raise SitePublishError(
                f"Existing published generation is not a valid immutable version: {destination_version}"
            ) from exc
        if existing_manifest.get("deterministic_content_sha256") != source_manifest.get(
            "deterministic_content_sha256"
        ):
            raise SitePublishError(f"Existing published generation differs: {destination_version}")
        for filename in GENERATION_FILES:
            if (destination_version / filename).read_bytes() != (source_version / filename).read_bytes():
                raise SitePublishError(f"Existing published generation differs: {filename}")
        pointer_status = "ALREADY_MIRRORED"
    else:
        destination_versions.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{generation_name}.staging-", dir=destination_versions))
        installed = False
        try:
            for filename in GENERATION_FILES:
                source_file = source_version / filename
                if source_file.is_symlink() or not source_file.is_file():
                    raise SitePublishError(f"Source generation file must be a real file: {source_file}")
                # copyfile follows the source and always creates a regular file,
                # so a published generation can never contain a symlink.
                shutil.copyfile(source_file, staging / filename)
                if compute_file_sha256(staging / filename) != compute_file_sha256(source_file):
                    raise SitePublishError(f"Copied publication file does not match its source: {filename}")
            extra = {path.name for path in staging.iterdir()} - set(GENERATION_FILES)
            if extra:
                raise SitePublishError(f"Staged generation contains unexpected files: {sorted(extra)}")

            os.replace(staging, destination_version)
            installed = True
            _fsync_directory(destination_versions)
            pointer_status = "MIRRORED"
        finally:
            if not installed and staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    # Validate what actually landed on the website side, independently of the
    # source.  Only a destination that passes on its own may be pointed at.
    destination_manifest = validate_publication_version(
        destination_version,
        expected_generation=generation_name,
        expected_manifest_sha256=compute_file_sha256(source_version / "manifest.json"),
    )
    if destination_manifest.get("deterministic_content_sha256") != source_manifest.get(
        "deterministic_content_sha256"
    ):
        raise SitePublishError("Copied generation does not match the certified source content hash")

    pointer_written = False
    if update_pointer:
        source_pointer = _read_pointer(source_root)
        pointer_payload = {
            "schema_version": source_pointer.get("schema_version"),
            "publication_state": "COMPLETE",
            "publication_generation": generation_name,
            "path": f"versions/{generation_name}",
            "manifest_sha256": compute_file_sha256(destination_version / "manifest.json"),
        }
        if pointer_payload["schema_version"] not in SUPPORTED_PUBLICATION_SCHEMA_VERSIONS:
            raise SitePublishError("Source pointer has an unsupported schema version")
        # current.json is the last write of a cross-repository publication.
        _write_json_atomic(destination_publication / "current.json", pointer_payload)
        _fsync_directory(destination_publication)
        pointer_written = True

    return {
        "status": pointer_status,
        "generation": generation_name,
        "source_version": str(source_version),
        "destination_version": str(destination_version),
        "files": list(GENERATION_FILES),
        "pointer_written": pointer_written,
        "pointer_path": str(destination_publication / "current.json") if pointer_written else None,
        "deterministic_content_sha256": destination_manifest.get("deterministic_content_sha256"),
        "deterministic_payload_sha256": destination_manifest.get("deterministic_payload_sha256"),
        "source_repository": destination_manifest.get("source_repository"),
        "source_git_commit": destination_manifest.get("source_git_commit"),
        "committed": False,
        "pushed": False,
    }


def sync_history_to_site(
    *,
    site_repo: Path | str,
    source_history_path: Path | str,
    destination_relative: Path | str = SITE_HISTORY_RELATIVE,
) -> dict[str, Any]:
    """Validate and atomically mirror the canonical history artifact.

    The publication generation and the history file are installed by separate
    helpers because the former has an immutable-version/pointer contract.  A
    history copy is still staged and validated before its destination is
    replaced, and equal bytes are left untouched so a no-op sync creates no
    website commit.
    """

    site_root = Path(site_repo).resolve()
    source = Path(source_history_path)
    destination = site_root / Path(destination_relative)
    if not site_root.is_dir():
        raise SitePublishError(f"--site-repo must be an existing directory: {site_root}")
    if not source.is_file() or source.is_symlink():
        raise SitePublishError(f"Source history must be a regular file: {source}")
    source = source.resolve()
    try:
        with source.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SitePublishError(f"Source history is not readable JSON: {source}") from exc
    try:
        validate_history_contract(payload)
    except ValueError as exc:
        raise SitePublishError(f"Source history failed contract validation: {exc}") from exc

    source_bytes = source.read_bytes()
    changed = not destination.is_file() or destination.read_bytes() != source_bytes
    if changed:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_bytes(source_bytes)
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        finally:
            if os.path.lexists(temporary):
                os.unlink(temporary)
    return {
        "status": "SYNCED" if changed else "UNCHANGED",
        "source": str(source),
        "destination": str(destination),
        "sha256": compute_file_sha256(destination),
        "changed": changed,
    }
