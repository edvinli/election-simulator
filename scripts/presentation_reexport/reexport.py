"""Migrate one certified schema-1.2 publication to schema 1.3.

This is a representation-only operation.  It starts by validating the pinned
immutable source publication, then reads an explicitly supplied preserved
integer seat matrix and adds exact coalition histograms to cloned contracts.
It never simulates, reads polling inputs, appends to the prospective archive,
or rewrites an existing immutable generation.

The source forecast's deterministic payload identity is intentionally carried
forward unchanged.  The vote-share draw matrix was not preserved, so this
utility must not pretend to recompute the current payload-hash algorithm.
New prospective simulations continue to use that algorithm normally.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np

from scripts.simulator.config import DEFAULT_MAJORITY_THRESHOLD, PARLIAMENTARY_PARTIES_8
from scripts.simulator.reproducibility import (
    GENERATION_ID_PATTERN,
    SOURCE_REPOSITORY,
    build_generation_id,
    compute_file_sha256,
    get_git_commit_hash,
    is_git_worktree_clean,
)
#: This module is the documented one-off 1.2 -> 1.3 representation migration
#: (docs/static_publication.md). It must write 1.3 and NOT follow
#: PUBLICATION_SCHEMA_VERSION forward: it performs no simulation and has no run
#: manifest, so it cannot supply the ElectionNoise identity that schema 1.4
#: requires. Historical publications are never rewritten to add newer fields.
REEXPORT_TARGET_SCHEMA_VERSION = "1.3"

from scripts.static_exporter.exporter import (
    PUBLICATION_FILES,
    PUBLICATION_SCHEMA_VERSION,  # noqa: F401  (kept for the supported-version check)
    _canonical_bytes,
    _coalition_draws,
    _compact_integer_seat_histogram,
    _histogram_count,
    _histogram_quantile,
    _sha256_bytes,
    _strip_runtime_timestamps,
    _write_json,
    validate_publication_contract,
    validate_publication_version,
)


SOURCE_GENERATION = "20260828T064703Z-1da59168"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PAYLOAD_SHA256 = "1da59168a8e1d56e759b4631b5ff0e11e6d42392f0e9354ded8aa5320febd45d"
EXPECTED_MATRIX_SHA256 = "7d5626506bb7cad1bf54378bafd7aa51937037da5c45d22756632f681ed221cd"
EXPECTED_SAMPLES = 100_000
EXPECTED_PARTY_ORDER = tuple(PARLIAMENTARY_PARTIES_8)
DEFAULT_SOURCE_VERSION = REPOSITORY_ROOT / "files" / "election-simulator" / "versions" / SOURCE_GENERATION
DEFAULT_MATRIX_PATH = (
    Path.home()
    / "Documents"
    / "election-simulator-audit"
    / "20260828-coalition-covariance-audit"
    / "seats_matrix.npy"
)

SPOT_CHECKS: dict[int, dict[str, Any]] = {
    84: {
        "parties": ["C", "S", "MP"],
        "majority_count": 10_778,
        "prob_majority": 0.10778,
        "min_seats": 141,
        "max_seats": 190,
    },
    112: {
        "parties": ["S", "V", "MP"],
        "majority_count": 2_216,
        "prob_majority": 0.02216,
        "min_seats": 144,
        "max_seats": 188,
    },
    139: {
        "parties": ["M", "L", "KD", "SD"],
        "majority_count": 286,
        "prob_majority": 0.00286,
        "min_seats": 131,
        "max_seats": 179,
    },
}


def _read_contracts(version_dir: Path) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    for filename in PUBLICATION_FILES:
        with (version_dir / filename).open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"Publication contract is not a JSON object: {version_dir / filename}")
        contracts[filename] = value
    return contracts


def _load_preserved_matrix(path: Path) -> np.ndarray:
    actual_sha256 = compute_file_sha256(path)
    if actual_sha256 != EXPECTED_MATRIX_SHA256:
        raise ValueError(
            "Preserved seat matrix SHA-256 mismatch: "
            f"expected {EXPECTED_MATRIX_SHA256}, got {actual_sha256}"
        )
    try:
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:  # pragma: no cover - NumPy supplies the detail
        raise ValueError(f"Unable to load preserved seat matrix: {path}") from exc
    if matrix.shape != (EXPECTED_SAMPLES, len(EXPECTED_PARTY_ORDER)):
        raise ValueError(
            "Preserved seat matrix must have shape "
            f"{(EXPECTED_SAMPLES, len(EXPECTED_PARTY_ORDER))}, got {matrix.shape}"
        )
    if not np.issubdtype(matrix.dtype, np.integer):
        raise ValueError(f"Preserved seat matrix must have an integer dtype, got {matrix.dtype}")
    if np.any(matrix < 0):
        raise ValueError("Preserved seat matrix contains negative seats")
    row_totals = np.sum(matrix, axis=1, dtype=np.int64)
    if not np.all(row_totals == 349):
        raise ValueError("Preserved seat matrix rows must all sum to 349")
    return matrix


def _assert_summary_matches_histogram(
    entry: Mapping[str, Any], histogram: Mapping[str, Any], *, samples: int, key: str
) -> None:
    minimum = int(histogram["min_seats"])
    counts = histogram["counts"]
    mean = sum((minimum + offset) * count for offset, count in enumerate(counts)) / samples
    if not np.isclose(mean, float(entry["mean_seats"]), rtol=0.0, atol=1e-12):
        raise ValueError(f"coalition {key} mean_seats disagrees with preserved histogram")
    for field, quantile in (
        ("p05_seats", 0.05),
        ("p10_seats", 0.10),
        ("p25_seats", 0.25),
        ("median_seats", 0.50),
        ("p75_seats", 0.75),
        ("p90_seats", 0.90),
        ("p95_seats", 0.95),
    ):
        if _histogram_quantile(minimum, counts, quantile) != entry[field]:
            raise ValueError(f"coalition {key} {field} disagrees with preserved histogram")
    majority_count = sum(
        count
        for offset, count in enumerate(counts)
        if minimum + offset >= DEFAULT_MAJORITY_THRESHOLD
    )
    probability = majority_count / samples
    if not np.isclose(probability, float(entry["prob_majority"]), rtol=0.0, atol=1e-12):
        raise ValueError(f"coalition {key} prob_majority disagrees with preserved histogram")


def _audit_histograms(
    coalitions: Mapping[str, Mapping[str, Any]], *, samples: int
) -> dict[str, Any]:
    if list(coalitions) != [str(mask) for mask in range(256)]:
        raise ValueError("Schema 1.3 coalition lookup must contain masks 0 through 255 in order")

    histograms: dict[int, Mapping[str, Any]] = {}
    for mask in range(256):
        key = str(mask)
        entry = coalitions[key]
        histogram = entry.get("seat_histogram")
        if not isinstance(histogram, Mapping):
            raise ValueError(f"coalition {key} is missing seat_histogram")
        minimum = histogram.get("min_seats")
        counts = histogram.get("counts")
        if (
            not isinstance(minimum, int)
            or isinstance(minimum, bool)
            or not 0 <= minimum <= 349
            or not isinstance(counts, list)
            or not counts
        ):
            raise ValueError(f"coalition {key} has an invalid seat_histogram")
        if any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts):
            raise ValueError(f"coalition {key} seat_histogram counts must be non-negative integers")
        maximum = minimum + len(counts) - 1
        if maximum > 349 or counts[0] <= 0 or counts[-1] <= 0:
            raise ValueError(f"coalition {key} seat_histogram support is invalid")
        if sum(counts) != samples:
            raise ValueError(f"coalition {key} seat_histogram count total is not {samples}")
        _assert_summary_matches_histogram(entry, histogram, samples=samples, key=key)
        histograms[mask] = histogram

    full_mask = 255
    complement_comparisons = 0
    for mask in range(256):
        complement = full_mask ^ mask
        for seats in range(350):
            if _histogram_count(histograms[mask], seats) != _histogram_count(
                histograms[complement], 349 - seats
            ):
                raise ValueError(
                    f"coalition {mask} and complement {complement} violate the 349-seat identity"
                )
        complement_comparisons += 1

    spot_results: dict[str, dict[str, Any]] = {}
    for mask, expected in SPOT_CHECKS.items():
        histogram = histograms[mask]
        minimum = int(histogram["min_seats"])
        counts = histogram["counts"]
        maximum = minimum + len(counts) - 1
        majority_count = sum(
            count
            for offset, count in enumerate(counts)
            if minimum + offset >= DEFAULT_MAJORITY_THRESHOLD
        )
        actual = {
            "parties": list(expected["parties"]),
            "majority_count": majority_count,
            "prob_majority": majority_count / samples,
            "min_seats": minimum,
            "max_seats": maximum,
        }
        if actual != expected:
            raise ValueError(f"coalition {mask} spot check mismatch: expected {expected}, got {actual}")
        spot_results[str(mask)] = actual

    return {
        "status": "PASS",
        "masks_checked": 256,
        "complement_audit": f"{complement_comparisons}/256 PASS",
        "spot_checks": spot_results,
    }


def _flatten_leaves(value: Any, path: tuple[str, ...]) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _flatten_leaves(child, (*path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _flatten_leaves(child, (*path, str(index)))
    else:
        yield path, value


def _leaf_differences(before: Any, after: Any, path: tuple[str, ...] = ()) -> Iterable[dict[str, Any]]:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        keys = list(before)
        keys.extend(key for key in after if key not in before)
        for key in keys:
            child_path = (*path, str(key))
            if key not in before:
                for leaf_path, value in _flatten_leaves(after[key], child_path):
                    yield {"path": "/".join(leaf_path), "kind": "added", "after": value}
            elif key not in after:
                for leaf_path, value in _flatten_leaves(before[key], child_path):
                    yield {"path": "/".join(leaf_path), "kind": "removed", "before": value}
            else:
                yield from _leaf_differences(before[key], after[key], child_path)
    elif isinstance(before, list) and isinstance(after, list):
        for index in range(max(len(before), len(after))):
            child_path = (*path, str(index))
            if index >= len(before):
                for leaf_path, value in _flatten_leaves(after[index], child_path):
                    yield {"path": "/".join(leaf_path), "kind": "added", "after": value}
            elif index >= len(after):
                for leaf_path, value in _flatten_leaves(before[index], child_path):
                    yield {"path": "/".join(leaf_path), "kind": "removed", "before": value}
            else:
                yield from _leaf_differences(before[index], after[index], child_path)
    elif before != after:
        yield {"path": "/".join(path), "kind": "changed", "before": before, "after": after}


def _change_category(path: str) -> str:
    parts = path.split("/")
    filename = parts[0] if parts else ""
    if parts and parts[-1] == "schema_version":
        return "schema_version"
    if (
        filename == "groups.json"
        and len(parts) >= 6
        and parts[1] == "coalition_builder"
        and parts[2] == "coalitions"
        and parts[4] == "seat_histogram"
    ):
        return "coalition_seat_histogram"
    if filename == "metadata.json" and parts[-1] == "generated_at_utc":
        return "publication_timestamp"
    if filename == "metadata.json" and parts[-1] == "source_git_commit":
        return "reexport_source_commit"
    if filename == "manifest.json":
        if parts[1:2] in (["publication_files"], ["deterministic_content_hashes"]):
            return "manifest_file_hashes"
        if parts[-1] in {
            "publication_generation",
            "generated_at_utc",
            "deterministic_content_sha256",
        }:
            return "manifest_publication_identity"
        if parts[-1] == "source_git_commit":
            return "reexport_source_commit"
    if filename == "current.json":
        return "publication_pointer"
    return "scientific_or_unapproved"


def compare_source_and_reexport(
    source_contracts: Mapping[str, Any],
    reexport_contracts: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    reexport_manifest: Mapping[str, Any],
    *,
    source_pointer: Mapping[str, Any] | None = None,
    reexport_pointer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    differences: list[dict[str, Any]] = []
    for filename in PUBLICATION_FILES:
        differences.extend(
            {
                **difference,
                "category": _change_category(f"{filename}/{difference['path']}"),
                "path": f"{filename}/{difference['path']}",
            }
            for difference in _leaf_differences(source_contracts[filename], reexport_contracts[filename])
        )
    differences.extend(
        {
            **difference,
            "category": _change_category(f"manifest.json/{difference['path']}"),
            "path": f"manifest.json/{difference['path']}",
        }
        for difference in _leaf_differences(source_manifest, reexport_manifest)
    )
    if source_pointer is not None and reexport_pointer is not None:
        differences.extend(
            {
                **difference,
                "category": _change_category(f"current.json/{difference['path']}"),
                "path": f"current.json/{difference['path']}",
            }
            for difference in _leaf_differences(source_pointer, reexport_pointer)
        )

    category_counts = Counter(difference["category"] for difference in differences)
    unapproved = [
        difference
        for difference in differences
        if difference["category"] == "scientific_or_unapproved"
    ]
    if unapproved:
        raise ValueError(
            "Representation-only migration changed scientific or unapproved leaves: "
            + ", ".join(difference["path"] for difference in unapproved[:10])
        )
    return {
        "status": "PASS",
        "changed_leaf_categories": dict(sorted(category_counts.items())),
        "changed_leaf_count": len(differences),
        "scientific_changes": [],
        "payload_sha256_unchanged": (
            source_contracts["forecast.json"].get("deterministic_payload_sha256")
            == reexport_contracts["forecast.json"].get("deterministic_payload_sha256")
            == EXPECTED_PAYLOAD_SHA256
        ),
    }


def _read_pointer(publication_dir: Path) -> dict[str, Any] | None:
    path = publication_dir / "current.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        pointer = json.load(handle)
    if not isinstance(pointer, dict):
        raise ValueError("Publication pointer is not a JSON object")
    return pointer


def _write_pointer(publication_dir: Path, *, generation: str, manifest_sha256: str) -> dict[str, Any]:
    pointer = {
        "schema_version": REEXPORT_TARGET_SCHEMA_VERSION,
        "publication_state": "COMPLETE",
        "publication_generation": generation,
        "path": f"versions/{generation}",
        "manifest_sha256": manifest_sha256,
    }
    temporary = publication_dir / f".current-{uuid.uuid4().hex}.tmp"
    _write_json(temporary, pointer)
    try:
        os.replace(temporary, publication_dir / "current.json")
    except Exception:
        if os.path.lexists(temporary):
            os.unlink(temporary)
        raise
    return pointer


def _build_manifest(
    staged_version: Path,
    contracts: Mapping[str, Mapping[str, Any]],
    *,
    generation: str,
    generated_at_utc: str,
    source_git_commit: str,
) -> dict[str, Any]:
    file_hashes = {
        filename: compute_file_sha256(staged_version / filename) for filename in PUBLICATION_FILES
    }
    deterministic_file_hashes = {
        filename: _sha256_bytes(
            _canonical_bytes(_strip_runtime_timestamps(contracts[filename]))
        )
        for filename in PUBLICATION_FILES
    }
    return {
        "schema_version": REEXPORT_TARGET_SCHEMA_VERSION,
        "publication_state": "COMPLETE",
        "publication_generation": generation,
        "generated_at_utc": generated_at_utc,
        "publication_files": file_hashes,
        "deterministic_content_hashes": deterministic_file_hashes,
        "deterministic_content_sha256": _sha256_bytes(_canonical_bytes(deterministic_file_hashes)),
        "model_version": contracts["metadata.json"]["model"]["version"],
        "source_repository": SOURCE_REPOSITORY,
        "source_git_commit": source_git_commit,
        "source_worktree_clean": True,
        "deterministic_payload_sha256": EXPECTED_PAYLOAD_SHA256,
    }


def migrate_publication(
    *,
    source_version: Path | str = DEFAULT_SOURCE_VERSION,
    matrix_path: Path | str = DEFAULT_MATRIX_PATH,
    output_dir: Path | str = REPOSITORY_ROOT / "files" / "election-simulator",
    generated_at_utc: str | None = None,
    generation_id: str | None = None,
) -> dict[str, Any]:
    """Create and activate one new schema-1.3 presentation generation.

    The source version is validated before the matrix is opened or any output
    directory is created.  The worktree must be clean at that point; the
    resulting metadata and manifest therefore identify the committed utility
    code (Commit A), while the generated files themselves are intended for a
    later artifact-only Commit B.
    """

    source_version_path = Path(source_version).resolve()
    # This call is intentionally the first validation gate: no matrix load,
    # worktree check, or output mutation may happen before the source is known
    # to be a certified immutable publication.
    source_manifest = validate_publication_version(
        source_version_path,
        expected_generation=SOURCE_GENERATION,
    )
    if source_manifest.get("schema_version") != "1.2":
        raise ValueError("The pinned source generation must be schema 1.2")
    source_contracts = _read_contracts(source_version_path)
    if any(contract.get("schema_version") != "1.2" for contract in source_contracts.values()):
        raise ValueError("Every pinned source contract must be schema 1.2")
    if source_manifest.get("deterministic_payload_sha256") != EXPECTED_PAYLOAD_SHA256:
        raise ValueError("Pinned source generation has an unexpected deterministic payload hash")

    source_git_commit = get_git_commit_hash(REPOSITORY_ROOT)
    if not source_git_commit or source_git_commit == "unknown_git_commit":
        raise ValueError("A committed re-export source Git commit is required")
    if not is_git_worktree_clean(REPOSITORY_ROOT):
        raise ValueError("Presentation re-export requires a clean source worktree")

    matrix = _load_preserved_matrix(Path(matrix_path).expanduser().resolve())
    generated = generated_at_utc or datetime.now(timezone.utc).isoformat()
    parsed = datetime.fromisoformat(str(generated).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("generated_at_utc must include a timezone")
    generation = generation_id or build_generation_id(generated, EXPECTED_PAYLOAD_SHA256)
    if not GENERATION_ID_PATTERN.fullmatch(generation):
        raise ValueError(f"Publication generation id is not web-safe: {generation!r}")
    if generation == SOURCE_GENERATION:
        raise ValueError("Presentation re-export must create a new immutable generation")

    contracts = deepcopy(source_contracts)
    for contract in contracts.values():
        contract["schema_version"] = REEXPORT_TARGET_SCHEMA_VERSION
    contracts["metadata.json"]["generated_at_utc"] = generated
    contracts["metadata.json"]["source_git_commit"] = source_git_commit

    builder = contracts["groups.json"].get("coalition_builder")
    if not isinstance(builder, Mapping):
        raise ValueError("Pinned source groups.json has no coalition_builder")
    coalitions = builder.get("coalitions")
    if not isinstance(coalitions, Mapping):
        raise ValueError("Pinned source groups.json has no coalition lookup")
    for mask in range(256):
        key = str(mask)
        entry = coalitions.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"Pinned source groups.json has no coalition {key}")
        if "seat_histogram" in entry:
            raise ValueError(f"Pinned schema-1.2 coalition {key} unexpectedly has a histogram")
        draws = _coalition_draws(matrix, EXPECTED_PARTY_ORDER, mask)
        entry["seat_histogram"] = _compact_integer_seat_histogram(draws)

    # This production validator checks every cloned scientific field and every
    # histogram summary against the production percentile/probability rules.
    validate_publication_contract(contracts)
    audit = _audit_histograms(coalitions, samples=EXPECTED_SAMPLES)

    output_root = Path(output_dir).expanduser().resolve()
    versions_root = output_root / "versions"
    destination = versions_root / generation
    if os.path.lexists(destination):
        raise FileExistsError(f"Immutable publication version already exists: {destination}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(tempfile.mkdtemp(prefix=f".{generation}.staging-", dir=output_root.parent))
    staging_version = staging_parent / generation
    staging_version.mkdir()
    installed = False
    try:
        for filename in PUBLICATION_FILES:
            _write_json(staging_version / filename, contracts[filename])
        manifest = _build_manifest(
            staging_version,
            contracts,
            generation=generation,
            generated_at_utc=generated,
            source_git_commit=source_git_commit,
        )
        _write_json(staging_version / "manifest.json", manifest)
        staged_manifest_sha256 = compute_file_sha256(staging_version / "manifest.json")
        validate_publication_version(
            staging_version,
            expected_generation=generation,
            expected_manifest_sha256=staged_manifest_sha256,
        )

        versions_root.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(destination):
            raise FileExistsError(f"Immutable publication version already exists: {destination}")
        os.replace(staging_version, destination)
        installed = True
        destination_manifest = validate_publication_version(
            destination,
            expected_generation=generation,
            expected_manifest_sha256=staged_manifest_sha256,
        )

        source_pointer = _read_pointer(output_root)
        new_pointer = {
            "schema_version": REEXPORT_TARGET_SCHEMA_VERSION,
            "publication_state": "COMPLETE",
            "publication_generation": generation,
            "path": f"versions/{generation}",
            "manifest_sha256": staged_manifest_sha256,
        }
        comparison = compare_source_and_reexport(
            source_contracts,
            contracts,
            source_manifest,
            destination_manifest,
            source_pointer=source_pointer,
            reexport_pointer=new_pointer if source_pointer is not None else None,
        )
        if not comparison["payload_sha256_unchanged"]:
            raise ValueError("Representation-only re-export changed the certified payload identity")
        _write_pointer(output_root, generation=generation, manifest_sha256=staged_manifest_sha256)
        return {
            "status": "PUBLISHED",
            "source_generation": SOURCE_GENERATION,
            "generation": generation,
            "manifest_sha256": staged_manifest_sha256,
            "deterministic_content_sha256": destination_manifest["deterministic_content_sha256"],
            "deterministic_payload_sha256": destination_manifest["deterministic_payload_sha256"],
            "source_git_commit": source_git_commit,
            "matrix_sha256": EXPECTED_MATRIX_SHA256,
            "all_256_audit": audit,
            "recursive_comparison": comparison,
            "files": [*PUBLICATION_FILES, "manifest.json"],
            "pointer_path": str(output_root / "current.json"),
        }
    finally:
        if not installed and staging_parent.exists():
            shutil.rmtree(staging_parent, ignore_errors=True)
        elif staging_parent.exists():
            shutil.rmtree(staging_parent, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Migrate the pinned certified schema-1.2 publication using a preserved "
            "seat matrix; never simulates or mutates the prospective archive."
        )
    )
    parser.add_argument("--source-version", type=Path, default=DEFAULT_SOURCE_VERSION)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "files" / "election-simulator")
    parser.add_argument("--generated-at-utc", default=None)
    parser.add_argument("--generation", dest="generation_id", default=None)
    args = parser.parse_args(argv)
    report = migrate_publication(
        source_version=args.source_version,
        matrix_path=args.matrix,
        output_dir=args.output_dir,
        generated_at_utc=args.generated_at_utc,
        generation_id=args.generation_id,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Changed leaf categories:")
    for category, count in report["recursive_comparison"]["changed_leaf_categories"].items():
        print(f"  {category}: {count}")
    print("Scientific changes: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
