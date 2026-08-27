"""Reproducibility manifest generation and input hashing for ElectionSimulator v1."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from .config import MODEL_VERSION


# Generation ids are embedded in static publication URLs, so they are
# restricted to the character class the browser pointer validator accepts.
GENERATION_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
GENERATION_IDENTITY_PREFIX_LENGTH = 8


# Generated evidence is intentionally committed after the clean source commit
# and must not make later certification runs claim that the source itself was
# dirty.  All source, tests, configuration, and non-certification input data
# remain part of the cleanliness check.
GENERATED_OUTPUT_PREFIXES: tuple[str, ...] = (
    "data/processed/simulations/",
    "data/processed/seat_hindcasts/",
    "data/processed/prospective_forecasts/",
    "data/processed/botten_ada_benchmark/",
    # The stable consumer path contains the version store and may be a regular
    # directory or a legacy symlink on older checkouts.
    "files/election-simulator",
    # Retain compatibility with the short-lived sibling-store layout used by
    # earlier exporter builds; it is generated evidence, not source code.
    "files/.election-simulator.versions/",
)

# Repository that owns every artifact generated after the 2026-08-27 extraction.
SOURCE_REPOSITORY = "edvinli/election-simulator"

# Artifacts published before the extraction carry no ``source_repository`` key.
# They are still valid and are read as belonging to the original repository.
HISTORICAL_SOURCE_REPOSITORY = "edvinli/edvinli.github.io"

# Sentinel returned when ``git rev-parse`` cannot resolve a commit.  Certified
# publication and archive paths must treat this as a hard failure rather than
# writing it into provenance metadata.
UNRESOLVED_GIT_COMMIT = "unknown_git_commit"


def get_git_commit_hash(repo_dir: Path | str | None = None) -> str:
    """Retrieve current Git commit SHA-256 or HEAD hash."""
    r_dir = Path(repo_dir) if repo_dir else Path(__file__).resolve().parents[2]
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=r_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        commit = res.stdout.strip()
        return commit or UNRESOLVED_GIT_COMMIT
    except Exception:
        return UNRESOLVED_GIT_COMMIT


def resolve_source_repository(value: Any = None) -> str:
    """Read a ``source_repository`` field, defaulting historical artifacts.

    Artifacts published before the repository extraction have no
    ``source_repository``.  They remain valid and mean the original
    ``edvinli/edvinli.github.io`` repository; they are never rewritten.
    """

    if value is None or value == "":
        return HISTORICAL_SOURCE_REPOSITORY
    if not isinstance(value, str):
        raise ValueError("source_repository must be a string when present")
    return value


def require_resolvable_source_commit(manifest: Mapping[str, Any]) -> str:
    """Fail closed unless a manifest resolves to a real source Git commit.

    An unresolvable commit must never be written into an immutable artifact as
    a sentinel string, so this is a hard failure rather than a degraded field.
    """

    commit = manifest.get("source_git_commit", manifest.get("git_commit"))
    if not isinstance(commit, str) or not commit or commit == UNRESOLVED_GIT_COMMIT:
        raise ValueError(
            "Certified artifact requires a resolvable source Git commit; got "
            f"{commit!r}. Commit the repository before archiving or publishing."
        )
    return commit


def require_certified_source_provenance(manifest: Mapping[str, Any]) -> None:
    """Fail closed unless a manifest carries certifiable source provenance.

    Certification requires both a resolvable Git commit and a clean source
    worktree.
    """

    require_resolvable_source_commit(manifest)
    if manifest.get("source_worktree_clean") is not True:
        raise ValueError(
            "Certified publication metadata must record source_worktree_clean=true"
        )


def build_generation_id(generated_at_utc: str, identity: str) -> str:
    """Build the sortable ``YYYYMMDDTHHMMSSZ-<identity prefix>`` generation id.

    One canonical generation id is shared by the prospective archive snapshot
    and the static publication version directory it produces, so a published
    version can always be joined back to the archived forecast.  The format is
    lexicographically sortable and matches the ``[A-Za-z0-9_-]`` shape the
    browser publication pointer accepts.
    """

    parsed = datetime.fromisoformat(str(generated_at_utc).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("generated_at_utc must include a timezone")
    stamp = parsed.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = str(identity)[:GENERATION_IDENTITY_PREFIX_LENGTH]
    if not prefix or not re.fullmatch(r"[0-9a-zA-Z]+", prefix):
        raise ValueError(f"Generation identity prefix is not alphanumeric: {prefix!r}")
    generation = f"{stamp}-{prefix}"
    if not GENERATION_ID_PATTERN.fullmatch(generation):
        raise ValueError(f"Generated generation id is not web-safe: {generation!r}")
    return generation


def is_git_worktree_clean(repo_dir: Path | str | None = None) -> bool:
    """Return whether the source worktree was clean when an artifact was built."""
    r_dir = Path(repo_dir) if repo_dir else Path(__file__).resolve().parents[2]
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=r_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        dirty_source_paths: list[str] = []
        for raw_line in res.stdout.splitlines():
            # Porcelain v1 reports ordinary paths after the two status bytes;
            # rename entries contain an additional "old -> new" suffix and
            # are conservatively treated as source changes.
            path = raw_line[3:] if len(raw_line) >= 4 else raw_line
            if " -> " in path or not path:
                dirty_source_paths.append(path)
                continue
            is_generated_output = any(
                path == prefix.rstrip("/") or path.startswith(prefix.rstrip("/") + "/")
                for prefix in GENERATED_OUTPUT_PREFIXES
            )
            if not is_generated_output:
                dirty_source_paths.append(path)
        return not bool(dirty_source_paths)
    except Exception:
        return False


def compute_file_sha256(file_path: Path | str) -> str:
    """Compute SHA-256 checksum of a file on disk."""
    p = Path(file_path)
    if not p.exists():
        return "file_not_found"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dict_sha256(data_dict: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 checksum of a dictionary."""
    encoded = json.dumps(data_dict, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_reproducibility_manifest(
    as_of: str,
    election_date: str,
    samples: int,
    base_seed: int,
    poll_data_path: Path | str | None = None,
    election_data_path: Path | str | None = None,
    mandate_data_path: Path | str | None = None,
    geography_data_path: Path | str | None = None,
    model_config: dict[str, Any] | None = None,
    repo_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Generate canonical reproducibility manifest for a simulation execution."""
    root_dir = Path(repo_dir) if repo_dir else Path(__file__).resolve().parents[2]
    
    p_poll = Path(poll_data_path) if poll_data_path else root_dir / "data" / "processed" / "pollofpolls" / "swedishpolls_individual_polls.csv"
    p_elec = Path(election_data_path) if election_data_path else root_dir / "data" / "processed" / "elections" / "riksdag_election_results.csv"
    p_mand = Path(mandate_data_path) if mandate_data_path else root_dir / "data" / "processed" / "mandates" / "historical_certified_mandates.csv"
    p_geog = Path(geography_data_path) if geography_data_path else root_dir / "data" / "processed" / "geography" / "constituency_party_votes_2014_2022.csv"

    cfg = model_config or {
        "opinion_model": "OpinionState_v1.1",
        "dynamics_model": "symmetric_all_history",
        "noise_model": "pp_centered_noise",
        "geography_model": "GeographicProjection_v1",
        "mandate_model": "MandateAllocator_v1",
    }

    manifest = {
        "model_version": MODEL_VERSION,
        "as_of": str(as_of),
        "election_date": str(election_date),
        "samples": int(samples),
        "base_seed": int(base_seed),
        "poll_data_hash": compute_file_sha256(p_poll),
        "election_data_hash": compute_file_sha256(p_elec),
        "mandate_data_hash": compute_file_sha256(p_mand),
        "geography_data_hash": compute_file_sha256(p_geog),
        "model_config_hash": compute_dict_sha256(cfg),
        "model_config": cfg,
        # ``source_git_commit`` identifies the clean code/data commit used for
        # generation.  Artifacts may be committed afterward, so this must not
        # be interpreted as the commit containing the artifact itself.
        "source_git_commit": get_git_commit_hash(root_dir),
        "source_worktree_clean": is_git_worktree_clean(root_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    # Keep the original field for consumers of the v1 manifest schema.
    manifest["git_commit"] = manifest["source_git_commit"]
    return manifest


def compute_simulation_payload_sha256(
    national_matrix: Any,
    seats_matrix: Any,
    summary_dict: dict[str, Any],
) -> str:
    """Compute deterministic SHA-256 checksum over deterministic simulation payload."""
    import numpy as np
    h = hashlib.sha256()
    nat_arr = np.asarray(national_matrix, dtype=np.float64)
    seat_arr = np.asarray(seats_matrix, dtype=np.int64)
    # Include array metadata so concatenating different-shaped payloads cannot
    # accidentally produce the same byte stream.
    h.update(json.dumps({"dtype": str(nat_arr.dtype), "shape": nat_arr.shape}, separators=(",", ":")).encode("utf-8"))
    h.update(nat_arr.tobytes())
    h.update(json.dumps({"dtype": str(seat_arr.dtype), "shape": seat_arr.shape}, separators=(",", ":")).encode("utf-8"))
    h.update(seat_arr.tobytes())

    def _remove_run_specific(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                k: _remove_run_specific(v)
                for k, v in value.items()
                if k not in {
                    "generated_at",
                    "generated_at_utc",
                    "runtime_seconds",
                    "wall_clock_seconds",
                    # This is provenance metadata, not a simulation input;
                    # writing an artifact between two identical runs must not
                    # change the deterministic payload identity.
                    "source_worktree_clean",
                    "deterministic_payload_sha256",
                    "payload_sha256",
                }
            }
        if isinstance(value, list):
            return [_remove_run_specific(v) for v in value]
        return value

    clean_summary = _remove_run_specific(summary_dict)
    summary_bytes = json.dumps(clean_summary, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    h.update(summary_bytes)
    return h.hexdigest()
