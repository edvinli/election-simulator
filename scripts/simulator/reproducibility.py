"""Reproducibility manifest generation and input hashing for ElectionSimulator v1."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from .config import MODEL_VERSION


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
        return res.stdout.strip()
    except Exception:
        return "unknown_git_commit"


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
