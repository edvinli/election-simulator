"""Load an already-certified generation for rendering.

Publication is two-stage. Certification pushes the immutable generation; the
website's history curve is rendered afterwards. When rendering has to run
separately -- a retry, or the second workflow -- it must reproduce the exact
forecast that was certified, and must not produce a new one.

That rules out two tempting shortcuts. The authoritative simulator is never
invoked here: rerunning it would publish a different forecast than the one
certified, even with the same seed, because replay is not bit-stable across
runners (amendment 004 established that). And the joint distribution is never
approximated from the published marginal quantiles: a history point's coalition
intervals are computed from joint draws, so quantiles cannot stand in for them
without silently changing the model.

What remains is the exact-draw sidecar, retained for every certified generation
under amendment 007. This module loads it and refuses everything else.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from scripts.simulator.exact_draw_sidecar import (
    SIDECAR_DRAWS_FILENAME,
    SIDECAR_METADATA_FILENAME,
    ExactDrawSidecarError,
    load_verified_draw_sidecar,
)
from scripts.static_exporter import validate_publication_version

PUBLICATION_RELATIVE = "files/election-simulator"
ARCHIVE_RELATIVE = "data/processed/prospective_forecasts"


class CertifiedGenerationError(RuntimeError):
    """A certified generation cannot be loaded for rendering."""


@dataclass(frozen=True)
class CertifiedGeneration:
    """One certified generation, sufficient to render and nothing more.

    Deliberately duck-types the attributes the history boundary reads from a
    ``SimulationResult`` -- ``vote_shares_matrix``, ``seats_matrix`` and
    ``manifest`` -- because ``update_history_with_production_result`` accepts an
    already-computed result and never invokes the simulator. Passing this in
    keeps the certified current point exact while remaining, provably, not a
    simulation.
    """

    generation: str
    certification_commit: str | None
    vote_shares_matrix: np.ndarray
    seats_matrix: np.ndarray
    manifest: Mapping[str, Any]
    snapshot: Mapping[str, Any]
    publication_manifest: Mapping[str, Any]
    as_of: str
    source_git_commit: str

    @property
    def samples(self) -> int:
        return int(self.vote_shares_matrix.shape[0])


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise CertifiedGenerationError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CertifiedGenerationError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CertifiedGenerationError(f"{label} is not a JSON object: {path}")
    return value


def _git(repo: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False)


def _assert_in_certification_commit(
    repo: Path, *, commit: str, generation: str, require_draws: bool
) -> None:
    """Every artifact rendering depends on must be in the named commit.

    The commit is the caller's statement of *which* certification is being
    rendered. Checking the working tree instead would let a rendering retry
    silently pick up a different generation's files after a concurrent push.
    """

    resolved = _git(repo, ["rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}"])
    if resolved.returncode != 0:
        raise CertifiedGenerationError(
            f"certification commit {commit} does not exist in this checkout")
    required = [
        f"{PUBLICATION_RELATIVE}/versions/{generation}/manifest.json",
        f"{ARCHIVE_RELATIVE}/{generation}/snapshot.json",
    ]
    if require_draws:
        required.append(f"{ARCHIVE_RELATIVE}/{generation}/{SIDECAR_DRAWS_FILENAME}")
        required.append(f"{ARCHIVE_RELATIVE}/{generation}/{SIDECAR_METADATA_FILENAME}")
    for relative in required:
        present = _git(repo, ["cat-file", "-e", f"{commit}:{relative}"])
        if present.returncode != 0:
            raise CertifiedGenerationError(
                f"certification commit {commit[:12]} does not contain {relative}; "
                "it is not the commit that certified this generation"
            )


def load_certified_generation(
    repo_root: Path | str,
    *,
    generation: str,
    certification_commit: str | None = None,
) -> CertifiedGeneration:
    """Load and fully verify one certified generation for rendering.

    Raises :class:`CertifiedGenerationError` rather than falling back to
    anything. A generation certified before amendment 007 has no archived
    draws, and that is a clear refusal, not a cue to re-simulate.
    """

    repo = Path(repo_root)
    if Path(generation).name != generation or generation in {".", ".."}:
        raise CertifiedGenerationError(f"generation id is not a bare name: {generation!r}")

    version_dir = repo / PUBLICATION_RELATIVE / "versions" / generation
    if not version_dir.is_dir():
        raise CertifiedGenerationError(
            f"generation {generation} has no publication bundle at {version_dir}")
    try:
        validate_publication_version(version_dir, expected_generation=generation)
    except Exception as exc:  # the exporter raises its own error types
        raise CertifiedGenerationError(
            f"publication bundle for {generation} failed validation: {exc}") from exc
    publication_manifest = _read_json(
        version_dir / "manifest.json", label="publication manifest")

    archive_dir = repo / ARCHIVE_RELATIVE / generation
    snapshot = _read_json(archive_dir / "snapshot.json", label="archive snapshot")
    if snapshot.get("generation_id") != generation:
        raise CertifiedGenerationError(
            f"archive snapshot names generation {snapshot.get('generation_id')!r}, "
            f"not {generation!r}"
        )

    # The bundle and the archived snapshot must agree on the payload, or they
    # are not two views of one certification.
    payload_hash = publication_manifest.get("deterministic_payload_sha256")
    if not isinstance(payload_hash, str) or not payload_hash:
        raise CertifiedGenerationError(
            f"publication manifest for {generation} carries no deterministic_payload_sha256")
    if snapshot.get("deterministic_payload_sha256") != payload_hash:
        raise CertifiedGenerationError(
            f"generation {generation}: archive snapshot and publication manifest "
            "disagree on deterministic_payload_sha256"
        )

    draws_path = archive_dir / SIDECAR_DRAWS_FILENAME
    metadata_path = archive_dir / SIDECAR_METADATA_FILENAME
    if not draws_path.is_file() or not metadata_path.is_file():
        raise CertifiedGenerationError(
            f"generation {generation} has no archived exact-draw sidecar, so it "
            "cannot be rendered without re-running the authoritative forecast. "
            "Generations certified before amendment 007 are outside its "
            "prospective retention and are not renderable."
        )
    try:
        # Party ordering, the percentage-points vote unit and array dtypes are
        # all asserted by this loader against the sidecar's own metadata, so
        # they are not restated here.
        loaded = load_verified_draw_sidecar(
            draws_path,
            metadata_path,
            expected_generation_id=generation,
            expected_payload_hash=payload_hash,
            include_arrays=True,
        )
    except ExactDrawSidecarError as exc:
        raise CertifiedGenerationError(
            f"exact-draw sidecar for {generation} failed verification: {exc}") from exc

    votes = np.asarray(loaded["vote_shares_pct"])
    seats = np.asarray(loaded["seats"])
    if votes.shape[0] != seats.shape[0]:
        raise CertifiedGenerationError(
            f"generation {generation}: vote and seat draw counts differ")

    if certification_commit is not None:
        _assert_in_certification_commit(
            repo,
            commit=certification_commit,
            generation=generation,
            require_draws=True,
        )

    as_of = snapshot.get("as_of")
    if not isinstance(as_of, str) or not as_of:
        raise CertifiedGenerationError(
            f"archive snapshot for {generation} carries no as_of")
    source_commit = str(publication_manifest.get("source_git_commit") or "")
    if not source_commit:
        raise CertifiedGenerationError(
            f"publication manifest for {generation} carries no source_git_commit, so "
            "its model inputs cannot be pinned"
        )

    # The manifest handed on carries the certified provenance, so a history
    # point built from this reports the generation's own source revision rather
    # than whatever the renderer happens to be running.
    manifest = {
        "source_git_commit": source_commit,
        "git_commit": source_commit,
        "model_version": publication_manifest.get("model_version"),
        "deterministic_payload_sha256": payload_hash,
        "publication_generation": generation,
        "rendered_from": "archived_exact_draws",
    }
    return CertifiedGeneration(
        generation=generation,
        certification_commit=certification_commit,
        vote_shares_matrix=votes,
        seats_matrix=seats,
        manifest=manifest,
        snapshot=snapshot,
        publication_manifest=publication_manifest,
        as_of=as_of,
        source_git_commit=source_commit,
    )
