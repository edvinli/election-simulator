"""A frozen site publication, produced by the repository's real exporter.

PR #18 froze the forecast history and the polling inputs the publication tests
read. What remained was the site: ``_production_fixture`` copied the committed
``files/election-simulator`` tree, so the prior publication the tests start
from was whatever production last published. Generation ids begin with a UTC
timestamp and sort lexicographically, so a synthetic generation the tests
publish can sort *behind* the live one -- which is why the
``guard/history-publication-agreement`` generation-agreement guard rejects a
fixture publication that is otherwise perfectly valid. Choosing a later
synthetic timestamp would only move the collision.

This module instead freezes the site side. ``regenerate()`` runs the real
:func:`scripts.static_exporter.export_static_data` once, validates the result
through the real published-directory validator, and stores the output under
``tests/fixtures/site_publication``. Nothing here recomputes a manifest hash,
a deterministic content hash or a pointer hash: every hash in the fixture was
written by the exporter that writes production's.

Regenerate the committed fixture with:

    python -m tests.site_publication_fixture

Not named ``test_*``, so unittest discovery does not collect it.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from scripts.simulator.engine import simulate_election
from scripts.static_exporter import export_static_data, validate_published_directory
from scripts.static_exporter.exporter import validate_publication_version

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "site_publication"

# Frozen a clear margin before FROZEN_AS_OF in tests/history_fixtures.py, so
# every generation the tests publish sorts after the one the site starts from.
# A generation id is "<UTC timestamp>-<payload hash prefix>", so the timestamp
# decides the ordering.
FROZEN_PUBLICATION_ELECTION_DATE = "2026-09-13"
FROZEN_PUBLICATION_SAMPLES = 8
# Deliberately synthetic: a real commit hash would tie the committed fixture to
# whichever checkout happened to generate it.
FROZEN_SOURCE_GIT_COMMIT = "f0" * 20

# Two generations, oldest first, because a real site carries a prior version
# and at least one test needs a genuinely older generation to point a stale
# pointer at. The active publication is the last entry.
FROZEN_GENERATIONS: tuple[dict[str, Any], ...] = (
    {"generated_at_utc": "2026-08-30T00:00:00+00:00", "as_of": "2026-08-29", "seed": 4242},
    {"generated_at_utc": "2026-09-01T00:00:00+00:00", "as_of": "2026-08-31", "seed": 12345},
)
FROZEN_PUBLICATION_GENERATED_AT = FROZEN_GENERATIONS[-1]["generated_at_utc"]


def frozen_result(
    spec: Mapping[str, Any] = FROZEN_GENERATIONS[-1],
    *,
    source_git_commit: str = FROZEN_SOURCE_GIT_COMMIT,
) -> Any:
    """A small certified result for one frozen publication generation."""

    result = simulate_election(
        as_of=str(spec["as_of"]),
        election_date=FROZEN_PUBLICATION_ELECTION_DATE,
        samples=FROZEN_PUBLICATION_SAMPLES,
        seed=int(spec["seed"]),
    )
    # The exporter requires a certified clean-source claim, and a working
    # checkout legitimately carries unrelated uncommitted files.
    result.manifest["source_worktree_clean"] = True
    result.manifest["source_git_commit"] = source_git_commit
    return result


def _export_all(output_root: Path, *, source_git_commit: str) -> str:
    """Publish every frozen generation in order with the real exporter."""

    generation = ""
    for spec in FROZEN_GENERATIONS:
        manifest = export_static_data(
            frozen_result(spec, source_git_commit=source_git_commit),
            output_dir=output_root,
            generated_at_utc=str(spec["generated_at_utc"]),
        )
        generation = str(manifest["publication_generation"])
    # The active version is the one current.json addresses.
    validate_published_directory(output_root)
    for version in (output_root / "versions").iterdir():
        validate_publication_version(version, expected_generation=version.name)
    return generation


def frozen_site_generation() -> str:
    """The committed fixture's generation id, read from its own pointer."""

    pointer = json.loads((FIXTURE_ROOT / "current.json").read_text(encoding="utf-8"))
    return str(pointer["publication_generation"])


def _install_publication(source_root: Path, publication_root: Path) -> str:
    """Replace a publication root's pointer and versions, keeping ``history/``."""

    publication_root.mkdir(parents=True, exist_ok=True)
    versions = publication_root / "versions"
    if versions.exists():
        shutil.rmtree(versions)
    shutil.copytree(source_root / "versions", versions)
    shutil.copyfile(source_root / "current.json", publication_root / "current.json")
    return str(json.loads((publication_root / "current.json").read_text(encoding="utf-8"))[
        "publication_generation"
    ])


def install_frozen_site_publication(publication_root: Path | str) -> str:
    """Copy the committed frozen publication into a fixture repository."""

    return _install_publication(FIXTURE_ROOT, Path(publication_root))


def export_frozen_site_publication(
    publication_root: Path | str,
    *,
    source_git_commit: str,
) -> str:
    """Re-publish the frozen generation certified from ``source_git_commit``.

    Two tests need the baseline publication to be certified from the fixture
    repository's own HEAD, which cannot be known until that repository has been
    committed. The previous helper rewrote the metadata and then recomputed the
    manifest hash, every deterministic content hash and the pointer hash by
    hand -- a second implementation of the exporter's hashing living in the
    test file. This runs the exporter instead, so those hashes have exactly one
    implementation.

    The generation id is unchanged: it is derived from the frozen timestamp and
    the result's deterministic payload hash, and the source commit is audit
    provenance that enters neither.
    """

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "election-simulator"
        _export_all(staged, source_git_commit=source_git_commit)
        return _install_publication(staged, Path(publication_root))


def regenerate(fixture_root: Path | str = FIXTURE_ROOT) -> str:
    """Rebuild the committed fixture with the real exporter."""

    destination = Path(fixture_root)
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "election-simulator"
        # The fixture is only committed once the real validators accept it.
        generation = _export_all(staged, source_git_commit=FROZEN_SOURCE_GIT_COMMIT)
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staged, destination)
    validate_published_directory(destination)
    return generation


def main() -> int:
    generation = regenerate()
    print(f"frozen site publication regenerated: {generation}")
    print(f"  {FIXTURE_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
