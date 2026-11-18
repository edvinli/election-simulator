"""Integrity guard shared by the calibration-2030 harnesses.

``docs/calibration_2030_preregistration.md`` (frozen, body SHA-256 7707c2af...)
§1.2 says a scoring run refuses to start unless two things hold:

* every pinned input matches its frozen hash;
* no tracked file has uncommitted changes.

A run also records the commit it ran at.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "data" / "processed" / "calibration_2030" / "preregistration.json"
PREREG = ROOT / "docs" / "calibration_2030_preregistration.md"
FROZEN_BODY_SHA256 = "7707c2afb754ff3a5d6ea8f85034321a6f2e0de54117012220c6da46fa66ee53"
MARKER = b"<!-- FREEZE-BLOCK-" + b"START -->"


class IntegrityError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def verify_and_describe(allow_dirty: bool = False) -> dict:
    """Refuse to run on unpinned inputs or a modified checkout; return the provenance record."""

    body = PREREG.read_bytes().rsplit(MARKER, 1)[0]
    if hashlib.sha256(body).hexdigest() != FROZEN_BODY_SHA256:
        raise IntegrityError("preregistration body does not match its frozen hash")
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    if record.get("status") != "frozen" or record.get("body_sha256") != FROZEN_BODY_SHA256:
        raise IntegrityError("preregistration record is not the frozen record")
    expected = dict(record["inputs_sha256"])
    expected["scripts/election_residuals/eve_consensus.py"] = record["corrected_residuals"]["module_sha256"]
    mismatched = {p: h for p, h in expected.items() if sha256(ROOT / p) != h}
    if mismatched:
        raise IntegrityError(f"pinned inputs changed: {sorted(mismatched)}")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if dirty and not allow_dirty:
        raise IntegrityError(f"tracked files have uncommitted changes:\n{dirty}")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "tracked_clean": not dirty,
        "preregistration_body_sha256": FROZEN_BODY_SHA256,
        "preregistration_record_sha256": sha256(RECORD),
        "inputs_sha256": expected,
    }
