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


FINAL_ROUND_DOC = ROOT / "docs" / "calibration_2030_final_round_preregistration.md"
FINAL_ROUND_RECORD = ROOT / "data" / "processed" / "calibration_2030" / "final_round.json"
FINAL_ROUND_BODY_SHA256 = "5d070b4b21c823b9db40253efd46de5288ec6d8259f0a1c7a34c8f2f9485c18b"
T5_DOC = ROOT / "docs" / "calibration_2030_t5_shadow_preregistration.md"
T5_RECORD = ROOT / "data" / "processed" / "calibration_2030" / "t5_shadow.json"
T5_BODY_SHA256 = "bf93c12843f68c0ab24beead41356fa30a214c5c56b02aa7ba79ba38ef6a97a3"


def _frozen_body_ok(doc: Path, expected: str) -> bool:
    return hashlib.sha256(doc.read_bytes().rsplit(MARKER, 1)[0]).hexdigest() == expected


def verify_final_round(allow_dirty: bool = False) -> dict:
    """The parent guard plus the frozen final-round preregistration and every input it pins (§1.2)."""

    provenance = verify_and_describe(allow_dirty=allow_dirty)
    if not _frozen_body_ok(FINAL_ROUND_DOC, FINAL_ROUND_BODY_SHA256):
        raise IntegrityError("final-round preregistration body does not match its frozen hash")
    record = json.loads(FINAL_ROUND_RECORD.read_text(encoding="utf-8"))
    if record.get("status") != "frozen" or record.get("body_sha256") != FINAL_ROUND_BODY_SHA256:
        raise IntegrityError("final-round record is not the frozen record")
    mismatched = sorted(p for p, h in record["inputs_sha256"].items() if sha256(ROOT / p) != h)
    if mismatched:
        raise IntegrityError(f"final-round pinned inputs changed: {mismatched}")
    provenance.update({
        "final_round_body_sha256": FINAL_ROUND_BODY_SHA256,
        "final_round_record_sha256": sha256(FINAL_ROUND_RECORD),
        "final_round_inputs_sha256": record["inputs_sha256"],
    })
    return provenance
