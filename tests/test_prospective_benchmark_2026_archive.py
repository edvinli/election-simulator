"""Archive and wall-clock contract tests for the 2026 prospective benchmark."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.prospective_benchmark_2026.archive import (
    ArchiveValidationError,
    CaptureCollisionError,
    ModelCapture,
    append_capture,
    canonical_json_bytes,
    validate_archive,
)
from scripts.prospective_benchmark_2026.time_rules import (
    CaptureTimeError,
    STOCKHOLM,
    capture_id_for_date,
    classify_capture_time,
    scheduled_cutoff,
)


class TestCaptureTimeRules(unittest.TestCase):
    def test_stockholm_cutoffs_and_window_guards(self) -> None:
        cutoff = scheduled_cutoff("2026-09-04")
        self.assertEqual(cutoff.tzinfo, STOCKHOLM)
        self.assertEqual(cutoff.isoformat(), "2026-09-04T23:30:00+02:00")
        self.assertEqual(cutoff.astimezone().date(), datetime.fromisoformat(cutoff.isoformat()).astimezone().date())
        self.assertEqual(capture_id_for_date("2026-09-12"), "20260912T213000Z")
        with self.assertRaises(CaptureTimeError):
            scheduled_cutoff("2026-09-03")
        with self.assertRaises(CaptureTimeError):
            scheduled_cutoff("2026-09-13")

    def test_early_real_capture_is_prohibited(self) -> None:
        with self.assertRaises(CaptureTimeError):
            classify_capture_time("2026-09-04", "2026-09-04T21:29:59Z", durable=True)
        dry = classify_capture_time("2026-09-04", "2026-09-04T20:00:00Z", durable=False)
        self.assertEqual(dry.status, "DRY_RUN_BEFORE_CUTOFF")
        self.assertFalse(dry.eligible)

    def test_on_time_and_late_next_stockholm_day(self) -> None:
        on_time = classify_capture_time("2026-09-04", "2026-09-04T21:30:00Z", durable=True)
        self.assertEqual(on_time.status, "ON_TIME_ELIGIBLE")
        self.assertTrue(on_time.eligible)
        late = classify_capture_time("2026-09-04", "2026-09-04T22:00:00Z", durable=True)
        self.assertEqual(late.retrieved_at_local.date().isoformat(), "2026-09-05")
        self.assertEqual(late.status, "LATE_EXCLUDED")
        self.assertFalse(late.eligible)

    def test_later_capture_is_retroactive_and_never_durable(self) -> None:
        with self.assertRaises(CaptureTimeError):
            classify_capture_time("2026-09-04", "2026-09-06T00:00:00Z", durable=True)
        dry = classify_capture_time("2026-09-04", "2026-09-06T00:00:00Z", durable=False)
        self.assertEqual(dry.status, "RETROACTIVE_PROHIBITED")
        self.assertFalse(dry.eligible)


class TestProspectiveBenchmarkArchive(unittest.TestCase):
    def _root(self, base: Path) -> Path:
        root = base / "benchmark"
        root.mkdir()
        protocol = b'{"frozen":true}\n'
        digest = hashlib.sha256(protocol).hexdigest()
        (root / "protocol.json").write_bytes(protocol)
        (root / "protocol.sha256").write_text(f"{digest}  protocol.json\n", encoding="utf-8")
        (root / "index.json").write_text(
            json.dumps({
                "schema_version": "1.0",
                "protocol_path": "protocol.json",
                "protocol_sha256": digest,
                "captures": [],
            }) + "\n",
            encoding="utf-8",
        )
        return root

    def _models(self, *, marker: str = "today") -> dict[str, ModelCapture]:
        return {
            "election_simulator": ModelCapture(
                status="AVAILABLE",
                forecast={"published_central_prediction": {"M": 20.0}, "marker": marker},
                provenance={"source": "certified-generation"},
                files={"draws.npz": b"exact-es-draw-bytes"},
            ),
            "botten_ada": ModelCapture(
                status="PARITY_UNVERIFIED",
                forecast={"published_central_prediction": {"M": 19.8}, "draws": None, "marker": marker},
                provenance={"source": "official-machine-readable"},
                files={"raw/timeseries.csv": b"date,party,p50\n2026-09-13,M,19.8\n"},
            ),
        }

    def _timing(self, day: str) -> dict[str, object]:
        return classify_capture_time(day, f"{day}T21:31:00Z", durable=True).to_dict()

    def _add_amendment(self, root: Path, *, number: int = 1) -> dict[str, object]:
        payload = {
            "schema_version": "1.0",
            "amendment_number": number,
            "amendment_id": f"{number:03d}-fixture",
            "created_at_utc": "2026-09-03T20:00:00Z",
            "original_protocol_sha256": hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest(),
            "reason": "Fixture amendment for archive validation.",
            "primary_scoring_effect": "NONE",
            "immutable": True,
        }
        directory = root / "amendments"
        directory.mkdir()
        path = directory / f"{number:03d}-fixture.json"
        content = json.dumps(payload, indent=2) .encode("utf-8") + b"\n"
        path.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        (directory / f"{number:03d}-fixture.sha256").write_text(
            f"{digest}  {path.name}\n", encoding="utf-8"
        )
        index = json.loads((root / "index.json").read_text(encoding="utf-8"))
        index["amendments"] = [{
            "amendment_number": number,
            "path": f"amendments/{path.name}",
            "sha256": digest,
            "primary_scoring_effect": "NONE",
        }]
        (root / "index.json").write_bytes(canonical_json_bytes(index))
        return index["amendments"][0]

    def test_append_is_immutable_and_index_hashes_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(Path(tmp))
            destination, row = append_capture(
                root=root,
                capture_id=capture_id_for_date("2026-09-04"),
                timing=self._timing("2026-09-04"),
                models=self._models(),
            )
            old_hash = hashlib.sha256((destination / "manifest.json").read_bytes()).hexdigest()
            self.assertEqual(row["manifest_sha256"], old_hash)
            self.assertEqual(validate_archive(root)["capture_count"], 1)
            with self.assertRaises(CaptureCollisionError):
                append_capture(
                    root=root,
                    capture_id=capture_id_for_date("2026-09-04"),
                    timing=self._timing("2026-09-04"),
                    models=self._models(marker="replacement"),
                )
            self.assertEqual(hashlib.sha256((destination / "manifest.json").read_bytes()).hexdigest(), old_hash)

    def test_same_forecast_content_on_consecutive_days_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(Path(tmp))
            for day in ("2026-09-04", "2026-09-05"):
                append_capture(
                    root=root,
                    capture_id=capture_id_for_date(day),
                    timing=self._timing(day),
                    models=self._models(marker="same-published-content"),
                )
            self.assertEqual(validate_archive(root)["capture_count"], 2)

    def test_different_capture_id_cannot_claim_same_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(Path(tmp))
            append_capture(
                root=root,
                capture_id=capture_id_for_date("2026-09-04"),
                timing=self._timing("2026-09-04"),
                models=self._models(),
            )
            with self.assertRaises(CaptureCollisionError):
                append_capture(
                    root=root,
                    capture_id="different-id",
                    timing=self._timing("2026-09-04"),
                    models=self._models(),
                )

    def test_tampering_and_path_escape_fail_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(Path(tmp))
            destination, _ = append_capture(
                root=root,
                capture_id=capture_id_for_date("2026-09-04"),
                timing=self._timing("2026-09-04"),
                models=self._models(),
            )
            (destination / "election_simulator" / "forecast.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ArchiveValidationError):
                validate_archive(root)

    def test_index_failure_never_indexes_partial_capture_and_retry_recovers_exact_orphan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(Path(tmp))
            capture_id = capture_id_for_date("2026-09-04")
            with patch(
                "scripts.prospective_benchmark_2026.archive._atomic_replace",
                side_effect=OSError("simulated index fsync failure"),
            ):
                with self.assertRaises(OSError):
                    append_capture(
                        root=root,
                        capture_id=capture_id,
                        timing=self._timing("2026-09-04"),
                        models=self._models(),
                    )
            index = json.loads((root / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["captures"], [])
            orphan_forecast = (root / "captures" / capture_id / "election_simulator" / "forecast.json").read_bytes()
            append_capture(
                root=root,
                capture_id=capture_id,
                timing=self._timing("2026-09-04"),
                models=self._models(marker="must-not-replace-orphan"),
            )
            self.assertEqual(
                (root / "captures" / capture_id / "election_simulator" / "forecast.json").read_bytes(),
                orphan_forecast,
            )
            self.assertEqual(validate_archive(root)["capture_count"], 1)

    def test_manifest_and_index_bind_active_amendment_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(Path(tmp))
            amendment = self._add_amendment(root)
            destination, row = append_capture(
                root=root,
                capture_id=capture_id_for_date("2026-09-04"),
                timing=self._timing("2026-09-04"),
                models=self._models(),
            )
            manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["amendments"], [amendment])
            self.assertEqual(row["amendments"], [amendment])
            self.assertEqual(validate_archive(root)["active_amendments"], [amendment])

            amendment_path = root / str(amendment["path"])
            amendment_path.write_bytes(amendment_path.read_bytes() + b"tamper")
            with self.assertRaises(ArchiveValidationError):
                validate_archive(root)

    def test_missing_amendment_sidecar_or_index_reference_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(Path(tmp))
            self._add_amendment(root)
            (root / "amendments" / "001-fixture.sha256").unlink()
            with self.assertRaises(ArchiveValidationError):
                validate_archive(root)

        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(Path(tmp))
            self._add_amendment(root)
            index = json.loads((root / "index.json").read_text(encoding="utf-8"))
            index.pop("amendments")
            (root / "index.json").write_bytes(canonical_json_bytes(index))
            with self.assertRaises(ArchiveValidationError):
                validate_archive(root)

    def test_append_rejects_retroactive_timing_at_archive_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(Path(tmp))
            timing = classify_capture_time(
                "2026-09-04", "2026-09-06T00:00:00Z", durable=False
            ).to_dict()
            with self.assertRaises(ArchiveValidationError):
                append_capture(
                    root=root,
                    capture_id=capture_id_for_date("2026-09-04"),
                    timing=timing,
                    models=self._models(),
                )

    def test_unfinished_staging_directory_blocks_validation_and_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(Path(tmp))
            captures = root / "captures"
            captures.mkdir()
            (captures / ".unfinished.staging").mkdir()
            with self.assertRaises(ArchiveValidationError):
                validate_archive(root)
            with self.assertRaises(ArchiveValidationError):
                append_capture(
                    root=root,
                    capture_id=capture_id_for_date("2026-09-04"),
                    timing=self._timing("2026-09-04"),
                    models=self._models(),
                )

    def test_existing_archive_corruption_is_detected_before_new_install(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(Path(tmp))
            first_destination, _ = append_capture(
                root=root,
                capture_id=capture_id_for_date("2026-09-04"),
                timing=self._timing("2026-09-04"),
                models=self._models(),
            )
            index_before = (root / "index.json").read_bytes()
            (first_destination / "election_simulator" / "forecast.json").write_bytes(b"tampered\n")
            with self.assertRaises(ArchiveValidationError):
                append_capture(
                    root=root,
                    capture_id=capture_id_for_date("2026-09-05"),
                    timing=self._timing("2026-09-05"),
                    models=self._models(),
                )
            self.assertEqual((root / "index.json").read_bytes(), index_before)
            self.assertFalse((root / "captures" / capture_id_for_date("2026-09-05")).exists())


if __name__ == "__main__":
    unittest.main()
