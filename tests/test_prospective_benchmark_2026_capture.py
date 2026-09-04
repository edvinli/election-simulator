"""Offline orchestration tests for one prospective benchmark slot."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prospective_benchmark_2026.archive import CaptureCollisionError, validate_archive
from scripts.prospective_benchmark_2026.capture import run_capture


class TestProspectiveBenchmarkCapture(unittest.TestCase):
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

    def test_collector_outages_become_immutable_failure_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(Path(tmp))

            def unavailable(*_args, **_kwargs):
                raise RuntimeError("fixture source unavailable")

            result = run_capture(
                scheduled_date="2026-09-04",
                mode="capture",
                archive_root=root,
                es_archive_root=root,
                repo_root=root,
                es_collector=unavailable,
                ada_collector=unavailable,
                _clock=lambda: datetime.fromisoformat("2026-09-04T21:31:00+00:00"),
            )
            self.assertEqual(result["models"], {
                "election_simulator": "SOURCE_UNAVAILABLE",
                "botten_ada": "SOURCE_UNAVAILABLE",
            })
            self.assertEqual(validate_archive(root)["capture_count"], 1)
            capture_dir = root / "captures" / result["capture_id"]
            for system in ("election_simulator", "botten_ada"):
                forecast = json.loads((capture_dir / system / "forecast.json").read_text(encoding="utf-8"))
                self.assertFalse(forecast["available"])
                self.assertFalse(forecast["carry_forward"])

    def test_indexed_slot_is_rejected_before_collectors_are_called(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(Path(tmp))

            def unavailable(*_args, **_kwargs):
                raise RuntimeError("fixture source unavailable")

            kwargs = {
                "scheduled_date": "2026-09-04",
                "mode": "capture",
                "archive_root": root,
                "es_archive_root": root,
                "repo_root": root,
                "es_collector": unavailable,
                "ada_collector": unavailable,
                "_clock": lambda: datetime.fromisoformat("2026-09-04T21:31:00+00:00"),
            }
            run_capture(**kwargs)

            calls = 0

            def should_not_run(*_args, **_kwargs):
                nonlocal calls
                calls += 1
                raise AssertionError("indexed slot must fail before source collection")

            kwargs["es_collector"] = should_not_run
            kwargs["ada_collector"] = should_not_run
            with self.assertRaises(CaptureCollisionError):
                run_capture(**kwargs)
            self.assertEqual(calls, 0)

    def test_dry_run_before_cutoff_is_not_durable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(Path(tmp))

            def unavailable(*_args, **_kwargs):
                raise RuntimeError("fixture source unavailable")

            result = run_capture(
                scheduled_date="2026-09-04",
                mode="dry_run",
                archive_root=root,
                es_archive_root=root,
                repo_root=root,
                es_collector=unavailable,
                ada_collector=unavailable,
                _clock=lambda: datetime.fromisoformat("2026-09-04T20:00:00+00:00"),
            )
            self.assertFalse(result["durable_write"])
            self.assertFalse((root / "captures").exists())


if __name__ == "__main__":
    unittest.main()
