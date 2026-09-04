"""Offline orchestration tests for one prospective benchmark slot."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prospective_benchmark_2026.archive import CaptureCollisionError, validate_archive
from scripts.prospective_benchmark_2026.botten_ada_capture import BottenAdaCapture
from scripts.prospective_benchmark_2026.capture import (
    _normalize_botten_ada,
    _normalize_election_simulator,
    run_capture,
)


PARTIES = ("M", "L", "C", "KD", "S", "V", "MP", "SD")


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

    def test_es_point_forecast_is_published_p50_not_supplementary_mean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            distributions = {
                party: {
                    "quantiles": {
                        "p05": float(index),
                        "p25": float(index + 1),
                        "p50": float(index + 2),
                        "p75": float(index + 3),
                        "p95": float(index + 4),
                    }
                }
                for index, party in enumerate(PARTIES)
            }
            snapshot = {
                "generation_id": "fixture-generation",
                "generated_at_utc": "2026-09-04T20:00:00Z",
                "as_of": "2026-09-04",
                "election_date": "2026-09-13",
                "samples": 100_000,
                "seed": 12345,
                "model": {"version": "1.1.0-rc1"},
                "source_git_commit": "a" * 40,
                "source_worktree_clean": True,
                "deterministic_payload_sha256": "b" * 64,
                "hashes": {},
                "national_vote_summary": {
                    party: {"vote_share_mean": float(index + 20)}
                    for index, party in enumerate(PARTIES)
                },
                "national_vote_distributions": distributions,
                "threshold_probabilities_4pct": {party: 0.5 for party in PARTIES},
                "seat_summary": {party: {"mean": 1, "median": 1, "p05": 0, "p95": 2} for party in PARTIES},
            }
            normalized = _normalize_election_simulator(
                {
                    "forecast": snapshot,
                    "exact_draws": {"status": "UNAVAILABLE_NO_VERIFIED_DRAWS"},
                    "provenance": {},
                },
                repo_root=Path(tmp),
            )
            self.assertEqual(normalized.forecast["published_central_prediction"]["kind"], "published_vote_share_p50")
            self.assertEqual(normalized.forecast["published_central_prediction"]["values"]["M"], 2.0)
            self.assertEqual(normalized.forecast["supplementary_vote_share_mean"]["M"], 20.0)

    def test_available_ada_publication_is_not_downgraded_by_unverified_draws(self) -> None:
        election = {
            party: {
                "votes": {"p5": 0.01, "p50": 0.02, "p95": 0.03},
                "seats": {"p5": 0, "p50": 1, "p95": 2},
            }
            for party in PARTIES
        }
        captured = BottenAdaCapture(
            record={
                "status": "AVAILABLE",
                "election_date": "2026-09-13",
                "forecast": {
                    "metadata": {"run": "fixture", "model": "model8m10", "run_written": "2026-09-04 20:00:00"},
                    "election": election,
                },
                "threshold_probabilities_4pct": {},
                "latest_polls": None,
                "provenance": {},
                "errors": {},
            },
            raw_files={},
        )
        normalized = _normalize_botten_ada(captured)
        self.assertEqual(normalized.status, "AVAILABLE")
        self.assertTrue(normalized.forecast["available"])
        self.assertEqual(normalized.forecast["draws"]["status"], "PARITY_UNVERIFIED")
        self.assertFalse(normalized.forecast["draws"]["verified_predictive_vote_draws"])


if __name__ == "__main__":
    unittest.main()
