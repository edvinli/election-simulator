"""A parallel simulation is the serial simulation, byte for byte.

The worker count is an execution detail: national draws are made once, before
the per-draw loop is split, and every later step is a pure function of a draw.
These tests hold that to the bytes the certification hashes, on the real
production inputs, because a parity check on a toy fixture would not exercise
the exact-allocator fallback that a fifth of real draws take.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.election_automation import main as automation_main
from scripts.election_automation_base import PRODUCTION_SIMULATION_WORKERS
from scripts.publication_pipeline import run_publication_pipeline
from scripts.simulator.engine import _draw_chunks, resolve_simulation_workers, simulate_election
from scripts.simulator.pipeline import build_canonical_summary_dict

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# Large enough that every worker gets several chunks and some draws take the
# exact-allocator fallback; small enough to cost seconds.
SAMPLES = 240
SEED = 20_260_913


def _canonical_without_clock(result) -> str:
    canonical = json.loads(json.dumps(build_canonical_summary_dict(result), default=str))
    # The wall clock at the call. The payload hash excludes it by design.
    canonical["manifest"].pop("generated_at_utc", None)
    return json.dumps(canonical, sort_keys=True)


class ParallelParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.serial = simulate_election(samples=SAMPLES, seed=SEED)
        cls.parallel = simulate_election(samples=SAMPLES, seed=SEED, workers=3)

    def test_the_certified_arrays_are_byte_identical(self) -> None:
        for name in ("vote_shares_matrix", "seats_matrix", "threshold_flags"):
            with self.subTest(array=name):
                self.assertEqual(
                    getattr(self.serial, name).tobytes(), getattr(self.parallel, name).tobytes())

    def test_the_payload_hash_and_summary_are_identical(self) -> None:
        self.assertEqual(
            build_canonical_summary_dict(self.serial)["deterministic_payload_sha256"],
            build_canonical_summary_dict(self.parallel)["deterministic_payload_sha256"],
        )
        self.assertEqual(
            _canonical_without_clock(self.serial), _canonical_without_clock(self.parallel))

    def test_the_geography_and_derived_fields_are_identical(self) -> None:
        self.assertEqual(
            json.dumps(self.serial.geography_summary, sort_keys=True, default=str),
            json.dumps(self.parallel.geography_summary, sort_keys=True, default=str),
        )
        self.assertEqual(self.serial.largest_vote_parties, self.parallel.largest_vote_parties)
        self.assertEqual(self.serial.largest_seat_parties, self.parallel.largest_seat_parties)

    def test_the_worker_count_is_recorded_nowhere(self) -> None:
        """Recording it would make identical runs certify differently."""

        self.assertNotIn("workers", json.dumps(self.parallel.manifest, default=str))


class WorkerBoundsTests(unittest.TestCase):
    def test_chunks_cover_every_draw_exactly_once_in_order(self) -> None:
        for samples, workers in ((1, 4), (7, 4), (240, 3), (100_000, 4), (100_001, 4)):
            with self.subTest(samples=samples, workers=workers):
                chunks = _draw_chunks(samples, workers)
                self.assertEqual(chunks[0][0], 0)
                self.assertEqual(chunks[-1][1], samples)
                for (_, stop), (start, _) in zip(chunks, chunks[1:]):
                    self.assertEqual(stop, start)
                self.assertTrue(all(start < stop for start, stop in chunks))

    def test_several_chunks_per_worker_balance_the_fallback(self) -> None:
        self.assertGreaterEqual(len(_draw_chunks(100_000, 4)), 4 * 8)

    def test_the_worker_count_is_bounded_and_validated(self) -> None:
        self.assertEqual(resolve_simulation_workers(None), 1)
        self.assertEqual(resolve_simulation_workers(1), 1)
        with patch("scripts.simulator.engine.os.cpu_count", return_value=2):
            self.assertEqual(resolve_simulation_workers(4), 2)
        for bad in (0, -1, "four"):
            with self.subTest(requested=bad), self.assertRaises(ValueError):
                resolve_simulation_workers(bad)

    def test_the_quantization_audit_stays_serial(self) -> None:
        """It keeps the first mismatches in sample order, a one-pass property."""

        with self.assertRaises(ValueError):
            simulate_election(samples=8, seed=SEED, workers=2, collect_quantization_audit=True)


class ProductionWiringTests(unittest.TestCase):
    def _captured_runner_kwargs(self, **pipeline_kwargs) -> dict:
        captured: dict = {}

        def runner(**kwargs):
            captured.update(kwargs)
            raise RuntimeError("captured")

        run_publication_pipeline(
            processed_root=REPOSITORY_ROOT / "data" / "processed",
            append_archive=False,
            export_publication=False,
            simulation_runner=runner,
            **pipeline_kwargs,
        )
        return captured

    def test_the_pipeline_passes_workers_only_when_asked(self) -> None:
        """An injected runner that predates the option keeps its signature."""

        self.assertNotIn("workers", self._captured_runner_kwargs())
        self.assertEqual(self._captured_runner_kwargs(simulation_workers=4)["workers"], 4)

    def test_the_cli_carries_the_worker_count_to_the_automation(self) -> None:
        for argv, expected in (([], 1), (["--simulation-workers", "4"], 4)):
            with self.subTest(argv=argv), patch(
                "scripts.election_automation_base.run_automation"
            ) as run_automation, contextlib.redirect_stdout(io.StringIO()):
                run_automation.return_value.summary.render.return_value = ""
                run_automation.return_value.to_dict.return_value = {}
                run_automation.return_value.status = "PUBLISHED"
                automation_main(["--site-repo", ".", "--mode", "dry_run", *argv])
                self.assertEqual(run_automation.call_args.kwargs["simulation_workers"], expected)

    @staticmethod
    def _job(workflow: str, name: str) -> str:
        """One top-level job: from its key to the next two-space-indented key."""

        body = workflow.split(f"\n  {name}:\n", 1)[1]
        lines = body.splitlines()
        end = next(
            (i for i, line in enumerate(lines)
             if line.startswith("  ") and not line.startswith("   ") and line.rstrip().endswith(":")),
            len(lines),
        )
        return "\n".join(lines[:end])

    def test_the_workflow_runs_production_and_its_dry_run_on_four_workers(self) -> None:
        workflow = (
            REPOSITORY_ROOT / ".github/workflows/election-simulator-publication.yml"
        ).read_text(encoding="utf-8")
        self.assertEqual(PRODUCTION_SIMULATION_WORKERS, 4)
        for job in ("dry_run", "publish"):
            with self.subTest(job=job):
                self.assertIn(
                    f"--simulation-workers {PRODUCTION_SIMULATION_WORKERS}", self._job(workflow, job))

if __name__ == "__main__":
    unittest.main()
