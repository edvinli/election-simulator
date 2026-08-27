"""Deterministic contract and scoring tests for the Botten Ada benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.botten_ada_benchmark.adapters import bundle_from_simulation_result, load_bundle
from scripts.botten_ada_benchmark.config import PARTY_ORDER
from scripts.botten_ada_benchmark.harness import run_benchmark
from scripts.botten_ada_benchmark.metrics import continuous_crps, threshold_brier


def _bundle(candidate: str, offset: float = 0.0) -> dict:
    draws = [
        [20 + offset, 5, 7, 6, 30, 8, 5, 19],
        [21 + offset, 4, 7, 6, 29, 8, 5, 20],
        [19 + offset, 6, 7, 5, 31, 7, 5, 20],
        [20 + offset, 5, 6, 6, 30, 8, 6, 19],
    ]
    seats = [[70, 20, 31, 22, 100, 28, 16, 62] for _ in draws]
    return {
        "schema_version": "1.0",
        "candidate": candidate,
        "model_name": "Test model",
        "model_version": "test-1",
        "party_order": list(PARTY_ORDER),
        "source": {"fixture": True},
        "cases": [{
            "election_date": "2018-09-09",
            "as_of": "2018-05-20",
            "horizon_days": 112,
            "vote_draws": draws,
            "seat_draws": seats,
            "actual_vote": [19.84, 5.49, 8.61, 6.32, 28.26, 8.00, 4.41, 17.53],
            "actual_seats": [70, 20, 31, 22, 100, 28, 16, 62],
        }],
    }


class TestBottenAdaBenchmark(unittest.TestCase):
    def test_metric_formulas(self) -> None:
        self.assertAlmostEqual(continuous_crps(np.array([0.0]), 1.0), 1.0)
        self.assertAlmostEqual(threshold_brier(np.array([3.9, 4.1]), 4.2), 0.25)

    def test_bundle_requires_draws_and_fixed_party_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.json"
            path.write_text(json.dumps(_bundle("A")), encoding="utf-8")
            loaded = load_bundle(path, expected_candidate="A")
            self.assertEqual(loaded.party_order, PARTY_ORDER)
            self.assertEqual(loaded.cases[0].vote_draws.shape, (4, 8))
            self.assertEqual(loaded.cases[0].horizon_days, 112)

    def test_candidate_a_adapter_derives_common_cutoff_horizon(self) -> None:
        result = type(
            "Result",
            (),
            {
                "manifest": {"election_date": "2026-09-13", "as_of": "2026-08-23", "model_version": "test-1"},
                "vote_shares_matrix": np.zeros((2, 8)),
                "seats_matrix": np.zeros((2, 8), dtype=np.int64),
            },
        )()
        bundle = bundle_from_simulation_result(result, source={"fixture": True})
        self.assertEqual(bundle.cases[0].horizon_days, 21)

    def test_comparison_and_preregistered_pivot_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.json"
            b = root / "b.json"
            a.write_text(json.dumps(_bundle("A")), encoding="utf-8")
            b.write_text(json.dumps(_bundle("B", offset=0.5)), encoding="utf-8")
            report = run_benchmark(a, b)
            self.assertEqual(report["benchmark_status"], "COMPLETE")
            self.assertEqual(report["common_case_count"], 1)
            self.assertIn(report["pivot_decision"]["status"], {"KEEP_CANDIDATE_A_UNCHANGED", "TARGETED_LAYER_INVESTIGATION_ELIGIBLE", "NOT_ASSESSED_NO_SCORABLE_LATE_CASES"})

    def test_missing_external_model_is_honestly_not_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.json"
            path.write_text(json.dumps(_bundle("A")), encoding="utf-8")
            report = run_benchmark(path)
            self.assertEqual(report["benchmark_status"], "NOT_RUN")
            self.assertEqual(report["candidate_b"]["status"], "NOT_RUN")
            self.assertNotIn("cases", report["candidate_b"])


if __name__ == "__main__":
    unittest.main()
