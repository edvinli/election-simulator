"""Tests for the separately versioned Poll of Polls simulation baseline."""

from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.pop_baseline.config import DEFAULT_CONFIG, PARTY_ORDER, PoPBaselineConfig
from scripts.pop_baseline.model import apply_support_voting, load_stored_origin_pop, simulate_baseline


class PoPBaselineTests(unittest.TestCase):
    def test_exact_origin_is_required(self) -> None:
        with self.assertRaises(KeyError):
            load_stored_origin_pop(date(2014, 9, 14))

    def test_seeded_forecast_is_reproducible_and_compositional(self) -> None:
        kwargs = {
            "origin_date": date(2022, 8, 14),
            "horizon_days": 28,
            "samples_count": 128,
            "seed": 20260913,
        }
        first = simulate_baseline(**kwargs)
        second = simulate_baseline(**kwargs)
        np.testing.assert_array_equal(first.samples_matrix, second.samples_matrix)
        np.testing.assert_array_equal(first.raw_samples_matrix, second.raw_samples_matrix)
        self.assertEqual(first.model_version, "PoPBaseline-v1.0")
        self.assertEqual(first.party_order, PARTY_ORDER)
        self.assertTrue(np.allclose(first.samples_matrix.sum(axis=1), 100.0, atol=1e-10))
        self.assertTrue(np.all(first.samples_matrix >= 0.0))
        self.assertEqual(first.diagnostics["step_windows"], [21, 28, 35])
        self.assertEqual(first.diagnostics["current_state_uncertainty"], "none")
        self.assertEqual(first.diagnostics["election_residual"], "none")

    def test_support_voting_preserves_simplex_and_is_seeded(self) -> None:
        raw = np.array([
            [18.0, 3.8, 7.0, 3.5, 31.0, 3.7, 4.3, 25.0, 3.7],
            [18.0, 4.2, 7.0, 4.5, 31.0, 3.2, 4.1, 25.0, 3.0],
        ])
        a, diag = apply_support_voting(raw, seed=42)
        b, _ = apply_support_voting(raw, seed=42)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.allclose(a.sum(axis=1), 100.0, atol=1e-10))
        self.assertTrue(np.all(a >= 0.0))
        self.assertEqual(diag["formula_domain"], "2 < s_sim < 5")
        self.assertGreater(diag["active_draws_by_target"]["L"], 0)

    def test_support_voting_can_be_disabled_without_changing_raw_paths(self) -> None:
        config = PoPBaselineConfig(apply_support_voting=False)
        forecast = simulate_baseline(
            origin_date="2022-08-14",
            horizon_days=28,
            samples_count=96,
            seed=17,
            config=config,
        )
        np.testing.assert_array_equal(forecast.samples_matrix, forecast.raw_samples_matrix)
        self.assertFalse(forecast.diagnostics["support_voting"]["enabled"])

    def test_provenance_record_is_first_party_and_versioned(self) -> None:
        path = Path(__file__).resolve().parents[1] / "data" / "raw" / "pollofpolls" / "pop_baseline_provenance.json"
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        self.assertEqual(record["baseline_version"], "PoPBaseline-v1.0")
        self.assertGreaterEqual(len(record["sources"]), 3)
        self.assertTrue(all(source["url"].startswith("https://pollofpolls.se/") for source in record["sources"]))
        self.assertTrue(all(len(source["evidence_sha256"]) == 64 for source in record["sources"]))
        for source in record["sources"]:
            self.assertEqual(
                source["evidence_sha256"],
                hashlib.sha256(source["evidence_paraphrase"].encode("utf-8")).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
