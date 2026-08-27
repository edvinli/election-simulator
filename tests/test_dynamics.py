"""Unit, regression, and leakage tests for CLR compositional dynamics and empirical models."""

from __future__ import annotations

import csv
from datetime import date, timedelta
import math
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from scripts.pollofpolls.backtest import (
    derive_forecast_seed,
    get_temporal_split,
    run_backtest,
)
from scripts.pollofpolls.backtest_context import ForecastContext
from scripts.pollofpolls.backtest_models import (
    EmpiricalRawModel,
    EmpiricalSymmetricModel,
    PointPersistenceModel,
)
from scripts.pollofpolls.clr import (
    clr_to_composition,
    clr_to_composition_matrix,
    composition_to_clr,
)
from scripts.pollofpolls.state import load_timeseries_dataset
from scripts.pollofpolls.state_config import ALL_CATEGORIES, MIN_SHARE_PCT, PARTIES
from scripts.pollofpolls.transitions import (
    MIN_TRANSITIONS,
    HistoricalTransition,
    build_all_historical_transitions,
    filter_transitions_as_of,
    summarize_transition_pool,
)


class CLRMathTests(unittest.TestCase):
    """Test Centered Log-Ratio transformations and properties."""

    def test_clr_roundtrip_positive_composition(self) -> None:
        comp = {
            "M": 18.5,
            "L": 3.2,
            "C": 6.8,
            "KD": 5.4,
            "S": 31.2,
            "V": 7.9,
            "MP": 6.1,
            "SD": 19.3,
            "REST": 1.6,
        }
        clr_vec, was_floored = composition_to_clr(comp)
        self.assertFalse(was_floored)
        self.assertEqual(len(clr_vec), 9)

        # Sum of CLR coordinates must equal zero
        self.assertAlmostEqual(float(np.sum(clr_vec)), 0.0, places=9)

        # Reconstructed composition must match original within float tolerance
        recon = clr_to_composition(clr_vec)
        self.assertAlmostEqual(sum(recon.values()), 100.0, places=7)
        for cat in ALL_CATEGORIES:
            self.assertAlmostEqual(recon[cat], comp[cat], places=5)

    def test_clr_flooring_and_adjusted_roundtrip(self) -> None:
        comp_zero = {
            "M": 20.0,
            "L": 0.0,  # Zero share
            "C": 5.0,
            "KD": 5.0,
            "S": 35.0,
            "V": 8.0,
            "MP": 7.0,
            "SD": 20.0,
            "REST": 0.0,  # Zero share
        }
        clr_vec, was_floored = composition_to_clr(comp_zero)
        self.assertTrue(was_floored)
        self.assertAlmostEqual(float(np.sum(clr_vec)), 0.0, places=9)

        recon = clr_to_composition(clr_vec)
        self.assertAlmostEqual(sum(recon.values()), 100.0, places=7)
        self.assertGreaterEqual(recon["L"], MIN_SHARE_PCT * 0.99)
        self.assertGreaterEqual(recon["REST"], MIN_SHARE_PCT * 0.99)

        # Roundtrip of the adjusted composition must now match without flooring
        clr_vec2, was_floored2 = composition_to_clr(recon)
        self.assertFalse(was_floored2)
        recon2 = clr_to_composition(clr_vec2)
        for cat in ALL_CATEGORIES:
            self.assertAlmostEqual(recon2[cat], recon[cat], places=5)

    def test_clr_matrix_vectorization(self) -> None:
        clr_mat = np.array([
            [0.5, -0.2, 0.1, -0.4, 1.2, -0.3, -0.1, 0.2, -1.0],
            [-0.1, 0.4, -0.3, 0.2, 0.8, -0.5, 0.1, 0.0, -0.6],
        ])
        comp_mat = clr_to_composition_matrix(clr_mat)
        self.assertEqual(comp_mat.shape, (2, 9))
        self.assertAlmostEqual(float(np.sum(comp_mat[0])), 100.0, places=7)
        self.assertAlmostEqual(float(np.sum(comp_mat[1])), 100.0, places=7)
        self.assertTrue(np.all(comp_mat > 0.0))


class TransitionEngineTests(unittest.TestCase):
    """Test historical transition construction and structural leakage filtering."""

    def test_exact_horizon_transition_construction(self) -> None:
        base_dir = Path(__file__).resolve().parents[1] / "data" / "processed" / "pollofpolls"
        ts_data = load_timeseries_dataset(base_dir / "pollofpolls_timeseries.csv")
        transitions_by_h = build_all_historical_transitions(ts_data, horizons=(7, 14, 28))

        self.assertIn(7, transitions_by_h)
        self.assertIn(14, transitions_by_h)
        self.assertIn(28, transitions_by_h)

        t7 = transitions_by_h[7][0]
        self.assertEqual((t7.end_date - t7.start_date).days, 7)
        self.assertEqual(t7.clr_transition.shape, (9,))

    def test_structural_leakage_filter(self) -> None:
        ts = [
            HistoricalTransition(date(2020, 1, 1), date(2020, 1, 8), 7, np.zeros(9)),
            HistoricalTransition(date(2020, 1, 5), date(2020, 1, 12), 7, np.zeros(9)),
            HistoricalTransition(date(2020, 1, 10), date(2020, 1, 17), 7, np.zeros(9)),
        ]
        # At origin 2020-01-10 -> only 2020-01-08 is eligible (end <= 2020-01-10)
        filtered = filter_transitions_as_of(ts, date(2020, 1, 10))
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].end_date, date(2020, 1, 8))

        # At origin 2020-01-15 -> both 2020-01-08 and 2020-01-12 are eligible
        filtered2 = filter_transitions_as_of(ts, date(2020, 1, 15))
        self.assertEqual(len(filtered2), 2)


class DynamicsModelMechanismTests(unittest.TestCase):
    """Test model implementations, point persistence identity, and symmetric mechanics."""

    def test_point_persistence_crps_identity(self) -> None:
        # For degenerate point forecasts, CRPS == |PoP_t - PoP_{t+h}| == absolute error
        res = run_backtest(
            model="point_persistence",
            start_date="2024-01-01",
            end_date="2024-01-15",
            horizons=(7,),
            samples=100,
        )
        df = res["results_df"]
        self.assertGreater(len(df), 0)
        for _, row in df.iterrows():
            abs_err = abs(row["point_forecast"] - row["actual"])
            self.assertAlmostEqual(row["crps"], abs_err, places=4)
            self.assertAlmostEqual(row["absolute_error"], abs_err, places=4)

    def test_symmetric_model_sign_mechanism(self) -> None:
        origin_pop = {cat: 10.0 for cat in ALL_CATEGORIES}
        origin_pop["S"] = 20.0
        origin_pop = {cat: 100.0 * (v / sum(origin_pop.values())) for cat, v in origin_pop.items()}
        origin_clr, _ = composition_to_clr(origin_pop)

        # Create known synthetic transitions: [1.0, 0, ...]
        known_delta = np.zeros(9)
        known_delta[0] = 0.5
        known_delta[1] = -0.5
        synthetic_transitions = tuple([
            HistoricalTransition(date(2020, 1, 1), date(2020, 1, 8), 7, known_delta)
            for _ in range(50)
        ])

        ctx = ForecastContext(
            origin_date=date(2020, 1, 10),
            origin_pop=origin_pop,
            origin_clr=origin_clr,
            eligible_transitions_by_horizon={7: synthetic_transitions},
        )

        model = EmpiricalSymmetricModel()
        dist = model.forecast(ctx, horizon_days=7, samples_count=1000, seed=42)

        # In symmetric model, M samples will have two branches: origin_clr[0] + 0.5 and origin_clr[0] - 0.5
        m_samples = dist.samples_by_party["M"]
        # Max should be > origin_pop["M"] and Min should be < origin_pop["M"]
        self.assertGreater(float(np.max(m_samples)), origin_pop["M"])
        self.assertLess(float(np.min(m_samples)), origin_pop["M"])

    def test_minimum_transitions_skips_case(self) -> None:
        origin_pop = {cat: 100.0 / 9.0 for cat in ALL_CATEGORIES}
        origin_clr, _ = composition_to_clr(origin_pop)

        # Only 5 transitions (< 30)
        few_transitions = tuple([
            HistoricalTransition(date(2020, 1, 1), date(2020, 1, 8), 7, np.zeros(9))
            for _ in range(5)
        ])

        ctx = ForecastContext(
            origin_date=date(2020, 1, 10),
            origin_pop=origin_pop,
            origin_clr=origin_clr,
            eligible_transitions_by_horizon={7: few_transitions},
        )

        model = EmpiricalRawModel()
        with self.assertRaises(ValueError):
            model.forecast(ctx, horizon_days=7, samples_count=100, seed=123)

    def test_seed_determinism(self) -> None:
        seed1 = derive_forecast_seed(12345, "empirical_raw", date(2024, 1, 1), 14)
        seed2 = derive_forecast_seed(12345, "empirical_raw", date(2024, 1, 1), 14)
        seed3 = derive_forecast_seed(12345, "empirical_raw", date(2024, 1, 1), 28)
        self.assertEqual(seed1, seed2)
        self.assertNotEqual(seed1, seed3)

    def test_temporal_split_boundaries(self) -> None:
        self.assertEqual(get_temporal_split(date(2018, 5, 1)), "Development")
        self.assertEqual(get_temporal_split(date(2022, 12, 31)), "Development")
        self.assertEqual(get_temporal_split(date(2023, 1, 1)), "Validation")
        self.assertEqual(get_temporal_split(date(2023, 12, 31)), "Validation")
        self.assertEqual(get_temporal_split(date(2024, 1, 1)), "Holdout")
        self.assertEqual(get_temporal_split(date(2026, 8, 23)), "Holdout")


if __name__ == "__main__":
    unittest.main()
