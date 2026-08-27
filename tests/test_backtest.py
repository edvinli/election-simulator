"""Comprehensive unit, regression, and leakage safety tests for the backtesting framework."""

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
    derive_origin_seed,
    generate_forecast_origins,
    run_backtest,
)
from scripts.pollofpolls.backtest_context import (
    ForecastContext,
    HistoricalTransition,
    filter_transitions_as_of,
)
from scripts.pollofpolls.backtest_metrics import (
    calculate_crps,
    calculate_interval_metrics,
    calculate_point_error,
    precompute_crps_sample_term,
)
from scripts.pollofpolls.backtest_models import MODELS, ForecastDistribution, NoChangeModel
from scripts.pollofpolls.state import estimate_opinion


class CRPSAndMetricsTests(unittest.TestCase):
    """Test fast CRPS calculation against exact analytical solutions and slow double-loop reference."""

    def test_crps_hand_calculated_example(self) -> None:
        # Hand-calculated example: samples = [1.0, 3.0, 6.0], observation y = 4.0
        # n = 3
        # term1 = (|1-4| + |3-4| + |6-4|) / 3 = (3 + 1 + 2) / 3 = 6 / 3 = 2.0
        # term2 = sum_{i,j} |x_i - x_j| / (2 * 9)
        # Pairs:
        # |1-1|=0, |1-3|=2, |1-6|=5
        # |3-1|=2, |3-3|=0, |3-6|=3
        # |6-1|=5, |6-3|=3, |6-6|=0
        # Sum of diffs = 2 + 5 + 2 + 3 + 5 + 3 = 20
        # term2 = 20 / 18 = 10 / 9 = 1.111111...
        # CRPS = 2.0 - 1.111111... = 8 / 9 = 0.8888888...
        samples = [1.0, 3.0, 6.0]
        y = 4.0
        crps = calculate_crps(samples, y)
        self.assertAlmostEqual(crps, 8.0 / 9.0, places=7)

    def test_fast_crps_matches_double_loop(self) -> None:
        rng = np.random.default_rng(42)
        samples = rng.normal(loc=15.0, scale=2.5, size=200)
        y = 14.2
        n = len(samples)

        # Slow double loop O(n^2)
        term1 = float(np.mean(np.abs(samples - y)))
        term2_slow = float(np.sum(np.abs(samples[:, None] - samples[None, :])) / (2.0 * n * n))
        crps_slow = term1 - term2_slow

        # Fast O(n log n)
        crps_fast = calculate_crps(samples, y)
        self.assertAlmostEqual(crps_fast, crps_slow, places=9)

    def test_point_errors_and_interval_coverage(self) -> None:
        # P50 = 20.0, Actual = 18.5
        errs = calculate_point_error(20.0, 18.5)
        self.assertAlmostEqual(errs["error"], 1.5)
        self.assertAlmostEqual(errs["absolute_error"], 1.5)
        self.assertAlmostEqual(errs["squared_error"], 2.25)

        quantiles = {
            0.05: 16.0,
            0.10: 17.0,
            0.25: 18.0,
            0.50: 20.0,
            0.75: 22.0,
            0.90: 23.0,
            0.95: 24.0,
        }
        # Actual 18.5 is inside [18, 22] (50%), [17, 23] (80%), [16, 24] (90%)
        iv = calculate_interval_metrics(quantiles, 18.5)
        self.assertEqual(iv["interval50_contains_actual"], 1)
        self.assertEqual(iv["interval80_contains_actual"], 1)
        self.assertEqual(iv["interval90_contains_actual"], 1)
        self.assertAlmostEqual(iv["width_50"], 4.0)
        self.assertAlmostEqual(iv["width_80"], 6.0)
        self.assertAlmostEqual(iv["width_90"], 8.0)

        # Actual 16.5 is inside [16, 24] (90%), outside [18, 22] (50%) and outside [17, 23] (80%)
        iv_out = calculate_interval_metrics(quantiles, 16.5)
        self.assertEqual(iv_out["interval50_contains_actual"], 0)
        self.assertEqual(iv_out["interval80_contains_actual"], 0)
        self.assertEqual(iv_out["interval90_contains_actual"], 1)


class ContextAndLeakageTests(unittest.TestCase):
    """Test forecast context, transition filtering, and future data leakage prevention."""

    def test_filter_transitions_as_of(self) -> None:
        transitions = [
            HistoricalTransition(date(2020, 1, 1), date(2020, 1, 15), 14, {}),
            HistoricalTransition(date(2020, 1, 10), date(2020, 1, 25), 14, {}),
            HistoricalTransition(date(2020, 1, 20), date(2020, 2, 5), 14, {}),
        ]
        # Filter as of 2020-01-20
        filtered = filter_transitions_as_of(transitions, date(2020, 1, 20))
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].end_date, date(2020, 1, 15))

        # Filter as of 2020-01-25
        filtered_25 = filter_transitions_as_of(transitions, date(2020, 1, 25))
        self.assertEqual(len(filtered_25), 2)

    def test_future_poll_injection_does_not_alter_earlier_backtest_forecasts(self) -> None:
        base_dir = Path(__file__).resolve().parents[1] / "data" / "processed" / "pollofpolls"
        origin_date = date(2020, 5, 6)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            shutil.copy(base_dir / "pollofpolls_timeseries.csv", tmp_path / "pollofpolls_timeseries.csv")
            shutil.copy(base_dir / "individual_polls.csv", tmp_path / "individual_polls.csv")

            # 1. Baseline backtest on 2020-05-06
            res_baseline = run_backtest(
                model="no_change",
                start_date=origin_date,
                end_date=origin_date,
                horizons=(7, 14, 28),
                data_dir=tmp_path,
                output_dir=tmp_path / "out1",
                seed=12345,
            )
            df_base = res_baseline["results_df"]

            # 2. Inject future poll published in 2025 into individual_polls.csv
            with (tmp_path / "individual_polls.csv").open("a", encoding="utf-8") as f:
                writer = csv.writer(f)
                for party in ["M", "L", "C", "KD", "S", "V", "MP", "SD", "FI", "other"]:
                    writer.writerow([
                        "pop-leakage-test-fake",
                        "Novus",
                        "Novus",
                        "2025-01-01",
                        "2025-01-10",
                        "2025-01-12",
                        party,
                        "12.0",
                        "12.0",
                        "reported",
                        "1000",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "exact_span_match",
                        "[]",
                    ])

            # 3. Run backtest again on the same historical origin
            res_injected = run_backtest(
                model="no_change",
                start_date=origin_date,
                end_date=origin_date,
                horizons=(7, 14, 28),
                data_dir=tmp_path,
                output_dir=tmp_path / "out2",
                seed=12345,
            )
            df_inj = res_injected["results_df"]

            # 4. Verify all forecast rows, point forecasts, errors, and CRPS values match exactly
            self.assertEqual(len(df_base), len(df_inj))
            pd.testing.assert_frame_equal(df_base, df_inj)


class BacktestExecutionAndLogicTests(unittest.TestCase):
    """Test origin generation, seed stability, exact target matching, and aggregations."""

    def test_generate_forecast_origins_weekly(self) -> None:
        d1 = date(2024, 1, 1)
        d2 = date(2024, 1, 22)
        origins = generate_forecast_origins(d1, d2, step_days=7)
        expected = [
            date(2024, 1, 1),
            date(2024, 1, 8),
            date(2024, 1, 15),
            date(2024, 1, 22),
        ]
        self.assertEqual(origins, expected)

        with self.assertRaises(ValueError):
            generate_forecast_origins(date(2024, 2, 1), date(2024, 1, 1))

    def test_deterministic_seed_derivation(self) -> None:
        seed1 = derive_origin_seed(12345, "no_change", date(2024, 5, 1))
        seed2 = derive_origin_seed(12345, "no_change", date(2024, 5, 1))
        seed3 = derive_origin_seed(12345, "no_change", date(2024, 5, 8))
        seed4 = derive_origin_seed(99999, "no_change", date(2024, 5, 1))

        # Same inputs produce identical seed
        self.assertEqual(seed1, seed2)
        # Different origin produces different seed
        self.assertNotEqual(seed1, seed3)
        # Different base seed produces different seed
        self.assertNotEqual(seed1, seed4)

    def test_missing_target_observation_skipped_cleanly(self) -> None:
        # Origin = 2026-08-23 (the maximum date in dataset)
        # Horizon 7d -> Target 2026-08-30 (which does not exist)
        res = run_backtest(
            model="no_change",
            start_date="2026-08-23",
            end_date="2026-08-23",
            horizons=(7, 14),
            samples=500,
        )
        self.assertEqual(res["summary"]["evaluated_cases_count"], 0)
        self.assertEqual(res["summary"]["skipped_cases_count"], 2)
        self.assertEqual(len(res["results_df"]), 0)

    def test_no_change_samples_identical_across_horizons_for_same_origin(self) -> None:
        res = run_backtest(
            model="no_change",
            start_date="2024-01-01",
            end_date="2024-01-01",
            horizons=(7, 14, 28),
            samples=1000,
            seed=42,
        )
        df = res["results_df"]
        # Point forecasts (predictive P50) for party 'S' should be identical across horizons 7, 14, 28
        s_rows = df[df["party"] == "S"]
        p50_vals = s_rows["point_forecast"].tolist()
        self.assertEqual(len(p50_vals), 3)
        self.assertEqual(p50_vals[0], p50_vals[1])
        self.assertEqual(p50_vals[1], p50_vals[2])

        # Widths should also be identical
        w50_vals = s_rows["width_50"].tolist()
        self.assertEqual(w50_vals[0], w50_vals[1])


if __name__ == "__main__":
    unittest.main()
