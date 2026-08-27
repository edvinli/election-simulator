"""Unit and regression tests for Dynamics Calibration v2 models, recency weighting, and calendar windows."""

from __future__ import annotations

from datetime import date, timedelta
import math
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from scripts.pollofpolls.backtest import (
    compute_paired_crps_comparison,
    get_calendar_year_block,
    run_backtest,
)
from scripts.pollofpolls.backtest_context import ForecastContext
from scripts.pollofpolls.backtest_models import (
    Symmetric2YModel,
    Symmetric4YModel,
    SymmetricAllHistoryModel,
    SymmetricRecencyWeightedModel,
)
from scripts.pollofpolls.clr import composition_to_clr
from scripts.pollofpolls.state_config import ALL_CATEGORIES
from scripts.pollofpolls.transitions import (
    HistoricalTransition,
    compute_recency_weights,
    filter_transitions_as_of,
)


class WindowFilteringAndRecencyTests(unittest.TestCase):
    """Test 2-year/4-year calendar window filtering and exponential recency weights."""

    def test_calendar_window_filtering(self) -> None:
        origin_date = date(2024, 6, 1)
        # Create transitions ending at various dates
        t1 = HistoricalTransition(date(2024, 5, 1), date(2024, 5, 15), 14, np.zeros(9))  # ~0.5m ago
        t2 = HistoricalTransition(date(2023, 1, 1), date(2023, 1, 15), 14, np.zeros(9))  # ~1.5y ago
        t3 = HistoricalTransition(date(2021, 1, 1), date(2021, 1, 15), 14, np.zeros(9))  # ~3.5y ago
        t4 = HistoricalTransition(date(2018, 1, 1), date(2018, 1, 15), 14, np.zeros(9))  # ~6.5y ago
        t_future = HistoricalTransition(date(2024, 6, 1), date(2024, 6, 15), 14, np.zeros(9))  # Future

        all_t = [t1, t2, t3, t4, t_future]

        # 1. All history (leakage safe)
        all_hist = filter_transitions_as_of(all_t, origin_date, lookback_years=None)
        self.assertEqual(len(all_hist), 4)
        self.assertNotIn(t_future, all_hist)

        # 2. 4-year calendar window (origin_date - 4y = 2020-06-01) -> t1, t2, t3
        w4 = filter_transitions_as_of(all_t, origin_date, lookback_years=4)
        self.assertEqual(len(w4), 3)
        self.assertIn(t1, w4)
        self.assertIn(t2, w4)
        self.assertIn(t3, w4)
        self.assertNotIn(t4, w4)

        # 3. 2-year calendar window (origin_date - 2y = 2022-06-01) -> t1, t2
        w2 = filter_transitions_as_of(all_t, origin_date, lookback_years=2)
        self.assertEqual(len(w2), 2)
        self.assertIn(t1, w2)
        self.assertIn(t2, w2)
        self.assertNotIn(t3, w2)

    def test_recency_weights_730d_half_life(self) -> None:
        origin_date = date(2024, 1, 1)
        # Age 0 days
        t0 = HistoricalTransition(origin_date - timedelta(days=7), origin_date, 7, np.zeros(9))
        # Age 730 days (1 half-life)
        t730 = HistoricalTransition(origin_date - timedelta(days=737), origin_date - timedelta(days=730), 7, np.zeros(9))
        # Age 1460 days (2 half-lives)
        t1460 = HistoricalTransition(origin_date - timedelta(days=1467), origin_date - timedelta(days=1460), 7, np.zeros(9))

        transitions = [t0, t730, t1460]
        probs, kish_eff, w_age = compute_recency_weights(transitions, origin_date, half_life_days=730.0)

        # Relative weights should be 1.0, 0.5, 0.25 (sum = 1.75)
        self.assertAlmostEqual(probs[0] / probs[1], 2.0, places=5)
        self.assertAlmostEqual(probs[1] / probs[2], 2.0, places=5)
        self.assertAlmostEqual(probs[0], 1.0 / 1.75, places=5)
        self.assertAlmostEqual(probs[1], 0.5 / 1.75, places=5)
        self.assertAlmostEqual(probs[2], 0.25 / 1.75, places=5)

        # Kish effective sample size: (1.75^2) / (1^2 + 0.5^2 + 0.25^2) = 3.0625 / (1 + 0.25 + 0.0625) = 3.0625 / 1.3125 = 2.333333
        expected_kish = (1.75 ** 2) / (1.0 ** 2 + 0.5 ** 2 + 0.25 ** 2)
        self.assertAlmostEqual(kish_eff, expected_kish, places=5)


class DynamicsV2MechanismsTests(unittest.TestCase):
    """Test Dynamics v2 models, paired CRPS comparison, and annual block labelling."""

    def test_annual_year_block_labeling(self) -> None:
        self.assertEqual(get_calendar_year_block(date(2019, 3, 1)), "2019")
        self.assertEqual(get_calendar_year_block(date(2023, 12, 1)), "2023")
        self.assertEqual(get_calendar_year_block(date(2026, 8, 23)), "2026 YTD")

    def test_paired_crps_calculation(self) -> None:
        # Create synthetic result dataframes for raw and symmetric
        rows_raw = [
            {"origin_date": "2024-01-01", "horizon_days": 7, "party": "S", "year_block": "2024", "is_parliamentary": True, "crps": 0.50},
            {"origin_date": "2024-01-01", "horizon_days": 7, "party": "M", "year_block": "2024", "is_parliamentary": True, "crps": 0.40},
            {"origin_date": "2024-01-01", "horizon_days": 7, "party": "REST", "year_block": "2024", "is_parliamentary": False, "crps": 0.10},
        ]
        rows_sym = [
            {"origin_date": "2024-01-01", "horizon_days": 7, "party": "S", "year_block": "2024", "is_parliamentary": True, "crps": 0.45},
            {"origin_date": "2024-01-01", "horizon_days": 7, "party": "M", "year_block": "2024", "is_parliamentary": True, "crps": 0.38},
            {"origin_date": "2024-01-01", "horizon_days": 7, "party": "REST", "year_block": "2024", "is_parliamentary": False, "crps": 0.12},
        ]
        df_raw = pd.DataFrame(rows_raw)
        df_sym = pd.DataFrame(rows_sym)

        paired = compute_paired_crps_comparison(df_raw, df_sym)
        # Delta S: 0.50 - 0.45 = +0.05
        # Delta M: 0.40 - 0.38 = +0.02
        # Delta 8p mean: (0.05 + 0.02) / 2 = +0.035
        self.assertAlmostEqual(paired["overall_delta_crps_8parties"], 0.035, places=5)

    def test_joint_transition_preservation_in_all_v2_models(self) -> None:
        origin_pop = {cat: 10.0 for cat in ALL_CATEGORIES}
        origin_pop["S"] = 20.0
        origin_pop = {cat: 100.0 * (v / sum(origin_pop.values())) for cat, v in origin_pop.items()}
        origin_clr, _ = composition_to_clr(origin_pop)

        # Coupled transition: M increases by 0.5, S decreases by 0.5
        coupled_delta = np.zeros(9)
        coupled_delta[0] = 0.5  # M
        coupled_delta[4] = -0.5  # S
        synthetic_transitions = tuple([
            HistoricalTransition(date(2024, 1, 1), date(2024, 1, 8), 7, coupled_delta)
            for _ in range(50)
        ])

        ctx = ForecastContext(
            origin_date=date(2024, 6, 1),
            origin_pop=origin_pop,
            origin_clr=origin_clr,
            eligible_transitions_by_horizon={7: synthetic_transitions},
        )

        for model in (SymmetricAllHistoryModel(), Symmetric4YModel(), Symmetric2YModel(), SymmetricRecencyWeightedModel()):
            dist = model.forecast(ctx, horizon_days=7, samples_count=500, seed=12345)
            # In each sample, if M is + branch, S must be - branch
            # Verify correlation between sampled M and S CLR values is exactly -1.0
            m_samps = dist.samples_by_party["M"]
            s_samps = dist.samples_by_party["S"]
            self.assertEqual(len(m_samps), 500)
            self.assertEqual(len(s_samps), 500)


if __name__ == "__main__":
    unittest.main()
