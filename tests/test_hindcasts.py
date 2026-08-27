"""Unit and regression tests for Election Hindcast v1."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import unittest
import numpy as np

from scripts.elections.load import get_election_target, load_election_targets_for_forecasting
from scripts.hindcasts.hindcast import (
    EVALUATION_ELECTIONS,
    calculate_empirical_midrank_percentile,
    run_election_hindcasts,
)
from scripts.hindcasts.models import (
    derive_opinion_state_seed,
    derive_shared_dynamics_seed,
    hindcast_dynamics_only,
    hindcast_point_persistence,
    hindcast_state_plus_dynamics,
    sample_shared_symmetric_dynamics,
)
from scripts.pollofpolls.clr import composition_to_clr
from scripts.pollofpolls.state import estimate_opinion, load_timeseries_dataset
from scripts.pollofpolls.state_config import ALL_CATEGORIES
from scripts.pollofpolls.transitions import (
    HistoricalTransition,
    build_all_historical_transitions,
    filter_transitions_as_of,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "pollofpolls"
TS_FILE = DATA_DIR / "pollofpolls_timeseries.csv"


class HindcastTargetAlignmentTests(unittest.TestCase):
    """Test official election target loading and FI + OTHER -> REST alignment."""

    def test_target_alignment_and_100_percent_sum(self) -> None:
        targets = load_election_targets_for_forecasting()
        self.assertIn(date(2018, 9, 9), targets)
        self.assertIn(date(2022, 9, 11), targets)

        for el_date, comp in targets.items():
            self.assertEqual(len(comp), 9)
            self.assertIn("REST", comp)
            self.assertNotIn("FI", comp)
            self.assertNotIn("OTHER", comp)
            self.assertAlmostEqual(sum(comp.values()), 100.0, places=4)

    def test_2018_and_2022_official_target_values(self) -> None:
        t2018 = get_election_target("2018-09-09")
        self.assertAlmostEqual(t2018["S"], 28.2610, places=3)
        self.assertAlmostEqual(t2018["M"], 19.8356, places=3)
        self.assertAlmostEqual(t2018["SD"], 17.5340, places=3)
        self.assertAlmostEqual(t2018["REST"], 1.5307, places=3)

        t2022 = get_election_target("2022-09-11")
        self.assertAlmostEqual(t2022["S"], 30.3255, places=3)
        self.assertAlmostEqual(t2022["SD"], 20.5361, places=3)
        self.assertAlmostEqual(t2022["M"], 19.1021, places=3)
        self.assertAlmostEqual(t2022["REST"], 1.5476, places=3)


class ArbitraryHorizonDynamicsTests(unittest.TestCase):
    """Test arbitrary integer horizon transition construction and leakage boundaries."""

    def test_arbitrary_integer_horizon_transitions(self) -> None:
        ts_data = load_timeseries_dataset(TS_FILE)
        # Request arbitrary non-standard horizons: 5d, 12d, 45d
        transitions_by_h = build_all_historical_transitions(ts_data, horizons=(5, 12, 45))
        self.assertIn(5, transitions_by_h)
        self.assertIn(12, transitions_by_h)
        self.assertIn(45, transitions_by_h)

        # Check exact endpoint spacing for 45-day transitions
        for t in transitions_by_h[45][:10]:
            self.assertEqual((t.end_date - t.start_date).days, 45)
            self.assertEqual(t.horizon_days, 45)
            self.assertEqual(len(t.clr_transition), 9)
            self.assertAlmostEqual(float(np.sum(t.clr_transition)), 0.0, places=5)

    def test_structural_leakage_boundary(self) -> None:
        origin_date = date(2022, 6, 1)
        ts_data = load_timeseries_dataset(TS_FILE)
        t_56d = build_all_historical_transitions(ts_data, horizons=(56,))[56]

        filtered = filter_transitions_as_of(t_56d, origin_date)
        for t in filtered:
            self.assertLessEqual(t.end_date, origin_date)


class SharedDynamicsAndSeedMechanismsTests(unittest.TestCase):
    """Test shared Monte Carlo dynamics draws, deterministic seeds, and percentile conventions."""

    def test_shared_dynamics_identical_across_models(self) -> None:
        origin_date = date(2022, 8, 14)  # 28d before 2022 election
        h = 28
        base_seed = 12345

        ts_data = load_timeseries_dataset(TS_FILE)
        t_28d = build_all_historical_transitions(ts_data, horizons=(h,))[h]
        eligible = filter_transitions_as_of(t_28d, origin_date)

        dyn_seed = derive_shared_dynamics_seed(base_seed, origin_date, h)
        state_seed = derive_opinion_state_seed(base_seed, origin_date)

        # 1. Sample shared dynamics deltas twice with same seed -> must be identical
        deltas1 = sample_shared_symmetric_dynamics(eligible, 500, dyn_seed)
        deltas2 = sample_shared_symmetric_dynamics(eligible, 500, dyn_seed)
        np.testing.assert_array_equal(deltas1, deltas2)

        # 2. State seed must be independent and different from dynamics seed
        self.assertNotEqual(dyn_seed, state_seed)

    def test_empirical_midrank_percentile_calculation(self) -> None:
        samples = np.array([10.0, 20.0, 30.0, 40.0])
        # Actual = 5.0 -> 0%
        self.assertEqual(calculate_empirical_midrank_percentile(samples, 5.0), 0.0)
        # Actual = 50.0 -> 100%
        self.assertEqual(calculate_empirical_midrank_percentile(samples, 50.0), 100.0)
        # Actual = 25.0 -> 2 below, 2 above -> 50%
        self.assertEqual(calculate_empirical_midrank_percentile(samples, 25.0), 50.0)
        # Actual = 20.0 -> 1 below, 1 equal -> (1 + 0.5)/4 = 37.5%
        self.assertEqual(calculate_empirical_midrank_percentile(samples, 20.0), 37.5)

    def test_end_to_end_hindcast_execution(self) -> None:
        """Run a fast hindcast check across 7d horizon for both elections."""
        res = run_election_hindcasts(
            elections=(date(2018, 9, 9), date(2022, 9, 11)),
            horizons=(7,),
            models=("point_persistence", "dynamics_only", "state_plus_dynamics"),
            samples=200,
            seed=12345,
        )
        self.assertEqual(len(res["summary"]["skipped_cases"]), 0)
        self.assertEqual(len(res["cases_df"]), 2 * 1 * 3 * 9)  # 2 elections * 1 horizon * 3 models * 9 parties = 54 rows


if __name__ == "__main__":
    unittest.main()
