"""Tests for the retrospective campaign-path evaluation harness."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from scripts.campaign_path_eval.evaluate import (
    MODEL_IDS,
    build_origins,
    crps_matrix,
    evaluate_campaign_paths,
    write_evaluation_artifacts,
)
from scripts.pollofpolls.backtest_metrics import calculate_crps


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMESERIES = REPO_ROOT / "data" / "processed" / "pollofpolls" / "pollofpolls_timeseries.csv"


class CrpsEstimatorTests(unittest.TestCase):
    def test_vectorized_crps_matches_the_repository_scalar_estimator(self) -> None:
        generator = np.random.default_rng(11)
        draws = generator.normal(size=(3, 4, 250)) * 2.0 + 20.0
        actual = generator.normal(size=(3, 4)) * 2.0 + 20.0
        vectorized = crps_matrix(draws, actual)
        for i in range(draws.shape[0]):
            for j in range(draws.shape[1]):
                self.assertAlmostEqual(
                    float(vectorized[i, j]),
                    calculate_crps(draws[i, j], float(actual[i, j])),
                    places=12,
                )

    def test_a_point_mass_scores_the_absolute_error(self) -> None:
        draws = np.full((1, 500), 30.0)
        self.assertAlmostEqual(float(crps_matrix(draws, np.array([31.5]))[0]), 1.5, places=12)

    def test_a_single_draw_is_accepted(self) -> None:
        self.assertAlmostEqual(float(crps_matrix(np.array([[4.0]]), np.array([6.0]))[0]), 2.0)

    def test_empty_draws_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "without draws"):
            crps_matrix(np.zeros((2, 0)), np.zeros(2))


class OriginSelectionTests(unittest.TestCase):
    def test_origins_stop_early_enough_for_a_complete_trajectory(self) -> None:
        first = date(2020, 1, 1)
        dates = [first + timedelta(days=offset) for offset in range(100)]
        origins = build_origins(dates, path_days=28, stride_days=14, start=first)
        self.assertEqual(origins[0], first)
        for origin in origins:
            self.assertIn(origin + timedelta(days=28), dates)
        self.assertLessEqual(origins[-1] + timedelta(days=28), dates[-1])

    def test_a_missing_origin_date_is_skipped(self) -> None:
        first = date(2020, 1, 1)
        dates = [first + timedelta(days=offset) for offset in range(100)]
        dates.remove(first + timedelta(days=14))
        origins = build_origins(dates, path_days=28, stride_days=14, start=first)
        self.assertNotIn(first + timedelta(days=14), origins)


@unittest.skipUnless(TIMESERIES.is_file(), "processed Poll of Polls timeseries is not available")
class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evaluation = evaluate_campaign_paths(
            timeseries_file=TIMESERIES,
            path_days=14,
            stride_days=240,
            samples=200,
            start=date(2019, 1, 1),
        )

    def test_every_model_is_scored_at_every_horizon_for_every_party(self) -> None:
        horizons = {row["horizon_days"] for row in self.evaluation.by_horizon}
        models = {row["model"] for row in self.evaluation.by_horizon}
        parties = {row["party"] for row in self.evaluation.by_horizon}
        self.assertEqual(horizons, set(range(1, 15)))
        self.assertEqual(models, set(MODEL_IDS))
        self.assertEqual(len(parties), 8)
        self.assertEqual(len(self.evaluation.by_horizon), 14 * len(MODEL_IDS) * 8)

    def test_the_endpoint_is_bitwise_identical_at_every_origin(self) -> None:
        """The decisive parity evidence: no endpoint model change occurred."""

        self.assertTrue(self.evaluation.summary["endpoint_bitwise_identical_all_origins"])
        self.assertEqual(self.evaluation.summary["endpoint_max_abs_crps_difference"], 0.0)
        for row in self.evaluation.endpoint_parity:
            self.assertTrue(row["bitwise_identical"])
            self.assertEqual(row["crps_campaign_paths"], row["crps_dynamics_v2"])

    def test_the_path_model_beats_the_frozen_opinion_baseline(self) -> None:
        paths = self.evaluation.summary["campaign_paths_mean_crps_all_horizons"]
        frozen = self.evaluation.summary["frozen_state_mean_crps_all_horizons"]
        self.assertLess(paths, frozen)

    def test_the_path_model_is_better_calibrated_than_a_constant_width_fan(self) -> None:
        paths = self.evaluation.summary["campaign_paths_coverage_90_all_horizons"]
        constant = self.evaluation.summary["endpoint_fan_coverage_90_all_horizons"]
        self.assertLess(abs(paths - 0.90), abs(constant - 0.90))

    def test_interval_width_increases_with_the_horizon(self) -> None:
        rows = [
            row
            for row in self.evaluation.by_horizon
            if row["model"] == "campaign_paths" and row["party"] == "S"
        ]
        rows.sort(key=lambda row: row["horizon_days"])
        self.assertLess(rows[0]["mean_width_90"], rows[-1]["mean_width_90"])

    def test_the_frozen_baseline_is_a_point_mass(self) -> None:
        rows = [row for row in self.evaluation.by_horizon if row["model"] == "frozen_state"]
        self.assertTrue(all(row["mean_width_90"] == 0.0 for row in rows))

    def test_the_evaluation_is_deterministic(self) -> None:
        repeat = evaluate_campaign_paths(
            timeseries_file=TIMESERIES,
            path_days=14,
            stride_days=240,
            samples=200,
            start=date(2019, 1, 1),
        )
        self.assertEqual(repeat.summary, self.evaluation.summary)
        self.assertEqual(repeat.by_horizon, self.evaluation.by_horizon)

    def test_a_too_short_path_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two days"):
            evaluate_campaign_paths(timeseries_file=TIMESERIES, path_days=1)

    def test_artifacts_are_written_with_stable_names(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            written = write_evaluation_artifacts(
                self.evaluation,
                backtest_dir=root / "backtests",
                diagnostics_dir=root / "diagnostics",
            )
            self.assertEqual(
                set(written), {"by_horizon", "energy", "endpoint_parity", "summary"}
            )
            for path in written.values():
                self.assertTrue(Path(path).is_file(), path)
            summary = json.loads(Path(written["summary"]).read_text(encoding="utf-8"))
            self.assertEqual(summary, self.evaluation.summary)
            header = Path(written["by_horizon"]).read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(
                header,
                "model,horizon_days,party,cases,mean_crps,coverage_50,coverage_90,"
                "mean_width_50,mean_width_90",
            )


if __name__ == "__main__":
    unittest.main()
