"""Tests for the prospective 2026 fair scoring contract."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.prospective_benchmark_2026.scoring import (
    PRIMARY_PARTY_ORDER,
    PROBABILISTIC_TIER_FAIR_DRAWS,
    PROBABILISTIC_TIER_POINT_MAE,
    PROBABILISTIC_TIER_WIS,
    central_interval_metrics,
    compatible_quantile_forecasts,
    crps_v_statistic,
    energy_score_v_statistic,
    fair_crps,
    fair_energy_score,
    point_mae,
    score_forecast_pair,
    score_vote_ensemble,
    select_primary_scoring_tier,
    threshold_brier,
    threshold_brier_from_probability,
    weighted_interval_score,
)


def _quantile_map(offset: float = 0.0) -> dict[str, dict[float, float]]:
    """Build explicit, central quantile fixtures in percentage points."""

    probabilities = {
        0.025: 0.0 + offset,
        0.05: 0.0 + offset,
        0.10: 0.0 + offset,
        0.25: 0.0 + offset,
        0.50: 1.0 + offset,
        0.75: 2.0 + offset,
        0.90: 2.0 + offset,
        0.95: 2.0 + offset,
        0.975: 2.0 + offset,
    }
    return {party: dict(probabilities) for party in PRIMARY_PARTY_ORDER}


class FairUnivariateScoringTests(unittest.TestCase):
    def test_fair_crps_hand_computable_example(self) -> None:
        # First term = (|0-4| + |2-4|)/2 = 3.
        # Distinct-pair term = |0-2| / (2*1) = 1.
        self.assertAlmostEqual(fair_crps(np.array([0.0, 2.0]), 4.0), 2.0)

    def test_v_statistic_is_preserved_as_a_sensitivity_metric(self) -> None:
        # The V-statistic includes two zero self-pairs: pair term = 4/8=.5,
        # whereas the fair U-statistic pair term is 2/2=1.
        self.assertAlmostEqual(crps_v_statistic(np.array([0.0, 2.0]), 4.0), 2.5)
        self.assertAlmostEqual(fair_crps(np.array([0.0, 2.0]), 4.0), 2.0)

    def test_fair_crps_is_order_invariant(self) -> None:
        values = np.array([3.0, -1.0, 2.5, 8.0])
        self.assertAlmostEqual(fair_crps(values, 4.2), fair_crps(values[::-1], 4.2), places=12)

    def test_fair_crps_rejects_one_draw_instead_of_changing_estimand(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2"):
            fair_crps(np.array([1.0]), 2.0)


class FairMultivariateScoringTests(unittest.TestCase):
    def test_fair_energy_score_hand_computable_example(self) -> None:
        samples = np.array([[0.0], [2.0]])
        # In one dimension this equals fair CRPS for the same values.
        self.assertAlmostEqual(fair_energy_score(samples, np.array([4.0])), 2.0)

    def test_v_energy_score_is_preserved_as_sensitivity(self) -> None:
        samples = np.array([[0.0], [2.0]])
        self.assertAlmostEqual(energy_score_v_statistic(samples, np.array([4.0])), 2.5)

    def test_energy_score_is_order_invariant(self) -> None:
        samples = np.array([[0.0, 5.0], [2.0, 1.0], [9.0, -2.0]])
        actual = np.array([3.0, 2.0])
        self.assertAlmostEqual(
            fair_energy_score(samples, actual),
            fair_energy_score(samples[[2, 0, 1]], actual),
            places=12,
        )

    def test_explicit_pair_sampling_is_reproducible_and_handles_unequal_sizes(self) -> None:
        samples = np.array([[0.0, 5.0], [2.0, 1.0], [9.0, -2.0]])
        actual = np.array([3.0, 2.0])
        first = fair_energy_score(samples, actual, pair_sample_size=100, random_seed=17)
        second = fair_energy_score(samples, actual, pair_sample_size=100, random_seed=17)
        self.assertEqual(first, second)
        # The two models may use different ensemble sizes.  This direct check
        # exercises both exact and sampled paths without any equal-N gate.
        self.assertIsInstance(fair_energy_score(samples[:2], actual), float)
        self.assertIsInstance(fair_energy_score(samples, actual, pair_sample_size=100, random_seed=17), float)

    def test_energy_rejects_one_draw_for_fair_estimator(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2"):
            fair_energy_score(np.array([[1.0, 2.0]]), np.array([1.0, 2.0]))


class AuxiliaryMetricTests(unittest.TestCase):
    def test_threshold_brier_is_inclusive_at_exactly_four_percent(self) -> None:
        # Two of three draws satisfy >= 4.0 and the actual is exactly 4.0.
        self.assertAlmostEqual(threshold_brier([3.9, 4.0, 4.1], 4.0), 1.0 / 9.0)
        self.assertAlmostEqual(threshold_brier_from_probability(2.0 / 3.0, 4.0), 1.0 / 9.0)

    def test_threshold_probability_rejects_missing_or_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            threshold_brier_from_probability(None, 4.0)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            threshold_brier_from_probability(1.1, 4.0)

    def test_point_mae_requires_explicit_central_forecast(self) -> None:
        self.assertAlmostEqual(point_mae([20.0, 5.5], [19.0, 4.0]), 1.25)
        with self.assertRaises(ValueError):
            point_mae([20.0], [19.0, 4.0])

    def test_interval_coverage_and_width_are_explicit(self) -> None:
        scored = central_interval_metrics([0.0, 1.0, 2.0, 3.0], 2.0, levels=(0.5,))
        self.assertEqual(scored["0.5"]["lower"], 0.75)
        self.assertEqual(scored["0.5"]["upper"], 2.25)
        self.assertTrue(scored["0.5"]["covered"])
        self.assertEqual(scored["0.5"]["width"], 1.5)

    def test_wis_uses_only_reported_quantiles(self) -> None:
        quantiles = {
            0.025: 0.0,
            0.05: 0.0,
            0.10: 0.0,
            0.25: 0.0,
            0.50: 1.0,
            0.75: 2.0,
            0.90: 2.0,
            0.95: 2.0,
            0.975: 2.0,
        }
        # Default WIS uses the pre-registered 50/80/90/95% interval set.
        self.assertAlmostEqual(weighted_interval_score(quantiles, 1.0), 0.85 / 4.5)
        with self.assertRaisesRegex(ValueError, "missing required probability"):
            weighted_interval_score({0.5: 1.0, 0.1: 0.0, 0.9: 2.0}, 1.0)

    def test_wis_compatibility_requires_same_explicit_levels_for_all_parties(self) -> None:
        first = _quantile_map()
        second = _quantile_map(0.2)
        self.assertTrue(compatible_quantile_forecasts(first, second))
        second["L"] = {0.50: second["L"][0.50]}
        self.assertFalse(compatible_quantile_forecasts(first, second))

    def test_wis_fallback_scores_the_common_intersection_only(self) -> None:
        first = _quantile_map()
        second = _quantile_map(0.2)
        # Keep a median and 90% endpoints in both systems. The first system
        # has only 90%, while the second also has 50%; 50% is not common.
        for party in PRIMARY_PARTY_ORDER:
            first[party] = {key: value for key, value in first[party].items() if key in {0.05, 0.50, 0.95}}
            second[party] = {key: value for key, value in second[party].items() if key in {0.05, 0.25, 0.50, 0.75, 0.95}}
        result = score_forecast_pair(
            np.zeros(8),
            election_simulator_quantiles=first,
            botten_ada_quantiles=second,
        )
        self.assertEqual(result["primary_tier"], PROBABILISTIC_TIER_WIS)
        self.assertEqual(result["wis_common_interval_levels"], [0.9])
        self.assertEqual(set(result["election_simulator"]["quantiles"]["per_party"]["L"]["central_intervals"]), {"0.9"})

    def test_fallback_selector_never_promotes_unverified_draws(self) -> None:
        self.assertEqual(
            select_primary_scoring_tier(
                election_simulator_draws_verified=True,
                botten_ada_draws_verified=False,
            ),
            PROBABILISTIC_TIER_POINT_MAE,
        )
        self.assertEqual(
            select_primary_scoring_tier(
                election_simulator_draws_verified=False,
                botten_ada_draws_verified=False,
                election_simulator_quantiles=_quantile_map(),
                botten_ada_quantiles=_quantile_map(0.2),
            ),
            PROBABILISTIC_TIER_WIS,
        )


class VoteContractTests(unittest.TestCase):
    def _actual(self) -> np.ndarray:
        return np.array([20.0, 5.0, 7.0, 6.0, 30.0, 8.0, 4.0, 20.0])

    def _draws(self, n: int, offset: float = 0.0) -> np.ndarray:
        actual = self._actual()
        rows = [actual + offset + np.arange(actual.size, dtype=float) * 0.01 + i * 0.1 for i in range(n)]
        return np.asarray(rows)

    def test_eight_party_contract_does_not_renormalize_shares(self) -> None:
        actual = self._actual()
        draws = self._draws(3)
        # These rows intentionally do not sum to 100.  The scoring contract is
        # over official national valid-vote percentage points and must not
        # manufacture an eight-party simplex by renormalising them.
        self.assertNotAlmostEqual(float(np.sum(draws[0])), 100.0)
        score = score_vote_ensemble(
            draws,
            actual,
            central_forecast=actual,
            energy_pair_sample_size=200,
            energy_random_seed=3,
        )
        self.assertEqual(score["party_order"], list(PRIMARY_PARTY_ORDER))
        self.assertEqual(score["sample_count"], 3)
        self.assertEqual(score["point_forecast"]["values"], list(actual))
        self.assertEqual(score["point_forecast"]["mean_mae"], 0.0)
        threshold_score = score_vote_ensemble(
            draws,
            actual,
            threshold_parties=("L", "C", "KD", "MP"),
            energy_pair_sample_size=32,
        )
        self.assertEqual(threshold_score["threshold_4pct"]["parties"], ["L", "C", "KD", "MP"])

    def test_unequal_draw_counts_are_allowed_in_primary_pair(self) -> None:
        actual = self._actual()
        result = score_forecast_pair(
            actual,
            election_simulator_draws=self._draws(3),
            botten_ada_draws=self._draws(5, offset=0.2),
            election_simulator_draws_verified=True,
            botten_ada_draws_verified=True,
            election_simulator_central_forecast=actual,
            botten_ada_central_forecast=actual,
            energy_pair_sample_size=200,
            election_simulator_energy_seed=5,
            botten_ada_energy_seed=6,
        )
        self.assertEqual(result["primary_tier"], PROBABILISTIC_TIER_FAIR_DRAWS)
        self.assertEqual(result["election_simulator"]["sample_count"], 3)
        self.assertEqual(result["botten_ada"]["sample_count"], 5)

    def test_verified_draw_tier_requires_both_draw_arrays(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires draw arrays"):
            score_forecast_pair(
                self._actual(),
                election_simulator_draws=self._draws(3),
                election_simulator_draws_verified=True,
                botten_ada_draws_verified=True,
            )

    def test_unverified_draws_are_not_scored_as_fair_draws(self) -> None:
        actual = self._actual()
        result = score_forecast_pair(
            actual,
            election_simulator_draws=self._draws(3),
            botten_ada_draws=self._draws(5),
            election_simulator_draws_verified=False,
            botten_ada_draws_verified=False,
            election_simulator_central_forecast=actual,
            botten_ada_central_forecast=actual,
        )
        self.assertEqual(result["primary_tier"], PROBABILISTIC_TIER_POINT_MAE)
        self.assertEqual(result["status"], "SCORABLE")
        self.assertNotIn("fair_crps", result["election_simulator"])

    def test_missing_point_forecast_is_not_zeroed(self) -> None:
        result = score_forecast_pair(
            self._actual(),
            election_simulator_central_forecast=self._actual(),
            botten_ada_central_forecast=None,
        )
        self.assertEqual(result["primary_tier"], PROBABILISTIC_TIER_POINT_MAE)
        self.assertEqual(result["status"], "UNAVAILABLE_NO_COMMON_POINT_FORECASTS")
        self.assertIsNone(result["botten_ada"]["point_forecast"]["mean_mae"])

    def test_compatible_published_quantiles_use_wis_without_draw_inference(self) -> None:
        actual = self._actual()
        result = score_forecast_pair(
            actual,
            election_simulator_quantiles=_quantile_map(),
            botten_ada_quantiles=_quantile_map(0.2),
        )
        self.assertEqual(result["primary_tier"], PROBABILISTIC_TIER_WIS)
        self.assertIn("quantiles", result["election_simulator"])
        self.assertNotIn("fair_crps", result["election_simulator"])

    def test_bad_party_order_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires party order"):
            score_vote_ensemble(
                self._draws(3),
                self._actual(),
                party_order=tuple(reversed(PRIMARY_PARTY_ORDER)),
                energy_pair_sample_size=10,
            )


if __name__ == "__main__":
    unittest.main()
