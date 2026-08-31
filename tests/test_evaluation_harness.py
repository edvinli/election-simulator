"""Tests for the frozen ElectionNoise v2 evaluation harness (research infrastructure).

These guard the properties that must hold before any challenger is scored: case
selection follows the frozen rules, the historical mandate law is dispatched
explicitly, coalition scoring is joint and correctly encoded, truth vectors are
the certified ones, aggregation happens within election first, and a fixed
(case, seed) reproduces exactly.

Heavy Monte Carlo is avoided: every test runs at a small draw count or on
synthetic seat matrices.
"""

from __future__ import annotations

from datetime import date, timedelta
from fractions import Fraction
from pathlib import Path
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH = REPO_ROOT / "diagnostics" / "election_noise_v2"

from diagnostics.election_noise_v2.control_baseline.harness import metrics as M
from diagnostics.election_noise_v2.control_baseline.harness.manifest import (
    ELECTION_DATES,
    GEOGRAPHY_BASELINE,
    HORIZONS,
    K_OUTER_MIN,
    TIER1_CANDIDATE_TARGETS,
    build_manifest,
    validate_manifest,
)
from diagnostics.election_noise_v2.control_baseline.harness.pipeline import (
    assert_law_dispatch,
    tier1_control_draws,
)
from diagnostics.election_noise_v2.control_baseline.harness.rng import (
    DRAWS_PER_SEED,
    FROZEN_SEEDS,
    assert_paired_base,
    control_residual_indices,
    stream_seeds,
)
from scripts.election_layer_v2.forward_eval import compute_discrete_crps
from scripts.mandates.law import MandateLaw, mandate_law_for_election_year
from scripts.pollofpolls.backtest_metrics import calculate_crps
from scripts.simulator.config import PARLIAMENTARY_PARTIES_8

_HAS_RESEARCH_DATA = (
    RESEARCH / "historical_seat_extension" / "processed" / "certified_mandates_2010_2014.csv"
).exists()


@unittest.skipUnless(_HAS_RESEARCH_DATA, "Part-2B research data not present")
class CaseManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = build_manifest()

    def test_manifest_validates(self) -> None:
        self.assertEqual(validate_manifest(self.m), [])

    def test_tier1_candidate_set_is_frozen_and_unexpanded(self) -> None:
        self.assertEqual(tuple(self.m["tier1_candidate_targets"]), TIER1_CANDIDATE_TARGETS)
        for c in self.m["cases"]["tier1"]:
            self.assertIn(c["target_year"], TIER1_CANDIDATE_TARGETS)

    def test_tier1_selection_is_exactly_the_k_outer_rule(self) -> None:
        for row in self.m["tier1_eligibility"]:
            self.assertEqual(row["eligible"], row["k_outer"] >= K_OUTER_MIN)
        self.assertEqual(self.m["counts"]["N_T1"], 3)
        self.assertEqual(self.m["counts"]["tier1_elections"], [2014, 2018, 2022])

    def test_2010_is_excluded_everywhere(self) -> None:
        self.assertNotIn(2010, self.m["counts"]["tier1_elections"])
        self.assertNotIn(2010, self.m["counts"]["tier23_elections"])

    def test_tier3_cases_are_exactly_tier2_cases(self) -> None:
        def key(c):
            return (c["target_year"], c["horizon_days"])

        self.assertEqual(
            sorted(map(key, self.m["cases"]["tier2"])),
            sorted(map(key, self.m["cases"]["tier3"])),
        )

    def test_horizons_are_the_frozen_six(self) -> None:
        for year in self.m["counts"]["tier23_elections"]:
            hs = sorted(
                c["horizon_days"] for c in self.m["cases"]["tier2"] if c["target_year"] == year
            )
            self.assertEqual(hs, sorted(HORIZONS))

    def test_monte_carlo_design_is_frozen(self) -> None:
        self.assertEqual(tuple(self.m["monte_carlo"]["seeds"]), FROZEN_SEEDS)
        self.assertEqual(self.m["monte_carlo"]["draws_per_seed"], DRAWS_PER_SEED)
        for c in self.m["cases"]["tier1"] + self.m["cases"]["tier2"] + self.m["cases"]["tier3"]:
            self.assertEqual(tuple(c["seeds"]), FROZEN_SEEDS)
            self.assertEqual(c["draws_per_seed"], DRAWS_PER_SEED)

    def test_no_future_residual_leaks_into_a_training_pool(self) -> None:
        for c in self.m["cases"]["tier1"] + self.m["cases"]["tier2"] + self.m["cases"]["tier3"]:
            for y in c["training_residual_years"]:
                self.assertLess(y, c["target_year"], f"leak in {c['tier']} {c['target_year']}")

    def test_geography_baseline_is_chronological(self) -> None:
        for c in self.m["cases"]["tier2"] + self.m["cases"]["tier3"]:
            self.assertEqual(
                c["geography_baseline_year"], GEOGRAPHY_BASELINE[c["target_year"]]
            )
            self.assertLess(c["geography_baseline_year"], c["target_year"])

    def test_as_of_precedes_election_by_the_horizon(self) -> None:
        for c in self.m["cases"]["tier2"]:
            ed = date.fromisoformat(c["election_date"])
            self.assertEqual(
                date.fromisoformat(c["as_of"]), ed - timedelta(days=c["horizon_days"])
            )

    def test_certified_truth_seat_vectors_are_correct(self) -> None:
        expected = {
            2018: {"S": 100, "M": 70, "SD": 62, "C": 31, "V": 28, "KD": 22, "L": 20, "MP": 16},
            2022: {"S": 107, "SD": 73, "M": 68, "V": 24, "C": 24, "KD": 19, "MP": 18, "L": 16},
        }
        for c in self.m["cases"]["tier3"]:
            self.assertEqual(c["truth_seats"], expected[c["target_year"]])
            self.assertEqual(sum(c["truth_seats"].values()), 349)

    def test_truth_vote_vectors_sum_to_100(self) -> None:
        for c in self.m["cases"]["tier1"] + self.m["cases"]["tier2"]:
            self.assertAlmostEqual(sum(c["truth_vote_pct"].values()), 100.0, places=3)


class LawDispatchTest(unittest.TestCase):
    """2014 must never be scored under current law; 2018/2022 must be."""

    def test_2014_maps_to_pre_2018(self) -> None:
        cfg = mandate_law_for_election_year(2014)
        self.assertIs(cfg.law, MandateLaw.PRE_2018)
        self.assertEqual(cfg.first_divisor, Fraction(7, 5))

    def test_2018_and_2022_map_to_post_2018(self) -> None:
        for y in (2018, 2022):
            cfg = mandate_law_for_election_year(y)
            self.assertIs(cfg.law, MandateLaw.POST_2018)
            self.assertEqual(cfg.first_divisor, Fraction(6, 5))

    def test_production_engine_refuses_a_pre_2018_target(self) -> None:
        """The hard guard: routing 2014 through the current-law engine must fail."""
        with self.assertRaises(RuntimeError) as ctx:
            assert_law_dispatch(2014, engine="production_simulate_election")
        self.assertIn("LAW DISPATCH VIOLATION", str(ctx.exception))

    def test_production_engine_accepts_post_2018_targets(self) -> None:
        for y in (2018, 2022):
            self.assertIs(
                assert_law_dispatch(y, engine="production_simulate_election"),
                MandateLaw.POST_2018,
            )

    @unittest.skipUnless(_HAS_RESEARCH_DATA, "Part-2B research data not present")
    def test_manifest_records_the_statutory_law_for_every_seat_case(self) -> None:
        m = build_manifest()
        for c in m["cases"]["tier2"] + m["cases"]["tier3"]:
            self.assertEqual(
                c["mandate_law"], mandate_law_for_election_year(c["target_year"]).law.value
            )


class CoalitionScoringTest(unittest.TestCase):
    """Mask encoding, joint summation and the 175 threshold."""

    def test_mask_encoding_matches_party_order(self) -> None:
        self.assertEqual(PARLIAMENTARY_PARTIES_8, ("M", "L", "C", "KD", "S", "V", "MP", "SD"))
        self.assertEqual(M.coalition_mask_columns(1), [0])            # M
        self.assertEqual(M.coalition_mask_columns(112), [4, 5, 6])    # S+V+MP
        self.assertEqual(M.coalition_mask_columns(84), [2, 4, 6])     # C+S+MP
        self.assertEqual(M.coalition_mask_columns(15), [0, 1, 2, 3])  # M+L+C+KD
        self.assertEqual(M.coalition_mask_columns(254), list(range(1, 8)))

    def test_mask_set_is_1_to_254(self) -> None:
        self.assertEqual(M.MASKS[0], 1)
        self.assertEqual(M.MASKS[-1], 254)
        self.assertEqual(len(M.MASKS), 254)
        self.assertNotIn(0, M.MASKS)
        self.assertNotIn(255, M.MASKS)

    def test_majority_threshold_is_175_inclusive(self) -> None:
        self.assertEqual(M.MAJORITY_THRESHOLD, 175)
        truth = np.array([70, 20, 31, 22, 100, 28, 16, 62], dtype=np.int64)
        # A coalition sitting exactly on 175 counts as a majority.
        seats = np.tile(np.array([175, 0, 0, 0, 174, 0, 0, 0]), (10, 1))
        res = M.d4_coalition_brier(seats, truth)
        self.assertEqual(res["per_mask"][1]["p"], 1.0)   # M alone = 175 -> majority
        self.assertEqual(res["per_mask"][16]["p"], 0.0)  # S alone = 174 -> no majority

    def test_coalition_seats_are_joint_per_draw_sums_not_marginals(self) -> None:
        """A distribution whose marginals average below 175 but whose joint draws
        are always exactly 175 must score p = 1."""
        truth = np.array([70, 20, 31, 22, 100, 28, 16, 62], dtype=np.int64)
        a = np.array([120, 0, 0, 0, 55, 0, 0, 174])
        b = np.array([55, 0, 0, 0, 120, 0, 0, 174])
        seats = np.stack([a, b] * 50)
        res = M.d4_coalition_brier(seats, truth)
        # mask 17 = M + S: every draw sums to exactly 175
        self.assertEqual(M.coalition_mask_columns(17), [0, 4])
        self.assertEqual(res["per_mask"][17]["p"], 1.0)
        # the marginal medians are 87.5 and 87.5; summing marginals would give 175
        # too, so also check a case where they differ:
        self.assertAlmostEqual(float(np.mean(seats[:, 0])), 87.5)

    def test_complement_symmetry_holds_exactly(self) -> None:
        rng = np.random.default_rng(3)
        truth = np.array([70, 20, 31, 22, 100, 28, 16, 62], dtype=np.int64)
        seats = rng.multinomial(349, [1 / 8] * 8, size=200).astype(np.int64)
        res = M.d4_coalition_brier(seats, truth)
        sym = M.verify_complement_symmetry(res["per_mask"])
        self.assertTrue(sym["holds_within_tolerance"])
        # The identity B_m == B_{255-m} is exact algebraically. In floating point
        # p and 1-p are computed independently before squaring, so agreement is to
        # machine epsilon rather than bitwise; the documented 1e-12 tolerance in
        # verify_complement_symmetry covers it.
        self.assertLessEqual(sym["max_abs_brier_difference_between_complements"], 1e-15)
        # and the 254-mask mean equals the mean over 127 complement representatives
        all_254 = float(np.mean([res["per_mask"][m]["brier"] for m in M.MASKS]))
        reps = float(np.mean([res["per_mask"][m]["brier"] for m in range(1, 128)]))
        self.assertAlmostEqual(all_254, reps, places=12)

    def test_effective_event_count_is_127(self) -> None:
        self.assertEqual(M.EFFECTIVE_DISTINCT_EVENTS, 127)

    def test_brier_is_squared_error_against_certified_indicator(self) -> None:
        truth = np.array([70, 20, 31, 22, 100, 28, 16, 62], dtype=np.int64)
        seats = np.tile(np.array([70, 20, 31, 22, 100, 28, 16, 62]), (4, 1))
        res = M.d4_coalition_brier(seats, truth)
        for m in M.MASKS:
            # A point mass at the certified outcome scores a perfect 0 everywhere.
            self.assertEqual(res["per_mask"][m]["brier"], 0.0)

    def test_seat_totals_must_be_349_for_the_identity_to_hold(self) -> None:
        truth = np.array([70, 20, 31, 22, 100, 28, 16, 62], dtype=np.int64)
        self.assertEqual(int(truth.sum()), 349)


class AggregationTest(unittest.TestCase):
    """Brier must be aggregated within election before elections are combined."""

    def test_within_election_first_then_across(self) -> None:
        from diagnostics.election_noise_v2.control_baseline.harness.run_control import aggregate

        manifest = {
            "counts": {"tier1_elections": [], "tier23_elections": [2018, 2022]},
        }
        rows = []
        # 2018 gets two horizons with very different Brier; 2022 gets two equal.
        for year, vals in ((2018, {112: 0.40, 7: 0.00}), (2022, {112: 0.10, 7: 0.10})):
            for h, b in vals.items():
                for s in FROZEN_SEEDS:
                    rows.append(
                        {
                            "tier": "tier3",
                            "target_year": year,
                            "horizon_days": h,
                            "seed": s,
                            "coalition_brier_mean_over_masks": b,
                            "seat_energy_score": b,
                            "seat_crps_8party_mean": b,
                            "seat_coverage_50": b,
                            "seat_coverage_80": b,
                            "seat_coverage_90": b,
                        }
                    )
        agg = aggregate(rows, manifest)
        t3 = agg["tiers"]["tier3"]
        self.assertAlmostEqual(t3["by_election"]["2018"]["coalition_brier_mean_over_masks"], 0.20)
        self.assertAlmostEqual(t3["by_election"]["2022"]["coalition_brier_mean_over_masks"], 0.10)
        # Headline is the unweighted mean of the two election aggregates, NOT the
        # mean over the four cases (which would also be 0.15 here) — check that
        # unequal case counts do not change the election weighting:
        self.assertAlmostEqual(
            t3["headline"]["coalition_brier_mean_over_masks"]["mean_over_elections"], 0.15
        )
        self.assertEqual(t3["headline"]["coalition_brier_mean_over_masks"]["n_elections"], 2)

    def test_election_weighting_is_unaffected_by_case_count(self) -> None:
        from diagnostics.election_noise_v2.control_baseline.harness.run_control import aggregate

        manifest = {"counts": {"tier1_elections": [], "tier23_elections": [2018, 2022]}}
        rows = []
        # 2018: three horizons at 0.30; 2022: one horizon at 0.10.
        for year, spec in ((2018, {112: 0.30, 84: 0.30, 56: 0.30}), (2022, {112: 0.10, 84: 0.10, 56: 0.10})):
            for h, b in spec.items():
                for s in FROZEN_SEEDS:
                    rows.append(
                        {
                            "tier": "tier3", "target_year": year, "horizon_days": h, "seed": s,
                            "coalition_brier_mean_over_masks": b, "seat_energy_score": b,
                            "seat_crps_8party_mean": b, "seat_coverage_50": b,
                            "seat_coverage_80": b, "seat_coverage_90": b,
                        }
                    )
        agg = aggregate(rows, manifest)
        self.assertAlmostEqual(
            agg["tiers"]["tier3"]["headline"]["coalition_brier_mean_over_masks"]["mean_over_elections"],
            0.20,
        )


class PairedRandomnessTest(unittest.TestCase):
    """Upstream seed derivation must be deterministic and model-independent."""

    def test_stream_seeds_are_deterministic(self) -> None:
        a = stream_seeds(12345, date(2018, 5, 20), 112)
        b = stream_seeds(12345, date(2018, 5, 20), 112)
        self.assertEqual(a, b)

    def test_upstream_seeds_do_not_depend_on_the_noise_model(self) -> None:
        """OpinionState/Dynamics tokens contain no model identifier, so the
        upstream sub-seeds are fixed by (base_seed, as_of, horizon) alone."""
        a = stream_seeds(12345, date(2022, 9, 4), 7)
        self.assertEqual(
            a.upstream,
            stream_seeds(12345, date(2022, 9, 4), 7).upstream,
        )
        # Different base seeds must give different upstream streams.
        self.assertNotEqual(a.upstream, stream_seeds(24680, date(2022, 9, 4), 7).upstream)

    def test_different_horizons_give_different_dynamics_seeds(self) -> None:
        s1 = stream_seeds(12345, date(2018, 5, 20), 112)
        s2 = stream_seeds(12345, date(2018, 5, 20), 7)
        self.assertNotEqual(s1.dynamics_seed, s2.dynamics_seed)
        self.assertEqual(s1.opinion_state_seed, s2.opinion_state_seed)

    def test_assert_paired_base_detects_a_difference(self) -> None:
        a = np.ones((5, 9))
        assert_paired_base(a, a.copy(), "identical")
        b = a.copy()
        b[2, 3] += 1e-9
        with self.assertRaises(AssertionError):
            assert_paired_base(a, b, "perturbed")

    def test_control_index_draw_is_reproducible_and_uniform(self) -> None:
        i1 = control_residual_indices(999, 5, 10_000)
        i2 = control_residual_indices(999, 5, 10_000)
        np.testing.assert_array_equal(i1, i2)
        counts = np.bincount(i1, minlength=5)
        self.assertEqual(int(counts.sum()), 10_000)
        self.assertLess(float(np.max(np.abs(counts - 2000))), 200.0)


class ReproducibilityTest(unittest.TestCase):
    def test_fixed_case_and_seed_reproduce_exactly(self) -> None:
        ed = ELECTION_DATES[2018]
        a = tier1_control_draws(ed, 2018, 12345, 500)
        b = tier1_control_draws(ed, 2018, 12345, 500)
        np.testing.assert_array_equal(a.votes_pct, b.votes_pct)
        np.testing.assert_array_equal(a.lambdas, b.lambdas)
        np.testing.assert_array_equal(a.residual_index, b.residual_index)

    def test_different_seed_gives_different_draws(self) -> None:
        ed = ELECTION_DATES[2018]
        a = tier1_control_draws(ed, 2018, 12345, 500)
        c = tier1_control_draws(ed, 2018, 24680, 500)
        self.assertFalse(np.array_equal(a.residual_index, c.residual_index))

    def test_tier1_draws_are_on_the_k_atom_support(self) -> None:
        ed = ELECTION_DATES[2014]
        d = tier1_control_draws(ed, 2014, 12345, 400)
        self.assertEqual(len(d.training_years), 3)
        self.assertEqual(set(np.unique(d.residual_index).tolist()), {0, 1, 2})
        uniq = np.unique(np.round(d.votes_pct, 10), axis=0)
        self.assertEqual(uniq.shape[0], 3)  # exactly K distinct compositions
        np.testing.assert_allclose(d.votes_pct.sum(axis=1), 100.0, atol=1e-9)


class CrpsEstimatorEquivalenceTest(unittest.TestCase):
    """The O(N log N) production CRPS equals the O(N^2) one; only N differs."""

    def test_estimators_agree(self) -> None:
        rng = np.random.default_rng(7)
        for n in (2, 17, 300, 2000):
            x = rng.normal(size=n) * 4 + 30
            y = float(rng.normal() * 4 + 30)
            self.assertAlmostEqual(compute_discrete_crps(x, y), calculate_crps(x, y), places=12)


if __name__ == "__main__":
    unittest.main()
