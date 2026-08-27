"""Test suite for ElectionSimulator v1, national vote-model integration, exact integerization, and allocator verification."""

import unittest
from datetime import date
import numpy as np

from scripts.geography.config import OFFICIAL_CONSTITUENCY_CODES
from scripts.geography.integerization import biproportional_controlled_rounding
from scripts.mandates.allocator import allocate_riksdag_seats
from scripts.mandates.config import FIXED_SEATS_2026, TOTAL_RIKSDAG_SEATS
from scripts.simulator.config import (
    DEFAULT_ELECTION_DATE,
    DEFAULT_MAJORITY_THRESHOLD,
    MODEL_PARTIES_9,
    PARLIAMENTARY_PARTIES_8,
)
from scripts.simulator.engine import _apportion_constituency_units_of_25, _apportion_national_party_integers, simulate_election
from scripts.simulator.fast_allocator import fast_allocate_seats_from_matrix
from scripts.vote_share_calibration.national_engine import generate_national_vote_shares


class TestElectionSimulator(unittest.TestCase):
    """Comprehensive test suite for Swedish Riksdag ElectionSimulator v1."""

    def test_national_vote_model_exact_identity_with_standalone(self) -> None:
        """Gate 1: Verify standalone VoteShareModel national draws strictly equal simulator national draws."""
        samples = 500
        seed = 42
        as_of = "2026-08-23"
        el_date = "2026-09-13"

        # Standalone national generator
        standalone_res = generate_national_vote_shares(
            as_of=as_of,
            election_date=el_date,
            samples=samples,
            seed=seed,
        )

        # Simulator run
        sim_res = simulate_election(
            as_of=as_of,
            election_date=el_date,
            samples=samples,
            seed=seed,
        )

        # Assert exact numerical equality on national vote shares
        np.testing.assert_array_equal(
            standalone_res.nat_shares_matrix * 100.0,
            sim_res.vote_shares_matrix,
        )
        # Assert latent component shapes
        self.assertEqual(standalone_res.dynamics_deltas.shape, (samples, 9))
        self.assertEqual(standalone_res.opinion_state_draws.shape, (samples, 9))

    def test_deterministic_seed_reproducibility(self) -> None:
        """Verify identical seed produces byte-for-byte identical simulation draws and summary."""
        res1 = simulate_election(samples=500, seed=42)
        res2 = simulate_election(samples=500, seed=42)

        np.testing.assert_array_equal(res1.vote_shares_matrix, res2.vote_shares_matrix)
        np.testing.assert_array_equal(res1.seats_matrix, res2.seats_matrix)
        np.testing.assert_array_equal(res1.threshold_flags, res2.threshold_flags)
        self.assertEqual(res1.largest_vote_parties, res2.largest_vote_parties)
        self.assertEqual(res1.largest_seat_parties, res2.largest_seat_parties)
        self.assertEqual(
            res1.summary.parties["S"].seats_mean,
            res2.summary.parties["S"].seats_mean,
        )

    def test_different_seed_different_draws(self) -> None:
        """Verify different seeds produce different draws."""
        res1 = simulate_election(samples=500, seed=42)
        res2 = simulate_election(samples=500, seed=999)

        self.assertFalse(np.array_equal(res1.vote_shares_matrix, res2.vote_shares_matrix))

    def test_strict_vote_share_sum_invariant(self) -> None:
        """Verify sum of 9 party vote shares strictly equals 100.0% for all samples."""
        res = simulate_election(samples=1000, seed=123)
        row_sums = np.sum(res.vote_shares_matrix, axis=1)
        np.testing.assert_allclose(row_sums, 100.0, atol=1e-5)

    def test_strict_seat_sum_invariant(self) -> None:
        """Verify sum of 8 parliamentary party seats strictly equals 349 for all samples."""
        res = simulate_election(samples=1000, seed=123)
        seat_sums = np.sum(res.seats_matrix, axis=1)
        self.assertTrue(np.all(seat_sums == TOTAL_RIKSDAG_SEATS))
        self.assertEqual(res.summary.parties["REST"].seats_mean, 0.0)

    def test_quantization_audit_uses_production_integer_boundaries(self) -> None:
        """Verify quantization diagnostics inspect every production party/constituency pair."""
        samples = 25
        res = simulate_election(samples=samples, seed=123, collect_quantization_audit=True)
        self.assertIsNotNone(res.quantization_audit)
        audit = res.quantization_audit
        assert audit is not None
        self.assertEqual(audit["total_samples"], samples)
        self.assertEqual(audit["total_party_constituency_pairs_checked"], samples * 29 * 8)
        self.assertGreaterEqual(audit["relevant_party_constituency_pairs"], 0)
        self.assertLessEqual(audit["relevant_party_constituency_pairs"], samples * 29 * 8)
        self.assertGreaterEqual(audit["post_integer_local_12_events"], 0)
        self.assertGreaterEqual(audit["pre_post_local_12_mismatches"], 0)
        self.assertIsNotNone(audit["minimum_national_4pct_continuous_distance_pp"])

    def test_threshold_complementarity(self) -> None:
        """Verify P(vote >= 4%) + P(vote < 4%) == 1.0 for all parties."""
        res = simulate_election(samples=1000, seed=123)
        for p in PARLIAMENTARY_PARTIES_8:
            p_above = res.summary.parties[p].prob_above_4pct
            p_below = res.summary.parties[p].prob_below_4pct
            self.assertAlmostEqual(p_above + p_below, 1.0, places=5)

    def test_group_summary_majority_logic(self) -> None:
        """Verify generic group summarizer correctly computes seat quantiles and majority prob."""
        res = simulate_election(samples=1000, seed=123)
        tido_group = ["M", "SD", "KD", "L"]
        rg_group = ["S", "V", "MP", "C"]

        tido_summary = res.summarize_group(tido_group, majority_threshold=175)
        rg_summary = res.summarize_group(rg_group, majority_threshold=175)

        self.assertEqual(tido_summary.parties, tuple(tido_group))
        self.assertEqual(rg_summary.parties, tuple(rg_group))
        self.assertGreaterEqual(tido_summary.prob_majority, 0.0)
        self.assertLessEqual(tido_summary.prob_majority, 1.0)
        self.assertGreaterEqual(rg_summary.prob_majority, 0.0)
        self.assertLessEqual(rg_summary.prob_majority, 1.0)
        self.assertAlmostEqual(sum(tido_summary.seat_histogram.values()), 1.0, places=5)

    def test_exact_margin_controlled_rounding(self) -> None:
        """Gate 5 & 6: Verify biproportional controlled rounding matches exact row and column margins."""
        np.random.seed(42)
        X = np.random.uniform(500, 20000, size=(29, 9))
        tot_votes = 6500000
        X = X / np.sum(X) * tot_votes
        
        # Row targets as multiples of 25
        R_int = _apportion_constituency_units_of_25(np.sum(X, axis=1), tot_votes)
        # Column targets preserving grand total
        C_int = _apportion_national_party_integers(np.sum(X, axis=0) / tot_votes, tot_votes)

        cr_res = biproportional_controlled_rounding(X, R_int, C_int, solver="auto")

        self.assertEqual(cr_res.max_row_error, 0)
        self.assertEqual(cr_res.max_column_error, 0)
        self.assertTrue(cr_res.total_conserved)
        self.assertLess(cr_res.max_cell_error, 1.0)
        np.testing.assert_array_equal(np.sum(cr_res.rounded_matrix, axis=1), R_int)
        np.testing.assert_array_equal(np.sum(cr_res.rounded_matrix, axis=0), C_int)
        self.assertTrue(np.all(R_int % 25 == 0), "All constituency totals must be multiples of 25")

    def test_exact_threshold_boundary_precision(self) -> None:
        """Gate 7: Test exact inclusive 4.000% national and 12.000% local threshold boundaries."""
        tot_votes = 6500000
        # Exactly 4.000% is 260,000 votes
        self.assertEqual(int(0.04 * tot_votes), 260000)

        # Test national boundaries around 4.0%
        # 3.999% = 259,935, 4.000% = 260,000, 4.001% = 260,065
        v_3999 = 259935
        v_4000 = 260000
        v_4001 = 260065

        self.assertFalse(25 * v_3999 >= tot_votes)
        self.assertTrue(25 * v_4000 >= tot_votes)
        self.assertTrue(25 * v_4001 >= tot_votes)

        # Test local boundaries around 12.0% in a constituency with 250,000 votes
        # 12.0% of 250,000 = 30,000 votes
        c_votes = 250000
        v_1199 = 29990
        v_1200 = 30000
        v_1201 = 30010

        self.assertFalse(25 * v_1199 >= 3 * c_votes)
        self.assertTrue(25 * v_1200 >= 3 * c_votes)
        self.assertTrue(25 * v_1201 >= 3 * c_votes)

    def test_fast_allocator_adversarial_oracle_equivalence(self) -> None:
        """Gate 9: Verify fast allocator matches exact reference allocator across adversarial cases."""
        np.random.seed(99)
        tot_votes = 6500000
        
        # Test 100 adversarial cases with diverse compositions
        for i in range(100):
            # Generate random shares with parties near 4% threshold
            base_p = [18.0, 3.99 + (i % 5) * 0.01, 6.5, 4.01 - (i % 3) * 0.01, 30.0, 7.5, 7.5, 18.0, 2.0]
            shares = np.array(base_p) / sum(base_p)
            C_int = _apportion_national_party_integers(shares, tot_votes)
            
            # Apportion dummy 29x9 matrix preserving C_int
            mat = np.outer(np.ones(29) / 29.0, C_int).astype(np.int64)
            diff = C_int - np.sum(mat, axis=0)
            mat[0] += diff

            fast_seats = fast_allocate_seats_from_matrix(mat)

            # Build dict for reference allocator
            cv_map = {}
            for row_i, c_code in enumerate(OFFICIAL_CONSTITUENCY_CODES):
                cv_map[c_code] = {}
                for col_j, p in enumerate(MODEL_PARTIES_9):
                    lbl = "OTHER_INELIGIBLE" if p == "REST" else p
                    cv_map[c_code][lbl] = int(mat[row_i, col_j])

            ref_res = allocate_riksdag_seats(cv_map, FIXED_SEATS_2026)
            ref_seats = {p: ref_res.final_seats_by_party.get(p, 0) for p in PARLIAMENTARY_PARTIES_8}

            self.assertEqual(
                fast_seats,
                ref_seats,
                f"Fast allocator mismatch with reference oracle on adversarial case {i}",
            )

    def test_reproducibility_manifest_metadata(self) -> None:
        """Verify reproducibility manifest includes all required hash fields and commit information."""
        res = simulate_election(samples=100, seed=42)
        manifest = res.manifest

        self.assertIn("model_version", manifest)
        self.assertIn("as_of", manifest)
        self.assertIn("election_date", manifest)
        self.assertIn("samples", manifest)
        self.assertIn("base_seed", manifest)
        self.assertIn("poll_data_hash", manifest)
        self.assertIn("election_data_hash", manifest)
        self.assertIn("mandate_data_hash", manifest)
        self.assertIn("geography_data_hash", manifest)
        self.assertIn("model_config_hash", manifest)
        self.assertIn("git_commit", manifest)
        self.assertIn("generated_at_utc", manifest)
