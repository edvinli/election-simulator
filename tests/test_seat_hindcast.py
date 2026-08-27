"""Tests for SeatHindcast v1: metrics, leakage safety, baseline resolution, and invariants."""

import unittest
from datetime import date
from pathlib import Path
import numpy as np

from scripts.seat_hindcasts.config import EVALUATION_ELECTIONS, PARLIAMENTARY_PARTIES_8
from scripts.seat_hindcasts.metrics import (
    calculate_discrete_seat_crps,
    calculate_empirical_percentile,
    calculate_interval_coverage_and_width,
    calculate_multivariate_energy_score,
)
from scripts.seat_hindcasts.models import (
    evaluate_election_simulator_v1,
    evaluate_seat_point_baseline,
)
from scripts.vote_share_calibration.national_engine import generate_national_vote_shares


class TestSeatHindcast(unittest.TestCase):
    """Test suite for historical Riksdag seat hindcast metrics, pipeline, and invariance."""

    def test_discrete_crps_known_point_distribution(self) -> None:
        """Verify discrete CRPS for a degenerate (point) distribution F = delta_c."""
        # If all samples are 100, and actual is 100, CRPS must be 0.0
        samples_exact = np.full(1000, 100, dtype=int)
        crps_zero = calculate_discrete_seat_crps(samples_exact, 100)
        self.assertAlmostEqual(crps_zero, 0.0, places=4)

        # If all samples are 100, and actual is 105, CRPS must be |100 - 105| = 5.0
        crps_offset = calculate_discrete_seat_crps(samples_exact, 105)
        self.assertAlmostEqual(crps_offset, 5.0, places=4)

        # If all samples are 100, and actual is 95, CRPS must be |100 - 95| = 5.0
        crps_offset_low = calculate_discrete_seat_crps(samples_exact, 95)
        self.assertAlmostEqual(crps_offset_low, 5.0, places=4)

    def test_discrete_crps_two_point_mixture(self) -> None:
        """Verify discrete CRPS for 50/50 mixture of 10 and 20 when actual is 15."""
        # F = 0.5 * delta_10 + 0.5 * delta_20.
        # CRPS = E[|S - y|] - 0.5 * E[|S - S'|]
        # E[|S - 15|] = 0.5 * 5 + 0.5 * 5 = 5.0
        # E[|S - S'|] = 0.5 * 0.5 * |10 - 20| + 0.5 * 0.5 * |20 - 10| = 0.25 * 10 + 0.25 * 10 = 5.0
        # CRPS = 5.0 - 0.5 * 5.0 = 2.5
        samples_mix = np.array([10] * 500 + [20] * 500)
        crps = calculate_discrete_seat_crps(samples_mix, 15)
        self.assertAlmostEqual(crps, 2.5, places=2)

    def test_multivariate_energy_score_exact(self) -> None:
        """Verify multivariate Energy Score on degenerate vector distribution."""
        # If all sample vectors equal actual, Energy Score must be 0.0
        actual = np.array([70, 20, 31, 22, 100, 28, 16, 62], dtype=int)
        samples = np.tile(actual, (100, 1))
        es_zero = calculate_multivariate_energy_score(samples, actual)
        self.assertAlmostEqual(es_zero, 0.0, places=4)

        # Shifted samples by 3 in L2 norm
        shifted_actual = actual + 1
        # ||(actual+1) - actual||_2 = sqrt(8) = 2.8284
        es_shifted = calculate_multivariate_energy_score(samples, shifted_actual)
        self.assertAlmostEqual(es_shifted, np.sqrt(8), places=4)

    def test_coverage_and_percentiles(self) -> None:
        """Verify coverage, width, and mid-rank percentiles."""
        samples = np.arange(101, dtype=int)
        cov, width, q_low, q_high = calculate_interval_coverage_and_width(samples, actual=50, level=0.80)
        self.assertTrue(cov)
        self.assertEqual(q_low, 10)
        self.assertEqual(q_high, 90)
        self.assertEqual(width, 80)

        perc = calculate_empirical_percentile(samples, actual=50)
        self.assertAlmostEqual(perc, 50.0, places=1)

    def test_chronological_baseline_resolution_and_leakage_safety(self) -> None:
        """Verify that 2018 hindcast uses 2014 baseline and 2022 hindcast uses 2018 baseline."""
        self.assertEqual(EVALUATION_ELECTIONS["2018"]["geography_baseline_year"], 2014)
        self.assertEqual(EVALUATION_ELECTIONS["2022"]["geography_baseline_year"], 2018)

    def test_end_to_end_seat_hindcast_execution_and_invariants(self) -> None:
        """Run a test hindcast for 2022 at 28d horizon and verify all invariants."""
        elec_date = date(2022, 9, 11)
        as_of = date(2022, 8, 14)  # 28 days prior
        n_samples = 200
        seed = 12345

        # 1. Point baseline
        pt_seats = evaluate_seat_point_baseline(
            as_of=as_of,
            election_date=elec_date,
            baseline_year=2018,
        )
        self.assertEqual(sum(pt_seats.values()), 349)

        # 2. Simulator
        sim_res = evaluate_election_simulator_v1(
            as_of=as_of,
            election_date=elec_date,
            baseline_year=2018,
            samples=n_samples,
            seed=seed,
        )

        # Invariant 1: Total seats == 349 across all samples
        seat_totals = np.sum(sim_res.seats_matrix, axis=1)
        self.assertTrue(np.all(seat_totals == 349))

        # Invariant 2: Vote shares sum to 100.0%
        share_totals = np.sum(sim_res.vote_shares_matrix, axis=1)
        np.testing.assert_allclose(share_totals, 100.0, atol=1e-5)

        # Invariant 3: National vote draws identical to standalone generator
        standalone_res = generate_national_vote_shares(
            as_of=as_of,
            election_date=elec_date,
            samples=n_samples,
            seed=seed,
        )
        np.testing.assert_allclose(sim_res.vote_shares_matrix / 100.0, standalone_res.nat_shares_matrix, atol=1e-12)

    def test_strict_future_leakage_independence(self) -> None:
        """Verify that chronological hindcasts are 100% independent of future target-election electorate rows."""
        import tempfile
        import pandas as pd
        from scripts.geography.config import DEFAULT_PROCESSED_GEOGRAPHY_DIR

        elec_date = date(2018, 9, 9)
        as_of = date(2018, 8, 12)  # 28 days prior
        n_samples = 100
        seed = 42

        # 1. Baseline run with standard data
        res_baseline = evaluate_election_simulator_v1(
            as_of=as_of,
            election_date=elec_date,
            baseline_year=2014,
            samples=n_samples,
            seed=seed,
            geography_mode="chronological",
        )

        # 2. Create temp processed geography directory where 2018 & 2022 target electorate data is corrupted
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Copy baseline party votes
            pv_df = pd.read_csv(DEFAULT_PROCESSED_GEOGRAPHY_DIR / "constituency_party_votes_2014_2022.csv")
            pv_df.to_csv(tmp_path / "constituency_party_votes_2014_2022.csv", index=False)

            # Corrupt target year electorates (2018 and 2022) with arbitrary wild numbers
            el_df = pd.read_csv(DEFAULT_PROCESSED_GEOGRAPHY_DIR / "constituency_electorates_2014_2026.csv")
            el_df.loc[el_df["election_year"] >= 2018, "valid_votes"] = 999_999_999
            el_df.loc[el_df["election_year"] >= 2018, "eligible_voters"] = 999_999_999
            el_df.to_csv(tmp_path / "constituency_electorates_2014_2026.csv", index=False)

            # Re-run chronological simulator pointing to corrupted future data
            from scripts.simulator.engine import simulate_election
            res_corrupted = simulate_election(
                as_of=as_of,
                election_date=elec_date,
                samples=n_samples,
                seed=seed,
                baseline_year=2014,
                geography_mode="chronological",
                processed_geo_dir=tmp_path,
            )

            # Assert 100% bit-for-bit identical seat matrices and vote share matrices
            np.testing.assert_array_equal(res_baseline.seats_matrix, res_corrupted.seats_matrix)
            np.testing.assert_array_equal(res_baseline.vote_shares_matrix, res_corrupted.vote_shares_matrix)


if __name__ == "__main__":
    unittest.main()
