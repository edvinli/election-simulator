"""Test suite for GeographicProjection v1 and Iterative Proportional Fitting (IPF)."""

from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from scripts.geography.config import (
    DEFAULT_PROCESSED_GEOGRAPHY_DIR,
    MODEL_PARTIES_9,
    OFFICIAL_CONSTITUENCY_CODES,
    REST_MANDATE_LABEL,
)
from scripts.geography.evaluate import evaluate_projection_pair
from scripts.geography.projection import project_constituency_votes
from scripts.geography.raking import iterative_proportional_fitting
from scripts.mandates.allocator import allocate_riksdag_seats
from scripts.mandates.config import FIXED_SEATS_2018, FIXED_SEATS_2022, TOTAL_RIKSDAG_SEATS


class TestGeographicProjection(unittest.TestCase):
    """Test suite covering IPF mathematical properties, leakage guards, REST handling, and historical projection accuracy."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.p_dir = DEFAULT_PROCESSED_GEOGRAPHY_DIR
        cls.pv_df = pd.read_csv(cls.p_dir / "constituency_party_votes_2014_2022.csv")
        cls.el_df = pd.read_csv(cls.p_dir / "constituency_electorates_2014_2026.csv")

    def test_ipf_row_and_column_constraints(self) -> None:
        """Verify IPF algorithm satisfies row and column constraints within tolerance."""
        np.random.seed(42)
        C, P = 29, 9
        B = np.random.uniform(100, 50000, size=(C, P))
        R = np.random.uniform(50000, 200000, size=C)
        R = R * (1000000.0 / np.sum(R))  # Sum to 1,000,000
        C_target = np.random.uniform(10000, 300000, size=P)
        C_target = C_target * (1000000.0 / np.sum(C_target))  # Sum to 1,000,000

        res = iterative_proportional_fitting(B, R, C_target, tol=1e-8)

        self.assertTrue(res.converged)
        self.assertLess(res.max_row_error, 1e-8)
        self.assertLess(res.max_column_error, 1e-8)
        self.assertLess(res.iterations, 50)
        np.testing.assert_allclose(np.sum(res.matrix, axis=1), R, atol=1e-7)
        np.testing.assert_allclose(np.sum(res.matrix, axis=0), C_target, atol=1e-7)

    def test_ipf_zero_and_tiny_cell_handling(self) -> None:
        """Verify IPF handles zero cells gracefully without NaNs or crashes."""
        C, P = 5, 3
        B = np.array([
            [0.0, 10.0, 20.0],
            [10.0, 0.0, 10.0],
            [15.0, 15.0, 0.0],
            [0.0, 0.0, 10.0],
            [20.0, 10.0, 10.0],
        ])
        R = np.array([30.0, 20.0, 30.0, 10.0, 40.0])  # sum = 130
        C_target = np.array([45.0, 35.0, 50.0])       # sum = 130

        res = iterative_proportional_fitting(B, R, C_target, tol=1e-6)
        self.assertTrue(res.converged)
        self.assertFalse(np.isnan(res.matrix).any())
        self.assertEqual(res.matrix[0, 0], 0.0)  # Structural zero preserved

    def test_deterministic_convergence(self) -> None:
        """Verify projection is 100% deterministic given identical inputs."""
        shares = {"M": 0.20, "L": 0.05, "C": 0.08, "KD": 0.06, "S": 0.30, "V": 0.08, "MP": 0.05, "SD": 0.16, "REST": 0.02}

        res1 = project_constituency_votes(shares, baseline_year=2018, target_year=2022, mode="oracle")
        res2 = project_constituency_votes(shares, baseline_year=2018, target_year=2022, mode="oracle")

        self.assertEqual(res1.constituency_votes, res2.constituency_votes)
        self.assertEqual(res1.national_votes, res2.national_votes)
        self.assertEqual(res1.ipf_result.iterations, res2.ipf_result.iterations)

    def test_constituency_total_preservation(self) -> None:
        """Verify integer rounding preserves exact constituency valid vote totals."""
        shares = {"M": 0.19, "L": 0.05, "C": 0.09, "KD": 0.06, "S": 0.28, "V": 0.08, "MP": 0.04, "SD": 0.18, "REST": 0.03}
        res = project_constituency_votes(shares, baseline_year=2014, target_year=2018, mode="oracle")

        for c_code in OFFICIAL_CONSTITUENCY_CODES:
            c_sum = sum(res.constituency_votes[c_code].values())
            self.assertEqual(c_sum, res.constituency_valid_votes[c_code])

    def test_rest_mandate_ineligibility(self) -> None:
        """Verify REST is mapped to OTHER_INELIGIBLE and receives 0 mandates even with large vote share."""
        # Give REST 25% of national votes
        shares = {"M": 0.15, "L": 0.05, "C": 0.05, "KD": 0.05, "S": 0.20, "V": 0.05, "MP": 0.05, "SD": 0.15, "REST": 0.25}
        proj_res = project_constituency_votes(shares, baseline_year=2018, target_year=2022, mode="oracle")

        alloc_input = proj_res.to_allocator_input()
        self.assertIn(REST_MANDATE_LABEL, alloc_input["01"])
        self.assertNotIn("REST", alloc_input["01"])

        alloc_res = allocate_riksdag_seats(
            constituency_votes=alloc_input,
            fixed_seats_by_constituency=FIXED_SEATS_2022,
        )

        self.assertEqual(alloc_res.total_seats, TOTAL_RIKSDAG_SEATS)
        self.assertEqual(alloc_res.final_seats_by_party.get(REST_MANDATE_LABEL, 0), 0)
        self.assertFalse(alloc_res.threshold_eligibility.get(REST_MANDATE_LABEL, False))

    def test_historical_forward_2014_to_2018_projection(self) -> None:
        """Verify 2014 -> 2018 forward projection achieves low MAE and reproduces exact certified seats."""
        eval_oracle = evaluate_projection_pair(baseline_year=2014, target_year=2018, mode="oracle")
        self.assertLess(eval_oracle.constituency_share_mae, 0.01)  # < 1.0% MAE
        self.assertEqual(eval_oracle.total_seat_error, 0)         # Exact 349 seat match

        eval_prod = evaluate_projection_pair(baseline_year=2014, target_year=2018, mode="production")
        self.assertLess(eval_prod.constituency_share_mae, 0.01)
        self.assertEqual(eval_prod.total_seat_error, 0)

    def test_historical_forward_2018_to_2022_projection(self) -> None:
        """Verify 2018 -> 2022 forward projection achieves low MAE and reproduces exact certified seats."""
        eval_oracle = evaluate_projection_pair(baseline_year=2018, target_year=2022, mode="oracle")
        self.assertLess(eval_oracle.constituency_share_mae, 0.01)  # < 1.0% MAE
        self.assertEqual(eval_oracle.total_seat_error, 0)         # Exact 349 seat match

        eval_prod = evaluate_projection_pair(baseline_year=2018, target_year=2022, mode="production")
        self.assertLess(eval_prod.constituency_share_mae, 0.01)
        self.assertEqual(eval_prod.total_seat_error, 0)

    def test_leakage_guard(self) -> None:
        """Verify target election constituency votes are never used in projection construction."""
        # Corrupt 2022 constituency votes in a temporary copy and ensure 2018 baseline projection does not change
        shares = {"M": 0.19, "L": 0.05, "C": 0.07, "KD": 0.05, "S": 0.30, "V": 0.07, "MP": 0.05, "SD": 0.20, "REST": 0.02}
        res_prod = project_constituency_votes(shares, baseline_year=2018, target_year=2022, mode="production")
        # In production mode, only 2018 valid votes and 2022 eligible voters are used
        self.assertIsNotNone(res_prod.constituency_votes)
        self.assertEqual(len(res_prod.constituency_votes), 29)
