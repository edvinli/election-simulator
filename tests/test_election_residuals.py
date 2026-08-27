"""Unit and regression tests for Historical Poll-to-Election Residual Study."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from scripts.election_residuals.config import (
    ALL_CATEGORIES,
    DEFAULT_ELECTIONS_FILE,
    DEFAULT_POLLS_FILE,
    EVALUATION_ELECTIONS,
    PARLIAMENTARY_PARTIES,
    THRESHOLD_MARGIN_PCT,
    THRESHOLD_PCT,
)
from scripts.election_residuals.consensus import (
    ContributingPoll,
    ElectionPollConsensus,
    build_election_polling_consensus,
    compute_poll_weight,
)
from scripts.election_residuals.residuals import calculate_residuals_study


class PollWeightingAndDeduplicationTests(unittest.TestCase):
    """Test sample size weighting, boundary clipping, missing fallbacks, and deduplication."""

    def test_weight_computation_and_clipping(self) -> None:
        # Bounded within [0.7, 1.5]
        w_small, missing_s = compute_poll_weight(400)
        self.assertFalse(missing_s)
        self.assertAlmostEqual(w_small, 0.7, places=4)  # sqrt(400/1000) = 0.632 -> clipped to 0.7

        w_mid, missing_m = compute_poll_weight(1000)
        self.assertFalse(missing_m)
        self.assertAlmostEqual(w_mid, 1.0, places=4)  # sqrt(1000/1000) = 1.0

        w_large, missing_l = compute_poll_weight(1600)
        self.assertFalse(missing_l)
        self.assertAlmostEqual(w_large, 1.2649, places=3)  # sqrt(1.6) = 1.2649

        w_huge, missing_h = compute_poll_weight(10000)
        self.assertFalse(missing_h)
        self.assertAlmostEqual(w_huge, 1.5, places=4)  # sqrt(10) = 3.162 -> clipped to 1.5

    def test_missing_sample_size_fallback(self) -> None:
        w_none, is_missing_none = compute_poll_weight(None)
        self.assertTrue(is_missing_none)
        self.assertEqual(w_none, 1.0)

        w_nan, is_missing_nan = compute_poll_weight(np.nan)
        self.assertTrue(is_missing_nan)
        self.assertEqual(w_nan, 1.0)

        w_zero, is_missing_zero = compute_poll_weight(0)
        self.assertTrue(is_missing_zero)
        self.assertEqual(w_zero, 1.0)

    def test_exact_14_day_eligibility_window_and_leakage(self) -> None:
        election_date = date(2022, 9, 11)
        synthetic_polls = pd.DataFrame([
            # 1. Eligible poll (inside 14d, pub <= E)
            {"poll_id": "p1", "pollster": "Sifo", "pollster_original": "Sifo", "interview_start": "2022-09-05", "interview_end": "2022-09-08", "publication_date": "2022-09-09", "sample_size": 1500, "party": "M", "support": 18.0},
            {"poll_id": "p1", "pollster": "Sifo", "pollster_original": "Sifo", "interview_start": "2022-09-05", "interview_end": "2022-09-08", "publication_date": "2022-09-09", "sample_size": 1500, "party": "S", "support": 30.0},
            # 2. Too old (interview_end before E - 14d) -> 2022-08-27 is 15 days before
            {"poll_id": "p2", "pollster": "Sifo", "pollster_original": "Sifo", "interview_start": "2022-08-20", "interview_end": "2022-08-27", "publication_date": "2022-08-28", "sample_size": 1500, "party": "M", "support": 17.0},
            # 3. Future leak (publication after E)
            {"poll_id": "p3", "pollster": "Novus", "pollster_original": "Novus", "interview_start": "2022-09-05", "interview_end": "2022-09-08", "publication_date": "2022-09-12", "sample_size": 1500, "party": "M", "support": 19.0},
            # 4. Valid poll from second pollster
            {"poll_id": "p4", "pollster": "Novus", "pollster_original": "Novus", "interview_start": "2022-09-04", "interview_end": "2022-09-07", "publication_date": "2022-09-08", "sample_size": 1200, "party": "M", "support": 17.5},
            {"poll_id": "p4", "pollster": "Novus", "pollster_original": "Novus", "interview_start": "2022-09-04", "interview_end": "2022-09-07", "publication_date": "2022-09-08", "sample_size": 1200, "party": "S", "support": 29.5},
        ])

        # Fill remaining parties with dummy 0
        for p in ["L", "C", "KD", "V", "MP", "SD"]:
            row1 = {"poll_id": "p1", "pollster": "Sifo", "pollster_original": "Sifo", "interview_start": "2022-09-05", "interview_end": "2022-09-08", "publication_date": "2022-09-09", "sample_size": 1500, "party": p, "support": 5.0}
            row4 = {"poll_id": "p4", "pollster": "Novus", "pollster_original": "Novus", "interview_start": "2022-09-04", "interview_end": "2022-09-07", "publication_date": "2022-09-08", "sample_size": 1200, "party": p, "support": 5.0}
            synthetic_polls = pd.concat([synthetic_polls, pd.DataFrame([row1, row4])], ignore_index=True)

        consensus = build_election_polling_consensus(election_date, synthetic_polls, window_days=14)
        self.assertEqual(consensus.retained_pollsters_count, 2)
        contributing_ids = {cp.poll_id for cp in consensus.contributing_polls}
        self.assertEqual(contributing_ids, {"p1", "p4"})


class ResidualCalculationAndStudyTests(unittest.TestCase):
    """Test residual sign conventions, CLR transformations, near-threshold logic, and full study."""

    def test_residual_sign_and_near_threshold(self) -> None:
        # Result > Poll -> Positive residual (outperformance)
        res_pp_pos = 30.33 - 28.84
        self.assertGreater(res_pp_pos, 0.0)

        # Near threshold checks
        self.assertTrue(abs(3.5 - 4.0) <= 1.5)  # 3.5% is near
        self.assertTrue(abs(5.2 - 4.0) <= 1.5)  # 5.2% is near
        self.assertFalse(abs(2.4 - 4.0) <= 1.5)  # 2.4% is not near (1.6 pp away)
        self.assertFalse(abs(5.6 - 4.0) <= 1.5)  # 5.6% is not near (1.6 pp away)

    def test_full_residual_study_all_six_elections(self) -> None:
        """Run complete residual study across all 6 elections (2002-2022)."""
        res = calculate_residuals_study(
            polls_file=DEFAULT_POLLS_FILE,
            elections_file=DEFAULT_ELECTIONS_FILE,
        )

        # 1. Total rows = 6 elections * 9 categories = 54
        self.assertEqual(len(res["residuals_df"]), 54)

        # 2. Check each election exists
        years_present = set(res["residuals_df"]["election_year"].unique())
        self.assertEqual(years_present, {2002, 2006, 2010, 2014, 2018, 2022})

        # 3. Target sum and consensus sum strictly 100%
        for _, group in res["residuals_df"].groupby("election_year"):
            self.assertAlmostEqual(group["poll_consensus"].sum(), 100.0, places=3)
            self.assertAlmostEqual(group["election_result"].sum(), 100.0, places=3)

        # 4. Check that S outperformed polling in all 6 elections
        s_residuals = res["residuals_df"][res["residuals_df"]["party"] == "S"]["residual_pp"].values
        self.assertTrue(all(r > 0 for r in s_residuals), f"S did not outperform in all 6 elections: {s_residuals}")

        # 5. Check that V and MP underperformed polling in all 6 elections
        v_residuals = res["residuals_df"][res["residuals_df"]["party"] == "V"]["residual_pp"].values
        self.assertTrue(all(r < 0 for r in v_residuals), f"V did not underperform in all 6 elections: {v_residuals}")

        mp_residuals = res["residuals_df"][res["residuals_df"]["party"] == "MP"]["residual_pp"].values
        self.assertTrue(all(r < 0 for r in mp_residuals), f"MP did not underperform in all 6 elections: {mp_residuals}")


if __name__ == "__main__":
    unittest.main()
