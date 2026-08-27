"""Unit tests for historical party-election threshold events pipeline.

Tests verify party normalization, canonical eligibility, strict anti-leakage guards,
latest-poll-per-pollster rule, non-zero missing handling, sample-size weighting,
exact threshold band intervals, exact 4.0% legal pass boundary, quality grading,
and offline reproducibility.
"""
from datetime import date
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from scripts.threshold_events.config import (
    PROCESSED_DATA_DIR,
    TARGET_ELECTIONS,
    assign_threshold_band,
    grade_episode_quality,
    normalize_party_name,
)
from scripts.threshold_events.consensus import (
    build_final_polling_consensus,
    compute_sample_size_weight,
)
from scripts.threshold_events.election_results import (
    OfficialPartyResult,
    load_all_official_election_results,
)
import scripts.threshold_events.election_results as election_results_module
from scripts.threshold_events.episodes import (
    build_party_election_episodes,
    generate_and_save_canonical_datasets,
)
from scripts.threshold_events.qa import (
    compute_band_summary,
    compute_quadrant_diagnostics,
    run_all_threshold_qa,
    run_window_sensitivity_analysis,
)


class TestThresholdEventsPipeline(unittest.TestCase):
    """Test suite for threshold events data processing, consensus engine, and QA."""

    def setUp(self):
        self.processed_dir = PROCESSED_DATA_DIR
        self.episodes_file = self.processed_dir / "party_election_threshold_events.csv"
        self.details_file = self.processed_dir / "election_consensus_details.csv"
        self.report_file = self.processed_dir / "validation_report.json"

    def test_party_normalization_and_fp_to_l(self):
        """Verify party normalization maps historical FP/Folkpartiet to L and standardizes names."""
        self.assertEqual(normalize_party_name("FP"), "L")
        self.assertEqual(normalize_party_name("Folkpartiet"), "L")
        self.assertEqual(normalize_party_name("Folkpartiet liberalerna"), "L")
        self.assertEqual(normalize_party_name("L"), "L")
        self.assertEqual(normalize_party_name("Liberalerna"), "L")
        self.assertEqual(normalize_party_name("KDS"), "KD")
        self.assertEqual(normalize_party_name("Kristdemokraterna"), "KD")
        self.assertEqual(normalize_party_name("VPK"), "V")
        self.assertEqual(normalize_party_name("Vänsterpartiet"), "V")
        self.assertEqual(normalize_party_name("F!"), "FI")
        self.assertEqual(normalize_party_name("Feministiskt initiativ"), "FI")
        self.assertEqual(normalize_party_name("NYD"), "NYD")

    def test_sample_size_weighting_formula(self):
        """Verify sqrt(N / 1000) clipped to [0.7, 1.5] and missing N returns 1.0."""
        w_small, is_miss_small = compute_sample_size_weight(100)
        self.assertEqual(w_small, 0.7)  # sqrt(0.1) = 0.316 -> clipped to 0.7
        self.assertFalse(is_miss_small)

        w_base, is_miss_base = compute_sample_size_weight(1000)
        self.assertEqual(w_base, 1.0)
        self.assertFalse(is_miss_base)

        w_large, is_miss_large = compute_sample_size_weight(4000)
        self.assertEqual(w_large, 1.5)  # sqrt(4.0) = 2.0 -> clipped to 1.5
        self.assertFalse(is_miss_large)

        w_none, is_miss_none = compute_sample_size_weight(None)
        self.assertEqual(w_none, 1.0)
        self.assertTrue(is_miss_none)

        w_nan, is_miss_nan = compute_sample_size_weight(np.nan)
        self.assertEqual(w_nan, 1.0)
        self.assertTrue(is_miss_nan)

    def test_strict_anti_leakage_poison_tests(self):
        """Verify that poisoned future polls, post-election publications, and invalid dates are rejected."""
        elec_date = date(2022, 9, 11)
        
        # Base valid poll
        valid_rows = [
            {
                "poll_id": "valid_1",
                "pollster": "Sifo",
                "pollster_original": "Kantar Sifo",
                "interview_start": "2022-09-01",
                "interview_end": "2022-09-08",
                "publication_date": "2022-09-09",
                "sample_size": 1500,
                "party": "M",
                "support": 19.0,
            },
            {
                "poll_id": "valid_1",
                "pollster": "Sifo",
                "pollster_original": "Kantar Sifo",
                "interview_start": "2022-09-01",
                "interview_end": "2022-09-08",
                "publication_date": "2022-09-09",
                "sample_size": 1500,
                "party": "L",
                "support": 5.0,
            },
        ]
        
        # Poisoned polls that MUST be excluded
        poison_rows = [
            # Poison 1: Publication after election day
            {
                "poll_id": "poison_pub_post",
                "pollster": "Novus",
                "pollster_original": "Novus",
                "interview_start": "2022-09-05",
                "interview_end": "2022-09-09",
                "publication_date": "2022-09-12",  # Post-election!
                "sample_size": 1500,
                "party": "L",
                "support": 99.0,
            },
            # Poison 2: Interview ending after election day
            {
                "poll_id": "poison_end_post",
                "pollster": "Ipsos",
                "pollster_original": "Ipsos",
                "interview_start": "2022-09-05",
                "interview_end": "2022-09-12",  # Post-election!
                "publication_date": "2022-09-11",
                "sample_size": 1500,
                "party": "L",
                "support": 99.0,
            },
            # Poison 3: Interview outside 14-day window (ended 16 days before)
            {
                "poll_id": "poison_too_old",
                "pollster": "Demoskop",
                "pollster_original": "Demoskop",
                "interview_start": "2022-08-20",
                "interview_end": "2022-08-26",  # E - 16 days!
                "publication_date": "2022-08-27",
                "sample_size": 1500,
                "party": "L",
                "support": 99.0,
            },
            # Poison 4: Missing interview_end date
            {
                "poll_id": "poison_missing_date",
                "pollster": "Skop",
                "pollster_original": "Skop",
                "interview_start": "2022-09-01",
                "interview_end": None,  # Missing!
                "publication_date": "2022-09-09",
                "sample_size": 1500,
                "party": "L",
                "support": 99.0,
            },
        ]
        
        test_df = pd.DataFrame(valid_rows + poison_rows)
        res = build_final_polling_consensus(elec_date, 2022, test_df, window_days=14)
        
        self.assertIsNotNone(res)
        self.assertEqual(res.total_eligible_polls_in_window, 1)
        self.assertEqual(res.total_retained_pollsters, 1)
        self.assertEqual(res.contributing_records[0].poll_id, "valid_1")
        self.assertEqual(res.party_consensus["L"].consensus_pct, 5.0)

    def test_missing_party_support_is_never_zero(self):
        """Verify that an omitted party in a selected poll is excluded from that party's mean, not set to 0.0%."""
        elec_date = date(2022, 9, 11)
        polls_data = [
            # Pollster 1 reports M and FI
            {"poll_id": "p1", "pollster": "HouseA", "pollster_original": "HouseA", "interview_start": "2022-09-01", "interview_end": "2022-09-08", "publication_date": "2022-09-09", "sample_size": 1000, "party": "M", "support": 20.0},
            {"poll_id": "p1", "pollster": "HouseA", "pollster_original": "HouseA", "interview_start": "2022-09-01", "interview_end": "2022-09-08", "publication_date": "2022-09-09", "sample_size": 1000, "party": "FI", "support": 3.0},
            # Pollster 2 reports M only (omits FI)
            {"poll_id": "p2", "pollster": "HouseB", "pollster_original": "HouseB", "interview_start": "2022-09-02", "interview_end": "2022-09-08", "publication_date": "2022-09-09", "sample_size": 1000, "party": "M", "support": 22.0},
        ]
        df = pd.DataFrame(polls_data)
        res = build_final_polling_consensus(elec_date, 2022, df, window_days=14)
        
        self.assertIsNotNone(res)
        # M is reported by both (mean = 21.0%)
        self.assertEqual(res.party_consensus["M"].consensus_pct, 21.0)
        self.assertEqual(res.party_consensus["M"].party_pollster_count, 2)
        # FI is reported only by HouseA -> consensus MUST be 3.0%, NOT (3.0 + 0)/2 = 1.5%
        self.assertEqual(res.party_consensus["FI"].consensus_pct, 3.0)
        self.assertEqual(res.party_consensus["FI"].party_pollster_count, 1)

    def test_nullable_metadata_polls_are_retained_and_missing_n_is_weighted(self):
        """Eligible polls with missing N or interview_start must reach consensus."""
        elec_date = date(2022, 9, 11)
        polls_data = [
            # This is HouseA's latest poll.  Its missing sample size must use
            # the documented neutral fallback rather than being dropped.
            {
                "poll_id": "latest_missing_n",
                "pollster": "HouseA",
                "pollster_original": "HouseA",
                "interview_start": "2022-09-01",
                "interview_end": "2022-09-08",
                "publication_date": "2022-09-09",
                "sample_size": np.nan,
                "party": "M",
                "support": 20.0,
            },
            # An older HouseA poll confirms that latest-poll selection occurs
            # after eligibility, not after filtering on available N.
            {
                "poll_id": "older_known_n",
                "pollster": "HouseA",
                "pollster_original": "HouseA",
                "interview_start": "2022-08-30",
                "interview_end": "2022-09-07",
                "publication_date": "2022-09-08",
                "sample_size": 2000,
                "party": "M",
                "support": 80.0,
            },
            # Missing interview_start is valid under the documented date
            # eligibility rules because interview_end is present.
            {
                "poll_id": "missing_start",
                "pollster": "HouseB",
                "pollster_original": "HouseB",
                "interview_start": np.nan,
                "interview_end": "2022-09-07",
                "publication_date": "2022-09-08",
                "sample_size": 1000,
                "party": "M",
                "support": 40.0,
            },
        ]
        res = build_final_polling_consensus(elec_date, 2022, pd.DataFrame(polls_data), window_days=14)

        self.assertIsNotNone(res)
        self.assertEqual(res.total_eligible_polls_in_window, 3)
        self.assertEqual(res.total_retained_pollsters, 2)
        by_poll = {record.poll_id: record for record in res.contributing_records}
        self.assertEqual(set(by_poll), {"latest_missing_n", "missing_start"})
        self.assertTrue(by_poll["latest_missing_n"].sample_size_missing)
        self.assertIsNone(by_poll["latest_missing_n"].sample_size)
        self.assertEqual(by_poll["latest_missing_n"].weight, 1.0)
        self.assertIsNone(by_poll["missing_start"].interview_start)
        self.assertFalse(by_poll["missing_start"].sample_size_missing)
        self.assertEqual(res.contributing_records[0].poll_id, "latest_missing_n")
        self.assertEqual(res.party_consensus["M"].party_eligible_poll_count, 3)
        self.assertEqual(res.party_consensus["M"].party_contributing_poll_count, 2)
        self.assertEqual(res.party_consensus["M"].party_sample_size_coverage, 0.5)
        self.assertEqual(res.election_sample_size_coverage, 0.5)

    def test_latest_selection_is_deterministic_with_missing_metadata(self):
        """Tie-breakers remain deterministic when nullable metadata is present."""
        elec_date = date(2022, 9, 11)
        rows = [
            {
                "poll_id": "tie_missing_start",
                "pollster": "HouseA",
                "pollster_original": "HouseA",
                "interview_start": np.nan,
                "interview_end": "2022-09-08",
                "publication_date": "2022-09-09",
                "sample_size": 1000,
                "party": "M",
                "support": 21.0,
            },
            {
                "poll_id": "tie_known_start",
                "pollster": "HouseA",
                "pollster_original": "HouseA",
                "interview_start": "2022-09-01",
                "interview_end": "2022-09-08",
                "publication_date": "2022-09-09",
                "sample_size": 1000,
                "party": "M",
                "support": 22.0,
            },
        ]
        df = pd.DataFrame(rows)
        first = build_final_polling_consensus(elec_date, 2022, df, window_days=14)
        second = build_final_polling_consensus(elec_date, 2022, df.iloc[::-1].reset_index(drop=True), window_days=14)

        self.assertEqual(first.contributing_records[0].poll_id, "tie_known_start")
        self.assertEqual(second.contributing_records[0].poll_id, "tie_known_start")
        self.assertEqual(first.party_consensus["M"].consensus_pct, second.party_consensus["M"].consensus_pct)

    def test_duplicate_canonical_party_rows_fail_closed(self):
        """Aliases for one party must not create duplicate pivot cells."""
        elec_date = date(2022, 9, 11)
        rows = [
            {
                "poll_id": "duplicate_party",
                "pollster": "HouseA",
                "pollster_original": "HouseA",
                "interview_start": "2022-09-01",
                "interview_end": "2022-09-08",
                "publication_date": "2022-09-09",
                "sample_size": 1000,
                "party": "L",
                "support": 5.0,
            },
            {
                "poll_id": "duplicate_party",
                "pollster": "HouseA",
                "pollster_original": "HouseA",
                "interview_start": "2022-09-01",
                "interview_end": "2022-09-08",
                "publication_date": "2022-09-09",
                "sample_size": 1000,
                "party": "Liberalerna",
                "support": 6.0,
            },
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate party rows"):
            build_final_polling_consensus(elec_date, 2022, pd.DataFrame(rows), window_days=14)

    def test_conflicting_poll_metadata_fails_closed(self):
        """Party rows with contradictory poll metadata cannot be joined safely."""
        elec_date = date(2022, 9, 11)
        rows = [
            {
                "poll_id": "conflicting_metadata",
                "pollster": "HouseA",
                "pollster_original": "HouseA",
                "interview_start": "2022-09-01",
                "interview_end": "2022-09-08",
                "publication_date": "2022-09-09",
                "sample_size": 1000,
                "party": "M",
                "support": 20.0,
            },
            {
                "poll_id": "conflicting_metadata",
                "pollster": "HouseA",
                "pollster_original": "HouseA",
                "interview_start": "2022-09-01",
                "interview_end": "2022-09-08",
                "publication_date": "2022-09-09",
                "sample_size": 2000,
                "party": "L",
                "support": 5.0,
            },
        ]
        with self.assertRaisesRegex(ValueError, "Inconsistent poll metadata"):
            build_final_polling_consensus(elec_date, 2022, pd.DataFrame(rows), window_days=14)

    def test_official_results_archive_is_write_once(self):
        """A changed archive is rejected without overwriting prior evidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_dir = Path(tmpdir)
            original_raw_dir = election_results_module.RAW_DATA_DIR
            try:
                election_results_module.RAW_DATA_DIR = archive_dir
                load_all_official_election_results()
                archive_file = archive_dir / "official_election_results_archive.json"
                archive_file.write_text("tampered archive", encoding="utf-8")

                with self.assertRaisesRegex(RuntimeError, "Immutable official-results archive"):
                    load_all_official_election_results()
                self.assertEqual(archive_file.read_text(encoding="utf-8"), "tampered archive")
            finally:
                election_results_module.RAW_DATA_DIR = original_raw_dir

    def test_predefined_threshold_band_boundaries(self):
        """Assert exact half-open intervals for predefined threshold bands."""
        self.assertEqual(assign_threshold_band(1.99), "<2")
        self.assertEqual(assign_threshold_band(2.00), "2–3")
        self.assertEqual(assign_threshold_band(2.99), "2–3")
        self.assertEqual(assign_threshold_band(3.00), "3–3.5")
        self.assertEqual(assign_threshold_band(3.49), "3–3.5")
        self.assertEqual(assign_threshold_band(3.50), "3.5–4")
        self.assertEqual(assign_threshold_band(3.99), "3.5–4")
        self.assertEqual(assign_threshold_band(4.00), "4–4.5")
        self.assertEqual(assign_threshold_band(4.49), "4–4.5")
        self.assertEqual(assign_threshold_band(4.50), "4.5–5")
        self.assertEqual(assign_threshold_band(4.99), "4.5–5")
        self.assertEqual(assign_threshold_band(5.00), "5–6")
        self.assertEqual(assign_threshold_band(5.99), "5–6")
        self.assertEqual(assign_threshold_band(6.00), ">=6")
        self.assertEqual(assign_threshold_band(12.5), ">=6")

    def test_exact_4pct_pass_condition_legal_formula(self):
        """Verify that passed_4pct uses exact vote counts (25 * V_p >= V_valid)."""
        valid_tot = 1000000
        # Exactly 40,000 votes -> 25 * 40000 = 1000000 == valid_tot -> Passed!
        self.assertTrue(bool(25 * 40000 >= valid_tot))
        # 39,999 votes -> 25 * 39999 = 999975 < 1000000 -> Failed!
        self.assertFalse(bool(25 * 39999 >= valid_tot))

        # Check in processed episodes dataset
        df_ep = pd.read_csv(self.episodes_file)
        valid_ep = df_ep[df_ep["episode_quality"].isin(["HIGH", "MEDIUM", "LOW"])]
        for _, r in valid_ep.iterrows():
            votes = int(r["votes"])
            valid_votes = int(r["valid_votes_total"])
            expected_passed = bool(25 * votes >= valid_votes)
            self.assertEqual(bool(r["passed_4pct"]), expected_passed)
            self.assertEqual(bool(r["actual_side"]), expected_passed)

    def test_episode_threshold_side_uses_exact_votes_not_rounded_share(self):
        """A rounded 4.0% display value must not turn a legal failure into a pass."""
        elec_date = date(2022, 9, 11)
        polls = pd.DataFrame(
            [
                {
                    "poll_id": "boundary_poll",
                    "pollster": "HouseA",
                    "pollster_original": "HouseA",
                    "interview_start": "2022-09-01",
                    "interview_end": "2022-09-08",
                    "publication_date": "2022-09-09",
                    "sample_size": 1000,
                    "party": "M",
                    "support": 4.0,
                }
            ]
        )
        official = {
            2022: {
                "M": OfficialPartyResult(
                    election_year=2022,
                    election_date=elec_date,
                    party="M",
                    party_raw="M",
                    votes=39_999,
                    valid_votes_total=1_000_000,
                    # Deliberately rounded as 4.0%; the exact cross-product
                    # is below the legal threshold.
                    vote_share_pct=4.0,
                    source_url="synthetic",
                )
            }
        }

        episodes, _ = build_party_election_episodes(
            polls_df=polls,
            official_results=official,
            window_days=14,
        )
        row = episodes[(episodes["election_year"] == 2022) & (episodes["party"] == "M")].iloc[0]
        self.assertFalse(bool(row["passed_4pct"]))
        self.assertFalse(bool(row["actual_side"]))

    def test_episode_construction_rejects_nonpositive_valid_vote_total(self):
        """A missing/invalid denominator cannot be classified from a rounded share."""
        official = {
            1991: {
                "M": OfficialPartyResult(
                    election_year=1991,
                    election_date=date(1991, 9, 15),
                    party="M",
                    party_raw="M",
                    votes=4_000,
                    valid_votes_total=0,
                    # Deliberately plausible but unusable rounded display.
                    vote_share_pct=4.0,
                    source_url="synthetic",
                )
            }
        }
        empty_polls = pd.DataFrame(
            columns=[
                "poll_id",
                "pollster",
                "pollster_original",
                "interview_start",
                "interview_end",
                "publication_date",
                "sample_size",
                "party",
                "support",
            ]
        )

        with self.assertRaisesRegex(ValueError, "valid_votes_total"):
            build_party_election_episodes(
                polls_df=empty_polls,
                official_results=official,
                window_days=14,
            )

    def test_quality_grading_rules(self):
        """Assert deterministic quality grading logic."""
        # HIGH: >= 5 pollsters, >= 15 eligible polls, >= 80% N coverage
        self.assertEqual(grade_episode_quality(5, 15, 0.85, True), "HIGH")
        self.assertEqual(grade_episode_quality(8, 25, 1.0, True), "HIGH")

        # MEDIUM: >= 3 pollsters or (>= 2 pollsters and >= 5 polls)
        self.assertEqual(grade_episode_quality(3, 4, 0.5, True), "MEDIUM")
        self.assertEqual(grade_episode_quality(2, 6, 0.9, True), "MEDIUM")

        # LOW: 1 or 2 pollsters with < 5 polls
        self.assertEqual(grade_episode_quality(1, 1, 1.0, True), "LOW")
        self.assertEqual(grade_episode_quality(2, 3, 0.5, True), "LOW")

        # EXCLUDE: 0 pollsters or incomplete metadata
        self.assertEqual(grade_episode_quality(0, 0, 0.0, True), "EXCLUDE")
        self.assertEqual(grade_episode_quality(5, 15, 0.9, False), "EXCLUDE")

    def test_1998_excluded_from_all_windows_due_to_missing_dates(self):
        """Verify 1998 is excluded from canonical 14d and sensitivity windows due to missing interview dates."""
        df_ep = pd.read_csv(self.episodes_file)
        sub_1998 = df_ep[df_ep["election_year"] == 1998]
        self.assertTrue((sub_1998["episode_quality"] == "EXCLUDE").all())
        self.assertTrue((sub_1998["threshold_band"] == "EXCLUDED").all())

    def test_canonical_dataset_structure_and_quadrants(self):
        """Assert canonical dataset has all required columns and valid 4-quadrant assignments."""
        df_ep = pd.read_csv(self.episodes_file)
        required_cols = [
            "election_year", "election_date", "party", "party_raw",
            "final_poll_consensus_pct", "actual_result_pct", "votes", "valid_votes_total",
            "residual_pp", "distance_from_4_pp", "passed_4pct", "forecast_side", "actual_side",
            "quadrant", "threshold_crossing_distance_pp", "threshold_band",
            "party_eligible_poll_count", "party_contributing_poll_count", "party_pollster_count",
            "party_sample_size_coverage", "election_poll_count", "election_pollster_count",
            "consensus_window_days", "metadata_quality", "episode_quality", "source_notes",
        ]
        for col in required_cols:
            self.assertIn(col, df_ep.columns, f"Missing column {col} in episodes CSV")

        # Check quadrants
        valid_ep = df_ep[df_ep["episode_quality"].isin(["HIGH", "MEDIUM", "LOW"])]
        for _, r in valid_ep.iterrows():
            f_side = r["forecast_side"]
            a_side = r["actual_side"]
            q = r["quadrant"]
            if not f_side and not a_side:
                self.assertEqual(q, "below_to_below")
            elif not f_side and a_side:
                self.assertEqual(q, "below_to_above")
            elif f_side and not a_side:
                self.assertEqual(q, "above_to_below")
            else:
                self.assertEqual(q, "above_to_above")

    def test_offline_reproducibility(self):
        """Verify processing runs completely offline into a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            df_ep, df_det = generate_and_save_canonical_datasets(output_dir=tmp_path)
            self.assertEqual(len(df_ep), 77)
            self.assertGreater(len(df_det), 0)

            report_path = tmp_path / "validation_report.json"
            report = run_all_threshold_qa(processed_dir=tmp_path, output_file=report_path)
            self.assertTrue(report["assertions"]["all_assertions_passed"])


if __name__ == "__main__":
    unittest.main()
