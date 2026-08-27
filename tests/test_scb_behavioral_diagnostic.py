"""Unit tests for SCB behavioral threshold diagnostic pipeline (Step 3).

Tests verify linear and placebo kernel bounded properties, fixed-effects OLS/WLS
estimation, wave bootstrap determinism, coverage gate completeness, conversion
ratio floor enforcement, and offline reproducibility.
"""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from scripts.scb_behavioral_diagnostic.config import (
    CONVERSION_RATIO_FLOOR_PCT,
    FOCUS_THRESHOLD_PARTIES,
    PROCESSED_DATA_DIR,
    SCB_PANEL_FILE,
    kernel_gaussian_4pct,
    kernel_linear_4pct,
    kernel_placebo_7pct,
    kernel_step_4pct,
)
from scripts.scb_behavioral_diagnostic.models import (
    fit_and_bootstrap_fe_model,
    load_and_prepare_regression_data,
)
from scripts.scb_behavioral_diagnostic.profiles import (
    build_party_threshold_profiles,
    compute_identification_coverage_gate,
)
from scripts.scb_behavioral_diagnostic.qa import run_full_scb_behavioral_qa


class TestSCBBehavioralDiagnostic(unittest.TestCase):
    """Test suite for Step 3 behavioral regression models, bootstrap, and profiles."""

    def setUp(self):
        self.processed_dir = PROCESSED_DATA_DIR
        self.results_file = self.processed_dir / "scb_behavioral_regression_results.csv"
        self.gate_file = self.processed_dir / "identification_coverage_gate.csv"
        self.profiles_file = self.processed_dir / "party_threshold_profiles.csv"
        self.report_file = self.processed_dir / "scb_behavioral_validation_report.json"

    def test_primary_linear_kernel_4pct_properties(self):
        """Assert exact bounds and shape of primary linear proximity kernel K_4."""
        self.assertEqual(kernel_linear_4pct(4.0), 1.0)
        self.assertEqual(kernel_linear_4pct(2.0), 0.0)
        self.assertEqual(kernel_linear_4pct(6.0), 0.0)
        self.assertEqual(kernel_linear_4pct(1.0), 0.0)
        self.assertEqual(kernel_linear_4pct(8.5), 0.0)
        self.assertEqual(kernel_linear_4pct(3.0), 0.5)
        self.assertEqual(kernel_linear_4pct(5.0), 0.5)
        self.assertTrue(np.isnan(kernel_linear_4pct(np.nan)))

    def test_placebo_linear_kernel_7pct_properties(self):
        """Assert exact bounds and shape of placebo linear kernel K_7."""
        self.assertEqual(kernel_placebo_7pct(7.0), 1.0)
        self.assertEqual(kernel_placebo_7pct(5.0), 0.0)
        self.assertEqual(kernel_placebo_7pct(9.0), 0.0)
        self.assertEqual(kernel_placebo_7pct(4.0), 0.0)
        self.assertEqual(kernel_placebo_7pct(10.0), 0.0)
        self.assertEqual(kernel_placebo_7pct(6.0), 0.5)
        self.assertEqual(kernel_placebo_7pct(8.0), 0.5)
        self.assertTrue(np.isnan(kernel_placebo_7pct(np.nan)))

    def test_gaussian_and_step_kernel_properties(self):
        """Assert bounds and properties for sensitivity kernels."""
        self.assertEqual(kernel_gaussian_4pct(4.0, sigma=1.0), 1.0)
        self.assertAlmostEqual(kernel_gaussian_4pct(3.0, sigma=1.0), np.exp(-0.5), places=4)
        
        self.assertEqual(kernel_step_4pct(4.0), 1.0)
        self.assertEqual(kernel_step_4pct(3.0), 1.0)
        self.assertEqual(kernel_step_4pct(4.5), 1.0)
        self.assertEqual(kernel_step_4pct(2.9), 0.0)
        self.assertEqual(kernel_step_4pct(4.6), 0.0)

    def test_regression_dataset_preparation(self):
        """Verify cross-party filtering, variable construction, and no missing data leaks."""
        df = load_and_prepare_regression_data(SCB_PANEL_FILE)
        self.assertGreater(len(df), 500)
        self.assertTrue((df["donor_party"] != df["recipient_party"]).all())
        self.assertTrue((df["R"] >= 0.0).all())
        self.assertTrue((df["A"] >= 0.0).all())
        self.assertTrue((df["K4_symp"] >= 0.0).all() and (df["K4_symp"] <= 1.0).all())
        self.assertTrue((df["K7_symp"] >= 0.0).all() and (df["K7_symp"] <= 1.0).all())
        self.assertEqual(df["wave"].nunique(), 29)

    def test_wave_block_bootstrap_determinism(self):
        """Verify deterministic wave block bootstrap produces identical results with fixed seed."""
        df = load_and_prepare_regression_data(SCB_PANEL_FILE)
        x_cols = ["A", "K4_symp", "A_K4_symp"]
        
        # Run 1
        c1, r1, n1, se1, cil1, ciu1, _ = fit_and_bootstrap_fe_model(
            df, x_cols, n_replications=200, random_seed=42
        )
        # Run 2
        c2, r2, n2, se2, cil2, ciu2, _ = fit_and_bootstrap_fe_model(
            df, x_cols, n_replications=200, random_seed=42
        )
        
        self.assertEqual(c1["A_K4_symp"], c2["A_K4_symp"])
        self.assertEqual(se1["A_K4_symp"], se2["A_K4_symp"])
        self.assertEqual(cil1["A_K4_symp"], cil2["A_K4_symp"])
        self.assertEqual(ciu1["A_K4_symp"], ciu2["A_K4_symp"])

    def test_conversion_ratio_floor_enforcement(self):
        """Verify conversion ratio R / A is NaN when A < 2.0% and valid when A >= 2.0%."""
        panel_df = pd.read_csv(SCB_PANEL_FILE)
        df_prof = build_party_threshold_profiles(panel_df)
        
        # When top_donor_1_affinity < 2.0%, conversion ratio MUST be NaN
        low_a = df_prof[df_prof["top_donor_1_affinity_pct"] < CONVERSION_RATIO_FLOOR_PCT]
        if not low_a.empty:
            self.assertTrue(low_a["top_donor_1_conversion_ratio"].isna().all())
            
        # When top_donor_1_affinity >= 2.0% and vote is present, conversion ratio MUST be valid
        high_a = df_prof[
            (df_prof["top_donor_1_affinity_pct"] >= CONVERSION_RATIO_FLOOR_PCT) &
            df_prof["top_donor_1_vote_pct"].notna()
        ]
        if not high_a.empty:
            self.assertTrue(high_a["top_donor_1_conversion_ratio"].notna().all())
            for _, r in high_a.iterrows():
                expected_ratio = round(r["top_donor_1_vote_pct"] / r["top_donor_1_affinity_pct"], 4)
                self.assertAlmostEqual(r["top_donor_1_conversion_ratio"], expected_ratio, places=3)

    def test_identification_coverage_gate_contents(self):
        """Verify identification coverage gate covers all parliamentary parties with 29 waves."""
        panel_df = pd.read_csv(SCB_PANEL_FILE)
        df_gate = compute_identification_coverage_gate(panel_df)
        
        self.assertEqual(len(df_gate), 8)
        for p in FOCUS_THRESHOLD_PARTIES:
            row = df_gate[df_gate["recipient_party"] == p].iloc[0]
            self.assertEqual(row["total_waves"], 29)
            self.assertTrue(row["is_focus_threshold_party"])
            self.assertGreater(row["waves_in_k4_danger"], 0)

    def test_offline_reproducibility_and_assertions(self):
        """Verify full Step 3 QA execution in a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            report_file = tmp_path / "scb_behavioral_validation_report.json"
            report = run_full_scb_behavioral_qa(
                processed_dir=tmp_path,
                output_report_file=report_file,
                n_bootstrap_replications=100,
            )
            
            self.assertTrue(report["assertions"]["all_assertions_passed"])
            self.assertEqual(report["substantive_conclusion"], "CONCLUSION_A_NO_EVIDENCE")
            self.assertIn("primary_model", report)
            self.assertIn("placebo_comparison", report)


if __name__ == "__main__":
    unittest.main()
