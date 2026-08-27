"""Unit tests for SCB PSU support-voting data pipeline.

Verifies raw manifest integrity, exact wave ordering, canonical party mappings,
suppressed value handling, duplicate avoidance, percentage bounds, Table D selectors,
offline reproducibility, and joined panel correctness.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from scripts.scb_support_voting.config import (
    PARLIAMENTARY_PARTIES,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SCB_TABLES,
    WAVES_2010_2026,
    classify_category,
)
from scripts.scb_support_voting.process import (
    build_donor_recipient_panel,
    parse_scb_cell_value,
    process_all,
    process_table_a,
    process_table_b,
    process_table_c,
    process_table_d,
)
from scripts.scb_support_voting.qa import run_all_qa


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


class TestSCBSupportVotingPipeline(unittest.TestCase):
    """Test suite for SCB PSU data extraction, processing, and QA."""

    def setUp(self):
        self.raw_dir = RAW_DATA_DIR
        self.processed_dir = PROCESSED_DATA_DIR
        self.manifest_path = self.raw_dir / "manifest.json"

    def test_raw_manifest_structure_and_hashes(self):
        """Verify manifest.json exists, has all 4 tables, and recorded SHA-256 hashes match files on disk."""
        self.assertTrue(self.manifest_path.exists(), "manifest.json must exist in data/raw/scb_support_voting/")
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertIn("tables", manifest)
        self.assertEqual(set(manifest["tables"].keys()), set(SCB_TABLES.keys()))

        # Check each table's files and hashes
        for table_key, t_info in manifest["tables"].items():
            files = t_info["files"]
            for f_type in ["metadata", "query", "data"]:
                self.assertIn(f_type, files, f"Missing {f_type} in {table_key} manifest")
                f_meta = files[f_type]
                disk_path = self.raw_dir / f_meta["filename"]
                self.assertTrue(disk_path.exists(), f"File {f_meta['filename']} missing on disk")
                self.assertEqual(disk_path.stat().st_size, f_meta["byte_count"])
                self.assertEqual(compute_file_sha256(disk_path), f_meta["sha256"])

    def test_exact_29_waves_ordered(self):
        """Assert exact ordered list of 29 waves (2010M11 -> 2026M05) in all processed datasets."""
        for filename in [
            "vote_by_sympathy.csv",
            "second_choice_by_sympathy.csv",
            "overall_vote_intention.csv",
            "overall_party_sympathy.csv",
            "scb_donor_recipient_panel.csv",
        ]:
            df = pd.read_csv(self.processed_dir / filename)
            observed_waves = sorted(df["wave"].unique().tolist())
            self.assertEqual(
                observed_waves,
                WAVES_2010_2026,
                f"Wave sequence mismatch in {filename}",
            )
            self.assertEqual(len(observed_waves), 29)
            self.assertEqual(observed_waves[0], "2010M11")
            self.assertEqual(observed_waves[-1], "2026M05")

    def test_table_d_exact_selectors(self):
        """Verify Table D was queried with exact selectors Kon=TOT and Alder=tot18+ without demographic averaging."""
        # Check raw query file
        query_path = self.raw_dir / "table_d_overall_party_sympathy_query.json"
        with open(query_path, "r", encoding="utf-8") as f:
            query = json.load(f)

        query_vars = {item["code"]: item["selection"]["values"] for item in query["query"]}
        self.assertEqual(query_vars.get("Kon"), ["TOT"], "Kon selector must be ['TOT']")
        self.assertEqual(query_vars.get("Alder"), ["tot18+"], "Alder selector must be ['tot18+']")

        # Check processed dataset
        df_d = pd.read_csv(self.processed_dir / "overall_party_sympathy.csv")
        self.assertEqual(set(df_d["kon_code_raw"].unique()), {"TOT"})
        self.assertEqual(set(df_d["alder_code_raw"].unique()), {"tot18+"})
        # 10 parties * 29 waves = 290 rows
        self.assertEqual(len(df_d), 290)

    def test_suppressed_cell_handling(self):
        """Verify '..' cells are parsed as NaN with value_status='suppressed' and never as 0.0."""
        val_nan, status_supp = parse_scb_cell_value("..")
        self.assertTrue(np.isnan(val_nan))
        self.assertEqual(status_supp, "suppressed")

        val_obs, status_obs = parse_scb_cell_value("4.2")
        self.assertEqual(val_obs, 4.2)
        self.assertEqual(status_obs, "observed")

        # In processed CSVs, check that suppressed cells have NaN estimates
        df_a = pd.read_csv(self.processed_dir / "vote_by_sympathy.csv")
        suppressed_rows = df_a[df_a["value_status"] == "suppressed"]
        self.assertGreater(len(suppressed_rows), 0, "Expected suppressed cells in Table A")
        self.assertTrue(
            suppressed_rows["estimate_pct"].isna().all(),
            "Suppressed cells must have NaN estimate_pct",
        )
        self.assertTrue(
            suppressed_rows["margin_error_pp"].isna().all(),
            "Suppressed cells must have NaN margin_error_pp",
        )

    def test_canonical_party_mapping_and_fp_to_l(self):
        """Test canonical party classification, FP -> L mapping, and category separation."""
        # Test historical FP mapping
        canon_fp, type_fp = classify_category("fp", "Folkpartiet")
        self.assertEqual(canon_fp, "L")
        self.assertEqual(type_fp, "parliamentary_party")

        canon_l, type_l = classify_category("l", "Liberalerna")
        self.assertEqual(canon_l, "L")
        self.assertEqual(type_l, "parliamentary_party")

        # Test parliamentary parties
        for code in ["m", "c", "kd", "mp", "s", "v", "SD"]:
            canon, p_type = classify_category(code, code)
            self.assertEqual(canon, code.upper())
            self.assertEqual(p_type, "parliamentary_party")

        # Test non-party categories
        self.assertEqual(classify_category("blankt", "blankt")[1], "blank_vote")
        self.assertEqual(classify_category("vet ej", "vet ej")[1], "dont_know")
        self.assertEqual(classify_category("inget parti", "inget parti")[1], "no_second_choice")
        self.assertEqual(classify_category("hela väljarkåren", "hela väljarkåren")[1], "total_electorate")
        self.assertEqual(classify_category("ingen sympati/vet ej", "ingen sympati/vet ej")[1], "no_sympathy")

    def test_no_duplicate_cells(self):
        """Assert zero duplicates across all processed datasets."""
        df_a = pd.read_csv(self.processed_dir / "vote_by_sympathy.csv")
        df_b = pd.read_csv(self.processed_dir / "second_choice_by_sympathy.csv")
        df_c = pd.read_csv(self.processed_dir / "overall_vote_intention.csv")
        df_d = pd.read_csv(self.processed_dir / "overall_party_sympathy.csv")
        df_panel = pd.read_csv(self.processed_dir / "scb_donor_recipient_panel.csv")

        self.assertEqual(df_a.duplicated(subset=["wave", "best_party_code_raw", "vote_party_code_raw"]).sum(), 0)
        self.assertEqual(df_b.duplicated(subset=["wave", "best_party_code_raw", "second_choice_code_raw"]).sum(), 0)
        self.assertEqual(df_c.duplicated(subset=["wave", "party_code_raw"]).sum(), 0)
        self.assertEqual(df_d.duplicated(subset=["wave", "party_code_raw"]).sum(), 0)
        self.assertEqual(df_panel.duplicated(subset=["wave", "donor_party", "recipient_party"]).sum(), 0)

    def test_percentage_bounds(self):
        """Assert that observed estimates satisfy 0 <= estimate <= 100 and MOE >= 0."""
        for filename in [
            "vote_by_sympathy.csv",
            "second_choice_by_sympathy.csv",
            "overall_vote_intention.csv",
            "overall_party_sympathy.csv",
        ]:
            df = pd.read_csv(self.processed_dir / filename)
            obs = df[df["value_status"] == "observed"]["estimate_pct"]
            self.assertTrue((obs >= 0.0).all(), f"Negative estimate found in {filename}")
            self.assertTrue((obs <= 100.0).all(), f"Estimate > 100 found in {filename}")

            moe = df[df["margin_error_pp"].notna()]["margin_error_pp"]
            self.assertTrue((moe >= 0.0).all(), f"Negative MOE found in {filename}")

    def test_joined_donor_recipient_panel_structure(self):
        """Assert joined donor-recipient panel contains 1,856 rows and separate uncertainty fields."""
        df_panel = pd.read_csv(self.processed_dir / "scb_donor_recipient_panel.csv")
        # 8 donor parties * 8 recipient parties * 29 waves = 1856 rows
        self.assertEqual(len(df_panel), 1856)

        # Check required columns
        required_cols = [
            "wave", "survey_date", "period",
            "donor_party", "donor_party_code_raw",
            "recipient_party", "recipient_party_code_raw",
            "vote_estimate_pct", "vote_margin_error_pp", "vote_value_status",
            "second_choice_estimate_pct", "second_choice_margin_error_pp", "second_choice_value_status",
            "donor_overall_sympathy_pct", "donor_overall_sympathy_margin_error_pp", "donor_overall_sympathy_value_status",
            "recipient_overall_sympathy_pct", "recipient_overall_sympathy_margin_error_pp", "recipient_overall_sympathy_value_status",
            "recipient_overall_vote_pct", "recipient_overall_vote_margin_error_pp", "recipient_overall_vote_value_status",
        ]
        for col in required_cols:
            self.assertIn(col, df_panel.columns, f"Missing required column {col} in panel")

        # Check that party pairs are within parliamentary parties
        self.assertTrue(df_panel["donor_party"].isin(PARLIAMENTARY_PARTIES).all())
        self.assertTrue(df_panel["recipient_party"].isin(PARLIAMENTARY_PARTIES).all())

    def test_validation_report_assertions_and_reconciliation(self):
        """Assert validation report is generated and all QA assertions pass."""
        report_path = self.processed_dir / "validation_report.json"
        self.assertTrue(report_path.exists())
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        self.assertTrue(report["assertions"]["all_assertions_passed"])
        self.assertIn("reconciliation_diagnostics", report)
        self.assertIn("row_sum_diagnostics", report)
        self.assertIn("coverage_diagnostics", report)

    def test_offline_reproducibility(self):
        """Verify processing runs completely offline into a temporary directory without network access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            processed_dict = process_all(raw_dir=self.raw_dir, output_dir=tmp_path)
            self.assertEqual(len(processed_dict["donor_recipient_panel"]), 1856)

            report_file = tmp_path / "validation_report.json"
            report = run_all_qa(processed_dir=tmp_path, output_file=report_file)
            self.assertTrue(report["assertions"]["all_assertions_passed"])


if __name__ == "__main__":
    unittest.main()
