from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.pollofpolls.state import load_individual_polls_dataset
from scripts.poll_aggregate import replay as R


def scored_rows(**candidate_overrides) -> list[dict]:
    """Gated rows where both arms score identically unless overridden."""

    base = {
        "es_9cat": 1.0, "seat_energy_score": 5.0, "brier_coalition": 0.10,
        "brier_threshold": 0.02, "coverage_90": 0.90, "coverage_50": 0.50,
        "seat_total_always_349": True, "all_finite": True,
    }
    rows = []
    for arm in (R.CONTROL, R.CANDIDATE):
        for election in R.ELECTIONS:
            for horizon in R.HORIZONS:
                row = {**base, "arm": arm, "election": election.isoformat(), "horizon_days": horizon,
                       "as_of": "2018-08-12", "selected_estimate_date": "2018-08-12",
                       "aggregate_row_matches": True if arm == R.CANDIDATE else "",
                       "aggregate_information_date": "2018-08-10" if arm == R.CANDIDATE else "",
                       "gated": True}
                if arm == R.CANDIDATE:
                    row.update(candidate_overrides.get(election.isoformat(), {}))
                    row.update({k: v for k, v in candidate_overrides.items() if not isinstance(v, dict)})
                rows.append(row)
    return rows


class GateTests(unittest.TestCase):
    def test_equal_arms_pass_every_gate(self) -> None:
        result = R.evaluate_gates(scored_rows())
        self.assertEqual(result["decision"], "PASS")

    def test_pooled_accuracy_margin_is_five_percent(self) -> None:
        self.assertTrue(R.evaluate_gates(scored_rows(es_9cat=1.049))["gates"]["G1_accuracy"]["pass"])
        result = R.evaluate_gates(scored_rows(es_9cat=1.051))
        self.assertFalse(result["gates"]["G1_accuracy"]["pass"])
        self.assertEqual(result["decision"], "FAIL")

    def test_one_bad_election_fails_even_when_the_pool_passes(self) -> None:
        result = R.evaluate_gates(scored_rows(**{"2018-09-09": {"es_9cat": 1.16}, "2022-09-11": {"es_9cat": 0.9}}))
        self.assertTrue(result["gates"]["G1_accuracy"]["pass"])
        self.assertFalse(result["gates"]["G2_per_election"]["pass"])

    def test_calibration_has_absolute_and_relative_floors(self) -> None:
        self.assertFalse(R.evaluate_gates(scored_rows(coverage_90=0.79))["gates"]["G5_calibration"]["pass"])
        self.assertFalse(R.evaluate_gates(scored_rows(coverage_50=0.39))["gates"]["G5_calibration"]["pass"])

    def test_a_missing_case_fails_integrity(self) -> None:
        rows = scored_rows()
        rows.pop()
        self.assertFalse(R.evaluate_gates(rows)["gates"]["G6_integrity"]["pass"])

    def test_a_selected_estimate_not_dated_on_as_of_fails_integrity(self) -> None:
        rows = scored_rows()
        rows[0]["selected_estimate_date"] = "2018-08-11"  # a control row
        self.assertFalse(R.evaluate_gates(rows)["gates"]["G6_integrity"]["pass"])

    def test_a_candidate_row_that_is_not_the_aggregate_fails_integrity(self) -> None:
        self.assertFalse(
            R.evaluate_gates(scored_rows(aggregate_row_matches=False))["gates"]["G6_integrity"]["pass"])

    def test_an_aggregate_row_using_later_information_fails_integrity(self) -> None:
        self.assertFalse(
            R.evaluate_gates(scored_rows(aggregate_information_date="2018-08-13"))["gates"]["G6_integrity"]["pass"])

    def test_diagnostic_rows_never_enter_the_gates(self) -> None:
        rows = scored_rows()
        rows.append({**rows[0], "arm": R.DIAGNOSTIC, "gated": False, "es_9cat": 99.0})
        self.assertEqual(R.evaluate_gates(rows)["decision"], "PASS")


class HarnessTests(unittest.TestCase):
    def test_arms_differ_only_in_their_replaced_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            aggregate = Path(tmp) / "aggregate.csv"
            aggregate.write_text("date\n")
            roots = {arm: R.build_data_root(arm, Path(tmp), aggregate) for arm in R.ARMS}
            production = R.PROCESSED_ROOT / "pollofpolls"

            def target(arm: str, name: str) -> Path | None:
                path = roots[arm] / "pollofpolls" / name
                return Path(os.readlink(path)) if path.is_symlink() else None

            for name in ("pollofpolls_timeseries.csv", "individual_polls.csv", "swedishpolls_individual_polls.csv"):
                self.assertEqual(target(R.CONTROL, name), production / name)
            self.assertEqual(target(R.CANDIDATE, "pollofpolls_timeseries.csv"), aggregate)
            self.assertEqual(target(R.DIAGNOSTIC, "pollofpolls_timeseries.csv"), aggregate)
            self.assertIsNone(target(R.CANDIDATE, "individual_polls.csv"))  # written, not linked
            self.assertEqual(target(R.DIAGNOSTIC, "individual_polls.csv"), production / "individual_polls.csv")
            self.assertEqual(Path(os.readlink(roots[R.CANDIDATE] / "elections")), R.PROCESSED_ROOT / "elections")

    def test_converted_swedishpolls_polls_load_in_opinion_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "individual_polls.csv"
            written = R.write_swedishpolls_individual_polls(path)
            polls, issues = load_individual_polls_dataset(path)
        self.assertEqual(len(polls), written)
        self.assertEqual(issues["incomplete_main_party_values"], 0)
        self.assertTrue(all(p.publication_date is not None for p in polls))

    def test_candidate_selects_the_aggregate_row_of_its_as_of(self) -> None:
        from datetime import date
        aggregate = R.DEFAULT_OUTPUT_DIR / R.TIMESERIES_FILENAME
        with tempfile.TemporaryDirectory() as tmp:
            root = R.build_data_root(R.CANDIDATE, Path(tmp), aggregate)
            record = R.selected_estimate(
                {"data_root": str(root), "aggregate_file": str(aggregate)}, date(2018, 8, 12))
        self.assertEqual(record["selected_estimate_date"], "2018-08-12")
        self.assertIs(record["aggregate_row_matches"], True)
        self.assertLessEqual(record["aggregate_information_date"], "2018-08-12")

    def test_v2_is_agreed_to_its_frozen_protocol_and_baseline(self) -> None:
        self.assertTrue(R.protocol_is_agreed(R.REPLAY_V2))
        self.assertEqual(R.REPLAY_V2.baseline_commit, "7153bcbef80518a3c3e32f76ca35ec6034a69557")
        self.assertEqual(R.REPLAY_V2.aggregate_file, R.DEFAULT_OUTPUT_DIR / "v0_2" / R.TIMESERIES_FILENAME)
        self.assertNotEqual(R.REPLAY_V2.output_dir, R.REPLAY_V1.output_dir)

    def test_v2_gates_section_is_v1s_word_for_word(self) -> None:
        def section(path):
            text = path.read_text(encoding="utf-8")
            return text[text.index("## 6. Metrics"):text.index("### 7.1")]
        self.assertEqual(section(R.REPLAY_V1.protocol_path), section(R.REPLAY_V2.protocol_path))

    def test_v1_remains_agreed_to_its_frozen_protocol(self) -> None:
        self.assertTrue(R.protocol_is_agreed(R.REPLAY_V1))
        self.assertEqual(R.REPLAY_V1.aggregate_file, R.DEFAULT_OUTPUT_DIR / R.TIMESERIES_FILENAME)

    def test_baseline_check_refuses_a_dirty_checkout(self) -> None:
        def fake_git(*args):
            out = " M data/processed/poll_aggregate/x.csv\n" if args[0] == "status" else ""
            return subprocess.CompletedProcess(args, 0, out, "")
        with patch.object(R, "_git", side_effect=fake_git):
            with self.assertRaisesRegex(RuntimeError, "clean checkout"):
                R.verify_baseline()

    def test_baseline_check_refuses_inputs_changed_since_the_baseline(self) -> None:
        def fake_git(*args):
            out = "data/processed/poll_aggregate/x.csv\n" if args[0] == "diff" else ""
            return subprocess.CompletedProcess(args, 0, out, "")
        with patch.object(R, "_git", side_effect=fake_git):
            with self.assertRaisesRegex(RuntimeError, "differ from baseline"):
                R.verify_baseline()

    def test_baseline_check_excludes_only_the_harness(self) -> None:
        calls = []
        def fake_git(*args):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, "", "")
        with patch.object(R, "_git", side_effect=fake_git):
            R.verify_baseline()
        diff = next(c for c in calls if c[0] == "diff")
        self.assertIn(R.BASELINE_COMMIT, diff)
        self.assertEqual([a for a in diff if a.startswith(":(exclude)")], [f":(exclude){R.HARNESS_PATH}"])
        for path in ("data", "scripts", "diagnostics", "uv.lock"):
            self.assertIn(path, diff)

    def test_scoring_is_refused_until_the_protocol_is_agreed(self) -> None:
        from dataclasses import replace
        with patch.dict(R.REPLAYS, {"v1": replace(R.REPLAY_V1, agreed_sha256=None)}):
            self.assertFalse(R.protocol_is_agreed(R.REPLAYS["v1"]))
            with self.assertRaises(SystemExit):
                R.main(["--score"])
        self.assertFalse(R.protocol_is_agreed(replace(R.REPLAY_V1, agreed_sha256="0" * 64)))


if __name__ == "__main__":
    unittest.main()
