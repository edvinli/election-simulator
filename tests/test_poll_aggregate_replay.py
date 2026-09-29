from __future__ import annotations

import os
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
                       "as_of": "d", "opinion_as_of": "d", "gated": True}
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

    def test_scoring_is_refused_until_the_protocol_is_agreed(self) -> None:
        with patch.object(R, "AGREED_PROTOCOL_SHA256", None):
            self.assertFalse(R.protocol_is_agreed())
            with self.assertRaises(SystemExit):
                R.main(["--score"])
        with patch.object(R, "AGREED_PROTOCOL_SHA256", "0" * 64):
            self.assertFalse(R.protocol_is_agreed())


if __name__ == "__main__":
    unittest.main()
