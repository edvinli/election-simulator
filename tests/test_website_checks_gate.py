"""Tests for strict website acceptance checks and narrow baseline failure allowance."""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from scripts import election_automation as ea
from scripts.election_automation_base import AutomationError


EXACT_BASELINE_OUTPUT = """
[schema 1.2] desktop (1280x1000)
  ok    no legacy .eg-party cards in DOM
  ok    no horizontal overflow
  ok    no console errors
  ok    no uncaught exceptions

[schema 1.3] desktop (1280x1000)
  ok    schema 1.3 starts with hidden histogram
  ok    schema 1.3 has no histogram link initially
  ok    the statistics grid starts hidden
  ok    drag S to Regering
  ok    drag V to Regering
  ok    drag MP to Regering
  FAIL  government mask is 112
          expected "112", got "16"
  FAIL  S+V+MP median matches published
          expected "165 mandat", got "110 mandat"
  FAIL  S+V+MP probability matches published
          expected "2,2 %", got "0,0 %"
  ok    the statistics grid is visible with a government
  ok    statistics labels
  FAIL  median statistic is median_seats
          expected "165", got "110"
  FAIL  50 % statistic is p25–p75
          expected "162–168", got "107–113"
  FAIL  80 % statistic is p10–p90
          expected "157–171", got "105–117"
  FAIL  90 % statistic is p05–p95
          expected "155–173", got "104–119"
  ok    the prominent Majoritet result is still shown
  ok    summary includes link to histogram
  ok    histogram is visible with government
  FAIL  histogram mask follows government
          expected "112", got "16"
  ok    histogram has rendered seat bars
  ok    reset hides histogram in schema 1.3
  ok    reset hides the statistics grid
  ok    reset empties the statistics grid
  ok    schema 1.3 no horizontal overflow
  ok    schema 1.3 no console errors
  ok    schema 1.3 no uncaught exceptions

[schema 1.3] narrow-360 (360x900)
  ok    schema 1.3 starts with hidden histogram
  ok    schema 1.3 has no histogram link initially
  ok    the statistics grid starts hidden
  ok    drag S to Regering
  ok    drag V to Regering
  ok    drag MP to Regering
  FAIL  government mask is 112
          expected "112", got "16"
  FAIL  S+V+MP median matches published
          expected "165 mandat", got "110 mandat"
  FAIL  S+V+MP probability matches published
          expected "2,2 %", got "0,0 %"
  ok    the statistics grid is visible with a government
  ok    statistics labels
  FAIL  median statistic is median_seats
          expected "165", got "110"
  FAIL  50 % statistic is p25–p75
          expected "162–168", got "107–113"
  FAIL  80 % statistic is p10–p90
          expected "157–171", got "105–117"
  FAIL  90 % statistic is p05–p95
          expected "155–173", got "104–119"
  ok    the prominent Majoritet result is still shown
  ok    summary includes link to histogram
  ok    histogram is visible with government
  FAIL  histogram mask follows government
          expected "112", got "16"
  ok    histogram has rendered seat bars
  ok    reset hides histogram in schema 1.3
  ok    reset hides the statistics grid
  ok    reset empties the statistics grid
  ok    schema 1.3 no horizontal overflow
  ok    schema 1.3 no console errors
  ok    schema 1.3 no uncaught exceptions

[schema 1.1 fail-closed] desktop
  ok    builder section is hidden under schema 1.1
  ok    no console errors under schema 1.1
  ok    no uncaught exceptions under schema 1.1

269/285 checks passed

FAIL (16)
"""


class WebsiteChecksGateTests(unittest.TestCase):
    """Test suite ensuring website acceptance gate is strict and fail-closed."""

    def test_baseline_matcher_matches_exact_baseline(self) -> None:
        self.assertTrue(
            ea.is_known_baseline_government_builder_failure(EXACT_BASELINE_OUTPUT, 1)
        )

    def test_baseline_matcher_rejects_exit_code_zero(self) -> None:
        self.assertFalse(
            ea.is_known_baseline_government_builder_failure(EXACT_BASELINE_OUTPUT, 0)
        )

    def test_baseline_matcher_rejects_exit_code_two(self) -> None:
        self.assertFalse(
            ea.is_known_baseline_government_builder_failure(EXACT_BASELINE_OUTPUT, 2)
        )

    def test_baseline_matcher_rejects_missing_pass_count(self) -> None:
        tampered = EXACT_BASELINE_OUTPUT.replace("269/285 checks passed", "270/285 checks passed")
        self.assertFalse(
            ea.is_known_baseline_government_builder_failure(tampered, 1)
        )

    def test_baseline_matcher_rejects_extra_failure(self) -> None:
        extra_failure = "  FAIL  extra unexpected assertion failed\n" + EXACT_BASELINE_OUTPUT
        self.assertFalse(
            ea.is_known_baseline_government_builder_failure(extra_failure, 1)
        )

    def test_baseline_matcher_rejects_missing_failure(self) -> None:
        missing_failure = EXACT_BASELINE_OUTPUT.replace(
            "  FAIL  histogram mask follows government\n          expected \"112\", got \"16\"",
            "  ok    histogram mask follows government",
            1,
        )
        self.assertFalse(
            ea.is_known_baseline_government_builder_failure(missing_failure, 1)
        )

    def test_baseline_matcher_rejects_different_failure_name(self) -> None:
        different_failure = EXACT_BASELINE_OUTPUT.replace(
            "FAIL  government mask is 112",
            "FAIL  government mask is 84",
            1,
        )
        self.assertFalse(
            ea.is_known_baseline_government_builder_failure(different_failure, 1)
        )

    def test_baseline_matcher_rejects_console_error(self) -> None:
        with_console_error = EXACT_BASELINE_OUTPUT.replace(
            "ok    schema 1.3 no console errors",
            "FAIL  schema 1.3 no console errors",
        )
        self.assertFalse(
            ea.is_known_baseline_government_builder_failure(with_console_error, 1)
        )

    def test_baseline_matcher_rejects_uncaught_exception(self) -> None:
        with_exception = EXACT_BASELINE_OUTPUT.replace(
            "ok    schema 1.3 no uncaught exceptions",
            "FAIL  schema 1.3 no uncaught exceptions",
        )
        self.assertFalse(
            ea.is_known_baseline_government_builder_failure(with_exception, 1)
        )

    @patch("scripts.election_automation._run_command")
    @patch("scripts.election_automation._run_command_with_output")
    def test_known_baseline_condition_handled_when_exact(
        self,
        mock_run_output: MagicMock,
        mock_run_cmd: MagicMock,
    ) -> None:
        mock_run_output.return_value = (1, EXACT_BASELINE_OUTPUT)
        result = ea._run_website_checks_guarded(Path("/fake/site"))

        self.assertEqual(result["status"], "PASS")
        self.assertIn(
            "government-builder browser smoke (narrowly allowlisted known baseline drag-and-drop: 16 schema 1.3 assertions)",
            result["checks"],
        )

    @patch("scripts.election_automation._run_command")
    @patch("scripts.election_automation._run_command_with_output")
    def test_additional_failure_causes_gate_to_fail(
        self,
        mock_run_output: MagicMock,
        mock_run_cmd: MagicMock,
    ) -> None:
        extra_failure_output = "  FAIL  extra assertion failed\n" + EXACT_BASELINE_OUTPUT
        mock_run_output.return_value = (1, extra_failure_output)

        with self.assertRaises(AutomationError):
            ea._run_website_checks_guarded(Path("/fake/site"))

    @patch("scripts.election_automation._run_command")
    @patch("scripts.election_automation._run_command_with_output")
    def test_timeout_causes_gate_to_fail(
        self,
        mock_run_output: MagicMock,
        mock_run_cmd: MagicMock,
    ) -> None:
        mock_run_output.side_effect = AutomationError(
            "website command timed out: government-builder.smoke.mjs after 90.000s"
        )

        with self.assertRaises(AutomationError) as ctx:
            ea._run_website_checks_guarded(Path("/fake/site"))
        self.assertIn("timed out", str(ctx.exception))

    @patch("scripts.election_automation._run_command")
    @patch("scripts.election_automation._run_command_with_output")
    def test_different_government_builder_failure_causes_gate_to_fail(
        self,
        mock_run_output: MagicMock,
        mock_run_cmd: MagicMock,
    ) -> None:
        diff_failure_output = EXACT_BASELINE_OUTPUT.replace(
            "FAIL  government mask is 112",
            "FAIL  different assertion",
            1,
        )
        mock_run_output.return_value = (1, diff_failure_output)

        with self.assertRaises(AutomationError):
            ea._run_website_checks_guarded(Path("/fake/site"))

    @patch("scripts.election_automation._run_command")
    @patch("scripts.election_automation._run_command_with_output")
    def test_forecast_timeseries_failure_remains_fatal(
        self,
        mock_run_output: MagicMock,
        mock_run_cmd: MagicMock,
    ) -> None:
        mock_run_cmd.side_effect = AutomationError(
            "website command failed: forecast-timeseries.smoke.mjs (exit code 1)"
        )

        with self.assertRaises(AutomationError):
            ea._run_website_checks_guarded(Path("/fake/site"))

    @patch("scripts.election_automation._run_command")
    @patch("scripts.election_automation._run_command_with_output")
    def test_successful_website_checks_behave_as_before(
        self,
        mock_run_output: MagicMock,
        mock_run_cmd: MagicMock,
    ) -> None:
        mock_run_output.return_value = (0, "285/285 checks passed\nPASS\n")
        result = ea._run_website_checks_guarded(Path("/fake/site"))

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(
            result["checks"],
            [
                "jekyll build",
                "forecast-timeseries browser smoke",
                "government-builder browser smoke",
                "zero console errors and no mobile horizontal overflow (smoke assertions)",
            ],
        )


if __name__ == "__main__":
    unittest.main()
