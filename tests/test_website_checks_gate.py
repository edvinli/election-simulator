"""Tests for strict website acceptance checks without any baseline bypass."""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from scripts import election_automation as ea
from scripts.election_automation_base import AutomationError, run_website_checks


class StrictWebsiteChecksGateTests(unittest.TestCase):
    """Test suite ensuring website acceptance gate is 100% strict and fail-closed."""

    @patch("scripts.election_automation._run_command")
    def test_successful_website_checks_pass_cleanly(
        self,
        mock_run_cmd: MagicMock,
    ) -> None:
        result = ea.run_website_checks(Path("/fake/site"))

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
        self.assertEqual(mock_run_cmd.call_count, 3)

    @patch("scripts.election_automation._run_command")
    def test_government_builder_failure_is_strictly_fatal(
        self,
        mock_run_cmd: MagicMock,
    ) -> None:
        def side_effect(cmd, **kwargs):
            if "government-builder.smoke.mjs" in cmd[1]:
                raise AutomationError(
                    "website command failed: government-builder.smoke.mjs (exit code 1) after 22.100s"
                )
            return None

        mock_run_cmd.side_effect = side_effect

        with self.assertRaises(AutomationError) as ctx:
            ea.run_website_checks(Path("/fake/site"))
        self.assertIn("government-builder.smoke.mjs", str(ctx.exception))

    @patch("scripts.election_automation._run_command")
    def test_government_builder_timeout_is_fatal(
        self,
        mock_run_cmd: MagicMock,
    ) -> None:
        def side_effect(cmd, **kwargs):
            if "government-builder.smoke.mjs" in cmd[1]:
                raise AutomationError(
                    "website command timed out: government-builder.smoke.mjs after 90.000s"
                )
            return None

        mock_run_cmd.side_effect = side_effect

        with self.assertRaises(AutomationError) as ctx:
            ea.run_website_checks(Path("/fake/site"))
        self.assertIn("timed out", str(ctx.exception))

    @patch("scripts.election_automation._run_command")
    def test_forecast_timeseries_failure_remains_fatal(
        self,
        mock_run_cmd: MagicMock,
    ) -> None:
        def side_effect(cmd, **kwargs):
            if "forecast-timeseries.smoke.mjs" in cmd[1]:
                raise AutomationError(
                    "website command failed: forecast-timeseries.smoke.mjs (exit code 1) after 15.000s"
                )
            return None

        mock_run_cmd.side_effect = side_effect

        with self.assertRaises(AutomationError) as ctx:
            ea.run_website_checks(Path("/fake/site"))
        self.assertIn("forecast-timeseries.smoke.mjs", str(ctx.exception))

    @patch("scripts.election_automation._run_command")
    def test_jekyll_build_failure_remains_fatal(
        self,
        mock_run_cmd: MagicMock,
    ) -> None:
        def side_effect(cmd, **kwargs):
            if "jekyll" in cmd[0]:
                raise AutomationError("website command failed: jekyll build (exit code 1)")
            return None

        mock_run_cmd.side_effect = side_effect

        with self.assertRaises(AutomationError) as ctx:
            ea.run_website_checks(Path("/fake/site"))
        self.assertIn("jekyll build", str(ctx.exception))

    def test_no_baseline_allowance_attributes_exist_on_facade(self) -> None:
        """Confirm that all legacy allowance functions and constants are completely removed."""
        self.assertFalse(hasattr(ea, "is_known_baseline_government_builder_failure"))
        self.assertFalse(hasattr(ea, "KNOWN_BASELINE_GB_FAILURES"))
        self.assertFalse(hasattr(ea, "_run_website_checks_guarded"))


if __name__ == "__main__":
    unittest.main()
