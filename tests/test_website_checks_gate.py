"""Tests for strict website acceptance checks without any baseline bypass."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts import election_automation as ea
from scripts import election_automation_base as base
from scripts.election_automation_base import AutomationError, run_website_checks

try:  # pragma: no cover - import shape depends on how the suite is invoked
    from ._website_repo import SKIP_REASON, website_repo
except ImportError:  # pragma: no cover
    from tests._website_repo import SKIP_REASON, website_repo


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
                "party-timeseries browser smoke (--real-artifact)",
                "zero console errors and no mobile horizontal overflow (smoke assertions)",
            ],
        )
        self.assertEqual(mock_run_cmd.call_count, 4)

    @patch("scripts.election_automation._run_command")
    def test_party_view_is_gated_in_real_artifact_mode(
        self,
        mock_run_cmd: MagicMock,
    ) -> None:
        """The gate must run the party suite against the generated artifact.

        Its default mode overlays a committed fixture onto a copy of the site,
        so the fixture-mode command would gate every publication against the
        fixture instead of the artifact the run just produced. Only the
        --real-artifact form reads the history the site actually ships.
        """

        ea.run_website_checks(Path("/fake/site"))

        commands = [call.args[0] for call in mock_run_cmd.call_args_list]
        party = [cmd for cmd in commands if "party-timeseries.smoke.mjs" in cmd[1]]
        self.assertEqual(len(party), 1, commands)
        self.assertEqual(
            party[0],
            ["node", "browser-tests/party-timeseries.smoke.mjs", "_site", "--real-artifact"],
        )
        # Every suite keeps its path at argv index 1, which is what the
        # fail-closed assertions below match on.
        for cmd in commands[1:]:
            self.assertTrue(cmd[1].startswith("browser-tests/"), cmd)
            self.assertTrue(cmd[1].endswith(".mjs"), cmd)

    @patch("scripts.election_automation._run_command")
    def test_party_real_artifact_failure_is_strictly_fatal(
        self,
        mock_run_cmd: MagicMock,
    ) -> None:
        """An artifact that cannot back the party view must stop publication."""

        def side_effect(cmd, **kwargs):
            if "party-timeseries.smoke.mjs" in cmd[1]:
                raise AutomationError(
                    "website command failed: party-timeseries.smoke.mjs "
                    "(exit code 1) after 41.300s"
                )
            return None

        mock_run_cmd.side_effect = side_effect

        with self.assertRaises(AutomationError) as ctx:
            ea.run_website_checks(Path("/fake/site"))
        self.assertIn("party-timeseries.smoke.mjs", str(ctx.exception))

    @patch("scripts.election_automation._run_command")
    def test_party_real_artifact_timeout_is_fatal(
        self,
        mock_run_cmd: MagicMock,
    ) -> None:
        def side_effect(cmd, **kwargs):
            if "party-timeseries.smoke.mjs" in cmd[1]:
                raise AutomationError(
                    "website command timed out: party-timeseries.smoke.mjs after 90.000s"
                )
            return None

        mock_run_cmd.side_effect = side_effect

        with self.assertRaises(AutomationError) as ctx:
            ea.run_website_checks(Path("/fake/site"))
        self.assertIn("timed out", str(ctx.exception))

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


class WebsitePushGateTests(unittest.TestCase):
    """The second gate tier: everything a publication affects, before the push.

    The tier exists for ordering reasons. ``run_website_checks`` runs before
    the simulator's own durable commit, so suites added there delay
    certification of the forecast itself. These run after the forecast is
    certified and before the website is pushed, so a failure costs the website
    an update and costs the forecast nothing.
    """

    def _built_site(self) -> Path:
        root = Path(tempfile.mkdtemp(prefix="push-gate-site-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        (root / "_site").mkdir()
        return root

    @patch("scripts.election_automation._run_command")
    def test_push_gate_runs_every_publication_affected_suite(
        self,
        mock_run_cmd: MagicMock,
    ) -> None:
        result = ea.run_website_push_checks(self._built_site())

        self.assertEqual(result["status"], "PASS")
        commands = [call.args[0] for call in mock_run_cmd.call_args_list]
        self.assertEqual(commands, [list(c) for c in ea.WEBSITE_PUSH_GATE_COMMANDS])
        # No build: the caller has already built this tree for the first tier,
        # and rebuilding would charge every publication for nothing.
        self.assertNotIn("jekyll", [cmd[0] for cmd in commands])
        # Every suite keeps its path at argv index 1, which is what the
        # fail-closed assertions below match on.
        for cmd in commands:
            self.assertTrue(cmd[1].startswith("browser-tests/"), cmd)

    @patch("scripts.election_automation._run_command")
    def test_push_gate_refuses_an_unbuilt_tree(self, mock_run_cmd: MagicMock) -> None:
        """A missing _site would make every suite fail for the wrong reason."""

        root = Path(tempfile.mkdtemp(prefix="push-gate-unbuilt-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)

        with self.assertRaises(AutomationError) as ctx:
            ea.run_website_push_checks(root)
        self.assertIn("already-built", str(ctx.exception))
        mock_run_cmd.assert_not_called()

    @patch("scripts.election_automation._run_command")
    def test_every_push_gate_suite_failure_is_fatal(
        self,
        mock_run_cmd: MagicMock,
    ) -> None:
        site = self._built_site()
        for command in ea.WEBSITE_PUSH_GATE_COMMANDS:
            suite = command[1]
            with self.subTest(suite=suite):
                def side_effect(cmd, _suite=suite, **kwargs):
                    if cmd[1] == _suite:
                        raise AutomationError(
                            f"website command failed: {_suite.rsplit('/', 1)[-1]} "
                            "(exit code 1) after 12.000s"
                        )
                    return None

                mock_run_cmd.side_effect = side_effect
                with self.assertRaises(AutomationError) as ctx:
                    ea.run_website_push_checks(site)
                self.assertIn(suite.rsplit("/", 1)[-1], str(ctx.exception))

    def test_changes_baseline_is_gated_at_all(self) -> None:
        """The suite whose absence let a provenance regression reach master.

        A forecast sync replaces the history artifact, which is exactly what
        `changes-baseline` reads for the hero's polling provenance. It was in
        neither tier, so the only thing that ever ran it was post-push CI --
        which cannot prevent a bad publication.
        """

        gated = {
            command[1].rsplit("/", 1)[-1]
            for command in ea.WEBSITE_GATE_COMMANDS + ea.WEBSITE_PUSH_GATE_COMMANDS
        }
        self.assertIn("changes-baseline.smoke.mjs", gated)

    @patch("scripts.election_automation._run_command")
    def test_push_gate_worst_case_cannot_reach_the_publish_job_timeout(
        self,
        mock_run_cmd: MagicMock,
    ) -> None:
        """The benchmark's protected interval is derived from that timeout.

        The capture protocol assumes publish can hold the production lock for
        up to the job's 120-minute timeout and protects 18:30Z-22:00Z on that
        basis. If this tier could grow the worst case into the timeout it would
        move the assumption, so the ceiling is asserted rather than trusted.
        """

        ea.run_website_push_checks(self._built_site())

        timeouts = {
            call.kwargs["timeout_seconds"] for call in mock_run_cmd.call_args_list
        }
        self.assertEqual(timeouts, {ea.WEBSITE_PUSH_GATE_TIMEOUT_SECONDS})
        worst_case = len(ea.WEBSITE_PUSH_GATE_COMMANDS) * ea.WEBSITE_PUSH_GATE_TIMEOUT_SECONDS
        self.assertLess(worst_case, 120 * 60)
        # And tighter than the first tier, whose suites are the heavy ones.
        self.assertLess(
            ea.WEBSITE_PUSH_GATE_TIMEOUT_SECONDS, base.BROWSER_SMOKE_TIMEOUT_SECONDS)

    def test_the_two_tiers_do_not_overlap(self) -> None:
        first = {command[1] for command in ea.WEBSITE_GATE_COMMANDS}
        second = {command[1] for command in ea.WEBSITE_PUSH_GATE_COMMANDS}
        self.assertEqual(first & second, set())

    def test_the_push_gate_runs_after_certification_and_before_the_push(self) -> None:
        """Ordering is the whole point of the split, so assert it structurally.

        Invoking a real publication here would need a full simulator; what has
        to be guaranteed is position, and position is visible in the source:
        the gate call must sit between the simulator's own commit and the
        website's.
        """

        source = Path(base.__file__).read_text(encoding="utf-8")
        certify = source.index('f"chore: publish election forecast {as_of.isoformat()}"')
        gate = source.index('website["push_gate"] = website_push_check(staged_site)')
        sync = source.index('f"chore: sync election forecast {generation}"')
        self.assertLess(certify, gate, "the gate must not delay certification")
        self.assertLess(gate, sync, "the gate must precede the website push")

    def test_recovery_push_gate_failure_prevents_the_website_commit(self) -> None:
        """The recovery path mirrors an already-certified generation.

        Nothing is left to certify there, so the gate simply runs before
        anything is installed or pushed -- and a failure must stop the commit.
        """

        commits = MagicMock()
        failing = MagicMock(side_effect=AutomationError(
            "website command failed: changes-baseline.smoke.mjs (exit code 1) after 13.000s"))

        with patch.object(base, "_assert_clean"), \
                patch.object(base, "_copy_site_tree"), \
                patch.object(base, "_website_needs_recovery",
                             return_value=(True, {"generation": "20260909T044949Z-db320582"})), \
                patch.object(base, "_stage_site", return_value={"status": "PASS"}), \
                patch.object(base, "_install_site_outputs") as install, \
                patch.object(base, "_git_commit_paths", commits):
            with self.assertRaises(AutomationError) as ctx:
                base._recover_website_from_source(
                    source_repo=Path("/fake/simulator"),
                    site_repo=Path("/fake/site"),
                    website_check_fn=lambda _root: {"status": "PASS"},
                    website_push_check_fn=failing,
                    commit=True,
                    push=True,
                )

        self.assertIn("changes-baseline.smoke.mjs", str(ctx.exception))
        failing.assert_called_once()
        install.assert_not_called()
        commits.assert_not_called()


class WebsiteGateMatchesWebsiteSelectorTests(unittest.TestCase):
    """The gate's membership is the website's decision, not ours.

    `browser-tests/select-suites.mjs` already answers "which suites can this
    change affect", and the website repository owns that mapping. Restating it
    here by hand would let the two drift, and a gate that silently narrows is
    worse than no gate. So assert the equality directly against a real website
    checkout, and skip -- rather than pass -- when none is opted in.
    """

    #: Paths a forecast publication writes in the website repository.
    PUBLICATION_PATHS = (
        "files/election-simulator/current.json",
        "files/election-simulator/history/coalition-timeseries.json",
    )

    def test_both_tiers_together_are_exactly_the_affected_suites(self) -> None:
        website = website_repo()
        selector = website / "browser-tests" / "select-suites.mjs"
        if not selector.is_file():
            self.skipTest(SKIP_REASON)

        completed = subprocess.run(
            ["node", "browser-tests/select-suites.mjs", "--changed", *self.PUBLICATION_PATHS],
            cwd=website,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        affected = {
            entry["suite"] for entry in json.loads(completed.stdout)["include"]
        }
        gated = {
            command[1].rsplit("/", 1)[-1]
            for command in ea.WEBSITE_GATE_COMMANDS + ea.WEBSITE_PUSH_GATE_COMMANDS
        }
        self.assertEqual(
            gated,
            affected,
            "the publication gate and the website's own suite selector disagree; "
            "update WEBSITE_PUSH_GATE_COMMANDS rather than narrowing the gate",
        )
