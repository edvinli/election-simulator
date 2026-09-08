"""The fallback trigger for a missing mandatory daily publication.

GitHub's cron is best-effort, so the 04:00Z tick that carries the mandatory
daily recalculation can arrive late or not at all.  An external scheduler
therefore dispatches ``publish_if_stale``, and these tests pin the five
behaviours that make that safe:

1. a late primary trigger costs a no-op, not a second publication;
2. a missing primary trigger produces the mandatory recalculation;
3. a duplicate late cron arriving after the fallback publishes nothing;
4. the ordinary daily run is unchanged;
5. the fallback fails closed -- kill switch, benchmark window, unreadable
   pointer.

The layering is deliberate.  The policy functions are pure and are tested
without a repository, a network or a clock; the ``run_automation`` tests then
patch only the two pointer reads, because building a fully certified
publication pair would test the publication path rather than the fallback and
would say nothing extra about these decisions.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.election_automation import (
    DAILY_SCHEDULE_UTC,
    INTRADAY_SCHEDULE_UTC,
    AutomationError,
    PollingRefresh,
    automation_enabled_for_event,
    classify_run_type,
    resolve_mode,
    run_automation,
    should_publish,
)
from scripts.publication_fallback import (
    CAPTURE_COMPLETION_GRACE,
    FALLBACK_RUN_TYPE,
    PUBLICATION_MAX_RUNTIME,
    benchmark_protected_interval,
    benchmark_window_conflict,
    daily_publication_satisfied,
    stockholm_date_of,
)
from scripts.prospective_benchmark_2026.time_rules import (
    FINAL_CAPTURE_DATE,
    FIRST_CAPTURE_DATE,
    scheduled_cutoff,
)


TODAY = date(2026, 9, 7)
# 06:45 Europe/Stockholm: the first fallback tick, 45 minutes after the daily.
FIRST_TICK = datetime(2026, 9, 7, 4, 45, tzinfo=timezone.utc)
YESTERDAY_EVENING = datetime(2026, 9, 6, 18, 30, tzinfo=timezone.utc)
TODAY_MORNING = datetime(2026, 9, 7, 4, 2, tzinfo=timezone.utc)


def _forbidden_refresh(*args, **kwargs):
    raise AssertionError("acquisition must not run in this scenario")


def _seed_processed_inputs(root: Path) -> None:
    """The processed files the summary reads after a successful acquisition.

    Only the Poll-of-Polls timeseries needs real shape: the run reports its
    latest observation date.  Acquisition itself is replaced in these tests,
    so nothing here has to be a plausible polling snapshot.
    """

    timeseries = root / "data/processed/pollofpolls/pollofpolls_timeseries.csv"
    timeseries.parent.mkdir(parents=True, exist_ok=True)
    timeseries.write_text("date,M\n2026-09-05,20\n2026-09-06,21\n", encoding="utf-8")


class _Reached(RuntimeError):
    """Raised from a stubbed acquisition to mark the boundary as reached."""


class RunTypeTests(unittest.TestCase):
    """A fallback dispatch must be distinguishable from every other trigger."""

    def test_fallback_mode_has_its_own_run_type(self) -> None:
        self.assertEqual(
            classify_run_type(event_name="workflow_dispatch", mode="publish_if_stale"),
            FALLBACK_RUN_TYPE,
        )
        # Every pre-existing classification is untouched.
        self.assertEqual(classify_run_type(event_name="workflow_dispatch"), "MANUAL")
        self.assertEqual(
            classify_run_type(event_name="workflow_dispatch", mode="publish"), "MANUAL"
        )
        self.assertEqual(classify_run_type(schedule=DAILY_SCHEDULE_UTC), "DAILY")
        self.assertEqual(classify_run_type(schedule=INTRADAY_SCHEDULE_UTC), "POLL_CHANGE")

    def test_fallback_is_a_mutating_mode(self) -> None:
        self.assertEqual(
            resolve_mode(
                event_name="workflow_dispatch",
                mode="publish_if_stale",
                commit=True,
                push=True,
            ),
            "publish_if_stale",
        )
        with self.assertRaises(AutomationError):
            resolve_mode(
                event_name="workflow_dispatch",
                mode="publish_if_stale",
                commit=False,
                push=False,
            )

    def test_the_kill_switch_bypass_is_closed_for_the_fallback(self) -> None:
        """An operator dispatch may bypass the switch; an unattended one may not.

        This is the whole security question the fallback raises: without it,
        adding an external trigger would hand the automated path a way around
        an operator's "stop publishing".
        """

        self.assertTrue(
            automation_enabled_for_event(event_name="workflow_dispatch", enabled="false")
        )
        self.assertTrue(
            automation_enabled_for_event(
                event_name="workflow_dispatch", enabled="false", mode="publish"
            )
        )
        self.assertFalse(
            automation_enabled_for_event(
                event_name="workflow_dispatch", enabled="false", mode="publish_if_stale"
            )
        )
        self.assertTrue(
            automation_enabled_for_event(
                event_name="workflow_dispatch", enabled="true", mode="publish_if_stale"
            )
        )
        # Fail-closed on a missing repository variable, as for a cron tick.
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(
                automation_enabled_for_event(
                    event_name="workflow_dispatch", mode="publish_if_stale"
                )
            )


class LivenessTests(unittest.TestCase):
    """Has today's mandatory recalculation reached the public site?"""

    def test_both_sides_must_be_dated_today(self) -> None:
        self.assertTrue(
            daily_publication_satisfied(
                source_generated_at=TODAY_MORNING,
                site_generated_at=TODAY_MORNING,
                today=TODAY,
            )
        )
        self.assertFalse(
            daily_publication_satisfied(
                source_generated_at=YESTERDAY_EVENING,
                site_generated_at=YESTERDAY_EVENING,
                today=TODAY,
            )
        )

    def test_a_published_but_unsynced_generation_is_not_satisfied(self) -> None:
        """The condition the fallback exists for includes a failed sync.

        A simulation that certified a generation and then failed to push it to
        the website leaves the *public* forecast stale, which is what the
        requirement is about.
        """

        self.assertFalse(
            daily_publication_satisfied(
                source_generated_at=TODAY_MORNING,
                site_generated_at=YESTERDAY_EVENING,
                today=TODAY,
            )
        )

    def test_an_unreadable_pointer_reads_as_stale_not_as_fresh(self) -> None:
        """The check must fail towards a redundant run, never towards silence."""

        self.assertFalse(
            daily_publication_satisfied(
                source_generated_at=None, site_generated_at=TODAY_MORNING, today=TODAY
            )
        )
        self.assertFalse(
            daily_publication_satisfied(
                source_generated_at=TODAY_MORNING, site_generated_at=None, today=TODAY
            )
        )
        self.assertFalse(
            daily_publication_satisfied(
                source_generated_at=None, site_generated_at=None, today=TODAY
            )
        )

    def test_the_date_is_the_stockholm_date_not_the_utc_one(self) -> None:
        """22:30Z is already the next Stockholm day during the campaign."""

        late = datetime(2026, 9, 6, 22, 30, tzinfo=timezone.utc)
        self.assertEqual(stockholm_date_of(late), date(2026, 9, 7))
        self.assertTrue(
            daily_publication_satisfied(
                source_generated_at=late, site_generated_at=late, today=date(2026, 9, 7)
            )
        )
        with self.assertRaises(ValueError):
            stockholm_date_of(datetime(2026, 9, 7, 4, 0))


class BenchmarkWindowTests(unittest.TestCase):
    """The fallback must never hold the production lock across a cutoff.

    The prospective benchmark shares ``election-simulator-production`` by
    design, and amendment 005 records that a capture which is not ready before
    its frozen cutoff archives LATE_EXCLUDED and fails the run.  A publication
    holding that lock could cause exactly that, so the fallback stands down.
    """

    def test_the_interval_is_derived_from_the_frozen_protocol(self) -> None:
        start, end = benchmark_protected_interval(date(2026, 9, 7))
        cutoff = scheduled_cutoff(date(2026, 9, 7))
        self.assertEqual(end, cutoff + CAPTURE_COMPLETION_GRACE)
        # The pre-warm start is 22:30 Stockholm, one hour before the cutoff.
        self.assertEqual(start, cutoff - timedelta(hours=1) - PUBLICATION_MAX_RUNTIME)

    def test_a_dispatch_inside_the_window_conflicts(self) -> None:
        cutoff = scheduled_cutoff(date(2026, 9, 7))
        self.assertEqual(
            benchmark_window_conflict(cutoff - timedelta(minutes=30)), date(2026, 9, 7)
        )
        self.assertEqual(benchmark_window_conflict(cutoff), date(2026, 9, 7))
        # Just after the cutoff the capture is still running and still holds
        # the lock.
        self.assertEqual(
            benchmark_window_conflict(cutoff + timedelta(minutes=10)), date(2026, 9, 7)
        )

    def test_the_actual_fallback_ticks_never_conflict(self) -> None:
        """The guard is an invariant, not a restatement of the schedule."""

        for hour, minute in ((4, 45), (5, 30), (6, 30)):
            for day in range(4, 13):
                tick = datetime(2026, 9, day, hour, minute, tzinfo=timezone.utc)
                self.assertIsNone(
                    benchmark_window_conflict(tick),
                    f"fallback tick {tick.isoformat()} collides with a capture",
                )

    def test_dates_outside_the_frozen_window_hold_no_lock(self) -> None:
        before = scheduled_cutoff(FIRST_CAPTURE_DATE) - timedelta(days=1)
        after = scheduled_cutoff(FINAL_CAPTURE_DATE) + timedelta(days=1)
        self.assertIsNone(benchmark_window_conflict(before))
        self.assertIsNone(benchmark_window_conflict(after))

    def test_the_window_is_matched_across_local_midnight(self) -> None:
        """A slot's interval ends at local midnight, on the next local date.

        The closing instant therefore falls on the following Stockholm date,
        which is why the lookback considers the previous date as well as the
        current one.  A minute later belongs to nobody.
        """

        boundary = scheduled_cutoff(date(2026, 9, 7)) + CAPTURE_COMPLETION_GRACE
        self.assertEqual(stockholm_date_of(boundary), date(2026, 9, 8))
        self.assertEqual(benchmark_window_conflict(boundary), date(2026, 9, 7))
        self.assertIsNone(benchmark_window_conflict(boundary + timedelta(minutes=1)))


class ShouldPublishTests(unittest.TestCase):
    """The publication decision, trigger by trigger."""

    def test_scenario_missing_primary_trigger_publishes(self) -> None:
        """No publication today: the fallback performs the recalculation."""

        self.assertTrue(
            should_publish(
                FALLBACK_RUN_TYPE,
                model_inputs_changed=False,
                daily_already_satisfied=False,
            )
        )

    def test_scenario_late_primary_trigger_is_a_no_op(self) -> None:
        """The daily arrived before the tick, so there is nothing to cover."""

        self.assertFalse(
            should_publish(
                FALLBACK_RUN_TYPE,
                model_inputs_changed=False,
                daily_already_satisfied=True,
            )
        )

    def test_scenario_duplicate_late_cron_after_the_fallback_is_a_no_op(self) -> None:
        """The requirement: the fallback must not cause a double publication."""

        self.assertFalse(
            should_publish(
                "DAILY", model_inputs_changed=False, daily_already_satisfied=True
            )
        )

    def test_scenario_successful_normal_daily_run_is_unchanged(self) -> None:
        self.assertTrue(
            should_publish(
                "DAILY", model_inputs_changed=False, daily_already_satisfied=False
            )
        )
        self.assertTrue(
            should_publish(
                "DAILY", model_inputs_changed=True, daily_already_satisfied=False
            )
        )

    def test_a_covered_daily_still_publishes_real_news(self) -> None:
        """Idempotence must not swallow a changed input or a pending marker.

        This is what keeps the change from weakening publication semantics:
        the invariant is one mandatory recalculation per Stockholm day, and
        anything with something new to say still publishes.
        """

        self.assertTrue(
            should_publish(
                "DAILY", model_inputs_changed=True, daily_already_satisfied=True
            )
        )
        self.assertTrue(
            should_publish(
                "DAILY",
                model_inputs_changed=False,
                daily_already_satisfied=True,
                pending_publication=True,
            )
        )
        self.assertTrue(
            should_publish(
                FALLBACK_RUN_TYPE,
                model_inputs_changed=True,
                daily_already_satisfied=True,
            )
        )
        self.assertTrue(
            should_publish(
                FALLBACK_RUN_TYPE,
                model_inputs_changed=False,
                daily_already_satisfied=True,
                pending_publication=True,
            )
        )

    def test_untouched_decisions(self) -> None:
        # An operator dispatch is always honoured.
        self.assertTrue(
            should_publish(
                "MANUAL", model_inputs_changed=False, daily_already_satisfied=True
            )
        )
        # A poll-change check is still gated on a change.
        self.assertFalse(
            should_publish(
                "POLL_CHANGE", model_inputs_changed=False, daily_already_satisfied=False
            )
        )
        self.assertTrue(
            should_publish(
                "POLL_CHANGE", model_inputs_changed=True, daily_already_satisfied=False
            )
        )
        # A probe never publishes, whatever the trigger.
        self.assertFalse(
            should_publish(
                FALLBACK_RUN_TYPE,
                model_inputs_changed=True,
                mode="probe",
                daily_already_satisfied=False,
            )
        )


class FallbackRunTests(unittest.TestCase):
    """The same scenarios through ``run_automation``.

    Only the two pointer reads are patched.  Everything else -- run-type
    classification, the kill switch, the benchmark guard, the publication
    decision and the acquisition boundary -- is the production path.
    Acquisition is stubbed to raise at the moment it is entered, so a test can
    assert whether the real work would have started without performing it.
    """

    def _run(self, *, pointers, now, mode="publish_if_stale",
             event_name="workflow_dispatch", schedule=None, enabled="true",
             acquisition=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _seed_processed_inputs(root)
            with patch(
                "scripts.election_automation._certified_generated_at",
                side_effect=lambda _dir: pointers,
            ), patch(
                "scripts.election_automation.refresh_polling_snapshot",
                side_effect=acquisition or _forbidden_refresh,
            ):
                return run_automation(
                    root,
                    site_repo=root,
                    event_name=event_name,
                    schedule=schedule,
                    mode=mode,
                    commit=mode in {"publish", "publish_if_stale"},
                    now=now,
                    automation_enabled=enabled,
                )

    def _reaching_acquisition(self, **kwargs) -> bool:
        """Did the run get as far as acquiring polling data?"""

        reached = False

        def marker(*args, **kwargs_inner):
            nonlocal reached
            reached = True
            raise _Reached("acquisition boundary reached")

        try:
            self._run(acquisition=marker, **kwargs)
        except _Reached:
            pass
        return reached

    # --- scenario 2: missing primary trigger ----------------------------
    def test_missing_primary_trigger_does_the_real_work(self) -> None:
        self.assertTrue(
            self._reaching_acquisition(pointers=YESTERDAY_EVENING, now=FIRST_TICK),
            "a missing daily must be acquired and published",
        )

    # --- scenario 1: late primary trigger -------------------------------
    def test_late_primary_trigger_publishes_nothing(self) -> None:
        """The daily arrived before the tick, so the fallback is a no-op.

        It still acquires -- that is how it learns whether anything changed --
        but the publication decision comes out negative.
        """

        result = self._run(
            pointers=TODAY_MORNING,
            now=FIRST_TICK,
            acquisition=_unchanged_polling,
        )
        self.assertEqual(result.status, "SOURCE_CHECKED")
        self.assertEqual(result.summary.deployment_status, "NO_PUBLICATION_NEEDED")
        self.assertEqual(result.summary.run_type, FALLBACK_RUN_TYPE)
        self.assertEqual(result.summary.daily_publication_status, "SATISFIED_TODAY")
        self.assertEqual(result.summary.publication_generation, "NONE")

    # --- scenario 3: duplicate late cron after the fallback --------------
    def test_a_late_daily_cron_after_the_fallback_publishes_nothing(self) -> None:
        result = self._run(
            pointers=TODAY_MORNING,
            now=datetime(2026, 9, 7, 7, 30, tzinfo=timezone.utc),
            mode="publish",
            event_name="schedule",
            schedule=DAILY_SCHEDULE_UTC,
            acquisition=_unchanged_polling,
        )
        self.assertEqual(result.status, "SOURCE_CHECKED")
        self.assertEqual(result.summary.deployment_status, "NO_PUBLICATION_NEEDED")
        self.assertEqual(result.summary.run_type, "DAILY")
        self.assertEqual(result.summary.daily_publication_status, "SATISFIED_TODAY")

    # --- scenario 4: successful normal run -------------------------------
    def test_the_ordinary_daily_run_is_unchanged(self) -> None:
        self.assertTrue(
            self._reaching_acquisition(
                pointers=YESTERDAY_EVENING,
                now=datetime(2026, 9, 7, 4, 0, tzinfo=timezone.utc),
                mode="publish",
                event_name="schedule",
                schedule=DAILY_SCHEDULE_UTC,
            )
        )

    # --- scenario 5: the fallback itself fails ---------------------------
    def test_the_kill_switch_stops_the_fallback_before_acquisition(self) -> None:
        result = self._run(pointers=YESTERDAY_EVENING, now=FIRST_TICK, enabled="false")
        self.assertEqual(result.status, "DISABLED_BY_REPOSITORY_KILL_SWITCH")
        self.assertEqual(result.summary.run_type, FALLBACK_RUN_TYPE)
        self.assertEqual(
            result.summary.deployment_status, "DISABLED_BY_REPOSITORY_KILL_SWITCH"
        )
        self.assertIn(f"Run type: {FALLBACK_RUN_TYPE}", result.summary.render())

    def test_an_operator_dispatch_still_bypasses_the_kill_switch(self) -> None:
        """The pre-existing operator escape hatch is preserved."""

        self.assertTrue(
            self._reaching_acquisition(
                pointers=TODAY_MORNING,
                now=FIRST_TICK,
                mode="publish",
                enabled="false",
            )
        )

    def test_a_broken_pointer_still_recalculates_rather_than_going_quiet(self) -> None:
        """Scenario 5, the interesting half: the liveness check itself fails.

        An unreadable pointer must not be mistaken for a fresh one, or the
        fallback would stand down exactly when the site is broken.
        """

        self.assertTrue(
            self._reaching_acquisition(pointers=None, now=FIRST_TICK),
        )

    def test_after_election_day_the_fallback_stops_like_every_trigger(self) -> None:
        result = self._run(
            pointers=None,
            now=datetime(2026, 9, 20, 4, 45, tzinfo=timezone.utc),
        )
        self.assertEqual(result.status, "STOPPED_AFTER_ELECTION")


def _unchanged_polling(*args, **kwargs) -> PollingRefresh:
    """A successful acquisition that found nothing new.

    The acquisition mechanics are the existing suite's subject; what these
    tests need from it is only its verdict, so the boundary is replaced rather
    than driven.
    """

    return PollingRefresh(
        status="UNCHANGED",
        changed=False,
        old_hash="a" * 64,
        new_hash="a" * 64,
    )


class BenchmarkDeferralRunTests(unittest.TestCase):
    """A fallback inside the benchmark window stands down before doing work."""

    def _deferral_run(self, *, now, mode, event_name, schedule=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(
                "scripts.election_automation._certified_generated_at",
                side_effect=lambda _dir: YESTERDAY_EVENING,
            ), patch(
                "scripts.election_automation.refresh_polling_snapshot",
                side_effect=_forbidden_refresh,
            ):
                return run_automation(
                    root,
                    site_repo=root,
                    event_name=event_name,
                    schedule=schedule,
                    mode=mode,
                    commit=True,
                    now=now,
                    automation_enabled="true",
                )

    def test_the_fallback_defers_rather_than_taking_the_lock(self) -> None:
        cutoff = scheduled_cutoff(date(2026, 9, 7))
        result = self._deferral_run(
            now=cutoff - timedelta(minutes=20),
            mode="publish_if_stale",
            event_name="workflow_dispatch",
        )
        self.assertEqual(result.status, "DEFERRED_BENCHMARK_WINDOW")
        self.assertEqual(result.summary.deployment_status, "DEFERRED_BENCHMARK_WINDOW")
        self.assertEqual(result.summary.run_type, FALLBACK_RUN_TYPE)
        # A stand-down is a success, not a page-the-operator failure.
        self.assertIsNone(result.summary.failure)

    def test_deferral_happens_before_any_acquisition(self) -> None:
        """``_forbidden_refresh`` would raise if the boundary were crossed."""

        cutoff = scheduled_cutoff(date(2026, 9, 8))
        result = self._deferral_run(
            now=cutoff - timedelta(minutes=5),
            mode="publish_if_stale",
            event_name="workflow_dispatch",
        )
        self.assertEqual(result.status, "DEFERRED_BENCHMARK_WINDOW")

    def test_the_in_process_guard_still_does_not_defer_a_scheduled_cron(self) -> None:
        """``run_automation`` is deliberately unchanged.

        A scheduled cron IS now deferred inside the protected interval, but by
        the ``fallback_preflight`` job, before the publish job requests
        ``election-simulator-production`` -- see
        ``ScheduledPublicationDeferralTests``.  Deferring here instead would be
        useless for the benchmark: this code runs inside the publish job, which
        already holds the lock a capture would be waiting behind.  So the
        in-process behaviour stays as it was, and this test pins that it did
        not drift while the workflow-level gate was added.
        """

        cutoff = scheduled_cutoff(date(2026, 9, 7))
        result = self._deferral_run(
            now=cutoff - timedelta(minutes=20),
            mode="publish",
            event_name="schedule",
            schedule=INTRADAY_SCHEDULE_UTC,
        )
        self.assertNotEqual(result.status, "DEFERRED_BENCHMARK_WINDOW")
        # It went on to acquisition, which the stub refuses -- proof the guard
        # did not intercept it.
        self.assertEqual(result.summary.run_type, "POLL_CHANGE")
        self.assertIn("acquisition must not run", str(result.summary.failure))


PRODUCTION_GROUP = "election-simulator-production"
WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github/workflows/election-simulator-publication.yml"
)


def _workflow_job(workflow: str, name: str) -> str:
    """One top-level workflow job, without requiring a YAML package."""

    lines = workflow.splitlines()
    marker = f"  {name}:"
    try:
        start = lines.index(marker)
    except ValueError as exc:  # pragma: no cover - a missing job is the failure
        raise AssertionError(f"workflow job is missing: {name}") from exc
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.startswith("  ") and not line.startswith("    ") and line.endswith(":"):
            end = index
            break
    return "\n".join(lines[start:end])


class ProductionConcurrencyStructureTests(unittest.TestCase):
    """Where the production lock is taken, and by which job.

    This is a structural test on purpose.  The defect it pins is not a wrong
    value anywhere -- every decision function was correct -- it is *where* the
    decision ran.  ``election-simulator-production`` sat at workflow level, so
    a publish_if_stale run joined the group before any job started and the
    fallback's stand-down check ran while it already held the lock a capture
    was waiting for.  No unit test of that check could see the problem,
    because the check itself was right.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")
        cls.defaults = cls.workflow.split("\njobs:\n", 1)[0]
        cls.preflight = _workflow_job(cls.workflow, "fallback_preflight")
        cls.publish = _workflow_job(cls.workflow, "publish")

    def test_the_production_group_is_not_held_at_workflow_level(self) -> None:
        """The whole point: a run must not join the group before a job runs."""

        # The directive, not the name: the prose above `jobs:` explains why
        # the group moved and necessarily mentions it.
        self.assertNotIn(
            f"group: {PRODUCTION_GROUP}", self.defaults,
            "the production group is held at workflow level again, so the "
            "fallback joins it before it can decide to stand down",
        )

    def test_the_mutating_job_alone_owns_the_production_group(self) -> None:
        self.assertIn(f"group: {PRODUCTION_GROUP}", self.publish)
        self.assertIn("cancel-in-progress: false", self.publish)
        # Exactly one mention in the whole workflow, and it is that job's.
        self.assertEqual(
            self.workflow.count(f"group: {PRODUCTION_GROUP}"), 1,
            "more than one job takes the production lock",
        )
        for name in ("probe", "dry_run", "browser_diagnostic", "fallback_preflight"):
            with self.subTest(job=name):
                self.assertNotIn(
                    f"group: {PRODUCTION_GROUP}", _workflow_job(self.workflow, name),
                    f"{name} does not mutate anything and must not take the lock",
                )
        # And it is still the job that writes, so ordinary publication keeps
        # serializing with the benchmark exactly as before.
        self.assertIn("contents: write", self.publish)
        self.assertEqual(self.workflow.count("contents: write"), 1)

    def test_the_preflight_runs_outside_the_group_and_gates_the_publish(self) -> None:
        self.assertIn("concurrency:", self.preflight)
        self.assertIn("group: election-simulator-fallback-preflight", self.preflight)
        self.assertIn("needs: [fallback_preflight]", self.publish)
        # always(), or a non-fallback trigger would be skipped along with the
        # preflight it does not use.
        self.assertIn("always()", self.publish)
        # A preflight that failed is not a pass.
        self.assertIn("needs.fallback_preflight.result == 'success'", self.publish)
        self.assertIn("needs.fallback_preflight.outputs.proceed == 'true'", self.publish)
        self.assertIn("needs.fallback_preflight.result == 'skipped'", self.publish)

    def test_the_preflight_covers_every_unattended_path(self) -> None:
        """Both unattended triggers, and only those.

        It began as the fallback's gate alone.  The scheduled crons are
        unattended too, and the 20:00Z intraday tick lands inside the
        protected interval once GitHub's delivery lateness is taken into
        account, so they are gated here as well.
        """

        self.assertIn("github.event.inputs.mode == 'publish_if_stale'", self.preflight)
        self.assertIn("github.event_name == 'schedule'", self.preflight)
        # It reads runs, and writes nothing.
        self.assertIn("actions: read", self.preflight)
        self.assertNotIn("contents: write", self.preflight)
        self.assertIn("persist-credentials: false", self.preflight)

    def test_the_preflight_enforces_the_kill_switch_and_asks_about_real_runs(self) -> None:
        """Both guards, and the reason there are two.

        A query of current runs cannot see a capture GitHub has not created
        yet; the frozen window can. A clock cannot see a capture delivered
        hours late or queued behind something else; the query can.
        """

        self.assertIn("ELECTION_AUTOMATION_ENABLED", self.preflight)
        self.assertIn("DISABLED_BY_REPOSITORY_KILL_SWITCH", self.preflight)
        self.assertIn("prospective-benchmark-2026.yml", self.preflight)
        for status in ("queued", "in_progress", "waiting"):
            with self.subTest(status=status):
                self.assertIn(f'.status == "{status}"', self.preflight)
        self.assertIn("DEFERRED_BENCHMARK_RUN_ACTIVE", self.preflight)
        self.assertIn("benchmark_window_conflict", self.preflight)
        self.assertIn("DEFERRED_BENCHMARK_WINDOW", self.preflight)

    def test_an_operator_publish_never_reaches_the_preflight(self) -> None:
        """The generic manual bypass is preserved.

        mode=publish is an explicit operator action: it skips the preflight
        entirely, so the kill-switch enforcement added there cannot leak into
        it.
        """

        self.assertNotIn("inputs.mode == 'publish'\n", self.preflight)
        self.assertIn("github.event.inputs.mode == 'publish'", self.publish)
        self.assertIn("github.event.inputs.mode == 'publish_if_stale'", self.publish)
        self.assertIn("github.event_name == 'schedule'", self.publish)


def _cron_utc_hours(expression: str) -> list[int]:
    """The UTC hours a ``0 H[,H...] * * *`` publication cron fires at.

    Derived from the workflow's own constants rather than restated, so a
    schedule change cannot leave these tests asserting a schedule that is no
    longer configured.
    """

    minute, hours = expression.split()[:2]
    assert minute == "0", f"unexpected cron minute field: {expression}"
    return [int(part) for part in hours.split(",")]


class ScheduledPublicationDeferralTests(unittest.TestCase):
    """A scheduled publication defers rather than burning a benchmark slot.

    The failure this pins is not a wrong value; it is a scheduled tick
    *reaching* the production lock at all during the protected interval.
    Publish holds ``election-simulator-production`` for up to its 120-minute
    timeout, so a capture created punctually at 20:30Z waits behind it and
    archives ``LATE_EXCLUDED`` -- permanently, because the slot is frozen.

    That 120-minute hold is also what sets the interval's start: 120 minutes
    before the 20:30Z pre-warm, so 18:30Z.  The intraday cron's 20:00Z tick is
    inside it already, with no delivery lateness assumed.

    The asymmetry is the whole argument: a deferred poll-change publication is
    repaired by the next tick; a missed capture slot is not repairable at all.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")
        cls.preflight = _workflow_job(cls.workflow, "fallback_preflight")
        cls.publish = _workflow_job(cls.workflow, "publish")
        # The real jq filter, executed rather than pattern-matched.
        match = re.search(
            r"--jq '(\[\.workflow_runs\[\].*?length)'", cls.workflow, re.DOTALL
        )
        assert match, "the preflight's active-run filter was not found"
        cls.active_filter = " ".join(match.group(1).split())

    def _frozen_dates(self) -> list[date]:
        span = (FINAL_CAPTURE_DATE - FIRST_CAPTURE_DATE).days
        return [FIRST_CAPTURE_DATE + timedelta(days=n) for n in range(span + 1)]

    def _active_count(self, statuses: list[str]) -> int:
        if not shutil.which("jq"):
            self.skipTest("jq is unavailable")
        payload = json.dumps({"workflow_runs": [{"status": s} for s in statuses]})
        result = subprocess.run(
            ["jq", self.active_filter],
            input=payload, capture_output=True, text=True, check=True,
        )
        return int(result.stdout.strip())

    # --- requirement: before the protected interval, a scheduled tick proceeds

    def test_an_intraday_tick_before_the_protected_interval_proceeds(self) -> None:
        for day in self._frozen_dates():
            start, _ = benchmark_protected_interval(day)
            for hour in _cron_utc_hours(INTRADAY_SCHEDULE_UTC):
                tick = datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)
                if tick >= start:
                    continue
                with self.subTest(day=day, hour=hour):
                    self.assertIsNone(
                        benchmark_window_conflict(tick),
                        f"{tick.isoformat()} precedes the interval and must publish",
                    )
        # The nearest miss is 18:00Z, 30 minutes clear of the 18:30Z start --
        # which is exactly how little lateness it takes to pull it inside.
        self.assertIsNone(
            benchmark_window_conflict(datetime(2026, 9, 8, 18, 0, tzinfo=timezone.utc))
        )
        self.assertEqual(
            benchmark_window_conflict(
                datetime(2026, 9, 8, 18, 31, tzinfo=timezone.utc)
            ),
            date(2026, 9, 8),
        )

    def test_the_mandatory_daily_tick_never_falls_inside_the_interval(self) -> None:
        """The mandatory daily publication is preserved.

        Its 04:00Z tick is ~14.5 hours clear of the 18:30Z interval start, so
        the deferral cannot suppress the one publication that must happen every
        day.  Only the intraday poll-change checks are ever deferred.
        """

        for day in self._frozen_dates():
            for hour in _cron_utc_hours(DAILY_SCHEDULE_UTC):
                tick = datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)
                with self.subTest(day=day, hour=hour):
                    self.assertIsNone(benchmark_window_conflict(tick))

    # --- requirement: inside the interval, it stands down without the lock

    def test_the_intraday_tick_inside_the_protected_interval_stands_down(self) -> None:
        """The 20:00Z poll-change check collides by the clock alone.

        The pre-warm starts at 20:30Z and publish may hold the lock for 120
        minutes, so the interval begins at 18:30Z.  20:00Z is already inside
        it: this tick needs no delivery lateness whatsoever to block a
        capture.  It is the structural collision, not an edge case.
        """

        conflicting = []
        for day in self._frozen_dates():
            for hour in _cron_utc_hours(INTRADAY_SCHEDULE_UTC):
                tick = datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)
                if benchmark_window_conflict(tick) is not None:
                    conflicting.append((day, hour))
        self.assertEqual(
            sorted({hour for _, hour in conflicting}), [20],
            "only the 20:00Z intraday tick should conflict nominally",
        )
        self.assertEqual(len(conflicting), len(self._frozen_dates()))

    def test_any_instant_inside_the_interval_stands_down(self) -> None:
        """The interval is closed on both ends, and half-open outside them.

        This pins the boundaries themselves.  Which nominal ticks fall inside
        is the subject of the two tests above; delivery lateness only widens
        that set, because an arrival time is just another instant here.
        """

        day = date(2026, 9, 8)
        start, end = benchmark_protected_interval(day)
        for offset in (timedelta(0), timedelta(minutes=1), timedelta(hours=1)):
            with self.subTest(offset=offset):
                self.assertEqual(benchmark_window_conflict(start + offset), day)
        self.assertEqual(benchmark_window_conflict(end), day)
        self.assertIsNone(benchmark_window_conflict(start - timedelta(seconds=1)))
        self.assertIsNone(benchmark_window_conflict(end + timedelta(seconds=1)))

    def test_the_stand_down_decision_is_made_outside_the_production_group(self) -> None:
        """Why this is a structural test.

        Every decision function here was already correct.  What matters is
        *where* the decision runs: a job that has entered
        ``election-simulator-production`` cannot protect a capture that is
        already queued behind it.
        """

        self.assertIn("group: election-simulator-fallback-preflight", self.preflight)
        self.assertNotIn(f"group: {PRODUCTION_GROUP}", self.preflight)
        self.assertIn(f"group: {PRODUCTION_GROUP}", self.publish)
        self.assertIn("needs: [fallback_preflight]", self.publish)
        self.assertIn("needs.fallback_preflight.outputs.proceed == 'true'", self.publish)
        # The scheduled path is gated by that output rather than skipping it.
        self.assertIn("github.event_name == 'schedule'", self.preflight)
        self.assertIn("Production lock: NOT_REQUESTED", self.preflight)

    # --- requirement: an active capture defers regardless of the clock

    def test_an_active_capture_stands_down_whatever_the_clock_says(self) -> None:
        """Every status GitHub can show a live-or-pending run under."""

        for status in ("queued", "in_progress", "waiting", "pending", "requested"):
            with self.subTest(status=status):
                self.assertEqual(self._active_count([status]), 1)

    def test_the_active_run_filter_consults_no_clock(self) -> None:
        """A queued capture is invisible to a wall-clock check.

        This is why the run query exists alongside the window rule: a capture
        delivered hours late, or queued behind another run, is not in any
        nominal window at all.
        """

        self.assertNotIn("created_at", self.active_filter)
        self.assertEqual(self._active_count([]), 0)
        self.assertEqual(self._active_count(["completed", "completed"]), 0)
        self.assertEqual(self._active_count(["completed", "in_progress"]), 1)
        self.assertIn("DEFERRED_BENCHMARK_RUN_ACTIVE", self.preflight)

    # --- requirement: manual publish stays explicit

    def test_a_manual_publish_remains_operator_controlled(self) -> None:
        """An operator's publish skips the preflight entirely.

        It is an explicit human action: it bypasses the kill switch today and
        must not acquire a benchmark deferral it did not ask for.  The
        kill-switch stand-down is likewise still scoped to the fallback
        dispatch, so the scheduled path gained deferral and nothing else.
        """

        self.assertNotIn("inputs.mode == 'publish'\n", self.preflight)
        self.assertIn("github.event.inputs.mode == 'publish'", self.publish)
        self.assertIn("needs.fallback_preflight.result == 'skipped'", self.publish)
        self.assertIn('IS_FALLBACK: ${{ github.event_name ==', self.preflight)
        self.assertIn('if [[ "$IS_FALLBACK" == "true" && ', self.preflight)

    # --- requirement: outside Sep 4-12 nothing changes

    def test_outside_the_frozen_window_no_tick_is_ever_deferred(self) -> None:
        every_hour = sorted(
            set(_cron_utc_hours(INTRADAY_SCHEDULE_UTC))
            | set(_cron_utc_hours(DAILY_SCHEDULE_UTC))
        )
        for day in (
            FIRST_CAPTURE_DATE - timedelta(days=1),
            FINAL_CAPTURE_DATE + timedelta(days=1),
            date(2026, 1, 15),
        ):
            for hour in every_hour:
                tick = datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)
                with self.subTest(day=day, hour=hour):
                    self.assertIsNone(benchmark_window_conflict(tick))

    def test_the_timing_constants_are_not_restated_in_yaml(self) -> None:
        """The interval is imported, never copied into shell."""

        self.assertIn("benchmark_window_conflict", self.preflight)
        self.assertIn("scripts.publication_fallback", self.preflight)
        for literal in ("22:30", "23:30", "20:30", "120", "minutes=120"):
            with self.subTest(literal=literal):
                self.assertNotIn(f"timedelta({literal}", self.preflight)


if __name__ == "__main__":
    unittest.main()
