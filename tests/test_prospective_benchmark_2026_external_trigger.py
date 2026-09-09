"""The independent external scheduled trigger (amendment 006).

GitHub's cron delivered the 2026-09-04, 05 and 06 capture ticks 1h42m, 1h41m
and 1h44m late, the last of those with amendment 005's one-hour pre-warm
already live, and all three slots archived LATE_EXCLUDED. The timing-eligible
window is thirty minutes wide and the job runs in under a minute, so the
defect is in trigger delivery. Amendment 006 adds a second, independent
deliverer of the same unchanged trigger.

The property these tests exist to protect is narrow and easy to lose. The
external scheduler supplies *no scheduled date and no timestamp*: it sends
mode=scheduled_capture and nothing else, and the slot is resolved from
GitHub's recorded run created_at by the same frozen rule the cron path uses.
That is what keeps an operator-supplied claim from ever becoming timing
evidence -- not a validation step that could be relaxed, but the absence of an
input. Adding a scheduled_date input to the Worker would silently undo it,
which is why the payload shape is asserted here.

Note what created_at does and does not establish. It is GitHub's own
authoritative record of when a run was created, and slot attribution rests on
it. It is not evidence of scheduler identity: an authorized operator can
choose when to initiate a workflow_dispatch. Scientific timing eligibility
rests solely on retrieved_at_utc from the capture process clock.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from scripts.prospective_benchmark_2026.time_rules import (
    CUTOFF_LOCAL_TIME,
    CaptureTimeError,
    SCHEDULED_START_LOCAL_TIME,
    STOCKHOLM,
    UTC,
    classify_capture_time,
    resolve_scheduled_slot,
    scheduled_cutoff,
    seconds_until_cutoff,
)

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "prospective-benchmark-2026.yml"
PUBLICATION = ROOT / ".github" / "workflows" / "election-simulator-publication.yml"
WORKER_DIR = ROOT / "ops" / "benchmark-trigger-worker"
PUBLICATION_WORKER_DIR = ROOT / "ops" / "publication-fallback-worker"
ARCHIVE = ROOT / "data" / "processed" / "prospective_benchmark_2026"

SLOT = "2026-09-07"


def _utc(moment: str) -> datetime:
    """An aware UTC instant written as YYYY-MM-DDTHH:MM:SS."""

    return datetime.fromisoformat(moment).replace(tzinfo=UTC)


def _worker_crons() -> list[str]:
    text = (WORKER_DIR / "wrangler.toml").read_text(encoding="utf-8")
    match = re.search(r"^crons\s*=\s*\[(.*?)\]", text, re.MULTILINE | re.DOTALL)
    assert match, "wrangler.toml declares no crons"
    return re.findall(r'"([^"]+)"', match.group(1))


def _manifest_for(row: dict) -> dict:
    """The capture manifest an index row points at.

    Eligibility is a property of the capture, recorded in its own manifest;
    the index row carries the pointer. Read it rather than trusting a copy.
    """

    return json.loads((ARCHIVE / row["path"]).read_text(encoding="utf-8"))


def _baseline_ref() -> str | None:
    """`origin/main`, when this checkout actually has it.

    A pull-request checkout may hold only the merge ref, so a comparison
    against the baseline is real coverage where it is possible and must not
    become a spurious failure where it is not. The invariants that can be
    asserted without git are asserted without git instead.
    """

    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", "origin/main^{commit}"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    return "origin/main" if resolved.returncode == 0 else None


def _strip_js_comments(source: str) -> str:
    """Source with // and /* */ comments removed.

    Crude but sufficient here: this file contains no string literal holding a
    comment marker, and the assertion it serves is about what the Worker
    *sends*, which must not be confused with what its comments *discuss*.
    """

    without_blocks = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return "\n".join(re.sub(r"//.*$", "", line) for line in without_blocks.splitlines())


def _cron_utc_time(expression: str) -> time:
    minute, hour = expression.split()[0], expression.split()[1]
    return time(int(hour), int(minute))


class ExternalTriggerSlotBoundary(unittest.TestCase):
    """The slot boundary is 22:30 Europe/Stockholm, i.e. 20:30Z in-window.

    This is the rule that decides which night's evidence a trigger produces,
    and it is the reason no external tick may be configured before 20:30Z. A
    trigger one second earlier does not merely start early: it resolves to the
    PREVIOUS Stockholm slot, and a durable capture there would occupy a slot
    that can never be filled again.
    """

    def test_one_second_before_the_boundary_resolves_the_previous_slot(self) -> None:
        self.assertEqual(
            resolve_scheduled_slot(_utc("2026-09-07T20:29:59")),
            date(2026, 9, 6),
        )

    def test_the_boundary_itself_resolves_its_own_slot(self) -> None:
        self.assertEqual(
            resolve_scheduled_slot(_utc("2026-09-07T20:30:00")),
            date(2026, 9, 7),
        )

    def test_the_boundary_is_the_frozen_local_scheduled_start(self) -> None:
        # 20:30Z is not a chosen number: it is SCHEDULED_START_LOCAL_TIME
        # expressed in UTC during the frozen window. Read it from the rule
        # rather than restating it, so the two cannot drift apart.
        boundary_local = datetime.combine(
            date.fromisoformat(SLOT), SCHEDULED_START_LOCAL_TIME, tzinfo=STOCKHOLM
        )
        self.assertEqual(boundary_local.astimezone(UTC).timetz().replace(tzinfo=None), time(20, 30))

    def test_each_configured_tick_resolves_its_own_stockholm_slot(self) -> None:
        for expression in _worker_crons():
            tick = _cron_utc_time(expression)
            created = datetime.combine(date.fromisoformat(SLOT), tick, tzinfo=UTC)
            with self.subTest(cron=expression):
                self.assertEqual(
                    resolve_scheduled_slot(created),
                    date.fromisoformat(SLOT),
                    f"{expression} does not resolve to its own Stockholm date",
                )

    def test_a_trigger_delayed_past_local_midnight_resolves_the_previous_slot(self) -> None:
        # 22:10Z is 00:10 the next day in Stockholm.
        created = _utc("2026-09-07T22:10:00")
        self.assertEqual(resolve_scheduled_slot(created), date(2026, 9, 7))
        # ...and its retrieval is durably LATE_EXCLUDED, never relabelled.
        timing = classify_capture_time(SLOT, created, durable=True)
        self.assertEqual(timing.status, "LATE_EXCLUDED")
        self.assertFalse(timing.eligible)


class ExternalTriggerTimingWindow(unittest.TestCase):
    """The eligible window is thirty minutes wide and unchanged."""

    def test_the_window_opens_at_the_frozen_cutoff_and_closes_at_local_midnight(self) -> None:
        cutoff = scheduled_cutoff(SLOT)
        self.assertEqual(cutoff.timetz().replace(tzinfo=None), CUTOFF_LOCAL_TIME)
        opens = cutoff.astimezone(UTC)
        closes = datetime.combine(
            cutoff.date() + timedelta(days=1), time(0, 0), tzinfo=STOCKHOLM
        ).astimezone(UTC)
        self.assertEqual(closes - opens, timedelta(minutes=30))
        self.assertEqual(opens, _utc("2026-09-07T21:30:00"))
        self.assertEqual(closes, _utc("2026-09-07T22:00:00"))

    def test_every_configured_tick_falls_inside_the_usable_interval(self) -> None:
        cutoff = scheduled_cutoff(SLOT).astimezone(UTC)
        boundary = _utc("2026-09-07T20:30:00")
        closes = _utc("2026-09-07T22:00:00")
        for expression in _worker_crons():
            tick = datetime.combine(date.fromisoformat(SLOT), _cron_utc_time(expression), tzinfo=UTC)
            with self.subTest(cron=expression):
                self.assertGreaterEqual(
                    tick, boundary,
                    "a tick before 20:30Z resolves the PREVIOUS slot and could occupy it",
                )
                self.assertLess(
                    tick, closes,
                    "a tick at or after 22:00Z can only ever produce LATE_EXCLUDED evidence",
                )

    def test_a_pre_cutoff_tick_waits_and_a_post_cutoff_tick_does_not(self) -> None:
        cutoff = scheduled_cutoff(SLOT).astimezone(UTC)
        for expression in _worker_crons():
            tick = datetime.combine(date.fromisoformat(SLOT), _cron_utc_time(expression), tzinfo=UTC)
            waited = seconds_until_cutoff(SLOT, tick)
            with self.subTest(cron=expression):
                if tick < cutoff:
                    self.assertAlmostEqual(waited, (cutoff - tick).total_seconds())
                else:
                    self.assertEqual(waited, 0.0)

    def test_a_trigger_after_the_window_is_late_excluded_not_eligible(self) -> None:
        timing = classify_capture_time(SLOT, _utc("2026-09-07T22:00:01"), durable=True)
        self.assertEqual(timing.status, "LATE_EXCLUDED")
        self.assertFalse(timing.eligible)

    def test_a_capture_reaching_the_cutoff_after_the_prewarm_is_eligible(self) -> None:
        # The 20:30Z tick pre-warms for an hour, then retrieves just after the
        # frozen cutoff. That is the intended path and it must be eligible.
        timing = classify_capture_time(SLOT, _utc("2026-09-07T21:30:41"), durable=True)
        self.assertEqual(timing.status, "ON_TIME_ELIGIBLE")
        self.assertTrue(timing.eligible)


class ExternalTriggerCannotForgeTiming(unittest.TestCase):
    """No externally supplied value may make a late capture look on time."""

    def test_the_worker_supplies_no_scheduled_date_and_no_timestamp(self) -> None:
        source = (WORKER_DIR / "worker.mjs").read_text(encoding="utf-8")
        self.assertIn('inputs: { mode: "scheduled_capture" }', source)
        # Assert against the code, not the prose: the comments name these
        # things precisely in order to say the Worker must not send them.
        #
        # The absence of the inputs IS the safety property. A validation step
        # could be relaxed later; an input that was never added cannot be.
        code = _strip_js_comments(source)
        self.assertIn("scheduled_capture", code)
        for forbidden in ("scheduled_date", "retrieved_at", "Date.now", "toISOString", "getTime"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, code)

    def test_the_guard_ignores_dispatch_inputs_on_the_unattended_path(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        guard = text[text.index("Resolve intended scheduled slot"):text.index("  dry_run:")]
        # The unattended branch resolves from created_at; MANUAL_DATE is only
        # ever read on the operator branch.
        unattended = guard[guard.index('if [[ "$UNATTENDED" == "true" ]]; then'):guard.index("else\n            SCHEDULED_DATE=\"$MANUAL_DATE\"")]
        self.assertIn("resolve-slot --run-created-at", unattended)
        self.assertNotIn("MANUAL_DATE", unattended)

    def test_forged_timing_cannot_enter_through_the_capture_cli(self) -> None:
        # Unchanged from amendment 005, reasserted because an external trigger
        # is exactly the actor that would want such a flag.
        rejected = subprocess.run(
            [
                "python", "-m", "scripts.prospective_benchmark_2026", "capture",
                "--mode", "dry_run", "--scheduled-date", SLOT,
                "--retrieved-at", "2026-09-07T21:31:00Z",
            ],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(rejected.returncode, 2)

    def test_a_replayed_old_slot_cannot_produce_eligible_evidence(self) -> None:
        # An operator naming a stale date, days later, is retroactive: the
        # durable path raises before any source request rather than writing.
        with self.assertRaises(CaptureTimeError) as caught:
            classify_capture_time("2026-09-04", _utc("2026-09-07T21:31:00"), durable=True)
        self.assertIn("retroactive", str(caught.exception).lower())
        # The immediately preceding slot is the one case amendment 001 allows,
        # and it is durable but explicitly NOT eligible.
        previous = classify_capture_time("2026-09-06", _utc("2026-09-07T21:31:00"), durable=True)
        self.assertEqual(previous.status, "LATE_EXCLUDED")
        self.assertFalse(previous.eligible)

    def test_a_future_slot_cannot_be_claimed_before_its_cutoff(self) -> None:
        with self.assertRaises(CaptureTimeError) as caught:
            classify_capture_time("2026-09-08", _utc("2026-09-07T21:31:00"), durable=True)
        self.assertIn("before its scheduled cutoff", str(caught.exception))


class DuplicateTriggerStandsDownGreen(unittest.TestCase):
    """Redundant ticks are the design; the late ones must be a quiet no-op.

    The stand-down filter is executed here rather than pattern-matched, so a
    silent change to it fails rather than passing a substring assertion.
    """

    @classmethod
    def setUpClass(cls) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        match = re.search(r"'(\[\.captures\[\][^']*length > 0)'", text, re.DOTALL)
        assert match, "the guard's stand-down filter was not found"
        cls.filter = " ".join(match.group(1).split())

    def _stands_down(self, captures: list[dict], slot: str) -> bool:
        if not shutil.which("jq"):
            self.skipTest("jq is unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "index.json"
            index.write_text(json.dumps({"captures": captures}), encoding="utf-8")
            result = subprocess.run(
                ["jq", "-e", "--arg", "slot", slot, self.filter, str(index)],
                capture_output=True, text=True, check=False,
            )
            return result.returncode == 0

    def test_a_slot_with_an_eligible_capture_stands_down(self) -> None:
        self.assertTrue(self._stands_down(
            [{"scheduled_date": SLOT, "timing_eligible": True}], SLOT))

    def test_a_slot_with_an_ineligible_capture_stays_loud(self) -> None:
        # A LATE_EXCLUDED record occupies the slot but is not success. It must
        # NOT be suppressed: the collision has to surface.
        self.assertFalse(self._stands_down(
            [{"scheduled_date": SLOT, "timing_eligible": False}], SLOT))

    def test_an_empty_slot_proceeds_to_capture(self) -> None:
        self.assertFalse(self._stands_down([], SLOT))
        self.assertFalse(self._stands_down(
            [{"scheduled_date": "2026-09-06", "timing_eligible": True}], SLOT))

    def test_the_real_archive_stands_down_exactly_the_slots_already_captured(self) -> None:
        """Stand-down must track eligibility, not a moment in the campaign.

        This asserted that no slot in 2026-09-07..12 stands down, which held
        only while every capture in the archive was LATE_EXCLUDED. 2026-09-08
        is the first on-time success, so it now stands down -- correctly, and
        exactly as `test_a_slot_with_an_eligible_capture_stands_down` below
        requires. The old form would have gone on failing for every slot the
        campaign successfully captured.

        The property that actually matters is the equivalence, in both
        directions: a slot holding a timing-eligible capture stands down, and a
        slot without one never does. That is what protects the remaining
        schedule, and it does not depend on which day it is asserted.
        """

        captures = json.loads((ARCHIVE / "index.json").read_text())["captures"]
        eligible_slots = {
            row["scheduled_date"]
            for row in captures
            if _manifest_for(row).get("timing_eligible") is True
        }
        scheduled = json.loads((ARCHIVE / "protocol.json").read_text())["schedule"]
        for slot in scheduled["scheduled_dates"]:
            with self.subTest(slot=slot):
                expected = slot in eligible_slots
                self.assertEqual(
                    self._stands_down(captures, slot),
                    expected,
                    f"{slot} stand-down disagrees with its eligibility: an "
                    "eligible capture must suppress the slot and nothing else may",
                )
        # Not vacuous in either direction: the campaign has at least one
        # captured slot and at least one still to run.
        self.assertTrue(eligible_slots, "no eligible capture, so the True branch is untested")
        self.assertTrue(
            set(scheduled["scheduled_dates"]) - eligible_slots,
            "every slot is captured, so the False branch is untested",
        )


class BothSchedulersMayDeliver(unittest.TestCase):
    """The GitHub cron is retained, so neither scheduler is a single point."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = WORKFLOW.read_text(encoding="utf-8")

    def test_the_github_cron_is_retained_at_the_frozen_start(self) -> None:
        self.assertIn('cron: "30 20 * 9 *"', self.text)
        self.assertNotIn('cron: "30 21 * 9 *"', self.text)

    def test_both_delivery_mechanisms_take_the_same_created_at_branch(self) -> None:
        self.assertIn('if [[ "$EVENT_NAME" == "schedule" ]]; then\n            UNATTENDED=true', self.text)
        self.assertIn("scheduled_capture)", self.text)
        self.assertIn('if [[ "$UNATTENDED" == "true" ]]; then', self.text)

    def test_both_are_held_to_the_strict_eligibility_assertion(self) -> None:
        self.assertIn('if [[ "$UNATTENDED" != "true" ]]; then\n            ARGS+=(--warn-only)', self.text)
        # The old event-name test would have exempted an external dispatch.
        self.assertNotIn('if [[ "$EVENT_NAME" != "schedule" ]]; then\n            ARGS+=(--warn-only)', self.text)

    def test_the_external_mode_runs_the_capture_job_not_the_dry_run(self) -> None:
        self.assertIn("capture:\n    if: needs.schedule_guard.outputs.active == 'true' && needs.schedule_guard.outputs.mode == 'capture'", self.text)
        self.assertIn("scheduled_capture", self.text)

    def test_an_out_of_window_unattended_tick_is_a_green_no_op(self) -> None:
        # Both crons are yearless, so ticks outside 2026-09-04..12 are routine
        # and must not go red every night.
        self.assertIn("Unattended trigger is outside the frozen 2026-09-04..12 window.", self.text)


class ProductionLockIsUnchanged(unittest.TestCase):
    """A capture still serialises with publication, exactly as before."""

    def test_the_benchmark_still_shares_the_production_group(self) -> None:
        self.assertIn(
            "concurrency:\n  group: election-simulator-production\n  cancel-in-progress: false",
            WORKFLOW.read_text(encoding="utf-8"),
        )

    def test_the_publication_preflight_still_watches_this_workflow(self) -> None:
        # Design B would have added a second workflow, invisible to this
        # query, letting the publication fallback take the lock while a
        # capture was queued. Keeping one workflow is what preserves it.
        publication = PUBLICATION.read_text(encoding="utf-8")
        self.assertIn("BENCHMARK_WORKFLOW: prospective-benchmark-2026.yml", publication)
        self.assertEqual(WORKFLOW.name, "prospective-benchmark-2026.yml")

    def test_a_capture_released_by_a_publication_before_the_close_is_eligible(self) -> None:
        # Queued behind a publication that holds the lock until 21:45Z: the
        # cutoff is already past, so it retrieves at once and still counts.
        self.assertEqual(seconds_until_cutoff(SLOT, _utc("2026-09-07T21:45:00")), 0.0)
        timing = classify_capture_time(SLOT, _utc("2026-09-07T21:45:40"), durable=True)
        self.assertEqual(timing.status, "ON_TIME_ELIGIBLE")
        self.assertTrue(timing.eligible)


class WorkerIsSeparateFromPublicationFallback(unittest.TestCase):
    """One leaked credential must not be able to start both paths."""

    def test_the_two_workers_are_distinct_deployments(self) -> None:
        self.assertTrue((PUBLICATION_WORKER_DIR / "wrangler.toml").is_file())
        mine = (WORKER_DIR / "wrangler.toml").read_text(encoding="utf-8")
        theirs = (PUBLICATION_WORKER_DIR / "wrangler.toml").read_text(encoding="utf-8")
        self.assertIn('name = "election-benchmark-trigger"', mine)
        self.assertIn('name = "election-publication-fallback"', theirs)
        self.assertNotEqual(_worker_crons(), re.findall(r'"([^"]+)"', theirs.split("crons = [")[1].split("]")[0]))

    def test_each_worker_dispatches_only_its_own_workflow(self) -> None:
        mine = (WORKER_DIR / "worker.mjs").read_text(encoding="utf-8")
        theirs = (PUBLICATION_WORKER_DIR / "worker.mjs").read_text(encoding="utf-8")
        self.assertIn('const WORKFLOW = "prospective-benchmark-2026.yml";', mine)
        self.assertNotIn("election-simulator-publication.yml", mine)
        self.assertIn('const WORKFLOW = "election-simulator-publication.yml";', theirs)
        self.assertNotIn("prospective-benchmark-2026.yml", theirs)

    def test_the_publication_fallback_keeps_its_own_blast_radius(self) -> None:
        """Structural separation, not a freeze on the other Worker.

        This began as "the whole fallback directory is byte-identical to
        origin/main", which was good evidence that amendment 006 touched
        nothing of the fallback's.  As a standing invariant it was the wrong
        shape twice over: it forbade any future operational change to the
        fallback -- Workers Free allows only five cron triggers per account,
        so the fallback had to move to one coarse cron and select its three
        dispatch times in its handler -- and it would have failed a *benchmark*
        test for a reason with nothing to do with the benchmark.

        What matters for blast radius is not that the fallback never changes.
        It is that the fallback stays pointed at its own workflow, through its
        own credential, and that neither Worker can reach the other's. That is
        what is asserted here, against the code rather than against a hash, so
        it keeps holding while the fallback is maintained.
        """

        theirs = _strip_js_comments(
            (PUBLICATION_WORKER_DIR / "worker.mjs").read_text(encoding="utf-8")
        )
        mine = _strip_js_comments(
            (WORKER_DIR / "worker.mjs").read_text(encoding="utf-8")
        )

        # One target workflow, and it is the publication's.
        self.assertEqual(theirs.count("const WORKFLOW ="), 1)
        self.assertIn(
            'const WORKFLOW = "election-simulator-publication.yml";', theirs
        )
        self.assertNotIn("prospective-benchmark-2026.yml", theirs)
        self.assertIn('mode: "publish_if_stale"', theirs)
        self.assertNotIn("scheduled_capture", theirs)

        # Its own credential, and its own manual-endpoint guard.
        self.assertIn("env.GITHUB_TOKEN", theirs)
        self.assertIn("env.FALLBACK_SECRET", theirs)
        self.assertNotIn("TRIGGER_SECRET", theirs)

        # And the benchmark Worker never reaches for the fallback's secret.
        self.assertNotIn("FALLBACK_SECRET", mine)
        self.assertNotIn("election-simulator-publication.yml", mine)

        # Two deployments, not one: separate directories, and the distinct
        # names and schedules asserted in
        # test_the_two_workers_are_distinct_deployments.
        self.assertNotEqual(WORKER_DIR, PUBLICATION_WORKER_DIR)
        self.assertTrue((PUBLICATION_WORKER_DIR / "wrangler.toml").is_file())

    def test_the_worker_reports_to_a_dead_mans_switch(self) -> None:
        # The failure this Worker exists to catch is "nothing ran at all",
        # which is invisible in an Actions history that has no run in it.
        source = (WORKER_DIR / "worker.mjs").read_text(encoding="utf-8")
        self.assertIn("HEARTBEAT_URL", source)
        self.assertIn("ctx.waitUntil(reportOutcome(env, result))", source)

    def test_the_manual_endpoint_is_secret_guarded(self) -> None:
        source = (WORKER_DIR / "worker.mjs").read_text(encoding="utf-8")
        self.assertIn("X-Trigger-Secret", source)
        self.assertIn("status: 403", source)
        self.assertIn('request.method !== "POST"', source)


class FrozenProtocolIsUntouched(unittest.TestCase):
    """Amendment 006 changes delivery, and nothing else."""

    def test_the_cutoff_and_schedule_fields_are_unchanged(self) -> None:
        protocol = json.loads((ARCHIVE / "protocol.json").read_text())
        self.assertEqual(protocol["schedule"]["local_cutoff_time"], "23:30:00")
        self.assertEqual(protocol["schedule"]["scheduled_utc_time_during_window"], "21:30:00Z")

    def test_the_protocol_bytes_still_hash_to_the_frozen_value(self) -> None:
        # Asserted without git, so it holds in any checkout: the protocol
        # every capture and every amendment references is the original one.
        frozen = "16f920a88eb956b42f24307e4ada3b5b1133bffb11a6561e5a9e9ce75c5e3b23"
        digest = hashlib.sha256((ARCHIVE / "protocol.json").read_bytes()).hexdigest()
        self.assertEqual(digest, frozen)
        self.assertTrue((ARCHIVE / "protocol.sha256").read_text().startswith(frozen))
        index = json.loads((ARCHIVE / "index.json").read_text())
        self.assertEqual(index["protocol_sha256"], frozen)
        for ref in index["amendments"]:
            with self.subTest(amendment=ref["amendment_number"]):
                amendment = json.loads((ARCHIVE / ref["path"]).read_text())
                self.assertEqual(amendment["original_protocol_sha256"], frozen)

    # The modules that implement the frozen rules the protocol hash pins.
    # Narrower than "all Python" on purpose: see the test below.
    #
    # Each entry earns its place by implementing protocol language, not by
    # being nearby. `time_rules` implements `timing_eligibility`; `scoring`
    # and `results` implement `primary_scoring.fair_crps_formula` and the
    # official-result handling; `report` implements
    # `primary_scoring.winner` ("the system with the lower arithmetic mean
    # fair CRPS"), its `tie_rule`, `campaign_scoring`'s "number of dates won
    # by each system", and the `timing_eligible` filter that decides which
    # captures are scored at all.
    #
    # `report` was missing, and its absence was not academic: reversing the
    # comparison in `_winner` inverts every head-to-head outcome in the
    # campaign report and passed this class untouched.
    # `test_a_winner_rule_change_is_rejected` below is the negative case.
    #
    # Deliberately still excluded: `archive` and `__main__` persist and print
    # decisions the modules above have already made, and `capture` /
    # `botten_ada_capture` do retrieval. Pulling those in would drift back
    # towards freezing the whole package, which is what this test stopped
    # doing.
    FROZEN_RULE_SOURCES = (
        "scripts/prospective_benchmark_2026/time_rules.py",
        "scripts/prospective_benchmark_2026/scoring.py",
        "scripts/prospective_benchmark_2026/results.py",
        "scripts/prospective_benchmark_2026/report.py",
    )

    @classmethod
    def _changed_frozen_modules(cls, baseline: str, *, cwd: Path) -> list[str]:
        """The frozen-rule modules that differ from the baseline."""

        return subprocess.run(
            ["git", "diff", "--name-only", baseline, "--", *cls.FROZEN_RULE_SOURCES],
            cwd=cwd, capture_output=True, text=True, check=True,
        ).stdout.split()

    def test_the_frozen_rule_modules_are_unchanged(self) -> None:
        """The frozen rules themselves, not every line of Python in the repo.

        This began as "``scripts/`` is byte-identical to origin/main", which
        was true and useful evidence *about amendment 006*: that amendment
        changes delivery only. As a standing invariant it was the wrong shape.
        It froze the entire Python surface of the repository against any
        branch, so the first unrelated change to any module -- parallelising
        the history backfill, say, after a publication was lost to a job
        timeout -- failed a benchmark test for a reason with nothing to do
        with the benchmark, and the honest fix would have looked like weakening
        a frozen-protocol check.

        What actually has to hold while the campaign is live is that the
        modules implementing the pinned rules do not drift: the cutoff and slot
        attribution in ``time_rules``, and the scoring in ``scoring``/
        ``results``. Those are what ``protocol.sha256`` stands behind, and they
        are what is asserted here. The rest of the amendment's "delivery only"
        claim is carried by the sibling assertions in this class -- the
        protocol hash, the cutoff and schedule fields, and the recorded timing
        contract -- which check the artifacts rather than a diff.
        """

        baseline = _baseline_ref()
        if baseline is None:
            self.skipTest("no origin/main baseline in this checkout")
        changed = self._changed_frozen_modules(baseline, cwd=ROOT)
        self.assertEqual(
            changed, [],
            f"a frozen benchmark rule module changed: {changed}. These implement "
            "the rules protocol.sha256 pins; changing one during the campaign "
            "needs an amendment, not a code review.",
        )

    def test_a_winner_rule_change_is_rejected(self) -> None:
        """The negative case, without which the scope above is only a claim.

        A freeze test that never sees a violation cannot distinguish "nothing
        changed" from "the thing that changed was not being watched". That is
        exactly the failure this list already had: `report.py` was absent, so
        reversing the comparison in `_winner` -- the smallest edit that flips
        every head-to-head outcome the campaign reports -- passed every
        assertion in this class.

        The probe runs in a throwaway worktree of HEAD, so the checkout under
        test is never modified. `git worktree` shares refs, so the baseline
        resolves there exactly as it does here.
        """

        baseline = _baseline_ref()
        if baseline is None:
            self.skipTest("no origin/main baseline in this checkout")

        relative = "scripts/prospective_benchmark_2026/report.py"
        needle = '    return "election_simulator" if first < second else "botten_ada"'
        original = (ROOT / relative).read_text(encoding="utf-8")
        # If the comparison is rewritten, this probe must be updated rather
        # than silently stop probing anything.
        self.assertIn(
            needle, original,
            f"the winner comparison in {relative} moved; update this probe",
        )

        with tempfile.TemporaryDirectory(prefix="frozen-rule-probe-") as tmp:
            probe = Path(tmp) / "worktree"
            subprocess.run(
                ["git", "worktree", "add", "--detach", "-q", str(probe), "HEAD"],
                cwd=ROOT, check=True, capture_output=True, text=True,
            )
            try:
                target = probe / relative
                # Reverse the comparison: a tie stays a tie, and every decided
                # date changes hands.
                target.write_text(
                    original.replace(needle, needle.replace("<", ">")),
                    encoding="utf-8",
                )
                changed = self._changed_frozen_modules(baseline, cwd=probe)
                self.assertIn(
                    relative, changed,
                    "reversing the winner rule was not detected by the freeze; "
                    "FROZEN_RULE_SOURCES does not cover winner selection",
                )
                # And the freeze fails on it, rather than merely noticing.
                with self.assertRaises(AssertionError):
                    self.assertEqual(changed, [])
            finally:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(probe)],
                    cwd=ROOT, check=False, capture_output=True, text=True,
                )

        # The real checkout is untouched by the probe.
        self.assertEqual((ROOT / relative).read_text(encoding="utf-8"), original)

    def test_the_frozen_rule_modules_all_exist(self) -> None:
        """A path typo would make the freeze above silently vacuous."""

        for relative in self.FROZEN_RULE_SOURCES:
            with self.subTest(module=relative):
                self.assertTrue((ROOT / relative).is_file(), relative)

    def test_the_recorded_timing_contract_is_unchanged(self) -> None:
        # Pinned to explicit values rather than to another call of the same
        # function, so that a change to what a capture records fails here
        # instead of comparing equal to itself. These are the fields the
        # archive stores and every downstream report reads; an identical
        # eligible capture must produce exactly this, whichever scheduler
        # delivered the trigger.
        self.assertEqual(
            classify_capture_time(SLOT, _utc("2026-09-07T21:30:41"), durable=True).to_dict(),
            {
                "scheduled_date": "2026-09-07",
                "benchmark_cutoff": "2026-09-07T21:30:00Z",
                "benchmark_cutoff_europe_stockholm": "2026-09-07T23:30:00+02:00",
                "retrieved_at_utc": "2026-09-07T21:30:41Z",
                "retrieved_at_europe_stockholm": "2026-09-07T23:30:41+02:00",
                "timing_status": "ON_TIME_ELIGIBLE",
                "timing_eligible": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
