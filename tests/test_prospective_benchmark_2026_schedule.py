"""Scheduling tests for the prewarmed prospective benchmark capture.

The frozen cutoff is 23:30 Europe/Stockholm and a durable capture is
prohibited before it, yet a capture stops being timing-eligible once the local
calendar day ends. The eligible window is therefore thirty minutes wide. The
workflow used to *start* at the cutoff and spend that window on checkout,
dependency install and gate tests, which is why 2026-09-04 and 2026-09-05
produced successful runs and zero timing-eligible captures.

These tests pin the scheme that replaces it: start an hour early, resolve the
intended slot from GitHub's recorded run creation time, wait for the exact
frozen cutoff, and fail the run if the slot did not produce eligible evidence.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

from scripts.prospective_benchmark_2026.archive import slot_timing_outcome
from scripts.prospective_benchmark_2026.capture import run_capture
from scripts.prospective_benchmark_2026.time_rules import (
    CUTOFF_LOCAL_TIME,
    MAX_CUTOFF_WAIT,
    SCHEDULED_START_LOCAL_TIME,
    STOCKHOLM,
    CaptureTimeError,
    classify_capture_time,
    resolve_scheduled_slot,
    scheduled_cutoff,
    seconds_until_cutoff,
    wait_for_cutoff,
)


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "prospective-benchmark-2026.yml"
ARCHIVE = ROOT / "data" / "processed" / "prospective_benchmark_2026"
SLOT = "2026-09-06"


def _local(day: str, clock: str) -> datetime:
    """A Europe/Stockholm wall-clock instant during the frozen window."""

    return datetime.combine(date.fromisoformat(day), time.fromisoformat(clock), tzinfo=STOCKHOLM)


class TestFrozenCutoffIsUnchanged(unittest.TestCase):
    def test_the_cutoff_itself_is_not_moved_by_the_earlier_start(self) -> None:
        self.assertEqual(CUTOFF_LOCAL_TIME, time(23, 30))
        self.assertLess(SCHEDULED_START_LOCAL_TIME, CUTOFF_LOCAL_TIME)
        cutoff = scheduled_cutoff(SLOT)
        self.assertEqual(cutoff, _local(SLOT, "23:30"))
        # 21:30Z during the window, exactly as protocol.json records it. Both
        # assertions read the cutoff's own Stockholm offset (CEST, +02:00
        # throughout the frozen window) rather than the machine's timezone.
        self.assertEqual(cutoff.utcoffset(), timedelta(hours=2))
        self.assertEqual(cutoff.utctimetuple()[3:5], (21, 30))


class TestSlotAttribution(unittest.TestCase):
    """Which Stockholm slot a scheduled run belongs to, at any delay."""

    def test_prewarmed_start_belongs_to_its_own_stockholm_date(self) -> None:
        created = _local(SLOT, "22:30")
        self.assertEqual(resolve_scheduled_slot(created), date.fromisoformat(SLOT))
        # A full hour of prewarm remains before retrieval is permitted.
        self.assertEqual(seconds_until_cutoff(SLOT, created), 3600.0)

    def test_start_delayed_to_the_exact_cutoff_keeps_its_slot_and_waits_no_longer(self) -> None:
        created = _local(SLOT, "23:30")
        self.assertEqual(resolve_scheduled_slot(created), date.fromisoformat(SLOT))
        self.assertEqual(seconds_until_cutoff(SLOT, created), 0.0)
        timing = classify_capture_time(SLOT, created, durable=True)
        self.assertEqual(timing.status, "ON_TIME_ELIGIBLE")
        self.assertTrue(timing.eligible)

    def test_delayed_but_same_day_start_still_captures_eligibly(self) -> None:
        created = _local(SLOT, "23:50")
        self.assertEqual(resolve_scheduled_slot(created), date.fromisoformat(SLOT))
        self.assertEqual(seconds_until_cutoff(SLOT, created), 0.0)
        timing = classify_capture_time(SLOT, _local(SLOT, "23:52"), durable=True)
        self.assertEqual(timing.status, "ON_TIME_ELIGIBLE")
        self.assertTrue(timing.eligible)

    def test_after_midnight_late_start_is_attributed_to_the_intended_slot(self) -> None:
        """The case the old cutoff-keyed boundary got right and must keep."""

        created = _local("2026-09-07", "00:20")
        self.assertEqual(resolve_scheduled_slot(created), date.fromisoformat(SLOT))
        timing = classify_capture_time(SLOT, _local("2026-09-07", "00:25"), durable=True)
        self.assertEqual(timing.status, "LATE_EXCLUDED")
        self.assertFalse(timing.eligible)

    def test_attribution_holds_across_the_whole_plausible_delay_range(self) -> None:
        cases = {
            ("2026-09-06", "22:30"): "2026-09-06",  # on time
            ("2026-09-06", "22:31"): "2026-09-06",  # one minute late
            ("2026-09-06", "23:29"): "2026-09-06",  # 59 minutes late, still pre-cutoff
            ("2026-09-06", "23:30"): "2026-09-06",  # exactly at the cutoff
            ("2026-09-06", "23:59"): "2026-09-06",  # last eligible minute
            ("2026-09-07", "00:00"): "2026-09-06",  # first minute past midnight
            ("2026-09-07", "02:20"): "2026-09-06",  # badly delayed
            ("2026-09-07", "22:29"): "2026-09-06",  # absurdly delayed, still yesterday's
            ("2026-09-07", "22:30"): "2026-09-07",  # the next slot's own start
        }
        for (day, clock), expected in cases.items():
            with self.subTest(created=f"{day} {clock}"):
                self.assertEqual(
                    resolve_scheduled_slot(_local(day, clock)),
                    date.fromisoformat(expected),
                )

    def test_a_slot_is_never_resolved_into_the_future(self) -> None:
        for day, clock in (("2026-09-06", "22:30"), ("2026-09-07", "00:20")):
            created = _local(day, clock)
            resolved = resolve_scheduled_slot(created)
            self.assertLessEqual(scheduled_cutoff(resolved), created + timedelta(hours=1))
            self.assertLessEqual(resolved, created.date())


class TestCutoffWait(unittest.TestCase):
    def test_the_wait_ends_exactly_at_the_frozen_cutoff(self) -> None:
        slept: list[float] = []
        waited = wait_for_cutoff(SLOT, now=_local(SLOT, "22:30"), sleep=slept.append)
        self.assertEqual(waited, 3600.0)
        self.assertEqual(slept, [3600.0])

    def test_a_run_already_past_the_cutoff_does_not_wait(self) -> None:
        slept: list[float] = []
        self.assertEqual(
            wait_for_cutoff(SLOT, now=_local(SLOT, "23:45"), sleep=slept.append), 0.0
        )
        self.assertEqual(slept, [])

    def test_an_implausible_wait_fails_closed_rather_than_idling(self) -> None:
        far = _local(SLOT, "23:30") - MAX_CUTOFF_WAIT - timedelta(minutes=1)
        with self.assertRaisesRegex(CaptureTimeError, "resolved wrongly"):
            seconds_until_cutoff(SLOT, far)


class TestNoPrematureDurableCapture(unittest.TestCase):
    def _root(self, base: Path) -> Path:
        root = base / "benchmark"
        root.mkdir()
        protocol = b'{"frozen":true}\n'
        digest = hashlib.sha256(protocol).hexdigest()
        (root / "protocol.json").write_bytes(protocol)
        (root / "protocol.sha256").write_text(f"{digest}  protocol.json\n", encoding="utf-8")
        (root / "index.json").write_text(
            json.dumps({
                "schema_version": "1.0",
                "protocol_path": "protocol.json",
                "protocol_sha256": digest,
                "captures": [],
            }) + "\n",
            encoding="utf-8",
        )
        return root

    def test_durable_classification_before_the_cutoff_is_refused(self) -> None:
        with self.assertRaisesRegex(CaptureTimeError, "cannot be created before its scheduled cutoff"):
            classify_capture_time(SLOT, _local(SLOT, "23:29"), durable=True)

    def test_prewarm_time_cannot_retrieve_or_record_anything(self) -> None:
        """A durable capture attempted during prewarm touches no source."""

        calls: list[str] = []

        def es_collector(*_args, **_kwargs):
            calls.append("election_simulator")
            raise AssertionError("no source may be contacted before the cutoff")

        def ada_collector(*_args, **_kwargs):
            calls.append("botten_ada")
            raise AssertionError("no source may be contacted before the cutoff")

        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(Path(tmp))
            with self.assertRaises(CaptureTimeError):
                run_capture(
                    scheduled_date=SLOT,
                    mode="capture",
                    archive_root=root,
                    es_archive_root=root,
                    repo_root=root,
                    es_collector=es_collector,
                    ada_collector=ada_collector,
                    _clock=lambda: _local(SLOT, "22:31"),
                )
            self.assertEqual(calls, [])
            self.assertFalse((root / "captures").exists())
            self.assertEqual(json.loads((root / "index.json").read_text())["captures"], [])


class TestSlotTimingOutcome(unittest.TestCase):
    """"Archive intact but nothing scorable" must not read as success."""

    def test_an_archived_but_ineligible_slot_is_named_distinctly(self) -> None:
        for day in ("2026-09-04", "2026-09-05"):
            with self.subTest(day):
                outcome = slot_timing_outcome(ARCHIVE, day)
                self.assertEqual(outcome["outcome"], "ARCHIVED_NOT_TIMING_ELIGIBLE")
                self.assertEqual(outcome["timing_status"], "LATE_EXCLUDED")
                self.assertFalse(outcome["timing_eligible"])
                self.assertEqual(outcome["capture_status"], "COMPLETE")

    def test_a_slot_with_no_capture_is_distinguished_from_an_ineligible_one(self) -> None:
        outcome = slot_timing_outcome(ARCHIVE, "2026-09-12")
        self.assertEqual(outcome["outcome"], "NO_DURABLE_CAPTURE")
        self.assertIsNone(outcome["capture_id"])
        self.assertFalse(outcome["timing_eligible"])

    def test_the_earlier_start_does_not_reclassify_september_4_or_5(self) -> None:
        index = json.loads((ARCHIVE / "index.json").read_text())
        recorded = {
            row["scheduled_date"]: (row["timing_status"], row["timing_eligible"])
            for row in index["captures"]
        }
        self.assertEqual(recorded["2026-09-04"], ("LATE_EXCLUDED", False))
        self.assertEqual(recorded["2026-09-05"], ("LATE_EXCLUDED", False))


class TestWorkflowSchedulingContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = WORKFLOW.read_text(encoding="utf-8")

    def test_cron_starts_before_the_cutoff_and_matches_the_frozen_constant(self) -> None:
        start_utc = _local(SLOT, SCHEDULED_START_LOCAL_TIME.isoformat()).utctimetuple()
        self.assertIn(f'cron: "{start_utc[4]} {start_utc[3]} * 9 *"', self.text)
        # The old schedule began at the cutoff itself.
        self.assertNotIn('cron: "30 21 * 9 *"', self.text)

    def test_the_slot_guard_uses_the_shared_python_rule(self) -> None:
        self.assertIn("time_rules.py \\\n              resolve-slot --run-created-at", self.text)
        self.assertIn('gh api "repos/$REPOSITORY/actions/runs/$RUN_ID" --jq .created_at', self.text)
        # No shell reimplementation of the attribution boundary.
        self.assertNotIn('if [[ "$STOCKHOLM_TIME" < "23:30" ]]', self.text)
        # A failed resolution must not fall through to a skipped slot.
        self.assertIn("Cannot resolve the intended Stockholm slot", self.text)

    def test_the_capture_waits_for_the_cutoff_after_the_gates(self) -> None:
        gates = self.text.index("Run prospective capture gates")
        wait = self.text.index("Wait for the frozen 23:30 Europe/Stockholm cutoff")
        capture = self.text.index("Capture official contemporaneous evidence")
        self.assertLess(gates, wait)
        self.assertLess(wait, capture)
        self.assertIn("wait-for-cutoff --scheduled-date", self.text)

    def test_the_timeout_accommodates_the_deliberate_wait(self) -> None:
        budget = timedelta(minutes=120)
        self.assertIn(f"timeout-minutes: {int(budget.total_seconds() // 60)}", self.text)
        self.assertGreater(
            budget,
            MAX_CUTOFF_WAIT,
            "the job must be allowed to outlive its longest permitted cutoff wait",
        )

    def test_the_external_trigger_shares_the_scheduled_slot_rule(self) -> None:
        # An external dispatch is a different event with different provenance,
        # so the one thing it must not do is name its own slot. It takes the
        # same created_at branch, which is why its slot provenance is
        # identical to the cron's rather than merely equivalent to it.
        self.assertIn("scheduled_capture", self.text)
        self.assertIn('if [[ "$UNATTENDED" == "true" ]]; then', self.text)
        guard = self.text[self.text.index("Resolve intended scheduled slot"):self.text.index("  dry_run:")]
        unattended = guard[guard.index('if [[ "$UNATTENDED" == "true" ]]; then'):]
        self.assertIn("resolve-slot --run-created-at", unattended[:unattended.index("else")])

    def test_no_external_tick_may_precede_the_slot_attribution_boundary(self) -> None:
        # The boundary is SCHEDULED_START_LOCAL_TIME, 20:30Z in-window. A tick
        # before it resolves the PREVIOUS slot and could durably occupy it.
        boundary = _local(SLOT, SCHEDULED_START_LOCAL_TIME.isoformat()).astimezone(timezone.utc)
        crons = re.findall(
            r'"(\d+) (\d+) [^"]*"',
            (ROOT / "ops" / "benchmark-trigger-worker" / "wrangler.toml").read_text(),
        )
        self.assertTrue(crons, "the external Worker declares no crons")
        for minute, hour in crons:
            with self.subTest(cron=f"{minute} {hour}"):
                self.assertGreaterEqual(
                    (int(hour), int(minute)), (boundary.hour, boundary.minute))

    def test_an_ineligible_scheduled_slot_fails_the_run(self) -> None:
        push = self.text.index("Commit and push the append-only capture")
        assertion = self.text.index("Require a timing-eligible capture for the scheduled slot")
        # The evidence is committed first: an append-only archive keeps the
        # late record, and only the run's status goes red.
        self.assertLess(push, assertion)
        self.assertIn("assert-eligible", self.text)
        self.assertIn("ARGS+=(--warn-only)", self.text)


class TestAmendmentRecordsTheOperationalChange(unittest.TestCase):
    def test_amendment_005_is_append_only_and_changes_no_scoring_rule(self) -> None:
        path = ARCHIVE / "amendments" / "005-prewarmed-capture-start.json"
        amendment = json.loads(path.read_text())
        index = json.loads((ARCHIVE / "index.json").read_text())
        self.assertEqual(amendment["amendment_number"], 5)
        self.assertIs(amendment["immutable"], True)
        # protocol.json is superseded additively, never rewritten.
        self.assertEqual(amendment["original_protocol_sha256"], index["protocol_sha256"])
        self.assertTrue(amendment["primary_scoring_effect"].startswith("NONE"))
        self.assertEqual(amendment["replacement_rule"]["frozen_cutoff_unchanged_utc"], "21:30:00Z")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sidecar = (ARCHIVE / "amendments" / "005-prewarmed-capture-start.sha256").read_text()
        self.assertTrue(sidecar.startswith(digest))
        self.assertIn(
            {"amendment_number": 5, "path": f"amendments/{path.name}", "sha256": digest,
             "primary_scoring_effect": amendment["primary_scoring_effect"]},
            index["amendments"],
        )

    def test_the_frozen_protocol_cutoff_field_is_untouched(self) -> None:
        protocol = json.loads((ARCHIVE / "protocol.json").read_text())
        self.assertEqual(protocol["schedule"]["local_cutoff_time"], "23:30:00")
        self.assertEqual(protocol["schedule"]["scheduled_utc_time_during_window"], "21:30:00Z")

    def test_amendment_006_is_append_only_and_changes_no_scoring_rule(self) -> None:
        # Amendment 005 moved the start and the slot was still lost, because
        # the delay is applied to the start. 006 adds a second deliverer of
        # the same trigger; it must change nothing else.
        path = ARCHIVE / "amendments" / "006-external-independent-capture-trigger.json"
        amendment = json.loads(path.read_text())
        index = json.loads((ARCHIVE / "index.json").read_text())
        self.assertEqual(amendment["amendment_number"], 6)
        self.assertIs(amendment["immutable"], True)
        # protocol.json is superseded additively, never rewritten.
        self.assertEqual(amendment["original_protocol_sha256"], index["protocol_sha256"])
        self.assertTrue(amendment["primary_scoring_effect"].startswith("NONE"))
        rule = amendment["replacement_rule"]
        self.assertIn("retrieved_at_utc", rule["authoritative_timing_evidence"])
        # created_at attributes the slot; it is not proof of who fired the run.
        self.assertIn("not evidence of scheduler identity", rule["authoritative_timing_evidence"])
        self.assertIn("RETAINED", rule["github_cron_status"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sidecar = (ARCHIVE / "amendments" / "006-external-independent-capture-trigger.sha256").read_text()
        self.assertTrue(sidecar.startswith(digest))
        self.assertIn(
            {"amendment_number": 6, "path": f"amendments/{path.name}", "sha256": digest,
             "primary_scoring_effect": amendment["primary_scoring_effect"]},
            index["amendments"],
        )

    def test_amendment_006_does_not_rewrite_its_predecessors(self) -> None:
        directory = ARCHIVE / "amendments"
        for number, stem in enumerate(
            ("001-operational-clarifications", "002-standard-wis-normalization",
             "003-exact-deterministic-replay", "004-intraday-exact-draw-sidecar-retention",
             "005-prewarmed-capture-start"),
            start=1,
        ):
            with self.subTest(amendment=number):
                amendment = json.loads((directory / f"{stem}.json").read_text())
                self.assertEqual(amendment["amendment_number"], number)
                digest = hashlib.sha256((directory / f"{stem}.json").read_bytes()).hexdigest()
                self.assertTrue((directory / f"{stem}.sha256").read_text().startswith(digest))

    def test_the_september_4_to_6_records_keep_their_late_label(self) -> None:
        # Amendment 006 must not reclassify, recapture or disturb them.
        recorded = {
            row["scheduled_date"]: (row["timing_status"], row["timing_eligible"])
            for row in json.loads((ARCHIVE / "index.json").read_text())["captures"]
        }
        for slot in ("2026-09-04", "2026-09-05", "2026-09-06"):
            with self.subTest(slot=slot):
                self.assertEqual(recorded[slot], ("LATE_EXCLUDED", False))


if __name__ == "__main__":
    unittest.main()
