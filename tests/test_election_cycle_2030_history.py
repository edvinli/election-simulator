"""History and automation for a new election cycle.

The 2026 history is decided and frozen; the website keeps it as an archive of
its own. What these tests guard is that a history for the 2030 election:

* starts as a fresh artifact at the first Poll of Polls estimate after the
  2026 election, and contains no 2026 point;
* is scheduled from its own election (weekly anchors, then daily from its own
  dynamics cap), and has every publication day filled in, not only the daily
  part -- otherwise the chart would break its line between weekly anchors;
* carries no future views outside the final 112 days;
* is never certified on a pre-election estimate.

Every 2026 answer is asserted to be what it was before.
"""

from __future__ import annotations

import csv
from datetime import date
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from scripts.election_automation_base import (
    AwaitingPostElectionPolls,
    future_views_required,
    guard_cycle_started,
    required_history_views,
)
from scripts.forecast_history.generate import (
    HISTORY_CAP_DATE,
    HISTORY_START_DATE,
    _archive_point_from_record,
    _load_archive_records,
    build_history_dates,
    first_post_election_timeseries_date,
    missing_curve_dates,
)
from scripts.forecast_history.contract import DEFAULT_COALITIONS, deterministic_history_sha256
from scripts.forecast_history.future_projection import roll_in_certified_point
from scripts.publication_pipeline.pipeline import _load_prior_snapshot
from scripts.site_publisher import sync_history_to_site
from scripts.simulator.model_inputs import AGGREGATE_POLLS, AGGREGATE_TIMESERIES

try:
    from . import history_fixtures as fixtures
except ImportError:  # pragma: no cover - direct module execution
    import history_fixtures as fixtures


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
ARCHIVE = PROCESSED / "prospective_forecasts"
# After the certified 2026 result, before the first 2030 publication
# (2026-09-29). The committed archive and polling files move with every
# production publication; tests about "the first 2030 forecast" pin their
# inputs to this moment instead of reading whatever was published last.
FROZEN_BEFORE_FIRST_2030 = "2026-09-20"
# The last commit whose live history artifact was the 2026 one.
LAST_2026_HISTORY_COMMIT = "0906e6442ad39c1e39c21192909323a51e3de67f"


def frozen_archive(parent: Path, as_of: str = FROZEN_BEFORE_FIRST_2030) -> Path:
    """The committed archive as it stood on ``as_of``: pruned index, linked generations."""

    root = parent / "prospective_forecasts"
    root.mkdir()
    for child in ARCHIVE.iterdir():
        if child.name == "index.json":
            shutil.copyfile(child, root / child.name)
        else:
            (root / child.name).symlink_to(child, target_is_directory=child.is_dir())
    fixtures.freeze_archive_inputs(root, as_of=as_of)
    return root
E2026 = date(2026, 9, 13)
E2030 = date(2030, 9, 8)


class ScheduleTests(unittest.TestCase):
    def test_the_2026_schedule_is_unchanged(self) -> None:
        explicit = build_history_dates(start_date=HISTORY_START_DATE, latest_date=E2026,
                                       election_date=E2026, dynamics_cap_date=HISTORY_CAP_DATE)
        self.assertEqual(build_history_dates(latest_date=E2026, election_date=E2026), explicit)

    def test_the_2030_schedule_is_its_own(self) -> None:
        dates = build_history_dates(latest_date=date(2026, 10, 18), election_date=E2030)
        self.assertEqual(dates[0], date(2026, 9, 20))
        self.assertEqual([d.isoformat() for d in dates],
                         ["2026-09-20", "2026-09-27", "2026-10-04", "2026-10-11", "2026-10-18"])
        daily = build_history_dates(latest_date=date(2030, 5, 21), election_date=E2030)
        self.assertEqual(daily[-3:], [date(2030, 5, 19), date(2030, 5, 20), date(2030, 5, 21)])


def _payload(points: list[tuple[str, str]], *, election: date, start: str | None) -> dict:
    return {
        "election_date": election.isoformat(),
        "schedule": {"cycle_start_date": start} if start else {},
        "series": [{"date": day, "provenance": provenance} for day, provenance in points],
    }


class MissingCurveDatesTests(unittest.TestCase):
    def test_publication_days_between_weekly_anchors_are_holes(self) -> None:
        # Weekly regime: each daily publication leaves an archived-only date
        # behind, and the chart would break its line at every one of them.
        payload = _payload([
            ("2026-09-28", "reconstructed_current_model"),
            ("2026-09-30", "prospective_archived"),
            ("2026-10-01", "prospective_archived"),
            ("2026-10-02", "current_production"),
        ], election=E2030, start="2026-09-28")
        self.assertEqual(missing_curve_dates(payload), [date(2026, 9, 30), date(2026, 10, 1)])

    def test_a_due_weekly_anchor_is_a_hole(self) -> None:
        payload = _payload([
            ("2026-09-28", "reconstructed_current_model"),
            ("2026-10-06", "current_production"),
        ], election=E2030, start="2026-09-28")
        self.assertEqual(missing_curve_dates(payload), [date(2026, 10, 5)])

    def test_the_schedule_starts_where_the_history_says(self) -> None:
        payload = _payload([("2026-10-02", "current_production")], election=E2030, start="2026-10-02")
        self.assertEqual(missing_curve_dates(payload), [])

    def test_the_deployed_2026_history_is_complete(self) -> None:
        # The live path holds the 2030 history since the first 2030
        # publication (ec416d5); the 2026 history is the file as of the last
        # commit before it, which the website keeps frozen at history/2026/.
        shown = subprocess.run(
            ["git", "show", f"{LAST_2026_HISTORY_COMMIT}:files/election-simulator/history/coalition-timeseries.json"],
            cwd=ROOT, capture_output=True, text=True)
        if shown.returncode != 0:
            self.skipTest("the last 2026 history commit is not in this checkout")
        live = json.loads(shown.stdout)
        self.assertEqual(live["election_date"], "2026-09-13")
        self.assertEqual(missing_curve_dates(live), [])


class ElectionFiltersTests(unittest.TestCase):
    def setUp(self) -> None:
        self.archive = frozen_archive(Path(self.enterContext(tempfile.TemporaryDirectory())))

    def test_a_2026_snapshot_is_never_a_2030_point(self) -> None:
        records = [record for record in _load_archive_records(self.archive) if isinstance(record.get("groups"), dict)]
        self.assertTrue(records)
        coalitions = {key: list(value) for key, value in DEFAULT_COALITIONS.items()}
        record = records[-1]
        self.assertIsNotNone(_archive_point_from_record(record, election_date=E2026, coalitions=coalitions))
        self.assertIsNone(_archive_point_from_record(record, election_date=E2030, coalitions=coalitions))

    def test_the_first_2030_forecast_has_no_prior(self) -> None:
        self.assertIsNotNone(_load_prior_snapshot(self.archive, "2026-10-01", "2026-09-13"))
        self.assertIsNone(_load_prior_snapshot(self.archive, "2026-10-01", "2030-09-08"))

    def test_a_published_2030_snapshot_is_never_a_2026_point_or_prior(self) -> None:
        # The other direction, on the live archive once a 2030 generation exists.
        coalitions = {key: list(value) for key, value in DEFAULT_COALITIONS.items()}
        records = [r for r in _load_archive_records(ARCHIVE)
                   if isinstance(r.get("groups"), dict) and r.get("election_date") == "2030-09-08"]
        if not records:
            self.skipTest("no 2030 generation is archived yet")
        for record in records:
            self.assertIsNone(_archive_point_from_record(record, election_date=E2026, coalitions=coalitions))
            self.assertIsNotNone(_archive_point_from_record(record, election_date=E2030, coalitions=coalitions))
        prior = _load_prior_snapshot(ARCHIVE, "2030-01-01", "2030-09-08")
        self.assertEqual(prior["election_date"], "2030-09-08")


class FutureViewsWindowTests(unittest.TestCase):
    def test_the_2026_answers_are_unchanged(self) -> None:
        self.assertEqual(required_history_views(origin_date="2026-09-10", election_date="2026-09-13"),
                         ("future_projection", "future_campaign_paths"))
        self.assertEqual(required_history_views(origin_date="2026-09-13", election_date="2026-09-13"),
                         ("future_projection",))

    def test_no_views_outside_the_final_112_days(self) -> None:
        self.assertFalse(future_views_required(origin_date="2026-10-01", election_date="2030-09-08"))
        self.assertEqual(required_history_views(origin_date="2030-05-18", election_date="2030-09-08"), ())
        self.assertEqual(required_history_views(origin_date="2030-05-19", election_date="2030-09-08"),
                         ("future_projection", "future_campaign_paths"))


class CycleStartGateTests(unittest.TestCase):
    def test_a_pre_election_estimate_cannot_start_the_cycle(self) -> None:
        with self.assertRaises(AwaitingPostElectionPolls):
            guard_cycle_started("2026-09-11", E2030)
        with self.assertRaises(AwaitingPostElectionPolls):
            guard_cycle_started("2026-09-13", E2030)
        guard_cycle_started("2026-09-14", E2030)

    def test_the_2026_cycle_is_unaffected(self) -> None:
        guard_cycle_started("2026-09-11", E2026)
        guard_cycle_started("UNAVAILABLE", E2030)


class NewCycleBootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.timeseries = self.tmp / "pollofpolls_timeseries.csv"
        source = PROCESSED / "pollofpolls" / "pollofpolls_timeseries.csv"
        with source.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.fields = list(rows[0].keys())
        # The series as it stood before any post-election estimate: the tests
        # add their own post-election rows. The committed series gains real
        # ones as pollofpolls.se publishes them.
        self.rows = [row for row in rows if row["date"] <= E2026.isoformat()]
        self.archive = frozen_archive(self.tmp)

    def _write_timeseries(self, extra_dates: list[str]) -> None:
        rows = list(self.rows)
        for day in extra_dates:
            rows.append({**rows[-1], "date": day})
        with self.timeseries.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(rows)

    def _result(self, as_of: str) -> SimpleNamespace:
        votes = np.tile(fixtures._VOTES, (2, 1))
        seats = np.tile(fixtures._SEATS, (2, 1))
        return SimpleNamespace(
            summary=SimpleNamespace(as_of=as_of, total_samples=len(votes)),
            vote_shares_matrix=votes,
            seats_matrix=seats,
            manifest={"source_git_commit": "a" * 40, "source_worktree_clean": True, "base_seed": 12345},
        )

    def _roll(self, as_of: str) -> dict:
        return roll_in_certified_point(
            fixtures.make_history_fixture(),
            self._result(as_of),
            poll_file=PROCESSED / "pollofpolls" / "swedishpolls_individual_polls.csv",
            timeseries_file=self.timeseries,
            archive_dir=self.archive,
            election_date=E2030,
            publication_generation="20261005T040000Z-0000c0de",
            deterministic_payload_sha256="b" * 64,
            generated_at_utc="2026-10-05T04:00:00+00:00",
            model_commit="a" * 40,
            source_worktree_clean=True,
        )

    def test_the_first_2030_publication_starts_a_fresh_history(self) -> None:
        self._write_timeseries(["2026-09-28", "2026-10-05"])
        self.assertEqual(first_post_election_timeseries_date(self.timeseries, E2030), date(2026, 9, 28))
        history = self._roll("2026-10-05")
        self.assertEqual(history["election_date"], "2030-09-08")
        self.assertEqual([(p["date"], p["provenance"]) for p in history["series"]],
                         [("2026-10-05", "current_production")])
        self.assertEqual(history["series"][0]["publication_generation"], "20261005T040000Z-0000c0de")
        # The live forecast waited for the 2026-09-28 poll; the history is
        # scheduled from the day after the election, so the curve joins the
        # 2026 history with no hole, and the backfill fills the weekly anchors.
        self.assertEqual(history["schedule"]["cycle_start_date"], "2026-09-14")
        self.assertEqual(missing_curve_dates(history),
                         [date(2026, 9, 14), date(2026, 9, 21), date(2026, 9, 28)])
        self.assertNotIn("future_projection", history)

    def test_publication_renders_a_complete_first_2030_history(self) -> None:
        """The shared publication/render pipeline, end to end, on the new cycle."""

        from scripts.election_automation_base import render_history_for_generation

        self._write_timeseries(["2026-09-28", "2026-10-05"])
        history, report = render_history_for_generation(
            fixtures.make_history_fixture(),
            self._result("2026-10-05"),
            poll_file=PROCESSED / "pollofpolls" / "swedishpolls_individual_polls.csv",
            timeseries_file=self.timeseries,
            archive_dir=self.archive,
            model_data_dir=PROCESSED,
            election_date=E2030,
            publication_generation="20261005T040000Z-0000c0de",
            deterministic_payload_sha256="b" * 64,
            generated_at_utc="2026-10-05T04:00:00+00:00",
            model_commit="a" * 40,
            history_workers=1,
            stage_callback=None,
        )
        self.assertEqual([(p["date"], p["provenance"]) for p in history["series"]],
                         [("2026-09-14", "reconstructed_current_model"),
                          ("2026-09-21", "reconstructed_current_model"),
                          ("2026-09-28", "reconstructed_current_model"),
                          ("2026-10-05", "current_production")])
        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(report["views"]["secondary_projection"], "NOT_REQUIRED_OUTSIDE_CAMPAIGN_WINDOW")
        self.assertNotIn("future_projection", history)
        self.assertNotIn("future_campaign_paths", history)
        self.assertTrue(all(p["date"] > "2026-09-13" for p in history["series"]))

    def test_no_post_election_estimate_no_cycle(self) -> None:
        self._write_timeseries([])
        with self.assertRaisesRegex(ValueError, "cannot start yet"):
            self._roll("2026-10-05")

    # Model 1.2.0: the production input is the SwedishPolls aggregate, which
    # writes a row every day; the cycle starts at the first publication of an
    # eligible poll whose fieldwork began after the previous election.
    def _aggregate_inputs(self, polls_published_by: str) -> Path:
        directory = self.tmp / "aggregate"
        directory.mkdir(exist_ok=True)
        for relative, column in ((AGGREGATE_TIMESERIES, "date"), (AGGREGATE_POLLS, "publication_date")):
            with (PROCESSED / relative).open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields, rows = reader.fieldnames, [r for r in reader if (r[column] or "") <= polls_published_by]
            with (directory / relative.name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        return directory / AGGREGATE_TIMESERIES.name

    def test_the_aggregate_cycle_starts_at_the_first_post_election_poll(self) -> None:
        # Sentio, fielded 2026-09-21..25, published 2026-09-28.
        self.timeseries = self._aggregate_inputs("2026-10-05")
        self.assertEqual(first_post_election_timeseries_date(self.timeseries, E2030), date(2026, 9, 28))
        history = self._roll("2026-10-05")
        self.assertEqual(history["schedule"]["cycle_start_date"], "2026-09-14")

    def test_a_published_later_cycle_start_is_moved_back_on_roll_in(self) -> None:
        # The first 2030 history was published with cycle_start_date
        # 2026-09-28 (its first post-election poll), leaving 09-14..09-27
        # unscheduled. Rolling a new point into it reschedules from 09-14.
        self._write_timeseries(["2026-09-28", "2026-10-05"])
        first = self._roll("2026-10-05")
        first["schedule"]["cycle_start_date"] = "2026-09-28"
        first["deterministic_content_sha256"] = deterministic_history_sha256(first)
        rolled = roll_in_certified_point(
            first, self._result("2026-10-06"),
            poll_file=PROCESSED / "pollofpolls" / "swedishpolls_individual_polls.csv",
            timeseries_file=self.timeseries, archive_dir=self.archive, election_date=E2030,
            publication_generation="20261006T040000Z-0000c0de", deterministic_payload_sha256="c" * 64,
            generated_at_utc="2026-10-06T04:00:00+00:00", model_commit="a" * 40, source_worktree_clean=True,
        )
        self.assertEqual(rolled["schedule"]["cycle_start_date"], "2026-09-14")
        self.assertEqual(missing_curve_dates(rolled)[:2], [date(2026, 9, 14), date(2026, 9, 21)])

    def test_daily_aggregate_rows_alone_never_start_a_cycle(self) -> None:
        # Rows exist for every day after the election, but no poll fielded
        # after it was published by 2026-09-27.
        self.timeseries = self._aggregate_inputs("2026-09-27")
        self.assertIsNone(first_post_election_timeseries_date(self.timeseries, E2030))
        with self.assertRaisesRegex(ValueError, "cannot start yet"):
            self._roll("2026-10-05")


class ArchivedHistoryIsUntouchedTests(unittest.TestCase):
    """The website's frozen 2026 history survives every history sync."""

    def test_sync_leaves_the_archive_alone(self) -> None:
        site = Path(self.enterContext(tempfile.TemporaryDirectory())) / "website"
        history_dir = site / "files" / "election-simulator" / "history"
        (history_dir / "2026").mkdir(parents=True)
        archive = history_dir / "2026" / "coalition-timeseries.json"
        index = history_dir / "cycles.json"
        archive.write_bytes(b'{"frozen": true}\n')
        index.write_bytes(b'{"cycles": []}\n')
        live = Path(self.enterContext(tempfile.TemporaryDirectory())) / "live.json"
        shutil.copyfile(ROOT / "files" / "election-simulator" / "history" / "coalition-timeseries.json", live)
        sync_history_to_site(site_repo=site, source_history_path=live)
        self.assertEqual(archive.read_bytes(), b'{"frozen": true}\n')
        self.assertEqual(index.read_bytes(), b'{"cycles": []}\n')
        self.assertEqual((history_dir / "coalition-timeseries.json").read_bytes(), live.read_bytes())


if __name__ == "__main__":
    unittest.main()
