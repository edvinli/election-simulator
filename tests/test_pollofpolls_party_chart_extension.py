from __future__ import annotations

import csv
import unittest
from datetime import date
from pathlib import Path

from scripts.pollofpolls.normalize import (
    MAX_DISAGREEMENT_RATE,
    STRUCTURAL_DISAGREEMENT_PP,
    PartyChartDisagreement,
    extract_party_chart_pop_timeseries,
    parse_party_chart_pop_series,
)
from scripts.pollofpolls.validate import PARTY_CHART_TIMESERIES_FIELDS


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPOSITORY_ROOT / "data" / "raw" / "pollofpolls"
PROCESSED_TIMESERIES_PATH = (
    REPOSITORY_ROOT / "data" / "processed" / "pollofpolls" / "pollofpolls_timeseries.csv"
)
PARLIAMENTARY_PARTIES = ("M", "L", "C", "KD", "S", "V", "MP", "SD")

#: The series is daily and contiguous, so its length is fully determined by its
#: endpoints -- asserting a row count as well only restated the size of
#: whatever data happened to be checked in, and broke on every polling refresh.
#: The first date is a real claim (the party chart begins here) and is pinned.
#: The last date is bounded instead: it may only move forward, and it may never
#: reach into the future, which is the property a poll-of-polls series has to
#: hold and which an exact literal never checked.
SERIES_FIRST_DATE = date(2009, 1, 2)
SERIES_COVERED_THROUGH_AT_LEAST = date(2026, 8, 24)


class PollofpollsPartyChartExtensionTests(unittest.TestCase):
    def test_parse_party_chart_pop_series_returns_dates_and_floats(self) -> None:
        csv_sample = (
            "date,Val,pofp\n"
            "2009-01-02,,27.2\n"
            "2009-01-03,,27.2\n"
        ).encode("utf-8")
        parsed = parse_party_chart_pop_series(csv_sample, "M")
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[date(2009, 1, 2)], 27.2)
        self.assertEqual(parsed[date(2009, 1, 3)], 27.2)

    def test_extract_party_chart_pop_timeseries_starts_in_2009(self) -> None:
        rows = extract_party_chart_pop_timeseries(RAW_DIR)
        dates = [date.fromisoformat(r["date"]) for r in rows]

        self.assertEqual(dates[0], SERIES_FIRST_DATE)
        self.assertGreaterEqual(
            dates[-1], SERIES_COVERED_THROUGH_AT_LEAST,
            "the party chart series lost coverage it previously had")
        self.assertLessEqual(
            dates[-1], date.today(),
            "the party chart series extends into the future")

        self.assertEqual(dates, sorted(dates))
        self.assertEqual(len(set(dates)), len(dates))
        # Daily and gapless: with sortedness and uniqueness above, this pins
        # the row count exactly without naming a number that drifts.
        self.assertEqual(len(rows), (dates[-1] - dates[0]).days + 1)

        first = rows[0]
        for party in PARLIAMENTARY_PARTIES:
            self.assertIn(party, first)
            self.assertIsInstance(first[party], float)
            self.assertGreater(first[party], 0.0)

    def test_the_overlap_is_taken_from_the_canonical_series(self) -> None:
        """Canonical is authoritative wherever the two feeds overlap.

        The party charts exist to reach back past the canonical 2014+ series.
        Letting them also restate the overlap is what made a provider revision
        able to stop a publication, so on overlapping dates the returned rows
        carry the canonical value and the party charts contribute only the
        years canonical does not cover.
        """

        with PROCESSED_TIMESERIES_PATH.open("r", encoding="utf-8") as handle:
            canonical_rows = list(csv.DictReader(handle))
        canonical_by_date = {r["date"]: r for r in canonical_rows}

        rows = extract_party_chart_pop_timeseries(
            RAW_DIR, canonical_timeseries=canonical_rows)

        overlap = [row for row in rows if row["date"] in canonical_by_date]
        self.assertEqual(len(overlap), len(canonical_rows))
        for row in overlap:
            canon = canonical_by_date[row["date"]]
            for party in PARLIAMENTARY_PARTIES:
                if canon.get(party) not in (None, ""):
                    self.assertEqual(
                        row[party], float(canon[party]),
                        f"{row['date']} {party} did not come from canonical")

        # ...and the reach-back is still there, which is the entire point of
        # consulting the party charts at all.
        before_canonical = [row for row in rows if row["date"] not in canonical_by_date]
        self.assertTrue(before_canonical)
        self.assertLess(
            min(row["date"] for row in before_canonical),
            min(canonical_by_date),
        )

    def test_a_provider_revision_is_reported_and_does_not_raise(self) -> None:
        """The 2026-09-11 blocker, as a test.

        Upstream revised one party chart value by a tenth of a point and left
        the canonical series alone. Acquisition refused the whole dataset, no
        simulation ran, and nothing published. A tenth of a point is a
        revision, not corruption: it is recorded, canonical wins, and the
        refresh continues.
        """

        revised_date = "2024-03-22"
        party_value = self._party_chart_value(revised_date, "M")
        canonical_value = round(party_value + 0.1, 6)
        canonical = [self._canonical_row(revised_date, M=canonical_value)]

        disagreements: list[dict] = []
        rows = extract_party_chart_pop_timeseries(
            RAW_DIR,
            canonical_timeseries=canonical,
            disagreements=disagreements,
        )

        self.assertEqual(len(disagreements), 1)
        reported = disagreements[0]
        self.assertEqual(reported["date"], revised_date)
        self.assertEqual(reported["party"], "M")
        self.assertEqual(reported["party_chart_pofp"], party_value)
        self.assertEqual(reported["canonical"], canonical_value)
        self.assertAlmostEqual(reported["difference_pp"], 0.1, places=6)

        revised_row = next(row for row in rows if row["date"] == revised_date)
        self.assertEqual(revised_row["M"], canonical_value)

    def test_an_agreeing_overlap_reports_nothing(self) -> None:
        """The report has to stay empty when there is nothing to report."""

        agreeing = "2024-03-22"
        canonical = [self._canonical_row(
            agreeing,
            **{p: self._party_chart_value(agreeing, p) for p in PARLIAMENTARY_PARTIES},
        )]
        disagreements: list[dict] = []
        extract_party_chart_pop_timeseries(
            RAW_DIR,
            canonical_timeseries=canonical,
            disagreements=disagreements,
        )
        self.assertEqual(disagreements, [])

    def test_a_disagreement_past_the_per_value_limit_still_fails(self) -> None:
        """Reconciling the overlap must not absorb an unexplained jump.

        A value off by points rather than tenths is past what a revision
        explains, and the code is in no position to say what caused it -- so it
        stops and says so, without claiming to have diagnosed corruption.
        """

        broken_date = "2026-08-24"
        party_value = self._party_chart_value(broken_date, "M")
        canonical_value = round(party_value - STRUCTURAL_DISAGREEMENT_PP - 2.0, 6)
        canonical = [self._canonical_row(broken_date, M=canonical_value)]

        with self.assertRaises(PartyChartDisagreement) as context:
            extract_party_chart_pop_timeseries(
                RAW_DIR, canonical_timeseries=canonical)
        message = str(context.exception)
        self.assertIn("per-value limit", message)
        self.assertIn("not established", message)
        self.assertIn(broken_date, message)
        self.assertIn("party M", message)
        # A ValueError subclass, so acquisition error handling is unchanged.
        self.assertIsInstance(context.exception, ValueError)

    def test_a_rejection_still_carries_the_whole_inventory(self) -> None:
        """The point at which the inventory matters most is the rejection.

        A probe that refuses the feed and reports one example is not a
        diagnostic. Every disagreement travels on the exception, and the
        message carries a bounded summary of them -- count, worst value, the
        parties and dates touched -- because the message is what reaches a CI
        log.
        """

        broken_date = "2026-08-24"
        # One value far past the per-value limit, and a second, smaller one on
        # another party that on its own would have been absorbed silently.
        big = round(self._party_chart_value(broken_date, "M") - 3.0, 6)
        small = round(self._party_chart_value(broken_date, "S") + 0.2, 6)
        canonical = [self._canonical_row(broken_date, M=big, S=small)]

        with self.assertRaises(PartyChartDisagreement) as context:
            extract_party_chart_pop_timeseries(
                RAW_DIR, canonical_timeseries=canonical)

        caught = context.exception
        self.assertEqual(
            {(d["party"], d["date"]) for d in caught.disagreements},
            {("M", broken_date), ("S", broken_date)},
            "the sub-limit disagreement was dropped from the inventory",
        )
        self.assertEqual(caught.compared, 2)
        self.assertIn("2 of 2 compared values differ", str(caught))
        self.assertIn("parties M, S", str(caught))

    def test_the_inventory_reaches_the_caller_when_it_rejects(self) -> None:
        """The out-parameter is filled before the failure propagates."""

        broken_date = "2026-08-24"
        canonical = [self._canonical_row(
            broken_date,
            M=round(self._party_chart_value(broken_date, "M") - 3.0, 6),
        )]
        collected: list[dict] = []
        with self.assertRaises(PartyChartDisagreement):
            extract_party_chart_pop_timeseries(
                RAW_DIR,
                canonical_timeseries=canonical,
                disagreements=collected,
            )
        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0]["party"], "M")

    def test_many_small_disagreements_still_fail(self) -> None:
        """Prevalence is the other half of the check.

        No single value here crosses the per-value limit, so that limit alone
        would absorb the whole feed. Prevalence matters because before 2014
        there is no canonical series to compare against: a party feed that
        changed character takes the years nobody can check along with it.
        """

        with PROCESSED_TIMESERIES_PATH.open("r", encoding="utf-8") as handle:
            canonical_rows = list(csv.DictReader(handle))
        nudged = []
        for row in canonical_rows:
            shifted = dict(row)
            for party in PARLIAMENTARY_PARTIES:
                if shifted.get(party) not in (None, ""):
                    shifted[party] = f"{float(shifted[party]) + 0.1:.6f}"
            nudged.append(shifted)

        with self.assertRaises(PartyChartDisagreement) as context:
            extract_party_chart_pop_timeseries(
                RAW_DIR, canonical_timeseries=nudged)
        message = str(context.exception)
        self.assertIn("prevalence limit", message)
        self.assertIn(f"{MAX_DISAGREEMENT_RATE:.4%}", message)
        # Stated as a limit that was crossed, not as a diagnosis.
        self.assertIn("not established", message)
        self.assertNotIn("corrupt", message.lower())

    @staticmethod
    def _party_chart_value(iso_date: str, party: str) -> float:
        """The committed party chart's own value, so fixtures never hardcode it."""

        rows = extract_party_chart_pop_timeseries(RAW_DIR)
        row = next(row for row in rows if row["date"] == iso_date)
        return float(row[party])

    @staticmethod
    def _canonical_row(iso_date: str, **values: float) -> dict:
        """A canonical row carrying only the parties the caller names.

        Parties left out are blank, which the reconciliation treats as "the
        canonical feed does not know" -- neither a disagreement nor an
        override.
        """

        row = {"date": iso_date}
        row.update({party: "" for party in PARLIAMENTARY_PARTIES})
        row.update({party: f"{value:.6f}" for party, value in values.items()})
        return row


if __name__ == "__main__":
    unittest.main()
