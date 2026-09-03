from __future__ import annotations

import csv
import unittest
from datetime import date
from pathlib import Path

from scripts.pollofpolls.normalize import (
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

    def test_party_chart_matches_canonical_timeseries_with_zero_discrepancy(self) -> None:
        with PROCESSED_TIMESERIES_PATH.open("r", encoding="utf-8") as handle:
            canonical_rows = list(csv.DictReader(handle))

        # This will raise ValueError if any discrepancy exceeds tolerance=1e-4
        rows = extract_party_chart_pop_timeseries(
            RAW_DIR,
            canonical_timeseries=canonical_rows,
            discrepancy_tolerance=1e-4,
        )

        canonical_by_date = {r["date"]: r for r in canonical_rows}
        overlap_count = 0
        max_diff = 0.0
        for row in rows:
            iso = row["date"]
            if iso in canonical_by_date:
                overlap_count += 1
                canon = canonical_by_date[iso]
                for party in PARLIAMENTARY_PARTIES:
                    if canon.get(party) not in (None, ""):
                        diff = abs(float(row[party]) - float(canon[party]))
                        if diff > max_diff:
                            max_diff = diff

        self.assertEqual(overlap_count, len(canonical_rows))
        self.assertEqual(max_diff, 0.0)

    def test_discrepancy_tolerance_enforcement_raises_on_mismatch(self) -> None:
        synthetic_canonical = [
            {
                "date": "2026-08-24",
                "M": "15.0",  # actual is 18.2 in party_M.csv
                "L": "2.0",
                "C": "7.0",
                "KD": "6.5",
                "S": "30.4",
                "V": "7.5",
                "MP": "7.4",
                "SD": "19.1",
            }
        ]
        with self.assertRaises(ValueError) as context:
            extract_party_chart_pop_timeseries(
                RAW_DIR,
                canonical_timeseries=synthetic_canonical,
                discrepancy_tolerance=0.01,
            )
        self.assertIn("Discrepancy on 2026-08-24 for party M", str(context.exception))


if __name__ == "__main__":
    unittest.main()
