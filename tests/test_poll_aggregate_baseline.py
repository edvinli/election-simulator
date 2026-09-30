from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from scripts.poll_aggregate import consensus_baseline as B
from scripts.poll_aggregate import replay as R
from scripts.poll_aggregate.data import Observation

BASE = (19.0, 5.0, 7.0, 6.0, 28.0, 8.0, 6.0, 18.0)


def poll(key, house, mid, published, m=19.0, n=1000.0) -> Observation:
    shares = (m,) + BASE[1:]
    return Observation(key, "poll", mid, published, shares, house, n, 0.1)


class ConsensusBaselineTests(unittest.TestCase):
    def test_latest_poll_per_pollster_weighted_by_sample(self) -> None:
        d = date(2026, 9, 1)
        polls = [
            poll("sifo-old", "Sifo", d - timedelta(days=20), d - timedelta(days=18), m=10.0),
            poll("sifo-new", "Sifo", d - timedelta(days=5), d - timedelta(days=3), m=20.0, n=3000.0),
            poll("novus", "Novus", d - timedelta(days=10), d - timedelta(days=8), m=24.0, n=1000.0),
        ]
        (row,) = B.consensus_rows(polls, d, d)
        self.assertAlmostEqual(row["shares"][0], (20.0 * 3000 + 24.0 * 1000) / 4000)
        self.assertEqual(row["pollsters"], 2)

    def test_polls_outside_the_window_or_unpublished_are_ignored(self) -> None:
        d = date(2026, 9, 1)
        polls = [
            poll("stale", "Skop", d - timedelta(days=B.WINDOW_DAYS + 1), d - timedelta(days=55), m=50.0),
            poll("future", "Ipsos", d - timedelta(days=2), d + timedelta(days=1), m=50.0),
            poll("ok", "Sifo", d - timedelta(days=3), d - timedelta(days=1), m=20.0),
        ]
        (row,) = B.consensus_rows(polls, d, d)
        self.assertEqual((row["shares"][0], row["pollsters"]), (20.0, 1))

    def test_later_polls_never_change_earlier_rows(self) -> None:
        d = date(2026, 9, 1)
        polls = [poll(f"p{i}", h, d - timedelta(days=i), d - timedelta(days=i - 1))
                 for i, h in enumerate(("Sifo", "Novus", "Ipsos", "Sifo", "Novus"), start=2)]
        late = poll("late", "Skop", d - timedelta(days=4), d + timedelta(days=2), m=40.0)
        a = B.consensus_rows(polls, d - timedelta(days=5), d + timedelta(days=1))
        b = B.consensus_rows([*polls, late], d - timedelta(days=5), d + timedelta(days=1))
        self.assertEqual(a, b)

    def test_committed_series_is_daily_and_causal(self) -> None:
        import csv, json
        path = B.DEFAULT_BASELINE_DIR / "swedishpolls_aggregate_timeseries.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        days = [date.fromisoformat(r["date"]) for r in rows]
        self.assertEqual(days, [days[0] + timedelta(days=i) for i in range(len(days))])
        self.assertTrue(all(json.loads(r["source_extra_json"])["information_date"] <= r["date"] for r in rows))


class ReplayV4ConfigTests(unittest.TestCase):
    def test_v4_controls_on_the_consensus_baseline_not_pop(self) -> None:
        self.assertEqual(R.REPLAY_V4.arms, (R.BASELINE, R.CANDIDATE))
        self.assertEqual(R.REPLAY_V4.control, R.BASELINE)
        self.assertEqual(R.REPLAY_V4.aggregate_file, R.REPLAY_V2.aggregate_file)

    def test_baseline_arm_differs_from_the_candidate_only_in_its_timeseries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            roots = {arm: R.build_data_root(arm, Path(tmp), R.REPLAY_V4.aggregate_file,
                                            R.REPLAY_V4.control_timeseries) for arm in R.REPLAY_V4.arms}
            link = lambda arm, name: Path(os.readlink(roots[arm] / "pollofpolls" / name))
            self.assertEqual(link(R.BASELINE, "pollofpolls_timeseries.csv"), R.REPLAY_V4.control_timeseries)
            self.assertEqual(link(R.CANDIDATE, "pollofpolls_timeseries.csv"), R.REPLAY_V4.aggregate_file)
            for arm in R.REPLAY_V4.arms:
                self.assertFalse((roots[arm] / "pollofpolls" / "individual_polls.csv").is_symlink())
            a = (roots[R.BASELINE] / "pollofpolls" / "individual_polls.csv").read_bytes()
            b = (roots[R.CANDIDATE] / "pollofpolls" / "individual_polls.csv").read_bytes()
            self.assertEqual(a, b)

    def test_a_series_control_must_match_its_own_file(self) -> None:
        row = {"arm": R.BASELINE, "as_of": "2018-08-12", "selected_estimate_date": "2018-08-12",
               "aggregate_row_matches": False, "aggregate_information_date": "2018-08-10"}
        self.assertFalse(R.case_dates_ok(row))
        row["aggregate_row_matches"] = True
        self.assertTrue(R.case_dates_ok(row))


if __name__ == "__main__":
    unittest.main()
