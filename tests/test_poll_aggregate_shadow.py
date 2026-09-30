from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from scripts.poll_aggregate import shadow as S
from scripts.poll_aggregate import shadow_score as SC
from scripts.poll_aggregate.config import DEFAULT_POLLS_FILE, PARTIES, SPEC_V02, TIMESERIES_FILENAME
from scripts.poll_aggregate.data import Observation, load_elections, load_polls
from scripts.pollofpolls.config import SWEDISHPOLLS_CSV, SWEDISHPOLLS_SOURCES_CSV, TIMESERIES_CSV

RAW = S.REPOSITORY_ROOT / "data" / "raw" / "pollofpolls"
BASE = (19.0, 5.0, 7.0, 6.0, 28.0, 8.0, 6.0, 18.0)


def pop_csv(rows: list[tuple[str, tuple]]) -> bytes:
    lines = ["Datum,M,L,C,KD,S,V,MP,SD"] + [f"{d}," + ",".join(str(v) for v in vals) for d, vals in rows]
    return ("\n".join(lines) + "\n").encode()


def fetcher(pop: bytes | Exception):
    payloads = {
        SWEDISHPOLLS_CSV: (RAW / "swedishpolls_polls.csv").read_bytes(),
        SWEDISHPOLLS_SOURCES_CSV: (RAW / "swedishpolls_sources.csv").read_bytes(),
    }

    def fetch(url: str) -> bytes:
        if url == TIMESERIES_CSV:
            if isinstance(pop, Exception):
                raise pop
            return pop
        return payloads[url]

    return fetch


NOON_0928 = datetime(2026, 9, 28, 10, 0, tzinfo=timezone.utc)


class CaptureTests(unittest.TestCase):
    def test_pop_value_is_the_latest_complete_row_on_or_before_the_day(self) -> None:
        payload = pop_csv([("2026-09-20", BASE), ("2026-09-25", BASE), ("2026-09-30", BASE)])
        record = S.pop_record(payload, date(2026, 9, 28))
        self.assertEqual(record["row_date"], "2026-09-25")
        self.assertEqual(record["staleness_days"], 3)

    def test_a_failed_pop_fetch_is_missing_and_has_no_value(self) -> None:
        record = S.capture(NOON_0928, fetcher(TimeoutError("timed out")))
        self.assertEqual(record["pop"]["status"], "missing")
        self.assertNotIn("values", record["pop"])
        self.assertIn("TimeoutError", record["pop"]["error"])
        self.assertEqual(record["v02"]["status"], "ok")

    def test_a_block_page_is_missing_not_parsed(self) -> None:
        record = S.capture(NOON_0928, fetcher(b"<html><body>Access denied</body></html>"))
        self.assertEqual(record["pop"]["status"], "missing")

    def test_capture_is_dated_in_stockholm(self) -> None:
        late = datetime(2026, 9, 28, 22, 30, tzinfo=timezone.utc)  # 00:30 on 09-29 in Stockholm
        record = S.capture(late, fetcher(pop_csv([("2026-09-11", BASE)])))
        self.assertEqual(record["stockholm_date"], "2026-09-29")
        self.assertEqual(record["pop"]["staleness_days"], 18)

    def test_v02_value_equals_the_committed_v02_build(self) -> None:
        elections = load_elections(S.DEFAULT_ELECTION_RESULTS_FILE, [S.DEFAULT_2026_RESULT_MANIFEST])
        polls = load_polls(DEFAULT_POLLS_FILE, [e.obs_date for e in elections]).polls
        value = S.v02_value(polls, elections, date(2026, 9, 28), S.frozen_v02_segment())
        with (SPEC_V02.output_dir / TIMESERIES_FILENAME).open(encoding="utf-8", newline="") as handle:
            row = next(r for r in csv.DictReader(handle) if r["date"] == "2026-09-28")
        for p in PARTIES:
            self.assertAlmostEqual(round(value["values"][p], 2), float(row[p]), places=9)

    def test_raw_payloads_clean_to_the_committed_snapshot(self) -> None:
        elections = load_elections(S.DEFAULT_ELECTION_RESULTS_FILE, [S.DEFAULT_2026_RESULT_MANIFEST])
        live = S.swedishpolls_observations((RAW / "swedishpolls_polls.csv").read_bytes(),
                                           (RAW / "swedishpolls_sources.csv").read_bytes(), elections)
        committed = load_polls(DEFAULT_POLLS_FILE, [e.obs_date for e in elections]).polls
        self.assertEqual(sorted(o.key for o in live), sorted(o.key for o in committed))

    def test_dates_outside_the_frozen_segment_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            S.v02_value([], [], date(2026, 9, 18), S.frozen_v02_segment())

    def test_append_never_touches_earlier_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "captures.jsonl"
            S.append_capture({"a": 1}, path)
            first = path.read_bytes()
            S.append_capture({"b": 2}, path)
            self.assertTrue(path.read_bytes().startswith(first))
            self.assertEqual([json.loads(line) for line in path.read_text().splitlines()], [{"a": 1}, {"b": 2}])


def record(day: str, stamp: str, pop=None, v02=None) -> dict:
    def entry(values):
        return {"status": "missing"} if values is None else {"status": "ok", "values": dict(zip(PARTIES, values))}
    return {"stockholm_date": day, "captured_at_utc": stamp, "pop": entry(pop), "v02": entry(v02)}


def target_poll(key, house, start, end, published, shares=BASE, n=1000.0) -> Observation:
    mid = start + timedelta(days=(end - start).days // 2)
    return Observation(key, "poll", mid, published, tuple(shares), house, n, 0.1, fieldwork_start=start)


class ScoringRuleTests(unittest.TestCase):
    def test_latest_successful_capture_wins_and_nothing_carries_over(self) -> None:
        low, high = tuple(v - 1 for v in BASE), BASE
        captures = [
            record("2026-10-01", "2026-10-01T04:00:00+00:00", pop=low, v02=low),
            record("2026-10-01", "2026-10-01T16:00:00+00:00", pop=high, v02=None),  # v02 failed later
            record("2026-10-03", "2026-10-03T04:00:00+00:00", pop=None, v02=high),
        ]
        f = SC.forecasts_by_day(captures)
        np.testing.assert_array_equal(f["pop"][date(2026, 10, 1)], high)
        np.testing.assert_array_equal(f["v02"][date(2026, 10, 1)], low)
        self.assertNotIn(date(2026, 10, 3), f["pop"])
        self.assertNotIn(date(2026, 10, 2), f["v02"])

    def test_target_uses_only_later_fieldwork_in_the_window(self) -> None:
        d = date(2026, 10, 1)
        polls = [
            target_poll("a", "Sifo", d + timedelta(days=6), d + timedelta(days=10), d + timedelta(days=12)),
            target_poll("b", "Novus", d + timedelta(days=8), d + timedelta(days=12), d + timedelta(days=14)),
            target_poll("c", "Ipsos", d + timedelta(days=5), d + timedelta(days=13), d + timedelta(days=15)),
            target_poll("known", "Demoskop", d, d + timedelta(days=16), d + timedelta(days=18)),   # started on d
            target_poll("far", "Skop", d + timedelta(days=15), d + timedelta(days=20), d + timedelta(days=21)),
            target_poll("late", "YouGov", d + timedelta(days=8), d + timedelta(days=10), d + timedelta(days=29)),
        ]
        target = SC.target_for(d, polls)
        self.assertEqual((target.pollsters, target.polls), (3, 3))

    def test_repeated_polls_from_one_pollster_are_one_reading(self) -> None:
        d = date(2026, 10, 1)
        high = tuple(v + (4.0 if i == 0 else 0.0) for i, v in enumerate(BASE))
        polls = [target_poll(f"sifo{i}", "Sifo", d + timedelta(days=7 + i), d + timedelta(days=8 + i),
                             d + timedelta(days=10 + i), high) for i in range(5)]
        polls += [target_poll(h, h, d + timedelta(days=8), d + timedelta(days=10), d + timedelta(days=12))
                  for h in ("Novus", "Ipsos")]
        target = SC.target_for(d, polls)
        # Five Sifo polls of 1000 are one reading of weight 5000 among 7000.
        self.assertAlmostEqual(target.shares[0], BASE[0] + 4.0 * 5 / 7)
        self.assertAlmostEqual(target.equal_weight_shares[0], BASE[0] + 4.0 / 3)

    def test_fewer_than_three_pollsters_is_no_target(self) -> None:
        d = date(2026, 10, 1)
        polls = [target_poll(h, h, d + timedelta(days=8), d + timedelta(days=10), d + timedelta(days=12))
                 for h in ("Sifo", "Novus")]
        self.assertIsNone(SC.target_for(d, polls))

    def test_score_pairs_days_and_leaves_unscorable_days_out(self) -> None:
        d0 = date(2026, 10, 1)
        captures, polls = [], []
        for i in range(40):
            d = d0 + timedelta(days=i)
            captures.append(record(d.isoformat(), f"{d.isoformat()}T04:00:00+00:00",
                                   pop=tuple(v + 0.5 for v in BASE), v02=BASE))
        for i in range(60):
            for h in ("Sifo", "Novus", "Ipsos"):
                s = d0 + timedelta(days=i)
                polls.append(target_poll(f"{h}{i}", h, s, s + timedelta(days=2), s + timedelta(days=3)))
        result = SC.score(captures, polls, scored_through=d0 + timedelta(days=45))
        self.assertEqual(result["scorable_days"], 18)       # days 0..17 have passed their 28-day deadline
        self.assertEqual(result["paired_days"], 18)
        self.assertAlmostEqual(result["primary"]["mean_difference_v02_minus_pop"], -0.5)
        self.assertFalse(result["first_analysis_ready"])
        self.assertAlmostEqual(result["secondary"]["offset_adjusted_difference_v02_minus_pop"], 0.0)

    def test_bootstrap_interval_is_deterministic_and_brackets_the_mean(self) -> None:
        values = np.sin(np.arange(200) / 7.0) + 0.3
        a, b = SC.block_bootstrap_ci(values), SC.block_bootstrap_ci(values)
        self.assertEqual(a, b)
        self.assertLess(a[0], values.mean())
        self.assertGreater(a[1], values.mean())


if __name__ == "__main__":
    unittest.main()
