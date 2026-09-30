from __future__ import annotations

import subprocess
import unittest
from datetime import date, timedelta
from unittest.mock import patch

import numpy as np

from diagnostics.poll_aggregate_level import lookahead as LA
from diagnostics.poll_aggregate_level import vintages as V
from scripts.poll_aggregate.build import build_aggregate
from tests.test_poll_aggregate import PARAMS, SHIFTED, fixed_fit, poll, synthetic_history


class LookaheadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.polls, self.elections = synthetic_history()
        # build_aggregate uses the latest known election as each segment's reference.
        self.segments = [
            {"fit_date": e.available, "reference_election": e.key, "params": PARAMS} for e in self.elections[1:]
        ]

    def test_zero_lookahead_reproduces_the_causal_series(self) -> None:
        causal = build_aggregate(self.polls, self.elections, fit=fixed_fit, end=date(2022, 2, 20))
        diagnostic = LA.build_lookahead(self.polls, self.elections, self.segments, 0, date(2022, 2, 20))
        self.assertEqual([r.day for r in causal.rows], [r.day for r in diagnostic.rows])
        for a, b in zip(causal.rows, diagnostic.rows):
            np.testing.assert_allclose(a.mean, b.mean, rtol=0, atol=1e-9)
            np.testing.assert_allclose(a.sd, b.sd, rtol=0, atol=1e-9)

    def test_a_later_poll_moves_the_smoothed_estimate_but_only_within_the_lag(self) -> None:
        target = date(2021, 6, 1)
        late = poll("late", target + timedelta(days=5), target + timedelta(days=6), SHIFTED, "Novus")

        def value(k: int, polls) -> np.ndarray:
            rows = LA.build_lookahead(polls, self.elections, self.segments, k, date(2021, 7, 1)).rows
            return next(r.mean for r in rows if r.day == target)

        without, with_late = self.polls, [*self.polls, late]
        np.testing.assert_allclose(value(3, without), value(3, with_late), rtol=0, atol=1e-12)
        self.assertFalse(np.allclose(value(7, without), value(7, with_late)))

    def test_filter_only_variant_ignores_fieldwork_after_the_date(self) -> None:
        target = date(2021, 6, 1)
        late = poll("late", target + timedelta(days=2), target + timedelta(days=3), SHIFTED, "Novus")
        rows = lambda polls: LA.build_lookahead(polls, self.elections, self.segments, 7, date(2021, 7, 1),
                                                smooth=False).rows
        a = next(r.mean for r in rows(self.polls) if r.day == target)
        b = next(r.mean for r in rows([*self.polls, late]) if r.day == target)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)

    def test_the_module_declares_itself_diagnostic(self) -> None:
        self.assertTrue(LA.DIAGNOSTIC_ONLY)


class VintageTests(unittest.TestCase):
    def test_a_snapshot_counts_from_its_stockholm_commit_date(self) -> None:
        log = "\n".join([
            "a" * 40 + " 2026-08-28T23:49:32+02:00",
            "b" * 40 + " 2026-08-30T22:30:00+00:00",  # 00:30 on 08-31 in Stockholm
        ])
        with patch.object(V, "_git", return_value=log):
            self.assertEqual(V.snapshot_known_on(date(2026, 8, 30)).commit, "a" * 40)
            self.assertEqual(V.snapshot_known_on(date(2026, 8, 31)).commit, "b" * 40)
            self.assertIsNone(V.snapshot_known_on(date(2026, 8, 27)))

    def test_value_on_or_before_never_reads_a_later_date(self) -> None:
        series = {date(2026, 8, 24): {"M": 1.0}, date(2026, 9, 4): {"M": 2.0}}
        self.assertEqual(V.value_on_or_before(series, date(2026, 8, 30)), (date(2026, 8, 24), {"M": 1.0}))
        self.assertIsNone(V.value_on_or_before(series, date(2026, 8, 1)))

    def test_committed_2026_case_snapshots(self) -> None:
        try:
            first = V.snapshot_known_on(date(2026, 8, 30))
            second = V.snapshot_known_on(date(2026, 9, 6))
        except subprocess.CalledProcessError:
            self.skipTest("origin/main history is not available in this checkout")
        self.assertEqual(first.commit, "34c52d68a0ebfb84e4b1d62d0eae3658476eda0a")
        self.assertEqual(second.commit, "bc581a32e2cb3bf28c33b0e6fde20b9e44e770c9")


if __name__ == "__main__":
    unittest.main()
