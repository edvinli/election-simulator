from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np

from scripts.poll_aggregate.build import build_aggregate
from scripts.simulator import model_inputs as MI
from tests.test_poll_aggregate import fixed_fit, synthetic_history
from scripts.poll_aggregate.config import SPEC_V02


def touch(root: Path, relative: Path) -> None:
    (root / relative).parent.mkdir(parents=True, exist_ok=True)
    (root / relative).write_text("x\n")


class OpinionInputResolutionTests(unittest.TestCase):
    def test_the_aggregate_wins_when_both_revisions_are_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in (MI.AGGREGATE_TIMESERIES, MI.AGGREGATE_POLLS, MI.LEGACY_TIMESERIES, MI.LEGACY_POLLS):
                touch(root, rel)
            inputs = MI.opinion_inputs(root)
            self.assertEqual((inputs.timeseries, inputs.polls), (root / MI.AGGREGATE_TIMESERIES, root / MI.AGGREGATE_POLLS))
            self.assertFalse(inputs.is_legacy)

    def test_a_pre_migration_revision_resolves_to_its_legacy_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            touch(root, MI.LEGACY_TIMESERIES)
            touch(root, MI.LEGACY_POLLS)
            self.assertTrue(MI.opinion_inputs(root).is_legacy)
            with self.assertRaises(MI.MissingOpinionInputs):
                MI.opinion_inputs(root, allow_legacy=False)

    def test_half_an_aggregate_never_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            touch(root, MI.AGGREGATE_TIMESERIES)
            with self.assertRaises(MI.MissingOpinionInputs):
                MI.opinion_inputs(root)

    def test_the_committed_checkout_uses_the_aggregate(self) -> None:
        processed = MI.Path(__file__).resolve().parents[1] / "data" / "processed"
        self.assertEqual(MI.opinion_inputs(processed, allow_legacy=False).source, MI.AGGREGATE_SOURCE)


class StoredFitTests(unittest.TestCase):
    def test_stored_fits_reproduce_the_fitted_series(self) -> None:
        polls, elections = synthetic_history()
        fitted = build_aggregate(polls, elections, fit=fixed_fit, spec=SPEC_V02)
        stored = {s.fit_date: (s.params, s.loglik, s.optimizer_iterations, s.optimizer_converged)
                  for s in fitted.segments}

        def refuse(*args, **kwargs):
            raise AssertionError("a stored segment must not be refitted")

        reused = build_aggregate(polls, elections, fit=refuse, spec=SPEC_V02, stored_fits=stored)
        for a, b in zip(fitted.rows, reused.rows):
            np.testing.assert_array_equal(a.mean, b.mean)


if __name__ == "__main__":
    unittest.main()


class OpinionSeriesSerializationTests(unittest.TestCase):
    """The chart's ``poll_of_polls`` field from the aggregate and from legacy PoP."""

    HEADER = "date,M,L,C,KD,S,V,MP,SD\n"

    def write(self, root: Path, name: str, rows: list[str]) -> Path:
        path = root / name
        path.write_text(self.HEADER + "".join(r + "\n" for r in rows), encoding="utf-8")
        return path

    def test_the_aggregate_carries_its_latest_row_to_the_end_of_the_range(self) -> None:
        from scripts.forecast_history.generate import serialize_poll_of_polls_timeseries
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(Path(tmp), MI.AGGREGATE_TIMESERIES.name, [
                "2026-09-27,18,4,7,6,28,8,6,19",
                "2026-09-29,19,4,7,6,28,8,6,18",
            ])
            records = serialize_poll_of_polls_timeseries(path, start_date="2026-09-30", end_date="2026-10-01")
        self.assertEqual([r["date"] for r in records], ["2026-09-30", "2026-10-01"])
        self.assertEqual(records[0]["parties"]["M"], 19.0)

    def test_a_legacy_series_is_read_strictly_within_the_range(self) -> None:
        from scripts.forecast_history.generate import serialize_poll_of_polls_timeseries
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(Path(tmp), "pollofpolls_timeseries.csv", ["2026-09-11,17,4,8,6,27,8,7,19"])
            self.assertEqual(
                serialize_poll_of_polls_timeseries(path, start_date="2026-09-30", end_date="2026-10-01"), [])
