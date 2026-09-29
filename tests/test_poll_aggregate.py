from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from scripts.poll_aggregate.build import (
    TIMESERIES_FIELDS,
    _CheckpointedRunner,
    _known,
    build_aggregate,
    fit_hyperparameters,
    write_timeseries,
)
from scripts.poll_aggregate.config import HYPERPARAMETER_BOUNDS, HYPERPARAMETER_NAMES, PARTIES
from scripts.poll_aggregate.data import AggregateInputError, Observation, load_elections, load_polls
from scripts.poll_aggregate.model import Hyperparameters, SupportFilter
from scripts.pollofpolls.validate import TIMESERIES_FIELDS as POP_TIMESERIES_FIELDS

BASE = (19.0, 5.0, 7.0, 6.0, 28.0, 8.0, 6.0, 18.0)  # REST = 3.0
SHIFTED = (19.0, 5.0, 7.0, 6.0, 24.0, 8.0, 6.0, 22.0)  # four points from S to SD
PARAMS = Hyperparameters(
    process=1.3e-5, house_prior=5e-4, house_drift=1e-6, design_effect=1.0, election_n=1500.0
)
POLL_COLUMNS = (
    "poll_id", "pollster", "pollster_original", "house_original", "publication_period",
    "publication_date", "interview_start", "interview_end", "collection_period_approximate",
    "party", "support", "source_value", "support_status", "uncertain_share", "sample_size",
    "dataset_source_url", "row_source_references_json", "sources_index_url", "source_row",
    "retrieved_at",
)


def write_polls(path: Path, polls: list[dict]) -> None:
    """Long-format rows shaped like swedishpolls_individual_polls.csv."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POLL_COLUMNS)
        writer.writeheader()
        for poll in polls:
            shares = dict(zip(PARTIES, poll.get("shares", BASE)))
            for party in (*PARTIES, "FI", "other"):
                value = shares.get(party)
                writer.writerow(
                    {
                        "poll_id": poll["id"],
                        "pollster": poll.get("house", "Sifo"),
                        "publication_date": poll["published"],
                        "interview_start": poll["start"],
                        "interview_end": poll["end"],
                        "collection_period_approximate": "False",
                        "party": party,
                        "support": "" if value is None else value,
                        "sample_size": poll.get("n", 1000),
                    }
                )


def poll(key: str, obs: date, available: date, shares=BASE, house="Sifo", n=1000.0) -> Observation:
    return Observation(key, "poll", obs, available, tuple(shares), house, n, 0.1)


def election(day: date, shares=BASE, lag: int = 7) -> Observation:
    return Observation(f"election-{day}", "election", day, day + timedelta(days=lag), tuple(shares))


def fixed_fit(observations, houses, reference, initial=None):
    return PARAMS, 0.0, 0, True


def synthetic_history(seed: int = 7) -> tuple[list[Observation], list[Observation]]:
    """Three elections a year apart with weekly polls from two houses."""

    rng = np.random.default_rng(seed)
    elections = [election(date(2020, 1, 5)), election(date(2021, 1, 3)), election(date(2022, 1, 2))]
    polls = []
    day = date(2020, 1, 12)
    while day < date(2022, 3, 1):
        for house, bias in (("Sifo", 0.5), ("Novus", -0.5)):
            shares = np.array(BASE) + rng.normal(0.0, 0.6, len(BASE))
            shares[0] += bias
            polls.append(poll(f"{house}-{day}", day, day + timedelta(days=4), shares, house))
        day += timedelta(days=7)
    return polls, elections


class PollLoadingTests(unittest.TestCase):
    def load(self, polls: list[dict], elections=(date(2022, 9, 11),)):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "polls.csv"
            write_polls(path, polls)
            return load_polls(path, elections)

    def test_other_parties_are_the_remainder_not_rescaled(self) -> None:
        # Ipsos 2026-09-29: whole percentages that sum to 98.
        ipsos = (22, 4, 7, 6, 28, 8, 6, 17)
        dataset = self.load([{"id": "a", "published": "2026-09-29", "start": "2026-09-15",
                              "end": "2026-09-27", "shares": ipsos, "house": "Ipsos"}])
        (obs,) = dataset.polls
        self.assertEqual(obs.shares, tuple(float(v) for v in ipsos))
        self.assertEqual(obs.rounding_pp, 1.0)
        self.assertEqual(obs.obs_date, date(2026, 9, 21))
        self.assertEqual(obs.available, date(2026, 9, 29))

    def test_polls_whose_fieldwork_contains_election_day_are_excluded(self) -> None:
        dataset = self.load([
            {"id": "valu", "published": "2022-09-11", "start": "2022-08-26", "end": "2022-09-11"},
            {"id": "eve", "published": "2022-09-10", "start": "2022-09-05", "end": "2022-09-09"},
        ])
        self.assertEqual([o.key for o in dataset.polls], ["eve"])
        self.assertEqual(dataset.exclusions["election_day_poll"], ["valu"])

    def test_rolling_tracker_counts_only_new_fieldwork_days(self) -> None:
        dataset = self.load([
            {"id": f"d{i}", "published": f"2022-08-{10 + i}", "start": f"2022-08-{i + 1:02d}",
             "end": f"2022-08-{i + 3:02d}", "n": 3000}
            for i in range(3)
        ])
        self.assertEqual([o.sample_size for o in dataset.polls], [3000.0, 1000.0, 1000.0])

    def test_overlap_ignores_same_house_polls_published_later(self) -> None:
        dataset = self.load([
            {"id": "late", "published": "2022-08-20", "start": "2022-08-01", "end": "2022-08-03", "n": 3000},
            {"id": "early", "published": "2022-08-10", "start": "2022-08-02", "end": "2022-08-04", "n": 3000},
        ])
        by_key = {o.key: o.sample_size for o in dataset.polls}
        self.assertEqual(by_key["early"], 3000.0)
        self.assertEqual(by_key["late"], 1000.0)

    def test_missing_sample_size_is_imputed_from_earlier_polls_only(self) -> None:
        dataset = self.load([
            {"id": "first", "published": "2022-01-10", "start": "2022-01-01", "end": "2022-01-05", "n": 800},
            {"id": "gap", "published": "2022-02-10", "start": "2022-02-01", "end": "2022-02-05", "n": ""},
            {"id": "later", "published": "2022-03-10", "start": "2022-03-01", "end": "2022-03-05", "n": 5000},
        ])
        by_key = {o.key: o.sample_size for o in dataset.polls}
        self.assertEqual(by_key["gap"], 800.0)
        self.assertEqual(dataset.imputed_sample_size, ["gap"])

    def test_structural_exclusions_are_recorded_by_reason(self) -> None:
        dataset = self.load([
            {"id": "nodate", "published": "", "start": "2022-01-01", "end": "2022-01-05"},
            {"id": "future", "published": "2022-01-03", "start": "2022-01-01", "end": "2022-01-05"},
            {"id": "nosd", "published": "2022-01-10", "start": "2022-01-01", "end": "2022-01-05",
             "shares": BASE[:7] + (None,)},
            {"id": "old", "published": "2005-01-10", "start": "2005-01-01", "end": "2005-01-05"},
            {"id": "keep", "published": "2022-01-10", "start": "2022-01-02", "end": "2022-01-06"},
            {"id": "again", "published": "2022-01-11", "start": "2022-01-02", "end": "2022-01-06"},
        ])
        self.assertEqual([o.key for o in dataset.polls], ["keep"])
        self.assertEqual(
            {k: v for k, v in dataset.exclusions.items()},
            {
                "missing_dates": ["nodate"],
                "inconsistent_dates": ["future"],
                "party_not_reported": ["nosd"],
                "before_history_start": ["old"],
                "duplicate_house_period": ["again"],
            },
        )


class ElectionLoadingTests(unittest.TestCase):
    def write_inputs(self, tmp: Path, manifest_day: str = "2026-09-13") -> tuple[Path, Path]:
        results = tmp / "results.csv"
        with results.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["election_date", "party", "vote_share"])
            for day in ("2002-09-15", "2022-09-11"):
                for party, share in zip(PARTIES, BASE):
                    writer.writerow([day, party, share])
                writer.writerow([day, "OTHER", 3.0])
        manifest = tmp / "manifest.json"
        manifest.write_text(json.dumps({
            "certification_status": "FINAL_CERTIFIED",
            "election_date": manifest_day,
            "retrieved_at_utc": "2026-09-19T17:32:49Z",
            "parties": {p: {"vote_share_percentage_points": s} for p, s in zip(PARTIES, BASE)},
        }))
        return results, manifest

    def test_availability_is_explicit_per_election(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results, manifest = self.write_inputs(Path(tmp))
            elections = load_elections(results, [manifest])
        self.assertEqual(
            [(e.obs_date, e.available) for e in elections],
            [(date(2022, 9, 11), date(2022, 9, 18)), (date(2026, 9, 13), date(2026, 9, 19))],
        )
        self.assertTrue(all(e.house is None for e in elections))

    def test_an_election_supplied_twice_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results, manifest = self.write_inputs(Path(tmp), manifest_day="2022-09-11")
            with self.assertRaises(AggregateInputError):
                load_elections(results, [manifest])


class FilterTests(unittest.TestCase):
    def test_a_poll_pulls_support_toward_its_reading_net_of_nothing_learned(self) -> None:
        model = SupportFilter(["Sifo"], BASE, PARAMS)
        state = model.initial_state(date(2022, 1, 1), BASE)
        model.update(state, election(date(2022, 1, 1)))
        high = list(BASE)
        high[0] += 3.0
        model.update(state, poll("p", date(2022, 2, 1), date(2022, 2, 3), high))
        self.assertGreater(state.mean[0], BASE[0])
        self.assertLess(state.mean[0], BASE[0] + 3.0)
        # The unexplained part is split between support and Sifo's house effect.
        self.assertGreater(state.mean[len(PARTIES)], 0.0)
        self.assertTrue(np.isfinite(state.loglik))

    def test_unobserved_houses_do_not_change_the_estimate(self) -> None:
        polls, elections = synthetic_history()
        observations = _known([*polls, *elections], date(2021, 6, 1))
        results = []
        for houses in (["Novus", "Sifo"], ["Novus", "Sifo", "Skop", "YouGov"]):
            model = SupportFilter(houses, BASE, PARAMS)
            state = model.run(model.initial_state(elections[0].obs_date, BASE), observations)
            results.append(model.support_at(state, date(2021, 6, 1)))
        np.testing.assert_allclose(results[0][0], results[1][0], rtol=0, atol=1e-10)
        np.testing.assert_allclose(results[0][1], results[1][1], rtol=0, atol=1e-10)

    def test_checkpointed_runs_match_full_reruns(self) -> None:
        polls, elections = synthetic_history()
        everything = [*polls, *elections]
        # A late-published poll with old fieldwork forces a resume far back.
        everything.append(poll("late", date(2020, 3, 1), date(2021, 2, 1), house="Novus"))
        model = SupportFilter(["Novus", "Sifo"], BASE, PARAMS)
        runner = _CheckpointedRunner(model, elections[0].obs_date, BASE)
        for as_of in sorted({o.available for o in everything}):
            observations = _known(everything, as_of)
            resumed = runner.run(observations)
            fresh = model.run(model.initial_state(elections[0].obs_date, BASE), observations)
            np.testing.assert_allclose(resumed.mean, fresh.mean, rtol=0, atol=1e-9)
            np.testing.assert_allclose(resumed.cov, fresh.cov, rtol=0, atol=1e-9)


class BuildTests(unittest.TestCase):
    def test_later_information_never_changes_earlier_rows(self) -> None:
        polls, elections = synthetic_history()
        cutoff = date(2021, 8, 1)
        base = build_aggregate(polls, elections, fit=fixed_fit, end=date(2022, 2, 20))
        # Fieldwork before the cutoff, published after it.
        extra = [
            poll("revealed", date(2021, 5, 1), cutoff + timedelta(days=1), SHIFTED, "Skop"),
            poll("next", cutoff + timedelta(days=3), cutoff + timedelta(days=5), house="Novus"),
        ]
        extended = build_aggregate([*polls, *extra], elections, fit=fixed_fit, end=date(2022, 2, 20))
        before = [(r.day, r.mean, r.sd) for r in base.rows if r.day <= cutoff]
        after = [(r.day, r.mean, r.sd) for r in extended.rows if r.day <= cutoff]
        self.assertEqual(len(before), len(after))
        for (d1, m1, s1), (d2, m2, s2) in zip(before, after):
            self.assertEqual(d1, d2)
            np.testing.assert_allclose(m1, m2, rtol=0, atol=1e-10)
            np.testing.assert_allclose(s1, s2, rtol=0, atol=1e-10)
        changed = next(r for r in extended.rows if r.day == cutoff + timedelta(days=1))
        unchanged = next(r for r in base.rows if r.day == cutoff + timedelta(days=1))
        self.assertFalse(np.allclose(changed.mean, unchanged.mean))

    def test_an_election_counts_only_from_its_availability_date(self) -> None:
        polls, elections = synthetic_history()
        moved = [*elections[:2], election(elections[2].obs_date, SHIFTED)]
        a = build_aggregate(polls, elections, fit=fixed_fit, end=date(2022, 2, 1))
        b = build_aggregate(polls, moved, fit=fixed_fit, end=date(2022, 2, 1))
        available = elections[2].available
        for ra, rb in zip(a.rows, b.rows):
            if ra.day < available:
                np.testing.assert_allclose(ra.mean, rb.mean, rtol=0, atol=1e-10)
            else:
                self.assertFalse(np.allclose(ra.mean, rb.mean), ra.day)

    def test_rows_start_at_the_second_result_and_use_that_segments_fit(self) -> None:
        polls, elections = synthetic_history()
        seen = []

        def recording_fit(observations, houses, reference, initial=None):
            seen.append(max(o.available for o in observations))
            return fixed_fit(observations, houses, reference, initial)

        result = build_aggregate(polls, elections, fit=recording_fit)
        self.assertEqual(result.rows[0].day, elections[1].available)
        # Each fit sees nothing published after its own date.
        self.assertTrue(all(s <= f for s, f in zip(seen, [e.available for e in elections[1:]])))
        self.assertEqual([s.fit_date for s in result.segments], [e.available for e in elections[1:]])
        days = [r.day for r in result.rows]
        self.assertEqual(days, [days[0] + timedelta(days=i) for i in range(len(days))])

    def test_fit_stays_within_bounds_and_improves_on_the_start(self) -> None:
        polls, elections = synthetic_history()
        observations = _known([*polls, *elections], date(2021, 1, 10))
        params, loglik, _, _ = fit_hyperparameters(
            observations, ["Novus", "Sifo"], BASE, max_iterations=60
        )
        for name in HYPERPARAMETER_NAMES:
            lo, hi = HYPERPARAMETER_BOUNDS[name]
            self.assertTrue(lo <= getattr(params, name) <= hi, name)
        model = SupportFilter(["Novus", "Sifo"], BASE, Hyperparameters.from_log(
            [np.log(v) for v in (2e-6, 5e-4, 1e-8, 1.5, 5000.0)]))
        start = model.run(model.initial_state(elections[0].obs_date, BASE), observations)
        self.assertGreaterEqual(loglik, start.loglik)

    def test_output_uses_the_pollofpolls_timeseries_columns(self) -> None:
        polls, elections = synthetic_history()
        result = build_aggregate(polls, elections, fit=fixed_fit, end=date(2021, 2, 1))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "series.csv"
            write_timeseries(result, path)
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fields = reader.fieldnames
        self.assertEqual(tuple(fields), TIMESERIES_FIELDS)
        self.assertEqual(TIMESERIES_FIELDS, POP_TIMESERIES_FIELDS)
        first = rows[0]
        named = sum(float(first[p]) for p in PARTIES)
        self.assertAlmostEqual(named + float(first["other"]), 100.0, delta=0.011)
        self.assertEqual(first["FI"], "")
        extra = json.loads(first["source_extra_json"])
        self.assertEqual(set(extra["sd"]), set(PARTIES))


if __name__ == "__main__":
    unittest.main()
