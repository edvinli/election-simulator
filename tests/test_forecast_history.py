"""Tests for the offline historical coalition forecast generator."""

from __future__ import annotations

from copy import deepcopy
import csv
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from scripts.forecast_history.contract import (
    DEFAULT_COALITIONS,
    build_groups_from_matrices,
    coalition_seat_draws,
    coalition_vote_draws,
)
from scripts.forecast_history.generate import (
    backfill_reconstructed_curve,
    build_history,
    build_history_dates,
    filter_swedishpolls_as_of,
    filter_swedishpolls_period,
    first_changed_poll_date,
    missing_curve_dates,
    serialize_poll_of_polls_timeseries,
    serialize_swedishpolls,
    _poll_identity,
)
from scripts.simulator.config import PARLIAMENTARY_PARTIES_8


class ForecastHistoryTests(unittest.TestCase):
    """Unit tests for history assembly, isolated from the repository's archive.

    Every ``build_history`` call here passes ``archive_dir=None``. The default
    is ``data/processed/prospective_forecasts``, and ``build_history`` folds
    every date it finds there into ``observation_dates``. These tests pin
    explicit ``dates`` and a ``latest_result`` for 2026-05-24, so once the
    publication automation archived anything later, ``latest_result`` no longer
    matched ``max(observation_dates)`` and the calls raised. The tests were
    reading production state they never meant to depend on, and they broke a
    little further with every publish.

    Archive substitution itself is still covered: the cases that exercise it
    inject ``archived_points`` directly, which is the behaviour under test
    rather than whatever happens to be on disk.
    """

    @staticmethod
    def _matrices() -> tuple[np.ndarray, np.ndarray]:
        # The first eight columns are the parliamentary parties in canonical
        # order.  The ninth vote column is REST and must not enter the poll or
        # coalition denominator.
        votes = np.array(
            [
                [20, 5, 10, 5, 30, 10, 8, 12, 0],
                [18, 6, 11, 4, 32, 9, 7, 13, 0],
                [22, 4, 9, 6, 28, 11, 9, 11, 0],
                [19, 7, 12, 5, 31, 8, 6, 12, 5],
            ],
            dtype=float,
        )
        seats = np.array(
            [
                [40, 20, 30, 20, 80, 50, 30, 79],
                [39, 21, 31, 19, 82, 48, 31, 78],
                [42, 18, 29, 22, 78, 52, 29, 79],
                [41, 19, 32, 21, 81, 49, 28, 78],
            ],
            dtype=np.int64,
        )
        return votes, seats

    @staticmethod
    def _poll_csv(path: Path) -> None:
        fields = [
            "poll_id",
            "pollster",
            "publication_date",
            "interview_start",
            "interview_end",
            "party",
            "support",
            "support_status",
            "sample_size",
        ]
        values = {
            "M": 20,
            "L": 5,
            "C": 10,
            "KD": 5,
            "S": 30,
            "V": 10,
            "MP": 8,
            "SD": 12,
        }
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for party, support in values.items():
                writer.writerow(
                    {
                        "poll_id": "poll-1",
                        "pollster": "Testinstitut",
                        "publication_date": "2026-05-25",
                        "interview_start": "2026-05-20",
                        "interview_end": "2026-05-24",
                        "party": party,
                        "support": support,
                        "support_status": "reported",
                        "sample_size": 1000,
                    }
                )
            # Neither Uncertain nor a made-up Other value may become a party.
            writer.writerow(
                {
                    "poll_id": "poll-1",
                    "pollster": "Testinstitut",
                    "publication_date": "2026-05-25",
                    "interview_start": "2026-05-20",
                    "interview_end": "2026-05-24",
                    "party": "Uncertain",
                    "support": 99,
                    "support_status": "uncertain",
                    "sample_size": 1000,
                }
            )
            writer.writerow(
                {
                    "poll_id": "poll-1",
                    "pollster": "Testinstitut",
                    "publication_date": "2026-05-25",
                    "interview_start": "2026-05-20",
                    "interview_end": "2026-05-24",
                    "party": "Other",
                    "support": 3,
                    "support_status": "reported",
                    "sample_size": 1000,
                }
            )

    def test_vote_denominator_is_exactly_the_eight_parliamentary_parties(self) -> None:
        votes, _ = self._matrices()
        red_green = coalition_vote_draws(votes, DEFAULT_COALITIONS["red_green_center"])
        # The fourth draw has 5% REST.  It must still be 58% (not 55.1%)
        # because REST is excluded from the denominator.
        np.testing.assert_allclose(red_green, [58.0, 59.0, 57.0, 57.0])

    def test_seat_combination_uses_each_original_joint_draw(self) -> None:
        _, seats = self._matrices()
        draws = coalition_seat_draws(seats, DEFAULT_COALITIONS["red_green_center"])
        np.testing.assert_array_equal(draws, [190, 192, 188, 190])
        # This is intentionally a draw-level assertion, rather than a sum of
        # party medians or separate marginal distributions.
        self.assertEqual(int(np.median(draws)), 190)

    def test_every_exported_quantile_matches_draw_level_source(self) -> None:
        votes, seats = self._matrices()
        groups = build_groups_from_matrices(votes, seats)
        for coalition, members in DEFAULT_COALITIONS.items():
            vote_draws = coalition_vote_draws(votes, members)
            seat_draws = coalition_seat_draws(seats, members)
            for field, quantile in (("p05", 0.05), ("p25", 0.25), ("p50", 0.50), ("p75", 0.75), ("p95", 0.95)):
                self.assertEqual(
                    groups[coalition]["vote"][field],
                    round(float(np.quantile(vote_draws, quantile)), 6),
                )
                self.assertEqual(
                    groups[coalition]["seats"][field],
                    int(np.quantile(seat_draws, quantile)),
                )

    def test_history_schedule_marks_cap_boundary(self) -> None:
        dates = build_history_dates(latest_date="2026-05-26")
        self.assertIn(date(2026, 5, 23), dates)
        self.assertIn(date(2026, 5, 24), dates)
        self.assertIn(date(2026, 5, 25), dates)
        self.assertNotIn(date(2026, 5, 22), dates)
        self.assertEqual(dates[0], date(2022, 9, 18))

    def test_serialized_polls_ignore_uncertain_and_other(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "polls.csv"
            self._poll_csv(path)
            polls = serialize_swedishpolls(path)
        self.assertEqual(len(polls), 1)
        self.assertEqual(list(polls[0]["parties"]), list(PARLIAMENTARY_PARTIES_8))
        self.assertNotIn("Uncertain", polls[0]["parties"])
        self.assertNotIn("Other", polls[0]["parties"])

    def test_historical_poll_filter_excludes_future_publication_and_interview_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "polls.csv"
            self._poll_csv(path)
            polls = serialize_swedishpolls(path)
        self.assertEqual(filter_swedishpolls_as_of(polls, "2026-05-23"), [])
        admissible = filter_swedishpolls_as_of(polls, "2026-05-25")
        self.assertEqual(len(admissible), 1)
        self.assertEqual(admissible[0]["publication_date"], "2026-05-25")

    def test_chart_poll_layer_is_bounded_to_history_period(self) -> None:
        base = {
            "poll_id": "p",
            "company": "Test",
            "fieldwork_start": None,
            "fieldwork_end": None,
            "n": 1000,
            "parties": {party: 1.0 for party in PARLIAMENTARY_PARTIES_8},
        }
        before = {**base, "poll_id": "before", "publication_date": "2022-09-10"}
        inside = {**base, "poll_id": "inside", "publication_date": "2022-09-11"}
        after = {**base, "poll_id": "after", "publication_date": "2026-08-27"}
        bounded = filter_swedishpolls_period(
            [before, inside, after], "2022-09-11", "2026-08-24"
        )
        self.assertEqual([poll["poll_id"] for poll in bounded], ["inside"])

    def test_build_history_drops_polls_before_chart_start(self) -> None:
        votes, seats = self._matrices()

        def runner(*, as_of: str, election_date: str, samples: int, seed: int):
            return SimpleNamespace(
                vote_shares_matrix=votes,
                seats_matrix=seats,
                manifest={"source_git_commit": "8" * 40},
            )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "polls.csv"
            self._poll_csv(path)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
                fields = list(rows[0])
            before = [dict(row, poll_id="before", publication_date="2022-09-10") for row in rows]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows([*rows, *before])
            payload = build_history(
                archive_dir=None,
                dates=["2022-09-11"],
                poll_file=path,
                samples=4,
                production_latest_samples=4,
                simulation_runner=runner,
                model_commit="8" * 40,
                source_worktree_clean=True,
            )
        self.assertEqual({poll["poll_id"] for poll in payload["polls"]}, {"poll-1"})

    def test_history_is_deterministic_and_runner_receives_strict_as_of_dates(self) -> None:
        votes, seats = self._matrices()
        calls: list[tuple[str, int]] = []

        def runner(*, as_of: str, election_date: str, samples: int, seed: int):
            calls.append((as_of, samples))
            return SimpleNamespace(
                vote_shares_matrix=votes,
                seats_matrix=seats,
                manifest={"source_git_commit": "a" * 40},
            )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "polls.csv"
            self._poll_csv(path)
            kwargs = {
                "dates": ["2026-05-23", "2026-05-24"],
                "samples": 4,
                "seed": 7,
                "poll_file": path,
                "model_commit": "a" * 40,
                "simulation_runner": runner,
                "source_worktree_clean": True,
                "production_latest_samples": 4,
                "archive_dir": None,
            }
            first = build_history(**kwargs)
            second = build_history(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(calls, [("2026-05-23", 4), ("2026-05-24", 4), ("2026-05-23", 4), ("2026-05-24", 4)])
        self.assertEqual([point["samples"] for point in first["series"]], [4, 4])
        self.assertEqual(first["series"][0]["horizon_days"], 113)
        self.assertEqual(first["series"][0]["dynamics_horizon_days"], 112)
        self.assertEqual(first["series"][1]["horizon_days"], 112)
        self.assertEqual(first["series"][1]["dynamics_horizon_days"], 112)

    def test_rich_archived_point_replaces_reconstruction_but_legacy_snapshot_does_not(self) -> None:
        votes, seats = self._matrices()
        groups = build_groups_from_matrices(votes, seats)
        calls: list[tuple[str, int]] = []

        def runner(*, as_of: str, election_date: str, samples: int, seed: int):
            calls.append((as_of, samples))
            return SimpleNamespace(
                vote_shares_matrix=votes,
                seats_matrix=seats,
                manifest={"source_git_commit": "b" * 40},
            )

        archived = {
            "as_of": "2026-05-24",
            "samples": 4,
            "source_git_commit": "c" * 40,
            "groups": groups,
        }
        legacy = {
            "as_of": "2026-05-23",
            "samples": 100_000,
            "group_probabilities": {"red_green_center": {"mean_seats": 170}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "polls.csv"
            self._poll_csv(path)
            payload = build_history(
                archive_dir=None,
                dates=["2026-05-23", "2026-05-24"],
                poll_file=path,
                samples=4,
                archived_points=[archived, legacy],
                simulation_runner=runner,
                model_commit="b" * 40,
                source_worktree_clean=True,
                production_latest_samples=4,
            )
        self.assertEqual(calls, [("2026-05-23", 4)])
        by_date = {point["date"]: point for point in payload["series"]}
        self.assertEqual(by_date["2026-05-24"]["provenance"], "prospective_archived")
        self.assertEqual(by_date["2026-05-23"]["provenance"], "reconstructed_current_model")

    def test_archived_point_does_not_displace_the_reconstructed_curve(self) -> None:
        """An archived point is a second observation, not a substitute.

        The chart draws one continuous line through the non-archived points, so
        an archived point that took the reconstructed point's place left a hole
        in the curve on its date -- one per publication, widening forever.
        """

        votes, seats = self._matrices()
        groups = build_groups_from_matrices(votes, seats)
        calls: list[tuple[str, int]] = []

        def runner(*, as_of: str, election_date: str, samples: int, seed: int):
            calls.append((as_of, samples))
            return SimpleNamespace(
                vote_shares_matrix=votes,
                seats_matrix=seats,
                manifest={"source_git_commit": "b" * 40},
            )

        # 2026-05-25 is not the latest requested date, so the archived record
        # there is a genuine historical publication sitting behind the curve.
        archived = {
            "as_of": "2026-05-25",
            "samples": 4,
            "source_git_commit": "c" * 40,
            "groups": groups,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "polls.csv"
            self._poll_csv(path)
            payload = build_history(
                archive_dir=None,
                dates=["2026-05-24", "2026-05-25", "2026-05-26"],
                poll_file=path,
                samples=4,
                archived_points=[archived],
                simulation_runner=runner,
                model_commit="b" * 40,
                source_worktree_clean=True,
                production_latest_samples=4,
            )
        # The archived date is simulated like any other: it needs a curve point.
        self.assertIn(("2026-05-25", 4), calls)
        provenances = {
            (point["date"], point["provenance"]) for point in payload["series"]
        }
        self.assertIn(("2026-05-25", "reconstructed_current_model"), provenances)
        self.assertIn(("2026-05-25", "prospective_archived"), provenances)
        curve = [
            point["date"]
            for point in payload["series"]
            if point["provenance"] != "prospective_archived"
        ]
        self.assertEqual(curve, ["2026-05-24", "2026-05-25", "2026-05-26"])

    def test_missing_curve_dates_finds_archive_only_days(self) -> None:
        votes, seats = self._matrices()
        groups = build_groups_from_matrices(votes, seats)

        def point(day: str, provenance: str) -> dict:
            return {
                "date": day,
                "samples": 4,
                "horizon_days": (date(2026, 9, 13) - date.fromisoformat(day)).days,
                "dynamics_horizon_days": 0,
                "provenance": provenance,
                "groups": groups,
            }

        payload = {
            "series": [
                point("2026-05-24", "reconstructed_current_model"),
                point("2026-05-25", "prospective_archived"),
                point("2026-05-26", "prospective_archived"),
                point("2026-05-27", "current_production"),
            ]
        }
        self.assertEqual(
            missing_curve_dates(payload),
            [date(2026, 5, 25), date(2026, 5, 26)],
        )
        # A curve with no archive-only day needs nothing.
        continuous = {
            "series": [
                point("2026-05-24", "reconstructed_current_model"),
                point("2026-05-25", "reconstructed_current_model"),
                point("2026-05-25", "prospective_archived"),
                point("2026-05-26", "current_production"),
            ]
        }
        self.assertEqual(missing_curve_dates(continuous), [])

    def test_backfill_closes_the_hole_the_roll_in_leaves(self) -> None:
        votes, seats = self._matrices()
        calls: list[str] = []

        def runner(*, as_of: str, election_date: str, samples: int, seed: int):
            calls.append(as_of)
            return SimpleNamespace(
                vote_shares_matrix=votes,
                seats_matrix=seats,
                manifest={"source_git_commit": "b" * 40},
            )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "polls.csv"
            self._poll_csv(path)
            published = build_history(
                archive_dir=None,
                dates=["2026-05-24", "2026-05-25", "2026-05-26"],
                poll_file=path,
                samples=4,
                simulation_runner=runner,
                model_commit="b" * 40,
                source_worktree_clean=True,
                production_latest_samples=4,
            )
            # Simulate what the daily roll-in does to the previous official
            # point: relabel it, leaving its date with no curve point.
            holed = deepcopy(published)
            for point in holed["series"]:
                if point["date"] == "2026-05-25":
                    point["provenance"] = "prospective_archived"
            holed.pop("deterministic_content_sha256", None)
            self.assertEqual(missing_curve_dates(holed), [date(2026, 5, 25)])

            calls.clear()
            filled, backfilled = backfill_reconstructed_curve(
                holed,
                poll_file=path,
                archive_dir=None,
                samples=4,
                simulation_runner=runner,
            )
        self.assertEqual(backfilled, [date(2026, 5, 25)])
        self.assertEqual(calls, ["2026-05-25"])
        self.assertEqual(missing_curve_dates(filled), [])
        provenances = {
            (point["date"], point["provenance"]) for point in filled["series"]
        }
        self.assertIn(("2026-05-25", "reconstructed_current_model"), provenances)
        self.assertIn(("2026-05-25", "prospective_archived"), provenances)

        # Calling it on a continuous curve runs no simulation at all, which is
        # what makes it safe on every publication.
        calls.clear()
        unchanged, nothing = backfill_reconstructed_curve(
            filled, simulation_runner=runner
        )
        self.assertEqual(nothing, [])
        self.assertEqual(calls, [])
        self.assertEqual(unchanged["series"], filled["series"])

    # --- reconstruction reuse across an upstream refresh -------------------
    #
    # A poll_id is a SHA-256 over an identity dict that includes the poll's row
    # number in the upstream CSV, and SwedishPolls publishes newest-first. One
    # appended poll therefore renumbers every older row and changes every id in
    # the file. Reuse must survive that: the observations are the same, so every
    # point whose visible polls are unchanged has to be carried over.
    #
    # Pairing by poll_id made a single append look like the whole history had
    # been replaced -- no poll matched, every in-window date registered as
    # changed, the window start came back, and the entire curve was
    # re-simulated. These tests pin the observable contract rather than the
    # mechanism, so they hold whatever the id scheme does next.

    @staticmethod
    def _write_polls(path: Path, polls, *, id_prefix: str, alias: str = "Testinstitut") -> None:
        """Write a SwedishPolls-shaped CSV, controlling the synthetic ids.

        ``id_prefix`` stands in for upstream row renumbering: the same
        observations written with a different prefix are the same polls with
        different ``poll_id`` values, which is exactly what a refresh produces.
        """

        fields = [
            "poll_id", "pollster", "pollster_original", "publication_date",
            "interview_start", "interview_end", "party", "support",
            "support_status", "sample_size",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, (published, values) in enumerate(polls):
                for party, support in values.items():
                    writer.writerow({
                        "poll_id": f"{id_prefix}-{index}",
                        "pollster": "Testinstitut",
                        # The upstream display alias. Varying it is the
                        # negative control below; it is never the canonical
                        # name the model keys on.
                        "pollster_original": alias,
                        "publication_date": published,
                        "interview_start": published,
                        "interview_end": published,
                        "party": party,
                        "support": support,
                        "support_status": "reported",
                        "sample_size": 1000,
                    })

    # Three curve dates, and a poll on each of the first two. The last date is
    # the official current point: build_history exempts a current_production
    # point from poll-driven invalidation by design, since it is the published
    # result rather than something re-derived. So the middle date is the one
    # that has to demonstrate invalidation, and it needs a poll of its own.
    CURVE_DATES = ("2026-05-23", "2026-05-24", "2026-05-25")
    ORIGINAL_POLLS = (
        ("2026-05-23", {"M": 20, "L": 5, "C": 10, "KD": 5, "S": 30, "V": 10, "MP": 8, "SD": 12}),
        ("2026-05-24", {"M": 21, "L": 5, "C": 10, "KD": 5, "S": 29, "V": 10, "MP": 8, "SD": 12}),
    )

    @staticmethod
    def _support(**overrides):
        values = {"M": 20, "L": 5, "C": 10, "KD": 5, "S": 30, "V": 10, "MP": 8, "SD": 12}
        values.update(overrides)
        return values

    def _build_then_refresh(self, refreshed_polls, *, id_prefix: str):
        """Build a two-point curve, then rebuild against a refreshed source.

        Returns the first payload, the rebuilt payload, and the dates the
        simulation runner was asked for during the rebuild.
        """

        votes, seats = self._matrices()
        # The payload records only polls inside the chart period, which starts
        # at the first curve date -- so the fixture's polls sit on the curve
        # dates themselves, as the real source's do.
        original = list(self.ORIGINAL_POLLS)
        dates = list(self.CURVE_DATES)
        rebuild_calls: list[str] = []

        def runner(*, as_of: str, election_date: str, samples: int, seed: int):
            rebuild_calls.append(as_of)
            return SimpleNamespace(
                vote_shares_matrix=votes,
                seats_matrix=seats,
                manifest={"source_git_commit": "d" * 40},
            )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "polls.csv"
            self._write_polls(path, original, id_prefix="orig")
            first = build_history(
                archive_dir=None,
                dates=dates,
                poll_file=path,
                samples=4,
                simulation_runner=runner,
                model_commit="d" * 40,
                source_worktree_clean=True,
                production_latest_samples=4,
            )
            self.assertEqual(rebuild_calls, dates)
            rebuild_calls.clear()

            self._write_polls(path, refreshed_polls, id_prefix=id_prefix)
            second = build_history(
                archive_dir=None,
                dates=dates,
                poll_file=path,
                samples=4,
                existing_payload=first,
                simulation_runner=runner,
                model_commit="d" * 40,
                source_worktree_clean=True,
                production_latest_samples=4,
            )
        return first, second, rebuild_calls

    def test_appended_polls_and_renumbered_ids_reuse_every_earlier_point(self) -> None:
        """The regression: a later poll plus id churn must recompute nothing."""

        # A poll published strictly after both curve dates, and every id
        # rewritten, exactly as an upstream prepend does.
        refreshed = [*self.ORIGINAL_POLLS, ("2026-05-30", self._support(M=22))]
        first, second, calls = self._build_then_refresh(refreshed, id_prefix="renumbered")

        self.assertEqual(
            calls, [],
            "an append after the curve, with renumbered ids, recomputed a point",
        )
        # Byte for byte, not merely equal-looking.
        self.assertEqual(
            json.dumps(second["series"], sort_keys=True),
            json.dumps(first["series"], sort_keys=True),
        )
        self.assertEqual(
            second["resume_diagnostics"]["existing_points_reused"], len(self.CURVE_DATES)
        )
        self.assertEqual(second["resume_diagnostics"]["new_points_generated"], 0)
        # The source hash did change -- reuse survived that, which is the point.
        self.assertNotEqual(second["poll_source_sha256"], first["poll_source_sha256"])

    def test_renumbering_alone_reuses_every_point(self) -> None:
        """Identical observations under new ids are not a change at all."""

        first, second, calls = self._build_then_refresh(
            list(self.ORIGINAL_POLLS), id_prefix="renumbered"
        )
        self.assertEqual(calls, [])
        self.assertEqual(
            json.dumps(second["series"], sort_keys=True),
            json.dumps(first["series"], sort_keys=True),
        )

    def test_a_revision_at_or_before_a_point_recomputes_it(self) -> None:
        """The other half of the contract: a real change must invalidate.

        Revising the poll published on the first curve date invalidates that
        date onward -- even though the ids are renumbered at the same time, so
        a reuse rule that went by id could not tell this from the harmless
        case above.
        """

        revised = [("2026-05-23", self._support(M=25)), self.ORIGINAL_POLLS[1]]
        _first, _second, calls = self._build_then_refresh(revised, id_prefix="renumbered")
        self.assertEqual(calls, ["2026-05-23", "2026-05-24"])

    def test_a_revision_invalidates_only_from_its_own_date(self) -> None:
        """Reuse is incremental, not all-or-nothing.

        Revising the later poll leaves the earlier point untouched, which is
        the property that makes a refresh cheap.
        """

        revised = [self.ORIGINAL_POLLS[0], ("2026-05-24", self._support(M=25))]
        first, second, calls = self._build_then_refresh(revised, id_prefix="renumbered")
        self.assertEqual(calls, ["2026-05-24"])
        self.assertNotIn("2026-05-23", calls)
        kept = next(p for p in second["series"] if p["date"] == "2026-05-23")
        original_kept = next(p for p in first["series"] if p["date"] == "2026-05-23")
        self.assertEqual(
            json.dumps(kept, sort_keys=True), json.dumps(original_kept, sort_keys=True)
        )

    def test_a_removed_poll_recomputes_from_its_date(self) -> None:
        remaining = [self.ORIGINAL_POLLS[1]]
        _first, _second, calls = self._build_then_refresh(remaining, id_prefix="renumbered")
        self.assertEqual(calls, ["2026-05-23", "2026-05-24"])

    def test_the_identity_covers_every_serialized_field_but_the_identifier(self) -> None:
        """Completeness, pinned so a new field cannot slip past the identity.

        The reuse comparison is only sound if the identity carries everything
        about a poll that can reach the model.  ``serialize_swedishpolls`` is
        the whole of what reaches it -- the columns it drops (``source_row``,
        ``retrieved_at``, ``support_status``, ``uncertain_share``, the source
        URLs) never appear on the forecast path -- so the identity has to cover
        every serialized field except the one deliberately excluded.

        ``poll_id`` is that exclusion, and it is the point of the exercise: it
        is a SHA-256 over an identity dict containing the poll's upstream row
        number, so it changes for every poll whenever one is appended.  Its
        *cardinality* still matters, because the model counts distinct polls,
        and a multiset comparison preserves that.

        If a field is ever added to ``_new_poll``, this fails and whoever adds
        it has to decide whether it belongs in the identity rather than
        silently widening what reuse ignores.
        """

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "polls.csv"
            self._write_polls(path, list(self.ORIGINAL_POLLS), id_prefix="orig")
            polls = serialize_swedishpolls(path)
        serialized_fields = set(polls[0])
        self.assertEqual(
            serialized_fields,
            {
                "poll_id", "company", "house", "publication_date",
                "fieldwork_start", "fieldwork_end", "n", "parties",
            },
        )

        # Vary one field at a time and require the identity to notice. The
        # identity is opaque by design, so this asks the only question that
        # matters: does changing this field change the identity?
        baseline = dict(polls[0])
        for field, altered in (
            ("company", "Someone Else"),
            ("house", "Another House"),
            ("publication_date", date(2026, 5, 24)),
            ("fieldwork_start", date(2026, 5, 21)),
            ("fieldwork_end", date(2026, 5, 22)),
            ("n", 2000),
            ("parties", {**baseline["parties"], "M": 99.0}),
        ):
            with self.subTest(field=field):
                self.assertNotEqual(
                    _poll_identity({**baseline, field: altered}),
                    _poll_identity(baseline),
                    f"{field} can reach the model but does not affect the reuse identity",
                )
        # ...and the identifier alone must not.
        self.assertEqual(
            _poll_identity({**baseline, "poll_id": "swp-completely-different"}),
            _poll_identity(baseline),
        )
        self.assertEqual(
            serialized_fields - {"poll_id"},
            {"company", "house", "publication_date", "fieldwork_start",
             "fieldwork_end", "n", "parties"},
            "a serialized field is neither in the identity nor deliberately excluded",
        )

    def test_the_display_alias_changes_no_model_input_and_no_boundary(self) -> None:
        """Negative control for the one path-adjacent field the identity omits.

        ``pollster_original`` is the upstream display name -- Synovate and TEMO
        are both Ipsos, Gallup is Novus -- and it is not in the reuse identity.
        It is worth encoding rather than arguing, because it *does* appear on
        the consensus surface: it is part of the pivot index in
        ``scripts.election_residuals.consensus``.

        So vary only the alias, holding the canonical pollster and every
        numeric field fixed, and require three things to be unchanged: what
        the history path serializes, what the consensus computes, and the
        reuse boundary. The test also asserts the alias really does reach the
        consensus record, so it cannot pass by the field being ignored
        everywhere -- which would make the control vacuous.
        """

        import pandas as pd

        from scripts.election_residuals.consensus import build_election_polling_consensus

        # --- what the history path serializes ---
        with tempfile.TemporaryDirectory() as temporary:
            plain = Path(temporary) / "plain.csv"
            renamed = Path(temporary) / "renamed.csv"
            self._write_polls(plain, list(self.ORIGINAL_POLLS), id_prefix="orig")
            self._write_polls(
                renamed, list(self.ORIGINAL_POLLS), id_prefix="orig", alias="Synovate Temo"
            )
            self.assertNotEqual(
                plain.read_text(encoding="utf-8"), renamed.read_text(encoding="utf-8"),
                "the fixture did not actually vary the alias",
            )
            before = serialize_swedishpolls(plain)
            after = serialize_swedishpolls(renamed)

        self.assertEqual(before, after, "the alias leaked into the serialized poll")
        # ...so it cannot move the boundary either.
        self.assertIsNone(first_changed_poll_date(before, after))

        # --- what the consensus computes ---
        def frame(alias: str) -> pd.DataFrame:
            rows = []
            for index, (published, values) in enumerate(self.ORIGINAL_POLLS):
                for party, support in values.items():
                    rows.append({
                        "poll_id": f"orig-{index}",
                        "pollster": "Testinstitut",
                        "pollster_original": alias,
                        "publication_date": published,
                        "interview_start": published,
                        "interview_end": published,
                        "party": party,
                        "support": float(support),
                        "sample_size": 1000,
                    })
            return pd.DataFrame(rows)

        election = date(2026, 6, 1)
        plain_consensus = build_election_polling_consensus(election, frame("Testinstitut"))
        renamed_consensus = build_election_polling_consensus(election, frame("Synovate Temo"))

        self.assertEqual(
            plain_consensus.consensus_composition,
            renamed_consensus.consensus_composition,
            "the display alias changed the consensus composition",
        )
        self.assertEqual(
            (
                plain_consensus.total_eligible_polls_in_window,
                plain_consensus.retained_pollsters_count,
            ),
            (
                renamed_consensus.total_eligible_polls_in_window,
                renamed_consensus.retained_pollsters_count,
            ),
        )
        # The control is only meaningful if the alias reaches the record at
        # all. It does -- which is exactly why it is worth pinning that it
        # changes nothing numeric.
        self.assertEqual(
            [poll.pollster_original for poll in renamed_consensus.contributing_polls],
            ["Synovate Temo"] * len(renamed_consensus.contributing_polls),
        )
        self.assertNotEqual(
            [poll.pollster_original for poll in plain_consensus.contributing_polls],
            [poll.pollster_original for poll in renamed_consensus.contributing_polls],
        )
        # And the canonical name, which every numeric step keys on, is the one
        # that stayed fixed.
        self.assertEqual(
            {poll.pollster for poll in renamed_consensus.contributing_polls},
            {"Testinstitut"},
        )

    def test_first_changed_poll_date_pairs_on_content_not_identifier(self) -> None:
        """The unit-level statement of the same rule."""

        def poll(poll_id: str, published: str, m: float = 20.0) -> dict:
            return {
                "poll_id": poll_id,
                "company": "Test",
                "house": "Test",
                "publication_date": published,
                "fieldwork_start": published,
                "fieldwork_end": published,
                "n": 1000,
                "parties": {"M": m, "S": 30.0},
            }

        previous = [poll("a", "2026-05-24"), poll("b", "2026-05-26")]
        # Every id rewritten, nothing else: not a change.
        renumbered = [poll("z1", "2026-05-24"), poll("z2", "2026-05-26")]
        self.assertIsNone(first_changed_poll_date(previous, renumbered))
        # Renumbered *and* appended past the window: still not a change.
        self.assertIsNone(
            first_changed_poll_date(
                previous, [*renumbered, poll("z3", "2026-06-01")]
            )
        )
        # Renumbering does not hide a revision.
        self.assertEqual(
            first_changed_poll_date(
                previous, [poll("z1", "2026-05-24"), poll("z2", "2026-05-26", m=21.0)]
            ),
            date(2026, 5, 26),
        )
        # A duplicated observation is a change: the multiset differs even
        # though the set does not.
        self.assertEqual(
            first_changed_poll_date(previous, [*previous, poll("z9", "2026-05-26")]),
            date(2026, 5, 26),
        )
        # A poll with no id at all still participates.
        idless = [dict(p, poll_id=None) for p in previous]
        self.assertIsNone(first_changed_poll_date(previous, idless))

    def test_first_changed_poll_date_ignores_appends_after_the_window(self) -> None:
        def poll(poll_id: str, published: str, m: float = 20.0) -> dict:
            return {
                "poll_id": poll_id,
                "company": "Test",
                "house": "Test",
                "publication_date": published,
                "fieldwork_start": published,
                "fieldwork_end": published,
                "n": 1000,
                "parties": {"M": m, "S": 30.0},
            }

        previous = [poll("a", "2026-05-24"), poll("b", "2026-05-26")]
        self.assertIsNone(first_changed_poll_date(previous, list(previous)))
        # An append past the recorded window cannot have changed any point.
        self.assertIsNone(
            first_changed_poll_date(previous, [*previous, poll("c", "2026-06-01")])
        )
        # A revision inside the window invalidates that date onward.
        revised = [poll("a", "2026-05-24"), poll("b", "2026-05-26", m=21.0)]
        self.assertEqual(
            first_changed_poll_date(previous, revised), date(2026, 5, 26)
        )
        # So does a removal.
        self.assertEqual(
            first_changed_poll_date(previous, [poll("b", "2026-05-26")]),
            date(2026, 5, 24),
        )

    def test_resume_reuses_existing_points_and_generates_only_missing_dates(self) -> None:
        votes, seats = self._matrices()
        calls: list[tuple[str, int]] = []

        def runner(*, as_of: str, election_date: str, samples: int, seed: int):
            calls.append((as_of, samples))
            return SimpleNamespace(
                vote_shares_matrix=votes,
                seats_matrix=seats,
                manifest={"source_git_commit": "d" * 40},
            )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "polls.csv"
            self._poll_csv(path)
            first = build_history(
                archive_dir=None,
                dates=["2026-05-23"],
                poll_file=path,
                samples=4,
                simulation_runner=runner,
                model_commit="d" * 40,
                source_worktree_clean=True,
                production_latest_samples=4,
            )
            second = build_history(
                archive_dir=None,
                dates=["2026-05-23", "2026-05-24"],
                poll_file=path,
                samples=4,
                existing_payload=first,
                simulation_runner=runner,
                model_commit="d" * 40,
                source_worktree_clean=True,
                production_latest_samples=4,
            )
        self.assertEqual(calls, [("2026-05-23", 4), ("2026-05-24", 4)])
        self.assertEqual(len(second["series"]), 2)
        self.assertEqual(second["resume_diagnostics"]["existing_points_reused"], 1)
        self.assertEqual(second["resume_diagnostics"]["new_points_generated"], 1)

    def test_latest_result_is_used_as_exact_official_current_point(self) -> None:
        votes, seats = self._matrices()
        calls: list[tuple[str, int]] = []

        def runner(*, as_of: str, election_date: str, samples: int, seed: int):
            calls.append((as_of, samples))
            return SimpleNamespace(
                vote_shares_matrix=votes,
                seats_matrix=seats,
                manifest={"source_git_commit": "f" * 40},
            )

        latest = SimpleNamespace(
            vote_shares_matrix=votes,
            seats_matrix=seats,
            summary=SimpleNamespace(as_of="2026-05-24"),
            manifest={"source_git_commit": "f" * 40},
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "polls.csv"
            self._poll_csv(path)
            payload = build_history(
                archive_dir=None,
                dates=["2026-05-23", "2026-05-24"],
                poll_file=path,
                samples=4,
                latest_result=latest,
                production_latest_samples=4,
                simulation_runner=runner,
                model_commit="f" * 40,
                source_worktree_clean=True,
            )
        self.assertEqual(calls, [("2026-05-23", 4)])
        current = payload["series"][-1]
        self.assertEqual(current["date"], "2026-05-24")
        self.assertEqual(current["samples"], 4)
        self.assertEqual(current["provenance"], "current_production")

    def test_latest_result_replaces_cached_official_point(self) -> None:
        """A certified rerun must replace, rather than reuse, the cached latest point."""
        votes, seats = self._matrices()

        def runner(*, as_of: str, election_date: str, samples: int, seed: int):
            return SimpleNamespace(
                vote_shares_matrix=votes,
                seats_matrix=seats,
                manifest={"source_git_commit": "f" * 40},
            )

        latest = SimpleNamespace(
            vote_shares_matrix=votes,
            seats_matrix=seats,
            summary=SimpleNamespace(as_of="2026-05-24"),
            manifest={"source_git_commit": "f" * 40},
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "polls.csv"
            self._poll_csv(path)
            initial = build_history(
                archive_dir=None,
                dates=["2026-05-23", "2026-05-24"],
                poll_file=path,
                samples=4,
                production_latest_samples=4,
                simulation_runner=runner,
                model_commit="f" * 40,
                source_worktree_clean=True,
            )
            replaced = build_history(
                archive_dir=None,
                dates=["2026-05-23", "2026-05-24"],
                poll_file=path,
                samples=4,
                existing_payload=initial,
                latest_result=latest,
                production_latest_samples=4,
                simulation_runner=lambda **kwargs: self.fail("cached point must not be rerun"),
                production_metadata={
                    "publication_generation": "latest-generation",
                    "deterministic_payload_sha256": "a" * 64,
                    "generated_at_utc": "2026-05-24T12:00:00+00:00",
                },
                model_commit="f" * 40,
                source_worktree_clean=True,
            )
        current = next(point for point in replaced["series"] if point["date"] == "2026-05-24")
        self.assertEqual(current["publication_generation"], "latest-generation")
        self.assertEqual(
            current["groups"],
            build_groups_from_matrices(votes, seats),
        )

    def test_default_runner_requests_production_draw_count_only_for_latest_date(self) -> None:
        votes, seats = self._matrices()
        calls: list[tuple[str, int]] = []

        def runner(*, as_of: str, election_date: str, samples: int, seed: int):
            calls.append((as_of, samples))
            multiplier = 2 if samples == 8 else 1
            return SimpleNamespace(
                vote_shares_matrix=np.tile(votes, (multiplier, 1)),
                seats_matrix=np.tile(seats, (multiplier, 1)),
                manifest={"source_git_commit": "9" * 40},
            )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "polls.csv"
            self._poll_csv(path)
            payload = build_history(
                archive_dir=None,
                dates=["2026-05-23", "2026-05-24"],
                poll_file=path,
                samples=4,
                production_latest_samples=8,
                simulation_runner=runner,
                model_commit="9" * 40,
                source_worktree_clean=True,
            )
        self.assertEqual(calls, [("2026-05-23", 4), ("2026-05-24", 8)])
        self.assertEqual([point["samples"] for point in payload["series"]], [4, 8])

    def test_serialize_poll_of_polls_timeseries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pop.csv"
            path.write_text(
                "date,M,L,C,KD,S,V,MP,SD,FI,other\n"
                "2022-09-17,19.0,4.5,6.5,5.0,30.0,6.5,5.0,20.0,3.0,\n"
                "2022-09-18,19.1,4.6,6.7,5.3,30.3,6.7,5.1,20.5,3.1,\n"
                "2022-09-19,19.2,4.7,6.8,5.4,30.4,6.8,5.2,20.6,3.2,\n",
                encoding="utf-8",
            )
            pop = serialize_poll_of_polls_timeseries(path, start_date="2022-09-18", end_date="2022-09-18")
            self.assertEqual(len(pop), 1)
            self.assertEqual(pop[0]["date"], "2022-09-18")
            self.assertEqual(list(pop[0]["parties"]), list(PARLIAMENTARY_PARTIES_8))
            self.assertEqual(pop[0]["parties"]["M"], 19.1)
            self.assertEqual(pop[0]["parties"]["S"], 30.3)

    def test_pop_coalition_share_uses_exact_8_party_denominator(self) -> None:
        parties = {
            "M": 20.0,
            "L": 5.0,
            "C": 10.0,
            "KD": 5.0,
            "S": 30.0,
            "V": 10.0,
            "MP": 8.0,
            "SD": 12.0,
        }
        denom = sum(parties.values())  # 100.0
        rgc_parties = DEFAULT_COALITIONS["red_green_center"]  # V, MP, S, C
        rgc_share = 100.0 * sum(parties[p] for p in rgc_parties) / denom
        self.assertAlmostEqual(rgc_share, 58.0)

    def test_serial_and_parallel_execution_determinism(self) -> None:
        dates = ["2026-05-23", "2026-05-24"]
        serial = build_history(
            archive_dir=None,
            dates=dates,
            samples=50,
            production_latest_samples=50,
            workers=1,
            seed=12345,
            model_commit="a" * 40,
            source_worktree_clean=True,
        )
        parallel = build_history(
            archive_dir=None,
            dates=dates,
            samples=50,
            production_latest_samples=50,
            workers=4,
            seed=12345,
            model_commit="a" * 40,
            source_worktree_clean=True,
        )
        # Runtime timestamp is the only transient metadata
        serial.pop("generated_at_utc", None)
        parallel.pop("generated_at_utc", None)
        self.assertEqual(serial, parallel)
        self.assertEqual(
            serial["deterministic_content_sha256"],
            parallel["deterministic_content_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
