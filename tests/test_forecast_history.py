"""Tests for the offline historical coalition forecast generator."""

from __future__ import annotations

from copy import deepcopy
import csv
from datetime import date
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from scripts.forecast_history.contract import (
    DEFAULT_COALITIONS,
    HISTORY_PARTY_ORDER,
    build_groups_from_matrices,
    coalition_seat_draws,
    coalition_vote_draws,
)
from scripts.forecast_history.generate import (
    DEFAULT_HISTORY_WORKERS,
    PRODUCTION_HISTORY_WORKERS,
    backfill_reconstructed_curve,
    build_history,
    build_history_dates,
    filter_swedishpolls_as_of,
    filter_swedishpolls_period,
    first_changed_poll_date,
    missing_curve_dates,
    resolve_history_workers,
    serialize_poll_of_polls_timeseries,
    serialize_swedishpolls,
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
            holed["reconstruction_inputs"]["dates"].pop("2026-05-25")
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



REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _stub_result(as_of, samples: int, seed: int):
    """A deterministic joint result in the shape the contract requires."""

    # The two matrices are deliberately different widths: a vote share is
    # defined on the nine-category model composition (the eight parties plus
    # REST), while seats exist only for the eight that can hold them.
    width = len(HISTORY_PARTY_ORDER)
    votes = np.tile(np.full(width + 1, 100 / (width + 1)), (samples, 1))
    # Every seat draw must total exactly 349, so the remainder is distributed
    # rather than dropped: 8 * 43 is 344, not a chamber.
    row = np.full(width, 349 // width, dtype=int)
    row[: 349 - (349 // width) * width] += 1
    seats = np.tile(row, (samples, 1))
    return SimpleNamespace(
        summary=SimpleNamespace(as_of=as_of, total_samples=samples),
        vote_shares_matrix=votes,
        seats_matrix=seats,
        manifest={"source_git_commit": "0" * 40, "source_worktree_clean": True,
                  "base_seed": seed},
    )


class HistoryBackfillWorkersTests(unittest.TestCase):
    """Parallel backfill must be an execution detail, never a result detail.

    The reconstructed curve is the one part of a publication whose cost is
    unbounded by the schedule: a retroactive upstream revision can invalidate a
    hundred points at once. On 2026-09-08 that ran 119 points serially and
    exhausted the 120-minute job timeout, so the certified forecast never
    published at all.

    Parallelism is the remedy, which makes the load-bearing question not "is it
    faster" but "is it the same". These tests pin that, and pin that an injected
    runner still cannot be handed to a subprocess.

    ``production_latest_samples`` is pinned down to the reconstruction draw
    count throughout. Its 100_000 default applies to whichever date is the
    latest, and simulating that exercises the exact-Fraction allocator fallback
    for ~200s -- the certified forecast's cost, not the backfill's, and not
    this suite's subject.
    """

    ELECTION = "2026-09-13"
    POLL_FILE = REPOSITORY_ROOT / "data/processed/pollofpolls/swedishpolls_individual_polls.csv"
    TIMESERIES = REPOSITORY_ROOT / "data/processed/pollofpolls/pollofpolls_timeseries.csv"
    # Small but real: the parallel path refuses an injected runner by design, so
    # proving equivalence requires the actual simulator.
    DATES = ("2026-03-01", "2026-03-08", "2026-03-15")
    SAMPLES = 300
    SEED = 12345

    def _build(self, *, workers: int, **extra):
        return build_history(
            election_date=self.ELECTION,
            dates=list(self.DATES),
            samples=self.SAMPLES,
            production_latest_samples=self.SAMPLES,
            seed=self.SEED,
            poll_file=self.POLL_FILE,
            timeseries_file=self.TIMESERIES,
            archive_dir=None,
            workers=workers,
            **extra,
        )

    def test_the_resolver_bounds_a_request_by_available_cpus(self) -> None:
        self.assertEqual(DEFAULT_HISTORY_WORKERS, 1, "callers must default to serial")
        self.assertEqual(resolve_history_workers(None), DEFAULT_HISTORY_WORKERS)
        self.assertEqual(resolve_history_workers(1), 1)
        available = os.cpu_count() or 1
        self.assertEqual(resolve_history_workers(available + 64), available)
        self.assertLessEqual(resolve_history_workers(PRODUCTION_HISTORY_WORKERS), available)
        for refused in (0, -1):
            with self.subTest(refused=refused):
                with self.assertRaises(ValueError):
                    resolve_history_workers(refused)
        with self.assertRaises(ValueError):
            resolve_history_workers("four")

    def test_the_same_seed_and_inputs_give_the_same_artifact_either_way(self) -> None:
        """The whole justification for enabling parallelism in production.

        Compared on the artifact's own deterministic hash and then on the full
        payload, so a difference anywhere -- draws, ordering, resume
        diagnostics -- fails, not only a difference this test thought to name.
        """

        serial_plan: list[tuple[int, int]] = []
        parallel_plan: list[tuple[int, int]] = []
        serial = self._build(
            workers=1,
            workload_callback=lambda dates, workers: serial_plan.append((dates, workers)),
        )
        parallel = self._build(
            workers=4,
            workload_callback=lambda dates, workers: parallel_plan.append((dates, workers)),
        )
        # Without this the test could pass vacuously: a request that quietly
        # fell back to serial would trivially agree with serial.
        self.assertEqual(serial_plan, [(len(self.DATES), 1)])
        self.assertEqual(
            parallel_plan,
            [(len(self.DATES), min(4, os.cpu_count() or 1))],
            "the parallel path was not entered, so equivalence proves nothing",
        )
        self.assertEqual(
            serial["deterministic_content_sha256"],
            parallel["deterministic_content_sha256"],
        )
        self.assertEqual(serial, parallel)
        # The comparison is only worth anything if work actually happened, and
        # on more than one date, or the parallel path was never entered.
        self.assertEqual(
            [point["date"] for point in serial["series"]], list(self.DATES)
        )
        reconstructed = [
            point for point in serial["series"]
            if point["provenance"] == "reconstructed_current_model"
        ]
        # The latest date is the certified point; the rest are reconstruction.
        self.assertEqual(len(reconstructed), len(self.DATES) - 1)
        self.assertGreater(len(reconstructed), 1)

    def test_an_injected_runner_stays_on_the_serial_seam(self) -> None:
        """A closure cannot cross a process boundary.

        Every test here that injects a runner relies on it being called
        in-process. Requesting workers must not silently move that work into a
        subprocess, where the injected runner would not exist and the real
        simulator would run instead.
        """

        seen: list[str] = []

        def runner(**kwargs):
            seen.append(str(kwargs.get("as_of")))
            return _stub_result(kwargs.get("as_of"), int(kwargs["samples"]), self.SEED)

        observed: list[tuple[int, int]] = []
        self._build(
            workers=8,
            simulation_runner=runner,
            workload_callback=lambda dates, workers: observed.append((dates, workers)),
        )
        self.assertEqual(sorted(seen), sorted(self.DATES))
        # Reported as serial, because that is what it was.
        self.assertEqual(observed, [(len(self.DATES), 1)])

    def test_the_workload_is_reported_before_any_simulation_runs(self) -> None:
        """What makes the next incident diagnosable.

        The count has to come from inside the reconstruction: a caller counting
        holes sees only the gaps, and on 2026-09-08 the gap was one day while
        the real workload was 119 points.
        """

        order: list[str] = []

        def runner(**kwargs):
            order.append("simulate")
            return _stub_result(kwargs.get("as_of"), int(kwargs["samples"]), self.SEED)

        self._build(
            workers=1,
            simulation_runner=runner,
            workload_callback=lambda dates, workers: order.append(
                f"plan dates={dates} workers={workers}"
            ),
        )
        self.assertEqual(order[0], f"plan dates={len(self.DATES)} workers=1")
        self.assertEqual(order.count("simulate"), len(self.DATES))

    def test_a_continuous_curve_reports_nothing_and_simulates_nothing(self) -> None:
        """Backfill runs on every publication; silence is the normal case."""

        def runner(**kwargs):
            return _stub_result(kwargs.get("as_of"), int(kwargs["samples"]), self.SEED)

        built = self._build(workers=1, simulation_runner=runner)
        observed: list[tuple[int, int]] = []
        payload, backfilled = backfill_reconstructed_curve(
            built,
            poll_file=self.POLL_FILE,
            timeseries_file=self.TIMESERIES,
            archive_dir=None,
            election_date=self.ELECTION,
            samples=self.SAMPLES,
            production_latest_samples=self.SAMPLES,
            workers=PRODUCTION_HISTORY_WORKERS,
            workload_callback=lambda dates, workers: observed.append((dates, workers)),
        )
        self.assertEqual(backfilled, [])
        self.assertEqual(observed, [], "a continuous curve must not announce work")
        self.assertEqual(payload, built)


if __name__ == "__main__":
    unittest.main()
