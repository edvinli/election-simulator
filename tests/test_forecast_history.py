"""Tests for the offline historical coalition forecast generator."""

from __future__ import annotations

from copy import deepcopy
import csv
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
    build_history,
    build_history_dates,
    filter_swedishpolls_as_of,
    filter_swedishpolls_period,
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
