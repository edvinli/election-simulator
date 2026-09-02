"""Regression tests for the conditional future forecast fan."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from scripts.forecast_history.contract import DEFAULT_COALITIONS, build_groups_from_matrices
from scripts.forecast_history.future_projection import (
    ELECTION_NOISE_RNG_POLICY,
    PROJECTION_ASSUMPTION,
    build_future_projection,
    election_day_label_sv,
    projection_tooltip_sv,
    validate_future_projection_contract,
)
from scripts.forecast_history.projection_simulator import simulate_conditional_projection
from scripts.simulator.engine import simulate_election
from scripts.vote_share_calibration.national_engine import generate_national_vote_shares


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "data" / "processed"


class FutureProjectionContractTests(unittest.TestCase):
    @staticmethod
    def _matrices() -> tuple[np.ndarray, np.ndarray]:
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

    def _fixture(self):
        votes, seats = self._matrices()
        groups = build_groups_from_matrices(votes, seats)
        anchor = {
            "date": "2026-09-02",
            "samples": 100_000,
            "horizon_days": 11,
            "dynamics_horizon_days": 11,
            "provenance": "current_production",
            "groups": groups,
        }
        calls: list[dict] = []

        def runner(**kwargs):
            calls.append(dict(kwargs))
            return SimpleNamespace(
                summary=SimpleNamespace(as_of=kwargs["as_of"]),
                vote_shares_matrix=votes,
                seats_matrix=seats,
            )

        projection = build_future_projection(
            origin_date="2026-09-02",
            election_date="2026-09-13",
            anchor_point=anchor,
            samples=4,
            seed=7,
            coalitions=DEFAULT_COALITIONS,
            projection_runner=runner,
        )
        history = {
            "election_date": "2026-09-13",
            "coalitions": {key: list(value) for key, value in DEFAULT_COALITIONS.items()},
            "series": [anchor],
            "future_projection": projection,
        }
        return history, projection, calls, votes, seats

    def test_every_projection_call_freezes_as_of_and_only_shrinks_horizon(self) -> None:
        history, projection, calls, _, _ = self._fixture()
        validate_future_projection_contract(history)
        self.assertEqual(len(calls), 11)
        self.assertEqual({call["as_of"] for call in calls}, {"2026-09-02"})
        self.assertEqual(
            [call["dynamics_horizon_days"] for call in calls],
            list(range(10, -1, -1)),
        )
        self.assertTrue(all(call["election_date"] == "2026-09-13" for call in calls))
        self.assertEqual(projection["assumption"], PROJECTION_ASSUMPTION)
        self.assertEqual(projection["election_noise_rng_policy"], ELECTION_NOISE_RNG_POLICY)
        self.assertFalse(projection["future_measurements_known"])

    def test_projection_dates_are_strictly_future_daily_and_end_on_election_day(self) -> None:
        history, projection, _, _, _ = self._fixture()
        validate_future_projection_contract(history)
        dates = [date.fromisoformat(point["date"]) for point in projection["series"]]
        self.assertEqual(dates[0], date(2026, 9, 3))
        self.assertEqual(dates[-1], date(2026, 9, 13))
        self.assertEqual(len(dates), 11)
        self.assertEqual(
            [point["remaining_horizon_days"] for point in projection["series"]],
            list(range(10, -1, -1)),
        )
        self.assertEqual(projection["series"][-1]["remaining_horizon_days"], 0)

    def test_anchor_is_exact_current_point_and_future_points_keep_joint_quantiles(self) -> None:
        history, projection, _, votes, seats = self._fixture()
        validate_future_projection_contract(history)
        self.assertEqual(projection["anchor"]["groups"], history["series"][0]["groups"])
        expected = build_groups_from_matrices(votes, seats)
        for point in projection["series"]:
            self.assertEqual(point["groups"], expected)
            for coalition in DEFAULT_COALITIONS:
                self.assertEqual(list(point["groups"][coalition]), ["vote", "seats"])
                self.assertEqual(
                    list(point["groups"][coalition]["vote"]),
                    ["p05", "p25", "p50", "p75", "p95"],
                )
                self.assertEqual(
                    list(point["groups"][coalition]["seats"]),
                    ["p05", "p25", "p50", "p75", "p95"],
                )

    def test_rendering_contract_extends_axis_and_forbids_future_poll_dots(self) -> None:
        history, projection, _, _, _ = self._fixture()
        validate_future_projection_contract(history)
        rendering = projection["rendering"]
        self.assertEqual(rendering["x_axis_max"], "2026-09-13")
        self.assertEqual(rendering["future_region"]["start"], "2026-09-02")
        self.assertEqual(rendering["future_region"]["end"], "2026-09-13")
        self.assertEqual(rendering["latest_forecast_label"], "Senaste prognos")
        self.assertEqual(rendering["election_day_label"], "Valdag 13 sep")
        self.assertEqual(rendering["election_day_label"], election_day_label_sv("2026-09-13"))
        self.assertEqual(rendering["legend_label"], "Framåtblickande projektion")
        self.assertEqual(rendering["units"], ["vote", "seats"])
        self.assertFalse(rendering["poll_observations_in_future"])
        self.assertFalse(rendering["poll_of_polls_observations_in_future"])
        self.assertIn("framtida mätningar är okända", projection["tooltip_sv"])
        self.assertEqual(projection["tooltip_sv"], projection_tooltip_sv("2026-09-02"))

    def test_validator_rejects_future_data_leakage_and_broken_election_boundary(self) -> None:
        history, _, _, _, _ = self._fixture()
        leaked = deepcopy(history)
        leaked["series"].append(
            {
                **deepcopy(leaked["series"][0]),
                "date": "2026-09-03",
                "provenance": "reconstructed_current_model",
            }
        )
        with self.assertRaisesRegex(ValueError, "mixed into historical series"):
            validate_future_projection_contract(leaked)

        bad_end = deepcopy(history)
        bad_end["future_projection"]["series"].pop()
        with self.assertRaisesRegex(ValueError, "exactly one point"):
            validate_future_projection_contract(bad_end)

        bad_horizon = deepcopy(history)
        bad_horizon["future_projection"]["series"][-1]["remaining_horizon_days"] = 1
        with self.assertRaisesRegex(ValueError, "remaining horizon"):
            validate_future_projection_contract(bad_horizon)

    def test_validator_rejects_historical_poll_after_projection_origin(self) -> None:
        history, _, _, _, _ = self._fixture()
        history["polls"] = [{"publication_date": "2026-09-03"}]
        with self.assertRaisesRegex(ValueError, r"history\.polls.*after"):
            validate_future_projection_contract(history)

    def test_validator_rejects_poll_of_polls_after_projection_origin(self) -> None:
        history, _, _, _, _ = self._fixture()
        history["poll_of_polls"] = [{"date": "2026-09-03"}]
        with self.assertRaisesRegex(ValueError, r"history\.poll_of_polls.*after"):
            validate_future_projection_contract(history)

    def test_election_day_label_is_derived_from_contract_date(self) -> None:
        votes, seats = self._matrices()
        anchor = {
            "date": "2026-10-03",
            "samples": 100_000,
            "provenance": "current_production",
            "groups": build_groups_from_matrices(votes, seats),
        }
        projection = build_future_projection(
            origin_date="2026-10-03",
            election_date="2026-10-04",
            anchor_point=anchor,
            samples=4,
            projection_runner=lambda **kwargs: SimpleNamespace(
                summary=SimpleNamespace(as_of=kwargs["as_of"]),
                vote_shares_matrix=votes,
                seats_matrix=seats,
            ),
        )
        self.assertEqual(projection["rendering"]["election_day_label"], "Valdag 4 okt")

    def test_election_day_origin_has_no_hypothetical_future_points(self) -> None:
        votes, seats = self._matrices()
        groups = build_groups_from_matrices(votes, seats)
        anchor = {
            "date": "2026-09-13",
            "samples": 100_000,
            "horizon_days": 0,
            "dynamics_horizon_days": 0,
            "provenance": "current_production",
            "groups": groups,
        }
        projection = build_future_projection(
            origin_date="2026-09-13",
            election_date="2026-09-13",
            anchor_point=anchor,
            samples=4,
            projection_runner=lambda **_: self.fail("election day should need no future simulation"),
        )
        history = {
            "election_date": "2026-09-13",
            "coalitions": {key: list(value) for key, value in DEFAULT_COALITIONS.items()},
            "series": [anchor],
            "future_projection": projection,
        }
        validate_future_projection_contract(history)
        self.assertEqual(projection["series"], [])


@unittest.skipUnless(
    (PROCESSED / "pollofpolls" / "pollofpolls_timeseries.csv").is_file(),
    "processed forecast inputs are not available",
)
class ProjectionScientificParityTests(unittest.TestCase):
    def test_natural_horizon_matches_frozen_production_path(self) -> None:
        """The projection path must reproduce production at the natural horizon."""

        origin = "2026-08-24"
        election = "2026-09-13"
        horizon = (date.fromisoformat(election) - date.fromisoformat(origin)).days
        production = simulate_election(
            as_of=origin,
            election_date=election,
            samples=2,
            seed=12345,
            data_dir=PROCESSED,
        )
        projected = simulate_conditional_projection(
            as_of=origin,
            election_date=election,
            dynamics_horizon_days=horizon,
            samples=2,
            seed=12345,
            data_dir=PROCESSED,
        )
        np.testing.assert_allclose(
            projected.vote_shares_matrix,
            production.vote_shares_matrix,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_array_equal(projected.seats_matrix, production.seats_matrix)
        self.assertEqual(
            projected.diagnostics["election_noise_seed_horizon_days"],
            horizon,
        )
        self.assertEqual(
            projected.diagnostics["election_noise_rng_policy"],
            ELECTION_NOISE_RNG_POLICY,
        )

    def test_election_noise_draws_are_common_across_projection_horizons(self) -> None:
        natural = simulate_conditional_projection(
            as_of="2026-09-02",
            election_date="2026-09-13",
            dynamics_horizon_days=11,
            samples=2,
            seed=12345,
            data_dir=PROCESSED,
        )
        election_day = simulate_conditional_projection(
            as_of="2026-09-02",
            election_date="2026-09-13",
            dynamics_horizon_days=0,
            samples=2,
            seed=12345,
            data_dir=PROCESSED,
        )
        self.assertEqual(
            natural.diagnostics["election_noise_seed"],
            election_day.diagnostics["election_noise_seed"],
        )
        self.assertEqual(
            natural.diagnostics["election_noise_seed_horizon_days"],
            11,
        )
        self.assertEqual(
            election_day.diagnostics["election_noise_seed_horizon_days"],
            11,
        )

    def test_election_day_projection_has_exactly_zero_dynamics(self) -> None:
        projected = simulate_conditional_projection(
            as_of="2026-09-02",
            election_date="2026-09-13",
            dynamics_horizon_days=0,
            samples=2,
            seed=12345,
            data_dir=PROCESSED,
        )
        self.assertEqual(projected.diagnostics["dynamics_horizon_days"], 0)
        self.assertEqual(projected.diagnostics["dynamics_eval_horizon"], 0)
        self.assertEqual(projected.diagnostics["eligible_transitions_count"], 0)
        self.assertTrue(np.all(projected.seats_matrix.sum(axis=1) == 349))

        production = generate_national_vote_shares(
            as_of="2026-09-13",
            election_date="2026-09-13",
            samples=2,
            seed=12345,
            data_dir=PROCESSED,
        )
        self.assertEqual(production.horizon_days, 1)
        self.assertEqual(projected.diagnostics["dynamics_horizon_days"], 0)


if __name__ == "__main__":
    unittest.main()
