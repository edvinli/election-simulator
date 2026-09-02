"""Scientific and contract tests for the coherent campaign-path projection.

The decisive assertions here are the *parity* ones: the election-day endpoint
of the path model must reproduce the frozen production Dynamics v2 +
ElectionNoise forecast bitwise, so the new visualization cannot change any
published forecast probability.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from scripts.forecast_history.campaign_paths import (
    CAMPAIGN_PATH_MODEL_ID,
    DEFAULT_REPRESENTATIVE_PATHS,
    build_campaign_path_pool,
    campaign_paths_tooltip_sv,
    draw_trajectory_indices_and_signs,
    election_day_tooltip_sv,
    resolve_endpoint_horizon,
    simulate_campaign_paths,
)
from scripts.forecast_history.campaign_paths_contract import (
    PRIMARY_ROLE,
    SECONDARY_DESCRIPTION_SV,
    SECONDARY_ROLE,
    build_future_campaign_paths,
    mark_secondary_projection,
    validate_future_campaign_paths_contract,
    validate_secondary_projection_role,
)
from scripts.forecast_history.contract import DEFAULT_COALITIONS, build_groups_from_matrices
from scripts.hindcasts.models import (
    derive_shared_dynamics_seed,
    sample_shared_symmetric_dynamics,
)
from scripts.pollofpolls.state import load_timeseries_dataset
from scripts.pollofpolls.transitions import (
    build_all_historical_transitions,
    filter_transitions_as_of,
)
from scripts.simulator.engine import simulate_election
from scripts.vote_share_calibration.national_engine import generate_national_vote_shares


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "data" / "processed"
TIMESERIES = PROCESSED / "pollofpolls" / "pollofpolls_timeseries.csv"

ORIGIN = "2026-09-02"
ELECTION = "2026-09-13"
SEED = 12345

INPUTS_AVAILABLE = TIMESERIES.is_file()


# ---------------------------------------------------------------------------
# Contract-only tests: no processed inputs required
# ---------------------------------------------------------------------------


def _stub_matrices() -> tuple[np.ndarray, np.ndarray]:
    votes = np.array(
        [
            [20.0, 5.0, 10.0, 5.0, 30.0, 10.0, 8.0, 12.0, 0.0],
            [18.0, 6.0, 11.0, 4.0, 32.0, 9.0, 7.0, 13.0, 0.0],
            [22.0, 4.0, 9.0, 6.0, 28.0, 11.0, 9.0, 11.0, 0.0],
            [19.0, 7.0, 12.0, 5.0, 31.0, 8.0, 6.0, 12.0, 0.0],
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


class CampaignPathContractTests(unittest.TestCase):
    """Contract construction and every fail-closed validation branch."""

    PATH_DAYS = 11

    def _stub_simulation(
        self,
        samples: int,
        *,
        parity_verified: bool = True,
        parity_difference: float | None = 0.0,
    ) -> SimpleNamespace:
        origin = date.fromisoformat(ORIGIN)
        days = tuple(origin + timedelta(days=offset) for offset in range(self.PATH_DAYS + 1))
        generator = np.random.default_rng(4)
        draws = {
            key: 40.0 + generator.random((self.PATH_DAYS + 1, samples)) * 10.0
            for key in DEFAULT_COALITIONS
        }
        return SimpleNamespace(
            origin_date=origin,
            election_date=date.fromisoformat(ELECTION),
            path_days=self.PATH_DAYS,
            samples=samples,
            seed=SEED,
            day_dates=days,
            coalition_draws=draws,
            representative_indices=tuple(range(min(4, samples))),
            endpoint_national_shares=np.full((samples, 9), 1.0 / 9.0),
            endpoint_opinion_composition=np.full((samples, 9), 100.0 / 9.0),
            diagnostics={
                "model_id": CAMPAIGN_PATH_MODEL_ID,
                "eligible_trajectories": 4357,
                "earliest_trajectory_start": "2014-09-15",
                "latest_trajectory_end": "2026-08-30",
                "endpoint_horizon_days": self.PATH_DAYS,
                "time_warp": "identity",
                "opinion_state_seed": 1,
                "dynamics_seed": 2,
                "election_noise_seed": 3,
                "endpoint_parity_verified": parity_verified,
                "endpoint_parity_max_abs_difference_pp": parity_difference,
                "endpoint_parity_reference": "generate_national_vote_shares",
            },
        )

    def _fixture(self) -> tuple[dict, dict]:
        votes, seats = _stub_matrices()
        anchor = {
            "date": ORIGIN,
            "samples": 4,
            "horizon_days": self.PATH_DAYS,
            "dynamics_horizon_days": self.PATH_DAYS,
            "provenance": "current_production",
            "groups": build_groups_from_matrices(votes, seats),
        }
        campaign_paths = build_future_campaign_paths(
            origin_date=ORIGIN,
            election_date=ELECTION,
            anchor_point=anchor,
            seed=SEED,
            coalitions=DEFAULT_COALITIONS,
            path_simulator=lambda **kwargs: self._stub_simulation(kwargs["samples"]),
        )
        history = {
            "election_date": ELECTION,
            "coalitions": {key: list(value) for key, value in DEFAULT_COALITIONS.items()},
            "series": [anchor],
            "future_campaign_paths": campaign_paths,
        }
        return history, campaign_paths

    def test_contract_declares_the_primary_forward_opinion_view(self) -> None:
        history, paths = self._fixture()
        validate_future_campaign_paths_contract(history)
        self.assertEqual(paths["role"], PRIMARY_ROLE)
        self.assertEqual(paths["model_id"], CAMPAIGN_PATH_MODEL_ID)
        self.assertEqual(paths["quantity"], "underlying_opinion_share")
        self.assertEqual(paths["path_days"], self.PATH_DAYS)
        self.assertFalse(paths["future_measurements_known"])
        self.assertEqual(paths["tooltip_sv"], campaign_paths_tooltip_sv(ORIGIN, ELECTION))

    def test_construction_metadata_disclaims_the_rejected_alternatives(self) -> None:
        history, paths = self._fixture()
        validate_future_campaign_paths_contract(history)
        construction = paths["path_construction"]
        self.assertEqual(construction["space"], "clr")
        self.assertEqual(construction["categories"], 9)
        self.assertEqual(construction["sign_policy"], "single_sign_per_whole_trajectory")
        self.assertEqual(construction["transition_pool"], "all_history_leakage_safe")
        self.assertEqual(construction["leakage_rule"], "trajectory_end_le_origin")
        self.assertFalse(construction["synthesized_future_polls"])
        self.assertFalse(construction["daily_independent_random_walk"])
        self.assertFalse(construction["directional_momentum"])
        self.assertLessEqual(
            date.fromisoformat(construction["latest_trajectory_end"]),
            date.fromisoformat(ORIGIN),
        )

    def test_bands_are_daily_from_the_origin_and_vote_only(self) -> None:
        history, paths = self._fixture()
        validate_future_campaign_paths_contract(history)
        bands = paths["bands"]
        self.assertEqual(len(bands), self.PATH_DAYS + 1)
        self.assertEqual(bands[0]["date"], ORIGIN)
        self.assertEqual(bands[0]["path_day"], 0)
        self.assertEqual(bands[-1]["date"], ELECTION)
        self.assertEqual(bands[-1]["path_day"], self.PATH_DAYS)
        for band in bands:
            for coalition in DEFAULT_COALITIONS:
                self.assertEqual(list(band["groups"][coalition]), ["vote"])
                self.assertEqual(
                    list(band["groups"][coalition]["vote"]),
                    ["p05", "p25", "p50", "p75", "p95"],
                )

    def test_election_day_reuses_the_certified_production_summaries(self) -> None:
        history, paths = self._fixture()
        validate_future_campaign_paths_contract(history)
        election_day = paths["election_day"]
        self.assertEqual(election_day["groups"], history["series"][0]["groups"])
        self.assertEqual(election_day["samples"], history["series"][0]["samples"])
        self.assertTrue(election_day["includes_election_noise"])
        self.assertTrue(election_day["includes_geography_and_mandates"])
        self.assertEqual(election_day["provenance"], "current_production")
        self.assertEqual(election_day["label_sv"], "Valdagsprognos")
        self.assertEqual(election_day["tooltip_sv"], election_day_tooltip_sv(ELECTION))

    def test_rendering_metadata_forbids_future_observations_and_seat_paths(self) -> None:
        history, paths = self._fixture()
        validate_future_campaign_paths_contract(history)
        rendering = paths["rendering"]
        self.assertEqual(rendering["x_axis_max"], ELECTION)
        self.assertEqual(rendering["future_region"]["start"], ORIGIN)
        self.assertEqual(rendering["future_region"]["end"], ELECTION)
        self.assertEqual(rendering["future_region"]["background"], "light_distinct")
        self.assertEqual(rendering["future_region"]["label"], "Möjliga opinionsbanor")
        self.assertEqual(rendering["interval_bands"], ["p25_p75", "p05_p95"])
        self.assertEqual(rendering["path_units"], ["vote"])
        self.assertEqual(rendering["election_day_units"], ["vote", "seats"])
        self.assertTrue(rendering["median_may_be_flat"])
        self.assertFalse(rendering["intermediate_seat_trajectory"])
        self.assertFalse(rendering["poll_observations_in_future"])
        self.assertFalse(rendering["poll_of_polls_observations_in_future"])
        self.assertEqual(rendering["continues_from"], "poll_of_polls_opinion_series")

    def test_representative_paths_are_ordered_unique_and_in_range(self) -> None:
        history, paths = self._fixture()
        validate_future_campaign_paths_contract(history)
        indices = paths["paths"]["sample_indices"]
        self.assertEqual(indices, sorted(set(indices)))
        self.assertEqual(paths["paths"]["count"], len(paths["paths"]["series"]))
        for item in paths["paths"]["series"]:
            self.assertIn(item["sample_index"], indices)
            for coalition in DEFAULT_COALITIONS:
                track = item["values"][coalition]
                self.assertEqual(len(track), self.PATH_DAYS + 1)
                self.assertTrue(all(0.0 <= value <= 100.0 for value in track))

    def test_validator_rejects_a_leaked_trajectory_end(self) -> None:
        history, _ = self._fixture()
        broken = deepcopy(history)
        broken["future_campaign_paths"]["path_construction"]["latest_trajectory_end"] = "2026-09-03"
        with self.assertRaisesRegex(ValueError, "ending after the origin"):
            validate_future_campaign_paths_contract(broken)

    def test_validator_rejects_a_declared_random_walk_or_momentum(self) -> None:
        for flag in (
            "synthesized_future_polls",
            "daily_independent_random_walk",
            "directional_momentum",
        ):
            history, _ = self._fixture()
            history["future_campaign_paths"]["path_construction"][flag] = True
            with self.assertRaisesRegex(ValueError, flag):
                validate_future_campaign_paths_contract(history)

    def test_validator_rejects_a_nonzero_verified_parity_difference(self) -> None:
        history, _ = self._fixture()
        parity = history["future_campaign_paths"]["endpoint_parity"]
        parity["verified"] = True
        parity["max_abs_vote_share_difference_pp"] = 1e-9
        with self.assertRaisesRegex(ValueError, "exactly zero difference"):
            validate_future_campaign_paths_contract(history)

    def test_validator_rejects_election_day_summaries_that_drift_from_production(self) -> None:
        history, _ = self._fixture()
        groups = history["future_campaign_paths"]["election_day"]["groups"]
        first = next(iter(groups))
        groups[first]["seats"]["p50"] = int(groups[first]["seats"]["p50"]) + 1
        with self.assertRaisesRegex(ValueError, "reproduce the certified production"):
            validate_future_campaign_paths_contract(history)

    def test_validator_rejects_missing_election_noise_at_election_day(self) -> None:
        history, _ = self._fixture()
        history["future_campaign_paths"]["election_day"]["includes_election_noise"] = False
        with self.assertRaisesRegex(ValueError, "must include ElectionNoise"):
            validate_future_campaign_paths_contract(history)

    def test_validator_rejects_seat_quantiles_inside_the_path_bands(self) -> None:
        history, _ = self._fixture()
        band = history["future_campaign_paths"]["bands"][3]["groups"]
        first = next(iter(band))
        band[first]["seats"] = {"p05": 1, "p25": 2, "p50": 3, "p75": 4, "p95": 5}
        with self.assertRaisesRegex(ValueError, "opinion vote shares only"):
            validate_future_campaign_paths_contract(history)

    def test_validator_rejects_an_implied_future_seat_trajectory(self) -> None:
        history, _ = self._fixture()
        history["future_campaign_paths"]["rendering"]["intermediate_seat_trajectory"] = True
        with self.assertRaisesRegex(ValueError, "smooth future seat trajectory"):
            validate_future_campaign_paths_contract(history)

    def test_validator_rejects_a_missing_or_short_band_series(self) -> None:
        history, _ = self._fixture()
        history["future_campaign_paths"]["bands"].pop()
        with self.assertRaisesRegex(ValueError, "every campaign day"):
            validate_future_campaign_paths_contract(history)

    def test_validator_rejects_a_non_monotone_band(self) -> None:
        history, _ = self._fixture()
        band = history["future_campaign_paths"]["bands"][2]["groups"]
        first = next(iter(band))
        band[first]["vote"]["p50"] = 0.0
        with self.assertRaisesRegex(ValueError, "monotone"):
            validate_future_campaign_paths_contract(history)

    def test_validator_rejects_a_poll_observation_after_the_origin(self) -> None:
        history, _ = self._fixture()
        history["polls"] = [{"publication_date": "2026-09-05"}]
        with self.assertRaisesRegex(ValueError, r"history\.polls.*after"):
            validate_future_campaign_paths_contract(history)

    def test_validator_rejects_a_poll_of_polls_observation_after_the_origin(self) -> None:
        history, _ = self._fixture()
        history["poll_of_polls"] = [{"date": "2026-09-05"}]
        with self.assertRaisesRegex(ValueError, r"history\.poll_of_polls.*after"):
            validate_future_campaign_paths_contract(history)

    def test_validator_rejects_a_demoted_or_absent_primary_role(self) -> None:
        history, _ = self._fixture()
        history["future_campaign_paths"]["role"] = SECONDARY_ROLE
        with self.assertRaisesRegex(ValueError, "primary future view"):
            validate_future_campaign_paths_contract(history)

    def test_validator_rejects_a_missing_object(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be an object"):
            validate_future_campaign_paths_contract({"election_date": ELECTION})

    def test_identity_warp_requires_matching_endpoint_horizon(self) -> None:
        history, _ = self._fixture()
        history["future_campaign_paths"]["path_construction"]["endpoint_horizon_days"] = 7
        with self.assertRaisesRegex(ValueError, "identity time warp"):
            validate_future_campaign_paths_contract(history)

    def test_builder_refuses_an_origin_on_or_after_election_day(self) -> None:
        votes, seats = _stub_matrices()
        anchor = {
            "date": ELECTION,
            "samples": 4,
            "provenance": "current_production",
            "groups": build_groups_from_matrices(votes, seats),
        }
        with self.assertRaisesRegex(ValueError, "strictly before"):
            build_future_campaign_paths(
                origin_date=ELECTION,
                election_date=ELECTION,
                anchor_point=anchor,
                path_simulator=lambda **_: self.fail("election day needs no path simulation"),
            )

    def _anchor(self) -> dict:
        votes, seats = _stub_matrices()
        return {
            "date": ORIGIN,
            "samples": 4,
            "provenance": "current_production",
            "groups": build_groups_from_matrices(votes, seats),
        }

    def test_builder_fails_closed_when_the_endpoint_gate_reports_a_difference(self) -> None:
        with self.assertRaisesRegex(ValueError, "not bitwise identical"):
            build_future_campaign_paths(
                origin_date=ORIGIN,
                election_date=ELECTION,
                anchor_point=self._anchor(),
                path_simulator=lambda **kwargs: self._stub_simulation(
                    kwargs["samples"], parity_verified=True, parity_difference=1e-12
                ),
            )

    def test_builder_fails_closed_when_a_verified_gate_reports_nothing(self) -> None:
        with self.assertRaisesRegex(ValueError, "not bitwise identical"):
            build_future_campaign_paths(
                origin_date=ORIGIN,
                election_date=ELECTION,
                anchor_point=self._anchor(),
                path_simulator=lambda **kwargs: self._stub_simulation(
                    kwargs["samples"], parity_verified=True, parity_difference=None
                ),
            )

    def test_an_unverified_gate_publishes_no_difference(self) -> None:
        paths = build_future_campaign_paths(
            origin_date=ORIGIN,
            election_date=ELECTION,
            anchor_point=self._anchor(),
            path_simulator=lambda **kwargs: self._stub_simulation(
                kwargs["samples"], parity_verified=False, parity_difference=None
            ),
        )
        history = {
            "election_date": ELECTION,
            "coalitions": {key: list(value) for key, value in DEFAULT_COALITIONS.items()},
            "series": [self._anchor()],
            "future_campaign_paths": paths,
        }
        validate_future_campaign_paths_contract(history)
        self.assertFalse(paths["endpoint_parity"]["verified"])
        self.assertIsNone(paths["endpoint_parity"]["max_abs_vote_share_difference_pp"])

    def test_secondary_projection_is_demoted_and_described(self) -> None:
        marked = mark_secondary_projection({"projection_type": "conditional_forward_projection"})
        validate_secondary_projection_role(marked)
        self.assertEqual(marked["role"], SECONDARY_ROLE)
        self.assertFalse(marked["primary"])
        self.assertEqual(marked["description_sv"], SECONDARY_DESCRIPTION_SV)
        self.assertIn("står stilla", marked["description_sv"])
        with self.assertRaisesRegex(ValueError, "secondary analytical role"):
            validate_secondary_projection_role({"projection_type": "x"})


# ---------------------------------------------------------------------------
# Scientific tests against the real frozen production components
# ---------------------------------------------------------------------------


@unittest.skipUnless(INPUTS_AVAILABLE, "processed forecast inputs are not available")
class CampaignPathScienceTests(unittest.TestCase):
    """Endpoint parity, leakage safety, sign symmetry and reproducibility."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.timeseries = load_timeseries_dataset(TIMESERIES)
        cls.origin = date.fromisoformat(ORIGIN)
        cls.election = date.fromisoformat(ELECTION)
        cls.path_days = (cls.election - cls.origin).days
        cls.pool = build_campaign_path_pool(cls.timeseries, cls.origin, cls.path_days)

    # ---- parity ---------------------------------------------------------

    def test_pool_is_production_transition_pool_in_the_same_order(self) -> None:
        horizon, production_pool = resolve_endpoint_horizon(
            self.timeseries, self.origin, self.path_days
        )
        self.assertEqual(horizon, self.path_days)
        self.assertEqual(
            [transition.start_date for transition in production_pool],
            list(self.pool.start_dates),
        )
        np.testing.assert_array_equal(
            np.array([transition.clr_transition for transition in production_pool]),
            self.pool.delta_tensor[:, -1, :],
        )

    def test_endpoint_delta_equals_production_dynamics_bitwise(self) -> None:
        """The whole-path model's d=n delta *is* the production Dynamics v2 draw."""

        samples = 512
        dynamics_seed = derive_shared_dynamics_seed(
            base_seed=SEED, origin_date=self.origin, horizon_days=self.path_days
        )
        indices, signs = draw_trajectory_indices_and_signs(self.pool.size, samples, dynamics_seed)
        endpoint = signs * self.pool.delta_tensor[:, -1, :][indices]
        production = sample_shared_symmetric_dynamics(
            eligible_transitions=filter_transitions_as_of(
                build_all_historical_transitions(self.timeseries, horizons=[self.path_days])[
                    self.path_days
                ],
                self.origin,
            ),
            samples_count=samples,
            seed=dynamics_seed,
        )
        self.assertTrue(np.array_equal(endpoint, production))

    def test_endpoint_national_shares_equal_production_bitwise(self) -> None:
        samples = 256
        paths = simulate_campaign_paths(
            as_of=ORIGIN,
            election_date=ELECTION,
            samples=samples,
            seed=SEED,
            data_dir=PROCESSED,
        )
        production = generate_national_vote_shares(
            as_of=ORIGIN,
            election_date=ELECTION,
            samples=samples,
            seed=SEED,
            data_dir=PROCESSED,
        )
        self.assertTrue(
            np.array_equal(paths.endpoint_opinion_composition, production.base_comp_matrix),
            "pre-noise election-day composition must be identical",
        )
        self.assertTrue(
            np.array_equal(paths.endpoint_national_shares, production.nat_shares_matrix),
            "post-ElectionNoise election-day vote shares must be identical",
        )

    def test_endpoint_reproduces_full_production_votes_and_seats(self) -> None:
        samples = 24
        production = simulate_election(
            as_of=ORIGIN,
            election_date=ELECTION,
            samples=samples,
            seed=SEED,
            data_dir=PROCESSED,
        )
        paths = simulate_campaign_paths(
            as_of=ORIGIN,
            election_date=ELECTION,
            samples=samples,
            seed=SEED,
            data_dir=PROCESSED,
        )
        self.assertTrue(
            np.array_equal(
                paths.endpoint_national_shares * 100.0, production.vote_shares_matrix
            )
        )
        # Seats are a deterministic function of the national vote matrix, so
        # bitwise vote parity is bitwise seat parity.  Assert it anyway.
        from scripts.forecast_history.projection_simulator import _allocate_seats

        seats = _allocate_seats(
            paths.endpoint_national_shares,
            election_date=self.election,
            processed_geo_dir=PROCESSED / "geography",
            baseline_year=2022,
            total_national_votes=6_500_000,
        )
        np.testing.assert_array_equal(seats, production.seats_matrix)

    def test_the_endpoint_parity_gate_is_load_bearing(self) -> None:
        """A perturbed canonical reference must stop the simulation dead."""

        import scripts.vote_share_calibration.national_engine as engine

        original = engine.generate_national_vote_shares

        def perturbed(**kwargs):
            reference = original(**kwargs)
            shifted = reference.nat_shares_matrix.copy()
            shifted[0, 0] += 1e-15
            return replace(reference, nat_shares_matrix=shifted)

        with mock.patch.object(engine, "generate_national_vote_shares", perturbed):
            with self.assertRaisesRegex(ValueError, "not bitwise identical"):
                simulate_campaign_paths(
                    as_of=ORIGIN,
                    election_date=ELECTION,
                    samples=32,
                    seed=SEED,
                    data_dir=PROCESSED,
                )

    def test_parity_verification_can_be_skipped_explicitly(self) -> None:
        paths = simulate_campaign_paths(
            as_of=ORIGIN,
            election_date=ELECTION,
            samples=32,
            seed=SEED,
            data_dir=PROCESSED,
            verify_endpoint_parity=False,
        )
        self.assertFalse(paths.diagnostics["endpoint_parity_verified"])
        self.assertIsNone(paths.diagnostics["endpoint_parity_max_abs_difference_pp"])

    def test_the_verified_gate_reports_exactly_zero(self) -> None:
        paths = simulate_campaign_paths(
            as_of=ORIGIN,
            election_date=ELECTION,
            samples=32,
            seed=SEED,
            data_dir=PROCESSED,
        )
        self.assertTrue(paths.diagnostics["endpoint_parity_verified"])
        self.assertEqual(paths.diagnostics["endpoint_parity_max_abs_difference_pp"], 0.0)
        self.assertEqual(
            paths.diagnostics["endpoint_parity_reference"], "generate_national_vote_shares"
        )

    def test_endpoint_parity_holds_beyond_the_112_day_dynamics_cap(self) -> None:
        """Above the cap the path is a monotone stretch with an exact endpoint."""

        origin = "2026-01-01"
        samples = 64
        paths = simulate_campaign_paths(
            as_of=origin,
            election_date=ELECTION,
            samples=samples,
            seed=SEED,
            data_dir=PROCESSED,
        )
        production = generate_national_vote_shares(
            as_of=origin,
            election_date=ELECTION,
            samples=samples,
            seed=SEED,
            data_dir=PROCESSED,
        )
        self.assertEqual(paths.diagnostics["time_warp"], "monotone_stretch")
        self.assertEqual(paths.diagnostics["endpoint_horizon_days"], 112)
        self.assertEqual(
            paths.diagnostics["endpoint_horizon_days"],
            production.diagnostics["dynamics_eval_horizon"],
        )
        self.assertTrue(
            np.array_equal(paths.endpoint_national_shares, production.nat_shares_matrix)
        )

    # ---- leakage safety -------------------------------------------------

    def test_pool_never_reads_an_observation_after_the_origin(self) -> None:
        for start in self.pool.start_dates:
            self.assertLessEqual(start + timedelta(days=self.pool.endpoint_horizon_days), self.origin)
        truncated = [row for row in self.timeseries if row["date"] <= self.origin]
        truncated_pool = build_campaign_path_pool(truncated, self.origin, self.path_days)
        self.assertEqual(truncated_pool.start_dates, self.pool.start_dates)
        np.testing.assert_array_equal(truncated_pool.delta_tensor, self.pool.delta_tensor)

    def test_an_earlier_origin_cannot_see_a_later_trajectory(self) -> None:
        earlier = date(2020, 1, 1)
        earlier_pool = build_campaign_path_pool(self.timeseries, earlier, self.path_days)
        latest_end = max(earlier_pool.start_dates) + timedelta(days=earlier_pool.endpoint_horizon_days)
        self.assertLessEqual(latest_end, earlier)
        self.assertLess(earlier_pool.size, self.pool.size)

    def test_pool_fails_closed_on_a_gap_in_the_poll_of_polls_series(self) -> None:
        removed = self.origin - timedelta(days=4)
        punctured = [row for row in self.timeseries if row["date"] != removed]
        with self.assertRaisesRegex(ValueError, "has a gap at"):
            build_campaign_path_pool(punctured, self.origin, self.path_days)

    # ---- path structure -------------------------------------------------

    def test_one_sign_applies_to_the_whole_trajectory(self) -> None:
        samples = 400
        dynamics_seed = derive_shared_dynamics_seed(
            base_seed=SEED, origin_date=self.origin, horizon_days=self.path_days
        )
        indices, signs = draw_trajectory_indices_and_signs(self.pool.size, samples, dynamics_seed)
        raw = self.pool.delta_tensor[indices]
        signed = signs[:, :, np.newaxis] * raw
        self.assertTrue(np.array_equal(signed, raw * signs[:, :, np.newaxis]))
        # Every non-degenerate coordinate of a draw must carry the same sign
        # ratio, i.e. the path is never re-signed part-way through.
        significant = np.abs(raw) > 1e-12
        ratios = np.where(significant, signed / np.where(significant, raw, 1.0), np.nan)
        for draw in range(samples):
            observed = ratios[draw][significant[draw]]
            self.assertEqual(len(np.unique(np.round(observed, 12))), 1)
            self.assertIn(float(observed[0]), {-1.0, 1.0})
        self.assertEqual(set(np.unique(signs)), {-1.0, 1.0})

    def test_path_is_an_exact_historical_trajectory_not_a_random_walk(self) -> None:
        """Given the index and sign, no additional per-day randomness exists."""

        samples = 200
        dynamics_seed = derive_shared_dynamics_seed(
            base_seed=SEED, origin_date=self.origin, horizon_days=self.path_days
        )
        indices, signs = draw_trajectory_indices_and_signs(self.pool.size, samples, dynamics_seed)
        for draw in (0, 7, 199):
            start = self.pool.start_dates[int(indices[draw])]
            rows = {row["date"]: row for row in self.timeseries}
            from scripts.pollofpolls.clr import composition_to_clr

            base, _ = composition_to_clr(rows[start]["composition"])
            for day in range(1, self.path_days + 1):
                observed, _ = composition_to_clr(rows[start + timedelta(days=day)]["composition"])
                expected = float(signs[draw, 0]) * (observed - base)
                np.testing.assert_array_equal(
                    self.pool.delta_tensor[int(indices[draw]), day - 1, :] * signs[draw, 0],
                    expected,
                )

    def test_every_day_stays_on_the_nine_category_simplex(self) -> None:
        paths = simulate_campaign_paths(
            as_of=ORIGIN,
            election_date=ELECTION,
            samples=512,
            seed=SEED,
            data_dir=PROCESSED,
        )
        self.assertLess(paths.diagnostics["max_composition_sum_error_pp"], 1e-9)
        row_sums = np.sum(paths.endpoint_opinion_composition, axis=1)
        np.testing.assert_allclose(row_sums, 100.0, rtol=0.0, atol=1e-9)
        self.assertTrue(np.all(paths.endpoint_opinion_composition > 0.0))
        national_sums = np.sum(paths.endpoint_national_shares, axis=1)
        np.testing.assert_allclose(national_sums, 1.0, rtol=0.0, atol=1e-12)

    def test_joint_coalition_draws_stay_within_bounds_every_day(self) -> None:
        paths = simulate_campaign_paths(
            as_of=ORIGIN,
            election_date=ELECTION,
            samples=256,
            seed=SEED,
            data_dir=PROCESSED,
        )
        for key, draws in paths.coalition_draws.items():
            self.assertEqual(draws.shape, (self.path_days + 1, 256), key)
            self.assertTrue(np.all(draws >= 0.0) and np.all(draws <= 100.0), key)
        complement = (
            paths.coalition_draws["red_green_center"] + paths.coalition_draws["tido"]
        )
        np.testing.assert_allclose(complement, 100.0, rtol=0.0, atol=1e-9)

    def test_median_stays_approximately_flat_under_sign_symmetry(self) -> None:
        paths = simulate_campaign_paths(
            as_of=ORIGIN,
            election_date=ELECTION,
            samples=20_000,
            seed=SEED,
            data_dir=PROCESSED,
        )
        medians = np.median(paths.coalition_draws["red_green_center"], axis=1)
        self.assertLess(float(np.max(np.abs(medians - medians[0]))), 0.2)

    def test_interval_width_grows_with_the_remaining_horizon(self) -> None:
        paths = simulate_campaign_paths(
            as_of="2026-05-25",
            election_date=ELECTION,
            samples=8_000,
            seed=SEED,
            data_dir=PROCESSED,
        )
        draws = paths.coalition_draws["red_green_center"]
        widths = np.quantile(draws, 0.95, axis=1) - np.quantile(draws, 0.05, axis=1)
        self.assertGreater(widths[-1], widths[0] * 1.5)
        # Monotone up to Monte Carlo noise.
        self.assertGreaterEqual(float(np.min(np.diff(widths))), -0.02)

    # ---- reproducibility ------------------------------------------------

    def test_paths_are_bit_for_bit_reproducible_for_one_seed(self) -> None:
        first = simulate_campaign_paths(
            as_of=ORIGIN, election_date=ELECTION, samples=128, seed=SEED, data_dir=PROCESSED
        )
        second = simulate_campaign_paths(
            as_of=ORIGIN, election_date=ELECTION, samples=128, seed=SEED, data_dir=PROCESSED
        )
        for key in first.coalition_draws:
            np.testing.assert_array_equal(first.coalition_draws[key], second.coalition_draws[key])
        np.testing.assert_array_equal(
            first.endpoint_national_shares, second.endpoint_national_shares
        )
        self.assertEqual(first.representative_indices, second.representative_indices)
        self.assertEqual(first.diagnostics, second.diagnostics)

    def test_a_different_seed_produces_different_paths(self) -> None:
        first = simulate_campaign_paths(
            as_of=ORIGIN, election_date=ELECTION, samples=128, seed=SEED, data_dir=PROCESSED
        )
        other = simulate_campaign_paths(
            as_of=ORIGIN, election_date=ELECTION, samples=128, seed=SEED + 1, data_dir=PROCESSED
        )
        self.assertFalse(
            np.array_equal(
                first.coalition_draws["red_green_center"],
                other.coalition_draws["red_green_center"],
            )
        )

    def test_representative_selection_is_deterministic_and_bounded(self) -> None:
        paths = simulate_campaign_paths(
            as_of=ORIGIN, election_date=ELECTION, samples=1_000, seed=SEED, data_dir=PROCESSED
        )
        self.assertEqual(len(paths.representative_indices), DEFAULT_REPRESENTATIVE_PATHS)
        self.assertEqual(paths.representative_indices[0], 0)
        self.assertEqual(paths.representative_indices[-1], 999)
        self.assertEqual(list(paths.representative_indices), sorted(paths.representative_indices))

    def test_origin_day_carries_state_uncertainty_and_no_dynamics(self) -> None:
        paths = simulate_campaign_paths(
            as_of=ORIGIN, election_date=ELECTION, samples=4_000, seed=SEED, data_dir=PROCESSED
        )
        draws = paths.coalition_draws["red_green_center"]
        origin_width = float(np.quantile(draws[0], 0.95) - np.quantile(draws[0], 0.05))
        self.assertGreater(origin_width, 0.0)
        self.assertLess(
            origin_width,
            float(np.quantile(draws[-1], 0.95) - np.quantile(draws[-1], 0.05)),
        )

    def test_simulator_refuses_an_origin_on_or_after_election_day(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly before"):
            simulate_campaign_paths(
                as_of=ELECTION,
                election_date=ELECTION,
                samples=8,
                seed=SEED,
                data_dir=PROCESSED,
            )


if __name__ == "__main__":
    unittest.main()
