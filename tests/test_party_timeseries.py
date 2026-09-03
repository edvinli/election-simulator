"""Contract tests for the additive per-party time-series family.

Three properties carry the scientific weight and each has its own test class:

1. **Definition.** A party vote share is the national share over all nine model
   categories. It is *not* the coalition renormalization, and the difference is
   large enough to move every party away from the 4 % threshold.
2. **Election-day parity.** The party values at the certified point are the
   published production party forecast, derived from the same draw matrices by
   the same quantile rule.
3. **Leakage and structure.** Nothing observed after the origin enters a party
   path, and no intermediate future mandate trajectory is ever published.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import json
from pathlib import Path
import unittest

import numpy as np

from scripts.forecast_history.campaign_paths_contract import (
    build_future_campaign_paths,
    validate_future_campaign_paths_contract,
)
from scripts.forecast_history.contract import (
    DEFAULT_COALITIONS,
    HISTORY_PARTY_ORDER,
    build_groups_from_matrices,
    coalition_vote_draws,
    validate_history_contract,
)
from scripts.forecast_history.generate import _archive_point_from_record, _point_from_result
from scripts.forecast_history.party_contract import (
    NATIONAL_THRESHOLD_PCT,
    PARTY_DEFINITION_ORDER,
    PARTY_VOTE_DENOMINATOR,
    assert_election_day_party_parity,
    build_parties_from_matrices,
    build_party_vote_quantiles,
    parties_view_metadata,
    party_point_from_archive_record,
    party_seat_draws,
    party_vote_draws,
    validate_parties_view,
    validate_party_summaries,
    validate_party_vote_only,
)
from scripts.prospective_archive.archive import _histogram
from scripts.simulator.config import MODEL_PARTIES_9

REPO_ROOT = Path(__file__).resolve().parents[1]
QUANTILE_KEYS = ("p05", "p25", "p50", "p75", "p95")


def _matrices(samples: int = 400, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic but structurally valid joint draw pair.

    Vote rows sum to exactly 100 over nine categories and seat rows to exactly
    349 over eight, which is what every helper here is entitled to assume.
    """

    rng = np.random.default_rng(seed)
    raw = rng.gamma(shape=[18.0, 2.0, 7.0, 6.5, 30.0, 7.5, 7.5, 19.0, 2.0], size=(samples, 9))
    votes = 100.0 * raw / raw.sum(axis=1, keepdims=True)
    seats = np.zeros((samples, 8), dtype=np.int64)
    eligible = votes[:, :8].copy()
    eligible[votes[:, :8] < 4.0] = 0.0
    for row in range(samples):
        weights = eligible[row]
        if weights.sum() <= 0:
            weights = votes[row, :8]
        allocation = np.floor(349 * weights / weights.sum()).astype(np.int64)
        allocation[int(np.argmax(weights))] += 349 - int(allocation.sum())
        seats[row] = allocation
    return votes, seats


class PartyVoteShareDefinitionTests(unittest.TestCase):
    def test_party_share_is_the_nine_category_national_share(self) -> None:
        votes, _ = _matrices()
        for index, party in enumerate(PARTY_DEFINITION_ORDER):
            np.testing.assert_array_equal(party_vote_draws(votes, party), votes[:, index])

    def test_party_share_is_not_the_coalition_renormalization(self) -> None:
        # The coalition denominator excludes REST, so a one-party "coalition"
        # is systematically larger. Publishing that as a party share would move
        # every party away from the threshold by the REST mass.
        votes, _ = _matrices()
        party = party_vote_draws(votes, "L")
        renormalized = coalition_vote_draws(votes, ("L",))
        self.assertTrue(np.all(renormalized > party))
        self.assertGreater(float(np.mean(renormalized - party)), 0.01)

    def test_an_eight_column_matrix_is_rejected(self) -> None:
        votes, _ = _matrices()
        with self.assertRaisesRegex(ValueError, "nine-category"):
            party_vote_draws(votes[:, :8], "M")

    def test_rest_is_never_a_party_definition(self) -> None:
        self.assertNotIn("REST", PARTY_DEFINITION_ORDER)
        self.assertEqual(list(PARTY_DEFINITION_ORDER), list(HISTORY_PARTY_ORDER))
        self.assertEqual(list(MODEL_PARTIES_9)[:8], list(PARTY_DEFINITION_ORDER))
        votes, _ = _matrices()
        with self.assertRaises(ValueError):
            party_vote_draws(votes, "REST")

    def test_seat_draws_are_one_column_of_the_joint_allocation(self) -> None:
        _, seats = _matrices()
        for index, party in enumerate(PARTY_DEFINITION_ORDER):
            np.testing.assert_array_equal(party_seat_draws(seats, party), seats[:, index])

    def test_seat_draws_reject_a_chamber_that_is_not_349(self) -> None:
        _, seats = _matrices()
        broken = seats.copy()
        broken[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "349 seats"):
            party_seat_draws(broken, "M")

    def test_the_published_threshold_is_on_the_published_scale(self) -> None:
        # The 4 % threshold is defined on the national share. Publishing it
        # alongside a renormalized share would draw the line in the wrong place.
        self.assertEqual(NATIONAL_THRESHOLD_PCT, 4.0)
        self.assertEqual(
            parties_view_metadata()["vote_share_denominator"], PARTY_VOTE_DENOMINATOR
        )
        self.assertIn("including_rest", PARTY_VOTE_DENOMINATOR)


class PartySummaryShapeTests(unittest.TestCase):
    def test_every_party_gets_vote_and_seat_quantiles(self) -> None:
        votes, seats = _matrices()
        parties = build_parties_from_matrices(votes, seats)
        self.assertEqual(list(parties), list(PARTY_DEFINITION_ORDER))
        validate_party_summaries(parties, name="parties")
        for party in PARTY_DEFINITION_ORDER:
            self.assertEqual(list(parties[party]), ["vote", "seats"])
            self.assertEqual(list(parties[party]["vote"]), list(QUANTILE_KEYS))
            for key in QUANTILE_KEYS:
                self.assertIsInstance(parties[party]["seats"][key], int)

    def test_quantiles_are_monotone_for_every_party(self) -> None:
        votes, seats = _matrices()
        parties = build_parties_from_matrices(votes, seats)
        for party, entry in parties.items():
            for metric in ("vote", "seats"):
                values = [entry[metric][key] for key in QUANTILE_KEYS]
                self.assertEqual(values, sorted(values), f"{party}.{metric}")

    def test_party_medians_need_not_sum_to_a_hundred(self) -> None:
        # Marginal medians are computed independently, exactly as the published
        # party forecast does. Asserting the opposite would be asserting a bug.
        votes, seats = _matrices()
        parties = build_parties_from_matrices(votes, seats)
        total = sum(float(parties[p]["vote"]["p50"]) for p in PARTY_DEFINITION_ORDER)
        self.assertLess(total, 100.0)

    def test_building_is_deterministic(self) -> None:
        votes, seats = _matrices()
        self.assertEqual(
            build_parties_from_matrices(votes, seats),
            build_parties_from_matrices(votes.copy(), seats.copy()),
        )

    def test_the_series_point_carries_parties_from_the_same_matrices(self) -> None:
        votes, seats = _matrices()

        class _Result:
            vote_shares_matrix = votes
            seats_matrix = seats
            manifest: dict = {}

        point = _point_from_result(
            _Result(),
            point_date=date(2026, 6, 1),
            election_date=date(2026, 9, 13),
            coalitions=DEFAULT_COALITIONS,
        )
        self.assertIn("parties", point)
        self.assertEqual(point["parties"], build_parties_from_matrices(votes, seats))
        # The coalition family is untouched.
        self.assertEqual(point["groups"], build_groups_from_matrices(votes, seats))


class ElectionDayParityTests(unittest.TestCase):
    """The published party point must *be* the certified production forecast."""

    @classmethod
    def setUpClass(cls) -> None:
        from scripts.simulator.engine import simulate_election

        cls.result = simulate_election(
            as_of="2026-09-01", election_date="2026-09-13", samples=2000, seed=12345
        )
        cls.parties = build_parties_from_matrices(
            cls.result.vote_shares_matrix, cls.result.seats_matrix
        )

    def _certified_rows(self) -> list[dict]:
        rows = []
        for party in PARTY_DEFINITION_ORDER:
            summary = self.result.summary.parties[party]
            rows.append(
                {
                    "party": party,
                    "vote_share_p05": round(float(summary.vote_share_p05 * 100.0), 3),
                    "vote_share_p25": round(float(summary.vote_share_p25 * 100.0), 3),
                    "vote_share_median": round(float(summary.vote_share_median * 100.0), 3),
                    "vote_share_p75": round(float(summary.vote_share_p75 * 100.0), 3),
                    "vote_share_p95": round(float(summary.vote_share_p95 * 100.0), 3),
                    "seats_p05": int(summary.seats_p05),
                    "seats_p25": int(summary.seats_p25),
                    "seats_median": int(summary.seats_median),
                    "seats_p75": int(summary.seats_p75),
                    "seats_p95": int(summary.seats_p95),
                }
            )
        return rows

    def test_history_party_values_match_the_certified_publication(self) -> None:
        # This is the gate the product requirement names: the election-day party
        # values a reader sees on the chart are the ones the forecast publishes.
        assert_election_day_party_parity(self.parties, self._certified_rows())

    def test_seat_quantiles_are_integer_identical_not_merely_close(self) -> None:
        for party in PARTY_DEFINITION_ORDER:
            summary = self.result.summary.parties[party]
            for key, expected in (
                ("p05", summary.seats_p05),
                ("p25", summary.seats_p25),
                ("p50", summary.seats_median),
                ("p75", summary.seats_p75),
                ("p95", summary.seats_p95),
            ):
                self.assertEqual(self.parties[party]["seats"][key], int(expected), f"{party}.{key}")

    def test_a_drifted_value_fails_closed(self) -> None:
        drifted = deepcopy(self.parties)
        drifted["M"]["vote"]["p50"] += 0.01
        with self.assertRaisesRegex(ValueError, "M election-day vote p50"):
            assert_election_day_party_parity(drifted, self._certified_rows())

    def test_a_drifted_seat_value_fails_closed(self) -> None:
        drifted = deepcopy(self.parties)
        drifted["S"]["seats"]["p95"] += 1
        with self.assertRaisesRegex(ValueError, "S election-day seats p95"):
            assert_election_day_party_parity(drifted, self._certified_rows())


class ArchiveQuantileAgreementTests(unittest.TestCase):
    """The party quantile rule is the archive's rule, not a second one."""

    def test_the_quantile_rule_matches_the_prospective_archive(self) -> None:
        votes, seats = _matrices()
        parties = build_parties_from_matrices(votes, seats)
        for index, party in enumerate(PARTY_DEFINITION_ORDER):
            archived = _histogram(votes[:, index], lower=0.0, upper=100.0, width=0.25)["quantiles"]
            self.assertEqual(
                {key: parties[party]["vote"][key] for key in QUANTILE_KEYS},
                {key: archived[key] for key in QUANTILE_KEYS},
                f"{party} disagrees with the archived quantile rule",
            )

    def test_an_archived_snapshot_recovers_party_marginals(self) -> None:
        # Coalition intervals are joint and cannot be recovered from an archive;
        # party marginals are stored there and can.
        snapshots = sorted(
            (REPO_ROOT / "data" / "processed" / "prospective_forecasts").glob("*/snapshot.json")
        )
        if not snapshots:
            self.skipTest("no prospective archive snapshot is committed")
        record = json.loads(snapshots[-1].read_text(encoding="utf-8"))
        parties = party_point_from_archive_record(record)
        self.assertIsNotNone(parties)
        validate_party_summaries(parties, name="archived.parties")
        for party in PARTY_DEFINITION_ORDER:
            self.assertEqual(
                parties[party]["vote"]["p50"],
                round(float(record["national_vote_distributions"][party]["quantiles"]["p50"]), 6),
            )

    def test_a_legacy_snapshot_without_marginals_yields_no_party_block(self) -> None:
        self.assertIsNone(party_point_from_archive_record({"groups": {}}))
        self.assertIsNone(party_point_from_archive_record({}))

    def test_an_archived_point_carries_parties_when_recoverable(self) -> None:
        snapshots = sorted(
            (REPO_ROOT / "data" / "processed" / "prospective_forecasts").glob("*/snapshot.json")
        )
        if not snapshots:
            self.skipTest("no prospective archive snapshot is committed")
        record = json.loads(snapshots[-1].read_text(encoding="utf-8"))
        point = _archive_point_from_record(
            record,
            election_date=date(2026, 9, 13),
            coalitions={key: list(value) for key, value in DEFAULT_COALITIONS.items()},
        )
        if point is None:
            self.skipTest("the newest snapshot carries no joint coalition groups")
        self.assertIn("parties", point)
        validate_party_summaries(point["parties"], name="archived point")


def _history_fixture(with_parties: bool = True) -> dict:
    votes, seats = _matrices()
    origin = date(2026, 9, 3)
    election = date(2026, 9, 13)

    class _Result:
        vote_shares_matrix = votes
        seats_matrix = seats
        manifest: dict = {}

    point = _point_from_result(
        _Result(),
        point_date=origin,
        election_date=election,
        coalitions={key: list(value) for key, value in DEFAULT_COALITIONS.items()},
        provenance="current_production",
    )
    point["source_git_commit"] = "a" * 40
    if not with_parties:
        point.pop("parties")
    payload = {
        "schema_version": "1.1",
        "election_date": election.isoformat(),
        "model_commit": "b" * 40,
        "poll_source_sha256": "c" * 64,
        "party_order": list(HISTORY_PARTY_ORDER),
        "coalitions": {key: list(value) for key, value in DEFAULT_COALITIONS.items()},
        "series": [point],
        "poll_of_polls": [
            {
                "date": (origin - timedelta(days=1)).isoformat(),
                "parties": {party: 12.5 for party in HISTORY_PARTY_ORDER},
            }
        ],
        "polls": [],
    }
    if with_parties:
        payload["parties_view"] = parties_view_metadata()
    return payload


class PartiesViewValidationTests(unittest.TestCase):
    def test_a_well_formed_payload_validates(self) -> None:
        validate_history_contract(_history_fixture())

    def test_a_payload_without_the_party_family_is_still_valid(self) -> None:
        # Backward compatibility: the artifact the previous website consumed.
        validate_history_contract(_history_fixture(with_parties=False))

    def test_party_data_without_a_declaration_is_rejected(self) -> None:
        payload = _history_fixture()
        payload.pop("parties_view")
        with self.assertRaisesRegex(ValueError, "without a parties_view declaration"):
            validate_parties_view(payload)

    def test_a_declaration_with_no_party_data_is_rejected(self) -> None:
        payload = _history_fixture()
        payload["series"][0].pop("parties")
        with self.assertRaisesRegex(ValueError, "no series point carries party summaries"):
            validate_parties_view(payload)

    def test_a_renormalized_denominator_is_rejected(self) -> None:
        payload = _history_fixture()
        payload["parties_view"]["vote_share_denominator"] = "eight_parliamentary_parties"
        with self.assertRaisesRegex(ValueError, "nine-category denominator"):
            validate_parties_view(payload)

    def test_reconstructing_from_coalitions_is_rejected(self) -> None:
        payload = _history_fixture()
        payload["parties_view"]["election_day_parity"]["reconstructed_from_coalitions"] = True
        with self.assertRaisesRegex(ValueError, "never reconstructed from coalition"):
            validate_parties_view(payload)

    def test_declaring_rest_as_a_party_is_rejected(self) -> None:
        payload = _history_fixture()
        payload["parties_view"]["rest_is_a_party"] = True
        with self.assertRaisesRegex(ValueError, "aggregate vote mass"):
            validate_parties_view(payload)

    def test_declaring_an_intermediate_seat_trajectory_is_rejected(self) -> None:
        payload = _history_fixture()
        payload["parties_view"]["intermediate_seat_trajectory"] = True
        with self.assertRaisesRegex(ValueError, "intermediate seat trajectory"):
            validate_parties_view(payload)

    def test_a_missing_party_is_rejected(self) -> None:
        payload = _history_fixture()
        payload["series"][0]["parties"].pop("MP")
        with self.assertRaisesRegex(ValueError, "eight parliamentary parties in order"):
            validate_parties_view(payload)

    def test_a_certified_point_without_parties_is_rejected(self) -> None:
        payload = _history_fixture()
        payload["series"].insert(
            0,
            {
                "date": "2026-09-02",
                "samples": 10,
                "horizon_days": 11,
                "dynamics_horizon_days": 11,
                "provenance": "reconstructed_current_model",
                "groups": payload["series"][0]["groups"],
                "parties": payload["series"][0]["parties"],
            },
        )
        payload["series"][1].pop("parties")
        with self.assertRaisesRegex(ValueError, "current_production point must carry"):
            validate_parties_view(payload)

    def test_a_seat_quantile_in_an_opinion_band_is_rejected(self) -> None:
        votes, seats = _matrices()
        bands = {
            party: {"vote": build_party_vote_quantiles(votes[:, index])}
            for index, party in enumerate(PARTY_DEFINITION_ORDER)
        }
        validate_party_vote_only(bands, name="bands")
        bands["M"]["seats"] = {key: 10 for key in QUANTILE_KEYS}
        with self.assertRaisesRegex(ValueError, "intermediate future mandate trajectory"):
            validate_party_vote_only(bands, name="bands")


class _StubSimulation:
    """A campaign-path simulation double with a coherent party composition."""

    def __init__(self, origin: date, election: date, samples: int = 64) -> None:
        self.origin_date = origin
        self.election_date = election
        self.path_days = (election - origin).days
        self.samples = samples
        self.seed = 12345
        self.day_dates = tuple(
            origin + timedelta(days=offset) for offset in range(self.path_days + 1)
        )
        rng = np.random.default_rng(3)
        base = np.array([18.0, 2.0, 7.0, 6.5, 30.0, 7.5, 7.5, 19.0, 2.5])
        self.party_draws = {p: np.empty((self.path_days + 1, samples)) for p in PARTY_DEFINITION_ORDER}
        self.coalition_draws = {k: np.empty((self.path_days + 1, samples)) for k in DEFAULT_COALITIONS}
        for day in range(self.path_days + 1):
            raw = np.abs(base + rng.normal(0, 0.3, size=(samples, 9)))
            composition = 100.0 * raw / raw.sum(axis=1, keepdims=True)
            for index, party in enumerate(PARTY_DEFINITION_ORDER):
                self.party_draws[party][day] = composition[:, index]
            for key, members in DEFAULT_COALITIONS.items():
                self.coalition_draws[key][day] = coalition_vote_draws(composition, members)
        self.representative_indices = (0, samples // 2, samples - 1)
        self.diagnostics = {
            "eligible_trajectories": 4000,
            "earliest_trajectory_start": "2014-09-15",
            "latest_trajectory_end": (origin - timedelta(days=1)).isoformat(),
            "endpoint_horizon_days": self.path_days,
            "time_warp": "identity",
            "opinion_state_seed": 1,
            "dynamics_seed": 2,
            "election_noise_seed": 3,
            "endpoint_parity_verified": True,
            "endpoint_parity_max_abs_difference_pp": 0.0,
            "endpoint_parity_reference": "certified_production_result",
        }


def _campaign_history(with_parties: bool = True) -> tuple[dict, dict]:
    payload = _history_fixture(with_parties=with_parties)
    anchor = payload["series"][0]
    origin = date.fromisoformat(anchor["date"])
    election = date.fromisoformat(payload["election_date"])
    built = build_future_campaign_paths(
        origin_date=origin,
        election_date=election,
        anchor_point=anchor,
        samples=64,
        coalitions={key: tuple(value) for key, value in DEFAULT_COALITIONS.items()},
        path_simulator=lambda **_: _StubSimulation(origin, election),
    )
    payload["future_campaign_paths"] = built
    return payload, built


class PartyCampaignPathTests(unittest.TestCase):
    def test_the_party_family_is_published_alongside_the_coalitions(self) -> None:
        payload, built = _campaign_history()
        validate_future_campaign_paths_contract(payload)
        self.assertEqual(len(built["bands"]), built["path_days"] + 1)
        for band in built["bands"]:
            self.assertEqual(list(band["parties"]), list(PARTY_DEFINITION_ORDER))
            validate_party_vote_only(band["parties"], name="band")

    def test_party_keys_never_leak_into_the_coalition_groups(self) -> None:
        # A consumer that validates `groups` against the coalition list exactly
        # must keep working byte for byte. This is the backward-compatibility
        # guarantee the additive design rests on.
        _, built = _campaign_history()
        for band in built["bands"]:
            self.assertEqual(list(band["groups"]), list(DEFAULT_COALITIONS))
        for item in built["paths"]["series"]:
            self.assertEqual(list(item["values"]), list(DEFAULT_COALITIONS))
        self.assertEqual(list(built["election_day"]["groups"]), list(DEFAULT_COALITIONS))

    def test_each_trajectory_carries_one_coherent_composition(self) -> None:
        _, built = _campaign_history()
        for item in built["paths"]["series"]:
            self.assertEqual(list(item["party_values"]), list(PARTY_DEFINITION_ORDER))
            for track in item["party_values"].values():
                self.assertEqual(len(track), built["path_days"] + 1)
            # One draw is one nine-category composition, so the eight
            # parliamentary parties plus REST fill the day; the eight alone
            # must therefore fall short of 100 but stay close to it.
            for day in range(built["path_days"] + 1):
                total = sum(item["party_values"][p][day] for p in PARTY_DEFINITION_ORDER)
                self.assertLess(total, 100.0)
                self.assertGreater(total, 90.0)

    def test_election_day_parties_are_a_verbatim_copy_of_the_certified_point(self) -> None:
        payload, built = _campaign_history()
        anchor = payload["series"][0]
        self.assertEqual(built["election_day"]["parties"], anchor["parties"])
        # A copy, not an alias: mutating one must not move the other.
        built["election_day"]["parties"]["M"]["vote"]["p50"] = 0.0
        self.assertNotEqual(built["election_day"]["parties"], anchor["parties"])

    def test_a_drifted_election_day_party_value_fails_closed(self) -> None:
        payload, built = _campaign_history()
        built["election_day"]["parties"]["S"]["seats"]["p50"] += 1
        with self.assertRaisesRegex(ValueError, "party vote and seat summaries exactly"):
            validate_future_campaign_paths_contract(payload)

    def test_no_intermediate_party_mandate_trajectory_is_published(self) -> None:
        payload, built = _campaign_history()
        self.assertEqual(built["rendering"]["party_units"], ["vote"])
        self.assertFalse(built["rendering"]["party_intermediate_seat_trajectory"])
        for band in built["bands"]:
            for entry in band["parties"].values():
                self.assertEqual(list(entry), ["vote"])
        built["rendering"]["party_intermediate_seat_trajectory"] = True
        with self.assertRaisesRegex(ValueError, "smooth future mandate trajectory"):
            validate_future_campaign_paths_contract(payload)

    def test_a_seat_quantile_in_a_party_band_fails_closed(self) -> None:
        payload, built = _campaign_history()
        built["bands"][1]["parties"]["C"]["seats"] = {key: 20 for key in QUANTILE_KEYS}
        with self.assertRaisesRegex(ValueError, "intermediate future mandate trajectory"):
            validate_future_campaign_paths_contract(payload)

    def test_the_threshold_is_published_for_the_party_view(self) -> None:
        _, built = _campaign_history()
        self.assertEqual(built["rendering"]["national_threshold_pct"], NATIONAL_THRESHOLD_PCT)
        self.assertTrue(built["rendering"]["national_threshold_label_sv"].strip())

    def test_the_denominator_is_declared_in_the_construction(self) -> None:
        _, built = _campaign_history()
        self.assertEqual(
            built["path_construction"]["party_vote_share_denominator"], PARTY_VOTE_DENOMINATOR
        )

    def test_no_party_observation_may_be_dated_after_the_origin(self) -> None:
        payload, built = _campaign_history()
        origin = date.fromisoformat(built["origin_date"])
        self.assertLessEqual(
            date.fromisoformat(built["path_construction"]["latest_trajectory_end"]), origin
        )
        payload["polls"] = [
            {
                "poll_id": "future",
                "company": "Institut",
                "publication_date": (origin + timedelta(days=2)).isoformat(),
                "fieldwork_start": None,
                "fieldwork_end": None,
                "n": 1000,
                "parties": {party: 12.5 for party in HISTORY_PARTY_ORDER},
            }
        ]
        with self.assertRaisesRegex(ValueError, "after the campaign-path origin"):
            validate_future_campaign_paths_contract(payload)

    def test_bands_cover_every_day_from_the_origin(self) -> None:
        _, built = _campaign_history()
        origin = date.fromisoformat(built["origin_date"])
        for index, band in enumerate(built["bands"]):
            self.assertEqual(band["date"], (origin + timedelta(days=index)).isoformat())
            self.assertEqual(band["path_day"], index)
        self.assertEqual(built["bands"][0]["date"], built["origin_date"])
        self.assertEqual(built["bands"][-1]["date"], built["election_date"])

    def test_the_party_family_is_omitted_without_a_certified_party_point(self) -> None:
        # Half a family would be worse than none: opinion bands with no
        # certified endpoint to meet.
        payload, built = _campaign_history(with_parties=False)
        self.assertNotIn("parties", built["election_day"])
        self.assertNotIn("party_units", built["rendering"])
        for band in built["bands"]:
            self.assertNotIn("parties", band)
        for item in built["paths"]["series"]:
            self.assertNotIn("party_values", item)
        validate_future_campaign_paths_contract(payload)

    def test_a_partial_party_family_fails_closed(self) -> None:
        payload, built = _campaign_history()
        built["bands"][2].pop("parties")
        with self.assertRaisesRegex(ValueError, "missing its party opinion bands"):
            validate_future_campaign_paths_contract(payload)

    def test_party_bands_without_a_certified_endpoint_fail_closed(self) -> None:
        payload, built = _campaign_history(with_parties=False)
        built["bands"][0]["parties"] = {
            party: {"vote": {key: 5.0 for key in QUANTILE_KEYS}}
            for party in PARTY_DEFINITION_ORDER
        }
        with self.assertRaisesRegex(ValueError, "certified party election-day"):
            validate_future_campaign_paths_contract(payload)

    def test_building_is_deterministic(self) -> None:
        _, first = _campaign_history()
        _, second = _campaign_history()
        self.assertEqual(
            json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True)
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
