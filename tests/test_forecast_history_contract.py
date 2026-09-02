"""Contract and fail-closed validation tests for history schema 1.0."""

from __future__ import annotations

from copy import deepcopy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from scripts.forecast_history.contract import (
    DEFAULT_COALITIONS,
    deterministic_history_sha256,
    validate_history_contract,
    write_history_json,
)
from scripts.forecast_history.generate import build_history


class ForecastHistoryContractTests(unittest.TestCase):
    @staticmethod
    def _payload() -> dict:
        votes = np.tile(np.array([[20, 5, 10, 5, 30, 10, 8, 12, 0]], dtype=float), (3, 1))
        seats = np.tile(np.array([[40, 20, 30, 20, 80, 50, 30, 79]], dtype=np.int64), (3, 1))

        def runner(*, as_of: str, election_date: str, samples: int, seed: int):
            return SimpleNamespace(
                vote_shares_matrix=votes,
                seats_matrix=seats,
                manifest={"source_git_commit": "e" * 40},
            )

        with tempfile.TemporaryDirectory() as temporary:
            poll_path = Path(temporary) / "polls.csv"
            poll_path.write_text(
                "poll_id,pollster,publication_date,interview_start,interview_end,party,support\n"
                + "\n".join(
                    f"p,Test,2026-05-24,2026-05-20,2026-05-23,{party},{value}"
                    for party, value in zip(
                        ("M", "L", "C", "KD", "S", "V", "MP", "SD"),
                        (20, 5, 10, 5, 30, 10, 8, 12),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            return build_history(
                # Hermetic: the default archive_dir is the repository's
                # prospective-forecast archive, and build_history folds every
                # date it finds there into the observation dates. This test
                # pins a single date and its own runner, so reading production
                # archives would only make it fail as the archive grows.
                archive_dir=None,
                dates=["2026-05-24"],
                poll_file=poll_path,
                simulation_runner=runner,
                model_commit="e" * 40,
                source_worktree_clean=True,
                production_latest_samples=3,
            )

    def test_valid_payload_and_deterministic_hash(self) -> None:
        payload = self._payload()
        validate_history_contract(payload)
        self.assertEqual(payload["deterministic_content_sha256"], deterministic_history_sha256(payload))

    def test_write_is_compact_and_revalidates(self) -> None:
        payload = self._payload()
        with tempfile.TemporaryDirectory() as temporary:
            destination = write_history_json(Path(temporary) / "history.json", payload)
            content = destination.read_text(encoding="utf-8")
            self.assertLess(len(content), len(__import__("json").dumps(payload, indent=2)) + 1)
            self.assertTrue(content.endswith("\n"))

    def test_rejects_non_commit_and_non_sha_provenance(self) -> None:
        payload = self._payload()
        invalid_commit = deepcopy(payload)
        invalid_commit["model_commit"] = "not-a-commit"
        with self.assertRaises(ValueError):
            validate_history_contract(invalid_commit)
        invalid_hash = deepcopy(payload)
        invalid_hash["poll_source_sha256"] = "not-a-hash"
        with self.assertRaises(ValueError):
            validate_history_contract(invalid_hash)

    def test_rejects_bad_horizon_and_duplicate_point_identity(self) -> None:
        payload = self._payload()
        bad_horizon = deepcopy(payload)
        bad_horizon["series"][0]["dynamics_horizon_days"] = 113
        bad_horizon.pop("deterministic_content_sha256", None)
        with self.assertRaises(ValueError):
            validate_history_contract(bad_horizon)

        duplicate = deepcopy(payload)
        duplicate["series"].append(deepcopy(duplicate["series"][0]))
        duplicate.pop("deterministic_content_sha256", None)
        with self.assertRaises(ValueError):
            validate_history_contract(duplicate)

    def test_accepts_all_valid_provenance_types(self) -> None:
        payload = self._payload()
        for prov in ("reconstructed_current_model", "prospective_archived", "current_production"):
            test_payload = deepcopy(payload)
            test_payload["series"][0]["provenance"] = prov
            test_payload["deterministic_content_sha256"] = deterministic_history_sha256(test_payload)
            validate_history_contract(test_payload)

    def test_rejects_unknown_provenance(self) -> None:
        payload = self._payload()
        invalid = deepcopy(payload)
        invalid["series"][0]["provenance"] = "fabricated_archive"
        invalid.pop("deterministic_content_sha256", None)
        with self.assertRaises(ValueError):
            validate_history_contract(invalid)

    def test_rejects_other_category_in_poll(self) -> None:
        payload = self._payload()
        invalid = deepcopy(payload)
        invalid["polls"][0]["parties"]["Other"] = 1
        invalid.pop("deterministic_content_sha256", None)
        with self.assertRaises(ValueError):
            validate_history_contract(invalid)

    def test_rejects_non_monotone_quantiles(self) -> None:
        payload = self._payload()
        invalid = deepcopy(payload)
        vote = invalid["series"][0]["groups"]["tido"]["vote"]
        vote["p50"], vote["p75"] = 99.0, 1.0
        invalid.pop("deterministic_content_sha256", None)
        with self.assertRaises(ValueError):
            validate_history_contract(invalid)

    def test_rejects_missing_or_invalid_poll_of_polls(self) -> None:
        payload = self._payload()
        invalid_missing = deepcopy(payload)
        invalid_missing.pop("poll_of_polls", None)
        with self.assertRaises(ValueError):
            validate_history_contract(invalid_missing)

        invalid_empty = deepcopy(payload)
        invalid_empty["poll_of_polls"] = []
        with self.assertRaises(ValueError):
            validate_history_contract(invalid_empty)

    def test_rejects_unsorted_or_duplicate_pop_dates(self) -> None:
        payload = self._payload()
        invalid_duplicate = deepcopy(payload)
        invalid_duplicate["poll_of_polls"].append(deepcopy(invalid_duplicate["poll_of_polls"][0]))
        with self.assertRaises(ValueError):
            validate_history_contract(invalid_duplicate)

    def test_rejects_invalid_pop_parties_or_negative_denominator(self) -> None:
        payload = self._payload()
        invalid_party = deepcopy(payload)
        invalid_party["poll_of_polls"][0]["parties"]["Other"] = 1.0
        with self.assertRaises(ValueError):
            validate_history_contract(invalid_party)

        invalid_negative = deepcopy(payload)
        invalid_negative["poll_of_polls"][0]["parties"]["M"] = -5.0
        with self.assertRaises(ValueError):
            validate_history_contract(invalid_negative)

        invalid_zero = deepcopy(payload)
        for p in invalid_zero["poll_of_polls"][0]["parties"]:
            invalid_zero["poll_of_polls"][0]["parties"][p] = 0.0
        with self.assertRaises(ValueError):
            validate_history_contract(invalid_zero)


if __name__ == "__main__":
    unittest.main()
