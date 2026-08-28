"""The deterministic payload hash must track science, not provenance.

``compute_simulation_payload_sha256`` is the reproducibility identity of a
simulation: two runs that produced the same matrices from the same inputs must
carry the same hash, and any change to the matrices or to a real simulation
input must change it.

Git commit identifiers are provenance — they say which code produced a run, not
what the run consumed.  They were being folded into the digest, so a code-only
commit (documentation, comments, an unrelated module) moved the payload
identity of a bit-identical simulation.  That made the freeze audit's
``payload_equal`` check a false-alarm generator and would have let the same
forecast past the prospective archive's payload dedupe as a "new" snapshot.

These tests use tiny fixed matrices; no simulation is run.
"""

from __future__ import annotations

import copy
from typing import Any
import unittest

import numpy as np

from scripts.simulator.reproducibility import compute_simulation_payload_sha256

COMMIT_A = "f6ae4d13eb36a156fd56b5cea52767ff8b856923"
COMMIT_B = "577ebbf9e0997d66c193d9b4adfde633ecc0329a"


def _matrices() -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic stand-ins for the national and seat draws."""
    rng = np.random.default_rng(20260828)
    return rng.random((12, 9)), rng.integers(0, 60, size=(12, 9))


def _summary(
    *,
    commit: str = COMMIT_A,
    generated_at_utc: str = "2026-08-27T20:58:28.998301+00:00",
    worktree_clean: bool = True,
    poll_data_hash: str = "a" * 64,
    model_config_hash: str = "e" * 64,
) -> dict[str, Any]:
    """A canonical summary dict shaped like the one the pipeline hashes."""
    return {
        "as_of": "2026-08-24",
        "election_date": "2026-09-13",
        "total_samples": 12,
        "parties": {"M": {"vote_share_median": 18.6, "seats_median": 68}},
        "blocs": {"tido": {"parties": ["M", "SD", "KD", "L"], "prob_majority": 0.00286}},
        "manifest": {
            "model_version": "1.0.0-rc1",
            "as_of": "2026-08-24",
            "election_date": "2026-09-13",
            "samples": 12,
            "base_seed": 12345,
            "poll_data_hash": poll_data_hash,
            "election_data_hash": "b" * 64,
            "mandate_data_hash": "c" * 64,
            "geography_data_hash": "d" * 64,
            "model_config_hash": model_config_hash,
            "source_git_commit": commit,
            "git_commit": commit,
            "source_worktree_clean": worktree_clean,
            "generated_at_utc": generated_at_utc,
        },
    }


class PayloadHashIgnoresProvenance(unittest.TestCase):
    """Provenance-only differences must not move the payload identity."""

    def test_git_commit_change_alone_keeps_the_payload_hash(self) -> None:
        national, seats = _matrices()
        self.assertEqual(
            compute_simulation_payload_sha256(national, seats, _summary(commit=COMMIT_A)),
            compute_simulation_payload_sha256(national, seats, _summary(commit=COMMIT_B)),
            "a code-only commit must not change the deterministic payload identity",
        )

    def test_legacy_git_commit_alias_alone_keeps_the_payload_hash(self) -> None:
        """The v1 manifest alias ``git_commit`` must be excluded too."""
        national, seats = _matrices()
        baseline = _summary(commit=COMMIT_A)
        aliased = copy.deepcopy(baseline)
        aliased["manifest"]["git_commit"] = COMMIT_B
        self.assertEqual(
            compute_simulation_payload_sha256(national, seats, baseline),
            compute_simulation_payload_sha256(national, seats, aliased),
        )

    def test_generated_at_utc_change_keeps_the_payload_hash(self) -> None:
        national, seats = _matrices()
        self.assertEqual(
            compute_simulation_payload_sha256(national, seats, _summary()),
            compute_simulation_payload_sha256(
                national, seats, _summary(generated_at_utc="2199-01-01T00:00:00+00:00")
            ),
        )

    def test_worktree_clean_flag_keeps_the_payload_hash(self) -> None:
        national, seats = _matrices()
        self.assertEqual(
            compute_simulation_payload_sha256(national, seats, _summary()),
            compute_simulation_payload_sha256(national, seats, _summary(worktree_clean=False)),
        )

    def test_git_commit_is_only_dropped_from_the_hashed_copy(self) -> None:
        """The caller's summary dict must keep its provenance untouched."""
        national, seats = _matrices()
        summary = _summary(commit=COMMIT_A)
        compute_simulation_payload_sha256(national, seats, summary)
        self.assertEqual(summary["manifest"]["source_git_commit"], COMMIT_A)
        self.assertEqual(summary["manifest"]["git_commit"], COMMIT_A)
        self.assertTrue(summary["manifest"]["source_worktree_clean"])


class PayloadHashTracksScience(unittest.TestCase):
    """Anything the simulation actually consumed or produced must move it."""

    def test_one_changed_seat_changes_the_payload_hash(self) -> None:
        national, seats = _matrices()
        perturbed = seats.copy()
        perturbed[0, 0] += 1
        self.assertNotEqual(
            compute_simulation_payload_sha256(national, seats, _summary()),
            compute_simulation_payload_sha256(national, perturbed, _summary()),
        )

    def test_one_changed_vote_share_draw_changes_the_payload_hash(self) -> None:
        national, seats = _matrices()
        perturbed = national.copy()
        perturbed[0, 0] = np.nextafter(perturbed[0, 0], 1.0)
        self.assertNotEqual(
            compute_simulation_payload_sha256(national, seats, _summary()),
            compute_simulation_payload_sha256(perturbed, seats, _summary()),
        )

    def test_changed_input_data_hash_changes_the_payload_hash(self) -> None:
        """poll_data_hash pins a real simulation input, not provenance."""
        national, seats = _matrices()
        self.assertNotEqual(
            compute_simulation_payload_sha256(national, seats, _summary()),
            compute_simulation_payload_sha256(
                national, seats, _summary(poll_data_hash="f" * 64)
            ),
        )

    def test_changed_model_config_hash_changes_the_payload_hash(self) -> None:
        national, seats = _matrices()
        self.assertNotEqual(
            compute_simulation_payload_sha256(national, seats, _summary()),
            compute_simulation_payload_sha256(
                national, seats, _summary(model_config_hash="9" * 64)
            ),
        )

    def test_changed_scientific_summary_field_changes_the_payload_hash(self) -> None:
        national, seats = _matrices()
        moved = _summary()
        moved["parties"]["M"]["seats_median"] = 69
        self.assertNotEqual(
            compute_simulation_payload_sha256(national, seats, _summary()),
            compute_simulation_payload_sha256(national, seats, moved),
        )


if __name__ == "__main__":
    unittest.main()
