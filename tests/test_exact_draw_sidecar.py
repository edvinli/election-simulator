"""Tests for certified ElectionSimulator draw sidecars and cutoff selection."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from scripts.simulator.engine import simulate_election
from scripts.simulator.exact_draw_sidecar import (
    ExactDrawSidecarError,
    _array_digest,
    _discover_verified_sidecar,
    _npz_bytes,
    build_exact_draw_sidecar_files,
    collect_latest_certified_generation,
    load_verified_draw_sidecar,
    replay_certified_generation,
    write_exact_draw_sidecar,
)
from scripts.simulator.pipeline import build_canonical_summary_dict


class ExactDrawSidecarTests(unittest.TestCase):
    @staticmethod
    def _result():
        result = simulate_election(
            as_of="2026-09-03",
            election_date="2026-09-13",
            samples=4,
            seed=12345,
        )
        # This test checkout contains the test file itself as an uncommitted
        # file while unittest is running.  The production boundary records a
        # clean committed source; set that field explicitly for this isolated
        # result fixture.
        result.manifest["source_worktree_clean"] = True
        return result

    def _snapshot(self, result, generation: str) -> dict:
        payload = build_canonical_summary_dict(result)["deterministic_payload_sha256"]
        return {
            "snapshot_id": "s" * 64,
            "generation_id": generation,
            "as_of": result.manifest["as_of"],
            "election_date": result.manifest["election_date"],
            "generated_at_utc": "2026-09-03T21:20:00Z",
            "model": {"name": "ElectionSimulator", "version": result.manifest["model_version"]},
            "source_git_commit": result.manifest["source_git_commit"],
            "source_worktree_clean": True,
            "samples": int(result.manifest["samples"]),
            "deterministic_payload_sha256": payload,
        }

    def test_sidecar_round_trip_is_tied_to_payload_and_generation(self) -> None:
        result = self._result()
        generation = "20260903T212000Z-testgen"
        snapshot = self._snapshot(result, generation)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = write_exact_draw_sidecar(
                result,
                root,
                generation_id=generation,
                certified_snapshot=snapshot,
            )
            self.assertEqual(first["status"], "WRITTEN_VERIFIED")
            loaded = load_verified_draw_sidecar(
                root / "draws.npz",
                root / "draws.json",
                expected_generation_id=generation,
                expected_payload_hash=snapshot["deterministic_payload_sha256"],
            )
            np.testing.assert_array_equal(loaded["vote_shares_pct"], result.vote_shares_matrix)
            np.testing.assert_array_equal(loaded["seats"], result.seats_matrix)
            second = write_exact_draw_sidecar(
                result,
                root,
                generation_id=generation,
                certified_snapshot=snapshot,
            )
            self.assertEqual(second["status"], "ALREADY_PRESENT_VERIFIED")

    def test_sidecar_refuses_partial_or_different_existing_evidence(self) -> None:
        result = self._result()
        generation = "20260903T212000Z-testgen"
        snapshot = self._snapshot(result, generation)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "draws.npz").write_bytes(b"partial")
            with self.assertRaises(FileExistsError):
                write_exact_draw_sidecar(
                    result,
                    root,
                    generation_id=generation,
                    certified_snapshot=snapshot,
                )

    def test_sidecar_files_can_be_passed_to_an_atomic_capture_materializer(self) -> None:
        result = self._result()
        files = build_exact_draw_sidecar_files(
            result,
            generation_id="20260903T212000Z-testgen",
            certified_snapshot=self._snapshot(result, "20260903T212000Z-testgen"),
        )
        self.assertEqual(set(files), {"draws.npz", "draws.json"})
        self.assertGreater(len(files["draws.npz"]), 0)
        metadata = json.loads(files["draws.json"])
        self.assertEqual(metadata["vote_party_order"], ["M", "L", "C", "KD", "S", "V", "MP", "SD", "REST"])
        self.assertEqual(metadata["samples"], 4)

    def test_sidecar_rejects_generation_or_payload_mismatch(self) -> None:
        result = self._result()
        generation = "20260903T212000Z-testgen"
        snapshot = self._snapshot(result, generation)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ExactDrawSidecarError, "generation mismatch"):
                write_exact_draw_sidecar(
                    result,
                    root,
                    generation_id="other-generation",
                    certified_snapshot=snapshot,
                )

    def test_discovery_rejects_self_consistent_but_certified_summary_tampering(self) -> None:
        result = self._result()
        generation = "20260903T212000Z-testgen"
        snapshot = self._snapshot(result, generation)
        canonical = build_canonical_summary_dict(result)
        snapshot.update(
            {
                "seed": int(result.manifest["base_seed"]),
                "national_vote_summary": canonical["parties"],
                "group_probabilities": canonical["blocs"],
                "threshold_probabilities_4pct": {
                    party: canonical["parties"][party]["prob_above_4pct"]
                    for party in ("M", "L", "C", "KD", "S", "V", "MP", "SD")
                },
                "seat_summary": {
                    party: {
                        "mean": canonical["parties"][party]["seats_mean"],
                        "median": canonical["parties"][party]["seats_median"],
                        "p05": canonical["parties"][party]["seats_p05"],
                        "p95": canonical["parties"][party]["seats_p95"],
                    }
                    for party in ("M", "L", "C", "KD", "S", "V", "MP", "SD")
                },
            }
        )
        files = build_exact_draw_sidecar_files(
            result,
            generation_id=generation,
            certified_snapshot=snapshot,
        )
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp)
            generation_dir = archive / generation
            generation_dir.mkdir()
            metadata = json.loads(files["draws.json"])
            with np.load(io.BytesIO(files["draws.npz"]), allow_pickle=False) as loaded:
                votes = np.asarray(loaded["vote_shares_pct"]).copy()
                seats = np.asarray(loaded["seats"]).copy()
            # Keep the sidecar internally self-consistent by updating its
            # array and file hashes, while preserving the selected snapshot's
            # payload hash.  Only the independent compact-summary commitment
            # should reject this tampering.
            votes[0, 0] += 0.5
            votes[0, 8] -= 0.5
            tampered_npz = _npz_bytes(votes, seats)
            metadata["draws_file_sha256"] = hashlib.sha256(tampered_npz).hexdigest()
            metadata["arrays"]["vote_shares_pct"]["sha256"] = _array_digest(votes)
            (generation_dir / "draws.npz").write_bytes(tampered_npz)
            (generation_dir / "draws.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            discovered = _discover_verified_sidecar(
                archive_root=archive,
                snapshot=snapshot,
                provenance={},
                include_draws=False,
                include_sidecar_bytes=False,
            )
        self.assertEqual(discovered["status"], "UNVERIFIED_SIDECAR")
        self.assertIn("summary parity failed", discovered["reason"])


class CertifiedGenerationSelectionTests(unittest.TestCase):
    @staticmethod
    def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
        merged = {"GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@example.invalid", "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@example.invalid"}
        if env:
            merged.update(env)
        return subprocess.check_output(["git", *args], cwd=repo, env={**__import__("os").environ, **merged}, text=True).strip()

    def _make_repo(self, *, archive_commit_time: str) -> tuple[Path, str, dict]:
        root = Path(tempfile.mkdtemp(prefix="exact-draw-selection-"))
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        self._git(root, "init", "-q")
        (root / "README").write_text("source\n", encoding="utf-8")
        source_commit = self._git(
            root,
            "add", "README",
        )
        self._git(root, "commit", "-qm", "source", env={"GIT_AUTHOR_DATE": "2026-09-03T20:00:00Z", "GIT_COMMITTER_DATE": "2026-09-03T20:00:00Z"})
        source_commit = self._git(root, "rev-parse", "HEAD")
        generation = "20260903T210000Z-testgen"
        archive = root / "data" / "processed" / "prospective_forecasts"
        generation_dir = archive / generation
        generation_dir.mkdir(parents=True)
        snapshot = {
            "schema_version": "1.2",
            "snapshot_id": "s" * 64,
            "information_set_id": "i" * 64,
            "duplicate_payload_allowed": False,
            "generation_id": generation,
            "snapshot_date": "2026-09-03",
            "generated_at_utc": "2026-09-03T21:00:00Z",
            "as_of": "2026-09-03",
            "election_date": "2026-09-13",
            "model": {"name": "ElectionSimulator", "version": "1.1.0-rc1"},
            "source_git_commit": source_commit,
            "source_worktree_clean": True,
            "deterministic_payload_sha256": "a" * 64,
        }
        snapshot_path = generation_dir / "snapshot.json"
        snapshot_path.write_text(json.dumps(snapshot) + "\n", encoding="utf-8")
        entry = {
            "snapshot_id": snapshot["snapshot_id"],
            "generation_id": generation,
            "snapshot_date": snapshot["snapshot_date"],
            "as_of": snapshot["as_of"],
            "election_date": snapshot["election_date"],
            "generated_at_utc": snapshot["generated_at_utc"],
            "source_git_commit": source_commit,
            "model_version": snapshot["model"]["version"],
            "seed": 12345,
            "deterministic_payload_sha256": snapshot["deterministic_payload_sha256"],
            "duplicate_payload_allowed": False,
            "snapshot_file_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
            "path": f"{generation}/snapshot.json",
        }
        (archive / "index.json").write_text(json.dumps({"schema_version": "1.2", "archive": "ElectionSimulator prospective forecasts", "snapshots": [entry]}) + "\n", encoding="utf-8")
        self._git(root, "add", "data")
        self._git(root, "commit", "-qm", "archive", env={"GIT_AUTHOR_DATE": archive_commit_time, "GIT_COMMITTER_DATE": archive_commit_time})
        return root, generation, snapshot

    def test_archive_commit_must_exist_by_cutoff(self) -> None:
        root, _generation, _snapshot = self._make_repo(archive_commit_time="2026-09-03T21:30:00Z")
        selected = collect_latest_certified_generation(
            root / "data" / "processed" / "prospective_forecasts",
            "2026-09-03T21:00:00Z",
            root,
        )
        self.assertEqual(selected["status"], "NO_CERTIFIED_GENERATION")
        selected = collect_latest_certified_generation(
            root / "data" / "processed" / "prospective_forecasts",
            "2026-09-03T22:00:00Z",
            root,
        )
        self.assertEqual(selected["status"], "FOUND_NO_VERIFIED_DRAWS")
        self.assertEqual(
            selected["provenance"]["first_archive_commit"],
            selected["provenance"]["first_index_commit"],
        )
        self.assertEqual(selected["provenance"]["source_commit_resolved"], True)

    def test_replay_rejects_later_committed_simulator_code(self) -> None:
        root, _generation, snapshot = self._make_repo(
            archive_commit_time="2026-09-03T21:30:00Z"
        )
        changed = root / "scripts" / "changed.py"
        changed.parent.mkdir()
        changed.write_text("changed = True\n", encoding="utf-8")
        self._git(root, "add", "scripts/changed.py")
        self._git(
            root,
            "commit",
            "-qm",
            "later simulator code",
            env={
                "GIT_AUTHOR_DATE": "2026-09-03T22:00:00Z",
                "GIT_COMMITTER_DATE": "2026-09-03T22:00:00Z",
            },
        )
        with self.assertRaisesRegex(ExactDrawSidecarError, "code or model inputs"):
            replay_certified_generation(snapshot, root)

    def test_same_day_generations_are_selected_by_cutoff_not_fitness(self) -> None:
        root, first_generation, first_snapshot = self._make_repo(
            archive_commit_time="2026-09-03T20:30:00Z"
        )
        archive = root / "data" / "processed" / "prospective_forecasts"
        second_generation = "20260903T220000Z-second"
        second_snapshot = dict(first_snapshot)
        second_snapshot.update(
            {
                "snapshot_id": "t" * 64,
                "generation_id": second_generation,
                "generated_at_utc": "2026-09-03T22:00:00Z",
                "deterministic_payload_sha256": "b" * 64,
            }
        )
        second_dir = archive / second_generation
        second_dir.mkdir()
        second_path = second_dir / "snapshot.json"
        second_path.write_text(json.dumps(second_snapshot) + "\n", encoding="utf-8")
        index_path = archive / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["snapshots"].append(
            {
                "snapshot_id": second_snapshot["snapshot_id"],
                "generation_id": second_generation,
                "snapshot_date": second_snapshot["snapshot_date"],
                "as_of": second_snapshot["as_of"],
                "election_date": second_snapshot["election_date"],
                "generated_at_utc": second_snapshot["generated_at_utc"],
                "source_git_commit": second_snapshot["source_git_commit"],
                "model_version": second_snapshot["model"]["version"],
                "seed": 12345,
                "deterministic_payload_sha256": second_snapshot["deterministic_payload_sha256"],
                "duplicate_payload_allowed": False,
                "snapshot_file_sha256": hashlib.sha256(second_path.read_bytes()).hexdigest(),
                "path": f"{second_generation}/snapshot.json",
            }
        )
        index_path.write_text(json.dumps(index) + "\n", encoding="utf-8")
        self._git(root, "add", "data")
        self._git(
            root,
            "commit",
            "-qm",
            "second same-day generation",
            env={
                "GIT_AUTHOR_DATE": "2026-09-03T22:30:00Z",
                "GIT_COMMITTER_DATE": "2026-09-03T22:30:00Z",
            },
        )
        early = collect_latest_certified_generation(
            archive,
            "2026-09-03T21:30:00Z",
            root,
        )
        self.assertEqual(early["forecast"]["generation_id"], first_generation)
        late = collect_latest_certified_generation(archive, "2026-09-03T23:00:00Z", root)
        self.assertEqual(late["forecast"]["generation_id"], second_generation)


if __name__ == "__main__":
    unittest.main()
