"""REFERENCE_CONTRACT_TEST — portable contract check, not the real consumer.

A publication produced by ``scripts.static_exporter`` is handed to an
independent Node *reimplementation* of the website's acceptance rules
(``scripts/static_exporter/contract/reference_publication_validator.js``).

This is deliberately NOT the production consumer. It exists because it is
dependency-free and runs anywhere, including CI machines with no website
checkout. Two independently maintained validators can drift, so this module
alone does not close the exporter/browser contract gap.

The gap is closed by ``tests/test_actual_browser_consumer.py``
(ACTUAL_BROWSER_CONSUMER_TEST), which evaluates the deployed website source
byte for byte. ``ReferenceValidatorDriftTests`` below is the tripwire that
keeps this reference aligned with production in between.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from scripts.simulator.engine import simulate_election
from scripts.static_exporter import export_static_data

from ._website_repo import website_consumer_path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CLI = REPOSITORY_ROOT / "scripts" / "static_exporter" / "contract" / "reference_validate_cli.js"
LEGACY_FLAT_FIXTURE = REPOSITORY_ROOT / "tests" / "fixtures" / "legacy_flat_publication_2026_08_27"
WEBSITE_CONSUMER = website_consumer_path()

NODE = shutil.which("node")


def run_reference_validator(publication_dir: Path) -> dict:
    """Run the REFERENCE validator under Node and return its verdict."""

    completed = subprocess.run(
        [NODE, str(REFERENCE_CLI), str(publication_dir)],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
    )
    if not completed.stdout.strip():
        raise AssertionError(f"Reference validator produced no verdict: {completed.stderr}")
    return json.loads(completed.stdout)


@unittest.skipIf(NODE is None, "Node is required for the reference publication contract test")
class ReferencePublicationContractTests(unittest.TestCase):
    """The exporter's output must satisfy the reference acceptance rules."""

    @classmethod
    def setUpClass(cls) -> None:
        result = simulate_election(as_of="2026-08-23", election_date="2026-09-13", samples=8, seed=12345)
        # The working tree carries unrelated research artifacts during
        # development; certified publication requires the boolean true.
        result.manifest["source_worktree_clean"] = True
        cls.result = result
        cls._tmp = tempfile.TemporaryDirectory()
        cls.publication = Path(cls._tmp.name) / "publication"
        cls.manifest = export_static_data(
            result,
            output_dir=cls.publication,
            generated_at_utc="2026-08-27T00:00:00+00:00",
            calibration_dir=REPOSITORY_ROOT / "data" / "processed",
        )
        cls.version = cls.publication / "versions" / cls.manifest["publication_generation"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _mutated_copy(self, mutate) -> Path:
        """Copy the certified publication into a scratch dir and corrupt it."""

        target = Path(self.enterContext(tempfile.TemporaryDirectory())) / "publication"
        shutil.copytree(self.publication, target)
        mutate(target, target / "versions" / self.manifest["publication_generation"])
        return target

    # ---- happy path -----------------------------------------------------

    def test_exporter_output_is_accepted_by_the_reference_validator(self) -> None:
        verdict = run_reference_validator(self.publication)
        self.assertTrue(verdict["accepted"], verdict)
        self.assertEqual(verdict["mode"], "pointer")
        self.assertEqual(verdict["generation"], self.manifest["publication_generation"])
        self.assertTrue(verdict["certified"])
        self.assertEqual(verdict["status_text"], "Certified forecast loaded.")
        self.assertEqual(verdict["seat_allocation_source"], "representative_joint_simulation_draw")
        self.assertEqual(verdict["seat_total"], 349)
        self.assertEqual(verdict["schema_version"], "1.1")
        self.assertEqual(verdict["source_repository"], "edvinli/election-simulator")

    def test_published_version_contains_seven_real_files_and_no_symlinks(self) -> None:
        self.assertEqual(
            {path.name for path in self.version.iterdir()},
            {
                "forecast.json",
                "parties.json",
                "seats.json",
                "groups.json",
                "calibration.json",
                "metadata.json",
                "manifest.json",
            },
        )
        for path in self.version.iterdir():
            self.assertFalse(path.is_symlink(), f"{path} must be a real file")
            self.assertTrue(path.is_file())
        # No flat aliases are created any more; only the pointer and the store.
        self.assertEqual(
            {path.name for path in self.publication.iterdir()},
            {"current.json", "versions"},
        )
        for path in self.publication.rglob("*"):
            self.assertFalse(path.is_symlink(), f"{path} must not be a symlink")

    def test_pointer_is_written_last(self) -> None:
        pointer = self.publication / "current.json"
        pointer_written = pointer.stat().st_mtime_ns
        for path in self.version.iterdir():
            self.assertLessEqual(
                path.stat().st_mtime_ns,
                pointer_written,
                f"{path.name} must be durable before current.json is written",
            )
        # A pointer with no complete version behind it must never be accepted.
        orphan = self._mutated_copy(lambda root, version: shutil.rmtree(version))
        verdict = run_reference_validator(orphan)
        self.assertFalse(verdict["accepted"])

    # ---- pointer rejection ----------------------------------------------

    def _reject_pointer(self, mutate_pointer, expected: str) -> None:
        def mutate(root: Path, version: Path) -> None:
            pointer = json.loads((root / "current.json").read_text(encoding="utf-8"))
            mutate_pointer(pointer)
            (root / "current.json").write_text(json.dumps(pointer, indent=2) + "\n", encoding="utf-8")

        verdict = run_reference_validator(self._mutated_copy(mutate))
        self.assertFalse(verdict["accepted"], verdict)
        self.assertIn(expected, verdict["error"])

    def test_malformed_pointer_is_rejected_and_never_falls_back(self) -> None:
        cases = [
            ("incomplete state", lambda p: p.__setitem__("publication_state", "PENDING")),
            ("path not under versions", lambda p: p.__setitem__("path", p["publication_generation"])),
            ("path traversal", lambda p: p.__setitem__("path", "versions/../versions")),
            ("generation mismatch", lambda p: p.__setitem__("path", "versions/other-generation")),
            ("missing manifest hash", lambda p: p.pop("manifest_sha256")),
            ("non-string generation", lambda p: p.__setitem__("publication_generation", 1)),
        ]
        for label, mutate_pointer in cases:
            with self.subTest(case=label):
                self._reject_pointer(mutate_pointer, "Current publication pointer is invalid")

    def test_manifest_byte_tampering_is_rejected_by_the_pointer_hash(self) -> None:
        def mutate(root: Path, version: Path) -> None:
            manifest_path = version / "manifest.json"
            # A single whitespace byte: identical parsed JSON, different bytes.
            manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

        verdict = run_reference_validator(self._mutated_copy(mutate))
        self.assertFalse(verdict["accepted"])
        self.assertIn("pointer hash does not match", verdict["error"])

    def test_pointer_falls_back_only_on_a_literal_404(self) -> None:
        # A publication with no pointer at all is a legacy publication.
        def remove_pointer(root: Path, version: Path) -> None:
            (root / "current.json").unlink()
            for name in (
                "forecast.json",
                "parties.json",
                "seats.json",
                "groups.json",
                "calibration.json",
                "metadata.json",
                "manifest.json",
            ):
                shutil.copyfile(version / name, root / name)

        verdict = run_reference_validator(self._mutated_copy(remove_pointer))
        self.assertTrue(verdict["accepted"], verdict)
        self.assertEqual(verdict["mode"], "legacy_flat_fallback")

        # An unreadable-but-present pointer is a hard error, never a fallback.
        def corrupt_pointer(root: Path, version: Path) -> None:
            (root / "current.json").write_text("{ not json", encoding="utf-8")
            for name in ("forecast.json", "manifest.json"):
                shutil.copyfile(version / name, root / name)

        verdict = run_reference_validator(self._mutated_copy(corrupt_pointer))
        self.assertFalse(verdict["accepted"])
        self.assertIn("current.json", verdict["error"])

    # ---- payload contract rejection -------------------------------------

    def test_representative_allocation_must_total_349(self) -> None:
        for label, delta in (("348 seats", -1), ("350 seats", 1)):
            with self.subTest(case=label):

                def mutate(root: Path, version: Path, delta=delta) -> None:
                    seats_path = version / "seats.json"
                    seats = json.loads(seats_path.read_text(encoding="utf-8"))
                    seats["representative_allocation"]["seats"]["M"] += delta
                    seats_path.write_text(json.dumps(seats, indent=2) + "\n", encoding="utf-8")
                    self._repair_manifest(version)

                verdict = run_reference_validator(self._mutated_copy(mutate))
                self.assertFalse(verdict["accepted"], verdict)
                self.assertIn("representative joint allocation", verdict["error"])

    def test_missing_representative_allocation_is_rejected_under_a_pointer(self) -> None:
        def mutate(root: Path, version: Path) -> None:
            seats_path = version / "seats.json"
            seats = json.loads(seats_path.read_text(encoding="utf-8"))
            seats.pop("representative_allocation")
            seats_path.write_text(json.dumps(seats, indent=2) + "\n", encoding="utf-8")
            self._repair_manifest(version)

        verdict = run_reference_validator(self._mutated_copy(mutate))
        self.assertFalse(verdict["accepted"])
        self.assertIn("representative joint allocation", verdict["error"])

    def test_all_six_contracts_must_share_the_deterministic_payload_hash(self) -> None:
        for name in ("forecast.json", "parties.json", "seats.json", "groups.json", "calibration.json", "metadata.json"):
            with self.subTest(contract=name):

                def mutate(root: Path, version: Path, name=name) -> None:
                    path = version / name
                    contract = json.loads(path.read_text(encoding="utf-8"))
                    contract.pop("deterministic_payload_sha256")
                    path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
                    self._repair_manifest(version)

                verdict = run_reference_validator(self._mutated_copy(mutate))
                self.assertFalse(verdict["accepted"], verdict)
                self.assertIn("deterministic simulation payload", verdict["error"])

    def test_dirty_source_provenance_is_rejected_under_a_pointer(self) -> None:
        def mutate(root: Path, version: Path) -> None:
            for name in ("metadata.json", "manifest.json"):
                path = version / name
                value = json.loads(path.read_text(encoding="utf-8"))
                value["source_worktree_clean"] = False
                path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
            self._repair_manifest(version)

        verdict = run_reference_validator(self._mutated_copy(mutate))
        self.assertFalse(verdict["accepted"])
        self.assertIn("dirty or incomplete source provenance", verdict["error"])

    @staticmethod
    def _repair_manifest(version: Path) -> None:
        """Re-point current.json at a hand-mutated version.

        Mutating a contract changes the manifest hash, which the pointer would
        catch first.  These tests target the payload checks specifically, so
        the pointer is repaired to let the mutation reach them.
        """

        import hashlib

        root = version.parent.parent
        pointer_path = root / "current.json"
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        manifest_bytes = (version / "manifest.json").read_bytes()
        pointer["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
        pointer_path.write_text(json.dumps(pointer, indent=2) + "\n", encoding="utf-8")

    # ---- legacy regression ----------------------------------------------

    def test_legacy_flat_august_27_payload_still_loads_through_the_fallback(self) -> None:
        """The pre-extraction website payload must stay renderable, unmodified."""

        verdict = run_reference_validator(LEGACY_FLAT_FIXTURE)
        self.assertTrue(verdict["accepted"], verdict)
        self.assertEqual(verdict["mode"], "legacy_flat_fallback")
        self.assertFalse(verdict["certified"])
        self.assertEqual(verdict["status_text"], "Forecast loaded, but it is not certified.")
        self.assertEqual(verdict["seat_allocation_source"], "legacy_normalized_marginal_medians")
        # Schema 1.0 with no source_repository means the original repository.
        self.assertEqual(verdict["schema_version"], "1.0")
        self.assertIsNone(verdict["source_repository"])
        self.assertEqual(
            verdict["deterministic_payload_sha256"],
            "967795912c05a00e364d62c84430f664290fc8f390a370369482e8c38a625473",
        )

    def test_legacy_flat_fixture_bytes_are_unchanged(self) -> None:
        """The fixture is a frozen copy and must never be regenerated."""

        expected = {
            "forecast.json": "16223f73e7a3044477d74e4d61a3879bc933509993930a8d549ec657d6af3dc9",
            "parties.json": "ebfd6debd3c488d8af3838947c95369a285829319284b5b9807594ac920a1313",
            "seats.json": "315c4c865dfb0f6fbbd4f373320122805f2e8fd89f9fa40293ab98e6b481ae66",
            "groups.json": "62c7a100976108bc6aa86ff112ff9eed96acdc5ae0ed414cf175ba7a60991706",
            "calibration.json": "5ffc22b34148dfe3f19ed9327413ae8630edc0c5bae87b90d5977cc749f2ae66",
            "metadata.json": "62e941d0ad67529927eee5d023a6447c26e9f6b649361786f3120a82294e3d30",
        }
        import hashlib

        for name, digest in expected.items():
            actual = hashlib.sha256((LEGACY_FLAT_FIXTURE / name).read_bytes()).hexdigest()
            self.assertEqual(actual, digest, f"{name} has been modified")


@unittest.skipUnless(WEBSITE_CONSUMER.is_file(), "Website checkout is not available for drift comparison")
class ReferenceValidatorDriftTests(unittest.TestCase):
    """The reference validator must not drift from the deployed website consumer."""

    def test_reference_validator_matches_the_website_acceptance_predicates(self) -> None:
        website = WEBSITE_CONSUMER.read_text(encoding="utf-8")
        reference_source = (
            REPOSITORY_ROOT / "scripts" / "static_exporter" / "contract" / "reference_publication_validator.js"
        ).read_text(encoding="utf-8")
        # Operand-level fragments, so an equivalent restatement of a guard
        # (negated rejection vs positive acceptance) does not read as drift,
        # but a changed rule does.
        predicates = [
            '"versions/" + pointer.publication_generation',
            "/^versions\\/[A-Za-z0-9_-]+$/",
            'manifest.publication_state !== "COMPLETE"',
            "manifestHash !== pointer.manifest_sha256",
            "data[5].source_worktree_clean !== true",
            "identities.length !== 6",
            "value !== expected",
            "total === 349 && representative.total_seats === 349",
            "error.status !== 404",
        ]
        normalize = lambda text: "".join(text.split())
        website_normalized = normalize(website)
        reference_normalized = normalize(reference_source)
        for predicate in predicates:
            with self.subTest(predicate=predicate):
                needle = normalize(predicate)
                self.assertTrue(
                    needle in website_normalized,
                    f"Predicate is no longer present in the website consumer: {predicate}",
                )
                self.assertTrue(
                    needle in reference_normalized,
                    f"Reference validator has drifted from the website consumer: {predicate}",
                )


if __name__ == "__main__":
    unittest.main()
