"""ACTUAL_BROWSER_CONSUMER_TEST — drives the deployed website source verbatim.

This module is the one that genuinely closes the exporter/browser contract
gap. It does not reimplement any acceptance rule. It reads

    edvinli.github.io/assets/js/election-simulator.js

byte for byte, evaluates it, and asserts on the user-visible outcome the
production file produces. Every rule exercised here is the deployed rule.

Contrast with ``tests/test_reference_publication_contract.py``
(REFERENCE_CONTRACT_TEST), which runs an independent reimplementation. That
one is portable and always runs; this one is authoritative but requires the
website checkout.

The website repository is deliberately NOT modified by this branch, so this
test is a cross-repository integration test: it skips loudly when the
checkout is unavailable rather than failing. Point it elsewhere with
``ELECTION_SIMULATOR_WEBSITE_REPO``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from scripts.simulator.engine import simulate_election
from scripts.static_exporter import export_static_data

from ._website_repo import SKIP_REASON, website_consumer_path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HARNESS = REPOSITORY_ROOT / "scripts" / "static_exporter" / "contract" / "actual_consumer_harness.js"
LEGACY_FLAT_FIXTURE = REPOSITORY_ROOT / "tests" / "fixtures" / "legacy_flat_publication_2026_08_27"

NODE = shutil.which("node")


CONSUMER = website_consumer_path()


def run_actual_consumer(publication_dir: Path) -> dict:
    """Evaluate the production website file against a publication directory."""

    completed = subprocess.run(
        [NODE, str(HARNESS), str(CONSUMER), str(publication_dir)],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
    )
    if not completed.stdout.strip():
        raise AssertionError(f"Production consumer harness produced no verdict: {completed.stderr}")
    return json.loads(completed.stdout)


@unittest.skipIf(NODE is None, "Node is required to evaluate the production consumer")
@unittest.skipUnless(CONSUMER.is_file(), SKIP_REASON)
class ActualBrowserConsumerTests(unittest.TestCase):
    """The deployed website file must accept what the Python exporter writes."""

    @classmethod
    def setUpClass(cls) -> None:
        result = simulate_election(as_of="2026-08-23", election_date="2026-09-13", samples=8, seed=12345)
        result.manifest["source_worktree_clean"] = True
        cls._tmp = tempfile.TemporaryDirectory()
        cls.publication = Path(cls._tmp.name) / "publication"
        cls.manifest = export_static_data(
            result,
            output_dir=cls.publication,
            generated_at_utc="2026-08-27T00:00:00+00:00",
            calibration_dir=REPOSITORY_ROOT / "data" / "processed",
        )
        cls.generation = cls.manifest["publication_generation"]
        cls.version = cls.publication / "versions" / cls.generation

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _mutated_copy(self, mutate) -> Path:
        target = Path(self.enterContext(tempfile.TemporaryDirectory())) / "publication"
        shutil.copytree(self.publication, target)
        mutate(target, target / "versions" / self.generation)
        return target

    @staticmethod
    def _repair_pointer_hash(root: Path, version: Path) -> None:
        """Re-point current.json after a deliberate contract mutation.

        The manifest hash check would otherwise fire first and mask the
        payload-level rule under test.
        """

        pointer_path = root / "current.json"
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        pointer["manifest_sha256"] = hashlib.sha256((version / "manifest.json").read_bytes()).hexdigest()
        pointer_path.write_text(json.dumps(pointer, indent=2) + "\n", encoding="utf-8")

    # ---- provenance of what is under test -------------------------------

    def test_harness_evaluates_the_unmodified_deployed_source(self) -> None:
        """Guard against the harness silently testing a local substitute."""

        verdict = run_actual_consumer(self.publication)
        self.assertEqual(verdict["source_file"], str(CONSUMER.resolve()))
        self.assertEqual(
            verdict["source_sha256"],
            hashlib.sha256(CONSUMER.read_bytes()).hexdigest(),
            "The harness must evaluate the website file byte for byte",
        )
        # It is the production consumer, not the reference reimplementation.
        reference = REPOSITORY_ROOT / "scripts" / "static_exporter" / "contract" / "reference_publication_validator.js"
        self.assertNotEqual(
            verdict["source_sha256"],
            hashlib.sha256(reference.read_bytes()).hexdigest(),
        )

    # ---- happy path -----------------------------------------------------

    def test_production_consumer_accepts_exporter_output(self) -> None:
        verdict = run_actual_consumer(self.publication)
        self.assertTrue(verdict["accepted"], verdict)
        self.assertEqual(verdict["status_text"], "Certified forecast loaded.")
        self.assertTrue(verdict["certified"])
        self.assertEqual(verdict["seat_nodes"], 349)
        self.assertIn("representative joint simulation draw", verdict["parliament_aria_label"])

    def test_production_consumer_requests_the_canonical_paths(self) -> None:
        verdict = run_actual_consumer(self.publication)
        self.assertEqual(
            verdict["requested_paths"],
            ["current.json"]
            + [
                f"versions/{self.generation}/{name}"
                for name in (
                    "forecast.json",
                    "parties.json",
                    "seats.json",
                    "groups.json",
                    "calibration.json",
                    "metadata.json",
                    "manifest.json",
                )
            ],
        )

    # ---- rejection under the production rules ---------------------------

    def test_production_consumer_rejects_a_malformed_pointer(self) -> None:
        cases = {
            "state not COMPLETE": lambda p: p.__setitem__("publication_state", "PENDING"),
            "path outside versions/": lambda p: p.__setitem__("path", p["publication_generation"]),
            "path traversal": lambda p: p.__setitem__("path", "versions/../versions"),
            "missing manifest hash": lambda p: p.pop("manifest_sha256"),
            "non-string generation": lambda p: p.__setitem__("publication_generation", 1),
        }
        for label, mutate_pointer in cases.items():
            with self.subTest(case=label):

                def mutate(root: Path, version: Path, mutate_pointer=mutate_pointer) -> None:
                    pointer_path = root / "current.json"
                    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
                    mutate_pointer(pointer)
                    pointer_path.write_text(json.dumps(pointer, indent=2) + "\n", encoding="utf-8")

                verdict = run_actual_consumer(self._mutated_copy(mutate))
                self.assertFalse(verdict["accepted"], verdict)
                self.assertEqual(verdict["error"], "Current publication pointer is invalid")

    def test_production_consumer_rejects_manifest_byte_tampering(self) -> None:
        def mutate(root: Path, version: Path) -> None:
            manifest_path = version / "manifest.json"
            # Same parsed JSON, different bytes.
            manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

        verdict = run_actual_consumer(self._mutated_copy(mutate))
        self.assertFalse(verdict["accepted"])
        self.assertEqual(verdict["error"], "Current publication pointer hash does not match the manifest")

    def test_production_consumer_requires_a_349_seat_representative_allocation(self) -> None:
        for label, delta in (("348 seats", -1), ("350 seats", 1)):
            with self.subTest(case=label):

                def mutate(root: Path, version: Path, delta=delta) -> None:
                    seats_path = version / "seats.json"
                    seats = json.loads(seats_path.read_text(encoding="utf-8"))
                    seats["representative_allocation"]["seats"]["M"] += delta
                    seats_path.write_text(json.dumps(seats, indent=2) + "\n", encoding="utf-8")
                    self._repair_pointer_hash(root, version)

                verdict = run_actual_consumer(self._mutated_copy(mutate))
                self.assertFalse(verdict["accepted"], verdict)
                self.assertEqual(
                    verdict["error"], "Published seat contract has no valid representative joint allocation"
                )

    def test_production_consumer_requires_all_six_payload_hashes(self) -> None:
        for name in ("forecast.json", "parties.json", "seats.json", "groups.json", "calibration.json", "metadata.json"):
            with self.subTest(contract=name):

                def mutate(root: Path, version: Path, name=name) -> None:
                    path = version / name
                    contract = json.loads(path.read_text(encoding="utf-8"))
                    contract.pop("deterministic_payload_sha256")
                    path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
                    self._repair_pointer_hash(root, version)

                verdict = run_actual_consumer(self._mutated_copy(mutate))
                self.assertFalse(verdict["accepted"], verdict)
                self.assertEqual(
                    verdict["error"], "Publication files do not all link the deterministic simulation payload"
                )

    def test_production_consumer_rejects_dirty_source_provenance(self) -> None:
        def mutate(root: Path, version: Path) -> None:
            for name in ("metadata.json", "manifest.json"):
                path = version / name
                value = json.loads(path.read_text(encoding="utf-8"))
                value["source_worktree_clean"] = False
                path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
            self._repair_pointer_hash(root, version)

        verdict = run_actual_consumer(self._mutated_copy(mutate))
        self.assertFalse(verdict["accepted"])
        self.assertEqual(verdict["error"], "Certified publication has dirty or incomplete source provenance")

    # ---- fallback semantics ---------------------------------------------

    def test_production_consumer_falls_back_only_on_a_literal_404(self) -> None:
        def remove_pointer(root: Path, version: Path) -> None:
            (root / "current.json").unlink()
            for path in version.iterdir():
                shutil.copyfile(path, root / path.name)

        verdict = run_actual_consumer(self._mutated_copy(remove_pointer))
        self.assertTrue(verdict["accepted"], verdict)

        def corrupt_pointer(root: Path, version: Path) -> None:
            (root / "current.json").write_text("{ not json", encoding="utf-8")
            for path in version.iterdir():
                shutil.copyfile(path, root / path.name)

        verdict = run_actual_consumer(self._mutated_copy(corrupt_pointer))
        self.assertFalse(verdict["accepted"], "A present-but-unreadable pointer must not fall back")

    def test_production_consumer_still_renders_the_frozen_legacy_payload(self) -> None:
        """The August 27 flat payload must stay renderable, unmodified."""

        verdict = run_actual_consumer(LEGACY_FLAT_FIXTURE)
        self.assertTrue(verdict["accepted"], verdict)
        self.assertEqual(verdict["status_text"], "Forecast loaded, but it is not certified.")
        self.assertFalse(verdict["certified"])
        self.assertIn("legacy normalized marginal medians", verdict["parliament_aria_label"])
        self.assertEqual(verdict["seat_nodes"], 349)
        self.assertEqual(verdict["requested_paths"][0], "current.json")


if __name__ == "__main__":
    unittest.main()
