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

try:
    from ._website_repo import SKIP_REASON, website_consumer_path
except (ImportError, ValueError):
    from tests._website_repo import SKIP_REASON, website_consumer_path


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

    @staticmethod
    def _browser_probability(value: float) -> str:
        parsed = float(value)
        if parsed == 0:
            return "0,0\xa0%"
        if parsed == 1:
            return "100,0\xa0%"
        percent = parsed * 100
        if percent < 0.005:
            return "<0,01\xa0%"
        if percent > 99.995:
            return ">99,99\xa0%"
        rendered = f"{percent:.2f}" if percent < 1 or percent > 99 else f"{percent:.1f}"
        return rendered.replace(".", ",") + "\xa0%"

    @staticmethod
    def _assert_histogram_matches_entry(test_case: unittest.TestCase, snapshot: dict, entry: dict, mask: int) -> None:
        """Assert the rendered SVG bins are the published contiguous histogram."""

        histogram = entry["seat_histogram"]
        bins = snapshot["histogram_bins"]
        counts = histogram["counts"]
        total = sum(counts)
        test_case.assertFalse(snapshot["histogram_hidden"])
        test_case.assertEqual(snapshot["histogram_mask"], str(mask))
        test_case.assertEqual(snapshot["histogram_total_count"], str(total))
        test_case.assertEqual(snapshot["histogram_sample_count"], str(total))
        test_case.assertEqual(snapshot["histogram_min_seats"], str(histogram["min_seats"]))
        test_case.assertEqual(
            snapshot["histogram_max_seats"],
            str(histogram["min_seats"] + len(counts) - 1),
        )
        test_case.assertEqual(len(bins), len(counts))
        test_case.assertEqual([item["seat"] for item in bins], list(range(histogram["min_seats"], histogram["min_seats"] + len(counts))))
        test_case.assertEqual([item["count"] for item in bins], counts)
        test_case.assertEqual(sum(item["count"] for item in bins), total)
        test_case.assertEqual({item["coalition_mask"] for item in bins}, {str(mask)})
        test_case.assertTrue(all(item["majority"] == ("majority" if item["seat"] >= 175 else "below") for item in bins))
        test_case.assertEqual(snapshot["histogram_threshold"], 175)
        majority_count = sum(count for seat, count in zip(range(histogram["min_seats"], histogram["min_seats"] + len(counts)), counts) if seat >= 175)
        test_case.assertAlmostEqual(majority_count / total, entry["prob_majority"], places=12)
        test_case.assertIn("175 mandat", snapshot["histogram_text"])

    def test_production_consumer_renders_and_resolves_the_joint_coalition_lookup(self) -> None:
        verdict = run_actual_consumer(self.publication)
        initial = verdict["builder_initial"]
        self.assertTrue(initial["available"])
        self.assertFalse(initial["empty_hidden"])
        self.assertTrue(initial["results_hidden"])
        self.assertTrue(initial["histogram_hidden"])
        self.assertEqual(
            [tile["party"] for tile in initial["pool_tiles"]],
            ["M", "L", "C", "KD", "S", "V", "MP", "SD"],
        )
        self.assertEqual(initial["government_tiles"], [])
        self.assertEqual(initial["support_tiles"], [])
        self.assertTrue(all({action["action"] for action in tile["actions"]} == {"government", "support"} for tile in initial["pool_tiles"]))

        groups = json.loads((self.version / "groups.json").read_text(encoding="utf-8"))
        coalitions = groups["coalition_builder"]["coalitions"]

        government = verdict["builder_government"]
        government_entry = coalitions["137"]  # M + KD + SD
        self.assertTrue(government["available"])
        self.assertTrue(government["empty_hidden"])
        self.assertFalse(government["results_hidden"])
        self.assertEqual(government["government_mask"], "137")
        self.assertEqual(government["selected_support_mask"], "0")
        self.assertEqual(government["coalition_mask"], "137")
        self.assertIn("Sannolikhet för minst 175 mandat", government["results_html"])
        self.assertIn(self._browser_probability(government_entry["prob_majority"]), government["results_html"])
        self._assert_histogram_matches_entry(self, government, government_entry, 137)
        self.assertEqual({tile["party"] for tile in government["government_tiles"]}, {"M", "KD", "SD"})
        self.assertEqual({tile["party"] for tile in government["pool_tiles"]}, {"L", "C", "S", "V", "MP"})
        self.assertTrue(
            all(
                {action["action"] for action in tile["actions"]} == {"support", "pool"}
                for tile in government["government_tiles"]
            )
        )

        with_support = verdict["builder_with_support"]
        union_entry = coalitions["139"]  # M + KD + SD + L
        self.assertEqual(with_support["government_mask"], "137")
        self.assertEqual(with_support["selected_support_mask"], "2")
        self.assertEqual(with_support["coalition_mask"], "139")
        self.assertIn("M + L + KD + SD", with_support["histogram_context"])
        self.assertIn(self._browser_probability(union_entry["prob_majority"]), with_support["results_html"])
        self._assert_histogram_matches_entry(self, with_support, union_entry, 139)
        self.assertEqual({tile["party"] for tile in with_support["support_tiles"]}, {"L"})

    def test_production_consumer_degrades_schema_1_2_without_inventing_histogram_data(self) -> None:
        def mutate(root: Path, version: Path) -> None:
            for name in ("forecast.json", "parties.json", "seats.json", "groups.json", "calibration.json", "metadata.json"):
                path = version / name
                contract = json.loads(path.read_text(encoding="utf-8"))
                contract["schema_version"] = "1.2"
                if name == "groups.json":
                    for entry in contract["coalition_builder"]["coalitions"].values():
                        entry.pop("seat_histogram", None)
                path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
            manifest_path = version / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = "1.2"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            pointer_path = root / "current.json"
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            pointer["schema_version"] = "1.2"
            pointer_path.write_text(json.dumps(pointer, indent=2) + "\n", encoding="utf-8")
            self._repair_pointer_hash(root, version)

        legacy_builder = self._mutated_copy(mutate)
        verdict = run_actual_consumer(legacy_builder)
        self.assertTrue(verdict["accepted"], verdict)
        self.assertTrue(verdict["builder_initial"]["available"])
        self.assertTrue(verdict["builder_initial"]["histogram_hidden"])
        self.assertEqual(verdict["builder_initial"]["histogram_bins"], [])
        self.assertTrue(verdict["builder_government"]["histogram_hidden"])
        self.assertEqual(verdict["builder_government"]["histogram_bins"], [])
        self.assertTrue(verdict["builder_with_support"]["histogram_hidden"])
        self.assertEqual(verdict["builder_with_support"]["histogram_bins"], [])

    def test_production_consumer_hides_builder_for_a_schema_1_1_publication_without_lookup(self) -> None:
        def mutate(root: Path, version: Path) -> None:
            for name in ("forecast.json", "parties.json", "seats.json", "groups.json", "calibration.json", "metadata.json"):
                path = version / name
                contract = json.loads(path.read_text(encoding="utf-8"))
                contract["schema_version"] = "1.1"
                if name == "groups.json":
                    contract.pop("coalition_builder", None)
                path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
            manifest_path = version / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = "1.1"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            pointer_path = root / "current.json"
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            pointer["schema_version"] = "1.1"
            pointer_path.write_text(json.dumps(pointer, indent=2) + "\n", encoding="utf-8")
            self._repair_pointer_hash(root, version)

        legacy = self._mutated_copy(mutate)
        verdict = run_actual_consumer(legacy)
        self.assertTrue(verdict["accepted"], verdict)
        self.assertFalse(verdict["builder_initial"]["available"])
        self.assertIsNone(verdict["builder_government"])
        self.assertIsNone(verdict["builder_with_support"])

    def test_production_consumer_hides_a_malformed_coalition_histogram(self) -> None:
        def mutate(root: Path, version: Path) -> None:
            groups_path = version / "groups.json"
            groups = json.loads(groups_path.read_text(encoding="utf-8"))
            groups["coalition_builder"]["coalitions"]["7"]["seat_histogram"]["counts"][0] = -1
            groups_path.write_text(json.dumps(groups, indent=2) + "\n", encoding="utf-8")
            self._repair_pointer_hash(root, version)

        malformed = self._mutated_copy(mutate)
        verdict = run_actual_consumer(malformed)
        self.assertTrue(verdict["accepted"], verdict)
        self.assertFalse(verdict["builder_initial"]["available"])
        self.assertIsNone(verdict["builder_government"])
        self.assertIsNone(verdict["builder_with_support"])

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
