"""Determinism and fail-safe tests for the static publication contract."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.simulator.config import PARLIAMENTARY_PARTIES_8
from scripts.simulator.engine import simulate_election
from scripts.simulator.pipeline import build_canonical_summary_dict
from scripts.simulator.reproducibility import (
    SOURCE_REPOSITORY,
    UNRESOLVED_GIT_COMMIT,
    get_git_commit_hash,
    is_git_worktree_clean,
    require_certified_source_provenance,
    resolve_source_repository,
)
from scripts.static_exporter import export_static_data, validate_published_directory
from scripts.static_exporter.exporter import PUBLICATION_SCHEMA_VERSION
from scripts.static_exporter.exporter import (
    COALITION_BUILDER_ENTRY_FIELDS,
    COALITION_BUILDER_SUMMARY_FIELDS,
    _histogram_quantile,
    validate_publication_contract,
)


class StaticExporterTests(unittest.TestCase):
    @staticmethod
    def _clean_result(*, seed: int = 12345):
        result = simulate_election(as_of="2026-08-23", election_date="2026-09-13", samples=8, seed=seed)
        # The test process has unrelated uncommitted research artifacts.  The
        # production exporter requires this provenance bit to be an actual
        # boolean true, so use a synthetic clean-source result for positive
        # contract tests.
        result.manifest["source_worktree_clean"] = True
        return result

    @staticmethod
    def _version_dir(publication_root: Path) -> Path:
        """Resolve the immutable version the pointer addresses.

        The canonical contract publishes no flat aliases, so every read goes
        through current.json exactly as the browser consumer does.
        """
        pointer = json.loads((publication_root / "current.json").read_text())
        return publication_root / pointer["path"]

    def test_export_contract_is_complete_and_deterministic_excluding_timestamp(self) -> None:
        result = self._clean_result()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = export_static_data(
                result,
                output_dir=root / "first",
                generated_at_utc="2026-08-27T00:00:00+00:00",
                calibration_dir=Path("data/processed"),
            )
            second = export_static_data(
                result,
                output_dir=root / "second",
                generated_at_utc="2026-08-27T01:00:00+00:00",
                calibration_dir=Path("data/processed"),
            )
            self.assertEqual(first["deterministic_content_sha256"], second["deterministic_content_sha256"])
            self.assertNotEqual(first["generated_at_utc"], second["generated_at_utc"])
            publication_root = root / "first"
            pointer = json.loads((publication_root / "current.json").read_text())
            version = publication_root / pointer["path"]
            self.assertEqual(
                {path.name for path in version.iterdir()},
                {"forecast.json", "parties.json", "seats.json", "groups.json", "calibration.json", "metadata.json", "manifest.json"},
            )
            self.assertEqual(
                first["deterministic_content_sha256"],
                validate_published_directory(publication_root)["deterministic_content_sha256"],
            )
            self.assertEqual(pointer["publication_generation"], version.name)
            self.assertEqual(
                sum(json.loads((version / "seats.json").read_text())["representative_allocation"]["seats"].values()),
                349,
            )

    def test_schema_1_3_contains_all_joint_coalitions_histograms_and_preserves_payload_hash(self) -> None:
        result = self._clean_result()
        payload_before = build_canonical_summary_dict(result)["deterministic_payload_sha256"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = export_static_data(
                result,
                output_dir=root / "first",
                generated_at_utc="2026-08-27T00:00:00+00:00",
            )
            second = export_static_data(
                result,
                output_dir=root / "second",
                generated_at_utc="2026-08-27T01:00:00+00:00",
            )
            first_version = self._version_dir(root / "first")
            second_version = self._version_dir(root / "second")
            first_groups = json.loads((first_version / "groups.json").read_text())
            second_groups = json.loads((second_version / "groups.json").read_text())

            self.assertEqual(first["schema_version"], PUBLICATION_SCHEMA_VERSION)
            self.assertEqual(first_groups["schema_version"], PUBLICATION_SCHEMA_VERSION)
            self.assertEqual(first_groups["groups"]["tido"]["parties"], ["M", "SD", "KD", "L"])
            builder = first_groups["coalition_builder"]
            self.assertEqual(builder["party_order"], list(PARLIAMENTARY_PARTIES_8))
            self.assertEqual(builder["encoding"], "bitmask")
            self.assertEqual(builder["majority_threshold"], 175)
            self.assertEqual(len(builder["coalitions"]), 256)
            self.assertEqual(list(builder["coalitions"]), [str(mask) for mask in range(256)])
            self.assertEqual(
                (first_version / "groups.json").read_bytes(),
                (second_version / "groups.json").read_bytes(),
            )
            self.assertEqual(
                payload_before,
                build_canonical_summary_dict(result)["deterministic_payload_sha256"],
            )
            self.assertEqual(
                payload_before,
                first_groups["deterministic_payload_sha256"],
            )

            metric_fields = tuple(field for field in COALITION_BUILDER_SUMMARY_FIELDS if field != "mask")
            all_draws = result.seats_matrix
            for mask in range(256):
                key = str(mask)
                entry = builder["coalitions"][key]
                parties = [
                    party
                    for index, party in enumerate(PARLIAMENTARY_PARTIES_8)
                    if mask & (1 << index)
                ]
                self.assertEqual(list(entry), list(COALITION_BUILDER_ENTRY_FIELDS))
                self.assertEqual(entry["mask"], mask)
                self.assertEqual(entry["parties"], parties)

                indices = [PARLIAMENTARY_PARTIES_8.index(party) for party in parties]
                coalition_draws = (
                    all_draws[:, indices].sum(axis=1)
                    if indices
                    else np.zeros(all_draws.shape[0], dtype=np.int64)
                )
                complement_mask = 255 ^ mask
                complement_indices = [
                    index
                    for index in range(len(PARLIAMENTARY_PARTIES_8))
                    if complement_mask & (1 << index)
                ]
                complement_draws = all_draws[:, complement_indices].sum(axis=1)
                np.testing.assert_array_equal(coalition_draws + complement_draws, 349)
                self.assertTrue(np.all((coalition_draws >= 0) & (coalition_draws <= 349)))

                histogram = entry["seat_histogram"]
                self.assertEqual(list(histogram), ["min_seats", "counts"])
                minimum = int(coalition_draws.min())
                maximum = int(coalition_draws.max())
                expected_counts = np.bincount(
                    coalition_draws.astype(np.int64), minlength=maximum + 1
                )[minimum:]
                self.assertEqual(histogram["min_seats"], minimum)
                self.assertEqual(histogram["counts"], [int(count) for count in expected_counts])
                self.assertEqual(sum(histogram["counts"]), all_draws.shape[0])
                self.assertGreaterEqual(histogram["min_seats"], 0)
                self.assertLessEqual(
                    histogram["min_seats"] + len(histogram["counts"]) - 1,
                    349,
                )
                self.assertTrue(
                    all(isinstance(count, int) and count >= 0 for count in histogram["counts"])
                )
                expanded = np.repeat(
                    np.arange(
                        histogram["min_seats"],
                        histogram["min_seats"] + len(histogram["counts"]),
                    ),
                    histogram["counts"],
                )
                self.assertEqual(int(np.median(expanded)), entry["median_seats"])
                self.assertEqual(int(np.percentile(expanded, 5)), entry["p05_seats"])
                self.assertEqual(int(np.percentile(expanded, 95)), entry["p95_seats"])
                majority_count = int(np.sum(coalition_draws >= 175))
                self.assertAlmostEqual(
                    majority_count / all_draws.shape[0],
                    entry["prob_majority"],
                    places=15,
                )

                expected_summary = result.summarize_group(parties)
                expected = expected_summary.__dict__
                for field in metric_fields:
                    if field == "parties":
                        self.assertEqual(entry[field], list(expected[field]))
                    else:
                        self.assertEqual(entry[field], expected[field])

                complement_entry = builder["coalitions"][str(complement_mask)]
                self.assertAlmostEqual(
                    entry["prob_majority"] + complement_entry["prob_majority"],
                    1.0,
                    places=12,
                )
                complement_histogram = complement_entry["seat_histogram"]
                for seats in range(350):
                    coalition_count = (
                        histogram["counts"][seats - histogram["min_seats"]]
                        if histogram["min_seats"] <= seats < histogram["min_seats"] + len(histogram["counts"])
                        else 0
                    )
                    reflected_seats = 349 - seats
                    complement_count = (
                        complement_histogram["counts"][reflected_seats - complement_histogram["min_seats"]]
                        if complement_histogram["min_seats"] <= reflected_seats < complement_histogram["min_seats"] + len(complement_histogram["counts"])
                        else 0
                    )
                    self.assertEqual(coalition_count, complement_count)
                self.assertGreaterEqual(entry["mean_seats"], 0)
                self.assertLessEqual(entry["mean_seats"], 349)
                for field in (
                    "median_seats",
                    "p05_seats",
                    "p10_seats",
                    "p25_seats",
                    "p75_seats",
                    "p90_seats",
                    "p95_seats",
                ):
                    self.assertGreaterEqual(entry[field], 0)
                    self.assertLessEqual(entry[field], 349)

            empty = builder["coalitions"]["0"]
            for field in (
                "mean_seats",
                "median_seats",
                "p05_seats",
                "p10_seats",
                "p25_seats",
                "p75_seats",
                "p90_seats",
                "p95_seats",
                "prob_majority",
            ):
                self.assertEqual(empty[field], 0)
            self.assertEqual(
                empty["seat_histogram"],
                {"min_seats": 0, "counts": [all_draws.shape[0]]},
            )
            full = builder["coalitions"]["255"]
            for field in (
                "mean_seats",
                "median_seats",
                "p05_seats",
                "p10_seats",
                "p25_seats",
                "p75_seats",
                "p90_seats",
                "p95_seats",
            ):
                self.assertEqual(full[field], 349)
            self.assertEqual(full["prob_majority"], 1)
            self.assertEqual(
                full["seat_histogram"],
                {"min_seats": 349, "counts": [all_draws.shape[0]]},
            )

            for group_name, parties in (
                ("tido", ["M", "SD", "KD", "L"]),
                ("red_green_center", ["S", "V", "MP", "C"]),
            ):
                mask = sum(1 << PARLIAMENTARY_PARTIES_8.index(party) for party in parties)
                entry = builder["coalitions"][str(mask)]
                existing = first_groups["groups"][group_name]
                for field in (
                    "mean_seats",
                    "median_seats",
                    "p05_seats",
                    "p10_seats",
                    "p25_seats",
                    "p75_seats",
                    "p90_seats",
                    "p95_seats",
                    "prob_majority",
                ):
                    self.assertEqual(entry[field], existing[field])

    def test_schema_1_0_and_1_1_contracts_remain_accepted_without_builder(self) -> None:
        result = self._clean_result()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "publication"
            manifest = export_static_data(result, output_dir=output)
            version = output / "versions" / manifest["publication_generation"]
            current = {
                name: json.loads((version / name).read_text())
                for name in ("forecast.json", "parties.json", "seats.json", "groups.json", "calibration.json", "metadata.json")
            }
            for schema in ("1.0", "1.1"):
                contracts = deepcopy(current)
                for contract in contracts.values():
                    contract["schema_version"] = schema
                contracts["groups.json"].pop("coalition_builder")
                validate_publication_contract(contracts)

    def test_schema_1_3_builder_validation_rejects_missing_or_malformed_entries(self) -> None:
        result = self._clean_result()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "publication"
            manifest = export_static_data(result, output_dir=output)
            version = output / "versions" / manifest["publication_generation"]
            current = {
                name: json.loads((version / name).read_text())
                for name in ("forecast.json", "parties.json", "seats.json", "groups.json", "calibration.json", "metadata.json")
            }
            for label, mutate in (
                ("missing mask", lambda coalitions: coalitions.pop("7")),
                ("wrong membership", lambda coalitions: coalitions["7"]["parties"].append("REST")),
                ("malformed histogram", lambda coalitions: coalitions["7"].__setitem__("seat_histogram", {})),
                (
                    "entry field order",
                    lambda coalitions: coalitions.__setitem__(
                        "7",
                        {field: coalitions["7"][field] for field in reversed(COALITION_BUILDER_ENTRY_FIELDS)},
                    ),
                ),
                (
                    "histogram count total",
                    lambda coalitions: coalitions["7"]["seat_histogram"]["counts"].__setitem__(0, 0),
                ),
                (
                    "histogram support",
                    lambda coalitions: coalitions["7"]["seat_histogram"].__setitem__("min_seats", 350),
                ),
                (
                    "histogram probability",
                    lambda coalitions: coalitions["7"].__setitem__("prob_majority", 0.5),
                ),
                (
                    "histogram quantile",
                    lambda coalitions: coalitions["7"].__setitem__("p95_seats", 0),
                ),
                ("out of bounds", lambda coalitions: coalitions["7"].__setitem__("median_seats", 350)),
            ):
                with self.subTest(case=label):
                    contracts = deepcopy(current)
                    mutate(contracts["groups.json"]["coalition_builder"]["coalitions"])
                    with self.assertRaises(ValueError):
                        validate_publication_contract(contracts)

    def test_histogram_quantiles_match_numpy_lerp_upper_branch(self) -> None:
        """Integer truncation must preserve NumPy's upper-end _lerp branch."""

        adversarial = (
            ([84, 341, 278, 215, 199, 89, 281, 157], 0.95, 319),
            ([71, 10, 241], 0.95, 224),
        )
        for values, quantile, expected in adversarial:
            with self.subTest(values=values):
                minimum = min(values)
                counts = [values.count(seats) for seats in range(min(values), max(values) + 1)]
                self.assertEqual(_histogram_quantile(minimum, counts, quantile), expected)

    def test_schema_1_3_validator_requires_deterministic_object_key_order(self) -> None:
        result = self._clean_result()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "publication"
            manifest = export_static_data(result, output_dir=output)
            version = output / "versions" / manifest["publication_generation"]
            current = {
                name: json.loads((version / name).read_text())
                for name in ("forecast.json", "parties.json", "seats.json", "groups.json", "calibration.json", "metadata.json")
            }
            # Rebuild each object with reversed insertion order.  Dict.update
            # alone cannot change an existing key's position, so replacement
            # is performed explicitly below for a focused order regression.
            for label in ("builder", "histogram"):
                with self.subTest(case=label):
                    contracts = deepcopy(current)
                    groups = contracts["groups.json"]
                    builder = groups["coalition_builder"]
                    if label == "builder":
                        groups["coalition_builder"] = {
                            field: builder[field] for field in reversed(list(builder))
                        }
                    else:
                        entry = builder["coalitions"]["7"]
                        entry["seat_histogram"] = {
                            field: entry["seat_histogram"][field]
                            for field in reversed(list(entry["seat_histogram"]))
                        }
                    with self.assertRaises(ValueError):
                        validate_publication_contract(contracts)

    def test_schema_1_2_builder_remains_accepted_without_histograms(self) -> None:
        result = self._clean_result()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "publication"
            manifest = export_static_data(result, output_dir=output)
            version = output / "versions" / manifest["publication_generation"]
            contracts = {
                name: json.loads((version / name).read_text())
                for name in (
                    "forecast.json",
                    "parties.json",
                    "seats.json",
                    "groups.json",
                    "calibration.json",
                    "metadata.json",
                )
            }
            for contract in contracts.values():
                contract["schema_version"] = "1.2"
            builder = contracts["groups.json"]["coalition_builder"]
            for entry in builder["coalitions"].values():
                entry.pop("seat_histogram")
            validate_publication_contract(contracts)

    def test_dirty_source_is_rejected_for_certified_publication(self) -> None:
        result = simulate_election(as_of="2026-08-23", election_date="2026-09-13", samples=8, seed=12345)
        with tempfile.TemporaryDirectory() as tmp:
            for value in (False, 1, "true", None):
                result.manifest["source_worktree_clean"] = value
                with self.assertRaisesRegex(ValueError, "source_worktree_clean"):
                    export_static_data(result, output_dir=Path(tmp) / f"publication-{value}")

    def test_failed_validation_does_not_replace_existing_publication(self) -> None:
        result = self._clean_result()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "publication"
            export_static_data(result, output_dir=output, generated_at_utc="2026-08-27T00:00:00+00:00")
            version = self._version_dir(output)
            old_manifest = (version / "manifest.json").read_bytes()
            # A malformed result fails while contracts are built, before any
            # staging directory is swapped into the live output.
            with self.assertRaises((AttributeError, KeyError, ValueError)):
                export_static_data(object(), output_dir=output, generated_at_utc="2026-08-27T01:00:00+00:00")
            self.assertEqual((version / "manifest.json").read_bytes(), old_manifest)

    def test_change_since_prior_is_explicit_and_uses_median_deltas(self) -> None:
        result = self._clean_result()
        prior = {
            "as_of": "2026-08-22",
            "snapshot_id": "prior-id",
            "deterministic_payload_sha256": "prior-payload",
            "national_vote_summary": {
                party: {"vote_share_median": 1.0} for party in result.summary.parties
            },
            "seat_summary": {
                party: {"median": 1} for party in result.summary.parties if party != "REST"
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            contracts = export_static_data(
                result,
                output_dir=Path(tmp) / "publication",
                generated_at_utc="2026-08-27T00:00:00+00:00",
                prior_snapshot=prior,
            )
            self.assertEqual(contracts["deterministic_content_sha256"].__class__, str)
            version = self._version_dir(Path(tmp) / "publication")
            forecast = json.loads((version / "forecast.json").read_text())
            self.assertEqual(forecast["change_since_prior"]["status"], "AVAILABLE")
            self.assertEqual(forecast["change_since_prior"]["prior_as_of"], "2026-08-22")
            self.assertEqual(
                forecast["change_since_prior"]["seat_median_change"]["M"],
                result.summary.parties["M"].seats_median - 1,
            )
            validate_publication_contract({
                name: json.loads((version / name).read_text())
                for name in ("forecast.json", "parties.json", "seats.json", "groups.json", "calibration.json", "metadata.json")
            })

    def test_pointer_failure_keeps_previous_version_loadable(self) -> None:
        first_result = self._clean_result(seed=12345)
        second_result = self._clean_result(seed=54321)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "publication"
            export_static_data(first_result, output_dir=output, generated_at_utc="2026-08-27T00:00:00+00:00")
            old_pointer = (output / "current.json").read_bytes()
            old_pointer_data = json.loads(old_pointer)
            old_target = output / old_pointer_data["path"]
            old_manifest = (old_target / "manifest.json").read_bytes()
            with patch("scripts.static_exporter.exporter._atomic_replace_pointer", side_effect=OSError("injected pointer failure")):
                with self.assertRaisesRegex(OSError, "injected pointer failure"):
                    export_static_data(second_result, output_dir=output, generated_at_utc="2026-08-27T01:00:00+00:00")
            self.assertEqual((output / "current.json").read_bytes(), old_pointer)
            self.assertEqual((output / old_pointer_data["path"] / "manifest.json").read_bytes(), old_manifest)
            self.assertEqual(validate_published_directory(output)["publication_generation"], old_target.name)
            self.assertEqual(validate_published_directory(old_target)["publication_generation"], old_target.name)
            self.assertEqual(len(list((output / "versions").iterdir())), 2)

    def test_generation_collision_does_not_overwrite_immutable_version(self) -> None:
        result = self._clean_result()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "publication"
            export_static_data(
                result,
                output_dir=output,
                generated_at_utc="2026-08-27T00:00:00+00:00",
                generation_id="fixed-generation",
            )
            old_pointer = (output / "current.json").read_bytes()
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                export_static_data(
                    result,
                    output_dir=output,
                    generated_at_utc="2026-08-27T01:00:00+00:00",
                    generation_id="fixed-generation",
                )
            self.assertEqual((output / "current.json").read_bytes(), old_pointer)
            self.assertEqual(validate_published_directory(output)["publication_generation"], "fixed-generation")

    def test_existing_real_directory_is_supported_without_directory_swap(self) -> None:
        result = self._clean_result()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "publication"
            output.mkdir()
            marker = output / "marker"
            marker.write_text("legacy", encoding="utf-8")
            export_static_data(result, output_dir=output)
            self.assertEqual(marker.read_text(encoding="utf-8"), "legacy")
            self.assertEqual(validate_published_directory(output)["publication_state"], "COMPLETE")

    def test_repository_with_no_resolvable_commit_cannot_certify(self) -> None:
        """An unresolvable commit must never reach a published artifact."""

        result = self._clean_result()
        result.manifest["source_git_commit"] = UNRESOLVED_GIT_COMMIT
        result.manifest.pop("git_commit", None)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "publication"
            with self.assertRaisesRegex(ValueError, "resolvable source Git commit"):
                export_static_data(result, output_dir=output, generated_at_utc="2026-08-27T00:00:00+00:00")
            self.assertFalse(output.exists())

        # The same gate rejects an empty or missing commit field outright.
        for missing in ("", None):
            result.manifest["source_git_commit"] = missing
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, "resolvable source Git commit"):
                    export_static_data(result, output_dir=Path(tmp) / "publication")

    def test_real_repository_without_commits_cannot_certify(self) -> None:
        """End-to-end: a freshly initialised repo has no resolvable commit."""

        with tempfile.TemporaryDirectory() as tmp:
            empty_repo = Path(tmp) / "empty-repo"
            empty_repo.mkdir()
            init = subprocess.run(
                ["git", "init", "--quiet"], cwd=empty_repo, capture_output=True, text=True
            )
            if init.returncode != 0:
                self.skipTest("git is not available")

            commit = get_git_commit_hash(empty_repo)
            self.assertEqual(commit, UNRESOLVED_GIT_COMMIT)
            with self.assertRaisesRegex(ValueError, "resolvable source Git commit"):
                require_certified_source_provenance(
                    {"source_git_commit": commit, "source_worktree_clean": True}
                )

            # And the exporter refuses a result carrying that provenance.
            result = self._clean_result()
            result.manifest["source_git_commit"] = commit
            result.manifest.pop("git_commit", None)
            output = Path(tmp) / "publication"
            with self.assertRaisesRegex(ValueError, "resolvable source Git commit"):
                export_static_data(result, output_dir=output)
            self.assertFalse(output.exists())

    def test_publication_contains_no_symlinks_and_no_flat_aliases(self) -> None:
        result = self._clean_result()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "publication"
            manifest = export_static_data(result, output_dir=output, generated_at_utc="2026-08-27T00:00:00+00:00")
            self.assertEqual({path.name for path in output.iterdir()}, {"current.json", "versions"})
            for path in output.rglob("*"):
                self.assertFalse(path.is_symlink(), f"{path} must not be a symlink")
            version = output / "versions" / manifest["publication_generation"]
            self.assertEqual(len(list(version.iterdir())), 7)
            # A symlinked contract must be refused even if its target is valid.
            forecast = version / "forecast.json"
            replacement = version / "forecast.real.json"
            forecast.rename(replacement)
            forecast.symlink_to(replacement.name)
            with self.assertRaisesRegex(ValueError, "real file, not a symlink"):
                validate_published_directory(output)

    def test_generation_id_is_sortable_and_web_safe(self) -> None:
        result = self._clean_result()
        with tempfile.TemporaryDirectory() as tmp:
            first = export_static_data(
                result, output_dir=Path(tmp) / "a", generated_at_utc="2026-08-27T00:00:00+00:00"
            )
            second = export_static_data(
                result, output_dir=Path(tmp) / "b", generated_at_utc="2026-08-27T09:15:30+00:00"
            )
        for manifest in (first, second):
            generation = manifest["publication_generation"]
            self.assertRegex(generation, r"^[A-Za-z0-9_-]+$")
            self.assertRegex(generation, r"^\d{8}T\d{6}Z-[0-9a-f]{8}$")
        self.assertLess(first["publication_generation"], second["publication_generation"])
        self.assertTrue(first["publication_generation"].startswith("20260827T000000Z-"))

    def test_previous_generation_is_byte_identical_after_a_new_publish(self) -> None:
        first_result = self._clean_result(seed=12345)
        second_result = self._clean_result(seed=54321)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "publication"
            first = export_static_data(
                first_result, output_dir=output, generated_at_utc="2026-08-27T00:00:00+00:00"
            )
            first_version = output / "versions" / first["publication_generation"]
            before = {path.name: path.read_bytes() for path in first_version.iterdir()}

            second = export_static_data(
                second_result, output_dir=output, generated_at_utc="2026-08-27T01:00:00+00:00"
            )
            after = {path.name: path.read_bytes() for path in first_version.iterdir()}
            self.assertEqual(after, before, "A new publish must never rewrite an older generation")
            self.assertNotEqual(first["publication_generation"], second["publication_generation"])
            # Only the pointer moved; both versions remain independently valid.
            self.assertEqual(
                validate_published_directory(output)["publication_generation"],
                second["publication_generation"],
            )
            self.assertEqual(
                validate_published_directory(first_version)["publication_generation"],
                first["publication_generation"],
            )

    def test_publication_records_the_owning_source_repository(self) -> None:
        result = self._clean_result()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "publication"
            manifest = export_static_data(result, output_dir=output, generated_at_utc="2026-08-27T00:00:00+00:00")
            version = self._version_dir(output)
            metadata = json.loads((version / "metadata.json").read_text())
            self.assertEqual(manifest["schema_version"], PUBLICATION_SCHEMA_VERSION)
            self.assertEqual(metadata["schema_version"], PUBLICATION_SCHEMA_VERSION)
            self.assertEqual(manifest["source_repository"], SOURCE_REPOSITORY)
            self.assertEqual(metadata["source_repository"], SOURCE_REPOSITORY)

    def test_published_json_never_leaks_local_filesystem_paths(self) -> None:
        """A freshly generated publication is free of machine-local paths.

        The public contract is served to browsers, so a calibration source is
        addressed by its stable logical name rather than by wherever the
        generating machine happened to keep it.  Historical publications and
        the legacy fixtures are immutable and deliberately not scanned here.
        """

        repository_root = Path(__file__).resolve().parents[1]
        forbidden = ("/Users/", "/home/", "file://", "C:\\Users", "C:/Users")
        result = self._clean_result()
        with tempfile.TemporaryDirectory() as tmp:
            # Case 1: the production layout, calibrated from the repository.
            repository_output = Path(tmp) / "from-repository"
            export_static_data(
                result,
                output_dir=repository_output,
                generated_at_utc="2026-08-27T00:00:00+00:00",
                calibration_dir=repository_root / "data" / "processed",
            )
            calibration = json.loads(
                (self._version_dir(repository_output) / "calibration.json").read_text()
            )
            self.assertEqual(calibration["status"], "AVAILABLE_IF_ARTIFACTS_EXIST")
            self.assertEqual(
                {key: entry["path"] for key, entry in calibration["source_files"].items()},
                {
                    "seat_hindcast": "data/processed/seat_hindcasts/seat_hindcast_summary.json",
                    "vote_share_hindcast": "data/processed/vote_share_calibration/vote_share_summary_2018_2022.json",
                    "pop_head_to_head": "data/processed/pop_baseline_benchmark/benchmark_report.json",
                },
            )
            for entry in calibration["source_files"].values():
                self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")

            # Case 2: calibration artifacts outside the repository must not be
            # serialised by their absolute location either.
            outside_root = Path(tmp) / "outside" / "processed"
            for relative in (
                ("seat_hindcasts", "seat_hindcast_summary.json"),
                ("vote_share_calibration", "vote_share_summary_2018_2022.json"),
                ("pop_baseline_benchmark", "benchmark_report.json"),
            ):
                artifact = outside_root.joinpath(*relative)
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_text(json.dumps({"summary": {"stub": True}}), encoding="utf-8")
            outside_output = Path(tmp) / "from-outside"
            export_static_data(
                result,
                output_dir=outside_output,
                generated_at_utc="2026-08-27T00:00:00+00:00",
                calibration_dir=outside_root,
            )
            outside_calibration = json.loads(
                (self._version_dir(outside_output) / "calibration.json").read_text()
            )
            self.assertEqual(
                {key: entry["path"] for key, entry in outside_calibration["source_files"].items()},
                {
                    "seat_hindcast": "data/processed/seat_hindcasts/seat_hindcast_summary.json",
                    "vote_share_hindcast": "data/processed/vote_share_calibration/vote_share_summary_2018_2022.json",
                    "pop_head_to_head": "data/processed/pop_baseline_benchmark/benchmark_report.json",
                },
            )

            for output in (repository_output, outside_output):
                published = sorted(output.rglob("*.json"))
                self.assertTrue(published)
                for published_file in published:
                    payload = published_file.read_text(encoding="utf-8")
                    for needle in (*forbidden, str(repository_root), tmp):
                        self.assertNotIn(
                            needle,
                            payload,
                            f"{published_file.name} leaks a local path fragment: {needle}",
                        )

    def test_historical_schema_1_0_publications_remain_valid(self) -> None:
        """A pre-extraction 1.0 version validates and means the old repository."""

        legacy = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "legacy_flat_publication_2026_08_27"
        contracts = {
            name: json.loads((legacy / name).read_text())
            for name in ("forecast.json", "parties.json", "seats.json", "groups.json", "calibration.json", "metadata.json")
        }
        self.assertEqual(contracts["metadata.json"]["schema_version"], "1.0")
        self.assertNotIn("source_repository", contracts["metadata.json"])
        self.assertEqual(
            resolve_source_repository(contracts["metadata.json"].get("source_repository")),
            "edvinli/edvinli.github.io",
        )
        # The 1.0 fixture is uncertified and lacks the pointer-era fields, so
        # it must fail the certified contract while staying readable as 1.0.
        with self.assertRaises(ValueError):
            validate_publication_contract(contracts)

    def test_generated_static_output_does_not_dirty_source_provenance(self) -> None:
        generated_status = SimpleNamespace(stdout="?? files/election-simulator\n?? files/election-simulator/versions/abc/forecast.json\n?? files/.election-simulator.versions/abc/seats.json\n")
        with patch("scripts.simulator.reproducibility.subprocess.run", return_value=generated_status):
            self.assertTrue(is_git_worktree_clean(Path(".")))
        for changed_path in ("scripts/changed.py", "tests/changed.py", "data/processed/input.csv"):
            with patch(
                "scripts.simulator.reproducibility.subprocess.run",
                return_value=SimpleNamespace(stdout=f"?? {changed_path}\n"),
            ):
                self.assertFalse(is_git_worktree_clean(Path(".")))


if __name__ == "__main__":
    unittest.main()
