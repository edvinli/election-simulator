"""Focused tests for the schema-1.3 representation-only migration."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.presentation_reexport.reexport import (
    DEFAULT_MATRIX_PATH,
    DEFAULT_SOURCE_VERSION,
    EXPECTED_MATRIX_SHA256,
    EXPECTED_PAYLOAD_SHA256,
    SOURCE_GENERATION,
    compare_source_and_reexport,
    migrate_publication,
)
from scripts.simulator.reproducibility import compute_file_sha256
from scripts.static_exporter import validate_publication_version


def _preserved_matrix_available() -> bool:
    """Return whether the optional external audit artifact is available."""

    return (
        DEFAULT_MATRIX_PATH.is_file()
        and compute_file_sha256(DEFAULT_MATRIX_PATH) == EXPECTED_MATRIX_SHA256
    )


class PresentationReexportTests(unittest.TestCase):
    def test_source_validation_is_the_first_gate(self) -> None:
        with patch(
            "scripts.presentation_reexport.reexport.validate_publication_version",
            side_effect=ValueError("source is not certified"),
        ) as validate_source:
            with patch(
                "scripts.presentation_reexport.reexport._load_preserved_matrix",
                side_effect=AssertionError("matrix must not be loaded before source validation"),
            ):
                with self.assertRaisesRegex(ValueError, "source is not certified"):
                    migrate_publication(
                        source_version=DEFAULT_SOURCE_VERSION,
                        matrix_path=DEFAULT_MATRIX_PATH,
                        output_dir=Path(self.enterContext(tempfile.TemporaryDirectory())) / "out",
                    )
        validate_source.assert_called_once_with(
            DEFAULT_SOURCE_VERSION.resolve(),
            expected_generation=SOURCE_GENERATION,
        )

    def test_clean_worktree_guard_is_retained(self) -> None:
        with patch(
            "scripts.presentation_reexport.reexport.is_git_worktree_clean",
            return_value=False,
        ):
            with patch(
                "scripts.presentation_reexport.reexport._load_preserved_matrix",
                side_effect=AssertionError("matrix must not be loaded from a dirty source tree"),
            ):
                with self.assertRaisesRegex(ValueError, "clean source worktree"):
                    migrate_publication(
                        source_version=DEFAULT_SOURCE_VERSION,
                        matrix_path=DEFAULT_MATRIX_PATH,
                        output_dir=Path(self.enterContext(tempfile.TemporaryDirectory())) / "out",
                    )

    @unittest.skipUnless(
        _preserved_matrix_available(),
        "preserved coalition audit matrix is not available on this runner",
    )
    def test_migration_adds_exact_histograms_and_preserves_scientific_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "publication"
            source_before = {
                name: (DEFAULT_SOURCE_VERSION / name).read_bytes()
                for name in (
                    "forecast.json",
                    "parties.json",
                    "seats.json",
                    "groups.json",
                    "calibration.json",
                    "metadata.json",
                    "manifest.json",
                )
            }
            # The repository-wide test suite may leave unrelated generated
            # evidence dirty before this optional artifact test runs.  The
            # dedicated guard test above still covers the production gate;
            # this test scopes the provenance patch to its own integration
            # call so it tests migration semantics rather than test ordering.
            with patch(
                "scripts.presentation_reexport.reexport.is_git_worktree_clean",
                return_value=True,
            ):
                report = migrate_publication(
                    source_version=DEFAULT_SOURCE_VERSION,
                    matrix_path=DEFAULT_MATRIX_PATH,
                    output_dir=output,
                    generated_at_utc="2026-08-28T17:00:00+00:00",
                )
            generation = report["generation"]
            version = output / "versions" / generation
            self.assertEqual(report["status"], "PUBLISHED")
            self.assertEqual(report["source_generation"], SOURCE_GENERATION)
            self.assertEqual(report["matrix_sha256"], EXPECTED_MATRIX_SHA256)
            self.assertEqual(report["deterministic_payload_sha256"], EXPECTED_PAYLOAD_SHA256)
            self.assertEqual(report["all_256_audit"]["complement_audit"], "256/256 PASS")
            self.assertEqual(report["recursive_comparison"]["status"], "PASS")
            self.assertEqual(report["recursive_comparison"]["scientific_changes"], [])
            self.assertTrue(report["recursive_comparison"]["payload_sha256_unchanged"])

            self.assertEqual(
                {path.name for path in version.iterdir()},
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
            manifest = validate_publication_version(version, expected_generation=generation)
            self.assertEqual(manifest["deterministic_payload_sha256"], EXPECTED_PAYLOAD_SHA256)
            self.assertEqual(manifest["source_git_commit"], report["source_git_commit"])
            self.assertTrue(manifest["source_worktree_clean"])

            source_groups = json.loads((DEFAULT_SOURCE_VERSION / "groups.json").read_text())
            migrated_groups = json.loads((version / "groups.json").read_text())
            source_forecast = json.loads((DEFAULT_SOURCE_VERSION / "forecast.json").read_text())
            migrated_forecast = json.loads((version / "forecast.json").read_text())
            self.assertEqual(
                migrated_forecast["deterministic_payload_sha256"],
                source_forecast["deterministic_payload_sha256"],
            )
            self.assertEqual(
                migrated_groups["groups"],
                source_groups["groups"],
                "named groups and their existing summaries/histograms must be cloned",
            )
            builder = migrated_groups["coalition_builder"]
            self.assertEqual(len(builder["coalitions"]), 256)
            for mask in range(256):
                entry = builder["coalitions"][str(mask)]
                self.assertEqual(list(entry)[-1], "seat_histogram")
                self.assertEqual(list(entry["seat_histogram"]), ["min_seats", "counts"])
                self.assertEqual(sum(entry["seat_histogram"]["counts"]), 100_000)

            self.assertEqual(
                report["all_256_audit"]["spot_checks"],
                {
                    "84": {
                        "parties": ["C", "S", "MP"],
                        "majority_count": 10_778,
                        "prob_majority": 0.10778,
                        "min_seats": 141,
                        "max_seats": 190,
                    },
                    "112": {
                        "parties": ["S", "V", "MP"],
                        "majority_count": 2_216,
                        "prob_majority": 0.02216,
                        "min_seats": 144,
                        "max_seats": 188,
                    },
                    "139": {
                        "parties": ["M", "L", "KD", "SD"],
                        "majority_count": 286,
                        "prob_majority": 0.00286,
                        "min_seats": 131,
                        "max_seats": 179,
                    },
                },
            )

            for name, before in source_before.items():
                self.assertEqual((DEFAULT_SOURCE_VERSION / name).read_bytes(), before)

    @unittest.skipUnless(
        _preserved_matrix_available(),
        "preserved coalition audit matrix is not available on this runner",
    )
    def test_comparison_rejects_a_scientific_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "publication"
            with patch(
                "scripts.presentation_reexport.reexport.is_git_worktree_clean",
                return_value=True,
            ):
                migrate_publication(
                    source_version=DEFAULT_SOURCE_VERSION,
                    matrix_path=DEFAULT_MATRIX_PATH,
                    output_dir=output,
                    generated_at_utc="2026-08-28T17:01:00+00:00",
                )
            pointer = json.loads((output / "current.json").read_text())
            generation = pointer["publication_generation"]
            source_contracts = {
                name: json.loads((DEFAULT_SOURCE_VERSION / name).read_text())
                for name in (
                    "forecast.json",
                    "parties.json",
                    "seats.json",
                    "groups.json",
                    "calibration.json",
                    "metadata.json",
                )
            }
            reexport_contracts = {
                name: json.loads((output / "versions" / generation / name).read_text())
                for name in source_contracts
            }
            reexport_contracts["forecast.json"]["as_of"] = "2099-01-01"
            source_manifest = json.loads((DEFAULT_SOURCE_VERSION / "manifest.json").read_text())
            reexport_manifest = json.loads((output / "versions" / generation / "manifest.json").read_text())
            with self.assertRaisesRegex(ValueError, "scientific or unapproved"):
                compare_source_and_reexport(
                    source_contracts,
                    reexport_contracts,
                    source_manifest,
                    reexport_manifest,
                )


if __name__ == "__main__":
    unittest.main()
