"""Hostile, offline tests for the 2026 prospective benchmark boundary.

These tests intentionally exercise evidence that has been made internally
consistent by an attacker (or by a later repair) but is not sufficiently tied
to the frozen protocol, the certified source, or the public artifact.  They
are kept separate from the model-specific unit tests so a methods review can
be read and maintained independently.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.prospective_benchmark_2026.archive import (
    ArchiveValidationError,
    ModelCapture,
    append_capture,
    canonical_json_bytes,
)
from scripts.prospective_benchmark_2026.botten_ada_capture import (
    DEFAULT_SOURCE_SPECS,
    RDS_URL,
    BottenAdaCaptureError,
    SourceArtifact,
    fetch_source,
    parity_evaluate,
    parse_public_bundle,
    verify_official_draws,
)
from scripts.prospective_benchmark_2026.report import ReportError, _score_capture, _verified_draws
from scripts.prospective_benchmark_2026.results import OfficialResultError, PARTY_ORDER, load_official_result
from scripts.prospective_benchmark_2026.time_rules import capture_id_for_date, classify_capture_time


PARTIES = ("M", "L", "C", "KD", "S", "V", "MP", "SD")


def _published_forecast(value: float = 0.1, *, n_draws: int | None = 4) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if n_draws is not None:
        metadata["n_draws"] = n_draws
    return {
        "metadata": metadata,
        "election": {
            party: {
                "votes": {"p5": value, "p50": value, "p95": value},
            }
            for party in PARTIES
        },
    }


class TestBottenEvidenceBoundary(unittest.TestCase):
    def test_default_parity_tolerance_is_not_looser_than_frozen_protocol(self) -> None:
        # A 0.07 percentage-point discrepancy is larger than the frozen
        # 0.051-point allowance.  It must not be certified merely because the
        # display values happen to be rounded to one decimal place.
        draws = np.full((4, len(PARTIES)), 0.1007, dtype=float)
        parity = parity_evaluate(draws, _published_forecast())
        self.assertNotEqual(
            parity["status"],
            "VERIFIED",
            "parity accepted an error larger than protocol.json permits",
        )

    def test_verified_draws_require_hashable_raw_artifact_provenance(self) -> None:
        draws = np.full((4, len(PARTIES)), 0.1, dtype=float)
        # URL/role/reference alone do not bind these arrays to the official
        # contemporaneous RDS bytes.  A later scorer must not be able to mark
        # an arbitrary local matrix as official with three free-form strings.
        parity = verify_official_draws(
            draws,
            draw_provenance={
                "source_url": RDS_URL,
                "draw_role": "election_day_predictive_draws",
                "semantic_evidence_reference": "fixture:official-data-page",
            },
            published_forecast=_published_forecast(),
        )
        self.assertNotEqual(parity["status"], "VERIFIED")

    def test_source_artifact_url_must_be_one_of_the_preregistered_urls(self) -> None:
        # A parser caller can otherwise feed bytes from an unrelated endpoint
        # while retaining a plausible JSON shape and a successful status code.
        artifact = SourceArtifact(
            url="https://attacker.invalid/latest_forecast/seats--all.json",
            body=json.dumps({
                "metadata": {"election_day": "2026-09-13"},
                "election": {
                    party: {"votes": {"p5": 0.1, "p50": 0.1, "p95": 0.1}}
                    for party in PARTIES
                },
            }).encode(),
            retrieved_at_utc="2026-09-04T21:31:00Z",
            status_code=200,
        )
        with self.assertRaises(BottenAdaCaptureError):
            parse_public_bundle({"forecast": artifact})

    def test_malformed_content_length_is_a_durable_source_failure_not_an_uncaught_value_error(self) -> None:
        class FakeResponse:
            status = 200
            headers = {"Content-Length": "not-an-integer"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _size: int) -> bytes:
                return b"{}"

        with patch("scripts.prospective_benchmark_2026.botten_ada_capture.urlopen", return_value=FakeResponse()):
            artifact = fetch_source(DEFAULT_SOURCE_SPECS["forecast"].url)
        self.assertIsNone(artifact.body)
        self.assertIsNotNone(artifact.error)


class TestArchiveAppendBoundary(unittest.TestCase):
    def _archive(self, root: Path) -> Path:
        archive = root / "archive"
        archive.mkdir()
        protocol = b'{"frozen":true}\n'
        digest = hashlib.sha256(protocol).hexdigest()
        (archive / "protocol.json").write_bytes(protocol)
        (archive / "protocol.sha256").write_text(f"{digest}  protocol.json\n", encoding="utf-8")
        (archive / "index.json").write_bytes(canonical_json_bytes({
            "schema_version": "1.0",
            "protocol_path": "protocol.json",
            "protocol_sha256": digest,
            "captures": [],
        }))
        return archive

    def _models(self, marker: str) -> dict[str, ModelCapture]:
        return {
            "election_simulator": ModelCapture(
                status="AVAILABLE",
                forecast={"system": "election_simulator", "marker": marker},
                provenance={"source": "fixture"},
                files={"source_snapshot.json": b"snapshot"},
            ),
            "botten_ada": ModelCapture(
                status="PARITY_UNVERIFIED",
                forecast={"system": "botten_ada", "marker": marker},
                provenance={"source": "fixture"},
                files={"raw/source.json": b"source"},
            ),
        }

    def test_append_refuses_to_extend_a_corrupt_index_before_installing_new_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._archive(Path(tmp))
            first_day = "2026-09-04"
            append_capture(
                root=root,
                capture_id=capture_id_for_date(first_day),
                timing=classify_capture_time(first_day, "2026-09-04T21:31:00Z", durable=True).to_dict(),
                models=self._models("first"),
            )
            index_path = root / "index.json"
            tampered = json.loads(index_path.read_text(encoding="utf-8"))
            tampered["captures"][0]["entry_sha256"] = "0" * 64
            index_path.write_bytes(canonical_json_bytes(tampered))
            before = index_path.read_bytes()

            with self.assertRaises(ArchiveValidationError):
                append_capture(
                    root=root,
                    capture_id=capture_id_for_date("2026-09-05"),
                    timing=classify_capture_time("2026-09-05", "2026-09-05T21:31:00Z", durable=True).to_dict(),
                    models=self._models("must-not-install"),
                )
            self.assertEqual(index_path.read_bytes(), before)
            self.assertFalse((root / "captures" / capture_id_for_date("2026-09-05")).exists())


class TestCaptureTimeInputBoundary(unittest.TestCase):
    def test_cli_has_no_user_supplied_retrieval_timestamp(self) -> None:
        command = [
            sys.executable,
            "-m",
            "scripts.prospective_benchmark_2026",
            "capture",
            "--mode",
            "dry_run",
            "--scheduled-date",
            "2026-09-04",
            "--retrieved-at",
            "2026-09-04T21:31:00Z",
        ]
        rejected = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(rejected.returncode, 2)

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.prospective_benchmark_2026",
                "capture",
                "--help",
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertNotIn("--retrieved-at", completed.stdout)


class TestReportDrawTrustBoundary(unittest.TestCase):
    def test_report_does_not_score_a_draw_file_without_verified_sidecar_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp) / "capture"
            model_dir = capture / "election_simulator"
            model_dir.mkdir(parents=True)
            np.savez(model_dir / "draws.npz", vote_shares_pct=np.ones((2, 8), dtype=np.float64))
            forecast = {
                "system": "election_simulator",
                "draws": {
                    "status": "VERIFIED",
                    "verified_predictive_vote_draws": True,
                    "path": "draws.npz",
                },
            }
            with self.assertRaises(ReportError):
                _verified_draws(capture, forecast)

    def test_report_rejects_available_forecast_with_proportion_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp) / "capture"
            for model in ("election_simulator", "botten_ada"):
                directory = capture / model
                directory.mkdir(parents=True)
                forecast = {
                    "system": model,
                    "available": True,
                    "election_date": "2026-09-13",
                    "party_order": list(PARTIES),
                    # This must never be interpreted as percentage points by
                    # the report scorer.
                    "vote_share_unit": "proportion",
                    "vote_share_denominator": "official_national_valid_votes",
                    "published_central_prediction": None,
                    "published_quantiles": None,
                    "threshold_probabilities_4pct": {},
                    "draws": {"verified_predictive_vote_draws": False, "path": None},
                }
                (directory / "forecast.json").write_text(json.dumps(forecast), encoding="utf-8")
            # _score_capture only needs the actual result shares for this
            # contract check; the point is that it must fail before scoring.
            from scripts.prospective_benchmark_2026.results import OfficialResult

            result = OfficialResult(
                manifest_path=Path(tmp) / "result.json",
                manifest_sha256="1" * 64,
                raw_path=Path(tmp) / "raw.json",
                raw_sha256="0" * 64,
                official_source_url="https://resultat.val.se/final",
                retrieved_at_utc="2026-09-30T12:00:00Z",
                valid_national_votes=1,
                vote_shares={party: 0.0 for party in PARTIES},
                votes={party: 0 for party in PARTIES},
                seats={party: 0 for party in PARTIES},
            )
            with self.assertRaises(ReportError):
                _score_capture(capture, {"capture_id": "fixture", "scheduled_date": "2026-09-12"}, result)


class TestOfficialResultIdentityBoundary(unittest.TestCase):
    def _manifest(self, root: Path, *, url: str, election_date: str = "2026-09-13") -> Path:
        raw = root / "official.json"
        raw.write_bytes(b"final result fixture\n")
        denominator = 1_000_000
        counts = {party: 100_000 - index * 5_000 for index, party in enumerate(PARTY_ORDER)}
        payload = {
            "schema_version": "1.0",
            "authority": "Valmyndigheten",
            "certification_status": "FINAL_CERTIFIED",
            "election_date": election_date,
            "official_source_url": url,
            "retrieved_at_utc": "2026-09-30T12:00:00Z",
            "raw_path": raw.name,
            "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "valid_national_votes": denominator,
            "parties": {
                party: {
                    "votes": count,
                    "vote_share_percentage_points": 100.0 * count / denominator,
                    "seats": 10 + index,
                }
                for index, (party, count) in enumerate(counts.items())
            },
        }
        path = root / "result-manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_result_source_must_be_an_official_valmyndigheten_host(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(OfficialResultError):
                load_official_result(self._manifest(Path(tmp), url="https://attacker.invalid/final"))

    def test_result_url_with_malformed_port_fails_as_official_result_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(OfficialResultError):
                load_official_result(self._manifest(Path(tmp), url="https://resultat.val.se:not-a-port/final"))

    def test_result_raw_artifact_symlink_is_not_accepted_as_immutable_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._manifest(root, url="https://resultat.val.se/final")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            original = root / "official.json"
            outside = root / "outside.json"
            outside.write_bytes(original.read_bytes())
            original.unlink()
            original.symlink_to(outside)
            manifest["raw_sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(OfficialResultError):
                load_official_result(manifest_path)

    def test_nonfinite_result_share_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._manifest(root, url="https://resultat.val.se/final")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["parties"]["L"]["vote_share_percentage_points"] = float("nan")
            manifest_path.write_text(json.dumps(manifest, allow_nan=True), encoding="utf-8")
            with self.assertRaises(OfficialResultError):
                load_official_result(manifest_path)

    def test_result_must_identify_the_2026_election(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(OfficialResultError):
                load_official_result(
                    self._manifest(Path(tmp), url="https://resultat.val.se/final", election_date="2022-09-11")
                )


class TestPackageEntrypoint(unittest.TestCase):
    def test_required_post_election_module_entrypoint_exists(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.prospective_benchmark_2026", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("capture", result.stdout)
        self.assertIn("score", result.stdout)


if __name__ == "__main__":
    unittest.main()
