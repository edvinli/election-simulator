"""Offline contract tests for the prospective Botten Ada capture layer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.prospective_benchmark_2026.botten_ada_capture import (
    ADA_REPOSITORY_COMMIT,
    DEFAULT_SOURCE_SPECS,
    RDS_URL,
    STATUS_PARITY_UNVERIFIED,
    STATUS_PARITY_VERIFIED,
    STATUS_PARSE_FAILED,
    STATUS_SOURCE_UNAVAILABLE,
    BottenAdaCapture,
    BottenAdaDrawsNotVerified,
    SourceArtifact,
    capture_botten_ada,
    parse_forecast_json,
    parse_public_bundle,
    parity_evaluate,
    require_verified_draws,
    verify_official_draws,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "prospective_benchmark_2026" / "botten_ada"


class TestProspectiveBottenAdaCapture(unittest.TestCase):
    def _fixture_fetcher(self, *, broken: str | None = None, outage: str | None = None):
        fixture_names = {
            "forecast": "forecast.json",
            "latest_polls": "latest_polls.json",
            "timeseries": "timeseries.csv",
            "threshold_L": "threshold_L.json",
            "threshold_C": "threshold_C.json",
            "threshold_KD": "threshold_KD.json",
            "threshold_MP": "threshold_MP.json",
            "homepage": "homepage.html",
        }
        payloads = {
            key: (FIXTURE_ROOT / name).read_bytes()
            for key, name in fixture_names.items()
        }

        def fetch(url: str, *, head_only: bool = False) -> SourceArtifact:
            key = next((key for key, spec in DEFAULT_SOURCE_SPECS.items() if spec.url == url), None)
            if key == "rds":
                return SourceArtifact(
                    url=url,
                    body=None,
                    retrieved_at_utc="2026-09-04T21:31:00Z",
                    status_code=200,
                    method="HEAD",
                    headers={
                        "ETag": '"fixture-rds-etag"',
                        "Last-Modified": "Thu, 30 Apr 2026 04:33:45 GMT",
                        "Content-Length": "1685140142",
                        "Content-Type": "binary/octet-stream",
                    },
                )
            if key is None:
                raise AssertionError(f"unexpected source URL: {url}")
            if key == outage:
                return SourceArtifact(
                    url=url,
                    body=None,
                    retrieved_at_utc="2026-09-04T21:31:00Z",
                    status_code=503,
                    error="fixture outage",
                )
            body = b"not json" if key == broken else payloads[key]
            return SourceArtifact(
                url=url,
                body=body,
                retrieved_at_utc="2026-09-04T21:31:01Z",
                status_code=200,
                headers={
                    "ETag": f'"fixture-{key}"',
                    "Last-Modified": "Fri, 04 Sep 2026 21:30:00 GMT",
                    "Content-Type": "text/csv" if key == "timeseries" else "application/json",
                },
            )

        return fetch

    def test_capture_is_json_safe_and_keeps_exact_raw_bytes_and_transport_metadata(self) -> None:
        capture = capture_botten_ada(
            "2026-09-04T23:30:00+02:00",
            fetcher=self._fixture_fetcher(),
        )
        self.assertIsInstance(capture, BottenAdaCapture)
        self.assertEqual(capture.record["status"], STATUS_PARITY_UNVERIFIED)
        json.dumps(capture.jsonable(), allow_nan=False)
        self.assertIn("raw/latest_forecast_seats--all.json", capture.raw_files)
        self.assertIn("raw/latest_pop_timeseries.csv", capture.raw_files)
        self.assertEqual(
            capture.record["provenance"]["sources"]["forecast"]["content_sha256"],
            hashlib.sha256(capture.raw_files["raw/latest_forecast_seats--all.json"]).hexdigest(),
        )
        self.assertEqual(
            capture.record["provenance"]["sources"]["forecast"]["last_modified_utc"],
            "2026-09-04T21:30:00Z",
        )
        self.assertEqual(
            capture.record["provenance"]["sources"]["forecast"]["source_reported_update"]["run"],
            "fixture_fc2026",
        )
        self.assertFalse(capture.record["capabilities"]["verified_predictive_vote_draws"])
        self.assertEqual(capture.record["provenance"]["sources"]["rds"]["byte_size"], 1685140142)
        self.assertIsNone(capture.record["provenance"]["sources"]["rds"]["content_sha256"])
        self.assertNotIn("raw/pop.rds", capture.raw_files)

    def test_standalone_writer_refuses_overwrite(self) -> None:
        capture = capture_botten_ada(
            "2026-09-04T23:30:00+02:00",
            fetcher=self._fixture_fetcher(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "botten_ada"
            capture.write(destination)
            with self.assertRaises(FileExistsError):
                capture.write(destination)
            self.assertEqual(
                (destination / "raw/latest_forecast_seats--all.json").read_bytes(),
                capture.raw_files["raw/latest_forecast_seats--all.json"],
            )

    def test_machine_parser_exposes_election_forecast_and_does_not_mix_timeseries(self) -> None:
        forecast = parse_forecast_json((FIXTURE_ROOT / "forecast.json").read_bytes())
        self.assertEqual(forecast["metadata"]["model"], "model8m10")
        self.assertEqual(forecast["election"]["L"]["votes"]["p50"], 0.031)
        self.assertEqual(forecast["election"]["L"]["seats"]["p50"], 0.0)
        capture = capture_botten_ada(
            "2026-09-04T23:30:00+02:00",
            fetcher=self._fixture_fetcher(),
        )
        # The latest_pop CSV is retained independently, even though it also
        # has an election-date row.  Its values are not substituted for the
        # latest_forecast election JSON.
        self.assertNotEqual(
            capture.record["forecast"]["election"]["L"]["votes"]["p50"],
            capture.record["timeseries"]["election_day_row"]["parties"]["L"]["p50"],
        )

    def test_source_outage_is_explicit_and_never_carries_forward_a_previous_value(self) -> None:
        good = capture_botten_ada(
            "2026-09-04T23:30:00+02:00",
            fetcher=self._fixture_fetcher(),
        )
        outage = capture_botten_ada(
            "2026-09-05T23:30:00+02:00",
            fetcher=self._fixture_fetcher(outage="forecast"),
        )
        self.assertEqual(outage.record["status"], STATUS_SOURCE_UNAVAILABLE)
        self.assertIsNone(outage.record["forecast"])
        self.assertIn("forecast", outage.record["errors"])
        self.assertNotEqual(
            outage.record["capture"]["benchmark_cutoff"],
            good.record["capture"]["benchmark_cutoff"],
        )
        # An unavailable source has no copied forecast, even if a caller has a
        # successful prior capture in memory.
        self.assertIsNot(outage.record.get("forecast"), good.record.get("forecast"))

    def test_malformed_source_is_parse_failed_and_keeps_diagnostics(self) -> None:
        capture = capture_botten_ada(
            "2026-09-04T23:30:00+02:00",
            fetcher=self._fixture_fetcher(broken="forecast"),
        )
        self.assertEqual(capture.record["status"], STATUS_PARSE_FAILED)
        self.assertIsNone(capture.record["forecast"])
        self.assertIn("forecast", capture.record["errors"])
        self.assertIn("raw/latest_forecast_seats--all.json", capture.raw_files)

    def test_explicit_freshness_policy_marks_stale_without_guessing_a_threshold(self) -> None:
        capture = capture_botten_ada(
            "2026-09-04T23:30:00+02:00",
            fetcher=self._fixture_fetcher(),
            stale_before="2026-09-05T00:00:00Z",
        )
        self.assertEqual(capture.record["status"], "SOURCE_STALE")
        self.assertEqual(capture.record["provenance"]["freshness_policy"]["status"], "SOURCE_STALE")
        self.assertIn("freshness", capture.record["errors"])

    def test_parity_is_a_draw_only_check_and_inclusive_threshold(self) -> None:
        draws = np.array(
            [
                [0.04, 0.04, 0.064, 0.053, 0.314, 0.083, 0.064, 0.195],
                [0.16, 0.032, 0.065, 0.054, 0.300, 0.084, 0.065, 0.190],
                [0.178, 0.030, 0.063, 0.052, 0.320, 0.082, 0.063, 0.200],
                [0.20, 0.031, 0.064, 0.053, 0.314, 0.083, 0.064, 0.230],
            ],
            dtype=float,
        )
        published = {
            "metadata": {"n_draws": 4},
            "election": {
                party: {
                    "votes": {
                        quantile: float(np.quantile(draws[:, index], level))
                        for quantile, level in (("p5", 0.05), ("p50", 0.50), ("p95", 0.95))
                    }
                }
                for index, party in enumerate(("M", "L", "C", "KD", "S", "V", "MP", "SD"))
            },
        }
        thresholds = {
            "L": {"probability": 0.25},  # exactly 4.0% is inclusive
            "C": {"probability": 1.0},
        }
        parity = parity_evaluate(
            draws,
            published,
            threshold_probabilities=thresholds,
            expected_n_draws=4,
            tolerance=1e-12,
        )
        self.assertEqual(parity["status"], STATUS_PARITY_VERIFIED)
        self.assertTrue(parity["eligible_for_probabilistic_scoring"])
        self.assertEqual(parity["thresholds"]["L"]["draw_probability"], 0.25)
        bad = parity_evaluate(draws, published, expected_n_draws=5)
        self.assertEqual(bad["status"], STATUS_PARITY_UNVERIFIED)
        self.assertFalse(bad["eligible_for_probabilistic_scoring"])

    def test_rds_posterior_samples_are_not_eligible_without_semantic_evidence(self) -> None:
        capture = capture_botten_ada(
            "2026-09-04T23:30:00+02:00",
            fetcher=self._fixture_fetcher(),
        )
        published = capture.record["forecast"]
        draws = np.tile(
            [0.178, 0.031, 0.064, 0.053, 0.314, 0.083, 0.064, 0.195],
            (4, 1),
        )
        parity = verify_official_draws(
            draws,
            draw_provenance={
                "source_url": RDS_URL,
                "draw_role": "posterior_draws",
                "semantic_evidence_reference": "fixture:publisher-data-page",
            },
            published_forecast=published,
        )
        self.assertEqual(parity["status"], STATUS_PARITY_UNVERIFIED)
        with self.assertRaises(BottenAdaDrawsNotVerified):
            require_verified_draws(parity)

    def test_verified_draw_gate_requires_rds_url_role_and_reference(self) -> None:
        capture = capture_botten_ada(
            "2026-09-04T23:30:00+02:00",
            fetcher=self._fixture_fetcher(),
        )
        draws = np.array(
            [[0.178, 0.031, 0.064, 0.053, 0.314, 0.083, 0.064, 0.195]] * 4,
            dtype=float,
        )
        for provenance in (
            {},
            {"source_url": "https://example.test/pop.rds", "draw_role": "election_day_predictive_draws", "semantic_evidence_reference": "x"},
            {"source_url": RDS_URL, "draw_role": "election_day_predictive_draws"},
        ):
            parity = verify_official_draws(
                draws,
                draw_provenance=provenance,
                published_forecast=capture.record["forecast"],
            )
            self.assertEqual(parity["status"], STATUS_PARITY_UNVERIFIED)

    def test_missing_required_source_is_not_a_zero_forecast(self) -> None:
        artifacts = {
            "forecast": SourceArtifact(
                url=DEFAULT_SOURCE_SPECS["forecast"].url,
                body=None,
                retrieved_at_utc="2026-09-04T21:31:00Z",
                status_code=503,
                error="fixture outage",
            )
        }
        record, raw_files = parse_public_bundle(artifacts)
        self.assertEqual(record["status"], STATUS_SOURCE_UNAVAILABLE)
        self.assertIsNone(record["forecast"])
        self.assertEqual(raw_files, {})
        self.assertNotEqual(record["capabilities"]["published_central_predictions"], True)


if __name__ == "__main__":
    unittest.main()
