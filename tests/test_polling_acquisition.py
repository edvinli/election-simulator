from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.election_automation import _source_provenance, refresh_polling_snapshot
from scripts.pollofpolls.acquire import (
    AcquisitionError,
    _record,
    _validate_source_payload,
    acquire_all,
    sha256_bytes,
)
from scripts.pollofpolls.config import Source


FIXTURES = Path(__file__).parent / "fixtures"


def _response(url: str, *, content_type: str, status: int = 200) -> dict:
    return {
        "http_status": status,
        "final_url": url,
        "http_headers": {"content-type": content_type},
    }


class SemanticAcquisitionTests(unittest.TestCase):
    @staticmethod
    def _source(kind: str, filename: str, *, party: str | None = None) -> Source:
        return Source(
            key="party_M" if kind == "party_chart" else kind,
            url=(
                "http://pollofpolls.se/poll_img/data_big_2.csv"
                if kind == "party_chart"
                else "http://pollofpolls.se/"
            ),
            raw_filename=filename,
            kind=kind,
            party=party,
            page_url="http://pollofpolls.se/",
        )

    @staticmethod
    def _old_manifest(source: Source, payload: bytes) -> dict:
        record = _record(
            source,
            payload,
            _response(source.url, content_type="text/html"),
            "first_party_http",
            source.url,
            None,
        )
        return {"schema_version": 1, "sources": {source.key: record}}

    def test_invalid_200_homepage_is_rejected_and_fallback_is_used(self) -> None:
        source = self._source("homepage", "homepage.html")
        valid = (FIXTURES / "homepage.html").read_bytes()
        invalid = b"<!doctype html><html><body>access denied</body></html>"

        def fake_read(url: str, _timeout: float):
            if url == source.url:
                return invalid, _response(url, content_type="text/html")
            self.assertEqual(url, source.archive_url)
            return valid, _response(url, content_type="text/html")

        with tempfile.TemporaryDirectory() as tmp, patch(
            "scripts.pollofpolls.acquire.SOURCES", (source,)
        ), patch("scripts.pollofpolls.acquire._read_url", fake_read):
            manifest, _messages = acquire_all(Path(tmp))
            self.assertEqual((Path(tmp) / source.raw_filename).read_bytes(), valid)

        diagnostics = manifest["acquisition_diagnostics"]
        self.assertEqual(
            [item["semantic_validation"] for item in diagnostics], ["FAIL", "PASS"]
        )
        self.assertEqual(
            [item["retrieval_method"] for item in diagnostics],
            ["first_party_http", "wayback_preserved_first_party_response"],
        )
        self.assertEqual(manifest["sources"][source.key]["retrieval_method"], diagnostics[-1]["retrieval_method"])
        self.assertEqual(
            _source_provenance(manifest, []), "VERIFIED_STALE_FALLBACK"
        )
        self.assertTrue(all("payload" not in item for item in diagnostics))

    def test_valid_200_homepage_is_direct_live_without_archive_attempt(self) -> None:
        source = self._source("homepage", "homepage.html")
        valid = (FIXTURES / "homepage.html").read_bytes()
        calls: list[str] = []

        def fake_read(url: str, _timeout: float):
            calls.append(url)
            self.assertEqual(url, source.url)
            return valid, _response(url, content_type="text/html")

        with tempfile.TemporaryDirectory() as tmp, patch(
            "scripts.pollofpolls.acquire.SOURCES", (source,)
        ), patch("scripts.pollofpolls.acquire._read_url", fake_read):
            manifest, messages = acquire_all(Path(tmp))

        self.assertEqual(calls, [source.url])
        self.assertEqual(
            manifest["sources"][source.key]["retrieval_method"], "first_party_http"
        )
        self.assertEqual(_source_provenance(manifest, messages), "DIRECT_LIVE_FETCH")
        self.assertEqual(manifest["acquisition_diagnostics"][0]["semantic_validation"], "PASS")
        self.assertFalse(manifest["acquisition_diagnostics"][0]["retained_previous"])

    def test_semantic_homepage_failure_does_not_suppress_later_party_live_attempt(self) -> None:
        homepage = self._source("homepage", "homepage.html")
        party = self._source("party_chart", "party_M.csv", party="M")
        valid_homepage = (FIXTURES / "homepage.html").read_bytes()
        valid_party = (FIXTURES / "party_M.csv").read_bytes()
        invalid_homepage = b"<html><body>homepage block</body></html>"
        calls: list[str] = []

        def fake_read(url: str, _timeout: float):
            calls.append(url)
            if url == homepage.url:
                return invalid_homepage, _response(url, content_type="text/html")
            if url == homepage.archive_url:
                return valid_homepage, _response(url, content_type="text/html")
            if url == party.url:
                return valid_party, _response(url, content_type="text/csv")
            self.fail(f"unexpected archive request after source-scoped semantic failure: {url}")

        with tempfile.TemporaryDirectory() as tmp, patch(
            "scripts.pollofpolls.acquire.SOURCES", (homepage, party)
        ), patch("scripts.pollofpolls.acquire._read_url", fake_read):
            manifest, _messages = acquire_all(Path(tmp))

        self.assertIn(party.url, calls)
        self.assertNotIn(party.archive_url, calls)
        self.assertEqual(
            manifest["sources"][party.key]["retrieval_method"], "first_party_http"
        )
        homepage_diagnostics = [
            item for item in manifest["acquisition_diagnostics"] if item["source_key"] == homepage.key
        ]
        self.assertEqual(
            [item["semantic_validation"] for item in homepage_diagnostics], ["FAIL", "PASS"]
        )

    def test_invalid_200_timeseries_csv_is_rejected_before_acceptance(self) -> None:
        source = self._source("timeseries", "timeseries.dat")
        valid = (FIXTURES / "pollofpolls_timeseries.html").read_bytes()
        invalid = b"Datum,M\n2026-09-01,20\n"

        def fake_read(url: str, _timeout: float):
            if url == source.url:
                return invalid, _response(url, content_type="text/csv")
            return valid, _response(url, content_type="text/html")

        with tempfile.TemporaryDirectory() as tmp, patch(
            "scripts.pollofpolls.acquire.SOURCES", (source,)
        ), patch("scripts.pollofpolls.acquire._read_url", fake_read):
            manifest, _messages = acquire_all(Path(tmp))
            self.assertEqual((Path(tmp) / source.raw_filename).read_bytes(), valid)
        diagnostics = manifest["acquisition_diagnostics"]
        self.assertEqual(diagnostics[0]["semantic"], "FAIL")
        self.assertIn("missing required party columns", diagnostics[0]["error"])
        self.assertEqual(diagnostics[1]["semantic"], "PASS")

    def test_invalid_200_party_csv_is_rejected_before_acceptance(self) -> None:
        source = self._source("party_chart", "party_M.csv", party="M")
        valid = (FIXTURES / "party_M.csv").read_bytes()
        # A date column alone is not enough: pofp is required for the model.
        invalid = b"date,Demoskop\n2026-09-01,20\n"

        def fake_read(url: str, _timeout: float):
            if url == source.url:
                return invalid, _response(url, content_type="text/csv")
            return valid, _response(url, content_type="text/csv")

        with tempfile.TemporaryDirectory() as tmp, patch(
            "scripts.pollofpolls.acquire.SOURCES", (source,)
        ), patch("scripts.pollofpolls.acquire._read_url", fake_read):
            manifest, _messages = acquire_all(Path(tmp))
            self.assertEqual((Path(tmp) / source.raw_filename).read_bytes(), valid)
        diagnostics = manifest["acquisition_diagnostics"]
        self.assertEqual(diagnostics[0]["semantic_validation"], "FAIL")
        self.assertIn("missing date or pofp", diagnostics[0]["error"])
        self.assertEqual(diagnostics[1]["semantic_validation"], "PASS")

    def test_invalid_live_and_archive_retain_verified_payload_with_diagnostics(self) -> None:
        source = self._source("homepage", "homepage.html")
        valid = (FIXTURES / "homepage.html").read_bytes()
        invalid_live = b"<html><body>blocked</body></html>"
        invalid_archive = b"not a polling page"

        def fake_read(url: str, _timeout: float):
            if url == source.url:
                return invalid_live, _response(url, content_type="text/html")
            return invalid_archive, _response(url, content_type="text/html")

        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            destination = raw / source.raw_filename
            destination.write_bytes(valid)
            old_manifest = self._old_manifest(source, valid)
            (raw / "retrieval_manifest.json").write_text(
                json.dumps(old_manifest), encoding="utf-8"
            )
            before = destination.read_bytes()
            with patch("scripts.pollofpolls.acquire.SOURCES", (source,)), patch(
                "scripts.pollofpolls.acquire._read_url", fake_read
            ):
                manifest, messages = acquire_all(raw)

            self.assertEqual(destination.read_bytes(), before)
            self.assertIn("retained verified raw file", " ".join(messages))
            self.assertEqual(manifest["source_outcomes"][source.key], "retained_verified_previous")
            diagnostics = manifest["acquisition_diagnostics"]
            self.assertEqual(len(diagnostics), 2)
            self.assertTrue(all(item["retained_previous"] for item in diagnostics))
            self.assertTrue(all(item["semantic_validation"] == "FAIL" for item in diagnostics))
            self.assertEqual(manifest["sources"][source.key]["sha256"], sha256_bytes(valid))
            self.assertEqual(
                manifest["sources"][source.key], old_manifest["sources"][source.key]
            )

    def test_invalid_previous_payload_fails_closed_without_overwriting_it_or_manifest(self) -> None:
        source = self._source("homepage", "homepage.html")
        invalid_previous = b"<html><body>previous payload is also invalid</body></html>"
        invalid_live = b"<html><body>live block</body></html>"
        invalid_archive = b"<html><body>archive block</body></html>"

        def fake_read(url: str, _timeout: float):
            payload = invalid_live if url == source.url else invalid_archive
            return payload, _response(url, content_type="text/html")

        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            destination = raw / source.raw_filename
            destination.write_bytes(invalid_previous)
            old_manifest = self._old_manifest(source, invalid_previous)
            manifest_path = raw / "retrieval_manifest.json"
            manifest_path.write_text(json.dumps(old_manifest), encoding="utf-8")
            manifest_before = manifest_path.read_bytes()
            with patch("scripts.pollofpolls.acquire.SOURCES", (source,)), patch(
                "scripts.pollofpolls.acquire._read_url", fake_read
            ):
                with self.assertRaises(AcquisitionError):
                    acquire_all(raw)

            self.assertEqual(destination.read_bytes(), invalid_previous)
            self.assertEqual(manifest_path.read_bytes(), manifest_before)

    def test_unrecoverable_invalid_response_raises_with_payload_free_diagnostics(self) -> None:
        source = self._source("homepage", "homepage.html")
        invalid = b"<html><body>still blocked</body></html>"

        def fake_read(url: str, _timeout: float):
            return invalid, _response(url, content_type="text/html")

        with tempfile.TemporaryDirectory() as tmp, patch(
            "scripts.pollofpolls.acquire.SOURCES", (source,)
        ), patch("scripts.pollofpolls.acquire._read_url", fake_read):
            with self.assertRaises(AcquisitionError) as caught:
                acquire_all(Path(tmp))

        diagnostics = caught.exception.diagnostics
        self.assertEqual(len(diagnostics), 2)
        self.assertEqual(diagnostics[0]["http_status"], 200)
        self.assertEqual(diagnostics[0]["final_host"], "pollofpolls.se")
        self.assertEqual(diagnostics[0]["final_path"], "/")
        self.assertEqual(diagnostics[0]["content_type"], "text/html")
        self.assertEqual(diagnostics[0]["byte_length"], len(invalid))
        self.assertEqual(diagnostics[0]["source_key"], source.key)
        self.assertEqual(diagnostics[0]["retrieval_method"], "first_party_http")
        self.assertIn("homepage has no latest-polls table", diagnostics[0]["error"])
        self.assertFalse(any(item["retained_previous"] for item in diagnostics))
        self.assertTrue(all(item["semantic_validation"] == "FAIL" for item in diagnostics))
        self.assertTrue(all("payload" not in item for item in diagnostics))
        rendered = json.dumps(diagnostics, ensure_ascii=False)
        self.assertNotIn(invalid.decode("utf-8"), rendered)
        self.assertNotIn("still blocked", rendered)

    def test_semantic_validation_accepts_harmless_extra_columns_and_html_changes(self) -> None:
        timeseries = (
            "date,M,L,C,KD,S,V,MP,SD,FI,harmless_extra\n"
            "2026-09-01,20,5,7,4,30,8,6,18,2,1.5\n"
        ).encode("utf-8")
        timeseries_source = self._source("timeseries", "timeseries.dat")
        _validate_source_payload(
            timeseries_source,
            timeseries,
            _response(timeseries_source.url, content_type="text/csv"),
        )

        homepage_source = self._source("homepage", "homepage.html")
        homepage = (FIXTURES / "homepage.html").read_text(encoding="utf-8")
        harmless_html_change = (
            "<!-- harmless wrapper content -->\n"
            + homepage.replace("<table ", '<table data-probe-fixture="true" ', 1)
        ).encode("utf-8")
        _validate_source_payload(
            homepage_source,
            harmless_html_change,
            _response(homepage_source.url, content_type="text/html"),
        )

    def test_retained_diagnostics_produce_unavailable_stale_fallback_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in (
                "data/processed/pollofpolls/pollofpolls_timeseries.csv",
                "data/processed/pollofpolls/individual_polls.csv",
                "data/processed/pollofpolls/swedishpolls_individual_polls.csv",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("header\nverified\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(
                ["git", "config", "user.name", "Acquisition Test"], cwd=root, check=True
            )
            subprocess.run(
                ["git", "config", "user.email", "acquisition@example.test"],
                cwd=root,
                check=True,
            )
            subprocess.run(["git", "add", "-A"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)

            def refresh(_raw, _processed, **_kwargs):
                return {
                    "messages": [],
                    "acquisition_diagnostics": [
                        {
                            "source_key": "homepage",
                            "retrieval_method": "verified_previous_snapshot",
                            "semantic_validation": "FAIL",
                            "retained_previous": True,
                        }
                    ],
                }

            polling = refresh_polling_snapshot(root, refresh_fn=refresh)
            self.assertEqual(polling.status, "SOURCE_UNAVAILABLE_USING_VERIFIED_SNAPSHOT")
            self.assertEqual(polling.source_provenance, "VERIFIED_STALE_FALLBACK")
            self.assertTrue(polling.acquisition_diagnostics[0]["retained_previous"])


if __name__ == "__main__":
    unittest.main()
