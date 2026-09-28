"""Tests for mirroring the certified election result into the website."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.elections.config import ELECTIONS
from scripts.site_publisher.election_result import (
    DEFAULT_RESULT_MANIFEST,
    build_site_election_result,
    ordinary_election_date,
    site_result_relative,
    sync_election_result_to_site,
)
from scripts.site_publisher.publisher import SitePublishError

try:
    from ._website_repo import website_repo
except ImportError:  # pragma: no cover - direct module execution
    from tests._website_repo import website_repo


PARTIES = ["M", "L", "C", "KD", "S", "V", "MP", "SD"]


class OrdinaryElectionDateTests(unittest.TestCase):
    def test_every_recorded_election_is_on_its_ordinary_day(self) -> None:
        # Third Sunday in September through 2010, second from 2014.
        for year, metadata in ELECTIONS.items():
            with self.subTest(year=year):
                self.assertEqual(ordinary_election_date(year), metadata.election_date)

    def test_2026_and_the_next_election(self) -> None:
        self.assertEqual(ordinary_election_date(2026), date(2026, 9, 13))
        self.assertEqual(ordinary_election_date(2030), date(2030, 9, 8))


class BuildSiteElectionResultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = build_site_election_result()
        self.manifest = json.loads(DEFAULT_RESULT_MANIFEST.read_text(encoding="utf-8"))

    def test_the_site_contract_fields(self) -> None:
        self.assertEqual(self.payload["schema_version"], "1.0")
        self.assertEqual(self.payload["role"], "official_election_result")
        self.assertEqual(self.payload["certification_status"], "FINAL_CERTIFIED")
        self.assertEqual(self.payload["vote_share_definition"], "national_vote_share")
        self.assertEqual(self.payload["election_date"], "2026-09-13")
        self.assertEqual(self.payload["next_election_date"], "2030-09-08")
        self.assertEqual(self.payload["party_order"], PARTIES)
        self.assertEqual(self.payload["total_seats"], 349)

    def test_every_number_is_the_certified_one(self) -> None:
        for party in PARTIES:
            with self.subTest(party=party):
                published = self.payload["parties"][party]
                certified = self.manifest["parties"][party]
                self.assertEqual(published["votes"], certified["votes"])
                self.assertEqual(published["seats"], certified["seats"])
                self.assertAlmostEqual(
                    published["vote_share_pct"], certified["vote_share_percentage_points"], places=4)
        self.assertEqual(sum(self.payload["parties"][p]["seats"] for p in PARTIES), 349)
        self.assertEqual(self.payload["raw_sha256"], self.manifest["raw_sha256"])
        self.assertEqual(self.payload["turnout_pct"], 84.89)

    def test_other_parties_close_the_denominator(self) -> None:
        eight = sum(self.payload["parties"][p]["votes"] for p in PARTIES)
        self.assertEqual(self.payload["other_parties"]["votes"] + eight,
                         self.payload["valid_national_votes"])
        self.assertEqual(self.payload["other_parties"]["seats"], 0)


class RejectsUncertifiedEvidenceTests(unittest.TestCase):
    """The page must never print a result the benchmark would refuse to score."""

    def setUp(self) -> None:
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.evidence = self.tmp / "val2026"
        shutil.copytree(DEFAULT_RESULT_MANIFEST.parent, self.evidence)
        self.manifest_path = self.evidence / DEFAULT_RESULT_MANIFEST.name
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _rewrite(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_the_copied_evidence_is_accepted(self) -> None:
        self.assertEqual(build_site_election_result(self.manifest_path)["election_date"], "2026-09-13")

    def test_a_preliminary_count_is_refused(self) -> None:
        self.manifest["certification_status"] = "PRELIMINARY"
        self._rewrite()
        with self.assertRaises(SitePublishError):
            build_site_election_result(self.manifest_path)

    def test_a_tampered_raw_file_is_refused(self) -> None:
        raw = self.evidence / self.manifest["raw_path"]
        raw.write_bytes(raw.read_bytes() + b" ")
        with self.assertRaises(SitePublishError):
            build_site_election_result(self.manifest_path)

    def test_seats_that_do_not_fill_the_chamber_are_refused(self) -> None:
        self.manifest["parties"]["M"]["seats"] -= 1
        self._rewrite()
        with self.assertRaisesRegex(SitePublishError, "348 seats"):
            build_site_election_result(self.manifest_path)


class SyncElectionResultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.site = Path(self.enterContext(tempfile.TemporaryDirectory())) / "website"
        self.site.mkdir()

    def test_sync_installs_then_leaves_equal_bytes_alone(self) -> None:
        first = sync_election_result_to_site(site_repo=self.site)
        destination = self.site / site_result_relative(2026)
        self.assertEqual(first["status"], "SYNCED")
        self.assertTrue(destination.is_file())
        self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), build_site_election_result())
        before = destination.stat().st_mtime_ns
        second = sync_election_result_to_site(site_repo=self.site)
        self.assertEqual(second["status"], "UNCHANGED")
        self.assertEqual(destination.stat().st_mtime_ns, before)
        self.assertFalse(first["committed"] or first["pushed"])

    def test_a_missing_site_is_refused(self) -> None:
        with self.assertRaises(SitePublishError):
            sync_election_result_to_site(site_repo=self.site / "absent")


class DeployedResultMatchesTests(unittest.TestCase):
    """The deployed file is exactly what this exporter builds.

    Opt-in, like the other cross-repository tests: set
    ``ELECTION_SIMULATOR_WEBSITE_REPO`` to the website checkout.
    """

    def test_the_website_file_is_byte_identical(self) -> None:
        deployed = website_repo() / site_result_relative(2026)
        if not deployed.is_file():
            self.skipTest("no website checkout with results/2026.json is opted in")
        site = Path(self.enterContext(tempfile.TemporaryDirectory()))
        sync_election_result_to_site(site_repo=site)
        self.assertEqual((site / site_result_relative(2026)).read_bytes(), deployed.read_bytes())


if __name__ == "__main__":
    unittest.main()
