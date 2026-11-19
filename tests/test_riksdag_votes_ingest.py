"""Riksdag roll-call ingest: parsing, completeness checks, bulk/API reconciliation, cache."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.riksdag_votes.fetch import (
    RawStore,
    bulk_url,
    register_url,
    rows_by_beteckning_url,
    session_slug,
    vote_metadata_url,
)
from scripts.riksdag_votes.ingest import (
    IngestError,
    group_rows_by_vote,
    load_session,
    parse_bulk_archive,
    parse_register,
    parse_vote_document,
    validate_vote_rows,
    vote_from_rows,
)
from tests.riksdag_votes_fixtures import (
    FakeHttp,
    bulk_archive,
    chamber,
    metadata_payload,
    register_payload,
    rows_for_vote,
    voteringlista_rows_payload,
)

GOV = {"M": "Ja", "KD": "Ja", "L": "Ja", "SD": "Ja", "S": "Nej", "V": "Nej", "MP": "Nej", "C": "Avstår"}
RM = "2025/26"


class ParsingTests(unittest.TestCase):
    def test_single_row_dict_becomes_list(self) -> None:
        row = rows_for_vote("V1", GOV)[0]
        self.assertEqual(parse_vote_document({"dokvotering": {"votering": row}}), [row])

    def test_missing_body_raises(self) -> None:
        with self.assertRaises(IngestError):
            parse_vote_document({"dokvotering": {}})

    def test_bulk_archive_decodes_utf8_sig(self) -> None:
        votes = [rows_for_vote("V1", GOV), rows_for_vote("V2", GOV, punkt="2")]
        parsed = parse_bulk_archive(bulk_archive(votes, bom=True))
        self.assertEqual([rows[0]["votering_id"] for rows in parsed], ["V1", "V2"])
        self.assertEqual((parsed[0][0]["parti"], parsed[0][0]["rost"]), ("S", "Nej"))
        self.assertEqual(len(parsed[1]), 349)

    def test_vote_from_rows_orders_by_bench_and_reads_fields(self) -> None:
        rows = rows_for_vote("V1", GOV, beteckning="UFöU3", datum="2025-11-05")
        vote = vote_from_rows(list(reversed(rows)), source="bulk")
        self.assertEqual([b.banknummer for b in vote.ballots], list(range(1, 350)))
        self.assertEqual(vote.committee, "UFöU")
        self.assertEqual((vote.date, vote.date_source), ("2025-11-05", "row_datum"))
        self.assertTrue(vote.in_analysis_set)
        self.assertEqual(vote.counts()["Avstår"], 24)

    def test_analysis_set_flag(self) -> None:
        self.assertFalse(vote_from_rows(rows_for_vote("V1", GOV, avser="motivfrågan"), source="bulk").in_analysis_set)
        self.assertFalse(vote_from_rows(rows_for_vote("V1", GOV, avser="frågan", datum=None), source="bulk").in_analysis_set)
        self.assertFalse(vote_from_rows(rows_for_vote("V1", GOV, votering="kontra"), source="bulk").in_analysis_set)


class CompletenessTests(unittest.TestCase):
    def test_exactly_349_rows_required(self) -> None:
        rows = rows_for_vote("V1", GOV)
        validate_vote_rows(rows)
        with self.assertRaisesRegex(IngestError, "348 rows"):
            validate_vote_rows(rows[:-1])
        with self.assertRaisesRegex(IngestError, "350 rows"):
            validate_vote_rows(rows + [dict(rows[0], intressent_id="X", banknummer="350")])

    def test_duplicate_pair_detected(self) -> None:
        rows = rows_for_vote("V1", GOV)
        rows[5] = dict(rows[5], intressent_id=rows[4]["intressent_id"])
        with self.assertRaisesRegex(IngestError, "duplicate"):
            validate_vote_rows(rows)

    def test_bench_numbers_must_cover_1_to_349(self) -> None:
        rows = rows_for_vote("V1", GOV)
        rows[0] = dict(rows[0], banknummer="2")
        with self.assertRaisesRegex(IngestError, "banknummer"):
            validate_vote_rows(rows)

    def test_unknown_rost_and_mixed_vote_fields_rejected(self) -> None:
        rows = rows_for_vote("V1", GOV)
        rows[0] = dict(rows[0], rost="Kanske")
        with self.assertRaisesRegex(IngestError, "unknown rost"):
            validate_vote_rows(rows)
        rows = rows_for_vote("V1", GOV)
        rows[0] = dict(rows[0], punkt="9")
        with self.assertRaisesRegex(IngestError, "punkt"):
            validate_vote_rows(rows)

    def test_truncated_voteringlista_rejected(self) -> None:
        rows = rows_for_vote("V1", GOV)
        payload = json.loads(voteringlista_rows_payload(rows[:200], declared=349))
        with self.assertRaisesRegex(IngestError, "truncated"):
            group_rows_by_vote(payload)

    def test_register_parses_blank_counts_as_zero(self) -> None:
        register = parse_register(json.loads(register_payload([rows_for_vote("V1", {"M": "Ja"})])))
        self.assertEqual(register["V1"], {"Ja": 68, "Nej": 0, "Avstår": 0, "Frånvarande": 281})


class SessionReconciliationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def store(self, responses: dict[str, bytes], **kwargs) -> tuple[RawStore, FakeHttp]:
        http = FakeHttp(responses)
        store = RawStore(
            cache_dir=self.root / "cache",
            manifest_path=self.root / "out" / "manifest.json",
            http_get=http,
            now=lambda: "2026-11-19T12:00:00Z",
            **kwargs,
        )
        return store, http

    def fixture(self):
        a = rows_for_vote("A", GOV, hangar_id="1", punkt="1")
        b = rows_for_vote("B", GOV, avser="frågan", beteckning="1015-1", datum=None, hangar_id="2", punkt="0")
        withdrawn = rows_for_vote("W", {"M": "Ja"}, hangar_id="3", beteckning="SkU3")
        late = rows_for_vote("C", GOV, beteckning="KU32", punkt="2", hangar_id="4", datum=None)
        other_punkt = rows_for_vote("C2", GOV, beteckning="KU32", punkt="3", hangar_id="4", datum=None)
        responses = {
            bulk_url(RM): bulk_archive([a, b, withdrawn]),
            register_url(RM): register_payload([a, b, late]),
            vote_metadata_url("B"): metadata_payload("1015-1", "2025-10-15"),
            vote_metadata_url("C"): metadata_payload("KU32", "2026-06-11"),
            rows_by_beteckning_url(RM, "KU32"): voteringlista_rows_payload(late + other_punkt),
        }
        return responses

    def test_register_is_authoritative(self) -> None:
        store, _http = self.store(self.fixture())
        session = load_session(RM, store)
        self.assertEqual(sorted(v.votering_id for v in session.votes), ["A", "B", "C"])
        self.assertEqual([e["votering_id"] for e in session.excluded], ["W"])
        by_id = {v.votering_id: v for v in session.votes}
        self.assertEqual(by_id["C"].source, "voteringlista")
        self.assertEqual((by_id["C"].date, by_id["C"].date_source), ("2026-06-11", "votering_metadata"))
        self.assertEqual((by_id["B"].date, by_id["B"].date_source), ("2025-10-15", "votering_metadata"))
        self.assertIsNone(by_id["B"].committee)
        self.assertEqual(by_id["C"].committee, "KU")
        self.assertEqual([v.votering_id for v in session.votes], ["A", "B", "C"])  # date order

    def test_count_mismatch_with_register_fails(self) -> None:
        responses = self.fixture()
        a = rows_for_vote("A", GOV, hangar_id="1")
        a_changed = rows_for_vote("A", GOV, hangar_id="1", overrides={"M000": "Nej"})
        responses[register_url(RM)] = register_payload([a_changed])
        responses[bulk_url(RM)] = bulk_archive([a])
        store, _ = self.store(responses)
        with self.assertRaisesRegex(IngestError, "differ from the register"):
            load_session(RM, store)

    def test_missing_bulk_falls_back_to_voteringlista(self) -> None:
        responses = self.fixture()
        del responses[bulk_url(RM)]
        a = rows_for_vote("A", GOV, hangar_id="1")
        responses[register_url(RM)] = register_payload([a])
        responses[vote_metadata_url("A")] = metadata_payload("FiU1", "2025-10-01")
        responses[rows_by_beteckning_url(RM, "FiU1")] = voteringlista_rows_payload(a)
        store, _ = self.store(responses)
        session = load_session(RM, store)
        self.assertEqual([v.source for v in session.votes], ["voteringlista"])
        self.assertTrue(any("not found" in note for note in session.notes))

    def test_cache_reused_and_manifest_records_hashes(self) -> None:
        store, http = self.store(self.fixture())
        load_session(RM, store)
        store.write_manifest(["202526"])
        first_requests = len(http.requests)
        manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["role"], "riksdag_votes_source_manifest")
        entry = manifest["files"]["votering-202526.json.zip"]
        self.assertEqual(entry["url"], bulk_url(RM))
        self.assertEqual(entry["retrieved_at_utc"], "2026-11-19T12:00:00Z")
        self.assertEqual(len(entry["sha256"]), 64)

        offline, http2 = self.store({}, offline=True)
        session = load_session(RM, offline)
        self.assertEqual(len(session.votes), 3)
        self.assertEqual(http2.requests, [])
        self.assertGreater(first_requests, 0)

    def test_tampered_cache_is_redownloaded(self) -> None:
        store, _ = self.store(self.fixture())
        load_session(RM, store)
        store.write_manifest(["202526"])
        (self.root / "cache" / "votering-202526.json.zip").write_bytes(b"tampered")
        offline, _ = self.store({}, offline=True)
        with self.assertRaisesRegex(RuntimeError, "offline"):
            load_session(RM, offline)

    def test_slug(self) -> None:
        self.assertEqual(session_slug("2026/27"), "2026-27")
        with self.assertRaises(ValueError):
            session_slug("2026-27")


if __name__ == "__main__":
    unittest.main()
