"""Riksdag vote titles, polite fetching, unchanged-output detection and the site layout (no network)."""

import io
import json
import tempfile
import unittest
import urllib.error
from dataclasses import replace
from datetime import date
from email.message import Message
from pathlib import Path

from scripts.riksdag_votes.build import build, content_of, current_term_sessions, write_json
from scripts.riksdag_votes.fetch import (
    NotFound,
    PoliteHttpGet,
    RawStore,
    bulk_url,
    dokumentlista_by_beteckning_url,
    register_url,
    utskottsforslag_url,
)
from scripts.riksdag_votes.ingest import SessionData, vote_from_rows
from scripts.riksdag_votes.site import build_site, export_site
from scripts.riksdag_votes.summarize import build_summary, build_votes_payload
from scripts.riksdag_votes.titles import (
    attach_titles,
    guess_dok_id,
    parse_dokumentlista_dok_id,
    parse_utskottsforslag,
)
from tests.riksdag_votes_fixtures import (
    FakeHttp,
    bulk_archive,
    dokumentlista_payload,
    register_payload,
    rows_for_vote,
    utskottsforslag_payload,
)
from tests.test_riksdag_votes_build import SCOPE, PROVENANCE, synthetic_rows, synthetic_votes

RM = "2025/26"
GOV = {"M": "Ja", "KD": "Ja", "L": "Ja", "SD": "Ja", "S": "Nej", "V": "Nej", "MP": "Nej", "C": "Avstår"}


def session_votes():
    specs = [
        ("A", "FiU1", "1", "sakfrågan"),
        ("A2", "FiU1", "2", "sakfrågan"),
        ("B", "1015-1", "1", "frågan"),
        ("C", "KU32", "2", "sakfrågan"),
        ("D", "SoU5", "3", "sakfrågan"),
        ("E", "TU9", "1", "sakfrågan"),
    ]
    return [
        vote_from_rows(rows_for_vote(vid, GOV, beteckning=bet, punkt=punkt, avser=avser, hangar_id=str(i)), source="bulk")
        for i, (vid, bet, punkt, avser) in enumerate(specs)
    ]


def title_responses() -> dict[str, bytes]:
    return {
        utskottsforslag_url("HD01FiU1"): utskottsforslag_payload(
            "HD01FiU1", RM, "FiU1", "Statens  budget 2026 –\n Rambeslutet", {"1": "Riktlinjer", "2": " Utgiftstak "}
        ),
        utskottsforslag_url("HD191015-1"): utskottsforslag_payload(
            "HD191015-1", RM, "1015-1", "Omröstning: Prövning av förslag till statsminister", doktyp="votering"
        ),
        # KU32's derived id is absent; dokumentlista gives the real one.
        dokumentlista_by_beteckning_url(RM, "KU32"): dokumentlista_payload(
            [
                {"dok_id": "HD01KU32x", "rm": RM, "beteckning": "KU32", "doktyp": "bet"},
                {"dok_id": "HD02KU32", "rm": RM, "beteckning": "KU32", "doktyp": "mot"},
                {"dok_id": "HC01KU32", "rm": "2024/25", "beteckning": "KU32", "doktyp": "bet"},
            ]
        ),
        utskottsforslag_url("HD01KU32x"): utskottsforslag_payload("HD01KU32x", RM, "KU32", "Grundlag", {"2": "Tryckfrihet"}),
        # SoU5 is published without the voted punkt 3.
        utskottsforslag_url("HD01SoU5"): utskottsforslag_payload("HD01SoU5", RM, "SoU5", "Vård", {"1": "Tandvård"}),
        # TU9: nothing anywhere.
    }


class TitleParsingTests(unittest.TestCase):
    def test_guess_dok_id(self) -> None:
        self.assertEqual(guess_dok_id("2022/23", "FiU1", "committee"), "HA01FiU1")
        self.assertEqual(guess_dok_id("2025/26", "CU13", "committee"), "HD01CU13")
        self.assertEqual(guess_dok_id("2024/25", "UFöU1", "committee"), "HC01UFöU1")
        self.assertEqual(guess_dok_id("2022/23", "1017-1", "numeric"), "HA191017-1")
        self.assertEqual(guess_dok_id("2021/22", "FiU1", "committee"), "H901FiU1")
        self.assertIsNone(guess_dok_id("2011/12", "FiU1", "committee"))
        self.assertIsNone(guess_dok_id("2025/26", "", "missing"))

    def test_parse_single_punkt_dict_and_whitespace(self) -> None:
        payload = json.loads(utskottsforslag_payload("HD01X1", RM, "X1", " A\n  B ", {"1": "  Rubrik  ett "}, single_as_dict=True))
        document = parse_utskottsforslag(payload)
        self.assertEqual((document.title, document.punkter), ("A B", {"1": "Rubrik ett"}))
        self.assertEqual(document.url, "https://data.riksdagen.se/dokument/HD01X1")

    def test_parse_rejects_malformed(self) -> None:
        self.assertIsNone(parse_utskottsforslag({}))
        self.assertIsNone(parse_utskottsforslag({"utskottsforslag": {"dokument": {}}}))

    def test_dokumentlista_requires_one_matching_document(self) -> None:
        one = json.loads(dokumentlista_payload([{"dok_id": "X", "rm": RM, "beteckning": "KU1", "doktyp": "bet"}]))
        self.assertEqual(parse_dokumentlista_dok_id(one, RM, "KU1"), "X")
        two = json.loads(
            dokumentlista_payload(
                [
                    {"dok_id": "X", "rm": RM, "beteckning": "KU1", "doktyp": "bet"},
                    {"dok_id": "Y", "rm": RM, "beteckning": "KU1", "doktyp": "bet"},
                ]
            )
        )
        self.assertIsNone(parse_dokumentlista_dok_id(two, RM, "KU1"))
        self.assertIsNone(parse_dokumentlista_dok_id(one, RM, "KU2"))


class AttachTitlesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def store(self, responses, **kwargs):
        http = FakeHttp(responses)
        return (
            RawStore(
                cache_dir=self.root / "cache",
                manifest_path=self.root / "out" / "manifest.json",
                http_get=http,
                now=lambda: "2026-11-19T12:00:00Z",
                **kwargs,
            ),
            http,
        )

    def attach(self, store):
        session = SessionData(rm=RM, votes=session_votes())
        attach_titles(session, store)
        return session, {vote.votering_id: vote for vote in session.votes}

    def test_titles_from_derived_id_numeric_vote_and_fallback(self) -> None:
        store, _http = self.store(title_responses())
        session, votes = self.attach(store)
        self.assertEqual((votes["A"].title, votes["A"].punkt_title), ("Statens budget 2026 – Rambeslutet", "Riktlinjer"))
        self.assertEqual(votes["A2"].punkt_title, "Utgiftstak")
        self.assertEqual(votes["A"].dok_id, "HD01FiU1")
        self.assertEqual((votes["B"].title, votes["B"].punkt_title), ("Prövning av förslag till statsminister", None))
        self.assertEqual((votes["C"].title, votes["C"].punkt_title, votes["C"].dok_id), ("Grundlag", "Tryckfrihet", "HD01KU32x"))
        self.assertEqual((votes["D"].title, votes["D"].punkt_title), ("Vård", None))
        self.assertEqual((votes["E"].title, votes["E"].dok_id), (None, None))
        self.assertTrue(any("TU9" in note and "no title document" in note for note in session.notes))
        self.assertTrue(any("SoU5 punkt 3" in note for note in session.notes))
        self.assertEqual(
            session.title_files,
            {
                "utskottsforslag-202526-HD01FiU1.json",
                "utskottsforslag-202526-HD191015-1.json",
                "dokumentlista-202526-bet-KU32.json",
                "utskottsforslag-202526-HD01KU32x.json",
                "utskottsforslag-202526-HD01SoU5.json",
            },
        )
        # Votes keep their order and everything else.
        self.assertEqual([v.votering_id for v in session.votes], ["A", "A2", "B", "C", "D", "E"])

    def test_incremental_cache_manifest_and_offline(self) -> None:
        store, http = self.store(title_responses())
        self.attach(store)
        store.write_manifest(["202526"])
        manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))["files"]
        entry = manifest["utskottsforslag-202526-HD01FiU1.json"]
        self.assertEqual(entry["url"], utskottsforslag_url("HD01FiU1"))
        self.assertEqual(len(entry["sha256"]), 64)
        first = list(http.requests)
        self.assertEqual(len(first), len(set(first)))  # one request per file

        # Second online run: only what is still missing is asked for again
        # (SoU5 lacks punkt 3; TU9 and KU32's derived id were never found).
        again, http2 = self.store(title_responses())
        _session, online = self.attach(again)
        self.assertNotIn(utskottsforslag_url("HD01FiU1"), http2.requests)
        self.assertNotIn(utskottsforslag_url("HD191015-1"), http2.requests)
        self.assertIn(utskottsforslag_url("HD01SoU5"), http2.requests)
        self.assertEqual(http2.requests.count(utskottsforslag_url("HD01SoU5")), 1)

        offline, http3 = self.store({}, offline=True)
        session, votes = self.attach(offline)
        self.assertEqual(http3.requests, [])
        self.assertEqual({k: (v.title, v.punkt_title) for k, v in votes.items()}, {k: (v.title, v.punkt_title) for k, v in online.items()})

    def test_cached_document_refetched_when_voted_punkt_appears(self) -> None:
        responses = title_responses()
        store, _ = self.store(responses)
        self.attach(store)
        store.write_manifest(["202526"])
        responses[utskottsforslag_url("HD01SoU5")] = utskottsforslag_payload(
            "HD01SoU5", RM, "SoU5", "Vård", {"1": "Tandvård", "3": "Läkemedel"}
        )
        later, _ = self.store(responses)
        session, votes = self.attach(later)
        self.assertEqual(votes["D"].punkt_title, "Läkemedel")
        self.assertFalse(any("SoU5" in note for note in session.notes))

    def test_payloads_carry_titles(self) -> None:
        store, _ = self.store(title_responses())
        session, _votes = self.attach(store)
        payload = build_votes_payload(session, PROVENANCE)
        first = payload["votes"][0]
        self.assertEqual(first["title"], "Statens budget 2026 – Rambeslutet")
        self.assertEqual(first["punkt_title"], "Riktlinjer")
        self.assertEqual(first["dok_id"], "HD01FiU1")
        self.assertIn("title", payload["encoding"])
        self.assertIn("https://data.riksdagen.se/dokument/<dok_id>", payload["encoding"]["dok_id"])
        self.assertNotIn("document_url", first)

    def test_top_loading_patterns_name_their_votes(self) -> None:
        votes = [replace(v, title=f"Titel {v.beteckning}", punkt_title=f"Punkt {v.punkt}") for v in synthetic_votes()]
        summary = build_summary(votes, scope=SCOPE, provenance=PROVENANCE)
        pattern = summary["map"]["top_loading_patterns"]["axis_1"][0]
        example = pattern["example_votes"][0]
        self.assertEqual(example["id"], pattern["example_vote_ids"][0])
        self.assertTrue(example["title"].startswith("Titel "))
        self.assertTrue(example["punkt_title"].startswith("Punkt "))
        self.assertIn("title", summary["map"]["top_loading_votes"]["axis_1"][0])


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://x", code, "err", headers, io.BytesIO(b""))


class PoliteHttpTests(unittest.TestCase):
    def make(self, outcomes):
        clock = FakeClock()
        calls = []

        def opener(url):
            calls.append((url, clock.now))
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        return PoliteHttpGet(opener=opener, sleep=clock.sleep, clock=clock.clock), calls, clock

    def test_requests_are_spaced(self) -> None:
        get, calls, _clock = self.make([b"a", b"b", b"c"])
        for url in ("u1", "u2", "u3"):
            get(url)
        starts = [t for _url, t in calls]
        self.assertTrue(all(b - a >= 0.25 - 1e-9 for a, b in zip(starts, starts[1:])), starts)

    def test_backoff_on_server_errors_then_success(self) -> None:
        get, calls, clock = self.make([http_error(503), urllib.error.URLError("down"), b"ok"])
        self.assertEqual(get("u"), b"ok")
        self.assertEqual(len(calls), 3)
        self.assertEqual(clock.sleeps[:1], [2.0])
        self.assertIn(4.0, clock.sleeps)

    def test_retry_after_is_honoured(self) -> None:
        get, _calls, clock = self.make([http_error(429, "7"), b"ok"])
        self.assertEqual(get("u"), b"ok")
        self.assertEqual(clock.sleeps[0], 7.0)

    def test_404_and_client_errors_are_not_retried(self) -> None:
        get, calls, _ = self.make([http_error(404)])
        with self.assertRaises(NotFound):
            get("u")
        self.assertEqual(len(calls), 1)
        get, calls, _ = self.make([http_error(400)])
        with self.assertRaises(RuntimeError):
            get("u")
        self.assertEqual(len(calls), 1)

    def test_gives_up_after_attempts(self) -> None:
        get, calls, _ = self.make([http_error(500)] * 5)
        with self.assertRaisesRegex(RuntimeError, "after 5 attempts"):
            get("u")
        self.assertEqual(len(calls), 5)


class StoreAndOutputStabilityTests(unittest.TestCase):
    def test_identical_redownload_keeps_retrieval_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            http = FakeHttp({"https://x/a": b"one"})
            times = iter(["2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"])
            store = RawStore(cache_dir=root / "c", manifest_path=root / "m.json", http_get=http, now=lambda: next(times))
            store.fetch_listing("a-202526", "https://x/a")
            store.refresh_listings = True
            store.downloaded.clear()
            store.fetch_listing("a-202526", "https://x/a")
            self.assertEqual(store.entries["a-202526"]["retrieved_at_utc"], "2026-01-01T00:00:00Z")
            http.responses["https://x/a"] = b"two"
            store.downloaded.clear()
            store.fetch_listing("a-202526", "https://x/a")
            self.assertEqual(store.entries["a-202526"]["retrieved_at_utc"], "2026-02-01T00:00:00Z")

    def test_write_json_ignores_provenance_only_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.json"
            first = {"a": 1, "provenance": {"generated_at_utc": "T1", "code_commit": "c1", "code_dirty": False, "sources": [1]}}
            write_json(path, first)
            original = path.read_bytes()
            # New run stamp and a re-zipped source with the same content.
            second = {"a": 1, "provenance": {"generated_at_utc": "T2", "code_commit": "c2", "code_dirty": True, "sources": [2]}}
            write_json(path, second)
            self.assertEqual(path.read_bytes(), original)
            write_json(path, dict(second, a=2))
            written = json.loads(path.read_bytes())
            self.assertEqual((written["a"], written["provenance"]["generated_at_utc"]), (2, "T2"))
            self.assertEqual(content_of(second), {"a": 1})

    def test_current_term_sessions(self) -> None:
        self.assertEqual(current_term_sessions(date(2026, 11, 19)), ["2026/27"])
        self.assertEqual(current_term_sessions(date(2026, 8, 31)), ["2022/23", "2023/24", "2024/25", "2025/26"])
        self.assertEqual(current_term_sessions(date(2027, 8, 31)), ["2026/27"])
        self.assertEqual(current_term_sessions(date(2027, 9, 1)), ["2026/27", "2027/28"])
        self.assertEqual(current_term_sessions(date(2030, 6, 1)), ["2026/27", "2027/28", "2028/29", "2029/30"])
        self.assertEqual(current_term_sessions(date(2030, 9, 2)), ["2030/31"])


class SiteExportTests(unittest.TestCase):
    def build_outputs(self, root: Path, generated: str, code: str) -> Path:
        rows = synthetic_rows(n_votes=30, rm="2026/27")
        responses = {bulk_url("2026/27"): bulk_archive(rows), register_url("2026/27"): register_payload(rows)}
        out = root / "out"
        store = RawStore(cache_dir=root / "cache", manifest_path=out / "manifest.json", http_get=FakeHttp(responses), now=lambda: "2026-11-19T12:00:00Z")
        build(["2026/27"], store, out, generated_at_utc=generated, code={"code_commit": code, "code_dirty": False}, today=date(2026, 11, 19))
        return out

    def test_layout_index_and_stability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = self.build_outputs(root, "2026-11-19T12:00:00Z", "abc")
            site = root / "site"
            export_site(out, site)
            names = sorted(str(p.relative_to(site)) for p in site.rglob("*") if p.is_file())
            self.assertEqual(names, ["2026-27/summary.json", "2026-27/votes.json", "index.json", "term-2026-30/summary.json"])
            self.assertEqual((site / "2026-27/votes.json").read_bytes(), (out / "2026-27/votes.json").read_bytes())
            index = json.loads((site / "index.json").read_text(encoding="utf-8"))
            self.assertEqual((index["schema_version"], index["role"]), ("1.0", "riksdag_votes_index"))
            self.assertEqual(index["generated_at_utc"], "2026-11-19T12:00:00Z")
            self.assertEqual(index["latest_vote_date"], max(r[0]["datum"] for r in synthetic_rows(n_votes=30, rm="2026/27")))
            session = index["sessions"][0]
            self.assertEqual((session["rm"], session["slug"], session["n_votes"]), ("2026/27", "2026-27", 30))
            import hashlib

            for record in session["files"].values():
                data = (site / record["path"]).read_bytes()
                self.assertEqual((record["bytes"], record["sha256"]), (len(data), hashlib.sha256(data).hexdigest()))
            self.assertEqual(index["terms"][0]["label"], "2026-30")
            self.assertEqual(index["terms"][0]["files"]["summary"]["path"], "term-2026-30/summary.json")
            before = {p: p.read_bytes() for p in site.rglob("*") if p.is_file()}

            # A later run with no new votes changes nothing, on disk or on the site.
            self.build_outputs(root, "2026-11-20T06:41:00Z", "def")
            files, index_again = build_site(out)
            self.assertEqual(index_again, index)
            export_site(out, site)
            self.assertEqual({p: p.read_bytes() for p in site.rglob("*") if p.is_file()}, before)


if __name__ == "__main__":
    unittest.main()
