"""Riksdag roll-call outputs: summary statistics, compact votes encoding, determinism, CLI build."""

import json
import random
import tempfile
import unittest
from datetime import date
from pathlib import Path

from scripts.riksdag_votes.build import build, dumps, terms_to_write
from scripts.riksdag_votes.fetch import RawStore, bulk_url, register_url
from scripts.riksdag_votes.ingest import SessionData, vote_from_rows
from scripts.riksdag_votes.summarize import build_summary, build_votes_payload
from tests.riksdag_votes_fixtures import FakeHttp, bulk_archive, chamber, register_payload, rows_for_vote

PARTIES = ("M", "L", "C", "KD", "S", "V", "MP", "SD")
PROVENANCE = {"code_commit": "test", "generated_at_utc": "2026-11-19T00:00:00Z", "sources": []}
SCOPE = {"kind": "session", "label": "2025/26", "sessions": ["2025/26"]}


def synthetic_rows(n_votes: int = 120, seed: int = 5, rm: str = "2025/26") -> list[list[dict]]:
    """Random two-bloc-ish party lines; a substitute; an independent; a frequent absentee; one rebel."""
    rng = random.Random(seed)
    base = chamber(independents=2)
    substitute = dict(base[200], intressent_id="SUB001", namn="Sub Stitute")  # bench 201 (an SD seat)
    votes = []
    for i in range(n_votes):
        right = rng.choice(["Ja", "Nej"])
        left = "Nej" if right == "Ja" else "Ja"
        lines = {party: right for party in ("M", "KD", "L", "SD")}
        lines.update({party: left for party in ("S", "V", "MP")})
        lines["C"] = rng.choice(["Ja", "Nej", "Avstår"])
        if rng.random() < 0.25:
            lines["V"] = right  # some cross-bloc structure
        lines["-"] = rng.choice(["Ja", "Nej"])
        people = [substitute if p["banknummer"] == "201" and i % 4 == 0 else p for p in base]
        overrides = {"M001": "Frånvarande"} if i % 2 else {}
        if i == 3:
            overrides["S010"] = "Ja" if lines["S"] == "Nej" else "Nej"
        avser = "motivfrågan" if i == n_votes - 1 else "sakfrågan"
        votes.append(
            rows_for_vote(
                f"V{i:03d}",
                lines,
                people=people,
                overrides=overrides,
                rm=rm,
                beteckning=["FiU", "JuU", "SoU"][i % 3] + str(i % 7 + 1),
                punkt=str(i % 5 + 1),
                avser=avser,
                datum=f"2025-{10 + i // 30:02d}-{i % 28 + 1:02d}",
                hangar_id=str(1000 + i),
            )
        )
    return votes


def synthetic_votes(**kwargs):
    return [vote_from_rows(rows, source="bulk") for rows in synthetic_rows(**kwargs)]


class SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.votes = synthetic_votes()
        cls.summary = build_summary(cls.votes, scope=SCOPE, provenance=PROVENANCE)
        cls.members = {m["id"]: m for m in cls.summary["members"]}

    def test_schema_and_caveats(self) -> None:
        self.assertEqual(self.summary["schema_version"], "1.0")
        self.assertEqual(self.summary["role"], "riksdag_votes_summary")
        self.assertIn("kvittning", self.summary["caveats"]["kvittning"])
        self.assertIn("feltryckningar", self.summary["caveats"]["feltryckning"])

    def test_analysis_set_and_parties(self) -> None:
        self.assertEqual(self.summary["counts"]["votes_all"], 120)
        self.assertEqual(self.summary["counts"]["votes_analysis"], 119)
        self.assertEqual(self.summary["parties"], list(PARTIES))
        self.assertEqual(self.summary["agreement"]["overall"]["n_votes"], 119)
        # every vote keeps its party lines, analysis or not
        self.assertEqual(len(self.summary["party_lines"]["votes"]), 120)
        self.assertNotIn("-", self.summary["party_lines"]["parties"])

    def test_agreement_matches_construction(self) -> None:
        agreement = self.summary["agreement"]
        i, j = PARTIES.index("M"), PARTIES.index("S")
        self.assertEqual(agreement["overall"]["three_way"]["share"][i][j], 0.0)
        self.assertEqual(agreement["overall"]["three_way"]["n"][i][j], 119)
        k = PARTIES.index("KD")
        self.assertEqual(agreement["overall"]["three_way"]["share"][i][k], 1.0)
        self.assertEqual(set(agreement["by_committee"]), {"FiU", "JuU", "SoU"})

    def test_participation_and_absentee(self) -> None:
        absentee = self.members["M001"]
        self.assertEqual(absentee["seated"], 119)
        self.assertEqual(absentee["cast"], 60)
        self.assertEqual(absentee["participation"], round(60 / 119, 4))
        substitute = self.members["SUB001"]
        self.assertEqual(substitute["seated"], 30)
        self.assertEqual(self.members[chamber()[200]["intressent_id"]]["seated"], 89)
        self.assertNotIn("rank", json.dumps(self.summary["participation_distribution"]))

    def test_against_line(self) -> None:
        rebel = self.members["S010"]
        self.assertEqual(rebel["against_line"], 1)
        self.assertEqual(rebel["against_line_vote_ids"], ["V003"])
        self.assertEqual(self.summary["party_against_line"]["S"]["against"], 1)
        independent = next(m for m in self.summary["members"] if m["party"] == "-")
        self.assertEqual(independent["against_line_comparable"], 0)
        self.assertEqual(self.summary["party_against_line"]["M"]["against"], 0)

    def test_map(self) -> None:
        section = self.summary["map"]
        self.assertTrue(section["available"])
        self.assertEqual(len(section["explained_variance_ratio"]), 2)
        self.assertGreater(section["party_coordinates"]["M"][0], 0)
        self.assertEqual(len(section["top_loading_votes"]["axis_1"]), 10)
        loyal = self.members["M002"]["map"]
        np_close = abs(loyal["x"] - section["party_coordinates"]["M"][0]) + abs(loyal["y"] - section["party_coordinates"]["M"][1])
        self.assertLess(np_close, 1e-3)
        absentee = self.members["M001"]["map"]
        self.assertLess(abs(absentee["x"] - section["party_coordinates"]["M"][0]), 0.05)
        self.assertIn("unplaced_reason", self.members["SUB001"]["map"])  # 30 votes < 50

    def test_deterministic_and_order_independent(self) -> None:
        again = build_summary(list(reversed(synthetic_votes())), scope=SCOPE, provenance=PROVENANCE)
        self.assertEqual(dumps(self.summary), dumps(again))


class VotesPayloadTests(unittest.TestCase):
    def test_compact_encoding_round_trips(self) -> None:
        votes = synthetic_votes(n_votes=8)
        payload = build_votes_payload(SessionData(rm="2025/26", votes=votes), PROVENANCE)
        self.assertEqual(payload["role"], "riksdag_votes_session_votes")
        for vote, entry in zip(votes, payload["votes"]):
            self.assertEqual(len(entry["ballots"]), 349)
            roster = payload["rosters"][entry["roster"]]
            for ballot in vote.ballots:
                person_index, parti, _valkrets = payload["seat_holders"][roster[ballot.banknummer - 1]]
                self.assertEqual(payload["people"][person_index]["id"], ballot.intressent_id)
                self.assertEqual(parti, ballot.parti)
                self.assertEqual(entry["ballots"][ballot.banknummer - 1], {"Ja": "J", "Nej": "N", "Avstår": "A", "Frånvarande": "F"}[ballot.rost])
        self.assertEqual(len(payload["rosters"]), 2)  # with and without the substitute
        self.assertLess(len(dumps(payload)), 1_000_000)


class BuildTests(unittest.TestCase):
    def test_terms_written_only_when_all_begun_sessions_present(self) -> None:
        today = date(2026, 11, 19)
        all_sessions = ["2022/23", "2023/24", "2024/25", "2025/26", "2026/27"]
        self.assertEqual(set(terms_to_write(all_sessions, today)), {"2022-26", "2026-30"})
        self.assertEqual(set(terms_to_write(["2025/26", "2026/27"], today)), {"2026-30"})
        self.assertEqual(terms_to_write(["2026/27"], today)["2026-30"], ["2026/27"])
        self.assertEqual(terms_to_write(["2026/27"], date(2027, 9, 2)), {})

    def test_end_to_end_build_is_deterministic(self) -> None:
        rows = synthetic_rows(n_votes=30, rm="2026/27")
        responses = {bulk_url("2026/27"): bulk_archive(rows), register_url("2026/27"): register_payload(rows)}
        outputs = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for run in range(2):
                store = RawStore(
                    cache_dir=root / "cache",
                    manifest_path=root / f"out{run}" / "manifest.json",
                    http_get=FakeHttp(responses),
                    now=lambda: "2026-11-19T12:00:00Z",
                )
                written = build(
                    ["2026/27"],
                    store,
                    root / f"out{run}",
                    generated_at_utc="2026-11-19T12:00:00Z",
                    code={"code_commit": "abc", "code_dirty": False},
                    today=date(2026, 11, 19),
                )
                self.assertEqual(len(written), 4)  # votes, summary, term summary, manifest
                outputs.append(
                    {
                        name: (root / f"out{run}" / name).read_bytes()
                        for name in ("2026-27/votes.json", "2026-27/summary.json", "term-2026-30/summary.json", "manifest.json")
                    }
                )
            self.assertEqual(outputs[0], outputs[1])
            summary = json.loads(outputs[0]["2026-27/summary.json"])
            self.assertEqual({s["file"] for s in summary["provenance"]["sources"]}, {"voteringlista-202627-register.json", "votering-202627.json.zip"})
            term = json.loads(outputs[0]["term-2026-30/summary.json"])
            self.assertFalse(term["scope"]["term_complete"])


if __name__ == "__main__":
    unittest.main()
