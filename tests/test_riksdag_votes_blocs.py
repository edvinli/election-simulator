"""Map orientation (term anchors, session alignment), blocs from the PM vote, per-vote flags (no network)."""

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np

from scripts.riksdag_votes.blocs import cross_bloc, find_blocs, gov_split, is_pm_vote, pm_votes_considered, vote_flags
from scripts.riksdag_votes.build import build, dumps
from scripts.riksdag_votes.fetch import RawStore, bulk_url, register_url
from scripts.riksdag_votes.ideal_points import _orient_axis_by_anchor, align_to_reference, fit_party_map
from scripts.riksdag_votes.ingest import vote_from_rows
from scripts.riksdag_votes.party_lines import PartyLine, party_lines_for_vote
from scripts.riksdag_votes.summarize import build_summary
from tests.riksdag_votes_fixtures import FakeHttp, bulk_archive, register_payload, rows_for_vote
from tests.test_riksdag_votes_build import PROVENANCE, SCOPE, synthetic_rows, synthetic_votes

RIGHT = {"M": "Ja", "KD": "Ja", "L": "Ja", "SD": "Ja", "S": "Nej", "V": "Nej", "MP": "Nej", "C": "Nej"}
PM_TITLE = "Prövning av talmannens förslag till statsminister den 15 oktober"


def line(value):
    return PartyLine(value, False, value is None, (0, 0, 0, 0))


def pm_vote(vote_id, party_rost, *, date_, hangar, title=PM_TITLE, rm="2026/27"):
    rows = rows_for_vote(
        vote_id, party_rost, rm=rm, beteckning=date_[5:7] + date_[8:10] + "-1", punkt="0",
        avser="frågan", datum=None, hangar_id=hangar,
    )
    return replace(vote_from_rows(rows, source="bulk", date=date_, date_source="votering_metadata"), title=title)


class OrientationTests(unittest.TestCase):
    def test_anchor_at_zero_falls_back_to_the_next_party(self) -> None:
        parties = ["M", "V", "S"]
        for v_score in (0.0, 0.01):  # exactly zero, and below 10 % of the axis's largest |score|
            scores = np.array([[0.0, -0.5], [0.0, v_score], [0.0, 0.3]])
            loadings = np.ones((4, 2))
            used = _orient_axis_by_anchor(scores, loadings, parties, 1, "V")
            self.assertEqual(used, "M")  # first party in PARTY_ORDER that clears the bar
            self.assertGreater(scores[0, 1], 0)
            self.assertTrue(np.all(loadings[:, 1] == -1))
        scores = np.array([[0.0, -0.5], [0.0, -0.2], [0.0, 0.3]])
        self.assertEqual(_orient_axis_by_anchor(scores, np.ones((4, 2)), parties, 1, "V"), "V")
        self.assertGreater(scores[1, 1], 0)

    def test_term_map_puts_m_on_the_right_of_axis_1_and_v_up_on_axis_2(self) -> None:
        summary = build_summary(synthetic_votes(), scope=SCOPE, provenance=PROVENANCE)
        coordinates = summary["map"]["party_coordinates"]
        self.assertGreater(coordinates["M"][0], 0)
        self.assertGreater(coordinates["V"][1], 0)
        self.assertIsNone(summary["map"]["orientation_reference"])
        self.assertEqual([a["anchor"] for a in summary["map"]["orientation_axes"]], ["M", "V"])

    def test_session_axes_follow_the_term(self) -> None:
        rng = np.random.default_rng(3)
        parties = ("M", "L", "C", "KD", "S", "V", "MP", "SD")
        matrix = rng.choice([-1.0, 0.0, 1.0], size=(8, 80))
        term = fit_party_map(matrix, parties)
        reference = {p: term.party_scores[i] for i, p in enumerate(parties)}
        session = fit_party_map(matrix[:, :60], parties)
        session.party_scores[:, :] *= -1  # whatever sign the SVD gave
        session.loadings[:, :] *= -1
        records = align_to_reference(session, reference)
        for axis, record in enumerate(records):
            if record["method"] == "aligned":
                self.assertGreater(float(session.party_scores[:, axis] @ term.party_scores[:, axis]), 0)
        # A reference orthogonal to the session axis falls back to the anchor rule.
        orthogonal = {p: (0.0, 0.0) for p in parties}
        records = align_to_reference(session, orthogonal)
        self.assertEqual([r["method"] for r in records], ["anchor", "anchor"])
        self.assertGreater(session.party_scores[parties.index("M"), 0], 0)

    def test_session_summary_records_its_reference_and_is_deterministic(self) -> None:
        votes = synthetic_votes()
        term = build_summary(votes, scope=SCOPE, provenance=PROVENANCE)
        reference = {"label": "term-2022-26", "party_coordinates": term["map"]["party_coordinates"]}
        flipped = {p: [-x, -y] for p, (x, y) in reference["party_coordinates"].items()}
        session = build_summary(votes[:100], scope=SCOPE, provenance=PROVENANCE, map_reference=reference)
        self.assertEqual(session["map"]["orientation_reference"], "term-2022-26")
        self.assertIn("term-2022-26", session["map"]["orientation"])
        for axis in (0, 1):
            dot = sum(session["map"]["party_coordinates"][p][axis] * reference["party_coordinates"][p][axis] for p in flipped)
            self.assertGreater(dot, 0)
        against_flipped = build_summary(
            votes[:100], scope=SCOPE, provenance=PROVENANCE, map_reference={"label": "x", "party_coordinates": flipped}
        )
        self.assertEqual(
            against_flipped["map"]["party_coordinates"]["V"], [-c for c in session["map"]["party_coordinates"]["V"]]
        )
        again = build_summary(list(reversed(votes[:100])), scope=SCOPE, provenance=PROVENANCE, map_reference=reference)
        self.assertEqual(dumps(session), dumps(again))


class BlocsTests(unittest.TestCase):
    def test_latest_approved_pm_vote_defines_the_sides(self) -> None:
        approved = pm_vote("PM1", dict(RIGHT, C="Avstår"), date_="2026-10-15", hangar="1")  # Nej 149
        rejected = pm_vote("PM2", {**RIGHT, "SD": "Nej"}, date_="2026-11-03", hangar="2")  # Nej 222
        other = vote_from_rows(rows_for_vote("X", RIGHT, hangar_id="3"), source="bulk")
        blocs = find_blocs([other, rejected, approved])
        self.assertEqual(blocs["source_vote_id"], "PM1")
        self.assertEqual(blocs["government_side"], ["M", "L", "KD", "SD"])
        self.assertEqual(blocs["opposition"], ["S", "V", "MP"])
        self.assertEqual(blocs["abstained"], ["C"])
        self.assertEqual(blocs["no_line"], [])
        self.assertEqual([v["approved"] for v in pm_votes_considered([approved, rejected, other])], [True, False])

    def test_rejection_needs_175_nej_and_the_2022_title_form_counts(self) -> None:
        # S 107 + SD 73 = 180 Nej: rejected.
        rejected = pm_vote("PM", {"S": "Nej", "SD": "Nej", "M": "Ja"}, date_="2026-10-15", hangar="1")
        self.assertIsNone(find_blocs([rejected]))
        # S 107 + V 24 + C 24 + MP 18 = 173 Nej: approved (as on 2022-10-17).
        old_title = pm_vote("PM", RIGHT, date_="2022-10-17", hangar="1", title="Prövning av förslag till statsminister", rm="2022/23")
        self.assertTrue(is_pm_vote(old_title))
        self.assertEqual(find_blocs([old_title])["government_side"], ["M", "L", "KD", "SD"])
        no_confidence = pm_vote("MV", RIGHT, date_="2024-01-17", hangar="1", title="Prövning av yrkande om misstroendeförklaring")
        self.assertFalse(is_pm_vote(no_confidence))
        self.assertIsNone(find_blocs([no_confidence]))

    BLOCS = {"government_side": ["M", "KD", "L", "SD"], "opposition": ["S", "V", "MP", "C"], "abstained": []}
    ALIGNED = dict(M="Ja", KD="Ja", L="Ja", SD="Ja", S="Nej", V="Nej", MP="Nej", C="Nej")

    def flags(self, ja, nej, blocs=BLOCS, **lines):
        party_lines = {p: line(v) for p, v in lines.items()}
        return cross_bloc(party_lines, blocs, ja, nej), gov_split(party_lines, blocs)

    def test_cross_bloc_and_gov_split(self) -> None:
        aligned = self.ALIGNED
        # Bloc against bloc, either way round: neither flag.
        self.assertEqual(self.flags(180, 160, **aligned), (False, False))
        self.assertEqual(self.flags(160, 180, **{p: ("Nej" if v == "Ja" else "Ja") for p, v in aligned.items()}), (False, False))
        # S + SD vs the rest: the outcome cut across both blocs.
        s_sd = dict(M="Nej", KD="Nej", L="Nej", SD="Ja", S="Ja", V="Nej", MP="Nej", C="Nej")
        self.assertEqual(self.flags(180, 160, **s_sd), (True, True))
        self.assertEqual(self.flags(160, 180, **s_sd), (True, True))  # also when S+SD lose
        # A united government joined by some of the opposition: no losing government party.
        self.assertEqual(self.flags(200, 140, **dict(aligned, C="Ja", MP="Ja")), (False, False))
        self.assertEqual(self.flags(170, 170, **s_sd), (False, True))  # tie: no outcome
        # The government split only through Avstår counts for neither flag.
        avstar = dict(aligned, SD="Avstår", C="Ja")
        self.assertEqual(self.flags(150, 140, **avstar), (False, False))
        # Ties and missing lines are ignored like Avstår.
        self.assertEqual(self.flags(150, 140, **dict(aligned, SD=None, C="Ja")), (False, False))
        self.assertEqual(self.flags(150, 140, **dict(aligned, KD="Nej")), (False, True))
        self.assertEqual(self.flags(10, 20, blocs=None, **aligned), (None, None))
        # An abstained-group party is not on either side.
        with_abstained = dict(self.BLOCS, government_side=["M", "KD", "L"], abstained=["SD"])
        self.assertEqual(self.flags(180, 160, blocs=with_abstained, **dict(s_sd, M="Ja", KD="Ja", L="Ja")), (False, False))

    def test_vote_flags_columns(self) -> None:
        votes = synthetic_votes(n_votes=8)
        lines = {v.votering_id: party_lines_for_vote(v) for v in votes}
        flags = vote_flags(votes, lines, {"government_side": ["M", "KD", "L", "SD"], "opposition": ["S", "V", "MP"]})
        columns = flags["columns"]
        self.assertEqual(
            flags["fields"], ["ja", "nej", "avstar", "franvarande", "margin", "against_line", "cross_bloc", "gov_split"]
        )
        self.assertTrue(all(len(columns[f]) == 8 for f in flags["fields"]))
        for i, vote in enumerate(votes):
            counts = vote.counts()
            self.assertEqual(columns["ja"][i] + columns["nej"][i] + columns["avstar"][i] + columns["franvarande"][i], 349)
            self.assertEqual(columns["margin"][i], abs(counts["Ja"] - counts["Nej"]))
        self.assertEqual(columns["against_line"][3], 1)  # the rebel S010 on V003
        self.assertTrue(set(columns["cross_bloc"]) <= {0, 1})
        self.assertTrue(set(columns["gov_split"]) <= {0, 1})


class BuildContextTests(unittest.TestCase):
    def test_sessions_take_blocs_and_orientation_from_their_term(self) -> None:
        rows = synthetic_rows(n_votes=60, rm="2026/27")
        pm_rows = rows_for_vote("PM1", RIGHT, rm="2026/27", beteckning="1015-1", punkt="0", avser="frågan", hangar_id="1")
        responses = {
            bulk_url("2026/27"): bulk_archive(rows + [pm_rows]),
            register_url("2026/27"): register_payload(rows + [pm_rows]),
            "https://data.riksdagen.se/utskottsforslag/HE191015-1.json": json.dumps(
                {"utskottsforslag": {"dokument": {"dok_id": "HE191015-1", "rm": "2026/27", "beteckning": "1015-1",
                                                  "doktyp": "votering", "titel": "Omröstning: " + PM_TITLE}}}
            ).encode(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"

            def run(sessions, today):
                store = RawStore(cache_dir=Path(tmp) / "cache", manifest_path=out / "manifest.json",
                                 http_get=FakeHttp(responses), now=lambda: "2026-11-19T12:00:00Z")
                build(sessions, store, out, generated_at_utc="2026-11-19T12:00:00Z",
                      code={"code_commit": "c", "code_dirty": False}, today=today)

            run(["2026/27"], date(2026, 11, 19))
            term = json.loads((out / "term-2026-30/summary.json").read_text(encoding="utf-8"))
            session = json.loads((out / "2026-27/summary.json").read_text(encoding="utf-8"))
            self.assertEqual(term["blocs"]["source_vote_id"], "PM1")
            self.assertEqual(session["blocs"], term["blocs"])
            self.assertEqual(session["map"]["orientation_reference"], "term-2026-30")
            self.assertEqual(len(session["vote_flags"]["columns"]["cross_bloc"]), 61)
            self.assertEqual(term["pm_votes_considered"][0]["id"], "PM1")

            # A run without the whole term reads the term context from disk:
            # on 2027-09-02 the term has two begun sessions, only one is rebuilt.
            before = (out / "2026-27/summary.json").read_bytes()
            run(["2026/27"], date(2027, 9, 2))
            self.assertEqual((out / "2026-27/summary.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
