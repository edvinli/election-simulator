"""Riksdag roll-call rules: party lines, against-line, agreement and committee parsing."""

import unittest

from scripts.riksdag_votes.agreement import agreement_matrices
from scripts.riksdag_votes.committees import committee_key, parse_committee
from scripts.riksdag_votes.ingest import vote_from_rows
from scripts.riksdag_votes.party_lines import against_line, order_parties, party_line, party_lines_for_vote
from tests.riksdag_votes_fixtures import chamber, rows_for_vote


class PartyLineTests(unittest.TestCase):
    def test_plurality_ignores_absence(self) -> None:
        line = party_line(["Ja"] * 3 + ["Nej"] * 2 + ["Frånvarande"] * 10)
        self.assertEqual((line.line, line.tie, line.no_cast), ("Ja", False, False))
        self.assertEqual(line.counts, (3, 2, 0, 10))

    def test_avstar_is_a_stance(self) -> None:
        line = party_line(["Avstår"] * 5 + ["Ja"] * 2 + ["Nej"] * 2)
        self.assertEqual(line.line, "Avstår")

    def test_tie_gives_null_line_with_flag(self) -> None:
        line = party_line(["Ja", "Ja", "Avstår", "Avstår", "Nej"])
        self.assertIsNone(line.line)
        self.assertTrue(line.tie)
        self.assertFalse(line.no_cast)

    def test_all_absent_gives_null_line_no_cast(self) -> None:
        line = party_line(["Frånvarande"] * 4)
        self.assertIsNone(line.line)
        self.assertFalse(line.tie)
        self.assertTrue(line.no_cast)

    def test_independents_have_no_line(self) -> None:
        people = chamber(independents=3)
        vote = vote_from_rows(rows_for_vote("V", {"S": "Nej", "-": "Ja", "M": "Ja"}, people=people), source="bulk")
        lines = party_lines_for_vote(vote)
        self.assertNotIn("-", lines)
        self.assertEqual(lines["S"].counts[1], 104)
        independent = next(b for b in vote.ballots if b.parti == "-")
        self.assertIsNone(against_line(independent, lines))

    def test_against_line(self) -> None:
        vote = vote_from_rows(
            rows_for_vote("V", {"S": "Nej", "M": "Ja"}, overrides={"S000": "Ja", "S001": "Avstår", "S002": "Frånvarande"}),
            source="bulk",
        )
        lines = party_lines_for_vote(vote)
        verdicts = {b.intressent_id: against_line(b, lines) for b in vote.ballots}
        self.assertTrue(verdicts["S000"])
        self.assertTrue(verdicts["S001"])  # Avstår against a Nej line is a departure
        self.assertIsNone(verdicts["S002"])  # absence is not a departure
        self.assertFalse(verdicts["S003"])
        self.assertIsNone(verdicts["V000"])  # V entirely absent: no line

    def test_party_order(self) -> None:
        self.assertEqual(order_parties(["SD", "S", "-", "M", "X"]), ["M", "S", "SD", "X"])


class AgreementTests(unittest.TestCase):
    def test_three_way_and_ja_nej_with_n(self) -> None:
        votes = [
            {"A": "Ja", "B": "Ja"},
            {"A": "Ja", "B": "Avstår"},
            {"A": "Avstår", "B": "Avstår"},
            {"A": "Nej", "B": "Ja"},
            {"A": "Nej", "B": None},
        ]
        result = agreement_matrices(votes, ["A", "B"], min_n=1)
        three, two = result["three_way"], result["ja_nej"]
        self.assertEqual((three["agree"][0][1], three["n"][0][1]), (2, 4))
        self.assertEqual(three["share"][0][1], 0.5)
        self.assertEqual((two["agree"][0][1], two["n"][0][1]), (1, 2))
        self.assertEqual(two["share"][0][1], 0.5)
        self.assertEqual(three["n"][0][0], 5)  # diagonal: votes with a line
        self.assertEqual(three["n"][1][1], 4)
        self.assertEqual(result["n_votes"], 5)

    def test_small_cells_hide_share_keep_n(self) -> None:
        result = agreement_matrices([{"A": "Ja", "B": "Ja"}] * 3, ["A", "B"], min_n=10)
        self.assertIsNone(result["three_way"]["share"][0][1])
        self.assertEqual(result["three_way"]["n"][0][1], 3)

    def test_symmetric(self) -> None:
        votes = [{"A": "Ja", "B": "Nej", "C": "Ja"}, {"A": "Avstår", "B": "Avstår", "C": None}] * 6
        result = agreement_matrices(votes, ["A", "B", "C"])
        for key in ("three_way", "ja_nej"):
            for name in ("agree", "n", "share"):
                matrix = result[key][name]
                self.assertEqual(matrix, [list(row) for row in zip(*matrix)])


class CommitteeTests(unittest.TestCase):
    def test_known_committees(self) -> None:
        for beteckning, expected in [
            ("FiU1", "FiU"),
            ("JuU12", "JuU"),
            ("UFöU3", "UFöU"),
            ("MJU20", "MJU"),
            ("KU4", "KU"),
            ("FöU8", "FöU"),
            ("SoU38", "SoU"),
        ]:
            with self.subTest(beteckning=beteckning):
                parsed = parse_committee(beteckning)
                self.assertEqual((parsed.committee, parsed.kind), (expected, "committee"))

    def test_numeric_and_missing(self) -> None:
        self.assertEqual(parse_committee("0311-1").kind, "numeric")
        self.assertIsNone(parse_committee("0311-1").committee)
        self.assertEqual(parse_committee(None).kind, "missing")
        self.assertEqual(committee_key(None), "övrigt")

    def test_unknown_prefix_logged(self) -> None:
        with self.assertLogs("scripts.riksdag_votes.committees", level="WARNING") as logs:
            parsed = parse_committee("XyU2")
        self.assertEqual((parsed.committee, parsed.kind, parsed.prefix), (None, "unknown_prefix", "XyU"))
        self.assertIn("XyU", logs.output[0])


if __name__ == "__main__":
    unittest.main()
