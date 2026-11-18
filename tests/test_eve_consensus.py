"""The amended election-eve consensus (election-noise residual amendments 001 and 002)."""

from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from scripts.election_residuals.config import DEFAULT_POLLS_FILE
from scripts.election_residuals.consensus import build_election_polling_consensus
from scripts.election_residuals.eve_consensus import build_eve_polling_consensus, eligible_eve_rows

ELECTIONS = (
    date(2002, 9, 15), date(2006, 9, 17), date(2010, 9, 19), date(2014, 9, 14),
    date(2018, 9, 9), date(2022, 9, 11), date(2026, 9, 13),
)
PARTIES = ("M", "L", "C", "KD", "S", "V", "MP", "SD")


def _poll(pid, pollster, start, end, published, n, value=10.0):
    return [
        {"poll_id": pid, "pollster": pollster, "pollster_original": pollster, "interview_start": start,
         "interview_end": end, "publication_date": published, "sample_size": n, "party": p, "support": value}
        for p in PARTIES
    ]


class EveConsensusOnCommittedPolls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.polls = pd.read_csv(DEFAULT_POLLS_FILE)

    def test_equals_committed_builder_when_no_rule_applies(self):
        # Removing the polls either amendment acts on (election-day polls, and
        # polls missing a key the committed pivot drops) leaves data on which the
        # committed builder and the amended builder must agree exactly.
        for e in ELECTIONS:
            eligible = eligible_eve_rows(e, self.polls)
            clean_ids = set(eligible["poll_id"]) - set(
                eligible.loc[eligible["sample_size"].isna() | eligible["interview_start"].isna(), "poll_id"])
            clean = eligible[eligible["poll_id"].isin(clean_ids)]
            old = build_election_polling_consensus(e, clean, window_days=14)
            new = build_eve_polling_consensus(e, clean, window_days=14)
            self.assertEqual(old.consensus_composition, new.consensus_composition, e)
            self.assertEqual(old.contributing_polls, new.contributing_polls, e)

    def test_no_contributing_poll_is_from_election_day(self):
        for e in ELECTIONS:
            for p in build_eve_polling_consensus(e, self.polls).contributing_polls:
                self.assertLess(p.interview_end, e)
                self.assertLess(p.publication_date, e)

    def test_missing_key_polls_now_contribute(self):
        expected = {  # amendment 002, section 3
            date(2002, 9, 15): ("Skop", date(2002, 9, 13)),
            date(2006, 9, 17): ("Ipsos", date(2006, 9, 16)),
            date(2022, 9, 11): ("Infostat", date(2022, 9, 7)),
        }
        for e, (pollster, end) in expected.items():
            by_pollster = {p.pollster: p for p in build_eve_polling_consensus(e, self.polls).contributing_polls}
            self.assertIn(pollster, by_pollster, e)
            self.assertEqual(by_pollster[pollster].interview_end, end, e)


class EveConsensusRules(unittest.TestCase):
    E = date(2030, 9, 8)

    def test_election_day_and_same_day_publication_are_excluded(self):
        rows = (_poll("a", "A", "2030-09-01", "2030-09-05", "2030-09-07", 1000, 10.0)
                + _poll("b", "A", "2030-09-08", "2030-09-08", "2030-09-08", 9000, 20.0)
                + _poll("c", "B", "2030-09-01", "2030-09-06", "2030-09-08", 1000, 30.0)
                + _poll("d", "B", "2030-08-30", "2030-09-02", "2030-09-04", 1000, 11.0))
        c = build_eve_polling_consensus(self.E, pd.DataFrame(rows))
        self.assertEqual(sorted(p.poll_id for p in c.contributing_polls), ["a", "d"])

    def test_missing_sample_size_gets_weight_one(self):
        rows = _poll("a", "A", "2030-09-01", "2030-09-05", "2030-09-06", None, 10.0)
        (poll,) = build_eve_polling_consensus(self.E, pd.DataFrame(rows)).contributing_polls
        self.assertIsNone(poll.sample_size)
        self.assertTrue(poll.sample_size_missing)
        self.assertEqual(poll.weight, 1.0)

    def test_missing_start_sorts_last_in_tie_break(self):
        rows = (_poll("z", "A", None, "2030-09-05", "2030-09-06", 1000, 10.0)
                + _poll("a", "A", "2030-09-01", "2030-09-05", "2030-09-06", 1000, 12.0))
        (poll,) = build_eve_polling_consensus(self.E, pd.DataFrame(rows)).contributing_polls
        self.assertEqual(poll.poll_id, "a")

    def test_missing_start_poll_is_used_when_it_is_latest(self):
        rows = (_poll("z", "A", None, "2030-09-06", "2030-09-07", 1000, 10.0)
                + _poll("a", "A", "2030-09-01", "2030-09-05", "2030-09-06", 1000, 12.0))
        (poll,) = build_eve_polling_consensus(self.E, pd.DataFrame(rows)).contributing_polls
        self.assertEqual(poll.poll_id, "z")
        self.assertIsNone(poll.interview_start)


if __name__ == "__main__":
    unittest.main()
