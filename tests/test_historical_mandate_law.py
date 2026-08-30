"""Tests for historically versioned Riksdag mandate law (PRE_2018 / POST_2018).

The overriding requirement is that adding the historical version changed nothing
about production: ``allocate_riksdag_seats`` called without a ``law`` argument
must behave exactly as before, with first divisor 1.2 and mandate return.

The certified 2010 and 2014 golden reproductions live with the research data in
``diagnostics/election_noise_v2/historical_seat_extension`` and are skipped here
when that research data is absent, so this module never makes CI depend on it.
"""

from __future__ import annotations

import csv
from fractions import Fraction
import json
from pathlib import Path
import unittest

from scripts.mandates.allocator import allocate_riksdag_seats
from scripts.mandates.config import (
    DEFAULT_PROCESSED_DIR,
    FIXED_SEATS_2018,
    FIXED_SEATS_2022,
    TOTAL_RIKSDAG_SEATS,
)
from scripts.mandates.law import (
    FIRST_DIVISOR_BY_LAW,
    MandateLaw,
    mandate_law_for_election_year,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = REPO_ROOT / "diagnostics" / "election_noise_v2" / "historical_seat_extension" / "processed"
SEAT_PARTIES = ("M", "L", "C", "KD", "S", "V", "MP", "SD")


def _votes_from_csv(path: Path, year: int) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["election_year"]) != year:
                continue
            cc = f"{int(row['constituency_code']):02d}"
            out.setdefault(cc, {})[row["party"]] = int(row["votes"])
    return out


def _seats_from_csv(path: Path, year: int, column: str) -> dict[str, int]:
    totals: dict[str, int] = {p: 0 for p in SEAT_PARTIES}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["election_year"]) != year:
                continue
            if row["party"] in totals:
                totals[row["party"]] += int(row[column])
    return totals


class MandateLawVersionTest(unittest.TestCase):
    """The law module maps election years deterministically and never uses the clock."""

    def test_year_mapping_is_deterministic_and_correct(self) -> None:
        for year in (2010, 2014):
            cfg = mandate_law_for_election_year(year)
            self.assertIs(cfg.law, MandateLaw.PRE_2018)
            self.assertEqual(cfg.first_divisor, Fraction(7, 5))
            self.assertFalse(cfg.has_mandate_return)
        for year in (2018, 2022, 2026, 2030):
            cfg = mandate_law_for_election_year(year)
            self.assertIs(cfg.law, MandateLaw.POST_2018)
            self.assertEqual(cfg.first_divisor, Fraction(6, 5))
            self.assertTrue(cfg.has_mandate_return)

    def test_mapping_is_pure(self) -> None:
        self.assertEqual(
            mandate_law_for_election_year(2014), mandate_law_for_election_year(2014)
        )
        self.assertNotEqual(
            mandate_law_for_election_year(2014).law,
            mandate_law_for_election_year(2018).law,
        )

    def test_divisors_are_exact_fractions(self) -> None:
        self.assertEqual(FIRST_DIVISOR_BY_LAW[MandateLaw.POST_2018], Fraction(6, 5))
        self.assertEqual(FIRST_DIVISOR_BY_LAW[MandateLaw.PRE_2018], Fraction(7, 5))

    def test_rejects_bad_year(self) -> None:
        with self.assertRaises(ValueError):
            mandate_law_for_election_year(1968)
        with self.assertRaises(TypeError):
            mandate_law_for_election_year("2014")  # type: ignore[arg-type]


class ProductionDefaultUnchangedTest(unittest.TestCase):
    """Adding PRE_2018 must not perturb the production path in any way."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.votes_csv = DEFAULT_PROCESSED_DIR / "historical_constituency_votes.csv"
        cls.mandates_csv = DEFAULT_PROCESSED_DIR / "historical_certified_mandates.csv"

    def _golden(self, year: int, fixed: dict[str, int], expected: dict[str, int]) -> None:
        votes = _votes_from_csv(self.votes_csv, year)
        # Production call signature: no law argument.
        res = allocate_riksdag_seats(constituency_votes=votes, fixed_seats_by_constituency=fixed)
        self.assertEqual(res.total_seats, TOTAL_RIKSDAG_SEATS)
        self.assertEqual(res.law, MandateLaw.POST_2018.value)
        self.assertEqual(res.set_aside_parties, ())
        for p, exp in expected.items():
            self.assertEqual(res.final_seats_by_party.get(p, 0), exp, f"{year} {p}")
        certified = _seats_from_csv(self.mandates_csv, year, "total_seats")
        for p in SEAT_PARTIES:
            self.assertEqual(res.final_seats_by_party.get(p, 0), certified[p], f"{year} {p}")

    def test_2018_golden_unchanged_via_default_path(self) -> None:
        self._golden(
            2018,
            FIXED_SEATS_2018,
            {"S": 100, "M": 70, "SD": 62, "C": 31, "V": 28, "KD": 22, "L": 20, "MP": 16},
        )

    def test_2022_golden_unchanged_via_default_path(self) -> None:
        self._golden(
            2022,
            FIXED_SEATS_2022,
            {"S": 107, "SD": 73, "M": 68, "V": 24, "C": 24, "KD": 19, "MP": 18, "L": 16},
        )

    def test_explicit_post_2018_matches_default(self) -> None:
        votes = _votes_from_csv(self.votes_csv, 2022)
        a = allocate_riksdag_seats(votes, FIXED_SEATS_2022)
        b = allocate_riksdag_seats(votes, FIXED_SEATS_2022, law=MandateLaw.POST_2018)
        self.assertEqual(a.final_seats_by_party, b.final_seats_by_party)
        self.assertEqual(a.final_seats_by_party_constituency, b.final_seats_by_party_constituency)
        self.assertEqual(a.national_entitlement, b.national_entitlement)
        self.assertEqual(a.returned_or_reallocated_seats, b.returned_or_reallocated_seats)
        self.assertEqual(len(a.event_log), len(b.event_log))
        for ea, eb in zip(a.event_log, b.event_log):
            self.assertEqual(ea, eb)

    def test_default_first_divisor_is_still_one_point_two(self) -> None:
        votes = _votes_from_csv(self.votes_csv, 2022)
        res = allocate_riksdag_seats(votes, FIXED_SEATS_2022)
        first_seat = next(e for e in res.event_log if e.phase == "fixed")
        self.assertEqual(first_seat.divisor, Fraction(6, 5))

    def test_deterministic_tie_breaking_is_stable(self) -> None:
        votes = _votes_from_csv(self.votes_csv, 2018)
        a = allocate_riksdag_seats(votes, FIXED_SEATS_2018)
        b = allocate_riksdag_seats(votes, FIXED_SEATS_2018)
        self.assertEqual(
            [(e.sequence, e.phase, e.party, e.constituency_code) for e in a.event_log],
            [(e.sequence, e.phase, e.party, e.constituency_code) for e in b.event_log],
        )


@unittest.skipUnless(
    (RESEARCH_DIR / "certified_mandates_2010_2014.csv").exists(),
    "historical seat-extension research data not present",
)
class PreTwentyEighteenGoldenTest(unittest.TestCase):
    """PRE_2018 reproduces the certified 2010 and 2014 allocations exactly."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.votes_2010 = _votes_from_csv(
            RESEARCH_DIR / "constituency_party_votes_2006_2010.csv", 2010
        )
        cls.votes_2014 = _votes_from_csv(
            REPO_ROOT / "data" / "processed" / "geography" / "constituency_party_votes_2014_2022.csv",
            2014,
        )
        cls.fixed = {
            int(y): {k: int(v) for k, v in d.items()}
            for y, d in json.loads((RESEARCH_DIR / "fixed_seats_by_year.json").read_text())[
                "fixed_seats_by_year"
            ].items()
        }
        cls.certified_csv = RESEARCH_DIR / "certified_mandates_2010_2014.csv"

    def _run(self, year: int, votes: dict[str, dict[str, int]]):
        cfg = mandate_law_for_election_year(year)
        return allocate_riksdag_seats(
            votes,
            self.fixed[year],
            first_divisor=cfg.first_divisor,
            law=cfg.law,
            scenario_id=f"test_{year}",
        )

    def test_2010_certified_exact(self) -> None:
        res = self._run(2010, self.votes_2010)
        self.assertEqual(res.total_seats, TOTAL_RIKSDAG_SEATS)
        self.assertEqual(res.law, MandateLaw.PRE_2018.value)
        expected = {"S": 112, "M": 107, "MP": 25, "L": 24, "C": 23, "SD": 20, "KD": 19, "V": 19}
        for p, exp in expected.items():
            self.assertEqual(res.final_seats_by_party.get(p, 0), exp, f"2010 {p}")
        certified = _seats_from_csv(self.certified_csv, 2010, "total_seats")
        self.assertEqual({p: res.final_seats_by_party.get(p, 0) for p in SEAT_PARTIES}, certified)
        # Prop. 2013/14:48 §4.1.1 records that S and M were over-represented in 2010.
        self.assertEqual(set(res.set_aside_parties), {"S", "M"})

    def test_2014_certified_exact(self) -> None:
        res = self._run(2014, self.votes_2014)
        self.assertEqual(res.total_seats, TOTAL_RIKSDAG_SEATS)
        expected = {"S": 113, "M": 84, "SD": 49, "MP": 25, "C": 22, "V": 21, "L": 19, "KD": 16}
        for p, exp in expected.items():
            self.assertEqual(res.final_seats_by_party.get(p, 0), exp, f"2014 {p}")
        certified = _seats_from_csv(self.certified_csv, 2014, "total_seats")
        self.assertEqual({p: res.final_seats_by_party.get(p, 0) for p in SEAT_PARTIES}, certified)

    def test_pre_2018_never_returns_seats(self) -> None:
        res = self._run(2014, self.votes_2014)
        self.assertEqual(
            res.final_fixed_seats_by_party_constituency,
            res.initial_fixed_seats_by_party_constituency,
        )
        self.assertTrue(
            all(v == 0 for c in res.returned_or_reallocated_seats.values() for v in c.values())
        )
        self.assertFalse(any(e.phase == "excess_retracted" for e in res.event_log))
        self.assertFalse(any(e.phase == "returned_reallocated" for e in res.event_log))

    def test_set_aside_parties_keep_their_fixed_seats(self) -> None:
        res = self._run(2010, self.votes_2010)
        for p in res.set_aside_parties:
            self.assertEqual(res.national_entitlement[p], res.final_national_fixed_seats[p])
            self.assertEqual(res.national_adjustment_seats.get(p, 0), 0)

    def test_wrong_law_does_not_reproduce_certified(self) -> None:
        """Guards against silently allocating a pre-2018 election under current law."""
        res = allocate_riksdag_seats(
            self.votes_2010, self.fixed[2010], law=MandateLaw.POST_2018
        )
        certified = _seats_from_csv(self.certified_csv, 2010, "total_seats")
        self.assertNotEqual(
            {p: res.final_seats_by_party.get(p, 0) for p in SEAT_PARTIES}, certified
        )


if __name__ == "__main__":
    unittest.main()
