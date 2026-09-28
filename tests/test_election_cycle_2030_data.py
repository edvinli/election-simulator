"""The 2030 cycle's data and cycle rules, and that the 2026 cycle is unchanged.

Three kinds of check:

* the 2026 constituency result added to the geography tables is the certified
  one: it sums to the certified national votes, it reproduces the existing
  2022 rows from the same file's own "previous election" fields, and the
  statutory allocator turns it into the certified seats;
* the cycle rules reproduce every published 2026-cycle date and declare the
  2030 stand-ins explicitly;
* a forecast can never use its own election as its geography baseline.
"""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import unittest

import pandas as pd

from scripts.elections.config import ELECTIONS
from scripts.geography.config import CONSTITUENCY_NAME_TO_CODE, REST_MANDATE_LABEL
from scripts.geography.process import VAL2026_RESULT_MANIFEST, load_val2026_constituency_votes
from scripts.mandates.allocator import allocate_riksdag_seats
from scripts.mandates.config import FIXED_SEATS_2022, FIXED_SEATS_2026
from scripts.simulator import election_cycles as cycles


ROOT = Path(__file__).resolve().parents[1]
GEOGRAPHY = ROOT / "data" / "processed" / "geography"
PARTIES_8 = ("M", "L", "C", "KD", "S", "V", "MP", "SD")


class Val2026ConstituencyVotesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.votes = load_val2026_constituency_votes()
        cls.manifest = json.loads(VAL2026_RESULT_MANIFEST.read_text(encoding="utf-8"))
        cls.raw = json.loads((VAL2026_RESULT_MANIFEST.parent / cls.manifest["raw_path"]).read_text(encoding="utf-8"))

    def test_the_constituencies_sum_to_the_certified_national_result(self) -> None:
        for party in PARTIES_8:
            with self.subTest(party=party):
                self.assertEqual(sum(c["votes"][party] for c in self.votes.values()),
                                 self.manifest["parties"][party]["votes"])
        self.assertEqual(sum(c["valid"] for c in self.votes.values()),
                         self.manifest["valid_national_votes"])

    def test_the_same_file_reproduces_the_existing_2022_rows(self) -> None:
        """The raw file's previous-election fields are the 2022 result.

        If they match the 2022 rows already in the table cell for cell, the
        2026 rows read from the same fields' current values share their
        definition: same denominator, same REST.
        """

        table = pd.read_csv(GEOGRAPHY / "constituency_party_votes_2014_2022.csv", dtype={"constituency_code": str})
        rows_2022 = table[table["election_year"] == 2022]
        for constituency in self.raw["valkretsar"]:
            code = CONSTITUENCY_NAME_TO_CODE[constituency["namn"].strip()]
            counted = constituency["rosterPaverkaMandat"]
            previous: dict[str, int] = {}
            for row in counted["partiroster"]:
                party = row["partiforkortning"] if row["partiforkortning"] in PARTIES_8 else "REST"
                previous[party] = previous.get(party, 0) + int(row["antalRosterForegaendeVal"])
            recorded = rows_2022[rows_2022["constituency_code"] == code]
            with self.subTest(constituency=code):
                self.assertEqual(int(recorded["constituency_valid_votes"].iloc[0]),
                                 int(counted["antalRosterForegaendeVal"]))
                for party, value in previous.items():
                    self.assertEqual(int(recorded[recorded["party"] == party]["votes"].iloc[0]), value, party)

    def test_the_allocator_reproduces_the_certified_2026_seats(self) -> None:
        constituency_votes = {
            code: {(REST_MANDATE_LABEL if party == "REST" else party): count
                   for party, count in entry["votes"].items()}
            for code, entry in self.votes.items()
        }
        allocation = allocate_riksdag_seats(constituency_votes, FIXED_SEATS_2026)
        self.assertEqual({p: allocation.final_seats_by_party.get(p, 0) for p in PARTIES_8},
                         {p: self.manifest["parties"][p]["seats"] for p in PARTIES_8})


class GeographyTablesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.votes = pd.read_csv(GEOGRAPHY / "constituency_party_votes_2014_2022.csv", dtype={"constituency_code": str})
        cls.electorates = pd.read_csv(GEOGRAPHY / "constituency_electorates_2014_2026.csv", dtype={"constituency_code": str})

    def test_2026_votes_are_in_the_table(self) -> None:
        rows = self.votes[self.votes["election_year"] == 2026]
        self.assertEqual(len(rows), 29 * 9)
        loaded = load_val2026_constituency_votes()
        for code, entry in loaded.items():
            sub = rows[rows["constituency_code"] == code]
            with self.subTest(constituency=code):
                self.assertEqual(dict(zip(sub["party"], sub["votes"].astype(int))), entry["votes"])

    def test_2026_electorate_rows_carry_their_certified_valid_votes(self) -> None:
        rows = self.electorates[self.electorates["election_year"] == 2026].set_index("constituency_code")
        loaded = load_val2026_constituency_votes()
        for code, entry in loaded.items():
            with self.subTest(constituency=code):
                self.assertEqual(int(rows.loc[code, "valid_votes"]), entry["valid"])
                self.assertAlmostEqual(rows.loc[code, "turnout_rate"],
                                       entry["valid"] / rows.loc[code, "eligible_voters"], places=6)

    def test_the_2030_electorate_is_the_declared_2026_stand_in(self) -> None:
        by_year = {year: frame.set_index("constituency_code")
                   for year, frame in self.electorates.groupby("election_year")}
        self.assertEqual(len(by_year[2030]), 29)
        self.assertTrue(by_year[2030]["valid_votes"].isna().all())
        self.assertEqual(by_year[2030]["eligible_voters"].to_dict(), by_year[2026]["eligible_voters"].to_dict())


class ElectionCycleRulesTests(unittest.TestCase):
    def test_ordinary_election_days(self) -> None:
        for year, metadata in ELECTIONS.items():
            with self.subTest(year=year):
                self.assertEqual(cycles.ordinary_election_date(year), metadata.election_date)
        self.assertEqual(cycles.ordinary_election_date(2026), date(2026, 9, 13))
        self.assertEqual(cycles.ordinary_election_date(2030), date(2030, 9, 8))

    def test_the_2026_cycle_dates_are_reproduced(self) -> None:
        self.assertEqual(cycles.history_start_for("2026-09-13"), date(2022, 9, 18))
        self.assertEqual(cycles.dynamics_cap_start_for("2026-09-13"), date(2026, 5, 24))

    def test_the_2030_cycle_dates(self) -> None:
        self.assertEqual(cycles.previous_election_date("2030-09-08"), date(2026, 9, 13))
        self.assertEqual(cycles.history_start_for("2030-09-08"), date(2026, 9, 20))
        self.assertEqual(cycles.dynamics_cap_start_for("2030-09-08"), date(2030, 5, 19))

    def test_baselines_are_the_previous_election(self) -> None:
        self.assertEqual({y: cycles.geography_baseline_year_for(y) for y in (2018, 2022, 2026, 2030)},
                         {2018: 2014, 2022: 2018, 2026: 2022, 2030: 2026})

    def test_fixed_seats(self) -> None:
        self.assertIs(cycles.fixed_seats_for(2022), FIXED_SEATS_2022)
        self.assertIs(cycles.fixed_seats_for(2026), FIXED_SEATS_2026)
        self.assertIs(cycles.fixed_seats_for(2030), FIXED_SEATS_2026)
        self.assertIn(2030, cycles.FIXED_SEATS_STAND_IN_YEARS)
        # Historical behaviour for pre-2018 targets is kept.
        self.assertIs(cycles.fixed_seats_for(2014), FIXED_SEATS_2026)
        with self.assertRaises(ValueError):
            cycles.fixed_seats_for(2034)


class Cycle2030SimulationTests(unittest.TestCase):
    def test_a_2030_forecast_runs_on_the_2026_baseline(self) -> None:
        from scripts.simulator.engine import simulate_election

        baseline = cycles.geography_baseline_year_for(2030)
        result = simulate_election(as_of="2026-09-11", election_date="2030-09-08",
                                   samples=200, seed=7, baseline_year=baseline)
        self.assertEqual(result.manifest["model_config"]["geography_baseline_year"], 2026)
        self.assertTrue((result.seats_matrix.sum(axis=1) == 349).all())


class FingerprintScopeTests(unittest.TestCase):
    """The fingerprint hashes what a target's forecast may depend on, only."""

    def _static(self, election: date, data: Path) -> str:
        from scripts.forecast_history.effective_inputs import EffectiveInputs

        return EffectiveInputs(data, election_date=election, seed=1).static

    def test_2026_and_2030_targets_hash_different_geography(self) -> None:
        data = ROOT / "data" / "processed"
        self.assertNotEqual(self._static(date(2026, 9, 13), data), self._static(date(2030, 9, 8), data))

    def test_a_target_ignores_its_own_outcome_and_later_elections(self) -> None:
        """Certifying 2026, or adding 2030 rows, leaves 2026 fingerprints alone."""

        import shutil
        import tempfile

        data = ROOT / "data" / "processed"
        before = self._static(date(2026, 9, 13), data)
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "processed"
            shutil.copytree(data, copy, ignore=shutil.ignore_patterns("prospective_forecasts", "simulations"))
            electorates = copy / "geography" / "constituency_electorates_2014_2026.csv"
            frame = pd.read_csv(electorates, dtype={"constituency_code": str})
            own = frame["election_year"] == 2026
            frame.loc[own, ["valid_votes", "turnout_rate"]] = None
            frame = frame[frame["election_year"] <= 2026]
            frame.to_csv(electorates, index=False)
            self.assertEqual(self._static(date(2026, 9, 13), copy), before)
            # The 2030 fingerprint does move with its baseline's certified votes.
            self.assertNotEqual(self._static(date(2030, 9, 8), copy), self._static(date(2030, 9, 8), data))


if __name__ == "__main__":
    unittest.main()
