"""Unit and regression tests for official Swedish parliamentary election results pipeline."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import unittest
import pandas as pd

from scripts.elections.config import (
    CANONICAL_PARTIES,
    ELECTIONS,
    normalize_party_name_or_code,
)
from scripts.elections.normalize import (
    build_canonical_results_table,
    build_source_parties_table,
    normalize_all_elections,
)
from scripts.elections.parse import (
    ElectionParsedData,
    SourcePartyResult,
    parse_election_by_year,
)
from scripts.elections.validate import (
    ElectionValidationError,
    validate_election_results,
    validate_processed_files,
)


RAW_ELECTIONS_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "elections"
PROCESSED_ELECTIONS_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "elections"


class PartyNormalizationTests(unittest.TestCase):
    """Test party abbreviation and name normalization rules."""

    def test_canonical_party_mappings(self) -> None:
        self.assertEqual(normalize_party_name_or_code("M"), "M")
        self.assertEqual(normalize_party_name_or_code("Moderaterna"), "M")
        self.assertEqual(normalize_party_name_or_code("Moderata Samlingspartiet"), "M")

        # Folkpartiet / Liberalerna mapping to L
        self.assertEqual(normalize_party_name_or_code("L"), "L")
        self.assertEqual(normalize_party_name_or_code("Liberalerna"), "L")
        self.assertEqual(normalize_party_name_or_code("Liberalerna (tidigare Folkpartiet)"), "L")
        self.assertEqual(normalize_party_name_or_code("FP"), "L")
        self.assertEqual(normalize_party_name_or_code("Folkpartiet"), "L")
        self.assertEqual(normalize_party_name_or_code("Folkpartiet liberalerna"), "L")

        # Left / Greens / Christian Democrats
        self.assertEqual(normalize_party_name_or_code("S"), "S")
        self.assertEqual(normalize_party_name_or_code("Arbetarepartiet-Socialdemokraterna"), "S")
        self.assertEqual(normalize_party_name_or_code("V"), "V")
        self.assertEqual(normalize_party_name_or_code("Vänsterpartiet"), "V")
        self.assertEqual(normalize_party_name_or_code("MP"), "MP")
        self.assertEqual(normalize_party_name_or_code("Miljöpartiet de gröna"), "MP")
        self.assertEqual(normalize_party_name_or_code("KD"), "KD")
        self.assertEqual(normalize_party_name_or_code("Kristdemokraterna"), "KD")
        self.assertEqual(normalize_party_name_or_code("C"), "C")
        self.assertEqual(normalize_party_name_or_code("Centerpartiet"), "C")
        self.assertEqual(normalize_party_name_or_code("SD"), "SD")
        self.assertEqual(normalize_party_name_or_code("Sverigedemokraterna"), "SD")
        self.assertEqual(normalize_party_name_or_code("FI"), "FI")
        self.assertEqual(normalize_party_name_or_code("Feministiskt initiativ"), "FI")

    def test_other_party_mapping(self) -> None:
        self.assertEqual(normalize_party_name_or_code("Piratpartiet"), "OTHER")
        self.assertEqual(normalize_party_name_or_code("Medborgerlig Samling"), "OTHER")
        self.assertEqual(normalize_party_name_or_code("Alternativ för Sverige"), "OTHER")
        self.assertEqual(normalize_party_name_or_code("Övriga partier"), "OTHER")
        self.assertEqual(normalize_party_name_or_code("Random Party 123"), "OTHER")


class ElectionIntegrityAndFixtureTests(unittest.TestCase):
    """Test official election fixtures, date offsets, vote sums, and validation suite."""

    def test_election_date_fixtures(self) -> None:
        """Verify official election date fixtures."""
        self.assertEqual(ELECTIONS[2022].election_date, date(2022, 9, 11))
        self.assertEqual(ELECTIONS[2018].election_date, date(2018, 9, 9))
        self.assertEqual(ELECTIONS[2014].election_date, date(2014, 9, 14))
        self.assertEqual(ELECTIONS[2010].election_date, date(2010, 9, 19))
        self.assertEqual(ELECTIONS[2006].election_date, date(2006, 9, 17))
        self.assertEqual(ELECTIONS[2002].election_date, date(2002, 9, 15))

    def test_offline_parsing_and_exact_vote_sums_all_years(self) -> None:
        """Verify offline parsing produces exact vote sums equal to valid_votes_total."""
        for year in (2002, 2006, 2010, 2014, 2018, 2022):
            parsed = parse_election_by_year(year, raw_dir=RAW_ELECTIONS_DIR)
            self.assertEqual(parsed.election_year, year)
            self.assertEqual(parsed.election_date, ELECTIONS[year].election_date)

            # Sum of source party votes must equal valid_votes_total exactly
            src_votes_sum = sum(p.votes for p in parsed.source_parties)
            self.assertEqual(
                src_votes_sum,
                parsed.valid_votes_total,
                f"Year {year}: source party votes sum {src_votes_sum} != {parsed.valid_votes_total}",
            )

    def test_known_2022_fixture(self) -> None:
        """Verify known official 2022 election results."""
        parsed = parse_election_by_year(2022, raw_dir=RAW_ELECTIONS_DIR)
        self.assertEqual(parsed.valid_votes_total, 6477970)

        party_votes = {p.canonical_party: 0 for p in parsed.source_parties}
        for p in parsed.source_parties:
            party_votes[p.canonical_party] += p.votes

        self.assertEqual(party_votes["S"], 1964474)
        self.assertEqual(party_votes["SD"], 1330325)
        self.assertEqual(party_votes["M"], 1237428)
        self.assertEqual(party_votes["V"], 437050)
        self.assertEqual(party_votes["C"], 434945)
        self.assertEqual(party_votes["KD"], 345712)
        self.assertEqual(party_votes["MP"], 329242)
        self.assertEqual(party_votes["L"], 298542)
        self.assertEqual(party_votes["FI"], 3157)
        self.assertEqual(party_votes["OTHER"], 97095)

    def test_known_2018_fixture(self) -> None:
        """Verify known official 2018 election results."""
        parsed = parse_election_by_year(2018, raw_dir=RAW_ELECTIONS_DIR)
        self.assertEqual(parsed.valid_votes_total, 6476725)

        party_votes = {p.canonical_party: 0 for p in parsed.source_parties}
        for p in parsed.source_parties:
            party_votes[p.canonical_party] += p.votes

        self.assertEqual(party_votes["S"], 1830386)
        self.assertEqual(party_votes["M"], 1284698)
        self.assertEqual(party_votes["SD"], 1135627)
        self.assertEqual(party_votes["C"], 557500)
        self.assertEqual(party_votes["V"], 518454)
        self.assertEqual(party_votes["KD"], 409478)
        self.assertEqual(party_votes["L"], 355546)
        self.assertEqual(party_votes["MP"], 285899)
        self.assertEqual(party_votes["FI"], 29665)
        self.assertEqual(party_votes["OTHER"], 69472)

    def test_duplicate_detection_in_validator(self) -> None:
        """Validator must detect duplicate (election_year, party) entries."""
        canonical_csv = PROCESSED_ELECTIONS_DIR / "riksdag_election_results.csv"
        df = pd.read_csv(canonical_csv)

        # Inject a duplicate row
        dup_df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ElectionValidationError):
            validate_election_results(dup_df)

    def test_vote_sum_mismatch_detected(self) -> None:
        """Validator must reject election where votes sum does not match valid_votes_total."""
        canonical_csv = PROCESSED_ELECTIONS_DIR / "riksdag_election_results.csv"
        df = pd.read_csv(canonical_csv).copy()

        # Modify one vote count
        df.loc[0, "votes"] = df.loc[0, "votes"] + 100
        with self.assertRaises(ElectionValidationError):
            validate_election_results(df)

    def test_full_processed_dataset_passes_validation(self) -> None:
        """Full processed dataset must pass validation with 0 errors."""
        report = validate_processed_files(processed_dir=PROCESSED_ELECTIONS_DIR)
        self.assertEqual(report["status"], "PASSED")
        self.assertEqual(len(report["errors"]), 0)
        self.assertEqual(len(report["elections_checked"]), 6)


if __name__ == "__main__":
    unittest.main()
