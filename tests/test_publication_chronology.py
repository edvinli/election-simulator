"""Table-driven tests for the publication chronology invariant.

The rule this module covers used to have no name. Its violations arrived as
"future_projection points must never be mixed into historical series", which
reads as data corruption -- and an investigation into a failing consecutive
publication duly went looking for a corrupt projection row that did not
exist. The cause was a certified date behind published history. Naming the
rule and testing it directly is what keeps that misreading from recurring.
"""

from __future__ import annotations

from datetime import date
from typing import Any
import unittest

from scripts.forecast_history.contract import validate_publication_chronology

# Imported deliberately: the chronology rule is only correct when it is applied
# to a payload whose leaked projection rows have already been discarded, and
# that ordering is a property worth holding under test rather than a comment.
from scripts.forecast_history.future_projection import _discard_persisted_projection_rows


ELECTION = "2026-09-13"
LATEST_PUBLISHED = "2026-09-06"


def _history(*dates: str, election: str | None = ELECTION) -> dict[str, Any]:
    """The fields the rule reads, and nothing else."""

    payload: dict[str, Any] = {
        "series": [{"date": day, "provenance": "reconstructed_current_model"} for day in dates]
    }
    if election is not None:
        payload["election_date"] = election
    return payload


class PublicationChronologyTests(unittest.TestCase):
    # (label, certified date, expected error fragment or None for "permitted")
    CASES = (
        ("same-day rerun replaces the certified point", LATEST_PUBLISHED, None),
        ("next-day publication extends history", "2026-09-07", None),
        ("a gap of several days is still forward", "2026-09-09", None),
        ("election day is a valid publication date", ELECTION, None),
        ("one day behind history is refused", "2026-09-05", "is behind published history through 2026-09-06"),
        ("far behind history is refused", "2026-08-01", "is behind published history through 2026-09-06"),
        ("after election day is refused", "2026-09-14", "occurs after election day 2026-09-13"),
    )

    def test_chronology_table(self) -> None:
        history = _history("2026-08-30", "2026-09-01", LATEST_PUBLISHED)
        for label, certified, expected in self.CASES:
            with self.subTest(label):
                if expected is None:
                    self.assertIsNone(validate_publication_chronology(history, certified))
                else:
                    with self.assertRaisesRegex(ValueError, expected):
                        validate_publication_chronology(history, certified)

    def test_a_date_object_and_an_iso_string_are_equivalent(self) -> None:
        history = _history(LATEST_PUBLISHED)
        self.assertIsNone(validate_publication_chronology(history, date(2026, 9, 7)))
        with self.assertRaisesRegex(ValueError, "is behind published history"):
            validate_publication_chronology(history, date(2026, 9, 5))

    def test_history_without_series_or_election_date_constrains_nothing(self) -> None:
        self.assertIsNone(validate_publication_chronology(_history(), "2026-09-07"))
        self.assertIsNone(
            validate_publication_chronology(_history(LATEST_PUBLISHED, election=None), ELECTION)
        )

    def test_repaired_projection_rows_must_be_discarded_before_the_rule_applies(self) -> None:
        """Leaked future rows would refuse a publication that is properly ordered."""

        leaked = _history(LATEST_PUBLISHED)
        leaked["future_projection"] = {
            "origin_date": LATEST_PUBLISHED,
            "series": [{"date": "2026-09-07"}, {"date": "2026-09-08"}],
        }
        leaked["series"].extend(
            [
                {"date": "2026-09-07", "provenance": "reconstructed_current_model"},
                {"date": "2026-09-08", "provenance": "reconstructed_current_model"},
            ]
        )

        # Applied to the raw payload, the hypothetical rows read as published
        # history and a legitimate next-day publication is refused.
        with self.assertRaisesRegex(ValueError, "is behind published history through 2026-09-08"):
            validate_publication_chronology(leaked, "2026-09-07")

        # Applied in the order the rule documents, the same publication passes.
        repaired = _discard_persisted_projection_rows(leaked)
        self.assertEqual(
            [point["date"] for point in repaired["series"]], [LATEST_PUBLISHED]
        )
        self.assertIsNone(validate_publication_chronology(repaired, "2026-09-07"))


if __name__ == "__main__":
    unittest.main()
