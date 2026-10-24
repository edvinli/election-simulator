"""Constituency display boundaries: municipality assignment, dissolve and the committed output."""

import json
import unittest

from scripts.geography.boundaries import (
    DEFAULT_OUTPUT,
    DEFAULT_SOURCE,
    SPLIT_CONSTITUENCY_MUNICIPALITIES,
    assign_municipalities,
    build_constituency_boundaries,
    dissolve_rings,
)
from scripts.geography.config import CODE_TO_CONSTITUENCY_NAME, OFFICIAL_CONSTITUENCY_CODES


class ConstituencyBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.features = json.loads(DEFAULT_SOURCE.read_text(encoding="utf-8"))["features"]

    def test_every_municipality_has_exactly_one_constituency(self) -> None:
        assignment = assign_municipalities(self.features)
        self.assertEqual(len(assignment), 290)
        self.assertEqual(set(assignment.values()), set(OFFICIAL_CONSTITUENCY_CODES))
        self.assertEqual(assignment["0180"], "01")
        self.assertEqual(assignment["0138"], "02")
        self.assertEqual(assignment["1280"], "11")
        self.assertEqual(assignment["1480"], "16")
        self.assertEqual(assignment["0980"], "09")
        for constituency, names in SPLIT_CONSTITUENCY_MUNICIPALITIES.items():
            self.assertEqual(
                sum(1 for value in assignment.values() if value == constituency), len(names), constituency,
            )
        # Skåne and Västra Götaland are fully partitioned.
        self.assertEqual(sum(1 for code in assignment if code.startswith("12")), 33)
        self.assertEqual(sum(1 for code in assignment if code.startswith("14")), 49)

    def test_unlisted_split_county_municipality_fails_closed(self) -> None:
        features = [feature for feature in self.features if feature["properties"]["kom_namn"] != "Lund"]
        with self.assertRaisesRegex(ValueError, "Lund"):
            assign_municipalities(features)

    def test_dissolve_cancels_shared_borders(self) -> None:
        left = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        right = [(1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
        (outline,) = dissolve_rings([left, right])
        self.assertEqual(len(outline) - 1, 6)
        self.assertNotIn(((1.0, 0.0), (1.0, 1.0)), set(zip(outline, outline[1:])))
        # Two islands touching at one vertex stay two outline rings.
        corner = [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0), (1.0, 1.0)]
        self.assertEqual(len(dissolve_rings([left, corner])), 2)

    def test_committed_output_is_reproducible(self) -> None:
        built = build_constituency_boundaries()
        committed = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
        self.assertEqual(built, committed)
        self.assertEqual(
            [(row["code"], row["name"]) for row in committed["constituencies"]],
            [(code, CODE_TO_CONSTITUENCY_NAME[code]) for code in OFFICIAL_CONSTITUENCY_CODES],
        )
        for row in committed["constituencies"]:
            self.assertRegex(row["path"], r"^(M[-0-9.,L]+Z)+$")


if __name__ == "__main__":
    unittest.main()
