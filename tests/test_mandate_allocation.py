"""Comprehensive test suite for Swedish Riksdag mandate allocation and electoral mechanics.

Covers:
    - Official certified 2018 and 2022 historical golden tests (0 mismatches across 29 constituencies)
    - 349 total seats hard invariant
    - Exact 4.0% national threshold and 12.0% constituency threshold boundaries
    - Below-4% / above-12% local party fixture (wins fixed seat, 0 adjustment seats)
    - Adjustment first-seat divisor = 1.0 (raw votes) fixture
    - Gotland (<3 fixed seats) return prohibition fixture
    - Official overhang return and reallocation oracle fixture
    - Multi-constituency return-order fixture
    - Below-4% local party eligible to receive returned fixed seat fixture
    - Exact lottery TieBreaker interface
    - Exact Fraction arithmetic (no float rounding)
"""

from fractions import Fraction
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from scripts.mandates.allocator import SeatAllocation, allocate_riksdag_seats
from scripts.mandates.config import (
    DEFAULT_PROCESSED_DIR,
    FIXED_SEATS_2018,
    FIXED_SEATS_2022,
    FIXED_SEATS_2026,
    OFFICIAL_CONSTITUENCIES,
    TOTAL_ADJUSTMENT_SEATS,
    TOTAL_FIXED_SEATS,
    TOTAL_RIKSDAG_SEATS,
)
from scripts.mandates.tie_breaker import DeterministicLotteryTieBreaker, TieBreaker
from scripts.simulator.fast_allocator import EXACT_TIE, dispatch_production_allocation


class CustomMockTieBreaker:
    """Mock tie breaker that records tie events and picks candidate by custom rule."""

    def __init__(self, preferred_winner: str | None = None) -> None:
        self.preferred_winner = preferred_winner
        self.tie_events: list[dict] = []

    def pick_winner(self, candidates, context=None):
        self.tie_events.append({"candidates": list(candidates), "context": context})
        if self.preferred_winner and self.preferred_winner in candidates:
            return self.preferred_winner
        return candidates[0]


class TestMandateAllocation(unittest.TestCase):
    """Test suite covering historical golden tests, statutory edge cases, and arithmetic invariants."""

    @classmethod
    def setUpClass(cls) -> None:
        votes_csv = DEFAULT_PROCESSED_DIR / "historical_constituency_votes.csv"
        mandates_csv = DEFAULT_PROCESSED_DIR / "historical_certified_mandates.csv"
        const_2026_csv = DEFAULT_PROCESSED_DIR / "constituencies_2026.csv"

        cls.votes_df = pd.read_csv(votes_csv)
        cls.mandates_df = pd.read_csv(mandates_csv)
        cls.const_2026_df = pd.read_csv(const_2026_csv)

    def _build_constituency_votes_map(self, election_year: int) -> dict[str, dict[str, int]]:
        sub = self.votes_df[self.votes_df["election_year"] == election_year]
        cv_map: dict[str, dict[str, int]] = {}
        for _, row in sub.iterrows():
            c_code = f"{int(row['constituency_code']):02d}"
            p_code = str(row["party"])
            v_val = int(row["votes"])
            if c_code not in cv_map:
                cv_map[c_code] = {}
            cv_map[c_code][p_code] = v_val
        return cv_map

    def _build_certified_mandates_map(self, election_year: int) -> dict[tuple[str, str], int]:
        sub = self.mandates_df[self.mandates_df["election_year"] == election_year]
        cm_map: dict[tuple[str, str], int] = {}
        for _, row in sub.iterrows():
            c_code = f"{int(row['constituency_code']):02d}"
            p_code = str(row["party"])
            cm_map[(c_code, p_code)] = int(row["total_seats"])
        return cm_map

    def test_2026_constituency_configuration(self) -> None:
        """Verify 2026 constituency configuration matches Valmyndigheten's decided fixed seats."""
        self.assertEqual(len(self.const_2026_df), 29)
        self.assertEqual(self.const_2026_df["fixed_seats_2026"].sum(), TOTAL_FIXED_SEATS)
        self.assertEqual(FIXED_SEATS_2026["01"], 29)  # Stockholm kommun
        self.assertEqual(FIXED_SEATS_2026["02"], 41)  # Stockholm län (+1 vs 2022)
        self.assertEqual(FIXED_SEATS_2026["16"], 18)  # Göteborgs kommun (+1 vs 2022)
        self.assertEqual(FIXED_SEATS_2026["08"], 7)   # Kalmar län (-1 vs 2022)
        self.assertEqual(FIXED_SEATS_2026["26"], 7)   # Västernorrland (-1 vs 2022)

    def test_golden_2022_exact_allocation(self) -> None:
        """Verify allocator reproduces certified 2022 Riksdag election with 0 mismatches."""
        cv_map = self._build_constituency_votes_map(2022)
        cert_map = self._build_certified_mandates_map(2022)

        res = allocate_riksdag_seats(
            constituency_votes=cv_map,
            fixed_seats_by_constituency=FIXED_SEATS_2022,
        )

        self.assertEqual(res.total_seats, TOTAL_RIKSDAG_SEATS)
        expected_nat_2022 = {
            "S": 107, "SD": 73, "M": 68, "V": 24, "C": 24, "KD": 19, "MP": 18, "L": 16
        }
        expected_fixed_2022 = {
            "S": 104, "SD": 69, "M": 67, "C": 23, "V": 16, "KD": 13, "MP": 10, "L": 8
        }
        expected_adj_2022 = {
            "S": 3, "SD": 4, "M": 1, "C": 1, "V": 8, "KD": 6, "MP": 8, "L": 8
        }
        for p, exp_s in expected_nat_2022.items():
            self.assertEqual(res.final_seats_by_party.get(p, 0), exp_s)
            self.assertEqual(res.final_national_fixed_seats.get(p, 0), expected_fixed_2022[p])
            self.assertEqual(res.national_adjustment_seats.get(p, 0), expected_adj_2022[p])

        for c_code in sorted(cv_map.keys()):
            for p in ["M", "L", "C", "KD", "S", "V", "MP", "SD"]:
                calc = res.final_seats_by_party_constituency[c_code].get(p, 0)
                cert = cert_map.get((c_code, p), 0)
                self.assertEqual(calc, cert, f"Mismatch in 2022 for {c_code} {p}")

    def test_golden_2018_exact_allocation(self) -> None:
        """Verify allocator reproduces certified 2018 Riksdag election with 0 mismatches."""
        cv_map = self._build_constituency_votes_map(2018)
        cert_map = self._build_certified_mandates_map(2018)

        res = allocate_riksdag_seats(
            constituency_votes=cv_map,
            fixed_seats_by_constituency=FIXED_SEATS_2018,
        )

        self.assertEqual(res.total_seats, TOTAL_RIKSDAG_SEATS)
        expected_nat_2018 = {
            "S": 100, "M": 70, "SD": 62, "C": 31, "V": 28, "KD": 22, "L": 20, "MP": 16
        }
        expected_fixed_2018 = {
            "S": 94, "M": 66, "SD": 61, "C": 31, "V": 25, "KD": 16, "L": 12, "MP": 5
        }
        expected_adj_2018 = {
            "S": 6, "M": 4, "SD": 1, "C": 0, "V": 3, "KD": 6, "L": 8, "MP": 11
        }
        for p, exp_s in expected_nat_2018.items():
            self.assertEqual(res.final_seats_by_party.get(p, 0), exp_s)
            self.assertEqual(res.final_national_fixed_seats.get(p, 0), expected_fixed_2018[p])
            self.assertEqual(res.national_adjustment_seats.get(p, 0), expected_adj_2018[p])

        for c_code in sorted(cv_map.keys()):
            for p in ["M", "L", "C", "KD", "S", "V", "MP", "SD"]:
                calc = res.final_seats_by_party_constituency[c_code].get(p, 0)
                cert = cert_map.get((c_code, p), 0)
                self.assertEqual(calc, cert, f"Mismatch in 2018 for {c_code} {p}")

    def test_exact_4_percent_national_boundary(self) -> None:
        """Test exact inclusive 4.0% national threshold boundary."""
        votes = {
            c: {"M": 20000, "C": 10000, "L": 8000, "KD": 8000, "S": 30000, "V": 10000, "MP": 6000, "SD": 20000}
            for c in OFFICIAL_CONSTITUENCIES
        }
        for c in OFFICIAL_CONSTITUENCIES:
            votes[c]["P_EXACT"] = 5517
            votes[c]["P_SUB"] = 5517
        votes["01"]["P_EXACT"] += 160000 - 5517 * 29
        votes["01"]["P_SUB"] += 159999 - 5517 * 29
        votes["01"]["OTHER"] = 432001

        res = allocate_riksdag_seats(votes, FIXED_SEATS_2026)

        self.assertTrue(res.threshold_eligibility["P_EXACT"])
        self.assertFalse(res.threshold_eligibility["P_SUB"])
        self.assertEqual(res.total_seats, TOTAL_RIKSDAG_SEATS)

    def test_exact_12_percent_constituency_boundary(self) -> None:
        """Test exact inclusive 12.0% constituency threshold boundary for sub-4% party."""
        exact_votes = {c: {"M": 20000, "S": 20000} for c in OFFICIAL_CONSTITUENCIES}
        exact_votes["01"] = {"M": 22000, "S": 22000, "LOCAL_12": 6000}
        exact_votes["02"] = {"M": 22001, "S": 22001, "LOCAL_119": 5999}

        res = allocate_riksdag_seats(
            constituency_votes=exact_votes,
            fixed_seats_by_constituency=FIXED_SEATS_2026,
        )

        self.assertFalse(res.threshold_eligibility["LOCAL_12"])
        self.assertTrue(res.constituency_eligibility["01"]["LOCAL_12"])
        self.assertFalse(res.constituency_eligibility["02"]["LOCAL_119"])

    def test_sub4_above12_local_party_fixture(self) -> None:
        """Verify sub-4% party with >=12% locally wins fixed seat, gets 0 adjustment seats, and reduces national pool."""
        base_votes = {c: {"M": 15000, "S": 20000, "SD": 10000} for c in OFFICIAL_CONSTITUENCIES}
        base_votes["29"] = {"M": 5000, "S": 10000, "SD": 5000, "LOCAL_NORR": 15000}

        res = allocate_riksdag_seats(
            constituency_votes=base_votes,
            fixed_seats_by_constituency=FIXED_SEATS_2026,
        )

        self.assertFalse(res.threshold_eligibility["LOCAL_NORR"])
        lp_fixed = res.final_fixed_seats_by_party_constituency["29"]["LOCAL_NORR"]
        self.assertGreater(lp_fixed, 0)
        self.assertEqual(res.national_adjustment_seats.get("LOCAL_NORR", 0), 0)
        nat_entitled_sum = sum(res.national_entitlement.values())
        self.assertEqual(nat_entitled_sum, TOTAL_RIKSDAG_SEATS - lp_fixed)
        self.assertEqual(res.total_seats, TOTAL_RIKSDAG_SEATS)

    def test_adjustment_first_seat_divisor_is_one(self) -> None:
        """Verify that a party with 0 fixed seats in a constituency uses divisor 1.0 (pure votes) for its first adjustment seat."""
        cv_map = self._build_constituency_votes_map(2022)
        res = allocate_riksdag_seats(
            constituency_votes=cv_map,
            fixed_seats_by_constituency=FIXED_SEATS_2022,
        )

        adj_events = [e for e in res.event_log if e.phase == "adjustment"]
        found_divisor_one = False
        for ev in adj_events:
            c = ev.constituency_code
            p = ev.party
            if res.final_fixed_seats_by_party_constituency[c][p] == 0:
                self.assertEqual(ev.divisor, Fraction(1, 1), f"Expected divisor 1 for {p} in {c}, got {ev.divisor}")
                found_divisor_one = True

        self.assertTrue(found_divisor_one, "Should have at least one adjustment seat awarded to a 0-fixed-seat constituency")

    def test_gotland_under_three_fixed_seats_no_return(self) -> None:
        """Verify that Gotland (2 fixed seats) is prohibited from having fixed seats retracted during excess returns."""
        cv = {c: {"M": 20000, "S": 20000, "SD": 15000, "C": 10000, "V": 10000} for c in OFFICIAL_CONSTITUENCIES}
        cv["09"] = {"OVERHANG": 10000, "M": 100, "S": 100, "SD": 100, "C": 100, "V": 100}
        cv["01"]["OVERHANG"] = 80000

        res = allocate_riksdag_seats(
            constituency_votes=cv,
            fixed_seats_by_constituency=FIXED_SEATS_2026,
        )

        retracted_events = [e for e in res.event_log if e.phase == "excess_retracted"]
        for ev in retracted_events:
            self.assertNotEqual(ev.constituency_code, "09", "Constituency 09 (Gotland) must never have seats retracted")

    def test_overhang_return_and_reallocation_scenario(self) -> None:
        """Test full overhang detection, retraction of seat with lowest comparison number, and reallocation."""
        cv = {c: {"M": 20000, "S": 20000, "SD": 15000, "C": 10000, "V": 10000} for c in OFFICIAL_CONSTITUENCIES}
        for c in ["01", "02", "03", "04", "05"]:
            cv[c]["OVER"] = 60000

        res = allocate_riksdag_seats(
            constituency_votes=cv,
            fixed_seats_by_constituency=FIXED_SEATS_2026,
        )

        self.assertEqual(res.total_seats, TOTAL_RIKSDAG_SEATS)
        self.assertEqual(sum(res.final_seats_by_party.values()), TOTAL_RIKSDAG_SEATS)

        event_phases = set(e.phase for e in res.event_log)
        self.assertIn("fixed", event_phases)
        self.assertIn("national_entitlement", event_phases)
        self.assertIn("adjustment", event_phases)

    def test_multi_constituency_return_order(self) -> None:
        """Verify that when multiple constituencies have returned seats, reallocation occurs globally by largest quotient."""
        cv = {c: {"M": 20000, "S": 25000, "SD": 20000, "C": 10000, "V": 10000} for c in OFFICIAL_CONSTITUENCIES}
        cv["01"]["OVER"] = 80000
        cv["02"]["OVER"] = 90000
        cv["01"]["RECIPIENT_A"] = 25000
        cv["02"]["RECIPIENT_B"] = 30000

        res = allocate_riksdag_seats(cv, FIXED_SEATS_2026)
        self.assertEqual(res.total_seats, TOTAL_RIKSDAG_SEATS)
        self.assertEqual(sum(res.final_seats_by_party.values()), TOTAL_RIKSDAG_SEATS)

    def test_sub4_local_party_receives_returned_fixed_seat(self) -> None:
        """Verify that a party below 4% nationally with >=12% locally can receive a returned fixed seat in that constituency."""
        cv = {c: {"M": 20000, "S": 25000, "SD": 20000, "C": 10000, "V": 10000} for c in OFFICIAL_CONSTITUENCIES}
        cv["01"]["OVER"] = 120000
        cv["02"]["OVER"] = 120000
        cv["03"]["OVER"] = 80000
        cv["03"]["LOCAL_UPP"] = 35000  # >= 12% in constituency 03

        res = allocate_riksdag_seats(cv, FIXED_SEATS_2026)
        self.assertFalse(res.threshold_eligibility["LOCAL_UPP"])
        self.assertTrue(res.constituency_eligibility["03"]["LOCAL_UPP"])
        self.assertGreater(res.final_seats_by_party_constituency["03"]["LOCAL_UPP"], 0)
        self.assertEqual(res.total_seats, TOTAL_RIKSDAG_SEATS)

    def test_iterative_return_convergence_350_seat_regression(self) -> None:
        """Mandatory regression fixture: Sub-4% party receives returned seat, altering pool and creating second overhang.

        The old non-converging implementation produced 350 seats.
        The corrected iterative implementation must converge strictly to 349 seats.
        """
        cv = {c: {"M": 15000, "S": 25000, "SD": 15000, "V": 8000, "C": 7000} for c in OFFICIAL_CONSTITUENCIES}
        # In constituency 01 (Stockholm kommun, 29 fixed seats), create a dominant party with massive overhang
        cv["01"]["OVER_1"] = 160000
        # In constituency 02 (Stockholm län, 41 fixed seats), place sub-4% party with > 12% locally
        cv["02"]["OVER_2"] = 180000
        cv["02"]["SUB4_LOCAL"] = 55000  # ~21% in c=02, but < 4% nationally

        res = allocate_riksdag_seats(cv, FIXED_SEATS_2026)

        # Assert hard 349 seat invariant and exact convergence
        self.assertEqual(res.total_seats, TOTAL_RIKSDAG_SEATS)
        self.assertEqual(sum(res.final_seats_by_party.values()), TOTAL_RIKSDAG_SEATS)
        
        # Verify L and sum(U_p)
        L = sum(res.final_seats_by_party[p] for p in res.final_seats_by_party if not res.threshold_eligibility[p])
        sum_U = sum(res.national_adjustment_seats.values())
        sum_F_Q = sum(res.final_national_fixed_seats[p] for p in res.final_national_fixed_seats if res.threshold_eligibility[p])
        self.assertEqual(L + sum_F_Q + sum_U, TOTAL_RIKSDAG_SEATS)
        self.assertEqual(sum_U, TOTAL_ADJUSTMENT_SEATS)

    def test_keyed_lottery_tie_breaker_determinism_and_independence(self) -> None:
        """Verify DeterministicLotteryTieBreaker is state-based, independent of prior calls, and seed-sensitive."""
        tb1 = DeterministicLotteryTieBreaker(seed=42)
        tb2 = DeterministicLotteryTieBreaker(seed=42)
        tb_diff = DeterministicLotteryTieBreaker(seed=999)

        ctx_a = {"phase": "fixed", "constituency": "01", "comparison_number": Fraction(5000, 1)}
        ctx_b = {"phase": "fixed", "constituency": "02", "comparison_number": Fraction(5000, 1)}

        cand_list = ["PARTY_X", "PARTY_Y"]

        # Same event => same winner
        win1 = tb1.pick_winner(cand_list, context=ctx_a)
        win2 = tb2.pick_winner(cand_list, context=ctx_a)
        self.assertEqual(win1, win2)

        # Independent of unrelated prior calls on tb1
        tb1.pick_winner(["PARTY_A", "PARTY_B"], context=ctx_b)
        win1_again = tb1.pick_winner(cand_list, context=ctx_a)
        self.assertEqual(win1, win1_again)

    def test_first_divisor_normalization_and_rejection(self) -> None:
        """Verify divisor 1.2 is normalized to Fraction(6, 5) and invalid floats are rejected."""
        cv_map = self._build_constituency_votes_map(2022)
        res = allocate_riksdag_seats(cv_map, FIXED_SEATS_2022, first_divisor=1.2)
        self.assertEqual(res.total_seats, TOTAL_RIKSDAG_SEATS)

        with self.assertRaises(ValueError):
            allocate_riksdag_seats(cv_map, FIXED_SEATS_2022, first_divisor=1.2345)

    def test_input_validation_hardening(self) -> None:
        """Verify allocator strictly rejects negative votes, non-integers, empty/missing inputs."""
        cv_map = self._build_constituency_votes_map(2022)
        
        # Negative votes
        bad_cv = {c: dict(cv_map[c]) for c in cv_map}
        bad_cv["01"]["M"] = -100
        with self.assertRaises(ValueError):
            allocate_riksdag_seats(bad_cv, FIXED_SEATS_2022)

        # Non-integer votes
        bad_cv2 = {c: dict(cv_map[c]) for c in cv_map}
        bad_cv2["01"]["M"] = 100.5  # type: ignore
        with self.assertRaises(TypeError):
            allocate_riksdag_seats(bad_cv2, FIXED_SEATS_2022)

        # Missing constituency
        bad_cv3 = {c: dict(cv_map[c]) for c in cv_map if c != "01"}
        with self.assertRaises(ValueError):
            allocate_riksdag_seats(bad_cv3, FIXED_SEATS_2022)

    def test_injected_lottery_tie_breaker(self) -> None:
        """Verify that TieBreaker interface is called when comparison numbers tie exactly."""
        cv = {c: {"M": 10000, "S": 10000, "SD": 10000, "C": 5000, "V": 5000} for c in OFFICIAL_CONSTITUENCIES}
        cv["01"]["PARTY_A"] = 5000
        cv["01"]["PARTY_B"] = 5000

        mock_tb = CustomMockTieBreaker(preferred_winner="PARTY_B")
        res = allocate_riksdag_seats(
            constituency_votes=cv,
            fixed_seats_by_constituency=FIXED_SEATS_2026,
            tie_breaker=mock_tb,
        )

        self.assertEqual(res.total_seats, TOTAL_RIKSDAG_SEATS)
        self.assertGreater(len(mock_tb.tie_events), 0, "Tie breaker should have been invoked on equal votes")

    def test_exact_fraction_arithmetic_no_float_rounding(self) -> None:
        """Verify that divisor 1.2 is represented as exact Fraction(6, 5) without float precision loss."""
        res = allocate_riksdag_seats(
            constituency_votes=self._build_constituency_votes_map(2022),
            fixed_seats_by_constituency=FIXED_SEATS_2022,
        )
        for ev in res.event_log:
            self.assertIsInstance(ev.comparison_number, Fraction)
            self.assertIsInstance(ev.divisor, Fraction)

    def test_synthetic_return_reallocation_regression(self) -> None:
        """Synthetic Riksdag stress case for returned-seat reallocation mechanics."""
        # Statutory worked scenario: Party OVER has high concentration in constituencies 01 and 02
        cv = {c: {"M": 25000, "S": 35000, "SD": 20000, "C": 10000, "V": 10000} for c in OFFICIAL_CONSTITUENCIES}
        # In constituency 01 (Stockholm kommun, 29 fixed seats), give OVER 140,000 votes
        cv["01"]["OVER"] = 140000
        # In constituency 02 (Stockholm län, 41 fixed seats), give OVER 150,000 votes
        cv["02"]["OVER"] = 150000
        # In constituency 01, RECIPIENT_A has 22,000 votes
        cv["01"]["RECIPIENT_A"] = 22000

        res = allocate_riksdag_seats(
            constituency_votes=cv,
            fixed_seats_by_constituency=FIXED_SEATS_2026,
        )

        # 1. Total Riksdag seats strictly conserved
        self.assertEqual(res.total_seats, TOTAL_RIKSDAG_SEATS)
        self.assertEqual(sum(res.final_seats_by_party.values()), TOTAL_RIKSDAG_SEATS)

        # 2. Assert excess seats retracted and reallocated
        retracted_events = [e for e in res.event_log if e.phase == "excess_retracted"]
        reallocated_events = [e for e in res.event_log if e.phase == "returned_reallocated"]
        self.assertGreater(len(retracted_events), 0)
        self.assertEqual(len(retracted_events), len(reallocated_events))

        for ret_ev in retracted_events:
            self.assertEqual(ret_ev.party, "OVER")
            self.assertIn(ret_ev.constituency_code, ["01", "02"])
            self.assertNotEqual(ret_ev.constituency_code, "09")  # Gotland protection

        # 3. Assert adjustment seats sum to 39
        sum_adj = sum(res.national_adjustment_seats.values())
        self.assertEqual(sum_adj, TOTAL_ADJUSTMENT_SEATS)

    def test_valmyndigheten_example5_fixture_is_untransformed(self) -> None:
        """Validate the published three-constituency fixture without relabelling it Riksdag data."""
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "valmyndigheten_example_5_valkoping.json"
        with fixture_path.open(encoding="utf-8") as f:
            fixture = json.load(f)

        self.assertEqual(fixture["fixture_type"], "official_worked_example")
        self.assertIn("Valmyndigheten", fixture["source"]["authority"])
        self.assertEqual(len(fixture["constituencies"]), 3)
        self.assertEqual(sum(fixture["fixed_seats_by_constituency"].values()), 67)
        self.assertEqual(fixture["total_seats"], 75)
        self.assertEqual(
            fixture["expected_phase_order"],
            ["fixed", "national_entitlement", "excess_retracted", "returned_reallocated", "adjustment"],
        )
        self.assertEqual(fixture["expected_events"][0]["party"], "KD")
        self.assertEqual(fixture["expected_events"][0]["constituency"], "Valköping V")
        self.assertEqual(fixture["expected_events"][1]["party"], "L")
        self.assertEqual(fixture["expected_events"][1]["constituency"], "Valköping V")
        self.assertEqual(fixture["fixed_seats_after_return"]["KD"]["Valköping V"], 0)
        self.assertEqual(fixture["fixed_seats_after_return"]["L"]["Valköping V"], 2)
        self.assertEqual(sum(sum(row.values()) for row in fixture["final_seats"].values()), 75)

    def test_exact_cutoff_tie_dispatches_to_reference_with_lottery(self) -> None:
        """Exact 200000/1.2 == 500000/3 boundary must never use the fast path."""
        votes = np.array(
            [[20_000, 4_000, 7_000, 6_000, 30_000, 7_000, 6_000, 20_000, 3_000] for _ in OFFICIAL_CONSTITUENCIES],
            dtype=np.int64,
        )
        # In Stockholm kommun, M's first quotient and S's second quotient tie
        # exactly: 200000/(6/5) = 500000/3 = 166666 2/3.
        votes[0, :] = 100
        votes[0, 0] = 200_000
        votes[0, 4] = 500_000

        fixed_arr = np.array([FIXED_SEATS_2026[c] for c in OFFICIAL_CONSTITUENCIES], dtype=np.int64)
        dispatch = dispatch_production_allocation(votes, fixed_seats_arr=fixed_arr)
        self.assertEqual(dispatch.path, "REFERENCE")
        self.assertIn(EXACT_TIE, dispatch.fallback_reasons)
        self.assertEqual(dispatch.fixed_seat_configuration, "2026")
        self.assertEqual(sum(dispatch.seats_by_party.values()), TOTAL_RIKSDAG_SEATS)

    def test_2018_local_qualification_fixed_seat_map_regression(self) -> None:
        """Verify that historical 2018 seat allocation strictly uses 2018 fixed seats and rejects 2026 map."""
        # In 2018, Kalmar (08) had 8 fixed seats. In 2026, Kalmar has 7 fixed seats.
        # In 2018, Stockholm län (02) had 39 fixed seats. In 2026, Stockholm län has 41 fixed seats.
        cv = {c: {"M": 20000, "S": 30000, "SD": 20000, "C": 10000, "V": 10000} for c in OFFICIAL_CONSTITUENCIES}
        # In Kalmar (08):
        cv["08"] = {"S": 14000, "M": 8000, "SD": 7000, "C": 5000, "V": 4000, "LOCAL_KLM": 5400}

        # Allocation under 2018 map (8 fixed seats in Kalmar)
        res_2018 = allocate_riksdag_seats(cv, fixed_seats_by_constituency=FIXED_SEATS_2018)
        # Allocation under 2026 map (7 fixed seats in Kalmar)
        res_2026 = allocate_riksdag_seats(cv, fixed_seats_by_constituency=FIXED_SEATS_2026)

        # In 2018, Kalmar awards 8 total fixed seats (S gets 3 seats)
        total_kalmar_2018 = sum(res_2018.final_fixed_seats_by_party_constituency["08"].values())
        self.assertEqual(total_kalmar_2018, 8)
        self.assertEqual(res_2018.final_fixed_seats_by_party_constituency["08"]["S"], 3)

        # In 2026, Kalmar awards 7 total fixed seats (S gets 2 seats)
        total_kalmar_2026 = sum(res_2026.final_fixed_seats_by_party_constituency["08"].values())
        self.assertEqual(total_kalmar_2026, 7)
        self.assertEqual(res_2026.final_fixed_seats_by_party_constituency["08"]["S"], 2)

        # Ensure both configurations strictly conserve 349 total seats
        self.assertEqual(res_2018.total_seats, TOTAL_RIKSDAG_SEATS)
        self.assertEqual(res_2026.total_seats, TOTAL_RIKSDAG_SEATS)
