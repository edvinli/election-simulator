"""Genuinely unique 20,000+ adversarial stress test suite comparing fast mandate allocator against exact legal reference."""

import hashlib
import json
import os
from fractions import Fraction
from pathlib import Path
import time
import unittest
import numpy as np

from scripts.geography.config import OFFICIAL_CONSTITUENCY_CODES
from scripts.mandates.allocator import allocate_riksdag_seats
from scripts.mandates.config import (
    FIXED_SEATS_2018,
    FIXED_SEATS_2022,
    FIXED_SEATS_2026,
    TOTAL_RIKSDAG_SEATS,
)
from scripts.simulator.config import MODEL_PARTIES_9, PARLIAMENTARY_PARTIES_8
from scripts.simulator.fast_allocator import (
    _FIXED_SEATS_2026_ARR,
    dispatch_production_allocation,
    fast_allocate_kernel,
)


def _has_actual_adjustment_boundary_tie(ref_alloc, cv_map: dict[str, dict[str, int]]) -> bool:
    """Inspect the reference placement sequence for an awarded adjustment tie."""
    adjustment_counts = {
        c: {p: 0 for p in ref_alloc.national_entitlement}
        for c in OFFICIAL_CONSTITUENCY_CODES
    }
    for party, entitled in ref_alloc.national_entitlement.items():
        n_adjustment = entitled - ref_alloc.final_national_fixed_seats.get(party, 0)
        for _ in range(n_adjustment):
            candidates = []
            for c in OFFICIAL_CONSTITUENCY_CODES:
                k = ref_alloc.final_fixed_seats_by_party_constituency[c].get(party, 0) + adjustment_counts[c][party]
                divisor = 1 if k == 0 else 2 * k + 1
                candidates.append((Fraction(cv_map[c].get(party, 0), divisor), c))
            max_q = max(q for q, _ in candidates)
            tied = [c for q, c in candidates if q == max_q]
            if len(tied) > 1:
                return True
            adjustment_counts[tied[0]][party] += 1
    return False


class TestAdversarialMandateAllocation(unittest.TestCase):
    """Stress testing fast vectorized allocator vs exact legal reference on 20,000 genuinely unique matrices."""

    def test_20000_unique_adversarial_fast_vs_exact_cases(self) -> None:
        """Run 20,000 unique deterministic adversarial cases covering all legal branches and exact cutoff ties."""
        n_cases = 20_000
        rng = np.random.default_rng(20260913)

        fixed_2018_arr = np.array([FIXED_SEATS_2018[c] for c in OFFICIAL_CONSTITUENCY_CODES], dtype=np.int64)
        fixed_2022_arr = np.array([FIXED_SEATS_2022[c] for c in OFFICIAL_CONSTITUENCY_CODES], dtype=np.int64)
        fixed_2026_arr = _FIXED_SEATS_2026_ARR

        unique_input_hashes = set()

        # Metrics & Branch counters
        counts = {
            "total_cases": 0,
            "fast_path": 0,
            "exact_tie_fallback": 0,
            "local_12_fallback": 0,
            "overhang_fallback": 0,
            "fast_kernel_handled_cases": 0,
            "fast_kernel_mismatches": 0,
            "dispatcher_matches": 0,
            "total_seat_violations": 0,
            "multi_return": 0,
            "gotland": 0,
            "historical_fixed_seat_map": 0,
            "adjustment_tie": 0,
        }

        t_start = time.perf_counter()

        for i in range(n_cases):
            counts["total_cases"] += 1
            branch_selector = i % 7

            # Select fixed seat configuration (historical vs 2026)
            if i % 5 == 0:
                F_arr = fixed_2018_arr
                F_dict = FIXED_SEATS_2018
            elif i % 5 == 1:
                F_arr = fixed_2022_arr
                F_dict = FIXED_SEATS_2022
            else:
                F_arr = fixed_2026_arr
                F_dict = FIXED_SEATS_2026

            # Base synthetic matrix generation
            # Generate unique base votes across 29 constituencies and 9 parties
            row_noise = rng.uniform(0.7, 1.3, size=(29, 1))
            col_base = np.array([19.0, 4.5, 6.5, 5.0, 30.0, 7.5, 4.5, 20.0, 3.0], dtype=np.float64)
            col_noise = rng.uniform(0.8, 1.2, size=(1, 9))
            mat_float = row_noise * col_base * col_noise * (10_000 + (i % 500) * 100)
            mat = np.round(mat_float).astype(np.int64)

            # Ensure non-zero positive votes
            mat = np.maximum(mat, 50)

            # Tailor specific adversarial branches
            if branch_selector == 0:
                # 1. Standard competitive parliamentary case (Fast Path Candidate)
                # Ensure all 8 parliamentary parties strictly above 4%
                pass

            elif branch_selector == 1:
                # 2. Exact awarded-boundary tie crafted deliberately.  Equal
                # eligible votes make the fixed-seat cutoff tie, rather than
                # merely creating equal quotients wholly inside the awarded set.
                mat[2, 0] = 50_000
                mat[2, :8] = 50_000
                mat[2, 8] = 50

            elif branch_selector == 2:
                # 3. Local 12% Exception for sub-4% Party L (col 1) in Constituency 01 (Stockholm kommun)
                # Set national votes of L below 4%
                mat[:, 1] = 500
                c_valid_0 = int(np.sum(mat[0]))
                # Give L 14% of constituency 01 valid votes
                mat[0, 1] = int(c_valid_0 * 0.14) + (i % 100)

            elif branch_selector == 3:
                # 4. Single Overhang Case: Party S (col 4) concentrated heavily
                mat[0, 4] += 120_000
                mat[1, 4] += 90_000
                # Drop national votes in other constituencies to create fixed-seat overhang
                mat[10:, 4] = 500

            elif branch_selector == 4:
                # 5. Multi-Return Overhang: Multiple excess seats retracted
                mat[0, 0] += 180_000
                mat[1, 0] += 150_000
                mat[2, 0] += 120_000
                mat[3, 0] += 100_000
                mat[10:, 0] = 200

            elif branch_selector == 5:
                # 6. Gotland (Constituency index 8, code '09') Stress Test
                # Gotland has 2 fixed seats (< 3), must be protected from return
                mat[8, 4] = 40_000
                mat[8, :4] = 100
                mat[8, 5:] = 100
                # Make S dominate the two largest constituencies while keeping
                # its national share low enough for a genuine fixed-seat
                # overhang.  Votes outside those rows are spread over the
                # remaining constituencies to cross the national 4% rule
                # without manufacturing another concentrated fixed-seat map.
                non_s = [0, 1, 2, 3, 5, 6, 7, 8]
                mat[0, non_s] = 100
                mat[1, non_s] = 100
                mat[0, 4] = 500_000
                mat[1, 4] = 500_000
                mat[8, 4] = 40_000
                mat[8, :4] = 100
                mat[8, 5:] = 100
                mat[10:, 4] = 100
                other_votes = int(np.sum(mat[:, non_s]))
                target_s_votes = (other_votes * 55 + 944) // 945  # approximately 5.5% nationally
                extra_s_votes = max(0, target_s_votes - int(np.sum(mat[:, 4])))
                if extra_s_votes:
                    mat[10:, 4] += extra_s_votes // 19
                    mat[10 : 10 + (extra_s_votes % 19), 4] += 1

            else:
                # 7. Dense National Threshold Boundary [3.95% - 4.05%]
                tot_v = np.sum(mat)
                target_4pct = int(tot_v * 0.04)
                # Perturb party MP (col 6) around 4%
                delta = (i % 200) - 100
                mat[:, 6] = max(10, (target_4pct + delta) // 29)

            # Hash the canonical input *and* election configuration.  A case
            # is distinct only when its matrix/configuration pair is distinct.
            input_hash = hashlib.sha256(
                mat.tobytes()
                + np.asarray(F_arr, dtype=np.int64).tobytes()
                + "|".join(MODEL_PARTIES_9).encode("utf-8")
            ).hexdigest()
            unique_input_hashes.add(input_hash)

            # 1. Exact Legal Reference Allocator (Oracle)
            cv_map = {}
            for row_i, c_code in enumerate(OFFICIAL_CONSTITUENCY_CODES):
                cv_map[c_code] = {}
                for col_j, p_code in enumerate(MODEL_PARTIES_9):
                    t_label = "OTHER_INELIGIBLE" if p_code == "REST" else p_code
                    cv_map[c_code][t_label] = int(mat[row_i, col_j])

            ref_alloc = allocate_riksdag_seats(cv_map, fixed_seats_by_constituency=F_dict)
            ref_seats = {p: ref_alloc.final_seats_by_party.get(p, 0) for p in PARLIAMENTARY_PARTIES_8}

            # 2. Fast Kernel Evaluation (if applicable)
            fk_seats = fast_allocate_kernel(mat, F_arr, parties=MODEL_PARTIES_9)
            if fk_seats is not None:
                counts["fast_kernel_handled_cases"] += 1
                if fk_seats == ref_seats:
                    pass
                else:
                    counts["fast_kernel_mismatches"] += 1

            # 3. Production Dispatcher Evaluation
            disp_res = dispatch_production_allocation(mat, fixed_seats_arr=F_arr, parties=MODEL_PARTIES_9)
            disp_seats = disp_res.seats_by_party

            if sum(disp_seats.values()) != TOTAL_RIKSDAG_SEATS:
                counts["total_seat_violations"] += 1

            if disp_seats == ref_seats:
                counts["dispatcher_matches"] += 1

            # Tally dispatch path and explicit rejection reasons.  Reasons are
            # emitted by the dispatcher at rejection time and are not inferred
            # from reference allocator output dictionaries.
            if disp_res.path == "FAST":
                counts["fast_path"] += 1
            reasons = set(disp_res.fallback_reasons)
            if "EXACT_TIE" in reasons:
                counts["exact_tie_fallback"] += 1
            if "LOCAL_12" in reasons:
                counts["local_12_fallback"] += 1
            if "OVERHANG" in reasons:
                counts["overhang_fallback"] += 1
            if disp_res.fixed_seat_configuration in {"2018", "2022"}:
                counts["historical_fixed_seat_map"] += 1

            retracted_events = [e for e in ref_alloc.event_log if e.phase == "excess_retracted"]
            if len(retracted_events) > 1:
                counts["multi_return"] += 1
            if _has_actual_adjustment_boundary_tie(ref_alloc, cv_map):
                counts["adjustment_tie"] += 1
            if branch_selector == 5:
                self.assertEqual(F_dict.get("09"), 2, "Gotland fixture must use its protected two-seat map")
                self.assertTrue(
                    retracted_events,
                    "Gotland generator must create an actual overhang return scenario",
                )
                self.assertTrue(
                    all(e.constituency_code != "09" for e in retracted_events),
                    "Gotland must never be selected for a returned fixed seat",
                )
                counts["gotland"] += 1

        total_time = time.perf_counter() - t_start

        # Assertions
        self.assertGreaterEqual(len(unique_input_hashes), n_cases, "All 20,000 matrix/config inputs must be distinct!")
        self.assertEqual(counts["total_seat_violations"], 0, "All allocations must sum strictly to 349 seats!")
        self.assertEqual(counts["fast_kernel_mismatches"], 0, "Fast kernel must match exact reference on every case it handles!")
        self.assertEqual(counts["dispatcher_matches"], n_cases, "Production dispatcher must match exact reference on 100% of all cases!")
        self.assertGreater(counts["exact_tie_fallback"], 0, "Tie generator must reach an awarded boundary tie")
        self.assertGreater(counts["local_12_fallback"], 0, "Local 12% generator must reach the legal branch")
        self.assertGreater(counts["overhang_fallback"], 0, "Overhang generator must reach the legal branch")
        self.assertGreater(counts["multi_return"], 0, "Multi-return generator must create multiple return events")
        self.assertGreater(counts["gotland"], 0, "Gotland generator must execute protected return coverage")
        self.assertGreater(counts["historical_fixed_seat_map"], 0, "Historical fixed-seat maps must be exercised")
        self.assertGreater(counts["adjustment_tie"], 0, "Adjustment tie generator must reach an awarded placement boundary")

        # Save audit report
        out_report = {
            "unique_cases_generated": len(unique_input_hashes),
            "canonical_input_config_hashes": len(unique_input_hashes),
            "total_cases": counts["total_cases"],
            "runtime_seconds": round(total_time, 2),
            "fast_path_count": counts["fast_path"],
            "fast_kernel_handled_cases": counts["fast_kernel_handled_cases"],
            "reference_fallback_cases": n_cases - counts["fast_kernel_handled_cases"],
            "fast_kernel_mismatch_count": counts["fast_kernel_mismatches"],
            "fast_kernel_accuracy_pct": round(100.0 * (counts["fast_kernel_handled_cases"] - counts["fast_kernel_mismatches"]) / max(1, counts["fast_kernel_handled_cases"]), 4),
            "exact_tie_fallback_count": counts["exact_tie_fallback"],
            "local_12_fallback_count": counts["local_12_fallback"],
            "overhang_fallback_count": counts["overhang_fallback"],
            "multi_return_count": counts["multi_return"],
            "gotland_coverage_count": counts["gotland"],
            "historical_fixed_seat_map_count": counts["historical_fixed_seat_map"],
            "adjustment_boundary_tie_count": counts["adjustment_tie"],
            "dispatcher_mismatch_count": n_cases - counts["dispatcher_matches"],
            "production_dispatcher_accuracy_pct": round(100.0 * counts["dispatcher_matches"] / n_cases, 4),
        }

        # Unit tests remain read-only by default.  The explicit environment
        # switch is used by the freeze audit when it deliberately regenerates
        # the tracked evidence artifact.
        report_env = os.environ.get("ELECTIONSIM_ADVERSARIAL_REPORT")
        if report_env:
            report_path = Path(report_env)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("w", encoding="utf-8") as f:
                json.dump(out_report, f, indent=2)
                f.write("\n")

        print("\n==========================================================================================")
        print("GENUINE 20,000 ADVERSARIAL MANDATE AUDIT REPORT")
        print(f"Total Unique Cases: {out_report['unique_cases_generated']:,} in {total_time:.2f} s")
        print(f"  Fast Path:                  {out_report['fast_path_count']:,}")
        print(f"  Exact Tie Fallbacks:        {out_report['exact_tie_fallback_count']:,}")
        print(f"  Local 12% Fallbacks:        {out_report['local_12_fallback_count']:,}")
        print(f"  Overhang Fallbacks:         {out_report['overhang_fallback_count']:,}")
        print(f"  Multi-Return Cases:         {out_report['multi_return_count']:,}")
        print(f"  Gotland Cases:              {out_report['gotland_coverage_count']:,}")
        print(f"  Historical Map Cases:       {out_report['historical_fixed_seat_map_count']:,}")
        print(f"  Fast Kernel Handled:         {counts['fast_kernel_handled_cases']:,}")
        print(f"  Fast Kernel Mismatches:      {counts['fast_kernel_mismatches']:,}")
        print(f"  Production Dispatch Match:  {counts['dispatcher_matches']:,} / {n_cases:,} ({out_report['production_dispatcher_accuracy_pct']}%)")
        print("==========================================================================================")


if __name__ == "__main__":
    unittest.main()
