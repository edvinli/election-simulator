"""Ultra-fast, vectorized Riksdag mandate allocator with 100% legal equivalence and exact reference fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np

from scripts.geography.config import OFFICIAL_CONSTITUENCY_CODES
from scripts.mandates.allocator import SeatAllocation, allocate_riksdag_seats
from scripts.mandates.config import (
    FIXED_SEATS_2018,
    FIXED_SEATS_2022,
    FIXED_SEATS_2026,
    OFFICIAL_CONSTITUENCIES,
    TOTAL_RIKSDAG_SEATS,
)
from .config import MODEL_PARTIES_9, PARLIAMENTARY_PARTIES_8


# Precompute standard Sainte-Laguë divisors (first divisor 1.2 = 6/5, then 3, 5, 7...)
# Up to 350 seats
_DIVISORS_350 = np.empty(350, dtype=np.float64)
_DIVISORS_350[0] = 1.2
_DIVISORS_350[1:] = 2 * np.arange(1, 350) + 1.0

# Precompute fixed seats array for 2026 ordered by OFFICIAL_CONSTITUENCY_CODES
_FIXED_SEATS_2026_ARR = np.array([FIXED_SEATS_2026[c] for c in OFFICIAL_CONSTITUENCY_CODES], dtype=np.int64)

# Structured rejection labels are deliberately independent.  A single input can,
# for example, contain both a local 12% exception and an exact quotient tie.
EXACT_TIE = "EXACT_TIE"
LOCAL_12 = "LOCAL_12"
OVERHANG = "OVERHANG"
INSUFFICIENT_CANDIDATES = "INSUFFICIENT_CANDIDATES"


@dataclass(frozen=True)
class FastKernelEvaluation:
    """Result of the optimized kernel, including reasons for a safe fallback."""

    seats_by_party: dict[str, int] | None
    fallback_reasons: frozenset[str]


@dataclass(frozen=True)
class FastAllocationDispatchResult:
    """Result of fast allocation including dispatch diagnostics and local 12% tracking."""

    seats_by_party: dict[str, int]
    dispatch_path: str  # Compatibility label: "fast_path" or "reference_fallback"
    local_12pct_qualified: bool
    local_12pct_constituencies: list[str]
    received_seat_via_12pct: bool
    # ``dispatch_path`` is retained for compatibility with existing callers.
    # These fields are the authoritative structured diagnostics for audits.
    path: str  # "FAST" or "REFERENCE"
    fallback_reasons: tuple[str, ...]
    fixed_seat_configuration: str

    @property
    def diagnostics(self) -> dict[str, object]:
        """Return JSON-friendly dispatch diagnostics recorded at rejection time."""
        return {
            "path": self.path,
            "fallback_reasons": list(self.fallback_reasons),
            "fixed_seat_configuration": self.fixed_seat_configuration,
        }


def _fixed_seat_configuration(F_arr: np.ndarray) -> str:
    """Identify the election fixed-seat map without treating it as a fallback reason."""
    arr = np.asarray(F_arr, dtype=np.int64)
    known = {
        "2018": np.array([FIXED_SEATS_2018[c] for c in OFFICIAL_CONSTITUENCY_CODES], dtype=np.int64),
        "2022": np.array([FIXED_SEATS_2022[c] for c in OFFICIAL_CONSTITUENCY_CODES], dtype=np.int64),
        "2026": _FIXED_SEATS_2026_ARR,
    }
    for label, known_arr in known.items():
        if arr.shape == known_arr.shape and np.array_equal(arr, known_arr):
            return label
    return "custom"


def _modified_sainte_lague_numerators_denominators(
    votes: np.ndarray,
    seat_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build exact integer numerator/denominator arrays for modified Sainte-Laguë.

    The first comparison number is ``votes / (6/5) = 5*votes/6``.  All later
    comparison numbers are ``votes / (2*k+1)``.  Object dtype is used only for
    unusually large vote totals where an int64 cross-product could overflow;
    normal production totals remain vectorized int64 operations.
    """
    votes_arr = np.asarray(votes, dtype=np.int64)
    use_object = bool(np.any(np.abs(votes_arr) > np.iinfo(np.int64).max // 5))
    dtype = object if use_object else np.int64
    nums = np.repeat(votes_arr[:, np.newaxis], seat_count, axis=1).astype(dtype, copy=True)
    dens = np.empty((len(votes_arr), seat_count), dtype=dtype)
    if seat_count > 0:
        nums[:, 0] = nums[:, 0] * 5
        dens[:, 0] = 6
    if seat_count > 1:
        dens[:, 1:] = (2 * np.arange(1, seat_count, dtype=np.int64) + 1)[np.newaxis, :]
    return nums, dens


def _compare_exact_quotients(
    num_a: object,
    den_a: object,
    num_b: object,
    den_b: object,
) -> int:
    """Compare two nonnegative rational quotients through integer cross-products."""
    left = int(num_a) * int(den_b)
    right = int(num_b) * int(den_a)
    return (left > right) - (left < right)


def _exact_boundary_is_unambiguous(
    numerators: np.ndarray,
    denominators: np.ndarray,
    float_quotients: np.ndarray,
    selected_indices: np.ndarray,
) -> bool:
    """Verify a NumPy-selected boundary exactly.

    ``selected_indices`` comes from the floating-point/NumPy selection.  We then
    find the exact minimum awarded quotient and exact maximum unawarded quotient.
    The fast result is admissible only when max(unselected) < min(selected).
    Equality invokes the statutory lottery through the reference allocator.
    """
    selected = np.asarray(selected_indices, dtype=np.int64)
    if selected.size == 0:
        return False
    selected_mask = np.zeros(numerators.size, dtype=bool)
    selected_mask[selected] = True
    unselected = np.flatnonzero(~selected_mask)
    if unselected.size == 0:
        return True

    # A correctly rounded quotient can move by at most one ulp.  Restrict the
    # exact scan to the ulp-sized neighbourhood of each NumPy boundary; any
    # candidate outside it cannot reverse the boundary ordering.  This keeps
    # normal production runs vectorized while still making the final decision
    # with integer cross-products.
    selected_float = np.asarray(float_quotients[selected], dtype=np.float64)
    unselected_float = np.asarray(float_quotients[unselected], dtype=np.float64)
    min_float = float(np.min(selected_float))
    max_float = float(np.max(unselected_float))
    scale = max(1.0, abs(min_float), abs(max_float))
    tolerance = 8.0 * np.finfo(np.float64).eps * scale
    min_candidates = selected[selected_float <= min_float + tolerance]
    max_candidates = unselected[unselected_float >= max_float - tolerance]

    min_selected = int(min_candidates[0])
    for raw_idx in min_candidates[1:]:
        idx = int(raw_idx)
        if _compare_exact_quotients(
            numerators[idx], denominators[idx], numerators[min_selected], denominators[min_selected]
        ) < 0:
            min_selected = idx

    max_unselected = int(max_candidates[0])
    for raw_idx in max_candidates[1:]:
        idx = int(raw_idx)
        if _compare_exact_quotients(
            numerators[idx], denominators[idx], numerators[max_unselected], denominators[max_unselected]
        ) > 0:
            max_unselected = idx

    # Strict inequality is the only case in which the floating selection is
    # provably unambiguous.  Both equality and reversal require reference tie
    # handling (or exact re-ranking) and therefore fall back.
    return _compare_exact_quotients(
        numerators[max_unselected],
        denominators[max_unselected],
        numerators[min_selected],
        denominators[min_selected],
    ) < 0


def _verify_adjustment_boundaries(
    X: np.ndarray,
    qual_indices: np.ndarray,
    fixed_seats_won: np.ndarray,
    entitlement: np.ndarray,
) -> bool:
    """Check every adjustment-seat placement boundary with exact quotients."""
    adjustment_counts = np.zeros_like(fixed_seats_won)
    for p_idx in qual_indices:
        n_adjustment = int(entitlement[p_idx] - fixed_seats_won[:, p_idx].sum())
        if n_adjustment < 0:
            return False
        for _ in range(n_adjustment):
            seats_in_constituency = fixed_seats_won[:, p_idx] + adjustment_counts[:, p_idx]
            denominators = np.where(seats_in_constituency == 0, 1, 2 * seats_in_constituency + 1).astype(np.int64)
            votes = X[:, p_idx].astype(np.int64, copy=False)
            float_quotients = votes / denominators
            winner = int(np.argmax(float_quotients))
            # One selected candidate versus every other constituency.  Keeping
            # the exact helper's boundary contract makes reversal detection and
            # exact equality behave identically to fixed/national allocation.
            if not _exact_boundary_is_unambiguous(
                votes,
                denominators,
                float_quotients,
                np.array([winner], dtype=np.int64),
            ):
                return False
            adjustment_counts[winner, p_idx] += 1
    return True


def _evaluate_fast_kernel(
    X: np.ndarray,
    F_arr: np.ndarray,
    parties: Sequence[str] = MODEL_PARTIES_9,
) -> FastKernelEvaluation:
    """Run the fast kernel and retain the exact reason for any safe fallback.

    Returns:
        A ``FastKernelEvaluation``.  ``seats_by_party`` is populated only when
        the fast allocation is unambiguous and has no legal exceptional branch.
    """
    n_constituencies, n_parties = X.shape
    rest_col_idx = n_parties - 1  # Assume REST is last column

    # 1. National Vote Totals & Threshold Eligibility
    nat_votes = np.sum(X, axis=0)
    total_valid = np.sum(nat_votes)
    if total_valid <= 0:
        raise ValueError("Total valid votes must be strictly positive")

    # Check 4% national threshold for parliamentary parties (excluding REST)
    above_4_mask = np.zeros(n_parties, dtype=bool)
    above_4_mask[:rest_col_idx] = (25 * nat_votes[:rest_col_idx] >= total_valid)

    # 2. Check for local 12% exception on sub-4% parties
    sub_4_indices = np.where(~above_4_mask[:rest_col_idx])[0]
    const_valid = np.sum(X, axis=1)

    if len(sub_4_indices) > 0:
        for p_idx in sub_4_indices:
            if np.any(25 * X[:, p_idx] >= 3 * const_valid):
                return FastKernelEvaluation(None, frozenset({LOCAL_12}))

    # 3. Fixed Constituency Seats Allocation
    fixed_seats_won = np.zeros((n_constituencies, n_parties), dtype=np.int64)
    qual_indices = np.where(above_4_mask)[0]

    if len(qual_indices) == 0:
        return FastKernelEvaluation(None, frozenset({INSUFFICIENT_CANDIDATES}))

    for c in range(n_constituencies):
        f_c = int(F_arr[c])
        c_votes = X[c, qual_indices]  # shape (n_qual,)

        # Build comparison quotients for all seats up to f_c for qualifying parties
        divs = _DIVISORS_350[:f_c]  # shape (f_c,)
        quotients = c_votes[:, np.newaxis] / divs[np.newaxis, :]
        flat_q = quotients.ravel()

        if len(flat_q) < f_c:
            return FastKernelEvaluation(None, frozenset({INSUFFICIENT_CANDIDATES}))

        # Find top f_c quotients
        top_k_indices = np.argpartition(-flat_q, f_c - 1)[:f_c]
        exact_nums, exact_dens = _modified_sainte_lague_numerators_denominators(c_votes, f_c)
        if not _exact_boundary_is_unambiguous(
            exact_nums.ravel(),
            exact_dens.ravel(),
            flat_q,
            top_k_indices,
        ):
            return FastKernelEvaluation(None, frozenset({EXACT_TIE}))
        winning_party_local_idx = top_k_indices // f_c
        for w_l_idx in winning_party_local_idx:
            w_p_idx = qual_indices[w_l_idx]
            fixed_seats_won[c, w_p_idx] += 1

    nat_fixed_won = np.sum(fixed_seats_won, axis=0)

    # 4. National Proportional Entitlement (349 seats among qualifying parties)
    total_entitlement_seats = TOTAL_RIKSDAG_SEATS
    qual_nat_votes = nat_votes[qual_indices]

    divs_nat = _DIVISORS_350[:total_entitlement_seats]
    nat_quotients = qual_nat_votes[:, np.newaxis] / divs_nat[np.newaxis, :]
    flat_nat_q = nat_quotients.ravel()

    if len(flat_nat_q) < total_entitlement_seats:
        return FastKernelEvaluation(None, frozenset({INSUFFICIENT_CANDIDATES}))

    top_nat_indices = np.argpartition(-flat_nat_q, total_entitlement_seats - 1)[:total_entitlement_seats]
    exact_nat_nums, exact_nat_dens = _modified_sainte_lague_numerators_denominators(
        qual_nat_votes,
        total_entitlement_seats,
    )
    if not _exact_boundary_is_unambiguous(
        exact_nat_nums.ravel(),
        exact_nat_dens.ravel(),
        flat_nat_q,
        top_nat_indices,
    ):
        return FastKernelEvaluation(None, frozenset({EXACT_TIE}))
    winning_nat_party_local = top_nat_indices // total_entitlement_seats

    entitlement = np.zeros(n_parties, dtype=np.int64)
    for w_l in winning_nat_party_local:
        entitlement[qual_indices[w_l]] += 1

    # 5. Overhang Check
    has_overhang = np.any(nat_fixed_won[qual_indices] > entitlement[qual_indices])
    if has_overhang:
        return FastKernelEvaluation(None, frozenset({OVERHANG}))

    # Adjustment seats are constituency-level comparisons under Vallagen 14
    # kap. 5 §.  The fast path returns national party totals, but it may only
    # do so after proving every placement boundary is unambiguous; otherwise
    # reference allocation is required for the statutory lottery.
    if not _verify_adjustment_boundaries(X, qual_indices, fixed_seats_won, entitlement):
        return FastKernelEvaluation(None, frozenset({EXACT_TIE}))

    # Clean non-overhang case
    final_seats = {p: 0 for p in parties if p != "REST"}
    for p_idx in qual_indices:
        final_seats[parties[p_idx]] = int(entitlement[p_idx])

    return FastKernelEvaluation(final_seats, frozenset())


def fast_allocate_kernel(
    X: np.ndarray,
    F_arr: np.ndarray,
    parties: Sequence[str] = MODEL_PARTIES_9,
) -> dict[str, int] | None:
    """Pure fast allocation result, preserving the historical public API.

    Use ``dispatch_production_allocation`` when structured fallback diagnostics
    are needed by production or audit code.
    """
    return _evaluate_fast_kernel(X, F_arr, parties=parties).seats_by_party


def dispatch_production_allocation(
    votes_matrix_29x9: np.ndarray,
    fixed_seats_arr: np.ndarray | None = None,
    parties: Sequence[str] = MODEL_PARTIES_9,
) -> FastAllocationDispatchResult:
    """Production mandate dispatcher guaranteed to match exact legal reference allocator 100%.

    Dispatches cleanly:
    - Runs fast_allocate_kernel.
    - If fast_allocate_kernel returns a result, dispatches via fast path.
    - Otherwise, falls back to exact Fraction reference allocator passing election-specific fixed seats map.
    """
    X = np.asarray(votes_matrix_29x9, dtype=np.int64)
    F_arr = fixed_seats_arr if fixed_seats_arr is not None else _FIXED_SEATS_2026_ARR

    n_constituencies, n_parties = X.shape
    rest_col_idx = n_parties - 1

    # National totals
    nat_votes = np.sum(X, axis=0)
    total_valid = np.sum(nat_votes)
    if total_valid <= 0:
        raise ValueError("Total valid votes must be strictly positive")

    above_4_mask = np.zeros(n_parties, dtype=bool)
    above_4_mask[:rest_col_idx] = (25 * nat_votes[:rest_col_idx] >= total_valid)

    # Local 12% exception checks
    sub_4_indices = np.where(~above_4_mask[:rest_col_idx])[0]
    const_valid = np.sum(X, axis=1)

    local_12_qualified = False
    local_12_constituencies: list[str] = []

    if len(sub_4_indices) > 0:
        for p_idx in sub_4_indices:
            for c_idx in range(n_constituencies):
                if 25 * X[c_idx, p_idx] >= 3 * const_valid[c_idx]:
                    local_12_qualified = True
                    c_code = OFFICIAL_CONSTITUENCY_CODES[c_idx]
                    if c_code not in local_12_constituencies:
                        local_12_constituencies.append(c_code)

    # Attempt fast kernel.  The evaluation object records rejection reasons at
    # the point the kernel proves that reference allocation is required.
    kernel_eval = _evaluate_fast_kernel(X, F_arr, parties=parties)
    fixed_config = _fixed_seat_configuration(F_arr)
    if kernel_eval.seats_by_party is not None:
        return FastAllocationDispatchResult(
            seats_by_party=kernel_eval.seats_by_party,
            dispatch_path="fast_path",
            local_12pct_qualified=False,
            local_12pct_constituencies=[],
            received_seat_via_12pct=False,
            path="FAST",
            fallback_reasons=(),
            fixed_seat_configuration=fixed_config,
        )

    # Do not infer reasons from the reference allocator's returned-seat map.
    # The kernel's rejection set is authoritative; LOCAL_12 is also recorded
    # here because this dispatcher computes the full set of qualifying cells.
    fallback_reasons = set(kernel_eval.fallback_reasons)
    if local_12_qualified:
        fallback_reasons.add(LOCAL_12)
    if not fallback_reasons:
        fallback_reasons.add(INSUFFICIENT_CANDIDATES)

    # Fall back to exact reference allocator with election-specific fixed seats
    ref_res, ref_alloc = _fallback_to_reference_allocator_with_details(X, parties, F_arr)

    # Determine if any sub-4% party actually received a seat via 12%
    received_seat_via_12 = False
    if local_12_qualified:
        for p_idx in sub_4_indices:
            p_code = parties[p_idx]
            if ref_res.get(p_code, 0) > 0:
                received_seat_via_12 = True
                break

    return FastAllocationDispatchResult(
        seats_by_party=ref_res,
        dispatch_path="reference_fallback",
        local_12pct_qualified=local_12_qualified,
        local_12pct_constituencies=local_12_constituencies,
        received_seat_via_12pct=received_seat_via_12,
        path="REFERENCE",
        fallback_reasons=tuple(sorted(fallback_reasons)),
        fixed_seat_configuration=fixed_config,
    )


def fast_allocate_seats_from_matrix(
    votes_matrix_29x9: np.ndarray,
    fixed_seats_arr: np.ndarray | None = None,
    parties: Sequence[str] = MODEL_PARTIES_9,
) -> dict[str, int]:
    """Convenience wrapper for fast allocation dispatch."""
    disp_res = dispatch_production_allocation(votes_matrix_29x9, fixed_seats_arr=fixed_seats_arr, parties=parties)
    return disp_res.seats_by_party


def _fallback_to_reference_allocator_with_details(
    X: np.ndarray,
    parties: Sequence[str],
    F_arr: np.ndarray,
) -> tuple[dict[str, int], SeatAllocation]:
    """Fallback handler that constructs constituency vote dict and exact election-specific fixed seats map."""
    cv_map: dict[str, dict[str, int]] = {}
    fixed_seats_dict: dict[str, int] = {}
    for i, c_code in enumerate(OFFICIAL_CONSTITUENCY_CODES):
        fixed_seats_dict[c_code] = int(F_arr[i])
        cv_map[c_code] = {}
        for j, p_code in enumerate(parties):
            target_label = "OTHER_INELIGIBLE" if p_code == "REST" else p_code
            cv_map[c_code][target_label] = int(X[i, j])

    res = allocate_riksdag_seats(
        constituency_votes=cv_map,
        fixed_seats_by_constituency=fixed_seats_dict,
    )
    final_p = {p: res.final_seats_by_party.get(p, 0) for p in parties if p != "REST"}
    return final_p, res
