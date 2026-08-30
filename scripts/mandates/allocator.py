"""Exact deterministic Riksdag mandate-allocation algorithm according to Swedish electoral law.

References:
    - Regeringsformen (1974:152) 3 kap. 7 § (4% national threshold and 12% constituency exception)
    - Vallagen (2005:837) 14 kap. 1–5 §§ (Modified Sainte-Laguë, fixed seats, national entitlement, excess returns, adjustment seats)

Two law versions are supported, selected explicitly via the ``law`` argument and
never inferred from the wall clock; see :mod:`scripts.mandates.law`.
``MandateLaw.POST_2018`` is the production default and its behaviour is
unchanged by the availability of the historical version.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any, Mapping, Sequence

from .config import (
    CONSTITUENCY_THRESHOLD_PCT,
    DEFAULT_INELIGIBLE_PARTIES,
    NATIONAL_THRESHOLD_PCT,
    OFFICIAL_CONSTITUENCIES,
    OFFICIAL_CONSTITUENCY_CODES,
    TOTAL_ADJUSTMENT_SEATS,
    TOTAL_FIXED_SEATS,
    TOTAL_RIKSDAG_SEATS,
)
from .law import MandateLaw
from .tie_breaker import DeterministicLotteryTieBreaker, TieBreaker

MAX_RETURN_ITERATIONS: int = 50


@dataclass(frozen=True)
class SeatAllocationEvent:
    """Audit log entry for each allocated, retracted, or reallocated mandate."""

    sequence: int
    phase: str  # "fixed", "excess_retracted", "returned_reallocated", "national_entitlement", "adjustment"
    constituency_code: str | None
    constituency_name: str | None
    party: str
    votes: int
    divisor: Fraction
    comparison_number: Fraction
    action: str  # "won", "retracted", "reassigned"
    reason: str


@dataclass(frozen=True)
class SeatAllocation:
    """Complete certified Riksdag mandate allocation result and full audit trail."""

    national_votes: dict[str, int]
    national_vote_shares: dict[str, Fraction]
    total_valid_votes: int
    threshold_eligibility: dict[str, bool]
    constituency_eligibility: dict[str, dict[str, bool]]
    initial_fixed_seats_by_party_constituency: dict[str, dict[str, int]]
    initial_national_fixed_seats: dict[str, int]
    national_entitlement: dict[str, int]
    returned_or_reallocated_seats: dict[str, dict[str, int]]
    final_fixed_seats_by_party_constituency: dict[str, dict[str, int]]
    final_national_fixed_seats: dict[str, int]
    adjustment_seats_by_party_constituency: dict[str, dict[str, int]]
    national_adjustment_seats: dict[str, int]
    final_seats_by_party_constituency: dict[str, dict[str, int]]
    final_seats_by_party: dict[str, int]
    total_seats: int
    event_log: tuple[SeatAllocationEvent, ...]
    law: str = MandateLaw.POST_2018.value
    #: Parties set aside as over-represented under PRE_2018 (empty under POST_2018).
    set_aside_parties: tuple[str, ...] = ()


def _normalize_first_divisor(divisor: Fraction | float | int) -> Fraction:
    """Normalize first divisor to exact Fraction(6, 5) or validate exact custom fraction."""
    if isinstance(divisor, Fraction):
        return divisor
    if isinstance(divisor, int):
        return Fraction(divisor, 1)
    if isinstance(divisor, float):
        if math.isclose(divisor, 1.2, rel_tol=1e-9):
            return Fraction(6, 5)
        if math.isclose(divisor, 1.4, rel_tol=1e-9):
            return Fraction(7, 5)
        if math.isclose(divisor, 1.0, rel_tol=1e-9):
            return Fraction(1, 1)
        raise ValueError(
            f"Unsupported floating-point first divisor {divisor}. Pass exact Fraction(numerator, denominator)."
        )
    raise TypeError(f"Invalid type for first divisor: {type(divisor)}")


def _validate_allocator_inputs(
    constituency_votes: Mapping[str, Mapping[str, int]],
    fixed_seats_by_constituency: Mapping[str, int],
    total_seats: int,
) -> None:
    """Strict input validation for mandate allocator."""
    if total_seats != TOTAL_RIKSDAG_SEATS:
        raise ValueError(f"Total Riksdag seats must be {TOTAL_RIKSDAG_SEATS}, got {total_seats}")

    if not constituency_votes:
        raise ValueError("constituency_votes mapping cannot be empty")

    missing_constituencies = set(OFFICIAL_CONSTITUENCY_CODES) - set(constituency_votes.keys())
    if missing_constituencies:
        raise ValueError(f"Missing official constituencies in vote data: {sorted(missing_constituencies)}")

    for c, c_votes in constituency_votes.items():
        if not isinstance(c_votes, Mapping):
            raise TypeError(f"Constituency {c} votes must be a mapping of party -> votes")
        for p, v in c_votes.items():
            if not isinstance(v, int) or isinstance(v, bool):
                raise TypeError(f"Vote count for party {p} in constituency {c} must be an integer, got {type(v)}")
            if v < 0:
                raise ValueError(f"Negative vote count ({v}) for party {p} in constituency {c}")
        c_tot = sum(c_votes.values())
        if c_tot <= 0:
            raise ValueError(f"Constituency {c} has zero total valid votes")

    missing_fixed = set(OFFICIAL_CONSTITUENCY_CODES) - set(fixed_seats_by_constituency.keys())
    if missing_fixed:
        raise ValueError(f"Missing fixed seats configuration for constituencies: {sorted(missing_fixed)}")

    for c, f in fixed_seats_by_constituency.items():
        if not isinstance(f, int) or isinstance(f, bool):
            raise TypeError(f"Fixed seat count for constituency {c} must be an integer, got {type(f)}")
        if f <= 0:
            raise ValueError(f"Invalid fixed seat count ({f}) for constituency {c}")

    tot_fixed = sum(fixed_seats_by_constituency.values())
    if tot_fixed != TOTAL_FIXED_SEATS:
        raise ValueError(f"Total fixed seats must equal {TOTAL_FIXED_SEATS}, got {tot_fixed}")


def _compute_national_entitlement(
    national_votes: Mapping[str, int],
    eligible_parties: Sequence[str],
    seats_to_distribute: int,
    first_divisor: Fraction,
    tie_breaker: TieBreaker,
    seq_counter: int,
    event_log: list[SeatAllocationEvent],
    phase_label: str = "national_entitlement",
    scenario_id: str = "production",
) -> tuple[dict[str, int], int]:
    """Compute national proportional entitlement across qualifying parties using modified Sainte-Laguë."""
    entitlement: dict[str, int] = {p: 0 for p in eligible_parties}
    if seats_to_distribute <= 0 or not eligible_parties:
        return entitlement, seq_counter

    for seat_idx in range(seats_to_distribute):
        comp_numbers: dict[str, tuple[Fraction, Fraction]] = {}
        for p in eligible_parties:
            k = entitlement[p]
            div = first_divisor if k == 0 else Fraction(2 * k + 1, 1)
            comp_val = Fraction(national_votes[p], 1) / div
            comp_numbers[p] = (comp_val, div)

        max_comp = max(c_val for c_val, _ in comp_numbers.values())
        tied_parties = [p for p, (c_val, _) in comp_numbers.items() if c_val == max_comp]

        if len(tied_parties) == 1:
            winner = tied_parties[0]
        else:
            winner = tie_breaker.pick_winner(
                tied_parties,
                context={
                    "phase": phase_label,
                    "scenario_id": scenario_id,
                    "constituency": "national",
                    "seat_index": seat_idx,
                    "comparison_number": max_comp,
                    "current_seats": tuple((p, entitlement[p]) for p in sorted(eligible_parties)),
                },
            )

        win_comp, win_div = comp_numbers[winner]
        entitlement[winner] += 1
        seq_counter += 1

        event_log.append(
            SeatAllocationEvent(
                sequence=seq_counter,
                phase=phase_label,
                constituency_code=None,
                constituency_name="National",
                party=winner,
                votes=national_votes[winner],
                divisor=win_div,
                comparison_number=win_comp,
                action="won",
                reason=f"Won national entitlement seat {seat_idx + 1}/{seats_to_distribute}",
            )
        )

    return entitlement, seq_counter


def _pre_2018_national_entitlement(
    national_votes: Mapping[str, int],
    nationally_eligible_parties: Sequence[str],
    fixed_seats_by_party: Mapping[str, int],
    seats_to_distribute: int,
    first_divisor: Fraction,
    tie_breaker: TieBreaker,
    seq_counter: int,
    event_log: list[SeatAllocationEvent],
    scenario_id: str,
) -> tuple[dict[str, int], int, list[str]]:
    """National entitlement under the law in force before SFS 2014:1384.

    There is no mandate return. A party whose fixed constituency seats exceed its
    nationwide proportional entitlement keeps those seats and is set aside,
    together with them, from the remaining distribution, which is then carried
    out among the other participating parties so that those are proportional
    among themselves (3 kap. 8 § RF; 14 kap. 5 § vallagen in its pre-2018
    wording; prop. 2013/14:48 §4.1.1).

    The set-aside test is iterated because removing one over-represented party
    can push another above its recomputed entitlement.

    Returns:
        (entitlement including set-aside parties at their fixed-seat count,
         updated sequence counter, set-aside parties in deterministic order)
    """
    set_aside: list[str] = []
    for _ in range(len(nationally_eligible_parties) + 1):
        remaining = [p for p in nationally_eligible_parties if p not in set_aside]
        aside_fixed = sum(fixed_seats_by_party[p] for p in set_aside)
        entitlement, seq_counter = _compute_national_entitlement(
            national_votes=national_votes,
            eligible_parties=remaining,
            seats_to_distribute=seats_to_distribute - aside_fixed,
            first_divisor=first_divisor,
            tie_breaker=tie_breaker,
            seq_counter=seq_counter,
            event_log=event_log,
            phase_label="national_entitlement",
            scenario_id=scenario_id,
        )
        newly_over = sorted(
            p for p in remaining if fixed_seats_by_party[p] > entitlement[p]
        )
        if not newly_over:
            full_entitlement: dict[str, int] = {p: fixed_seats_by_party[p] for p in set_aside}
            full_entitlement.update(entitlement)
            return full_entitlement, seq_counter, set_aside
        set_aside.extend(newly_over)

    raise RuntimeError(
        "Pre-2018 set-aside iteration failed to converge: every nationally eligible "
        "party was set aside as over-represented."
    )


def allocate_riksdag_seats(
    constituency_votes: Mapping[str, Mapping[str, int]],
    fixed_seats_by_constituency: Mapping[str, int],
    first_divisor: Fraction | float | int = Fraction(6, 5),
    national_threshold_pct: float = NATIONAL_THRESHOLD_PCT,
    constituency_threshold_pct: float = CONSTITUENCY_THRESHOLD_PCT,
    total_seats: int = TOTAL_RIKSDAG_SEATS,
    tie_breaker: TieBreaker | None = None,
    ineligible_parties: set[str] | Sequence[str] | None = None,
    scenario_id: str = "production",
    law: MandateLaw = MandateLaw.POST_2018,
) -> SeatAllocation:
    """Execute complete certified Riksdag seat allocation according to Swedish electoral law.

    Law reference: Vallagen 14 kap. 1–5 §§.

    Parameters:
        law: Version of Vallagen 14 kap. to apply. ``POST_2018`` (default,
            production) uses mandate return (14 kap. 4a–4c §§). ``PRE_2018``
            uses the set-aside rule that governed the 2010 and 2014 elections.
            The caller is responsible for passing the matching ``first_divisor``
            (1.2 post-2018, 1.4 pre-2018); use
            :func:`scripts.mandates.law.mandate_law_for_election_year` to obtain
            a self-consistent pair. Never inferred from the current date.
    """
    f_div = _normalize_first_divisor(first_divisor)
    _validate_allocator_inputs(constituency_votes, fixed_seats_by_constituency, total_seats)

    tb = tie_breaker if tie_breaker is not None else DeterministicLotteryTieBreaker(seed=12345)
    ineligible_set = set(ineligible_parties) if ineligible_parties is not None else set(DEFAULT_INELIGIBLE_PARTIES)

    constituencies = [c for c in OFFICIAL_CONSTITUENCY_CODES if c in constituency_votes]
    all_parties = sorted({p for c in constituencies for p in constituency_votes[c].keys()})

    # 1. National Vote Totals & Threshold Eligibility (Regeringsformen 3 kap. 7 §)
    national_votes: dict[str, int] = {
        p: sum(constituency_votes[c].get(p, 0) for c in constituencies) for p in all_parties
    }
    total_valid_votes = sum(national_votes.values())

    national_vote_shares: dict[str, Fraction] = {
        p: Fraction(national_votes[p], total_valid_votes) for p in all_parties
    }

    # National 4.0% threshold (inclusive: 25 * votes >= total_valid_votes)
    threshold_eligibility: dict[str, bool] = {
        p: (p not in ineligible_set and (25 * national_votes[p] >= total_valid_votes)) for p in all_parties
    }

    # Constituency 12.0% exception (inclusive: 25 * p_votes >= 3 * c_valid_votes)
    constituency_eligibility: dict[str, dict[str, bool]] = {}
    const_valid_votes: dict[str, int] = {}
    for c in constituencies:
        c_valid = sum(constituency_votes[c].values())
        const_valid_votes[c] = c_valid
        constituency_eligibility[c] = {}
        for p in all_parties:
            p_votes = constituency_votes[c].get(p, 0)
            is_12_pct = (25 * p_votes >= 3 * c_valid) if c_valid > 0 else False
            constituency_eligibility[c][p] = (
                p not in ineligible_set and (threshold_eligibility[p] or is_12_pct)
            )

    event_log: list[SeatAllocationEvent] = []
    seq_counter = 0

    # 2. Fixed Constituency Seats Allocation (Vallagen 14 kap. 3 §)
    initial_fixed_alloc: dict[str, dict[str, int]] = {
        c: {p: 0 for p in all_parties} for c in constituencies
    }
    won_fixed_seats_records: dict[str, list[dict[str, Any]]] = {c: [] for c in constituencies}

    for c in constituencies:
        f_c = fixed_seats_by_constituency[c]
        c_name = OFFICIAL_CONSTITUENCIES.get(c, c)
        participating_parties = [
            p for p in all_parties if constituency_eligibility[c][p] and constituency_votes[c].get(p, 0) > 0
        ]

        for seat_idx in range(f_c):
            comp_numbers: dict[str, tuple[Fraction, Fraction]] = {}
            for p in participating_parties:
                k = initial_fixed_alloc[c][p]
                div = f_div if k == 0 else Fraction(2 * k + 1, 1)
                comp_val = Fraction(constituency_votes[c][p], 1) / div
                comp_numbers[p] = (comp_val, div)

            max_comp = max(c_val for c_val, _ in comp_numbers.values())
            tied_parties = [p for p, (c_val, _) in comp_numbers.items() if c_val == max_comp]

            if len(tied_parties) == 1:
                winner = tied_parties[0]
            else:
                winner = tb.pick_winner(
                    tied_parties,
                    context={
                        "phase": "fixed",
                        "scenario_id": scenario_id,
                        "constituency": c,
                        "seat_index": seat_idx,
                        "comparison_number": max_comp,
                        "current_seats": tuple((p, initial_fixed_alloc[c][p]) for p in sorted(participating_parties)),
                    },
                )

            win_comp, win_div = comp_numbers[winner]
            initial_fixed_alloc[c][winner] += 1
            seq_counter += 1

            won_fixed_seats_records[c].append({
                "party": winner,
                "seat_in_constituency": initial_fixed_alloc[c][winner],
                "comparison_number": win_comp,
                "divisor": win_div,
            })

            event_log.append(
                SeatAllocationEvent(
                    sequence=seq_counter,
                    phase="fixed",
                    constituency_code=c,
                    constituency_name=c_name,
                    party=winner,
                    votes=constituency_votes[c][winner],
                    divisor=win_div,
                    comparison_number=win_comp,
                    action="won",
                    reason=f"Won fixed constituency seat {seat_idx + 1}/{f_c} in {c_name}",
                )
            )

    initial_national_fixed: dict[str, int] = {
        p: sum(initial_fixed_alloc[c][p] for c in constituencies) for p in all_parties
    }

    # 3. Iterative Excess Fixed Seat Returns and Reallocation Convergence Loop
    # (Vallagen 14 kap. 4a–4c §§)
    active_fixed_alloc: dict[str, dict[str, int]] = {
        c: dict(initial_fixed_alloc[c]) for c in constituencies
    }
    returned_or_reallocated: dict[str, dict[str, int]] = {
        c: {p: 0 for p in all_parties} for c in constituencies
    }

    nationally_eligible_parties = [p for p in all_parties if threshold_eligibility[p]]
    returned_seats_pending: dict[str, int] = {c: 0 for c in constituencies}

    iteration = 0
    seen_states: set[str] = set()
    national_entitlement: dict[str, int] = {p: 0 for p in nationally_eligible_parties}
    set_aside_parties: list[str] = []

    while True:
        iteration += 1
        if iteration > MAX_RETURN_ITERATIONS:
            raise RuntimeError(f"Mandate return/reallocation failed to converge within {MAX_RETURN_ITERATIONS} iterations")

        # L = fixed seats held by parties qualifying only locally (sub-4% parties with >= 12% locally)
        L = sum(
            sum(active_fixed_alloc[c][p] for c in constituencies)
            for p in all_parties
            if not threshold_eligibility[p]
        )
        seats_to_distribute = total_seats - L

        if law is MandateLaw.PRE_2018:
            # Pre-SFS 2014:1384: no return. Over-represented parties keep their
            # fixed seats and are set aside from the remaining distribution.
            # ``active_fixed_alloc`` is therefore never modified.
            national_entitlement, seq_counter, set_aside_parties = _pre_2018_national_entitlement(
                national_votes=national_votes,
                nationally_eligible_parties=nationally_eligible_parties,
                fixed_seats_by_party={
                    p: sum(active_fixed_alloc[c][p] for c in constituencies)
                    for p in nationally_eligible_parties
                },
                seats_to_distribute=seats_to_distribute,
                first_divisor=f_div,
                tie_breaker=tb,
                seq_counter=seq_counter,
                event_log=event_log,
                scenario_id=scenario_id,
            )
            break

        # Step 3A: Compute national entitlement E_p for qualifying parties
        national_entitlement, seq_counter = _compute_national_entitlement(
            national_votes=national_votes,
            eligible_parties=nationally_eligible_parties,
            seats_to_distribute=seats_to_distribute,
            first_divisor=f_div,
            tie_breaker=tb,
            seq_counter=seq_counter,
            event_log=event_log,
            phase_label="national_entitlement",
            scenario_id=scenario_id,
        )

        # Step 3B: Detect excess fixed seats (overhang) for qualifying parties: F_p > E_p
        current_national_fixed = {
            p: sum(active_fixed_alloc[c][p] for c in constituencies) for p in nationally_eligible_parties
        }
        excess_by_party = {
            p: max(0, current_national_fixed[p] - national_entitlement.get(p, 0))
            for p in nationally_eligible_parties
        }

        total_excess = sum(excess_by_party.values())
        if total_excess == 0 and sum(returned_seats_pending.values()) == 0:
            # Stable convergence reached!
            break

        # State representation for cycle detection
        state_key = f"{L}|" + ",".join(
            f"{c}:{p}:{active_fixed_alloc[c][p]}" for c in sorted(constituencies) for p in sorted(all_parties)
        )
        if state_key in seen_states:
            raise RuntimeError("Mandate return loop detected an allocation state cycle")
        seen_states.add(state_key)

        # Step 3C: Retract excess seats from lowest winning comparison quotient (Vallagen 14 kap. 4a §)
        for p, num_excess in excess_by_party.items():
            if num_excess <= 0:
                continue

            for _ in range(num_excess):
                candidate_returns: list[tuple[str, Fraction, dict[str, Any]]] = []
                for c in constituencies:
                    if fixed_seats_by_constituency[c] < 3:
                        # 14 kap. 4a § st. 2: cannot return seat from constituency with fewer than 3 fixed seats
                        continue
                    if active_fixed_alloc[c][p] > 0:
                        p_records = [rec for rec in won_fixed_seats_records[c] if rec["party"] == p]
                        if p_records:
                            candidate_returns.append((c, p_records[-1]["comparison_number"], p_records[-1]))

                if not candidate_returns:
                    raise RuntimeError(
                        f"Party {p} has {num_excess} excess seats but no eligible constituency (>=3 fixed seats) to return from."
                    )

                min_comp_val = min(c_comp for _, c_comp, _ in candidate_returns)
                tied_retracts = [
                    (c, rec) for c, c_comp, rec in candidate_returns if c_comp == min_comp_val
                ]

                if len(tied_retracts) == 1:
                    retract_c, retract_rec = tied_retracts[0]
                else:
                    win_idx = tb.pick_winner(
                        [i for i in range(len(tied_retracts))],
                        context={
                            "phase": "excess_retracted",
                            "scenario_id": scenario_id,
                            "party": p,
                            "comparison_number": min_comp_val,
                            "tied_constituencies": tuple(sorted(c for c, _ in tied_retracts)),
                        },
                    )
                    retract_c, retract_rec = tied_retracts[win_idx]

                active_fixed_alloc[retract_c][p] -= 1
                returned_or_reallocated[retract_c][p] -= 1
                returned_seats_pending[retract_c] += 1
                won_fixed_seats_records[retract_c].remove(retract_rec)
                seq_counter += 1

                retract_c_name = OFFICIAL_CONSTITUENCIES.get(retract_c, retract_c)
                event_log.append(
                    SeatAllocationEvent(
                        sequence=seq_counter,
                        phase="excess_retracted",
                        constituency_code=retract_c,
                        constituency_name=retract_c_name,
                        party=p,
                        votes=constituency_votes[retract_c][p],
                        divisor=retract_rec["divisor"],
                        comparison_number=retract_rec["comparison_number"],
                        action="retracted",
                        reason=f"Excess fixed seat retracted from {p} in {retract_c_name} (original comp: {retract_rec['comparison_number']})",
                    )
                )

        # Step 3D: Reallocate returned seats within originating constituencies (Vallagen 14 kap. 4b §)
        total_pending = sum(returned_seats_pending.values())
        for realloc_idx in range(total_pending):
            candidate_recipients: list[tuple[str, str, Fraction, Fraction]] = []

            for c, count_ret in returned_seats_pending.items():
                if count_ret <= 0:
                    continue

                for q in all_parties:
                    if not constituency_eligibility[c][q]:
                        continue
                    if constituency_votes[c].get(q, 0) <= 0:
                        continue

                    # If q is nationally eligible, do not award if party already holds its entitlement
                    if threshold_eligibility[q]:
                        q_nat_fixed = sum(active_fixed_alloc[k][q] for k in constituencies)
                        if q_nat_fixed >= national_entitlement[q]:
                            continue

                    k_q = active_fixed_alloc[c][q]
                    div_q = f_div if k_q == 0 else Fraction(2 * k_q + 1, 1)
                    comp_q = Fraction(constituency_votes[c][q], 1) / div_q
                    candidate_recipients.append((c, q, comp_q, div_q))

            if not candidate_recipients:
                raise RuntimeError(
                    f"No eligible recipient party found for returned seats (reallocation {realloc_idx + 1}/{total_pending})"
                )

            max_realloc_comp = max(c_val for _, _, c_val, _ in candidate_recipients)
            tied_recipients = [
                (c, q, c_val, div) for c, q, c_val, div in candidate_recipients if c_val == max_realloc_comp
            ]

            if len(tied_recipients) == 1:
                win_c, win_q, win_c_val, win_div = tied_recipients[0]
            else:
                win_idx = tb.pick_winner(
                    [i for i in range(len(tied_recipients))],
                    context={
                        "phase": "returned_reallocated",
                        "scenario_id": scenario_id,
                        "step": realloc_idx,
                        "comparison_number": max_realloc_comp,
                        "tied_candidates": tuple(sorted((c, q) for c, q, _, _ in tied_recipients)),
                    },
                )
                win_c, win_q, win_c_val, win_div = tied_recipients[win_idx]

            active_fixed_alloc[win_c][win_q] += 1
            returned_or_reallocated[win_c][win_q] += 1
            returned_seats_pending[win_c] -= 1
            seq_counter += 1

            won_fixed_seats_records[win_c].append({
                "party": win_q,
                "seat_in_constituency": active_fixed_alloc[win_c][win_q],
                "comparison_number": win_c_val,
                "divisor": win_div,
            })

            win_c_name = OFFICIAL_CONSTITUENCIES.get(win_c, win_c)
            event_log.append(
                SeatAllocationEvent(
                    sequence=seq_counter,
                    phase="returned_reallocated",
                    constituency_code=win_c,
                    constituency_name=win_c_name,
                    party=win_q,
                    votes=constituency_votes[win_c][win_q],
                    divisor=win_div,
                    comparison_number=win_c_val,
                    action="reassigned",
                    reason=f"Returned fixed seat reassigned to {win_q} in {win_c_name}",
                )
            )

    # 4. Mandatory Pre-Adjustment Invariant Verifications (Vallagen 14 kap. 4c §)
    final_fixed_alloc = active_fixed_alloc
    final_national_fixed: dict[str, int] = {
        p: sum(final_fixed_alloc[c][p] for c in constituencies) for p in all_parties
    }
    L_final = sum(final_national_fixed[p] for p in all_parties if not threshold_eligibility[p])

    # Hard Invariants
    total_fixed_final = sum(final_national_fixed.values())
    if total_fixed_final != TOTAL_FIXED_SEATS:
        raise RuntimeError(f"Total fixed seats invariant violated: {total_fixed_final} != {TOTAL_FIXED_SEATS}")

    sum_E_Q = sum(national_entitlement[p] for p in nationally_eligible_parties)
    if sum_E_Q != (total_seats - L_final):
        raise RuntimeError(f"National entitlement sum invariant violated: {sum_E_Q} != {total_seats - L_final}")

    for p in nationally_eligible_parties:
        if final_national_fixed[p] > national_entitlement[p]:
            raise RuntimeError(
                f"Overhang invariant violated at convergence for party {p}: {final_national_fixed[p]} > {national_entitlement[p]}"
            )

    # 5. Adjustment Seat Allocation to Constituencies (Utjämningsmandat - Vallagen 14 kap. 5 §)
    adjustment_seats_needed: dict[str, int] = {
        p: national_entitlement[p] - final_national_fixed[p]
        for p in nationally_eligible_parties
    }
    total_adjustment_needed = sum(adjustment_seats_needed.values())
    if total_adjustment_needed != TOTAL_ADJUSTMENT_SEATS:
        raise RuntimeError(
            f"Total adjustment seats needed ({total_adjustment_needed}) != {TOTAL_ADJUSTMENT_SEATS}"
        )

    adj_alloc: dict[str, dict[str, int]] = {
        c: {p: 0 for p in all_parties} for c in constituencies
    }

    for p in nationally_eligible_parties:
        u_p = adjustment_seats_needed[p]
        for adj_idx in range(u_p):
            comp_by_const: dict[str, tuple[Fraction, Fraction]] = {}
            for c in constituencies:
                v_cp = constituency_votes[c].get(p, 0)
                tot_seats_in_c = final_fixed_alloc[c][p] + adj_alloc[c][p]
                # Divisor is 1 if party has 0 seats in the constituency, else 2*k + 1 (Vallagen 14 kap. 5 §)
                div_adj = Fraction(1, 1) if tot_seats_in_c == 0 else Fraction(2 * tot_seats_in_c + 1, 1)
                comp_val_adj = Fraction(v_cp, 1) / div_adj
                comp_by_const[c] = (comp_val_adj, div_adj)

            max_comp_adj = max(c_val for c_val, _ in comp_by_const.values())
            tied_consts = [c for c, (c_val, _) in comp_by_const.items() if c_val == max_comp_adj]

            if len(tied_consts) == 1:
                win_c = tied_consts[0]
            else:
                win_c = tb.pick_winner(
                    tied_consts,
                    context={
                        "phase": "adjustment",
                        "scenario_id": scenario_id,
                        "party": p,
                        "adj_seat_index": adj_idx,
                        "comparison_number": max_comp_adj,
                        "current_adj_seats": tuple((c, adj_alloc[c][p]) for c in sorted(constituencies)),
                    },
                )

            win_comp_adj, win_div_adj = comp_by_const[win_c]
            adj_alloc[win_c][p] += 1
            seq_counter += 1

            win_c_name = OFFICIAL_CONSTITUENCIES.get(win_c, win_c)
            event_log.append(
                SeatAllocationEvent(
                    sequence=seq_counter,
                    phase="adjustment",
                    constituency_code=win_c,
                    constituency_name=win_c_name,
                    party=p,
                    votes=constituency_votes[win_c][p],
                    divisor=win_div_adj,
                    comparison_number=win_comp_adj,
                    action="won",
                    reason=f"Won adjustment seat {adj_idx + 1}/{u_p} for {p} in {win_c_name}",
                )
            )

    # 6. Final Seat Aggregations and Integrity Verifications
    final_seats_by_party_constituency: dict[str, dict[str, int]] = {
        c: {p: final_fixed_alloc[c][p] + adj_alloc[c][p] for p in all_parties}
        for c in constituencies
    }
    final_seats_by_party: dict[str, int] = {
        p: sum(final_seats_by_party_constituency[c][p] for c in constituencies)
        for p in all_parties
    }
    national_adjustment_seats: dict[str, int] = {
        p: sum(adj_alloc[c][p] for c in constituencies) for p in all_parties
    }

    # Final Total Verification: L + sum(F_p + U_p) == 349
    total_allocated = sum(final_seats_by_party.values())
    if total_allocated != total_seats:
        raise RuntimeError(f"Total allocated seats ({total_allocated}) != required {total_seats}")

    # Verify REST / ineligible parties receive 0 seats
    for p in ineligible_set:
        if final_seats_by_party.get(p, 0) != 0:
            raise RuntimeError(f"Ineligible party {p} received {final_seats_by_party[p]} seats")

    return SeatAllocation(
        national_votes=national_votes,
        national_vote_shares=national_vote_shares,
        total_valid_votes=total_valid_votes,
        threshold_eligibility=threshold_eligibility,
        constituency_eligibility=constituency_eligibility,
        initial_fixed_seats_by_party_constituency=initial_fixed_alloc,
        initial_national_fixed_seats=initial_national_fixed,
        national_entitlement=national_entitlement,
        returned_or_reallocated_seats=returned_or_reallocated,
        final_fixed_seats_by_party_constituency=final_fixed_alloc,
        final_national_fixed_seats=final_national_fixed,
        adjustment_seats_by_party_constituency=adj_alloc,
        national_adjustment_seats=national_adjustment_seats,
        final_seats_by_party_constituency=final_seats_by_party_constituency,
        final_seats_by_party=final_seats_by_party,
        total_seats=total_seats,
        event_log=tuple(event_log),
        law=law.value,
        set_aside_parties=tuple(set_aside_parties),
    )
