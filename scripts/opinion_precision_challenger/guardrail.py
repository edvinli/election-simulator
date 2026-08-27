"""Two-stage election-level guardrail evaluation (Experiment 2).

Stage 2 evaluates the precision challenger and the frozen RC1 baseline through
the same national state/dynamics/election-noise layers, geographic projection,
integerization, and production mandate dispatcher.  The dispatcher is the
production fast path with an exact legal allocator fallback; its 349-seat
invariant is checked for every draw.

The guardrail is deliberately fail-closed.  ``EVALUATED`` is returned only
when all 2 elections x 6 horizons have finite vote and seat scores for both
arms.  Missing data or an exception is reported as ``FAILED`` and can never
be interpreted as a passed guardrail.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals
from scripts.elections.load import load_election_targets_for_forecasting
from scripts.geography.config import OFFICIAL_CONSTITUENCY_CODES
from scripts.geography.integerization import biproportional_controlled_rounding
from scripts.geography.projection import _get_cached_geography_structures
from scripts.hindcasts.models import (
    derive_opinion_state_seed,
    derive_shared_dynamics_seed,
    sample_shared_symmetric_dynamics,
)
from scripts.mandates.config import FIXED_SEATS_2018, FIXED_SEATS_2022, FIXED_SEATS_2026
from scripts.pollofpolls.state import OpinionState, load_individual_polls_dataset, load_timeseries_dataset
from scripts.pollofpolls.state_config import ALL_CATEGORIES
from scripts.pollofpolls.transitions import build_all_historical_transitions, filter_transitions_as_of
from scripts.pop_baseline.metrics import continuous_crps, energy_score
from scripts.seat_hindcasts.config import EVALUATION_ELECTIONS as SEAT_EVALUATION_ELECTIONS
from scripts.seat_hindcasts.metrics import calculate_discrete_seat_crps, calculate_multivariate_energy_score
from scripts.simulator.config import PARLIAMENTARY_PARTIES_8
from scripts.simulator.engine import _apportion_constituency_units_of_25, _apportion_national_party_integers
from scripts.simulator.fast_allocator import dispatch_production_allocation
from scripts.vote_share_calibration.config import DEFAULT_ELECTIONS_FILE, DEFAULT_POLLS_FILE, MIN_SHARE_PCT
from scripts.vote_share_calibration.models import apply_vote_share_models, derive_vote_share_layer_seeds

from .config import ALL_CATEGORIES_9, GUARDRAIL_ELECTIONS, GUARDRAIL_HORIZONS, GUARDRAIL_MAX_DEGRADATION, POLLS_FILE, POP_TIMESERIES_FILE
from .opinion_state import estimate_opinion_with_precision_arm
from .precision import estimate_pollster_precision


def _fixed_seat_array(target_year: int) -> np.ndarray:
    """Return the official fixed-seat map in canonical constituency order."""
    if target_year == 2018:
        fixed = FIXED_SEATS_2018
    elif target_year == 2022:
        fixed = FIXED_SEATS_2022
    else:
        fixed = FIXED_SEATS_2026
    return np.array([fixed[c] for c in OFFICIAL_CONSTITUENCY_CODES], dtype=np.int64)


def _generate_national_samples_from_state(
    opinion_state: OpinionState,
    election_date: date,
    samples: int,
    seed: int,
    timeseries_data: Sequence[dict[str, Any]],
    polls_file: Path | str = DEFAULT_POLLS_FILE,
    elections_file: Path | str = DEFAULT_ELECTIONS_FILE,
) -> np.ndarray:
    """Run all national layers with an explicitly supplied OpinionState.

    This mirrors ``generate_national_vote_shares`` exactly after its state fit,
    allowing the challenger to replace only OpinionState while sharing every
    random stream and every downstream layer with RC1.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")

    as_of_date = opinion_state.as_of
    horizon_days = max(1, (election_date - as_of_date).days)
    state_seed = derive_opinion_state_seed(seed, as_of_date)
    dyn_seed = derive_shared_dynamics_seed(seed, as_of_date, horizon_days)
    idx_seed, sign_seed = derive_vote_share_layer_seeds(seed, as_of_date, horizon_days)

    state_samples = opinion_state.sample(n=samples, seed=state_seed)
    state_matrix = np.array([[sample[c] for c in ALL_CATEGORIES] for sample in state_samples], dtype=np.float64)
    state_fractions = state_matrix / np.sum(state_matrix, axis=1, keepdims=True)
    state_clr = np.log(state_fractions)
    state_clr -= np.mean(state_clr, axis=1, keepdims=True)

    eval_h = min(horizon_days, 112) if horizon_days > 112 else horizon_days
    all_transitions = build_all_historical_transitions(timeseries_data, horizons=[eval_h])
    eligible_transitions = filter_transitions_as_of(all_transitions[eval_h], as_of_date)
    if len(eligible_transitions) < 30:
        for fallback_h in (28, 14, 7):
            fallback = build_all_historical_transitions(timeseries_data, horizons=[fallback_h])
            fallback_pool = filter_transitions_as_of(fallback[fallback_h], as_of_date)
            if len(fallback_pool) >= 30:
                eligible_transitions = fallback_pool
                break
    if len(eligible_transitions) < 30:
        raise RuntimeError(f"insufficient historical transitions for {as_of_date} ({len(eligible_transitions)} < 30)")

    symmetric_deltas = sample_shared_symmetric_dynamics(
        eligible_transitions=eligible_transitions,
        samples_count=samples,
        seed=dyn_seed,
    )
    base_clr = state_clr + symmetric_deltas
    max_vals = np.max(base_clr, axis=1, keepdims=True)
    base_comp = 100.0 * np.exp(base_clr - max_vals)
    base_comp /= np.sum(base_comp, axis=1, keepdims=True)

    training_pool = load_chronological_pp_residuals(
        target_election_year=election_date.year,
        polls_file=polls_file,
        elections_file=elections_file,
    )
    model_draws = apply_vote_share_models(
        base_comp_matrix=base_comp,
        training_pool=training_pool,
        samples_count=samples,
        index_seed=idx_seed,
        sign_seed=sign_seed,
        eps=MIN_SHARE_PCT,
    )
    nat_shares, _ = model_draws["pp_centered_noise"]
    return nat_shares / np.sum(nat_shares, axis=1, keepdims=True)


def _project_and_allocate_seats(
    national_shares: np.ndarray,
    election_date: date,
    baseline_year: int,
    geography_mode: str = "chronological",
    total_national_votes: int = 6_500_000,
) -> np.ndarray:
    """Project national draws and allocate every draw through the legal path."""
    nat = np.asarray(national_shares, dtype=np.float64)
    if nat.ndim != 2 or nat.shape[1] != len(ALL_CATEGORIES_9) or nat.shape[0] == 0:
        raise ValueError("national_shares must have shape (N, 9) with N > 0")
    if not np.isfinite(nat).all() or np.any(nat < 0.0):
        raise ValueError("national_shares must be finite and non-negative")
    nat = nat / np.sum(nat, axis=1, keepdims=True)

    geo_dir = Path(__file__).resolve().parents[2] / "data" / "processed" / "geography"
    B_base, R_base = _get_cached_geography_structures(
        baseline_year=baseline_year,
        target_year=election_date.year,
        mode=geography_mode,
        processed_dir_str=str(geo_dir),
    )
    R_int = _apportion_constituency_units_of_25(R_base, total_national_votes)
    R_col = R_int[:, np.newaxis].astype(np.float64)
    fixed_seats = _fixed_seat_array(election_date.year)
    seats = np.zeros((len(nat), len(PARLIAMENTARY_PARTIES_8)), dtype=np.int64)
    X_buf = np.empty_like(B_base)

    for row_idx, share_row in enumerate(nat):
        C_int = _apportion_national_party_integers(share_row, total_national_votes)
        C_row = C_int[np.newaxis, :].astype(np.float64)
        np.copyto(X_buf, B_base)
        for _ in range(8):
            row_sums = np.sum(X_buf, axis=1, keepdims=True)
            X_buf *= R_col / np.maximum(row_sums, 1e-12)
            col_sums = np.sum(X_buf, axis=0, keepdims=True)
            X_buf *= C_row / np.maximum(col_sums, 1e-12)
        rounded = biproportional_controlled_rounding(X_buf, R_int, C_int, solver="auto").rounded_matrix
        dispatch = dispatch_production_allocation(rounded, fixed_seats_arr=fixed_seats)
        for party_idx, party in enumerate(PARLIAMENTARY_PARTIES_8):
            seats[row_idx, party_idx] = int(dispatch.seats_by_party.get(party, 0))

    totals = np.sum(seats, axis=1)
    if not np.all(totals == 349):
        bad = np.where(totals != 349)[0]
        raise RuntimeError(f"exact seat-total invariant failed for {len(bad)} draws: {totals[bad[:10]].tolist()}")
    return seats


def _score_election_case(
    rc1_vote_shares: np.ndarray,
    challenger_vote_shares: np.ndarray,
    rc1_seats: np.ndarray,
    challenger_seats: np.ndarray,
    actual_vote_shares: np.ndarray,
    actual_seats: np.ndarray,
) -> Dict[str, Any]:
    """Calculate finite vote/seat proper scores and hard seat invariants."""
    vote_scores: Dict[str, float] = {}
    seat_scores: Dict[str, float] = {}
    for model, votes, seats in (("rc1", rc1_vote_shares, rc1_seats), ("precision", challenger_vote_shares, challenger_seats)):
        votes_pct = np.asarray(votes, dtype=np.float64) * 100.0
        vote_scores[f"{model}_vote_crps_8"] = float(np.mean([continuous_crps(votes_pct[:, idx], actual_vote_shares[idx]) for idx in range(8)]))
        vote_scores[f"{model}_vote_energy_9"] = float(energy_score(votes_pct, actual_vote_shares))
        seat_scores[f"{model}_seat_crps_8"] = float(np.mean([calculate_discrete_seat_crps(seats[:, idx], actual_seats[idx]) for idx in range(8)]))
        seat_scores[f"{model}_seat_energy_8"] = float(calculate_multivariate_energy_score(seats, actual_seats))
        totals = np.sum(seats, axis=1)
        seat_scores[f"{model}_seat_total_min"] = int(np.min(totals))
        seat_scores[f"{model}_seat_total_max"] = int(np.max(totals))
        seat_scores[f"{model}_all_seat_totals_349"] = bool(np.all(totals == 349))

    numeric_values = [value for value in {**vote_scores, **seat_scores}.values() if isinstance(value, (int, float))]
    if not all(np.isfinite(float(value)) for value in numeric_values):
        raise RuntimeError("non-finite vote or seat score in guardrail case")
    return {**vote_scores, **seat_scores}


def _relative_degradation(candidate: float, baseline: float) -> float:
    """Return candidate-minus-baseline relative score change (positive is worse)."""
    return float((candidate - baseline) / max(abs(baseline), 1e-12))


def evaluate_election_guardrail(
    is_rolling_gate_passed: bool,
    samples: int = 5000,
    seed: int = 12345,
    elections: Sequence[date] = GUARDRAIL_ELECTIONS,
    horizons: Sequence[int] = GUARDRAIL_HORIZONS,
) -> Dict[str, Any]:
    """Execute the paired election/seat guardrail.

    The default is all 12 election-by-horizon cases.  ``elections`` and
    ``horizons`` are injectable solely for bounded offline smoke evaluations;
    production QA keeps the defaults and records the requested case count.
    """
    requested_elections = tuple(elections)
    requested_horizons = tuple(horizons)
    expected_cases = len(requested_elections) * len(requested_horizons)
    if not is_rolling_gate_passed:
        return {
            "status": "SKIPPED_ROLLING_GATE_FAILED",
            "evaluation_status": "NOT_RUN",
            "message": "Stage 2 election guardrail was not run because Stage 1 rolling gate failed.",
            "guardrail_passed": False,
            "evaluated_case_count": 0,
            "expected_case_count": expected_cases,
        }
    if samples <= 0:
        return {
            "status": "FAILED",
            "evaluation_status": "FAIL",
            "reason": "samples must be positive",
            "guardrail_passed": False,
            "evaluated_case_count": 0,
            "expected_case_count": expected_cases,
        }

    case_rows: List[Dict[str, Any]] = []
    try:
        individual_polls, _ = load_individual_polls_dataset(POLLS_FILE)
        pop_timeseries = load_timeseries_dataset(POP_TIMESERIES_FILE)
        targets = load_election_targets_for_forecasting()
        pop_by_date = {row["date"]: row["composition"] for row in pop_timeseries}

        for election_date in requested_elections:
            if election_date not in targets or str(election_date.year) not in SEAT_EVALUATION_ELECTIONS:
                raise KeyError(f"missing official target or seat metadata for {election_date}")
            metadata = SEAT_EVALUATION_ELECTIONS[str(election_date.year)]
            actual_vote = np.array([targets[election_date][category] for category in ALL_CATEGORIES_9], dtype=float)
            actual_seat = np.array([metadata["actual_seats"][party] for party in PARLIAMENTARY_PARTIES_8], dtype=np.int64)

            for horizon in requested_horizons:
                as_of = election_date - timedelta(days=horizon)
                rc1_state = estimate_opinion_with_precision_arm(
                    as_of,
                    individual_polls,
                    pop_timeseries,
                    weighting_arm="rc1_baseline",
                    data_dir=POLLS_FILE.parent,
                )
                precision_state = estimate_pollster_precision(as_of, individual_polls, pop_by_date)
                challenger_state = estimate_opinion_with_precision_arm(
                    as_of,
                    individual_polls,
                    pop_timeseries,
                    weighting_arm="precision_challenger",
                    precision_state=precision_state,
                )

                # Both arms use the same base seed and therefore identical
                # state, dynamics-index, dynamics-sign, election-index, and
                # election-sign random streams.  Only OpinionState covariance
                # differs between the matrices.
                rc1_votes = _generate_national_samples_from_state(rc1_state, election_date, samples, seed, pop_timeseries)
                challenger_votes = _generate_national_samples_from_state(challenger_state, election_date, samples, seed, pop_timeseries)
                baseline_year = int(metadata["geography_baseline_year"])
                rc1_seats = _project_and_allocate_seats(rc1_votes, election_date, baseline_year)
                challenger_seats = _project_and_allocate_seats(challenger_votes, election_date, baseline_year)
                scores = _score_election_case(rc1_votes, challenger_votes, rc1_seats, challenger_seats, actual_vote, actual_seat)
                score_changes = {
                    "relative_degradation_vote_crps_8": _relative_degradation(scores["precision_vote_crps_8"], scores["rc1_vote_crps_8"]),
                    "relative_degradation_vote_energy_9": _relative_degradation(scores["precision_vote_energy_9"], scores["rc1_vote_energy_9"]),
                    "relative_degradation_seat_crps_8": _relative_degradation(scores["precision_seat_crps_8"], scores["rc1_seat_crps_8"]),
                    "relative_degradation_seat_energy_8": _relative_degradation(scores["precision_seat_energy_8"], scores["rc1_seat_energy_8"]),
                }
                case_rows.append({"election_date": election_date.isoformat(), "origin_date": as_of.isoformat(), "horizon_days": horizon, "samples": samples, **scores, **score_changes})

        if len(case_rows) != expected_cases:
            raise RuntimeError(f"incomplete guardrail: {len(case_rows)} cases, expected {expected_cases}")

        degradation_values = [
            row[key]
            for row in case_rows
            for key in ("relative_degradation_vote_crps_8", "relative_degradation_vote_energy_9", "relative_degradation_seat_crps_8", "relative_degradation_seat_energy_8")
        ]
        max_degradation = float(max(degradation_values))
        all_seat_invariants = all(row["rc1_all_seat_totals_349"] and row["precision_all_seat_totals_349"] for row in case_rows)
        guardrail_passed = bool(max_degradation <= GUARDRAIL_MAX_DEGRADATION and all_seat_invariants)
        return {
            "status": "EVALUATED",
            "evaluation_status": "PASS" if guardrail_passed else "FAIL",
            "guardrail_passed": guardrail_passed,
            "max_allowable_degradation_pct": GUARDRAIL_MAX_DEGRADATION * 100.0,
            "max_observed_degradation_pct": max_degradation * 100.0,
            "elections_evaluated": [election.isoformat() for election in requested_elections],
            "horizons_evaluated": list(requested_horizons),
            "evaluated_case_count": len(case_rows),
            "expected_case_count": expected_cases,
            "all_seat_invariants_349": all_seat_invariants,
            "cases": case_rows,
        }
    except Exception as err:
        return {
            "status": "FAILED",
            "evaluation_status": "FAIL",
            "reason": f"{type(err).__name__}: {err}",
            "guardrail_passed": False,
            "evaluated_case_count": len(case_rows),
            "expected_case_count": expected_cases,
            "cases": case_rows,
        }
