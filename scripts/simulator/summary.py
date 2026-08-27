"""Statistical summary objects and party-group aggregators for ElectionSimulator v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np

from .config import DEFAULT_MAJORITY_THRESHOLD, MODEL_PARTIES_9, PARLIAMENTARY_PARTIES_8


@dataclass(frozen=True)
class PartySummary:
    party: str
    vote_share_mean: float
    vote_share_median: float
    vote_share_p05: float
    vote_share_p10: float
    vote_share_p25: float
    vote_share_p75: float
    vote_share_p90: float
    vote_share_p95: float
    prob_above_4pct: float
    prob_below_4pct: float
    prob_largest_vote_party: float
    seats_mean: float
    seats_median: int
    seats_p05: int
    seats_p10: int
    seats_p25: int
    seats_p75: int
    seats_p90: int
    seats_p95: int
    prob_largest_seat_party: float
    prob_any_seats: float
    prob_local_12pct_exception_sub_4pct: float
    seat_histogram: dict[int, float]


@dataclass(frozen=True)
class GroupSummary:
    parties: tuple[str, ...]
    majority_threshold: int
    mean_seats: float
    median_seats: int
    p05_seats: int
    p10_seats: int
    p25_seats: int
    p75_seats: int
    p90_seats: int
    p95_seats: int
    prob_majority: float
    seat_histogram: dict[int, float]


@dataclass(frozen=True)
class SimulationSummary:
    as_of: str
    election_date: str
    total_samples: int
    base_seed: int
    parties: dict[str, PartySummary]
    largest_vote_party_probabilities: dict[str, float]
    largest_seat_party_probabilities: dict[str, float]
    manifest: dict[str, Any]

    def summarize_group(
        self,
        party_seats_matrix: np.ndarray,
        party_indices: Mapping[str, int],
        parties: Sequence[str],
        majority_threshold: int = DEFAULT_MAJORITY_THRESHOLD,
    ) -> GroupSummary:
        """Compute aggregate seats and majority probability for an arbitrary group of parties."""
        cols = [party_indices[p] for p in parties if p in party_indices]
        group_seats = np.sum(party_seats_matrix[:, cols], axis=1)

        unique_seats, counts = np.unique(group_seats, return_counts=True)
        hist = {int(k): float(v / len(group_seats)) for k, v in zip(unique_seats, counts)}

        return GroupSummary(
            parties=tuple(parties),
            majority_threshold=majority_threshold,
            mean_seats=float(np.mean(group_seats)),
            median_seats=int(np.median(group_seats)),
            p05_seats=int(np.percentile(group_seats, 5)),
            p10_seats=int(np.percentile(group_seats, 10)),
            p25_seats=int(np.percentile(group_seats, 25)),
            p75_seats=int(np.percentile(group_seats, 75)),
            p90_seats=int(np.percentile(group_seats, 90)),
            p95_seats=int(np.percentile(group_seats, 95)),
            prob_majority=float(np.mean(group_seats >= majority_threshold)),
            seat_histogram=hist,
        )


def compute_simulation_summary(
    as_of: str,
    election_date: str,
    vote_shares_matrix: np.ndarray,  # shape (N, 9)
    seats_matrix: np.ndarray,        # shape (N, 8)
    manifest: dict[str, Any],
    local_12_pct_flags: np.ndarray | None = None,  # shape (N, 8)
) -> tuple[SimulationSummary, GroupSummaryHelper]:
    """Compute canonical statistical summaries across all simulation draws."""
    n_samples, n_parties_9 = vote_shares_matrix.shape
    party_9_idx = {p: i for i, p in enumerate(MODEL_PARTIES_9)}
    party_8_idx = {p: i for i, p in enumerate(PARLIAMENTARY_PARTIES_8)}

    # Largest vote party per sample (among parliamentary parties)
    largest_vote_idx = np.argmax(vote_shares_matrix[:, :8], axis=1)
    # Largest seat party per sample
    largest_seat_idx = np.argmax(seats_matrix, axis=1)

    party_summaries: dict[str, PartySummary] = {}
    largest_vote_probs: dict[str, float] = {}
    largest_seat_probs: dict[str, float] = {}

    for i, p in enumerate(PARLIAMENTARY_PARTIES_8):
        v_shares = vote_shares_matrix[:, i]
        s_counts = seats_matrix[:, i]

        p_above_4 = float(np.mean(v_shares >= 0.04))
        p_below_4 = float(np.mean(v_shares < 0.04))
        p_largest_v = float(np.mean(largest_vote_idx == i))
        p_largest_s = float(np.mean(largest_seat_idx == i))
        p_any_seats = float(np.mean(s_counts > 0))

        if local_12_pct_flags is not None:
            p_12_sub4 = float(np.mean(local_12_pct_flags[:, i] & (v_shares < 0.04)))
        else:
            p_12_sub4 = 0.0

        unique_s, counts_s = np.unique(s_counts, return_counts=True)
        seat_hist = {int(k): float(v / n_samples) for k, v in zip(unique_s, counts_s)}

        largest_vote_probs[p] = p_largest_v
        largest_seat_probs[p] = p_largest_s

        party_summaries[p] = PartySummary(
            party=p,
            vote_share_mean=float(np.mean(v_shares)),
            vote_share_median=float(np.median(v_shares)),
            vote_share_p05=float(np.percentile(v_shares, 5)),
            vote_share_p10=float(np.percentile(v_shares, 10)),
            vote_share_p25=float(np.percentile(v_shares, 25)),
            vote_share_p75=float(np.percentile(v_shares, 75)),
            vote_share_p90=float(np.percentile(v_shares, 90)),
            vote_share_p95=float(np.percentile(v_shares, 95)),
            prob_above_4pct=p_above_4,
            prob_below_4pct=p_below_4,
            prob_largest_vote_party=p_largest_v,
            seats_mean=float(np.mean(s_counts)),
            seats_median=int(np.median(s_counts)),
            seats_p05=int(np.percentile(s_counts, 5)),
            seats_p10=int(np.percentile(s_counts, 10)),
            seats_p25=int(np.percentile(s_counts, 25)),
            seats_p75=int(np.percentile(s_counts, 75)),
            seats_p90=int(np.percentile(s_counts, 90)),
            seats_p95=int(np.percentile(s_counts, 95)),
            prob_largest_seat_party=p_largest_s,
            prob_any_seats=p_any_seats,
            prob_local_12pct_exception_sub_4pct=p_12_sub4,
            seat_histogram=seat_hist,
        )

    # REST summary
    rest_shares = vote_shares_matrix[:, 8]
    party_summaries["REST"] = PartySummary(
        party="REST",
        vote_share_mean=float(np.mean(rest_shares)),
        vote_share_median=float(np.median(rest_shares)),
        vote_share_p05=float(np.percentile(rest_shares, 5)),
        vote_share_p10=float(np.percentile(rest_shares, 10)),
        vote_share_p25=float(np.percentile(rest_shares, 25)),
        vote_share_p75=float(np.percentile(rest_shares, 75)),
        vote_share_p90=float(np.percentile(rest_shares, 90)),
        vote_share_p95=float(np.percentile(rest_shares, 95)),
        prob_above_4pct=float(np.mean(rest_shares >= 0.04)),
        prob_below_4pct=float(np.mean(rest_shares < 0.04)),
        prob_largest_vote_party=0.0,
        seats_mean=0.0,
        seats_median=0,
        seats_p05=0,
        seats_p10=0,
        seats_p25=0,
        seats_p75=0,
        seats_p90=0,
        seats_p95=0,
        prob_largest_seat_party=0.0,
        prob_any_seats=0.0,
        prob_local_12pct_exception_sub_4pct=0.0,
        seat_histogram={0: 1.0},
    )

    summary_obj = SimulationSummary(
        as_of=as_of,
        election_date=election_date,
        total_samples=n_samples,
        base_seed=manifest.get("base_seed", 0),
        parties=party_summaries,
        largest_vote_party_probabilities=largest_vote_probs,
        largest_seat_party_probabilities=largest_seat_probs,
        manifest=manifest,
    )

    helper = GroupSummaryHelper(seats_matrix=seats_matrix, party_indices=party_8_idx)
    return summary_obj, helper


class GroupSummaryHelper:
    """Helper class to quickly summarize arbitrary party groups on a completed simulation."""

    def __init__(self, seats_matrix: np.ndarray, party_indices: Mapping[str, int]):
        self.seats_matrix = seats_matrix
        self.party_indices = party_indices

    def summarize_group(
        self,
        parties: Sequence[str],
        majority_threshold: int = DEFAULT_MAJORITY_THRESHOLD,
    ) -> GroupSummary:
        cols = [self.party_indices[p] for p in parties if p in self.party_indices]
        group_seats = np.sum(self.seats_matrix[:, cols], axis=1)

        unique_seats, counts = np.unique(group_seats, return_counts=True)
        hist = {int(k): float(v / len(group_seats)) for k, v in zip(unique_seats, counts)}

        return GroupSummary(
            parties=tuple(parties),
            majority_threshold=majority_threshold,
            mean_seats=float(np.mean(group_seats)),
            median_seats=int(np.median(group_seats)),
            p05_seats=int(np.percentile(group_seats, 5)),
            p10_seats=int(np.percentile(group_seats, 10)),
            p25_seats=int(np.percentile(group_seats, 25)),
            p75_seats=int(np.percentile(group_seats, 75)),
            p90_seats=int(np.percentile(group_seats, 90)),
            p95_seats=int(np.percentile(group_seats, 95)),
            prob_majority=float(np.mean(group_seats >= majority_threshold)),
            seat_histogram=hist,
        )
