"""Seat uncertainty diagnostics and factual variance indicators for historical Riksdag seat hindcasts."""

from __future__ import annotations

from typing import Any
import numpy as np

from scripts.simulator.config import PARLIAMENTARY_PARTIES_8


def calculate_seat_uncertainty_diagnostics(
    vote_shares_matrix: np.ndarray,  # shape (N, 9) in percent
    seats_matrix: np.ndarray,        # shape (N, 8) in integer seats
    parties: tuple[str, ...] = PARLIAMENTARY_PARTIES_8,
) -> dict[str, dict[str, Any]]:
    """Compute factual seat uncertainty diagnostics and empirical variance indicators.

    Parameters:
        vote_shares_matrix: (N, 9) percentage vote share draws.
        seats_matrix: (N, 8) integer seat draws.
        parties: 8 parliamentary party codes.

    Returns:
        Dictionary mapping party -> diagnostic dictionary.
    """
    n_samples = len(seats_matrix)
    diagnostics = {}

    for idx, p in enumerate(parties):
        v = vote_shares_matrix[:, idx]
        s = seats_matrix[:, idx]

        v_mean = float(np.mean(v))
        v_std = float(np.std(v))
        s_mean = float(np.mean(s))
        s_med = int(np.median(s))
        s_std = float(np.std(s))

        # Threshold crossing and zero-seat probabilities
        p_qual = float(np.mean(v >= 4.0))
        p_zero = float(np.mean(s == 0))
        threshold_cliff_score = float(4.0 * p_qual * (1.0 - p_qual))

        # Conditional variance given qualification
        if p_qual > 0.01:
            s_qual = s[v >= 4.0]
            s_std_conditional = float(np.std(s_qual))
            bimodality_ratio = float(s_std / max(1e-6, s_std_conditional))
        else:
            s_std_conditional = 0.0
            bimodality_ratio = 1.0

        diagnostics[p] = {
            "vote_mean": round(v_mean, 2),
            "vote_std": round(v_std, 2),
            "seats_mean": round(s_mean, 2),
            "seats_median": s_med,
            "seats_std": round(s_std, 2),
            "prob_qualify_4pct": round(p_qual, 4),
            "prob_zero_seats": round(p_zero, 4),
            "threshold_cliff_score": round(threshold_cliff_score, 4),
            "seats_std_given_qualify": round(s_std_conditional, 2),
            "bimodality_ratio": round(bimodality_ratio, 2),
        }

    return diagnostics


# Backward compatibility alias
attribute_seat_uncertainty = calculate_seat_uncertainty_diagnostics
