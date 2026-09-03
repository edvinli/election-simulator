"""Scoring utilities for the prospective 2026 ElectionSimulator comparison.

This package deliberately does not alter the historical Botten Ada harness.
The scoring functions here are provenance-agnostic: callers must provide
verified predictive draws or explicitly published quantiles/probabilities and
must keep that evidence decision outside this module.
"""

from .scoring import (
    DEFAULT_INTERVAL_LEVELS,
    PRIMARY_PARTY_ORDER,
    PROBABILISTIC_TIER_FAIR_DRAWS,
    PROBABILISTIC_TIER_POINT_MAE,
    PROBABILISTIC_TIER_WIS,
    WIS_CANDIDATE_INTERVAL_LEVELS,
    central_interval_metrics,
    common_wis_interval_levels,
    compatible_quantile_forecasts,
    crps_v_statistic,
    fair_crps,
    fair_energy_score,
    energy_score_v_statistic,
    interval_coverage_width,
    point_mae,
    score_forecast_pair,
    score_vote_ensemble,
    select_primary_scoring_tier,
    threshold_brier,
    threshold_brier_from_probability,
    threshold_probability,
    weighted_interval_score,
)

__all__ = [
    "DEFAULT_INTERVAL_LEVELS",
    "PRIMARY_PARTY_ORDER",
    "PROBABILISTIC_TIER_FAIR_DRAWS",
    "PROBABILISTIC_TIER_POINT_MAE",
    "PROBABILISTIC_TIER_WIS",
    "WIS_CANDIDATE_INTERVAL_LEVELS",
    "central_interval_metrics",
    "common_wis_interval_levels",
    "compatible_quantile_forecasts",
    "crps_v_statistic",
    "fair_crps",
    "fair_energy_score",
    "energy_score_v_statistic",
    "interval_coverage_width",
    "point_mae",
    "score_forecast_pair",
    "score_vote_ensemble",
    "select_primary_scoring_tier",
    "threshold_brier",
    "threshold_brier_from_probability",
    "threshold_probability",
    "weighted_interval_score",
]
