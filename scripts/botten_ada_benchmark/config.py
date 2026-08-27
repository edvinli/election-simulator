"""Pinned benchmark configuration and the pre-registered adoption rule."""

from __future__ import annotations

from typing import Any

from scripts.simulator.config import PARLIAMENTARY_PARTIES_8

PARTY_ORDER: tuple[str, ...] = PARLIAMENTARY_PARTIES_8
HISTORICAL_HORIZONS: tuple[int, ...] = (112, 84, 56, 28, 14, 7)
HISTORICAL_ELECTION_DATES: dict[int, str] = {2018: "2018-09-09", 2022: "2022-09-11"}

# The repository and data page are primary Botten Ada sources.  The commit is
# pinned from the public repository's main branch on 2026-08-27.  The R/Stan
# model is intentionally not vendored or silently reimplemented here.
BOTTEN_ADA_SOURCE: dict[str, Any] = {
    "model_name": "Botten Ada",
    "repository_url": "https://github.com/MansMeg/ada_code",
    "repository_commit": "2dfe246b86c5cab517e4a0cb87fd57e5a9c62512",
    "official_data_url": "https://www.bottenada.se/data",
    "official_forecast_url": "https://www.bottenada.se/",
    "documentation_url": "https://www.bottenada.se/faq",
    "license": "CC BY-NC-SA 4.0 (official Botten Ada data/code page)",
    "pin_method": "git ls-remote HEAD; external bundle must record its own file SHA-256",
    "retrieved_date": "2026-08-27",
    "status": "EXTERNAL_BUNDLE_REQUIRED",
}

# Compatibility alias retained for any early local callers.
BOTten_ADA_SOURCE = BOTTEN_ADA_SOURCE

# This rule is deliberately conservative and is part of the benchmark input,
# not a result-derived recommendation.  "Material" requires an improvement of
# at least 0.10 score units and at least two of the three late horizons, with a
# separate threshold-Brier requirement.  The candidate remains frozen either
# way; this rule only determines whether to investigate one layer.
PIVOT_RULE: dict[str, Any] = {
    "rule_version": "1.0",
    "candidate_a_action_if_tied_or_better": "KEEP_CANDIDATE_A_UNCHANGED",
    "candidate_b_action_if_materially_better": "INVESTIGATE_TARGETED_LAYER_ONLY",
    "priority_horizons_days": [7, 14, 28],
    "minimum_late_horizon_wins": 2,
    "minimum_score_improvement": 0.10,
    "minimum_threshold_brier_improvement": 0.005,
    "tie_tolerance": 0.05,
    "required_metrics": [
        "vote_crps",
        "joint_vote_energy_score",
        "threshold_brier",
        "mean_vote_mae",
        "median_vote_mae",
        "coverage_and_width",
        "seat_crps",
        "joint_seat_energy_score",
    ],
    "non_adoption_note": "This rule does not authorize automatic retuning or wholesale replacement of Candidate A.",
}
