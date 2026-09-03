"""Offline forecast history publication helpers (coalitions and parties)."""

from importlib import import_module

from .contract import (
    DEFAULT_COALITIONS,
    HISTORY_DYNAMICS_CAP_DAYS,
    HISTORY_PARTY_ORDER,
    HISTORY_SCHEMA_VERSION,
    build_groups_from_matrices,
    coalition_seat_draws,
    coalition_vote_draws,
    deterministic_history_sha256,
    summarize_coalition_draws,
    validate_history_contract,
    write_history_json,
)


_FUTURE_PROJECTION_EXPORTS = {
    "DEFAULT_PROJECTION_SAMPLES",
    "ELECTION_NOISE_RNG_POLICY",
    "build_future_projection",
    "election_day_label_sv",
    "projection_tooltip_sv",
    "update_history_with_future_projection",
    "validate_future_projection_contract",
}
_PROJECTION_SIMULATOR_EXPORTS = {"simulate_conditional_projection"}
_CAMPAIGN_PATH_EXPORTS = {
    "CAMPAIGN_PATH_MODEL_ID",
    "DEFAULT_REPRESENTATIVE_PATHS",
    "build_campaign_path_pool",
    "campaign_paths_tooltip_sv",
    "election_day_tooltip_sv",
    "resolve_endpoint_horizon",
    "simulate_campaign_paths",
}
_PARTY_CONTRACT_EXPORTS = {
    "NATIONAL_THRESHOLD_PCT",
    "PARTY_DEFINITION_ORDER",
    "PARTY_VIEW_SCHEMA_VERSION",
    "PARTY_VOTE_DENOMINATOR",
    "assert_election_day_party_parity",
    "build_parties_from_matrices",
    "build_party_vote_quantiles",
    "parties_view_metadata",
    "party_point_from_archive_record",
    "party_seat_draws",
    "party_vote_draws",
    "validate_parties_view",
    "validate_party_summaries",
    "validate_party_vote_only",
}
_CAMPAIGN_PATH_CONTRACT_EXPORTS = {
    "PRIMARY_ROLE",
    "SECONDARY_ROLE",
    "SECONDARY_DESCRIPTION_SV",
    "build_future_campaign_paths",
    "mark_secondary_projection",
    "validate_future_campaign_paths_contract",
    "validate_secondary_projection_role",
}


def __getattr__(name: str):
    if name in _FUTURE_PROJECTION_EXPORTS:
        module = import_module("scripts.forecast_history.future_projection")
        if name == "update_history_with_future_projection":
            return module.update_history_with_production_result
        return getattr(module, name)
    if name in _PROJECTION_SIMULATOR_EXPORTS:
        module = import_module("scripts.forecast_history.projection_simulator")
        return getattr(module, name)
    if name in _CAMPAIGN_PATH_EXPORTS:
        return getattr(import_module("scripts.forecast_history.campaign_paths"), name)
    if name in _PARTY_CONTRACT_EXPORTS:
        return getattr(import_module("scripts.forecast_history.party_contract"), name)
    if name in _CAMPAIGN_PATH_CONTRACT_EXPORTS:
        return getattr(import_module("scripts.forecast_history.campaign_paths_contract"), name)
    module = import_module("scripts.forecast_history.generate")
    if hasattr(module, name):
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CAMPAIGN_PATH_MODEL_ID",
    "NATIONAL_THRESHOLD_PCT",
    "PARTY_DEFINITION_ORDER",
    "PARTY_VIEW_SCHEMA_VERSION",
    "PARTY_VOTE_DENOMINATOR",
    "assert_election_day_party_parity",
    "build_parties_from_matrices",
    "build_party_vote_quantiles",
    "parties_view_metadata",
    "party_point_from_archive_record",
    "party_seat_draws",
    "party_vote_draws",
    "validate_parties_view",
    "validate_party_summaries",
    "validate_party_vote_only",
    "DEFAULT_COALITIONS",
    "DEFAULT_REPRESENTATIVE_PATHS",
    "PRIMARY_ROLE",
    "SECONDARY_DESCRIPTION_SV",
    "SECONDARY_ROLE",
    "build_campaign_path_pool",
    "build_future_campaign_paths",
    "campaign_paths_tooltip_sv",
    "election_day_tooltip_sv",
    "mark_secondary_projection",
    "resolve_endpoint_horizon",
    "simulate_campaign_paths",
    "validate_future_campaign_paths_contract",
    "validate_secondary_projection_role",
    "DEFAULT_ARCHIVE_DIR",
    "DEFAULT_HISTORY_OUTPUT",
    "DEFAULT_HISTORY_SAMPLES",
    "DEFAULT_PROJECTION_SAMPLES",
    "ELECTION_NOISE_RNG_POLICY",
    "HISTORY_CAP_DATE",
    "HISTORY_DYNAMICS_CAP_DAYS",
    "HISTORY_PARTY_ORDER",
    "HISTORY_SCHEMA_VERSION",
    "HISTORY_START_DATE",
    "build_future_projection",
    "build_groups_from_matrices",
    "build_history",
    "update_history_with_production_result",
    "update_history_with_future_projection",
    "build_history_dates",
    "coalition_seat_draws",
    "coalition_vote_draws",
    "deterministic_history_sha256",
    "election_day_label_sv",
    "generate_history",
    "generate_history_artifact",
    "filter_swedishpolls_as_of",
    "filter_swedishpolls_period",
    "projection_tooltip_sv",
    "serialize_swedishpolls",
    "simulate_conditional_projection",
    "summarize_coalition_draws",
    "validate_future_projection_contract",
    "validate_history_contract",
    "write_history_json",
]
