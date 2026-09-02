"""Offline coalition forecast history publication helpers."""

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


def __getattr__(name: str):
    if name in _FUTURE_PROJECTION_EXPORTS:
        module = import_module("scripts.forecast_history.future_projection")
        if name == "update_history_with_future_projection":
            return module.update_history_with_production_result
        return getattr(module, name)
    if name in _PROJECTION_SIMULATOR_EXPORTS:
        module = import_module("scripts.forecast_history.projection_simulator")
        return getattr(module, name)
    module = import_module("scripts.forecast_history.generate")
    if hasattr(module, name):
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_COALITIONS",
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
