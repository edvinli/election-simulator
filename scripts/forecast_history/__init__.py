"""Offline coalition forecast history publication helpers."""

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
from .future_projection import (
    DEFAULT_PROJECTION_SAMPLES,
    build_future_projection,
    projection_tooltip_sv,
    update_history_with_production_result as update_history_with_future_projection,
    validate_future_projection_contract,
)
from .projection_simulator import simulate_conditional_projection


def __getattr__(name: str):
    from . import generate
    if hasattr(generate, name):
        return getattr(generate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_COALITIONS",
    "DEFAULT_ARCHIVE_DIR",
    "DEFAULT_HISTORY_OUTPUT",
    "DEFAULT_HISTORY_SAMPLES",
    "DEFAULT_PROJECTION_SAMPLES",
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
