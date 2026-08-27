"""Acquire, normalize, and validate public Pollofpolls data, and estimate opinion states."""

from typing import TYPE_CHECKING

from .normalize import normalize_party, parse_date, parse_percentage

if TYPE_CHECKING:
    from .state import OpinionState, estimate_opinion
    from .state_math import alr_to_composition, composition_to_alr


def estimate_opinion(*args, **kwargs):
    from .state import estimate_opinion as _estimate_opinion

    return _estimate_opinion(*args, **kwargs)


def __getattr__(name: str):
    if name == "OpinionState":
        from .state import OpinionState

        return OpinionState
    if name == "composition_to_alr":
        from .state_math import composition_to_alr

        return composition_to_alr
    if name == "alr_to_composition":
        from .state_math import alr_to_composition

        return alr_to_composition
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "OpinionState",
    "alr_to_composition",
    "composition_to_alr",
    "estimate_opinion",
    "normalize_party",
    "parse_date",
    "parse_percentage",
]


