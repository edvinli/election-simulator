"""Forecast context and structural leakage boundaries for historical backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence
import numpy as np

from .state import OpinionState
from .transitions import HistoricalTransition, filter_transitions_as_of


@dataclass(frozen=True)
class ForecastContext:
    """Strict leakage-safe context provided to forecast models at an origin date.

    Contains:
        origin_date: The forecast date t.
        opinion_state: Optional OpinionState v1.1 for measurement-uncertainty models.
        origin_pop: The exact Poll of Polls composition at date t.
        origin_clr: The exact CLR vector of origin_pop.
        eligible_transitions_by_horizon: Dict mapping horizon_days -> already-filtered
                                         historical transitions (transition_end <= origin_date).
        transitions: Generic tuple of transitions for backwards compatibility.
        data_dir: Optional data directory path.
    """

    origin_date: date
    opinion_state: OpinionState | None = None
    origin_pop: dict[str, float] | None = None
    origin_clr: np.ndarray | None = None
    eligible_transitions_by_horizon: dict[int, tuple[HistoricalTransition, ...]] | None = None
    transitions: tuple[HistoricalTransition, ...] = ()
    data_dir: Path | None = None
