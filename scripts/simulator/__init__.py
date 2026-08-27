"""Swedish Riksdag Election Simulator v1 Package."""

from .config import (
    DEFAULT_ELECTION_DATE,
    DEFAULT_GEOGRAPHY_BASELINE_YEAR,
    DEFAULT_MAJORITY_THRESHOLD,
    DEFAULT_SIMULATION_SAMPLES,
    DEFAULT_SIMULATION_SEED,
    MODEL_PARTIES_9,
    PARLIAMENTARY_PARTIES_8,
)
from .engine import SimulationResult, simulate_election
from .fast_allocator import fast_allocate_seats_from_matrix
from .reproducibility import build_reproducibility_manifest
from .summary import GroupSummary, GroupSummaryHelper, PartySummary, SimulationSummary

__all__ = [
    "DEFAULT_ELECTION_DATE",
    "DEFAULT_GEOGRAPHY_BASELINE_YEAR",
    "DEFAULT_MAJORITY_THRESHOLD",
    "DEFAULT_SIMULATION_SAMPLES",
    "DEFAULT_SIMULATION_SEED",
    "MODEL_PARTIES_9",
    "PARLIAMENTARY_PARTIES_8",
    "SimulationResult",
    "simulate_election",
    "fast_allocate_seats_from_matrix",
    "build_reproducibility_manifest",
    "GroupSummary",
    "GroupSummaryHelper",
    "PartySummary",
    "SimulationSummary",
]
