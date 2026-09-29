"""Configuration for the SwedishPolls causal poll aggregate."""

from __future__ import annotations

from datetime import date
from pathlib import Path

AGGREGATE_VERSION: str = "SwedishPollsAggregate-v0.1"

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLLS_FILE = (
    REPOSITORY_ROOT / "data" / "processed" / "pollofpolls" / "swedishpolls_individual_polls.csv"
)
DEFAULT_ELECTION_RESULTS_FILE = (
    REPOSITORY_ROOT / "data" / "processed" / "elections" / "riksdag_election_results.csv"
)
DEFAULT_2026_RESULT_MANIFEST = (
    REPOSITORY_ROOT / "data" / "raw" / "elections" / "val2026" / "official_result_manifest.json"
)
DEFAULT_POP_TIMESERIES_FILE = (
    REPOSITORY_ROOT / "data" / "processed" / "pollofpolls" / "pollofpolls_timeseries.csv"
)
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "data" / "processed" / "poll_aggregate"
TIMESERIES_FILENAME = "swedishpolls_aggregate_timeseries.csv"
METADATA_FILENAME = "swedishpolls_aggregate_metadata.json"
SOURCE_URL = "https://raw.githubusercontent.com/MansMeg/SwedishPolls/master/Data/Polls.csv"

PARTIES: tuple[str, ...] = ("M", "L", "C", "KD", "S", "V", "MP", "SD")

# The first anchor election. SwedishPolls reports all eight parties (SD
# included) consistently only from late 2006, so the filter starts here.
HISTORY_START_ELECTION: date = date(2006, 9, 17)

# When a historical final count is treated as known: election day plus seven
# days. The 2026 final count came six days after election day; historical
# publication dates are not recorded, so this is an assumption to verify. An
# election with a certified manifest uses that manifest's retrieval date.
HISTORICAL_RESULT_AVAILABILITY_LAG_DAYS: int = 7

# Named-party sums above this are implausible (the remainder is other parties).
MAX_NAMED_PARTY_SUM: float = 101.0

# Sample-size fallback when no earlier poll from any house has a reported size.
DEFAULT_SAMPLE_SIZE: int = 1000

# Diffuse prior before the first anchor election (percentage points).
INITIAL_STATE_SD_PP: float = 5.0

# Hyperparameters are fitted by maximum likelihood on log scale within these
# bounds. Each is dimensionless and multiplies the multinomial shape
# 1e4 * (diag(pi) - pi pi^T), so small parties move and err less in points.
#   process: daily latent-support variance
#   house_prior: variance of a pollster's house effect when first seen
#   house_drift: daily variance of house-effect drift
#   design_effect: poll variance relative to simple random sampling
#   election_n: effective sample size of an election as a reading of support
HYPERPARAMETER_NAMES: tuple[str, ...] = (
    "process",
    "house_prior",
    "house_drift",
    "design_effect",
    "election_n",
)
HYPERPARAMETER_START: dict[str, float] = {
    "process": 2e-6,
    "house_prior": 5e-4,
    "house_drift": 1e-8,
    "design_effect": 1.5,
    "election_n": 5000.0,
}
HYPERPARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "process": (1e-9, 1e-4),
    "house_prior": (1e-6, 1e-2),
    "house_drift": (1e-12, 1e-5),
    "design_effect": (0.5, 10.0),
    "election_n": (200.0, 1e6),
}
OPTIMIZER_MAX_ITERATIONS: int = 400
OPTIMIZER_TOLERANCE: float = 1e-4

# Filter checkpoints retained between consecutive as-of runs. A new
# information set that only adds recent observations restarts from the last
# checkpoint before them instead of from 2006.
CHECKPOINT_WINDOW: int = 96

OUTPUT_SHARE_DIGITS: int = 2
