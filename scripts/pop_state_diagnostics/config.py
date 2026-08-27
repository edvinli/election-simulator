"""Configuration constants and paths for Step 4A PoP State-Dependence Diagnostic."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Tuple

# Directories and Data Paths
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR: Path = REPO_ROOT / "data" / "processed" / "pop_state_diagnostics"
POP_TIMESERIES_FILE: Path = REPO_ROOT / "data" / "processed" / "pollofpolls" / "pollofpolls_timeseries.csv"

# Model 9-party composition vector
MODEL_PARTIES_8: Tuple[str, ...] = ("M", "L", "C", "KD", "S", "V", "MP", "SD")
REFERENCE_CATEGORY: str = "REST"
ALL_CATEGORIES_9: Tuple[str, ...] = MODEL_PARTIES_8 + (REFERENCE_CATEGORY,)

# Forecast Horizons & Backtest Setup
DEFAULT_HORIZONS: Tuple[int, ...] = (7, 14, 28, 56, 84, 112)
DEFAULT_ORIGIN_STEP_DAYS: int = 7
START_ORIGIN_DATE: date = date(2014, 1, 1)
MIN_CANDIDATE_TRANSITIONS: int = 50

# Nearest-Neighbor Hyperparameters
PRIMARY_K_NEIGHBORS: int = 50
SENSITIVITY_K_NEIGHBORS: Tuple[int, ...] = (25, 100)

# Monte Carlo Simulation Parameters
EVALUATION_DRAWS_COUNT: int = 1_000
BASE_RANDOM_SEED: int = 42
BOOTSTRAP_REPLICATIONS: int = 2_000

# Calendar-Block Bootstrap
BLOCK_LENGTH_MONTHS: int = 6
SENSITIVITY_BLOCK_LENGTH_MONTHS: int = 12

# Threshold Starting Bins (Half-open intervals [low, high))
THRESHOLD_STARTING_BINS: Tuple[Tuple[float, float, str], ...] = (
    (2.0, 3.0, "[2.0, 3.0)"),
    (3.0, 3.5, "[3.0, 3.5)"),
    (3.5, 4.0, "[3.5, 4.0)"),
    (4.0, 4.5, "[4.0, 4.5)"),
    (4.5, 5.0, "[4.5, 5.0)"),
    (5.0, 6.0, "[5.0, 6.0)"),
)

# Step 4B Decision Gate Thresholds
GATE_MIN_ENERGY_SCORE_IMPROVEMENT: float = 0.005
GATE_MAX_CRPS_DEGRADATION: float = 0.001
GATE_MIN_HORIZONS_BEATING_V2: int = 4
