"""Configuration parameters for Empirical Pollster Precision Challenger (Experiment 2)."""

from datetime import date
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed" / "opinion_precision_challenger"
POP_TIMESERIES_FILE = DATA_DIR / "processed" / "pollofpolls" / "pollofpolls_timeseries.csv"
POLLS_FILE = DATA_DIR / "processed" / "pollofpolls" / "individual_polls.csv"
HISTORICAL_POLLS_FILE = DATA_DIR / "processed" / "pollofpolls" / "swedishpolls_individual_polls.csv"
ELECTIONS_FILE = DATA_DIR / "processed" / "elections" / "riksdag_election_results.csv"

# Parties & Categories
PARLIAMENTARY_PARTIES = ("M", "L", "C", "KD", "S", "V", "MP", "SD")
ALL_CATEGORIES_9 = ("M", "L", "C", "KD", "S", "V", "MP", "SD", "REST")
REFERENCE_CATEGORY = "REST"

# Evaluation Horizons & Origins
DEFAULT_HORIZONS = (7, 14, 28, 56, 84, 112)
START_ORIGIN_DATE = date(2014, 1, 1)
DEFAULT_ORIGIN_STEP_DAYS = 7

# Frozen OpinionState v1.1 Parameters
COVARIANCE_LOOKBACK_YEARS = 4
RECENT_POLL_LOOKBACK_DAYS = 60
RECENCY_HALF_LIFE_DAYS = 21.0
MAX_EFFECTIVE_POLLS = 8.0
SAMPLE_SIZE_BENCHMARK = 1000.0
MIN_SAMPLE_WEIGHT = 0.7
MAX_SAMPLE_WEIGHT = 1.5
COVARIANCE_DIAGONAL_SHRINKAGE = 0.10
MIN_POLLS_FOR_HOUSE_EFFECT = 20

# Precision Challenger Hyperparameters
M_MIN_HISTORY = 20          # Minimum historical polls required for house-specific variance
M0_PRIMARY = 10.0          # Primary empirical-Bayes prior sample weight
M0_SENSITIVITY = 25.0      # Sensitivity empirical-Bayes prior sample weight
Q_MIN = 0.5                # Minimum precision multiplier bound
Q_MAX = 2.0                # Maximum precision multiplier bound

# Predefined Decision Gate Thresholds (not independent preregistration)
GATE_MIN_RELATIVE_ES_IMPROVEMENT = 0.005   # >= +0.5% relative Energy Score improvement
GATE_MIN_RELATIVE_CRPS_IMPROVEMENT = 0.0  # No CRPS degradation
GATE_MIN_HORIZONS_WON = 4                  # At least 4 of 6 horizons
BOOTSTRAP_REPLICATIONS = 2000
BOOTSTRAP_BLOCK_MONTHS = 6
EVALUATION_DRAWS_COUNT = 1000
BASE_RANDOM_SEED = 12345
MIN_CASES_PER_HORIZON = 30

# Election Guardrail Thresholds (2018 & 2022)
GUARDRAIL_ELECTIONS = (date(2018, 9, 9), date(2022, 9, 11))
GUARDRAIL_HORIZONS = (7, 14, 28, 56, 84, 112)
GUARDRAIL_MAX_DEGRADATION = 0.01  # <= 1.0% maximum allowable degradation on election proper scores
