"""Configuration constants and hyperparameters for Opinion State Estimator v1."""

from __future__ import annotations

PARTIES: tuple[str, ...] = ("M", "L", "C", "KD", "S", "V", "MP", "SD")
REFERENCE_CATEGORY: str = "REST"
ALL_CATEGORIES: tuple[str, ...] = PARTIES + (REFERENCE_CATEGORY,)

# Numerical and compositional parameters
MIN_SHARE_PCT: float = 0.01
FLOATING_POINT_TOLERANCE: float = 1e-5

# Residual matching parameters
MAX_ESTIMATE_MATCH_LAG_DAYS: int = 3

# Covariance estimation parameters
COVARIANCE_LOOKBACK_YEARS: int = 4
MIN_RESIDUAL_POLLS: int = 100
MIN_POLLS_FOR_HOUSE_EFFECT: int = 20
COVARIANCE_DIAGONAL_SHRINKAGE: float = 0.0


# Recent polling and effective sample size parameters
RECENT_POLL_LOOKBACK_DAYS: int = 60
RECENCY_HALF_LIFE_DAYS: float = 21.0
SAMPLE_SIZE_BENCHMARK: float = 1000.0
MIN_SAMPLE_WEIGHT: float = 0.70
MAX_SAMPLE_WEIGHT: float = 1.50
MAX_EFFECTIVE_POLLS: float = 8.0

# Cholesky decomposition jitter search factors relative to mean diagonal variance
CHOLESKY_JITTER_FACTORS: tuple[float, ...] = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4)
