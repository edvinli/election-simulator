"""Configuration and constants for SCB behavioral threshold diagnostic (Step 3)."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCB_PANEL_FILE = PROJECT_ROOT / "data" / "processed" / "scb_support_voting" / "scb_donor_recipient_panel.csv"
VOTE_INTENTION_FILE = PROJECT_ROOT / "data" / "processed" / "scb_support_voting" / "vote_by_sympathy.csv"
SECOND_CHOICE_FILE = PROJECT_ROOT / "data" / "processed" / "scb_support_voting" / "second_choice_by_sympathy.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "scb_behavioral_diagnostic"

# Key Parliamentary Parties and Focus Threshold Parties
PARLIAMENTARY_PARTIES = ["M", "C", "L", "KD", "MP", "S", "V", "SD"]
FOCUS_THRESHOLD_PARTIES = ["L", "KD", "MP", "C"]

# Pre-registered Linear Proximity Kernel (Primary: width = 2.0 pp, centered at 4.0%)
def kernel_linear_4pct(s: float) -> float:
    """Primary linear proximity kernel: K_4(s) = max(0, 1 - |s - 4.0| / 2.0).
    
    Properties:
      - Peaks at 1.0 when s = 4.0%
      - Equals 0.0 when s <= 2.0% or s >= 6.0%
      - Linearly ramps between 2% and 4%, and 4% and 6%
    """
    if pd_isna(s):
        return np.nan
    return float(max(0.0, 1.0 - abs(float(s) - 4.0) / 2.0))


# Placebo Linear Kernel (Centered at 7.0%, width = 2.0 pp)
def kernel_placebo_7pct(s: float) -> float:
    """Placebo linear proximity kernel: K_7(s) = max(0, 1 - |s - 7.0| / 2.0).
    
    Properties:
      - Peaks at 1.0 when s = 7.0%
      - Equals 0.0 when s <= 5.0% or s >= 9.0%
      - Identical 2.0 pp width to test whether near-4% effects exceed a non-threshold baseline
    """
    if pd_isna(s):
        return np.nan
    return float(max(0.0, 1.0 - abs(float(s) - 7.0) / 2.0))


# Gaussian Sensitivity Kernel (sigma = 1.0 pp)
def kernel_gaussian_4pct(s: float, sigma: float = 1.0) -> float:
    """Gaussian proximity kernel sensitivity: K_gauss(s) = exp(-0.5 * ((s - 4.0) / sigma)^2)."""
    if pd_isna(s):
        return np.nan
    return float(np.exp(-0.5 * ((float(s) - 4.0) / sigma) ** 2))


# Step Indicator Sensitivity Kernel (3.0% <= s <= 4.5%)
def kernel_step_4pct(s: float) -> float:
    """Step danger indicator sensitivity: I(3.0% <= s <= 4.5%)."""
    if pd_isna(s):
        return np.nan
    return 1.0 if (3.0 <= float(s) <= 4.5) else 0.0


def pd_isna(val: object) -> bool:
    """Check if value is NaN or None."""
    if val is None:
        return True
    try:
        return bool(np.isnan(float(val)))
    except (ValueError, TypeError):
        return False


# Election Cycles mapping for Leave-One-Cycle-Out sensitivity
ELECTION_CYCLES = {
    "cycle_2010_2014": ["2010M11", "2011M05", "2011M11", "2012M05", "2012M11", "2013M05", "2013M11", "2014M05"],
    "cycle_2014_2018": ["2014M11", "2015M05", "2015M11", "2016M05", "2016M11", "2017M05", "2017M11", "2018M05"],
    "cycle_2018_2022": ["2018M11", "2019M05", "2019M11", "2020M05", "2020M11", "2021M05", "2021M11", "2022M05"],
    "cycle_2022_2026": ["2022M11", "2023M05", "2024M05", "2025M05", "2026M05"],
}

# Number of bootstrap iterations
BOOTSTRAP_REPLICATIONS = 2000
BOOTSTRAP_RANDOM_SEED = 42

# Conversion ratio denominator floor (minimum A_jpt to compute R / A)
CONVERSION_RATIO_FLOOR_PCT = 2.0
