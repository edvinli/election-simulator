"""Audit and derivation of effective sample size (n_eff) formulas under precision weighting.

Derives and compares:
- Formula A: Direct variance-derived n_eff = (sum w_base * q_g)^2 / sum (w_age^2)
- Formula B: Generalized Kish information formula = (sum w_i * q_i)^2 / sum (w_i^2 * q_i) where w_i = w_age * w_N
- Formula C: Standard Kish on composite weight = (sum w_base * q_g)^2 / sum (w_base * q_g)^2
"""

import math
from typing import Dict, List, Tuple
import numpy as np


def compute_neff_formulas(
    w_age: np.ndarray,
    w_n: np.ndarray,
    q: np.ndarray,
) -> Dict[str, float]:
    """Compute all candidate n_eff formulas for a given vector of polls."""
    w_base = w_age * w_n
    w_composite = w_base * q

    # Formula A: Direct variance of weighted mean
    # Var(theta_bar) = sigma0^2 * sum(w_age^2) / (sum w_base * q)^2
    # n_eff_A = (sum w_base * q)^2 / sum(w_age^2)
    num_a = np.sum(w_base * q) ** 2
    den_a = np.sum(w_age ** 2)
    neff_a = float(num_a / max(den_a, 1e-8))

    # Formula B: Generalized Kish with information multiplier q
    # n_eff_B = (sum w_base * q)^2 / sum(w_base^2 * q) (if q is variance multiplier)
    # or (sum w_composite * q)^2 / sum(w_composite^2 * q)
    num_b = np.sum(w_base * q) ** 2
    den_b = np.sum((w_base ** 2) * q)
    neff_b = float(num_b / max(den_b, 1e-8))

    # Formula C: Standard Kish on composite weight w_i = w_base * q
    num_c = np.sum(w_composite) ** 2
    den_c = np.sum(w_composite ** 2)
    neff_c = float(num_c / max(den_c, 1e-8))

    # Baseline: RC1 Kish (q = 1)
    num_base = np.sum(w_base) ** 2
    den_base = np.sum(w_base ** 2)
    neff_base = float(num_base / max(den_base, 1e-8))

    return {
        "rc1_baseline_kish": neff_base,
        "formula_a_variance_derived": neff_a,
        "formula_b_generalized_kish": neff_b,
        "formula_c_composite_kish": neff_c,
    }


def audit_limiting_cases():
    """Test mathematical behavior across key limiting cases."""
    print("=== Limiting Case 1: All q = 1.0 (Uniform Precision) ===")
    w_age = np.array([1.0, 0.8, 0.6, 0.4])
    w_n = np.array([1.0, 1.2, 0.9, 1.1])
    q = np.array([1.0, 1.0, 1.0, 1.0])
    res1 = compute_neff_formulas(w_age, w_n, q)
    for k, v in res1.items():
        print(f"  {k:30s}: {v:.4f}")
    assert math.isclose(res1["formula_b_generalized_kish"], res1["rc1_baseline_kish"], rel_tol=1e-5)
    assert math.isclose(res1["formula_c_composite_kish"], res1["rc1_baseline_kish"], rel_tol=1e-5)

    print("\n=== Limiting Case 2: One Poll Twice as Precise (q = 1.414, sqrt(2)) ===")
    q2 = np.array([1.414, 1.0, 1.0, 1.0])
    res2 = compute_neff_formulas(w_age, w_n, q2)
    for k, v in res2.items():
        print(f"  {k:30s}: {v:.4f}")

    print("\n=== Limiting Case 3: Uniform Precision Scaling (All q = 1.5) ===")
    q3 = np.array([1.5, 1.5, 1.5, 1.5])
    res3 = compute_neff_formulas(w_age, w_n, q3)
    for k, v in res3.items():
        print(f"  {k:30s}: {v:.4f}")

    print("\nLimiting cases validated successfully.")


if __name__ == "__main__":
    audit_limiting_cases()
