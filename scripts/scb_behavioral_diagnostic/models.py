"""Statistical estimation engine for SCB behavioral threshold diagnostic.

Implements fixed-effects OLS, WLS uncertainty-weighting, deterministic wave-level
block bootstrap, placebo testing (7%), and sensitivity models.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from scripts.scb_behavioral_diagnostic.config import (
    BOOTSTRAP_RANDOM_SEED,
    BOOTSTRAP_REPLICATIONS,
    ELECTION_CYCLES,
    FOCUS_THRESHOLD_PARTIES,
    SCB_PANEL_FILE,
    kernel_gaussian_4pct,
    kernel_linear_4pct,
    kernel_placebo_7pct,
    kernel_step_4pct,
)


@dataclass(frozen=True)
class RegressionModelResult:
    """Results from a fixed-effects regression specification."""
    model_name: str
    model_category: str  # PRIMARY, PLACEBO, SENSITIVITY, DESCRIPTIVE_PARTY, LOO_CYCLE
    n_observations: int
    n_waves: int
    n_pairs: int
    r_squared: float
    coefficients: Dict[str, float]
    bootstrap_se: Dict[str, float]
    bootstrap_ci_lower: Dict[str, float]
    bootstrap_ci_upper: Dict[str, float]
    prob_alpha_positive: Optional[float]
    notes: str


def load_and_prepare_regression_data(
    panel_file: Path = SCB_PANEL_FILE,
) -> pd.DataFrame:
    """Load SCB donor-recipient panel and construct regression variables."""
    panel = pd.read_csv(panel_file)
    
    # 1. Retain only cross-party pairs (j != p)
    df = panel[panel["donor_party"] != panel["recipient_party"]].copy()
    
    # 2. Filter observed vote flow cells
    df = df[df["vote_value_status"] == "observed"].copy()
    df = df.dropna(subset=[
        "vote_estimate_pct",
        "second_choice_estimate_pct",
        "recipient_overall_sympathy_pct",
    ]).copy()
    
    # 3. Sort chronologically for lagging
    df["pair"] = df["donor_party"] + "_" + df["recipient_party"]
    df = df.sort_values(by=["pair", "survey_date"]).reset_index(drop=True)
    
    # 4. Construct primary regression variables (Partisympati-based)
    df["R"] = df["vote_estimate_pct"].astype(float)
    df["A"] = df["second_choice_estimate_pct"].astype(float)
    df["s_symp"] = df["recipient_overall_sympathy_pct"].astype(float)
    df["K4_symp"] = df["s_symp"].apply(kernel_linear_4pct)
    df["A_K4_symp"] = df["A"] * df["K4_symp"]
    
    # 5. Placebo kernel at 7.0%
    df["K7_symp"] = df["s_symp"].apply(kernel_placebo_7pct)
    df["A_K7_symp"] = df["A"] * df["K7_symp"]
    
    # 6. Vid10 sensitivity variables
    df["s_vid10"] = pd.to_numeric(df["recipient_overall_vote_pct"], errors="coerce")
    df["K4_vid10"] = df["s_vid10"].apply(kernel_linear_4pct)
    df["A_K4_vid10"] = df["A"] * df["K4_vid10"]
    
    # 7. Additional kernel sensitivities
    df["K4_gauss"] = df["s_symp"].apply(kernel_gaussian_4pct)
    df["A_K4_gauss"] = df["A"] * df["K4_gauss"]
    df["K4_step"] = df["s_symp"].apply(kernel_step_4pct)
    df["A_K4_step"] = df["A"] * df["K4_step"]
    
    # 8. Lagged second choice affinity A_{jp, t-1}
    df["A_lag"] = df.groupby("pair")["A"].shift(1)
    df["A_lag_K4_symp"] = df["A_lag"] * df["K4_symp"]
    
    # 9. Standard errors and WLS weights
    moe_clean = pd.to_numeric(df["vote_margin_error_pp"], errors="coerce").fillna(0.5)
    se_clean = np.maximum(moe_clean / 1.96, 0.1)
    df["wls_weight"] = 1.0 / (se_clean ** 2)
    
    return df


def fit_and_bootstrap_fe_model(
    df_sample: pd.DataFrame,
    x_columns: List[str],
    y_column: str = "R",
    weights_col: Optional[str] = None,
    n_replications: int = BOOTSTRAP_REPLICATIONS,
    random_seed: int = BOOTSTRAP_RANDOM_SEED,
) -> Tuple[Dict[str, float], float, int, Dict[str, float], Dict[str, float], Dict[str, float], Optional[float]]:
    """Fit two-way fixed effects (pair + wave) model and compute vectorized wave block bootstrap."""
    df_clean = df_sample.dropna(subset=x_columns + [y_column]).reset_index(drop=True)
    
    # Pre-build design matrices
    X_pairs = pd.get_dummies(df_clean["pair"], drop_first=True, dtype=float)
    X_waves = pd.get_dummies(df_clean["wave"], drop_first=True, dtype=float)
    X_main = df_clean[x_columns].astype(float)
    
    X_full = pd.concat([pd.Series(1.0, index=df_clean.index, name="intercept"), X_main, X_pairs, X_waves], axis=1).values
    y_full = df_clean[y_column].astype(float).values
    w_full = df_clean[weights_col].astype(float).values if weights_col else None
    
    # Point estimate
    if w_full is not None:
        sqrt_w = np.sqrt(w_full)
        X_fit = X_full * sqrt_w[:, np.newaxis]
        y_fit = y_full * sqrt_w
    else:
        X_fit = X_full
        y_fit = y_full
        
    beta_point, _, _, _ = np.linalg.lstsq(X_fit, y_fit, rcond=None)
    y_pred = X_full @ beta_point
    res = y_full - y_pred
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((y_full - np.mean(y_full)) ** 2)
    r2 = float(max(0.0, 1.0 - ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    point_coefs: Dict[str, float] = {}
    for idx, col in enumerate(x_columns):
        point_coefs[col] = float(beta_point[1 + idx])
        
    # Vectorized wave-level block bootstrap
    unique_waves = np.array(sorted(df_clean["wave"].unique()))
    n_unique_waves = len(unique_waves)
    wave_to_indices = {w: np.where(df_clean["wave"] == w)[0] for w in unique_waves}
    
    rng = np.random.default_rng(random_seed)
    boot_coefs_matrix = np.zeros((n_replications, len(x_columns)))
    
    for b in range(n_replications):
        sampled_waves = rng.choice(unique_waves, size=n_unique_waves, replace=True)
        idx_b = np.concatenate([wave_to_indices[w] for w in sampled_waves])
        
        X_b = X_full[idx_b]
        y_b = y_full[idx_b]
        
        if w_full is not None:
            w_b = np.sqrt(w_full[idx_b])
            X_b_fit = X_b * w_b[:, np.newaxis]
            y_b_fit = y_b * w_b
        else:
            X_b_fit = X_b
            y_b_fit = y_b
            
        try:
            b_beta, _, _, _ = np.linalg.lstsq(X_b_fit, y_b_fit, rcond=None)
            for idx in range(len(x_columns)):
                boot_coefs_matrix[b, idx] = b_beta[1 + idx]
        except Exception:
            for idx in range(len(x_columns)):
                boot_coefs_matrix[b, idx] = np.nan
                
    se_dict: Dict[str, float] = {}
    ci_lower: Dict[str, float] = {}
    ci_upper: Dict[str, float] = {}
    prob_pos: Optional[float] = None
    
    for idx, col in enumerate(x_columns):
        vals = boot_coefs_matrix[:, idx]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            se_dict[col] = float(np.std(vals))
            ci_lower[col] = float(np.percentile(vals, 2.5))
            ci_upper[col] = float(np.percentile(vals, 97.5))
            if "K4" in col or "K7" in col:
                prob_pos = float(np.mean(vals > 0))
        else:
            se_dict[col] = np.nan
            ci_lower[col] = np.nan
            ci_upper[col] = np.nan
            
    return point_coefs, r2, len(y_full), se_dict, ci_lower, ci_upper, prob_pos


def compute_paired_placebo_difference_bootstrap(
    df_sample: pd.DataFrame,
    n_replications: int = BOOTSTRAP_REPLICATIONS,
    random_seed: int = BOOTSTRAP_RANDOM_SEED,
) -> Dict[str, Any]:
    """Compute paired wave-level block bootstrap for difference delta_alpha = (alpha_4 - alpha_7)."""
    df_clean = df_sample.dropna(subset=["A", "K4_symp", "A_K4_symp", "K7_symp", "A_K7_symp", "R"]).reset_index(drop=True)
    
    x_prim = ["A", "K4_symp", "A_K4_symp"]
    x_p7 = ["A", "K7_symp", "A_K7_symp"]
    
    X_pairs = pd.get_dummies(df_clean["pair"], drop_first=True, dtype=float)
    X_waves = pd.get_dummies(df_clean["wave"], drop_first=True, dtype=float)
    
    X_prim_full = pd.concat([pd.Series(1.0, index=df_clean.index, name="intercept"), df_clean[x_prim], X_pairs, X_waves], axis=1).values
    X_p7_full = pd.concat([pd.Series(1.0, index=df_clean.index, name="intercept"), df_clean[x_p7], X_pairs, X_waves], axis=1).values
    y_full = df_clean["R"].astype(float).values
    
    # Point estimates
    b_prim_pt = np.linalg.lstsq(X_prim_full, y_full, rcond=None)[0]
    b_p7_pt = np.linalg.lstsq(X_p7_full, y_full, rcond=None)[0]
    point_diff = float(b_prim_pt[3] - b_p7_pt[3])
    
    # Paired block bootstrap over waves
    unique_waves = np.array(sorted(df_clean["wave"].unique()))
    n_unique_waves = len(unique_waves)
    wave_to_indices = {w: np.where(df_clean["wave"] == w)[0] for w in unique_waves}
    
    rng = np.random.default_rng(random_seed)
    diff_alphas = np.zeros(n_replications)
    
    for b in range(n_replications):
        sampled_waves = rng.choice(unique_waves, size=n_unique_waves, replace=True)
        idx_b = np.concatenate([wave_to_indices[w] for w in sampled_waves])
        
        try:
            b_beta_prim = np.linalg.lstsq(X_prim_full[idx_b], y_full[idx_b], rcond=None)[0]
            b_beta_p7 = np.linalg.lstsq(X_p7_full[idx_b], y_full[idx_b], rcond=None)[0]
            diff_alphas[b] = b_beta_prim[3] - b_beta_p7[3]
        except Exception:
            diff_alphas[b] = np.nan
            
    valid_diffs = diff_alphas[~np.isnan(diff_alphas)]
    return {
        "point_difference_alpha4_minus_alpha7": round(point_diff, 5),
        "paired_bootstrap_se": round(float(np.std(valid_diffs)), 5),
        "paired_bootstrap_ci_95": [round(float(np.percentile(valid_diffs, 2.5)), 5), round(float(np.percentile(valid_diffs, 97.5)), 5)],
        "prob_alpha4_greater_than_alpha7": round(float(np.mean(valid_diffs > 0)), 4),
    }


def evaluate_all_specifications(
    df: Optional[pd.DataFrame] = None,
    n_bootstrap_replications: int = BOOTSTRAP_REPLICATIONS,
) -> List[RegressionModelResult]:
    """Estimate all primary, placebo, and sensitivity regression specifications."""
    if df is None:
        df = load_and_prepare_regression_data()
        
    results: List[RegressionModelResult] = []
    
    # 1. PRIMARY MODEL: Partisympati, Linear K4 (width 2pp, center 4%)
    x_prim = ["A", "K4_symp", "A_K4_symp"]
    c_prim, r2_prim, n_prim, se_prim, cil_prim, ciu_prim, ppos_prim = fit_and_bootstrap_fe_model(
        df, x_prim, n_replications=n_bootstrap_replications
    )
    results.append(
        RegressionModelResult(
            model_name="Primary: Partisympati + Linear K4 (OLS)",
            model_category="PRIMARY",
            n_observations=n_prim,
            n_waves=df["wave"].nunique(),
            n_pairs=df["pair"].nunique(),
            r_squared=r2_prim,
            coefficients=c_prim,
            bootstrap_se=se_prim,
            bootstrap_ci_lower=cil_prim,
            bootstrap_ci_upper=ciu_prim,
            prob_alpha_positive=ppos_prim,
            notes="Preregistered primary model with pair FE and wave FE.",
        )
    )
    
    # 2. PLACEBO MODEL: Partisympati, Linear K7 (width 2pp, center 7%)
    x_p7 = ["A", "K7_symp", "A_K7_symp"]
    c_p7, r2_p7, n_p7, se_p7, cil_p7, ciu_p7, ppos_p7 = fit_and_bootstrap_fe_model(
        df, x_p7, n_replications=n_bootstrap_replications
    )
    results.append(
        RegressionModelResult(
            model_name="Placebo: Partisympati + Linear K7 (OLS)",
            model_category="PLACEBO",
            n_observations=n_p7,
            n_waves=df["wave"].nunique(),
            n_pairs=df["pair"].nunique(),
            r_squared=r2_p7,
            coefficients=c_p7,
            bootstrap_se=se_p7,
            bootstrap_ci_lower=cil_p7,
            bootstrap_ci_upper=ciu_p7,
            prob_alpha_positive=ppos_p7,
            notes="Placebo test centered at 7.0% (away from 4% threshold).",
        )
    )
    
    # 3. SENSITIVITY: WLS Uncertainty-Weighted Regression
    c_wls, r2_wls, n_wls, se_wls, cil_wls, ciu_wls, ppos_wls = fit_and_bootstrap_fe_model(
        df, x_prim, weights_col="wls_weight", n_replications=n_bootstrap_replications
    )
    results.append(
        RegressionModelResult(
            model_name="Sensitivity: WLS Uncertainty-Weighted",
            model_category="SENSITIVITY",
            n_observations=n_wls,
            n_waves=df["wave"].nunique(),
            n_pairs=df["pair"].nunique(),
            r_squared=r2_wls,
            coefficients=c_wls,
            bootstrap_se=se_wls,
            bootstrap_ci_lower=cil_wls,
            bootstrap_ci_upper=ciu_wls,
            prob_alpha_positive=ppos_wls,
            notes="Weighted by inverse SCB margin-of-error variance (1/SE^2).",
        )
    )
    
    # 4. SENSITIVITY: Vid10 Overall Vote Intention State Variable
    df_vid = df.dropna(subset=["s_vid10"]).copy()
    x_vid = ["A", "K4_vid10", "A_K4_vid10"]
    c_vid, r2_vid, n_vid, se_vid, cil_vid, ciu_vid, ppos_vid = fit_and_bootstrap_fe_model(
        df_vid, x_vid, n_replications=n_bootstrap_replications
    )
    results.append(
        RegressionModelResult(
            model_name="Sensitivity: Vid10 Overall Vote Intention",
            model_category="SENSITIVITY",
            n_observations=n_vid,
            n_waves=df_vid["wave"].nunique(),
            n_pairs=df_vid["pair"].nunique(),
            r_squared=r2_vid,
            coefficients=c_vid,
            bootstrap_se=se_vid,
            bootstrap_ci_lower=cil_vid,
            bootstrap_ci_upper=ciu_vid,
            prob_alpha_positive=ppos_vid,
            notes="Uses Vid10 vote intention as threshold state variable.",
        )
    )
    
    # 5. SENSITIVITY: Lagged Second Choice Affinity A_{t-1}
    df_lag = df.dropna(subset=["A_lag"]).copy()
    x_lag = ["A_lag", "K4_symp", "A_lag_K4_symp"]
    c_lag, r2_lag, n_lag, se_lag, cil_lag, ciu_lag, ppos_lag = fit_and_bootstrap_fe_model(
        df_lag, x_lag, n_replications=n_bootstrap_replications
    )
    results.append(
        RegressionModelResult(
            model_name="Sensitivity: Lagged Affinity A(t-1)",
            model_category="SENSITIVITY",
            n_observations=n_lag,
            n_waves=df_lag["wave"].nunique(),
            n_pairs=df_lag["pair"].nunique(),
            r_squared=r2_lag,
            coefficients=c_lag,
            bootstrap_se=se_lag,
            bootstrap_ci_lower=cil_lag,
            bootstrap_ci_upper=ciu_lag,
            prob_alpha_positive=ppos_lag,
            notes="Uses lagged second-choice affinity to reduce same-wave survey noise.",
        )
    )
    
    # 6. SENSITIVITY: Gaussian Proximity Kernel (sigma = 1.0 pp)
    x_gauss = ["A", "K4_gauss", "A_K4_gauss"]
    c_gauss, r2_gauss, n_gauss, se_gauss, cil_gauss, ciu_gauss, ppos_gauss = fit_and_bootstrap_fe_model(
        df, x_gauss, n_replications=n_bootstrap_replications
    )
    results.append(
        RegressionModelResult(
            model_name="Sensitivity: Gaussian Kernel (sigma=1.0pp)",
            model_category="SENSITIVITY",
            n_observations=n_gauss,
            n_waves=df["wave"].nunique(),
            n_pairs=df["pair"].nunique(),
            r_squared=r2_gauss,
            coefficients=c_gauss,
            bootstrap_se=se_gauss,
            bootstrap_ci_lower=cil_gauss,
            bootstrap_ci_upper=ciu_gauss,
            prob_alpha_positive=ppos_gauss,
            notes="Gaussian proximity kernel sensitivity.",
        )
    )
    
    # 7. SENSITIVITY: Step Indicator Kernel (3.0% <= s <= 4.5%)
    x_step = ["A", "K4_step", "A_K4_step"]
    c_step, r2_step, n_step, se_step, cil_step, ciu_step, ppos_step = fit_and_bootstrap_fe_model(
        df, x_step, n_replications=n_bootstrap_replications
    )
    results.append(
        RegressionModelResult(
            model_name="Sensitivity: Step Danger Indicator [3.0%, 4.5%]",
            model_category="SENSITIVITY",
            n_observations=n_step,
            n_waves=df["wave"].nunique(),
            n_pairs=df["pair"].nunique(),
            r_squared=r2_step,
            coefficients=c_step,
            bootstrap_se=se_step,
            bootstrap_ci_lower=cil_step,
            bootstrap_ci_upper=ciu_step,
            prob_alpha_positive=ppos_step,
            notes="Step indicator danger region sensitivity.",
        )
    )
    
    # 8. LEAVE-ONE-ELECTION-CYCLE-OUT ROBUSTNESS
    for cycle_name, cycle_waves in sorted(ELECTION_CYCLES.items()):
        df_loo = df[~df["wave"].isin(cycle_waves)].copy()
        c_loo, r2_loo, n_loo, se_loo, cil_loo, ciu_loo, ppos_loo = fit_and_bootstrap_fe_model(
            df_loo, x_prim, n_replications=n_bootstrap_replications
        )
        results.append(
            RegressionModelResult(
                model_name=f"LOO Cycle: Exclude {cycle_name}",
                model_category="LOO_CYCLE",
                n_observations=n_loo,
                n_waves=df_loo["wave"].nunique(),
                n_pairs=df_loo["pair"].nunique(),
                r_squared=r2_loo,
                coefficients=c_loo,
                bootstrap_se=se_loo,
                bootstrap_ci_lower=cil_loo,
                bootstrap_ci_upper=ciu_loo,
                prob_alpha_positive=ppos_loo,
                notes=f"Primary model excluding election cycle {cycle_name}.",
            )
        )
        
    # 9. RECIPIENT-SPECIFIC DESCRIPTIVE COEFFICIENTS (L, KD, MP, C)
    for p in FOCUS_THRESHOLD_PARTIES:
        df_p = df[df["recipient_party"] == p].copy()
        c_p, r2_p, n_p, se_p, cil_p, ciu_p, ppos_p = fit_and_bootstrap_fe_model(
            df_p, x_prim, n_replications=n_bootstrap_replications
        )
        results.append(
            RegressionModelResult(
                model_name=f"Descriptive Party: Recipient {p}",
                model_category="DESCRIPTIVE_PARTY",
                n_observations=n_p,
                n_waves=df_p["wave"].nunique(),
                n_pairs=df_p["pair"].nunique(),
                r_squared=r2_p,
                coefficients=c_p,
                bootstrap_se=se_p,
                bootstrap_ci_lower=cil_p,
                bootstrap_ci_upper=ciu_p,
                prob_alpha_positive=ppos_p,
                notes=f"Descriptive sub-sample regression for recipient {p} only.",
            )
        )
        
    return results


def export_regression_results_table(
    results: List[RegressionModelResult],
    output_file: Path,
) -> pd.DataFrame:
    """Format regression results into a structured summary dataframe and save to CSV."""
    rows: List[Dict[str, Any]] = []
    
    for res in results:
        param_names = list(res.coefficients.keys())
        a_param = param_names[0]
        k_param = param_names[1]
        inter_param = param_names[2]
        
        rows.append({
            "model_name": res.model_name,
            "model_category": res.model_category,
            "n_obs": res.n_observations,
            "n_waves": res.n_waves,
            "n_pairs": res.n_pairs,
            "r_squared": round(res.r_squared, 4),
            "theta_affinity": round(res.coefficients[a_param], 5),
            "theta_se": round(res.bootstrap_se[a_param], 5),
            "theta_ci_95": f"[{res.bootstrap_ci_lower[a_param]:.4f}, {res.bootstrap_ci_upper[a_param]:.4f}]",
            "delta_threshold": round(res.coefficients[k_param], 5),
            "delta_se": round(res.bootstrap_se[k_param], 5),
            "delta_ci_95": f"[{res.bootstrap_ci_lower[k_param]:.4f}, {res.bootstrap_ci_upper[k_param]:.4f}]",
            "alpha_interaction": round(res.coefficients[inter_param], 5),
            "alpha_se": round(res.bootstrap_se[inter_param], 5),
            "alpha_ci_95": f"[{res.bootstrap_ci_lower[inter_param]:.4f}, {res.bootstrap_ci_upper[inter_param]:.4f}]",
            "prob_alpha_positive": round(res.prob_alpha_positive, 4) if res.prob_alpha_positive is not None else None,
            "notes": res.notes,
        })
        
    df_out = pd.DataFrame(rows)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_file, index=False, encoding="utf-8")
    return df_out
