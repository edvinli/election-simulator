# Empirical Pollster Precision Challenger Evidence Report (Experiment 2)

This report presents the empirical findings and predefined decision-gate results for the **Empirical Pollster Precision Challenger (OpinionState v1.2-candidate)** evaluated against **OpinionState v1.1 (RC1 Baseline)**. The gate was specified in the experiment configuration; this repository does not claim independent preregistration unless an immutable pre-result reference is supplied.

---

## 1. Executive Summary & Final Decision

```text
==================================================================================================
PRE-LAUNCH CHALLENGER SUITE FINAL DECISION:
• Experiment 0 & 1 (Industry Bias):            ALREADY_REJECTED (docs/industry_bias_audit.md)
• Experiment 2 (Empirical Pollster Precision): REJECTED_KEEP_RC1
==================================================================================================
FINAL MODEL STATUS: ELECTION SIMULATOR v1.0-RC1 RETAINED; CANDIDATE NOT ADOPTED
==================================================================================================
```

### Key Statistical Results (3,695 Canonical Rolling Cases across 622 Weekly Origins, 2014–2026)
* **Energy Score (9-Party Simplex)**:
  * **Arm A (RC1 Baseline)**: `1.32117`
  * **Arm B (Equal-Weight Control)**: `1.32103`
  * **Arm C (Precision Challenger, $M_0=10$)**: `1.32040` (Relative improvement: **$+0.059\%$**, 95% block bootstrap CI: `[-0.018%, +0.148%]`)
  * **Sensitivity Arm C25 ($M_0=25$)**: `1.32127` (Degraded vs RC1 baseline `1.32117`)
* **Marginal CRPS (8 Parliamentary Parties)**:
  * **Arm A (RC1 Baseline)**: `0.34692`
  * **Arm C (Precision Challenger)**: `0.34673` (Relative improvement: **$+0.054\%$**, 95% block bootstrap CI: `[-0.006%, +0.126%]`)
* **Reference Invariance Hard Test**: **PASSED** (Max $q_g$ difference across ALR reference bases = `0.00000000`).
* **Predefined Adoption Gate**: **FAILED** (Materiality threshold $+0.50\%$ not met; block bootstrap 95% CI spans zero; sensitivity with $M_0=25$ inverts and degrades).

---

## 2. Experimental Methodology & Rigorous Formulation

### A. Reference-Invariant CLR Dispersion & $N$-Deconfounding
To prevent reference-base dependence and sample-size confounding:
1. For each historical poll $j$ within the trailing 4-year window $[o - 4\text{y}, o]$, construct a contemporaneous CLR reference from eligible polls by **other polling houses** at the same reference date (or the nearest prior date within the three-day matching window):
   $$\mathbf{r}_j^{\text{CLR}} = \text{clr}(\text{poll}_j) - \text{clr}(\text{LOO\text{-}reference}_{-g(j),\text{ref}(j)})$$
   $$\mathbf{h}_g^{\text{CLR}} = \frac{1}{M_g} \sum_{j \in g} \mathbf{r}_j^{\text{CLR}} \quad (\text{for houses with } M_g \ge 20)$$
   $$\mathbf{u}_j^{\text{CLR}} = \mathbf{r}_j^{\text{CLR}} - \mathbf{h}_g^{\text{CLR}}$$
   $$D_j = \frac{1}{9} \sum_{p=1}^9 (u_{j, p}^{\text{CLR}})^2$$
2. $N$-standardized dispersion removes sample size confounding:
   $$w_{N, j} = \text{clip}\left(\sqrt{\frac{N_j}{1000}}, 0.7, 1.5\right) \quad (\text{default 1.0 if missing})$$
   $$D_j^{\text{adj}} = D_j \cdot w_{N, j}^2$$
3. Empirical-Bayes shrinkage toward pooled historical dispersion:
   $$s_g^2 = \frac{1}{M_g} \sum_{j \in g} D_j^{\text{adj}}, \quad s_{\text{pool}}^2 = \frac{1}{M_{\text{total}}} \sum D_j^{\text{adj}}$$
   $$s_{g, \text{shrunk}}^2 = \frac{M_g}{M_g + M_0} s_g^2 + \frac{M_0}{M_g + M_0} s_{\text{pool}}^2$$
4. Standard-deviation precision multiplier:
   $$q_g = \text{clip}\left(\frac{s_{\text{pool}} / s_{g, \text{shrunk}}}{\text{Mean}(s_{\text{pool}} / s_{g, \text{shrunk}})}, 0.5, 2.0\right) \quad (\text{default } 1.0 \text{ if } M_g < 20)$$

### B. Coherent Information-Equivalent Effective Poll Count ($n_{\text{eff}}^{\text{precision}}$)
Under precision-weighted aggregation $w_i = w_{\text{age}, i} w_{N, i} q_{g(i)}$:
$$n_{\text{eff}}^{\text{precision}} = \frac{\left(\sum_i w_{\text{base}, i} q_{g(i)}\right)^2}{\sum_i \left(w_{\text{base}, i} q_{g(i)}\right)^2}, \quad \text{where } w_{\text{base}, i} = w_{\text{age}, i} w_{N, i}$$
*(This is Kish's effective sample size applied to the weights actually used by the precision arm, and reduces identically to standard Kish when all $q_g = 1.0$.)*

---

## 3. Horizon-by-Horizon Performance Summary

| Horizon | Cases | Arm A (RC1 ES) | Arm B (Equal ES) | Arm C (Precision ES) | Rel ES Imp (%) | Arm A CRPS | Arm C CRPS |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **7 days** | 622 | 0.40824 | 0.40780 | **0.40725** | +0.241% | 0.10532 | **0.10509** |
| **14 days** | 621 | 0.54928 | 0.54896 | **0.54833** | +0.173% | 0.14345 | **0.14324** |
| **28 days** | 619 | 0.87801 | 0.87792 | **0.87717** | +0.096% | 0.23138 | **0.23119** |
| **56 days** | 615 | 1.53483 | 1.53485 | **1.53413** | +0.046% | 0.40455 | **0.40437** |
| **84 days** | 611 | 2.07945 | 2.07945 | **2.07884** | +0.030% | 0.54728 | **0.54712** |
| **112 days** | 607 | 2.51853 | 2.51852 | **2.51800** | +0.021% | 0.66040 | **0.66026** |
| **POOLED** | **3,695** | **1.32117** | **1.32103** | **1.32040** | **+0.059%** | **0.34692** | **0.34673** |

---

## 4. Predefined Decision Gate Evaluation

| Decision Gate Check | Predefined Threshold | Measured Empirical Result | Status |
| :--- | :--- | :--- | :---: |
| **Reference Invariance Hard Test** | Identical $q_g$ across ALR reference bases | Max $\Delta q_g = 0.00000000$ | **PASSED** |
| **Check 1: Material Relative ES Improvement** | $I_{\text{ES}} \ge +0.50\%$ relative improvement | $+0.059\%$ relative improvement | **FAILED** |
| **Check 2: No Marginal CRPS Degradation** | $I_{\text{CRPS}} \ge 0.00\%$ | $+0.054\%$ relative improvement | **PASSED** |
| **Check 3: Multi-Horizon Consistency** | Beats RC1 in $\ge 4$ of 6 horizons | Won in 6 of 6 horizons | **PASSED** |
| **Check 4: Calendar-Block Bootstrap Inference** | 95% 6-month block CI lower bound $> 0.0\%$ | 95% CI: `[-0.018%, +0.148%]` | **FAILED** |
| **Check 5: Prior Shrinkage Sensitivity** | $M_0 = 25$ sensitivity must improve over RC1 | $M_0 = 25$ ES = `1.32127` (vs RC1 `1.32117`) | **FAILED** |

---

## 5. Statistical Interpretation & Modeling Philosophy

1. **Why the Improvement is Negligible**:
   In Swedish polling, major institutes (Sifo, Novus, Demoskop, SVT/Verian, Ipsos) maintain comparable sampling standards once sample sizes ($N$) are accounted for. The residual variance across houses after $N$-adjustment is remarkably uniform.
2. **Reference leakage control**:
   The precision residuals use a leave-one-pollster-out contemporaneous CLR reference. The evaluated house therefore cannot improve its apparent precision merely by contributing to the consensus against which it is scored. Standardizing for $N$ still relies on the stated approximate inverse-$N$ sampling-variance assumption and bounded weights.
3. **Occam's Razor & Model Discipline**:
   An improvement of $+0.059\%$ that is not distinguishable from zero under block bootstrap and that fails under prior sensitivity does not meet our high evidentiary bar for adding model complexity.

---

## 6. Scope and model-status note

For the evaluated challenger suite:
1. **Experiment 1 (Industry Bias)** is closed as `ALREADY_REJECTED`.
2. **Experiment 2 (Pollster Precision)** is closed as `REJECTED_KEEP_RC1`.
3. The precision candidate was not adopted because its measured improvement did not meet the configured evidentiary gate.
4. **`ElectionSimulator v1.0-rc1` remains the retained baseline.** This experiment does not prove that all future model families have been searched or that RC1 is a universal optimum.
