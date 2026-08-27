# Residual Robustness and Election Result Layer v1 — Methodology and Report

## 1. Executive Summary

This study evaluates whether an empirical poll-to-election residual layer applied in Additive Log-Ratio / Centered Log-Ratio space improves election hindcasts over the frozen `state_plus_dynamics` model (`OpinionState v1.1` + `Dynamics v2 = symmetric_all_history`).

### Key Findings
1. **Window Robustness**: The historical residual patterns (S outperformance, V/MP underperformance, KD/L underperformance) are **highly robust** across 7-day, 14-day, and 21-day trailing consensus windows.
2. **Model Variant Comparison**:
   - `base` (`state_plus_dynamics`): **8-party mean CRPS = 0.8494** (1.0623 in 2018, 0.6364 in 2022).
   - `bias_only`: **8-party mean CRPS = 6.1842** (severe degradation).
   - `noise_only`: **8-party mean CRPS = 2.5503** (severe degradation).
   - `bias_plus_noise`: **8-party mean CRPS = 1.9950** (severe degradation).
3. **Recommendation**: **Retain `base` (`state_plus_dynamics`)**. Naive empirical CLR residual bootstrapping fails because historical structural transitions (specifically Sverigedemokraterna being unpolled at 0.0% in 2002) create severe log-ratio leverage that distorts the simplex when transferred to modern multi-party elections.

---

## 2. Residual-Window Robustness Audit (7d, 14d, 21d)

Historical residuals were recomputed across all six modern general elections ($2002, 2006, 2010, 2014, 2018, 2022$) for three trailing lookback windows:

| Party | 7-Day Mean Residual | 7-Day Sign | 14-Day Mean Residual (Canonical) | 14-Day Sign | 21-Day Mean Residual | 21-Day Sign |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S** | **+1.82 pp** | 6+ / 0- | **+1.97 pp** | 6+ / 0- | **+1.89 pp** | 6+ / 0- |
| **V** | **-1.02 pp** | 0+ / 6- | **-1.06 pp** | 0+ / 6- | **-1.02 pp** | 0+ / 6- |
| **MP** | **-0.89 pp** | 0+ / 6- | **-0.83 pp** | 0+ / 6- | **-0.81 pp** | 0+ / 6- |
| **M** | **+0.45 pp** | 5+ / 1- | **+0.45 pp** | 5+ / 1- | **+0.46 pp** | 5+ / 1- |
| **KD** | **-0.51 pp** | 1+ / 5- | **-0.50 pp** | 1+ / 5- | **-0.49 pp** | 1+ / 5- |
| **L** | **-0.38 pp** | 1+ / 5- | **-0.35 pp** | 1+ / 5- | **-0.35 pp** | 1+ / 5- |
| **C** | **+0.05 pp** | 3+ / 3- | **+0.16 pp** | 3+ / 3- | **+0.11 pp** | 4+ / 2- |
| **SD** | **+0.91 pp** | 4+ / 2- | **+0.66 pp** | 4+ / 2- | **+0.66 pp** | 4+ / 2- |
| **REST** | **-0.43 pp** | 1+ / 5- | **-0.50 pp** | 1+ / 5- | **-0.45 pp** | 1+ / 5- |

### Overall Window MAE
* **7-day Window**: Overall 8-party MAE = **0.98 pp**
* **14-day Window** (Canonical): Overall 8-party MAE = **1.02 pp**
* **21-day Window**: Overall 8-party MAE = **1.01 pp**

> **Conclusion**: The 100% sign-consistent outperformance of S alongside the 100% sign-consistent underperformance of V and MP is completely invariant to the consensus window length.

---

## 3. Election Layer Mathematical Formulation

Starting from intermediate sample $z^{(i)}$ from `state_plus_dynamics`:
$$\text{CLR}(z^{(i)}) = \text{CLR}(\theta_t^{(i)}) + S^{(i)}\mathbf{\Delta}_h^{(i)}$$

Four variants were evaluated:

1. **`base`**:
   $$\text{CLR}(z_E^{(i)}) = \text{CLR}(z^{(i)})$$
2. **`bias_only`**:
   $$\bar{\mathbf{r}} = \frac{1}{K} \sum_{k=1}^K \mathbf{r}_{e_k}, \quad \text{CLR}(z_E^{(i)}) = \text{CLR}(z^{(i)}) + \bar{\mathbf{r}}$$
3. **`noise_only`**:
   $$\text{CLR}(z_E^{(i)}) = \text{CLR}(z^{(i)}) + (\mathbf{r}_{e^{(i)}} - \bar{\mathbf{r}})$$
4. **`bias_plus_noise`**:
   $$\text{CLR}(z_E^{(i)}) = \text{CLR}(z^{(i)}) + \mathbf{r}_{e^{(i)}}$$

### Chronological Training Sets
* **2018 Hindcasts**: Training pool strictly $\{2002, 2006, 2010, 2014\}$ ($K=4$).
* **2022 Hindcasts**: Training pool strictly $\{2002, 2006, 2010, 2014, 2018\}$ ($K=5$).

---

## 4. Empirical Hindcast Results (2018 and 2022)

### Overall Performance Across Both Elections (12 Forecast Cases / 108 Party Rows per Variant)

| Variant | 8-Party MAE | 8-Party Mean CRPS | All-9 Mean CRPS | 50% Coverage (Width) | 80% Coverage (Width) | 90% Coverage (Width) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`base`** | **1.11%** | **0.8494** | **0.8183** | **25.9% (0.87 pp)** | **43.5% (1.69 pp)** | **54.6% (2.29 pp)** |
| `bias_only` | 6.56% | 6.1842 | 5.5295 | 2.8% (0.78 pp) | 4.6% (1.53 pp) | 9.3% (2.04 pp) |
| `noise_only` | 2.75% | 2.5503 | 2.3352 | 35.2% (3.26 pp) | 93.5% (19.55 pp) | 93.5% (19.99 pp) |
| `bias_plus_noise` | 1.39% | 1.9950 | 1.8157 | 49.1% (3.21 pp) | 64.8% (17.83 pp) | 71.3% (18.21 pp) |

### By Election

| Election | Variant | 8-Party MAE | 8-Party Mean CRPS | 90% Coverage (Width) |
| :--- | :--- | :---: | :---: | :---: |
| **2018** | **`base`** | **1.34%** | **1.0623** | **38.9% (2.20 pp)** |
| 2018 | `bias_only` | 8.10% | 7.7362 | 7.4% (1.92 pp) |
| 2018 | `noise_only` | 2.58% | 2.6014 | 90.7% (19.75 pp) |
| 2018 | `bias_plus_noise` | 1.98% | 2.6561 | 55.6% (17.74 pp) |
| **2022** | **`base`** | **0.89%** | **0.6364** | **70.4% (2.38 pp)** |
| 2022 | `bias_only` | 5.03% | 4.6323 | 11.1% (2.16 pp) |
| 2022 | `noise_only` | 2.93% | 2.4993 | 96.3% (20.23 pp) |
| 2022 | `bias_plus_noise` | 0.79% | 1.3338 | 87.0% (18.68 pp) |

---

## 5. Diagnostic Analysis: Why Non-Parametric CLR Bootstrapping Fails

1. **Log-Ratio Outlier Leverage**:
   - In 2002, SD was not reported as an individual category by major pollsters (0.0% polled, floored to 0.01% in CLR), but received 1.44% in certified returns.
   - This produced a CLR residual of $\mathbf{r}_{\text{SD}, 2002} = +4.48$.
   - When averaged into $\bar{\mathbf{r}}$ for 2018 ($+1.29$), it acts as a $3.6\times$ multiplier on SD's modern ~18% support, projecting SD at ~48% and deflating all other parties.
2. **Interval Ballooning in Noise Variants**:
   - In `noise_only` and `bias_plus_noise`, drawing the 2002 residual introduces wild swings ($19.55$ pp interval widths), destroying sharp probabilistic resolution.
3. **Double Counting**:
   - The 14-day pre-election residual represents not only pure polling bias, but also late-breaking opinion movement and turnout. Because `Dynamics v2` already explicitly models transition uncertainty over the forecast horizon, unconstrained residual bootstrapping partially double-counts late movement.

---

## 6. Substantive Clarification on Historical Patterns

1. **Intra-Bloc Consolidation**:
   - The historical pattern where S systematically outperforms pre-election polls (+1.97 pp) while V (-1.06 pp) and MP (-0.83 pp) underperform (-1.89 pp combined) is **consistent with intra-bloc consolidation**.
   - Aggregate residuals alone cannot prove individual tactical voter switching, as differential turnout or late decision-making within the left-green voting pool could produce similar net shifts.
2. **Residual Composition**:
   - 14-day poll-to-election residuals reflect a mixture of measurement error, house effects, turnout differential, and genuine late campaign shifts.

---

## 7. Recommendation for Production Simulator

* **Retain `base` (`state_plus_dynamics`)** as the standard production election forecast model.
* Do not apply unconstrained empirical CLR residual bootstrapping.
* Any future strategic voting or threshold adjustments must be modeled structurally rather than via historical whole-vector CLR replay.
