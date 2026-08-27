# Opinion Dynamics & Historical CLR Transition Models

## 1. Purpose of Opinion Dynamics

While the **Opinion State Estimator** quantifies uncertainty about *current* public opinion, forecasting future election-day or future polling outcomes requires modeling *opinion movement over time*.

This document describes the foundational empirical dynamics models evaluated against future Poll of Polls consensus trajectories:

1. **`point_persistence`**: Deterministic benchmark ($\theta_{t+h} = \text{PoP}_t$).
2. **`empirical_raw`**: Direct resampling of historical joint CLR transitions.
3. **`symmetric_all_history`** (formerly `empirical_symmetric`): Sign-symmetric resampling of all eligible historical CLR transitions ($\text{drift} = 0$).
4. **`symmetric_4y`**: Sign-symmetric resampling restricted to trailing 4-calendar-year window.
5. **`symmetric_2y`**: Sign-symmetric resampling restricted to trailing 2-calendar-year window.
6. **`symmetric_recency_weighted`**: Sign-symmetric resampling with 730-day exponential recency weighting ($w = \exp(-\ln(2) \cdot \text{age\_days} / 730)$).

> [!NOTE]
> **Development Target Disclaimer**:
> *Poll of Polls future observations are used here as the development target for opinion-dynamics modeling. They are not assumed to be perfect observations of latent voter opinion.*

---

## 2. Compositional Representation: Centered Log-Ratio (CLR)

To model multi-party transitions without reference-category asymmetries (such as those observed with small categories in ALR), dynamics are modeled in **Centered Log-Ratio (CLR)** space across all nine canonical categories ($D=9$):

$$\text{clr}_i(\mathbf{p}) = \ln\left(\frac{p_i}{g(\mathbf{p})}\right) = \ln(p_i) - \frac{1}{D}\sum_{j=1}^D \ln(p_j)$$

where $g(\mathbf{p}) = \exp(\frac{1}{D}\sum_{j=1}^D \ln(p_j))$ is the geometric mean.

### Mathematical Properties
* $\sum_{i=1}^D \text{clr}_i = 0$ (elements lie on the zero-sum hyperplane in $\mathbb{R}^D$).
* Coordinate symmetric: no single category acts as an asymmetric divisor.
* Inverse CLR mapping:
  $$p_i = 100 \cdot \frac{\exp(\text{clr}_i - m)}{\sum_{j=1}^D \exp(\text{clr}_j - m)}, \quad m = \max(\mathbf{clr})$$

---

## 3. Direct Historical Transition Construction

For each forecast horizon $h \in \{7, 14, 28, 56, 84, 112\}$ days:

$$\mathbf{\Delta}_{s,h} = \text{CLR}(\text{PoP}_{s+h}) - \text{CLR}(\text{PoP}_s)$$

where both $s$ and $s+h$ exist exactly in `pollofpolls_timeseries.csv`.

### Structural Leakage Boundary
For any forecast origin $t$, a transition $(s, s+h)$ is eligible if and only if:

$$\text{transition\_end} = s + h \le t$$

Transitions ending after $t$ are structurally filtered before reaching the forecasting models.

### Minimum Historical Data Threshold
$$\text{MIN\_TRANSITIONS} = 30$$

If fewer than 30 eligible transitions exist for a given $(t, h)$ pair within the model's window, the case is skipped and recorded in diagnostics.

---

## 4. Dynamics Models

### A. Point Persistence (`point_persistence`)
$$\theta_{t+h} = \text{PoP}_t$$
Uses the exact origin Poll of Polls composition as a deterministic point forecast with zero dispersion.

### B. Empirical Raw Transitions (`empirical_raw`)
Resamples complete 9-party transition vectors uniformly with replacement from the eligible pool:

$$\text{CLR}(\theta_{t+h}^{(i)}) = \text{CLR}(\text{PoP}_t) + \mathbf{\Delta}_h^{(i)}$$

* Complete 9-party transitions are sampled jointly; party movements are never sampled independently.
* Preserves empirical historical directional drift and empirical cross-party correlation structure.

### C. Symmetric Recency-Adaptive Variants
Resamples transition vectors and independently applies a random sign $S^{(i)} \in \{-1, +1\}$ with equal probability ($P(S=1) = 0.5$):

$$\text{CLR}(\theta_{t+h}^{(i)}) = \text{CLR}(\text{PoP}_t) + S^{(i)} \cdot \mathbf{\Delta}_h^{(i)}$$

* Enforces zero expected historical drift ($\mathbb{E}[\mathbf{\Delta}] = \mathbf{0}$) while preserving empirical movement magnitudes and correlations.
* **`symmetric_all_history`**: Uses all eligible historical transitions ($\text{transition\_end} \le t$).
* **`symmetric_4y`**: Restricts to transitions with $t - 4\text{ years} \le \text{transition\_end} \le t$.
* **`symmetric_2y`**: Restricts to transitions with $t - 2\text{ years} \le \text{transition\_end} \le t$.
* **`symmetric_recency_weighted`**: Samples all eligible transitions with probability proportional to $w = \exp(-\ln(2) \cdot \text{age\_days} / 730)$.

---

## 5. Rolling Historical Evaluation Protocol

Because historical data through 2026 was inspected during exploratory iterations, evaluation is structured as **rolling historical evaluation across annual blocks** (`2019`, `2020`, `2021`, `2022`, `2023`, `2024`, `2025`, `2026 YTD`):

* Avoids over-interpreting a single arbitrary split.
* Tests whether recency-adaptive transition pools perform consistently across varying political volatility regimes.

### Primary Model Selection Criterion
To avoid distortion from the small, volatile derived `REST` category, the **primary model-selection metric** is:

$$\text{Equal-Weighted Annual Mean CRPS across the 8 parliamentary parties: M, L, C, KD, S, V, MP, SD}$$
with central 50%, 80%, and 90% interval coverage acting as calibration guardrails.
