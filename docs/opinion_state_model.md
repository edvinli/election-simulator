# Opinion State Estimator v1.1

## 1. Purpose of the Model

The **Opinion State Estimator v1.1** answers a single, specific foundational question for Swedish parliamentary election forecasting:

> *Given all polling and consensus information available as of a specified date, what is our estimate of current Swedish party support, and what uncertainty distribution should we attach to that estimate?*

This model is **not** an election-day forecast. It does not predict future voter movement, seat distributions, strategic voting, or coalition outcomes. Instead, it provides a strictly historical, leakage-free empirical snapshot of latent party support and polling consensus uncertainty at any requested point in time.

---

## 2. Input Datasets

The model consumes the processed datasets created by the Pollofpolls data acquisition pipeline:

1. `data/processed/pollofpolls/pollofpolls_timeseries.csv`:
   - Daily consensus estimates from 2014-09-15 through 2026-08-23.
   - Contains published support percentages for canonical Swedish parliamentary parties.
2. `data/processed/pollofpolls/individual_polls.csv`:
   - 1,437 reconstructed individual polls (14,370 long-format rows) spanning 2009 through 2026.
   - Primary support values are preserved strictly from first-party Pollofpolls sources.
   - Enriched with supplementary metadata (`publication_date`, `sample_size`) from SwedishPolls via exact interview-span crosswalk matching.

---

## 3. Canonical Party Representation & REST

The model tracks the eight modern parliamentary parties:

* **M** — Moderaterna
* **L** — Liberalerna (formerly Folkpartiet, FP)
* **C** — Centerpartiet
* **KD** — Kristdemokraterna
* **S** — Socialdemokraterna
* **V** — Vänsterpartiet
* **MP** — Miljöpartiet
* **SD** — Sverigedemokraterna

A ninth derived category, **REST**, represents all other vote intentions (including Feministiskt Initiativ / FI, Piratpartiet, Medborgerlig Samling, and other minor parties):

$$\text{REST} = 100 - (\text{M} + \text{L} + \text{C} + \text{KD} + \text{S} + \text{V} + \text{MP} + \text{SD})$$

### Data Integrity & Anomaly Handling
- `REST` is derived only when all eight main parties are present.
- If calculated $\text{REST} < -10^{-5}$ (materially negative), the observation is rejected as an invalid composition and recorded in diagnostics. Tiny floating-point imprecision in $[-10^{-5}, 0.0)$ is clamped to $0.0$.
- Non-negative shares smaller than $\text{MIN\_SHARE\_PCT} = 0.01\%$ are floored to $0.01\%$, followed by exact renormalization of all 9 parts to $100.0\%$.

---

## 4. Compositional Geometry (Additive Log-Ratio / ALR)

Party support data is compositional: shares must be strictly positive and sum to $100\%$. Standard unconstrained multivariate normal models on percentage points violate these constraints.

We use the **Additive Log-Ratio (ALR)** transformation with `REST` as the reference category. For the 8 canonical parties:

$$z_i = \ln\left(\frac{P_i}{P_{\text{REST}}}\right), \quad i \in \{\text{M}, \text{L}, \text{C}, \text{KD}, \text{S}, \text{V}, \text{MP}, \text{SD}\}$$

### Inverse Transformation
The inverse mapping from $\mathbf{z} \in \mathbb{R}^8$ to simplex percentages $\mathbf{P} \in \Delta^9$ is computed via a numerically stable max-shifted softmax:

$$m = \max(z_1, \dots, z_8, 0)$$

$$P_i = 100 \cdot \frac{\exp(z_i - m)}{\exp(-m) + \sum_{j=1}^8 \exp(z_j - m)}, \quad P_{\text{REST}} = 100 \cdot \frac{\exp(-m)}{\exp(-m) + \sum_{j=1}^8 \exp(z_j - m)}$$

This guarantees that all party shares $P_i > 0$ and $\sum_{i=1}^8 P_i + P_{\text{REST}} \equiv 100\%$.

---

## 5. Strict `as_of` Behavior & Leakage Prevention

To support honest historical backtesting, every calculation is strictly conditioned on an `as_of` date:

1. **Poll Known Date**: A poll is considered known only if its `publication_date <= as_of`. Polls without verified publication dates are excluded from strict historical calculations.
2. **Fieldwork Cutoff**: A poll is included only if its `interview_end <= as_of`.
3. **Historical Residuals Pool**: To prevent future lookahead bias in covariance and house effect estimation, residuals are calculated strictly from polls with `publication_date < as_of` (strictly prior to `as_of`).
4. **Mandatory Leakage Safety**: Introducing future polls or running the model on future datasets cannot alter any historical opinion state estimate.

---

## 6. Central Point Estimate

The model uses the published **Poll of Polls consensus** as its central mean estimate:

$$\boldsymbol{\mu}_{\text{ALR}} = \text{ALR}\left(\text{PollOfPolls}_{\le \text{as\_of}}\right)$$

If no Poll of Polls timeseries observation exists on or before `as_of`, an explicit error is raised.

---

## 7. Historical Poll Residuals & House Effects

For each eligible historical poll:
1. Determine reference date:

$$\text{reference\_date} = \text{interview\_start} + \lfloor(\text{interview\_end} - \text{interview\_start}) / 2\rfloor$$

2. Match the latest Poll of Polls observation on or before `reference_date`.
3. If the matching estimate is older than $\text{MAX\_ESTIMATE\_MATCH\_LAG\_DAYS} = 3\text{ days}$, skip the residual.
4. Calculate 8-dimensional residual vector: $\mathbf{r} = \mathbf{z}_{\text{poll}} - \mathbf{z}_{\text{PoP}}$.

### Trailing Covariance Window & Fallback
- Default window: 4 calendar years prior to `as_of` ($\text{as\_of} - 4\text{ years} \le \text{publication\_date} < \text{as\_of}$).
- If fewer than $\text{MIN\_RESIDUAL\_POLLS} = 100$ eligible polls exist in the 4-year window, the pool expands backward to include all prior eligible residuals, and a warning diagnostic is recorded.

### Pollster House Effects
From the active residual pool:
- For pollsters with $\ge \text{MIN\_POLLS\_FOR\_HOUSE\_EFFECT} = 20$ observations:

$$\mathbf{h}_{\text{pollster}} = \frac{1}{N_{\text{pollster}}} \sum \mathbf{r}$$

- For pollsters with $< 20$ observations: $\mathbf{h} = \mathbf{0}$.
- Compute adjusted residuals: $\mathbf{r}_{\text{adj}} = \mathbf{r} - \mathbf{h}_{\text{pollster}}$.

---

## 8. Residual Covariance: The v1.1 Correction

Sample covariance is calculated with Bessel's correction $(N - 1)$ on adjusted residuals:

$$\mathbf{\Sigma}_{\text{residual}} = \frac{1}{N - 1} \sum_{k=1}^N (\mathbf{r}_{\text{adj}, k} - \bar{\mathbf{r}}_{\text{adj}})(\mathbf{r}_{\text{adj}, k} - \bar{\mathbf{r}}_{\text{adj}})^T$$

### Why Covariance Shrinkage Was Removed in v1.1
In the initial v1 prototype, fixed 20% diagonal shrinkage ($\mathbf{\Sigma} \to 0.80\mathbf{\Sigma} + 0.20\text{diag}(\mathbf{\Sigma})$) was applied in coordinate space.

A formal statistical audit demonstrated that:
1. Because the reference category `REST` is small ($\sim 1.8\%$), its log-variance is large ($\approx 0.41$), creating a shared common-mode shift $-\ln(P_{\text{REST}})$ across all eight ALR coordinates (average raw off-diagonal correlation $+0.975$).
2. In an un-shrunk covariance matrix, this common shift cancels out completely during inverse-ALR softmax transformation.
3. However, diagonal shrinkage in coordinate space reduced off-diagonal covariances by 20% while preserving diagonal variances. This broke the common-mode cancellation, injecting $\approx 0.20 \times \text{Var}(\ln \text{REST})$ of artificial independent variance into each party and inflating party standard deviations by $5\times$ to $6\times$ (e.g. S SD jumped from $0.59\text{ pp}$ to $3.52\text{ pp}$).
4. With shrinkage disabled ($\text{COVARIANCE\_DIAGONAL\_SHRINKAGE} = 0.0$), percentage-space uncertainty becomes strictly **reference-category invariant** (producing identical party standard deviations whether `REST`, `S`, or `M` is used as reference), and uncertainty returns to empirically plausible levels matching binomial sampling theory.

In v1.1, the empirical house-effect-adjusted covariance is used directly without coordinate-space diagonal shrinkage. If regularization is needed in future versions, it will be designed on coordinate-invariant orthogonal subspaces (e.g., ILR or CLR) after historical backtesting.

---

## 9. Current Polling Information & Effective Poll Count

State uncertainty reflects the volume and recency of recent polls:
- Eligible recent polls: $\text{publication\_date} \le \text{as\_of}$, $\text{interview\_end} \le \text{as\_of}$, $\text{as\_of} - 60\text{ days} \le \text{reference\_date} \le \text{as\_of}$.
- Recency weight ($\text{half-life} = 21\text{ days}$):

$$w_{\text{recency}} = \exp\left(-\frac{\ln(2) \cdot \text{age\_days}}{21}\right)$$

- Sample size weight (benchmarked to 1,000 respondents):

$$w_{\text{sample}} = \text{clip}\left(\sqrt{\frac{N}{1000}}, 0.70, 1.50\right)$$

- Total poll weight: $w = w_{\text{recency}} \cdot w_{\text{sample}}$.
- Kish effective sample count:

$$n_{\text{eff}} = \frac{(\sum w)^2}{\sum w^2}$$

- Capped effective count:

$$n_{\text{eff\_used}} = \min(\max(n_{\text{eff}}, 1.0), 8.0)$$

---

## 10. State Covariance & Deterministic Sampling

The state uncertainty covariance in ALR space is:

$$\mathbf{\Sigma}_{\text{state}} = \frac{\mathbf{\Sigma}_{\text{residual}}}{n_{\text{eff\_used}}}$$

### Bounded-Jitter Cholesky Decomposition
1. Attempt lower-triangular factorization $\mathbf{L}\mathbf{L}^T = \mathbf{\Sigma}_{\text{state}}$.
2. If non-positive-definite due to floating-point conditioning, search over bounded diagonal jitter factors relative to the average diagonal variance: $\bar{\sigma}^2 \times [10^{-8}, 10^{-7}, 10^{-6}, 10^{-5}, 10^{-4}]$.
3. Record jitter amount in diagnostics (empirically $0.0$ on all tested historical datasets).

### Monte Carlo Sampling API
1. Draw standard normal vector $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_8)$ via seeded generator.
2. Form ALR sample: $\mathbf{x} = \boldsymbol{\mu}_{\text{ALR}} + \mathbf{L}\mathbf{z}$.
3. Transform to composition: $\mathbf{P} = \text{alr\_to\_composition}(\mathbf{x})$.

---

## 11. Configurable Hyperparameters (v1.1)

All constants are centralized in `scripts/pollofpolls/state_config.py`:

| Parameter | Value | Purpose |
| :--- | :--- | :--- |
| `MIN_SHARE_PCT` | `0.01` | Minimum percentage floor before ALR transformation |
| `MAX_ESTIMATE_MATCH_LAG_DAYS` | `3` | Maximum allowed days between poll reference date and matching PoP estimate |
| `COVARIANCE_LOOKBACK_YEARS` | `4` | Trailing window of historical residuals for covariance estimation |
| `MIN_RESIDUAL_POLLS` | `100` | Minimum residual observations required before triggering historical fallback |
| `MIN_POLLS_FOR_HOUSE_EFFECT` | `20` | Minimum polls required for an institute to receive an empirical house effect |
| `COVARIANCE_DIAGONAL_SHRINKAGE` | `0.0` | Disabled in v1.1 to preserve reference-category invariance and common-mode cancellation |
| `RECENT_POLL_LOOKBACK_DAYS` | `60` | Maximum age of polls included in current polling volume calculation |
| `RECENCY_HALF_LIFE_DAYS` | `21` | Exponential half-life for poll recency weighting |
| `MAX_EFFECTIVE_POLLS` | `8.0` | Maximum ceiling on Kish effective poll count |

---

## 12. Important Limitations & Modeling Caveats

1. **Not Yet Calibrated Against Election Outcomes**: The v1.1 model provides an empirical, uncalibrated representation of opinion state uncertainty. Historical backtesting and calibration against actual general election results will be conducted in the next modeling stage.
2. **Endogenous Baseline Bias**: Because the Poll of Polls estimate used as the baseline may include the individual poll whose residual is being measured, empirical residual variance may be somewhat understated.
3. **Consensus Reliance**: The model relies on the published Poll of Polls point estimate as its mean and does not re-weight or re-estimate the underlying latent trajectory.
4. **Empirical Proxy**: Polling disagreement and historical variance around the consensus serve as the proxy for state uncertainty without separating sampling error from non-sampling error.
5. **Simple House Effects**: Pollster house effects are estimated as straightforward empirical means without Bayesian hierarchical shrinkage across institutes.
6. **No Election Forecast**: This model reflects current opinion state uncertainty only. It does **not** include drift, voter volatility, or time-to-election uncertainty.
