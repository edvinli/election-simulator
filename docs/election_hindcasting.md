# Election Hindcast v1 — Methodology and Retrospective Evaluation Report

> Interpretation: **Retrospective historical evaluation (not independent holdout validation)**. Model-family choices and polling calibration used evidence from the same 2018/2022 period evaluated below. Coverage and horizon patterns are descriptive and should not be read as formal calibration or guaranteed monotonic improvement.

## 1. Overview and Model Formulations

Election Hindcast v1 evaluates probabilistic election-day forecasts for Swedish Riksdag general elections against certified official election returns, combining frozen components:
1. **`OpinionState v1.1`**: Latent opinion state estimation on the forecast origin date $t$.
2. **`Dynamics v2` (`symmetric_all_history`)**: Sign-symmetric historical CLR transitions over horizon $h = E - t$.

### Models Evaluated

1. **`point_persistence`** (Deterministic Baseline):
   $$\theta_E = \text{PoP}_t$$
   Uses the exact Poll of Polls composition at origin date $t$ as a point prediction.

2. **`dynamics_only`**:
   $$\text{CLR}(\theta_E^{(i)}) = \text{CLR}(\text{PoP}_t) + S^{(i)} \cdot \mathbf{\Delta}_h^{(i)}$$
   Where $S^{(i)} \in \{-1, +1\}$ with equal probability, and $\mathbf{\Delta}_h^{(i)}$ is drawn with replacement from historical CLR transitions $(s, s+h)$ satisfying $s+h \le t$.

3. **`state_plus_dynamics`**:
   $$\text{CLR}(\theta_E^{(i)}) = \text{CLR}(\theta_t^{(i)}) + S^{(i)} \cdot \mathbf{\Delta}_h^{(i)}$$
   Where $\theta_t^{(i)}$ is sampled independently from `OpinionState v1.1` estimated as of $t$, and combined with the shared symmetric historical dynamics draw in CLR space.

---

## 2. Category Alignment and Target Space

Forecast and evaluation occur in a 9-category composition space:

$$\text{M, L, C, KD, S, V, MP, SD, REST}$$

Official election returns from `data/processed/elections/riksdag_election_results.csv` are aligned using integer vote totals:

$$\text{votes}_{\text{REST}} = \text{votes}_{\text{FI}} + \text{votes}_{\text{OTHER}}$$
$$\text{actual\_share} = \frac{\text{votes}}{\text{valid\_votes\_total}} \times 100$$

Every election target strictly sums to $100.0000\%$.

---

## 3. Evaluation Schedule and Leakage Boundaries

### Elections Evaluated
* **2018 General Election**: `2018-09-09` (6,476,725 valid votes)
* **2022 General Election**: `2022-09-11` (6,477,970 valid votes)
*(2014 is excluded because the continuous daily PoP series begins after the 2014 election).*

### Horizons Evaluated
* **$h \in \{112, 84, 56, 28, 14, 7\}$ days** before election day.
* All 12 origin dates exist exactly in `pollofpolls_timeseries.csv` (0 skipped cases).

### Structural Leakage Safety
For each forecast origin $t = E - h$:
1. `OpinionState v1.1` receives only polls published on or before $t$.
2. Historical transition pool includes only pairs $(s, s+h)$ where $s+h \le t$.
3. Seed generation is deterministic based on `(base_seed, origin_date, horizon_days)`.
4. `dynamics_only` and `state_plus_dynamics` share the exact same sampled transition indices and $\pm$ signs to isolate the pure marginal effect of latent state uncertainty.

---

## 4. Key Performance Summary

### Overall Comparison (2018 + 2022, 108 Party-Horizon Forecasts)

| Model | Parliamentary MAE | Parliamentary CRPS | All-9 CRPS | 50% Coverage (Width) | 80% Coverage (Width) | 90% Coverage (Width) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `point_persistence` | 1.12% | 1.1180 | 1.0699 | 0.0% (0.00) | 0.0% (0.00) | 0.0% (0.00) |
| `dynamics_only` | 1.11% | 0.8820 | 0.8495 | 13.9% (0.70) | 31.5% (1.37) | 42.6% (1.93) |
| `state_plus_dynamics` | **1.11%** | **0.8494** | **0.8183** | **25.9% (0.87)** | **43.5% (1.69)** | **54.6% (2.29)** |

### By Election

| Election | Model | Parliamentary MAE | Parliamentary CRPS | 90% Coverage (Width) |
| :--- | :--- | :---: | :---: | :---: |
| **2018** | `point_persistence` | 1.34% | 1.3388 | 0.0% (0.00) |
| **2018** | `dynamics_only` | 1.34% | 1.1067 | 33.3% (1.88) |
| **2018** | `state_plus_dynamics` | **1.34%** | **1.0623** | **38.9% (2.20)** |
| **2022** | `point_persistence` | 0.90% | 0.8972 | 0.0% (0.00) |
| **2022** | `dynamics_only` | 0.89% | 0.6574 | 51.9% (1.99) |
| **2022** | `state_plus_dynamics` | **0.89%** | **0.6364** | **70.4% (2.38)** |

---

## 5. Diagnostic Findings

1. **Marginal Value of OpinionState Uncertainty**:
   - Adding OpinionState uncertainty was associated with lower CRPS in both elections (+0.0444 in 2018, +0.0210 in 2022).
   - It improves 90% interval coverage from 42.6% to 54.6% (and up to 70.4% in 2022) with only a modest width increase from 1.93 to 2.29 percentage points.
   - Interpretation: this retrospective comparison supports retaining OpinionState uncertainty in the production architecture, but is not independent validation of a universally beneficial effect.

2. **Persistent Directional Polling Biases**:
   - **V (Vänsterpartiet)** was systematically over-polled relative to election day returns in both cycles (+1.09% in 2018, +1.38% in 2022), resulting in actual percentiles $\le 1.3\%$.
   - **REST (Minor Parties)** was over-polled in 2018 (+1.11% bias), driven by pre-election polling overstating FI support.
   - **SD (Sverigedemokraterna)** exhibited opposite shifts: over-polled in 2018 (+2.12%) and under-polled in 2022 (-1.87%).

3. **4% Threshold Dynamics (`L`, `MP`, `KD`)**:
   - In 2018, KD polled below 4% at $h \ge 56$d before surging to 6.32% on election day.
   - In 2022, both L (3.69% at 112d) and MP (3.19% at 112d) polled below 4% in early origins before recovering to 4.61% and 5.08% on election day.
   - Early horizon forecasts correctly reflected high probability mass straddling the threshold.

---

## 6. Reproduction Commands

```bash
# Run full election hindcasts
make hindcast

# Run unit tests
make test-pollofpolls
```
