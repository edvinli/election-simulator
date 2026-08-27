# Historical Backtesting Framework

## 1. Purpose of the Framework

The **Historical Backtesting Framework** evaluates the predictive accuracy, calibration, and probabilistic scoring of Swedish election forecasting models across multiple historical forecast origins and horizons:

> *If we had only known information available as of historical date $t$, how well would our probabilistic forecast distribution have predicted the Poll of Polls state at $t+h$?*

The framework provides an honest, leakage-safe testbed for comparing competing opinion-dynamics models against standard baselines.

> [!NOTE]
> **Development Target Disclaimer**:
> *Poll of Polls future observations are used here as the development target for opinion-dynamics modeling. They are not assumed to be perfect observations of latent voter opinion.*

---

## 2. Leakage Rules & Data Conditioning

To prevent future lookahead bias, all data conditioning is strictly enforced:

1. **Origin As-Of Boundary**: For forecast origin $t$, no polling publication date, fieldwork end date, or consensus observation after $t$ may influence the forecast.
2. **Transition Filtering**: For dynamic transition models, historical training transitions are filtered strictly by:
   $$\text{transition\_end} \le \text{origin\_date}$$
3. **Deterministic Seed Derivation**: Random seeds are derived per origin using cryptographic hashing (`hashlib.sha256(f"{base_seed}:{model_id}:{origin_date}".encode())`), preventing random state coupling while guaranteeing complete reproducibility.

---

## 3. Exact Target-Date Rule

For forecast origin $t$ and horizon $h \in \{7, 14, 28, 56, 84, 112\}$ days:
* The evaluation target is the **exact** published Poll of Polls consensus on `target_date = origin_date + horizon_days`.
* If no exact Poll of Polls observation exists on that exact calendar date (e.g. beyond the dataset horizon), the case is skipped and tracked in diagnostics.
* **No substitution**: The framework does not interpolate or substitute nearby dates.

---

## 4. No-Change Baseline Model (`no_change`)

The initial reference baseline assumes that latent opinion remains constant:

$$\theta_{t+h} = \theta_t$$

### Procedure:
1. Obtain the `OpinionState` at origin date $t$ using Opinion State Estimator v1.1.
2. Draw $N$ deterministic Monte Carlo samples from the origin state's predictive distribution.
3. Carry those samples forward unchanged to all forecast horizons ($h = 7, 14, 28, 56, 84, 112$ days).
4. Evaluate the samples against the actual target observations.

This baseline captures current measurement uncertainty only; it intentionally contains **no future opinion movement**.

---

## 5. Evaluation Metrics

### Point Error Metrics
For predictive point forecast $\hat{y}$ (defined strictly as predictive P50 / median) and actual observation $y$:
* **Error**: $e = \hat{y} - y$
* **Mean Absolute Error (MAE)**: $\text{MAE} = \operatorname{mean}(|e|)$
* **Root Mean Squared Error (RMSE)**: $\text{RMSE} = \sqrt{\operatorname{mean}(e^2)}$

### Probabilistic Metric: Continuous Ranked Probability Score (CRPS)
For $n$ Monte Carlo samples $x_1, \dots, x_n$ and observed outcome $y$, the empirical CRPS is:

$$\text{CRPS}(F, y) = \frac{1}{n}\sum_{i=1}^n |x_i - y| - \frac{1}{2n^2}\sum_{i=1}^n \sum_{j=1}^n |x_i - x_j|$$

Using sorted samples $x_{(0)} \le x_{(1)} \le \dots \le x_{(n-1)}$, the second term is computed in $O(n)$:

$$\frac{1}{2n^2}\sum_{i=0}^{n-1} \sum_{j=0}^{n-1} |x_i - x_j| = \frac{1}{n^2}\sum_{i=0}^{n-1} (2i + 1 - n) x_{(i)}$$

### Probabilistic Calibration & Interval Widths
For central intervals derived from predictive quantiles:
* **50% Central Interval**: $[P_{25}, P_{75}]$, empirical coverage $\mathbb{I}(y \in [P_{25}, P_{75}])$, width $P_{75} - P_{25}$.
* **80% Central Interval**: $[P_{10}, P_{90}]$, empirical coverage $\mathbb{I}(y \in [P_{10}, P_{90}])$, width $P_{90} - P_{10}$.
* **90% Central Interval**: $[P_{05}, P_{95}]$, empirical coverage $\mathbb{I}(y \in [P_{05}, P_{95}])$, width $P_{95} - P_{05}$.

---

## 6. Output Schema

The framework outputs two primary files to `data/processed/backtests/`:

1. **Per-Case Raw Forecasts CSV** (`backtest_cases_{model}_{start}_{end}.csv`):
   Contains one row per `(model, origin_date, target_date, horizon_days, party)`:
   - `model`, `origin_date`, `target_date`, `horizon_days`, `party`
   - `point_forecast` (P50), `predictive_mean`, `actual`
   - `error`, `absolute_error`, `squared_error`
   - `p05`, `p10`, `p25`, `p50`, `p75`, `p90`, `p95`
   - `interval50_contains_actual`, `interval80_contains_actual`, `interval90_contains_actual`
   - `width_50`, `width_80`, `width_90`
   - `crps`, `samples_count`, `seed`
   - `origin_estimate_date`, `origin_estimate_age_days`

2. **Aggregated Horizon Summary** (`backtest_by_horizon_{model}_{start}_{end}.csv` and `.json`):
   Aggregated metrics across all evaluated cases for each horizon.

---

## 7. Model Extension Interface

New forecasting models plug into the framework by implementing the `ForecastModel` protocol in [`scripts/pollofpolls/backtest_models.py`](file:///Users/edvinli/Documents/Git/edvinli.github.io/scripts/pollofpolls/backtest_models.py):

```python
class MyOpinionDynamicsModel:
    model_id: str = "my_model"

    def forecast(
        self,
        context: ForecastContext,
        horizon_days: int,
        samples_count: int,
        seed: int,
    ) -> ForecastDistribution:
        # 1. Use context.opinion_state for origin state
        # 2. Use filter_transitions_as_of(context.transitions, context.origin_date)
        # 3. Simulate future paths for horizon_days
        # 4. Return ForecastDistribution
        ...
```
