# Historical Poll-to-Election Residual Study (2002–2022)

## 1. Overview and Study Scope

This study examines the empirical residuals between final pre-election polling consensus and certified election results across six modern Swedish general elections:

$$\text{Elections Evaluated: } 2002, 2006, 2010, 2014, 2018, 2022$$

### Purpose
To diagnose whether election-day errors are:
1. **Generic / Broad**: Broad sampling noise across all parties and elections.
2. **Party-Specific / Asymmetric**: Systematic directional discrepancies for specific parties (e.g. S, V, MP).
3. **Threshold-Concentrated**: Outsized volatility or tactical behavior near the 4.0% parliamentary threshold.
4. **Structured Within Blocs**: Zero-sum intra-bloc voter consolidation on election day.

---

## 2. Polling Consensus Construction

For each election date $E$:

1. **Eligibility Window**:
   - $\text{publication\_date} \le E$
   - $\text{interview\_end} \le E$
   - $\text{interview\_end} \ge E - 14\text{ days}$ (strictly a trailing 14-day window).
2. **Pollster Deduplication**:
   - For each pollster, only its **latest eligible poll** is retained.
   - Deterministic tie-breaking: `interview_end` $\to$ `publication_date` $\to$ `interview_start` $\to$ `sample_size` $\to$ `poll_id`.
3. **Sample-Size Weighting**:
   $$w = \text{clip}\left(\sqrt{\frac{n}{1000}}, 0.7, 1.5\right)$$
   (If sample size is unavailable, weight defaults to 1.0).
4. **Composition Calculation**:
   - Weighted average computed for parliamentary parties $\text{M, L, C, KD, S, V, MP, SD}$.
   - Residual category derived: $\text{REST} = 100.0 - \sum_{p \in \text{PARLIAMENTARY}} \text{PollConsensus}_p$.
   - Compositions strictly sum to $100.0000\%$.

---

## 3. Election Target Alignment & Residual Metrics

Targets are loaded from certified election returns in `data/processed/elections/riksdag_election_results.csv`:

$$\text{votes}_{\text{REST}} = \text{votes}_{\text{FI}} + \text{votes}_{\text{OTHER}}$$
$$\text{TargetShare}_p = \frac{\text{votes}_p}{\text{valid\_votes\_total}} \times 100$$

### Residual Formulations

* **Percentage-Point Residual ($r^{\text{pp}}$)**:
  $$r^{\text{pp}}_{E,p} = \text{Result}_{E,p} - \text{PollConsensus}_{E,p}$$
  *(Positive value indicates the party **outperformed polling** on election day).*

* **Compositional CLR Residual ($r^{\text{clr}}$)**:
  $$r^{\text{clr}}_{E,p} = \text{CLR}(\text{Result}_E)_p - \text{CLR}(\text{PollConsensus}_E)_p$$

---

## 4. Election-Level Residual Results

| Election | Window Dates | Total Polls | Retained Pollsters | 8-Party MAE | All-9 MAE | Max Miss Party (Diff) | REST Diff |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2002-09-15** | 2002-09-01 to 2002-09-15 | 44 | 5 (Sifo, Ipsos, Novus, Skop, Demoskop) | 1.60% | 1.51% | **S (+3.47 pp)** | -0.75 pp |
| **2006-09-17** | 2006-09-03 to 2006-09-17 | 23 | 5 (SVT, Sifo, Skop, Ipsos, Demoskop) | 0.70% | 0.64% | **S (+1.30 pp)** | -0.15 pp |
| **2010-09-19** | 2010-09-05 to 2010-09-19 | 27 | 7 (SVT, United Minds, Novus, Sifo, Skop, Ipsos, Demoskop) | 0.60% | 0.55% | **MP (-1.32 pp)** | -0.17 pp |
| **2014-09-14** | 2014-08-31 to 2014-09-14 | 23 | 9 (SVT, Skop, Novus, Demoskop, Sifo, United Minds, Ipsos, YouGov, Sentio) | 1.26% | 1.19% | **SD (+2.88 pp)** | -0.65 pp |
| **2018-09-09** | 2018-08-26 to 2018-09-09 | 35 | 10 (SVT, TV4, Skop, Inizio, Sifo, Novus, Demoskop, Ipsos, YouGov, Sentio) | 1.19% | 1.23% | **S (+3.17 pp)** | -1.59 pp |
| **2022-09-11** | 2022-08-28 to 2022-09-11 | 47 | 7 (SVT, Sifo, Skop, Demoskop, Ipsos, Novus, Sentio) | 0.77% | 0.72% | **S (+1.49 pp)** | +0.30 pp |

---

## 5. Party-Level Residual Summary (Across All 6 Elections)

| Party | Mean Residual ($r^{\text{pp}}$) | Median Residual | Std Dev | MAE | Sign Consistency | Mean CLR Diff | Consistent Directional Pattern |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **S** | **+1.97 pp** | **+1.40 pp** | 1.05 pp | 1.97% | **6+ / 0-** | +0.016 | **Systematic Outperformance**: Outperformed in 100% of elections. |
| **V** | **-1.06 pp** | **-1.01 pp** | 0.72 pp | 1.06% | **0+ / 6-** | -0.183 | **Systematic Underperformance**: Underperformed in 100% of elections. |
| **MP** | **-0.83 pp** | **-0.56 pp** | 0.66 pp | 0.83% | **0+ / 6-** | -0.173 | **Systematic Underperformance**: Underperformed in 100% of elections. |
| **SD** | **+0.66 pp** | **+0.60 pp** | 1.40 pp | 1.18% | **4+ / 2-** | +0.867 | **Regime Shift**: Under-polled in 2002–2014 (+2.88 pp max), slight over-polled in 2018–2022. |
| **M** | **+0.45 pp** | **+0.97 pp** | 1.77 pp | 1.41% | **5+ / 1-** | -0.030 | Outperformed in 5 of 6 elections (all except 2002). |
| **KD** | **-0.50 pp** | **-0.49 pp** | 0.48 pp | 0.58% | **1+ / 5-** | -0.120 | Underperformed in 5 of 6 elections. |
| **L** | **-0.35 pp** | **-0.37 pp** | 0.70 pp | 0.63% | **1+ / 5-** | -0.112 | Underperformed in 5 of 6 elections. |
| **C** | **+0.16 pp** | **-0.03 pp** | 0.66 pp | 0.49% | **3+ / 3-** | -0.021 | Balanced / Minimal systematic bias. |
| **REST** | **-0.50 pp** | **-0.41 pp** | 0.66 pp | 0.60% | **1+ / 5-** | -0.244 | Underperformed in 5 of 6 elections (minor party polling fade). |

---

## 6. Key Diagnostic Insights

### 1. Left/Green Bloc Internal Zero-Sum Transfer
* In all six elections, **S systematically outperformed polling** by an average of **+1.97 pp**, while **V (-1.06 pp)** and **MP (-0.83 pp)** systematically underperformed polling by a combined **-1.89 pp**.
* Net Left/Green bloc residual (S + V + MP) averaged only **+0.08 pp** across all elections, with virtually zero net bloc error in 2010 (-0.35 pp) and 2022 (+0.03 pp).
* This reveals an unmistakable **tactical voter consolidation mechanism**: voters who express support for V or MP in pre-election surveys disproportionately cast ballots for the prime ministerial anchor party (S) when entering the polling booth.

### 2. Threshold Dynamics (4.0% Benchmark)
* **Near-Threshold Cases** ($|\text{PollConsensus} - 4.0\%| \le 1.5\%$):
  - Average MAE is **0.62 pp** (compared to **1.09 pp** for parties away from threshold).
  - Absolute errors for threshold parties are smaller in percentage points because of their smaller baseline scale, but relatively critical in terms of parliamentary survival.
  - KD underperformed polling in 5 of 6 elections (averaging -0.50 pp), and L underperformed in 5 of 6 elections (averaging -0.35 pp).

### 3. SD Evolution: Taboo Under-Polling to Standard Polling
* From 2002 to 2014, SD consistently outperformed its final pre-election polling:
  - 2002: +1.44 pp (unpolled in standard reports)
  - 2006: +0.62 pp
  - 2010: +0.58 pp
  - 2014: +2.88 pp (major industry miss: polled 9.97% vs actual 12.86%).
* Following pollster methodological adjustments (weighting on past vote and online panels), SD was slightly **over-polled** in 2018 (-1.05 pp) and 2022 (-0.50 pp).

### 4. Difficulty Across Elections
* **2002** (MAE 1.60%) and **2018** (MAE 1.19%) / **2014** (MAE 1.26%) were the most difficult polling cycles.
* **2006** (MAE 0.70%), **2010** (MAE 0.60%), and **2022** (MAE 0.77%) showed remarkably high overall polling accuracy.

---

## 7. Limitations & Usage Guidelines

> [!NOTE]
> This study is strictly **diagnostic**. With only six election cycles ($N=6$), these residuals should not be used to hardcode arbitrary static bias offsets into future forecasting models, but rather to inform structured uncertainty distributions (such as correlated block covariance and threshold volatility).
