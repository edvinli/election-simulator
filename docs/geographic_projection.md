# Geographic Projection v1 (IPF / Raking)

## 1. Executive Summary

`GeographicProjection v1` translates simulated 9-category national vote totals (`M, L, C, KD, S, V, MP, SD, REST`) into plausible constituency-level vote matrices across all 29 Swedish Riksdag constituencies.

The algorithm uses deterministic **Iterative Proportional Fitting (IPF / Biproportional Raking)** anchored to a historical baseline election matrix ($B_{c,p}$).

In historical forward evaluations ($2014 \to 2018$ and $2018 \to 2022$), deterministic IPF achieves:
* **Constituency party-share MAE**: **0.65% – 0.67%**
* **National share reproduction error**: $< 10^{-6}$ (exact conservation)
* **Total Riksdag Seat Error**: **0 seats** (exact certified seat reproduction for all parties in both Oracle and Production modes).

These are retrospective historical forward-evaluation results for the stated
2014→2018 and 2018→2022 fixtures. They do not certify every cell-level
tie-break outcome in arbitrary forecast scenarios; the simulator's separate
freeze audit reports national-seat agreement and cell-level diagnostics
explicitly.

---

## 2. Mathematical Methodology (IPF / Raking)

### 2.1 Problem Formulation
Let:
* $B_{c,p} \ge 0$: Baseline constituency $\times$ party vote count (from previous election, e.g. 2014 for 2018, 2018 for 2022).
* $R_c > 0$: Target total valid votes in constituency $c \in \{1 \dots 29\}$.
* $C_p \ge 0$: Target total national votes for party $p \in \{1 \dots 9\}$.
* Conservation invariant: $\sum_{c=1}^{29} R_c = \sum_{p=1}^9 C_p = T_{\text{total}}$.

The biproportional scaling problem seeks multipliers $a_c$ and $b_p$ such that:
$$X_{c,p} = a_c B_{c,p} b_p$$
subject to:
$$\sum_{p=1}^9 X_{c,p} = R_c \quad \forall c \in \{1 \dots 29\}$$
$$\sum_{c=1}^{29} X_{c,p} = C_p \quad \forall p \in \{1 \dots 9\}$$

### 2.2 Numerical Algorithm
Initialize $X^{(0)} = B$.
Iterate for $k = 0, 1, 2, \dots$:
1. **Row Step (Constituency Scaling)**:
   $$X_{c,p}^{(k+1/2)} = X_{c,p}^{(k)} \times \frac{R_c}{\sum_q X_{c,q}^{(k)}}$$
2. **Column Step (National Party Scaling)**:
   $$X_{c,p}^{(k+1)} = X_{c,p}^{(k+1/2)} \times \frac{C_p}{\sum_d X_{d,p}^{(k+1/2)}}$$
3. **Convergence Criterion**:
   $$\max\left(\max_c \left|\sum_p X_{c,p}^{(k+1)} - R_c\right|, \max_p \left|\sum_c X_{c,p}^{(k+1)} - C_p\right|\right) < 10^{-8}$$

Convergence is deterministic and typically achieved in **9 to 10 iterations**.

---

## 3. REST Category Handling

`REST` is **not a single political party**. It represents the aggregate of all minor non-parliamentary parties (e.g. Medborgerlig Samling, Alternativ för Sverige, Piratpartiet, local lists).

1. **In Geographic Projection**: `REST` is treated as a modeled category in IPF, receiving proportional geographic allocation based on historical minor-party strength.
2. **In Mandate Allocation**: `REST` is automatically mapped to `OTHER_INELIGIBLE`:
   * Its votes contribute to valid-vote denominators in constituencies and nationally.
   * It is strictly ineligible for the 4% national threshold and 12% constituency threshold.
   * It never receives seats.

---

## 4. Constituency Vote Total Modes

### 4.1 Oracle Mode
Uses the actual certified valid vote total $R_c$ for each constituency in the target election. This mode isolates the pure geographic party-distribution error from turnout volume error.

### 4.2 Production Mode
Predicts constituency valid votes $R_c^{\text{target}}$ prior to the election using the target electorate and prior turnout rate:
$$R_c^{\text{target}} = \text{Eligible}_c^{\text{target}} \times \frac{\text{ValidVotes}_c^{\text{previous}}}{\text{Eligible}_c^{\text{previous}}}$$

---

## 5. Historical Forward Evaluations

### 5.1 Evaluation: 2014 Geography $\to$ 2018 Election
* **National Votes Target**: Actual certified 2018 national party votes ($N=6,476,725$).
* **Baseline**: 2014 certified constituency $\times$ party matrix.

| Metric | Oracle Mode | Production Mode |
|---|---|---|
| **IPF Iterations** | 10 | 10 |
| **Constituency Party-Share MAE** | **0.656%** | **0.656%** |
| **Constituency Volume MAPE** | 0.00% | 1.67% (Max: 10,913 votes) |
| **National Share Error** | $3.09 \times 10^{-7}$ | $7.52 \times 10^{-7}$ |
| **Seat Errors by Party** | M: 0, L: 0, C: 0, KD: 0, S: 0, V: 0, MP: 0, SD: 0 | M: 0, L: 0, C: 0, KD: 0, S: 0, V: 0, MP: 0, SD: 0 |
| **Total Absolute Seat Error** | **0 seats** | **0 seats** |

#### Party-Level Constituency Share MAEs (2014 $\to$ 2018):
* **M**: 1.15%
* **S**: 0.90%
* **C**: 0.98%
* **SD**: 0.83%
* **KD**: 0.73%
* **V**: 0.54%
* **REST**: 0.34%
* **L**: 0.24%
* **MP**: 0.21%

---

### 5.2 Evaluation: 2018 Geography $\to$ 2022 Election
* **National Votes Target**: Actual certified 2022 national party votes ($N=6,477,970$).
* **Baseline**: 2018 certified constituency $\times$ party matrix.

| Metric | Oracle Mode | Production Mode |
|---|---|---|
| **IPF Iterations** | 9 | 9 |
| **Constituency Party-Share MAE** | **0.670%** | **0.671%** |
| **Constituency Volume MAPE** | 0.00% | 3.42% (Max: 43,120 votes) |
| **National Share Error** | $4.63 \times 10^{-7}$ | $5.30 \times 10^{-7}$ |
| **Seat Errors by Party** | M: 0, L: 0, C: 0, KD: 0, S: 0, V: 0, MP: 0, SD: 0 | M: 0, L: 0, C: 0, KD: 0, S: 0, V: 0, MP: 0, SD: 0 |
| **Total Absolute Seat Error** | **0 seats** | **0 seats** |

#### Party-Level Constituency Share MAEs (2018 $\to$ 2022):
* **S**: 1.33%
* **SD**: 1.07%
* **M**: 0.95%
* **V**: 0.67%
* **C**: 0.63%
* **KD**: 0.58%
* **MP**: 0.38%
* **L**: 0.23%
* **REST**: 0.22%

---

## 6. Primary Practical Findings

> **Question**: If the future national vote result were known exactly, how much seat error is introduced solely by the geographic projection?

**Finding**: **Zero seat error**.
In both the 2018 and 2022 elections, deterministic IPF raking on prior-election geographic baselines introduces **0 total seat error** when evaluated against certified Riksdag outcomes.

### Why Deterministic Raking is Exceptionally Accurate in Sweden
1. **Constitutional Proportionality**: 39 adjustment seats (utjämningsmandat) absorb minor constituency-level swing differences and restore strict nationwide proportionality among qualifying parties.
2. **Persistence of Regional Relative Strength**: Relative regional party bastions (e.g. S in Norrland, M/L in Stockholm, KD in Jönköping, SD in Skåne) shift gradually over 4-year cycles.
3. **No Overhang Distortion**: Because no party reached an overhang threshold in 2018 or 2022, national party seat totals were dictated entirely by national proportional entitlement.

### Conclusion on Geographic Residual Uncertainty
Deterministic geography via IPF is **fully adequate** for the baseline Riksdag simulator. Additional geographic stochastic noise is unnecessary and would introduce artificial variance without improving national seat calibration.
