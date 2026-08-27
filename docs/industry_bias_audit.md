# Audit of Previous ElectionNoise Bias Experiments (Experiment 0 & 1)

This audit report documents the examination of previous historical poll-to-election residual experiments under the challenger-suite protocol. The protocol is described here for reproducibility; the repository does not claim independent preregistration without an immutable pre-result reference.

---

## 1. Audit Inquiries & Precise Findings

| Inquiry | Finding from `docs/election_layer_v2.md` & `scripts/election_layer_v2/` |
| :--- | :--- |
| **What `bias` meant** | The static chronological mean residual vector $\bar{\mathbf{r}} = \mu_E = \frac{1}{K}\sum_{k=1}^K \mathbf{r}_{e_k}$ computed strictly from prior elections $e < E$. |
| **What `raw` (`pp_bias_plus_noise`) meant** | The empirical historical residual bootstrap retaining both the chronological industry bias $\mu_E$ and the centered residual fluctuation: $\mathbf{r}^* = \mu_E + (\mathbf{r}_j - \mu_E)$. |
| **Which elections were used** | The 6 modern general elections with verified final polling: **2002, 2006, 2010, 2014, 2018, 2022**. |
| **Chronological / leakage safety** | **Strictly enforced**. Target elections were strictly excluded from training: <br>• 2010 evaluated with train $\{2002, 2006\}$ ($K=2$)<br>• 2014 evaluated with train $\{2002, 2006, 2010\}$ ($K=3$)<br>• 2018 evaluated with train $\{2002, 2006, 2010, 2014\}$ ($K=4$)<br>• 2022 evaluated with train $\{2002, 2006, 2010, 2014, 2018\}$ ($K=5$). |
| **Target election isolation** | **100% isolated**. No target election residuals ever entered their own bias estimation. |
| **Newer consensus data status** | The consensus dataset already utilizes the verified 14-day deduplicated, sample-weighted polling consensus with certified Valmyndigheten targets across all 6 modern elections. |

---

## 2. Empirical Performance Comparison

| Variant | 8-Party Mean CRPS | 2018 CRPS | 2022 CRPS | 90% Coverage (Width) | Result |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`pp_noise_only` (Centered Noise, RC1)** | **0.8001** | **0.9845** | **0.6158** | **76.9% (3.51 pp)** | **WINNER (Adopted into RC1)** |
| **`pp_bias_plus_noise` (Raw Industry Bias)** | **0.8568** | **1.0720** | **0.6415** | **80.6% (3.48 pp)** | Degraded (+7.1% worse CRPS) |
| **`pp_bias_only` (Static Bias Alone)** | **0.8886** | **1.1154** | **0.6618** | **63.0% (2.29 pp)** | Degraded (+11.1% worse CRPS) |
| **`base` (No Residual Layer)** | **0.8494** | **1.0623** | **0.6364** | **54.6% (2.29 pp)** | Under-dispersed |

### Key Takeaway
Adding the chronological industry bias $\mu_E$ degraded overall 8-party CRPS from **0.8001** to **0.8568** (+7.1% error) and degraded performance across **both** 2018 (0.9845 $\to$ 1.0720) and 2022 (0.6158 $\to$ 0.6415) individually. Party-specific polling biases vary across political eras; imposing historical mean shifts introduces out-of-sample directional error.

---

## 3. Formal Decision

```text
========================================================================================
EXPERIMENT 0 AUDIT: IDENTICAL TO PREVIOUS RIGOROUS EVALUATION
EXPERIMENT 1 INDUSTRY BIAS CHALLENGER: ALREADY_REJECTED
========================================================================================
```
Experiment 1 is closed as a historical comparison of the listed ElectionNoise variants. Its result supports retaining the centered-noise choice for the evaluated cases; it does not establish that every possible bias model has been searched. The separate **Experiment 2 (Empirical Pollster Precision Weighting in OpinionState)** remains an experiment-only candidate and was evaluated under its own gate.
