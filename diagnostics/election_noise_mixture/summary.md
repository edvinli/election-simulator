# Diagnostic: the multimodal C+S+MP coalition-seat distribution

**Status: diagnostic only.** No production component, parameter, seed, dataset or
published artifact was modified. `main` is untouched, nothing was published,
`current.json` was not moved, and no prospective snapshot was regenerated.

Branch: `diagnostic/election-noise-mixture`.

---

## 0. What the repository says the production forecast is

Established from the repository itself, not from the task description:

| Item | Value | Source |
|---|---|---|
| Canonical published generation | `20260828T201250Z-1da59168` | `files/election-simulator/current.json` |
| `as_of` | **2026-08-24** | `versions/.../metadata.json` (= max date in `pollofpolls_timeseries.csv`) |
| Election date | **2026-09-13** | `metadata.json`, `scripts/simulator/config.py::DEFAULT_ELECTION_DATE` |
| Samples | **100 000** | `forecast.json::total_samples`, `DEFAULT_SIMULATION_SAMPLES` |
| Base seed | **12345** | `DEFAULT_SIMULATION_SEED`, `publication_pipeline` default, archive index |
| ElectionNoise model | `pp_centered_noise` | `scripts/vote_share_calibration/models.py::apply_vote_share_models` |
| Residual pool | `load_chronological_pp_residuals(target_election_year=2026)` | `scripts/election_layer_v2/residuals_pool.py` |
| Eligible residual elections | **2002, 2006, 2010, 2014, 2018, 2022 (K = 6)** | all of `ALL_HISTORICAL_ELECTIONS` with `year < 2026` |
| Residual index sampled at | `models.py:apply_vote_share_models` → `np.random.default_rng(index_seed).integers(0, K, size=N)` | uniform, one index per draw |
| Geography + mandates | `scripts/simulator/engine.py::simulate_election` (IPF → biproportional controlled rounding → `dispatch_production_allocation`) | fully deterministic given the national 9-vector |

The published artifact was produced at commit `2697c18`; `HEAD` (`34c52d6`) only
blanked `retrieved_at` / `metadata_retrieved_at` provenance columns in the
processed polling CSVs. Column-wise comparison of `pollofpolls_timeseries.csv`
and `swedishpolls_individual_polls.csv` between `2697c18` and `HEAD` shows the
files are identical once `retrieved_at` is excluded, so no model input changed.
This is confirmed empirically in §1.

**Party order (index → party), used by the published `coalition_builder`
bitmask and by `seats_matrix`:** `M(0) L(1) C(2) KD(3) S(4) V(5) MP(6) SD(7)`.
`C+S+MP` = mask 84 = columns {2, 4, 6}. `S+V+MP` = mask 112 = columns {4, 5, 6}.

---

## 1. Reproduction of the published 100k forecast — exact

`diagnostics/election_noise_mixture/run_simulation.py --mode production` calls the
frozen `simulate_election` unchanged. Instrumentation is passive:

* a wrapper around `apply_vote_share_models` that records its **inputs**
  (`base_comp_matrix`, the pool, the seeds) and draws no randomness of its own;
* the residual index is **reconstructed**, not re-drawn, as
  `np.random.default_rng(index_seed).integers(0, 6, N)` — production computes it on
  a dedicated generator instance, so recomputing it outside cannot perturb any stream.

`diagnostics/election_noise_mixture/test_instrumentation.py` (5 tests, all pass)
proves the reconstruction is the index production actually consumed: feeding
`centered_residuals_matrix[reconstructed_index]` through
`apply_batch_simplex_transfer` reproduces production's `pp_centered_noise` output
and its λ vector under `assert_array_equal` (bit-for-bit, not approximately).

Result of the instrumented 100 000-draw run at `as_of=2026-08-24`, `seed=12345`:

| Coalition | published mean | reproduced | published median | reproduced | published P(≥175) | reproduced | histogram |
|---|---|---|---|---|---|---|---|
| C+S+MP | 162.71929 | 162.71929 | 162 | 162 | 0.10778 (10 778) | 0.10778 (10 778) | **exact match** |
| S+V+MP | 164.52760 | 164.52760 | 165 | 165 | 0.02216 (2 216) | 0.02216 (2 216) | **exact match** |

All **256** published `coalition_builder` seat histograms match bin-for-bin
(`max_abs_bin_difference = 0`, `mismatched_masks = []`). Reproduction is exact,
not Monte-Carlo-close, so every number below describes the published forecast.

---

## 2. Conditional decomposition by sampled residual election

Full table (with p05/p10/p25/p75/p90/p95, min, max) in
`conditional_by_residual_year.csv`.

### C+S+MP — overall mean 162.72, median 162, p90 **175**, p95 177, min 141, max 190

| residual year | n | P(year) | mean | median | p05 | p95 | min | max | count ≥175 | P(≥175 \| year) | share of all majority draws | P(year)·P(≥175\|year) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2002 | 16 616 | 0.16616 | **175.66** | **176** | 171 | 181 | 161 | 190 | **10 742** | **64.649 %** | **99.666 %** | 0.107420 |
| 2006 | 16 504 | 0.16504 | 163.06 | 163 | 158 | 168 | 151 | 176 | 4 | 0.0242 % | 0.037 % | 0.000040 |
| 2010 | 16 883 | 0.16883 | 158.28 | 158 | 153 | 163 | 146 | 170 | 0 | 0 % | 0 % | 0.000000 |
| 2014 | 16 611 | 0.16611 | 153.40 | 153 | 148 | 158 | 141 | 169 | 0 | 0 % | 0 % | 0.000000 |
| 2018 | 16 868 | 0.16868 | 165.64 | 166 | 161 | 171 | 154 | 177 | 32 | 0.1897 % | 0.297 % | 0.000320 |
| 2022 | 16 518 | 0.16518 | 160.29 | 160 | 155 | 165 | 147 | 172 | 0 | 0 % | 0 % | 0.000000 |
| **total** | 100 000 | 1.0 | 162.72 | 162 | 152 | 177 | 141 | 190 | **10 778** | **10.778 %** | 100 % | **0.107780** ✓ |

> **10 742 / 10 778 = 99.666 % of every C+S+MP majority simulation in the
> published forecast comes from the single residual year 2002.**
> Three of the six residual years produce **zero** majority draws out of ~16 700 each.

### S+V+MP — overall mean 164.53, median 165, p90 171, p95 173, min 144, max 188

| residual year | n | P(year) | mean | median | p05 | p95 | min | max | count ≥175 | P(≥175 \| year) | share of all majority draws | P(year)·P(≥175\|year) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2002 | 16 616 | 0.16616 | **171.26** | 171 | 167 | 176 | 160 | 188 | **2 148** | 12.927 % | **96.931 %** | 0.021480 |
| 2006 | 16 504 | 0.16504 | 165.38 | 165 | 161 | 170 | 153 | 177 | 20 | 0.1212 % | 0.903 % | 0.000200 |
| 2010 | 16 883 | 0.16883 | 163.81 | 164 | 159 | 169 | 153 | 176 | 3 | 0.0178 % | 0.135 % | 0.000030 |
| 2014 | 16 611 | 0.16611 | 156.09 | 156 | 151 | 161 | 144 | 170 | 0 | 0 % | 0 % | 0.000000 |
| 2018 | 16 868 | 0.16868 | 164.78 | 165 | 160 | 169 | 153 | 176 | 7 | 0.0415 % | 0.316 % | 0.000070 |
| 2022 | 16 518 | 0.16518 | 165.86 | 166 | 161 | 171 | 155 | 178 | 38 | 0.2301 % | 1.715 % | 0.000380 |
| **total** | 100 000 | 1.0 | 164.53 | 165 | 155 | 173 | 144 | 188 | **2 216** | **2.216 %** | 100 % | **0.022160** ✓ |

Both unconditional decompositions sum exactly to the published probability.

---

## 3. Is the second mode present before ElectionNoise? No.

`--mode prenoise` substitutes the `OpinionState + Dynamics` composition
(`base_comp_matrix`, normalised by its own row sum — exactly the normalisation
production applies after the transfer) for the post-noise composition and then
runs the **same** geography, integerisation and exact mandate allocator. Both
sides below are seat distributions from the same 100 000 base draws.

| | C+S+MP pre | C+S+MP final | S+V+MP pre | S+V+MP final |
|---|---|---|---|---|
| mean | 162.704 | 162.719 | 164.512 | 164.528 |
| median | 163 | 162 | 165 | 165 |
| SD | **3.06** | **7.58** | **2.87** | **5.30** |
| p05 / p95 | 158 / 168 | 152 / 177 | 160 / 169 | 155 / 173 |
| min / max | 148 / 178 | 141 / 190 | 152 / 179 | 144 / 188 |
| count ≥175 | **10** | **10 778** | **43** | **2 216** |
| P(≥175) | **0.010 %** | **10.778 %** | **0.043 %** | **2.216 %** |

The pre-ElectionNoise distribution is unimodal and narrow. ElectionNoise leaves
the mean unchanged (it is mean-zero by construction) and multiplies the SD by
2.5× / 1.8×, raising C+S+MP's majority probability by a factor of **≈ 1 078**.

**Variance decomposition of the final distribution:**

| | total var | between residual years | within residual year | between share |
|---|---|---|---|---|
| C+S+MP | 57.39 | 48.03 | 9.36 | **83.7 %** |
| S+V+MP | 28.13 | 19.87 | 8.25 | **70.7 %** |

Conditional SDs are 3.00–3.09 (C+S+MP) and 2.82–2.90 (S+V+MP) — i.e. essentially
the pre-noise SD in every branch. **ElectionNoise contributes almost no extra
within-branch spread; it contributes six discrete location shifts.** The final
distribution is a six-component location mixture of nearly identically shaped
components.

**Placebo check.** Conditioning the *pre-noise* seats on the (unused) residual
year gives means 162.69, 162.70, 162.69, 162.70, 162.73, 162.72 — no separation.
The strata are exchangeable until the residual is applied, which is what makes the
conditional decomposition in §2 causally interpretable.

---

## 4. Vote-space mechanism (`coalition_residual_shocks.csv`)

Centered ElectionNoise shocks, in percentage points, using exactly production's
centering (`r_e − mean_e r_e`, then the zero-sum cleaning in
`load_chronological_pp_residuals`). Mean bias removed (pp):
M +0.452, L −0.351, C +0.157, KD −0.500, S +1.973, V −1.056, MP −0.834, SD +0.662, REST −0.503.

| year | C | S | V | MP | **C+S+MP** | **S+V+MP** | M | L | KD | SD | REST |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **2002** | +0.871 | **+1.502** | −0.816 | +0.718 | **+3.091** | +1.404 | **−3.335** | +1.195 | −0.664 | +0.776 | −0.246 |
| 2006 | +0.685 | −0.670 | +0.826 | +0.309 | +0.325 | +0.465 | −0.322 | −0.836 | −0.296 | −0.047 | +0.350 |
| 2010 | −0.294 | −0.668 | +0.725 | −0.487 | −1.449 | −0.429 | +0.127 | +0.193 | +0.154 | −0.079 | +0.329 |
| 2014 | −0.314 | −0.870 | −0.070 | −1.085 | **−2.269** | **−2.025** | +0.915 | −0.518 | −0.135 | +2.222 | −0.145 |
| 2018 | −0.079 | +1.192 | −0.817 | +0.266 | +1.380 | +0.641 | +1.611 | −0.116 | +0.748 | −1.714 | −1.092 |
| 2022 | −0.869 | −0.486 | +0.153 | +0.279 | −1.077 | −0.055 | +1.004 | +0.082 | +0.193 | −1.159 | +0.804 |

Two facts explain everything:

1. **C+S+MP's shock range is 5.36 pp (−2.27 … +3.09); S+V+MP's is 3.43 pp
   (−2.03 … +1.40).** Swapping V for C is what widens it. V's centered residual
   is −0.816 in 2002 and −0.817 in 2018 — the only two S-positive years — so V
   cancels roughly half of S's gain in exactly the years that matter for the upper
   tail. C's residual in 2002 is +0.871, i.e. it *adds* where V subtracts: a
   1.69 pp swing between the two coalitions inside the single decisive year.
2. **2002 is an outlier in the pool.** No other year's C+S+MP shock exceeds
   +1.4 pp; the next-largest is 2018 at +1.380 against 2002's +3.091. In seat
   terms that 1.71 pp gap becomes a 10-seat gap between the top branch (175.66)
   and the second branch (165.64).

### The pp → seats map is essentially linear here

Treating the coalition's seat share as `349 · c/E` with `E` = vote mass of parties
above 4 % (mean 95.98 pp pre-noise), the first-order prediction
`349·(Δc/E − c·ΔE/E²)` reproduces every realised branch shift:

| year | C+S+MP shock (pp) | ΔE (pp) | predicted Δseats | realised Δseats | ratio |
|---|---|---|---|---|---|
| 2002 | +3.091 | −0.948 | +12.85 | **+12.96** | 1.01 |
| 2006 | +0.325 | +0.486 | +0.36 | +0.36 | 1.00 |
| 2010 | −1.449 | −0.521 | −4.39 | −4.42 | 1.01 |
| 2014 | −2.269 | +0.663 | −9.37 | −9.31 | 0.99 |
| 2018 | +1.380 | +1.207 | +2.97 | +2.93 | 0.99 |
| 2022 | −1.077 | −0.887 | −2.41 | −2.42 | 1.00 |

(S+V+MP ratios are 0.99–1.06; full table in `diagnostic_report.json` →
`vote_to_seat_first_order_map`.)

2002's +12.96-seat shift is amplified beyond the naive 3.6 seats/pp because the
2002 residual also moves **+1.195 pp into L**, which stays below 4 % in 99.99 % of
draws. That vote is wasted, shrinking the eligible denominator `E` and raising
everyone else's seats-per-vote. This is a real effect of the composition, not a
nonlinearity of geography or the allocator — the linear map already captures it.

As instructed, this is explanatory evidence, not an identity. What it *does*
establish is the negative result needed for §6: geography, integerisation, the
Sainte-Laguë allocator and the 4 % threshold add ≲2 % correction to branch
locations and therefore cannot be the source of the second mode.

---

## 5. Sensitivity to the six-election bootstrap (`reweighting_sensitivity.csv`)

### 5a. Conditional reweighting diagnostic — **not a new model**

Purely descriptive: drop the draws assigned a given residual year from the
completed production run and renormalise the remainder. This changes only the
estimator's weighting, not the model.

| removed year | C+S+MP P(≥175) | Δ (pp) | rel. | S+V+MP P(≥175) | Δ (pp) | rel. |
|---|---|---|---|---|---|---|
| — (production) | 10.778 % | — | — | 2.216 % | — | — |
| **2002** | **0.0432 %** | **−10.73** | **−99.6 %** | **0.0816 %** | **−2.13** | **−96.3 %** |
| 2006 | 12.904 % | +2.13 | +19.7 % | 2.630 % | +0.41 | +18.7 % |
| 2010 | 12.967 % | +2.19 | +20.3 % | 2.663 % | +0.45 | +20.1 % |
| 2014 | 12.925 % | +2.15 | +19.9 % | 2.657 % | +0.44 | +19.9 % |
| 2018 | 12.926 % | +2.15 | +19.9 % | 2.657 % | +0.44 | +19.9 % |
| 2022 | 12.911 % | +2.13 | +19.8 % | 2.609 % | +0.39 | +17.7 % |

Removing any year other than 2002 simply re-weights the surviving 2002 draws from
1/6 to 1/5 (+20 %). Removing 2002 removes the phenomenon.

### 5b. True leave-one-election-out re-runs (6 × 100 000 draws)

These required **no** production-code change. `ALL_HISTORICAL_ELECTIONS` in
`scripts/election_layer_v2/residuals_pool` is filtered to five elections and the
production loader itself performs the re-centering (mean over the remaining five,
plus its zero-sum cleaning). Everything else — seed 12345, `as_of`, data,
OpinionState, Dynamics, transfer, geography, allocator — is unchanged. K drops
from 6 to 5, so the index stream necessarily differs; this is the faithful
"same algorithm, smaller pool" counterfactual, not a seed change.

| removed year | C+S+MP P(≥175) | rel. | median | S+V+MP P(≥175) | rel. | median |
|---|---|---|---|---|---|---|
| — (production) | 10.778 % | — | 162 | 2.216 % | — | 165 |
| **2002** | **0.392 %** | **−96.4 %** | 163 | **0.261 %** | **−88.2 %** | 165 |
| 2006 | 13.064 % | +21.2 % | 161 | 2.861 % | +29.1 % | 165 |
| 2010 | 10.664 % | −1.1 % | 162 | 2.458 % | +10.9 % | 165 |
| **2014** | **8.022 %** | **−25.6 %** | 161 | **0.857 %** | **−61.3 %** | 164 |
| 2018 | 14.175 % | +31.5 % | 161 | 2.683 % | +21.1 % | 165 |
| 2022 | 11.741 % | +8.9 % | 162 | 3.094 % | +39.6 % | 165 |

The LOO numbers differ from the reweighting numbers for a principled reason.
Re-centering on five elections gives, for each surviving year *e*,
`c'_e = c_e + c_j/5` where *j* is the removed year: **every surviving branch moves
by one fifth of the removed year's centered residual.** Hence

* removing 2002 (`c = +3.091`) lifts all five survivors by +0.618 pp ≈ +2.6 seats,
  which is why C+S+MP retains 0.392 % rather than the 0.043 % of pure reweighting;
* removing **2014** (`c = −2.269`) lowers all survivors by −0.454 pp ≈ −1.9 seats
  and cuts C+S+MP from 10.78 % to 8.02 % and S+V+MP from 2.22 % to 0.86 % — even
  though the 2014 branch itself contributed **zero** majority draws.

The overall mean is invariant across all LOO runs (162.68–162.71), as it must be:
a centered pool is mean-zero whatever its size. The tail probability is not.

---

## 6. Implementation-bug audit

Every item requested was checked. **No bug was found.** Nothing below is called a
bug, because nothing below could be demonstrated as one.

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Wrong coalition bitmask / party selection | **Ruled out** | `parties_8` order equals the published `coalition_builder.party_order` exactly; C+S+MP uses columns {2,4,6}, S+V+MP {4,5,6}; both reproduce the published mask-84 / mask-112 histograms bin-for-bin. |
| Summing marginal medians instead of joint draws | **Ruled out** | Coalition seats are `seats_matrix[:, cols].sum(axis=1)` per draw. Joint median S+V+MP = 165 vs sum-of-marginal-medians = 164 — they differ, and the joint value is the published one. |
| Non-uniform residual-year sampling | **Ruled out** | counts 16 616 / 16 504 / 16 883 / 16 611 / 16 868 / 16 518; χ² = 8.49 on 5 df (p ≈ 0.13). Consistent with uniform. |
| Duplicate residual years / vectors | **Ruled out** | `training_years = (2002, 2006, 2010, 2014, 2018, 2022)`, all distinct; six distinct residual vectors. |
| Residual centering error | **Ruled out** | column means of the centered matrix ≤ 1.1e−16 pp; every row sums to 0 within 5.3e−15 pp; raw rows also zero-sum. |
| Incorrect party ordering | **Ruled out** | `ALL_CATEGORIES` (M, L, C, KD, S, V, MP, SD, REST) is used consistently by the residual loader, the transfer and the engine; the realised per-stratum mean vote shift equals the year's centered residual vector componentwise (below). |
| RNG / index ↔ seat-row misalignment | **Ruled out** | Within each residual-year stratum, mean(final − base) equals that year's centered residual to ≤ 1.05e−12 pp for 2002, 2006, 2010, 2014, 2022. The 2018 stratum deviates by 0.0137 pp — fully accounted for by λ-attenuation (next row). Geography and allocation are deterministic given the national vector, so row *i*'s residual and row *i*'s seats cannot desynchronise. |
| Clipping / simplex-transfer discontinuity | **Ruled out as a cause** | λ < 1 in only 0.981 % of all draws (mean λ = 0.99865). It occurs **exclusively in the 2018 branch** (5.82 % of that branch, min λ = 0.439), and the binding donor is always **REST** (2018's REST residual is −1.092 pp against a base REST that can fall to 0.40 pp). λ < 1 *shrinks* a shock toward zero, so it pulls the 2018 branch toward the centre. In the 2002 branch λ ≡ 1.0 exactly. Attenuation cannot create the upper mode. |
| Geography / allocator creating the second mode | **Ruled out** | §4: a first-order `349·(Δc/E − c·ΔE/E²)` vote-space map predicts every branch's realised seat shift to within 1 % (2002: +12.85 predicted vs +12.96 realised). The mode separation exists in vote space before any seat machinery. |
| Threshold effects (any party near 4 %) | **Ruled out as a cause** | Only L and KD ever cross: L ≥ 4 % in 10 of 100 000 draws (0.01 %), KD < 4 % in 10 draws (0.01 %). C min 4.37 %, MP min 4.42 %, V min 5.21 %, S min 25.11 % — no coalition member is ever near the threshold. 20 boundary draws cannot produce a 10 778-draw mode. (REST exceeds 4 % in 2.36 % of draws but is modelled as ineligible by design — documented in `docs/election_simulator.md` §5, unrelated to this question.) |
| Histogram aggregation / binning artifact | **Ruled out** | The support is integer seats and every histogram is a raw per-seat count with no binning. Published counts and reproduced counts agree exactly. The trough is a genuine density feature: C+S+MP densities run 168 → 2.427 %, 169 → 1.822 %, 170 → 1.289 %, **171 → 1.160 % (antimode)**, 172 → 1.398 %, 174 → 1.960 %, **176 → 2.131 % (secondary mode)**. |

Other invariants confirmed on the 100 000-draw run: every draw allocates exactly
349 seats; every national vote row sums to 100 % within 4.3e−14; no parliamentary
party is ever pinned at the ε = 0.01 % floor.

---

## 7. Direct answers

**Is C+S+MP genuinely multimodal?**
Yes. On the raw integer-seat support the density falls monotonically from the
primary mode at 161 (6.113 %) to an antimode at **171 (1.160 %)**, then rises again
to a secondary mode at **176 (2.131 %)** — 1.84× the trough — before falling away.
It is not a binning artifact and not Monte Carlo noise: the trough and the
secondary mode differ by 971 draws per bin at N = 100 000. A 3-bin smoothed
peak-finder returns local maxima at {162, 176} for C+S+MP and {165} only for
S+V+MP. S+V+MP's density is strictly decreasing above its mode at 165
(9.457 % → 0.997 % at 175) with no secondary rise anywhere.

**Is the multimodality already present before ElectionNoise?**
No. The pre-ElectionNoise seat distribution — same base draws, same geography,
same exact allocator — is unimodal with SD 3.06 seats and P(≥175) = 0.010 %.
The second mode is created entirely by the ElectionNoise layer.

**Which historical residual year(s) create the upper mode?**
**2002, alone.** Its centered C+S+MP shock is +3.091 pp, which maps to +12.96
seats and places its conditional distribution at mean 175.66 / median 176 —
straddling the majority line. The next-highest branch (2018, +1.380 pp) sits at
165.6, nine seats lower. Because within-branch SD is only ~3.1 seats, the 2002
component barely overlaps the other five, and the mixture separates.

**Of all C+S+MP majority simulations, what percentage comes from each residual year?**

| 2002 | 2018 | 2006 | 2010 | 2014 | 2022 |
|---|---|---|---|---|---|
| **10 742 / 10 778 = 99.666 %** | 32 / 10 778 = 0.297 % | 4 / 10 778 = 0.037 % | 0 % | 0 % | 0 % |

For S+V+MP: 2002 **2 148 / 2 216 = 96.931 %**, 2022 1.715 %, 2006 0.903 %,
2018 0.316 %, 2010 0.135 %, 2014 0 %.

**Why does S+V+MP behave differently?**
Not because it is better centred — it is not; its mean (164.53) is *higher* than
C+S+MP's (162.72). Two reasons:

1. **Smaller residual spread.** In 2002 — the decisive year — V's centered
   residual is −0.816 while C's is +0.871, so substituting C for V moves the
   coalition shock by +1.69 pp in that year alone. The C+S+MP shock range is
   5.36 pp against 3.43 pp for S+V+MP; final SD 7.58 vs 5.30 seats.
2. **Where the top branch lands.** The 2002 branch sits at 171.26 for S+V+MP —
   3.7 seats *below* the threshold — so only its upper tail crosses
   (P = 12.9 %, 2 148 draws). For C+S+MP the same branch sits at 175.66, *above*
   the threshold, so most of it crosses (P = 64.6 %, 10 742 draws).

The two coalitions are therefore driven by the *same* single historical year; they
differ in whether that year's branch clears 175 in the middle or in the tail.
S+V+MP's smaller and lower shock also means its mixture components overlap enough
to stay unimodal.

**How sensitive are 10.78 % and 2.22 % to individual historical residual years?**
Extremely, and only to 2002 and (via re-centering) 2014.

* Reweighting away 2002: 10.778 % → 0.043 % (−99.6 %) and 2.216 % → 0.082 % (−96.3 %).
* True leave-2002-out re-run: 10.778 % → **0.392 %** (−96.4 %) and 2.216 % → **0.261 %** (−88.2 %).
* True leave-2014-out re-run (a year contributing zero majority draws):
  10.778 % → **8.022 %** (−25.6 %) and 2.216 % → **0.857 %** (−61.3 %).
* Across the six LOO re-runs, C+S+MP's majority probability spans **0.39 % – 14.18 %**
  and S+V+MP's spans **0.26 % – 3.09 %**.

The point estimates 10.78 % and 2.22 % are stable to *sampling* noise (the run is
exactly reproducible) but not to the *composition of the six-election pool*.

**Is there any evidence of an implementation bug?**
No. Every hypothesis in §6 was tested and none survived. The reproduction is
bit-exact against the published contract across all 256 coalitions; centering is
exact to machine precision; sampling is uniform; the index-to-seat mapping is
verified componentwise; λ-attenuation is rare, confined to one branch, and works
in the wrong direction to explain the mode; threshold flips number 20 draws in
100 000. **The behaviour is what this code is specified to do.**

**Does this reveal a meaningful model fragility even if the implementation is correct?**
Yes, and it is worth stating plainly. `pp_centered_noise` is a six-atom empirical
distribution. With K = 6 the layer cannot produce a smooth error distribution; it
produces six point shifts. When one atom's projection onto a coalition is an
outlier — as 2002's +3.091 pp is — and the within-atom spread (~3 seats) is small
relative to the between-atom spread (~13 seats to the next branch), the mixture
separates visibly and the tail probability becomes, in effect, a statement about
one election.

The concrete consequences:

* C+S+MP's published **p90 is exactly 175** — the reported 90th percentile sits on
  the decision boundary, entirely because the 2002 branch carries 16.6 % of the mass.
* "P(C+S+MP majority) = 10.8 %" is arithmetically close to
  "P(the 2002-type polling error recurs) × P(majority | 2002-type error)"
  = 1/6 × 64.6 % = 10.74 %. The headline number is dominated by the assumption that
  a 2002-magnitude error has probability exactly 1/6.
* The tail is also sensitive through re-centering to years that contribute no
  majority draws at all (2014), because the centering constant is estimated from
  the same six observations.
* Each atom carries 1/6 = 16.7 % probability with a standard error of ±1.5 pp from
  six observations. Nothing in the layer smooths, shrinks, or interpolates between
  the six observed error patterns; a 2026 error that is a mixture or a scaled
  version of two historical patterns has probability zero under this layer.

This is model-assumption sensitivity, not a defect. It was a deliberate,
documented and empirically validated choice: `docs/election_layer_v2.md` shows
`pp_noise_only` improved out-of-sample CRPS in all four forward-evaluated
elections and lifted 90 % coverage from 54.6 % to 76.9 %. Those validations scored
*marginal party* distributions, where six atoms average out well. The diagnostic
above shows that the same six atoms behave very differently for a *coalition tail
probability*, where a single atom can straddle a threshold. Marginal calibration
does not transfer to a threshold functional of a sum.

**No change is proposed here.** Smoothing, shrinkage, recency weighting, or a
parametric joint residual model would each be a different model and would need a
separate preregistered task with its own out-of-sample evaluation.

---

## 8. Files

| File | Contents |
|---|---|
| `summary.md` | this document |
| `conditional_by_residual_year.csv` | overall + per-residual-year stats for both coalitions (n, mean, median, p05/p10/p25/p75/p90/p95, min, max, count ≥175, P(≥175), share of majority draws, unconditional contribution) |
| `coalition_residual_shocks.csv` | raw and centered pp residuals per historical election for C, S, V, MP (+ M, L, KD, SD, REST) and the C+S+MP / S+V+MP sums |
| `reweighting_sensitivity.csv` | conditional reweighting diagnostic **and** true leave-one-election-out re-runs |
| `diagnostic_report.json` | full machine-readable report incl. bug audit, λ diagnostics, variance decomposition, vote→seat map |
| `plots/overall_*.png` | pre-ElectionNoise vs final seat distribution, 175 marked |
| `plots/by_residual_year_*.png` | final distribution faceted by residual year, 175 marked |
| `plots/mixture_decomposition_*.png` | the pooled distribution decomposed into its six weighted components |
| `run_simulation.py` | instrumented production / pre-noise / LOO runner (passive wrappers only) |
| `analyze.py`, `plots.py` | analysis and figures |
| `test_instrumentation.py` | 5 targeted tests proving the instrumentation is RNG-neutral and index-exact |
| `_runs/*.npz` | raw 100 000-draw artifacts (uncommitted; ~14 MB each, 109 MB total) |

Reproduce:

```bash
for m in production prenoise loo:2002 loo:2006 loo:2010 loo:2014 loo:2018 loo:2022; do
  uv run python -m diagnostics.election_noise_mixture.run_simulation \
    --mode $m --samples 100000 --as-of 2026-08-24 --seed 12345 \
    --out diagnostics/election_noise_mixture/_runs/$(echo $m | tr ':' '_').npz
done
uv run python -m diagnostics.election_noise_mixture.analyze
uv run python -m unittest diagnostics.election_noise_mixture.test_instrumentation
```
