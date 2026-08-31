# Preregistration — ElectionNoise v2 challenger competition

**PREREGISTRATION STATUS: FROZEN — AMENDMENT 2** (see §J: J.1 records the
superseded Amendment-0 freeze, J.2 the Amendment-1 rationale, J.3 its superseded
freeze record, J.4 the Amendment-2 rationale, J.5 the operative freeze record).

**Written before any challenger implementation and before any challenger score
exists.** No challenger has been implemented, run, or scored. The 2026 forecast
has not been computed under any challenger.

**RC1 is not modified by this document.** `ElectionSimulator v1.0.0-rc1` remains
the frozen production control. This document authorizes *evaluation only*.
Passing the adoption gate in §F does not release anything; release remains a
separate, explicit decision.

| | |
|---|---|
| Document version | 2.2 — Amendment 2 (pre-challenger leakage-safety correction of the seat tier) |
| Written | 2026-08-30 |
| Frozen | Amendment 0: 2026-08-30T20:08:21Z · Amendment 1: §J.3 · Amendment 2: §J.5 |
| Author | Diagnostic work on branch `diagnostic/election-noise-mixture` |
| Motivating evidence | `diagnostics/election_noise_mixture/summary.md` (commit `f55c3c9`) |
| Component under test | ElectionNoise only |
| Frozen control | `pp_centered_noise` as implemented at commit `34c52d6` |

---

## 0. Provenance and preconditions

### 0.1 What triggered this

The Part-1 diagnostic reproduced published generation `20260828T201250Z-1da59168`
exactly (all 256 `coalition_builder` seat histograms bin-for-bin identical) and
established, with no implementation bug demonstrated on any audited hypothesis:

* C+S+MP P(≥175) = 10.778 %, of which **10 742 / 10 778 = 99.666 %** of majority
  draws come from the single residual year 2002.
* S+V+MP P(≥175) = 2.216 %, of which **96.93 %** come from 2002.
* C+S+MP is bimodal after ElectionNoise (modes 161 and 176, antimode 171) and
  unimodal before it. Pre-ElectionNoise P(≥175) is 0.010 % / 0.043 %.
* **83.7 %** of final C+S+MP seat variance is *between* residual-year branches;
  within-branch SD (~3.05) is essentially the pre-noise SD.
* True leave-one-election-out re-runs: dropping 2002 moves C+S+MP 10.778 % → 0.392 %
  and S+V+MP 2.216 % → 0.261 %; dropping 2014 (which contributes zero majority
  draws) moves them to 8.022 % and 0.857 % through re-centering alone.

The existing validation of this layer (`docs/election_layer_v2.md`) established
that `pp_noise_only` improves *marginal* party CRPS in all four forward-evaluated
elections and lifts 90 % coverage from 54.6 % to 76.9 %. The diagnostic shows that
this marginal validation does not, on its own, establish calibration of a
**threshold functional of a sum of parties**. That gap — not the shape of any
particular 2026 histogram — is what reopens the component.

### 0.2 Data-provenance caveat (maintenance item, not part of this competition)

`data/processed/election_residuals/contributing_polls_audit.csv` is tracked from
commit `f55bf36` and carries `poll_id` values from a superseded id scheme; the
`poll_id`s in the current `swedishpolls_individual_polls.csv` differ. Re-running
`tests/test_election_residuals.py` regenerates the audit file with current ids.

**This does not invalidate any residual value.** Pollster, interview and
publication dates, sample sizes, Kish weights and every support value were
verified identical between the committed audit file and the regenerated one; only
the id column differs. The residual pool under test is unaffected.

**Reviewer decision (§I item 10): the repair is explicitly out of scope for this
preregistration and must not be mixed into any commit that carries it.** The
recommended follow-up is a separate maintenance change that regenerates the audit
file under the current id scheme and demonstrates, as an explicit test, that the
resulting residual pool (`residuals_matrix`, `mean_bias_pp`,
`centered_residuals_matrix`) is unchanged to machine precision. Until then,
anyone auditing the residual pool should regenerate this artifact rather than
read the committed copy.

### 0.3 Preconditions before any challenger code is written

1. This document is frozen (§J) — satisfied.
2. No challenger has been implemented, run, or scored — satisfied.
3. The 2026 forecast under any challenger has not been computed — satisfied.
4. The Amendment-0 open clause (the non-operative Tier-1 "3 of 4" threshold) has
   been resolved by the reviewer and replaced by the general `N_T1 − 1` rule of
   §F.3 G5 — satisfied under Amendment 1.
5. The Part-2 historical-data feasibility search has not been run. It may change
   `N_T1` and `N_seat` **only** through the mechanical eligibility rules frozen
   here (§E.6). *(Satisfied and closed: Part 2 `cb39e84`, Part 2B `61d6d3b`.)*
6. **Amendment 2 only:** the isolated seat/coalition path of §E.2a has passed its
   leakage and sufficiency audit, recorded at
   `diagnostics/election_noise_v2/isolated_seat_evaluation/processed/isolated_path_audit.json`
   — satisfied. No challenger score existed when Amendment 2 was made.

---

## A. Control

The control is the current frozen production ElectionNoise, used verbatim, with
no reimplementation:

| Property | Specification | Location |
|---|---|---|
| Model id | `pp_centered_noise` | `scripts/vote_share_calibration/models.py` |
| Residual source | certified election result minus the 14-day final-poll consensus | `scripts/election_layer_v2/residuals_pool.py`, `scripts/election_residuals/consensus.py` |
| Window | `CANONICAL_WINDOW_DAYS = 14` | `scripts/election_layer_v2/config.py` |
| Space | percentage points, 9 categories, zero-sum | — |
| Category order | `(M, L, C, KD, S, V, MP, SD, REST)` = `ALL_CATEGORIES` | `scripts/election_residuals/config.py` |
| Pool for target `E` | all of `ALL_HISTORICAL_ELECTIONS` with `year < year(E)` (strictly chronological) | `residuals_pool.py` |
| Centering | subtract the pool mean bias, with zero-sum cleaning at 1e-12 | `residuals_pool.py` |
| Sampling law | **uniform over the K centered atoms**, one whole joint vector per draw | `models.py` |
| Index draw | `np.random.default_rng(index_seed).integers(0, K, size=N)` | `models.py` |
| Transfer | bounded simplex transfer `x' = x + λr`, `λ = min(1, min_{r_p<0}(x_p−ε)/(−r_p))`, `ε = 0.01` pp | `scripts/election_layer_v2/transfer.py` |
| REST | 9th category; participates in the residual vector and the vote denominator; never receives seats | `docs/election_simulator.md` §5 |
| Free parameters | **0** | — |

For the 2026 target, K = 6: `{2002, 2006, 2010, 2014, 2018, 2022}`.

The control is also the reference point for the "unsmoothed" end of Challenger
A's family: **`h = 0` is deliberately not in A's grid, because CONTROL already is
that model** (§C).

Every other frozen component is unchanged (§G).

---

## B. Scientific question

> **Can the predictive distribution of final-election error be improved — in its
> joint structure and specifically in its coalition-threshold behaviour — without
> degrading the already-validated marginal-party and seat-level performance of
> RC1?**

Explicitly **not** the question: "can the C+S+MP histogram be made smoother?"

The 2026 C+S+MP result is the *motivation* for reopening the component. It is
**not a target**. No specification, hyperparameter, tolerance or metric in this
document may be chosen, revised, or justified by reference to the 2026 forecast
under any model. Multimodality is not treated as an error in itself: a genuinely
bimodal predictive distribution can be correct. What is under test is whether a
6-atom empirical law is the best available representation of election-day error
given six observations, judged by proper scoring rules on historical outcomes.

The three models are chosen to make one clean comparison:

| Model | Joint residual law |
|---|---|
| CONTROL | discrete empirical |
| A | smoothed empirical, empirical covariance preserved exactly, singular (historical span only) |
| B | regularized continuous, full rank on the zero-sum subspace |

No heavy-tail hypothesis is under test here (§C, Challenger B).

---

## C. Candidate model families

Exactly three models are preregistered. **No fourth family may be added after any
result is seen.** If a preregistered family proves unimplementable as specified,
that fact is reported and the family is dropped — it is not replaced.

### Common notation

For target election `E` with training pool `P` (`K = |P|` elections):

* `r_e ∈ R⁹` — raw residual in pp for election `e`, zero-sum.
* `r̄_P = (1/K) Σ_{e∈P} r_e` — pool mean bias (zero-sum cleaned).
* `c_e = r_e − r̄_P` — centered residual.
* `C_P ∈ R^{K×9}` — matrix whose rows are `c_e`. Since `Σ_e c_e = 0`,
  `rank(C_P) ≤ K−1`.
* `S_P = C_Pᵀ C_P / K` — **maximum-likelihood** covariance about the fitted mean
  (divisor `K`, no Bessel correction). This is exactly the covariance of the
  CONTROL's atom distribution.
* `P₉ = I₉ − (1/9) 𝟙𝟙ᵀ` — orthogonal projector onto the zero-sum subspace, rank 8.
* `R` — the residual vector drawn by the layer for one simulation draw.

All three models emit a zero-sum `R ∈ R⁹` in pp and hand it to the **unchanged**
production transfer `apply_batch_simplex_transfer(x, R, eps=0.01)` (§G).

### CONTROL — six-atom centered empirical bootstrap

```
R = c_{k},   k ~ Uniform{1, …, K}
```

Free/tunable hyperparameters: **0**. Support: K points. Covariance: `S_P` exactly.

### CHALLENGER A — variance-corrected smoothed empirical bootstrap

A conservative continuous smoothing of the control that preserves the historical
joint structure exactly and replaces the K point masses with K Gaussian kernels.

```
k ~ Uniform{1, …, K}
z ~ N(0, I_K)                         (independent of k)
ε = (1/√K) Σ_{j=1}^{K} z_j c_j        (so ε ~ N(0, S_P))
R = (c_k + h·ε) / √(1 + h²)
```

**Coordinate system.** Percentage points, 9 categories, `ALL_CATEGORIES` order.
Not CLR: `docs/election_layer_v2.md` §1 records that CLR residuals produced
catastrophic distortion through log-ratio leverage on historically unpolled
parties, and the frozen layer, its validation artifacts and the transfer all
operate in pp space. Introducing a second, separately untested transformation
would confound the thing being tested.

**Centering.** Identical to the control: performed by the production
`load_chronological_pp_residuals`, unchanged.

**Covariance convention — binding.** A uses `S_P` with divisor `K`
(maximum-likelihood, **no** Bessel correction). This is not a stylistic choice:
it makes `Cov(R) = S_P` hold *exactly*, so **A nests CONTROL as h → 0**. That
nesting is the point of the family — A is the control plus one smoothing dial and
nothing else. **This divisor may not be changed during implementation.**

Verification: `E[R] = 0` since `Σ_j c_j = 0`; and
`Cov(R) = (Cov(c_k) + h²S_P)/(1+h²) = (S_P + h²S_P)/(1+h²) = S_P`, using
`Cov(c_k) = (1/K)Σ_j c_j c_jᵀ = S_P` and `Cov(ε) = (1/K)Σ_j c_j c_jᵀ = S_P`.

**Smoothing / bandwidth rule.** One scalar `h`, grid

```
H = {0.25, 0.50, 0.75, 1.00}
```

**`h = 0` is deliberately excluded: CONTROL already is the unsmoothed empirical
model, and including it would let A trivially collapse onto the control.**

Selection is by **LOEO-FIT**, run strictly inside the **outer** training pool
`P` and never touching the target election (§E.4):

```
for each h in H:
    score(h) = (1/K_outer) Σ_{j∈P}  ES( F^A(h, P\{j}),  r_j − r̄_{P\{j}} )
h* = argmin_h score(h);  ties broken toward the smallest h (most conservative)
```

`F^A(h, P\{j})` is Challenger A fitted on the **inner** pool `P\{j}`
(`K_inner = K_outer − 1`) and re-centered by the production centering algorithm
on `P\{j}`; the held-out target is the held-out residual expressed in that same
centering. `ES` is the energy score of §D1, estimated from 20 000 draws under
seed token `":election_noise_v2_a_loeo"` for each of the five seeds of §D0.

**Nested pool-size rule — binding (§I item 3, clarified by Amendment 1).**

Two distinct pool sizes exist and each has its own frozen minimum:

| | Definition | Frozen minimum |
|---|---|---|
| `K_outer` | size of the training pool for an evaluation target — all elections strictly earlier than the target | **`K_outer ≥ 3`** |
| `K_inner` | size of a pool inside LOEO-FIT, i.e. `K_outer − 1` | **`K_inner ≥ 2`** |

> **Outer evaluation eligibility.** A comparative historical target is eligible
> only if `K_outer ≥ 3`. **There is no `K_outer = 2` fallback**; such a target is
> excluded outright, and the identical common-case exclusion is applied to
> **CONTROL, Challenger A and Challenger B** so that every model ranking compares
> exactly the same historical cases.
>
> **Inner fitting.** For an eligible outer target, Challenger A selects `h` by the
> preregistered inner leave-one-out procedure over the outer pool. A target with
> `K_outer = 3` therefore necessarily produces inner fits with `K_inner = 2`.
> **This is explicitly allowed.** The Challenger A distribution is fitted normally
> at `K_inner = 2` for the purpose of scoring candidate `h` values.
>
> This is **not** the removed "K = 2 fallback": `h` is not fixed arbitrarily, the
> `K_inner = 2` fit exists only inside the already-preregistered inner validation,
> and candidate `h` values are still chosen from the frozen grid by held-out
> scoring.
>
> **`K_inner = 1` is prohibited.** That is precisely why an outer `K_outer = 2`
> case is excluded: its inner tuning would require a `K_inner = 1` fit, which
> provides no meaningful residual covariance and no meaningful smoothing fit.

Disclosed consequence at the smallest eligible size: with `K_inner = 2` the two
centered inner residuals satisfy `c_1 + c_2 = 0`, so `rank(C) = 1` and the inner
kernel covariance is rank 1 — the inner smoothing is one-dimensional. This is a
known and accepted property of the smallest eligible fold, not a defect, and it
affects only the *scoring of candidate `h` values*, never the outer model that is
actually evaluated (which is fitted at `K_outer ≥ 3`).

**Tail behaviour.** The marginal law is an equal-weight K-component Gaussian
mixture with component means `c_k/√(1+h²)` and common covariance
`h²S_P/(1+h²)`. Tails are Gaussian. **No heavy-tailed component is included, and
none may be substituted after results are seen.**

**Support / degeneracy — disclosed.** `ε ∈ span{c_1,…,c_K}`, dimension `≤ K−1`
(≤ 5 for the 2026 pool; ≤ 2 at the smallest eligible outer pool `K_outer = 3`).
A is therefore continuous but **singular**: supported on a `(K−1)`-dimensional
affine subspace of the 8-dimensional zero-sum hyperplane.
It cannot produce an error pattern outside the historical span. This is the
deliberate "conservative" property of the family and the main scientific contrast
with B.

**Simplex / composition handling.** `R` is zero-sum by construction (a linear
combination of zero-sum vectors), to floating-point residue only; the production
zero-sum cleaning in `apply_simplex_transfer` handles the residue. The production
λ rule and `ε = 0.01` pp floor are unchanged and invariant (§G).

**REST.** Identical to the control: REST is the 9th residual component, moves
with every draw, enters the vote denominator, and never receives seats.

**Free/tunable hyperparameters: exactly 1 (`h`), selected inside the historical
design.**

### CHALLENGER B — Ledoit–Wolf-regularized joint Gaussian residual model

One regularized continuous joint model with **no tunable hyperparameter at all**:
the regularization intensity is a closed-form function of the training pool.

```
R ~ N(0, Σ̃_P)
```

with `Σ̃_P` built as follows, using the normalized Frobenius norm
`‖A‖² := tr(A Aᵀ)/8` (8 = rank of the zero-sum subspace):

```
τ²   = tr(S_P) / 8
T    = τ² · P₉                                    (isotropic target on the zero-sum subspace)
d²   = ‖S_P − T‖²
b̄²   = (1/K²) Σ_{j=1}^{K} ‖c_j c_jᵀ − S_P‖²
b²   = min(b̄², d²)
δ    = b² / d²        (δ := 1 if d² = 0)
Σ_LW = δ·T + (1−δ)·S_P
Σ̃_P  = (K/(K−1)) · Σ_LW                           (single Bessel correction, applied once, at the end)
```

**Distribution — frozen as Gaussian (§I item 5).** Student-t is explicitly
rejected for this competition. A fixed or estimated degrees-of-freedom parameter
would introduce a separate heavy-tail hypothesis that the Part-1 diagnostic does
not motivate, and would make the CONTROL / A / B comparison harder to interpret.
A heavy-tail challenger may be proposed only in a future, separately preregistered
experiment, if evidence warrants it. **No substitution is permitted after any
result is seen.**

**Covariance conventions — binding, and not to be silently changed during
implementation.** Three separate conventions are in play and each is fixed here:

1. **Inside the Ledoit–Wolf estimator, `S_P` uses divisor `K`** (maximum
   likelihood), which is the convention Ledoit–Wolf (2004) is stated in. `b̄²`
   uses the `1/K²` prefactor of that paper.
2. **The Bessel correction `K/(K−1)` is applied exactly once, at the very end**,
   to the already-shrunk `Σ_LW`. It corrects the downward bias of `S_P` arising
   from estimating the mean from the same `K` points. It is a fixed formula, not
   a tunable dial. It is **not** applied inside `d²`, `b̄²`, `τ²`, or `δ`.
3. **Challenger A uses divisor `K` with no Bessel correction at all**, because A's
   defining property is exact preservation of the CONTROL's covariance and exact
   nesting at `h → 0`. The asymmetry between A and B is deliberate: A is a
   smoothing of an empirical law and must match it; B is a fitted parametric law
   and is bias-corrected.

Consequences that must hold in the implementation and be asserted by tests:

* The `1/8` normalization cancels in the ratio `δ = b²/d²`, so `δ` does not depend
  on the choice of `p_eff`. It is written explicitly only to make the adaptation
  of Ledoit–Wolf's scaled-identity target to a rank-8 projector target
  unambiguous.
* `S_P 𝟙 = 0` because every `c_j` is zero-sum; hence `S_P P₉ = S_P`,
  `τ² = tr(S_P P₉)/tr(P₉) = tr(S_P)/8` is exactly the projection coefficient of
  `S_P` onto `P₉`, and `Σ̃_P 𝟙 = 0`.
* Because `Σ̃_P 𝟙 = 0` and the symmetric square root shares its eigenvectors,
  `Σ̃_P^{1/2} 𝟙 = 0`, so `𝟙ᵀR = (Σ̃_P^{1/2}𝟙)ᵀ z = 0`: **every draw is zero-sum
  almost surely.**
* `δ ∈ [0, 1]` by construction, since `b² = min(b̄², d²) ≤ d²`.

Sampling: symmetric PSD square root by eigendecomposition of `Σ̃_P` with
eigenvalues clipped at 0; `R = Σ̃_P^{1/2} z`, `z ~ N(0, I₉)`; then
`R ← R − mean(R)` to remove floating-point residue, matching the control's
zero-sum cleaning.

**Coordinate system / centering / REST / simplex handling / λ:** identical to
Challenger A and to the control.

**Rank.** Unlike A, B is **full rank 8** on the zero-sum hyperplane whenever
`δ > 0`, and can therefore produce error patterns outside the historical span.

**Free/tunable hyperparameters: 0.**

### Complexity budget

With six elections, the total budget across both challengers is **one** scalar,
selected inside the historical design. Any specification requiring more is out of
scope.

---

## D. Evaluation metrics

### D0. Monte Carlo design — binding for every model and every case (§I item 4)

| Item | Value |
|---|---|
| Seeds | **exactly five**: `12345, 24680, 98765, 54321, 13579` |
| Draws | **N = 20 000 per seed, per case, per model** (100 000 draws total per case per model) |
| Identical across models | yes — same seed set, same N, same cases, no exceptions |

Scoring procedure for every adoption metric:

1. Compute the score **separately for each of the five seeds**.
2. Report the **five-seed mean** and the **five-seed standard deviation**.
3. **The adoption gate in §F is applied to the five-seed mean.**

**Choosing a favourable seed, dropping a seed, or pooling only favourable runs is
prohibited.** All five per-seed values must appear in the results artifact for
every model and every case.

### D1. PRIMARY — joint vote metric

**Energy score on the 9-category vote composition, in percentage points, with
Euclidean distance.**

* Implementation: `scripts/vote_share_calibration/energy_score.py::compute_energy_score`, unchanged.
* Representation: `x ∈ R⁹`, pp units summing to 100, order `(M, L, C, KD, S, V, MP, SD, REST)`, **including REST**.
* Distance: `‖·‖₂` on that vector.
* `ES(F, y) = E‖X − y‖₂ − ½·E‖X − X'‖₂`; strictly proper for the joint law; lower is better.
* Truth `y`: the certified election composition in the same 9 categories and units.

**Justification of the representation.** This is the geometry the layer acts in:
residuals are pp, the transfer is additive in pp, and the frozen layer's own
validation is reported in pp. `docs/election_layer_v2.md` §1 records that the CLR
alternative was tested and rejected as catastrophic. Scoring in pp/Euclidean
introduces no untested transformation between the model and the score.

Secondary read, reported alongside but not the primary gate: the 8-party energy
score (REST excluded), since REST never reaches the seat layer.

### D2. Marginal vote metrics

* **Per-party discrete CRPS** via `scripts/election_layer_v2/forward_eval.py::compute_discrete_crps`, unchanged, for all 9 categories. Headline: mean over the 8 parliamentary parties.
* **Central interval coverage and mean width** at 50 %, 80 % and 90 %, computed as in the existing `election_layer_v2` / `vote_share_calibration` artifacts.
* Existing artifact schemas are retained so new numbers sit beside the frozen ones.

### D3. Joint seat metric

* **Energy score on the 8-dimensional integer seat vector** via `scripts/seat_hindcasts/metrics.py::calculate_multivariate_energy_score`, unchanged.
* Every vote draw passes through the **identical** deterministic downstream: geographic IPF projection → exact-margin biproportional controlled rounding → `dispatch_production_allocation` (349-seat Sainte-Laguë with the legal fallback). No part of that path is re-implemented or re-parameterized.
* Truth: the certified seat vectors in `scripts/seat_hindcasts/config.py::EVALUATION_ELECTIONS`, extended only by elections that pass the Part-2 admission requirements of §E.3. **(Amendment 2)** For Tier 3-ISO this is 2018 and 2022 from that source plus 2014 from the Part-2B certified artifact `diagnostics/election_noise_v2/historical_seat_extension/processed/certified_mandates_2010_2014.csv`.
* Supporting: per-party discrete seat CRPS via `calculate_discrete_seat_crps`.

### D4. Coalition-threshold metric

* **Mask set — fixed in advance, exhaustive.** All `m ∈ {1, …, 254}` over the fixed
  party order `(M, L, C, KD, S, V, MP, SD)`. Masks 0 (empty) and 255 (all eight)
  are excluded, explicitly: they yield 0 and 349 seats in every draw and every
  realized outcome, so their Brier score is identically 0 for every model and
  carries no discriminating information. **Coalitions are never selected on the
  basis of today's interesting examples.**
* Forecast: `p_m = (1/N) Σ_i 1{ Σ_{p∈m} seats_{i,p} ≥ 175 }`, computed per seed.
* Outcome: `y_m = 1{ Σ_{p∈m} actual_seats_p ≥ 175 }`.
* Score: `B_m = (p_m − y_m)²`.

**Complement duplication — decided before any scoring.** Seats sum to exactly 349
in every draw and every realized outcome, and 349 is odd, so `{s_m ≥ 175}` is the
exact complement of `{s_{255−m} ≥ 175}`. Therefore `p_{255−m} = 1 − p_m`,
`y_{255−m} = 1 − y_m`, and `B_{255−m} = B_m` **identically**.

> **Decision: retain all 254 nontrivial masks symmetrically.** The mean over 254
> masks equals the mean over any set of 127 complement representatives, so this
> cannot change any ranking. It is chosen over a representative rule because the
> obvious deterministic representative set (masks 1–127) excludes SD from every
> retained coalition, which reads as a structural bias even though it is
> statistically inert. **The effective number of distinct binary events per
> election is 127, not 254**, and no uncertainty statement may use 254. Applied
> identically to CONTROL, A and B.

**Aggregation — masks are never treated as independent observations.**

1. Per `(election, horizon, seed)` case: `B̄ = (1/254) Σ_m B_m`.
2. Per `(election, horizon)`: mean over the five seeds (with the seed SD reported).
3. Per election: mean over horizons → the **election-level aggregate coalition Brier**.
4. Headline: unweighted mean of the election-level aggregates over evaluation elections.

**(Amendment 2)** On **Tier 3-ISO** there is exactly one case per election — the
isolated path has a single forecast origin and therefore no horizon dimension — so
step 3 is the identity and steps 1, 2 and 4 apply unchanged: mask mean, then
five-seed mean, then the unweighted mean over the three elections. Elections remain
equally weighted and masks are still never treated as independent. The aggregation
rule itself is not modified.

Per-case, per-seed and per-election values are always reported alongside the
headline. The number of independent realized outcomes is the number of
elections, and every reported summary must state it.

### D5. Mandatory λ diagnostics (recorded, never used for selection) (§I item 6)

For **every evaluation model, case and fold**, record at minimum:

* fraction of draws with `λ < 1`
* mean λ
* low quantiles of λ: p01, p05, p10
* minimum λ
* binding donor/category counts (which category's `ε`-floor bound `λ`)

These exist because the continuous challengers can generate residuals more
extreme than any historical atom and so may trigger simplex attenuation more
often than the control.

**These are mandatory diagnostics, not model-selection criteria.** No
clipping-frequency gate may be introduced after seeing results. If a challenger
passes the statistical gate in §F but exhibits qualitatively extreme clipping,
that must be **reported for scientific review before any production promotion**;
the transfer rule must never be silently altered to accommodate it.

---

## E. Historical evaluation design

### E.1 Leakage rule (all tiers)

For target election `E`, the residual pool is exactly `{e : year(e) < year(E)}`.
This is the production rule and is a leave-all-future-out design. The target
election's own residual never enters the pool, the bandwidth selection, or the
covariance estimate.

### E.2 Tiers, the frozen Tier-1 target set, and the `K_outer ≥ 3` eligibility rule

`K_outer` depends only on the target election year (the pool is all elections
strictly earlier than the target), never on the horizon.

**Tier 1 — standalone forward evaluation from the 14-day polling consensus.**
Vote level only. The base composition is the poll consensus, so OpinionState and
Dynamics are absent; this tier isolates ElectionNoise. Mirrors
`docs/election_layer_v2.md` §4.

> **FROZEN TIER-1 CANDIDATE TARGET SET — `{2010, 2014, 2018, 2022}`, and only
> those four.** This set is fixed by this document and **must not expand**. Part 2
> may find credible older residual elections that enter historical *training*
> pools under the frozen data-inclusion criteria of §E.3; **older discovered
> elections never become new Tier-1 target elections**, and no target may be added
> because its result is favourable.

Eligibility of each candidate target is then decided **mechanically** by
`K_outer ≥ 3`:

| Candidate target | Pool under the current residual history | `K_outer` | Eligible? |
|---|---|---|---|
| 2010 | {2002, 2006} | 2 | **NO** — excluded |
| 2014 | {2002, 2006, 2010} | 3 | yes (inner folds run at `K_inner = 2`, allowed) |
| 2018 | {2002, 2006, 2010, 2014} | 4 | yes |
| 2022 | {2002, 2006, 2010, 2014, 2018} | 5 | yes |

**Definition.** `N_T1` = the number of **eligible** targets among the frozen
candidate set `{2010, 2014, 2018, 2022}`, after applying the frozen historical-data
inclusion criteria of §E.3 and the `K_outer ≥ 3` rule.

> **`N_T1 = 3` under the current residual history** (eligible: `{2014, 2018, 2022}`;
> 2010 excluded because `K_outer = 2`).
>
> `N_T1` becomes **4** if and only if Part 2 admits at least one pre-2002 residual
> election under §E.3, which would raise the 2010 target to `K_outer ≥ 3`. That is
> the *only* mechanism by which `N_T1` can change. The same eligibility set is
> applied identically to CONTROL, A and B.
>
> **If `N_T1 < 3`: STOP.** The preregistered Tier-1 evidence base is deemed
> insufficient for the challenger competition, and a new methodological decision
> is required before implementation. **A two-election rule must not be invented.**

**Tier 2 — full-pipeline hindcast (rolling origin).** Vote level, full frozen
pipeline. Mirrors `docs/election_layer_v2.md` §5 and `scripts/seat_hindcasts`.

| Target | Pool | K | Horizons | Status |
|---|---|---|---|---|
| 2018 | 4 elections | 4 | 112, 84, 56, 28, 14, 7 | all 6 included |
| 2022 | 5 elections | 5 | 112, 84, 56, 28, 14, 7 | all 6 included |

> **Frozen Tier-2 case set: 12 cases over 2 elections. No case is excluded by the
> `K_outer ≥ 3` rule** (2018 has `K_outer = 4`, 2022 has `K_outer = 5`).
>
> **AMENDMENT 2 — Tier 2 is retained but leaves the adoption gate.** Its historical
> Poll-of-Polls state input is not publication-time leakage-safe (§E.5 item 7), so
> Tier-2 results are computed, preserved and reported as **retrospective
> diagnostics** and are **not used for adoption**. Existing Tier-2 results are not
> discarded or rewritten.

**Tier 3 — seat and coalition level.** The same 12 cases as Tier 2, with
geography baselines 2014→2018 and 2018→2022, through the identical geography +
exact mandate allocator. Metrics D3, D4, D5.

> **Frozen Tier-3 case set: 12 cases over 2 elections ({2018, 2022}).**
>
> **AMENDMENT 2 — Tier 3 is likewise retained as a retrospective diagnostic and
> leaves the adoption gate**, for the same reason as Tier 2: it is defined on
> Tier-2 cases and therefore inherits their leaky upstream state. Its existing
> results are preserved, not rewritten. The authoritative seat/coalition
> evaluation is **Tier 3-ISO** (§E.2a).

### E.2a Tier 3-ISO — the authoritative seat/coalition evaluation (Amendment 2)

The isolated ElectionNoise path, with no OpinionState and no Dynamics:

```
historical final 14-day polling consensus      (publication-date safe)
  -> ElectionNoise                             (the component under test)
  -> unchanged bounded simplex transfer         (λ rule, ε = 0.01 pp)
  -> frozen deterministic geography             (chronological mode only)
  -> historically correct mandate law
  -> joint per-draw seat vectors
```

The consensus is **exactly** the existing
`scripts/election_residuals/consensus.py::build_election_polling_consensus`, the
same estimator used to construct the historical ElectionNoise residuals. **No new
polling estimator is introduced.** It admits a poll only if
`publication_date <= election_date` **and** `interview_end <= election_date`, so
every input was published at the forecast origin.

| Target | Residual training years | `K_outer` | Geography baseline | Mandate law | First divisor |
|---|---|---|---|---|---|
| **2014** | 2002, 2006, 2010 | 3 | 2010 | **PRE_2018** | 7/5 |
| **2018** | 2002, 2006, 2010, 2014 | 4 | 2014 | POST_2018 | 6/5 |
| **2022** | 2002, 2006, 2010, 2014, 2018 | 5 | 2018 | POST_2018 | 6/5 |

> **Frozen Tier 3-ISO case set: 3 cases over 3 elections. `N_seat` = 3.**
> One case per election: the isolated path has a single forecast origin (the final
> 14-day consensus) and therefore **no horizon dimension**.

**Geography mode is restricted to `chronological`. `oracle` mode is prohibited on
this path**, because its row margins come from the target election's realized
constituency valid votes. In chronological mode with a target year ≤ 2022 the row
margins are `R = sum(B, axis=1)`, derived entirely from the baseline election, and
the electorates file is not read into the result at all — proven by perturbation
test in the Amendment-2 audit.

Metrics: **D3, D4, D5**. Truth is the certified historical seat vector: 2018 and
2022 from `scripts/seat_hindcasts/config.py::EVALUATION_ELECTIONS`, 2014 from the
Part-2B certified artifact
`diagnostics/election_noise_v2/historical_seat_extension/processed/certified_mandates_2010_2014.csv`
(M 84, L 19, C 22, KD 16, S 113, V 21, MP 25, SD 49 = 349), which passed the seven
§E.3 admission requirements.

**Why 2014 is admissible here but was not in Tier 2/Tier 3.** Part 2B established
that 2014 satisfies every §E.3 admission requirement, including exact vote
reconciliation, an unambiguous PRE_2018 law, a boundary-clean 2010 baseline, and
zero disagreement with the certified coalition-majority indicator across all 254
masks. Part 3 nevertheless excluded it because Tier 2 is defined as a
*full-pipeline* hindcast and OpinionState v1.1 cannot be estimated for any 2014
`as_of` (the Poll-of-Polls daily series begins 2014-09-15, one day after the
election). Tier 3-ISO does not use OpinionState, so the §E.3 admission is
sufficient and 2014 enters.

Total draw budget per model: **adoption** = (`N_T1` + `N_seat`) = 3 + 3 = 6 cases
× 5 seeds × 20 000 draws; **diagnostics** = a further 12 Tier-2 and 12 Tier-3
cases on the same seed and draw design.

### E.3 Backward extension of the seat/coalition evidence — a Part-2 investigation, not a licence (§I item 7)

The current Tier-3 restriction to 2018 and 2022 follows from two hard repository
facts:

* `data/processed/mandates/historical_certified_mandates.csv` covers **only 2018 and 2022**.
* `data/processed/geography/constituency_party_votes_2014_2022.csv` covers **2014, 2018, 2022**.

A 2014 target would need a 2010 constituency baseline, which does not exist here.

> **This restriction is explicitly NOT treated as permanent.** Part 2 must
> investigate whether official historical constituency vote data and certified
> seat results can credibly extend the exact end-to-end evaluation backward, for:
>
> *(Amendment 2: the seven requirements below now govern admission to **Tier 3-ISO**,
> the authoritative seat tier, as well as to the full-pipeline diagnostic tiers.)*
>
> * **2014 target** — requires a 2010 constituency baseline plus certified 2014 mandates and results;
> * **2010 target** — requires a 2006 constituency baseline plus certified 2010 mandates and results.

**Inclusion / provenance requirements, declared before the search begins.** An
older election may be admitted to Tier 2/Tier 3 only if **all** of the following
hold:

1. **Authoritative source.** Constituency-level party votes and certified mandate
   allocations come from the official election authority (Valmyndigheten) or an
   equally authoritative certified record — not a secondary compilation, not a
   reconstruction.
2. **Completeness.** Full coverage of every constituency in force for that
   election, with constituency-valid-vote totals, for all nine model categories
   (with REST reconstructible as the residual mass).
3. **Constituency-structure resolution.** The constituency set and its fixed-seat
   allocation for that election are documented and representable in the existing
   `OFFICIAL_CONSTITUENCY_CODES` / `FIXED_SEATS_*` structures, with any historical
   boundary or seat-count change explicitly mapped and recorded.
4. **Algorithm invariance.** The election runs through the **identical**
   chronological geography → IPF → exact-margin controlled rounding → exact
   mandate allocator path with **no modification to the geography or allocation
   algorithm.** *Altering the geography algorithm merely to admit an older
   election is prohibited.*
5. **Leakage safety.** The chronological baseline rule is respected (target `E`
   uses the immediately preceding election's constituency baseline), and the
   residual pool rule of §E.1 still holds.
6. **Reproducibility.** The new inputs are hashed into the reproducibility
   manifest exactly as the existing ones are, with a documented retrieval
   provenance.
7. **Verification.** The pipeline reproduces the certified national seat vector
   for that election when fed the certified national vote shares, as a
   correctness precondition before any scoring.

If an election fails **any** of these, it is **rejected from Tier 2 and Tier 3**
and the rejection is documented with the specific criterion that failed. A
rejected election may not be admitted with a weakened criterion.

The number of seat-evaluable elections that survives this process determines
which branch of the frozen conditional coalition-Brier rule in §F.3 G5 applies.
**That rule is
fixed now, before the search, precisely so that the outcome of the search cannot
select a favourable rule after the fact.**

### E.4 Two distinct leave-one-out loops — named to prevent confusion

* **LOEO-FIT** — inside the **outer** training pool, for Challenger A's bandwidth only (§C). Never sees the target election. Operates at `K_inner = K_outer − 1`, with **`K_inner ≥ 2`** permitted and **`K_inner = 1` prohibited**. It is the outer rule `K_outer ≥ 3` that guarantees `K_inner ≥ 2`.
* **LOEO-EVAL** — over the *evaluation* set, used only by the robustness criterion G5: each evaluation election is dropped in turn and the headline metrics are recomputed. A stability check on the conclusion, not a model-fitting step.

### E.6 What Part 2 may and may not determine (Amendment 1)

Part 2 is a **data-feasibility investigation only**. It may determine which
historical observations satisfy the frozen data-quality criteria of §E.3.

**Part 2 may NOT change any of the following. All are frozen by this document:**

* the Tier-1 candidate target set `{2010, 2014, 2018, 2022}`
* the `K_outer` minimum (3)
* the `K_inner` minimum (2)
* Challenger A's `h` grid `{0.25, 0.50, 0.75, 1.00}`
* any scoring metric (§D)
* any numerical adoption threshold (§F.1)
* the `N_T1 − 1` Tier-1 robustness rule (§F.3 G5)
* the conditional coalition-Brier rule (§F.3 G5)
* horizon weights (§F.3 G4b and the primary Tier-2 summary)
* the seed policy or sample counts (§D0)
* the Challenger A and Challenger B specifications (§C)

**Data availability may change `N_T1` or `N_seat` only through the mechanical
eligibility rules frozen above** — `K_outer ≥ 3` for Tier-1 targets (§E.2) and the
seven admission requirements for seat-evaluable elections (§E.3). No discretionary
judgement enters either count.

> **Amendment-2 clarification.** This section bounds **Part 2**, and Part 2 stayed
> inside it: it left `N_seat` at 2. `N_seat` became 3 under Amendment 2 not through
> data availability but through an explicit, reviewer-authorised redefinition of
> which tier carries the seat/coalition evaluation (§E.2a), made before any
> challenger score existed. §E.3's seven admission requirements continue to govern
> which elections may enter the seat tier, and 2014 satisfies all seven.

### E.5 Limitations — to be restated in every report produced under this preregistration

1. The residual pool has at most **six** observations under the current history.
   Every covariance and bandwidth quantity is estimated from ≤6 points in a
   9-dimensional zero-sum space of rank ≤5, and at the smallest eligible outer
   pool (`K_outer = 3`) from 3 points of rank ≤2, with inner LOEO-FIT folds at
   `K_inner = 2` of rank 1 (§C).
2. The coalition-threshold gate rests on **two** realized elections as of this
   freeze (possibly more after §E.3), and on 127 effectively distinct, strongly
   dependent binary events per election. It is a **decision rule, not a hypothesis
   test.** No p-values, confidence intervals, or significance claims will be made.
3. Tiers 2 and 3 are **retrospective, not independent holdout**: the model family,
   the polling calibration, and the frozen components were all chosen with
   knowledge of 2018 and 2022.
4. **No prospective validation is claimed.** The 2026 election is unobserved; its
   outcome cannot enter any part of this competition.
5. **Six elections is not large-N evidence and must never be described as such.**
   Any summary that reads as a general claim about Swedish polling error, rather
   than a claim about a handful of observations, is a reporting error.
6. **(Amendment 2)** The Tier-1 and Tier 3-ISO metrics are **not statistically
   independent evidence** — see §E.7.
7. **(Amendment 2) The historical Poll-of-Polls state series is not
   publication-time leakage-safe.** Part 3B established that it is a
   fieldwork-dated rolling aggregate which the provider retrospectively completes
   as later-published polls arrive: two archived snapshots retrieved one day apart
   differ in 176 of 34 888 cells, and the largest revision (2026-08-22, MP
   7.5 → 7.3) is explicable only by a poll published 2026-08-27. Across the six
   `as_of` dates of each full-pipeline election, 19 poll-instances leak for 2018
   (max lead 37 days) and 13 for 2022 (max 18 days). This is a property of the
   frozen production input, is common-mode across models, and is **not** repaired
   here. Its consequence is confined to scope: the full-pipeline Tier-2/Tier-3
   results are retained as retrospective diagnostics and excluded from the
   adoption gate, and the authoritative seat/coalition evaluation moves to the
   leakage-safe Tier 3-ISO path (§E.2a).

### E.7 How the vote and seat evaluations relate (Amendment 2)

Tier 1 and Tier 3-ISO score **the same predictive distribution**. Tier 1 scores it
directly in vote space; Tier 3-ISO scores deterministic, nonlinear downstream
functionals of that identical distribution — the geography projection, the
integerisation and the statutory allocator are a fixed pushforward applied to the
very same draws.

> **They are therefore related evaluations of one forecast, not independent
> evidence, and must never be reported or aggregated as if they were.** No
> statement of the form "the challenger won on two independent tiers" is
> admissible.

Their purpose is different from adding evidence: it is to check that an
improvement in marginal and joint *vote* prediction also behaves sensibly for the
downstream quantities that exposed the six-atom problem in the first place. A
challenger could improve vote-space scores while degrading a coalition-threshold
functional, and the seat tier exists to catch exactly that.

Two consequences of the isolated construction must be stated openly, because they
shape how the seat metrics should be read:

* **CONTROL's predictive law on this path is exactly `K` equal atoms** (`K` = 3, 4,
  5 for 2014, 2018, 2022). Verified: the path produces exactly `K` distinct vote
  compositions and exactly `K` distinct seat vectors. So CONTROL's coalition
  probability `p_m` can only take values in `{0, 1/K, …, 1}` — at 2014, only
  `{0, ⅓, ⅔, 1}`.
* A continuous challenger can express intermediate probabilities that CONTROL
  structurally cannot. The Brier score is strictly proper and the truth is
  external, so this is a legitimate comparison and not circular — but the
  coalition-Brier improvement threshold is likely to be **easy** for any
  well-behaved continuous law to clear. The informative content of the gate
  therefore sits in the **non-inferiority guards** (G3, G4) and the
  per-election robustness conditions (G5), not in the headline Brier improvement
  alone. Readers of the eventual result should weight it accordingly.

---

## F. Adoption gate — fixed before any challenger score exists

### F.1 Tolerances (§I item 2) — frozen

All comparisons are relative to CONTROL on the identical case set, with the
identical five seeds and identical N, applied to the **five-seed mean**.

| Term | Definition |
|---|---|
| **"improves"** | the challenger's five-seed mean metric is strictly lower and by **≥ 2.0 % relative** to CONTROL |
| **"does not materially worsen"** (non-inferiority) | the challenger's five-seed mean metric is **not more than 1.0 % relative** above CONTROL |
| **"coverage does not materially worsen"** | at each of 50 %, 80 %, 90 %, `abs(coverage − nominal)` increases by **no more than 3.0 percentage points** versus CONTROL |

These are deliberately simple practical thresholds, declared before any challenger
result exists. **They may not be retuned later on the basis of whether A or B
almost passes.**

### F.2 Tier roles (§I item 1) — frozen

| Tier | Role in the gate (as amended) |
|---|---|
| **Tier 1** (3 cases, ElectionNoise isolated, vote level) | **Required improvement** on the primary joint vote energy score, **plus** hard non-inferiority on the marginal and interval metrics |
| **Tier 3-ISO** (3 cases, ElectionNoise isolated, seat level) | Coalition-majority Brier is a **separate required-improvement criterion**; seat-vector ES is a **hard non-inferiority check** |
| **Tier 2 and Tier 3** (12 + 12 cases, full pipeline) | **Retrospective diagnostics only — not part of the gate** (§E.5 item 7). Computed, preserved and reported; never used for adoption. |

Rationale for putting the improvement requirement on Tier 1: it isolates
ElectionNoise, which is the component under test, and it spans **more independent
elections** (3: 2014, 2018, 2022) than Tier 2 (2: 2018, 2022). Tier 2's 12 cases
are 6 horizons on only 2 realized outcomes, and each case carries the additional
uncertainty of OpinionState and Dynamics, so Tier 2 has less power to identify
whether ElectionNoise itself improved. Case count is not the relevant measure of
evidence here; the number of independent realized elections is.

> **No compensation.** A required improvement may **never** offset a material
> degradation elsewhere. Every non-inferiority check is a hard gate evaluated
> independently; failing any one of them fails the candidate regardless of how
> large the improvement is. This principle is unchanged by Amendment 2; only the
> tiers the checks are evaluated on changed.

**Amendment-2 note on the rationale above.** The argument for putting the
improvement requirement on Tier 1 rather than the full pipeline was that Tier 1
isolates the component under test and spans more independent elections. Amendment 2
strengthens that argument rather than weakening it: both gate tiers are now
isolated-ElectionNoise evaluations over the same three elections, and the two
full-pipeline tiers — which carried both the extra OpinionState/Dynamics
uncertainty *and* the leaky historical state — are out of the gate entirely.

### F.3 The gate — all criteria must hold

**G1 — Tier-1 joint vote performance improves (required improvement). Unchanged.**
Tier-1 five-seed mean 9-category energy score (D1) **improves** (≥ 2.0 % better
than CONTROL).

**G2 — coalition-majority Brier improves (required improvement). Re-pointed by
Amendment 2 to Tier 3-ISO.**
The **Tier 3-ISO** headline coalition Brier (D4), computed after the identical
frozen geography and the historically correct mandate allocator, **improves**
(≥ 2.0 % better than CONTROL). Its per-election conditions are in G5, which now
runs on the `N_seat ≥ 3` branch because `N_seat` = 3.

**G3 — Tier-1 marginal and interval non-inferiority (hard). Re-pointed by
Amendment 2 from Tier 2 to Tier 1.**
On the 3 Tier-1 cases, none of the following may materially worsen (each
≤ +1.0 % relative): the 9-category energy score; the 8-party energy score; the
mean 8-party CRPS. Coverage must not materially worsen at 50/80/90 % (≤ +3.0 pp
absolute deviation from nominal).
*The tolerances are identical to the ones this criterion carried before; only the
tier it is evaluated on changed, because the previous tier's upstream state is not
publication-time leakage-safe.*

**G4 — seat non-inferiority (hard). Re-pointed by Amendment 2 to Tier 3-ISO.**
The **Tier 3-ISO** mean 8-dimensional seat-vector energy score does not materially
worsen (≤ +1.0 % relative).

**G4b — RETIRED FROM THE GATE by Amendment 2; retained as a diagnostic.**
G4b aggregated the Tier-2 cases at horizons 14 and 28 days. The isolated path has a
**single forecast origin** and therefore no horizon dimension, so no Tier 3-ISO
analogue of G4b exists. It is **not** replaced by an invented substitute. G4b
continues to be **computed and reported on the Tier-2 diagnostic set** under the
same 1.0 % tolerance, and the primary Tier-2 diagnostic summary keeps equal
weighting over all six preregistered horizons.

> **Recorded consequence, stated plainly:** retiring G4b removes one hard gate. The
> partial mitigation is that Tier 1 and Tier 3-ISO are both built on the final
> 14-day consensus, i.e. the shortest and operationally most relevant origin, so
> the regime G4b was added to protect is the regime the gate now evaluates
> throughout. That is a weaker guarantee than G4b provided, because G4b tested the
> full pipeline at a short horizon whereas the isolated path omits OpinionState and
> Dynamics entirely. The reviewer should treat this as a known reduction in gate
> coverage, accepted in exchange for removing a leaky input from the adoption
> decision.

**G5 — robustness to individual elections (§I item 9; Tier-1 clause replaced by
Amendment 1).**

*Tier-1 joint vote — general `N_T1 − 1` rule, declared before Part 2 so that
historical-data availability cannot select a favourable robustness criterion
afterwards.* Let `N_T1` be as defined in §E.2.

**A. Aggregate requirement (unchanged).** The challenger improves the frozen
Tier-1 aggregate primary joint-vote energy score by **≥ 2.0 %** (this is G1).

**B. Leave-one-target-out robustness.** For each of the `N_T1` eligible Tier-1
targets: remove that target from the evaluation aggregate and recompute challenger
versus CONTROL over the remaining eligible targets. Then

* **every** leave-one-target-out relative improvement must be **> 0** — the
  challenger must beat CONTROL directionally in **all** `N_T1` recomputations;
  **and**
* at least **`N_T1 − 1`** of the `N_T1` leave-one-target-out recomputations must
  show at least **1.0 % relative improvement**.

**C. Individual-election consistency.** At least **`N_T1 − 1`** of the `N_T1`
eligible individual Tier-1 target elections must have a **lower** primary
joint-vote energy score under the challenger than under CONTROL.

Instantiations (both fixed now; neither is chosen after seeing data):

| | `N_T1 = 3` (current) | `N_T1 = 4` (if Part 2 restores 2010) |
|---|---|---|
| B — all LOO aggregates favour challenger | all **3** | all **4** |
| B — LOO aggregates improving ≥ 1 % | at least **2** of 3 | at least **3** of 4 |
| C — individual elections favouring challenger | at least **2** of 3 | at least **3** of 4 |

The `N_T1 = 4` instantiation reproduces the originally intended "3 of 4"
robustness requirement exactly; the `N_T1 = 3` instantiation is defined by the
same formula rather than by post-hoc discretion.

**If `N_T1 < 3`: STOP.** The Tier-1 evidence base is deemed insufficient and a new
methodological decision is required before implementation. **A two-election rule
must not be invented.**

*Coalition Brier — frozen conditional rule, declared before the §E.3 historical-data
search so that data availability cannot select a favourable rule after the fact.*
Let `N_elections` (also written `N_seat`) be the number of seat-evaluable
elections admitted under §E.3. **This rule is unchanged by Amendment 1 and
unchanged by Amendment 2.** Under Amendment 2 it is evaluated on Tier 3-ISO with
`N_seat` = 3, so the **`N_elections ≥ 3` branch is the live one**: aggregate
improvement ≥ 2 %, improvement in at least `ceil(3/2) = 2` of the 3 elections, and
removing any one election must not turn the aggregate delta into a degradation of
more than 1.0 %.

* **If `N_elections == 2`:**
  * aggregate coalition Brier must improve **≥ 2 %**; **and**
  * the challenger must beat CONTROL on the election-level aggregate coalition Brier in **both** historical elections.
* **If `N_elections ≥ 3`:**
  * aggregate coalition Brier must improve **≥ 2 %**; **and**
  * the challenger must improve the election-level aggregate coalition Brier in at least `ceil(N_elections / 2)` elections; **and**
  * removing any one evaluation election must not turn the aggregate challenger-vs-CONTROL Brier delta materially negative (a degradation of more than 1.0 % relative).

**G6 — determinism and reproducibility (§I item 4, clarified).**
The canonical test is: **repeating the exact same (model, case, seed) must produce
exactly identical deterministic output and the identical deterministic hash.**
**Different seeds are not expected to produce identical payload hashes, and are
not required to.** All challenger randomness is drawn from generators seeded
through the existing SHA-256 token convention
(`f"{base_seed}:{origin_date}:{horizon_days}:{token}"` → `int(digest[:8],16) % 2_147_483_647`),
with these reserved new tokens and no others:

```
election_noise_v2_a_index     Challenger A atom index
election_noise_v2_a_kernel    Challenger A kernel noise z
election_noise_v2_a_loeo      Challenger A bandwidth-selection scoring
election_noise_v2_b_normal    Challenger B Gaussian draw
```

The control's `residual_index` and `sign_draw` tokens are untouched. No
wall-clock, no PID, no environment-dependent value, and no unordered-set
iteration may enter any new code path. A targeted test asserting byte-identical
repeat runs at fixed (model, case, seed) is required.

**G7 — no tuning on 2026.**
Challenger A's `h*` is produced solely by LOEO-FIT inside training pools;
Challenger B has no tunable hyperparameter. **The 2026 forecast under any
challenger may be computed only after the gate has been evaluated and its result
recorded**, and its value may not be used to revise any specification, tolerance
or metric in this document. Any deviation voids the preregistration and requires
a new one.

### F.4 Resolution rules — also fixed now

* **Neither challenger passes** → RC1 remains production. This is an acceptable,
  fully preregistered outcome and is not a failure of the exercise.
* **Both pass** → prefer the lower Tier-1 five-seed mean ES. If within 0.5 %
  relative, prefer the model with fewer free parameters (B over A); if still tied,
  prefer the more conservative (A at the smaller `h`).
* **A passing challenger is not thereby released.** Adoption requires a separate
  release decision, a new freeze audit, and a new immutable publication — none of
  which this document authorizes.
* **λ disclosure (§D5):** extreme clipping in a passing challenger is reported for
  scientific review before promotion. It is never a gate and never a reason to
  change the transfer.
* **Reported last, gated on nothing:** the 2026 C+S+MP and S+V+MP seat
  distributions under each model, for transparency only.

---

## G. Invariants

Every comparison holds these fixed, byte-for-byte, across CONTROL, A and B:

* OpinionState v1.1 (estimation, house effects, Kish weighting, ALR covariance)
* Dynamics v2 (`symmetric_all_history`, ± sign symmetry, no `√h` scaling)
* GeographicProjection v1 (IPF on the chronological baseline) and its baselines
* Exact-margin biproportional controlled rounding; 6 500 000 pseudo-votes;
  constituency totals as multiples of 25
* MandateAllocator v1 (310 fixed + 39 adjustment, `Fraction(6,5)` first divisor,
  integer cross-product thresholds, keyed deterministic lottery)
* **The bounded simplex transfer (§I item 6).** `apply_batch_simplex_transfer`,
  the λ rule and `ε = 0.01` pp are an **invariant downstream transformation**.
  Every candidate passes through the identical rule; **continuous challengers are
  not given a different feasibility mapping.** Only the distribution generating
  `R` changes. λ behaviour is recorded as a mandatory diagnostic (§D5) and never
  used for selection.
* Production seed conventions and the base seed `12345`
* The five-seed set and `N = 20 000` per seed (§D0); `100 000` draws for any 2026
  forecast computed after the gate
* Historical as-of construction, the 14-day consensus window, and the strict
  chronological pool rule
* The frozen Tier-1 **candidate target set** `{2010, 2014, 2018, 2022}` (§E.2),
  which may never expand, and the frozen Tier-2/Tier-3 and Tier-3-ISO case sets
* **(Amendment 2)** On Tier 3-ISO: the consensus estimator
  (`build_election_polling_consensus`, 14-day window, unchanged), geography mode
  restricted to `chronological` with `oracle` prohibited, `total_national_votes`
  left unset so the scale comes from the baseline election, and the statutory law
  taken from `mandate_law_for_election_year`
* The nested pool-size minima `K_outer ≥ 3` and `K_inner ≥ 2`, with `K_inner = 1`
  prohibited, applied identically to all three models (§C, §E.2, §E.4)
* The tactical-voting decision (rejected; no threshold behaviour is modelled)
* Pollster treatment, deduplication, and house-effect handling
* REST semantics (aggregate ineligible mass, never receives seats)
* Every other component of RC1

---

## H. Explicit non-goals

This preregistration does **not**:

* delete, down-weight, or exclude the 2002 residual;
* introduce recency weighting of any kind;
* cap, winsorize, or clip large residuals;
* alter the current published forecast, `current.json`, the prospective archive,
  or any immutable publication;
* publish a challenger or touch the website;
* change displayed probability precision or any presentation contract;
* tune anything for C+S+MP or any other named coalition;
* claim that multimodality is itself an error;
* alter the geography or mandate-allocation algorithm for any reason, including to
  admit an older election (§E.3);
* alter the simplex transfer or λ rule for any reason (§G);
* **(Amendment 2)** introduce a new polling estimator, reconstruct a historical
  Poll-of-Polls state series, treat the retrospective `pofp` series as
  leakage-safe, alter production polling inputs, or discard or rewrite the existing
  full-pipeline Tier-2/Tier-3 results.

---

## I. Reviewer decisions — resolved and frozen

| # | Decision | Resolution |
|---|---|---|
| 1 | Tier roles | *(Amendment-1 record — **superseded in part by item 12**.)* **Tier 1 = required improvement** on the primary joint vote ES; **Tier 2 = hard non-inferiority** on integrated metrics; coalition Brier a **separate required-improvement** criterion after the identical geography + exact allocator. **No compensation** of a degradation by an improvement — this principle survives Amendment 2 unchanged. Amendment 2 moved the non-inferiority family from Tier 2 to Tier 1 and the seat criteria to Tier 3-ISO; the tolerances did not change. (§F.2) |
| 2 | Tolerances | Frozen: improve ≥ **2.0 %** relative; non-inferiority ≤ **1.0 %** relative; coverage deviation may not grow by more than **3.0 pp**. Not to be retuned. (§F.1) |
| 3 | A's grid and small-K rule | Grid `{0.25, 0.50, 0.75, 1.00}` retained; **`h = 0` not added** (CONTROL is the unsmoothed model). **The K = 2 outer fallback is removed.** **Amendment 1 clarification:** outer evaluation eligibility requires **`K_outer ≥ 3`** (applied identically to CONTROL, A and B); inner LOEO-FIT folds may run at **`K_inner = 2`**, which is explicitly allowed and is not the removed fallback; **`K_inner = 1` is prohibited**, which is exactly why an outer `K_outer = 2` target is excluded. (§C, §E.2, §E.4) |
| 4 | Monte Carlo design | **Five fixed seeds** `12345, 24680, 98765, 54321, 13579`; **N = 20 000 per seed per case per model**; per-seed scores reported with the five-seed mean and SD; **gate applied to the five-seed mean**; no seed selection or favourable pooling. G6 clarified: determinism is per (model, case, seed); different seeds need not match. (§D0, §F.3 G6) |
| 5 | B's distribution | **Gaussian, frozen. Student-t rejected** for this competition; a heavy-tail challenger only in a future preregistered experiment. Covariance conventions (K vs K−1 / Bessel) written out explicitly and declared binding. (§C) |
| 6 | Simplex transfer / λ | **Invariant downstream transformation** for every candidate; identical feasibility mapping for discrete and continuous models. λ diagnostics (fraction < 1, mean, p01/p05/p10, min, binding donor counts) are **mandatory records, never selection criteria**; **no clipping-frequency gate may be added after results**; extreme clipping in a passing challenger is reported for scientific review before promotion, never fixed by silently changing the transfer. (§D5, §G) |
| 7 | Historical seat/geography evidence | The 2018/2022-only Tier-3 set is **not permanently fixed**. Part 2 must investigate backward extension to 2014 (2010 baseline) and 2010 (2006 baseline) under the explicit provenance/inclusion requirements declared in §E.3 **before** the search. The geography algorithm may not be altered to admit an older election; an election that cannot run the identical chronological path with authoritative inputs is rejected from Tier 2. (§E.3) |
| 8 | Horizon weighting | **Equal weighting over all six preregistered horizons retained** for the primary Tier-2 summary; primary weights not changed on the basis of the current 2026 date. **Added:** a short-horizon operational **non-inferiority guard** on the aggregated 14-day and 28-day Tier-2 cases at the same 1.0 % tolerance (G4b). |
| 9 | G5 robustness | Frozen **before** the §E.3 data search. **Amendment 1** replaces the non-operative Tier-1 "3 of 4" threshold with the general **`N_T1 − 1`** rule of §F.3 G5: all `N_T1` leave-one-target-out aggregates must favour the challenger directionally, at least `N_T1 − 1` of them by ≥ 1.0 %, and at least `N_T1 − 1` individual targets must favour the challenger. At `N_T1 = 4` this reproduces the original "3 of 4" exactly; at `N_T1 = 3` it yields "2 of 3" by the same formula. `N_T1 < 3` ⇒ STOP. Coalition Brier: the `N_elections` (= `N_seat`) `== 2` / `≥ 3` conditional rule is **unchanged**. |
| 12 | Leakage-safe seat evaluation (Amendment 2) | The historical PoP state is not publication-time leakage-safe (§E.5 item 7), discovered **before** any challenger implementation or score. The authoritative seat/coalition evaluation moves to the isolated ElectionNoise path **Tier 3-ISO** (§E.2a): final 14-day consensus → ElectionNoise → unchanged transfer → frozen geography (chronological only) → historically correct law. Targets `{2014, 2018, 2022}`, **`N_seat` = 3**. The full-pipeline Tier-2/Tier-3 results are preserved and reported as retrospective diagnostics and excluded from the gate. G2 and G4 re-point to Tier 3-ISO; G3 re-points to Tier 1; G4b is retired from the gate and kept as a diagnostic. **No threshold, seed, sample count, Challenger A/B definition, h grid, λ rule or Tier-1 vote gate changed.** (§E.2a, §E.7, §F.2, §F.3) |
| 13 | Vote and seat metrics are not independent (Amendment 2) | Tier 1 and Tier 3-ISO score the same predictive distribution — the seat metrics are a deterministic nonlinear pushforward of the identical draws. They may never be reported or aggregated as independent evidence. CONTROL's law on this path is exactly `K` atoms, so its coalition probabilities are confined to multiples of `1/K`; the informative content of the gate therefore sits in the non-inferiority guards and per-election robustness, not the headline Brier improvement. (§E.7) |
| 11 | Nested-CV K ambiguity (Amendment 1) | Discovered **before** Part 2 and before any challenger implementation or score. `K_outer ≥ 3` governs evaluation eligibility; `K_inner ≥ 2` governs fitting inside the already-preregistered `h` selection; `K_inner = 1` prohibited. Tier-1 candidate targets frozen at `{2010, 2014, 2018, 2022}` and may never expand; `N_T1` is determined mechanically. (§C, §E.2, §E.6, §F.3 G5, §J.2) |
| 10 | Stale audit file | Documented as a **provenance-maintenance caveat** in §0.2. It does not invalidate residual values. **Its repair is explicitly excluded from this preregistration and from any commit carrying it**; a separate maintenance change is recommended, which must prove the regenerated residual pool is unchanged. |

---

## J. Freeze block and amendment history

### J.1 Amendment 0 — superseded freeze record (preserved audit trail)

The preregistration was first frozen as **Amendment 0**. That freeze is valid and
is **not erased**; it is superseded only by Amendment 1 below.

```
PREREGISTRATION STATUS (superseded): FROZEN — AMENDMENT 0

FREEZE TIMESTAMP (UTC): 2026-08-30T20:08:21Z
FREEZE BASE COMMIT:     f4c169dcb25f907cac602bbc3b6c436dde193eaa
FREEZE COMMIT:          ef5aaeba02b5a3d43334d5743381d219c2e80c71
BODY SHA-256:           7222f9a27755b41a4689d639b46f6675808de42ca2a5186a953b9b17965ad1f0
WHOLE-FILE SHA-256:     f8c68d75e220cbb203fe0d2ebdb7b35cf765ca7cdab62325f495dab015b197f8
REVIEWER:               repository owner (decisions of 2026-08-30, applied in full at §I items 1–10)
```

Amendment 0 froze everything in §§A–I except one clause, which it explicitly
recorded as **non-operative** rather than silently rewriting it: the Tier-1
individual-case threshold in G5, written as *"at least 3 of the 4 preregistered
Tier-1 held-out election cases"*, which had no referent once the `K ≥ 3` rule
excluded the 2010 target and left 3 Tier-1 elections. Amendment 0 recorded three
candidate readings and adopted none. **That is the ambiguity Amendment 1 resolves.**

### J.2 Amendment 1 — rationale and scope

**When it was made.** Before the Part-2 historical-data feasibility search, before
any challenger implementation, before any challenger score, and before any 2026
challenger forecast. Nothing empirical about challenger performance exists, so
nothing empirical could have motivated it.

**Why it was made.** Two items required resolution:

1. **An explicit ambiguity.** The Amendment-0 Tier-1 G5 threshold was recorded as
   non-operative and had to be given an operative form before any evaluation.
2. **A nested-validation clarification.** Amendment 0's phrase "Challenger A
   requires `K ≥ 3`" admitted an impossible reading: applied to *every* fit, it
   would forbid the inner leave-one-out folds that A's own already-preregistered
   `h`-selection procedure depends on (an eligible `K_outer = 3` target
   necessarily produces `K_inner = 2` folds). Amendment 1 separates the outer
   eligibility minimum from the inner fitting minimum so the preregistered
   procedure is executable as written.

**What it does and does not do.**

* It **generalizes** the originally intended robustness requirement rather than
  weakening or strengthening it after seeing performance. At `N_T1 = 4` the new
  `N_T1 − 1` rule reproduces the original "3 of 4" **exactly**; at `N_T1 = 3` it
  yields "2 of 3" from the same formula, with no post-hoc discretion.
* It **fixes the Tier-1 candidate target set** at `{2010, 2014, 2018, 2022}` and
  forbids it from expanding, so no target can be added after the data search
  because its result is favourable.
* It makes `N_T1` and `N_seat` **mechanically determined** by frozen eligibility
  rules, and forbids Part 2 from changing any other rule (§E.6).
* It **does not** alter the coalition-Brier conditional rule, the metrics, the
  tolerances, the seed policy, the sample counts, the horizon weights, the `h`
  grid, or the Challenger A/B specifications.
* It **does not** touch the stale `contributing_polls_audit.csv` maintenance item
  (§0.2, §I item 10), which remains separate and does not block Part 2.

**Current instantiation.** Under the existing residual history `N_T1 = 3`
(eligible: `{2014, 2018, 2022}`; 2010 excluded at `K_outer = 2`). If Part 2 admits
a pre-2002 residual election under §E.3, 2010 reaches `K_outer ≥ 3`, `N_T1` becomes
4, and the `N_T1 = 4` column of the §F.3 G5 table applies automatically with no
further decision.

### J.3 Amendment 1 — superseded freeze record (preserved audit trail)

```
PREREGISTRATION STATUS (superseded): FROZEN — AMENDMENT 1

FREEZE TIMESTAMP (UTC): 2026-08-30T20:21:28Z
AMENDMENT 0 COMMIT:     ef5aaeba02b5a3d43334d5743381d219c2e80c71  (superseded, §J.1)
FREEZE BASE COMMIT:     ef5aaeba02b5a3d43334d5743381d219c2e80c71
FREEZE COMMIT:          the commit that introduces this block on branch
                        diagnostic/election-noise-mixture (reported with the
                        amendment; it cannot be embedded in the file it hashes)
FREEZE COMMIT (actual):  80b1c671c4b6d879a888f28a859ee392e8f59bc5
BODY SHA-256:           bac3ca06e52cc07fe74ca9e5aa785d94e30934db32193c7f948e95a49a6ae075
WHOLE-FILE SHA-256:     03dc843bd73d12c51a8deb7f727a7a4a29a198ee8513d0df5dd8c1fd309e5a97
REVIEWER:               repository owner (Amendment-1 instruction of 2026-08-30)
SCOPE:                  ambiguity resolution only; pre-data, pre-challenger
OPEN CLAUSES:           none
```

Amendment 1 resolved the non-operative Tier-1 G5 threshold and split the nested
`K_outer` / `K_inner` minima. Everything it froze remains in force except where
Amendment 2 explicitly re-points a gate criterion; **no threshold, seed, sample
count, challenger specification, `h` grid, λ rule or Tier-1 vote gate was changed
by Amendment 2.**

**BODY SHA-256** — the hash covers every byte of this document **strictly before
the last occurrence** of the freeze-block start marker, which is the real marker
at the end of the file (earlier occurrences of the same string are prose and code
references and are inside the hashed body). The hash value is printed immediately
after that marker. This scope is used because a whole-file hash cannot be embedded
in the file it hashes, so the value must sit outside the region it covers. It is
verifiable with:

```bash
python3 - <<'PY'
import hashlib, pathlib
b = pathlib.Path("docs/election_noise_v2_preregistration.md").read_bytes()
body = b.rsplit(b"<!-- FREEZE-BLOCK-START -->", 1)[0]   # LAST occurrence
print(hashlib.sha256(body).hexdigest())
PY
```

The whole-file SHA-256 of the frozen file and the Amendment-1 commit hash are
recorded in the amendment commit message and, for Amendment 1, above.

### J.4 Amendment 2 — rationale and scope

**When it was made.** After Part 3B (`89d3408`) and **before any challenger was
implemented, before any challenger score existed, and before any 2026 challenger
forecast**. Nothing empirical about challenger performance existed, so nothing
empirical about challenger performance could have motivated it.

**Why it was made.** Part 3B established that the historical Poll-of-Polls state
series consumed by the full-pipeline hindcasts is a fieldwork-dated rolling
aggregate which the provider retrospectively completes as later-published polls
arrive (§E.5 item 7). The seat and coalition metrics that the adoption gate turns
on were therefore computed from a state a forecaster could not have observed at the
forecast origin. That the leakage is common-mode across models makes the
*comparison* survivable, but it does not make an adoption decision on a
non-leakage-safe evaluation acceptable, and it was not accepted on that basis.

**What it does.**

* Introduces **Tier 3-ISO** (§E.2a) as the authoritative seat/coalition evaluation:
  the final 14-day polling consensus — the existing publication-date-safe estimator
  already used to build the ElectionNoise residuals, with **no new estimator
  invented** — through ElectionNoise, the unchanged simplex transfer, the frozen
  geography in `chronological` mode only, and the historically correct mandate law.
* Sets the seat/coalition targets to `{2014, 2018, 2022}`, **`N_seat` = 3**. 2014
  becomes admissible because it satisfied every §E.3 requirement in Part 2B and its
  earlier exclusion was caused solely by Tier 2's dependence on OpinionState.
* Moves the full-pipeline Tier-2 and Tier-3 evaluations out of the gate and
  **preserves** them as retrospective diagnostics. Their existing Part-3 results are
  not discarded or rewritten.
* Re-points G2 and G4 to Tier 3-ISO, re-points G3 to Tier 1, and retires G4b from
  the gate while keeping it as a Tier-2 diagnostic — with the resulting reduction in
  gate coverage recorded openly rather than papered over.
* Records in §E.7 that the vote and seat evaluations are **not independent
  evidence**, and that CONTROL's coalition probabilities on this path are confined
  to multiples of `1/K`.

**What it does not do.** No threshold, tolerance, seed, sample count, mask set,
aggregation rule, `h` grid, λ rule, simplex-transfer rule, Challenger A or
Challenger B specification, or Tier-1 vote gate is changed. The coalition-Brier
conditional rule is untouched; `N_seat` = 3 merely selects the `≥ 3` branch that
was already frozen in Amendment 1. No new polling estimator, no reconstructed PoP
series, no production polling input, and no `compute_discrete_energy_score` change.

**Audit evidence.** `diagnostics/election_noise_v2/isolated_seat_evaluation/` —
geography input classification with an electorates perturbation test, consensus
publication-safety per target, a poll-archive revision test, and a three-target
smoke test producing valid 349-seat draws. No predictive score was computed.

### J.5 Amendment 2 — operative freeze record

```
PREREGISTRATION STATUS: FROZEN — AMENDMENT 2

FREEZE TIMESTAMP (UTC): 2026-08-31T08:54:20Z
AMENDMENT 0 COMMIT:     ef5aaeba02b5a3d43334d5743381d219c2e80c71  (superseded, §J.1)
AMENDMENT 1 COMMIT:     80b1c671c4b6d879a888f28a859ee392e8f59bc5  (superseded, §J.3)
FREEZE BASE COMMIT:     89d340880a4bdb389f94ce61fa3333799b58d81a
FREEZE COMMIT:          the commit that introduces this block on branch
                        research/isolated-seat-evaluation (reported with the
                        amendment; it cannot be embedded in the file it hashes)
BODY SHA-256:           recorded after the marker at the end of this file
WHOLE-FILE SHA-256:     recorded in the amendment commit message (it cannot be
                        embedded in the file it covers)
REVIEWER:               repository owner (Amendment-2 instruction of 2026-08-31)
SCOPE:                  leakage-safety correction of the seat/coalition tier;
                        pre-challenger, pre-score
OPEN CLAUSES:           none
```

### J.6 What may happen next

Re-certification of the CONTROL baseline under the amended evaluation (Tier 1
unchanged, Tier 3-ISO added, full-pipeline results retained as diagnostics), and
only then Challenger A/B implementation. Challenger scoring and any 2026 challenger
forecast remain prohibited until that recertified baseline has been reviewed.

<!-- FREEZE-BLOCK-START -->
All bytes above this marker line are covered by BODY SHA-256.

BODY SHA-256: 5a9a6dc8ef6f26ce3ce152155af0ed288fb8d2d97c81a2606e513cf20e1b058b
<!-- FREEZE-BLOCK-END -->
