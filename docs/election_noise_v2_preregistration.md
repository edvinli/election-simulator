# Preregistration — ElectionNoise v2 challenger competition

**Status: PREREGISTRATION. Written before any challenger implementation and before
any challenger score exists.**

**RC1 is not modified by this document.** `ElectionSimulator v1.0.0-rc1` remains the
frozen production control. This document authorizes *evaluation only*. Passing the
adoption gate defined in §F does not itself release anything; release remains a
separate, explicit decision.

| | |
|---|---|
| Document version | 1.0 (unfrozen draft — requires sign-off, see §I) |
| Written | 2026-08-30 |
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

### 0.2 Data-provenance caveat found during Part 1

`data/processed/election_residuals/contributing_polls_audit.csv` is tracked from
commit `f55bf36` and carries `poll_id` values from a superseded id scheme; the
`poll_id`s in the current `swedishpolls_individual_polls.csv` differ. Re-running
`tests/test_election_residuals.py` regenerates the audit file with current ids.
**Only the id column is stale — pollster, dates, sample sizes, weights and every
support value are unchanged, so the residual pool itself is unaffected.** The file
was restored to `HEAD` and not modified. Anyone auditing the residual pool should
regenerate this artifact rather than read the committed copy. This is recorded
here because the residual pool is the object under test; it is not part of the
competition.

### 0.3 Preconditions that must hold before any challenger code is written

1. This document is reviewed and frozen (a `FROZEN` marker plus the reviewer's
   sign-off appended to §J, and its content hash recorded).
2. No challenger has been implemented, run, or scored.
3. The 2026 forecast under any challenger has not been computed.

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
  `rank(C_P) ≤ K−1` (≤ 5 for the 2026 pool).
* `S_P = C_Pᵀ C_P / K` — maximum-likelihood covariance about the fitted mean.
  This is exactly the covariance of the CONTROL's atom distribution.
* `P₉ = I₉ − (1/9) 𝟙𝟙ᵀ` — orthogonal projector onto the zero-sum subspace, rank 8.
* `R` — the residual vector drawn by the layer for one simulation draw.

All three models emit a zero-sum `R ∈ R⁹` in pp and hand it to the **unchanged**
production transfer `apply_batch_simplex_transfer(x, R, eps=0.01)`.

### CONTROL — six-atom centered empirical bootstrap

```
R = c_{k},   k ~ Uniform{1, …, K}
```

Free/tunable hyperparameters: **0**.
Support: K points. Covariance: `S_P` exactly.

### CHALLENGER A — variance-corrected smoothed empirical bootstrap

A conservative continuous smoothing of the control that preserves the historical
joint structure exactly and replaces the six point masses with six Gaussian
kernels.

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
into the comparison would confound the thing being tested.

**Centering.** Identical to the control: the production
`load_chronological_pp_residuals` performs it, unchanged.

**Covariance.** `S_P`, the maximum-likelihood covariance — deliberately *not*
Bessel-corrected, so that `Cov(R) = S_P` exactly and **A nests the CONTROL as
h → 0**. This nesting is the point of the family: A is the control plus one
smoothing dial and nothing else.

**Smoothing / bandwidth rule.** One scalar `h`. Grid
`H = {0.25, 0.50, 0.75, 1.00}`. Selection is by **LOEO-FIT**, run strictly inside
the training pool and never touching the target election (§E):

```
for each h in H:
    score(h) = (1/K) Σ_{j∈P}  ES( F^A(h, P\{j}),  r_j − r̄_{P\{j}} )
h* = argmin_h score(h);  ties broken toward the smallest h (most conservative)
```

`F^A(h, P\{j})` is Challenger A fitted on the pool with election `j` removed,
re-centered by the production centering algorithm on `P\{j}`; the held-out target
is the held-out residual expressed in that same centering. `ES` is the energy
score of §D1, estimated from `M = 20 000` draws under seed token
`":election_noise_v2_a_loeo"`. Requires `K ≥ 3`; for `K = 2` (target election
2010) the rule is undefined and `h := 0.25` (the smallest grid value) is used.
This fallback is preregistered here and may not be changed later.

**Tail behaviour.** The marginal law is an equal-weight K-component Gaussian
mixture with component means `c_k/√(1+h²)` and common covariance
`h²S_P/(1+h²)`. Tails are Gaussian. **No heavy-tailed component is included, and
none may be substituted after results are seen.** Six observations cannot support
tail-index estimation; this is a stated limitation, not an oversight.

**Support / degeneracy — disclosed.** `ε ∈ span{c_1,…,c_K}`, which has dimension
`≤ K−1 = 5`. A is therefore continuous but **singular**: it is supported on a
5-dimensional affine subspace of the 8-dimensional zero-sum hyperplane. It cannot
produce an error pattern outside the historical span. This is the deliberate
"conservative" property of the family and the main scientific contrast with B.

**Simplex / composition handling.** `R` is zero-sum by construction (every `c_j`
is zero-sum and `R` is a linear combination of them), to floating-point residue
only; the production zero-sum cleaning in `apply_simplex_transfer` handles the
residue. The production λ rule and `ε = 0.01` pp floor are unchanged.

**REST.** Identical to the control: REST is the 9th residual component, moves with
every draw, enters the vote denominator, and never receives seats.

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
τ²  = tr(S_P) / 8
T   = τ² · P₉                                     (isotropic target on the zero-sum subspace)
d²  = ‖S_P − T‖²
b̄²  = (1/K²) Σ_{j=1}^{K} ‖c_j c_jᵀ − S_P‖²
b²  = min(b̄², d²)
δ   = b² / d²        (δ := 1 if d² = 0)
Σ_LW = δ·T + (1−δ)·S_P
Σ̃_P  = (K/(K−1)) · Σ_LW                            (single Bessel correction)
```

(The `1/8` in the normalized norm cancels in the ratio `δ = b²/d²`, so `δ` does
not depend on the choice of `p_eff`; it is written explicitly only to make the
adaptation of Ledoit–Wolf's scaled-identity target to a rank-8 projector target
unambiguous. `S_P 𝟙 = 0` because every `c_j` is zero-sum, hence `S_P P₉ = S_P`,
`Σ̃_P 𝟙 = 0`, and every draw is zero-sum almost surely.)

Sampling: symmetric PSD square root by eigendecomposition of `Σ̃_P` with
eigenvalues clipped at 0; `R = Σ̃_P^{1/2} z`, `z ~ N(0, I₉)`; then
`R ← R − mean(R)` to remove floating-point residue, matching the control's
zero-sum cleaning.

**Coordinate system / centering / REST / simplex handling:** identical to
Challenger A and to the control.

**Covariance / regularization.** Ledoit–Wolf (2004) shrinkage applied verbatim to
the maximum-likelihood `S_P`, with the scaled-identity target replaced by the
scaled zero-sum projector `τ²P₉` so that every draw is zero-sum almost surely
(`S_P P₉ = S_P`, hence `τ² = tr(S_P)/8` is exactly the projection coefficient).
The single `(K/(K−1))` factor corrects the mean-estimation bias of `S_P` once, at
the end; it is a fixed formula, not a dial. Unlike A, B is **full rank 8** on the
zero-sum hyperplane whenever `δ > 0` and can therefore produce error patterns
outside the historical span.

**Tail behaviour.** Gaussian. Same limitation and same prohibition on post-hoc
substitution as A. The choice of Gaussian over, e.g., a fixed-ν multivariate t is
a preregistered structural decision (§I item 5).

**Free/tunable hyperparameters: 0.**

### Complexity budget

With six elections, the total budget across both challengers is **one** scalar
selected inside the historical design. Any specification requiring more is out of
scope for this competition.

---

## D. Evaluation metrics

All metrics use existing repository implementations, unchanged. Every model is
scored on the same cases, with the same draw count `N = 20 000` per case, using
the existing SHA-256 seed-derivation convention.

### D1. PRIMARY — joint vote metric

**Energy score on the 9-category vote composition, in percentage points, with
Euclidean distance.**

* Implementation: `scripts/vote_share_calibration/energy_score.py::compute_energy_score`, unchanged.
* Representation: `x ∈ R⁹`, pp units summing to 100, order `(M, L, C, KD, S, V, MP, SD, REST)`, **including REST**.
* Distance: `‖·‖₂` on that vector.
* `ES(F, y) = E‖X − y‖₂ − ½·E‖X − X'‖₂`; strictly proper for the joint law; lower is better.
* Truth `y`: the certified election composition in the same 9 categories and units.

**Justification of the representation.** This is the geometry the layer itself
acts in: residuals are pp, the transfer is additive in pp, and the frozen layer's
own validation (`docs/election_layer_v2.md`) is reported in pp. `docs/election_layer_v2.md`
§1 records that the CLR alternative was tested and rejected as catastrophic.
Scoring in pp/Euclidean therefore introduces no untested transformation between
the model and the score.

Secondary read, reported alongside but not the primary gate: the 8-party energy
score (REST excluded), since REST never reaches the seat layer.

### D2. Marginal vote metrics

* **Per-party discrete CRPS** via `scripts/election_layer_v2/forward_eval.py::compute_discrete_crps`, unchanged, for all 9 categories. Headline: mean over the 8 parliamentary parties.
* **Central interval coverage and mean width** at 50 %, 80 % and 90 %, computed as in the existing `election_layer_v2` / `vote_share_calibration` artifacts.
* Existing artifact schemas are retained so the new numbers sit beside the frozen ones.

### D3. Joint seat metric

* **Energy score on the 8-dimensional integer seat vector** via `scripts/seat_hindcasts/metrics.py::calculate_multivariate_energy_score`, unchanged.
* Every vote draw is passed through the **identical** deterministic downstream: geographic IPF projection → exact-margin biproportional controlled rounding → `dispatch_production_allocation` (349-seat Sainte-Laguë with the legal fallback). No part of that path is re-implemented or re-parameterized.
* Truth: the certified seat vectors in `scripts/seat_hindcasts/config.py::EVALUATION_ELECTIONS`.
* Supporting: per-party discrete seat CRPS via `calculate_discrete_seat_crps`.

### D4. Coalition-threshold metric

* **Mask set — fixed in advance, exhaustive.** All `m ∈ {1, …, 254}` over the fixed
  party order `(M, L, C, KD, S, V, MP, SD)`. Masks 0 (empty) and 255 (all eight)
  are excluded and this is stated explicitly: they yield 0 and 349 seats in every
  draw and every realized outcome, so their Brier score is identically 0 for every
  model and carries no discriminating information.
  **Coalitions are never selected on the basis of today's interesting examples.**
* Forecast: `p_m = (1/N) Σ_i 1{ Σ_{p∈m} seats_{i,p} ≥ 175 }`.
* Outcome: `y_m = 1{ Σ_{p∈m} actual_seats_p ≥ 175 }`.
* Score: `B_m = (p_m − y_m)²`.

**Complement duplication — decided before any scoring.** Seats sum to exactly 349
in every draw and every realized outcome, and 349 is odd, so
`{s_m ≥ 175}` is the exact complement of `{s_{255−m} ≥ 175}`. Therefore
`p_{255−m} = 1 − p_m`, `y_{255−m} = 1 − y_m`, and `B_{255−m} = B_m` **identically**.

> **Decision: retain all 254 nontrivial masks symmetrically.** The mean over 254
> masks equals the mean over any set of 127 complement representatives, so this
> choice cannot change any ranking. It is chosen over a representative rule
> because the obvious deterministic representative set (masks 1–127) excludes SD
> from every retained coalition, which reads as a structural bias even though it
> is statistically inert. **The effective number of distinct binary events per
> election is 127, not 254**, and no uncertainty statement may use 254. Applied
> identically to CONTROL, A and B.

**Aggregation — masks are never treated as independent observations.**

1. Per `(election, horizon)` case: `B̄_case = (1/254) Σ_{m} B_m`.
2. Per election: `B̄_E = mean over horizons of B̄_case`.
3. Headline: unweighted mean of `B̄_E` over evaluation elections.

Per-case and per-election values are always reported alongside the headline. The
number of independent realized outcomes is the number of elections (§E), and every
reported summary must state it.

### D5. Mandatory descriptive diagnostics (reported, not gated)

For every model: mean λ, min λ, fraction λ < 1, fraction λ < 0.99, and the binding
donor category when λ < 1; plus the fraction of draws in which any of the eight
parliamentary parties crosses 4 %. These exist because the continuous challengers
can generate residuals more extreme than any historical atom and so may trigger
simplex attenuation more often than the control.

---

## E. Historical evaluation design

### E.1 Leakage rule (all tiers)

For target election `E`, the residual pool is exactly `{e : year(e) < year(E)}`.
This is the production rule and is a leave-all-future-out design. The target
election's own residual never enters the pool, the bandwidth selection, or the
covariance estimate.

### E.2 Three evaluation tiers

| Tier | What varies | Target elections | Cases | Metrics |
|---|---|---|---|---|
| **1 — Standalone forward evaluation** | ElectionNoise only; base composition is the 14-day poll consensus, so OpinionState and Dynamics are absent | 2010 (K=2), 2014 (K=3), 2018 (K=4), 2022 (K=5) | 4 | D1, D2 |
| **2 — Full-pipeline hindcast** | full frozen pipeline | 2018, 2022 × horizons {112, 84, 56, 28, 14, 7} | 12 | D1, D2 |
| **3 — Seat and coalition level** | full frozen pipeline + geography + exact allocator | same 12 cases; geography baselines 2014→2018, 2018→2022 | 12 | D3, D4, D5 |

Tier 1 mirrors `docs/election_layer_v2.md` §4; Tiers 2–3 mirror §5 and
`scripts/seat_hindcasts`. The six horizons constitute the **rolling-origin**
design; no additional origins are introduced.

### E.3 Why Tier 3 stops at two elections — a hard repository constraint

* `data/processed/mandates/historical_certified_mandates.csv` covers **only 2018 and 2022**.
* `data/processed/geography/constituency_party_votes_2014_2022.csv` covers **2014, 2018, 2022**.

A 2014 seat-level target would require a 2010 constituency baseline, which does
not exist in the repository. Adding 2010/2014 seat-level evaluation would require
new data acquisition and is **out of scope** for this competition (§I item 7).
Consequently the coalition-threshold gate rests on **two realized elections**.

### E.4 Two distinct leave-one-out loops — named to prevent confusion

* **LOEO-FIT** — inside the training pool, for Challenger A's bandwidth only (§C). Never sees the target election.
* **LOEO-EVAL** — over the *evaluation* set, used only for the robustness criterion G5: each evaluation election is dropped in turn and the headline metrics are recomputed. This is a stability check on the conclusion, not a model-fitting step.

### E.5 Limitations — to be restated in every report produced under this preregistration

1. The residual pool has at most **six** observations. Every covariance and
   bandwidth quantity is estimated from ≤6 points in a 9-dimensional zero-sum
   space of rank ≤5.
2. The coalition-threshold gate rests on **two** realized elections and 127
   effectively distinct (and strongly dependent) binary events per election. It
   is a **decision rule, not a hypothesis test**. No p-values, confidence
   intervals, or significance claims will be made.
3. Tiers 2 and 3 are **retrospective, not independent holdout**: the model family,
   the polling calibration, and the frozen components were all chosen with
   knowledge of 2018 and 2022.
4. **No prospective validation is claimed.** The 2026 election is unobserved; its
   outcome cannot enter any part of this competition.
5. **Six elections is not large-N evidence and must never be described as such.**
   Any summary that reads as a general claim about Swedish polling error, rather
   than a claim about six observations, is a reporting error.

---

## F. Adoption gate — fixed before any challenger score exists

### F.1 Definitions of the tolerances

All comparisons are relative to CONTROL on the same cases with the same `N`.

| Term | Definition |
|---|---|
| **"improves"** | the challenger's mean metric is strictly lower and by **≥ 2.0 % relative** to CONTROL |
| **"does not materially worsen"** | the challenger's mean metric is **not more than 1.0 % relative** above CONTROL |
| **"coverage does not materially worsen"** | at each of 50 %, 80 %, 90 %, `abs(coverage − nominal)` increases by **no more than 3.0 percentage points** versus CONTROL |

These tolerances are set now, without reference to any challenger score. They are
judgement calls, not calibrated quantities, and are flagged for sign-off (§I
item 2). Once frozen they may not be revised.

### F.2 The gate — all seven criteria must hold

**G1 — joint vote performance improves.**
Tier-1 mean 9-category energy score (D1) **improves** (≥ 2.0 % better), **and**
Tier-2 mean 9-category energy score does not materially worsen (≤ +1.0 %).
*(Tier 1 carries the improvement requirement because it has four elections and
isolates the layer; Tier 2 carries a non-degradation requirement because it has
only two elections but exercises the full pipeline.)*

**G2 — coalition-majority Brier improves.**
Tier-3 headline coalition Brier (D4) **improves** (≥ 2.0 % better).

**G3 — marginal party performance is not materially worsened.**
Mean 8-party CRPS does not materially worsen (≤ +1.0 %) in **both** Tier 1 and
Tier 2; the 8-party energy score does not materially worsen (≤ +1.0 %); and
coverage does not materially worsen at 50/80/90 %.

**G4 — joint seat performance is not materially worsened.**
Tier-3 mean 8-dimensional seat-vector energy score does not materially worsen
(≤ +1.0 %).

**G5 — the improvement does not come from one historical election.**
Under LOEO-EVAL:
* Tier-1 ES (G1): recomputed with each of the four evaluation elections dropped in
  turn, the challenger is strictly better than CONTROL in **all four**
  recomputations, and better by ≥ 1.0 % in **at least three of four**.
* Tier-3 Brier (G2): with only two elections available, the challenger's
  **per-election** Brier must be strictly better than CONTROL's for **2018 and for
  2022 separately**. A challenger better in aggregate but worse in either election
  fails.

**G6 — determinism and reproducibility.**
Two independent executions with identical seed and configuration produce a
byte-identical `deterministic_payload_sha256`. All challenger randomness is drawn
from generators seeded through the existing SHA-256 token convention
(`f"{base_seed}:{origin_date}:{horizon_days}:{token}"` → `int(digest[:8],16) % 2_147_483_647`),
with these reserved new tokens and no others:

```
election_noise_v2_a_index     Challenger A atom index
election_noise_v2_a_kernel    Challenger A kernel noise z
election_noise_v2_a_loeo      Challenger A bandwidth-selection scoring
election_noise_v2_b_normal    Challenger B Gaussian draw
```

The control's `residual_index` and `sign_draw` tokens are untouched. No wall-clock,
no PID, no environment-dependent value, and no unordered-set iteration may enter
any new code path. A targeted test asserting byte-identical repeat runs is required.

**G7 — no tuning on 2026.**
Challenger A's `h*` is produced solely by LOEO-FIT inside training pools;
Challenger B has no tunable hyperparameter. **The 2026 forecast under any
challenger may be computed only after the gate has been evaluated and its result
recorded**, and its value may not be used to revise any specification, tolerance
or metric in this document. Any deviation voids the preregistration and requires a
new one.

### F.3 Resolution rules — also fixed now

* **Neither challenger passes** → RC1 remains production. This is an acceptable,
  fully preregistered outcome and is not a failure of the exercise.
* **Both pass** → prefer the lower Tier-1 ES. If the two are within 0.5 % relative,
  prefer the model with fewer free parameters (B over A); if still tied, prefer
  the more conservative (A at the smaller `h`).
* **A passing challenger is not thereby released.** Adoption requires a separate
  release decision, a new freeze audit, and a new immutable publication — none of
  which this document authorizes.
* **Disclosure requirement (not a gate):** any model with mean λ < 0.98 or
  fraction λ < 1 exceeding 5 % must be flagged prominently in the report, because
  it indicates the continuous residual is being attenuated by the simplex floor
  materially more often than the control.
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
* **The bounded simplex transfer**: the λ rule and `ε = 0.01` pp are held fixed for
  all models; only the *distribution generating `R`* changes. (Flagged for sign-off,
  §I item 6 — this materially constrains the challengers.)
* Production seed conventions and the base seed `12345`
* Simulation counts (`N = 20 000` per evaluation case; `100 000` for any 2026
  forecast computed after the gate)
* Historical as-of construction, the 14-day consensus window, and the strict
  chronological pool rule
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
* claim that multimodality is itself an error.

---

## I. Open methodological decisions requiring sign-off before freeze

These are the choices a reviewer should actively accept or overrule. Each is
currently set to the stated default in the text above.

| # | Decision | Current default | Why it matters |
|---|---|---|---|
| 1 | Which tier carries the improvement requirement for the joint vote metric | Tier 1 (4 elections, layer isolated) improves; Tier 2 (2 elections, full pipeline) must not degrade | Tier 1 has twice the elections but omits OpinionState/Dynamics; Tier 2 is realistic but n = 2 |
| 2 | Numeric tolerances: 2.0 % improvement, 1.0 % non-degradation, 3 pp coverage | as stated | These are judgement calls, calibrated to nothing. With n = 2–4 elections they are the difference between a permissive and an unpassable gate |
| 3 | Challenger A grid `H = {0.25, 0.50, 0.75, 1.00}` and the `K = 2` fallback `h = 0.25` | as stated | A wider grid increases selection noise at K ≤ 5; a narrower one may miss the useful range |
| 4 | `N = 20 000` draws per evaluation case; single seed | as stated | Compute vs Monte Carlo noise. The repo elsewhere uses 5 seeds (`12345, 24680, 98765, 54321, 13579`) for stability; adding that here multiplies cost by 5 but would make small ES differences credible. **Recommended:** add the 5-seed stability check for the two headline metrics only |
| 5 | Challenger B distribution: Gaussian vs fixed-ν multivariate t | Gaussian | Must be decided now — substituting a heavier tail after seeing coalition-tail results would invalidate the preregistration. A fixed `ν = 5` is defensible a priori but is an extra structural constant |
| 6 | Is the simplex-transfer λ rule part of ElectionNoise (mutable) or an invariant? | Invariant | Continuous challengers will hit the ε floor more often than six fixed atoms. Freezing λ isolates the distribution cleanly but may handicap the challengers in a way that is an artifact of the transfer, not of the residual law |
| 7 | Whether to acquire 2010/2014 constituency data to extend Tier 3 | Out of scope | Would take the coalition Brier from n = 2 to n = 4 elections — the single largest available improvement to the evidence base, at the cost of new data acquisition and a new leakage audit |
| 8 | Equal weighting of the six horizons in Tiers 2–3 | Equal | The 2026 production forecast runs at a ~20-day horizon, so 14 and 28 days are the operationally relevant cases. Equal weighting lets 112-day cases, where the layer matters least, carry a third of the weight |

---

## J. Freeze block

This document is **not yet frozen**. On sign-off, append below: the reviewer, the
date, the resolution of every §I item, the commit hash of this file, and the
literal marker `FROZEN`. No challenger implementation may begin before that
marker exists.

```
STATUS:            DRAFT — awaiting review
FROZEN AT COMMIT:  <to be filled on freeze>
SHA-256 OF FILE:   <to be filled on freeze>
REVIEWER:          <to be filled on freeze>
DATE:              <to be filled on freeze>
§I RESOLUTIONS:    <to be filled on freeze>
```
