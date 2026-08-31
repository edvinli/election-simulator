# Amendment-2 CONTROL baseline and frozen evaluator — method

**Research infrastructure. CONTROL only.** No challenger implemented or scored, no
adoption gate evaluated, no 2026 forecast produced, the preregistration not edited.

| | |
|---|---|
| Preregistration | **FROZEN — AMENDMENT 2**, commit `00f7030`, body SHA-256 `5a9a6dc8ef6f26ce3ce152155af0ed288fb8d2d97c81a2606e513cf20e1b058b` |
| Predecessors | Part 3 baseline `998a200` · Part 3B `89d3408` · Part 3C audit `7f37e12` |
| Model | `CONTROL_pp_centered_noise` — unmodified production ElectionNoise |
| Case set | Tier 1: 2014, 2018, 2022 (**`N_T1` = 3**) · Tier 3-ISO: 2014, 2018, 2022 (**`N_seat` = 3**) |

---

## 1. Case set

Both gate tiers now run on the same three elections, from the same
publication-safe origin.

| Tier | Level | Cases | Metrics |
|---|---|---|---|
| **Tier 1** | vote | 3 (one per election) | D1, D2, D5 |
| **Tier 3-ISO** | seat / coalition | 3 (one per election) | D3, D4, D5 |

Tier 3-ISO, per Amendment 2 §E.2a:

| Target | Base | Training residuals | `K` | Geography baseline | Mode | Law | Divisor |
|---|---|---|---|---|---|---|---|
| **2014** | final 14-day publication-safe consensus | 2002, 2006, 2010 | 3 | 2010 | chronological | **PRE_2018** | **7/5** |
| **2018** | final 14-day publication-safe consensus | 2002, 2006, 2010, 2014 | 4 | 2014 | chronological | POST_2018 | 6/5 |
| **2022** | final 14-day publication-safe consensus | 2002, 2006, 2010, 2014, 2018 | 5 | 2018 | chronological | POST_2018 | 6/5 |

**Oracle geography mode is forbidden** and enforced by
`isolated.assert_geography_mode`, which raises `GEOGRAPHY MODE VIOLATION`; the guard
is tested. `total_national_votes` is left unset so the projection scale comes from
the baseline election. 2010 is excluded everywhere.

The consensus builder re-checks, for every retained poll, that
`publication_date <= election_date` and `interview_end <= election_date`, and raises
if not — the leakage guarantee is asserted at run time, not just documented.

### Superseded full-pipeline results

The Part-3 Tier-2/Tier-3 outputs are **preserved byte-for-byte** at
`diagnostics/election_noise_v2/control_baseline/`, never recomputed and never
deleted. The manifest records their SHA-256 and labels their role
*"RETROSPECTIVE DIAGNOSTICS ONLY — excluded from the adoption gate"*, with the
reason (their historical Poll-of-Polls state input is not publication-time
leakage-safe). `validate_manifest` re-hashes them on every run and fails if any
changed; a test asserts the same.

## 2. Monte Carlo design

The frozen design, unchanged: seeds **12345, 24680, 98765, 54321, 13579**,
**N = 20 000 draws per seed** per case per model. Every case is scored separately
for every seed; the five-seed mean, SD and all five individual values are reported.

Because the vote → seat map is deterministic, identical vote rows are **memoised**.
For CONTROL this collapses 20 000 draws onto `K` distinct geography+allocator
evaluations. `verify_memoisation_is_exact` asserts the memoised result equals a
per-draw evaluation row for row, so this is a pure optimisation, not an
approximation. A continuous challenger simply misses the cache on every draw, so the
runner does not depend on the collapse.

## 3. Exact finite-support CONTROL oracle

CONTROL's law on this path is uniform over the `K` centered residual atoms, and
**λ ≡ 1 at all three targets** (verified), so each atom maps deterministically
through consensus → transfer → geography → allocator. The predictive distribution is
an exact `K`-point law, and every quantity is computed analytically:

* the `K` vote vectors and the `K` seat vectors, one per residual year;
* exact mean vote and mean seat vectors;
* exact per-party seat support probabilities;
* exact coalition-majority probability for all 254 masks — necessarily a multiple
  of `1/K`;
* exact coalition Brier against the certified truth;
* exact seat-vector energy score.

This is a **validation artifact**, not a replacement for the preregistered
five-seed Monte Carlo baseline.

Two definitional notes carried from Part 3:

* The energy score of a uniform `K`-atom law uses the `1/K²` dispersion
  normalisation, which is the limit of `compute_energy_score` on Monte Carlo draws.
  `compute_discrete_energy_score` normalises by `K(K−1)` and is deliberately
  **not** used anywhere.
* Quantiles of a discrete law are step functions of the empirical atom weights, so
  **interval coverage is not a continuous functional of the atom probabilities** and
  cannot be expected to converge smoothly. It is reported and flagged rather than
  numerically compared against the oracle.

## 4. Monte Carlo versus exact

Compared per election and per seed, with an a-priori tolerance rather than one
chosen after seeing the numbers: the atom-frequency standard error is
`sqrt(p(1−p)/N)` with `p = 1/K`, and the pass condition is that the worst
coalition-probability error stays inside **5σ**. Every reported quantity — max and
mean |Δp_m|, Brier error, seat-mean error, vote-mean error, seat-ES error and
atom-frequency error — is in `monte_carlo_vs_exact.json`. Any systematic discrepancy
is a blocker; none was found.

## 5. Paired randomness contract (frozen)

Recorded in `evaluator_freeze.json`. The key structural fact:

> On the Tier 1 / Tier 3-ISO isolated path **every non-ElectionNoise input is
> deterministic**. The consensus is a fixed function of the archived polls;
> geography, integerisation and the allocator are deterministic maps. There are
> therefore **no upstream random draws to pair** — pairing is exact by construction,
> which is strictly stronger than the full-pipeline case where OpinionState and
> Dynamics draws had to be matched.

Identical and immutable for every model: the consensus per election, the
chronological geography baseline and mode, law dispatch, the certified truth
vectors, case selection, N, the seed list, the mask set and threshold, and the
simplex transfer.

Seed derivation is unchanged:
`token = f"{base_seed}:{origin}:{horizon}:{label}"`, `subseed = int(sha256(token)[:8], 16) % 2_147_483_647`,
with origin = election date and horizon = 14 on this path. CONTROL consumes
`residual_index` / `sign_draw`; challengers consume only
`election_noise_v2_a_index`, `..._a_kernel`, `..._a_loeo`, `..._b_normal`.

**Common random numbers — available, deliberately not used.** Challenger A's atom
index has the same marginal law as CONTROL's, so reusing CONTROL's `residual_index`
stream would be a mathematically valid CRN pairing and would reduce comparison
variance. The preregistration reserves A its own stream, so reusing CONTROL's would
be a preregistration change and is prohibited. The forgone variance reduction is
accepted and recorded rather than taken silently. No artificial pairing is imposed
on A's kernel noise or B's Gaussian draw, which have no CONTROL counterpart.

## 6. Brier interpretation, carried forward

CONTROL's coalition probabilities are **structurally coarse**: `K` is only 3, 4 or 5,
so `p_m` is confined to multiples of `1/K` — at 2014, only `{0, ⅓, ⅔, 1}`. A
continuous challenger can express intermediate probabilities CONTROL structurally
cannot, and may therefore clear the ≥ 2 % aggregate Brier improvement threshold
relatively easily.

**The threshold is not changed and no gate is added.** The eventual decision must
rest on the complete frozen gate: Tier-1 primary joint vote improvement, marginal
non-inferiority, seat-vector non-inferiority, election-level robustness, and
coalition-Brier robustness across elections.

## 7. Evaluator freeze

`evaluator_freeze.json` records 46 hashes: the Amendment-2 preregistration body and
whole-file hashes, the case manifest, every metric and path implementation file
(including the reused Part-3 metric module and the production estimators,
allocator, law module, geography and consensus), every truth input, the exact
oracle, every baseline artifact, and the preserved Part-3 diagnostics — plus the
seed/N policy, the case set with its law and geography restrictions, and the paired
randomness contract.

**Part 4 must call `freeze.verify()` before implementing a challenger.** Any changed
hash is a hard stop: either the evaluator drifted, or the change must be
preregistered first.

## 8. Outputs

| File | Contents |
|---|---|
| `evaluation_case_manifest.json` | authoritative Amendment-2 case set, eligibility, seed streams, truth vectors, input hashes, preserved-diagnostic hashes |
| `control_scores_by_case_seed.csv` | one row per (tier, election, seed) — every metric, unpooled |
| `control_scores_by_election.csv` | election-level aggregates per tier |
| `control_scores_summary.json` | full aggregation, the Tier-1-unchanged check, oracle agreement, Brier interpretation |
| `coalition_brier_by_election.csv` | D4 per election with the exact value beside it, and the headline |
| `mask_level/coalition_brier_by_mask.csv` | every (election, seed, mask) row for independent verification |
| `exact_control_support.csv` | the `K` atoms per election: residual year, vote vector, seat vector, selected coalition sums |
| `exact_control_oracle.json` | full exact oracle incl. all 254 per-mask exact probabilities and the memoisation check |
| `monte_carlo_vs_exact.json` | per-election and per-seed MC-vs-exact residuals with the 5σ tolerance |
| `lambda_diagnostics.csv` | D5 per case and seed |
| `evaluator_freeze.json` | the freeze artifact |
| `evaluation_method.md` | this document |

The original Part-3 baseline directory is **not** overwritten.
