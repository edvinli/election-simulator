# ElectionNoise v2 — CONTROL vs Challenger A vs Challenger B

**Decision: ADOPT_B** — Challenger B, the Ledoit–Wolf-regularized joint Gaussian
residual model. Both challengers passed every frozen gate; the frozen resolution
rule selects the lower Tier-1 primary joint vote energy score, and B leads A there
by 5.24 % relative. No discretionary override was applied, and no threshold, seed,
case, model definition or bandwidth was changed after scores were observed.

The 2026 forecast was not run and **was not an adoption input**.

## Design as executed

90 runs = 3 models × 2 tiers × 3 elections × 5 seeds, 20 000 draws per seed
(1.8 M draws per model). Seeds 12345, 24680, 98765, 54321, 13579. Tier 1 and
Tier 3-ISO on {2014, 2018, 2022}; `N_T1` = 3, `N_seat` = 3. 2014 under PRE_2018 law
(first divisor 7/5), 2018 and 2022 under POST_2018 (6/5). Chronological geography
only; oracle mode forbidden and unused. Challenger A at its frozen `h* = 0.75` for
all three pools; Challenger B has no tunable hyperparameter.

The full-pipeline Tier-2/Tier-3 results remain retrospective diagnostics and were
not used.

## Headline metrics — five-seed means, unweighted mean over the three elections

| metric | CONTROL | A | B | A vs CONTROL | B vs CONTROL |
|---|---|---|---|---|---|
| **Tier-1 ES 9-category (primary)** | 3.133231 | 2.981095 | **2.832796** | **+4.86 %** | **+9.59 %** |
| Tier-1 ES 8-party | 3.022573 | 2.875321 | 2.744143 | +4.87 % | +9.21 % |
| Tier-1 mean 8-party CRPS | 0.777206 | 0.760843 | 0.764472 | +2.11 % | +1.64 % |
| Tier-3-ISO seat-vector ES | 10.920287 | 10.234760 | 9.722860 | +6.28 % | +10.97 % |
| Tier-3-ISO coalition Brier | 0.026677 | 0.025124 | 0.022230 | +5.82 % | +16.67 % |

Positive = better than CONTROL. Every metric is lower-is-better.

## Coverage — deviation from nominal (the quantity the gate uses)

CONTROL's law on this path has only K = 3/4/5 atoms, so its central intervals are
badly under-nominal. Both challengers move coverage substantially toward nominal.

| level | CONTROL cov | CONTROL dev | A cov | A dev | A change | B cov | B dev | B change |
|---|---|---|---|---|---|---|---|---|
| 50 % | 0.2889 | 21.11 pp | 0.2593 | 24.07 pp | +2.96 pp | 0.3704 | 12.96 pp | −8.15 pp |
| 80 % | 0.4815 | 31.85 pp | 0.5926 | 20.74 pp | −11.11 pp | 0.7407 | 5.93 pp | −25.93 pp |
| 90 % | 0.4815 | 41.85 pp | 0.7407 | 15.93 pp | −25.93 pp | 0.8148 | 8.52 pp | −33.33 pp |

Tolerance is +3.0 pp. A's 50 % level increases deviation by 2.96 pp — inside the
tolerance, but only just, and it is A's tightest gate.

## Tier-1 robustness (G5)

Per election, all six challenger-election pairs beat CONTROL:

| election | CONTROL | A | A rel | B | B rel |
|---|---|---|---|---|---|
| 2014 | 3.567105 | 3.402428 | +4.62 % | 3.211186 | +9.98 % |
| 2018 | 3.865710 | 3.720654 | +3.75 % | 3.512770 | +9.13 % |
| 2022 | 1.966877 | 1.820203 | +7.46 % | 1.774433 | +9.78 % |

Leave-one-target-out aggregates (all must be > 0; at least 2 of 3 must reach 1 %):

| dropped | A | B |
|---|---|---|
| 2014 | +5.00 % | +9.35 % |
| 2018 | +5.63 % | +9.91 % |
| 2022 | +4.17 % | +9.54 % |

All three positive and all three above 1 % for both challengers.

## Coalition Brier (G2, G5)

| election | CONTROL | A | B |
|---|---|---|---|
| 2014 | 0.044615 | 0.045251 *(A loses)* | 0.040105 |
| 2018 | 0.015270 | 0.013969 | 0.011568 |
| 2022 | 0.020148 | 0.016152 | 0.015016 |
| **aggregate** | **0.026677** | **0.025124** (+5.82 %) | **0.022230** (+16.67 %) |

A improves 2 of 3 elections, meeting `ceil(3/2) = 2`; it is worse than CONTROL on
2014 by 1.4 % relative. Leave-one-election-out deltas (none may degrade by more
than 1 %):

| dropped | A | B |
|---|---|---|
| 2014 | +14.96 % | +24.94 % |
| 2018 | +5.19 % | +14.89 % |
| 2022 | **+1.11 %** | +13.71 % |

A's drop-2022 recomputation falls to +1.11 %, still an improvement, so the rule is
satisfied — but it is the point where A is closest to failing.

**Frozen caveat, carried forward unchanged.** CONTROL's coalition probabilities are
confined to multiples of 1/K with K = 3/4/5, so a continuous challenger can clear
the 2 % Brier threshold relatively easily. The threshold was **not** changed on that
basis, and the decision rests on the complete gate with Tier-1 joint vote ES as the
primary criterion — where both challengers also win comfortably.

## Gate summary

| | Challenger A | Challenger B |
|---|---|---|
| G1 Tier-1 improvement ≥ 2 % | PASS +4.86 % | PASS +9.59 % |
| G2 coalition Brier ≥ 2 % | PASS +5.82 % | PASS +16.67 % |
| G3 marginal/interval non-inferiority (6 checks) | PASS | PASS |
| G4 seat-vector non-inferiority | PASS +6.28 % | PASS +10.97 % |
| G5-C individual elections ≥ 2 of 3 | PASS 3 of 3 | PASS 3 of 3 |
| G5-B1 all 3 LOO > 0 | PASS | PASS |
| G5-B2 ≥ 2 of 3 LOO ≥ 1 % | PASS 3 of 3 | PASS 3 of 3 |
| G5-Brier elections ≥ 2 of 3 | PASS 2 of 3 | PASS 3 of 3 |
| G5-Brier LOO ≥ −1 % | PASS worst +1.11 % | PASS worst +13.71 % |
| **All gates** | **PASS** | **PASS** |

G4b was retired from the gate by Amendment 2 and has no Tier-3-ISO analogue.

## Decision

Both passed, so the frozen rule (§F.4) selects the lower Tier-1 primary joint vote
energy score: B at 2.832796 versus A at 2.981095, a 5.24 % relative gap — well
outside the 0.5 % band that would have triggered the fewer-parameters tie rule. B
would have been preferred under that rule too, since it has zero tunable
hyperparameters against A's one.

**ADOPT_B.**

## Monte Carlo stability

Five-seed relative standard deviations are at most 0.69 % (Tier-1 ES ≤ 0.40 %,
coalition Brier ≤ 0.69 %, seat ES ≤ 0.42 %), one to two orders of magnitude below
the 4.9–16.7 % effects the gates turn on. No conclusion here is seed-sensitive, and
all five per-seed values are retained in `scores_by_model_case_seed.csv`.

## Integrity

`score_audit.json`: run valid, no problems. 90 runs, identical case and seed sets
across all three models, no NaN/Inf, every draw summing to 100 with non-negative
shares, every seat vector totalling 349, λ ∈ [0,1] throughout, chronological
geography everywhere with oracle mode unused, correct law dispatch, no future
residual in any training pool, A's bandwidth pinned at 0.75 in every row, all 254
masks present in all 45 seat cells, complement symmetry holding, and all six
re-run cells bit-identical. CONTROL reproduced its certified Part-3D baseline
exactly across 135 metric values. Both freezes verified drift-free before and after
scoring.
