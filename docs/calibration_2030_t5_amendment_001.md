# T5 amendment 001: add shadow arm `Ul`, log-ratio drift

**Status: DRAFT, not in effect.** Drafted 2026-11-18 on branch
`prereg/calibration-2030-drafts`. It takes effect only when all of the
following hold:

- the user approves it;
- it is finalized with its actual timestamp;
- it is committed with its JSON record, the record's SHA-256 companion and an
  entry in `data/processed/calibration_2030/t5_amendments/index.json`.

It is append-only. It amends `docs/calibration_2030_t5_shadow_preregistration.md`
(frozen, body SHA-256 `bf93c128…`) §2 by adding one arm. Nothing else in T5
changes: cadence, targets, metrics, seeds of existing arms, archive rules,
decision rule.

## Why

**The problem.** The final round's T2 selected `Ua`: the filter posterior × k,
then a random walk in percentage points. A Gaussian random walk in pp is
symmetric in points, whatever a party's size. At multi-year horizons it puts
real probability below zero for parties near the 4 % threshold, which the
simplex map then clips to 0.01 pp.

**Measured on the committed code.** The figures come from
`t2_arms.state_dynamics_draws("Ua", …)` with 20,000 draws, computed on
2026-11-18. No outcome is involved. The origin is 2026-11-08, the last T2
origin, with k = 0.5; the target is 2030-09-08.

| party | draws at ≤ 0.02 % | 5th percentile |
| --- | --- | --- |
| L | 7.6 % | 0.01 % |
| C | 2.5 % | |
| KD | 3.0 % | |
| V | 1.8 % | |
| MP | 2.1 % | |

P(L < 4 %) is 0.52.

P(share < 4 %) is a headline quantity, and seat outcomes hinge on it.

**Why the gates missed it.** T2 scored CRPS averaged over eight parties, and
T4 gates threshold Brier only up to 112 days. Neither isolates this.

**How it was found.** This was found after T2's results existed, while
planning the production integration. The one-round budget (final round §1.1)
forbids a retrospective redesign. Under T5 §1 the only admissible route to a
2030-cycle change is prospective evidence. Hence a shadow arm.

## The arm

`Ul` is `Ua` with its total pp deviation applied **multiplicatively**, on a
log-ratio scale, instead of additively.

For each capture origin `O`, target `D` and draw `j`:

1. **Ua's pre-map draw.** Take Ua's draw before the simplex map:
   `y_j = m + k·(z_state,j Lᵀ_state) + z_dyn,j Lᵀ_dyn` (eight parties, pp).
   Everything is as in `diagnostics/calibration_2030/t2_arms.py::state_dynamics_draws`
   for arm `Ua`:
   - `m` is the v0.3 reading;
   - `Ua`'s dynamics include its election-day shocks over the open interval
     `(O, D)`.
2. **Deviations.** Set `ε_j = y_j − m` for the eight parties, and
   `ε_j,REST = −Σ ε_j`. Set `π = (m, 100 − Σ m)`, the nine-part reading.
3. **The multiplicative map.** Map each category
   `x_j,i = π_i · exp(ε_j,i / max(π_i, 0.5))`, then renormalize `x_j` to sum
   to 100.
4. **Ω.** For poll targets, add the common poll-average term `N(0, Ω_O)` after
   step 3, as for every T2 arm. For the election day, apply the interim noise
   law as T5 §2.1 does.

**Properties.**

- **To first order, `Ul` equals `Ua`.** `π·exp(ε/π) ≈ π + ε`, so for short
  horizons and large parties the two arms coincide.
- **Away from first order, shares stay positive.** A party's spread becomes
  proportional on a log scale. A small party can fall toward zero but never
  reaches it, and its upside is not cut.
- **The 0.5 pp floor** applies only inside the exponent's denominator. It
  bounds the transform for a category whose reading is tiny (REST in some
  years). It is not a clip on the draw.

**Inflation k.** `Ul` computes its own k by final round §3.5, exactly as each
U arm does for itself. The procedure is unchanged; only the arm's own k = 1
predictive enters.

**Random numbers.** `Ul` uses exactly `Ua`'s `z_state` and `z_dyn` at every
(capture, target), from T5 §3.3's generator. So `Ul` is a deterministic
transform of `Ua`'s draws, and the `Ua`–`Ul` comparison is perfectly paired.

**Logged.** The same as every T2 arm in T5 §2.1:

- poll targets h ∈ {30, 91, 182, 365, 730} as 1 % quantiles;
- the 2030 election-day draws as uint16 centi-pp;
- k in `diagnostics.json`.

**From when.** From the first capture whose origin is on or after the date
this amendment takes effect. No backfill.

## Disclosure and limits

- **Post hoc.** The arm was proposed after T2's results were seen. That is why
  it is a prospective shadow arm and not a retrospective challenger.
- **No retrospective scoring.** `Ul` will not be scored on 2010–2026 data.
  Implementation tests may compare `Ul` with `Ua` numerically (the
  first-order equivalence, positivity), but never against outcomes.
- **Decision use.** Adopting `Ul` later needs exactly what T5 §6 requires for
  any change:
  - a new preregistration citing T5;
  - at least 26 weekly captures with closed targets at h ≥ 91 days;
  - CRPS as the metric.

  That preregistration should also name a threshold-specific metric, Brier on
  P(share < 4 %) per party, before seeing it (T5 §5.3).
- **Code change.** This amendment authorizes the T5 runner change that adds
  the arm (T5 §4.4). Earlier captures keep their recorded code commit.
