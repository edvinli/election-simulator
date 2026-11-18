# Calibration programme for the 2030 cycle: final round preregistration (T2, T3-rev, CAND, T4)

**Status: FROZEN.** Drafted on 2026-11-18 on branch
`prereg/calibration-2030-drafts` from the binding design spec the user and the
lead decided on 2026-11-18. It was frozen the same day; the timestamp, body
hash and base commit are in the freeze block at the end and in the freeze
record. Changes after the freeze need an append-only amendment.

- **Parent.** `docs/calibration_2030_preregistration.md` (FROZEN, body SHA-256
  `7707c2afb754ff3a5d6ea8f85034321a6f2e0de54117012220c6da46fa66ee53`). It
  defines T1, T3 and the T4 gates G1–G7. This document does not change any of
  them.
- **Amendment in effect.** `docs/calibration_2030_preregistration_amendment_001.md`
  (interim noise law: Gaussian, β = 0.5, post hoc).
- **T5** (the prospective shadow run) has its own preregistration and is not
  covered here.
- **Nothing scored.** No T2, T3-rev or T4 arm has been implemented or scored.
- **Resolved items.** The points the spec left ambiguous, or that the code
  could not execute as written, were resolved by the lead on 2026-11-18. The
  text states the resolutions, and §9 lists them.

This round is not a protocol change of the 2026 prospective benchmark. That
benchmark's dates, selection, weighting and scoring are untouched.

## 0. Why, and what is already known

T1 and T3 are complete (`docs/calibration_2030_t1_t3_results.md`):

- **T1** selected filter `SH` plus rolling consensus weights `W`. Together
  they are the candidate aggregate `SwedishPollsAggregate-v0.3`.
- **T3** selected Gaussian β = 1, and its 2026 flag tripped. The user chose
  Gaussian β = 0.5 as the interim law, post hoc (calibration amendment 001).

Two questions remain open from the parent:

1. **State uncertainty and dynamics** (parent §0 cause 4, §5). OpinionState is
   narrower than the aggregate's own posterior, and dynamics are capped at
   112 days. This is T2.
2. **The noise law after the 2026 flag.** T3-rev tests a hierarchical,
   size-scaled law against the interim law.

The results are combined into one candidate, `CAND` (§5), and scored end to
end by T4 under the parent's frozen gates.

## 1. Rules of the final round

Every rule of the parent §1 applies unchanged: pinned control, pinned data,
causality, procedures locked rather than numbers, 2026 held out where it
motivated a change, common random numbers, report everything, and the
effective-sample caveat. The scoring guard, the clean-checkout check
(`git status --porcelain --untracked-files=no` empty) and the recording of
the commit apply to every run here. The rules below are added.

### 1.1 One-round budget (binding)

This is the **last retrospective design round** on 2010–2026 data for the
2030 model.

- After it, a model change for the 2030 cycle may be justified **only by T5
  prospective evidence**, under a new preregistration that cites that
  evidence.
- No further retrospective tier, arm or rerun on 2010–2026 data may be used
  to justify a 2030-cycle model change.

### 1.2 Pinned inputs

**Control.** `C0` stays production `MODEL_VERSION` 1.2.0-rc1 at commit
`6c4271c`.

**Aggregate.** Every T2, T3-rev and T4 candidate arm uses aggregate **v0.3**
(§2): T1's `SH` filter with its fitted segment hyperparameters, plus rolling
weights `W`.

**Inputs from the parent §1.** Unchanged:

| input | SHA-256 |
| --- | --- |
| `data/processed/pollofpolls/swedishpolls_individual_polls.csv` | `18e84e56eff64a5133eb4eeca9767893a7f74b15724552ef573fd9314e8d2828` |
| `data/processed/elections/riksdag_election_results.csv` | `3a99ffb1b0fbef60d61712764a2136982ecf01c4879a43edc6d9a38ddb4ba0fc` |
| `data/raw/elections/val2026/official_result_manifest.json` | `6bdcbd284f602d9c9b1b9605ab3d52b5700528b54232565f3242d01d107f788d` |
| `data/processed/poll_aggregate/v0_2/swedishpolls_aggregate_timeseries.csv` | `e04bca4a6f90340993f2ca79bfe5d2f17d3ddf8a9e36d885a278048986943761` |
| `data/processed/poll_aggregate/v0_2/swedishpolls_aggregate_metadata.json` | `5c409da0ddac139deb3c5be9709f66ea4903b17ed9c2a9b727ae0aa82082da0a` |

**Added: everything committed on this branch that the round reads.**

| input | SHA-256 |
| --- | --- |
| `scripts/election_residuals/eve_consensus.py` | `781bf689cc024f9a874e0b6eddd5ff34795377e99da1ca35dbc05d90040b7d20` |
| `diagnostics/calibration_2030/t1_results.json` | `d5f402541f2067f60382c537c41f9482285fef727f96ada4d12a53f4dea0feaa` |
| `diagnostics/calibration_2030/t3_results.json` | `85f4e4109fa2475598e825bf077790adf76d5f9d42c7f3c8fd29a40be434fc8d` |
| `data/processed/calibration_2030/amendments/001-t3-flag-interim-beta-half.json` | `a29f0d6e293aaf5ba10f0d78f47f46e911517ec1681fe2e2516de0ed87814f88` |
| `data/processed/election_residuals/amendments/index.json` | `10aba2f176a2a6747ad0fad4bf2716fe52bfc99772a2980228105cf8b514fb5b` |
| `data/processed/poll_aggregate/v0_2/individual_polls.csv` | `cc86b8ea42caeca43d2e9bbde70612e7e93002b8f1593b1f71c9844a1ab1fda7` |

The last row is the polls file that production OpinionState reads, through
`scripts/simulator/model_inputs.py`. Arms `D0`, `C0`, `F1` and `FB` read it.

**Referenced, not re-hashed by the guard.** Already pinned elsewhere:

- the parent record `data/processed/calibration_2030/preregistration.json`
  (SHA-256 `e3f59c4b824e1aefdc2ea58a410b6bb1b85abb8aca74d367eeeb9029f792b0fd`);
- residual amendment records 001 (`f17cfbde…9e27`) and 002 (`988e4b06…981e`),
  hashed in the parent record;
- the frozen metric module
  `diagnostics/election_noise_v2/control_baseline/harness/metrics.py`
  (`ce872a25cf2d56d6f3db18259834bb279cf828ad4cba1865e88b79999f0c3810`).

Polls published after 2026-11-18 are not used.

### 1.3 Disclosure

- **Everything here is retrospective.**
- **Post hoc proposals.** The hierarchical noise model and the size scaling
  (T3-rev) were proposed after T3's results were seen.
- **Party-size pattern.** It was computed on the 2002–2026 corrected
  election-eve residuals. RMS residual by consensus share:

  | share | RMS (pp) |
  | --- | --- |
  | small (< 7 %) | 0.66 |
  | 7–15 % | 1.28 |
  | ≥ 15 % | 1.87 |

  The maximum-likelihood exponent is γ ≈ 0.76.
- **2026 was seen.** This covers its polls, its result and every T1/T3 score
  on it.

Under these conditions, no result here is independent validation. The only
independent evidence will be prospective (T5).

### 1.4 Predeclared fallback

- If `CAND` fails T4, `FB` is run through the same T4 gates. `FB` is:
  - `F1`;
  - aggregate v0.3;
  - the interim noise law (Gaussian, β = 0.5, γ = 0);
  - production OpinionState and dynamics, with the 112-day cap.
- If `FB` passes, `FB` is proposed.
- If `FB` also fails, production stays 1.2.0-rc1. Only `F1` may be proposed,
  on correctness grounds.
- **No other candidate may be constructed after T4 scores exist.**

**Coincidence.** If T2 selects `D0` and T3-rev selects `N0`, then `CAND` and
`FB` are the same model. It is run once, and the result is recorded for both.

## 2. Aggregate v0.3 (the aggregate of every candidate arm)

v0.3 is defined by the T1 harness `diagnostics/calibration_2030/t1_aggregate.py`
at commit `b27dce4` (the run recorded in `t1_results.json`). It has two parts:

- the filter `ArmFilter` with arm `SH`;
- the readings of `build_rows(..., weights_mode="rolling")`. Each day `T`
  reports `consensus_reading_at(state, T, consensus_weights_at(polls, T, 365))`.

Hyperparameters are not refitted. Each segment uses `SH`'s fit recorded in
`t1_results.json` under `arms.SH.fits[i].theta`:

| fit date | process | house_prior | house_drift | design_effect | election_n | s_shock | h_reset |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2010-09-26 | 1.144e-5 | 8.697e-4 | 1.029e-6 | 1.051 | 4325 | 2.839e-4 | 8.035e-6 |
| 2014-09-21 | 1.254e-5 | 4.633e-4 | 5.970e-7 | 1.004 | 1238 | 9.366e-4 | 1.336e-4 |
| 2018-09-16 | 1.265e-5 | 5.933e-4 | 9.526e-7 | 0.953 | 898.0 | 1.378e-3 | 2.564e-4 |
| 2022-09-18 | 1.353e-5 | 5.802e-4 | 1.015e-6 | 0.876 | 1071 | 1.641e-3 | 2.383e-4 |
| 2026-09-19 | 1.285e-5 | 5.163e-4 | 9.276e-7 | 0.860 | 984.8 | 1.454e-3 | 2.309e-4 |

The exact values are the JSON floats in the file, copied into the record
`data/processed/calibration_2030/final_round.json`.

**Range.** Rows start at 2010-09-26, the first fit date, as in v0.2.

**Files.** Arms that read the aggregate as a file (`D0`, `FB`, and the T4
engine) get the rows in production's timeseries schema, written as
`scripts/poll_aggregate/build.py::write_timeseries` writes them, with 2-decimal
rounding. The file's SHA-256 is recorded with every run. The arms that use the
filter state directly (`Ua`, `Ub1`, `Ub2`, `Uc`) use the unrounded state.

**Self-check.** Before anything is scored, the regenerated v0.3 must
reproduce the `SH` M readings that `t1_results.json` reports with rolling
weights (`m_reading_2026_rolling_weights`).

## 3. T2: state uncertainty and dynamics (poll-prediction test)

**Question.** Do the filter's own posterior and a longer-horizon dynamics law
predict future poll averages better than OpinionState and the capped
bootstrap?

### 3.1 Aggregate, origins, horizons

- **Aggregate.** v0.3 for every arm, so the aggregate is not the variable.
- **Origins.** Every 14th day from 2010-10-03 through 2026-11-18 (421 origins,
  the last on 2026-11-08), on days where the v0.3 reading exists. Each origin
  uses only data available that day.
- **Horizons.** h ∈ {7, 28, 91, 182, 365, 730, 1095, 1460} days.
- **Target existence.** A target exists only if its window closes on or before
  2026-11-18, that is, D + 7 ≤ 2026-11-18, with D = origin + h.
- **2010.** v0.3 rows start 2010-09-26, so T2 origins start at 2010-10-03.
  The 2006–2010 cycle is not in T2.

**First scored origin.** At the first grid dates, not every quantity can be
computed:

- OpinionState raises unless at least 10 residual polls match a v0.3 row
  within 3 days.
- The production bootstrap raises below 30 transitions even at its 7-day
  fallback.
- Ω needs at least 30 centres (§3.3).
- θ needs at least 3 variogram lags with pairs (§3.5).

The first scored origin is the first grid date at which every T2 arm, Ω and
θ are all defined. It is the same for all arms and is recorded. Earlier grid
dates are unscored. No pre-2010 substitute rows (§6.3) are used in T2.

### 3.2 Target

At D = origin + h:

1. Take every poll the aggregate uses whose fieldwork midpoint
   (`Observation.obs_date`) is in [D − 7, D + 7]. These are the cleaned
   SwedishPolls observations of `scripts/poll_aggregate/data.py::load_polls`.
2. Compute the effective-sample-weighted mean of their raw eight named-party
   shares (pp). The weight is `Observation.sample_size`, the
   overlap-discounted or imputed effective sample.
3. At least 3 polls are required. Otherwise there is no target.

On the pinned inputs this yields 2,020 targets, 2,003 of them dated before
2026-09-13 (a count of targets only; no score was computed).

### 3.3 Common poll-average term

A zero-mean Gaussian with covariance Ω_origin is added **identically** to
every arm's predictive draws, so the arms differ only in state and dynamics.

- **Ω_origin** is the sample covariance (8 × 8, pp) of the difference
  between two readings at each window centre c:
  - the target-style window poll average at c, under the §3.2 rules,
    including the 3-poll minimum;
  - minus the v0.3 reading at c, taken from c's causal row.
- **Centres.** An expanding window over a **daily** grid: every calendar day
  c from 2010-09-26 (the first v0.3 row) with c + 7 ≤ origin that meets the
  3-poll minimum. Only data known at the origin are used, that is, polls
  published on or before it.
  - Daily centres have overlapping windows, so neighbouring differences are
    correlated. Ω is used only as a covariance estimate, never as a count of
    independent cases.
  - A daily grid is used so that Ω's minimum does not decide the first scored
    origin. θ and `D0` decide it.
- **Minimum.** At least 30 such centres are required. An origin with fewer is
  unscored (§3.1). There is no fallback.

### 3.4 Arms

Each arm produces N = 4,000 draws of the eight named-party shares (pp) at D.

| arm | state uncertainty at origin | dynamics origin → D |
| --- | --- | --- |
| `D0` (control) | OpinionState v1.1 covariance as production computes it (ALR, shrunk residual covariance / n_eff), with the v0.3 reading as the mean | production symmetric CLR bootstrap at evaluation horizon min(h, 112), with production's fallback rule (28, 14, 7 days when fewer than 30 transitions) |
| `Ua` | the covariance of the consensus reading at the origin, `consensus_reading_at(state, origin, w)`, times k² | the filter's own forecast (below), plus election-day shocks |
| `Ub1` | as `Ua` | OU / damped random walk anchored at the current reading: mean unchanged; variance σ²(1 − e^{−θ·days})/θ in the shape metric; plus the same election-day shocks |
| `Ub2` | as `Ua` | OU anchored at the last certified election result A known at the origin: mean A + (x − A)e^{−θ·days}; the same variance and the same θ as `Ub1` |
| `Uc` | as `Ua` | symmetric CLR bootstrap of v0.3 transitions at horizon h exactly, with no cap |

**Common definitions** (`Ua`, `Ub1`, `Ub2`, `Uc`):

- `state` is the v0.3 filter state on the origin, run over the observations
  known that day with the segment's `SH` parameters.
- `w` is the rolling weights at the origin, `consensus_weights_at(polls,
  origin, 365)`. They are held fixed to D.
- The mean is the v0.3 reading at the origin, except for `Ub2`.

**`D0` in detail.** `D0` is production's code path, run on the v0.3 rows:

- `estimate_opinion(as_of=origin, timeseries_file=<v0.3 file>,
  polls_file=<production polls file>)`;
- `build_all_historical_transitions` on the same v0.3 rows, then
  `filter_transitions_as_of`;
- `sample_shared_symmetric_dynamics`;
- CLR addition as in `national_engine.py`. The eight named parties are taken
  from the result.

The v0.3 file therefore sets the mean, the OpinionState residual covariance
(polls against v0.3 rows) and the transitions. This is how the replay harness
substitutes an aggregate (`scripts/poll_aggregate/replay.py::build_data_root`),
and it is consistent with T4 and with `FB`.

**`Ua` dynamics.** The dynamics covariance is what
`consensus_reading_at(state, D, w)` adds to the covariance at the origin:

- the process variance × days × `multinomial_shape(s)`, at the latent mean
  `s` on the origin;
- plus house drift: house_drift × days × Σ_h (w_h / Σw)² × reference_shape.

**Election-day shocks.** For every election day E in the open interval
(origin, D), the reading's covariance gains:

- `s_shock × multinomial_shape(s)`;
- `h_reset × Σ_h (w_h / Σw)² × reference_shape`.

The parameters and shapes:

- `s_shock` and `h_reset` are the origin segment's;
- `s` is the latent mean at the origin;
- `reference_shape` is the origin's.

An election on the target date D itself gets no shock. Its jump would
otherwise be counted twice with the noise law, for example in T5's
election-day target.

These are the effects of T1's `SH` injection on the latent support and on the
weighted houses. `consensus_reading_at` does not include them, so the harness
adds them, as `ArmFilter` does. Election days are calendar facts, known in
advance.

**`Ub1` and `Ub2` dynamics.**

- **Variance.** (σ²/θ)(1 − e^{−θ·days}) × the full `multinomial_shape(s)`
  matrix, with σ² the origin segment's process variance. The `Ub` arms carry
  no house drift.
- **Ub2 anchor.** A is the eight named shares of the last election whose
  `available` date is on or before the origin.
- **"Vague prior" anchor.** It reduces to `Ub1` in the mean, so it is not a
  separate arm.

**`Uc` fallback.** If there are fewer than 30 eligible transitions at h
(those with `end_date` ≤ origin), take the largest h′ < h on the horizon grid
with at least 30, and scale the deltas by √(h/h′).

**Simplex map.** One rule applies to every arm and to `CAND`'s base
composition (§5):

1. Complete the draw with REST = 100 − Σ of the eight parties.
2. Clip each of the nine categories at 0.01 pp.
3. Renormalize to 100.

The map is applied as follows:

- **`Ua`, `Ub1`, `Ub2` and `D0`.** The map is applied to the state + dynamics
  draw.
- **`Uc`.** The state draw is mapped, then converted to CLR. The delta is
  added, and the result is converted back to a composition.
- **Ω.** The poll-average draw (§3.3) is added only after the map, and is not
  clipped, because the targets are poll averages in pp.

### 3.5 Inflation k and decay θ

**k** (`Ua`, `Ub1`, `Ub2`, `Uc`, each for itself):

- **Equation.** At each origin, k solves

  mean over eligible past targets of (1/8) · Σ_p e_p² / (k² · S_pp + D_pp + Ω_pp) = 1.

  The root is found on k ∈ [0.5, 3] by 1-D root-finding. If there is no root,
  k is the nearer bound.
- **Terms**, for each past target (origin′, h):
  - e_p = the target minus the mean of the arm's k = 1 predictive draws;
  - S_pp = the state variance (pp²) at origin′;
  - Ω_pp = the poll-average variance at origin′;
  - D_pp = max(0, Var_p(k = 1 predictive draws) − S_pp − Ω_pp). This works
    for every arm, `Uc` included.
- **Eligible targets.** Past (origin′, h) targets with h ∈ {7, 28},
  D′ + 7 ≤ origin, and every poll in the window published on or before the
  origin. The unit is a target, which carries 8 party errors.
- k = 1 until 20 eligible targets exist.
- k is recorded over time.

**θ** (`Ub1`, `Ub2`):

- A single scalar for all parties, fitted at each origin.
- **Empirical variogram.** For each lag h_v ∈ {28, 91, 182, 365, 730}, γ̂(h_v)
  is computed from the v0.3 reading x:
  - for each pair and party, the uncentred squared increment
    (x_{t+h_v,p} − x_{t,p})², divided by `multinomial_shape(x_t)[p, p]`;
  - averaged over parties and pairs.

  Pairs are expanding: all pairs whose later date is on or before the origin.
  Lags with no pairs are dropped. At least 3 lags are required; an origin with
  fewer is unscored (§3.1).
- **Model.** (σ²/θ)(1 − e^{−θ·h_v}), with σ² fixed to the filter process
  variance (as `Ua`).
- **Fit.** Least squares on the log scale, minimizing
  Σ_{h_v} (log γ̂(h_v) − log model(h_v))².
- **Bounds.** θ ∈ [1e-5, 1e-1] per day.
- θ is recorded over time.

Because σ² is fixed and θ ≥ 1e-5 > 0, `Ub1`'s dynamics variance never
exceeds `Ua`'s random-walk variance at any horizon. By design, `Ub` is the
saturating alternative.

### 3.6 Draws and seeds

The derivation is analogous to T3's. Arms share standard-normal draws where
their laws allow.

- **Seed per (origin, h).** int(sha256("calibration-2030:T2:<origin
  ISO>:<h>")[:8], 16).
- **Draw order.** One `numpy.random.default_rng(seed)` draws Z_state, then
  Z_dyn, then Z_poll, each 4,000 × 8, and then `Uc`'s bootstrap indices and
  signs.
- **Shared draws.** Z_state is shared by every U arm, Z_dyn by `Ua`, `Ub1`
  and `Ub2`, and Z_poll by all five arms.
- **`D0`.** Production samples OpinionState with Python's `random.Random`, so
  it cannot share Z_state. It is run with production's own sub-seed functions,
  using the case seed as `base_seed`, and takes Z_poll from the shared
  generator.
- **Quantiles.** Central intervals use `numpy.quantile` (linear) at 0.05,
  0.25, 0.75 and 0.95, with inclusive bounds as in
  `calculate_interval_metrics`.
- **Block bootstrap.** 10,000 resamples of election cycles, with a seed
  derived by the same scheme.

### 3.7 Metrics

- **Primary.** CRPS per party (pp) from the draws
  (`scripts/pollofpolls/backtest_metrics.py::calculate_crps`), averaged over
  the 8 parties, for each (origin, h).
- **Also.** Central 50 % and 90 % coverage per party.
- **Horizon groups.**

  | group | horizons (days) |
  | --- | --- |
  | short | 7, 28 |
  | medium | 91, 182, 365 |
  | long | 730, 1095, 1460 |

- **Group values.** A group's value is the mean over its (origin, h) cells.
  Coverage is the share of party intervals that contain the target.
- **Decision targets.** Targets dated before 2026-09-13. Later targets are
  reported only.

### 3.8 Gates (each U arm against `D0`, on decision targets)

| gate | condition |
| --- | --- |
| T2-G1 long | mean CRPS over the long group < `D0`'s |
| T2-G2 short | mean CRPS over the short group ≤ 1.05 × `D0`'s |
| T2-G3 calibration | 90 % coverage in [0.80, 0.97] for the medium group **and** for the long group |

### 3.9 Selection

1. Among passing arms, take the one with the lowest long-group CRPS.
2. If a simpler passing arm is within 2 % of it (its long-group CRPS
   ≤ 1.02 × the best), choose the simpler. Simplicity order: `Ua` < `Ub1` <
   `Ub2` < `Uc`.
3. If no arm passes, `CAND` keeps production state and dynamics (`D0`).

### 3.10 Reported, no decision weight

- **Effective sample caveat.** There are about four independent four-year
  windows. A block bootstrap by election cycle of each U arm's long-group
  CRPS difference against `D0` is reported only.
- k and θ over time, and the first scored origin (§3.1).
- Targets dated on or after 2026-09-13.

## 4. T3-rev: the noise law (election-eve residual test)

**Question.** Does a hierarchical law with size scaling beat the interim law
on the corrected residuals?

The design is T3's (parent §3.2), with these changes.

### 4.1 Data and design

- **Residuals.** The corrected residuals of amendments 001 and 002: nine
  categories, pp, zero-sum cleaned. They are `corrected_residuals_pp` and
  `corrected_consensus_pct` in `t3_results.json`, recomputed and checked equal.
- **Cases.**
  - Rolling targets 2014, 2018 and 2022 (K = 3, 4, 5): decision.
  - Leave-one-out over 2002–2022 (six folds, K = 5): check.
  - 2026 with pool 2002–2022 (K = 6): reported.
- **Forecast.** The target's corrected consensus plus 20,000 draws of r,
  through `apply_batch_simplex_transfer` with `eps` = 0.01 pp.
- **Seed.** int(sha256("calibration-2030:T3rev:<year>:<design>")[:8], 16),
  with `<design>` ∈ {`rolling`, `loo`, `heldout`}.
- **Draw order.** One `numpy.random.default_rng(seed)` draws Z (20,000 × 9).
  Every arm of a case shares Z.
- **Monte Carlo sensitivity.** Four more seeds, made by appending `:1` to
  `:4` to the seed string. A gate whose outcome differs across the five seeds
  is Monte-Carlo-fragile and is treated as **failed**.
- **Metrics.** The frozen metric module as in T3.

### 4.2 Arms

| arm | law |
| --- | --- |
| `N0` (control) | the interim law (calibration amendment 001): `r = 0.5 · r̄_P + ε`, `ε ~ N(0, Σ̃)`, `Σ̃` = `fit_election_noise_b` on the centred pool; γ = 0 |
| `NH0` | hierarchical normal–normal law, γ = 0 |
| `NH0.5` | hierarchical normal–normal law, γ = 0.5 |
| `NH1` | hierarchical normal–normal law, γ = 1 |

### 4.3 Size scaling

The scaling covers all nine categories. REST is scaled like the parties, with
its own consensus share.

For each pool election e:

1. Let c_e be the consensus shares of e's own election (pp), floored at
   1.0 pp.
2. Standardize elementwise: u_e = r_e / c_e^γ.
3. **Unreported categories.** A category that no contributing poll of e's
   consensus reported is set to u = 0. This applies in every NH arm
   (γ = 0, 0.5 and 1), so the γ comparison is clean.
4. Project u_e to zero-sum by subtracting its mean.
5. Fit in u-space (§4.4).

At the target T:

1. Draw ũ (zero-sum).
2. Map r = ũ ⊙ c_T^γ, with c_T the target's corrected consensus, floored at
   1.0 pp.
3. Restore zero-sum by subtracting (Σr) · c_T^γ / Σ c_T^γ.

γ = 0 is the unscaled law.

**Where the unreported-category rule applies.** On the pinned data it applies
to one cell only, SD in 2002:

- SD's corrected consensus in 2002 is 0.00 pp, because no contributing poll
  reported the party. Its residual is +1.44 pp.
- With only the 1.0 pp floor, u_SD,2002 would be 1.44 at any γ. At γ = 1,
  every other |u| is at most 0.50, so that one cell would dominate every pool
  containing 2002.

These values were computed from the published T3 residuals. No T3-rev score
was computed.

**Disclosure.** `N0` is unchanged: it is the tested interim law, and keeps the
SD-2002 cell. So `NH0` differs from `N0` in this one cell as well as in the
hierarchical law.

### 4.4 Hierarchical law in u-space

- ū = the pool mean, zero-sum cleaned.
- Σ̂ = `fit_election_noise_b` on the centred pool (u_e − ū). It requires
  K ≥ 2, and its rows are zero-sum, as its Ledoit–Wolf target τ²P₉ assumes.
- κ̂ = max(0, tr(ū ūᵀ)/tr(Σ̂) − 1/K), where K is the pool size.
- β = Kκ̂/(Kκ̂ + 1).
- **Predictive.** ũ ~ N(β ū, (1 + β/K) Σ̂), drawn as
  β ū + √(1 + β/K) · Z Lᵀ, with L the symmetric factor of Σ̂
  (`symmetric_factor`).

### 4.5 Metrics

- **Primary.** The nine-category energy score, `d1_joint_vote_energy_score`.
- **Secondary.** `d2`: CRPS and 50/90 % coverage.
- **Additional: Brier_threshold at election eve.** The mean over the eight
  parties of (P(share ≥ 4 %) − 1[actual ≥ 4 %])², from the transferred
  draws. This is the protocol v4 §6 definition, as in
  `scripts/poll_aggregate/replay.py`.

### 4.6 Gates (each NH arm against `N0`)

| gate | condition |
| --- | --- |
| T3r-G1 | rolling mean ES < `N0`'s |
| T3r-G2 | each rolling target's ES ≤ 1.10 × `N0`'s |
| T3r-G3 | leave-one-out mean ES ≤ 1.02 × `N0`'s |
| T3r-G4 | pooled rolling Brier_threshold ≤ `N0`'s + 0.005 |

### 4.7 Selection and final refit

1. Among passing arms, take the one with the lowest rolling mean ES.
2. If a simpler passing arm is within 1 % of it, choose the simpler. Order:
   `NH0` < `NH0.5` < `NH1`.
3. If no arm passes, keep `N0`.
4. **Final refit.** The selected law is refitted by its frozen procedure on
   all seven corrected residuals (2002–2026).

**Reported.** κ̂ and β for every case, and every arm's 2026 scores.

## 5. The unified model, `CAND`

`CAND` is built from these parts:

| layer | definition |
| --- | --- |
| aggregate | v0.3 (§2) |
| state uncertainty and dynamics | the T2 selection. If a U arm wins, the filter posterior × k replaces OpinionState, and that arm's dynamics replace the bootstrap. If `D0` is retained, production OpinionState and dynamics run on v0.3. The base composition is the state + dynamics draw after the §3.4 simplex map. |
| election noise | the T3-rev selection, refitted for each T4 target on the elections strictly before it |
| geography, mandate allocation, simplex transfer | production, unchanged |

**Order in T4.** The noise is applied to the mapped base composition by
production's simplex transfer.

**Size scaling in T4.** At a T4 forecast origin, the target election's eve
consensus does not yet exist. c_T is each draw's base composition (state +
dynamics after the §3.4 map), REST included, floored at 1.0 pp.

**Values carried into T4.** k and θ are taken at the T4 origin by the §3.5
procedures, from data known then.

## 6. T4: end-to-end replay

The gates are frozen in parent §4.4 (G1–G7, the 2026 flag) and parent §4.5
(decisions). They are not restated or changed here. `CAND`, and `FB` if it is
run, must pass all of them. Pooled values are unweighted means over the cases
of a set (parent §4.3).

### 6.1 Arms

| arm | definition |
| --- | --- |
| `C0` | production 1.2.0-rc1 at `6c4271c` |
| `F1` | `C0` with the noise pool built from the corrected residuals (amendments 001 and 002). Reported against `C0`, not gated (parent §4.1). |
| `CAND` | §5 |
| `FB` | §1.4, run only if `CAND` fails |

### 6.2 Cases, draws, seed

- **Set D and set L.** As frozen: elections 2014, 2018 and 2022. Set D at 112,
  84, 56, 28, 14 and 7 days; set L at 365, 730 and 1,095 days.
- **2026.** Both horizon sets, reported, with the parent's 2026 flag.
- **2010.** Reported only (§6.3). It enters no gate.
- **Draws.** 20,000 draws, seed 20260929, the same for every arm.
- **Noise pools.** Each target's pool contains only earlier elections.
- **G7's calibration-improvement clause** applies if `CAND`'s dynamics differ
  from `D0`.

**Sub-seeds.**

- **Unchanged layers** use production's derivation functions.
- **New layers** use
  `int(sha256(f"{seed}:{origin_iso}:{horizon_days}:calibration-2030:{layer}")[:8], 16)`,
  with `layer` ∈ {`state`, `dynamics`, `bootstrap`, `ou`}.
- There is one `numpy.random.default_rng` per layer.

### 6.3 2010, reported only, with three declared substitutions

1. **v0.3 rows before 2010-09-26.** They come from the `SH` filter run in
   causal mode with `HYPERPARAMETER_START` plus the `SH` new-parameter starts
   (`s_shock` = 3.162e-4, `h_reset` = 1e-5), with rolling weights.
   - There is no non-anchor election before 2010, so the two new parameters
     have no effect on these rows.
   - Rows run from 2007-01-01 to 2010-09-25. Every 2010 case's origin is on
     or after 2007-09-20 (the 1,095-day horizon), so the weights at every
     origin are informed by polls known in the 365 days before it.
2. **v0.2 rows before 2010-09-26, for `C0` and `F1`.** v0.2 also has no
   rows before that date.
   - **How.** These rows come from the v0.2 filter (no `SH` injection) run in
     causal mode with `HYPERPARAMETER_START`, from 2007-01-01 to 2010-09-25.
   - **Weights.** These substitute rows use **daily rolling weights**, the
     same rule as v0.3's `W`. v0.2's frozen-per-segment weights cannot be
     formed: there is no segment fit date before 2010-09-26, and on
     2006-09-24 no poll is yet known. This is a declared difference from
     production's v0.2, confined to these reported-only rows.
   - `FB` uses the v0.3 substitution of item 1.
3. **The 2010 noise pool has K = 2** (2002 and 2006). Ledoit–Wolf is
   degenerate there, which is acceptable for reporting.

All 2010 cases are reported only and enter no gate.

### 6.4 Runtime

- **Candidate path.** The engine needs a candidate path that plugs in state,
  dynamics and noise. It is built as new modules under
  `scripts/calibration_2030/` or `diagnostics/calibration_2030/`, without
  modifying hashed production files (`engine.py`, `national_engine.py`,
  `consensus.py`, `state.py`, …).
- **Self-check.** Configured as `C0`, the candidate path must reproduce
  `C0`'s bytes, like T1's self-check.

## 7. Order

1. Freeze this document.
2. Regenerate v0.3 and run its self-check (§2).
3. Run T2 and T3-rev. They are independent.
4. Build `CAND` (§5), and the candidate path with its `C0` self-check.
5. Run T4. If `CAND` fails, run `FB` (§1.4).
6. A release proposal on a pass, through its own amendment (parent §4.5).

## 8. Executability notes (checked against the code at `0188f75`)

**`consensus_reading_at`** (`scripts/poll_aggregate/model.py`) returns the
reading's mean and 8 × 8 covariance. That covariance includes:

- the process variance × gap × the shape at the latent mean;
- each weighted house's drift since it was last read.

So `Ua`'s state (at the origin) and random-walk dynamics (at D minus at the
origin) are both available. Election shocks are not included, and are added
as in `ArmFilter`.

**OpinionState** (`scripts/pollofpolls/state.py`):

- It reads a timeseries CSV, so v0.3 must be written as a file in that schema.
- It samples the ALR Gaussian with Python's `random.Random` and inverts it to
  nine parts.

**Metrics.** `calculate_crps` and `calculate_interval_metrics` exist. With
4,000 draws they are cheap.

**`fit_election_noise_b`** takes any K × 9 pool with K ≥ 2. Centred u-space
rows are zero-sum, so it applies as written.

**T2 cost.** 421 origins, 2,020 targets and 5 arms, at 4,000 × 8 draws each,
come to about 3 × 10⁸ values. The dominant costs are:

- 421 `estimate_opinion` calls, each with a pure-Python poll × row scan;
- regenerating v0.3 (the T1 harness's `build_rows`).

Feasible in well under an hour on one core.

## 9. Resolved items

1. `v0_2/individual_polls.csv` is pinned (§1.2): resolved by the lead, 2026-11-18.
2. If `CAND` ≡ `FB`, the model is run once and recorded for both (§1.4): resolved by the lead, 2026-11-18.
3. The first scored origin is common to all arms; no pre-2010 rows are used in T2 (§3.1): resolved by the lead, 2026-11-18.
4. Ω uses an expanding window and needs at least 30 centres; there is no fallback (§3.3): resolved by the lead, 2026-11-18.
5. `D0`: the v0.3 file sets the mean, the residual covariance and the transitions (§3.4): resolved by the lead, 2026-11-18.
6. Election shocks use origin parameters and shapes, over the open interval (origin, D) (§3.4): resolved by the lead, 2026-11-18.
7. `Ub` uses the full shape matrix and is saturating by design (§3.4, §3.5): resolved by the lead, 2026-11-18.
8. One simplex map serves every arm and `CAND`'s base; Ω is added after it (§3.4, §5): resolved by the lead, 2026-11-18.
9. k solves a variance-matching equation over causally eligible targets (§3.5): resolved by the lead, 2026-11-18.
10. θ uses an uncentred, shape-normalized variogram with at least 3 lags (§3.5): resolved by the lead, 2026-11-18.
11. Seeds, draw order, quantiles and a 10,000-resample bootstrap are adopted as proposed (§3.6): resolved by the lead, 2026-11-18.
12. REST is scaled; unreported categories (SD 2002) are set to u = 0 in NH arms (§4.3): resolved by the lead, 2026-11-18.
13. In T4, c_T is each draw's mapped base composition, floored at 1.0 pp (§5): resolved by the lead, 2026-11-18.
14. T4 sub-seeds: production derivation for unchanged layers, new tokens per layer (§6.2): resolved by the lead, 2026-11-18.
15. 2010 cases: a parallel v0.2 substitution for `C0` and `F1` (§6.3): resolved by the lead, 2026-11-18.
16. Ω centres on a daily grid, so its 30-centre minimum does not push the first scored origin to 2012 (§3.3): resolved by the lead, 2026-11-18.
17. The 2010 substitute rows (v0.2 and v0.3) use daily rolling weights from 2007-01-01, because frozen v0.2 weights cannot be formed before the first fit (§6.3): resolved by the lead, 2026-11-18.

## 10. Freeze procedure

The scheme is the parent's §7:

- **`BODY SHA-256`** covers every byte before the last freeze-block marker.
- **The freeze record** is `data/processed/calibration_2030/final_round.json`,
  with its `.sha256` companion created at the freeze.
- **The freeze commit** is reported with the freeze.

A verification snippet must split the marker string, as the parent's does.

<!-- FREEZE-BLOCK-START -->

```
PREREGISTRATION STATUS: FROZEN

FREEZE TIMESTAMP (UTC): 2026-11-18T18:54:43Z
FREEZE BASE COMMIT:     0188f75b9dfa9315343ba5aec2b5444774e1ddf7
FREEZE COMMIT:          the commit that adds this block (reported with the
                        freeze; it cannot be embedded in the file it hashes)
BODY SHA-256:           5d070b4b21c823b9db40253efd46de5288ec6d8259f0a1c7a34c8f2f9485c18b
DECIDED BY:             user (plan approval of 2026-11-18); open items resolved by the lead
RECORD:                 data/processed/calibration_2030/final_round.json
```

BODY SHA-256 covers every byte of this file strictly before the last
occurrence of the freeze-block marker. Verify with:

```bash
python3 - <<'PY'
import hashlib, pathlib
b = pathlib.Path("docs/calibration_2030_final_round_preregistration.md").read_bytes()
marker = b"<!-- FREEZE-BLOCK-" + b"START -->"
print(hashlib.sha256(b.rsplit(marker, 1)[0]).hexdigest())
PY
```
