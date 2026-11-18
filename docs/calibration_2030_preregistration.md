# Calibration programme for the 2030 cycle: preregistration (T1, T3, T4 gates)

**Status: FROZEN.** The user approved the freeze on 2026-11-18. The document
was drafted the same day on branch `prereg/calibration-2030-drafts` from
`origin/main` `6c4271c`.

- **Freeze record.** The freeze timestamp, body hash and base commit are in
  the freeze block at the end of this file and in the freeze record (§7).
- **Nothing scored before the freeze.** No challenger was implemented or
  scored before it.
- **Changes after the freeze** need an append-only amendment.

This programme is not a protocol change of the 2026 prospective benchmark.
That benchmark's dates, selection, weighting and scoring are untouched.

## 0. Why, and what is already known

The production model (1.2.0-rc1) is too confident. In replay v4 its 90 %
intervals covered 0.784 of outcomes
(`docs/poll_aggregate_replay_v4_results.md`). The review of 2026-11-18 found
four contributing causes:

1. **The noise residuals leak election-day polls and silently drop polls.**
   They measure the result against a consensus that includes exit polls and
   other election-day polls, and that drops polls missing a sample size or a
   fieldwork start. Two amendments fix this, and both are in effect:
   - `docs/election_noise_residual_amendment_001.md`, finalized
     2026-11-18T16:41:53Z;
   - `docs/election_noise_residual_amendment_002.md`, finalized
     2026-11-18T16:44:12Z.

   They are implemented by `build_eve_polling_consensus` in
   `scripts/election_residuals/eve_consensus.py`. On the production pool
   (2002–2022) the corrected residuals raise the noise law's total variance
   from 7.29 to 8.39 pp².
2. **The noise law discards systematic poll bias.** Its draws are centred,
   although S was underestimated in 6 of 6 elections (2002–2022), and V and
   MP were overestimated in 6 of 6.
3. **The poll aggregate absorbs post-election shifts slowly.** After 2026 the
   first five M polls had standardized innovations of +2.8 to +3.9.
4. **The uncertainty layers above the aggregate are too narrow.** OpinionState
   is narrower than the aggregate's own posterior, and dynamics are capped at
   112 days. These belong to T2 and are not preregistered here.

**Disclosure of evidence already seen.** Every test here is retrospective.
The following was computed before this preregistration was drafted and cannot
be unseen:

- **T1.** With the 2026 segment's hyperparameters held fixed, the author ran a
  grid. An election shock to latent support (1e-3 to 3e-3 × shape) plus a
  house-effect reset (3e-4 × shape) raised the summed log-likelihood of polls
  before 2026-09-13 by about 58 nats. Two more results come from the same
  session:
  - **Earlier elections:** in the first 120 days after the 2010, 2014, 2018
    and 2022 elections, the mean standardized innovation per party stayed
    within ±0.72.
  - **2026:** the mean for M was +2.28.

  These were in-sample diagnostics. The T1 test differs: hyperparameters are
  refitted causally and scored out of sample.
- **T3.**
  - The residuals of all seven elections are known under both rules
    (`diagnostics/election_noise_residual_eve_rule/results.json`).
  - So are the pool mean biases.
  - The adopted law `pp_lw_gaussian` was itself selected on the same rolling
    targets (2014, 2018, 2022) by `docs/election_noise_v2_preregistration.md`.
- **T4.** Replay v4's scores for 2018, 2022 and 2026 at 7–112 days are known.

Under these conditions, no result here is independent validation. The only
independent evidence will be prospective (T5).

## 1. Rules that bind every tier

1. **Pinned control.** The control `C0` is production `MODEL_VERSION`
   1.2.0-rc1 at commit `6c4271c`:
   - aggregate `SwedishPollsAggregate-v0.2`;
   - OpinionState v1.1;
   - Dynamics v2 with the 112-day cap;
   - noise law `pp_lw_gaussian` on residuals built by the current rule.
2. **Pinned data.** Every tier reads exactly these bytes:

   | input | SHA-256 |
   | --- | --- |
   | `data/processed/pollofpolls/swedishpolls_individual_polls.csv` | `18e84e56eff64a5133eb4eeca9767893a7f74b15724552ef573fd9314e8d2828` |
   | `data/processed/elections/riksdag_election_results.csv` | `3a99ffb1b0fbef60d61712764a2136982ecf01c4879a43edc6d9a38ddb4ba0fc` |
   | `data/raw/elections/val2026/official_result_manifest.json` | `6bdcbd284f602d9c9b1b9605ab3d52b5700528b54232565f3242d01d107f788d` |
   | `data/processed/poll_aggregate/v0_2/swedishpolls_aggregate_timeseries.csv` | `e04bca4a6f90340993f2ca79bfe5d2f17d3ddf8a9e36d885a278048986943761` |
   | `data/processed/poll_aggregate/v0_2/swedishpolls_aggregate_metadata.json` | `5c409da0ddac139deb3c5be9709f66ea4903b17ed9c2a9b727ae0aa82082da0a` |

   The corrected residuals come from
   `scripts/election_residuals/eve_consensus.py` (SHA-256
   `781bf689cc024f9a874e0b6eddd5ff34795377e99da1ca35dbc05d90040b7d20`), as finalized with amendment 002.

   Polls published after 2026-11-18 are not used. A scoring run refuses to
   start if any hash differs or any tracked file has uncommitted changes
   (`git status --porcelain --untracked-files=no` is not empty). It records
   the commit it ran at.
3. **Causality.**
   - Every forecast origin uses only polls published by that origin, and only
     election results available by then.
   - Every fitted quantity is fitted only on data known at its fit date.
4. **Lock procedures, not numbers.** A tier may estimate parameters only
   through the mechanical procedure written here: maximum likelihood, the
   Ledoit–Wolf fit, or a pool mean. No value is chosen by looking at scores.
   This document fixes every option set in advance.
5. **2026 is held out wherever it motivated the change.**
   - T1 and T3 decide on pre-2026 data alone.
   - 2026 is reported for every arm as "held out, but already seen" (§0). It
     cannot change a selection; it can only raise a flag that the user must
     resolve.
   - T3's final refit then adds 2026 to the pool (decision of 2026-11-18).
6. **Common random numbers.** Arms are compared on identical random draws
   wherever their laws allow, so a difference between arms is not simulation
   noise.
7. **Report everything.** Every arm, metric and gate outcome is published,
   including failures and flags. Nothing is rerun to change an outcome.
8. **Effective sample size.**
   - The end-to-end evidence is three decision elections, plus 2026 reported
     separately. End-to-end tests therefore gate against harm
     (non-inferiority) and require better calibration.
   - Evidence of improvement must come from T1 (about 1,600 polls) and T3
     (residual-level scoring).

## 2. T1: post-election behaviour of the poll aggregate

**Question.** Does letting the filter expect a jump when an election result
arrives improve its out-of-sample prediction of polls after elections,
without harming it mid-cycle?

### 2.1 Arms

All arms run the filter in `scripts/poll_aggregate/model.py` and the
segment-refit design in `build.py`, unchanged except as stated.

| arm | change after each election observation (the first anchor election, 2006, excepted) |
| --- | --- |
| `C0` | none (v0.2) |
| `S` | `Cov[s, s] += s_shock × multinomial_shape(s)`, at the post-update mean |
| `H` | every house block `+= h_reset × reference_shape` |
| `SH` | both |

**`C0` uses production's stored segment fits.** These are the
`hyperparameters_exact` in the pinned v0.2 metadata, and are not refitted.
Its log-likelihoods are recomputed by the T1 harness with the same filter
code as the other arms.

`s_shock` and `h_reset` are hyperparameters estimated with the five existing
ones. Each segment is fitted by maximum likelihood on its fit date, using only
data known then. The procedure is production's: Nelder–Mead from the
configured start and from the previous segment's fit, then a restart from the
best point.

| parameter | bounds (log scale) | start |
| --- | --- | --- |
| `s_shock` | [1e-6, 1e-1] | the geometric midpoint, 3.162e-4 |
| `h_reset` | [1e-8, 1e-2] | the geometric midpoint, 1e-5 |

The starts are not the grid values in §0.

**One extra start for the non-control arms.** `C0`'s fitted values for the
same segment, with the new parameters at their starts. The arms nest `C0`
(at their lower bounds they add negligible variance), so this start keeps a
poor Nelder–Mead run from showing up as a model difference.

### 2.2 Scores

- **Score.** Each poll's one-step-ahead log predictive density: the filter's
  innovation log-likelihood for that poll, eight-dimensional Gaussian, in
  nats. It is computed prequentially in fieldwork order, with the
  hyperparameters of the segment the poll falls in. Election observations
  are not scored.
- **Segments.** A poll belongs to the segment of the last election `e` whose
  day is on or before its fieldwork midpoint. It is scored with the fit made
  when `e`'s result became available.
- **Out of sample only.** Polls published on or before that fit date were in
  the fit's data, so they are not scored.
- **Post-election window `W_e`.** Scored polls whose fieldwork midpoint is
  within 120 days on or after election day `e`.
- **Mid-cycle `M_e`.** The remaining scored polls of segment `e`.
- **Decision windows: `e ∈ {2014, 2018, 2022}`.** Each is scored with a fit
  that saw at least one complete earlier post-election window.
  - 2010 is reported but carries no decision weight. Its fit, on
    2010-09-26, has only one week of post-election polls, so `s_shock` and
    `h_reset` are not identified there. This is a design reason, fixed before
    any score.
  - 2026 is reported separately (§1.5).

### 2.3 Gates (each non-control arm against `C0`)

| gate | condition |
| --- | --- |
| T1-G1 post-election gain | summed log score over `W_e` higher than `C0` in at least 2 of the 3 decision windows, **and** higher summed over the three |
| T1-G2 no mid-cycle harm | for each of `M_2010`, `M_2014`, `M_2018`, `M_2022`: mean log score per poll ≥ `C0`'s − 0.01 nats |
| T1-G3 integrity | at every fit date, the arm's maximized in-sample log-likelihood is at least `C0`'s − 0.5 nats (it nests `C0`, so a shortfall is an optimizer failure); the aggregate stays on the simplex; earlier rows are unchanged when later data are appended (the existing causality test) |

**Selection.**

- If several filter arms pass, `SH` is chosen only if its sum over the
  decision windows beats the better single arm by at least 2.0 nats.
- Otherwise the single arm with the larger sum is chosen; on an exact tie,
  `S`.
- If no arm passes, the filter stays `C0`.

### 2.4 Rolling consensus weights (reported level only)

v0.2 freezes each pollster's consensus weight at the segment's fit date for
the whole segment. In 2026 this keeps 13 % weight on Demoskop, which has not
published since 2026-09-09.

- **Arm `W`.** Recompute the weights daily: each pollster's share of
  effective sample among polls known on day `T` with fieldwork midpoint in
  the preceding 365 days. The test is v0.2's own,
  `(T − midpoint).days < 365`. `W` sits on top of the selected filter arm,
  and changes no likelihood.
- **Effective sample.** A poll's sample size after the aggregate's own
  missing-size imputation and rolling-tracker overlap discount
  (`Observation.sample_size`).
- **Metric.** For each day `T` with at least three polls published in
  `(T, T + 30]`:
  - take the squared error between the reading on `T` and the
    effective-sample-weighted mean of those polls' raw shares;
  - average it over the eight parties.

  A segment's score is the mean over its qualifying days.
- **T1-GW.** `W` is adopted if its score beats frozen weights in at least 3
  of the 4 segments 2010, 2014, 2018 and 2022, **and** on the pooled score.
  The 2026 segment is reported only.

### 2.5 Reported, no decision weight

- Fitted `s_shock` and `h_reset` per segment.
- Standardized innovation mean and SD by window and party.
- For 2026, the M reading on 2026-09-29, 2026-10-14 and 2026-11-18 under
  every arm.

The result becomes the candidate `SwedishPollsAggregate-v0.3`. v0.2 and its
files are not modified.

## 3. T3: the election-noise law

**Precondition: satisfied.** Amendments 001 and 002 are both in effect. Every
residual in T3 and T4 is
`result − build_eve_polling_consensus(E, polls, window_days = 14)`, with:

- the pinned polls (§1);
- `result` the certified nine-category result (for 2026, REST is 100 minus
  the eight parties);
- the vector zero-sum cleaned as in
  `scripts/election_layer_v2/residuals_pool.py`.

These are the corrected residuals.

### 3.1 Arms

A residual vector `r` (nine categories, pp, zero-sum) is drawn as
`r = μ_β + ε`, where:

- `r̄_P` is the pool mean bias, computed and zero-sum cleaned exactly as in
  `scripts/election_layer_v2/residuals_pool.py`;
- `μ_β = β · r̄_P`;
- `Σ̃` is the covariance from `fit_election_noise_b` on the centred pool. The
  fitting procedure is unchanged.

| factor | options |
| --- | --- |
| bias `β` | 0 (the control: centred), 0.5, 1 |
| tails | Gaussian `ε ~ N(0, Σ̃)`; or Student-t with ν = 5 scaled to the same covariance: `ε = √((ν−2)/ν) · Z Lᵀ / √(V/ν)`, with `V ~ χ²_ν` and `L Lᵀ = Σ̃` (the existing symmetric factor) |

Six arms. `(Gaussian, β = 0)` is the control, `pp_lw_gaussian`. Nothing is
tuned: the options are exactly these.

### 3.2 Evaluation

- **Primary: rolling origin.** Targets 2014, 2018 and 2022, each scored with
  a pool of the elections strictly before it (K = 3, 4, 5).
- **Secondary: leave one election out** over 2002–2022 (six folds, K = 5).
  This uses later elections for earlier targets, so it is a robustness check
  only.
- **Held out: 2026,** with the pool 2002–2022 (K = 6). Reported for every
  arm.

Each case:

- **Forecast.** The predictive vote distribution is the target's corrected
  consensus plus 20,000 draws of `r`, applied exactly as the engine applies
  noise: `apply_batch_simplex_transfer` with `eps` = 0.01 pp. The truth is the
  certified result.
- **Primary score.** The nine-category energy score,
  `d1_joint_vote_energy_score`, from
  `diagnostics/election_noise_v2/control_baseline/harness/metrics.py`
  (SHA-256 `ce872a25…3810`).
- **Secondary scores** (`d2`): CRPS and 50/90 % coverage.
- **Seeds.** The seed is the integer value of the first 8 hex digits of
  `sha256("calibration-2030:T3:<target year>:<design>")`, where `<design>` is
  `rolling`, `loo` or `heldout`.
- **Draw order.** One `numpy.random.default_rng(seed)` draws `Z` (20,000 × 9
  standard normals) and then `V` (20,000 values of χ²₅). All arms of a case
  share the same `Z` and `V`.
- **Monte Carlo sensitivity.** Every case is repeated with four more seeds,
  made by appending `:1` to `:4` to the seed string.
  If a gate outcome differs across the five seeds, the gate is reported as
  Monte-Carlo-fragile and is treated as **failed**.

### 3.3 Gates (each arm against the control)

| gate | condition |
| --- | --- |
| T3-G1 | mean ES over the rolling targets lower than the control's |
| T3-G2 | on each rolling target, ES ≤ 1.10 × the control's |
| T3-G3 | mean leave-one-out ES ≤ 1.02 × the control's |

**Selection.**

1. Among passing arms, choose the lowest mean rolling ES.
2. If a less complex arm passes and is within 1 % of it, choose the less
   complex arm. Complexity:

   | complexity | arms |
   | --- | --- |
   | 0 | (Gaussian, 0) |
   | 1 | (Gaussian, 0.5), (Gaussian, 1), (t, 0) |
   | 2 | (t, 0.5), (t, 1) |

3. Ties at equal complexity go to the smaller `β`, then to Gaussian.
4. If no arm passes, the control is retained.

**2026 flag.** If the selected arm's 2026 ES exceeds 1.15 × the control's,
the result is flagged. The user decides whether T4 proceeds with it. The
selection itself is not changed.

### 3.4 Final refit (decision of 2026-11-18)

- **2030 candidate law.** The selected specification, refitted by its frozen
  procedure on all seven corrected residuals (2002–2026).
- **T4's historical cases** use the same specification, fitted on the
  elections strictly before each target.

## 4. T4: end-to-end replay (gates frozen with this document)

### 4.1 Arms

| arm | definition |
| --- | --- |
| `C0` | production 1.2.0-rc1 as pinned in §1 |
| `F1` | `C0`, with the noise pool built from the corrected residuals (amendments 001 and 002) |
| `CAND` | `F1` plus the T1 selection, the T3 selection, and any T2 selection under its own preregistration |

`F1` isolates the correction. It removes an information leak, so it is
adopted on correctness grounds, not by score. Its scores are reported against
`C0`, and are not gated.

### 4.2 Cases

- **Runner.** `scripts.simulator.engine.simulate_election` end to end,
  through the replay harness.
- **Draws.** 20,000 draws, seed 20260929, the same for every arm.
- **Decision set `D`.** Elections 2014, 2018 and 2022 at 112, 84, 56, 28, 14
  and 7 days (18 cases).
- **Long-horizon set `L`.** The same elections at 365, 730 and 1,095 days
  (9 cases).
- **2026.** Both horizon sets, reported separately.
- **Noise pools.** Each target's pool contains only earlier elections.

### 4.3 Metrics

As in protocol v4 §6, with the same frozen metric module:

- ES_vote (primary);
- CRPS_vote;
- ES_seat;
- Brier_coalition;
- Brier_threshold;
- Coverage_50 and Coverage_90.

Pooled values are unweighted means over the cases of a set.

### 4.4 Gates (`CAND` must pass all of them)

| gate | condition, on set `D` unless stated |
| --- | --- |
| G1 accuracy | pooled ES_vote ≤ 1.05 × `C0` **and** ≤ 1.05 × `F1` |
| G2 per election | for each of 2014, 2018 and 2022, mean ES_vote over its horizons ≤ 1.15 × `C0` |
| G3 seats | pooled ES_seat ≤ 1.05 × `C0` and ≤ 1.05 × `F1`; Brier_coalition ≤ `C0` + 0.01 |
| G4 thresholds | Brier_threshold ≤ `C0` + 0.01 |
| G5 calibration | Coverage_90 in [0.80, 0.97], **and** \|Coverage_90 − 0.90\| < \|Coverage_90(`C0`) − 0.90\|; Coverage_50 in [0.35, 0.65] |
| G6 integrity | for every arm: every case completes; every draw allocates 349 seats; each case uses an estimate dated on its `as_of` and not informed after it (protocol v4 §7.1); the checkout is clean; the inputs match §1 |
| G7 long horizon (set `L`) | pooled ES_vote ≤ 1.05 × `C0`. If `CAND` includes a T2 dynamics change, also \|Coverage_90 − 0.90\| < \|Coverage_90(`C0`) − 0.90\| |

**2026 flag.** If `CAND`'s mean ES_vote on 2026 (set `D` horizons) exceeds
1.15 × `C0`'s, the result is flagged for the user's decision.

### 4.5 Decisions T4 can produce

- **Pass.** A proposal for a new model release: a new `MODEL_VERSION`
  (expected 1.3.0-rc1) and an aggregate v0.3 if T1 selected a change, both
  adopted through their own amendment. Production keeps 1.2.0-rc1 until the
  user adopts it.
- **Fail.** No production change. The results are published, and `F1` alone
  may still be proposed on correctness grounds.

## 5. Not preregistered here

- **T2:** OpinionState uncertainty, the cross-election 60-day window and the
  dynamics horizon cap. This needs its own preregistration, frozen before T4
  runs, or an explicit user decision to run T4 without T2.
- **T5:** a prospective shadow run of the T4 candidate. Its own
  preregistration is to be frozen before any shadow forecast is archived.

## 6. Order

1. Amendments 001 and 002 into effect (done 2026-11-18).
2. Freeze this document.
3. Run T1 and T3, which are independent: T3's residuals do not read the
   aggregate.
4. T2: preregister and run, or skip by decision.
5. T4.
6. A release proposal, on a pass.

**A T4 interaction to watch.** The forecast centre is the aggregate's
consensus reading, while the noise is defined against a final-poll consensus.
Arm `W` can change the reading's level, and T4 is where that interaction is
scored.

## 7. Freeze procedure

The same body-hash scheme as `docs/election_noise_v2_preregistration.md`
§J.3:

- **`BODY SHA-256`** covers every byte before the last
  `<!-- FREEZE-BLOCK-START -->` marker.
- **The freeze record** is
  `data/processed/calibration_2030/preregistration.json`. It holds:
  - the body hash;
  - the actual UTC freeze timestamp;
  - the base commit;
  - the input hashes from §1.
- **The record's own hash** goes in its `.sha256` companion.
- **The freeze commit** is reported with the freeze. It cannot be embedded in
  the file it hashes.

<!-- FREEZE-BLOCK-START -->

```
PREREGISTRATION STATUS: FROZEN

FREEZE TIMESTAMP (UTC): 2026-11-18T16:46:25Z
FREEZE BASE COMMIT:     254fb676cb5bbc18b5dc7f19530d95c5e4632db9
FREEZE COMMIT:          the commit that adds this block (reported with the
                        freeze; it cannot be embedded in the file it hashes)
BODY SHA-256:           7707c2afb754ff3a5d6ea8f85034321a6f2e0de54117012220c6da46fa66ee53
DECIDED BY:             user (approval of 2026-11-18)
RECORD:                 data/processed/calibration_2030/preregistration.json
```

BODY SHA-256 covers every byte of this file strictly before the last
occurrence of the freeze-block marker. Verify with:

```bash
python3 - <<'PY'
import hashlib, pathlib
b = pathlib.Path("docs/calibration_2030_preregistration.md").read_bytes()
marker = b"<!-- FREEZE-BLOCK-" + b"START -->"
print(hashlib.sha256(b.rsplit(marker, 1)[0]).hexdigest())
PY
```
