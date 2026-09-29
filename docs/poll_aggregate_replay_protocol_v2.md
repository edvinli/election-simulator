# SwedishPolls aggregate: replay protocol v2

**Status: frozen before scoring.** Agreement is recorded by setting
`REPLAY_V2.agreed_sha256` (and `REPLAY_V2.baseline_commit`) in
`scripts/poll_aggregate/replay.py` to this file's SHA-256, never by editing
this file. The user authorized scoring this protocol once frozen, with the six
v1 margins unchanged. The harness refuses to score unless the hash matches.

This fixes, before v0.2 is scored against any outcome, how the full forecast
driven by `SwedishPollsAggregate-v0.2` is compared with the production
forecast driven by pollofpolls.se. **Everything except the aggregate and its
baseline is identical to replay protocol v1**
(`docs/poll_aggregate_replay_protocol.md`, SHA-256
`3a34221883158a76b2b1021605c2cb1f636a44bba26b06063b2e727c6dc1be5c`): arms,
cases, draws, seed, truth, metrics, and the six gates with their margins.
Replay v1 and its FAIL result stand unchanged.

## 1. What is frozen

| item | value |
| --- | --- |
| Aggregate | `SwedishPollsAggregate-v0.2`, commit `7153bcb` (`data/processed/poll_aggregate/v0_2/`, timeseries SHA-256 `23ce416d2f76cad53acbf0da1eb67fc101b028b346c4e63c3065e98cecc7fb8f`) |
| Simulator | production code at the same commit: `MODEL_VERSION` 1.1.0-rc1, noise law `pp_lw_gaussian` |
| Other inputs | the committed processed geography, mandates, election results and SwedishPolls snapshot |

No aggregate hyperparameter, cleaning rule or model choice may change after
this protocol is agreed. A change is a new aggregate version and needs a new
protocol version and a new run.

## 2. Arms

Both arms run `scripts.simulator.engine.simulate_election` end to end
(OpinionState → dynamics → election noise → geography → seats). They differ
only in two files of the `pollofpolls/` data folder:

| arm | `pollofpolls_timeseries.csv` | `individual_polls.csv` | gated |
| --- | --- | --- | --- |
| `CONTROL_POP` | production PoP series | production PoP-reconstructed polls | reference |
| `CANDIDATE_SWEDISHPOLLS` | the v0.2 aggregate | SwedishPolls polls that pass the aggregate's cleaning, in the same schema | **yes** |
| `DIAGNOSTIC_TS_ONLY` | the v0.2 aggregate | production PoP-reconstructed polls | no (attribution) |

The candidate arm is what production would run after migration: no PoP input
at all. It replaces PoP as OpinionState's centre, as the reference for
house effects and covariance, and as the history for dynamics transitions.

## 3. Election noise and the sparse-history question

The election-noise pool is **unchanged in every arm**. Its residuals are the
certified result minus a consensus of each pollster's last poll
(`scripts/election_residuals/consensus.py`), built from SwedishPolls; they do
not use PoP or the aggregate. The chronological pool therefore has
K = 4, 5 and 6 elections for 2018, 2022 and 2026, above the law's minimum of
two, and no sparse-history fallback is needed.

The noise is centred (`election_noise_b.py` draws `N(0, Σ̃)`), so this
replay does not correct any level difference between the aggregate and
PoP; that difference is exactly what the replay measures. Adding an industry
bias term was evaluated and rejected before (`docs/industry_bias_audit.md`).

Redefining residuals as result minus the aggregate's pre-election estimate is
**out of scope**. The aggregate has pre-election estimates only for 2014,
2018, 2022 and 2026, so a chronological 2018 fit would have one residual.
If that is pursued later, its fallback must be declared in its own protocol
before scoring; one natural rule is to use poll-consensus residuals for
elections without an aggregate estimate, with a minimum K of 2.

## 4. Cases

Elections 2018-09-09, 2022-09-11 and 2026-09-13, at the frozen competition
horizons of 112, 84, 56, 28, 14 and 7 days: 18 cases, with `as_of` equal to
election day minus the horizon. Every arm runs the same 18 cases with 20,000
draws and base seed 20260929.

2014 is excluded. The canonical PoP series starts on 2014-09-15, so there is
no control, and the production geography has no 2010 constituency baseline,
so the full forecast cannot be run for 2014 in either arm.

## 5. Truth

- Vote shares: nine categories (the eight parties plus the remainder), in
  percent of valid votes, from `data/processed/elections/riksdag_election_results.csv`
  (2018, 2022) and the certified 2026 manifest.
- Seats: national totals from `data/processed/mandates/historical_certified_mandates.csv`
  (2018, 2022) and the certified 2026 manifest.

## 6. Metrics

Computed with the frozen metric module
`diagnostics/election_noise_v2/control_baseline/harness/metrics.py`, except
the threshold Brier score, which is new and defined here.

| metric | definition |
| --- | --- |
| **ES_vote (primary)** | nine-category vote energy score in pp (`d1_joint_vote_energy_score`) |
| CRPS_vote | eight-party mean CRPS in pp (`d2_marginal_vote_metrics`) |
| ES_seat | eight-party seat energy score (`d3_seat_metrics`) |
| Brier_coalition | mean majority Brier score over coalition masks 1–254 (`d4_coalition_brier`) |
| Brier_threshold | mean over the eight parties of `(P(share ≥ 4 %) − 1[actual ≥ 4 %])²` |
| Coverage_50, Coverage_90 | share of the 9 × 18 central 50 % and 90 % intervals containing the outcome |

Pooled values are unweighted means over the 18 cases.

## 7. Gates

The candidate **passes** only if every gate holds.

| gate | condition |
| --- | --- |
| G1 accuracy | pooled ES_vote(candidate) ≤ 1.05 × pooled ES_vote(control) |
| G2 per election | for each election, mean ES_vote over its six horizons, candidate ≤ 1.15 × control |
| G3 seats | pooled ES_seat(candidate) ≤ 1.05 × control, and pooled Brier_coalition(candidate) ≤ control + 0.01 |
| G4 thresholds | pooled Brier_threshold(candidate) ≤ control + 0.01 |
| G5 calibration | Coverage_90(candidate) ≥ 0.80 and ≥ control − 0.05; Coverage_50(candidate) ≥ 0.35 and ≥ control − 0.10 |
| G6 integrity | all 18 cases complete in both arms; every draw allocates 349 seats; the estimate each case actually uses is dated on its `as_of` (section 7.1) |

The gates test **non-inferiority**, not superiority, because the control is
advantaged (section 8).

### 7.1 How G6's date condition is checked

The simulation summary's `as_of` only echoes the requested date, so it is not
used. For every case the harness calls `estimate_opinion` exactly as the
engine does, on the same data folder, and records the date and values of the
timeseries row it selects (`selected_estimate_date`, `selected_estimate_pct`).
A gated case satisfies G6's date condition only if:

- the selected row is dated on the case's `as_of` (both arms); and
- for the candidate, the selected values equal the v0.2 aggregate's row for that
  date (within 1e-6 pp per party), and that row's `information_date` is not
  after `as_of`.

### 7.2 Scored inputs must be the fixed baseline

The protocol hash locks this document, not the files the harness reads, so
`--score` additionally refuses to run unless:

- the checkout is clean: no modified, staged or untracked file; and
- `git diff` between the baseline commit
  `7153bcbef80518a3c3e32f76ca35ec6034a69557` and HEAD is empty under `data/`,
  `scripts/`, `diagnostics/`, `pyproject.toml` and `uv.lock`, with the single
  exception of the harness `scripts/poll_aggregate/replay.py`.

With a clean checkout, the aggregate, the polling snapshot, the election and
mandate data, the geography and the simulator code read during scoring are
then byte-identical to the baseline. The HEAD commit and these checks are
written into the scored provenance.

## 8. Why every case is retrospective

- **The control has look-ahead.** The stored PoP series is fieldwork-dated
  and backfilled as later polls are published; the repository's audit
  attributes a revision to a poll published five days after the date it
  changed and measured look-ahead of up to 21 days at 2014 horizons
  (`docs/election_noise_v2_historical_pop_extension.md`, section 4). No
  real-time PoP vintages exist for 2018 or 2022; the 2026 values are subject
  to the same backfill.
  The level investigation quantified it on the only real vintages that exist:
  for the 2026 7- and 14-day cases, the production forecast rerun on the PoP
  snapshot actually committed by each date scores ES_vote 3.28, against 2.53
  on the stored series (`docs/poll_aggregate_level_investigation.md`).
- **The candidate was designed after the outcomes.** v0.1 was built with all
  three results known. v0.2 exists because v0.1 failed replay v1, and the
  investigation that led to it used outcome-scored diagnostics (per-party
  forecast errors). The specific v0.2 change, reporting the sample-weighted
  consensus reading, and every check made on v0.2 before this protocol were
  outcome-free (`diagnostics/poll_aggregate_level/level_structure.py`), and
  v0.2 reuses v0.1's causal hyperparameter fits unchanged. It is still not
  independent of these outcomes.

So no case is a true out-of-sample test for either arm, and a v2 result must
not be described as independent validation. The first prospective evidence is
the 2030 cycle, run in shadow.

## 9. Decisions this replay can produce

- **PASS:** the aggregate may proceed to step 3 (migrating every consumer
  together). Passing does not itself switch production.
- **FAIL:** the aggregate stays in shadow mode. The gates are not relaxed,
  production is not switched to release the 2030 forecast, and v0.2 is not
  tuned in response to its score.

Results are written to `data/processed/poll_aggregate/replay_v2/` with
per-case rows, pooled metrics, the gate table and the decision
(`uv run python -m scripts.poll_aggregate.replay --replay v2 --score`).
