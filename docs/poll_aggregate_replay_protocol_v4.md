# SwedishPolls aggregate: replay protocol v4 (benchmark against a SwedishPolls-only baseline)

**Status: frozen before scoring.** Agreement is recorded by setting
`REPLAY_V4.agreed_sha256` and `REPLAY_V4.baseline_commit` in
`scripts/poll_aggregate/replay.py`, never by editing this file. The user
directed this benchmark and asked for its acceptance criteria to be fixed
before the new scores are run.

v4 benchmarks the frozen `SwedishPollsAggregate-v0.2` against **election
outcomes and a simple SwedishPolls-only poll-consensus baseline**, using only
information published by each forecast date. It replaces backfilled PoP as the
release comparison, because the level investigation showed that the stored
PoP series carries hindsight a forecaster never had
(`docs/poll_aggregate_level_investigation.md`). Replays v1 and v2, and their
FAIL results against stored PoP, are preserved unchanged.

## 1. What is frozen

| item | value |
| --- | --- |
| Candidate | `SwedishPollsAggregate-v0.2` (`data/processed/poll_aggregate/v0_2/`, timeseries SHA-256 `23ce416d2f76cad53acbf0da1eb67fc101b028b346c4e63c3065e98cecc7fb8f`) |
| Baseline | `SwedishPollsConsensusBaseline-v1` (`data/processed/poll_aggregate/consensus_baseline/`, timeseries SHA-256 `fae2877e76ec9455eb8ff11b00d7314ae71c822d90944edba9e6e48c255bc5aa`) |
| Code and inputs | baseline commit `286b6a274cb31e188f557d3592d88207f46b39f8`: simulator `MODEL_VERSION` 1.1.0-rc1, noise law `pp_lw_gaussian`, committed geography, mandates, election results and SwedishPolls snapshot |

## 2. The baseline

For each day `T`, `scripts/poll_aggregate/consensus_baseline.py` does the
following:

- it takes, for each pollster, the latest poll published by `T` whose
  fieldwork midpoint lies in the 60 days up to `T`;
- it uses the aggregate's own poll-cleaning rules;
- it averages those polls weighted by effective sample.

There is no filter, no house effect and no election input. It is the
election-noise layer's consensus rule applied daily, the simplest defensible
opinion estimate from SwedishPolls alone. Every row uses only polls published
by its date.

## 3. Arms

Both arms run `scripts.simulator.engine.simulate_election` end to end. Both
read the same SwedishPolls individual polls. They differ **only** in
`pollofpolls_timeseries.csv`. No PoP input enters either arm.

| arm | timeseries | role |
| --- | --- | --- |
| `CONTROL_SWEDISHPOLLS_CONSENSUS` | the consensus baseline | control |
| `CANDIDATE_SWEDISHPOLLS` | v0.2 | gated candidate |

The election-noise pool is unchanged. It is built from SwedishPolls, with
K = 4, 5 and 6 for 2018, 2022 and 2026.

## 4. Cases

The same 18 cases as v1 and v2: the 2018, 2022 and 2026 elections at 112,
84, 56, 28, 14 and 7 days, with 20,000 draws and seed 20260929.

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

The control in every gate is the consensus baseline. The margins are v1's,
agreed by the user, and are unchanged.

### 7.1 How G6's date condition is checked

The simulation summary's `as_of` only echoes the requested date, so it is not
used. For every case the harness calls `estimate_opinion` exactly as the
engine does, on the same data folder, and records the date and values of the
timeseries row it selects (`selected_estimate_date`, `selected_estimate_pct`).
A gated case satisfies G6's date condition only if:

- the selected row is dated on the case's `as_of` (both arms); and
- for each arm, the selected values equal that arm's timeseries row for that
  date (within 1e-6 pp per party), and that row's `information_date` is not
  after `as_of`.

### 7.2 Scored inputs must be the fixed baseline

`--score` refuses to run unless the checkout is clean and `git diff` between
the baseline commit `286b6a274cb31e188f557d3592d88207f46b39f8` and HEAD is
empty under `data/`, `scripts/`, `diagnostics/`, `pyproject.toml` and
`uv.lock`, except for `scripts/poll_aggregate/replay.py`.

## 8. What is already known, and why this is retrospective

- **v0.2's own scores are already known.** The candidate arm runs the same
  code, data, draws and seed as replay v2's candidate, so its scores will
  reproduce replay v2's: pooled ES_vote 3.725, Coverage_90 0.784,
  Coverage_50 0.420.
  - Coverage_90 0.784 is below G5's absolute floor of 0.80. The floor is kept
    unchanged because it was agreed, so **G5 is expected to fail**.
  - The new information in v4 is the baseline's scores. That means G1–G4 and
    G5's relative part, which were not known when these criteria were fixed.
- **Every case is retrospective.** v0.2 was developed in September 2026,
  after the 2018, 2022 and 2026 elections, with their outcomes known. The
  investigation that led to it used outcome-scored diagnostics. So the
  baseline is tested out of sample in design terms, but v0.2 is not. A v4
  result must not be described as independent validation.

## 9. Decisions this benchmark can produce

- **PASS** (all six gates): migrating every production PoP dependency may
  proceed together, with the cycle guard and model version updated and the
  full publication path tested. Passing does not itself switch production.
- **FAIL:** production is not switched. The failing gates identify the
  weakness. It is fixed in a new aggregate version with a stated,
  outcome-independent rationale where possible, and scored under a new
  protocol version before any switch. The gates are not relaxed.

Results are written to `data/processed/poll_aggregate/replay_v4/`
(`uv run python -m scripts.poll_aggregate.replay --replay v4 --score`).
