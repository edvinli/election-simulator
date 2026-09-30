# SwedishPolls aggregate v0.1

`SwedishPollsAggregate-v0.1` is a causal daily estimate of party support built
only from [SwedishPolls](https://github.com/MansMeg/SwedishPolls) and certified
election results. It exists so the simulator can stop depending on
pollofpolls.se for its opinion input.

**Status: production opinion input from model 1.2.0-rc1 (v0.2).** Replays v1
and v2 (against stored Poll of Polls) and v4 (against a SwedishPolls-only
consensus baseline) all remain FAIL. v4 failed only G5 calibration, and v0.2
was released under an explicit exception recorded in
`docs/poll_aggregate_replay_protocol_v4_amendment_001.md`. That is not a gate
pass, and 90 % interval calibration remains an open issue. All historical
testing is retrospective, because v0.2 was developed after the 2018, 2022 and
2026 elections.

## Versions

| version | reported level | build |
| --- | --- | --- |
| v0.1 | latent support `s`, whose level is tied to past results through the house effects | `uv run python -m scripts.poll_aggregate` (writes `data/processed/poll_aggregate/`) |
| v0.2 | the consensus reading `s + Σ_h w_h b_h` | `uv run python -m scripts.poll_aggregate --version v0.2` (writes `data/processed/poll_aggregate/v0_2/`) |

Both run the same observations, filter and hyperparameter fits; v0.2's
fitted hyperparameters equal v0.1's. For v0.2, `w_h` is each pollster's share
of effective sample among polls known at the segment's fit date whose
fieldwork midpoint is in the preceding 365 days, fixed for the segment. The
reading is what a sample-weighted mix of current pollsters would report, the
scale on which the election-noise residuals are defined (result minus a
sample-weighted pollster consensus). Why v0.2 exists is in
`docs/poll_aggregate_level_investigation.md`; it is scored under
`docs/poll_aggregate_replay_protocol_v2.md`.

## Usage

```bash
uv run python -m scripts.poll_aggregate
```

Reads `data/processed/pollofpolls/swedishpolls_individual_polls.csv`,
`data/processed/elections/riksdag_election_results.csv` and the certified 2026
manifest, and writes `data/processed/poll_aggregate/`:

- `swedishpolls_aggregate_timeseries.csv`: one row per day, in exactly the
  columns of `pollofpolls_timeseries.csv`. Other parties are in `other`; `FI`
  is empty. `source_extra_json` carries each row's per-party standard
  deviation, information date, hyperparameter fit date, number of polls known
  and latest poll publication date.
- `swedishpolls_aggregate_metadata.json`: input hashes, every exclusion by
  reason and poll id, imputed sample sizes, and each segment's fitted
  hyperparameters and likelihood.

A full rebuild takes a few minutes.

## What "causal" means here

- **Availability, not fieldwork, decides what is known.** A poll is known from
  its publication date; it describes opinion at its fieldwork midpoint. Ipsos
  fieldwork 15–27 September 2026 is a reading of 21 September that first
  exists on 29 September.
- **Each row is its own replay.** The row for day *T* runs the filter over
  exactly the observations available by *T*, in fieldwork order, and reports
  the filtered (not smoothed) state on *T*. A poll published later, whatever
  its fieldwork dates, cannot change it; earlier rows of a rebuild with more
  data are identical. `tests/test_poll_aggregate.py` asserts this.
- **Election results have availability dates.** The 2026 result is known from
  its manifest's retrieval, 19 September 2026, six days after election day.
  Historical results are treated as known seven days after election day. The
  actual historical publication dates are not recorded in the repository;
  if a final count came out later than that, the lag must be raised.
- **Hyperparameters are frozen per segment.** They are refitted by maximum
  likelihood on the day each election result becomes available, using only
  what was known then, and held until the next result. Rows begin at the
  first refit (26 September 2010); 2006–2010 is burn-in.
- **Sample-size imputation and overlap discounts look backwards only** (see
  below).

## Model

The state is the eight named-party shares in percentage points (other parties
are the remainder) plus one additive house-effect vector per pollster.

- Support follows a random walk. Daily variance is
  `process × 1e4 × (diag(π) − ππᵀ)` at the current estimate, so small parties
  move less in points.
- A poll reads support plus its pollster's house effect, with noise
  `design_effect × 1e4 × (diag(π) − ππᵀ) / n_eff` plus rounding variance
  (`res²/12` per party, where `res` is 1 pp for whole-number polls and 0.1 pp
  otherwise).
- An election reads support with no house effect and noise of an
  `election_n`-respondent poll. That finite noise reflects that votes and poll
  responses are different quantities; the election is an informative anchor,
  not a hard reset.
- House effects start at `N(0, house_prior × shape)` and drift by
  `house_drift × shape` per day, in the multinomial shape of the latest
  election known at the fit date.

Shares are modelled directly rather than in log-ratio coordinates: that avoids
taking logs of a poll's remainder, which is small, rounded and sometimes zero.
The filter reports mean and covariance; downstream code can still move to
log-ratio space as `OpinionState` does.

Nelder-Mead is started from the default hyperparameters and from the previous
segment's fit, then restarted from the best point.

## Cleaning rules

| rule | reason |
| --- | --- |
| missing publication or fieldwork dates | availability cannot be established; not guessed |
| fieldwork ending after publication, or ending before it starts | inconsistent record (mostly 2008–2012 Sentio monthly rows) |
| fieldwork ending before the 2006 election | SD is not reported consistently before late 2006 |
| any named party missing | the eight-party composition is incomplete |
| named parties summing above 101 % | implausible; the remainder is other parties |
| fieldwork containing an election day | exit and election-day polls (SVT VALU, TV4) are superseded by the count |
| same pollster and identical fieldwork as an earlier-published poll | double listing; the first published is kept |

Named parties are **not** required to sum to 100: the remainder is other
parties (the 29 September 2026 Ipsos row sums to 98 %).

- **Rolling trackers.** A poll's sample size is scaled by the share of its
  fieldwork days not already covered by same-pollster polls published no
  later than it. A daily three-day tracker counts one third of its sample
  after the first day.
- **Missing sample sizes** are the median of that pollster's earlier-published
  reported sizes, else all pollsters', else 1,000.

## First diagnostics (not the validation)

These are sanity checks from the committed build, not the replay
evaluation that decides the switch. PoP's timeseries starts 15 September
2014.

Build of 29 September 2026 (polling snapshot of 28 September; 1,632 polls
used):

| check | aggregate | PoP |
| --- | --- | --- |
| mean absolute daily difference from PoP, 2014-09-15 to 2026-09-11 (per party) | M 0.45, L 0.24, C 0.28, KD 0.25, S 0.66, V 0.41, MP 0.40, SD 0.84 | – |
| mean signed difference from PoP | SD +0.76, S −0.35, MP −0.36, V −0.33, others within ±0.15 | – |
| mean absolute daily change (average over parties, pp) | 0.032 | 0.016 |
| eve of 2018: MAE / max error vs result (pp) | 1.44 / 3.62 (S) | 0.60 / 1.16 |
| eve of 2022 | 0.69 / 1.80 (S) | 0.26 / 0.65 |
| eve of 2026 | 0.98 / 2.31 (M) | 1.05 / 2.55 |

Fitted hyperparameters are stable across the five segments (`process`
1.1–1.5e-5, `design_effect` 0.86–1.05, `election_n` 1,000–4,300). An
`election_n` near 1,400 means an election moves the estimate roughly as much
as one large poll: when the 2026 result became available on 19 September, M
moved from 17.54 to 18.42 against a result of 19.85.

## Known limitations and open questions for the replay

- **Level relative to PoP.** On the eves of 2018 and 2022 PoP was much
  closer to the result than this aggregate, which tracks the raw poll average
  (2018 S: aggregate 24.6, raw two-week poll mean 24.6, PoP 27.2, result
  28.3). PoP applies corrections this model does not have. The stored PoP
  values are also not what was visible in real time: the series is
  fieldwork-dated and backfilled as later polls are published, with measured
  look-ahead of up to 21 days
  (`docs/election_noise_v2_historical_pop_extension.md`, section 4), so the
  PoP side of these comparisons is retrospective and advantaged. On the eve of
  2026 the two are comparable.
- **The election-noise layer does not depend on PoP.** (An earlier version of
  this document said it did.) Its residuals are the result minus a consensus
  of each pollster's last poll, built from SwedishPolls, and its draws are
  centred, so it neither needs recalibrating for this series nor corrects a
  level difference. The full-forecast comparison is fixed in
  `docs/poll_aggregate_replay_protocol.md`.
- **Persistent house effects were tried and rejected**: forcing near-zero
  house drift lowers the likelihood substantially and does not improve eve
  accuracy.
- **Volatility.** The aggregate moves about twice as much per day as PoP.
  This is a diagnostic, not a tuning target; the replay scores the full
  forecast.
- **Rolling-tracker overlap** discounts sample size but ignores the
  correlation between overlapping polls' errors.
- **Pre-2013 exclusions** remove a few dozen polls without publication dates;
  imputing availability would risk leakage.
