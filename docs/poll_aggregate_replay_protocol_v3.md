# SwedishPolls aggregate: protocol v3 (prospective shadow evaluation)

**Status: frozen before any target poll exists.** Captures begin after this
protocol is committed. No future poll has been examined, because none has yet
been fielded after the first capture.

## 1. Purpose and scope

v3 compares, prospectively and in real time, how well two opinion estimates
predict the polls fielded after each day:

- the **PoP value actually available** that day, from pollofpolls.se; and
- the **v0.2 value**: `SwedishPollsAggregate-v0.2`'s consensus reading
  (baseline `7153bcb`).

It removes the look-ahead that confounded replays v1 and v2, because both
values are recorded as they stood at the time.

**v3 evaluates the opinion estimate only. It does not validate the full
election forecast.** Production stays on PoP and the 2030 cycle guard is
unchanged. Replacing PoP before the 2030 election would need a separate,
explicit decision about the forecast-level uncertainty v3 cannot resolve.
The v1 and v2 results and their gates stand unchanged.

## 2. Captures

`uv run python -m scripts.poll_aggregate.shadow capture` appends one record to
`data/shadow/v3/captures.jsonl`.

**What a record holds.**

- the UTC timestamp and the Stockholm date;
- the code commit, and whether tracked files were modified;
- **PoP:** the latest complete row of `pollofpolls.se/poll_img/data_table_tot.csv`
  dated on or before the Stockholm date, with that row's date, its staleness
  and the payload's SHA-256;
- **v0.2:** the consensus reading on the Stockholm date, from SwedishPolls
  (`Data/Polls.csv` and `Sources/sources.csv`, fetched at capture time and
  cleaned with the aggregate's own rules). It uses the frozen 2026 segment:
  the hyperparameters, reference election and consensus weights in
  `data/processed/poll_aggregate/v0_2/swedishpolls_aggregate_metadata.json`.
  That equals what a full v0.2 build would report for that date from the same
  polls (tested). The record also holds the payload hashes, the number of
  polls known and the latest publication date.

**Rules.**

- **Missing is missing.** A failed fetch, an invalid payload (for example a
  block page) or any error makes that source's record `missing`, with an
  error class and no values. The two sources fail independently. A missing
  value is never filled from PoP's later, revised history or from any later
  capture.
- **Append-only.** Records are never edited or deleted, and no record is ever
  created for a past date.
- **Schedule.** Captures run at least daily; the target is 06:30 and 18:30
  Europe/Stockholm. Extra captures are allowed. Days with no successful
  capture are missing for that source.
- **End date.** Captures end on 2030-09-07, the day before the 2030
  election, because the frozen segment is only valid until then.

## 3. Frozen scoring rules

Implemented in `scripts/poll_aggregate/shadow_score.py` as committed with this
protocol. Changing them requires a v3 amendment committed **before** the
analysis it affects.

**Forecast values.** For each Stockholm date `D` and each source, the value is
the latest successful capture dated `D`. There is none if no capture that day
succeeded. Values are never carried across dates.

**Target polls for `D`.** Polls from a SwedishPolls snapshot retrieved at
scoring time, cleaned with the aggregate's rules, and meeting all of:

- fieldwork starting after `D`, so the poll was unknown on `D`;
- fieldwork midpoint between `D + 7` and `D + 14` days; and
- published by `D + 28`.

A date is scorable only once `D + 28` has passed.

**Repeated polls from one pollster.** A pollster's eligible polls are
averaged, weighted by effective sample (rolling trackers already count only
new fieldwork days), into one reading. The target is:

- the effective-sample-weighted mean of the pollster readings (primary); and
- their equal-weighted mean (secondary).

A date with fewer than three pollsters has no target.

**Error.** The mean absolute error over the eight parties, in pp, of the
source's value minus the target.

**Primary endpoint.** The mean, over paired dates (both sources present and a
target exists), of `MAE(v0.2) − MAE(PoP)`. Its 95 % interval is a
moving-block bootstrap with 14-day blocks, 10,000 resamples and seed
20260930. Blocks are needed because neighbouring dates share targets.

**Secondary endpoints.**

- **Offset-adjusted MAE:** each source's own mean signed error per party over
  the analysis period is subtracted first. This measures tracking,
  independent of a constant level choice.
- **The equal-pollster target.**
- **Per-party mean signed error.**
- **Availability:** each source's share of scorable dates, and PoP
  staleness.

## 4. Analyses

- **Interim looks are descriptive only.** Captures, availability and
  staleness may be inspected at any time.
- **The first analysis comes after at least 26 weeks of captures and at least
  100 paired dates**, whichever is later. The report states the capture file
  hash, the target snapshot's hash and retrieval date, and the code commit.
- **Later analyses** repeat the same rules on a cumulative basis. Every one is
  reported, not only favourable ones.

## 5. Interpretation, fixed in advance

These are bands for describing the result. They are not a production gate.

- **Upper 95 % limit of the primary difference below 0:** v0.2 predicted
  subsequent polls better than the PoP value available at the time.
- **Interval within ±0.10 pp:** the two are comparable for this purpose.
- **Lower limit above 0:** v0.2 predicted subsequent polls worse.
- **Otherwise:** inconclusive at that analysis.

## 6. Known biases of this design

- **Polls as the target favour a consensus-scale estimate.** v0.2 is defined
  on the sample-weighted pollster consensus. PoP departs from that consensus
  on purpose, and its departures may help predict elections while hurting
  here. The offset-adjusted secondary endpoint and the per-party signed
  errors are there to separate level from tracking. A v0.2 advantage on the
  primary endpoint alone is not evidence of better election forecasts.
- **SwedishPolls is maintained by hand.** Late entry of polls lowers v0.2's
  freshness in the captures, and the same file supplies the targets. That is
  why targets are read from a later snapshot, with a 28-day deadline.
- **PoP availability depends on network access to pollofpolls.se** from the
  capture host. Failures reduce paired dates; they are never imputed.
- **Captures in the first weeks after the 2026 election** include PoP values
  from before the election. They are kept as they were, and staleness is
  reported.
