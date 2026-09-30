# The 2030 election cycle

The 2026 election (13 September 2026) is decided. The same model now forecasts
the next ordinary election, **8 September 2030**. This note records what
changes between the cycles, what does not, and which inputs are stand-ins
that must be replaced before 2030.

It is a design note, not a benchmark amendment. The 2026 prospective benchmark
(its dates, selection, weighting and scoring) is untouched, and all 2026
evidence stays pinned to 2026.

## What stays the same

- **The model.** `MODEL_VERSION` is unchanged, and so is the methodology: the
  OpinionState, Dynamics v2 with its 112-day cap, ElectionNoise B, the
  geographic projection and the mandate allocator. The frozen implementation
  files are byte-identical.
- **The election-noise pool.** It stays K = 6 (2002–2022). Adding 2026 would
  change the model, and that is a separate, governed decision.
- **2026-target forecasts are bit-identical.** A 2026 target still resolves
  to the 2022 baseline and the 2026 fixed seats. Only the recorded geography
  data hash moves, because the tables gained rows.

## What is derived from the target election

`scripts/simulator/election_cycles.py` derives every cycle-dependent value
from the target election rather than from 2026 constants:

| | 2026 cycle | 2030 cycle |
|---|---|---|
| Ordinary election day | 2026-09-13 | 2030-09-08 |
| Geography baseline | 2022 | 2026 |
| Fixed seats | `FIXED_SEATS_2026` | `FIXED_SEATS_2026` (stand-in) |
| First history point | 2022-09-18 | 2026-09-20 |
| Dynamics cap starts | 2026-05-24 | 2030-05-19 |

**Production callers pass the resolved baseline explicitly.** That covers
the publication pipeline and the projection simulator. `engine.py` itself is
unchanged in this step, because it is a hashed model file: editing it would
invalidate every reconstructed point of the deployed 2026 history. Two engine
changes come with the switch to 2030: resolving its own default baseline, and
refusing a baseline that does not precede its target.

**The input fingerprint hashes only what a forecast may depend on.** That
means:

- the baseline election's constituency votes;
- the electorate history up to the target, with the target's own outcome
  masked;
- a declared fixed-seat stand-in, by value.

Adding the certified 2026 rows therefore leaves every fingerprint in the
deployed 2026 history unchanged. That was verified against `main`.

## Data added

- `data/processed/geography/constituency_party_votes_2014_2022.csv` now also
  holds the certified **2026** constituency votes: 29 × 9 rows. The name stays
  because the frozen projection module reads the file by name. The rows are
  parsed from Valmyndigheten's committed `RD_S.json` by
  `scripts/geography/process.py` and verified by the benchmark's strict
  result loader. The tests check that:
  - they sum to the certified national votes;
  - the same file's previous-election fields reproduce the existing 2022 rows
    cell for cell;
  - the statutory allocator turns them into the certified seats.
- `constituency_electorates_2014_2026.csv`:
  - The 2026 rows now carry certified valid votes and turnout. The electorate
    is still the pre-election count.
  - 29 rows for **2030** repeat the 2026 electorate as a stand-in.

## History and automation for a new cycle

None of this changes anything while the target is 2026. Each 2026 answer is
asserted unchanged by `tests/test_election_cycle_2030_history.py`.

**A new cycle starts a fresh history.** When the existing artifact forecasts
an earlier election, the roll-in builds a new history holding the certified
point alone. Nothing of the old history is carried over: its points forecast
a decided election, and the website keeps it as a frozen archive at
`history/2026/`. The history's `schedule.cycle_start_date` is the first
publication date of an eligible poll fielded after the previous election. (It
was the first Poll of Polls estimate after it before model 1.2.0-rc1.) The
backfill fills the schedule from there.

**No forecast of the next election on a pre-election estimate.** Publication
refuses to certify until an eligible poll whose fieldwork began after the
previous election has been published. The model's opinion aggregate writes a
row every day, so a row date proves nothing. It reports
`AWAITING_POST_ELECTION_POLLS`, a successful no-op, until then.

**The schedule is the election's own.** Weekly anchors run from the cycle
start, and dates are daily from the election's dynamics cap.
`missing_curve_dates` also treats every publication day that carries only an
archived point as a hole, wherever it falls. In the weekly part of a cycle,
daily publications leave such days between the anchors, and the chart would
otherwise break its line at each of them.

**Elections are kept apart:**

- A snapshot made for an earlier election is never a point of a later
  election's history.
- The first forecast of a cycle has no "change since prior".
- Rendering takes the election from the generation's own snapshot and
  refuses a caller that names another.
- The completeness check reads the election from the deployed history.

**Future views only in the final 112 days.** Both the secondary fan and the
campaign paths are built and required only from the election's dynamics cap
(2030-05-19). Four years out, they would take about 1,440 simulations and
about 17 GB of path arrays per publication, to describe movement the model
caps at 112 days. The website already works without them.

## The switch

`DEFAULT_ELECTION_DATE` is `2030-09-08`, and the baseline label
`DEFAULT_GEOGRAPHY_BASELINE_YEAR` is 2026. The engine now:

- resolves its own default baseline from the target;
- refuses a baseline that does not precede the target outside oracle mode;
- takes fixed seats from the cycle rules.

`config.py` is recorded as intentional drift in the publication freeze
(`CYCLE_2030_CHANGED`). `engine.py` was already known drift in every
freeze.

Tests that exercise the 2026 campaign on 2026 fixtures pin that election
explicitly rather than relying on the default. That covers the automation,
fallback and backfill-kill scenarios. The live-artifact reuse lane is
skipped while the committed history belongs to a decided election, since no
publication resumes it. It runs again once the first 2030 history is
committed.

### Merging it safely

Merging arms the hourly publication cron: the date guard now passes, and
`ELECTION_AUTOMATION_ENABLED` is `true`. The render workflow has no kill
switch and fires after every publication run. The sequence is:

1. Set the repository variable `ELECTION_AUTOMATION_ENABLED=false`.
2. Merge.
3. Dispatch a manual `dry_run`, then a manual `publish`. Manual dispatch
   bypasses the kill switch by design.
4. Check the website:
   - the 2026 archive and the new 2030 history are joined on one chart;
   - `results/2026.json` is still served;
   - the pointer names the 2030 generation.
5. Re-enable the variable.

Until a poll fielded after the previous election has been published, every
run reports `AWAITING_POST_ELECTION_POLLS` and changes nothing.

## Stand-ins to replace

| Input | Stand-in | Replace when |
|---|---|---|
| 2030 fixed seats per constituency | the 2026 distribution | Valmyndigheten decides them (spring 2030) |
| 2030 electorate per constituency | the 2026 electorate | the 2030 count is published (August 2030) |

With the 2026 turnout as the baseline rate, the projected 2030 constituency
totals equal the 2026 valid votes exactly.

A side effect of the fixed-seat stand-in: the frozen `fast_allocator` labels
a 2030 dispatch's fixed-seat configuration "2026", because the arrays are
equal. That is expected and is not a 2026 target.

## Known limitation

Dynamics are capped at 112 days, with no √h scaling. A forecast made years
before the election therefore carries only 112 days of movement uncertainty.
The website says so ("Mer än 112 dagar före ett val …"). This is unchanged
from how the 2022–2026 history was built.
