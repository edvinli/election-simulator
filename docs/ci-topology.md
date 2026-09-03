# Test and CI topology

Five layers. Each one exists because it answers a question the others cannot
answer quickly enough, and each one states what it is allowed to skip.

The rule that holds the whole thing together: **no layer is the only cover for
a property a change can break.** Deferring work to a slower layer is allowed;
deferring the only check of a property is not. Where a cheap layer runs a
reduced form of an expensive check, the reduced form keeps the same assertions
and only shrinks the sample.

## Layer 1 - Developer loop (seconds to ~2 min)

```sh
make check              # compile + selector tests + the tests your diff affects
make test-affected      # just the affected tests
make test-shard-plan     # see how the full suite splits
```

`make check` byte-compiles `scripts` and `tests`, runs the CI selector's own
tests, then runs the tests your change actually affects -- including
uncommitted and untracked work, because a test that a staged edit breaks should
fail before the push and not after it.

Not here: the exhaustive 20,000-case allocator audit, and any 100,000-draw
production simulation. The allocator is still covered, at
`ADVERSARIAL_CASES=700`.

## Layer 2 - Pull request (`.github/workflows/pr.yml`, a few minutes)

Triggers on `pull_request` and `merge_group` only. The workflow this replaced
also triggered on `push` to every branch, so a branch with an open pull request
ran the entire suite twice for the same commit; that duplication is gone.

1. `static` - byte-compile, and run `tests.test_ci_topology`.
2. `plan` - ask `scripts/ci/test_topology.py` which modules the diff affects.
3. `unit` - run them across 2 shards, with
   `ELECTIONSIM_ADVERSARIAL_CASES=700` so that
   `test_adversarial_mandates` -- which *is* in this matrix whenever the diff
   selects it -- audits allocator parity at the reduced size.
4. `required` - one status for branch protection.

Selection walks the **import graph**, transitively, rather than a hand-written
path table: a table goes stale silently, and its failure mode is a test that
quietly stopped running. Three gaps are handled explicitly rather than hoped
away, and all three fail toward running more:

- Seven modules import nothing from `scripts` and assert against tracked
  artifacts instead. The graph can never reach them, so they always run
  (`ALWAYS_RUN`, about seven seconds in total).
- Changes to `uv.lock`, `pyproject.toml`, `Makefile`, `data/`, `files/`, test
  fixtures, the selector itself or the workflows escalate to the whole suite.
- Any path the rules do not recognise escalates to the whole suite.

Point branch protection at **`PR required checks`**, not at the matrix jobs:
their names depend on the selection, so a required check named after a shard
would block whenever that shard is not selected. That job also fails when
`unit` is skipped for any reason other than a provably empty selection, so a
broken matrix cannot read as success.

## Layer 3 - Full CI (`.github/workflows/full.yml`)

Every module, across 4 shards balanced by measured duration
(`scripts/ci/timings.json`). Runs on push to `main`.

"Every" is literal. No module is held back for nightly -- see *Nightly is a
re-run, never a substitute* below.

Sharding by measured cost rather than by name or file count matters: one module
was 70% of the suite's wall-clock, so a naive split leaves three runners idle
while the fourth carries everything.

- **Required:** the 4 unit shards.
- **Informational** (`continue-on-error`): `cross-repo-coverage`, which records
  that `tests.test_actual_browser_consumer` skips here because there is no
  `edvinli.github.io` checkout. That test is *enforced* by the publication
  gate, which does have one. The job exists so the skip is recorded rather than
  passing unnoticed as if it were coverage.

## Layer 4 - Publication gate (`.github/workflows/election-simulator-publication.yml`)

**Unchanged by the topology work, deliberately.** It remains strict and
fail-closed:

- the real 100,000-draw production simulation (`PRODUCTION_SAMPLES`);
- history and projection generation;
- simulator to website sync validation;
- required browser acceptance (`forecast-timeseries.smoke.mjs` and
  `government-builder.smoke.mjs`) against the freshly built site;
- the backend and history contract suites, run directly and by name;
- no baseline-failure tolerance of any kind.

The modules the gate names -- `test_publication_pipeline`,
`test_prospective_archive`, `test_forecast_history_contract`,
`test_forecast_history`, `test_site_publisher`, `test_production_freeze`,
`test_publication_freeze`, `test_election_automation` -- never included
`test_adversarial_mandates`, so moving the exhaustive audit to nightly does not
touch what the gate proves.

## Layer 5 - Nightly (`.github/workflows/nightly.yml`, 02:40 UTC)

### Nightly is a re-run, never a substitute

Nightly may run a module *harder*. It may never be the only place a module runs.

This rule is written down because the first version of this topology broke it.
`test_adversarial_mandates` was declared `NIGHTLY_ONLY` and removed from the
per-change suite, while `pr.yml` and `full.yml` went on setting
`ELECTIONSIM_ADVERSARIAL_CASES=700` for a module their matrices no longer
contained. The allocator parity audit ran nowhere on a pull request, and every
job reported success. The topology test of the time checked that the case count
was large enough -- never that the module was present -- so it green-lit the
configuration.

`NIGHTLY_EXHAUSTIVE` now names modules that run in **both** places: at the
reduced size on every change, and again on the schedule without the override.
`tests.test_ci_topology` asserts the module is in the suite, that an allocator
change selects it, that it lands in a shard of the full matrix, and that nightly
invokes it by name with no override.

- **Required:** the exhaustive 20,000-case allocator audit; the expensive
  scientific and parity suites.
- **Informational:** `whole-suite-serial`, which runs `unittest discover` in a
  single process. Sharding means no other job imports the whole suite into one
  interpreter, so this is the only place that still sees cross-module state
  leakage and order dependence.

## The allocator audit split

`tests/test_adversarial_mandates.py` compares the fast vectorised allocator
against the exact legal reference. At 20,000 cases it takes about nine minutes
-- roughly 70% of the entire unit suite.

`ELECTIONSIM_ADVERSARIAL_CASES` lowers the case count. The per-change layers
run the module at 700 cases (about 17 seconds); nightly runs the same module
again at the full 20,000. The module is in the per-change suite either way --
the knob changes its size, never whether it runs. Every assertion is written
against the configured count, so the reduced run is a smaller audit of the same
kind, not a weaker one:

- the production dispatcher must match the exact legal reference on **100%** of
  cases;
- every allocation must sum to 349 seats;
- every input must be distinct;
- **every legal branch must still be reached** -- awarded cutoff tie, local 12%
  rule, overhang, multi-return. At 700 cases these are hit 86, 129, 419 and 434
  times respectively.

Two guards keep the knob from becoming a way to weaken the audit:

- values below 350 are **rejected**, because the generators cycle the branch
  selector modulo 7 and the fixed-seat map modulo 5, and a shorter run could
  silently stop covering them; values above 20,000 are rejected too;
- the test **refuses to write** the tracked audit evidence artifact from a
  reduced run, and `scripts/simulator/freeze_audit.py` drops
  `ELECTIONSIM_ADVERSARIAL_CASES` from the environment it passes down, so the
  freeze audit cannot inherit a reduced count from the shell and record it as
  the full audit.

## Baseline and drift failures

Eight modules failed on `main` before this work, none of them because of a
product regression, all of them because a test read live state it never meant
to depend on. They are fixed rather than quarantined; nothing is skipped and no
expectation was relaxed.

| Cause | Modules | Fix |
| --- | --- | --- |
| Freeze verifiers reported the refreshed polling snapshot as evaluator drift | `test_challenger_freeze`, `test_evaluator_freeze_reconstructible`, `test_control_baseline_amendment2`, `test_production_challenger_b` | Judge drift per freeze **group** (`tests/_freeze_drift.py`): code drift stays strictly enumerated, a refreshed `truth_input` is expected |
| `build_history` folded the repository's growing forecast archive into the observation dates | `test_forecast_history`, `test_forecast_history_contract` | Pass `archive_dir=None`; archive substitution is still covered by injecting `archived_points` |
| Hard-coded row count and end date of a growing series | `test_pollofpolls_party_chart_extension` | Pin the start date, bound the end date forward-only and forbid future dates; the count is already implied by contiguity |
| Origin pinned to a literal "last date in the dataset" that the data outgrew | `test_backtest` | Read the last observation from the dataset |

Widening each module's `KNOWN_POST_FREEZE_CHANGES` to include the CSV would
have been the smaller change and the wrong one: those sets match on filename
with no notion of group, so the polling file would have been excused as *code*
drift too.

### The underlying defect, not worked around

`poll_id` in `data/processed/pollofpolls/swedishpolls_individual_polls.csv` is a
content hash that includes each row's upstream `source_row` index. When
upstream inserts a poll, every row is renumbered, every id is regenerated, and
all ~26,400 rows change even though no poll's numbers did. Commit `34c52d6`
("Stabilize processed polling outputs across semantically identical refreshes")
attacked this and it is still present.

This is left alone on purpose: `poll_id` reaches published payloads and
provenance hashes, so changing its derivation is a publication-integrity
decision and not a CI clean-up. The freeze tests now describe the churn
accurately instead of failing on it.

## Refreshing the timings table

`scripts/ci/timings.json` only balances shards; it is not a performance budget
and does not need routine updates.

One entry is deliberately not the module's default cost: sharding happens only
in the layers that set `ELECTIONSIM_ADVERSARIAL_CASES=700`, so
`test_adversarial_mandates` is recorded at that reduced cost (17s), not the
exhaustive 511s. Recording the exhaustive figure would make the planner hand
the module its own shard and leave the other three nearly idle. The exhaustive
number is kept alongside, under `exhaustive_seconds`, for reference only;
nightly runs it as a single unsharded job where no balancing is needed.

Refresh the table when a suite's cost changes materially:

```sh
for t in tests/test_*.py; do
  m=$(basename "$t" .py)
  /usr/bin/time -f "%e $m" uv run python -m unittest "tests.$m"
done
```

A module missing from the table is treated as average cost, so adding a test
module does not break sharding -- it just balances slightly less well until the
table is refreshed.
