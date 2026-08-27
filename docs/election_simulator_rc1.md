# ElectionSimulator v1.0-rc1 certification and prospective evidence

`ElectionSimulator v1.0-rc1` is the frozen Candidate A model. The statistical
model, polling calibration, geographic projection, turnout assumptions,
election residuals, strategic-voting treatment, and mandate rules are not
changed by the comparative benchmark.

## Two-commit provenance

Release certification uses two commits:

1. a clean source-code commit containing implementation, tests, fixtures,
   benchmark code, and documentation;
2. a separate artifact commit containing forecasts, audits, hindcasts, and
   prospective snapshots generated from the first commit.

Every generated artifact records `source_git_commit` and
`source_worktree_clean=true`. The cleanliness flag covers source, tests,
configuration, and input data; only the explicitly generated certification,
hindcast, prospective-archive, and benchmark output directories are ignored
for this flag. It is intentionally impossible for an artifact
to contain the hash of the later commit that stores that artifact.

The historical report claiming L 3.56%, S 31.25%, and a 31.8% Tidö majority
has no reproducible source artifact or command log. It is marked
`UNREPRODUCIBLE_INVALID` and is not a canonical forecast.

## Daily prospective archive

The archive is append-only by `as_of` date. A collision, duplicate
deterministic payload, or duplicate identity fails closed; existing snapshots
are never overwritten. Snapshots contain compact quantiles and histograms,
not the 100,000 raw draws, and link to the canonical forecast through both its
file SHA-256 and deterministic payload SHA-256.

Run one snapshot after producing the canonical forecast:

```text
uv run python -m scripts.prospective_archive \
  --as-of 2026-08-23 \
  --election-date 2026-09-13 \
  --samples 100000 --seed 12345
```

The command writes `data/processed/prospective_forecasts/<as_of>/snapshot.json`
and updates `index.json`. Do not rerun the same `as_of` date into the same
archive; the command is designed to refuse that collision. Preserve the
poll-data hash, source commit, model/config hashes, and timestamp from every
run.

`REST` is an aggregate residual category for otherwise modeled-as-ineligible
parties. It is never independently eligible for the 4% threshold or seats.

## Botten Ada comparison

The benchmark harness treats this model as Candidate A and does not mutate it.
Botten Ada is pinned to the public `MansMeg/ada_code` repository commit
`2dfe246b86c5cab517e4a0cb87fd57e5a9c62512`, with source and data URLs recorded
in the benchmark output. The current repository does not contain an
independently generated Botten Ada predictive-draw bundle, so the default
benchmark status is `NOT_RUN`. No point forecast is converted into an invented
distribution.

An external adapter bundle must use the standardized schema and fixed party
order. It must provide identical election date, as-of cutoff, horizon, and
sample count for both candidates. Retrospective 2018/2022 scores are
comparative historical evidence, not holdout validation. The harness computes:

- vote CRPS by party and macro-average;
- joint vote Energy Score;
- party-specific inclusive 4% threshold Brier score;
- mean and median vote-share MAE;
- central predictive coverage and interval width;
- seat CRPS and joint seat Energy Score when independently generated seat
  draws and realized seats are available.

The pre-registered pivot rule is stored in the machine-readable report. A tie
or Candidate A win keeps Candidate A unchanged. Only a consistent, material
Candidate B advantage at late horizons and on threshold scores permits
targeted layer investigation; it does not authorize automatic adoption.

Generate a Candidate A exchange bundle (raw draws are for benchmark exchange,
not the compact prospective archive) with:

```text
uv run python -m scripts.botten_ada_benchmark.candidate_a \
  --as-of 2026-08-23 --output /tmp/candidate_a.json
```

Then compare it with an independently produced Botten Ada bundle:

```text
uv run python -m scripts.botten_ada_benchmark \
  --candidate-a /tmp/candidate_a.json \
  --candidate-b /path/to/botten_ada_bundle.json \
  --output data/processed/botten_ada_benchmark/benchmark_report.json
```
