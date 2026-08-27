# Threshold-loss attribution (diagnostic)

This page documents the final, bounded research cycle for the inclusive
national 4% threshold.  It is retrospective evidence only.  It does not
change ElectionSimulator v1.0-rc1, PoPBaseline v1, the mandate allocator, or
the publication contract.

## What is evaluated

`scripts/pop_baseline/threshold_attribution.py` runs the six predeclared
variants below on exact 2018/2022 origin dates at the six standard horizons.
Every scored row has the same origin, target, horizon, outcome, sample count,
and seed across all variants.

| Variant | Start state | Dynamics | Election residual | Support rule |
| --- | --- | --- | --- | --- |
| A | stored PoP point | PoPBaseline raw CLR paths | none | PoPBaseline support transfer |
| B | OpinionState draws | frozen RC1 dynamics | `pp_centered_noise` | none |
| C | deterministic OpinionState mean | B's exact dynamics | B's exact centered residual draws | none |
| D | B's exact state draws | B's exact dynamics | removed | none |
| E | B's exact state draws | PoPBaseline raw CLR paths | B's exact centered residual draws | none |
| F | stored PoP point | A's exact raw CLR paths | none | disabled |

The harness uses the deterministic OpinionState mean for C, not a substitute
stored point estimate.  It also records an explicit center-preservation check.
If any layer cannot be isolated from an existing implementation surface, the
variant is marked `NOT_RUN` and the case is not scored.

## Threshold evidence

`threshold_metrics.py` expands each scored case into one row per model and
threshold party with:

* election and horizon;
* party;
* forecast probability of `share >= 4.0%`;
* fixed probability bin;
* actual pass/fail outcome;
* row-level Brier score.

Reliability tables are descriptive fixed-bin summaries.  No bin merging,
smoothing, fitting, or retrospective calibration is performed.  In a
multi-seed run, probabilities are first averaged within the exact
case/model/party key so the same election outcome is not counted once per
seed.

The existing 2002–2022 final-poll consensus study is also validated for all
six elections.  Its 3–5% table is a deterministic poll-to-election residual
diagnostic, not a probabilistic Brier benchmark.  The report explicitly marks
probabilistic evaluation `NOT_RUN` where exact historical PoP origins and
Candidate-A pools do not exist.

## Frozen diagnostic results

All results in this section are retrospective comparative diagnostics, not
holdout validation and not production forecasts.  The machine-readable
outputs are in [`precision_election_v1`](../data/processed/pop_baseline_benchmark/precision_election_v1/),
[`precision_rolling_slice_v1`](../data/processed/pop_baseline_benchmark/precision_rolling_slice_v1/),
and [`threshold_attribution_v1`](../data/processed/pop_baseline_benchmark/threshold_attribution_v1/).

The higher-precision A/B election run used three fixed seeds and 5,000 draws
per model/case (12 exact 2018/2022 election-origin cases).  Scores are lower
when better; seed variation is shown as a stability diagnostic, not an
inferential interval.

| Model | Vote CRPS (8) | Energy Score (9) | Threshold Brier (8) | Median vote MAE (8) |
| --- | ---: | ---: | ---: | ---: |
| RC1 (`B`) | 0.802467 | 2.869141 | 0.055609 | 1.115286 |
| PoPBaseline (`A`) | 0.850086 | 3.000501 | 0.018588 | 1.054197 |
| RC1 minus PoPBaseline | -0.047619 | -0.131360 | +0.037021 | +0.061090 |

The threshold result favors PoPBaseline on these 12 retrospective cases,
while CRPS and joint Energy Score favor RC1.  The paired summary is
descriptive because seeds repeat cases and historical origins are not
independent.

The six-variant run used one fixed seed and 256 draws per case.  It is a
component diagnostic, not a model-selection exercise:

| Variant | Threshold Brier (8) | Vote CRPS (8) | Energy Score (9) |
| --- | ---: | ---: | ---: |
| A PoPBaseline | 0.017080 | 0.846535 | 2.991111 |
| B RC1 full | 0.056038 | 0.798797 | 2.861143 |
| C RC1 without OpinionState uncertainty | 0.056019 | 0.810256 | 2.900664 |
| D RC1 without `pp_centered_noise` | 0.057554 | 0.847283 | 2.915408 |
| E RC1 with PoP-style dynamics | 0.054035 | 0.804175 | 2.887457 |
| F PoPBaseline support disabled | 0.058795 | 0.907249 | 3.126103 |

The predeclared C–F gate produced `FAIL` for every candidate and therefore
`NO_CLEAR_ATTRIBUTION_STOP_KEEP_RC1`.  The gate statistics were:

| Candidate vs RC1 | Mean threshold improvement | Gate-qualified case win rate (improvement ≥ 0.005) | CRPS delta | Energy delta | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| C vs B | +0.000020 | 0.000 | +0.011459 | +0.039522 | FAIL |
| D vs B | -0.001515 | 0.333 | +0.048486 | +0.054265 | FAIL |
| E vs B | +0.002004 | 0.083 | +0.005378 | +0.026314 | FAIL |
| F vs B | -0.002757 | 0.333 | +0.108452 | +0.264960 | FAIL |

The explicit reference pairings add the missing attribution context.  Deltas
are candidate minus reference, so positive values mean that the candidate or
removed layer has the worse loss:

| Component comparison | Threshold Brier delta | CRPS delta | Energy delta | Threshold candidate wins |
| --- | ---: | ---: | ---: | ---: |
| Support transfer: F minus A | +0.041715 | +0.060713 | +0.134992 | 0/12 |
| OpinionState uncertainty: C minus B | -0.000020 | +0.011459 | +0.039522 | 8/12 |
| `pp_centered_noise`: D minus B | +0.001515 | +0.048486 | +0.054265 | 9/12 |
| Dynamics: E minus B | -0.002004 | +0.005378 | +0.026314 | 9/12 |

The A/F comparison indicates that PoPBaseline's support transfer accounts for
much of its observed 2018/2022 threshold advantage within this diagnostic.
That is not evidence that a new tactical-voting rule should be implemented:
the 3–5% historical support table contains only three observations, all pass,
and its probabilistic evaluation status is explicitly `NOT_RUN` because exact
A/B probabilities for the early elections are unavailable.  The conservative
conclusion is to keep RC1.

The rolling precision artifact is only a bounded current-data slice: three
fixed seeds, 1,000 draws, origins from 2026-07-01 through 2026-08-23 at a
7-day step, and horizons 7/14/28 days.  It has 17 scored cases per seed and
seven explicit missing-exact-date skips per seed.  It is marked `PARTIAL` and
is not the planned full 2014–2026 hardened rerun; the missing dates and the
runtime/data-availability boundary were not repaired with nearby observations.

## Adoption gate

The gate is declared before reading variant scores:

* mean threshold Brier improvement of at least 0.005;
* improvement in at least 75% of paired cases;
* mean CRPS degradation no greater than 0.01;
* mean joint Energy Score degradation no greater than 0.02.

Only one variant satisfying every criterion could become a diagnostic
candidate.  Even then this cycle does not alter production RC1.  Multiple
passing variants or no passing variant yields `NO_CLEAR_ATTRIBUTION_STOP_KEEP_RC1`.

## Reproducible commands

```text
uv run python -m scripts.pop_baseline.threshold_attribution \
  --samples 256 --seed 12345 \
  --output data/processed/pop_baseline_benchmark/threshold_attribution_v1
```

For higher Monte Carlo precision, use the non-mutating paired wrapper:

```python
from scripts.pop_baseline.paired_precision import run_paired_precision_benchmark

report = run_paired_precision_benchmark(
    seeds=(12345, 24680, 98765),
    rolling_samples=1000,
    election_samples=5000,
)
```

Seed and rolling-origin summaries are descriptive: repeated seeds reuse the
same realized case outcomes, and rolling cases are temporally dependent.  No
inferential confidence intervals are reported.

The earlier 256-draw full benchmark's 116 skips are independently explained
in `data/processed/pop_baseline_benchmark/skip_audit_v1/skip_audit.json`.
Forty-three are missing exact stored origin/target observations and 73 are
chronological Candidate-A transition-pool shortages; neither category is
repaired by substituting data.
