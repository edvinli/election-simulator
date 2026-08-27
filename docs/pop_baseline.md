# PoPBaseline v1

`PoPBaseline-v1.0` is a separate, opt-in opponent for the frozen
`ElectionSimulator v1.0-rc1` (Candidate A). It is not a poll aggregator and it
does not change Candidate A or any certified RC1 artifact.

## Reconstructed method

The baseline starts with exactly one stored Poll of Polls composition at the
requested origin. It then samples historical joint changes over 21-, 28-, and
35-day windows, reverses each sampled change with equal probability, and
combines equal-sized batches from the three windows. Changes are represented
in CLR coordinates, so the perturbation preserves a positive nine-category
composition and sums to 100 percent. No current-state uncertainty, election
residual, geography, or seats are added.

The first-party 2018 page also specifies a simple support-vote rule:

```text
s_support = s_sim + (-0.6 * abs(s_sim - 4) + 1.2) * X,  X ~ Normal(1, 0.7)
```

for `2 < s_sim < 5`, otherwise unchanged. Positive transfers are taken
proportionally from same-block parties' support above 4 percent. The
canonical data model folds FI into `REST`, so FI is not invented as a separate
recipient; the representable default recipients are L, C, KD, MP, and V.
Support voting can be disabled in the configuration for the dynamics-only
benchmark. Transfers are applied sequentially in the declared target order
(`L`, `C`, `KD`, `MP`, `V`); that order is part of the baseline version's
deterministic configuration because the source does not specify a simultaneous
multi-party reconciliation rule.

For exact 7/14/28/56/84/112-day benchmark horizons, a final historical step is
fractionalized linearly in CLR space when the horizon is not divisible by the
selected window. This is an explicit approximation because the source pages
do not specify how to handle partial final steps.

## Source record

The reproducibility record, URL list, retrieval date, paraphrased evidence, and
hashes are in
[`pop_baseline_provenance.json`](../data/raw/pollofpolls/pop_baseline_provenance.json).
The primary sources are the first-party [2018 simulation
method](https://pollofpolls.se/simulering-av-valresultat-eller-valet-kommer-inte-avgoras-pa-valdagen/),
[2022 simulation](https://pollofpolls.se/simulering-av-valresultat-infor-valet-2022/),
[2026 simulation](https://pollofpolls.se/simulering-av-valresultat-2026/), and
[method page](https://pollofpolls.se/metod/), retrieved 2026-08-27.

The 2022 post reports a later 88-day step. That later setting is intentionally
not silently substituted: this baseline is the documented three-window method
that the project set out to improve, and all approximations are explicit.

## Usage

```bash
uv run python -m scripts.pop_baseline --origin 2022-08-14 --horizon 28 --samples 5000 --seed 12345
```

The module returns deterministic seeded draws and machine-readable diagnostics.
It requires an exact date in the processed PoP series when `--origin-pop` is
not supplied; nearby dates are never substituted silently.

## Matched benchmark

The separate `scripts.pop_baseline.benchmark` harness writes
`benchmark_report.json` and `benchmark_case_metrics.csv` under
`data/processed/pop_baseline_benchmark/`. It uses the same exact stored origin,
target date, horizon, sample count, seed, and outcome for the baseline and
frozen RC1 dynamics. The rolling component skips missing exact rows and origins
with fewer than the Candidate A minimum transition pool; it never imputes a
nearby date. Election 2018/2022 results are retrospective comparative evidence,
not independent holdout validation. Seat metrics are explicitly unavailable
because PoPBaseline does not fabricate geographic or seat draws.

The current checked-in run uses 256 draws per case and 3,634 scored cases out
of 3,750; the remainder are explicit early-history/end-of-series skips. This is
benchmark evidence for the harness and should not be presented as a universal
winner claim. The report's proper-score tables and implementation/input hashes
are the source of any later model decision.
