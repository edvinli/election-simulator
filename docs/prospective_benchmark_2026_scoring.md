# Prospective 2026 scoring contract

This note defines the scoring helpers in
`scripts/prospective_benchmark_2026/scoring.py`. It is a scoring
implementation note, not the capture protocol; the protocol remains the
authoritative pre-registration for the campaign.

## Vote-share target and party order

The target is the final certified national valid-vote share, in percentage
points. The eight-party order is fixed as

`M, L, C, KD, S, V, MP, SD`.

The eight values are scored directly on that official denominator. They are
not renormalized to sum to 100%; omitted non-parliamentary parties remain
omitted from this eight-party comparison.

## Fair finite-ensemble scores

For predictive draws \(x_1,\ldots,x_n\) and realized value \(y\), the fair
CRPS estimator is

\[
 \widehat{\operatorname{CRPS}}_{\mathrm{fair}}
 = {1\over n}\sum_i |x_i-y|
 - {1\over 2n(n-1)}\sum_{i\ne j}|x_i-x_j|,
 \qquad n\ge2.
\]

The fair multivariate Energy Score uses the same correction:

\[
 \widehat{\operatorname{ES}}_{\mathrm{fair}}
 = {1\over n}\sum_i \lVert x_i-y\rVert_2
 - {1\over 2n(n-1)}\sum_{i\ne j}\lVert x_i-x_j\rVert_2.
\]

These are U-statistics over distinct draw pairs. The correction is important
when ElectionSimulator and Botten Ada have different ensemble sizes: the
finite-sample bias from zero self-pairs is not allowed to favor the larger
ensemble. A one-draw ensemble is therefore rejected for fair scoring rather
than silently treated as a point mass.

The existing V-statistics are retained as named sensitivity outputs:

\[
 {1\over n}\sum_i|x_i-y| - {1\over 2n^2}\sum_{i,j}|x_i-x_j|
\]

and its multivariate norm analogue. V-statistic values must not replace the
fair primary score in the prospective comparison.

The univariate pair sum is evaluated exactly after sorting. Exact multivariate
pair sums are chunked but remain quadratic. For a 100,000-draw ensemble, a
caller may explicitly request uniform ordered distinct-pair Monte Carlo via
`pair_sample_size` and record its seed and count in the capture/report. This
is an unbiased estimator of the declared U-statistic, not an unannounced
change of score.

## Secondary metrics

`threshold_brier` uses the inclusive event `share >= 4.0`. The probability may
come from verified draws or from an explicitly published event probability;
missing probabilities are errors/status values, never zeroes.

Point MAE accepts only explicitly published central predictions. It never
turns a mean, median, interval, or `3.1 ± 2.3` display into an assumed law.
Central interval coverage and width can be computed from verified draws or
from explicitly published lower/upper bounds.

WIS is available only when both models publish a median and at least one
compatible central interval from the pre-registered 50%, 80%, 90%, and 95%
levels. The scorer uses all and only the common levels (and requires each
level for every scored party). No quantile interpolation or joint
distribution reconstruction is performed.

Per amendment 002, the weighted sum uses the standard WIS normalization
`K + 1/2`; the earlier weighted-average denominator is not used.

## Fallback hierarchy

`select_primary_scoring_tier` encodes the preregistered hierarchy:

1. both models have independently verified predictive draws → fair CRPS (and
   fair Energy Score as the joint secondary metric);
2. otherwise, both models publish a median plus at least one common interval
   level from 50%, 80%, 90%, and 95% → WIS over that intersection;
3. otherwise → no probabilistic winner; report MAE only when both explicit
   central forecasts exist.

The draw verification flags are supplied by the capture/provenance layer.
An RDS object, posterior draw table, or locally rerun model does not become a
verified Botten Ada predictive ensemble merely because it contains 1,000
rows.
