# 2026 prospective ElectionSimulator–Botten Ada benchmark protocol

Status: frozen before the first scored capture. Created 2026-09-03 at 19:06:02 UTC.

The machine-readable protocol in `data/processed/prospective_benchmark_2026/protocol.json` is normative. Its exact byte hash is stored in `protocol.sha256`, copied into the archive index, and required in every capture manifest. This document explains the same contract in human terms.

## Question and design

The primary question is: **Which forecasting system, as actually published during the 2026 Swedish election campaign, performed better?** It is not the retrospective question of which mathematical model would win if both were rerun with identical inputs.

The experiment therefore uses a common public wall-clock cutoff, not a forced common internal `as_of`. At every scheduled cutoff it separately records ElectionSimulator's model `as_of` and latest poll input date, and Botten Ada's source update time and latest poll date. Forecast quality and source freshness can then be described separately without pretending that the two public systems had the same information set. The existing historical Botten Ada benchmark remains the appropriate home for a controlled identical-information-set experiment.

## Schedule and immutable slots

There are nine primary campaign slots: 2026-09-04 through 2026-09-12 inclusive at 23:30 `Europe/Stockholm` (21:30 UTC during this window). The 2026-09-12 slot is the primary final forecast. There is no election-morning primary snapshot.

One and only one durable capture may occupy each scheduled date. A real capture cannot be created before its cutoff or retroactively. A job that starts after the scheduled Stockholm calendar date is durably recorded as late and excluded; it is never silently relabelled on-time. Dry runs are never indexed. Failures such as `SOURCE_UNAVAILABLE`, `PARSE_FAILED`, `SOURCE_STALE`, or `PARITY_UNVERIFIED` occupy their slot and cannot be replaced. Retries are allowed only before any durable capture has been indexed.

Identical content on consecutive days is legitimate evidence that a system had not changed. The capture process must retrieve and hash the source each day; it may never copy yesterday's record into today's slot.

## ElectionSimulator selection and exact draws

For each cutoff, the benchmark selects the latest certified production generation that both has `generated_at_utc <= cutoff` and was durably committed by the cutoff. The existing prospective archive, its index, the snapshot hash, source commit, input hashes, deterministic payload hash, model version, seed, and exact 100,000-simulation count must validate. Multiple same-day generations are resolved mechanically by generation timestamp and ID; performance never enters selection.

Eligible ElectionSimulator draws must be the exact joint draws exported from the same `SimulationResult` used by production and cryptographically tied to the certified generation. A benchmark-specific rerun or changed configuration is prohibited. The sidecar is checked against the immutable production summaries and deterministic payload hash.

## Botten Ada evidence and verified draws

The evidence hierarchy is: (1) official contemporaneous predictive draws whose parity with the public forecast is verified; (2) official machine-readable point estimates, probabilities, quantiles, and intervals; then (3) official webpage values with the raw response retained. Each raw artifact records its URL, UTC retrieval time, SHA-256, byte size, HTTP validators when available, source update time, extraction version, relevant upstream commit when identifiable, and licensing/attribution.

“Verified Ada draws” means that the official object is identified as election-day predictive draws and reproduces every simultaneously published central forecast, compatible interval endpoint, and threshold probability within the display-rounding tolerances frozen in the machine protocol. A downloadable R object containing 1,000 posterior samples is not automatically the site's 1,000 election simulations. If parity fails or semantics cannot be established, the object is retained as evidence but its samples are not scored as predictive draws.

The benchmark never creates draws from intervals, interprets unexplained plus/minus values as Gaussian, reconstructs a joint distribution from marginal quantiles, or substitutes a local Ada reimplementation for Botten Ada's publication.

## Scoring and unequal ensembles

The primary final-forecast result is available only if both systems have verified predictive vote-share draws. It is the lower arithmetic mean of party-level **fair finite-ensemble CRPS** over `M, L, C, KD, S, V, MP, SD`, in percentage points of official national valid votes. The eight parties are not renormalized to 100%.

For ensemble members \(x_1,\ldots,x_n\) and result \(y\), fair CRPS is

\[
\frac{1}{n}\sum_i |x_i-y| - \frac{1}{2n(n-1)}\sum_{i\ne j}|x_i-x_j|.
\]

The joint fair Energy Score replaces absolute values with Euclidean norms and uses the same U-statistic denominator. Removing the self-pairs corrects the ensemble-size-dependent downward bias of the ordinary empirical/V-statistic pair term, so 100,000 ElectionSimulator members and roughly 1,000 Ada members can be compared without requiring equal counts. The old V-statistic is retained only as an explicitly labelled sensitivity result.

Secondary metrics are per-party fair CRPS, joint fair Energy Score, MAE of each system's actual published central prediction, compatible central interval coverage/width, and 4% threshold Brier scores. Seats are scored only if both systems publish genuine compatible predictive seat distributions. Government or majority probabilities are scored only if the events are proven identical.

## Frozen winner and fallback hierarchy

For the 2026-09-12 final capture:

1. If both systems have verified predictive draws, the probabilistic winner is lower mean fair CRPS.
2. Otherwise, if both publish a median and at least one compatible central interval among 50%, 80%, 90%, and 95%, the probabilistic winner is lower mean weighted interval score using all and only their common levels. The exact WIS formula is frozen in `protocol.json`.
3. Otherwise no probabilistic winner is declared. The point-forecast comparison is eight-party MAE using each model's identified published central predictions.

The campaign score applies the same hierarchy to paired eligible daily captures and gives every scheduled date equal weight. A model receives no extra weight for additional intraday updates. Reports include every excluded date and reason, the equal-weight mean, per-date scores, and daily win counts.

The inclusive threshold event is official vote share `>= 4.0%`. `L`, `C`, `KD`, and `MP` are pre-registered; `L` must be discussed explicitly when comparable probabilities exist. Missing forecasts never become zeros.

## Results, reporting, and limitations

Scoring requires the final certified Valmyndigheten result—not election-night or preliminary totals—with the official source, retrieval time, raw hash, valid-vote denominator, party votes/shares, final seats, and certification status. Adding result evidence never modifies prospective captures.

The generated report must distinguish a probabilistic winner from a point-forecast winner, quantify score differences, list missingness and source freshness, and avoid turning a tiny difference into a claim of universal model superiority. The command is:

```bash
python -m scripts.prospective_benchmark_2026 score --results <official-result-file>
```

## Amendments and attribution

The original protocol is never rewritten. An unavoidable change requires a sequentially numbered, separately timestamped and hashed amendment that states the reason and whether primary scoring changes.

Archived Botten Ada material remains attributable to Botten Ada and its authors. Captures retain upstream URLs and discovered license metadata. Repository inclusion is limited to the evidence needed for independent audit; an artifact whose license does not permit redistribution must be represented by metadata and a cryptographic hash rather than copied beyond what the license permits.
