# 2026 prospective benchmark operations

## Scope

This infrastructure answers one prospective question: **which forecasting system, as actually published during the 2026 Swedish campaign, performed better?** It does not answer which mathematical model would win if both were rerun later on an identical information set. The existing historical Botten Ada harness remains separate for that controlled question.

The normative rules are the immutable machine protocol in `data/processed/prospective_benchmark_2026/protocol.json`, its SHA-256 sidecar, and the numbered amendments. Every capture manifest binds the original protocol hash and the full amendment catalog. The original protocol is never rewritten.

## Repository audit and architecture

The pre-existing ElectionSimulator prospective archive is immutable and content-hashed, accepts multiple generations on one calendar day, and records generation ID, UTC generation time, `as_of`, source commit, worktree state, model version, seed, sample count, input hashes, and deterministic payload hash. Scheduled publication produces a daily generation even when poll inputs are unchanged. Production is `1.1.0-rc1`, uses the adopted `pp_lw_gaussian` election-noise model, seed 12345, and 100,000 simulations. Existing snapshots intentionally store compact marginal distributions rather than the full joint matrices.

The historical `scripts/botten_ada_benchmark` package expects an independently supplied external bundle, has not completed a contemporaneous Botten Ada head-to-head, requires equal draw counts in its common-case validator, and uses finite-sample V-statistics. It is not reused as the prospective evidence archive.

The new system has four boundaries:

1. `scripts/simulator/exact_draw_sidecar.py` mechanically selects the latest certified ElectionSimulator generation at or before the cutoff and validates or exactly reproduces its joint draws under amendment 003.
2. `scripts/prospective_benchmark_2026/botten_ada_capture.py` retrieves only pinned official Botten Ada artifacts and archives their exact bytes and transport/source metadata.
3. `scripts/prospective_benchmark_2026/archive.py` atomically installs one immutable capture per scheduled slot and appends a hash-chained index entry only after complete validation.
4. `scripts/prospective_benchmark_2026/scoring.py` and `report.py` apply the frozen evidence hierarchy and generate machine-readable and Markdown reports without a hard-coded winner.

No component writes to or deploys the public website.

## Schedule and slot selection

The nine scheduled cutoffs are 2026-09-04 through 2026-09-12 at 23:30 `Europe/Stockholm`. They are 21:30 UTC in this window. The final primary forecast is the 2026-09-12 slot; there is no primary election-morning capture.

The workflow uses GitHub's externally recorded workflow-run creation timestamp, converts it through the Stockholm time zone, and shares the `election-simulator-production` concurrency group with publication. The selected ElectionSimulator forecast is the latest certified generation whose generation, source, archive-index, and Git-availability evidence all precede the cutoff. Intraday update frequency cannot add campaign weight.

A durable capture before cutoff is rejected. A run retrieved on the immediately following Stockholm date is stored as `LATE_EXCLUDED`; a later attempt is retroactive and cannot write. A failed source occupies its slot. Retries can only occur before an index entry exists.

## ElectionSimulator exact-draw evidence

A sidecar from the same production `SimulationResult` is preferred. Because the existing publication boundary did not retain that in-process object, amendment 003 permits a narrow `REPLAY_VERIFIED` path. Replay must prove the exact certified snapshot and index row existed by cutoff; source ancestry; a clean checkout; unchanged simulator code, inputs, lockfile, configuration, version, seed, and hashes; and parity with the deterministic payload plus every published vote, threshold, seat, and coalition summary. Discovery repeats the parity checks. Any failed check leaves the draws unverified.

Only the nine mechanically selected generations receive sidecars. At approximately 7.3–7.4 MiB per compressed 100,000-draw sidecar, this avoids changing the production archive or retaining every intraday matrix.

## Botten Ada evidence and parity finding

The pinned official sources are the Botten Ada [site](https://www.bottenada.se/), [data page](https://www.bottenada.se/data), [FAQ](https://www.bottenada.se/faq), official S3 JSON/CSV endpoints, the downloadable `pop.rds`, and the public [`ada_code`](https://github.com/MansMeg/ada_code) repository. Current machine-readable evidence includes:

- `latest_forecast/seats--all.json`, containing election-day p5/p50/p95 vote and seat forecasts plus run metadata;
- `latest_forecast/latest_polls--all.json`;
- `latest_pop/timeseries.csv`, archived but kept separate from the election forecast;
- official `is_{party}_above_4_pct` JSON for L, C, KD, and MP;
- homepage/data/FAQ HTML where it supplies semantic evidence;
- a metadata-only HEAD capture for the very large `pop.rds` object.

The public data page describes the R object as containing model/data/Stan output and 1,000 posterior draws, while the FAQ describes 1,000 simulated elections. Those statements do not establish that the RDS posterior samples are the election-day predictive simulations displayed by the website. The live deployment also reports model `8m10`, while the pinned public code configuration is not sufficient to reproduce that deployment. Consequently current captures use model/source status `AVAILABLE` while the nested Ada draw status remains `PARITY_UNVERIFIED`; they preserve published medians, p5/p95 intervals, threshold probabilities, and raw evidence. The forecast and four threshold objects must share the same `run`, `model`, and `run_written` identity, and each decision-bearing object's HTTP `Last-Modified` must be at or before the benchmark cutoff. Mixed or post-cutoff evidence is retained raw but removed from scoring. No Gaussian, marginal-joint, or quantile-derived draws are created.

Future Ada draws may become `VERIFIED` only if exact official object bytes, pinned semantic evidence bytes, extraction metadata, canonical draw-matrix hashes, and simultaneous public-value parity all validate. Frozen tolerances are 0.051 percentage points for displayed vote quantiles and 0.0051 on the probability scale for threshold probabilities. The normal capture path does not currently elevate any Ada object to verified-draw status.

Archived Botten Ada material is attributed to Botten Ada and its authors. The site data page reports CC BY-NC-SA 4.0 for its data/material; the public code repository reports MIT for code. Large or redistribution-sensitive artifacts are represented by contemporaneous headers, hashes, sizes, and URLs rather than copied into Git.

## Frozen scoring contract

For draws \(x_1,\ldots,x_n\) and outcome \(y\), fair CRPS is

\[
\frac{1}{n}\sum_i |x_i-y|-
\frac{1}{2n(n-1)}\sum_{i\ne j}|x_i-x_j|.
\]

Joint fair Energy Score replaces absolute differences with Euclidean norms across `M,L,C,KD,S,V,MP,SD`. The distinct-pair U-statistic removes the self-pair bias that otherwise depends on ensemble size, so 100,000 and roughly 1,000 verified members can be compared without equalizing or resampling them. Energy Score uses the amendment-001 deterministic Monte Carlo estimator with 1,000,000 uniformly sampled ordered distinct pairs and seed 20260903. The ordinary V-statistics remain labelled sensitivity metrics only.

All shares are percentage points on the official national valid-vote denominator. The eight parties are never renormalized to 100%. Point MAE uses each system's actually published central prediction: ElectionSimulator's headline p50 and Botten Ada's p50. ES means are retained only as supplementary metadata. Threshold Brier events are inclusive `share >= 4.0%` for L, C, KD, and MP; additional archived ES party probabilities are preserved but are not scored. Central interval coverage/width is reported where meaningful. Seat and government-event scores are omitted unless both predictive objects and event semantics are genuinely compatible.

The final and per-date hierarchy is:

1. If both systems have verified election-day predictive draws, lower mean fair CRPS over the eight fixed parties is the probabilistic winner.
2. Otherwise, if both publish a median and at least one complete common central interval among 50%, 80%, 90%, and 95%, lower standard WIS over all common levels is the probabilistic winner.
3. Otherwise no probabilistic winner is declared; lower eight-party point MAE is reported as the point-forecast comparison only.

Campaign aggregates give each paired eligible scheduled date one equal weight. Losses from different fallback tiers are not averaged into a synthetic campaign winner.

## Failure and freshness policy

`SOURCE_UNAVAILABLE`, `PARSE_FAILED`, and `SOURCE_STALE` are model/source evidence, not reasons to substitute another date. `PARITY_UNVERIFIED` is separately retained as draw evidence and never prevents compatible published intervals from being represented as available. Exact raw bytes and diagnostics are retained when available. Yesterday's Ada forecast is never copied into today's slot. An unchanged official response may legitimately recur on consecutive dates and is freshly retrieved and re-hashed each time.

No arbitrary age threshold is applied. `SOURCE_STALE` is used only when the official source marks the forecast obsolete or the payload is not a current 2026-election forecast. Source-reported timestamps, latest poll dates, HTTP validators, and unchanged hashes remain visible so freshness can be analyzed separately from forecast loss.

## Commands

Validate the frozen empty/current archive:

```bash
uv run python -m scripts.prospective_benchmark_2026 validate
```

Manual non-durable rehearsal for a scheduled slot (safe before cutoff):

```bash
uv run python -m scripts.prospective_benchmark_2026 capture \
  --mode dry_run \
  --scheduled-date 2026-09-04
```

First real capture, run only at or after 23:30 Europe/Stockholm on 2026-09-04:

```bash
uv run python -m scripts.prospective_benchmark_2026 capture \
  --mode capture \
  --scheduled-date 2026-09-04
uv run python -m scripts.prospective_benchmark_2026 validate
```

The preferred operational route is **Actions → Prospective benchmark 2026 capture → Run workflow**, with `mode=capture` and the exact scheduled date. A real run commits only the capture directory and `index.json` as `github-actions[bot]`.

After final certification, prepare a normalized result manifest with schema `1.0`, authority `Valmyndigheten`, election date `2026-09-13`, status `FINAL_CERTIFIED`, an exact `https://resultat.val.se/...` source, UTC retrieval time, a safe relative raw-file path and SHA-256, national valid-vote denominator, and exact votes/share/seats for all eight parties. Shares must equal `100 * party_votes / valid_national_votes`; other-party votes remain in the denominator. Commit the raw artifact, normalized manifest, and generated report together so the reported manifest SHA-256 and raw SHA-256 are independently auditable.

Generate both reports with:

```bash
uv run python -m scripts.prospective_benchmark_2026 score \
  --results <path-to-final-certified-result-manifest.json>
```

Outputs are `diagnostics/prospective_benchmark_2026/final_report.json` and `final_report.md`.

## Before the first capture

1. Merge this branch to `main` without rewriting history and ensure the dedicated workflow is enabled.
2. Run the normal production publication once from that merged commit before the Sep 4 cutoff. Exact replay fails closed if the certified generation predates the benchmark simulator/integration code.
3. Run the dry-run command and inspect the Action summary: ES generation/provenance should validate; Ada should contain current published p50/p5/p95 and threshold evidence with draws still `PARITY_UNVERIFIED`.
4. Confirm branch protection permits `github-actions[bot]` to append the two expected benchmark paths, or arrange the manual real command and normal non-force push immediately after cutoff.
5. Monitor the 21:30 UTC run. If it cannot create a durable entry, do not backfill after the immediately following Stockholm date.

## Known limitations

- The current official Ada evidence does not justify predictive draws, so the expected probabilistic comparison is the compatible 90% interval WIS fallback unless stronger contemporaneous evidence passes the frozen gate.
- Git commit dates are self-asserted. The capture proves what the workflow observed in reachable `main`; the GitHub run timestamp and repository history supply external operational evidence, but this is not a third-party timestamping service.
- The result loader validates the official host, certification label, raw and normalized-manifest hashes, denominator arithmetic, and values. The normalized values are not yet parsed directly from an unknown future Valmyndigheten raw format; the raw artifact and manifest must therefore be committed and independently checked together. If a deterministic official-format parser is added, do so transparently before scoring.
- The pinned result host is `resultat.val.se`. If Valmyndigheten publishes the certified 2026 artifact under another official host, amend the allowlist transparently before importing results rather than weakening it to arbitrary HTTPS.
- Tiny score differences should be reported as such and do not establish universal forecasting superiority.
