# SwedishPolls aggregate: replay v4 results

**Decision: FAIL, on G5 calibration only.** Against the SwedishPolls-only
consensus baseline, v0.2 passes accuracy (G1), every election (G2), seats
(G3), thresholds (G4) and integrity (G6). Its 90 % intervals cover 0.784 of
outcomes, below G5's absolute floor of 0.80. That was known in advance and
recorded in the protocol. Production is not switched.

Everything here is **retrospective**: v0.2 was developed after the 2018,
2022 and 2026 elections, with their outcomes known. It is not independent
validation.

| | |
| --- | --- |
| Protocol | `docs/poll_aggregate_replay_protocol_v4.md`, SHA-256 `daaac76c…`, agreed in `7ecd550` |
| Scored at | HEAD `7ecd550`, clean checkout, inputs identical to baseline `286b6a2` |
| Outputs | `data/processed/poll_aggregate/replay_v4/` |

## Gates

| gate | v0.2 | consensus baseline | result |
| --- | --- | --- | --- |
| G1 pooled ES_vote (≤ 1.05×) | 3.725 | 3.695 | pass (1.008×) |
| G2 per election (≤ 1.15×) | 2018 4.680 · 2022 2.366 · 2026 4.128 | 4.321 · 2.521 · 4.245 | pass (2018 1.08×) |
| G3 ES_seat / Brier_coalition | 17.02 / 0.0264 | 16.98 / 0.0259 | pass |
| G4 Brier_threshold | 0.0849 | 0.0849 | pass |
| G5 Coverage_90 / Coverage_50 | 0.784 / 0.420 | 0.784 / 0.377 | **fail**: Coverage_90 below 0.80 |
| G6 integrity | complete; every case's selected row matches its arm's series | | pass |

## The weakness is calibration, and it is not the input's

- **v0.2 and the naive consensus are equally accurate and equally
  under-covered.** The filter adds little over the simple consensus on these
  cases: better in 2022 and 2026, worse in 2018.
- **Under-coverage is concentrated in 2018 and 2026.** Coverage_90 in 2018 is
  0.61 for v0.2 and 0.69 for the consensus; in 2026, 0.78 and 0.70. In 2022
  both reach 0.96. Interval widths are similar across arms, 3.5–5.6 pp
  depending on horizon.
- **The production model has the same weakness with a real-time input.** For
  the only real-time PoP cases (2026 at 14 and 7 days, from committed
  snapshots), production covers 0.78 and 0.78 with 90 % intervals. v0.2
  covers 0.89 and 0.78; the consensus 0.78 and 0.67. Stored PoP's 0.87 in
  replays v1 and v2 is flattered by hindsight.

So G5 fails because the forecast model's uncertainty is too narrow when it
starts from a real-time poll-based estimate. The candidate aggregate is not
what makes it narrow. Fixing G5 means changing the model's uncertainty
layers, most directly the adopted election-noise law (`pp_lw_gaussian`, K =
4–6 residuals, Gaussian by preregistration). That law is used in production
regardless of the polling input.
