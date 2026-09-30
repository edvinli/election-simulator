# SwedishPolls aggregate: replay v1 results

**Decision: FAIL.** `SwedishPollsAggregate-v0.1` stays in shadow mode.
Production is not switched and the gates are not relaxed.

| | |
| --- | --- |
| Protocol | `docs/poll_aggregate_replay_protocol.md`, SHA-256 `3a34221883158a76b2b1021605c2cb1f636a44bba26b06063b2e727c6dc1be5c`, agreed in commit `0a50073` |
| Scored at | HEAD `0a50073`, clean checkout, inputs identical to baseline `fd40555` |
| Aggregate SHA-256 | `d54cf2ba6b2fc66daa309ce0e2601fff4be08814e9ba39ac003945feeae86e90` |
| Cases | 18 per arm, 20,000 draws, seed 20260929 |
| Outputs | `data/processed/poll_aggregate/replay_v1/` (`cases_scored.csv`, `decision.json`, provenance) |

## Gates

| gate | candidate | control | result |
| --- | --- | --- | --- |
| G1 pooled ES_vote (≤ 1.05×) | 3.749 | 3.027 | fail (+24 %) |
| G2 per election (≤ 1.15×) | 2018 4.648 · 2022 2.342 · 2026 4.257 | 3.420 · 2.156 · 3.505 | fail (2018 +36 %, 2026 +21 %; 2022 +9 % passes) |
| G3 ES_seat (≤ 1.05×) / Brier_coalition (≤ +0.01) | 17.26 / 0.0244 | 13.83 / 0.0216 | fail (seat ES +25 %) |
| G4 Brier_threshold (≤ +0.01) | 0.0881 | 0.0635 | fail |
| G5 Coverage_90 / Coverage_50 | 0.772 / 0.364 | 0.870 / 0.481 | fail (90 % below 0.80) |
| G6 integrity | complete; every selected estimate dated on its `as_of` and equal to the aggregate row | | pass |

## What the failure is (diagnostic, not gated)

- **It comes from the daily series, not the poll list.** The diagnostic arm,
  which swaps only `pollofpolls_timeseries.csv`, scores within 0.02 of the
  candidate in every election.
- **The centre is further from the result; the spread is not too narrow.**
  The mean absolute error of the selected starting estimate against the
  result is 1.46 pp for the candidate and 1.21 pp for the control. The
  candidate's 90 % intervals are slightly wider (4.3–4.5 pp vs 4.0–4.3 pp),
  yet cover less.
- **The gap is concentrated close to the election.** At 7 and 14 days, 2018
  ES_vote is 4.7 vs 1.9–2.5, and 2026 is 3.3–3.6 vs 2.5–2.6. At 84–112 days
  the candidate is level or better (2022: 2.72/3.04 vs 3.04/3.14; 2026 at
  112 days: 3.97 vs 4.43).

That pattern fits both explanations the protocol recorded, and this replay
cannot separate them:

1. **PoP's corrections to the level.** The aggregate tracks the raw poll
   average; PoP moves away from it (step-1 diagnostics).
2. **PoP's look-ahead.** The stored PoP series is backfilled with polls
   published after each date, by days to weeks. That matters most at short
   horizons, where the candidate loses most.

## Next: investigate the level gap

These are proposals, not yet done. Any change is a new aggregate version,
scored under a new protocol.

- **Size the look-ahead.** Build diagnostic variants of the aggregate that
  also see polls published up to *k* days after each date (*k* = 3, 7, 14).
  If the short-horizon gap closes at a realistic *k*, look-ahead explains
  much of it.
- **A partly real-time control for 2026.** PoP snapshots committed from
  2026-08-27 onwards (12 commits) are real vintages. They cover the 2026
  cases at 14 and 7 days out and nothing earlier.
- **Level structure.** Decompose the candidate-minus-PoP difference by party
  (SD +0.76 pp on average; S, V and MP about −0.35) and test aggregate
  variants that tie house effects more strongly to election results.
