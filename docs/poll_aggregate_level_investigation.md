# SwedishPolls aggregate: level investigation

**Outcome.** Replay v2 scored `SwedishPollsAggregate-v0.2` under its frozen
protocol and it **FAILS**, like v0.1 under replay v1. Both aggregates stay in
shadow mode; production still runs on pollofpolls.se (PoP). The main reason is
now measured: most of the stored PoP series' short-horizon advantage is
hindsight that no causal aggregate can have. The remaining gap comes from
PoP's own departures from the poll consensus. Every result here is
retrospective. None of it is independent validation.

| | |
| --- | --- |
| Start | `7a63bbe` (v0.1 and the failed replay v1, both unchanged) |
| Diagnostics | `90a422b`: `diagnostics/poll_aggregate_level/` |
| v0.2 baseline | `7153bcb` (v0.1 still rebuilds byte-identically: timeseries SHA-256 `d54cf2ba…`) |
| Protocol v2 | `4d922f3`, SHA-256 `fe6c5229…`; agreement recorded in `dfdf925` |
| Replay v2 | scored at `dfdf925`, clean checkout, inputs identical to `7153bcb`; outputs in `data/processed/poll_aggregate/replay_v2/` |

Reproduce: `uv run python -m diagnostics.poll_aggregate_level.run` (about
14 min), `uv run python -m diagnostics.poll_aggregate_level.level_structure`,
`uv run python -m scripts.poll_aggregate --version v0.2` and
`uv run python -m scripts.poll_aggregate.replay --replay v2 --score`
(about 14 min).

## 1. How much stored PoP gains from look-ahead

**Diagnostic arms.** v0.1 rebuilt as a fixed-lag smoother: each date's
estimate may use polls published up to *k* days later. The code is under
`diagnostics/` and is never a candidate. v0.1's causal hyperparameters are
reused unchanged. At *k* = 0 the series equals v0.1 exactly.

| arm | pooled ES_vote | ES_vote at 7–14 days | ES_vote at 84–112 days | Coverage_90 | centre MAE (pp) |
| --- | --- | --- | --- | --- | --- |
| stored PoP (control) | **3.027** | **1.992** | 3.890 | 0.870 | 1.214 |
| v0.1 (causal) | 3.749 | 3.339 | 3.898 | 0.772 | 1.458 |
| + 3 days look-ahead | 3.686 | 3.187 | 4.014 | 0.796 | 1.446 |
| + 7 days | 3.642 | 3.046 | 4.002 | 0.803 | 1.429 |
| + 14 days | 3.448 | 2.529 | 4.015 | 0.821 | 1.345 |

- **Fourteen days of hindsight recovers about 42 % of the pooled gap** and
  about 60 % of it at 7–14 days. Even so it still fails G1 by a wide margin
  (1.14× control against the 1.05× gate).
- **At 84–112 days the arms are level**, where look-ahead cannot help.
- **The gain comes from later fieldwork, not late publication.** A variant
  that sees only late-published polls with fieldwork on or before the date
  gains almost nothing (14 days: centre MAE 1.429, against 1.345 smoothed). So
  the hindsight comes from polls fielded after the date. That fits PoP drawing
  its line between polls on fieldwork dates.
- **About three weeks of hindsight matches stored PoP.** Shifting v0.1's
  causal series back 21 days gives a pooled centre MAE of 1.227 (stored PoP:
  1.214).

## 2. PoP as it was actually available (2026)

Real vintages exist only from 2026-08-27, in the committed polling snapshots
(12 commits). A snapshot is committed only when its content changes, so the
last one committed by a date is the PoP state known then.

| 2026 case | snapshot | selected PoP row | as published | stored today |
| --- | --- | --- | --- | --- |
| 14 days (as of 08-30) | `34c52d6`, known 08-28 | 08-24 (6 days stale) | L 2.0, S 30.4 | L 3.0, S 28.9 |
| 7 days (as of 09-06) | `bc581a3`, known 09-04 | 09-04 | L 2.8, S 27.7 | L 4.0, S 27.9 |

- **Revisions are material.** Across all snapshots, the published latest value
  has since been revised by 0.39 pp on average per party, and by up to
  1.5 pp (S; L up to 1.4). The polls of the day did read L ≈ 2 and S ≈ 30; the stored series was
  later pulled toward the polls that caught L's late rise.
- **Stored PoP scores about 23 % better than the real-time value.** Rerunning
  the production forecast on the snapshot folders gives ES_vote 3.28 (mean of
  the two cases), against 2.53 for the stored series.
- **Against the real-time value, the gap is at the margin.** v0.1 scores 3.46
  and v0.2 3.41, so 1.04–1.06× real-time PoP, against 1.37× stored PoP. Two
  correlated cases cannot support a decision.

## 3. Errors and the gap by party

(Tables: `diagnostics/poll_aggregate_level/results/party_breakdowns.csv`.)

- **v0.1 overestimates SD in every election.** Its forecast-mean error on SD
  averages +1.44 pp across the 18 cases (control +0.64). Against stored PoP it
  sits +0.77 pp higher on SD, and −0.33 to −0.36 on S, V and MP, over 12
  years.
- **The SD level is not a learned election correction.** Removing election
  observations left the SD gap unchanged (+0.78). The level is set by the
  house-effect priors alone. It sits above the sample-weighted pollster
  consensus on SD (+0.36) and below it on MP (−0.35) and V (−0.24). Most
  pollsters read SD below v0.1 (Sifo −1.2, Ipsos −1.0 pp). The two above it
  are the SD-high web panels Sentio (+2.6) and YouGov (+2.9). That is
  consistent with a prior-set level that does not follow sample share, though
  the exact mechanism was not isolated.
- **Stored PoP departs from that consensus itself** (S +0.51, SD −0.42 pp). That
  departure is PoP's method, not the polls.

## 4. v0.2 and its result

**The change.** v0.2 runs v0.1's filter and fits unchanged, with identical
hyperparameters. It reports the **consensus reading** `s + Σ_h w_h b_h`: the
level a sample-weighted mix of the pollsters active in the year before each
fit date would read. That is the scale the election-noise residuals are
defined on.

**How it was chosen.** On outcome-free evidence only (section 3 and
`level_structure.py`). Two other designs were built and discarded on
outcome-free grounds before protocol v2 existed:

- dropping election observations did not move SD;
- a hard sum-to-zero constraint on house effects cost about 1,000
  log-likelihood points, because it stopped a heavily weighted pollster's
  bias from drifting.

**Outcome-free effect.** v0.2 sits on the consensus for every party (SD
−0.05, MP 0.00), and its mean absolute gap to stored PoP falls from 0.44 to
0.36 pp.

**Replay v2** (protocol identical to v1 except the aggregate):

| gate | v0.2 | control | v0.1 (replay v1) |
| --- | --- | --- | --- |
| G1 pooled ES_vote (≤ 1.05×) | 3.725 | 3.027 | 3.749 |
| G2 per election (≤ 1.15×) | 2018 4.680 · 2022 2.366 · 2026 4.128 | 3.420 · 2.156 · 3.505 | 4.648 · 2.342 · 4.257 |
| G3 ES_seat / Brier_coalition | 17.02 / 0.0264 | 13.83 / 0.0216 | 17.26 / 0.0244 |
| G4 Brier_threshold | 0.0849 | 0.0635 | 0.0881 |
| G5 Coverage_90 / Coverage_50 | 0.784 / 0.420 | 0.870 / 0.481 | 0.772 / 0.364 |
| G6 integrity | pass | | pass |

G1–G5 fail, so the **decision is FAIL**. Against v0.1, v0.2 improves
calibration and 2026 modestly, and is slightly worse in 2018 and 2022
overall. The level
correction is real but small next to the look-ahead gap. Per the protocol,
v0.2 was not tuned after this score.

## 5. Diagnosis

1. **The v1/v2 gates cannot be passed by any causal aggregate in
   retrospect.** They are non-inferiority against stored PoP, whose 7–14 day
   values carry roughly two to three weeks of hindsight. An aggregate given
   14 days of hindsight still fails G1.
2. **The level is fixed to the extent the evidence allows.** v0.2's consensus
   reading removes v0.1's arbitrary prior-set level. The rest of the gap to
   PoP is PoP's own correction, and it can't be reverse-engineered without
   fitting to a series that itself has look-ahead.
3. **The only fair comparison available is at the margin.** On the two
   real-time 2026 cases, the causal aggregates are within about 5 % of real
   PoP; on long horizons they are level with stored PoP.

## 6. Limitations

- **Few, correlated cases.** The 18 cases come from three elections, and cases
  within an election are strongly correlated. The real-time evidence is two
  cases from one election.
- **Nothing here is independent.** Both aggregates were designed with the
  outcomes known. The look-ahead and party diagnostics used outcomes.
- **Look-ahead variants use v0.1's hyperparameters.** They isolate
  information, not a refit, so a refit smoother could do somewhat better.
- **Untested assumptions:**
  - the historical result-availability lag of seven days;
  - v0.2's 365-day weight window, chosen a priori and not varied.
- **Snapshot dates bound availability from above.** Snapshot commit times are
  an upper bound on retrieval; true availability may have been slightly
  earlier.
- **Only the full-forecast target was measured.** Nowcast accuracy against
  later polls was not.

## 7. Recommendation and next action

- **Keep production on PoP.** Do not switch, and do not change the v1/v2
  gates to release the 2030 forecast.
- **The decision needed from you: the basis of comparison.** Retrospective
  replays against stored PoP cannot answer the original question, because
  they grade PoP's hindsight. A fair basis has to be prospective and use
  real-time PoP vintages.
- **Recommended next step:** a protocol v3 that archives, every day, the
  real-time PoP vintage and v0.2's row with timestamps, and scores both
  prospectively. The target would be how well each predicts the polls
  published over the following 7–14 days. That gives evidence within months;
  the full-forecast comparison follows at the 2030 election. Writing and
  freezing v3, including its gates, needs your approval, because it changes
  what "better than PoP" means.
