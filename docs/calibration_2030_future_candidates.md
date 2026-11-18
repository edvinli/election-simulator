# Calibration programme 2030: candidate improvements to test later

A planning backlog, written 2026-11-18. It is **not** a preregistration, and
it binds nothing.

**The governing rule** (final round §1.1, T5 §1): after the final round, a
2030-cycle model change may be justified only by T5 prospective evidence,
under a new preregistration that cites it. That means:

- at least 26 weekly captures with closed targets at h ≥ 91 days;
- CRPS as the metric;
- the earliest possible decision is about September 2027.

Each candidate below therefore names how it would earn that evidence.

**Not model changes** (T5 §1), and so outside the rule:

- scheduled refits by frozen procedures (the 2030 result, K = 8);
- stand-in replacements;
- new data;
- correctness amendments.

## Ranked candidates

| # | Candidate | Why | How it would be tested | Effort | Priority |
|---|---|---|---|---|---|
| 1 | **`Ul`: log-ratio drift.** Ua's deviation applied multiplicatively | Ua's pp random walk clips small parties at multi-year horizons: L 7.6 % of draws ≈ 0 for 2030 | T5 shadow arm, drafted in `docs/calibration_2030_t5_amendment_001.md`; paired with `Ua` | small: one transform in the runner | **high** |
| 2 | **A threshold-specific metric**: per-party Brier on P(share < 4 %), at all horizons | Seat outcomes hinge on L, KD, MP and C at 4 %; T2 and T4 gate it weakly | Named in the next preregistration before it is seen (T5 §5.3); computable from the archived T5 draws | trivial | **high** |
| 3 | **Cross-country pooling of election-eve errors** (Norway, Denmark, Finland, Germany, Netherlands) | The noise law and bias persistence (β, κ̂) rest on K = 7 Swedish elections. Pooling could give dozens | A data project first: harmonized poll archives and certified results. Then a down-weighted pooled law as a T5 arm, election-day target only | large: weeks of data work | **high value**, start early |
| 4 | **Recover the 1998 fieldwork dates** from primary sources | It would take K from 7 to 8. 1994 stays excluded: a single pollster | Preregister the inclusion rule and any era down-weighting **before** computing 1998's residual; then a T5 arm or a correctness-style data amendment | medium: archive research | medium |
| 5 | **The k bound.** k sits at its 0.5 lower bound at every recent origin (227 of 408 T2 origins) | The filter's posterior looks over-wide at short horizons. The bound is binding, so the true k may be lower | Monitor k in T5 diagnostics. A wider bound would be a new arm, not a retune | trivial | low–medium |
| 6 | **Pollster behaviour**: reweighting after elections, method and mode changes | T1's house reset (`h_reset`) helped. Pollster metadata could explain more | New aggregate arm (v0.4 candidate) in T5, scored on poll targets | medium | medium |
| 7 | **Fundamentals at long horizons**: governing parties' cost of ruling, post-election swings, economy | The only route to sharper 2–4-year forecasts. The simple reversion `Ub2` did not beat `Ua` | Shadow arm only; high overfitting risk with few Swedish cycles; needs external covariates | medium–large | speculative |
| 8 | **Late-campaign tactical voting near 4 %** (*stödröstning*) | Historically, parties near the threshold were **over**-estimated by polls. It is only checkable at elections | Effectively testable only at 2030, as a reported diagnostic | small | low |
| 9 | **Constituency-level uncertainty** in geography | The projection is a deterministic swing. National seats are proportional, so it matters only for the 12 % rule and per-constituency display | T4-style replay after 2030; not a T5 target | medium | low |
| 10 | **Government-formation probabilities** | What readers care about. Political modelling, not vote statistics | A separate product layer; out of scope for calibration | large | product decision |

## Required engineering, not model changes

- **Port the campaign paths and the future fan to the Ua state model before
  2030-05-19.** Production builds them only in the final 112 days, and they
  currently assume OpinionState and Dynamics v2. See the integration plan.
- **The refit after the 2030 result:** the SH segment refit (needs v0.2 to keep
  running), NH0 at K = 8, and the next cycle's history.

## Order of work

1. Finalize T5 amendment 001 (`Ul`) before the first T5 capture, so it
   accumulates evidence from the start.
2. Start the data collection for item 3 now. It has the longest lead time.
3. Write the next preregistration (items 1, 2 and 5) by mid-2027, before its
   26-capture threshold is reached, so its metrics and margins are fixed before
   the data is.
4. Item 4 only if the 1998 dates can be recovered from primary sources.
