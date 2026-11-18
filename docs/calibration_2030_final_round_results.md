# Calibration programme 2030: final-round results (T2, T3-rev, T4)

Scored under `docs/calibration_2030_final_round_preregistration.md` (frozen,
body `5d070b4b…`) and calibration amendments 001 and 002. Every run was made
from a clean, committed tree, with the integrity guard passing.

| tier | commit | ran (UTC) | output | SHA-256 |
| --- | --- | --- | --- | --- |
| T3-rev | `eb46b51` | 2026-11-18 19:04 → 19:46 | `diagnostics/calibration_2030/t3rev_results.json` | `16385cdf…43e6` |
| T2 | `ef0a5c7` | 2026-11-18, about 20:50 → 20:55 | `diagnostics/calibration_2030/t2_results.json` | `320cd411…afe` |
| T4 | `0459047` | 2026-11-18 21:45 → 23:09 | `diagnostics/calibration_2030/t4_results.json` | `d1be9e16…a6c8` |

T2's run overlapped a merge (`e50941d`). That merge only added files T2 never
imports (`git diff ef0a5c7 e50941d`: three new T4 files), so T2's results
remain attributable to `ef0a5c7`.

**Everything here is retrospective** (final round §1.3).

## T3-rev: selected `NH0` (hierarchical, no size scaling)

| arm | rolling mean ES | leave-one-out mean ES | 2026 ES (reported) | gates |
| --- | --- | --- | --- | --- |
| `N0` (interim, β = 0.5) | 2.642 | 2.413 | 2.471 | control |
| **`NH0`** | **2.589** | 2.380 | 2.573 | **pass** |
| `NH0.5` | 2.665 | 2.384 | 2.645 | fails G1 |
| `NH1` | 2.903 | 2.528 | 2.831 | fails G1–G3 |

**Size scaling fails.** The 2030 refit gives β = 0.703 (κ̂ = 0.338).

## T2: selected `Ua` (the filter posterior, plus a random walk)

| group | `D0` CRPS / 90 % coverage | `Ua` CRPS / 90 % coverage |
| --- | --- | --- |
| long (2–4 years) | 2.025 / **0.59** | 1.899 / **0.92** |
| medium | 0.956 / 0.86 | 0.955 / 0.91 |
| short | 0.441 / 0.94 | 0.442 / 0.93 |

- `Ub1` and `Ub2` also pass; `Uc` fails G1.
- In the block bootstrap by cycle, `Ua`'s long-group difference against `D0`
  is negative in 98 % of resamples.
- k sits at its 0.5 lower bound at most recent origins.

## T4: `CAND` and `FB` both FAIL, on G4 only

- **Self-check.** `C0` reproduced production byte for byte.
- **2014.** It ran on amendment 002's seat path.

| gate | `C0` | `CAND` (v0.3 + Ua + NH0) | `FB` (v0.3 + D0 + N0) |
| --- | --- | --- | --- |
| G1 ES_vote (set D) | 3.673 | **3.421** pass | **3.359** pass |
| G2 per election (2014 / 2018 / 2022) | 3.97 / 4.68 / 2.36 | 3.69 / 4.05 / 2.53 pass | 3.59 / 4.10 / 2.39 pass |
| G3 seats / coalition Brier | 15.84 / 0.0290 | 16.14 / 0.0269 pass | 15.92 / 0.0265 pass |
| **G4 Brier_threshold** | **0.0572** | **0.0772 FAIL** (margin +0.01) | **0.0773 FAIL** |
| G5 coverage 90 / 50 | 0.784 / 0.438 | 0.821 / 0.401 pass | 0.821 / 0.389 pass |
| G6 integrity | pass | pass | pass |
| G7 long horizon: ES / cov90 | 5.37 / 0.741 | 5.15 / **0.901** pass | 5.20 / 0.778 pass |
| 2026 flag (ES ratio to `C0`) | | **1.21, tripped** | 1.14 |

**Frozen decision (final round §1.4).** Production stays 1.2.0-rc1. Only `F1`
may be proposed, on correctness grounds.

### Why G4 fails (diagnostic, reported only)

**It is not F1.** F1, the corrected residuals alone, matches `C0` on
Brier_threshold in every case, to within about 0.002.

**It is the bias correction.** `CAND` and `FB` share v0.3 and a bias
correction (β·r̄). The pool mean is negative for the small parties: L −0.44,
KD −0.53, MP −0.78. So both arms move L, KD and MP down by 0.3–0.5 pp from
nearly identical starting readings.

**What actually happened.** In 2014, 2018 and 2022 **no party fell below 4 %**,
and every near-threshold party cleared it:

| election | party | poll reading two months out | result |
| --- | --- | --- | --- |
| 2014 | KD | 4.2 | 4.57 |
| 2018 | KD | 3.0 | 6.32 |
| 2018 | MP | 4.1 | 4.41 |
| 2022 | MP | 3.7 | 5.08 |
| 2022 | L | 4.4 | 4.61 |

Shifting these parties toward the line raised P(< 4 %) for events that never
happened.

## Two design mistakes, owned by the lead

1. **The fallback was not a clean fallback.** `FB` kept a bias correction
   (`N0`) and production dynamics. **No T4 arm isolated the strongest
   finding**, `Ua` without bias correction: v0.3 + `Ua` + LW0 on the corrected
   residuals. That combination was never scored, so T4 cannot say whether
   `Ua` alone passes. Under §1.4, no candidate may be constructed after T4
   scores exist.
2. **G4 is weak on these elections.** In the decision set there are zero
   threshold events (no party below 4 %). Brier_threshold therefore rewards
   any shift of probability **above** 4 % for parties that cleared. That
   includes overconfidence, and it penalizes honest widening near the line. On
   three elections with no events, G4 is closer to a test of "do not doubt
   the small parties" than to a calibration test. The +0.01 margin was set
   without examining this.

## What the evidence still supports (not a decision)

- **`Ua`'s calibration gain is consistent across T2 and T4.** Long-horizon
  90 % coverage rises from 0.59 to 0.92 in T2, and from 0.74 to 0.90 in T4's
  set L. Accuracy is also better (G1 −7 %).
- **The bias correction**, in any form tested:
  - helped on election-eve energy score in 2014–2022 (T3, T3-rev);
  - hurt threshold forecasts end to end (G4);
  - hurt 2026 (T3 flag, T4 flag).

  It is the component the evidence most argues against.
- **The small-party clipping** of `Ua` at multi-year horizons (T5 amendment
  001 draft) is a separate known issue, not the cause of G4. G4's set D tops
  out at 112 days.

Options for what follows are for the user to decide. They are recorded with
the decision.
