# Calibration programme 2030: T4 exception arm `U0` (amendment 003): FAIL on G5

Scored under calibration amendment 003 (record SHA-256 `b6ea2163…`), with the
unchanged frozen T4 gates of `docs/calibration_2030_preregistration.md` §4.4
and `docs/calibration_2030_final_round_preregistration.md`.

| item | value |
| --- | --- |
| commit | `4109291` (clean tracked tree; integrity guard passed) |
| ran (UTC) | 2026-11-19 07:59:15 → 09:04:28 (3,913 s, 12 workers, 5 metric slots) |
| output | `diagnostics/calibration_2030/t4_exception_u0_results.json` |
| SHA-256 | `c89ca59fff37516bfa655ebd51f53fd1c8e6e142968a8681b7c198129812b5ef` |
| draws / seed | 20,000 / 20260929 |

**Before the official run**, a one-case smoke run (2018, 28 days, 500 draws)
was made to check that the harness completed. It is not used.

**Self-check and reproduction.**

- The `C0` self-check passed.
- `C0` and `F1` reproduced the recorded T4 run (`t4_results.json`,
  `d1be9e16…a6c8`) exactly: 630 of 630 compared metric values were identical.

**Everything here is retrospective, and `U0` is a post-hoc arm** (amendment
003, Disclosure).

## Gates (decision set `D`, 2014–2022; long set `L`)

| gate | `C0` | `U0` | frozen rule | result |
| --- | --- | --- | --- | --- |
| G1 ES_vote (D) | 3.673 | 3.496 | ≤ 1.05 × `C0` and ≤ 1.05 × `F1` (3.667) | pass |
| G2 per election (2014 / 2018 / 2022) | 3.97 / 4.68 / 2.36 | 3.90 / 4.11 / 2.48 | each ≤ 1.15 × `C0` | pass |
| G3 seat ES / coalition Brier | 15.84 / 0.0290 | 15.05 / 0.0276 | ≤ 1.05 × `C0` and `F1` / ≤ `C0` + 0.01 | pass |
| G4 Brier_threshold | 0.0572 | **0.0520** | ≤ `C0` + 0.01 | pass |
| **G5 coverage 90 / 50** | 0.784 / 0.438 | **0.772** / 0.407 | cov90 in [0.80, 0.97] and closer to 0.90 than `C0`; cov50 in [0.35, 0.65] | **FAIL** |
| G6 integrity | pass | pass | | pass |
| G7 long horizon ES / cov90 | 5.37 / 0.741 | 5.22 / **0.926** | ES ≤ 1.05 × `C0`; cov90 closer to 0.90 | pass |
| 2026 flag (ES ratio to `C0`) | | 1.03 | > 1.15 trips | not tripped |

**Frozen decision (amendment 003).** `U0` fails. Production stays 1.2.0-rc1.
Only `F1` may be proposed, on correctness grounds. This was the only exception
arm. No further arm may be constructed from T4 or `U0` scores, and later
changes go only through T5.

## Why G5 fails (diagnostic, reported only)

G5 counts 90 % intervals over 9 categories in 18 set-`D` cases, so 162
intervals in all:

| arm | intervals covering the result | mean 90 % width (pp) | k |
| --- | --- | --- | --- |
| `C0` | 127 / 162 (0.784) | 4.38 | |
| `F1` | 127 / 162 | 4.47 | |
| `U0` | **125 / 162** (0.772) | **4.06** | 0.5 in every case |
| `CAND` (recorded T4) | 133 / 162 (0.821) | 4.21 | 0.5 in every case |

- **The miss is two intervals.** `U0` covers 2 fewer than `C0`, and 8 short
  of the 0.80 floor. `C0` itself is below that floor.
- **The intervals are narrower.** At horizons of 112 days or less, `U0`'s
  intervals are about 7 % narrower than `C0`'s. In every set-`D` case, `Ua`'s
  scale factor k sits at its lower bound of 0.5. This shrinks the filter
  posterior below production's residual-based state uncertainty. Without the
  `NH0` widening factor (1 + β/K) that `CAND` carried, nothing compensates.
- **Where the misses fall.** They are in 2022 (28–84 days) and in 2014 at 112
  days.
- **This is the k-bound item** in `docs/calibration_2030_future_candidates.md`.
  It can only be pursued through T5.

## What the exception shows (not a decision)

- **Removing the bias correction fixes G4,** as diagnosed. `U0`'s
  Brier_threshold is 0.052, against 0.077 for `CAND`. The 2026 flag no longer
  trips (1.03, against 1.21 for `CAND`).
- **`Ua` again improves accuracy and long-horizon calibration.**
  - G1 improves by 5 %.
  - The seat energy score improves by 5 %.
  - Long-horizon 90 % coverage rises from 0.74 to 0.93.
- **But at short horizons, `Ua`'s k = 0.5 makes intervals too narrow** for the
  frozen G5 rule.
- **No arm passed every gate.** The frozen outcome stands.

## Decision (user, 2026-11-19)

- **`U0` is not adopted**, per the frozen rule. It is logged prospectively as T5
  shadow arm `U0` (T5 amendment 002), and `Ul` is logged as T5 amendment 001.
- **`F1` is shipped as a correctness fix:** production model 1.2.1-rc1, for
  2030 targets only, with K = 6 unchanged. It is in
  `edvinli/election-simulator` PR #55, with methods text in
  `edvinli/edvinli.github.io` PR #47. A data-corruption fix is not held to the
  "must improve the backtest" bar. In T4, `F1` matched `C0` (G1 3.667 against
  3.673; coverage 0.784 for both).
- **`shadow/2030` is pushed, with scheduled CI** (T5 amendment 003;
  `edvinli/election-simulator` PR #54).
- **Backlog items 11 and 12** are added to
  `docs/calibration_2030_future_candidates.md`.
