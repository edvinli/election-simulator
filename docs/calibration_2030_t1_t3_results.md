# Calibration programme 2030: T1 and T3 results

Scored under `docs/calibration_2030_preregistration.md`, frozen at
2026-11-18T16:46:25Z (body SHA-256 `7707c2af…`).

Both runs:

- used the harnesses at commit `b27dce4`, with no uncommitted changes to
  tracked files;
- ran on inputs that the guard verified against the frozen record.

| tier | ran (UTC) | output | SHA-256 |
| --- | --- | --- | --- |
| T1 | 16:51:12 → 17:29:28 | `diagnostics/calibration_2030/t1_results.json` | `d5f402541f2067f60382c537c41f9482285fef727f96ada4d12a53f4dea0feaa` |
| T3 | 16:51:12 → 17:52:54 | `diagnostics/calibration_2030/t3_results.json` | `85f4e4109fa2475598e825bf077790adf76d5f9d42c7f3c8fd29a40be434fc8d` |

**Everything here is retrospective (§0 of the preregistration).** This
includes one disclosure. Before the scored run, a smoke test of the T3
harness at N = 400 draws showed the same direction as the result below:
β = 1 best on the rolling targets, with the 2026 flag tripping. The harness
was not changed afterwards. The result is the N = 20,000 run.

## T1: post-election behaviour of the poll aggregate

**Harness self-check.** `C0` reproduces all 5,898 pinned v0.2 daily rows.

Out-of-sample prequential log score against `C0`, in nats:

| arm | gain in post-election windows 2014 / 2018 / 2022 | total | change in mid-cycle mean per poll, 2010–2022 | G1 | G2 | G3 |
| --- | --- | --- | --- | --- | --- | --- |
| `S` | +12.1 / +12.5 / +6.3 | +30.8 | −0.001 to +0.007 | pass | pass | pass |
| `H` | +3.1 / +0.4 / +6.0 | +9.5 | 0.000 to +0.021 | pass | pass | pass |
| `SH` | +13.8 / +11.7 / +10.3 | **+35.8** | 0.000 to +0.025 | pass | pass | pass |

**Selected filter arm: `SH`.** It beats the better single arm (`S`) by 5.0
nats. The frozen threshold is 2.0.

**Integrity (G3).**

- Every fit converged.
- Every arm's in-sample likelihood was at least `C0`'s at every fit date. At
  2010 they are equal, as expected: the new parameters are not identified
  there.
- Simplex: held.
- Causality: 5,880 rows were identical under truncated data.

**Fitted parameters.** `SH`'s support shock is about 0.9e-3, 1.4e-3, 1.6e-3
and 1.5e-3 (2014–2026 fits). Its house reset is about 1.3e-4 to 2.6e-4.

**Rolling weights, `W` (T1-GW): pass.** `W` wins 2010, 2014 and 2018, and
loses 2022. Pooled squared error falls from 0.4504 to 0.4457, a gain of
about 1 %.

**Reported only: 2026.** The M reading, `C0` → `SH`:

| date | `C0` | `SH` |
| --- | --- | --- |
| 2026-09-29 | 19.08 | 21.01 |
| 2026-10-14 | 21.97 | 22.70 |
| 2026-11-18 | 24.73 | 24.76 |

The nine post-election polls of 2026 score −101.9 nats under `C0` and −85.8
under `SH`. The lag is reduced, not removed.

**T1 outcome.** The candidate aggregate `SwedishPollsAggregate-v0.3` is
filter `SH` plus rolling weights `W`.

## T3: the election-noise law

Energy score (nine categories, pp; lower is better), primary seed. Rolling
targets use only earlier elections.

| arm | 2014 | 2018 | 2022 | rolling mean | leave-one-out mean | 2026 (held out) |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian, β = 0 (control) | 2.895 | 4.206 | 1.929 | 3.010 | 2.748 | **2.334** |
| Gaussian, β = 0.5 | 2.504 | 3.854 | 1.571 | 2.643 | 2.418 | 2.460 |
| **Gaussian, β = 1** | 2.432 | 3.764 | 1.583 | **2.593** | 2.401 | 2.897 |
| t₅, β = 0 | 2.930 | 4.275 | 1.928 | 3.044 | 2.763 | 2.368 |
| t₅, β = 0.5 | 2.531 | 3.918 | 1.543 | 2.664 | 2.411 | 2.499 |
| t₅, β = 1 | 2.465 | 3.827 | 1.580 | 2.624 | 2.404 | 2.955 |

**Gates.** Gaussian β = 0.5 and β = 1, and t₅ β = 0.5 and β = 1, pass G1–G3
in all five seeds. t₅ with β = 0 fails G1. No gate outcome is
Monte-Carlo-fragile: the rolling means vary by at most 0.01 across seeds.

**Selected: Gaussian, β = 1.** It has the lowest rolling mean. No less
complex passing arm is within 1 %: β = 0.5 is 1.9 % worse.

**Findings.**

- Heavy tails never help. At every β, t₅ scores worse than the Gaussian with
  the same covariance.
- Bias correction helps strongly on 2014–2022, where S was underestimated
  every time.

**2026 flag: TRIPPED.** The selected arm's 2026 energy score is 2.897
against the control's 2.334, a ratio of 1.24, above the frozen 1.15. Other
2026 scores:

| arm | CRPS, 8 parties | 90 % coverage |
| --- | --- | --- |
| control (β = 0) | 0.645 | 0.89 |
| β = 0.5 | 0.705 | 0.89 |
| β = 1 | 0.863 | 0.67 |

The cause is that the pattern the correction relies on broke in 2026:

| | pool mean bias, 2002–2022 | 2026 residual |
| --- | --- | --- |
| S | +1.90 | +0.18 |
| M | +0.42 | +2.61 |
| SD | +0.58 | −1.53 |
| V | −0.96 | +0.68 |

Under §3.3 the selection is unchanged, and **the user decides whether T4
proceeds with it.**

**Final refit (§3.4)** of Gaussian β = 1 on all seven corrected residuals,
2002–2026:

| | M | L | C | KD | S | V | MP | SD | REST |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean shift (pp) | +0.74 | −0.33 | +0.16 | −0.48 | +1.66 | −0.73 | −0.79 | +0.28 | −0.50 |
| sd (pp) | 1.37 | 0.99 | 1.02 | 0.96 | 1.18 | 1.06 | 0.99 | 1.23 | 1.02 |

This is the 2030 candidate law, if T4 proceeds with it.

## What this does not establish

- **Neither result is independent validation.** T3's rolling targets are
  three elections, the same three on which `pp_lw_gaussian` was itself
  selected.
- **The 2026 flag is one held-out election against three.** It is evidence
  of instability in the bias, not proof that the correction is wrong. Equally,
  three elections are not proof that it is right.
- **Calibration of the full forecast is T4's question, not T3's.** Bias
  correction narrows the centre of the distribution. In 2018 it lowered 90 %
  coverage at election eve (from 0.67 to 0.44), so T4's G5 is a real test
  for it.

## Next

1. **User decision on the T3 2026 flag.** T4 can proceed with Gaussian β = 1,
   or the user can choose otherwise. Any choice other than the frozen
   selection needs an amendment that records why.
2. **T2.** Preregister it, or skip it by decision, before T4.
3. **T4.**
