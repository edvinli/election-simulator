# Calibration preregistration, amendment 001: T3's 2026 flag, interim noise law β = 0.5

**Status: FINAL.** Decided by the user on 2026-11-18. Finalized at
2026-11-18T18:38:17Z.

It takes effect when committed with all of the following:

- its JSON record,
  `data/processed/calibration_2030/amendments/001-t3-flag-interim-beta-half.json`;
- that record's SHA-256 companion;
- its entry in `data/processed/calibration_2030/amendments/index.json`.

It is append-only. It does not edit `docs/calibration_2030_preregistration.md`
(frozen, body SHA-256 `7707c2af…`), `docs/calibration_2030_t1_t3_results.md`,
or any result.

## What happened

T3 ran under the frozen §3. The selection rule chose Gaussian noise with full
bias correction (β = 1). Its 2026 held-out energy score exceeded the
control's by more than the frozen 1.15 ratio, so the §3.3 flag tripped. §3.3
leaves the decision to the user.

| arm | rolling mean ES (2014, 2018, 2022) | leave-one-out mean ES | 2026 ES | 2026 ratio to control |
| --- | --- | --- | --- | --- |
| Gaussian, β = 0 (control) | 3.0102 | 2.7480 | 2.3339 | 1.00 |
| Gaussian, β = 0.5 | 2.6431 | 2.4182 | 2.4595 | 1.05 |
| Gaussian, β = 1 (frozen selection) | 2.5927 | 2.4010 | 2.8969 | **1.24** |

Source: `diagnostics/calibration_2030/t3_results.json`
(SHA-256 `85f4e4109fa2475598e825bf077790adf76d5f9d42c7f3c8fd29a40be434fc8d`).

## The decision

For T4, and as the interim noise law of the T5 shadow run, the noise law is
**Gaussian with β = 0.5** instead of the frozen selection (β = 1). The law is
exactly the tested T3 arm:

- `r = 0.5 · r̄_P + ε`, with `ε ~ N(0, Σ̃)`;
- `Σ̃` from `fit_election_noise_b` on the centred pool;
- the engine's simplex transfer.

It has **no** additional `(1 + β/K)` variance inflation. That variant was
not tested.

**This choice was made after the 2026 held-out result was seen.** It is a
post hoc decision, not a preregistered selection:

- β = 0.5 passed every T3 gate (G1–G3) in all five seeds. Its rolling mean
  ES is 1.9 % above β = 1's.
- Its 2026 ratio is 1.05.
- It keeps most of the gain on 2014–2022 and roughly matches the control's
  2026 score.

**The 2030 interim law (§3.4 refit on all seven corrected residuals,
2002–2026).** The mean shift is `0.5 · r̄`, in pp:

| M | L | C | KD | S | V | MP | SD | REST |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| +0.368 | −0.166 | +0.081 | −0.242 | +0.829 | −0.363 | −0.396 | +0.138 | −0.250 |

The covariance is unchanged by β. Its standard deviations, in pp:

| M | L | C | KD | S | V | MP | SD | REST |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.372 | 0.989 | 1.019 | 0.962 | 1.175 | 1.057 | 0.987 | 1.227 | 1.015 |

## What it does not do

- It does not change T3's record or its frozen selection. The T3 result still
  reads "selected Gaussian β = 1, 2026 flag tripped".
- It does not change any T4 gate, margin or case.
- It does not adopt anything in production.
- The final round, `docs/calibration_2030_final_round_preregistration.md`,
  tests further noise laws against this interim law (T3-rev). This amendment
  does not pre-empt that test.
