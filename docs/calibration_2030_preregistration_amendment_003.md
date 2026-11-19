# Calibration preregistration, amendment 003: one post-hoc exception arm in T4 (`U0`: Ua without bias correction)

**Status: FINAL.** Authorized by the user on 2026-11-19. Finalized at
2026-11-19T07:55:03Z.

It takes effect when committed with all of the following:

- its JSON record,
  `data/processed/calibration_2030/amendments/003-t4-exception-arm-u0.json`;
- that record's SHA-256 companion;
- its entry in `data/processed/calibration_2030/amendments/index.json`.

It is append-only. It edits no frozen document and no result.

**This is an explicit exception, not a preregistered test.** It is the kind
of step the final round's §1.4 forbids ("no other candidate may be
constructed after T4 scores exist"). It is made openly, by user decision,
under the precedent of `docs/poll_aggregate_replay_protocol_v4_amendment_001.md`
(a recorded release exception).

## What happened

T4 (`diagnostics/calibration_2030/t4_results.json`, SHA-256 `d1be9e16…a6c8`)
failed both `CAND` and `FB`, **on G4 (Brier_threshold) alone**:

| arm | Brier_threshold | `C0` | margin |
| --- | --- | --- | --- |
| `CAND` | 0.0772 | 0.0572 | +0.01 |
| `FB` | 0.0773 | 0.0572 | +0.01 |

Both passed G1–G3 and G5–G7. `docs/calibration_2030_final_round_results.md`
records the diagnostic:

- **What both arms share.** `CAND` and `FB` share a bias correction
  (β·r̄, with β = 0.5 or about 0.7). It moves L, KD and MP down by
  0.3–0.5 pp.
- **What happened instead.** Every near-threshold party cleared 4 % in 2014,
  2018 and 2022.
- **What it is not.** F1 alone (the corrected residuals) matches `C0` on G4.

**The lead's design error.** No T4 arm isolated the strongest finding of the
round, T2's `Ua`, without a bias correction. `FB` kept one.

## The exception

**One additional T4 arm, `U0`,** is run once:

| layer | `U0` |
| --- | --- |
| aggregate | v0.3 (final round §2) |
| state and dynamics | **`Ua`**, T2's selection (final round §3, with k and the sub-seeds of §6.2) |
| election noise | **LW0**: production's own unbiased law `pp_lw_gaussian` (centred, β = 0, γ = 0), fitted per target on the **corrected** pool (residual amendments 001 and 002) of the elections strictly before it. For the 2030 target, it is refitted on all seven corrected residuals, 2002–2026. |
| residuals | F1's (corrected) |
| geography, allocation, transfer | as every T4 arm; 2014 per amendment 002 |

**Every component has already been implemented and tested.** LW0 is
byte-identical to production's law (`tests/test_calibration_2030_t3rev.py`),
and `Ua` is T2's selection. Nothing is tuned. The arm only **removes** the
component that the T4 diagnostic implicates.

**How it is judged.** `U0` is judged by the **unchanged** frozen T4 gates:

- G1–G7, with every margin as frozen. **G4 keeps its +0.01 margin**, despite
  its weakness on these elections.
- The 2026 flag.
- The same cases, including amendment 002's 2014 path and 2010 vote-only
  cases.
- 20,000 draws and seed 20260929.

`C0` and `F1` are rerun in the same run. Their metrics must reproduce the
recorded T4 run exactly. Any difference is reported, and the comparison is
made within the new run.

**Decisions.**

- **`U0` passes every gate and the 2026 flag does not trip.** `U0` is proposed
  as `MODEL_VERSION` 1.3.0-rc1. The user authorized a production update on a
  demonstrated improvement on 2026-11-18.
- **`U0` passes, but the 2026 flag trips.** The user decides.
- **`U0` fails any gate.** Production stays 1.2.0-rc1, and only F1 may be
  proposed.

**This is the only exception arm.** No further arm may be constructed from
T4 or `U0` scores. Later changes go only through T5 (T5 §6).

## Disclosure

- `U0` was specified **after** T4's scores and its G4 diagnostic were seen.
  It is a post-hoc construction, and it is therefore more likely to pass
  than a preregistered arm would be. Any pass is retrospective evidence of
  the weakest kind in this programme.
- `U0` inherits `Ua`'s known small-party clipping at multi-year horizons
  (T5 amendment 001). At the 2030 horizon, L has about 7.6 % of draws near
  zero before election noise. That property is not gated here, and it is
  tracked prospectively by the T5 arm `Ul`.
- The strongest evidence for `Ua` comes from T2: about 2,000 poll-prediction
  targets, with the long-horizon gain present in 98 % of block-bootstrap
  resamples. That evidence was not produced by this exception.
