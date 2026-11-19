# T5 amendment 002: add full-pipeline shadow arm `U0` (Ua without bias correction)

**Status: FINAL.** Authorized by the user on 2026-11-19. Finalized at
2026-11-19T09:08:59Z.

**Authorization.** The user chose "One exception arm" and "Yes, finalize Ul",
whose option text reads: "If you choose the exception arm above, also add 'Ua
without bias' as a shadow arm."

**Effect.** It takes effect when committed with all of the following:

- its JSON record,
  `data/processed/calibration_2030/t5_amendments/002-u0-shadow-arm.json`;
- that record's SHA-256 companion;
- its entry in `data/processed/calibration_2030/t5_amendments/index.json`.

**Scope.** It is append-only. It amends
`docs/calibration_2030_t5_shadow_preregistration.md` (frozen, body SHA-256
`bf93c128…`) §2.2 by adding one full-pipeline arm. Nothing else in T5
changes:

- cadence;
- targets;
- metrics;
- the seeds of existing arms;
- archive rules;
- the decision rule.

## Why

Calibration amendment 003 ran `U0` once through T4 as a post-hoc exception
arm. `U0` is v0.3 + `Ua` + production's unbiased noise law LW0 on the
corrected residual pool.

`U0` failed the frozen gates on **G5 alone** (`docs/calibration_2030_t4_exception_u0_results.md`):

- Its set-`D` 90 % coverage was 0.772. The floor is 0.80, and `C0` scored
  0.784.
- It passed G1–G4, G6 and G7, and the 2026 flag did not trip.

Production therefore stays 1.2.0-rc1. Under the one-round budget, `U0` can
reach production only through T5 prospective evidence and a later
preregistration (T5 §6). Logging it from now on is the only admissible way to
collect that evidence.

## The arm

| item | `U0` |
| --- | --- |
| definition | exactly `diagnostics/calibration_2030/t4_exception_u0.py::config_u0()`, as committed with calibration amendment 003 (`4109291`): aggregate v0.3, state and dynamics `Ua`, noise LW0 (production's `pp_lw_gaussian`, centred, β = 0, γ = 0), corrected residual pool |
| noise pool | `candidate_engine.pool_years(2030)`, the seven corrected residuals 2002–2026 (K = 7, as `FB` and `CAND`) |
| path | the T4 candidate path, as `FB` and `CAND` (`t5_shadow.run_engine_arm`) |
| k | `Ua`'s k at the capture (the engine reads it from the capture's T2 context) |
| draws and seed | N = 4,000; the shared engine seed `calibration-2030:T5:<O>:engine` (T5 §3.3), so `U0`, `FB` and `CAND` use common random numbers |
| logged | the 2030 election-day distribution only, `U0.npz` with `election_draws_centipp` (uint16, 4,000 × 9) |
| poll targets | none: a full-pipeline arm's poll-target distribution is that of its T2 arm (`Ua`), which §2.1 already logs |
| label | copied into every manifest that carries `U0`: `{"verdict": "T4X_FAIL", "failed_gates": ["G5_calibration"], "flag_2026_raised": false}` and the SHA-256 of `diagnostics/calibration_2030/t4_exception_u0_results.json` (`c89ca59f…12b5ef`) |
| from when | the first capture whose origin is on or after the date this amendment takes effect, 2026-11-19; no backfill (the week-47 capture, origin 2026-11-16, does not carry it) |

## Runner change authorized

This amendment, with T5 amendment 001, authorizes the T5 runner changes that:

- add `U0` and `Ul`;
- read in-force T5 amendments from
  `data/processed/calibration_2030/t5_amendments/index.json`, verifying each
  record against its index entry and companion hash;
- record the amendments in force in each manifest's `preregistration`
  block.

Earlier captures keep their recorded code commit (T5 §4.4).

## Disclosure and limits

- **`U0` is post hoc twice over.** It was specified after T4's scores were
  seen, and it is logged after its own T4 result was seen. Its T5 record is
  prospective, but the choice to log it is not.
- **Its election-day distribution shares `Ua`'s known small-party clipping**
  at multi-year horizons, before election noise. `Ul` (T5 amendment 001)
  tracks the alternative.
- **Decision use is as T5 §6.** Any later adoption needs a new
  preregistration citing T5, with gates stated before the comparison is seen.
  Because `U0` failed G5 on short horizons, that preregistration should name
  short-horizon interval coverage explicitly.
