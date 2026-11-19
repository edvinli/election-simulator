# T5 amendment 003: operations — `shadow/2030` pushed, captures scheduled in CI

**Status: FINAL.** Decided by the user on 2026-11-19 ("push `shadow/2030` with
scheduled CI"). Finalized at 2026-11-19T09:19:37Z.

**Effect.** It takes effect when committed with all of the following:

- its JSON record,
  `data/processed/calibration_2030/t5_amendments/003-operations-ci.json`;
- that record's SHA-256 companion;
- its entry in `data/processed/calibration_2030/t5_amendments/index.json`.

**Scope.** It is append-only. `docs/calibration_2030_t5_shadow_preregistration.md`
(frozen, body SHA-256 `bf93c128…`) §7 left pushing `shadow/2030` and CI
scheduling to separate user decisions. This amendment records that decision
and how it is carried out. It changes no arm, target, seed, metric, cadence,
archive rule or decision rule.

## What changes

**The branch.** `shadow/2030` is pushed to `origin`.

- It is never force-pushed, rewritten or deleted (§4.1).
- Captures W47 onwards keep their commits.

**The schedule.** The workflow is
`.github/workflows/calibration-2030-t5-shadow.yml`. A scheduled workflow runs
only from the default branch, so the file lives on `main`. It:

- **Runs** daily at 07:23 UTC. That is after the 04:00 UTC always-publish run
  and its render, and outside 18:30–22:00 UTC. `workflow_dispatch` is
  allowed.
- **Checks out `shadow/2030`** with full history and runs:
  - `uv sync --frozen`;
  - `uv run python -m diagnostics.calibration_2030.t5_shadow capture --workers 4`;
  - then `verify`;
  - then pushes `shadow/2030` without force.
- **Skips without running** when the current ISO week already has a committed
  capture. A daily schedule gives up to seven attempts per week; §3.1 counts
  the week as missed only if none succeeds.
- **Tolerates one refusal until Sunday.** "No certified production generation
  this week yet" is reported, but not as a failure, until Sunday. On Sunday,
  the last day of the week, it fails. Every other refusal fails.
- **Opens or updates a GitHub issue on failure**, titled "T5 shadow capture
  failing". That covers fetch, capture, verify and push failures, so a broken
  pipeline is visible within a day.
- **Stays isolated.** It has its own concurrency group
  (`calibration-2030-t5-shadow`). It writes nothing outside
  `data/processed/shadow_2030/` on `shadow/2030`, and never invokes
  publication, certification, rendering or deployment (§7).
- **Triggers nothing else.** No workflow triggers on a push to `shadow/2030`,
  as checked on 2026-11-19:
  - `full.yml` runs on pushes to `main` only;
  - `pr.yml` runs on pull requests;
  - publication runs on a schedule or by dispatch;
  - render runs on `workflow_run` of publication, from the default branch.

  A push made with `GITHUB_TOKEN` starts no workflows in any case. The
  website (`edvinli/edvinli.github.io`) is a separate repository, and is
  updated only by the render workflow.

**The runtime.**

- CI captures run on `ubuntu-24.04`, CPython 3.12, with the locked
  dependencies.
- The W47 capture (2026-11-16) ran locally on CPython 3.11.15.
- Each manifest records its runtime, and arms are compared only within a
  capture (one runtime for every arm). Replay is not claimed bit-stable
  across runtimes (§4.5).

**Timing evidence.** From W48, a capture's push to `origin` gives an external
time record, alongside the self-reported committer time. W47 was committed
while the branch was local only, as its verification note says.

## Production model versions

The user also decided on 2026-11-19 to ship `F1` (the corrected election-eve
residuals) to production as a new `MODEL_VERSION`. T5 needs no change for it:

- `C0` is production as published.
- Each manifest records the anchor's `model_version`.
- Reports split `C0` by production model version (§2.4).
- `C0p` is computed from the production model files at the anchor's commit
  whenever they differ from the T5 code commit (§2.4 a).

The first capture whose anchor carries the new version marks the cut.

## Code change authorized

The runner accepts in-force T5 amendments that add no arm, such as this one.
Earlier captures keep their recorded code commit (§4.4).
