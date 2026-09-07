# Benchmark capture trigger

GitHub Actions cron is best-effort. For the prospective benchmark it has not
been good enough: every delivered tick so far arrived 1h40m–1h44m late, against
a timing-eligible window that is **thirty minutes wide**.

| slot | cron | run `created_at` | delay | outcome |
| --- | --- | --- | --- | --- |
| 2026-09-04 | 21:30Z | 23:12:22Z | +1h42m | `LATE_EXCLUDED` |
| 2026-09-05 | 21:30Z | 23:10:38Z | +1h41m | `LATE_EXCLUDED` |
| 2026-09-06 | **20:30Z** | 22:13:47Z | +1h44m | `LATE_EXCLUDED` |

The third row is the important one: amendment 005's one-hour pre-warm was
already live, and the slot was still lost. Starting earlier cannot help,
because the delay is applied to the start. The capture job itself completes in
about 45 seconds. The defect is in trigger *delivery*, so the remedy is a
second, independent *deliverer*. See amendment 006.

## What it does, and what it deliberately does not

It POSTs one `workflow_dispatch` to `prospective-benchmark-2026.yml` with
`mode=scheduled_capture`, three times each September evening. That is all.

It supplies **no `scheduled_date` and no timestamp**, and this is the core of
the design rather than an economy. Every decision belongs to the frozen
implementation, where it is tested:

| decision | where it lives |
| --- | --- |
| which Stockholm slot is this | `resolve_scheduled_slot`, from GitHub's recorded run `created_at` |
| is the capture timing-eligible | `classify_capture_time`, from `retrieved_at_utc` |
| is this inside the frozen window | the workflow's `2026-09-04..12` guard |
| is the slot already captured | the guard's stand-down, then `append_capture` |
| may retrieval happen yet | `wait_for_cutoff`, and `run_capture`'s own refusal |
| may two runs mutate at once | the workflow's `election-simulator-production` group |

Because the Worker names no slot and no instant, an external trigger's slot
provenance is **identical to the GitHub cron's**, not merely equivalent to it.
There is no operator-supplied claim for the frozen rules to have to distrust.

**Do not add a `scheduled_date` input.** It would introduce exactly the
unverified claim this shape exists to avoid.

## Why these three ticks

```
crons = ["30 20 * 9 *", "20 21 * 9 *", "40 21 * 9 *"]
```

- **20:30Z** — the frozen scheduled start (22:30 Europe/Stockholm, amendment
  005). Preserves the full one-hour pre-warm before the cutoff.
- **21:20Z** — a second independent trigger that can still acquire the
  production lock *before* the 21:30Z cutoff if the first tick was lost.
- **21:40Z** — a final post-cutoff retry, with ~20 minutes still left in the
  timing-eligible window.

### No tick may precede 20:30Z

`resolve_scheduled_slot` attributes a run to the Stockholm date whose **22:30
local** scheduled start it was created at or after — and 22:30 local is 20:30Z
during the frozen window. A tick at 20:00Z would therefore resolve to the
**previous day's** slot, and could durably occupy it with a `LATE_EXCLUDED`
record. Nor may a tick fall at or after 22:00Z, when the eligible window
closes at local midnight.

Both bounds are asserted against these exact expressions in
`tests/test_prospective_benchmark_2026_external_trigger.py`. They are not
stylistic.

## Redundancy, not replacement

The GitHub cron `30 20 * 9 *` is **retained**. Disabling it would only move the
single point of failure from GitHub to Cloudflare. When both arrive, whichever
lands first captures and the other stands down green.

## Verifying without burning a slot

The guarded `POST` endpoint dispatches a **real** `scheduled_capture`. Before
20:30Z it would resolve to the previous Stockholm slot, so it is not a
rehearsal tool. The safe order is:

1. Verify deployment, auth and configuration **without** dispatching:
   ```sh
   npx wrangler deployments list
   npx wrangler secret list          # names only, never values
   npx wrangler triggers list        # confirm the three crons
   ```
2. Let the first real trigger be the next scheduled 20:30Z tick.
3. **Only after** an `ON_TIME_ELIGIBLE` capture exists for that slot, use the
   guarded POST to prove the Cloudflare → GitHub path end to end:
   ```sh
   curl -sS -X POST -H "X-Trigger-Secret: $TRIGGER_SECRET" \
     https://election-benchmark-trigger.<subdomain>.workers.dev
   ```
   Expect `202` and `{"ok":true,"status":204}`, then a green run whose summary
   reads `Stand-down: SLOT_ALREADY_TIMING_ELIGIBLE`. That single run proves
   both the dispatch path and the duplicate-suppression path at once, and
   captures nothing.

## Deploy

```sh
cd ops/benchmark-trigger-worker
npx wrangler secret put GITHUB_TOKEN      # fine-grained PAT, see wrangler.toml
npx wrangler secret put TRIGGER_SECRET    # any long random string
npx wrangler secret put HEARTBEAT_URL     # optional
npx wrangler deploy
```

## Separate from the publication fallback, on purpose

`ops/publication-fallback-worker` is a different Worker with a **different
credential**, and the two must not be merged.

- **Blast radius.** One leaked secret must not be able to start both
  production paths. Each PAT can start exactly one workflow.
- **Schedules that must not converge.** The publication fallback's three ticks
  are in the morning *specifically* because they cannot contend for the
  production lock with a capture. Putting a 20:30Z tick on that Worker would
  place a publication-capable credential on a cron inside the benchmark's own
  protected window — the precise coupling that design avoids.
- **Different failure semantics.** A missed publication is repaired by
  publishing later. A missed capture is a permanently empty frozen slot.

The code shape is shared because it was already reviewed; the deployment,
token and schedule are not.

## Token scope

A fine-grained PAT limited to `edvinli/election-simulator` with **Actions: read
and write** and nothing else. It can start this workflow; it cannot push code,
read other secrets, or write to the archive. The capture's own `contents:
write` stays inside GitHub, granted by the workflow.

Note what `created_at` does and does not establish. It is GitHub's own
authoritative record of when the run was created, and it is what the slot is
attributed from. It is **not** evidence of scheduler identity: an authorized
operator can choose when to initiate a `workflow_dispatch`. Scientific timing
eligibility does not rest on it — that remains `retrieved_at_utc`, taken from
the capture process clock, and no externally supplied value participates.
