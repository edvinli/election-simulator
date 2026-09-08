# Publication fallback trigger

GitHub Actions cron is best-effort. When the `0 4 * * *` tick that carries the
mandatory daily recalculation is delayed or dropped, the public forecast stays
on yesterday's generation and nothing notices. This Worker is the trigger of
last resort, and it runs on Cloudflare's scheduler because a fallback on
GitHub's own scheduler cannot compensate for GitHub's own scheduler.

## What it does, and what it deliberately does not

It POSTs one `workflow_dispatch` to `election-simulator-publication.yml` with
`mode=publish_if_stale`, three times each morning. That is all it does.

It holds **no policy**. Whether anything is actually stale, whether the kill
switch permits publishing, whether a benchmark capture is imminent, and whether
a publication would be a duplicate are all decided by the automation, where
they are unit-tested:

| decision | where it lives |
| --- | --- |
| does the kill switch apply | `fallback_preflight`, then `automation_enabled_for_event` |
| is a benchmark capture queued or in flight | `fallback_preflight`, via the Actions API |
| is a capture imminent but not yet created | `benchmark_window_conflict`, from the frozen protocol |
| is today's publication live on both source and site | `daily_publication_satisfied` |
| should this publish at all | `should_publish` |
| may two publications run at once | the **publish job's** `election-simulator-production` group |

The first three run in `fallback_preflight`, which is deliberately **outside**
the production concurrency group. That matters: the group is shared with the
prospective benchmark, and a run that joins it before deciding to stand down
is already holding the lock a capture is waiting for. Only the publish job
takes that lock, so ordinary scheduled and manual publication still serialize
with the benchmark exactly as before.

The consequence is that a fallback tick on a normal day spends one short
workflow run and publishes nothing. That is the intended trade: the rules stay
in one tested place, and the no-op is visible in the Actions history.

## Schedule: three logical retries, one Cloudflare cron

The dispatch times are **04:45Z, 05:30Z and 06:30Z** -- 45, 90 and 150 minutes
after the `0 4 * * *` daily cron. They are delivered by a single coarse
trigger:

```toml
crons = ["*/15 4-6 * * *"]
```

Cloudflare's Workers Free plan allows **5 cron triggers per _account_**, not
per Worker, and `ops/benchmark-trigger-worker` holds three for the frozen 2026
benchmark campaign. Three expressions here would make six, which Cloudflare
rejects outright -- the schedules call is all-or-nothing per script, so the
result is *no* crons registered rather than some of them (observed: HTTP 400,
`code: 10072`). One coarse expression costs one slot and carries all three
times.

`isDispatchTick()` in `worker.mjs` selects them from the trigger's twelve daily
invocations, reading `event.scheduledTime` as **UTC** (`getUTCHours` /
`getUTCMinutes`, never the local-time getters -- the Worker's zone is not
guaranteed). The other nine invocations are a clean no-op:

| | |
| --- | --- |
| GitHub dispatch | none |
| heartbeat, success **or** `/fail` | none |
| publication action | none |

The heartbeat exclusion is deliberate. A ping from a tick that was never meant
to publish would report health this Worker has not established, and a `/fail`
ping would invent an incident.

**This selection is scheduler plumbing, not publication policy.** It answers
only "is this the tick I meant?". Staleness, the kill switch, the benchmark
window, double-publish and concurrency remain decisions of the GitHub workflow
-- see the table above. The guarded `POST` endpoint bypasses the filter
entirely and dispatches immediately, because an operator asking for a dispatch
is not a scheduled tick.

## Publication semantics

`publish_if_stale` is a distinct mode, not a flag on `publish`, because it
differs in exactly two ways that matter:

- **It is subject to the kill switch.** An operator dispatch bypasses
  `ELECTION_AUTOMATION_ENABLED` because a human asked for it. An unattended
  external trigger must not, or adding a fallback would quietly grant the
  automated path a way around "stop publishing".
- **It only covers an absent daily.** `FALLBACK_DAILY` publishes
  unconditionally when today's recalculation is missing — that is the mandatory
  recalculation, not a poll-change check — and falls back to poll-change
  semantics once the daily is accounted for.

## Deploy

```sh
cd ops/publication-fallback-worker
npx wrangler secret put GITHUB_TOKEN      # fine-grained PAT, see wrangler.toml
npx wrangler secret put FALLBACK_SECRET   # any long random string
npx wrangler secret put HEARTBEAT_URL     # optional
npx wrangler deploy
```

Verify the dispatch path without waiting for a tick:

```sh
curl -sS -X POST -H "X-Fallback-Secret: $FALLBACK_SECRET" \
  https://election-publication-fallback.<subdomain>.workers.dev
```

Expect `202` and `{"ok":true,"status":204}`, then a `FALLBACK_DAILY` run in the
Actions history whose summary reads `Daily publication: SATISFIED_TODAY` and
`Deployment status: NO_PUBLICATION_REQUIRED` on a day that already published.

## Token scope

A fine-grained PAT limited to `edvinli/election-simulator` with
**Actions: read and write** and nothing else. It can start this workflow; it
cannot read the repository's other secrets, push code, or publish anything
directly. The publication itself still runs under the workflow's own
`contents: write` permission inside GitHub.
