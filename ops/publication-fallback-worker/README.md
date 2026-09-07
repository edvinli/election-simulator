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
| is today's publication live on both source and site | `daily_publication_satisfied` |
| does the kill switch apply | `automation_enabled_for_event` |
| would this collide with a benchmark cutoff | `benchmark_window_conflict` |
| should this publish at all | `should_publish` |
| may two publications run at once | the workflow's `election-simulator-production` group |

The consequence is that a fallback tick on a normal day spends one short
workflow run and publishes nothing. That is the intended trade: the rules stay
in one tested place, and the no-op is visible in the Actions history.

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
