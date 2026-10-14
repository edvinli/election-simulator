// External fallback trigger for the mandatory daily publication.
//
// GitHub's cron is best-effort. A delayed or dropped 04:00Z tick leaves the
// public forecast on yesterday's generation, and a fallback that runs on the
// same scheduler cannot fix that -- so this runs on Cloudflare's.
//
// It contains NO policy, on purpose. It does not read the forecast, does not
// decide whether anything is stale, and does not know what a publication is.
// It pokes the existing publication workflow with mode=publish_if_stale and
// lets the automation decide, because every one of those decisions is a tested
// rule in scripts/publication_fallback.py and scripts/election_automation_base.py
// and a second copy here would be a second answer:
//
//   * liveness         -> daily_publication_satisfied()
//   * kill switch      -> automation_enabled_for_event(mode="publish_if_stale")
//   * benchmark window -> benchmark_window_conflict()
//   * double-publish   -> should_publish(daily_already_satisfied=...)
//   * concurrency      -> the workflow's election-simulator-production group
//
// The cost of that choice is a workflow run that no-ops when the daily already
// happened. That is the right trade: it is cheap, it is visible in the Actions
// history, and it keeps the rules in one place with the tests.
//
// ONE COARSE CRON, THREE LOGICAL RETRIES.
//
// Cloudflare's Workers Free plan allows 5 cron triggers per ACCOUNT, not per
// Worker, and ops/benchmark-trigger-worker holds three of them for the frozen
// 2026 benchmark campaign. Three separate expressions here would need six in
// total, which Cloudflare rejects outright -- the schedules call is
// all-or-nothing, so the effect is no crons at all rather than some of them.
//
// So the deployed schedule is a single coarse cron, "*/15 * * * *", and the
// three intended dispatch times are selected from its 96 daily invocations by
// isDispatchTick() below. One account cron slot, three logical retries, and
// the dispatch times are unchanged: 04:45Z, 05:30Z, 06:30Z.
//
// That filtering is SCHEDULER PLUMBING, not publication policy. It answers
// only "is this the tick I meant?" -- never "should anything be published?".
// Every rule in the list above still lives in the workflow, and the manual
// POST endpoint deliberately bypasses the filter entirely.
//
// THE POLL-CHANGE WATCHER rides on the same cron.
//
// GitHub has also been delivering the hourly poll-change ticks late or not at
// all, so a poll merged upstream could wait most of a day. The cron therefore
// fires every 15 minutes, all day, and on every invocation watchPolls() asks
// GitHub for the newest commit touching the upstream poll file. When that
// commit is new and has stayed the newest for POLL_QUIET_MS, it dispatches the
// workflow with mode=poll_change -- an hourly tick in every respect except
// when it arrives. The quiet period is there because upstream often merges a
// batch and then a correction minutes later (2026-10-14: the October polls at
// 07:10Z, a fix to Sifo's SD figure at 07:27Z); publishing the first would put
// a wrong number on the site until the next run.
//
// This is scheduler plumbing in the same sense as the tick filter: it decides
// only when to ask. Whether the change is model-relevant, the kill switch, the
// benchmark window and concurrency are the workflow's, exactly as for a cron
// tick. The watcher's state is one KV key; without the POLL_STATE binding it
// does nothing at all.
//
// Deploy: see README.md in this directory.

const WORKFLOW = "election-simulator-publication.yml";
const OWNER = "edvinli";
const REPO = "election-simulator";
const REF = "main";
const UA = "election-simulator-publication-fallback";

// The upstream poll file the publication acquires, and the branch it is read
// from. Only commits touching this path wake the watcher.
export const POLL_SOURCE = {
  owner: "MansMeg",
  repo: "SwedishPolls",
  path: "Data/Polls.csv",
  ref: "master",
};
// How long a new upstream commit must stay the newest before it is published.
export const POLL_QUIET_MS = 15 * 60 * 1000;
const POLL_STATE_KEY = "poll-watch";

// The three intended dispatch times, as "HH:MM" in UTC: 45, 90 and 150 minutes
// after the 04:00Z daily cron. These are the schedule; "*/15 * * * *" is
// merely the delivery mechanism that can afford one account cron slot.
export const DISPATCH_TICKS_UTC = ["04:45", "05:30", "06:30"];

/**
 * The "HH:MM" of a Cloudflare scheduledTime, in UTC, or null if unreadable.
 *
 * Cloudflare passes event.scheduledTime as milliseconds since the epoch. UTC
 * is read explicitly with getUTCHours/getUTCMinutes rather than getHours: the
 * Worker's local zone is not guaranteed and the schedule is defined in UTC, so
 * a local-time reading would drift the dispatch times by the offset.
 */
export function scheduledTickUtc(scheduledTime) {
  const instant =
    scheduledTime instanceof Date ? scheduledTime : new Date(scheduledTime);
  if (Number.isNaN(instant.getTime())) return null;
  const hours = String(instant.getUTCHours()).padStart(2, "0");
  const minutes = String(instant.getUTCMinutes()).padStart(2, "0");
  return `${hours}:${minutes}`;
}

/**
 * Is this invocation one of the three intended dispatch times?
 *
 * Pure, and the only thing standing between the coarse cron's 96 daily
 * invocations and the three that mean something. Anything else -- including a
 * missing or unreadable scheduledTime -- is false, so the handler no-ops
 * rather than guessing at a dispatch nobody asked for.
 */
export function isDispatchTick(scheduledTime) {
  return DISPATCH_TICKS_UTC.includes(scheduledTickUtc(scheduledTime));
}

/**
 * Ask GitHub to run the publication workflow in fallback mode.
 *
 * Returns a plain result rather than throwing, so a scheduled invocation can
 * report a failed dispatch to the dead-man's-switch instead of vanishing.
 */
export async function dispatchFallback(env, fetchImpl = fetch) {
  return dispatchWorkflow(env, { mode: "publish_if_stale" }, fetchImpl);
}

/** Ask GitHub to run the publication workflow as a poll-change check. */
export async function dispatchPollChange(env, fetchImpl = fetch) {
  return dispatchWorkflow(env, { mode: "poll_change" }, fetchImpl);
}

async function dispatchWorkflow(env, inputs, fetchImpl) {
  if (!env || !env.GITHUB_TOKEN) {
    return { ok: false, status: 0, reason: "GITHUB_TOKEN is not configured" };
  }
  let response;
  try {
    response = await fetchImpl(
      `https://api.github.com/repos/${OWNER}/${REPO}/actions/workflows/${WORKFLOW}/dispatches`,
      {
        method: "POST",
        headers: {
          Accept: "application/vnd.github+json",
          Authorization: `Bearer ${env.GITHUB_TOKEN}`,
          "X-GitHub-Api-Version": "2022-11-28",
          "Content-Type": "application/json",
          "User-Agent": UA,
        },
        body: JSON.stringify({ ref: REF, inputs }),
      },
    );
  } catch (error) {
    return { ok: false, status: 0, reason: `dispatch request failed: ${error}` };
  }
  // 204 No Content is the documented success for a workflow dispatch.
  if (response.status === 204) return { ok: true, status: 204 };
  let detail = "";
  try {
    detail = (await response.text()).slice(0, 300);
  } catch {
    detail = "<unreadable body>";
  }
  return { ok: false, status: response.status, reason: detail };
}

/**
 * The newest upstream commit touching the poll file, or a failure result.
 *
 * Authenticated with the Worker's own token: unauthenticated calls share a
 * 60-an-hour limit per IP, and Workers share IPs. A fine-grained token can
 * read any public repository, so this needs no scope beyond the dispatch's.
 */
export async function latestPollCommit(env, fetchImpl = fetch) {
  if (!env || !env.GITHUB_TOKEN) {
    return { ok: false, status: 0, reason: "GITHUB_TOKEN is not configured" };
  }
  const { owner, repo, path, ref } = POLL_SOURCE;
  const query = new URLSearchParams({ path, sha: ref, per_page: "1" });
  let response;
  try {
    response = await fetchImpl(
      `https://api.github.com/repos/${owner}/${repo}/commits?${query}`,
      {
        headers: {
          Accept: "application/vnd.github+json",
          Authorization: `Bearer ${env.GITHUB_TOKEN}`,
          "X-GitHub-Api-Version": "2022-11-28",
          "User-Agent": UA,
        },
      },
    );
  } catch (error) {
    return { ok: false, status: 0, reason: `commit lookup failed: ${error}` };
  }
  if (response.status !== 200) {
    return { ok: false, status: response.status, reason: "commit lookup refused" };
  }
  let commits;
  try {
    commits = await response.json();
  } catch {
    return { ok: false, status: 200, reason: "commit lookup was not JSON" };
  }
  const sha = Array.isArray(commits) && commits[0] && commits[0].sha;
  if (typeof sha !== "string" || !/^[0-9a-f]{40}$/.test(sha)) {
    return { ok: false, status: 200, reason: "commit lookup returned no commit" };
  }
  return { ok: true, sha };
}

/**
 * What the watcher should do, given the newest upstream commit and its state.
 *
 * Pure. The state is { dispatched, pending: { sha, seenAt } }: the commit last
 * handed to the workflow, and a newer one waiting out the quiet period. The
 * quiet period runs on the watcher's own clock from when it first saw the
 * commit, not on the commit's date -- a pull request merged today can carry a
 * commit dated last week, and that must still wait.
 *
 * Returns { action, state }, where action is one of:
 *   initialise -- no state yet: adopt the current commit, publish nothing
 *   unchanged  -- the newest commit is the one already dispatched
 *   settling   -- a new commit, still inside its quiet period
 *   dispatch   -- a new commit that has stayed the newest for the period
 */
export function pollWatchDecision(latestSha, state, now) {
  if (!state || typeof state.dispatched !== "string") {
    return { action: "initialise", state: { dispatched: latestSha, pending: null } };
  }
  if (latestSha === state.dispatched) {
    return { action: "unchanged", state: { dispatched: state.dispatched, pending: null } };
  }
  const pending = state.pending;
  if (!pending || pending.sha !== latestSha || !Number.isFinite(pending.seenAt)) {
    return {
      action: "settling",
      state: { dispatched: state.dispatched, pending: { sha: latestSha, seenAt: now } },
    };
  }
  if (now - pending.seenAt < POLL_QUIET_MS) {
    return { action: "settling", state };
  }
  return { action: "dispatch", state: { dispatched: latestSha, pending: null } };
}

/**
 * One watcher pass: look upstream, decide, and dispatch when it is time.
 *
 * The new state is written only after a successful dispatch, so a refused
 * dispatch is retried on the next invocation rather than forgotten.
 */
export async function watchPolls(env, now, fetchImpl = fetch) {
  const store = env && env.POLL_STATE;
  if (!store) return { action: "disabled" };
  const latest = await latestPollCommit(env, fetchImpl);
  if (!latest.ok) {
    console.error(`poll watch lookup failed: status=${latest.status} ${latest.reason}`);
    return { action: "lookup_failed", status: latest.status };
  }
  let state = null;
  try {
    state = JSON.parse((await store.get(POLL_STATE_KEY)) || "null");
  } catch {
    state = null;
  }
  const decision = pollWatchDecision(latest.sha, state, now);
  if (decision.action === "dispatch") {
    const result = await dispatchPollChange(env, fetchImpl);
    if (!result.ok) {
      console.error(
        `poll-change dispatch failed: status=${result.status} ${result.reason}`,
      );
      return { action: "dispatch_failed", status: result.status };
    }
  }
  if (JSON.stringify(decision.state) !== JSON.stringify(state)) {
    await store.put(POLL_STATE_KEY, JSON.stringify(decision.state));
  }
  return { action: decision.action, sha: latest.sha };
}

/**
 * Report the outcome to a dead-man's-switch, if one is configured.
 *
 * A fallback that fails silently is not a fallback. This is best-effort by
 * construction: the dispatch has already happened or already failed, and
 * nothing here may change that outcome.
 */
export async function reportOutcome(env, result, fetchImpl = fetch) {
  const base = env && env.HEARTBEAT_URL;
  if (!base) return { reported: false };
  const url = result.ok ? base : `${base.replace(/\/$/, "")}/fail`;
  try {
    await fetchImpl(url, { method: "POST", headers: { "User-Agent": UA } });
    return { reported: true };
  } catch {
    return { reported: false };
  }
}

export default {
  async scheduled(event, env, ctx) {
    const scheduledTime = event && event.scheduledTime;
    // The watcher runs on every invocation, independently of the tick filter
    // below; it makes no outbound call unless POLL_STATE is bound.
    const now = new Date(scheduledTime).getTime();
    if (Number.isFinite(now)) {
      ctx.waitUntil(watchPolls(env, now));
    }
    // Scheduler plumbing only. 93 of the coarse cron's 96 daily invocations
    // are not one of the intended times, and for the fallback each of those
    // must be a clean no-op: no dispatch, no heartbeat, nothing observable. In
    // particular NOT a heartbeat -- a dead-man's-switch ping from a tick that
    // was never meant to publish would report health this Worker has not
    // established, and a /fail ping would invent an incident.
    if (!isDispatchTick(scheduledTime)) {
      return;
    }
    const result = await dispatchFallback(env);
    if (!result.ok) {
      console.error(
        `publication fallback dispatch failed: status=${result.status} ${result.reason}`,
      );
    }
    // Keep the worker alive for the heartbeat without making the dispatch
    // outcome depend on it.
    ctx.waitUntil(reportOutcome(env, result));
  },

  // A manual poke for verifying credentials and the dispatch path end to end.
  // Guarded by a shared secret so the endpoint is not a public publish button.
  async fetch(request, env) {
    if (request.method !== "POST") {
      return new Response("POST with X-Fallback-Secret to trigger\n", { status: 405 });
    }
    const provided = request.headers.get("X-Fallback-Secret") || "";
    const expected = env.FALLBACK_SECRET || "";
    if (!expected || provided !== expected) {
      return new Response("forbidden\n", { status: 403 });
    }
    const result = await dispatchFallback(env);
    return new Response(`${JSON.stringify(result)}\n`, {
      status: result.ok ? 202 : 502,
      headers: { "Content-Type": "application/json" },
    });
  },
};
