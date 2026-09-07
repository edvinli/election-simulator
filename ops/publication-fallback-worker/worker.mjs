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
// Deploy: see README.md in this directory.

const WORKFLOW = "election-simulator-publication.yml";
const OWNER = "edvinli";
const REPO = "election-simulator";
const REF = "main";
const UA = "election-simulator-publication-fallback";

/**
 * Ask GitHub to run the publication workflow in fallback mode.
 *
 * Returns a plain result rather than throwing, so a scheduled invocation can
 * report a failed dispatch to the dead-man's-switch instead of vanishing.
 */
export async function dispatchFallback(env, fetchImpl = fetch) {
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
        body: JSON.stringify({ ref: REF, inputs: { mode: "publish_if_stale" } }),
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
