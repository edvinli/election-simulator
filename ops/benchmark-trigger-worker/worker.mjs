// Independent scheduled trigger for the frozen 2026 prospective benchmark.
//
// GitHub's cron is best-effort, and for this workflow it has been late by
// roughly 1h40m-1h44m on every delivered tick, against a timing-eligible
// window thirty minutes wide. A trigger that runs on the scheduler it is
// compensating for cannot fix that -- so this runs on Cloudflare's.
//
// It contains NO policy, on purpose, and in particular it supplies NO
// scheduled_date and NO timestamp. That is the whole point of the design:
//
//   * slot attribution -> the workflow's schedule_guard, from GitHub's own
//                         recorded workflow-run created_at, by the frozen
//                         rule in scripts/prospective_benchmark_2026/time_rules.py
//   * timing eligibility -> classify_capture_time(), from retrieved_at_utc
//                         taken inside the capture process
//   * frozen window      -> the guard's 2026-09-04..12 check
//   * duplicate slots    -> the guard's stand-down, then append_capture()
//   * cutoff wait        -> wait_for_cutoff(), and run_capture()'s own refusal
//   * concurrency        -> the workflow's election-simulator-production group
//
// Because this Worker names no slot and no instant, an external trigger's
// provenance is identical to the cron's rather than merely equivalent to it,
// and there is no operator-supplied claim for the frozen rules to distrust.
// Adding a scheduled_date input here would undo that; do not add one.
//
// Deploy: see README.md in this directory. This is deliberately a SEPARATE
// Worker from ops/publication-fallback-worker, with its own credential.

const WORKFLOW = "prospective-benchmark-2026.yml";
const OWNER = "edvinli";
const REPO = "election-simulator";
const REF = "main";
const UA = "election-simulator-benchmark-trigger";

/**
 * Ask GitHub to run the benchmark capture workflow as an unattended
 * scheduled trigger.
 *
 * Returns a plain result rather than throwing, so a scheduled invocation can
 * report a failed dispatch to the dead-man's-switch instead of vanishing.
 */
export async function dispatchScheduledCapture(env, fetchImpl = fetch) {
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
        // mode only. No scheduled_date, no timestamp, deliberately.
        body: JSON.stringify({ ref: REF, inputs: { mode: "scheduled_capture" } }),
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
 * A trigger that fails silently is not a trigger, and the failure this whole
 * Worker exists to detect is "the capture never started". Best-effort by
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
    const result = await dispatchScheduledCapture(env);
    if (!result.ok) {
      console.error(
        `benchmark trigger dispatch failed: status=${result.status} ${result.reason}`,
      );
    }
    // Keep the worker alive for the heartbeat without making the dispatch
    // outcome depend on it.
    ctx.waitUntil(reportOutcome(env, result));
  },

  // A manual poke for verifying credentials and the dispatch path end to end.
  // Guarded by a shared secret so the endpoint is not a public capture button.
  //
  // NOTE: this dispatches a real scheduled_capture. Before the frozen 20:30Z
  // boundary it would resolve to the PREVIOUS Stockholm slot, so do not use it
  // to rehearse: see README.md, "Verifying without burning a slot".
  async fetch(request, env) {
    if (request.method !== "POST") {
      return new Response("POST with X-Trigger-Secret to dispatch\n", { status: 405 });
    }
    const provided = request.headers.get("X-Trigger-Secret") || "";
    const expected = env.TRIGGER_SECRET || "";
    if (!expected || provided !== expected) {
      return new Response("forbidden\n", { status: 403 });
    }
    const result = await dispatchScheduledCapture(env);
    return new Response(`${JSON.stringify(result)}\n`, {
      status: result.ok ? 202 : 502,
      headers: { "Content-Type": "application/json" },
    });
  },
};
