"""The rendering workflow's contract with certification and the benchmark.

Rendering is a separate workflow because the expensive, retryable half of
publication must not be able to lose the half that cannot be redone. These
tests pin the three properties that make the separation real: how rendering is
triggered, which lock it takes, and that it renders an explicit generation
rather than whatever is newest.

Asserted against the workflow text, as the other workflow tests in this suite
are: PyYAML is not a dependency of this project and adding one to read two
files would be the wrong trade. Where a comment could satisfy a naive substring
search, the assertion matches the YAML key as well as the value.
"""

from __future__ import annotations

from pathlib import Path
import re
import unittest

from scripts.forecast_history.generate import (
    PRODUCTION_HISTORY_WORKERS,
    resolve_history_workers,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPOSITORY_ROOT / ".github/workflows"
RENDER = WORKFLOWS / "election-simulator-render.yml"
PUBLICATION = WORKFLOWS / "election-simulator-publication.yml"
RECONSTRUCTION_CANARY = WORKFLOWS / "election-simulator-reconstruction-canary.yml"
JEKYLL_ACTION = (
    REPOSITORY_ROOT / ".github/actions/setup-jekyll-and-chromium/action.yml"
)
# The website's _config.yml `plugins:` list, plus jekyll itself. `gem install
# jekyll` alone provisions a runner that fails at `jekyll build` on the first
# unresolved plugin.
REQUIRED_GEMS = (
    "jekyll",
    "jekyll-paginate",
    "jekyll-sitemap",
    "jekyll-gist",
    "jekyll-feed",
    "jekyll-redirect-from",
)


def _without_comments(workflow: str) -> str:
    """The YAML with whole-line comments removed.

    This suite deliberately asserts against workflow text, and the render
    workflow now *explains* the conclusion filter it does not have -- so a
    naive substring search finds the explanation. Assertions about what the
    YAML does are made against this; assertions about what it documents are
    made against the raw text.
    """

    return "\n".join(
        line for line in workflow.splitlines() if not line.lstrip().startswith("#")
    )


def _gem_install_blocks(workflow: str) -> list[str]:
    """Every `gem install` invocation in a workflow, backslash joins included."""

    blocks = []
    lines = workflow.splitlines()
    for index, line in enumerate(lines):
        # The invocation, not a comment that mentions it.
        if not re.match(r"\s*gem install\b", line):
            continue
        block = [line]
        cursor = index
        while block[-1].rstrip().endswith("\\") and cursor + 1 < len(lines):
            cursor += 1
            block.append(lines[cursor])
        blocks.append("\n".join(block))
    return blocks


class RenderWorkflowTriggerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.render = RENDER.read_text(encoding="utf-8")
        cls.publication = PUBLICATION.read_text(encoding="utf-8")

    def test_it_is_triggered_by_workflow_run_not_by_push(self) -> None:
        """The mechanism has to actually fire.

        A push made with GITHUB_TOKEN does not trigger other workflows, so the
        certification workflow's push cannot start rendering via `push`.
        `workflow_run` is not subject to that restriction. Asserting this stops
        someone "simplifying" it back to a trigger that silently never runs.
        """

        self.assertRegex(self.render, r"(?m)^  workflow_run:$")
        self.assertRegex(self.render, r"(?m)^  workflow_dispatch:$")
        self.assertNotRegex(
            self.render, r"(?m)^  push:$",
            "a GITHUB_TOKEN push cannot trigger this workflow",
        )

    def test_the_workflow_run_trigger_names_the_publication_workflow_exactly(self) -> None:
        """A name mismatch would make the trigger silently dead."""

        declared = re.search(r"(?m)^name:\s*(.+?)\s*$", self.publication)
        self.assertIsNotNone(declared)
        expected = declared.group(1)
        self.assertIn(f'workflows: ["{expected}"]', self.render)
        self.assertIn("types: [completed]", self.render)

    def test_rendering_never_takes_the_production_lock(self) -> None:
        """The benchmark's protected interval is derived from the publish job.

        The capture serializes against `election-simulator-production`, and the
        protected 18:30Z-22:00Z interval is computed from that job's
        120-minute timeout. A rendering retry queued inside that window must
        not be able to delay a capture, so rendering serializes only against
        other renders.
        """

        self.assertRegex(self.render, r"(?m)^  group: election-simulator-render$")
        self.assertRegex(self.render, r"(?m)^  cancel-in-progress: false$")
        # Keyed on `group:` so the comment explaining which lock is *not* used
        # cannot satisfy this.
        self.assertNotRegex(
            self.render, r"(?m)^\s*group:\s*election-simulator-production\s*$")

    def test_the_publication_workflow_still_owns_the_production_lock(self) -> None:
        self.assertRegex(
            self.publication, r"(?m)^      group: election-simulator-production$")

    def test_it_renders_an_explicit_generation(self) -> None:
        """Never "whatever is latest" at the moment rendering starts."""

        for flag in ("--mode render", "--render-generation",
                     "--render-certification-commit"):
            with self.subTest(flag=flag):
                self.assertIn(flag, self.render)
        dispatch = self.render[self.render.index("workflow_dispatch:"):]
        generation = dispatch[dispatch.index("generation:"):]
        self.assertRegex(
            generation[:generation.index("certification_commit:")],
            r"required: true",
        )

    def test_it_needs_full_history_to_pin_model_inputs(self) -> None:
        """`materialize_pinned_model_inputs` reads the certified revision."""

        self.assertEqual(
            len(re.findall(r"(?m)^\s+uses: actions/checkout@", self.render)), 2)
        self.assertEqual(
            len(re.findall(r"(?m)^\s+fetch-depth: 0$", self.render)), 2,
            "both checkouts need full history",
        )

    def test_it_writes_to_both_repositories(self) -> None:
        self.assertRegex(self.render, r"(?m)^      contents: write$")
        self.assertIn("repository: edvinli/edvinli.github.io", self.render)
        self.assertIn("secrets.WEBSITE_REPO_TOKEN", self.render)

    def test_it_follows_every_completed_publication_whatever_its_conclusion(self) -> None:
        """The recovery case is a publication that certified and then failed.

        Certification pushes the immutable generation before any history,
        projection or browser work, so `certify, push, fail in rendering` is
        the shape this workflow exists to recover -- and a job gated on
        `conclusion == 'success'` skips precisely that run. The condition and
        the comment above it used to contradict each other: the comment said a
        failed publication may still have certified, and the condition then
        declined to look.

        Nothing is inferred from the run's outcome. The target comes from the
        durable certified pointer and `deployed_render_state` decides whether
        it still needs rendering, so following a genuinely empty failure costs
        one cheap RENDER_NOT_NEEDED.
        """

        # No job-level conclusion filter of any kind, checked against the
        # YAML rather than the prose that explains its absence.
        yaml = _without_comments(self.render)
        self.assertNotIn("workflow_run.conclusion", yaml)
        self.assertNotRegex(
            yaml, r"(?m)^    if:",
            "the render job must not gate on the publication run's outcome",
        )
        # And the trigger still fires on completion rather than on success,
        # which is what makes every conclusion reach the job at all.
        self.assertIn("types: [completed]", yaml)
        # The reasoning is recorded where the condition used to be, so the
        # next reader does not restore it.
        self.assertIn("may already have certified", self.render)

    def test_every_conclusion_and_a_dispatch_reach_the_render_job(self) -> None:
        """Enumerated, because "no filter" is easy to regress into "one filter".

        There is no expression to evaluate once the job carries no `if:`, so
        this asserts the property that makes that true: the workflow names no
        conclusion anywhere, for any of the values a completed run can carry.
        """

        yaml = _without_comments(self.render)
        for conclusion in ("success", "failure", "cancelled", "timed_out",
                           "skipped", "action_required", "neutral", "stale"):
            with self.subTest(conclusion=conclusion):
                self.assertNotIn(
                    f"conclusion == '{conclusion}'", yaml,
                    f"a {conclusion} publication must still reach the render job",
                )
        # workflow_dispatch reaches it too, and remains the forcing path.
        self.assertRegex(yaml, r"(?m)^  workflow_dispatch:$")
        self.assertIn(
            "RENDER_REPAIR: ${{ github.event_name == 'workflow_dispatch' }}",
            yaml,
        )

    def test_the_workflow_does_not_decide_whether_rendering_is_needed(self) -> None:
        """That decision moved into `deployed_render_state`, and had to.

        The step here compared current.json to the target generation and
        skipped on equality. The curve backfill is never allowed to block a
        certified forecast, so a render whose reconstruction failed installs
        the publication and flips the pointer anyway -- and a skip keyed on
        the pointer then declines to ever repair the hole, including when an
        operator dispatches the workflow explicitly for that generation.

        Keyed on the step output the old gate wrote, so restoring the gate
        fails this rather than merely reading differently.
        """

        self.assertNotIn("needed=false", self.render)
        self.assertNotIn("needed=true", self.render)
        self.assertNotRegex(
            self.render, r"(?m)^\s+if: steps\.needed\.outputs\.needed",
            "the render step must run and let the renderer decide",
        )

    def test_an_explicit_dispatch_forces_the_render_and_the_follow_on_does_not(self) -> None:
        """Repair is what a dispatch is for; the automatic trigger forces nothing.

        Without this the retry story is incomplete: an operator who knows the
        curve is short has no way to ask for it to be redone, because the only
        generation they would name is the one already in the pointer.
        """

        self.assertIn(
            "RENDER_REPAIR: ${{ github.event_name == 'workflow_dispatch' }}",
            self.render,
        )
        self.assertIn("args+=(--repair)", self.render)
        # Not passed unconditionally: a workflow_run follow-on that forced the
        # work would re-render every publication the publish job already
        # rendered, which is the cost the split exists to avoid paying twice.
        self.assertNotRegex(self.render, r"(?m)^\s+--repair$")


class RenderWorkflowRuntimeTests(unittest.TestCase):
    """What the render job actually invokes, not what it could invoke.

    Both failures pinned here are the same shape: a step that looks correct
    and is only wrong at the moment it does real work. The 2026-09-10 render
    run passed while exercising neither, because it skipped rendering -- the
    website already served the generation.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.render = RENDER.read_text(encoding="utf-8")
        cls.publication = PUBLICATION.read_text(encoding="utf-8")
        cls.action = JEKYLL_ACTION.read_text(encoding="utf-8")

    def test_the_render_invocation_requests_the_production_worker_count(self) -> None:
        """`--history-workers` defaults to 1, and the backfill is why this job exists.

        The 2026-09-10 backfill resolved 123 dates in 58m54s on four workers.
        Serial, that does not fit in this job's 90-minute timeout -- so a
        workflow that omits the flag discards the parallelism the run depended
        on and reintroduces the timeout the split was created to fix.

        Asserted on the argument array the step builds, not merely on the
        string appearing somewhere in the file: a comment mentioning the flag
        must not satisfy this.
        """

        args = self.render[self.render.index('args=(--site-repo'):]
        args = args[:args.index(')')]
        self.assertIn(f"--history-workers {PRODUCTION_HISTORY_WORKERS}", args)
        self.assertIn("--mode render", args)

    def test_rendering_and_publication_request_the_same_worker_count(self) -> None:
        """One pipeline, one worker budget.

        `render_history_for_generation` is shared, so a rendering retry that
        ran it with different concurrency than publication would have a
        different runtime profile for identical work.
        """

        expected = f"--history-workers {PRODUCTION_HISTORY_WORKERS}"
        self.assertIn(expected, self.render)
        self.assertIn(expected, self.publication)
        self.assertLessEqual(
            resolve_history_workers(PRODUCTION_HISTORY_WORKERS), 8,
            "bounded, not all cores: the runner also has the render to do",
        )

    def test_a_dispatch_can_stage_a_render_without_installing_it(self) -> None:
        """The safe way to exercise this path on a real runner.

        Forcing a render against a healthy deployment can only leave it
        unchanged or strip a view that could not be rebuilt, so the canary
        needs a mode that runs the whole expensive, environment-specific path
        and stops before installing. Dispatch-only, and off unless asked for.
        """

        yaml = _without_comments(self.render)
        dispatch = yaml[yaml.index("workflow_dispatch:"):]
        dispatch = dispatch[:dispatch.index("\npermissions:")]
        self.assertIn("dry_run:", dispatch)
        self.assertIn("default: false", dispatch)
        self.assertIn("type: boolean", dispatch)
        self.assertIn(
            "RENDER_DRY_RUN: ${{ github.event.inputs.dry_run == 'true' }}", yaml)
        self.assertIn("args+=(--render-dry-run)", yaml)
        # Never unconditional: a scheduled follow-on must still really render.
        self.assertNotRegex(yaml, r"(?m)^\s+--render-dry-run$")

    def test_rendering_installs_every_gem_the_website_build_needs(self) -> None:
        """`gem install jekyll` alone is a runner that fails at `jekyll build`.

        This workflow shipped exactly that. Nothing caught it because its only
        run skipped rendering, so no Jekyll build ever ran on it.
        """

        self.assertIn(
            "uses: ./simulator/.github/actions/setup-jekyll-and-chromium",
            self.render,
            "rendering must use the shared dependency setup",
        )
        # The path is relative to $GITHUB_WORKSPACE, and the simulator is
        # checked out to `simulator/`, so the action has to be reachable there.
        self.assertIn("path: simulator", self.render)
        self.assertTrue(JEKYLL_ACTION.is_file(), JEKYLL_ACTION)
        for gem in REQUIRED_GEMS:
            with self.subTest(gem=gem):
                self.assertRegex(self.action, rf"(?m)^\s+{re.escape(gem)} \\$")

    def test_the_shared_action_still_provides_a_browser(self) -> None:
        """The browser suites resolve Chromium through CHROME_BIN."""

        self.assertIn('echo "CHROME_BIN=$CHROME_BIN_PATH" >> "$GITHUB_ENV"', self.action)
        self.assertIn("using: composite", self.action)
        self.assertIn("ruby/setup-ruby@v1", self.action)

    def test_the_action_says_which_browser_it_chose(self) -> None:
        """A silent choice is an undiagnosable failure.

        The browser is picked from whatever the runner image carries, and the
        selection was previously invisible: a render that failed with "Chrome
        did not expose CDP" gave no way to tell which binary had been
        launched, so the first diagnosis had to be inferred from a *passing*
        publication's timing instead.
        """

        self.assertIn('echo "browser: $CHROME_BIN_PATH ($CHROME_SOURCE)"', self.action)
        self.assertIn('--version', self.action)
        # Each branch of the selection labels itself, or the log names a path
        # without saying how it was found.
        self.assertEqual(self.action.count("CHROME_SOURCE="), 4)

    def test_the_action_fails_closed_when_no_browser_is_found(self) -> None:
        """Exporting an empty CHROME_BIN defers the failure to a suite.

        The harness would then launch its own default and fail somewhere less
        obvious, which is how an infrastructure problem gets reported as a
        browser-test failure.
        """

        self.assertIn('if [ -z "$CHROME_BIN_PATH" ]; then', self.action)
        self.assertIn("::error::no usable browser was found or installed", self.action)

    def test_ci_asks_the_harness_for_its_bounded_launch_deadline(self) -> None:
        """Configurable in the website repo, requested here.

        The harness defaults to 30s and bounds the setting at 60s. A passing
        publication launched in 23.7s -- a 6.3s margin -- and a rendering job,
        which drives the browser after reconstruction and projection work in
        the same job, lost that race at 30.2s. Asking for the bound is not a
        fix for a browser that cannot start; the harness fails that case
        immediately whatever the deadline says.
        """

        self.assertIn('echo "CDP_READY_TIMEOUT_MS=60000" >> "$GITHUB_ENV"', self.action)

    def test_no_jekyll_install_anywhere_is_missing_a_plugin(self) -> None:
        """One job cannot use the shared action, so pin its copy to it.

        `browser_diagnostic` checks out only the website, so
        ./simulator/.github/actions is not on disk in that job and its install
        stays inline. Every `gem install` in either workflow is therefore
        checked directly, which also catches a new one added by hand.
        """

        blocks = (
            _gem_install_blocks(self.render)
            + _gem_install_blocks(self.publication)
            + _gem_install_blocks(self.action)
        )
        self.assertTrue(blocks, "no gem install found; the assertion has gone blind")
        for block in blocks:
            for gem in REQUIRED_GEMS:
                with self.subTest(gem=gem, block=block[:60]):
                    self.assertRegex(block, rf"(?m)^\s*{re.escape(gem)}\s*\\?$")


class ReconstructionCanaryWorkflowTests(unittest.TestCase):
    """The one criterion the ordinary dry-run canary cannot reach.

    Against a healthy deployment there is nothing to reconstruct, and the
    rendering path only does real work when a publication certified and then
    failed. So this workflow manufactures the gap. Everything asserted here is
    about it being unable to reach production while doing so.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.canary = RECONSTRUCTION_CANARY.read_text(encoding="utf-8")
        cls.yaml = _without_comments(cls.canary)
        cls.script = (
            REPOSITORY_ROOT / "tests/support/reconstruction_canary.py"
        ).read_text(encoding="utf-8")

    def test_it_is_manual_only(self) -> None:
        """A canary that manufactures a gap must never fire on its own."""

        self.assertRegex(self.yaml, r"(?m)^  workflow_dispatch:$")
        for trigger in ("schedule:", "workflow_run:", "push:"):
            with self.subTest(trigger=trigger):
                self.assertNotIn(trigger, self.yaml)

    def test_it_cannot_write_to_either_repository(self) -> None:
        """The strongest available guarantee, and it is available.

        Both repositories are public, so the script clones them over HTTPS
        with no credential at all. The job therefore needs no write scope, and
        is granted none -- which makes "installs nothing" a property of the
        token rather than only of the code path.
        """

        self.assertIn("contents: read", self.yaml)
        self.assertNotIn("contents: write", self.yaml)
        self.assertNotIn("WEBSITE_REPO_TOKEN", self.yaml)
        self.assertNotIn("secrets.", self.yaml)

    def test_it_renders_dry_and_holds_its_own_lock(self) -> None:
        # The renderer must install nothing even though the token could not
        # let it: two independent reasons, not one.
        self.assertIn("--render-dry-run", self.script)
        self.assertRegex(
            self.yaml, r"(?m)^  group: election-simulator-reconstruction-canary$")
        # Never the production lock the benchmark capture serializes against,
        # and not the render lock either: this conflicts with nothing.
        self.assertNotIn("election-simulator-production", self.yaml)
        self.assertNotIn("group: election-simulator-render", self.yaml)

    def test_it_requires_the_generation_and_the_target_date(self) -> None:
        """Never "whatever is latest", and never a guessed gap."""

        dispatch = self.yaml[self.yaml.index("workflow_dispatch:"):]
        dispatch = dispatch[:dispatch.index("\npermissions:")]
        for required in ("generation:", "target_date:"):
            with self.subTest(input=required):
                block = dispatch[dispatch.index(required):]
                self.assertRegex(
                    block[:block.index("type:")], r"required: true", required)

    def test_the_verdict_is_tested_behaviourally_elsewhere(self) -> None:
        """This file asserts the workflow; the verdict needs more than text.

        There used to be a test here listing the script's assertion messages
        and checking they appeared in its source. That style cannot catch a
        wrong comparison, and it missed three: protected-path baselines that
        collided between the two repositories, safety checks skipped when the
        renderer's output could not be parsed, and a "both views rebuilt"
        verdict that never looked at the views.

        The verdict is a pure function now, driven by crafted state in
        tests/test_reconstruction_canary.py. This only pins that the seam
        exists, so the behavioural tests cannot be quietly bypassed.
        """

        self.assertIn("def evaluate(", self.script)
        self.assertIn("FULL_ACCEPTANCE", self.script)
        self.assertIn("RECONSTRUCTION_ONLY", self.script)
        behavioural = REPOSITORY_ROOT / "tests/test_reconstruction_canary.py"
        self.assertTrue(behavioural.is_file(), behavioural)
        self.assertIn("from tests.support.reconstruction_canary import",
                      behavioural.read_text(encoding="utf-8"))

    def test_the_script_never_pushes(self) -> None:
        """Keyed on argv tokens, not on prose.

        Every git invocation in the script is a list of quoted arguments, so
        `"push"` as a literal is the thing to forbid. An earlier version of
        this test searched for the bare word and matched the docstring
        explaining that nothing is pushed.
        """

        self.assertNotIn('"push"', self.script)
        self.assertNotIn("'push'", self.script)
        self.assertNotIn("git push", self.script)
        # And the remote it clones from is never written to: no credential is
        # embedded in a URL, which is the only way a public clone could be.
        self.assertNotIn("@github.com", self.script)
        self.assertNotIn("x-access-token", self.script)


if __name__ == "__main__":
    unittest.main()
