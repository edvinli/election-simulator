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

    def test_it_declines_a_failed_certification_but_not_a_dispatch(self) -> None:
        self.assertIn("github.event_name == 'workflow_dispatch'", self.render)
        self.assertIn(
            "github.event.workflow_run.conclusion == 'success'", self.render)

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


if __name__ == "__main__":
    unittest.main()
