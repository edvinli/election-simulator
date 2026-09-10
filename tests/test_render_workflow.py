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

    def test_it_skips_when_the_website_already_serves_the_generation(self) -> None:
        """Cheap guard before the expensive one in `_reject_stale_render`."""

        self.assertIn("already serves", self.render)
        self.assertIn("needed=false", self.render)


class RenderWorkflowRuntimeTests(unittest.TestCase):
    """What the render job actually invokes, not what it could invoke.

    The 2026-09-10 render run passed while exercising none of this, because it
    skipped rendering -- the website already served the generation.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.render = RENDER.read_text(encoding="utf-8")
        cls.publication = PUBLICATION.read_text(encoding="utf-8")

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


if __name__ == "__main__":
    unittest.main()
