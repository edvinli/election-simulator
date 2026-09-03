"""Tests for the affected-test selector and shard planner.

This module is the safety net under the CI topology. The selector decides what
runs on a pull request and the planner decides how the full suite is split, so
a defect in either silently removes coverage while every job still reports
green -- the one failure mode that a test suite cannot catch by itself.

The properties asserted here are the ones the topology's assurance argument
rests on: sharding is a partition (nothing dropped, nothing duplicated), the
nightly split is a partition of the whole suite, and every escalation rule
resolves to the full suite rather than to a guess.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.ci.test_topology import (
    ALWAYS_RUN,
    NIGHTLY_EXHAUSTIVE,
    all_test_modules,
    build_graph,
    select,
    shard,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


class ShardPlanTests(unittest.TestCase):
    """Sharding must never lose or duplicate a module."""

    def test_shards_partition_the_suite_exactly(self) -> None:
        modules = all_test_modules(_REPO_ROOT)
        for count in (1, 2, 3, 4, 5, 8):
            bins = shard(modules, count)
            self.assertEqual(len(bins), count)
            flat = [m for group in bins for m in group]
            self.assertEqual(
                sorted(flat), sorted(modules),
                f"{count} shards did not reproduce the suite exactly",
            )
            self.assertEqual(
                len(flat), len(set(flat)),
                f"{count} shards duplicated a module",
            )

    def test_shard_count_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            shard(["test_a"], 0)

    def test_more_shards_than_modules_yields_empty_bins_not_loss(self) -> None:
        bins = shard(["test_a", "test_b"], 5)
        flat = [m for group in bins for m in group]
        self.assertEqual(sorted(flat), ["test_a", "test_b"])
        self.assertEqual(len(bins), 5)

    def test_shards_are_balanced_by_measured_cost(self) -> None:
        # With the allocator audit at its reduced size no module dominates, so
        # the bins should come out close together. A wide spread means the
        # planner is packing by name or count rather than by duration -- or
        # that timings.json has drifted badly.
        from scripts.ci.test_topology import _timings

        timings = _timings()
        bins = shard(all_test_modules(_REPO_ROOT), 4)
        loads = [sum(timings.get(m, 0.0) for m in b) for b in bins]
        self.assertGreater(min(loads), 0, "a shard was given no measured work")
        self.assertLess(
            max(loads) / min(loads), 1.5,
            f"shard loads are poorly balanced: "
            f"{[round(x, 1) for x in loads]}",
        )


class NightlyIsAReRunNotAnExclusion(unittest.TestCase):
    """The defect these tests exist for.

    An earlier version of the topology removed test_adversarial_mandates from
    the per-change suite and called it "covered by nightly". The pull-request
    matrix went on setting ELECTIONSIM_ADVERSARIAL_CASES=700 for a module that
    was not in the matrix, so the allocator parity audit ran nowhere on a pull
    request while every job reported success. The old assertions passed,
    because they only checked that the case count was large enough -- never
    that the module was present.

    Nightly may run a module *harder*. It may never be the only place a module
    runs.
    """

    def test_every_module_runs_on_every_change(self) -> None:
        everything = set(all_test_modules(_REPO_ROOT))
        # There is no per-change subset any more; this is the guarantee.
        self.assertEqual(len(everything), len(all_test_modules(_REPO_ROOT)))
        for module in NIGHTLY_EXHAUSTIVE:
            self.assertIn(
                module, everything,
                f"{module} is named for a nightly re-run but is not a test module",
            )

    def test_nightly_exhaustive_modules_are_not_withheld(self) -> None:
        modules = all_test_modules(_REPO_ROOT)
        for module in NIGHTLY_EXHAUSTIVE:
            self.assertIn(
                module, modules,
                f"{module} was withheld from the per-change suite; nightly is a "
                f"re-run, not a substitute",
            )

    def test_the_allocator_audit_is_in_the_per_change_suite(self) -> None:
        self.assertIn("test_adversarial_mandates", all_test_modules(_REPO_ROOT))

    def test_an_allocator_change_selects_the_audit(self) -> None:
        modules, _ = select(["scripts/mandates/allocator.py"])
        self.assertIn(
            "test_adversarial_mandates", modules,
            "a change to the allocator must run the parity audit",
        )

    def test_the_audit_appears_in_a_shard_of_the_full_matrix(self) -> None:
        bins = shard(all_test_modules(_REPO_ROOT), 4)
        holding = [b for b in bins if "test_adversarial_mandates" in b]
        self.assertEqual(
            len(holding), 1,
            "the allocator audit must land in exactly one shard of the full run",
        )

    def test_the_audit_is_balanced_at_its_reduced_cost(self) -> None:
        # Sharding only happens in the layers that set the reduced case count,
        # so the recorded duration must be the reduced one. If the exhaustive
        # 511s were recorded here the planner would give the module its own
        # shard and leave the others idle.
        from scripts.ci.test_topology import _timings

        recorded = _timings().get("test_adversarial_mandates")
        self.assertIsNotNone(recorded, "the audit has no recorded duration")
        self.assertLess(
            recorded, 120,
            "timings.json records the exhaustive audit cost; sharding needs the "
            "reduced one",
        )


class SelectionTests(unittest.TestCase):
    """Selection must escalate to the full suite whenever it cannot be sure."""

    def test_documentation_only_change_selects_nothing(self) -> None:
        modules, reason = select(["docs/ci-topology.md", "README.md"])
        self.assertEqual(modules, [])
        self.assertIn("can affect", reason)

    def test_dependency_change_escalates_to_full_suite(self) -> None:
        for path in ("uv.lock", "pyproject.toml"):
            modules, reason = select([path])
            self.assertEqual(
                sorted(modules), sorted(all_test_modules(_REPO_ROOT)),
                f"{path} must select the whole suite",
            )
            self.assertIn("full suite", reason)

    def test_tracked_data_change_escalates_to_full_suite(self) -> None:
        modules, _ = select(["data/processed/simulations/whatever.json"])
        self.assertEqual(
            sorted(modules), sorted(all_test_modules(_REPO_ROOT)))

    def test_unmapped_path_escalates_to_full_suite(self) -> None:
        modules, reason = select(["some/new/thing.txt"])
        self.assertEqual(
            sorted(modules), sorted(all_test_modules(_REPO_ROOT)))
        self.assertIn("unmapped", reason)

    def test_non_python_file_under_scripts_escalates(self) -> None:
        modules, reason = select(["scripts/simulator/notes.txt"])
        self.assertEqual(
            sorted(modules), sorted(all_test_modules(_REPO_ROOT)))
        self.assertIn("full suite", reason)

    def test_artifact_only_tests_are_always_selected(self) -> None:
        # These import nothing from scripts/, so the import graph can never
        # reach them; they must be added unconditionally.
        modules, _ = select(["scripts/simulator/engine.py"])
        for module in ALWAYS_RUN:
            self.assertIn(module, modules)

    def test_a_changed_test_module_runs_itself(self) -> None:
        modules, _ = select(["tests/test_backtest.py"])
        self.assertIn("test_backtest", modules)

    def test_selection_is_transitive_through_the_import_graph(self) -> None:
        # test_actual_browser_consumer imports scripts.simulator.engine, which
        # imports scripts.geography.projection. The test never names
        # projection, so only a transitive walk selects it -- a direct-import
        # table would drop this test on a projection change.
        graph = build_graph(_REPO_ROOT)
        self.assertIn("scripts.simulator.engine", graph["tests.test_actual_browser_consumer"])
        self.assertNotIn(
            "scripts.geography.projection",
            graph["tests.test_actual_browser_consumer"],
            "chain is no longer two-hop; pick another example",
        )
        self.assertIn("scripts.geography.projection", graph["scripts.simulator.engine"])

        modules, _ = select(["scripts/geography/projection.py"])
        self.assertIn("test_actual_browser_consumer", modules)

    def test_allocator_change_selects_the_allocator_tests(self) -> None:
        modules, _ = select(["scripts/mandates/allocator.py"])
        for expected in (
            "test_mandate_allocation",
            "test_adversarial_mandates",
            "test_historical_mandate_law",
        ):
            self.assertIn(expected, modules)

    def test_selection_never_returns_an_unknown_module(self) -> None:
        everything = set(all_test_modules(_REPO_ROOT))
        for changed in (
            ["scripts/mandates/allocator.py"],
            ["scripts/simulator/engine.py"],
            ["tests/test_backtest.py"],
            ["uv.lock"],
        ):
            modules, _ = select(changed)
            self.assertTrue(
                set(modules) <= everything,
                f"{changed} selected something that is not a test module",
            )


class GraphTests(unittest.TestCase):
    def test_graph_covers_every_test_module(self) -> None:
        graph = build_graph(_REPO_ROOT)
        for module in all_test_modules(_REPO_ROOT):
            self.assertIn(f"tests.{module}", graph)

    def test_no_artifact_only_module_is_left_out_of_always_run(self) -> None:
        """A module the graph cannot reach must be declared, or it never runs.

        ALWAYS_RUN is a hand-written list, and a hand-written list is the same
        shape of hazard as the nightly exclusion that dropped the allocator
        audit: add a test module that asserts against tracked artifacts without
        importing anything from `scripts`, forget to list it here, and no
        change under `scripts/` will ever select it. Nothing would fail.
        """
        graph = build_graph(_REPO_ROOT)
        unreachable = {
            module
            for module in all_test_modules(_REPO_ROOT)
            if not any(
                dep.startswith("scripts")
                for dep in graph.get(f"tests.{module}", set())
            )
        }
        missing = sorted(unreachable - set(ALWAYS_RUN))
        self.assertEqual(
            missing, [],
            f"these modules import nothing from scripts/, so no scripts change "
            f"can select them, and they are not in ALWAYS_RUN: {missing}",
        )

    def test_relative_imports_are_resolved_to_absolute_names(self) -> None:
        graph = build_graph(_REPO_ROOT)
        # Every recorded edge is a dotted first-party name, never a bare
        # relative fragment.
        for importer, imported in graph.items():
            for target in imported:
                self.assertTrue(
                    target.startswith(("scripts", "tests")),
                    f"{importer} recorded non-first-party edge {target}",
                )


if __name__ == "__main__":
    unittest.main()


class WorkflowCoverageTests(unittest.TestCase):
    """Every module must run in some required job of some workflow.

    The tier functions can be a correct partition while the workflows still
    fail to invoke one side of it, which would drop a whole tier silently. The
    workflows are read as text rather than parsed, to avoid adding a YAML
    dependency for four assertions.
    """

    WORKFLOWS = _REPO_ROOT / ".github" / "workflows"

    @classmethod
    def setUpClass(cls) -> None:
        if not cls.WORKFLOWS.is_dir():
            raise unittest.SkipTest("no workflows directory")
        cls.text = {
            path.name: path.read_text(encoding="utf-8")
            for path in cls.WORKFLOWS.glob("*.yml")
        }

    def test_full_ci_runs_every_module(self) -> None:
        full = " ".join(self.text["full.yml"].split())
        self.assertIn("--all --shards 4", full)
        # A tier filter here is what silently dropped the allocator audit.
        self.assertNotIn(
            "--tier", full,
            "full.yml must not filter the suite down to a subset",
        )

    def test_pr_ci_does_not_filter_the_suite_to_a_subset(self) -> None:
        self.assertNotIn("--tier", self.text["pr.yml"])

    def test_the_reduced_audit_actually_runs_where_its_knob_is_set(self) -> None:
        """A case count set for a module the matrix omits is not coverage.

        This is the assertion the previous version lacked. It checks the two
        facts together: the knob is set, *and* the module it applies to is
        genuinely reachable in that layer's matrix.
        """
        for name in ("pr.yml", "full.yml"):
            body = self.text[name]
            sets_knob = any(
                "ELECTIONSIM_ADVERSARIAL_CASES:" in line
                and not line.lstrip().startswith("#")
                for line in body.splitlines()
            )
            self.assertTrue(sets_knob, f"{name} sets no audit size")

            # The layer builds its matrix from the selector, so the module is
            # reachable exactly when the selector can return it.
            self.assertIn(
                "test_adversarial_mandates", all_test_modules(_REPO_ROOT),
                f"{name} sets ELECTIONSIM_ADVERSARIAL_CASES for a module that "
                f"is not in the suite its matrix is built from",
            )
            selected, _ = select(["scripts/mandates/allocator.py"])
            self.assertIn(
                "test_adversarial_mandates", selected,
                f"{name} sets the audit size but no diff can select the audit",
            )

    def test_nightly_reruns_every_exhaustive_module_by_name(self) -> None:
        nightly = self.text["nightly.yml"]
        for module in NIGHTLY_EXHAUSTIVE:
            self.assertIn(
                f"tests.{module}", nightly,
                f"{module} is named for a nightly re-run but nightly never "
                f"invokes it",
            )

    def test_nightly_does_not_pin_the_audit_below_its_full_size(self) -> None:
        # The exhaustive job must not set ELECTIONSIM_ADVERSARIAL_CASES; the
        # test's own default is the full 20,000.
        nightly = self.text["nightly.yml"]
        audit_job = nightly.split("allocator-audit:", 1)[1].split("\n  scientific-stress:", 1)[0]
        # Ignore comments; the job explains in prose why it sets nothing.
        effective = [
            line for line in audit_job.splitlines()
            if not line.lstrip().startswith("#")
        ]
        offenders = [l for l in effective if "ELECTIONSIM_ADVERSARIAL_CASES" in l]
        self.assertEqual(
            offenders, [],
            "the exhaustive audit job must not override the case count",
        )

    def test_per_change_layers_use_a_branch_covering_audit_size(self) -> None:
        from tests.test_adversarial_mandates import (
            _FULL_AUDIT_CASES,
            _MIN_BRANCH_COVERAGE_CASES,
        )

        for name in ("pr.yml", "full.yml"):
            values = [
                int(line.split(":", 1)[1].strip().strip('"'))
                for line in self.text[name].splitlines()
                if "ELECTIONSIM_ADVERSARIAL_CASES:" in line
                and not line.lstrip().startswith("#")
            ]
            self.assertTrue(
                values, f"{name} does not set an audit size at all")
            for value in values:
                self.assertGreaterEqual(
                    value, _MIN_BRANCH_COVERAGE_CASES,
                    f"{name} runs the audit below full branch coverage")
                self.assertLessEqual(value, _FULL_AUDIT_CASES, name)

    def test_the_publication_gate_still_names_its_own_suites(self) -> None:
        """The gate must not have been rewired to depend on the new layers."""
        gate = self.text["election-simulator-publication.yml"]
        for module in (
            "tests.test_publication_pipeline",
            "tests.test_prospective_archive",
            "tests.test_forecast_history_contract",
            "tests.test_forecast_history",
            "tests.test_site_publisher",
            "tests.test_production_freeze",
            "tests.test_publication_freeze",
            "tests.test_election_automation",
            "tests.test_polling_acquisition",
        ):
            self.assertIn(
                module, gate,
                f"the publication gate no longer runs {module}",
            )

    def test_the_publication_gate_still_requires_browser_acceptance(self) -> None:
        """The gate's browser suites live in the automation runner, not the YAML."""
        runner = (_REPO_ROOT / "scripts" / "election_automation_base.py").read_text(
            encoding="utf-8")
        for suite in ("forecast-timeseries.smoke.mjs", "government-builder.smoke.mjs",
                      "party-timeseries.smoke.mjs"):
            self.assertIn(
                suite, runner,
                f"the publication gate no longer runs {suite}",
            )
        # The party suite is only a gate in its real-artifact mode; its default
        # mode would validate the committed fixture rather than the artifact
        # the publication just generated.
        self.assertIn(
            "--real-artifact", runner,
            "the publication gate no longer runs the party suite against the real artifact",
        )

    def test_the_publication_gate_runs_the_real_production_sample_count(self) -> None:
        from scripts.election_automation import PRODUCTION_SAMPLES

        self.assertEqual(
            PRODUCTION_SAMPLES, 100_000,
            "the publication gate's production simulation was reduced",
        )

    def test_pr_workflow_does_not_also_trigger_on_push(self) -> None:
        """The duplicate push + pull_request run must stay gone."""
        pr = self.text["pr.yml"]
        trigger_block = pr.split("on:", 1)[1].split("concurrency:", 1)[0]
        self.assertNotIn("push:", trigger_block)
        self.assertIn("pull_request", trigger_block)

    def test_no_module_is_covered_only_by_an_informational_job(self) -> None:
        # cross-repo-coverage and whole-suite-serial are continue-on-error, so
        # nothing may depend on them. full.yml's required shards are built from
        # the whole suite, which is what makes that true.
        full = " ".join(self.text["full.yml"].split())
        self.assertIn("--all --shards 4", full)
        self.assertNotIn("--tier", full)
