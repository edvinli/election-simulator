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
    NIGHTLY_ONLY,
    all_test_modules,
    build_graph,
    select,
    shard,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


class ShardPlanTests(unittest.TestCase):
    """Sharding must never lose or duplicate a module."""

    def test_shards_partition_the_suite_exactly(self) -> None:
        modules = all_test_modules(_REPO_ROOT, tier="per-change")
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

    def test_balancing_keeps_the_dominant_module_alone(self) -> None:
        # The point of measured sharding: the 519-second audit must not be
        # bin-packed alongside other work when it is included.
        modules = all_test_modules(_REPO_ROOT, tier="all")
        bins = shard(modules, 4)
        heavy = [b for b in bins if "test_adversarial_mandates" in b]
        self.assertEqual(len(heavy), 1)
        self.assertEqual(
            heavy[0], ["test_adversarial_mandates"],
            "the exhaustive audit should occupy its own shard when present",
        )


class NightlySplitTests(unittest.TestCase):
    """The nightly split must be a partition, not a subtraction."""

    def test_tiers_partition_the_suite(self) -> None:
        everything = set(all_test_modules(_REPO_ROOT, tier="all"))
        per_change = set(all_test_modules(_REPO_ROOT, tier="per-change"))
        nightly = set(all_test_modules(_REPO_ROOT, tier="nightly"))
        self.assertEqual(per_change | nightly, everything)
        self.assertEqual(per_change & nightly, set())

    def test_nightly_only_modules_exist(self) -> None:
        everything = set(all_test_modules(_REPO_ROOT, tier="all"))
        for module in NIGHTLY_ONLY:
            self.assertIn(
                module, everything,
                f"NIGHTLY_ONLY names {module}, which is not a test module",
            )

    def test_unknown_tier_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            all_test_modules(_REPO_ROOT, tier="sometimes")


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
                sorted(modules), sorted(all_test_modules(_REPO_ROOT, tier="all")),
                f"{path} must select the whole suite",
            )
            self.assertIn("full suite", reason)

    def test_tracked_data_change_escalates_to_full_suite(self) -> None:
        modules, _ = select(["data/processed/simulations/whatever.json"])
        self.assertEqual(
            sorted(modules), sorted(all_test_modules(_REPO_ROOT, tier="all")))

    def test_unmapped_path_escalates_to_full_suite(self) -> None:
        modules, reason = select(["some/new/thing.txt"])
        self.assertEqual(
            sorted(modules), sorted(all_test_modules(_REPO_ROOT, tier="all")))
        self.assertIn("unmapped", reason)

    def test_non_python_file_under_scripts_escalates(self) -> None:
        modules, reason = select(["scripts/simulator/notes.txt"])
        self.assertEqual(
            sorted(modules), sorted(all_test_modules(_REPO_ROOT, tier="all")))
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

    def test_per_change_tier_drops_nightly_modules_from_selection(self) -> None:
        modules, _ = select(["scripts/mandates/allocator.py"], tier="per-change")
        for module in NIGHTLY_ONLY:
            self.assertNotIn(module, modules)

    def test_selection_never_returns_an_unknown_module(self) -> None:
        everything = set(all_test_modules(_REPO_ROOT, tier="all"))
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
        for module in all_test_modules(_REPO_ROOT, tier="all"):
            self.assertIn(f"tests.{module}", graph)

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

    def test_full_ci_runs_the_whole_per_change_tier(self) -> None:
        full = self.text["full.yml"]
        self.assertIn("--all --shards 4 --tier per-change", " ".join(full.split()))

    def test_nightly_runs_every_nightly_only_module_by_name(self) -> None:
        nightly = self.text["nightly.yml"]
        for module in NIGHTLY_ONLY:
            self.assertIn(
                f"tests.{module}", nightly,
                f"{module} is deferred to nightly but nightly never names it",
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
        for suite in ("forecast-timeseries.smoke.mjs", "government-builder.smoke.mjs"):
            self.assertIn(
                suite, runner,
                f"the publication gate no longer runs {suite}",
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
        # cross-repo-coverage and whole-suite-serial are continue-on-error.
        # Anything they run must also run in a required job.
        everything = set(all_test_modules(_REPO_ROOT, tier="all"))
        per_change = set(all_test_modules(_REPO_ROOT, tier="per-change"))
        nightly = set(all_test_modules(_REPO_ROOT, tier="nightly"))
        # full.yml (required) covers per_change; nightly.yml's required jobs
        # cover NIGHTLY_ONLY by name.
        self.assertEqual(per_change | nightly, everything)
