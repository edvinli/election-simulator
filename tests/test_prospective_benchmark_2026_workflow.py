"""Static safety checks for the dedicated prospective benchmark workflow."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "prospective-benchmark-2026.yml"


class TestProspectiveBenchmarkWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = WORKFLOW.read_text(encoding="utf-8")

    def test_schedule_dispatch_and_shared_production_lock(self) -> None:
        self.assertIn('cron: "30 21 * 9 *"', self.text)
        self.assertIn("workflow_dispatch:", self.text)
        self.assertIn("Europe/Stockholm", self.text)
        self.assertIn("2026-09-(0[4-9]|1[0-2])", self.text)
        self.assertIn("group: election-simulator-production", self.text)
        self.assertIn("cancel-in-progress: false", self.text)
        self.assertIn("actions: read", self.text)
        self.assertIn('gh api "repos/$REPOSITORY/actions/runs/$RUN_ID" --jq .created_at', self.text)
        self.assertIn('date -d "$RUN_CREATED_AT"', self.text)

    def test_dry_run_and_real_capture_modes_are_explicit(self) -> None:
        self.assertIn("- dry_run", self.text)
        self.assertIn("- capture", self.text)
        self.assertIn("--mode \"$CAPTURE_MODE\"", self.text)
        self.assertNotIn("--retrieved-at", self.text)
        self.assertIn("Verify dry run left the repository untouched", self.text)
        self.assertIn("schedule_guard:", self.text)
        self.assertIn("needs: schedule_guard", self.text)
        self.assertIn("tests.test_prospective_benchmark_2026_capture", self.text)
        self.assertIn("tests.test_prospective_benchmark_2026_botten_ada_capture", self.text)
        self.assertIn("tests.test_prospective_benchmark_2026_adversarial", self.text)
        self.assertNotIn("tests.test_prospective_botten_ada_capture", self.text)

    def test_dry_run_has_read_only_token_and_capture_has_write_token(self) -> None:
        self.assertIn("dry_run:\n    if:", self.text)
        self.assertIn("dry_run:\n    if: needs.schedule_guard", self.text)
        self.assertIn("capture:\n    if: needs.schedule_guard", self.text)
        self.assertIn("persist-credentials: false", self.text)
        self.assertIn("capture:\n    if: needs.schedule_guard.outputs.active == 'true' && needs.schedule_guard.outputs.mode == 'capture'\n    needs: schedule_guard\n    permissions:\n      contents: write", self.text)

    def test_no_website_checkout_or_write(self) -> None:
        lower = self.text.lower()
        self.assertNotIn("edvinli.github.io", lower)
        self.assertNotIn("website_repo_token", lower)
        self.assertNotIn("site_repo", lower)

    def test_commit_is_restricted_to_the_slot_and_index(self) -> None:
        self.assertIn("Verify only the expected benchmark slot changed", self.text)
        self.assertIn("data/processed/prospective_benchmark_2026/index.json", self.text)
        self.assertIn('data/processed/prospective_benchmark_2026/captures/$CAPTURE_ID', self.text)
        self.assertIn("git diff --cached --name-only", self.text)
        self.assertNotIn("git add .", self.text)

    def test_locked_dependencies_and_capture_gates_precede_commit(self) -> None:
        self.assertIn("uv sync --frozen", self.text)
        self.assertLess(self.text.index("Run prospective capture gates"), self.text.index("Commit and push"))
        self.assertLess(self.text.index("Validate the complete immutable archive"), self.text.index("Commit and push"))


if __name__ == "__main__":
    unittest.main()
