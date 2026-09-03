"""Targeted offline tests for the scheduled ElectionSimulator publication."""

from __future__ import annotations

from copy import deepcopy
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np

from scripts.election_automation import (
    DAILY_SCHEDULE_UTC,
    ELECTION_DAY,
    BROWSER_SMOKE_TIMEOUT_SECONDS,
    JEKYLL_BUILD_TIMEOUT_SECONDS,
    INTRADAY_SCHEDULE_UTC,
    AutomationError,
    _run_command,
    automation_enabled_for_event,
    classify_run_type,
    current_stockholm_date,
    guard_election_date,
    _log_stage,
    latest_pop_observation_date,
    model_relevant_snapshot_sha256,
    refresh_polling_snapshot,
    resolve_mode,
    run_automation,
    run_production_event,
    run_website_checks,
    should_publish,
)
from scripts.forecast_history.campaign_paths import CampaignPathSimulation
from scripts.forecast_history.contract import DEFAULT_COALITIONS, build_groups_from_matrices, validate_history_contract
from scripts.forecast_history.generate import build_history, update_history_with_production_result
from scripts.publication_pipeline.pipeline import run_publication_pipeline
from scripts.site_publisher import GENERATION_FILES, publish_generation_to_site, sync_history_to_site
from scripts.static_exporter import validate_published_directory
from scripts.simulator.engine import SimulationResult, simulate_election
from scripts.simulator.summary import compute_simulation_summary


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40


class ElectionAutomationTests(unittest.TestCase):
    @staticmethod
    def _workflow_job(workflow: str, name: str) -> str:
        """Return one top-level workflow job without requiring a YAML package."""

        lines = workflow.splitlines()
        marker = f"  {name}:"
        try:
            start = lines.index(marker)
        except ValueError as exc:
            raise AssertionError(f"workflow job is missing: {name}") from exc
        end = len(lines)
        for index in range(start + 1, len(lines)):
            line = lines[index]
            if line.startswith("  ") and not line.startswith("    ") and line.endswith(":"):
                end = index
                break
        return "\n".join(lines[start:end])

    @staticmethod
    def _matrices() -> tuple[np.ndarray, np.ndarray]:
        votes = np.array(
            [
                [20, 5, 10, 5, 30, 10, 8, 12, 0],
                [18, 6, 11, 4, 32, 9, 7, 13, 0],
                [22, 4, 9, 6, 28, 11, 9, 11, 0],
                [19, 7, 12, 5, 31, 8, 6, 12, 5],
            ],
            dtype=float,
        )
        seats = np.array(
            [
                [40, 20, 30, 20, 80, 50, 30, 79],
                [39, 21, 31, 19, 82, 48, 31, 78],
                [42, 18, 29, 22, 78, 52, 29, 79],
                [41, 19, 32, 21, 81, 49, 28, 78],
            ],
            dtype=np.int64,
        )
        return votes, seats

    @classmethod
    def _result(cls, as_of: str) -> SimpleNamespace:
        votes, seats = cls._matrices()
        return SimpleNamespace(
            summary=SimpleNamespace(as_of=as_of, total_samples=len(votes)),
            vote_shares_matrix=votes,
            seats_matrix=seats,
            manifest={
                "source_git_commit": COMMIT,
                "source_worktree_clean": True,
                "base_seed": 12345,
            },
        )

    @classmethod
    def _projection_runner(cls, **kwargs) -> SimpleNamespace:
        """Return cheap joint draws through the dedicated projection seam."""

        votes, seats = cls._matrices()
        samples = int(kwargs["samples"])
        repeats = (samples + len(votes) - 1) // len(votes)
        return SimpleNamespace(
            summary=SimpleNamespace(as_of=kwargs["as_of"]),
            vote_shares_matrix=np.tile(votes, (repeats, 1))[:samples],
            seats_matrix=np.tile(seats, (repeats, 1))[:samples],
        )

    @classmethod
    def _campaign_path_simulator(cls, **kwargs) -> CampaignPathSimulation:
        """Return cheap opinion paths through the campaign-path seam.

        These orchestration tests stage placeholder model inputs, so the real
        simulator cannot run here and the scientific parity gate has nothing
        to verify.  The parity mathematics is covered end to end against the
        canonical engine in ``tests/test_campaign_paths.py``; this stub only
        keeps the *publication wiring* under test.
        """

        origin = date.fromisoformat(str(kwargs["as_of"]))
        election = date.fromisoformat(str(kwargs["election_date"]))
        path_days = (election - origin).days
        samples = int(kwargs["samples"])
        coalitions = kwargs["coalitions"]
        draws = {
            key: np.linspace(45.0, 55.0, samples, dtype=np.float64)[np.newaxis, :]
            .repeat(path_days + 1, axis=0)
            for key in coalitions
        }
        return CampaignPathSimulation(
            origin_date=origin,
            election_date=election,
            path_days=path_days,
            samples=samples,
            seed=int(kwargs["seed"]),
            day_dates=tuple(origin + timedelta(days=offset) for offset in range(path_days + 1)),
            coalition_draws=draws,
            representative_indices=tuple(range(min(4, samples))),
            endpoint_national_shares=np.full((samples, 9), 1.0 / 9.0),
            endpoint_opinion_composition=np.full((samples, 9), 100.0 / 9.0),
            diagnostics={
                "model_id": "coherent_campaign_paths_v1",
                "eligible_trajectories": 4357,
                "earliest_trajectory_start": "2014-09-15",
                "latest_trajectory_end": origin.isoformat(),
                "endpoint_horizon_days": min(path_days, 112),
                "time_warp": "identity" if path_days <= 112 else "monotone_stretch",
                "opinion_state_seed": 1,
                "dynamics_seed": 2,
                "election_noise_seed": 3,
                "endpoint_parity_verified": True,
                "endpoint_parity_max_abs_difference_pp": 0.0,
                "endpoint_parity_reference": "generate_national_vote_shares",
            },
        )

    @staticmethod
    def _init_git(root: Path) -> None:
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Automation Test"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "automation@example.test"], cwd=root, check=True)
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)

    @staticmethod
    def _write_model_inputs(root: Path, *, suffix: str = "") -> None:
        for relative in (
            "data/processed/pollofpolls/pollofpolls_timeseries.csv",
            "data/processed/pollofpolls/individual_polls.csv",
            "data/processed/pollofpolls/swedishpolls_individual_polls.csv",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"header\nvalue{suffix}\n", encoding="utf-8")

    @staticmethod
    def _change_normalized_poll_support(path: Path) -> None:
        """Change one real normalized poll value without changing its schema."""
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
        changed = False
        for row in rows:
            raw = row.get("support")
            if raw not in (None, ""):
                updated = f"{float(raw) + 0.1:.1f}"
                row["support"] = updated
                if row.get("source_value") not in (None, ""):
                    row["source_value"] = updated
                changed = True
                break
        if not changed:
            raise AssertionError(f"fixture has no numeric normalized poll support: {path}")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @classmethod
    def _production_result(cls, as_of: str) -> SimulationResult:
        """Build a compact, valid 100k-draw result for orchestration tests."""

        base = simulate_election(
            as_of=as_of,
            election_date="2026-09-13",
            samples=4,
            seed=12345,
        )
        votes = np.tile(base.vote_shares_matrix, (25_000, 1))
        seats = np.tile(base.seats_matrix, (25_000, 1))
        threshold_flags = np.tile(base.threshold_flags, (25_000, 1))
        manifest = dict(base.manifest)
        manifest.update(
            {
                "as_of": as_of,
                "samples": 100_000,
                "source_git_commit": COMMIT,
                "source_worktree_clean": True,
            }
        )
        summary, helper = compute_simulation_summary(
            as_of,
            "2026-09-13",
            votes,
            seats,
            manifest,
            local_12_pct_flags=np.zeros_like(threshold_flags, dtype=bool),
        )
        return SimulationResult(
            summary=summary,
            vote_shares_matrix=votes,
            seats_matrix=seats,
            threshold_flags=threshold_flags,
            largest_vote_parties=base.largest_vote_parties * 25_000,
            largest_seat_parties=base.largest_seat_parties * 25_000,
            group_helper=helper,
            manifest=manifest,
            quantization_audit=None,
        )

    @classmethod
    def _production_fixture(cls, parent: Path) -> tuple[Path, Path]:
        """Create two clean repositories with real validated artifacts.

        The large retrospective diagnostics are represented by read-only
        symlinks; polling, archive, and publication inputs are copied because
        the orchestration must be able to commit and validate them locally.
        """

        source = parent / "simulator"
        site = parent / "website"
        source.mkdir()
        site.mkdir()
        shutil.copytree(REPOSITORY_ROOT / "files/election-simulator", source / "files/election-simulator")
        shutil.copytree(REPOSITORY_ROOT / "files/election-simulator", site / "files/election-simulator")

        processed = source / "data/processed"
        processed.mkdir(parents=True)
        for directory in (
            "elections",
            "mandates",
            "geography",
            "seat_hindcasts",
            "vote_share_calibration",
            "pop_baseline_benchmark",
        ):
            (processed / directory).symlink_to(REPOSITORY_ROOT / "data/processed" / directory, target_is_directory=True)
        shutil.copytree(
            REPOSITORY_ROOT / "data/processed/pollofpolls",
            processed / "pollofpolls",
        )
        shutil.copytree(
            REPOSITORY_ROOT / "data/processed/prospective_forecasts",
            processed / "prospective_forecasts",
        )
        raw = source / "data/raw/pollofpolls"
        shutil.copytree(REPOSITORY_ROOT / "data/raw/pollofpolls", raw)
        shutil.copyfile(REPOSITORY_ROOT / "data/README.md", source / "data/README.md")
        cls._init_git(source)
        cls._init_git(site)
        return source, site

    @staticmethod
    def _git_status(root: Path) -> str:
        return subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    @staticmethod
    def _rewrite_certified_source_commit(publication_root: Path, source_commit: str) -> None:
        """Update only audit provenance while preserving publication contracts."""

        pointer_path = publication_root / "current.json"
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        version = publication_root / pointer["path"]
        metadata_path = version / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["source_git_commit"] = source_commit
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        def canonical(value):
            return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

        def without_runtime_timestamps(value):
            if isinstance(value, dict):
                return {
                    key: without_runtime_timestamps(item)
                    for key, item in value.items()
                    if key not in {"generated_at_utc", "published_at_utc", "updated_at_utc"}
                }
            if isinstance(value, list):
                return [without_runtime_timestamps(item) for item in value]
            return value

        manifest_path = version / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source_git_commit"] = source_commit
        manifest["publication_files"]["metadata.json"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        manifest["deterministic_content_hashes"]["metadata.json"] = hashlib.sha256(
            canonical(without_runtime_timestamps(metadata))
        ).hexdigest()
        manifest["deterministic_content_sha256"] = hashlib.sha256(
            canonical(manifest["deterministic_content_hashes"])
        ).hexdigest()
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        pointer["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        pointer_path.write_text(
            json.dumps(pointer, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def test_unchanged_source_content_does_not_trigger_intraday_simulation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_model_inputs(root)
            self._init_git(root)

            def refresh(raw, processed, **kwargs):
                return {"messages": []}

            polling = refresh_polling_snapshot(root, refresh_fn=refresh)
            self.assertEqual(polling.status, "SOURCE_UNCHANGED")
            self.assertFalse(polling.changed)
            self.assertFalse(should_publish("POLL_CHANGE", model_inputs_changed=polling.changed))
            self.assertEqual(
                subprocess.run(
                    ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
                ).stdout,
                "",
            )

            def unexpected_runner(**kwargs):
                raise AssertionError("unchanged intraday polling must not simulate")

            with patch(
                "scripts.election_automation.latest_pop_observation_date",
                return_value="2026-09-05",
            ):
                result = run_automation(
                    root,
                    site_repo=root,
                    schedule=INTRADAY_SCHEDULE_UTC,
                    now=datetime(2026, 9, 5, 8, tzinfo=timezone.utc),
                    automation_enabled="true",
                    refresh_fn=refresh,
                    simulation_runner=unexpected_runner,
                )
            self.assertEqual(result.status, "SOURCE_CHECKED")
            self.assertEqual(result.summary.deployment_status, "NO_PUBLICATION_NEEDED")
            self.assertEqual(self._git_status(root), "")

    def test_changed_model_relevant_polling_content_triggers_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_model_inputs(root)
            self._init_git(root)
            calls: list[dict[str, object]] = []

            def refresh(raw, processed, **kwargs):
                calls.append(kwargs)
                raw.mkdir(parents=True, exist_ok=True)
                path = processed / "individual_polls.csv"
                path.write_text("header\nchanged\n", encoding="utf-8")
                return {"messages": []}

            polling = refresh_polling_snapshot(root, refresh_fn=refresh)
            self.assertEqual(polling.status, "SOURCE_UPDATED")
            self.assertTrue(polling.changed)
            self.assertFalse(polling.installed)
            self.assertTrue(should_publish("POLL_CHANGE", model_inputs_changed=polling.changed))
            self.assertEqual(len(calls), 1)
            self.assertEqual(self._git_status(root), "")

    def test_changed_polling_fixture_runs_one_100k_publication_and_keeps_both_repos_clean(self) -> None:
        """Exercise acquire -> commit -> production -> mirror as one event."""

        with tempfile.TemporaryDirectory() as tmp:
            source, site = self._production_fixture(Path(tmp))
            calls: list[int] = []

            def refresh(raw, processed, **kwargs):
                raw.mkdir(parents=True, exist_ok=True)
                path = processed / "individual_polls.csv"
                self._change_normalized_poll_support(path)
                return {"messages": []}

            production_result = self._production_result("2026-09-05")

            def runner(**kwargs):
                calls.append(int(kwargs["samples"]))
                commit = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=source,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                production_result.manifest["source_git_commit"] = commit
                production_result.manifest["git_commit"] = commit
                return production_result

            projection_calls: list[int] = []

            def projection_runner(**kwargs):
                projection_calls.append(int(kwargs["dynamics_horizon_days"]))
                return self._projection_runner(**kwargs)

            with patch(
                "scripts.publication_pipeline.pipeline.DEFAULT_PROCESSED_ROOT",
                source / "data/processed",
            ):
                result = run_automation(
                    source,
                    site_repo=site,
                    schedule=INTRADAY_SCHEDULE_UTC,
                    now=datetime(2026, 9, 5, 8, tzinfo=timezone.utc),
                    automation_enabled="true",
                    commit=True,
                    refresh_fn=refresh,
                    simulation_runner=runner,
                    projection_runner=projection_runner,
                    campaign_path_simulator=self._campaign_path_simulator,
                    website_check_fn=lambda _: {"status": "PASS"},
                    generated_at_utc="2026-09-05T06:00:00+00:00",
                )

            self.assertEqual(result.status, "PUBLISHED")
            self.assertEqual(result.summary.run_type, "POLL_CHANGE")
            self.assertEqual(result.summary.simulation_samples, 100_000)
            self.assertEqual(calls, [100_000])
            self.assertEqual(projection_calls, list(range(7, -1, -1)))
            current = next(
                point for point in result.history["series"]
                if point["provenance"] == "current_production"
            )
            self.assertEqual(
                current["groups"],
                build_groups_from_matrices(
                    production_result.vote_shares_matrix,
                    production_result.seats_matrix,
                ),
            )
            self.assertEqual(self._git_status(source), "")
            self.assertEqual(self._git_status(site), "")
            source_pointer = json.loads(
                (source / "files/election-simulator/current.json").read_text(encoding="utf-8")
            )
            source_before = (source / "files/election-simulator/current.json").read_bytes()
            site_pointer = json.loads(
                (site / "files/election-simulator/current.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                source_pointer["publication_generation"],
                site_pointer["publication_generation"],
            )
            self.assertEqual(
                (source / "files/election-simulator/current.json").read_bytes(),
                (site / "files/election-simulator/current.json").read_bytes(),
            )
            source_history_path = (
                source / "files/election-simulator/history/coalition-timeseries.json"
            )
            site_history_path = (
                site / "files/election-simulator/history/coalition-timeseries.json"
            )
            source_history = json.loads(source_history_path.read_text(encoding="utf-8"))
            site_history = json.loads(site_history_path.read_text(encoding="utf-8"))
            self.assertIn("future_projection", result.history)
            self.assertEqual(source_history["future_projection"], result.history["future_projection"])
            self.assertEqual(site_history["future_projection"], result.history["future_projection"])
            self.assertEqual(source_history_path.read_bytes(), site_history_path.read_bytes())

    def test_changed_dry_run_runs_full_pipeline_once_without_dirtying_live_repos(self) -> None:
        """Dry-run consumes staged changed inputs and leaves both live repos clean."""

        with tempfile.TemporaryDirectory() as tmp:
            source, site = self._production_fixture(Path(tmp))
            before_source_pointer = (source / "files/election-simulator/current.json").read_bytes()
            before_site_pointer = (site / "files/election-simulator/current.json").read_bytes()
            calls: list[int] = []
            production_result = self._production_result("2026-09-05")

            def refresh(raw, processed, **kwargs):
                raw.mkdir(parents=True, exist_ok=True)
                self._change_normalized_poll_support(processed / "individual_polls.csv")
                return {"messages": []}

            def runner(**kwargs):
                calls.append(int(kwargs["samples"]))
                return production_result

            result = run_automation(
                source,
                site_repo=site,
                schedule=INTRADAY_SCHEDULE_UTC,
                now=datetime(2026, 9, 5, 8, tzinfo=timezone.utc),
                automation_enabled="true",
                refresh_fn=refresh,
                simulation_runner=runner,
                projection_runner=self._projection_runner,
                    campaign_path_simulator=self._campaign_path_simulator,
                website_check_fn=lambda _: {"status": "PASS"},
                mode="dry_run",
                generated_at_utc="2026-09-05T08:00:00+00:00",
            )
            self.assertEqual(result.status, "PUBLISHED")
            self.assertEqual(result.summary.deployment_status, "STAGED_NOT_INSTALLED")
            self.assertEqual(calls, [100_000])
            self.assertTrue(result.polling and result.polling.changed)
            self.assertEqual(
                next(point for point in result.history["series"] if point["provenance"] == "current_production")["groups"],
                build_groups_from_matrices(
                    production_result.vote_shares_matrix,
                    production_result.seats_matrix,
                ),
            )
            self.assertEqual(
                (source / "files/election-simulator/current.json").read_bytes(),
                before_source_pointer,
            )
            self.assertEqual(
                (site / "files/election-simulator/current.json").read_bytes(),
                before_site_pointer,
            )
            self.assertEqual(self._git_status(source), "")
            self.assertEqual(self._git_status(site), "")

    def test_dry_run_does_not_short_circuit_on_a_stale_website(self) -> None:
        """Dry-run always performs its one simulation, even when site lags."""

        with tempfile.TemporaryDirectory() as tmp:
            source, site = self._production_fixture(Path(tmp))
            source_before = (source / "files/election-simulator/current.json").read_bytes()
            generations = sorted(
                path.name for path in (site / "files/election-simulator/versions").iterdir()
            )
            old_generation = generations[0]
            old_manifest = site / "files/election-simulator/versions" / old_generation / "manifest.json"
            stale_pointer = {
                "schema_version": json.loads(old_manifest.read_text(encoding="utf-8"))["schema_version"],
                "publication_state": "COMPLETE",
                "publication_generation": old_generation,
                "path": f"versions/{old_generation}",
                "manifest_sha256": __import__("hashlib").sha256(old_manifest.read_bytes()).hexdigest(),
            }
            site_pointer_path = site / "files/election-simulator/current.json"
            site_pointer_path.write_text(json.dumps(stale_pointer, indent=2) + "\n", encoding="utf-8")
            subprocess.run(["git", "add", "files/election-simulator/current.json"], cwd=site, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture: make website stale"], cwd=site, check=True)
            site_before = site_pointer_path.read_bytes()
            calls: list[int] = []
            production_result = self._production_result("2026-09-05")

            def runner(**kwargs):
                calls.append(int(kwargs["samples"]))
                return production_result

            result = run_automation(
                source,
                site_repo=site,
                schedule=INTRADAY_SCHEDULE_UTC,
                now=datetime(2026, 9, 5, 8, tzinfo=timezone.utc),
                automation_enabled="true",
                mode="dry_run",
                refresh_fn=lambda raw, processed, **kwargs: {"messages": []},
                simulation_runner=runner,
                projection_runner=self._projection_runner,
                    campaign_path_simulator=self._campaign_path_simulator,
                website_check_fn=lambda _: {"status": "PASS"},
                generated_at_utc="2026-09-05T04:00:00+00:00",
            )
            self.assertEqual(result.status, "PUBLISHED")
            self.assertEqual(calls, [100_000])
            self.assertEqual(site_pointer_path.read_bytes(), site_before)
            self.assertEqual(
                (source / "files/election-simulator/current.json").read_bytes(),
                source_before,
            )
            self.assertEqual(self._git_status(source), "")
            self.assertEqual(self._git_status(site), "")

    def test_website_gate_failure_keeps_both_live_current_pointers_byte_identical(self) -> None:
        """A late website failure cannot publish a partially staged forecast."""

        with tempfile.TemporaryDirectory() as tmp:
            source, site = self._production_fixture(Path(tmp))
            source_current = source / "files/election-simulator/current.json"
            site_current = site / "files/election-simulator/current.json"
            source_before = source_current.read_bytes()
            site_before = site_current.read_bytes()

            def refresh(raw, processed, **kwargs):
                raw.mkdir(parents=True, exist_ok=True)
                path = processed / "individual_polls.csv"
                self._change_normalized_poll_support(path)
                return {"messages": []}

            production_result = self._production_result("2026-09-05")
            calls: list[int] = []

            def runner(**kwargs):
                calls.append(int(kwargs["samples"]))
                commit = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=source,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                production_result.manifest["source_git_commit"] = commit
                production_result.manifest["git_commit"] = commit
                return production_result

            def failed_website_check(_):
                raise AutomationError("browser smoke failed")

            with patch(
                "scripts.publication_pipeline.pipeline.DEFAULT_PROCESSED_ROOT",
                source / "data/processed",
            ):
                result = run_automation(
                    source,
                    site_repo=site,
                    schedule=INTRADAY_SCHEDULE_UTC,
                    now=datetime(2026, 9, 5, 8, tzinfo=timezone.utc),
                    automation_enabled="true",
                    commit=True,
                    refresh_fn=refresh,
                    simulation_runner=runner,
                    projection_runner=self._projection_runner,
                    campaign_path_simulator=self._campaign_path_simulator,
                    website_check_fn=failed_website_check,
                    generated_at_utc="2026-09-05T06:00:00+00:00",
                )

            self.assertEqual(result.status, "FAILED")
            self.assertIn("browser smoke failed", result.summary.failure or "")
            self.assertEqual(result.summary.simulation_samples, 100_000)
            self.assertIn("Simulation samples: 100000", result.summary.render())
            self.assertEqual(calls, [100_000])
            self.assertEqual(source_current.read_bytes(), source_before)
            self.assertEqual(site_current.read_bytes(), site_before)
            self.assertEqual(self._git_status(source), "")
            self.assertEqual(self._git_status(site), "")

    def test_daily_dry_run_publishes_without_poll_change(self) -> None:
        """Daily mode still executes one production event on an unchanged snapshot."""

        with tempfile.TemporaryDirectory() as tmp:
            source, site = self._production_fixture(Path(tmp))
            calls: list[int] = []
            production_result = self._production_result("2026-09-05")

            def refresh(raw, processed, **kwargs):
                return {"messages": []}

            def runner(**kwargs):
                calls.append(int(kwargs["samples"]))
                commit = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=source,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                production_result.manifest["source_git_commit"] = commit
                production_result.manifest["git_commit"] = commit
                return production_result

            with patch(
                "scripts.publication_pipeline.pipeline.DEFAULT_PROCESSED_ROOT",
                source / "data/processed",
            ):
                result = run_automation(
                    source,
                    site_repo=site,
                    schedule=DAILY_SCHEDULE_UTC,
                    now=datetime(2026, 9, 5, 4, tzinfo=timezone.utc),
                    automation_enabled="true",
                    refresh_fn=refresh,
                    simulation_runner=runner,
                    projection_runner=self._projection_runner,
                    campaign_path_simulator=self._campaign_path_simulator,
                    website_check_fn=lambda _: {"status": "PASS"},
                    generated_at_utc="2026-09-05T04:00:00+00:00",
                )

            self.assertEqual(result.status, "PUBLISHED")
            self.assertEqual(result.summary.run_type, "DAILY")
            self.assertEqual(result.summary.deployment_status, "STAGED_NOT_INSTALLED")
            self.assertEqual(calls, [100_000])
            self.assertTrue(result.polling and not result.polling.changed)
            self.assertEqual(self._git_status(source), "")
            self.assertEqual(self._git_status(site), "")

    def test_retained_verified_snapshot_is_not_reported_as_current(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_model_inputs(root)
            self._init_git(root)

            def refresh(raw, processed, **kwargs):
                return {"messages": ["timeseries: acquisition failed; retained verified raw file"]}

            polling = refresh_polling_snapshot(root, refresh_fn=refresh)
            self.assertEqual(polling.status, "SOURCE_UNAVAILABLE_USING_VERIFIED_SNAPSHOT")
            self.assertFalse(polling.changed)
            self.assertEqual(polling.source_provenance, "VERIFIED_STALE_FALLBACK")

    def test_summary_distinguishes_direct_live_and_verified_fallback_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_model_inputs(root)
            self._init_git(root)

            def live_refresh(raw, processed, **kwargs):
                return {"messages": [], "manifest": {"sources": {
                    "pop": {"retrieval_method": "direct_repository_http"},
                }}}

            with patch(
                "scripts.election_automation.latest_pop_observation_date",
                return_value="2026-09-05",
            ):
                live = run_automation(
                    root,
                    site_repo=root,
                    schedule=INTRADAY_SCHEDULE_UTC,
                    now=datetime(2026, 9, 5, 8, tzinfo=timezone.utc),
                    automation_enabled="true",
                    refresh_fn=live_refresh,
                    mode="probe",
                )
            self.assertEqual(live.summary.polling_source_provenance, "DIRECT_LIVE_FETCH")
            self.assertIn("Polling source provenance: DIRECT_LIVE_FETCH", live.summary.render())

            def fallback_refresh(raw, processed, **kwargs):
                return {"messages": ["first-party host unavailable; retained verified raw file"]}

            with patch(
                "scripts.election_automation.latest_pop_observation_date",
                return_value="2026-09-05",
            ):
                fallback = run_automation(
                    root,
                    site_repo=root,
                    schedule=INTRADAY_SCHEDULE_UTC,
                    now=datetime(2026, 9, 5, 8, tzinfo=timezone.utc),
                    automation_enabled="true",
                    refresh_fn=fallback_refresh,
                    mode="probe",
                )
            self.assertEqual(fallback.summary.polling_source_provenance, "VERIFIED_STALE_FALLBACK")
            self.assertIn("Polling source provenance: VERIFIED_STALE_FALLBACK", fallback.summary.render())

    def test_mixed_fallback_refresh_preserves_a_real_semantic_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_model_inputs(root)
            self._init_git(root)

            def mixed_refresh(raw, processed, **kwargs):
                (processed / "individual_polls.csv").write_text("header\nnew poll\n", encoding="utf-8")
                return {"messages": ["timeseries: first-party host unavailable; retained verified raw file"]}

            polling = refresh_polling_snapshot(root, refresh_fn=mixed_refresh)
            self.assertEqual(polling.status, "SOURCE_UNAVAILABLE_USING_VERIFIED_SNAPSHOT")
            self.assertTrue(polling.changed)
            self.assertEqual(polling.source_provenance, "VERIFIED_STALE_FALLBACK")

    def test_repository_kill_switch_stops_schedule_before_acquisition_but_manual_bypasses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            called = False

            def forbidden_refresh(*args, **kwargs):
                nonlocal called
                called = True
                raise AssertionError("kill switch must stop before acquisition")

            disabled = run_automation(
                root,
                site_repo=root,
                event_name="schedule",
                schedule=DAILY_SCHEDULE_UTC,
                mode="publish",
                commit=True,
                now=datetime(2026, 9, 5, 4, tzinfo=timezone.utc),
                automation_enabled="false",
                refresh_fn=forbidden_refresh,
            )
            self.assertEqual(disabled.status, "DISABLED_BY_REPOSITORY_KILL_SWITCH")
            self.assertFalse(called)
            self.assertEqual(disabled.summary.run_type, "DAILY")
            self.assertEqual(disabled.summary.deployment_status, "DISABLED_BY_REPOSITORY_KILL_SWITCH")
            self.assertEqual(disabled.summary.polling_source_provenance, "NOT_ACQUIRED")
            self.assertIn("Run type: DAILY", disabled.summary.render())
            self.assertIn(
                "Polling source status: DISABLED_BY_REPOSITORY_KILL_SWITCH",
                disabled.summary.render(),
            )
            self.assertIn("Simulation samples: 0", disabled.summary.render())
            self.assertTrue(automation_enabled_for_event(event_name="workflow_dispatch", enabled="false"))
            intraday_disabled = run_automation(
                root,
                site_repo=root,
                event_name="schedule",
                schedule=INTRADAY_SCHEDULE_UTC,
                mode="publish",
                commit=True,
                now=datetime(2026, 9, 5, 8, tzinfo=timezone.utc),
                automation_enabled="false",
                refresh_fn=forbidden_refresh,
            )
            self.assertEqual(intraday_disabled.status, "DISABLED_BY_REPOSITORY_KILL_SWITCH")
            self.assertEqual(intraday_disabled.summary.run_type, "POLL_CHANGE")
            self.assertIn("Run type: POLL_CHANGE", intraday_disabled.summary.render())
            with patch.dict(os.environ, {}, clear=True):
                self.assertFalse(automation_enabled_for_event(event_name="schedule"))

    def test_explicit_modes_are_unambiguous(self) -> None:
        self.assertEqual(resolve_mode(event_name="workflow_dispatch", mode="probe", commit=False, push=False), "probe")
        self.assertEqual(resolve_mode(event_name="workflow_dispatch", mode="dry_run", commit=False, push=False), "dry_run")
        self.assertEqual(resolve_mode(event_name="workflow_dispatch", mode="publish", commit=True, push=True), "publish")
        with self.assertRaises(AutomationError):
            resolve_mode(event_name="workflow_dispatch", mode=None, commit=False, push=False)
        with self.assertRaises(AutomationError):
            resolve_mode(event_name="workflow_dispatch", mode="dry_run", commit=True, push=False)
        for read_only_mode in ("probe", "dry_run"):
            with self.assertRaises(AutomationError):
                resolve_mode(
                    event_name="workflow_dispatch",
                    mode=read_only_mode,
                    commit=False,
                    push=True,
                )
        workflow = (REPOSITORY_ROOT / ".github/workflows/election-simulator-publication.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("type: choice", workflow)
        self.assertIn("- probe", workflow)
        self.assertIn("- dry_run", workflow)
        self.assertIn("- publish", workflow)
        self.assertNotIn("force_run", workflow)
        self.assertIn("ELECTION_AUTOMATION_ENABLED", workflow)
        self.assertIn('EVENT_SCHEDULE: ${{ github.event.schedule }}', workflow)
        self.assertIn('if [[ "$EVENT_SCHEDULE" == "0 4 * * *" ]]', workflow)
        self.assertIn("vars.ELECTION_AUTOMATION_ENABLED || 'false'", workflow)

    def test_workflow_partitions_permissions_and_website_secret_by_mode(self) -> None:
        workflow = (REPOSITORY_ROOT / ".github/workflows/election-simulator-publication.yml").read_text(
            encoding="utf-8"
        )
        workflow_defaults = workflow.split("\njobs:\n", 1)[0]
        probe = self._workflow_job(workflow, "probe")
        dry_run = self._workflow_job(workflow, "dry_run")
        browser_diagnostic = self._workflow_job(workflow, "browser_diagnostic")
        publish = self._workflow_job(workflow, "publish")

        self.assertIn("permissions:\n  contents: read", workflow_defaults)
        self.assertNotIn("contents: write", workflow_defaults)
        self.assertIn("permissions:\n      contents: read", probe)
        self.assertIn("permissions:\n      contents: read", dry_run)
        self.assertIn("permissions:\n      contents: read", browser_diagnostic)
        self.assertIn("permissions:\n      contents: write", publish)
        self.assertEqual(workflow.count("contents: write"), 1)

        self.assertIn(
            "if: github.event_name == 'workflow_dispatch' && github.event.inputs.mode == 'probe'",
            probe,
        )
        self.assertIn(
            "if: github.event_name == 'workflow_dispatch' && github.event.inputs.mode == 'dry_run'",
            dry_run,
        )
        self.assertIn(
            "if: github.event_name == 'schedule' || "
            "(github.event_name == 'workflow_dispatch' && github.event.inputs.mode == 'publish')",
            publish,
        )
        self.assertNotIn("github.event_name == 'schedule'", probe)
        self.assertNotIn("github.event_name == 'schedule'", dry_run)

        self.assertNotIn("WEBSITE_REPO_TOKEN", probe)
        self.assertNotIn("WEBSITE_REPO_TOKEN", dry_run)
        self.assertNotIn("WEBSITE_REPO_TOKEN", browser_diagnostic)
        self.assertEqual(publish.count("WEBSITE_REPO_TOKEN"), 1)
        self.assertEqual(workflow.count("WEBSITE_REPO_TOKEN"), 1)
        self.assertIn("persist-credentials: false", probe)
        self.assertIn("persist-credentials: false", dry_run)
        self.assertNotIn("persist-credentials: true", probe)
        self.assertNotIn("persist-credentials: true", dry_run)
        self.assertIn("Clone public website without credentials", dry_run)
        self.assertIn("https://github.com/edvinli/edvinli.github.io.git", dry_run)
        self.assertIn("GIT_CONFIG_GLOBAL: /dev/null", dry_run)
        self.assertIn("GIT_CONFIG_NOSYSTEM: \"1\"", dry_run)
        self.assertIn("--mode probe", probe)
        self.assertIn("--mode dry_run", dry_run)
        self.assertIn("--mode publish", publish)
        self.assertIn("github.event.inputs.mode == 'browser_diagnostic'", browser_diagnostic)
        self.assertIn("persist-credentials: false", browser_diagnostic)
        self.assertIn("forecast-timeseries.smoke.mjs", browser_diagnostic)
        self.assertIn("government-builder.smoke.mjs", browser_diagnostic)
        self.assertIn("--kill-after=5s 5m", browser_diagnostic)
        self.assertEqual(browser_diagnostic.count("--kill-after=5s 15m"), 2)
        self.assertIn("github.event.inputs.website_ref || 'master'", browser_diagnostic)
        self.assertIn("git -C website rev-parse HEAD", browser_diagnostic)
        self.assertIn('echo "Website commit: $WEBSITE_COMMIT" >> "$GITHUB_STEP_SUMMARY"', browser_diagnostic)
        self.assertNotIn("scripts.election_automation", browser_diagnostic)
        self.assertNotIn("100000", browser_diagnostic)
        self.assertNotIn("ELECTION_AUTOMATION_ENABLED", browser_diagnostic)
        self.assertIn("timeout-minutes: 45", probe)
        self.assertIn("timeout-minutes: 120", dry_run)
        self.assertIn("timeout-minutes: 120", publish)
        self.assertEqual(workflow.count("timeout-minutes: 120"), 2)
        self.assertNotIn("git push", probe)
        self.assertNotIn("git push", dry_run)
        self.assertIn("token: ${{ secrets.WEBSITE_REPO_TOKEN }}", publish)

    def test_stage_logger_flushes_start_and_elapsed_completion(self) -> None:
        with patch("builtins.print") as mocked_print:
            _log_stage("acquisition", "START")
            _log_stage("acquisition", "DONE", 1.25)

        mocked_print.assert_any_call("[election-automation] acquisition START", flush=True)
        mocked_print.assert_any_call(
            "[election-automation] acquisition DONE elapsed=1.250s",
            flush=True,
        )

    def test_website_checks_log_and_bound_each_required_command(self) -> None:
        events: list[tuple[str, str, float | None]] = []
        with patch("scripts.election_automation._run_command") as command:
            result = run_website_checks(
                Path("/website"),
                chrome_bin="/test/chromium",
                stage_callback=lambda stage, event, elapsed: events.append((stage, event, elapsed)),
            )

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(
            [(item.args[0], item.kwargs["name"], item.kwargs["timeout_seconds"]) for item in command.call_args_list],
            [
                (
                    ["jekyll", "build", "--config", "_config.yml,_config.dev.yml"],
                    "jekyll build",
                    JEKYLL_BUILD_TIMEOUT_SECONDS,
                ),
                (
                    ["node", "browser-tests/forecast-timeseries.smoke.mjs", "_site"],
                    "forecast-timeseries.smoke.mjs",
                    BROWSER_SMOKE_TIMEOUT_SECONDS,
                ),
                (
                    ["node", "browser-tests/government-builder.smoke.mjs", "_site"],
                    "government-builder.smoke.mjs",
                    BROWSER_SMOKE_TIMEOUT_SECONDS,
                ),
            ],
        )
        self.assertEqual(
            [(stage, event) for stage, event, _ in events],
            [
                ("jekyll build", "START"),
                ("jekyll build", "DONE"),
                ("forecast-timeseries.smoke.mjs", "START"),
                ("forecast-timeseries.smoke.mjs", "DONE"),
                ("government-builder.smoke.mjs", "START"),
                ("government-builder.smoke.mjs", "DONE"),
            ],
        )
        self.assertTrue(all(elapsed is not None for _, event, elapsed in events if event == "DONE"))

    def test_website_command_streams_output_with_a_sanitized_environment(self) -> None:
        process = Mock(pid=4321, returncode=0)
        process.communicate.return_value = (None, None)
        with patch("scripts.election_automation.subprocess.Popen", return_value=process) as popen:
            _run_command(
                ["node", "browser-tests/forecast-timeseries.smoke.mjs", "_site"],
                name="forecast-timeseries.smoke.mjs",
                timeout_seconds=900,
                cwd=Path("/website"),
                env={
                    "PATH": "/bin",
                    "CHROME_BIN": "/test/chromium",
                    "WEBSITE_REPO_TOKEN": "must-not-reach-child",
                    "AWS_ACCESS_KEY_ID": "must-not-reach-child",
                    "SAFE_VALUE": "visible",
                },
            )

        child_env = popen.call_args.kwargs["env"]
        self.assertEqual(child_env["PATH"], "/bin")
        self.assertEqual(child_env["CHROME_BIN"], "/test/chromium")
        self.assertEqual(child_env["SAFE_VALUE"], "visible")
        self.assertNotIn("WEBSITE_REPO_TOKEN", child_env)
        self.assertNotIn("AWS_ACCESS_KEY_ID", child_env)
        self.assertIsNone(popen.call_args.kwargs["stdout"])
        self.assertIsNone(popen.call_args.kwargs["stderr"])

    def test_website_command_timeout_names_command_elapsed_and_terminates_group(self) -> None:
        process = Mock(pid=4321, returncode=None)
        process.communicate.side_effect = [
            subprocess.TimeoutExpired(["node", "fixture"], 900),
            ("", ""),
        ]
        with (
            patch("scripts.election_automation.subprocess.Popen", return_value=process),
            patch("scripts.election_automation.os.killpg") as killpg,
            patch("scripts.election_automation.monotonic", side_effect=[10.0, 11.25]),
        ):
            with self.assertRaisesRegex(
                AutomationError,
                r"^website command timed out: forecast-timeseries\.smoke\.mjs after 1\.250s$",
            ):
                _run_command(
                    ["node", "browser-tests/forecast-timeseries.smoke.mjs", "_site"],
                    name="forecast-timeseries.smoke.mjs",
                    timeout_seconds=900,
                    cwd=Path("/website"),
                )

        self.assertEqual(
            killpg.call_args_list,
            [call(4321, signal.SIGTERM), call(4321, signal.SIGKILL)],
        )
        self.assertEqual(process.communicate.call_count, 2)

    def test_website_command_timeout_escalates_to_kill_for_stuck_children(self) -> None:
        process = Mock(pid=4321, returncode=None)
        process.communicate.side_effect = [
            subprocess.TimeoutExpired(["node", "fixture"], 900),
            subprocess.TimeoutExpired(["node", "fixture"], 5),
            ("", ""),
        ]
        with (
            patch("scripts.election_automation.subprocess.Popen", return_value=process),
            patch("scripts.election_automation.os.killpg") as killpg,
            patch("scripts.election_automation.monotonic", side_effect=[20.0, 21.0]),
        ):
            with self.assertRaisesRegex(AutomationError, r"government-builder\.smoke\.mjs after 1\.000s"):
                _run_command(
                    ["node", "browser-tests/government-builder.smoke.mjs", "_site"],
                    name="government-builder.smoke.mjs",
                    timeout_seconds=900,
                    cwd=Path("/website"),
                )

        self.assertEqual(
            killpg.call_args_list,
            [call(4321, signal.SIGTERM), call(4321, signal.SIGKILL)],
        )
        self.assertEqual(process.communicate.call_count, 3)

    def test_website_command_failure_names_command_elapsed_and_cleans_children(self) -> None:
        process = Mock(pid=4321, returncode=7)
        process.communicate.side_effect = [("", ""), ("", "")]
        with (
            patch("scripts.election_automation.subprocess.Popen", return_value=process),
            patch("scripts.election_automation.os.killpg") as killpg,
            patch("scripts.election_automation.monotonic", side_effect=[30.0, 30.5]),
        ):
            with self.assertRaisesRegex(
                AutomationError,
                r"^website command failed: jekyll build \(exit code 7\) after 0\.500s$",
            ):
                _run_command(
                    ["jekyll", "build"],
                    name="jekyll build",
                    timeout_seconds=300,
                    cwd=Path("/website"),
                )

        self.assertEqual(
            killpg.call_args_list,
            [call(4321, signal.SIGTERM), call(4321, signal.SIGKILL)],
        )

    @unittest.skipUnless(os.name == "posix", "process-group cleanup is a POSIX runner contract")
    def test_website_command_timeout_reaps_a_real_child_process(self) -> None:
        parent_source = """
import signal
import subprocess
import sys
import time
from pathlib import Path

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
Path(sys.argv[1]).write_text(str(child.pid), encoding="utf-8")

def stop(_signum, _frame):
    try:
        child.wait(timeout=2)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
    raise SystemExit(143)

signal.signal(signal.SIGTERM, stop)
time.sleep(60)
"""
        with tempfile.TemporaryDirectory() as tmp:
            child_pid_path = Path(tmp) / "child.pid"
            with self.assertRaisesRegex(AutomationError, r"real child fixture after [0-9.]+s"):
                _run_command(
                    [sys.executable, "-c", parent_source, str(child_pid_path)],
                    name="real child fixture",
                    timeout_seconds=1,
                    cwd=Path(tmp),
                )

            child_pid = int(child_pid_path.read_text(encoding="utf-8"))
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)

    def test_cross_repository_consumer_testing_is_explicitly_opt_in(self) -> None:
        from tests._website_repo import DEFAULT_WEBSITE_REPO, ENV_OVERRIDE, website_repo

        self.assertEqual(website_repo(), DEFAULT_WEBSITE_REPO)
        self.assertNotEqual(DEFAULT_WEBSITE_REPO, Path.home() / "Documents" / "Git" / "edvinli.github.io")
        self.assertIn(ENV_OVERRIDE, (REPOSITORY_ROOT / "docs/election_simulator_automation.md").read_text())

    def test_committed_polling_without_publication_forces_next_unchanged_retry(self) -> None:
        """A durable polling commit is retried after the first publish fails."""

        with tempfile.TemporaryDirectory() as tmp:
            source, site = self._production_fixture(Path(tmp))
            baseline_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=source, check=True,
                capture_output=True, text=True,
            ).stdout.strip()
            self._rewrite_certified_source_commit(
                source / "files/election-simulator", baseline_commit
            )
            self._rewrite_certified_source_commit(
                site / "files/election-simulator", baseline_commit
            )
            subprocess.run(["git", "add", "files/election-simulator"], cwd=source, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture: certify baseline"], cwd=source, check=True)
            subprocess.run(["git", "add", "files/election-simulator"], cwd=site, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture: certify baseline"], cwd=site, check=True)

            changed_poll = source / "data/processed/pollofpolls/individual_polls.csv"
            self._change_normalized_poll_support(changed_poll)
            subprocess.run(["git", "add", str(changed_poll.relative_to(source))], cwd=source, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture: committed polling refresh"], cwd=source, check=True)

            calls: list[int] = []
            production_result = self._production_result("2026-09-05")

            def refresh(raw, processed, **kwargs):
                return {"messages": []}

            def failing_runner(**kwargs):
                calls.append(int(kwargs["samples"]))
                raise AssertionError("simulated production failure")

            with patch(
                "scripts.publication_pipeline.pipeline.DEFAULT_PROCESSED_ROOT",
                source / "data/processed",
            ):
                first = run_automation(
                    source,
                    site_repo=site,
                    schedule=INTRADAY_SCHEDULE_UTC,
                    now=datetime(2026, 9, 5, 8, tzinfo=timezone.utc),
                    automation_enabled="true",
                    mode="publish",
                    commit=True,
                    refresh_fn=refresh,
                    simulation_runner=failing_runner,
                    website_check_fn=lambda _: {"status": "PASS"},
                    generated_at_utc="2026-09-05T08:00:00+00:00",
                )
            self.assertEqual(first.status, "FAILED")
            self.assertEqual(first.summary.recovery_status, "POLLING_PUBLICATION_PENDING")
            self.assertEqual(self._git_status(source), "")

            def succeeding_runner(**kwargs):
                calls.append(int(kwargs["samples"]))
                commit = subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=source, check=True,
                    capture_output=True, text=True,
                ).stdout.strip()
                production_result.manifest["source_git_commit"] = commit
                production_result.manifest["git_commit"] = commit
                return production_result

            with patch(
                "scripts.publication_pipeline.pipeline.DEFAULT_PROCESSED_ROOT",
                source / "data/processed",
            ):
                second = run_automation(
                    source,
                    site_repo=site,
                    schedule=INTRADAY_SCHEDULE_UTC,
                    now=datetime(2026, 9, 5, 8, tzinfo=timezone.utc),
                    automation_enabled="true",
                    mode="publish",
                    commit=True,
                    refresh_fn=refresh,
                    simulation_runner=succeeding_runner,
                    projection_runner=self._projection_runner,
                    campaign_path_simulator=self._campaign_path_simulator,
                    website_check_fn=lambda _: {"status": "PASS"},
                    generated_at_utc="2026-09-05T08:05:00+00:00",
                )
            self.assertEqual(second.status, "PUBLISHED")
            self.assertEqual(calls, [100_000, 100_000])
            self.assertEqual(self._git_status(source), "")
            self.assertEqual(self._git_status(site), "")

    def test_website_recovery_syncs_certified_generation_without_simulation(self) -> None:
        """A source-ahead/site-behind retry mirrors bytes without a second run."""

        with tempfile.TemporaryDirectory() as tmp:
            source, site = self._production_fixture(Path(tmp))
            old_site_tree = Path(tmp) / "old-site-publication"
            shutil.copytree(site / "files/election-simulator", old_site_tree)
            production_result = self._production_result("2026-09-05")

            def runner(**kwargs):
                commit = subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=source, check=True,
                    capture_output=True, text=True,
                ).stdout.strip()
                production_result.manifest["source_git_commit"] = commit
                production_result.manifest["git_commit"] = commit
                return production_result

            with patch(
                "scripts.publication_pipeline.pipeline.DEFAULT_PROCESSED_ROOT",
                source / "data/processed",
            ):
                first = run_production_event(
                    source,
                    site_repo=site,
                    forecast_as_of="2026-09-05",
                    simulation_runner=runner,
                    projection_runner=self._projection_runner,
                    campaign_path_simulator=self._campaign_path_simulator,
                    website_check_fn=lambda _: {"status": "PASS"},
                    generated_at_utc="2026-09-05T09:00:00+00:00",
                    commit=True,
                    push=False,
                    allow_duplicate_payload=True,
                )
            new_source_pointer = (source / "files/election-simulator/current.json").read_bytes()
            shutil.rmtree(site / "files/election-simulator")
            shutil.copytree(old_site_tree, site / "files/election-simulator")
            subprocess.run(["git", "add", "-A", "files/election-simulator"], cwd=site, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture: simulate website push failure"], cwd=site, check=True)
            self.assertNotEqual(
                (site / "files/election-simulator/current.json").read_bytes(),
                new_source_pointer,
            )

            calls = 0

            def forbidden_runner(**kwargs):
                nonlocal calls
                calls += 1
                raise AssertionError("website recovery must not simulate")

            recovered = run_automation(
                source,
                site_repo=site,
                schedule=INTRADAY_SCHEDULE_UTC,
                now=datetime(2026, 9, 5, 10, tzinfo=timezone.utc),
                automation_enabled="true",
                mode="publish",
                commit=True,
                refresh_fn=lambda raw, processed, **kwargs: {"messages": []},
                simulation_runner=forbidden_runner,
                website_check_fn=lambda _: {"status": "PASS"},
            )
            self.assertEqual(recovered.status, "WEBSITE_RECOVERED")
            self.assertEqual(calls, 0)
            self.assertEqual(
                (site / "files/election-simulator/current.json").read_bytes(),
                new_source_pointer,
            )
            self.assertEqual(self._git_status(source), "")
            self.assertEqual(self._git_status(site), "")

    def test_daily_mode_publishes_without_polling_change(self) -> None:
        self.assertTrue(should_publish("DAILY", model_inputs_changed=False))
        self.assertFalse(should_publish("POLL_CHANGE", model_inputs_changed=False))
        self.assertTrue(should_publish("MANUAL", model_inputs_changed=False, mode="publish"))

    def test_workflow_configures_git_identity_for_both_checkouts(self) -> None:
        workflow = (REPOSITORY_ROOT / ".github/workflows/election-simulator-publication.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn('git -C "$repo" config user.name "github-actions[bot]"', workflow)
        self.assertIn(
            'git -C "$repo" config user.email "41898282+github-actions[bot]@users.noreply.github.com"',
            workflow,
        )
        self.assertIn('for repo in simulator website; do', workflow)
        self.assertIn('test "$(git -C website config user.name)" = "github-actions[bot]"', workflow)
        self.assertIn(
            'test "$(git -C website config user.email)" = "41898282+github-actions[bot]@users.noreply.github.com"',
            workflow,
        )

    def test_run_type_and_stockholm_date_guard(self) -> None:
        self.assertEqual(classify_run_type(schedule=DAILY_SCHEDULE_UTC), "DAILY")
        self.assertEqual(classify_run_type(schedule=INTRADAY_SCHEDULE_UTC), "POLL_CHANGE")
        self.assertEqual(classify_run_type(event_name="workflow_dispatch"), "MANUAL")
        # 22:30 UTC is 00:30 the next calendar day in Stockholm during DST.
        self.assertEqual(
            current_stockholm_date(datetime(2026, 8, 31, 22, 30, tzinfo=timezone.utc)).isoformat(),
            "2026-09-01",
        )
        with self.assertRaises(Exception):
            guard_election_date(ELECTION_DAY.replace(day=14))

    def test_future_as_of_uses_latest_available_pop_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeseries.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["date", "M"])
                writer.writerow(["2026-09-04", "20"])
                writer.writerow(["2026-09-05", "21"])
            self.assertEqual(
                latest_pop_observation_date(path, as_of="2026-09-06"),
                "2026-09-05",
            )

    def test_date_guard_stops_before_acquisition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            called = False

            def no_acquisition(*args, **kwargs):
                nonlocal called
                called = True
                raise AssertionError("date guard must run before acquisition")

            with patch("scripts.election_automation.refresh_polling_snapshot", no_acquisition):
                result = run_automation(
                    tmp,
                    site_repo=tmp,
                    now=datetime(2026, 9, 14, 8, tzinfo=timezone.utc),
                )
            self.assertEqual(result.status, "STOPPED_AFTER_ELECTION")
            self.assertFalse(called)

    def test_existing_reconstructed_points_are_reused_without_rerunning(self) -> None:
        existing = json.loads(
            (REPOSITORY_ROOT / "files/election-simulator/history/coalition-timeseries.json").read_text()
        )

        def unexpected_runner(**kwargs):
            raise AssertionError(f"existing point was rerun: {kwargs}")

        rebuilt = build_history(
            election_date="2026-09-13",
            dates=[point["date"] for point in existing["series"]],
            existing_payload=existing,
            poll_file=REPOSITORY_ROOT / "data/processed/pollofpolls/swedishpolls_individual_polls.csv",
            timeseries_file=REPOSITORY_ROOT / "data/processed/pollofpolls/pollofpolls_timeseries.csv",
            archive_dir=REPOSITORY_ROOT / "data/processed/prospective_forecasts",
            model_commit=existing["model_commit"],
            simulation_runner=unexpected_runner,
            workers=1,
        )
        before = [
            point
            for point in existing["series"]
            if point["provenance"] == "reconstructed_current_model"
        ]
        after = [
            point
            for point in rebuilt["series"]
            if point["provenance"] == "reconstructed_current_model"
        ]
        self.assertEqual(after, before)

    def test_production_history_rollover_and_same_day_replacement(self) -> None:
        existing = json.loads(
            (REPOSITORY_ROOT / "files/election-simulator/history/coalition-timeseries.json").read_text()
        )
        # Rollover is a property of the updater, not of whatever the committed
        # artifact happens to hold. A freshly generated history deliberately
        # carries no certified point -- when the archive already has a snapshot
        # for the latest date at the production draw count, build_history uses
        # that record verbatim -- so the prior certified point is established
        # here rather than assumed. Depending on the shipped artifact's shape
        # made this test fail on a legitimate regeneration.
        existing = update_history_with_production_result(
            existing,
            self._result("2026-08-26"),
            poll_file=REPOSITORY_ROOT / "data/processed/pollofpolls/swedishpolls_individual_polls.csv",
            timeseries_file=REPOSITORY_ROOT / "data/processed/pollofpolls/pollofpolls_timeseries.csv",
            archive_dir=REPOSITORY_ROOT / "data/processed/prospective_forecasts",
            election_date="2026-09-13",
            publication_generation="seed-generation",
            deterministic_payload_sha256="a" * 64,
            generated_at_utc="2026-08-31T21:00:00+00:00",
            model_commit=COMMIT,
            source_worktree_clean=True,
        )
        self.assertEqual(
            sum(point["provenance"] == "current_production" for point in existing["series"]), 1,
            "the seeded payload must carry exactly one certified point",
        )
        # Every reconstructed point *other than the one this update replaces*
        # must survive byte for byte. Excluding the target date explicitly is
        # the actual invariant; the previous version relied on that date not
        # being covered by the artifact, which a denser regeneration changes.
        rollover_date = "2026-08-25"
        reconstructed = {
            point["date"]: deepcopy(point)
            for point in existing["series"]
            if point["provenance"] == "reconstructed_current_model"
            and point["date"] != rollover_date
        }
        new_day = update_history_with_production_result(
            existing,
            self._result(rollover_date),
            poll_file=REPOSITORY_ROOT / "data/processed/pollofpolls/swedishpolls_individual_polls.csv",
            timeseries_file=REPOSITORY_ROOT / "data/processed/pollofpolls/pollofpolls_timeseries.csv",
            archive_dir=REPOSITORY_ROOT / "data/processed/prospective_forecasts",
            election_date="2026-09-13",
            publication_generation="new-generation",
            deterministic_payload_sha256="b" * 64,
            generated_at_utc="2026-08-31T22:00:00+00:00",
            model_commit=COMMIT,
            source_worktree_clean=True,
        )
        self.assertEqual(sum(point["provenance"] == "current_production" for point in new_day["series"]), 1)
        self.assertEqual(len({point["date"] for point in new_day["series"]}), len(new_day["series"]))
        old_current = next(point for point in existing["series"] if point["provenance"] == "current_production")
        rolled = next(point for point in new_day["series"] if point["date"] == old_current["date"])
        self.assertEqual(rolled["provenance"], "prospective_archived")
        self.assertEqual(
            {point["date"]: point for point in new_day["series"] if point["date"] in reconstructed},
            reconstructed,
        )

        same_day = update_history_with_production_result(
            existing,
            self._result(old_current["date"]),
            poll_file=REPOSITORY_ROOT / "data/processed/pollofpolls/swedishpolls_individual_polls.csv",
            timeseries_file=REPOSITORY_ROOT / "data/processed/pollofpolls/pollofpolls_timeseries.csv",
            archive_dir=REPOSITORY_ROOT / "data/processed/prospective_forecasts",
            election_date="2026-09-13",
            publication_generation="same-day-new-generation",
            deterministic_payload_sha256="c" * 64,
            generated_at_utc="2026-08-31T23:00:00+00:00",
            model_commit=COMMIT,
            source_worktree_clean=True,
        )
        self.assertEqual(len(same_day["series"]), len(existing["series"]))
        self.assertEqual(len({point["date"] for point in same_day["series"]}), len(same_day["series"]))
        self.assertEqual(
            next(point for point in same_day["series"] if point["date"] == old_current["date"])["provenance"],
            "current_production",
        )

    def test_coalition_quantities_use_same_joint_draws(self) -> None:
        existing = json.loads(
            (REPOSITORY_ROOT / "files/election-simulator/history/coalition-timeseries.json").read_text()
        )
        result = self._result("2026-08-25")
        updated = update_history_with_production_result(
            existing,
            result,
            poll_file=REPOSITORY_ROOT / "data/processed/pollofpolls/swedishpolls_individual_polls.csv",
            timeseries_file=REPOSITORY_ROOT / "data/processed/pollofpolls/pollofpolls_timeseries.csv",
            archive_dir=REPOSITORY_ROOT / "data/processed/prospective_forecasts",
            publication_generation="joint-generation",
            deterministic_payload_sha256="d" * 64,
            generated_at_utc="2026-08-31T22:00:00+00:00",
            model_commit=COMMIT,
            source_worktree_clean=True,
        )
        current = next(point for point in updated["series"] if point["date"] == "2026-08-25")
        self.assertEqual(current["groups"], build_groups_from_matrices(result.vote_shares_matrix, result.seats_matrix))
        validate_history_contract(updated)

    def test_simulator_and_website_history_and_publication_bytes_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "simulator"
            site = root / "website"
            shutil.copytree(REPOSITORY_ROOT / "files/election-simulator", source / "files/election-simulator")
            shutil.copytree(REPOSITORY_ROOT / "files/election-simulator", site / "files/election-simulator")
            # The site starts with a prior generation; mirroring a new source
            # generation exercises the immutable-copy path and history sync.
            source_pointer = json.loads((source / "files/election-simulator/current.json").read_text())
            generation = source_pointer["publication_generation"]
            (site / "files/election-simulator/versions" / generation).rename(
                site / "files/election-simulator/versions" / (generation + "-old")
            )
            source_history = source / "files/election-simulator/history/coalition-timeseries.json"
            site_history = site / "files/election-simulator/history/coalition-timeseries.json"
            site_history.unlink()
            publish_generation_to_site(site_repo=site, source_publication_dir=source / "files/election-simulator", generation=generation)
            sync_history_to_site(site_repo=site, source_history_path=source_history)
            validate_published_directory(site / "files/election-simulator")
            for filename in GENERATION_FILES:
                self.assertEqual(
                    (source / "files/election-simulator/versions" / generation / filename).read_bytes(),
                    (site / "files/election-simulator/versions" / generation / filename).read_bytes(),
                )
            self.assertEqual(source_history.read_bytes(), site_history.read_bytes())

    def test_one_production_publication_requests_one_100k_simulation(self) -> None:
        calls: list[int] = []

        base = simulate_election(
            as_of="2026-09-05",
            election_date="2026-09-13",
            samples=4,
            seed=12345,
        )
        # A compact deterministic fake retains the exact result shape and
        # canonical summary contracts while avoiding a multi-minute allocator
        # run inside the unit suite.  The production boundary still requests
        # (and asserts) exactly 100,000 draws.
        votes = np.tile(base.vote_shares_matrix, (25_000, 1))
        seats = np.tile(base.seats_matrix, (25_000, 1))
        threshold_flags = np.tile(base.threshold_flags, (25_000, 1))
        manifest = dict(base.manifest)
        manifest["as_of"] = "2026-09-05"
        summary, helper = compute_simulation_summary(
            "2026-09-05",
            "2026-09-13",
            votes,
            seats,
            manifest,
            local_12_pct_flags=np.zeros_like(threshold_flags, dtype=bool),
        )
        production_result = SimulationResult(
            summary=summary,
            vote_shares_matrix=votes,
            seats_matrix=seats,
            threshold_flags=threshold_flags,
            largest_vote_parties=base.largest_vote_parties * 25_000,
            largest_seat_parties=base.largest_seat_parties * 25_000,
            group_helper=helper,
            manifest=manifest,
            quantization_audit=None,
        )

        def runner(**kwargs):
            calls.append(int(kwargs["samples"]))
            return production_result

        run = run_publication_pipeline(
            as_of="2026-09-05",
            election_date="2026-09-13",
            samples=100_000,
            seed=12345,
            processed_root=REPOSITORY_ROOT / "data/processed",
            append_archive=False,
            export_publication=False,
            simulation_runner=runner,
        )
        self.assertEqual(run.status, "SIMULATED")
        self.assertEqual(calls, [100_000])
        self.assertIsNotNone(run.simulation_result)

    def test_failure_before_completion_leaves_previous_live_pointer_intact(self) -> None:
            # The staging behavior is exercised by the site publisher contract:
        # an invalid source is rejected before destination/current.json exists.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            site = root / "site"
            (site / "files").mkdir(parents=True)
            shutil.copytree(REPOSITORY_ROOT / "files/election-simulator", source / "files/election-simulator")
            # Seed the website with a certified prior publication, then make a
            # staged source fail validation.  The previous pointer is retained.
            pointer_source = json.loads((source / "files/election-simulator/current.json").read_text())
            generation = pointer_source["publication_generation"]
            publish_generation_to_site(site_repo=site, source_publication_dir=source / "files/election-simulator")
            pointer_path = site / "files/election-simulator/current.json"
            pointer_before = pointer_path.read_bytes()
            bad_manifest = source / "files/election-simulator/versions" / generation / "manifest.json"
            original = bad_manifest.read_bytes()
            bad_manifest.write_text("{\"publication_state\":\"PENDING\"}\n", encoding="utf-8")
            try:
                with self.assertRaises(Exception):
                    publish_generation_to_site(
                        site_repo=site,
                        source_publication_dir=source / "files/election-simulator",
                        generation=generation,
                    )
            finally:
                bad_manifest.write_bytes(original)
            self.assertEqual(pointer_path.read_bytes(), pointer_before)


if __name__ == "__main__":
    unittest.main()
