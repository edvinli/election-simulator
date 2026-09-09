"""Targeted offline tests for the scheduled ElectionSimulator publication."""

from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
from datetime import date, datetime, timedelta, timezone
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np

from scripts import election_automation_base as base
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
from scripts.forecast_history.effective_inputs import EffectiveInputs, previous_fingerprints
from scripts.forecast_history.generate import (
    DEFAULT_SIMULATION_SEED,
    build_history,
    missing_curve_dates,
    update_history_with_production_result,
)
from scripts.publication_pipeline.pipeline import run_publication_pipeline
from scripts.site_publisher import GENERATION_FILES, publish_generation_to_site, sync_history_to_site
from scripts.static_exporter import validate_published_directory
from scripts.simulator.engine import SimulationResult, simulate_election
from scripts.simulator.reproducibility import compute_file_sha256
from scripts.rendering import (
    CertifiedGenerationError,
    load_certified_generation,
)
from scripts.simulator.exact_draw_sidecar import (
    SIDECAR_DRAWS_FILENAME,
    SIDECAR_METADATA_FILENAME,
    collect_latest_certified_generation,
    load_verified_draw_sidecar,
    validate_exact_draw_sidecar,
)
from scripts.simulator.summary import compute_simulation_summary
from tests.history_fixtures import (
    FROZEN_AS_OF,
    FROZEN_ELECTION_DATE,
    freeze_poll_inputs,
    make_history_fixture,
)
from tests.site_publication_fixture import (
    export_frozen_site_publication,
    frozen_site_generation,
    install_frozen_site_publication,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40

# The publication fixture is frozen (see tests/history_fixtures.py): the
# history it rolls a certified point into, and the polling inputs that history
# agrees with, are built by the tests rather than read from the committed
# artifact. Deriving the forecast date from the artifact removed the hardcoded
# drift that broke these tests once; freezing the inputs removes the coupling
# itself. One integration test still exercises the live committed artifact.
LIVE_HISTORY_ARTIFACT = (
    REPOSITORY_ROOT / "files" / "election-simulator" / "history" / "coalition-timeseries.json"
)
FORECAST_AS_OF = FROZEN_AS_OF
FORECAST_DAY = date.fromisoformat(FORECAST_AS_OF)


def _reuse_partition(existing: dict) -> tuple[set[str], set[str]]:
    """Split an artifact's reconstructed dates by whether its inputs still hold.

    Derived with the very mechanism ``build_history`` gates reuse on rather
    than restated, so it cannot drift from the rule under test and cannot
    freeze today's data into an expectation.

    Returns ``(input_stable, input_revised)``: dates whose recorded
    effective-inputs fingerprint still matches what the current sources
    produce, and dates where it genuinely differs.
    """

    recorded = previous_fingerprints(existing)
    inputs = EffectiveInputs(
        REPOSITORY_ROOT / "data/processed",
        election_date=date.fromisoformat(existing["election_date"]),
        seed=int(existing["model"]["seed"]),
    )
    stable: set[str] = set()
    revised: set[str] = set()
    for point in existing["series"]:
        if point["provenance"] != "reconstructed_current_model":
            continue
        day = str(point["date"])
        current = inputs.fingerprint(date.fromisoformat(day))
        (stable if recorded.get(day) == current else revised).add(day)
    return stable, revised


def _forecast_utc(hour: int, minute: int = 0) -> datetime:
    """A UTC instant on the forecast day the committed artifact implies."""

    return datetime(
        FORECAST_DAY.year, FORECAST_DAY.month, FORECAST_DAY.day, hour, minute, tzinfo=timezone.utc
    )


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
            votes / 100.0,
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
        # The frozen publication from tests/fixtures/site_publication, not the
        # committed site tree: the prior generation these tests start from must
        # not be whatever production last published, or a synthetic generation
        # id can sort behind it.
        for repository in (source, site):
            install_frozen_site_publication(repository / "files/election-simulator")

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

        # Freeze the inputs the publication reads. The committed artifact and
        # the committed polling CSVs both move with every publication; a test
        # that rolls its certified point into them fails when they move, for
        # reasons unrelated to what it asserts.
        frozen_history = json.dumps(make_history_fixture(), separators=(",", ":"), sort_keys=False)
        for repository in (source, site):
            history_path = repository / "files/election-simulator/history/coalition-timeseries.json"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(frozen_history, encoding="utf-8")
        freeze_poll_inputs(processed / "pollofpolls")

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

            production_result = self._production_result(FORECAST_AS_OF)

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
                    now=_forecast_utc(8),
                    automation_enabled="true",
                    commit=True,
                    refresh_fn=refresh,
                    simulation_runner=runner,
                    projection_runner=projection_runner,
                    campaign_path_simulator=self._campaign_path_simulator,
                    website_check_fn=lambda _: {"status": "PASS"},
                    generated_at_utc=f"{FORECAST_AS_OF}T06:00:00+00:00",
                )

            self.assertEqual(result.status, "PUBLISHED")
            self.assertEqual(result.summary.run_type, "POLL_CHANGE")
            self.assertEqual(result.summary.simulation_samples, 100_000)
            self.assertEqual(calls, [100_000])
            # One projection per remaining calendar day, shrinking to zero on
            # election day. The count follows the forecast date, so it is
            # derived rather than written out.
            future_days = (ELECTION_DAY - FORECAST_DAY).days
            self.assertEqual(projection_calls, list(range(future_days - 1, -1, -1)))
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
            production_result = self._production_result(FORECAST_AS_OF)

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
                now=_forecast_utc(8),
                automation_enabled="true",
                refresh_fn=refresh,
                simulation_runner=runner,
                projection_runner=self._projection_runner,
                    campaign_path_simulator=self._campaign_path_simulator,
                website_check_fn=lambda _: {"status": "PASS"},
                mode="dry_run",
                generated_at_utc=f"{FORECAST_AS_OF}T08:00:00+00:00",
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
                # The repository's own hash helper, not a second implementation.
                "manifest_sha256": compute_file_sha256(old_manifest),
            }
            site_pointer_path = site / "files/election-simulator/current.json"
            site_pointer_path.write_text(json.dumps(stale_pointer, indent=2) + "\n", encoding="utf-8")
            subprocess.run(["git", "add", "files/election-simulator/current.json"], cwd=site, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture: make website stale"], cwd=site, check=True)
            site_before = site_pointer_path.read_bytes()
            calls: list[int] = []
            production_result = self._production_result(FORECAST_AS_OF)

            def runner(**kwargs):
                calls.append(int(kwargs["samples"]))
                return production_result

            result = run_automation(
                source,
                site_repo=site,
                schedule=INTRADAY_SCHEDULE_UTC,
                now=_forecast_utc(8),
                automation_enabled="true",
                mode="dry_run",
                refresh_fn=lambda raw, processed, **kwargs: {"messages": []},
                simulation_runner=runner,
                projection_runner=self._projection_runner,
                    campaign_path_simulator=self._campaign_path_simulator,
                website_check_fn=lambda _: {"status": "PASS"},
                generated_at_utc=f"{FORECAST_AS_OF}T04:00:00+00:00",
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

    def test_a_generation_certified_after_the_benchmark_window_retains_its_draws(self) -> None:
        """Amendment 007, at the date that motivated it.

        Amendment 004 permitted per-generation exact-draw sidecars only through
        2026-09-12, after which amendment 003's prohibition would return. A
        renderer needs those joint draws -- coalition intervals are computed
        from them and cannot be recovered from published marginal quantiles --
        so on 2026-09-13, election day, rendering an already-certified
        generation would have had no valid input.

        Certified here on 2026-09-13 -- election day, and the day after the
        benchmark's final scheduled date of 2026-09-12 -- asserting the sidecar
        is exported and hash-linked from the archive entry rather than merely
        present. The history contract permits a publication on election day
        itself and refuses one after it, so 2026-09-13 is both the last date a
        forecast can be certified and the first date amendment 004's window
        would have denied it a sidecar.
        """

        for as_of in ("2026-09-13",):
            with self.subTest(as_of=as_of):
                with tempfile.TemporaryDirectory() as tmp:
                    source, site = self._production_fixture(Path(tmp))
                    production_result = self._production_result(as_of)

                    def refresh(raw, processed, **kwargs):
                        raw.mkdir(parents=True, exist_ok=True)
                        self._change_normalized_poll_support(
                            processed / "individual_polls.csv")
                        return {"messages": []}

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
                        result = run_automation(
                            source,
                            site_repo=site,
                            schedule=INTRADAY_SCHEDULE_UTC,
                            now=datetime.fromisoformat(f"{as_of}T08:00:00+00:00"),
                            automation_enabled="true",
                            mode="publish",
                            commit=True,
                            refresh_fn=refresh,
                            simulation_runner=runner,
                            projection_runner=self._projection_runner,
                            campaign_path_simulator=self._campaign_path_simulator,
                            website_check_fn=lambda _root: {"status": "PASS"},
                            website_push_check_fn=lambda _root: {"status": "PASS"},
                            generated_at_utc=f"{as_of}T08:00:00+00:00",
                        )
                    self.assertEqual(result.status, "PUBLISHED", result.summary.render())

                    generation = json.loads(
                        (source / "files/election-simulator/current.json").read_text()
                    )["publication_generation"]
                    archive = source / "data/processed/prospective_forecasts" / generation
                    draws = archive / SIDECAR_DRAWS_FILENAME
                    meta = archive / SIDECAR_METADATA_FILENAME
                    self.assertTrue(draws.is_file(), f"no exact draws for {generation}")
                    self.assertTrue(meta.is_file(), f"no sidecar metadata for {generation}")

                    # VERIFIED, not merely present: the archive's own
                    # validator ties the sidecar to the certified snapshot, so
                    # this runs that check rather than re-deriving a hash.
                    snapshot = json.loads((archive / "snapshot.json").read_text())
                    validate_exact_draw_sidecar(
                        draws, meta, certified_snapshot=snapshot)
                    index = json.loads(
                        (source / "data/processed/prospective_forecasts/index.json").read_text())
                    entry = next(e for e in index["snapshots"]
                                 if e.get("generation_id") == generation)
                    self.assertEqual(entry["path"], f"{generation}/snapshot.json", entry)

                    # And the snapshot it belongs to is in the same commit.
                    files = subprocess.run(
                        ["git", "show", "--name-only", "--format=", "HEAD~1"],
                        cwd=source, check=True, capture_output=True, text=True,
                    ).stdout.split()
                    relative = f"data/processed/prospective_forecasts/{generation}"
                    self.assertIn(f"{relative}/snapshot.json", files, files)
                    self.assertIn(f"{relative}/{SIDECAR_DRAWS_FILENAME}", files, files)

    def test_backfill_failure_after_certification_keeps_the_forecast_on_the_remote(self) -> None:
        """The 2026-09-09 incident, as an acceptance test.

        That run finished its 100_000-draw simulation at 16:36:33, exported the
        snapshot, entered the history backfill, and was killed by the job
        timeout at 18:20:02 having pushed nothing. The forecast was complete and
        was thrown away.

        Here the backfill is failed deliberately at its first call -- the
        earliest point after certification -- and the surviving state is
        asserted from FRESH CLONES of both remotes, which is the only tree
        production ever gives a retry:

          * the simulator's remote holds the certified generation;
          * its certification commit excludes the history artifact;
          * the website's remote is untouched;
          * rendering republishes that same generation with no second
            authoritative simulation.
        """

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkouts = root / "checkouts"
            checkouts.mkdir()
            source, site = self._production_fixture(checkouts)
            source_remote = root / "simulator.git"
            site_remote = root / "website.git"
            self._publish_to_bare_remote(source, source_remote, "main")
            self._publish_to_bare_remote(site, site_remote, "master")
            site_tree_before = subprocess.run(
                ["git", "rev-parse", "master^{tree}"], cwd=site_remote, check=True,
                capture_output=True, text=True,
            ).stdout.strip()
            site_subjects_before = self._remote_subjects(site_remote, "master")

            simulations: list[int] = []
            production_result = self._production_result(FORECAST_AS_OF)

            def refresh(raw, processed, **kwargs):
                raw.mkdir(parents=True, exist_ok=True)
                self._change_normalized_poll_support(processed / "individual_polls.csv")
                return {"messages": []}

            def runner(**kwargs):
                simulations.append(int(kwargs["samples"]))
                commit = subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=source, check=True,
                    capture_output=True, text=True,
                ).stdout.strip()
                production_result.manifest["source_git_commit"] = commit
                production_result.manifest["git_commit"] = commit
                return production_result

            # What the incident actually was: the process was KILLED inside the
            # backfill, so no in-process handler ran. A backfill exception is
            # already tolerated here, which is why merely raising proves
            # nothing. The property that makes a kill survivable is that the
            # certified generation is on the remote BEFORE this point, so the
            # probe records the remote's state at backfill entry.
            remote_at_backfill: list[str | None] = []

            def exploding_backfill(*args, **kwargs):
                shown = subprocess.run(
                    ["git", "show", "main:files/election-simulator/current.json"],
                    cwd=source_remote, capture_output=True, text=True, check=False,
                )
                remote_at_backfill.append(
                    json.loads(shown.stdout)["publication_generation"]
                    if shown.returncode == 0 else None
                )
                raise AutomationError("history curve backfill exhausted its budget")

            with patch(
                "scripts.publication_pipeline.pipeline.DEFAULT_PROCESSED_ROOT",
                source / "data/processed",
            ), patch.object(base, "backfill_reconstructed_curve", exploding_backfill):
                blocked = run_automation(
                    source,
                    site_repo=site,
                    schedule=INTRADAY_SCHEDULE_UTC,
                    now=_forecast_utc(8),
                    automation_enabled="true",
                    mode="publish",
                    commit=True,
                    push=True,
                    refresh_fn=refresh,
                    simulation_runner=runner,
                    projection_runner=self._projection_runner,
                    campaign_path_simulator=self._campaign_path_simulator,
                    website_check_fn=lambda _root: {"status": "PASS"},
                    website_push_check_fn=lambda _root: (_ for _ in ()).throw(
                        AutomationError("website command failed: changes-baseline.smoke.mjs")),
                    generated_at_utc=f"{FORECAST_AS_OF}T08:00:00+00:00",
                )
            self.assertNotEqual(blocked.status, "PUBLISHED")
            self.assertEqual(simulations, [100_000], "one authoritative simulation")

            # "Forecast certified; website update failed" has to be
            # distinguishable from "forecast failed", and it has to survive the
            # generic failure handler -- which is the only path a rendering
            # exception takes.
            self.assertEqual(blocked.summary.certification_status, "CERTIFIED_AND_PUSHED")
            self.assertTrue(
                blocked.summary.certification_remote_verified,
                blocked.summary.render(),
            )
            self.assertNotEqual(blocked.summary.certification_commit, "NONE")
            self.assertIn("Certification: CERTIFIED_AND_PUSHED", blocked.summary.render())

            # THE LOAD-BEARING ASSERTION. When the backfill began, the
            # certified generation was already retrievable from the simulator's
            # remote. Had the process been killed at that instant -- as it was
            # on 2026-09-09 at 18:20:02 -- the forecast would still exist.
            self.assertEqual(len(remote_at_backfill), 1, remote_at_backfill)
            self.assertIsNotNone(
                remote_at_backfill[0],
                "no certified pointer on the remote when the backfill started: a "
                "process kill here would discard the finished forecast again",
            )

            # Certified, and durable on the remote.
            certified = self._remote_pointer(
                source_remote, "main")["publication_generation"]
            self.assertEqual(
                remote_at_backfill[0], certified,
                "the generation on the remote at backfill entry is the certified one",
            )
            subjects = self._remote_subjects(source_remote, "main")
            self.assertTrue(
                any(s.startswith("chore: publish election forecast") for s in subjects),
                subjects,
            )
            # The CERTIFICATION commit specifically -- not HEAD, which by now
            # is the separate render commit. That separation is the point: the
            # certification commit carries the generation and never the
            # history, so a rendering failure cannot have touched it.
            certification_sha = subprocess.run(
                ["git", "log", "--format=%H", "--grep",
                 "^chore: publish election forecast", "main"],
                cwd=source_remote, check=True, capture_output=True, text=True,
            ).stdout.split()[0]
            files = subprocess.run(
                ["git", "show", "--name-only", "--format=", certification_sha],
                cwd=source_remote, check=True, capture_output=True, text=True,
            ).stdout.split()
            self.assertTrue(
                any(f.startswith(f"files/election-simulator/versions/{certified}/")
                    for f in files), files)
            self.assertNotIn(
                "files/election-simulator/history/coalition-timeseries.json", files, files)

            # The website remote is untouched.
            self.assertEqual(self._remote_subjects(site_remote, "master"),
                             site_subjects_before)
            self.assertEqual(
                subprocess.run(
                    ["git", "rev-parse", "master^{tree}"], cwd=site_remote,
                    check=True, capture_output=True, text=True,
                ).stdout.strip(),
                site_tree_before,
            )

            # Rendering retried from fresh clones publishes that generation,
            # with a runner that raises if any simulation is attempted.
            fresh_source = self._clone_from_remote(
                source_remote, "main", root / "fresh-simulator")
            fresh_site = self._clone_from_remote(
                site_remote, "master", root / "fresh-website")

            def must_not_simulate(**kwargs):
                raise AssertionError("rendering must not run a simulation")

            with patch(
                "scripts.publication_pipeline.pipeline.DEFAULT_PROCESSED_ROOT",
                fresh_source / "data/processed",
            ):
                rendered = run_automation(
                    fresh_source,
                    site_repo=fresh_site,
                    schedule=INTRADAY_SCHEDULE_UTC,
                    now=_forecast_utc(10),
                    automation_enabled="true",
                    mode="publish",
                    commit=True,
                    push=True,
                    refresh_fn=lambda raw, processed, **kwargs: {"messages": []},
                    simulation_runner=must_not_simulate,
                    website_check_fn=lambda _root: {"status": "PASS"},
                    website_push_check_fn=lambda _root: {"status": "PASS"},
                    generated_at_utc=f"{FORECAST_AS_OF}T10:00:00+00:00",
                )
            self.assertNotEqual(rendered.status, "FAILED", rendered.summary.render())
            self.assertEqual(
                self._remote_pointer(site_remote, "master")["publication_generation"],
                certified,
            )

    def test_website_gate_failure_certifies_the_forecast_and_leaves_the_site(self) -> None:
        """The boundary, at the pointers.

        This asserted that BOTH live pointers stayed byte-identical after a
        late website failure, which was the right guarantee while certification
        and rendering were one transaction: a half-published forecast was the
        only other outcome.

        Certification now pushes before any rendering runs, so the source
        pointer *must* move -- that is the whole point, and on 2026-09-09 a
        completed 100_000-draw simulation was discarded precisely because it
        did not. The website pointer must still not move, and the guarantee
        the original name protected -- that no partially staged forecast is
        ever published -- is now stronger, not weaker: the forecast is fully
        certified and the website is untouched.
        """

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

            production_result = self._production_result(FORECAST_AS_OF)
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
                    now=_forecast_utc(8),
                    automation_enabled="true",
                    commit=True,
                    refresh_fn=refresh,
                    simulation_runner=runner,
                    projection_runner=self._projection_runner,
                    campaign_path_simulator=self._campaign_path_simulator,
                    website_check_fn=failed_website_check,
                    generated_at_utc=f"{FORECAST_AS_OF}T06:00:00+00:00",
                )

            self.assertEqual(result.status, "FAILED")
            self.assertIn("browser smoke failed", result.summary.failure or "")
            self.assertEqual(result.summary.simulation_samples, 100_000)
            self.assertIn("Simulation samples: 100000", result.summary.render())
            self.assertEqual(calls, [100_000])

            # The forecast is certified: the source pointer advanced, and it
            # names the generation the run produced.
            self.assertNotEqual(source_current.read_bytes(), source_before)
            certified = json.loads(source_current.read_text())["publication_generation"]
            self.assertIn(
                f"chore: publish election forecast",
                self._git_subjects(source)[0],
            )
            self.assertTrue(
                (source / "files/election-simulator/versions" / certified).is_dir(),
                certified,
            )

            # The website is untouched, by pointer and by commit subject.
            self.assertEqual(site_current.read_bytes(), site_before)
            self.assertFalse(
                any(subject.startswith("chore: sync election forecast")
                    for subject in self._git_subjects(site)),
                self._git_subjects(site),
            )

            # And the history was NOT swept into the certification commit: it
            # is a rendering output, and rendering is what failed.
            certification_files = subprocess.run(
                ["git", "show", "--name-only", "--format=", "HEAD"],
                cwd=source, check=True, capture_output=True, text=True,
            ).stdout.split()
            self.assertNotIn(
                "files/election-simulator/history/coalition-timeseries.json",
                certification_files,
                certification_files,
            )

            self.assertEqual(self._git_status(source), "")
            self.assertEqual(self._git_status(site), "")

    def test_daily_dry_run_publishes_without_poll_change(self) -> None:
        """Daily mode still executes one production event on an unchanged snapshot."""

        with tempfile.TemporaryDirectory() as tmp:
            source, site = self._production_fixture(Path(tmp))
            calls: list[int] = []
            production_result = self._production_result(FORECAST_AS_OF)

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
                    now=_forecast_utc(4),
                    automation_enabled="true",
                    refresh_fn=refresh,
                    simulation_runner=runner,
                    projection_runner=self._projection_runner,
                    campaign_path_simulator=self._campaign_path_simulator,
                    website_check_fn=lambda _: {"status": "PASS"},
                    generated_at_utc=f"{FORECAST_AS_OF}T04:00:00+00:00",
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
        # The gate is folded across lines, so compare on normalised
        # whitespace: publish runs for every schedule tick, for an operator
        # dispatch, and for the external fallback -- and for nothing else. It
        # is additionally gated on the fallback preflight, which runs outside
        # the production concurrency group; the structural properties of that
        # split are pinned in tests.test_publication_fallback.
        self.assertIn(
            "if: >- always() && (github.event_name == 'schedule' || "
            "(github.event_name == 'workflow_dispatch' && "
            "(github.event.inputs.mode == 'publish' || "
            "github.event.inputs.mode == 'publish_if_stale'))) && "
            "(needs.fallback_preflight.result == 'skipped' || "
            "(needs.fallback_preflight.result == 'success' && "
            "needs.fallback_preflight.outputs.proceed == 'true'))",
            " ".join(publish.split()),
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
        # The publish job serves three triggers, so it forwards the resolved
        # mode rather than a literal: a schedule tick is always a full
        # publish, and a dispatch carries whichever mode it chose.
        self.assertIn('--mode "$MODE"', publish)
        self.assertIn(
            "MODE: ${{ github.event_name == 'schedule' && 'publish' "
            "|| github.event.inputs.mode }}",
            publish,
        )
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
                (
                    # Real-artifact mode, not the fixture default: the gate has
                    # to validate the artifact this publication just generated.
                    ["node", "browser-tests/party-timeseries.smoke.mjs", "_site", "--real-artifact"],
                    "party-timeseries.smoke.mjs",
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
                ("party-timeseries.smoke.mjs", "START"),
                ("party-timeseries.smoke.mjs", "DONE"),
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

    def _git_subjects(self, root: Path) -> list[str]:
        return subprocess.run(
            ["git", "log", "--format=%s"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.split("\n")

    @staticmethod
    def _publish_to_bare_remote(repo: Path, remote: Path, branch: str) -> None:
        """Give a fixture checkout a real remote on the branch it pushes to."""

        subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
        subprocess.run(["git", "branch", "-M", branch], cwd=repo, check=True)
        subprocess.run(["git", "remote", "add", "origin", str(remote)], cwd=repo, check=True)
        subprocess.run(["git", "push", "-q", "origin", f"HEAD:{branch}"], cwd=repo, check=True)

    @staticmethod
    def _remote_subjects(remote: Path, branch: str) -> list[str]:
        return subprocess.run(
            ["git", "log", "--format=%s", branch], cwd=remote, check=True,
            capture_output=True, text=True,
        ).stdout.split("\n")

    @staticmethod
    def _remote_pointer(remote: Path, branch: str) -> dict:
        return json.loads(subprocess.run(
            ["git", "show", f"{branch}:files/election-simulator/current.json"],
            cwd=remote, check=True, capture_output=True, text=True,
        ).stdout)

    @classmethod
    def _clone_from_remote(cls, remote: Path, branch: str, destination: Path) -> Path:
        subprocess.run(
            ["git", "clone", "-q", "-b", branch, str(remote), str(destination)],
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Automation Test"], cwd=destination, check=True)
        subprocess.run(
            ["git", "config", "user.email", "automation@example.test"],
            cwd=destination, check=True)
        return destination

    def test_gate_failure_leaves_the_forecast_on_the_remote_and_the_website_untouched(self) -> None:
        """Remote durability, which `push=False` cannot show.

        A local commit proves the run got that far; what the split actually
        promises is that a failed website gate costs nothing already earned.
        That is a claim about the remotes: the forecast has to be durable
        somewhere the next fresh checkout will find it, and the website has to
        be exactly where it was. So this pushes for real, to temporary bare
        remotes, and then finishes the job from fresh clones -- which is how
        production reaches recovery, every run starting from the remote.
        """

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkouts = root / "checkouts"
            checkouts.mkdir()
            source, site = self._production_fixture(checkouts)
            source_remote = root / "simulator.git"
            site_remote = root / "website.git"
            self._publish_to_bare_remote(source, source_remote, "main")
            self._publish_to_bare_remote(site, site_remote, "master")

            site_subjects_before = self._remote_subjects(site_remote, "master")
            site_pointer_before = self._remote_pointer(site_remote, "master")
            site_tree_before = subprocess.run(
                ["git", "rev-parse", "master^{tree}"], cwd=site_remote, check=True,
                capture_output=True, text=True,
            ).stdout.strip()

            simulations: list[int] = []
            production_result = self._production_result(FORECAST_AS_OF)

            def refresh(raw, processed, **kwargs):
                raw.mkdir(parents=True, exist_ok=True)
                self._change_normalized_poll_support(processed / "individual_polls.csv")
                return {"messages": []}

            def runner(**kwargs):
                simulations.append(int(kwargs["samples"]))
                commit = subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=source, check=True,
                    capture_output=True, text=True,
                ).stdout.strip()
                production_result.manifest["source_git_commit"] = commit
                production_result.manifest["git_commit"] = commit
                return production_result

            def failing_push_gate(site_root):
                raise AutomationError(
                    "website command failed: changes-baseline.smoke.mjs "
                    "(exit code 1) after 13.000s"
                )

            with patch(
                "scripts.publication_pipeline.pipeline.DEFAULT_PROCESSED_ROOT",
                source / "data/processed",
            ):
                blocked = run_automation(
                    source,
                    site_repo=site,
                    schedule=INTRADAY_SCHEDULE_UTC,
                    now=_forecast_utc(8),
                    automation_enabled="true",
                    mode="publish",
                    commit=True,
                    push=True,
                    refresh_fn=refresh,
                    simulation_runner=runner,
                    projection_runner=self._projection_runner,
                    campaign_path_simulator=self._campaign_path_simulator,
                    website_check_fn=lambda _root: {"status": "PASS"},
                    website_push_check_fn=failing_push_gate,
                    generated_at_utc=f"{FORECAST_AS_OF}T08:00:00+00:00",
                )
            self.assertNotEqual(blocked.status, "PUBLISHED")
            self.assertEqual(simulations, [100_000])

            # The forecast reached the simulator's remote: durable where the
            # next fresh checkout will find it.
            source_subjects = self._remote_subjects(source_remote, "main")
            self.assertTrue(
                any(subject.startswith("chore: publish election forecast")
                    for subject in source_subjects),
                source_subjects,
            )
            generation = self._remote_pointer(
                source_remote, "main")["publication_generation"]

            # The website's remote is byte-identical: same subjects, same
            # pointer, same tree object.
            self.assertEqual(self._remote_subjects(site_remote, "master"),
                             site_subjects_before)
            self.assertEqual(self._remote_pointer(site_remote, "master"),
                             site_pointer_before)
            self.assertEqual(
                subprocess.run(
                    ["git", "rev-parse", "master^{tree}"], cwd=site_remote,
                    check=True, capture_output=True, text=True,
                ).stdout.strip(),
                site_tree_before,
            )
            self.assertNotEqual(
                site_pointer_before["publication_generation"], generation)

            # Recovery from fresh clones, which is the only tree production
            # ever gives it. Nothing is carried over from the failed run.
            fresh_source = self._clone_from_remote(
                source_remote, "main", root / "fresh-simulator")
            fresh_site = self._clone_from_remote(
                site_remote, "master", root / "fresh-website")
            self.assertEqual(self._git_status(fresh_source), "")
            self.assertEqual(self._git_status(fresh_site), "")

            def must_not_simulate(**kwargs):
                raise AssertionError("recovery must not run a simulation")

            with patch(
                "scripts.publication_pipeline.pipeline.DEFAULT_PROCESSED_ROOT",
                fresh_source / "data/processed",
            ):
                recovered = run_automation(
                    fresh_source,
                    site_repo=fresh_site,
                    schedule=INTRADAY_SCHEDULE_UTC,
                    now=_forecast_utc(10),
                    automation_enabled="true",
                    mode="publish",
                    commit=True,
                    push=True,
                    refresh_fn=lambda raw, processed, **kwargs: {"messages": []},
                    simulation_runner=must_not_simulate,
                    website_check_fn=lambda _root: {"status": "PASS"},
                    website_push_check_fn=lambda _root: {"status": "PASS"},
                    generated_at_utc=f"{FORECAST_AS_OF}T10:00:00+00:00",
                )

            self.assertEqual(recovered.status, "WEBSITE_RECOVERED")
            # That same generation, published to the website's remote, with no
            # forecast recomputed anywhere.
            self.assertEqual(
                self._remote_pointer(site_remote, "master")["publication_generation"],
                generation,
            )
            self.assertEqual(
                self._remote_pointer(source_remote, "main")["publication_generation"],
                generation,
            )
            self.assertEqual(self._git_status(fresh_source), "")
            self.assertEqual(self._git_status(fresh_site), "")

    def test_dry_run_fails_on_a_second_tier_failure_without_writing_anything(self) -> None:
        """A dry run that cannot fail the way a publication would is a lie.

        The committed path runs the second tier after certification, which a
        dry run cannot imitate because there is nothing to certify. Running it
        only there would leave `--mode dry_run` reporting success while the
        real publication failed on one of those seven suites -- and a dry run
        exists to be believed before a publication is attempted.
        """

        with tempfile.TemporaryDirectory() as tmp:
            source, site = self._production_fixture(Path(tmp))
            source_before = self._git_status(source)
            site_before = self._git_status(site)
            source_head = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=source, check=True,
                capture_output=True, text=True,
            ).stdout.strip()
            site_head = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=site, check=True,
                capture_output=True, text=True,
            ).stdout.strip()
            site_pointer_before = (
                site / "files/election-simulator/current.json").read_bytes()
            production_result = self._production_result(FORECAST_AS_OF)

            def refresh(raw, processed, **kwargs):
                raw.mkdir(parents=True, exist_ok=True)
                path = processed / "individual_polls.csv"
                self._change_normalized_poll_support(path)
                return {"messages": []}

            def runner(**kwargs):
                commit = subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=source, check=True,
                    capture_output=True, text=True,
                ).stdout.strip()
                production_result.manifest["source_git_commit"] = commit
                production_result.manifest["git_commit"] = commit
                return production_result

            first_tier: list[Path] = []
            second_tier: list[Path] = []

            def passing_first_tier(site_root):
                first_tier.append(Path(site_root))
                return {"status": "PASS"}

            def failing_second_tier(site_root):
                second_tier.append(Path(site_root))
                raise AutomationError(
                    "website command failed: changes-baseline.smoke.mjs "
                    "(exit code 1) after 13.000s"
                )

            with patch(
                "scripts.publication_pipeline.pipeline.DEFAULT_PROCESSED_ROOT",
                source / "data/processed",
            ):
                result = run_automation(
                    source,
                    site_repo=site,
                    schedule=INTRADAY_SCHEDULE_UTC,
                    now=_forecast_utc(8),
                    automation_enabled="true",
                    mode="dry_run",
                    commit=False,
                    refresh_fn=refresh,
                    simulation_runner=runner,
                    projection_runner=self._projection_runner,
                    campaign_path_simulator=self._campaign_path_simulator,
                    website_check_fn=passing_first_tier,
                    website_push_check_fn=failing_second_tier,
                    generated_at_utc=f"{FORECAST_AS_OF}T08:00:00+00:00",
                )

            # The dry run reached the second tier and failed on it, rather
            # than reporting a success the publication would not reproduce.
            self.assertEqual(len(second_tier), 1, second_tier)
            self.assertNotEqual(result.status, "PUBLISHED")
            self.assertNotIn(result.status, {"DRY_RUN", "STAGED", "OK"})

            # Both tiers ran against the same disposable tree, never the live
            # website checkout.
            self.assertEqual(len(first_tier), 1, first_tier)
            self.assertEqual(first_tier, second_tier)
            self.assertNotEqual(second_tier[0], site)

            # And a dry run writes nothing, in either repository.
            self.assertEqual(self._git_status(source), source_before)
            self.assertEqual(self._git_status(site), site_before)
            for repository, head in ((source, source_head), (site, site_head)):
                self.assertEqual(
                    subprocess.run(
                        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
                        capture_output=True, text=True,
                    ).stdout.strip(),
                    head,
                )
            self.assertEqual(
                (site / "files/election-simulator/current.json").read_bytes(),
                site_pointer_before,
            )

    def test_website_push_gate_failure_certifies_the_forecast_but_never_pushes(self) -> None:
        """The second tier's whole justification, asserted as behaviour.

        Source order shows where the call sits; only a real publication shows
        what a failure there costs. The claim is that the split is not merely
        tidy but load-bearing: when the tier fails, the forecast is already
        durable and the website is untouched, and the same generation can then
        be published by recovery without simulating anything again.

        Scoped to one repository pair with ``push=False``, so "durable" here
        means committed, and recovery is shown from a reset checkout. The
        remote half of the claim, with real clones from bare remotes, belongs
        to the sibling test below.
        """

        with tempfile.TemporaryDirectory() as tmp:
            source, site = self._production_fixture(Path(tmp))
            simulations: list[int] = []
            production_result = self._production_result(FORECAST_AS_OF)

            def refresh(raw, processed, **kwargs):
                raw.mkdir(parents=True, exist_ok=True)
                path = processed / "individual_polls.csv"
                self._change_normalized_poll_support(path)
                return {"messages": []}

            def runner(**kwargs):
                simulations.append(int(kwargs["samples"]))
                commit = subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=source, check=True,
                    capture_output=True, text=True,
                ).stdout.strip()
                production_result.manifest["source_git_commit"] = commit
                production_result.manifest["git_commit"] = commit
                return production_result

            gate_calls: list[Path] = []

            def failing_push_gate(site_root):
                gate_calls.append(Path(site_root))
                raise AutomationError(
                    "website command failed: changes-baseline.smoke.mjs "
                    "(exit code 1) after 13.000s"
                )

            with patch(
                "scripts.publication_pipeline.pipeline.DEFAULT_PROCESSED_ROOT",
                source / "data/processed",
            ):
                blocked = run_automation(
                    source,
                    site_repo=site,
                    schedule=INTRADAY_SCHEDULE_UTC,
                    now=_forecast_utc(8),
                    automation_enabled="true",
                    commit=True,
                    refresh_fn=refresh,
                    simulation_runner=runner,
                    projection_runner=self._projection_runner,
                    campaign_path_simulator=self._campaign_path_simulator,
                    website_check_fn=lambda _root: {"status": "PASS"},
                    website_push_check_fn=failing_push_gate,
                    generated_at_utc=f"{FORECAST_AS_OF}T08:00:00+00:00",
                )

            # The gate ran, and it ran against a staged tree rather than the
            # live website checkout.
            self.assertEqual(len(gate_calls), 1, gate_calls)
            self.assertNotEqual(gate_calls[0], site)

            self.assertNotEqual(blocked.status, "PUBLISHED")
            self.assertEqual(simulations, [100_000])

            # The forecast is durable: the simulator carries its own
            # publication commit even though the website was never pushed.
            source_subjects = self._git_subjects(source)
            self.assertTrue(
                any(subject.startswith("chore: publish election forecast")
                    for subject in source_subjects),
                source_subjects,
            )
            certified = json.loads(
                (source / "files/election-simulator/current.json").read_text())
            generation = certified["publication_generation"]

            # And the website was not pushed, by any of the three names that
            # would indicate it had been.
            site_subjects = self._git_subjects(site)
            for forbidden in ("chore: sync election forecast",
                              "chore: recover election forecast"):
                self.assertFalse(
                    any(subject.startswith(forbidden) for subject in site_subjects),
                    site_subjects,
                )
            # Nothing was committed, so nothing would be served.
            committed_pointer = json.loads(subprocess.run(
                ["git", "show", "HEAD:files/election-simulator/current.json"],
                cwd=site, check=True, capture_output=True, text=True,
            ).stdout)
            self.assertNotEqual(
                committed_pointer["publication_generation"], generation)
            # And the live pointer *file* was put back. Installation and the
            # pointer write both precede this gate, so without the rollback
            # the working tree would claim a generation that was never
            # pushed -- the state the module's own recovery path refuses to
            # leave behind, in the same words.
            #
            # Only the pointer. The checkout is deliberately still dirty with
            # the installed version files, which is why the recovery phase
            # below has to reset it rather than simply proceeding.
            live_pointer = json.loads(
                (site / "files/election-simulator/current.json").read_text())
            self.assertEqual(
                live_pointer["publication_generation"],
                committed_pointer["publication_generation"],
            )

            # Recovery publishes that exact generation with no further
            # simulation: the cost of a failed push gate is one website
            # update, never a recomputed forecast.
            #
            # Recovery is demonstrated from a *reset* checkout, not from the
            # tree the failed run left behind: restoring the pointer does not
            # clean that tree, and `_assert_clean` would refuse it. That is
            # not a gap being papered over -- production never sees the tree
            # again, since every workflow run starts from a fresh checkout of
            # the durable remote. `git checkout`/`clean` is how a local
            # fixture spells "fresh checkout"; the sibling remote-durability
            # test does it properly, with real clones from bare remotes.
            subprocess.run(["git", "checkout", "--", "."], cwd=site, check=True)
            subprocess.run(["git", "clean", "-qfd"], cwd=site, check=True)
            self.assertEqual(self._git_status(site), "")
            simulations.clear()

            def must_not_simulate(**kwargs):
                raise AssertionError("recovery must not run a simulation")

            with patch(
                "scripts.publication_pipeline.pipeline.DEFAULT_PROCESSED_ROOT",
                source / "data/processed",
            ):
                recovered = run_automation(
                    source,
                    site_repo=site,
                    schedule=INTRADAY_SCHEDULE_UTC,
                    now=_forecast_utc(10),
                    automation_enabled="true",
                    commit=True,
                    refresh_fn=lambda raw, processed, **kwargs: {"messages": []},
                    simulation_runner=must_not_simulate,
                    website_check_fn=lambda _root: {"status": "PASS"},
                    website_push_check_fn=lambda _root: {"status": "PASS"},
                    generated_at_utc=f"{FORECAST_AS_OF}T10:00:00+00:00",
                )

            self.assertEqual(recovered.status, "WEBSITE_RECOVERED")
            self.assertEqual(simulations, [])
            recovered_live = json.loads(
                (site / "files/election-simulator/current.json").read_text())
            self.assertEqual(recovered_live["publication_generation"], generation)
            self.assertEqual(self._git_status(source), "")
            self.assertEqual(self._git_status(site), "")

    def test_committed_polling_without_publication_forces_next_unchanged_retry(self) -> None:
        """A durable polling commit is retried after the first publish fails."""

        with tempfile.TemporaryDirectory() as tmp:
            source, site = self._production_fixture(Path(tmp))
            baseline_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=source, check=True,
                capture_output=True, text=True,
            ).stdout.strip()
            # Re-published by the real exporter, certified from this
            # fixture repository's own HEAD. The generation id does not move:
            # the source commit is audit provenance and enters neither the
            # frozen timestamp nor the deterministic payload hash the id is
            # built from.
            for repository in (source, site):
                export_frozen_site_publication(
                    repository / "files/election-simulator",
                    source_git_commit=baseline_commit,
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
            production_result = self._production_result(FORECAST_AS_OF)

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
                    now=_forecast_utc(8),
                    automation_enabled="true",
                    mode="publish",
                    commit=True,
                    refresh_fn=refresh,
                    simulation_runner=failing_runner,
                    website_check_fn=lambda _: {"status": "PASS"},
                    generated_at_utc=f"{FORECAST_AS_OF}T08:00:00+00:00",
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
                    now=_forecast_utc(8),
                    automation_enabled="true",
                    mode="publish",
                    commit=True,
                    refresh_fn=refresh,
                    simulation_runner=succeeding_runner,
                    projection_runner=self._projection_runner,
                    campaign_path_simulator=self._campaign_path_simulator,
                    website_check_fn=lambda _: {"status": "PASS"},
                    generated_at_utc=f"{FORECAST_AS_OF}T08:05:00+00:00",
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
            production_result = self._production_result(FORECAST_AS_OF)

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
                    forecast_as_of=FORECAST_AS_OF,
                    simulation_runner=runner,
                    projection_runner=self._projection_runner,
                    campaign_path_simulator=self._campaign_path_simulator,
                    website_check_fn=lambda _: {"status": "PASS"},
                    generated_at_utc=f"{FORECAST_AS_OF}T09:00:00+00:00",
                    commit=True,
                    push=False,
                    allow_duplicate_payload=True,
                )
            generation = str(first[0].snapshot["generation_id"])
            generation_dir = source / "data/processed/prospective_forecasts" / generation
            loaded_sidecar = load_verified_draw_sidecar(
                generation_dir / "draws.npz",
                generation_dir / "draws.json",
                expected_generation_id=generation,
                expected_payload_hash=first[0].snapshot["deterministic_payload_sha256"],
            )
            np.testing.assert_array_equal(
                loaded_sidecar["vote_shares_pct"],
                production_result.vote_shares_matrix,
            )
            np.testing.assert_array_equal(loaded_sidecar["seats"], production_result.seats_matrix)
            selected = collect_latest_certified_generation(
                source / "data/processed/prospective_forecasts",
                "2100-01-01T00:00:00Z",
                source,
            )
            self.assertEqual(selected["status"], "FOUND_WITH_VERIFIED_DRAWS")
            self.assertEqual(selected["exact_draws"]["status"], "VERIFIED")
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
                now=_forecast_utc(10),
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

    def test_exact_sidecar_failure_leaves_live_repositories_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source, site = self._production_fixture(Path(tmp))
            production_result = self._production_result("2026-09-05")
            source_pointer = (source / "files/election-simulator/current.json").read_bytes()
            site_pointer = (site / "files/election-simulator/current.json").read_bytes()

            def runner(**kwargs):
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

            with (
                patch(
                    "scripts.publication_pipeline.pipeline.DEFAULT_PROCESSED_ROOT",
                    source / "data/processed",
                ),
                patch(
                    "scripts.election_automation_base.write_exact_draw_sidecar",
                    side_effect=ValueError("sidecar gate failed"),
                ),
                self.assertRaisesRegex(ValueError, "sidecar gate failed"),
            ):
                run_production_event(
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

            self.assertEqual((source / "files/election-simulator/current.json").read_bytes(), source_pointer)
            self.assertEqual((site / "files/election-simulator/current.json").read_bytes(), site_pointer)
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
        """The live committed artifact, deliberately: this is the integration lane.

        Everything else in this module publishes into the frozen fixture from
        tests/history_fixtures.py. This test and the rollover test below are
        the exceptions, and they are exceptions on purpose: the property they
        assert is about the shape the *shipped* artifact actually has -- which
        points already carry a reconstructed curve, and which carry only an
        archived prospective point. A synthetic history proves nothing about
        that. Both read their dates out of the artifact rather than hardcoding
        them, so a denser or longer artifact does not break them.
        """

        existing = json.loads(LIVE_HISTORY_ARTIFACT.read_text())

        # A date that already carries a reconstructed point must not be
        # resimulated *while its effective model inputs are unchanged*. That
        # qualifier is the invariant, not a softening of it: reconstruction is
        # a function of those inputs, so a point whose inputs were genuinely
        # revised upstream is stale and reusing it would publish a curve built
        # from numbers that no longer exist. The partition is derived from the
        # same fingerprint the production gate uses.
        #
        # A date carrying only an archived prospective point is in neither
        # set: it has no curve point yet, and simulating it is how the hole
        # each publication leaves gets closed.
        input_stable, input_revised = _reuse_partition(existing)
        resimulated: list[str] = []

        def unexpected_runner(**kwargs):
            as_of = str(kwargs.get("as_of"))
            if as_of in input_stable:
                raise AssertionError(
                    f"point with unchanged effective inputs was rerun: {kwargs}"
                )
            resimulated.append(as_of)
            votes, seats = self._matrices()
            requested = int(kwargs["samples"])
            repeats = (requested + len(votes) - 1) // len(votes)
            return SimpleNamespace(
                summary=SimpleNamespace(as_of=as_of, total_samples=requested),
                vote_shares_matrix=np.tile(votes, (repeats, 1))[:requested],
                seats_matrix=np.tile(seats, (repeats, 1))[:requested],
                manifest={
                    "source_git_commit": COMMIT,
                    "source_worktree_clean": True,
                    "base_seed": 12345,
                },
            )

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
        before = {
            point["date"]: point
            for point in existing["series"]
            if point["provenance"] == "reconstructed_current_model"
        }
        after = {
            point["date"]: point
            for point in rebuilt["series"]
            if point["provenance"] == "reconstructed_current_model"
        }
        # Every point whose inputs still hold is carried over byte for byte.
        # The rebuild may legitimately *add* points on dates the curve was
        # missing, so the two sets are compared per date rather than as whole
        # lists.
        for point_date, point in before.items():
            if point_date in input_stable:
                self.assertEqual(after.get(point_date), point, point_date)
        # Nothing with unchanged inputs was resimulated, and everything that
        # was resimulated is accounted for: either it had no curve point, or
        # its inputs were genuinely revised.
        self.assertTrue(set(resimulated).isdisjoint(input_stable))
        self.assertEqual(
            set(resimulated),
            {day.isoformat() for day in missing_curve_dates(existing)} | input_revised,
        )
        # The reuse cache must still be doing its job. A refresh that
        # invalidated everything would satisfy the assertions above while
        # making reconstruction a full rebuild -- the failure mode the resume
        # cache exists to prevent -- so the reused majority is asserted too.
        self.assertTrue(
            input_stable,
            "no point survived the refresh; the resume cache reused nothing",
        )
        for point_date in input_stable:
            self.assertEqual(after.get(point_date), before[point_date], point_date)

    def test_production_history_rollover_and_same_day_replacement(self) -> None:
        # The second half of the live-artifact integration lane; see the note on
        # test_existing_reconstructed_points_are_reused_without_rerunning.
        existing = json.loads(LIVE_HISTORY_ARTIFACT.read_text())
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
        # A date carries at most one point *per provenance*: the reconstructed
        # curve and the archived publication for that day coexist.
        self.assertEqual(
            len({(point["date"], point["provenance"]) for point in new_day["series"]}),
            len(new_day["series"]),
        )
        old_current = next(point for point in existing["series"] if point["provenance"] == "current_production")
        rolled = [
            point for point in new_day["series"]
            if point["date"] == old_current["date"]
            and point["provenance"] == "prospective_archived"
        ]
        self.assertEqual(len(rolled), 1)
        self.assertEqual(
            {
                point["date"]: point
                for point in new_day["series"]
                if point["provenance"] == "reconstructed_current_model"
                and point["date"] in reconstructed
            },
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
        self.assertEqual(
            len({(point["date"], point["provenance"]) for point in same_day["series"]}),
            len(same_day["series"]),
        )
        # A same-day rerun replaces the point for its own date; the reconstructed
        # point that may share that date is not what the rollover produces.
        self.assertEqual(
            [
                point["provenance"]
                for point in same_day["series"]
                if point["date"] == old_current["date"]
                and point["provenance"] != "reconstructed_current_model"
            ],
            ["current_production"],
        )

    def test_coalition_quantities_use_same_joint_draws(self) -> None:
        existing = make_history_fixture()
        result = self._result(FORECAST_AS_OF)
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
        current = next(point for point in updated["series"] if point["date"] == FORECAST_AS_OF)
        self.assertEqual(current["groups"], build_groups_from_matrices(result.vote_shares_matrix, result.seats_matrix))
        validate_history_contract(updated)

    def test_simulator_and_website_history_and_publication_bytes_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "simulator"
            site = root / "website"
            for repository in (source, site):
                install_frozen_site_publication(repository / "files/election-simulator")
            # The site starts with a prior generation; mirroring a new source
            # generation exercises the immutable-copy path and history sync.
            source_pointer = json.loads((source / "files/election-simulator/current.json").read_text())
            generation = source_pointer["publication_generation"]
            (site / "files/election-simulator/versions" / generation).rename(
                site / "files/election-simulator/versions" / (generation + "-old")
            )
            source_history = source / "files/election-simulator/history/coalition-timeseries.json"
            site_history = site / "files/election-simulator/history/coalition-timeseries.json"
            source_history.parent.mkdir(parents=True, exist_ok=True)
            source_history.write_text(
                json.dumps(make_history_fixture(), separators=(",", ":")), encoding="utf-8"
            )
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
            install_frozen_site_publication(source / "files/election-simulator")
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



class CertifiedGenerationLoaderTests(unittest.TestCase):
    """Loading a certified generation for rendering, and refusing the rest.

    The renderer must reproduce the forecast that was certified, so the loader
    is verified against a generation this pipeline actually produced rather
    than against a hand-built fixture. Exactness is the load-bearing property:
    the draws it returns have to be the certified draws, bit for bit, or the
    rendered current point is a different forecast.
    """

    @classmethod
    def _certify_one(cls, tmp: Path) -> tuple[Path, str, Any]:
        """Certify one generation in a throwaway repo pair."""

        source, site = ElectionAutomationTests._production_fixture(tmp)
        production_result = ElectionAutomationTests._production_result(FORECAST_AS_OF)

        def refresh(raw, processed, **kwargs):
            raw.mkdir(parents=True, exist_ok=True)
            ElectionAutomationTests._change_normalized_poll_support(
                processed / "individual_polls.csv")
            return {"messages": []}

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
            result = run_automation(
                source,
                site_repo=site,
                schedule=INTRADAY_SCHEDULE_UTC,
                now=_forecast_utc(8),
                automation_enabled="true",
                mode="publish",
                commit=True,
                refresh_fn=refresh,
                simulation_runner=runner,
                projection_runner=ElectionAutomationTests._projection_runner,
                campaign_path_simulator=ElectionAutomationTests._campaign_path_simulator,
                website_check_fn=lambda _root: {"status": "PASS"},
                website_push_check_fn=lambda _root: {"status": "PASS"},
                generated_at_utc=f"{FORECAST_AS_OF}T08:00:00+00:00",
            )
        assert result.status == "PUBLISHED", result.summary.render()
        generation = json.loads(
            (source / "files/election-simulator/current.json").read_text()
        )["publication_generation"]
        return source, generation, production_result

    def test_it_returns_the_certified_draws_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source, generation, produced = self._certify_one(Path(tmp))
            loaded = load_certified_generation(source, generation=generation)

            self.assertEqual(loaded.generation, generation)
            # Bit-for-bit. Anything else is a different forecast.
            np.testing.assert_array_equal(
                loaded.vote_shares_matrix, produced.vote_shares_matrix)
            np.testing.assert_array_equal(
                loaded.seats_matrix, produced.seats_matrix)
            # Percentage points, not fractions.
            self.assertAlmostEqual(
                float(loaded.vote_shares_matrix[0].sum()), 100.0, places=6)
            # Provenance is the certified generation's own source revision,
            # never the renderer's checkout.
            self.assertEqual(
                loaded.source_git_commit,
                json.loads((source / "files/election-simulator/versions" / generation
                            / "manifest.json").read_text())["source_git_commit"],
            )
            self.assertEqual(loaded.manifest["publication_generation"], generation)
            self.assertEqual(loaded.manifest["rendered_from"], "archived_exact_draws")

    def test_it_accepts_an_explicit_certification_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source, generation, _ = self._certify_one(Path(tmp))
            certification = subprocess.run(
                ["git", "log", "--format=%H", "--grep",
                 "^chore: publish election forecast", "-1"],
                cwd=source, check=True, capture_output=True, text=True,
            ).stdout.strip()
            loaded = load_certified_generation(
                source, generation=generation, certification_commit=certification)
            self.assertEqual(loaded.certification_commit, certification)

    def test_it_refuses_a_commit_that_did_not_certify_the_generation(self) -> None:
        """The commit is the caller's statement of which certification to render."""

        with tempfile.TemporaryDirectory() as tmp:
            source, generation, _ = self._certify_one(Path(tmp))
            earlier = subprocess.run(
                ["git", "rev-list", "--max-parents=0", "HEAD"],
                cwd=source, check=True, capture_output=True, text=True,
            ).stdout.split()[0]
            with self.assertRaises(CertifiedGenerationError) as ctx:
                load_certified_generation(
                    source, generation=generation, certification_commit=earlier)
            self.assertIn("does not contain", str(ctx.exception))

    def test_it_refuses_a_generation_without_archived_draws(self) -> None:
        """Pre-amendment-007 generations fail clearly instead of re-simulating."""

        with tempfile.TemporaryDirectory() as tmp:
            source, generation, _ = self._certify_one(Path(tmp))
            archive = source / "data/processed/prospective_forecasts" / generation
            (archive / SIDECAR_DRAWS_FILENAME).unlink()
            with self.assertRaises(CertifiedGenerationError) as ctx:
                load_certified_generation(source, generation=generation)
            message = str(ctx.exception)
            self.assertIn("no archived exact-draw sidecar", message)
            self.assertIn("amendment 007", message)

    def test_it_refuses_a_snapshot_that_disagrees_with_the_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source, generation, _ = self._certify_one(Path(tmp))
            snapshot_path = (source / "data/processed/prospective_forecasts"
                             / generation / "snapshot.json")
            snapshot = json.loads(snapshot_path.read_text())
            snapshot["deterministic_payload_sha256"] = "0" * 64
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
            with self.assertRaises(CertifiedGenerationError) as ctx:
                load_certified_generation(source, generation=generation)
            self.assertIn("disagree on deterministic_payload_sha256", str(ctx.exception))

    def test_it_refuses_an_unknown_or_unsafe_generation_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source, _generation, _ = self._certify_one(Path(tmp))
            for bad in ("../escape", "not-a-generation"):
                with self.subTest(generation=bad):
                    with self.assertRaises(CertifiedGenerationError):
                        load_certified_generation(source, generation=bad)

    def test_the_loader_never_reaches_for_the_simulator(self) -> None:
        """Structural: no replay or simulation fallback exists to be reached."""

        source = Path(base.__file__).parent / "rendering/certified_generation.py"
        text = source.read_text(encoding="utf-8")
        for forbidden in ("simulate_election", "replay_certified_generation",
                          "reproduce_missing_draws"):
            self.assertNotIn(forbidden, text, forbidden)


class HistoryInputRevisionReuseTests(unittest.TestCase):
    """What a polling refresh may and may not invalidate.

    Reproduces the shape of ``9007c19 chore: refresh polling snapshot``, which
    took the publication down. That refresh rewrote the whole poll-of-polls
    table, and buried in it were two *retroactive revisions* of historical
    trend estimates -- 2026-02-18 SD 21.0 -> 21.1 and 2026-02-20 M 17.8 -> 17.9
    with L 2.0 -> 2.1. Those are effective model inputs, so every
    reconstructed point whose dynamics window reaches them became stale, and
    the gate's absolute "an existing point is never rerun" was violated for a
    legitimate reason.

    The two halves of the rule are asserted separately, because the incident
    turned on telling them apart:

    * a genuine revision invalidates the affected dates and *only* those,
      leaving everything earlier reusable -- otherwise a refresh degenerates
      into a ~300-point rebuild, which is how the curve stopped being
      extended before the resume cache existed;
    * churn in transport identity, row order and acquisition metadata --
      ``poll_id``, ``source_row``, ``retrieved_at`` -- invalidates nothing,
      even though it rewrites most bytes in the file.

    Hermetic: the real input tables are copied to a temporary tree and edited
    there, so nothing in the repository is touched and the assertions do not
    depend on today's snapshot.
    """

    INPUT_FILES = (
        "pollofpolls/individual_polls.csv",
        "pollofpolls/pollofpolls_timeseries.csv",
        "pollofpolls/swedishpolls_individual_polls.csv",
        "elections/riksdag_election_results.csv",
        "geography/constituency_party_votes_2014_2022.csv",
        "geography/constituency_electorates_2014_2026.csv",
    )
    TIMESERIES = "pollofpolls/pollofpolls_timeseries.csv"
    ELECTION = date(2026, 9, 13)

    def _inputs_tree(self) -> Path:
        destination = Path(self.enterContext(tempfile.TemporaryDirectory())) / "processed"
        for relative in self.INPUT_FILES:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPOSITORY_ROOT / "data/processed" / relative, target)
        return destination

    def _fingerprints(self, data_dir: Path, probes: list[date]) -> dict[date, str]:
        inputs = EffectiveInputs(
            data_dir, election_date=self.ELECTION, seed=DEFAULT_SIMULATION_SEED
        )
        return {probe: inputs.fingerprint(probe) for probe in probes}

    @staticmethod
    def _timeseries_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or []), list(reader)

    @staticmethod
    def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _probe_dates(self, path: Path) -> list[date]:
        """Observation dates spread across the campaign year, from the data."""

        _, rows = self._timeseries_rows(path)
        campaign = [
            date.fromisoformat(row["date"])
            for row in rows
            if row["date"].startswith("2026-")
        ]
        self.assertGreater(len(campaign), 40, "too few 2026 rows to probe")
        return campaign[:: max(1, len(campaign) // 12)]

    def test_a_retroactive_revision_invalidates_only_from_its_own_date(self) -> None:
        data_dir = self._inputs_tree()
        timeseries = data_dir / self.TIMESERIES
        probes = self._probe_dates(timeseries)
        baseline = self._fingerprints(data_dir, probes)

        # The incident's shape: one historical estimate nudged by 0.1, exactly
        # as pollofpolls.se revised 2026-02-18. REST is derived, so it absorbs
        # the change the way the real refresh did.
        fieldnames, rows = self._timeseries_rows(timeseries)
        revised_date = probes[len(probes) // 2]
        for row in rows:
            if row["date"] == revised_date.isoformat():
                row["SD"] = f"{float(row['SD']) + 0.1:.1f}"
                break
        else:  # pragma: no cover - the probe came from this file
            self.fail(f"no timeseries row for {revised_date}")
        self._write_rows(timeseries, fieldnames, rows)

        after = self._fingerprints(data_dir, probes)
        changed = {probe for probe in probes if baseline[probe] != after[probe]}

        # It is detected at all: a real input change must not be reused away.
        self.assertTrue(changed, "a revised historical estimate went undetected")
        # And it is detected on the revised date itself, which is what makes
        # "only from its own date" literal rather than merely "somewhere at or
        # after it". The date's own opinion.central reads that very row, so
        # this is the strongest single point in the assertion.
        self.assertIn(
            revised_date, changed,
            f"the revised date {revised_date} was not itself invalidated",
        )
        # And it is contained: nothing before the revision is invalidated.
        self.assertTrue(
            all(probe >= revised_date for probe in changed),
            f"a point before {revised_date} was invalidated: {sorted(changed)}",
        )
        earlier = [probe for probe in probes if probe < revised_date]
        self.assertTrue(earlier, "no probe precedes the revision")
        for probe in earlier:
            self.assertEqual(baseline[probe], after[probe], probe)

    def test_identity_order_and_refresh_metadata_churn_invalidates_nothing(self) -> None:
        """The half that makes a daily refresh survivable.

        A refresh rewrites nearly every row of both poll files: ``poll_id`` is
        a hash that includes ``source_row``, so inserting one poll renumbers
        and re-identifies thousands of unrelated observations. None of that is
        a model input, and treating it as one would rebuild the curve daily.
        """

        data_dir = self._inputs_tree()
        probes = self._probe_dates(data_dir / self.TIMESERIES)
        baseline = self._fingerprints(data_dir, probes)

        for relative in ("pollofpolls/swedishpolls_individual_polls.csv",
                         "pollofpolls/individual_polls.csv"):
            path = data_dir / relative
            fieldnames, rows = self._timeseries_rows(path)
            # One new id per poll, kept consistent across that poll's rows so
            # the file stays loadable -- this is churn, not corruption.
            remapped = {
                old: f"churn-{index:08d}"
                for index, old in enumerate(sorted({row["poll_id"] for row in rows}))
            }
            for offset, row in enumerate(rows):
                row["poll_id"] = remapped[row["poll_id"]]
                if "source_row" in row and row["source_row"]:
                    row["source_row"] = str(offset + 10_000)
                if "retrieved_at" in row and row["retrieved_at"]:
                    row["retrieved_at"] = "2026-09-08T04:15:00Z"
            rows.reverse()
            self._write_rows(path, fieldnames, rows)
            self.assertNotEqual(
                compute_file_sha256(path),
                compute_file_sha256(REPOSITORY_ROOT / "data/processed" / relative),
                "the churn fixture must actually change the file",
            )

        after = self._fingerprints(data_dir, probes)
        for probe in probes:
            self.assertEqual(
                baseline[probe], after[probe],
                f"{probe} was invalidated by identity/order/metadata churn alone",
            )



PUBLICATION_WORKFLOW = (
    REPOSITORY_ROOT / ".github/workflows/election-simulator-publication.yml"
)


class HistoryWorkersPlumbingTests(unittest.TestCase):
    """One named path from the CLI to the backfill, and a serial default.

    The 2026-09-08 publication died because the reconstructed-curve backfill
    ran 119 points on one core and exhausted the job timeout. The remedy is a
    worker count production can ask for -- but only production, and only
    explicitly: forking processes is not something a library does behind an
    in-process caller's back.
    """

    def test_every_hop_defaults_to_serial(self) -> None:
        """A default that forked would change behaviour for every caller."""

        import inspect

        from scripts import election_automation_base as base
        from scripts.forecast_history import generate

        self.assertEqual(generate.DEFAULT_HISTORY_WORKERS, 1)
        for function, name in (
            (generate.build_history, "workers"),
            (generate.backfill_reconstructed_curve, "workers"),
            (base.run_automation, "history_workers"),
        ):
            with self.subTest(function=function.__name__):
                parameter = inspect.signature(function).parameters[name]
                self.assertEqual(parameter.default, 1)
        # run_production_event is reached through the facade's projection
        # wrapper, which forwards **kwargs and so reports no signature of its
        # own. Its default is asserted at the definition instead.
        source = Path(base.__file__).read_text(encoding="utf-8")
        self.assertIn("    history_workers: int = DEFAULT_HISTORY_WORKERS,\n", source)
        self.assertIn("history_workers=history_workers,", source)
        self.assertIn("workers=resolve_history_workers(history_workers),", source)

    def test_the_cli_exposes_the_flag_and_defaults_to_serial(self) -> None:
        from scripts.election_automation_base import build_parser

        parsed = build_parser().parse_args(["--site-repo", "/tmp/site"])
        self.assertEqual(parsed.history_workers, 1)
        parsed = build_parser().parse_args(
            ["--site-repo", "/tmp/site", "--history-workers", "4"]
        )
        self.assertEqual(parsed.history_workers, 4)

    def test_main_forwards_the_flag_to_run_automation(self) -> None:
        from scripts import election_automation_base as base

        captured: dict[str, object] = {}

        def fake_run_automation(*args, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                summary=SimpleNamespace(render=lambda: ""),
                to_dict=lambda: {},
                status="SOURCE_CHECKED",
            )

        # main() prints the rendered summary and the result JSON; swallow both
        # so the suite's output stays readable.
        with patch.object(base, "run_automation", fake_run_automation), \
                contextlib.redirect_stdout(io.StringIO()):
            base.main([
                "--site-repo", "/tmp/site", "--mode", "probe", "--history-workers", "4",
            ])
        self.assertEqual(captured.get("history_workers"), 4)

    def test_production_requests_a_bounded_count(self) -> None:
        from scripts.forecast_history.generate import (
            PRODUCTION_HISTORY_WORKERS,
            resolve_history_workers,
        )

        workflow = PUBLICATION_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn(
            f"--history-workers {PRODUCTION_HISTORY_WORKERS}", workflow,
            "the publication workflow must request the production worker count",
        )
        # Bounded, not "all cores": the runner also has the publication to do.
        self.assertLessEqual(PRODUCTION_HISTORY_WORKERS, 8)
        self.assertGreater(PRODUCTION_HISTORY_WORKERS, 1)
        self.assertLessEqual(
            resolve_history_workers(PRODUCTION_HISTORY_WORKERS), os.cpu_count() or 1
        )

    def test_the_job_timeout_still_matches_the_benchmark_guard(self) -> None:
        """The constraint that makes parallelism the right fix, not more time.

        publication_fallback.PUBLICATION_MAX_RUNTIME is not an independent
        constant: it is this job's timeout, and the benchmark's protected
        interval is derived by subtracting it from the 20:30Z pre-warm to get
        18:30Z. Raising the timeout without raising that value would let a
        publication hold election-simulator-production across a frozen
        capture, which is exactly the contention the preflight guard exists to
        prevent -- so the two must move together or not at all.
        """

        from scripts.publication_fallback import PUBLICATION_MAX_RUNTIME

        workflow = PUBLICATION_WORKFLOW.read_text(encoding="utf-8")
        publish = workflow.split("\n  publish:\n", 1)[1]
        declared = re.search(r"timeout-minutes:\s*(\d+)", publish)
        self.assertIsNotNone(declared, "the publish job must declare a timeout")
        self.assertEqual(
            int(declared.group(1)),
            int(PUBLICATION_MAX_RUNTIME.total_seconds() // 60),
            "publish timeout-minutes and PUBLICATION_MAX_RUNTIME disagree; the "
            "benchmark protected interval is computed from the latter",
        )


if __name__ == "__main__":
    unittest.main()
