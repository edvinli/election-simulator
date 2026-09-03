"""Scheduled, fail-closed ElectionSimulator publication orchestration.

The GitHub Actions entry point in ``.github/workflows/election-simulator-
publication.yml`` calls this module.  The module intentionally keeps the
scientific pipeline in :mod:`scripts.publication_pipeline`: acquisition is
staged and validated first, the polling change is committed, and one
certified ``SimulationResult`` is then handed to the existing archive and
static exporters.  History and the website mirror consume that same result;
neither path invokes the simulator.

No operation in this module performs a live fetch by default when imported.
The CLI is the scheduled boundary and its explicit ``probe``, ``dry_run``,
and ``publish`` modes make the externally-mutating version-control operations
unambiguous.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import tempfile
from time import monotonic
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

from scripts.forecast_history.contract import (
    deterministic_history_sha256,
    validate_history_contract,
    write_history_json,
)
from scripts.forecast_history.generate import update_history_with_production_result
from scripts.pollofpolls.__main__ import (
    PollingValidationError,
    refresh_snapshot,
)
from scripts.pollofpolls.acquire import AcquisitionError
from scripts.publication_pipeline.pipeline import (
    PipelineRun,
    run_publication_pipeline,
)
from scripts.prospective_archive.archive import _validate_index
from scripts.site_publisher.publisher import (
    GENERATION_FILES,
    SITE_HISTORY_RELATIVE,
    SITE_PUBLICATION_RELATIVE,
    publish_generation_to_site,
    sync_history_to_site,
)
from scripts.static_exporter import (
    validate_publication_version,
    validate_published_directory,
)
from scripts.simulator.config import DEFAULT_ELECTION_DATE, DEFAULT_SIMULATION_SEED
from scripts.simulator.reproducibility import compute_file_sha256, get_git_commit_hash


STOCKHOLM = ZoneInfo("Europe/Stockholm")
ELECTION_DAY = date.fromisoformat(DEFAULT_ELECTION_DATE)
DAILY_SCHEDULE_UTC = "0 4 * * *"
INTRADAY_SCHEDULE_UTC = "0 6,8,10,12,14,16,18,20 * * *"
PRODUCTION_SAMPLES = 100_000
VALID_MODES = ("probe", "dry_run", "publish")
AUTOMATION_ENABLED_ENV = "ELECTION_AUTOMATION_ENABLED"
SOURCE_PROVENANCE_DIRECT_LIVE = "DIRECT_LIVE_FETCH"
SOURCE_PROVENANCE_VERIFIED_FALLBACK = "VERIFIED_STALE_FALLBACK"
StageCallback = Callable[[str, str, float | None], None]
JEKYLL_BUILD_TIMEOUT_SECONDS = 5 * 60
BROWSER_SMOKE_TIMEOUT_SECONDS = 15 * 60
COMMAND_TERMINATION_GRACE_SECONDS = 5.0
SENSITIVE_ENV_NAME_FRAGMENTS = (
    "ACCESS_KEY",
    "AUTHORIZATION",
    "CREDENTIAL",
    "PASSWORD",
    "PRIVATE_KEY",
    "SECRET",
    "TOKEN",
)

MODEL_RELEVANT_INPUTS: tuple[Path, ...] = (
    Path("data/processed/pollofpolls/pollofpolls_timeseries.csv"),
    Path("data/processed/pollofpolls/individual_polls.csv"),
    Path("data/processed/pollofpolls/swedishpolls_individual_polls.csv"),
)

POLLING_STATUS_VALUES = {
    "SOURCE_UPDATED",
    "SOURCE_UNCHANGED",
    "SOURCE_UNAVAILABLE_USING_VERIFIED_SNAPSHOT",
}


class AutomationError(RuntimeError):
    """A fail-closed automation stage failed."""


def _log_stage(stage: str, event: str, elapsed: float | None = None) -> None:
    """Emit one immediately visible Actions log line without buffered output."""

    suffix = f" elapsed={elapsed:.3f}s" if elapsed is not None else ""
    print(f"[election-automation] {stage} {event}{suffix}", flush=True)


@contextmanager
def _timed_stage(stage: str, callback: StageCallback | None):
    """Report a stage boundary while leaving its implementation untouched."""

    if callback is None:
        yield
        return
    started = monotonic()
    callback(stage, "START", None)
    try:
        yield
    except Exception:
        callback(stage, "FAIL", monotonic() - started)
        raise
    callback(stage, "DONE", monotonic() - started)


class AfterElectionDay(AutomationError):
    """The explicit stop guard has passed the election date."""


@dataclass
class PollingRefresh:
    """Evidence from one staged acquisition/normalization/validation run."""

    status: str
    changed: bool
    old_hash: str | None
    new_hash: str | None
    messages: list[str] = field(default_factory=list)
    source_commit: str | None = None
    pushed: bool = False
    installed: bool = False
    source_provenance: str = SOURCE_PROVENANCE_DIRECT_LIVE
    staged_root: str | None = None
    acquisition_diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "model_relevant_inputs_changed": self.changed,
            "old_hash": self.old_hash,
            "new_hash": self.new_hash,
            "messages": list(self.messages),
            "source_commit": self.source_commit,
            "pushed": self.pushed,
            "installed": self.installed,
            "source_provenance": self.source_provenance,
            "staged_root": self.staged_root,
            "acquisition_diagnostics": list(self.acquisition_diagnostics),
        }


@dataclass
class AutomationSummary:
    """Stable GitHub Actions summary fields."""

    run_type: str
    forecast_as_of: str
    mode: str = "publish"
    pop_estimate_date: str = "UNAVAILABLE"
    polling_source_status: str = "UNAVAILABLE"
    polling_source_provenance: str = "UNAVAILABLE"
    model_inputs_changed: bool = False
    simulation_samples: int = 0
    publication_generation: str = "NONE"
    history_current_point: str = "NONE"
    simulator_commit: str = "UNAVAILABLE"
    website_commit: str = "NONE"
    deployment_status: str = "NOT_RUN"
    recovery_status: str = "NONE"
    failure: str | None = None

    def render(self) -> str:
        lines = [
            f"Run type: {self.run_type}",
            f"Mode: {self.mode.upper()}",
            f"Forecast as_of: {self.forecast_as_of}",
            f"PoP estimate date: {self.pop_estimate_date}",
            f"Polling source status: {self.polling_source_status}",
            f"Polling source provenance: {self.polling_source_provenance}",
            f"Model-relevant inputs changed: {'yes' if self.model_inputs_changed else 'no'}",
            f"Simulation samples: {self.simulation_samples}",
            f"Publication generation: {self.publication_generation}",
            f"History current point: {self.history_current_point}",
            f"Simulator commit: {self.simulator_commit}",
            f"Website commit: {self.website_commit}",
            f"Deployment status: {self.deployment_status}",
            f"Recovery status: {self.recovery_status}",
        ]
        if self.failure:
            lines.append(f"Failure: {self.failure}")
        return "\n".join(lines) + "\n"


@dataclass
class AutomationResult:
    """Machine-readable result returned by :func:`run_automation`."""

    status: str
    summary: AutomationSummary
    polling: PollingRefresh | None = None
    pipeline: PipelineRun | None = None
    history: dict[str, Any] | None = None
    website: dict[str, Any] | None = None
    acquisition_diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary.render(),
            "polling": self.polling.to_dict() if self.polling else None,
            "acquisition_diagnostics": list(self.acquisition_diagnostics)
            or (self.polling.to_dict().get("acquisition_diagnostics", []) if self.polling else []),
            "pipeline": self.pipeline.to_dict() if self.pipeline else None,
            "history": {
                "deterministic_content_sha256": self.history.get("deterministic_content_sha256"),
                "current_point": next(
                    (
                        {
                            "date": point["date"],
                            "provenance": point["provenance"],
                        }
                        for point in self.history.get("series", [])
                        if point.get("provenance") == "current_production"
                    ),
                    None,
                ),
            }
            if self.history
            else None,
            "website": self.website,
        }


def current_stockholm_date(now: datetime | None = None) -> date:
    """Return the current calendar date in the required Stockholm zone."""

    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        raise ValueError("now must include a timezone")
    return value.astimezone(STOCKHOLM).date()


def guard_election_date(today: date, election_date: date = ELECTION_DAY) -> None:
    """Stop cleanly after election day, before acquisition or simulation."""

    if today > election_date:
        raise AfterElectionDay(
            f"automation stopped after election day {election_date.isoformat()} "
            f"(Stockholm date is {today.isoformat()})"
        )


def classify_run_type(
    *,
    event_name: str | None = None,
    schedule: str | None = None,
) -> str:
    """Map GitHub event context to the three public run-type labels."""

    if event_name == "workflow_dispatch":
        return "MANUAL"
    if schedule == DAILY_SCHEDULE_UTC:
        return "DAILY"
    return "POLL_CHANGE"


def resolve_mode(
    *,
    event_name: str | None,
    mode: str | None,
    commit: bool,
    push: bool,
) -> str:
    """Resolve one explicit operating mode and reject ambiguous mutations."""

    if mode is None:
        # Direct Python callers from the pre-mode API retain a safe default;
        # the workflow_dispatch boundary always supplies an explicit choice.
        resolved = "publish" if commit else "dry_run"
    else:
        resolved = str(mode).strip().lower()
    if resolved not in VALID_MODES:
        raise AutomationError(f"mode must be one of {', '.join(VALID_MODES)}")
    if event_name == "workflow_dispatch" and mode is None:
        raise AutomationError("workflow_dispatch requires an explicit mode")
    if resolved in {"probe", "dry_run"} and (commit or push):
        raise AutomationError(f"{resolved} mode cannot commit or push")
    if resolved == "publish" and not commit:
        raise AutomationError("publish mode requires commit=True")
    if push and not commit:
        raise AutomationError("push requires commit=True")
    return resolved


def automation_enabled_for_event(
    *,
    event_name: str | None,
    enabled: str | bool | None = None,
) -> bool:
    """Apply the repository kill switch only to scheduled events.

    Manual dispatch is an explicit operator action and is therefore always
    allowed.  The repository variable is fail-closed: a missing value is
    treated as ``false``, and only an explicit non-false value enables
    scheduled events.
    """

    if event_name == "workflow_dispatch":
        return True
    if enabled is None:
        enabled = os.environ.get(AUTOMATION_ENABLED_ENV, "false")
    return str(enabled).strip().lower() != "false"


def should_publish(
    run_type: str,
    *,
    model_inputs_changed: bool,
    mode: str = "publish",
    pending_publication: bool = False,
) -> bool:
    """Decide whether the already-validated snapshot merits production."""

    if mode == "probe":
        return False
    if run_type == "DAILY" or run_type == "MANUAL":
        return True
    return model_inputs_changed or pending_publication


def model_relevant_snapshot_sha256(repo_root: Path | str) -> str:
    """Hash only deterministic model-relevant processed polling content."""

    root = Path(repo_root).resolve()
    digest = hashlib.sha256()
    for relative in MODEL_RELEVANT_INPUTS:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise AutomationError(f"Missing model-relevant polling input: {path}")
        content = path.read_bytes()
        # Include the relative name and length so concatenation cannot be
        # ambiguous while keeping the hash independent of local filesystem
        # paths and retrieval timestamps outside these files.
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(content)).encode("ascii"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def latest_pop_observation_date(
    timeseries_file: Path | str,
    *,
    as_of: date | str,
) -> str:
    """Find the latest actual PoP row on or before ``as_of``.

    No row is synthesized for ``as_of`` when the source has not published one.
    """

    target = as_of if isinstance(as_of, date) else date.fromisoformat(str(as_of))
    latest: date | None = None
    with Path(timeseries_file).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = str(row.get("date") or "").strip()
            if not raw:
                continue
            try:
                observed = date.fromisoformat(raw)
            except ValueError as exc:
                raise AutomationError(f"Invalid PoP date in {timeseries_file}: {raw!r}") from exc
            if observed <= target and (latest is None or observed > latest):
                latest = observed
    if latest is None:
        raise AutomationError(
            f"No Poll of Polls observation exists on or before {target.isoformat()}"
        )
    return latest.isoformat()


def _run_git(
    repo: Path,
    args: Sequence[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run Git without emitting command output that could contain credentials."""

    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            check=check,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        command = "git " + " ".join(str(item) for item in args[:2])
        raise AutomationError(f"{command} failed") from exc


def _assert_clean(repo: Path, *, label: str) -> None:
    result = _run_git(repo, ["status", "--porcelain", "--untracked-files=all"])
    if result.stdout.strip():
        raise AutomationError(f"{label} worktree is not clean")


def _copy_tree_contents(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    if source.is_symlink() or not source.is_dir():
        raise AutomationError(f"Expected a regular directory: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)


def _copy_file_atomic(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise AutomationError(f"Expected a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sync_tree_contents(source: Path, destination: Path) -> None:
    """Install an already validated staged tree without removing old files."""

    if not source.is_dir():
        raise AutomationError(f"Staged tree is missing: {source}")
    for item in sorted(source.rglob("*")):
        if item.is_dir():
            continue
        if item.is_symlink() or not item.is_file():
            raise AutomationError(f"Staged polling tree contains a non-file: {item}")
        _copy_file_atomic(item, destination / item.relative_to(source))


def _git_commit_paths(
    repo: Path,
    paths: Sequence[str],
    message: str,
    *,
    commit: bool,
    push: bool,
    push_ref: str,
) -> tuple[str | None, bool]:
    """Commit/push exactly the requested generated or polling paths."""

    if not commit:
        return None, False
    _run_git(repo, ["add", "--", *paths])
    staged = _run_git(repo, ["diff", "--cached", "--quiet"], check=False)
    if staged.returncode == 0:
        return get_git_commit_hash(repo), False
    _run_git(repo, ["commit", "-m", message])
    commit_hash = _run_git(repo, ["rev-parse", "HEAD"]).stdout.strip()
    pushed = False
    if push:
        # Do not use --verbose: remote output can echo a credential-bearing
        # URL when a runner's credential helper is configured unexpectedly.
        _run_git(repo, ["push", "origin", f"HEAD:{push_ref}"])
        pushed = True
    _assert_clean(repo, label="post-commit")
    return commit_hash, pushed


def _source_provenance(
    refreshed: Mapping[str, Any],
    messages: Sequence[str],
) -> str:
    """Classify live versus verified-stale acquisition evidence.

    The refresh manifest is authoritative when present.  Fallback messages
    are also recognized because test/adapter refreshers may return only the
    public acquisition messages.  A stale retained payload is never called a
    direct live fetch merely because normalization succeeded.
    """

    lowered = [str(message).lower() for message in messages]
    if any(
        "retained verified" in message
        or "wayback" in message
        or "archive replay" in message
        or "fallback" in message
        or "first-party host unavailable" in message
        for message in lowered
    ):
        return SOURCE_PROVENANCE_VERIFIED_FALLBACK
    diagnostics = refreshed.get("acquisition_diagnostics")
    if isinstance(diagnostics, Sequence):
        methods = {
            str(diagnostic.get("retrieval_method", "")).lower()
            for diagnostic in diagnostics
            if isinstance(diagnostic, Mapping)
        }
        if any("wayback" in method or "archive" in method for method in methods):
            return SOURCE_PROVENANCE_VERIFIED_FALLBACK
        if any(bool(diagnostic.get("retained_previous")) for diagnostic in diagnostics if isinstance(diagnostic, Mapping)):
            return SOURCE_PROVENANCE_VERIFIED_FALLBACK
    manifest = refreshed.get("manifest")
    if isinstance(manifest, Mapping):
        sources = manifest.get("sources")
        if isinstance(sources, Mapping):
            methods = {
                str(record.get("retrieval_method", "")).lower()
                for record in sources.values()
                if isinstance(record, Mapping)
            }
            if methods and any(
                "wayback" in method or "archive" in method or "fallback" in method
                for method in methods
            ):
                return SOURCE_PROVENANCE_VERIFIED_FALLBACK
            if methods and all(
                "http" in method or "live" in method or "direct" in method
                for method in methods
            ):
                return SOURCE_PROVENANCE_DIRECT_LIVE
    return SOURCE_PROVENANCE_DIRECT_LIVE


def refresh_polling_snapshot(
    repo_root: Path | str,
    *,
    commit: bool = False,
    push: bool = False,
    push_ref: str = "main",
    allow_archive_fallback: bool = True,
    timeout: float = 45.0,
    refresh_fn: Callable[..., dict[str, Any]] = refresh_snapshot,
    install: bool | None = None,
    staging_directory: Path | str | None = None,
    stage_callback: StageCallback | None = None,
) -> PollingRefresh:
    """Acquire into a temporary tree, validate, then conditionally install.

    A semantic no-op leaves both raw/processed content and manifests in the
    checked-out repository untouched.  A source failure that retains the old
    verified raw payload is reported as unavailable, never as proof that the
    source is current.
    """

    root = Path(repo_root).resolve()
    # A non-committing invocation is a dry-run by default.  Acquisition still
    # runs in a disposable, validated tree and semantic status is returned,
    # but the checked-out repository is never left dirty.  Callers that need
    # to install an uncommitted fixture can opt in explicitly with
    # ``install=True``; the production workflow always commits the snapshot.
    if install is None:
        install = commit
    if push and not commit:
        raise AutomationError("push requires commit=True")
    _assert_clean(root, label="simulator")
    old_hash = model_relevant_snapshot_sha256(root)
    # The constants resolve against the source checkout where this module was
    # imported.  Derive repository-relative paths explicitly for a clean
    # worktree or a test fixture rooted elsewhere.
    current_raw = root / Path("data/raw/pollofpolls")
    current_processed = root / Path("data/processed/pollofpolls")
    current_readme = root / Path("data/README.md")

    temporary_context = (
        tempfile.TemporaryDirectory(prefix="election-poll-refresh-")
        if staging_directory is None
        else nullcontext(str(Path(staging_directory).resolve()))
    )
    with temporary_context as temporary_name:
        staging_root = Path(temporary_name)
        if staging_root.exists():
            if any(staging_root.iterdir()):
                raise AutomationError(f"Polling staging directory is not empty: {staging_root}")
        else:
            staging_root.mkdir(parents=True)
        staged_raw = staging_root / "data" / "raw" / "pollofpolls"
        staged_processed = staging_root / "data" / "processed" / "pollofpolls"
        staged_readme = staging_root / "data" / "README.md"
        _copy_tree_contents(current_raw, staged_raw)
        _copy_tree_contents(current_processed, staged_processed)
        if current_readme.is_file():
            staged_readme.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(current_readme, staged_readme)

        try:
            refreshed = refresh_fn(
                staged_raw,
                staged_processed,
                readme_path=staged_readme,
                offline=False,
                allow_archive_fallback=allow_archive_fallback,
                timeout=timeout,
                stage_callback=stage_callback,
            )
        except PollingValidationError:
            # The original checkout has not been touched.  Do not turn an
            # invalid acquisition into a polling commit or a production run.
            raise
        new_hash = model_relevant_snapshot_sha256(staging_root)
        messages = list(refreshed.get("messages", []))
        acquisition_diagnostics = list(refreshed.get("acquisition_diagnostics", []))
        source_provenance = _source_provenance(refreshed, messages)
        unavailable = any(
            "retained verified raw file" in message
            for message in messages
        ) or any(
            bool(diagnostic.get("retained_previous"))
            for diagnostic in acquisition_diagnostics
            if isinstance(diagnostic, Mapping)
        )
        status = (
            "SOURCE_UNAVAILABLE_USING_VERIFIED_SNAPSHOT"
            if unavailable
            else "SOURCE_UPDATED"
            if new_hash != old_hash
            else "SOURCE_UNCHANGED"
        )
        if status not in POLLING_STATUS_VALUES:
            raise AutomationError(f"Unknown polling source status: {status}")
        semantic_changed = new_hash != old_hash
        # Fallback provenance describes freshness, not semantic content.  A
        # validated mixed-source refresh may still contain a genuine change
        # from another source, and that deterministic change must be staged
        # and committed.  The caller will report the fallback explicitly and
        # must never describe it as a direct current fetch.
        changed = semantic_changed
        source_commit: str | None = None
        pushed = False
        installed = not semantic_changed
        if changed and install:
            _sync_tree_contents(staged_raw, current_raw)
            _sync_tree_contents(staged_processed, current_processed)
            if staged_readme.is_file():
                _copy_file_atomic(staged_readme, current_readme)
            installed_hash = model_relevant_snapshot_sha256(root)
            if installed_hash != new_hash:
                raise AutomationError("Installed polling snapshot hash differs from staged content")
            source_commit, pushed = _git_commit_paths(
                root,
                ["data/raw/pollofpolls", "data/processed/pollofpolls", "data/README.md"],
                "chore: refresh polling snapshot",
                commit=commit,
                push=push,
                push_ref=push_ref,
            )
            if commit:
                _assert_clean(root, label="simulator after polling commit")
            installed = True
        else:
            _assert_clean(root, label="simulator after unchanged polling check")
        return PollingRefresh(
            status=status,
            changed=changed,
            old_hash=old_hash,
            new_hash=new_hash,
            messages=messages,
            source_commit=source_commit,
            pushed=pushed,
            installed=installed,
            source_provenance=source_provenance,
            staged_root=str(staging_root) if staging_directory is not None else None,
            acquisition_diagnostics=acquisition_diagnostics,
        )


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise AutomationError(f"Expected a JSON object: {path}")
    return value


def _validate_archive_directory(archive_dir: Path, *, expected_generation: str | None = None) -> dict[str, Any]:
    """Validate index/path/hash links, including the newly appended snapshot."""

    index_path = archive_dir / "index.json"
    if not index_path.is_file() or index_path.is_symlink():
        raise AutomationError(f"Prospective archive index is missing: {index_path}")
    index = _load_json_object(index_path)
    try:
        _validate_index(index)
    except ValueError as exc:
        raise AutomationError(f"Prospective archive index failed validation: {exc}") from exc
    matching: dict[str, Any] | None = None
    for entry in index.get("snapshots", []):
        relative = entry.get("path")
        if not isinstance(relative, str):
            raise AutomationError("Prospective archive entry has no path")
        snapshot_path = (archive_dir / relative).resolve()
        if snapshot_path.parent.parent != archive_dir.resolve() or snapshot_path.name != "snapshot.json":
            raise AutomationError("Prospective archive entry escapes its generation directory")
        if not snapshot_path.is_file() or snapshot_path.is_symlink():
            raise AutomationError(f"Prospective archive snapshot is missing: {snapshot_path}")
        if compute_file_sha256(snapshot_path) != entry.get("snapshot_file_sha256"):
            raise AutomationError(f"Prospective archive snapshot hash mismatch: {snapshot_path}")
        snapshot = _load_json_object(snapshot_path)
        if snapshot.get("snapshot_id") != entry.get("snapshot_id"):
            raise AutomationError("Prospective archive snapshot identity mismatch")
        if snapshot.get("deterministic_payload_sha256") != entry.get("deterministic_payload_sha256"):
            raise AutomationError("Prospective archive payload hash mismatch")
        if bool(snapshot.get("duplicate_payload_allowed")) != bool(
            entry.get("duplicate_payload_allowed")
        ):
            raise AutomationError("Prospective archive duplicate-payload marker mismatch")
        if expected_generation and entry.get("generation_id") == expected_generation:
            matching = snapshot
    if expected_generation and matching is None:
        raise AutomationError(f"Prospective archive lacks generation {expected_generation}")
    return matching or {}


def _stage_copy_generation(
    *,
    staged_publication: Path,
    destination_publication: Path,
    generation: str,
) -> None:
    """Install one immutable version, but defer the current pointer switch."""

    source_version = staged_publication / "versions" / generation
    destination_version = destination_publication / "versions" / generation
    validate_publication_version(source_version, expected_generation=generation)
    if os.path.lexists(destination_version):
        if not destination_version.is_dir():
            raise AutomationError(f"Existing publication generation is not a directory: {destination_version}")
        validate_publication_version(destination_version, expected_generation=generation)
        for filename in GENERATION_FILES:
            if (destination_version / filename).read_bytes() != (source_version / filename).read_bytes():
                raise AutomationError(f"Existing publication generation differs: {destination_version}")
        if (destination_version / "manifest.json").read_bytes() != (source_version / "manifest.json").read_bytes():
            raise AutomationError(f"Existing publication manifest differs: {destination_version}")
        return
    versions = destination_publication / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{generation}.staging-", dir=versions))
    installed = False
    try:
        for filename in GENERATION_FILES:
            _copy_file_atomic(source_version / filename, temporary / filename)
        os.replace(temporary, destination_version)
        installed = True
        try:
            validate_publication_version(destination_version, expected_generation=generation)
        except Exception:
            # The destination did not exist before this call, so removing only
            # this just-installed generation restores the prior publication.
            shutil.rmtree(destination_version, ignore_errors=True)
            raise
    finally:
        if not installed and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _write_publication_pointer(publication_dir: Path, generation: str) -> None:
    """Atomically switch the static consumer pointer after all stage gates."""

    pointer = {
        "schema_version": _load_json_object(publication_dir / "versions" / generation / "manifest.json")["schema_version"],
        "publication_state": "COMPLETE",
        "publication_generation": generation,
        "path": f"versions/{generation}",
        "manifest_sha256": compute_file_sha256(publication_dir / "versions" / generation / "manifest.json"),
    }
    destination = publication_dir / "current.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(pointer, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _replace_bytes_atomic(destination: Path, content: bytes) -> None:
    """Atomically restore a pointer after a cross-repository gate fails."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.restore.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _capture_pointer(path: Path) -> bytes | None:
    """Capture one existing regular pointer, rejecting unsafe symlinks."""

    if os.path.lexists(path) and (path.is_symlink() or not path.is_file()):
        raise AutomationError(f"Publication pointer is not a regular file: {path}")
    return path.read_bytes() if path.exists() else None


def _restore_pointer(path: Path, content: bytes | None) -> None:
    """Restore exactly the pointer state captured before installation."""

    if content is None:
        if os.path.lexists(path):
            path.unlink()
        return
    _replace_bytes_atomic(path, content)


def _stage_copy_archive(staged_archive: Path, destination_archive: Path) -> None:
    """Merge the validated append-only archive without rewriting old entries."""

    destination_archive.mkdir(parents=True, exist_ok=True)
    staged_index = _load_json_object(staged_archive / "index.json")
    for entry in staged_index.get("snapshots", []):
        relative = entry.get("path")
        if not isinstance(relative, str):
            raise AutomationError("Prospective archive entry has no path")
        source = (staged_archive / relative).resolve()
        generation_dir = source.parent
        generation_name = generation_dir.name
        destination = destination_archive / generation_name
        if destination.exists():
            if not destination.is_dir() or (destination / "snapshot.json").read_bytes() != source.read_bytes():
                raise AutomationError(f"Existing archive generation differs: {destination}")
            continue
        temporary = Path(tempfile.mkdtemp(prefix=f".{generation_name}.staging-", dir=destination_archive))
        installed = False
        try:
            _copy_file_atomic(source, temporary / "snapshot.json")
            os.replace(temporary, destination)
            installed = True
        finally:
            if not installed and temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
    _copy_file_atomic(staged_archive / "index.json", destination_archive / "index.json")


def _load_history(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path)
    try:
        validate_history_contract(payload)
    except ValueError as exc:
        raise AutomationError(f"History artifact failed contract validation: {exc}") from exc
    return payload


def _copy_site_tree(source: Path, destination: Path) -> None:
    """Copy a website checkout to an isolated build tree, excluding caches."""

    ignored = shutil.ignore_patterns(".git", ".jekyll-cache", ".bundle", "_site", "_shots")
    shutil.copytree(source, destination, ignore=ignored)


def _terminate_process_group(
    process: subprocess.Popen[str],
    *,
    grace_seconds: float = COMMAND_TERMINATION_GRACE_SECONDS,
) -> None:
    """Terminate the command and every child it created, then reap its pipes."""

    def send(sig: signal.Signals) -> None:
        try:
            if os.name == "posix":
                # _run_command starts a new session, so the leader PID is also
                # the process-group ID shared by Node and Chromium descendants.
                os.killpg(process.pid, sig)
            elif sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
        except (ProcessLookupError, OSError):
            pass

    send(signal.SIGTERM)
    try:
        process.communicate(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        send(signal.SIGKILL)
        process.communicate()
    else:
        # The leader can exit while a detached Chromium child in the same
        # group ignores SIGTERM.  Kill any remaining group members even after
        # the leader and its captured pipes have been reaped.
        send(signal.SIGKILL)


def _run_command(
    command: Sequence[str],
    *,
    name: str,
    timeout_seconds: float,
    cwd: Path,
    env: Mapping[str, str] | None = None,
) -> None:
    started = monotonic()
    command_env = (
        {
            key: value
            for key, value in env.items()
            if not any(fragment in key.upper() for fragment in SENSITIVE_ENV_NAME_FRAGMENTS)
        }
        if env is not None
        else None
    )
    try:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=command_env,
            # These are fixed, allowlisted website commands. Their sanitized
            # environment contains no token/secret/password variables, so
            # inherit the Actions streams and expose browser progress live.
            stdout=None,
            stderr=None,
            text=True,
            start_new_session=os.name == "posix",
        )
    except OSError as exc:
        elapsed = monotonic() - started
        raise AutomationError(
            f"website command failed to start: {name} after {elapsed:.3f}s"
        ) from exc

    try:
        process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process)
        elapsed = monotonic() - started
        raise AutomationError(
            f"website command timed out: {name} after {elapsed:.3f}s"
        ) from exc

    elapsed = monotonic() - started
    if process.returncode != 0:
        # A failed Node parent may still have a live Chromium child.  Always
        # clean the whole isolated group before surfacing the command failure.
        _terminate_process_group(process)
        # Never include captured stderr: a misconfigured credential helper can
        # print a remote URL containing a token.
        raise AutomationError(
            f"website command failed: {name} (exit code {process.returncode}) "
            f"after {elapsed:.3f}s"
        )


def run_website_checks(
    site_root: Path,
    *,
    chrome_bin: str | None = None,
    stage_callback: StageCallback | None = _log_stage,
) -> dict[str, Any]:
    """Build Jekyll and run both real-browser smoke tests."""

    env = os.environ.copy()
    if chrome_bin:
        env["CHROME_BIN"] = chrome_bin
    with _timed_stage("jekyll build", stage_callback):
        _run_command(
            ["jekyll", "build", "--config", "_config.yml,_config.dev.yml"],
            name="jekyll build",
            timeout_seconds=JEKYLL_BUILD_TIMEOUT_SECONDS,
            cwd=site_root,
            env=env,
        )
    for test_name in ("forecast-timeseries.smoke.mjs", "government-builder.smoke.mjs"):
        with _timed_stage(test_name, stage_callback):
            _run_command(
                ["node", f"browser-tests/{test_name}", "_site"],
                name=test_name,
                timeout_seconds=BROWSER_SMOKE_TIMEOUT_SECONDS,
                cwd=site_root,
                env=env,
            )
    return {
        "status": "PASS",
        "checks": [
            "jekyll build",
            "forecast-timeseries browser smoke",
            "government-builder browser smoke",
            "zero console errors and no mobile horizontal overflow (smoke assertions)",
        ],
    }


def _stage_site(
    *,
    site_root: Path,
    staged_publication: Path,
    staged_history: Path,
    generation: str,
    website_check_fn: Callable[[Path], dict[str, Any]],
    stage_callback: StageCallback | None = None,
) -> dict[str, Any]:
    """Mirror publication/history into a disposable website tree and test it."""

    with _timed_stage("website sync/validation", stage_callback):
        publish_generation_to_site(
            site_repo=site_root,
            source_publication_dir=staged_publication,
            generation=generation,
            update_pointer=True,
        )
        sync_history_to_site(site_repo=site_root, source_history_path=staged_history)
        validate_published_directory(site_root / SITE_PUBLICATION_RELATIVE)
        validate_history_contract(_load_json_object(site_root / SITE_HISTORY_RELATIVE))
    with _timed_stage("Jekyll/browser tests", stage_callback):
        checks = website_check_fn(site_root)
    if not isinstance(checks, Mapping) or checks.get("status") != "PASS":
        raise AutomationError("Website checks did not return PASS")
    return dict(checks)


def _verify_website_artifacts(
    *,
    source_publication: Path,
    source_history: Path,
    site_root: Path,
    generation: str,
) -> None:
    source_version = source_publication / "versions" / generation
    site_version = site_root / SITE_PUBLICATION_RELATIVE / "versions" / generation
    for filename in GENERATION_FILES:
        if (source_version / filename).read_bytes() != (site_version / filename).read_bytes():
            raise AutomationError(f"Simulator and website publication differ: {filename}")
    if source_history.read_bytes() != (site_root / SITE_HISTORY_RELATIVE).read_bytes():
        raise AutomationError("Simulator and website history artifacts differ")


def _certified_current_generation(publication_dir: Path) -> dict[str, Any]:
    """Read and independently validate the source's durable live pointer."""

    pointer_path = publication_dir / "current.json"
    if not pointer_path.is_file() or pointer_path.is_symlink():
        raise AutomationError(f"Certified source publication pointer is missing: {pointer_path}")
    validate_published_directory(publication_dir)
    pointer = _load_json_object(pointer_path)
    generation = pointer.get("publication_generation")
    if not isinstance(generation, str) or not generation:
        raise AutomationError("Certified source publication pointer has no generation")
    version = publication_dir / "versions" / generation
    manifest = validate_publication_version(version, expected_generation=generation)
    if pointer.get("manifest_sha256") != compute_file_sha256(version / "manifest.json"):
        raise AutomationError("Certified source publication pointer does not match its manifest")
    return {"pointer": pointer, "generation": generation, "version": version, "manifest": manifest}


def _is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    result = _run_git(
        repo,
        ["merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
    )
    return result.returncode == 0


def _latest_model_input_commit(repo: Path) -> str | None:
    result = _run_git(
        repo,
        ["log", "-1", "--format=%H", "--", *(path.as_posix() for path in MODEL_RELEVANT_INPUTS)],
        check=False,
    )
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def _has_unpublished_polling_commit(repo: Path, publication_dir: Path) -> bool:
    """Detect a committed polling refresh not represented by the live forecast.

    This marker is intentionally derived from Git and the certified current
    manifest.  It survives an Actions retry and does not depend on an
    ephemeral workspace flag.  A later code-only commit does not count when
    the latest model-input commit is already an ancestor of the certified
    source commit.
    """

    try:
        current = _certified_current_generation(publication_dir)
    except (AutomationError, OSError, ValueError):
        return False
    source_commit = current["manifest"].get("source_git_commit")
    latest_input_commit = _latest_model_input_commit(repo)
    if not isinstance(source_commit, str) or not latest_input_commit:
        return False
    if source_commit == latest_input_commit:
        return False
    # If the current publication was built after the latest input commit, the
    # input is already represented.  Otherwise a descendant (or divergent
    # branch) is conservatively treated as pending and will be re-certified.
    return not _is_ancestor(repo, latest_input_commit, source_commit)


def _website_needs_recovery(*, source_repo: Path, site_repo: Path) -> tuple[bool, dict[str, Any]]:
    """Compare durable source/site artifacts for a no-simulation recovery."""

    source_publication = source_repo / "files" / "election-simulator"
    source_history = source_publication / "history" / "coalition-timeseries.json"
    source = _certified_current_generation(source_publication)
    generation = str(source["generation"])
    site_publication = site_repo / SITE_PUBLICATION_RELATIVE
    site_history = site_repo / SITE_HISTORY_RELATIVE
    site_pointer = site_publication / "current.json"
    needs_recovery = False
    if not site_pointer.is_file() or site_pointer.is_symlink():
        needs_recovery = True
    else:
        try:
            site = _certified_current_generation(site_publication)
            needs_recovery = site["generation"] != generation
        except (AutomationError, OSError, ValueError):
            needs_recovery = True
    source_version = source["version"]
    site_version = site_publication / "versions" / generation
    if not site_version.is_dir():
        needs_recovery = True
    else:
        for filename in GENERATION_FILES:
            if not (site_version / filename).is_file() or (
                site_version / filename
            ).read_bytes() != (source_version / filename).read_bytes():
                needs_recovery = True
                break
    if (
        not site_history.is_file()
        or not source_history.is_file()
        or site_history.read_bytes() != source_history.read_bytes()
    ):
        needs_recovery = True
    return needs_recovery, source


def _recover_website_from_source(
    *,
    source_repo: Path,
    site_repo: Path,
    website_check_fn: Callable[[Path], dict[str, Any]],
    commit: bool,
    push: bool,
    website_push_ref: str = "master",
) -> dict[str, Any] | None:
    """Mirror an already-certified source generation without simulation."""

    _assert_clean(source_repo, label="simulator before website recovery")
    _assert_clean(site_repo, label="website before website recovery")
    source_publication = source_repo / "files" / "election-simulator"
    source_history = source_publication / "history" / "coalition-timeseries.json"
    needs_recovery, source = _website_needs_recovery(
        source_repo=source_repo,
        site_repo=site_repo,
    )
    if not needs_recovery:
        return None
    generation = str(source["generation"])
    with tempfile.TemporaryDirectory(prefix="election-website-recovery-") as temporary_name:
        staged_site = Path(temporary_name) / "website"
        _copy_site_tree(site_repo, staged_site)
        checks = _stage_site(
            site_root=staged_site,
            staged_publication=source_publication,
            staged_history=source_history,
            generation=generation,
            website_check_fn=website_check_fn,
        )
    if not commit:
        checks.update(
            {
                "status": "RECOVERY_STAGED_NOT_INSTALLED",
                "deployment": "dry-run",
                "generation": generation,
            }
        )
        return checks

    pointer = site_repo / SITE_PUBLICATION_RELATIVE / "current.json"
    pointer_before = _capture_pointer(pointer)
    try:
        mirrored = _install_site_outputs(
            site_repo=site_repo,
            source_publication=source_publication,
            source_history=source_history,
            generation=generation,
            update_pointer=True,
        )
        checks.update(mirrored)
        _git_commit_paths(
            site_repo,
            ["files/election-simulator"],
            f"chore: recover election forecast {generation}",
            commit=True,
            push=push,
            push_ref=website_push_ref,
        )
        checks["status"] = "RECOVERED"
        checks["deployment"] = "PUSHED" if push else "COMMITTED_NOT_PUSHED"
        return checks
    except Exception:
        # A website gate or pointer failure must not leave a new live pointer.
        # Generated files may be left for diagnosis only after a successful
        # gate; the next workflow checkout starts from the durable remote.
        try:
            _restore_pointer(pointer, pointer_before)
        except Exception as restore_error:
            raise AutomationError("failed to restore website recovery pointer") from restore_error
        raise


def _install_source_outputs(
    *,
    staged_archive: Path,
    destination_archive: Path,
    staged_publication: Path,
    destination_publication: Path,
    staged_history: Path,
    destination_history: Path,
    generation: str,
    update_pointer: bool = True,
) -> None:
    """Install archive/history/version, then optionally switch current.json."""

    _stage_copy_archive(staged_archive, destination_archive)
    _stage_copy_generation(
        staged_publication=staged_publication,
        destination_publication=destination_publication,
        generation=generation,
    )
    # The history file is validated in the temporary tree before this point.
    _copy_file_atomic(staged_history, destination_history)
    if update_pointer:
        _write_publication_pointer(destination_publication, generation)
        validate_published_directory(destination_publication)
    else:
        validate_publication_version(
            destination_publication / "versions" / generation,
            expected_generation=generation,
        )
    validate_history_contract(_load_json_object(destination_history))
    _validate_archive_directory(destination_archive, expected_generation=generation)


def _install_site_outputs(
    *,
    site_repo: Path,
    source_publication: Path,
    source_history: Path,
    generation: str,
    update_pointer: bool = True,
) -> dict[str, Any]:
    """Install website files with an optional deferred pointer switch."""

    destination = site_repo / SITE_PUBLICATION_RELATIVE
    publish_generation_to_site(
        site_repo=site_repo,
        source_publication_dir=source_publication,
        generation=generation,
        update_pointer=False,
        allow_existing=True,
    )
    sync_history_to_site(site_repo=site_repo, source_history_path=source_history)
    if update_pointer:
        publish_generation_to_site(
            site_repo=site_repo,
            source_publication_dir=source_publication,
            generation=generation,
            update_pointer=True,
            allow_existing=True,
        )
    else:
        validate_publication_version(
            destination / "versions" / generation,
            expected_generation=generation,
        )
        # The pointer remains at the previous generation until both
        # repositories have passed their installation gates.
    if update_pointer:
        validate_published_directory(destination)
    validate_history_contract(_load_json_object(site_repo / SITE_HISTORY_RELATIVE))
    _verify_website_artifacts(
        source_publication=source_publication,
        source_history=source_history,
        site_root=site_repo,
        generation=generation,
    )
    return {
        "status": "MIRRORED",
        "generation": generation,
        "history": str(site_repo / SITE_HISTORY_RELATIVE),
    }


def run_production_event(
    repo_root: Path | str,
    *,
    site_repo: Path | str,
    forecast_as_of: date | str,
    election_date: date | str = ELECTION_DAY,
    generated_at_utc: str | None = None,
    seed: int = DEFAULT_SIMULATION_SEED,
    simulation_runner: Callable[..., Any] | None = None,
    projection_runner: Callable[..., Any] | None = None,
    campaign_path_simulator: Callable[..., Any] | None = None,
    history_updater: Callable[..., dict[str, Any]] | None = None,
    website_check_fn: Callable[[Path], dict[str, Any]] | None = None,
    commit: bool = False,
    push: bool = False,
    allow_duplicate_payload: bool = False,
    processed_root: Path | str | None = None,
    allow_custom_processed_root: bool = False,
    source_push_ref: str = "main",
    website_push_ref: str = "master",
    stage_callback: StageCallback | None = None,
    pipeline_observer: Callable[[PipelineRun], None] | None = None,
) -> tuple[PipelineRun, dict[str, Any], dict[str, Any] | None]:
    """Run exactly one production simulation and stage all consumers.

    All writes are deferred until the temporary publication, archive, history,
    website build, and browser checks have passed.  A failed check therefore
    leaves the previous source ``current.json`` and website pointer untouched.
    """

    root = Path(repo_root).resolve()
    site = Path(site_repo).resolve()
    if root == site:
        raise AutomationError("Simulator and website repositories must be different directories")
    _assert_clean(root, label="simulator before production")
    _assert_clean(site, label="website before production")
    as_of = forecast_as_of if isinstance(forecast_as_of, date) else date.fromisoformat(str(forecast_as_of))
    election = election_date if isinstance(election_date, date) else date.fromisoformat(str(election_date))
    # Keep sub-second provenance in the manifest.  Generation ids remain
    # second-sortable; the archive's explicit duplicate-payload marker handles
    # same-seed reruns while a true same-second generation collision still
    # fails closed.
    generated = generated_at_utc or datetime.now(timezone.utc).isoformat()
    selected_processed_root = (
        Path(processed_root).resolve()
        if processed_root is not None
        else root / "data" / "processed"
    )
    if selected_processed_root != (root / "data" / "processed").resolve() and not allow_custom_processed_root:
        raise AutomationError(
            "custom processed_root requires an explicit dry-run opt-in"
        )
    processed_root = selected_processed_root
    archive_destination = processed_root / "prospective_forecasts"
    publication_destination = root / "files" / "election-simulator"
    history_destination = publication_destination / "history" / "coalition-timeseries.json"
    if not history_destination.is_file():
        raise AutomationError(f"Existing history artifact is required: {history_destination}")
    existing_history = _load_history(history_destination)

    with tempfile.TemporaryDirectory(prefix="election-production-") as temporary_name:
        temporary = Path(temporary_name)
        staged_archive = temporary / "prospective_forecasts"
        _copy_tree_contents(archive_destination, staged_archive)
        staged_publication = temporary / "publication"
        staged_history = temporary / "history" / "coalition-timeseries.json"
        run = run_publication_pipeline(
            as_of=as_of.isoformat(),
            election_date=election.isoformat(),
            samples=PRODUCTION_SAMPLES,
            seed=seed,
            processed_root=processed_root,
            archive_dir=staged_archive,
            publication_dir=staged_publication,
            generated_at_utc=generated,
            append_archive=True,
            export_publication=True,
            simulation_runner=simulation_runner,
            allow_duplicate_payload=allow_duplicate_payload,
            allow_custom_processed_root=allow_custom_processed_root,
            simulation_repo_root=root,
            stage_callback=stage_callback,
        )
        if pipeline_observer is not None:
            pipeline_observer(run)
        if run.status != "PUBLISHED" or run.simulation_result is None:
            detail = run.error.get("message") if run.error else run.status
            raise AutomationError(f"production pipeline failed: {detail}")
        if (
            run.simulation_validation is None
            or int(run.simulation_validation.get("samples", 0)) != PRODUCTION_SAMPLES
        ):
            raise AutomationError("production publication did not produce exactly 100,000 draws")
        if run.snapshot is None or run.publication_manifest is None:
            raise AutomationError("production pipeline returned no certified publication metadata")
        result_as_of = getattr(getattr(run.simulation_result, "summary", None), "as_of", None)
        if str(result_as_of) != as_of.isoformat():
            raise AutomationError(
                "production simulation result as_of does not match the explicit forecast date"
            )
        generation = str(run.snapshot["generation_id"])
        validate_published_directory(staged_publication)
        _validate_archive_directory(staged_archive, expected_generation=generation)
        result = run.simulation_result
        payload_hash = str(run.snapshot["deterministic_payload_sha256"])
        source_commit = str(result.manifest.get("source_git_commit", ""))
        with _timed_stage("history update", stage_callback):
            selected_history_updater = history_updater or update_history_with_production_result
            history_kwargs = {
                "poll_file": processed_root / "pollofpolls" / "swedishpolls_individual_polls.csv",
                "timeseries_file": processed_root / "pollofpolls" / "pollofpolls_timeseries.csv",
                "archive_dir": staged_archive,
                "election_date": election,
                "publication_generation": generation,
                "deterministic_payload_sha256": payload_hash,
                "generated_at_utc": generated,
                "model_commit": source_commit,
                "source_worktree_clean": True,
            }
            if history_updater is not None:
                history_kwargs["projection_runner"] = projection_runner
                history_kwargs["campaign_path_simulator"] = campaign_path_simulator
            history = selected_history_updater(existing_history, result, **history_kwargs)
            write_history_json(staged_history, history)
            validate_history_contract(_load_json_object(staged_history))
            # The self-hash in the history payload is checked once more at the
            # production boundary so a future writer cannot accidentally mutate it
            # between construction and staging.
            if history["deterministic_content_sha256"] != deterministic_history_sha256(history):
                raise AutomationError("History deterministic hash is not self-consistent")

        website_check = website_check_fn or run_website_checks
        staged_site = temporary / "website"
        _copy_site_tree(site, staged_site)
        website = _stage_site(
            site_root=staged_site,
            staged_publication=staged_publication,
            staged_history=staged_history,
            generation=generation,
            website_check_fn=website_check,
            stage_callback=stage_callback,
        )

        if not commit:
            # A local invocation without --commit is a genuine dry-run: all
            # production gates execute against disposable trees, while the
            # live simulator and website repositories remain byte-for-byte
            # untouched.  The GitHub workflow supplies --commit.
            website["status"] = "STAGED_NOT_INSTALLED"
            website["deployment"] = "dry-run"
            return run, history, website

        # All publication gates passed.  The source pointer remains untouched
        # until archive/history/version installation succeeds.  Website staging
        # has already run the build and both real-browser checks.
        source_pointer = publication_destination / "current.json"
        website_pointer = site / SITE_PUBLICATION_RELATIVE / "current.json"
        source_pointer_before = _capture_pointer(source_pointer)
        website_pointer_before = _capture_pointer(website_pointer)
        _install_source_outputs(
            staged_archive=staged_archive,
            destination_archive=archive_destination,
            staged_publication=staged_publication,
            destination_publication=publication_destination,
            staged_history=staged_history,
            destination_history=history_destination,
            generation=generation,
            update_pointer=False,
        )
        website_live = _install_site_outputs(
            site_repo=site,
            source_publication=publication_destination,
            source_history=history_destination,
            generation=generation,
            update_pointer=False,
        )
        website.update(website_live)

        # There is no cross-filesystem transaction spanning two repositories,
        # so prepare and validate both complete trees first, then switch the
        # two tiny pointers.  If either pointer write or its final validation
        # fails, restore the exact pre-run bytes in both repositories.
        try:
            _write_publication_pointer(publication_destination, generation)
            _write_publication_pointer(site / SITE_PUBLICATION_RELATIVE, generation)
            validate_published_directory(publication_destination)
            validate_published_directory(site / SITE_PUBLICATION_RELATIVE)
            validate_history_contract(_load_json_object(history_destination))
            validate_history_contract(_load_json_object(site / SITE_HISTORY_RELATIVE))
            _verify_website_artifacts(
                source_publication=publication_destination,
                source_history=history_destination,
                site_root=site,
                generation=generation,
            )
        except Exception:
            try:
                _restore_pointer(source_pointer, source_pointer_before)
                _restore_pointer(website_pointer, website_pointer_before)
            except Exception as restore_error:
                raise AutomationError("failed to restore live publication pointers") from restore_error
            raise

        _git_commit_paths(
            root,
            ["data/processed/prospective_forecasts", "files/election-simulator"],
            f"chore: publish election forecast {as_of.isoformat()}",
            commit=True,
            push=push,
            push_ref=source_push_ref,
        )
        _git_commit_paths(
            site,
            ["files/election-simulator"],
            f"chore: sync election forecast {generation}",
            commit=True,
            push=push,
            push_ref=website_push_ref,
        )
        return run, history, website


def run_automation(
    repo_root: Path | str,
    *,
    site_repo: Path | str,
    event_name: str | None = None,
    schedule: str | None = None,
    mode: str | None = None,
    automation_enabled: str | bool | None = None,
    now: datetime | None = None,
    election_date: date | str = ELECTION_DAY,
    commit: bool = False,
    push: bool = False,
    website_check_fn: Callable[[Path], dict[str, Any]] | None = None,
    refresh_fn: Callable[..., dict[str, Any]] = refresh_snapshot,
    simulation_runner: Callable[..., Any] | None = None,
    projection_runner: Callable[..., Any] | None = None,
    campaign_path_simulator: Callable[..., Any] | None = None,
    generated_at_utc: str | None = None,
    stage_callback: StageCallback | None = None,
) -> AutomationResult:
    """Execute one explicit probe, dry-run, or publish event.

    ``probe`` ends after acquisition/normalization/validation.  ``dry_run``
    runs the complete simulation and website gates in disposable trees.  Only
    ``publish`` may install, commit, or push artifacts, and its caller must
    explicitly provide ``commit=True``.
    """

    root = Path(repo_root).resolve()
    site = Path(site_repo).resolve()
    election = election_date if isinstance(election_date, date) else date.fromisoformat(str(election_date))
    today = current_stockholm_date(now)
    run_type = classify_run_type(event_name=event_name, schedule=schedule)
    resolved_mode = mode or ("publish" if commit else "dry_run")
    summary = AutomationSummary(
        run_type=run_type,
        forecast_as_of=today.isoformat(),
        mode=resolved_mode,
    )
    polling: PollingRefresh | None = None

    def observe_pipeline(run: PipelineRun) -> None:
        if run.simulation_validation is not None:
            summary.simulation_samples = int(run.simulation_validation.get("samples", 0))

    try:
        guard_election_date(today, election)
        resolved_mode = resolve_mode(
            event_name=event_name,
            mode=mode,
            commit=commit,
            push=push,
        )
        summary.mode = resolved_mode
        if not automation_enabled_for_event(
            event_name=event_name,
            enabled=automation_enabled,
        ):
            summary.polling_source_status = "DISABLED_BY_REPOSITORY_KILL_SWITCH"
            summary.polling_source_provenance = "NOT_ACQUIRED"
            summary.deployment_status = "DISABLED_BY_REPOSITORY_KILL_SWITCH"
            return AutomationResult(
                status="DISABLED_BY_REPOSITORY_KILL_SWITCH",
                summary=summary,
            )

        acquisition_staging: Path | None = None
        acquisition_context = (
            tempfile.TemporaryDirectory(prefix="election-automation-acquisition-")
            if resolved_mode == "dry_run"
            else nullcontext(None)
        )
        with acquisition_context as acquisition_name:
            if acquisition_name is not None:
                acquisition_staging = Path(acquisition_name)
            effective_commit = commit and resolved_mode == "publish"
            effective_push = push and effective_commit
            polling = refresh_polling_snapshot(
                root,
                commit=effective_commit,
                push=effective_push,
                refresh_fn=refresh_fn,
                install=effective_commit,
                staging_directory=acquisition_staging,
                stage_callback=stage_callback,
            )
            summary.polling_source_status = polling.status
            summary.polling_source_provenance = polling.source_provenance
            summary.model_inputs_changed = polling.changed
            pop_timeseries = (
                Path(polling.staged_root) / "data/processed/pollofpolls/pollofpolls_timeseries.csv"
                if polling.staged_root
                else root / "data/processed/pollofpolls/pollofpolls_timeseries.csv"
            )
            summary.pop_estimate_date = latest_pop_observation_date(
                pop_timeseries,
                as_of=today,
            )

            # A source commit without a successful publication is a durable
            # pending marker.  It must force the next retry even though the
            # next network refresh produces identical semantic inputs.
            pending_publication = _has_unpublished_polling_commit(
                root,
                root / "files/election-simulator",
            )
            if pending_publication:
                summary.recovery_status = "POLLING_PUBLICATION_PENDING"

            if resolved_mode == "probe":
                summary.deployment_status = "PROBE_ONLY"
                return AutomationResult(status="SOURCE_CHECKED", summary=summary, polling=polling)

            publication_needed = should_publish(
                run_type,
                model_inputs_changed=polling.changed,
                mode=resolved_mode,
                pending_publication=pending_publication,
            )
            # An explicitly selected manual dry-run is itself a request to
            # exercise the complete production path, even when the caller
            # supplies an intraday-style schedule in a local/test context.
            # The legacy no-mode API retains its no-op intraday behavior so
            # source-change probes cannot unexpectedly simulate.
            if resolved_mode == "dry_run" and mode is not None:
                publication_needed = True

            # A certified source generation may already be ahead of the
            # website after a prior website push failure.  Repair that durable
            # mismatch before considering a new production event; this path
            # never invokes the simulator.  A new daily/manual/poll-change
            # publication takes precedence: it must produce the newest
            # forecast and mirror that result in one event rather than
            # short-circuiting on a stale website generation from an older
            # failed run.
            recovery: dict[str, Any] | None = None
            if not publication_needed and root != site and resolved_mode == "publish":
                # Only skip the recovery probe when there is no certified
                # source pointer to compare.  Once a source generation is
                # certified, a website build/browser failure is a real gate
                # failure and must not be silently converted to a no-op.
                try:
                    _certified_current_generation(root / "files/election-simulator")
                except (AutomationError, OSError, ValueError):
                    source_is_certified = False
                else:
                    source_is_certified = True
                if source_is_certified:
                    recovery = _recover_website_from_source(
                        source_repo=root,
                        site_repo=site,
                        website_check_fn=website_check_fn or run_website_checks,
                        commit=effective_commit,
                        push=effective_push,
                    )
            if recovery is not None:
                summary.recovery_status = "WEBSITE_RECOVERED" if effective_commit else "WEBSITE_RECOVERY_STAGED"
                summary.publication_generation = str(recovery.get("generation", "NONE"))
                summary.website_commit = get_git_commit_hash(site) if effective_commit else "NOT_COMMITTED"
                summary.deployment_status = str(
                    recovery.get("deployment", "STAGED_NOT_INSTALLED")
                )
                return AutomationResult(
                    status="WEBSITE_RECOVERED",
                    summary=summary,
                    polling=polling,
                    website=recovery,
                )

            if not publication_needed:
                summary.deployment_status = "NO_PUBLICATION_NEEDED"
                return AutomationResult(status="SOURCE_CHECKED", summary=summary, polling=polling)

            if resolved_mode == "dry_run":
                if acquisition_staging is None:
                    raise AutomationError("dry_run acquisition did not retain its staging tree")
                # Keep the entire validated processed snapshot disposable.  A
                # changed polling tree replaces only its corresponding staged
                # subtree, so the simulator sees precisely the inputs that
                # acquisition validated without dirtying the live checkout.
                with tempfile.TemporaryDirectory(prefix="election-automation-processed-") as processed_name:
                    staged_processed = Path(processed_name) / "processed"
                    _copy_tree_contents(root / "data/processed", staged_processed)
                    staged_polling = Path(polling.staged_root) / "data/processed/pollofpolls"
                    _sync_tree_contents(staged_polling, staged_processed / "pollofpolls")
                    _assert_clean(root, label="simulator before dry-run production")
                    _assert_clean(site, label="website before dry-run production")
                    run, history, website = run_production_event(
                        root,
                        site_repo=site,
                        forecast_as_of=today,
                        election_date=election,
                        generated_at_utc=generated_at_utc,
                        simulation_runner=simulation_runner,
                        projection_runner=projection_runner,
                        campaign_path_simulator=campaign_path_simulator,
                        website_check_fn=website_check_fn,
                        commit=False,
                        push=False,
                        allow_duplicate_payload=True,
                        processed_root=staged_processed,
                        allow_custom_processed_root=True,
                        stage_callback=stage_callback,
                        pipeline_observer=observe_pipeline,
                    )
            else:
                _assert_clean(root, label="simulator before production")
                run, history, website = run_production_event(
                    root,
                    site_repo=site,
                    forecast_as_of=today,
                    election_date=election,
                    generated_at_utc=generated_at_utc,
                    simulation_runner=simulation_runner,
                    projection_runner=projection_runner,
                    campaign_path_simulator=campaign_path_simulator,
                    website_check_fn=website_check_fn,
                    commit=effective_commit,
                    push=effective_push,
                    # A daily or manual publication is still a distinct
                    # immutable generation when its seeded draw payload is
                    # unchanged.  The archive marker preserves that event.
                    allow_duplicate_payload=True,
                    stage_callback=stage_callback,
                    pipeline_observer=observe_pipeline,
                )
        summary.simulation_samples = int(run.simulation_validation["samples"]) if run.simulation_validation else 0
        summary.publication_generation = str(run.snapshot["generation_id"]) if run.snapshot else "NONE"
        current = next(
            (point for point in history["series"] if point.get("provenance") == "current_production"),
            None,
        )
        summary.history_current_point = (
            f"{current['date']} ({current['provenance']})" if current else "NONE"
        )
        summary.simulator_commit = str(run.simulation_result.manifest.get("source_git_commit")) if run.simulation_result else "UNAVAILABLE"
        summary.website_commit = get_git_commit_hash(site) if effective_commit else "NOT_COMMITTED"
        summary.deployment_status = (
            "PUSHED"
            if effective_push
            else "PUBLISHED_NOT_PUSHED"
            if effective_commit
            else "STAGED_NOT_INSTALLED"
        )
        return AutomationResult(
            status="PUBLISHED",
            summary=summary,
            polling=polling,
            pipeline=run,
            history=history,
            website=website,
        )
    except AfterElectionDay as exc:
        summary.deployment_status = "STOPPED_AFTER_ELECTION"
        summary.failure = str(exc)
        return AutomationResult(status="STOPPED_AFTER_ELECTION", summary=summary)
    except Exception as exc:  # noqa: BLE001 - CLI converts all gates to one summary
        summary.deployment_status = "FAILED"
        summary.failure = str(exc)
        diagnostics = exc.diagnostics if isinstance(exc, AcquisitionError) else []
        return AutomationResult(
            status="FAILED",
            summary=summary,
            polling=polling,
            acquisition_diagnostics=list(diagnostics),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--site-repo", type=Path, required=True)
    parser.add_argument("--event-name", default=None)
    parser.add_argument("--schedule", default=None)
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        required=False,
        help="probe (acquire only), dry_run (full disposable gates), or publish (commit/push)",
    )
    parser.add_argument(
        "--automation-enabled",
        default=None,
        help="Override ELECTION_AUTOMATION_ENABLED for this invocation",
    )
    parser.add_argument("--election-date", default=ELECTION_DAY.isoformat())
    parser.add_argument("--summary-path", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "publish":
        commit = True
        push = True
    else:
        commit = False
        push = False
    result = run_automation(
        args.repo_root,
        site_repo=args.site_repo,
        event_name=args.event_name,
        schedule=args.schedule,
        mode=args.mode,
        automation_enabled=args.automation_enabled,
        election_date=args.election_date,
        commit=commit,
        push=push,
        stage_callback=_log_stage,
    )
    rendered = result.summary.render()
    if args.summary_path:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False))
    return 0 if result.status in {
        "PUBLISHED",
        "SOURCE_CHECKED",
        "STOPPED_AFTER_ELECTION",
        "DISABLED_BY_REPOSITORY_KILL_SWITCH",
        "WEBSITE_RECOVERED",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AutomationError",
    "AutomationResult",
    "AutomationSummary",
    "DAILY_SCHEDULE_UTC",
    "ELECTION_DAY",
    "INTRADAY_SCHEDULE_UTC",
    "MODEL_RELEVANT_INPUTS",
    "PollingRefresh",
    "PRODUCTION_SAMPLES",
    "AfterElectionDay",
    "AUTOMATION_ENABLED_ENV",
    "automation_enabled_for_event",
    "classify_run_type",
    "current_stockholm_date",
    "guard_election_date",
    "latest_pop_observation_date",
    "model_relevant_snapshot_sha256",
    "refresh_polling_snapshot",
    "run_automation",
    "run_production_event",
    "run_website_checks",
    "resolve_mode",
    "should_publish",
]
