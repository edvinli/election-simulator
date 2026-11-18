"""T5 of the calibration-2030 programme: the prospective shadow run.

Implements ``docs/calibration_2030_t5_shadow_preregistration.md`` (FROZEN, body
SHA-256 bf93c128...; record ``data/processed/calibration_2030/t5_shadow.json``).
The arms are those of ``docs/calibration_2030_final_round_preregistration.md``;
this module only wires them to a weekly production snapshot.

    uv run python -m diagnostics.calibration_2030.t5_shadow capture [--dry-run --dry-run-out DIR]
    uv run python -m diagnostics.calibration_2030.t5_shadow verify [--capture <O>] [--replay]
    uv run python -m diagnostics.calibration_2030.t5_shadow report --quarter <YYYY>Q<q> [--dry-run]
    uv run python -m diagnostics.calibration_2030.t5_shadow activate-cand [--t4-results PATH] [--dry-run]

Isolation (§7). The runner reads production files only as Git objects of
``origin/main`` (``git show`` / ``git archive`` into a temporary directory),
never takes a production or benchmark lock, never calls publication,
certification, rendering or deployment code, and writes only under
``data/processed/shadow_2030/``. It refuses to run in 18:30–22:00 UTC, on a
dirty tracked tree, or (when writing) on any branch but ``shadow/2030``.

The T2 context is built from the capture's snapshot without editing
``t2_arms.py``: ``v03.load_inputs`` (the loader ``t2_context`` reaches through
``v03_context``) is replaced for the duration of the call by a loader that
reads the snapshot's files and keeps only observations available by ``O``;
the polls file, the target-poll arrays, the election days and the continued
k grid are then set on the returned ``ctx`` dictionary.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import multiprocessing
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .common import (
    ROOT,
    T5_BODY_SHA256,
    T5_DOC,
    T5_RECORD,
    IntegrityError,
    sha256,
)

# --------------------------------------------------------------------------- constants

SHADOW_BRANCH = "shadow/2030"
PRODUCTION_REMOTE = "origin"
PRODUCTION_BRANCH = "main"
PRODUCTION_REF = f"{PRODUCTION_REMOTE}/{PRODUCTION_BRANCH}"
SHADOW_REL = Path("data") / "processed" / "shadow_2030"
INDEX_NAME = "index.json"
ACTIVATIONS_REL = SHADOW_REL / "activations"
CAND_ACTIVATION_REL = ACTIVATIONS_REL / "cand.json"
FB_VERDICT_REL = ACTIVATIONS_REL / "fb_t4_verdict.json"
REPORTS_REL = SHADOW_REL / "reports"
TARGETS_REL = SHADOW_REL / "closed_targets.json"
DEFAULT_T4_RESULTS = "diagnostics/calibration_2030/t4_results.json"
PROD_ARCHIVE_REL = "data/processed/prospective_forecasts"
PROD_INDEX_REL = f"{PROD_ARCHIVE_REL}/index.json"
CALIBRATION_AMENDMENT_INDEX = ROOT / "data" / "processed" / "calibration_2030" / "amendments" / "index.json"
T5_DOC_REL = "docs/calibration_2030_t5_shadow_preregistration.md"

ELECTION_2030 = date(2030, 9, 8)
LAST_ORIGIN = date(2030, 9, 7)
POLL_HORIZONS = (30, 91, 182, 365, 730)
TARGET_NAMES = tuple(f"h{h}" for h in POLL_HORIZONS) + ("election",)
T2_ARMS = ("D0", "Ua", "Ub1", "Ub2", "Uc")
U_ARMS = ("Ua", "Ub1", "Ub2", "Uc")
ENGINE_ARMS = ("FB", "CAND")
N_DRAWS = 4000
LEVELS = np.round(np.arange(1, 100) / 100.0, 2)
PARTIES8 = ("M", "L", "C", "KD", "S", "V", "MP", "SD")
COLUMNS9 = (*PARTIES8, "REST")
NOISE_YEARS = (2002, 2006, 2010, 2014, 2018, 2022, 2026)
PROTECTED_START = dtime(18, 30)
PROTECTED_END = dtime(22, 0)
C0_SEED = 12345
C0_SAMPLES = 100_000
NPZ_DATE = (1980, 1, 1, 0, 0, 0)
GENESIS = "0" * 64
CLOSURE_LAG_DAYS = 14
WINDOW_HALF = 7
INTERVAL_LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

# Snapshot paths (relative to a production commit) read for every capture (§3.2).
SNAP_POLLS_CSV = "data/processed/pollofpolls/swedishpolls_individual_polls.csv"
SNAP_RAW = ("data/raw/pollofpolls/swedishpolls_polls.csv", "data/raw/pollofpolls/swedishpolls_sources.csv",
            "data/raw/pollofpolls/swedishpolls_retrieval_manifest.json")
SNAP_V02_TS = "data/processed/poll_aggregate/v0_2/swedishpolls_aggregate_timeseries.csv"
SNAP_V02_META = "data/processed/poll_aggregate/v0_2/swedishpolls_aggregate_metadata.json"
SNAP_V02_POLLS = "data/processed/poll_aggregate/v0_2/individual_polls.csv"
SNAP_ELECTIONS = "data/processed/elections/riksdag_election_results.csv"
SNAP_MANIFEST_2026 = "data/raw/elections/val2026/official_result_manifest.json"
SNAPSHOT_EXPORT = ("data/processed", "data/raw/pollofpolls", "data/raw/elections", "scripts", "pyproject.toml")
# Large production outputs the arms never read; left out of the temporary export.
SNAPSHOT_EXCLUDE = ("data/processed/prospective_forecasts", "data/processed/prospective_benchmark_2026",
                    "data/processed/backtests", "data/processed/pop_baseline_benchmark",
                    "data/processed/pop_state_diagnostics")
CODE_INPUTS = ("diagnostics/calibration_2030/t1_results.json",
               "data/processed/calibration_2030/amendments/001-t3-flag-interim-beta-half.json",
               "scripts/election_residuals/eve_consensus.py", "uv.lock")
INTERIM_LAW_RECORD = ROOT / "data" / "processed" / "calibration_2030" / "amendments" / "001-t3-flag-interim-beta-half.json"


class T5Refusal(RuntimeError):
    """The runner refuses to start, write or commit (§3.1, §4.1, §7)."""


# --------------------------------------------------------------------------- small pure helpers

def canonical_bytes(value: Any) -> bytes:
    """Canonical JSON (§3.5): sorted keys, no whitespace, UTF-8."""

    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def seed_string(origin: date, target: str) -> str:
    """§3.3: ``calibration-2030:T5:<O>:<target>``; target ∈ h30…h730, election, engine."""

    if target not in (*TARGET_NAMES, "engine"):
        raise ValueError(f"unknown T5 seed target {target!r}")
    return f"calibration-2030:T5:{origin.isoformat()}:{target}"


def seed_int(text: str) -> int:
    """The integer value of the first 8 hex digits of sha256(text)."""

    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def t5_seed(origin: date, target: str) -> int:
    return seed_int(seed_string(origin, target))


def iso_week_label(day: date) -> str:
    year, week, _ = day.isocalendar()
    return f"{year}-W{week:02d}"


def iso_week_start(moment: datetime) -> datetime:
    """Monday 00:00:00 UTC of the ISO week containing ``moment`` (§3.1)."""

    utc = moment.astimezone(timezone.utc)
    monday = utc.date() - timedelta(days=utc.weekday())
    return datetime.combine(monday, dtime(0, 0), tzinfo=timezone.utc)


def week_of_label(label: str) -> date:
    """The Monday of an ISO week label ``YYYY-Www``."""

    year, week = label.split("-W")
    return date.fromisocalendar(int(year), int(week), 1)


def weeks_between(previous_label: str, current_label: str) -> list[str]:
    """ISO weeks strictly between two week labels."""

    out, day = [], week_of_label(previous_label) + timedelta(days=7)
    end = week_of_label(current_label)
    while day < end:
        out.append(iso_week_label(day))
        day += timedelta(days=7)
    return out


def in_protected_interval(moment: datetime) -> bool:
    """18:30 ≤ t < 22:00 UTC (§3.1, CLAUDE.md)."""

    t = moment.astimezone(timezone.utc).time()
    return PROTECTED_START <= t < PROTECTED_END


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp without timezone: {value!r}")
    return parsed.astimezone(timezone.utc)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def fmt_utc(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_centipp(shares_pp: np.ndarray) -> np.ndarray:
    """§3.4: rint(share_pp × 100) clipped to [0, 10000], uint16."""

    return np.clip(np.rint(np.asarray(shares_pp, dtype=np.float64) * 100.0), 0, 10000).astype(np.uint16)


def from_centipp(centi: np.ndarray) -> np.ndarray:
    return np.asarray(centi, dtype=np.float64) / 100.0


def poll_quantiles(draws8: np.ndarray) -> np.ndarray:
    """(99, 8): numpy.quantile, method linear, at 0.01…0.99 (§3.4)."""

    return np.quantile(np.asarray(draws8, dtype=np.float64), LEVELS, axis=0, method="linear")


def quantile_crps(q: Sequence[float], y: float, levels: Sequence[float] = LEVELS) -> float:
    """§5.3: (2/99) Σ_τ QS_τ(q_τ, y), QS_τ(q, y) = (1{y < q} − τ)(q − y)."""

    q = np.asarray(q, dtype=np.float64)
    tau = np.asarray(levels, dtype=np.float64)
    if q.shape != tau.shape:
        raise ValueError("quantiles and levels differ in length")
    return float((2.0 / len(tau)) * np.sum(((y < q).astype(np.float64) - tau) * (q - y)))


def npz_bytes(members: Sequence[tuple[str, np.ndarray]]) -> bytes:
    """Deterministic NPZ exactly as ``exact_draw_sidecar._npz_bytes`` writes one (§3.4).

    ZIP_DEFLATED, fixed member timestamps, ``create_system`` 3, no pickle; the
    members are written in the order given.
    """

    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, array in members:
            buffer = io.BytesIO()
            np.save(buffer, np.asarray(array), allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=NPZ_DATE)
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 3
            archive.writestr(member, buffer.getvalue())
    return stream.getvalue()


def load_npz(data: bytes | Path) -> dict[str, np.ndarray]:
    source = io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data
    with np.load(source, allow_pickle=False) as loaded:
        return {k: np.asarray(loaded[k]) for k in loaded.files}


# --------------------------------------------------------------------------- git helpers

def git(repo: Path, *args: str, check: bool = True, binary: bool = False, env: dict | None = None):
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, check=False, text=not binary,
                            env={**os.environ, **(env or {})})
    if check and result.returncode != 0:
        err = result.stderr if not binary else result.stderr.decode("utf-8", "replace")
        raise RuntimeError(f"git {' '.join(args[:3])} failed: {err.strip()}")
    return result


def git_out(repo: Path, *args: str) -> str:
    return git(repo, *args).stdout.strip()


def git_show_bytes(repo: Path, ref: str, path: str) -> bytes | None:
    result = git(repo, "show", f"{ref}:{path}", check=False, binary=True)
    return result.stdout if result.returncode == 0 else None


def current_branch(repo: Path) -> str | None:
    result = git(repo, "symbolic-ref", "--quiet", "--short", "HEAD", check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def tracked_dirty(repo: Path) -> str:
    return git_out(repo, "status", "--porcelain", "--untracked-files=no")


def commit_time(repo: Path, commit: str) -> datetime:
    return parse_utc(git_out(repo, "show", "-s", "--format=%cI", commit))


def first_commit_with_blob(repo: Path, ref: str, path: str, expected_sha256: str) -> tuple[str, datetime] | None:
    """The first commit on ``ref`` whose ``path`` blob hashes to ``expected_sha256``."""

    result = git(repo, "log", ref, "--reverse", "--format=%H%x00%cI", "--", path, check=False)
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        commit, _, stamp = line.partition("\x00")
        blob = git_show_bytes(repo, commit, path)
        if blob is not None and sha256_bytes(blob) == expected_sha256:
            return commit, parse_utc(stamp)
    return None


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return git(repo, "merge-base", "--is-ancestor", ancestor, descendant, check=False).returncode == 0


def export_tree(repo: Path, commit: str, dest: Path, paths: Sequence[str] = SNAPSHOT_EXPORT,
                exclude: Sequence[str] = SNAPSHOT_EXCLUDE) -> Path:
    """``git archive`` of ``paths`` at ``commit`` into ``dest`` (Git objects only, no checkout)."""

    present = [p for p in paths if git(repo, "cat-file", "-e", f"{commit}:{p}", check=False).returncode == 0]
    spec = [*present, *(f":(exclude){p}" for p in exclude)]
    archive = git(repo, "archive", "--format=tar", commit, "--", *spec, binary=True).stdout
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")
    return dest


# --------------------------------------------------------------------------- guard (frozen T5 body)

def verify_t5(allow_dirty: bool = False) -> dict:
    """The final-round guard plus the frozen T5 preregistration, its record and companion hash."""

    from .common import _frozen_body_ok, verify_final_round

    provenance = verify_final_round(allow_dirty=allow_dirty)
    if not _frozen_body_ok(T5_DOC, T5_BODY_SHA256):
        raise IntegrityError("T5 preregistration body does not match its frozen hash")
    record = json.loads(T5_RECORD.read_text(encoding="utf-8"))
    if record.get("status") != "frozen" or record.get("frozen") is not True or record.get("body_sha256") != T5_BODY_SHA256:
        raise IntegrityError("T5 record is not the frozen record")
    companion = T5_RECORD.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    record_sha = sha256(T5_RECORD)
    if companion != record_sha:
        raise IntegrityError("T5 record does not match its .sha256 companion")
    provenance.update({"t5_body_sha256": T5_BODY_SHA256, "t5_record_sha256": record_sha,
                       "t5_amendments_in_force": t5_amendments_in_force()})
    return provenance


def t5_amendments_in_force() -> list[dict]:
    """Calibration amendments whose ``amends`` is the T5 document (none at freeze)."""

    if not CALIBRATION_AMENDMENT_INDEX.exists():
        return []
    index = json.loads(CALIBRATION_AMENDMENT_INDEX.read_text(encoding="utf-8"))
    return [{"amendment_id": a["amendment_id"], "record_sha256": a.get("record_sha256")}
            for a in index.get("amendments", []) if a.get("amends") == T5_DOC_REL]


# --------------------------------------------------------------------------- refusals

def check_clock(now: datetime) -> None:
    if in_protected_interval(now):
        raise T5Refusal(f"refusing: {fmt_utc(now)} is inside the protected interval 18:30–22:00 UTC")


def preflight(repo: Path, now: datetime, *, writing: bool) -> list[str]:
    """§7 refusals. Returns warnings (for a dry run) instead of refusing on branch/dirty when not writing."""

    check_clock(now)
    problems = []
    branch = current_branch(repo)
    if branch != SHADOW_BRANCH:
        problems.append(f"checked-out branch is {branch!r}, not {SHADOW_BRANCH!r}")
    dirty = tracked_dirty(repo)
    if dirty:
        problems.append(f"tracked files have uncommitted changes:\n{dirty}")
    if writing and problems:
        raise T5Refusal("refusing: " + "; ".join(problems))
    return problems


# --------------------------------------------------------------------------- index (§4.1, §4.2)

def empty_index() -> dict:
    return {"schema": "calibration-2030-t5-shadow-index-v1", "captures": [], "missed_weeks": []}


def entry_hash(entry: Mapping) -> str:
    return sha256_bytes(canonical_bytes(entry))


def make_entry(index: Mapping, *, capture_date: date, manifest_sha256: str, arms: Sequence[str],
               code_commit: str) -> dict:
    captures = index.get("captures", [])
    return {
        "capture_date": capture_date.isoformat(),
        "iso_week": iso_week_label(capture_date),
        "path": (SHADOW_REL / capture_date.isoformat()).as_posix(),
        "manifest_sha256": manifest_sha256,
        "arms": sorted(arms),
        "code_commit": code_commit,
        "previous_entry_sha256": entry_hash(captures[-1]) if captures else GENESIS,
    }


def append_capture(index: Mapping, entry: Mapping, current_week: str) -> dict:
    """A new index: the old one plus the skipped weeks (§4.2) and the new entry."""

    new = json.loads(canonical_bytes(index))
    captures = new["captures"]
    if any(c["iso_week"] == current_week for c in captures):
        raise T5Refusal(f"refusing: {current_week} already has a committed capture")
    if captures:
        recorded = {m["iso_week"] for m in new["missed_weeks"]}
        for week in weeks_between(captures[-1]["iso_week"], current_week):
            if week not in recorded:
                new["missed_weeks"].append({"iso_week": week, "recorded_by": entry["capture_date"]})
    expected_prev = entry_hash(captures[-1]) if captures else GENESIS
    if entry["previous_entry_sha256"] != expected_prev:
        raise T5Refusal("refusing: the new entry does not chain to the last committed entry")
    captures.append(dict(entry))
    return new


def check_chain(index: Mapping) -> list[str]:
    errors, prev = [], GENESIS
    for i, entry in enumerate(index.get("captures", [])):
        if entry.get("previous_entry_sha256") != prev:
            errors.append(f"index entry {i} ({entry.get('capture_date')}) breaks the chain")
        prev = entry_hash(entry)
    return errors


def is_prefix(committed: Mapping, new: Mapping) -> bool:
    """§4.1: the committed index is a prefix of the index about to be written."""

    old_c, new_c = committed.get("captures", []), new.get("captures", [])
    old_m, new_m = committed.get("missed_weeks", []), new.get("missed_weeks", [])
    if len(new_c) < len(old_c) or len(new_m) < len(old_m):
        return False
    same_head = {k: v for k, v in committed.items() if k not in ("captures", "missed_weeks")} == \
        {k: v for k, v in new.items() if k not in ("captures", "missed_weeks")}
    return same_head and canonical_bytes(new_c[:len(old_c)]) == canonical_bytes(old_c) and \
        canonical_bytes(new_m[:len(old_m)]) == canonical_bytes(old_m)


def committed_json(repo: Path, rel: Path | str, ref: str = "HEAD") -> dict | None:
    blob = git_show_bytes(repo, ref, Path(rel).as_posix())
    return None if blob is None else json.loads(blob.decode("utf-8"))


def committed_index(repo: Path, ref: str = "HEAD") -> dict:
    return committed_json(repo, SHADOW_REL / INDEX_NAME, ref) or empty_index()


# --------------------------------------------------------------------------- anchor (§3.1)

def production_snapshots(repo: Path, ref: str = PRODUCTION_REF) -> list[dict]:
    blob = git_show_bytes(repo, ref, PROD_INDEX_REL)
    if blob is None:
        raise T5Refusal(f"refusing: {ref}:{PROD_INDEX_REL} does not exist")
    return list(json.loads(blob.decode("utf-8")).get("snapshots", []))


def select_anchor(snapshots: Iterable[Mapping], week_start: datetime, run_started: datetime,
                  first_commit: Callable[[Mapping], tuple[str, datetime] | None]) -> tuple[dict, tuple[str, datetime]] | None:
    """§3.1: the first certified 2030 generation with generated_at_utc ≥ Monday 00:00 UTC (and in the
    week), whose snapshot was first committed on origin/main before the run started."""

    week_end = week_start + timedelta(days=7)
    candidates = []
    for s in snapshots:
        if s.get("election_date") != ELECTION_2030.isoformat() or not s.get("generation_id"):
            continue
        generated = parse_utc(s["generated_at_utc"])
        if week_start <= generated < week_end and generated < run_started:
            candidates.append((generated, s["generation_id"], s))
    for _, _, s in sorted(candidates, key=lambda t: (t[0], t[1])):
        first = first_commit(s)
        if first is not None and first[1] < run_started:
            return dict(s), first
    return None


def archive_first_commit(repo: Path, snapshot: Mapping, ref: str = PRODUCTION_REF) -> tuple[str, datetime] | None:
    return first_commit_with_blob(repo, ref, f"{PROD_ARCHIVE_REL}/{snapshot['path']}", snapshot["snapshot_file_sha256"])


# --------------------------------------------------------------------------- C0 (§2.4 b)

def verify_c0_sidecar(repo: Path, anchor: Mapping, workdir: Path, ref: str = PRODUCTION_REF,
                      include_arrays: bool = False) -> dict:
    """Extract the anchor's draws.npz/draws.json from ``ref`` and run ``load_verified_draw_sidecar``."""

    from scripts.simulator.exact_draw_sidecar import load_verified_draw_sidecar

    gen = anchor["generation_id"]
    out = workdir / "c0" / gen
    out.mkdir(parents=True, exist_ok=True)
    for name in ("draws.npz", "draws.json"):
        blob = git_show_bytes(repo, ref, f"{PROD_ARCHIVE_REL}/{gen}/{name}")
        if blob is None:
            raise FileNotFoundError(f"{gen}/{name} is not in {ref}")
        (out / name).write_bytes(blob)
    loaded = load_verified_draw_sidecar(out / "draws.npz", out / "draws.json", expected_generation_id=gen,
                                        expected_payload_hash=anchor["deterministic_payload_sha256"],
                                        include_arrays=include_arrays)
    loaded["draws_path_local"] = str(out / "draws.npz")
    return loaded


C0_RECOMPUTE_DRIVER = r"""
import json, sys
import numpy as np
spec = json.loads(sys.argv[1])
from scripts.simulator.engine import simulate_election
result = simulate_election(as_of=spec["as_of"], election_date=spec["election_date"], samples=spec["samples"],
                           seed=spec["seed"], data_dir=spec["data_dir"])
np.save(spec["out"], np.asarray(result.vote_shares_matrix, dtype=np.float64), allow_pickle=False)
"""


def recompute_c0(export_root: Path, origin: date, workdir: Path) -> np.ndarray:
    """§2.4: production at the anchor's commit and snapshot, seed 12345, 100,000 draws (subprocess)."""

    out = workdir / "c0_recomputed.npy"
    spec = {"as_of": origin.isoformat(), "election_date": ELECTION_2030.isoformat(), "samples": C0_SAMPLES,
            "seed": C0_SEED, "data_dir": str(export_root / "data" / "processed"), "out": str(out)}
    script = workdir / "c0_driver.py"
    script.write_text(C0_RECOMPUTE_DRIVER, encoding="utf-8")
    subprocess.run([sys.executable, str(script), json.dumps(spec)], cwd=export_root, check=True,
                   env={**os.environ, "PYTHONPATH": str(export_root)})
    return np.load(out, allow_pickle=False)


# --------------------------------------------------------------------------- snapshot context (§2.1, §3.2)

def production_model_files_identical(repo: Path, anchor_commit: str, code_commit: str) -> tuple[bool, list[str]]:
    """The hashed production model files (``effective_inputs.MODEL_FILES``) at both commits."""

    from scripts.forecast_history.effective_inputs import MODEL_FILES

    # PREREG-AMBIGUITY: §2.4 says "the hashed production model files" without
    # a list. Reading implemented: scripts.forecast_history.effective_inputs.
    # MODEL_FILES, production's own list of numerical model files whose hashes
    # gate draw reuse.
    differing = []
    for path in MODEL_FILES:
        a = git(repo, "rev-parse", f"{anchor_commit}:{path}", check=False).stdout.strip()
        b = git(repo, "rev-parse", f"{code_commit}:{path}", check=False).stdout.strip()
        if a != b:
            differing.append(path)
    return not differing, differing


@contextlib.contextmanager
def _patched(module, name: str, value):
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


def t5_grid(origin: date) -> list[date]:
    """T2's 14-day grid from 2010-10-03, continued past 2026-11-18 up to the origin (§2.1)."""

    from . import t2_arms as A

    out, day = [], A.GRID_FIRST
    while day <= origin:
        out.append(day)
        day += timedelta(days=A.GRID_STEP)
    return out


def build_snapshot_context(export_root: Path, origin: date, workdir: Path, n_draws: int = N_DRAWS) -> dict:
    """``t2_arms.t2_context`` on the capture's snapshot, causal at ``origin``."""

    from scripts.poll_aggregate.data import load_elections, load_polls

    from . import t2_arms as A
    from . import v03

    counts: dict[str, int] = {}

    def load_inputs_from_snapshot():
        elections_all = load_elections(export_root / SNAP_ELECTIONS, [export_root / SNAP_MANIFEST_2026])
        polls_all = load_polls(export_root / SNAP_POLLS_CSV, [e.obs_date for e in elections_all]).polls
        polls = [p for p in polls_all if p.available <= origin]
        elections = [e for e in elections_all if e.available <= origin]
        counts.update(polls_in_snapshot=len(polls_all), polls_after_origin_excluded=len(polls_all) - len(polls),
                      elections_after_origin_excluded=len(elections_all) - len(elections))
        everything = sorted([*polls, *elections], key=lambda o: o.sort_key())
        houses = sorted({o.house for o in polls})
        return elections, polls, everything, houses

    with _patched(v03, "load_inputs", load_inputs_from_snapshot):
        ctx = A.t2_context(n_draws=n_draws, v03_dir=workdir / "v03")
    ctx["polls_file"] = export_root / SNAP_V02_POLLS
    # §3.2: the target polls known at O (t2_context cut them at the T2 cutoff 2026-11-18).
    tp = [o for o in ctx["polls"] if o.available <= origin]
    ctx["tp_obs"] = np.array([o.obs_date.toordinal() for o in tp], dtype=np.int64)
    ctx["tp_avail"] = np.array([o.available.toordinal() for o in tp], dtype=np.int64)
    ctx["tp_n"] = np.array([float(o.sample_size) for o in tp])
    ctx["tp_shares"] = np.array([o.shares for o in tp], dtype=float)
    ctx["target_polls_excluded_after_cutoff"] = len(ctx["polls"]) - len(tp)
    ctx["cache"]["target"].clear()
    ctx["cache"]["window"].clear()
    # Scheduled election days are calendar facts: 2030-09-08 enters the open-interval shocks
    # of poll targets dated after it, never the election-day target itself (§2.1).
    ctx["election_days"] = sorted(set(ctx["election_days"]) | {ELECTION_2030})
    ctx["t5_counts"] = counts
    ctx["t5_origin"] = origin
    return ctx


_CTX: dict | None = None


def _phase_a(origin: date) -> tuple:
    from . import t2_arms as A

    ctx = _CTX
    try:
        state = A.d0_opinion_state(ctx, origin)
    except ValueError as exc:
        state = exc
    return origin, state, A.omega_at(ctx, origin), A.theta_details(ctx, origin)


def _phase_b(cell: tuple[date, int]) -> tuple:
    from . import t2_arms as A

    o, h = cell
    return (o, h, _CTX["n_draws"]), A.k1_cell(_CTX, o, h)


def _map(workers: int, fn, items: list) -> list:
    if workers <= 1 or len(items) < 2:
        return [fn(x) for x in items]
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(processes=workers) as pool:
        return pool.map(fn, items, chunksize=max(1, len(items) // (workers * 4)))


def prepare_k(ctx: dict, origin: date, workers: int = 1) -> dict:
    """Per-origin intermediates over the continued grid, then k per U arm at ``origin`` (§2.1).

    Workers only fill ``ctx``'s caches with the same deterministic recomputations
    ``t2_arms`` would make lazily (byte-identical caching, §2.1).
    """

    global _CTX
    from . import t2_arms as A

    for h in (7, 14, 28, 112):
        A._d0_transitions(ctx, h)
    for h in A.HORIZONS:
        A._uc_table(ctx, h)
    A._variogram_tables(ctx)
    A._centre_latest_publication(ctx)
    grid = [o for o in t5_grid(origin) if o < origin]
    _CTX = ctx
    for o, state, omega, theta in _map(workers, _phase_a, grid):
        ctx["cache"]["d0_state"][o] = state
        ctx["cache"]["omega"][o] = omega
        ctx["cache"]["theta"][o] = theta
    first = A.first_scored_origin(ctx)
    ctx["scored_origins"] = [o for o in grid if o >= first and A.origin_defined(ctx, o)["all"]]
    cells = A.k_eligible_cells(ctx, origin)
    _CTX = ctx
    for key, value in _map(workers, _phase_b, cells):
        ctx["cache"]["k1"][key] = value
    _CTX = None
    return {arm: A.inflation_k(arm, ctx, origin) for arm in U_ARMS}


def interim_law():
    """Amendment 001's interim law refitted on the seven corrected residuals; checked against the record."""

    from . import noise_laws as N

    law = N.fit_law("N0", N.corrected_pool(list(NOISE_YEARS)))
    record = json.loads(INTERIM_LAW_RECORD.read_text(encoding="utf-8"))["interim_2030_law"]
    s = law.summary()
    if not (np.allclose(s["mean_pp"], record["mean_shift_pp"], rtol=0, atol=1e-12)
            and np.allclose(s["sd_pp"], record["sd_pp"], rtol=0, atol=1e-12)):
        raise IntegrityError("the refitted interim law does not reproduce amendment 001's record")
    return law


def apply_noise(base9: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """The engine's transfer: apply_batch_simplex_transfer(eps = 0.01 pp), then renormalize to 100."""

    from scripts.election_layer_v2.transfer import apply_batch_simplex_transfer
    from scripts.vote_share_calibration.config import MIN_SHARE_PCT

    nat, _ = apply_batch_simplex_transfer(np.asarray(base9, dtype=np.float64), np.asarray(residuals, dtype=np.float64),
                                          eps=MIN_SHARE_PCT)
    return 100.0 * nat / np.sum(nat, axis=1, keepdims=True)


C0P_DRIVER = r"""
import json, sys
from datetime import date
import numpy as np
spec = json.loads(sys.argv[1])
from diagnostics.calibration_2030 import t2_arms as A
ctx = {"v03_file": spec["timeseries"], "polls_file": spec["polls"], "cache": {"d0_state": {}}}
origin = date.fromisoformat(spec["origin"])
out = {}
for name, (target, seed) in spec["targets"].items():
    draws = A._d0_draws(ctx, origin, date.fromisoformat(target), spec["n"], seed)
    np.save(f"{spec['out']}/{name}.npy", draws, allow_pickle=False)
    st = A.d0_opinion_state(ctx, origin)
    out[name] = A.d0_eligible(ctx, st.as_of, max(1, (date.fromisoformat(target) - st.as_of).days))[1]
print(json.dumps(out))
"""


def c0p_ctx(export_root: Path) -> dict:
    """D0's production path on production's own v0.2 files at the anchor (§2.4 a)."""

    return {"v03_file": export_root / SNAP_V02_TS, "polls_file": export_root / SNAP_V02_POLLS,
            "cache": {"d0_state": {}}}


def c0p_base_draws(export_root: Path, origin: date, targets: Mapping[str, tuple[date, int]], n: int,
                   in_process: bool, workdir: Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """C0p state + dynamics per poll target (no Ω); production code in-process or at the anchor commit.

    # PREREG-AMBIGUITY: §2.4 lists C0p's state, dynamics and common term but no
    # simplex map, and also says C0p "differs from D0 only in the aggregate".
    # Reading implemented: C0p is D0's code path (t2_arms._d0_draws, which ends
    # with the §3.4 map, a no-op unless a share is below 0.01 pp) on the v0.2
    # files, so the two differ only in the aggregate.
    """

    from . import t2_arms as A

    if in_process:
        ctx = c0p_ctx(export_root)
        draws, fallback = {}, {}
        for name, (target, seed) in targets.items():
            draws[name] = A._d0_draws(ctx, origin, target, n, seed)
            st = A.d0_opinion_state(ctx, origin)
            fallback[name] = A.d0_eligible(ctx, st.as_of, max(1, (target - st.as_of).days))[1]
        return draws, fallback
    # Production files differ from the T5 code commit: run D0's path with the anchor's
    # scripts/ first on sys.path (a git-archive export at the anchor commit).
    out = workdir / "c0p"
    out.mkdir(parents=True, exist_ok=True)
    script = workdir / "c0p_driver.py"
    script.write_text(C0P_DRIVER, encoding="utf-8")
    spec = {"timeseries": str(export_root / SNAP_V02_TS), "polls": str(export_root / SNAP_V02_POLLS),
            "origin": origin.isoformat(), "n": n, "out": str(out),
            "targets": {k: [t.isoformat(), s] for k, (t, s) in targets.items()}}
    result = subprocess.run([sys.executable, str(script), json.dumps(spec)], cwd=workdir, check=True,
                            capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": f"{export_root}{os.pathsep}{ROOT}"})
    fallback = json.loads(result.stdout.strip().splitlines()[-1])
    return {k: np.load(out / f"{k}.npy", allow_pickle=False) for k in targets}, fallback


# --------------------------------------------------------------------------- arms

def target_dates(origin: date) -> dict[str, date]:
    out = {f"h{h}": origin + timedelta(days=h) for h in POLL_HORIZONS}
    out["election"] = ELECTION_2030
    return out


def run_t2_targets(ctx: dict, origin: date, k: Mapping[str, tuple[float, int]], law, n: int,
                   clock: Callable[[], datetime]) -> tuple[dict, dict, dict]:
    """Every T2 arm at every target with §3.3's shared generator. Returns (quantiles, election draws, info)."""

    from . import t2_arms as A

    quantiles: dict[str, dict[str, np.ndarray]] = {arm: {} for arm in T2_ARMS}
    election: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}
    info: dict[str, dict] = {}
    seconds = dict.fromkeys(T2_ARMS, 0.0)
    for name, target in target_dates(origin).items():
        check_clock(clock())
        seed = t5_seed(origin, name)
        rng = np.random.default_rng(seed)
        # §3.3: T2's draw order — Z_state, Z_dyn, Z_poll, then Uc's bootstrap — then, for
        # the election target, Z_noise (N × 9) from the same generator.
        z_state = rng.standard_normal((n, len(PARTIES8)))
        z_dyn = rng.standard_normal((n, len(PARTIES8)))
        z_poll = rng.standard_normal((n, len(PARTIES8)))
        cell_info: dict[str, Any] = {"target_date": target.isoformat(), "seed_string": seed_string(origin, name),
                                     "seed": seed}
        base: dict[str, np.ndarray] = {}
        for arm in T2_ARMS:
            if arm in errors:
                continue
            t0 = time.time()
            try:
                if arm == "D0":
                    base[arm] = A.state_dynamics_draws(arm, ctx, origin, target, z_state=z_state, z_dyn=z_dyn,
                                                       boot_rng=None, d0_base_seed=seed)
                    as_of = A.d0_opinion_state(ctx, origin).as_of
                    cell_info["D0_eval_horizon"] = A.d0_eligible(ctx, as_of, max(1, (target - as_of).days))[1]
                else:
                    base[arm] = A.state_dynamics_draws(arm, ctx, origin, target, z_state=z_state, z_dyn=z_dyn,
                                                       boot_rng=rng if arm == "Uc" else None, d0_base_seed=None,
                                                       k=k[arm][0])
                    if arm == "Uc":
                        deltas, hp, scale = A.uc_pool(ctx, origin, (target - origin).days)
                        cell_info["Uc_eligible_transitions"] = int(len(deltas))
                        cell_info["Uc_horizon_used"] = int(hp)
                        cell_info["Uc_scale"] = float(scale)
            except Exception as exc:  # noqa: BLE001 - recorded as FAILED (§4.3)
                errors[arm] = f"{name}: {type(exc).__name__}: {exc}"
            seconds[arm] += time.time() - t0
        # PREREG-AMBIGUITY: if Uc fails, its bootstrap draws are not consumed and
        # Z_noise follows Z_poll directly; the spec orders the draws but does not
        # say what a failed arm consumes. Reading implemented: a failed arm draws nothing.
        if name == "election":
            z_noise = rng.standard_normal((n, len(COLUMNS9)))
            for arm, b in base.items():
                t0 = time.time()
                try:
                    election[arm] = apply_noise(b, law.draw(z_noise, b))
                except Exception as exc:  # noqa: BLE001
                    errors[arm] = f"election noise: {type(exc).__name__}: {exc}"
                seconds[arm] += time.time() - t0
        else:
            try:
                poll = A.poll_term(ctx, origin, z_poll)
            except Exception as exc:  # noqa: BLE001 - Ω undefined: every arm fails this target
                for arm in base:
                    errors.setdefault(arm, f"{name}: {type(exc).__name__}: {exc}")
                poll = None
            if poll is not None:
                for arm, b in base.items():
                    quantiles[arm][name] = poll_quantiles(b[:, :len(PARTIES8)] + poll)
            cell_info["_z_poll"] = z_poll
        info[name] = cell_info
    return quantiles, election, {"errors": errors, "targets": info, "seconds": seconds}


def run_c0p(export_root: Path, ctx: dict, origin: date, n: int, in_process: bool, workdir: Path,
            z_polls: Mapping[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    from . import t2_arms as A

    targets = {name: (d, t5_seed(origin, name)) for name, d in target_dates(origin).items() if name != "election"}
    base, fallback = c0p_base_draws(export_root, origin, targets, n, in_process, workdir)
    out = np.stack([poll_quantiles(base[name][:, :len(PARTIES8)] + A.poll_term(ctx, origin, z_polls[name]))
                    for name in targets]).astype(np.float32)
    return out, {"eval_horizon": fallback}


def run_engine_arm(config, export_root: Path, v03_file: Path, origin: date, n: int, seed: int, workdir: Path,
                   ctx: dict) -> np.ndarray:
    """FB / CAND through the T4 candidate path on the snapshot's data root (§2.2)."""

    from . import candidate_engine as ce

    parent = workdir / f"engine_{config.name}"
    parent.mkdir(parents=True, exist_ok=True)
    with _patched(ce, "PROCESSED_ROOT", export_root / "data" / "processed"):
        root = ce.build_opinion_root(parent, "root_v03", v03_file)
    # A U-arm CAND reads ce.t2_context(); hand it the snapshot context.
    with _patched(ce, "_T2_CTX", ctx):
        result, _ = ce.simulate_candidate(config, as_of=origin, election_date=ELECTION_2030, samples=n, seed=seed,
                                          data_dir=root)
    return np.asarray(result.vote_shares_matrix, dtype=np.float64)


# --------------------------------------------------------------------------- activation (§2.3)

def load_activation(repo: Path, rel: Path = CAND_ACTIVATION_REL, ref: str = "HEAD") -> dict | None:
    record = committed_json(repo, rel, ref)
    if record is None:
        return None
    blob = git_show_bytes(repo, record["t4_results_commit"], record["t4_results_path"])
    if blob is None or sha256_bytes(blob) != record["t4_results_sha256"]:
        raise IntegrityError(f"{rel}: the T4 results record does not match its recorded hash and commit")
    return record


def t4_completion(repo: Path, t4_path: str, ref: str = "HEAD") -> tuple[str, datetime] | None:
    """The first commit on ``ref`` that contains the T4 results record (§2.3)."""

    result = git(repo, "log", ref, "--reverse", "--format=%H%x00%cI", "--", t4_path, check=False)
    for line in result.stdout.splitlines():
        commit, _, stamp = line.partition("\x00")
        if git_show_bytes(repo, commit, t4_path) is not None:
            return commit, parse_utc(stamp)
    return None


def build_activation(repo: Path, t4_path: str = DEFAULT_T4_RESULTS) -> tuple[dict, dict | None]:
    """cand.json (and fb_t4_verdict.json if FB was run through T4) from the committed T4 record."""

    first = t4_completion(repo, t4_path)
    if first is None:
        raise T5Refusal(f"refusing: {t4_path} is not committed on HEAD; T4 has not completed")
    commit, committed_at = first
    blob = git_show_bytes(repo, commit, t4_path)
    record = json.loads(blob.decode("utf-8"))
    # PREREG-AMBIGUITY: §2.3 names the label (T4_PASS/T4_FAIL and the 2026 flag)
    # but not where t4_results.json stores them. Reading implemented: t4_replay's
    # layout, gates.CAND.pass and gates.CAND.flag_2026.flagged, with the selection
    # from cand_selection.
    gates = record.get("gates", {})
    cand = gates.get("CAND")
    if record.get("smoke") or cand is None or record.get("cand_selection") is None:
        raise T5Refusal("refusing: the T4 record has no CAND verdict (smoke run or CAND not configured)")
    common = {"t4_results_path": t4_path, "t4_results_sha256": sha256_bytes(blob), "t4_results_commit": commit,
              "t4_completed_utc": fmt_utc(committed_at), "completion_day_utc": committed_at.date().isoformat(),
              "code_commit": git_out(repo, "rev-parse", "HEAD")}
    cand_record = {**common, "arm": "CAND", "selection": record["cand_selection"],
                   "cand_equals_fb": bool(record.get("cand_equals_fb")),
                   "label": {"verdict": "T4_PASS" if cand.get("pass") else "T4_FAIL",
                             "flag_2026_raised": bool(cand.get("flag_2026", {}).get("flagged"))}}
    fb = gates.get("FB")
    fb_record = None
    if fb is not None and str(record.get("fb_status", "")).startswith("run because"):
        fb_record = {**common, "arm": "FB", "label": {"verdict": "T4_PASS" if fb.get("pass") else "T4_FAIL",
                                                      "flag_2026_raised": bool(fb.get("flag_2026", {}).get("flagged"))}}
    return cand_record, fb_record


# --------------------------------------------------------------------------- writing and committing

def write_and_commit(repo: Path, files: Mapping[Path, bytes], new_index: dict | None, message: str,
                     clock: Callable[[], datetime], expected_head: str | None = None,
                     week_end: datetime | None = None) -> str:
    """Write ``files`` (and the index) under shadow_2030 and commit them alone on shadow/2030."""

    now = clock()
    preflight(repo, now, writing=True)
    if week_end is not None and now >= week_end:
        raise T5Refusal("refusing: the capture's ISO week has ended; the week is missed (§3.1)")
    head = git_out(repo, "rev-parse", "HEAD")
    if expected_head is not None and head != expected_head:
        raise T5Refusal("refusing: HEAD moved during the run")
    rels = list(files)
    if new_index is not None:
        committed = committed_index(repo)
        if not is_prefix(committed, new_index):
            raise T5Refusal("refusing: the committed index is not a prefix of the index about to be written")
        rels.append(SHADOW_REL / INDEX_NAME)
        files = {**files, SHADOW_REL / INDEX_NAME: canonical_bytes(new_index)}
    for rel in rels:
        if not Path(rel).as_posix().startswith(SHADOW_REL.as_posix() + "/"):
            raise T5Refusal(f"refusing: {rel} is outside {SHADOW_REL}")
        if rel != SHADOW_REL / INDEX_NAME and git(repo, "cat-file", "-e", f"HEAD:{Path(rel).as_posix()}",
                                                   check=False).returncode == 0:
            raise T5Refusal(f"refusing: {rel} is already committed (append-only)")
    for rel, data in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    paths = [Path(r).as_posix() for r in rels]
    git(repo, "add", "--", *paths)
    staged = git_out(repo, "diff", "--cached", "--name-only").splitlines()
    if sorted(staged) != sorted(paths):
        raise T5Refusal(f"refusing: staged paths {staged} differ from the capture's paths")
    git(repo, "commit", "--quiet", "-m", message, "--", *paths)
    return git_out(repo, "rev-parse", "HEAD")


# --------------------------------------------------------------------------- capture

def input_hashes(export_root: Path, ctx: dict) -> dict:
    out = {p: sha256(export_root / p) for p in (SNAP_POLLS_CSV, *SNAP_RAW, SNAP_V02_TS, SNAP_V02_META, SNAP_V02_POLLS,
                                                SNAP_ELECTIONS, SNAP_MANIFEST_2026)}
    snapshot = {f"{p}@anchor": h for p, h in out.items()}
    code = {f"{p}@code": sha256(ROOT / p) for p in CODE_INPUTS}
    return {**snapshot, **code, "v0.3_timeseries_generated": ctx["v03_file_sha256"],
            "swedishpolls_snapshot_sha256": out[SNAP_POLLS_CSV]}


def runtime() -> dict:
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "numpy": np.__version__, "uv_lock_sha256": sha256(ROOT / "uv.lock")}


def capture(repo: Path = ROOT, *, dry_run: bool = False, dry_run_out: Path | None = None, fetch: bool = True,
            workers: int = 1, clock: Callable[[], datetime] = utc_now, n_draws: int = N_DRAWS,
            prod_ref: str = PRODUCTION_REF) -> dict:
    """One weekly capture (§3). Returns a summary; writes and commits only when not a dry run."""

    run_started = clock()
    timings: dict[str, float] = {}
    t_all = time.time()
    warnings = preflight(repo, run_started, writing=not dry_run)
    provenance = verify_t5(allow_dirty=dry_run)
    code_commit = git_out(repo, "rev-parse", "HEAD")
    if fetch:
        git(repo, "fetch", "--quiet", PRODUCTION_REMOTE,
            f"+refs/heads/{PRODUCTION_BRANCH}:refs/remotes/{PRODUCTION_REMOTE}/{PRODUCTION_BRANCH}")
    prod_tip = git_out(repo, "rev-parse", prod_ref)

    week_start = iso_week_start(run_started)
    week = iso_week_label(week_start.date())
    picked = select_anchor(production_snapshots(repo, prod_ref), week_start, run_started,
                           lambda s: archive_first_commit(repo, s, prod_ref))
    if picked is None:
        raise T5Refusal(f"refusing: no certified 2030 production generation since {fmt_utc(week_start)} "
                        f"is committed on {prod_ref} yet (week {week} has no anchor)")
    anchor, (archive_commit, archive_time) = picked
    origin = date.fromisoformat(anchor["as_of"])
    if origin > LAST_ORIGIN:
        raise T5Refusal(f"refusing: origin {origin} is after the last capture origin {LAST_ORIGIN}")
    index = committed_index(repo)
    if any(c["iso_week"] == iso_week_label(origin) for c in index["captures"]):
        raise T5Refusal(f"refusing: {iso_week_label(origin)} already has a committed capture")
    if iso_week_label(origin) != week:
        # PREREG-AMBIGUITY: the directory and the index use O's ISO week; an anchor
        # whose as_of falls in another week than its generated_at_utc is refused.
        raise T5Refusal(f"refusing: anchor as_of {origin} is not in the run's ISO week {week}")
    activation = load_activation(repo)
    fb_verdict = committed_json(repo, FB_VERDICT_REL)

    with tempfile.TemporaryDirectory(prefix="calibration2030-t5-") as tmp:
        work = Path(tmp)
        t0 = time.time()
        snap_commit = anchor["source_git_commit"]
        export_root = export_tree(repo, snap_commit, work / "snapshot")
        snapshot_blob = git_show_bytes(repo, prod_ref, f"{PROD_ARCHIVE_REL}/{anchor['path']}")
        snapshot_json = json.loads(snapshot_blob.decode("utf-8"))
        poll_hash = sha256(export_root / SNAP_POLLS_CSV)
        if poll_hash != snapshot_json["hashes"]["poll_data_sha256"]:
            raise T5Refusal("refusing: the polls CSV at the anchor commit does not match poll_data_sha256")
        timings["snapshot_export"] = time.time() - t0

        # C0 (§2.4 b)
        t0 = time.time()
        c0: dict[str, Any] = {"generation_id": anchor["generation_id"], "seed": C0_SEED}
        c0_draws_file = None
        try:
            side = verify_c0_sidecar(repo, anchor, work, prod_ref)
            if archive_time >= run_started:
                raise ValueError("first archive commit is not before the run start")
            c0.update(status="VERIFIED", draws_file_sha256=side["metadata"]["draws_file_sha256"],
                      rows=int(side["metadata"]["samples"]))
        except Exception as exc:  # noqa: BLE001 - recomputed and labelled (§2.4)
            c0.update(status="C0_RECOMPUTED", verification_error=f"{type(exc).__name__}: {exc}")
            if dry_run and os.environ.get("T5_DRY_RUN_SKIP_C0_RECOMPUTE"):
                c0_draws = None
            else:
                c0_draws = recompute_c0(export_root, origin, work)
            if c0_draws is not None:
                c0_draws_file = npz_bytes([("election_draws_centipp", to_centipp(c0_draws))])
                c0.update(rows=int(c0_draws.shape[0]))
        timings["C0"] = time.time() - t0

        # Context and per-capture fits (§2.1)
        t0 = time.time()
        check_clock(clock())
        ctx = build_snapshot_context(export_root, origin, work, n_draws)
        from . import t2_arms as A
        from .v03 import self_check

        timings["context"] = time.time() - t0
        t0 = time.time()
        k = prepare_k(ctx, origin, workers)
        timings["k_theta_omega"] = time.time() - t0
        t0 = time.time()
        law = interim_law()
        timings["interim_law"] = time.time() - t0

        t0 = time.time()
        quantiles, election, t2_info = run_t2_targets(ctx, origin, k, law, n_draws, clock)
        timings["T2_arms"] = time.time() - t0
        timings.update({f"arm_{a}": s for a, s in t2_info["seconds"].items()})

        arm_status: dict[str, dict] = {}
        files: dict[str, bytes] = {}
        target_names = [f"h{h}" for h in POLL_HORIZONS]
        for arm in T2_ARMS:
            seeds = {t: {"string": seed_string(origin, t), "int": t5_seed(origin, t)} for t in TARGET_NAMES}
            err = t2_info["errors"].get(arm)
            if err is None and (set(quantiles[arm]) != set(target_names) or arm not in election):
                err = "missing target output"
            if err:
                arm_status[arm] = {"status": "FAILED", "error": err, "n": n_draws, "seeds": seeds}
                continue
            q = np.stack([quantiles[arm][t] for t in target_names]).astype(np.float32)
            files[f"{arm}.npz"] = npz_bytes([("election_draws_centipp", to_centipp(election[arm])),
                                             ("poll_quantiles", q)])
            arm_status[arm] = {"status": "OK", "n": n_draws, "seeds": seeds, "file": f"{arm}.npz"}

        # C0p (§2.4 a)
        t0 = time.time()
        check_clock(clock())
        identical, differing = production_model_files_identical(repo, snap_commit, code_commit)
        seeds = {t: {"string": seed_string(origin, t), "int": t5_seed(origin, t)} for t in target_names}
        try:
            z_polls = {t: t2_info["targets"][t]["_z_poll"] for t in target_names}
            q, c0p_diag = run_c0p(export_root, ctx, origin, n_draws, identical, work, z_polls)
            files["C0p.npz"] = npz_bytes([("poll_quantiles", q)])
            arm_status["C0p"] = {"status": "OK", "n": n_draws, "seeds": seeds, "file": "C0p.npz",
                                 "code": "in-process" if identical else "anchor-commit export",
                                 "model_files_differing": differing}
        except Exception as exc:  # noqa: BLE001
            c0p_diag = {}
            arm_status["C0p"] = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "n": n_draws,
                                 "seeds": seeds}
        timings["C0p"] = time.time() - t0

        # FB and CAND (§2.2, §2.3)
        from . import candidate_engine as ce

        engine_seed = t5_seed(origin, "engine")
        engine_seeds = {"engine": {"string": seed_string(origin, "engine"), "int": engine_seed}}
        engine_arms = [("FB", ce.config_fb(), fb_verdict["label"] if fb_verdict else None)]
        if activation is not None and origin >= date.fromisoformat(activation["completion_day_utc"]):
            engine_arms.append(("CAND", ce.config_cand(activation["selection"]), activation["label"]))
        for name, config, label in engine_arms:
            t0 = time.time()
            check_clock(clock())
            status: dict[str, Any] = {"n": n_draws, "seeds": engine_seeds, "config": config.to_dict()}
            if label is not None:
                status["label"] = label
            try:
                votes = run_engine_arm(config, export_root, Path(ctx["v03_file"]), origin, n_draws, engine_seed,
                                       work, ctx)
                files[f"{name}.npz"] = npz_bytes([("election_draws_centipp", to_centipp(votes))])
                status.update(status="OK", file=f"{name}.npz")
            except Exception as exc:  # noqa: BLE001
                status.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            arm_status[name] = status
            timings[name] = time.time() - t0

        if c0_draws_file is not None:
            files["C0.npz"] = c0_draws_file
        for arm, st in arm_status.items():
            if st.get("file"):
                st["file_sha256"] = sha256_bytes(files[st["file"]])

        st0 = A.origin_state(ctx, origin)
        omega = A.omega_at(ctx, origin)
        theta = A.theta_details(ctx, origin)
        v03_check = self_check(ctx["rows"])
        diagnostics = {
            "capture_date": origin, "h_E": (ELECTION_2030 - origin).days,
            "v03_reading_at_O": dict(zip(PARTIES8, st0["mean"].tolist())),
            "v03_self_check_ok_reported_only": bool(v03_check["ok"]),
            "omega_O": None if omega is None else omega[0], "omega_centres": None if omega is None else omega[1],
            "k": {arm: {"k": k[arm][0], "eligible_targets": k[arm][1], "errors": 8 * k[arm][1]} for arm in U_ARMS},
            "theta": theta,
            "first_scored_origin": A.first_scored_origin(ctx), "k_grid_scored_origins": len(ctx["scored_origins"]),
            "targets": {t: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                        for t, v in t2_info["targets"].items()},
            "C0p": c0p_diag,
            "snapshot_counts": ctx["t5_counts"],
            "target_polls_excluded_after_origin": ctx["target_polls_excluded_after_cutoff"],
            "election_days": ctx["election_days"],
        }
        files["diagnostics.json"] = canonical_bytes(diagnostics)
        inputs = input_hashes(export_root, ctx)

    run_finished = clock()
    timings["total"] = time.time() - t_all
    c0_reference = {
        **c0, "snapshot_file_sha256": anchor["snapshot_file_sha256"],
        "deterministic_payload_sha256": anchor["deterministic_payload_sha256"],
        "first_archive_commit": archive_commit, "first_archive_commit_time": fmt_utc(archive_time),
    }
    if c0_draws_file is not None:
        c0_reference["file"] = "C0.npz"
        c0_reference["file_sha256"] = sha256_bytes(c0_draws_file)
    arm_status["C0"] = c0_reference
    manifest = {
        "schema": "calibration-2030-t5-shadow-capture-v1",
        "identity": {"capture_date": origin, "iso_week": iso_week_label(origin), "h_E": (ELECTION_2030 - origin).days,
                     "run_started_utc": fmt_utc(run_started), "run_finished_utc": fmt_utc(run_finished),
                     "code_commit": code_commit, "tracked_clean": not tracked_dirty(repo), "runtime": runtime()},
        "anchor": {"generation_id": anchor["generation_id"], "generated_at_utc": anchor["generated_at_utc"],
                   "as_of": anchor["as_of"], "model_version": anchor.get("model_version"),
                   "source_git_commit": snap_commit, "snapshot_path": anchor["path"],
                   "snapshot_file_sha256": anchor["snapshot_file_sha256"],
                   "draws_file_sha256": c0.get("draws_file_sha256"),
                   "deterministic_payload_sha256": anchor["deterministic_payload_sha256"],
                   "first_archive_commit": archive_commit, "first_archive_commit_time": fmt_utc(archive_time),
                   "production_ref": prod_ref, "production_ref_commit": prod_tip, "C0_status": c0["status"]},
        "inputs": inputs,
        "arms": arm_status,
        "activation": None if activation is None else {"cand": activation, "cand_active": "CAND" in arm_status},
        "preregistration": {"document": T5_DOC_REL, "body_sha256": T5_BODY_SHA256,
                            "record_sha256": provenance["t5_record_sha256"],
                            "amendments_in_force": provenance["t5_amendments_in_force"]},
        "files": {name: sha256_bytes(data) for name, data in sorted(files.items())},
        "timings_seconds": {k2: round(v, 1) for k2, v in timings.items()},
        "dry_run": bool(dry_run), "dry_run_refusals": warnings,
    }
    manifest_bytes = canonical_bytes(manifest)
    ok_arms = [a for a, s in arm_status.items() if s.get("status") in ("OK", "VERIFIED", "C0_RECOMPUTED")]
    entry = make_entry(index, capture_date=origin, manifest_sha256=sha256_bytes(manifest_bytes), arms=ok_arms,
                       code_commit=code_commit)
    new_index = append_capture(index, entry, iso_week_label(origin))
    files["manifest.json"] = manifest_bytes
    summary = {"capture_date": origin.isoformat(), "iso_week": week, "anchor": anchor["generation_id"],
               "anchor_generated_at_utc": anchor["generated_at_utc"], "source_git_commit": snap_commit,
               "C0": c0["status"], "arms": {a: s.get("status") for a, s in arm_status.items()},
               "bytes": {n: len(b) for n, b in files.items()}, "total_bytes": sum(len(b) for b in files.values()),
               "timings_seconds": manifest["timings_seconds"], "dry_run": dry_run, "refusals_if_writing": warnings}
    if dry_run:
        if dry_run_out is not None:
            out = Path(dry_run_out).resolve()
            if out == ROOT or ROOT in out.parents or repo.resolve() in (out, *out.parents):
                raise T5Refusal("refusing: --dry-run-out must be outside the repository")
            target = out / origin.isoformat()
            target.mkdir(parents=True, exist_ok=True)
            for name, data in files.items():
                (target / name).write_bytes(data)
            (out / INDEX_NAME).write_bytes(canonical_bytes(new_index))
            summary["dry_run_out"] = str(target)
        return summary
    rel_files = {SHADOW_REL / origin.isoformat() / name: data for name, data in files.items()}
    commit = write_and_commit(repo, rel_files, new_index, f"t5: shadow capture {origin.isoformat()} ({week})",
                              clock, expected_head=code_commit, week_end=week_start + timedelta(days=7))
    summary["commit"] = commit
    return summary


# --------------------------------------------------------------------------- verify (§4.5)

def verify(repo: Path = ROOT, capture_date: str | None = None, prod_ref: str = PRODUCTION_REF,
           replay: bool = False, ref: str = SHADOW_BRANCH) -> dict:
    index = committed_index(repo, ref)
    report: dict[str, Any] = {"ref": ref, "chain_errors": check_chain(index), "captures": {}}
    for entry in index["captures"]:
        if capture_date and entry["capture_date"] != capture_date:
            continue
        problems, notes = [], []
        base = entry["path"]
        manifest_blob = git_show_bytes(repo, ref, f"{base}/manifest.json")
        if manifest_blob is None:
            report["captures"][entry["capture_date"]] = {"ok": False, "problems": ["manifest missing"]}
            continue
        manifest = json.loads(manifest_blob.decode("utf-8"))
        # 1. files
        if sha256_bytes(manifest_blob) != entry["manifest_sha256"]:
            problems.append("manifest hash differs from its index entry")
        blobs = {}
        for name, digest in manifest["files"].items():
            blob = git_show_bytes(repo, ref, f"{base}/{name}")
            blobs[name] = blob
            if blob is None or sha256_bytes(blob) != digest:
                problems.append(f"{name}: hash mismatch or missing")
        for arm, st in manifest["arms"].items():
            if st.get("file") and manifest["files"].get(st["file"]) != st.get("file_sha256"):
                problems.append(f"{arm}: per-arm file hash differs from the files table")
        # 2. timing
        first = first_commit_with_blob(repo, ref, f"{base}/manifest.json", entry["manifest_sha256"])
        anchor = manifest["anchor"]
        if first is None:
            problems.append("no commit on the branch contains the manifest")
        else:
            commit, when = first
            monday = week_of_label(manifest["identity"]["iso_week"])
            start = datetime.combine(monday, dtime(0, 0), tzinfo=timezone.utc)
            if not (start <= when < start + timedelta(days=7)):
                problems.append(f"manifest first committed at {fmt_utc(when)}, outside {manifest['identity']['iso_week']}")
            if when <= parse_utc(anchor["first_archive_commit_time"]):
                problems.append("manifest committed before the anchor's first archive commit")
            notes.append("committer times are self-reported while shadow/2030 is local only")
        # 3. inputs
        snaps = {s.get("generation_id"): s for s in production_snapshots(repo, prod_ref)}
        snap = snaps.get(anchor["generation_id"])
        if snap is None or snap.get("snapshot_file_sha256") != anchor["snapshot_file_sha256"] or \
                snap.get("deterministic_payload_sha256") != anchor["deterministic_payload_sha256"]:
            problems.append("anchor generation missing from production's archive or hashes differ")
        csv_blob = git_show_bytes(repo, anchor["source_git_commit"], SNAP_POLLS_CSV)
        if csv_blob is None or sha256_bytes(csv_blob) != manifest["inputs"]["swedishpolls_snapshot_sha256"]:
            problems.append("polls CSV at the anchor commit does not match the recorded snapshot hash")
        # 4. C0
        if anchor["C0_status"] != "C0_RECOMPUTED":
            with tempfile.TemporaryDirectory(prefix="calibration2030-t5-verify-") as tmp:
                try:
                    verify_c0_sidecar(repo, {**anchor, "path": anchor["snapshot_path"]}, Path(tmp), prod_ref)
                except Exception as exc:  # noqa: BLE001
                    problems.append(f"C0 sidecar fails verification: {exc}")
        # 5. code
        code = manifest["identity"]["code_commit"]
        if git(repo, "cat-file", "-e", f"{code}^{{commit}}", check=False).returncode != 0 or \
                not is_ancestor(repo, code, ref):
            problems.append("code commit missing or not an ancestor of the branch")
        # CAND ≡ FB (§2.3)
        act = (manifest.get("activation") or {}).get("cand") or {}
        if act.get("cand_equals_fb") and "CAND.npz" in blobs and "FB.npz" in blobs:
            if blobs["CAND.npz"] != blobs["FB.npz"]:
                problems.append("CAND and FB are the same model but their draws differ")
        # 6. replay (diagnostic only)
        if replay:
            notes.append(_replay_note(repo, manifest))
        report["captures"][entry["capture_date"]] = {"ok": not problems, "problems": problems, "notes": notes}
    report["ok"] = not report["chain_errors"] and all(c["ok"] for c in report["captures"].values())
    return report


def _replay_note(repo: Path, manifest: Mapping) -> str:
    # PREREG-AMBIGUITY: §4.5.6 replays "at its code commit and anchor". A replay
    # needs the code commit checked out and the original clock; it is reported
    # as a diagnostic only. Reading implemented: verify reports whether the
    # checked-out code equals the recorded code commit and leaves the rerun to an
    # explicit capture --dry-run in a worktree at that commit.
    code = manifest["identity"]["code_commit"]
    head = git_out(repo, "rev-parse", "HEAD")
    same = git_out(repo, "rev-parse", f"{head}^{{tree}}") == git_out(repo, "rev-parse", f"{code}^{{tree}}")
    return (f"replay: HEAD tree {'equals' if same else 'differs from'} code commit {code[:12]}; "
            f"runtime here {runtime()} vs recorded {manifest['identity']['runtime']}; a mismatch on another "
            "runtime is reported, not a failure")


# --------------------------------------------------------------------------- report (§5)

def _scoring_snapshot(repo: Path, prod_ref: str) -> tuple[dict, bytes, Path | None]:
    """The latest certified 2030 generation on production: its as_of and its polls CSV (§5.1)."""

    # PREREG-AMBIGUITY: §5.1 scores from "the snapshot used for scoring" without
    # naming it. Reading implemented: the latest certified 2030 generation in
    # production's archive at report time; its as_of is the snapshot date and
    # the CSV at its source_git_commit the polls.
    snaps = [s for s in production_snapshots(repo, prod_ref) if s.get("election_date") == ELECTION_2030.isoformat()]
    latest = max(snaps, key=lambda s: parse_utc(s["generated_at_utc"]))
    blob = git_show_bytes(repo, latest["source_git_commit"], SNAP_POLLS_CSV)
    return latest, blob, None


def target_ctx_from_csv(csv_path: Path, elections_csv: Path, manifest_2026: Path) -> dict:
    from scripts.poll_aggregate.data import load_elections, load_polls

    elections = load_elections(elections_csv, [manifest_2026])
    tp = load_polls(csv_path, [e.obs_date for e in elections]).polls
    return {"tp_obs": np.array([o.obs_date.toordinal() for o in tp], dtype=np.int64),
            "tp_avail": np.array([o.available.toordinal() for o in tp], dtype=np.int64),
            "tp_n": np.array([float(o.sample_size) for o in tp]),
            "tp_shares": np.array([o.shares for o in tp], dtype=float),
            "cache": {"target": {}, "window": {}}}


def score_quantiles(q: np.ndarray, y: np.ndarray) -> dict:
    """Quantile CRPS per party and 50/90 % coverage (inclusive, as calculate_interval_metrics)."""

    from scripts.pollofpolls.backtest_metrics import calculate_interval_metrics

    crps, c50, c90 = [], [], []
    for p in range(len(PARTIES8)):
        crps.append(quantile_crps(q[:, p], float(y[p])))
        qd = {lv: float(q[int(round(lv * 100)) - 1, p]) for lv in INTERVAL_LEVELS}
        iv = calculate_interval_metrics(qd, float(y[p]))
        c50.append(int(iv["interval50_contains_actual"]))
        c90.append(int(iv["interval90_contains_actual"]))
    return {"crps": crps, "crps_mean": float(np.mean(crps)), "cov50": c50, "cov90": c90}


def report(repo: Path = ROOT, quarter: str = "", *, dry_run: bool = False, clock: Callable[[], datetime] = utc_now,
           prod_ref: str = PRODUCTION_REF, fetch: bool = True, ref: str = "HEAD") -> dict:
    """§5.4: the quarterly report (machine record + Markdown under shadow_2030/reports/)."""

    from . import t2_arms as A

    if len(quarter) != 6 or quarter[4] != "Q" or quarter[5] not in "1234":
        raise ValueError("--quarter must look like 2027Q1")
    now = clock()
    preflight(repo, now, writing=not dry_run)
    if fetch:
        git(repo, "fetch", "--quiet", PRODUCTION_REMOTE,
            f"+refs/heads/{PRODUCTION_BRANCH}:refs/remotes/{PRODUCTION_REMOTE}/{PRODUCTION_BRANCH}")
    json_rel = REPORTS_REL / f"t5_report_{quarter}.json"
    if git(repo, "cat-file", "-e", f"{ref}:{json_rel.as_posix()}", check=False).returncode == 0:
        raise T5Refusal(f"refusing: a report for {quarter} already exists (at most one per quarter)")
    index = committed_index(repo, ref)
    frozen = committed_json(repo, TARGETS_REL, ref) or {"schema": "calibration-2030-t5-closed-targets-v1",
                                                        "targets": []}
    known = {(t["capture_date"], t["target"]): t for t in frozen["targets"]}
    snap, csv_blob, _ = _scoring_snapshot(repo, prod_ref)
    snap_date = date.fromisoformat(snap["as_of"])
    csv_sha = sha256_bytes(csv_blob)
    new_targets = []
    with tempfile.TemporaryDirectory(prefix="calibration2030-t5-report-") as tmp:
        root = export_tree(repo, snap["source_git_commit"], Path(tmp) / "snap",
                           paths=(SNAP_POLLS_CSV, SNAP_ELECTIONS, SNAP_MANIFEST_2026), exclude=())
        tctx = target_ctx_from_csv(root / SNAP_POLLS_CSV, root / SNAP_ELECTIONS, root / SNAP_MANIFEST_2026)
        for entry in index["captures"]:
            o = date.fromisoformat(entry["capture_date"])
            for h in POLL_HORIZONS:
                key = (entry["capture_date"], f"h{h}")
                d = o + timedelta(days=h)
                if key in known or d + timedelta(days=WINDOW_HALF + CLOSURE_LAG_DAYS) > snap_date:
                    continue
                value = A.poll_target(tctx, d)
                n_polls = int(A._window_mask(tctx, d).sum())
                rec = {"capture_date": entry["capture_date"], "target": f"h{h}", "target_date": d.isoformat(),
                       "value": None if value is None else [float(v) for v in value], "window_polls": n_polls,
                       "scoring_generation_id": snap["generation_id"], "scoring_snapshot_date": snap["as_of"],
                       "polls_csv_sha256": csv_sha}
                new_targets.append(rec)
                known[key] = rec
    frozen_new = {**frozen, "targets": [*frozen["targets"], *new_targets]}

    cells, failures, k_theta, fb_d0, c0_versions = [], [], {}, {}, {}
    for entry in index["captures"]:
        base = entry["path"]
        manifest = json.loads(git_show_bytes(repo, ref, f"{base}/manifest.json").decode("utf-8"))
        diag = json.loads(git_show_bytes(repo, ref, f"{base}/diagnostics.json").decode("utf-8"))
        k_theta[entry["capture_date"]] = {"k": diag["k"], "theta": diag["theta"].get("theta")}
        c0_versions[entry["capture_date"]] = manifest["anchor"].get("model_version")
        for arm, st in manifest["arms"].items():
            if st.get("status") == "FAILED":
                failures.append({"capture_date": entry["capture_date"], "arm": arm, "error": st.get("error")})
        arrays = {}
        for arm in (*T2_ARMS, "C0p", "FB"):
            blob = git_show_bytes(repo, ref, f"{base}/{arm}.npz")
            if blob is not None:
                arrays[arm] = load_npz(blob)
        if "FB" in arrays and "D0" in arrays:
            fb_d0[entry["capture_date"]] = {
                "mean_diff_pp": (from_centipp(arrays["FB"]["election_draws_centipp"]).mean(axis=0)
                                 - from_centipp(arrays["D0"]["election_draws_centipp"]).mean(axis=0)).tolist(),
                "sd_ratio": (from_centipp(arrays["FB"]["election_draws_centipp"]).std(axis=0)
                             / from_centipp(arrays["D0"]["election_draws_centipp"]).std(axis=0)).tolist()}
        for i, h in enumerate(POLL_HORIZONS):
            t = known.get((entry["capture_date"], f"h{h}"))
            if t is None or t["value"] is None:
                continue
            y = np.asarray(t["value"])
            for arm, arr in arrays.items():
                if "poll_quantiles" in arr:
                    cells.append({"capture_date": entry["capture_date"], "h": h, "arm": arm,
                                  "code_commit": entry["code_commit"],
                                  **score_quantiles(arr["poll_quantiles"][i].astype(np.float64), y)})
    by_h: dict[str, dict] = {}
    for h in POLL_HORIZONS:
        for arm in (*T2_ARMS, "C0p"):
            sel = [c for c in cells if c["h"] == h and c["arm"] == arm]
            if sel:
                by_h.setdefault(str(h), {})[arm] = _pool_cells(sel)
    pooled = {arm: _pool_cells([c for c in cells if c["arm"] == arm and c["h"] >= 91])
              for arm in (*T2_ARMS, "C0p") if any(c["arm"] == arm and c["h"] >= 91 for c in cells)}
    counted = sorted({c["capture_date"] for c in cells if c["h"] == 91})
    record = {
        "schema": "calibration-2030-t5-report-v1", "quarter": quarter, "generated_utc": fmt_utc(now),
        "preregistration_body_sha256": T5_BODY_SHA256, "shadow_ref_commit": git_out(repo, "rev-parse", ref),
        "scoring_snapshot": {"generation_id": snap["generation_id"], "as_of": snap["as_of"],
                             "polls_csv_sha256": csv_sha},
        "captures": len(index["captures"]), "missed_weeks": index["missed_weeks"],
        "captures_with_scored_h91": len(counted),
        "by_horizon": by_h, "pooled_h_ge_91": pooled, "cells": cells,
        "election_day": "not scored: the certified 2030 result is not committed (§5.2)",
        "k_theta_over_time": k_theta, "failures": failures,
        "fb_d0_implementation_check_reported_only": fb_d0, "c0_model_version_by_capture": c0_versions,
        "note": "reports never change arms, horizons, targets, metrics, cadence or captures (§5.4)",
    }
    # PREREG-AMBIGUITY: §5.4 names docs/calibration_2030_t5_report_<YYYY>Q<q>.md,
    # but §7 lets the runner write only under data/processed/shadow_2030/.
    # Reading implemented: both the Markdown and the machine record are written
    # under shadow_2030/reports/; copying the Markdown to docs/ is a separate step.
    md = _report_markdown(record)
    files = {json_rel: canonical_bytes(record), REPORTS_REL / f"calibration_2030_t5_report_{quarter}.md":
             md.encode("utf-8")}
    if new_targets:
        files[TARGETS_REL] = canonical_bytes(frozen_new)
    summary = {"quarter": quarter, "captures": len(index["captures"]), "new_closed_targets": len(new_targets),
               "scored_cells": len(cells), "dry_run": dry_run}
    if dry_run:
        return {**summary, "record": record}
    # closed_targets.json is append-only by content: its old entries must be a prefix.
    if frozen["targets"] != frozen_new["targets"][:len(frozen["targets"])]:
        raise T5Refusal("refusing: frozen targets would change")
    summary["commit"] = _commit_report(repo, files, clock, quarter)
    return summary


def _commit_report(repo: Path, files: Mapping[Path, bytes], clock, quarter: str) -> str:
    now = clock()
    preflight(repo, now, writing=True)
    for rel, data in files.items():
        if rel != TARGETS_REL and git(repo, "cat-file", "-e", f"HEAD:{rel.as_posix()}", check=False).returncode == 0:
            raise T5Refusal(f"refusing: {rel} already committed")
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_bytes(data)
    paths = [r.as_posix() for r in files]
    git(repo, "add", "--", *paths)
    git(repo, "commit", "--quiet", "-m", f"t5: shadow report {quarter}", "--", *paths)
    return git_out(repo, "rev-parse", "HEAD")


def _pool_cells(sel: list[dict]) -> dict:
    return {"cells": len(sel), "crps": float(np.mean([c["crps_mean"] for c in sel])),
            "coverage50": float(np.mean([v for c in sel for v in c["cov50"]])),
            "coverage90": float(np.mean([v for c in sel for v in c["cov90"]]))}


def _report_markdown(record: Mapping) -> str:
    lines = [f"# Calibration 2030, T5 shadow run: report {record['quarter']}", "",
             f"Generated {record['generated_utc']} from `shadow/2030` at `{record['shadow_ref_commit'][:12]}`. "
             f"Scoring snapshot: production generation `{record['scoring_snapshot']['generation_id']}` "
             f"(as of {record['scoring_snapshot']['as_of']}). Reports change nothing (§5.4).", "",
             f"- Captures: {record['captures']}; missed weeks: {len(record['missed_weeks'])}",
             f"- Captures with a scored 91-day target: {record['captures_with_scored_h91']} (26 needed, §6)",
             f"- Failures: {len(record['failures'])}", f"- Election day: {record['election_day']}", "",
             "## Quantile CRPS and coverage by horizon", "",
             "| h | arm | cells | CRPS (pp) | cov50 | cov90 |", "| --- | --- | --- | --- | --- | --- |"]
    for h, arms in record["by_horizon"].items():
        for arm, v in arms.items():
            lines.append(f"| {h} | {arm} | {v['cells']} | {v['crps']:.4f} | {v['coverage50']:.3f} | {v['coverage90']:.3f} |")
    lines += ["", "## Pooled, h ≥ 91", "", "| arm | cells | CRPS (pp) | cov50 | cov90 |", "| --- | --- | --- | --- | --- |"]
    for arm, v in record["pooled_h_ge_91"].items():
        lines.append(f"| {arm} | {v['cells']} | {v['crps']:.4f} | {v['coverage50']:.3f} | {v['coverage90']:.3f} |")
    lines += ["", "k, θ, failures, missed weeks, the FB–D0 implementation check and C0's model versions are in the "
              "machine record.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- activation command

def activate_cand(repo: Path = ROOT, t4_path: str = DEFAULT_T4_RESULTS, *, dry_run: bool = False,
                  clock: Callable[[], datetime] = utc_now) -> dict:
    preflight(repo, clock(), writing=not dry_run)
    if committed_json(repo, CAND_ACTIVATION_REL) is not None:
        raise T5Refusal("refusing: CAND is already activated (append-only record)")
    cand_record, fb_record = build_activation(repo, t4_path)
    files = {CAND_ACTIVATION_REL: canonical_bytes(cand_record)}
    if fb_record is not None and committed_json(repo, FB_VERDICT_REL) is None:
        files[FB_VERDICT_REL] = canonical_bytes(fb_record)
    if dry_run:
        return {"cand": cand_record, "fb": fb_record, "dry_run": True}
    commit = write_and_commit(repo, files, None, "t5: activate CAND (section 2.3)", clock)
    return {"cand": cand_record, "fb": fb_record, "commit": commit}


# --------------------------------------------------------------------------- CLI

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture", help="one weekly capture (§3)")
    cap.add_argument("--dry-run", action="store_true", help="compute everything; write and commit nothing")
    cap.add_argument("--dry-run-out", type=Path, default=None, help="with --dry-run: write the would-be files here")
    cap.add_argument("--no-fetch", action="store_true", help="use the current origin/main without fetching")
    cap.add_argument("--workers", type=int, default=1, help="processes for the per-origin k/θ/Ω recomputation")
    ver = sub.add_parser("verify", help="verify the archive (§4.5)")
    ver.add_argument("--capture", default=None)
    ver.add_argument("--replay", action="store_true")
    ver.add_argument("--ref", default=SHADOW_BRANCH)
    rep = sub.add_parser("report", help="quarterly report (§5.4)")
    rep.add_argument("--quarter", required=True)
    rep.add_argument("--dry-run", action="store_true")
    rep.add_argument("--no-fetch", action="store_true")
    act = sub.add_parser("activate-cand", help="record CAND's activation after T4 completes (§2.3)")
    act.add_argument("--t4-results", default=DEFAULT_T4_RESULTS)
    act.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "capture":
            out = capture(ROOT, dry_run=args.dry_run, dry_run_out=args.dry_run_out, fetch=not args.no_fetch,
                          workers=args.workers)
        elif args.command == "verify":
            out = verify(ROOT, args.capture, replay=args.replay, ref=args.ref)
        elif args.command == "report":
            out = report(ROOT, args.quarter, dry_run=args.dry_run, fetch=not args.no_fetch)
            out.pop("record", None)
        else:
            out = activate_cand(ROOT, args.t4_results, dry_run=args.dry_run)
    except T5Refusal as exc:
        print(str(exc), file=sys.stderr)
        return 3
    print(json.dumps(_jsonable(out), indent=2, sort_keys=True))
    return 0 if out.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
