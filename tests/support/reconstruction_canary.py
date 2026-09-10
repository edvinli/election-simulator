"""Prove on a real runner that a render reconstructs a missing curve date.

The rendering path is hard to exercise. It only does real work when a
publication certified and then failed, and against a healthy deployment there
is nothing to reconstruct -- so the one acceptance criterion a runner cannot
otherwise reach is "the historical reconstruction actually runs here".

This punches a single, known gap into a *disposable clone* of the simulator's
source history and dry-renders it. The renderer installs nothing, but the
fixture commit is a real commit, so the safety property is that the clones are
throwaway rather than that the command is read-only: both repositories are
always cloned fresh into a temporary workspace that is removed on the way out,
and no existing checkout is ever touched. Nothing is pushed, and the run
asserts afterwards that neither clone's HEAD moved past the fixture.

Written as a script rather than a procedure because both of its failure modes
present as success. Rehearsing it by hand produced two false greens: a fixture
the history contract rejected let the run continue and report "nothing to
reconstruct" against the unmodified history, and a later `set -e` truncated the
run at the renderer's *expected* non-zero exit so the safety assertions never
executed. Every step below is therefore asserted, and the renderer's exit code
is captured without short-circuiting what follows.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_HISTORY = Path("files/election-simulator/history/coalition-timeseries.json")
SITE_POINTER = Path("files/election-simulator/current.json")

#: Paths a curve repair must never touch. The certified publication and the
#: archive are what make a render a *rendering* rather than a new forecast.
PROTECTED = (
    "files/election-simulator/versions",
    "files/election-simulator/current.json",
    "data/processed/prospective_forecasts",
)


class CanaryError(RuntimeError):
    """A canary that cannot establish its own preconditions."""


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise CanaryError(
            f"{' '.join(command)} failed ({completed.returncode}):\n{completed.stderr}")
    return completed.stdout.strip()


def _clone(remote: str, branch: str, destination: Path) -> Path:
    _run(["git", "clone", "--quiet", "--branch", branch, remote, str(destination)])
    for key, value in (("user.name", "Reconstruction canary"),
                       ("user.email", "canary@example.invalid")):
        _run(["git", "config", key, value], cwd=destination)
    return destination


def punch_gap(source: Path, target_date: str) -> None:
    """Remove one reconstructed curve point, leaving a contract-valid history.

    Four structures are coupled and all four have to move together, which is
    what the hand-run version got wrong: the series, the observation count,
    the reconstruction fingerprint registry, and the payload digest. Dropping
    the fingerprint is load-bearing rather than bookkeeping -- it is what
    tells the resume cache to re-simulate this date instead of reusing it.
    """

    sys.path.insert(0, str(source))
    from scripts.forecast_history.contract import (  # noqa: PLC0415
        deterministic_history_sha256,
        validate_history_contract,
    )
    from scripts.forecast_history.generate import missing_curve_dates  # noqa: PLC0415

    path = source / SOURCE_HISTORY
    payload = json.loads(path.read_text(encoding="utf-8"))

    before = [day.isoformat() for day in missing_curve_dates(payload)]
    if before:
        raise CanaryError(
            f"the source history already has missing curve dates {before}; "
            "this canary needs a known-good starting point")

    kept = [
        point for point in payload["series"]
        if not (point["date"] == target_date
                and point["provenance"] == "reconstructed_current_model")
    ]
    if len(kept) != len(payload["series"]) - 1:
        raise CanaryError(
            f"expected exactly one reconstructed point on {target_date}, "
            f"removed {len(payload['series']) - len(kept)}")
    payload["series"] = kept
    payload["schedule"]["observation_count"] = len(kept)
    registry = (payload.get("reconstruction_inputs") or {}).get("dates")
    if isinstance(registry, dict):
        registry.pop(target_date, None)
    payload["deterministic_content_sha256"] = deterministic_history_sha256(payload)
    validate_history_contract(payload)

    after = [day.isoformat() for day in missing_curve_dates(payload)]
    if after != [target_date]:
        raise CanaryError(
            f"the fixture must leave exactly one gap on {target_date}, left {after}")
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    _run(["git", "add", str(SOURCE_HISTORY)], cwd=source)
    _run(["git", "commit", "--quiet", "-m",
          f"canary fixture: remove the reconstructed point for {target_date}"], cwd=source)
    touched = _run(["git", "diff", "--name-only", "HEAD~1", "HEAD"], cwd=source).splitlines()
    if touched != [str(SOURCE_HISTORY)]:
        raise CanaryError(f"the fixture touched more than the history: {touched}")


#: What a run established. A recognised parity omission is a real result --
#: reconstruction ran and the curve closed -- but it is not the full runner
#: acceptance, which requires both future views rebuilt. Keeping them distinct
#: stops a known limitation from being read as a clean pass.
FULL_ACCEPTANCE = "FULL_ACCEPTANCE"
RECONSTRUCTION_ONLY = "RECONSTRUCTION_ONLY"
FAILED = "FAILED"

#: Exit codes. 2 is deliberately not 0: reconstruction-only is not acceptance.
EXIT_CODES = {FULL_ACCEPTANCE: 0, FAILED: 1, RECONSTRUCTION_ONLY: 2}


@dataclass(frozen=True)
class RepositoryState:
    """One repository's observable state, as the canary compares it.

    ``protected`` is keyed by path *within this repository*. It used to be a
    single dict keyed by path across both, so the website's entries silently
    overwrote the simulator's and both were then compared against the
    website's tree hashes -- which differ by construction, so the check would
    have reported a change that had not happened.
    """

    head: str
    status: str
    protected: dict[str, str]


def _baseline(repository: Path, protected: tuple[str, ...]) -> RepositoryState:
    return RepositoryState(
        head=_run(["git", "rev-parse", "HEAD"], cwd=repository),
        status=_run(["git", "status", "--porcelain"], cwd=repository),
        protected={
            name: _run(["git", "rev-parse", f"HEAD:{name}"], cwd=repository)
            for name in protected
            if (repository / name).exists()
        },
    )


def evaluate(
    *,
    target_date: str,
    expected_generation: str,
    rendered: Mapping[str, Any] | None,
    parse_error: str | None,
    exit_code: int,
    before: Mapping[str, RepositoryState],
    after: Mapping[str, RepositoryState],
    pointer_before: bytes,
    pointer_after: bytes,
) -> tuple[list[str], str]:
    """Decide what a run established, from state alone.

    Pure so it can be tested without a render. The safety checks run first and
    unconditionally: an unreadable renderer result is a failure *of the render*
    and must not stop the canary from reporting whether production state was
    disturbed, which is the more important question and used to be skipped
    entirely when result parsing raised.
    """

    failures: list[str] = []

    # --- safety, always, whatever the renderer said -----------------------
    for label in sorted(set(before) | set(after)):
        base, now = before.get(label), after.get(label)
        if base is None or now is None:
            failures.append(f"{label} state was not captured on both sides")
            continue
        if now.head != base.head:
            failures.append(f"{label} HEAD moved: {base.head} -> {now.head}")
        if now.status != base.status:
            failures.append(f"{label} working tree changed: {now.status!r}")
        for name, tree in base.protected.items():
            if now.protected.get(name) != tree:
                failures.append(
                    f"{label} protected path changed: {name} "
                    f"({tree} -> {now.protected.get(name)})")
        for name in set(now.protected) - set(base.protected):
            failures.append(f"{label} protected path appeared: {name}")
    if pointer_after != pointer_before:
        failures.append("the website pointer changed")

    # --- what the render itself reported ----------------------------------
    if parse_error is not None or not isinstance(rendered, Mapping):
        failures.append(parse_error or "the renderer produced no usable result")
        return failures, FAILED

    curve = rendered.get("curve") or {}
    views = curve.get("views") or {}

    if rendered.get("generation") != expected_generation:
        failures.append(
            f"the render reports generation {rendered.get('generation')!r}, "
            f"expected {expected_generation!r}")
    if rendered.get("status") != "RENDER_STAGED_NOT_INSTALLED":
        failures.append(
            f"expected a staged dry run, got status {rendered.get('status')!r}")
    if target_date not in (curve.get("reconstructed") or []):
        failures.append(
            f"{target_date} was not reconstructed "
            f"(reconstructed={curve.get('reconstructed')})")
    if curve.get("missing"):
        failures.append(f"curve dates still missing: {curve['missing']}")
    if curve.get("status") != "COMPLETE":
        failures.append(f"curve status is {curve.get('status')!r}")

    primary = views.get("primary_campaign_paths")
    parity_omission = (
        primary == "OMITTED"
        and "bitwise identical" in str(views.get("error") or ""))
    if primary == "OMITTED" and not parity_omission:
        failures.append(
            f"the primary view was omitted for an unexpected reason: "
            f"{views.get('error')!r}")
    elif primary not in {"REBUILT", "OMITTED", "NOT_REQUIRED_ON_ELECTION_DAY"}:
        failures.append(f"unrecognised primary view state {primary!r}")

    both_views_rebuilt = (
        primary == "REBUILT"
        and views.get("secondary_projection") == "REBUILT"
        and views.get("status") == "COMPLETE"
        and not (views.get("missing") or []))

    # A non-zero renderer exit is expected only for the recognised omission.
    if exit_code != 0 and not parity_omission:
        failures.append(
            f"the renderer exited {exit_code} with nothing recognised as short")
    if exit_code == 0 and not both_views_rebuilt:
        failures.append(
            f"the renderer exited 0 without both views rebuilt (views={views})")

    if failures:
        return failures, FAILED
    return failures, FULL_ACCEPTANCE if both_views_rebuilt else RECONSTRUCTION_ONLY


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator-remote", required=True)
    parser.add_argument("--website-remote", required=True)
    parser.add_argument("--simulator-ref", default="main")
    parser.add_argument("--website-ref", default="master")
    parser.add_argument("--generation", required=True)
    parser.add_argument("--certification-commit", default=None)
    parser.add_argument("--target-date", required=True,
                        help="The reconstructed curve date to remove and expect back")
    parser.add_argument("--history-workers", type=int, default=4)
    args = parser.parse_args(argv)

    workspace = Path(tempfile.mkdtemp(prefix="reconstruction-canary-"))
    try:
        source = _clone(args.simulator_remote, args.simulator_ref, workspace / "simulator")
        site = _clone(args.website_remote, args.website_ref, workspace / "website")
        print(f"simulator {_run(['git', 'rev-parse', 'HEAD'], cwd=source)}", flush=True)
        print(f"website   {_run(['git', 'rev-parse', 'HEAD'], cwd=site)}", flush=True)

        punch_gap(source, args.target_date)
        print(f"fixture   one gap on {args.target_date}", flush=True)

        # Taken *after* the fixture: the fixture commit is expected, anything
        # the renderer adds on top of it is not. Keyed per repository, because
        # the same path has different content in each.
        repositories = {"simulator": (source, PROTECTED), "website": (site, PROTECTED[:2])}
        before = {
            label: _baseline(repository, protected)
            for label, (repository, protected) in repositories.items()
        }
        pointer_before = (site / SITE_POINTER).read_bytes()

        command = [
            sys.executable, "-m", "scripts.election_automation",
            "--repo-root", str(source), "--site-repo", str(site),
            "--mode", "render", "--render-generation", args.generation,
            "--history-workers", str(args.history_workers),
            "--repair", "--render-dry-run",
            "--summary-path", str(workspace / "summary.txt"),
        ]
        if args.certification_commit:
            command += ["--render-certification-commit", args.certification_commit]

        started = time.monotonic()
        # The exit code is captured, never allowed to short-circuit what
        # follows: a non-zero exit is an expected outcome when a required view
        # cannot be verified, and the assertions are the point.
        completed = subprocess.run(
            command, cwd=source, capture_output=True, text=True, check=False)
        elapsed = time.monotonic() - started
        sys.stderr.write(completed.stderr)
        print(f"renderer  exit {completed.returncode} in {elapsed:.1f}s", flush=True)

        # Parsing failure is recorded, not raised. The safety checks matter
        # more than the render's own report and must still run.
        rendered, parse_error = None, None
        for line in reversed(completed.stdout.splitlines()):
            if line.startswith("{") and '"generation"' in line:
                try:
                    rendered = json.loads(line)
                except json.JSONDecodeError as error:
                    parse_error = f"the renderer result is not valid JSON: {error}"
                break
        if rendered is None and parse_error is None:
            parse_error = (
                "the renderer produced no result JSON; last output:\n"
                + completed.stdout[-4000:])

        after = {
            label: _baseline(repository, protected)
            for label, (repository, protected) in repositories.items()
        }
        failures, outcome = evaluate(
            target_date=args.target_date,
            expected_generation=args.generation,
            rendered=rendered,
            parse_error=parse_error,
            exit_code=completed.returncode,
            before=before,
            after=after,
            pointer_before=pointer_before,
            pointer_after=(site / SITE_POINTER).read_bytes(),
        )

        curve = (rendered or {}).get("curve") or {}
        print(json.dumps({
            "outcome": outcome,
            "generation": (rendered or {}).get("generation"),
            "certification_commit": (rendered or {}).get("certification_commit"),
            "status": (rendered or {}).get("status"),
            "renderer_exit_code": completed.returncode,
            "elapsed_seconds": round(elapsed, 1),
            "reconstructed": curve.get("reconstructed"),
            "curve_status": curve.get("status"),
            "views": curve.get("views"),
            "failures": failures,
        }, indent=2), flush=True)

        for failure in failures:
            print(f"FAIL  {failure}", flush=True)
        if outcome == FULL_ACCEPTANCE:
            print("PASS  reconstruction verified, both views rebuilt, "
                  "every gate passed, production untouched", flush=True)
        elif outcome == RECONSTRUCTION_ONLY:
            print("PARTIAL  reconstruction verified and production untouched, but "
                  "the primary campaign-path view was omitted on the known parity "
                  "limitation -- not a full runner acceptance", flush=True)
        return EXIT_CODES[outcome]
    finally:
        os.chdir(REPOSITORY_ROOT)
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
