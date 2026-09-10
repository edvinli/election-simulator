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
from pathlib import Path

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


def _baseline(repository: Path) -> dict[str, str]:
    return {
        "head": _run(["git", "rev-parse", "HEAD"], cwd=repository),
        "status": _run(["git", "status", "--porcelain"], cwd=repository),
    }


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
        # the renderer adds on top of it is not.
        base_source, base_site = _baseline(source), _baseline(site)
        pointer_before = (site / SITE_POINTER).read_bytes()
        protected_before = {
            name: _run(["git", "rev-parse", f"HEAD:{name}"], cwd=repository)
            for repository, names in ((source, PROTECTED), (site, PROTECTED[:2]))
            for name in names
            if (repository / name).exists()
        }

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
        # The exit code is captured, never allowed to short-circuit the
        # assertions below: a non-zero exit is an *expected* outcome when a
        # required view cannot be verified, and the assertions are the point.
        completed = subprocess.run(
            command, cwd=source, capture_output=True, text=True, check=False)
        elapsed = time.monotonic() - started
        sys.stderr.write(completed.stderr)
        print(f"renderer  exit {completed.returncode} in {elapsed:.1f}s", flush=True)

        rendered = None
        for line in reversed(completed.stdout.splitlines()):
            if line.startswith("{") and '"generation"' in line:
                rendered = json.loads(line)
                break
        if rendered is None:
            raise CanaryError(
                f"the renderer produced no result JSON:\n{completed.stdout[-4000:]}")

        curve = rendered.get("curve") or {}
        views = curve.get("views") or {}
        failures: list[str] = []

        # 1. The requested date was reconstructed, and nothing is left short.
        if args.target_date not in (curve.get("reconstructed") or []):
            failures.append(
                f"{args.target_date} was not reconstructed "
                f"(reconstructed={curve.get('reconstructed')})")
        if curve.get("missing"):
            failures.append(f"curve dates still missing: {curve['missing']}")
        if curve.get("status") != "COMPLETE":
            failures.append(f"curve status is {curve.get('status')}")

        # 2. Nothing was installed, in either repository, beyond the fixture.
        for label, repository, base in (("simulator", source, base_source),
                                        ("website", site, base_site)):
            now = _baseline(repository)
            if now["head"] != base["head"]:
                failures.append(f"{label} HEAD moved: {base['head']} -> {now['head']}")
            if now["status"] != base["status"]:
                failures.append(f"{label} working tree changed:\n{now['status']}")
        if (site / SITE_POINTER).read_bytes() != pointer_before:
            failures.append("the website pointer changed")
        for name, tree in protected_before.items():
            for repository in (source, site):
                if (repository / name).exists():
                    if _run(["git", "rev-parse", f"HEAD:{name}"], cwd=repository) != tree:
                        failures.append(f"protected path changed: {name}")

        # 3. An expected parity omission is not an unrelated failure.
        omitted_view = views.get("primary_campaign_paths") == "OMITTED"
        parity = "bitwise identical" in str(views.get("error") or "")
        if omitted_view and not parity:
            failures.append(
                f"the primary view was omitted for an unexpected reason: "
                f"{views.get('error')}")
        if completed.returncode != 0 and not (omitted_view and parity) and not failures:
            failures.append(
                f"the renderer exited {completed.returncode} with nothing reported short")

        print(json.dumps({
            "generation": rendered.get("generation"),
            "certification_commit": rendered.get("certification_commit"),
            "status": rendered.get("status"),
            "elapsed_seconds": round(elapsed, 1),
            "reconstructed": curve.get("reconstructed"),
            "curve_status": curve.get("status"),
            "views": views,
            "expected_parity_omission": bool(omitted_view and parity),
        }, indent=2), flush=True)

        if failures:
            for failure in failures:
                print(f"FAIL  {failure}", flush=True)
            return 1
        if omitted_view:
            print("PASS  reconstruction verified; primary view omitted on the known "
                  "campaign-path parity limitation, production untouched", flush=True)
        else:
            print("PASS  reconstruction verified and both views rebuilt, "
                  "production untouched", flush=True)
        return 0
    finally:
        os.chdir(REPOSITORY_ROOT)
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
