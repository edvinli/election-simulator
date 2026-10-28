"""Decide whether the publication's contract suite would only repeat Full CI.

The publish job runs ten backend and history contract modules -- ten to
eighteen minutes -- before it acquires a single poll. Those modules test code,
and code reaches main only through human pushes, each of which Full CI runs in
full. The bot's own publication and benchmark commits never start Full CI (a
`GITHUB_TOKEN` push triggers no workflow), but they write nothing except
forecast data and site artifacts. So when every commit since a green Full CI
run is a bot commit confined to those paths, the suite at HEAD is the suite
Full CI already passed, and running it again only delays the publication.

What a skip gives up is narrow, and stated so it stays a decision rather than
an accident: the suite would re-read a few bot-written files against the
current tree (one historical snapshot pinned by `test_publication_freeze`, the
poll file `test_forecast_history` backfills March dates from). The automation
validates its own inputs and outputs either way.

Everything else runs the suite: an unreadable run list, a run that is not on
main or did not succeed, a commit that is not an ancestor of HEAD, a human
commit, a bot commit touching any other path, a git error. "Verified" is
returned only when it can be shown.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

BOT_EMAIL = "41898282+github-actions[bot]@users.noreply.github.com"
FULL_CI_WORKFLOW_PATH = ".github/workflows/full.yml"

# Every path the publication, rendering and benchmark workflows commit, and
# nothing else. Deliberately narrower than `data/`: the calibration inputs
# under data/ are human-owned and read by the suite. A bot commit outside this
# list is not proof of anything, so it runs the suite rather than widening it.
BOT_OWNED_PATHS = (
    "data/README.md",
    "data/processed/poll_aggregate/",
    "data/processed/pollofpolls/",
    "data/processed/prospective_benchmark_2026/",
    "data/processed/prospective_forecasts/",
    "data/raw/pollofpolls/",
    "files/election-simulator/",
)


@dataclass(frozen=True)
class Verdict:
    verified: bool
    reason: str
    commit: str | None = None
    run_id: int | None = None


def is_bot_owned(path: str) -> bool:
    return any(
        path == owned or (owned.endswith("/") and path.startswith(owned))
        for owned in BOT_OWNED_PATHS
    )


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True,
    ).stdout


def _is_ancestor(repo: Path, commit: str, head: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", commit, head],
        capture_output=True, text=True,
    )
    if result.returncode not in (0, 1):
        raise subprocess.CalledProcessError(
            result.returncode, result.args, result.stdout, result.stderr)
    return result.returncode == 0


def _foreign_commit(repo: Path, base: str, head: str) -> str | None:
    """The first commit in ``base..head`` the bot-only rule does not cover."""

    log = _git(repo, "log", "--format=%x00%H %ae %ce", "--name-only",
               "--no-renames", "-m", f"{base}..{head}")
    for record in log.split("\0")[1:]:
        header, *paths = record.strip("\n").split("\n")
        sha, author, committer = header.split(" ")
        if author != BOT_EMAIL or committer != BOT_EMAIL:
            return f"{sha[:12]} is not a bot commit"
        outside = [p for p in paths if p and not is_bot_owned(p)]
        if outside:
            return f"{sha[:12]} touches {outside[0]}"
    return None


def _successful_main_runs(runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [
        run for run in runs
        if run.get("conclusion") == "success"
        and run.get("status") == "completed"
        and run.get("head_branch") == "main"
        and run.get("path") == FULL_CI_WORKFLOW_PATH
        and isinstance(run.get("head_sha"), str)
    ]
    return sorted(eligible, key=lambda run: run.get("created_at") or "", reverse=True)


def verify(repo: Path, runs: Iterable[dict[str, Any]], head: str = "HEAD") -> Verdict:
    """Is HEAD's code exactly what some green Full CI run on main tested?

    Tries the newest successful runs first and stops at the first that is an
    ancestor of HEAD: an older run can only be separated from HEAD by more
    commits, and the newest ancestor already carries every human change.
    """

    try:
        head_sha = _git(repo, "rev-parse", "--verify", f"{head}^{{commit}}").strip()
        for run in _successful_main_runs(runs):
            commit = run["head_sha"]
            try:
                if not _is_ancestor(repo, commit, head_sha):
                    continue
            except subprocess.CalledProcessError:
                # Not in this clone: a run for a commit force-pushed away.
                continue
            foreign = _foreign_commit(repo, commit, head_sha)
            if foreign:
                return Verdict(False, f"CODE_CHANGED_SINCE_FULL_CI: {foreign}",
                               commit, run.get("id"))
            return Verdict(True, "VERIFIED_BY_FULL_CI", commit, run.get("id"))
        return Verdict(False, "NO_SUCCESSFUL_FULL_CI_ANCESTOR")
    except (subprocess.CalledProcessError, KeyError, ValueError, OSError) as error:
        return Verdict(False, f"CHECK_FAILED: {type(error).__name__}")


def _summary_line(verdict: Verdict) -> str:
    if verdict.verified:
        return (f"Backend contract tests: SKIPPED_VERIFIED_BY_FULL_CI "
                f"(commit {verdict.commit}, run {verdict.run_id})\n")
    return f"Backend contract tests: REQUIRED ({verdict.reason})\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Write ``verified=true|false`` and ``reason=...``; never fail the step.

    An unreadable runs file is not an error to surface but a reason to run the
    suite, which is the safe outcome, so this always exits 0.
    """

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--runs-json", type=Path, required=True,
                        help="GitHub's workflow-runs response for full.yml")
    parser.add_argument("--output", type=Path, help="the step's $GITHUB_OUTPUT")
    parser.add_argument("--summary-path", type=Path)
    args = parser.parse_args(argv)

    try:
        runs = json.loads(args.runs_json.read_text(encoding="utf-8"))["workflow_runs"]
        if not isinstance(runs, list):
            raise ValueError("workflow_runs is not a list")
    except (OSError, ValueError, KeyError, TypeError) as error:
        verdict = Verdict(False, f"RUNS_UNREADABLE: {type(error).__name__}")
    else:
        verdict = verify(args.repo_root, runs)

    line = f"verified={'true' if verdict.verified else 'false'}\nreason={verdict.reason}\n"
    if args.output:
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(line)
    print(line, end="")
    if args.summary_path:
        with args.summary_path.open("a", encoding="utf-8") as handle:
            handle.write(_summary_line(verdict))
    return 0


if __name__ == "__main__":
    sys.exit(main())
