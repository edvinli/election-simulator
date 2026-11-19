"""Git steps of the scheduled riksdag workflow (``.github/workflows/riksdag-votes.yml``).

    python -m scripts.riksdag_votes.publish prepare --repo simulator --worktree data-branch
    python -m scripts.riksdag_votes.publish status --repo data-branch --path data/processed/riksdag_votes
    python -m scripts.riksdag_votes.publish commit --repo data-branch --path data/processed/riksdag_votes \\
        --branch data/riksdag-votes --message "..."
    python -m scripts.riksdag_votes.publish wait --repository edvinli/election-simulator
    python -m scripts.riksdag_votes.publish commit --repo website --path files/riksdag --branch master --message "..."

The workflow never writes simulator ``main``. It keeps its record of the
outputs on the append-only branch ``data/riksdag-votes`` and publishes a copy
to the website's ``master``; those are the only two (path, branch) targets
``commit`` accepts.

``prepare`` adds a git worktree of ``data/riksdag-votes`` next to the simulator
checkout: the remote branch if it exists, otherwise a new local branch at the
checked-out main commit (so the branch shares main's history and is never an
orphan). The build reads the previous outputs from, and writes the new ones
to, that worktree.

``status`` fails if anything outside the allowed path changed and prints
``changed=yes|no`` (a change to the source manifest alone is ``no``: a
re-zipped source with the same votes is not worth a commit).

``commit`` stages exactly that path, commits, and pushes with a plain (never
forced) push; a rejected push is retried once after ``git pull --rebase``.
After the push, the remote branch must point at the local commit.

``wait`` is for the website push only. It blocks until no run of the
workflows that push to website master (publication, rendering; the
prospective benchmark is included for safety) is queued or in progress -- two
consecutive quiet checks -- and until the clock is outside the protected
18:30-22:00 UTC interval. Those workflows push with a plain ``git push`` that
does not retry, so pushing while one of them is between its checkout and its
push would make *it* fail. It exits 3 (deferred) if that does not happen
before the deadline. No other workflow writes ``data/riksdag-votes``, so that
push does not wait.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, time as clock_time, timezone
from pathlib import Path
from typing import Callable, Sequence

from .constants import (
    DATA_BRANCH,
    MANIFEST_FILENAME,
    PROCESSED_DIR_RELATIVE,
    WEBSITE_BRANCH,
    WEBSITE_DIR_RELATIVE,
)

# The only places the workflow may write: path -> branch.
ALLOWED_TARGETS = {PROCESSED_DIR_RELATIVE: DATA_BRANCH, WEBSITE_DIR_RELATIVE: WEBSITE_BRANCH}
ALLOWED_PATHS = tuple(ALLOWED_TARGETS)
PROTECTED_BRANCHES = ("main",)
# Committed with the outputs when they change, never on its own.
NOT_WORTH_A_COMMIT_ALONE = (f"{PROCESSED_DIR_RELATIVE}/{MANIFEST_FILENAME}",)
# Workflows whose commits land on website master (or simulator main).
CONFLICTING_WORKFLOWS = (
    ".github/workflows/election-simulator-publication.yml",
    ".github/workflows/election-simulator-render.yml",
    ".github/workflows/prospective-benchmark-2026.yml",
)
ACTIVE_STATUSES = ("queued", "in_progress", "waiting", "requested", "pending")
PROTECTED_START = clock_time(18, 30)
PROTECTED_END = clock_time(22, 0)
EXIT_DEFERRED = 3


class PublishError(RuntimeError):
    pass


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        raise PublishError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def _under(path: str, allowed: str) -> bool:
    allowed = allowed.rstrip("/")
    return path == allowed or path.startswith(allowed + "/")


def changed_paths(repo: Path) -> list[str]:
    """Every modified, added, deleted or untracked path (renames as both sides)."""
    output = _git(repo, "status", "--porcelain", "--untracked-files=all", "-z").stdout
    entries = [entry for entry in output.split("\0") if entry]
    paths = []
    index = 0
    while index < len(entries):
        entry = entries[index]
        status, path = entry[:2], entry[3:]
        paths.append(path)
        if "R" in status or "C" in status:
            index += 1
            paths.append(entries[index])
        index += 1
    return sorted(set(paths))


def check_path(path: str) -> str:
    path = path.strip().rstrip("/")
    if path not in ALLOWED_PATHS:
        raise PublishError(f"{path!r} is not one of the allowed paths {ALLOWED_PATHS}")
    return path


def check_target(path: str, branch: str) -> tuple[str, str]:
    path = check_path(path)
    if branch in PROTECTED_BRANCHES or ALLOWED_TARGETS[path] != branch:
        raise PublishError(f"{path} may only be pushed to {ALLOWED_TARGETS[path]!r}, not {branch!r}")
    return path, branch


def status(repo: Path, path: str) -> bool:
    """Whether ``path`` changed in a way worth a commit; raises if anything
    outside ``path`` changed. A change to the source manifest alone (a
    re-zipped source with the same votes) is not worth one."""
    path = check_path(path)
    changed = changed_paths(repo)
    outside = [p for p in changed if not _under(p, path)]
    if outside:
        raise PublishError(f"unexpected changes outside {path}: {outside[:10]}")
    return any(p not in NOT_WORTH_A_COMMIT_ALONE for p in changed)


def _remote_branch_exists(repo: Path, branch: str, remote: str) -> bool:
    result = _git(repo, "ls-remote", "--exit-code", "--heads", remote, branch, check=False)
    if result.returncode not in (0, 2):
        raise PublishError(f"git ls-remote {remote} {branch} failed: {result.stderr.strip()}")
    return result.returncode == 0


def prepare_data_branch(
    repo: Path, worktree: Path, *, branch: str = DATA_BRANCH, remote: str = "origin"
) -> bool:
    """Add a worktree of ``branch`` at ``worktree``; True if the branch already
    existed on the remote, False if it was started at ``repo``'s HEAD."""
    if branch in PROTECTED_BRANCHES or branch != DATA_BRANCH:
        raise PublishError(f"refusing to prepare {branch!r}; only {DATA_BRANCH!r}")
    if _remote_branch_exists(repo, branch, remote):
        shallow = _git(repo, "rev-parse", "--is-shallow-repository").stdout.strip() == "true"
        tracking = f"refs/remotes/{remote}/{branch}"
        _git(repo, "fetch", "-q", *(["--depth=1"] if shallow else []), remote, f"+refs/heads/{branch}:{tracking}")
        _git(repo, "worktree", "add", "-q", "-B", branch, str(worktree), tracking)
        return True
    _git(repo, "worktree", "add", "-q", "-b", branch, str(worktree), "HEAD")
    return False


def commit_and_push(
    repo: Path,
    path: str,
    message: str,
    *,
    branch: str,
    push: bool = True,
    remote: str = "origin",
) -> str | None:
    """Commit exactly ``path`` and push it to ``branch`` (its one allowed
    target); returns the pushed commit, or None if nothing changed."""
    path, branch = check_target(path, branch)
    if not status(repo, path):
        return None
    _git(repo, "add", "--", path)
    staged = [p for p in _git(repo, "diff", "--cached", "--name-only", "-z").stdout.split("\0") if p]
    if any(not _under(p, path) for p in staged):
        raise PublishError(f"staged paths outside {path}: {staged[:10]}")
    if not staged:
        return None
    _git(repo, "commit", "-q", "-m", message)
    if not push:
        return _git(repo, "rev-parse", "HEAD").stdout.strip()
    ref = f"HEAD:refs/heads/{branch}"
    first = _git(repo, "push", remote, ref, check=False)
    if first.returncode != 0:
        print(f"push rejected, rebasing onto {remote}/{branch}: {first.stderr.strip()}", file=sys.stderr)
        rebase = _git(repo, "pull", "--rebase", "--no-autostash", remote, branch, check=False)
        if rebase.returncode != 0:
            _git(repo, "rebase", "--abort", check=False)
            raise PublishError(f"rebase onto {remote}/{branch} failed: {rebase.stderr.strip()}")
        _git(repo, "push", remote, ref)
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    remote_head = _git(repo, "ls-remote", remote, f"refs/heads/{branch}").stdout.split()
    if not remote_head or remote_head[0] != head:
        raise PublishError(f"{remote}/{branch} is {remote_head[:1]} after the push, expected {head}")
    return head


def in_protected_interval(now: datetime) -> bool:
    current = now.astimezone(timezone.utc).time()
    return PROTECTED_START <= current < PROTECTED_END


def gh_active_runs(repository: str) -> int | None:
    """Queued or running runs of the conflicting workflows; None if unknown."""
    total = 0
    for state in ACTIVE_STATUSES:
        result = subprocess.run(
            ["gh", "api", f"repos/{repository}/actions/runs?status={state}&per_page=100"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"gh api failed: {result.stderr.strip()}", file=sys.stderr)
            return None
        try:
            runs = json.loads(result.stdout).get("workflow_runs", [])
        except ValueError:
            return None
        total += sum(run.get("path") in CONFLICTING_WORKFLOWS for run in runs)
    return total


def wait_until_quiet(
    active_runs: Callable[[], int | None],
    *,
    timeout: float,
    interval: float = 60.0,
    quiet_checks: int = 2,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    log: Callable[[str], None] = lambda message: print(message, file=sys.stderr),
) -> bool:
    """True once ``quiet_checks`` consecutive checks see no active run outside
    the protected interval; False at the deadline. An unknown count is busy."""
    deadline = monotonic() + timeout
    quiet = 0
    while True:
        count = active_runs()
        if count == 0 and not in_protected_interval(now()):
            quiet += 1
            if quiet >= quiet_checks:
                return True
        else:
            quiet = 0
            log(f"waiting: active conflicting runs={count}, protected interval={in_protected_interval(now())}")
        if monotonic() + interval > deadline:
            return False
        sleep(interval)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--repo", type=Path, required=True, help="the simulator checkout (main)")
    prepare_parser.add_argument("--worktree", type=Path, required=True)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--repo", type=Path, required=True)
    status_parser.add_argument("--path", required=True)
    wait_parser = commands.add_parser("wait")
    wait_parser.add_argument("--repository", required=True, help="owner/name whose runs to watch")
    wait_parser.add_argument("--timeout-minutes", type=float, default=60.0)
    wait_parser.add_argument("--interval-seconds", type=float, default=60.0)
    commit_parser = commands.add_parser("commit")
    commit_parser.add_argument("--repo", type=Path, required=True)
    commit_parser.add_argument("--path", required=True)
    commit_parser.add_argument("--branch", required=True)
    commit_parser.add_argument("--message", required=True)
    args = parser.parse_args(argv)

    try:
        if args.command == "prepare":
            existed = prepare_data_branch(args.repo, args.worktree)
            print(f"data_branch_existed={'yes' if existed else 'no'}")
            return 0
        if args.command == "status":
            print(f"changed={'yes' if status(args.repo, args.path) else 'no'}")
            return 0
        if args.command == "wait":
            quiet = wait_until_quiet(
                lambda: gh_active_runs(args.repository),
                timeout=args.timeout_minutes * 60,
                interval=args.interval_seconds,
            )
            print(f"quiet={'yes' if quiet else 'no'}")
            return 0 if quiet else EXIT_DEFERRED
        commit = commit_and_push(args.repo, args.path, args.message, branch=args.branch)
        print(f"commit={commit or ''}")
        return 0
    except PublishError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
