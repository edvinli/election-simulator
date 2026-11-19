"""Git steps of the scheduled riksdag workflow (``.github/workflows/riksdag-votes.yml``).

    python -m scripts.riksdag_votes.publish status --repo simulator --path data/processed/riksdag_votes
    python -m scripts.riksdag_votes.publish wait --repository edvinli/election-simulator
    python -m scripts.riksdag_votes.publish commit --repo simulator --path data/processed/riksdag_votes \\
        --branch main --message "..."

``status`` fails if anything outside the allowed path changed and prints
``changed=yes|no``. ``commit`` stages exactly that path, commits, and pushes
with a plain (never forced) push; a rejected push is retried once after
``git pull --rebase``. The commit touches only riksdag paths, so the rebase
cannot conflict with the publication or rendering commits that may have landed
in between. After the push, the remote branch must point at the local commit.

``wait`` blocks until no run of the workflows that push to the same branches
(publication, rendering, the prospective benchmark) is queued or in progress
-- two consecutive quiet checks -- and until the clock is outside the
protected 18:30-22:00 UTC interval. Those workflows push with a plain
``git push`` that does not retry, so pushing while one of them is between its
checkout and its push would make *it* fail. It exits 3 (deferred) if that does
not happen before the deadline.
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

from .constants import PROCESSED_DIR_RELATIVE, WEBSITE_DIR_RELATIVE

ALLOWED_PATHS = (PROCESSED_DIR_RELATIVE, WEBSITE_DIR_RELATIVE)
# Workflows whose commits land on simulator main or website master.
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


def status(repo: Path, path: str) -> bool:
    """Whether ``path`` changed; raises if anything else in ``repo`` did."""
    path = check_path(path)
    changed = changed_paths(repo)
    outside = [p for p in changed if not _under(p, path)]
    if outside:
        raise PublishError(f"unexpected changes outside {path}: {outside[:10]}")
    return bool(changed)


def commit_and_push(
    repo: Path,
    path: str,
    message: str,
    *,
    branch: str,
    push: bool = True,
    remote: str = "origin",
) -> str | None:
    """Commit exactly ``path`` and push it; returns the pushed commit or None if nothing changed."""
    if not status(repo, path):
        return None
    path = check_path(path)
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
