"""Decide which unit tests a change needs, and split a run into balanced shards.

Two jobs, one module, because both need the same picture of the suite.

`select` answers "what does this diff affect?" for the pull-request layer. It
walks the import graph rather than a hand-written path table, because a table
goes stale silently and the failure mode is a test that stopped running without
anyone noticing. The graph is built from `scripts/**` and `tests/**` and closed
transitively: a change to `scripts/simulator/config.py` selects every test that
reaches it through any chain of imports, not only the ones naming it directly.

Import analysis alone is not sufficient, and the gaps are handled explicitly
rather than hoped away:

  * Seven modules (the freeze and reference-math tests) import nothing from
    `scripts` at all -- they assert against tracked artifacts. Nothing in a
    diff of `scripts/**` would ever select them, so they are always run. They
    cost about seven seconds together.
  * Changes that can invalidate any test -- dependency pins, fixtures, tracked
    data, the test harness, this selector, the workflows -- escalate to the
    full suite instead of being resolved through the graph.
  * A path the rules do not recognise also escalates. The default is the whole
    suite, so an unmapped change over-runs rather than under-runs.

`shard` splits a module list into N groups of comparable wall-clock using
measured durations from `timings.json`, longest-first. Sharding by name or by
file count puts the 519-second allocator audit in one bin and leaves the rest
idle; sharding by measured cost keeps the bins even.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TIMINGS_PATH = Path(__file__).with_name("timings.json")

# Tests that assert against tracked artifacts instead of importing the code
# that produced them. The import graph cannot see the dependency, so these run
# on every change.
ALWAYS_RUN = (
    "test_challenger_b",
    "test_challenger_freeze",
    "test_challenger_reference_math",
    "test_competition_gates",
    "test_evaluator_freeze_reconstructible",
    "test_production_freeze",
    "test_publication_freeze",
)

# Modules whose full form is too expensive for the per-change layers and runs
# on the nightly schedule instead. Nothing here may be the only cover for a
# property: test_adversarial_mandates keeps proving allocator parity on every
# change through ELECTIONSIM_ADVERSARIAL_CASES, and only the exhaustive
# 20,000-case sweep is deferred. See docs/ci-topology.md.
NIGHTLY_ONLY = ("test_adversarial_mandates",)

# Path prefixes whose change can invalidate any test in the suite.
FULL_RUN_PREFIXES = (
    "pyproject.toml",
    "uv.lock",
    "Makefile",
    "data/",
    "files/",
    "tests/fixtures/",
    "tests/__init__.py",
    "tests/_website_repo.py",
    "scripts/ci/",
    ".github/workflows/",
)

# Paths that cannot affect the Python unit suite.
IGNORED_PREFIXES = (
    "docs/",
    "diagnostics/",
    "README.md",
    ".gitignore",
)


def _module_name(path: Path) -> str | None:
    """Map a repository-relative .py path to its dotted module name."""
    if path.suffix != ".py":
        return None
    parts = list(path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) if parts else None


def _imports_of(path: Path) -> set[str]:
    """Return the first-party modules a file imports.

    Relative imports are resolved against the file's own package so that
    `from .config import X` inside `scripts/simulator/` records
    `scripts.simulator.config`.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return set()

    package = list(path.parent.parts)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - node.level + 1]
                target = base + ([node.module] if node.module else [])
                found.add(".".join(target))
                # `from . import sibling` names the members, not the module.
                for alias in node.names:
                    found.add(".".join(target + [alias.name]))
            elif node.module:
                found.add(node.module)
                for alias in node.names:
                    found.add(f"{node.module}.{alias.name}")
    return {m for m in found if m.startswith(("scripts", "tests"))}


def build_graph(repo_root: Path) -> dict[str, set[str]]:
    """Map each first-party module to the first-party modules it imports."""
    graph: dict[str, set[str]] = {}
    for directory in ("scripts", "tests"):
        for path in sorted((repo_root / directory).rglob("*.py")):
            rel = path.relative_to(repo_root)
            name = _module_name(rel)
            if name:
                graph[name] = _imports_of(path)
    return graph


def _reverse_closure(graph: dict[str, set[str]], seeds: set[str]) -> set[str]:
    """Every module that reaches any seed through a chain of imports."""
    # `a -> b` in the graph means a imports b, so dependents are found by
    # walking the reversed edges outward from the changed modules.
    reverse: dict[str, set[str]] = {}
    for importer, imported in graph.items():
        for target in imported:
            reverse.setdefault(target, set()).add(importer)

    seen = set(seeds)
    queue = list(seeds)
    while queue:
        current = queue.pop()
        for dependent in reverse.get(current, ()):
            if dependent not in seen:
                seen.add(dependent)
                queue.append(dependent)
    return seen


def all_test_modules(repo_root: Path = _REPO_ROOT, *, tier: str = "all") -> list[str]:
    """Every unit test module, by bare name.

    `tier` selects which side of the nightly split to return: "per-change"
    drops the modules deferred to nightly, "nightly" returns only those, and
    "all" returns everything.
    """
    modules = sorted(p.stem for p in (repo_root / "tests").glob("test_*.py"))
    if tier == "per-change":
        return [m for m in modules if m not in NIGHTLY_ONLY]
    if tier == "nightly":
        return [m for m in modules if m in NIGHTLY_ONLY]
    if tier != "all":
        raise ValueError(f"unknown tier {tier!r}")
    return modules


def select(
    changed: list[str],
    repo_root: Path = _REPO_ROOT,
    *,
    tier: str = "all",
) -> tuple[list[str], str]:
    """Choose the test modules a change requires.

    Returns the module names and a one-line reason, so CI can print why it ran
    what it ran. `tier` is passed through to `all_test_modules`, so the
    pull-request layer can ask for the per-change tier and have the nightly
    modules dropped after selection rather than by a second filter it might
    forget to apply.
    """
    everything = all_test_modules(repo_root, tier=tier)
    considered = [p for p in changed if not p.startswith(IGNORED_PREFIXES)]
    if not considered:
        return [], "no path in the diff can affect the Python unit suite"

    for path in considered:
        if path.startswith(FULL_RUN_PREFIXES):
            return everything, f"full suite: {path} can invalidate any test"

    unmapped = [
        p for p in considered
        if not (p.startswith("scripts/") or p.startswith("tests/")) or not p.endswith(".py")
    ]
    if unmapped:
        return everything, f"full suite: unmapped path {unmapped[0]}"

    graph = build_graph(repo_root)
    seeds = set()
    for path in considered:
        name = _module_name(Path(path))
        if name:
            seeds.add(name)

    reached = _reverse_closure(graph, seeds)
    selected = {m.split(".")[-1] for m in reached if m.startswith("tests.")}
    # A changed test module runs even when nothing imports it.
    selected |= {Path(p).stem for p in considered if p.startswith("tests/")}
    selected |= set(ALWAYS_RUN)
    selected &= set(everything)

    ordered = sorted(selected)
    if len(ordered) == len(everything):
        return ordered, "full suite: the change reaches every test"
    return ordered, f"{len(ordered)} of {len(everything)} modules affected"


def _timings() -> dict[str, float]:
    try:
        return json.loads(_TIMINGS_PATH.read_text(encoding="utf-8"))["seconds"]
    except (OSError, KeyError, json.JSONDecodeError):
        return {}


def shard(modules: list[str], count: int) -> list[list[str]]:
    """Split modules into `count` bins of comparable measured duration."""
    if count < 1:
        raise ValueError("shard count must be at least 1")
    timings = _timings()
    default = (sum(timings.values()) / len(timings)) if timings else 1.0
    # Longest-processing-time first: the classic greedy makespan heuristic, and
    # good enough here because one module dominates the distribution.
    ordered = sorted(modules, key=lambda m: (-timings.get(m, default), m))
    bins: list[list[str]] = [[] for _ in range(count)]
    loads = [0.0] * count
    for module in ordered:
        target = loads.index(min(loads))
        bins[target].append(module)
        loads[target] += timings.get(module, default)
    return [sorted(b) for b in bins]


def _changed_from_git(base: str) -> list[str]:
    merge_base = subprocess.run(
        ["git", "merge-base", base, "HEAD"],
        capture_output=True, text=True, cwd=_REPO_ROOT,
    )
    ref = merge_base.stdout.strip() or base
    diff = subprocess.run(
        ["git", "diff", "--name-only", f"{ref}...HEAD"],
        capture_output=True, text=True, check=True, cwd=_REPO_ROOT,
    )
    return [line for line in diff.stdout.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_select = sub.add_parser("select", help="print the test modules a diff affects")
    source = p_select.add_mutually_exclusive_group(required=True)
    source.add_argument("--base", help="git ref to diff HEAD against")
    source.add_argument("--changed", nargs="*", help="explicit changed paths")
    p_select.add_argument("--shards", type=int, default=1)
    p_select.add_argument("--shard-index", type=int)
    p_select.add_argument("--format", choices=("lines", "unittest", "json"), default="lines")
    p_select.add_argument("--tier", choices=("all", "per-change", "nightly"), default="all")

    p_shard = sub.add_parser("shard", help="split the whole suite into balanced shards")
    p_shard.add_argument("--shards", type=int, required=True)
    p_shard.add_argument("--shard-index", type=int, required=True)
    p_shard.add_argument("--format", choices=("lines", "unittest", "json"), default="lines")
    p_shard.add_argument("--tier", choices=("all", "per-change", "nightly"), default="all")

    p_matrix = sub.add_parser(
        "matrix",
        help="emit a GitHub Actions matrix of shards with their modules baked in",
    )
    m_source = p_matrix.add_mutually_exclusive_group(required=True)
    m_source.add_argument("--base", help="git ref to diff HEAD against")
    m_source.add_argument("--changed", nargs="*", help="explicit changed paths")
    m_source.add_argument("--all", action="store_true", help="use the whole tier")
    p_matrix.add_argument("--shards", type=int, required=True)
    p_matrix.add_argument("--tier", choices=("all", "per-change", "nightly"), default="all")

    p_plan = sub.add_parser("plan", help="describe the shard plan for humans")
    p_plan.add_argument("--shards", type=int, required=True)
    p_plan.add_argument("--tier", choices=("all", "per-change", "nightly"), default="all")

    args = parser.parse_args(argv)

    if args.command == "plan":
        timings = _timings()
        bins = shard(all_test_modules(_REPO_ROOT, tier=args.tier), args.shards)
        for index, group in enumerate(bins):
            cost = sum(timings.get(m, 0.0) for m in group)
            print(f"shard {index}: {len(group):2d} modules, {cost:7.1f}s")
            for module in group:
                print(f"    {timings.get(module, 0.0):7.1f}s  {module}")
        return 0

    if args.command == "matrix":
        if args.all:
            modules = all_test_modules(_REPO_ROOT, tier=args.tier)
            reason = f"{args.tier} tier, all modules"
        else:
            changed = args.changed if args.changed is not None else _changed_from_git(args.base)
            modules, reason = select(changed, tier=args.tier)
        # Never open more shards than there is work for; an empty shard is a
        # runner spun up to do nothing.
        count = max(1, min(args.shards, len(modules)))
        groups = shard(modules, count) if modules else []
        include = [
            {
                "shard": index,
                "modules": " ".join(f"tests.{m}" for m in group),
                "count": len(group),
            }
            for index, group in enumerate(groups)
            if group
        ]
        print(json.dumps({"include": include}))
        print(f"selection: {reason}", file=sys.stderr)
        print(f"shards: {len(include)}", file=sys.stderr)
        return 0

    if args.command == "select":
        changed = args.changed if args.changed is not None else _changed_from_git(args.base)
        modules, reason = select(changed, tier=args.tier)
        print(f"selection: {reason}", file=sys.stderr)
    else:
        modules = all_test_modules(_REPO_ROOT, tier=args.tier)
        reason = f"{args.tier} tier"

    if getattr(args, "shard_index", None) is not None and args.shards > 1:
        modules = shard(modules, args.shards)[args.shard_index]

    if args.format == "json":
        print(json.dumps(modules))
    elif args.format == "unittest":
        print(" ".join(f"tests.{m}" for m in modules))
    else:
        for module in modules:
            print(module)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
