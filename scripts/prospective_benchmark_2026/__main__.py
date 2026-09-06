"""Command-line entry points for the 2026 prospective benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from .archive import DEFAULT_ARCHIVE_ROOT, slot_timing_outcome, validate_archive
from .capture import DEFAULT_ES_ARCHIVE, REPOSITORY_ROOT, run_capture
from .report import DEFAULT_OUTPUT_DIR, write_report


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _write_summary(path: str | None, value: Any) -> None:
    if not path:
        return
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write("### Prospective benchmark command\n\n")
        handle.write("```json\n")
        handle.write(_json_text(value))
        handle.write("```\n")


def _capture_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "capture",
        help="capture one frozen scheduled slot (or perform a non-durable dry run)",
    )
    parser.add_argument("--mode", choices=("dry_run", "capture"), required=True)
    parser.add_argument("--scheduled-date", required=True, help="YYYY-MM-DD in Europe/Stockholm")
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--es-archive-root", type=Path, default=DEFAULT_ES_ARCHIVE)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--summary-path", help="optional GitHub Actions summary file")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.prospective_benchmark_2026")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _capture_parser(subparsers)

    validate = subparsers.add_parser("validate", help="validate the complete immutable archive")
    validate.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)

    eligible = subparsers.add_parser(
        "assert-eligible",
        help="fail unless a scheduled slot recorded a timing-eligible durable capture",
    )
    eligible.add_argument("--scheduled-date", required=True, help="YYYY-MM-DD in Europe/Stockholm")
    eligible.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    eligible.add_argument("--summary-path", help="optional GitHub Actions summary file")
    eligible.add_argument(
        "--warn-only",
        action="store_true",
        help="report the outcome without failing (manual dispatch, not an expected slot)",
    )

    score = subparsers.add_parser("score", help="score immutable captures against a final result manifest")
    score.add_argument("--results", type=Path, required=True, help="FINAL_CERTIFIED Valmyndigheten manifest")
    score.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    score.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "capture":
        result = run_capture(
            scheduled_date=args.scheduled_date,
            mode=args.mode,
            archive_root=args.archive_root,
            es_archive_root=args.es_archive_root,
            repo_root=args.repo_root,
        )
        _write_summary(args.summary_path, result)
        print(_json_text(result), end="")
        return 0
    if args.command == "assert-eligible":
        outcome = slot_timing_outcome(args.archive_root, args.scheduled_date)
        _write_summary(args.summary_path, outcome)
        print(_json_text(outcome), end="")
        if outcome["timing_eligible"]:
            return 0
        # An append that is not timing-eligible is a real operational failure
        # even though the archive is intact, so it is reported loudly and the
        # two cases are named differently rather than collapsed into "failed".
        message = (
            f"Scheduled slot {outcome['scheduled_date']} produced no timing-eligible capture "
            f"(outcome {outcome['outcome']}, timing status {outcome['timing_status']})."
        )
        print(f"::error::{message}", file=sys.stderr)
        if args.summary_path:
            with Path(args.summary_path).open("a", encoding="utf-8") as handle:
                handle.write(f"\n> [!CAUTION]\n> {message}\n")
        return 0 if args.warn_only else 4
    if args.command == "validate":
        result = validate_archive(args.archive_root)
        print(_json_text(result), end="")
        return 0
    if args.command == "score":
        json_path, markdown_path, result = write_report(
            archive_root=args.archive_root,
            result_manifest=args.results,
            output_dir=args.output_dir,
        )
        summary = {
            "status": "SCORED",
            "json_report": str(json_path),
            "markdown_report": str(markdown_path),
            "final_probabilistic_winner": result.get("final_probabilistic_winner"),
            "final_point_winner": result.get("final_point_winner"),
        }
        print(_json_text(summary), end="")
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f"prospective benchmark: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
