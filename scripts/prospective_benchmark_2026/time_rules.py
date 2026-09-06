"""Frozen wall-clock rules for the 2026 prospective benchmark.

Deliberately stdlib-only, and deliberately runnable as a file:

    python scripts/prospective_benchmark_2026/time_rules.py resolve-slot <iso>

The workflow's slot guard runs before any dependency install, so it cannot
import the package (whose ``__init__`` pulls in numpy through the archive and
scoring modules). Invoking this module by path skips the package import
entirely, which lets the guard and the capture job share one implementation of
the scheduling rules instead of reimplementing them in shell.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import sys
import time as _time
from zoneinfo import ZoneInfo


STOCKHOLM = ZoneInfo("Europe/Stockholm")
UTC = timezone.utc
FIRST_CAPTURE_DATE = date(2026, 9, 4)
FINAL_CAPTURE_DATE = date(2026, 9, 12)
CUTOFF_LOCAL_TIME = time(23, 30)
LATE_CAPTURE_MAX_LOCAL_DATE = timedelta(days=1)

# The scheduled job starts an hour before the cutoff so that checkout,
# dependency install and the capture gates finish while retrieval is still
# prohibited. The cutoff itself is frozen and unchanged: the job waits for it.
SCHEDULED_START_LOCAL_TIME = time(22, 30)
# The scheduled start is one hour before the cutoff, so a legitimate wait never
# exceeds that by much. A longer one means the slot was resolved wrongly: fail
# closed rather than idle a runner and capture at an unintended time. The bound
# also stays below the job's own timeout, so a waiting job is never killed
# mid-wait and left looking like an infrastructure failure.
MAX_CUTOFF_WAIT = timedelta(minutes=90)


class CaptureTimeError(ValueError):
    """Raised when a durable capture would violate the frozen schedule."""


@dataclass(frozen=True)
class CaptureTiming:
    scheduled_date: date
    cutoff_local: datetime
    cutoff_utc: datetime
    retrieved_at_utc: datetime
    retrieved_at_local: datetime
    status: str
    eligible: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "scheduled_date": self.scheduled_date.isoformat(),
            "benchmark_cutoff": self.cutoff_utc.isoformat().replace("+00:00", "Z"),
            "benchmark_cutoff_europe_stockholm": self.cutoff_local.isoformat(),
            "retrieved_at_utc": self.retrieved_at_utc.isoformat().replace("+00:00", "Z"),
            "retrieved_at_europe_stockholm": self.retrieved_at_local.isoformat(),
            "timing_status": self.status,
            "timing_eligible": self.eligible,
        }


def parse_aware_datetime(value: str | datetime) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CaptureTimeError("Timestamp must include an explicit timezone")
    return parsed


def scheduled_cutoff(scheduled_date: date | str) -> datetime:
    day = scheduled_date if isinstance(scheduled_date, date) else date.fromisoformat(scheduled_date)
    if not FIRST_CAPTURE_DATE <= day <= FINAL_CAPTURE_DATE:
        raise CaptureTimeError(
            f"Scheduled date {day} is outside {FIRST_CAPTURE_DATE} through {FINAL_CAPTURE_DATE}"
        )
    return datetime.combine(day, CUTOFF_LOCAL_TIME, tzinfo=STOCKHOLM)


def classify_capture_time(
    scheduled_date: date | str,
    retrieved_at: str | datetime,
    *,
    durable: bool,
) -> CaptureTiming:
    cutoff_local = scheduled_cutoff(scheduled_date)
    retrieved_utc = parse_aware_datetime(retrieved_at).astimezone(UTC)
    retrieved_local = retrieved_utc.astimezone(STOCKHOLM)
    if retrieved_utc < cutoff_local.astimezone(UTC):
        if durable:
            raise CaptureTimeError("A durable real capture cannot be created before its scheduled cutoff")
        status, eligible = "DRY_RUN_BEFORE_CUTOFF", False
    elif retrieved_local.date() == cutoff_local.date():
        status, eligible = "ON_TIME_ELIGIBLE", True
    elif retrieved_local.date() == cutoff_local.date() + LATE_CAPTURE_MAX_LOCAL_DATE:
        # Amendment 001 permits retaining a delayed scheduled run only on the
        # immediately following Stockholm calendar date.  It remains excluded
        # from primary scoring, but is useful evidence that the run was late.
        status, eligible = "LATE_EXCLUDED", False
    else:
        # A later run cannot honestly represent the missing historical cutoff.
        # Dry-run callers receive a diagnostic status; durable callers fail
        # before any source evidence is written or indexed.
        status, eligible = "RETROACTIVE_PROHIBITED", False
        if durable:
            raise CaptureTimeError(
                "A durable capture may be late only on the immediately following "
                "Europe/Stockholm calendar date; later captures are retroactive"
            )
    return CaptureTiming(
        scheduled_date=cutoff_local.date(),
        cutoff_local=cutoff_local,
        cutoff_utc=cutoff_local.astimezone(UTC),
        retrieved_at_utc=retrieved_utc,
        retrieved_at_local=retrieved_local,
        status=status,
        eligible=eligible,
    )


def capture_id_for_date(scheduled_date: date | str) -> str:
    cutoff = scheduled_cutoff(scheduled_date).astimezone(UTC)
    return cutoff.strftime("%Y%m%dT%H%M%SZ")


def resolve_scheduled_slot(run_created_at: str | datetime) -> date:
    """Return the Stockholm slot a scheduled workflow run belongs to.

    ``run_created_at`` must be GitHub's externally recorded run creation time,
    not the runner's wall clock: the runner may start well after the cron
    instant, and the point of this rule is to survive that delay.

    The job is scheduled at :data:`SCHEDULED_START_LOCAL_TIME`, so a run
    created at or after that local time belongs to that Stockholm date -- and
    still does when the scheduler delays it past the 23:30 cutoff, which is
    exactly the case a boundary at the cutoff got wrong. A run created before
    the start time has been delayed across local midnight and belongs to the
    previous date, whose capture will be durably LATE_EXCLUDED.

    The rule deliberately never invents a slot in the future: a run cannot be
    attributed to a date whose cutoff has not yet arrived.
    """

    created_local = parse_aware_datetime(run_created_at).astimezone(STOCKHOLM)
    if created_local.time() >= SCHEDULED_START_LOCAL_TIME:
        return created_local.date()
    return created_local.date() - timedelta(days=1)


def seconds_until_cutoff(
    scheduled_date: date | str,
    now: str | datetime | None = None,
) -> float:
    """Seconds to wait before the frozen cutoff for ``scheduled_date``.

    Zero once the cutoff has passed, so a delayed run captures immediately
    instead of waiting for a boundary that is already behind it.
    """

    cutoff = scheduled_cutoff(scheduled_date).astimezone(UTC)
    moment = datetime.now(UTC) if now is None else parse_aware_datetime(now).astimezone(UTC)
    remaining = (cutoff - moment).total_seconds()
    if remaining <= 0:
        return 0.0
    if remaining > MAX_CUTOFF_WAIT.total_seconds():
        raise CaptureTimeError(
            f"Cutoff for {scheduled_date} is {remaining:.0f}s away, beyond the "
            f"{MAX_CUTOFF_WAIT.total_seconds():.0f}s bound; the slot is resolved wrongly"
        )
    return remaining


def wait_for_cutoff(
    scheduled_date: date | str,
    *,
    now: str | datetime | None = None,
    sleep: object = None,
) -> float:
    """Block until the frozen cutoff, and report how long that took."""

    remaining = seconds_until_cutoff(scheduled_date, now)
    if remaining > 0:
        (sleep or _time.sleep)(remaining)  # type: ignore[operator]
    return remaining


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/prospective_benchmark_2026/time_rules.py",
        description="Frozen scheduling rules, callable without installing dependencies.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    resolve = sub.add_parser("resolve-slot", help="Stockholm slot for a scheduled run creation time")
    resolve.add_argument("--run-created-at", required=True)
    wait = sub.add_parser("wait-for-cutoff", help="block until the frozen cutoff for a slot")
    wait.add_argument("--scheduled-date", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "resolve-slot":
            print(resolve_scheduled_slot(args.run_created_at).isoformat())
            return 0
        waited = wait_for_cutoff(args.scheduled_date)
        print(f"waited {waited:.0f}s for the {args.scheduled_date} cutoff")
        return 0
    except CaptureTimeError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_main())
