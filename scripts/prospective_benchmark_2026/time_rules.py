"""Frozen wall-clock rules for the 2026 prospective benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


STOCKHOLM = ZoneInfo("Europe/Stockholm")
UTC = timezone.utc
FIRST_CAPTURE_DATE = date(2026, 9, 4)
FINAL_CAPTURE_DATE = date(2026, 9, 12)
CUTOFF_LOCAL_TIME = time(23, 30)
LATE_CAPTURE_MAX_LOCAL_DATE = timedelta(days=1)


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
