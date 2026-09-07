"""Liveness policy for the mandatory daily publication.

GitHub's cron is best-effort.  A delayed or dropped ``0 4 * * *`` tick leaves
the public forecast showing yesterday's generation with nothing to notice it,
which is the failure this module exists to detect.  The remedy is a trigger
outside GitHub Actions that pokes the existing publication workflow; a fallback
that shares the scheduler it is compensating for is not a fallback.

Everything here is pure and stdlib-only, in the manner of
``scripts.prospective_benchmark_2026.time_rules``: the caller reads the
artifacts and passes the two publication instants in, so the decisions can be
tested without a repository, a network or a clock.  It deliberately contains no
publication logic at all -- the fallback's whole design is to invoke the one
existing workflow rather than to grow a second path into production.

Two rules live here.

``daily_publication_satisfied`` answers "has today's mandatory recalculation
already reached the public site?".  It requires *both* the source's certified
generation and the website's live pointer to be dated today, so a publication
that simulated successfully and then failed to sync still reads as unsatisfied
and is repaired rather than reported as fresh.

``benchmark_window_conflict`` keeps the fallback out of the prospective
benchmark's protected window.  The benchmark shares the
``election-simulator-production`` concurrency group precisely so a capture can
never race a publication, and amendment 005 records that a capture which is not
ready before its frozen cutoff archives LATE_EXCLUDED and fails the run.  A
publication holding that lock across the cutoff could cause exactly that, so
the fallback refuses to dispatch into the window instead.  The interval is
derived from the frozen protocol module rather than restated here, and the
fallback's own firing window is hours away from it -- the guard is the
invariant, not the schedule.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from scripts.prospective_benchmark_2026.time_rules import (
    FINAL_CAPTURE_DATE,
    FIRST_CAPTURE_DATE,
    SCHEDULED_START_LOCAL_TIME,
    STOCKHOLM,
    scheduled_cutoff,
)

__all__ = [
    "CAPTURE_COMPLETION_GRACE",
    "FALLBACK_RUN_TYPE",
    "PUBLICATION_MAX_RUNTIME",
    "benchmark_protected_interval",
    "benchmark_window_conflict",
    "daily_publication_satisfied",
    "stockholm_date_of",
]

# The publication workflow's own ``timeout-minutes``.  A publication that
# started this long before the benchmark's pre-warm could still be holding the
# production lock when the capture job wants it.
PUBLICATION_MAX_RUNTIME = timedelta(minutes=120)

# Headroom after the frozen cutoff for the capture itself to finish and release
# the lock.
CAPTURE_COMPLETION_GRACE = timedelta(minutes=30)

# The run-type label a fallback dispatch carries.  It is deliberately distinct
# from MANUAL: an operator dispatch is an explicit human action and bypasses
# the repository kill switch, while this one is automated and must not.
FALLBACK_RUN_TYPE = "FALLBACK_DAILY"


def stockholm_date_of(instant: datetime) -> date:
    """The Stockholm calendar date of an aware instant."""

    if instant.tzinfo is None:
        raise ValueError("instant must include a timezone")
    return instant.astimezone(STOCKHOLM).date()


def daily_publication_satisfied(
    *,
    source_generated_at: datetime | None,
    site_generated_at: datetime | None,
    today: date,
) -> bool:
    """Has today's mandatory recalculation reached the public site?

    Both instants must be dated today.  A missing instant on either side is
    unsatisfied: an unreadable pointer is treated as stale rather than as
    fresh, so the failure mode of the check itself is a redundant publication
    and never a silently stale forecast.

    Note what this does *not* claim.  No published artifact records which cron
    produced it, so this cannot attest that the DAILY run specifically ran --
    only that a full publication carrying today's date is live.  That is the
    property the public page depends on, and the mandatory-recalculation
    semantics are preserved on the other side: a fallback dispatch publishes
    unconditionally rather than behaving as a poll-change check.
    """

    if source_generated_at is None or site_generated_at is None:
        return False
    return (
        stockholm_date_of(source_generated_at) == today
        and stockholm_date_of(site_generated_at) == today
    )


def benchmark_protected_interval(scheduled_date: date | str) -> tuple[datetime, datetime]:
    """The interval in which a publication must not hold the production lock.

    From ``PUBLICATION_MAX_RUNTIME`` before the benchmark's pre-warm start
    until ``CAPTURE_COMPLETION_GRACE`` after its frozen cutoff.  The cutoff
    comes from the frozen protocol module, so this cannot drift from the
    protocol it protects.
    """

    day = scheduled_date if isinstance(scheduled_date, date) else date.fromisoformat(str(scheduled_date))
    start_local = datetime.combine(day, SCHEDULED_START_LOCAL_TIME, tzinfo=STOCKHOLM)
    return (
        start_local - PUBLICATION_MAX_RUNTIME,
        scheduled_cutoff(day) + CAPTURE_COMPLETION_GRACE,
    )


def benchmark_window_conflict(now: datetime) -> date | None:
    """The benchmark slot a dispatch at ``now`` could push past its cutoff.

    ``None`` when there is no conflict.  Only the frozen capture dates are
    considered; outside that window the benchmark holds no lock.
    """

    if now.tzinfo is None:
        raise ValueError("now must include a timezone")
    # A protected interval can begin on the previous Stockholm date, so check
    # the local date and the one before it.
    local_date = stockholm_date_of(now)
    for candidate in (local_date, local_date - timedelta(days=1)):
        if candidate < FIRST_CAPTURE_DATE or candidate > FINAL_CAPTURE_DATE:
            continue
        start, end = benchmark_protected_interval(candidate)
        if start <= now <= end:
            return candidate
    return None
