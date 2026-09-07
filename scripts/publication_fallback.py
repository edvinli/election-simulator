"""Liveness policy for the mandatory daily publication.

GitHub's cron is best-effort.  A delayed or dropped ``0 4 * * *`` tick leaves
the public forecast showing yesterday's generation with nothing to notice it,
which is the failure this module exists to detect.  The remedy is a trigger
outside GitHub Actions that pokes the existing publication workflow; a fallback
that shares the scheduler it is compensating for is not a fallback.

Everything here is pure: the caller reads the artifacts and passes the two
publication instants in, so the decisions can be tested without a repository,
a network or a clock.  The module's own code needs nothing but the standard
library, though importing it reaches numpy through the benchmark package's
``__init__`` -- the cutoff is read from the frozen protocol module rather than
restated here, and that is worth an installed environment.  It deliberately contains no
publication logic at all -- the fallback's whole design is to invoke the one
existing workflow rather than to grow a second path into production.

Two rules live here.

``daily_publication_satisfied`` answers "has today's mandatory recalculation
already reached the public site?".  It requires *both* the source's certified
generation and the website's live pointer to be dated today, so a publication
that simulated successfully and then failed to sync still reads as unsatisfied
and is repaired rather than reported as fresh.

``benchmark_window_conflict`` reports whether an instant falls in the window
around a frozen capture slot.  Read what it is and is not.

It is *not* what protects the benchmark.  Protection has to happen before the
fallback enters or queues for ``election-simulator-production``, and this
function is called from inside the publication path -- by which point the lock
is already held and a capture may already be waiting behind it.  That job
belongs to the ``fallback_preflight`` job in the publication workflow, which
runs outside the group and asks GitHub directly whether a capture is queued or
in flight.  Late delivery is the norm rather than the exception: the 20:30Z
capture cron has been arriving 1h43m to 2h42m late, so a capture can be live
well outside any nominal window, and one queued behind another run is not
visible to a clock at all.

What this function is: the forward-looking half of that decision, and a local
safety net.  A query of current runs cannot see a capture GitHub has not
created yet but is about to, and the frozen window is exactly the interval in
which that is imminent.  The interval is derived from the protocol module, so
it cannot drift from the protocol it describes, and it is not a widened
blackout -- widening it would trade one blind spot for a suppressed recovery.
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
