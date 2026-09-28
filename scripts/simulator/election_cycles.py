"""Which election a forecast targets, and what that implies.

Every date and year here is derived from the target election rather than
stated as a 2026 constant, so the same code serves the 2026 cycle -- whose
outputs stay bit-identical -- and the 2030 cycle.

This module sits outside every implementation freeze on purpose. The frozen
modules (the projection, the allocators, the national engines) are unchanged;
what changes between cycles is which inputs they are handed.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Mapping

from scripts.mandates.config import FIXED_SEATS_2018, FIXED_SEATS_2022, FIXED_SEATS_2026


#: Ordinary Riksdag elections are held every fourth year.
ELECTION_INTERVAL_YEARS = 4

#: From the 2014 election the ordinary election day is the second Sunday in
#: September; before that it was the third.
SECOND_SUNDAY_FROM_YEAR = 2014

#: The longest horizon the dynamics layer has empirical support for. Beyond it
#: the model uses the capped 112-day movement rather than the full remaining
#: time; the history schedule turns daily from that day.
DYNAMICS_CAP_DAYS = 112

#: The first reconstructed history point is a week after the previous
#: election. For the 2026 cycle this is 2022-09-18, the published start.
HISTORY_START_OFFSET_DAYS = 7

#: The official fixed-seat distribution per target election. 2030 is a
#: declared stand-in: Valmyndigheten decides the 2030 distribution in the
#: spring of 2030, and until then the 2026 distribution is used.
FIXED_SEATS_BY_ELECTION_YEAR: Mapping[int, Mapping[str, int]] = {
    2018: FIXED_SEATS_2018,
    2022: FIXED_SEATS_2022,
    2026: FIXED_SEATS_2026,
    2030: FIXED_SEATS_2026,
}
FIXED_SEATS_STAND_IN_YEARS: frozenset[int] = frozenset({2030})

#: The last target year with a declared seat distribution. Anything later has
#: to be added here deliberately rather than silently reuse an older one.
LAST_SUPPORTED_ELECTION_YEAR = max(FIXED_SEATS_BY_ELECTION_YEAR)


def ordinary_election_date(year: int) -> date:
    """The ordinary Riksdag election day in ``year``."""

    first = date(year, 9, 1)
    first_sunday = first + timedelta(days=(6 - first.weekday()) % 7)
    return first_sunday + timedelta(days=7 if year >= SECOND_SUNDAY_FROM_YEAR else 14)


def _as_date(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def previous_election_date(election_date: str | date) -> date:
    """The ordinary election before ``election_date``."""

    return ordinary_election_date(_as_date(election_date).year - ELECTION_INTERVAL_YEARS)


def geography_baseline_year_for(election_year: int) -> int:
    """The constituency baseline for a target: the previous ordinary election.

    A forecast never uses its own election's constituency result as the
    baseline; that would be the target leaking into its own prediction.
    """

    return election_year - ELECTION_INTERVAL_YEARS


def fixed_seats_for(election_year: int) -> Mapping[str, int]:
    """The fixed seats per constituency the allocator uses for a target year.

    Years before 2018 keep the historical behaviour of the engine, which fell
    back to the 2026 distribution for them. Years after the last declared
    election raise instead of silently reusing a stale distribution.
    """

    if election_year in FIXED_SEATS_BY_ELECTION_YEAR:
        return FIXED_SEATS_BY_ELECTION_YEAR[election_year]
    if election_year < min(FIXED_SEATS_BY_ELECTION_YEAR):
        return FIXED_SEATS_2026
    raise ValueError(
        f"No fixed-seat distribution is declared for the {election_year} election; "
        "add it to FIXED_SEATS_BY_ELECTION_YEAR")


def history_start_for(election_date: str | date) -> date:
    """The first reconstructed history point for a target election."""

    return previous_election_date(election_date) + timedelta(days=HISTORY_START_OFFSET_DAYS)


def dynamics_cap_start_for(election_date: str | date) -> date:
    """The first day on which the full remaining horizon is modelled."""

    return _as_date(election_date) - timedelta(days=DYNAMICS_CAP_DAYS)
