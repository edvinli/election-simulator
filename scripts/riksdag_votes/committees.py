"""Committee (utskott) attribution from a vote's ``beteckning``.

``FiU1`` -> ``FiU``, ``UFöU3`` -> ``UFöU``. Numeric designations such as
``0311-1`` (chamber votes on a question, e.g. electing the prime minister)
belong to no committee. A letter prefix that is not a known utskott is logged
and treated as no committee rather than guessed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .constants import COMMITTEES, OTHER_COMMITTEE

LOGGER = logging.getLogger(__name__)

_COMMITTEE_PATTERN = re.compile(r"^(?P<prefix>[A-Za-zÅÄÖåäö]+)(?P<number>\d+)$")
_NUMERIC_PATTERN = re.compile(r"^\d+(-\d+)?$")


@dataclass(frozen=True)
class CommitteeParse:
    committee: str | None  # official abbreviation, or None
    kind: str  # "committee", "numeric", "unknown_prefix", "missing", "unparsed"
    prefix: str | None = None


def parse_committee(beteckning: str | None) -> CommitteeParse:
    text = (beteckning or "").strip()
    if not text:
        return CommitteeParse(None, "missing")
    if _NUMERIC_PATTERN.match(text):
        return CommitteeParse(None, "numeric")
    match = _COMMITTEE_PATTERN.match(text)
    if not match:
        LOGGER.warning("unparsed beteckning %r: no committee assigned", text)
        return CommitteeParse(None, "unparsed", None)
    prefix = match.group("prefix")
    if prefix in COMMITTEES:
        return CommitteeParse(prefix, "committee", prefix)
    LOGGER.warning("unknown committee prefix %r in beteckning %r", prefix, text)
    return CommitteeParse(None, "unknown_prefix", prefix)


def committee_key(committee: str | None) -> str:
    """Grouping key for per-committee tables: the abbreviation or ``övrigt``."""
    return committee if committee is not None else OTHER_COMMITTEE
