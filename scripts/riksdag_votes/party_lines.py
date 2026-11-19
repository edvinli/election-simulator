"""Party lines (rule 2) and votes against the line (rule 4).

A party's line on a vote is the plurality position among its members who
voted Ja, Nej or Avstår. Avstår is a stance, not an absence; Frånvarande is
excluded. A tie for the plurality gives no line (``line=None, tie=True``); a
party none of whose members cast a vote has no line (``line=None,
no_cast=True``). Independents (parti ``-``) form no party and have no line.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .constants import AVSTAR, CAST_VALUES, FRANVARANDE, INDEPENDENT_PARTY, JA, NEJ, PARTY_ORDER


@dataclass(frozen=True)
class PartyLine:
    line: str | None  # "Ja", "Nej", "Avstår" or None
    tie: bool
    no_cast: bool
    counts: tuple[int, int, int, int]  # Ja, Nej, Avstår, Frånvarande

    @property
    def cast(self) -> int:
        return self.counts[0] + self.counts[1] + self.counts[2]


def party_line(rosts: Iterable[str]) -> PartyLine:
    tally = Counter(rosts)
    counts = (tally.get(JA, 0), tally.get(NEJ, 0), tally.get(AVSTAR, 0), tally.get(FRANVARANDE, 0))
    cast = {rost: tally.get(rost, 0) for rost in CAST_VALUES}
    best = max(cast.values())
    if best == 0:
        return PartyLine(None, False, True, counts)
    leaders = [rost for rost, count in cast.items() if count == best]
    if len(leaders) > 1:
        return PartyLine(None, True, False, counts)
    return PartyLine(leaders[0], False, False, counts)


def party_lines_for_vote(vote) -> dict[str, PartyLine]:
    by_party: dict[str, list[str]] = {}
    for ballot in vote.ballots:
        if ballot.parti == INDEPENDENT_PARTY:
            continue
        by_party.setdefault(ballot.parti, []).append(ballot.rost)
    return {party: party_line(rosts) for party, rosts in by_party.items()}


def order_parties(parties: Iterable[str]) -> list[str]:
    present = {party for party in parties if party != INDEPENDENT_PARTY}
    known = [party for party in PARTY_ORDER if party in present]
    return known + sorted(present - set(PARTY_ORDER))


def against_line(ballot, lines: dict[str, PartyLine]) -> bool | None:
    """True/False when the ballot can be compared with a party line, else None.

    Comparable means: the member cast Ja/Nej/Avstår, belongs to a party on
    that vote, and the party has a (non-null) line.
    """
    if ballot.rost not in CAST_VALUES or ballot.parti == INDEPENDENT_PARTY:
        return None
    line = lines.get(ballot.parti)
    if line is None or line.line is None:
        return None
    return ballot.rost != line.line
