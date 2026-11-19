"""Government and opposition sides from the prime-minister vote, and per-vote flags.

**Blocs.** A term's sides are read off its statsministeromröstning: the
chamber vote (numeric beteckning, ``avser == "frågan"``) titled ``Prövning av
(talmannens) förslag till statsminister …``. Under RF 6:4 the speaker's
proposal is rejected only if more than half of all members -- at least 175 of
349 -- vote Nej; otherwise it is approved. The latest approved PM vote of the
term is used. Its party lines give ``government_side`` (line Ja),
``opposition`` (line Nej) and ``abstained`` (line Avstår); parties with no line
(tie or nobody cast) are listed in ``no_line``; independents have no line and
are not placed. A term with no approved PM vote yet has ``blocs = null``.

The sides are those of the PM vote. A party that tolerated the government by
abstaining is in ``abstained`` and is ignored by the bloc flags.

**Per-vote flags** (``vote_flags``), for every vote of a summary in the order
of ``party_lines.votes``: totals, ``margin = |Ja - Nej|``, ``against_line``
(members casting Ja/Nej/Avstår different from their party's non-null line),
and two bloc flags that use only party lines Ja and Nej (Avstår, ties and no
line are ignored), both null when the term has no blocs:

* ``cross_bloc`` = 1 iff the outcome cut across both blocs: with the winning
  line the one with more Ja or Nej ballots in the chamber (ja == nej: no
  outcome, 0), at least one government_side party and at least one opposition
  party have the winning line AND at least one of each has the losing line;
* ``gov_split`` = 1 iff government_side parties have both Ja and Nej lines.
"""

from __future__ import annotations

import re
from typing import Sequence

from .constants import (
    AVSTAR,
    FRANVARANDE,
    JA,
    NEJ,
    PM_REJECTION_NEJ_THRESHOLD,
    ROST_VALUES,
)
from .party_lines import PartyLine, against_line, order_parties, party_lines_for_vote

PM_VOTE_TITLE = re.compile(r"^Prövning av (talmannens )?förslag till statsminister\b", re.IGNORECASE)

BLOCS_RULE = (
    "latest approved statsministeromröstning of the term (chamber vote, avser frågan, title 'Prövning av "
    f"(talmannens) förslag till statsminister'; rejected only if Nej >= {PM_REJECTION_NEJ_THRESHOLD} of 349); "
    "government_side = parties whose line was Ja, opposition = Nej, abstained = Avstår, no_line = tie or "
    "nobody cast; independents not placed"
)
CROSS_BLOC_RULE = (
    "1 iff the outcome cut across both blocs: using party lines Ja/Nej only (Avstår, ties, no line ignored) and "
    "the winning line = the side with more Ja vs Nej ballots in the chamber (ja == nej: no outcome, 0), at least "
    "one government_side party and at least one opposition party have the winning line, and at least one "
    "government_side party and at least one opposition party have the losing line; abstained parties ignored; "
    "null when blocs is null"
)
GOV_SPLIT_RULE = (
    "1 iff government_side parties have both Ja and Nej lines on the vote (Avstår, ties, no line not counted); "
    "null when blocs is null"
)
VOTE_FLAG_FIELDS = ("ja", "nej", "avstar", "franvarande", "margin", "against_line", "cross_bloc", "gov_split")


def is_pm_vote(vote) -> bool:
    return (
        vote.committee_kind == "numeric"
        and vote.avser == "frågan"
        and bool(vote.title)
        and PM_VOTE_TITLE.match(vote.title) is not None
    )


def pm_vote_approved(vote) -> bool:
    return vote.counts()[NEJ] < PM_REJECTION_NEJ_THRESHOLD


def find_blocs(votes: Sequence) -> dict | None:
    """The blocs of the latest approved PM vote among ``votes``, or None."""
    candidates = [vote for vote in votes if is_pm_vote(vote) and pm_vote_approved(vote)]
    if not candidates:
        return None
    vote = max(candidates, key=lambda v: v.sort_key())
    lines = party_lines_for_vote(vote)
    sides: dict[str, list[str]] = {JA: [], NEJ: [], AVSTAR: [], None: []}
    for party in order_parties(lines):
        sides[lines[party].line].append(party)
    counts = vote.counts()
    return {
        "source_vote_id": vote.votering_id,
        "rm": vote.rm,
        "date": vote.date,
        "title": vote.title,
        "counts": [counts[rost] for rost in ROST_VALUES],
        "government_side": sides[JA],
        "opposition": sides[NEJ],
        "abstained": sides[AVSTAR],
        "no_line": sides[None],
        "rule": BLOCS_RULE,
    }


def pm_votes_considered(votes: Sequence) -> list[dict]:
    """Every PM vote among ``votes`` with whether it was approved (for the record)."""
    return [
        {
            "id": vote.votering_id,
            "date": vote.date,
            "title": vote.title,
            "nej": vote.counts()[NEJ],
            "approved": pm_vote_approved(vote),
        }
        for vote in sorted(votes, key=lambda v: v.sort_key())
        if is_pm_vote(vote)
    ]


def _ja_nej_lines(vote_lines: dict[str, PartyLine], parties: Sequence[str]) -> set[str]:
    return {vote_lines[p].line for p in parties if p in vote_lines and vote_lines[p].line in (JA, NEJ)}


def cross_bloc(vote_lines: dict[str, PartyLine], blocs: dict | None, ja: int, nej: int) -> bool | None:
    """Whether the outcome cut across both blocs (see ``CROSS_BLOC_RULE``)."""
    if blocs is None:
        return None
    if ja == nej:
        return False
    winning, losing = (JA, NEJ) if ja > nej else (NEJ, JA)
    government = _ja_nej_lines(vote_lines, blocs["government_side"])
    opposition = _ja_nej_lines(vote_lines, blocs["opposition"])
    return winning in government and winning in opposition and losing in government and losing in opposition


def gov_split(vote_lines: dict[str, PartyLine], blocs: dict | None) -> bool | None:
    """Whether government_side parties have both Ja and Nej lines."""
    if blocs is None:
        return None
    return _ja_nej_lines(vote_lines, blocs["government_side"]) == {JA, NEJ}


def vote_flags(votes: Sequence, lines: dict[str, dict[str, PartyLine]], blocs: dict | None) -> dict:
    """Columnar per-vote flags in the order of ``votes``."""
    columns: dict[str, list] = {field: [] for field in VOTE_FLAG_FIELDS}
    for vote in votes:
        counts = vote.counts()
        vote_lines = lines[vote.votering_id]
        columns["ja"].append(counts[JA])
        columns["nej"].append(counts[NEJ])
        columns["avstar"].append(counts[AVSTAR])
        columns["franvarande"].append(counts[FRANVARANDE])
        columns["margin"].append(abs(counts[JA] - counts[NEJ]))
        columns["against_line"].append(sum(bool(against_line(b, vote_lines)) for b in vote.ballots))
        for field, flag in (
            ("cross_bloc", cross_bloc(vote_lines, blocs, counts[JA], counts[NEJ])),
            ("gov_split", gov_split(vote_lines, blocs)),
        ):
            columns[field].append(None if flag is None else int(flag))
    return {
        "order": "one entry per vote, in the order of party_lines.votes (all votes; party_lines.votes[i].analysis "
        "marks the analysis set)",
        "fields": list(VOTE_FLAG_FIELDS),
        "columns": columns,
        "definitions": {
            "ja/nej/avstar/franvarande": "ballot totals",
            "margin": "|ja - nej|",
            "against_line": "members casting Ja/Nej/Avstår different from their party's non-null line",
            "cross_bloc": CROSS_BLOC_RULE,
            "gov_split": GOV_SPLIT_RULE,
        },
    }


__all__ = [
    "BLOCS_RULE",
    "CROSS_BLOC_RULE",
    "GOV_SPLIT_RULE",
    "VOTE_FLAG_FIELDS",
    "cross_bloc",
    "gov_split",
    "find_blocs",
    "is_pm_vote",
    "pm_vote_approved",
    "pm_votes_considered",
    "vote_flags",
]
