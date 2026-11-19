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
abstaining is in ``abstained`` and is ignored by the cross-bloc test.

**Per-vote flags** (``vote_flags``), for every vote of a summary in the order
of ``party_lines.votes``: totals, ``margin = |Ja - Nej|``, ``against_line``
(members casting Ja/Nej/Avstår different from their party's non-null line) and
``cross_bloc``:

* parties considered: those with a (non-null) line on the vote;
* unanimous: they all share one line;
* bloc-aligned: every government_side party has one line X and every
  opposition party one line Y != X (abstained-group parties ignored);
* cross_bloc = not unanimous and not bloc-aligned and at least one
  government_side party shares its line with at least one opposition party;
  null when the term has no blocs.
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
    "among parties with a line: not unanimous, and not bloc-aligned (all government_side parties on one line X, "
    "all opposition parties on one line Y != X; abstained parties ignored), and at least one government_side "
    "party shares its line with at least one opposition party; null when blocs is null"
)
VOTE_FLAG_FIELDS = ("ja", "nej", "avstar", "franvarande", "margin", "against_line", "cross_bloc")


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


def cross_bloc(vote_lines: dict[str, PartyLine], blocs: dict | None) -> bool | None:
    if blocs is None:
        return None
    with_line = {party: line.line for party, line in vote_lines.items() if line.line is not None}
    if len(set(with_line.values())) <= 1:
        return False  # unanimous (or no party with a line)
    government = {with_line[p] for p in blocs["government_side"] if p in with_line}
    opposition = {with_line[p] for p in blocs["opposition"] if p in with_line}
    aligned = len(government) == 1 and len(opposition) == 1 and government != opposition
    shared = bool(government & opposition)
    return (not aligned) and shared


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
        flag = cross_bloc(vote_lines, blocs)
        columns["cross_bloc"].append(None if flag is None else int(flag))
    return {
        "order": "one entry per vote, in the order of party_lines.votes (all votes; party_lines.votes[i].analysis "
        "marks the analysis set)",
        "fields": list(VOTE_FLAG_FIELDS),
        "columns": columns,
        "definitions": {
            "ja/nej/avstar/franvarande": "ballot totals",
            "margin": "|ja - nej|",
            "against_line": "members casting Ja/Nej/Avstår different from their party's non-null line",
            "cross_bloc": "1 = cross-bloc, 0 = not, null = no blocs; " + CROSS_BLOC_RULE,
        },
    }


__all__ = [
    "BLOCS_RULE",
    "CROSS_BLOC_RULE",
    "VOTE_FLAG_FIELDS",
    "cross_bloc",
    "find_blocs",
    "is_pm_vote",
    "pm_vote_approved",
    "pm_votes_considered",
    "vote_flags",
]
