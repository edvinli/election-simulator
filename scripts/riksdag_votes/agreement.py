"""Pairwise party agreement (rule 2).

Primary, "three-way": over the votes where both parties have a line, the
share on which the lines are identical among {Ja, Nej, Avstår}. Avstår against
Ja counts as disagreement, Avstår against Avstår as agreement.

Secondary, "Ja/Nej only": over the votes where both lines are Ja or Nej, the
share with identical lines. It ignores every vote on which either party
abstained, so it answers a narrower question and has a smaller n.

Every cell carries its vote count ``n``; the share is published only when
``n >= MIN_AGREEMENT_CELL_N``.
"""

from __future__ import annotations

from typing import Sequence

from .constants import JA, MIN_AGREEMENT_CELL_N, NEJ, ROUND_DIGITS

JA_NEJ = (JA, NEJ)


def agreement_matrices(
    lines_by_vote: Sequence[dict[str, str | None]],
    parties: Sequence[str],
    *,
    min_n: int = MIN_AGREEMENT_CELL_N,
) -> dict:
    """``lines_by_vote``: per vote, party -> line (None when the party has no line)."""
    size = len(parties)
    agree3 = [[0] * size for _ in range(size)]
    n3 = [[0] * size for _ in range(size)]
    agree2 = [[0] * size for _ in range(size)]
    n2 = [[0] * size for _ in range(size)]
    for lines in lines_by_vote:
        values = [lines.get(party) for party in parties]
        for i, a in enumerate(values):
            if a is None:
                continue
            for j, b in enumerate(values):
                if b is None:
                    continue
                n3[i][j] += 1
                agree3[i][j] += a == b
                if a in JA_NEJ and b in JA_NEJ:
                    n2[i][j] += 1
                    agree2[i][j] += a == b

    def shares(agree, n):
        return [
            [round(agree[i][j] / n[i][j], ROUND_DIGITS) if n[i][j] >= min_n and n[i][j] > 0 else None for j in range(size)]
            for i in range(size)
        ]

    return {
        "n_votes": len(lines_by_vote),
        "three_way": {"agree": agree3, "n": n3, "share": shares(agree3, n3)},
        "ja_nej": {"agree": agree2, "n": n2, "share": shares(agree2, n2)},
    }
