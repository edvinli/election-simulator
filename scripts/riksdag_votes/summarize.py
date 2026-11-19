"""Assemble the derived JSON payloads (votes.json and summary.json).

Pure functions of the ingested votes plus explicit provenance, so the same
inputs always give byte-identical output.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Iterable, Sequence

import numpy as np

from .agreement import agreement_matrices
from .committees import committee_key
from .constants import (
    ANALYSIS_AVSER,
    ANALYSIS_VOTERING,
    AVSTAR,
    CAST_VALUES,
    CAVEAT_FELTRYCKNING,
    CAVEAT_KVITTNING,
    CAVEAT_MAP,
    COMMITTEES,
    FRANVARANDE,
    JA,
    MAP_AXIS_ANCHORS,
    MIN_AGREEMENT_CELL_N,
    MIN_MAP_VOTES_FOR_FIT,
    MIN_MP_MAP_VOTES,
    NEAR_UNANIMOUS_MINORITY_SHARE,
    NEJ,
    OTHER_COMMITTEE,
    ROLE_SESSION_VOTES,
    ROLE_SUMMARY,
    ROST_CODES,
    ROST_NUMERIC,
    ROST_VALUES,
    ROUND_DIGITS,
    SCHEMA_VERSION,
    SEATS,
    TOP_LOADING_VOTES,
)
from .ideal_points import fit_party_map, is_near_unanimous, project_member
from .ingest import SessionData, Vote
from .party_lines import PartyLine, against_line, order_parties, party_lines_for_vote

LINE_CODES = {JA: "J", NEJ: "N", AVSTAR: "A"}

RULES = {
    "analysis_set": f"votering == {ANALYSIS_VOTERING!r} and avser == {ANALYSIS_AVSER!r}",
    "party_line": (
        "plurality among the party's members voting Ja/Nej/Avstår (Frånvarande excluded); "
        "tie -> null and listed in ties; nobody cast -> null and listed in no_cast; parti '-' has no line"
    ),
    "agreement_three_way": "share of votes, where both parties have a line, with identical lines among Ja/Nej/Avstår",
    "agreement_ja_nej": "share of votes, where both lines are Ja or Nej, with identical lines",
    "min_agreement_cell_n": MIN_AGREEMENT_CELL_N,
    "against_line": "cast Ja/Nej/Avstår that differs from the member's party's non-null line (analysis set)",
    "participation": "share of analysis-set votes the member was seated for on which rost != Frånvarande",
    "map_encoding": {"Ja": 1, "Avstår": 0, "Nej": -1, "Frånvarande": None},
    "near_unanimous_minority_share": NEAR_UNANIMOUS_MINORITY_SHARE,
    "min_map_votes_for_fit": MIN_MAP_VOTES_FOR_FIT,
    "min_mp_map_votes": MIN_MP_MAP_VOTES,
    "map_axis_anchors": list(MAP_AXIS_ANCHORS),
}


def _round(value: float) -> float:
    rounded = round(float(value), ROUND_DIGITS)
    return 0.0 if rounded == 0 else rounded


def _line_code(line: PartyLine | None) -> str | None:
    if line is None or line.line is None:
        return None
    return LINE_CODES[line.line]


# --- votes.json --------------------------------------------------------------


def build_votes_payload(session: SessionData, provenance: dict) -> dict:
    """Per-session vote metadata and a compact 349-bench ballot matrix.

    Encoding: ``people[i]`` is a person; ``seat_holders[j] = [person index,
    parti, valkrets]`` is a person with the affiliation they had on a vote;
    ``rosters[k]`` is a list of 349 seat-holder indices, element b-1 being the
    occupant of bench b; each vote names its roster and gives ``ballots``, a
    349-character string whose character b-1 is the bench-b ballot (J = Ja,
    N = Nej, A = Avstår, F = Frånvarande).
    """
    latest_person: dict[str, dict] = {}
    for vote in session.votes:
        for ballot in vote.ballots:
            latest_person[ballot.intressent_id] = {
                "id": ballot.intressent_id,
                "name": ballot.namn,
                "fornamn": ballot.fornamn,
                "efternamn": ballot.efternamn,
                "iort": ballot.iort,
                "kon": ballot.kon,
                "fodd": ballot.fodd,
            }
    people = [latest_person[person_id] for person_id in sorted(latest_person)]
    person_index = {person["id"]: index for index, person in enumerate(people)}

    holder_keys = sorted(
        {(person_index[b.intressent_id], b.parti, b.valkrets or "") for vote in session.votes for b in vote.ballots}
    )
    holder_index = {key: index for index, key in enumerate(holder_keys)}
    rosters: list[tuple[int, ...]] = []
    roster_index: dict[tuple[int, ...], int] = {}
    votes = []
    for vote in session.votes:
        roster = tuple(holder_index[(person_index[b.intressent_id], b.parti, b.valkrets or "")] for b in vote.ballots)
        if roster not in roster_index:
            roster_index[roster] = len(rosters)
            rosters.append(roster)
        counts = vote.counts()
        votes.append(
            {
                "id": vote.votering_id,
                "beteckning": vote.beteckning,
                "punkt": vote.punkt,
                "avser": vote.avser,
                "votering": vote.votering,
                "hangar_id": vote.hangar_id,
                "date": vote.date,
                "date_source": vote.date_source,
                "committee": vote.committee,
                "analysis": vote.in_analysis_set,
                "source": vote.source,
                "roster": roster_index[roster],
                "ballots": "".join(ROST_CODES[b.rost] for b in vote.ballots),
                "counts": [counts[rost] for rost in ROST_VALUES],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "role": ROLE_SESSION_VOTES,
        "rm": session.rm,
        "seats": SEATS,
        "encoding": {
            "ballots": "349 characters, character b-1 = ballot of bench b: J=Ja, N=Nej, A=Avstår, F=Frånvarande",
            "counts": list(ROST_VALUES),
            "seat_holders": "[person index, parti, valkrets]",
            "rosters": "349 seat_holders indices, element b-1 = occupant of bench b (member or substitute)",
        },
        "people": people,
        "seat_holders": [[p, parti, valkrets or None] for p, parti, valkrets in holder_keys],
        "rosters": [list(roster) for roster in rosters],
        "votes": votes,
        "excluded_votes": session.excluded,
        "unknown_committee_prefixes": session.unknown_committee_prefixes,
        "notes": session.notes,
        "provenance": provenance,
    }


# --- summary.json ------------------------------------------------------------


def _agreement_section(analysis: Sequence[Vote], lines: dict[str, dict[str, PartyLine]], parties: list[str]) -> dict:
    def as_codes(vote: Vote) -> dict[str, str | None]:
        vote_lines = lines[vote.votering_id]
        return {party: vote_lines[party].line if party in vote_lines else None for party in parties}

    by_committee_votes: dict[str, list[Vote]] = defaultdict(list)
    for vote in analysis:
        by_committee_votes[committee_key(vote.committee)].append(vote)
    by_committee = {}
    for key in sorted(by_committee_votes):
        section = agreement_matrices([as_codes(vote) for vote in by_committee_votes[key]], parties)
        section["name"] = COMMITTEES.get(key, OTHER_COMMITTEE)
        by_committee[key] = section
    return {
        "parties": parties,
        "min_cell_n": MIN_AGREEMENT_CELL_N,
        "overall": agreement_matrices([as_codes(vote) for vote in analysis], parties),
        "by_committee": by_committee,
    }


def _descending(text: str) -> tuple[int, ...]:
    """Sort key that orders strings descending inside an ascending sort."""
    return tuple(-ord(char) for char in text)


def _vote_brief(vote: Vote, vote_lines: dict[str, PartyLine], parties: list[str]) -> dict:
    by_line = defaultdict(list)
    for party in parties:
        line = vote_lines.get(party)
        by_line[line.line if line and line.line else "none"].append(party)
    return {
        "id": vote.votering_id,
        "rm": vote.rm,
        "beteckning": vote.beteckning,
        "punkt": vote.punkt,
        "committee": vote.committee,
        "date": vote.date,
        "parties_ja": by_line[JA],
        "parties_nej": by_line[NEJ],
        "parties_avstar": by_line[AVSTAR],
        "parties_no_line": by_line["none"],
    }


def _map_section(analysis: Sequence[Vote], lines: dict[str, dict[str, PartyLine]], parties: list[str]):
    """Returns (map section, {person id: member map entry})."""
    candidates = []
    near_unanimous = missing_line = 0
    for vote in analysis:
        counts = vote.counts()
        if is_near_unanimous(counts[JA], counts[NEJ]):
            near_unanimous += 1
            continue
        vote_lines = lines[vote.votering_id]
        if any(party not in vote_lines or vote_lines[party].line is None for party in parties):
            missing_line += 1
            continue
        candidates.append(vote)
    counts_section = {
        "analysis": len(analysis),
        "near_unanimous_dropped": near_unanimous,
        "missing_party_line_dropped": missing_line,
        "map": len(candidates),
    }
    if len(parties) <= 2 or len(candidates) < MIN_MAP_VOTES_FOR_FIT:
        return {
            "available": False,
            "reason": f"needs at least {MIN_MAP_VOTES_FOR_FIT} map votes and three parties",
            "n_votes": counts_section,
        }, {}

    matrix = np.array(
        [[ROST_NUMERIC[lines[vote.votering_id][party].line] for vote in candidates] for party in parties], dtype=float
    )
    fitted = fit_party_map(matrix, parties)
    column = {vote.votering_id: index for index, vote in enumerate(candidates)}
    member_codes: dict[str, np.ndarray] = {}
    for vote in candidates:
        position = column[vote.votering_id]
        for ballot in vote.ballots:
            codes = member_codes.get(ballot.intressent_id)
            if codes is None:
                codes = member_codes[ballot.intressent_id] = np.full(len(candidates), np.nan)
            if ballot.rost != FRANVARANDE:
                codes[position] = ROST_NUMERIC[ballot.rost]
    scale = 1.0 / math.sqrt(len(candidates))
    members = {}
    for person_id in sorted(member_codes):
        point, n_cast, reason = project_member(member_codes[person_id], fitted.mu, fitted.loadings)
        entry = {"n_cast_map_votes": n_cast}
        if point is None:
            entry["unplaced_reason"] = reason
        else:
            entry["x"], entry["y"] = _round(point[0] * scale), _round(point[1] * scale)
        members[person_id] = entry

    top_votes, top_patterns = {}, {}
    for axis in range(fitted.loadings.shape[1]):
        loadings = fitted.loadings[:, axis]
        # With a handful of parties many votes share one party-line pattern and
        # hence one loading; ties are broken by most recent date, then id.
        order = sorted(
            range(len(candidates)),
            key=lambda i: (-round(abs(loadings[i]), 10), _descending(candidates[i].date or ""), candidates[i].votering_id),
        )
        top_votes[f"axis_{axis + 1}"] = [
            dict(_vote_brief(candidates[i], lines[candidates[i].votering_id], parties), loading=_round(loadings[i]))
            for i in order[:TOP_LOADING_VOTES]
        ]
        patterns: dict[tuple, list[int]] = defaultdict(list)
        for i in order:
            patterns[tuple(lines[candidates[i].votering_id][party].line for party in parties)].append(i)
        ranked = sorted(patterns.items(), key=lambda item: (-round(abs(loadings[item[1][0]]), 10), -len(item[1]), item[0]))
        top_patterns[f"axis_{axis + 1}"] = [
            {
                "loading": _round(loadings[members_[0]]),
                "n_votes": len(members_),
                "parties_ja": [party for party, line in zip(parties, pattern) if line == JA],
                "parties_nej": [party for party, line in zip(parties, pattern) if line == NEJ],
                "parties_avstar": [party for party, line in zip(parties, pattern) if line == AVSTAR],
                "example_vote_ids": [candidates[i].votering_id for i in members_[:3]],
            }
            for pattern, members_ in ranked[:TOP_LOADING_VOTES]
        ]
    section = {
        "available": True,
        "parties": parties,
        "coordinate_scale": (
            "PCA scores and member solutions divided by sqrt(n map votes), so positions are comparable "
            "across periods of different length; loadings are the unit-norm PCA vectors"
        ),
        "party_coordinates": {
            party: [_round(fitted.party_scores[i, 0] * scale), _round(fitted.party_scores[i, 1] * scale)]
            for i, party in enumerate(parties)
        },
        "explained_variance_ratio": [_round(value) for value in fitted.explained_variance_ratio[:2]],
        "explained_variance_ratio_all_components": [_round(value) for value in fitted.explained_variance_ratio],
        "n_votes": counts_section,
        "orientation": (
            f"axis 1 oriented so {MAP_AXIS_ANCHORS[0]} scores positive, axis 2 so {MAP_AXIS_ANCHORS[1]} scores "
            "positive; signs are a reproducibility convention, axes are not left/right"
        ),
        "top_loading_votes": top_votes,
        "top_loading_patterns": top_patterns,
        "member_rule": f"least squares on the member's own cast map votes; at least {MIN_MP_MAP_VOTES} required",
    }
    return section, members


def build_summary(
    votes: Sequence[Vote],
    *,
    scope: dict,
    provenance: dict,
    notes: Iterable[str] = (),
) -> dict:
    votes = sorted(votes, key=Vote.sort_key)
    analysis = [vote for vote in votes if vote.in_analysis_set]
    lines = {vote.votering_id: party_lines_for_vote(vote) for vote in votes}
    parties = order_parties(party for vote in analysis for party in lines[vote.votering_id])
    line_parties = order_parties(party for vote in votes for party in lines[vote.votering_id])

    party_lines_votes = []
    for vote in votes:
        vote_lines = lines[vote.votering_id]
        party_lines_votes.append(
            {
                "id": vote.votering_id,
                "rm": vote.rm,
                "analysis": vote.in_analysis_set,
                "lines": [_line_code(vote_lines.get(party)) for party in line_parties],
                "ties": [party for party in line_parties if party in vote_lines and vote_lines[party].tie],
                "no_cast": [party for party in line_parties if party in vote_lines and vote_lines[party].no_cast],
            }
        )

    # Per-member statistics over the analysis set.
    stats: dict[str, dict] = {}
    for vote in analysis:
        vote_lines = lines[vote.votering_id]
        for ballot in vote.ballots:
            entry = stats.setdefault(
                ballot.intressent_id,
                {"seated": 0, "cast": 0, "against": [], "comparable": 0, "parties": Counter()},
            )
            entry["latest"] = ballot
            entry["seated"] += 1
            entry["parties"][ballot.parti] += 1
            if ballot.rost in CAST_VALUES:
                entry["cast"] += 1
            verdict = against_line(ballot, vote_lines)
            if verdict is not None:
                entry["comparable"] += 1
                if verdict:
                    entry["against"].append(vote.votering_id)

    map_section, member_map = _map_section(analysis, lines, parties)

    unplaced_default = (
        {"n_cast_map_votes": 0, "unplaced_reason": "not seated for any map vote"} if map_section["available"] else None
    )
    members = []
    party_against: dict[str, dict[str, int]] = {party: {"against": 0, "comparable": 0} for party in parties}
    for person_id in sorted(stats):
        entry = stats[person_id]
        latest = entry["latest"]
        participation = entry["cast"] / entry["seated"] if entry["seated"] else None
        member = {
            "id": person_id,
            "name": latest.namn,
            "iort": latest.iort,
            "party": latest.parti,
            "parties": dict(sorted(entry["parties"].items())),
            "valkrets": latest.valkrets,
            "seated": entry["seated"],
            "cast": entry["cast"],
            "absent": entry["seated"] - entry["cast"],
            "participation": _round(participation) if participation is not None else None,
            "against_line": len(entry["against"]),
            "against_line_comparable": entry["comparable"],
            "against_line_vote_ids": entry["against"],
            "map": member_map.get(person_id, unplaced_default),
        }
        members.append(member)
    # Party totals attribute each against-line ballot to the party on that vote.
    for vote in analysis:
        vote_lines = lines[vote.votering_id]
        for ballot in vote.ballots:
            verdict = against_line(ballot, vote_lines)
            if verdict is None or ballot.parti not in party_against:
                continue
            party_against[ballot.parti]["comparable"] += 1
            party_against[ballot.parti]["against"] += int(verdict)

    participations = [member["participation"] for member in members if member["participation"] is not None]
    avser_counts = Counter(vote.avser for vote in votes)
    return {
        "schema_version": SCHEMA_VERSION,
        "role": ROLE_SUMMARY,
        "scope": scope,
        "rules": RULES,
        "counts": {
            "votes_all": len(votes),
            "votes_analysis": len(analysis),
            "votes_by_avser": dict(sorted(avser_counts.items())),
            "votes_by_committee_analysis": dict(sorted(Counter(committee_key(v.committee) for v in analysis).items())),
            "members": len(members),
        },
        "parties": parties,
        "party_lines": {
            "parties": line_parties,
            "codes": {"J": JA, "N": NEJ, "A": AVSTAR, "null": "no line (see ties / no_cast)"},
            "votes": party_lines_votes,
        },
        "agreement": _agreement_section(analysis, lines, parties),
        "party_against_line": party_against,
        "members": members,
        "participation_distribution": {
            "min": min(participations) if participations else None,
            "median": _round(statistics.median(participations)) if participations else None,
            "max": max(participations) if participations else None,
        },
        "map": map_section,
        "caveats": {"kvittning": CAVEAT_KVITTNING, "feltryckning": CAVEAT_FELTRYCKNING, "map": CAVEAT_MAP},
        "notes": list(notes),
        "provenance": provenance,
    }


__all__ = ["RULES", "build_summary", "build_votes_payload"]
