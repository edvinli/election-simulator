"""Parse, validate and reconcile riksdag roll-call votes for one session.

Sources, in order of use:

* the bulk session archive ``votering-<YYYYZZ>.json.zip`` -- one JSON file per
  vote, ``{"dokvotering": {"votering": rows}}`` (``rows`` is a dict when the
  file holds a single row; decoded as ``utf-8-sig``);
* the ``voteringlista`` register (``gruppering=votering_id``), which lists
  every vote the API currently recognises with its Ja/Nej/Frånvarande/Avstår
  totals. It is the authority on *which* votes exist: the bulk archives are
  regenerated irregularly and on 2026-11-19 lacked late-registered votes (for
  example the 2022-10-17 prime-minister vote) while still containing votes the
  API has since withdrawn (aborted votes with ~340 absent, a re-run SfU31 vote);
* for a registered vote absent from the bulk archive, its ``beteckning`` from
  ``votering/<id>/json`` and then its rows from ``voteringlista`` filtered on
  ``rm`` and ``bet`` (the ``votering_id`` query parameter is silently ignored by
  that endpoint, so it is never used), grouped by ``votering_id``;
* ``votering/<id>/json`` metadata also supplies the date of votes whose rows
  carry ``datum: null`` (``frågan`` votes such as the prime-minister vote).

Every vote must have exactly 349 rows, one per bench 1..349, no duplicate
(votering_id, intressent_id) pair, a known ``rost``, consistent vote-level
fields, and totals equal to the register's. Any violation raises
``IngestError``: a truncated or duplicated answer is never analysed.
"""

from __future__ import annotations

import io
import json
import logging
import urllib.parse
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .committees import parse_committee
from .constants import (
    ANALYSIS_AVSER,
    ANALYSIS_VOTERING,
    ROST_VALUES,
    SEATS,
)
from .fetch import (
    NotFound,
    RawStore,
    bulk_url,
    register_url,
    rows_by_beteckning_url,
    session_code,
    vote_metadata_url,
)

LOGGER = logging.getLogger(__name__)

VOTE_LEVEL_FIELDS = ("votering_id", "rm", "beteckning", "punkt", "avser", "votering", "hangar_id")


class IngestError(ValueError):
    """Raised when source data fails a completeness or consistency check."""


@dataclass(frozen=True)
class Ballot:
    banknummer: int
    intressent_id: str
    rost: str
    parti: str
    valkrets: str | None
    namn: str
    fornamn: str | None
    efternamn: str | None
    kon: str | None
    fodd: str | None
    iort: str | None


@dataclass(frozen=True)
class Vote:
    votering_id: str
    rm: str
    beteckning: str
    punkt: str
    avser: str
    votering: str
    hangar_id: str
    date: str | None
    date_source: str | None
    committee: str | None
    committee_kind: str
    source: str  # "bulk" or "voteringlista"
    ballots: tuple[Ballot, ...]  # ordered by banknummer 1..349
    # What was voted on (``titles.attach_titles``): the document's title, the
    # heading of the förslagspunkt and the document id.
    title: str | None = None
    punkt_title: str | None = None
    dok_id: str | None = None

    @property
    def in_analysis_set(self) -> bool:
        return self.votering == ANALYSIS_VOTERING and self.avser == ANALYSIS_AVSER

    def counts(self) -> dict[str, int]:
        tally = Counter(ballot.rost for ballot in self.ballots)
        return {rost: tally.get(rost, 0) for rost in ROST_VALUES}

    def sort_key(self) -> tuple:
        return (
            self.date or "9999-99-99",
            int(self.hangar_id) if self.hangar_id.isdigit() else 0,
            int(self.punkt) if self.punkt.isdigit() else 0,
            self.beteckning,
            self.avser,
            self.votering_id,
        )


@dataclass
class SessionData:
    rm: str
    votes: list[Vote]
    excluded: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    unknown_committee_prefixes: list[str] = field(default_factory=list)
    source_files: set[str] = field(default_factory=set)
    title_files: set[str] = field(default_factory=set)


def _blank_to_none(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_vote_document(document: dict) -> list[dict]:
    """Rows of one bulk per-vote JSON document (a dict row becomes a list)."""
    try:
        rows = document["dokvotering"]["votering"]
    except (KeyError, TypeError) as error:
        raise IngestError("document lacks dokvotering.votering") from error
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        raise IngestError(f"dokvotering.votering has unexpected type {type(rows).__name__}")
    return rows


def decode_json(data: bytes) -> dict:
    return json.loads(data.decode("utf-8-sig"))


def validate_vote_rows(rows: list[dict]) -> None:
    """Fail loudly unless ``rows`` is one complete vote (rule 5)."""
    if len(rows) != SEATS:
        ids = sorted({str(row.get("votering_id")) for row in rows})
        raise IngestError(f"vote {ids} has {len(rows)} rows, expected {SEATS}")
    pairs = Counter((row.get("votering_id"), row.get("intressent_id")) for row in rows)
    duplicates = sorted(str(pair) for pair, count in pairs.items() if count > 1)
    if duplicates:
        raise IngestError(f"duplicate (votering_id, intressent_id) pairs: {duplicates[:5]}")
    for name in VOTE_LEVEL_FIELDS:
        values = {str(row.get(name)) for row in rows}
        if len(values) != 1:
            raise IngestError(f"vote-level field {name!r} is not constant within a vote: {sorted(values)[:5]}")
    vote_id = rows[0]["votering_id"]
    unknown = sorted({str(row.get("rost")) for row in rows} - set(ROST_VALUES))
    if unknown:
        raise IngestError(f"vote {vote_id}: unknown rost values {unknown}")
    try:
        benches = sorted(int(row["banknummer"]) for row in rows)
    except (KeyError, TypeError, ValueError) as error:
        raise IngestError(f"vote {vote_id}: missing or non-integer banknummer") from error
    if benches != list(range(1, SEATS + 1)):
        raise IngestError(f"vote {vote_id}: banknummer is not exactly 1..{SEATS}")
    if any(not row.get("intressent_id") for row in rows):
        raise IngestError(f"vote {vote_id}: row without intressent_id")
    if any(not row.get("parti") for row in rows):
        raise IngestError(f"vote {vote_id}: row without parti")


def vote_from_rows(
    rows: list[dict],
    *,
    source: str,
    date: str | None = None,
    date_source: str | None = None,
) -> Vote:
    validate_vote_rows(rows)
    first = rows[0]
    if date is None:
        row_date = _blank_to_none(first.get("datum"))
        if row_date is not None:
            date, date_source = row_date[:10], "row_datum"
    committee = parse_committee(first.get("beteckning"))
    ballots = tuple(
        sorted(
            (
                Ballot(
                    banknummer=int(row["banknummer"]),
                    intressent_id=str(row["intressent_id"]),
                    rost=row["rost"],
                    parti=str(row["parti"]).strip(),
                    valkrets=_blank_to_none(row.get("valkrets")),
                    namn=str(row.get("namn") or "").strip(),
                    fornamn=_blank_to_none(row.get("fornamn")),
                    efternamn=_blank_to_none(row.get("efternamn")),
                    kon=_blank_to_none(row.get("kon")),
                    fodd=_blank_to_none(row.get("fodd")),
                    iort=_blank_to_none(row.get("iort")),
                )
                for row in rows
            ),
            key=lambda ballot: ballot.banknummer,
        )
    )
    return Vote(
        votering_id=str(first["votering_id"]),
        rm=str(first["rm"]),
        beteckning=str(first["beteckning"]),
        punkt=str(first["punkt"]),
        avser=str(first["avser"]),
        votering=str(first["votering"]),
        hangar_id=str(first["hangar_id"]),
        date=date,
        date_source=date_source if date else None,
        committee=committee.committee,
        committee_kind=committee.kind,
        source=source,
        ballots=ballots,
    )


def parse_bulk_archive(data: bytes) -> list[list[dict]]:
    """Every vote's rows in a bulk archive, in archive member order."""
    votes = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith("/"):
                continue
            votes.append(parse_vote_document(decode_json(archive.read(name))))
    return votes


def _voteringlista_rows(payload: dict) -> list[dict]:
    body = payload.get("voteringlista")
    if not isinstance(body, dict):
        raise IngestError("voteringlista response lacks the voteringlista object")
    rows = body.get("votering") or []
    if isinstance(rows, dict):
        rows = [rows]
    declared = int(body.get("@antal") or 0)
    if declared != len(rows):
        raise IngestError(f"voteringlista returned {len(rows)} rows but declares @antal={declared} (truncated?)")
    return rows


def parse_register(payload: dict) -> dict[str, dict[str, int]]:
    """``votering_id -> {Ja, Nej, Frånvarande, Avstår}`` from a gruppering=votering_id list."""
    register: dict[str, dict[str, int]] = {}
    for row in _voteringlista_rows(payload):
        vote_id = str(row["votering_id"])
        if vote_id in register:
            raise IngestError(f"register lists {vote_id} twice")
        register[vote_id] = {rost: int(row.get(rost) or 0) for rost in ROST_VALUES}
    return register


def group_rows_by_vote(payload: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in _voteringlista_rows(payload):
        grouped[str(row["votering_id"])].append(row)
    return dict(grouped)


def parse_vote_metadata(payload: dict) -> dict | None:
    document = (payload.get("votering") or {}).get("dokument") if isinstance(payload, dict) else None
    if not isinstance(document, dict):
        return None
    return document


def check_against_register(vote: Vote, register: dict[str, dict[str, int]]) -> None:
    expected = register[vote.votering_id]
    actual = vote.counts()
    if actual != expected:
        raise IngestError(f"vote {vote.votering_id}: row totals {actual} differ from the register's {expected}")


def load_session(rm: str, store: RawStore) -> SessionData:
    code = session_code(rm)
    session = SessionData(rm=rm, votes=[])

    register_name = f"voteringlista-{code}-register.json"
    register = parse_register(decode_json(store.fetch_listing(register_name, register_url(rm))))
    session.source_files.add(register_name)

    bulk_name = f"votering-{code}.json.zip"
    bulk_votes: dict[str, list[dict]] = {}
    try:
        archive = store.fetch_listing(bulk_name, bulk_url(rm))
    except NotFound:
        session.notes.append(f"bulk archive {bulk_url(rm)} not found; all votes fetched from voteringlista")
    else:
        session.source_files.add(bulk_name)
        for rows in parse_bulk_archive(archive):
            validate_vote_rows(rows)
            vote_id = str(rows[0]["votering_id"])
            if vote_id in bulk_votes:
                raise IngestError(f"bulk archive contains vote {vote_id} twice")
            if rows[0]["rm"] != rm:
                raise IngestError(f"bulk archive for {rm} contains a vote from {rows[0]['rm']}")
            bulk_votes[vote_id] = rows

    metadata_cache: dict[str, dict | None] = {}

    def metadata(vote_id: str) -> dict | None:
        if vote_id not in metadata_cache:
            name = f"votering-meta-{code}-{vote_id}.json"
            metadata_cache[vote_id] = parse_vote_metadata(decode_json(store.fetch(name, vote_metadata_url(vote_id))))
            session.source_files.add(name)
        return metadata_cache[vote_id]

    for vote_id in sorted(set(bulk_votes) - set(register)):
        rows = bulk_votes[vote_id]
        counts = Counter(row["rost"] for row in rows)
        session.excluded.append(
            {
                "votering_id": vote_id,
                "beteckning": rows[0].get("beteckning"),
                "punkt": rows[0].get("punkt"),
                "avser": rows[0].get("avser"),
                "date": _blank_to_none(rows[0].get("datum")),
                "counts": {rost: counts.get(rost, 0) for rost in ROST_VALUES},
                "reason": "in the bulk archive but absent from the voteringlista register (withdrawn or re-run vote)",
            }
        )

    rows_by_bet_cache: dict[str, dict[str, list[dict]]] = {}
    for vote_id in sorted(register):
        if vote_id in bulk_votes:
            rows, source = bulk_votes[vote_id], "bulk"
        else:
            meta = metadata(vote_id)
            if meta is None or not meta.get("beteckning"):
                raise IngestError(f"registered vote {vote_id} is absent from the bulk archive and has no metadata")
            bet = str(meta["beteckning"])
            name = f"voteringlista-{code}-bet-{urllib.parse.quote(bet, safe='')}.json"
            if bet not in rows_by_bet_cache:
                rows_by_bet_cache[bet] = group_rows_by_vote(decode_json(store.fetch(name, rows_by_beteckning_url(rm, bet))))
                session.source_files.add(name)
            if vote_id not in rows_by_bet_cache[bet] and not store.offline and not store.is_fresh(name):
                # A cached answer predates this vote (a later vote on the same
                # betänkande); ask again once.
                rows_by_bet_cache[bet] = group_rows_by_vote(
                    decode_json(store.fetch(name, rows_by_beteckning_url(rm, bet), refresh=True))
                )
            rows = rows_by_bet_cache[bet].get(vote_id)
            if rows is None:
                raise IngestError(f"registered vote {vote_id} ({bet}) not returned by voteringlista")
            source = "voteringlista"
            session.notes.append(f"vote {vote_id} ({bet}) missing from the bulk archive; rows from voteringlista")
        date = date_source = None
        if not _blank_to_none(rows[0].get("datum")):
            meta = metadata(vote_id)
            meta_date = _blank_to_none((meta or {}).get("datum"))
            if meta_date:
                date, date_source = meta_date[:10], "votering_metadata"
        vote = vote_from_rows(rows, source=source, date=date, date_source=date_source)
        if vote.rm != rm:
            raise IngestError(f"vote {vote_id} belongs to {vote.rm}, not {rm}")
        check_against_register(vote, register)
        if vote.committee_kind in ("unknown_prefix", "unparsed"):
            session.unknown_committee_prefixes.append(vote.beteckning)
        session.votes.append(vote)

    session.votes.sort(key=Vote.sort_key)
    session.unknown_committee_prefixes = sorted(set(session.unknown_committee_prefixes))
    LOGGER.info(
        "%s: %d votes (%d bulk, %d voteringlista), %d excluded",
        rm,
        len(session.votes),
        sum(vote.source == "bulk" for vote in session.votes),
        sum(vote.source != "bulk" for vote in session.votes),
        len(session.excluded),
    )
    return session


__all__ = [
    "Ballot",
    "IngestError",
    "SessionData",
    "Vote",
    "check_against_register",
    "group_rows_by_vote",
    "load_session",
    "parse_bulk_archive",
    "parse_register",
    "parse_vote_document",
    "validate_vote_rows",
    "vote_from_rows",
]
