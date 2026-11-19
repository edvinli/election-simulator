"""What each vote was about: the document's title and the förslagspunkt heading.

Source: ``https://data.riksdagen.se/utskottsforslag/<dok_id>.json`` -- one
request per document (about 20 kB), giving

* ``utskottsforslag.dokument.titel``: the betänkande's title, e.g.
  ``Fastighetsrätt`` for 2025/26:CU13, and
* ``utskottsforslag.dokutskottsforslag.utskottsforslag[]``: one entry per
  förslagspunkt with ``punkt`` and ``rubrik`` (the heading), e.g. punkt 6
  ``Förköpslag``.

It was chosen over the alternatives (checked on 2026-11-19):

* ``votering/<id>/json`` has the betänkande title but no punkt heading, needs
  one request per vote instead of per document, and embeds all 349 rows
  (~290 kB a vote);
* ``dokumentstatus/<dok_id>.json`` has the same title and headings but embeds
  the full document HTML (~330 kB a document);
* ``dokumentlista`` has the title and the press headline (``notisrubrik``) but
  no punkt headings.

Chamber votes that are not on a betänkande (numeric beteckning such as
``1017-1``, e.g. the prime-minister vote) have a ``votering`` document on the
same endpoint whose title is ``Omröstning: <subject>``; the ``Omröstning:``
prefix is dropped and there is no punkt heading.

The dok_id is derived from the riksdag's numbering -- ``H`` + session
character (2012/13 = ``0`` ... 2022/23 = ``A``, 2025/26 = ``D``) + series
(``01`` betänkande, ``19`` votering) + beteckning, e.g. ``HD01CU13`` -- and the
answer is accepted only if its ``rm`` and ``beteckning`` match the vote. If the
derived id is absent or does not match, the id is looked up with
``dokumentlista?rm=<rm>&bet=<beteckning>``.

Files are cached per document id like every other raw file (SHA-256 in the
manifest), so a rebuild fetches only documents it has not seen. A cached
document that lacks a voted punkt (fetched before the riksdag published the
proposal) is downloaded again once per run. Offline, a document that is not in
the cache simply has no title; the same note is written either way, so online
and offline builds of the same cache agree.
"""

from __future__ import annotations

import hashlib
import logging
import re
import urllib.parse
from dataclasses import dataclass, replace

from .constants import DOCUMENT_URL_TEMPLATE
from .fetch import (
    NotFound,
    RawStore,
    dokumentlista_by_beteckning_url,
    session_code,
    utskottsforslag_url,
)
from .ingest import SessionData, Vote, decode_json

LOGGER = logging.getLogger(__name__)

_SESSION_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_SESSION_CHAR_BASE_YEAR = 2012  # 2012/13 -> "0", 2022/23 -> "A"
SERIES_BETANKANDE = "01"
SERIES_VOTERING = "19"
_VOTERING_TITLE_PREFIX = re.compile(r"^\s*Omröstning\s*:\s*", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class DocumentTitles:
    dok_id: str
    rm: str
    beteckning: str
    doktyp: str | None
    title: str | None
    url: str
    punkter: dict[str, str | None]


def clean_text(value) -> str | None:
    if value is None:
        return None
    text = _WHITESPACE.sub(" ", str(value)).strip()
    return text or None


def session_dok_char(rm: str) -> str | None:
    index = int(session_code(rm)[:4]) - _SESSION_CHAR_BASE_YEAR
    return _SESSION_CHARS[index] if 0 <= index < len(_SESSION_CHARS) else None


def guess_dok_id(rm: str, beteckning: str, committee_kind: str) -> str | None:
    """The dok_id the riksdag's numbering gives, or None when it cannot be derived."""
    char = session_dok_char(rm)
    if char is None or not beteckning:
        return None
    if committee_kind == "numeric":
        series = SERIES_VOTERING
    elif committee_kind in ("committee", "unknown_prefix"):
        series = SERIES_BETANKANDE
    else:
        return None
    return f"H{char}{series}{beteckning}"


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return value
    return []


def parse_utskottsforslag(payload: dict) -> DocumentTitles | None:
    """Title, URL and punkt headings of one ``utskottsforslag`` answer."""
    body = payload.get("utskottsforslag") if isinstance(payload, dict) else None
    document = body.get("dokument") if isinstance(body, dict) else None
    if not isinstance(document, dict) or not document.get("dok_id"):
        return None
    dok_id = str(document["dok_id"]).strip()
    doktyp = clean_text(document.get("doktyp"))
    title = clean_text(document.get("titel"))
    if title and doktyp == "votering":
        title = clean_text(_VOTERING_TITLE_PREFIX.sub("", title))
    proposals = body.get("dokutskottsforslag")
    items = _as_list(proposals.get("utskottsforslag")) if isinstance(proposals, dict) else []
    punkter: dict[str, str | None] = {}
    for item in items:
        if not isinstance(item, dict) or item.get("punkt") in (None, ""):
            continue
        punkter.setdefault(str(item["punkt"]).strip(), clean_text(item.get("rubrik")))
    url = clean_text(document.get("dokument_url_html")) or DOCUMENT_URL_TEMPLATE.format(dok_id=dok_id)
    return DocumentTitles(
        dok_id=dok_id,
        rm=str(document.get("rm") or "").strip(),
        beteckning=str(document.get("beteckning") or "").strip(),
        doktyp=doktyp,
        title=title,
        url=url,
        punkter=punkter,
    )


def parse_dokumentlista_dok_id(payload: dict, rm: str, beteckning: str) -> str | None:
    """The single dok_id a ``dokumentlista`` search gives for ``rm`` + ``beteckning``."""
    body = payload.get("dokumentlista") if isinstance(payload, dict) else None
    hits = _as_list(body.get("dokument")) if isinstance(body, dict) else []
    ids = set()
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        if str(hit.get("rm") or "").strip() != rm or str(hit.get("beteckning") or "").strip() != beteckning:
            continue
        if hit.get("doktyp") not in ("bet", "votering") or not hit.get("dok_id"):
            continue
        ids.add(str(hit["dok_id"]).strip())
    return ids.pop() if len(ids) == 1 else None


class _TitleSource:
    def __init__(self, session: SessionData, store: RawStore):
        self.session = session
        self.store = store
        self.code = session_code(session.rm)
        self.files: set[str] = set()

    def _get(self, name: str, url: str, *, refresh: bool = False) -> bytes | None:
        if self.store.offline:
            return self.store.cached(name, url)
        try:
            return self.store.fetch(name, url, refresh=refresh or None)
        except NotFound:
            return None

    def _utskottsforslag(self, dok_id: str, beteckning: str, *, refresh: bool = False) -> DocumentTitles | None:
        name = f"utskottsforslag-{self.code}-{urllib.parse.quote(dok_id, safe='')}.json"
        data = self._get(name, utskottsforslag_url(dok_id), refresh=refresh)
        if data is None:
            return None
        document = parse_utskottsforslag(decode_json(data))
        if document is None or document.rm != self.session.rm or document.beteckning != beteckning:
            return None
        self.files.add(name)
        return document

    def document(self, beteckning: str, committee_kind: str, punkter: set[str]) -> DocumentTitles | None:
        guess = guess_dok_id(self.session.rm, beteckning, committee_kind)
        document = self._utskottsforslag(guess, beteckning) if guess else None
        if document is None:
            name = f"dokumentlista-{self.code}-bet-{urllib.parse.quote(beteckning, safe='')}.json"
            data = self._get(name, dokumentlista_by_beteckning_url(self.session.rm, beteckning))
            dok_id = parse_dokumentlista_dok_id(decode_json(data), self.session.rm, beteckning) if data else None
            if dok_id is not None:
                self.files.add(name)
                if dok_id != guess:
                    document = self._utskottsforslag(dok_id, beteckning)
        if document is None:
            return None
        missing = punkter - set(document.punkter)
        if missing and document.doktyp != "votering" and not self.store.offline:
            name = f"utskottsforslag-{self.code}-{urllib.parse.quote(document.dok_id, safe='')}.json"
            if not self.store.is_fresh(name):
                document = self._utskottsforslag(document.dok_id, beteckning, refresh=True) or document
        return document


def attach_titles(session: SessionData, store: RawStore) -> None:
    """Set ``title``, ``punkt_title`` and ``dok_id`` on every vote.

    Adds the title files to ``session.title_files`` and a note for every
    beteckning without a document and every voted punkt without a heading.
    """
    source = _TitleSource(session, store)
    by_beteckning: dict[str, list[Vote]] = {}
    for vote in session.votes:
        by_beteckning.setdefault(vote.beteckning, []).append(vote)
    documents: dict[str, DocumentTitles | None] = {}
    for beteckning in sorted(by_beteckning):
        kind = by_beteckning[beteckning][0].committee_kind
        punkter = {vote.punkt for vote in by_beteckning[beteckning]}
        documents[beteckning] = source.document(beteckning, kind, punkter)

    missing_documents: list[str] = []
    missing_punkter: list[str] = []
    updated = []
    for vote in session.votes:
        document = documents.get(vote.beteckning)
        if document is None:
            missing_documents.append(vote.beteckning)
            updated.append(vote)
            continue
        punkt_title = document.punkter.get(vote.punkt)
        if document.doktyp != "votering" and punkt_title is None:
            missing_punkter.append(f"{vote.beteckning} punkt {vote.punkt}")
        updated.append(
            replace(vote, title=document.title, punkt_title=punkt_title, dok_id=document.dok_id)
        )
    session.votes = updated
    session.title_files = set(source.files)
    if missing_documents:
        session.notes.append(f"no title document found for {sorted(set(missing_documents))}")
    if missing_punkter:
        session.notes.append(f"no förslagspunkt heading for {sorted(set(missing_punkter))}")
    LOGGER.info(
        "%s: titles for %d of %d votes from %d documents",
        session.rm,
        sum(vote.title is not None for vote in session.votes),
        len(session.votes),
        sum(document is not None for document in documents.values()),
    )


def title_sources_digest(store: RawStore, names: set[str]) -> dict:
    """A compact provenance record for the title files (each one is in the manifest)."""
    lines = "".join(f"{name} {store.entries[name]['sha256']}\n" for name in sorted(names))
    return {
        "count": len(names),
        "sha256_of_listing": hashlib.sha256(lines.encode("utf-8")).hexdigest(),
        "listing": "one line per file, sorted by name: '<file> <sha256>\\n'; every file is in manifest.json",
    }


__all__ = [
    "DocumentTitles",
    "attach_titles",
    "clean_text",
    "guess_dok_id",
    "parse_dokumentlista_dok_id",
    "parse_utskottsforslag",
    "session_dok_char",
    "title_sources_digest",
]
