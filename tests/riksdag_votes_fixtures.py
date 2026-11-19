"""Synthetic riksdag roll-call fixtures (no network)."""

from __future__ import annotations

import io
import json
import zipfile

# Seats per party in a 349-seat chamber (2022 result).
PARTY_SEATS = (("S", 107), ("SD", 73), ("M", 68), ("V", 24), ("C", 24), ("KD", 19), ("MP", 18), ("L", 16))
assert sum(seats for _, seats in PARTY_SEATS) == 349


def chamber(independents: int = 0) -> list[dict]:
    """349 people by bench; the last ``independents`` S seats become parti '-'."""
    people = []
    bench = 1
    for party, seats in PARTY_SEATS:
        for k in range(seats):
            people.append(
                {
                    "banknummer": str(bench),
                    "intressent_id": f"{party}{k:03d}",
                    "parti": party,
                    "namn": f"{party} Member {k}",
                    "fornamn": party,
                    "efternamn": f"Member {k}",
                    "valkrets": "Testlän",
                    "kon": "kvinna" if k % 2 else "man",
                    "fodd": "1970",
                }
            )
            bench += 1
    for person in people[107 - independents:107]:
        person["parti"] = "-"
    return people


def rows_for_vote(
    vote_id: str,
    party_rost: dict[str, str],
    *,
    people: list[dict] | None = None,
    overrides: dict[str, str] | None = None,
    rm: str = "2025/26",
    beteckning: str = "FiU1",
    punkt: str = "1",
    avser: str = "sakfrågan",
    votering: str = "huvud",
    datum: str | None = "2025-10-01",
    hangar_id: str = "1000",
    default: str = "Frånvarande",
) -> list[dict]:
    """Every member votes their party's ``party_rost`` unless overridden by intressent_id."""
    people = people if people is not None else chamber()
    overrides = overrides or {}
    rows = []
    for person in people:
        rost = overrides.get(person["intressent_id"], party_rost.get(person["parti"], default))
        rows.append(
            dict(
                person,
                rm=rm,
                beteckning=beteckning,
                hangar_id=hangar_id,
                votering_id=vote_id,
                punkt=punkt,
                rost=rost,
                avser=avser,
                votering=votering,
                datum=datum,
                valkretsnummer="1",
                iort=None,
            )
        )
    return rows


def bulk_archive(votes: list[list[dict]], *, bom: bool = True) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for rows in votes:
            payload = {"dokvotering": {"votering": rows[0] if len(rows) == 1 else rows}}
            text = json.dumps(payload, ensure_ascii=False)
            archive.writestr(f"{rows[0]['votering_id']}.json", ("﻿" if bom else "") + text)
    return buffer.getvalue()


def register_payload(votes: list[list[dict]]) -> bytes:
    entries = []
    for rows in votes:
        counts = {key: 0 for key in ("Ja", "Nej", "Frånvarande", "Avstår")}
        for row in rows:
            counts[row["rost"]] += 1
        entries.append({"votering_id": rows[0]["votering_id"], **{k: str(v) if v else "" for k, v in counts.items()}})
    return json.dumps({"voteringlista": {"@antal": str(len(entries)), "votering": entries}}).encode()


def voteringlista_rows_payload(rows: list[dict], *, declared: int | None = None) -> bytes:
    api_rows = [{k: v for k, v in row.items() if k not in ("datum", "valkretsnummer")} for row in rows]
    return json.dumps(
        {"voteringlista": {"@antal": str(len(api_rows) if declared is None else declared), "votering": api_rows}}
    ).encode()


def metadata_payload(beteckning: str, datum: str) -> bytes:
    return json.dumps({"votering": {"dokument": {"beteckning": beteckning, "datum": f"{datum} 00:00:00"}}}).encode()


def utskottsforslag_payload(
    dok_id: str,
    rm: str,
    beteckning: str,
    titel: str,
    punkter: dict[str, str] | None = None,
    *,
    doktyp: str = "bet",
    single_as_dict: bool = False,
) -> bytes:
    """An ``utskottsforslag/<dok_id>.json`` answer (CRLF-indented like the API)."""
    items = [
        {"punkt": punkt, "rubrik": rubrik, "beslutstyp": "röstning", "votering_id": "", "rm": rm, "bet": beteckning}
        for punkt, rubrik in (punkter or {}).items()
    ]
    body: dict = {
        "dokument": {
            "dok_id": dok_id,
            "rm": rm,
            "beteckning": beteckning,
            "doktyp": doktyp,
            "titel": titel,
            "dokument_url_html": f"https://data.riksdagen.se/dokument/{dok_id}",
            "html": None,
        }
    }
    if items:
        body["dokutskottsforslag"] = {"utskottsforslag": items[0] if single_as_dict and len(items) == 1 else items}
    return json.dumps({"utskottsforslag": body}, ensure_ascii=False, indent=2).replace("\n", "\r\n").encode("utf-8")


def dokumentlista_payload(hits: list[dict]) -> bytes:
    return json.dumps(
        {"dokumentlista": {"@ms": "12.0", "@traffar": str(len(hits)), "dokument": hits}}, ensure_ascii=False
    ).encode("utf-8")


class FakeHttp:
    """URL -> bytes map; records requests; raises NotFound for unknown URLs."""

    def __init__(self, responses: dict[str, bytes]):
        self.responses = dict(responses)
        self.requests: list[str] = []

    def __call__(self, url: str) -> bytes:
        from scripts.riksdag_votes.fetch import NotFound

        self.requests.append(url)
        if url not in self.responses:
            raise NotFound(url)
        return self.responses[url]
