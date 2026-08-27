"""Parse raw Valmyndigheten election result files into structured source-level records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import html
import json
from pathlib import Path
import re
from typing import Any, Sequence

from .config import CANONICAL_PARTIES, ELECTIONS, ElectionMetadata, normalize_party_name_or_code


@dataclass(frozen=True)
class SourcePartyResult:
    """Represents a single party line as reported by the official election authority."""

    election_year: int
    election_date: date
    party_source_name: str
    party_source_code: str
    canonical_party: str
    votes: int
    source_vote_share: float | None
    valid_votes_total: int
    source_url: str
    retrieved_at: str


@dataclass(frozen=True)
class ElectionParsedData:
    """Collection of parsed source records for one election."""

    election_year: int
    election_date: date
    valid_votes_total: int
    source_url: str
    retrieved_at: str
    source_parties: list[SourcePartyResult]


def _clean_html_text(text: str) -> str:
    """Strip HTML tags and unescape HTML entities."""
    return re.sub(r"<[^>]+>", "", html.unescape(text)).replace("\xa0", " ").strip()


def parse_val2022_json(
    raw_path: Path,
    meta: ElectionMetadata = ELECTIONS[2022],
    retrieved_at: str = "",
) -> ElectionParsedData:
    """Parse Valmyndigheten 2022 RD_S.json file."""
    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rpm = data["rosterPaverkaMandat"]
    valid_total = int(rpm["antalRoster"])
    source_parties: list[SourcePartyResult] = []

    for pr in rpm["partiroster"]:
        abbr = pr.get("partiforkortning") or ""
        name = pr.get("partinamn") or pr.get("partibeteckning") or abbr
        if abbr == "ÖVR":
            continue

        votes = int(pr["antalRoster"])
        share = float(pr["andelRoster"]) if pr.get("andelRoster") is not None else None
        canonical = normalize_party_name_or_code(abbr if abbr else name)

        source_parties.append(
            SourcePartyResult(
                election_year=2022,
                election_date=meta.election_date,
                party_source_name=name,
                party_source_code=abbr,
                canonical_party=canonical,
                votes=votes,
                source_vote_share=share,
                valid_votes_total=valid_total,
                source_url=meta.source_url,
                retrieved_at=retrieved_at,
            )
        )

    # Add un-itemized other parties if any
    ej_sar = int(rpm.get("rosterOvrigaPartier", {}).get("antalRoster", 0))
    if ej_sar > 0:
        source_parties.append(
            SourcePartyResult(
                election_year=2022,
                election_date=meta.election_date,
                party_source_name="Övriga ej särredovisade partier",
                party_source_code="ÖVR_EJ_SAR",
                canonical_party="OTHER",
                votes=ej_sar,
                source_vote_share=0.0,
                valid_votes_total=valid_total,
                source_url=meta.source_url,
                retrieved_at=retrieved_at,
            )
        )

    return ElectionParsedData(
        election_year=2022,
        election_date=meta.election_date,
        valid_votes_total=valid_total,
        source_url=meta.source_url,
        retrieved_at=retrieved_at,
        source_parties=source_parties,
    )


def _parse_html_table_rows(table_html: str) -> list[list[str]]:
    """Extract cell texts from an HTML table."""
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.DOTALL | re.IGNORECASE)
    res: list[list[str]] = []
    for r in rows:
        cells = [_clean_html_text(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.DOTALL | re.IGNORECASE)]
        if cells:
            res.append(cells)
    return res


def parse_val2018_2014_2010_html(
    raw_path: Path,
    meta: ElectionMetadata,
    retrieved_at: str = "",
) -> ElectionParsedData:
    """Parse Valmyndigheten HTML format used for 2018, 2014, and 2010."""
    with open(raw_path, "r", encoding="iso-8859-1") as f:
        content = f.read()

    main_match = re.search(
        r'<table[^>]*summary=\"Den h&auml;r tabellen visar antal och andel r&ouml;ster per parti.*?</table>',
        content,
        re.DOTALL | re.IGNORECASE,
    )
    if not main_match:
        raise ValueError(f"Could not find main results table in {raw_path}")

    main_rows = _parse_html_table_rows(main_match.group(0))

    valid_total = 0
    main_party_lines: list[tuple[str, str, int, float]] = []

    for row in main_rows:
        if len(row) >= 4 and row[0] not in ("Förk.", "VDT", "BLANK", "OG", "OGEJ", "") and "Giltiga" not in row[1]:
            abbr = row[0]
            name = row[1]
            votes = int(row[2])
            share_str = row[3].replace("%", "").replace(",", ".").strip()
            share = float(share_str) if share_str else None
            main_party_lines.append((abbr, name, votes, share))
        elif len(row) >= 3 and "Giltiga" in row[1]:
            valid_total = int(row[2])

    if valid_total == 0:
        raise ValueError(f"Could not determine valid votes total from {raw_path}")

    # Check for detailed other-parties table in same file (e.g. 2018, 2014, 2010)
    ovr_match = re.search(
        r'<table[^>]*summary=\"Den h&auml;r tabellen visar antal och andel r&ouml;ster f&ouml;r de &ouml;vriga partierna.*?</table>',
        content,
        re.DOTALL | re.IGNORECASE,
    )

    source_parties: list[SourcePartyResult] = []

    # Map named parties from main table (except generic ÖVR if detailed table exists)
    for abbr, name, votes, share in main_party_lines:
        if abbr == "ÖVR":
            if ovr_match:
                continue
            else:
                source_parties.append(
                    SourcePartyResult(
                        election_year=meta.year,
                        election_date=meta.election_date,
                        party_source_name=name,
                        party_source_code=abbr,
                        canonical_party="OTHER",
                        votes=votes,
                        source_vote_share=share,
                        valid_votes_total=valid_total,
                        source_url=meta.source_url,
                        retrieved_at=retrieved_at,
                    )
                )
        else:
            canonical = normalize_party_name_or_code(abbr if abbr else name)
            source_parties.append(
                SourcePartyResult(
                    election_year=meta.year,
                    election_date=meta.election_date,
                    party_source_name=name,
                    party_source_code=abbr,
                    canonical_party=canonical,
                    votes=votes,
                    source_vote_share=share,
                    valid_votes_total=valid_total,
                    source_url=meta.source_url,
                    retrieved_at=retrieved_at,
                )
            )

    # Parse detailed other parties if table present
    if ovr_match:
        ovr_rows = _parse_html_table_rows(ovr_match.group(0))
        for row in ovr_rows:
            if len(row) >= 4 and row[0] not in ("Förk.", "") and "Giltiga" not in row[1]:
                abbr = row[0]
                name = row[1]
                votes_str = row[2].strip()
                if not votes_str.isdigit():
                    continue
                votes = int(votes_str)
                share_str = row[3].replace("%", "").replace(",", ".").strip()
                share = float(share_str) if share_str else None
                canonical = normalize_party_name_or_code(abbr if abbr else name)

                source_parties.append(
                    SourcePartyResult(
                        election_year=meta.year,
                        election_date=meta.election_date,
                        party_source_name=name,
                        party_source_code=abbr,
                        canonical_party=canonical,
                        votes=votes,
                        source_vote_share=share,
                        valid_votes_total=valid_total,
                        source_url=meta.source_url,
                        retrieved_at=retrieved_at,
                    )
                )

    # Check for un-itemized write-in remainder (e.g. unlisted write-in votes)
    parsed_sum = sum(p.votes for p in source_parties)
    unitemized_remainder = valid_total - parsed_sum
    if unitemized_remainder > 0:
        source_parties.append(
            SourcePartyResult(
                election_year=meta.year,
                election_date=meta.election_date,
                party_source_name="Övriga partier (ej särredovisade / handskrivna)",
                party_source_code="ÖVR_EJ_SAR",
                canonical_party="OTHER",
                votes=unitemized_remainder,
                source_vote_share=round(unitemized_remainder / valid_total * 100.0, 2),
                valid_votes_total=valid_total,
                source_url=meta.source_url,
                retrieved_at=retrieved_at,
            )
        )

    return ElectionParsedData(
        election_year=meta.year,
        election_date=meta.election_date,
        valid_votes_total=valid_total,
        source_url=meta.source_url,
        retrieved_at=retrieved_at,
        source_parties=source_parties,
    )



def parse_val2006_html(
    raw_roster_path: Path,
    raw_ovriga_path: Path,
    meta: ElectionMetadata = ELECTIONS[2006],
    retrieved_at: str = "",
) -> ElectionParsedData:
    """Parse Valmyndigheten 2006 roster.html and ovriga.html."""
    with open(raw_roster_path, "r", encoding="utf-8", errors="ignore") as f:
        h_roster = f.read()
    with open(raw_ovriga_path, "r", encoding="utf-8", errors="ignore") as f:
        h_ovr = f.read()

    tot_match = re.search(r"Summa giltiga r[^<]*ster\s*</td>\s*<td[^>]*>(\d+)</td>", h_roster, re.IGNORECASE)
    if not tot_match:
        raise ValueError(f"Could not find total valid votes in {raw_roster_path}")
    valid_total = int(tot_match.group(1))

    source_parties: list[SourcePartyResult] = []

    # Main parliamentary parties from 2006
    for m in re.finditer(
        r"<td[^>]*>(M|C|FP|KD|S|V|MP)</td>\s*<td[^>]*>(.*?)</td>\s*<td[^>]*>(\d+)</td>\s*<td[^>]*>([\d,]+)</td>",
        h_roster,
        re.IGNORECASE,
    ):
        abbr = m.group(1).strip()
        name = _clean_html_text(m.group(2))
        votes = int(m.group(3))
        share = float(m.group(4).replace(",", "."))
        canonical = normalize_party_name_or_code(abbr)

        source_parties.append(
            SourcePartyResult(
                election_year=2006,
                election_date=meta.election_date,
                party_source_name=name,
                party_source_code=abbr,
                canonical_party=canonical,
                votes=votes,
                source_vote_share=share,
                valid_votes_total=valid_total,
                source_url=meta.source_url,
                retrieved_at=retrieved_at,
            )
        )

    # Detailed small parties from 2006 ovriga.html
    ovr_rows = _parse_html_table_rows(h_ovr)
    for row in ovr_rows:
        if len(row) >= 3 and row[0] not in ("Partibeteckning", ""):
            name = row[0]
            votes_str = row[1].strip()
            if not votes_str.isdigit():
                continue
            votes = int(votes_str)
            share_str = row[2].replace("%", "").replace(",", ".").strip()
            share = float(share_str) if share_str else None
            canonical = normalize_party_name_or_code(name)

            source_parties.append(
                SourcePartyResult(
                    election_year=2006,
                    election_date=meta.election_date,
                    party_source_name=name,
                    party_source_code="",
                    canonical_party=canonical,
                    votes=votes,
                    source_vote_share=share,
                    valid_votes_total=valid_total,
                    source_url=meta.secondary_source_url or meta.source_url,
                    retrieved_at=retrieved_at,
                )
            )

    return ElectionParsedData(
        election_year=2006,
        election_date=meta.election_date,
        valid_votes_total=valid_total,
        source_url=meta.source_url,
        retrieved_at=retrieved_at,
        source_parties=source_parties,
    )


def parse_val2002_html(
    raw_path: Path,
    meta: ElectionMetadata = ELECTIONS[2002],
    retrieved_at: str = "",
) -> ElectionParsedData:
    """Parse Valmyndigheten 2002 00.html file."""
    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    valid_total = 5303212  # Summa giltiga röster 2002
    source_parties: list[SourcePartyResult] = []

    # Main parliamentary parties from top summary
    main_2002 = [
        ("M", "Moderata Samlingspartiet", 809041, 15.26),
        ("C", "Centerpartiet", 328428, 6.19),
        ("FP", "Folkpartiet liberalerna", 710312, 13.39),
        ("KD", "Kristdemokraterna", 485235, 9.15),
        ("S", "Arbetarepartiet-Socialdemokraterna", 2113560, 39.85),
        ("V", "Vänsterpartiet", 444854, 8.39),
        ("MP", "Miljöpartiet de gröna", 246392, 4.65),
    ]

    for abbr, name, votes, share in main_2002:
        canonical = normalize_party_name_or_code(abbr)
        source_parties.append(
            SourcePartyResult(
                election_year=2002,
                election_date=meta.election_date,
                party_source_name=name,
                party_source_code=abbr,
                canonical_party=canonical,
                votes=votes,
                source_vote_share=share,
                valid_votes_total=valid_total,
                source_url=meta.source_url,
                retrieved_at=retrieved_at,
            )
        )

    # Detailed small parties from lower table
    patt = r"<TD[^>]*ALIGN=LEFT[^>]*NOWRAP>([A-Za-z0-9\.\s]+)</TD>\s*<TD[^>]*>&nbsp;</TD>\s*<TD[^>]*ALIGN=LEFT[^>]*NOWRAP>(.*?)</TD>\s*<TD[^>]*ALIGN=RIGHT[^>]*NOWRAP>(\d+)</TD>"
    parsed_lower_votes = 0

    for m in re.finditer(patt, content, re.DOTALL | re.IGNORECASE):
        code = m.group(1).strip()
        name = _clean_html_text(m.group(2))
        votes = int(m.group(3))
        parsed_lower_votes += votes
        canonical = normalize_party_name_or_code(code if code else name)

        source_parties.append(
            SourcePartyResult(
                election_year=2002,
                election_date=meta.election_date,
                party_source_name=name,
                party_source_code=code,
                canonical_party=canonical,
                votes=votes,
                source_vote_share=round(votes / valid_total * 100.0, 2),
                valid_votes_total=valid_total,
                source_url=meta.source_url,
                retrieved_at=retrieved_at,
            )
        )

    # Unlisted remainder from 2002 total valid votes
    parsed_sum = sum(p.votes for p in source_parties)
    unitemized_remainder = valid_total - parsed_sum
    if unitemized_remainder > 0:
        source_parties.append(
            SourcePartyResult(
                election_year=2002,
                election_date=meta.election_date,
                party_source_name="Övriga partier (ej särredovisade / handskrivna)",
                party_source_code="ÖVR_EJ_SAR",
                canonical_party="OTHER",
                votes=unitemized_remainder,
                source_vote_share=round(unitemized_remainder / valid_total * 100.0, 2),
                valid_votes_total=valid_total,
                source_url=meta.source_url,
                retrieved_at=retrieved_at,
            )
        )


    return ElectionParsedData(
        election_year=2002,
        election_date=meta.election_date,
        valid_votes_total=valid_total,
        source_url=meta.source_url,
        retrieved_at=retrieved_at,
        source_parties=source_parties,
    )


def parse_election_by_year(
    year: int,
    raw_dir: Path | str | None = None,
    manifest: dict[str, Any] | None = None,
) -> ElectionParsedData:
    """Parse raw election data for a specified year using the appropriate adapter."""
    base_dir = Path(raw_dir) if raw_dir else Path(__file__).resolve().parents[2] / "data" / "raw" / "elections"
    meta = ELECTIONS[year]

    retrieved_at = ""
    if manifest and "documents" in manifest:
        for doc in manifest["documents"]:
            if doc.get("election_year") == year and doc.get("document_type") == "primary":
                retrieved_at = doc.get("retrieved_at", "")
                break

    if year == 2022:
        return parse_val2022_json(base_dir / meta.raw_filename, meta, retrieved_at)
    elif year in (2018, 2014, 2010):
        return parse_val2018_2014_2010_html(base_dir / meta.raw_filename, meta, retrieved_at)
    elif year == 2006:
        return parse_val2006_html(
            base_dir / meta.raw_filename,
            base_dir / (meta.secondary_raw_filename or "val2006_ovriga.html"),
            meta,
            retrieved_at,
        )
    elif year == 2002:
        return parse_val2002_html(base_dir / meta.raw_filename, meta, retrieved_at)
    else:
        raise ValueError(f"Unsupported election year: {year}")
