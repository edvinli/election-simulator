"""Configuration, constants, canonical party mappings, and election metadata."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping


DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "elections"
DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "elections"


CANONICAL_PARTIES: tuple[str, ...] = (

    "M",
    "L",
    "C",
    "KD",
    "S",
    "V",
    "MP",
    "SD",
    "FI",
    "OTHER",
)

PARLIAMENTARY_PARTIES: tuple[str, ...] = (
    "M",
    "L",
    "C",
    "KD",
    "S",
    "V",
    "MP",
    "SD",
)


@dataclass(frozen=True)
class ElectionMetadata:
    year: int
    election_date: date
    source_url: str
    secondary_source_url: str | None = None
    raw_filename: str = ""
    secondary_raw_filename: str | None = None


ELECTIONS: dict[int, ElectionMetadata] = {
    2022: ElectionMetadata(
        year=2022,
        election_date=date(2022, 9, 11),
        source_url="https://resultat.val.se/data/resultat/val2022/RD_S.json",
        raw_filename="val2022_RD_S.json",
    ),
    2018: ElectionMetadata(
        year=2018,
        election_date=date(2018, 9, 9),
        source_url="https://historik.val.se/val/val2018/slutresultat/R/rike/index.html",
        raw_filename="val2018_slutresultat.html",
    ),
    2014: ElectionMetadata(
        year=2014,
        election_date=date(2014, 9, 14),
        source_url="https://historik.val.se/val/val2014/slutresultat/R/rike/index.html",
        raw_filename="val2014_slutresultat.html",
    ),
    2010: ElectionMetadata(
        year=2010,
        election_date=date(2010, 9, 19),
        source_url="https://historik.val.se/val/val2010/slutresultat/R/rike/index.html",
        raw_filename="val2010_slutresultat.html",
    ),
    2006: ElectionMetadata(
        year=2006,
        election_date=date(2006, 9, 17),
        source_url="https://historik.val.se/val/val2006/slutlig/R/rike/roster.html",
        secondary_source_url="https://historik.val.se/val/val2006/slutlig/R/rike/ovriga.html",
        raw_filename="val2006_roster.html",
        secondary_raw_filename="val2006_ovriga.html",
    ),
    2002: ElectionMetadata(
        year=2002,
        election_date=date(2002, 9, 15),
        source_url="https://historik.val.se/val/val_02/slutresultat/00R/00.html",
        raw_filename="val2002_slutresultat.html",
    ),
}


PARTY_NAME_MAPPINGS: Mapping[str, str] = {
    # Moderaterna
    "M": "M",
    "MODERATERNA": "M",
    "MODERATA SAMLINGSPARTIET": "M",
    # Liberalerna / Folkpartiet
    "L": "L",
    "LIBERALERNA": "L",
    "LIBERALERNA (TIDIGARE FOLKPARTIET)": "L",
    "FP": "L",
    "FOLKPARTIET": "L",
    "FOLKPARTIET LIBERALERNA": "L",
    # Centerpartiet
    "C": "C",
    "CENTERPARTIET": "C",
    # Kristdemokraterna
    "KD": "KD",
    "KRISTDEMOKRATERNA": "KD",
    "KRISTDEMOKRATISKA SAMHÄLLSPARTIET": "KD",
    "KDS": "KD",
    # Socialdemokraterna
    "S": "S",
    "SOCIALDEMOKRATERNA": "S",
    "ARBETAREPARTIET-SOCIALDEMOKRATERNA": "S",
    "SVERIGES SOCIALDEMOKRATISKA ARBETAREPARTI": "S",
    # Vänsterpartiet
    "V": "V",
    "VÄNSTERPARTIET": "V",
    "VÄNSTERPARTIET KOMMUNISTERNA": "V",
    "VPK": "V",
    # Miljöpartiet
    "MP": "MP",
    "MILJÖPARTIET": "MP",
    "MILJÖPARTIET DE GRÖNA": "MP",
    # Sverigedemokraterna
    "SD": "SD",
    "SVERIGEDEMOKRATERNA": "SD",
    # Feministiskt initiativ
    "FI": "FI",
    "F!": "FI",
    "FEMINISTISKT INITIATIV": "FI",
}


def normalize_party_name_or_code(name_or_code: str) -> str:
    """Normalize a raw source party name or abbreviation to canonical code (or 'OTHER')."""
    cleaned = name_or_code.strip().upper()
    # Direct lookup
    if cleaned in PARTY_NAME_MAPPINGS:
        return PARTY_NAME_MAPPINGS[cleaned]

    # Simplified lookup (strip punctuation)
    simplified = cleaned.replace("-", " ").replace("!", "").strip()
    if simplified in PARTY_NAME_MAPPINGS:
        return PARTY_NAME_MAPPINGS[simplified]

    return "OTHER"
