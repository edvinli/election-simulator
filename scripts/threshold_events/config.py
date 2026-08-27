"""Configuration and constants for historical party-election threshold events study."""
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLLS_FILE = PROJECT_ROOT / "data" / "processed" / "pollofpolls" / "swedishpolls_individual_polls.csv"
ELECTIONS_FILE = PROJECT_ROOT / "data" / "processed" / "elections" / "riksdag_election_results.csv"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "threshold_events"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "threshold_events"

# Historical Election Definitions
@dataclass(frozen=True)
class TargetElection:
    year: int
    election_date: date
    canonical_window_days: int = 14
    canonical_inclusion_status: str = "INCLUDED"  # INCLUDED, EXCLUDE_NO_POLLS, EXCLUDE_MISSING_DATES
    source_url: str = ""
    notes: str = ""


TARGET_ELECTIONS: Dict[int, TargetElection] = {
    1991: TargetElection(
        year=1991,
        election_date=date(1991, 9, 15),
        canonical_inclusion_status="EXCLUDE_NO_POLLS",
        source_url="https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3",
        notes="Single pre-election poll ended E-18d (outside canonical 14d; eligible for 21d sensitivity).",
    ),
    1994: TargetElection(
        year=1994,
        election_date=date(1994, 9, 18),
        canonical_inclusion_status="INCLUDED",
        source_url="https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3",
        notes="Included with LOW quality grade (single pollster Ipsos/TEMO in 14-day window).",
    ),
    1998: TargetElection(
        year=1998,
        election_date=date(1998, 9, 20),
        canonical_inclusion_status="EXCLUDE_MISSING_DATES",
        source_url="https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3",
        notes="Excluded from canonical 14d and all sensitivity windows due to missing/corrupt interview dates in SwedishPolls.",
    ),
    2002: TargetElection(
        year=2002,
        election_date=date(2002, 9, 15),
        canonical_inclusion_status="INCLUDED",
        source_url="https://historik.val.se/val/val_02/slutresultat/00R/00.html",
        notes="High-volume multi-house consensus (44 polls, 7 pollsters).",
    ),
    2006: TargetElection(
        year=2006,
        election_date=date(2006, 9, 17),
        canonical_inclusion_status="INCLUDED",
        source_url="https://historik.val.se/val/val2006/slutlig/R/rike/roster.html",
        notes="Multi-house consensus (25 polls, 5 pollsters).",
    ),
    2010: TargetElection(
        year=2010,
        election_date=date(2010, 9, 19),
        canonical_inclusion_status="INCLUDED",
        source_url="https://historik.val.se/val/val2010/slutresultat/R/rike/index.html",
        notes="Multi-house consensus (27 polls, 7 pollsters).",
    ),
    2014: TargetElection(
        year=2014,
        election_date=date(2014, 9, 14),
        canonical_inclusion_status="INCLUDED",
        source_url="https://historik.val.se/val/val2014/slutresultat/R/rike/index.html",
        notes="Multi-house consensus (23 polls, 9 pollsters).",
    ),
    2018: TargetElection(
        year=2018,
        election_date=date(2018, 9, 9),
        canonical_inclusion_status="INCLUDED",
        source_url="https://historik.val.se/val/val2018/slutresultat/R/rike/index.html",
        notes="Multi-house consensus (35 polls, 10 pollsters).",
    ),
    2022: TargetElection(
        year=2022,
        election_date=date(2022, 9, 11),
        canonical_inclusion_status="INCLUDED",
        source_url="https://resultat.val.se/data/resultat/val2022/RD_S.json",
        notes="Multi-house consensus (47 polls, 8 pollsters).",
    ),
}

# Canonical Parliamentary Parties and Named Minor Parties
PARLIAMENTARY_PARTIES = ["M", "C", "L", "KD", "MP", "S", "V", "SD"]
KNOWN_NAMED_MINOR_PARTIES = ["FI", "NYD"]

# Canonical Party Mapping Dictionary
PARTY_SYNONYMS: Dict[str, str] = {
    "M": "M",
    "MODERATERNA": "M",
    "MODERATA SAMLINGSPARTIET": "M",
    "L": "L",
    "LIBERALERNA": "L",
    "FOLKPARTIET": "L",
    "FOLKPARTIET LIBERALERNA": "L",
    "FP": "L",
    "C": "C",
    "CENTERPARTIET": "C",
    "KD": "KD",
    "KRISTDEMOKRATERNA": "KD",
    "KRISTDEMOKRATISKA SAMHÄLLSPARTIET": "KD",
    "KDS": "KD",
    "S": "S",
    "SOCIALDEMOKRATERNA": "S",
    "ARBETAREPARTIET-SOCIALDEMOKRATERNA": "S",
    "V": "V",
    "VÄNSTERPARTIET": "V",
    "VPK": "V",
    "VÄNSTERPARTIET KOMMUNISTERNA": "V",
    "MP": "MP",
    "MILJÖPARTIET": "MP",
    "MILJÖPARTIET DE GRÖNA": "MP",
    "SD": "SD",
    "SVERIGEDEMOKRATERNA": "SD",
    "FI": "FI",
    "F!": "FI",
    "FEMINISTISKT INITIATIV": "FI",
    "NYD": "NYD",
    "NY DEMOKRATI": "NYD",
}


def normalize_party_name(raw_name: str) -> str:
    """Normalize raw party name or abbreviation to canonical string."""
    clean = str(raw_name).strip().upper()
    if clean in PARTY_SYNONYMS:
        return PARTY_SYNONYMS[clean]
    clean_no_punct = clean.replace("-", " ").replace("!", "").strip()
    if clean_no_punct in PARTY_SYNONYMS:
        return PARTY_SYNONYMS[clean_no_punct]
    return clean


# Pre-registered Threshold Bands (Exact Half-Open Intervals)
THRESHOLD_BANDS = [
    ("<2", lambda x: x < 2.0),
    ("2–3", lambda x: 2.0 <= x < 3.0),
    ("3–3.5", lambda x: 3.0 <= x < 3.5),
    ("3.5–4", lambda x: 3.5 <= x < 4.0),
    ("4–4.5", lambda x: 4.0 <= x < 4.5),
    ("4.5–5", lambda x: 4.5 <= x < 5.0),
    ("5–6", lambda x: 5.0 <= x < 6.0),
    (">=6", lambda x: x >= 6.0),
]


def assign_threshold_band(consensus_pct: float) -> str:
    """Assign a polling consensus value to exactly one pre-registered threshold band."""
    for band_name, condition in THRESHOLD_BANDS:
        if condition(consensus_pct):
            return band_name
    return ">=6"


# Objective Quality Grading Rules (Party-Specific)
def grade_episode_quality(
    party_pollster_count: int,
    party_eligible_poll_count: int,
    sample_size_coverage: float,
    metadata_complete: bool,
) -> str:
    """Assign an objective, outcome-blind quality grade to a party-election episode.
    
    Grading rules:
      - HIGH: >= 5 distinct pollsters for this party, >= 15 eligible polls in window, >= 80% sample size coverage, and required dates present.
      - MEDIUM: >= 3 distinct pollsters for this party (or >= 2 pollsters and >= 5 eligible polls in window).
      - LOW: 1 or 2 pollsters for this party (e.g. 1994 single-pollster or low-coverage minor party).
      - EXCLUDE: 0 pollsters for this party in window or incomplete/indeterminate required metadata.

    ``interview_start`` is optional source metadata; the anti-leakage filter
    requires ``interview_end`` and ``publication_date`` only.  Missing
    ``interview_start`` is retained and audited rather than treated as an
    exclusion.
    """
    if party_pollster_count == 0 or not metadata_complete:
        return "EXCLUDE"
    if party_pollster_count >= 5 and party_eligible_poll_count >= 15 and sample_size_coverage >= 0.80:
        return "HIGH"
    if party_pollster_count >= 3 or (party_pollster_count >= 2 and party_eligible_poll_count >= 5):
        return "MEDIUM"
    return "LOW"
