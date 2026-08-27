"""Official historical election results loader for Sweden (1991–2022).

Combines official Valmyndigheten election results (2002–2022) with official SCB
historical election records (1991–1998), preserving exact vote counts and valid totals.
"""
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from scripts.threshold_events.config import (
    ELECTIONS_FILE,
    RAW_DATA_DIR,
    TARGET_ELECTIONS,
    normalize_party_name,
)


@dataclass(frozen=True)
class OfficialPartyResult:
    election_year: int
    election_date: date
    party: str
    party_raw: str
    votes: int
    valid_votes_total: int
    vote_share_pct: float
    source_url: str


# Official SCB / Valmyndigheten Historical Results for 1991, 1994, 1998
# Source: SCB Statistikdatabasen table ME0104T3 & Valmyndigheten Historical Archives
HISTORICAL_SCB_RESULTS_1991_1998 = [
    # 1991 (1991-09-15) - Total valid votes: 5,470,761
    {"year": 1991, "party_raw": "S", "party": "S", "votes": 2062761, "valid_votes": 5470761, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},
    {"year": 1991, "party_raw": "M", "party": "M", "votes": 1199394, "valid_votes": 5470761, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},
    {"year": 1991, "party_raw": "FP", "party": "L", "votes": 499356, "valid_votes": 5470761, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},
    {"year": 1991, "party_raw": "C", "party": "C", "votes": 465175, "valid_votes": 5470761, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},
    {"year": 1991, "party_raw": "KDS", "party": "KD", "votes": 390351, "valid_votes": 5470761, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},
    {"year": 1991, "party_raw": "NYD", "party": "NYD", "votes": 368281, "valid_votes": 5470761, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},
    {"year": 1991, "party_raw": "V", "party": "V", "votes": 246905, "valid_votes": 5470761, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},
    {"year": 1991, "party_raw": "MP", "party": "MP", "votes": 185051, "valid_votes": 5470761, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},
    {"year": 1991, "party_raw": "SD", "party": "SD", "votes": 4968, "valid_votes": 5470761, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},
    {"year": 1991, "party_raw": "ÖVRIGA", "party": "OTHER", "votes": 48519, "valid_votes": 5470761, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},

    # 1994 (1994-09-18) - Total valid votes: 5,555,540
    {"year": 1994, "party_raw": "S", "party": "S", "votes": 2513905, "valid_votes": 5555540, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},
    {"year": 1994, "party_raw": "M", "party": "M", "votes": 1243253, "valid_votes": 5555540, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},
    {"year": 1994, "party_raw": "C", "party": "C", "votes": 425153, "valid_votes": 5555540, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},
    {"year": 1994, "party_raw": "FP", "party": "L", "votes": 399556, "valid_votes": 5555540, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},
    {"year": 1994, "party_raw": "V", "party": "V", "votes": 342988, "valid_votes": 5555540, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},
    {"year": 1994, "party_raw": "MP", "party": "MP", "votes": 279042, "valid_votes": 5555540, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},
    {"year": 1994, "party_raw": "KDS", "party": "KD", "votes": 225974, "valid_votes": 5555540, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},
    {"year": 1994, "party_raw": "NYD", "party": "NYD", "votes": 68663, "valid_votes": 5555540, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},
    {"year": 1994, "party_raw": "SD", "party": "SD", "votes": 13950, "valid_votes": 5555540, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},
    {"year": 1994, "party_raw": "ÖVRIGA", "party": "OTHER", "votes": 43056, "valid_votes": 5555540, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},

    # 1998 (1998-09-20) - Total valid votes: 5,260,109
    {"year": 1998, "party_raw": "S", "party": "S", "votes": 1914426, "valid_votes": 5260109, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},
    {"year": 1998, "party_raw": "M", "party": "M", "votes": 1204926, "valid_votes": 5260109, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},
    {"year": 1998, "party_raw": "V", "party": "V", "votes": 631011, "valid_votes": 5260109, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},
    {"year": 1998, "party_raw": "KD", "party": "KD", "votes": 618033, "valid_votes": 5260109, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},
    {"year": 1998, "party_raw": "C", "party": "C", "votes": 269762, "valid_votes": 5260109, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},
    {"year": 1998, "party_raw": "FP", "party": "L", "votes": 248076, "valid_votes": 5260109, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},
    {"year": 1998, "party_raw": "MP", "party": "MP", "votes": 236699, "valid_votes": 5260109, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},
    {"year": 1998, "party_raw": "SD", "party": "SD", "votes": 19624, "valid_votes": 5260109, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},
    {"year": 1998, "party_raw": "ÖVRIGA", "party": "OTHER", "votes": 117552, "valid_votes": 5260109, "source_url": "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C/ME0104T3"},
]


def load_all_official_election_results() -> Dict[int, Dict[str, OfficialPartyResult]]:
    """Load official election results for all target elections from authoritative source datasets.
    
    Returns:
        {election_year: {canonical_party: OfficialPartyResult}}
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_archive_file = RAW_DATA_DIR / "official_election_results_archive.json"
    
    results: Dict[int, Dict[str, OfficialPartyResult]] = {}
    
    # 1. Load 1991, 1994, 1998 SCB results
    for row in HISTORICAL_SCB_RESULTS_1991_1998:
        yr = row["year"]
        if yr not in results:
            results[yr] = {}
        elec_date = TARGET_ELECTIONS[yr].election_date
        share = round(100.0 * row["votes"] / row["valid_votes"], 4)
        results[yr][row["party"]] = OfficialPartyResult(
            election_year=yr,
            election_date=elec_date,
            party=row["party"],
            party_raw=row["party_raw"],
            votes=row["votes"],
            valid_votes_total=row["valid_votes"],
            vote_share_pct=share,
            source_url=row["source_url"],
        )
        
    # 2. Load 2002–2022 from processed riksdag_election_results.csv
    if ELECTIONS_FILE.exists():
        df = pd.read_csv(ELECTIONS_FILE)
        for _, r in df.iterrows():
            yr = int(r["election_year"])
            if yr not in results:
                results[yr] = {}
            p_canon = normalize_party_name(r["party"])
            elec_date = date.fromisoformat(r["election_date"])
            votes = int(r["votes"])
            valid_tot = int(r["valid_votes_total"])
            share = round(100.0 * votes / valid_tot, 4) if valid_tot > 0 else float(r["vote_share"])
            results[yr][p_canon] = OfficialPartyResult(
                election_year=yr,
                election_date=elec_date,
                party=p_canon,
                party_raw=str(r["party_source_name"]),
                votes=votes,
                valid_votes_total=valid_tot,
                vote_share_pct=share,
                source_url=str(r["source_url"]),
            )
            
    # Save a write-once raw archive snapshot.  Re-running the loader should be
    # idempotent, while a changed source must not silently overwrite evidence
    # used by a previous evaluation.  A deliberate source revision can be
    # archived under a new path after review.
    serializable = {}
    for yr, p_dict in results.items():
        serializable[yr] = {
            p: {
                "election_year": res.election_year,
                "election_date": res.election_date.isoformat(),
                "party": res.party,
                "party_raw": res.party_raw,
                "votes": res.votes,
                "valid_votes_total": res.valid_votes_total,
                "vote_share_pct": res.vote_share_pct,
                "source_url": res.source_url,
            }
            for p, res in p_dict.items()
        }
    rendered_archive = json.dumps(serializable, indent=2, ensure_ascii=False)
    if raw_archive_file.exists():
        existing_archive = raw_archive_file.read_text(encoding="utf-8")
        if existing_archive != rendered_archive:
            raise RuntimeError(
                f"Immutable official-results archive differs from current source: "
                f"{raw_archive_file}. Preserve it and write a reviewed revision "
                "under a new path."
            )
    else:
        # Exclusive creation protects against replacing a concurrently-created
        # archive.  The file is intentionally never opened with mode ``w``.
        with open(raw_archive_file, "x", encoding="utf-8") as f:
            f.write(rendered_archive)
        
    return results
