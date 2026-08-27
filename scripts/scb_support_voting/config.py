"""Configuration and constants for SCB Partisympatiundersökningen (PSU) support-voting pipeline.
"""
from pathlib import Path
from typing import Dict, List, Tuple

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "scb_support_voting"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "scb_support_voting"

# SCB API Base URL
SCB_API_BASE_URL = "https://api.scb.se/OV0104/v1/doris/sv/ssd"

# Table Identifiers and Endpoints
SCB_TABLES = {
    "table_a_vote_by_sympathy": {
        "table_id": "Rostningssympati170",
        "path": "ME/ME0201/ME0201A/Rostningssympati170",
        "title": "Röstningssympati (parti man skulle rösta på) efter partisympati (bästa parti)",
        "description": "Cross-tabulation of intended vote (röstningssympati / val idag) conditional on party sympathy (bästa parti)",
        "contents_codes": {
            "000001IS": "estimate_pct",
            "000001IR": "margin_error_pp",
        },
    },
    "table_b_second_choice_by_sympathy": {
        "table_id": "Nastbastaparti190",
        "path": "ME/ME0201/ME0201D/Nastbastaparti190",
        "title": "Näst bästa parti efter partisympati (bästa parti)",
        "description": "Cross-tabulation of second-best party conditional on party sympathy (bästa parti)",
        "contents_codes": {
            "000001IU": "estimate_pct",
            "000001IT": "margin_error_pp",
        },
    },
    "table_c_overall_vote_intention": {
        "table_id": "Vid10",
        "path": "ME/ME0201/ME0201A/Vid10",
        "title": "Valresultat om det varit val idag (PSU)",
        "description": "Overall headline vote intention ('Val idag') by party and wave among decided voters",
        "contents_codes": {
            "ME0201B1": "estimate_pct",
            "ME0201B4": "margin_error_pp",
        },
    },
    "table_d_overall_party_sympathy": {
        "table_id": "Partisympati051",
        "path": "ME/ME0201/ME0201B/Partisympati051",
        "title": "Partisympati efter kön, ålder, parti (urvalsundersökning)",
        "description": "Overall party sympathy ('bästa parti') for total electorate (Kon=TOT, Alder=tot18+)",
        "contents_codes": {
            "ME0201CM": "estimate_pct",
            "ME0201CN": "margin_error_pp",
        },
        # Explicit selectors for Table D to extract the overall electorate distribution
        "fixed_selectors": {
            "Kon": ["TOT"],      # Män och kvinnor totalt
            "Alder": ["tot18+"], # Totalt 18+ år
        },
    },
}

# The exact 29 PSU waves covering 2010M11 -> 2026M05
# Note: PSU was biannual (May + Nov) 2010-2022, and became annual (May only) starting in 2023.
WAVES_2010_2026: List[str] = [
    "2010M11",
    "2011M05", "2011M11",
    "2012M05", "2012M11",
    "2013M05", "2013M11",
    "2014M05", "2014M11",
    "2015M05", "2015M11",
    "2016M05", "2016M11",
    "2017M05", "2017M11",
    "2018M05", "2018M11",
    "2019M05", "2019M11",
    "2020M05", "2020M11",
    "2021M05", "2021M11",
    "2022M05", "2022M11",
    "2023M05",
    "2024M05",
    "2025M05",
    "2026M05",
]

# Canonical 8 parliamentary parties in Sweden
PARLIAMENTARY_PARTIES = ["M", "C", "L", "KD", "MP", "S", "V", "SD"]

# Raw SCB code & label mapping to canonical party & category classification
RAW_CODE_TO_CANONICAL: Dict[str, Tuple[str, str]] = {
    # Parliamentary parties (lower / upper case variants)
    "m": ("M", "parliamentary_party"),
    "M": ("M", "parliamentary_party"),
    "c": ("C", "parliamentary_party"),
    "C": ("C", "parliamentary_party"),
    "l": ("L", "parliamentary_party"),
    "L": ("L", "parliamentary_party"),
    "fp": ("L", "parliamentary_party"),  # Historical Folkpartiet -> L
    "FP": ("L", "parliamentary_party"),
    "kd": ("KD", "parliamentary_party"),
    "KD": ("KD", "parliamentary_party"),
    "mp": ("MP", "parliamentary_party"),
    "MP": ("MP", "parliamentary_party"),
    "s": ("S", "parliamentary_party"),
    "S": ("S", "parliamentary_party"),
    "v": ("V", "parliamentary_party"),
    "V": ("V", "parliamentary_party"),
    "sd": ("SD", "parliamentary_party"),
    "SD": ("SD", "parliamentary_party"),
    
    # Historical / other named parties
    "nyd": ("NYD", "historical_party"),  # Ny Demokrati (in historical SCB tables)
    "NYD": ("NYD", "historical_party"),
    "övr": ("OTHER", "other_party"),
    "ÖVR": ("OTHER", "other_party"),
    "övriga": ("OTHER", "other_party"),
    "övriga partier": ("OTHER", "other_party"),

    # Non-party categories
    "ingen sympati/vet ej": ("NO_SYMPATHY_OR_DONT_KNOW", "no_sympathy"),
    "hela väljarkåren": ("TOTAL_ELECTORATE", "total_electorate"),
    "blankt": ("BLANK_VOTE", "blank_vote"),
    "vet ej": ("DONT_KNOW", "dont_know"),
    "inget parti": ("NO_SECOND_CHOICE", "no_second_choice"),
}


def classify_category(raw_code: str, raw_label: str) -> Tuple[str, str]:
    """Classify a category given its raw code and label.
    
    Returns:
        (canonical_name, category_type)
    """
    code_key = str(raw_code).strip()
    if code_key in RAW_CODE_TO_CANONICAL:
        return RAW_CODE_TO_CANONICAL[code_key]
    
    label_key = str(raw_label).strip().lower()
    if label_key in RAW_CODE_TO_CANONICAL:
        return RAW_CODE_TO_CANONICAL[label_key]
    
    # Handle composite blocks in Vid10 (e.g. 'c+fp+m+kd', 'borgerligt block')
    if any(b in label_key for b in ["block", "+"]):
        return (code_key, "composite_block")
    
    # Default fallback
    return (code_key.upper(), "unclassified")


def parse_wave_period(wave: str) -> Tuple[str, str]:
    """Parse SCB wave format (e.g. '2010M11') to (period, date_str).
    
    Returns:
        (period '2010-11', date_str '2010-11-01')
    """
    if "M" in wave:
        parts = wave.split("M")
        year = parts[0]
        month = parts[1].zfill(2)
        return f"{year}-{month}", f"{year}-{month}-01"
    return wave, wave


# SCB Methodology metadata notes
METHODOLOGY_METADATA = {
    "agency": "Statistiska centralbyrån (SCB)",
    "survey_name": "Partisympatiundersökningen (PSU)",
    "panel_coverage": "2010M11 - 2026M05",
    "number_of_waves": len(WAVES_2010_2026),
    "frequency_change_note": (
        "PSU was conducted semiannually in May and November from 1972 through November 2022. "
        "Starting in 2023, SCB changed the frequency to an annual survey conducted exclusively in May."
    ),
    "weighting_revision_2020": (
        "In 2020, SCB implemented an updated calibration and weighting model for 'Val idag' "
        "and party sympathy estimates. SCB recalculated historical 'Vid10' (Val idag) estimates "
        "back to 2010 using this revised methodology. Other tables may reflect the weighting methodology "
        "in effect at their respective publication periods."
    ),
    "concept_distinction": (
        "CRITICAL DISTINCTION: 'Partisympati' (bästa parti) measures emotional / ideological party preference, "
        "while 'Röstningssympati / Val idag' measures intended vote in a hypothetical parliamentary election today "
        "(reweighted to expected electoral turnout). Aggregate differences between overall sympathy and overall "
        "vote intention must not be interpreted directly as tactical voting."
    ),
}
