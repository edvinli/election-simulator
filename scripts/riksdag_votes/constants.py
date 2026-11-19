"""Named constants for the riksdag roll-call pipeline.

Every threshold the analysis depends on lives here so that the module
docstring, ``docs/riksdag_votes.md`` and the emitted JSON can all cite one
value. Nothing here is imported from the election forecast.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "raw" / "riksdag_votes"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "riksdag_votes"
MANIFEST_FILENAME = "manifest.json"
INDEX_FILENAME = "index.json"
# Paths the scheduled workflow (.github/workflows/riksdag-votes.yml) may write:
# the committed outputs in this repository and the published copy on the
# website (edvinli/edvinli.github.io, branch master). Nothing else.
PROCESSED_DIR_RELATIVE = "data/processed/riksdag_votes"
WEBSITE_DIR_RELATIVE = "files/riksdag"
WORKFLOW_CONCURRENCY_GROUP = "riksdag-votes"

SCHEMA_VERSION = "1.0"
ROLE_SESSION_VOTES = "riksdag_votes_session_votes"
ROLE_SUMMARY = "riksdag_votes_summary"
ROLE_MANIFEST = "riksdag_votes_source_manifest"
ROLE_INDEX = "riksdag_votes_index"

# --- Sources -----------------------------------------------------------------
BULK_URL_TEMPLATE = "https://data.riksdagen.se/dataset/votering/votering-{code}.json.zip"
VOTERINGLISTA_URL = "https://data.riksdagen.se/voteringlista/"
VOTE_METADATA_URL_TEMPLATE = "https://data.riksdagen.se/votering/{votering_id}/json"
# Page size requested from voteringlista. Every response is checked against
# its own ``@antal`` so a truncated answer fails instead of being trusted.
VOTERINGLISTA_PAGE_SIZE = 20000
# Vote titles: the committee's proposals for one document -- its title and one
# heading (rubrik) per förslagspunkt -- about 20 kB, one request per document.
UTSKOTTSFORSLAG_URL_TEMPLATE = "https://data.riksdagen.se/utskottsforslag/{dok_id}.json"
# Fallback lookup of a document's dok_id by riksmöte + beteckning.
DOKUMENTLISTA_URL = "https://data.riksdagen.se/dokumentlista/"
DOCUMENT_URL_TEMPLATE = "https://data.riksdagen.se/dokument/{dok_id}"
HTTP_TIMEOUT_SECONDS = 120
HTTP_ATTEMPTS = 5
# Politeness: at most one request per HTTP_MIN_INTERVAL_SECONDS (4 requests/s)
# and exponential backoff (2, 4, 8, 16 s, or the server's Retry-After, capped)
# after a 429, a 5xx or a network error.
HTTP_MIN_INTERVAL_SECONDS = 0.25
HTTP_BACKOFF_BASE_SECONDS = 2.0
HTTP_BACKOFF_MAX_SECONDS = 60.0
USER_AGENT = "riksdag-votes-pipeline/1.0"

# --- Chamber -----------------------------------------------------------------
SEATS = 349
JA = "Ja"
NEJ = "Nej"
AVSTAR = "Avstår"
FRANVARANDE = "Frånvarande"
ROST_VALUES = (JA, NEJ, AVSTAR, FRANVARANDE)
CAST_VALUES = (JA, NEJ, AVSTAR)
# One-character codes for the compact per-vote ballot strings in votes.json.
ROST_CODES = {JA: "J", NEJ: "N", AVSTAR: "A", FRANVARANDE: "F"}
# Numeric encoding for the map (rule 3). Frånvarande is missing, never 0.
ROST_NUMERIC = {JA: 1.0, AVSTAR: 0.0, NEJ: -1.0}

INDEPENDENT_PARTY = "-"
# Display/ordering convention for the eight riksdag parties (the same order the
# rest of the repository uses). Unknown parties are appended alphabetically.
PARTY_ORDER = ("M", "L", "C", "KD", "S", "V", "MP", "SD")

# --- Vote selection (rule 1) -------------------------------------------------
ANALYSIS_VOTERING = "huvud"
ANALYSIS_AVSER = "sakfrågan"
AVSER_VALUES = ("sakfrågan", "motivfrågan", "frågan")

# --- Agreement ---------------------------------------------------------------
# A matrix cell is shown as a share only when it rests on at least this many
# votes; the count is always kept.
MIN_AGREEMENT_CELL_N = 10

# --- Map (rule 3) ------------------------------------------------------------
# NOMINATE convention: drop votes whose minority side is below 2.5 % of the
# Ja+Nej ballots cast.
NEAR_UNANIMOUS_MINORITY_SHARE = 0.025
# A member is placed only with at least this many cast (Ja/Nej/Avstår) map votes.
MIN_MP_MAP_VOTES = 50
# The party geometry is fitted only when at least this many map votes remain.
MIN_MAP_VOTES_FOR_FIT = 20
MAP_DIMENSIONS = 2
TOP_LOADING_VOTES = 10
# Sign convention: axis k is oriented so that ANCHOR[k] has a positive score.
# If the anchor's score is numerically zero the first party in PARTY_ORDER with
# a non-zero score is made positive instead. The anchors are a convention for
# reproducibility only; they carry no left/right meaning.
MAP_AXIS_ANCHORS = ("M", "SD")
ROUND_DIGITS = 4

# --- Terms -------------------------------------------------------------------
# Four-year electoral terms counted from the 2022 election: "2022-26" is
# 2022/23..2025/26, "2026-30" is 2026/27..2029/30, and so on. A riksmöte is
# treated as begun on 1 September of its first year.
FIRST_TERM_START = 2022
LAST_TERM_START = 2050
SESSION_START = (9, 1)  # (month, day)


def session_label(start_year: int) -> str:
    """``2026`` -> ``2026/27``."""
    return f"{start_year}/{(start_year + 1) % 100:02d}"


TERMS = {
    f"{start}-{(start + 4) % 100:02d}": tuple(session_label(year) for year in range(start, start + 4))
    for start in range(FIRST_TERM_START, LAST_TERM_START + 1, 4)
}

# --- Committees --------------------------------------------------------------
OTHER_COMMITTEE = "övrigt"
COMMITTEES = {
    "AU": "Arbetsmarknadsutskottet",
    "CU": "Civilutskottet",
    "FiU": "Finansutskottet",
    "FöU": "Försvarsutskottet",
    "JuU": "Justitieutskottet",
    "KU": "Konstitutionsutskottet",
    "KrU": "Kulturutskottet",
    "MJU": "Miljö- och jordbruksutskottet",
    "NU": "Näringsutskottet",
    "SfU": "Socialförsäkringsutskottet",
    "SkU": "Skatteutskottet",
    "SoU": "Socialutskottet",
    "TU": "Trafikutskottet",
    "UbU": "Utbildningsutskottet",
    "UU": "Utrikesutskottet",
    "UFöU": "Sammansatta utrikes- och försvarsutskottet",
}

# --- Caveats -----------------------------------------------------------------
CAVEAT_KVITTNING = (
    "Frånvarande includes kvittning (pairing), where a member stays out by "
    "agreement with a member of the opposing side so the balance is unchanged. "
    "The open data does not distinguish kvittning from other absence, so "
    "participation is not a measure of diligence and is not ranked."
)
CAVEAT_FELTRYCKNING = (
    "Votes against the party line are counted as recorded. Isolated cases can be "
    "feltryckningar (accidental button presses), which the riksdag keeps in the "
    "official record and the API reports unchanged."
)
CAVEAT_MAP = (
    "Coordinates come from a principal-component fit to party lines; axes are "
    "statistical summaries of voting patterns, not a left/right or any other "
    "named scale. Interpret them through the listed top-loading votes."
)
