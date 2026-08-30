"""Deterministic retrieval of official Valmyndigheten historical Riksdag results (research only).

Fetches, for the Part-2B seat-evaluation extension:

* 2006 constituency party votes + certified constituency mandates
      historik.val.se/val/val2006/slutlig/R/riksdagsvalkrets/<cc>/roster.html
* 2010 constituency party votes
      historik.val.se/val/val2010/slutresultat/R/rvalkrets/<cc>/index.html
* 2010 certified constituency mandates
      historik.val.se/val/val2010/slutresultat/R/rvalkrets/<cc>/valda.html
* 2014 certified constituency mandates
      historik.val.se/val/val2014/slutresultat/R/rvalkrets/<cc>/valda.html

2014 constituency *votes* are already in the repository
(``data/processed/geography/constituency_party_votes_2014_2022.csv``) and are not
re-fetched.

Nothing under ``data/`` is written or modified. Every page is stored verbatim with
its URL, retrieval timestamp and SHA-256 in ``raw/fetch_manifest.json``.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.geography.config import OFFICIAL_CONSTITUENCY_CODES

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
USER_AGENT = "Mozilla/5.0 (election-simulator research; Part 2B historical seat extension)"

PAGE_SPECS: list[dict[str, str]] = [
    {
        "key": "votes_mandates_2006",
        "year": "2006",
        "purpose": "2006 constituency party votes (geography baseline for the 2010 target) and certified 2006 constituency mandates",
        "url_template": "https://historik.val.se/val/val2006/slutlig/R/riksdagsvalkrets/{cc}/roster.html",
        "parsing_notes": (
            "ISO-8859-1. Table[1] is 'Röstfördelning' with a two-row header; data rows are "
            "[förk, parti, antal_2006, andel_2006, mandat_2006, antal_2002, andel_2002, mandat_2002]. "
            "Only the 2006 columns are used. Table[2] 'Röstredovisning' row 'Summa giltiga röster' "
            "gives the constituency valid-vote total (2006 column)."
        ),
    },
    {
        "key": "ovriga_2006",
        "year": "2006",
        "purpose": "2006 per-constituency breakdown of the aggregate 'ÖVR' row, needed because the 2006 result pages do not break out Sverigedemokraterna",
        "url_template": "https://historik.val.se/val/val2006/slutlig/R/riksdagsvalkrets/{cc}/ovriga.html",
        "parsing_notes": (
            "UTF-8 (unlike the 2010/2014 pages, which are ISO-8859-1). Single table: "
            "[partibeteckning, antal, andel]. 'Sverigedemokraterna' is extracted as SD; "
            "every other row stays inside REST."
        ),
    },
    {
        "key": "votes_2010",
        "year": "2010",
        "purpose": "2010 constituency party votes (target results, and geography baseline for the 2014 target)",
        "url_template": "https://historik.val.se/val/val2010/slutresultat/R/rvalkrets/{cc}/index.html",
        "parsing_notes": (
            "ISO-8859-1. Same layout as the 2014 pages already parsed by "
            "scripts/geography/fetch.py: table[3] rows are [förk, parti, antal, andel, ...]; "
            "vote count is cell index 2. Non-party rows (BLANK, OG, VDT, &nbsp;) are excluded."
        ),
    },
    {
        "key": "mandates_2010",
        "year": "2010",
        "purpose": "Certified 2010 per-constituency per-party mandates (golden target)",
        "url_template": "https://historik.val.se/val/val2010/slutresultat/R/rvalkrets/{cc}/valda.html",
        "parsing_notes": (
            "ISO-8859-1. Table[1] 'Mandatfördelning' rows are "
            "[förk, parti, mandat_2010, +/-, mandat_2006]; the final row is the 'Totalt' control total."
        ),
    },
    {
        "key": "mandates_2014",
        "year": "2014",
        "purpose": "Certified 2014 per-constituency per-party mandates (golden target)",
        "url_template": "https://historik.val.se/val/val2014/slutresultat/R/rvalkrets/{cc}/valda.html",
        "parsing_notes": "ISO-8859-1. Same layout as the 2010 valda pages.",
    },
]


def fetch(url: str, attempts: int = 4) -> bytes:
    last: Exception | None = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=45) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - transient network failures are expected
            last = exc
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last}")


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []

    for spec in PAGE_SPECS:
        out_dir = RAW / spec["key"]
        out_dir.mkdir(parents=True, exist_ok=True)
        for cc in OFFICIAL_CONSTITUENCY_CODES:
            url = spec["url_template"].format(cc=cc)
            dest = out_dir / f"{cc}.html"
            if dest.exists():
                payload = dest.read_bytes()
                retrieved = "cached (see prior manifest entry)"
            else:
                payload = fetch(url)
                dest.write_bytes(payload)
                retrieved = datetime.now(timezone.utc).isoformat()
                time.sleep(0.25)
            entries.append(
                {
                    "key": spec["key"],
                    "election_year": int(spec["year"]),
                    "constituency_code": cc,
                    "file": str(dest.relative_to(HERE)),
                    "source_url": url,
                    "retrieved_at_utc": retrieved,
                    "byte_count": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "purpose": spec["purpose"],
                    "parsing_notes": spec["parsing_notes"],
                }
            )
            print(f"  {spec['key']} {cc}: {len(payload):,d} bytes")

    manifest = {
        "schema_version": "1.0",
        "status": "RESEARCH ONLY - nothing under data/ is written or modified",
        "publisher": "Valmyndigheten (historik.val.se) - official historical election results",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pages": entries,
    }
    (RAW / "fetch_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {RAW / 'fetch_manifest.json'} with {len(entries)} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
