"""Fetch raw geographical and electoral baseline data for GeographicProjection v1."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request

from .config import DEFAULT_RAW_GEOGRAPHY_DIR, OFFICIAL_CONSTITUENCY_CODES


def fetch_raw_geography_data(raw_dir: Path | str | None = None) -> Path:
    """Fetch 2014 constituency results and 2026 electorate data from Valmyndigheten."""
    target_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_GEOGRAPHY_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, Any]] = []

    # 1. Fetch 2026 Electorate dataset
    url_2026 = "http://val.se/download/18.1a2972da19f159e73fd3b4a/1787064446298/antal-rostberattigade-per-valdistrikt-uppdelat-pa-kon-och-alder-kvalifikationsdagen-14-augusti-2026-val-till-riksdagen.xlsx"
    dest_2026 = target_dir / "antal_rostberattigade_2026_riksdagen.xlsx"
    print(f"Fetching 2026 electorate data from {url_2026} ...")

    req = urllib.request.Request(url_2026, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data_2026 = resp.read()

    with open(dest_2026, "wb") as f:
        f.write(data_2026)

    sha_2026 = hashlib.sha256(data_2026).hexdigest()
    manifest_entries.append({
        "filename": "antal_rostberattigade_2026_riksdagen.xlsx",
        "source_url": url_2026,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "byte_count": len(data_2026),
        "sha256": sha_2026,
    })
    print(f"  -> Saved {dest_2026} ({len(data_2026):,d} bytes, sha256: {sha_2026[:12]}...)")

    # 2. Fetch 2014 Constituency Results (all 29 constituencies)
    print("Fetching 2014 constituency results from historik.val.se ...")
    votes_2014_payload: dict[str, Any] = {}

    for code in OFFICIAL_CONSTITUENCY_CODES:
        url_c = f"https://historik.val.se/val/val2014/slutresultat/R/rvalkrets/{code}/index.html"
        req_c = urllib.request.Request(url_c, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req_c, timeout=30) as resp_c:
            html_c = resp_c.read().decode("iso-8859-1")

        tables = re.findall(r"<table[^>]*>(.*?)</table>", html_c, re.DOTALL | re.IGNORECASE)
        t3 = tables[3]
        rows_3 = re.findall(r"<tr[^>]*>(.*?)</tr>", t3, re.DOTALL | re.IGNORECASE)

        c_votes: dict[str, int] = {}
        for r in rows_3[1:]:
            cells = [
                re.sub(r"<[^>]+>", "", c).strip()
                for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.DOTALL | re.IGNORECASE)
            ]
            if len(cells) >= 3:
                party_raw = cells[0]
                votes_val = int(cells[2].replace(" ", ""))
                c_votes[party_raw] = votes_val

        # Also extract eligible voters from summary table
        eligible_val = c_votes.get("&nbsp;", 0)
        valid_val = sum(v for k, v in c_votes.items() if k not in ["&nbsp;", "BLANK", "OG", "VDT"])

        votes_2014_payload[code] = {
            "source_url": url_c,
            "party_votes": {k: v for k, v in c_votes.items() if k not in ["&nbsp;", "BLANK", "OG", "VDT"]},
            "eligible_voters": eligible_val,
            "valid_votes": valid_val,
            "total_votes_cast": c_votes.get("VDT", 0),
        }

    data_2014_bytes = json.dumps(votes_2014_payload, indent=2, ensure_ascii=False).encode("utf-8")
    dest_2014 = target_dir / "val2014_constituencies_votes.json"
    with open(dest_2014, "wb") as f:
        f.write(data_2014_bytes)

    sha_2014 = hashlib.sha256(data_2014_bytes).hexdigest()
    manifest_entries.append({
        "filename": "val2014_constituencies_votes.json",
        "source_url": "https://historik.val.se/val/val2014/slutresultat/R/rvalkrets/XX/index.html (29 constituencies)",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "byte_count": len(data_2014_bytes),
        "sha256": sha_2014,
    })
    print(f"  -> Saved {dest_2014} ({len(data_2014_bytes):,d} bytes, sha256: {sha_2014[:12]}...)")

    # Write Manifest
    manifest_path = target_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"sources": manifest_entries}, f, indent=2, ensure_ascii=False)

    print(f"Manifest written to {manifest_path}")
    return target_dir
