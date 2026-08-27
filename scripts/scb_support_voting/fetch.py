"""Raw data fetcher for SCB Partisympatiundersökningen (PSU) tables.

Downloads authoritative table metadata, queries, and responses directly from
SCB Statistikdatabasen (SSD API) and creates an immutable raw archive with SHA-256 provenance.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict
import urllib.request

from scripts.scb_support_voting.config import (
    METHODOLOGY_METADATA,
    RAW_DATA_DIR,
    SCB_API_BASE_URL,
    SCB_TABLES,
    WAVES_2010_2026,
)


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def fetch_scb_metadata(table_path: str) -> Dict[str, Any]:
    """Fetch table schema and variable metadata from SCB API."""
    url = f"{SCB_API_BASE_URL}/{table_path}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AntigravitySCBClient/1.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def submit_scb_query(table_path: str, query_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Submit a POST query to SCB API and return response data."""
    url = f"{SCB_API_BASE_URL}/{table_path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(query_dict).encode("utf-8"),
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AntigravitySCBClient/1.0",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_table_query(table_meta: Dict[str, Any], table_conf: Dict[str, Any]) -> Dict[str, Any]:
    """Construct deterministic JSON query payload for an SCB table.
    
    Explicitly filters:
    - Tid: exact 29 waves (2010M11 -> 2026M05)
    - ContentsCode: required estimate and margin of error codes
    - Fixed selectors (e.g. Kon='TOT', Alder='tot18+' for Table D)
    - All other variables: select '*'
    """
    fixed_selectors = table_conf.get("fixed_selectors", {})
    contents_codes = list(table_conf["contents_codes"].keys())
    
    query_items = []
    for var in table_meta.get("variables", []):
        code = var["code"]
        if code == "Tid":
            query_items.append({
                "code": "Tid",
                "selection": {"filter": "item", "values": WAVES_2010_2026},
            })
        elif code in fixed_selectors:
            query_items.append({
                "code": code,
                "selection": {"filter": "item", "values": fixed_selectors[code]},
            })
        elif code == "ContentsCode":
            query_items.append({
                "code": "ContentsCode",
                "selection": {"filter": "item", "values": contents_codes},
            })
        else:
            query_items.append({
                "code": code,
                "selection": {"filter": "all", "values": ["*"]},
            })
            
    return {
        "query": query_items,
        "response": {"format": "json"},
    }


def fetch_all_scb_tables(raw_dir: Path = RAW_DATA_DIR) -> Dict[str, Any]:
    """Fetch all 4 authoritative SCB PSU tables and generate manifest.json."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    retrieval_timestamp = datetime.now(timezone.utc).isoformat()
    
    manifest: Dict[str, Any] = {
        "title": "SCB PSU Support Voting Raw Data Archive",
        "retrieval_timestamp_utc": retrieval_timestamp,
        "api_base_url": SCB_API_BASE_URL,
        "time_range": {
            "start_wave": WAVES_2010_2026[0],
            "end_wave": WAVES_2010_2026[-1],
            "total_waves": len(WAVES_2010_2026),
            "waves": WAVES_2010_2026,
        },
        "methodology": METHODOLOGY_METADATA,
        "tables": {},
    }
    
    for table_key, conf in SCB_TABLES.items():
        print(f"Fetching metadata for {table_key} ({conf['table_id']})...")
        meta = fetch_scb_metadata(conf["path"])
        meta_file = raw_dir / f"{table_key}_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
            
        print(f"Building query for {table_key}...")
        query = build_table_query(meta, conf)
        query_file = raw_dir / f"{table_key}_query.json"
        with open(query_file, "w", encoding="utf-8") as f:
            json.dump(query, f, indent=2, ensure_ascii=False)
            
        print(f"Submitting query for {table_key} to SCB API...")
        data = submit_scb_query(conf["path"], query)
        data_file = raw_dir / f"{table_key}_data.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        table_manifest = {
            "table_id": conf["table_id"],
            "table_path": conf["path"],
            "source_url": f"{SCB_API_BASE_URL}/{conf['path']}",
            "title": meta.get("title", conf["title"]),
            "contents_codes": conf["contents_codes"],
            "fixed_selectors": conf.get("fixed_selectors", {}),
            "records_count": len(data.get("data", [])),
            "columns": [c["code"] for c in data.get("columns", [])],
            "files": {
                "metadata": {
                    "filename": meta_file.name,
                    "byte_count": meta_file.stat().st_size,
                    "sha256": compute_sha256(meta_file),
                },
                "query": {
                    "filename": query_file.name,
                    "byte_count": query_file.stat().st_size,
                    "sha256": compute_sha256(query_file),
                },
                "data": {
                    "filename": data_file.name,
                    "byte_count": data_file.stat().st_size,
                    "sha256": compute_sha256(data_file),
                },
            },
            "query_payload": query,
        }
        manifest["tables"][table_key] = table_manifest
        print(f"  -> Saved {table_manifest['records_count']} rows for {table_key}")

    manifest_file = raw_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        
    print(f"Raw archive completed. Manifest saved to {manifest_file}")
    return manifest


if __name__ == "__main__":
    fetch_all_scb_tables()
