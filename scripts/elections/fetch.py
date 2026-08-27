"""Fetch raw official election result documents from Valmyndigheten."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
import urllib.request

from .config import ELECTIONS, ElectionMetadata


DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "elections"
USER_AGENT = "SwedishElectionForecastingBot/1.0 (Research/Academic Pair Programming Pipeline)"


def fetch_and_save_document(
    url: str,
    target_path: Path,
    user_agent: str = USER_AGENT,
    timeout: int = 15,
) -> dict[str, Any]:
    """Download a raw document from an authoritative URL and return provenance metadata."""
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content_bytes = resp.read()

    sha256_hash = hashlib.sha256(content_bytes).hexdigest()
    byte_count = len(content_bytes)
    retrieved_at = datetime.now(timezone.utc).isoformat()

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "wb") as f:
        f.write(content_bytes)

    return {
        "source_url": url,
        "local_filename": target_path.name,
        "local_path": str(target_path),
        "retrieved_at": retrieved_at,
        "sha256": sha256_hash,
        "byte_count": byte_count,
        "http_status": resp.status,
    }


def fetch_all_elections(raw_dir: Path | str | None = None) -> dict[str, Any]:
    """Fetch all official election result documents across configured election years."""
    base_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, Any]] = []

    for year, meta in sorted(ELECTIONS.items()):
        # Primary document
        primary_path = base_dir / meta.raw_filename
        print(f"Fetching {year} official results from {meta.source_url} -> {primary_path.name} ...")
        entry = fetch_and_save_document(meta.source_url, primary_path)
        entry["election_year"] = year
        entry["document_type"] = "primary"
        manifest_entries.append(entry)

        # Secondary document (if any, e.g. 2006 ovriga)
        if meta.secondary_source_url and meta.secondary_raw_filename:
            sec_path = base_dir / meta.secondary_raw_filename
            print(f"Fetching {year} secondary results from {meta.secondary_source_url} -> {sec_path.name} ...")
            sec_entry = fetch_and_save_document(meta.secondary_source_url, sec_path)
            sec_entry["election_year"] = year
            sec_entry["document_type"] = "secondary"
            manifest_entries.append(sec_entry)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elections_count": len(ELECTIONS),
        "documents": manifest_entries,
    }

    manifest_path = base_dir / "retrieval_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nSaved retrieval manifest to {manifest_path}")
    return manifest


if __name__ == "__main__":
    fetch_all_elections()
