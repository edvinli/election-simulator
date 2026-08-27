"""Fetch official raw electoral datasets from Valmyndigheten and write an immutable manifest."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

from .config import DEFAULT_RAW_DIR

RAW_SOURCE_URLS: dict[str, str] = {
    "valkretsmandat_riksdag_1988_2026.xlsx": "http://val.se/download/18.4005a7d19dee20a8ea544/1778074856144/valkretsmandat-riksdag-1988-2026.xlsx",
    "fasta_valkretsmandat_val_2026.xlsx": "http://val.se/download/18.4005a7d19dee20a8ea531/1778074822037/fasta-valkretsmandat-val-2026.xlsx",
    "slutligt_valresultat_riksdagen_2018_2022.xlsx": "http://val.se/download/18.162047b519a91d05331197bd/1786611369096/slutligt-valresultat-riksdagen-jamforande-statistik-2018-2022.xlsx",
    "val2018_valda.html": "https://historik.val.se/val/val2018/slutresultat/R/rike/valda.html",
    "val2022_RD_S.json": "https://resultat.val.se/data/resultat/val2022/RD_S.json",
}


def fetch_all_mandate_raw_data(raw_dir: Path | str | None = None) -> Path:
    """Download raw electoral data from Valmyndigheten, compute SHA-256 digests, and write manifest.json."""
    target_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, Any]] = []

    for filename, url in RAW_SOURCE_URLS.items():
        dest_path = target_dir / filename
        print(f"Fetching {filename} from {url} ...")

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()

        with open(dest_path, "wb") as f:
            f.write(data)

        sha256_hash = hashlib.sha256(data).hexdigest()
        retrieved_at = datetime.now(timezone.utc).isoformat()

        manifest_entries.append({
            "filename": filename,
            "source_url": url,
            "retrieved_at_utc": retrieved_at,
            "byte_count": len(data),
            "sha256": sha256_hash,
        })
        print(f"  -> Saved {dest_path} ({len(data):,d} bytes, sha256: {sha256_hash[:12]}...)")

    manifest_path = target_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"sources": manifest_entries}, f, indent=2, ensure_ascii=False)

    print(f"Manifest written to {manifest_path}")
    return target_dir
