"""Raw downloads from data.riksdagen.se with a local cache and a hash manifest.

Raw files live in a cache directory that is not committed. Every file the
build uses is recorded in ``manifest.json`` (committed) with its source URL,
the UTC time it was retrieved and its SHA-256. A cached file is reused only
when the manifest attests to exactly those bytes; otherwise it is downloaded
again and the entry is replaced, so the manifest never vouches for a file it
did not see arrive.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .constants import (
    BULK_URL_TEMPLATE,
    HTTP_ATTEMPTS,
    HTTP_TIMEOUT_SECONDS,
    ROLE_MANIFEST,
    SCHEMA_VERSION,
    USER_AGENT,
    VOTE_METADATA_URL_TEMPLATE,
    VOTERINGLISTA_PAGE_SIZE,
    VOTERINGLISTA_URL,
)


class NotFound(Exception):
    """The source answered 404 for a URL that may legitimately be absent."""


HttpGet = Callable[[str], bytes]


def default_http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(HTTP_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise NotFound(url) from error
            last_error = error
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def session_code(rm: str) -> str:
    """``2025/26`` -> ``202526`` (the bulk archive naming)."""
    start, end = rm.split("/")
    if len(start) != 4 or len(end) != 2 or not (start + end).isdigit():
        raise ValueError(f"riksmöte must look like 2025/26, got {rm!r}")
    return start + end


def session_slug(rm: str) -> str:
    """``2025/26`` -> ``2025-26`` (output directory naming)."""
    session_code(rm)
    return rm.replace("/", "-")


def bulk_url(rm: str) -> str:
    return BULK_URL_TEMPLATE.format(code=session_code(rm))


def register_url(rm: str) -> str:
    query = {"rm": rm, "sz": VOTERINGLISTA_PAGE_SIZE, "utformat": "json", "gruppering": "votering_id"}
    return VOTERINGLISTA_URL + "?" + urllib.parse.urlencode(query)


def rows_by_beteckning_url(rm: str, beteckning: str) -> str:
    query = {"rm": rm, "bet": beteckning, "sz": VOTERINGLISTA_PAGE_SIZE, "utformat": "json", "gruppering": ""}
    return VOTERINGLISTA_URL + "?" + urllib.parse.urlencode(query)


def vote_metadata_url(votering_id: str) -> str:
    return VOTE_METADATA_URL_TEMPLATE.format(votering_id=votering_id)


@dataclass
class RawStore:
    cache_dir: Path
    manifest_path: Path
    http_get: HttpGet = default_http_get
    refresh: bool = False
    offline: bool = False
    now: Callable[[], str] = utc_now_iso
    entries: dict[str, dict] = field(default_factory=dict)
    used: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.manifest_path = Path(self.manifest_path)
        if self.manifest_path.exists():
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self.entries = dict(payload.get("files", {}))

    def fetch(self, name: str, url: str, *, refresh: bool | None = None) -> bytes:
        """Return the bytes of ``url``, cached under ``name``.

        Raises ``NotFound`` for a 404. The manifest entry is (re)written only
        when the bytes were actually downloaded in this call.
        """
        if "/" in name or name.startswith("."):
            raise ValueError(f"bad cache name {name!r}")
        path = self.cache_dir / name
        entry = self.entries.get(name)
        use_refresh = self.refresh if refresh is None else refresh
        if not use_refresh and path.exists() and entry and entry.get("url") == url:
            data = path.read_bytes()
            if sha256_hex(data) == entry.get("sha256"):
                self.used.add(name)
                return data
        if self.offline:
            raise RuntimeError(f"offline: {name} is not in the cache with a matching manifest entry")
        data = self.http_get(url)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        self.entries[name] = {
            "url": url,
            "retrieved_at_utc": self.now(),
            "sha256": sha256_hex(data),
            "bytes": len(data),
        }
        self.used.add(name)
        return data

    def provenance(self, names: set[str] | list[str]) -> list[dict]:
        return [dict(self.entries[name], file=name) for name in sorted(names)]

    def write_manifest(self, rebuilt_session_codes: list[str] | tuple[str, ...] = ()) -> None:
        """Write the manifest: every file used now, plus earlier entries for
        sessions this run did not rebuild (each cache name embeds its session
        code). Entries of rebuilt sessions that were not used are dropped."""
        def keep(name: str) -> bool:
            if name in self.used:
                return True
            return not any(f"-{code}" in name for code in rebuilt_session_codes)

        self.entries = {name: entry for name, entry in self.entries.items() if keep(name)}
        payload = {
            "schema_version": SCHEMA_VERSION,
            "role": ROLE_MANIFEST,
            "description": (
                "Raw files downloaded from data.riksdagen.se and used to build the derived "
                "riksdag_votes outputs. The files themselves are a local, uncommitted cache."
            ),
            "files": {name: self.entries[name] for name in sorted(self.entries)},
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1) + "\n", encoding="utf-8"
        )
