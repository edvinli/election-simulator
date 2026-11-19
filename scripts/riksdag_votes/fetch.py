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
import http.client
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
    DOKUMENTLISTA_URL,
    HTTP_ATTEMPTS,
    HTTP_BACKOFF_BASE_SECONDS,
    HTTP_BACKOFF_MAX_SECONDS,
    HTTP_MIN_INTERVAL_SECONDS,
    HTTP_TIMEOUT_SECONDS,
    ROLE_MANIFEST,
    SCHEMA_VERSION,
    USER_AGENT,
    UTSKOTTSFORSLAG_URL_TEMPLATE,
    VOTE_METADATA_URL_TEMPLATE,
    VOTERINGLISTA_PAGE_SIZE,
    VOTERINGLISTA_URL,
)


class NotFound(Exception):
    """The source answered 404 for a URL that may legitimately be absent."""


HttpGet = Callable[[str], bytes]


def _urlopen(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        return response.read()


def _retry_after_seconds(error: urllib.error.HTTPError) -> float | None:
    value = error.headers.get("Retry-After") if error.headers is not None else None
    try:
        return max(0.0, float(value)) if value is not None else None
    except ValueError:
        return None


@dataclass
class PoliteHttpGet:
    """GET with a minimum spacing between requests and backoff on failure.

    At most one request starts per ``min_interval`` seconds (4 requests/s by
    default). A 404 raises ``NotFound`` at once; a 429, a 5xx or a network
    error is retried after ``backoff_base * 2**attempt`` seconds (or the
    server's ``Retry-After``), capped at ``backoff_max``, up to ``attempts``
    tries. Other 4xx responses are not retried.
    """

    opener: Callable[[str], bytes] = _urlopen
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    min_interval: float = HTTP_MIN_INTERVAL_SECONDS
    attempts: int = HTTP_ATTEMPTS
    backoff_base: float = HTTP_BACKOFF_BASE_SECONDS
    backoff_max: float = HTTP_BACKOFF_MAX_SECONDS
    _last_start: float | None = None

    def _wait_turn(self) -> None:
        if self._last_start is not None:
            remaining = self._last_start + self.min_interval - self.clock()
            if remaining > 0:
                self.sleep(remaining)
        self._last_start = self.clock()

    def __call__(self, url: str) -> bytes:
        last_error: Exception | None = None
        for attempt in range(self.attempts):
            self._wait_turn()
            delay = min(self.backoff_max, self.backoff_base * 2**attempt)
            try:
                return self.opener(url)
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    raise NotFound(url) from error
                if error.code != 429 and error.code < 500:
                    raise RuntimeError(f"failed to fetch {url}: HTTP {error.code}") from error
                retry_after = _retry_after_seconds(error)
                if retry_after is not None:
                    delay = min(self.backoff_max, retry_after)
                last_error = error
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as error:
                last_error = error
            if attempt + 1 < self.attempts:
                self.sleep(delay)
        raise RuntimeError(f"failed to fetch {url} after {self.attempts} attempts: {last_error}")


_DEFAULT_HTTP = PoliteHttpGet()


def default_http_get(url: str) -> bytes:
    """The shared, rate-limited getter (one limiter per process)."""
    return _DEFAULT_HTTP(url)


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


def utskottsforslag_url(dok_id: str) -> str:
    return UTSKOTTSFORSLAG_URL_TEMPLATE.format(dok_id=urllib.parse.quote(dok_id, safe=""))


def dokumentlista_by_beteckning_url(rm: str, beteckning: str) -> str:
    query = {"rm": rm, "bet": beteckning, "sz": 50, "utformat": "json"}
    return DOKUMENTLISTA_URL + "?" + urllib.parse.urlencode(query)


@dataclass
class RawStore:
    """Cached raw downloads.

    ``refresh`` re-downloads every file. ``refresh_listings`` re-downloads only
    the session-level listings that grow as the riksdag votes (the register and
    the bulk archive, fetched through ``fetch_listing``); per-id files (vote
    metadata, document titles) are immutable and stay cached.
    """

    cache_dir: Path
    manifest_path: Path
    http_get: HttpGet = default_http_get
    refresh: bool = False
    offline: bool = False
    refresh_listings: bool = False
    now: Callable[[], str] = utc_now_iso
    entries: dict[str, dict] = field(default_factory=dict)
    used: set[str] = field(default_factory=set)
    downloaded: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.manifest_path = Path(self.manifest_path)
        if self.manifest_path.exists():
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self.entries = dict(payload.get("files", {}))

    def fetch(self, name: str, url: str, *, refresh: bool | None = None) -> bytes:
        """Return the bytes of ``url``, cached under ``name``.

        Raises ``NotFound`` for a 404. The manifest entry is written only when
        the bytes were actually downloaded in this call. A re-download that
        returns exactly the bytes the manifest already attests to keeps the
        entry unchanged: ``retrieved_at_utc`` is when these exact bytes were
        first retrieved, so an unchanged source does not churn the manifest.
        In offline mode a cache miss raises and nothing is downloaded.
        """
        if "/" in name or name.startswith("."):
            raise ValueError(f"bad cache name {name!r}")
        path = self.cache_dir / name
        entry = self.entries.get(name)
        use_refresh = self.refresh if refresh is None else refresh
        if self.offline:
            use_refresh = False
        if name in self.downloaded:
            use_refresh = False  # never download the same file twice in one run
        if not use_refresh and path.exists() and entry and entry.get("url") == url:
            data = path.read_bytes()
            if sha256_hex(data) == entry.get("sha256"):
                self.used.add(name)
                return data
        if self.offline:
            raise RuntimeError(f"offline: {name} is not in the cache with a matching manifest entry")
        data = self.http_get(url)
        self.downloaded.add(name)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        digest = sha256_hex(data)
        if not (entry and entry.get("url") == url and entry.get("sha256") == digest):
            self.entries[name] = {
                "url": url,
                "retrieved_at_utc": self.now(),
                "sha256": digest,
                "bytes": len(data),
            }
        self.used.add(name)
        return data

    def cached(self, name: str, url: str) -> bytes | None:
        """The cached bytes the manifest attests to, or None; never downloads."""
        path = self.cache_dir / name
        entry = self.entries.get(name)
        if not (path.exists() and entry and entry.get("url") == url):
            return None
        data = path.read_bytes()
        if sha256_hex(data) != entry.get("sha256"):
            return None
        self.used.add(name)
        return data

    def fetch_listing(self, name: str, url: str) -> bytes:
        """``fetch`` for a session-level listing that grows over time."""
        return self.fetch(name, url, refresh=self.refresh or self.refresh_listings)

    def is_fresh(self, name: str) -> bool:
        """Whether ``name`` was downloaded during this run."""
        return name in self.downloaded

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
