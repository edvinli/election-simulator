"""Pure parsing and normalization for saved Pollofpolls payloads."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import unicodedata
from collections import defaultdict
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

from .config import (
    PARTY_SOURCES,
    SITE_HOME,
    SOURCE_BY_KEY,
    SWEDISHPOLLS_CSV,
    SWEDISHPOLLS_SOURCES_CSV,
    TIMESERIES_CSV,
)


PARTY_ORDER = ("M", "L", "C", "KD", "S", "V", "MP", "SD", "FI", "other")
MISSING_MARKERS = {"", "nan", "na", "n/a", "null", "none", "-", ".."}


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.strip().casefold())
    return "".join(character for character in normalized if not unicodedata.combining(character))


PARTY_ALIASES = {
    "m": "M",
    "moderaterna": "M",
    "moderata samlingspartiet": "M",
    "l": "L",
    "liberalerna": "L",
    "fp": "L",
    "folkpartiet": "L",
    "folkpartiet liberalerna": "L",
    "c": "C",
    "centerpartiet": "C",
    "kd": "KD",
    "kristdemokraterna": "KD",
    "s": "S",
    "socialdemokraterna": "S",
    "v": "V",
    "vansterpartiet": "V",
    "mp": "MP",
    "miljopartiet": "MP",
    "sd": "SD",
    "sverigedemokraterna": "SD",
    "fi": "FI",
    "f!": "FI",
    "feministiskt initiativ": "FI",
    "other": "other",
    "ovriga": "other",
    "ovrigt": "other",
    "ovr": "other",
}


POLLSTER_ALIASES = {
    "um": "United Minds",
    "united minds": "United Minds",
    "yougov": "YouGov",
    "tns-sifo": "Sifo",
    "tns sifo": "Sifo",
    "kantar-sifo": "Sifo",
    "kantar sifo": "Sifo",
    "sifo": "Sifo",
    "demoskop (inizio)": "Demoskop",
    "demoskop/inizio": "Demoskop",
}


def normalize_party(value: str) -> str:
    """Return the canonical identifier, retaining unknown labels verbatim."""

    folded = _fold(value).replace("_", " ")
    folded = re.sub(r"\s+", " ", folded)
    return PARTY_ALIASES.get(folded, value.strip())


def normalize_pollster(value: str) -> str:
    folded = re.sub(r"\s+", " ", _fold(value).replace("_", " "))
    return POLLSTER_ALIASES.get(folded, value.strip())


def parse_percentage(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace("\u2212", "-")
    if text.casefold() in MISSING_MARKERS:
        return None
    text = text.removesuffix("%").strip().replace(" ", "").replace(",", ".")
    return float(text)


def parse_date(value: str) -> date:
    text = value.strip()
    for pattern in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            pass
    raise ValueError(f"unsupported date: {value!r}")


def parse_reference_date(value: str | None) -> date:
    if not value:
        return datetime.now().date()
    try:
        return parsedate_to_datetime(value).date()
    except (TypeError, ValueError):
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()


def parse_interview_period(value: str, reference: date) -> tuple[date, date]:
    """Parse source d/m - d/m spans without inventing publication dates."""

    match = re.fullmatch(r"\s*(\d{1,2})/(\d{1,2})\s*[-–]\s*(\d{1,2})/(\d{1,2})\s*", value)
    if not match:
        raise ValueError(f"unsupported interview period: {value!r}")
    start_day, start_month, end_day, end_month = map(int, match.groups())
    end_year = reference.year
    end = date(end_year, end_month, end_day)
    if end > reference + timedelta(days=7):
        end = date(end_year - 1, end_month, end_day)
    start_year = end.year - 1 if start_month > end_month else end.year
    start = date(start_year, start_month, start_day)
    return start, end


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table_depth = 0
        self._rows: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._rows = []
        elif self._table_depth == 1 and tag == "tr":
            self._row = []
        elif self._table_depth == 1 and tag in {"td", "th"}:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._table_depth == 1 and tag in {"td", "th"} and self._cell is not None:
            assert self._row is not None
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif self._table_depth == 1 and tag == "tr" and self._row is not None:
            if self._row:
                assert self._rows is not None
                self._rows.append(self._row)
            self._row = None
        elif tag == "table" and self._table_depth:
            if self._table_depth == 1 and self._rows is not None:
                self.tables.append(self._rows)
                self._rows = None
            self._table_depth -= 1


def extract_html_tables(payload: str) -> list[list[list[str]]]:
    parser = _TableParser()
    parser.feed(payload)
    return parser.tables


def _decode(payload: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", payload, 0, 1, "unsupported source encoding")


def _table_to_dicts(rows: list[list[str]]) -> list[dict[str, str]]:
    if not rows:
        return []
    headers = [header.strip() for header in rows[0]]
    return [dict(zip(headers, row)) for row in rows[1:] if len(row) == len(headers)]


def parse_timeseries_payload(payload: bytes) -> tuple[list[dict[str, Any]], dict[str, str]]:
    text = _decode(payload)
    if "<table" in text.casefold():
        candidates = extract_html_tables(text)
        matching = [table for table in candidates if table and normalize_party(table[0][0]) == "Datum"]
        if not matching:
            matching = [table for table in candidates if table and _fold(table[0][0]) == "datum"]
        if not matching:
            raise ValueError("no Datum party table found in HTML source")
        source_rows = _table_to_dicts(matching[0])
    else:
        source_rows = list(csv.DictReader(io.StringIO(text)))

    if not source_rows:
        raise ValueError("timeseries source has no observations")
    date_header = next(
        (header for header in source_rows[0] if _fold(header) in {"datum", "date"}),
        None,
    )
    if date_header is None:
        raise ValueError("timeseries source has no date column")

    source_labels: dict[str, str] = {}
    normalized: list[dict[str, Any]] = []
    for source_row in source_rows:
        result: dict[str, Any] = {"date": parse_date(source_row[date_header]).isoformat()}
        extras: dict[str, float | None] = {}
        for header, value in source_row.items():
            if header == date_header:
                continue
            party = normalize_party(header)
            parsed = parse_percentage(value)
            if party in PARTY_ORDER:
                result[party] = parsed
                source_labels.setdefault(party, header.strip())
            else:
                extras[header.strip()] = parsed
        for party in PARTY_ORDER:
            result.setdefault(party, None)
        result["source_extra_json"] = (
            json.dumps(extras, ensure_ascii=False, sort_keys=True) if extras else ""
        )
        normalized.append(result)
    return normalized, source_labels


def parse_party_chart_payload(payload: bytes, party: str) -> dict[date, dict[str, float | None]]:
    text = _decode(payload)
    rows = csv.DictReader(io.StringIO(text))
    if not rows.fieldnames or "date" not in rows.fieldnames:
        raise ValueError(f"party {party} chart CSV has no date column")
    pollster_columns = [
        column for column in rows.fieldnames if column not in {"date", "pofp", "Val"}
    ]
    parsed: dict[date, dict[str, float | None]] = {}
    for row in rows:
        observation_date = parse_date(row["date"])
        if observation_date in parsed:
            raise ValueError(f"duplicate date in {party} chart: {observation_date}")
        parsed[observation_date] = {
            pollster: parse_percentage(row.get(pollster)) for pollster in pollster_columns
        }
    return parsed


def parse_homepage_polls(payload: bytes, reference: date) -> list[dict[str, Any]]:
    tables = extract_html_tables(_decode(payload))
    matching = [
        table
        for table in tables
        if table
        and table[0]
        and _fold(table[0][0]) == "institut"
        and any(_fold(header) == "intervjuperiod" for header in table[0])
    ]
    if not matching:
        raise ValueError("homepage has no latest-polls table")
    rows = _table_to_dicts(matching[0])
    results: list[dict[str, Any]] = []
    for row in rows:
        institute_header = next(key for key in row if _fold(key) == "institut")
        period_header = next(key for key in row if _fold(key) == "intervjuperiod")
        start, end = parse_interview_period(row[period_header], reference)
        values: dict[str, float | None] = {}
        for header, value in row.items():
            if header in {institute_header, period_header}:
                continue
            values[normalize_party(header)] = parse_percentage(value)
        results.append(
            {
                "pollster": normalize_pollster(row[institute_header]),
                "pollster_original": row[institute_header].strip(),
                "interview_start": start,
                "interview_end": end,
                "values": values,
            }
        )
    return results


def _poll_id(
    pollster: str, start: date, end: date, values: dict[str, float | None]
) -> str:
    serialized = json.dumps(
        [pollster, start.isoformat(), end.isoformat(), values],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "pop-" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:20]


def reconstruct_chart_polls(
    party_payloads: dict[str, dict[date, dict[str, float | None]]]
) -> list[dict[str, Any]]:
    """Reconstruct poll spans from values repeated across interview dates.

    Each party chart uses one column per polling institute.  A poll's support
    value is repeated on every date in its interview span.  A new segment is
    therefore identified by a date gap or a change in the cross-party vector.
    """

    all_dates = sorted({day for payload in party_payloads.values() for day in payload})
    raw_pollsters = sorted(
        {
            pollster
            for payload in party_payloads.values()
            for day_values in payload.values()
            for pollster in day_values
        }
    )
    polls: list[dict[str, Any]] = []
    for raw_pollster in raw_pollsters:
        segment_start: date | None = None
        segment_end: date | None = None
        segment_values: dict[str, float | None] | None = None

        def finish() -> None:
            nonlocal segment_start, segment_end, segment_values
            if segment_start is None or segment_end is None or segment_values is None:
                return
            canonical = normalize_pollster(raw_pollster)
            polls.append(
                {
                    "poll_id": _poll_id(canonical, segment_start, segment_end, segment_values),
                    "pollster": canonical,
                    "pollster_original": raw_pollster,
                    "interview_start": segment_start,
                    "interview_end": segment_end,
                    "values": dict(segment_values),
                    "value_sources": {
                        party: SOURCE_BY_KEY[f"party_{party}"].url
                        for party in party_payloads
                    },
                }
            )
            segment_start = segment_end = None
            segment_values = None

        for day in all_dates:
            vector = {
                party: payload.get(day, {}).get(raw_pollster)
                for party, payload in party_payloads.items()
            }
            active = any(value is not None for value in vector.values())
            consecutive = segment_end is not None and day == segment_end + timedelta(days=1)
            if not active:
                finish()
            elif segment_values is None:
                segment_start = segment_end = day
                segment_values = vector
            elif consecutive and vector == segment_values:
                segment_end = day
            else:
                finish()
                segment_start = segment_end = day
                segment_values = vector
        finish()
    return polls


def merge_homepage_polls(
    chart_polls: list[dict[str, Any]], homepage_polls: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_key = {
        (poll["pollster"], poll["interview_start"], poll["interview_end"]): poll
        for poll in chart_polls
    }
    for homepage_poll in homepage_polls:
        key = (
            homepage_poll["pollster"],
            homepage_poll["interview_start"],
            homepage_poll["interview_end"],
        )
        if key in by_key:
            poll = by_key[key]
            poll["pollster_original"] = homepage_poll["pollster_original"]
            poll["values"].update(homepage_poll["values"])
        else:
            poll = {
                **homepage_poll,
                "value_sources": {},
            }
            chart_polls.append(poll)
            by_key[key] = poll
        for party in homepage_poll["values"]:
            poll["value_sources"][party] = SITE_HOME
        poll["poll_id"] = _poll_id(
            poll["pollster"],
            poll["interview_start"],
            poll["interview_end"],
            poll["values"],
        )
    return sorted(
        chart_polls,
        key=lambda poll: (poll["interview_end"], poll["pollster"], poll["interview_start"]),
    )


def polls_to_long_rows(
    polls: Iterable[dict[str, Any]],
    retrieved_at_by_url: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for poll in polls:
        values = poll["values"]
        for party in PARTY_ORDER:
            source_value = values.get(party)
            support = source_value
            status = "reported"
            if source_value is None:
                status = "unreported"
            elif party == "FI" and source_value == 0:
                # Pollofpolls explicitly warns that some institutes include FI
                # in Other.  Preserve the displayed zero separately but do not
                # turn that ambiguous display value into a modeled observation.
                support = None
                status = "ambiguous_zero_or_included_in_other"
            source_url = poll.get("value_sources", {}).get(party)
            if source_url is None and party in SOURCE_BY_KEY:
                source_url = SOURCE_BY_KEY[party].url
            rows.append(
                {
                    "poll_id": poll["poll_id"],
                    "pollster": poll["pollster"],
                    "pollster_original": poll["pollster_original"],
                    "interview_start": poll["interview_start"].isoformat(),
                    "interview_end": poll["interview_end"].isoformat(),
                    "publication_date": None,
                    "party": party,
                    "support": support,
                    "source_value": source_value,
                    "support_status": status,
                    "sample_size": None,
                    "poll_method": None,
                    "source_url": source_url or SITE_HOME,
                    "retrieved_at": "",
                    "metadata_source_url": None,
                    "metadata_retrieved_at": None,
                    "metadata_match_status": "no_match",
                    "metadata_row_source_references_json": "[]",
                }
            )
    return rows


def normalize_raw_dataset(
    raw_dir: Path, manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_records = manifest["sources"]
    timeseries_source = SOURCE_BY_KEY["timeseries"]
    timeseries, source_labels = parse_timeseries_payload(
        (raw_dir / timeseries_source.raw_filename).read_bytes()
    )
    for row in timeseries:
        row["source_url"] = TIMESERIES_CSV
        row["retrieved_at"] = ""

    party_payloads = {
        source.party: parse_party_chart_payload(
            (raw_dir / source.raw_filename).read_bytes(), source.party
        )
        for source in PARTY_SOURCES
        if source.party is not None
    }
    chart_polls = reconstruct_chart_polls(party_payloads)
    homepage_record = source_records["homepage"]
    reference = parse_reference_date(
        homepage_record.get("upstream_capture_at") or homepage_record.get("retrieved_at")
    )
    homepage_polls = parse_homepage_polls(
        (raw_dir / SOURCE_BY_KEY["homepage"].raw_filename).read_bytes(), reference
    )
    polls = merge_homepage_polls(chart_polls, homepage_polls)
    individual_rows = polls_to_long_rows(polls)
    metadata = {
        "timeseries_source_party_labels": source_labels,
        "party_identifier_normalization": {"FP": "L", "Folkpartiet": "L", "Liberalerna": "L"},
        "support_unit": "percentage_points",
        "poll_reconstruction": (
            "Contiguous identical pollster values across party chart CSV dates are "
            "treated as one interview span; homepage rows are merged by pollster and span."
        ),
    }
    return timeseries, individual_rows, metadata


def _optional_date(value: Any) -> date | None:
    if value is None or str(value).strip().casefold() in MISSING_MARKERS:
        return None
    return parse_date(str(value))


def _optional_int(value: Any) -> int | None:
    parsed = parse_percentage(value)
    return None if parsed is None else int(parsed)


def parse_swedishpolls_payloads(
    polls_payload: bytes,
    sources_payload: bytes,
    *,
    retrieved_at: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize the CC0 SwedishPolls table without conflating it with Pollofpolls."""

    source_reader = csv.DictReader(io.StringIO(_decode(sources_payload)), delimiter=";")
    references: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for row in source_reader:
        key = (
            (row.get("Company") or "").strip(),
            (row.get("PublYearMonth") or "").strip(),
            "" if (row.get("PublDate") or "").strip().casefold() in MISSING_MARKERS else (row.get("PublDate") or "").strip(),
        )
        reference = (row.get("source") or "").strip()
        if reference and reference not in references[key]:
            references[key].append(reference)

    wide_polls: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    reader = csv.DictReader(io.StringIO(_decode(polls_payload)))
    required = {
        "PublYearMonth",
        "Company",
        "PublDate",
        "collectPeriodFrom",
        "collectPeriodTo",
        "approxPeriod",
        "house",
    }
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise ValueError("SwedishPolls CSV is missing required columns")

    for source_row_number, row in enumerate(reader, start=2):
        company = (row.get("Company") or "").strip()
        house_original = (row.get("house") or "").strip()
        if house_original.casefold() in MISSING_MARKERS:
            house_original = company
        pollster = normalize_pollster(house_original)
        publication = _optional_date(row.get("PublDate"))
        interview_start = _optional_date(row.get("collectPeriodFrom"))
        interview_end = _optional_date(row.get("collectPeriodTo"))
        values = {party: parse_percentage(row.get(party)) for party in PARTY_ORDER if party != "other"}
        values["other"] = None
        reference_key = (
            company,
            (row.get("PublYearMonth") or "").strip(),
            publication.isoformat() if publication else "",
        )
        row_references = references.get(reference_key, [])
        identity = {
            "company": company,
            "publication_period": (row.get("PublYearMonth") or "").strip(),
            "publication_date": publication.isoformat() if publication else None,
            "interview_start": interview_start.isoformat() if interview_start else None,
            "interview_end": interview_end.isoformat() if interview_end else None,
            "values": values,
            "source_row": source_row_number,
        }
        poll_id = "swp-" + hashlib.sha256(
            json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:20]
        poll = {
            "poll_id": poll_id,
            "pollster": pollster,
            "pollster_original": company,
            "house_original": house_original,
            "publication_period": identity["publication_period"],
            "publication_date": publication,
            "interview_start": interview_start,
            "interview_end": interview_end,
            "collection_period_approximate": (row.get("approxPeriod") or "").strip().casefold() == "true",
            "values": values,
            "uncertain_share": parse_percentage(row.get("Uncertain")),
            "sample_size": _optional_int(row.get("n")),
            "row_source_references": row_references,
            "source_row": source_row_number,
        }
        wide_polls.append(poll)
        for party in PARTY_ORDER:
            support = values.get(party)
            long_rows.append(
                {
                    "poll_id": poll_id,
                    "pollster": pollster,
                    "pollster_original": company,
                    "house_original": house_original,
                    "publication_period": poll["publication_period"],
                    "publication_date": publication.isoformat() if publication else None,
                    "interview_start": interview_start.isoformat() if interview_start else None,
                    "interview_end": interview_end.isoformat() if interview_end else None,
                    "collection_period_approximate": poll["collection_period_approximate"],
                    "party": party,
                    "support": support,
                    "source_value": support,
                    "support_status": "reported" if support is not None else "unreported",
                    "uncertain_share": poll["uncertain_share"],
                    "sample_size": poll["sample_size"],
                    "dataset_source_url": SWEDISHPOLLS_CSV,
                    "row_source_references_json": json.dumps(
                        row_references, ensure_ascii=False, sort_keys=True
                    ),
                    "sources_index_url": SWEDISHPOLLS_SOURCES_CSV,
                    "source_row": source_row_number,
                    "retrieved_at": "",
                }
            )
    return wide_polls, long_rows


def enrich_with_swedishpolls(
    pollofpolls_rows: list[dict[str, Any]],
    swedishpolls_wide: list[dict[str, Any]],
    *,
    metadata_retrieved_at: str | None = None,
) -> list[dict[str, Any]]:
    """Enrich metadata only for unique pollster + exact interview-span matches."""

    pop_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pollofpolls_rows:
        row.setdefault("metadata_source_url", None)
        row.setdefault("metadata_retrieved_at", None)
        row.setdefault("metadata_match_status", "no_match")
        row.setdefault("metadata_row_source_references_json", "[]")
        pop_by_id[row["poll_id"]].append(row)

    sw_index: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for poll in swedishpolls_wide:
        if poll["interview_start"] is None or poll["interview_end"] is None:
            continue
        sw_index[
            (
                poll["pollster"],
                poll["interview_start"].isoformat(),
                poll["interview_end"].isoformat(),
            )
        ].append(poll)

    crosswalk: list[dict[str, Any]] = []
    for poll_id, rows in sorted(pop_by_id.items()):
        first = rows[0]
        key = (first["pollster"], first["interview_start"], first["interview_end"])
        candidates = sw_index.get(key, [])
        match: dict[str, Any] | None = candidates[0] if len(candidates) == 1 else None
        if not candidates:
            status = "no_match"
        elif len(candidates) > 1:
            status = "ambiguous_span_match"
        else:
            status = (
                "exact_span_match_approximate_supplementary_period"
                if match and match["collection_period_approximate"]
                else "exact_span_match"
            )

        differences: dict[str, float] = {}
        if match is not None:
            for row in rows:
                pop_value = row.get("source_value")
                sw_value = match["values"].get(row["party"])
                if pop_value is not None and sw_value is not None:
                    difference = abs(float(pop_value) - float(sw_value))
                    if difference > 1e-9:
                        differences[row["party"]] = round(difference, 6)
            refs_json = json.dumps(match["row_source_references"], ensure_ascii=False, sort_keys=True)
            for row in rows:
                row["publication_date"] = (
                    match["publication_date"].isoformat() if match["publication_date"] else None
                )
                row["sample_size"] = match["sample_size"]
                row["metadata_source_url"] = SWEDISHPOLLS_CSV
                row["metadata_retrieved_at"] = None
                row["metadata_match_status"] = status
                row["metadata_row_source_references_json"] = refs_json

        crosswalk.append(
            {
                "pollofpolls_poll_id": poll_id,
                "swedishpolls_poll_id": match["poll_id"] if match else None,
                "pollster": first["pollster"],
                "interview_start": first["interview_start"],
                "interview_end": first["interview_end"],
                "match_status": status,
                "candidate_count": len(candidates),
                "max_party_absolute_difference": max(differences.values()) if differences else 0.0 if match else None,
                "party_differences_json": json.dumps(differences, sort_keys=True),
                "metadata_enriched": match is not None,
            }
        )
    return crosswalk
