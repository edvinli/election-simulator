"""Automated, non-correcting validation for normalized datasets."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
from typing import Any, Iterable

from .normalize import PARTY_ORDER, parse_date


TIMESERIES_FIELDS = (
    "date",
    "M",
    "L",
    "C",
    "KD",
    "S",
    "V",
    "MP",
    "SD",
    "FI",
    "other",
    "source_extra_json",
    "source_url",
    "retrieved_at",
)

PARTY_CHART_TIMESERIES_FIELDS = (
    "date",
    "M",
    "L",
    "C",
    "KD",
    "S",
    "V",
    "MP",
    "SD",
    "FI",
)

SWEDISHPOLLS_FIELDS = (
    "poll_id",
    "pollster",
    "pollster_original",
    "house_original",
    "publication_period",
    "publication_date",
    "interview_start",
    "interview_end",
    "collection_period_approximate",
    "party",
    "support",
    "source_value",
    "support_status",
    "uncertain_share",
    "sample_size",
    "dataset_source_url",
    "row_source_references_json",
    "sources_index_url",
    "source_row",
    "retrieved_at",
)

CROSSWALK_FIELDS = (
    "pollofpolls_poll_id",
    "swedishpolls_poll_id",
    "pollster",
    "interview_start",
    "interview_end",
    "match_status",
    "candidate_count",
    "max_party_absolute_difference",
    "party_differences_json",
    "metadata_enriched",
)

INDIVIDUAL_FIELDS = (
    "poll_id",
    "pollster",
    "pollster_original",
    "interview_start",
    "interview_end",
    "publication_date",
    "party",
    "support",
    "source_value",
    "support_status",
    "sample_size",
    "poll_method",
    "source_url",
    "retrieved_at",
    "metadata_source_url",
    "metadata_retrieved_at",
    "metadata_match_status",
    "metadata_row_source_references_json",
)


def _issue(
    issues: list[dict[str, Any]], severity: str, code: str, message: str, **details: Any
) -> None:
    item = {"severity": severity, "code": code, "message": message}
    if details:
        item["details"] = details
    issues.append(item)


def validate_timeseries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not rows:
        _issue(issues, "error", "empty_timeseries", "No Poll of Polls observations parsed")
        return issues
    missing_fields = set(TIMESERIES_FIELDS) - set(rows[0])
    if missing_fields:
        _issue(
            issues,
            "error",
            "timeseries_schema",
            "Required timeseries fields are absent",
            fields=sorted(missing_fields),
        )

    parsed_dates: list[date] = []
    invalid_dates: list[str] = []
    date_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    negative_samples: list[dict[str, Any]] = []
    nonnumeric_samples: list[dict[str, Any]] = []
    total_samples: list[dict[str, Any]] = []
    for row in rows:
        raw_date = str(row.get("date", ""))
        date_counts[raw_date] += 1
        try:
            parsed_dates.append(parse_date(raw_date))
        except ValueError:
            invalid_dates.append(raw_date)
        values: list[float] = []
        complete = True
        for party in PARTY_ORDER:
            value = row.get(party)
            if value is None:
                missing_counts[party] += 1
                complete = False
                continue
            if not isinstance(value, (int, float)):
                nonnumeric_samples.append({"date": raw_date, "party": party, "value": value})
                complete = False
                continue
            if value < 0:
                negative_samples.append({"date": raw_date, "party": party, "value": value})
            values.append(float(value))
        if complete:
            total = sum(values)
            if not 98.5 <= total <= 101.5:
                total_samples.append({"date": raw_date, "total": round(total, 3)})

    if invalid_dates:
        _issue(
            issues,
            "error",
            "invalid_timeseries_dates",
            "Timeseries dates failed ISO parsing",
            count=len(invalid_dates),
            sample=invalid_dates[:10],
        )
    if parsed_dates != sorted(parsed_dates):
        _issue(
            issues,
            "error",
            "unordered_timeseries_dates",
            "Timeseries dates are not ordered after normalization",
        )
    duplicate_dates = sorted(day for day, count in date_counts.items() if count > 1)
    if duplicate_dates:
        _issue(
            issues,
            "error",
            "duplicate_timeseries_dates",
            "Duplicate dates were retained and surfaced",
            count=len(duplicate_dates),
            sample=duplicate_dates[:10],
        )
    if nonnumeric_samples:
        _issue(
            issues,
            "error",
            "nonnumeric_party_support",
            "Party support contains nonnumeric values",
            count=len(nonnumeric_samples),
            sample=nonnumeric_samples[:10],
        )
    if negative_samples:
        _issue(
            issues,
            "error",
            "negative_party_support",
            "Party support below zero was found",
            count=len(negative_samples),
            sample=negative_samples[:10],
        )
    if total_samples:
        _issue(
            issues,
            "warning",
            "implausible_complete_timeseries_totals",
            "Complete source-category totals are not approximately 100",
            count=len(total_samples),
            sample=total_samples[:10],
        )
    if missing_counts:
        _issue(
            issues,
            "warning",
            "timeseries_missing_values",
            "Missing values are preserved, not converted to zero",
            counts=dict(sorted(missing_counts.items())),
            known_unavailable_category="other is not present in the downloadable source",
        )
    return issues


def validate_individual_polls(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not rows:
        _issue(issues, "error", "empty_individual_polls", "No individual poll rows parsed")
        return issues
    missing_fields = set(INDIVIDUAL_FIELDS) - set(rows[0])
    if missing_fields:
        _issue(
            issues,
            "error",
            "individual_schema",
            "Required individual-poll fields are absent",
            fields=sorted(missing_fields),
        )

    invalid_spans: list[dict[str, str]] = []
    out_of_range: list[dict[str, Any]] = []
    missing_original: list[str] = []
    poll_parties: dict[str, Counter[str]] = defaultdict(Counter)
    poll_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    natural_keys: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        poll_id = str(row.get("poll_id", ""))
        start = parse_date(str(row["interview_start"]))
        end = parse_date(str(row["interview_end"]))
        if start > end:
            invalid_spans.append(
                {"poll_id": poll_id, "start": start.isoformat(), "end": end.isoformat()}
            )
        support = row.get("support")
        if support is not None and not 0 <= float(support) <= 100:
            out_of_range.append(
                {"poll_id": poll_id, "party": row.get("party"), "support": support}
            )
        if not str(row.get("pollster_original", "")).strip():
            missing_original.append(poll_id)
        poll_parties[poll_id][str(row.get("party"))] += 1
        poll_rows[poll_id].append(row)
        natural_keys[(str(row.get("pollster")), start.isoformat(), end.isoformat())].add(poll_id)

    if invalid_spans:
        _issue(
            issues,
            "error",
            "invalid_interview_spans",
            "Interview start is after interview end",
            count=len(invalid_spans),
            sample=invalid_spans[:10],
        )
    if out_of_range:
        _issue(
            issues,
            "error",
            "individual_support_out_of_range",
            "Individual-poll support lies outside [0, 100]",
            count=len(out_of_range),
            sample=out_of_range[:10],
        )
    if missing_original:
        _issue(
            issues,
            "error",
            "missing_original_pollster",
            "Normalized pollster lacks its original source label",
            count=len(missing_original),
            sample=missing_original[:10],
        )
    repeated_party_rows = [
        {"poll_id": poll_id, "party": party, "count": count}
        for poll_id, counts in poll_parties.items()
        for party, count in counts.items()
        if count > 1
    ]
    if repeated_party_rows:
        _issue(
            issues,
            "error",
            "duplicate_poll_party_rows",
            "A poll contains repeated party rows",
            count=len(repeated_party_rows),
            sample=repeated_party_rows[:10],
        )
    duplicate_natural_keys = [
        {"pollster": key[0], "interview_start": key[1], "interview_end": key[2], "poll_ids": sorted(ids)}
        for key, ids in natural_keys.items()
        if len(ids) > 1
    ]
    if duplicate_natural_keys:
        _issue(
            issues,
            "warning",
            "duplicate_poll_candidates",
            "Multiple poll IDs share pollster and interview span; none were discarded",
            count=len(duplicate_natural_keys),
            sample=duplicate_natural_keys[:10],
        )

    implausible_totals: list[dict[str, Any]] = []
    for poll_id, grouped in poll_rows.items():
        if set(row["party"] for row in grouped) != set(PARTY_ORDER):
            continue
        # Validate exact displayed source values. This still checks recent
        # complete homepage rows when FI=0 is conservatively normalized to
        # support=null because its reporting status is ambiguous.
        if any(row.get("source_value") is None for row in grouped):
            continue
        total = sum(float(row["source_value"]) for row in grouped)
        if not 97.0 <= total <= 103.0:
            implausible_totals.append({"poll_id": poll_id, "total": round(total, 3)})
    if implausible_totals:
        _issue(
            issues,
            "warning",
            "implausible_complete_poll_totals",
            "Complete individual-poll party totals are not plausibly close to 100",
            count=len(implausible_totals),
            sample=implausible_totals[:10],
        )
    return issues


def validation_report(
    timeseries: list[dict[str, Any]],
    individual: list[dict[str, Any]],
    swedishpolls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    issues = validate_timeseries(timeseries) + validate_individual_polls(individual)
    if swedishpolls is not None:
        issues += validate_swedishpolls(swedishpolls)
    counts = Counter(issue["severity"] for issue in issues)
    return {
        "valid": counts["error"] == 0,
        "error_count": counts["error"],
        "warning_count": counts["warning"],
        "issues": issues,
    }


def validate_swedishpolls(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not rows:
        _issue(issues, "error", "empty_swedishpolls", "No supplementary SwedishPolls rows parsed")
        return issues
    missing_fields = set(SWEDISHPOLLS_FIELDS) - set(rows[0])
    if missing_fields:
        _issue(
            issues,
            "error",
            "swedishpolls_schema",
            "Required SwedishPolls fields are absent",
            fields=sorted(missing_fields),
        )
    poll_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid_spans: list[dict[str, Any]] = []
    invalid_support: list[dict[str, Any]] = []
    invalid_samples: list[dict[str, Any]] = []
    for row in rows:
        poll_rows[str(row["poll_id"])].append(row)
        start_raw, end_raw = row.get("interview_start"), row.get("interview_end")
        if start_raw and end_raw and parse_date(str(start_raw)) > parse_date(str(end_raw)):
            invalid_spans.append({"poll_id": row["poll_id"], "start": start_raw, "end": end_raw})
        support = row.get("support")
        if support is not None and not 0 <= float(support) <= 100:
            invalid_support.append(
                {"poll_id": row["poll_id"], "party": row["party"], "support": support}
            )
        sample = row.get("sample_size")
        if sample is not None and int(sample) <= 0:
            invalid_samples.append({"poll_id": row["poll_id"], "sample_size": sample})
    if invalid_spans:
        _issue(
            issues,
            "error",
            "swedishpolls_invalid_spans",
            "Supplementary interview start is after interview end",
            count=len(invalid_spans),
            sample=invalid_spans[:10],
        )
    if invalid_support:
        _issue(
            issues,
            "error",
            "swedishpolls_support_out_of_range",
            "Supplementary support lies outside [0, 100]",
            count=len(invalid_support),
            sample=invalid_support[:10],
        )
    if invalid_samples:
        _issue(
            issues,
            "error",
            "swedishpolls_invalid_sample_size",
            "Supplementary sample size is not positive",
            count=len(invalid_samples),
            sample=invalid_samples[:10],
        )
    wrong_row_counts = [poll_id for poll_id, grouped in poll_rows.items() if len(grouped) != len(PARTY_ORDER)]
    if wrong_row_counts:
        _issue(
            issues,
            "error",
            "swedishpolls_poll_row_count",
            "Each supplementary poll must have exactly one row per canonical category",
            count=len(wrong_row_counts),
            sample=wrong_row_counts[:10],
        )
    natural_keys: dict[tuple[str, str, str, str, str], list[str]] = defaultdict(list)
    for poll_id, grouped in poll_rows.items():
        first = grouped[0]
        key = (
            str(first.get("pollster") or ""),
            str(first.get("publication_period") or ""),
            str(first.get("publication_date") or ""),
            str(first.get("interview_start") or ""),
            str(first.get("interview_end") or ""),
        )
        natural_keys[key].append(poll_id)
    duplicate_candidates = [
        {
            "pollster": key[0],
            "publication_period": key[1] or None,
            "publication_date": key[2] or None,
            "interview_start": key[3] or None,
            "interview_end": key[4] or None,
            "poll_ids": poll_ids[:10],
        }
        for key, poll_ids in natural_keys.items()
        if len(poll_ids) > 1
    ]
    if duplicate_candidates:
        _issue(
            issues,
            "warning",
            "swedishpolls_duplicate_candidates",
            "Supplementary polls share pollster and available date fields; none were discarded",
            count=len(duplicate_candidates),
            sample=duplicate_candidates[:10],
        )
    implausible: list[dict[str, Any]] = []
    for poll_id, grouped in poll_rows.items():
        # SwedishPolls currently has no Other column. Check a total only if a
        # future source version reports every canonical category; a nine-party
        # subtotal below 100 is not a complete total and is not anomalous.
        if any(row.get("source_value") is None for row in grouped):
            continue
        total = sum(float(row["source_value"]) for row in grouped)
        if not 97 <= total <= 103:
            implausible.append({"poll_id": poll_id, "total": round(total, 3)})
    if implausible:
        _issue(
            issues,
            "warning",
            "swedishpolls_implausible_totals",
            "Complete supplementary party totals are not plausibly close to 100",
            count=len(implausible),
            sample=implausible[:10],
        )
    return issues
