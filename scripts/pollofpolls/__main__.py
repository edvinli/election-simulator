"""Command-line entry point for the complete Pollofpolls refresh."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable

from .acquire import AcquisitionError, acquire_all
from .config import (
    PARTY_SOURCES,
    SITE_HOME,
    SOURCE_BY_KEY,
    SWEDISHPOLLS_CSV,
    SWEDISHPOLLS_SOURCES_CSV,
    TIMESERIES_CSV,
    TIMESERIES_PAGE,
)
from .normalize import (
    enrich_with_swedishpolls,
    extract_party_chart_pop_timeseries,
    normalize_raw_dataset,
    parse_swedishpolls_payloads,
)
from .validate import (
    CROSSWALK_FIELDS,
    INDIVIDUAL_FIELDS,
    PARTY_CHART_TIMESERIES_FIELDS,
    SWEDISHPOLLS_FIELDS,
    TIMESERIES_FIELDS,
    validation_report,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = REPOSITORY_ROOT / "data" / "raw" / "pollofpolls"
DEFAULT_PROCESSED = REPOSITORY_ROOT / "data" / "processed" / "pollofpolls"
DEFAULT_README = REPOSITORY_ROOT / "data" / "README.md"


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _summary(
    timeseries: list[dict[str, Any]],
    individual: list[dict[str, Any]],
    swedishpolls: list[dict[str, Any]],
    crosswalk: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    poll_ids = {row["poll_id"] for row in individual}
    source_parties = sorted(
        {row["party"] for row in individual if row.get("source_value") is not None},
        key=lambda value: (value == "other", value),
    )
    pollsters = sorted({row["pollster"] for row in individual})
    supplementary_poll_ids = {row["poll_id"] for row in swedishpolls}
    publication_dates = [row["publication_date"] for row in swedishpolls if row["publication_date"]]
    supplementary_parties = sorted(
        {row["party"] for row in swedishpolls if row.get("source_value") is not None},
        key=lambda value: (value == "other", value),
    )
    supplementary_pollsters = sorted({row["pollster"] for row in swedishpolls})
    supplementary_original_pollsters = sorted(
        {row["pollster_original"] for row in swedishpolls}
    )
    supplementary_interview_starts = [
        row["interview_start"] for row in swedishpolls if row["interview_start"]
    ]
    supplementary_interview_ends = [
        row["interview_end"] for row in swedishpolls if row["interview_end"]
    ]
    enriched = sum(bool(row["metadata_enriched"]) for row in crosswalk)
    match_status_counts: dict[str, int] = {}
    for row in crosswalk:
        status = str(row["match_status"])
        match_status_counts[status] = match_status_counts.get(status, 0) + 1
    value_disagreements = [
        row for row in crosswalk if row["party_differences_json"] != "{}"
    ]
    maximum_difference = max(
        (
            float(row["max_party_absolute_difference"])
            for row in crosswalk
            if row["max_party_absolute_difference"] is not None
        ),
        default=0.0,
    )
    supplementary_publication_ids = {
        row["poll_id"] for row in swedishpolls if row["publication_date"]
    }
    supplementary_span_ids = {
        row["poll_id"]
        for row in swedishpolls
        if row["interview_start"] and row["interview_end"]
    }
    supplementary_sample_ids = {
        row["poll_id"] for row in swedishpolls if row["sample_size"] is not None
    }
    supplementary_reference_ids = {
        row["poll_id"]
        for row in swedishpolls
        if row["row_source_references_json"] != "[]"
    }
    return {
        "timeseries_observations": len(timeseries),
        "timeseries_earliest": min(row["date"] for row in timeseries),
        "timeseries_latest": max(row["date"] for row in timeseries),
        "individual_polls": len(poll_ids),
        "individual_long_rows": len(individual),
        "individual_earliest_interview_start": min(row["interview_start"] for row in individual),
        "individual_latest_interview_end": max(row["interview_end"] for row in individual),
        "parties_and_categories": source_parties,
        "pollsters": pollsters,
        "supplementary_individual_polls": len(supplementary_poll_ids),
        "supplementary_long_rows": len(swedishpolls),
        "supplementary_earliest_publication": min(publication_dates),
        "supplementary_latest_publication": max(publication_dates),
        "supplementary_earliest_interview_start": min(supplementary_interview_starts),
        "supplementary_latest_interview_end": max(supplementary_interview_ends),
        "supplementary_parties": supplementary_parties,
        "supplementary_pollsters": supplementary_pollsters,
        "supplementary_original_pollsters": supplementary_original_pollsters,
        "metadata_enriched_polls": enriched,
        "metadata_unmatched_polls": len(crosswalk) - enriched,
        "metadata_match_status_counts": dict(sorted(match_status_counts.items())),
        "metadata_value_disagreement_polls": len(value_disagreements),
        "metadata_max_party_absolute_difference": maximum_difference,
        "supplementary_polls_with_publication_date": len(supplementary_publication_ids),
        "supplementary_polls_with_interview_span": len(supplementary_span_ids),
        "supplementary_polls_with_sample_size": len(supplementary_sample_ids),
        "supplementary_polls_with_source_references": len(supplementary_reference_ids),
        "retrieval_methods": sorted(
            {record["retrieval_method"] for record in manifest["sources"].values()}
        ),
    }


def _render_readme(summary: dict[str, Any], metadata: dict[str, Any]) -> str:
    party_sources = "\n".join(
        f"- `{source.url}` — {source.party}; declared by `{source.page_url}`."
        for source in PARTY_SOURCES
    )
    parties = ", ".join(f"`{party}`" for party in summary["parties_and_categories"])
    pollsters = ", ".join(summary["pollsters"])
    supplementary_parties = ", ".join(
        f"`{party}`" for party in summary["supplementary_parties"]
    )
    supplementary_pollsters = ", ".join(summary["supplementary_pollsters"])
    retrieval_methods = ", ".join(f"`{method}`" for method in summary["retrieval_methods"])
    return f"""# Pollofpolls data

This directory contains only data acquisition, normalization, and validation for the
Swedish election project. It does **not** contain forecasting or simulation code.

## Snapshot summary

- Poll of Polls estimates: **{summary['timeseries_observations']:,}** daily observations,
  {summary['timeseries_earliest']} through {summary['timeseries_latest']}.
- Reconstructed individual polls: **{summary['individual_polls']:,}** polls
  ({summary['individual_long_rows']:,} long-format rows), with interview spans from
  {summary['individual_earliest_interview_start']} through
  {summary['individual_latest_interview_end']}.
- Supplementary SwedishPolls dataset: **{summary['supplementary_individual_polls']:,}** polls
  ({summary['supplementary_long_rows']:,} long-format rows), published from
  {summary['supplementary_earliest_publication']} through
  {summary['supplementary_latest_publication']}; available interview spans run from
  {summary['supplementary_earliest_interview_start']} through
  {summary['supplementary_latest_interview_end']}.
- Unique pollster/interview-span metadata matches: **{summary['metadata_enriched_polls']:,}**;
  unmatched or ambiguous Pollofpolls polls: **{summary['metadata_unmatched_polls']:,}**.
- Supplementary field coverage: publication date on
  **{summary['supplementary_polls_with_publication_date']:,}** polls, interview span on
  **{summary['supplementary_polls_with_interview_span']:,}**, sample size on
  **{summary['supplementary_polls_with_sample_size']:,}**, and row source references on
  **{summary['supplementary_polls_with_source_references']:,}**.
- Parties/categories present in source values: {parties}.
- Pollsters present: {pollsters}.
- Supplementary parties reported: {supplementary_parties}.
- Supplementary pollsters present: {supplementary_pollsters}.
- Retrieval method(s) for this raw snapshot: {retrieval_methods}.

These counts describe the checked-in raw snapshot and are regenerated by the refresh
command.

## Exact Pollofpolls sources

- `{SITE_HOME}` — rendered table of recent individual polls and their interview periods.
- `{TIMESERIES_PAGE}` — first-party full HTML table of Poll of Polls estimates.
- `{TIMESERIES_CSV}` — CSV linked from that table page as "CSV fil för tabellen".
{party_sources}

## Supplementary individual-poll source

- `{SWEDISHPOLLS_CSV}` — the SwedishPolls project poll table.
- `{SWEDISHPOLLS_SOURCES_CSV}` — its poll-level source-reference index.
- `https://github.com/MansMeg/SwedishPolls` — documentation, license, and methodology.

SwedishPolls is retained as a separate supplementary dataset. Its support values do not
replace Pollofpolls values. A unique normalized-pollster plus exact interview-start/end
match may enrich the Pollofpolls row with publication date and sample size; the
crosswalk records the match and any party-value differences. The original Pollofpolls
`source_url` and support remain unchanged, while supplementary provenance is stored in
dedicated `metadata_*` fields. SwedishPolls publishes its data under CC0 and code under
MIT; its citation requests the repository revision/hash.

Of the matched rows in this snapshot, **{summary['metadata_value_disagreement_polls']:,}**
polls have at least one non-identical party value across the two sources; the largest
absolute party difference is **{summary['metadata_max_party_absolute_difference']}**
percentage points. This can reflect source revisions, rounding, FI/Other treatment, or
an imperfect chart-based reconstruction. It is surfaced in the crosswalk and is a
reason not to combine the two support series silently.

The home and party pages were inspected in normal page source. Each party page calls
`AmCharts.loadFile` with the listed CSV. No endpoint name was guessed. During initial
inspection on 2026-08-26 the HTTPS host presented the wrong certificate and a subsequent
HTTP request received a transient 455 firewall response; the actual refresh later
succeeded over the site's canonical public HTTP URLs. The fetcher makes at most one
live-host attempt after a refusal and opens a circuit breaker if it is blocked. It can
then use a pinned public Wayback replay of the same first-party response. The retrieval
manifest records both the original
Pollofpolls URL and archive replay URL, final URL, capture time, selected HTTP headers,
byte length, retrieval time, and SHA-256 hash.

## Files and fields

`raw/pollofpolls/` stores exact response payloads plus `retrieval_manifest.json`.
Raw payloads are never edited. Exact acquisition timestamps, response hashes, HTTP
metadata, and byte counts live in `data/raw/pollofpolls/retrieval_manifest.json` and the
raw snapshot.

`processed/pollofpolls/` contains:

- `pollofpolls_timeseries.csv`: `date`, M, L, C, KD, S, V, MP, SD, FI,
  `other`, `source_extra_json`, `source_url`, and `retrieved_at`.
- `pollofpolls_party_chart_timeseries.csv`: `date`, M, L, C, KD, S, V, MP, SD, FI
  extracted from first-party party chart CSVs (2009+), verified to match the canonical
  2014+ timeseries exactly.
- `individual_polls.csv`: deterministic `poll_id`, normalized and original pollster,
  separate interview/publication dates, party, normalized `support`, exact numeric
  `source_value`, reporting status, sample/method, source URL, `retrieved_at`, and
  supplementary-metadata provenance (`metadata_source_url`, `metadata_retrieved_at`,
  `metadata_match_status`, and `metadata_row_source_references_json`).
- `swedishpolls_individual_polls.csv`: separately normalized long-format copy of the
  supplementary table, including original company/house, approximate-period flag,
  sample size, source row, poll-level source references, and `retrieved_at`.
- `pollofpolls_swedishpolls_crosswalk.csv`: every Pollofpolls poll's match status,
  candidate count, linked SwedishPolls ID, and source-value differences.
- `validation_report.json`, `dataset_summary.json`, and
  `normalization_metadata.json`.

Dates use ISO `YYYY-MM-DD`. Support values are percentage points: `28.4` means 28.4%,
not 0.284. `poll_method` remains null because neither acquired table supplies it.
Unmatched Pollofpolls publication dates and sample sizes remain null; they are never
inferred from another date or a non-unique match.

Processed CSV schemas retain `retrieved_at` and `metadata_retrieved_at` columns for
schema compatibility, but those observation-row fields are intentionally blank so that
processed rows are byte-stable when polling content is semantically unchanged. Exact
network acquisition timestamps live in the raw retrieval manifest, while observation
and source provenance (such as source URLs, dataset references, and matching status)
remains in processed rows.

## Historical naming and missingness

Folkpartiet (`FP`), Folkpartiet Liberalerna, and Liberalerna normalize to canonical
`L`; the source header/label remains in raw data and normalization metadata. Pollster
names normalize conservatively while `pollster_original` preserves the displayed name.
For example, TNS-Sifo/Kantar-Sifo map to `Sifo`, and `UM` maps to `United Minds`.

FI begins only where its separate source series exists. Pollofpolls warns that some
institutes include FI in Övriga. A displayed FI zero in an individual poll is therefore
stored as `source_value=0.0`, `support=null`, and
`support_status=ambiguous_zero_or_included_in_other`; it is not treated as a genuine
zero. Other absent values remain null. The Poll of Polls CSV does not contain Övriga,
so `other` is null there. Unknown future source categories are retained in
`source_extra_json` rather than discarded.

SwedishPolls has no Övriga/Other column. Its `Uncertain` field is retained as
`uncertain_share` and is not misclassified as party support. Consequently, a missing
supplementary `other` is null and nine reported party values are only a subtotal.

## Individual-poll reconstruction and limitations

The party-chart CSVs repeat each poll's party value on every day of its interview
period. The normalizer joins all nine party files and treats a contiguous, unchanged
cross-party vector for one institute as a poll span. Recent homepage rows are merged by
normalized pollster and exact interview span; this also supplies Övriga where displayed.
This follows the public chart encoding, but two overlapping polls from the same
institute cannot both be represented in one CSV column. Historical tracking polls or
adjacent polls with an identical full party vector may therefore be merged, truncated,
or otherwise ambiguous. No source value is corrected to address that ambiguity.

SwedishPolls documents uneven historical completeness: early data are lower quality,
before 2000 the repository currently includes only Sifo, and the post-2008 period is
more complete. It also documents that some Ipsos rows were normalized from totals of
101 and that sample size represents respondents rather than everyone contacted. Its
source-reference index does not provide a URL for every row. These caveats remain
supplementary metadata and are not used to alter Pollofpolls observations.

Each Poll of Polls row date is the model-estimate date; the source says the series ends
at the latest included poll's interview end, rather than its later publication date.
The homepage supplies day/month interview spans but no publication date; years are
anchored to the preserved response date and cross-year spans are handled explicitly. No
date field is substituted for another.

## Refresh and tests

From the repository root:

```bash
make fetch-pollofpolls
make test-pollofpolls
```

For a deterministic parse/validation run without network access:

```bash
make process-pollofpolls
```

The refresh stages writes atomically, records exact raw retrieval metadata in the
manifest, normalizes both datasets deterministically, runs all validations, writes
reports, and prints a concise coverage summary. Unit tests use saved fixtures; no normal
test makes a live request.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="parse verified saved raw files")
    parser.add_argument(
        "--no-archive-fallback",
        action="store_true",
        help="fail instead of using a preserved first-party response",
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--timeout", type=float, default=45.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        manifest, messages = acquire_all(
            args.raw_dir,
            offline=args.offline,
            allow_archive_fallback=not args.no_archive_fallback,
            timeout=args.timeout,
        )
    except AcquisitionError as exc:
        print(f"Pollofpolls acquisition failed: {exc}")
        return 2
    for message in messages:
        print(f"- {message}")

    timeseries, individual, metadata = normalize_raw_dataset(args.raw_dir, manifest)
    swedishpolls_wide, swedishpolls = parse_swedishpolls_payloads(
        (args.raw_dir / SOURCE_BY_KEY["swedishpolls"].raw_filename).read_bytes(),
        (args.raw_dir / SOURCE_BY_KEY["swedishpolls_sources"].raw_filename).read_bytes(),
    )
    crosswalk = enrich_with_swedishpolls(
        individual,
        swedishpolls_wide,
    )
    report = validation_report(timeseries, individual, swedishpolls)
    summary = _summary(timeseries, individual, swedishpolls, crosswalk, manifest)
    party_chart_timeseries = extract_party_chart_pop_timeseries(
        args.raw_dir,
        canonical_timeseries=timeseries,
    )

    _write_csv(args.processed_dir / "pollofpolls_timeseries.csv", TIMESERIES_FIELDS, timeseries)
    _write_csv(
        args.processed_dir / "pollofpolls_party_chart_timeseries.csv",
        PARTY_CHART_TIMESERIES_FIELDS,
        party_chart_timeseries,
    )
    _write_csv(args.processed_dir / "individual_polls.csv", INDIVIDUAL_FIELDS, individual)
    _write_csv(
        args.processed_dir / "swedishpolls_individual_polls.csv",
        SWEDISHPOLLS_FIELDS,
        swedishpolls,
    )
    _write_csv(
        args.processed_dir / "pollofpolls_swedishpolls_crosswalk.csv",
        CROSSWALK_FIELDS,
        crosswalk,
    )
    _atomic_text(
        args.processed_dir / "validation_report.json",
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(
        args.processed_dir / "dataset_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(
        args.processed_dir / "normalization_metadata.json",
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(DEFAULT_README, _render_readme(summary, metadata))

    print(
        "Pollofpolls: "
        f"{summary['timeseries_observations']:,} estimates "
        f"({summary['timeseries_earliest']}..{summary['timeseries_latest']}), "
        f"{summary['individual_polls']:,} individual polls, "
        f"{summary['supplementary_individual_polls']:,} supplementary polls, "
        f"{summary['metadata_enriched_polls']:,} metadata matches, "
        f"{report['error_count']} errors / {report['warning_count']} warnings."
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
