"""Build one aggregate version and write everything the model reads from it.

Used by the command line and by the production polling refresh, so both write
byte-identical outputs from the same inputs: the daily timeseries, its
metadata, and ``individual_polls.csv`` (the cleaned SwedishPolls polls the
aggregate was built from, in the schema OpinionState reads).
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from typing import Sequence

from .build import AggregateResult, build_aggregate, write_timeseries
from .config import (
    HISTORICAL_RESULT_AVAILABILITY_LAG_DAYS,
    HISTORY_START_ELECTION,
    METADATA_FILENAME,
    PARTIES,
    TIMESERIES_FILENAME,
    AggregateSpec,
)
from .data import PollDataset, load_elections, load_polls, sha256_file
from .model import Hyperparameters

POLLS_FILENAME = "individual_polls.csv"
INDIVIDUAL_POLL_FIELDS = (
    "poll_id", "pollster", "pollster_original", "interview_start", "interview_end",
    "publication_date", "party", "support", "source_value", "support_status", "sample_size",
    "poll_method", "source_url", "retrieved_at", "metadata_source_url", "metadata_retrieved_at",
    "metadata_match_status", "metadata_row_source_references_json",
)


def write_individual_polls(polls_file: Path, eligible: set[str], dest: Path) -> int:
    """The eligible SwedishPolls polls in individual_polls.csv's schema.

    Sample sizes are the reported ones (blank when unreported); the
    aggregate's overlap discount and imputation are its own concerns.
    """

    written = set()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with Path(polls_file).open(encoding="utf-8", newline="") as src, tmp.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=INDIVIDUAL_POLL_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in csv.DictReader(src):
            if row["poll_id"] not in eligible or row["party"] not in PARTIES:
                continue
            written.add(row["poll_id"])
            writer.writerow({
                "poll_id": row["poll_id"],
                "pollster": row["pollster"],
                "pollster_original": row["pollster_original"],
                "interview_start": row["interview_start"],
                "interview_end": row["interview_end"],
                "publication_date": row["publication_date"],
                "party": row["party"],
                "support": row["support"],
                "source_value": row["source_value"],
                "support_status": row["support_status"],
                "sample_size": row["sample_size"],
                "source_url": row["dataset_source_url"],
                "metadata_match_status": "swedishpolls",
                "metadata_row_source_references_json": "[]",
            })
    tmp.replace(dest)
    return len(written)


def stored_fits(metadata_file: Path, spec: AggregateSpec) -> dict[date, tuple[Hyperparameters, float, int, bool]]:
    """Recorded segment fits of a previous build of the same version, if any."""

    if not metadata_file.is_file():
        return {}
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata.get("aggregate_version") != spec.version:
        return {}
    if not all("hyperparameters_exact" in s for s in metadata["segments"]):
        return {}  # rounded fits would not reproduce the series; refit instead
    return {
        date.fromisoformat(s["hyperparameters_fit_date"]): (
            Hyperparameters(**s.get("hyperparameters_exact") or s["hyperparameters"]), float(s["log_likelihood"]),
            int(s["optimizer_iterations"]), bool(s["optimizer_converged"]),
        )
        for s in metadata["segments"]
    }


def aggregate_metadata(result: AggregateResult, dataset: PollDataset, elections, spec: AggregateSpec,
                       polls_file: Path, election_results: Path, manifests: Sequence[Path]) -> dict:
    return {
        "aggregate_version": spec.version,
        "reported_level": "consensus_reading" if spec.consensus_window_days else "latent_support",
        "consensus_window_days": spec.consensus_window_days,
        "inputs": {
            "polls": {"path": str(Path(polls_file).name), "sha256": sha256_file(polls_file)},
            "election_results": {"path": str(Path(election_results).name), "sha256": sha256_file(election_results)},
            "certified_results": [{"path": str(Path(m).name), "sha256": sha256_file(m)} for m in manifests],
        },
        "history_start_election": HISTORY_START_ELECTION.isoformat(),
        "historical_result_availability_lag_days": HISTORICAL_RESULT_AVAILABILITY_LAG_DAYS,
        "elections": [{"date": e.obs_date.isoformat(), "available": e.available.isoformat()} for e in elections],
        "houses": result.houses,
        "polls_used": len(dataset.polls),
        "exclusions": {reason: {"count": len(ids), "poll_ids": sorted(ids)}
                       for reason, ids in sorted(dataset.exclusions.items())},
        "imputed_sample_size": sorted(dataset.imputed_sample_size),
        "approximate_fieldwork": sorted(dataset.approximate_fieldwork),
        "overlap_discounted_count": len(dataset.overlap_discounted),
        "segments": [
            {
                "hyperparameters_fit_date": s.fit_date.isoformat(),
                "reference_election": s.reference_election,
                "observations": s.n_obs,
                "log_likelihood": round(s.loglik, 4),
                "optimizer_iterations": s.optimizer_iterations,
                "optimizer_converged": s.optimizer_converged,
                "hyperparameters": {k: float(f"{v:.6g}") for k, v in s.params.to_dict().items()},
                # Full precision, so a stored fit reproduces the series exactly.
                "hyperparameters_exact": dict(s.params.to_dict()),
                "consensus_weights": (
                    {h: round(w, 6) for h, w in s.consensus_weights.items()} if s.consensus_weights else None
                ),
            }
            for s in result.segments
        ],
        "rows": len(result.rows),
        "first_date": result.rows[0].day.isoformat() if result.rows else None,
        "last_date": result.rows[-1].day.isoformat() if result.rows else None,
        "stats": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.stats.items()},
    }


def build_outputs(polls_file: Path, election_results: Path, manifests: Sequence[Path], output_dir: Path,
                  spec: AggregateSpec, *, reuse_stored_fits: bool = False) -> dict:
    """Build ``spec`` from the inputs and write its timeseries, metadata and polls."""

    output_dir = Path(output_dir)
    elections = load_elections(election_results, manifests)
    dataset = load_polls(polls_file, [e.obs_date for e in elections])
    fits = stored_fits(output_dir / METADATA_FILENAME, spec) if reuse_stored_fits else None
    result = build_aggregate(dataset.polls, elections, spec=spec, stored_fits=fits)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_timeseries(result, output_dir / TIMESERIES_FILENAME, version=spec.version)
    metadata = aggregate_metadata(result, dataset, elections, spec, polls_file, election_results, manifests)
    metadata_path = output_dir / METADATA_FILENAME
    tmp = metadata_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(metadata_path)
    write_individual_polls(polls_file, {o.key for o in dataset.polls}, output_dir / POLLS_FILENAME)
    return metadata
