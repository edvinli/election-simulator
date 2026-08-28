"""Export compact, validated static forecast JSON atomically.

The exporter is deliberately downstream of the frozen Python simulator.  It
does not run Monte Carlo in a browser and it never writes into the immutable
prospective archive.  A complete publication is assembled in a sibling
temporary directory, validated, and published as an immutable version behind
one atomically replaced current.json pointer.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

from scripts.simulator.config import (
    DEFAULT_MAJORITY_THRESHOLD,
    MODEL_PARTIES_9,
    PARLIAMENTARY_PARTIES_8,
)
from scripts.simulator.pipeline import build_canonical_summary_dict
from scripts.simulator.reproducibility import (
    GENERATION_ID_PATTERN,
    SOURCE_REPOSITORY,
    build_generation_id,
    compute_file_sha256,
    require_certified_source_provenance,
    resolve_source_repository,
)


# 1.0 is the pre-extraction publication schema.  1.1 adds ``source_repository``
# to metadata and manifests.  1.2 adds the precomputed coalition-builder
# summaries to groups.json.  1.3 adds exact contiguous integer seat
# histograms for those coalitions.  Validators accept every historical version
# so existing immutable publications stay readable and are never rewritten;
# only 1.3 is written by this exporter.
PUBLICATION_SCHEMA_VERSION = "1.3"
SUPPORTED_PUBLICATION_SCHEMA_VERSIONS: tuple[str, ...] = ("1.0", "1.1", "1.2", "1.3")
PUBLICATION_FILES: tuple[str, ...] = (
    "forecast.json",
    "parties.json",
    "seats.json",
    "groups.json",
    "calibration.json",
    "metadata.json",
)

# The publication is a public artifact, so it must never carry the local
# filesystem layout of whichever machine generated it.  Calibration sources are
# addressed by their stable logical name under the processed data root instead.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOGICAL_CALIBRATION_ROOT = "data/processed"
CALIBRATION_SOURCE_RELATIVE_PARTS: dict[str, tuple[str, ...]] = {
    "seat_hindcast": ("seat_hindcasts", "seat_hindcast_summary.json"),
    "vote_share_hindcast": ("vote_share_calibration", "vote_share_summary_2018_2022.json"),
    "pop_head_to_head": ("pop_baseline_benchmark", "benchmark_report.json"),
}

COALITION_BUILDER_SUMMARY_FIELDS: tuple[str, ...] = (
    "mask",
    "parties",
    "mean_seats",
    "median_seats",
    "p05_seats",
    "p10_seats",
    "p25_seats",
    "p75_seats",
    "p90_seats",
    "p95_seats",
    "prob_majority",
)
COALITION_BUILDER_HISTOGRAM_FIELD = "seat_histogram"
COALITION_BUILDER_ENTRY_FIELDS: tuple[str, ...] = (
    *COALITION_BUILDER_SUMMARY_FIELDS,
    COALITION_BUILDER_HISTOGRAM_FIELD,
)


def _public_source_path(path: Path, relative_parts: Sequence[str]) -> str:
    """Return a stable POSIX path safe to publish for a calibration artifact.

    A file inside the repository is serialised repo-relative.  Anything else --
    a temporary directory in tests, a mounted artifact store -- falls back to
    the logical ``data/processed`` name of that artifact.  Neither form can
    contain a local absolute path, and both are byte-identical across machines
    so the deterministic publication hash stays reproducible.
    """

    try:
        return path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return "/".join((LOGICAL_CALIBRATION_ROOT, *relative_parts))


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strip_runtime_timestamps(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_runtime_timestamps(item)
            for key, item in value.items()
            if key not in {"generated_at_utc", "published_at_utc", "updated_at_utc"}
        }
    if isinstance(value, list):
        return [_strip_runtime_timestamps(item) for item in value]
    return value


def _histogram(values: np.ndarray, *, lower: float, upper: float, width: float) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Cannot export an empty distribution")
    edges = np.arange(lower, upper + width, width, dtype=np.float64)
    if edges[-1] < upper:
        edges = np.append(edges, upper)
    counts, actual_edges = np.histogram(arr, bins=edges)
    return {
        "bin_edges": [round(float(x), 6) for x in actual_edges],
        "counts": [int(x) for x in counts],
        "total": int(arr.size),
        "quantiles": {
            "p05": round(float(np.quantile(arr, 0.05)), 6),
            "p25": round(float(np.quantile(arr, 0.25)), 6),
            "p50": round(float(np.quantile(arr, 0.50)), 6),
            "p75": round(float(np.quantile(arr, 0.75)), 6),
            "p95": round(float(np.quantile(arr, 0.95)), 6),
        },
    }


def _representative_seat_allocation(result: Any) -> dict[str, Any]:
    """Select one coherent simulated parliament for the public display.

    Marginal party medians are useful summaries, but they need not form a
    legal parliament when added together.  The representative display is the
    simulated row with minimum squared distance to the vector of marginal
    medians.  ``argmin`` is deterministic and therefore stable for a fixed
    simulation payload; every row has already passed the 349-seat invariant in
    the production pipeline.
    """

    seats = np.asarray(result.seats_matrix)
    if seats.ndim != 2 or seats.shape[1] != len(PARLIAMENTARY_PARTIES_8) or seats.shape[0] == 0:
        raise ValueError("Cannot select a representative allocation from an invalid seat matrix")
    if not np.issubdtype(seats.dtype, np.integer) or np.any(seats < 0):
        raise ValueError("Cannot select a representative allocation from invalid seat values")
    seat_totals = np.sum(seats, axis=1)
    if not np.all(seat_totals == 349):
        raise ValueError("Cannot select a representative allocation before the 349-seat invariant holds")
    marginal_medians = np.median(seats, axis=0)
    distances = np.sum((seats.astype(np.float64) - marginal_medians) ** 2, axis=1)
    draw_index = int(np.argmin(distances))
    allocation = {
        party: int(seats[draw_index, index])
        for index, party in enumerate(PARLIAMENTARY_PARTIES_8)
    }
    total_seats = int(sum(allocation.values()))
    if total_seats != 349:
        raise ValueError("Representative allocation does not contain exactly 349 seats")
    return {
        "method": "joint_simulation_draw_closest_to_marginal_medians",
        "draw_index": draw_index,
        "seats": allocation,
        "total_seats": total_seats,
    }


def _compact_integer_seat_histogram(group_seats: np.ndarray) -> dict[str, Any]:
    """Encode integer seat draws as a contiguous, exact count vector.

    ``group_seats`` is deliberately supplied by the caller after summing the
    original joint ``seats_matrix``.  This helper never sees marginal
    summaries and never samples or reconstructs a distribution.  The first
    and last bins are the observed minimum and maximum, making the encoding
    compact while retaining zero-count seats in between them.
    """

    values = np.asarray(group_seats)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Cannot encode an empty coalition seat distribution")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Coalition seat draws must be integer values")
    if np.any(values < 0) or np.any(values > 349):
        raise ValueError("Coalition seat draws must be within 0–349")

    min_seats = int(np.min(values))
    max_seats = int(np.max(values))
    counts = np.bincount(values.astype(np.int64), minlength=max_seats + 1)[min_seats:]
    if counts.size != max_seats - min_seats + 1:
        raise ValueError("Coalition seat histogram is not contiguous")
    if int(counts[0]) <= 0 or int(counts[-1]) <= 0:
        raise ValueError("Coalition seat histogram bounds must be observed")
    if int(np.sum(counts, dtype=np.int64)) != int(values.size):
        raise ValueError("Coalition seat histogram counts do not sum to the draws")
    return {
        "min_seats": min_seats,
        "counts": [int(count) for count in counts],
    }


def _coalition_draws(seats_matrix: np.ndarray, party_order: Sequence[str], mask: int) -> np.ndarray:
    """Return one coalition's seats in every original joint simulation draw."""

    indices = [index for index in range(len(party_order)) if mask & (1 << index)]
    if not indices:
        return np.zeros(seats_matrix.shape[0], dtype=np.int64)
    # The source matrix is already validated as integer seat draws by the
    # caller.  Summing columns preserves draw-by-draw dependence exactly.
    return np.sum(seats_matrix[:, indices], axis=1, dtype=np.int64)


def _build_coalition_builder(result: Any) -> dict[str, Any]:
    """Build the compact coalition lookup from the joint seat draws.

    ``SimulationResult.summarize_group`` delegates to ``GroupSummaryHelper``
    over the result's original ``seats_matrix``.  Calling that method once for
    every bitmask intentionally keeps coalition quantiles and majority logic
    identical to the existing published group summaries.  No marginal party
    summaries, reconstructed distributions, or browser-side sampling enter
    this contract.  Histograms are counted directly from the same matrix,
    rather than by multiplying the existing floating-point probability map by
    the sample count.
    """

    party_order = list(PARLIAMENTARY_PARTIES_8)
    seats_matrix = np.asarray(result.seats_matrix)
    if seats_matrix.ndim != 2 or seats_matrix.shape[1] != len(party_order) or seats_matrix.shape[0] == 0:
        raise ValueError("Cannot publish coalition histograms from an invalid seat matrix")
    if not np.issubdtype(seats_matrix.dtype, np.integer):
        raise ValueError("Cannot publish coalition histograms from non-integer seat draws")
    if np.any(seats_matrix < 0) or np.any(seats_matrix > 349):
        raise ValueError("Cannot publish coalition histograms from out-of-range seat draws")
    if not np.all(np.sum(seats_matrix, axis=1) == 349):
        raise ValueError("Cannot publish coalition histograms before the 349-seat invariant holds")
    expected_samples = getattr(result.summary, "total_samples", None)
    if expected_samples != seats_matrix.shape[0]:
        raise ValueError("Coalition histogram sample count does not match the simulation result")

    coalitions: dict[str, dict[str, Any]] = {}
    for mask in range(1 << len(party_order)):
        parties = [
            party
            for index, party in enumerate(party_order)
            if mask & (1 << index)
        ]
        summary = result.summarize_group(
            parties,
            majority_threshold=DEFAULT_MAJORITY_THRESHOLD,
        )
        group_seats = _coalition_draws(seats_matrix, party_order, mask)
        seat_histogram = _compact_integer_seat_histogram(group_seats)
        coalitions[str(mask)] = {
            "mask": mask,
            "parties": parties,
            "mean_seats": float(summary.mean_seats),
            "median_seats": int(summary.median_seats),
            "p05_seats": int(summary.p05_seats),
            "p10_seats": int(summary.p10_seats),
            "p25_seats": int(summary.p25_seats),
            "p75_seats": int(summary.p75_seats),
            "p90_seats": int(summary.p90_seats),
            "p95_seats": int(summary.p95_seats),
            "prob_majority": float(summary.prob_majority),
            COALITION_BUILDER_HISTOGRAM_FIELD: seat_histogram,
        }
    return {
        "party_order": party_order,
        "encoding": "bitmask",
        "majority_threshold": DEFAULT_MAJORITY_THRESHOLD,
        "coalitions": coalitions,
    }


def _is_finite_number(value: Any) -> bool:
    """Return whether a JSON-compatible numeric value is finite and non-bool."""

    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def _histogram_count(histogram: Mapping[str, Any], seats: int) -> int:
    """Return one integer bin count, treating seats outside support as zero."""

    minimum = int(histogram["min_seats"])
    counts = histogram["counts"]
    offset = seats - minimum
    if offset < 0 or offset >= len(counts):
        return 0
    return int(counts[offset])


def _histogram_value_at_order_index(minimum: int, counts: Sequence[int], index: int) -> int:
    """Return the sorted integer value at a zero-based order statistic index."""

    remaining = index
    for offset, count in enumerate(counts):
        if remaining < count:
            return minimum + offset
        remaining -= count
    raise ValueError("Histogram order-statistic index is outside its support")


def _histogram_quantile(minimum: int, counts: Sequence[int], quantile: float) -> int:
    """Match NumPy's default linear percentile followed by integer truncation.

    Existing group summaries use ``int(np.percentile(values, q))`` with the
    default ``linear`` method.  This computes the same convention directly
    from counts without materialising raw draws in a validator.
    """

    total = int(sum(counts))
    if total <= 0:
        raise ValueError("Cannot calculate a quantile from an empty histogram")
    position = (total - 1) * float(quantile)
    lower_index = int(np.floor(position))
    upper_index = int(np.ceil(position))
    lower = _histogram_value_at_order_index(minimum, counts, lower_index)
    upper = _histogram_value_at_order_index(minimum, counts, upper_index)
    gamma = position - lower_index
    # NumPy's private _lerp helper deliberately evaluates the upper-end
    # expression for gamma >= 0.5.  This avoids a one-ULP downward drift in
    # cases such as an interpolated value that is mathematically integral;
    # that drift would change the existing int(np.percentile(...)) result.
    if gamma >= 0.5:
        interpolated = upper - (upper - lower) * (1.0 - gamma)
    else:
        interpolated = lower + (upper - lower) * gamma
    return int(interpolated)


def _validate_coalition_seat_histogram(
    value: Any,
    *,
    expected_samples: int,
    entry: Mapping[str, Any],
    key: str,
) -> dict[str, Any]:
    """Validate one exact contiguous histogram and its published summaries."""

    if not isinstance(value, Mapping) or list(value) != ["min_seats", "counts"]:
        raise ValueError(f"coalition {key} has an invalid seat_histogram")
    minimum = value.get("min_seats")
    counts = value.get("counts")
    if (
        not isinstance(minimum, int)
        or isinstance(minimum, bool)
        or not 0 <= minimum <= 349
    ):
        raise ValueError(f"coalition {key} seat_histogram min_seats is outside 0–349")
    if not isinstance(counts, list) or not counts:
        raise ValueError(f"coalition {key} seat_histogram counts must be a non-empty list")
    if any(
        not isinstance(count, int) or isinstance(count, bool) or count < 0
        for count in counts
    ):
        raise ValueError(f"coalition {key} seat_histogram counts must be non-negative integers")
    if counts[0] <= 0 or counts[-1] <= 0:
        raise ValueError(f"coalition {key} seat_histogram bounds must be observed")
    maximum = minimum + len(counts) - 1
    if maximum > 349:
        raise ValueError(f"coalition {key} seat_histogram support exceeds 349 seats")
    if sum(counts) != expected_samples:
        raise ValueError(f"coalition {key} seat_histogram counts do not sum to samples")

    mean = sum((minimum + offset) * count for offset, count in enumerate(counts)) / expected_samples
    if not np.isclose(mean, float(entry["mean_seats"]), rtol=0.0, atol=1e-12):
        raise ValueError(f"coalition {key} mean_seats disagrees with seat_histogram")
    quantile_fields = (
        ("p05_seats", 0.05),
        ("p10_seats", 0.10),
        ("p25_seats", 0.25),
        ("median_seats", 0.50),
        ("p75_seats", 0.75),
        ("p90_seats", 0.90),
        ("p95_seats", 0.95),
    )
    for field, quantile in quantile_fields:
        if _histogram_quantile(minimum, counts, quantile) != entry[field]:
            raise ValueError(f"coalition {key} {field} disagrees with seat_histogram")

    threshold = DEFAULT_MAJORITY_THRESHOLD
    majority_count = sum(
        count
        for offset, count in enumerate(counts)
        if minimum + offset >= threshold
    )
    probability = majority_count / expected_samples
    if not np.isclose(probability, float(entry["prob_majority"]), rtol=0.0, atol=1e-12):
        raise ValueError(f"coalition {key} prob_majority disagrees with seat_histogram")
    return {"min_seats": minimum, "counts": counts}


def _validate_coalition_builder(
    value: Mapping[str, Any],
    *,
    expected_samples: int | None = None,
    require_histogram: bool = False,
) -> None:
    """Validate a coalition lookup without seeing raw draws.

    Schema 1.2 entries contain only the original compact summaries.  Schema
    1.3 adds an exact contiguous integer histogram and cross-checks every
    summary value against it.
    """

    expected_keys = ["party_order", "encoding", "majority_threshold", "coalitions"]
    if list(value) != expected_keys:
        raise ValueError("coalition_builder has unexpected fields")
    if value.get("party_order") != list(PARLIAMENTARY_PARTIES_8):
        raise ValueError("coalition_builder has incorrect canonical party order")
    if value.get("encoding") != "bitmask":
        raise ValueError("coalition_builder must use bitmask encoding")
    threshold = value.get("majority_threshold")
    if (
        not isinstance(threshold, int)
        or isinstance(threshold, bool)
        or threshold != DEFAULT_MAJORITY_THRESHOLD
    ):
        raise ValueError("coalition_builder has incorrect majority threshold")

    coalitions = value.get("coalitions")
    if not isinstance(coalitions, Mapping):
        raise ValueError("coalition_builder.coalitions must be an object")
    expected_mask_keys = [str(mask) for mask in range(1 << len(PARLIAMENTARY_PARTIES_8))]
    if list(coalitions) != expected_mask_keys:
        raise ValueError("coalition_builder must contain keys \"0\" through \"255\" in order")

    if require_histogram and (
        not isinstance(expected_samples, int)
        or isinstance(expected_samples, bool)
        or expected_samples <= 0
    ):
        raise ValueError("Schema 1.3 coalition histograms require a positive sample count")

    expected_entry_fields = (
        COALITION_BUILDER_ENTRY_FIELDS if require_histogram else COALITION_BUILDER_SUMMARY_FIELDS
    )
    validated_histograms: dict[int, dict[str, Any]] = {}
    for mask, key in enumerate(expected_mask_keys):
        entry = coalitions[key]
        if not isinstance(entry, Mapping):
            raise ValueError(f"coalition {key} must be an object")
        if list(entry) != list(expected_entry_fields):
            raise ValueError(f"coalition {key} has unexpected or unordered fields")
        entry_mask = entry.get("mask")
        if entry_mask != mask or isinstance(entry_mask, bool) or not isinstance(entry_mask, int):
            raise ValueError(f"coalition {key} has an inconsistent mask")
        expected_parties = [
            party
            for index, party in enumerate(PARLIAMENTARY_PARTIES_8)
            if mask & (1 << index)
        ]
        if entry.get("parties") != expected_parties:
            raise ValueError(f"coalition {key} does not match the canonical bitmask membership")

        mean = entry.get("mean_seats")
        if not _is_finite_number(mean) or not 0 <= float(mean) <= 349:
            raise ValueError(f"coalition {key} mean_seats is outside 0–349")
        quantiles = [entry.get(field) for field in (
            "p05_seats",
            "p10_seats",
            "p25_seats",
            "median_seats",
            "p75_seats",
            "p90_seats",
            "p95_seats",
        )]
        if any(
            not isinstance(seats, int)
            or isinstance(seats, bool)
            or not 0 <= seats <= 349
            for seats in quantiles
        ):
            raise ValueError(f"coalition {key} quantiles must be integer seats in 0–349")
        if quantiles != sorted(quantiles):
            raise ValueError(f"coalition {key} quantiles are not monotone")
        probability = entry.get("prob_majority")
        if not _is_finite_number(probability) or not 0 <= float(probability) <= 1:
            raise ValueError(f"coalition {key} probability must be between 0 and 1")

        if require_histogram:
            validated_histograms[mask] = _validate_coalition_seat_histogram(
                entry.get(COALITION_BUILDER_HISTOGRAM_FIELD),
                expected_samples=expected_samples,
                entry=entry,
                key=key,
            )

        if mask == 0:
            if any(seats != 0 for seats in quantiles) or mean != 0 or probability != 0:
                raise ValueError("Empty coalition must have zero seats and zero majority probability")
            if require_histogram and validated_histograms[mask] != {
                "min_seats": 0,
                "counts": [expected_samples],
            }:
                raise ValueError("Empty coalition must contain only zero-seat draws")
        elif mask == (1 << len(PARLIAMENTARY_PARTIES_8)) - 1:
            if any(seats != 349 for seats in quantiles) or mean != 349 or probability != 1:
                raise ValueError("Full coalition must have 349 seats and certainty of majority")
            if require_histogram and validated_histograms[mask] != {
                "min_seats": 349,
                "counts": [expected_samples],
            }:
                raise ValueError("Full coalition must contain only 349-seat draws")

    if require_histogram:
        assert expected_samples is not None
        # Every underlying draw has exactly 349 seats.  In the compact public
        # representation this is the reflection identity between each mask
        # and its complement: count(mask, s) == count(~mask, 349-s).
        full_mask = (1 << len(PARLIAMENTARY_PARTIES_8)) - 1
        for mask in range(1 << len(PARLIAMENTARY_PARTIES_8)):
            complement = full_mask ^ mask
            if mask > complement:
                continue
            for seats in range(350):
                if _histogram_count(validated_histograms[mask], seats) != _histogram_count(
                    validated_histograms[complement], 349 - seats
                ):
                    raise ValueError(
                        f"coalition {mask} and complement {complement} violate the 349-seat identity"
                    )


def _build_contracts(
    result: Any,
    *,
    generated_at_utc: str,
    calibration_dir: Path | None,
    prior_snapshot: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    summary = build_canonical_summary_dict(result)
    manifest = dict(result.manifest)
    if manifest.get("source_worktree_clean") is not True:
        raise ValueError(
            "Certified static publication requires source_worktree_clean to be the boolean true"
        )
    deterministic_payload_sha256 = summary["deterministic_payload_sha256"]
    # ``build_canonical_summary_dict`` is intentionally compact for the
    # archive.  The publication contract additionally exposes the inner
    # quantiles needed to display all three predictive intervals.
    parties = {party: dict(value) for party, value in summary["parties"].items()}
    for party in MODEL_PARTIES_9:
        party_summary = result.summary.parties[party]
        parties[party].update({
            "vote_share_p10": round(float(party_summary.vote_share_p10 * 100.0), 3),
            "vote_share_p25": round(float(party_summary.vote_share_p25 * 100.0), 3),
            "vote_share_p75": round(float(party_summary.vote_share_p75 * 100.0), 3),
            "vote_share_p90": round(float(party_summary.vote_share_p90 * 100.0), 3),
            "seats_p10": int(party_summary.seats_p10),
            "seats_p25": int(party_summary.seats_p25),
            "seats_p75": int(party_summary.seats_p75),
            "seats_p90": int(party_summary.seats_p90),
        })
    for party in PARLIAMENTARY_PARTIES_8:
        if party not in parties:
            raise ValueError(f"Canonical result is missing party {party}")
    if "REST" not in parties:
        raise ValueError("Canonical result is missing aggregate REST")

    party_rows: list[dict[str, Any]] = []
    for party in MODEL_PARTIES_9:
        row = dict(parties[party])
        row["party"] = party
        row["eligible_for_national_threshold"] = party in PARLIAMENTARY_PARTIES_8
        row["threshold_probability_defined"] = party in PARLIAMENTARY_PARTIES_8
        party_rows.append(row)

    vote_distributions = {
        party: _histogram(result.vote_shares_matrix[:, index], lower=0.0, upper=100.0, width=0.25)
        for index, party in enumerate(MODEL_PARTIES_9)
    }
    seat_distributions = {
        party: _histogram(result.seats_matrix[:, index], lower=0.0, upper=349.0, width=1.0)
        for index, party in enumerate(PARLIAMENTARY_PARTIES_8)
    }
    representative_allocation = _representative_seat_allocation(result)
    change_since_prior: dict[str, Any]
    if prior_snapshot is None:
        change_since_prior = {
            "status": "NOT_AVAILABLE_NO_PRIOR_SNAPSHOT",
            "prior_as_of": None,
            "prior_snapshot_id": None,
            "prior_deterministic_payload_sha256": None,
            "vote_share_median_change_pp": {},
            "seat_median_change": {},
        }
    else:
        prior_vote_summary = prior_snapshot.get("national_vote_summary", {})
        prior_seat_summary = prior_snapshot.get("seat_summary", {})
        vote_changes: dict[str, float] = {}
        for party in MODEL_PARTIES_9:
            current_value = parties[party].get("vote_share_median")
            prior_value = prior_vote_summary.get(party, {}).get("vote_share_median")
            if current_value is not None and prior_value is not None:
                vote_changes[party] = round(float(current_value) - float(prior_value), 3)
        seat_changes: dict[str, int] = {}
        for party in PARLIAMENTARY_PARTIES_8:
            current_value = parties[party].get("seats_median")
            prior_value = prior_seat_summary.get(party, {}).get("median")
            if current_value is not None and prior_value is not None:
                seat_changes[party] = int(current_value) - int(prior_value)
        change_since_prior = {
            "status": "AVAILABLE",
            "prior_as_of": prior_snapshot.get("as_of"),
            "prior_snapshot_id": prior_snapshot.get("snapshot_id"),
            "prior_deterministic_payload_sha256": prior_snapshot.get("deterministic_payload_sha256"),
            "vote_share_median_change_pp": vote_changes,
            "seat_median_change": seat_changes,
        }
    groups = {
        "tido": result.summarize_group(["M", "SD", "KD", "L"]).__dict__,
        "red_green_center": result.summarize_group(["S", "V", "MP", "C"]).__dict__,
    }
    for row in groups.values():
        row["parties"] = list(row["parties"])
    coalition_builder = _build_coalition_builder(result)

    calibration: dict[str, Any] = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "status": "AVAILABLE_IF_ARTIFACTS_EXIST",
        "evidence_type": "retrospective_historical_not_holdout",
        "source_files": {},
    }
    if calibration_dir is not None:
        processed_root = calibration_dir
        calibration_paths = {
            key: processed_root.joinpath(*relative_parts)
            for key, relative_parts in CALIBRATION_SOURCE_RELATIVE_PARTS.items()
        }
        for key, path in calibration_paths.items():
            path = path.resolve()
            if path.exists():
                with path.open(encoding="utf-8") as handle:
                    source_value = json.load(handle)
                # Keep the publication compact: retain status/aggregates, not
                # per-draw or per-case evidence in the website contract.
                if key == "pop_head_to_head":
                    source_value = {
                        "benchmark_status": source_value.get("benchmark_status"),
                        "aggregate_by_evaluation": source_value.get("aggregate_by_evaluation"),
                        "aggregate_by_horizon": source_value.get("aggregate_by_horizon"),
                        "threshold_support_diagnostic": source_value.get("threshold_support_diagnostic"),
                        "underdispersion_diagnostic": source_value.get("underdispersion_diagnostic"),
                        "comparison_decision": source_value.get("comparison_decision"),
                        "evidence_type": source_value.get("evidence_type"),
                    }
                else:
                    source_value = {
                        "evidence_type": calibration["evidence_type"],
                        "summary": source_value.get("summary", source_value.get("by_model_overall", source_value)),
                    }
                calibration["source_files"][key] = {
                    "path": _public_source_path(path, CALIBRATION_SOURCE_RELATIVE_PARTS[key]),
                    "sha256": compute_file_sha256(path),
                    "summary": source_value,
                }
    if not calibration["source_files"]:
        calibration["status"] = "NOT_AVAILABLE"
        calibration["reason"] = "No validation summary artifacts were found"

    forecast = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "as_of": summary["as_of"],
        "election_date": summary["election_date"],
        "model": {"name": "ElectionSimulator", "version": manifest.get("model_version"), "candidate": "A"},
        "total_samples": int(summary["total_samples"]),
        "parties": parties,
        "threshold_probabilities_4pct": {
            party: parties[party]["prob_above_4pct"] for party in PARLIAMENTARY_PARTIES_8
        },
        "groups": summary["blocs"],
        "change_since_prior": change_since_prior,
        "deterministic_payload_sha256": deterministic_payload_sha256,
        "rest_semantics": "REST is aggregate vote mass for modeled-as-ineligible parties; it cannot independently qualify or receive seats.",
    }
    party_contract = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "party_order": list(MODEL_PARTIES_9),
        "parties": party_rows,
        "rest_semantics": forecast["rest_semantics"],
        "deterministic_payload_sha256": deterministic_payload_sha256,
    }
    seat_contract = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "party_order": list(PARLIAMENTARY_PARTIES_8),
        "total_seats": 349,
        "seat_distributions": seat_distributions,
        "seat_summary": {
            party: {
                "mean": parties[party]["seats_mean"],
                "median": parties[party]["seats_median"],
                "p05": parties[party]["seats_p05"],
                "p95": parties[party]["seats_p95"],
            }
            for party in PARLIAMENTARY_PARTIES_8
        },
        "representative_allocation": representative_allocation,
        "rest_semantics": forecast["rest_semantics"],
        "deterministic_payload_sha256": deterministic_payload_sha256,
    }
    group_contract = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "majority_threshold": 175,
        "groups": groups,
        "coalition_builder": coalition_builder,
        "note": "Groups are configurable summaries over parliamentary-party seat draws; REST is never included as an eligible party.",
        "deterministic_payload_sha256": deterministic_payload_sha256,
    }
    calibration["deterministic_payload_sha256"] = deterministic_payload_sha256
    metadata = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc,
        "as_of": summary["as_of"],
        "election_date": summary["election_date"],
        "model": {"name": "ElectionSimulator", "version": manifest.get("model_version"), "candidate": "A"},
        "source_repository": SOURCE_REPOSITORY,
        "source_git_commit": manifest.get("source_git_commit", manifest.get("git_commit")),
        "source_worktree_clean": manifest.get("source_worktree_clean"),
        "input_hashes": {
            key: manifest.get(key)
            for key in ("poll_data_hash", "election_data_hash", "mandate_data_hash", "geography_data_hash", "model_config_hash")
        },
        "deterministic_payload_sha256": deterministic_payload_sha256,
        "rest_semantics": forecast["rest_semantics"],
        "interval_semantics": "Central empirical predictive intervals at 50%, 80%, and 90%; they are not confidence intervals.",
        "validation_note": "Historical validation is retrospective and not independent holdout validation.",
    }
    return {
        "forecast.json": forecast,
        "parties.json": party_contract,
        "seats.json": seat_contract,
        "groups.json": group_contract,
        "calibration.json": calibration,
        "metadata.json": metadata,
    }


def validate_publication_contract(contracts: Mapping[str, Mapping[str, Any]]) -> None:
    """Validate all compact publication contracts before any directory swap."""
    if set(contracts) != set(PUBLICATION_FILES):
        raise ValueError(f"Publication files must be exactly {list(PUBLICATION_FILES)}")
    for name, value in contracts.items():
        if value.get("schema_version") not in SUPPORTED_PUBLICATION_SCHEMA_VERSIONS:
            raise ValueError(f"{name} has unsupported schema version")
    forecast = contracts["forecast.json"]
    parties = contracts["parties.json"]
    seats = contracts["seats.json"]
    groups = contracts["groups.json"]
    if parties["party_order"] != list(MODEL_PARTIES_9):
        raise ValueError("parties.json has incorrect canonical party order")
    if seats["party_order"] != list(PARLIAMENTARY_PARTIES_8) or seats["total_seats"] != 349:
        raise ValueError("seats.json has incorrect party order or seat total")
    if "REST" not in parties["party_order"] or "REST" in seats["party_order"]:
        raise ValueError("REST must be present in vote contracts and absent from seat contracts")
    rest_rows = [row for row in parties["parties"] if row["party"] == "REST"]
    if len(rest_rows) != 1 or rest_rows[0]["eligible_for_national_threshold"]:
        raise ValueError("REST must be explicitly marked ineligible")
    if forecast["threshold_probabilities_4pct"].keys() != set(PARLIAMENTARY_PARTIES_8):
        raise ValueError("Threshold probabilities must cover exactly the eight parliamentary parties")
    group_threshold = groups.get("majority_threshold")
    if (
        not isinstance(group_threshold, int)
        or isinstance(group_threshold, bool)
        or group_threshold != DEFAULT_MAJORITY_THRESHOLD
    ):
        raise ValueError("Group majority threshold must be 175")
    if groups.get("schema_version") == "1.2":
        coalition_builder = groups.get("coalition_builder")
        if not isinstance(coalition_builder, Mapping):
            raise ValueError("Schema 1.2 groups.json must include coalition_builder")
        _validate_coalition_builder(coalition_builder)
    elif groups.get("schema_version") == "1.3":
        coalition_builder = groups.get("coalition_builder")
        if not isinstance(coalition_builder, Mapping):
            raise ValueError("Schema 1.3 groups.json must include coalition_builder")
        total_samples = forecast.get("total_samples")
        if (
            not isinstance(total_samples, int)
            or isinstance(total_samples, bool)
            or total_samples <= 0
        ):
            raise ValueError("Schema 1.3 forecast.json must include a positive total_samples")
        _validate_coalition_builder(
            coalition_builder,
            expected_samples=total_samples,
            require_histogram=True,
        )
    elif "coalition_builder" in groups:
        raise ValueError("coalition_builder is only valid in schema 1.2 or 1.3 groups.json")
    metadata = contracts["metadata.json"]
    if not metadata.get("deterministic_payload_sha256"):
        raise ValueError("metadata.json must link the deterministic simulation payload")
    # An unresolvable Git commit and a dirty worktree are both hard
    # certification failures; neither may reach a published artifact.
    require_certified_source_provenance(metadata)
    # Missing on historical 1.0 artifacts, where it means the original
    # repository; required on everything this exporter writes.
    if metadata.get("schema_version") != "1.0" and metadata.get("source_repository") != SOURCE_REPOSITORY:
        raise ValueError(f"metadata.json must record source_repository={SOURCE_REPOSITORY!r}")
    payload_hash = metadata["deterministic_payload_sha256"]
    for name, contract in contracts.items():
        if contract.get("deterministic_payload_sha256") != payload_hash:
            raise ValueError(f"{name} does not link the common deterministic simulation payload")
    representative = seats.get("representative_allocation")
    if not isinstance(representative, dict):
        raise ValueError("seats.json must include a representative joint allocation")
    representative_seats = representative.get("seats")
    if not isinstance(representative_seats, dict):
        raise ValueError("Representative allocation must contain a seat mapping")
    if set(representative_seats) != set(PARLIAMENTARY_PARTIES_8):
        raise ValueError("Representative allocation must cover exactly the eight parliamentary parties")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in representative_seats.values()):
        raise ValueError("Representative allocation seats must be non-negative integers")
    if sum(representative_seats.values()) != 349 or representative.get("total_seats") != 349:
        raise ValueError("Representative allocation must contain exactly 349 seats")


def _validate_publication_version(
    root: Path,
    *,
    expected_generation: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate one immutable version directory and its manifest hashes."""

    contracts: dict[str, dict[str, Any]] = {}
    for filename in PUBLICATION_FILES:
        path = root / filename
        # Every published version file must be a real file.  A symlink would
        # not survive static hosting and would break the immutability promise.
        if path.is_symlink():
            raise ValueError(f"Published contract must be a real file, not a symlink: {path}")
        if not path.is_file():
            raise ValueError(f"Published contract is missing {path}")
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"Published contract is not a JSON object: {path}")
        contracts[filename] = value
    validate_publication_contract(contracts)
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError(f"Published manifest must be a real file, not a symlink: {manifest_path}")
    if not manifest_path.is_file():
        raise ValueError(f"Published manifest is missing: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") not in SUPPORTED_PUBLICATION_SCHEMA_VERSIONS:
        raise ValueError("Published manifest has unsupported schema version")
    if manifest.get("schema_version") != contracts["metadata.json"].get("schema_version"):
        raise ValueError("Published manifest and metadata disagree on schema version")
    # Historical 1.0 artifacts legitimately carry no source_repository and are
    # read as belonging to the original repository; they are never rewritten.
    if resolve_source_repository(manifest.get("source_repository")) != resolve_source_repository(
        contracts["metadata.json"].get("source_repository")
    ):
        raise ValueError("Published manifest and metadata disagree on source repository")
    if manifest.get("publication_state") != "COMPLETE":
        raise ValueError("Published manifest is not marked COMPLETE")
    generation = manifest.get("publication_generation")
    if not isinstance(generation, str) or not generation:
        raise ValueError("Published manifest has no immutable publication generation")
    if root.name != generation:
        raise ValueError("Publication version path does not match the manifest generation")
    if expected_generation is not None and generation != expected_generation:
        raise ValueError("Current pointer does not match the manifest generation")
    actual_manifest_sha256 = compute_file_sha256(manifest_path)
    if expected_manifest_sha256 is not None and actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError("Current pointer manifest hash does not match the active version")
    if manifest.get("source_worktree_clean") is not True:
        raise ValueError("Published manifest is not certified from a clean source worktree")
    expected_file_hashes = {
        filename: compute_file_sha256(root / filename) for filename in PUBLICATION_FILES
    }
    if manifest.get("publication_files") != expected_file_hashes:
        raise ValueError("Published file hashes do not match manifest")
    expected_deterministic_hashes = {
        filename: _sha256_bytes(_canonical_bytes(_strip_runtime_timestamps(contracts[filename])))
        for filename in PUBLICATION_FILES
    }
    if manifest.get("deterministic_content_hashes") != expected_deterministic_hashes:
        raise ValueError("Published deterministic content hashes do not match manifest")
    expected_manifest_hash = _sha256_bytes(_canonical_bytes(expected_deterministic_hashes))
    if manifest.get("deterministic_content_sha256") != expected_manifest_hash:
        raise ValueError("Published deterministic manifest hash does not match content")
    if manifest.get("deterministic_payload_sha256") != contracts["metadata.json"].get("deterministic_payload_sha256"):
        raise ValueError("Published payload hash is not linked through metadata")
    return manifest


def validate_publication_version(
    version_dir: Path | str,
    *,
    expected_generation: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate one immutable version directory in isolation.

    Public entry point for consumers that address a version directly rather
    than through ``current.json`` — notably the cross-repository site
    publisher, which validates the source and the copied destination
    independently.
    """

    return _validate_publication_version(
        Path(version_dir),
        expected_generation=expected_generation,
        expected_manifest_sha256=expected_manifest_sha256,
    )


def validate_published_directory(output_dir: Path | str) -> dict[str, Any]:
    """Validate the active immutable version addressed by ``current.json``."""

    root = Path(output_dir)
    pointer_path = root / "current.json"
    if not pointer_path.is_file():
        if os.path.lexists(pointer_path):
            raise ValueError("Current publication pointer is not a regular readable file")
        # Backward-compatible read path for an already materialized version.
        # New canonical publications always contain current.json at their
        # stable root, while callers may validate a version directly.
        return _validate_publication_version(root)
    with pointer_path.open(encoding="utf-8") as handle:
        pointer = json.load(handle)
    if not isinstance(pointer, dict):
        raise ValueError("Current publication pointer is not a JSON object")
    if pointer.get("schema_version") not in SUPPORTED_PUBLICATION_SCHEMA_VERSIONS:
        raise ValueError("Current publication pointer has unsupported schema version")
    if pointer.get("publication_state") != "COMPLETE":
        raise ValueError("Current publication pointer is not marked COMPLETE")
    generation = pointer.get("publication_generation")
    relative_path = pointer.get("path")
    manifest_sha256 = pointer.get("manifest_sha256")
    if not isinstance(generation, str) or not generation:
        raise ValueError("Current publication pointer has no generation")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("Current publication pointer has no version path")
    if not isinstance(manifest_sha256, str) or not manifest_sha256:
        raise ValueError("Current publication pointer has no manifest hash")
    relative = Path(relative_path)
    if relative.is_absolute() or relative.parts[:1] != ("versions",) or len(relative.parts) != 2:
        raise ValueError("Current publication pointer path must address one direct versions child")
    versions_root = (root / "versions").resolve()
    version_path = (root / relative).resolve()
    if version_path.parent != versions_root:
        raise ValueError("Current publication pointer escapes the immutable versions directory")
    return _validate_publication_version(
        version_path,
        expected_generation=generation,
        expected_manifest_sha256=manifest_sha256,
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    """Durably flush a directory entry on filesystems that support it."""

    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_replace_pointer(pointer: Path, output: Path) -> None:
    """Replace the consumer pointer in one filesystem operation."""

    os.replace(pointer, output)


def _write_pointer(output: Path, *, generation: str, version_path: Path) -> None:
    """Atomically commit the active immutable version through ``current.json``."""

    pointer_payload = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "publication_state": "COMPLETE",
        "publication_generation": generation,
        "path": str(version_path.relative_to(output)).replace(os.sep, "/"),
        "manifest_sha256": compute_file_sha256(version_path / "manifest.json"),
    }
    pointer = output / f".current-{uuid.uuid4().hex}.tmp"
    _write_json(pointer, pointer_payload)
    try:
        _atomic_replace_pointer(pointer, output / "current.json")
        _fsync_directory(output)
    except Exception:
        if os.path.lexists(pointer):
            os.unlink(pointer)
        raise


def _swap_directory(staging: Path, output: Path, *, generation: str | None = None) -> None:
    """Publish an immutable version by atomically replacing ``current.json``.

    ``output`` is a normal, stable directory so static hosting works without
    special configuration.  Each complete staged payload is moved into
    ``output/versions/<generation>``.  The only authoritative switch is the
    atomic replacement of ``output/current.json``, which is always written
    last; a crash before that rename leaves the previous pointer and version
    loadable, while a crash after it exposes only the complete new version.

    No flat top-level aliases are written.  The pointer plus the immutable
    version directory is the entire canonical contract; any flat files already
    present in ``output`` are legacy artifacts and are left untouched.
    """

    output.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output) and output.is_symlink():
        raise ValueError(
            f"Publication path {output} is a symlink; migrate it to a normal directory with current.json"
        )
    output.mkdir(parents=True, exist_ok=True)
    versions = output / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    if not generation:
        raise ValueError("An immutable publication version requires a generation id")
    generation_name = generation
    version_path = versions / generation_name
    # Never overwrite an existing immutable version.  UUID collisions are
    # unlikely in normal operation, but a deterministic test seed, retry, or
    # caller-supplied generation must fail closed rather than corrupting a
    # version that an older current.json may still address.
    if os.path.lexists(version_path):
        raise FileExistsError(f"Immutable publication version already exists: {version_path}")
    os.replace(staging, version_path)
    _fsync_directory(versions)
    # The version is complete and immutable on disk before the pointer moves.
    # current.json is always the last write of a publication.
    _write_pointer(output, generation=generation_name, version_path=version_path)


def export_static_data(
    result: Any,
    *,
    output_dir: Path | str,
    generated_at_utc: str | None = None,
    calibration_dir: Path | str | None = None,
    prior_snapshot: Mapping[str, Any] | None = None,
    generation_id: str | None = None,
) -> dict[str, Any]:
    """Validate and atomically publish one immutable version behind a pointer.

    ``generation_id`` is the canonical generation shared with the prospective
    archive snapshot this publication was built from.  When omitted it is
    derived from the generation timestamp and the deterministic payload hash,
    so a publication always has a sortable, content-linked identity.
    """
    generated = generated_at_utc or datetime.now(timezone.utc).isoformat()
    parsed = datetime.fromisoformat(generated.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("generated_at_utc must include a timezone")
    contracts = _build_contracts(
        result,
        generated_at_utc=generated,
        calibration_dir=Path(calibration_dir) if calibration_dir else None,
        prior_snapshot=prior_snapshot,
    )
    validate_publication_contract(contracts)
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        for filename in PUBLICATION_FILES:
            _write_json(staging / filename, contracts[filename])
        file_hashes = {
            filename: compute_file_sha256(staging / filename)
            for filename in PUBLICATION_FILES
        }
        deterministic_file_hashes = {
            filename: _sha256_bytes(_canonical_bytes(_strip_runtime_timestamps(contracts[filename])))
            for filename in PUBLICATION_FILES
        }
        deterministic_manifest_hash = _sha256_bytes(_canonical_bytes(deterministic_file_hashes))
        if generation_id is None:
            generation = build_generation_id(
                generated, contracts["metadata.json"]["deterministic_payload_sha256"]
            )
        else:
            generation = str(generation_id)
            if not GENERATION_ID_PATTERN.fullmatch(generation):
                raise ValueError(f"Publication generation id is not web-safe: {generation!r}")
        manifest = {
            "schema_version": PUBLICATION_SCHEMA_VERSION,
            "publication_state": "COMPLETE",
            "publication_generation": generation,
            "generated_at_utc": generated,
            "publication_files": file_hashes,
            "deterministic_content_hashes": deterministic_file_hashes,
            "deterministic_content_sha256": deterministic_manifest_hash,
            "model_version": contracts["metadata.json"]["model"]["version"],
            "source_repository": contracts["metadata.json"]["source_repository"],
            "source_git_commit": contracts["metadata.json"]["source_git_commit"],
            "source_worktree_clean": contracts["metadata.json"]["source_worktree_clean"],
            "deterministic_payload_sha256": contracts["metadata.json"]["deterministic_payload_sha256"],
        }
        _write_json(staging / "manifest.json", manifest)
        _swap_directory(staging, output, generation=generation)
        staging = None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return manifest
