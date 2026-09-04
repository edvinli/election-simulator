"""One-cutoff orchestration without website or publication mutations."""

from __future__ import annotations

import base64
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping

from scripts.simulator.config import PARLIAMENTARY_PARTIES_8
from scripts.simulator.exact_draw_sidecar import collect_latest_certified_generation

from .archive import (
    DEFAULT_ARCHIVE_ROOT,
    ArchiveValidationError,
    CaptureCollisionError,
    ModelCapture,
    append_capture,
    canonical_json_bytes,
    validate_archive,
)
from .botten_ada_capture import BottenAdaCapture, capture_botten_ada
from .time_rules import capture_id_for_date, classify_capture_time, scheduled_cutoff


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ES_ARCHIVE = REPOSITORY_ROOT / "data" / "processed" / "prospective_forecasts"
PARTY_ORDER = tuple(PARLIAMENTARY_PARTIES_8)


class CaptureSourceError(ValueError):
    """Raised when source output cannot be represented honestly."""


def _failed_model_capture(
    *,
    system: str,
    status: str,
    reason: str,
    files: Mapping[str, bytes] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> ModelCapture:
    """Represent a source/parser failure without manufacturing a forecast."""

    return ModelCapture(
        status=status,
        forecast={
            "schema_version": "1.0",
            "system": system,
            "available": False,
            "reason": reason,
            "carry_forward": False,
        },
        provenance={
            "source_error": reason,
            "carry_forward": False,
            **dict(provenance or {}),
        },
        files=files,
    )


def _durable_slot_is_indexed(archive_root: Path, scheduled_date: str) -> bool:
    """Fail closed before collection when a durable slot already exists."""

    index_path = archive_root / "index.json"
    if not index_path.exists():
        return False
    # Validate the whole archive before reading the slot.  A malformed or
    # tampered index must never be treated as an empty archive and replaced.
    # A complete directory may have been installed immediately before a
    # process died while replacing index.json.  Treat that as a recoverable
    # orphan; append_capture will validate and index that exact directory
    # without overwriting it.  Malformed or partial orphans still fail closed.
    validate_archive(archive_root, allow_unindexed_orphans=True)
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArchiveValidationError(f"Cannot read benchmark index before collection: {exc}") from exc
    return any(
        isinstance(row, Mapping) and row.get("scheduled_date") == scheduled_date
        for row in index.get("captures", [])
    )


def _git_blob(repo_root: Path, commit: str, relative_path: str) -> bytes | None:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def _latest_es_poll_input_date(repo_root: Path, snapshot: Mapping[str, Any]) -> tuple[str | None, str]:
    source_commit = str(snapshot.get("source_git_commit", ""))
    relative = "data/processed/pollofpolls/pollofpolls_timeseries.csv"
    blob = _git_blob(repo_root, source_commit, relative)
    if blob is None:
        return None, "UNAVAILABLE_AT_CERTIFIED_SOURCE_COMMIT"
    try:
        rows = list(csv.DictReader(io.StringIO(blob.decode("utf-8-sig"))))
        dates = [row["date"] for row in rows if row.get("date") and row["date"] <= str(snapshot["as_of"])]
    except (UnicodeDecodeError, csv.Error, KeyError):
        return None, "UNPARSEABLE_AT_CERTIFIED_SOURCE_COMMIT"
    return (max(dates), "VERIFIED_FROM_CERTIFIED_POLL_TIMESERIES") if dates else (None, "NO_ELIGIBLE_ROW")


def _normalize_election_simulator(selected: Mapping[str, Any], *, repo_root: Path) -> ModelCapture:
    snapshot = selected.get("forecast")
    if not isinstance(snapshot, Mapping):
        return ModelCapture(
            status="SOURCE_UNAVAILABLE",
            forecast={
                "schema_version": "1.0",
                "system": "election_simulator",
                "available": False,
                "reason": selected.get("status"),
            },
            provenance={
                "selection": selected.get("provenance"),
                "diagnostics": selected.get("diagnostics", []),
                "carry_forward": False,
            },
        )
    if snapshot.get("samples") != 100_000:
        raise CaptureSourceError("Selected ElectionSimulator generation is not the certified 100,000-draw production run")
    summary = snapshot.get("national_vote_summary")
    distributions = snapshot.get("national_vote_distributions")
    if not isinstance(summary, Mapping) or not isinstance(distributions, Mapping):
        raise CaptureSourceError("Selected ElectionSimulator snapshot lacks published vote summaries")
    published_mean = {party: float(summary[party]["vote_share_mean"]) for party in PARTY_ORDER}
    quantiles = {
        party: {
            "0.05": float(distributions[party]["quantiles"]["p05"]),
            "0.25": float(distributions[party]["quantiles"]["p25"]),
            "0.50": float(distributions[party]["quantiles"]["p50"]),
            "0.75": float(distributions[party]["quantiles"]["p75"]),
            "0.95": float(distributions[party]["quantiles"]["p95"]),
        }
        for party in PARTY_ORDER
    }
    exact = selected.get("exact_draws")
    if not isinstance(exact, Mapping):
        exact = {"status": "UNAVAILABLE_NO_VERIFIED_DRAWS"}
    exact_status = str(exact.get("status", "UNAVAILABLE_NO_VERIFIED_DRAWS"))
    draws_verified = exact_status in {"VERIFIED", "REPLAY_VERIFIED"}
    files: dict[str, bytes] = {"source_snapshot.json": canonical_json_bytes(dict(snapshot))}
    if draws_verified:
        try:
            files["draws.npz"] = base64.b64decode(str(exact["draws_bytes_base64"]), validate=True)
            files["draws.json"] = base64.b64decode(str(exact["metadata_bytes_base64"]), validate=True)
        except (KeyError, ValueError) as exc:
            raise CaptureSourceError("Verified ElectionSimulator draws were not supplied as exact sidecar bytes") from exc
    latest_poll_date, latest_poll_status = _latest_es_poll_input_date(repo_root, snapshot)
    forecast = {
        "schema_version": "1.0",
        "system": "election_simulator",
        "available": True,
        "election_date": snapshot.get("election_date"),
        "party_order": list(PARTY_ORDER),
        "vote_share_unit": "percentage_points",
        "vote_share_denominator": "official_national_valid_votes",
        "published_central_prediction": {
            "kind": "published_vote_share_p50",
            "values": {party: quantiles[party]["0.50"] for party in PARTY_ORDER},
        },
        "supplementary_vote_share_mean": published_mean,
        "published_quantiles": quantiles,
        "threshold_probabilities_4pct": {
            party: float(snapshot["threshold_probabilities_4pct"][party])
            for party in PARTY_ORDER
        },
        "seat_quantiles": {
            party: dict(snapshot["seat_summary"][party]) for party in PARTY_ORDER
        },
        "draws": {
            "status": exact_status,
            "verified_predictive_vote_draws": draws_verified,
            "verified_predictive_seat_draws": draws_verified,
            "path": "draws.npz" if draws_verified else None,
            "metadata_path": "draws.json" if draws_verified else None,
        },
    }
    provenance = {
        "generation_id": snapshot.get("generation_id"),
        "generated_at_utc": snapshot.get("generated_at_utc"),
        "model_as_of": snapshot.get("as_of"),
        "latest_poll_input_date": latest_poll_date,
        "latest_poll_input_date_status": latest_poll_status,
        "model_version": snapshot.get("model", {}).get("version"),
        "source_git_commit": snapshot.get("source_git_commit"),
        "source_worktree_clean": snapshot.get("source_worktree_clean"),
        "deterministic_payload_sha256": snapshot.get("deterministic_payload_sha256"),
        "input_hashes": snapshot.get("hashes"),
        "seed": snapshot.get("seed"),
        "sample_count": snapshot.get("samples"),
        "selection": selected.get("provenance"),
        "exact_draw_evidence": {
            key: value for key, value in exact.items()
            if key not in {"draws_bytes_base64", "metadata_bytes_base64", "vote_shares_pct", "seats"}
        },
        "carry_forward": False,
    }
    return ModelCapture(status="AVAILABLE", forecast=forecast, provenance=provenance, files=files)


def _normalize_botten_ada(captured: BottenAdaCapture) -> ModelCapture:
    record = captured.jsonable()
    published = record.get("forecast")
    central: dict[str, float] | None = None
    quantiles: dict[str, dict[str, float]] | None = None
    seat_quantiles: dict[str, Any] | None = None
    if isinstance(published, Mapping):
        election = published.get("election")
        if not isinstance(election, Mapping) or set(election) != set(PARTY_ORDER):
            raise CaptureSourceError("Parsed Botten Ada forecast does not contain the fixed eight-party set")
        central = {party: 100.0 * float(election[party]["votes"]["p50"]) for party in PARTY_ORDER}
        quantiles = {
            party: {
                "0.05": 100.0 * float(election[party]["votes"]["p5"]),
                "0.50": 100.0 * float(election[party]["votes"]["p50"]),
                "0.95": 100.0 * float(election[party]["votes"]["p95"]),
            }
            for party in PARTY_ORDER
        }
        if all("seats" in election[party] for party in PARTY_ORDER):
            seat_quantiles = {party: dict(election[party]["seats"]) for party in PARTY_ORDER}
    threshold_records = record.get("threshold_probabilities_4pct")
    threshold_probabilities = {
        party: float(value["probability"])
        for party, value in (threshold_records.items() if isinstance(threshold_records, Mapping) else [])
        if party in {"L", "C", "KD", "MP"} and isinstance(value, Mapping) and value.get("probability") is not None
    }
    latest_polls = record.get("latest_polls")
    latest_poll_date = latest_polls.get("latest_poll_date") if isinstance(latest_polls, Mapping) else None
    source_metadata = published.get("metadata", {}) if isinstance(published, Mapping) else {}
    source_updated = source_metadata.get("run_written") if isinstance(source_metadata, Mapping) else None
    forecast = {
        "schema_version": "1.0",
        "system": "botten_ada",
        "available": published is not None,
        "election_date": record.get("election_date"),
        "party_order": list(PARTY_ORDER),
        "vote_share_unit": "percentage_points",
        "vote_share_denominator": "official_national_valid_votes",
        "published_central_prediction": None if central is None else {
            "kind": "published_p50",
            "values": central,
        },
        "published_quantiles": quantiles,
        "threshold_probabilities_4pct": threshold_probabilities,
        "seat_quantiles": seat_quantiles,
        "draws": {
            "status": "PARITY_UNVERIFIED",
            "verified_predictive_vote_draws": False,
            "verified_predictive_seat_draws": False,
            "path": None,
            "reason": "No official election-day predictive draw matrix passed semantic and public-value parity gates.",
        },
    }
    provenance = {
        **dict(record.get("provenance", {})),
        "source_status": record.get("status"),
        "source_updated_at": source_updated,
        "source_updated_at_timezone": "UNSPECIFIED_BY_SOURCE" if source_updated else None,
        "latest_poll_date": latest_poll_date,
        "parse_errors": record.get("errors", {}),
        "carry_forward": False,
    }
    files = dict(captured.raw_files)
    files["parsed_publication.json"] = canonical_json_bytes(record)
    status = str(record.get("status", "PARSE_FAILED"))
    return ModelCapture(status=status, forecast=forecast, provenance=provenance, files=files)


def run_capture(
    *,
    scheduled_date: str,
    mode: str,
    archive_root: Path | str = DEFAULT_ARCHIVE_ROOT,
    es_archive_root: Path | str = DEFAULT_ES_ARCHIVE,
    repo_root: Path | str = REPOSITORY_ROOT,
    es_collector: Callable[..., Mapping[str, Any]] = collect_latest_certified_generation,
    ada_collector: Callable[..., BottenAdaCapture] = capture_botten_ada,
    _clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Collect both systems and optionally append one immutable scheduled slot.

    The durable command has no timestamp override: retrieval times always come
    from the process clock.  ``_clock`` is intentionally private and exists
    only for deterministic offline tests.
    """

    if mode not in {"dry_run", "capture"}:
        raise ValueError("mode must be dry_run or capture")
    durable = mode == "capture"
    clock = _clock or (lambda: datetime.now(timezone.utc))
    started = clock()
    # This guard happens before any source request, so no real capture contains
    # evidence retrieved before the preregistered cutoff.
    initial_timing = classify_capture_time(scheduled_date, started, durable=durable)
    archive_path = Path(archive_root)
    if durable and _durable_slot_is_indexed(archive_path, str(scheduled_date)):
        raise CaptureCollisionError(
            f"Scheduled benchmark slot is already indexed: {scheduled_date}"
        )
    cutoff = scheduled_cutoff(scheduled_date).astimezone(timezone.utc)
    try:
        selected = es_collector(
            Path(es_archive_root),
            cutoff,
            Path(repo_root),
            include_sidecar_bytes=True,
            reproduce_missing_draws=True,
        )
        election_simulator = _normalize_election_simulator(selected, repo_root=Path(repo_root))
    except (CaptureSourceError, KeyError, TypeError, IndexError, ValueError) as exc:
        election_simulator = _failed_model_capture(
            system="election_simulator",
            status="PARSE_FAILED",
            reason=f"{type(exc).__name__}: {exc}",
        )
    except Exception as exc:  # noqa: BLE001 - capture the failure as immutable evidence.
        election_simulator = _failed_model_capture(
            system="election_simulator",
            status="SOURCE_UNAVAILABLE",
            reason=f"{type(exc).__name__}: {exc}",
        )
    try:
        botten_capture = ada_collector(cutoff)
        botten_ada = _normalize_botten_ada(botten_capture)
    except (CaptureSourceError, KeyError, TypeError, IndexError, ValueError) as exc:
        botten_ada = _failed_model_capture(
            system="botten_ada",
            status="PARSE_FAILED",
            reason=f"{type(exc).__name__}: {exc}",
        )
    except Exception as exc:  # noqa: BLE001 - preserve an outage/transport failure.
        botten_ada = _failed_model_capture(
            system="botten_ada",
            status="SOURCE_UNAVAILABLE",
            reason=f"{type(exc).__name__}: {exc}",
        )
    completed = clock()
    timing = classify_capture_time(scheduled_date, completed, durable=durable)
    summary = {
        "mode": mode,
        "capture_id": capture_id_for_date(scheduled_date),
        "timing": timing.to_dict(),
        "started_timing_status": initial_timing.status,
        "models": {
            "election_simulator": election_simulator.status,
            "botten_ada": botten_ada.status,
        },
        "draws": {
            "election_simulator": election_simulator.forecast.get("draws"),
            "botten_ada": botten_ada.forecast.get("draws"),
        },
        "audit": {
            "election_simulator": {
                "generation_id": election_simulator.provenance.get("generation_id"),
                "generated_at_utc": election_simulator.provenance.get("generated_at_utc"),
                "deterministic_payload_sha256": election_simulator.provenance.get(
                    "deterministic_payload_sha256"
                ),
                "published_central_prediction_kind": (
                    election_simulator.forecast.get("published_central_prediction") or {}
                ).get("kind"),
            },
            "botten_ada": {
                "decision_generation_identity": botten_ada.provenance.get(
                    "decision_generation_identity"
                ),
                "decision_cutoff": botten_ada.provenance.get("decision_cutoff"),
                "source_updated_at": botten_ada.provenance.get("source_updated_at"),
                "latest_poll_date": botten_ada.provenance.get("latest_poll_date"),
            },
        },
    }
    if durable:
        destination, row = append_capture(
            root=archive_path,
            capture_id=summary["capture_id"],
            timing=timing.to_dict(),
            models={
                "election_simulator": election_simulator,
                "botten_ada": botten_ada,
            },
        )
        summary["capture_path"] = str(destination)
        summary["index_entry_sha256"] = row["entry_sha256"]
    else:
        summary["capture_path"] = None
        summary["durable_write"] = False
    return summary
