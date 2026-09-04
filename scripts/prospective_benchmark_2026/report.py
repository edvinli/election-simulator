"""Post-election scoring and deterministic report generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .archive import DEFAULT_ARCHIVE_ROOT, validate_archive
from .results import OfficialResult, PARTY_ORDER, load_official_result
from .scoring import (
    PROBABILISTIC_TIER_FAIR_DRAWS,
    PROBABILISTIC_TIER_POINT_MAE,
    PROBABILISTIC_TIER_WIS,
    score_forecast_pair,
    threshold_brier_from_probability,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "diagnostics" / "prospective_benchmark_2026"
ENERGY_PAIR_SAMPLE_SIZE = 1_000_000
ENERGY_RANDOM_SEED = 20_260_903
THRESHOLD_PARTIES = ("L", "C", "KD", "MP")
SCHEDULED_DATES = tuple(f"2026-09-{day:02d}" for day in range(4, 13))


class ReportError(ValueError):
    """Raised when immutable capture evidence cannot be scored as declared."""


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"Cannot read JSON evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportError(f"Expected a JSON object: {path}")
    return value


def _ordered_values(forecast: Mapping[str, Any], field: str) -> list[float] | None:
    block = forecast.get(field)
    if block is None:
        return None
    if field == "published_central_prediction":
        block = block.get("values") if isinstance(block, Mapping) else None
    if not isinstance(block, Mapping) or not all(party in block for party in PARTY_ORDER):
        return None
    return [float(block[party]) for party in PARTY_ORDER]


def _quantiles(forecast: Mapping[str, Any]) -> Mapping[str, Mapping[str, float]] | None:
    value = forecast.get("published_quantiles")
    if not isinstance(value, Mapping) or set(value) != set(PARTY_ORDER):
        return None
    if not all(isinstance(value[party], Mapping) for party in PARTY_ORDER):
        return None
    return value  # type: ignore[return-value]


def _validate_forecast_contract(forecast: Mapping[str, Any], *, expected_system: str) -> None:
    """Reject an available forecast whose units/order identify another target.

    Archive hashes establish what was captured, not that the captured JSON
    means percentage points for the fixed eight parties.  This semantic gate
    prevents a malformed or malicious source from exploiting the scorer's
    numeric assumptions.  Explicit source-failure records intentionally lack
    this payload and remain valid evidence of missingness.
    """

    if forecast.get("system") != expected_system:
        raise ReportError("Forecast model identity does not match its capture directory")
    if forecast.get("election_date") != "2026-09-13":
        raise ReportError("Available forecast does not identify the 2026 election")
    if forecast.get("party_order") != list(PARTY_ORDER):
        raise ReportError("Available forecast party order is not the frozen benchmark order")
    if forecast.get("vote_share_unit") != "percentage_points":
        raise ReportError("Available forecast vote-share unit is not percentage_points")
    if forecast.get("vote_share_denominator") != "official_national_valid_votes":
        raise ReportError("Available forecast denominator is not official national valid votes")

    central = forecast.get("published_central_prediction")
    if central is not None:
        if not isinstance(central, Mapping) or not isinstance(central.get("values"), Mapping):
            raise ReportError("Available forecast central prediction has an invalid shape")
        if set(central["values"]) != set(PARTY_ORDER):
            raise ReportError("Available forecast central prediction lacks the fixed eight parties")
        try:
            values = np.asarray([central["values"][party] for party in PARTY_ORDER], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ReportError("Available forecast central prediction is not numeric") from exc
        if not np.isfinite(values).all():
            raise ReportError("Available forecast central prediction is not finite")

    quantiles = forecast.get("published_quantiles")
    if quantiles is not None:
        if not isinstance(quantiles, Mapping) or set(quantiles) != set(PARTY_ORDER):
            raise ReportError("Available forecast quantiles lack the fixed eight parties")
        for party in PARTY_ORDER:
            values = quantiles[party]
            if not isinstance(values, Mapping):
                raise ReportError(f"Available forecast quantiles for {party} are malformed")
            try:
                numeric = np.asarray(list(values.values()), dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ReportError(f"Available forecast quantiles for {party} are not numeric") from exc
            if not np.isfinite(numeric).all():
                raise ReportError(f"Available forecast quantiles for {party} are not finite")

    thresholds = forecast.get("threshold_probabilities_4pct")
    if thresholds is not None:
        if not isinstance(thresholds, Mapping):
            raise ReportError("Available forecast threshold probabilities are malformed")
        for party, probability in thresholds.items():
            # A publisher may expose threshold probabilities for more parties
            # than the four preregistered scoring events. Preserve and validate
            # those official values; _threshold_scores deliberately evaluates
            # only THRESHOLD_PARTIES.
            if party not in PARTY_ORDER:
                raise ReportError(f"Available forecast has an unknown-party threshold event: {party}")
            try:
                numeric = float(probability)
            except (TypeError, ValueError) as exc:
                raise ReportError(f"Available forecast threshold probability for {party} is not numeric") from exc
            if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                raise ReportError(f"Available forecast threshold probability for {party} is invalid")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_digest(array: np.ndarray) -> str:
    """Match the exact-draw sidecar's dtype/shape-bound array commitment."""

    values = np.ascontiguousarray(array)
    descriptor = json.dumps(
        {"dtype": values.dtype.str, "shape": [int(item) for item in values.shape]},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(descriptor)
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _is_hex_digest(value: Any, *, lengths: tuple[int, ...]) -> bool:
    return (
        isinstance(value, str)
        and len(value) in lengths
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _safe_draw_path(base: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ReportError(f"Verified draw evidence has no {label}")
    relative = Path(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ReportError(f"Verified draw evidence has an unsafe {label}")
    path = base.joinpath(*relative.parts)
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError as exc:
        raise ReportError(f"Verified draw evidence {label} escapes its model directory") from exc
    return path


def _validate_draw_metadata(
    *,
    metadata: Mapping[str, Any],
    draws: np.ndarray,
    draws_path: Path,
    expected_system: str,
) -> None:
    """Require an independently hashed and semantically identified sidecar.

    The archive manifest protects the metadata file after capture, while this
    function checks that the metadata itself commits to the exact NPZ bytes,
    array shape/order, and an auditable source.  A boolean in forecast.json is
    not evidence by itself.
    """

    if metadata.get("schema_version") != "1.0":
        raise ReportError("Verified draw metadata has an unsupported schema")
    declared_file_hash = metadata.get("draws_file_sha256")
    if not _is_hex_digest(declared_file_hash, lengths=(64,)) or declared_file_hash != _sha256_file(draws_path):
        raise ReportError("Verified draw metadata does not hash-match the NPZ file")
    samples = metadata.get("samples")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
        raise ReportError("Verified draw metadata has an invalid sample count")
    if samples != int(draws.shape[0]):
        raise ReportError("Verified draw metadata sample count disagrees with the NPZ")
    if metadata.get("vote_share_unit") != "percentage_points":
        raise ReportError("Verified draw metadata has an unsupported vote-share unit")

    party_order = metadata.get("vote_party_order", metadata.get("party_order"))
    if not isinstance(party_order, list):
        raise ReportError("Verified draw metadata lacks vote party order")
    if party_order[: len(PARTY_ORDER)] != list(PARTY_ORDER):
        raise ReportError("Verified draw metadata vote party order is not canonical")
    if draws.shape[1] == len(PARTY_ORDER) + 1:
        if party_order != [*PARTY_ORDER, "REST"]:
            raise ReportError("Nine-column verified draws must place REST after the eight parties")
    elif draws.shape[1] == len(PARTY_ORDER):
        if party_order != list(PARTY_ORDER):
            raise ReportError("Eight-column verified draws must use the canonical party order")
    else:
        raise ReportError("Verified vote draws have an invalid column count")

    arrays = metadata.get("arrays")
    if not isinstance(arrays, Mapping):
        raise ReportError("Verified draw metadata lacks array descriptors")
    descriptor = arrays.get("vote_shares_pct")
    if not isinstance(descriptor, Mapping):
        raise ReportError("Verified draw metadata lacks vote-share array descriptor")
    if descriptor.get("dtype") != draws.dtype.str:
        raise ReportError("Verified draw metadata dtype disagrees with the NPZ")
    if descriptor.get("shape") != [int(item) for item in draws.shape]:
        raise ReportError("Verified draw metadata shape disagrees with the NPZ")
    if descriptor.get("sha256") != _array_digest(draws):
        raise ReportError("Verified draw metadata array hash disagrees with the NPZ")

    model = metadata.get("model")
    if not isinstance(model, Mapping):
        raise ReportError("Verified draw metadata lacks model identity")
    model_name = model.get("name")
    expected_names = {
        "election_simulator": {"ElectionSimulator"},
        "botten_ada": {"Botten Ada", "BottenAda"},
    }
    if model_name not in expected_names.get(expected_system, set()):
        raise ReportError("Verified draw metadata model identity disagrees with the capture")

    if expected_system == "election_simulator":
        generation_id = metadata.get("generation_id")
        if not isinstance(generation_id, str) or not generation_id or "/" in generation_id or "\\" in generation_id:
            raise ReportError("ElectionSimulator draw metadata lacks a certified generation identity")
        if not _is_hex_digest(metadata.get("deterministic_payload_sha256"), lengths=(64,)):
            raise ReportError("ElectionSimulator draw metadata lacks a deterministic payload hash")
        source_commit = metadata.get("source_git_commit")
        if not _is_hex_digest(source_commit, lengths=(40, 64)) or source_commit == "unknown_git_commit":
            raise ReportError("ElectionSimulator draw metadata lacks source commit provenance")
        if metadata.get("source_worktree_clean") is not True:
            raise ReportError("ElectionSimulator draw metadata is not tied to a clean source")
    else:
        # A future Ada sidecar may choose either a flat or nested provenance
        # shape; both forms still require all three immutable source facts.
        source = metadata.get("provenance")
        if not isinstance(source, Mapping):
            source = metadata
        source_url = source.get("source_url")
        source_hash = source.get("content_sha256", source.get("source_content_sha256"))
        retrieved = source.get("retrieved_at_utc")
        if not (
            isinstance(source_url, str)
            and source_url.startswith("https://")
            and _is_hex_digest(source_hash, lengths=(64,))
            and isinstance(retrieved, str)
            and retrieved
        ):
            raise ReportError("Botten Ada draw metadata lacks URL/hash/retrieval provenance")


def _verified_draws(
    capture_dir: Path,
    forecast: Mapping[str, Any],
    *,
    expected_system: str | None = None,
) -> tuple[np.ndarray | None, bool]:
    draw_info = forecast.get("draws")
    if (
        not isinstance(draw_info, Mapping)
        or draw_info.get("verified_predictive_vote_draws") is not True
        or draw_info.get("status") not in {"VERIFIED", "REPLAY_VERIFIED"}
    ):
        return None, False
    system = expected_system or forecast.get("system")
    if system not in {"election_simulator", "botten_ada"}:
        raise ReportError("Verified draw evidence has an unknown model identity")
    if expected_system is not None and forecast.get("system") != expected_system:
        raise ReportError("Forecast model identity does not match its capture directory")
    model_dir = capture_dir / str(system)
    relative = draw_info.get("path")
    path = _safe_draw_path(model_dir, relative, label="draw path")
    metadata_path = _safe_draw_path(model_dir, draw_info.get("metadata_path"), label="metadata path")
    if not path.is_file() or path.is_symlink():
        raise ReportError("Verified draw evidence is missing")
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise ReportError("Verified draw metadata is missing")
    metadata = _read_object(metadata_path)
    try:
        loaded = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - normalize malformed immutable evidence
        raise ReportError(f"Cannot load verified draw evidence: {path}") from exc
    try:
        if "vote_shares_pct" not in loaded.files:
            raise ReportError("Verified draw archive lacks vote_shares_pct")
        draws = np.asarray(loaded["vote_shares_pct"], dtype=np.float64)
    finally:
        loaded.close()
    if draws.ndim != 2 or draws.shape[1] not in {8, 9} or draws.shape[0] < 2:
        raise ReportError("Verified vote draws have an invalid shape")
    if not np.isfinite(draws).all():
        raise ReportError("Verified vote draws contain non-finite values")
    _validate_draw_metadata(
        metadata=metadata,
        draws=draws,
        draws_path=path,
        expected_system=str(system),
    )
    return np.ascontiguousarray(draws[:, :8]), True


def _winner(first: float | None, second: float | None) -> str | None:
    if first is None or second is None:
        return None
    if first == second:
        return "tie"
    return "election_simulator" if first < second else "botten_ada"


def _metric_values(scored: Mapping[str, Any]) -> tuple[str | None, float | None, float | None]:
    tier = scored.get("primary_tier")
    election_simulator = scored.get("election_simulator") or {}
    botten_ada = scored.get("botten_ada") or {}
    if tier == PROBABILISTIC_TIER_FAIR_DRAWS:
        return (
            "mean_fair_crps",
            election_simulator.get("fair_crps_mean_8parties"),
            botten_ada.get("fair_crps_mean_8parties"),
        )
    if tier == PROBABILISTIC_TIER_WIS:
        return (
            "mean_wis",
            (election_simulator.get("quantiles") or {}).get("mean_wis"),
            (botten_ada.get("quantiles") or {}).get("mean_wis"),
        )
    if tier == PROBABILISTIC_TIER_POINT_MAE:
        return (
            "point_mae",
            (election_simulator.get("point_forecast") or {}).get("mean_mae"),
            (botten_ada.get("point_forecast") or {}).get("mean_mae"),
        )
    return None, None, None


def _point_values(scored: Mapping[str, Any]) -> tuple[float | None, float | None]:
    return tuple(
        ((scored.get(model) or {}).get("point_forecast") or {}).get("mean_mae")
        for model in ("election_simulator", "botten_ada")
    )  # type: ignore[return-value]


def _threshold_scores(
    forecasts: Mapping[str, Mapping[str, Any]],
    result: OfficialResult,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for party in THRESHOLD_PARTIES:
        outcome = result.vote_shares[party] >= 4.0
        row: dict[str, Any] = {"outcome_share_gte_4pct": outcome}
        for model in ("election_simulator", "botten_ada"):
            probabilities = forecasts[model].get("threshold_probabilities_4pct")
            probability = probabilities.get(party) if isinstance(probabilities, Mapping) else None
            row[model] = {
                "probability": probability,
                "brier": None if probability is None else threshold_brier_from_probability(
                    float(probability), result.vote_shares[party], threshold=4.0
                ),
            }
        row["winner"] = _winner(row["election_simulator"]["brier"], row["botten_ada"]["brier"])
        output[party] = row
    return output


def _score_capture(capture_dir: Path, manifest: Mapping[str, Any], result: OfficialResult) -> dict[str, Any]:
    forecasts: dict[str, dict[str, Any]] = {}
    for model in ("election_simulator", "botten_ada"):
        forecast = _read_object(capture_dir / model / "forecast.json")
        if forecast.get("available") is True:
            _validate_forecast_contract(forecast, expected_system=model)
        elif forecast.get("system") != model:
            raise ReportError("Unavailable forecast model identity does not match its capture directory")
        forecasts[model] = forecast
    actual = [result.vote_shares[party] for party in PARTY_ORDER]
    election_draws, election_verified = _verified_draws(
        capture_dir,
        forecasts["election_simulator"],
        expected_system="election_simulator",
    )
    ada_draws, ada_verified = _verified_draws(
        capture_dir,
        forecasts["botten_ada"],
        expected_system="botten_ada",
    )
    scored = score_forecast_pair(
        actual,
        election_simulator_draws=election_draws,
        botten_ada_draws=ada_draws,
        election_simulator_draws_verified=election_verified,
        botten_ada_draws_verified=ada_verified,
        election_simulator_quantiles=_quantiles(forecasts["election_simulator"]),
        botten_ada_quantiles=_quantiles(forecasts["botten_ada"]),
        election_simulator_central_forecast=_ordered_values(
            forecasts["election_simulator"], "published_central_prediction"
        ),
        botten_ada_central_forecast=_ordered_values(
            forecasts["botten_ada"], "published_central_prediction"
        ),
        party_order=PARTY_ORDER,
        threshold_parties=THRESHOLD_PARTIES,
        energy_pair_sample_size=ENERGY_PAIR_SAMPLE_SIZE,
        election_simulator_energy_seed=ENERGY_RANDOM_SEED,
        botten_ada_energy_seed=ENERGY_RANDOM_SEED,
    )
    metric_name, election_value, ada_value = _metric_values(scored)
    point_election, point_ada = _point_values(scored)
    return {
        "capture_id": manifest.get("capture_id"),
        "scheduled_date": manifest.get("scheduled_date"),
        "timing_status": manifest.get("timing_status"),
        "model_statuses": manifest.get("model_statuses"),
        "score_status": scored.get("status"),
        "primary_tier": scored.get("primary_tier"),
        "decision_metric": metric_name,
        "decision_values": {
            "election_simulator": election_value,
            "botten_ada": ada_value,
        },
        "decision_winner": _winner(election_value, ada_value),
        "probabilistic_winner": (
            _winner(election_value, ada_value)
            if scored.get("primary_tier") in {PROBABILISTIC_TIER_FAIR_DRAWS, PROBABILISTIC_TIER_WIS}
            else None
        ),
        "point_mae": {
            "election_simulator": point_election,
            "botten_ada": point_ada,
            "winner": _winner(point_election, point_ada),
        },
        "threshold_4pct": _threshold_scores(forecasts, result),
        "metrics": scored,
    }


def _campaign_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scorable = [row for row in rows if row["decision_winner"] is not None]
    by_metric: dict[str, Any] = {}
    for metric in sorted({str(row["decision_metric"]) for row in scorable}):
        selected = [row for row in scorable if row["decision_metric"] == metric]
        election_values = [float(row["decision_values"]["election_simulator"]) for row in selected]
        ada_values = [float(row["decision_values"]["botten_ada"]) for row in selected]
        by_metric[metric] = {
            "capture_count": len(selected),
            "dates": [row["scheduled_date"] for row in selected],
            "equal_weight_mean": {
                "election_simulator": float(np.mean(election_values)),
                "botten_ada": float(np.mean(ada_values)),
            },
            "winner": _winner(float(np.mean(election_values)), float(np.mean(ada_values))),
            "dates_won": {
                "election_simulator": sum(row["decision_winner"] == "election_simulator" for row in selected),
                "botten_ada": sum(row["decision_winner"] == "botten_ada" for row in selected),
                "ties": sum(row["decision_winner"] == "tie" for row in selected),
            },
        }
    uniform_metric = next(iter(by_metric)) if len(by_metric) == 1 else None
    return {
        "scheduled_capture_count": 9,
        "timing_eligible_capture_count": len(rows),
        "scorable_capture_count": len(scorable),
        "single_comparable_metric": uniform_metric,
        "winner": by_metric[uniform_metric]["winner"] if uniform_metric else None,
        "mixed_metric_warning": None if uniform_metric else "No single loss is averaged across different fallback tiers.",
        "by_metric": by_metric,
        "per_date": rows,
    }


def build_report(
    *,
    archive_root: Path | str,
    result_manifest: Path | str,
) -> dict[str, Any]:
    root = Path(archive_root)
    archive_validation = validate_archive(root)
    result = load_official_result(result_manifest)
    index = _read_object(root / "index.json")
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for entry in index.get("captures", []):
        capture_id = str(entry["capture_id"])
        capture_dir = root / "captures" / capture_id
        manifest = _read_object(capture_dir / "manifest.json")
        if manifest.get("timing_eligible") is not True:
            excluded.append({
                "scheduled_date": manifest.get("scheduled_date"),
                "capture_id": capture_id,
                "reason": manifest.get("timing_status"),
            })
            continue
        rows.append(_score_capture(capture_dir, manifest, result))
    represented_dates = {
        str(item["scheduled_date"])
        for item in [*rows, *excluded]
        if item.get("scheduled_date") is not None
    }
    excluded.extend(
        {"scheduled_date": day, "capture_id": None, "reason": "MISSING_SCHEDULED_CAPTURE"}
        for day in SCHEDULED_DATES
        if day not in represented_dates
    )
    excluded.sort(key=lambda item: str(item["scheduled_date"]))
    final = next((row for row in rows if row["scheduled_date"] == "2026-09-12"), None)
    report = {
        "schema_version": "1.0",
        "benchmark": "2026 prospective ElectionSimulator vs Botten Ada",
        "protocol_sha256": archive_validation["protocol_sha256"],
        "active_amendments": index.get("amendments", []),
        "party_order": list(PARTY_ORDER),
        "vote_share_unit": "percentage_points",
        "vote_share_denominator": "official national valid votes",
        "eight_party_renormalization": False,
        "official_result": {
            "authority": "Valmyndigheten",
            "certification_status": "FINAL_CERTIFIED",
            "normalized_manifest_sha256": result.manifest_sha256,
            "official_source_url": result.official_source_url,
            "retrieved_at_utc": result.retrieved_at_utc,
            "raw_sha256": result.raw_sha256,
            "valid_national_votes": result.valid_national_votes,
            "vote_shares": result.vote_shares,
            "seats": result.seats,
        },
        "final_forecast": final,
        "final_probabilistic_winner": None if final is None else final["probabilistic_winner"],
        "final_point_winner": None if final is None else final["point_mae"]["winner"],
        "campaign": _campaign_summary(rows),
        "excluded_captures": excluded,
        "limitations": [
            "The prospective contest compares public wall-clock forecasts, not identical information sets.",
            "No probabilistic winner is declared from point forecasts alone.",
            "Botten Ada draws enter fair scoring only after semantic and public-value parity verification.",
            "Fair Energy Score is secondary and uses the fixed Monte Carlo pair sample in amendment 001; mean fair CRPS is exact.",
            "Small score differences should not be interpreted as universal model superiority.",
        ],
    }
    return report


def _format_number(value: Any) -> str:
    return "unavailable" if value is None else f"{float(value):.4f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    final = report.get("final_forecast")
    lines = [
        "# 2026 prospective forecast benchmark",
        "",
        "This report compares what ElectionSimulator and Botten Ada actually published at the same pre-registered wall-clock cutoffs. It is not an identical-information-set model rerun.",
        "",
        "## Final scheduled forecast (2026-09-12)",
        "",
    ]
    if not isinstance(final, Mapping):
        lines.append("No timing-eligible final scheduled capture is available; no final winner can be declared.")
    else:
        values = final["decision_values"]
        lines.extend([
            f"Scoring tier: `{final['primary_tier']}`. Decision metric: `{final['decision_metric']}`.",
            "",
            f"ElectionSimulator: {_format_number(values['election_simulator'])}; Botten Ada: {_format_number(values['botten_ada'])}. Probabilistic winner: `{final['probabilistic_winner'] or 'not declared'}`.",
            "",
            f"Point MAE — ElectionSimulator: {_format_number(final['point_mae']['election_simulator'])}; Botten Ada: {_format_number(final['point_mae']['botten_ada'])}. Point winner: `{final['point_mae']['winner'] or 'not available'}`.",
            "",
            "### Pre-registered 4% events",
            "",
            "| Party | ES probability | Ada probability | Outcome >=4% | ES Brier | Ada Brier |",
            "|---|---:|---:|:---:|---:|---:|",
        ])
        for party in THRESHOLD_PARTIES:
            row = final["threshold_4pct"][party]
            lines.append(
                f"| {party} | {_format_number(row['election_simulator']['probability'])} | {_format_number(row['botten_ada']['probability'])} | {'yes' if row['outcome_share_gte_4pct'] else 'no'} | {_format_number(row['election_simulator']['brier'])} | {_format_number(row['botten_ada']['brier'])} |"
            )
    campaign = report["campaign"]
    lines.extend([
        "",
        "## Campaign",
        "",
        f"Timing-eligible captures: {campaign['timing_eligible_capture_count']} of 9; scorable captures: {campaign['scorable_capture_count']}.",
        "",
    ])
    if campaign["single_comparable_metric"]:
        aggregate = campaign["by_metric"][campaign["single_comparable_metric"]]
        lines.append(
            f"Equal-weight `{campaign['single_comparable_metric']}` means: ElectionSimulator {_format_number(aggregate['equal_weight_mean']['election_simulator'])}, Botten Ada {_format_number(aggregate['equal_weight_mean']['botten_ada'])}; winner `{aggregate['winner']}`. Dates won: ElectionSimulator {aggregate['dates_won']['election_simulator']}, Botten Ada {aggregate['dates_won']['botten_ada']}, ties {aggregate['dates_won']['ties']}."
        )
    else:
        lines.append("No single campaign winner is declared because different fallback metrics cannot be averaged into one loss.")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def write_report(
    *,
    archive_root: Path | str = DEFAULT_ARCHIVE_ROOT,
    result_manifest: Path | str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path, dict[str, Any]]:
    report = build_report(archive_root=archive_root, result_manifest=result_manifest)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "final_report.json"
    markdown_path = destination / "final_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path, report
