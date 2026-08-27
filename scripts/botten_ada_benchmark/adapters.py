"""Strict adapters for frozen Candidate A and externally supplied Botten Ada data."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from datetime import date
from typing import Any, Mapping, Sequence

import numpy as np

from .config import BOTTEN_ADA_SOURCE, PARTY_ORDER


SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class ForecastCase:
    election_date: str
    as_of: str
    horizon_days: int
    vote_draws: np.ndarray
    seat_draws: np.ndarray | None
    actual_vote: np.ndarray | None = None
    actual_seats: np.ndarray | None = None

    @property
    def key(self) -> tuple[str, str, int]:
        return self.election_date, self.as_of, self.horizon_days


@dataclass(frozen=True)
class ForecastBundle:
    candidate: str
    model_name: str
    model_version: str
    party_order: tuple[str, ...]
    source: dict[str, Any]
    cases: tuple[ForecastCase, ...]
    source_file_sha256: str | None = None


def _array(value: Any, *, name: str, ndim: int, dtype: Any = np.float64) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim != ndim or arr.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty {ndim}-D array")
    return arr


def _validate_case(raw: Mapping[str, Any], party_order: tuple[str, ...]) -> ForecastCase:
    for field in ("election_date", "as_of", "horizon_days", "vote_draws"):
        if field not in raw:
            raise ValueError(f"Forecast case missing {field}")
    vote_draws = _array(raw["vote_draws"], name="vote_draws", ndim=2)
    if vote_draws.shape[1] != len(party_order):
        raise ValueError("vote_draws columns must exactly match party_order")
    seat_value = raw.get("seat_draws")
    seat_draws = None if seat_value is None else _array(seat_value, name="seat_draws", ndim=2, dtype=np.int64)
    if seat_draws is not None and (seat_draws.shape[0] != vote_draws.shape[0] or seat_draws.shape[1] != len(party_order)):
        raise ValueError("seat_draws must have the same sample count and party columns as vote_draws")
    actual_vote = None if raw.get("actual_vote") is None else np.asarray(raw["actual_vote"], dtype=np.float64)
    actual_seats = None if raw.get("actual_seats") is None else np.asarray(raw["actual_seats"], dtype=np.int64)
    if actual_vote is not None and actual_vote.shape != (len(party_order),):
        raise ValueError("actual_vote must match party_order")
    if actual_seats is not None and actual_seats.shape != (len(party_order),):
        raise ValueError("actual_seats must match party_order")
    return ForecastCase(str(raw["election_date"]), str(raw["as_of"]), int(raw["horizon_days"]), vote_draws, seat_draws, actual_vote, actual_seats)


def load_bundle(path: Path | str, *, expected_candidate: str | None = None) -> ForecastBundle:
    """Load a standardized forecast bundle; never infer draws from quantiles."""
    p = Path(path)
    with p.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported benchmark bundle schema: {raw.get('schema_version')}")
    candidate = str(raw.get("candidate", ""))
    if expected_candidate and candidate != expected_candidate:
        raise ValueError(f"Expected Candidate {expected_candidate}, got {candidate}")
    party_order = tuple(raw.get("party_order", ()))
    if party_order != PARTY_ORDER:
        raise ValueError(f"party_order must be exactly {list(PARTY_ORDER)}")
    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValueError("Benchmark bundle must contain at least one case")
    cases = tuple(_validate_case(case, party_order) for case in cases_raw)
    keys = [case.key for case in cases]
    if len(keys) != len(set(keys)):
        raise ValueError("Benchmark bundle contains duplicate case identities")
    return ForecastBundle(candidate, str(raw.get("model_name", "")), str(raw.get("model_version", "")), party_order, dict(raw.get("source", {})), cases, _sha256(p))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def bundle_from_simulation_result(result: Any, *, source: Mapping[str, Any], actual_vote: Sequence[float] | None = None, actual_seats: Sequence[int] | None = None) -> ForecastBundle:
    """Adapt a frozen SimulationResult without changing its draws or model."""
    election_date = str(result.manifest["election_date"])
    as_of = str(result.manifest["as_of"])
    case = ForecastCase(
        election_date=election_date,
        as_of=as_of,
        # The horizon is part of the common-information identity.  Derive it
        # from the explicit dates in the immutable simulation manifest rather
        # than emitting a sentinel that would make a valid paired comparison
        # fail closed.
        horizon_days=(date.fromisoformat(election_date) - date.fromisoformat(as_of)).days,
        vote_draws=np.asarray(result.vote_shares_matrix[:, :8], dtype=np.float64),
        seat_draws=np.asarray(result.seats_matrix, dtype=np.int64),
        actual_vote=None if actual_vote is None else np.asarray(actual_vote, dtype=np.float64),
        actual_seats=None if actual_seats is None else np.asarray(actual_seats, dtype=np.int64),
    )
    return ForecastBundle("A", "ElectionSimulator", str(result.manifest.get("model_version", "")), PARTY_ORDER, dict(source), (case,))


def bundle_to_json(bundle: ForecastBundle, *, include_draws: bool = True) -> dict[str, Any]:
    """Serialize a bundle for benchmark exchange; compact archives should omit draws."""
    cases = []
    for case in bundle.cases:
        row: dict[str, Any] = {
            "election_date": case.election_date,
            "as_of": case.as_of,
            "horizon_days": case.horizon_days,
            "actual_vote": None if case.actual_vote is None else case.actual_vote.tolist(),
            "actual_seats": None if case.actual_seats is None else case.actual_seats.tolist(),
        }
        if include_draws:
            row["vote_draws"] = case.vote_draws.tolist()
            row["seat_draws"] = None if case.seat_draws is None else case.seat_draws.tolist()
        cases.append(row)
    return {"schema_version": SCHEMA_VERSION, "candidate": bundle.candidate, "model_name": bundle.model_name, "model_version": bundle.model_version, "party_order": list(bundle.party_order), "source": bundle.source, "cases": cases}


def write_bundle(bundle: ForecastBundle, path: Path | str) -> None:
    """Write a benchmark exchange bundle; this is separate from compact archives."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(bundle_to_json(bundle), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def unavailable_botten_ada_status(reason: str = "No independently exported Botten Ada bundle was supplied") -> dict[str, Any]:
    return {"status": "NOT_RUN", "reason": reason, "source": BOTTEN_ADA_SOURCE, "required_input": "A standardized JSON bundle with independently generated predictive draws and file SHA-256"}
