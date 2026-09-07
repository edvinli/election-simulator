"""Canonical input identities for the frozen historical simulation path.

Transport poll IDs and acquisition metadata are deliberately absent. The
fingerprint lives beside (not inside) certified points. Legacy artifacts can
be upgraded only from their hash-verified source revision; absence of that
revision is a cache miss, never permission to assume fresh inputs are old ones.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Iterator, Mapping

import pandas as pd

from scripts.election_layer_v2.config import ALL_HISTORICAL_ELECTIONS, CANONICAL_WINDOW_DAYS
from scripts.election_residuals.consensus import build_election_polling_consensus
from scripts.elections.load import load_election_targets_for_forecasting
from scripts.pollofpolls.state import load_individual_polls_dataset, load_timeseries_dataset
from scripts.simulator.config import DEFAULT_GEOGRAPHY_BASELINE_YEAR

VERSION = 1
ROOT = Path(__file__).resolve().parents[2]
# Numerical implementation/configuration, separate from ingestion and chart
# serialization. A numerical code change must not silently reuse old draws.
MODEL_FILES = tuple(
    f"scripts/{name}.py" for name in (
        "simulator/engine", "simulator/config", "simulator/fast_allocator",
        "geography/config", "geography/projection", "geography/integerization", "geography/raking",
        "mandates/config", "mandates/allocator", "mandates/law", "mandates/tie_breaker",
        "vote_share_calibration/national_engine", "vote_share_calibration/election_noise_b",
        "vote_share_calibration/config", "vote_share_calibration/models",
        "election_layer_v2/residuals_pool", "election_layer_v2/transfer", "election_layer_v2/config",
        "election_residuals/consensus", "election_residuals/config", "elections/load",
        "hindcasts/models", "pollofpolls/state", "pollofpolls/state_config",
        "pollofpolls/clr", "pollofpolls/transitions", "pollofpolls/state_math",
        "elections/config", "elections/parse",
    )
)
DATA_FILES = (
    "pollofpolls/individual_polls.csv", "pollofpolls/pollofpolls_timeseries.csv",
    "pollofpolls/swedishpolls_individual_polls.csv", "elections/riksdag_election_results.csv",
    "geography/constituency_party_votes_2014_2022.csv",
    "geography/constituency_electorates_2014_2026.csv",
)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _ordered(values: Any) -> list[Any]:
    # Sorting a list (not a set) retains multiplicity.
    return sorted(values, key=lambda value: json.dumps(value, sort_keys=True))


class EffectiveInputs:
    """Read once, then fingerprint the eligible inputs for each observation date.

    The complete eligible opinion history is retained: the estimator can fall
    back beyond its trailing covariance window. Dynamics likewise uses all
    transitions ending on/before as_of. No future rows or undated polls enter
    those identities. Election-noise windows follow the canonical selector.
    """

    def __init__(self, data_dir: Path = ROOT / "data/processed", *,
                 source_root: Path = ROOT, election_date: date, seed: int):
        self.election = election_date
        self.seed = seed
        self.timeseries = load_timeseries_dataset(data_dir / "pollofpolls/pollofpolls_timeseries.csv")
        self.polls, _ = load_individual_polls_dataset(data_dir / "pollofpolls/individual_polls.csv")
        polls_df = pd.read_csv(data_dir / "pollofpolls/swedishpolls_individual_polls.csv")
        targets = load_election_targets_for_forecasting(data_dir / "elections/riksdag_election_results.csv")
        self.training = []
        for election in ALL_HISTORICAL_ELECTIONS:
            if election.year >= election_date.year:
                continue
            consensus = build_election_polling_consensus(election, polls_df, window_days=CANONICAL_WINDOW_DAYS)
            # The numeric target and selected observations include weights,
            # rather than poll_id (which only transports source identity).
            self.training.append((election, {
                "election": election.isoformat(), "target": targets[election],
                "polls": _ordered({
                    "pollster": p.pollster, "publication_date": p.publication_date.isoformat(),
                    "interview_start": p.interview_start.isoformat() if p.interview_start else None,
                    "interview_end": p.interview_end.isoformat(), "sample_size": p.sample_size,
                    "weight": p.weight, "parties": p.party_support,
                } for p in consensus.contributing_polls),
            }))
        # Geography's numerical baseline and chronological electorate projection
        # consume these tables. Canonical records ignore serialization order.
        geography = {}
        for filename in DATA_FILES[-2:]:
            frame = pd.read_csv(data_dir / filename)
            if "party_votes" in filename:
                frame = frame[frame["election_year"] == DEFAULT_GEOGRAPHY_BASELINE_YEAR]
            geography[filename] = _ordered(json.loads(frame.to_json(orient="records")))
        self.static = digest({
            "model": {f: hashlib.sha256((source_root / f).read_bytes()).hexdigest() for f in MODEL_FILES},
            "geography": geography,
        })

    def fingerprint(self, as_of: date) -> str:
        # The current engine trains election noise strictly before the target
        # election. Refuse to certify a history date predating that training.
        if any(election > as_of for election, _ in self.training):
            raise ValueError("Historical as_of precedes an election-noise training outcome")
        polls = []
        for p in self.polls:
            if p.publication_date is None or p.publication_date > as_of:
                continue
            if p.interview_end is not None and p.interview_end > as_of:
                continue
            if p.reference_date is None or p.reference_date > as_of:
                continue
            polls.append({
                "pollster": p.pollster, "publication_date": p.publication_date.isoformat(),
                "start": p.interview_start.isoformat() if p.interview_start else None,
                "end": p.interview_end.isoformat() if p.interview_end else None,
                "reference": p.reference_date.isoformat(), "sample_size": p.sample_size,
                "composition": p.composition,
            })
        return digest({
            "version": VERSION, "as_of": as_of.isoformat(), "election": self.election.isoformat(),
            "seed": self.seed, "static": self.static, "opinion_polls": _ordered(polls),
            "timeseries": [{"date": r["date"].isoformat(), "composition": r["composition"]}
                           for r in self.timeseries if r["date"] <= as_of],
            "training": [value for _, value in self.training],
        })


@contextmanager
def legacy_inputs(payload: Mapping[str, Any], *, repo: Path = ROOT) -> Iterator[EffectiveInputs | None]:
    """Resolve legacy provenance locally, with no fetch or fallback to HEAD."""
    commit = str(payload.get("model_commit", ""))
    if payload.get("source_worktree_clean") is not True or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
        yield None
        return
    with tempfile.TemporaryDirectory(prefix="history-inputs-") as tmp:
        root = Path(tmp)
        try:
            for relative in (*MODEL_FILES, *("data/processed/" + f for f in DATA_FILES)):
                content = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=repo,
                                         check=True, capture_output=True).stdout
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            for name, field in ((DATA_FILES[2], "poll_source_sha256"),
                                (DATA_FILES[1], "timeseries_source_sha256")):
                expected = (payload.get("source_hashes") or {}).get(field, payload.get(field))
                if hashlib.sha256((root / "data/processed" / name).read_bytes()).hexdigest() != expected:
                    raise ValueError("Legacy source revision does not match recorded input hashes")
            inputs = EffectiveInputs(root / "data/processed", source_root=root,
                                     election_date=date.fromisoformat(payload["election_date"]),
                                     seed=int(payload["model"]["seed"]))
        except (subprocess.CalledProcessError, ValueError, KeyError, OSError):
            inputs = None
        yield inputs


def previous_fingerprints(payload: Mapping[str, Any]) -> dict[str, str]:
    metadata = payload.get("reconstruction_inputs")
    if metadata is not None:
        if metadata.get("version") != VERSION:
            return {}
        return dict(metadata["dates"])
    with legacy_inputs(payload) as previous:
        if previous is None:
            return {}
        return {point["date"]: previous.fingerprint(date.fromisoformat(point["date"]))
                for point in payload["series"] if point["provenance"] == "reconstructed_current_model"}
