"""Configuration, constants, and paths for Swedish Riksdag ElectionSimulator v1."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

DEFAULT_ELECTION_DATE: str = "2026-09-13"
DEFAULT_SIMULATION_SAMPLES: int = 100_000
DEFAULT_SIMULATION_SEED: int = 12345
DEFAULT_GEOGRAPHY_BASELINE_YEAR: int = 2022
DEFAULT_MAJORITY_THRESHOLD: int = 175

# Advanced from 1.0.0-rc1 when the preregistered historical evaluation selected the
# regularized joint Gaussian ElectionNoise law (Challenger B) as the production
# default. The ElectionNoise layer changed, which is a model change rather than a
# fix, so the minor version advances and the candidate letter follows the adopted
# challenger. RC status is retained: the repository convention has not declared a
# stable release.
MODEL_VERSION: str = "1.1.0-rc1"
RELEASE_TAG: str = "election-simulator-v1.1-rc1"

# TWO DISTINCT NAMESPACES SHARE THE LETTER "B". Do not merge them.
#
# BENCHMARK_LINEAGE_CANDIDATE is the published artifact's ``candidate`` field: this
# simulator's identity in the botten-ada comparative benchmark, where Candidate A is
# this model and Candidate B would be a rival external model
# (docs/election_simulator_rc1.md). It is unrelated to ElectionNoise and does not
# change when an ElectionNoise challenger is adopted.
#
# ADOPTED_ELECTION_NOISE_CANDIDATE is the challenger selected by the preregistered
# ElectionNoise v2 competition (docs/election_noise_v2_preregistration.md).
BENCHMARK_LINEAGE_CANDIDATE: str = "A"
ADOPTED_ELECTION_NOISE_CANDIDATE: str = "B"

PARLIAMENTARY_PARTIES_8: tuple[str, ...] = (
    "M",
    "L",
    "C",
    "KD",
    "S",
    "V",
    "MP",
    "SD",
)

MODEL_PARTIES_9: tuple[str, ...] = (
    "M",
    "L",
    "C",
    "KD",
    "S",
    "V",
    "MP",
    "SD",
    "REST",
)

DEFAULT_SIMULATIONS_DIR: Path = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "simulations"
)
