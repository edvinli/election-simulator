"""Configuration, constants, and paths for Swedish Riksdag ElectionSimulator v1."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

# The election the production forecast targets. The 2026 election is decided;
# the cycle moved to the next ordinary election (docs/election_cycle_2030.md).
DEFAULT_ELECTION_DATE: str = "2030-09-08"
DEFAULT_SIMULATION_SAMPLES: int = 100_000
DEFAULT_SIMULATION_SEED: int = 12345
# The baseline for the default target only. Callers resolve the baseline from
# their own target through scripts.simulator.election_cycles, and the engine
# refuses a baseline that does not precede its target.
DEFAULT_GEOGRAPHY_BASELINE_YEAR: int = 2026
DEFAULT_MAJORITY_THRESHOLD: int = 175

# Advanced from 1.0.0-rc1 when the preregistered historical evaluation selected the
# regularized joint Gaussian ElectionNoise law (Challenger B) as the production
# default. The ElectionNoise layer changed, which is a model change rather than a
# fix, so the minor version advances and the candidate letter follows the adopted
# challenger. RC status is retained: the repository convention has not declared a
# stable release.
#
# Advanced from 1.1.0-rc1 when the opinion inputs moved from pollofpolls.se to
# SwedishPolls only (SwedishPollsAggregate-v0.2 and its cleaned polls). The
# ElectionNoise law is unchanged. This is a model-input change, so the minor
# version advances. It was released under an explicit exception: replay v4
# failed only G5 calibration (docs/poll_aggregate_replay_protocol_v4_amendment_001.md).
#
# Advanced from 1.2.0-rc1 when ElectionNoise B was refitted on the corrected
# election-noise residuals of residual amendments 001 and 002: the certified
# result minus the election-eve poll consensus, with no election-day poll and no
# poll silently dropped (docs/election_simulator_model_1_2_1.md). The law, its
# K = 6 pool of 2002-2022 and every other layer are unchanged. A correction of
# the training inputs is a fix, so the patch version advances.
#
# 1.2.1-rc1 applies to forecasts that target the 2030 election or later. A
# forecast of an earlier election is still produced under 1.2.0-rc1, bit for
# bit: model_version_for() routes the version and residual_consensus_rule_for()
# the pool, both by target year.
MODEL_VERSION: str = "1.2.1-rc1"
RELEASE_TAG: str = "election-simulator-v1.2.1-rc1"
PREVIOUS_MODEL_VERSION: str = "1.2.0-rc1"
PREVIOUS_RELEASE_TAG: str = "election-simulator-v1.2-rc1"

#: The first target election forecast under MODEL_VERSION.
CORRECTED_RESIDUALS_FIRST_TARGET_YEAR: int = 2030
#: The residual consensus a target's election-noise pool is fitted on. The
#: corrected rule is ``scripts/election_residuals/eve_consensus.py::RULE_ID``.
UNAMENDED_RESIDUAL_CONSENSUS_RULE: str = "unamended-election-polling-consensus"
CORRECTED_RESIDUAL_CONSENSUS_RULE: str = "election-eve-consensus-amendments-001-002"


def _target_year(election_year: int) -> int:
    if not isinstance(election_year, int) or isinstance(election_year, bool):
        raise TypeError(f"target election year must be an int: {election_year!r}")
    return election_year


def model_version_for(election_year: int) -> str:
    """The model version a forecast of ``election_year`` is produced under."""
    if _target_year(election_year) >= CORRECTED_RESIDUALS_FIRST_TARGET_YEAR:
        return MODEL_VERSION
    return PREVIOUS_MODEL_VERSION


def release_tag_for(election_year: int) -> str:
    """The release tag matching ``model_version_for(election_year)``."""
    if _target_year(election_year) >= CORRECTED_RESIDUALS_FIRST_TARGET_YEAR:
        return RELEASE_TAG
    return PREVIOUS_RELEASE_TAG


def residual_consensus_rule_for(election_year: int) -> str:
    """The residual consensus the election-noise pool of ``election_year`` uses."""
    if _target_year(election_year) >= CORRECTED_RESIDUALS_FIRST_TARGET_YEAR:
        return CORRECTED_RESIDUAL_CONSENSUS_RULE
    return UNAMENDED_RESIDUAL_CONSENSUS_RULE

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
