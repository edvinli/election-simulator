"""The election-noise training pool a production forecast uses, by target election.

Model 1.2.1-rc1 fits ElectionNoise B on the corrected residuals of
election-noise residual amendments 001 and 002
(``docs/election_noise_residual_amendment_001.md`` and ``..._002.md``): the
certified result minus the election-eve poll consensus of
``scripts/election_residuals/eve_consensus.py``. It applies to forecasts that
target the 2030 election or later.

A forecast of an earlier election keeps the unamended pool of
``scripts/election_layer_v2/residuals_pool.py``, which stays byte-identical, so
every 2026-target forecast, archived generation, benchmark replay and hindcast
reproduces bit for bit. The routing rule itself lives in
``scripts/simulator/config.py`` (``residual_consensus_rule_for``), a hashed
model file.

The corrected pool differs from the unamended one only in the consensus
builder. ``_load_pp_residuals`` is ``load_chronological_pp_residuals`` with the
builder as a parameter; ``tests/test_production_residuals_pool.py`` asserts
that, given the unamended builder, it reproduces that function exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from scripts.election_residuals.config import (
    ALL_CATEGORIES,
    DEFAULT_ELECTIONS_FILE,
    DEFAULT_POLLS_FILE,
)
from scripts.election_residuals.consensus import (
    ElectionPollConsensus,
    build_election_polling_consensus,
)
from scripts.election_residuals.eve_consensus import RULE_ID as EVE_RULE_ID
from scripts.election_residuals.eve_consensus import build_eve_polling_consensus
from scripts.elections.load import load_election_targets_for_forecasting

from .config import ALL_HISTORICAL_ELECTIONS, CANONICAL_WINDOW_DAYS
from .residuals_pool import ChronologicalPPResidualsPool, load_chronological_pp_residuals

ConsensusBuilder = Callable[..., ElectionPollConsensus]


def _load_pp_residuals(
    builder: ConsensusBuilder,
    target_election_year: int,
    window_days: int = CANONICAL_WINDOW_DAYS,
    polls_file: Path | str | None = None,
    elections_file: Path | str | None = None,
) -> ChronologicalPPResidualsPool:
    """``load_chronological_pp_residuals`` with the consensus builder as a parameter.

    Every other step -- the strictly-prior election set, the zero-sum checks
    and their clean-ups, the mean bias and the centring -- is the same code in
    the same order, so a builder swap is the only possible difference.
    """
    p_file = Path(polls_file) if polls_file else DEFAULT_POLLS_FILE
    e_file = Path(elections_file) if elections_file else DEFAULT_ELECTIONS_FILE

    polls_df = pd.read_csv(p_file)
    election_targets = load_election_targets_for_forecasting(e_file)

    eligible_elections = [
        el for el in sorted(ALL_HISTORICAL_ELECTIONS) if el.year < target_election_year
    ]

    if not eligible_elections:
        raise ValueError(f"No historical elections available prior to {target_election_year}")

    residuals_list: list[np.ndarray] = []
    training_years: list[int] = []

    for el_date in eligible_elections:
        target_comp = election_targets[el_date]
        consensus = builder(el_date, polls_df, window_days=window_days)

        t_vec = np.array([target_comp[c] for c in ALL_CATEGORIES], dtype=float)
        c_vec = np.array([consensus.consensus_composition[c] for c in ALL_CATEGORIES], dtype=float)
        r_vec = t_vec - c_vec

        res_sum = float(np.sum(r_vec))
        if abs(res_sum) > 0.05:
            raise ValueError(f"Election {el_date.year} residual sum ({res_sum:.6f}) deviates from zero")

        if abs(res_sum) > 1e-12:
            r_vec = r_vec - (res_sum / len(r_vec))

        residuals_list.append(r_vec)
        training_years.append(el_date.year)

    residuals_mat = np.array(residuals_list, dtype=float)
    mean_bias = np.mean(residuals_mat, axis=0)

    mb_sum = float(np.sum(mean_bias))
    if abs(mb_sum) > 1e-12:
        mean_bias = mean_bias - (mb_sum / len(mean_bias))

    centered_mat = residuals_mat - mean_bias

    return ChronologicalPPResidualsPool(
        target_election_year=target_election_year,
        training_years=tuple(training_years),
        residuals_matrix=residuals_mat,
        mean_bias_pp=mean_bias,
        centered_residuals_matrix=centered_mat,
    )


def load_eve_pp_residuals(
    target_election_year: int,
    window_days: int = CANONICAL_WINDOW_DAYS,
    polls_file: Path | str | None = None,
    elections_file: Path | str | None = None,
) -> ChronologicalPPResidualsPool:
    """The corrected pool: residuals against the amendment 001 + 002 eve consensus."""
    return _load_pp_residuals(
        build_eve_polling_consensus, target_election_year,
        window_days=window_days, polls_file=polls_file, elections_file=elections_file,
    )


def _rule_for(election_year: int) -> str:
    # Imported when called rather than at module load: importing the
    # scripts.simulator package runs its engine import, which reaches this
    # module through the national engine.
    from scripts.simulator.config import (
        CORRECTED_RESIDUAL_CONSENSUS_RULE,
        UNAMENDED_RESIDUAL_CONSENSUS_RULE,
        residual_consensus_rule_for,
    )

    if CORRECTED_RESIDUAL_CONSENSUS_RULE != EVE_RULE_ID:
        raise RuntimeError("config.py names a residual consensus rule eve_consensus.py does not implement")
    rule = residual_consensus_rule_for(election_year)
    if rule not in (CORRECTED_RESIDUAL_CONSENSUS_RULE, UNAMENDED_RESIDUAL_CONSENSUS_RULE):
        raise ValueError(f"unknown residual consensus rule {rule!r}")
    return rule


def consensus_builder_for(election_year: int) -> ConsensusBuilder:
    """The residual consensus builder a forecast of ``election_year`` uses."""
    if _rule_for(election_year) == EVE_RULE_ID:
        return build_eve_polling_consensus
    return build_election_polling_consensus


def load_production_pp_residuals(
    target_election_year: int,
    window_days: int = CANONICAL_WINDOW_DAYS,
    polls_file: Path | str | None = None,
    elections_file: Path | str | None = None,
) -> ChronologicalPPResidualsPool:
    """The training pool for a forecast of ``target_election_year``.

    2030 and later: the corrected pool. Earlier targets: the unamended
    ``load_chronological_pp_residuals``, called exactly as before.
    """
    if _rule_for(target_election_year) == EVE_RULE_ID:
        return load_eve_pp_residuals(
            target_election_year, window_days=window_days,
            polls_file=polls_file, elections_file=elections_file,
        )
    return load_chronological_pp_residuals(
        target_election_year=target_election_year, window_days=window_days,
        polls_file=polls_file, elections_file=elections_file,
    )


__all__ = [
    "consensus_builder_for",
    "load_eve_pp_residuals",
    "load_production_pp_residuals",
]
