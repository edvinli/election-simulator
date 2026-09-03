"""Rolling, leakage-safe retrospective evaluation of forward opinion paths.

Two distinct questions are answered, and the distinction matters:

1. **Did the election-day endpoint model change?**  It must not.  The
   ``campaign_paths`` endpoint draws are compared against the frozen
   ``dynamics_v2`` production sample at the same origin, horizon and seed.
   The evaluation records the maximum absolute CRPS difference, which is
   exactly zero when the endpoint is bitwise identical.  This is the evidence
   that the new visualization did not silently adopt a different endpoint
   model.

2. **Are the newly published intermediate days calibrated?**  Only the path
   model publishes an opinion distribution for ``t + d`` with ``0 < d < n``,
   so that part *is* new and needs its own out-of-sample evidence.  It is
   scored against the realized Poll-of-Polls trajectory, the same development
   target used throughout ``docs/opinion_dynamics.md``, and compared with
   three reference models:

   ``frozen_state``
       what the incumbent shrinking-horizon view implies about opinion:
       ``theta[t + d] = PoP_t`` for every intermediate day (a point mass).
   ``endpoint_fan``
       the naive alternative of reusing the *election-day* dynamics spread at
       every intermediate day, i.e. a constant-width fan.
   ``independent_walk``
       the explicitly rejected alternative: ``d`` independently signed
       one-day CLR steps accumulated into a random walk.

State uncertainty is deliberately excluded: every model is conditioned on
``theta_t = PoP_t``.  ``OpinionState`` is common to all four models, so
including it would add identical variance everywhere and dilute the
comparison of the dynamics layer, which is the only thing that differs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from scripts.hindcasts.models import (
    derive_shared_dynamics_seed,
    sample_shared_symmetric_dynamics,
)
from scripts.pollofpolls.clr import clr_to_composition_matrix, composition_to_clr
from scripts.pollofpolls.state import load_timeseries_dataset
from scripts.pollofpolls.state_config import ALL_CATEGORIES
from scripts.pollofpolls.transitions import (
    MIN_TRANSITIONS,
    build_all_historical_transitions,
    filter_transitions_as_of,
)
from scripts.simulator.config import PARLIAMENTARY_PARTIES_8
from scripts.vote_share_calibration.energy_score import compute_energy_score


MODEL_IDS: tuple[str, ...] = (
    "campaign_paths",
    "frozen_state",
    "endpoint_fan",
    "independent_walk",
)

DEFAULT_PATH_DAYS = 28
DEFAULT_ORIGIN_STRIDE_DAYS = 14
DEFAULT_SAMPLES = 2_000
DEFAULT_BASE_SEED = 12345
DEFAULT_START = date(2019, 1, 1)
#: Energy scores are multivariate and quadratic in the draw count, so they are
#: computed on a fixed subsample at a few checkpoint horizons only.
ENERGY_SUBSAMPLE = 300

_PARTY_INDEX = {party: ALL_CATEGORIES.index(party) for party in PARLIAMENTARY_PARTIES_8}
_QUANTILES = (0.05, 0.25, 0.75, 0.95)


def crps_matrix(samples: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Exact empirical CRPS over the last axis of ``samples``.

    ``samples`` has shape ``(..., N)`` and ``actual`` shape ``(...)``.  The
    estimator is the one used across this repository:

        CRPS = mean|X - y| - (1 / N^2) * sum_i (2i + 1 - N) * x_(i)

    ``tests/test_campaign_path_eval.py`` locks it against the scalar
    ``scripts.pollofpolls.backtest_metrics.calculate_crps``.
    """

    draws = np.asarray(samples, dtype=np.float64)
    target = np.asarray(actual, dtype=np.float64)
    count = draws.shape[-1]
    if count < 1:
        raise ValueError("cannot compute CRPS without draws")
    term1 = np.mean(np.abs(draws - target[..., np.newaxis]), axis=-1)
    if count == 1:
        return term1
    ordered = np.sort(draws, axis=-1)
    positions = np.arange(count, dtype=np.float64)
    weights = (2.0 * positions + 1.0 - count) / (count * count)
    return term1 - ordered @ weights


@dataclass
class _Accumulator:
    """Running sums for one ``(model, horizon, party)`` cell."""

    cases: int = 0
    crps: float = 0.0
    covered_50: int = 0
    covered_90: int = 0
    width_50: float = 0.0
    width_90: float = 0.0


@dataclass
class CampaignPathEvaluation:
    """Aggregated retrospective evaluation output."""

    path_days: int
    samples: int
    base_seed: int
    origins: tuple[date, ...]
    by_horizon: list[dict[str, Any]]
    energy: list[dict[str, Any]]
    endpoint_parity: list[dict[str, Any]]
    summary: dict[str, Any] = field(default_factory=dict)


def _clr_series(timeseries: Sequence[dict[str, Any]]) -> tuple[list[date], np.ndarray, dict[date, int]]:
    dates = [row["date"] for row in timeseries]
    matrix = np.empty((len(dates), len(ALL_CATEGORIES)), dtype=np.float64)
    for index, row in enumerate(timeseries):
        matrix[index, :], _ = composition_to_clr(row["composition"])
    return dates, matrix, {value: index for index, value in enumerate(dates)}


def _trajectory_tensor(
    clr_matrix: np.ndarray,
    start_positions: np.ndarray,
    path_days: int,
) -> np.ndarray:
    """Vectorized ``CLR(PoP[s + d]) - CLR(PoP[s])`` for ``d = 1 .. path_days``.

    Equivalent to ``build_campaign_path_pool``'s tensor for a gap-free series;
    the equivalence is asserted once per run in ``evaluate_campaign_paths``.
    """

    base = clr_matrix[start_positions]  # (M, 9)
    offsets = np.arange(1, path_days + 1)
    forward = clr_matrix[start_positions[:, None] + offsets[None, :]]  # (M, n, 9)
    return forward - base[:, None, :]


def _eligible_start_positions(
    dates: Sequence[date],
    origin: date,
    horizon: int,
) -> np.ndarray:
    """Positions ``p`` with ``dates[p] + horizon`` present and ``<= origin``.

    On the gap-free production series this is exactly production's eligible
    Dynamics v2 pool at ``horizon``, in the same order.
    """

    index = {value: position for position, value in enumerate(dates)}
    positions = []
    for position, value in enumerate(dates):
        end = value + timedelta(days=horizon)
        end_position = index.get(end)
        if end_position is None or end > origin:
            continue
        if end_position != position + horizon:
            # A gap between ``s`` and ``s + horizon`` would break the
            # index alignment the endpoint-parity argument depends on.
            continue
        positions.append(position)
    return np.asarray(positions, dtype=np.int64)


def build_origins(
    dates: Sequence[date],
    *,
    path_days: int,
    stride_days: int,
    start: date,
) -> tuple[date, ...]:
    """Rolling origins whose full realized trajectory exists in the series."""

    available = set(dates)
    last = max(dates)
    origins: list[date] = []
    cursor = start
    while cursor + timedelta(days=path_days) <= last:
        if cursor in available:
            origins.append(cursor)
        cursor += timedelta(days=stride_days)
    return tuple(origins)


def _model_clr_paths(
    model: str,
    *,
    base_clr: np.ndarray,
    trajectory: np.ndarray,
    one_day: np.ndarray,
    indices: np.ndarray,
    signs: np.ndarray,
    path_days: int,
    samples: int,
    walk_seed: int,
) -> np.ndarray:
    """Predictive CLR draws of shape ``(path_days, samples, 9)``."""

    if model == "campaign_paths":
        return base_clr[None, None, :] + (signs[None, :, :] * trajectory[indices].transpose(1, 0, 2))
    if model == "frozen_state":
        return np.broadcast_to(
            base_clr[None, None, :], (path_days, samples, base_clr.shape[0])
        ).copy()
    if model == "endpoint_fan":
        endpoint = signs * trajectory[indices][:, -1, :]
        return base_clr[None, None, :] + np.broadcast_to(
            endpoint[None, :, :], (path_days, samples, base_clr.shape[0])
        ).copy()
    if model == "independent_walk":
        generator = np.random.default_rng(walk_seed)
        step_indices = generator.integers(0, one_day.shape[0], size=(samples, path_days))
        step_signs = generator.choice([-1.0, 1.0], size=(samples, path_days, 1))
        steps = step_signs * one_day[step_indices]
        return base_clr[None, None, :] + np.cumsum(steps, axis=1).transpose(1, 0, 2)
    raise ValueError(f"unknown evaluation model {model!r}")


def evaluate_campaign_paths(
    *,
    timeseries_file: Path | str,
    path_days: int = DEFAULT_PATH_DAYS,
    stride_days: int = DEFAULT_ORIGIN_STRIDE_DAYS,
    samples: int = DEFAULT_SAMPLES,
    base_seed: int = DEFAULT_BASE_SEED,
    start: date = DEFAULT_START,
    energy_checkpoints: Iterable[int] | None = None,
    verify_pool_against_production: bool = True,
) -> CampaignPathEvaluation:
    """Score the path model and its reference models over rolling origins."""

    if path_days < 2:
        raise ValueError("path_days must be at least two days to have an interior")
    timeseries = load_timeseries_dataset(Path(timeseries_file))
    dates, clr_matrix, position_of = _clr_series(timeseries)
    origins = build_origins(dates, path_days=path_days, stride_days=stride_days, start=start)
    if not origins:
        raise ValueError("no rolling origin has a complete realized trajectory")

    checkpoints = tuple(
        sorted({value for value in (energy_checkpoints or (1, 7, 14, path_days)) if 1 <= value <= path_days})
    )
    endpoint_horizon = min(path_days, 112)

    cells: dict[tuple[str, int, str], _Accumulator] = {}
    energy_cells: dict[tuple[str, int], list[float]] = {}
    parity_rows: list[dict[str, Any]] = []
    used_origins: list[date] = []
    verified_pool = not verify_pool_against_production

    for origin in origins:
        start_positions = _eligible_start_positions(dates, origin, endpoint_horizon)
        if start_positions.size < MIN_TRANSITIONS:
            continue
        one_day_positions = _eligible_start_positions(dates, origin, 1)
        if one_day_positions.size < MIN_TRANSITIONS:
            continue

        trajectory = _trajectory_tensor(clr_matrix, start_positions, path_days)
        one_day = _trajectory_tensor(clr_matrix, one_day_positions, 1)[:, 0, :]

        if not verified_pool:
            from scripts.forecast_history.campaign_paths import build_campaign_path_pool

            reference = build_campaign_path_pool(timeseries, origin, path_days)
            if reference.start_dates != tuple(dates[position] for position in start_positions):
                raise ValueError("evaluation pool disagrees with the production campaign-path pool")
            np.testing.assert_array_equal(reference.delta_tensor, trajectory)
            verified_pool = True

        base_clr = clr_matrix[position_of[origin]]
        dynamics_seed = derive_shared_dynamics_seed(
            base_seed=base_seed, origin_date=origin, horizon_days=path_days
        )
        generator = np.random.default_rng(dynamics_seed)
        indices = generator.integers(0, start_positions.size, size=samples)
        signs = generator.choice([-1.0, 1.0], size=(samples, 1))
        walk_seed = derive_shared_dynamics_seed(
            base_seed=base_seed + 1, origin_date=origin, horizon_days=path_days
        )

        realized = np.empty((path_days, len(ALL_CATEGORIES)), dtype=np.float64)
        for day in range(1, path_days + 1):
            row = timeseries[position_of[origin + timedelta(days=day)]]
            realized[day - 1, :] = [row["composition"][category] for category in ALL_CATEGORIES]

        endpoint_draws: dict[str, np.ndarray] = {}
        for model in MODEL_IDS:
            clr_paths = _model_clr_paths(
                model,
                base_clr=base_clr,
                trajectory=trajectory,
                one_day=one_day,
                indices=indices,
                signs=signs,
                path_days=path_days,
                samples=samples,
                walk_seed=walk_seed,
            )
            shares = clr_to_composition_matrix(clr_paths.reshape(-1, len(ALL_CATEGORIES))).reshape(
                path_days, samples, len(ALL_CATEGORIES)
            )
            endpoint_draws[model] = shares[-1]

            for party, column in _PARTY_INDEX.items():
                draws = shares[:, :, column]  # (path_days, samples)
                truth = realized[:, column]
                scores = crps_matrix(draws, truth)
                quantiles = np.quantile(draws, _QUANTILES, axis=-1)
                low50, low90 = quantiles[1], quantiles[0]
                high50, high90 = quantiles[2], quantiles[3]
                inside50 = (truth >= low50) & (truth <= high50)
                inside90 = (truth >= low90) & (truth <= high90)
                for day in range(path_days):
                    cell = cells.setdefault((model, day + 1, party), _Accumulator())
                    cell.cases += 1
                    cell.crps += float(scores[day])
                    cell.covered_50 += int(inside50[day])
                    cell.covered_90 += int(inside90[day])
                    cell.width_50 += float(high50[day] - low50[day])
                    cell.width_90 += float(high90[day] - low90[day])

            for checkpoint in checkpoints:
                subsample = shares[checkpoint - 1][:ENERGY_SUBSAMPLE]
                energy_cells.setdefault((model, checkpoint), []).append(
                    compute_energy_score(subsample, realized[checkpoint - 1])
                )

        # ---- endpoint parity against the frozen production sample ----------
        production_pool = filter_transitions_as_of(
            build_all_historical_transitions(timeseries, horizons=[endpoint_horizon])[endpoint_horizon],
            origin,
        )
        production_deltas = sample_shared_symmetric_dynamics(
            eligible_transitions=production_pool,
            samples_count=samples,
            seed=dynamics_seed,
        )
        production_shares = clr_to_composition_matrix(base_clr[None, :] + production_deltas)
        path_shares = endpoint_draws["campaign_paths"]
        bitwise = bool(np.array_equal(path_shares, production_shares))
        for party, column in _PARTY_INDEX.items():
            truth = float(realized[-1, column])
            parity_rows.append(
                {
                    "origin": origin.isoformat(),
                    "party": party,
                    "bitwise_identical": bitwise,
                    "crps_campaign_paths": float(crps_matrix(path_shares[:, column], np.float64(truth))),
                    "crps_dynamics_v2": float(
                        crps_matrix(production_shares[:, column], np.float64(truth))
                    ),
                }
            )
        used_origins.append(origin)

    if not used_origins:
        raise ValueError("no origin produced a sufficient leakage-safe transition pool")

    by_horizon = [
        {
            "model": model,
            "horizon_days": horizon,
            "party": party,
            "cases": cell.cases,
            "mean_crps": round(cell.crps / cell.cases, 6),
            "coverage_50": round(cell.covered_50 / cell.cases, 6),
            "coverage_90": round(cell.covered_90 / cell.cases, 6),
            "mean_width_50": round(cell.width_50 / cell.cases, 6),
            "mean_width_90": round(cell.width_90 / cell.cases, 6),
        }
        for (model, horizon, party), cell in sorted(cells.items())
    ]
    energy = [
        {
            "model": model,
            "horizon_days": horizon,
            "cases": len(values),
            "mean_energy_score": round(float(np.mean(values)), 6),
        }
        for (model, horizon), values in sorted(energy_cells.items())
    ]

    parity_difference = max(
        abs(row["crps_campaign_paths"] - row["crps_dynamics_v2"]) for row in parity_rows
    )
    summary = {
        "path_days": path_days,
        "stride_days": stride_days,
        "samples": samples,
        "base_seed": base_seed,
        "origins": len(used_origins),
        "first_origin": used_origins[0].isoformat(),
        "last_origin": used_origins[-1].isoformat(),
        "endpoint_horizon_days": endpoint_horizon,
        "endpoint_bitwise_identical_all_origins": all(row["bitwise_identical"] for row in parity_rows),
        "endpoint_max_abs_crps_difference": parity_difference,
        "energy_checkpoints": list(checkpoints),
        "energy_subsample": ENERGY_SUBSAMPLE,
        "target": "realized_poll_of_polls_trajectory",
        "state_uncertainty": "excluded_common_to_all_models",
        "models": list(MODEL_IDS),
    }
    for model in MODEL_IDS:
        rows = [row for row in by_horizon if row["model"] == model]
        interior = [row for row in rows if row["horizon_days"] < path_days]
        summary[f"{model}_mean_crps_all_horizons"] = round(
            float(np.mean([row["mean_crps"] for row in rows])), 6
        )
        summary[f"{model}_mean_crps_interior_days"] = round(
            float(np.mean([row["mean_crps"] for row in interior])), 6
        )
        summary[f"{model}_coverage_50_all_horizons"] = round(
            float(np.mean([row["coverage_50"] for row in rows])), 6
        )
        summary[f"{model}_coverage_90_all_horizons"] = round(
            float(np.mean([row["coverage_90"] for row in rows])), 6
        )

    return CampaignPathEvaluation(
        path_days=path_days,
        samples=samples,
        base_seed=base_seed,
        origins=tuple(used_origins),
        by_horizon=by_horizon,
        energy=energy,
        endpoint_parity=parity_rows,
        summary=summary,
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"refusing to write an empty artifact to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row[column]) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_evaluation_artifacts(
    evaluation: CampaignPathEvaluation,
    *,
    backtest_dir: Path | str,
    diagnostics_dir: Path | str,
) -> dict[str, str]:
    """Write the by-horizon, energy, parity and summary artifacts."""

    backtests = Path(backtest_dir)
    diagnostics = Path(diagnostics_dir)
    tag = (
        f"n{evaluation.path_days}_"
        f"{evaluation.origins[0].isoformat()}_{evaluation.origins[-1].isoformat()}"
    )
    written = {
        "by_horizon": str(
            _write_csv(backtests / f"campaign_paths_by_horizon_{tag}.csv", evaluation.by_horizon)
        ),
        "energy": str(
            _write_csv(backtests / f"campaign_paths_energy_{tag}.csv", evaluation.energy)
        ),
        "endpoint_parity": str(
            _write_csv(
                backtests / f"campaign_paths_endpoint_parity_{tag}.csv",
                evaluation.endpoint_parity,
            )
        ),
    }
    diagnostics.mkdir(parents=True, exist_ok=True)
    summary_path = diagnostics / f"campaign_paths_summary_{tag}.json"
    summary_path.write_text(
        json.dumps(evaluation.summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    written["summary"] = str(summary_path)
    return written


__all__ = [
    "CampaignPathEvaluation",
    "DEFAULT_BASE_SEED",
    "DEFAULT_ORIGIN_STRIDE_DAYS",
    "DEFAULT_PATH_DAYS",
    "DEFAULT_SAMPLES",
    "ENERGY_SUBSAMPLE",
    "MODEL_IDS",
    "build_origins",
    "crps_matrix",
    "evaluate_campaign_paths",
    "write_evaluation_artifacts",
]
