"""Forecast model protocol, distributions, and dynamics variants for Dynamics Calibration v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence
import numpy as np

from .backtest_context import ForecastContext
from .backtest_metrics import precompute_crps_sample_term
from .clr import clr_to_composition_matrix, composition_to_clr
from .state_config import ALL_CATEGORIES
from .transitions import (
    MIN_TRANSITIONS,
    RECENCY_HALF_LIFE_DAYS,
    HistoricalTransition,
    compute_recency_weights,
    filter_transitions_as_of,
)


QUANTILES_TO_TRACK: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


@dataclass(frozen=True)
class ForecastDistribution:
    """Probabilistic forecast output for all canonical party categories."""

    model_id: str
    parties: tuple[str, ...]
    samples_by_party: dict[str, np.ndarray]
    point_forecast: dict[str, float]  # Strictly defined as predictive P50 (median)
    predictive_mean: dict[str, float]
    quantiles_by_party: dict[str, dict[float, float]]
    crps_sample_terms: dict[str, float]
    samples_count: int
    seed: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


class ForecastModel(Protocol):
    """Protocol for all forecasting models in the backtesting framework."""

    model_id: str

    def forecast(
        self,
        context: ForecastContext,
        horizon_days: int,
        samples_count: int,
        seed: int,
    ) -> ForecastDistribution:
        """Produce a probabilistic forecast distribution for a given horizon."""
        ...


def _build_distribution_from_matrix(
    model_id: str,
    samples_matrix: np.ndarray,
    samples_count: int,
    seed: int,
    categories: Sequence[str] = ALL_CATEGORIES,
    diagnostics: dict[str, Any] | None = None,
) -> ForecastDistribution:
    """Helper to compute sorted samples, quantiles, predictive P50, and CRPS sample terms from (N, D) matrix."""
    samples_by_party: dict[str, np.ndarray] = {}
    point_forecast: dict[str, float] = {}
    predictive_mean: dict[str, float] = {}
    quantiles_by_party: dict[str, dict[float, float]] = {}
    crps_sample_terms: dict[str, float] = {}

    for i, party in enumerate(categories):
        sorted_arr = np.sort(samples_matrix[:, i])
        samples_by_party[party] = sorted_arr

        # Exact point forecast = predictive P50 (median)
        q_vals = np.quantile(sorted_arr, QUANTILES_TO_TRACK, method="linear")
        q_dict = {q: float(val) for q, val in zip(QUANTILES_TO_TRACK, q_vals)}

        point_forecast[party] = q_dict[0.50]
        predictive_mean[party] = float(np.mean(sorted_arr))
        quantiles_by_party[party] = q_dict
        crps_sample_terms[party] = precompute_crps_sample_term(sorted_arr)

    return ForecastDistribution(
        model_id=model_id,
        parties=tuple(categories),
        samples_by_party=samples_by_party,
        point_forecast=point_forecast,
        predictive_mean=predictive_mean,
        quantiles_by_party=quantiles_by_party,
        crps_sample_terms=crps_sample_terms,
        samples_count=samples_count,
        seed=seed,
        diagnostics=diagnostics or {},
    )


class PointPersistenceModel:
    """Deterministic Point Persistence baseline (theta_{t+h} = PoP_t)."""

    model_id: str = "point_persistence"

    def forecast(
        self,
        context: ForecastContext,
        horizon_days: int,
        samples_count: int,
        seed: int,
    ) -> ForecastDistribution:
        if context.origin_pop is None:
            raise ValueError("PointPersistenceModel requires context.origin_pop")

        base_row = np.array([context.origin_pop[cat] for cat in ALL_CATEGORIES], dtype=float)
        matrix = np.tile(base_row, (samples_count, 1))

        return _build_distribution_from_matrix(
            model_id=self.model_id,
            samples_matrix=matrix,
            samples_count=samples_count,
            seed=seed,
        )


class EmpiricalRawModel:
    """Direct historical CLR transition resampling (preserves empirical historical drift)."""

    model_id: str = "empirical_raw"

    def forecast(
        self,
        context: ForecastContext,
        horizon_days: int,
        samples_count: int,
        seed: int,
    ) -> ForecastDistribution:
        if context.origin_clr is None:
            raise ValueError("EmpiricalRawModel requires context.origin_clr")

        transitions: Sequence[HistoricalTransition]
        if context.eligible_transitions_by_horizon and horizon_days in context.eligible_transitions_by_horizon:
            transitions = context.eligible_transitions_by_horizon[horizon_days]
        else:
            transitions = [t for t in context.transitions if t.horizon_days == horizon_days and t.end_date <= context.origin_date]

        if len(transitions) < MIN_TRANSITIONS:
            raise ValueError(
                f"Insufficient historical transitions ({len(transitions)} < {MIN_TRANSITIONS}) "
                f"for horizon {horizon_days}d at origin {context.origin_date}"
            )

        delta_matrix = np.array([t.clr_transition for t in transitions], dtype=float)

        rng = np.random.default_rng(seed)
        sampled_indices = rng.integers(0, len(transitions), size=samples_count)
        sampled_deltas = delta_matrix[sampled_indices]

        sampled_clr = context.origin_clr + sampled_deltas
        samples_matrix = clr_to_composition_matrix(sampled_clr)

        earliest_end = min(t.end_date for t in transitions).isoformat()
        latest_end = max(t.end_date for t in transitions).isoformat()

        diagnostics = {
            "eligible_transition_count": len(transitions),
            "earliest_transition_end": earliest_end,
            "latest_transition_end": latest_end,
            "weighted_mean_age_days": round(float(np.mean([(context.origin_date - t.end_date).days for t in transitions])), 2),
            "kish_effective_transition_count": float(len(transitions)),
        }

        return _build_distribution_from_matrix(
            model_id=self.model_id,
            samples_matrix=samples_matrix,
            samples_count=samples_count,
            seed=seed,
            diagnostics=diagnostics,
        )


class BaseSymmetricModel:
    """Base class for all sign-symmetric CLR transition models."""

    model_id: str = "symmetric_all_history"
    lookback_years: int | None = None
    use_recency_weighting: bool = False
    half_life_days: float = RECENCY_HALF_LIFE_DAYS

    def forecast(
        self,
        context: ForecastContext,
        horizon_days: int,
        samples_count: int,
        seed: int,
    ) -> ForecastDistribution:
        if context.origin_clr is None:
            raise ValueError(f"{self.model_id} requires context.origin_clr")

        raw_pool: Sequence[HistoricalTransition]
        if context.eligible_transitions_by_horizon and horizon_days in context.eligible_transitions_by_horizon:
            raw_pool = context.eligible_transitions_by_horizon[horizon_days]
        else:
            raw_pool = [t for t in context.transitions if t.horizon_days == horizon_days and t.end_date <= context.origin_date]

        # Apply calendar lookback window if configured
        if self.lookback_years is not None:
            transitions = filter_transitions_as_of(raw_pool, context.origin_date, lookback_years=self.lookback_years)
        else:
            transitions = tuple(raw_pool)

        if len(transitions) < MIN_TRANSITIONS:
            raise ValueError(
                f"Insufficient historical transitions ({len(transitions)} < {MIN_TRANSITIONS}) "
                f"for {self.model_id} horizon {horizon_days}d at origin {context.origin_date}"
            )

        delta_matrix = np.array([t.clr_transition for t in transitions], dtype=float)
        rng = np.random.default_rng(seed)

        if self.use_recency_weighting:
            probs, kish_eff, w_age = compute_recency_weights(transitions, context.origin_date, half_life_days=self.half_life_days)
            sampled_indices = rng.choice(len(transitions), size=samples_count, p=probs)
        else:
            kish_eff = float(len(transitions))
            w_age = float(np.mean([(context.origin_date - t.end_date).days for t in transitions]))
            sampled_indices = rng.integers(0, len(transitions), size=samples_count)

        sampled_deltas = delta_matrix[sampled_indices]
        signs = rng.choice([-1.0, 1.0], size=(samples_count, 1))
        sampled_clr = context.origin_clr + (signs * sampled_deltas)

        samples_matrix = clr_to_composition_matrix(sampled_clr)

        earliest_end = min(t.end_date for t in transitions).isoformat()
        latest_end = max(t.end_date for t in transitions).isoformat()

        diagnostics = {
            "eligible_transition_count": len(transitions),
            "earliest_transition_end": earliest_end,
            "latest_transition_end": latest_end,
            "weighted_mean_age_days": round(w_age, 2),
            "kish_effective_transition_count": round(kish_eff, 2),
        }

        return _build_distribution_from_matrix(
            model_id=self.model_id,
            samples_matrix=samples_matrix,
            samples_count=samples_count,
            seed=seed,
            diagnostics=diagnostics,
        )


class SymmetricAllHistoryModel(BaseSymmetricModel):
    """Sign-symmetric historical CLR transitions using all available history."""
    model_id: str = "symmetric_all_history"
    lookback_years: int | None = None
    use_recency_weighting: bool = False


# Backwards-compatible alias for Dynamics v1
EmpiricalSymmetricModel = SymmetricAllHistoryModel



class Symmetric4YModel(BaseSymmetricModel):
    """Sign-symmetric historical CLR transitions restricted to trailing 4-calendar-year window."""
    model_id: str = "symmetric_4y"
    lookback_years: int | None = 4
    use_recency_weighting: bool = False


class Symmetric2YModel(BaseSymmetricModel):
    """Sign-symmetric historical CLR transitions restricted to trailing 2-calendar-year window."""
    model_id: str = "symmetric_2y"
    lookback_years: int | None = 2
    use_recency_weighting: bool = False


class SymmetricRecencyWeightedModel(BaseSymmetricModel):
    """Sign-symmetric historical CLR transitions with 730-day exponential recency weighting."""
    model_id: str = "symmetric_recency_weighted"
    lookback_years: int | None = None
    use_recency_weighting: bool = True
    half_life_days: float = RECENCY_HALF_LIFE_DAYS


class NoChangeModel:
    """No-Change baseline model (theta_{t+h} = theta_t)."""

    model_id: str = "no_change"

    def forecast(
        self,
        context: ForecastContext,
        horizon_days: int,
        samples_count: int,
        seed: int,
    ) -> ForecastDistribution:
        if context.opinion_state is None:
            raise ValueError("NoChangeModel requires context.opinion_state")

        raw_samples = context.opinion_state.sample(n=samples_count, seed=seed)
        samples_matrix = np.array([[s[cat] for cat in ALL_CATEGORIES] for s in raw_samples], dtype=float)

        return _build_distribution_from_matrix(
            model_id=self.model_id,
            samples_matrix=samples_matrix,
            samples_count=samples_count,
            seed=seed,
        )


MODELS: dict[str, ForecastModel] = {
    "point_persistence": PointPersistenceModel(),
    "empirical_raw": EmpiricalRawModel(),
    "symmetric_all_history": SymmetricAllHistoryModel(),
    "empirical_symmetric": SymmetricAllHistoryModel(),  # Backwards-compatible alias
    "symmetric_4y": Symmetric4YModel(),
    "symmetric_2y": Symmetric2YModel(),
    "symmetric_recency_weighted": SymmetricRecencyWeightedModel(),
    "no_change": NoChangeModel(),
}
