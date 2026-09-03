"""Prospective 2026 scoring contract.

The public benchmark compares forecasts on eight Swedish parliamentary
parties, in percentage points of the official national valid-vote
denominator, in this fixed order::

    M, L, C, KD, S, V, MP, SD

The primary draw-based estimators are *fair* finite-ensemble estimators.  For
an ensemble of independent draws, the sample-only term is a U-statistic over
distinct pairs, not the usual V-statistic that includes zero self-pairs.  This
matters when ensemble sizes differ.  The V-statistic remains available under
explicit ``*_v_statistic`` names as a sensitivity result.

No function in this module creates a predictive distribution from a point,
standard deviation, interval, or marginal quantiles.  Draw scoring is only
valid when its caller has independently established that the archived draws
are the model's verified predictive draws.  The fallback selector therefore
requires explicit verification flags and compatible published quantiles.

For large multivariate ensembles the exact Energy Score U-statistic is
quadratic in the number of draws.  ``fair_energy_score`` supports an explicit
uniform pair sample, which is an unbiased Monte Carlo estimator of the same
U-statistic.  If used, callers should record the pair-sampling seed and count
in provenance; no automatic sampling is performed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np


PRIMARY_PARTY_ORDER: tuple[str, ...] = ("M", "L", "C", "KD", "S", "V", "MP", "SD")
# The protocol pre-registers these central levels for WIS. Draw-based
# interval reporting uses the same defaults; callers may request a strict
# subset when a source genuinely publishes fewer intervals.
DEFAULT_INTERVAL_LEVELS: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95)
WIS_CANDIDATE_INTERVAL_LEVELS: tuple[float, ...] = DEFAULT_INTERVAL_LEVELS

PROBABILISTIC_TIER_FAIR_DRAWS = "fair_draws"
PROBABILISTIC_TIER_WIS = "compatible_published_quantiles_wis"
PROBABILISTIC_TIER_POINT_MAE = "point_mae_only"

_EPSILON = 1e-12


def _finite_vector(values: Sequence[float] | np.ndarray, *, name: str, min_size: int = 1) -> np.ndarray:
    """Convert a vector input and reject empty, malformed, or non-finite data."""

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size < min_size:
        raise ValueError(f"{name} requires a one-dimensional array with at least {min_size} value(s)")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _finite_matrix(values: np.ndarray | Sequence[Sequence[float]], *, name: str, min_rows: int = 1) -> np.ndarray:
    """Convert a draw matrix and reject empty, malformed, or non-finite data."""

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < min_rows or arr.shape[1] == 0:
        raise ValueError(
            f"{name} requires a two-dimensional array with at least {min_rows} row(s) and one column"
        )
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _finite_scalar(value: float, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_interval_level(level: float) -> float:
    value = _finite_scalar(level, name="interval level")
    if not 0.0 < value < 1.0:
        raise ValueError("interval levels must lie strictly between 0 and 1")
    return value


def _normalise_interval_levels(levels: Sequence[float]) -> tuple[float, ...]:
    normalised = tuple(_validate_interval_level(level) for level in levels)
    if not normalised:
        raise ValueError("at least one interval level is required")
    if len(set(normalised)) != len(normalised):
        raise ValueError("interval levels must be unique")
    return tuple(sorted(normalised))


def _validate_party_order(party_order: Sequence[str]) -> tuple[str, ...]:
    parties = tuple(str(party) for party in party_order)
    if parties != PRIMARY_PARTY_ORDER:
        raise ValueError(
            "the prospective vote-share contract requires party order "
            f"{PRIMARY_PARTY_ORDER!r}; do not substitute or reorder parties"
        )
    return parties


def _pairwise_abs_sum_sorted(values: np.ndarray) -> float:
    """Return ``sum(i < j, |x_i - x_j|)`` in O(n log n) time."""

    ordered = np.sort(values)
    n = ordered.size
    # For x_(0) <= ... <= x_(n-1), each x_(k) appears with coefficient
    # (2*k - n + 1) in the sum over unordered pair distances.
    coefficients = 2.0 * np.arange(n, dtype=np.float64) - n + 1.0
    return float(np.dot(coefficients, ordered))


def crps_v_statistic(samples: Sequence[float] | np.ndarray, actual: float) -> float:
    """Return the empirical CRPS using the finite-ensemble V-statistic.

    This is retained solely as a sensitivity metric.  Its sample-only term is
    ``(2 n^2)^-1 sum_{i,j} |x_i-x_j|`` and includes the zero diagonal.
    """

    values = _finite_vector(samples, name="CRPS samples")
    target = _finite_scalar(actual, name="CRPS actual")
    n = values.size
    first_term = float(np.mean(np.abs(values - target)))
    sample_term = _pairwise_abs_sum_sorted(values) / float(n * n)
    return float(first_term - sample_term)


def fair_crps(samples: Sequence[float] | np.ndarray, actual: float) -> float:
    r"""Return the fair finite-ensemble CRPS (the pairwise U-statistic).

    For ``n >= 2`` draws ``x_i`` and outcome ``y`` this is

    .. math::

       n^{-1} \sum_i |x_i-y|
       - [2n(n-1)]^{-1} \sum_{i\ne j}|x_i-x_j|.

    Unlike the V-statistic, no self-pair is included in the sample-only term.
    The estimator is intentionally undefined for one draw: silently falling
    back to a point-mass CRPS would change the declared estimator.
    """

    values = _finite_vector(samples, name="fair CRPS samples", min_size=2)
    target = _finite_scalar(actual, name="fair CRPS actual")
    n = values.size
    first_term = float(np.mean(np.abs(values - target)))
    unordered_pair_sum = _pairwise_abs_sum_sorted(values)
    sample_term = unordered_pair_sum / float(n * (n - 1))
    return float(first_term - sample_term)


def _validate_energy_inputs(
    samples: np.ndarray | Sequence[Sequence[float]],
    actual: Sequence[float] | np.ndarray,
    *,
    min_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = _finite_matrix(samples, name="Energy Score samples", min_rows=min_rows)
    target = _finite_vector(actual, name="Energy Score actual")
    if values.shape[1] != target.size:
        raise ValueError(
            "Energy Score samples and actual must have matching dimensions "
            f"({values.shape[1]} != {target.size})"
        )
    return values, target


def _energy_pair_sum(values: np.ndarray, *, chunk_size: int) -> float:
    """Calculate all ordered off-diagonal pair distances in chunks."""

    if not isinstance(chunk_size, (int, np.integer)) or int(chunk_size) < 1:
        raise ValueError("chunk_size must be a positive integer")
    n = values.shape[0]
    total = 0.0
    block = int(chunk_size)
    for start in range(0, n, block):
        chunk = values[start : start + block]
        distances = np.linalg.norm(chunk[:, None, :] - values[None, :, :], axis=2)
        total += float(np.sum(distances, dtype=np.float64))
    # Diagonal distances are exactly zero, so the sum above is also the sum
    # over i != j.  Keeping this explicit documents the U-statistic mapping.
    return total


def _sample_energy_pair_distances(
    values: np.ndarray,
    *,
    pair_sample_size: int,
    random_seed: int,
) -> np.ndarray:
    """Draw uniform ordered distinct index pairs and return their distances."""

    if not isinstance(pair_sample_size, (int, np.integer)) or int(pair_sample_size) < 1:
        raise ValueError("pair_sample_size must be a positive integer")
    if not isinstance(random_seed, (int, np.integer)):
        raise ValueError("random_seed must be an integer")
    n = values.shape[0]
    # The caller only reaches here for n >= 2.  Mapping j from [0,n-2] to a
    # value other than i gives an exact uniform draw from the n-1 alternatives
    # without rejection, even for a small ensemble.
    rng = np.random.default_rng(int(random_seed))
    i = rng.integers(0, n, size=int(pair_sample_size), dtype=np.int64)
    j = rng.integers(0, n - 1, size=int(pair_sample_size), dtype=np.int64)
    j += j >= i
    return np.linalg.norm(values[i] - values[j], axis=1)


def fair_energy_score(
    samples: np.ndarray | Sequence[Sequence[float]],
    actual: Sequence[float] | np.ndarray,
    *,
    chunk_size: int = 512,
    pair_sample_size: int | None = None,
    random_seed: int = 0,
) -> float:
    r"""Return the fair finite-ensemble multivariate Energy Score.

    The exact estimator is

    .. math::

       n^{-1}\sum_i \|x_i-y\|_2
       - [2n(n-1)]^{-1}\sum_{i\ne j}\|x_i-x_j\|_2.

    ``pair_sample_size=None`` computes the exact U-statistic by chunking the
    full pair matrix.  For very large ensembles, pass an explicit positive
    ``pair_sample_size``.  The resulting uniform-pair average is an unbiased
    Monte Carlo estimator of the same U-statistic and never silently changes
    the estimator.  Its seed and sample count should be archived by the
    caller.  At least two draws are required for the fair correction.
    """

    values, target = _validate_energy_inputs(samples, actual, min_rows=2)
    n = values.shape[0]
    first_term = float(np.mean(np.linalg.norm(values - target[None, :], axis=1)))
    if pair_sample_size is None:
        pair_sum = _energy_pair_sum(values, chunk_size=chunk_size)
        pair_mean = pair_sum / float(n * (n - 1))
    else:
        distances = _sample_energy_pair_distances(
            values,
            pair_sample_size=pair_sample_size,
            random_seed=random_seed,
        )
        pair_mean = float(np.mean(distances))
    return float(first_term - 0.5 * pair_mean)


def energy_score_v_statistic(
    samples: np.ndarray | Sequence[Sequence[float]],
    actual: Sequence[float] | np.ndarray,
    *,
    chunk_size: int = 512,
    pair_sample_size: int | None = None,
    random_seed: int = 0,
) -> float:
    """Return the existing finite-ensemble V-statistic Energy Score.

    This is a sensitivity metric only.  The pair denominator is ``2 n^2``;
    diagonal self-pairs are included as zero distances.  ``pair_sample_size``
    may be supplied for a disclosed uniform-pair Monte Carlo estimate when an
    exact quadratic calculation is impractical; the deterministic conversion
    from the distinct-pair mean to the V-statistic mean is applied.
    """

    values, target = _validate_energy_inputs(samples, actual, min_rows=1)
    n = values.shape[0]
    first_term = float(np.mean(np.linalg.norm(values - target[None, :], axis=1)))
    if pair_sample_size is None:
        pair_sum = _energy_pair_sum(values, chunk_size=chunk_size)
        pair_mean = pair_sum / float(n * n)
    else:
        if n < 2:
            # There are no off-diagonal pairs, so the V-statistic pair term is
            # zero even though the fair U-statistic is undefined for n=1.
            pair_mean = 0.0
        else:
            distances = _sample_energy_pair_distances(
                values,
                pair_sample_size=pair_sample_size,
                random_seed=random_seed,
            )
            # Convert the ordered-distinct-pair mean to the V-statistic mean:
            # the n diagonal terms are zero, hence E_V = (n-1)/n E_U.
            pair_mean = float((n - 1) / n * np.mean(distances))
    return float(first_term - 0.5 * pair_mean)


def threshold_probability(
    samples: Sequence[float] | np.ndarray,
    threshold: float = 4.0,
) -> float:
    """Return ``P(vote share >= threshold)`` from explicit predictive draws."""

    values = _finite_vector(samples, name="threshold samples")
    cutoff = _finite_scalar(threshold, name="threshold")
    return float(np.mean(values >= cutoff))


def threshold_brier(
    samples: Sequence[float] | np.ndarray,
    actual: float,
    threshold: float = 4.0,
) -> float:
    """Return the inclusive threshold Brier score from predictive draws."""

    probability = threshold_probability(samples, threshold=threshold)
    return threshold_brier_from_probability(probability, actual, threshold=threshold)


def threshold_brier_from_probability(
    probability: float,
    actual: float,
    *,
    threshold: float = 4.0,
) -> float:
    """Return Brier score from a published event probability.

    This accepts an already published probability and never treats a missing
    probability as zero.  The event is inclusive: ``actual >= threshold``.
    """

    p = _finite_scalar(probability, name="threshold probability")
    if not 0.0 <= p <= 1.0:
        raise ValueError("threshold probability must be between 0 and 1")
    y = _finite_scalar(actual, name="threshold actual")
    cutoff = _finite_scalar(threshold, name="threshold")
    event = float(y >= cutoff)
    return float((p - event) ** 2)


def point_mae(point_forecast: Sequence[float] | np.ndarray | float, actual: Sequence[float] | np.ndarray | float) -> float:
    """Return mean absolute error for explicitly published central forecasts."""

    forecast = np.asarray(point_forecast, dtype=np.float64)
    target = np.asarray(actual, dtype=np.float64)
    if forecast.shape != target.shape or forecast.ndim == 0:
        if forecast.ndim == 0 and target.ndim == 0:
            forecast = forecast.reshape(1)
            target = target.reshape(1)
        else:
            raise ValueError("point forecast and actual must have the same non-scalar shape")
    if forecast.size == 0:
        raise ValueError("point MAE requires at least one forecast value")
    if not np.isfinite(forecast).all() or not np.isfinite(target).all():
        raise ValueError("point forecast and actual must contain only finite values")
    return float(np.mean(np.abs(forecast - target)))


def central_interval_metrics(
    samples: Sequence[float] | np.ndarray,
    actual: float,
    *,
    levels: Sequence[float] = DEFAULT_INTERVAL_LEVELS,
) -> dict[str, dict[str, float | bool]]:
    """Return empirical central interval coverage and width from verified draws."""

    values = _finite_vector(samples, name="interval samples")
    target = _finite_scalar(actual, name="interval actual")
    normalised = _normalise_interval_levels(levels)
    out: dict[str, dict[str, float | bool]] = {}
    for level in normalised:
        alpha = 1.0 - level
        lower, upper = np.quantile(
            values,
            [alpha / 2.0, 1.0 - alpha / 2.0],
            method="linear",
        )
        out[f"{level:g}"] = {
            "level": float(level),
            "lower": float(lower),
            "upper": float(upper),
            "width": float(upper - lower),
            "covered": bool(float(lower) <= target <= float(upper)),
        }
    return out


def interval_coverage_width(lower: float, upper: float, actual: float) -> dict[str, float | bool]:
    """Score one explicitly published interval without manufacturing a law."""

    low = _finite_scalar(lower, name="interval lower bound")
    high = _finite_scalar(upper, name="interval upper bound")
    target = _finite_scalar(actual, name="interval actual")
    if high < low:
        raise ValueError("interval upper bound must be at least its lower bound")
    return {
        "lower": low,
        "upper": high,
        "width": float(high - low),
        "covered": bool(low <= target <= high),
    }


def _probability_key(value: float | int | str) -> float:
    """Normalise numeric quantile keys without inferring a distribution."""

    if isinstance(value, str):
        text = value.strip().lower()
        if text.startswith("p"):
            text = text[1:]
            try:
                number = float(text) / 100.0
            except ValueError as exc:
                raise ValueError(f"invalid quantile key: {value!r}") from exc
        else:
            try:
                number = float(text)
            except ValueError as exc:
                raise ValueError(f"invalid quantile key: {value!r}") from exc
    else:
        number = float(value)
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"quantile key must represent a probability in [0, 1], got {value!r}")
    return number


def _normalise_quantiles(quantiles: Mapping[float | int | str, float]) -> dict[float, float]:
    if not isinstance(quantiles, Mapping) or not quantiles:
        raise ValueError("published quantiles must be a non-empty mapping")
    out: dict[float, float] = {}
    for raw_key, raw_value in quantiles.items():
        key = _probability_key(raw_key)
        if any(abs(key - existing) <= _EPSILON for existing in out):
            raise ValueError("published quantiles contain duplicate probability keys")
        out[key] = _finite_scalar(raw_value, name=f"published quantile {raw_key!r}")
    return out


def _quantile_for(quantiles: Mapping[float, float], probability: float) -> float:
    for key, value in quantiles.items():
        if abs(key - probability) <= _EPSILON:
            return value
    raise ValueError(f"published quantiles are missing required probability {probability:g}")


def weighted_interval_score(
    quantiles: Mapping[float | int | str, float],
    actual: float,
    *,
    interval_levels: Sequence[float] = DEFAULT_INTERVAL_LEVELS,
) -> float:
    """Return WIS for explicitly published, compatible central quantiles.

    For central interval level ``1-alpha`` the interval score is

    ``(upper-lower) + 2/alpha*(lower-y)*I(y<lower) +
    2/alpha*(y-upper)*I(y>upper)``.

    The median has weight ``1/2`` and each interval has weight ``alpha/2``;
    the weighted sum is normalised by the sum of those weights.  This function
    requires a median and both endpoints for every requested level.  It does
    not interpolate, fit, or otherwise manufacture unreported quantiles.
    """

    values = _normalise_quantiles(quantiles)
    target = _finite_scalar(actual, name="WIS actual")
    levels = _normalise_interval_levels(interval_levels)
    median = _quantile_for(values, 0.5)
    numerator = 0.5 * abs(median - target)
    denominator = 0.5
    for level in levels:
        alpha = 1.0 - level
        lower = _quantile_for(values, alpha / 2.0)
        upper = _quantile_for(values, 1.0 - alpha / 2.0)
        if upper < lower:
            raise ValueError("published quantile interval bounds are not ordered")
        interval_score = upper - lower
        if target < lower:
            interval_score += 2.0 / alpha * (lower - target)
        elif target > upper:
            interval_score += 2.0 / alpha * (target - upper)
        weight = alpha / 2.0
        numerator += weight * interval_score
        denominator += weight
    return float(numerator / denominator)


def quantile_levels_compatible(
    first: Mapping[float | int | str, float],
    second: Mapping[float | int | str, float],
    *,
    interval_levels: Sequence[float] = DEFAULT_INTERVAL_LEVELS,
) -> bool:
    """Return whether two forecasts expose the same required quantile levels."""

    try:
        first_values = _normalise_quantiles(first)
        second_values = _normalise_quantiles(second)
        levels = _normalise_interval_levels(interval_levels)
        required = {0.5}
        for level in levels:
            alpha = 1.0 - level
            required.update((alpha / 2.0, 1.0 - alpha / 2.0))
        return all(any(abs(key - required_key) <= _EPSILON for key in first_values) for required_key in required) and all(
            any(abs(key - required_key) <= _EPSILON for key in second_values) for required_key in required
        )
    except (TypeError, ValueError):
        return False


def compatible_quantile_forecasts(
    first: Mapping[str, Mapping[float | int | str, float]] | None,
    second: Mapping[str, Mapping[float | int | str, float]] | None,
    *,
    party_order: Sequence[str] = PRIMARY_PARTY_ORDER,
    interval_levels: Sequence[float] = DEFAULT_INTERVAL_LEVELS,
) -> bool:
    """Return whether both models expose a non-empty common WIS level."""

    return bool(
        common_wis_interval_levels(
            first,
            second,
            party_order=party_order,
            candidate_levels=interval_levels,
        )
    )


def common_wis_interval_levels(
    first: Mapping[str, Mapping[float | int | str, float]] | None,
    second: Mapping[str, Mapping[float | int | str, float]] | None,
    *,
    party_order: Sequence[str] = PRIMARY_PARTY_ORDER,
    candidate_levels: Sequence[float] = WIS_CANDIDATE_INTERVAL_LEVELS,
) -> tuple[float, ...]:
    """Return common WIS levels available for every benchmark party.

    The fallback protocol permits any non-empty intersection of the
    pre-registered 50%, 80%, 90%, and 95% central levels. Both forecasts must
    expose an explicit median and both endpoints at each returned level. A
    level present for only one party is excluded rather than converted to a
    missing value or silently imputed. An empty tuple means WIS is not
    supported by the available evidence.
    """

    try:
        parties = _validate_party_order(party_order)
        levels = _normalise_interval_levels(candidate_levels)
        if first is None or second is None:
            return ()
        if set(first) != set(parties) or set(second) != set(parties):
            return ()
        first_values = {party: _normalise_quantiles(first[party]) for party in parties}
        second_values = {party: _normalise_quantiles(second[party]) for party in parties}
        # WIS always includes the median absolute-error term. Do not return a
        # probabilistic tier if either source omits it for any party.
        if any(
            not any(abs(key - 0.5) <= _EPSILON for key in first_values[party])
            or not any(abs(key - 0.5) <= _EPSILON for key in second_values[party])
            for party in parties
        ):
            return ()

        common: list[float] = []
        for level in levels:
            alpha = 1.0 - level
            lower_probability = alpha / 2.0
            upper_probability = 1.0 - alpha / 2.0
            if all(
                any(abs(key - lower_probability) <= _EPSILON for key in first_values[party])
                and any(abs(key - upper_probability) <= _EPSILON for key in first_values[party])
                and any(abs(key - lower_probability) <= _EPSILON for key in second_values[party])
                and any(abs(key - upper_probability) <= _EPSILON for key in second_values[party])
                for party in parties
            ):
                common.append(level)
        return tuple(common)
    except (TypeError, ValueError, KeyError):
        return ()


def select_primary_scoring_tier(
    *,
    election_simulator_draws_verified: bool,
    botten_ada_draws_verified: bool,
    election_simulator_quantiles: Mapping[str, Mapping[float | int | str, float]] | None = None,
    botten_ada_quantiles: Mapping[str, Mapping[float | int | str, float]] | None = None,
    party_order: Sequence[str] = PRIMARY_PARTY_ORDER,
    interval_levels: Sequence[float] = DEFAULT_INTERVAL_LEVELS,
) -> Literal["fair_draws", "compatible_published_quantiles_wis", "point_mae_only"]:
    """Choose the preregistered probabilistic fallback tier.

    Draw arrays are deliberately not inspected here.  The caller must supply
    an evidence-backed ``*_draws_verified`` flag; an RDS object or an array
    reconstructed from quantiles cannot opt itself into the draw tier.
    """

    # Do not interpret truthy strings (for example ``"false"`` from a loose
    # manifest parser) as evidence.  JSON provenance should provide booleans.
    if election_simulator_draws_verified is True and botten_ada_draws_verified is True:
        return PROBABILISTIC_TIER_FAIR_DRAWS
    if compatible_quantile_forecasts(
        election_simulator_quantiles,
        botten_ada_quantiles,
        party_order=party_order,
        interval_levels=interval_levels,
    ):
        return PROBABILISTIC_TIER_WIS
    return PROBABILISTIC_TIER_POINT_MAE


def _score_point_forecast(
    point_forecast: Sequence[float] | np.ndarray | None,
    actual: np.ndarray,
    *,
    party_order: tuple[str, ...],
) -> dict[str, Any]:
    if point_forecast is None:
        return {
            "status": "UNAVAILABLE_NOT_PUBLISHED",
            "values": None,
            "per_party_absolute_error": None,
            "mean_mae": None,
        }
    values = _finite_vector(point_forecast, name="published central forecast")
    if values.shape != actual.shape:
        raise ValueError("published central forecast must match the eight-party actual vector")
    errors = np.abs(values - actual)
    return {
        "status": "AVAILABLE_EXPLICIT",
        "values": [float(value) for value in values],
        "per_party_absolute_error": {
            party: float(errors[index]) for index, party in enumerate(party_order)
        },
        "mean_mae": float(np.mean(errors)),
    }


def _score_published_quantiles(
    quantiles: Mapping[str, Mapping[float | int | str, float]],
    actual: np.ndarray,
    *,
    party_order: tuple[str, ...],
    interval_levels: Sequence[float],
) -> dict[str, Any]:
    if set(quantiles) != set(party_order):
        raise ValueError("published quantiles must contain exactly the eight benchmark parties")
    per_party: dict[str, Any] = {}
    wis_values: list[float] = []
    medians: list[float] = []
    for index, party in enumerate(party_order):
        values = _normalise_quantiles(quantiles[party])
        wis = weighted_interval_score(values, actual[index], interval_levels=interval_levels)
        median = _quantile_for(values, 0.5)
        intervals: dict[str, Any] = {}
        for level in _normalise_interval_levels(interval_levels):
            alpha = 1.0 - level
            lower = _quantile_for(values, alpha / 2.0)
            upper = _quantile_for(values, 1.0 - alpha / 2.0)
            intervals[f"{level:g}"] = interval_coverage_width(lower, upper, actual[index]) | {
                "level": float(level)
            }
        per_party[party] = {
            "wis": wis,
            "median": median,
            "point_absolute_error_from_published_median": float(abs(median - actual[index])),
            "central_intervals": intervals,
        }
        wis_values.append(wis)
        medians.append(median)
    return {
        "status": "AVAILABLE_EXPLICIT_QUANTILES",
        "interval_levels": [float(level) for level in _normalise_interval_levels(interval_levels)],
        "per_party": per_party,
        "mean_wis": float(np.mean(wis_values)),
        "median_mae": float(np.mean(np.abs(np.asarray(medians) - actual))),
    }


def score_vote_ensemble(
    samples: np.ndarray | Sequence[Sequence[float]],
    actual: Sequence[float] | np.ndarray,
    *,
    central_forecast: Sequence[float] | np.ndarray | None = None,
    party_order: Sequence[str] = PRIMARY_PARTY_ORDER,
    threshold_parties: Sequence[str] | None = None,
    threshold: float = 4.0,
    interval_levels: Sequence[float] = DEFAULT_INTERVAL_LEVELS,
    energy_pair_sample_size: int | None = None,
    energy_random_seed: int = 0,
    energy_chunk_size: int = 512,
) -> dict[str, Any]:
    """Score one model's verified vote-share draw ensemble.

    ``samples`` must be the model's actual predictive draws.  This function
    does not validate provenance or decide whether a source's draws are
    genuine; the pair-level API requires the caller to pass that evidence as
    an explicit verification flag.  Ensemble sizes need not match another
    model's size.
    """

    parties = _validate_party_order(party_order)
    values = _finite_matrix(samples, name="vote-share draws", min_rows=2)
    target = _finite_vector(actual, name="actual vote shares")
    if values.shape[1] != len(parties) or target.shape != (len(parties),):
        raise ValueError("vote-share draws and actual must each contain exactly the eight benchmark parties")
    levels = _normalise_interval_levels(interval_levels)
    threshold_set = set(parties if threshold_parties is None else threshold_parties)
    if not threshold_set.issubset(set(parties)):
        raise ValueError("threshold_parties must be drawn from the fixed benchmark party order")
    per_party: dict[str, Any] = {}
    fair_values: list[float] = []
    v_values: list[float] = []
    brier_values: list[float] = []
    for index, party in enumerate(parties):
        draws = values[:, index]
        fair = fair_crps(draws, target[index])
        v_stat = crps_v_statistic(draws, target[index])
        intervals = central_interval_metrics(draws, target[index], levels=levels)
        threshold_data: dict[str, Any]
        if party in threshold_set:
            probability = threshold_probability(draws, threshold=threshold)
            threshold_data = {
                "probability": probability,
                "brier": threshold_brier_from_probability(probability, target[index], threshold=threshold),
                "event": bool(target[index] >= float(threshold)),
            }
            brier_values.append(float(threshold_data["brier"]))
        else:
            threshold_data = {"probability": None, "brier": None, "event": None}
        per_party[party] = {
            "fair_crps": fair,
            "crps_v_statistic": v_stat,
            "threshold_4pct": threshold_data,
            "central_intervals": intervals,
            "sample_count": int(values.shape[0]),
        }
        fair_values.append(fair)
        v_values.append(v_stat)

    fair_energy = fair_energy_score(
        values,
        target,
        chunk_size=energy_chunk_size,
        pair_sample_size=energy_pair_sample_size,
        random_seed=energy_random_seed,
    )
    v_energy = energy_score_v_statistic(
        values,
        target,
        chunk_size=energy_chunk_size,
        pair_sample_size=energy_pair_sample_size,
        random_seed=energy_random_seed,
    )
    interval_aggregate: dict[str, Any] = {}
    for level in levels:
        key = f"{level:g}"
        interval_aggregate[key] = {
            "coverage_rate": float(
                np.mean([bool(per_party[party]["central_intervals"][key]["covered"]) for party in parties])
            ),
            "mean_width": float(
                np.mean([float(per_party[party]["central_intervals"][key]["width"]) for party in parties])
            ),
        }
    return {
        "status": "SCORED_VERIFIED_DRAWS",
        "units": "percentage_points",
        "party_order": list(parties),
        "sample_count": int(values.shape[0]),
        "per_party": per_party,
        "fair_crps_mean_8parties": float(np.mean(fair_values)),
        "crps_v_statistic_mean_8parties": float(np.mean(v_values)),
        "fair_energy_score": fair_energy,
        "energy_score_v_statistic": v_energy,
        "threshold_4pct": {
            "parties": [party for party in parties if party in threshold_set],
            "mean_brier": float(np.mean(brier_values)) if brier_values else None,
        },
        "central_intervals": interval_aggregate,
        "point_forecast": _score_point_forecast(central_forecast, target, party_order=parties),
        "energy_estimator": (
            "fair_u_statistic_exact"
            if energy_pair_sample_size is None
            else "fair_u_statistic_uniform_pair_monte_carlo"
        ),
        "energy_pair_sample_size": (
            None if energy_pair_sample_size is None else int(energy_pair_sample_size)
        ),
        "energy_random_seed": None if energy_pair_sample_size is None else int(energy_random_seed),
    }


def score_forecast_pair(
    actual: Sequence[float] | np.ndarray,
    *,
    election_simulator_draws: np.ndarray | Sequence[Sequence[float]] | None = None,
    botten_ada_draws: np.ndarray | Sequence[Sequence[float]] | None = None,
    election_simulator_draws_verified: bool = False,
    botten_ada_draws_verified: bool = False,
    election_simulator_quantiles: Mapping[str, Mapping[float | int | str, float]] | None = None,
    botten_ada_quantiles: Mapping[str, Mapping[float | int | str, float]] | None = None,
    election_simulator_central_forecast: Sequence[float] | np.ndarray | None = None,
    botten_ada_central_forecast: Sequence[float] | np.ndarray | None = None,
    party_order: Sequence[str] = PRIMARY_PARTY_ORDER,
    threshold_parties: Sequence[str] | None = None,
    threshold: float = 4.0,
    interval_levels: Sequence[float] = DEFAULT_INTERVAL_LEVELS,
    energy_pair_sample_size: int | None = None,
    election_simulator_energy_seed: int = 0,
    botten_ada_energy_seed: int = 0,
    energy_chunk_size: int = 512,
) -> dict[str, Any]:
    """Apply the preregistered tier hierarchy to one paired cutoff.

    The first tier is selected only when both models' draw artifacts are
    explicitly verified as predictive draws.  There is intentionally no
    equal-draw-count check.  If that tier is unavailable, both models need
    compatible published quantiles for WIS; otherwise the result reports only
    explicit point forecasts.  Missing point forecasts remain ``None``.
    """

    parties = _validate_party_order(party_order)
    target = _finite_vector(actual, name="actual vote shares")
    if target.shape != (len(parties),):
        raise ValueError("actual vote shares must contain exactly the eight benchmark parties")
    tier = select_primary_scoring_tier(
        election_simulator_draws_verified=election_simulator_draws_verified,
        botten_ada_draws_verified=botten_ada_draws_verified,
        election_simulator_quantiles=election_simulator_quantiles,
        botten_ada_quantiles=botten_ada_quantiles,
        party_order=parties,
        interval_levels=interval_levels,
    )
    result: dict[str, Any] = {
        "status": "SCORABLE",
        "primary_tier": tier,
        "party_order": list(parties),
        "units": "percentage_points",
        "election_simulator": None,
        "botten_ada": None,
    }
    if tier == PROBABILISTIC_TIER_FAIR_DRAWS:
        if election_simulator_draws is None or botten_ada_draws is None:
            raise ValueError("verified fair-draw scoring requires draw arrays for both models")
        result["election_simulator"] = score_vote_ensemble(
            election_simulator_draws,
            target,
            central_forecast=election_simulator_central_forecast,
            party_order=parties,
            threshold_parties=threshold_parties,
            threshold=threshold,
            interval_levels=interval_levels,
            energy_pair_sample_size=energy_pair_sample_size,
            energy_random_seed=election_simulator_energy_seed,
            energy_chunk_size=energy_chunk_size,
        )
        result["botten_ada"] = score_vote_ensemble(
            botten_ada_draws,
            target,
            central_forecast=botten_ada_central_forecast,
            party_order=parties,
            threshold_parties=threshold_parties,
            threshold=threshold,
            interval_levels=interval_levels,
            energy_pair_sample_size=energy_pair_sample_size,
            energy_random_seed=botten_ada_energy_seed,
            energy_chunk_size=energy_chunk_size,
        )
        return result
    if tier == PROBABILISTIC_TIER_WIS:
        # Compatibility was checked by the selector.  Revalidate while
        # scoring so a concurrent mutable mapping cannot silently pass through
        # with a different shape.
        if not compatible_quantile_forecasts(
            election_simulator_quantiles,
            botten_ada_quantiles,
            party_order=parties,
            interval_levels=interval_levels,
        ):
            raise ValueError("published quantile mappings changed after compatibility validation")
        assert election_simulator_quantiles is not None
        assert botten_ada_quantiles is not None
        common_levels = common_wis_interval_levels(
            election_simulator_quantiles,
            botten_ada_quantiles,
            party_order=parties,
            candidate_levels=interval_levels,
        )
        if not common_levels:
            raise ValueError("published quantile mappings have no common WIS interval level")
        result["election_simulator"] = {
            "quantiles": _score_published_quantiles(
                election_simulator_quantiles,
                target,
                party_order=parties,
                interval_levels=common_levels,
            ),
            "point_forecast": _score_point_forecast(
                election_simulator_central_forecast,
                target,
                party_order=parties,
            ),
        }
        result["botten_ada"] = {
            "quantiles": _score_published_quantiles(
                botten_ada_quantiles,
                target,
                party_order=parties,
                interval_levels=common_levels,
            ),
            "point_forecast": _score_point_forecast(
                botten_ada_central_forecast,
                target,
                party_order=parties,
            ),
        }
        result["wis_common_interval_levels"] = [float(level) for level in common_levels]
        return result

    result["election_simulator"] = {
        "point_forecast": _score_point_forecast(
            election_simulator_central_forecast,
            target,
            party_order=parties,
        )
    }
    result["botten_ada"] = {
        "point_forecast": _score_point_forecast(
            botten_ada_central_forecast,
            target,
            party_order=parties,
        )
    }
    if (
        result["election_simulator"]["point_forecast"]["mean_mae"] is None
        or result["botten_ada"]["point_forecast"]["mean_mae"] is None
    ):
        result["status"] = "UNAVAILABLE_NO_COMMON_POINT_FORECASTS"
    return result


__all__ = [
    "DEFAULT_INTERVAL_LEVELS",
    "PRIMARY_PARTY_ORDER",
    "PROBABILISTIC_TIER_FAIR_DRAWS",
    "PROBABILISTIC_TIER_POINT_MAE",
    "PROBABILISTIC_TIER_WIS",
    "central_interval_metrics",
    "common_wis_interval_levels",
    "compatible_quantile_forecasts",
    "crps_v_statistic",
    "energy_score_v_statistic",
    "fair_crps",
    "fair_energy_score",
    "interval_coverage_width",
    "point_mae",
    "quantile_levels_compatible",
    "score_forecast_pair",
    "score_vote_ensemble",
    "select_primary_scoring_tier",
    "threshold_brier",
    "threshold_brier_from_probability",
    "threshold_probability",
    "weighted_interval_score",
]
