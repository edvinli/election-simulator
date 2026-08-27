"""Faithful reconstruction of the historical Pollofpolls simulation.

This module intentionally models only the simulation *after* a Poll of Polls
point estimate has been supplied.  It is not a poll aggregator.  The default
path is therefore directly comparable with Candidate A at a common stored PoP
origin while remaining an opt-in, separately versioned baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.pollofpolls.clr import clr_to_composition_matrix, composition_to_clr
from scripts.pollofpolls.state import load_timeseries_dataset
from scripts.pollofpolls.state_config import ALL_CATEGORIES, PARTIES, REFERENCE_CATEGORY
from scripts.pollofpolls.transitions import (
    HistoricalTransition,
    build_all_historical_transitions,
    filter_transitions_as_of,
)

from .config import (
    BASELINE_VERSION,
    DEFAULT_CONFIG,
    DEFAULT_STEP_WINDOWS,
    LEFT_BLOCK,
    MIN_TRANSITIONS,
    MIN_SHARE_PCT,
    MODEL_ID,
    PARTY_ORDER,
    RIGHT_BLOCK,
    PoPBaselineConfig,
)


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "pollofpolls"


@dataclass(frozen=True)
class BaselineForecast:
    """A reproducible baseline forecast and its pre/post support-vote draws."""

    model_id: str
    model_version: str
    origin_date: date
    horizon_days: int
    samples_count: int
    seed: int
    party_order: tuple[str, ...]
    samples_matrix: np.ndarray
    raw_samples_matrix: np.ndarray
    diagnostics: dict[str, Any]

    @property
    def samples_by_party(self) -> dict[str, np.ndarray]:
        """Return party columns as arrays, preserving the declared order."""
        return {party: self.samples_matrix[:, i] for i, party in enumerate(self.party_order)}

    @property
    def raw_samples_by_party(self) -> dict[str, np.ndarray]:
        """Return pre-support-vote party columns for diagnostic comparisons."""
        return {party: self.raw_samples_matrix[:, i] for i, party in enumerate(self.party_order)}


def derive_baseline_seed(base_seed: int, origin_date: date, horizon_days: int, label: str) -> int:
    """Derive an independent deterministic NumPy seed for a baseline layer."""
    token = f"{base_seed}:{origin_date.isoformat()}:{horizon_days}:{label}".encode("utf-8")
    digest = hashlib.sha256(token).hexdigest()
    return int(digest[:8], 16) % 2_147_483_647


def _validate_origin_composition(origin_pop: Mapping[str, float]) -> np.ndarray:
    """Validate and convert a stored PoP composition to a percentage vector."""
    missing = set(PARTY_ORDER) - set(origin_pop)
    if missing:
        raise ValueError(f"origin_pop is missing categories: {sorted(missing)}")
    values = np.asarray([float(origin_pop[p]) for p in PARTY_ORDER], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("origin_pop contains non-finite values")
    if np.any(values < 0.0):
        raise ValueError("origin_pop cannot contain negative shares")
    if not math.isclose(float(np.sum(values)), 100.0, abs_tol=1e-5):
        raise ValueError(f"origin_pop must sum to 100%, got {float(np.sum(values))}")
    if np.any(values < MIN_SHARE_PCT):
        # The canonical CLR implementation floors tiny values.  Applying the
        # same floor here makes the baseline's handling explicit and stable.
        values = np.maximum(values, MIN_SHARE_PCT)
        values *= 100.0 / float(np.sum(values))
    return values


def load_stored_origin_pop(
    origin_date: str | date,
    *,
    data_dir: Path | str | None = None,
) -> dict[str, float]:
    """Load the exact stored PoP point estimate for an origin date.

    A baseline forecast must not silently substitute a nearby date: callers
    receive an error when the requested date is absent.  This keeps matched
    benchmarks aligned with Candidate A's information cutoff.
    """
    d = date.fromisoformat(origin_date) if isinstance(origin_date, str) else origin_date
    base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    rows = load_timeseries_dataset(base / "pollofpolls_timeseries.csv")
    for row in rows:
        if row["date"] == d:
            return {party: float(row["composition"][party]) for party in PARTY_ORDER}
    raise KeyError(f"No exact stored Poll of Polls point estimate for {d.isoformat()}")


def _sample_window_paths(
    origin_clr: np.ndarray,
    transitions: Sequence[HistoricalTransition],
    *,
    horizon_days: int,
    samples_count: int,
    seed: int,
    random_sign: bool = True,
) -> np.ndarray:
    """Draw random-walk paths from one historical step window.

    For exact horizons not divisible by a historical window, the final step is
    multiplied by its elapsed-day fraction in CLR space.  This is an explicit
    approximation required by the benchmark's exact 7/14/28/56/84/112-day
    targets; the source material leaves this edge case unspecified.
    """
    if samples_count <= 0:
        raise ValueError("samples_count must be positive")
    if horizon_days < 0:
        raise ValueError("horizon_days cannot be negative")
    if not transitions:
        raise ValueError("Cannot simulate a step window without eligible transitions")

    window_days = int(transitions[0].horizon_days)
    if any(int(t.horizon_days) != window_days for t in transitions):
        raise ValueError("All transitions in one baseline window must have the same horizon")

    rng = np.random.default_rng(seed)
    current = np.repeat(np.asarray(origin_clr, dtype=np.float64)[None, :], samples_count, axis=0)
    remaining = int(horizon_days)
    step_count = 0
    while remaining > 0:
        elapsed = min(remaining, window_days)
        fraction = float(elapsed / window_days)
        indices = rng.integers(0, len(transitions), size=samples_count)
        delta_matrix = np.asarray([t.clr_transition for t in transitions], dtype=np.float64)
        deltas = delta_matrix[indices]
        signs = (
            rng.choice(np.asarray([-1.0, 1.0]), size=samples_count)
            if random_sign
            else np.ones(samples_count, dtype=np.float64)
        )
        current += signs[:, None] * deltas * fraction
        remaining -= elapsed
        step_count += 1

    # Keep the diagnostic useful without changing the returned deterministic
    # array.  The caller records this per-window separately.
    return current


def _block_for_target(target: str) -> tuple[str, ...] | None:
    if target in RIGHT_BLOCK:
        return RIGHT_BLOCK
    if target in LEFT_BLOCK:
        return LEFT_BLOCK
    return None


def _apply_signed_support_transfer(
    matrix: np.ndarray,
    *,
    target: str,
    requested_amount: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Apply one formula-derived support-vote transfer while preserving 100%.

    Positive transfers draw proportionally from same-block parties' support
    above 4%, as described in the first-party 2018 methodology.  A negative
    normal draw is handled as the reverse transfer rather than allowing a
    negative share or silently changing the total.
    """
    target_idx = PARTY_ORDER.index(target)
    block = _block_for_target(target)
    if block is None:
        return requested_amount * 0.0, 0

    donor_indices = [PARTY_ORDER.index(p) for p in block if p != target]
    if not donor_indices:
        return requested_amount * 0.0, 0

    before = matrix[:, target_idx].copy()
    positive = np.maximum(requested_amount, 0.0)
    donor_excess = np.maximum(matrix[:, donor_indices] - 4.0, 0.0)
    donor_total = np.sum(donor_excess, axis=1)
    positive_applied = np.minimum(positive, donor_total)
    positive_mask = positive_applied > 0.0
    if np.any(positive_mask):
        weights = np.divide(
            donor_excess,
            donor_total[:, None],
            out=np.zeros_like(donor_excess),
            where=donor_total[:, None] > 0.0,
        )
        matrix[:, donor_indices] -= positive_applied[:, None] * weights
        matrix[:, target_idx] += positive_applied

    negative = np.minimum(requested_amount, 0.0)
    target_available = np.maximum(matrix[:, target_idx] - MIN_SHARE_PCT, 0.0)
    negative_applied = -np.minimum(-negative, target_available)
    negative_mask = negative_applied < 0.0
    if np.any(negative_mask):
        # Use the same above-threshold donor weighting where possible.  If all
        # donors are below 4%, use their current positive shares instead.
        donor_weights = np.maximum(matrix[:, donor_indices] - 4.0, 0.0)
        donor_weight_totals = np.sum(donor_weights, axis=1)
        fallback_weights = np.maximum(matrix[:, donor_indices], 0.0)
        fallback_totals = np.sum(fallback_weights, axis=1)
        use_fallback = donor_weight_totals <= 0.0
        donor_weights = np.where(use_fallback[:, None], fallback_weights, donor_weights)
        donor_weight_totals = np.where(use_fallback, fallback_totals, donor_weight_totals)
        weights = np.divide(
            donor_weights,
            donor_weight_totals[:, None],
            out=np.zeros_like(donor_weights),
            where=donor_weight_totals[:, None] > 0.0,
        )
        matrix[:, donor_indices] -= negative_applied[:, None] * weights
        matrix[:, target_idx] += negative_applied

    # Floating arithmetic is only used for percentages; enforce the exact
    # simplex after each target so later target transfers see a valid state.
    matrix[:] *= 100.0 / np.sum(matrix, axis=1, keepdims=True)
    applied = matrix[:, target_idx] - before
    return applied, int(np.count_nonzero(np.abs(applied) > 1e-12))


def apply_support_voting(
    raw_samples_matrix: np.ndarray,
    *,
    seed: int,
    targets: Sequence[str] = DEFAULT_CONFIG.support_voting_targets,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the documented first-party support-vote formula to draws.

    The canonical dataset folds FI and other non-parliamentary parties into
    ``REST``.  FI therefore cannot be modelled as a separate recipient; the
    default target list intentionally contains only representable parties.
    """
    matrix = np.asarray(raw_samples_matrix, dtype=np.float64).copy()
    if matrix.ndim != 2 or matrix.shape[1] != len(PARTY_ORDER):
        raise ValueError(f"Expected an (N, {len(PARTY_ORDER)}) sample matrix")
    if matrix.shape[0] == 0:
        raise ValueError("raw_samples_matrix cannot be empty")
    if any(target not in PARTY_ORDER for target in targets):
        raise ValueError("support-voting targets must be canonical party categories")

    rng = np.random.default_rng(seed)
    active_counts: dict[str, int] = {}
    applied_counts: dict[str, int] = {}
    requested_by_target: dict[str, float] = {}
    for target in targets:
        idx = PARTY_ORDER.index(target)
        support = matrix[:, idx].copy()
        active = (support > 2.0) & (support < 5.0)
        # s_support - s_sim = (-0.6 * |s_sim - 4| + 1.2) * X,
        # X ~ N(1, 0.7), exactly as transcribed by the first-party source.
        x = rng.normal(loc=1.0, scale=0.7, size=matrix.shape[0])
        coefficient = -0.6 * np.abs(support - 4.0) + 1.2
        requested = np.where(active, coefficient * x, 0.0)
        _, applied_count = _apply_signed_support_transfer(
            matrix,
            target=target,
            requested_amount=requested,
        )
        active_counts[target] = int(np.count_nonzero(active))
        applied_counts[target] = applied_count
        requested_by_target[target] = float(np.mean(requested))

    # The transfer operations preserve the simplex to floating precision.
    matrix[:] *= 100.0 / np.sum(matrix, axis=1, keepdims=True)
    return matrix, {
        "targets": list(targets),
        "active_draws_by_target": active_counts,
        "applied_draws_by_target": applied_counts,
        "mean_requested_transfer_pp": requested_by_target,
        "transfer_application": "sequential_in_declared_target_order",
        "normal_mean": 1.0,
        "normal_sd": 0.7,
        "formula_domain": "2 < s_sim < 5",
        "donor_policy": "same_block shares above 4%; reverse transfer for negative normal draw",
    }


def simulate_baseline(
    *,
    origin_date: str | date,
    horizon_days: int,
    samples_count: int = 5_000,
    seed: int = 12345,
    origin_pop: Mapping[str, float] | None = None,
    data_dir: Path | str | None = None,
    config: PoPBaselineConfig = DEFAULT_CONFIG,
    _timeseries_data: Sequence[dict[str, Any]] | None = None,
    _transitions_by_window: Mapping[int, Sequence[HistoricalTransition]] | None = None,
) -> BaselineForecast:
    """Generate deterministic PoPBaseline v1 draws from a stored origin.

    ``origin_pop`` is optional only for convenience; when omitted it is loaded
    from the exact requested date in the processed PoP series.  No estimator,
    current-state uncertainty, election residual, geographic projection, or
    seat allocation is applied here.
    """
    d = date.fromisoformat(origin_date) if isinstance(origin_date, str) else origin_date
    if not isinstance(d, date):
        raise TypeError("origin_date must be an ISO date string or datetime.date")
    if int(horizon_days) < 0:
        raise ValueError("horizon_days cannot be negative")
    if int(samples_count) <= 0:
        raise ValueError("samples_count must be positive")
    if origin_pop is None:
        origin_pop = load_stored_origin_pop(d, data_dir=data_dir)
    origin_values = _validate_origin_composition(origin_pop)
    origin_clr, origin_was_floored = composition_to_clr(
        {party: float(origin_values[i]) for i, party in enumerate(PARTY_ORDER)},
        categories=PARTY_ORDER,
        min_share_pct=MIN_SHARE_PCT,
    )

    base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    ts_rows = list(_timeseries_data) if _timeseries_data is not None else load_timeseries_dataset(base / "pollofpolls_timeseries.csv")
    all_transitions = (
        dict(_transitions_by_window)
        if _transitions_by_window is not None
        else build_all_historical_transitions(ts_rows, horizons=config.step_windows)
    )

    # Match the source's equal-sized batches by splitting N across windows.
    counts = [samples_count // len(config.step_windows)] * len(config.step_windows)
    for i in range(samples_count % len(config.step_windows)):
        counts[i] += 1

    raw_parts: list[np.ndarray] = []
    window_diagnostics: list[dict[str, Any]] = []
    for window, count in zip(config.step_windows, counts):
        eligible = filter_transitions_as_of(all_transitions[int(window)], d)
        if len(eligible) < MIN_TRANSITIONS:
            raise ValueError(
                f"Insufficient historical transitions for {window}d at {d.isoformat()}: "
                f"{len(eligible)} < {MIN_TRANSITIONS}"
            )
        window_seed = derive_baseline_seed(seed, d, int(horizon_days), f"window-{window}")
        raw_parts.append(
            _sample_window_paths(
                origin_clr,
                eligible,
                horizon_days=int(horizon_days),
                samples_count=count,
                seed=window_seed,
                random_sign=config.random_sign,
            )
        )
        window_diagnostics.append({
            "window_days": int(window),
            "draw_count": count,
            "eligible_transition_count": len(eligible),
            "earliest_transition_end": min(t.end_date for t in eligible).isoformat(),
            "latest_transition_end": max(t.end_date for t in eligible).isoformat(),
            "sampling_seed": window_seed,
            "partial_step_policy": config.partial_step_policy,
        })

    raw_clr = np.vstack(raw_parts)
    raw_matrix = clr_to_composition_matrix(raw_clr)

    support_diag: dict[str, Any]
    final_matrix = raw_matrix
    if config.apply_support_voting:
        support_seed = derive_baseline_seed(seed, d, int(horizon_days), "support-voting")
        final_matrix, support_diag = apply_support_voting(
            raw_matrix,
            seed=support_seed,
            targets=config.support_voting_targets,
        )
        support_diag["seed"] = support_seed
    else:
        support_diag = {
            "enabled": False,
            "targets": list(config.support_voting_targets),
        }

    if not np.all(np.isfinite(final_matrix)):
        raise RuntimeError("PoPBaseline generated non-finite draws")
    if not np.allclose(np.sum(final_matrix, axis=1), 100.0, atol=1e-10):
        raise RuntimeError("PoPBaseline composition invariant violated")
    if np.any(final_matrix < 0.0):
        raise RuntimeError("PoPBaseline generated negative shares")

    diagnostics = {
        "baseline_version": BASELINE_VERSION,
        "origin_pop_source": "stored_processed_pollofpolls_timeseries_exact_date",
        "origin_was_floored": bool(origin_was_floored),
        "step_windows": list(config.step_windows),
        "random_sign": bool(config.random_sign),
        "compositional_space": config.compositional_space,
        "partial_step_policy": config.partial_step_policy,
        "support_voting": support_diag,
        "window_diagnostics": window_diagnostics,
        "current_state_uncertainty": "none",
        "election_residual": "none",
        "geography": "none",
        "seat_allocation": "none",
    }
    return BaselineForecast(
        model_id=MODEL_ID,
        model_version=BASELINE_VERSION,
        origin_date=d,
        horizon_days=int(horizon_days),
        samples_count=int(samples_count),
        seed=int(seed),
        party_order=PARTY_ORDER,
        samples_matrix=final_matrix,
        raw_samples_matrix=raw_matrix,
        diagnostics=diagnostics,
    )
