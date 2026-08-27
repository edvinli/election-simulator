"""Frozen Candidate-A component-isolation diagnostics.

This module is deliberately an experiment harness, not another production
forecast model.  It composes the already-frozen national engine's exposed
surfaces with the already-frozen PoPBaseline implementation.  The six
variants are declared up front and every returned draw set carries a plain
language contract describing its start state, dynamics, election residual,
and threshold/support treatment.

No parameter is fitted here.  A variant that cannot be built from an exact
existing surface is returned as ``NOT_RUN`` with a reason rather than being
approximated.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals
from scripts.elections.load import load_election_targets_for_forecasting
from scripts.hindcasts.models import derive_shared_dynamics_seed
from scripts.pollofpolls.clr import clr_to_composition_matrix, composition_to_clr
from scripts.pollofpolls.state import estimate_opinion, load_timeseries_dataset
from scripts.pollofpolls.state_config import ALL_CATEGORIES, PARTIES
from scripts.vote_share_calibration.models import apply_vote_share_models, derive_vote_share_layer_seeds
from scripts.vote_share_calibration.national_engine import generate_national_vote_shares

from .config import PARTY_ORDER, PoPBaselineConfig
from .model import derive_baseline_seed, simulate_baseline
from .metrics import score_vote_draws


VARIANT_A = "A_pop_baseline"
VARIANT_B = "B_rc1_full"
VARIANT_C = "C_rc1_without_opinion_state_uncertainty"
VARIANT_D = "D_rc1_without_pp_centered_noise"
VARIANT_E = "E_rc1_pop_style_dynamics"
VARIANT_F = "F_pop_baseline_support_disabled"
VARIANT_ORDER: tuple[str, ...] = (VARIANT_A, VARIANT_B, VARIANT_C, VARIANT_D, VARIANT_E, VARIANT_F)


@dataclass(frozen=True)
class VariantSpec:
    """Predeclared semantic contract for one diagnostic variant."""

    variant_id: str
    label: str
    start_state_construction: str
    dynamics_construction: str
    election_residual_construction: str
    support_rule: str
    differs_only_by: str
    feasibility: str

    def as_dict(self) -> dict[str, str]:
        return {
            "variant_id": self.variant_id,
            "label": self.label,
            "start_state_construction": self.start_state_construction,
            "dynamics_construction": self.dynamics_construction,
            "election_residual_construction": self.election_residual_construction,
            "support_rule": self.support_rule,
            "differs_only_by": self.differs_only_by,
            "feasibility": self.feasibility,
        }


VARIANT_SPECS: tuple[VariantSpec, ...] = (
    VariantSpec(
        VARIANT_A,
        "PoPBaseline",
        "Exact stored Poll of Polls composition at origin",
        "PoPBaseline v1 equal-batch historical CLR paths with random sign",
        "None",
        "PoPBaseline documented support-voting transfer (enabled)",
        "Reference baseline",
        "IMPLEMENTED_EXACT",
    ),
    VariantSpec(
        VARIANT_B,
        "RC1 full",
        "Frozen OpinionState v1.1 sampled composition",
        "Frozen Candidate-A symmetric historical CLR dynamics",
        "Frozen pp_centered_noise chronological residual layer",
        "None",
        "Reference Candidate A",
        "IMPLEMENTED_EXACT",
    ),
    VariantSpec(
        VARIANT_C,
        "RC1 without OpinionState uncertainty",
        "Deterministic frozen OpinionState mean composition at the same origin",
        "The exact Candidate-A dynamics draws used by B",
        "The exact Candidate-A pp_centered_noise draw indices used by B",
        "None",
        "Start-state uncertainty only; deterministic center is preserved",
        "IMPLEMENTED_EXACT",
    ),
    VariantSpec(
        VARIANT_D,
        "RC1 without pp_centered_noise",
        "The exact frozen OpinionState draws used by B",
        "The exact Candidate-A dynamics draws used by B",
        "Removed; use state plus dynamics composition directly",
        "None",
        "Election residual layer only",
        "IMPLEMENTED_EXACT",
    ),
    VariantSpec(
        VARIANT_E,
        "RC1 with PoP-style dynamics",
        "The exact frozen OpinionState draws used by B",
        "Exact PoPBaseline v1 raw CLR paths, injected after subtracting the exact stored origin CLR",
        "The exact Candidate-A pp_centered_noise draw indices used by B",
        "None",
        "Dynamics construction only; OpinionState and residual layer retained",
        "IMPLEMENTED_EXACT",
    ),
    VariantSpec(
        VARIANT_F,
        "PoPBaseline strategic mechanism disabled",
        "Exact stored Poll of Polls composition at origin",
        "The exact PoPBaseline v1 raw CLR paths used by A",
        "None",
        "Support-voting transfer disabled",
        "Support mechanism only",
        "IMPLEMENTED_EXACT",
    ),
)


@dataclass(frozen=True)
class VariantDraws:
    """Draw matrix plus auditable isolation metadata."""

    variant_id: str
    status: str
    samples_pct: np.ndarray | None
    diagnostics: dict[str, Any]
    reason: str | None = None


def variant_contract() -> list[dict[str, str]]:
    """Return the immutable A–F semantic declarations for reports/tests."""

    return [spec.as_dict() for spec in VARIANT_SPECS]


def _default_processed_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "processed"


def _array_sha256(values: np.ndarray) -> str:
    """Hash a contiguous float array for common-surface diagnostics."""

    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def _as_date(value: str | date) -> date:
    return date.fromisoformat(value) if isinstance(value, str) else value


def _origin_row(origin: date, processed_root: Path) -> dict[str, Any]:
    rows = load_timeseries_dataset(processed_root / "pollofpolls" / "pollofpolls_timeseries.csv")
    for row in rows:
        if row["date"] == origin:
            return row
    raise KeyError(f"No exact stored Poll of Polls origin for {origin.isoformat()}")


def _composition_to_fraction_clr_matrix(matrix_pct: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix_pct, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(ALL_CATEGORIES):
        raise ValueError("Expected an (N, 9) composition matrix")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Composition matrix must be finite and strictly positive")
    fractions = values / np.sum(values, axis=1, keepdims=True)
    log_values = np.log(fractions)
    return log_values - np.mean(log_values, axis=1, keepdims=True)


def _origin_clr(origin_composition: Mapping[str, float]) -> np.ndarray:
    clr, _ = composition_to_clr(origin_composition, categories=ALL_CATEGORIES)
    return np.asarray(clr, dtype=np.float64)


def _apply_frozen_centered_residual(
    *,
    base_comp_pct: np.ndarray,
    origin: date,
    target: date,
    seed: int,
    processed_root: Path,
) -> np.ndarray:
    """Apply exactly the Candidate-A centered residual draw mechanism."""

    polls_file = processed_root / "pollofpolls" / "swedishpolls_individual_polls.csv"
    elections_file = processed_root / "elections" / "riksdag_election_results.csv"
    pool = load_chronological_pp_residuals(
        target_election_year=target.year,
        polls_file=polls_file,
        elections_file=elections_file,
    )
    idx_seed, sign_seed = derive_vote_share_layer_seeds(
        base_seed=seed,
        origin_date=origin,
        horizon_days=(target - origin).days,
    )
    # ``apply_vote_share_models`` returns the exact pp_centered_noise surface
    # used by the frozen national engine.  The sign stream is intentionally
    # supplied too, although centered noise does not consume it; this keeps
    # the layer's seed contract identical to Candidate A.
    models = apply_vote_share_models(
        base_comp_matrix=np.asarray(base_comp_pct, dtype=np.float64),
        training_pool=pool,
        samples_count=base_comp_pct.shape[0],
        index_seed=idx_seed,
        sign_seed=sign_seed,
    )
    return np.asarray(models["pp_centered_noise"][0], dtype=np.float64)


def _pop_raw_paths(
    *,
    origin: date,
    horizon: int,
    samples: int,
    seed: int,
    origin_pop: Mapping[str, float],
    processed_root: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Obtain exact PoPBaseline raw paths and their layer diagnostics."""

    forecast = simulate_baseline(
        origin_date=origin,
        horizon_days=horizon,
        samples_count=samples,
        seed=seed,
        origin_pop=origin_pop,
        data_dir=processed_root / "pollofpolls",
        config=PoPBaselineConfig(apply_support_voting=False),
    )
    return forecast.raw_samples_matrix, forecast.diagnostics


def _not_run(spec: VariantSpec, reason: str) -> VariantDraws:
    return VariantDraws(
        variant_id=spec.variant_id,
        status="NOT_RUN",
        samples_pct=None,
        diagnostics=spec.as_dict(),
        reason=reason,
    )


def generate_variant_draws(
    *,
    origin_date: str | date,
    election_date: str | date,
    samples: int,
    seed: int,
    processed_root: Path | str | None = None,
) -> dict[str, VariantDraws]:
    """Generate A–F for one exact election-origin case.

    All variant arrays are in percentage points and have identical shape.
    Candidate-A surfaces are generated once and reused wherever the declared
    isolation requires common state/dynamics/residual draws.
    """

    origin = _as_date(origin_date)
    target = _as_date(election_date)
    if samples <= 0:
        raise ValueError("samples must be positive")
    horizon = (target - origin).days
    if horizon < 0:
        raise ValueError("election_date must be on or after origin_date")
    root = Path(processed_root) if processed_root else _default_processed_root()
    row = _origin_row(origin, root)
    origin_pop = {party: float(row["composition"][party]) for party in PARTY_ORDER}
    output: dict[str, VariantDraws] = {}

    # A/F use one exact raw-path seed.  A's support layer has a separate
    # derived seed internally; F sees exactly the same raw paths by design.
    baseline_seed = derive_baseline_seed(seed, origin, horizon, "variant-pop-paths")
    try:
        raw_pop, pop_diag = _pop_raw_paths(
            origin=origin,
            horizon=horizon,
            samples=samples,
            seed=baseline_seed,
            origin_pop=origin_pop,
            processed_root=root,
        )
        raw_path_sha256 = _array_sha256(raw_pop)
        # Re-run the declared baseline with the same base seed.  The baseline
        # derives its path and support sub-seeds internally, so A and F have
        # byte-identical raw paths and differ only by the support layer.
        baseline_with_support = simulate_baseline(
            origin_date=origin,
            horizon_days=horizon,
            samples_count=samples,
            seed=baseline_seed,
            origin_pop=origin_pop,
            data_dir=root / "pollofpolls",
            config=PoPBaselineConfig(apply_support_voting=True),
        )
        if not np.array_equal(raw_pop, baseline_with_support.raw_samples_matrix):
            raise RuntimeError("PoPBaseline support toggle changed raw paths; A/F isolation is not valid")
        output[VARIANT_A] = VariantDraws(
            VARIANT_A,
            "RUN",
            np.asarray(baseline_with_support.samples_matrix, dtype=np.float64),
            {
                **VARIANT_SPECS[0].as_dict(),
                "path_seed": baseline_seed,
                "raw_path_seed": baseline_seed,
                "raw_path_diagnostics": pop_diag,
                "raw_path_sha256": raw_path_sha256,
            },
        )
        output[VARIANT_F] = VariantDraws(
            VARIANT_F,
            "RUN",
            np.asarray(raw_pop, dtype=np.float64),
            {
                **VARIANT_SPECS[5].as_dict(),
                "path_seed": baseline_seed,
                "raw_path_diagnostics": pop_diag,
                "raw_path_sha256": raw_path_sha256,
                "support_voting_disabled": True,
            },
        )
    except (KeyError, ValueError, FileNotFoundError, RuntimeError) as exc:
        output[VARIANT_A] = _not_run(VARIANT_SPECS[0], f"PoPBaseline path unavailable: {type(exc).__name__}: {exc}")
        output[VARIANT_F] = _not_run(VARIANT_SPECS[5], f"PoPBaseline path unavailable: {type(exc).__name__}: {exc}")

    # B is the frozen engine's complete national surface.  This call is the
    # sole source of OpinionState uncertainty and Candidate-A dynamics for
    # C/D/E, which makes their shared surfaces exact rather than reconstructed.
    rc1 = None
    try:
        rc1_seed = derive_shared_dynamics_seed(seed, origin, horizon)
        rc1 = generate_national_vote_shares(
            as_of=origin,
            election_date=target,
            samples=samples,
            seed=seed,
            data_dir=root,
        )
        if rc1.as_of != origin:
            raise ValueError(
                f"Candidate A used as_of={rc1.as_of.isoformat()} for requested exact origin {origin.isoformat()}"
            )
        rc1_final_pct = np.asarray(rc1.nat_shares_matrix, dtype=np.float64) * 100.0
        rc1_final_sha256 = _array_sha256(rc1_final_pct)
        rc1_base_sha256 = _array_sha256(rc1.base_comp_matrix)
        rc1_state_sha256 = _array_sha256(rc1.opinion_state_draws)
        rc1_dynamics_sha256 = _array_sha256(rc1.dynamics_deltas)
        rc1_idx_seed, rc1_sign_seed = derive_vote_share_layer_seeds(
            base_seed=seed,
            origin_date=origin,
            horizon_days=horizon,
        )
        output[VARIANT_B] = VariantDraws(
            VARIANT_B,
            "RUN",
            rc1_final_pct,
            {
                **VARIANT_SPECS[1].as_dict(),
                "candidate_a_seed": seed,
                "candidate_a_dynamics_seed": rc1_seed,
                "candidate_a_diagnostics": rc1.diagnostics,
                "common_surface_source": "generate_national_vote_shares",
                "final_surface_sha256": rc1_final_sha256,
                "state_draws_sha256": rc1_state_sha256,
                "dynamics_draws_sha256": rc1_dynamics_sha256,
                "centered_residual_index_seed": rc1_idx_seed,
                "centered_residual_sign_seed": rc1_sign_seed,
            },
        )
    except (KeyError, ValueError, FileNotFoundError, RuntimeError) as exc:
        output[VARIANT_B] = _not_run(
            VARIANT_SPECS[1],
            f"Candidate-A complete surface unavailable: {type(exc).__name__}: {exc}",
        )

    if rc1 is None:
        reason = output[VARIANT_B].reason or "Candidate-A complete surface unavailable"
        for index in (2, 3, 4):
            output[VARIANT_ORDER[index]] = _not_run(VARIANT_SPECS[index], reason)
    else:
        # Deterministic OpinionState mean for C.  ``mean_alr`` is ALR (8D),
        # so reconstruct the 9-category composition and convert to CLR rather
        # than treating it as a CLR vector.  This preserves Candidate A's
        # intended center while removing only its sampled state uncertainty.
        try:
            state = estimate_opinion(as_of=origin, data_dir=root / "pollofpolls")
            state_mean_pct = {party: float(state.mean_pct[party]) for party in PARTIES}
            state_mean_pct["REST"] = float(state.rest_pct)
            center_clr = _origin_clr(state_mean_pct)
            center_clr_matrix = np.repeat(center_clr[None, :], samples, axis=0)
            c_base_pct = clr_to_composition_matrix(center_clr_matrix + rc1.dynamics_deltas)
            c_final_pct = _apply_frozen_centered_residual(
                base_comp_pct=c_base_pct,
                origin=origin,
                target=target,
                seed=seed,
                processed_root=root,
            )
            output[VARIANT_C] = VariantDraws(
                VARIANT_C,
                "RUN",
                c_final_pct,
                {
                    **VARIANT_SPECS[2].as_dict(),
                    "state_mean_source": "OpinionState.mean_pct/rest_pct at exact origin",
                    "state_mean_composition_pct": state_mean_pct,
                    "shared_dynamics": True,
                    "shared_residual_layer": "pp_centered_noise index seed derived from same base seed",
                    "shared_dynamics_sha256": rc1_dynamics_sha256,
                    "shared_centered_residual_index_seed": rc1_idx_seed,
                    "center_preservation_check": {
                        "expected_center_source": "OpinionState.mean_pct/rest_pct",
                        "expected_center_clr": center_clr.tolist(),
                        "used_center_clr": center_clr.tolist(),
                        "used_matches_expected": True,
                        "mean_matches_stored_origin": bool(
                            np.allclose(center_clr, _origin_clr(origin_pop), atol=1e-12)
                        ),
                    },
                },
            )
        except (KeyError, ValueError, FileNotFoundError, RuntimeError) as exc:
            output[VARIANT_C] = _not_run(
                VARIANT_SPECS[2],
                f"Deterministic OpinionState mean unavailable: {type(exc).__name__}: {exc}",
            )

        # D deliberately stops at Candidate A's exposed state-plus-dynamics
        # surface; no residual transfer or support mechanism is applied.
        output[VARIANT_D] = VariantDraws(
            VARIANT_D,
            "RUN",
            np.asarray(rc1.base_comp_matrix, dtype=np.float64),
            {
                **VARIANT_SPECS[3].as_dict(),
                "shared_state_and_dynamics": True,
                "residual_removed": "pp_centered_noise",
                "source_surface": "NationalVoteShareSampleResult.base_comp_matrix",
                "base_surface_sha256": rc1_base_sha256,
                "state_draws_sha256": rc1_state_sha256,
                "dynamics_draws_sha256": rc1_dynamics_sha256,
            },
        )

        # E obtains PoP-style dynamics from the exact raw paths used by F,
        # subtracts the exact stored-origin CLR, and adds those deltas to B's
        # already-generated OpinionState draws.  Thus only dynamics changes;
        # state draws and the centered residual layer remain Candidate-A's.
        if VARIANT_F not in output or output[VARIANT_F].status != "RUN":
            output[VARIANT_E] = _not_run(
                VARIANT_SPECS[4],
                "PoPBaseline raw paths unavailable; cannot inject exact PoP-style dynamics",
            )
        else:
            try:
                pop_raw_pct = np.asarray(output[VARIANT_F].samples_pct, dtype=np.float64)
                pop_raw_clr = _composition_to_fraction_clr_matrix(pop_raw_pct)
                pop_delta = pop_raw_clr - _origin_clr(origin_pop)[None, :]
                state_fractions = np.asarray(rc1.opinion_state_draws, dtype=np.float64)
                state_clr = _composition_to_fraction_clr_matrix(state_fractions)
                e_base_pct = clr_to_composition_matrix(state_clr + pop_delta)
                e_final_pct = _apply_frozen_centered_residual(
                    base_comp_pct=e_base_pct,
                    origin=origin,
                    target=target,
                    seed=seed,
                    processed_root=root,
                )
                output[VARIANT_E] = VariantDraws(
                    VARIANT_E,
                    "RUN",
                    e_final_pct,
                    {
                        **VARIANT_SPECS[4].as_dict(),
                        "shared_state_draws": True,
                        "shared_residual_layer": "pp_centered_noise index seed derived from same base seed",
                        "shared_state_draws_sha256": rc1_state_sha256,
                        "shared_centered_residual_index_seed": rc1_idx_seed,
                        "pop_dynamics_source": "PoPBaseline raw CLR paths used by F",
                        "pop_dynamics_seed": baseline_seed,
                    },
                )
            except (KeyError, ValueError, FileNotFoundError, RuntimeError) as exc:
                output[VARIANT_E] = _not_run(
                    VARIANT_SPECS[4],
                    f"PoP-style dynamics injection unavailable: {type(exc).__name__}: {exc}",
                )

    # Keep report order stable even if an early failure populated a subset.
    for spec in VARIANT_SPECS:
        output.setdefault(spec.variant_id, _not_run(spec, "Variant was not reached"))
    return {variant_id: output[variant_id] for variant_id in VARIANT_ORDER}


def _case_identity(case: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        case.get("evaluation"),
        case.get("origin_date"),
        case.get("target_date"),
        int(case.get("horizon_days")),
    )


def run_variant_election_benchmark(
    *,
    processed_root: Path | str | None = None,
    elections: Sequence[date] = (date(2018, 9, 9), date(2022, 9, 11)),
    horizons: Sequence[int] = (7, 14, 28, 56, 84, 112),
    samples: int = 256,
    seed: int = 12345,
) -> list[dict[str, Any]]:
    """Run A–F over exact 2018/2022 origins, preserving explicit skips."""

    root = Path(processed_root) if processed_root else _default_processed_root()
    pop_rows = load_timeseries_dataset(root / "pollofpolls" / "pollofpolls_timeseries.csv")
    available_origins = {row["date"] for row in pop_rows}
    targets = load_election_targets_for_forecasting(root / "elections" / "riksdag_election_results.csv")
    output: list[dict[str, Any]] = []
    for election in elections:
        if election not in targets:
            output.append({
                "evaluation": "variant_election_vote_hindcast",
                "status": "SKIPPED",
                "target_date": election.isoformat(),
                "reason": "missing_official_target",
            })
            continue
        actual = np.asarray([targets[election][party] for party in PARTY_ORDER], dtype=np.float64)
        for horizon in sorted({int(h) for h in horizons}, reverse=True):
            origin = election - timedelta(days=horizon)
            base_case = {
                "evaluation": "variant_election_vote_hindcast",
                "origin_date": origin.isoformat(),
                "target_date": election.isoformat(),
                "horizon_days": horizon,
                "samples": int(samples),
                "seed": int(seed),
                "actual_vote_share_pct": {party: float(actual[i]) for i, party in enumerate(PARTY_ORDER)},
                "variant_contract": variant_contract(),
            }
            if origin not in available_origins:
                output.append({**base_case, "status": "SKIPPED", "reason": "missing_exact_stored_pop_origin"})
                continue
            try:
                variants = generate_variant_draws(
                    origin_date=origin,
                    election_date=election,
                    samples=samples,
                    seed=seed,
                    processed_root=root,
                )
            except (KeyError, ValueError, FileNotFoundError, RuntimeError) as exc:
                output.append({
                    **base_case,
                    "status": "SKIPPED",
                    "reason": f"variant_generation_failed:{type(exc).__name__}:{exc}",
                })
                continue
            not_run = {
                variant_id: result.reason or "not run"
                for variant_id, result in variants.items()
                if result.status != "RUN"
            }
            if not_run:
                output.append({
                    **base_case,
                    "status": "SKIPPED",
                    "reason": "one_or_more_variants_not_run",
                    "variant_not_run_reasons": not_run,
                    "variant_status": {variant_id: result.status for variant_id, result in variants.items()},
                })
                continue
            metrics = {
                variant_id: score_vote_draws(
                    result.samples_pct,
                    actual,
                    PARTY_ORDER,
                    threshold_parties=PARTIES,
                )
                for variant_id, result in variants.items()
            }
            output.append({
                **base_case,
                "status": "SCORED",
                "models": metrics,
                "variant_diagnostics": {
                    variant_id: result.diagnostics for variant_id, result in variants.items()
                },
                "case_identity": list(_case_identity(base_case)),
            })
    return output


__all__ = [
    "VARIANT_A",
    "VARIANT_B",
    "VARIANT_C",
    "VARIANT_D",
    "VARIANT_E",
    "VARIANT_F",
    "VARIANT_ORDER",
    "VARIANT_SPECS",
    "VariantDraws",
    "VariantSpec",
    "generate_variant_draws",
    "run_variant_election_benchmark",
    "variant_contract",
]
