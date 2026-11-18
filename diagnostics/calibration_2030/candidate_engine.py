"""The T4 candidate path: production's simulator with a pluggable national stage.

``docs/calibration_2030_final_round_preregistration.md`` §5 and §6.4 ask for
a path that runs ``scripts.simulator.engine.simulate_election`` end to end,
but lets three layers of the national stage be swapped:

(a) **the aggregate data root.** Production's v0.2 files, or a v0.3 file
    (``v03.py``), or a 2010 substitute file (§6.3). The engine and
    OpinionState read whatever ``opinion_inputs(data_root)`` resolves.
(b) **the state + dynamics provider.** Production (OpinionState and the
    capped symmetric bootstrap, exactly as ``generate_national_vote_shares``
    computes them) or a T2 U arm (``t2_arms.state_dynamics_draws``).
(c) **the noise law and its pool.** Production's own ``pp_lw_gaussian`` code,
    or a ``noise_laws`` law fitted on a pool of the elections strictly before
    the target. Every law gets production's standard normals ``z``.

Nothing downstream of the national stage is reimplemented. The candidate
national stage is installed in place of the engine's
``generate_national_vote_shares`` for the duration of one call, and the
engine then runs its own geography projection, integerization, mandate
allocation, seat checks and summary on it. No production file is modified.

``t2_arms`` and ``noise_laws`` are other agents' modules. They are imported
lazily, so this module imports, and its tests run, without them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import os
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np

# The engine first: importing national_engine on its own hits a circular
# import through scripts.simulator.__init__.
import scripts.simulator.engine  # noqa: F401  (import order)
from scripts.election_layer_v2.transfer import apply_batch_simplex_transfer
from scripts.vote_share_calibration.config import MIN_SHARE_PCT
from scripts.vote_share_calibration.election_noise_b import derive_election_noise_b_seed
from scripts.vote_share_calibration.national_engine import (
    NationalVoteShareSampleResult,
    generate_national_vote_shares,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = ROOT / "data" / "processed"
N_CATEGORIES = 9
N_PARTIES = 8
ELECTION_YEARS = (2002, 2006, 2010, 2014, 2018, 2022, 2026)
T2_ARMS = ("D0", "Ua", "Ub1", "Ub2", "Uc")
U_ARMS = ("Ua", "Ub1", "Ub2", "Uc")
NOISE_LAWS = ("LW0", "N0", "NH0", "NH0.5", "NH1")
NEW_LAYERS = ("state", "dynamics", "bootstrap", "ou")
SIMPLEX_FLOOR_PP = 0.01

NOISE_LAWS_MODULE = "diagnostics.calibration_2030.noise_laws"
T2_ARMS_MODULE = "diagnostics.calibration_2030.t2_arms"


# --------------------------------------------------------------------------- #
# Lazy imports of the parallel agents' modules
# --------------------------------------------------------------------------- #


def _optional_module(name: str):
    """The module if it exists; ``None`` if it (and only it) is missing."""

    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name == name:
            return None
        raise


def noise_laws_module():
    return _optional_module(NOISE_LAWS_MODULE)


def t2_arms_module():
    module = _optional_module(T2_ARMS_MODULE)
    if module is None:
        raise RuntimeError(f"{T2_ARMS_MODULE} is required for a T2 U arm but is not present")
    return module


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class CandidateConfig:
    """One arm of the candidate path.

    ``state_dynamics``: ``"production"`` or a T2 U arm (``Ua`` … ``Uc``).
    ``noise``: ``"production"`` (the engine's own pp_lw_gaussian code and pool)
    or a ``noise_laws`` law name. ``pool``: ``"production"``,
    ``"current_rule"`` or ``"corrected"``. ``aggregate``: which opinion file the
    harness must hand the arm (``"v0.2"`` or ``"v0.3"``); the path itself is
    passed per case. ``simplex_map``: apply the §3.4 map to a production base.
    """

    name: str
    aggregate: str = "v0.2"
    state_dynamics: str = "production"
    noise: str = "production"
    pool: str = "production"
    simplex_map: bool = False
    t2_arm: str | None = None

    def __post_init__(self) -> None:
        if self.aggregate not in ("v0.2", "v0.3"):
            raise ValueError(f"unknown aggregate {self.aggregate!r}")
        if self.state_dynamics != "production" and self.state_dynamics not in U_ARMS:
            raise ValueError(f"unknown state/dynamics provider {self.state_dynamics!r}")
        if self.noise != "production" and self.noise not in NOISE_LAWS:
            raise ValueError(f"unknown noise law {self.noise!r}")
        if (self.noise == "production") != (self.pool == "production"):
            raise ValueError("production noise goes with the production pool, and only with it")
        if self.pool not in ("production", "current_rule", "corrected"):
            raise ValueError(f"unknown pool {self.pool!r}")

    @property
    def dynamics_differ_from_d0(self) -> bool:
        return self.state_dynamics in U_ARMS

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: dict) -> "CandidateConfig":
        return cls(**record)


def config_c0(use_noise_laws: bool | None = None) -> CandidateConfig:
    """Production 1.2.0-rc1.

    With ``noise_laws`` present the noise is LW0 on the current-rule pool, drawn
    from production's ``z`` (so the self-check also verifies LW0); without it,
    production's own noise code.
    """

    if use_noise_laws is None:
        use_noise_laws = noise_laws_module() is not None
    if use_noise_laws:
        return CandidateConfig("C0", noise="LW0", pool="current_rule")
    return CandidateConfig("C0")


def config_f1() -> CandidateConfig:
    """C0 with the corrected-residual pool (amendments 001 and 002), law LW0."""

    return CandidateConfig("F1", noise="LW0", pool="corrected")


def config_fb() -> CandidateConfig:
    """§1.4: F1 + aggregate v0.3 + interim law N0 + production state and dynamics."""

    # PREREG-AMBIGUITY: §5 applies the §3.4 simplex map to CAND's base
    # composition, and §1.4 says CAND ≡ FB when T2 → D0 and T3-rev → N0. For
    # that identity to hold FB must apply the same map to production's base.
    # The map is applied (it is a no-op unless a share is below 0.01 pp).
    return CandidateConfig("FB", aggregate="v0.3", noise="N0", pool="corrected", simplex_map=True, t2_arm="D0")


def config_cand(selection: dict) -> CandidateConfig:
    """§5 from a selection record ``{"t2_arm": ..., "noise_law": ...}``."""

    t2_arm, law = selection["t2_arm"], selection["noise_law"]
    if t2_arm not in T2_ARMS:
        raise ValueError(f"T2 selection {t2_arm!r} is not one of {T2_ARMS}")
    if law not in ("N0", "NH0", "NH0.5", "NH1"):
        raise ValueError(f"T3-rev selection {law!r} is not a T3-rev arm")
    return CandidateConfig(
        "CAND", aggregate="v0.3",
        state_dynamics="production" if t2_arm == "D0" else t2_arm,
        noise=law, pool="corrected", simplex_map=True, t2_arm=t2_arm,
    )


def cand_equals_fb(selection: dict) -> bool:
    """§1.4 coincidence: T2 → D0 and T3-rev → N0 make CAND and FB the same model."""

    return selection["t2_arm"] == "D0" and selection["noise_law"] == "N0"


# --------------------------------------------------------------------------- #
# Seeds, maps, pools
# --------------------------------------------------------------------------- #


def layer_seed(seed: int, origin: date, horizon_days: int, layer: str) -> int:
    """§6.2 sub-seed of a new layer (no modulus, as frozen)."""

    if layer not in NEW_LAYERS:
        raise ValueError(f"unknown new layer {layer!r}")
    token = f"{seed}:{origin.isoformat()}:{horizon_days}:calibration-2030:{layer}"
    return int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)


def layer_rngs(seed: int, origin: date, horizon_days: int) -> dict[str, np.random.Generator]:
    """One ``default_rng`` per new layer."""

    return {layer: np.random.default_rng(layer_seed(seed, origin, horizon_days, layer)) for layer in NEW_LAYERS}


def production_noise_z(seed: int, origin: date, horizon_days: int, samples: int) -> np.ndarray:
    """Production's standard normals for the noise layer: the same generator and
    draw call as ``draw_election_noise_b`` (unchanged layer, production seed)."""

    rng = np.random.default_rng(derive_election_noise_b_seed(seed, origin, horizon_days))
    return rng.standard_normal((samples, N_CATEGORIES))


def simplex_map(draws_pp: np.ndarray) -> np.ndarray:
    """§3.4: REST = 100 − Σ eight parties, clip all nine at 0.01 pp, renormalize to 100."""

    x = np.array(draws_pp, dtype=np.float64, copy=True)
    if x.ndim != 2 or x.shape[1] not in (N_PARTIES, N_CATEGORIES):
        raise ValueError(f"expected (n, 8) or (n, 9) draws, got {x.shape}")
    out = np.empty((x.shape[0], N_CATEGORIES), dtype=np.float64)
    out[:, :N_PARTIES] = x[:, :N_PARTIES]
    out[:, N_PARTIES] = 100.0 - x[:, :N_PARTIES].sum(axis=1)
    np.maximum(out, SIMPLEX_FLOOR_PP, out=out)
    return out / out.sum(axis=1, keepdims=True) * 100.0


def pool_years(target_year: int) -> tuple[int, ...]:
    """The elections strictly before the target (the noise pool)."""

    return tuple(y for y in ELECTION_YEARS if y < target_year)


_POOL_CACHE: dict[tuple[str, tuple[int, ...]], Any] = {}
_T2_CTX: dict | None = None


def _pool(kind: str, years: tuple[int, ...]):
    key = (kind, years)
    if key not in _POOL_CACHE:
        module = noise_laws_module()
        if module is None:
            raise RuntimeError(f"{NOISE_LAWS_MODULE} is required for pool {kind!r} but is not present")
        builder = {"current_rule": module.current_rule_pool, "corrected": module.corrected_pool}[kind]
        _POOL_CACHE[key] = builder(list(years))
    return _POOL_CACHE[key]


def t2_context() -> dict:
    """``t2_arms.t2_context()``, built once per process."""

    global _T2_CTX
    if _T2_CTX is None:
        _T2_CTX = t2_arms_module().t2_context()
    return _T2_CTX


# --------------------------------------------------------------------------- #
# The national stage
# --------------------------------------------------------------------------- #


def candidate_national_vote_shares(
    config: CandidateConfig,
    *,
    as_of: str | date | None,
    election_date: str | date,
    samples: int,
    seed: int,
    data_dir: Path | str | None = None,
    t2_k: tuple[float, int] | None = None,
) -> NationalVoteShareSampleResult:
    """The national stage of one arm, in production's result type.

    ``t2_k``: ``inflation_k(arm, ctx, as_of)`` computed beforehand (it is
    deterministic and costly, so the harness computes it once per origin);
    ``None`` computes it here.

    Production's ``generate_national_vote_shares`` is always called on the
    arm's data root. It fixes ``as_of`` and the horizon exactly as production
    does, and supplies the production base composition (OpinionState +
    capped bootstrap, production sub-seeds) and production's own noise.
    """

    prod = generate_national_vote_shares(
        as_of=as_of, election_date=election_date, samples=samples, seed=seed, data_dir=data_dir,
    )
    as_of_date, horizon_days = prod.as_of, prod.horizon_days
    diagnostics: dict[str, Any] = {
        "candidate_arm": config.name,
        "candidate_config": config.to_dict(),
        "production_diagnostics": dict(prod.diagnostics),
    }

    # (b) state + dynamics
    if config.state_dynamics == "production":
        base = prod.base_comp_matrix
        if config.simplex_map:
            base = simplex_map(base)
        opinion_draws, deltas = prod.opinion_state_draws, prod.dynamics_deltas
    else:
        t2 = t2_arms_module()
        ctx = t2_context()
        rngs = layer_rngs(seed, as_of_date, horizon_days)
        z_state = rngs["state"].standard_normal((samples, N_PARTIES))
        z_dyn = rngs["dynamics"].standard_normal((samples, N_PARTIES))
        # PREREG-AMBIGUITY: §6.2 lists an `ou` layer, but the T2 interface
        # draws the OU dynamics from z_dyn (T2 §3.6 shares Z_dyn across Ua,
        # Ub1 and Ub2). The `ou` generator is derived and its seed recorded,
        # but nothing is drawn from it.
        k, k_count = t2_k if t2_k is not None else t2.inflation_k(config.state_dynamics, ctx, as_of_date)
        base = np.asarray(
            t2.state_dynamics_draws(
                config.state_dynamics, ctx, as_of_date, prod.election_date,
                z_state=z_state, z_dyn=z_dyn, boot_rng=rngs["bootstrap"], d0_base_seed=seed, k=k,
            ),
            dtype=np.float64,
        )
        if base.shape != (samples, N_CATEGORIES):
            raise ValueError(f"t2_arms returned shape {base.shape}, expected {(samples, N_CATEGORIES)}")
        # The T2 output is already after the §3.4 map; it is not mapped twice.
        opinion_draws = np.full((samples, N_CATEGORIES), np.nan)
        deltas = np.full((samples, N_CATEGORIES), np.nan)
        diagnostics.update({
            "t2_arm": config.state_dynamics, "k": float(k), "k_eligible_targets": int(k_count),
            "layer_seeds": {layer: layer_seed(seed, as_of_date, horizon_days, layer) for layer in NEW_LAYERS},
        })

    # (c) noise
    if config.noise == "production":
        if config.state_dynamics != "production" or config.simplex_map:
            raise ValueError("production noise is only defined on production's own base composition")
        nat, lambdas, training_years = prod.nat_shares_matrix, prod.lambdas, prod.training_years
    else:
        years = pool_years(prod.election_date.year)
        pool = _pool(config.pool, years)
        law = noise_laws_module().fit_law(config.noise, pool)
        z = production_noise_z(seed, as_of_date, horizon_days, samples)
        residuals = np.asarray(law.draw(z, base), dtype=np.float64)
        if residuals.shape != (samples, N_CATEGORIES):
            raise ValueError(f"noise law returned shape {residuals.shape}")
        nat, lambdas = apply_batch_simplex_transfer(base, residuals, eps=MIN_SHARE_PCT)
        nat = nat / np.sum(nat, axis=1, keepdims=True)
        training_years = tuple(int(y) for y in pool.get("years", years))
        diagnostics.update({
            "noise_law": config.noise, "noise_pool": config.pool, "noise_pool_years": list(training_years),
            "noise_seed": derive_election_noise_b_seed(seed, as_of_date, horizon_days),
            "mean_lambda": float(np.mean(lambdas)), "min_lambda": float(np.min(lambdas)),
        })

    return dataclasses.replace(
        prod,
        opinion_state_draws=opinion_draws,
        dynamics_deltas=deltas,
        base_comp_matrix=base,
        nat_shares_matrix=nat,
        lambdas=lambdas,
        training_years=training_years,
        diagnostics=diagnostics,
    )


@contextmanager
def _engine_national_stage(stage: Callable[..., NationalVoteShareSampleResult]):
    """Install ``stage`` as the engine's national stage for one call.

    The engine resolves ``generate_national_vote_shares`` from its module
    globals at call time, so everything after it is production's own code.
    Process-local; the harness runs cases in separate processes.
    """

    import scripts.simulator.engine as engine

    original = engine.generate_national_vote_shares
    engine.generate_national_vote_shares = stage
    try:
        yield engine
    finally:
        engine.generate_national_vote_shares = original


def simulate_candidate(
    config: CandidateConfig,
    *,
    as_of: date,
    election_date: date,
    samples: int,
    seed: int,
    data_dir: Path | str | None = None,
    processed_geo_dir: Path | str | None = None,
    t2_k: tuple[float, int] | None = None,
):
    """``simulate_election`` with the arm's national stage. Returns ``(result, national)``."""

    captured: dict[str, NationalVoteShareSampleResult] = {}

    def stage(*, as_of, election_date, samples, seed, data_dir, noise_model, **extra):
        if extra:
            raise TypeError(f"unexpected national-stage arguments {sorted(extra)}")
        result = candidate_national_vote_shares(
            config, as_of=as_of, election_date=election_date, samples=samples, seed=seed, data_dir=data_dir,
            t2_k=t2_k,
        )
        captured["national"] = result
        return result

    with _engine_national_stage(stage) as engine:
        result = engine.simulate_election(
            as_of=as_of, election_date=election_date, samples=samples, seed=seed,
            data_dir=data_dir, processed_geo_dir=processed_geo_dir,
        )
    return result, captured["national"]


# --------------------------------------------------------------------------- #
# Data roots
# --------------------------------------------------------------------------- #


def build_opinion_root(parent: Path, name: str, timeseries_file: Path) -> Path:
    """A processed-data folder identical to production except the opinion timeseries.

    The analogue of ``scripts.poll_aggregate.replay.build_data_root`` for the
    1.2.0 layout, where ``opinion_inputs`` reads
    ``poll_aggregate/v0_2/swedishpolls_aggregate_timeseries.csv``. The polls
    file (``poll_aggregate/v0_2/individual_polls.csv``, pinned) and every other
    input are symlinks to production's.
    """

    from scripts.simulator.model_inputs import AGGREGATE_DIR, AGGREGATE_TIMESERIES

    root = Path(parent) / name
    root.mkdir(parents=True)
    for child in PROCESSED_ROOT.iterdir():
        if child.name != AGGREGATE_DIR.parts[0]:
            os.symlink(child, root / child.name)
    aggregate_parent = PROCESSED_ROOT / AGGREGATE_DIR.parts[0]
    (root / AGGREGATE_DIR).mkdir(parents=True)
    for child in aggregate_parent.iterdir():
        if child.name != AGGREGATE_DIR.name:
            os.symlink(child, root / AGGREGATE_DIR.parts[0] / child.name)
    for child in (PROCESSED_ROOT / AGGREGATE_DIR).iterdir():
        if child.name != AGGREGATE_TIMESERIES.name:
            os.symlink(child, root / AGGREGATE_DIR / child.name)
    os.symlink(Path(timeseries_file).resolve(), root / AGGREGATE_TIMESERIES)
    return root
