"""Authoritative Amendment-2 evaluation case manifest, derived mechanically.

Tier 1 (vote level) and Tier 3-ISO (seat/coalition level) are the two gate tiers.
The full-pipeline Tier-2/Tier-3 results from Part 3 are **preserved byte-for-byte**
and recorded here as retrospective diagnostics only; they are never recomputed and
never enter the gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.election_layer_v2.config import CANONICAL_WINDOW_DAYS
from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals
from scripts.election_residuals.config import ALL_CATEGORIES
from scripts.elections.load import load_election_targets_for_forecasting
from scripts.mandates.law import mandate_law_for_election_year
from scripts.simulator.config import PARLIAMENTARY_PARTIES_8

from diagnostics.election_noise_v2.control_baseline.harness.rng import (
    DRAWS_PER_SEED,
    FROZEN_SEEDS,
    stream_seeds,
    tier1_origin,
)

from .isolated import (
    FORBIDDEN_GEOGRAPHY_MODES,
    GEOGRAPHY_MODE,
    TIER3_ISO_TARGETS,
    certified_seats,
    fixed_seats,
)

PART3 = REPO_ROOT / "diagnostics/election_noise_v2/control_baseline"
PART2B = REPO_ROOT / "diagnostics/election_noise_v2/historical_seat_extension/processed"

TIER1_CANDIDATE_TARGETS: tuple[int, ...] = (2010, 2014, 2018, 2022)
K_OUTER_MIN: int = 3

AMENDMENT2 = {
    "status": "FROZEN - AMENDMENT 2",
    "commit": "00f7030eda65e64efa3253798eb8cc6c0b8fe9cc",
    "body_sha256": "5a9a6dc8ef6f26ce3ce152155af0ed288fb8d2d97c81a2606e513cf20e1b058b",
    "whole_file_sha256": "126a8f4902f8af36f007dca4442f8e41bed5a507b190d30960b38f33c6345572",
    "freeze_timestamp_utc": "2026-08-31T08:54:20Z",
    "edited_by_this_task": False,
}

TRUTH_INPUTS = {
    "elections": REPO_ROOT / "data/processed/elections/riksdag_election_results.csv",
    "certified_mandates_2018_2022": REPO_ROOT / "data/processed/mandates/historical_certified_mandates.csv",
    "certified_mandates_2010_2014": PART2B / "certified_mandates_2010_2014.csv",
    "fixed_seats_by_year": PART2B / "fixed_seats_by_year.json",
    "polls": REPO_ROOT / "data/processed/pollofpolls/swedishpolls_individual_polls.csv",
    "geography_votes_prod": REPO_ROOT / "data/processed/geography/constituency_party_votes_2014_2022.csv",
    "geography_votes_research": PART2B / "research_geography/constituency_party_votes_2014_2022.csv",
}

PRESERVED_DIAGNOSTICS = [
    "evaluation_case_manifest.json",
    "control_scores_by_case_seed.csv",
    "control_scores_by_election.csv",
    "control_scores_summary.json",
    "coalition_brier_by_election.csv",
    "monte_carlo_stability.csv",
    "lambda_diagnostics.csv",
    "harness_validation.json",
    "mask_level/coalition_brier_by_mask.csv",
]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@dataclass
class Case:
    tier: str
    target_year: int
    election_date: str
    base: str
    consensus_window_days: int
    origin_date: str
    horizon_days: int
    training_residual_years: list[int]
    k_outer: int
    metrics: list[str]
    seeds: list[int]
    draws_per_seed: int
    seed_streams: dict[str, dict[str, int]] = field(default_factory=dict)
    truth_vote_pct: dict[str, float] | None = None
    truth_seats: dict[str, int] | None = None
    geography_baseline_year: int | None = None
    geography_mode: str | None = None
    forbidden_geography_modes: list[str] | None = None
    mandate_law: str | None = None
    first_divisor: str | None = None
    fixed_seats: dict[str, int] | None = None


def _pool(year: int) -> tuple[list[int], int]:
    p = load_chronological_pp_residuals(target_election_year=year)
    ys = [int(y) for y in p.training_years]
    return ys, len(ys)


def _truth_vote(year: int, ed: str) -> dict[str, float]:
    from datetime import date

    comp = load_election_targets_for_forecasting()[date.fromisoformat(ed)]
    return {c: round(float(comp[c]), 6) for c in ALL_CATEGORIES}


def _streams(seed_list, origin, horizon) -> dict[str, dict[str, int]]:
    out = {}
    for s in seed_list:
        ss = stream_seeds(s, origin, horizon)
        out[str(s)] = {
            "opinion_state_seed": ss.opinion_state_seed,
            "dynamics_seed": ss.dynamics_seed,
            "election_noise_index_seed": ss.election_noise_index_seed,
            "election_noise_sign_seed": ss.election_noise_sign_seed,
        }
    return out


def build_manifest() -> dict:
    tier1_elig, tier1_cases = [], []
    for year in TIER1_CANDIDATE_TARGETS:
        ys, k = _pool(year)
        ok = k >= K_OUTER_MIN
        tier1_elig.append(
            {"target_year": year, "training_residual_years": ys, "k_outer": k,
             "eligible": ok, "reason": "" if ok else f"K_outer={k} < {K_OUTER_MIN}"}
        )
        if not ok:
            continue
        ed = TIER3_ISO_TARGETS[year]["election_date"]
        origin, horizon = tier1_origin(ed)
        tier1_cases.append(Case(
            tier="tier1", target_year=year, election_date=ed.isoformat(),
            base="14-day publication-safe pre-election polling consensus (deterministic)",
            consensus_window_days=CANONICAL_WINDOW_DAYS,
            origin_date=origin.isoformat(), horizon_days=horizon,
            training_residual_years=ys, k_outer=k,
            metrics=["D1_joint_vote_energy_score", "D2_party_crps", "D2_interval_coverage", "D5_lambda"],
            seeds=list(FROZEN_SEEDS), draws_per_seed=DRAWS_PER_SEED,
            seed_streams=_streams(FROZEN_SEEDS, origin, horizon),
            truth_vote_pct=_truth_vote(year, ed.isoformat()),
        ))

    iso_cases = []
    for year, spec in TIER3_ISO_TARGETS.items():
        ys, k = _pool(year)
        ed = spec["election_date"]
        origin, horizon = tier1_origin(ed)
        cfg = mandate_law_for_election_year(year)
        iso_cases.append(Case(
            tier="tier3_iso", target_year=year, election_date=ed.isoformat(),
            base="14-day publication-safe pre-election polling consensus (deterministic; no OpinionState, no Dynamics)",
            consensus_window_days=CANONICAL_WINDOW_DAYS,
            origin_date=origin.isoformat(), horizon_days=horizon,
            training_residual_years=ys, k_outer=k,
            metrics=["D3_seat_vector_energy_score", "D3_seat_crps", "D3_seat_coverage",
                     "D4_coalition_majority_brier", "D5_lambda"],
            seeds=list(FROZEN_SEEDS), draws_per_seed=DRAWS_PER_SEED,
            seed_streams=_streams(FROZEN_SEEDS, origin, horizon),
            truth_vote_pct=_truth_vote(year, ed.isoformat()),
            truth_seats=certified_seats(year),
            geography_baseline_year=spec["geography_baseline_year"],
            geography_mode=GEOGRAPHY_MODE,
            forbidden_geography_modes=list(FORBIDDEN_GEOGRAPHY_MODES),
            mandate_law=cfg.law.value,
            first_divisor=str(cfg.first_divisor),
            fixed_seats=fixed_seats(year),
        ))

    return {
        "schema_version": "2.0",
        "status": "AUTHORITATIVE AMENDMENT-2 EVALUATION MANIFEST - research only; no production artifact modified",
        "preregistration": AMENDMENT2,
        "predecessors": {
            "part3_baseline": "998a20047cf9bae1e9b8a59d4ec4888684842fd5",
            "part3b": "89d340880a4bdb389f94ce61fa3333799b58d81a",
            "part3c_audit": "7f37e127a81b2bbdccaa26a27b7275ba39e96dec",
            "amendment2": AMENDMENT2["commit"],
        },
        "monte_carlo": {
            "seeds": list(FROZEN_SEEDS),
            "draws_per_seed": DRAWS_PER_SEED,
            "draws_per_case_per_model": DRAWS_PER_SEED * len(FROZEN_SEEDS),
        },
        "coalition_masks": {
            "mask_range": [1, 254], "excluded_masks": [0, 255],
            "party_order": list(PARLIAMENTARY_PARTIES_8),
            "majority_threshold": 175,
            "effective_distinct_events_per_election": 127,
        },
        "gate_tiers": ["tier1", "tier3_iso"],
        "tier1_candidate_targets": list(TIER1_CANDIDATE_TARGETS),
        "tier1_eligibility": tier1_elig,
        "counts": {
            "N_T1": len(tier1_cases),
            "N_seat": len(iso_cases),
            "tier1_elections": sorted(c.target_year for c in tier1_cases),
            "tier3_iso_elections": sorted(c.target_year for c in iso_cases),
            "tier1_cases": len(tier1_cases),
            "tier3_iso_cases": len(iso_cases),
        },
        "cases": {"tier1": [asdict(c) for c in tier1_cases],
                  "tier3_iso": [asdict(c) for c in iso_cases]},
        "truth_input_provenance": {
            k: {"path": str(v.relative_to(REPO_ROOT)), "sha256": sha256_file(v)}
            for k, v in TRUTH_INPUTS.items()
        },
        "preserved_full_pipeline_diagnostics": {
            "role": "RETROSPECTIVE DIAGNOSTICS ONLY - excluded from the adoption gate (Amendment 2 §E.2, §F.2)",
            "reason": "their historical Poll-of-Polls state input is not publication-time leakage-safe (Amendment 2 §E.5 item 7)",
            "source_commit": "998a20047cf9bae1e9b8a59d4ec4888684842fd5",
            "location": "diagnostics/election_noise_v2/control_baseline/",
            "never_recomputed_or_deleted": True,
            "files": {
                f: sha256_file(PART3 / f) for f in PRESERVED_DIAGNOSTICS
            },
        },
    }


def validate_manifest(m: dict) -> list[str]:
    probs: list[str] = []
    c = m["counts"]
    if c["N_T1"] != 3:
        probs.append(f"N_T1 must be 3, got {c['N_T1']}")
    if c["N_seat"] != 3:
        probs.append(f"N_seat must be 3, got {c['N_seat']}")
    if c["tier1_elections"] != [2014, 2018, 2022]:
        probs.append(f"Tier-1 elections wrong: {c['tier1_elections']}")
    if c["tier3_iso_elections"] != [2014, 2018, 2022]:
        probs.append(f"Tier 3-ISO elections wrong: {c['tier3_iso_elections']}")
    if set(m["tier1_candidate_targets"]) != set(TIER1_CANDIDATE_TARGETS):
        probs.append("Tier-1 candidate target set modified")
    if 2010 in c["tier1_elections"] or 2010 in c["tier3_iso_elections"]:
        probs.append("2010 was included; the frozen rules exclude it")
    if m["gate_tiers"] != ["tier1", "tier3_iso"]:
        probs.append("gate tiers modified")

    for case in m["cases"]["tier1"] + m["cases"]["tier3_iso"]:
        if case["seeds"] != list(FROZEN_SEEDS):
            probs.append(f"seed list altered in {case['tier']} {case['target_year']}")
        if case["draws_per_seed"] != DRAWS_PER_SEED:
            probs.append(f"draw count altered in {case['tier']} {case['target_year']}")
        if case["k_outer"] < K_OUTER_MIN:
            probs.append(f"below K_outer minimum: {case['target_year']}")
        if any(y >= case["target_year"] for y in case["training_residual_years"]):
            probs.append(f"LEAKAGE: future residual year in the {case['target_year']} pool")
        if abs(sum(case["truth_vote_pct"].values()) - 100.0) > 1e-3:
            probs.append(f"truth vote vector does not sum to 100 for {case['target_year']}")

    for case in m["cases"]["tier3_iso"]:
        exp = mandate_law_for_election_year(case["target_year"])
        if case["mandate_law"] != exp.law.value:
            probs.append(f"wrong law for {case['target_year']}")
        if case["first_divisor"] != str(exp.first_divisor):
            probs.append(f"wrong divisor for {case['target_year']}")
        if case["geography_mode"] != GEOGRAPHY_MODE:
            probs.append(f"geography mode must be {GEOGRAPHY_MODE} for {case['target_year']}")
        if "oracle" not in (case["forbidden_geography_modes"] or []):
            probs.append(f"oracle mode not recorded as forbidden for {case['target_year']}")
        if case["geography_baseline_year"] >= case["target_year"]:
            probs.append(f"non-chronological baseline for {case['target_year']}")
        if sum(case["truth_seats"].values()) != 349:
            probs.append(f"certified seats do not sum to 349 for {case['target_year']}")
        if sum(case["fixed_seats"].values()) != 310:
            probs.append(f"fixed seats do not sum to 310 for {case['target_year']}")

    # Preserved diagnostics must still hash to what Part 3 committed.
    for f, h in m["preserved_full_pipeline_diagnostics"]["files"].items():
        if sha256_file(PART3 / f) != h:
            probs.append(f"preserved diagnostic changed: {f}")
    return probs
