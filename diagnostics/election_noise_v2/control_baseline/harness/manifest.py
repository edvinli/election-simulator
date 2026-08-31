"""Build and validate the frozen ElectionNoise v2 evaluation case manifest.

Cases are derived from the frozen preregistration (§E.2) plus the Part-2 and
Part-2B findings. Nothing is selected, added or removed on the basis of a score.
Every derivation is re-checked here rather than trusted from an upstream file:
``part3_seat_cases.json`` from Part 2B is loaded and **validated**, and a case it
proposes is dropped with a recorded reason if it cannot actually be run.

Tier definitions (preregistration §E.2)
    Tier 1  standalone forward evaluation from the 14-day polling consensus;
            ElectionNoise isolated; vote level only (D1, D2).
    Tier 2  full-pipeline hindcast at six horizons; vote level (D1, D2).
    Tier 3  the same cases as Tier 2, at seat and coalition level (D3, D4, D5).

Eligibility
    Tier 1  candidate targets are frozen at {2010, 2014, 2018, 2022} and may never
            expand; a target is eligible iff K_outer >= 3.
    Tier 2/3  a target must additionally satisfy the §E.3 admission requirements
            *and* be runnable through the full frozen pipeline. The second
            condition is checked here, not assumed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.election_layer_v2.config import CANONICAL_WINDOW_DAYS
from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals
from scripts.election_residuals.config import ALL_CATEGORIES
from scripts.elections.load import load_election_targets_for_forecasting
from scripts.mandates.law import mandate_law_for_election_year
from scripts.simulator.config import PARLIAMENTARY_PARTIES_8

from .rng import DRAWS_PER_SEED, FROZEN_SEEDS, stream_seeds, tier1_origin

REPO_ROOT = Path(__file__).resolve().parents[4]
PART2B = REPO_ROOT / "diagnostics" / "election_noise_v2" / "historical_seat_extension" / "processed"

#: Frozen Tier-1 candidate target set (preregistration §E.2, Amendment 1). Never expand.
TIER1_CANDIDATE_TARGETS: tuple[int, ...] = (2010, 2014, 2018, 2022)
K_OUTER_MIN: int = 3

#: Frozen rolling-origin horizons (preregistration §E.2).
HORIZONS: tuple[int, ...] = (112, 84, 56, 28, 14, 7)

ELECTION_DATES: dict[int, date] = {
    2010: date(2010, 9, 19),
    2014: date(2014, 9, 14),
    2018: date(2018, 9, 9),
    2022: date(2022, 9, 11),
}

#: Chronological geography baseline (the immediately preceding election).
GEOGRAPHY_BASELINE: dict[int, int] = {2010: 2006, 2014: 2010, 2018: 2014, 2022: 2018}

INPUT_FILES: dict[str, Path] = {
    "polls_swedishpolls": REPO_ROOT / "data/processed/pollofpolls/swedishpolls_individual_polls.csv",
    "polls_individual": REPO_ROOT / "data/processed/pollofpolls/individual_polls.csv",
    "pop_timeseries": REPO_ROOT / "data/processed/pollofpolls/pollofpolls_timeseries.csv",
    "elections": REPO_ROOT / "data/processed/elections/riksdag_election_results.csv",
    "geography_votes": REPO_ROOT / "data/processed/geography/constituency_party_votes_2014_2022.csv",
    "certified_mandates": REPO_ROOT / "data/processed/mandates/historical_certified_mandates.csv",
}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@dataclass
class Tier1Case:
    tier: str
    target_year: int
    election_date: str
    base: str
    consensus_window_days: int
    origin_date: str
    horizon_days: int
    training_residual_years: list[int]
    k_outer: int
    truth_vote_pct: dict[str, float]
    seeds: list[int]
    draws_per_seed: int
    metrics: list[str]
    seed_streams: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class Tier23Case:
    tier: str
    target_year: int
    election_date: str
    as_of: str
    horizon_days: int
    training_residual_years: list[int]
    k_outer: int
    mandate_law: str
    first_divisor: str
    geography_baseline_year: int
    geography_mode: str
    truth_vote_pct: dict[str, float]
    truth_seats: dict[str, int]
    seeds: list[int]
    draws_per_seed: int
    metrics: list[str]
    seed_streams: dict[str, dict[str, int]] = field(default_factory=dict)


def _pop_timeseries_min_date() -> date:
    df = pd.read_csv(INPUT_FILES["pop_timeseries"], usecols=["date"])
    return date.fromisoformat(str(df["date"].min()))


def _training_pool(target_year: int) -> tuple[list[int], int]:
    pool = load_chronological_pp_residuals(target_election_year=target_year)
    years = [int(y) for y in pool.training_years]
    return years, len(years)


def _truth_vote(target_year: int) -> dict[str, float]:
    targets = load_election_targets_for_forecasting()
    comp = targets[ELECTION_DATES[target_year]]
    return {c: round(float(comp[c]), 6) for c in ALL_CATEGORIES}


def _truth_seats(target_year: int) -> dict[str, int]:
    if target_year in (2010, 2014):
        df = pd.read_csv(PART2B / "certified_mandates_2010_2014.csv")
    else:
        df = pd.read_csv(INPUT_FILES["certified_mandates"])
    sub = df[df["election_year"] == target_year]
    return {p: int(sub[sub["party"] == p]["total_seats"].sum()) for p in PARLIAMENTARY_PARTIES_8}


def build_manifest() -> dict[str, Any]:
    pop_min = _pop_timeseries_min_date()

    tier1_eligibility: list[dict[str, Any]] = []
    tier1_cases: list[Tier1Case] = []
    for year in TIER1_CANDIDATE_TARGETS:
        years, k = _training_pool(year)
        eligible = k >= K_OUTER_MIN
        tier1_eligibility.append(
            {
                "target_year": year,
                "training_residual_years": years,
                "k_outer": k,
                "eligible": eligible,
                "reason": "" if eligible else f"K_outer={k} < {K_OUTER_MIN}",
            }
        )
        if not eligible:
            continue
        ed = ELECTION_DATES[year]
        origin, horizon = tier1_origin(ed)
        case = Tier1Case(
            tier="tier1",
            target_year=year,
            election_date=ed.isoformat(),
            base="14-day pre-election polling consensus (deterministic; no upstream randomness)",
            consensus_window_days=CANONICAL_WINDOW_DAYS,
            origin_date=origin.isoformat(),
            horizon_days=horizon,
            training_residual_years=years,
            k_outer=k,
            truth_vote_pct=_truth_vote(year),
            seeds=list(FROZEN_SEEDS),
            draws_per_seed=DRAWS_PER_SEED,
            metrics=["D1_joint_vote_energy_score", "D2_party_crps", "D2_interval_coverage"],
        )
        for s in FROZEN_SEEDS:
            ss = stream_seeds(s, origin, horizon)
            case.seed_streams[str(s)] = {
                "opinion_state_seed": ss.opinion_state_seed,
                "dynamics_seed": ss.dynamics_seed,
                "election_noise_index_seed": ss.election_noise_index_seed,
                "election_noise_sign_seed": ss.election_noise_sign_seed,
            }
        tier1_cases.append(case)

    # ---- Tier 2 / Tier 3 -------------------------------------------------
    part2b = json.loads((PART2B / "part3_seat_cases.json").read_text())
    proposed = sorted(int(y) for y in part2b["cases"])

    seat_eligibility: list[dict[str, Any]] = []
    tier23_years: list[int] = []
    for year in proposed:
        years, k = _training_pool(year)
        checks: dict[str, Any] = {
            "part2b_admitted": True,
            "k_outer_at_least_3": k >= K_OUTER_MIN,
        }
        # A Tier-2 case is a FULL-PIPELINE hindcast. OpinionState v1.1 requires a
        # Poll of Polls daily observation on or before as_of, at every horizon.
        earliest_as_of = ELECTION_DATES[year] - timedelta(days=max(HORIZONS))
        latest_as_of = ELECTION_DATES[year] - timedelta(days=min(HORIZONS))
        checks["pop_timeseries_covers_all_horizons"] = pop_min <= earliest_as_of
        checks["pop_timeseries_min_date"] = pop_min.isoformat()
        checks["required_earliest_as_of"] = earliest_as_of.isoformat()
        checks["required_latest_as_of"] = latest_as_of.isoformat()
        runnable = all(v for k_, v in checks.items() if isinstance(v, bool))
        seat_eligibility.append(
            {
                "target_year": year,
                "checks": checks,
                "runnable_as_tier2_tier3_case": runnable,
                "reason": ""
                if runnable
                else (
                    "OpinionState v1.1 cannot be estimated: the Poll of Polls daily timeseries "
                    f"begins {pop_min.isoformat()}, after the latest required as_of "
                    f"{latest_as_of.isoformat()}. Tier 2 is defined as a full-pipeline hindcast, "
                    "and Tier 3 is defined as the same cases as Tier 2, so the target cannot "
                    "enter either tier."
                ),
            }
        )
        if runnable:
            tier23_years.append(year)

    tier2_cases: list[Tier23Case] = []
    tier3_cases: list[Tier23Case] = []
    for year in tier23_years:
        years, k = _training_pool(year)
        ed = ELECTION_DATES[year]
        cfg = mandate_law_for_election_year(year)
        for h in HORIZONS:
            as_of = ed - timedelta(days=h)
            streams = {}
            for s in FROZEN_SEEDS:
                ss = stream_seeds(s, as_of, h)
                streams[str(s)] = {
                    "opinion_state_seed": ss.opinion_state_seed,
                    "dynamics_seed": ss.dynamics_seed,
                    "election_noise_index_seed": ss.election_noise_index_seed,
                    "election_noise_sign_seed": ss.election_noise_sign_seed,
                }
            common = dict(
                target_year=year,
                election_date=ed.isoformat(),
                as_of=as_of.isoformat(),
                horizon_days=h,
                training_residual_years=years,
                k_outer=k,
                mandate_law=cfg.law.value,
                first_divisor=str(cfg.first_divisor),
                geography_baseline_year=GEOGRAPHY_BASELINE[year],
                geography_mode="chronological",
                truth_vote_pct=_truth_vote(year),
                truth_seats=_truth_seats(year),
                seeds=list(FROZEN_SEEDS),
                draws_per_seed=DRAWS_PER_SEED,
                seed_streams=streams,
            )
            tier2_cases.append(
                Tier23Case(
                    tier="tier2",
                    metrics=["D1_joint_vote_energy_score", "D2_party_crps", "D2_interval_coverage"],
                    **common,
                )
            )
            tier3_cases.append(
                Tier23Case(
                    tier="tier3",
                    metrics=[
                        "D3_seat_vector_energy_score",
                        "D3_seat_crps",
                        "D4_coalition_majority_brier",
                        "D5_lambda_diagnostics",
                    ],
                    **common,
                )
            )

    manifest = {
        "schema_version": "1.0",
        "status": "FROZEN EVALUATION CASE MANIFEST - research only, no production artifact modified",
        "preregistration": {
            "freeze": "FROZEN - AMENDMENT 1",
            "commit": "80b1c671c4b6d879a888f28a859ee392e8f59bc5",
            "body_sha256": "bac3ca06e52cc07fe74ca9e5aa785d94e30934db32193c7f948e95a49a6ae075",
            "edited_by_this_task": False,
        },
        "predecessors": {
            "part2": "cb39e84074def993e804ba4d2ec478d59c27fa4a",
            "part2b": "61d6d3b127b5140fcf4ef5c4708ddb83cc68781e",
        },
        "monte_carlo": {
            "seeds": list(FROZEN_SEEDS),
            "draws_per_seed": DRAWS_PER_SEED,
            "draws_per_case_per_model": DRAWS_PER_SEED * len(FROZEN_SEEDS),
        },
        "coalition_masks": {
            "mask_range": [1, 254],
            "excluded_masks": [0, 255],
            "party_order": list(PARLIAMENTARY_PARTIES_8),
            "majority_threshold": 175,
            "effective_distinct_events_per_election": 127,
        },
        "tier1_candidate_targets": list(TIER1_CANDIDATE_TARGETS),
        "tier1_eligibility": tier1_eligibility,
        "tier23_eligibility": seat_eligibility,
        "counts": {
            "N_T1": len(tier1_cases),
            "tier1_cases": len(tier1_cases),
            "N_seat": len(tier23_years),
            "tier2_cases": len(tier2_cases),
            "tier3_cases": len(tier3_cases),
            "tier1_elections": sorted(c.target_year for c in tier1_cases),
            "tier23_elections": sorted(tier23_years),
        },
        "input_provenance": {
            name: {"path": str(p.relative_to(REPO_ROOT)), "sha256": sha256_file(p)}
            for name, p in INPUT_FILES.items()
        },
        "part2b_derived_inputs": {
            name: {
                "path": f"diagnostics/election_noise_v2/historical_seat_extension/processed/{name}",
                "sha256": sha256_file(PART2B / name),
            }
            for name in (
                "certified_mandates_2010_2014.csv",
                "constituency_party_votes_2006_2010.csv",
                "fixed_seats_by_year.json",
                "part3_seat_cases.json",
            )
        },
        "cases": {
            "tier1": [asdict(c) for c in tier1_cases],
            "tier2": [asdict(c) for c in tier2_cases],
            "tier3": [asdict(c) for c in tier3_cases],
        },
    }
    return manifest


def validate_manifest(m: dict[str, Any]) -> list[str]:
    """Structural checks that must hold before any score is computed."""
    problems: list[str] = []
    counts = m["counts"]

    if set(m["tier1_candidate_targets"]) != set(TIER1_CANDIDATE_TARGETS):
        problems.append("Tier-1 candidate target set was modified")
    for row in m["tier1_eligibility"]:
        if row["eligible"] != (row["k_outer"] >= K_OUTER_MIN):
            problems.append(f"Tier-1 eligibility inconsistent for {row['target_year']}")
    if counts["N_T1"] != sum(1 for r in m["tier1_eligibility"] if r["eligible"]):
        problems.append("N_T1 does not equal the number of eligible Tier-1 targets")
    if counts["tier2_cases"] != counts["N_seat"] * len(HORIZONS):
        problems.append("Tier-2 case count is not N_seat x horizons")
    if counts["tier3_cases"] != counts["tier2_cases"]:
        problems.append("Tier 3 must have exactly the same cases as Tier 2")

    for c in m["cases"]["tier1"] + m["cases"]["tier2"] + m["cases"]["tier3"]:
        if c["seeds"] != list(FROZEN_SEEDS):
            problems.append(f"seed set altered in {c['tier']} {c['target_year']}")
        if c["draws_per_seed"] != DRAWS_PER_SEED:
            problems.append(f"draws_per_seed altered in {c['tier']} {c['target_year']}")
        if c["k_outer"] < K_OUTER_MIN:
            problems.append(f"case below K_outer minimum: {c['tier']} {c['target_year']}")
        # No future election may leak into a training pool.
        if any(y >= c["target_year"] for y in c["training_residual_years"]):
            problems.append(f"LEAKAGE: training pool for {c['target_year']} contains a non-past year")
        vt = c["truth_vote_pct"]
        if abs(sum(vt.values()) - 100.0) > 1e-3:
            problems.append(f"truth vote vector does not sum to 100 for {c['target_year']}")

    for c in m["cases"]["tier2"] + m["cases"]["tier3"]:
        expected = mandate_law_for_election_year(c["target_year"])
        if c["mandate_law"] != expected.law.value:
            problems.append(f"wrong law recorded for {c['target_year']}")
        if c["first_divisor"] != str(expected.first_divisor):
            problems.append(f"wrong first divisor recorded for {c['target_year']}")
        if sum(c["truth_seats"].values()) != 349:
            problems.append(f"certified seats do not sum to 349 for {c['target_year']}")
        if c["geography_baseline_year"] != GEOGRAPHY_BASELINE[c["target_year"]]:
            problems.append(f"non-chronological geography baseline for {c['target_year']}")

    if 2010 in counts["tier1_elections"] or 2010 in counts["tier23_elections"]:
        problems.append("2010 was included; the frozen rules exclude it")

    return problems
