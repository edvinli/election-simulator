"""Validate the harness against the repository's existing historical artifacts.

Two independent reproduction checks, both run at the *legacy* configuration so
that exact equality is the expected outcome rather than something excused by
Monte Carlo noise:

1. **Tier 1 vs `data/processed/election_layer_v2/forward_eval_2010_2022.json`.**
   The legacy `pp_noise_only` evaluation is exact, not sampled: it enumerates the
   K support points and scores them. CONTROL at Tier 1 has that same predictive
   law, so the harness's closed-form Tier-1 scores must equal the frozen artifact
   **exactly**. The Monte Carlo values reported in the baseline are then compared
   against that exact anchor to quantify the sampling error at N = 20 000.

2. **Tier 3 vs `data/processed/seat_hindcasts/seat_hindcast_summary.json`.**
   Re-runs the frozen simulator at the legacy configuration (5 000 samples,
   seed 12345) and compares the 8-party seat-vector energy score for all 12
   legacy cases. Exact equality is required.

Any unexplained mismatch is a blocker. Nothing is written to ``data/``.
"""

from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.election_noise_v2.control_baseline.harness import metrics as M
from diagnostics.election_noise_v2.control_baseline.harness.manifest import ELECTION_DATES
from diagnostics.election_noise_v2.control_baseline.harness.pipeline import (
    tier1_control_draws,
    tier1_support,
)
from diagnostics.election_noise_v2.control_baseline.harness.rng import FROZEN_SEEDS
from scripts.election_layer_v2.forward_eval import compute_discrete_crps
from scripts.elections.load import load_election_targets_for_forecasting
from scripts.election_residuals.config import ALL_CATEGORIES
from scripts.pollofpolls.backtest_metrics import calculate_crps
from scripts.seat_hindcasts.config import EVALUATION_ELECTIONS
from scripts.seat_hindcasts.metrics import calculate_multivariate_energy_score
from scripts.seat_hindcasts.models import evaluate_election_simulator_v1
from scripts.simulator.config import PARLIAMENTARY_PARTIES_8

OUT = Path(__file__).resolve().parents[1]
LEGACY_TIER1 = REPO_ROOT / "data/processed/election_layer_v2/forward_eval_2010_2022.json"
LEGACY_SEAT = REPO_ROOT / "data/processed/seat_hindcasts/seat_hindcast_summary.json"


def check_crps_estimator_equivalence() -> dict:
    """The O(N log N) production CRPS must equal the O(N^2) one at feasible N."""
    rng = np.random.default_rng(11)
    worst = 0.0
    for n in (5, 50, 500, 3000):
        x = rng.normal(size=n) * 3 + 25
        y = float(rng.normal() * 3 + 25)
        a = compute_discrete_crps(x, y)
        b = calculate_crps(x, y)
        worst = max(worst, abs(a - b))
    return {
        "check": "compute_discrete_crps vs calculate_crps",
        "max_abs_difference": worst,
        "tolerance": 1e-12,
        "passes": worst <= 1e-12,
    }


def check_tier1_against_legacy() -> dict:
    """Closed-form Tier-1 CONTROL scores must equal the frozen exact artifact."""
    legacy = json.loads(LEGACY_TIER1.read_text())
    by_el = {
        (r["election_year"], r["variant"]): r
        for r in legacy["by_variant_election"]
    }
    targets = load_election_targets_for_forecasting()
    rows = []
    ok = True
    for year in (2014, 2018, 2022):
        ed = ELECTION_DATES[year]
        support = tier1_support(ed, year)
        truth = np.array([targets[ed][c] for c in ALL_CATEGORIES], dtype=float)
        crps = [float(calculate_crps(support[:, i], float(truth[i]))) for i in range(9)]
        mine8 = round(float(np.mean(crps[:8])), 4)
        mine9 = round(float(np.mean(crps)), 4)
        ref = by_el[(year, "pp_noise_only")]
        r = {
            "election_year": year,
            "k_support_points": int(support.shape[0]),
            "legacy_training_pool_size": ref["training_pool_size"],
            "exact_crps_8party": mine8,
            "legacy_crps_8party": ref["mean_CRPS_8parties"],
            "crps_8party_exact_match": mine8 == ref["mean_CRPS_8parties"],
            "exact_crps_all9": mine9,
            "legacy_crps_all9": ref["mean_CRPS_all9"],
            "crps_all9_exact_match": mine9 == ref["mean_CRPS_all9"],
            "support_points_match_pool_size": int(support.shape[0]) == ref["training_pool_size"],
        }
        ok &= r["crps_8party_exact_match"] and r["crps_all9_exact_match"] and r["support_points_match_pool_size"]
        rows.append(r)
    return {"check": "Tier 1 closed form vs forward_eval_2010_2022.json (pp_noise_only)", "rows": rows, "passes": ok}


def check_tier1_monte_carlo_error(draws: int = 20_000) -> dict:
    """Quantify the sampling error of the reported Tier-1 MC scores against the exact anchor."""
    targets = load_election_targets_for_forecasting()
    rows = []
    for year in (2014, 2018, 2022):
        ed = ELECTION_DATES[year]
        truth = np.array([targets[ed][c] for c in ALL_CATEGORIES], dtype=float)
        support = tier1_support(ed, year)
        exact = M.exact_tier1_metrics(support, truth)
        mc = []
        for s in FROZEN_SEEDS:
            d = tier1_control_draws(ed, year, s, draws)
            mc.append(
                {
                    **M.d1_joint_vote_energy_score(d.votes_pct, truth),
                    "crps_8party_mean": M.d2_marginal_vote_metrics(d.votes_pct, truth)["crps_8party_mean"],
                }
            )
        rows.append(
            {
                "election_year": year,
                "exact_es_9cat": exact["exact_es_9cat"],
                "mc_es_9cat_five_seed_mean": float(np.mean([m["es_9cat"] for m in mc])),
                "mc_es_9cat_five_seed_sd": float(np.std([m["es_9cat"] for m in mc], ddof=1)),
                "es_9cat_abs_mc_error": abs(float(np.mean([m["es_9cat"] for m in mc])) - exact["exact_es_9cat"]),
                "es_9cat_rel_mc_error_pct": 100.0
                * abs(float(np.mean([m["es_9cat"] for m in mc])) - exact["exact_es_9cat"])
                / exact["exact_es_9cat"],
                "exact_crps_8party": exact["exact_crps_8party_mean"],
                "mc_crps_8party_five_seed_mean": float(np.mean([m["crps_8party_mean"] for m in mc])),
                "crps_8party_rel_mc_error_pct": 100.0
                * abs(float(np.mean([m["crps_8party_mean"] for m in mc])) - exact["exact_crps_8party_mean"])
                / exact["exact_crps_8party_mean"],
            }
        )
    return {
        "check": "Tier 1 Monte Carlo error vs exact anchor",
        "draws_per_seed": draws,
        "rows": rows,
        "note": (
            "CONTROL's Tier-1 law is exactly K equal atoms, so the exact score is the "
            "limit of the Monte Carlo score. The deviation below is pure sampling error "
            "from drawing K atoms N times; it is not a modelling difference."
        ),
    }


def check_tier3_against_legacy() -> dict:
    """Reproduce all 12 legacy seat-hindcast energy scores at the legacy configuration."""
    legacy = json.loads(LEGACY_SEAT.read_text())
    cases = {(c["election_year"], c["horizon_days"], c["seed"]): c for c in legacy["cases"]}
    meta = legacy["summary"]["metadata"] if "summary" in legacy else legacy["metadata"]
    rows = []
    ok = True
    for (year, h, seed), ref in sorted(cases.items()):
        info = EVALUATION_ELECTIONS[str(year)]
        ed = info["election_date"]
        res = evaluate_election_simulator_v1(
            as_of=ed - timedelta(days=h),
            election_date=ed,
            baseline_year=info["geography_baseline_year"],
            samples=int(meta["samples_per_case"]),
            seed=seed,
        )
        actual = np.array([info["actual_seats"][p] for p in PARLIAMENTARY_PARTIES_8], dtype=np.int64)
        es = calculate_multivariate_energy_score(res.seats_matrix, actual)
        match = round(es, 4) == ref["joint_energy_score"]
        ok &= match
        rows.append(
            {
                "election_year": year,
                "horizon_days": h,
                "seed": seed,
                "samples": int(meta["samples_per_case"]),
                "reproduced_seat_energy_score": round(es, 4),
                "legacy_seat_energy_score": ref["joint_energy_score"],
                "exact_match": match,
            }
        )
    return {
        "check": "Tier 3 vs seat_hindcast_summary.json (joint_energy_score), legacy configuration",
        "legacy_samples_per_case": int(meta["samples_per_case"]),
        "legacy_seed": 12345,
        "rows": rows,
        "passes": ok,
    }


def main() -> int:
    report = {
        "status": "HARNESS VALIDATION - research only",
        "crps_estimator_equivalence": check_crps_estimator_equivalence(),
        "tier1_exact_vs_legacy": check_tier1_against_legacy(),
        "tier3_vs_legacy_seat_hindcast": check_tier3_against_legacy(),
        "tier1_monte_carlo_error": check_tier1_monte_carlo_error(),
    }
    blockers = [
        k
        for k, v in report.items()
        if isinstance(v, dict) and v.get("passes") is False
    ]
    report["blockers"] = blockers
    report["all_reproduction_checks_pass"] = not blockers
    (OUT / "harness_validation.json").write_text(json.dumps(report, indent=2) + "\n")

    print("CRPS estimator equivalence:", report["crps_estimator_equivalence"])
    print("\nTier 1 exact vs legacy:")
    for r in report["tier1_exact_vs_legacy"]["rows"]:
        print("  ", r)
    print("\nTier 3 vs legacy seat hindcast:")
    for r in report["tier3_vs_legacy_seat_hindcast"]["rows"]:
        print(f"   {r['election_year']} h={r['horizon_days']:>3} seed={r['seed']}: "
              f"{r['reproduced_seat_energy_score']} vs {r['legacy_seat_energy_score']} exact={r['exact_match']}")
    print("\nTier 1 Monte Carlo error at N=20000:")
    for r in report["tier1_monte_carlo_error"]["rows"]:
        print(f"   {r['election_year']}: exact ES {r['exact_es_9cat']:.4f} vs MC {r['mc_es_9cat_five_seed_mean']:.4f} "
              f"(rel err {r['es_9cat_rel_mc_error_pct']:.3f}%, seed sd {r['mc_es_9cat_five_seed_sd']:.4f})")
    print("\nALL REPRODUCTION CHECKS PASS:", report["all_reproduction_checks_pass"])
    return 0 if report["all_reproduction_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
