"""Pipeline-sufficiency smoke test for the six preregistered 2014 horizons.

Determines whether the candidate historical PoP state supplies everything the
frozen model needs to *run* a 2014 hindcast:

    OpinionState v1.1 -> Dynamics v2 -> ElectionNoise CONTROL
        -> frozen geography (2010 baseline) -> PRE_2018 mandate law

**No predictive score against the certified 2014 outcome is computed or reported.**
This task must not score 2014. Only structural facts are recorded: whether each
case runs, which fallbacks fire, transition counts, and whether the draws are
valid 349-seat allocations.

The frozen production input ``data/processed/pollofpolls/pollofpolls_timeseries.csv``
is never modified. A research data root is assembled under a gitignored
``_scratch/`` directory: the extended timeseries is written there and every other
input is symlinked to its unchanged production copy.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals
from scripts.geography.projection import project_constituency_votes
from scripts.mandates.allocator import allocate_riksdag_seats
from scripts.mandates.law import MandateLaw, mandate_law_for_election_year
from scripts.pollofpolls.state import estimate_opinion, load_timeseries_dataset
from scripts.pollofpolls.transitions import (
    build_all_historical_transitions,
    filter_transitions_as_of,
)
from scripts.simulator.config import MODEL_PARTIES_9, PARLIAMENTARY_PARTIES_8
from scripts.vote_share_calibration.national_engine import generate_national_vote_shares

HERE = Path(__file__).resolve().parent
PROCESSED = HERE / "processed"
SCRATCH = HERE / "_scratch"
PROD = REPO_ROOT / "data" / "processed"
PART2B = REPO_ROOT / "diagnostics" / "election_noise_v2" / "historical_seat_extension" / "processed"

PARTIES = ("M", "L", "C", "KD", "S", "V", "MP", "SD")
ELECTION_2014 = date(2014, 9, 14)
HORIZONS = (112, 84, 56, 28, 14, 7)
SMOKE_DRAWS = 300  # structural check only; the frozen N is never used for an unscored run
SMOKE_SEED = 12345


def build_research_root() -> Path:
    """Research data root: extended timeseries written, everything else symlinked."""
    root = SCRATCH / "data_processed"
    (root / "pollofpolls").mkdir(parents=True, exist_ok=True)
    (root / "elections").mkdir(parents=True, exist_ok=True)

    cand = list(csv.DictReader((PROCESSED / "candidate_pop_state_2009_2026.csv").open()))
    out = root / "pollofpolls" / "pollofpolls_timeseries.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["date", *PARTIES, "FI", "other", "source_extra_json", "source_url", "retrieved_at"],
        )
        w.writeheader()
        for r in cand:
            w.writerow(
                {
                    "date": r["date"],
                    **{p: r[p] for p in PARTIES},
                    "FI": r["FI"],
                    "other": "",
                    "source_extra_json": "",
                    "source_url": r["source_url"],
                    "retrieved_at": "",
                }
            )

    for rel in (
        "pollofpolls/individual_polls.csv",
        "pollofpolls/swedishpolls_individual_polls.csv",
        "elections/riksdag_election_results.csv",
    ):
        dst = root / rel
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(PROD / rel)
    return root


def fixed_seats_2014() -> dict[str, int]:
    payload = json.loads((PART2B / "fixed_seats_by_year.json").read_text())
    return {k: int(v) for k, v in payload["fixed_seats_by_year"]["2014"].items()}


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    root = build_research_root()
    ts = load_timeseries_dataset(root / "pollofpolls" / "pollofpolls_timeseries.csv")
    report: dict = {
        "status": "SMOKE TEST ONLY - no predictive score against the 2014 outcome is computed",
        "research_timeseries_rows": len(ts),
        "research_timeseries_range": [ts[0]["date"].isoformat(), ts[-1]["date"].isoformat()],
        "draws_per_case": SMOKE_DRAWS,
        "seed": SMOKE_SEED,
        "cases": [],
    }

    cfg = mandate_law_for_election_year(2014)
    assert cfg.law is MandateLaw.PRE_2018, cfg
    pool = load_chronological_pp_residuals(target_election_year=2014)
    report["residual_pool"] = {
        "training_years": [int(y) for y in pool.training_years],
        "k_outer": len(pool.training_years),
        "no_future_year_in_pool": all(int(y) < 2014 for y in pool.training_years),
    }
    report["mandate_law"] = {"law": cfg.law.value, "first_divisor": str(cfg.first_divisor)}

    all_ok = True
    for h in HORIZONS:
        as_of = ELECTION_2014 - timedelta(days=h)
        case: dict = {"horizon_days": h, "as_of": as_of.isoformat()}
        try:
            # 1. OpinionState v1.1
            st = estimate_opinion(as_of=as_of, data_dir=root / "pollofpolls")
            case["opinion_state_as_of"] = st.as_of.isoformat()
            case["opinion_state_runs"] = True

            # 2. Dynamics v2 at the exact horizon, with the frozen fallback ladder
            eval_h = min(h, 112) if h > 112 else h
            trans = build_all_historical_transitions(ts, horizons=[eval_h])
            elig = filter_transitions_as_of(trans[eval_h], as_of)
            case["dynamics_exact_horizon"] = eval_h
            case["dynamics_eligible_transitions"] = len(elig)
            case["dynamics_meets_minimum_30"] = len(elig) >= 30
            fallback = None
            if len(elig) < 30:
                for fb in (28, 14, 7):
                    t2 = build_all_historical_transitions(ts, horizons=[fb])
                    e2 = filter_transitions_as_of(t2[fb], as_of)
                    if len(e2) >= 30:
                        fallback, elig = fb, e2
                        break
            case["dynamics_fallback_horizon"] = fallback
            case["dynamics_transitions_used"] = len(elig)
            case["max_transition_end"] = max(
                (t.end_date.isoformat() for t in elig), default=None
            )
            case["no_transition_ends_after_as_of"] = all(
                t.end_date <= as_of for t in elig
            )

            # 3. National vote draws through the frozen national engine
            nat = generate_national_vote_shares(
                as_of=as_of,
                election_date=ELECTION_2014,
                samples=SMOKE_DRAWS,
                seed=SMOKE_SEED,
                data_dir=root,
            )
            v = nat.nat_shares_matrix
            case["national_engine_runs"] = True
            case["draw_rows"] = int(v.shape[0])
            case["vote_rows_sum_to_one"] = bool(
                np.allclose(v.sum(axis=1), 1.0, atol=1e-12)
            )
            case["training_years_used"] = [int(y) for y in nat.training_years]
            case["mean_lambda"] = float(np.mean(nat.lambdas))

            # 4. Frozen geography (2010 baseline) + PRE_2018 allocator, per draw
            seat_rows = []
            for i in range(min(25, v.shape[0])):  # structural check on a subset
                shares = {p: float(v[i, j]) for j, p in enumerate(MODEL_PARTIES_9)}
                proj = project_constituency_votes(
                    national_vote_shares=shares,
                    baseline_year=2010,
                    target_year=2014,
                    mode="chronological",
                    total_national_votes=None,
                    processed_dir=PART2B / "research_geography",
                )
                alloc = allocate_riksdag_seats(
                    proj.to_allocator_input(),
                    fixed_seats_2014(),
                    first_divisor=cfg.first_divisor,
                    law=cfg.law,
                    scenario_id=f"smoke_2014_h{h}_{i}",
                )
                seat_rows.append([alloc.final_seats_by_party.get(p, 0) for p in PARLIAMENTARY_PARTIES_8])
            S = np.array(seat_rows, dtype=np.int64)
            case["seat_draws_checked"] = int(S.shape[0])
            case["all_seat_totals_349"] = bool(np.all(S.sum(axis=1) == 349))
            case["seat_law_recorded"] = alloc.law
            case["runs"] = True
        except Exception as exc:  # noqa: BLE001 - the point is to record whether it runs
            case["runs"] = False
            case["error"] = f"{type(exc).__name__}: {exc}"
            all_ok = False
        report["cases"].append(case)

    report["all_six_horizons_run"] = all_ok and all(c.get("runs") for c in report["cases"])
    (PROCESSED / "pipeline_sufficiency_2014.json").write_text(json.dumps(report, indent=2) + "\n")

    for c in report["cases"]:
        if c.get("runs"):
            print(
                f"h={c['horizon_days']:>3} as_of={c['as_of']} OpinionState=OK "
                f"transitions={c['dynamics_transitions_used']:>4} (exact_h={c['dynamics_exact_horizon']}, "
                f"fallback={c['dynamics_fallback_horizon']}) "
                f"max_transition_end={c['max_transition_end']} causal={c['no_transition_ends_after_as_of']} "
                f"seats349={c['all_seat_totals_349']} law={c['seat_law_recorded']}"
            )
        else:
            print(f"h={c['horizon_days']:>3} as_of={c['as_of']} FAILED: {c['error']}")
    print(f"\nall six 2014 horizons run: {report['all_six_horizons_run']}")
    print("(no predictive score against the 2014 outcome was computed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
