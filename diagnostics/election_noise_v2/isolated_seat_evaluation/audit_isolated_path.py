"""Audit the proposed isolated (OpinionState/Dynamics-free) seat evaluation path.

Path under audit, for targets 2014, 2018, 2022::

    historical final 14-day polling consensus
      -> ElectionNoise (CONTROL law, unchanged)
      -> unchanged bounded simplex transfer
      -> frozen deterministic geography (chronological mode)
      -> historically correct mandate law
      -> joint seat draws

Four probes, all read-only:

A. **Geography input classification.** Enumerate every input the geography and
   allocator path consumes and classify each as prior-election information, fixed
   historical/legal metadata, or target-election realized information. The
   electorates file is additionally *perturbation-tested* to prove it does not
   enter chronological-mode output.
B. **Consensus publication-safety.** Confirm every retained poll satisfies
   ``publication_date <= election_date``, and quantify the polls that were
   fieldwork-eligible but excluded because they published after election day —
   the polls the leaky PoP state series would have absorbed.
C. **Archive revision test** on the individual-poll inputs, the same snapshot
   comparison that exposed the PoP series in Part 3B.
D. **Smoke test** of the full isolated path for all three targets. No predictive
   score against any certified outcome is computed.

Writes ``processed/isolated_path_audit.json``.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from scripts.election_layer_v2.config import CANONICAL_WINDOW_DAYS, MIN_SHARE_PCT
from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals
from scripts.election_layer_v2.transfer import apply_batch_simplex_transfer
from scripts.election_residuals.config import ALL_CATEGORIES, DEFAULT_POLLS_FILE
from scripts.election_residuals.consensus import build_election_polling_consensus
from scripts.geography.projection import project_constituency_votes
from scripts.mandates.allocator import allocate_riksdag_seats
from scripts.mandates.law import mandate_law_for_election_year
from scripts.simulator.config import MODEL_PARTIES_9, PARLIAMENTARY_PARTIES_8
from scripts.vote_share_calibration.models import derive_vote_share_layer_seeds

HERE = Path(__file__).resolve().parent
PROCESSED = HERE / "processed"
SCRATCH = HERE / "_scratch"
PART2B = REPO_ROOT / "diagnostics/election_noise_v2/historical_seat_extension/processed"
RESEARCH_GEO = PART2B / "research_geography"

TARGETS = {
    2014: {"election_date": date(2014, 9, 14), "baseline": 2010},
    2018: {"election_date": date(2018, 9, 9), "baseline": 2014},
    2022: {"election_date": date(2022, 9, 11), "baseline": 2018},
}
SMOKE_DRAWS = 400
SMOKE_SEED = 12345


# ----------------------------------------------------------------- probe A ----
def geography_input_classification() -> dict:
    """Enumerate and classify every input the geography + allocator path consumes."""
    inputs = [
        {
            "input": "baseline matrix B (29 x 9 constituency party votes)",
            "source": "constituency_party_votes_*.csv rows for the BASELINE election year",
            "used_for": "IPF seed structure",
            "classification": "prior-election information",
            "target_realized": False,
            "evidence": "_get_cached_geography_structures selects rows where election_year == baseline_year",
        },
        {
            "input": "target constituency row totals R",
            "source": "R = np.sum(B, axis=1) in chronological mode when target_year <= 2022",
            "used_for": "IPF row margins",
            "classification": "prior-election information",
            "target_realized": False,
            "evidence": "projection.py: 'Strictly chronological: row totals are derived entirely from baseline valid votes. Zero information from target election electorate or valid votes is accessed!'",
        },
        {
            "input": "total national valid votes",
            "source": "sum(R), i.e. the baseline election total, when total_national_votes=None",
            "used_for": "IPF column margin scale",
            "classification": "prior-election information",
            "target_realized": False,
            "evidence": "project_constituency_votes leaves R unscaled when total_national_votes is None",
        },
        {
            "input": "national vote shares C",
            "source": "the forecast draw itself (consensus + ElectionNoise)",
            "used_for": "IPF column margins",
            "classification": "forecast output, not an observation",
            "target_realized": False,
            "evidence": "passed in as national_vote_shares",
        },
        {
            "input": "constituency electorates file",
            "source": "constituency_electorates_*.csv",
            "used_for": "nothing in chronological mode with target_year <= 2022",
            "classification": "read but unused - proven by perturbation test",
            "target_realized": False,
            "evidence": "see perturbation_test below",
        },
        {
            "input": "fixed constituency seats for the target year",
            "source": "Valmyndigheten valkretsmandat workbook",
            "used_for": "allocator seat quota per constituency",
            "classification": "fixed historical/legal metadata, decided and published BEFORE the election (Vallagen 4 kap. 3 §: apportioned from eligible voters as of 1 March of the election year)",
            "target_realized": False,
            "evidence": "published pre-election; available at any forecast origin in the election year",
        },
        {
            "input": "mandate law version and first divisor",
            "source": "mandate_law_for_election_year(target_year)",
            "used_for": "allocation rules",
            "classification": "fixed historical/legal metadata",
            "target_realized": False,
            "evidence": "statutory; SFS 2014:1384 effective from the 2018 election",
        },
        {
            "input": "4% national and 12% constituency thresholds, 349/310/39 seat split",
            "source": "Regeringsformen 3 kap. 7 §, Vallagen 14 kap.",
            "used_for": "eligibility and allocation",
            "classification": "fixed historical/legal metadata",
            "target_realized": False,
            "evidence": "statutory constants",
        },
        {
            "input": "constituency code set (29 constituencies)",
            "source": "OFFICIAL_CONSTITUENCY_CODES",
            "used_for": "matrix indexing",
            "classification": "fixed historical/legal metadata; unchanged since 1998",
            "target_realized": False,
            "evidence": "Valmyndigheten workbook footnotes place changes at 1994, 1998, 2006, 2018 only",
        },
        {
            "input": "deterministic tie-break seed",
            "source": "DeterministicLotteryTieBreaker(seed=12345)",
            "used_for": "statutory lottery",
            "classification": "fixed model convention",
            "target_realized": False,
            "evidence": "constant",
        },
        {
            "input": "oracle-mode target valid votes",
            "source": "constituency_electorates_*.csv rows for the TARGET year",
            "used_for": "IPF row margins - ONLY in mode='oracle'",
            "classification": "TARGET-ELECTION REALIZED - therefore oracle mode is prohibited on this path",
            "target_realized": True,
            "evidence": "_get_cached_geography_structures branch for mode == 'oracle'",
        },
    ]
    return {
        "mode_used": "chronological",
        "oracle_mode_prohibited": True,
        "inputs": inputs,
        "any_target_realized_input_in_chronological_mode": any(
            i["target_realized"] for i in inputs if "oracle" not in i["input"]
        ),
    }


def electorates_perturbation_test() -> dict:
    """Prove the electorates file does not enter chronological-mode output."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    pert = SCRATCH / "geo_perturbed"
    if pert.exists():
        shutil.rmtree(pert)
    pert.mkdir(parents=True)
    shutil.copy(RESEARCH_GEO / "constituency_party_votes_2014_2022.csv", pert)
    el = pd.read_csv(RESEARCH_GEO / "constituency_electorates_2014_2026.csv")
    for col in ("eligible_voters", "valid_votes"):
        if col in el.columns:
            el[col] = pd.to_numeric(el[col], errors="coerce") * 7.77 + 12345  # gross perturbation
    el.to_csv(pert / "constituency_electorates_2014_2026.csv", index=False)

    shares = {p: v for p, v in zip(MODEL_PARTIES_9, [0.20, 0.05, 0.07, 0.06, 0.30, 0.07, 0.06, 0.17, 0.02])}
    out = {}
    for target, spec in TARGETS.items():
        a = project_constituency_votes(
            national_vote_shares=shares, baseline_year=spec["baseline"], target_year=target,
            mode="chronological", total_national_votes=None, processed_dir=RESEARCH_GEO,
        )
        b = project_constituency_votes(
            national_vote_shares=shares, baseline_year=spec["baseline"], target_year=target,
            mode="chronological", total_national_votes=None, processed_dir=pert,
        )
        same = a.constituency_votes == b.constituency_votes and a.constituency_valid_votes == b.constituency_valid_votes
        out[str(target)] = {
            "identical_under_grossly_perturbed_electorates": bool(same),
            "constituency_valid_votes_total": int(sum(a.constituency_valid_votes.values())),
        }
    return {
        "perturbation": "eligible_voters and valid_votes multiplied by 7.77 and offset by 12345",
        "per_target": out,
        "electorates_file_proven_unused": all(
            v["identical_under_grossly_perturbed_electorates"] for v in out.values()
        ),
    }


# ----------------------------------------------------------------- probe B ----
def consensus_publication_safety() -> dict:
    polls = pd.read_csv(DEFAULT_POLLS_FILE)
    per_poll = polls.drop_duplicates("poll_id")[
        ["poll_id", "pollster", "interview_start", "interview_end", "publication_date"]
    ]
    out = {}
    for target, spec in TARGETS.items():
        ed = spec["election_date"]
        eds = ed.isoformat()
        cons = build_election_polling_consensus(ed, polls, window_days=CANONICAL_WINDOW_DAYS)
        retained = [
            {
                "pollster": p.pollster,
                "interview_end": p.interview_end.isoformat(),
                "publication_date": p.publication_date.isoformat(),
                "published_on_or_before_election": p.publication_date <= ed,
                "fieldwork_ended_on_or_before_election": p.interview_end <= ed,
                "sample_size": p.sample_size,
            }
            for p in cons.contributing_polls
        ]
        ws = (ed - timedelta(days=CANONICAL_WINDOW_DAYS)).isoformat()
        # Polls that were fieldwork-eligible but published after election day:
        # exactly the information a fieldwork-dated state series would absorb.
        excluded = per_poll[
            (per_poll.interview_end >= ws)
            & (per_poll.interview_end <= eds)
            & (per_poll.publication_date > eds)
        ]
        out[str(target)] = {
            "election_date": eds,
            "window": [ws, eds],
            "retained_pollsters": cons.retained_pollsters_count,
            "eligible_polls_in_window": cons.total_eligible_polls_in_window,
            "all_retained_published_on_or_before_election": all(
                r["published_on_or_before_election"] for r in retained
            ),
            "all_retained_fieldwork_ended_on_or_before_election": all(
                r["fieldwork_ended_on_or_before_election"] for r in retained
            ),
            "consensus_composition": {c: cons.consensus_composition[c] for c in ALL_CATEGORIES},
            "consensus_sums_to_100": abs(sum(cons.consensus_composition.values()) - 100.0) < 1e-6,
            "excluded_because_published_after_election_day": int(len(excluded)),
            "excluded_detail": [
                {
                    "pollster": r.pollster,
                    "interview_end": str(r.interview_end),
                    "publication_date": str(r.publication_date),
                    "days_after_election": (
                        date.fromisoformat(str(r.publication_date)) - ed
                    ).days,
                }
                for _, r in excluded.iterrows()
            ],
            "retained": retained,
        }
    out["all_targets_publication_safe"] = all(
        v["all_retained_published_on_or_before_election"]
        and v["all_retained_fieldwork_ended_on_or_before_election"]
        for k, v in out.items()
        if k.isdigit()
    )
    return out


# ----------------------------------------------------------------- probe C ----
def poll_archive_revision_test() -> dict:
    """Same snapshot comparison that exposed the PoP series, applied to the poll archive."""
    path = "data/processed/pollofpolls/swedishpolls_individual_polls.csv"
    results = []
    for a, b in (("f55bf36", "f6ae4d1"), ("f6ae4d1", "34c52d6")):
        def load(commit: str) -> dict[tuple, float]:
            txt = subprocess.check_output(["git", "show", f"{commit}:{path}"]).decode("utf-8", "replace")
            r = csv.DictReader(io.StringIO(txt))
            out = {}
            for row in r:
                pub = (row.get("publication_date") or "").strip()
                if not pub or pub > "2022-12-31":
                    continue  # restrict to the historical region the isolated path uses
                key = (row.get("poll_id"), row.get("party"))
                v = (row.get("support") or "").strip()
                if v not in ("", "NaN"):
                    out[key] = float(v)
            return out
        A, B = load(a), load(b)
        common = set(A) & set(B)
        changed = [k for k in common if abs(A[k] - B[k]) > 1e-9]
        results.append(
            {
                "file": path,
                "snapshot_a": a,
                "snapshot_b": b,
                "restricted_to_publication_date": "<= 2022-12-31",
                "keys_in_a": len(A),
                "keys_in_b": len(B),
                "common_keys": len(common),
                "keys_only_in_a": len(set(A) - set(B)),
                "keys_only_in_b": len(set(B) - set(A)),
                "revised_support_values": len(changed),
            }
        )
    return {
        "probes": results,
        "no_historical_support_value_revised": all(r["revised_support_values"] == 0 for r in results),
        "note": (
            "The consensus filters on publication_date <= election_date, so even an archive "
            "assembled later cannot inject a poll that was unpublished at the forecast origin. "
            "This is the structural difference from the PoP state series, which carries no "
            "publication filter."
        ),
    }


# ----------------------------------------------------------------- probe D ----
def smoke_isolated_path() -> dict:
    polls = pd.read_csv(DEFAULT_POLLS_FILE)
    fixed_all = json.loads((PART2B / "fixed_seats_by_year.json").read_text())["fixed_seats_by_year"]
    cert = {}
    df14 = pd.read_csv(PART2B / "certified_mandates_2010_2014.csv")
    df_prod = pd.read_csv(REPO_ROOT / "data/processed/mandates/historical_certified_mandates.csv")
    for y in TARGETS:
        src = df14 if y == 2014 else df_prod
        s = src[src["election_year"] == y]
        cert[y] = {p: int(s[s["party"] == p]["total_seats"].sum()) for p in PARLIAMENTARY_PARTIES_8}

    out = {}
    for target, spec in TARGETS.items():
        ed = spec["election_date"]
        cons = build_election_polling_consensus(ed, polls, window_days=CANONICAL_WINDOW_DAYS)
        base = np.array([cons.consensus_composition[c] for c in ALL_CATEGORIES], dtype=float)
        pool = load_chronological_pp_residuals(target_election_year=target)
        k = len(pool.training_years)
        idx_seed, _ = derive_vote_share_layer_seeds(
            base_seed=SMOKE_SEED, origin_date=ed, horizon_days=CANONICAL_WINDOW_DAYS
        )
        idx = np.random.default_rng(idx_seed).integers(0, k, size=SMOKE_DRAWS)
        votes_pct, lambdas = apply_batch_simplex_transfer(
            np.tile(base, (SMOKE_DRAWS, 1)), pool.centered_residuals_matrix[idx], eps=MIN_SHARE_PCT
        )
        cfg = mandate_law_for_election_year(target)
        fixed = {kk: int(vv) for kk, vv in fixed_all[str(target)].items()}

        seats = []
        for i in range(SMOKE_DRAWS):
            shares = {p: float(votes_pct[i, j] / 100.0) for j, p in enumerate(MODEL_PARTIES_9)}
            proj = project_constituency_votes(
                national_vote_shares=shares,
                baseline_year=spec["baseline"],
                target_year=target,
                mode="chronological",
                total_national_votes=None,
                processed_dir=RESEARCH_GEO,
            )
            alloc = allocate_riksdag_seats(
                proj.to_allocator_input(), fixed,
                first_divisor=cfg.first_divisor, law=cfg.law,
                scenario_id=f"isolated_{target}_{i}",
            )
            seats.append([alloc.final_seats_by_party.get(p, 0) for p in PARLIAMENTARY_PARTIES_8])
        S = np.array(seats, dtype=np.int64)
        out[str(target)] = {
            "election_date": ed.isoformat(),
            "geography_baseline_year": spec["baseline"],
            "mandate_law": cfg.law.value,
            "first_divisor": str(cfg.first_divisor),
            "training_residual_years": [int(y) for y in pool.training_years],
            "k_outer": k,
            "no_future_year_in_pool": all(int(y) < target for y in pool.training_years),
            "retained_pollsters": cons.retained_pollsters_count,
            "draws": SMOKE_DRAWS,
            "distinct_support_points_in_votes": int(np.unique(np.round(votes_pct, 10), axis=0).shape[0]),
            "vote_rows_sum_to_100": bool(np.allclose(votes_pct.sum(axis=1), 100.0, atol=1e-9)),
            "mean_lambda": float(np.mean(lambdas)),
            "all_seat_totals_349": bool(np.all(S.sum(axis=1) == 349)),
            "distinct_seat_vectors": int(np.unique(S, axis=0).shape[0]),
            "certified_truth_seats": cert[target],
            "certified_sums_to_349": sum(cert[target].values()) == 349,
            "runs": True,
        }
    out["all_three_targets_run"] = all(v.get("runs") for k, v in out.items() if k.isdigit())
    return out


def main() -> int:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "AUDIT ONLY - no predictive score against any certified outcome is computed",
        "path_under_audit": (
            "final 14-day polling consensus -> ElectionNoise -> unchanged simplex transfer "
            "-> frozen geography (chronological) -> historically correct mandate law -> joint seat draws"
        ),
        "probe_a_geography_inputs": geography_input_classification(),
        "probe_a_electorates_perturbation": electorates_perturbation_test(),
        "probe_b_consensus_publication_safety": consensus_publication_safety(),
        "probe_c_poll_archive_revision": poll_archive_revision_test(),
        "probe_d_smoke": smoke_isolated_path(),
    }
    report["verdict"] = {
        "no_target_realized_geography_input": not report["probe_a_geography_inputs"][
            "any_target_realized_input_in_chronological_mode"
        ],
        "electorates_file_proven_unused": report["probe_a_electorates_perturbation"][
            "electorates_file_proven_unused"
        ],
        "consensus_publication_safe": report["probe_b_consensus_publication_safety"][
            "all_targets_publication_safe"
        ],
        "poll_archive_not_retrospectively_revised": report["probe_c_poll_archive_revision"][
            "no_historical_support_value_revised"
        ],
        "all_three_targets_run": report["probe_d_smoke"]["all_three_targets_run"],
    }
    report["isolated_path_valid"] = all(report["verdict"].values())
    (PROCESSED / "isolated_path_audit.json").write_text(json.dumps(report, indent=2) + "\n")

    print("probe A: target-realized geography input in chronological mode:",
          report["probe_a_geography_inputs"]["any_target_realized_input_in_chronological_mode"])
    print("probe A: electorates file proven unused:",
          report["probe_a_electorates_perturbation"]["electorates_file_proven_unused"])
    b = report["probe_b_consensus_publication_safety"]
    for y in ("2014", "2018", "2022"):
        v = b[y]
        print(f"probe B: {y} retained={v['retained_pollsters']} pollsters, all published<=E={v['all_retained_published_on_or_before_election']}, "
              f"excluded because published after E: {v['excluded_because_published_after_election_day']}")
    print("probe C: no historical support value revised:",
          report["probe_c_poll_archive_revision"]["no_historical_support_value_revised"])
    d = report["probe_d_smoke"]
    for y in ("2014", "2018", "2022"):
        v = d[y]
        print(f"probe D: {y} law={v['mandate_law']} K={v['k_outer']} pool={v['training_residual_years']} "
              f"support_points={v['distinct_support_points_in_votes']} seats349={v['all_seat_totals_349']} "
              f"distinct_seat_vectors={v['distinct_seat_vectors']}")
    print(f"\nISOLATED PATH VALID: {report['isolated_path_valid']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
