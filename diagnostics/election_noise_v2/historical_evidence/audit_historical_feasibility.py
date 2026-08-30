"""Part 2 historical-data feasibility audit (research only).

Reproduces every empirical claim in ``docs/election_noise_v2_historical_evidence.md``.

This module is read-only with respect to production: it imports the frozen
production residual/consensus code and applies it unchanged. It never writes to
``data/``, never adds a residual to the production pool, and never simulates.

Outputs ``findings.json`` next to this file.
"""

from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals
from scripts.election_residuals.config import (
    ALL_CATEGORIES,
    DEFAULT_POLLS_FILE,
    EVALUATION_ELECTIONS,
    LOOKBACK_WINDOW_DAYS,
    PARLIAMENTARY_PARTIES,
)
from scripts.election_residuals.consensus import build_election_polling_consensus
from scripts.elections.load import load_election_targets_for_forecasting

HERE = Path(__file__).resolve().parent

# Candidate pre-2002 elections, with official dates from the repository's own
# threshold-events configuration (scripts/threshold_events/config.py).
CANDIDATE_PRE_2002 = {
    1991: date(1991, 9, 15),
    1994: date(1994, 9, 18),
    1998: date(1998, 9, 20),
}

# Frozen Tier-1 candidate target set (preregistration §E.2, Amendment 1).
TIER1_CANDIDATE_TARGETS = (2010, 2014, 2018, 2022)
K_OUTER_MIN = 3


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def window_eligibility(polls: pd.DataFrame, election_date: date, window_days: int) -> dict:
    """Apply the production eligibility filter verbatim and report what survives."""
    ws = (election_date - timedelta(days=window_days)).isoformat()
    we = election_date.isoformat()
    sub = polls[
        (polls["interview_end"] >= ws)
        & (polls["interview_end"] <= we)
        & (polls["publication_date"] <= we)
    ]
    return {
        "window_start": ws,
        "window_end": we,
        "eligible_polls": int(sub["poll_id"].nunique()),
        "distinct_pollsters": int(sub["pollster"].nunique()),
        "pollsters": sorted(sub["pollster"].unique().tolist()),
    }


def metadata_census(polls: pd.DataFrame, election_date: date, lookback_days: int = 120) -> dict:
    """Census of every poll published in the run-up, and its metadata completeness."""
    lo = (election_date - timedelta(days=lookback_days)).isoformat()
    hi = election_date.isoformat()
    ids = polls[(polls["publication_date"] >= lo) & (polls["publication_date"] <= hi)]["poll_id"].unique()
    sub = polls[polls["poll_id"].isin(ids)]
    per_poll = sub.groupby("poll_id").agg(
        pollster=("pollster", "first"),
        interview_start=("interview_start", "first"),
        interview_end=("interview_end", "first"),
        publication_date=("publication_date", "first"),
        sample_size=("sample_size", "first"),
    )
    missing_end = per_poll["interview_end"].isna()
    missing_n = per_poll["sample_size"].isna()
    # Fieldwork dated after publication is physically impossible -> corrupt record.
    corrupt = per_poll.apply(
        lambda r: bool(
            pd.notnull(r["interview_end"])
            and pd.notnull(r["publication_date"])
            and str(r["interview_end"]) > str(r["publication_date"])
        ),
        axis=1,
    )
    recs = []
    for pid, r in per_poll.iterrows():
        ie = r["interview_end"]
        recs.append(
            {
                "pollster": str(r["pollster"]),
                "interview_start": None if pd.isna(r["interview_start"]) else str(r["interview_start"]),
                "interview_end": None if pd.isna(ie) else str(ie),
                "publication_date": str(r["publication_date"]),
                "sample_size": None if pd.isna(r["sample_size"]) else int(r["sample_size"]),
                "days_before_election": (
                    None if pd.isna(ie) else (election_date - date.fromisoformat(str(ie))).days
                ),
                "interview_end_after_publication": bool(corrupt.loc[pid]),
            }
        )
    recs.sort(key=lambda d: d["publication_date"])
    return {
        "lookback_days": lookback_days,
        "polls_published_in_lookback": int(len(per_poll)),
        "missing_interview_end": int(missing_end.sum()),
        "missing_sample_size": int(missing_n.sum()),
        "corrupt_interview_end_after_publication": int(corrupt.sum()),
        "polls": recs,
    }


def existing_pool_profile(polls: pd.DataFrame) -> list[dict]:
    """Track C: descriptive comparability attributes of the frozen residual pool."""
    targets = load_election_targets_for_forecasting()
    pool = load_chronological_pp_residuals(target_election_year=2026)
    out = []
    for ed in EVALUATION_ELECTIONS:
        c = build_election_polling_consensus(ed, polls, window_days=LOOKBACK_WINDOW_DAYS)
        cp = c.contributing_polls
        ages = [(ed - p.interview_end).days for p in cp]
        ns = [p.sample_size for p in cp if p.sample_size]
        j = pool.training_years.index(ed.year)
        r = pool.residuals_matrix[j]
        unpolled = [
            p for p in PARLIAMENTARY_PARTIES if not any(p in q.party_support for q in cp)
        ]
        out.append(
            {
                "year": ed.year,
                "election_date": ed.isoformat(),
                "retained_pollsters": c.retained_pollsters_count,
                "eligible_polls_in_window": c.total_eligible_polls_in_window,
                "pollsters": sorted({p.pollster for p in cp}),
                "summed_sample_size": int(sum(ns)) if ns else 0,
                "polls_missing_sample_size": sum(1 for p in cp if p.sample_size_missing),
                "mean_consensus_age_days": round(float(np.mean(ages)), 3),
                "min_consensus_age_days": int(min(ages)),
                "max_consensus_age_days": int(max(ages)),
                "structurally_unpolled_parties": unpolled,
                "unpolled_party_official_pct": {
                    p: round(targets[ed][p], 4) for p in unpolled
                },
                "mean_abs_residual_pp": round(float(np.mean(np.abs(r))), 4),
                "max_abs_residual_pp": round(float(np.max(np.abs(r))), 4),
                "residual_l2_norm_pp": round(float(np.linalg.norm(r)), 4),
                "centered_residual_l2_norm_pp": round(
                    float(np.linalg.norm(pool.centered_residuals_matrix[j])), 4
                ),
            }
        )
    return out


def main() -> int:
    polls = pd.read_csv(DEFAULT_POLLS_FILE)

    findings: dict = {
        "purpose": "Part 2 historical-data feasibility audit (research only)",
        "preregistration_freeze_commit": "80b1c671c4b6d879a888f28a859ee392e8f59bc5",
        "preregistration_body_sha256": (
            "bac3ca06e52cc07fe74ca9e5aa785d94e30934db32193c7f948e95a49a6ae075"
        ),
        "production_window_days": LOOKBACK_WINDOW_DAYS,
        "polls_file": str(DEFAULT_POLLS_FILE.relative_to(REPO_ROOT)),
        "polls_file_sha256": sha256_file(DEFAULT_POLLS_FILE),
        "polls_file_coverage": {
            "rows": int(len(polls)),
            "distinct_polls": int(polls["poll_id"].nunique()),
            "earliest_interview_end": str(polls["interview_end"].min()),
            "latest_interview_end": str(polls["interview_end"].max()),
        },
    }

    # ---- Track A: candidate pre-2002 elections -------------------------------
    track_a: dict = {}
    for year, ed in CANDIDATE_PRE_2002.items():
        track_a[str(year)] = {
            "election_date": ed.isoformat(),
            "canonical_window": window_eligibility(polls, ed, LOOKBACK_WINDOW_DAYS),
            "metadata_census": metadata_census(polls, ed),
        }
    findings["track_a_candidate_elections"] = track_a

    # Existing pool, same filter, for a like-for-like comparison.
    findings["track_a_incumbent_pool"] = {
        str(ed.year): window_eligibility(polls, ed, LOOKBACK_WINDOW_DAYS)
        for ed in EVALUATION_ELECTIONS
    }

    # ---- Mechanical consequence tables ---------------------------------------
    accepted_new_residual_years: list[int] = []  # populated only by an ACCEPT verdict
    pool = load_chronological_pp_residuals(target_election_year=2026)
    residual_years = sorted(set(pool.training_years) | set(accepted_new_residual_years))
    findings["residual_pool"] = {
        "production_years": list(pool.training_years),
        "accepted_new_years": accepted_new_residual_years,
        "resulting_years": residual_years,
        "K": len(residual_years),
    }

    tier1 = []
    for t in TIER1_CANDIDATE_TARGETS:
        prior = [y for y in residual_years if y < t]
        tier1.append(
            {
                "target": t,
                "prior_accepted_residual_years": prior,
                "K_outer": len(prior),
                "eligible": len(prior) >= K_OUTER_MIN,
            }
        )
    findings["tier1"] = {
        "candidate_targets": list(TIER1_CANDIDATE_TARGETS),
        "K_outer_min": K_OUTER_MIN,
        "rows": tier1,
        "N_T1": sum(1 for r in tier1 if r["eligible"]),
    }

    # ---- Track C -------------------------------------------------------------
    profile = existing_pool_profile(polls)
    findings["track_c_pool_profile"] = profile
    yrs = np.array([p["year"] for p in profile], dtype=float)
    l2 = np.array([p["residual_l2_norm_pp"] for p in profile], dtype=float)
    order_y = np.argsort(np.argsort(yrs))
    order_l = np.argsort(np.argsort(l2))
    findings["track_c_descriptive_trend"] = {
        "n": int(len(yrs)),
        "pearson_year_vs_residual_l2": round(float(np.corrcoef(yrs, l2)[0, 1]), 4),
        "spearman_year_vs_residual_l2": round(
            float(np.corrcoef(order_y, order_l)[0, 1]), 4
        ),
        "note": (
            "Descriptive only, n=6. No trend is fitted, no recency weighting is "
            "implied, and no residual is altered."
        ),
    }

    out = HERE / "findings.json"
    out.write_text(json.dumps(findings, indent=2, sort_keys=False) + "\n")
    print(f"wrote {out}")
    print(json.dumps(findings["tier1"], indent=2))
    print(json.dumps(findings["residual_pool"], indent=2))
    for y, v in track_a.items():
        cw = v["canonical_window"]
        mc = v["metadata_census"]
        print(
            f"{y}: eligible_polls={cw['eligible_polls']} pollsters={cw['distinct_pollsters']} "
            f"| lookback polls={mc['polls_published_in_lookback']} "
            f"missing_interview_end={mc['missing_interview_end']} "
            f"missing_n={mc['missing_sample_size']} "
            f"corrupt_dates={mc['corrupt_interview_end_after_publication']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
