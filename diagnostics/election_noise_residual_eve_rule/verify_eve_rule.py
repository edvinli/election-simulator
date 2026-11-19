"""Evidence for election-noise residual amendment 001 (the election-eve information set).

For every residual election 2002-2026 this compares three eligibility rules for
the final-poll consensus that election-noise residuals are measured against:

* ``current``: ``scripts/election_residuals/consensus.py`` as committed:
  publication_date <= E and interview_end <= E, with interview_end >= E - 14.
* ``eve``: the amended rule, publication_date < E and interview_end < E, with
  the same window lower bound. It is applied here as a pre-filter in front of
  the unchanged consensus builder, so everything else (latest poll per
  pollster, tie-break, weights, REST) is byte-for-byte the committed code.
* ``aggregate``: the exclusion ``scripts/poll_aggregate/data.py`` uses: drop a
  poll whose fieldwork interval contains election day (start <= E <= end),
  keep publication_date <= E. Recorded only to show where it differs from
  ``eve``; it is not a candidate rule.

It changes nothing. Run with
``uv run python -m diagnostics.election_noise_residual_eve_rule.verify_eve_rule``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.election_layer_v2.config import CANONICAL_WINDOW_DAYS
from scripts.election_residuals.config import ALL_CATEGORIES, DEFAULT_ELECTIONS_FILE, DEFAULT_POLLS_FILE
from scripts.election_residuals.consensus import build_election_polling_consensus
from scripts.elections.load import load_election_targets_for_forecasting
from scripts.vote_share_calibration.election_noise_b import fit_election_noise_b

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_2026 = ROOT / "data" / "raw" / "elections" / "val2026" / "official_result_manifest.json"
OUTPUT = Path(__file__).resolve().parent / "results.json"
ELECTIONS = (
    date(2002, 9, 15), date(2006, 9, 17), date(2010, 9, 19), date(2014, 9, 14),
    date(2018, 9, 9), date(2022, 9, 11), date(2026, 9, 13),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def truth_for(election: date, targets: dict) -> dict[str, float]:
    if election.year != 2026:
        return dict(targets[election])
    manifest = json.loads(MANIFEST_2026.read_text(encoding="utf-8"))
    shares = {p: float(v["vote_share_percentage_points"]) for p, v in manifest["parties"].items()}
    shares["REST"] = 100.0 - sum(shares.values())
    return shares


def eve_filter(polls: pd.DataFrame, election: date) -> pd.DataFrame:
    e = election.isoformat()
    return polls[(polls["publication_date"] < e) & (polls["interview_end"] < e)]


def aggregate_filter(polls: pd.DataFrame, election: date) -> pd.DataFrame:
    e = election.isoformat()
    contains = polls["interview_start"].notna() & (polls["interview_start"] <= e) & (polls["interview_end"] >= e)
    return polls[~contains]


def consensus_record(election: date, polls: pd.DataFrame, truth: dict[str, float]) -> dict:
    c = build_election_polling_consensus(election, polls, window_days=CANONICAL_WINDOW_DAYS)
    comp = np.array([c.consensus_composition[k] for k in ALL_CATEGORIES])
    resid = np.array([truth[k] for k in ALL_CATEGORIES]) - comp
    resid = resid - resid.sum() / len(resid)
    return {
        "eligible_polls_in_window": c.total_eligible_polls_in_window,
        "contributing_polls": [
            {
                "poll_id": p.poll_id, "pollster": p.pollster,
                "interview_start": p.interview_start.isoformat() if p.interview_start else None,
                "interview_end": p.interview_end.isoformat(),
                "publication_date": p.publication_date.isoformat(),
                "sample_size": p.sample_size, "weight": p.weight,
            }
            for p in c.contributing_polls
        ],
        "consensus_pct": {k: round(float(v), 4) for k, v in zip(ALL_CATEGORIES, comp)},
        "residual_pp": {k: round(float(v), 4) for k, v in zip(ALL_CATEGORIES, resid)},
        "residual_norm_pp": round(float(np.linalg.norm(resid)), 4),
        "_residual": resid,
    }


def pool_summary(residuals: np.ndarray) -> dict:
    centered = residuals - residuals.mean(axis=0)
    fit = fit_election_noise_b(centered)
    sigma = fit.delta * fit.target + (1.0 - fit.delta) * fit.s_p
    return {
        "k": int(residuals.shape[0]),
        "noise_sd_pp": {k: round(float(v), 4) for k, v in zip(ALL_CATEGORIES, np.sqrt(np.diag(sigma)))},
        "total_variance_pp2": round(float(np.trace(sigma)), 4),
        "mean_bias_pp": {k: round(float(v), 4) for k, v in zip(ALL_CATEGORIES, residuals.mean(axis=0))},
        "lw_delta": round(float(fit.delta), 6),
    }


def main() -> int:
    polls = pd.read_csv(DEFAULT_POLLS_FILE)
    targets = load_election_targets_for_forecasting(DEFAULT_ELECTIONS_FILE)
    window = pd.Timedelta(days=CANONICAL_WINDOW_DAYS)
    elections, residuals = [], {"current": [], "eve": []}
    for election in ELECTIONS:
        truth = truth_for(election, targets)
        current = consensus_record(election, polls, truth)
        eve = consensus_record(election, eve_filter(polls, election), truth)
        aggregate = consensus_record(election, aggregate_filter(polls, election), truth)
        residuals["current"].append(current.pop("_residual"))
        residuals["eve"].append(eve.pop("_residual"))
        aggregate.pop("_residual")

        e, lo = election.isoformat(), (pd.Timestamp(election) - window).date().isoformat()
        in_window = polls[(polls["interview_end"] >= lo) & (polls["interview_end"] <= e) & (polls["publication_date"] <= e)]
        wide = in_window.drop_duplicates("poll_id")
        removed = wide[(wide["publication_date"] >= e) | (wide["interview_end"] >= e)]
        # Polls the eve rule drops but the aggregate's exclusion would keep:
        # published on election day with fieldwork ending before it.
        published_on_day = wide[(wide["publication_date"] >= e) & (wide["interview_end"] < e)]
        elections.append({
            "election": e,
            "removed_by_eve_rule": [
                {"poll_id": r.poll_id, "pollster": r.pollster, "interview_start": r.interview_start,
                 "interview_end": r.interview_end, "publication_date": r.publication_date}
                for r in removed.itertuples()
            ],
            "eve_rule_identical_to_current": eve["contributing_polls"] == current["contributing_polls"]
            and eve["consensus_pct"] == current["consensus_pct"],
            "eve_rule_identical_to_aggregate_exclusion": eve["contributing_polls"] == aggregate["contributing_polls"],
            "published_on_election_day_with_fieldwork_ending_before": published_on_day["poll_id"].tolist(),
            "current": current,
            "eve": eve,
        })

    by_year = {r["election"][:4]: r for r in elections}
    record = {
        "purpose": "Evidence for docs/election_noise_residual_amendment_001.md; changes nothing.",
        "inputs_sha256": {
            "data/processed/pollofpolls/swedishpolls_individual_polls.csv": sha256(DEFAULT_POLLS_FILE),
            "data/processed/elections/riksdag_election_results.csv": sha256(DEFAULT_ELECTIONS_FILE),
            "data/raw/elections/val2026/official_result_manifest.json": sha256(MANIFEST_2026),
            "scripts/election_residuals/consensus.py": sha256(ROOT / "scripts/election_residuals/consensus.py"),
        },
        "window_days": CANONICAL_WINDOW_DAYS,
        "checks": {
            "2002_identical_under_eve_rule": by_year["2002"]["eve_rule_identical_to_current"]
            and not by_year["2002"]["removed_by_eve_rule"],
            "elections_changed_by_eve_rule": [r["election"] for r in elections if not r["eve_rule_identical_to_current"]],
            "eve_rule_equals_aggregate_exclusion_for_all": all(r["eve_rule_identical_to_aggregate_exclusion"] for r in elections),
            "any_poll_published_on_election_day_with_earlier_fieldwork": any(
                r["published_on_election_day_with_fieldwork_ending_before"] for r in elections
            ),
        },
        "pools": {
            rule: {
                "2002-2022": pool_summary(np.array(residuals[rule][:6])),
                "2002-2026": pool_summary(np.array(residuals[rule])),
            }
            for rule in ("current", "eve")
        },
        "elections": elections,
    }
    OUTPUT.write_text(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(record["checks"], indent=2))
    for rule in ("current", "eve"):
        for span, s in record["pools"][rule].items():
            print(rule, span, "total variance", s["total_variance_pp2"])
    for r in elections:
        print(r["election"], "removed:", [(p["pollster"], p["interview_end"], p["publication_date"]) for p in r["removed_by_eve_rule"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
