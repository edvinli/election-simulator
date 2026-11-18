"""Evidence for election-noise residual amendment 002 (polls missing a pivot key).

Compares, for every residual election 2002-2026, the consensus under
amendment 001 alone with the consensus under amendments 001 and 002:

* 001 alone: the committed builder fed amendment 001's eligible rows, as in
  ``verify_eve_rule.py``. Its pivot drops any poll missing ``sample_size`` or
  ``interview_start``.
* 001 + 002: ``scripts/election_residuals/eve_consensus.py``.

It changes nothing, and writes ``results_002.json`` beside this file
(``results.json``, the evidence of amendment 001, is not touched). Run with
``uv run python -m diagnostics.election_noise_residual_eve_rule.verify_missing_keys``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.election_residuals.config import ALL_CATEGORIES, DEFAULT_ELECTIONS_FILE, DEFAULT_POLLS_FILE
from scripts.election_residuals.consensus import build_election_polling_consensus
from scripts.election_residuals.eve_consensus import build_eve_polling_consensus, eligible_eve_rows
from scripts.elections.load import load_election_targets_for_forecasting

from .verify_eve_rule import ELECTIONS, MANIFEST_2026, ROOT, pool_summary, sha256, truth_for

OUTPUT = Path(__file__).resolve().parent / "results_002.json"


def summarize(consensus, truth: dict[str, float]) -> tuple[dict, np.ndarray]:
    comp = np.array([consensus.consensus_composition[k] for k in ALL_CATEGORIES])
    resid = np.array([truth[k] for k in ALL_CATEGORIES]) - comp
    resid = resid - resid.sum() / len(resid)
    return {
        "contributing_polls": [
            {"poll_id": p.poll_id, "pollster": p.pollster, "interview_end": p.interview_end.isoformat(),
             "publication_date": p.publication_date.isoformat(), "sample_size": p.sample_size, "weight": p.weight}
            for p in consensus.contributing_polls
        ],
        "consensus_pct": {k: round(float(v), 4) for k, v in zip(ALL_CATEGORIES, comp)},
        "residual_pp": {k: round(float(v), 4) for k, v in zip(ALL_CATEGORIES, resid)},
        "residual_norm_pp": round(float(np.linalg.norm(resid)), 4),
    }, resid


def main() -> int:
    polls = pd.read_csv(DEFAULT_POLLS_FILE)
    targets = load_election_targets_for_forecasting(DEFAULT_ELECTIONS_FILE)
    elections, residuals = [], {"001": [], "001+002": []}
    for election in ELECTIONS:
        truth = truth_for(election, targets)
        only_001, r1 = summarize(build_election_polling_consensus(election, eligible_eve_rows(election, polls), window_days=14), truth)
        both, r2 = summarize(build_eve_polling_consensus(election, polls, window_days=14), truth)
        residuals["001"].append(r1)
        residuals["001+002"].append(r2)
        eligible = eligible_eve_rows(election, polls).drop_duplicates("poll_id")
        missing = eligible[eligible["sample_size"].isna() | eligible["interview_start"].isna()]
        elections.append({
            "election": election.isoformat(),
            "eligible_polls_missing_a_key": [
                {"poll_id": r.poll_id, "pollster": r.pollster, "interview_start": None if pd.isna(r.interview_start) else r.interview_start,
                 "interview_end": r.interview_end, "sample_size": None if pd.isna(r.sample_size) else int(r.sample_size)}
                for r in missing.itertuples()
            ],
            "consensus_changed_by_002": only_001["consensus_pct"] != both["consensus_pct"],
            "amendment_001_only": only_001,
            "amendments_001_002": both,
        })

    record = {
        "purpose": "Evidence for docs/election_noise_residual_amendment_002.md; changes nothing.",
        "inputs_sha256": {
            "data/processed/pollofpolls/swedishpolls_individual_polls.csv": sha256(DEFAULT_POLLS_FILE),
            "data/processed/elections/riksdag_election_results.csv": sha256(DEFAULT_ELECTIONS_FILE),
            "data/raw/elections/val2026/official_result_manifest.json": sha256(MANIFEST_2026),
            "scripts/election_residuals/consensus.py": sha256(ROOT / "scripts/election_residuals/consensus.py"),
            "scripts/election_residuals/eve_consensus.py": sha256(ROOT / "scripts/election_residuals/eve_consensus.py"),
        },
        "elections_changed_by_002": [r["election"] for r in elections if r["consensus_changed_by_002"]],
        "pools": {
            rule: {"2002-2022": pool_summary(np.array(res[:6])), "2002-2026": pool_summary(np.array(res))}
            for rule, res in residuals.items()
        },
        "elections": elections,
    }
    OUTPUT.write_text(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print("changed by 002:", record["elections_changed_by_002"])
    for rule in residuals:
        for span, s in record["pools"][rule].items():
            print(rule, span, "total variance", s["total_variance_pp2"], "sd", s["noise_sd_pp"])
    for r in elections:
        if r["consensus_changed_by_002"]:
            a, b = r["amendment_001_only"], r["amendments_001_002"]
            print(r["election"], "norm", a["residual_norm_pp"], "->", b["residual_norm_pp"],
                  {k: (a["residual_pp"][k], b["residual_pp"][k]) for k in ALL_CATEGORIES if abs(a["residual_pp"][k] - b["residual_pp"][k]) >= 0.005})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
