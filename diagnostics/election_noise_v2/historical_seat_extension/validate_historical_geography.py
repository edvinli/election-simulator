"""Run the frozen geography model chronologically on the two new historical chains.

    2006 baseline -> actual 2010 national vote  -> PRE_2018 allocator
    2010 baseline -> actual 2014 national vote  -> PRE_2018 allocator

The geography model itself is untouched: ``project_constituency_votes`` is imported
and called unchanged. Only its ``processed_dir`` is redirected to a research-only
directory that contains the production CSVs **plus** the newly normalized 2006 and
2010 rows. Nothing under ``data/`` is written or modified.

The 2014->2018 and 2018->2022 chains are re-run identically as a control, so the
new numbers can be read against the pairs the repository already evaluates.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from scripts.geography.config import OFFICIAL_CONSTITUENCY_CODES
from scripts.geography.projection import project_constituency_votes
from scripts.mandates.allocator import allocate_riksdag_seats
from scripts.mandates.law import mandate_law_for_election_year
from scripts.simulator.config import MODEL_PARTIES_9

HERE = Path(__file__).resolve().parent
PROCESSED = HERE / "processed"
RESEARCH_GEO = PROCESSED / "research_geography"
PROD_GEO = REPO_ROOT / "data" / "processed" / "geography"
SEAT_PARTIES = ("M", "L", "C", "KD", "S", "V", "MP", "SD")

CHAINS = [(2006, 2010), (2010, 2014), (2014, 2018), (2018, 2022)]


def build_research_processed_dir() -> None:
    """Production CSVs plus the new 2006/2010 rows, in a research-only location."""
    RESEARCH_GEO.mkdir(parents=True, exist_ok=True)

    prod_votes = pd.read_csv(PROD_GEO / "constituency_party_votes_2014_2022.csv")
    new_votes = pd.read_csv(PROCESSED / "constituency_party_votes_2006_2010.csv")
    new_votes["constituency_code"] = new_votes["constituency_code"].astype(int)
    combined = pd.concat([new_votes, prod_votes], ignore_index=True)
    combined = combined.sort_values(["election_year", "constituency_code", "party"]).reset_index(drop=True)
    # The production rows must survive unchanged.
    check = combined[combined["election_year"].isin([2014, 2018, 2022])].reset_index(drop=True)
    ref = prod_votes.sort_values(["election_year", "constituency_code", "party"]).reset_index(drop=True)
    assert len(check) == len(ref), (len(check), len(ref))
    assert np.array_equal(check["votes"].to_numpy(), ref["votes"].to_numpy())
    combined.to_csv(RESEARCH_GEO / "constituency_party_votes_2014_2022.csv", index=False)

    # Electorates: copy production verbatim, then append valid-vote rows for the two
    # new baselines so 'oracle' mode is also runnable. Chronological mode with a
    # target year <= 2022 never reads this file's contents.
    el = pd.read_csv(PROD_GEO / "constituency_electorates_2014_2026.csv")
    extra = []
    for year in (2006, 2010):
        sub = new_votes[new_votes["election_year"] == year]
        for cc in OFFICIAL_CONSTITUENCY_CODES:
            rows = sub[sub["constituency_code"] == int(cc)]
            extra.append(
                {
                    "election_year": year,
                    "constituency_code": int(cc),
                    "constituency_name": rows["constituency_name"].iloc[0],
                    "eligible_voters": np.nan,
                    "valid_votes": int(rows["constituency_valid_votes"].iloc[0]),
                }
            )
    el_ext = pd.concat([pd.DataFrame(extra), el], ignore_index=True)
    el_ext.to_csv(RESEARCH_GEO / "constituency_electorates_2014_2026.csv", index=False)


def actual_national(target_year: int) -> tuple[dict[str, float], int]:
    df = pd.read_csv(RESEARCH_GEO / "constituency_party_votes_2014_2022.csv")
    sub = df[df["election_year"] == target_year]
    votes = {p: int(sub[sub["party"] == p]["votes"].sum()) for p in MODEL_PARTIES_9}
    total = sum(votes.values())
    return {p: votes[p] / total for p in MODEL_PARTIES_9}, total


def certified(target_year: int) -> dict[str, int]:
    if target_year in (2010, 2014):
        df = pd.read_csv(PROCESSED / "certified_mandates_2010_2014.csv")
    else:
        df = pd.read_csv(REPO_ROOT / "data" / "processed" / "mandates" / "historical_certified_mandates.csv")
    sub = df[df["election_year"] == target_year]
    return {p: int(sub[sub["party"] == p]["total_seats"].sum()) for p in SEAT_PARTIES}


def fixed_seats(year: int) -> dict[str, int]:
    payload = json.loads((PROCESSED / "fixed_seats_by_year.json").read_text())
    return {k: int(v) for k, v in payload["fixed_seats_by_year"][str(year)].items()}


def evaluate(baseline_year: int, target_year: int, mode: str) -> dict:
    nat_shares, total_valid = actual_national(target_year)
    proj = project_constituency_votes(
        national_vote_shares=nat_shares,
        baseline_year=baseline_year,
        target_year=target_year,
        mode=mode,
        total_national_votes=total_valid if mode == "oracle" else None,
        processed_dir=RESEARCH_GEO,
    )

    # Constituency party-share MAE, exactly as scripts/geography/evaluate.py defines it.
    df = pd.read_csv(RESEARCH_GEO / "constituency_party_votes_2014_2022.csv")
    sub = df[df["election_year"] == target_year]
    c_idx = {c: i for i, c in enumerate(OFFICIAL_CONSTITUENCY_CODES)}
    p_idx = {p: i for i, p in enumerate(MODEL_PARTIES_9)}
    actual = np.zeros((29, 9))
    for _, r in sub.iterrows():
        actual[c_idx[f"{int(r['constituency_code']):02d}"], p_idx[str(r["party"])]] = float(r["party_share"])
    projected = np.zeros((29, 9))
    for cc, pmap in proj.constituency_votes.items():
        cv = proj.constituency_valid_votes[cc]
        for p, v in pmap.items():
            projected[c_idx[cc], p_idx[p]] = v / cv if cv > 0 else 0.0
    diffs = np.abs(projected - actual)

    cfg = mandate_law_for_election_year(target_year)
    alloc = allocate_riksdag_seats(
        constituency_votes=proj.to_allocator_input(),
        fixed_seats_by_constituency=fixed_seats(target_year),
        first_divisor=cfg.first_divisor,
        law=cfg.law,
        scenario_id=f"geo_{baseline_year}_{target_year}_{mode}",
    )
    cert = certified(target_year)
    produced = {p: alloc.final_seats_by_party.get(p, 0) for p in SEAT_PARTIES}
    seat_diff = {p: produced[p] - cert[p] for p in SEAT_PARTIES}

    return {
        "baseline_year": baseline_year,
        "target_year": target_year,
        "mode": mode,
        "law": cfg.law.value,
        "first_divisor": str(cfg.first_divisor),
        "constituency_share_mae": float(np.mean(diffs)),
        "constituency_share_max_abs_error": float(np.max(diffs)),
        "party_share_mae": {p: float(np.mean(diffs[:, p_idx[p]])) for p in MODEL_PARTIES_9},
        "national_share_max_error": float(
            max(abs(proj.national_vote_shares[p] - nat_shares[p]) for p in MODEL_PARTIES_9)
        ),
        "national_votes_reconciliation_diff": int(
            sum(proj.constituency_valid_votes.values()) - (total_valid if mode == "oracle" else sum(proj.constituency_valid_votes.values()))
        ),
        "ipf_iterations": proj.ipf_result.iterations,
        "projected_seats": produced,
        "certified_seats": cert,
        "seat_differences": seat_diff,
        "total_absolute_seat_error": int(sum(abs(v) for v in seat_diff.values())),
        "exact_seat_match": all(v == 0 for v in seat_diff.values()),
        "set_aside_parties": list(alloc.set_aside_parties),
    }


def main() -> int:
    build_research_processed_dir()
    report = {"status": "RESEARCH ONLY", "chains": []}
    for base, target in CHAINS:
        for mode in ("chronological", "oracle"):
            report["chains"].append(evaluate(base, target, mode))
    (PROCESSED / "geography_validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )

    print(f"{'chain':>14} {'mode':>14} {'law':>10} {'share MAE':>10} {'seat err':>9} {'exact':>6}  seats")
    for r in report["chains"]:
        chain = f"{r['baseline_year']}->{r['target_year']}"
        print(
            f"{chain:>14} {r['mode']:>14} {r['law']:>10} {r['constituency_share_mae']:>10.5f} "
            f"{r['total_absolute_seat_error']:>9d} {str(r['exact_seat_match']):>6}  {r['projected_seats']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
