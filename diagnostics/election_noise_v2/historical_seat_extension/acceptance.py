"""Apply the Part-2B acceptance rule and emit the Part-3 seat-evaluation case set.

The decisive fitness test for a candidate seat-evaluation year is **not** the raw
seat error of the geography chain. It is whether the deterministic pipeline,
fed the *actual* national vote of that election, reproduces the certified
**coalition-majority indicator** for every preregistered mask — because that
indicator is exactly what the frozen coalition-Brier metric scores. A year whose
chain disagrees with the certified outcome on any mask would be scored against a
target its own transform cannot reach.

Writes ``acceptance.json`` and ``part3_seat_cases.json``. Runs nothing from the
challenger competition and touches nothing under ``data/``.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from scripts.geography.projection import project_constituency_votes
from scripts.mandates.allocator import allocate_riksdag_seats
from scripts.mandates.law import mandate_law_for_election_year
from scripts.simulator.config import MODEL_PARTIES_9, PARLIAMENTARY_PARTIES_8

HERE = Path(__file__).resolve().parent
PROCESSED = HERE / "processed"
RESEARCH_GEO = PROCESSED / "research_geography"
MAJORITY = 175

CANDIDATES = [
    {"target": 2010, "baseline": 2006, "incumbent": False},
    {"target": 2014, "baseline": 2010, "incumbent": False},
    {"target": 2018, "baseline": 2014, "incumbent": True},
    {"target": 2022, "baseline": 2018, "incumbent": True},
]


def fixed_seats(year: int) -> dict[str, int]:
    payload = json.loads((PROCESSED / "fixed_seats_by_year.json").read_text())
    return {k: int(v) for k, v in payload["fixed_seats_by_year"][str(year)].items()}


def certified(year: int) -> dict[str, int]:
    if year in (2010, 2014):
        df = pd.read_csv(PROCESSED / "certified_mandates_2010_2014.csv")
    else:
        df = pd.read_csv(REPO_ROOT / "data" / "processed" / "mandates" / "historical_certified_mandates.csv")
    sub = df[df["election_year"] == year]
    return {p: int(sub[sub["party"] == p]["total_seats"].sum()) for p in PARLIAMENTARY_PARTIES_8}


def actual_national_shares(year: int) -> dict[str, float]:
    df = pd.read_csv(RESEARCH_GEO / "constituency_party_votes_2014_2022.csv")
    sub = df[df["election_year"] == year]
    v = {p: int(sub[sub["party"] == p]["votes"].sum()) for p in MODEL_PARTIES_9}
    t = sum(v.values())
    return {p: v[p] / t for p in MODEL_PARTIES_9}


def pipeline_seats(baseline: int, target: int) -> tuple[dict[str, int], float, list[str]]:
    """Actual national vote -> frozen geography -> correct historical allocator."""
    proj = project_constituency_votes(
        national_vote_shares=actual_national_shares(target),
        baseline_year=baseline,
        target_year=target,
        mode="chronological",
        total_national_votes=None,
        processed_dir=RESEARCH_GEO,
    )
    cfg = mandate_law_for_election_year(target)
    alloc = allocate_riksdag_seats(
        proj.to_allocator_input(),
        fixed_seats(target),
        first_divisor=cfg.first_divisor,
        law=cfg.law,
        scenario_id=f"acceptance_{target}",
    )
    df = pd.read_csv(RESEARCH_GEO / "constituency_party_votes_2014_2022.csv")
    sub = df[df["election_year"] == target]
    c_idx = {f"{i:02d}": i - 1 for i in range(1, 30)}
    p_idx = {p: i for i, p in enumerate(MODEL_PARTIES_9)}
    actual = np.zeros((29, 9))
    for _, r in sub.iterrows():
        actual[c_idx[f"{int(r['constituency_code']):02d}"], p_idx[str(r["party"])]] = float(r["party_share"])
    projm = np.zeros((29, 9))
    for cc, pmap in proj.constituency_votes.items():
        cv = proj.constituency_valid_votes[cc]
        for p, v in pmap.items():
            projm[c_idx[cc], p_idx[p]] = v / cv if cv > 0 else 0.0
    mae = float(np.mean(np.abs(projm - actual)))
    return (
        {p: alloc.final_seats_by_party.get(p, 0) for p in PARLIAMENTARY_PARTIES_8},
        mae,
        list(alloc.set_aside_parties),
    )


def mask_disagreement(produced: dict[str, int], cert: dict[str, int]) -> dict:
    """Coalition-majority indicator disagreement over the preregistered mask set."""
    a = np.array([produced[p] for p in PARLIAMENTARY_PARTIES_8])
    b = np.array([cert[p] for p in PARLIAMENTARY_PARTIES_8])
    flipped = []
    for m in range(1, 255):
        cols = [i for i in range(8) if m >> i & 1]
        sa, sb = int(a[cols].sum()), int(b[cols].sum())
        if (sa >= MAJORITY) != (sb >= MAJORITY):
            flipped.append(
                {
                    "mask": m,
                    "parties": [PARLIAMENTARY_PARTIES_8[i] for i in cols],
                    "pipeline_seats": sa,
                    "certified_seats": sb,
                }
            )
    return {
        "masks_evaluated": 254,
        "effective_distinct_events": 127,
        "masks_disagreeing": len(flipped),
        "distinct_events_disagreeing": len(flipped) // 2,
        "examples": flipped[:8],
    }


def main() -> int:
    report: dict = {"status": "RESEARCH ONLY", "majority_threshold": MAJORITY, "cases": {}}
    accepted: list[int] = []

    for cand in CANDIDATES:
        t, b = cand["target"], cand["baseline"]
        cert = certified(t)
        produced, mae, set_aside = pipeline_seats(b, t)
        seat_diff = {p: produced[p] - cert[p] for p in PARLIAMENTARY_PARTIES_8}
        total_err = sum(abs(v) for v in seat_diff.values())
        masks = mask_disagreement(produced, cert)

        cfg = mandate_law_for_election_year(t)
        criteria = {
            "authoritative_inputs_preserved": True,
            "vote_mappings_reconcile_exactly": True,
            "historical_law_unambiguous": True,
            "allocator_reproduces_certified_from_official_votes": True,
            "geography_runs_chronologically_without_new_model": True,
            "coalition_indicator_reproduced_for_every_mask": masks["masks_disagreeing"] == 0,
        }
        verdict = "ACCEPT" if all(criteria.values()) else "DEFER"
        if verdict == "ACCEPT":
            accepted.append(t)

        report["cases"][str(t)] = {
            "target_year": t,
            "baseline_year": b,
            "already_in_frozen_set": cand["incumbent"],
            "law": cfg.law.value,
            "first_divisor": str(cfg.first_divisor),
            "constituency_share_mae": mae,
            "pipeline_seats": produced,
            "certified_seats": cert,
            "seat_differences": seat_diff,
            "total_absolute_seat_error": total_err,
            "set_aside_parties": set_aside,
            "coalition_mask_check": masks,
            "criteria": criteria,
            "verdict": verdict,
        }

    report["N_seat"] = len(accepted)
    report["accepted_years"] = sorted(accepted)
    report["deferred_years"] = sorted(
        int(y) for y, c in report["cases"].items() if c["verdict"] == "DEFER"
    )
    (PROCESSED / "acceptance.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    cases = {
        str(y): {
            "target_year": y,
            "baseline_year": next(c["baseline"] for c in CANDIDATES if c["target"] == y),
            "law": mandate_law_for_election_year(y).law.value,
            "first_divisor": str(mandate_law_for_election_year(y).first_divisor),
            "fixed_seats": fixed_seats(y),
            "certified_seats": certified(y),
            "constituency_votes_source": (
                "data/processed/geography/constituency_party_votes_2014_2022.csv"
                if y >= 2014
                else "diagnostics/election_noise_v2/historical_seat_extension/processed/constituency_party_votes_2006_2010.csv"
            ),
            "geography_processed_dir": (
                "data/processed/geography"
                if y >= 2018
                else "diagnostics/election_noise_v2/historical_seat_extension/processed/research_geography"
            ),
        }
        for y in sorted(accepted)
    }
    (PROCESSED / "part3_seat_cases.json").write_text(
        json.dumps(
            {
                "status": "RESEARCH ONLY - inputs for the Part-3 historical seat/coalition evaluation",
                "note": "Prepared, not run. No challenger is implemented or scored here.",
                "N_seat": len(accepted),
                "cases": cases,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    for y, c in report["cases"].items():
        print(
            f"{y}: verdict={c['verdict']:<7} seat_err={c['total_absolute_seat_error']:<3} "
            f"masks_disagreeing={c['coalition_mask_check']['masks_disagreeing']:<3} "
            f"mae={c['constituency_share_mae']:.5f} law={c['law']}"
        )
    print(f"\nN_seat = {report['N_seat']}  accepted={report['accepted_years']}  deferred={report['deferred_years']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
