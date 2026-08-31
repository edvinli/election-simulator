"""Leakage audit of the Poll-of-Polls state series. Reproduces every §4 number.

Three independent probes:

1. **Retrospective revision.** Compare two archived snapshots of the same series
   from git history and report every revised cell and its date range.
2. **Mechanism attribution.** For the largest revised cell, list every poll whose
   fieldwork covers that date together with its publication date, showing that the
   revision can only be explained by a poll published after the row date.
3. **Magnitude per evaluation `as_of`.** For 2014, and for the two elections already
   in the certified Tier-2/Tier-3 set, count polls whose fieldwork covers `as_of`
   but which were published after it.

Read-only with respect to the repository. Writes ``processed/leakage_audit.json``.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
import io
import json
from pathlib import Path
import statistics as st
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

HERE = Path(__file__).resolve().parent
PROCESSED = HERE / "processed"
PARTIES = ("M", "L", "C", "KD", "S", "V", "MP", "SD")

SNAPSHOT_PAIRS = [
    ("f55bf36", "f6ae4d1", "data/raw/pollofpolls/pollofpolls_timeseries_source.dat"),
    ("f55bf36", "34c52d6", "data/processed/pollofpolls/pollofpolls_timeseries.csv"),
]
HORIZONS = (112, 84, 56, 28, 14, 7)
ELECTIONS = {2014: date(2014, 9, 14), 2018: date(2018, 9, 9), 2022: date(2022, 9, 11)}


def _snapshot(commit: str, path: str) -> dict[str, dict[str, float]]:
    txt = subprocess.check_output(["git", "show", f"{commit}:{path}"]).decode("utf-8", "replace")
    reader = csv.DictReader(io.StringIO(txt))
    key = "date" if "date" in (reader.fieldnames or []) else "Datum"
    out: dict[str, dict[str, float]] = {}
    for row in reader:
        d = (row.get(key) or "").strip()
        if len(d) != 10:
            continue
        vals = {}
        for p in PARTIES:
            v = (row.get(p) or "").strip()
            if v not in ("", "NaN"):
                try:
                    vals[p] = float(v)
                except ValueError:
                    pass
        out[d] = vals
    return out


def revision_probe() -> list[dict]:
    results = []
    for a, b, path in SNAPSHOT_PAIRS:
        A, B = _snapshot(a, path), _snapshot(b, path)
        ov = sorted(set(A) & set(B))
        diffs = []
        cells = 0
        for d in ov:
            for p in PARTIES:
                if p in A[d] and p in B[d]:
                    cells += 1
                    delta = B[d][p] - A[d][p]
                    if abs(delta) > 1e-9:
                        diffs.append({"date": d, "party": p, "from": A[d][p], "to": B[d][p], "delta": round(delta, 6)})
        diffs.sort(key=lambda r: -abs(r["delta"]))
        rev_dates = sorted({r["date"] for r in diffs})
        results.append(
            {
                "file": path,
                "snapshot_a": a,
                "snapshot_b": b,
                "a_max_date": max(A),
                "b_max_date": max(B),
                "overlapping_dates": len(ov),
                "cells_compared": cells,
                "revised_cells": len(diffs),
                "revised_fraction": len(diffs) / cells if cells else None,
                "max_abs_revision": abs(diffs[0]["delta"]) if diffs else 0.0,
                "mean_abs_revision_over_revised": st.mean(abs(r["delta"]) for r in diffs) if diffs else 0.0,
                "revised_distinct_dates": len(rev_dates),
                "revised_date_range": [rev_dates[0], rev_dates[-1]] if rev_dates else None,
                "oldest_unrevised_boundary": rev_dates[0] if rev_dates else None,
                "largest_revisions": diffs[:8],
            }
        )
    return results


def _polls() -> pd.DataFrame:
    ip = pd.read_csv(REPO_ROOT / "data/processed/pollofpolls/individual_polls.csv")
    p = ip.drop_duplicates("poll_id")[
        ["pollster", "interview_start", "interview_end", "publication_date"]
    ].dropna()
    # Drop the one upstream record whose fieldwork is dated after publication
    # (the corrupt 1998 TEMO row documented in Part 2).
    return p[p.publication_date >= p.interview_end]


def covering_polls(polls: pd.DataFrame, t: str) -> pd.DataFrame:
    return polls[(polls.interview_start <= t) & (polls.interview_end >= t)]


def mechanism_probe(revisions: list[dict]) -> dict:
    polls = _polls()
    top = revisions[0]["largest_revisions"][0]
    t = top["date"]
    rows = []
    for _, r in covering_polls(polls, t).sort_values("publication_date").iterrows():
        rows.append(
            {
                "pollster": r.pollster,
                "interview_start": str(r.interview_start),
                "interview_end": str(r.interview_end),
                "publication_date": str(r.publication_date),
                "published_after_row_date": str(r.publication_date) > t,
            }
        )
    return {
        "revised_cell": top,
        "polls_whose_fieldwork_covers_that_date": rows,
        "all_such_polls_published_after_the_row_date": all(x["published_after_row_date"] for x in rows),
        "conclusion": (
            "The historical value for this date changed between two retrievals one day apart, "
            "and every poll whose fieldwork covers it was published after it. The series is a "
            "fieldwork-dated rolling aggregate retrospectively completed as later-published polls "
            "arrive, so pofp(t) incorporates information published after t."
        ),
    }


def magnitude_probe() -> dict:
    polls = _polls()
    lag = (pd.to_datetime(polls.publication_date) - pd.to_datetime(polls.interview_end)).dt.days
    out: dict = {
        "publication_lag_after_fieldwork_end_days": {
            "n": int(len(lag)),
            "median": float(lag.median()),
            "mean": round(float(lag.mean()), 2),
            "p95": float(lag.quantile(0.95)),
            "max": int(lag.max()),
        },
        "by_election": {},
    }
    for year, ed in ELECTIONS.items():
        per_h = []
        total = 0
        max_lead = 0
        for h in HORIZONS:
            t = (ed - timedelta(days=h)).isoformat()
            m = covering_polls(polls, t)
            m = m[m.publication_date > t]
            leads = [
                (date.fromisoformat(str(r.publication_date)) - date.fromisoformat(t)).days
                for _, r in m.iterrows()
            ]
            total += len(m)
            max_lead = max(max_lead, max(leads) if leads else 0)
            per_h.append(
                {
                    "horizon_days": h,
                    "as_of": t,
                    "leaking_polls": len(m),
                    "max_publication_lead_days": max(leads) if leads else 0,
                    "detail": [
                        {
                            "pollster": r.pollster,
                            "interview_start": str(r.interview_start),
                            "interview_end": str(r.interview_end),
                            "publication_date": str(r.publication_date),
                            "lead_days": l,
                        }
                        for (_, r), l in zip(m.iterrows(), leads)
                    ],
                }
            )
        out["by_election"][str(year)] = {
            "in_certified_tier23_set": year in (2018, 2022),
            "total_leaking_poll_instances": total,
            "max_publication_lead_days": max_lead,
            "by_horizon": per_h,
        }
    return out


def main() -> int:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    revisions = revision_probe()
    report = {
        "status": "RESEARCH ONLY - read-only audit; nothing under data/ modified",
        "question": "Can a PoP state value at date t incorporate polling information published after t?",
        "probe_1_retrospective_revision": revisions,
        "probe_2_mechanism_attribution": mechanism_probe(revisions),
        "probe_3_magnitude_per_as_of": magnitude_probe(),
    }
    report["leakage_detected"] = report["probe_2_mechanism_attribution"][
        "all_such_polls_published_after_the_row_date"
    ]
    (PROCESSED / "leakage_audit.json").write_text(json.dumps(report, indent=2) + "\n")

    r0 = revisions[0]
    print(f"probe 1: {r0['revised_cells']} of {r0['cells_compared']} cells revised "
          f"({100*r0['revised_fraction']:.2f}%), dates {r0['revised_date_range'][0]}..{r0['revised_date_range'][1]}, "
          f"max |revision| {r0['max_abs_revision']}")
    m = report["probe_2_mechanism_attribution"]
    print(f"probe 2: cell {m['revised_cell']['date']} {m['revised_cell']['party']} "
          f"{m['revised_cell']['from']} -> {m['revised_cell']['to']}; "
          f"all covering polls published later: {m['all_such_polls_published_after_the_row_date']}")
    for y, v in report["probe_3_magnitude_per_as_of"]["by_election"].items():
        tag = " (in certified set)" if v["in_certified_tier23_set"] else ""
        print(f"probe 3: {y}{tag}: {v['total_leaking_poll_instances']} leaking poll-instances, "
              f"max lead {v['max_publication_lead_days']}d")
    print(f"\nLEAKAGE DETECTED: {report['leakage_detected']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
