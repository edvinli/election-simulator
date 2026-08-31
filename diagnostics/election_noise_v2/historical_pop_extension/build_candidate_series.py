"""Build a research-only historical Poll-of-Polls state series from archived first-party charts.

Source: the already-archived first-party PoP party-chart CSVs
``data/raw/pollofpolls/party_<PARTY>.csv`` (``pollofpolls.se/poll_img/data_big_N.csv``),
specifically their ``pofp`` column.

Construction is purely mechanical:

* the eight parliamentary parties are taken on the dates where **all eight** have a
  ``pofp`` value — no interpolation, no forward or backward filling, no imputation;
* ``REST`` is derived as ``100 - sum(eight)``, the same rule
  ``scripts/pollofpolls/state.py::load_timeseries_dataset`` applies to the canonical
  series;
* the output schema matches the canonical ``pollofpolls_timeseries.csv`` so the frozen
  loader can read it unchanged.

Nothing under ``data/`` is written or modified. The canonical production
``pollofpolls_timeseries.csv`` is untouched.

Outputs, under ``processed/``:
    candidate_pop_state_2009_2026.csv   the normalized series
    overlap_reconciliation.json         candidate vs canonical over the full overlap
    provenance.json                     source URLs, retrieval timestamps, SHA-256
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import statistics as st
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HERE = Path(__file__).resolve().parent
PROCESSED = HERE / "processed"
RAW = REPO_ROOT / "data" / "raw" / "pollofpolls"
CANONICAL_RAW = RAW / "pollofpolls_timeseries_source.dat"
CANONICAL_PROCESSED = REPO_ROOT / "data" / "processed" / "pollofpolls" / "pollofpolls_timeseries.csv"

#: Order of the canonical processed timeseries schema.
PARTIES = ("M", "L", "C", "KD", "S", "V", "MP", "SD")
#: FI is carried by the source but maps to REST in the model's 9 categories.
EXTRA = ("FI",)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_chart_pofp(party: str) -> dict[str, float]:
    """The ``pofp`` column of one archived party chart, verbatim."""
    text = (RAW / f"party_{party}.csv").read_text(encoding="utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or "pofp" not in reader.fieldnames:
        raise ValueError(f"party_{party}.csv has no pofp column")
    out: dict[str, float] = {}
    for row in reader:
        d = (row.get("date") or "").strip()
        v = (row.get("pofp") or "").strip()
        if len(d) != 10 or v in ("", "NaN"):
            continue
        if d in out:
            raise ValueError(f"duplicate date {d} in party_{party}.csv")
        out[d] = float(v)
    return out


def read_canonical() -> dict[str, dict[str, float]]:
    text = CANONICAL_RAW.read_text(encoding="utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    out: dict[str, dict[str, float]] = {}
    for row in reader:
        d = (row.get("Datum") or "").strip()
        if len(d) != 10:
            continue
        vals = {}
        for p in PARTIES + EXTRA:
            v = (row.get(p) or "").strip()
            if v not in ("", "NaN"):
                vals[p] = float(v)
        out[d] = vals
    return out


def main() -> int:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    charts = {p: read_chart_pofp(p) for p in PARTIES}
    fi = read_chart_pofp("FI")

    # Only dates where all eight parliamentary parties are present. No filling.
    common = sorted(set.intersection(*(set(v) for v in charts.values())))
    per_party_dates = {p: (min(charts[p]), max(charts[p]), len(charts[p])) for p in PARTIES}

    rows = []
    rest_negative: list[tuple[str, float]] = []
    for d in common:
        eight = [charts[p][d] for p in PARTIES]
        rest = round(100.0 - sum(eight), 10)
        if rest < 0.0:
            rest_negative.append((d, rest))
        rows.append(
            {
                "date": d,
                **{p: charts[p][d] for p in PARTIES},
                "FI": fi.get(d, ""),
                "REST_derived": max(rest, 0.0),
                "source_url": "http://pollofpolls.se/poll_img/data_big_N.csv (pofp column)",
                "retrieved_at": "",
            }
        )

    with open(PROCESSED / "candidate_pop_state_2009_2026.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- overlap reconciliation against the canonical series -----------------
    canon = read_canonical()
    ov = sorted(set(common) & set(canon))
    recon: dict = {
        "candidate": {
            "dates": len(common),
            "min_date": common[0],
            "max_date": common[-1],
            "per_party_coverage": {
                p: {"min": a, "max": b, "n": n} for p, (a, b, n) in per_party_dates.items()
            },
        },
        "canonical": {"dates": len(canon), "min_date": min(canon), "max_date": max(canon)},
        "overlap": {"dates": len(ov), "min_date": ov[0] if ov else None, "max_date": ov[-1] if ov else None},
        "by_category": {},
        "rest_derived_negative_dates": rest_negative[:20],
        "rest_derived_negative_count": len(rest_negative),
    }
    for p in PARTIES + EXTRA:
        diffs = []
        exact = 0
        n = 0
        src = charts[p] if p in charts else fi
        for d in ov:
            a = src.get(d)
            b = canon[d].get(p)
            if a is None or b is None:
                continue
            n += 1
            dd = abs(a - b)
            diffs.append(dd)
            if dd < 1e-9:
                exact += 1
        recon["by_category"][p] = {
            "matched_dates": n,
            "exact_matches": exact,
            "exact_match_fraction": (exact / n) if n else None,
            "mean_absolute_difference": st.mean(diffs) if diffs else None,
            "max_absolute_difference": max(diffs) if diffs else None,
        }
    # REST comparison uses the canonical loader's own rule on both sides.
    r_diffs = []
    for d in ov:
        a = 100.0 - sum(charts[p][d] for p in PARTIES)
        b = 100.0 - sum(canon[d][p] for p in PARTIES if p in canon[d])
        r_diffs.append(abs(a - b))
    recon["by_category"]["REST"] = {
        "matched_dates": len(r_diffs),
        "exact_matches": sum(1 for x in r_diffs if x < 1e-9),
        "exact_match_fraction": sum(1 for x in r_diffs if x < 1e-9) / len(r_diffs),
        "mean_absolute_difference": st.mean(r_diffs),
        "max_absolute_difference": max(r_diffs),
        "note": "derived as 100 - sum(eight) on both sides, the canonical loader's rule",
    }
    recon["all_categories_exact_over_full_overlap"] = all(
        v["exact_match_fraction"] == 1.0 for v in recon["by_category"].values()
    )
    (PROCESSED / "overlap_reconciliation.json").write_text(
        json.dumps(recon, indent=2) + "\n"
    )

    # ---- provenance ---------------------------------------------------------
    manifest = json.loads((RAW / "retrieval_manifest.json").read_text())
    prov = {
        "status": "RESEARCH ONLY - derived from already-archived first-party sources; nothing under data/ modified",
        "canonical_production_series": {
            "path": str(CANONICAL_PROCESSED.relative_to(REPO_ROOT)),
            "sha256": sha256_file(CANONICAL_PROCESSED),
            "modified_by_this_task": False,
        },
        "sources": [
            {
                "party": p,
                "file": f"data/raw/pollofpolls/party_{p}.csv",
                "source_url": manifest["sources"][f"party_{p}"]["final_url"],
                "retrieved_at": manifest["sources"][f"party_{p}"]["retrieved_at"],
                "sha256": manifest["sources"][f"party_{p}"]["sha256"],
                "column_used": "pofp",
            }
            for p in PARTIES + EXTRA
        ],
        "canonical_source_for_reconciliation": {
            "file": "data/raw/pollofpolls/pollofpolls_timeseries_source.dat",
            "source_url": manifest["sources"]["timeseries"]["final_url"],
            "retrieved_at": manifest["sources"]["timeseries"]["retrieved_at"],
            "sha256": manifest["sources"]["timeseries"]["sha256"],
        },
        "construction": [
            "take the pofp column verbatim from each archived party chart",
            "keep only dates where all eight parliamentary parties have a value",
            "derive REST as 100 - sum(eight), the canonical loader's rule",
            "no interpolation, no forward or backward filling, no manual values",
        ],
    }
    (PROCESSED / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")

    print(f"candidate series: {len(common)} dates {common[0]}..{common[-1]}")
    print(f"overlap with canonical: {len(ov)} dates")
    for p, v in recon["by_category"].items():
        print(
            f"  {p:5s} matched={v['matched_dates']:5d} exact={v['exact_matches']:5d} "
            f"({(v['exact_match_fraction'] or 0)*100:6.3f}%) MAD={v['mean_absolute_difference']:.2e} "
            f"maxAD={v['max_absolute_difference']:.2e}"
        )
    print(f"all categories exact over full overlap: {recon['all_categories_exact_over_full_overlap']}")
    print(f"derived REST negative on {len(rest_negative)} dates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
