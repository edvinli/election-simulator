"""Parse and reconcile the fetched Valmyndigheten historical pages (research only).

Produces, under ``processed/``:

* ``constituency_party_votes_2006_2010.csv`` — same schema as the repository's
  ``data/processed/geography/constituency_party_votes_2014_2022.csv``.
* ``certified_mandates_2010_2014.csv`` — same schema as the repository's
  ``data/processed/mandates/historical_certified_mandates.csv``.
* ``reconciliation.json`` — every check and its exact residual.

Reconciliation is exact-or-fail: any nonzero unexplained difference against the
official national totals in ``data/processed/elections/riksdag_election_results.csv``
raises, and the affected year cannot be ACCEPTed.

Nothing under ``data/`` is written or modified.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from scripts.geography.config import OFFICIAL_CONSTITUENCY_CODES
from scripts.mandates.config import OFFICIAL_CONSTITUENCIES

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
PROCESSED = HERE / "processed"

MODEL_PARTIES = ("M", "L", "C", "KD", "S", "V", "MP", "SD")
CATEGORIES = MODEL_PARTIES + ("REST",)

#: Official abbreviation on historik.val.se -> canonical model category.
#: 'FP' (Folkpartiet liberalerna) is the pre-2015 name of 'L' (Liberalerna); the
#: rename carried no organisational change, so the mapping is deterministic.
ABBREV_TO_CATEGORY = {
    "M": "M",
    "C": "C",
    "FP": "L",
    "L": "L",
    "KD": "KD",
    "S": "S",
    "V": "V",
    "MP": "MP",
    "SD": "SD",
}
#: Rows that are not party results and must never enter a vote total.
NON_PARTY_ROWS = {"BLANK", "OG", "VDT", "", "&nbsp;", "\xa0"}


def read_html(path: Path) -> str:
    raw = path.read_bytes()
    m = re.search(rb"charset=([A-Za-z0-9_\-]+)", raw)
    enc = m.group(1).decode("ascii").lower() if m else "iso-8859-1"
    if enc in {"utf-8", "utf8"}:
        return raw.decode("utf-8", errors="strict")
    return raw.decode("iso-8859-1", errors="strict")


def table_rows(doc: str) -> list[list[list[str]]]:
    out = []
    for t in re.findall(r"<table[^>]*>(.*?)</table>", doc, re.DOTALL | re.I):
        rows = []
        for r in re.findall(r"<tr[^>]*>(.*?)</tr>", t, re.DOTALL | re.I):
            cells = [
                html.unescape(re.sub(r"<[^>]+>", "", c)).replace("\xa0", " ").strip()
                for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.DOTALL | re.I)
            ]
            rows.append(cells)
        out.append(rows)
    return out


def to_int(text: str) -> int:
    cleaned = text.replace(" ", "").replace(" ", "").replace("+", "").strip()
    if cleaned in {"", "-", "–"}:
        return 0
    return int(cleaned)


def parse_votes_2006(cc: str) -> tuple[dict[str, int], int]:
    """2006 constituency: party votes (SD folded out of ÖVR) and valid-vote total."""
    doc = read_html(RAW / "votes_mandates_2006" / f"{cc}.html")
    tabs = table_rows(doc)
    votes = {c: 0 for c in CATEGORIES}
    ovr_total = 0
    for row in tabs[1]:
        if len(row) < 3 or row[0] in NON_PARTY_ROWS or row[0] in ("Röstfördelning", "Antal"):
            continue
        abbrev = row[0]
        n = to_int(row[2])
        if abbrev in ABBREV_TO_CATEGORY:
            votes[ABBREV_TO_CATEGORY[abbrev]] += n
        elif abbrev.upper().startswith("ÖVR"):
            ovr_total += n
        else:
            raise ValueError(f"2006 cc={cc}: unmapped abbreviation {abbrev!r}")

    # 2006 result pages do not break SD out of ÖVR; the per-constituency
    # 'ovriga' page does. Everything else in ÖVR stays inside REST.
    sd = 0
    odoc = read_html(RAW / "ovriga_2006" / f"{cc}.html")
    otabs = table_rows(odoc)
    ovr_listed = 0
    for row in otabs[0]:
        if len(row) < 2 or row[0] in {"Partibeteckning", ""}:
            continue
        n = to_int(row[1])
        ovr_listed += n
        if row[0].strip().lower() == "sverigedemokraterna":
            sd += n
    if ovr_listed != ovr_total:
        raise ValueError(
            f"2006 cc={cc}: 'ovriga' page lists {ovr_listed} votes but the result page "
            f"reports ÖVR = {ovr_total}"
        )
    votes["SD"] = sd
    votes["REST"] = ovr_total - sd

    valid = None
    for row in tabs[2]:
        if row and row[0].lower().startswith("summa giltiga"):
            valid = to_int(row[1])
    if valid is None:
        raise ValueError(f"2006 cc={cc}: 'Summa giltiga röster' not found")
    if sum(votes.values()) != valid:
        raise ValueError(
            f"2006 cc={cc}: party votes sum to {sum(votes.values())} but valid total is {valid}"
        )
    return votes, valid


def parse_votes_2010(cc: str) -> tuple[dict[str, int], int]:
    """2010 constituency party votes and valid-vote total."""
    doc = read_html(RAW / "votes_2010" / f"{cc}.html")
    tabs = table_rows(doc)
    votes = {c: 0 for c in CATEGORIES}
    for row in tabs[3]:
        if len(row) < 3 or row[0] in NON_PARTY_ROWS or row[0] == "Förk.":
            continue
        abbrev = row[0]
        n = to_int(row[2])
        if abbrev in ABBREV_TO_CATEGORY:
            votes[ABBREV_TO_CATEGORY[abbrev]] += n
        elif abbrev.upper().startswith("ÖVR"):
            votes["REST"] += n
        else:
            raise ValueError(f"2010 cc={cc}: unmapped abbreviation {abbrev!r}")
    return votes, sum(votes.values())


def parse_mandates(key: str, cc: str) -> dict[str, int]:
    """Certified per-constituency mandates from a 'valda' page, with its own control total."""
    doc = read_html(RAW / key / f"{cc}.html")
    tabs = table_rows(doc)
    seats = {p: 0 for p in MODEL_PARTIES}
    stated_total = None
    for row in tabs[1]:
        if len(row) < 3 or row[1] in {"Parti", ""}:
            continue
        if row[1].strip().lower() == "totalt":
            stated_total = to_int(row[2])
            continue
        abbrev = row[0]
        if abbrev not in ABBREV_TO_CATEGORY:
            raise ValueError(f"{key} cc={cc}: unmapped abbreviation {abbrev!r} in mandate table")
        seats[ABBREV_TO_CATEGORY[abbrev]] += to_int(row[2])
    if stated_total is None:
        raise ValueError(f"{key} cc={cc}: no 'Totalt' control row")
    if sum(seats.values()) != stated_total:
        raise ValueError(
            f"{key} cc={cc}: parsed seats {sum(seats.values())} != stated total {stated_total}"
        )
    return seats


def official_national(year: int) -> dict[str, int]:
    df = pd.read_csv(REPO_ROOT / "data" / "processed" / "elections" / "riksdag_election_results.csv")
    sub = df[df["election_year"] == year]
    if sub.empty:
        raise ValueError(f"No official national results for {year}")
    out = {c: 0 for c in CATEGORIES}
    for _, r in sub.iterrows():
        p = str(r["party"]).strip().upper()
        cat = p if p in MODEL_PARTIES else "REST"
        out[cat] += int(r["votes"])
    out["_valid_total"] = int(sub["valid_votes_total"].iloc[0])
    return out


def main() -> int:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    report: dict = {"status": "RESEARCH ONLY", "checks": {}}

    vote_rows: list[dict] = []
    for year, parser in ((2006, parse_votes_2006), (2010, parse_votes_2010)):
        per_cc: dict[str, tuple[dict[str, int], int]] = {}
        for cc in OFFICIAL_CONSTITUENCY_CODES:
            per_cc[cc] = parser(cc)
        assert len(per_cc) == 29

        totals = {c: sum(v[0][c] for v in per_cc.values()) for c in CATEGORIES}
        valid_sum = sum(v[1] for v in per_cc.values())
        off = official_national(year)
        diffs = {c: totals[c] - off[c] for c in CATEGORIES}
        valid_diff = valid_sum - off["_valid_total"]

        report["checks"][f"votes_{year}"] = {
            "constituencies": len(per_cc),
            "constituency_valid_sum": valid_sum,
            "official_valid_total": off["_valid_total"],
            "valid_total_difference": valid_diff,
            "per_party_difference": diffs,
            "max_abs_party_difference": max(abs(d) for d in diffs.values()),
            "exact": valid_diff == 0 and all(d == 0 for d in diffs.values()),
        }

        for cc in OFFICIAL_CONSTITUENCY_CODES:
            v, valid = per_cc[cc]
            for cat in CATEGORIES:
                vote_rows.append(
                    {
                        "election_year": year,
                        "constituency_code": cc,
                        "constituency_name": OFFICIAL_CONSTITUENCIES[cc],
                        "party": cat,
                        "votes": v[cat],
                        "constituency_valid_votes": valid,
                        "party_share": v[cat] / valid,
                    }
                )

    seat_rows: list[dict] = []
    for year, key in ((2010, "mandates_2010"), (2014, "mandates_2014")):
        per_cc = {cc: parse_mandates(key, cc) for cc in OFFICIAL_CONSTITUENCY_CODES}
        nat = {p: sum(per_cc[cc][p] for cc in per_cc) for p in MODEL_PARTIES}
        report["checks"][f"mandates_{year}"] = {
            "constituencies": len(per_cc),
            "national_seat_vector": nat,
            "total_seats": sum(nat.values()),
            "sums_to_349": sum(nat.values()) == 349,
        }
        for cc in OFFICIAL_CONSTITUENCY_CODES:
            for p in MODEL_PARTIES:
                seat_rows.append(
                    {
                        "election_year": year,
                        "constituency_code": cc,
                        "constituency_name": OFFICIAL_CONSTITUENCIES[cc],
                        "party": p,
                        "total_seats": per_cc[cc][p],
                    }
                )

    with open(PROCESSED / "constituency_party_votes_2006_2010.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(vote_rows[0].keys()))
        w.writeheader()
        w.writerows(vote_rows)
    with open(PROCESSED / "certified_mandates_2010_2014.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(seat_rows[0].keys()))
        w.writeheader()
        w.writerows(seat_rows)

    (PROCESSED / "reconciliation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    failures = [k for k, v in report["checks"].items() if "exact" in v and not v["exact"]]
    failures += [k for k, v in report["checks"].items() if "sums_to_349" in v and not v["sums_to_349"]]
    if failures:
        raise SystemExit(f"RECONCILIATION FAILED for: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
