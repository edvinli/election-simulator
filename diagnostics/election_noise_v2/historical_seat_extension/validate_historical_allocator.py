"""Validate the PRE_2018 mandate law against the certified 2010 and 2014 results.

Golden target: the certified final per-party seat vector, nationally and per
constituency, as published by Valmyndigheten. Historical `valda` pages do not
publish a fixed-vs-adjustment phase split, so no phase-level expectation is
asserted — only the final allocation, which is what the official source certifies.

Also runs each year under the wrong (POST_2018) law as a contrast, to show that
the law version is load-bearing rather than incidental.

Nothing under ``data/`` is written or modified; the production allocator's
default path is not exercised differently by anything here.
"""

from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from scripts.mandates.allocator import allocate_riksdag_seats
from scripts.mandates.config import FIXED_SEATS_2018, FIXED_SEATS_2022
from scripts.mandates.law import MandateLaw, mandate_law_for_election_year

HERE = Path(__file__).resolve().parent
PROCESSED = HERE / "processed"
MODEL_PARTIES = ("M", "L", "C", "KD", "S", "V", "MP", "SD")


def load_fixed_seats() -> dict[int, dict[str, int]]:
    payload = json.loads((PROCESSED / "fixed_seats_by_year.json").read_text())
    return {int(y): {k: int(v) for k, v in d.items()} for y, d in payload["fixed_seats_by_year"].items()}


def load_constituency_votes(year: int) -> dict[str, dict[str, int]]:
    if year in (2006, 2010):
        df = pd.read_csv(PROCESSED / "constituency_party_votes_2006_2010.csv")
    else:
        df = pd.read_csv(REPO_ROOT / "data" / "processed" / "geography" / "constituency_party_votes_2014_2022.csv")
    sub = df[df["election_year"] == year]
    if sub.empty:
        raise ValueError(f"no constituency votes for {year}")
    out: dict[str, dict[str, int]] = {}
    for _, r in sub.iterrows():
        cc = f"{int(r['constituency_code']):02d}"
        out.setdefault(cc, {})[str(r["party"])] = int(r["votes"])
    return out


def load_certified(year: int) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    if year in (2010, 2014):
        df = pd.read_csv(PROCESSED / "certified_mandates_2010_2014.csv")
    else:
        df = pd.read_csv(REPO_ROOT / "data" / "processed" / "mandates" / "historical_certified_mandates.csv")
    sub = df[df["election_year"] == year]
    per_cc: dict[str, dict[str, int]] = {}
    for _, r in sub.iterrows():
        cc = f"{int(r['constituency_code']):02d}"
        per_cc.setdefault(cc, {})[str(r["party"])] = int(r["total_seats"])
    national = {p: sum(per_cc[c].get(p, 0) for c in per_cc) for p in MODEL_PARTIES}
    return national, per_cc


def run(year: int, law: MandateLaw, first_divisor: Fraction) -> dict:
    votes = load_constituency_votes(year)
    fixed = load_fixed_seats()[year]
    res = allocate_riksdag_seats(
        votes,
        fixed,
        first_divisor=first_divisor,
        law=law,
        scenario_id=f"historical_{year}_{law.value}",
    )
    return {
        "seats_by_party": {p: res.final_seats_by_party.get(p, 0) for p in MODEL_PARTIES},
        "per_constituency": {
            c: {p: res.final_seats_by_party_constituency[c].get(p, 0) for p in MODEL_PARTIES}
            for c in res.final_seats_by_party_constituency
        },
        "national_entitlement": dict(res.national_entitlement),
        "initial_national_fixed": {
            p: res.initial_national_fixed_seats.get(p, 0) for p in MODEL_PARTIES
        },
        "national_adjustment_seats": {
            p: res.national_adjustment_seats.get(p, 0) for p in MODEL_PARTIES
        },
        "set_aside_parties": list(res.set_aside_parties),
        "law": res.law,
        "total_seats": res.total_seats,
    }


def compare(year: int, produced: dict) -> dict:
    certified_nat, certified_cc = load_certified(year)
    nat_diff = {p: produced["seats_by_party"][p] - certified_nat[p] for p in MODEL_PARTIES}
    cc_mismatches = []
    for cc, exp in certified_cc.items():
        got = produced["per_constituency"][cc]
        for p in MODEL_PARTIES:
            if got.get(p, 0) != exp.get(p, 0):
                cc_mismatches.append(
                    {"constituency": cc, "party": p, "produced": got.get(p, 0), "certified": exp.get(p, 0)}
                )
    return {
        "certified_national": certified_nat,
        "produced_national": produced["seats_by_party"],
        "national_difference": nat_diff,
        "national_exact_match": all(v == 0 for v in nat_diff.values()),
        "total_seats": produced["total_seats"],
        "sums_to_349": produced["total_seats"] == 349,
        "constituency_mismatches": cc_mismatches,
        "constituency_exact_match": len(cc_mismatches) == 0,
    }


def main() -> int:
    report: dict = {"status": "RESEARCH ONLY", "results": {}}

    for year in (2010, 2014):
        cfg = mandate_law_for_election_year(year)
        assert cfg.law is MandateLaw.PRE_2018 and cfg.first_divisor == Fraction(7, 5), cfg
        correct = run(year, cfg.law, cfg.first_divisor)
        cmp_correct = compare(year, correct)

        wrong = run(year, MandateLaw.POST_2018, Fraction(6, 5))
        cmp_wrong = compare(year, wrong)

        report["results"][str(year)] = {
            "law_config": {
                "law": cfg.law.value,
                "first_divisor": str(cfg.first_divisor),
                "statute": cfg.statute,
            },
            "correct_law": {**cmp_correct, "detail": correct},
            "wrong_law_contrast_post_2018": {
                "produced_national": cmp_wrong["produced_national"],
                "national_difference": cmp_wrong["national_difference"],
                "national_exact_match": cmp_wrong["national_exact_match"],
                "total_absolute_seat_error": sum(abs(v) for v in cmp_wrong["national_difference"].values()),
            },
        }

    # Regression contrast: the two production years must still be exact under the
    # default law, and must be produced by the default code path.
    for year, fixed in ((2018, FIXED_SEATS_2018), (2022, FIXED_SEATS_2022)):
        cfg = mandate_law_for_election_year(year)
        assert cfg.law is MandateLaw.POST_2018 and cfg.first_divisor == Fraction(6, 5), cfg
        votes = load_constituency_votes(year)
        res_default = allocate_riksdag_seats(votes, fixed)  # production defaults, no law kwarg
        certified_nat, certified_cc = load_certified(year)
        nat = {p: res_default.final_seats_by_party.get(p, 0) for p in MODEL_PARTIES}
        mism = [
            (cc, p)
            for cc, exp in certified_cc.items()
            for p in MODEL_PARTIES
            if res_default.final_seats_by_party_constituency[cc].get(p, 0) != exp.get(p, 0)
        ]
        report["results"][str(year)] = {
            "law_config": {"law": cfg.law.value, "first_divisor": str(cfg.first_divisor)},
            "invoked": "allocate_riksdag_seats(votes, fixed)  # no law kwarg -> production default",
            "produced_national": nat,
            "certified_national": certified_nat,
            "national_exact_match": nat == certified_nat,
            "constituency_exact_match": len(mism) == 0,
            "set_aside_parties": list(res_default.set_aside_parties),
            "law_recorded": res_default.law,
        }

    (PROCESSED / "allocator_validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )

    for y in ("2010", "2014"):
        r = report["results"][y]["correct_law"]
        w = report["results"][y]["wrong_law_contrast_post_2018"]
        print(f"--- {y} under PRE_2018 (divisor 7/5)")
        print(f"    national exact: {r['national_exact_match']}   constituency exact: {r['constituency_exact_match']}   total={r['total_seats']}")
        print(f"    produced : {r['produced_national']}")
        print(f"    certified: {r['certified_national']}")
        print(f"    set aside as over-represented: {report['results'][y]['correct_law']['detail']['set_aside_parties']}")
        print(f"    contrast under POST_2018: exact={w['national_exact_match']} total_abs_seat_error={w['total_absolute_seat_error']}")
    for y in ("2018", "2022"):
        r = report["results"][y]
        print(f"--- {y} default path: national exact={r['national_exact_match']} constituency exact={r['constituency_exact_match']} law={r['law_recorded']}")

    ok = all(
        report["results"][y]["correct_law"]["national_exact_match"]
        and report["results"][y]["correct_law"]["constituency_exact_match"]
        for y in ("2010", "2014")
    ) and all(
        report["results"][y]["national_exact_match"] and report["results"][y]["constituency_exact_match"]
        for y in ("2018", "2022")
    )
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
