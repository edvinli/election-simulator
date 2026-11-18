"""T4 of the calibration-2030 final round: end-to-end replay of C0, F1, CAND (and FB).

Implements ``docs/calibration_2030_final_round_preregistration.md`` §6 with
the gates G1–G7 and the 2026 flag of the parent
``docs/calibration_2030_preregistration.md`` §4.4, which are protocol v4 §6's
metrics. Every case runs ``scripts.simulator.engine.simulate_election`` end to
end through ``candidate_engine.simulate_candidate``.

    uv run python -m diagnostics.calibration_2030.t4_replay [--workers N] [--out PATH]
        [--smoke] [--cand-config PATH] [--geography {production,historical}]

``--smoke`` runs three cases at n = 1,000, writes under /tmp and may run on a
modified checkout. It never writes the results file.

Reused from ``scripts/poll_aggregate/replay.py``: ``load_truth``,
``case_dates_ok`` (G6's date condition, protocol v4 §7.1), ``_aggregate_row``,
the threshold and the frozen metric calls. ``selected_estimate`` is reproduced
for the 1.2.0 input layout (see ``selected_estimate_v12``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np

from scripts.poll_aggregate.config import HYPERPARAMETER_START, PARTIES
from scripts.poll_aggregate.replay import THRESHOLD_PCT, _aggregate_row, case_dates_ok, load_truth

from . import candidate_engine as ce
from .common import ROOT, sha256

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "t4_results.json"
SMOKE_OUTPUT = Path("/tmp/calibration_2030_t4_smoke.json")
T2_RESULTS = HERE / "t2_results.json"
T3REV_RESULTS = HERE / "t3rev_results.json"
PART2B = ROOT / "diagnostics" / "election_noise_v2" / "historical_seat_extension" / "processed"
PART2B_VOTES = PART2B / "constituency_party_votes_2006_2010.csv"
PART2B_MANDATES = PART2B / "certified_mandates_2010_2014.csv"
GEOGRAPHY_VOTES_NAME = "constituency_party_votes_2014_2022.csv"

DRAWS = 20_000
SEED = 20260929
SELF_CHECK_DRAWS = 2_000
ELECTION_DAYS = {2010: date(2010, 9, 19), 2014: date(2014, 9, 14), 2018: date(2018, 9, 9),
                 2022: date(2022, 9, 11), 2026: date(2026, 9, 13)}
DECISION_YEARS = (2014, 2018, 2022)
HORIZONS = {"D": (112, 84, 56, 28, 14, 7), "L": (365, 730, 1095)}
SUBSTITUTE_START = date(2007, 1, 1)
SUBSTITUTE_END = date(2010, 9, 25)
SELF_CHECK_CASES = ((2018, 28), (2026, 7))
SMOKE_CASES = ((2018, 28, "D"), (2022, 365, "L"), (2010, 112, "D"))
CATEGORIES = (*PARTIES, "REST")


def group_of(year: int) -> str:
    return "decision" if year in DECISION_YEARS else f"reported_{year}"


def all_cases() -> list[tuple[int, int, str]]:
    """(election year, horizon, set): set D and set L for 2014–2022, 2026 and 2010."""

    return [(y, h, s) for y in (2014, 2018, 2022, 2026, 2010) for s in ("D", "L") for h in HORIZONS[s]]


# --------------------------------------------------------------------------- #
# §6.3: 2010 substitute rows
# --------------------------------------------------------------------------- #


def substitute_rows(kind: str, ctx: dict, start: date = SUBSTITUTE_START, end: date = SUBSTITUTE_END) -> dict:
    """Causal daily rows before the first fit date, through T1's ``build_rows``.

    ``kind`` ``"v0.3"``: the SH filter, HYPERPARAMETER_START plus the SH
    new-parameter starts (s_shock 3.162e-4, h_reset 1e-5), rolling weights.
    ``kind`` ``"v0.2"``: the v0.2 filter (no SH injection), HYPERPARAMETER_START,
    daily rolling weights. One synthetic segment starting at ``start`` with the
    latest election known then as reference; each row reads only observations
    available by its own date. ``ctx`` is ``v03.v03_context()`` (or a
    truncated copy). Returns {day: (start, eight shares)}.
    """

    from scripts.poll_aggregate.build import _known

    from .t1_aggregate import NEW_STARTS, build_rows

    if kind == "v0.3":
        arm, theta = "SH", {**HYPERPARAMETER_START, **NEW_STARTS}
    elif kind == "v0.2":
        arm, theta = "C0", dict(HYPERPARAMETER_START)
    else:
        raise ValueError(f"unknown substitute kind {kind!r}")
    known = _known(ctx["everything"], start)
    reference = max((o for o in known if o.kind == "election"), key=lambda o: o.obs_date)
    segment = {"fit_date": start, "reference": reference, "known": known}
    # PREREG-AMBIGUITY: the house list is v0.3's (every house in the pinned
    # polls), as in T1's own causality check; houses not yet polled carry only
    # their prior and receive no consensus weight.
    rows = build_rows(arm, ctx["houses"], ctx["polls"], ctx["everything"], [segment], [theta], end, "rolling")
    return {d: (f, [float(x) for x in m]) for d, (f, m) in rows.items() if start <= d <= end}


def information_dates(everything, days, extra_change_dates=(), end: date | None = None) -> dict[date, date]:
    """The information date of each row: the last change date (publication or fit) on or before it,
    exactly the ``info_date`` from which ``build_rows`` reads the row's filter state."""

    end = end or max(days)
    changes = sorted({o.available for o in everything if o.available <= end} | set(extra_change_dates))
    out, i = {}, -1
    for d in sorted(days):
        while i + 1 < len(changes) and changes[i + 1] <= d:
            i += 1
        out[d] = changes[i]
    return out


def write_combined_timeseries(sub_rows: dict, info: dict, label: str, base_file: Path, path: Path) -> Path:
    """Substitute rows (2-decimal, production schema) followed by ``base_file``'s rows verbatim."""

    from .v03 import TIMESERIES_FIELDS

    with base_file.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != TIMESERIES_FIELDS:
            raise ValueError(f"{base_file} is not in the production timeseries schema")
        base_rows = list(reader)
    first = date.fromisoformat(base_rows[0]["date"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMESERIES_FIELDS, lineterminator="\n")
        writer.writeheader()
        for day in sorted(d for d in sub_rows if d < first):
            fit_date, mean = sub_rows[day]
            record = {p: f"{round(v, 2):.2f}" for p, v in zip(PARTIES, mean)}
            record.update({
                "date": day.isoformat(), "FI": "", "other": f"{round(100.0 - sum(mean), 2):.2f}",
                "source_extra_json": json.dumps({
                    "aggregate_version": label, "substitute": "calibration-2030 final round 6.3",
                    "hyperparameters": "HYPERPARAMETER_START" + (" + SH new-parameter starts" if "v0.3" in label else ""),
                    "information_date": info[day].isoformat(),
                }, sort_keys=True, separators=(",", ":")),
                "source_url": "", "retrieved_at": "",
            })
            writer.writerow(record)
        writer.writerows(base_rows)
    return path


# --------------------------------------------------------------------------- #
# Truth, geography, G6 dates
# --------------------------------------------------------------------------- #


def load_truth_t4() -> dict[int, dict]:
    """``replay.load_truth`` plus 2010 and 2014 national seats.

    ``replay.load_truth`` leaves 2010 and 2014 seats at zero, because the
    production mandates file starts in 2018.
    """

    # PREREG-AMBIGUITY: the 2010 and 2014 seat truth comes from the Part 2B
    # research file of certified mandates (Valmyndigheten source); it is not in
    # the §1.2 pinned inputs. Its SHA-256 is recorded with the run.
    truth = load_truth()
    seats: dict[int, dict[str, int]] = {}
    with PART2B_MANDATES.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            year = int(row["election_year"])
            if row["party"] in PARTIES:
                seats.setdefault(year, {p: 0 for p in PARTIES})[row["party"]] += int(row["total_seats"])
    for year in (2010, 2014):
        if sum(seats[year].values()) != 349:
            raise ValueError(f"Part 2B seats for {year} do not sum to 349")
        truth[year]["seats"] = seats[year]
    return truth


def build_geography_dir(parent: Path) -> Path:
    """Production's geography folder plus the 2006 and 2010 constituency baselines (Part 2B)."""

    from scripts.geography.config import DEFAULT_PROCESSED_GEOGRAPHY_DIR as GEO

    out = Path(parent) / "geography_historical"
    out.mkdir(parents=True)
    for child in GEO.iterdir():
        if child.name != GEOGRAPHY_VOTES_NAME:
            (out / child.name).symlink_to(child.resolve())
    production = (GEO / GEOGRAPHY_VOTES_NAME).read_text(encoding="utf-8").splitlines(keepends=True)
    extra = PART2B_VOTES.read_text(encoding="utf-8").splitlines(keepends=True)
    if production[0] != extra[0]:
        raise ValueError("Part 2B geography is not in production's schema")
    body = production + ([] if production[-1].endswith("\n") else ["\n"]) + extra[1:]
    (out / GEOGRAPHY_VOTES_NAME).write_text("".join(body), encoding="utf-8")
    return out


def selected_estimate_v12(data_root: str, as_of: date, info_lookup: dict[str, str]) -> dict:
    """``replay.selected_estimate`` for the 1.2.0 layout.

    The engine reads ``opinion_inputs(data_root)``, not ``pollofpolls/``, so
    ``estimate_opinion`` is called with those two files, exactly as the engine
    calls it. The selected values must equal that file's row; its information
    date comes from the row's ``source_extra_json`` or, for v0.3 rows written by
    ``v03.write_timeseries`` (which carry none), from ``info_lookup``.
    """

    # PREREG-AMBIGUITY: replay.selected_estimate reads pollofpolls/ (the
    # pre-1.2.0 layout), which production 1.2.0 no longer reads; the date check
    # is applied to the file the 1.2.0 engine actually reads.
    from scripts.pollofpolls.state import estimate_opinion
    from scripts.simulator.model_inputs import opinion_inputs

    inputs = opinion_inputs(Path(data_root) if data_root else ce.PROCESSED_ROOT)
    state = estimate_opinion(as_of=as_of, timeseries_file=inputs.timeseries, polls_file=inputs.polls)
    record = {
        "selected_estimate_date": state.estimate_date.isoformat(),
        "selected_estimate_pct": "|".join(f"{state.mean_pct[p]:.4f}" for p in PARTIES),
        "aggregate_file_sha256": sha256(inputs.timeseries),
    }
    row = _aggregate_row(Path(inputs.timeseries), record["selected_estimate_date"])
    record["aggregate_row_matches"] = bool(
        row is not None and all(abs(float(row[p]) - state.mean_pct[p]) <= 1e-6 for p in PARTIES))
    info = ""
    if row is not None:
        info = json.loads(row["source_extra_json"]).get("information_date", "") or info_lookup.get(row["date"], "")
    record["aggregate_information_date"] = info
    return record


# --------------------------------------------------------------------------- #
# One case
# --------------------------------------------------------------------------- #


def run_case(job: dict) -> dict:
    """One (arm, election, horizon): simulate, check integrity, score."""

    started = time.time()
    config = ce.CandidateConfig.from_dict(job["config"])
    election = date.fromisoformat(job["election"])
    as_of = election - timedelta(days=job["horizon"])
    row = {
        "arm": job["arm"], "election": job["election"], "year": election.year, "horizon_days": job["horizon"],
        "set": job["set"], "group": group_of(election.year), "as_of": as_of.isoformat(),
        "data_root": job["data_root"] or "production", "geography": job["geo_dir"] or "production",
    }
    try:
        result, national = ce.simulate_candidate(
            config, as_of=as_of, election_date=election, samples=job["draws"], seed=job["seed"],
            data_dir=job["data_root"] or None, processed_geo_dir=job["geo_dir"] or None,
        )
        row.update(selected_estimate_v12(job["data_root"], as_of, job["info_lookup"]))
    except Exception as exc:  # recorded; G6 requires every case to complete
        row.update({"completed": False, "error": f"{type(exc).__name__}: {exc}",
                    "seconds": round(time.time() - started, 1)})
        return row
    votes = np.asarray(result.vote_shares_matrix, dtype=float)
    seats = np.asarray(result.seats_matrix, dtype=np.int64)
    diag = national.diagnostics
    row.update({
        "completed": True,
        "opinion_as_of": str(result.summary.as_of),
        "seat_total_always_349": bool(np.all(seats.sum(axis=1) == 349)),
        "all_finite": bool(np.all(np.isfinite(votes))),
        "mean_vote_pct": "|".join(f"{v:.4f}" for v in votes.mean(axis=0)),
        "mean_seats": "|".join(f"{v:.3f}" for v in seats.mean(axis=0)),
        "noise_pool_years": list(national.training_years),
        "mean_lambda": float(np.mean(national.lambdas)),
        "k": diag.get("k"), "k_eligible_targets": diag.get("k_eligible_targets"),
        "layer_seeds": diag.get("layer_seeds"),
        "dynamics_eval_horizon": diag.get("production_diagnostics", {}).get("dynamics_eval_horizon"),
    })
    row["dates_ok"] = case_dates_ok(row)
    if job["truth"] is not None:
        from diagnostics.election_noise_v2.control_baseline.harness import metrics as M

        truth_vote = np.asarray(job["truth"]["vote"], dtype=float)
        truth_seats = np.asarray(job["truth"]["seats"], dtype=np.int64)
        row.update(M.d1_joint_vote_energy_score(votes, truth_vote))
        marginal = M.d2_marginal_vote_metrics(votes, truth_vote)
        row.update({k: v for k, v in marginal.items() if k != "per_party"})
        seat = M.d3_seat_metrics(seats, truth_seats)
        row.update({f"seat_{k}" if not k.startswith("seat") else k: v for k, v in seat.items() if k != "per_party"})
        row["brier_coalition"] = M.d4_coalition_brier(seats, truth_seats)["brier_mean_over_masks"]
        p_threshold = (votes[:, : len(PARTIES)] >= THRESHOLD_PCT).mean(axis=0)
        y_threshold = (truth_vote[: len(PARTIES)] >= THRESHOLD_PCT).astype(float)
        row["brier_threshold"] = float(np.mean((p_threshold - y_threshold) ** 2))
    row["seconds"] = round(time.time() - started, 1)
    return row


# --------------------------------------------------------------------------- #
# Gates (parent §4.4) — pure arithmetic on case rows
# --------------------------------------------------------------------------- #


def _ok(row: dict) -> bool:
    return bool(row.get("completed")) and bool(row.get("seat_total_always_349")) and bool(row.get("all_finite")) \
        and bool(row.get("dates_ok"))


def pooled(rows: list[dict], arm: str, metric: str, set_name: str, years: Sequence[int] = DECISION_YEARS) -> float:
    """Unweighted mean over the completed cases of a set (NaN if none)."""

    values = [float(r[metric]) for r in rows
              if r["arm"] == arm and r["set"] == set_name and r["year"] in years and r.get("completed")
              and r.get(metric) is not None]
    return float(np.mean(values)) if values else math.nan


def integrity(rows: list[dict], arm: str, sets: Sequence[str] = ("D", "L"), years: Sequence[int] = DECISION_YEARS) -> dict:
    expected = {(y, h, s) for s in sets for y in years for h in HORIZONS[s]}
    mine = [r for r in rows if r["arm"] == arm and (r["year"], r["horizon_days"], r["set"]) in expected]
    seen = {(r["year"], r["horizon_days"], r["set"]) for r in mine}
    failed = [f"{r['year']}/{r['set']}{r['horizon_days']}" for r in mine if not _ok(r)]
    return {"complete": seen == expected and len(mine) == len(expected), "failed_cases": failed,
            "missing_cases": sorted(f"{y}/{s}{h}" for y, h, s in expected - seen),
            "pass": seen == expected and len(mine) == len(expected) and not failed}


def evaluate_gates(rows: list[dict], arm: str, *, dynamics_changed: bool, provenance_ok: bool,
                   control: str = "C0", f1: str = "F1") -> dict:
    """G1–G7 for ``arm`` against C0 (and F1 where stated), plus the 2026 flag."""

    def p(a, metric, s="D", years=DECISION_YEARS):
        return pooled(rows, a, metric, s, years)

    g: dict[str, dict] = {}
    es, es_c, es_f = p(arm, "es_9cat"), p(control, "es_9cat"), p(f1, "es_9cat")
    g["G1_accuracy"] = {"arm": es, "C0": es_c, "F1": es_f, "pass": es <= 1.05 * es_c and es <= 1.05 * es_f}
    per = {}
    for y in DECISION_YEARS:
        a, c = p(arm, "es_9cat", years=(y,)), p(control, "es_9cat", years=(y,))
        per[str(y)] = {"arm": a, "C0": c, "pass": a <= 1.15 * c}
    g["G2_per_election"] = {"elections": per, "pass": all(v["pass"] for v in per.values())}
    seat, seat_c, seat_f = p(arm, "seat_energy_score"), p(control, "seat_energy_score"), p(f1, "seat_energy_score")
    coal, coal_c = p(arm, "brier_coalition"), p(control, "brier_coalition")
    g["G3_seats"] = {"seat_energy_score": {"arm": seat, "C0": seat_c, "F1": seat_f},
                     "brier_coalition": {"arm": coal, "C0": coal_c},
                     "pass": seat <= 1.05 * seat_c and seat <= 1.05 * seat_f and coal <= coal_c + 0.01}
    thr, thr_c = p(arm, "brier_threshold"), p(control, "brier_threshold")
    g["G4_thresholds"] = {"arm": thr, "C0": thr_c, "pass": thr <= thr_c + 0.01}
    c90, c90_c, c50 = p(arm, "coverage_90"), p(control, "coverage_90"), p(arm, "coverage_50")
    g["G5_calibration"] = {"coverage_90": {"arm": c90, "C0": c90_c}, "coverage_50": {"arm": c50},
                           "pass": (0.80 <= c90 <= 0.97 and abs(c90 - 0.90) < abs(c90_c - 0.90)
                                    and 0.35 <= c50 <= 0.65)}
    # PREREG-AMBIGUITY: G6 "every case completes" is read over the gated sets
    # (D and L, 2014–2022) for every compared arm (C0, F1 and the gated arm);
    # the reported 2010 and 2026 cases are recorded separately.
    checks = {a: integrity(rows, a) for a in dict.fromkeys((control, f1, arm))}
    g["G6_integrity"] = {"arms": checks, "provenance_clean_and_inputs_pinned": bool(provenance_ok),
                         "pass": bool(provenance_ok) and all(v["pass"] for v in checks.values())}
    es_l, es_l_c = p(arm, "es_9cat", "L"), p(control, "es_9cat", "L")
    c90_l, c90_l_c = p(arm, "coverage_90", "L"), p(control, "coverage_90", "L")
    g7_cal = abs(c90_l - 0.90) < abs(c90_l_c - 0.90)
    g["G7_long_horizon"] = {"es_vote": {"arm": es_l, "C0": es_l_c},
                            "coverage_90": {"arm": c90_l, "C0": c90_l_c},
                            "calibration_clause_applies": bool(dynamics_changed),
                            "pass": es_l <= 1.05 * es_l_c and (g7_cal if dynamics_changed else True)}
    e26, e26_c = p(arm, "es_9cat", "D", (2026,)), p(control, "es_9cat", "D", (2026,))
    flag = {"arm": e26, "C0": e26_c, "ratio": e26 / e26_c if e26_c else math.nan, "flagged": e26 > 1.15 * e26_c}
    passed = all(v["pass"] for v in g.values())
    return {"arm": arm, "gates": g, "pass": passed, "decision": "PASS" if passed else "FAIL", "flag_2026": flag}


def pooled_table(rows: list[dict], arms: Sequence[str]) -> dict:
    metrics = ("es_9cat", "crps_8party_mean", "seat_energy_score", "brier_coalition", "brier_threshold",
               "coverage_50", "coverage_90")
    out = {}
    for a in arms:
        out[a] = {}
        for label, s, years in (("D", "D", DECISION_YEARS), ("L", "L", DECISION_YEARS),
                                ("2026_D", "D", (2026,)), ("2026_L", "L", (2026,)),
                                ("2010_D", "D", (2010,)), ("2010_L", "L", (2010,))):
            out[a][label] = {m: pooled(rows, a, m, s, years) for m in metrics}
    return out


# --------------------------------------------------------------------------- #
# CAND selection
# --------------------------------------------------------------------------- #


def _selected_value(record: dict):
    # Interface assumption: T2 and T3-rev record their selection under
    # "selected" at the top level or under "decision"/"selection".
    for path in (("selected",), ("decision", "selected"), ("selection", "selected"),
                 ("decision", "selected_arm"), ("selection", "selected_arm"), ("selected_arm",)):
        node = record
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, str):
            return node
    raise KeyError("no selection found in the results record")


def load_selection(cand_config: Path | None) -> tuple[dict | None, dict]:
    """The CAND selection ``{"t2_arm", "noise_law"}`` and where it came from."""

    if cand_config is not None:
        record = json.loads(Path(cand_config).read_text(encoding="utf-8"))
        return {"t2_arm": record["t2_arm"], "noise_law": record["noise_law"]}, {
            "source": str(cand_config), "sha256": sha256(Path(cand_config))}
    if T2_RESULTS.exists() and T3REV_RESULTS.exists():
        t2 = json.loads(T2_RESULTS.read_text(encoding="utf-8"))
        t3 = json.loads(T3REV_RESULTS.read_text(encoding="utf-8"))
        return {"t2_arm": _selected_value(t2), "noise_law": _selected_value(t3)}, {
            "source": "t2_results.json + t3rev_results.json",
            "t2_results_sha256": sha256(T2_RESULTS), "t3rev_results_sha256": sha256(T3REV_RESULTS)}
    return None, {"source": None}


# --------------------------------------------------------------------------- #
# Inputs for one run
# --------------------------------------------------------------------------- #


def prepare_inputs(tmp: Path, *, need_v03: bool, need_2010: bool, geography: str) -> dict:
    """v0.3 file, 2010 substitute files, data roots and geography, with their hashes."""

    from scripts.simulator.model_inputs import AGGREGATE_TIMESERIES

    from . import v03

    out: dict = {"roots": {"v0.2": ""}, "files": {"v0.2": str(ce.PROCESSED_ROOT / AGGREGATE_TIMESERIES)},
                 "info": {"v0.2": {}}, "sha256": {}, "geo_dir": ""}
    ctx = v03.v03_context()
    if need_v03 or need_2010:
        rows = v03.v03_rows(ctx)
        check = v03.self_check(rows)
        if not check["ok"]:
            raise RuntimeError(f"v0.3 self-check failed: {check}")
        out["v03_self_check"] = check
        v03_file = v03.write_timeseries(rows, tmp / "v03" / "swedishpolls_aggregate_timeseries.csv")
        out["files"]["v0.3"] = str(v03_file)
        out["roots"]["v0.3"] = str(ce.build_opinion_root(tmp, "root_v03", v03_file))
        out["info"]["v0.3"] = {d.isoformat(): i.isoformat() for d, i in information_dates(
            ctx["everything"], list(rows), [s["fit_date"] for s in ctx["segments"]]).items()}
    if need_2010:
        for kind, base in (("v0.2", Path(out["files"]["v0.2"])), ("v0.3", Path(out["files"]["v0.3"]))):
            sub = substitute_rows(kind, ctx)
            info = information_dates(ctx["everything"], list(sub), [SUBSTITUTE_START], SUBSTITUTE_END)
            label = f"SwedishPollsAggregate-{kind}-substitute-2007"
            path = write_combined_timeseries(sub, info, label, base, tmp / f"{kind}-2010" / "timeseries.csv")
            key = f"{kind}-2010"
            out["files"][key] = str(path)
            out["roots"][key] = str(ce.build_opinion_root(tmp, f"root_{kind}_2010", path))
            out["info"][key] = dict(out["info"].get(kind, {}))
    for key, path in out["files"].items():
        out["sha256"][key] = sha256(Path(path))
    if geography == "historical":
        out["geo_dir"] = str(build_geography_dir(tmp))
        out["sha256"]["part2b_geography"] = sha256(PART2B_VOTES)
    out["sha256"]["part2b_mandates"] = sha256(PART2B_MANDATES)
    return out


def make_jobs(arm: str, config: ce.CandidateConfig, cases, inputs: dict, truth, draws: int, seed: int) -> list[dict]:
    jobs = []
    for year, horizon, set_name in cases:
        election = ELECTION_DAYS[year]
        key = config.aggregate + ("-2010" if year == 2010 else "")
        as_of = (election - timedelta(days=horizon)).isoformat()
        case_truth = None
        if truth is not None:
            record = truth[year]
            case_truth = {"vote": [record["vote"][c] for c in CATEGORIES],
                          "seats": [record["seats"][p] for p in PARTIES]}
        jobs.append({
            "arm": arm, "config": config.to_dict(), "election": election.isoformat(), "horizon": horizon,
            "set": set_name, "data_root": inputs["roots"][key],
            "geo_dir": inputs["geo_dir"] if year in (2010, 2014) else "",
            "info_lookup": {as_of: inputs["info"][key].get(as_of, "")} if inputs["info"][key] else {},
            "draws": draws, "seed": seed, "truth": case_truth,
        })
    return jobs


def run_jobs(jobs: list[dict], workers: int) -> list[dict]:
    if workers <= 1:
        return [run_case(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_case, jobs))


def self_check(samples: int = SELF_CHECK_DRAWS, seed: int = SEED, cases=SELF_CHECK_CASES,
               config: ce.CandidateConfig | None = None) -> dict:
    """§6.4: the candidate path configured as C0 reproduces production's bytes."""

    from scripts.simulator.engine import simulate_election

    config = config or ce.config_c0()
    out = {"config": config.to_dict(), "samples": samples, "seed": seed, "cases": []}
    for year, horizon in cases:
        election = ELECTION_DAYS[year]
        as_of = election - timedelta(days=horizon)
        reference = simulate_election(as_of=as_of, election_date=election, samples=samples, seed=seed)
        candidate, _ = ce.simulate_candidate(config, as_of=as_of, election_date=election, samples=samples, seed=seed)
        out["cases"].append({
            "election": election.isoformat(), "horizon_days": horizon,
            "votes_identical": reference.vote_shares_matrix.tobytes() == candidate.vote_shares_matrix.tobytes(),
            "seats_identical": reference.seats_matrix.tobytes() == candidate.seats_matrix.tobytes(),
        })
    out["ok"] = all(c["votes_identical"] and c["seats_identical"] for c in out["cases"])
    return out


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (date, Path)):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="three cases at n = 1,000, written under /tmp")
    parser.add_argument("--cand-config", type=Path, default=None, help='JSON {"t2_arm": ..., "noise_law": ...}')
    parser.add_argument("--geography", choices=("production", "historical"), default="production",
                        help="production: the engine's geography (it has no 2006/2010 baseline, so 2010 and "
                             "2014 cases fail); historical: add the Part 2B 2006/2010 baselines (needs authorization)")
    args = parser.parse_args(argv)

    from .common import verify_final_round

    provenance = verify_final_round(allow_dirty=args.smoke)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.time()
    out_path = args.out or (SMOKE_OUTPUT if args.smoke else OUTPUT)
    if args.smoke and not str(out_path.resolve()).startswith("/tmp/"):
        parser.error("--smoke writes under /tmp only")
    draws = 1_000 if args.smoke else DRAWS
    cases = list(SMOKE_CASES) if args.smoke else all_cases()

    noise_available = ce.noise_laws_module() is not None
    selection, selection_source = load_selection(args.cand_config)
    if not args.smoke:
        if not noise_available:
            parser.error(f"{ce.NOISE_LAWS_MODULE} is required (F1, FB and CAND use it)")
        if selection is None:
            parser.error("CAND is not configured: no t2_results.json + t3rev_results.json and no --cand-config")
    configs: dict[str, ce.CandidateConfig] = {"C0": ce.config_c0()}
    if noise_available:
        configs["F1"] = ce.config_f1()
    if selection is not None:
        configs["CAND"] = ce.config_cand(selection)
    identical = selection is not None and ce.cand_equals_fb(selection)
    if noise_available:
        fb = ce.config_fb()
    else:
        fb = None

    check = self_check(samples=1_000 if args.smoke else SELF_CHECK_DRAWS,
                       cases=SELF_CHECK_CASES[:1] if args.smoke else SELF_CHECK_CASES)
    print(json.dumps({"self_check_ok": check["ok"]}))
    if not check["ok"] and not args.smoke:
        raise SystemExit(f"C0 self-check failed: {json.dumps(check)}")

    truth = load_truth_t4()
    need_v03 = any(c.aggregate == "v0.3" for c in configs.values()) or fb is not None
    need_2010 = any(y == 2010 for y, _, _ in cases)
    with tempfile.TemporaryDirectory(prefix="calibration-2030-t4-") as tmp:
        inputs = prepare_inputs(Path(tmp), need_v03=need_v03, need_2010=need_2010, geography=args.geography)
        phase1 = [j for arm, cfg in configs.items() for j in make_jobs(arm, cfg, cases, inputs, truth, draws, SEED)]
        if args.smoke and fb is not None and "CAND" not in configs:
            phase1 += make_jobs("FB", fb, cases, inputs, truth, draws, SEED)
        rows = run_jobs(phase1, args.workers)
        provenance_ok = bool(provenance["tracked_clean"])
        gate_results: dict[str, dict] = {}
        dynamics_changed = configs["CAND"].dynamics_differ_from_d0 if "CAND" in configs else False
        fb_status = "not run"
        if "CAND" in configs and "F1" in configs:
            gate_results["CAND"] = evaluate_gates(rows, "CAND", dynamics_changed=dynamics_changed,
                                                  provenance_ok=provenance_ok)
            if identical:
                gate_results["FB"] = {**gate_results["CAND"], "arm": "FB", "recorded_from": "CAND (section 1.4 coincidence)"}
                fb_status = "identical to CAND; run once and recorded for both"
            elif not gate_results["CAND"]["pass"] and fb is not None:
                rows += run_jobs(make_jobs("FB", fb, cases, inputs, truth, draws, SEED), args.workers)
                gate_results["FB"] = evaluate_gates(rows, "FB", dynamics_changed=False, provenance_ok=provenance_ok)
                fb_status = "run because CAND failed"
            else:
                fb_status = "not run: CAND passed"
        elif args.smoke and "F1" in configs and any(r["arm"] == "FB" for r in rows):
            gate_results["FB"] = evaluate_gates(rows, "FB", dynamics_changed=False, provenance_ok=provenance_ok)
            fb_status = "smoke stand-in (CAND not configured)"
        inputs_record = {k: v for k, v in inputs.items() if k not in ("info",)}

    if "CAND" in gate_results and gate_results["CAND"]["pass"]:
        decision = "CAND passes: propose CAND (parent 4.5)"
    elif "FB" in gate_results and gate_results["FB"]["pass"] and not identical:
        decision = "CAND fails, FB passes: propose FB (section 1.4)"
    elif "CAND" in gate_results:
        decision = "CAND and FB fail: production stays 1.2.0-rc1; only F1 may be proposed (section 1.4)"
    else:
        decision = "not evaluated"
    arms_run = list(dict.fromkeys(r["arm"] for r in rows))
    record = {
        "tier": "T4", "preregistration": "docs/calibration_2030_final_round_preregistration.md",
        "parent": "docs/calibration_2030_preregistration.md", "smoke": args.smoke,
        "started_utc": started, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - t0, 1), "provenance": provenance, "draws": draws, "seed": SEED,
        "geography": args.geography, "noise_laws_available": noise_available,
        "cand_selection": selection, "cand_selection_source": selection_source, "cand_equals_fb": identical,
        "configs": {a: c.to_dict() for a, c in {**configs, **({"FB": fb} if fb else {})}.items()},
        "self_check": check, "inputs": inputs_record, "fb_status": fb_status,
        "pooled": pooled_table(rows, arms_run), "gates": gate_results, "decision": decision,
        "cases": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_path), "decision": decision, "fb": fb_status,
                      "failed_cases": [f"{r['arm']} {r['year']} {r['set']}{r['horizon_days']}: {r.get('error')}"
                                       for r in rows if not r.get("completed")],
                      "seconds": record["seconds"]}, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
