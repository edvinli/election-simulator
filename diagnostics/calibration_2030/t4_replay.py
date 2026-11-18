"""T4 of the calibration-2030 final round: end-to-end replay of C0, F1, CAND (and FB).

Implements ``docs/calibration_2030_final_round_preregistration.md`` §6 with
the gates G1–G7 and the 2026 flag of the parent
``docs/calibration_2030_preregistration.md`` §4.4 (protocol v4 §6 metrics),
as amended by calibration amendment 002:

* 2018, 2022 and 2026 run ``scripts.simulator.engine.simulate_election`` end
  to end through ``candidate_engine.simulate_candidate``;
* 2014 uses the arm's national draws and the validated historical seat map
  ``isolated.votes_to_seats(votes_pct, 2014)`` (PRE_2018 law), with the
  certified 2014 seats as truth, and enters every gate;
* 2010 is reported only, with vote metrics only (no seats). An arm whose
  state cannot be computed at a 2010 origin records ``not computable``.

    uv run python -m diagnostics.calibration_2030.t4_replay [--workers N] [--out PATH]
        [--smoke] [--cand-config PATH]

``--smoke`` runs C0, F1 and CAND on four cases at n = 1,000, writes under
/tmp and may run on a modified checkout. It never writes the results file.

Reused from ``scripts/poll_aggregate/replay.py``: ``load_truth``,
``case_dates_ok`` (G6's date condition, protocol v4 §7.1), ``_aggregate_row``,
the threshold and the frozen metric calls. ``selected_estimate`` is reproduced
for the 1.2.0 input layout (see ``selected_estimate_v12``).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import multiprocessing
import os
import resource
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
AMENDMENT_002 = ROOT / "data" / "processed" / "calibration_2030" / "amendments" / "002-t4-2014-seat-path.json"

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
SMOKE_CASES = ((2014, 28, "D"), (2018, 7, "D"), (2022, 730, "L"), (2010, 112, "D"))
CATEGORIES = (*PARTIES, "REST")
ISO_YEAR = 2014       # amendment 002: national draws -> votes_to_seats (PRE_2018)
VOTES_ONLY_YEAR = 2010  # amendment 002: vote metrics only, reported
#: Rough per-case cost at 20,000 draws (seconds), only to schedule the longest cases first.
COST_HINT = {2014: 450, 2026: 330, 2018: 90, 2022: 90, 2010: 30}
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
#: Concurrent metric computations across workers. The frozen energy score at
#: 20,000 draws peaks at about 1.7 GB RSS (chunked pairwise distances), so with
#: 12 workers an unbounded peak could pass 20 GB; 6 slots keep the total near
#: 6 x 1.7 + 12 x ~0.6 GB.
DEFAULT_METRIC_SLOTS = 6
_METRIC_SLOT = None


def _init_worker(slot) -> None:
    global _METRIC_SLOT
    _METRIC_SLOT = slot


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
# Amendment 002: the 2014 seat path and 2010 vote-only cases
# --------------------------------------------------------------------------- #


def verify_amendment_002(path: Path = AMENDMENT_002) -> dict:
    """Refuse to run unless amendment 002 is in effect and every file it pins matches.

    Checks the record against its SHA-256 companion and the amendment index,
    the text against ``text_sha256``, and each ``pinned_sha256`` entry.
    """

    from .common import IntegrityError

    record_bytes = path.read_bytes()
    record_sha = hashlib.sha256(record_bytes).hexdigest()
    companion = path.with_suffix(".sha256")
    if not companion.exists() or companion.read_text(encoding="utf-8").split()[0] != record_sha:
        raise IntegrityError(f"amendment 002 record does not match its SHA-256 companion ({companion.name})")
    index = json.loads((path.parent / "index.json").read_text(encoding="utf-8"))
    entries = [e for e in index["amendments"] if e.get("record") == path.name]
    if len(entries) != 1 or entries[0].get("record_sha256") != record_sha:
        raise IntegrityError("amendment 002 is not indexed with this record's SHA-256")
    record = json.loads(record_bytes)
    if record.get("status") != "final" or record.get("in_effect") is not True:
        raise IntegrityError("amendment 002 is not final and in effect")
    if sha256(ROOT / record["text"]) != record["text_sha256"]:
        raise IntegrityError("amendment 002 text does not match text_sha256")
    mismatched = sorted(p for p, h in record["pinned_sha256"].items() if sha256(ROOT / p) != h)
    if mismatched:
        raise IntegrityError(f"amendment 002 pinned files changed: {mismatched}")
    return {"record": str(path.relative_to(ROOT)), "record_sha256": record_sha,
            "pinned_sha256": record["pinned_sha256"], "text_sha256": record["text_sha256"]}


def _isolated():
    from diagnostics.election_noise_v2.control_baseline_amendment2.harness2 import isolated

    return isolated


def load_truth_t4() -> dict[int, dict]:
    """``replay.load_truth`` with the certified 2014 national seats (amendment 002).

    ``replay.load_truth`` leaves 2010 and 2014 seats at zero, because the
    production mandates file starts in 2018. 2014 takes
    ``isolated.certified_seats(2014)``; 2010 has no seat truth (vote metrics only).
    """

    truth = load_truth()
    seats_2014 = _isolated().certified_seats(2014)
    if sum(seats_2014.values()) != 349:
        raise ValueError("certified 2014 seats do not sum to 349")
    truth[2014]["seats"] = {p: int(seats_2014[p]) for p in PARTIES}
    truth[2010]["seats"] = None
    return truth


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


def _vote_metrics(votes: np.ndarray, truth_vote: np.ndarray) -> dict:
    from diagnostics.election_noise_v2.control_baseline.harness import metrics as M

    out = dict(M.d1_joint_vote_energy_score(votes, truth_vote))
    marginal = M.d2_marginal_vote_metrics(votes, truth_vote)
    out.update({k: v for k, v in marginal.items() if k != "per_party"})
    p_threshold = (votes[:, : len(PARTIES)] >= THRESHOLD_PCT).mean(axis=0)
    y_threshold = (truth_vote[: len(PARTIES)] >= THRESHOLD_PCT).astype(float)
    out["brier_threshold"] = float(np.mean((p_threshold - y_threshold) ** 2))
    return out


def _seat_metrics(seats: np.ndarray, truth_seats: np.ndarray) -> dict:
    from diagnostics.election_noise_v2.control_baseline.harness import metrics as M

    seat = M.d3_seat_metrics(seats, truth_seats)
    out = {f"seat_{k}" if not k.startswith("seat") else k: v for k, v in seat.items() if k != "per_party"}
    out["brier_coalition"] = M.d4_coalition_brier(seats, truth_seats)["brier_mean_over_masks"]
    return out


def case_path(year: int) -> str:
    """Amendment 002: which path a case's election takes."""

    if year == ISO_YEAR:
        return "national + isolated.votes_to_seats (PRE_2018), amendment 002"
    if year == VOTES_ONLY_YEAR:
        return "national only, vote metrics only, amendment 002"
    return "simulate_election (production engine)"


def run_case(job: dict) -> dict:
    """One (arm, election, horizon): simulate, check integrity, score."""

    started = time.time()
    config = ce.CandidateConfig.from_dict(job["config"])
    election = date.fromisoformat(job["election"])
    year = election.year
    as_of = election - timedelta(days=job["horizon"])
    row = {
        "arm": job["arm"], "election": job["election"], "year": year, "horizon_days": job["horizon"],
        "set": job["set"], "group": group_of(year), "as_of": as_of.isoformat(),
        "data_root": job["data_root"] or "production", "path": case_path(year),
    }
    if job.get("not_computable"):
        row.update({"completed": False, "status": "not computable", "error": job["not_computable"],
                    "seconds": 0.0})
        return row
    t2_k = tuple(job["t2_k"]) if job.get("t2_k") else None
    try:
        if year in (ISO_YEAR, VOTES_ONLY_YEAR):
            national = ce.candidate_national_vote_shares(
                config, as_of=as_of, election_date=election, samples=job["draws"], seed=job["seed"],
                data_dir=job["data_root"] or None, t2_k=t2_k,
            )
            votes = national.nat_shares_matrix * 100.0  # exactly as the engine forms its vote matrix
            seats = _isolated().votes_to_seats(votes, ISO_YEAR) if year == ISO_YEAR else None
            opinion_as_of = national.as_of.isoformat()
        else:
            result, national = ce.simulate_candidate(
                config, as_of=as_of, election_date=election, samples=job["draws"], seed=job["seed"],
                data_dir=job["data_root"] or None, t2_k=t2_k,
            )
            votes, seats = result.vote_shares_matrix, result.seats_matrix
            opinion_as_of = str(result.summary.as_of)
        row.update(selected_estimate_v12(job["data_root"], as_of, job["info_lookup"]))
    except Exception as exc:  # recorded; G6 requires every case to complete
        row.update({"completed": False, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                    "seconds": round(time.time() - started, 1)})
        return row
    votes = np.asarray(votes, dtype=float)
    diag = national.diagnostics
    row.update({
        "completed": True, "status": "completed", "opinion_as_of": opinion_as_of,
        "all_finite": bool(np.all(np.isfinite(votes))),
        "vote_rows_sum_to_100": bool(np.allclose(votes.sum(axis=1), 100.0, rtol=0, atol=1e-9)),
        "mean_vote_pct": "|".join(f"{v:.4f}" for v in votes.mean(axis=0)),
        "noise_pool_years": list(national.training_years),
        "mean_lambda": float(np.mean(national.lambdas)),
        "k": diag.get("k"), "k_eligible_targets": diag.get("k_eligible_targets"),
        "layer_seeds": diag.get("layer_seeds"),
        "dynamics_eval_horizon": diag.get("production_diagnostics", {}).get("dynamics_eval_horizon"),
    })
    if seats is not None:
        seats = np.asarray(seats, dtype=np.int64)
        row["seat_total_always_349"] = bool(np.all(seats.sum(axis=1) == 349))
        row["mean_seats"] = "|".join(f"{v:.3f}" for v in seats.mean(axis=0))
    else:
        row["seat_total_always_349"] = None  # 2010: no seats (amendment 002)
    row["dates_ok"] = case_dates_ok(row)
    if job["truth"] is not None:
        waited = time.time()
        with _METRIC_SLOT if _METRIC_SLOT is not None else contextlib.nullcontext():
            row["metric_wait_seconds"] = round(time.time() - waited, 1)
            row.update(_vote_metrics(votes, np.asarray(job["truth"]["vote"], dtype=float)))
            if seats is not None:
                row.update(_seat_metrics(seats, np.asarray(job["truth"]["seats"], dtype=np.int64)))
    row["seconds"] = round(time.time() - started, 1)
    row["max_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
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


#: Where each results record states its selection (fixed by the lead, 2026-11-18).
T2_SELECTION_PATH = ("decision", "selection", "selected")
T3REV_SELECTION_PATH = ("decision", "selected")


def _at(record: dict, path: Sequence[str]) -> str:
    node = record
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"selection field {'.'.join(path)} missing")
        node = node[key]
    if not isinstance(node, str):
        raise TypeError(f"selection field {'.'.join(path)} is not a string")
    return node


def load_selection(cand_config: Path | None, t2_results: Path = T2_RESULTS,
                   t3rev_results: Path = T3REV_RESULTS) -> tuple[dict | None, dict]:
    """The CAND selection ``{"t2_arm", "noise_law"}`` and where it came from (with SHA-256)."""

    if cand_config is not None:
        record = json.loads(Path(cand_config).read_text(encoding="utf-8"))
        return {"t2_arm": record["t2_arm"], "noise_law": record["noise_law"]}, {
            "source": str(cand_config), "sha256": sha256(Path(cand_config))}
    if t2_results.exists() and t3rev_results.exists():
        t2 = json.loads(t2_results.read_text(encoding="utf-8"))
        t3 = json.loads(t3rev_results.read_text(encoding="utf-8"))
        return {"t2_arm": _at(t2, T2_SELECTION_PATH), "noise_law": _at(t3, T3REV_SELECTION_PATH)}, {
            "source": "t2_results.json decision.selection.selected + t3rev_results.json decision.selected",
            "t2_results_sha256": sha256(t2_results), "t3rev_results_sha256": sha256(t3rev_results)}
    return None, {"source": None}


# --------------------------------------------------------------------------- #
# Inputs for one run
# --------------------------------------------------------------------------- #


def prepare_inputs(tmp: Path, *, need_v03: bool, need_2010: bool) -> dict:
    """v0.3 file, 2010 substitute files and data roots, with their hashes."""

    from scripts.simulator.model_inputs import AGGREGATE_TIMESERIES

    from . import v03

    out: dict = {"roots": {"v0.2": ""}, "files": {"v0.2": str(ce.PROCESSED_ROOT / AGGREGATE_TIMESERIES)},
                 "info": {"v0.2": {}}, "sha256": {}}
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
    return out


def t2_origin_values(config: ce.CandidateConfig, origins: Sequence[date]) -> dict[str, dict]:
    """For a U arm: k at each origin (``inflation_k``), or why the state cannot be computed there.

    k is deterministic and costly (it scores every earlier k = 1 cell), so it
    is computed once here, in increasing origin order on one shared context,
    and handed to the cases.
    """

    if not config.dynamics_differ_from_d0:
        return {}
    t2 = ce.t2_arms_module()
    ctx = ce.t2_context()
    out = {}
    for origin in sorted(set(origins)):
        try:
            t2.origin_state(ctx, origin)
        except ValueError as exc:
            out[origin.isoformat()] = {"not_computable": f"{config.state_dynamics} state at {origin}: {exc}"}
            continue
        k, count = t2.inflation_k(config.state_dynamics, ctx, origin)
        out[origin.isoformat()] = {"t2_k": [float(k), int(count)]}
    return out


def make_jobs(arm: str, config: ce.CandidateConfig, cases, inputs: dict, truth, draws: int, seed: int,
              t2_values: dict | None = None) -> list[dict]:
    jobs = []
    for year, horizon, set_name in cases:
        election = ELECTION_DAYS[year]
        key = config.aggregate + ("-2010" if year == 2010 else "")
        as_of = (election - timedelta(days=horizon)).isoformat()
        case_truth = None
        if truth is not None:
            record = truth[year]
            case_truth = {"vote": [record["vote"][c] for c in CATEGORIES],
                          "seats": None if record["seats"] is None else [record["seats"][p] for p in PARTIES]}
        extra = (t2_values or {}).get(as_of, {})
        jobs.append({
            "arm": arm, "config": config.to_dict(), "election": election.isoformat(), "horizon": horizon,
            "set": set_name, "data_root": inputs["roots"][key],
            "info_lookup": {as_of: inputs["info"][key].get(as_of, "")} if inputs["info"][key] else {},
            "draws": draws, "seed": seed, "truth": case_truth,
            "t2_k": extra.get("t2_k"), "not_computable": extra.get("not_computable"),
        })
    return jobs


def _by_cost(jobs: list[dict]) -> list[dict]:
    return sorted(jobs, key=lambda j: -COST_HINT[date.fromisoformat(j["election"]).year])


def _pool(workers: int, metric_slots: int = DEFAULT_METRIC_SLOTS) -> ProcessPoolExecutor:
    """Spawned single-threaded workers (one BLAS thread each, so N workers use N
    cores) sharing a semaphore that bounds concurrent metric computations."""

    for var in THREAD_VARS:
        os.environ.setdefault(var, "1")
    context = multiprocessing.get_context("spawn")
    slot = context.Semaphore(max(1, min(metric_slots, workers)))
    return ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker, initargs=(slot,))


def run_jobs(jobs: list[dict], workers: int, metric_slots: int = DEFAULT_METRIC_SLOTS) -> list[dict]:
    if workers <= 1:
        return [run_case(j) for j in _by_cost(jobs)]
    with _pool(workers, metric_slots) as pool:
        return list(pool.map(run_case, _by_cost(jobs)))


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
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="C0, F1, CAND on four cases at n = 1,000, under /tmp")
    parser.add_argument("--cand-config", type=Path, default=None, help='JSON {"t2_arm": ..., "noise_law": ...}')
    parser.add_argument("--metric-slots", type=int, default=DEFAULT_METRIC_SLOTS,
                        help="concurrent metric computations (each peaks near 1.7 GB at 20,000 draws)")
    args = parser.parse_args(argv)

    from .common import verify_final_round

    provenance = verify_final_round(allow_dirty=args.smoke)
    amendment = verify_amendment_002()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.time()
    out_path = args.out or (SMOKE_OUTPUT if args.smoke else OUTPUT)
    if args.smoke and not str(out_path.resolve()).startswith("/tmp/"):
        parser.error("--smoke writes under /tmp only")
    draws = 1_000 if args.smoke else DRAWS
    cases = list(SMOKE_CASES) if args.smoke else all_cases()

    if ce.noise_laws_module() is None:
        parser.error(f"{ce.NOISE_LAWS_MODULE} is required (C0's LW0, F1, FB and CAND use it)")
    selection, selection_source = load_selection(args.cand_config)
    if selection is None:
        parser.error("CAND is not configured: no t2_results.json + t3rev_results.json and no --cand-config")
    configs = {"C0": ce.config_c0(), "F1": ce.config_f1(), "CAND": ce.config_cand(selection)}
    fb = ce.config_fb()
    identical = ce.cand_equals_fb(selection)

    check = self_check(samples=1_000 if args.smoke else SELF_CHECK_DRAWS,
                       cases=SELF_CHECK_CASES[:1] if args.smoke else SELF_CHECK_CASES)
    print(json.dumps({"self_check_ok": check["ok"]}), flush=True)
    if not check["ok"] and not args.smoke:
        raise SystemExit(f"C0 self-check failed: {json.dumps(check)}")

    truth = load_truth_t4()
    need_2010 = any(y == 2010 for y, _, _ in cases)
    timings: dict[str, float] = {}
    with tempfile.TemporaryDirectory(prefix="calibration-2030-t4-") as tmp:
        inputs = prepare_inputs(Path(tmp), need_v03=True, need_2010=need_2010)
        base_jobs = [j for arm in ("C0", "F1") for j in make_jobs(arm, configs[arm], cases, inputs, truth, draws, SEED)]
        origins = [ELECTION_DAYS[y] - timedelta(days=h) for y, h, _ in cases]
        pool = _pool(args.workers, args.metric_slots)
        try:
            # C0 and F1 run while the parent computes CAND's per-origin k.
            futures = [pool.submit(run_case, j) for j in _by_cost(base_jobs)]
            t_k = time.time()
            t2_values = t2_origin_values(configs["CAND"], origins)
            timings["cand_k_precompute_seconds"] = round(time.time() - t_k, 1)
            cand_jobs = make_jobs("CAND", configs["CAND"], cases, inputs, truth, draws, SEED, t2_values)
            futures += [pool.submit(run_case, j) for j in _by_cost(cand_jobs)]
            rows = [f.result() for f in futures]
        finally:
            pool.shutdown()
        timings["phase1_seconds"] = round(time.time() - t0, 1)
        provenance_ok = bool(provenance["tracked_clean"])
        gate_results: dict[str, dict] = {}
        fb_status = "not run (smoke)"
        if not args.smoke:
            gate_results["CAND"] = evaluate_gates(rows, "CAND", dynamics_changed=configs["CAND"].dynamics_differ_from_d0,
                                                  provenance_ok=provenance_ok)
            if identical:
                gate_results["FB"] = {**gate_results["CAND"], "arm": "FB",
                                      "recorded_from": "CAND (section 1.4 coincidence)"}
                fb_status = "identical to CAND; run once and recorded for both"
            elif not gate_results["CAND"]["pass"]:
                rows += run_jobs(make_jobs("FB", fb, cases, inputs, truth, draws, SEED), args.workers,
                                 args.metric_slots)
                gate_results["FB"] = evaluate_gates(rows, "FB", dynamics_changed=False, provenance_ok=provenance_ok)
                fb_status = "run because CAND failed"
            else:
                fb_status = "not run: CAND passed"
        inputs_record = {k: v for k, v in inputs.items() if k != "info"}

    if args.smoke:
        decision = "smoke: not evaluated"
    elif gate_results["CAND"]["pass"]:
        decision = "CAND passes: propose CAND (parent 4.5)"
    elif "FB" in gate_results and gate_results["FB"]["pass"] and not identical:
        decision = "CAND fails, FB passes: propose FB (section 1.4)"
    else:
        decision = "CAND and FB fail: production stays 1.2.0-rc1; only F1 may be proposed (section 1.4)"
    arms_run = list(dict.fromkeys(r["arm"] for r in rows))
    record = {
        "tier": "T4", "preregistration": "docs/calibration_2030_final_round_preregistration.md",
        "parent": "docs/calibration_2030_preregistration.md", "amendment_002": amendment, "smoke": args.smoke,
        "started_utc": started, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - t0, 1), "timings": timings, "workers": args.workers,
        "metric_slots": args.metric_slots,
        "provenance": provenance, "draws": draws, "seed": SEED,
        "cand_selection": selection, "cand_selection_source": selection_source, "cand_equals_fb": identical,
        "configs": {a: c.to_dict() for a, c in {**configs, "FB": fb}.items()},
        "self_check": check, "inputs": inputs_record, "fb_status": fb_status,
        "pooled": pooled_table(rows, arms_run), "gates": gate_results, "decision": decision,
        "cases": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_path), "decision": decision, "fb": fb_status, "timings": timings,
                      "cases": [f"{r['arm']} {r['year']} {r['set']}{r['horizon_days']}: {r['status']} "
                                f"{r['seconds']}s rss={r.get('max_rss_mb')}MB {r.get('error') or ''}" for r in rows],
                      "seconds": record["seconds"]}, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
