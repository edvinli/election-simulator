"""Calibration amendment 003: the one post-hoc exception arm U0 (Ua without bias correction).

U0 = v0.3 + Ua (T2's selection) + LW0 (production's unbiased law), fitted per
target on the corrected residual pool. It runs with C0 and F1 on T4's frozen
cases, seed, draws and gates. It reuses ``t4_replay``'s machinery unchanged:
the frozen T4 ``main()`` is not touched. C0 and F1 must reproduce the recorded
T4 run exactly, and any difference is reported.

Run with ``uv run python -m diagnostics.calibration_2030.t4_exception_u0 [--workers 12] [--metric-slots 5]``.
It writes ``t4_exception_u0_results.json`` beside this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Sequence

from . import candidate_engine as ce
from . import t4_replay as t4

HERE = Path(__file__).resolve().parent
AMENDMENT_003 = t4.ROOT / "data" / "processed" / "calibration_2030" / "amendments" / "003-t4-exception-arm-u0.json"
OUTPUT = HERE / "t4_exception_u0_results.json"
RECORDED_T4 = HERE / "t4_results.json"
COMPARED_METRICS = ("es_9cat", "crps_8party_mean", "brier_threshold", "coverage_50", "coverage_90",
                    "seat_energy_score", "brier_coalition")


def config_u0() -> ce.CandidateConfig:
    return ce.CandidateConfig("U0", aggregate="v0.3", state_dynamics="Ua", noise="LW0", pool="corrected",
                              simplex_map=True, t2_arm="Ua")


def verify_amendment_003(path: Path = AMENDMENT_003) -> dict:
    """Refuse unless amendment 003 is final, indexed with this record's hash, and its text matches."""

    from .common import IntegrityError

    record_bytes = path.read_bytes()
    record_sha = hashlib.sha256(record_bytes).hexdigest()
    companion = path.with_suffix(".sha256")
    if not companion.exists() or companion.read_text(encoding="utf-8").split()[0] != record_sha:
        raise IntegrityError("amendment 003 record does not match its SHA-256 companion")
    index = json.loads((path.parent / "index.json").read_text(encoding="utf-8"))
    entries = [e for e in index["amendments"] if e.get("record") == path.name]
    if len(entries) != 1 or entries[0].get("record_sha256") != record_sha:
        raise IntegrityError("amendment 003 is not indexed with this record's SHA-256")
    record = json.loads(record_bytes)
    if record.get("status") != "final" or record.get("in_effect") is not True:
        raise IntegrityError("amendment 003 is not final and in effect")
    if t4.sha256(t4.ROOT / record["text"]) != record["text_sha256"]:
        raise IntegrityError("amendment 003 text does not match text_sha256")
    arm = record["arm"]
    if (arm["state_dynamics"], arm["noise"], arm["pool"], arm["aggregate"]) != ("Ua", "LW0", "corrected", "v0.3"):
        raise IntegrityError("amendment 003 arm does not match the implemented U0")
    return {"record": str(path.relative_to(t4.ROOT)), "record_sha256": record_sha, "text_sha256": record["text_sha256"]}


def reproduction_check(rows: list[dict]) -> dict:
    """C0 and F1 against the recorded T4 run, case by case, on every compared metric."""

    recorded = json.loads(RECORDED_T4.read_text(encoding="utf-8"))
    key = lambda r: (r["arm"], r["year"], r["set"], r["horizon_days"])  # noqa: E731
    old = {key(r): r for r in recorded["cases"] if r["arm"] in ("C0", "F1")}
    diffs, compared = [], 0
    for r in rows:
        if r["arm"] not in ("C0", "F1"):
            continue
        o = old.get(key(r))
        if o is None:
            diffs.append({"case": list(key(r)), "issue": "missing in recorded run"})
            continue
        for m in COMPARED_METRICS:
            a, b = r.get(m), o.get(m)
            compared += 1
            if not (a == b or (isinstance(a, float) and isinstance(b, float) and a != a and b != b)):
                diffs.append({"case": list(key(r)), "metric": m, "new": a, "recorded": b})
    return {"recorded_t4_sha256": t4.sha256(RECORDED_T4), "compared_values": compared,
            "identical": not diffs, "differences": diffs[:50]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--metric-slots", type=int, default=t4.DEFAULT_METRIC_SLOTS)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    from .common import verify_final_round

    provenance = verify_final_round()
    amendment_002 = t4.verify_amendment_002()
    amendment_003 = verify_amendment_003()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.time()
    configs = {"C0": ce.config_c0(), "F1": ce.config_f1(), "U0": config_u0()}

    check = t4.self_check(samples=t4.SELF_CHECK_DRAWS, cases=t4.SELF_CHECK_CASES)
    print(json.dumps({"self_check_ok": check["ok"]}), flush=True)
    if not check["ok"]:
        raise SystemExit(f"C0 self-check failed: {json.dumps(check)}")

    cases = t4.all_cases()
    truth = t4.load_truth_t4()
    timings: dict[str, float] = {}
    with tempfile.TemporaryDirectory(prefix="calibration-2030-t4-u0-") as tmp:
        inputs = t4.prepare_inputs(Path(tmp), need_v03=True, need_2010=True)
        base_jobs = [j for arm in ("C0", "F1")
                     for j in t4.make_jobs(arm, configs[arm], cases, inputs, truth, t4.DRAWS, t4.SEED)]
        origins = [t4.ELECTION_DAYS[y] - timedelta(days=h) for y, h, _ in cases]
        pool = t4._pool(args.workers, args.metric_slots)
        try:
            futures = [pool.submit(t4.run_case, j) for j in t4._by_cost(base_jobs)]
            t_k = time.time()
            t2_values = t4.t2_origin_values(configs["U0"], origins)
            timings["u0_k_precompute_seconds"] = round(time.time() - t_k, 1)
            u0_jobs = t4.make_jobs("U0", configs["U0"], cases, inputs, truth, t4.DRAWS, t4.SEED, t2_values)
            futures += [pool.submit(t4.run_case, j) for j in t4._by_cost(u0_jobs)]
            rows = [f.result() for f in futures]
        finally:
            pool.shutdown()
        inputs_record = {k: v for k, v in inputs.items() if k != "info"}

    gates = t4.evaluate_gates(rows, "U0", dynamics_changed=configs["U0"].dynamics_differ_from_d0,
                              provenance_ok=bool(provenance["tracked_clean"]))
    flagged = bool(gates.get("flag_2026", {}).get("flagged"))
    if gates["pass"] and not flagged:
        decision = "U0 passes and the 2026 flag does not trip: propose U0 as 1.3.0-rc1 (calibration amendment 003)"
    elif gates["pass"]:
        decision = "U0 passes but the 2026 flag trips: user decides (calibration amendment 003)"
    else:
        decision = "U0 fails: production stays 1.2.0-rc1; only F1 may be proposed (calibration amendment 003)"
    arms_run = list(dict.fromkeys(r["arm"] for r in rows))
    record = {
        "tier": "T4-exception-U0", "amendment_003": amendment_003, "amendment_002": amendment_002,
        "started_utc": started, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - t0, 1), "timings": timings, "workers": args.workers,
        "metric_slots": args.metric_slots, "provenance": provenance, "draws": t4.DRAWS, "seed": t4.SEED,
        "configs": {a: c.to_dict() for a, c in configs.items()}, "self_check": check, "inputs": inputs_record,
        "reproduction_of_recorded_t4": reproduction_check(rows),
        "pooled": t4.pooled_table(rows, arms_run), "gates": {"U0": gates}, "decision": decision, "cases": rows,
    }
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True, default=t4._json_default) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "reproduction_identical": record["reproduction_of_recorded_t4"]["identical"],
                      "seconds": record["seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
