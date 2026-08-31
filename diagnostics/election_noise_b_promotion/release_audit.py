"""Release-safety audit for the promoted ElectionNoise law.

Only an implementation, data-integrity or mathematical failure may block promotion.
No probability value in the 2026 forecast is treated as a blocker: the adoption was
decided on frozen historical proper scoring, and a prospective probability that
looks surprising is not evidence of a defect.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.election_noise_v2.challengers import freeze_challengers as ch_freeze
from diagnostics.election_noise_v2.control_baseline_amendment2.harness2 import freeze as ev_freeze
from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals
from scripts.simulator.production_runner import simulate_election_with_noise_model
from scripts.vote_share_calibration.election_noise_b import LEGACY_MODEL_ID, MODEL_ID

OUT = Path(__file__).resolve().parent
PUB = REPO_ROOT / "files/election-simulator/versions/20260828T201250Z-1da59168"
AS_OF, ELECTION, SAMPLES, SEED = "2026-08-24", "2026-09-13", 100_000, 12345


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    problems: list[str] = []
    checks: dict = {}

    print("run 1 (B)", flush=True)
    r1, n1, d1 = simulate_election_with_noise_model(
        MODEL_ID, as_of=AS_OF, election_date=ELECTION, samples=SAMPLES, seed=SEED)
    print("run 2 (B, deterministic repeat)", flush=True)
    r2, n2, d2 = simulate_election_with_noise_model(
        MODEL_ID, as_of=AS_OF, election_date=ELECTION, samples=SAMPLES, seed=SEED)

    # --- determinism ---
    same = {
        "vote_shares_matrix": bool(np.array_equal(r1.vote_shares_matrix, r2.vote_shares_matrix)),
        "seats_matrix": bool(np.array_equal(r1.seats_matrix, r2.seats_matrix)),
        "lambdas": bool(np.array_equal(n1.lambdas, n2.lambdas)),
        "election_noise_residuals": bool(np.array_equal(d1.residuals_pp, d2.residuals_pp)),
        "sigma_tilde": bool(np.array_equal(d1.fit.sigma_tilde, d2.fit.sigma_tilde)),
        "election_noise_seed": d1.election_noise_seed == d2.election_noise_seed,
    }
    checks["deterministic_repeat"] = same
    if not all(same.values()):
        problems.append(f"deterministic repeat differs: {[k for k,v in same.items() if not v]}")

    votes, seats, lam = r1.vote_shares_matrix, r1.seats_matrix, n1.lambdas

    # --- numeric validity ---
    checks["no_nan_or_inf"] = bool(np.all(np.isfinite(votes)) and np.all(np.isfinite(lam)))
    checks["votes_non_negative"] = bool(votes.min() >= 0.0)
    checks["votes_sum_to_100"] = float(np.abs(votes.sum(axis=1) - 100.0).max())
    checks["seat_totals_all_349"] = bool(np.all(seats.sum(axis=1) == 349))
    checks["seats_non_negative"] = bool(seats.min() >= 0)
    checks["lambda_in_unit_interval"] = bool(np.all((lam >= 0.0) & (lam <= 1.0)))
    if not checks["no_nan_or_inf"]:
        problems.append("non-finite values in the forecast")
    if not checks["votes_non_negative"]:
        problems.append("negative vote share")
    if checks["votes_sum_to_100"] > 1e-9:
        problems.append("vote compositions do not sum to 100")
    if not checks["seat_totals_all_349"]:
        problems.append("a seat draw does not total 349")
    if not checks["lambda_in_unit_interval"]:
        problems.append("lambda outside [0,1]")

    # --- upstream unchanged: exact pairing against the legacy law ---
    print("run 3 (legacy CONTROL, for pairing)", flush=True)
    r0, n0, _ = simulate_election_with_noise_model(
        LEGACY_MODEL_ID, as_of=AS_OF, election_date=ELECTION, samples=SAMPLES, seed=SEED)
    pairing = {
        "opinion_state_draws_identical": bool(np.array_equal(n0.opinion_state_draws,
                                                             n1.opinion_state_draws)),
        "dynamics_deltas_identical": bool(np.array_equal(n0.dynamics_deltas, n1.dynamics_deltas)),
        "base_comp_matrix_identical": bool(np.array_equal(n0.base_comp_matrix,
                                                          n1.base_comp_matrix)),
        "as_of_identical": n0.as_of == n1.as_of,
        "horizon_identical": n0.horizon_days == n1.horizon_days,
        "training_years_identical": tuple(n0.training_years) == tuple(n1.training_years),
    }
    checks["upstream_unchanged"] = pairing
    if not all(pairing.values()):
        problems.append(f"upstream not paired: {[k for k,v in pairing.items() if not v]}")

    # --- geography / allocator / engine unchanged (frozen-file hashes) ---
    frozen_files = [
        "scripts/geography/projection.py", "scripts/geography/raking.py",
        "scripts/geography/integerization.py", "scripts/mandates/allocator.py",
        "scripts/mandates/law.py", "scripts/simulator/engine.py",
        "scripts/vote_share_calibration/national_engine.py",
        "scripts/vote_share_calibration/models.py",
        "scripts/election_layer_v2/transfer.py",
        "scripts/election_layer_v2/residuals_pool.py",
    ]
    ev = json.loads((REPO_ROOT / "diagnostics/election_noise_v2/control_baseline_amendment2"
                     / "evaluator_freeze.json").read_text())
    rec = {**ev["metric_implementation_hashes"],
           **{k: v["working_tree_sha256"] for k, v in ev["evaluator_import_closure_hashes"].items()}}
    drifted = [f for f in frozen_files if f in rec and sha(REPO_ROOT / f) != rec[f]]
    checks["downstream_and_upstream_code_unchanged"] = {
        "files_checked": len(frozen_files), "drifted": drifted}
    if drifted:
        problems.append(f"frozen production code changed: {drifted}")

    # --- historical residual pool unchanged, and no future data ---
    pool = load_chronological_pp_residuals(target_election_year=2026)
    years = tuple(int(y) for y in pool.training_years)
    checks["residual_pool"] = {
        "training_years": list(years), "k": len(years),
        "all_strictly_before_2026": all(y < 2026 for y in years),
        "centered_sha256": hashlib.sha256(
            np.ascontiguousarray(pool.centered_residuals_matrix).tobytes()).hexdigest(),
    }
    if not checks["residual_pool"]["all_strictly_before_2026"]:
        problems.append("future election in the 2026 training pool")

    # --- 2026 data snapshot unchanged vs the certified publication ---
    pub_meta = json.loads((PUB / "metadata.json").read_text())
    run_hashes = r1.manifest.get("input_hashes", {})
    snapshot = {"published": pub_meta["input_hashes"], "current_run": run_hashes,
                "as_of_published": pub_meta["as_of"], "as_of_run": str(n1.as_of)}
    mismatched = [k for k, v in pub_meta["input_hashes"].items()
                  if k in run_hashes and run_hashes[k] != v and k != "model_config_hash"]
    snapshot["data_hashes_match"] = not mismatched
    snapshot["mismatched"] = mismatched
    checks["data_snapshot"] = snapshot
    if mismatched:
        problems.append(f"2026 input data changed since publication: {mismatched}")
    if pub_meta["as_of"] != str(n1.as_of):
        problems.append("as_of advanced; polling inputs must not be refreshed in this task")

    # --- CONTROL still reproducible against the certified publication ---
    pub_fc = json.loads((PUB / "forecast.json").read_text())
    diffs = []
    for p, v in pub_fc["parties"].items():
        i = list(pub_fc["parties"]).index(p)
        run_med = float(np.median(r0.vote_shares_matrix[:, i]))
        diffs.append(abs(run_med - v["vote_share_median"]))
    checks["control_reproduces_published_rc1"] = {
        "max_abs_median_difference_pp": max(diffs),
        "published_artifact_rounding_pp": 0.001,
        "within_publication_rounding": max(diffs) <= 0.001,
    }
    if max(diffs) > 0.001:
        problems.append(f"CONTROL no longer reproduces the certified RC1 (max {max(diffs):.2e} pp)")

    # --- freezes ---
    a, b = ev_freeze.verify(), ch_freeze.verify()
    checks["evaluator_freeze"] = {"checks": a["checks"], "drift": a["drift"]}
    checks["challenger_freeze"] = {"checks": b["checks"], "drift": b["drift"]}
    if a["drift"]:
        problems.append("evaluator freeze drifted")
    if b["drift"]:
        problems.append("challenger freeze drifted")

    payload = {
        "artifact": "ELECTIONNOISE B PRODUCTION RELEASE-SAFETY AUDIT",
        "scope": ("implementation, data integrity and mathematical validity only; no "
                  "probability value in the 2026 forecast is a blocker"),
        "audited_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                              cwd=REPO_ROOT).decode().strip(),
        "configuration": {"as_of": AS_OF, "election_date": ELECTION,
                          "samples": SAMPLES, "seed": SEED},
        "checks": checks, "problems": problems, "promotion_safe": not problems,
    }
    (OUT / "release_audit.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print("promotion_safe:", payload["promotion_safe"])
    print("problems:", problems)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
