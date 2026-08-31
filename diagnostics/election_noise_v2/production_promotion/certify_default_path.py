"""Certify that the ORDINARY production default reproduces the Part-6A B forecast.

The default entry point is called with no ``noise_model`` argument, so what is
certified is the default itself and not an override. The result is compared field by
field, at exact equality, against the Part-6A certified same-input diagnostic.

The legacy CONTROL law is then invoked explicitly and compared against the same
diagnostic's CONTROL block, so the production flip is shown not to have destroyed
archived-forecast reproduction.

No polling data is refreshed and no new as_of is introduced.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.election_noise_b_promotion.run_same_input_2026 import summarise
from scripts.simulator.engine import simulate_election
from scripts.simulator.production_runner import simulate_election_with_noise_model
from scripts.vote_share_calibration.election_noise_b import LEGACY_MODEL_ID, MODEL_ID

OUT = Path(__file__).resolve().parent
PART6A = REPO_ROOT / "diagnostics/election_noise_b_promotion/same_input_2026.json"
AS_OF, ELECTION, SAMPLES, SEED = "2026-08-24", "2026-09-13", 100_000, 12345


def diff(a, b, path="$", out=None, tol=0.0):
    if out is None:
        out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append({"path": f"{path}.{k}", "issue": "key missing on one side"})
            else:
                diff(a[k], b[k], f"{path}.{k}", out, tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append({"path": path, "len_a": len(a), "len_b": len(b)})
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out, tol)
    elif isinstance(a, float) and isinstance(b, float):
        if a != b and abs(a - b) > tol:
            out.append({"path": path, "certified": a, "default_path": b})
    elif a != b:
        out.append({"path": path, "certified": a, "default_path": b})
    return out


def main() -> int:
    certified = json.loads(PART6A.read_text())

    print("running the ORDINARY default entry point (no noise_model argument)", flush=True)
    default_res = simulate_election(as_of=AS_OF, election_date=ELECTION,
                                    samples=SAMPLES, seed=SEED)
    # national result / detail for lambda and covariance diagnostics
    b_res, b_nat, b_detail = simulate_election_with_noise_model(
        MODEL_ID, as_of=AS_OF, election_date=ELECTION, samples=SAMPLES, seed=SEED)

    default_matches_explicit = {
        "vote_shares_matrix": bool(np.array_equal(default_res.vote_shares_matrix,
                                                  b_res.vote_shares_matrix)),
        "seats_matrix": bool(np.array_equal(default_res.seats_matrix, b_res.seats_matrix)),
    }

    got_b = summarise(default_res, b_nat, b_detail, MODEL_ID)
    b_diffs = diff(certified["models"]["B"], got_b)

    print("running the explicit legacy CONTROL law", flush=True)
    c_res, c_nat, _ = simulate_election_with_noise_model(
        LEGACY_MODEL_ID, as_of=AS_OF, election_date=ELECTION, samples=SAMPLES, seed=SEED)
    got_c = summarise(c_res, c_nat, None, LEGACY_MODEL_ID)
    c_diffs = diff(certified["models"]["CONTROL"], got_c)

    # metadata
    meta = {
        "default_manifest_noise_model": default_res.manifest["model_config"]["noise_model"],
        "default_manifest_model_version": default_res.manifest["model_version"],
        "control_manifest_noise_model": c_res.manifest["model_config"]["noise_model"],
        "default_is_adopted_b": default_res.manifest["model_config"]["noise_model"] == MODEL_ID,
    }

    def h(m):
        return hashlib.sha256(np.ascontiguousarray(m).tobytes()).hexdigest()

    payload = {
        "artifact": "PRODUCTION DEFAULT-PATH CERTIFICATION",
        "purpose": ("prove the ordinary production default reproduces the Part-6A certified "
                    "Challenger-B forecast exactly, and that explicit CONTROL still "
                    "reproduces the archived CONTROL forecast"),
        "certified_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                              cwd=REPO_ROOT).decode().strip(),
        "configuration": {"as_of": AS_OF, "election_date": ELECTION,
                          "samples": SAMPLES, "seed": SEED,
                          "noise_model_argument_passed": False,
                          "polling_inputs_refreshed": False},
        "part6a_reference": {
            "artifact": "diagnostics/election_noise_b_promotion/same_input_2026.json",
            "sha256": hashlib.sha256(PART6A.read_bytes()).hexdigest(),
            "commit": "b8705e33ba469be29962164edf96a7f558d127ba",
        },
        "default_path_equals_explicit_b": default_matches_explicit,
        "default_vs_certified_B": {"differences": b_diffs, "identical": not b_diffs},
        "control_vs_certified_CONTROL": {"differences": c_diffs, "identical": not c_diffs},
        "metadata": meta,
        "output_hashes": {
            "default_vote_shares_sha256": h(default_res.vote_shares_matrix),
            "default_seats_sha256": h(default_res.seats_matrix),
            "control_vote_shares_sha256": h(c_res.vote_shares_matrix),
            "control_seats_sha256": h(c_res.seats_matrix),
        },
        "certified": (not b_diffs and not c_diffs and all(default_matches_explicit.values())
                      and meta["default_is_adopted_b"]),
    }
    (OUT / "default_path_certification.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n")
    print("default == explicit B:", default_matches_explicit)
    print("B differences vs certified      :", len(b_diffs))
    print("CONTROL differences vs certified:", len(c_diffs))
    print("metadata:", meta)
    print("CERTIFIED:", payload["certified"])
    return 0 if payload["certified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
