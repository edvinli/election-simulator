"""Freeze the evaluator so Part-4 challenger work cannot silently alter it.

Writes ``evaluator_freeze.json``: hashes of the Amendment-2 preregistration, the
authoritative case manifest, every metric and path implementation file the
evaluator depends on, every truth input, the seed/N policy, the exact CONTROL
oracle and the CONTROL baseline summary.

Part 4 must re-run ``verify()`` before implementing a challenger. Any changed hash
is a hard stop: either the evaluator drifted, or the change must be preregistered.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.election_noise_v2.control_baseline.harness.rng import (
    CHALLENGER_RESERVED_TOKENS,
    CONTROL_TOKENS,
    DRAWS_PER_SEED,
    FROZEN_SEEDS,
)

from .manifest import AMENDMENT2, TRUTH_INPUTS

OUT = Path(__file__).resolve().parents[1]
PART3H = REPO_ROOT / "diagnostics/election_noise_v2/control_baseline/harness"
H2 = Path(__file__).resolve().parent

#: Every file whose content can change a reported metric value.
METRIC_IMPLEMENTATION_FILES = [
    # certified Part-3 metric implementations, reused unchanged
    PART3H / "metrics.py",
    PART3H / "rng.py",
    PART3H / "pipeline.py",
    # Amendment-2 path and oracle
    H2 / "isolated.py",
    H2 / "exact_oracle.py",
    H2 / "manifest.py",
    H2 / "run_control.py",
    # production estimators the evaluator calls
    REPO_ROOT / "scripts/vote_share_calibration/energy_score.py",
    REPO_ROOT / "scripts/pollofpolls/backtest_metrics.py",
    REPO_ROOT / "scripts/seat_hindcasts/metrics.py",
    REPO_ROOT / "scripts/election_layer_v2/forward_eval.py",
    REPO_ROOT / "scripts/election_layer_v2/transfer.py",
    REPO_ROOT / "scripts/election_layer_v2/residuals_pool.py",
    REPO_ROOT / "scripts/election_residuals/consensus.py",
    REPO_ROOT / "scripts/geography/projection.py",
    REPO_ROOT / "scripts/geography/raking.py",
    REPO_ROOT / "scripts/geography/integerization.py",
    REPO_ROOT / "scripts/mandates/allocator.py",
    REPO_ROOT / "scripts/mandates/law.py",
    REPO_ROOT / "scripts/mandates/tie_breaker.py",
    REPO_ROOT / "scripts/vote_share_calibration/models.py",
]

#: Baseline and manifest artifacts whose values Part 4 must compare against.
BASELINE_ARTIFACTS = [
    OUT / "evaluation_case_manifest.json",
    OUT / "control_scores_summary.json",
    OUT / "control_scores_by_case_seed.csv",
    OUT / "control_scores_by_election.csv",
    OUT / "coalition_brier_by_election.csv",
    OUT / "mask_level/coalition_brier_by_mask.csv",
    OUT / "exact_control_oracle.json",
    OUT / "exact_control_support.csv",
    OUT / "monte_carlo_vs_exact.json",
    OUT / "lambda_diagnostics.csv",
]

PART3_PRESERVED = REPO_ROOT / "diagnostics/election_noise_v2/control_baseline"
PART3_PRESERVED_FILES = [
    "control_scores_by_case_seed.csv",
    "control_scores_summary.json",
    "evaluation_case_manifest.json",
    "coalition_brier_by_election.csv",
    "monte_carlo_stability.csv",
    "lambda_diagnostics.csv",
    "harness_validation.json",
    "mask_level/coalition_brier_by_mask.csv",
]

PAIRED_RANDOMNESS = {
    "principle": (
        "For the same (target election, seed), every deterministic input is identical "
        "across CONTROL, Challenger A and Challenger B. Only the ElectionNoise draw law "
        "may differ."
    ),
    "why_pairing_is_exact_on_this_path": (
        "On the Tier 1 / Tier 3-ISO isolated path every non-ElectionNoise input is "
        "deterministic: the 14-day consensus is a fixed function of the archived polls, "
        "and geography, integerisation and the allocator are deterministic maps. There "
        "are therefore NO upstream random draws to pair - pairing is exact by "
        "construction, which is strictly stronger than the full-pipeline case where "
        "OpinionState and Dynamics draws had to be matched."
    ),
    "identical_and_immutable_for_every_model": [
        "the historical 14-day publication-safe polling consensus per election",
        "the chronological geography baseline, mode and processed inputs "
        "(oracle mode forbidden; total_national_votes left unset)",
        "law dispatch via mandate_law_for_election_year (PRE_2018 for 2014; POST_2018 for 2018/2022)",
        "the certified truth seat and vote vectors",
        "case selection (Tier 1 and Tier 3-ISO targets 2014/2018/2022)",
        "the sample count N per seed",
        "the seed list",
        "the mask set 1..254 and the 175 majority threshold",
        "the bounded simplex transfer (lambda rule, eps = 0.01 pp)",
    ],
    "seed_derivation": (
        'token = f"{base_seed}:{origin_date.isoformat()}:{horizon_days}:{label}"; '
        "subseed = int(sha256(token).hexdigest()[:8], 16) % 2_147_483_647"
    ),
    "origin_convention_on_the_isolated_path": {
        "origin_date": "the election date",
        "horizon_days": 14,
        "note": "frozen in Part 3; the isolated path has a single origin and no horizon dimension",
    },
    "control_streams": list(CONTROL_TOKENS),
    "challenger_reserved_streams": list(CHALLENGER_RESERVED_TOKENS),
    "common_random_numbers": {
        "available_but_not_used": (
            "Challenger A's atom index has the same marginal law as CONTROL's, so reusing "
            "CONTROL's residual_index stream would be a mathematically valid common-random-number "
            "pairing and would reduce comparison variance."
        ),
        "why_not_used": (
            "The preregistration reserves election_noise_v2_a_index as A's own stream. Reusing "
            "CONTROL's stream instead would be a preregistration change, which is prohibited. "
            "The forgone variance reduction is accepted and recorded."
        ),
        "not_forced_where_laws_differ": (
            "Challenger A's kernel noise and Challenger B's Gaussian draw have no CONTROL "
            "counterpart, so no artificial pairing is imposed on them."
        ),
    },
    "prohibited_for_challenger_implementations": [
        "perturbing the consensus, geography, law dispatch, certified truth, case selection, "
        "sample count or seed list",
        "consuming from CONTROL's residual_index or sign_draw streams",
        "introducing any seed token outside the reserved list",
        "using wall-clock, PID, environment values or unordered-set iteration",
    ],
}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT).decode().strip()


def _head_blob_sha256(rel: str) -> str | None:
    """SHA-256 of the committed (HEAD) content, so working-tree edits are visible."""
    try:
        blob = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None
    return hashlib.sha256(blob).hexdigest()


def import_closure_hashes() -> dict:
    """Hash every repository module the evaluator actually imports.

    A hand-curated file list silently misses transitive dependencies — during this
    task ``scripts/pollofpolls/normalize.py`` turned out to be in the evaluator's
    import chain while absent from the curated list. Walking ``sys.modules`` after
    importing the evaluator closes that gap and keeps itself up to date.

    Both the working-tree hash (what actually ran) and the committed HEAD hash are
    recorded, so an uncommitted edit is visible rather than absorbed.
    """
    import importlib

    for mod in (
        "diagnostics.election_noise_v2.control_baseline_amendment2.harness2.isolated",
        "diagnostics.election_noise_v2.control_baseline_amendment2.harness2.exact_oracle",
        "diagnostics.election_noise_v2.control_baseline_amendment2.harness2.manifest",
        "diagnostics.election_noise_v2.control_baseline_amendment2.harness2.run_control",
        "diagnostics.election_noise_v2.control_baseline.harness.metrics",
        "diagnostics.election_noise_v2.control_baseline.harness.pipeline",
        "diagnostics.election_noise_v2.control_baseline.harness.rng",
    ):
        importlib.import_module(mod)

    out: dict[str, dict] = {}
    for name, module in sorted(sys.modules.items()):
        f = getattr(module, "__file__", None)
        if not f:
            continue
        path = Path(f).resolve()
        try:
            rel = str(path.relative_to(REPO_ROOT))
        except ValueError:
            continue
        if not (rel.startswith("scripts/") or rel.startswith("diagnostics/")):
            continue
        wt = sha256_file(path)
        head = _head_blob_sha256(rel)
        out[rel] = {
            "module": name,
            "working_tree_sha256": wt,
            "head_sha256": head,
            "uncommitted_local_edit": head is not None and head != wt,
        }
    return out


def build() -> dict:
    return {
        "artifact": "ELECTIONNOISE V2 EVALUATOR FREEZE",
        "purpose": (
            "Part-4 challenger implementation must not silently alter the evaluator. "
            "Re-run verify() before implementing a challenger; any changed hash is a hard stop."
        ),
        "freeze_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": {
            "base_commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "freeze_commit": "the commit that introduces this artifact; reported with the commit",
        },
        "preregistration": AMENDMENT2,
        "monte_carlo_policy": {
            "seeds": list(FROZEN_SEEDS),
            "draws_per_seed": DRAWS_PER_SEED,
            "draws_per_case_per_model": DRAWS_PER_SEED * len(FROZEN_SEEDS),
            "seeds_may_not_be_reordered_extended_or_subset": True,
        },
        "case_set": {
            "gate_tiers": ["tier1", "tier3_iso"],
            "tier1_elections": [2014, 2018, 2022],
            "tier3_iso_elections": [2014, 2018, 2022],
            "N_T1": 3,
            "N_seat": 3,
            "geography_mode": "chronological",
            "forbidden_geography_modes": ["oracle"],
            "mandate_law": {"2014": "PRE_2018", "2018": "POST_2018", "2022": "POST_2018"},
            "first_divisor": {"2014": "7/5", "2018": "6/5", "2022": "6/5"},
        },
        "metric_implementation_hashes": {
            str(p.relative_to(REPO_ROOT)): sha256_file(p) for p in METRIC_IMPLEMENTATION_FILES
        },
        "evaluator_import_closure_hashes": import_closure_hashes(),
        "truth_input_hashes": {
            k: {"path": str(v.relative_to(REPO_ROOT)), "sha256": sha256_file(v)}
            for k, v in TRUTH_INPUTS.items()
        },
        "baseline_artifact_hashes": {
            str(p.relative_to(OUT)): sha256_file(p) for p in BASELINE_ARTIFACTS
        },
        "preserved_part3_full_pipeline_diagnostics": {
            "role": "retrospective diagnostics only; excluded from the adoption gate",
            "hashes": {
                f: sha256_file(PART3_PRESERVED / f) for f in PART3_PRESERVED_FILES
            },
        },
        "paired_randomness_contract": PAIRED_RANDOMNESS,
        "brier_interpretation_carried_forward": (
            "CONTROL's coalition probabilities are structurally coarse (K = 3/4/5 atoms, so p_m "
            "is confined to multiples of 1/K). A continuous challenger may clear the >=2% "
            "aggregate Brier improvement threshold relatively easily. The threshold is NOT "
            "changed and no gate is added. The decision must rest on the complete frozen gate: "
            "Tier-1 primary joint vote improvement, marginal non-inferiority, seat-vector "
            "non-inferiority, election-level robustness and coalition-Brier robustness across "
            "elections."
        ),
    }


def verify(frozen: dict | None = None) -> dict:
    """Recompute every hash and report drift. Part 4 must call this first."""
    if frozen is None:
        frozen = json.loads((OUT / "evaluator_freeze.json").read_text())
    drift: list[dict] = []
    for group, items in (
        ("metric_implementation", frozen["metric_implementation_hashes"]),
        ("baseline_artifact", frozen["baseline_artifact_hashes"]),
    ):
        for rel, expected in items.items():
            path = (REPO_ROOT / rel) if group == "metric_implementation" else (OUT / rel)
            actual = sha256_file(path) if path.exists() else None
            if actual != expected:
                drift.append({"group": group, "file": rel, "expected": expected, "actual": actual})
    for rel, v in frozen.get("evaluator_import_closure_hashes", {}).items():
        path = REPO_ROOT / rel
        actual = sha256_file(path) if path.exists() else None
        if actual != v["working_tree_sha256"]:
            drift.append({"group": "evaluator_import_closure", "file": rel,
                          "expected": v["working_tree_sha256"], "actual": actual,
                          "module": v["module"]})
    for k, v in frozen["truth_input_hashes"].items():
        path = REPO_ROOT / v["path"]
        actual = sha256_file(path) if path.exists() else None
        if actual != v["sha256"]:
            drift.append({"group": "truth_input", "file": v["path"], "expected": v["sha256"], "actual": actual})
    for f, expected in frozen["preserved_part3_full_pipeline_diagnostics"]["hashes"].items():
        path = PART3_PRESERVED / f
        actual = sha256_file(path) if path.exists() else None
        if actual != expected:
            drift.append({"group": "preserved_part3_diagnostic", "file": f,
                          "expected": expected, "actual": actual})
    return {"drift": drift, "evaluator_unchanged": not drift, "checks": (
        len(frozen["metric_implementation_hashes"]) + len(frozen["baseline_artifact_hashes"])
        + len(frozen["truth_input_hashes"])
        + len(frozen.get("evaluator_import_closure_hashes", {}))
        + len(frozen["preserved_part3_full_pipeline_diagnostics"]["hashes"]))}


def main() -> int:
    frozen = build()
    (OUT / "evaluator_freeze.json").write_text(json.dumps(frozen, indent=2) + "\n")
    res = verify(frozen)
    edits = {k: v for k, v in frozen["evaluator_import_closure_hashes"].items()
             if v["uncommitted_local_edit"]}
    print(f"evaluator freeze written; {res['checks']} hashes recorded")
    print(f"import closure: {len(frozen['evaluator_import_closure_hashes'])} repository modules")
    if edits:
        print("  NOTE - modules with an uncommitted working-tree edit at freeze time:")
        for k, v in edits.items():
            print(f"    {k}  (module {v['module']})")
    print("self-verification: evaluator_unchanged =", res["evaluator_unchanged"])
    if res["drift"]:
        for d in res["drift"]:
            print("  DRIFT:", d)
        return 1
    print("freeze timestamp:", frozen["freeze_timestamp_utc"])
    print("manifest sha256 :", frozen["baseline_artifact_hashes"]["evaluation_case_manifest.json"])
    print("oracle sha256   :", frozen["baseline_artifact_hashes"]["exact_control_oracle.json"])
    print("summary sha256  :", frozen["baseline_artifact_hashes"]["control_scores_summary.json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
