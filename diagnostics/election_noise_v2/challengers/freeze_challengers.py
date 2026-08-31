"""Freeze the challenger implementations so they cannot drift once scoring begins.

Built on the Part-3D-R reconstructibility lesson: a freeze that records a
*working-tree* hash which exists in no commit can only be verified on the machine
that produced it. Every module recorded here is therefore required to satisfy

    head_sha256 is not None
    working_tree_sha256 == head_sha256
    uncommitted_local_edit is False

and :func:`build` refuses to emit an artifact otherwise. Generate this only from a
clean worktree, after the implementation commit exists.

The freeze records the preregistration and evaluator references, the challenger and
dependency hashes, the frozen bandwidth grid, the covariance conventions, the
exact-tie rule and the d² = 0 rule, and the targeted test results. It deliberately
records **no** target-election challenger score.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.election_noise_v2.control_baseline.harness.rng import (
    CHALLENGER_RESERVED_TOKENS,
    CONTROL_TOKENS,
    DRAWS_PER_SEED,
    FROZEN_SEEDS,
)
from diagnostics.election_noise_v2.control_baseline_amendment2.harness2.manifest import AMENDMENT2

from .challenger_a import FROZEN_H_GRID

HERE = Path(__file__).resolve().parent
OUT = HERE / "challenger_implementation_freeze.json"
A2 = REPO_ROOT / "diagnostics/election_noise_v2/control_baseline_amendment2"

#: The evaluator refreeze this work is built on (Part 3D-R).
EVALUATOR_REFREEZE_COMMIT = "a5b8c7a234acf60cac71ef1ab1439343fae88639"

#: Challenger implementation files, grouped by role.
IMPLEMENTATION_FILES: dict[str, list[str]] = {
    "challenger_a": ["diagnostics/election_noise_v2/challengers/challenger_a.py"],
    "nested_loocv": ["diagnostics/election_noise_v2/challengers/loeo.py",
                     "diagnostics/election_noise_v2/challengers/select_bandwidths.py"],
    "challenger_b": ["diagnostics/election_noise_v2/challengers/challenger_b.py"],
    "rng": ["diagnostics/election_noise_v2/challengers/rng.py"],
    "downstream_integration": ["diagnostics/election_noise_v2/challengers/draws.py"],
    "package": ["diagnostics/election_noise_v2/challengers/__init__.py"],
    "freeze_mechanism": ["diagnostics/election_noise_v2/challengers/freeze_challengers.py"],
}

#: Frozen production dependencies the challengers call but must never modify.
FROZEN_DEPENDENCIES = [
    "scripts/election_layer_v2/transfer.py",          # apply_batch_simplex_transfer
    "scripts/election_layer_v2/residuals_pool.py",
    "scripts/election_layer_v2/config.py",
    "scripts/election_residuals/consensus.py",
    "scripts/vote_share_calibration/energy_score.py",
    "scripts/geography/projection.py",
    "scripts/mandates/allocator.py",
    "scripts/mandates/law.py",
    "diagnostics/election_noise_v2/control_baseline/harness/rng.py",
    "diagnostics/election_noise_v2/control_baseline_amendment2/harness2/isolated.py",
]

TEST_FILES = [
    "tests/test_challenger_reference_math.py",
    "tests/test_challenger_a.py",
    "tests/test_challenger_b.py",
    "tests/test_challenger_integration.py",
]


class UncommittedImplementation(RuntimeError):
    """Raised when the freeze would record content that is not committed."""


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _head_blob_sha256(rel: str) -> str | None:
    try:
        blob = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None
    return hashlib.sha256(blob).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT).decode().strip()


def _record(rel: str) -> dict:
    p = REPO_ROOT / rel
    wt = sha256_file(p) if p.exists() else None
    head = _head_blob_sha256(rel)
    return {"working_tree_sha256": wt, "head_sha256": head,
            "uncommitted_local_edit": head is not None and head != wt}


def import_closure_hashes() -> dict:
    """Hash every repository module the challenger path actually imports."""
    for mod in (
        "diagnostics.election_noise_v2.challengers.challenger_a",
        "diagnostics.election_noise_v2.challengers.challenger_b",
        "diagnostics.election_noise_v2.challengers.loeo",
        "diagnostics.election_noise_v2.challengers.rng",
        "diagnostics.election_noise_v2.challengers.draws",
        "diagnostics.election_noise_v2.challengers.select_bandwidths",
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
        rec = _record(rel)
        rec["module"] = name
        out[rel] = rec
    return out


def build(test_results: dict | None = None) -> dict:
    closure = import_closure_hashes()
    groups = {g: {rel: _record(rel) for rel in rels} for g, rels in IMPLEMENTATION_FILES.items()}
    deps = {rel: _record(rel) for rel in FROZEN_DEPENDENCIES}
    tests = {rel: _record(rel) for rel in TEST_FILES}

    bad = []
    for scope, table in (("implementation", {r: v for g in groups.values() for r, v in g.items()}),
                         ("dependency", deps), ("test", tests), ("import_closure", closure)):
        for rel, rec in table.items():
            if rec["head_sha256"] is None or rec["uncommitted_local_edit"]:
                bad.append({"scope": scope, "file": rel,
                            "head_sha256": rec["head_sha256"],
                            "uncommitted_local_edit": rec["uncommitted_local_edit"]})
    if bad:
        raise UncommittedImplementation(
            "the challenger freeze must record committed content only; offending entries:\n"
            + "\n".join(f"  {b['scope']}: {b['file']}" for b in bad)
            + "\nCommit the implementation first (Part 4 step 11), then regenerate."
        )

    bandwidths = None
    bw_path = HERE / "bandwidth_selection.json"
    if bw_path.exists():
        bw = json.loads(bw_path.read_text())
        bandwidths = {
            "artifact_sha256": sha256_file(bw_path),
            "h_star_by_target": {t: v["h_star"] for t, v in bw["by_target"].items()},
            "exact_tie_encountered": {t: v["exact_tie_encountered"] for t, v in bw["by_target"].items()},
            "note": ("LOEO-FIT inside the training pools only; not a score. Pinned here so the "
                     "bandwidth cannot drift between freezing and scoring."),
        }

    return {
        "artifact": "ELECTIONNOISE V2 CHALLENGER IMPLEMENTATION FREEZE",
        "purpose": ("Pin the Challenger A and Challenger B implementations and their "
                    "dependencies before any target-election score is computed. Any changed "
                    "hash after this point is a hard stop."),
        "contains_no_target_election_scores": True,
        "freeze_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": {
            "implementation_commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "worktree_clean_at_freeze": _git("status", "--porcelain") == "",
        },
        "preregistration": AMENDMENT2,
        "evaluator": {
            "refreeze_commit": EVALUATOR_REFREEZE_COMMIT,
            "evaluator_freeze_sha256": sha256_file(A2 / "evaluator_freeze.json"),
            "evaluation_case_manifest_sha256": sha256_file(A2 / "evaluation_case_manifest.json"),
            "control_scores_summary_sha256": sha256_file(A2 / "control_scores_summary.json"),
            "exact_control_oracle_sha256": sha256_file(A2 / "exact_control_oracle.json"),
            "clean_reproduction_comparison_sha256": sha256_file(A2 / "clean_reproduction_comparison.json"),
            "evaluator_not_modified_by_part4": True,
        },
        "monte_carlo_policy": {
            "seeds": list(FROZEN_SEEDS),
            "draws_per_seed": DRAWS_PER_SEED,
            "unchanged_from_the_evaluator_freeze": True,
        },
        "challenger_a": {
            "law": "R = (c_k + h*eps)/sqrt(1+h^2);  k ~ U{1..K};  eps = (1/sqrt K) sum_j z_j c_j",
            "epsilon_law": "eps ~ N(0, S_P)",
            "covariance_convention": "S_P = C^T C / K  (divisor K, maximum likelihood, NO Bessel)",
            "theoretical_mean": "0",
            "theoretical_covariance": "S_P, exactly, for every h on the grid",
            "variance_correction": "sqrt(1+h^2) denominator, binding",
            "h_grid": list(FROZEN_H_GRID),
            "h_zero_excluded": True,
            "free_parameters": 1,
            "prohibited": ["h=0", "additional h values", "interpolation", "continuous optimisation",
                           "party-specific smoothing", "recency weighting", "residual-year weighting",
                           "Student-t noise", "covariance regularization", "2026-dependent tuning"],
        },
        "nested_loocv": {
            "rule": ("score(h) = (1/K_outer) sum_{j in P} ES(F^A(h, P\\{j}), r_j - rbar_{P\\{j}}); "
                     "h* = argmin_h score(h)"),
            "exact_tie_rule": "on an exact tie choose the SMALLEST h (most conservative)",
            "seed_aggregation": ("five-seed mean per D0, so h* is a deterministic property of the "
                                 "training pool and does not vary with the evaluation seed"),
            "k_outer_minimum": 3,
            "k_inner_minimum": 2,
            "k_inner_one": "prohibited; fails loudly",
            "centering": "the production centering algorithm, re-applied on each inner pool",
            "held_out_target": "r_j - rbar_{P\\{j}}, expressed in the inner centering",
            "energy_score": "scripts/vote_share_calibration/energy_score.py::compute_energy_score, unchanged",
            "leakage": ("no outer target residual, no future residual and no 2026 information "
                        "enters tuning"),
            "bandwidth_selection": bandwidths,
        },
        "challenger_b": {
            "law": "R ~ N(0, Sigma_tilde)",
            "s_p": "S_P = C^T C / K",
            "projector": "P9 = I - 11^T/9 (exact zero-sum projector)",
            "tau_sq": "tr(S_P)/8",
            "target": "T = tau^2 * P9",
            "norm": "||A||^2 = tr(A A^T)/8; the 1/8 cancels in delta",
            "d_sq": "||S_P - T||^2",
            "bbar_sq": "(1/K^2) sum_j ||c_j c_j^T - S_P||^2",
            "b_sq": "min(bbar^2, d^2)",
            "delta": "b^2/d^2",
            "d_sq_zero_rule": "delta := 1 if d^2 == 0 (preregistered limit)",
            "sigma_tilde": "(K/(K-1)) * [delta*T + (1-delta)*S_P]",
            "bessel_correction": "K/(K-1), applied exactly once, at the final stage only",
            "free_parameters": 0,
            "distribution": "Gaussian only",
            "numerical_policy": ("symmetric eigendecomposition; a materially negative eigenvalue "
                                 "raises NonPSDCovariance rather than being clipped; only "
                                 "round-off-level structural zeros are set to exactly 0"),
            "prohibited": ["Student-t", "added ridge", "empirical tail multiplier",
                           "recency weighting", "eigenvalue clipping", "any tunable parameter"],
        },
        "downstream": {
            "transfer": "scripts/election_layer_v2/transfer.py::apply_batch_simplex_transfer, unchanged",
            "apply_batch_simplex_transfer_sha256": sha256_file(
                REPO_ROOT / "scripts/election_layer_v2/transfer.py"),
            "unchanged": ["epsilon/floor behaviour (eps = 0.01 pp)", "lambda computation",
                          "donor attenuation", "simplex constraints"],
            "lambda_role": "descriptive only; not a tuning parameter and has no adoption gate",
            "geography": "chronological only; oracle mode forbidden",
            "mandate_law": {"2014": "PRE_2018", "2018": "POST_2018", "2022": "POST_2018"},
        },
        "rng_contract": {
            "reserved_tokens": list(CHALLENGER_RESERVED_TOKENS),
            "control_tokens_forbidden": list(CONTROL_TOKENS),
            "derivation": ('token = f"{base_seed}:{origin_date.isoformat()}:{horizon_days}:{label}"; '
                           "subseed = int(sha256(token).hexdigest()[:8], 16) % 2_147_483_647"),
            "substreams": "numpy SeedSequence spawn_key on the reserved token; no new tokens",
            "a_streams": ["election_noise_v2_a_index (atom index)",
                          "election_noise_v2_a_kernel (Gaussian smoothing)",
                          "election_noise_v2_a_loeo (LOEO-FIT)"],
            "b_streams": ["election_noise_v2_b_normal (Gaussian residual draw)"],
            "common_random_numbers": ("none imposed; A's index stream is its own reserved token as "
                                      "the preregistration requires, and no artificial coupling is "
                                      "introduced between A, B or CONTROL"),
            "determinism": "identical (model, case, seed, N) is bit-identical; asserted by tests",
        },
        "implementation_hashes": groups,
        "frozen_dependency_hashes": deps,
        "test_file_hashes": tests,
        "import_closure_hashes": closure,
        "targeted_test_results": test_results or {},
        "reconstructibility": {
            "every_entry_committed": True,
            "working_tree_equals_head": True,
            "no_uncommitted_local_edit": True,
            "lesson": "Part 3D-R: a freeze recording working-tree-only content is unverifiable",
        },
    }


def verify(frozen: dict | None = None) -> dict:
    """Recompute every recorded hash and report drift."""
    if frozen is None:
        frozen = json.loads(OUT.read_text())
    drift: list[dict] = []
    checks = 0

    tables: list[tuple[str, dict]] = [
        ("frozen_dependency", frozen["frozen_dependency_hashes"]),
        ("test_file", frozen["test_file_hashes"]),
        ("import_closure", frozen["import_closure_hashes"]),
    ]
    for group, files in frozen["implementation_hashes"].items():
        tables.append((f"implementation:{group}", files))

    for group, table in tables:
        for rel, rec in table.items():
            checks += 1
            p = REPO_ROOT / rel
            actual = sha256_file(p) if p.exists() else None
            if actual != rec["working_tree_sha256"]:
                drift.append({"group": group, "file": rel,
                              "expected": rec["working_tree_sha256"], "actual": actual})
            elif rec["head_sha256"] is None or rec["uncommitted_local_edit"]:
                drift.append({"group": group, "file": rel,
                              "issue": "recorded entry is not committed content"})

    ev = frozen["evaluator"]
    for name, rel in (("evaluator_freeze_sha256", "evaluator_freeze.json"),
                      ("evaluation_case_manifest_sha256", "evaluation_case_manifest.json"),
                      ("control_scores_summary_sha256", "control_scores_summary.json"),
                      ("exact_control_oracle_sha256", "exact_control_oracle.json")):
        checks += 1
        actual = sha256_file(A2 / rel)
        if actual != ev[name]:
            drift.append({"group": "evaluator", "file": rel,
                          "expected": ev[name], "actual": actual})

    checks += 1
    tr = sha256_file(REPO_ROOT / "scripts/election_layer_v2/transfer.py")
    if tr != frozen["downstream"]["apply_batch_simplex_transfer_sha256"]:
        drift.append({"group": "downstream", "file": "scripts/election_layer_v2/transfer.py",
                      "expected": frozen["downstream"]["apply_batch_simplex_transfer_sha256"],
                      "actual": tr})

    return {"checks": checks, "drift": drift, "challengers_unchanged": not drift}


def main() -> int:
    frozen = build()
    OUT.write_text(json.dumps(frozen, indent=2) + "\n")
    res = verify(frozen)
    print(f"challenger implementation freeze written; {res['checks']} hashes recorded")
    print(f"import closure: {len(frozen['import_closure_hashes'])} repository modules")
    print("challengers_unchanged =", res["challengers_unchanged"])
    for d in res["drift"]:
        print("  DRIFT:", d)
    print("implementation commit:", frozen["git"]["implementation_commit"])
    print("freeze timestamp     :", frozen["freeze_timestamp_utc"])
    return 0 if res["challengers_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
