"""Post-adoption PRODUCTION freeze for the adopted ElectionNoise law.

Two freeze scopes exist and answer different questions. Conflating them is the
mistake this module is designed to prevent.

* The **historical research freezes**
  (``control_baseline_amendment2/evaluator_freeze.json`` and
  ``challengers/challenger_implementation_freeze.json``) certify the experiment that
  selected Challenger B. They are preserved byte-for-byte, are never regenerated
  here, and are verified against their **referenced historical commits**. After the
  Part-6B default flip they legitimately report drift against current HEAD, in
  exactly the four files that were changed on purpose.

* **This** freeze certifies the **current production configuration**: which law is
  the default, at which version, with which import closure, reproducing which
  certified same-input output.

Reconstructibility, carrying the Part-3D-R lesson forward: every recorded entry must
have a non-null committed HEAD hash equal to its working-tree hash, with no
uncommitted local edit. :func:`build` refuses to emit an artifact otherwise, so this
freeze can never depend on a state that exists in no commit.
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

from scripts.simulator.config import ADOPTED_CANDIDATE, MODEL_VERSION, RELEASE_TAG
from scripts.vote_share_calibration.election_noise_b import LEGACY_MODEL_ID, MODEL_ID

HERE = Path(__file__).resolve().parent
OUT = HERE / "production_freeze.json"
A2 = REPO_ROOT / "diagnostics/election_noise_v2/control_baseline_amendment2"
CH = REPO_ROOT / "diagnostics/election_noise_v2/challengers"
COMP = REPO_ROOT / "diagnostics/election_noise_v2/competition"
P6A = REPO_ROOT / "diagnostics/election_noise_b_promotion"

#: Authoritative upstream references, by full commit hash.
REFERENCES = {
    "adopt_b_decision_commit": "ff89621848c95ac9320804ffc4f148454f522284",
    "competition_bookkeeping_commit": "fb00f2ba6fc2613daefb3ca58b535ecf42ad1626",
    "evaluator_refreeze_commit": "a5b8c7a234acf60cac71ef1ab1439343fae88639",
    "challenger_freeze_commit": "1450e6f301a98d5d6e4af1357113435534b0e7a9",
    "part6a_production_implementation_commit": "8c8eaed20292961c8c262d1568b73a9ff1ebd679",
    "part6a_same_input_diagnostic_commit": "b8705e33ba469be29962164edf96a7f558d127ba",
}

#: Production files that define which ElectionNoise law runs, and how.
PRODUCTION_DEFAULT_FILES = [
    "scripts/vote_share_calibration/national_engine.py",
    "scripts/simulator/engine.py",
    "scripts/simulator/config.py",
    "scripts/simulator/reproducibility.py",
]

#: The adopted law and its production wrappers.
B_IMPLEMENTATION_FILES = [
    "scripts/vote_share_calibration/election_noise_b.py",
    "scripts/vote_share_calibration/production_national_engine.py",
    "scripts/simulator/production_runner.py",
]

#: The superseded law, retained unmodified for archived-forecast reproduction.
CONTROL_REPRODUCTION_FILES = [
    "scripts/vote_share_calibration/models.py",
    "scripts/election_layer_v2/transfer.py",
    "scripts/election_layer_v2/residuals_pool.py",
]

TEST_FILES = [
    "tests/test_production_default_is_b.py",
    "tests/test_production_challenger_b.py",
]

#: Files that must be byte-identical to their historical state, and are.
PRESERVED_HISTORICAL_ARTIFACTS = {
    "diagnostics/election_noise_v2/control_baseline_amendment2/evaluator_freeze.json":
        "3142f81c2494773448f2a48cbe57ccd964fa807891477155cc89e1c3b5f04bae",
    "diagnostics/election_noise_v2/challengers/challenger_implementation_freeze.json":
        "2454ac15309361443656fe1d00abd5cb655d5a8efc8ddaded9e8c7164d8c1c22",
    "diagnostics/election_noise_v2/competition/decision.json":
        "f0e534bb6f2a5c83dd32ae53a7dc916ed29ad85d1ea1910143f954b4689116bc",
    "diagnostics/election_noise_b_promotion/same_input_2026.json":
        "16206a6fd890cc08fb7571ba572f29017619458f13e27c3dc1c49fca3277636f",
    "diagnostics/election_noise_v2/challengers/challenger_b.py":
        None,   # resolved from the challenger freeze at build time
}

#: Drift against the historical freezes that the Part-6B flip deliberately created.
INTENTIONAL_HISTORICAL_DRIFT = set(PRODUCTION_DEFAULT_FILES)


class UncommittedProductionState(RuntimeError):
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


def _git(*a: str) -> str:
    return subprocess.check_output(["git", *a], cwd=REPO_ROOT).decode().strip()


def _record(rel: str) -> dict:
    p = REPO_ROOT / rel
    wt = sha256_file(p) if p.exists() else None
    head = _head_blob_sha256(rel)
    return {"working_tree_sha256": wt, "head_sha256": head,
            "uncommitted_local_edit": head is not None and head != wt}


def production_import_closure() -> dict:
    """Hash every repository module the production forecast path imports."""
    for mod in ("scripts.simulator.engine",
                "scripts.vote_share_calibration.national_engine",
                "scripts.vote_share_calibration.election_noise_b",
                "scripts.vote_share_calibration.production_national_engine",
                "scripts.simulator.production_runner",
                "scripts.publication_pipeline.pipeline"):
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


def historical_freeze_drift() -> dict:
    """Report, without repairing, how current HEAD differs from the historical state."""
    from diagnostics.election_noise_v2.challengers import freeze_challengers as cf
    from diagnostics.election_noise_v2.control_baseline_amendment2.harness2 import freeze as ev
    out = {}
    for name, res in (("evaluator", ev.verify()), ("challenger", cf.verify())):
        drifted = sorted({d["file"] for d in res["drift"]})
        out[name] = {
            "checks": res["checks"],
            "drifted_files": drifted,
            "all_drift_is_intentional": set(drifted) <= INTENTIONAL_HISTORICAL_DRIFT,
            "interpretation": (
                "Expected. These freezes certify the historical experiment and verify "
                "against their referenced historical commits, not against current HEAD."),
        }
    return out


def build(test_results: dict | None = None) -> dict:
    closure = production_import_closure()
    groups = {
        "production_default": {r: _record(r) for r in PRODUCTION_DEFAULT_FILES},
        "adopted_b_implementation": {r: _record(r) for r in B_IMPLEMENTATION_FILES},
        "control_reproduction": {r: _record(r) for r in CONTROL_REPRODUCTION_FILES},
        "tests": {r: _record(r) for r in TEST_FILES},
    }

    bad = []
    for scope, table in list(groups.items()) + [("import_closure", closure)]:
        for rel, rec in table.items():
            if rec["head_sha256"] is None or rec["uncommitted_local_edit"]:
                bad.append(f"{scope}: {rel}")
    if bad:
        raise UncommittedProductionState(
            "the production freeze must record committed content only; offending entries:\n"
            + "\n".join(f"  {b}" for b in bad)
            + "\nCommit the production-default implementation first, then regenerate.")

    cert_path = HERE / "default_path_certification.json"
    cert = json.loads(cert_path.read_text()) if cert_path.exists() else None

    preserved = dict(PRESERVED_HISTORICAL_ARTIFACTS)
    cfj = json.loads((CH / "challenger_implementation_freeze.json").read_text())
    preserved["diagnostics/election_noise_v2/challengers/challenger_b.py"] = (
        cfj["implementation_hashes"]["challenger_b"]
        ["diagnostics/election_noise_v2/challengers/challenger_b.py"]["working_tree_sha256"])
    preserved_check = {rel: {"expected": exp, "actual": sha256_file(REPO_ROOT / rel),
                             "preserved": sha256_file(REPO_ROOT / rel) == exp}
                       for rel, exp in preserved.items()}

    return {
        "artifact": "ELECTIONNOISE POST-ADOPTION PRODUCTION FREEZE",
        "purpose": ("Certify the CURRENT production configuration: which ElectionNoise law "
                    "is the default, at which version, with which import closure, "
                    "reproducing which certified same-input output."),
        "scope_note": (
            "This does NOT supersede or regenerate the historical research freezes. Those "
            "certify the experiment that selected Challenger B, are preserved byte-for-byte, "
            "and verify against their referenced historical commits."),
        "freeze_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": {
            "production_default_commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "worktree_clean_at_freeze": _git("status", "--porcelain") == "",
        },
        "references": REFERENCES,
        "adopted_model": {
            "election_noise_law": MODEL_ID,
            "candidate": ADOPTED_CANDIDATE,
            "superseded_law": LEGACY_MODEL_ID,
            "superseded_law_still_selectable": True,
            "selection_basis": ("preregistered historical evaluation over 2014/2018/2022; "
                                "the 2026 forecast was not an adoption input"),
            "tunable_hyperparameters": 0,
        },
        "model_version": {"model_version": MODEL_VERSION, "release_tag": RELEASE_TAG,
                          "release_status": "release candidate; not declared stable"},
        "historical_freeze_hashes": {
            "evaluator_freeze_sha256": sha256_file(A2 / "evaluator_freeze.json"),
            "challenger_freeze_sha256": sha256_file(CH / "challenger_implementation_freeze.json"),
            "decision_sha256": sha256_file(COMP / "decision.json"),
            "competition_gate_table_sha256": sha256_file(COMP / "gate_table.json"),
            "part6a_same_input_sha256": sha256_file(P6A / "same_input_2026.json"),
            "part6a_release_audit_sha256": sha256_file(P6A / "release_audit.json"),
        },
        "preserved_historical_artifacts": preserved_check,
        "historical_freeze_drift_against_head": historical_freeze_drift(),
        "same_input_certification": {
            "artifact": "default_path_certification.json",
            "sha256": sha256_file(cert_path) if cert_path.exists() else None,
            "configuration": cert["configuration"] if cert else None,
            "default_reproduces_part6a_b": cert["default_vs_certified_B"]["identical"] if cert else None,
            "control_reproduces_archived": cert["control_vs_certified_CONTROL"]["identical"] if cert else None,
            "output_hashes": cert["output_hashes"] if cert else None,
        },
        "production_file_hashes": groups,
        "production_import_closure_hashes": closure,
        "targeted_test_results": test_results or {},
        "reconstructibility": {
            "every_entry_committed": True,
            "working_tree_equals_head": True,
            "no_uncommitted_local_edit": True,
        },
    }


def verify(frozen: dict | None = None) -> dict:
    """Recompute every recorded hash and report drift in the CURRENT production state."""
    if frozen is None:
        frozen = json.loads(OUT.read_text())
    drift: list[dict] = []
    checks = 0

    tables = [(f"production:{g}", t) for g, t in frozen["production_file_hashes"].items()]
    tables.append(("production_import_closure", frozen["production_import_closure_hashes"]))
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

    for rel, rec in frozen["preserved_historical_artifacts"].items():
        checks += 1
        actual = sha256_file(REPO_ROOT / rel)
        if actual != rec["expected"]:
            drift.append({"group": "preserved_historical_artifact", "file": rel,
                          "expected": rec["expected"], "actual": actual})

    for name, rel in (("evaluator_freeze_sha256", A2 / "evaluator_freeze.json"),
                      ("challenger_freeze_sha256", CH / "challenger_implementation_freeze.json"),
                      ("decision_sha256", COMP / "decision.json"),
                      ("part6a_same_input_sha256", P6A / "same_input_2026.json")):
        checks += 1
        if sha256_file(rel) != frozen["historical_freeze_hashes"][name]:
            drift.append({"group": "historical_reference", "file": str(rel.name),
                          "expected": frozen["historical_freeze_hashes"][name],
                          "actual": sha256_file(rel)})

    return {"checks": checks, "drift": drift, "production_unchanged": not drift}


def main() -> int:
    frozen = build()
    OUT.write_text(json.dumps(frozen, indent=2) + "\n")
    res = verify(frozen)
    print(f"production freeze written; {res['checks']} hashes recorded")
    print(f"production import closure: {len(frozen['production_import_closure_hashes'])} modules")
    print("production_unchanged =", res["production_unchanged"])
    for d in res["drift"]:
        print("  DRIFT:", d)
    print("adopted law    :", frozen["adopted_model"]["election_noise_law"])
    print("model version  :", frozen["model_version"]["model_version"])
    print("default commit :", frozen["git"]["production_default_commit"])
    for k, v in frozen["historical_freeze_drift_against_head"].items():
        print(f"  historical {k} drift (intentional={v['all_drift_is_intentional']}):",
              v["drifted_files"])
    return 0 if res["production_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
