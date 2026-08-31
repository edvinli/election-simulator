"""Write decision.json and RESULTS.md from the frozen gate outcome. No new science."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.election_noise_v2.competition import gates as G

OUT = Path(__file__).resolve().parent


def main() -> int:
    man = json.loads((OUT / "competition_manifest.json").read_text())
    gt = json.loads((OUT / "gate_table.json").read_text())
    summ = json.loads((OUT / "scores_summary.json").read_text())
    audit = json.loads((OUT / "score_audit.json").read_text())

    def hl(m, t, k):
        return summ["models"][m][t]["headline"][k]["mean_over_elections"]

    passes = gt["all_gates_pass"]
    t1es = {m: hl(m, "tier1", "es_9cat") for m in ("A", "B")}
    decision = G.decide(passes, t1es)

    if not audit["run_valid"]:
        decision = {"decision": "INVALID_RUN",
                    "rule": "score audit reported integrity problems",
                    "problems": audit["problems"]}

    rec = {
        "artifact": "ELECTIONNOISE V2 ADOPTION DECISION",
        "decided_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit_of_results": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip(),
        "authoritative_commits": man["authoritative_commits"],
        "freeze_hashes": man["freeze_hashes"],
        "preregistration": man["preregistration"],
        "model_versions": man["models"],
        "seeds": man["seeds"],
        "draws_per_seed": man["draws_per_seed"],
        "challenger_a_bandwidths": man["challenger_a_bandwidths"],
        "challenger_b_hyperparameters": 0,
        "case_set": {"tier1": man["targets"], "tier3_iso": man["targets"],
                     "N_T1": man["N_T1"], "N_seat": man["N_seat"]},
        "headline_metrics": {
            m: {"tier1_es_9cat": hl(m, "tier1", "es_9cat"),
                "tier1_es_8party": hl(m, "tier1", "es_8party"),
                "tier1_crps_8party_mean": hl(m, "tier1", "crps_8party_mean"),
                "tier3_iso_seat_energy_score": hl(m, "tier3_iso", "seat_energy_score"),
                "tier3_iso_coalition_brier": hl(m, "tier3_iso", "coalition_brier_mean_over_masks")}
            for m in ("CONTROL", "A", "B")},
        "gate_outcomes": {
            m: {"all_gates_pass": passes[m],
                "gates": [{"gate": r["gate"], "metric": r["metric"],
                           "result": r["result"],
                           "relative_difference_pct": r["relative_difference_pct"],
                           "required_threshold": r["required_threshold"]}
                          for r in gt["rows"] if r["model"] == m]}
            for m in ("A", "B")},
        "selected_model": decision["decision"],
        "decision_rule_applied": decision.get("rule"),
        "decision_detail": decision,
        "discretionary_override": False,
        "override_statement": ("The decision follows the frozen rule mechanically. No "
                               "qualitative override was applied, no threshold was changed, "
                               "and no model parameter was altered after scores were observed."),
        "forecast_2026_statement": "2026 forecast was not an adoption input.",
        "forecast_2026_run": False,
        "excluded_from_adoption": man["excluded_from_adoption"],
        "score_audit": {"run_valid": audit["run_valid"], "problems": audit["problems"]},
        "brier_interpretation_caveat": (
            "CONTROL's coalition probabilities are structurally coarse - its law on this path "
            "has only K = 3/4/5 atoms, so p_m is confined to multiples of 1/K. A continuous "
            "challenger may therefore clear the 2% Brier threshold relatively easily. The "
            "threshold was NOT changed, and the decision rests on the complete frozen gate, "
            "with the Tier-1 joint vote energy score as the primary criterion."),
    }
    (OUT / "decision.json").write_text(json.dumps(rec, indent=2, default=str) + "\n")
    print("decision:", rec["selected_model"])
    print("rule    :", rec["decision_rule_applied"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
