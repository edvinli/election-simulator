"""The frozen adoption gate, implemented literally from preregistration §F.

Tolerances (§F.1), applied to the five-seed mean, always relative to CONTROL on the
identical case set:

* "improves"                  -> strictly lower AND >= 2.0 % relative improvement
* "does not materially worsen" -> not more than +1.0 % relative above CONTROL
* "coverage does not materially worsen" -> at each of 50/80/90, abs(coverage -
  nominal) increases by no more than 3.0 pp versus CONTROL

Gates (§F.3): G1 Tier-1 joint vote improvement; G2 Tier-3-ISO coalition Brier
improvement; G3 Tier-1 marginal/interval non-inferiority; G4 Tier-3-ISO seat-vector
non-inferiority; G5 robustness (Tier-1 leave-one-target-out and individual
elections, plus the coalition per-election conditions); G6 determinism. G4b is
retired from the gate by Amendment 2.

No compensation: every non-inferiority check is a hard gate. A challenger passes
only if ALL required gates pass.
"""

from __future__ import annotations

import math

IMPROVE_PCT = 2.0        # G1 / G2 required improvement
NONINFERIOR_PCT = 1.0    # G3 / G4 tolerance
COVERAGE_PP = 3.0        # G3 coverage tolerance, percentage points
LOO_IMPROVE_PCT = 1.0    # G5-B secondary threshold
BRIER_LOO_DEGRADE_PCT = 1.0  # G5 coalition leave-one-out tolerance
NOMINAL = {"50": 0.50, "80": 0.80, "90": 0.90}

#: Comparison slack for boundary cases only. The frozen thresholds above are NOT
#: changed; this exists because a value that is exactly on a threshold in decimal
#: is not exactly on it in binary. abs(0.87 - 0.90) * 100 evaluates to
#: 3.0000000000000027, so an exactly-3.0 pp increase would otherwise FAIL a rule
#: that permits "no more than 3.0 percentage points". The slack is ~1e-9 relative,
#: far below any difference that could be scientifically meaningful, and it is
#: applied identically to every gate and to both challengers. Fixed before any
#: target-election score existed.
EPS = 1e-9


def rel_improvement(control: float, challenger: float) -> float:
    """Percent improvement of the challenger over CONTROL. Positive = better.

    Every metric in this gate is lower-is-better, so improvement is
    (control - challenger) / control * 100.
    """
    if control == 0:
        raise ValueError("CONTROL metric is zero; relative improvement undefined")
    return (control - challenger) / control * 100.0


def rel_degradation(control: float, challenger: float) -> float:
    """Percent the challenger is WORSE than CONTROL. Positive = worse."""
    return -rel_improvement(control, challenger)


def _row(gate, metric, ctrl, chal, threshold, passed, *, absolute=None, relative=None,
         artifact="", detail=""):
    return {"gate": gate, "metric": metric, "control": ctrl, "challenger": chal,
            "absolute_difference": absolute, "relative_difference_pct": relative,
            "required_threshold": threshold, "result": "PASS" if passed else "FAIL",
            "artifact": artifact, "detail": detail}


def g1_tier1_improvement(ctrl_es: float, chal_es: float) -> dict:
    imp = rel_improvement(ctrl_es, chal_es)
    return _row("G1", "tier1_es_9cat (five-seed mean)", ctrl_es, chal_es,
                f">= {IMPROVE_PCT}% relative improvement",
                chal_es < ctrl_es and imp >= IMPROVE_PCT - EPS,
                absolute=chal_es - ctrl_es, relative=imp,
                artifact="scores_summary.json:tier1.es_9cat")


def g2_coalition_improvement(ctrl_b: float, chal_b: float) -> dict:
    imp = rel_improvement(ctrl_b, chal_b)
    return _row("G2", "tier3_iso coalition Brier (headline)", ctrl_b, chal_b,
                f">= {IMPROVE_PCT}% relative improvement",
                chal_b < ctrl_b and imp >= IMPROVE_PCT - EPS,
                absolute=chal_b - ctrl_b, relative=imp,
                artifact="coalition_brier_by_model_election.csv")


def g3_noninferiority(ctrl: dict, chal: dict) -> list[dict]:
    """Tier-1 marginal and interval non-inferiority. Each sub-check is hard."""
    rows = []
    for key, label in (("es_9cat", "tier1 es_9cat"),
                       ("es_8party", "tier1 es_8party"),
                       ("crps_8party_mean", "tier1 mean 8-party CRPS")):
        deg = rel_degradation(ctrl[key], chal[key])
        rows.append(_row("G3", label, ctrl[key], chal[key],
                         f"<= +{NONINFERIOR_PCT}% relative degradation",
                         deg <= NONINFERIOR_PCT + EPS,
                         absolute=chal[key] - ctrl[key], relative=-deg,
                         artifact="scores_summary.json:tier1"))
    for lvl, nom in NOMINAL.items():
        k = f"coverage_{lvl}"
        cd = abs(ctrl[k] - nom) * 100.0
        hd = abs(chal[k] - nom) * 100.0
        rows.append(_row("G3", f"tier1 coverage {lvl}% |dev from nominal|",
                         cd, hd, f"increase <= {COVERAGE_PP} pp",
                         (hd - cd) <= COVERAGE_PP + EPS,
                         absolute=hd - cd, relative=None,
                         artifact="coverage_gate.csv",
                         detail=f"control coverage {ctrl[k]:.4f}, challenger {chal[k]:.4f}, "
                                f"nominal {nom:.2f}"))
    return rows


def g4_seat_noninferiority(ctrl_es: float, chal_es: float) -> dict:
    deg = rel_degradation(ctrl_es, chal_es)
    return _row("G4", "tier3_iso seat-vector ES", ctrl_es, chal_es,
                f"<= +{NONINFERIOR_PCT}% relative degradation", deg <= NONINFERIOR_PCT + EPS,
                absolute=chal_es - ctrl_es, relative=-deg,
                artifact="scores_summary.json:tier3_iso.seat_energy_score")


def g5_tier1_robustness(ctrl_by_el: dict, chal_by_el: dict) -> tuple[list[dict], dict]:
    """G5-B leave-one-target-out and G5-C individual-election consistency."""
    years = sorted(ctrl_by_el)
    n = len(years)
    rows, detail = [], {}

    # C - individual elections
    wins = [y for y in years if chal_by_el[y] < ctrl_by_el[y]]
    detail["individual"] = {str(y): {"control": ctrl_by_el[y], "challenger": chal_by_el[y],
                                     "challenger_lower": chal_by_el[y] < ctrl_by_el[y],
                                     "relative_improvement_pct":
                                         rel_improvement(ctrl_by_el[y], chal_by_el[y])}
                            for y in years}
    rows.append(_row("G5-C", f"individual Tier-1 elections favouring challenger",
                     None, len(wins), f">= {n - 1} of {n}", len(wins) >= n - 1,
                     artifact="scores_by_model_election.csv",
                     detail="won: " + (", ".join(str(y) for y in wins) or "none")))

    # B - leave-one-target-out aggregates (unweighted mean over remaining elections)
    loo = {}
    for drop in years:
        keep = [y for y in years if y != drop]
        c = sum(ctrl_by_el[y] for y in keep) / len(keep)
        h = sum(chal_by_el[y] for y in keep) / len(keep)
        loo[drop] = {"dropped": drop, "kept": keep, "control": c, "challenger": h,
                     "relative_improvement_pct": rel_improvement(c, h)}
    detail["leave_one_out"] = {str(k): v for k, v in loo.items()}
    all_pos = all(v["relative_improvement_pct"] > 0 for v in loo.values())
    n_ge1 = sum(1 for v in loo.values()
                if v["relative_improvement_pct"] >= LOO_IMPROVE_PCT - EPS)
    rows.append(_row("G5-B1", "leave-one-target-out aggregates favouring challenger",
                     None, sum(1 for v in loo.values() if v["relative_improvement_pct"] > 0),
                     f"all {n} strictly > 0", all_pos,
                     artifact="tier1_leave_one_out.csv",
                     detail="; ".join(f"drop {k}: {v['relative_improvement_pct']:+.3f}%"
                                      for k, v in loo.items())))
    rows.append(_row("G5-B2", f"leave-one-target-out aggregates improving >= {LOO_IMPROVE_PCT}%",
                     None, n_ge1, f">= {n - 1} of {n}", n_ge1 >= n - 1,
                     artifact="tier1_leave_one_out.csv"))
    return rows, detail


def g5_coalition_robustness(ctrl_by_el: dict, chal_by_el: dict) -> tuple[list[dict], dict]:
    """Coalition per-election conditions, N_seat >= 3 branch."""
    years = sorted(ctrl_by_el)
    n = len(years)
    need = math.ceil(n / 2)
    rows, detail = [], {}

    wins = [y for y in years if chal_by_el[y] < ctrl_by_el[y]]
    detail["individual"] = {str(y): {"control": ctrl_by_el[y], "challenger": chal_by_el[y],
                                     "challenger_lower": chal_by_el[y] < ctrl_by_el[y],
                                     "relative_improvement_pct":
                                         rel_improvement(ctrl_by_el[y], chal_by_el[y])}
                            for y in years}
    rows.append(_row("G5-Brier-elections", "elections with lower coalition Brier",
                     None, len(wins), f">= ceil({n}/2) = {need}", len(wins) >= need,
                     artifact="coalition_brier_by_model_election.csv",
                     detail="won: " + (", ".join(str(y) for y in wins) or "none")))

    loo = {}
    for drop in years:
        keep = [y for y in years if y != drop]
        c = sum(ctrl_by_el[y] for y in keep) / len(keep)
        h = sum(chal_by_el[y] for y in keep) / len(keep)
        loo[drop] = {"dropped": drop, "kept": keep, "control": c, "challenger": h,
                     "relative_improvement_pct": rel_improvement(c, h)}
    detail["leave_one_out"] = {str(k): v for k, v in loo.items()}
    worst = min(v["relative_improvement_pct"] for v in loo.values())
    ok = all(v["relative_improvement_pct"] >= -BRIER_LOO_DEGRADE_PCT - EPS
             for v in loo.values())
    rows.append(_row("G5-Brier-LOO", "leave-one-election-out coalition Brier delta",
                     None, worst,
                     f"no LOO delta worse than -{BRIER_LOO_DEGRADE_PCT}% (degradation)", ok,
                     relative=worst, artifact="coalition_leave_one_out.csv",
                     detail="; ".join(f"drop {k}: {v['relative_improvement_pct']:+.3f}%"
                                      for k, v in loo.items())))
    return rows, detail


def g6_determinism(bit_identical: bool, detail: str) -> dict:
    return _row("G6", "repeat (model, case, seed) is bit-identical", None, bit_identical,
                "identical output", bit_identical,
                artifact="score_audit.json:determinism", detail=detail)


def decide(passes: dict[str, bool], tier1_es: dict[str, float]) -> dict:
    """Frozen resolution rule (§F.4).

    Neither passes -> RETAIN_CONTROL. Exactly one -> adopt it. Both -> the lower
    Tier-1 primary joint vote ES; if within 0.5 % relative, prefer fewer free
    parameters (B over A); if still tied, prefer the more conservative A.
    """
    winners = [m for m, ok in passes.items() if ok]
    if not winners:
        return {"decision": "RETAIN_CONTROL", "rule": "neither challenger passed every gate",
                "passing_models": []}
    if len(winners) == 1:
        m = winners[0]
        return {"decision": f"ADOPT_{m}", "rule": "exactly one challenger passed every gate",
                "passing_models": winners}
    a, b = tier1_es["A"], tier1_es["B"]
    lower, other = ("A", "B") if a < b else ("B", "A")
    gap = abs(a - b) / min(a, b) * 100.0
    if gap < 0.5:
        return {"decision": "ADOPT_B", "rule": ("both passed and Tier-1 ES within 0.5% relative; "
                                                "frozen tie rule prefers fewer free parameters (B)"),
                "passing_models": winners, "tier1_relative_gap_pct": gap}
    return {"decision": f"ADOPT_{lower}",
            "rule": "both passed; frozen rule selects the lower Tier-1 primary joint vote ES",
            "passing_models": winners, "tier1_relative_gap_pct": gap,
            "rejected": other}
