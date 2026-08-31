"""Same-input 2026 comparison: legacy ElectionNoise (CONTROL) vs the adopted law (B).

Post-adoption diagnostic. ADOPT_B was frozen in
``diagnostics/election_noise_v2/competition/decision.json`` before any of this was
computed, and nothing here may reopen model selection. Polling inputs are not
refreshed and ``as_of`` is not advanced: both runs use the certified production
configuration as_of 2026-08-24, election 2026-09-13, N = 100 000, seed 12345.

Both runs go through the unmodified production engine, and both consume the same
OpinionState and Dynamics draws, so every difference reported is attributable to
the ElectionNoise layer alone.
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

from scripts.simulator.config import MODEL_PARTIES_9, PARLIAMENTARY_PARTIES_8
from scripts.simulator.production_runner import simulate_election_with_noise_model
from scripts.vote_share_calibration.election_noise_b import (
    ADOPTED_ELECTION_NOISE_CANDIDATE,
    ADOPTED_MODEL_VERSION,
    LEGACY_MODEL_ID,
    MODEL_ID,
)

OUT = Path(__file__).resolve().parent
AS_OF, ELECTION, SAMPLES, SEED = "2026-08-24", "2026-09-13", 100_000, 12345

GROUPS = {
    "tido": ["M", "SD", "KD", "L"],
    "red_green_center": ["S", "V", "MP", "C"],
}
FOCUS_COALITIONS = {
    "C+S+MP": ["C", "S", "MP"],
    "S+V+MP": ["S", "V", "MP"],
}
QUANTILES = [5, 10, 25, 50, 75, 90, 95]


def _q(a: np.ndarray) -> dict:
    return {f"p{q:02d}": float(np.percentile(a, q)) for q in QUANTILES}


def _coalition_seats(seats: np.ndarray, parties: list[str]) -> np.ndarray:
    idx = [PARLIAMENTARY_PARTIES_8.index(p) for p in parties]
    return seats[:, idx].sum(axis=1)


def _modes(counts: dict[int, int], n: int, floor: float = 0.02) -> list[dict]:
    """Material modes: local maxima of the integer histogram holding >= 2% mass."""
    ks = sorted(counts)
    out = []
    for k in ks:
        c = counts[k]
        if c / n < floor:
            continue
        left = counts.get(k - 1, 0)
        right = counts.get(k + 1, 0)
        if c >= left and c >= right:
            out.append({"seats": int(k), "probability": c / n})
    return sorted(out, key=lambda d: -d["probability"])


def summarise(res, nat, detail, model_id: str) -> dict:
    votes = res.vote_shares_matrix          # (N, 9) already in percent
    seats = res.seats_matrix
    lambdas = nat.lambdas
    n = seats.shape[0]

    parties = {}
    for i, p in enumerate(MODEL_PARTIES_9):
        v = votes[:, i]
        rec = {"vote_mean": float(v.mean()), "vote_median": float(np.median(v)),
               "vote_p05": float(np.percentile(v, 5)), "vote_p95": float(np.percentile(v, 95))}
        if p != "REST":
            j = PARLIAMENTARY_PARTIES_8.index(p)
            s = seats[:, j]
            rec.update({
                "prob_above_4pct": float(np.mean(v >= 4.0)),
                "seats_mean": float(s.mean()), "seats_median": float(np.median(s)),
                "seats_p05": float(np.percentile(s, 5)), "seats_p95": float(np.percentile(s, 95)),
                "prob_any_seats": float(np.mean(s > 0)),
            })
        parties[p] = rec

    groups = {}
    for name, ps in GROUPS.items():
        cs = _coalition_seats(seats, ps)
        groups[name] = {"parties": ps, "mean_seats": float(cs.mean()),
                        "median_seats": float(np.median(cs)),
                        **_q(cs), "prob_majority": float(np.mean(cs >= 175))}

    focus = {}
    for name, ps in FOCUS_COALITIONS.items():
        cs = _coalition_seats(seats, ps)
        vals, cnts = np.unique(cs, return_counts=True)
        hist = {int(a): int(b) for a, b in zip(vals, cnts)}
        focus[name] = {
            "parties": ps, "mean_seats": float(cs.mean()),
            "median_seats": float(np.median(cs)), **_q(cs),
            "sd_seats": float(cs.std(ddof=1)), "variance_seats": float(cs.var(ddof=1)),
            "prob_majority": float(np.mean(cs >= 175)),
            "distinct_seat_values": int(len(hist)),
            "material_modes": _modes(hist, n),
            "mass_within_5_of_175": float(np.mean(np.abs(cs - 175) <= 5)),
            "mass_in_170_180": float(np.mean((cs >= 170) & (cs <= 180))),
            "histogram_top20": [{"seats": int(k), "probability": hist[k] / n}
                                for k in sorted(hist, key=lambda x: -hist[x])[:20]],
        }

    # All 254 non-trivial coalition masks, the same convention the evaluation used.
    masks = {}
    for m in range(1, 255):
        cols = [i for i in range(8) if m >> i & 1]
        cs = seats[:, cols].sum(axis=1)
        masks[str(m)] = {"parties": "+".join(PARLIAMENTARY_PARTIES_8[i] for i in cols),
                         "mean_seats": float(cs.mean()),
                         "prob_majority": float(np.mean(cs >= 175))}

    out = {
        "model_id": model_id, "samples": n,
        "parties": parties, "groups": groups, "focus_coalitions": focus,
        "coalition_masks": masks,
        "lambda": {"mean": float(lambdas.mean()), "min": float(lambdas.min()),
                   "max": float(lambdas.max()),
                   "fraction_lt_1": float(np.mean(lambdas < 1.0))},
        "integrity": {
            "vote_min_pct": float(votes.min()),
            "max_abs_sum_deviation_pct": float(np.abs(votes.sum(axis=1) - 100.0).max()),
            "all_finite": bool(np.all(np.isfinite(votes))),
            "seat_totals_all_349": bool(np.all(seats.sum(axis=1) == 349)),
            "lambda_in_unit_interval": bool(np.all((lambdas >= 0) & (lambdas <= 1))),
        },
    }
    if detail is not None:
        out["election_noise"] = {
            "k": detail.fit.k, "delta": detail.fit.delta, "tau_sq": detail.fit.tau_sq,
            "bessel_factor": detail.fit.bessel_factor,
            "election_noise_seed": detail.election_noise_seed,
            "residual_zero_sum_max_abs": float(np.abs(detail.residuals_pp.sum(axis=1)).max()),
        }
    return out


def main() -> int:
    print(f"running CONTROL (legacy) at as_of={AS_OF} N={SAMPLES} seed={SEED}", flush=True)
    ctl, ctl_nat, _ = simulate_election_with_noise_model(
        LEGACY_MODEL_ID, as_of=AS_OF, election_date=ELECTION, samples=SAMPLES, seed=SEED)
    print("running adopted law B", flush=True)
    b, b_nat, detail = simulate_election_with_noise_model(
        MODEL_ID, as_of=AS_OF, election_date=ELECTION, samples=SAMPLES, seed=SEED)

    paired = bool(np.array_equal(ctl_nat.base_comp_matrix, b_nat.base_comp_matrix))
    payload = {
        "artifact": "SAME-INPUT 2026 POST-ADOPTION DIAGNOSTIC",
        "status": ("post-adoption diagnostic only; ADOPT_B was frozen before this ran and "
                   "the 2026 forecast was not an adoption input"),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                              cwd=REPO_ROOT).decode().strip(),
        "configuration": {"as_of": AS_OF, "election_date": ELECTION, "samples": SAMPLES,
                          "seed": SEED, "polling_inputs_refreshed": False},
        "adopted_version": {"model_version": ADOPTED_MODEL_VERSION,
                            "candidate": ADOPTED_ELECTION_NOISE_CANDIDATE,
                            "applies_when": "the production default flip is finalised"},
        "upstream_pairing": {
            "base_comp_matrix_identical": paired,
            "meaning": ("OpinionState and Dynamics draws are bit-identical across the two "
                        "runs, so every difference below is the ElectionNoise layer alone"),
        },
        "models": {"CONTROL": summarise(ctl, ctl_nat, None, LEGACY_MODEL_ID),
                   "B": summarise(b, b_nat, detail, MODEL_ID)},
    }
    (OUT / "same_input_2026.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("base_comp_matrix identical:", paired)
    for name in FOCUS_COALITIONS:
        c = payload["models"]["CONTROL"]["focus_coalitions"][name]
        x = payload["models"]["B"]["focus_coalitions"][name]
        print(f"  {name}: CONTROL modes={len(c['material_modes'])} P(maj)={c['prob_majority']:.4f}"
              f" | B modes={len(x['material_modes'])} P(maj)={x['prob_majority']:.4f}")
    for g in GROUPS:
        c = payload["models"]["CONTROL"]["groups"][g]
        x = payload["models"]["B"]["groups"][g]
        print(f"  {g}: CONTROL P(maj)={c['prob_majority']:.4f} -> B {x['prob_majority']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
