"""CLI for the opt-in PoPBaseline v1 simulator."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .config import BASELINE_VERSION, DEFAULT_CONFIG, PARTY_ORDER, PoPBaselineConfig
from .model import simulate_baseline


def _summary(forecast: Any) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for i, party in enumerate(PARTY_ORDER):
        values = forecast.samples_matrix[:, i]
        rows[party] = {
            "mean": float(np.mean(values)),
            "p05": float(np.quantile(values, 0.05)),
            "p25": float(np.quantile(values, 0.25)),
            "p50": float(np.quantile(values, 0.50)),
            "p75": float(np.quantile(values, 0.75)),
            "p95": float(np.quantile(values, 0.95)),
            "prob_above_4pct": None if party == "REST" else float(np.mean(values >= 4.0)),
        }
    return {
        "schema_version": "1.0",
        "model_id": forecast.model_id,
        "model_version": forecast.model_version,
        "origin_date": forecast.origin_date.isoformat(),
        "horizon_days": forecast.horizon_days,
        "samples_count": forecast.samples_count,
        "seed": forecast.seed,
        "party_order": list(forecast.party_order),
        "parties": rows,
        "diagnostics": forecast.diagnostics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PoPBaseline v1 from an exact stored Poll of Polls origin")
    parser.add_argument("--origin", required=True, help="Exact origin date YYYY-MM-DD")
    parser.add_argument("--horizon", required=True, type=int, help="Forecast horizon in days (non-negative)")
    parser.add_argument("--samples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--disable-support-voting", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output including all draws")
    args = parser.parse_args(argv)

    try:
        origin = date.fromisoformat(args.origin)
        if args.horizon < 0 or args.samples <= 0:
            raise ValueError("horizon must be non-negative and samples must be positive")
        config = PoPBaselineConfig(
            step_windows=DEFAULT_CONFIG.step_windows,
            random_sign=DEFAULT_CONFIG.random_sign,
            compositional_space=DEFAULT_CONFIG.compositional_space,
            apply_support_voting=not args.disable_support_voting,
            support_voting_targets=DEFAULT_CONFIG.support_voting_targets,
            partial_step_policy=DEFAULT_CONFIG.partial_step_policy,
        )
        forecast = simulate_baseline(
            origin_date=origin,
            horizon_days=args.horizon,
            samples_count=args.samples,
            seed=args.seed,
            data_dir=args.data_dir,
            config=config,
        )
    except (KeyError, ValueError, TypeError, FileNotFoundError) as exc:
        parser.error(str(exc))

    summary = _summary(forecast)
    if args.output is not None:
        payload = dict(summary)
        payload["draws"] = forecast.samples_matrix.tolist()
        payload["raw_draws"] = forecast.raw_samples_matrix.tolist()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

