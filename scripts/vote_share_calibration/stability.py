"""Multi-seed stability audit at N=20,000 samples for top vote-share model candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd

from .config import (
    CANONICAL_MODELS,
    DEFAULT_OUTPUT_DIR,
    HIGH_SAMPLES,
    STABILITY_SEEDS,
)
from .hindcast import run_vote_share_hindcasts


def run_multi_seed_stability_audit(
    seeds: Sequence[int] = STABILITY_SEEDS,
    models: Sequence[str] = ("pp_centered_noise", "pp_symmetric_noise"),
    samples: int = HIGH_SAMPLES,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Execute high-sample stability runs across multiple fixed seeds to verify ranking stability."""
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    stability_rows: list[dict[str, Any]] = []

    for s in seeds:
        print(f"  --> Running stability check at samples={samples} with seed={s} ...")
        res = run_vote_share_hindcasts(
            models=models,
            samples=samples,
            seed=s,
            output_dir=out_dir / f"seed_{s}",
        )

        summ = res["summary"]
        for row in summ["by_model_overall"]:
            m_id = row["model"]
            # Extract 2018 and 2022 specific CRPS
            sub_18 = next(
                r for r in summ["by_election_model"]
                if r["model"] == m_id and r["election_date"] == "2018-09-09"
            )
            sub_22 = next(
                r for r in summ["by_election_model"]
                if r["model"] == m_id and r["election_date"] == "2022-09-11"
            )

            stability_rows.append({
                "seed": s,
                "samples": samples,
                "model": m_id,
                "CRPS_8parties_overall": round(row["mean_CRPS_8parties"], 4),
                "CRPS_8parties_2018": round(sub_18["mean_CRPS_8parties"], 4),
                "CRPS_8parties_2022": round(sub_22["mean_CRPS_8parties"], 4),
                "EnergyScore_all9": round(row["EnergyScore_all9"], 6),
                "coverage_90": round(row["coverage_90"], 4),
                "mean_width_90": round(row["mean_width_90"], 4),
                "mean_lambda": round(row["mean_lambda"], 4),
            })

    df_stability = pd.DataFrame(stability_rows)
    csv_path = out_dir / "stability_20k_seeds.csv"
    json_path = out_dir / "stability_20k_seeds.json"

    df_stability.to_csv(csv_path, index=False)

    # Average metrics across seeds per model
    model_averages: list[dict[str, Any]] = []
    for m in models:
        sub = df_stability[df_stability["model"] == m]
        model_averages.append({
            "model": m,
            "mean_CRPS_8parties_across_seeds": round(float(sub["CRPS_8parties_overall"].mean()), 4),
            "mean_CRPS_2018_across_seeds": round(float(sub["CRPS_8parties_2018"].mean()), 4),
            "mean_CRPS_2022_across_seeds": round(float(sub["CRPS_8parties_2022"].mean()), 4),
            "mean_EnergyScore_across_seeds": round(float(sub["EnergyScore_all9"].mean()), 6),
            "mean_coverage_90_across_seeds": round(float(sub["coverage_90"].mean()), 4),
        })

    report_data = {
        "seeds": list(seeds),
        "samples": samples,
        "models_evaluated": list(models),
        "by_seed_and_model": stability_rows,
        "model_averages_across_seeds": model_averages,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    return {
        "stability_df": df_stability,
        "report": report_data,
        "paths": {
            "csv": str(csv_path),
            "json": str(json_path),
        },
    }
