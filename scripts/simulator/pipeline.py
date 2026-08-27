"""Command-line pipeline runner and benchmark suite for ElectionSimulator v1."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Any, Sequence
import numpy as np

from scripts.geography.audit_sensitivity import (
    run_geography_baseline_sensitivity_audit,
    run_integerization_sensitivity_audit,
)
from .config import (
    DEFAULT_ELECTION_DATE,
    DEFAULT_GEOGRAPHY_BASELINE_YEAR,
    DEFAULT_SIMULATION_SAMPLES,
    DEFAULT_SIMULATION_SEED,
    DEFAULT_SIMULATIONS_DIR,
    PARLIAMENTARY_PARTIES_8,
)
from .engine import simulate_election
from .reproducibility import compute_simulation_payload_sha256


def build_canonical_summary_dict(sim_res: Any) -> dict[str, Any]:
    """Serialize the canonical forecast summary from a ``SimulationResult``."""
    tido = sim_res.summarize_group(["M", "SD", "KD", "L"])
    rg = sim_res.summarize_group(["S", "V", "MP", "C"])
    summary_dict: dict[str, Any] = {
        "as_of": sim_res.summary.as_of,
        "election_date": sim_res.summary.election_date,
        "total_samples": sim_res.summary.total_samples,
        "parties": {
            p: {
                "party": p_info.party,
                "vote_share_mean": round(p_info.vote_share_mean * 100, 3),
                "vote_share_median": round(p_info.vote_share_median * 100, 3),
                "vote_share_p05": round(p_info.vote_share_p05 * 100, 3),
                "vote_share_p95": round(p_info.vote_share_p95 * 100, 3),
                "prob_above_4pct": round(p_info.prob_above_4pct, 4),
                "prob_below_4pct": round(p_info.prob_below_4pct, 4),
                "prob_largest_vote_party": round(p_info.prob_largest_vote_party, 4),
                "seats_mean": round(p_info.seats_mean, 2),
                "seats_median": p_info.seats_median,
                "seats_p05": p_info.seats_p05,
                "seats_p95": p_info.seats_p95,
                "prob_largest_seat_party": round(p_info.prob_largest_seat_party, 4),
                "prob_any_seats": round(p_info.prob_any_seats, 4),
                "prob_local_12pct_exception_sub_4pct": round(p_info.prob_local_12pct_exception_sub_4pct, 6),
            }
            for p, p_info in sim_res.summary.parties.items()
        },
        "blocs": {
            "tido": {
                "parties": ["M", "SD", "KD", "L"],
                "mean_seats": round(tido.mean_seats, 2),
                "prob_majority": round(tido.prob_majority, 4),
            },
            "red_green_center": {
                "parties": ["S", "V", "MP", "C"],
                "mean_seats": round(rg.mean_seats, 2),
                "prob_majority": round(rg.prob_majority, 4),
            },
        },
        "manifest": sim_res.manifest,
    }
    if sim_res.quantization_audit is not None:
        summary_dict["quantization_audit"] = sim_res.quantization_audit
    summary_dict["deterministic_payload_sha256"] = compute_simulation_payload_sha256(
        sim_res.vote_shares_matrix,
        sim_res.seats_matrix,
        summary_dict,
    )
    return summary_dict


def run_benchmarks(
    as_of: str | None = None,
    election_date: str = DEFAULT_ELECTION_DATE,
    seed: int = DEFAULT_SIMULATION_SEED,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Run performance and memory benchmarks for 1k, 10k, and 100k simulations."""
    out_dir = Path(output_dir) if output_dir else DEFAULT_SIMULATIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_sizes = [1_000, 10_000, 100_000]
    results: dict[str, Any] = {}

    print("\n==========================================================================================")
    print("RUNNING SIMULATION PERFORMANCE & MEMORY BENCHMARKS (1k, 10k, 100k)")
    print("==========================================================================================")

    for n in benchmark_sizes:
        print(f"\n>>> Benchmarking N = {n:,d} simulations ...")
        tracemalloc.start()
        t0 = time.perf_counter()

        sim_res = simulate_election(
            as_of=as_of,
            election_date=election_date,
            samples=n,
            seed=seed,
        )

        t1 = time.perf_counter()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        elapsed_sec = t1 - t0
        rate = n / elapsed_sec if elapsed_sec > 0 else 0.0
        peak_mb = peak_mem / (1024 * 1024)

        print(f"  Completed N={n:,d} in {elapsed_sec:.3f} s ({rate:,.1f} sims/sec) | Peak memory: {peak_mb:.2f} MB")
        print("  Summary Seats (Mean):", {p: round(sim_res.summary.parties[p].seats_mean, 1) for p in PARLIAMENTARY_PARTIES_8})

        # Test Bloc Majorities
        tido = sim_res.summarize_group(["M", "SD", "KD", "L"])
        rg = sim_res.summarize_group(["S", "V", "MP", "C"])
        print(f"  P(Tidö >= 175): {tido.prob_majority * 100:.1f}% (Mean seats: {tido.mean_seats:.1f})")
        print(f"  P(Red-Green-Center >= 175): {rg.prob_majority * 100:.1f}% (Mean seats: {rg.mean_seats:.1f})")

        results[f"benchmark_{n}"] = {
            "samples": n,
            "wall_clock_seconds": round(elapsed_sec, 4),
            "simulations_per_second": round(rate, 1),
            "peak_memory_mb": round(peak_mb, 2),
            "tido_majority_prob": round(tido.prob_majority, 4),
            "red_green_majority_prob": round(rg.prob_majority, 4),
        }

    bench_path = out_dir / "simulation_benchmark_report.json"
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark report saved to {bench_path}")
    return results


def run_full_pipeline(
    as_of: str | None = None,
    election_date: str = DEFAULT_ELECTION_DATE,
    samples: int = DEFAULT_SIMULATION_SAMPLES,
    seed: int = DEFAULT_SIMULATION_SEED,
    baseline_year: int = DEFAULT_GEOGRAPHY_BASELINE_YEAR,
    benchmark: bool = False,
    audit_sensitivity: bool = False,
    output_dir: Path | str | None = None,
) -> int:
    """Execute complete simulation pipeline, audits, and exports."""
    out_dir = Path(output_dir) if output_dir else DEFAULT_SIMULATIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if audit_sensitivity:
        print(">>> Step 1: Running Geography Baseline Sensitivity Audit ...")
        run_geography_baseline_sensitivity_audit(output_dir=out_dir)

        print("\n>>> Step 2: Running Integerization Sensitivity Audit ...")
        run_integerization_sensitivity_audit(output_dir=out_dir)

    if benchmark:
        run_benchmarks(as_of=as_of, election_date=election_date, seed=seed, output_dir=out_dir)

    print(f"\n>>> Running Production Simulation: N = {samples:,d}, Seed = {seed}, As-Of = {as_of or 'latest'} ...")
    t0 = time.perf_counter()
    sim_res = simulate_election(
        as_of=as_of,
        election_date=election_date,
        samples=samples,
        seed=seed,
        baseline_year=baseline_year,
    )
    t1 = time.perf_counter()

    print(f"Simulation finished in {t1 - t0:.2f} s.")

    # Export canonical summary JSON. The deterministic payload hash is derived
    # from the simulation arrays plus this summary and excludes timestamps.
    summary_dict = build_canonical_summary_dict(sim_res)

    out_file = out_dir / f"simulation_summary_N{samples}_seed{seed}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
        f.write("\n")

    # Keep a small sidecar for reviewers and downstream jobs that need the
    # deterministic payload identity without parsing the full JSON artifact.
    # The sidecar contains only the payload hash; timestamps/runtime are not
    # part of that identity.
    hash_file = out_dir / "deterministic_payload.sha256"
    hash_file.write_text(f"{summary_dict['deterministic_payload_sha256']}\n", encoding="utf-8")

    print(f"Simulation summary exported to {out_file}")
    print(f"Deterministic payload hash exported to {hash_file}")
    return 0


def main(args_list: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Swedish Riksdag ElectionSimulator v1 Pipeline.")
    parser.add_argument("--as-of", type=str, default=None, help="Polls cutoff date (YYYY-MM-DD).")
    parser.add_argument("--election-date", type=str, default=DEFAULT_ELECTION_DATE, help="Election date (YYYY-MM-DD).")
    parser.add_argument("--samples", type=int, default=DEFAULT_SIMULATION_SAMPLES, help="Number of simulation samples.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SIMULATION_SEED, help="Base random seed.")
    parser.add_argument("--baseline-year", type=int, default=DEFAULT_GEOGRAPHY_BASELINE_YEAR, help="Baseline geography election year.")
    parser.add_argument("--benchmark", action="store_true", help="Run 1k, 10k, 100k benchmarks.")
    parser.add_argument("--audit-sensitivity", action="store_true", help="Run geography baseline and integerization sensitivity audits.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for simulation results.")

    args = parser.parse_args(args_list)
    return run_full_pipeline(
        as_of=args.as_of,
        election_date=args.election_date,
        samples=args.samples,
        seed=args.seed,
        baseline_year=args.baseline_year,
        benchmark=args.benchmark,
        audit_sensitivity=args.audit_sensitivity,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
