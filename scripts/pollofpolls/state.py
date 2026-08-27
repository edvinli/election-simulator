"""Opinion State Estimator v1 for Swedish parliamentary election polling.

This module provides:
1. Strict, leakage-safe as-of estimation of current Swedish party support.
2. The published Poll of Polls estimate as the central mean opinion state.
3. Empirical covariance estimation in compositional Additive Log-Ratio (ALR) coordinates.
4. House-effect adjustment and trailing covariance window with deterministic fallbacks.
5. Recency-weighted Kish effective poll count for state uncertainty.
6. Deterministic Monte Carlo sampling API respecting compositional constraints.
7. Formatted CLI output and machine-readable JSON diagnostic export.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Sequence

from .normalize import parse_date
from .state_config import (
    ALL_CATEGORIES,
    COVARIANCE_DIAGONAL_SHRINKAGE,
    COVARIANCE_LOOKBACK_YEARS,
    FLOATING_POINT_TOLERANCE,
    MAX_EFFECTIVE_POLLS,
    MAX_ESTIMATE_MATCH_LAG_DAYS,
    MAX_SAMPLE_WEIGHT,
    MIN_POLLS_FOR_HOUSE_EFFECT,
    MIN_RESIDUAL_POLLS,
    MIN_SAMPLE_WEIGHT,
    MIN_SHARE_PCT,
    PARTIES,
    RECENCY_HALF_LIFE_DAYS,
    RECENT_POLL_LOOKBACK_DAYS,
    REFERENCE_CATEGORY,
    SAMPLE_SIZE_BENCHMARK,
)
from .state_math import (
    alr_to_composition,
    apply_covariance_shrinkage,
    calculate_sample_covariance,
    calculate_sample_mean,
    cholesky_decomposition_with_jitter,
    composition_to_alr,
    sample_multivariate_normal,
    summarize_samples,
)


@dataclass(frozen=True)
class ReconstructedPoll:
    """A canonical, unique reconstructed polling observation."""

    poll_id: str
    pollster: str
    pollster_original: str
    interview_start: date | None
    interview_end: date | None
    publication_date: date | None
    sample_size: int | None
    party_values: dict[str, float]
    rest_value: float
    reference_date: date | None
    composition: dict[str, float]
    alr_vector: list[float]


def subtract_calendar_years(d: date, years: int = COVARIANCE_LOOKBACK_YEARS) -> date:
    """Subtract calendar years deterministically, handling leap year Feb 29."""
    target_year = d.year - years
    try:
        return date(target_year, d.month, d.day)
    except ValueError:
        # If Feb 29 in leap year shifts to non-leap year, fall back to Feb 28
        return date(target_year, d.month, 28)


def calculate_poll_reference_date(start: date | None, end: date | None) -> date | None:
    """Calculate deterministic reference date as integer midpoint or interview end."""
    if start is not None and end is not None:
        return start + ((end - start) // 2)
    if end is not None:
        return end
    return None


@dataclass
class OpinionState:
    """Programmatic representation of the estimated opinion state and uncertainty distribution."""

    as_of: date
    estimate_date: date
    estimate_age_days: int
    parties: tuple[str, ...]
    mean_pct: dict[str, float]
    rest_pct: float
    mean_alr: list[float]
    covariance_alr: list[list[float]]
    residual_covariance_alr: list[list[float]]
    recent_poll_count: int
    effective_poll_count: float
    residual_poll_count: int
    covariance_fallback_used: bool
    house_effects_alr: dict[str, list[float]]
    diagnostics: dict[str, Any]
    _cholesky_L: list[list[float]] = field(repr=False)

    def sample(self, n: int = 10_000, seed: int | None = None) -> list[dict[str, float]]:
        """Draw n deterministic Monte Carlo samples of 9-part party support compositions.

        Algorithm:
            1. Sample 8-dimensional multivariate normal in ALR coordinates: N(mean_alr, covariance_alr).
            2. Inverse-transform each vector to 9-part composition (M, L, C, KD, S, V, MP, SD, REST).
            3. Each sample guarantees all party shares > 0 and sum(all 9 categories) == 100.0%.

        Parameters:
            n: Number of samples to draw.
            seed: Optional integer random seed for reproducibility.

        Returns:
            List of n dictionaries mapping party names and 'REST' to support percentages.
        """
        if n <= 0:
            raise ValueError(f"Sample count n must be positive, got {n}")
        rng = random.Random(seed)
        samples: list[dict[str, float]] = []
        for _ in range(n):
            alr_draw = sample_multivariate_normal(self.mean_alr, self._cholesky_L, rng)
            comp = alr_to_composition(alr_draw)
            samples.append(comp)
        return samples

    def summary(self, n: int = 10_000, seed: int = 12345) -> dict[str, dict[str, float]]:
        """Compute human-readable summary statistics from deterministic Monte Carlo samples.

        Returns percentiles (P05, P25, P50, P75, P95), mean, and SD for each party and REST.
        """
        samples = self.sample(n=n, seed=seed)
        return summarize_samples(samples)

    def format_table(self, n: int = 10_000, seed: int = 12345) -> str:
        """Format human-readable opinion-state summary table."""
        stats = self.summary(n=n, seed=seed)
        lines: list[str] = [
            f"Requested as-of date:          {self.as_of.isoformat()}",
            f"Poll of Polls estimate date:  {self.estimate_date.isoformat()}",
            f"Estimate age:                 {self.estimate_age_days} days",
            f"Number of recent polls (60d): {self.recent_poll_count}",
            f"Effective poll count (n_eff): {self.effective_poll_count:.2f}",
            f"Residual polls used:          {self.residual_poll_count}",
            f"Covariance fallback used:     {'Yes (expanded backward)' if self.covariance_fallback_used else 'No (4-year window)'}",
            "",
            "Party  | Point Est (%) | State SD |   P05   |   P25   |   P50   |   P75   |   P95   ",
            "-------+---------------+----------+---------+---------+---------+---------+---------",
        ]
        for party in PARTIES:
            p_stat = stats[party]
            point = self.mean_pct[party]
            lines.append(
                f"{party:<6} | {point:>13.2f} | {p_stat['std_dev']:>8.2f} | "
                f"{p_stat['p05']:>7.2f} | {p_stat['p25']:>7.2f} | {p_stat['p50']:>7.2f} | "
                f"{p_stat['p75']:>7.2f} | {p_stat['p95']:>7.2f}"
            )
        rest_stat = stats[REFERENCE_CATEGORY]
        lines.append(
            f"{REFERENCE_CATEGORY:<6} | {self.rest_pct:>13.2f} | {rest_stat['std_dev']:>8.2f} | "
            f"{rest_stat['p05']:>7.2f} | {rest_stat['p25']:>7.2f} | {rest_stat['p50']:>7.2f} | "
            f"{rest_stat['p75']:>7.2f} | {rest_stat['p95']:>7.2f}"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to a JSON-serializable dictionary for diagnostics export."""
        return {
            "as_of": self.as_of.isoformat(),
            "estimate_date": self.estimate_date.isoformat(),
            "estimate_age_days": self.estimate_age_days,
            "parties": list(self.parties),
            "mean_pct": {k: round(v, 4) for k, v in self.mean_pct.items()},
            "rest_pct": round(self.rest_pct, 4),
            "mean_alr": [round(v, 6) for v in self.mean_alr],
            "covariance_alr": [[round(v, 6) for v in row] for row in self.covariance_alr],
            "residual_covariance_alr": [
                [round(v, 6) for v in row] for row in self.residual_covariance_alr
            ],
            "recent_poll_count": self.recent_poll_count,
            "effective_poll_count": round(self.effective_poll_count, 4),
            "residual_poll_count": self.residual_poll_count,
            "covariance_fallback_used": self.covariance_fallback_used,
            "house_effects_alr": {
                pollster: [round(v, 6) for v in vec]
                for pollster, vec in self.house_effects_alr.items()
            },
            "hyperparameters": self.diagnostics.get("hyperparameters", {}),
            "warnings": self.diagnostics.get("warnings", []),
            "exclusions_for_as_of": self.diagnostics.get("exclusions_for_as_of", {}),
            "cholesky_jitter_used": self.diagnostics.get("cholesky_jitter_used", 0.0),
        }


def load_timeseries_dataset(filepath: Path | str) -> list[dict[str, Any]]:
    """Load and parse the Poll of Polls timeseries CSV dataset."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Timeseries CSV not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, Any]] = []
        for r in reader:
            d = parse_date(r["date"])
            row_dict: dict[str, Any] = {"date": d}
            for party in PARTIES:
                val_str = r.get(party)
                if val_str is None or val_str.strip() == "":
                    raise ValueError(f"Missing {party} in timeseries row for date {d}")
                row_dict[party] = float(val_str)
            tot_8 = sum(row_dict[party] for party in PARTIES)
            rest_val = 100.0 - tot_8
            if rest_val < -FLOATING_POINT_TOLERANCE:
                raise ValueError(
                    f"Timeseries row for date {d} has negative REST: {rest_val}"
                )
            if rest_val < 0.0:
                rest_val = 0.0
            row_dict[REFERENCE_CATEGORY] = rest_val
            comp = {p: row_dict[p] for p in PARTIES}
            comp[REFERENCE_CATEGORY] = rest_val
            row_dict["composition"] = comp
            row_dict["alr"] = composition_to_alr(comp)
            rows.append(row_dict)

    rows.sort(key=lambda item: item["date"])
    return rows


def load_individual_polls_dataset(
    filepath: Path | str,
) -> tuple[list[ReconstructedPoll], dict[str, Any]]:
    """Load, reconstruct, and deduplicate individual polls from long-format CSV."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Individual polls CSV not found: {path}")

    raw_by_id: dict[str, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["poll_id"]
            raw_by_id.setdefault(pid, []).append(row)

    reconstructed_polls: list[ReconstructedPoll] = []
    loading_issues: dict[str, int] = {
        "total_polls_in_file": len(raw_by_id),
        "incomplete_main_party_values": 0,
        "invalid_composition_negative_rest": 0,
        "valid_reconstructed_polls": 0,
    }
    anomalous_polls: list[dict[str, Any]] = []

    for pid, rows in sorted(raw_by_id.items()):
        first = rows[0]
        pollster = first["pollster"].strip()
        pollster_original = first.get("pollster_original", "").strip() or pollster
        start_str = first.get("interview_start")
        end_str = first.get("interview_end")
        pub_str = first.get("publication_date")
        sample_str = first.get("sample_size")

        start = parse_date(start_str) if start_str and start_str.strip() else None
        end = parse_date(end_str) if end_str and end_str.strip() else None
        pub = parse_date(pub_str) if pub_str and pub_str.strip() else None
        sample_size = int(float(sample_str)) if sample_str and sample_str.strip() else None

        party_values: dict[str, float] = {}
        for r in rows:
            party_name = r["party"].strip()
            sup_str = r.get("support")
            if sup_str is not None and sup_str.strip() != "":
                try:
                    party_values[party_name] = float(sup_str)
                except ValueError:
                    pass

        # Require all 8 canonical parties to be present
        if not all(p in party_values for p in PARTIES):
            loading_issues["incomplete_main_party_values"] += 1
            continue

        tot_8 = sum(party_values[p] for p in PARTIES)
        rest_val = 100.0 - tot_8

        # Reject materially negative REST
        if rest_val < -FLOATING_POINT_TOLERANCE:
            loading_issues["invalid_composition_negative_rest"] += 1
            anomalous_polls.append({
                "poll_id": pid,
                "pollster": pollster,
                "interview_start": start.isoformat() if start else None,
                "interview_end": end.isoformat() if end else None,
                "total_8_parties": tot_8,
                "rest": rest_val,
            })
            continue

        if rest_val < 0.0:
            rest_val = 0.0

        comp = {p: party_values[p] for p in PARTIES}
        comp[REFERENCE_CATEGORY] = rest_val

        try:
            alr_vec = composition_to_alr(comp)
        except ValueError as err:
            loading_issues["invalid_composition_negative_rest"] += 1
            anomalous_polls.append({"poll_id": pid, "error": str(err)})
            continue

        ref_date = calculate_poll_reference_date(start, end)

        poll = ReconstructedPoll(
            poll_id=pid,
            pollster=pollster,
            pollster_original=pollster_original,
            interview_start=start,
            interview_end=end,
            publication_date=pub,
            sample_size=sample_size,
            party_values=party_values,
            rest_value=rest_val,
            reference_date=ref_date,
            composition=comp,
            alr_vector=alr_vec,
        )
        reconstructed_polls.append(poll)

    loading_issues["valid_reconstructed_polls"] = len(reconstructed_polls)
    loading_issues["anomalous_polls"] = anomalous_polls
    return reconstructed_polls, loading_issues


def estimate_opinion(
    as_of: str | date | None = None,
    *,
    data_dir: Path | str | None = None,
) -> OpinionState:
    """Estimate Swedish parliamentary party support and uncertainty as of a specific date.

    Parameters:
        as_of: Target historical or current date (ISO string 'YYYY-MM-DD' or date object).
               If omitted (None), defaults to the maximum date in pollofpolls_timeseries.csv.
        data_dir: Optional path to directory containing data/processed/pollofpolls.
                  If None, resolves relative to repository layout.

    Returns:
        OpinionState object containing point estimates, ALR covariance, diagnostics,
        and Monte Carlo sampling methods.
    """
    base_path = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "data" / "processed" / "pollofpolls"
    ts_file = base_path / "pollofpolls_timeseries.csv"
    ind_file = base_path / "individual_polls.csv"

    timeseries_data = load_timeseries_dataset(ts_file)
    individual_polls, load_diagnostics = load_individual_polls_dataset(ind_file)

    if not timeseries_data:
        raise ValueError("Poll of Polls timeseries contains no observations")

    max_ts_date = timeseries_data[-1]["date"]

    if as_of is None:
        target_as_of = max_ts_date
    elif isinstance(as_of, str):
        target_as_of = parse_date(as_of)
    elif isinstance(as_of, date):
        target_as_of = as_of
    else:
        raise TypeError(f"as_of must be str, date, or None; got {type(as_of)}")

    # 1. Central Point Estimate from Poll of Polls timeseries on or before as_of
    eligible_ts = [row for row in timeseries_data if row["date"] <= target_as_of]
    if not eligible_ts:
        raise ValueError(
            f"No Poll of Polls timeseries observation available on or before {target_as_of}"
        )
    central_row = eligible_ts[-1]
    estimate_date = central_row["date"]
    estimate_age_days = (target_as_of - estimate_date).days
    mean_pct = {p: central_row[p] for p in PARTIES}
    rest_pct = central_row[REFERENCE_CATEGORY]
    mean_alr = list(central_row["alr"])

    # 2. Strict Historical Residual Matching (publication_date < as_of)
    exclusions: dict[str, int] = {
        "missing_publication_date": 0,
        "missing_interview_end": 0,
        "future_publication_date": 0,
        "future_interview_end": 0,
        "no_reference_date": 0,
        "no_matching_timeseries_estimate": 0,
        "timeseries_match_lag_exceeded": 0,
    }
    warnings: list[str] = []

    eligible_residuals: list[dict[str, Any]] = []

    # Map dates for fast binary or sequential lookup
    ts_dates = [row["date"] for row in timeseries_data]

    for poll in individual_polls:
        if poll.publication_date is None:
            exclusions["missing_publication_date"] += 1
            continue

        # Prevent future leakage in residual estimation: require publication_date < as_of
        if poll.publication_date >= target_as_of:
            exclusions["future_publication_date"] += 1
            continue

        # Also require interview_end <= as_of when present
        if poll.interview_end is not None and poll.interview_end > target_as_of:
            exclusions["future_interview_end"] += 1
            continue

        if poll.interview_end is None:
            exclusions["missing_interview_end"] += 1
            continue

        if poll.reference_date is None:
            exclusions["no_reference_date"] += 1
            continue

        # Find latest PoP timeseries observation on or before poll reference_date
        # Binary or linear search
        candidates = [row for row in timeseries_data if row["date"] <= poll.reference_date]
        if not candidates:
            exclusions["no_matching_timeseries_estimate"] += 1
            continue

        matched_pop = candidates[-1]
        lag_days = (poll.reference_date - matched_pop["date"]).days
        if lag_days > MAX_ESTIMATE_MATCH_LAG_DAYS:
            exclusions["timeseries_match_lag_exceeded"] += 1
            continue

        # Compute 8D ALR residual: poll_alr - pop_alr
        residual = [
            poll.alr_vector[i] - matched_pop["alr"][i] for i in range(len(PARTIES))
        ]
        eligible_residuals.append({
            "poll_id": poll.poll_id,
            "pollster": poll.pollster,
            "publication_date": poll.publication_date,
            "reference_date": poll.reference_date,
            "matched_pop_date": matched_pop["date"],
            "lag_days": lag_days,
            "residual": residual,
        })

    # 3. Trailing 4-Year Window & Fallback
    window_start = subtract_calendar_years(target_as_of, COVARIANCE_LOOKBACK_YEARS)
    window_residuals = [
        r for r in eligible_residuals if r["publication_date"] >= window_start
    ]

    if len(window_residuals) >= MIN_RESIDUAL_POLLS:
        active_residuals = window_residuals
        fallback_used = False
    else:
        active_residuals = eligible_residuals
        fallback_used = True
        warnings.append(
            f"Fewer than {MIN_RESIDUAL_POLLS} residual polls in 4-year trailing window ({len(window_residuals)} found); "
            f"expanded backward to use all {len(eligible_residuals)} eligible prior residuals."
        )

    if len(active_residuals) < 10:
        raise ValueError(
            f"Insufficient eligible historical residual polls ({len(active_residuals)}) to estimate 8x8 covariance matrix reliably."
        )

    # 4. Pollster House Effects Estimation from Active Residual Pool
    pollster_residuals: dict[str, list[list[float]]] = {}
    for r in active_residuals:
        pollster_residuals.setdefault(r["pollster"], []).append(r["residual"])

    house_effects_alr: dict[str, list[float]] = {}
    for pollster, res_list in sorted(pollster_residuals.items()):
        if len(res_list) >= MIN_POLLS_FOR_HOUSE_EFFECT:
            house_effects_alr[pollster] = calculate_sample_mean(res_list)
        else:
            house_effects_alr[pollster] = [0.0] * len(PARTIES)

    # 5. Adjusted Residuals and Shrunk Covariance
    adjusted_residuals: list[list[float]] = []
    for r in active_residuals:
        he = house_effects_alr[r["pollster"]]
        adj = [r["residual"][i] - he[i] for i in range(len(PARTIES))]
        adjusted_residuals.append(adj)

    cov_raw = calculate_sample_covariance(adjusted_residuals)
    residual_cov = apply_covariance_shrinkage(cov_raw, COVARIANCE_DIAGONAL_SHRINKAGE)

    # 6. Current Polling Information & Effective Poll Count
    recent_lookback_start = target_as_of - timedelta(days=RECENT_POLL_LOOKBACK_DAYS)
    recent_polls_selected: list[ReconstructedPoll] = []
    recent_weights: list[float] = []

    for poll in individual_polls:
        # Strict as_of checks for current polls:
        # publication_date <= as_of
        # interview_end <= as_of (when present)
        # reference_date <= as_of
        # reference_date >= as_of - 60 days
        if poll.publication_date is None or poll.publication_date > target_as_of:
            continue
        if poll.interview_end is not None and poll.interview_end > target_as_of:
            continue
        if poll.reference_date is None:
            continue
        if not (recent_lookback_start <= poll.reference_date <= target_as_of):
            continue

        recent_polls_selected.append(poll)

        age_days = (target_as_of - poll.reference_date).days
        recency_weight = math.exp(-math.log(2.0) * age_days / RECENCY_HALF_LIFE_DAYS)

        if poll.sample_size is not None and poll.sample_size > 0:
            raw_sample_weight = math.sqrt(poll.sample_size / SAMPLE_SIZE_BENCHMARK)
            sample_weight = min(max(raw_sample_weight, MIN_SAMPLE_WEIGHT), MAX_SAMPLE_WEIGHT)
        else:
            sample_weight = 1.0

        total_weight = recency_weight * sample_weight
        recent_weights.append(total_weight)

    if recent_weights:
        sum_w = sum(recent_weights)
        sum_w_sq = sum(w * w for w in recent_weights)
        n_eff = (sum_w ** 2) / sum_w_sq
        n_eff_used = min(max(n_eff, 1.0), MAX_EFFECTIVE_POLLS)
    else:
        n_eff = 1.0
        n_eff_used = 1.0
        warnings.append(
            f"No eligible recent polls found in trailing {RECENT_POLL_LOOKBACK_DAYS} days; using minimum effective poll count 1.0."
        )

    # 7. State Covariance
    state_cov = [
        [residual_cov[i][j] / n_eff_used for j in range(len(PARTIES))]
        for i in range(len(PARTIES))
    ]

    # 8. Cholesky Decomposition with Bounded Jitter Search
    cholesky_L, jitter_used = cholesky_decomposition_with_jitter(state_cov)

    diagnostics: dict[str, Any] = {
        "hyperparameters": {
            "MIN_SHARE_PCT": MIN_SHARE_PCT,
            "MAX_ESTIMATE_MATCH_LAG_DAYS": MAX_ESTIMATE_MATCH_LAG_DAYS,
            "COVARIANCE_LOOKBACK_YEARS": COVARIANCE_LOOKBACK_YEARS,
            "MIN_RESIDUAL_POLLS": MIN_RESIDUAL_POLLS,
            "MIN_POLLS_FOR_HOUSE_EFFECT": MIN_POLLS_FOR_HOUSE_EFFECT,
            "COVARIANCE_DIAGONAL_SHRINKAGE": COVARIANCE_DIAGONAL_SHRINKAGE,
            "RECENT_POLL_LOOKBACK_DAYS": RECENT_POLL_LOOKBACK_DAYS,
            "RECENCY_HALF_LIFE_DAYS": RECENCY_HALF_LIFE_DAYS,
            "MAX_EFFECTIVE_POLLS": MAX_EFFECTIVE_POLLS,
        },
        "recent_poll_count": len(recent_polls_selected),
        "effective_poll_count_raw": round(n_eff, 4),
        "effective_poll_count_used": round(n_eff_used, 4),
        "residual_poll_count": len(active_residuals),
        "pollsters_represented_in_residuals": sorted(pollster_residuals.keys()),
        "pollsters_with_house_effects": sorted([p for p, vec in house_effects_alr.items() if any(v != 0.0 for v in vec)]),
        "warnings": warnings,
        "exclusions_for_as_of": exclusions,
        "data_loading_diagnostics": load_diagnostics,
        "cholesky_jitter_used": jitter_used,
        "window_start": window_start.isoformat(),
    }

    return OpinionState(
        as_of=target_as_of,
        estimate_date=estimate_date,
        estimate_age_days=estimate_age_days,
        parties=PARTIES,
        mean_pct=mean_pct,
        rest_pct=rest_pct,
        mean_alr=mean_alr,
        covariance_alr=state_cov,
        residual_covariance_alr=residual_cov,
        recent_poll_count=len(recent_polls_selected),
        effective_poll_count=n_eff_used,
        residual_poll_count=len(active_residuals),
        covariance_fallback_used=fallback_used,
        house_effects_alr=house_effects_alr,
        diagnostics=diagnostics,
        _cholesky_L=cholesky_L,
    )


def main(args_list: Sequence[str] | None = None) -> int:
    """CLI entry point for Opinion State Estimator v1."""
    parser = argparse.ArgumentParser(
        description="Estimate current Swedish party support and uncertainty as of a specified date."
    )
    parser.add_argument(
        "--as-of",
        dest="as_of",
        default=None,
        help="Target as-of date (YYYY-MM-DD). If omitted, defaults to latest available Poll of Polls date.",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Output machine-readable JSON diagnostics instead of formatted table.",
    )
    parser.add_argument(
        "--samples",
        dest="samples",
        type=int,
        default=10_000,
        help="Number of Monte Carlo samples for uncertainty summary (default: 10000).",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=12345,
        help="Random seed for reproducible Monte Carlo sampling (default: 12345).",
    )
    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        default=None,
        help="Custom directory path containing processed Pollofpolls CSV files.",
    )

    args = parser.parse_args(args_list)

    try:
        state = estimate_opinion(as_of=args.as_of, data_dir=args.data_dir)
        if args.json_output:
            out_dict = state.to_dict()
            out_dict["summary_statistics"] = state.summary(n=args.samples, seed=args.seed)
            print(json.dumps(out_dict, indent=2, ensure_ascii=False))
        else:
            print("=================================================================")
            print("        SWEDISH OPINION STATE ESTIMATOR v1.1 (AS-OF MODEL)       ")
            print("=================================================================")

            print(state.format_table(n=args.samples, seed=args.seed))
            print("-----------------------------------------------------------------")
            if state.diagnostics.get("warnings"):
                print("Warnings:")
                for w in state.diagnostics["warnings"]:
                    print(f"  * {w}")
                print("-----------------------------------------------------------------")
            print("House-Effect Pollsters (>= 20 historical polls):")
            he_active = [
                p for p, vec in state.house_effects_alr.items() if any(v != 0.0 for v in vec)
            ]
            if he_active:
                for p in he_active:
                    vec_str = ", ".join(
                        f"{party}: {val:+.3f}"
                        for party, val in zip(PARTIES, state.house_effects_alr[p])
                    )
                    print(f"  * {p:<10}: ALR diff = [{vec_str}]")
            else:
                print("  * None (no pollster met the 20-poll threshold)")
            print("=================================================================")
        return 0
    except Exception as err:
        sys.stderr.write(f"Error estimating opinion state: {err}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
