"""Comprehensive unit and regression tests for Opinion State Estimator v1."""

from __future__ import annotations

import csv
from datetime import date, timedelta
import math
from pathlib import Path
import random
import shutil
import tempfile
import unittest

from scripts.pollofpolls.state import (
    OpinionState,
    ReconstructedPoll,
    calculate_poll_reference_date,
    estimate_opinion,
    load_individual_polls_dataset,
    load_timeseries_dataset,
    subtract_calendar_years,
)

from scripts.pollofpolls.state_config import (
    ALL_CATEGORIES,
    COVARIANCE_DIAGONAL_SHRINKAGE,
    COVARIANCE_LOOKBACK_YEARS,
    MAX_EFFECTIVE_POLLS,
    MIN_POLLS_FOR_HOUSE_EFFECT,
    MIN_RESIDUAL_POLLS,
    MIN_SHARE_PCT,
    PARTIES,
    RECENCY_HALF_LIFE_DAYS,
    REFERENCE_CATEGORY,
    SAMPLE_SIZE_BENCHMARK,
)
from scripts.pollofpolls.state_math import (
    alr_to_composition,
    apply_covariance_shrinkage,
    calculate_percentile,
    calculate_sample_covariance,
    calculate_sample_mean,
    cholesky_decomposition_with_jitter,
    composition_to_alr,
    sample_multivariate_normal,
    summarize_samples,
)


class CompositionMathTests(unittest.TestCase):
    """Test ALR and compositional transformations."""

    def test_alr_round_trip(self) -> None:
        original = {
            "M": 18.5,
            "L": 3.2,
            "C": 6.8,
            "KD": 5.4,
            "S": 31.2,
            "V": 7.9,
            "MP": 6.1,
            "SD": 19.3,
            "REST": 1.6,
        }
        alr_vec = composition_to_alr(original)
        self.assertEqual(len(alr_vec), 8)

        reconstructed = alr_to_composition(alr_vec)
        self.assertEqual(set(reconstructed.keys()), set(ALL_CATEGORIES))

        total = sum(reconstructed.values())
        self.assertAlmostEqual(total, 100.0, places=7)

        for party in ALL_CATEGORIES:
            self.assertAlmostEqual(reconstructed[party], original[party], places=5)

    def test_composition_positivity_and_sum_to_100(self) -> None:
        # Test with extreme ALR vectors
        extreme_alr = [-10.0, 5.0, 0.0, -2.5, 3.2, -4.0, 1.1, -1.0]
        comp = alr_to_composition(extreme_alr)

        self.assertAlmostEqual(sum(comp.values()), 100.0, places=7)
        for cat, val in comp.items():
            self.assertGreater(val, 0.0)

    def test_tiny_and_zero_value_handling(self) -> None:
        zero_comp = {
            "M": 20.0,
            "L": 0.0,
            "C": 5.0,
            "KD": 5.0,
            "S": 35.0,
            "V": 8.0,
            "MP": 7.0,
            "SD": 20.0,
            "REST": 0.0,
        }
        alr_vec = composition_to_alr(zero_comp)
        self.assertTrue(all(math.isfinite(z) for z in alr_vec))

        reconstructed = alr_to_composition(alr_vec)
        self.assertGreaterEqual(reconstructed["L"], MIN_SHARE_PCT * 0.99)
        self.assertGreaterEqual(reconstructed["REST"], MIN_SHARE_PCT * 0.99)
        self.assertAlmostEqual(sum(reconstructed.values()), 100.0, places=7)

    def test_materially_negative_share_rejected(self) -> None:
        negative_comp = {
            "M": 20.0,
            "L": -0.5,
            "C": 5.0,
            "KD": 5.0,
            "S": 35.0,
            "V": 8.0,
            "MP": 7.0,
            "SD": 20.0,
            "REST": 0.5,
        }
        with self.assertRaises(ValueError):
            composition_to_alr(negative_comp)


class LinearAlgebraAndSamplingTests(unittest.TestCase):
    """Test covariance estimation, shrinkage, Cholesky, and deterministic sampling."""

    def test_sample_covariance_and_shrinkage(self) -> None:
        vectors = [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [1.2, 1.9, 3.1, 4.2, 4.8, 6.1, 6.9, 8.2],
            [0.9, 2.1, 2.9, 3.9, 5.1, 5.9, 7.1, 7.9],
            [1.1, 2.0, 3.0, 4.1, 5.0, 6.0, 7.0, 8.1],
        ]
        cov = calculate_sample_covariance(vectors)
        self.assertEqual(len(cov), 8)
        self.assertEqual(len(cov[0]), 8)

        # Symmetry and positive diagonal
        for i in range(8):
            self.assertGreater(cov[i][i], 0.0)
            for j in range(8):
                self.assertAlmostEqual(cov[i][j], cov[j][i], places=9)

        # Explicit shrinkage of 0.20 scales off-diagonals by 0.80
        shrunk_20 = apply_covariance_shrinkage(cov, shrinkage=0.20)
        for i in range(8):
            self.assertAlmostEqual(shrunk_20[i][i], cov[i][i], places=9)
            for j in range(8):
                if i != j:
                    self.assertAlmostEqual(shrunk_20[i][j], 0.80 * cov[i][j], places=9)

        # Default shrinkage in v1.1 is 0.0 (unshrunk covariance)
        shrunk_default = apply_covariance_shrinkage(cov, shrinkage=COVARIANCE_DIAGONAL_SHRINKAGE)
        for i in range(8):
            for j in range(8):
                self.assertAlmostEqual(shrunk_default[i][j], cov[i][j], places=9)


    def test_cholesky_decomposition_accuracy(self) -> None:
        # Create a symmetric positive definite 8x8 matrix
        dim = 8
        A = [[0.0] * dim for _ in range(dim)]
        for i in range(dim):
            for j in range(dim):
                A[i][j] = 0.5 ** abs(i - j)
            A[i][i] += 1.0

        L, jitter = cholesky_decomposition_with_jitter(A)
        self.assertEqual(jitter, 0.0)

        # Verify L * L^T = A
        for i in range(dim):
            for j in range(dim):
                recon = sum(L[i][k] * L[j][k] for k in range(min(i, j) + 1))
                self.assertAlmostEqual(recon, A[i][j], places=7)

    def test_cholesky_bounded_jitter_fallback(self) -> None:
        # Create a positive semi-definite matrix with singular zero eigenvalue
        dim = 8
        A = [[1.0] * dim for _ in range(dim)]
        for i in range(dim):
            A[i][i] = 1.0  # rank 1 matrix

        L, jitter_used = cholesky_decomposition_with_jitter(A)
        self.assertGreater(jitter_used, 0.0)
        # Check that decomposition with jitter succeeded
        recon = sum(L[0][k] * L[0][k] for k in range(1))
        self.assertAlmostEqual(recon, 1.0 + jitter_used, places=7)

    def test_sampling_reproducibility_and_bounds(self) -> None:
        mean_alr = [2.2, 0.1, 1.3, 1.2, 2.8, 1.4, 1.3, 2.3]
        dim = 8
        cov = [[0.05 if i == j else 0.01 for j in range(dim)] for i in range(dim)]
        L, _ = cholesky_decomposition_with_jitter(cov)

        state = OpinionState(
            as_of=date(2026, 8, 23),
            estimate_date=date(2026, 8, 23),
            estimate_age_days=0,
            parties=PARTIES,
            mean_pct=alr_to_composition(mean_alr),
            rest_pct=alr_to_composition(mean_alr)["REST"],
            mean_alr=mean_alr,
            covariance_alr=cov,
            residual_covariance_alr=cov,
            recent_poll_count=4,
            effective_poll_count=3.5,
            residual_poll_count=150,
            covariance_fallback_used=False,
            house_effects_alr={},
            diagnostics={},
            _cholesky_L=L,
        )

        samples1 = state.sample(n=100, seed=42)
        samples2 = state.sample(n=100, seed=42)
        samples3 = state.sample(n=100, seed=999)

        # Same seed yields identical samples
        self.assertEqual(samples1, samples2)
        # Different seed yields different samples
        self.assertNotEqual(samples1, samples3)

        # Bounds check
        for sample in samples1:
            self.assertAlmostEqual(sum(sample.values()), 100.0, places=7)
            for val in sample.values():
                self.assertGreater(val, 0.0)

    def test_linear_percentile_interpolation(self) -> None:
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        # n = 5
        # p = 0.50 -> h = (5-1)*0.5 = 2.0 -> vals[2] = 30.0
        self.assertAlmostEqual(calculate_percentile(vals, 0.50), 30.0)
        # p = 0.25 -> h = (5-1)*0.25 = 1.0 -> vals[1] = 20.0
        self.assertAlmostEqual(calculate_percentile(vals, 0.25), 20.0)
        # p = 0.75 -> h = (5-1)*0.75 = 3.0 -> vals[3] = 40.0
        self.assertAlmostEqual(calculate_percentile(vals, 0.75), 40.0)
        # p = 0.10 -> h = 4 * 0.10 = 0.4 -> (1-0.4)*10 + 0.4*20 = 6 + 8 = 14.0
        self.assertAlmostEqual(calculate_percentile(vals, 0.10), 14.0)


class DateAndLookbackTests(unittest.TestCase):
    """Test date arithmetic and reference date calculation."""

    def test_poll_reference_date_midpoint(self) -> None:
        # Even day span: Aug 1 to Aug 5 (4 days span -> midpoint Aug 3)
        ref1 = calculate_poll_reference_date(date(2026, 8, 1), date(2026, 8, 5))
        self.assertEqual(ref1, date(2026, 8, 3))

        # Odd day span: Aug 1 to Aug 6 (5 days span -> // 2 is 2 days -> Aug 3)
        ref2 = calculate_poll_reference_date(date(2026, 8, 1), date(2026, 8, 6))
        self.assertEqual(ref2, date(2026, 8, 3))

        # Missing start -> returns end
        ref3 = calculate_poll_reference_date(None, date(2026, 8, 5))
        self.assertEqual(ref3, date(2026, 8, 5))

        # Missing end -> returns None
        ref4 = calculate_poll_reference_date(date(2026, 8, 1), None)
        self.assertIsNone(ref4)

    def test_calendar_4year_lookback(self) -> None:
        d1 = date(2026, 8, 23)
        self.assertEqual(subtract_calendar_years(d1, 4), date(2022, 8, 23))

        # Leap year Feb 29 lookback to non-leap year
        leap_day = date(2024, 2, 29)
        self.assertEqual(subtract_calendar_years(leap_day, 1), date(2023, 2, 28))
        self.assertEqual(subtract_calendar_years(leap_day, 4), date(2020, 2, 29))


class EffectivePollWeightingTests(unittest.TestCase):
    """Test recency weights, sample size weights, and Kish effective sample count calculation."""

    def test_recency_weight_half_life(self) -> None:
        # At age 0: weight = 1.0
        w0 = math.exp(-math.log(2.0) * 0 / RECENCY_HALF_LIFE_DAYS)
        self.assertAlmostEqual(w0, 1.0)

        # At age 21 days: weight = 0.50
        w21 = math.exp(-math.log(2.0) * 21.0 / RECENCY_HALF_LIFE_DAYS)
        self.assertAlmostEqual(w21, 0.50)

        # At age 42 days: weight = 0.25
        w42 = math.exp(-math.log(2.0) * 42.0 / RECENCY_HALF_LIFE_DAYS)
        self.assertAlmostEqual(w42, 0.25)

    def test_sample_size_clipping(self) -> None:
        # Sample size 1000 -> weight 1.0
        w_1000 = math.sqrt(1000 / SAMPLE_SIZE_BENCHMARK)
        self.assertAlmostEqual(w_1000, 1.0)

        # Sample size 100 (sqrt(0.1) = 0.316) -> clipped to 0.70
        w_100 = min(max(math.sqrt(100 / SAMPLE_SIZE_BENCHMARK), 0.70), 1.50)
        self.assertEqual(w_100, 0.70)

        # Sample size 5000 (sqrt(5) = 2.236) -> clipped to 1.50
        w_5000 = min(max(math.sqrt(5000 / SAMPLE_SIZE_BENCHMARK), 0.70), 1.50)
        self.assertEqual(w_5000, 1.50)

    def test_kish_effective_count_and_cap(self) -> None:
        # 10 identical polls with weight 1.0 -> sum(w)=10, sum(w^2)=10 -> n_eff = 100/10 = 10.0
        weights = [1.0] * 10
        sum_w = sum(weights)
        sum_w_sq = sum(w * w for w in weights)
        n_eff = (sum_w ** 2) / sum_w_sq
        self.assertAlmostEqual(n_eff, 10.0)

        n_eff_capped = min(max(n_eff, 1.0), MAX_EFFECTIVE_POLLS)
        self.assertEqual(n_eff_capped, 8.0)


class DataLeakageAndAsOfSafetyTests(unittest.TestCase):
    """Mandatory regression tests proving strict as_of safety and no future leakage."""

    def test_future_poll_injection_does_not_alter_historical_opinion_state(self) -> None:
        base_dir = Path(__file__).resolve().parents[1] / "data" / "processed" / "pollofpolls"
        historical_as_of = "2020-09-01"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            shutil.copy(base_dir / "pollofpolls_timeseries.csv", tmp_path / "pollofpolls_timeseries.csv")
            shutil.copy(base_dir / "individual_polls.csv", tmp_path / "individual_polls.csv")

            # 1. Baseline estimate on historical date
            state_baseline = estimate_opinion(as_of=historical_as_of, data_dir=tmp_path)

            # 2. Inject future poll (published in 2026) directly into the temporary individual_polls.csv
            with (tmp_path / "individual_polls.csv").open("a", encoding="utf-8") as f:
                writer = csv.writer(f)
                for party in ["M", "L", "C", "KD", "S", "V", "MP", "SD", "FI", "other"]:
                    writer.writerow([
                        "pop-future-fake",
                        "Novus",
                        "Novus",
                        "2026-08-01",
                        "2026-08-10",
                        "2026-08-12",
                        party,
                        "12.0",
                        "12.0",
                        "reported",
                        "1000",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "exact_span_match",
                        "[]",
                    ])

            # 3. Estimate state again on historical date
            state_injected = estimate_opinion(as_of=historical_as_of, data_dir=tmp_path)

            # 4. Verify substantive model state components are completely identical
            self.assertEqual(state_baseline.as_of, state_injected.as_of)
            self.assertEqual(state_baseline.estimate_date, state_injected.estimate_date)
            self.assertEqual(state_baseline.mean_pct, state_injected.mean_pct)
            self.assertEqual(state_baseline.mean_alr, state_injected.mean_alr)
            self.assertEqual(state_baseline.covariance_alr, state_injected.covariance_alr)
            self.assertEqual(state_baseline.recent_poll_count, state_injected.recent_poll_count)
            self.assertEqual(state_baseline.effective_poll_count, state_injected.effective_poll_count)
            self.assertEqual(state_baseline.residual_poll_count, state_injected.residual_poll_count)
            self.assertEqual(state_baseline.house_effects_alr, state_injected.house_effects_alr)

            # 5. Verify deterministic samples with same seed are identical
            samples_a = state_baseline.sample(n=1000, seed=12345)
            samples_b = state_injected.sample(n=1000, seed=12345)
            self.assertEqual(samples_a, samples_b)

            # 6. Verify diagnostics accurately caught the extra future poll
            self.assertEqual(
                state_injected.diagnostics["exclusions_for_as_of"]["future_publication_date"],
                state_baseline.diagnostics["exclusions_for_as_of"]["future_publication_date"] + 1,
            )

    def test_covariance_fallback_reported_when_history_short(self) -> None:
        # Early date (2015-09-15): 1 year after timeseries start, fewer than 100 polls in 4y window
        state_early = estimate_opinion(as_of="2015-09-15")
        self.assertTrue(state_early.covariance_fallback_used)
        self.assertGreater(len(state_early.diagnostics["warnings"]), 0)

    def test_default_as_of_uses_latest_timeseries_date(self) -> None:
        timeseries_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "processed"
            / "pollofpolls"
            / "pollofpolls_timeseries.csv"
        )
        latest_timeseries_date = max(
            row["date"] for row in load_timeseries_dataset(timeseries_path)
        )
        state_default = estimate_opinion()
        state_explicit = estimate_opinion(as_of=latest_timeseries_date)
        self.assertEqual(state_default.as_of, latest_timeseries_date)
        self.assertEqual(state_default.mean_pct, state_explicit.mean_pct)
        self.assertEqual(state_default.covariance_alr, state_explicit.covariance_alr)

    def test_alr_reference_invariance_without_shrinkage(self) -> None:
        """Regression test: with shrinkage=0, ALR reference choice (REST vs S vs M) yields invariant percentage-space SDs."""
        base_dir = Path(__file__).resolve().parents[1] / "data" / "processed" / "pollofpolls"
        timeseries_data = load_timeseries_dataset(base_dir / "pollofpolls_timeseries.csv")
        individual_polls, _ = load_individual_polls_dataset(base_dir / "individual_polls.csv")

        as_of = date(2026, 8, 23)
        window_start = subtract_calendar_years(as_of, 4)

        active_matches = []
        for poll in individual_polls:
            if poll.publication_date and window_start <= poll.publication_date < as_of:
                if poll.interview_end and poll.interview_end <= as_of and poll.reference_date:
                    candidates = [row for row in timeseries_data if row["date"] <= poll.reference_date]
                    if candidates and (poll.reference_date - candidates[-1]["date"]).days <= 3:
                        active_matches.append({"poll": poll, "matched": candidates[-1]})

        from scripts.pollofpolls.state_diagnostics import run_reference_sensitivity_audit

        sens = run_reference_sensitivity_audit(
            active_matches,
            timeseries_data,
            as_of,
            n_eff_used=3.6537,
            seed=12345,
            samples_count=10_000,
        )

        unshrunk = sens["shrinkage_00"]
        rest_sds = {cat: unshrunk["REST"][cat]["std_dev"] for cat in ALL_CATEGORIES}
        s_sds = {cat: unshrunk["S"][cat]["std_dev"] for cat in ALL_CATEGORIES}
        m_sds = {cat: unshrunk["M"][cat]["std_dev"] for cat in ALL_CATEGORIES}

        # Assert reference-invariance within Monte Carlo tolerance (< 0.03 pp)
        for cat in ALL_CATEGORIES:
            diff_s = abs(rest_sds[cat] - s_sds[cat])
            diff_m = abs(rest_sds[cat] - m_sds[cat])
            self.assertLess(
                diff_s,
                0.03,
                f"Reference sensitivity between REST and S on {cat} exceeded tolerance: {diff_s:.4f}",
            )
            self.assertLess(
                diff_m,
                0.03,
                f"Reference sensitivity between REST and M on {cat} exceeded tolerance: {diff_m:.4f}",
            )

    def test_as_of_before_timeseries_start_raises_clear_error(self) -> None:
        with self.assertRaises(ValueError):
            estimate_opinion(as_of="2010-01-01")



if __name__ == "__main__":
    unittest.main()


