"""Unit tests for the Opinion State Estimator v1 diagnostic and audit tools."""

from __future__ import annotations

import math
import unittest

from scripts.pollofpolls.state_diagnostics import (
    calculate_distribution_stats,
    calculate_pearson_correlation,
    covariance_to_correlation,
    run_full_audit,
)


class DiagnosticMathTests(unittest.TestCase):
    """Test mathematical helper functions in state_diagnostics."""

    def test_pearson_correlation_known_values(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pos = [2.0, 4.0, 6.0, 8.0, 10.0]
        y_neg = [10.0, 8.0, 6.0, 4.0, 2.0]
        y_orth = [1.0, -1.0, 1.0, -1.0, 0.0]

        self.assertAlmostEqual(calculate_pearson_correlation(x, y_pos), 1.0, places=7)
        self.assertAlmostEqual(calculate_pearson_correlation(x, y_neg), -1.0, places=7)

        # Length mismatch raises ValueError
        with self.assertRaises(ValueError):
            calculate_pearson_correlation(x, [1.0, 2.0])

    def test_covariance_to_correlation(self) -> None:
        cov = [
            [4.0, 1.0, 0.0],
            [1.0, 9.0, 2.0],
            [0.0, 2.0, 16.0],
        ]
        corr = covariance_to_correlation(cov)

        # Diagonal entries must be 1.0
        for i in range(3):
            self.assertAlmostEqual(corr[i][i], 1.0, places=7)

        # Off-diagonal checks
        self.assertAlmostEqual(corr[0][1], 1.0 / (2.0 * 3.0), places=7)
        self.assertAlmostEqual(corr[1][2], 2.0 / (3.0 * 4.0), places=7)
        self.assertAlmostEqual(corr[0][2], 0.0, places=7)

    def test_distribution_stats(self) -> None:
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        st = calculate_distribution_stats(vals)
        self.assertEqual(st["N"], 5.0)
        self.assertAlmostEqual(st["mean"], 3.0)
        self.assertAlmostEqual(st["median"], 3.0)
        self.assertAlmostEqual(st["min"], 1.0)
        self.assertAlmostEqual(st["max"], 5.0)
        self.assertAlmostEqual(st["mad"], 1.0)
        # Sample variance = ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 4 = (4 + 1 + 0 + 1 + 4)/4 = 2.5
        self.assertAlmostEqual(st["std_dev"], math.sqrt(2.5), places=7)


class FullAuditIntegrationTests(unittest.TestCase):
    """Test full audit execution."""

    def test_run_full_audit_smoke_test(self) -> None:
        report = run_full_audit(as_of="2026-08-23")
        expected_sections = (
            "as_of",
            "reconstruction_report",
            "pp_stats_raw",
            "pp_stats_adjusted",
            "top_20_extreme_polls",
            "rest_distribution",
            "rest_alr_correlations",
            "alr_correlation_summary",
            "reference_sensitivity",
            "modeling_step_breakdown",
            "sampling_error_ratios",
            "time_stability",
            "pollster_audit",
        )
        for section in expected_sections:
            self.assertIn(section, report)

        self.assertEqual(report["reconstruction_report"]["residual_polls_count"], 190)
        self.assertEqual(len(report["top_20_extreme_polls"]), 20)


if __name__ == "__main__":
    unittest.main()
