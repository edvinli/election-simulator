"""Calibration amendment 003: the U0 exception arm harness."""

from __future__ import annotations

import json
import unittest

from diagnostics.calibration_2030 import t4_exception_u0 as u


class U0ExceptionHarness(unittest.TestCase):
    def test_u0_is_ua_with_unbiased_law_on_corrected_pool(self):
        cfg = u.config_u0().to_dict()
        self.assertEqual((cfg["aggregate"], cfg["state_dynamics"], cfg["noise"], cfg["pool"]),
                         ("v0.3", "Ua", "LW0", "corrected"))
        self.assertTrue(u.config_u0().dynamics_differ_from_d0)

    def test_amendment_003_is_in_effect_and_matches_the_arm(self):
        record = u.verify_amendment_003()
        self.assertEqual(len(record["record_sha256"]), 64)

    def test_reproduction_check_is_exact_on_the_recorded_rows(self):
        recorded = json.loads(u.RECORDED_T4.read_text(encoding="utf-8"))
        rows = [r for r in recorded["cases"] if r["arm"] in ("C0", "F1")]
        check = u.reproduction_check(rows)
        self.assertTrue(check["identical"])
        self.assertGreater(check["compared_values"], 0)

    def test_reproduction_check_reports_a_difference(self):
        recorded = json.loads(u.RECORDED_T4.read_text(encoding="utf-8"))
        rows = [dict(r) for r in recorded["cases"] if r["arm"] == "C0"][:1]
        rows[0]["es_9cat"] = rows[0]["es_9cat"] + 1e-9
        self.assertFalse(u.reproduction_check(rows)["identical"])


if __name__ == "__main__":
    unittest.main()
