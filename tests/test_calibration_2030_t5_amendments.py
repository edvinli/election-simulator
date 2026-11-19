"""T5 amendments 001 (shadow arm Ul) and 002 (shadow arm U0)."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np

from diagnostics.calibration_2030 import t2_arms as A
from diagnostics.calibration_2030 import t5_shadow as T
from diagnostics.calibration_2030 import ul_arm as U

M = np.array([18.0, 4.0, 5.0, 5.0, 34.0, 7.0, 5.0, 20.0])


class LogRatioMapTest(unittest.TestCase):
    def test_compositions_are_positive_and_sum_to_100(self):
        y = M + 10.0 * np.random.default_rng(1).standard_normal((5000, 8))
        x = U.log_ratio_map(y, M)
        self.assertEqual(x.shape, (5000, 9))
        self.assertTrue((x > 0).all())
        np.testing.assert_allclose(x.sum(axis=1), 100.0, atol=1e-9)

    def test_zero_deviation_returns_the_reading(self):
        x = U.log_ratio_map(M[None, :], M)
        np.testing.assert_allclose(x[0], [*M, 100.0 - M.sum()], atol=1e-12)

    def test_first_order_equals_ua_and_the_gap_is_second_order(self):
        z = np.random.default_rng(2).standard_normal((200, 8))
        gaps = []
        for scale in (0.01, 0.001):
            y = M + scale * z
            gaps.append(np.abs(U.log_ratio_map(y, M) - A.simplex_map(y)).max())
        self.assertLess(gaps[0], 0.01)
        self.assertAlmostEqual(gaps[0] / gaps[1], 100.0, delta=10.0)

    def test_floor_bounds_a_tiny_category(self):
        m = M.copy()
        m[4] += 1.9  # REST = 0.1 pp
        y = m + np.array([[0, 0, 0, 0, 0, 0, 0, -1.0]])  # REST deviation +1 pp
        x = U.log_ratio_map(y, m)
        expected_rest = 0.1 * np.exp(1.0 / U.FLOOR_PP)
        raw = np.concatenate([m * np.exp(np.r_[np.zeros(7), -1.0] / np.maximum(m, 0.5)), [expected_rest]])
        np.testing.assert_allclose(x[0], 100.0 * raw / raw.sum(), rtol=1e-12)


class AmendmentRecordsTest(unittest.TestCase):
    def test_both_amendments_are_in_force_with_their_arms(self):
        in_force = T.t5_amendments_in_force()
        arms = {a["amendment_id"]: a["arm"] for a in in_force}
        self.assertEqual(arms, {"001-ul-log-ratio-arm": "Ul", "002-u0-shadow-arm": "U0", "003-operations-ci": None})

    def test_activation_by_origin(self):
        in_force = [{"amendment_id": "001-ul-log-ratio-arm", "arm": "Ul", "in_effect_from": "2026-11-19"},
                    {"amendment_id": "002-u0-shadow-arm", "arm": "U0", "in_effect_from": "2026-11-19"},
                    {"amendment_id": "999-other", "arm": "X", "in_effect_from": "2026-01-01"}]
        self.assertEqual(T.amendment_arms_active(date(2026, 11, 16), in_force), {})
        self.assertEqual(set(T.amendment_arms_active(date(2026, 11, 23), in_force)), {"Ul", "U0"})

    def test_a_tampered_record_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for f in T.T5_AMENDMENT_DIR.iterdir():
                (d / f.name).write_bytes(f.read_bytes())
            rec = d / "001-ul-log-ratio-arm.json"
            rec.write_bytes(rec.read_bytes().replace(b'"Ul"', b'"Ux"', 1))
            with self.assertRaises(T.IntegrityError):
                T.t5_amendments_in_force(d)

    def test_u0_label_from_the_committed_exception_results(self):
        label = T.u0_label()
        self.assertEqual(label["verdict"], "T4X_FAIL")
        self.assertEqual(label["failed_gates"], ["G5_calibration"])
        self.assertFalse(label["flag_2026_raised"])
        record = json.loads((T.T5_AMENDMENT_DIR / "002-u0-shadow-arm.json").read_text(encoding="utf-8"))
        self.assertEqual(record["arm"]["label"]["t4_exception_results_sha256"],
                         label["t4_exception_results_sha256"])

    def test_u0_config_is_the_amendment_003_arm(self):
        from diagnostics.calibration_2030.t4_exception_u0 import config_u0

        cfg = config_u0().to_dict()
        record = json.loads((T.T5_AMENDMENT_DIR / "002-u0-shadow-arm.json").read_text(encoding="utf-8"))
        for key in ("aggregate", "state_dynamics", "noise", "pool"):
            self.assertEqual(cfg[key], record["arm"]["config"][key])


class UlOnContextTest(unittest.TestCase):
    """On T2's pinned context: Ul's pre-map draw is Ua's, and Ul's k is defined."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.ctx = A.t2_context(n_draws=64, v03_dir=Path(cls.tmp.name) / "v03")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_premap_reproduces_ua(self):
        origin, target = date(2026, 11, 8), date(2030, 9, 8)
        rng = np.random.default_rng(3)
        z_state, z_dyn = rng.standard_normal((64, 8)), rng.standard_normal((64, 8))
        ua = A.state_dynamics_draws("Ua", self.ctx, origin, target, z_state=z_state, z_dyn=z_dyn,
                                    boot_rng=None, d0_base_seed=None, k=0.5)
        y, m = U.ua_premap(self.ctx, origin, target, z_state=z_state, z_dyn=z_dyn, k=0.5)
        np.testing.assert_array_equal(A.simplex_map(y), ua)
        ul = U.ul_draws(self.ctx, origin, target, z_state=z_state, z_dyn=z_dyn, k=0.5)
        self.assertTrue((ul > 0).all())
        np.testing.assert_allclose(ul.sum(axis=1), 100.0, atol=1e-9)


class GuardTest(unittest.TestCase):
    def test_runner_records_the_t5_amendments(self):
        p = T.verify_t5(allow_dirty=True)
        self.assertEqual([a["amendment_id"] for a in p["t5_amendments_in_force"]],
                         ["001-ul-log-ratio-arm", "002-u0-shadow-arm", "003-operations-ci"])


if __name__ == "__main__":
    unittest.main()
