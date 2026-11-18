"""T4 candidate path and harness (calibration-2030 final round §5, §6).

The C0 self-check runs production ``simulate_election`` and the candidate
path at n = 2,000 on one 2018 and one 2026 case and requires identical bytes.
It is checked through the real ``noise_laws`` LW0 and, independently,
through a local stub that copies production's law. Amendment 002 cases
(2014 on ``isolated.votes_to_seats``, 2010 vote-only) and a small CAND draw
through the real ``t2_arms`` and ``noise_laws`` are checked at small n.
"""

from __future__ import annotations

import hashlib
from fractions import Fraction
import json
import shutil
import sys
import tempfile
import types
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import numpy as np

from diagnostics.calibration_2030 import candidate_engine as ce
from diagnostics.calibration_2030 import t4_replay as t4

N = 2_000
SEED = 20260929


def _stub_noise_laws() -> types.ModuleType:
    """LW0 exactly as production draws pp_lw_gaussian, on production's pool."""

    from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals
    from scripts.vote_share_calibration.election_noise_b import fit_election_noise_b, symmetric_factor

    def current_rule_pool(years):
        pool = load_chronological_pp_residuals(target_election_year=max(years) + 1)
        assert list(pool.training_years) == list(years)
        return {"residuals": pool.residuals_matrix, "centered": pool.centered_residuals_matrix,
                "consensus": None, "unreported": None, "years": list(pool.training_years)}

    class LW0:
        def __init__(self, pool):
            self.factor = symmetric_factor(fit_election_noise_b(pool["centered"]).sigma_tilde)

        def draw(self, z, base):
            return z @ self.factor.T

    def fit_law(name, pool):
        assert name == "LW0"
        return LW0(pool)

    module = types.ModuleType(ce.NOISE_LAWS_MODULE)
    module.current_rule_pool = current_rule_pool
    module.corrected_pool = current_rule_pool
    module.fit_law = fit_law
    return module


class C0SelfCheckTest(unittest.TestCase):
    """§6.4: configured as C0, the candidate path reproduces production's bytes."""

    @classmethod
    def setUpClass(cls):
        from scripts.simulator.engine import simulate_election

        cls.reference = {}
        for year, horizon in t4.SELF_CHECK_CASES:
            election = t4.ELECTION_DAYS[year]
            cls.reference[year] = (election, election - timedelta(days=horizon))
            ref = simulate_election(as_of=election - timedelta(days=horizon), election_date=election,
                                    samples=N, seed=SEED)
            cls.reference[year] += (ref,)

    def _assert_identical(self, config):
        for year, (election, as_of, ref) in self.reference.items():
            ce._POOL_CACHE.clear()
            got, national = ce.simulate_candidate(config, as_of=as_of, election_date=election, samples=N, seed=SEED)
            self.assertEqual(ref.vote_shares_matrix.tobytes(), got.vote_shares_matrix.tobytes(), year)
            self.assertEqual(ref.seats_matrix.tobytes(), got.seats_matrix.tobytes(), year)
            self.assertEqual(ref.summary.as_of, got.summary.as_of)

    def test_c0_with_production_noise_code(self):
        # Without noise_laws, C0's national stage is production's own result;
        # checked at the national stage (the downstream is covered below).
        from scripts.vote_share_calibration.national_engine import generate_national_vote_shares

        config = ce.config_c0(use_noise_laws=False)
        for year, (election, as_of, _) in self.reference.items():
            ref = generate_national_vote_shares(as_of=as_of, election_date=election, samples=N, seed=SEED)
            got = ce.candidate_national_vote_shares(config, as_of=as_of, election_date=election, samples=N, seed=SEED)
            self.assertEqual(ref.nat_shares_matrix.tobytes(), got.nat_shares_matrix.tobytes(), year)

    def test_c0_through_the_pluggable_noise_path(self):
        # LW0 on production's z must give production's bytes; this exercises
        # the generic (law.draw -> transfer -> renormalize) path end to end.
        with mock.patch.dict(sys.modules, {ce.NOISE_LAWS_MODULE: _stub_noise_laws()}):
            self._assert_identical(ce.config_c0(use_noise_laws=True))
        ce._POOL_CACHE.clear()

    def test_c0_with_the_real_noise_laws_module(self):
        config = ce.config_c0()
        self.assertEqual((config.noise, config.pool), ("LW0", "current_rule"))
        self._assert_identical(config)


class SubSeedTest(unittest.TestCase):
    def test_new_layer_seed_matches_the_frozen_formula(self):
        origin, horizon = date(2018, 8, 12), 28
        for layer in ("state", "dynamics", "bootstrap", "ou"):
            token = f"20260929:2018-08-12:28:calibration-2030:{layer}".encode()
            expected = int(hashlib.sha256(token).hexdigest()[:8], 16)
            self.assertEqual(ce.layer_seed(SEED, origin, horizon, layer), expected)
        seeds = {ce.layer_seed(SEED, origin, horizon, layer) for layer in ce.NEW_LAYERS}
        self.assertEqual(len(seeds), 4)
        with self.assertRaises(ValueError):
            ce.layer_seed(SEED, origin, horizon, "noise")

    def test_one_generator_per_layer(self):
        origin, horizon = date(2026, 9, 6), 7
        rngs = ce.layer_rngs(SEED, origin, horizon)
        for layer, rng in rngs.items():
            expected = np.random.default_rng(ce.layer_seed(SEED, origin, horizon, layer)).standard_normal(5)
            np.testing.assert_array_equal(rng.standard_normal(5), expected)

    def test_noise_z_is_productions(self):
        from scripts.vote_share_calibration.election_noise_b import derive_election_noise_b_seed

        z = ce.production_noise_z(SEED, date(2022, 7, 31), 42, 10)
        rng = np.random.default_rng(derive_election_noise_b_seed(SEED, date(2022, 7, 31), 42))
        np.testing.assert_array_equal(z, rng.standard_normal((10, 9)))

    def test_u_arm_receives_layer_draws(self):
        calls = {}

        def state_dynamics_draws(arm, ctx, origin, target, *, z_state, z_dyn, boot_rng, d0_base_seed, k=None):
            calls.update(arm=arm, origin=origin, target=target, z_state=z_state.copy(), z_dyn=z_dyn.copy(),
                         boot=boot_rng.integers(0, 2**31, 3), seed=d0_base_seed, k=k)
            base = np.tile([20.0, 5.0, 6.0, 5.0, 30.0, 7.0, 5.0, 18.0, 4.0], (z_state.shape[0], 1))
            return base

        t2 = types.ModuleType(ce.T2_ARMS_MODULE)
        t2.t2_context = lambda: {"stub": True}
        t2.state_dynamics_draws = state_dynamics_draws
        t2.inflation_k = lambda arm, ctx, origin: (1.25, 21)
        config = ce.CandidateConfig("U", aggregate="v0.3", state_dynamics="Ub1", noise="LW0", pool="corrected",
                                    simplex_map=True, t2_arm="Ub1")
        election, n = date(2018, 9, 9), 50
        as_of = election - timedelta(days=28)
        with mock.patch.dict(sys.modules, {ce.T2_ARMS_MODULE: t2, ce.NOISE_LAWS_MODULE: _stub_noise_laws()}), \
                mock.patch.object(ce, "_T2_CTX", None):
            ce._POOL_CACHE.clear()
            national = ce.candidate_national_vote_shares(config, as_of=as_of, election_date=election,
                                                         samples=n, seed=SEED)
        ce._POOL_CACHE.clear()
        rngs = ce.layer_rngs(SEED, as_of, 28)
        np.testing.assert_array_equal(calls["z_state"], rngs["state"].standard_normal((n, 8)))
        np.testing.assert_array_equal(calls["z_dyn"], rngs["dynamics"].standard_normal((n, 8)))
        np.testing.assert_array_equal(calls["boot"], rngs["bootstrap"].integers(0, 2**31, 3))
        self.assertEqual((calls["arm"], calls["origin"], calls["target"], calls["seed"], calls["k"]),
                         ("Ub1", as_of, election, SEED, 1.25))
        self.assertEqual(national.diagnostics["k_eligible_targets"], 21)
        np.testing.assert_allclose(national.nat_shares_matrix.sum(axis=1), 1.0)


class SimplexMapTest(unittest.TestCase):
    def test_map(self):
        x = np.array([[50.0, 0.0, 10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 999.0]])
        out = ce.simplex_map(x)
        self.assertAlmostEqual(out.sum(), 100.0, places=12)
        expected = np.array([50.0, 0.01, 10, 10, 10, 10, 5, 5, 0.01])
        np.testing.assert_allclose(out[0], expected / expected.sum() * 100.0)


class SubstituteRowsTest(unittest.TestCase):
    """§6.3 rows are causal: rows before X do not change when data after X are dropped."""

    @classmethod
    def setUpClass(cls):
        from diagnostics.calibration_2030.v03 import v03_context

        cls.ctx = v03_context()

    def test_causal(self):
        cut = date(2008, 3, 1)
        truncated = dict(self.ctx)
        truncated["polls"] = [o for o in self.ctx["polls"] if o.available <= cut]
        truncated["everything"] = [o for o in self.ctx["everything"] if o.available <= cut]
        for kind in ("v0.2", "v0.3"):
            full = t4.substitute_rows(kind, self.ctx)
            short = t4.substitute_rows(kind, truncated, end=cut)
            self.assertEqual(min(full), t4.SUBSTITUTE_START)
            self.assertEqual(max(full), t4.SUBSTITUTE_END)
            days = [d for d in full if d < cut]
            self.assertGreater(len(days), 400)
            for d in days:
                self.assertEqual(full[d][1], short[d][1], f"{kind} {d}")
        info = t4.information_dates(self.ctx["everything"], list(full), [t4.SUBSTITUTE_START], t4.SUBSTITUTE_END)
        self.assertTrue(all(i <= d for d, i in info.items()))

    def test_sh_starts_have_no_effect_before_2010(self):
        # §6.3: no non-anchor election is known before 2010-09-26, so the SH
        # injections never fire and the two substitutions coincide.
        v02 = t4.substitute_rows("v0.2", self.ctx, end=date(2007, 6, 1))
        v03 = t4.substitute_rows("v0.3", self.ctx, end=date(2007, 6, 1))
        self.assertEqual(v02, v03)


def _synthetic_rows(values: dict) -> list[dict]:
    """Rows for C0, F1, CAND on D and L (2014–2022) and 2026 D; values[arm] -> metric dict."""

    rows = []
    for arm, metrics in values.items():
        for year in (2014, 2018, 2022, 2026):
            for set_name in ("D", "L"):
                for h in t4.HORIZONS[set_name]:
                    r = {"arm": arm, "year": year, "set": set_name, "horizon_days": h, "completed": True,
                         "seat_total_always_349": True, "all_finite": True, "dates_ok": True}
                    r.update(metrics(year, set_name) if callable(metrics) else metrics)
                    rows.append(r)
    return rows


BASE = {"es_9cat": 4.0, "seat_energy_score": 10.0, "brier_coalition": 0.10, "brier_threshold": 0.02,
        "coverage_90": 0.78, "coverage_50": 0.40}


class GateArithmeticTest(unittest.TestCase):
    def test_all_pass(self):
        cand = dict(BASE, es_9cat=4.1, seat_energy_score=10.4, brier_coalition=0.105, coverage_90=0.86,
                    coverage_50=0.5)
        rows = _synthetic_rows({"C0": BASE, "F1": dict(BASE, es_9cat=3.95), "CAND": cand})
        out = t4.evaluate_gates(rows, "CAND", dynamics_changed=True, provenance_ok=True)
        self.assertTrue(out["pass"], out["gates"])
        self.assertFalse(out["flag_2026"]["flagged"])

    def test_each_gate_fails_on_its_margin(self):
        ok = dict(BASE, coverage_90=0.86, coverage_50=0.5)
        cases = {
            "G1_accuracy": {"F1": dict(BASE, es_9cat=3.8), "CAND": dict(ok, es_9cat=4.0)},   # > 1.05 x F1
            "G3_seats": {"CAND": dict(ok, brier_coalition=0.111)},                             # > C0 + 0.01
            "G4_thresholds": {"CAND": dict(ok, brier_threshold=0.0301)},
            "G5_calibration": {"CAND": dict(ok, coverage_90=0.975)},                           # outside [0.80, 0.97]
        }
        for gate, override in cases.items():
            values = {"C0": BASE, "F1": BASE, "CAND": ok}
            values.update(override)
            out = t4.evaluate_gates(_synthetic_rows(values), "CAND", dynamics_changed=False, provenance_ok=True)
            failing = [g for g, v in out["gates"].items() if not v["pass"]]
            self.assertEqual(failing, [gate], (gate, out["gates"]))

    def test_g2_per_election_and_2026_flag(self):
        def cand(year, set_name):
            return dict(BASE, coverage_90=0.86, coverage_50=0.5,
                        es_9cat={2014: 4.0, 2018: 4.7, 2022: 3.2, 2026: 4.7}[year])  # 2018: 1.175 x C0

        out = t4.evaluate_gates(_synthetic_rows({"C0": BASE, "F1": BASE, "CAND": cand}), "CAND",
                                dynamics_changed=False, provenance_ok=True)
        self.assertFalse(out["gates"]["G2_per_election"]["elections"]["2018"]["pass"])
        self.assertTrue(out["gates"]["G2_per_election"]["elections"]["2022"]["pass"])
        self.assertTrue(out["gates"]["G1_accuracy"]["pass"])  # pooled 3.967
        self.assertTrue(out["flag_2026"]["flagged"])           # 4.7 > 1.15 x 4.0

    def test_g7_calibration_clause_only_with_dynamics_change(self):
        def cand(year, set_name):
            return dict(BASE, coverage_90=0.86 if set_name == "D" else 0.70, coverage_50=0.5)

        rows = _synthetic_rows({"C0": BASE, "F1": BASE, "CAND": cand})
        self.assertTrue(t4.evaluate_gates(rows, "CAND", dynamics_changed=False, provenance_ok=True)["pass"])
        out = t4.evaluate_gates(rows, "CAND", dynamics_changed=True, provenance_ok=True)
        self.assertFalse(out["gates"]["G7_long_horizon"]["pass"])  # |0.70-0.9| >= |0.78-0.9|

    def test_g6_integrity(self):
        ok = dict(BASE, coverage_90=0.86, coverage_50=0.5)
        rows = _synthetic_rows({"C0": BASE, "F1": BASE, "CAND": ok})
        self.assertFalse(t4.evaluate_gates(rows, "CAND", dynamics_changed=False, provenance_ok=False)["pass"])
        dropped = [r for r in rows if not (r["arm"] == "F1" and r["year"] == 2014 and r["horizon_days"] == 730)]
        out = t4.evaluate_gates(dropped, "CAND", dynamics_changed=False, provenance_ok=True)
        self.assertEqual(out["gates"]["G6_integrity"]["arms"]["F1"]["missing_cases"], ["2014/L730"])
        self.assertFalse(out["gates"]["G6_integrity"]["pass"])
        rows[0]["dates_ok"] = False
        self.assertFalse(t4.evaluate_gates(rows, "CAND", dynamics_changed=False,
                                           provenance_ok=True)["gates"]["G6_integrity"]["pass"])


class SelectionTest(unittest.TestCase):
    def test_configs(self):
        self.assertTrue(ce.cand_equals_fb({"t2_arm": "D0", "noise_law": "N0"}))
        self.assertFalse(ce.cand_equals_fb({"t2_arm": "Ua", "noise_law": "N0"}))
        cand = ce.config_cand({"t2_arm": "D0", "noise_law": "N0"})
        fb = ce.config_fb()
        self.assertEqual({**cand.to_dict(), "name": "FB"}, fb.to_dict())
        self.assertTrue(ce.config_cand({"t2_arm": "Uc", "noise_law": "NH1"}).dynamics_differ_from_d0)
        with self.assertRaises(ValueError):
            ce.config_cand({"t2_arm": "D0", "noise_law": "LW0"})

    def test_selection_from_the_results_records(self):
        selection, source = t4.load_selection(None)
        self.assertEqual(selection, {"t2_arm": "Ua", "noise_law": "NH0"})
        self.assertIn("t2_results_sha256", source)
        cand = ce.config_cand(selection)
        self.assertEqual((cand.aggregate, cand.state_dynamics, cand.noise, cand.pool), ("v0.3", "Ua", "NH0", "corrected"))
        self.assertFalse(ce.cand_equals_fb(selection))


class Amendment002Test(unittest.TestCase):
    """2014 seats on the validated PRE_2018 path; 2010 votes only; pinned hashes enforced."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.inputs = t4.prepare_inputs(Path(cls.tmp.name), need_v03=True, need_2010=True)
        cls.truth = t4.load_truth_t4()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_amendment_verifies(self):
        record = t4.verify_amendment_002()
        self.assertIn("diagnostics/election_noise_v2/control_baseline_amendment2/harness2/isolated.py",
                      record["pinned_sha256"])

    def test_pinned_hash_mismatch_refuses(self):
        from diagnostics.calibration_2030.common import IntegrityError

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            record = json.loads(t4.AMENDMENT_002.read_text(encoding="utf-8"))
            record["pinned_sha256"]["scripts/mandates/law.py"] = "0" * 64
            body = json.dumps(record, sort_keys=True).encode()
            digest = hashlib.sha256(body).hexdigest()
            (folder / t4.AMENDMENT_002.name).write_bytes(body)
            (folder / t4.AMENDMENT_002.with_suffix(".sha256").name).write_text(f"{digest}  x\n")
            index = {"amendments": [{"record": t4.AMENDMENT_002.name, "record_sha256": digest}]}
            (folder / "index.json").write_text(json.dumps(index))
            with self.assertRaisesRegex(IntegrityError, "pinned files changed"):
                t4.verify_amendment_002(folder / t4.AMENDMENT_002.name)
            (folder / t4.AMENDMENT_002.with_suffix(".sha256").name).write_text("f" * 64 + "  x\n")
            with self.assertRaisesRegex(IntegrityError, "companion"):
                t4.verify_amendment_002(folder / t4.AMENDMENT_002.name)
        tampered = Path(self.tmp.name) / "amend"
        shutil.copytree(t4.AMENDMENT_002.parent, tampered)
        (tampered / t4.AMENDMENT_002.name).write_bytes(t4.AMENDMENT_002.read_bytes() + b" ")
        with self.assertRaises(IntegrityError):
            t4.verify_amendment_002(tampered / t4.AMENDMENT_002.name)

    def _job(self, arm, config, year, horizon, set_name, n=40, t2_values=None):
        (job,) = t4.make_jobs(arm, config, [(year, horizon, set_name)], self.inputs, self.truth, n, SEED, t2_values)
        return job

    def test_2014_seats_come_from_votes_to_seats_under_pre_2018(self):
        from diagnostics.election_noise_v2.control_baseline_amendment2.harness2 import isolated
        from scripts.mandates.law import MandateLaw, mandate_law_for_election_year

        self.assertIs(mandate_law_for_election_year(2014).law, MandateLaw.PRE_2018)
        self.assertEqual(mandate_law_for_election_year(2014).first_divisor, Fraction(7, 5))
        calls = []
        original = isolated.votes_to_seats

        def spy(votes, year, *args, **kwargs):
            calls.append((votes.copy(), year))
            return original(votes, year, *args, **kwargs)

        config = ce.config_c0()
        with mock.patch.object(isolated, "votes_to_seats", spy):
            row = t4.run_case(self._job("C0", config, 2014, 28, "D"))
        self.assertTrue(row["completed"], row.get("error"))
        self.assertEqual([y for _, y in calls], [2014])
        national = ce.candidate_national_vote_shares(config, as_of=date(2014, 8, 17), election_date=date(2014, 9, 14),
                                                     samples=40, seed=SEED)
        np.testing.assert_array_equal(calls[0][0], national.nat_shares_matrix * 100.0)
        seats = original(calls[0][0], 2014)
        self.assertEqual(row["mean_seats"], "|".join(f"{v:.3f}" for v in seats.mean(axis=0)))
        self.assertTrue(row["seat_total_always_349"] and row["dates_ok"])
        self.assertEqual(sum(self.truth[2014]["seats"].values()), 349)
        self.assertIn("seat_energy_score", row)
        self.assertIn("brier_coalition", row)

    def test_2010_has_vote_metrics_only(self):
        row = t4.run_case(self._job("C0", ce.config_c0(), 2010, 112, "D"))
        self.assertTrue(row["completed"], row.get("error"))
        for key in ("es_9cat", "crps_8party_mean", "brier_threshold", "coverage_50", "coverage_90"):
            self.assertIn(key, row)
        for key in ("seat_energy_score", "brier_coalition", "mean_seats"):
            self.assertNotIn(key, row)
        self.assertIsNone(row["seat_total_always_349"])
        self.assertEqual(row["noise_pool_years"], [2002, 2006])

    def test_u_arm_before_2010_09_26_is_not_computable(self):
        cand = ce.config_cand({"t2_arm": "Ua", "noise_law": "NH0"})
        values = t4.t2_origin_values(cand, [date(2010, 5, 30)])
        self.assertIn("not_computable", values["2010-05-30"])
        row = t4.run_case(self._job("CAND", cand, 2010, 112, "D", t2_values=values))
        self.assertEqual((row["completed"], row["status"]), (False, "not computable"))

    def test_cand_draw_through_the_real_modules(self):
        # k is passed in (inflation_k at a 2018 origin scores every earlier
        # k = 1 cell, minutes of work); the harness computes it with
        # t2_origin_values. Everything else is the real T2 and T3-rev code.
        cand = ce.config_cand({"t2_arm": "Ua", "noise_law": "NH0"})
        values = {"2018-08-12": {"t2_k": [0.5, 999]}}
        for year, horizon, set_name in ((2018, 28, "D"), (2014, 28, "D")):
            if year == 2014:
                values = {"2014-08-17": {"t2_k": [0.5, 999]}}
            row = t4.run_case(self._job("CAND", cand, year, horizon, set_name, n=60, t2_values=values))
            self.assertTrue(row["completed"], row.get("error"))
            self.assertTrue(row["vote_rows_sum_to_100"] and row["seat_total_always_349"], row)
            self.assertEqual((row["k"], row["k_eligible_targets"]), (0.5, 999))
            self.assertTrue(row["dates_ok"])

if __name__ == "__main__":
    unittest.main()
