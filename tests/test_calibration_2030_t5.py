"""Fast tests for the T5 prospective shadow runner (diagnostics/calibration_2030/t5_shadow.py).

Git behaviour is exercised in temporary repositories; nothing touches the
real repository's branches, production archive or ``data/processed/shadow_2030``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from diagnostics.calibration_2030 import t5_shadow as T

UTC = timezone.utc


def run_git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def init_repo(path: Path, branch: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    run_git(path, "init", "--quiet", "-b", branch)
    run_git(path, "config", "user.email", "t5-test@example.invalid")
    run_git(path, "config", "user.name", "T5 test")
    run_git(path, "config", "commit.gpgsign", "false")
    (path / "README").write_text("x\n", encoding="utf-8")
    run_git(path, "add", "README")
    run_git(path, "commit", "--quiet", "-m", "init")
    return path


def commit_file(repo: Path, rel: str, data: bytes, message: str, when: str | None = None) -> str:
    target = repo / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    run_git(repo, "add", rel)
    env = None
    if when:
        import os
        env = {**os.environ, "GIT_COMMITTER_DATE": when, "GIT_AUTHOR_DATE": when}
    subprocess.run(["git", "commit", "--quiet", "-m", message], cwd=repo, check=True, capture_output=True, env=env)
    return run_git(repo, "rev-parse", "HEAD")


def fixed_clock(moment: datetime):
    return lambda: moment


SAFE = datetime(2026, 11, 25, 10, 0, tzinfo=UTC)  # a Wednesday morning, outside 18:30–22:00


class SeedTest(unittest.TestCase):
    def test_seed_strings_and_integers(self):
        o = date(2026, 11, 23)
        self.assertEqual(T.seed_string(o, "h30"), "calibration-2030:T5:2026-11-23:h30")
        self.assertEqual(T.seed_string(o, "election"), "calibration-2030:T5:2026-11-23:election")
        self.assertEqual(T.seed_string(o, "engine"), "calibration-2030:T5:2026-11-23:engine")
        for target in (*T.TARGET_NAMES, "engine"):
            text = f"calibration-2030:T5:2026-11-23:{target}"
            self.assertEqual(T.t5_seed(o, target), int(hashlib.sha256(text.encode()).hexdigest()[:8], 16))
        self.assertEqual(T.TARGET_NAMES, ("h30", "h91", "h182", "h365", "h730", "election"))
        with self.assertRaises(ValueError):
            T.seed_string(o, "h7")
        self.assertNotEqual(T.t5_seed(o, "h30"), T.t5_seed(o + timedelta(days=7), "h30"))

    def test_target_dates(self):
        o = date(2026, 11, 23)
        d = T.target_dates(o)
        self.assertEqual(d["h30"], date(2026, 12, 23))
        self.assertEqual(d["h730"], date(2028, 11, 22))
        self.assertEqual(d["election"], date(2030, 9, 8))


class EncodingTest(unittest.TestCase):
    def test_npz_bytes_are_deterministic_and_match_the_sidecar_encoder(self):
        from scripts.simulator.exact_draw_sidecar import _npz_bytes

        rng = np.random.default_rng(3)
        votes = rng.uniform(0, 30, (50, 9))
        seats = rng.integers(0, 100, (50, 8)).astype("<i8")
        ours = T.npz_bytes([("vote_shares_pct", votes), ("seats", seats)])
        self.assertEqual(ours, _npz_bytes(votes, seats))
        self.assertEqual(ours, T.npz_bytes([("vote_shares_pct", votes.copy()), ("seats", seats.copy())]))
        q = rng.uniform(0, 30, (5, 99, 8)).astype(np.float32)
        c = T.to_centipp(rng.uniform(0, 40, (4000, 9)))
        a = T.npz_bytes([("election_draws_centipp", c), ("poll_quantiles", q)])
        self.assertEqual(a, T.npz_bytes([("election_draws_centipp", c), ("poll_quantiles", q)]))
        back = T.load_npz(a)
        np.testing.assert_array_equal(back["poll_quantiles"], q)
        np.testing.assert_array_equal(back["election_draws_centipp"], c)
        self.assertEqual(back["poll_quantiles"].dtype, np.float32)

    def test_no_pickle(self):
        with self.assertRaises(ValueError):
            T.npz_bytes([("bad", np.array([{"a": 1}], dtype=object))])

    def test_centipp_round_trip(self):
        exact = np.array([[0.0, 0.01, 12.34, 99.99, 100.0, 4.5, 33.33, 7.77, 0.05]])
        c = T.to_centipp(exact)
        self.assertEqual(c.dtype, np.uint16)
        np.testing.assert_array_equal(c, [[0, 1, 1234, 9999, 10000, 450, 3333, 777, 5]])
        np.testing.assert_allclose(T.from_centipp(c), exact, atol=1e-12)
        rng = np.random.default_rng(5)
        x = rng.uniform(0, 100, (1000, 9))
        np.testing.assert_array_less(np.abs(T.from_centipp(T.to_centipp(x)) - x), 0.005 + 1e-12)
        np.testing.assert_array_equal(T.to_centipp(np.array([-0.3, 100.7, 12.345, 12.355])), [0, 10000, 1234, 1236])

    def test_canonical_json(self):
        a = T.canonical_bytes({"b": np.float64(1.5), "a": [np.int64(2), date(2026, 1, 2)], "c": float("nan")})
        self.assertEqual(a, b'{"a":[2,"2026-01-02"],"b":1.5,"c":null}')


class QuantileCrpsTest(unittest.TestCase):
    def test_hand_computation(self):
        # τ = .25: (0 − .25)(1 − 2.5) = .375; τ = .5: (0 − .5)(2 − 2.5) = .25;
        # τ = .75: (1 − .75)(3 − 2.5) = .125; sum .75 × 2/3 = .5
        self.assertAlmostEqual(T.quantile_crps([1.0, 2.0, 3.0], 2.5, levels=[0.25, 0.5, 0.75]), 0.5, places=12)
        self.assertAlmostEqual(T.quantile_crps([1.0, 2.0, 3.0], 0.0, levels=[0.25, 0.5, 0.75]),
                               (2 / 3) * (0.75 * 1 + 0.5 * 2 + 0.25 * 3), places=12)

    def test_point_mass_gives_absolute_error(self):
        # Every quantile at c: (2/99) Σ (1{y<c} − τ)(c − y) = |c − y| because Σ τ = 49.5.
        q = np.full(99, 10.0)
        self.assertAlmostEqual(T.quantile_crps(q, 7.0), 3.0, places=12)
        self.assertAlmostEqual(T.quantile_crps(q, 12.5), 2.5, places=12)
        self.assertEqual(len(T.LEVELS), 99)
        self.assertAlmostEqual(float(T.LEVELS[0]), 0.01)
        self.assertAlmostEqual(float(T.LEVELS[-1]), 0.99)

    def test_close_to_draw_crps(self):
        from scripts.pollofpolls.backtest_metrics import calculate_crps

        draws = np.random.default_rng(1).normal(20.0, 1.5, 200_000)
        q = T.poll_quantiles(np.tile(draws[:, None], (1, 8)))[:, 0]
        self.assertAlmostEqual(T.quantile_crps(q, 21.0), calculate_crps(draws, 21.0), delta=0.03)

    def test_score_quantiles_coverage(self):
        q = np.tile(np.linspace(1, 99, 99)[:, None], (1, 8))  # q_τ = 100τ
        y = np.array([5.0, 25.0, 75.0, 95.0, 50.0, 4.9, 95.1, 24.9])
        s = T.score_quantiles(q, y)
        self.assertEqual(s["cov90"], [1, 1, 1, 1, 1, 0, 0, 1])
        self.assertEqual(s["cov50"], [0, 1, 1, 0, 1, 0, 0, 0])


class WeekAndClockTest(unittest.TestCase):
    def test_iso_weeks(self):
        self.assertEqual(T.iso_week_label(date(2026, 11, 23)), "2026-W48")
        self.assertEqual(T.iso_week_start(datetime(2026, 11, 25, 13, 0, tzinfo=UTC)),
                         datetime(2026, 11, 23, tzinfo=UTC))
        self.assertEqual(T.iso_week_start(datetime(2026, 11, 23, 0, 0, tzinfo=UTC)),
                         datetime(2026, 11, 23, tzinfo=UTC))
        self.assertEqual(T.weeks_between("2026-W48", "2026-W51"), ["2026-W49", "2026-W50"])
        self.assertEqual(T.weeks_between("2026-W52", "2027-W02"), ["2026-W53", "2027-W01"])
        self.assertEqual(T.weeks_between("2026-W48", "2026-W49"), [])

    def test_protected_interval(self):
        d = date(2026, 11, 25)
        inside = [datetime.combine(d, t, tzinfo=UTC) for t in (T.PROTECTED_START, datetime(1, 1, 1, 21, 59, 59).time())]
        outside = [datetime(2026, 11, 25, 18, 29, 59, tzinfo=UTC), datetime(2026, 11, 25, 22, 0, tzinfo=UTC)]
        for m in inside:
            self.assertTrue(T.in_protected_interval(m))
            with self.assertRaises(T.T5Refusal):
                T.check_clock(m)
        for m in outside:
            self.assertFalse(T.in_protected_interval(m))
            T.check_clock(m)
        # Another timezone's wall clock is converted to UTC first.
        cet = timezone(timedelta(hours=1))
        self.assertTrue(T.in_protected_interval(datetime(2026, 11, 25, 20, 0, tzinfo=cet)))


class IndexTest(unittest.TestCase):
    def _two(self):
        idx = T.empty_index()
        e1 = T.make_entry(idx, capture_date=date(2026, 11, 23), manifest_sha256="a" * 64, arms=["D0", "FB"],
                          code_commit="c" * 40)
        self.assertEqual(e1["previous_entry_sha256"], T.GENESIS)
        idx1 = T.append_capture(idx, e1, "2026-W48")
        e2 = T.make_entry(idx1, capture_date=date(2026, 12, 14), manifest_sha256="b" * 64, arms=["D0"],
                          code_commit="c" * 40)
        idx2 = T.append_capture(idx1, e2, "2026-W51")
        return idx, idx1, idx2

    def test_chain_and_missed_weeks(self):
        idx, idx1, idx2 = self._two()
        self.assertEqual(idx2["captures"][1]["previous_entry_sha256"], T.entry_hash(idx2["captures"][0]))
        self.assertEqual(idx2["captures"][0]["path"], "data/processed/shadow_2030/2026-11-23")
        self.assertEqual([m["iso_week"] for m in idx2["missed_weeks"]], ["2026-W49", "2026-W50"])
        self.assertEqual(T.check_chain(idx2), [])
        self.assertTrue(T.is_prefix(idx, idx1) and T.is_prefix(idx1, idx2) and T.is_prefix(idx2, idx2))
        self.assertFalse(T.is_prefix(idx2, idx1))
        tampered = json.loads(T.canonical_bytes(idx2))
        tampered["captures"][0]["manifest_sha256"] = "d" * 64
        self.assertEqual(len(T.check_chain(tampered)), 1)
        self.assertFalse(T.is_prefix(tampered, idx2))
        dropped = json.loads(T.canonical_bytes(idx2))
        dropped["missed_weeks"] = dropped["missed_weeks"][1:]
        self.assertFalse(T.is_prefix(idx2, dropped))

    def test_duplicate_week_and_bad_chain_refused(self):
        _, idx1, _ = self._two()
        again = T.make_entry(idx1, capture_date=date(2026, 11, 25), manifest_sha256="e" * 64, arms=[],
                             code_commit="c" * 40)
        with self.assertRaises(T.T5Refusal):
            T.append_capture(idx1, again, "2026-W48")
        stale = T.make_entry(T.empty_index(), capture_date=date(2026, 12, 7), manifest_sha256="e" * 64, arms=[],
                             code_commit="c" * 40)
        with self.assertRaises(T.T5Refusal):
            T.append_capture(idx1, stale, "2026-W50")


def snap(gen: str, generated: str, election: str = "2030-09-08", as_of: str | None = None) -> dict:
    return {"generation_id": gen, "generated_at_utc": generated, "election_date": election,
            "as_of": as_of or generated[:10], "path": f"{gen}/snapshot.json",
            "snapshot_file_sha256": hashlib.sha256(gen.encode()).hexdigest(),
            "deterministic_payload_sha256": "f" * 64, "source_git_commit": "0" * 40}


class AnchorSelectionTest(unittest.TestCase):
    MON = datetime(2026, 11, 23, tzinfo=UTC)

    def test_rule_on_a_synthetic_index(self):
        snaps = [
            snap("sun", "2026-11-22T04:50:00+00:00"),
            snap("mon_2026", "2026-11-23T03:00:00+00:00", election="2026-09-13"),
            snap("mon_late", "2026-11-23T09:00:00+00:00"),
            snap("mon_early", "2026-11-23T04:50:00+00:00"),
            snap("next_mon", "2026-11-30T04:50:00+00:00"),
        ]
        commit_at = {s["generation_id"]: ("c", T.parse_utc(s["generated_at_utc"]) + timedelta(minutes=5)) for s in snaps}
        start = datetime(2026, 11, 25, 10, 0, tzinfo=UTC)
        anchor, first = T.select_anchor(snaps, self.MON, start, lambda s: commit_at[s["generation_id"]])
        self.assertEqual(anchor["generation_id"], "mon_early")
        # The earliest one is not yet committed before the run starts: the next one is the anchor.
        commit_at["mon_early"] = ("c", start + timedelta(minutes=1))
        anchor, _ = T.select_anchor(snaps, self.MON, start, lambda s: commit_at[s["generation_id"]])
        self.assertEqual(anchor["generation_id"], "mon_late")
        # Missing from the archive history: skipped.
        commit_at["mon_late"] = None
        self.assertIsNone(T.select_anchor(snaps, self.MON, start, lambda s: commit_at[s["generation_id"]]))
        # A run on Monday 04:00, before anything this week: no anchor.
        early = datetime(2026, 11, 23, 4, 0, tzinfo=UTC)
        self.assertIsNone(T.select_anchor(snaps, self.MON, early, lambda s: ("c", early - timedelta(hours=1))))

    def test_anchor_from_a_git_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            prod = init_repo(Path(tmp) / "prod", "main")
            index = {"snapshots": []}
            stamps = {}
            for gen, gen_at, commit_at in (("g_sun", "2026-11-22T04:44:00+00:00", "2026-11-22T04:50:00+00:00"),
                                           ("g_mon", "2026-11-23T04:44:00+00:00", "2026-11-23T04:50:00+00:00"),
                                           ("g_mon2", "2026-11-23T09:00:00+00:00", "2026-11-23T09:10:00+00:00")):
                body = json.dumps({"generation_id": gen}).encode()
                commit_file(prod, f"{T.PROD_ARCHIVE_REL}/{gen}/snapshot.json", body, gen, when=commit_at)
                row = snap(gen, gen_at)
                row["snapshot_file_sha256"] = hashlib.sha256(body).hexdigest()
                index["snapshots"].append(row)
                stamps[gen] = commit_file(prod, T.PROD_INDEX_REL, json.dumps(index).encode(), f"index {gen}",
                                          when=commit_at)
            shadow = Path(tmp) / "shadow"
            subprocess.run(["git", "clone", "--quiet", str(prod), str(shadow)], check=True, capture_output=True)
            snaps = T.production_snapshots(shadow)
            self.assertEqual(len(snaps), 3)
            start = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)
            anchor, (commit, when) = T.select_anchor(snaps, T.iso_week_start(start), start,
                                                     lambda s: T.archive_first_commit(shadow, s))
            self.assertEqual(anchor["generation_id"], "g_mon")
            self.assertEqual(when, datetime(2026, 11, 23, 4, 50, tzinfo=UTC))
            self.assertEqual(run_git(shadow, "log", "-1", "--format=%s", commit), "g_mon")


class GitRefusalTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = init_repo(Path(self._tmp.name) / "repo", T.SHADOW_BRANCH)

    def tearDown(self):
        self._tmp.cleanup()

    def files(self, day: str) -> dict:
        return {T.SHADOW_REL / day / "manifest.json": b"{}", T.SHADOW_REL / day / "D0.npz": b"npz"}

    def index_for(self, base: dict, day: date) -> dict:
        entry = T.make_entry(base, capture_date=day, manifest_sha256=hashlib.sha256(b"{}").hexdigest(), arms=["D0"],
                             code_commit=run_git(self.repo, "rev-parse", "HEAD"))
        return T.append_capture(base, entry, T.iso_week_label(day))

    def test_commit_writes_only_its_paths_on_shadow_2030(self):
        (self.repo / "untracked.txt").write_text("not mine", encoding="utf-8")
        idx = self.index_for(T.committed_index(self.repo), date(2026, 11, 23))
        commit = T.write_and_commit(self.repo, self.files("2026-11-23"), idx, "capture", fixed_clock(SAFE))
        changed = run_git(self.repo, "show", "--name-only", "--format=", commit).splitlines()
        self.assertEqual(sorted(changed), sorted([
            "data/processed/shadow_2030/2026-11-23/D0.npz", "data/processed/shadow_2030/2026-11-23/manifest.json",
            "data/processed/shadow_2030/index.json"]))
        self.assertEqual(T.committed_index(self.repo), json.loads(T.canonical_bytes(idx)))
        self.assertIn("untracked.txt", run_git(self.repo, "status", "--porcelain"))
        # The next capture chains; rewriting a committed capture is refused.
        idx2 = self.index_for(T.committed_index(self.repo), date(2026, 12, 7))
        with self.assertRaises(T.T5Refusal):
            T.write_and_commit(self.repo, self.files("2026-11-23"), idx2, "rewrite", fixed_clock(SAFE))
        T.write_and_commit(self.repo, self.files("2026-12-07"), idx2, "capture 2", fixed_clock(SAFE))
        self.assertEqual(T.check_chain(T.committed_index(self.repo)), [])
        self.assertEqual(T.committed_index(self.repo)["missed_weeks"][0]["iso_week"], "2026-W49")

    def test_refuses_when_committed_index_is_not_a_prefix(self):
        idx = self.index_for(T.committed_index(self.repo), date(2026, 11, 23))
        T.write_and_commit(self.repo, self.files("2026-11-23"), idx, "capture", fixed_clock(SAFE))
        forged = self.index_for(T.empty_index(), date(2026, 12, 7))  # drops the committed entry
        with self.assertRaises(T.T5Refusal) as cm:
            T.write_and_commit(self.repo, self.files("2026-12-07"), forged, "capture", fixed_clock(SAFE))
        self.assertIn("prefix", str(cm.exception))
        self.assertFalse((self.repo / T.SHADOW_REL / "2026-12-07").exists())

    def test_refuses_paths_outside_the_shadow_archive(self):
        with self.assertRaises(T.T5Refusal):
            T.write_and_commit(self.repo, {Path("data/processed/prospective_forecasts/x.json"): b"{}"}, None, "bad",
                               fixed_clock(SAFE))

    def test_refuses_wrong_branch(self):
        run_git(self.repo, "checkout", "--quiet", "-b", "main")
        with self.assertRaises(T.T5Refusal) as cm:
            T.preflight(self.repo, SAFE, writing=True)
        self.assertIn("shadow/2030", str(cm.exception))
        with self.assertRaises(T.T5Refusal):
            T.write_and_commit(self.repo, self.files("2026-11-23"), None, "x", fixed_clock(SAFE))
        # A dry run only reports it.
        self.assertTrue(T.preflight(self.repo, SAFE, writing=False))

    def test_refuses_dirty_tracked_tree(self):
        (self.repo / "README").write_text("changed\n", encoding="utf-8")
        with self.assertRaises(T.T5Refusal) as cm:
            T.preflight(self.repo, SAFE, writing=True)
        self.assertIn("uncommitted", str(cm.exception))
        run_git(self.repo, "checkout", "--", "README")
        (self.repo / "new_untracked").write_text("ok", encoding="utf-8")
        self.assertEqual(T.preflight(self.repo, SAFE, writing=True), [])

    def test_refuses_in_protected_interval_with_injected_clock(self):
        evening = datetime(2026, 11, 25, 19, 0, tzinfo=UTC)
        for writing in (True, False):
            with self.assertRaises(T.T5Refusal):
                T.preflight(self.repo, evening, writing=writing)
        idx = self.index_for(T.committed_index(self.repo), date(2026, 11, 23))
        with self.assertRaises(T.T5Refusal):
            T.write_and_commit(self.repo, self.files("2026-11-23"), idx, "x", fixed_clock(evening))
        with self.assertRaises(T.T5Refusal):
            T.capture(self.repo, dry_run=True, fetch=False, clock=fixed_clock(evening))

    def test_refuses_after_the_week_ends(self):
        idx = self.index_for(T.committed_index(self.repo), date(2026, 11, 23))
        with self.assertRaises(T.T5Refusal):
            T.write_and_commit(self.repo, self.files("2026-11-23"), idx, "x", fixed_clock(SAFE),
                               week_end=SAFE - timedelta(seconds=1))


class ActivationTest(unittest.TestCase):
    def test_cand_label_from_committed_t4_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = init_repo(Path(tmp) / "repo", T.SHADOW_BRANCH)
            with self.assertRaises(T.T5Refusal):
                T.build_activation(repo)
            record = {"smoke": False, "cand_selection": {"t2_arm": "D0", "noise_law": "N0"}, "cand_equals_fb": True,
                      "fb_status": "identical to CAND; run once and recorded for both",
                      "gates": {"CAND": {"pass": False, "flag_2026": {"flagged": True}}}}
            commit = commit_file(repo, T.DEFAULT_T4_RESULTS, json.dumps(record).encode(), "t4",
                                 when="2027-01-12T15:00:00+00:00")
            cand, fb = T.build_activation(repo)
            self.assertEqual(cand["label"], {"verdict": "T4_FAIL", "flag_2026_raised": True})
            self.assertEqual(cand["completion_day_utc"], "2027-01-12")
            self.assertEqual(cand["t4_results_commit"], commit)
            self.assertTrue(cand["cand_equals_fb"])
            self.assertIsNone(fb)
            out = T.activate_cand(repo, clock=fixed_clock(SAFE))
            self.assertEqual(T.load_activation(repo)["label"]["verdict"], "T4_FAIL")
            self.assertTrue(out["commit"])
            with self.assertRaises(T.T5Refusal):
                T.activate_cand(repo, clock=fixed_clock(SAFE))


class SnapshotContextTest(unittest.TestCase):
    """The snapshot data root reproduces T2's pinned context and is causal at O."""

    def test_snapshot_context_matches_pinned_t2_and_cuts_at_origin(self):
        from diagnostics.calibration_2030 import t2_arms as A

        with tempfile.TemporaryDirectory() as tmp:
            export = T.export_tree(T.ROOT, "HEAD", Path(tmp) / "snap")
            ctx = T.build_snapshot_context(export, date(2026, 11, 18), Path(tmp) / "w", n_draws=16)
            pinned = A.t2_context(n_draws=16, v03_dir=Path(tmp) / "pinned")
            self.assertEqual(ctx["v03_file_sha256"], pinned["v03_file_sha256"])
            self.assertEqual(len(ctx["tp_n"]), len(pinned["tp_n"]))
            self.assertEqual(Path(ctx["polls_file"]), export / T.SNAP_V02_POLLS)
            self.assertEqual(ctx["election_days"], [*pinned["election_days"], T.ELECTION_2030])
            early = T.build_snapshot_context(export, date(2026, 10, 1), Path(tmp) / "w2", n_draws=16)
            self.assertGreater(early["t5_counts"]["polls_after_origin_excluded"], 0)
            self.assertLessEqual(int(early["tp_avail"].max()), date(2026, 10, 1).toordinal())
            self.assertLessEqual(early["row_days"][-1], date(2026, 10, 1))
            # The open-interval shock now includes 2030-09-08 for targets after it, never on it.
            o = date(2026, 10, 1)
            self.assertFalse(A.election_shock_cov(early, o, T.ELECTION_2030).any())
            self.assertTrue(A.election_shock_cov(early, o, T.ELECTION_2030 + timedelta(days=1)).any())


class GuardTest(unittest.TestCase):
    def test_frozen_t5_body_and_record(self):
        p = T.verify_t5(allow_dirty=True)
        self.assertEqual(p["t5_body_sha256"], T.T5_BODY_SHA256)
        # T5 amendments 001 (Ul) and 002 (U0) are in force from 2026-11-19 (none at freeze).
        self.assertEqual([a["amendment_id"] for a in p["t5_amendments_in_force"]],
                         ["001-ul-log-ratio-arm", "002-u0-shadow-arm"])


if __name__ == "__main__":
    unittest.main()
