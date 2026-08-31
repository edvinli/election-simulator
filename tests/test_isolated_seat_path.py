"""Audit-only tests guarding the Amendment-2 isolated seat/coalition path.

These assert the properties Amendment 2 relies on, so that a later Tier 3-ISO
runner cannot silently violate them:

* the geography path consumes no target-election realized information in
  chronological mode, and the electorates file does not enter its output;
* oracle mode — which does consume target-realized row margins — is distinguishable
  and therefore prohibitable;
* the final 14-day consensus admits only polls published on or before election day;
* the statutory law dispatches PRE_2018 for 2014 and POST_2018 for 2018/2022;
* the CONTROL law on this path is exactly K atoms in both vote and seat space.

No predictive score against any certified outcome is computed. Heavy Monte Carlo is
avoided; draw counts are small and the frozen N is never used here.
"""

from __future__ import annotations

from datetime import date
from fractions import Fraction
import json
from pathlib import Path
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PART2B = REPO_ROOT / "diagnostics/election_noise_v2/historical_seat_extension/processed"
RESEARCH_GEO = PART2B / "research_geography"

from scripts.election_layer_v2.config import CANONICAL_WINDOW_DAYS, MIN_SHARE_PCT
from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals
from scripts.election_layer_v2.transfer import apply_batch_simplex_transfer
from scripts.election_residuals.config import ALL_CATEGORIES, DEFAULT_POLLS_FILE
from scripts.election_residuals.consensus import build_election_polling_consensus
from scripts.mandates.law import MandateLaw, mandate_law_for_election_year
from scripts.simulator.config import MODEL_PARTIES_9, PARLIAMENTARY_PARTIES_8

_HAS_GEO = (RESEARCH_GEO / "constituency_party_votes_2014_2022.csv").exists()

TARGETS = {
    2014: {"election_date": date(2014, 9, 14), "baseline": 2010, "k": 3},
    2018: {"election_date": date(2018, 9, 9), "baseline": 2014, "k": 4},
    2022: {"election_date": date(2022, 9, 11), "baseline": 2018, "k": 5},
}


class LawDispatchOnIsolatedPathTest(unittest.TestCase):
    def test_statutory_law_per_target(self) -> None:
        self.assertIs(mandate_law_for_election_year(2014).law, MandateLaw.PRE_2018)
        self.assertEqual(mandate_law_for_election_year(2014).first_divisor, Fraction(7, 5))
        for y in (2018, 2022):
            self.assertIs(mandate_law_for_election_year(y).law, MandateLaw.POST_2018)
            self.assertEqual(mandate_law_for_election_year(y).first_divisor, Fraction(6, 5))


class ConsensusPublicationSafetyTest(unittest.TestCase):
    """The consensus may never admit a poll unpublished at the forecast origin."""

    @classmethod
    def setUpClass(cls) -> None:
        import pandas as pd

        cls.polls = pd.read_csv(DEFAULT_POLLS_FILE)

    def test_every_retained_poll_published_on_or_before_election_day(self) -> None:
        for year, spec in TARGETS.items():
            ed = spec["election_date"]
            cons = build_election_polling_consensus(ed, self.polls, window_days=CANONICAL_WINDOW_DAYS)
            self.assertGreater(cons.retained_pollsters_count, 0, year)
            for p in cons.contributing_polls:
                self.assertLessEqual(p.publication_date, ed, f"{year} {p.pollster}")
                self.assertLessEqual(p.interview_end, ed, f"{year} {p.pollster}")

    def test_consensus_is_a_valid_composition(self) -> None:
        for year, spec in TARGETS.items():
            cons = build_election_polling_consensus(
                spec["election_date"], self.polls, window_days=CANONICAL_WINDOW_DAYS
            )
            total = sum(cons.consensus_composition[c] for c in ALL_CATEGORIES)
            self.assertAlmostEqual(total, 100.0, places=6, msg=str(year))
            self.assertGreaterEqual(cons.consensus_composition["REST"], 0.0)

    def test_window_is_the_frozen_fourteen_days(self) -> None:
        self.assertEqual(CANONICAL_WINDOW_DAYS, 14)


class ResidualPoolLeakageTest(unittest.TestCase):
    def test_no_future_year_enters_a_training_pool(self) -> None:
        for year, spec in TARGETS.items():
            pool = load_chronological_pp_residuals(target_election_year=year)
            self.assertEqual(len(pool.training_years), spec["k"], year)
            for y in pool.training_years:
                self.assertLess(int(y), year, f"leak into {year}")


class ControlIsExactlyKAtomsTest(unittest.TestCase):
    """CONTROL's isolated-path law has exactly K support points in vote space."""

    def test_k_atoms_in_vote_space(self) -> None:
        import pandas as pd

        polls = pd.read_csv(DEFAULT_POLLS_FILE)
        for year, spec in TARGETS.items():
            cons = build_election_polling_consensus(
                spec["election_date"], polls, window_days=CANONICAL_WINDOW_DAYS
            )
            base = np.array([cons.consensus_composition[c] for c in ALL_CATEGORIES], dtype=float)
            pool = load_chronological_pp_residuals(target_election_year=year)
            k = len(pool.training_years)
            idx = np.random.default_rng(7).integers(0, k, size=600)
            votes, lam = apply_batch_simplex_transfer(
                np.tile(base, (600, 1)), pool.centered_residuals_matrix[idx], eps=MIN_SHARE_PCT
            )
            uniq = np.unique(np.round(votes, 10), axis=0)
            self.assertEqual(uniq.shape[0], k, f"{year}: expected {k} atoms")
            np.testing.assert_allclose(votes.sum(axis=1), 100.0, atol=1e-9)
            # lambda never binds at these targets, so the atom count survives intact
            self.assertTrue(np.all(lam == 1.0), f"{year}: simplex floor bound unexpectedly")


@unittest.skipUnless(_HAS_GEO, "Part-2B research geography not present")
class GeographyHasNoTargetRealizedInputTest(unittest.TestCase):
    """Chronological mode must not consume target-election realized information."""

    SHARES = dict(zip(MODEL_PARTIES_9, [0.20, 0.05, 0.07, 0.06, 0.30, 0.07, 0.06, 0.17, 0.02]))

    def _project(self, target: int, processed_dir: Path, mode: str = "chronological"):
        from scripts.geography.projection import project_constituency_votes

        return project_constituency_votes(
            national_vote_shares=self.SHARES,
            baseline_year=TARGETS[target]["baseline"],
            target_year=target,
            mode=mode,
            total_national_votes=None,
            processed_dir=processed_dir,
        )

    def test_electorates_file_does_not_enter_chronological_output(self) -> None:
        import shutil
        import tempfile

        import pandas as pd

        with tempfile.TemporaryDirectory() as td:
            pert = Path(td)
            shutil.copy(RESEARCH_GEO / "constituency_party_votes_2014_2022.csv", pert)
            el = pd.read_csv(RESEARCH_GEO / "constituency_electorates_2014_2026.csv")
            for col in ("eligible_voters", "valid_votes"):
                if col in el.columns:
                    el[col] = pd.to_numeric(el[col], errors="coerce") * 7.77 + 12345
            el.to_csv(pert / "constituency_electorates_2014_2026.csv", index=False)
            for target in TARGETS:
                a = self._project(target, RESEARCH_GEO)
                b = self._project(target, pert)
                self.assertEqual(a.constituency_votes, b.constituency_votes, target)
                self.assertEqual(a.constituency_valid_votes, b.constituency_valid_votes, target)

    def test_row_totals_come_from_the_baseline_election(self) -> None:
        """Projected constituency valid votes must equal the baseline election totals."""
        import pandas as pd

        votes = pd.read_csv(RESEARCH_GEO / "constituency_party_votes_2014_2022.csv")
        for target, spec in TARGETS.items():
            proj = self._project(target, RESEARCH_GEO)
            base = votes[votes["election_year"] == spec["baseline"]]
            baseline_total = int(base["votes"].sum())
            self.assertEqual(
                sum(proj.constituency_valid_votes.values()), baseline_total, target
            )

    def test_oracle_mode_differs_and_is_therefore_prohibitable(self) -> None:
        """Oracle mode consumes target-realized row margins, so it must differ."""
        # 2018 and 2022 have target valid votes in the electorates file; 2014 does not
        # in the research copy, so oracle mode is only checked where it is runnable.
        differed = 0
        for target in (2018, 2022):
            a = self._project(target, RESEARCH_GEO, mode="chronological")
            try:
                b = self._project(target, RESEARCH_GEO, mode="oracle")
            except Exception:
                continue
            if a.constituency_valid_votes != b.constituency_valid_votes:
                differed += 1
        self.assertGreater(
            differed, 0, "oracle and chronological modes must be distinguishable"
        )


@unittest.skipUnless(_HAS_GEO, "Part-2B research geography not present")
class IsolatedPathProducesValidSeatDrawsTest(unittest.TestCase):
    def test_all_three_targets_produce_349_seat_draws(self) -> None:
        import pandas as pd

        from scripts.geography.projection import project_constituency_votes
        from scripts.mandates.allocator import allocate_riksdag_seats

        fixed_all = json.loads((PART2B / "fixed_seats_by_year.json").read_text())["fixed_seats_by_year"]
        polls = pd.read_csv(DEFAULT_POLLS_FILE)
        for year, spec in TARGETS.items():
            cons = build_election_polling_consensus(
                spec["election_date"], polls, window_days=CANONICAL_WINDOW_DAYS
            )
            base = np.array([cons.consensus_composition[c] for c in ALL_CATEGORIES], dtype=float)
            pool = load_chronological_pp_residuals(target_election_year=year)
            cfg = mandate_law_for_election_year(year)
            fixed = {k: int(v) for k, v in fixed_all[str(year)].items()}
            seats = []
            for atom in range(len(pool.training_years)):
                v, _ = apply_batch_simplex_transfer(
                    base[None, :], pool.centered_residuals_matrix[atom][None, :], eps=MIN_SHARE_PCT
                )
                shares = {p: float(v[0, j] / 100.0) for j, p in enumerate(MODEL_PARTIES_9)}
                proj = project_constituency_votes(
                    national_vote_shares=shares,
                    baseline_year=spec["baseline"],
                    target_year=year,
                    mode="chronological",
                    total_national_votes=None,
                    processed_dir=RESEARCH_GEO,
                )
                alloc = allocate_riksdag_seats(
                    proj.to_allocator_input(), fixed,
                    first_divisor=cfg.first_divisor, law=cfg.law,
                    scenario_id=f"test_iso_{year}_{atom}",
                )
                self.assertEqual(alloc.law, cfg.law.value)
                seats.append([alloc.final_seats_by_party.get(p, 0) for p in PARLIAMENTARY_PARTIES_8])
            S = np.array(seats, dtype=np.int64)
            self.assertTrue(np.all(S.sum(axis=1) == 349), year)
            # exactly K atoms survive into seat space
            self.assertEqual(np.unique(S, axis=0).shape[0], len(pool.training_years), year)


if __name__ == "__main__":
    unittest.main()
