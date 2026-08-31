# Part 3D-R — evaluator freeze reconstructibility repair

**Scope: freeze metadata only.** No scientific evaluator file was modified. No
challenger was implemented or scored. The preregistration was not amended. Case
sets, metrics, truths, seeds, `N`, CONTROL, geography and mandate logic are
untouched, and the tests that guard them still pass.

## The defect

`freeze.verify()` compares each import-closure module against the recorded
`working_tree_sha256`. The Part-3D freeze (`aea30ba`) was built from a dirty
working tree, so that recorded reference was not reconstructible from any commit.
A clean checkout therefore failed verification — the condition Part 4 must clear
before implementing a challenger.

Two distinct reconstructibility defects were present. The audit named the first;
the second surfaced while regenerating the freeze.

**Defect 1 — a reference blob present in no commit.**
`scripts/pollofpolls/normalize.py` was captured while an unrelated additive edit
(the party-chart / forecast-history work) was uncommitted:

| | SHA-256 |
|---|---|
| recorded `working_tree_sha256` | `c6b0480d89cde0b892e1769a394e13d29bdde04d1a8abcce178cbe44cd4da09a` |
| committed `head_sha256` (any clean checkout) | `437a78560aea68163f425352348ba95e7d944d72350bd48e74e286feeec75b59` |

The edit is purely additive — it adds `parse_party_chart_pop_series` and
`extract_party_chart_pop_timeseries` and changes no existing function — so the
evaluator's *behaviour* was unaffected. The freeze contract is nevertheless a hash
contract, and it could only be satisfied on a machine still carrying that edit.

**Defect 2 — the evaluator's own modules had no committed reference.**
The Part-3D freeze recorded `head_sha256: null` for seven modules, because it was
built at `7f37e12`, a commit at which `control_baseline_amendment2/` was still
untracked:

```
diagnostics/election_noise_v2/control_baseline_amendment2/__init__.py
diagnostics/election_noise_v2/control_baseline_amendment2/harness2/__init__.py
diagnostics/election_noise_v2/control_baseline_amendment2/harness2/exact_oracle.py
diagnostics/election_noise_v2/control_baseline_amendment2/harness2/freeze.py
diagnostics/election_noise_v2/control_baseline_amendment2/harness2/isolated.py
diagnostics/election_noise_v2/control_baseline_amendment2/harness2/manifest.py
diagnostics/election_noise_v2/control_baseline_amendment2/harness2/run_control.py
```

These are the most load-bearing files in the closure — the isolated seat path, the
exact oracle, the case manifest, the CONTROL runner and the freeze mechanism
itself. The freeze pinned their working-tree bytes but held no committed
counterpart to check them against.

## The repair

The freeze was regenerated from a clean checkout of
`aea30bacfa8dfe342198d3d6ff8748e84c9ffd9c`, at which every closure module is
committed. **No verifier semantics were changed and `normalize.py` was not
special-cased**; the fix is a reconstructible freeze, not a looser check. The
regenerated artifact satisfies, for all 66 closure modules:

* `working_tree_sha256 == head_sha256`
* `head_sha256 is not null`
* `uncommitted_local_edit == false`

Everything else in the freeze is byte-identical to the Part-3D artifact. The only
other differing fields are the freeze timestamp and the `git.base_commit` /
`git.branch` provenance labels. In particular the Amendment-2 preregistration
reference, all 21 metric-implementation hashes, all 10 baseline-artifact hashes,
all 7 truth-input hashes, the 8 preserved Part-3 diagnostic hashes, the seed/`N`
policy and the paired-randomness contract are unchanged.

## Pre-repair scientific reproduction

Before the freeze was regenerated, the clean tree was required to reproduce the
certified Part-3D results — a full 5 seeds × 20 000 draws × 30 jobs CONTROL run
written to a scratch directory, never over the committed baseline.

All 10 artifacts matched. Nine were byte-identical as written. The tenth,
`control_scores_by_case_seed.csv`, differed **only** in `elapsed_seconds`, per-job
wall-clock runtime; with that single column dropped both sides hash to
`4dec36bdbae4549b41f0ceb1550d8730a5880696d0e24b79aa04739cd69f3744`. All 48
scientific columns matched exactly, including the Tier-1 and Tier-3-ISO scores,
coalition Brier, lambda diagnostics, `mandate_law`, `first_divisor`,
`geography_mode`, `geography_baseline_year`, `seat_total_always_349`,
`mc_mean_seats` and `mc_mean_vote_pct`.

Headline values, unchanged: Tier-1 joint vote ES `3.1332307726237563`; Tier-3-ISO
seat ES `10.920286666666668`; coalition Brier `0.026677477940944883`. Tier 1 was
again bit-identical to the Part-3 baseline, and the Monte Carlo again agreed with
the exact finite-support oracle.

The machine-readable comparison is `clean_reproduction_comparison.json`.

## Guarding the repair

`tests/test_evaluator_freeze_reconstructible.py` fails if either defect returns:
it asserts zero uncommitted-edit and zero null-`head_sha256` entries, working-tree
equality with HEAD for every closure module, that each recorded `head_sha256`
matches the blob actually retrievable from the commit, that the dirty
`normalize.py` variant was not absorbed, that `verify()` reports no drift, and
that the recorded reproduction is pinned to the current baseline file hashes so it
cannot go stale unnoticed.
