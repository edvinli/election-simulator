# Model 1.2.1-rc1: corrected election-noise residuals

**Release:** `MODEL_VERSION` 1.2.1-rc1, `RELEASE_TAG`
`election-simulator-v1.2.1-rc1`, from 1.2.0-rc1.
**Applies to:** forecasts that target the 2030 election or later.
**Authorized by:** the user, as a correctness fix (arm `F1` of the 2030
calibration programme).

## What changed

ElectionNoise B (`pp_lw_gaussian`) is fitted on historical election-day
residuals: the certified result minus the final pre-election poll consensus.
Two defects in how that consensus was built are fixed by the finalized,
append-only residual amendments:

- **001** (`docs/election_noise_residual_amendment_001.md`): election-day
  and exit polls leaked into the "final" consensus. A poll now counts only if
  `publication_date < E` and `interview_end < E`: the information a
  forecaster had on election eve.
- **002** (`docs/election_noise_residual_amendment_002.md`): the pandas pivot
  silently dropped polls with no sample size or no fieldwork start (2002,
  2006 and 2022). Every eligible poll now reaches deduplication.

The corrected builder is `scripts/election_residuals/eve_consensus.py`
(SHA-256 `781bf689…b7d20`, rule `election-eve-consensus-amendments-001-002`).
It and the amendment records were brought over byte-identical from
`prereg/calibration-2030-drafts`, and the amendment indexes and SHA-256
companions verify.

Nothing else changes. The law, its zero tunable parameters, the K = 6 pool
of 2002–2022, OpinionState, Dynamics, geography and the mandate allocator are
the same. Adding 2026 to the pool is a separate, governed decision and is
not part of this release.

Effect on the fitted law, from amendment 002's evidence:

| pool 2002–2022 | unamended | corrected |
| --- | --- | --- |
| total variance, `trace(δT + (1−δ)S_P)` (pp²) | 7.29 | 8.39 |

The corrected noise is about 15 % wider in variance. Its mean is still zero,
because the residuals are centred, so central forecasts barely move.

## Scope: by target election

| target | model version | residual consensus | pool |
| --- | --- | --- | --- |
| 2010–2026 | 1.2.0-rc1 | `unamended-election-polling-consensus` | `residuals_pool.load_chronological_pp_residuals` |
| 2030 and later | 1.2.1-rc1 | `election-eve-consensus-amendments-001-002` | `production_residuals_pool.load_eve_pp_residuals` |

`scripts/simulator/config.py` holds the routing: `model_version_for`,
`release_tag_for` and `residual_consensus_rule_for`. It is a hashed model
file, so the history fingerprint covers the rule.

**A 2026 target is bit-identical.** It still records `model_version`
1.2.0-rc1, the same `model_config` and hash, and draws from the same pool
through the same frozen function. This was verified in two ways:

- Same machine, before and after (Python 3.12, `origin/main` 5c7e87a against
  this branch). Three 3,000-draw 2026 forecasts, at 2026-06-01, 2026-08-24
  and 2026-09-12, gave byte-identical draw matrices, manifests and
  deterministic payload hashes. Every legacy pool for 2010–2030 was also
  byte-identical.
- `tests/test_production_residuals_pool.py`, which pins:
  - the 2026 pool and the 2026 national draws against the frozen loader;
  - the 2026 `model_config_hash`.

Archived generations, sidecars, benchmark captures and their hashes are not
touched.

**Historical tools keep their rule.** The rule is unchanged in:

- `forward_eval.py` and `hindcast.py`, in both `election_layer_v2` and
  `vote_share_calibration`;
- `pop_baseline`, `opinion_precision_challenger` and `election_layer`;
- the frozen harnesses.

They call `load_chronological_pp_residuals` or
`build_election_polling_consensus` directly, as before.

## Implementation

**New module.** `scripts/election_layer_v2/production_residuals_pool.py`
provides:

- `load_production_pp_residuals`, the target router;
- `load_eve_pp_residuals`, the corrected pool;
- `consensus_builder_for`.

The corrected pool is the frozen loader with the consensus builder as a
parameter. A test asserts that, given the unamended builder, it reproduces
`load_chronological_pp_residuals` exactly for every target.

**Frozen files are not edited.** `residuals_pool.py` and `consensus.py`
stay byte-identical; `test_production_challenger_b` and the challenger freeze
pin them. Amendment 001 §7 names `residuals_pool.py` as the production
consumer to switch. The switch is made in its callers instead, so that the
frozen file and the archived forecasts it reproduces keep their bytes.

**Changed files.** All production pool consumers now call the router:

- `vote_share_calibration/national_engine.py`;
- `vote_share_calibration/production_national_engine.py`, the diagnostic
  wrapper;
- `forecast_history/projection_simulator.py`, the future fan;
- `forecast_history/campaign_paths.py`, whose election-day parity check
  against the national engine needs the same pool.

The other changes:

- **`simulator/engine.py`** adds `model_config["residual_consensus_rule"]`,
  and only for a corrected-rule target. A 2026 `model_config` is unchanged.
- **`simulator/reproducibility.py`** records `model_version_for(target)`
  rather than the constant.
- **`vote_share_calibration/national_engine.py`** also adds
  `election_noise_residual_consensus_rule` to the diagnostics, again only
  for corrected-rule targets.
- **`forecast_history/effective_inputs.py`**: the fingerprint's training
  polls come from the target's own consensus builder, as amendment 001 §7
  requires. `MODEL_FILES` gains `production_residuals_pool` and
  `eve_consensus`, and a corrected-rule target's static part records the rule
  by value. `VERSION` stays 1.
- **Legacy recovery** (`legacy_inputs`, for an artifact without recorded
  fingerprints) may name a revision that predates the two new files. Those
  files are now hashed as absent (`None`) rather than making the revision
  unrecoverable. Such a fingerprint can never match a current one. The
  current tree still requires every model file, and the commit itself must
  be present.

**Freezes.** Every file above that sits in a freeze was already known
drift (`config.py`, `engine.py`, `reproducibility.py`, `national_engine.py`),
with one exception: `production_national_engine.py`. It is named in
`tests/test_production_freeze.py` as `CORRECTED_RESIDUALS_CHANGED`. No
freeze is re-issued.

## The next production run

**Certification** (publication workflow, 120-minute job, unchanged):

- one 100,000-draw 2030 forecast under 1.2.1-rc1, same cost as before;
- the snapshot records `model.version` 1.2.1-rc1 and the consensus rule;
- the exact-draw sidecar of amendment 007 is exported as before.

Its `replay model_config` check compares the snapshot with the same engine,
so it agrees. If Full CI has not verified the merged code, the contract suite
runs first. That includes the live-history lane described below.

**Rendering** (render workflow, 90-minute job, outside the production lock):

1. The roll-in turns the previous `current_production` point into
   `prospective_archived`, which leaves a hole.
2. Because `MODEL_FILES` changed, no recorded fingerprint matches, so the
   backfill rebuilds every reconstructed point as well as the hole.
3. Those are about 55 points of 10,000 draws on four workers: the 54
   reconstructed points from 2026-09-14 to 2026-11-18, plus the previous
   publication day.

Locally, one 10,000-draw 2030 point takes 61 s serial. The 2026-09-10 CI
render did 123 points in 58m54s on four workers, about 1.9 runner-minutes
per point. **Expect about 15 minutes locally and 25–30 minutes on a hosted
runner.** That is well inside the 90-minute render timeout, and it never
holds the production lock.

The future fan and campaign paths are not built before 2030-05-19. The
publication timeout is not changed.

**After that render**, every point carries a 1.2.1-rc1 fingerprint and the
cache works again. The archived prospective points keep the model they were
published under (1.2.0-rc1), as the history contract intends. The
reconstructed curve is a retrospective reconstruction under today's model.

**If the render's backfill fails**, the certified point is still installed,
because the curve never blocks a certified forecast. The stale points are
rebuilt by the next backfill.

### The live-history test lane

`tests/test_election_automation.py::test_existing_reconstructed_points_are_reused_without_rerunning`
runs against the committed 2030 history. Between the merge and the first
render, no recorded fingerprint matches the new code. The test excuses that
only when the recorded fingerprints are exactly what the previous code
produced.

The helper used to load the previous model files into *today's*
`effective_inputs.py`. That cannot reproduce anything once a change also
extends what a fingerprint covers, as this one does. It now runs the previous
revision's whole `scripts/` tree (`git archive <change>^ scripts`) in a
subprocess. The strict expectation is unchanged:

- a fingerprint that matches neither version still fails;
- a shallow clone still falls back to the strict check.

To keep "the previous code" well defined, every change to a fingerprinted
file is in a single commit. The amendment files arrive in an earlier commit.
A squash merge also satisfies this.

The lane needs Python 3.12 locally. Under 3.11 it fails even on `main`.

## Website

The site reads the model version from `metadata.json` (`model.version`) at
run time, so "Teknisk information" shows 1.2.1-rc1 without an edit. The
methods text should say which consensus the residuals are measured against.
These edits are suggested for `_pages/election_simulator.md` in the website
repository. The site is not changed by this release.

- **Step 3, first paragraph.** Current text: "Modellen använder därför
  historiska skillnader mellan det sista samlade mätläget och valresultatet:".
  Suggested replacement: "Modellen använder därför historiska skillnader
  mellan mätläget kvällen före valet och valresultatet. Mätläget är varje
  instituts senaste mätning under de sista 14 dagarna, med fältarbete och
  publicering före valdagen. Mätningar som görs eller publiceras på valdagen,
  till exempel vallokalsundersökningar, räknas inte:"
  - English: "The model therefore uses historical differences between the
    polling position on election eve and the result. That position is each
    pollster's latest poll in the final 14 days, fielded and published before
    election day; polls conducted or published on election day, such as exit
    polls, do not count."
- **Equation.** `\text{slutligt mätläge}_e` becomes
  `\text{mätläge kvällen före valet}_e`.
- **New sentence after the δ paragraph.** "Från modellversion 1.2.1
  (prognoser för valet 2030) räknas de historiska felen mot mätläget kvällen
  före valet, och mätningar som saknar urvalsstorlek tas med. Tidigare kom
  några mätningar från själva valdagen med, medan andra föll bort. Rättelsen
  gör valdagsfelet något större."
  - English: "From model version 1.2.1 (forecasts of the 2030 election), the
    historical errors are measured against the election-eve polling position,
    and polls without a reported sample size are included. Previously some
    election-day polls were included and some polls were dropped. The
    correction makes the election-day error somewhat larger."
- **Key assumption 3.** Current text: "Historiska skillnader mellan
  slutmätningar och valresultat …". Suggested replacement: "Historiska
  skillnader mellan mätläget kvällen före valet och valresultatet …".
