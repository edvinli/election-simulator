# Election-noise residuals, amendment 002: polls missing a pivot key

**Status: FINAL.** The user decided option (b) of amendment 001 §5 on
2026-11-18. This amendment was drafted on 2026-11-18 and finalized at 2026-11-18T16:44:12Z
on branch `prereg/calibration-2030-drafts`. It takes effect when committed
with all of the following:

- its JSON record,
  `data/processed/election_residuals/amendments/002-missing-key-polls.json`;
- that record's SHA-256 companion;
- its entry in `data/processed/election_residuals/amendments/index.json`.

It is append-only. It builds on amendment 001
(`docs/election_noise_residual_amendment_001.md`) and edits neither 001 nor
any earlier document or result. Together, 001 and 002 define the corrected
residuals: arm `F1` of `docs/calibration_2030_preregistration.md`.

## 1. The defect

`scripts/election_residuals/consensus.py` turns the long poll table into one
row per poll with `pivot_table`. It uses as index keys `poll_id`, `pollster`,
`pollster_original`, `interview_start`, `interview_end`, `publication_date`
and `sample_size`. pandas drops a group whose key is missing. So a poll with
no sample size or no fieldwork start silently disappears before
deduplication.

The code's own intent is the opposite. `compute_poll_weight` returns weight
1.0 and a `sample_size_missing` flag for a poll without `n`, and
`ContributingPoll.interview_start` is optional. No such poll ever reaches
them.

## 2. The amended rule

Every poll eligible under amendment 001 reaches deduplication:

- **Missing sample size.** Weight 1.0 and `sample_size_missing = true`, as
  `compute_poll_weight` specifies.
- **Missing fieldwork start.** The poll is kept with `interview_start = None`.
- **Tie-break.** The order is unchanged: `interview_end`, `publication_date`,
  `interview_start`, `sample_size`, `poll_id`, all descending. A missing value
  sorts **last** within its key.

Nothing else changes: the window, the latest poll per pollster, the weights
for reported sizes, REST, and the amendment 001 eligibility.

## 3. Implementation

**`scripts/election_residuals/eve_consensus.py`** implements 001 and 002
together. It pivots support on `poll_id` alone and joins one metadata row per
poll, so no missing value can drop a poll. `consensus.py` stays byte-identical
(SHA-256 `702f75a2…a4f46c`), so archived forecasts and frozen harnesses
reproduce.

**`tests/test_eve_consensus.py`** asserts the following:

- On the committed polls, for all seven elections 2002–2026, the two builders
  agree **exactly** (composition and contributing polls) once the polls either
  amendment acts on are removed. The amendments change nothing else.
- No contributing poll has fieldwork or publication on election day.
- The three polls in §4 contribute.
- On synthetic data:
  - election-day polls and same-day publications are excluded;
  - a poll missing `n` gets weight 1.0;
  - a missing start loses a tie but is used when it is the latest poll.

## 4. Effect

Evidence:
`diagnostics/election_noise_residual_eve_rule/verify_missing_keys.py`, which
writes `results_002.json`
(SHA-256 `1aae386cbcd484774841be5c932607f7c269bb727f31d4789b81076e3d6b4a3e`).
It is deterministic: a rerun gives the same bytes. It was computed from the
inputs pinned in amendment 001 §1, plus `eve_consensus.py`.

| election | poll now included | residual norm (001 → 001 + 002) |
| --- | --- | --- |
| 2002 | Skop, fieldwork to 2002-09-13, no `n` (replaces Skop to 2002-09-12) | 5.45 → 5.12 pp |
| 2006 | Ipsos / Synovate Temo, fieldwork to 2006-09-16, no start (replaces Ipsos to 2006-09-14) | 2.29 → 2.32 pp |
| 2022 | Infostat, fieldwork to 2022-09-07, no `n` (Infostat was absent) | 2.85 → 2.78 pp |
| 2010, 2014, 2018, 2026 | none | unchanged |

`pp_lw_gaussian` fitted to the residuals:

| pool | current rule | 001 | 001 + 002 (`F1`) |
| --- | --- | --- | --- |
| 2002–2022, total variance (pp²) | 7.29 | 8.90 | **8.39** |
| 2002–2026, total variance (pp²) | 8.12 | 9.64 | **9.28** |

002 moves the noise **narrower** than 001 alone. Taken together, the
corrected production pool (2002–2022) is 15 % wider than the current one.
Both amendments were adopted on correctness grounds: their direction was
measured after each rule was fixed, and neither rule was chosen for its
direction.

## 5. Scope

It **does**:

- complete the corrected residual definition used by T3 and T4 of
  `docs/calibration_2030_preregistration.md`.

It **does not**:

- change production forecasts, `MODEL_VERSION` or the adopted noise law;
- rewrite any recorded residual, replay, benchmark capture or archived
  forecast;
- alter amendment 001's evidence. `results.json` keeps SHA-256 `edd1e8cf…`.

The production consumers named in amendment 001 §7 adopt
`build_eve_polling_consensus` together, and only with a later model release.
