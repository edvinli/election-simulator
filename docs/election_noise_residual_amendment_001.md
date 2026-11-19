# Election-noise residuals, amendment 001: the election-eve information set

**Status: FINAL.** Approved by the user on 2026-11-18 and finalized at
2026-11-18T16:41:53Z. It was drafted at 2026-11-18T16:32:08Z on branch
`prereg/calibration-2030-drafts` from `origin/main` `6c4271c`.

It takes effect when committed with all of the following:

- its JSON record,
  `data/processed/election_residuals/amendments/001-eve-information-set.json`;
- that record's SHA-256 companion;
- its entry in `data/processed/election_residuals/amendments/index.json`.

Taking effect defines the residual rule for the 2030 calibration programme
(§6). It changes no production forecast, archived forecast or recorded
result.

It is append-only. It does not edit:

- `docs/election_noise_v2_preregistration.md`;
- `docs/election_noise_adopted_model.md`;
- the replay protocols;
- any result of those documents.

## 1. The defect

Every election-noise residual is the certified result minus a final-poll
consensus. `scripts/election_residuals/consensus.py` builds that consensus
from each pollster's latest poll whose fieldwork ended in the 14 days up to
election day `E`. Its eligibility test is:

```
publication_date <= E  and  interview_end <= E  and  interview_end >= E - 14
```

`<= E` admits polls taken or published **on election day**. These include
exit polls (SVT VALU, TV4), election-day polls (Sifo 2022, Novus 2026) and,
in 2018, one ordinary poll first published on election day. None of them
exists when an election-eve forecast is made. Exit polls also sit closer to
the result than final polls do, so the residuals are biased toward zero and
the noise law fitted to them is too narrow.

The poll aggregate (`scripts/poll_aggregate/data.py`) already excludes such
polls ("exit and election-day polls are superseded by the count").

Evidence: `diagnostics/election_noise_residual_eve_rule/verify_eve_rule.py`
and its output `results.json`
(SHA-256 `edd1e8cf5e533f02fc664da756ff456d412d7fefb1759cfa4ad939bc6e661c85`).
It was computed from:

| input | SHA-256 |
| --- | --- |
| SwedishPolls polls (`data/processed/pollofpolls/swedishpolls_individual_polls.csv`) | `18e84e56eff64a5133eb4eeca9767893a7f74b15724552ef573fd9314e8d2828` |
| Results 2002–2022 (`data/processed/elections/riksdag_election_results.csv`) | `3a99ffb1b0fbef60d61712764a2136982ecf01c4879a43edc6d9a38ddb4ba0fc` |
| 2026 certified manifest | `6bdcbd284f602d9c9b1b9605ab3d52b5700528b54232565f3242d01d107f788d` |
| `scripts/election_residuals/consensus.py` | `702f75a2371f08abba521112d7edd3188a6a7b7811b75c33e472f1a421a4f46c` |

## 2. The amended rule

For election day `E`, a poll is eligible for the final-poll consensus **if and
only if**

```
publication_date < E  and  interview_end < E  and  interview_end >= E - 14
```

with dates compared as ISO calendar dates. Everything after eligibility is
unchanged:

- latest poll per pollster;
- the five-key tie-break;
- weights `clip(sqrt(n / 1000), 0.7, 1.5)`;
- REST as the remainder.

Both conjuncts are kept on purpose. `publication_date < E` alone would admit
a row whose fieldwork ends after its publication. The aggregate rejects such
rows as inconsistent, and the window test here does not.

**Relation to `scripts/poll_aggregate/`.** The aggregate does not filter on a
date predicate. It drops any poll whose fieldwork interval contains an
election day (`start <= E <= end`), and then admits polls by availability
(`publication_date <= as_of`). The amended rule equals the aggregate's
**information set on election eve** (`as_of = E − 1`). It does not equal the
aggregate's exclusion rule.

The two differ for exactly one poll in 2002–2026: Skop
`swp-c1c5acbdcbe9d9d1ae01`, fieldwork 2018-09-02 to 2018-09-07, first
published 2018-09-09 (election day). The exclusion rule keeps it; the
amended rule drops it, because no forecaster had it on the eve.
`results.json` records `eve_rule_equals_aggregate_exclusion_for_all: false`
for this reason.

**Reference implementation.** Pre-filter the polls table with the predicate,
then call the unchanged `build_election_polling_consensus`. This is how
`verify_eve_rule.py` produced every number below. Production code will expose
the rule under a new name and keep the current rule selectable, so archived
forecasts stay reproducible. The noise law `pp_centered_noise` was retained
for the same reason.

## 3. Effect

Polls removed from the 14-day window (dates are `interview_end` /
`publication_date`):

| election | removed by the amended rule | consensus changed |
| --- | --- | --- |
| 2002 | SVT VALU, Demoskop *valdag*, Zapera exit (2002-09-15) | **no** (see §4) |
| 2006 | SVT VALU, Demoskop election day (2006-09-17) | yes |
| 2010 | SVT VALU (2010-09-19) | yes |
| 2014 | SVT VALU (2014-09-14) | yes |
| 2018 | SVT VALU, TV4 (2018-09-09); Skop (2018-09-07 / 2018-09-09) | yes |
| 2022 | SVT VALU, Sifo election day (2022-09-11) | yes |
| 2026 | SVT VALU, Novus (2026-09-13) | yes |

Fitting `pp_lw_gaussian` (centred, Ledoit–Wolf) to the residuals, nine
categories:

| pool | rule | total variance (pp²) | sd M | sd S | sd SD |
| --- | --- | --- | --- | --- | --- |
| 2002–2022 (K = 6) | current | 7.29 | 1.09 | 0.91 | 0.99 |
| 2002–2022 (K = 6) | amended | **8.90** | 1.14 | 1.05 | 1.05 |
| 2002–2026 (K = 7) | current | 8.12 | 1.24 | 1.00 | 1.10 |
| 2002–2026 (K = 7) | amended | **9.64** | 1.24 | 1.12 | 1.13 |

On the production pool (2002–2022), the amended rule raises total noise
variance by 22 %. Residual norms:

| election | current (pp) | amended (pp) |
| --- | --- | --- |
| 2018 | 4.69 | 5.51 |
| 2022 | 2.54 | 2.85 |
| 2026 | 3.21 | 3.40 |
| 2014 | 4.24 | 3.90 |

2014 falls: there the exit poll was further from the result on SD than the
final polls were.

## 4. Is every election defined identically?

**Yes, under the amended rule, and 2002 is unchanged by it.**

2002 does have election-day rows in the window (SVT VALU, Demoskop *valdag*,
Zapera exit). The current code never used them, but not because of its date
test. All three have **no sample size**. `consensus.py` pivots with
`sample_size` among the pandas index keys, and pandas drops rows whose index
key is missing. So 2002 was free of election-day polls by accident. Under the
amended rule the predicate itself excludes them. The 2002 consensus is
byte-identical under both rules (`2002 eve_rule_identical_to_current`), and
all seven elections are now defined by one explicit rule.

## 5. A second defect, found during verification, NOT changed by this amendment

The same side effect silently drops **any** poll with a missing sample size
or a missing fieldwork start. `compute_poll_weight` is written to give a
missing-size poll weight 1.0, but no such poll ever reaches it. Under the
amended rule this changes the consensus in three elections:

| election | dropped poll | effect |
| --- | --- | --- |
| 2002 | Skop's latest poll (fieldwork to 2002-09-13, no `n`) | an older Skop poll (to 2002-09-12, n = 1,572) is used instead |
| 2006 | Ipsos / Synovate Temo (to 2006-09-16, no fieldwork start) | Ipsos's earlier poll (to 2006-09-14) is used instead |
| 2022 | Infostat (to 2022-09-07, no `n`) | Infostat is missing from the consensus |

This amendment deliberately leaves that behaviour as it is, so that this
amendment's effect is attributable on its own. **It must be decided before
T3 of `docs/calibration_2030_preregistration.md` is frozen.** Either:

- **(a)** record it as intended behaviour; or
- **(b)** fix it in a separate amendment 002, so that every eligible poll
  reaches deduplication: missing `n` gets weight 1.0 as written, and a
  missing start sorts last in the tie-break.

## 6. What this amendment does and does not do

It **does**, once in effect:

- define the residual consensus for every election by the rule in §2;
- make the corrected residuals the residual definition for every evaluation
  in `docs/calibration_2030_preregistration.md` (T3 and T4).

It **does not**:

- change production forecasts, `MODEL_VERSION` or the adopted noise law.
  Production adopts the corrected residuals only with a later model release,
  after T4;
- rewrite any recorded residual, replay, benchmark capture, archived forecast
  or prospective-benchmark artifact. All keep their bytes, hashes and
  eligibility, and remain correct records of what was computed under the rule
  then in force;
- add 2026 to the production pool. That is decided in T3's final refit;
- change the 14-day window, deduplication, weights or REST.

## 7. Consumers of `build_election_polling_consensus`

The rule applies where the residual definition is used for the 2030
programme:

- the T3 and T4 harnesses;
- when a model release adopts it, both production consumers together:
  - `scripts/election_layer_v2/residuals_pool.py`, the production pool;
  - `scripts/forecast_history/effective_inputs.py`, which fingerprints the
    training residual polls a forecast actually used. It must use the same
    rule as the pool, or history reuse would match forecasts to the wrong
    inputs.

Historical analyses and frozen harnesses keep the rule they were run with:

- `scripts/election_layer/*`
- `scripts/election_residuals/residuals.py`
- `scripts/vote_share_calibration/forward_eval.py`
- `scripts/election_layer_v2/forward_eval.py`
- `scripts/pop_baseline/*`
- `diagnostics/election_noise_v2/*`
- `tests/test_isolated_seat_path.py`
