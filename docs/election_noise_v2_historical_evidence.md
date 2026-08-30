# ElectionNoise v2 — Part 2: historical data feasibility and evidence-base expansion

**Research only.** No production data was added, altered or replaced. RC1, `current.json`,
the prospective archive, the production residual pool, the geography module and the
production mandate allocator are all untouched. No challenger was implemented or scored.

| | |
|---|---|
| Preregistration | **FROZEN — AMENDMENT 1** |
| Freeze commit | `80b1c671c4b6d879a888f28a859ee392e8f59bc5` |
| Body SHA-256 | `bac3ca06e52cc07fe74ca9e5aa785d94e30934db32193c7f948e95a49a6ae075` |
| Whole-file SHA-256 | `03dc843bd73d12c51a8deb7f727a7a4a29a198ee8513d0df5dd8c1fd309e5a97` |
| Reproducible audit | `diagnostics/election_noise_v2/historical_evidence/audit_historical_feasibility.py` → `findings.json` |
| Provenance manifest | `diagnostics/election_noise_v2/historical_evidence/manifest.json` |

---

## A1. The current residual contract

Read from the implementation, not the documentation. Where documentation and code could
differ, the code is quoted. Sources: `scripts/election_residuals/consensus.py`,
`scripts/election_residuals/config.py`, `scripts/elections/load.py`,
`scripts/election_layer_v2/residuals_pool.py`.

| Element | Authoritative rule (code) |
|---|---|
| Final-poll window | `LOOKBACK_WINDOW_DAYS = 14`. Window is `[E − 14 days, E]`. |
| Eligible poll | **All three** must hold: `interview_end ≥ E − 14`, `interview_end ≤ E`, `publication_date ≤ E`. A poll with a null `interview_end` fails the comparison and is silently dropped. |
| Latest-per-pollster | After eligibility, polls are sorted **descending** on `(interview_end, publication_date, interview_start, sample_size, poll_id)` and `drop_duplicates(subset=["pollster"], keep="first")`. `poll_id` is the final deterministic tie-break. |
| Publication-date handling | Used only as an anti-leakage filter (`≤ E`) and as tie-break rank 2. Never used as a fieldwork proxy. |
| Fieldwork-date handling | `interview_end` is **mandatory** (it is the window variable). `interview_start` is optional: it is retained, used only as tie-break rank 3, and may be null. |
| Sample-size weighting | `compute_poll_weight`: `clip(sqrt(n / 1000), 0.7, 1.5)`. |
| Missing `n` | Not an exclusion. Weight is set to `1.0` and `sample_size_missing = True`. |
| Clipping | Weight clipped to `[0.7, 1.5]`. `SAMPLE_SIZE_BENCHMARK = 1000`. |
| Party mapping | Fixed 8: `M, L, C, KD, S, V, MP, SD`. Canonical 9 add `REST`. Ordering `ALL_CATEGORIES = (M, L, C, KD, S, V, MP, SD, REST)` is used everywhere. |
| Per-party consensus | Weighted mean **over only those retained polls that reported that party**: `Σ w·x / Σ w`, rounded to 4 dp. |
| Party reported by no retained poll | `consensus[p] = 0.0` — a **structural zero**, not a measurement. This branch is live in the frozen pool (see 2002/SD below). |
| REST (consensus side) | `REST = 100 − Σ(8 parties)`, rounded to 4 dp. Hard error if `< −0.01`; clamped to 0 if in `[−0.01, 0)`. |
| Consensus sum check | `abs(Σ − 100) > 0.001` raises. |
| Election-result source | `data/processed/elections/riksdag_election_results.csv` via `load_election_targets_for_forecasting`. Exact integer votes; `FI` and `OTHER` are summed into `REST`; the integer sum must equal `valid_votes_total` exactly or it raises; shares are `votes / valid_total × 100`. |
| Residual definition | `r_e = target_vector − consensus_vector`, elementwise, in **percentage points** over the 9 categories. |
| Representation | Percentage points, **not** compositional/CLR. `docs/election_layer_v2.md` §1 records CLR as tested and rejected. |
| Zero-sum enforcement | `abs(Σ r) > 0.05` raises; residue `> 1e-12` is removed by subtracting `Σr / 9` from every component. |
| Centering stage | Separate and later: `mean_bias = mean_e(r_e)` (zero-sum cleaned), `c_e = r_e − mean_bias`. Centering is over the **chronological pool for the target year**, `{e : year(e) < target}`. |
| Mandatory fields | `poll_id`, `pollster`, `interview_end`, `publication_date`, `party`, `support`. |
| Optional fields | `interview_start`, `sample_size`. |
| Pollster exclusions | **None.** There is no allow-list or block-list; every pollster present in the file competes for its latest-in-window slot. |

Two consequences matter for everything below:

1. **`interview_end` is load-bearing.** A poll with no fieldwork end date cannot enter the
   consensus under any circumstances, regardless of how well dated its publication is.
2. **The object is a multi-house consensus.** Deduplication to one poll per pollster,
   followed by a weighted mean across pollsters, is the mechanism by which house effects
   are averaged out. With one retained pollster the computation still runs, but the
   quantity produced is a single house's error, not a consensus error.

---

## A2. Inclusion rules — frozen before any candidate year was judged

Declared here in advance and applied uniformly. Rules are calibrated against **the frozen
production pool itself**, so that no candidate year is held to a standard the incumbent
six are not.

### Mandatory (any failure ⇒ REJECT)

| | Requirement |
|---|---|
| **M1** | Certified official election result from an authoritative statistical or electoral authority, with exact integer votes and a valid-vote total that reconciles exactly. |
| **M2** | **At least one poll eligible under the unmodified production filter.** The 14-day window may not be widened, narrowed or shifted for any election. |
| **M3** | Every retained poll has a non-null, non-contradictory `interview_end` and `publication_date`. Fieldwork dated *after* publication is a corrupt record and is not repairable by inference. |
| **M4** | Deterministic mapping of every reported party into the canonical 9 categories. **No invented support values**, no imputed fieldwork dates, no reconstructed party splits. |
| **M5** | Consensus sums to 100% within the production tolerance with `REST ≥ 0`; residual is zero-sum within the production tolerance. |
| **M6** | No election-specific parameter tweak of any kind. |
| **M7** | Provenance: an identifiable pollster per retained poll, and a citable source for every value. Manual transcription requires the source artifact plus an independent total check. |

### Quality (determines ACCEPT vs ACCEPT WITH DOCUMENTED LIMITATION vs REJECT)

| | Requirement | Failure consequence |
|---|---|---|
| **Q1** | **≥ 3 distinct retained pollsters.** The frozen pool ranges 5–10. | 2 pollsters ⇒ ACCEPT WITH LIMITATION. **1 pollster ⇒ REJECT**: with a single house the house effect is fully confounded with the poll-to-election error, so the observation is drawn from a materially different sampling distribution than every incumbent atom. Under a uniform K-atom bootstrap each atom carries `1/K` of the mass, and Part 1 showed a single atom can dominate a coalition tail — so a categorically different atom is not admissible. |
| **Q2** | All 8 modeled parties reported by ≥ 1 retained poll. | A structural zero is a **documented limitation** if the affected party's official result is `≤ 2.0 pp`; ⇒ REJECT above that. The bound is set from the incumbent pool: 2002 already carries a structural zero for SD whose official result was 1.44 pp, so a stricter bound would retroactively disqualify a frozen pool member. |
| **Q3** | Sample size present for ≥ 80% of retained polls. | Below ⇒ documented limitation. |
| **Q4** | Consensus recency comparable to the pool (mean age 2.4–3.7 days, max 13). | Materially older ⇒ documented limitation. |
| **Q5** | Category semantics comparable — no non-modeled party inside `REST` large enough to change what `REST` means. | A `REST` dominated by a party that won parliamentary seats ⇒ REJECT: the pp transfer would move mass in a category whose meaning differs from the target year's. |

### Edge cases — decided in advance, not per election

| Situation | Decision |
|---|---|
| Missing `n` | Not disqualifying (production sets weight 1.0). Counts against Q3. |
| Missing fieldwork dates | **Disqualifying for that poll** (M3). Not imputable from publication date. |
| Fieldwork dated after publication | **Corrupt record, excluded** (M3). A year correction is inference, not evidence, unless an authoritative source states the true dates. |
| Pollster renames / mergers (TEMO→Ipsos, Demoskop/Inizio) | Use the repository's existing canonical `pollster` field. A rename is one house; a genuine merger of two previously distinct houses collapses two slots into one and is recorded as a limitation. |
| Party did not yet exist | Consensus takes the structural-zero branch; subject to Q2. |
| Party splits / mergers | Only a **deterministic, officially documented** mapping is allowed (e.g. KDS→KD, FP→L, VPK→V). Any judgement-based apportionment ⇒ REJECT. |
| Predecessor party without a clean mapping | REJECT. |
| Support reported as "<x" | Not a number; treated as unreported. |
| Poll reporting a partial party set | Allowed — production averages per party over reporting polls only. Subject to Q2. |
| Archive table with no underlying poll metadata | Fails M3/M7. |

---

## A3. Backward search

### What already exists in the repository

The repository is itself a primary finding. It already contains, from earlier and
**outcome-blind** work:

* **Certified 1991/1994/1998 national results** from SCB Statistikdatabasen `ME0104T3`
  (`scripts/threshold_events/election_results.py`, archived to
  `data/raw/threshold_events/official_election_results_archive.json`). Integer votes for
  all parties incl. `NYD`, with valid-vote totals. Verified: the 1991 party votes sum to
  `5,470,761` exactly, matching the stated total. **M1 is satisfied for all three years.**
* **Polls back to 1973** in `data/processed/pollofpolls/swedishpolls_individual_polls.csv`
  (2,640 polls; `interview_end` from 1973-02-18). **The polling data was never the
  missing piece — its metadata is.**
* Pre-existing inclusion verdicts in `scripts/threshold_events/config.py`, made for a
  different study before this research program existed: 1991 `EXCLUDE_NO_POLLS`, 1994
  `INCLUDED` (LOW, single pollster), 1998 `EXCLUDE_MISSING_DATES`. These were treated as
  hypotheses and re-verified from data below, not taken on trust.

### Applying the unmodified production filter

`audit_historical_feasibility.py`, using the frozen `build_election_polling_consensus`
eligibility rule verbatim:

| Election | Eligible polls in `[E−14, E]` | Distinct pollsters | Polls published in `[E−120, E]` | Missing `interview_end` | Missing `n` | Corrupt dates |
|---|---|---|---|---|---|---|
| 1991-09-15 | **0** | 0 | 5 | 0 | 0 | 0 |
| 1994-09-18 | **1** | **1** (Ipsos/TEMO) | 6 | 0 | 0 | 0 |
| 1998-09-20 | **0** | 0 | 13 | **5** | **5** | **1** |
| 2002-09-15 | 44 | 5 | 90 | 2 | — | — |
| 2006-09-17 | 25 | 5 | — | — | — | — |
| 2010-09-19 | 27 | 7 | — | — | — | — |
| 2014-09-14 | 23 | 9 | — | — | — | — |
| 2018-09-09 | 35 | 10 | — | — | — | — |
| 2022-09-11 | 47 | 7 | — | — | — | — |

### Verification at the upstream source

`MansMeg/SwedishPolls` `Data/Polls.csv` was downloaded and inspected directly
(sha256 `4cd11c83…`, retrieved 2026-08-30T20:30:10Z) to establish whether the gaps are a
local ingestion defect or a genuine source limitation. They are genuine:

* **1998** — the four Demoskop polls published 1998-08-20, 08-26, 09-02, 09-09 and 09-16
  have `collectPeriodFrom = collectPeriodTo = NA` **and** `n = NA` *upstream*. The single
  TEMO poll published 1998-09-19 carries `collectPeriodFrom = 1999-09-07`,
  `collectPeriodTo = 1999-09-17` — fieldwork dated a full year **after** publication —
  *upstream*. The last correctly dated poll is Sifo, fieldwork ending 1998-08-31 = **E−20**,
  outside the window. The repository's ingestion is faithful: its 13 run-up polls match the
  upstream 13 exactly.
* **1991** — the final poll is TEMO, fieldwork 1991-08-19…08-28 = **E−18**, published
  09-13. Nothing at all falls inside `[E−14, E]`.
* **1994** — one poll inside the window: TEMO, fieldwork 09-11…09-14 (E−4), published
  09-16, `n = 1358`. Correct and complete, but alone.
* Upstream documentation independently states that pre-2000 records carry more NAs in
  sample size and collection period and that quality improves from 2008 — corroborating
  that this is an era-wide archival limitation, not a per-election accident.

A further structural fact from the upstream schema: the `SD` column is `NA` for **every**
pre-2000 poll, and there is **no `NYD` column at all**. In 1991 Ny Demokrati won 6.73% of
the vote and 25 Riksdag seats while being entirely absent from the poll instrument.

### Verdicts

| Year | Verdict | Binding reason |
|---|---|---|
| **1998** | **REJECT** | **M2 and M3.** Zero polls have a valid in-window `interview_end`. Four candidate polls have no fieldwork dates upstream; the one poll that would fall in-window has fieldwork dated after publication. Admitting it would require imputing dates (M4) or asserting a year correction the source does not support (M3). |
| **1994** | **REJECT** | **Q1.** M1–M7 are all satisfied — this is the one candidate that clears every mandatory rule — but the consensus retains a **single** pollster against 5–10 for every incumbent atom. The house effect is fully confounded with the polling error, and the resulting vector is a one-house error, not a consensus error. Also fails Q2 (SD structurally unpolled) and Q5 in 1994's milder form. |
| **1991** | **REJECT** | **M2** (zero eligible polls; the nearest is E−18) **and Q5** (`REST` in 1991 is ≈7.6% and is dominated by Ny Demokrati, a party that won 25 seats — a categorically different meaning of `REST` from the ≈1.5–4% non-parliamentary residue of 2002–2022). |
| **pre-1991** | **REJECT, not searched further** | The binding constraint is monotone: poll frequency and metadata completeness decline going backwards, while the window rule stays fixed. 1991 already has zero in-window polls; no earlier election can do better. Searching further would be theatre. |

**Per A4, no candidate residual vector was reconstructed for any rejected year.** Computing
a residual for a rejected election and then reporting it would invite exactly the post-hoc
reasoning the preregistration exists to prevent. The verdicts above rest solely on the A2
criteria.

---

## A5 / Track A consequence

Older elections are potential *training* residuals only; the Tier-1 candidate target set is
frozen at `{2010, 2014, 2018, 2022}` and did not change.

**No pre-2002 election was accepted.** The residual pool is unchanged at
`{2002, 2006, 2010, 2014, 2018, 2022}`, **K = 6**, and the 2010 target keeps
`K_outer = 2 < 3`, so it remains ineligible. **`N_T1` remains 3.**

---

## Track B — end-to-end seat evaluation

### B1. Seat-data inclusion rules, frozen before searching

| | Requirement |
|---|---|
| **S1** | Authoritative constituency-level party votes for the **previous** election (the geography baseline), with constituency valid-vote totals. |
| **S2** | Authoritative constituency-level and national results for the **target** election, reconciling exactly to the certified national totals. |
| **S3** | Deterministic mapping into the 9 model categories, with `REST` constructed the same way as production. |
| **S4** | Constituency definitions directly compatible between baseline and target, or mappable by a deterministic, officially documented transformation. |
| **S5** | Exact certified mandate outcome for the target (per party nationally; per constituency preferred). |
| **S6** | The mandate-allocation law actually in force for the target is known exactly and unambiguously from official sources. |
| **S7** | No geography parameter changed to force agreement; no hand-adjusted constituency inputs; every reconciliation difference explained. |
| Tolerance | Vote reconciliation must be **exact (0 votes)** at national and per-party level; seats must sum to exactly 349 and match the certified per-party vector exactly. Any unexplained mismatch blocks ACCEPT. |

### B2 / B3. Findings for the 2014 and 2010 targets

**The law is the binding constraint, and it is the same one for both.**
Proposition 2013/14:48, enacted as **SFS 2014:1384**, made two changes effective from the
**2018** general election:

1. the modified Sainte-Laguë **first divisor changed from 1.4 to 1.2**; and
2. **mandate return (`återföring`) was introduced** — Vallagen 14 kap. 4a–4c §§.

Under the pre-2018 law that governed **both 2010 and 2014**, the first divisor was **1.4**,
and there was **no return mechanism**: a party that won more fixed constituency seats than
its national proportional entitlement simply kept them, and that party together with its
fixed seats was excluded from the adjustment-seat distribution.

The production allocator implements `Fraction(6, 5) = 1.2` plus the `återföring` loop
(`docs/riksdag_mandate_allocation.md` §3.1 states the 1.2 divisor is "effective since
Jan 1, 2015 via SFS 2014:1384; applies to 2018, 2022, 2026"). It is therefore **legally
wrong for 2010 and 2014**, and neither year is usable with the existing allocator exactly.

Data availability, by contrast, is largely favourable:

| Input | 2014 target | 2010 target |
|---|---|---|
| Fixed constituency seats | **Already in repo** — `data/raw/mandates/valkretsmandat_riksdag_1988_2026.xlsx` (Valmyndigheten) gives 1988–2026, incl. 2010 and 2014, summing to 310. | **Already in repo**, same source. |
| Target constituency party votes | **Already in repo** — `data/processed/geography/constituency_party_votes_2014_2022.csv`, 29 constituencies, 9 categories. **Reconciles exactly**: 0-vote difference against the official national total (`6,231,573`) and against every one of the 9 party totals. | **Must be acquired** — available at `historik.val.se/val/val2010/slutresultat/R/rvalkrets/NN/index.html` with party votes and `Giltiga röster` per constituency (29 pages). |
| Geography baseline (previous election) | **Must be acquired** — 2010 constituency votes, same source as above. | **Must be acquired** — 2006 constituency votes, `historik.val.se/val/val2006/...` (29 pages). |
| Certified per-constituency mandates | **Must be acquired** — `.../rvalkrets/NN/valda.html`; verified present for 2010 (e.g. Stockholms kommun 2010: M 10, C 2, FP 3, KD 2, S 6, V 2, MP 3, SD 1). **No fixed/adjustment split is published on these pages**, so per-phase allocator validation would be weaker than the 2018/2022 case. | Same. |
| Constituency structure | **Compatible.** 29 constituencies, stable since 1998. The workbook's own footnotes list boundary changes only for 1994, 1998, 2006 (Heby moved to Uppsala) and 2018 (Västra Götaland internal boundaries). **2010→2014 spans no boundary change.** | **Compatible.** 2006→2010 spans no boundary change; the Heby move predates the 2006 baseline. |
| Law | **1.4, no `återföring`** | **1.4, no `återföring`** |

### B4. Allocator classification

| Target | Class | Justification |
|---|---|---|
| 2018, 2022 | **A — usable with the existing allocator exactly** | Already the frozen Tier-3 set; the allocator reproduces both certified results with 0 mismatches. |
| **2014** | **B — usable if a historically versioned allocator is implemented from unambiguous official rules** | Requires first divisor `Fraction(7,5)` and suppression of the `återföring` path. Both are officially documented (prop. 2013/14:48; SFS 2014:1384). The existing allocator already accepts `Fraction(7,5)` via `_normalize_first_divisor`, so the change is a configuration plus a legal-path switch, not a new algorithm. |
| **2010** | **B — same** | Same law, same mechanism. Additional data acquisition burden (both the 2006 baseline and the 2010 target must be scraped). |

No historically versioned allocator was implemented, and no proof-of-concept was needed:
the classification follows from the legal sources alone. **A historically versioned
allocator is not a scientific challenger** — it exists only to reproduce the law in force —
and building it is Part-3 work, gated on the specification and provenance assembled here.

### B5. Geography compatibility

The frozen geography method (2022-baseline IPF raking onto target constituencies, then
exact-margin biproportional controlled rounding) is **specification-compatible** with 2010
and 2014: 29 constituencies in both baseline and target, no boundary change across either
baseline→target pair, the same 9 categories, the same `REST` construction, and
constituency valid-vote totals published for every constituency. What is required is
**historical data normalization only** — scraping and mapping — not a different geography
model. No special geography model is needed, and none is proposed.

Two caveats to carry forward: (i) party availability differs — SD is present throughout
2006–2014, but per-constituency zero cells are more likely for small parties in the older
years and must be checked before controlled rounding; (ii) the 2018 Västra Götaland
boundary change means a 2014→2018 baseline pair is *not* boundary-clean, which is a
pre-existing property of the frozen 2018 case and not something this task may revisit.

### B6. Reconciliation performed

| Check | Result |
|---|---|
| 2014 constituency votes → official national valid votes | **Exact**, 0 difference (`6,231,573`). |
| 2014 constituency votes → official per-party national votes | **Exact**, 0 difference for all 9 categories. |
| 2014 constituency valid-vote totals → national valid votes | **Exact**, 0 difference. |
| 2018, 2022 (control) | **Exact**, 0 difference (all categories). |
| 1991 SCB party votes → stated valid total | **Exact**, `5,470,761`. |
| 2010 / 2006 constituency reconciliation | **Not performed — data not yet acquired.** Blocks ACCEPT for both years. |
| 2010 / 2014 certified per-party seat vectors | **Not acquired.** Blocks ACCEPT. |

Under S1/S2/S5 an unacquired input is an unmet requirement, so neither year can be ACCEPTed
in this task. Both are **DEFERRED**, not rejected.

---

## Track C — descriptive comparability of the existing residual history

Descriptive only. No trend is fitted, no recency weighting is implied, no residual is
altered, and 2002 is not excluded or down-weighted.

| Year | Retained pollsters | Eligible polls | Σ sample size | Missing `n` | Mean consensus age (d) | Max age | All 8 parties polled | Mean abs residual (pp) | Max abs residual (pp) | Residual L2 (pp) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2002 | 5 | 44 | 7,639 | 0 | 3.20 | 4 | **No — SD unpolled** (official 1.44%) | 1.508 | 3.474 | 5.446 |
| 2006 | 5 | 25 | 18,447 | 0 | 2.80 | 5 | Yes | 0.643 | 1.303 | 2.280 |
| 2010 | 7 | 27 | 22,231 | 0 | 2.43 | 4 | Yes | 0.548 | 1.321 | 2.104 |
| 2014 | 9 | 23 | 25,558 | 0 | 2.67 | 6 | Yes | 1.190 | 2.885 | 4.237 |
| 2018 | 10 | 35 | 32,739 | 0 | 3.70 | 12 | Yes | 1.234 | 3.165 | 4.694 |
| 2022 | 7 | 47 | 26,357 | 0 | 3.14 | 13 | Yes | 0.721 | 1.486 | 2.542 |

Observations, recorded as future research questions only:

1. **No era trend in residual magnitude.** Year vs residual L2 norm: Spearman −0.086,
   Pearson −0.194 (n = 6). The two largest residuals are 2002 and 2018 — not adjacent in
   time — and the two smallest are 2010 and 2006. There is **no descriptive support for
   recency weighting**, and none is proposed.
2. **Polling effort grew.** Summed sample size roughly quadrupled from 2002 to 2018;
   pollster count moved 5→10. Sample-size coverage is 100% throughout.
3. **2002 is the one incumbent atom with a structural zero** (SD unpolled, consensus 0.000
   vs official 1.44%). Part 1 showed 2002 is also the atom that produces essentially all
   C+S+MP majority draws. These two facts are **not shown to be causally related** — 2002's
   large residual is dominated by M (−2.88 pp) and S (+3.47 pp), not by SD — but the
   coincidence is recorded because it is the kind of thing a later reader would want
   flagged. It is **not** a reason to touch 2002, and nothing here does.
4. **Mode/methodology** is not documented per poll in the repository schema (no telephone /
   web / mixed field), so the mode transition across the 2002–2022 era cannot be described
   from available data. Recovering it would require pollster-level archival work.

---

## TABLE 1 — Residual history

| Year | Source quality | Poll coverage (14-day window) | Algorithm comparability | Status | Reason | Candidate residual available? | Usable as training residual? |
|---|---|---|---|---|---|---|---|
| 1991 | Result: HIGH (SCB `ME0104T3`, exact). Polls: LOW | **0 polls, 0 pollsters**; nearest E−18 | Incompatible — no consensus computable; `REST` ≈7.6% dominated by NYD (25 seats) | **REJECT** | M2 (no eligible poll) + Q5 (`REST` semantics) | No — not computed | **No** |
| 1994 | Result: HIGH (SCB, exact). Polls: LOW | **1 poll, 1 pollster** (Ipsos/TEMO, E−4, n=1358) | Computable but not a consensus: single house, SD structurally unpolled | **REJECT** | Q1 (single pollster ⇒ house effect fully confounded) | No — not computed | **No** |
| 1998 | Result: HIGH (SCB, exact). Polls: metadata UNUSABLE | **0 polls, 0 pollsters**; 4 Demoskop with no fieldwork dates and no `n` upstream, 1 TEMO with fieldwork dated after publication | Not computable without imputing dates | **REJECT** | M2 + M3 | No — not computed | **No** |
| 2002 | HIGH (Valmyndigheten + 5-house consensus) | 44 polls, 5 pollsters | Reference | **INCUMBENT** | Frozen production pool | Yes (in production) | **Yes** |
| 2006 | HIGH | 25 polls, 5 pollsters | Reference | **INCUMBENT** | Frozen production pool | Yes | **Yes** |
| 2010 | HIGH | 27 polls, 7 pollsters | Reference | **INCUMBENT** | Frozen production pool | Yes | **Yes** |
| 2014 | HIGH | 23 polls, 9 pollsters | Reference | **INCUMBENT** | Frozen production pool | Yes | **Yes** |
| 2018 | HIGH | 35 polls, 10 pollsters | Reference | **INCUMBENT** | Frozen production pool | Yes | **Yes** |
| 2022 | HIGH | 47 polls, 7 pollsters | Reference | **INCUMBENT** | Frozen production pool | Yes | **Yes** |

**Resulting residual pool: `{2002, 2006, 2010, 2014, 2018, 2022}` — K = 6 (unchanged).**

## TABLE 2 — Tier-1 consequence

Frozen candidate target set `{2010, 2014, 2018, 2022}`; eligibility rule `K_outer ≥ 3`.

| Target | Prior accepted residual years | `K_outer` | Eligible? |
|---|---|---|---|
| **2010** | 2002, 2006 | **2** | **No** |
| **2014** | 2002, 2006, 2010 | **3** | Yes |
| **2018** | 2002, 2006, 2010, 2014 | **4** | Yes |
| **2022** | 2002, 2006, 2010, 2014, 2018 | **5** | Yes |

### **N_T1 = 3** (eligible: 2014, 2018, 2022)

The `N_T1 = 3` column of the frozen §F.3 G5 table applies: all 3 leave-one-target-out
aggregates must favour the challenger, at least 2 of 3 by ≥ 1.0%, and at least 2 of 3
individual targets must favour it. `N_T1 = 3` is **not** below the `N_T1 < 3 ⇒ STOP`
threshold, so the competition remains viable.

## TABLE 3 — Seat history

| Target | Required baseline | Baseline data status | Target official data status | Law compatibility | Geography compatibility | Allocator requirement | Status |
|---|---|---|---|---|---|---|---|
| **2010** | 2006 constituency votes | **Not in repo**; available at `historik.val.se/val/val2006/…` (29 pages) — not acquired | Constituency votes not in repo (available); certified per-constituency mandates available without fixed/adjustment split; not acquired | **Incompatible** with production allocator — 2010 used first divisor **1.4** and had **no `återföring`** | Compatible — 29 constituencies, no boundary change 2006→2010 | **Historically versioned allocator required** (class **B**) | **DEFERRED** |
| **2014** | 2010 constituency votes | **Not in repo**; available at `historik.val.se/val/val2010/…` (29 pages) — not acquired | **Constituency votes already in repo and reconcile exactly (0 votes)**; certified per-constituency mandates available, not acquired | **Incompatible** — 2014 used **1.4**, no `återföring` | Compatible — 29 constituencies, no boundary change 2010→2014 | **Historically versioned allocator required** (class **B**) | **DEFERRED** |
| **2018** | 2014 constituency votes | In repo | In repo, certified | Compatible (post-SFS 2014:1384) | Compatible | None (class **A**) | **ACCEPTED (incumbent)** |
| **2022** | 2018 constituency votes | In repo | In repo, certified | Compatible | Compatible | None (class **A**) | **ACCEPTED (incumbent)** |

### **N_seat = 2** (2018, 2022)

The frozen `N_elections == 2` branch of the coalition-Brier rule applies: aggregate Brier
must improve ≥ 2%, **and** the challenger must beat CONTROL in **both** 2018 and 2022
separately.

## TABLE 4 — Data provenance

Archived under `diagnostics/election_noise_v2/historical_evidence/raw/`.

| Source | URL | Retrieved (UTC) | SHA-256 | Purpose | Years |
|---|---|---|---|---|---|
| MansMeg/SwedishPolls `Data/Polls.csv` | `https://raw.githubusercontent.com/MansMeg/SwedishPolls/master/Data/Polls.csv` | 2026-08-30T20:30:10Z | `4cd11c83e76d7b37425956d9d9944dafa86fcf273408207cd32f3d670a2253d8` | Verify at source whether 1991/1994/1998 fieldwork dates and sample sizes are recoverable | 1972–2026 |
| Valmyndigheten, *Mandatfördelning* manual (VAL V785 05) | `https://www.val.se/download/18.162047b519a91d05331183a9/1761747515752/manual-mandatfordelning-val-v785-05.pdf` | 2026-08-30T20:32:53Z | `e3a7661a58bf97abc4c0386a1cb93a5d170e9027f3dabd819d5931dfafa7e34f` | Authoritative statement of the currently-in-force allocation rules | 2018– |
| Regeringen, Prop. 2013/14:48 | `https://www.regeringen.se/contentassets/3bb36f6f751d4ed1ae9ada6aeab9f5ac/proportionell-fordelning-av-mandat-och-forhandsanmalan-av-partier-i-val-prop.-20131448` | 2026-08-30T20:33:07Z | `5aa84ff21840515a126928dd1300f847752ca2ed4c497ecc4115bee5bcf923cb` | Primary legal source for 1.4→1.2 and the introduction of `återföring`, effective 2018 (SFS 2014:1384) | law |

Cited but not archived (no transcription taken from them): the Valmyndigheten historical
results pages for 2010 national / constituency / elected, the Valmyndigheten index of past
results, the SFS 2014:1384 text at riksdagen.se, and one Linköping University page used
**only** as secondary corroboration of the pre-2018 algorithm. Full list with roles in
`manifest.json`.

Already in the repository and **not re-downloaded or modified**: the Valmyndigheten
`valkretsmandat_riksdag_1988_2026.xlsx` (fixed seats 1988–2026 and the authoritative record
of constituency boundary changes) and the SCB `ME0104T3` 1991/1994/1998 results archive.
No manual transcription was performed anywhere in this task.

---

## FINAL DECISION

# PROCEED_WITH_EXISTING_HISTORY

**1. Which pre-2002 residual elections, if any, qualify?**
None. 1998 and 1991 fail mandatory rules (no poll with a valid in-window fieldwork end
date). 1994 clears every mandatory rule but fails quality rule Q1: its consensus retains a
single pollster, so its house effect is fully confounded with the polling error, making it
an atom drawn from a materially different distribution than the incumbent five-to-ten-house
atoms. Pre-1991 was not searched further because the binding constraint is monotone in time.

**2. Resulting residual K?** **K = 6**, unchanged: `{2002, 2006, 2010, 2014, 2018, 2022}`.

**3. Resulting `N_T1`?** **`N_T1 = 3`** (2014, 2018, 2022). 2010 remains ineligible at
`K_outer = 2`. This is at, not below, the `N_T1 < 3 ⇒ STOP` threshold, so the competition
proceeds under the frozen `N_T1 = 3` instantiation.

**4. Can 2014 become an end-to-end seat evaluation case?**
Not yet, but it is the nearest. Its constituency votes are already in the repository and
reconcile exactly to the official national totals; its fixed-seat vector is already in the
repository; its constituency structure is boundary-clean against a 2010 baseline. Missing:
the 2010 constituency baseline, the certified 2014 per-constituency mandates, and a
historically versioned allocator. **DEFERRED, class B.**

**5. Can 2010 become one?**
Same legal answer, more data work: both the 2006 baseline and the 2010 target constituency
votes must be acquired, plus certified mandates. **DEFERRED, class B.**

**6. Does either require a historically versioned allocator?**
**Yes — both, for the same two reasons.** SFS 2014:1384 (prop. 2013/14:48) changed the
modified Sainte-Laguë first divisor from **1.4 to 1.2** and introduced mandate **return
(`återföring`)**, both effective from the 2018 election. 2010 and 2014 were held under the
old law: first divisor 1.4, and a party with more fixed seats than its national entitlement
kept them and was excluded, with its fixed seats, from the adjustment distribution. The
production allocator implements the post-2018 law and is legally wrong for both years. The
rules are unambiguous and officially documented, so a versioned allocator is specifiable —
class **B**, not C. The existing allocator already accepts `Fraction(7,5)`, so this is a
legal-path switch plus configuration, not a new algorithm.

**7. Resulting `N_seat`?** **`N_seat = 2`** (2018, 2022), unchanged. The frozen
`N_elections == 2` branch applies: aggregate Brier ≥ 2% improvement **and** the challenger
must beat CONTROL in 2018 **and** 2022 separately.

**8. Important comparability caveats.**
(a) The residual pool remains six observations, and Part 1 showed one atom can dominate a
coalition tail — Part 2 does not relieve that. (b) 2002 carries a structural zero for SD
(consensus 0.000 vs official 1.44%) and is also the atom driving the C+S+MP tail; no causal
link is demonstrated, and nothing was changed. (c) There is no descriptive era trend in
residual magnitude (Spearman −0.086, n = 6), so nothing supports recency weighting.
(d) Polling effort grew markedly across the pool (5→10 houses, sample size ~4×), a real
non-stationarity in *precision* even though residual magnitude shows no trend. (e) Poll
mode/methodology is not recorded in the schema, so era changes in method cannot be
described from available data. (f) The certified per-constituency mandate pages for
2010/2014 do not publish a fixed/adjustment split, so a versioned allocator could be
validated against final per-party seats but not phase-by-phase as 2018/2022 were.

**9. Is missing data realistically recoverable with additional archival work?**
Partly, with clearly different odds per target.
*Seat side — likely.* The 2006 and 2010 constituency votes and the 2010/2014 certified
mandates are all published by Valmyndigheten at stable `historik.val.se` URLs; this is
bounded scraping (≈29 pages per year plus mandate pages) against a primary source, and
would move `N_seat` from 2 to 4 — the single largest available improvement to the evidence
base.
*Residual side — unlikely to change any verdict.* 1991 cannot be rescued at all: no poll
exists inside the window, and the window may not move. 1998 is the only conceivable case —
it would need Demoskop's 1998 fieldwork dates and an authoritative correction of the TEMO
record, from pollster archives or contemporaneous press. Even if both were obtained, the
best attainable is **two** pollsters (TEMO plus Demoskop; Sifo's last poll ended E−20), so
1998 could at most reach ACCEPT WITH DOCUMENTED LIMITATION and would still carry a
structurally unpolled SD. 1994 is not recoverable in principle: no additional archival work
can create a second house that did not poll.

**10. Does anything discovered require stopping before Part 3?**
**No.** `N_T1 = 3 ≥ 3` and `N_seat = 2` both land on frozen, already-specified branches of
the adoption gate; no rule needed inventing and none was invented. Two findings should be
carried into Part 3 as explicit constraints rather than surprises: the seat/coalition
evidence stays at two elections unless the Track-B acquisition is authorised first, and any
future 2010/2014 extension is gated on a historically versioned allocator that must be
built from the official pre-2018 rules and validated before it is used for scoring.

---

## Guardrail compliance

Not edited: the preregistration, RC1, `current.json`, the prospective archive, production
polling data, geography, the production mandate allocator, seeds, sample counts, adoption
gates. No residual was added to the production pool. No challenger was implemented or
scored. No 2026 forecast was run. No recency weighting was introduced. 2002 was neither
removed nor down-weighted. The stale `contributing_polls_audit.csv` maintenance item
remains untouched and separate.

Targeted validation only: the reconciliation checks in
`audit_historical_feasibility.py`. The full test suite was not run.
