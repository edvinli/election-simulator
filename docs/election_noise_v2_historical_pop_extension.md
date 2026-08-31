# ElectionNoise v2 — Part 3B: historical Poll-of-Polls state extension feasibility

**Data/evaluation infrastructure only.** No challenger was implemented or scored, no
2026 challenger forecast was run, no predictive score against the 2014 outcome was
computed, the frozen preregistration was not edited, and the certified Part-3
evaluation manifest was not modified.

| | |
|---|---|
| Base commit | `998a20047cf9bae1e9b8a59d4ec4888684842fd5` |
| Branch | `research/historical-pop-extension` |
| Preregistration | **FROZEN — AMENDMENT 1**, body SHA-256 `bac3ca06e52cc07fe74ca9e5aa785d94e30934db32193c7f948e95a49a6ae075` — unchanged |
| **Decision** | **`HISTORICAL_POP_EXTENSION_REJECTED`** |
| **Resulting `N_seat`** | **2** (unchanged) |

---

## 1. Raw source semantics

| Question | Finding |
|---|---|
| What is `pofp`? | The Poll-of-Polls **own aggregate estimate** line of each first-party party chart (`pollofpolls.se/poll_img/data_big_N.csv`). Confirmed by §3: it is byte-identical to the canonical `data_table_tot.csv` series over their entire 12-year overlap. |
| Date semantics | One row per calendar day, `date` in ISO form, no gaps. |
| Daily estimate, or something else? | A daily aggregate estimate, in percent, per party. |
| Coverage synchronised across the eight parties? | **Yes.** All eight parliamentary parties have identical coverage: 6,444 rows each, 2009-01-02 … 2026-08-24, no missing `pofp` cell. FI is shorter (starts 2014-01-02) and maps to `REST` in the model's nine categories, so it is not required. |
| Earliest / latest usable dates | **2009-01-02 … 2026-08-24** (6,444 days). The canonical production series covers only 2014-09-15 … 2026-08-24 (4,362 days), so the extension would add **2,082 days** of pre-election-2014 history. |
| Rounding / precision | One decimal place, consistent with the canonical series. Derived `REST = 100 − Σ(eight)` is never negative on any of the 6,444 dates. |
| Is the source retrospectively recomputed or revised? | **Yes, within a trailing window.** See §4. |
| Can a historical value contain information from polls published later than the row date? | **Yes.** This is the binding blocker; see §4. |

Two structural facts about the charts matter and are easy to miss:

* The production parser `scripts/pollofpolls/normalize.py::parse_party_chart_payload`
  **explicitly excludes** the `pofp` column
  (`column not in {"date", "pofp", "Val"}`). The frozen pipeline has never consumed
  it. So `pofp` being present in an archived production input does not make it a
  validated production input.
* The per-pollster columns in the same charts are **fieldwork-dated**: the
  production docstring states "A poll's support value is repeated on every date in
  its interview span", and this reproduces exactly. Demoskop/M, 2022:

  | Chart run | Poll interview span | Poll **published** |
  |---|---|---|
  | 2022-08-01 … 08-02, M = 20.3 | 2022-07-24 … 08-02 | 2022-08-04 |
  | 2022-08-31 … 09-04, M = 18.3 | 2022-08-31 … 09-04 | 2022-09-05 |
  | 2022-09-26 … 09-30, M = 20.8 | 2022-09-26 … 10-04 | 2022-10-05 |

  Every chart run **ends before** the poll that produced it was published.

The name `pofp` was not taken as evidence of equivalence with the canonical
series; equivalence was established empirically in §3.

---

## 2. Research-only normalized candidate series

`diagnostics/election_noise_v2/historical_pop_extension/build_candidate_series.py`
produces `processed/candidate_pop_state_2009_2026.csv`.

Construction is purely mechanical, with every prohibited operation absent:

* `pofp` taken **verbatim** from each archived party chart;
* only dates where **all eight** parliamentary parties carry a value — 6,444 of
  6,444, so no date is dropped;
* `REST = 100 − Σ(eight)`, the same rule
  `scripts/pollofpolls/state.py::load_timeseries_dataset` applies to the canonical
  series;
* **no interpolation, no forward filling, no backward filling from later
  observations, no manually entered values, no imputation.**

Provenance in `processed/provenance.json`: per-party source URL, retrieval
timestamp and SHA-256, taken from the existing
`data/raw/pollofpolls/retrieval_manifest.json`, plus the SHA-256 of the canonical
production series recorded as unmodified. The schema matches the canonical
processed timeseries so the frozen loader reads it without change.

`data/processed/pollofpolls/pollofpolls_timeseries.csv` was **not** modified.

---

## 3. Overlap reconciliation

Candidate `pofp` versus the canonical archived source
(`data/raw/pollofpolls/pollofpolls_timeseries_source.dat`, i.e. `data_table_tot.csv`)
over the **entire** overlapping range 2014-09-15 … 2026-08-24.

| Category | Matched dates | Exact matches | Exact-match fraction | Mean abs diff | Max abs diff |
|---|---|---|---|---|---|
| M | 4362 | 4362 | **100.000 %** | 0 | 0 |
| L | 4362 | 4362 | **100.000 %** | 0 | 0 |
| C | 4362 | 4362 | **100.000 %** | 0 | 0 |
| KD | 4362 | 4362 | **100.000 %** | 0 | 0 |
| S | 4362 | 4362 | **100.000 %** | 0 | 0 |
| V | 4362 | 4362 | **100.000 %** | 0 | 0 |
| MP | 4362 | 4362 | **100.000 %** | 0 | 0 |
| SD | 4362 | 4362 | **100.000 %** | 0 | 0 |
| FI | 4361 | 4361 | **100.000 %** | 0 | 0 |
| REST (derived both sides) | 4362 | 4362 | **100.000 %** | 0 | 0 |

**Zero discrepancies of any size, in any category, on any of 4,362 dates.** No
tolerance was needed, so none was chosen — the question of picking a tolerance
that would make 2014 admissible never arose. The two files are two
representations of the same PoP estimate concept, and criterion 2 is satisfied.

---

## 4. Leakage audit — the binding blocker

**Result: leakage is present, with an attributed mechanism.**

### 4.1 The series is retrospectively revised

Two archived snapshots of the *same* series exist in git history, retrieved one day
apart. Comparing every overlapping cell:

| | |
|---|---|
| Snapshots | `f55bf36` (max date 2026-08-23) vs `f6ae4d1` (max date 2026-08-24) |
| Cells compared | 34,888 (4,361 dates × 8 parties) |
| **Cells revised** | **176 (0.50 %)** |
| Max revision | 0.2 pp; typical 0.1 pp |
| Revised date range | **2026-02-15 … 2026-08-23**, 79 distinct dates |
| Revisions older than 2026-02-15 | **none** |

Identical numbers for the raw source and the processed production file, so the
revision originates upstream, not in the repository's parser. The provider
recomputes roughly a **six-month trailing window** and leaves older history stable.

### 4.2 The mechanism, attributed to a specific poll

Take the largest revision: **2026-08-22, MP: 7.5 → 7.3**, between a retrieval on
2026-08-26 and one on 2026-08-27.

Every poll whose **fieldwork covers 2026-08-22**:

| Pollster | Interview span | Published |
|---|---|---|
| Ipsos | 2026-08-11 … 08-23 | **2026-08-25** |
| Indikator | 2026-08-06 … 08-23 | **2026-08-26** |
| Demoskop | 2026-08-13 … 08-24 | **2026-08-27** |

All three were published *after* 2026-08-22, and the Demoskop poll published
exactly on the second retrieval date is the only new information between the two
snapshots. The historical value for 2026-08-22 therefore changed **because a poll
published five days later arrived.**

Combined with §1 — the charts are fieldwork-dated and every chart run ends before
its poll is published — the conclusion is that `pofp(t)` is a **fieldwork-dated
rolling aggregate, retrospectively completed as later-published polls arrive.** A
forecaster standing at date `t` could not have observed the value the archived
series reports for `t`.

This is strong mechanistic evidence rather than mathematical proof: it shows the
observed behaviour is inconsistent with a causal, publication-dated series. An
attempted change-date attribution test was inconclusive and is not relied on —
`pofp` changes on publication-only dates (67.0 %) and on fieldwork-start-only dates
(67.3 %) at indistinguishable rates, because a rolling average moves daily as polls
enter and leave its window regardless of publication events.

### 4.3 Magnitude, for the six 2014 horizons

Polls whose fieldwork covers `as_of` but which were published after it:

| Horizon | `as_of` | Leaking polls | Max lead | Detail |
|---|---|---|---|---|
| 112 | 2014-05-25 | 4 | 15 d | Ipsos +5 d, Novus +4 d, SCB +2 d, United Minds +15 d |
| 84 | 2014-06-22 | 3 | 10 d | Ipsos +6 d, Novus +6 d, United Minds +10 d |
| 56 | 2014-07-20 | 1 | 21 d | YouGov +21 d |
| 28 | 2014-08-17 | 1 | 6 d | Demoskop +6 d |
| 14 | 2014-08-31 | 3 | 6 d | Novus +1 d, Demoskop +5 d, Skop +6 d |
| 7 | 2014-09-07 | 3 | 5 d | Novus +1 d, Sentio +2 d, Ipsos +5 d |
| **total** | | **15** | **21 d** | |

Publication lag after fieldwork end across all 1,181 dated polls: median 3 d,
mean 3.4 d, p95 8 d, max 30 d. The look-ahead is therefore bounded by days to a few
weeks, not by the ~6-month revision window — the 2009–2014 values are long since
outside that window and demonstrably stable.

### 4.4 The other two audited inputs are clean

* **Residual pool.** For target 2014 the chronological pool is exactly
  `{2002, 2006, 2010}`; no year ≥ 2014 enters. Verified.
* **Dynamics.** Every eligible transition satisfies the frozen boundary
  `end_date <= as_of`; the maximum transition end equals `as_of` at all six
  horizons. Verified.
* **OpinionState's individual-poll input** is the unchanged production
  `individual_polls.csv`, and `estimate_opinion` applies its existing
  reference-date filter. The leakage is in the **PoP state series**, not in the
  individual-poll residual inputs.

---

## 5. Pipeline sufficiency

`smoke_2014_pipeline.py` runs the frozen chain — OpinionState v1.1 → Dynamics v2 →
ElectionNoise CONTROL → frozen geography (2010 baseline) → PRE_2018 mandate law —
for all six horizons against the candidate history. **No predictive score against
the certified 2014 outcome was computed.**

| Horizon | `as_of` | OpinionState | Dynamics transitions (exact h) | Fallback | Max transition end | Causal | Seats = 349 | Law |
|---|---|---|---|---|---|---|---|---|
| 112 | 2014-05-25 | OK | 1,858 | none | 2014-05-25 | ✔ | ✔ | PRE_2018 |
| 84 | 2014-06-22 | OK | 1,914 | none | 2014-06-22 | ✔ | ✔ | PRE_2018 |
| 56 | 2014-07-20 | OK | 1,970 | none | 2014-07-20 | ✔ | ✔ | PRE_2018 |
| 28 | 2014-08-17 | OK | 2,026 | none | 2014-08-17 | ✔ | ✔ | PRE_2018 |
| 14 | 2014-08-31 | OK | 2,054 | none | 2014-08-31 | ✔ | ✔ | PRE_2018 |
| 7 | 2014-09-07 | OK | 2,068 | none | 2014-09-07 | ✔ | ✔ | PRE_2018 |

**All six horizons run.** The exact-horizon transition counts are far above the
frozen minimum of 30, so **no fallback horizon fires** at any horizon; the residual
pool is available with `K_outer = 3`; the PRE_2018 law dispatches correctly; every
sampled draw is a valid 349-seat allocation. No new scientific model and no ad-hoc
imputation were required.

The frozen production input was not modified: the research data root is assembled
under a gitignored `_scratch/`, with the extended timeseries written there and
every other input symlinked to its unchanged production copy.

---

## 6. Acceptance decision

# `HISTORICAL_POP_EXTENSION_REJECTED`

| # | Criterion | Verdict |
|---|---|---|
| 1 | `pofp` has a clear first-party PoP-estimate interpretation | ✔ satisfied (§1, §3) |
| 2 | Overlap with the canonical series is reconciled | ✔ satisfied — 100.000 % exact on all ten categories, 4,362 dates, zero discrepancies |
| 3 | The historical series can be normalized deterministically | ✔ satisfied — 6,444 dates, no interpolation, filling or manual values |
| **4** | **No leakage is detected** | ✖ **FAILS** |
| 5 | All six 2014 cases can run under the frozen model | ✔ satisfied — all six run, no fallback fires |
| 6 | No new scientific model or ad-hoc imputation required | ✔ satisfied |

**Binding reason.** Criterion 4 fails. The PoP state series is a fieldwork-dated
rolling aggregate that is retrospectively completed as later-published polls
arrive, demonstrated by a specific attributed revision (2026-08-22 MP 7.5 → 7.3,
explicable only by a poll published 2026-08-27). Across the six 2014 `as_of` dates,
15 poll-instances have fieldwork covering `as_of` but a publication date after it,
with leads up to 21 days. The task's requirement is unconditional — *"No
retrospective future information may enter a 2014 hindcast"* — and it is not met.

`N_seat` remains **2**. The certified Part-3 evaluation manifest is unchanged, and
2014 remains a Tier-1 case only.

**Everything except leakage is ready.** If a future task resolves the leakage — for
instance by reconstructing a *publication-dated* PoP-equivalent state series from
the individual-poll archive, which would be a new component requiring its own
preregistration — then criteria 1, 2, 3, 5 and 6 are already demonstrated, the
normalized series and its provenance are committed, and 2014 would become
technically eligible as the third Tier-2/Tier-3 election.

---

## 7. Escalation: the same leakage affects the two elections already certified

This is the more consequential finding, and it must not be buried in a rejection.

The leakage is **not a property of the candidate extension**. The candidate series
is byte-identical to the canonical production series over their entire overlap, so
the property belongs to the **frozen production input** that the certified Part-3
harness already consumes for 2018 and 2022.

Applying the identical audit to the elections already in the certified Tier-2/Tier-3
set:

| Election | Leaking poll-instances across its six `as_of` dates | Max publication lead |
|---|---|---|
| **2018** | **19** | **37 d** |
| **2022** | **13** | **18 d** |
| *(2014, rejected here)* | *15* | *21 d* |

2018 and 2022 leak **more** than 2014 does, not less. So the criterion that rejects
2014, applied consistently, indicts the two elections the harness is already
certified on.

What this does and does not mean:

* It **does not** invalidate a CONTROL-versus-challenger comparison. Every model
  receives the identical state input for the same `(case, horizon, seed)`, so the
  leakage is common-mode and the *relative* comparison the adoption gate rests on
  is unaffected.
* It **does** mean the absolute Tier-2/Tier-3 hindcast scores in the Part-3 CONTROL
  baseline are optimistic relative to a genuinely causal forecast, by an amount
  bounded by a few days to a few weeks of extra polling information.
* It **is** a pre-existing property of the frozen model, protected by the §G
  invariant that fixes "historical as-of construction". I have not changed it and
  cannot change it here.

**This requires reviewer decision before Part 4, and I am not making it.** The two
coherent options are:

1. **Accept the inherited leakage** as a documented, common-mode property of the
   frozen model. If 2018 and 2022 are acceptable on these terms, 2014 is acceptable
   on the same terms — which would reverse this rejection and take `N_seat` to 3.
   That is a preregistration decision, not a harness decision.
2. **Treat it as a defect** requiring a publication-dated state reconstruction for
   *all* historical cases, which would change the frozen model's inputs and require
   re-certifying the Part-3 baseline.

Choosing option 1 by myself would let a rejection be reversed by an argument I
generated, which is exactly the dynamic this program's preregistration discipline
exists to prevent. Hence the flag.

---

## 8. Recorded maintenance item (not fixed here)

`scripts/vote_share_calibration/energy_score.py::compute_discrete_energy_score`
normalises its dispersion term by `K(K−1)` rather than `K²`, so for a discrete law
whose support *is* the K points it returns a value that is too small — the
dispersion term is inflated by `K/(K−1)`, i.e. 1.5× at K = 3. It is the
without-replacement U-statistic, correct when the points are distinct *samples* from
a continuous law, and it is **not modified here**.

The certified evaluation harness does not use it: Part 3's D1 uses
`compute_energy_score` on the draws, exactly as the preregistration mandates, and
its closed-form anchor is `metrics.exact_uniform_atom_energy_score`. Recorded as a
separate maintenance item for a future task, which should also check whether any
other repository artifact relies on it.

The SwedishPolls latent-state idea was not touched.

---

## 9. Guardrail compliance

Not modified: the frozen preregistration (body hash verified unchanged), the
certified Part-3 evaluation manifest, RC1, `current.json`, the prospective archive,
`data/processed/pollofpolls/pollofpolls_timeseries.csv` or anything else under
`data/`, geography, the mandate allocator, seeds, sample counts, adoption gates. No
challenger implemented or scored; no 2026 challenger forecast; **no predictive score
against the 2014 outcome computed**. `compute_discrete_energy_score` left as is.

The unrelated untracked files `scripts/forecast_history/` and
`tests/test_forecast_history*.py` were not deleted, modified, opened beyond a
directory listing, or committed.

Targeted execution only: the reconciliation, leakage-audit and smoke-test scripts
in this directory. The full test suite was not run.
