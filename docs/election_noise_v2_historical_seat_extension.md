# ElectionNoise v2 — Part 2B: historical seat-evaluation extension

**Evaluation infrastructure, not model tuning.** No ElectionNoise challenger was
implemented or scored, no adoption metric was computed, and no 2026 forecast was
run. The frozen preregistration was not edited.

| | |
|---|---|
| Preregistration | **FROZEN — AMENDMENT 1**, `80b1c671c4b6d879a888f28a859ee392e8f59bc5`, body SHA-256 `bac3ca06e52cc07fe74ca9e5aa785d94e30934db32193c7f948e95a49a6ae075` — unchanged |
| Part 2 predecessor | `cb39e84074def993e804ba4d2ec478d59c27fa4a` |
| **Result** | **`N_seat` = 3** — 2014 **ACCEPT**, 2018/2022 incumbent **ACCEPT**, 2010 **DEFER** |

---

## 1. Files changed

**Production code (behaviour-preserving):**

| File | Change |
|---|---|
| `scripts/mandates/law.py` | **New.** `MandateLaw` enum, per-version first divisors, and `mandate_law_for_election_year()`. Carries the legal provenance in its docstring. |
| `scripts/mandates/allocator.py` | **Modified, additively.** New keyword-only `law` argument defaulting to `POST_2018`; a `_pre_2018_national_entitlement` helper; a single early-exit branch inside the existing convergence loop; two new audit fields (`law`, `set_aside_parties`) with defaults. Every shared rule is reused, not duplicated. |
| `tests/test_historical_mandate_law.py` | **New.** 14 tests: law mapping, production-default invariance, and the 2010/2014 golden reproductions (auto-skipped when the research data is absent, so CI never depends on it). |

**Research-only (nothing under `data/`):**

`docs/election_noise_v2_historical_seat_extension.md` (this report) and
`diagnostics/election_noise_v2/historical_seat_extension/` containing
`fetch_historical_results.py`, `normalize_historical_results.py`,
`validate_historical_allocator.py`, `validate_historical_geography.py`,
`acceptance.py`, `manifest.json`, `raw/` (145 archived pages + `fetch_manifest.json`),
and `processed/` (normalized CSVs, `fixed_seats_by_year.json`, `reconciliation.json`,
`allocator_validation.json`, `geography_validation.json`, `acceptance.json`,
`part3_seat_cases.json`, `research_geography/`).

---

## 2. Source provenance

All page retrieval is automated and deterministic (`fetch_historical_results.py`);
no value was copied by hand anywhere in this task. **145 pages**, each stored
verbatim with URL, retrieval timestamp, byte count and SHA-256 in
`raw/fetch_manifest.json`. Aggregate digest over all page hashes:
`e4041a79fa84c1cc…`.

| Group | Pages | URL pattern (Valmyndigheten, `historik.val.se`) | Purpose |
|---|---|---|---|
| `votes_mandates_2006` | 29 | `/val/val2006/slutlig/R/riksdagsvalkrets/<cc>/roster.html` | 2006 constituency party votes + valid totals (baseline for the 2010 target) |
| `ovriga_2006` | 29 | `/val/val2006/slutlig/R/riksdagsvalkrets/<cc>/ovriga.html` | Per-constituency breakdown of the aggregate `ÖVR` row — **required**, because the 2006 result pages do not break out Sverigedemokraterna |
| `votes_2010` | 29 | `/val/val2010/slutresultat/R/rvalkrets/<cc>/index.html` | 2010 constituency party votes (2010 target results; baseline for the 2014 target) |
| `mandates_2010` | 29 | `/val/val2010/slutresultat/R/rvalkrets/<cc>/valda.html` | Certified 2010 per-constituency mandates |
| `mandates_2014` | 29 | `/val/val2014/slutresultat/R/rvalkrets/<cc>/valda.html` | Certified 2014 per-constituency mandates |

Encoding differs by vintage and is detected per file: the 2006 pages are UTF-8,
the 2010/2014 pages ISO-8859-1.

Derived from artifacts **already in the repository and not modified**:

* `data/raw/mandates/valkretsmandat_riksdag_1988_2026.xlsx` (Valmyndigheten,
  sha256 `74d02c44…`) → fixed constituency seats for 2010 and 2014.
  **Independently validated:** the same parser reproduces the repository's
  hard-coded `FIXED_SEATS_2018` and `FIXED_SEATS_2022` exactly, and every year
  yields 29 constituencies summing to 310.
* `data/processed/geography/constituency_party_votes_2014_2022.csv` → 2014 target
  votes and the 2014/2018/2022 baselines (copied unchanged; byte-compared after copy).
* `data/processed/elections/riksdag_election_results.csv` → official national
  totals used as the exact reconciliation reference.

Legal provenance: Prop. 2013/14:48, archived in Part 2 at
`diagnostics/election_noise_v2/historical_evidence/raw/prop_2013_14_48_proportionell_fordelning.pdf`
(sha256 `5aa84ff2…`), enacted as SFS 2014:1384.

### Normalization and reconciliation

Party mapping is deterministic: `M, C, KD, S, V, MP, SD` map to themselves and
**`FP` (Folkpartiet liberalerna) → `L`** — a pure rename with no organisational
change. Everything else (including `ÖVR` net of SD in 2006) becomes `REST`; rows
`BLANK`, `OG`, `VDT` are never votes and are excluded. In 2006, SD is folded out
of `ÖVR` using the per-constituency `ovriga` page, and the parser **fails hard**
unless the `ovriga` listing sums exactly to the `ÖVR` figure on the result page.

| Check | 2006 | 2010 | 2014 (control) |
|---|---|---|---|
| Constituencies | 29 | 29 | 29 |
| Valid-vote total vs official | **0** | **0** | **0** |
| Per-party difference (all 9 categories) | **0** | **0** | **0** |
| Constituency valid totals sum to national | ✔ | ✔ | ✔ |

Certified mandates parsed, each with the page's own `Totalt` control row checked:

* **2010** — M 107, L 24, C 23, KD 19, S 112, V 19, MP 25, SD 20 = **349**
* **2014** — M 84, L 19, C 22, KD 16, S 113, V 21, MP 25, SD 49 = **349**

Boundary compatibility agrees with the Part-2 finding: the workbook's own
footnotes place constituency changes at 1994, 1998, 2006 (Heby → Uppsala) and
2018 only, so **2006→2010 and 2010→2014 are both boundary-clean** across a stable
set of 29 constituencies.

---

## 3. PRE_2018 vs POST_2018 — the legal differences implemented

The archived proposition was re-read before coding rather than assumed. Its
Riksdag section enumerates the changes, and they are **exactly two**:

1. **First divisor.** §4.1.4: *"Vid fördelningen av mandaten mellan partierna i
   riksdagsval ska den jämkade uddatalsmetoden tillämpas med 1,2 som första
   delningstal … i stället för 1,4."* → `Fraction(6,5)` post-2018,
   `Fraction(7,5)` pre-2018.
2. **Mandate return (`återföring`).** Introduced by the same reform as
   Vallagen 14 kap. 4a–4c §§. It **did not exist** before.

The pre-reform overhang rule is stated in §4.1.1: *"det parti som blev
överrepresenterat fick behålla de mandat som det fått i första omgången och att
utjämningsmandaten fördelades mellan övriga deltagande partier så att dessa blev
riksproportionellt representerade sinsemellan. Den reglering som beredningen
föreslog … finns i nuvarande lagstiftning intagen i 3 kap. 8 § RF och 14 kap. 5 §
vallagen."*

§4.5 independently confirms the reform did **not** reach the 2014 election.

| Rule | POST_2018 (production default) | PRE_2018 (2010, 2014) |
|---|---|---|
| Modified Sainte-Laguë first divisor | **1.2** = `Fraction(6,5)` | **1.4** = `Fraction(7,5)` |
| Subsequent divisors `2k+1` | same | same |
| Total seats / fixed / adjustment | 349 / 310 / 39 | **same** |
| National 4 % threshold, constituency 12 % exception | same | **same** |
| Excess fixed seats | **Returned** and reallocated (14 kap. 4a–4c §§); never returned from a constituency with < 3 fixed seats | **No return.** The over-represented party keeps its fixed seats and is **set aside together with them**; the remaining seats are redistributed among the other participating parties. Iterated, because setting one party aside can push another above its recomputed entitlement. |
| Adjustment-seat placement divisors (1 if the party holds no seat in the constituency, else `2k+1`) | same | **same** |
| Tie resolution by deterministic keyed lottery | same | **same** |

**Everything in the "same" rows is shared code, not a second implementation.** The
historical path is one early-exit branch inside the existing convergence loop plus
one helper; the fixed-seat allocation, threshold logic, adjustment placement,
invariants and tie-breaking are the production functions unchanged.

The version is **never inferred from the wall clock**:
`mandate_law_for_election_year(year)` maps the *target election year* to a
self-consistent `(law, first_divisor)` pair, and `allocate_riksdag_seats` takes
`law` as an explicit keyword defaulting to `POST_2018`.

Σ`U_p` = 39 remains an identity under PRE_2018: with `A` the set-aside parties,
Σ_remaining `E_p` = 349 − L − Σ_A `F_p` and Σ_remaining `F_p` = 310 − L − Σ_A `F_p`,
so the difference is 39 exactly. The production invariant checks therefore hold
unchanged in both versions.

---

## 4. 2010 allocator golden result

Official 2010 constituency votes → fixed seats 2010 → PRE_2018, divisor 7/5:

| | M | L | C | KD | S | V | MP | SD | Σ |
|---|---|---|---|---|---|---|---|---|---|
| Produced | 107 | 24 | 23 | 19 | 112 | 19 | 25 | 20 | **349** |
| Certified | 107 | 24 | 23 | 19 | 112 | 19 | 25 | 20 | 349 |

**Exact nationally and in every one of the 29 constituencies × 8 parties.** No
tuning was applied; this was the first and only run.

Parties set aside as over-represented: **M and S**. This is independent
corroboration of the *mechanism*, not merely the outcome: prop. 2013/14:48 §4.1.1
records that at the 2010 election *"Socialdemokraterna blev överrepresenterade med
tre mandat och Moderaterna med ett"* — the same two parties the implementation
identifies from the votes alone.

Contrast under the wrong (current) law: **8 seats of absolute error**. The law
version is load-bearing.

## 5. 2014 allocator golden result

Official 2014 constituency votes → fixed seats 2014 → PRE_2018, divisor 7/5:

| | M | L | C | KD | S | V | MP | SD | Σ |
|---|---|---|---|---|---|---|---|---|---|
| Produced | 84 | 19 | 22 | 16 | 113 | 21 | 25 | 49 | **349** |
| Certified | 84 | 19 | 22 | 16 | 113 | 21 | 25 | 49 | 349 |

**Exact nationally and per constituency.** Parties set aside: **S and SD**.
Contrast under the wrong law: **6 seats of absolute error**.

Historical `valda` pages publish only final per-constituency mandates, with **no
fixed-vs-adjustment split**. Accordingly **no phase-level expectation was
invented**: the golden target is the certified final allocation. The internal
phases that therefore cannot be independently certified for 2010/2014 are the
initial fixed-seat vector, the national entitlement vector, and the split of each
party's seats into fixed versus adjustment. They are internally consistent and
satisfy every production invariant, but only their sum is externally certified.
(For 2018/2022 the phase split *is* certified and is already tested.)

## 6. Current law is unchanged

* **Byte-identity against `main`.** `main`'s allocator was loaded side by side
  with the current one and both were run on four configurations. A SHA-256 over
  the complete result — national votes, eligibility maps, initial and final fixed
  allocations, entitlement, returns, adjustment allocation, final seats, and the
  **entire event log** — is identical in every case:

  | Configuration | events | identical |
  |---|---|---|
  | 2018 votes / `FIXED_SEATS_2018` | 698 | ✔ |
  | 2022 votes / `FIXED_SEATS_2022` | 698 | ✔ |
  | 2022 votes / `FIXED_SEATS_2026` | 698 | ✔ |
  | 2018 votes / `FIXED_SEATS_2026` | 1049 | ✔ |

* **Targeted tests pass:** `tests/test_mandate_allocation.py` (21), the new
  `tests/test_historical_mandate_law.py` (14), plus `test_geographic_projection`,
  `test_adversarial_mandates` and `test_threshold_attribution`. The full suite was
  not run.
* The 2018 and 2022 golden results are re-asserted **through the production call
  signature** (`allocate_riksdag_seats(votes, fixed)` with no `law` argument), and
  the result records `law == "POST_2018"` with `set_aside_parties == ()`.
* An explicit `law=POST_2018` call is asserted equal to the default call
  event-for-event.
* The first fixed-seat event under the default path still carries divisor
  `Fraction(6, 5)`.
* Deterministic tie-breaking is asserted stable across repeated runs.
* No 2026 production configuration was touched; `FIXED_SEATS_2026` and every
  simulator default are unchanged.

---

## 7. Geography validation on the new chains

Same frozen geography model (`project_constituency_votes`, unmodified), same
`chronological` mode the production simulator uses, actual national vote of the
target election, then the historically correct allocator. The two incumbent
chains are re-run identically as controls.

| Chain | Law | Constituency share MAE | Seat error | Certified reproduced |
|---|---|---|---|---|
| 2006 → 2010 | PRE_2018 | 0.00677 | **8** | No |
| **2010 → 2014** | PRE_2018 | **0.00628** | **4** | No |
| 2014 → 2018 (control) | POST_2018 | 0.00649 | 0 | Yes |
| 2018 → 2022 (control) | POST_2018 | 0.00665 | 0 | Yes |

**The geography model transfers cleanly.** The two new chains bracket the two
incumbent ones on the geography metric — 2010→2014 is in fact the *best* of the
four — so no new scientific model is required and none was built. National totals
reconcile exactly on every chain by construction of the IPF column targets.

The seat errors are therefore projection error, not a geography-model failure:

* **2010:** M +1, L +1, S **−4**, V +1, MP +1.
* **2014:** M −1, S +1, V −1, SD +1 — internal transfers that leave each bloc
  total unchanged (S+V+MP = 159 both ways; M+L+C+KD+SD = 190 both ways).

A hypothesis that the un-damped pre-2018 law amplifies projection error was
tested and **rejected**: running the projected 2010 chain under POST_2018 gives
the same 8-seat error, and the projected 2014 chain gives a *worse* 6. The
difference between the new and incumbent chains is the projection, not the law.

### The decisive test: does the pipeline reproduce the certified coalition indicator?

Raw seat error is not the quantity these cases exist to serve. The frozen
coalition-Brier metric scores `1{Σ seats ≥ 175}` over the preregistered mask set,
against the **certified** outcome. If the deterministic transform, fed the actual
national vote, disagrees with the certified indicator on any mask, then that mask
is scored against a target the pipeline cannot reach even with a perfect vote
forecast.

| Chain | Masks disagreeing / 254 | Distinct events / 127 |
|---|---|---|
| 2006 → 2010 | **10** | **5** |
| 2010 → 2014 | **0** | **0** |
| 2014 → 2018 | 0 | 0 |
| 2018 → 2022 | 0 | 0 |

The 2010 failures land exactly where they matter most — on the majority line:

| Mask | Coalition | Pipeline | Certified |
|---|---|---|---|
| 15 | M+L+C+KD (Alliansen) | 175 | 173 |
| 39 | M+L+C+V | 176 | 173 |
| 77 | M+C+KD+MP | 176 | 174 |
| 120 | KD+S+V+MP | 173 | 175 |
| 135 | M+L+C+SD | 176 | 174 |

The 2010 chain would have scored the actual 2010 Alliansen coalition as holding a
majority when certified seats say it did not.

---

## 8. Verdict — 2010: **DEFER**

| Criterion | Met |
|---|---|
| Authoritative inputs acquired and preserved | ✔ |
| Mappings reconcile exactly (0 votes) | ✔ |
| Historical law unambiguous | ✔ |
| Allocator reproduces certified seats from official votes | ✔ (exact, nationally and per constituency) |
| Geography runs chronologically without a new scientific model | ✔ (MAE 0.00677, in range) |
| **Coalition indicator reproduced for every mask** | ✖ — **10 of 254 masks disagree** |

Deferred, not rejected: nothing about the data, the law or the allocator failed.
The 2006→2010 projection carries a −4-seat error on S, and because S is large,
every S-containing coalition moves by 4 seats, flipping five distinct
majority events. The most likely cause is the party-system change across that
chain — SD polled 2.93 % in 2006 with no seats and 5.70 % in 2010 with 20, so the
2006 baseline carries almost no constituency structure for SD, and M surged over
the same period. Admitting 2010 would inject a case whose realized coalition
outcomes the pipeline provably cannot reproduce. That is exactly the standard the
task forbids lowering to raise `N_seat`. All artifacts are preserved so the case
can be reconsidered if a future task establishes a defensible treatment.

## 9. Verdict — 2014: **ACCEPT**

| Criterion | Met |
|---|---|
| Authoritative inputs acquired and preserved | ✔ |
| Mappings reconcile exactly (0 votes, all 9 categories) | ✔ |
| Historical law unambiguous | ✔ (divisor 1.4, no return; §4.5 confirms 2014 predates the reform) |
| Allocator reproduces certified seats from official votes | ✔ exact, nationally and in all 29 × 8 cells |
| Geography runs chronologically without a new model | ✔ **0.00628 MAE — the best of the four chains** |
| Coalition indicator reproduced for every mask | ✔ **0 of 254 disagree** |
| No unexplained discrepancies | ✔ |

The residual 4-seat marginal error is explained, bounded, and inert for the metric
this case serves: the four displacements are internal to their blocs and leave
every one of the 254 coalition-majority indicators correct.

**Stated limitation, carried forward honestly.** The mask check is evaluated *at*
the realized vote, not over a neighbourhood of it. A systematic 4-seat
displacement could in principle affect masks near the threshold elsewhere in the
predictive distribution. That risk is not measurable without running the
challenger evaluation, which this task must not do. It is recorded here so Part 3
reads it as a known property rather than discovering it.

## 10. Final `N_seat`

### **`N_seat` = 3** — accepted: **2014, 2018, 2022**; deferred: **2010**

Under the frozen preregistration (§F.3 G5, unchanged), `N_seat = 3` mechanically
selects the **`N_elections ≥ 3`** branch of the coalition-Brier rule:

* aggregate coalition Brier must improve **≥ 2 %**; **and**
* the challenger must improve the election-level aggregate in at least
  `ceil(3 / 2) = 2` of the 3 elections; **and**
* removing any one evaluation election must not turn the aggregate
  challenger-vs-CONTROL Brier delta into a degradation of more than 1.0 %.

This is a stricter and better-powered test than the `N_elections == 2` branch that
would have applied before this task. No preregistration edit was needed or made.

## 11. What blocks Part 3

**Nothing blocks it.** Four items should be carried in as known constraints:

1. **Tier-3 is now 18 cases, not 12** — 3 elections × 6 horizons. Whoever runs
   Part 3 must use `N_seat = 3` and the `≥ 3` Brier branch, taking the case set
   from `processed/part3_seat_cases.json` rather than re-deriving it.
2. **2014 must be allocated under PRE_2018.** Any Part-3 harness that calls the
   allocator for 2014 without `law=PRE_2018` and `first_divisor=Fraction(7,5)`
   will silently produce a legally wrong result — 6 seats off. Use
   `mandate_law_for_election_year()`; a regression test already guards the wrong-law
   case.
3. **The 2014 chain carries a 4-seat marginal displacement** (§9), inert at the
   realized point but unmeasured across the predictive distribution.
4. **Phase-level validation for 2014 is not available** (§5); only the final
   certified allocation is externally certified.

Tier 1 (`N_T1 = 3`) and the residual pool (`K = 6`) are untouched by this task.

---

## Guardrail compliance

Not modified: the preregistration (verified byte-identical to its freeze), RC1
configuration, `current.json`, the prospective archive, production polling data,
the geography model, `FIXED_SEATS_2026`, seeds, sample counts, adoption gates. The
production allocator's default behaviour is proven byte-identical to `main`
including full event logs. Nothing under `data/` was written. No ElectionNoise
challenger was implemented or scored; no 2026 challenger forecast was run.
Targeted tests only; the full suite was not run.
