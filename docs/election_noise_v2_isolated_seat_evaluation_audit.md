# ElectionNoise v2 — Part 3C: audit of the isolated seat/coalition evaluation

**Audit only.** No challenger was implemented or scored, no 2026 challenger forecast
was run, no predictive score against any certified outcome was computed, and no
CONTROL baseline was recertified.

| | |
|---|---|
| Base commit | `89d340880a4bdb389f94ce61fa3333799b58d81a` |
| Branch | `research/isolated-seat-evaluation` |
| **Verdict** | **`ISOLATED_SEAT_EVALUATION_READY`** |
| **Final `N_seat`** | **3** (2014, 2018, 2022) |
| Preregistration | **FROZEN — AMENDMENT 2** (§J.5), body SHA-256 `5a9a6dc8ef6f26ce3ce152155af0ed288fb8d2d97c81a2606e513cf20e1b058b` |

Machine-readable evidence:
`diagnostics/election_noise_v2/isolated_seat_evaluation/processed/isolated_path_audit.json`,
reproduced by `audit_isolated_path.py`.

---

## 1. The path audited

```
historical final 14-day polling consensus      publication-date safe
  -> ElectionNoise                             the component under test
  -> unchanged bounded simplex transfer         λ rule, ε = 0.01 pp
  -> frozen deterministic geography             chronological mode only
  -> historically correct mandate law
  -> joint per-draw seat vectors
  -> seat-vector ES and coalition-majority Brier
```

The consensus is **exactly**
`scripts/election_residuals/consensus.py::build_election_polling_consensus`, the
same estimator used to construct the historical ElectionNoise residuals. **No new
polling estimator was invented**, and none may be.

---

## 2. Geography and allocator inputs, classified

Every input the path consumes, with the classification the task asked for.

| Input | Source | Classification | Target-realized? |
|---|---|---|---|
| Baseline matrix `B` (29 × 9 constituency party votes) | rows for the **baseline** election year | prior-election information | **No** |
| Target constituency row totals `R` | `R = sum(B, axis=1)` in chronological mode for target ≤ 2022 | prior-election information | **No** |
| Total national valid votes (IPF scale) | `sum(R)`, the baseline total, with `total_national_votes=None` | prior-election information | **No** |
| National vote shares `C` (IPF column margins) | the forecast draw itself | forecast output, not an observation | **No** |
| Constituency electorates file | `constituency_electorates_*.csv` | read but **unused** in this mode — proven below | **No** |
| Fixed constituency seats for the target year | Valmyndigheten workbook | fixed legal metadata, **decided and published before the election** (Vallagen 4 kap. 3 §: apportioned from eligible voters as of 1 March of the election year) | **No** |
| Mandate law version and first divisor | `mandate_law_for_election_year` | fixed legal metadata (SFS 2014:1384) | **No** |
| 4 % / 12 % thresholds, 349 = 310 + 39 | Regeringsformen 3 kap. 7 §, Vallagen 14 kap. | fixed legal metadata | **No** |
| 29 constituency codes | `OFFICIAL_CONSTITUENCY_CODES` | fixed legal metadata, unchanged since 1998 | **No** |
| Deterministic tie-break seed | `DeterministicLotteryTieBreaker(seed=12345)` | fixed model convention | **No** |
| *Oracle-mode target valid votes* | electorates rows for the **target** year | **TARGET-ELECTION REALIZED** | **Yes — therefore `oracle` mode is prohibited on this path** |

**No target-election realized information enters the path in `chronological` mode.**
The single target-realized input in the geography module is oracle mode's row
margins, and Amendment 2 prohibits oracle mode here.

The code itself is explicit about the chronological branch — *"Strictly
chronological: row totals are derived entirely from baseline valid votes. Zero
information from target election electorate or valid votes is accessed!"* — but a
comment is not evidence, so the claim was tested.

### Electorates perturbation test

The electorates file is read unconditionally by
`_get_cached_geography_structures`, so "unused" had to be demonstrated rather than
asserted. Every numeric column was multiplied by 7.77 and offset by 12 345, and the
projection re-run for all three targets:

| Target | Projection identical under grossly perturbed electorates |
|---|---|
| 2014 | ✔ |
| 2018 | ✔ |
| 2022 | ✔ |

Constituency votes and constituency valid votes are bit-identical. The file does
not enter chronological-mode output.

---

## 3. Consensus publication-safety

`build_election_polling_consensus` admits a poll only if
`publication_date <= election_date` **and** `interview_end <= election_date`. This
is the structural difference from the Poll-of-Polls state series, which carries no
publication filter at all.

| Target | Retained pollsters | Eligible polls in window | All retained published ≤ E | All fieldwork ended ≤ E | Fieldwork-eligible but excluded for publishing after E |
|---|---|---|---|---|---|
| 2014 | 9 | 23 | ✔ | ✔ | **0** |
| 2018 | 10 | 35 | ✔ | ✔ | **0** |
| 2022 | 7 | 47 | ✔ | ✔ | **0** |

The last column is the information a fieldwork-dated series would have absorbed and
this path refuses. At these three targets it is **empty**: every poll whose
fieldwork fell in the final 14 days was also published by election day. So the
isolated path loses no available polling information relative to the leaky
construction — it simply cannot gain any unavailable information.

### Poll archive revision test

The same snapshot comparison that exposed the PoP series in Part 3B, applied to
`swedishpolls_individual_polls.csv` restricted to `publication_date <= 2022-12-31`
across two snapshot pairs (`f55bf36 → f6ae4d1`, `f6ae4d1 → 34c52d6`):

**Zero historical support values revised.** And even if the archive were revised,
the `publication_date <= election_date` filter would still prevent injecting a poll
that was unpublished at the forecast origin — the safety here is structural, not
merely empirical.

---

## 4. Three-target smoke test

No predictive score computed. Structural facts only.

| Target | Law | Divisor | Residual pool | `K` | Retained pollsters | Distinct vote support points | Distinct seat vectors | All totals 349 | Mean λ |
|---|---|---|---|---|---|---|---|---|---|
| 2014 | **PRE_2018** | 7/5 | 2002, 2006, 2010 | 3 | 9 | **3** | **3** | ✔ | 1.0000 |
| 2018 | POST_2018 | 6/5 | 2002, 2006, 2010, 2014 | 4 | 10 | **4** | **4** | ✔ | 1.0000 |
| 2022 | POST_2018 | 6/5 | 2002, 2006, 2010, 2014, 2018 | 5 | 7 | **5** | **5** | ✔ | 1.0000 |

All three run. No future year enters any training pool. The certified truth vectors
sum to 349 for all three.

**The distinct-support-point counts are the important line.** CONTROL's predictive
law on this path is exactly `K` atoms in vote space and exactly `K` atoms in seat
space. λ ≡ 1 at all three targets, so the simplex floor never binds and the atom
count is preserved end to end.

---

## 5. Interpretation — the vote and seat evaluations are not independent

Recorded in the preregistration as §E.7 and repeated here because it governs how
the eventual result must be read.

Tier 1 and Tier 3-ISO score **the same predictive distribution**. Tier 1 scores it
directly in vote space; Tier 3-ISO scores deterministic nonlinear pushforwards of
the identical draws through geography, integerisation and the statutory allocator.
**They are related evaluations of one forecast, not independent evidence**, and no
statement of the form "the challenger won on two independent tiers" is admissible.

Their purpose is to check that improved marginal and joint vote prediction also
behaves sensibly for the downstream quantities that exposed the six-atom problem —
a challenger could improve vote-space scores while degrading a coalition-threshold
functional, and the seat tier exists to catch that.

Two consequences must be stated openly:

* CONTROL's coalition probability `p_m` can only take values in `{0, 1/K, …, 1}` —
  at 2014, only `{0, ⅓, ⅔, 1}`. That is the six-atom pathology itself, now visible
  directly in the metric.
* A continuous challenger can express intermediate probabilities CONTROL
  structurally cannot. The Brier score is strictly proper and the truth is external,
  so the comparison is legitimate and not circular — but the ≥ 2 % coalition-Brier
  improvement threshold is likely **easy** for any well-behaved continuous law to
  clear. **The informative content of the gate therefore sits in the
  non-inferiority guards (G3, G4) and the per-election robustness conditions (G5),
  not in the headline Brier improvement.**

---

## 6. Amendment 2 — what changed, and what did not

**Changed (scope only):**

* New **§E.2a Tier 3-ISO**, the authoritative seat/coalition evaluation: targets
  `{2014, 2018, 2022}`, **`N_seat` = 3**, one case per election (no horizon
  dimension), chronological geography only, oracle prohibited, statutory law per
  target.
* Full-pipeline **Tier 2 and Tier 3 leave the gate** and are **preserved** as
  retrospective diagnostics. Their Part-3 results are not discarded or rewritten.
* **G2** and **G4** re-point to Tier 3-ISO. **G3** re-points from Tier 2 to Tier 1,
  with identical tolerances. **G4b is retired from the gate** and kept as a Tier-2
  diagnostic.
* New **§E.5 item 7** recording the leakage finding, and new **§E.7** recording
  non-independence.
* **D3** truth pointer extended to the Part-2B certified 2014 vector.
* §E.6, §E.3, §I item 1 and the D4 aggregation section annotated so no stale
  statement contradicts the amendment.

**Verified byte-identical to Amendment 1** by a programmatic diff of each block:
D0 (seeds and draw count), D1, D2, the D4 mask set and complement rule, D5, F.1
tolerances, the CONTROL specification, **Challenger A**, **Challenger B**, G5's
Tier-1 clause, G6, G7, F.4 and §B. No threshold, seed, sample count, `h` grid, λ
rule, simplex-transfer rule, challenger definition or Tier-1 vote gate changed.

**Recorded reduction in gate coverage.** Retiring G4b removes one hard gate. The
partial mitigation is that both gate tiers are built on the final 14-day consensus —
the shortest and operationally most relevant origin — so the regime G4b protected is
the regime the gate now evaluates throughout. That is genuinely weaker than G4b,
which tested the *full pipeline* at a short horizon whereas the isolated path omits
OpinionState and Dynamics. Accepted in exchange for removing a leaky input from the
adoption decision, and flagged rather than buried.

**Why this is not post-hoc.** Amendment 2 was made after a *data-property*
discovery and before any challenger existed. No challenger has been implemented, no
challenger score exists, and the 2026 forecast has not been run under any
challenger. The change cannot have been selected to favour an outcome, because no
outcome exists.

---

## 7. Blocking recertification of the CONTROL baseline

Nothing blocks it. Four items to carry in:

1. **Tier 3-ISO must be added** to the harness: 3 cases, one per election, at the
   frozen 5 seeds × 20 000 draws. Since CONTROL's law is exactly `K` atoms, its
   Monte Carlo error will be tiny (Part 3 measured 0.008–0.072 % on the analogous
   Tier-1 metrics), but N is frozen and must not be reduced.
2. **2014 must dispatch PRE_2018** with divisor 7/5. The Part-3 harness guard
   `assert_law_dispatch` raises if a PRE_2018 target is routed through the
   production engine — which is correct, and means the Tier 3-ISO runner must call
   the allocator directly with the law from `mandate_law_for_election_year`, as the
   audit script does.
3. **Geography mode must be `chronological` with `total_national_votes=None`.**
   Oracle mode is prohibited; a recertification runner should assert this.
4. **The existing Part-3 Tier-2/Tier-3 results must be retained** and relabelled as
   diagnostics, not deleted or recomputed.

The 2014 geography floor from Part 2B (4-seat displacement, 0 of 254 coalition
masks disagreeing) still applies to the 2014 Tier-3-ISO case and remains a
documented property of the frozen deterministic transform, not a leakage.

---

## 8. Guardrail compliance

Not done: no new historical PoP estimator; the retrospective `pofp` series was not
treated as leakage-safe and is not used; production PoP inputs untouched; the
SwedishPolls latent-state project untouched; `compute_discrete_energy_score`
untouched; the unrelated `scripts/forecast_history/` and
`tests/test_forecast_history*.py` files not deleted, modified or committed; no
Challenger A/B implementation; no challenger or adoption score; no 2026 challenger
forecast; the CONTROL baseline not rescored.
