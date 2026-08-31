# ElectionNoise v2 evaluation harness — method

**Research infrastructure. CONTROL only.** No challenger is implemented or scored,
no adoption gate is evaluated, and no 2026 forecast is produced. The frozen
preregistration is authoritative and was not edited.

| | |
|---|---|
| Preregistration | **FROZEN — AMENDMENT 1**, `80b1c671c4b6d879a888f28a859ee392e8f59bc5`, body SHA-256 `bac3ca06e52cc07fe74ca9e5aa785d94e30934db32193c7f948e95a49a6ae075` |
| Predecessors | Part 2 `cb39e84`, Part 2B `61d6d3b` |
| Model evaluated | `CONTROL_pp_centered_noise` — the unmodified production ElectionNoise |

---

## 1. Case set

Cases are derived from the frozen rules, never from a score. The manifest builder
re-derives every eligibility decision and **validates** Part 2B's
`part3_seat_cases.json` rather than trusting it.

| Tier | Definition (§E.2) | Elections | Cases | Metrics |
|---|---|---|---|---|
| **Tier 1** | standalone forward evaluation from the 14-day polling consensus; ElectionNoise isolated | 2014, 2018, 2022 | 3 | D1, D2 |
| **Tier 2** | full-pipeline hindcast, six horizons | 2018, 2022 | 12 | D1, D2 |
| **Tier 3** | the same cases as Tier 2, at seat/coalition level | 2018, 2022 | 12 | D3, D4, D5 |

**`N_T1` = 3.** `N_seat` = **2**.

### Why `N_seat` is 2 and not the 3 Part 2B reported

This is a correction to Part 2B, found while building the harness.

Part 2B applied the seven §E.3 admission criteria to the 2014 target and it
passed all of them: authoritative constituency votes reconciling exactly, an
unambiguous PRE_2018 law, a boundary-clean 2010 baseline, certified per-party
seats, and — the decisive test — zero disagreement with the certified
coalition-majority indicator over all 254 masks. All of that still holds.

What §E.3's criteria do not mention, and what Part 2B therefore did not check, is
whether the **upstream** half of the pipeline can run. Tier 2 is defined as a
*full-pipeline hindcast*, and Tier 3 as *the same cases as Tier 2*. The full
pipeline begins with OpinionState v1.1, which requires a Poll of Polls daily
observation on or before `as_of`. That series
(`data/processed/pollofpolls/pollofpolls_timeseries.csv`) **begins 2014-09-15 —
one day after the 2014 election.** Every 2014 horizon needs an `as_of` between
2014-05-25 and 2014-09-07, so `estimate_opinion` raises for all six, and Dynamics
v2 likewise has no eligible transition ending on or before any 2014 `as_of`.

2014 therefore cannot be a Tier-2 case, and because Tier 3 is defined on Tier-2
cases, it cannot be a Tier-3 case either. It remains a full Tier-1 case, where
the base is the deterministic polling consensus and no OpinionState is involved.

The manifest records this as an explicit, machine-readable eligibility check
(`tier23_eligibility`) with the dates that fail, rather than silently dropping the
case. `N_seat = 2` selects the already-frozen `N_elections == 2` branch of the
coalition-Brier rule; no preregistration amendment is needed.

**Recoverable in principle, deliberately not done here.** The PoP daily estimate
*is* present back to 2009-01-02 in already-archived raw data — the `pofp` column of
`data/raw/pollofpolls/party_*.csv` (from `data_big_N.csv`), with 2,082 complete
pre-2014-09-15 rows for each of the eight modelled parties. The processed
timeseries is built from a *different* upstream file (`data_table_tot.csv`) whose
own coverage starts 2014-09-15. Reconstructing 2009–2014 would change how a frozen
production input is constructed for a historical case and would need its own
normalisation, an overlap reconciliation against `data_table_tot` where both exist,
and a leakage audit. That is a separate task, not a harness decision.

**2010** remains excluded everywhere, by the frozen `K_outer >= 3` rule
(its pool is `{2002, 2006}`) and by Part 2B's coalition-indicator failure.

---

## 2. Monte Carlo design (§D0, frozen)

Seeds **12345, 24680, 98765, 54321, 13579**; **N = 20 000 draws per seed**, per
case, per model. Every case is scored separately for every seed. Reported: the
five-seed mean, the five-seed standard deviation, and all five individual values
(`control_scores_summary.json` carries the per-seed values; `monte_carlo_stability.csv`
carries mean/SD/relative SD per case and metric). Seeds are never pooled in a way
that hides variability, and no seed is selected or dropped.

---

## 3. Paired randomness (frozen scheme)

Stated once in `harness/rng.py` so a later challenger cannot alter it.

All production sub-seeds come from one SHA-256 token convention:

```
token   = f"{base_seed}:{origin_date.isoformat()}:{horizon_days}:{label}"
subseed = int(sha256(token).hexdigest()[:8], 16) % 2_147_483_647
```

| Stream | Derivation | Model-dependent? |
|---|---|---|
| OpinionState | `derive_opinion_state_seed(seed, as_of)` | **No** |
| Dynamics | `derive_shared_dynamics_seed(seed, as_of, horizon)` | **No** |
| ElectionNoise index / sign | `derive_vote_share_layer_seeds(seed, as_of, horizon)` | **Yes — the only stream under test** |
| Geography, integerisation | deterministic | No |
| Allocator lottery | keyed on canonical legal state | No |

Because the OpinionState and Dynamics tokens contain no model identifier, two
models run at the same `(as_of, horizon, base_seed)` observe a **bit-identical**
`base_comp_matrix`. Pairing is a property of the existing derivation, not
something the harness arranges. `paired_base_composition()` exposes that matrix
and `assert_paired_base()` turns the property into an executable check.

Challengers must draw only from the tokens the preregistration reserves for them
(`election_noise_v2_a_index`, `..._a_kernel`, `..._a_loeo`, `..._b_normal`) and
must not touch the upstream tokens or CONTROL's `residual_index` / `sign_draw`
streams.

**Tier-1 origin convention (frozen here):** Tier 1 has no `as_of`; its base is the
deterministic consensus, so pairing is automatic. For determinism the ElectionNoise
sub-seed uses `origin_date = election_date`, `horizon_days = 14`.

No upstream draw cache is committed. `_cache/` is gitignored.

---

## 4. Metrics

Every estimator is an existing repository function, used unchanged, with one
documented substitution and one documented anchor choice.

* **D1 primary — joint vote energy score.**
  `compute_energy_score` on the (N, 9) composition in pp, order
  `(M, L, C, KD, S, V, MP, SD, REST)`, Euclidean. The 8-party score is reported
  alongside as the secondary read.
* **D2 — marginal vote metrics.** Per-category CRPS; central 50/80/90 coverage from
  p25/p75, p10/p90, p05/p95 and their widths, exactly as the existing
  `vote_share_calibration` artifacts define them.
* **D3 — joint seat metric.** `calculate_multivariate_energy_score` on the (N, 8)
  integer seat vector, plus `calculate_discrete_seat_crps` and
  `calculate_interval_coverage_and_width` per party. Every draw passes through the
  identical production geography → integerisation → exact allocator path.
* **D4 — coalition-majority Brier.** See §5.
* **D5 — λ diagnostics.** `summarize_lambda_diagnostics` plus min, p01, p10 and the
  fraction below 1. Recorded for every case and seed; **never used for selection**.

### Substitution: the CRPS estimator

`compute_discrete_crps` builds the full `N × N` pairwise matrix — 3.2 GB per party
at N = 20 000, and not computable. The harness evaluates the **same estimator**
through the repository's own O(N log N) implementation,
`scripts/pollofpolls/backtest_metrics.py::calculate_crps`, whose docstring states
the algebraic identity. Both are production code; neither is modified. The two
agree to **6.7e-16** at feasible N (tested at N = 5, 50, 500, 3000). The difference
is summation order only.

### Anchor choice: the two energy-score normalisations are not the same

Found while validating, and worth recording because it is a live trap for anyone
building on this harness.

D1 defines `ES(F, y) = E‖X − y‖ − ½·E‖X − X′‖` with `X, X′` **iid** from `F`. For
`F` uniform on K atoms that is

```
ES = (1/K) Σ_m ‖s_m − y‖ − ½ · (1/K²) Σ_{m,l} ‖s_m − s_l‖
```

— a `1/K²` normalisation that includes the `m = l` pairs, each contributing 0.

The repository also has `compute_discrete_energy_score`, which normalises the
dispersion term by `K(K−1)`. That is the without-replacement U-statistic, unbiased
for `E‖X − X′‖` when the points are *distinct samples* from a continuous law — but
when the points **are** the support of a discrete law it is larger by `K/(K−1)`,
i.e. **1.5× at K = 3**. Using it as the Tier-1 anchor produced an apparent 14–27 %
"Monte Carlo error" with a five-seed SD of only 0.002 — bias masquerading as noise.
The correct closed form (`metrics.exact_uniform_atom_energy_score`) agrees with the
reported Monte Carlo values to 0.01–0.07 %.

`compute_discrete_energy_score` is **not a bug** and is not modified; it is simply
not the right anchor for a K-atom predictive law, and is deliberately unused here.
The reported D1 scores use `compute_energy_score` on the draws, exactly as D1
mandates.

---

## 5. Coalition-majority Brier (D4)

* Masks **1 … 254** over the fixed order `(M, L, C, KD, S, V, MP, SD)`; masks 0 and
  255 excluded as degenerate.
* Coalition seats are summed from the **joint per-draw seat vector**
  (`seats[:, cols].sum(axis=1)`), never from marginal summaries. A test constructs
  a distribution whose marginals mislead and whose joint draws do not, and asserts
  the joint answer.
* `p_m` = fraction of draws with coalition seats **≥ 175** (inclusive);
  `y_m` = indicator on the **certified** seat vector; `B_m = (p_m − y_m)²`.
* Truth is always the certified historical result — never the pipeline's own output.

**Aggregation, in this order:**

1. per `(election, horizon, seed)`: mean of `B_m` over all 254 masks;
2. per `(election, horizon)`: mean over the five seeds, with SD retained;
3. per **election**: mean over the six horizons — the election-level aggregate;
4. headline: unweighted mean of the election-level aggregates.

Masks are never treated as independent observations, and elections are weighted
equally regardless of how many cases each contributes. A test verifies step 4 is
insensitive to unequal case counts.

**Complement symmetry.** `B_{255−m} = B_m` algebraically. In floating point `p` and
`1 − p` are computed independently before squaring, so agreement is to machine
epsilon (**observed max 1.1e-16**), not bitwise. The documented 1e-12 tolerance
covers it, and the mean over 254 masks equals the mean over 127 complement
representatives to 1e-12. The effective number of distinct binary events per
election is **127**, never 254.

`mask_level/coalition_brier_by_mask.csv` retains every `(case, seed, mask)` row —
`p_majority`, the certified indicator, the certified coalition seat total and the
Brier — so any within-election aggregate can be recomputed independently.

---

## 6. Law dispatch

`mandate_law_for_election_year(year)` is the only source of the law version, and it
never consults the wall clock. 2014 → **PRE_2018** (first divisor 1.4, no mandate
return); 2018, 2022 → **POST_2018** (1.2, with return).

`pipeline.assert_law_dispatch(target_year, engine=...)` is a **hard guard**: the
production engine `simulate_election` always allocates under current law, so the
guard raises `LAW DISPATCH VIOLATION` if a PRE_2018 target is routed through it.
It is called before every Tier-2/3 simulation. A test asserts it fires for 2014
and passes for 2018/2022, so a future addition of 2014 to Tier 3 fails loudly
instead of silently producing a legally wrong seat vector (6 seats off, per
Part 2B).

---

## 7. Geography floor — do not attribute it to ElectionNoise

From Part 2B, feeding the **actual** national vote through the frozen geography and
the correct allocator gives:

| Chain | Constituency share MAE | Seat error vs certified | Coalition masks disagreeing |
|---|---|---|---|
| 2014 → 2018 | 0.00649 | **0** | 0 / 254 |
| 2018 → 2022 | 0.00665 | **0** | 0 / 254 |
| 2010 → 2014 | 0.00628 | 4 | 0 / 254 |
| 2006 → 2010 | 0.00677 | 8 | 10 / 254 |

The two elections in the frozen Tier-3 set have a **zero** deterministic floor: a
perfect vote forecast would reproduce the certified seat vector exactly, so all of
the observed seat and coalition score is attributable to the forecast, not to
geography. The 2014 row is recorded because 2014 is a Tier-1 case and because it
would matter if 2014 ever entered Tier 3.

---

## 8. Validation against existing artifacts

Both checks run at the *legacy* configuration, so exact equality is the expected
outcome rather than something excused by Monte Carlo noise. Results in
`harness_validation.json`.

1. **Tier 1 vs `election_layer_v2/forward_eval_2010_2022.json`.** The legacy
   `pp_noise_only` evaluation is exact, not sampled: it enumerates the K support
   points. CONTROL's Tier-1 law is that same law, so the harness's closed-form
   scores must equal the frozen artifact exactly — and do, for all three
   elections, on both the 8-party and all-9 CRPS, with support size matching the
   recorded pool size.
2. **Tier 3 vs `seat_hindcasts/seat_hindcast_summary.json`.** Re-running the frozen
   simulator at 5 000 samples / seed 12345 reproduces the 8-party seat-vector
   energy score **exactly for all 12 legacy cases**.
3. **Monte Carlo error**, quantified against the corrected exact Tier-1 anchor.

---

## 9. Known limitations

1. `N_seat = 2`. The coalition-Brier headline rests on two realized elections and
   127 effectively distinct, strongly dependent events per election. It is a
   decision input, not a hypothesis test.
2. Tier 1 has three elections but omits OpinionState and Dynamics; Tier 2/3 have
   two elections and the full pipeline. Neither tier has both properties.
3. 2014 is Tier-1 only, for the upstream-data reason in §1.
4. Tiers 2 and 3 are retrospective, not independent holdout: the frozen components
   were chosen with knowledge of 2018 and 2022.
5. CONTROL's Tier-1 predictive law has only K ∈ {3, 4, 5} atoms, so its Tier-1
   coverage figures are coarse by construction — a K-atom law cannot express a
   smooth 90 % interval. This is a property of the model under test, not of the
   harness, and is exactly what the challengers exist to probe.
6. Sampling noise is small but not zero; per-case relative five-seed SD is reported
   for every metric in `monte_carlo_stability.csv`.

## 10. Outputs

| File | Contents |
|---|---|
| `evaluation_case_manifest.json` | the frozen case set, eligibility derivations, seed streams per case, truth vectors, input hashes |
| `control_scores_by_case_seed.csv` | one row per (tier, case, seed) — every metric, unpooled |
| `control_scores_by_election.csv` | election-level aggregates per tier |
| `control_scores_summary.json` | full aggregation incl. per-seed values and the G4b short-horizon aggregate |
| `coalition_brier_by_election.csv` | D4 election aggregates and the headline |
| `mask_level/coalition_brier_by_mask.csv` | every (case, seed, mask) row for independent verification |
| `monte_carlo_stability.csv` | five-seed mean, SD and relative SD per case and metric |
| `lambda_diagnostics.csv` | D5 per case and seed |
| `harness_validation.json` | the reproduction checks of §8 |
