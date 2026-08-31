# Same-input 2026 post-adoption diagnostic — CONTROL vs the adopted law B

**Post-adoption diagnostic only.** `ADOPT_B` was frozen in
`diagnostics/election_noise_v2/competition/decision.json` before any of this was
computed. The 2026 forecast was not an adoption input and cannot reopen model
selection. Polling inputs were not refreshed and `as_of` was not advanced.

Configuration, identical for both runs: `as_of = 2026-08-24`,
`election_date = 2026-09-13`, `N = 100 000`, `seed = 12345`.

**Pairing is exact.** Both runs go through the unmodified production engine and the
adopted-law sampler reuses the very `base_comp_matrix` the legacy run produced.
OpinionState draws, Dynamics deltas and the base composition are **bit-identical**
across the two runs, so every difference below is attributable to the ElectionNoise
layer alone.

**Baseline validated.** The CONTROL arm reproduces the certified published RC1
(`files/election-simulator/versions/20260828T201250Z-1da59168`) to within
0.00047 pp on every party vote median — below the artifact's own 0.001 pp rounding —
and reproduces its group figures exactly (Tidö mean 159.03, P(majority) 0.00286 vs
published 0.0029).

Fitted layer for 2026: `K = 6` (2002–2022), `δ = 0.795693`, `τ² = 0.911807`,
Bessel factor 1.2, sub-seed 1788386760 from the reserved token.

## Parties

| party | vote median CTL → B | 90 % vote interval CTL | 90 % vote interval B | P(≥4 %) CTL → B | seat median CTL → B | 90 % seat interval CTL | 90 % seat interval B |
|---|---|---|---|---|---|---|---|
| M | 18.599 → 18.174 | 14.53–20.23 | 15.97–20.34 | 1.0000 → 1.0000 | 68 → 66 | 53–73 | 58–74 |
| L | 1.962 → 2.009 | 1.06–3.29 | 0.43–3.56 | **0.0001 → 0.0174** | 0 → 0 | 0–0 | 0–0 |
| C | 6.936 → 6.993 | 5.86–8.22 | 5.34–8.66 | 1.0000 → 0.9985 | 25 → 25 | 21–30 | 19–31 |
| KD | 6.486 → 6.499 | 5.44–7.63 | 4.81–8.19 | 0.9999 → 0.9931 | 24 → 24 | 20–28 | 18–30 |
| S | 30.162 → 30.339 | 28.51–32.51 | 28.23–32.43 | 1.0000 → 1.0000 | 110 → 110 | 104–119 | 103–118 |
| V | 7.521 → 7.490 | 6.32–8.64 | 5.85–9.13 | 1.0000 → 0.9998 | 27 → 27 | 23–31 | 21–33 |
| MP | 7.461 → 7.408 | 5.92–8.75 | 5.61–9.23 | 1.0000 → 0.9990 | 27 → 27 | 21–32 | 20–34 |
| SD | 18.977 → 19.060 | 16.86–21.65 | 16.96–21.18 | 1.0000 → 1.0000 | 69 → 69 | 61–78 | 62–77 |
| REST | 1.981 → 1.978 | 0.49–3.57 | 0.16–4.00 | — | — | — | — |

Central locations barely move. What changes is the **tails**. Under CONTROL the
six-atom law essentially cannot carry a party across the 4 % threshold unless one of
six historical residuals happens to do so, so `P(L ≥ 4 %) = 0.0001` and four parties
sit at a hard 1.0000. Under B the threshold becomes a genuinely probabilistic event:
L rises to 0.0174, and C, KD, V and MP fall marginally below certainty. This is the
substantive behavioural consequence of replacing discrete support with a continuous
regularized law, and it is the direction the historical evaluation rewarded.

## Groups

| group | metric | CONTROL | B |
|---|---|---|---|
| Tidö (M, SD, KD, L) | mean / median | 159.03 / 159 | 159.15 / 159 |
| | p05 / p95 | 147 / 170 | 149 / 169 |
| | **P(majority)** | **0.0029** | **0.0070** |
| Red-Green-Centre (S, V, MP, C) | mean / median | 189.97 / 190 | 189.85 / 190 |
| | p05 / p95 | 179 / 202 | 180 / 200 |
| | **P(majority)** | **0.9971** | **0.9929** |

The two are exact complements (349 is odd), as they must be. Both majority
probabilities move toward the middle under B, but the qualitative picture is
unchanged.

## The original multimodality diagnostic, revisited

Descriptive only. Unimodality was never a requirement and is not a release gate.

### C+S+MP

| | mean | median | p05 | p10 | p25 | p75 | p90 | p95 | sd | P(≥175) | distinct values | material modes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CONTROL | 162.72 | 162 | 152 | 154 | 157 | 167 | 175 | 177 | 7.575 | **0.1078** | 50 | **2** |
| B | 162.62 | 163 | 152 | 154 | 158 | 167 | 171 | 173 | 6.492 | **0.0336** | 64 | **1** |

This is the case the original diagnostic flagged. CONTROL's modes sit at 161
(6.11 % mass) and **176 (2.13 % mass)** — a second lump parked immediately above the
175-seat majority line, produced by one historical residual atom rather than by any
continuous accumulation of evidence. Its p90 lands exactly on 175 and its p95 on 177.
Under B that upper mode disappears: a single mode at 162, variance falls 57.4 → 42.1,
p90/p95 retreat to 171/173, and the majority probability falls from 10.8 % to 3.4 %.
Roughly two-thirds of CONTROL's majority probability for this coalition was carried
by the discrete atom.

### S+V+MP

| | mean | median | p05 | p25 | p75 | p95 | sd | P(≥175) | distinct values | material modes |
|---|---|---|---|---|---|---|---|---|---|---|
| CONTROL | 164.53 | 165 | 155 | 162 | 168 | 173 | 5.303 | 0.0222 | 43 | 1 |
| B | 164.44 | 164 | 155 | 161 | 168 | 174 | 5.861 | 0.0428 | 60 | 1 |

Both unimodal. Here B is **wider**, not narrower — variance rises 28.1 → 34.3 and the
majority probability roughly doubles from 2.2 % to 4.3 %. B is not a uniform
variance reduction: it removes support artefacts where the discrete law had them and
adds honest tail mass where the discrete law was artificially confident. Mass within
±5 seats of 175 moves 0.1727 → 0.1375 for C+S+MP and 0.1630 → 0.1856 for S+V+MP.

All 254 non-trivial coalition masks are exported in `same_input_2026.json` under
`coalition_masks`, matching the convention the historical evaluation used.

## λ diagnostics

| | mean | min | fraction < 1 |
|---|---|---|---|
| CONTROL | 0.998654 | 0.4387 | 0.98 % |
| B | 0.989492 | 0.2384 | 5.49 % |

B's continuous tails reach further, so the bounded transfer attenuates more often.
λ stays in [0, 1] throughout. λ is descriptive; it is not a tuning parameter and has
no gate.

## Release-safety audit

`release_audit.json` — **promotion_safe: true, no problems**.

Deterministic repeat at the full production configuration is bit-identical on vote
shares, seats, λ, the ElectionNoise residual draws, `Σ̃` and the sub-seed. No
NaN/Inf. Vote compositions non-negative and summing to 100 (max deviation 4.3e-14).
Every seat draw totals 349. λ ∈ [0, 1]. OpinionState, Dynamics, `as_of`, horizon and
the training-year set are identical to the legacy run. Ten frozen production files —
geography, raking, integerization, allocator, law, engine, national engine, models,
transfer, residual pool — are unchanged. The 2026 training pool is
{2002, 2006, 2010, 2014, 2018, 2022}, all strictly before 2026, so no future data
entered. The poll, election, mandate and geography input hashes match the certified
publication and `as_of` is still 2026-08-24, so no polling refresh occurred. CONTROL
remains reproducible against the published RC1. Both freezes verify: evaluator 112
checks drift `[]`, challengers 95 checks drift `[]`.

No probability in this document is treated as a blocker. Adoption rested on frozen
historical proper scoring; a prospective probability that looks surprising is not
evidence of a defect.
