# ElectionNoise — the adopted layer

## What changed

The ElectionNoise layer applies a joint, zero-sum, percentage-point election-error
vector to the `state_plus_dynamics` composition through the bounded simplex transfer
(`docs/election_layer_v2.md` §2, unchanged).

**Former layer — `pp_centered_noise`.** An empirical discrete bootstrap: one of the
`K` centered historical final-poll-to-election residual vectors was drawn uniformly
and applied. With `K = 3…6` the predictive law was supported on a handful of atoms.

**Adopted layer — `pp_lw_gaussian`.** A Ledoit–Wolf-regularized joint Gaussian on
the same residual pool, in the same 9-category percentage-point space:

```
S_P = Cᵀ C / K                      P₉  = I − 𝟙𝟙ᵀ/9
τ²  = tr(S_P)/8                     T   = τ² P₉
d²  = ‖S_P − T‖²                    b̄²  = (1/K²) Σ_j ‖c_j c_jᵀ − S_P‖²
b²  = min(b̄², d²)                   δ   = b²/d²   (δ := 1 if d² = 0)
Σ̃   = (K/(K−1))·[δT + (1−δ)S_P]     R   ~ N(0, Σ̃)
```

with `‖A‖² := tr(A Aᵀ)/8` and the Bessel correction applied exactly once, at the
end. **Zero tunable hyperparameters** — `δ` is a closed-form function of the pool.
Gaussian only: no Student-t, no ridge, no tail multiplier, no recency or
residual-year weighting.

B preserves the joint zero-sum structure of election error — `Σ̃𝟙 = 0`, so every
draw is zero-sum and enters the unchanged transfer exactly as the discrete law did —
while replacing the small discrete support with a continuous regularized
distribution that is full rank (8) on the zero-sum subspace.

## Why it was adopted

A preregistered comparison (`docs/election_noise_v2_preregistration.md`, frozen
before any challenger was scored) evaluated CONTROL against two challengers on
2014, 2018 and 2022, at 20 000 draws × 5 seeds, on the leakage-safe isolated path.
B improved the primary Tier-1 9-category joint vote energy score by **9.59 %**
relative to CONTROL (3.133231 → 2.832796) and passed every frozen adoption gate.
Full results: `diagnostics/election_noise_v2/competition/RESULTS.md`.

**Adoption was based on historical proper-scoring results, not on the 2026
forecast.** The decision `ADOPT_B` was frozen in
`diagnostics/election_noise_v2/competition/decision.json` before any prospective
forecast under B was computed or inspected.

## What this does not claim

This is not a claim of universal superiority. B won a specific preregistered
comparison on three elections under one frozen evaluation design, with `K = 3/4/5`
training residuals. The evidence base is small and the limitations recorded in
§E.5 of the preregistration continue to apply. In particular CONTROL's coalition
probabilities were structurally coarse (multiples of `1/K`), which the
preregistration flagged in advance as making the coalition-Brier threshold
relatively easy for any continuous challenger; the decision therefore rested on the
Tier-1 vote criterion, where B also won comfortably.

Challenger A — a variance-corrected smoothed empirical bootstrap — also passed every
gate, at +4.86 %. The frozen resolution rule selected B on the lower Tier-1 energy
score.

## Production status

**Challenger B is the production ElectionNoise law.** The ordinary entry points —
`generate_national_vote_shares` and `simulate_election` — default to
`noise_model="pp_lw_gaussian"`, and the reproducibility manifest records it.

## The former layer is preserved

`pp_centered_noise` is **unmodified** and remains explicitly selectable:

```python
simulate_election(as_of=..., noise_model="pp_centered_noise")
```

so **archived RC1 forecasts stay reproducible**. It is retained for historical
reproduction, regression testing and archived-forecast replay, and is still
exercised by the test suite. It was not removed, and it is not a defective
implementation: it computed exactly what it was specified to compute. What the
preregistered evaluation found was *model fragility* — a predictive law supported on
only `K = 3…6` discrete atoms is highly sensitive to individual historical residual
years, and can place material probability mass on a single atom that happens to sit
just past a decision threshold.

## Implementation

| role | file |
|---|---|
| adopted law (production) | `scripts/vote_share_calibration/election_noise_b.py` |
| vote-share sampler, either law | `scripts/vote_share_calibration/production_national_engine.py` |
| full-pipeline runner, either law | `scripts/simulator/production_runner.py` |
| frozen research implementation | `diagnostics/election_noise_v2/challengers/challenger_b.py` |
| equivalence tests | `tests/test_production_challenger_b.py` |

The production implementation is written independently of the frozen research one
and is asserted bit-identical to it on every intermediate quantity, on the generated
draws and after the simplex transfer, including on the real 2026 training pool.

The layer draws from the reserved seed token `election_noise_v2_b_normal` under the
unchanged SHA-256 token convention, so repeated runs at the same
`(as_of, election_date, N, seed)` are bit-identical.

## Version

Following the existing convention in `scripts/simulator/config.py` (semver + `-rcN`,
release tag mirroring it), the adopted layer carries:

```
MODEL_VERSION  1.1.0-rc1        (from 1.0.0-rc1)
RELEASE_TAG    election-simulator-v1.1-rc1
candidate      B                (from A)
```

These are declared in `scripts/vote_share_calibration/election_noise_b.py` and take
effect in published artifacts when the production default is flipped.

## Two freeze scopes — read this before interpreting a verifier

The default flip intentionally changed four files that sit inside the **historical**
research freeze closures. That drift is expected and is not a defect, but it means
the two freezes answer different questions and must not be conflated.

| scope | question it answers | artifact | state |
|---|---|---|---|
| **Historical research freeze** | *Is the experiment that selected B still reconstructible?* | `control_baseline_amendment2/evaluator_freeze.json`, `challengers/challenger_implementation_freeze.json` | Verifies at its referenced historical commits. Both artifacts are preserved byte-for-byte and were not regenerated. |
| **Post-adoption production freeze** | *Is the current production configuration the certified one?* | `production_promotion/production_freeze.json` | Certifies current HEAD. |

Running the historical verifiers against current HEAD reports drift in exactly these
four files, all changed deliberately:

| file | intentional change |
|---|---|
| `scripts/vote_share_calibration/national_engine.py` | `noise_model` dispatch; default is the adopted law |
| `scripts/simulator/engine.py` | passes `noise_model` through; manifest records the law actually used |
| `scripts/simulator/config.py` | `MODEL_VERSION`, `RELEASE_TAG`, `ADOPTED_CANDIDATE` |
| `scripts/simulator/reproducibility.py` | fallback manifest default |

The mathematical implementations of **both** laws are untouched:
`challengers/challenger_b.py`, `vote_share_calibration/models.py`,
`election_layer_v2/transfer.py` and `election_layer_v2/residuals_pool.py` all still
match their recorded hashes, as do every competition-result artifact and the frozen
`ADOPT_B` decision.
