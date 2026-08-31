# ElectionNoise v2 — challenger implementations (Part 4)

Implementation record for the two preregistered challengers. **No target-election
score is computed or reported here**, and none was inspected while this code was
written. The evaluator, CONTROL, case set, seeds, `N`, geography, mandate logic and
truths are untouched.

Code lives in `diagnostics/election_noise_v2/challengers/`. Nothing in that package
is imported by the frozen evaluator.

## Challenger A — variance-corrected smoothed empirical bootstrap

For a centered pool `C` (K × 9, pp, rows zero-sum) and `h` on the frozen grid:

```
S_P = Cᵀ C / K                       divisor K, maximum likelihood, NO Bessel
k   ~ Uniform({1..K})
z_j ~ iid N(0,1)
ε   = (1/√K) Σ_j z_j c_j             ⇒ ε ~ N(0, S_P)
R   = (c_k + h ε) / √(1 + h²)
```

`E[R] = 0` and `Cov(R) = (S_P + h² S_P)/(1 + h²) = S_P`, exactly, for every `h`. The
`√(1+h²)` denominator is binding — without it the covariance would be `(1+h²)·S_P`,
which a test demonstrates empirically. The divisor `K` is binding because it is what
makes `Cov(R) = S_P` exact and nests CONTROL as `h → 0`.

Grid: `{0.25, 0.50, 0.75, 1.00}`. `h = 0` is excluded — CONTROL already is the
unsmoothed empirical model. One free parameter, nothing else tunable.

Disclosed and tested: `ε ∈ span{c_j}`, so A is continuous but **singular**, supported
on a `(K−1)`-dimensional subspace of the 8-dimensional zero-sum hyperplane. That is
the deliberate contrast with B.

## Nested LOEO-FIT

```
for each h in H:
    score(h) = (1/K_outer) Σ_{j∈P} ES( F^A(h, P\{j}), r_j − r̄_{P\{j}} )
h* = argmin_h score(h);   exact ties → the SMALLEST h
```

`F^A(h, P\{j})` is fitted on the inner pool and re-centered by the production
centering algorithm on that inner pool; the held-out target is the held-out residual
in that same centering. `ES` is the unchanged production `compute_energy_score`,
applied in 9-category **residual** space.

`K_outer ≥ 3` for an eligible outer target (no `K_outer = 2` fallback); `K_inner = 2`
is allowed; `K_inner = 1` is prohibited and raises. A white-box test asserts each
fold is fitted on exactly `P\{j}` and that the held-out row never enters its own
inner pool.

### One interpretive decision, recorded

The preregistration estimates the LOEO energy score "for each of the five seeds of
§D0" but writes a single `score(h)` and a single `h*`. Two readings exist: `h*`
selected once from the five-seed mean, or `h*` selected separately per seed.

**Implemented: the five-seed mean, giving one `h*` per training pool.** §D0 makes the
five-seed mean the reported and decision quantity throughout, and §G7 speaks of
"Challenger A's `h*`" in the singular. Per-seed selection would make Challenger A
five different models rather than one, and the model's law would depend on which
evaluation seed happened to be running. This is recorded in the implementation
freeze so the choice is auditable rather than buried.

### Selected bandwidths

Run at the frozen design (5 seeds × 20 000 draws), inside the training pools only:

| target | K_outer | training years | h* | exact tie |
|---|---|---|---|---|
| 2014 | 3 | 2002, 2006, 2010 | **0.75** | no |
| 2018 | 4 | 2002, 2006, 2010, 2014 | **0.75** | no |
| 2022 | 5 | 2002, 2006, 2010, 2014, 2018 | **0.75** | no |

These are held-out **residual** scores computed strictly inside each target's own
training pool. No target-election outcome was read, and no CONTROL comparison was
made. They are pinned in the freeze so the bandwidth cannot drift before scoring.

## Challenger B — Ledoit–Wolf-regularized joint Gaussian

```
S_P  = Cᵀ C / K
P₉   = I − 𝟙𝟙ᵀ/9
τ²   = tr(S_P)/8
T    = τ² P₉
d²   = ‖S_P − T‖²                     ‖A‖² := tr(A Aᵀ)/8
b̄²   = (1/K²) Σ_j ‖c_j c_jᵀ − S_P‖²
b²   = min(b̄², d²)
δ    = b²/d²        (δ := 1 if d² = 0)
Σ̃    = (K/(K−1))·[δT + (1−δ)S_P]      single Bessel correction, at the end
R    ~ N(0, Σ̃)
```

Zero tunable hyperparameters. Gaussian only. The `1/8` cancels in `δ`, verified
against the unnormalized Frobenius convention. `Σ̃𝟙 = 0`, so every draw is zero-sum
almost surely, and rank on the zero-sum subspace is 8 — full rank where A is
singular.

**Numerical policy, deliberately strict.** `Σ̃` is PSD by construction and singular
(`𝟙` is exactly in its null space), so Cholesky is unavailable and a symmetric
eigendecomposition is used. A materially negative eigenvalue raises
`NonPSDCovariance` rather than being clipped; only round-off-level structural zeros
are set to exactly `0`. The model covariance is never altered to make factorization
succeed. No case encountered in validation required this.

## Downstream

Both challengers hand pp residual draws to the unmodified
`apply_batch_simplex_transfer` (ε = 0.01 pp floor, λ rule, donor attenuation and
simplex constraints all production), then on the seat path to the frozen
`isolated.votes_to_seats` — chronological geography only, oracle mode forbidden,
historically correct mandate law. λ is descriptive only.

## RNG

Streams come only from the four reserved tokens; `rng.py` refuses any other label,
including CONTROL's `residual_index` and `sign_draw`. Sub-streams within a token use
`SeedSequence` spawn keys, so LOEO-FIT separates (bandwidth, fold, seed, role)
without inventing tokens. A's index and kernel draws are independently reproducible;
B has one Gaussian stream. No common random numbers are imposed — A's index stream is
its own reserved token, as the preregistration requires, and no artificial coupling
is introduced to stabilise comparisons.
