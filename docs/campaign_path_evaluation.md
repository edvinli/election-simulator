# Retrospective evaluation of the coherent campaign-path model

Two separate questions, deliberately kept apart.

1. **Did the election-day endpoint model change?** It must not. The new
   visualization is only allowed to exist if it leaves every published
   forecast probability untouched.
2. **Is the newly published intermediate-day *dynamics mechanism*
   calibrated?** The path model is the first thing this project publishes for
   `t + d` with `0 < d < n`, so that part is genuinely new and needs its own
   out-of-sample evidence.

> [!IMPORTANT]
> **What this evaluation does and does not certify.** Every model below is
> conditioned on `theta_t = PoP_t`, a *deterministic* origin. These scores
> therefore validate the **campaign-dynamics path mechanism** — whether
> resampling one whole historical trajectory with one sign is a well-calibrated
> description of how opinion moves over `d` days. They do **not** measure the
> calibration of the fan the chart actually draws, which starts from the
> `OpinionState` posterior at `path_day = 0` and is correspondingly wider at
> every horizon. `OpinionState` is evaluated separately (see
> [`opinion_state_model.md`](opinion_state_model.md) and
> [`pop_baseline.md`](pop_baseline.md)); combining the two into one displayed
> interval is not scored here.

Reproduce with:

```bash
make run-campaign-path-eval                                    # n = 28 default
uv run python -m scripts.campaign_path_eval --path-days 11 --stride-days 7
```

Artifacts land in `data/processed/backtests/campaign_paths_*.csv` and
`diagnostics/campaign_paths/`.

---

## 1. Protocol

* **Rolling origins**, every 14 days (28-day paths) or 7 days (11-day paths)
  from 2019-01-01 to the last origin whose complete realized trajectory exists.
* **Leakage-safe pools**: at origin `t` only transitions with
  `transition_end <= t` are eligible, exactly as production filters them. The
  harness asserts once per run that its vectorized pool reproduces
  `build_campaign_path_pool` element for element.
* **Target**: the realized Poll-of-Polls trajectory `PoP[t + d]`, the same
  development target used throughout
  [`opinion_dynamics.md`](opinion_dynamics.md), with the same disclaimer — it
  is not assumed to be a perfect measurement of latent voter opinion.
* **State uncertainty is excluded.** Every model is conditioned on
  `theta_t = PoP_t`. `OpinionState` is common to all four models, so including
  it would add identical variance everywhere and dilute the only thing that
  differs, the dynamics layer.
* **Scores**: per-party CRPS over the eight parliamentary parties, central 50 %
  and 90 % interval coverage as calibration guardrails, and the multivariate
  Energy Score over the full nine-category composition at checkpoint horizons.
* **Seeds**: production's own `derive_shared_dynamics_seed(base_seed, t, n)`.
  The evaluation is deterministic and a repeat run is asserted equal.

### Models compared

| id | what it is |
| :-- | :-- |
| `campaign_paths` | the adopted model: one whole historical trajectory per draw, one sign for the whole path |
| `frozen_state` | the "opinion does not move" reference: `theta[t+d] = PoP_t`, a point mass at every intermediate day. It is **not** the old `future_projection` object, which is an election-day forecast carrying `OpinionState` *and* ElectionNoise at every displayed date; it isolates the one assumption that fan makes about *opinion* |
| `endpoint_fan` | the naive alternative of reusing the *election-day* dynamics spread at every intermediate day, i.e. a constant-width fan |
| `independent_walk` | the explicitly rejected alternative: `d` independently signed one-day CLR steps accumulated into a random walk |

---

## 2. Endpoint parity: nothing changed

Across **198 rolling origins** (`n = 28`) and **399 rolling origins**
(`n = 11`):

```
endpoint_bitwise_identical_all_origins : true
endpoint_max_abs_crps_difference       : 0.0
```

At every origin the path model's `d = n` composition draws are
`np.array_equal` to the frozen production `sample_shared_symmetric_dynamics`
draws, so every proper score at the endpoint is *numerically identical*, not
merely close. There is no endpoint model to adopt or reject: the endpoint is
the existing Dynamics v2 model, bit for bit. The mechanism is derived in
[`future_campaign_paths.md`](future_campaign_paths.md) §3 and gated at
publication time by an independent re-derivation through
`generate_national_vote_shares`.

The `endpoint_fan` row is the corroborating sanity check: it is *defined* to
use `Delta_n` at every day, and its scores converge to `campaign_paths`
exactly at `d = n` (CRPS 0.2103, coverage 0.523 / 0.919 for both).

---

## 3. Intermediate-day calibration

### Aggregate over all horizons (`n = 28`, 198 origins, 8 parties, 2 000 draws)

| model | mean CRPS | CRPS, interior days only | coverage 50 % | coverage 90 % |
| :-- | --: | --: | --: | --: |
| **`campaign_paths`** | **0.1199** | **0.1165** | 0.553 | **0.907** |
| `independent_walk` | 0.1220 | 0.1185 | 0.433 | 0.819 |
| `endpoint_fan` | 0.1418 | 0.1392 | 0.753 | 0.974 |
| `frozen_state` | 0.1602 | 0.1556 | 0.102 | 0.102 |

The adopted model has the best CRPS and the only 90 % coverage close to
nominal.

### By horizon (`n = 28`; CRPS / coverage 50 % / coverage 90 %)

| model | d = 1 | d = 7 | d = 14 | d = 21 | d = 28 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| `campaign_paths` | 0.0141 / .87 / .91 | 0.0688 / .55 / .88 | 0.1176 / .53 / **.92** | 0.1660 / .52 / **.91** | 0.2103 / .52 / **.92** |
| `independent_walk` | 0.0141 / .87 / .92 | 0.0689 / .45 / .86 | 0.1190 / .40 / .83 | 0.1693 / .36 / **.77** | 0.2162 / .34 / **.75** |
| `endpoint_fan` | 0.0897 / .99 / **1.00** | 0.1054 / .91 / **1.00** | 0.1335 / .76 / .99 | 0.1708 / .63 / .96 | 0.2103 / .52 / .92 |
| `frozen_state` | 0.0146 / .31 / .31 | 0.0885 / .13 / .13 | 0.1578 / .07 / .07 | 0.2246 / .05 / .05 | 0.2843 / .05 / .05 |

Energy Score over the full nine-category composition (`n = 28`):

| model | d = 1 | d = 7 | d = 14 | d = 28 |
| :-- | --: | --: | --: | --: |
| `campaign_paths` | 0.0872 | **0.2698** | **0.4498** | **0.7926** |
| `independent_walk` | 0.0871 | 0.2718 | 0.4556 | 0.8135 |

### The same picture at the live 11-day horizon (399 origins)

| model | mean CRPS | coverage 50 % | coverage 90 % |
| :-- | --: | --: | --: |
| **`campaign_paths`** | **0.0602** | 0.593 | **0.894** |
| `independent_walk` | 0.0604 | 0.531 | 0.869 |
| `endpoint_fan` | 0.0682 | 0.709 | 0.965 |
| `frozen_state` | 0.0770 | 0.172 | 0.172 |

---

## 4. What the numbers say

**Holding opinion fixed is the worst available assumption.** `frozen_state` is
a point mass, so its coverage is whatever fraction of days the Poll-of-Polls
consensus happens not to have moved: 31 % after one day, 5 % after four weeks.
Its CRPS is worse than the adopted model at every horizon beyond `d = 1`.

Read that as a statement about the *assumption*, not about the old object. The
shrinking-horizon `future_projection` is not this row: it is an election-day
forecast at every displayed date and carries `OpinionState` and ElectionNoise
uncertainty, so its intervals are wide. What it does not do is let the
underlying opinion move, and that is the assumption scored here. The case for
replacing the headline future view is that the assumption is empirically poor,
and that an election-day forecast repeated across intermediate dates answers a
different question from "where can opinion go from here".

**Whole-path resampling beats a daily random walk, and the gap grows with the
horizon.** The two models are identical at `d = 1` — a one-day path *is* a
one-day step — and diverge monotonically: coverage 90 % holds at .88–.92 for
`campaign_paths` while the random walk decays .92 → .86 → .83 → .77 → .75.
Accumulating independent daily steps discards the serial correlation of real
campaign movement and therefore understates how far opinion can travel in one
direction over several weeks. `campaign_paths` also wins on CRPS and on the
joint Energy Score at every checkpoint past `d = 1`. The requirement to avoid a
daily independent random walk is thus supported by evidence, not only by
construction.

**A constant-width fan is badly over-dispersed near the origin.** Reusing the
election-day spread at every intermediate day gives 99.9 % coverage of a
nominal 90 % interval one day out and a CRPS six times worse than the adopted
model at `d = 1`. Uncertainty about tomorrow's opinion is not uncertainty about
election day.

**Residual miscalibration.** `campaign_paths` runs a little wide at the 50 %
level (0.55 against a nominal 0.50 at `n = 28`, 0.59 at `n = 11`) and a little
narrow at `d = 1`–`d = 7` at the 90 % level (0.88–0.91). Both are small and
both are inherited from the frozen Dynamics v2 transition pool rather than
introduced by the path construction — the endpoint of that pool is the adopted
production model and is not retuned here. No parameter was fitted for this
evaluation. Again, these are the numbers for the dynamics mechanism at a
deterministic origin, not for the displayed fan.

---

## 5. Limitations

* Poll of Polls is the movement target, not a perfect observation of latent
  opinion. Every model in the table is scored against the same target, so the
  comparison is fair, but the absolute CRPS values inherit that target's
  smoothing.
* The deterministic origin is the largest single caveat. The published fan adds
  the `OpinionState` posterior at `path_day = 0`, which widens every interval,
  including at `d = n`. Nothing here says the *displayed* 90 % band covers 90 %
  of realized opinion; it says the dynamics increment that band is built from
  is close to nominal on its own.
* Rolling origins 14 (or 7) days apart share most of their transition pools and
  their realized trajectories overlap, so the 198 (399) cases are far from
  independent. The differences reported here are consistent across every
  horizon and both configurations, which is the evidence offered; no p-value is
  claimed.
* The historical data through 2026 was inspected during development of the
  underlying dynamics model, so this is **retrospective historical evaluation,
  not independent holdout validation** — the same label the SeatHindcast
  results carry.
* `independent_walk` is an evaluation-only reference model. It is not
  publishable in any case, because it cannot reproduce the production endpoint.
