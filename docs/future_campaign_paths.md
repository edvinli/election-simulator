# Coherent forward campaign paths

The headline future visualization simulates **complete opinion trajectories**
from the certified forecast origin to election day, and emphasizes the
election-day forecast distribution at the right-hand edge. It replaces the
former headline treatment — "freeze today's opinion and shrink the remaining
dynamics horizon" — which is retained only as a clearly labelled secondary
analytical view (see [`future_forecast_projection.md`](future_forecast_projection.md)).

The construction changes **no production forecast probability**. The
election-day endpoint is *bitwise* identical to the frozen production
Dynamics v2 + ElectionNoise draw, and the published election-day summaries are
copied verbatim from the certified `current_production` history point.

---

## 1. What question the future region answers

Two different quantities share the x-axis and the contract keeps them apart:

| Region | Quantity | Contains |
| :-- | :-- | :-- |
| Historical series | election-day forecast as of that date | OpinionState + Dynamics v2 + ElectionNoise + geography + mandates |
| Historical Poll of Polls overlay | measured opinion | the published PoP consensus |
| **Future region (`bands`, `paths`)** | **latent opinion** $\theta_{t+d}$ | OpinionState at $t$ + whole-path campaign dynamics |
| **Election day (`election_day`)** | **official election-day forecast** | the certified production distribution, unchanged |

The future paths are therefore a forward continuation of **latent opinion**,
not of the forecast line. The published contract states this explicitly as
`quantity = "underlying_opinion_share"` and
`rendering.continues_from = "current_opinion_state"`.

The origin of the fan is `bands[0]`, i.e. `path_day = 0`: the model's own
estimate of *today's* latent opinion, carrying current-state uncertainty and
nothing else. It is published with its own Swedish label
(`rendering.origin_state_label` = "Opinionsläge i dag") and its own disclosure,
because it is a **different quantity** from the certified `current_production`
forecast point that sits on the same calendar date. That point additionally
carries campaign dynamics and ElectionNoise and is therefore much wider. A
consumer must draw the origin as its own marker and let the fan emanate from
it; reusing the forecast dot as the fan's origin would silently equate two
different distributions. `path_construction.origin_day_quantity` is
`opinion_state_only` and the validator rejects anything else.

The chart no longer draws the aggregate Poll of Polls series at all, so the
contract does not claim to continue it.

ElectionNoise is a *poll-to-election* structural error. It realizes once, on
election day, and is deliberately absent from every intermediate day. The
visible widening between the last path day and the emphasized election-day
distribution is that structural layer, not an inconsistency.

---

## 2. Mathematical construction

Let $t$ be the certified origin, $E$ the election date and $n = E - t$ in days.
All movement is modelled jointly in centred-log-ratio space over the
$D = 9$ canonical categories `M, L, C, KD, S, V, MP, SD, REST`:

$$\operatorname{clr}_i(\mathbf p) = \ln p_i - \tfrac{1}{D}\textstyle\sum_{j} \ln p_j,
\qquad \textstyle\sum_i \operatorname{clr}_i(\mathbf p) = 0 .$$

**Step 1 — freeze the certified state.** Draw
$\theta_t^{(i)} \sim F_{\text{state}}$ from the frozen `OpinionState v1.1`
fitted at cutoff $t$, using production's sub-seed
`derive_opinion_state_seed(base_seed, t)`. The draws are the same draws
production uses.

**Step 2 — the eligible trajectory pool.** Let $h$ be production's Dynamics v2
endpoint horizon,

$$h = \min\big(\max(1, n),\, 112\big),$$

with production's `28 / 14 / 7` fallback ladder if fewer than
$\text{MIN\_TRANSITIONS} = 30$ leakage-safe transitions exist at $h$. The pool
is production's own eligible pool,

$$\mathcal S(t, h) = \{\, s : \text{PoP}[s] \text{ and } \text{PoP}[s+h] \text{ exist},\; s + h \le t \,\},$$

sorted by transition end date. Nothing observed after $t$ can enter it.

**Step 3 — the whole-path displacement.** For trajectory $s_j \in \mathcal S(t,h)$
and display day $d = 1 \dots n$,

$$\boldsymbol\Delta_{j,d} = \operatorname{CLR}\big(\text{PoP}[s_j + \tau(d)]\big) - \operatorname{CLR}\big(\text{PoP}[s_j]\big),
\qquad \tau(n) = h,$$

where $\tau$ is the monotone day map of §4. When $n \le 112$, $\tau$ is the
identity and $\boldsymbol\Delta_{j,d}$ is literally the $d$-day movement of a
real historical campaign window.

**Step 4 — one index and one sign per draw.**

$$J^{(i)} \sim \mathrm{Unif}\{1 \dots |\mathcal S|\}, \qquad
S^{(i)} \sim \mathrm{Unif}\{-1, +1\},$$

drawn from `numpy.random.default_rng(dyn_seed)` with
`dyn_seed = derive_shared_dynamics_seed(base_seed, t, max(1, n))`, consuming
`integers` before `choice` exactly as production does. **The single sign
multiplies the entire trajectory**, so a draw is either a historical campaign
movement or its exact mirror image — never a per-day re-randomization.

**Step 5 — the path.**

$$\boxed{\;\operatorname{CLR}\big(\theta^{(i)}_{t+d}\big) = \operatorname{CLR}\big(\theta^{(i)}_t\big) + S^{(i)} \cdot \boldsymbol\Delta_{J^{(i)},\,d}\;}$$

Because each $\boldsymbol\Delta$ is a complete nine-category vector, party
movements are never sampled independently: the joint composition and the
empirical cross-party correlation structure are preserved, and the inverse CLR
map returns each day to the simplex with the row summing to exactly $100\%$
(the published `max_composition_sum_error_pp` diagnostic is $<10^{-9}$).

**Step 6 — election day.** At $d = n$ the adopted `pp_lw_gaussian`
ElectionNoise law is applied with production's
`derive_election_noise_b_seed(base_seed, t, max(1, n))`, followed by the
GeographicProjection v1 IPF raking, exact-margin biproportional controlled
rounding and the statutory `MandateAllocator v1` — the canonical production
path, unmodified.

**Not in the model.** No future poll or Poll-of-Polls value is synthesized. No
directional momentum is introduced ($\mathbb E[S\boldsymbol\Delta] = \mathbf 0$
by sign symmetry, so the median path is flat by construction). No daily
independent random walk exists: conditional on $(J, S)$ the whole path is
deterministic.

---

## 3. How election-day parity is guaranteed

Parity is **exact and bitwise**, and every link in the chain is asserted at
runtime rather than assumed.

1. **Same pool, same order.** `build_campaign_path_pool` does not construct its
   own pool. It calls `resolve_endpoint_horizon`, which returns production's
   `build_all_historical_transitions` + `filter_transitions_as_of` pool, and it
   walks that pool in order. Index $j$ therefore denotes the same historical
   transition in both models.
2. **Same endpoint arithmetic.** For $d = n$ the path displacement reduces to
   $\operatorname{CLR}(\text{PoP}[s_j + h]) - \operatorname{CLR}(\text{PoP}[s_j])$,
   the identical floating-point subtraction of the identical two arrays that
   production stores in `HistoricalTransition.clr_transition`.
   `_assert_endpoint_pool_parity` compares the two matrices with
   `np.array_equal` and fails closed on any difference.
3. **Same randomness.** `draw_trajectory_indices_and_signs` uses the same
   generator, the same production sub-seed and the same consumption order as
   `sample_shared_symmetric_dynamics`, so $J^{(i)}$ and $S^{(i)}$ are identical
   draw for draw. A test asserts
   `signs * pool[:, -1, :][indices] == sample_shared_symmetric_dynamics(...)`
   element for element.
4. **Same state and same noise.** The OpinionState and ElectionNoise sub-seeds
   are production's, derived from the same origin and the same
   `max(1, n)` natural horizon.
5. **Series continuity is required, not hoped for.** Index alignment would break
   if the daily PoP series had a gap between $s$ and $s+h$. `clr_at` raises on
   a missing date, so a punctured series fails the publication rather than
   silently producing a misaligned pool. The production series is gap-free
   daily from 2014-09-15.
6. **Verified at every build.** `simulate_campaign_paths` refuses to return
   unless its own $d = n$ draws equal the canonical production draws bitwise.
   The published `endpoint_parity.max_abs_vote_share_difference_pp` is `0.0`,
   and the contract validator rejects any non-zero value on a verified check.

   The reference comes from one of two places, named in
   `endpoint_parity.reference`:

   * `generate_national_vote_shares` — an **independent re-derivation** through
     the canonical engine. This is the default and what the unit tests use; two
     separately written code paths sharing the same frozen primitives must
     agree to the last bit, which catches a wrong seed, a misaligned pool or a
     changed ElectionNoise horizon.
   * `certified_production_result` — the `(N, 9)` percentage matrix production
     already holds in `SimulationResult.vote_shares_matrix`. A publication
     passes this so it does not run a second 100 000-draw canonical simulation
     purely to check itself. Measured, that saves 1.7 s of the ~194 s history
     stage; ~97 % of that stage is the *secondary* shrinking-horizon fan
     computing geography, integerization and statutory allocation for ten
     intermediate dates at 10 000 draws.

   The comparison is in percentage points, because both sides then apply the
   identical `* 100.0` to their own fraction matrix. Recovering fractions by
   dividing that matrix by 100 does **not** round-trip bit-exactly and would
   break the very equality being asserted.
7. **Published summaries are copies, not recomputations.**
   `election_day.groups` is a deep copy of the certified
   `current_production` point's `groups`, and the validator requires exact
   equality. The published election-day vote and seat probabilities are
   therefore the production ones by identity.

Consequences: seats are a deterministic function of the national vote matrix,
so bitwise vote parity is bitwise seat parity — asserted directly against
`simulate_election` in `tests/test_campaign_paths.py`. Every proper score
computed at the endpoint is numerically identical for the two models; the
retrospective evaluation reports `endpoint_max_abs_crps_difference = 0.0`
across all rolling origins.

The frozen `simulate_election()` entrypoint is not modified.

---

## 4. The day map $\tau$ and the 112-day cap

Production caps its Dynamics v2 horizon at the 112-day empirical support. For
an origin inside the final 112 days — every origin from 2026-05-24 onward for
the 2026-09-13 election — $n \le 112$, $h = n$, and

$$\tau(d) = d, \qquad \texttt{time\_warp = "identity"}.$$

Outside that window production's endpoint delta is already a **112-day**
movement, not an $n$-day movement, so there is no $n$-day historical
trajectory whose final day equals production's draw. The path then stretches a
112-day trajectory monotonically over the $n$ display days,

$$\tau(d) = \max\big(1, \min(h, \operatorname{round}(d \cdot h / n))\big), \qquad \tau(n) = h,$$

published as `time_warp = "monotone_stretch"`. Endpoint parity is preserved
exactly; the cost is that the intermediate days are a smoothed, time-dilated
version of a shorter real campaign and therefore understate within-window
volatility.

The reader-facing disclosure tracks the day map rather than overstating the
construction. Under the identity map it says each path is "en hel historisk
opinionsrörelse **av samma längd**"; under a stretch it says the movement is
"på *h* dagar, **tidsutsträckt över perioden**". `campaign_paths_tooltip_sv`
takes the warp and the endpoint horizon, and the validator regenerates the
sentence from the published construction, so the two cannot drift apart. This
is a presentation limitation of the existing 112-day cap, not a new modeling
assumption.

---

## 5. Published contract

The history JSON gains one additive top-level object,
`future_campaign_paths`. The existing `series`, `polls`, `poll_of_polls` and
`future_projection` keys are unchanged apart from the secondary-role fields
added to `future_projection`.

| Field | Meaning |
| :-- | :-- |
| `projection_type` | `coherent_campaign_paths` |
| `model_id` | `coherent_campaign_paths_v1` |
| `role` | `primary_future_view` |
| `quantity` | `underlying_opinion_share` |
| `origin_date` / `state_cutoff_date` | the certified production date; equal by validation |
| `path_days`, `samples` | $n$ and the Monte Carlo draw count |
| `path_construction` | CLR space, sign policy, leakage rule, pool size, endpoint horizon, day map, and explicit `false` disclaimers for synthesized polls, daily random walk and directional momentum |
| `endpoint_parity` | the bitwise guarantee, the verification flag, the observed difference and the three shared production sub-seeds |
| `bands[]` | one point for `path_day = 0 … n`, each with per-coalition `vote` `p05/p25/p50/p75/p95`. **Vote only**; `path_day = 0` is state-only |
| `paths` | a limited, deterministically selected set of individual trajectories: `count`, `selection`, `sample_indices`, and per-coalition daily vote tracks |
| `election_day` | the certified production `groups` (vote **and** seats), `includes_election_noise: true`, `includes_geography_and_mandates: true`, `label_sv: "Valdagsprognos"` |
| `rendering` | axis maximum, future-region bounds and Swedish labels including `origin_state_label` and `origin_state_tooltip_sv`, `interval_bands`, `path_units: ["vote"]`, `election_day_units: ["vote","seats"]`, `median_may_be_flat: true`, `intermediate_seat_trajectory: false`, `continues_from`, and the two future-observation prohibitions |
| `tooltip_sv` | the reader-facing disclosure, regenerated from the actual dates **and the day map** |

`path_day = 0` is the origin itself, carrying **only** current-state
uncertainty. It is what makes the fan visibly emanate from today's opinion
rather than from the election-day forecast band, and it is rendered as its own
marker rather than as the certified forecast point.

Representative trajectories are selected as evenly spaced draw indices,
`round(linspace(0, N - 1, K))`. Draw order carries no information — the draws
are i.i.d. — so this is an unbiased subsample that is also deterministic and
reproducible.

### Mandates

No intermediate future seat trajectory is published. Seats appear only in the
historical series (forecast per date) and in the emphasized election-day
distribution. `rendering.intermediate_seat_trajectory` is `false` and the
validator rejects `true`, so a consumer cannot draw a smooth deterministic
future seat curve from this contract. Latent opinion has no seat allocation:
the statutory allocator is defined on an election result, not on a poll
average.

### Validation

`validate_future_campaign_paths_contract()` fails closed on, among other
things: a trajectory ending after the origin; a declared random walk,
momentum or synthesized poll; a non-zero verified parity difference;
election-day summaries that differ from the certified point; missing
ElectionNoise or mandate flags at election day; seat quantiles inside the path
bands; non-daily or non-monotone bands; an implied intermediate seat
trajectory; a poll or Poll-of-Polls observation dated after the origin; a
missing primary role; and an identity day map whose endpoint horizon does not
equal `path_days`.

---

## 6. Behaviour and interpretation

Sign symmetry gives $\mathbb E[\boldsymbol\Delta] = \mathbf 0$, so the median
path is flat up to Monte Carlo noise. The published
`rendering.median_may_be_flat` says so, and the reader-facing copy frames the
region as *possible* paths, not as a central expectation.

Interval width grows with the remaining horizon. Measured on the
`red_green_center` coalition (all-history pool, 8 000 draws):

| origin | $n$ | 90 % width at $d=0$ | at $d = n/2$ | at $d = n$ |
| :-- | --: | --: | --: | --: |
| 2026-09-02 | 11 | 1.47 pp | 1.53 pp | 1.65 pp |
| 2026-07-15 | 60 | 1.63 pp | 2.41 pp | 3.50 pp |
| 2026-05-25 | 111 | 2.10 pp | 3.61 pp | 5.27 pp |

With eleven days remaining the coherent opinion fan is genuinely narrow: the
Poll-of-Polls consensus is a heavily smoothed average and rarely moves far in
eleven days. The corresponding election-day *forecast* 90 % interval for the
same coalition is about 5.2 pp wide. That contrast is the honest decomposition
— close to an election, most remaining uncertainty is structural
poll-to-election error rather than pending opinion drift — and the chart shows
it as a narrow fan meeting a much wider emphasized election-day distribution.

## 7. Limitations

- Above the 112-day cap the intermediate days are a monotone stretch of a
  112-day trajectory and are smoother than a genuine $n$-day movement (§4).
  The published disclosure says so rather than claiming equal length.
- Poll of Polls is the movement target, not a perfect measurement of latent
  voter opinion; the disclaimer in
  [`opinion_dynamics.md`](opinion_dynamics.md) applies unchanged.
- Trajectories are resampled whole, so the pool of *distinct* campaign shapes
  is bounded by the number of eligible historical windows, and neighbouring
  windows overlap heavily. Path-to-path independence is therefore weaker than
  the draw count suggests; the marginal per-day distributions are unaffected.
- The published PoP series is rounded to 0.1 pp, so individual rendered paths
  are visibly stepwise. That is the source data, not a rendering artefact.
- The model has no campaign-event structure: it assumes the distribution of
  future campaign movements resembles the all-history distribution of past
  movements of the same length, with symmetric direction.
