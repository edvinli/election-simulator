# Conditional future forecast projection

The coalition history chart may show a **forward-looking conditional projection** from the latest certified forecast through election day. This is deliberately not part of the historical `series` and must never be described as future polling data.

## Interpretation

The projection answers one narrowly defined question:

> How would ElectionSimulator's election-day distribution change as the remaining campaign-dynamics horizon shrinks, if the underlying opinion state measured at the latest certified forecast remained unchanged?

For every projected calendar date:

- the `OpinionState` cutoff remains fixed at the latest certified production date;
- no future poll or Poll of Polls observation is synthesized;
- only the explicit remaining Dynamics v2 horizon changes;
- the adopted `pp_lw_gaussian` ElectionNoise law remains active;
- the normal geographic projection, exact controlled rounding and mandate allocation remain active;
- coalition vote and seat quantiles are computed from the same joint draw matrices.

The election-day point therefore has **zero Dynamics v2 horizon**, but still contains current polling-state uncertainty and ElectionNoise. It is not a deterministic point estimate. This is a deliberate projection-only boundary condition: canonical production uses `max(1, election_date - as_of)` and therefore floors its natural horizon at one day, including on election day. The frozen `simulate_election()` entrypoint is unchanged.

### ElectionNoise random-number policy

All dates in one fan use **common ElectionNoise draws**. The ElectionNoise seed is derived from the frozen origin and its natural production horizon, not from each projected date's shrinking Dynamics horizon. OpinionState draws are already common for the frozen origin; only Dynamics changes with the displayed remaining horizon. This prevents day-to-day fan movement from being dominated by independently reseeded election-error Monte Carlo draws.

At the natural horizon, the seed and complete projection path remain exactly equal to canonical production. The published `election_noise_rng_policy` value is `common_natural_horizon_seed`, and contract/scientific-parity tests guard both the common-draw policy and natural-horizon parity.

## Contract

The history JSON adds an optional top-level `future_projection` object. Existing historical `series`, `polls`, and `poll_of_polls` remain unchanged.

The projection contains:

- an immutable `anchor` copied exactly from the current production history point;
- one daily point strictly after the anchor date through the election date;
- `remaining_horizon_days` decreasing to zero;
- joint coalition `p05/p25/p50/p75/p95` summaries for both `vote` and `seats`;
- explicit rendering metadata extending the x-axis to election day;
- explicit flags prohibiting poll and Poll of Polls observations in the future region;
- Swedish disclosure text explaining the conditional assumption.

The canonical tooltip copy is:

> Framåtblickande projektion från opinionsläget 2 sep. Antar oförändrat underliggande opinionsläge; framtida mätningar är okända.

The date is generated from the actual projection origin.

## Rendering requirements

A consumer should render the future region separately from historical observations:

- x-axis maximum: election day;
- future background: very light neutral shading;
- vertical boundary at the origin labelled **Senaste prognos**;
- election-day line labelled dynamically from `election_date` (for example **Valdag 13 sep**);
- historical forecast medians remain solid;
- future medians are dashed/lighter;
- future 50% and 90% bands are visually lighter than historical bands;
- legend label: **Framåtblickande projektion**;
- no poll dots or Poll of Polls observations may appear after the projection origin.

The simulator repository publishes these semantics in the history contract. The website renderer is a separate repository and must consume these fields rather than infer future points from `series`.

## Scientific isolation

The frozen production `simulate_election()` entrypoint is not modified. `scripts.forecast_history.projection_simulator` composes the same frozen scientific components with an explicit Dynamics horizon. A regression test requires the projection path to reproduce the production vote and seat draws at the natural remaining horizon for identical inputs and seed.

This makes the new path a presentation-oriented conditional re-evaluation of the existing model, not a separately calibrated forecasting model.

## Validation

`validate_future_projection_contract()` fails closed on, among other things:

- projection dates inside the historical series;
- a state cutoff different from the current production date;
- missing or additional future dates;
- a final date other than election day;
- a non-zero election-day Dynamics horizon;
- non-monotone or invalid vote/seat quantiles;
- an anchor that differs from the certified current history point;
- future poll/PoP rendering flags set to true;
- any actual `polls` or `poll_of_polls` observation dated after the projection origin;
- missing vote or seat rendering support;
- changed disclosure or boundary labels.
