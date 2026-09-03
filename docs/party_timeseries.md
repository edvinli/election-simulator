# Per-party time series

*Vägen till valdagen* was built for coalitions. This document specifies the
additive **party** family published alongside them, so the same chart can show
one party's history, its forward campaign paths and its election-day
distribution without the consumer deriving anything.

The construction adds **no model**, **no simulation** and **no probability**.
Every published party number is a marginal of draws the forecast already made.

---

## 1. Why a party is not a one-party coalition

The two families answer different questions and use different denominators.
That difference is the whole reason this contract exists.

| | Coalition share | Party share |
| :-- | :-- | :-- |
| Denominator | the eight parliamentary parties | all nine model categories, `REST` included |
| Answers | "what fraction of the seats-eligible vote" | "what fraction of the electorate" |
| Published helper | `coalition_vote_draws` | `party_vote_draws` |
| Matches | the coalition majority arithmetic | `parties.json`, every poll, the 4 % threshold |

`REST` is roughly 2 % of the vote, so renormalizing a party over the eight
parliamentary parties inflates it by about 2 % of its own value — around
+0.04 pp for Liberalerna and +0.6 pp for Socialdemokraterna. For a party near
the threshold that is the difference between being drawn above and below the
4 % line, which is why `party_vote_draws` **refuses** an eight-column matrix
rather than silently accepting one:

```python
party_vote_draws(votes[:, :8], "M")
# ValueError: A party vote share is defined on the nine-category model
#             composition; got 8 columns, expected 9
```

The published contract states the denominator explicitly in
`parties_view.vote_share_denominator` and in
`future_campaign_paths.path_construction.party_vote_share_denominator`, and the
validator rejects any other value. A consumer never has to guess.

The individual poll observations already in the artifact are on the same scale:
`polls[].parties` and `poll_of_polls[].parties` are the published party numbers
as reported, whose remainder is exactly the `REST` mass the model carries. A
party poll dot is therefore the raw published value, drawn without any
arithmetic. Coalition mode's renormalization is the special case, not the norm.

`REST` itself is never published as a party. It is aggregate vote mass for
modelled-ineligible parties: it cannot cross the threshold, cannot hold seats,
and is not a party a reader can follow. `parties_view.rest_is_a_party` is
`false` and the validator rejects `true`.

---

## 2. What is published

Everything is **additive**. No existing key changes type, meaning or contents,
and every party surface is a *sibling* of the coalition surface it parallels —
never a new key inside it. That is deliberate: the deployed consumer validates
`bands[].groups`, `paths.series[].values` and `election_day.groups` against the
coalition list with an exact key-set comparison, so merging party ids into
those objects would make the previous website reject the whole future region.
A test asserts the separation directly.

### 2.1 `parties_view` — one top-level declaration

```json
"parties_view": {
  "schema_version": "1.0",
  "role": "party_time_series",
  "party_order": ["M","L","C","KD","S","V","MP","SD"],
  "party_names_sv": { "M": "Moderaterna", "...": "..." },
  "vote_share_definition": "national_vote_share",
  "vote_share_denominator": "all_nine_model_categories_including_rest",
  "seat_definition": "statutory_mandate_allocation",
  "national_threshold_pct": 4.0,
  "threshold_label_sv": "4 %-spärren",
  "rest_is_a_party": false,
  "election_day_parity": {
    "guarantee": "identical_to_certified_production_party_forecast",
    "source": "certified_production_result_draw_matrices",
    "reconstructed_from_coalitions": false
  },
  "intermediate_seat_trajectory": false,
  "provenance_note_sv": "…"
}
```

Its presence is the feature flag. A history artifact without it is valid and is
exactly the artifact the previous website consumed; the validator only has an
opinion once it is present, and then it is strict.

### 2.2 `series[].parties` — the historical party forecast

```json
"parties": {
  "M": { "vote": {"p05":…,"p25":…,"p50":…,"p75":…,"p95":…},
         "seats": {"p05":…,"p25":…,"p50":…,"p75":…,"p95":…} },
  "…": "…"
}
```

Built by `build_parties_from_matrices` from the **same** joint
`vote_shares_matrix` and `seats_matrix` the point's coalition `groups` come
from. Vote quantiles carry six decimals, seats are integers — the identical
rule `contract._quantiles` applies to coalitions.

The block is optional per point. A reconstructed point generated before this
contract simply has none, and the consumer feature-detects. Archived
prospective points recover theirs from the immutable snapshot's own
`national_vote_distributions` / `seat_distributions` quantiles: unlike joint
coalition intervals, party **marginals** are recoverable from an archive.

### 2.3 `future_campaign_paths` — the party campaign region

| Key | Contents |
| :-- | :-- |
| `bands[].parties` | per-party `vote` quantiles for `path_day = 0 … n`. **Vote only.** |
| `paths.series[].party_values` | per-party daily track for the same representative draws |
| `election_day.parties` | a deep copy of the certified point's `parties` |
| `path_construction.party_vote_share_denominator` | the nine-category denominator |
| `rendering.party_units` | `["vote"]` |
| `rendering.party_election_day_units` | `["vote","seats"]` |
| `rendering.party_intermediate_seat_trajectory` | `false` |
| `rendering.national_threshold_pct` / `_label_sv` | `4.0` / `"4 %-spärren"` |

`bands[].parties[p].vote` is the quantile of that party's own column of the
same daily nine-category composition the coalition bands are reduced from —
`composition[:, i]`, no renormalization. `paths.series[k].party_values` reads
out the *same* draw `k`, with the same trajectory index and the same whole-path
sign, so a rendered party path and a rendered coalition path are two views of
one coherent simulated campaign, not two independent samples.

The family is **all or nothing**, gated on the certified anchor carrying its own
`parties`. Half a family — opinion bands with no certified endpoint to meet —
would be worse than none, and the validator rejects it in both directions: a
band with party data and no certified party election day, and a certified party
election day with a band missing its party data.

---

## 3. Election-day parity

The requirement is that the party values a reader sees at the right-hand edge
of the chart *are* the published forecast. It holds by construction in two
steps, and both are asserted.

**Step 1 — the same draws, the same rule.** `build_parties_from_matrices` reads
`SimulationResult.vote_shares_matrix` and `SimulationResult.seats_matrix`, the
matrices the certified publication summarizes. Its quantile rule is
`contract._quantiles`, which is `np.quantile` at the five published levels.
`np.percentile(x, 5)` delegates to `np.quantile(x, 0.05)` on the identical
array, so the seat quantiles are integer-identical to `parties.json` and the
vote quantiles agree to the published precision.

`ArchiveQuantileAgreementTests` pins that the rule is the *same* rule the
prospective archive already uses for its party marginals, by running both over
one shared matrix and requiring equality — so a change to either drifts loudly
rather than silently.

**Step 2 — election day is a copy, not a recomputation.**
`future_campaign_paths.election_day.parties` is a deep copy of the certified
`current_production` point's `parties`, exactly as `election_day.groups` is a
copy of its `groups`. The validator requires exact equality, so the
election-day party distribution is the production one by identity.

`assert_election_day_party_parity` is the executable statement of the
requirement: it takes the rows of the publication's own `parties.json` and
fails closed on any disagreement — votes at the published three decimals, seats
exactly.

Measured against the certified generation `20260903T110151Z-68041c74`, all
nine parties × five vote quantiles and eight parties × five seat quantiles
agree exactly.

---

## 4. Mandates

The party view publishes seats in exactly two places: the historical series
(one forecast per date) and the emphasized election-day distribution. There is
**no intermediate future mandate trajectory** and there never will be from this
contract — latent opinion has no seat allocation, because the statutory
allocator is defined on an election result, not on a poll average.

Three separate checks enforce it: `parties_view.intermediate_seat_trajectory`
is `false`, `rendering.party_intermediate_seat_trajectory` is `false`, and
`validate_party_vote_only` rejects a `seats` key inside any opinion band with a
message that says why.

---

## 5. What did *not* change

- No coalition contract, value or key. `series[].groups`, the coalition bands,
  the coalition trajectories and `election_day.groups` are byte-identical.
- No forecast probability. Nothing here is simulated; every number is a
  marginal of existing draws.
- The frozen `OpinionState` semantics, the coherent historical-trajectory
  campaign paths, the bitwise election-day parity gate, the prohibition on
  future polls and Poll-of-Polls values, and the ElectionNoise / geography /
  allocation layers are untouched.
- `future_projection` — the secondary shrinking-horizon fan — gains nothing.
  The party view has one future interpretation: campaign opinion paths meeting
  the election-day forecast.
- The history schema version stays `1.1`. The party family carries its own
  `parties_view.schema_version`, so a reader that knows nothing about parties
  reads the artifact unchanged.

---

## 6. Cost

A full history artifact roughly doubles in size, because eight party
definitions replace seven coalition definitions per point. Vote quantiles keep
six decimals to stay identical to the archive's own rule rather than being
truncated to display precision.

Backfilling party data into existing reconstructed points requires a full
`scripts.forecast_history.generate` run — the resume cache keeps old points
byte-for-byte, party block or not — at about 20 seconds per point per core.
Daily publication is unaffected: `update_history_with_production_result` copies
existing points forward and derives the new certified point's party block from
matrices it already holds, at no measurable cost.

---

## 7. Testing

`tests/test_party_timeseries.py`:

- the denominator, and that a renormalized share is rejected and is measurably
  different from the published one;
- shape, order, monotonicity, integrality and determinism;
- election-day parity against the certified party forecast, plus fail-closed
  tests for a drifted vote value and a drifted seat value;
- agreement with the prospective archive's quantile rule, and recovery of party
  marginals from a committed snapshot;
- leakage: no trajectory ending after the origin, no poll dated after it;
- structure: no intermediate mandate trajectory, no seat quantile in a band, no
  party key inside the coalition `groups`, and both directions of the
  all-or-nothing gate.

```bash
make test-party-timeseries
```
