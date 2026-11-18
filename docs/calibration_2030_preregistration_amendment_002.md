# Calibration preregistration, amendment 002: the 2014 seat path and 2010 vote-only cases in T4

**Status: FINAL.** Decided by the user on 2026-11-18. Finalized at
2026-11-18T21:24:48Z.

It takes effect when committed with all of the following:

- its JSON record,
  `data/processed/calibration_2030/amendments/002-t4-2014-seat-path.json`;
- that record's SHA-256 companion;
- its entry in `data/processed/calibration_2030/amendments/index.json`.

It is append-only. It amends the T4 cases of
`docs/calibration_2030_preregistration.md` §4.2 (body `7707c2af…`) and
`docs/calibration_2030_final_round_preregistration.md` §6.2–6.3 (body
`5d070b4b…`). It does not edit either document, and it changes no gate,
margin, metric, arm, seed or draw count.

## What happened

T4's decision set includes the 2014 election. The engine projects national
vote shares to constituencies from the **previous** election's constituency
result, so a 2014 forecast needs the 2010 constituency baseline. Production's
geography file
(`data/processed/geography/constituency_party_votes_2014_2022.csv`) holds
2014–2026 only.

As frozen, every 2014 case therefore cannot run, for every arm, the control
included. Separately, production's allocator path applies the post-2018
mandate law with 2026 fixed seats, which is the wrong law for 2014.

The T4 harness found this while it was being built, before any T4 score
existed. No T4 result has been computed. National vote shares are produced
before any geography and are unaffected.

## The decision

### 2014: end to end on the validated historical seat path

For every arm and every 2014 case (sets D and L):

- **Vote shares.** The national vote draws are the arm's, exactly as for the
  other elections.
- **Seats.** Seats are the deterministic map
  `votes_to_seats(votes_pct, 2014)` in
  `diagnostics/election_noise_v2/control_baseline_amendment2/harness2/isolated.py`,
  used unchanged. That map is the noise-v2 "Tier 3-ISO" path. It consists of:
  - `project_constituency_votes` in chronological mode, from the 2010
    constituency baseline in the research geography;
  - `allocate_riksdag_seats` under `mandate_law_for_election_year(2014)`:
    PRE_2018 law, first divisor 1.4;
  - the 2014 fixed seats.
- **Seat truth.** The certified 2014 national seats from
  `certified_mandates_2010_2014.csv`.

The path's validation is recorded in
`docs/election_noise_v2_historical_seat_extension.md` §§4–9:

- the allocator reproduces the certified 2014 seats exactly, in all 29 × 8
  cells, from official votes;
- fed the actual 2014 national vote, the 2010 → 2014 projection disagrees with
  the certified majority indicator on **0 of 254** coalition masks;
- its party-level error is 4 seats, all internal to blocs;
- the projection's constituency-share MAE is 0.00628, the lowest of the four
  chains tested.

**Known limitation.** Part 2B checked the masks at the realized vote only. A
systematic displacement could affect near-threshold masks elsewhere in the
predictive distribution. The seat map is identical for every arm, so it cannot
favour one arm over another. It only adds error that all arms share.

The 2018 and 2022 cases keep the production engine path unchanged.
2014 enters every gate exactly as frozen (G1–G7).

### 2010: vote metrics only, reported

The 2010 cases remain reported only (final round §6.3).

- **What is computed.** Only metrics that do not need seats: ES_vote,
  CRPS_vote, Brier_threshold, Coverage_50 and Coverage_90, from the national
  stage.
- **Why no seats.** Part 2B deferred 2010 as a seat case: its 2006 → 2010
  projection flips 5 distinct majority events.
- **Arms that cannot run.** An arm whose state model cannot be computed at a
  2010 origin, such as a U arm before the first v0.3 fit on 2010-09-26, records
  those cases as `not computable`, with no substitute.

## Pinned files (SHA-256)

| file | SHA-256 |
| --- | --- |
| `diagnostics/election_noise_v2/control_baseline_amendment2/harness2/isolated.py` | `7ff6d111e5f9c8b76c22178d210d711b06dd9ce69bbea4620b1233242a73f832` |
| `diagnostics/election_noise_v2/historical_seat_extension/processed/research_geography/constituency_party_votes_2014_2022.csv` | `65d3eababd467794df5155b276f4322d326fce34c54a60c6e48acd691f341448` |
| `diagnostics/election_noise_v2/historical_seat_extension/processed/research_geography/constituency_electorates_2014_2026.csv` | `7f340a283dbf5c4f7c252a9531d7a89e6d60ef4ea68dfc49a99a29075a4903d4` |
| `diagnostics/election_noise_v2/historical_seat_extension/processed/fixed_seats_by_year.json` | `f5fbeff5a691f28364930b66e47bf6738651aa8c00afc1c1d594136020781d5a` |
| `diagnostics/election_noise_v2/historical_seat_extension/processed/certified_mandates_2010_2014.csv` | `3af0c56f0a94724242c39da205b86f121b8b8f2d64d2aaefefd86c70541c8fcc` |
| `diagnostics/election_noise_v2/historical_seat_extension/processed/geography_validation.json` | `3b987537dddffcbe2658b285d1e7d7076aa3fe654ead4e7bedffbd92d47e500d` |
| `scripts/mandates/law.py` | `c784cda3efd5eac325a26fe8bb720abe7f0c2e0f30b78e57e51e5cf79b399a1e` |

## What it does not do

- It changes no gate, margin, metric, arm, seed or draw count.
- It does not touch production geography, the production allocator default, or
  any production file.
- It does not admit 2010 as a seat case.
- It does not use the research geography for any election other than 2014.
