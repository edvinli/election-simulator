# Protocol v4, amendment 001: release exception for G5 calibration

Finalized at 2026-09-30T07:43:12Z. It takes effect when committed with its
JSON record, SHA-256 companion and index entry. It is append-only: it does not
edit protocol v4 or any result.

**Replay v4 remains a FAIL. This amendment is an explicit release exception,
not a G5 pass.**

## What v4 found

Against the SwedishPolls-only consensus baseline, `SwedishPollsAggregate-v0.2`
passed G1–G4 and G6 and failed G5. Its 90 % intervals contained the outcome in
127 of 162 party-case intervals (Coverage_90 0.784); the absolute floor of
0.80 needs three more. Protocol v4 recorded this in advance. The baseline
covered the same 127 of 162. The 162 intervals are heavily correlated, since
they come from only three elections.

## The exception

The user decided on 2026-09-30 to release the polling-input migration despite
the G5 failure, because removing the dependency on pollofpolls.se is the
priority. This amendment permits exactly one thing:

- **Scope:** production may replace the pollofpolls.se opinion inputs with
  `SwedishPollsAggregate-v0.2` and its cleaned SwedishPolls polls, all
  together, under a new model version (1.2.0-rc1).

It does not:

- pass G5, or change any gate, margin or result in protocols v1, v2 or v4;
- change the election-noise law. `pp_lw_gaussian` is kept exactly as
  adopted. The preregistration reserves alternatives such as Student-t for a
  separate experiment, which this amendment does not start;
- claim that 90 % intervals are calibrated. **Interval calibration remains
  unresolved** and is to be studied separately. It must not become a post-score
  way to clear G5;
- treat the two real-time PoP cases (2026 at 14 and 7 days) as evidence of
  calibration. They show only that production with a real-time PoP input also
  missed the floor on those cases.

## Retrospective status

Every v4 case is retrospective. v0.2 was developed in September 2026, after
the 2018, 2022 and 2026 elections, with their outcomes known. No result in
protocols v1, v2 or v4 is independent validation.
