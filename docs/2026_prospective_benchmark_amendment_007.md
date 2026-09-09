# Amendment 007: prospective exact-draw retention for rendering and recovery

Finalized at 2026-09-09T19:40:11Z. This records when the amendment text was completed; adoption takes
effect when it is committed to the archive alongside its index entry, and it governs generations
certified after that commit. No claim is made here about its position relative to any particular
durable capture.

Amendment 004 permitted an exact-draw sidecar for every certified production generation, but only
"for the bounded period 2026-09-04 through 2026-09-12", and it superseded amendment 003's
prohibition on per-generation sidecars only for that period. On 2026-09-13 that prohibition would
therefore return.

That boundary was written when the sidecar existed to serve the benchmark. It now also carries an
operational load the benchmark never anticipated. A certified forecast is published in two stages:
the immutable generation is certified and pushed, and the website's history curve is rendered
afterwards. Rendering an already-certified generation without rerunning the authoritative
100,000-draw simulation requires that generation's joint draw matrices, because a history point's
coalition intervals are computed from joint draws and cannot be recovered from published marginal
quantiles. The exact-draw sidecar is the only artifact that retains them.

Under amendment 004's boundary that capability would end on 2026-09-13. A rendering retry from
that date onward would be forced either to rerun the authoritative forecast, publishing a different
forecast than the one certified, or to approximate the joint distribution from quantiles, which
would be a different model.

This amendment extends the retention permission of amendment 004 forward without end, for
operational purposes only: an exact-draw sidecar is retained for every newly certified production
generation, solely to support rendering of, and recovery to, an already-certified generation. The
two conditions that make a sidecar `VERIFIED` rather than merely present are unchanged -- it must be
exported from the same in-memory `SimulationResult` used for that publication, and must first
appear in the same Git commit as its certified snapshot.

**Benchmark scoring is unchanged; operational retention is extended.** Scoring rules are
untouched, and retention behaviour within the scoring window is untouched: every scheduled date this
benchmark scores falls within 2026-09-04 through 2026-09-12, where amendment 004 already permitted
retention, so no scored date gains or loses a sidecar.

This amendment is not invisible to the archive, and the earlier draft of this paragraph overstated
that. Capture manifests recorded after its adoption list amendment 007 among the amendments in
force, exactly as the append-only chain requires; the remaining scheduled captures of 2026-09-09
through 2026-09-12 will therefore carry it. What that records is the state of the protocol at capture
time. It changes no capture's forecast, no capture's draws, no timing label and no weight. The
extension of retention itself reaches only generations certified after the benchmark's final
scheduled date, which the benchmark never selects, weights or scores. The scheduled capture dates, the mechanical selection of
one generation per scheduled date, the timing-eligibility labels, the late-capture limit, the
weighting, the fair CRPS primary metric and its availability condition, the amendment 002 standard
WIS formula, the amendment 003 deterministic replay fallback, the eight-party set, the final
forecast date and the winner hierarchy are all untouched.

Everything else stays in force verbatim. Retroactive sidecar backfill remains prohibited, and this
amendment is prospective only: no sidecar is written for any generation that did not export one
from its own `SimulationResult`, and the archive's sidecar requirement continues to apply to the
generation being certified rather than to generations already archived. Sidecars belonging to
unselected intraday generations continue to receive no scoring weight. Existing capture manifests
and the amendment lists they record are not modified.

`protocol.json` and amendments 001 through 006 are not rewritten. The chain is append-only: this
amendment records the extended rule additively, and amendment 004's bounded period remains readable
in its original file.
