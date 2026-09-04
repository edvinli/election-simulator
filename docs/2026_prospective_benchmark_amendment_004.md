# Amendment 004: same-result exact-draw sidecars for every certified production generation

Frozen at 2026-09-04 16:01:35 UTC, before the first durable 2026-09-04 capture (cutoff 23:30 Europe/Stockholm) and before election results were known.

Amendment 003 froze a narrow `REPLAY_VERIFIED` path on the premise that an exact deterministic replay could stand in for a sidecar exported from the certified production `SimulationResult`. Post-merge rehearsal of that path found the premise does not hold: replay is not bit-stable across separate GitHub-hosted runners. A rerun on a different runner did not reproduce the certified draws bitwise despite matching source commit, model inputs, seed, model version, and dependency lock.

That leaves amendment 003's storage policy in conflict with the experiment it was written to protect. Its sentence "Do not alter the existing compact production archive or emit sidecars for every intraday generation" would deny exact draws to any scheduled date whose mechanically selected generation is an intraday one, with replay no longer a dependable substitute.

This amendment therefore supersedes that one sentence, and only its prohibition on emitting sidecars for every intraday production generation. For the bounded period 2026-09-04 through 2026-09-12, an exact draw sidecar may be retained for every certified production generation, provided it is exported from the same in-memory `SimulationResult` used for that publication and first appears in the same Git commit as its certified snapshot. Those two conditions are what make the sidecar `VERIFIED` rather than merely present: the first ties it to the published forecast, the second makes its existence at publication time provable from the commit graph alone. Deterministic replay under amendment 003 remains available, but only as a fallback if it is ever needed.

Everything else is unchanged and stays in force verbatim. Benchmark selection remains one mechanically selected generation per scheduled date; sidecars belonging to unselected intraday generations receive no scoring weight; retroactive sidecar backfill remains prohibited; and the cutoff rules, the fair CRPS and standard WIS formulas, the eight-party set, the final forecast date, and the winner hierarchy are untouched. Amendment 003's eligibility list and its prohibition on partial, approximate, differently configured, or summary-fitted reruns also remain in force.

The effect on primary scoring is availability, not rule. Retaining these sidecars can change which draws exist and therefore which pre-registered fallback tier is reachable. It changes no scoring rule, and it does not relax the requirement that Botten Ada independently have verified predictive draws.

Amendments 001, 002 and 003 and `protocol.json` are not rewritten. The chain is append-only: this amendment records the superseding rule additively, and the superseded sentence remains readable in its original file.
