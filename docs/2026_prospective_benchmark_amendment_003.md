# Amendment 003: exact deterministic ElectionSimulator replay

Frozen at 2026-09-03 19:49:41 UTC, before the first scheduled capture and before election results were known.

The preferred ElectionSimulator evidence remains joint draws exported from the same `SimulationResult` used for the certified publication. Integration review found that this object exists only inside the production process, while the durable production archive intentionally stores compact distributions. Persisting every intraday generation would add substantial repository weight and couple the experiment to the publication boundary.

This amendment therefore permits one narrower fallback already contemplated by the original implementation brief: an exact deterministic replay of the mechanically selected certified production generation. Eligible replay requires the exact source ancestry, immutable snapshot and index row to have existed by the cutoff; a clean checkout; identical simulator code, model inputs, dependency lock, configuration, seed, model version, sample count, and input hashes; parity with the deterministic payload hash and every published national-vote, threshold, seat, and coalition summary; and a cryptographically bound sidecar that is revalidated on discovery and scoring. Any failed check makes the draws ineligible. Approximate, summary-fitted, or differently configured reruns remain prohibited.

This can affect whether ElectionSimulator satisfies its half of draw-based fallback level A, so the availability effect is explicitly material. It does not change the requirement that Botten Ada must independently have verified predictive draws, nor the fair-CRPS formula, final date, party set, or winner rule.

The machine-readable amendment also preserves and corrects an audit issue discovered during review: amendments 001 and 002 contain future-skewed `created_at_utc` literals. Their immutable files are not rewritten. Their Git commit timestamps—2026-09-03 19:21:56 UTC and 19:27:47 UTC—are the authoritative proof that both existed before this amendment and the first capture.
