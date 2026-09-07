# Reusing reconstructed history

`build_history()` compares a canonical model-input fingerprint for each cached
reconstructed date. It preserves an equal point verbatim. Certified prospective
and current production observations remain records of their original publication;
they are never treated as replaceable reconstructed draws.

The SwedishPolls chart feed is not OpinionState's input. The canonical simulator
reads `individual_polls.csv` for OpinionState, `pollofpolls_timeseries.csv` for its
central estimate and historical dynamics, and SwedishPolls for chronological
pre-election noise training. Consequently neither a changed chart file SHA nor a
changed chart poll ID establishes that a reconstructed forecast changed.

The versioned `reconstruction_inputs.dates` map is additive artifact metadata,
covered by the artifact's deterministic self-hash. Its entries include:

- observation date, target election, and random seed;
- canonical complete opinion poll observations eligible by publication,
  fieldwork end, and reference date; IDs and acquisition metadata are excluded;
- historical time-series compositions on/before the observation date (including
  history before the chart window, which the all-history dynamics can consume);
- the canonical selected polls, weights, and outcomes for the model's historical
  election-noise windows;
- numerical model/configuration source identity and geography inputs.

The eligible opinion history is deliberately retained beyond the usual trailing
window because the canonical estimator can expand backward when residual counts
are insufficient. Multiplicity is preserved when canonical records are sorted.
There is no rounding or tolerance in the identity comparison. Numerical source
identity is independent of the acquisition normalizer and presentation code;
`MODEL_FILES` in `effective_inputs.py` enumerates the numerical implementation.

For a legacy artifact without this metadata, the migration reads its recorded
`model_commit` using local `git show`, requires a recorded clean worktree, and
verifies the recorded SwedishPolls and time
series SHA-256 values, and fingerprints that source revision's model inputs.
It never substitutes HEAD for missing history and never fetches from the network.
If the revision or matching hashes cannot be established, reconstructed points
are cache misses. Full CI and publication checkouts retain Git history for this
reason. A successful build persists the fingerprints, so later publications do
not require the legacy source revision. Roll-in carries the original per-date
identities; it must not stamp old points with today's fingerprints.

`model_data_dir` explicitly binds the canonical simulator and the fingerprint
reader to the same processed data tree, including in process workers. The
`poll_file` and `timeseries_file` chart export arguments remain separate; callers
using alternative model data must specify `model_data_dir` too. An injected test
runner stands in for that same model/data contract.

New SwedishPolls IDs omit `source_row`; the row number remains acquisition
provenance. Chart comparisons accept legacy IDs by comparing semantic multisets,
not by joining on those IDs. That chart comparison is not a model reuse decision.

Verification includes insertion/reordering, future and undated poll exclusions,
fieldwork eligibility, real historical opinion and training revisions, missing
or mismatched legacy provenance, metadata validation, and the live committed
artifact integration test. At the September 7 refresh, the live integration test
requires only the genuinely missing September 5 curve point and preserves all
297 existing reconstructed points.
