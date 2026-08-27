# Static publication contract

`scripts.static_exporter` turns a completed frozen Python
`ElectionSimulator` result into compact files for GitHub Pages:

```text
forecast.json
parties.json
seats.json
groups.json
calibration.json
metadata.json
manifest.json
current.json
versions/<generation>/{forecast,parties,seats,groups,calibration,metadata,manifest}.json
```

The exporter never stores 100,000 raw draws in the site payload and never runs
Monte Carlo in JavaScript. The page at `/election-simulator/` consumes only
these files. REST is explicitly aggregate vote mass for modeled-ineligible
parties and is absent from threshold and seat eligibility surfaces.

## Safe production orchestration

Use `scripts.publication_pipeline` as the offline-first boundary around a
production run:

```text
validate saved processed inputs (no network/no writes)
  -> require the canonical processed-data root and clean source provenance
  -> run frozen Candidate A
  -> validate vote, seat, threshold, and REST invariants
  -> append one immutable prospective snapshot
  -> publish a complete immutable version behind one atomic pointer
```

The command intentionally does not fetch data. Run the separately approved
Pollofpolls refresh first, then invoke:

```bash
uv run python -m scripts.publication_pipeline \
  --as-of 2026-08-23 \
  --samples 100000 \
  --archive-dir data/processed/prospective_forecasts \
  --publication-dir files/election-simulator
```

The default is fail-closed: a missing or invalid processed input prevents
simulation, a non-canonical `--processed-root` is rejected because the frozen
simulator does not yet support alternate input bundles, a dirty source
worktree cannot be certified, and a duplicate as-of date prevents an archive
append. Neither a failed validation nor a collision replaces the previous
static publication. An archive append is never rolled back because the archive
is immutable; if a later export fails, the append remains available for
reconciliation. The JSON result identifies each stage and reports `PUBLISHED`,
`SIMULATED`, `COLLISION`, or `FAILED`.

## Fail-safe publication

All files are assembled in a sibling staging directory and validated before
they are moved into a new immutable version directory under
`files/election-simulator/versions/<generation>/`. The stable
`files/election-simulator/current.json` pointer is then atomically replaced in
one filesystem operation. A failure before that pointer replacement leaves the
previous complete version reachable, while a failure after it can only expose
the complete new version. Old versions are retained and are never rewritten.
The prospective archive under `data/processed/prospective_forecasts/` is not
modified by the exporter.

The pointer is the authoritative publication contract. For compatibility,
top-level JSON names are refreshed as best-effort aliases after the pointer is
committed; they must not be used to decide whether a publication is complete.
The browser resolves `current.json` first and fetches all contracts from the
addressed immutable version, so stale or interrupted aliases cannot be shown as
the certified publication.

The ordinary local command is:

```bash
uv run python -m scripts.static_exporter \
  --output-dir files/election-simulator \
  --samples 100000 --seed 12345 \
  --calibration-dir data/processed
```

The generated `manifest.json` contains ordinary file hashes plus
`deterministic_content_sha256`. The latter hashes canonical JSON values after
removing runtime timestamps, so two runs with identical model inputs can be
compared without treating publication time as model content.

## Frontend semantics

The MVP shows median national vote shares, central 50/80/90% predictive
intervals, inclusive 4% probabilities for parliamentary parties, seat ranges,
and a 349-seat parliament view using one coherent simulated allocation closest
to the marginal medians. It also shows published group majority probabilities,
change since the latest earlier immutable snapshot, validation status, and
source/payload metadata. Historical validation is labelled retrospective; no
“beats PoP” claim is emitted without a completed proper-score benchmark. A
publication is labelled certified only when both the manifest and metadata
record the boolean `source_worktree_clean: true`; dirty or incomplete output
must remain visibly uncertified.

The `forecast.json` `change_since_prior` object is computed before a pipeline
run appends its current snapshot, so it cannot accidentally compare a forecast
with itself. If no earlier snapshot exists, the object is explicitly marked
`NOT_AVAILABLE_NO_PRIOR_SNAPSHOT`.
