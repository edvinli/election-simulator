# Static publication contract

`scripts.static_exporter` turns a completed frozen Python
`ElectionSimulator` result into compact files for GitHub Pages:

```text
current.json
versions/<generation>/{forecast,parties,seats,groups,calibration,metadata,manifest}.json
```

That is the entire canonical contract. The pointer plus one immutable version
directory; nothing else is written. The seven files in a version directory are
always **real files, never symlinks**, so they survive static hosting and can be
committed to a Jekyll site unchanged.

`<generation>` is the sortable canonical generation id
`YYYYMMDDTHHMMSSZ-<snapshot_id_prefix>`, shared verbatim with the prospective
archive snapshot the publication was built from. A published version therefore
always joins back to exactly one archived forecast.

### Legacy flat layout (frozen, never regenerated)

Publications made before the 2026-08-27 repository extraction served seven flat
files from the publication root with no pointer. Those URLs are already public
and are **frozen exactly as they are**: they are never rewritten, never
regenerated, and never copied into `versions/`. The exporter no longer creates
flat aliases of any kind.

The browser consumer still reads that layout, but only through a fallback that
fires on a literal `404` from `current.json`. A pointer that exists but is
malformed is a hard error and is never bypassed. A read-only copy of the last
flat publication is kept at
`tests/fixtures/legacy_flat_publication_2026_08_27/` so this compatibility path
stays covered by tests.

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
simulator does not yet support alternate input bundles, an unresolvable source
commit or a dirty source worktree cannot be certified, and a repeated forecast
prevents an archive append. Neither a failed validation nor a collision
replaces the previous static publication.

Several immutable forecasts may be archived and published on the same calendar
day. What must stay unique is the forecast itself: the snapshot identity, the
deterministic payload hash, and the sortable generation id. Re-running an
identical forecast is a `COLLISION`; running a genuinely new one later the same
day is a normal second generation. An archive append is never rolled back because the archive
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

The pointer is the authoritative publication contract and is always the last
write of a publication. No flat aliases are produced, so there is no second,
ambiguous surface that could disagree with it. The browser resolves
`current.json` first and fetches every contract from the addressed immutable
version.

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

## Provenance

Every artifact this repository generates records:

| Field | Meaning |
| :--- | :--- |
| `source_repository` | `edvinli/election-simulator` for everything published after the extraction. **Absent on historical 1.0 artifacts, where it means `edvinli/edvinli.github.io`.** Historical payloads are never rewritten to add it. |
| `source_git_commit` | The clean source commit used for generation. An unresolvable commit is a **hard certification failure**: the sentinel `unknown_git_commit` must never reach a published or archived artifact. |
| `source_worktree_clean` | Must be the boolean `true` to certify. |

### Schema versions

Publication schema `1.1` adds `source_repository` to `metadata.json` and
`manifest.json`. Archive schema `1.1` adds `generation_id` to snapshots and
index entries and moves snapshots to `<generation_id>/snapshot.json`.

Only `1.1` is written. Validators accept both `1.0` and `1.1` so every
historical publication and archive entry stays readable and valid without
modification.

## Cross-repository publishing

Publishing to the website is a separate, explicit step. `scripts.site_publisher`
never simulates, never commits, and never pushes:

```bash
uv run python -m scripts.site_publisher --site-repo ../edvinli.github.io
```

It requires `--site-repo`, validates the certified source generation, refuses to
overwrite an existing destination generation, copies exactly seven real files,
validates the destination independently, and only then writes the website's
`current.json`. Review and commit the resulting working-tree change yourself,
committing the version directory before the pointer.

## Browser contract tests

Contract coverage is deliberately split into two layers, and they are not
interchangeable.

### ACTUAL_BROWSER_CONSUMER_TEST — authoritative

`tests/test_actual_browser_consumer.py` with
`scripts/static_exporter/contract/actual_consumer_harness.js`.

The harness reads `edvinli.github.io/assets/js/election-simulator.js` **byte for
byte**, evaluates that source, and asserts on the terminal
`#election-app-status` text the production file produces. Every acceptance rule
it exercises is the deployed rule, because the deployed source is what runs. The
harness records the source path and its SHA-256 in its verdict, and a test
asserts both match the real file, so it cannot silently drift onto a substitute.

It supplies only the small closed set of browser globals the file touches
(`document.getElementById` / `createElement`, `getAttribute` / `setAttribute`,
`querySelector`, `appendChild`, `addEventListener`, and the `hidden`,
`className`, `textContent`, `innerHTML`, `style`, `value` properties, plus
`fetch`, `crypto.subtle`, `TextEncoder`). It does not use jsdom: the file's
accept/reject decisions never read back from the DOM, so a real DOM would add a
heavy dependency without making the validation path more authentic.

This test requires an `edvinli.github.io` checkout. It **skips** when one is
absent — including in CI — and points at an alternative with
`ELECTION_SIMULATOR_WEBSITE_REPO`.

### REFERENCE_CONTRACT_TEST — portable, not authoritative

`tests/test_reference_publication_contract.py` with
`scripts/static_exporter/contract/reference_publication_validator.js`.

An **independent reimplementation** of the same acceptance rules. It is not
extracted from, shared with, or byte-identical to the production file. It exists
so the exporter has a dependency-free contract check that runs anywhere,
including CI with no website checkout.

Because it is a second implementation, it can drift from production in exactly
the way the original gap allowed. It must never be described as testing the real
consumer. `ReferenceValidatorDriftTests` is the tripwire: it asserts that a set
of operand-level acceptance predicates appears in both the production file and
the reference, and fails when either side changes a rule.

### Eliminating the duplication

The duplication is deliberate and temporary. It resolves when the website
repository is cleaned up: the production file's load-and-validate half moves
into one shared module that both the site and this repository import, replacing
the reference reimplementation rather than sitting alongside it. Until that
happens, the ACTUAL test is what closes the gap and the REFERENCE test is a
convenience.
