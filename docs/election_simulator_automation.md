# Election Simulator automation

`.github/workflows/election-simulator-publication.yml` is the scheduled
production boundary. It stages acquisition and normalization into a temporary
tree, validates that tree, compares the three model-relevant processed polling
files by a deterministic content hash, and commits a changed polling snapshot
before production starts. A source refresh that retains an old verified raw
file is reported as `SOURCE_UNAVAILABLE_USING_VERIFIED_SNAPSHOT`; it is never
reported as current merely because the fallback file exists.

The fixed schedules are:

- `0 4 * * *` UTC: 06:00 Europe/Stockholm, always publish.
- `0 6,8,10,12,14,16,18,20 * * *` UTC: 08:00–22:00 Europe/Stockholm,
  publish only after a semantic polling-input change.

The workflow also supports an explicit `workflow_dispatch` mode:
`probe` acquires/normalizes/validates only, `dry_run` runs the complete
simulation/publication/website gates in disposable trees, and `publish`
commits and pushes the certified outputs. There is no implicit force-run
switch; a manual `dry_run` or `publish` is always intentional. The repository
variable `ELECTION_AUTOMATION_ENABLED=false` disables scheduled events before
acquisition (and an absent variable is treated as false), while manual
dispatch is always allowed. Set it explicitly to `true` before enabling the
cron schedules.
`runs-on` reads the repository variable `ELECTION_SIMULATOR_RUNNER` and falls
back to `ubuntu-latest`, so a self-hosted runner can be selected if
`pollofpolls.se` blocks GitHub-hosted traffic. The explicit Stockholm date
guard stops all work after 2026-09-13.

The selected runner must provide outbound HTTPS access to the polling sources,
enough memory/time for the existing 100,000-draw production simulation, and a
working Ruby/Jekyll/Chromium toolchain. The workflow installs Chromium with
`apt` only as a GitHub-hosted Ubuntu fallback; a self-hosted runner without
`sudo apt-get` should preinstall Chromium and set `ELECTION_SIMULATOR_RUNNER`
to its label.

The publishing job checks the two repositories out as sibling directories:
`$GITHUB_WORKSPACE/simulator` and `$GITHUB_WORKSPACE/website`. This is
intentional: the website checkout must not appear as an untracked directory
inside the simulator worktree, otherwise the simulator's clean-source
certification would fail. The read-only dry-run job instead clones the public
website without credentials under `$RUNNER_TEMP`. The workflow verifies the
relevant Git roots and clean statuses before acquisition. The publishing job
also configures both fresh checkouts with the non-secret `github-actions[bot]`
identity before any polling or publication commit is possible.

Before a response is accepted, acquisition performs a source-kind semantic
probe. The homepage must contain a parseable latest-polls table; the canonical
timeseries must contain all required party columns; and every party chart must
contain parseable `date` and `pofp` observations (and parseable displayed chart
values). The two SwedishPolls CSVs are also checked for their required headers
and at least one row. A 200 response that is a block or error page therefore
fails the same gate as a transport error and is tried against the configured
preserved response before any raw file is written. If both attempts fail, a
hash-verified and semantically rechecked previous raw file is retained. The
refresh result and raw retrieval manifest expose payload-free
`acquisition_diagnostics` for each attempt: source key, method, status, final
host/path, content type, byte length, semantic PASS/FAIL, and whether the
previous file was retained. These diagnostics are safe to print in a probe run
and never include response bodies. A semantic failure is scoped to that source;
the Pollofpolls host circuit breaker remains reserved for transport/HTTP
failures, so later source keys still receive their own live attempt.

## Required GitHub setup

Enable Actions for the simulator repository and leave the workflow disabled
until its first reviewed run. Workflow permissions default to `contents: read`;
only the scheduled/manual publishing job elevates the normal `GITHUB_TOKEN` to
`contents: write`. Create a fine-grained `WEBSITE_REPO_TOKEN` scoped only to
`edvinli/edvinli.github.io` with `Contents: write`; add it as an Actions secret
in the simulator repository. Only the publishing job receives that secret.
Do not put either token in workflow arguments or logs.

The website checkout must retain its `master` branch. The workflow commits
only generated files in `files/election-simulator` and pushes that branch after
the staged Jekyll build and both browser smoke tests pass.

Cross-repository consumer tests are opt-in. They do not infer a website
checkout from a sibling path or a developer home directory. Set
`ELECTION_SIMULATOR_WEBSITE_REPO` explicitly when running the consumer harness
against a checked-out website; the normal simulator CI tests remain
deterministic and do not require that repository.

Each production event runs the existing 100,000-draw simulator once. Its
validated `SimulationResult` supplies the static publication, immutable
prospective snapshot, and history point. The history update keeps exactly one
point per calendar date *and provenance*: on a new date the previous current
point becomes `prospective_archived`; a same-day run replaces that date and
retains the old immutable archive generation. Reconstructed historical points
are copied, not rerun.

A date may therefore carry two points, and this is deliberate. The chart draws
one continuous line through the non-archived points, so the date whose official
point was just relabelled needs a reconstructed point of its own or the line
has a hole there. The `history curve backfill` stage, which runs immediately
before the history update, simulates exactly the daily dates that lack a curve
point -- normally the single previous publication day, and nothing at all when
the curve is already continuous. It is deliberately non-fatal: the curve is a
presentation of history, so a reconstruction failure logs and is skipped rather
than blocking a certified forecast.

Reconstruction also survives a poll refresh. A point reads only the polls
visible on its own date, so an appended poll leaves every earlier point valid;
`first_changed_poll_date` finds the earliest publication date whose polls
actually differ and only points from there on are recomputed. Rejecting the
whole cache on any source-hash change, as the earlier rule did, made
reconstruction a ~300-point rebuild after any refresh -- which is why the curve
stopped being extended and the gap at the end of the chart grew by a day per
publication.

### What counts as a change, and what does not

The comparison pairs polls by their **observed content** -- house, publication
date, fieldwork window, sample size and party values -- and never by
`poll_id`. A `poll_id` is a SHA-256 over an identity dict that includes the
poll's row number in the upstream CSV, and SwedishPolls publishes
newest-first, so a single appended poll renumbers every older row and changes
every id in the file. Pairing on the id reintroduced the exact collapse the
incremental rule was written to remove: no poll matched, every in-window date
registered as changed, the window start came back, and the whole curve was
re-simulated on every refresh. `2f29b61` is a worked example -- all 261 polls
the artifact recorded were still present with byte-identical content, and not
one `poll_id` among them survived.

So the invalidation rule is:

- **Not a change:** renumbered `poll_id` values, a different source hash, row
  reordering, an excluded column, or a poll appended after the recorded
  window. Every point is reused byte for byte.
- **A change:** a revised party value, sample size or fieldwork date; an added
  poll inside the window; a removed poll. Points from that publication date
  onward are recomputed, and earlier ones are still reused.
- **Exempt either way:** a `current_production` point is the published result
  rather than something re-derived, so polls never invalidate it.

A poll that arrives late -- published on a past date but absent from the
artifact -- is a genuine change and correctly invalidates that date onward.
The test for this property therefore asserts reuse *relative to*
`first_changed_poll_date` rather than forbidding recomputation outright.

The boundary itself is checked against an oracle recomputed in the test from
the raw observations, not against a percentage of the curve. A "most of the
curve must be reusable" threshold would catch the collapse above, but it would
also fail a legitimate large revision, which is a real thing that can happen
to a polling archive. Stating the boundary exactly costs nothing and cannot
misfire. One cheap guard names the specific regression on top of that:
rewriting every recorded `poll_id` and changing nothing else must leave the
boundary at `None`.

`pollster_original` has a negative control of its own, because it is the one
omitted field that does appear on a model-adjacent surface -- it is part of the
consensus pivot index. Varying only the alias, with the canonical `pollster`
and every numeric field held fixed, leaves the serialized poll, the consensus
composition, the eligible-poll count, the retained-pollster count and the reuse
boundary all unchanged. The same test asserts the alias really does reach the
consensus record, so it cannot pass by the field being ignored everywhere.

The identity's completeness is pinned too. `serialize_swedishpolls` is the
whole of what reaches the model from this file -- the columns it drops
(`source_row`, `retrieved_at`, `support_status`, `uncertain_share`, the source
URLs) never appear on the forecast path -- so a test varies each serialized
field in turn and requires the identity to notice, and requires `poll_id`
alone not to. Adding a field to the serialized poll therefore fails that test
until someone decides whether it belongs in the identity.

The archive's normal API still rejects duplicate information-set/payload
identities. Production daily and manual publish runs explicitly mark a duplicate
payload as an additional immutable generation, salted by its publication
timestamp; its deterministic payload and original draws remain unchanged.
Generation and path collisions still fail closed.

If a workflow fails before the final installation gate, the source and website
`current.json` pointers remain unchanged. The archive and generated outputs
are installed only after input, simulation, archive, static publication,
history, website build, and browser-contract validation have passed.

Recovery is derived from durable Git/artifact state. A committed polling
snapshot whose commit is newer than the source generation's
`source_git_commit` forces the next publish even when the next refresh is
semantically unchanged. If the source pointer already addresses a certified
generation but the website pointer/history or generation bytes lag, the next
run mirrors that generation without running the simulator again. This makes a
failed website push safe to retry from a fresh sibling checkout.

The summary distinguishes `DIRECT_LIVE_FETCH` from
`VERIFIED_STALE_FALLBACK`; a retained old payload is never described as a
current live source.
