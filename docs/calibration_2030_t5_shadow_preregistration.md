# Calibration programme for the 2030 cycle: T5 prospective shadow run (preregistration)

**Status: FROZEN.** Drafted on 2026-11-18 on branch
`prereg/calibration-2030-drafts` and frozen the same day; the timestamp, body
hash and base commit are in the freeze block at the end and in the freeze
record. Changes after the freeze need an append-only T5 amendment.

- **Parent.** `docs/calibration_2030_preregistration.md` (FROZEN, body
  SHA-256 `7707c2af…`). Its §5 requires this document to be frozen before any
  shadow forecast is archived.
- **Sibling.** `docs/calibration_2030_final_round_preregistration.md` (the
  final round: T2, T3-rev, T4). It defines the arms that T5 logs. Where this
  document names an arm, the definition is the final round's; nothing here
  redefines it.
- **Machine record.** `data/processed/calibration_2030/t5_shadow.json`
  (frozen with this document).
- **Resolved items.** The drafter's 14 open points were resolved by the
  lead on 2026-11-18 (§9). Where the final-round preregistration and this document
  differ on a point listed in §9, §9's resolution applies.

T5 is not a protocol change of the 2026 prospective benchmark. That
benchmark's dates, selection, weighting and scoring are untouched, and T5 never
reads or writes its archive.

## 0. Purpose

Every test of the calibration programme so far is retrospective, and much of
the 2010–2026 evidence was seen before the tests were designed (parent §0;
final-round disclosure). T5 accumulates genuinely out-of-sample evidence from
now until the 2030 election, by logging, every week, the predictive
distributions of a fixed set of arms on information that is known at the
time, and scoring them only once the truth arrives.

T5 decides nothing by itself. It is the only evidence admissible for a later
2030-cycle model change (§1).

## 1. Binding rule: the one-round budget

This rule restates global rule 1 of the final round and binds every later
decision of the 2030 cycle.

1. The final round (T2, T3-rev, T4 and the release proposal that may follow
   it) is the **last retrospective design round** on 2010–2026 data for the
   2030 model.
2. After it, a model change for the 2030 cycle may be justified **only** by
   T5 prospective evidence, under a **new preregistration that cites it**
   (§6).
3. No retrospective re-analysis of 2010–2026 data, and no new arm built after
   T4's scores exist, can justify a 2030-cycle model change.

**Outside the rule.** The following are not model changes in the sense of
this rule. Each is still documented when made:

- the scheduled stand-in replacements in `docs/election_cycle_2030.md` (2030
  fixed seats, 2030 electorate);
- new data, including new polls;
- refits by frozen procedures after a new certified election result;
- correctness amendments, such as adopting `F1` (parent §4.5).

## 2. Arms

The arm list is fixed by this document. Arms are never removed. The only arm
added later is `CAND`, under §2.3. Any other addition needs an append-only T5
amendment, and an added arm starts at its first capture; it is never
backfilled.

### 2.1 The T2 arms: poll targets and the 2030 election day

| arm | definition |
| --- | --- |
| `D0` | T2 control, as defined in the final round |
| `Ua` | as defined in T2 |
| `Ub1` | as defined in T2 |
| `Ub2` | as defined in T2 |
| `Uc` | as defined in T2 |

Every arm uses the aggregate **v0.3**: T1's SH filter with the fitted segment
hyperparameters recorded in `diagnostics/calibration_2030/t1_results.json`,
plus rolling weights `W` (final-round global rule 2). The 2026 segment's fit
applies to every capture. T5 captures end before the 2030 result exists
(§3.1), so no refit occurs within T5.

Each arm logs two kinds of distribution at every capture with origin `O`:

1. **Poll targets.** The eight named parties' poll-average share (pp) at
   `D = O + h`, for **h ∈ {30, 91, 182, 365, 730} days**. The predictive
   distribution is built exactly as in T2: the arm's state uncertainty at `O`,
   its dynamics `O → D`, plus T2's **common poll-average term** `N(0, Ω_O)`,
   added identically to every arm.
2. **The 2030 election day.** The eight parties' national vote shares on
   **2030-09-08**, with `h_E = 2030-09-08 − O`. The distribution is the arm's
   state uncertainty and dynamics to `h_E`, combined with the **interim noise
   law** of `docs/calibration_2030_preregistration_amendment_001.md`
   (Gaussian, `β = 0.5`, `γ = 0`, refitted on the seven corrected residuals
   2002–2026, as tabulated there). The noise is applied as the engine applies
   it: REST = 100 − the eight shares, then `apply_batch_simplex_transfer`
   with `eps` = 0.01 pp.

Two rules govern the election-day distribution:

- **No common poll-average term.** The election-day distribution adds no
  `N(0, Ω_O)` term. The noise law plays that role for a result.
- **Election-day shocks over the open interval.** As in the final round,
  `Ua`, `Ub1` and `Ub2` add SH's `s_shock` and `h_reset` effects only for
  election days in the open interval `(O, D)`. An election on the target date
  gets no shock. This holds for every target, so the 2030 election-day
  distribution carries no 2030 shock. Poll targets dated after 2030-09-08
  carry it.

**Fitted quantities at each capture.** `k` (per U arm), `θ` (for `Ub1` and
`Ub2`), `Ω_O`, `Uc`'s eligible transitions and `D0`'s fallback are computed
by T2's causal procedures at origin `O`, using only data known at `O` in the
capture's snapshot. `k` is computed over T2's origin grid (every 14th day
from 2010-10-03), continued past 2026-11-18. The grid origins enter only
where their `h ∈ {7, 28}` targets closed on or before `O`. Each grid origin's
forecast uses data known at that grid origin.

**Recomputed at every capture.** `k`, `θ` and `Ω_O` are recomputed at
every capture by the final round's causal procedures. For `k`, that means a
pass over about 420 historical origins, roughly an hour a week; the cost is
accepted. An implementation may cache per-origin intermediate results only
where they are byte-identical recomputations on the same pinned snapshot.

`Uc` at a T5 horizon off T2's grid (30 days, or `h_E`) bootstraps at that
horizon exactly. Its fewer-than-30 fallback steps down T2's horizon grid
{7, 28, 91, 182, 365, 730, 1095, 1460}, as T2 states.

### 2.2 Full-pipeline arms

| arm | definition | logged |
| --- | --- | --- |
| `FB` | `F1` + v0.3 + interim noise (Gaussian `β = 0.5`, `γ = 0`) + production OpinionState and dynamics (112-day cap), run end to end through the T4 candidate path | from the first capture |
| `CAND` | exactly T4's `CAND`, as configured in the commit that records T4's results; the noise law is the T3-rev selection refitted by its frozen procedure on all seven corrected residuals (2002–2026) | from the first capture on or after the day T4 completes (§2.3) |

Both log the **2030 election-day** distribution of the eight parties'
national shares (and REST), from the engine's national vote draws. Neither
logs poll targets: a full-pipeline arm's poll-target distribution is that of
its T2 state-and-dynamics arm (`D0` for `FB`; the T2 selection for `CAND`),
which §2.1 already logs.

### 2.3 Adding `CAND` when T4 completes

- **T4 completes** when the T4 results record is first committed. The record
  is the one named in the final-round preregistration. If that document names
  no path, the record is `diagnostics/calibration_2030/t4_results.json`, by
  the T1/T3 convention. The completion day is that commit's UTC date.
- `CAND` is added from the **first capture whose origin is on or after the
  completion day**, whatever T4's verdict. Earlier captures are not
  backfilled.
- The activation is an append-only record,
  `data/processed/shadow_2030/activations/cand.json`. It holds:
  - the T4 results record path, its SHA-256 and its commit;
  - the code commit that implements `CAND` in the runner;
  - the **label**: T4's verdict for `CAND` (`T4_PASS` or `T4_FAIL`), and
    whether the 2026 flag was raised.
- Every capture manifest that carries `CAND` copies the label.
- If the fallback ran `FB` through the T4 gates (final-round global rule 4),
  `FB`'s verdict is recorded the same way in
  `activations/fb_t4_verdict.json`. `FB`'s logging is unchanged by it.
- If `CAND` and `FB` are the same model, both are still logged under their
  own names. This happens when T2 selects `D0` and T3-rev selects `N0`. The
  two arms share the engine seed (§3.3), so their draws must be
  byte-identical, and `verify` checks this.

### 2.4 `C0`: production

**What production archives (inspected on `origin/main` at `6c4271c`,
2026-11-18).**

- **Location.** `data/processed/prospective_forecasts/<generation_id>/`,
  indexed by `data/processed/prospective_forecasts/index.json`. The archive
  holds 61 generations targeting the 2030-09-08 election. The first is
  `20260929T214644Z-3aa1fb10` (`as_of` 2026-09-29). From 2026-09-30 onwards
  every one is model 1.2.0-rc1.
- **Cadence.** Daily from the 04:00 UTC "always publish" run, which certifies
  around 04:50 UTC. More generations follow on days when the polls change.
- **Contents.** Each 2030 generation has:
  - a `snapshot.json` with marginal histograms and summary quantiles;
  - its input hashes, including `poll_data_sha256` of
    `data/processed/pollofpolls/swedishpolls_individual_polls.csv`;
  - an **exact-draw sidecar**: `draws.npz` holds `vote_shares_pct`,
    100,000 × 9 float64 (M, L, C, KD, S, V, MP, SD, REST), and `seats`,
    100,000 × 8. It is described by `draws.json` and is about 7.4 MiB.
  - All 61 have the sidecar. Amendment 007 of the 2026 benchmark retains a
    sidecar for every newly certified production generation, without end.
- **Target.** Production forecasts **only the election day**. It produces no
  distribution of a future poll average.

**Consequences.**

- **(a) Poll targets: production's archive does not suffice.** The spec
  requires recomputation from pinned production code. That is arm `C0p`:
  - **state.** Production's OpinionState v1.1 covariance on production's own
    v0.2 reading at `O`. Both come from the v0.2 aggregate files committed at
    the anchor generation's `source_git_commit` (§3.1).
  - **dynamics.** Production's symmetric CLR bootstrap at `min(h, 112)`, with
    production's fallback rule.
  - **common term.** The same common poll-average term `N(0, Ω_O)` as every
    T2 arm, with the same standard normals.
  - **code.** The production model files are read at the anchor's
    `source_git_commit`. The runner checks that the hashed production model
    files are byte-identical between that commit and the T5 code commit. If
    they are not, it computes `C0p` in a temporary detached worktree at the
    anchor commit.

  **`C0p` is reported only.** It differs from `D0` only in the aggregate
  (v0.2 against v0.3), and T2's poll-target control is `D0`. Its common term
  is `Ω_O`, identical across arms.
- **(b) The 2030 election day: production's archive suffices.** `C0` is the
  verified exact-draw sidecar of the anchor generation (§3.1). It is
  **referenced, not copied**: by generation id, the snapshot and draws-file
  SHA-256, the deterministic payload hash, and the commit that first contains
  the snapshot. `C0` is scored from all 100,000 rows of the eight named-party
  columns.
  - A sidecar fails verification when `load_verified_draw_sidecar` in
    `scripts/simulator/exact_draw_sidecar.py` rejects it, or when its first
    archive commit is later than the capture's run start.
  - If the anchor's sidecar fails verification, the runner recomputes `C0`
    at the anchor's `source_git_commit` and snapshot. It uses production's
    seed 12345 and 100,000 draws, retains the result as `C0` draws in the
    shadow archive (§3.4), and labels it `C0_RECOMPUTED`.
  - Replay is not bit-stable across machines (benchmark amendment 004). A
    recomputed `C0` is therefore never claimed to equal the certified
    forecast.

**`C0` when production changes model.** Production may adopt a new
`MODEL_VERSION` during the cycle, for example after a T4 pass. `C0` stays
production as published:

- `model_version` is recorded per capture;
- reports split `C0` by production model version;
- no pinned-1.2.0-rc1 arm is added unless an amendment adds it.

### 2.5 Two acknowledged properties

- **`FB` equals `D0`'s model at election day.** `FB`'s election-day
  distribution has the same state, dynamics and noise law as `D0`'s. The two
  differ only in implementation path and random numbers. The comparison is a
  pipeline cross-check, reported only. It is not independent evidence.
- **Different noise pools, by design.** Production's 2030 noise pool is
  K = 6 (2002–2022), per `docs/election_cycle_2030.md`. The T2 arms, `FB`
  and `CAND` use laws refitted on K = 7. `C0` is production as published, so
  this difference is part of what `C0` measures.

## 3. Capture

### 3.1 Cadence and the anchor

- **One capture per ISO calendar week.** A week runs from Monday 00:00:00
  UTC to the following Monday 00:00:00 UTC.
- **Anchor.** The capture is anchored on the **first certified production
  generation targeting 2030-09-08 whose `generated_at_utc` is on or after
  that Monday 00:00 UTC**, and whose snapshot is first committed on
  `origin/main` before the runner starts. Normally this is Monday's 04:00 UTC
  run.
  - The anchor fixes the **polling snapshot**: the
    `swedishpolls_individual_polls.csv` and the raw SwedishPolls files at the
    anchor's `source_git_commit`. The CSV's SHA-256 must equal the anchor's
    `hashes.poll_data_sha256`, or the capture is refused.
  - The anchor also fixes `C0` (§2.4).
- **Origin.** The origin `O` (the capture date) is the anchor's `as_of`. The
  archive directory is named by it.

**The weekly snapshot.** "The first polling snapshot published on or after
Monday 00:00 UTC" is the snapshot of production's first run on or after
Monday 00:00 UTC, as above. By construction it is the snapshot `C0` used, so
`C0` pairs with the shadow arms on one information set.

**A week with no production run is missed,** with no backfill. Production may
publish no 2030 generation in a week, for example with automation disabled.
The runner never acquires SwedishPolls itself.

**Window.** The runner may start at any time in the capture's ISO week. It
never starts in the protected interval **18:30–22:00 UTC**, and it must commit
the capture on `shadow/2030` (§7) before the week ends. If a capture is not
committed by the end of the week, the week is missed. A run later in the week still uses Monday's
anchor. No target window opens before `O + 23` days, so a run in the same week
sees nothing of any target.

**End.** The last capture is the last one whose origin is on or before
2030-09-07. Literally, the spec accumulates "until the 2030 election".
Scoring continues after that until every logged poll target has closed,
which is at most `O + 737` days, about September 2032.

### 3.2 Inputs pinned per capture

Every capture records the SHA-256 of each input it read:

| input | source |
| --- | --- |
| `data/processed/pollofpolls/swedishpolls_individual_polls.csv` | at the anchor's `source_git_commit`; equals the anchor's `poll_data_sha256` |
| `data/raw/pollofpolls/swedishpolls_polls.csv`, `swedishpolls_sources.csv`, `swedishpolls_retrieval_manifest.json` | the same commit |
| `data/processed/poll_aggregate/v0_2/` timeseries and metadata (for `C0p`) | the same commit |
| `data/processed/elections/riksdag_election_results.csv`, `data/raw/elections/val2026/official_result_manifest.json` | the same commit |
| `diagnostics/calibration_2030/t1_results.json` (SH hyperparameters) | the T5 code commit |
| `data/processed/calibration_2030/amendments/001-t3-flag-interim-beta-half.json` (interim law) | the T5 code commit |
| `scripts/election_residuals/eve_consensus.py` | the T5 code commit |
| `uv.lock` | the T5 code commit |

- **Polls.** Only polls with `Observation.available` ≤ `O` enter any arm.
- **Election results.** Only results available by `O` enter any arm.
- **Parent pin.** Parent §1's poll cutoff (2026-11-18) binds the
  retrospective tiers only. T5 is prospective, and reads each week's
  snapshot.

### 3.3 Draws and seeds

- **Draws per arm.** **N = 4,000 for every arm and every target**, which is
  T2's N. That includes the `FB` and `CAND` election-day pipelines. `C0` uses
  production's own archived 100,000 draws.
- **T2 arms and `C0p`.** The seed is the integer value of the first 8 hex
  digits of `sha256("calibration-2030:T5:<O>:<target>")`, with:
  - `<O>` the ISO date;
  - `<target>` one of `h30`, `h91`, `h182`, `h365`, `h730` or `election`.

  One `numpy.random.default_rng(seed)` per (capture, target) draws the
  standard normals in T2's draw order, from the final-round preregistration.
  All arms of that target share them wherever their laws allow, as in T2.
  For `election`, the same generator then draws the noise normals
  `Z_noise` (N × 9).
- **`FB` and `CAND`.** One engine seed per capture: the first 8 hex digits
  of `sha256("calibration-2030:T5:<O>:engine")`. Both arms use it, so they
  share random numbers wherever the engine path allows.
- **`C0`.** Production's own seed (12345), as archived.

### 3.4 What is retained, format and size

Per capture, `data/processed/shadow_2030/<O>/`:

| file | contents |
| --- | --- |
| `<arm>.npz`, one per T2 arm and `C0p` | `poll_quantiles`, float32, shape (5, 99, 8): horizons {30, 91, 182, 365, 730} × levels 0.01…0.99 × parties M, L, C, KD, S, V, MP, SD; `numpy.quantile` with method `linear` over the N draws. T2 arms also hold `election_draws_centipp` (below) |
| `FB.npz`, `CAND.npz` | `election_draws_centipp` only |
| `diagnostics.json` | v0.3 reading at `O`; `Ω_O` and the number of window centres it used (at least 30 are required, as in T2); `k` per U arm with the number of errors behind it (k = 1 below 20); `θ`; `Uc`'s eligible-transition counts and any `h'` fallback; `D0`'s and `C0p`'s fallback use; `h_E` |
| `manifest.json` | capture identity and hashes (§3.5) |

**Election draws.** `election_draws_centipp` is uint16, shape (4,000, 9),
with columns M, L, C, KD, S, V, MP, SD, REST. Each value is
`rint(share_pp × 100)`, clipped to [0, 10000]. The precision is 0.01 pp,
which equals the engine's simplex `eps`.

**Encoding.** NPZ written as in `exact_draw_sidecar._npz_bytes`: ZIP_DEFLATED,
fixed member timestamps and no pickle. A retry therefore yields identical
bytes.

**Why draws for the election day and quantiles for poll targets.**

- Poll targets are scored marginally (CRPS, coverage), so 1 % quantiles
  suffice, as the spec prefers.
- The election day is retained as joint draws, so that joint metrics stay
  possible. Joint distributions are never inferred from marginals.
- `C0`'s draws stay in production's archive and are not copied, except a
  `C0_RECOMPUTED` capture (§2.4), which adds `C0.npz` with
  `election_draws_centipp` at 100,000 × 9 (about 1.8 MB raw).

**Size estimate** (raw, before deflate):

| part | arithmetic | bytes |
| --- | --- | --- |
| poll quantiles | 6 arms × 5 horizons × 99 × 8 × 4 B | 95,040 |
| election draws | 7 arms (5 T2, `FB`, `CAND`) × 4,000 × 9 × 2 B | 504,000 |
| JSON | manifest + diagnostics | ≈ 25,000 |
| **per capture** | | **≈ 0.6 MB** |

- About 32 MB a year (52 captures).
- At most about 120 MB for the cycle: about 197 weeks from late November
  2026 to 2030-09-07.
- Production's archive grows by about 7.4 MiB per generation, around 2.7 GB
  a year at its daily cadence.

### 3.5 Manifest and identity

`manifest.json` (canonical JSON, sorted keys) records:

- **identity:**
  - `capture_date` (= `O`), the ISO week, `h_E`;
  - `run_started_utc`, `run_finished_utc`;
  - `code_commit`, the T5 runner's `HEAD`; tracked files must be clean, as
    `git status --porcelain --untracked-files=no` is empty;
  - the runtime (Python, NumPy, `uv.lock` SHA-256).
- **anchor:**
  - the production `generation_id`, `generated_at_utc`, `as_of`,
    `model_version`, `source_git_commit`;
  - `snapshot_file_sha256`, `draws_file_sha256`,
    `deterministic_payload_sha256`;
  - the first archive commit and its committer time;
  - `C0`'s status: `VERIFIED` or `C0_RECOMPUTED`.
- **inputs:** every hash in §3.2, including the SwedishPolls snapshot
  SHA-256.
- **per arm:**
  - status (`OK` or `FAILED` with the error);
  - N, the seed strings and their integer seeds;
  - output file and its SHA-256;
  - the label, for `CAND` (and `FB` once a verdict exists).
- **the preregistration:**
  - this document's frozen body SHA-256;
  - the record's SHA-256;
  - the T5 amendments in force.
- **files:** SHA-256 of every other file in the capture directory.

## 4. Archive rules

### 4.1 Append-only

- A capture directory, once committed, is **never rewritten, renamed or
  deleted**.
- `data/processed/shadow_2030/index.json` is append-only. Each entry holds:
  - `capture_date`, the ISO week, `path`;
  - the `manifest_sha256`;
  - the arms present;
  - `code_commit`;
  - the SHA-256 of the previous entry, which chains the entries.
- The runner refuses to write when the committed index is not a prefix of
  the index it is about to write.

### 4.2 Missed weeks

- **No backfill.** A week without a committed capture stays missing for
  every arm.
- The next capture appends the skipped ISO weeks to the index's
  `missed_weeks` list. That records the gap. It is not a capture.

### 4.3 Failures within a capture

- **Before the commit.** The runner may be retried within the same week,
  with the same anchor and the same code commit. Its output is deterministic.
- **At the commit.** An arm that still fails is committed as `FAILED` and
  stays missing for that capture.
- **After the commit,** nothing is recomputed.

### 4.4 Code changes

- The T5 code commit changes only through:
  - the `CAND` activation (§2.3); or
  - an append-only T5 amendment (for example a bug fix).
- Earlier captures keep their outputs and their recorded commit.
- Reports separate results by code commit wherever a change could affect an
  arm.

### 4.5 Verification

`uv run python -m diagnostics.calibration_2030.t5_shadow verify` checks
every capture, or one given by `--capture <O>`:

1. **Files.** Every file hash matches the manifest, and the manifest hash
   matches its index entry. The index chain is intact.
2. **Timing.** The first commit containing the manifest has a committer time
   within the capture's ISO week and after the anchor's first archive commit.
   It is reachable from `shadow/2030` (§7). While the branch is local only,
   committer times are the only timing evidence, and they are
   self-reported.
3. **Inputs.** The anchor generation exists in production's archive with the
   recorded hashes. The polls CSV at the anchor's `source_git_commit` hashes
   to the recorded SwedishPolls snapshot SHA-256.
4. **`C0`.** Unless `C0_RECOMPUTED`, the anchor's sidecar passes
   `load_verified_draw_sidecar` with the recorded generation id and payload
   hash.
5. **Code.** The code commit exists and is an ancestor of `shadow/2030`.
6. **Replay (diagnostic only).** Rerunning the capture at its code commit and
   anchor is expected to reproduce the bytes on the same runtime. A mismatch
   on another runtime is reported, not treated as a failure (§2.4).

## 5. Scoring

### 5.1 Poll targets

The target is T2's, unchanged. At `D = O + h`:

- take every cleaned SwedishPolls observation the aggregate uses whose
  fieldwork midpoint is in `[D − 7, D + 7]`;
- compute the mean of their raw eight named-party shares, weighted by
  `Observation.sample_size`, the effective sample after overlap discount and
  imputation;
- require **at least 3 polls**; with fewer, there is no target.

**Closure.** A target is scored **14 days after its window closes**, which
allows for publication lag:

- a target is closed when `D + 7 + 14 days` ≤ the date of the snapshot used
  for scoring;
- its value is computed once, from that snapshot, and recorded with the
  snapshot's hash;
- the value is then frozen: later reports reuse it and never rescore it.

### 5.2 The election day

- **When.** The 2030 election-day distributions are scored once the
  certified 2030 result is committed.
- **Truth.** The eight parties' certified national shares of valid votes, in
  pp. They come from the official result manifest, built as for
  `val2026`.

### 5.3 Metrics

- **Primary: CRPS**, per party (pp), averaged over the eight parties, per
  (capture, target). This is T2's metric.
  - **Election day:** computed from the retained draws as T2 computes CRPS
    from draws, with
    `scripts/pollofpolls/backtest_metrics.py::calculate_crps`. `C0` is
    computed from its 100,000 sidecar rows.
  - **Poll targets:** computed from the retained quantiles as
    `(2/99) Σ_{τ = 0.01…0.99} QS_τ(q_τ, y)`, where
    `QS_τ(q, y) = (1{y < q} − τ)(q − y)`.
- **Also:** central 50 % (`[q_0.25, q_0.75]`) and 90 % (`[q_0.05, q_0.95]`)
  coverage per party, with inclusive bounds as in
  `calculate_interval_metrics`, as T2.
- **Reporting.** Results are reported per horizon, and pooled over all
  closed targets with `h ≥ 91` (the decision set of §6).

**The quantile CRPS is an approximation.** It truncates the tails beyond
1 % and 99 %, so its level is not identical to T2's draw-based CRPS. It is
applied identically to every arm, and arms are compared within T5 only.

**Other metrics.** Joint metrics (for example the energy score) may be
computed from the election-day draws and **reported only**. Such a metric
can carry decision weight only if a later preregistration names it before
seeing it.

### 5.4 Reports

- **Frequency.** At most one report per calendar quarter, as
  `docs/calibration_2030_t5_report_<YYYY>Q<q>.md` with a machine record
  under `data/processed/shadow_2030/reports/`.
- **Content.** Every arm, every scored target, `k` and `θ` over time,
  failures, missed weeks and the `FB`–`D0` implementation check, all
  reported.
- **Reports never change arms,** horizons, targets, metrics, the cadence or
  any capture. They are not decisions.

## 6. Decision use (global rule 1)

After the final round, a 2030-cycle model change requires all of the
following:

1. **A new preregistration** that cites T5, frozen before it scores anything
   it decides on.
2. **At least 26 weekly captures with closed targets at h ≥ 91 days.**
3. **CRPS as the comparison metric,** as in T2.

**Counting.** A capture counts toward the 26 when its `h = 91` target
exists (at least 3 polls) and has been scored under §5.1. The comparison
pools every closed target with `h ≥ 91`.

**Margins and gates.** T5 fixes none. The later preregistration sets them,
and must state them before it sees any T5 comparison beyond the quarterly
reports (§5.4).

**Earliest date.** With a first capture on 2026-11-23, the 26th capture
falls on 2027-05-17. Its 91-day target closes at 2027-08-23, or about
2027-09-06 with the 14-day lag. Evidence for a decision cannot
exist before September 2027.

## 7. Operations

- **Runner (local).** `uv run python -m diagnostics.calibration_2030.t5_shadow
  capture`, after `uv sync --frozen`. The module also provides `verify` (§4.5)
  and `report --quarter <YYYY>Q<q>`. The runner:
  - fetches `origin/main` and selects the anchor (§3.1);
  - reads production files only as Git objects at the anchor's commits, or
    from a fresh temporary clone;
  - refuses a dirty checkout or the 18:30–22:00 UTC interval;
  - writes **only** under `data/processed/shadow_2030/`.
- **No interference with production.** The runner:
  - never takes the `election-simulator-production` or
    `election-simulator-render` concurrency groups, or any benchmark lock;
  - never invokes publication, certification, rendering or deployment
    code;
  - never writes `data/processed/prospective_forecasts/`, `files/`, any
    production data path or the website repository.

  T5 failures are T5 failures only. They never block, delay or alter a
  production generation.
- **Branch.** Captures are committed on a dedicated branch, `shadow/2030`,
  which is **local only for now**. The runner refuses to write when the
  checked-out branch is not `shadow/2030`, and in particular it never writes
  to `main`. The branch is never rewritten. Shadow commits therefore never
  race production's bot pushes to `main`.
- **Pushing `shadow/2030`, and CI scheduling** of the runner, are separate
  decisions for the user. This
  preregistration adds and changes no `.github/workflows` file. A CI job, if
  adopted later, uses its own concurrency group.

## 8. Preconditions and order

1. Freeze this document (§10). The parent §5 requires the freeze before any
   capture.
2. Freeze the final-round preregistration. The T2 arm definitions and draw
   order must be fixed.
3. Commit the runner, with:
   - the T2 arm implementations;
   - `C0p`;
   - the `FB` engine path, whose candidate path reproduces `C0` bytes when
     configured as `C0` (the final round's self-check).
4. Run the first capture in the next ISO week that has an anchor.

T5 does not wait for the T2 or T4 *results*. `CAND` joins under §2.3.

## 9. Resolved items

1. Scope of "model change" (§1): stand-in replacements, new data, refits by frozen procedures after a new election result and correctness amendments (like `F1`) are outside rule 1 — resolved by the lead, 2026-11-18.
2. No `Ω` poll-average term in the election-day distribution (§2.1) — resolved by the lead, 2026-11-18.
3. Election-day shocks over the open interval `(O, D)` everywhere; no shock for an election on the target date (§2.1) — resolved by the lead, 2026-11-18.
4. `k`, `θ`, `Ω` recomputed every capture by the final round's causal procedures; byte-identical per-origin caching only (§2.1) — resolved by the lead, 2026-11-18.
5. `C0p` is reported only, with `Ω_O` (§2.4) — resolved by the lead, 2026-11-18.
6. `C0` is production as published; reports split by production model version (§2.4) — resolved by the lead, 2026-11-18.
7. The weekly snapshot is production's first run on or after Monday 00:00 UTC (§3.1) — resolved by the lead, 2026-11-18.
8. A week with no production run is missed, no backfill (§3.1) — resolved by the lead, 2026-11-18.
9. N = 4,000 for every arm, including `FB` and `CAND`; `C0` uses its archived 100,000 (§3.3) — resolved by the lead, 2026-11-18.
10. Election-day draws retained as uint16 at 0.01 pp (§3.4) — resolved by the lead, 2026-11-18.
11. Poll targets scored 14 days after the window closes, then frozen (§5.1) — resolved by the lead, 2026-11-18.
12. CRPS from 1 % quantiles accepted, an approximation applied identically to every arm (§5.3) — resolved by the lead, 2026-11-18.
13. Count captures with a scored 91-day target; margins and gates set by the later preregistration before any T5 comparison beyond the quarterly reports (§6) — resolved by the lead, 2026-11-18.
14. Captures on a local-only `shadow/2030` branch; the runner refuses to write to `main`; pushing and CI are separate user decisions (§7) — resolved by the lead, 2026-11-18.

## 10. Freeze procedure (to be applied at freeze)

The parent's scheme (§7) applies:

- **`BODY SHA-256`** covers every byte before the last freeze-block marker.
  The marker is appended only at freeze.
- **The freeze record** is `data/processed/calibration_2030/t5_shadow.json`.
  At freeze it is set to:
  - `status` `frozen`, `frozen` `true`;
  - the body hash;
  - the actual UTC freeze timestamp;
  - the freeze base commit.
- **The record's own hash** goes in a `.sha256` companion, created at freeze.
- **The freeze commit** is reported with the freeze.

A verification snippet must split the marker string, as the parent's does.

<!-- FREEZE-BLOCK-START -->

```
PREREGISTRATION STATUS: FROZEN

FREEZE TIMESTAMP (UTC): 2026-11-18T18:54:43Z
FREEZE BASE COMMIT:     0188f75b9dfa9315343ba5aec2b5444774e1ddf7
FREEZE COMMIT:          the commit that adds this block (reported with the
                        freeze; it cannot be embedded in the file it hashes)
BODY SHA-256:           bf93c12843f68c0ab24beead41356fa30a214c5c56b02aa7ba79ba38ef6a97a3
DECIDED BY:             user (plan approval of 2026-11-18); open items resolved by the lead
RECORD:                 data/processed/calibration_2030/t5_shadow.json
```

BODY SHA-256 covers every byte of this file strictly before the last
occurrence of the freeze-block marker. Verify with:

```bash
python3 - <<'PY'
import hashlib, pathlib
b = pathlib.Path("docs/calibration_2030_t5_shadow_preregistration.md").read_bytes()
marker = b"<!-- FREEZE-BLOCK-" + b"START -->"
print(hashlib.sha256(b.rsplit(marker, 1)[0]).hexdigest())
PY
```
