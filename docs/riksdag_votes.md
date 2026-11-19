# Riksdag roll-call votes (phase 0 data pipeline)

A data pipeline for a riksdag voting tracker: party lines, party agreement, a
two-dimensional voting map and per-member statistics from the riksdag's open
roll-call data. It is independent of the election forecast — it neither
imports nor writes anything in the simulator, certification, publication or
rendering code.

```
uv run python -m scripts.riksdag_votes.build --sessions 2022/23 2023/24 2024/25 2025/26 2026/27
```

Options:

* `--current` instead of `--sessions`: the sessions a daily update rebuilds —
  every begun session of the electoral term containing today's UTC date
  (2026/27 on 2026-11-19), plus the whole previous term from 1 September to
  31 October of an election year, because its last session can still vote in
  September before the new riksdag convenes;
* `--update`: re-download the session listings (register and bulk archive) to
  pick up new votes; per-vote and per-document files stay cached;
* `--refresh`: re-download everything; `--offline`: use only cached raw files
  the manifest attests to (no network at all);
* `--site-dir DIR`: also write the website layout (below) into `DIR`;
* `--no-titles`: skip vote titles; `-v`: log progress.

`python -m scripts.riksdag_votes.site --site-dir DIR` writes the website layout
from the committed outputs without building.

## Sources

| Source | Used for |
| --- | --- |
| `https://data.riksdagen.se/dataset/votering/votering-<YYYYZZ>.json.zip` | Bulk rows: one JSON file per vote, `{"dokvotering": {"votering": rows}}` (`rows` is a dict for a single row; decoded as `utf-8-sig`). |
| `https://data.riksdagen.se/voteringlista/?rm=<rm>&gruppering=votering_id&utformat=json` | The register: every vote the API recognises, with its Ja/Nej/Frånvarande/Avstår totals. |
| `https://data.riksdagen.se/votering/<id>/json` | Metadata (beteckning, date) for votes missing from the bulk archive or with `datum: null`. |
| `https://data.riksdagen.se/voteringlista/?rm=<rm>&bet=<beteckning>&utformat=json` | Rows of a registered vote that is missing from the bulk archive. |
| `https://data.riksdagen.se/utskottsforslag/<dok_id>.json` | Vote titles: the document title and the heading (`rubrik`) of every förslagspunkt. |
| `https://data.riksdagen.se/dokumentlista/?rm=<rm>&bet=<beteckning>&utformat=json` | Fallback lookup of a document's `dok_id`. |

**The register decides which votes exist.** The bulk archives are regenerated
irregularly. On 2026-11-19 they disagreed with the API:

* missing from bulk (fetched from `voteringlista`): the 2022-10-17
  prime-minister vote and KU32 (2022/23), and the 2024-01-17 vote of no
  confidence (2023/24);
* in bulk but withdrawn from the API (excluded, listed in `excluded_votes`):
  four 2024/25 votes on SkU3 and FiU6 with 339–344 of 349 absent (aborted
  votes), and a 2026-06-15 SfU31 vote (2025/26) that was later re-run.

`voteringlista` ignores a `votering_id` query parameter and returns unrelated
rows, so rows for a single vote are fetched by `rm` + `bet` and grouped by
`votering_id`. Every `voteringlista` response is checked against its own
`@antal`; a short answer fails the build.

Raw files are cached in `data/raw/riksdag_votes/` (git-ignored).
`data/processed/riksdag_votes/manifest.json` (committed) records the source
URL, UTC retrieval time and SHA-256 of every raw file used. A cached file is
reused only when its bytes match the manifest. `retrieved_at_utc` is when
those exact bytes were first downloaded: re-downloading identical bytes (as
`--update` does daily) leaves the entry unchanged.

Requests are rate limited to 4 per second, with up to five attempts and
exponential backoff (2, 4, 8, 16 s, or the server's `Retry-After`) after a
429, a 5xx or a network error; a 404 is not retried.

## Vote titles

Every vote carries `title`, `punkt_title` and `dok_id`. Examples:

| Vote | `title` | `punkt_title` |
| --- | --- | --- |
| 2025/26:CU13 punkt 6 | Fastighetsrätt | Förköpslag |
| 2025/26:UbU7 punkt 15 | Övergripande skolfrågor | Skolor med konfessionell inriktning |
| 2025/26:TU12 punkt 3 | Luftfartsfrågor | Arlanda och Bromma flygplats |
| 2026/27:FöU2 punkt 2 | Personalförsörjning av det militära försvaret | Villkor för förlängd värnplikt för tidigare yrkes- och reservofficerare |
| 2026/27:1015-1 | Prövning av talmannens förslag till statsminister den 15 oktober | – |

**Source.** `utskottsforslag/<dok_id>.json`, one request per document
(~20 kB): `dokument.titel` is the betänkande title and
`dokutskottsforslag.utskottsforslag[]` gives each förslagspunkt's `punkt` and
`rubrik`. Betänkande title plus punkt heading is the clearest Swedish
description of what was voted on. Alternatives checked on 2026-11-19:
`votering/<id>/json` has the betänkande title but no punkt heading and embeds
all 349 rows (~290 kB per vote); `dokumentstatus/<dok_id>.json` has the same
title and headings but embeds the whole document HTML (~330 kB);
`dokumentlista` has titles and press headlines but no punkt headings.

Chamber votes with a numeric beteckning (prime-minister votes, votes of no
confidence) have a `votering` document on the same endpoint titled
`Omröstning: <subject>`; the prefix is dropped and `punkt_title` is null.

**dok_id.** Derived from the riksdag's numbering: `H` + session character
(2012/13 = `0`, 2022/23 = `A`, 2025/26 = `D`, 2026/27 = `E`) + series (`01`
betänkande, `19` votering) + beteckning, e.g. `HD01CU13`. The answer is used
only if its `rm` and `beteckning` match the vote; otherwise the id is looked up
with `dokumentlista`. The document is at
`https://data.riksdagen.se/dokument/<dok_id>`.

**Caching.** Each document is cached by id and recorded in the manifest like
every other raw file, so a run fetches only documents it has not seen. A
cached document that lacks a voted punkt (fetched before the riksdag published
it) is downloaded again, once per run. Offline, a document missing from the
cache has no title and the same note is written as online, so online and
offline builds of one cache agree. Outputs record the title files compactly in
`provenance.title_sources` (count, and a SHA-256 over the sorted
`<file> <sha256>` lines); the per-file hashes are in the manifest.

On 2026-11-19 all 2,618 votes of 2022/23–2026/27 had a title (905 documents)
and every vote on a betänkande had a punkt heading.

## Ingest checks (rule 5)

Each vote must have exactly 349 rows, benches (`banknummer`) exactly 1..349,
no duplicate (votering_id, intressent_id) pair, one value per vote-level field
(rm, beteckning, punkt, avser, votering, hangar_id), a known `rost`, and row
totals equal to the register's. Any failure raises `IngestError`. People are
tracked by `intressent_id`; a substitute sits in the bench of the member they
replace, so each vote's bench mapping shows who actually sat there.

## Rules

1. **Vote selection.** Every vote keeps `avser`, `votering`, `punkt`,
   `beteckning`, committee and date. The analysis set — used for agreement,
   map, against-line counts and participation — is `votering == "huvud"` and
   `avser == "sakfrågan"`. Motivfrågan and frågan votes (e.g. the PM vote)
   remain in `votes.json` and in the per-vote party lines.
2. **Avstår is a stance.** A party's line is the plurality among its members
   voting Ja, Nej or Avstår; Frånvarande is excluded. A tie gives `null` and
   the party is listed in `ties`; a party with nobody casting gives `null` and
   is listed in `no_cast`.
   * *Three-way agreement* (primary) between parties a and b =
     `#{votes: line_a = line_b} / n`, n = votes where both have a line.
     Avstår vs Ja is disagreement; Avstår vs Avstår is agreement.
   * *Ja/Nej agreement* (secondary) = the same over the votes where both lines
     are Ja or Nej. It drops every vote on which either party abstained.
   * Each cell stores `agree` and `n`; `share` is published only when
     `n >= 10` (`MIN_AGREEMENT_CELL_N`). Matrices are given overall and per
     committee.
3. **Map without centre-pull.** Ja = +1, Avstår = 0, Nej = −1,
   Frånvarande = missing, never imputed.
   * Map votes: analysis-set votes whose minority side is at least 2.5 % of
     Ja+Nej ballots (`min(Ja, Nej)/(Ja+Nej) >= 0.025`, the NOMINATE
     convention) and on which all parties have a line.
   * Geometry: X = parties × map votes of party-line codes (independents
     excluded); μ_i = mean of column i; X − μ = U S Vᵀ. Party scores
     (U S)[:, :2], loadings W = V[:, :2] (unit-norm columns). Explained
     variance of axis k = s_k² / Σ s².
   * Members: with O the map votes the member cast,
     `s = argmin Σ_{i∈O} (v_i − μ_i − W_i·s)² = (W_Oᵀ W_O)⁻¹ W_Oᵀ (v_O − μ_O)`.
     At least 50 cast map votes (`MIN_MP_MAP_VOTES`) are required, otherwise
     the member is unplaced with a reason. A member who always votes the party
     line lands on the party's point regardless of absences (tested).
   * Reported coordinates are divided by √(number of map votes) so that
     periods of different length are comparable.
   * Signs. A **term** map is oriented so M scores positive on axis 1 and V on
     axis 2. An anchor counts only if its |score| is at least 10 %
     (`MAP_ANCHOR_MIN_SHARE`) of the axis's largest |score|; otherwise the
     first party in the standard order that clears that bar is made positive.
     (Until 2026-11-19 axis 2 was anchored on SD, which sits at ~0 on axis 2,
     so its sign was arbitrary: V was −0.51 in 2022/23 and +0.40…+0.63
     elsewhere.) A **session** map is aligned with its term's map axis by
     axis: the sign of the dot product of the two maps' party coordinates over
     the parties in both; an axis whose |cosine| is below 0.1
     (`MAP_ALIGNMENT_MIN_COSINE`) falls back to the term rule. Each map
     records `orientation` (text), `orientation_reference` (`term-<label>`
     for a session map, null for a term map) and `orientation_axes` (per
     axis: `method` `aligned` with its `cosine`, or `anchor` with the party
     used). A session rebuilt without the rest of its term reads the term map
     from the term summary on disk. This is a reproducibility convention.
     **Axes are not left/right**;
     read them through `top_loading_votes` and `top_loading_patterns` (the
     party-line patterns with the largest |loading|, with vote counts and up
     to three `example_votes` with their titles — with eight parties many
     votes share one pattern and therefore one loading).
   * The map needs at least 20 map votes (`MIN_MAP_VOTES_FOR_FIT`);
     otherwise `available: false`.
4. **Independents and feltryckningar.** Members with parti `-` (politiska
   vildar) have no party line and are outside the party matrix, but keep
   per-member statistics and a map position. A vote against the party line is
   a cast Ja/Nej/Avstår different from the member's party's non-null line on
   that vote (the party the member had on that vote). Counts and vote ids are
   stored. Isolated cases can be feltryckningar (accidental presses); the
   record keeps them as cast.
5. **No truncation, correct people.** See *Ingest checks*.
6. **Attendance.** Participation = cast / seated over analysis-set votes,
   where "seated" counts the votes on which the member occupied a bench.
   No rankings: Frånvarande includes kvittning (pairing), which the data does
   not distinguish from other absence.

## Blocs and per-vote flags (`blocs.py`)

**Blocs.** A term's sides come from its statsministeromröstning: the chamber
vote (numeric beteckning, `avser: frågan`) titled `Prövning av (talmannens)
förslag till statsminister …`. The speaker's proposal is rejected only if at
least 175 of 349 members vote Nej (RF 6:4). The latest approved PM vote of the
term is used: `government_side` = parties whose line was Ja, `opposition` =
Nej, `abstained` = Avstår, `no_line` = tie or nobody cast; independents are not
placed. Every term summary has `blocs` (or null) and `pm_votes_considered`
(each PM vote with `nej` and `approved`); every session summary has its term's
`blocs`.

* term-2022-26: the 2022-10-17 vote (176 Ja, 173 Nej, approved):
  government_side M, L, KD, SD; opposition C, S, V, MP.
* term-2026-30: `null` on 2026-11-19 — both PM votes so far were rejected
  (2026-10-15: 248 Nej; 2026-11-03: 324 Nej).

`blocs` = `{source_vote_id, rm, date, title, counts [Ja, Nej, Avstår,
Frånvarande], government_side, opposition, abstained, no_line, rule}`.

**Per-vote flags** (`vote_flags` in every session and term summary), so the
website can filter without loading `votes.json`. Columnar, one entry per vote
in the order of `party_lines.votes` (all votes; `party_lines.votes[i].analysis`
marks the analysis set):

```
"vote_flags": {
  "order": "...", "fields": ["ja","nej","avstar","franvarande","margin","against_line","cross_bloc"],
  "columns": {"ja": [...], "nej": [...], ..., "cross_bloc": [1, 0, null, ...]},
  "definitions": {...}
}
```

* `ja`, `nej`, `avstar`, `franvarande`: ballot totals; `margin` = |ja − nej|;
* `against_line`: members casting Ja/Nej/Avstår different from their party's
  non-null line on that vote;
* `cross_bloc` (1/0, null when `blocs` is null): among parties with a line —
  not unanimous, and not bloc-aligned (all government_side parties on one line
  X and all opposition parties on one line Y ≠ X; abstained parties ignored),
  and at least one government_side party shares its line with at least one
  opposition party.

Because riksdag roll calls are almost never unanimous (unanimous decisions are
taken by acclamation) and the 2022–26 opposition rarely voted as one, most
votes are cross-bloc by this rule: 2,187 of 2,610 in term-2022-26 (152
bloc-aligned, 6 unanimous). In 534 of them the government side itself split.

## Committees

Committee = letter prefix of `beteckning` when it is one of AU, CU, FiU, FöU,
JuU, KU, KrU, MJU, NU, SfU, SkU, SoU, TU, UbU, UU, UFöU. Numeric designations
(`0311-1`, chamber votes on a question) have no committee and are grouped as
`övrigt`; an unknown prefix is logged, recorded in `notes` /
`unknown_committee_prefixes` and grouped as `övrigt`. No unknown prefix
occurred in 2022/23–2026/27.

## Outputs (`data/processed/riksdag_votes/`)

All files are UTF-8 JSON with sorted keys and compact separators;
`schema_version: "1.0"`; output is byte-identical for the same raw inputs
except the run stamp (`provenance.generated_at_utc`, `code_commit`,
`code_dirty`). A file whose content apart from `provenance` is unchanged is
not rewritten, so a run with no new votes changes no output. Its provenance
then stays that of the build that produced the content: the bulk archive URL
can serve differently zipped bytes with the same votes (on 2026-11-19 the
2026/27 archive alternated between two SHA-256s within an hour), and that
alone is not a change. The manifest does record the new
bytes, so offline rebuilds keep working; the scheduled workflow does not
commit a manifest-only change.

* `<YYYY-ZZ>/votes.json` (`role: riksdag_votes_session_votes`):
  `people[i]` (intressent_id, name, …); `seat_holders[j] = [person index,
  parti, valkrets]`; `rosters[k]` = 349 seat-holder indices, element b−1 =
  occupant of bench b; each vote has its metadata (including `title`,
  `punkt_title`, `dok_id`), `roster`, `ballots` (a
  349-character string, character b−1 = bench b: J/N/A/F) and `counts`
  [Ja, Nej, Avstår, Frånvarande]; `excluded_votes`; `provenance`.
* `<YYYY-ZZ>/summary.json` and `term-<YYYY-ZZ>/summary.json`
  (`role: riksdag_votes_summary`): `party_lines` for every vote, `agreement`
  (three-way and Ja/Nej, overall and per committee, with n), `members`
  (party, parties, valkrets, seated, cast, participation, against-line count
  and vote ids, map position or unplaced reason), `party_against_line`,
  `participation_distribution`, `map`, `blocs`, `vote_flags` (and
  `pm_votes_considered` in a term summary), `caveats`, `rules`, `notes`,
  `provenance` (source files with SHA-256, `title_sources`, `code_commit`,
  `code_dirty`, `generated_at_utc`). A term summary is written only when the
  run includes every session of the term that has begun, so a partial run
  cannot overwrite it with a partial one; `scope.term_complete` says whether
  all four sessions are in.
* `manifest.json` (`role: riksdag_votes_source_manifest`).

Sizes on 2026-11-19 in bytes, before → after titles: 2025-26/votes.json
596,376 → 698,528; 2025-26/summary.json 284,695 → 298,961;
term-2022-26/summary.json 599,171 → 613,146; manifest.json 6,524 → 246,139
(one entry per title document). With blocs and vote_flags (2026-11-19):
2025-26/summary.json 298,961 → 317,611; term-2022-26/summary.json 613,146 →
668,985 (+55,839).

## Website layout (`files/riksdag/` on edvinli.github.io)

`--site-dir` (or `python -m scripts.riksdag_votes.site`) writes:

```
index.json                    role riksdag_votes_index
<YYYY-ZZ>/votes.json          byte copy of data/processed/riksdag_votes/<YYYY-ZZ>/votes.json
<YYYY-ZZ>/summary.json        byte copy of .../<YYYY-ZZ>/summary.json
term-<YYYY-ZZ>/summary.json   byte copy of .../term-<YYYY-ZZ>/summary.json
```

`index.json` has `schema_version`, `role`, `generated_at_utc` (the newest
`provenance.generated_at_utc` of the listed files, i.e. when the content last
changed — not the export time, so re-exporting is byte-stable),
`latest_vote_date`, `source`, `sessions[]` (`rm`, `slug`, `n_votes`,
`n_votes_analysis`, `n_votes_titled`, `first_vote_date`, `latest_vote_date`,
`generated_at_utc`, and `files.votes` / `files.summary` each with `path`
relative to the index, `bytes` and `sha256`) and `terms[]` (`label`,
`sessions`, `term_complete`, counts, `latest_vote_date`, `files.summary`). The
manifest is not published. The export writes only inside the site directory
and deletes nothing.

## Scheduled update (`.github/workflows/riksdag-votes.yml`)

Daily at 06:41 UTC, and on `workflow_dispatch` (`dry_run`, default true;
optional `sessions`). Concurrency group `riksdag-votes` only. Steps: restore
the raw cache (`actions/cache`, keyed on the session list and date, falling
back to the newest), run the riksdag tests, check out the data branch, build
the `--current` sessions with `--update --output-dir <data branch> --site-dir`,
report both diffs, and — unless dry run — push the data branch, then push the
website copy. All git steps are `scripts/riksdag_votes/publish.py`: one
explicit path per target, no force, one retry after `git pull --rebase`, and a
check that the remote branch is at the pushed commit. A failure opens or
comments on the issue "Riksdag votes update failing".

**The workflow never commits or pushes to simulator `main`.** `publish.py`
accepts exactly two (path, branch) targets and refuses `main`:

| Path | Repository | Branch |
| --- | --- | --- |
| `data/processed/riksdag_votes` (with `manifest.json`) | edvinli/election-simulator | `data/riksdag-votes` |
| `files/riksdag` | edvinli/edvinli.github.io | `master` |

`data/riksdag-votes` is append-only: on the first run it is started at the
main commit the workflow checked out (so it shares main's history and starts
with main's outputs); every later run adds data commits on top. Main holds the
riksdag outputs only as updated through normal pull requests.

**Where the previous outputs are read from.** The build needs them to leave
unchanged files untouched (and so commit nothing on a day without new votes)
and to copy the completed sessions to the website. They are read from
`data/riksdag-votes` if it exists, otherwise from main (which is the same
thing on the first run, since the branch starts there). The branch is checked
out as a git worktree beside the simulator checkout (`publish.py prepare`), the
build writes into it with `--output-dir`, and the main checkout is asserted
clean afterwards. A rebuild that reaches main later through a PR does not flow
into the branch by itself: dispatch the workflow with `sessions` set to those
sessions to rebuild them on the branch (the same code gives the same output).

Completed sessions are otherwise not rebuilt; their outputs on the branch are
copied to the site unchanged. After a pipeline change, rebuild a completed
term by dispatching with `sessions` = `2022/23 2023/24 2024/25 2025/26` (and,
to update main too, run `build --sessions …` locally and open a PR).

**Waiting applies to the website push only.** The publication and rendering
workflows push to website master with a plain `git push` that is not retried.
Before the website push the workflow waits until no publication, rendering or
benchmark run is queued or running (two checks a minute apart, never inside
18:30–22:00 UTC) and defers after 60 minutes; the data branch is already
pushed then, and the next run sees the website behind and copies it. No other
workflow writes `data/riksdag-votes`, so that push does not wait.

Interactions with the other workflows:

* The data-branch push uses `GITHUB_TOKEN`, which starts no other workflow,
  and no workflow triggers on that branch (Full CI, `full.yml`, triggers on
  push to main only).
* Because main never moves, `scripts/ci/verified_code.py` and the publication
  workflow's skip of its contract suite are unaffected.
* The website push uses `WEBSITE_REPO_TOKEN` and starts the website's own
  push workflows (Full CI on master, Pages), as the render workflow's does.

## Caveats

* Participation is not diligence: kvittning, ministerial duty (ministers are
  replaced by substitutes) and party leaders' pairing all appear as absence or
  as a substitute in the bench.
* Against-line counts include feltryckningar.
* A party line is a plurality; a party split 10–9 has the same line as a
  united one. Use the per-vote `ballots` for the split.
* The map summarises co-voting among eight parties. With so few rows, many
  votes share one party-line pattern, and the geometry of the
  government/opposition period is dominated by bloc voting; it is not an
  ideological scale.
