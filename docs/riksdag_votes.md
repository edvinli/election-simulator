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

* `--current-term` instead of `--sessions`: every begun session of the
  electoral term containing today's UTC date (2026/27 on 2026-11-19);
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
   * Signs: axis 1 is oriented so M scores positive, axis 2 so SD scores
     positive (fallback: the first party in the standard order with a non-zero
     score). This is a reproducibility convention. **Axes are not left/right**;
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
  `participation_distribution`, `map`, `caveats`, `rules`, `notes`,
  `provenance` (source files with SHA-256, `title_sources`, `code_commit`,
  `code_dirty`, `generated_at_utc`). A term summary is written only when the
  run includes every session of the term that has begun, so a partial run
  cannot overwrite it with a partial one; `scope.term_complete` says whether
  all four sessions are in.
* `manifest.json` (`role: riksdag_votes_source_manifest`).

Sizes on 2026-11-19 in bytes, before → after titles: 2025-26/votes.json
596,376 → 698,528; 2025-26/summary.json 284,695 → 298,961;
term-2022-26/summary.json 599,171 → 613,146; manifest.json 6,524 → 246,139
(one entry per title document).

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
back to the newest), run the riksdag tests, build the current term's sessions
with `--update --site-dir`, report both diffs, and — unless dry run — wait
until no publication, rendering or benchmark run is queued or running (two
checks a minute apart, never inside 18:30–22:00 UTC, deferring to the next day
after 60 minutes), then commit `data/processed/riksdag_votes` to simulator main
and `files/riksdag` to website master with `scripts/riksdag_votes/publish.py`:
explicit path only, no force, one retry after `git pull --rebase`, and a check
that the remote branch is at the pushed commit. A failure opens or comments on
the issue "Riksdag votes update failing".

Completed sessions are not rebuilt daily; their committed outputs are copied to
the site unchanged. After a pipeline change, rebuild a completed term with
`build --sessions 2022/23 2023/24 2024/25 2025/26` locally and commit, or
dispatch the workflow with that `sessions` input.

Interactions with the other workflows:

* The simulator push uses `GITHUB_TOKEN`, which starts no other workflow, so
  Full CI (`full.yml`, on push to main) does not run on it.
* The website push uses `WEBSITE_REPO_TOKEN` and starts the website's own
  push workflows (Full CI on master, Pages), as the render workflow's does.
* The waiting step exists because the publication, rendering and benchmark
  workflows push with a plain `git push` that is not retried.
* `scripts/ci/verified_code.py` lets the publication workflow skip its contract
  suite only when every commit since the last green Full CI is a bot commit
  inside `BOT_OWNED_PATHS`, which does not include
  `data/processed/riksdag_votes/`. Until that path is added there, each
  riksdag commit makes later publication runs run the 10–18 minute contract
  suite until the next human push passes Full CI.

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
