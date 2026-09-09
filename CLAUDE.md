# Working in this repository

## Establish the current state
- Production is `edvinli/election-simulator`, branch `main`; the website is the separate `edvinli/edvinli.github.io` repository, branch `master`.
- Fetch before diagnosing. Record the exact commit, workflow run and generation inspected; local worktree names and old logs are not production evidence.
- Inspect `git status` and worktrees before editing. Preserve other work; use an isolated checkout when necessary. Never revert a whole file to remove a temporary probe if it contains other edits.
- Follow existing user authorization without repeatedly asking. Implementation authorization does not itself authorize merging, production dispatch or changes to repository protection.

## Forecast integrity
- Read the relevant `docs/2026_prospective_benchmark_*` documents and machine protocol/amendments before changing benchmark behavior. `scripts/simulator/config.py` identifies the current model; historical documents describe their recorded versions.
- Preserve historical capture bytes, hashes, amendment lists and eligibility. Never backdate evidence, retrofit sidecars, or change tests merely to admit excluded observations.
- Protocol changes require explicit authorization and an append-only amendment: Markdown, JSON, SHA-256 companion and index reference. Use the actual finalization timestamp, distinguish publication/adoption, and verify chronological claims.
- Authorized scope for the certification split: extend same-run exact-draw retention prospectively beyond September 12, including election day, with the required amendment. Benchmark dates, selection, weighting and scoring stay unchanged. Check whether this amendment already exists before creating another.

## Certification and rendering
- Target boundary: acquire/freeze inputs → one 100,000-draw authoritative simulation → validate/export → commit/push/verify certification → render.
- Certification retains the immutable generation, exact joint draws, snapshot and provenance together. Stage explicit paths; rendered history is a separate output and commit.
- Rendering consumes an explicit generation and pinned inputs. Never rerun the authoritative forecast or infer joint distributions from marginal quantiles. Historical reconstruction and projection simulations remain allowed.
- Distinguish certification, rendering and deployment failures. `CERTIFIED_NOT_RENDERED` needs rendering; `RENDERED_NOT_DEPLOYED` needs validated mirroring. Reject stale deployments.
- Keep rendering outside the certification/benchmark lock. Verify actual workflow triggers and credentials; do not assume a bot push triggers another workflow.
- For the 2026 campaign, preserve the protected 18:30–22:00 UTC interval and check capture state before an authorized dispatch. Do not increase the 120-minute publication timeout without reconciling its benchmark guard.

## Verification and delivery
- Use `uv sync --frozen` and targeted `uv run python -m unittest ...` tests. CI selects affected modules via `scripts.ci.test_topology`; green PR CI is not a full-suite result. Explicitly run relevant freeze/archive tests and explain skips.
- Full suite: `uv run python -m unittest discover -s tests -t . -v`. Preserve its exit status through logging wrappers. Test a quiescent, identified revision; overlapping edits invalidate attribution.
- Prove durability/recovery with real temporary bare remotes, `push=True`, interruption before history commits, and fresh clones. Disable authoritative simulation during rendering retries.
- Run website suites against the exact staged artifact and enforce parity with its `browser-tests/select-suites.mjs`. Tests share port 4000: use separate runners or run serially.
- Before merging, verify `baseRefName`, head SHA, checks and final diff. Rebase and retarget stacked PRs; preserve upstream fixes during conflict resolution. Never force-push shared production branches.
- Report what changed, tested revision/results, remote state and remaining limitations. Keep website branch protection and generation-owned polling-date provenance deferred unless the user changes that decision.
