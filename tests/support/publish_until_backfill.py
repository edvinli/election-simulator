"""Run a real publication and die at history-backfill entry.

The 2026-09-09 incident was not an exception inside the backfill -- those are
caught and tolerated -- it was the *process* being killed there by the job
timeout, with nothing pushed. No in-process handler runs in that case, so the
only way to test the surviving state honestly is to actually kill a process at
that point.

This module is the subprocess. It publishes for real against the repositories
it is given, signals the parent the moment the backfill is entered, and then
terminates itself with SIGKILL so that no cleanup, no `finally`, and no
exception handling can run.
"""

from __future__ import annotations

import json
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts import election_automation_base as base
from scripts.election_automation import run_automation



def main() -> int:
    payload = json.loads(sys.argv[1])
    source = Path(payload["source"])
    site = Path(payload["site"])
    as_of = payload["as_of"]
    ready = Path(payload["signal"])

    # The fixture's own result builder, so the pipeline sees exactly the shape
    # it sees in the in-process tests rather than a hand-rolled stand-in.
    from tests.test_election_automation import ElectionAutomationTests

    produced = ElectionAutomationTests._production_result(as_of)

    def refresh(raw, processed, **kwargs):
        # The fixture's own mutator: it changes one real normalized poll value
        # without breaking the schema, which appending a line does not.
        raw.mkdir(parents=True, exist_ok=True)
        ElectionAutomationTests._change_normalized_poll_support(
            processed / "individual_polls.csv")
        return {"messages": []}

    def runner(**kwargs):
        import subprocess as sp
        commit = sp.run(["git", "rev-parse", "HEAD"], cwd=source, check=True,
                        capture_output=True, text=True).stdout.strip()
        produced.manifest["source_git_commit"] = commit
        produced.manifest["git_commit"] = commit
        return produced

    def die_at_backfill(*args, **kwargs):
        # Certification has already happened and been pushed by this point.
        # Tell the parent, flush, and go straight down: SIGKILL cannot be
        # caught, so nothing below the boundary gets to tidy up.
        ready.write_text("backfill-entered\n", encoding="utf-8")
        os.kill(os.getpid(), signal.SIGKILL)

    base.backfill_reconstructed_curve = die_at_backfill
    # The frozen simulator refuses a processed root other than the one it
    # validated, so the fixture's root is pinned here exactly as the in-process
    # tests pin it.
    from scripts.publication_pipeline import pipeline as _pipeline
    _pipeline.DEFAULT_PROCESSED_ROOT = source / "data/processed"
    result = run_automation(
        source,
        site_repo=site,
        schedule=base.INTRADAY_SCHEDULE_UTC,
        now=datetime.fromisoformat(f"{as_of}T08:00:00+00:00").astimezone(timezone.utc),
        automation_enabled="true",
        mode="publish",
        commit=True,
        push=True,
        refresh_fn=refresh,
        simulation_runner=runner,
        projection_runner=ElectionAutomationTests._projection_runner,
        campaign_path_simulator=ElectionAutomationTests._campaign_path_simulator,
        website_check_fn=lambda _root: {"status": "PASS"},
        website_push_check_fn=lambda _root: {"status": "PASS"},
        generated_at_utc=f"{as_of}T08:00:00+00:00",
    )
    # Only reached if the kill never happened, which is itself the diagnosis.
    # Only reached if the kill never happened, which is itself the diagnosis.
    sys.stderr.write(
        "publication returned without entering the backfill: "
        + str(result.status)
        + chr(10)
        + result.summary.render()
        + chr(10)
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
