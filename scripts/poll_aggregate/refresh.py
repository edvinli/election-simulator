"""The production polling refresh: SwedishPolls only, then the v0.2 aggregate.

From model version 1.2.0 production no longer fetches pollofpolls.se. This
refresh acquires the two SwedishPolls files, with the existing retention of a
verified raw file when a fetch fails, normalizes them to
``pollofpolls/swedishpolls_individual_polls.csv``, and rebuilds
``SwedishPollsAggregate-v0.2`` with its cleaned polls into
``poll_aggregate/v0_2/``. Segment fits already recorded there are reused, so
only a newly available election result triggers a refit.

It keeps the orchestrator's refresh signature. The aggregate directory is the
``poll_aggregate/v0_2`` sibling of ``processed_dir``, so a staged tree that
mirrors ``data/processed`` stages it too.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from time import monotonic
from typing import Any, Callable

from scripts.pollofpolls.__main__ import PollingValidationError, _write_csv
from scripts.pollofpolls.acquire import acquire_all
from scripts.pollofpolls.config import SOURCE_BY_KEY
from scripts.pollofpolls.normalize import parse_swedishpolls_payloads
from scripts.pollofpolls.validate import SWEDISHPOLLS_FIELDS, validate_swedishpolls
from scripts.simulator.model_inputs import AGGREGATE_DIR, SWEDISHPOLLS_POLLS

from .config import DEFAULT_2026_RESULT_MANIFEST, DEFAULT_ELECTION_RESULTS_FILE, SPEC_V02
from .outputs import build_outputs

SWEDISHPOLLS_SOURCES = (SOURCE_BY_KEY["swedishpolls"], SOURCE_BY_KEY["swedishpolls_sources"])
MANIFEST_FILENAME = "swedishpolls_retrieval_manifest.json"
StageCallback = Callable[[str, str, float | None], None]


@contextmanager
def _stage(name: str, callback: StageCallback | None):
    started = monotonic()
    if callback:
        callback(name, "start", None)
    yield
    if callback:
        callback(name, "end", monotonic() - started)


def aggregate_dir_for(processed_dir: Path) -> Path:
    return Path(processed_dir).parent / AGGREGATE_DIR


def refresh_snapshot(
    raw_dir: Path,
    processed_dir: Path,
    *,
    readme_path: Path | None = None,
    offline: bool = False,
    allow_archive_fallback: bool = True,
    timeout: float = 45.0,
    stage_callback: StageCallback | None = None,
    election_results: Path = DEFAULT_ELECTION_RESULTS_FILE,
    certified_results: tuple[Path, ...] = (DEFAULT_2026_RESULT_MANIFEST,),
) -> dict[str, Any]:
    """Acquire SwedishPolls, validate, then write the processed table and aggregate.

    ``readme_path`` and ``allow_archive_fallback`` are accepted for the
    orchestrator's signature; SwedishPolls has no archive fallback and the
    README is not regenerated.
    """

    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    with _stage("acquisition", stage_callback):
        manifest, messages = acquire_all(
            raw_dir, offline=offline, allow_archive_fallback=False, timeout=timeout,
            sources=SWEDISHPOLLS_SOURCES, manifest_filename=MANIFEST_FILENAME,
        )
    with _stage("normalization/validation", stage_callback):
        _, rows = parse_swedishpolls_payloads(
            (raw_dir / SOURCE_BY_KEY["swedishpolls"].raw_filename).read_bytes(),
            (raw_dir / SOURCE_BY_KEY["swedishpolls_sources"].raw_filename).read_bytes(),
        )
        errors = [i for i in validate_swedishpolls(rows) if i.get("severity") == "error"]
        if errors:
            raise PollingValidationError({"valid": False, "error_count": len(errors), "issues": errors})
        polls_file = processed_dir / SWEDISHPOLLS_POLLS.name
        _write_csv(polls_file, SWEDISHPOLLS_FIELDS, rows)
    with _stage("aggregate", stage_callback):
        metadata = build_outputs(
            polls_file, election_results, certified_results, aggregate_dir_for(processed_dir), SPEC_V02,
            reuse_stored_fits=True,
        )
    messages.append(
        f"aggregate: {SPEC_V02.version} {metadata['first_date']}..{metadata['last_date']}, "
        f"{metadata['polls_used']} polls"
    )
    return {
        "manifest": manifest,
        "acquisition_diagnostics": manifest.get("acquisition_diagnostics", []),
        "messages": messages,
        "swedishpolls": rows,
        "aggregate_metadata": metadata,
    }
