"""Fail-safe orchestration for the ElectionSimulator publication contract.

The package intentionally keeps acquisition out of the default path.  A
publication run validates the checked-in source snapshot, executes the frozen
Candidate A simulator, appends one immutable prospective snapshot, and then
atomically publishes the compact static JSON contract.
"""

from .pipeline import (
    DEFAULT_ARCHIVE_DIR,
    DEFAULT_PROCESSED_ROOT,
    DEFAULT_PUBLICATION_DIR,
    PipelineRun,
    run_publication_pipeline,
    validate_existing_inputs,
    validate_simulation_result,
)

__all__ = [
    "DEFAULT_ARCHIVE_DIR",
    "DEFAULT_PROCESSED_ROOT",
    "DEFAULT_PUBLICATION_DIR",
    "PipelineRun",
    "run_publication_pipeline",
    "validate_existing_inputs",
    "validate_simulation_result",
]
