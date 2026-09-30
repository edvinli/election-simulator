"""Where the production model's opinion inputs live, resolved per data revision.

From model version 1.2.0 the opinion inputs are built only from SwedishPolls:
the daily ``SwedishPollsAggregate-v0.2`` series and the cleaned SwedishPolls
polls it was built from, both under ``poll_aggregate/v0_2/``. Earlier
revisions used the pollofpolls.se series and PoP-reconstructed polls under
``pollofpolls/``. A generation certified before the migration must still
render from its own pinned inputs, so the resolver falls back to those files
when a pinned revision has no aggregate. The publication path requires the
aggregate explicitly (``PRODUCTION_OPINION_INPUTS``), so a current run can
never fall back silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Relative to the processed-data root.
AGGREGATE_DIR = Path("poll_aggregate") / "v0_2"
AGGREGATE_TIMESERIES = AGGREGATE_DIR / "swedishpolls_aggregate_timeseries.csv"
AGGREGATE_POLLS = AGGREGATE_DIR / "individual_polls.csv"
AGGREGATE_METADATA = AGGREGATE_DIR / "swedishpolls_aggregate_metadata.json"
AGGREGATE_SOURCE = "SwedishPollsAggregate-v0.2"

LEGACY_TIMESERIES = Path("pollofpolls") / "pollofpolls_timeseries.csv"
LEGACY_POLLS = Path("pollofpolls") / "individual_polls.csv"
LEGACY_SOURCE = "pollofpolls.se"

#: The SwedishPolls table itself (election-noise pool and the chart's poll feed).
SWEDISHPOLLS_POLLS = Path("pollofpolls") / "swedishpolls_individual_polls.csv"

#: The inputs a current production run must have; never the legacy files.
PRODUCTION_OPINION_INPUTS: tuple[Path, ...] = (AGGREGATE_TIMESERIES, AGGREGATE_POLLS)


class MissingOpinionInputs(FileNotFoundError):
    """Neither the aggregate nor the legacy opinion inputs are present."""


@dataclass(frozen=True)
class OpinionInputs:
    timeseries: Path
    polls: Path
    source: str

    @property
    def is_legacy(self) -> bool:
        return self.source == LEGACY_SOURCE


def opinion_inputs(processed_root: Path | str, *, allow_legacy: bool = True) -> OpinionInputs:
    """The opinion timeseries and polls for one processed-data revision.

    The aggregate wins whenever both of its files exist. The legacy PoP files
    are used only for a revision that predates the aggregate, and only when
    ``allow_legacy`` is true.
    """

    root = Path(processed_root)
    timeseries, polls = root / AGGREGATE_TIMESERIES, root / AGGREGATE_POLLS
    if timeseries.is_file() and polls.is_file():
        return OpinionInputs(timeseries, polls, AGGREGATE_SOURCE)
    if allow_legacy:
        timeseries, polls = root / LEGACY_TIMESERIES, root / LEGACY_POLLS
        if timeseries.is_file() and polls.is_file():
            return OpinionInputs(timeseries, polls, LEGACY_SOURCE)
    raise MissingOpinionInputs(f"No opinion inputs under {root} (legacy allowed: {allow_legacy})")
