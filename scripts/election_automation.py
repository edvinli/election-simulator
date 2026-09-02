"""Compatibility facade for scheduled ElectionSimulator publication orchestration.

The historical automation implementation is preserved byte-for-byte in
``scripts.election_automation_base``. This facade changes exactly one production
boundary: real simulator runs use the additive future-projection history updater.
Injected simulation runners retain the historical updater so the existing
orchestration test harness remains isolated from extra Monte Carlo work.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from scripts import election_automation_base as _base
from scripts.forecast_history.future_projection import (
    update_history_with_production_result as _future_history_update,
)


_original_history_update = _base.update_history_with_production_result
_original_run_production_event = _base.run_production_event


def _history_update_with_projection(
    existing_payload: Any,
    production_result: Any,
    **kwargs: Any,
):
    """Attach the conditional fan using the exact processed input root in use."""

    poll_file = kwargs.get("poll_file")
    projection_data_dir = None
    if poll_file is not None:
        # <processed>/pollofpolls/swedishpolls_individual_polls.csv
        projection_data_dir = Path(poll_file).resolve().parents[1]
    return _future_history_update(
        existing_payload,
        production_result,
        projection_data_dir=projection_data_dir,
        **kwargs,
    )


def _run_production_event_with_projection(*args: Any, **kwargs: Any):
    """Use the fan for real runs while preserving injected-runner test semantics."""

    selected = (
        _history_update_with_projection
        if kwargs.get("simulation_runner") is None
        else _original_history_update
    )
    previous = _base.update_history_with_production_result
    _base.update_history_with_production_result = selected
    try:
        return _original_run_production_event(*args, **kwargs)
    finally:
        _base.update_history_with_production_result = previous


# Functions defined in election_automation_base resolve globals from that module,
# so patch the callable there rather than wrapping only the public import surface.
_base.run_production_event = _run_production_event_with_projection


if __name__ == "__main__":
    raise SystemExit(_base.main())

# Normal imports receive the preserved implementation module itself. This keeps
# unittest.mock patch targets and all existing private/public names behaving as
# before while retaining the two patched production-boundary globals above.
sys.modules[__name__] = _base
