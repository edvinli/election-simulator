"""Compatibility facade for scheduled ElectionSimulator publication orchestration.

The automation implementation lives in ``scripts.election_automation_base``.
This module used to change one production boundary: it patched
``run_production_event`` to inject the additive future-projection history
updater, which meant publication and ``--mode render`` ran two different
history pipelines and only publication got the projections. The pipeline now
lives in ``election_automation_base.render_history_for_generation`` and both
entry points call it, so there is no boundary left to patch and this module is
the module alias plus the ``python -m`` entry point.
"""

from __future__ import annotations

import sys

from scripts import election_automation_base as _base


if __name__ == "__main__":
    raise SystemExit(_base.main())

# Normal imports receive the implementation module itself. This keeps
# unittest.mock patch targets and all existing private/public names behaving as
# before.
sys.modules[__name__] = _base
