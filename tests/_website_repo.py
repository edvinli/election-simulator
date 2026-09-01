"""Locate the deployed website consumer for cross-repository contract tests.

The website repository is a separate checkout that this repository must not
modify. Tests that need the deployed consumer resolve it through here so the
location and the override are defined exactly once.

Not named ``test*.py``, so unittest discovery does not collect it.
"""

from __future__ import annotations

import os
from pathlib import Path


ENV_OVERRIDE = "ELECTION_SIMULATOR_WEBSITE_REPO"
# Never infer a sibling checkout from the developer's home directory.  That
# made portable CI accidentally exercise whichever website happened to be
# present on the runner.  Cross-repository tests must opt in explicitly.
DEFAULT_WEBSITE_REPO = Path("__website_checkout_not_opted_in__")


def website_repo() -> Path:
    override = os.environ.get(ENV_OVERRIDE)
    return Path(override) if override else DEFAULT_WEBSITE_REPO


def website_consumer_path() -> Path:
    """Path to the production consumer, edvinli.github.io/assets/js/…"""

    return website_repo() / "assets" / "js" / "election-simulator.js"


SKIP_REASON = (
    "The website checkout is unavailable, so the actual production consumer cannot be "
    f"exercised. Set {ENV_OVERRIDE} to the edvinli.github.io checkout. "
    "The portable REFERENCE contract test still runs."
)
