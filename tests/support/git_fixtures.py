"""Git fixture repositories that do not race their own cleanup.

Git runs automatic maintenance after commands that create objects, and it
detaches that work into a background process by default. A fixture repository
built inside a ``TemporaryDirectory`` therefore keeps being written to after
the test that built it has finished, and the cleanup races the maintenance
run:

    OSError: [Errno 39] Directory not empty: '/tmp/tmp.../simulator/.git/objects'

That is what blocked a publication on 2026-09-11: a green test suite reported
an error with nothing wrong in the test, in the temporary-directory teardown
of a test that had already made all of its assertions.

Turning maintenance off at init is the fix. Pinning ``gc.autoDetach`` off as
well is the belt to that braces: if anything ever does trigger a run, it
finishes inside the synchronous ``git`` call that started it instead of
outliving the fixture. Both are repository-local, so they apply to every later
``git`` invocation in that repository -- including the ones production code
makes while under test -- and they leak into no other repository.
"""

from __future__ import annotations

from pathlib import Path
import subprocess


#: Repository-local settings that keep a fixture repository quiescent once the
#: last synchronous git call against it has returned.
NO_BACKGROUND_MAINTENANCE = (
    ("gc.auto", "0"),
    ("maintenance.auto", "false"),
    ("gc.autoDetach", "false"),
)


def disable_background_maintenance(repository: Path | str) -> None:
    """Stop ``repository`` from writing to itself in the background.

    Safe on a worktree or a bare repository, and must be called immediately
    after ``git init`` -- before the first commit, which is what would
    otherwise trigger the first maintenance run.
    """

    for key, value in NO_BACKGROUND_MAINTENANCE:
        subprocess.run(
            ["git", "config", key, value],
            cwd=str(repository),
            check=True,
        )
