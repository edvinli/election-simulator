"""Shared policy for judging drift reported by the scientific freeze verifiers.

A freeze certifies an evaluator -- its rules, cases, seeds and truth -- at a
referenced commit. Its verifier reports every file whose hash no longer matches,
and each report carries the ``group`` the file belongs to. Those groups mean
different things, and collapsing them into one set of filenames is what made
four separate test modules fail on main after every publication:

  * ``evaluator_import_closure`` and its siblings are code. Drift there must be
    an enumerated, deliberate change, or the freeze no longer certifies what it
    claims to. This stays strict.

  * ``truth_input`` is the polling snapshot, which the publication automation
    re-fetches and rewrites on every publish. Its ``poll_id`` column is a
    content hash that includes each row's upstream ``source_row`` index, so an
    upstream insertion renumbers every row and regenerates every id: the whole
    file churns even when no poll's numbers changed. A refreshed truth input is
    therefore expected, and is not evidence that the frozen evaluator moved.

The alternative -- adding the CSV to each module's KNOWN_POST_FREEZE_CHANGES --
would have worked once and then hidden any future *code* drift in that same
file, because those sets are matched on filename alone with no notion of which
group the drift came from.

The ``poll_id`` instability is a real defect in the polling pipeline and is
tracked separately; it is deliberately not worked around here, because changing
how ``poll_id`` is derived changes published provenance.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

#: Freeze groups that hold inputs the publication automation refreshes by
#: design. Drift confined to these is expected.
REFRESHED_INPUT_GROUPS = frozenset({"truth_input"})


def unexpected_drift(
    result: Mapping[str, Any],
    known_changes: Iterable[str] = (),
) -> set[str]:
    """Return drift that no deliberate change or input refresh explains.

    Entries are returned as ``"<group>:<file>"`` so a failure message names the
    group it came from; a filename alone was ambiguous precisely because the
    same path can appear under more than one group.

    An entry whose group is absent or unrecognised is reported rather than
    assumed safe, so a new freeze group cannot quietly widen this exemption.
    """
    known = set(known_changes)
    unexpected: set[str] = set()
    for entry in result.get("drift", ()):
        group = entry.get("group")
        if group in REFRESHED_INPUT_GROUPS:
            continue
        if entry["file"] in known:
            continue
        unexpected.add(f"{group}:{entry['file']}")
    return unexpected
