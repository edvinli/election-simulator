"""Behavioural tests for the reconstruction canary's verdict.

The canary is an acceptance script, so the thing that matters is whether it
can misreport. Its earlier tests searched its source for assertion messages,
which cannot catch a wrong comparison: three defects survived that style --
protected-path baselines that collided between the two repositories, safety
checks skipped whenever the renderer's output could not be parsed, and a
"both views rebuilt" verdict that never checked the views.

`evaluate` is therefore pure, and driven here with crafted state.
"""

from __future__ import annotations

import unittest

from tests.support.reconstruction_canary import (
    FAILED,
    FULL_ACCEPTANCE,
    RECONSTRUCTION_ONLY,
    RepositoryState,
    evaluate,
)

TARGET = "2026-09-07"
GENERATION = "20260910T110717Z-5054d5b3"
POINTER = b'{"publication_generation": "20260910T110717Z-5054d5b3"}'


def state(head="abc", status="", **protected) -> RepositoryState:
    return RepositoryState(head=head, status=status, protected=dict(protected))


# The same path holds different content in each repository, which is exactly
# what the collided baseline got wrong.
CLEAN = {
    "simulator": state(head="sim1", versions="tree-sim", archive="tree-arch"),
    "website": state(head="web1", versions="tree-web"),
}


def rendered(*, curve=None, views=None, **overrides) -> dict:
    """A clean render result, with named parts overridable in place."""

    payload = {
        "generation": GENERATION,
        "status": "RENDER_STAGED_NOT_INSTALLED",
        "curve": {
            "status": "COMPLETE",
            "reconstructed": [TARGET],
            "missing": [],
            "error": None,
            "views": {
                "status": "COMPLETE",
                "secondary_projection": "REBUILT",
                "primary_campaign_paths": "REBUILT",
                "missing": [],
                "error": None,
            },
        },
    }
    payload.update(overrides)
    if curve:
        payload["curve"].update(curve)
    if views:
        payload["curve"]["views"].update(views)
    return payload


def verdict(*, curve=None, views=None, **overrides):
    """`evaluate` on a clean run, with any one part perturbed.

    `curve` and `views` are folded into the render result rather than passed
    to `evaluate`, which takes neither.
    """

    kwargs = dict(
        target_date=TARGET,
        expected_generation=GENERATION,
        rendered=rendered(curve=curve, views=views),
        parse_error=None,
        exit_code=0,
        before=CLEAN,
        after=CLEAN,
        pointer_before=POINTER,
        pointer_after=POINTER,
    )
    kwargs.update(overrides)
    return evaluate(**kwargs)


class HappyPathTests(unittest.TestCase):
    def test_a_clean_run_is_full_acceptance(self) -> None:
        failures, outcome = verdict()
        self.assertEqual(failures, [])
        self.assertEqual(outcome, FULL_ACCEPTANCE)


class ProtectedPathBaselineTests(unittest.TestCase):
    """Defect 1: baselines collided across repositories.

    `protected_before` was keyed by path alone, so the website's entries
    overwrote the simulator's and both repositories were then compared against
    the website's tree hashes. Those differ by construction, so the check
    would have reported a change that had not happened -- a false failure on
    the runner, and one no source-text test could see.
    """

    def test_identical_paths_with_different_content_do_not_collide(self) -> None:
        failures, outcome = verdict()
        self.assertEqual(failures, [], "the two repositories' trees were conflated")
        self.assertEqual(outcome, FULL_ACCEPTANCE)

    def test_a_change_in_one_repository_is_attributed_to_it(self) -> None:
        after = dict(CLEAN)
        after["website"] = state(head="web1", versions="tree-web-CHANGED")
        failures, outcome = verdict(after=after)
        self.assertEqual(outcome, FAILED)
        self.assertTrue(
            any("website protected path changed: versions" in f for f in failures),
            failures)
        self.assertFalse(
            any("simulator protected path changed" in f for f in failures),
            "the untouched repository was blamed too")

    def test_each_repository_is_compared_against_its_own_baseline(self) -> None:
        # Swapping the two repositories' tree hashes must be detected in both,
        # which a single shared baseline could not do.
        after = {
            "simulator": state(head="sim1", versions="tree-web", archive="tree-arch"),
            "website": state(head="web1", versions="tree-sim"),
        }
        failures, _ = verdict(after=after)
        self.assertTrue(any("simulator protected path changed" in f for f in failures))
        self.assertTrue(any("website protected path changed" in f for f in failures))

    def test_a_protected_path_appearing_is_a_failure(self) -> None:
        after = dict(CLEAN)
        after["website"] = state(head="web1", versions="tree-web", archive="new")
        failures, outcome = verdict(after=after)
        self.assertEqual(outcome, FAILED)
        self.assertTrue(any("protected path appeared: archive" in f for f in failures))


class SafetyChecksAlwaysRunTests(unittest.TestCase):
    """Defect 2: a parse failure skipped every safety check.

    The script raised while parsing the renderer's output, before comparing
    HEADs, working trees and the pointer. An unreadable render is a failure of
    the render; whether production state was disturbed is the more important
    question and was the one that went unasked.
    """

    def test_a_parse_failure_still_reports_disturbed_state(self) -> None:
        after = dict(CLEAN)
        after["website"] = state(head="web2", versions="tree-web")
        failures, outcome = verdict(
            rendered=None, parse_error="no result JSON", exit_code=1, after=after)
        self.assertEqual(outcome, FAILED)
        self.assertIn("no result JSON", failures)
        self.assertTrue(
            any("website HEAD moved" in f for f in failures),
            "the safety checks were skipped when parsing failed")

    def test_a_parse_failure_alone_is_still_a_failure(self) -> None:
        failures, outcome = verdict(rendered=None, parse_error="no result JSON")
        self.assertEqual(outcome, FAILED)
        self.assertEqual(failures, ["no result JSON"])

    def test_a_moved_head_fails_even_on_an_otherwise_perfect_render(self) -> None:
        after = dict(CLEAN)
        after["simulator"] = state(head="sim2", versions="tree-sim", archive="tree-arch")
        failures, outcome = verdict(after=after)
        self.assertEqual(outcome, FAILED)
        self.assertTrue(any("simulator HEAD moved" in f for f in failures))

    def test_a_dirty_working_tree_fails(self) -> None:
        after = dict(CLEAN)
        after["website"] = state(head="web1", status=" M some/file", versions="tree-web")
        failures, outcome = verdict(after=after)
        self.assertEqual(outcome, FAILED)
        self.assertTrue(any("website working tree changed" in f for f in failures))

    def test_a_moved_pointer_fails(self) -> None:
        failures, outcome = verdict(pointer_after=b'{"publication_generation": "other"}')
        self.assertEqual(outcome, FAILED)
        self.assertIn("the website pointer changed", failures)

    def test_state_captured_on_one_side_only_is_a_failure(self) -> None:
        failures, outcome = verdict(after={"simulator": CLEAN["simulator"]})
        self.assertEqual(outcome, FAILED)
        self.assertTrue(any("website state was not captured" in f for f in failures))


class VerdictIsActuallyAssertedTests(unittest.TestCase):
    """Defect 3: "both views rebuilt" was printed, never checked.

    The message branched only on whether the primary view was omitted, so it
    could claim both views were rebuilt without the views being complete, and
    without the reported generation or the dry-run status being what was
    asked for.
    """

    def test_a_recognised_parity_omission_is_reconstruction_only(self) -> None:
        failures, outcome = verdict(
            views={
                "status": "INCOMPLETE",
                "primary_campaign_paths": "OMITTED",
                "missing": ["future_campaign_paths"],
                "error": "ValueError: ... not bitwise identical to the canonical ...",
            },
            exit_code=1,
        )
        self.assertEqual(failures, [])
        self.assertEqual(
            outcome, RECONSTRUCTION_ONLY,
            "a known parity omission must not read as full acceptance")

    def test_an_unexplained_omission_is_a_failure(self) -> None:
        failures, outcome = verdict(
            views={
                "status": "INCOMPLETE",
                "primary_campaign_paths": "OMITTED",
                "missing": ["future_campaign_paths"],
                "error": "RuntimeError: something else entirely",
            },
            exit_code=1,
        )
        self.assertEqual(outcome, FAILED)
        self.assertTrue(any("unexpected reason" in f for f in failures))

    def test_incomplete_views_cannot_pass_as_acceptance(self) -> None:
        failures, outcome = verdict(views={"status": "INCOMPLETE"})
        self.assertEqual(outcome, FAILED)
        self.assertTrue(any("without both views rebuilt" in f for f in failures))

    def test_a_missing_view_cannot_pass_as_acceptance(self) -> None:
        failures, outcome = verdict(views={"missing": ["future_campaign_paths"]})
        self.assertEqual(outcome, FAILED)

    def test_a_secondary_view_that_was_not_rebuilt_fails(self) -> None:
        failures, outcome = verdict(views={"secondary_projection": "OMITTED"})
        self.assertEqual(outcome, FAILED)

    def test_the_wrong_generation_fails(self) -> None:
        failures, outcome = verdict(rendered=rendered(generation="20260101T000000Z-other"))
        self.assertEqual(outcome, FAILED)
        self.assertTrue(any("expected " + repr(GENERATION) in f for f in failures))

    def test_a_render_that_installed_something_fails(self) -> None:
        failures, outcome = verdict(rendered=rendered(status="RENDERED_AND_DEPLOYED"))
        self.assertEqual(outcome, FAILED)
        self.assertTrue(any("expected a staged dry run" in f for f in failures))

    def test_the_requested_date_must_be_the_one_reconstructed(self) -> None:
        failures, outcome = verdict(curve={"reconstructed": ["2026-09-01"]})
        self.assertEqual(outcome, FAILED)
        self.assertTrue(any(f"{TARGET} was not reconstructed" in f for f in failures))

    def test_reconstructing_nothing_fails(self) -> None:
        # The exact shape of the first false green: the fixture never applied,
        # so the render had nothing to do and said so.
        failures, outcome = verdict(curve={"reconstructed": []})
        self.assertEqual(outcome, FAILED)

    def test_a_remaining_curve_gap_fails(self) -> None:
        failures, outcome = verdict(curve={"missing": ["2026-09-08"]})
        self.assertEqual(outcome, FAILED)
        self.assertTrue(any("curve dates still missing" in f for f in failures))

    def test_a_zero_exit_without_rebuilt_views_fails(self) -> None:
        failures, outcome = verdict(
            views={"primary_campaign_paths": "OMITTED", "status": "INCOMPLETE",
                   "error": "not bitwise identical"},
            exit_code=0,
        )
        self.assertEqual(outcome, FAILED)
        self.assertTrue(any("exited 0 without both views rebuilt" in f for f in failures))

    def test_a_nonzero_exit_with_nothing_short_fails(self) -> None:
        failures, outcome = verdict(exit_code=1)
        self.assertEqual(outcome, FAILED)
        self.assertTrue(
            any("nothing recognised as short" in f for f in failures), failures)

    def test_an_unrecognised_view_state_fails(self) -> None:
        failures, outcome = verdict(views={"primary_campaign_paths": "SOMETHING_NEW"})
        self.assertEqual(outcome, FAILED)
        self.assertTrue(any("unrecognised primary view state" in f for f in failures))


class ExitCodeTests(unittest.TestCase):
    def test_the_three_outcomes_have_distinct_exit_codes(self) -> None:
        from tests.support.reconstruction_canary import EXIT_CODES

        self.assertEqual(EXIT_CODES[FULL_ACCEPTANCE], 0)
        self.assertNotEqual(EXIT_CODES[RECONSTRUCTION_ONLY], 0)
        self.assertNotEqual(EXIT_CODES[FAILED], 0)
        self.assertNotEqual(
            EXIT_CODES[RECONSTRUCTION_ONLY], EXIT_CODES[FAILED],
            "a known limitation and a real failure must be distinguishable")


if __name__ == "__main__":
    unittest.main()
