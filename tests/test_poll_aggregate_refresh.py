from __future__ import annotations

import filecmp
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.poll_aggregate import refresh as RF
from scripts.pollofpolls.__main__ import PollingValidationError
from scripts.simulator.model_inputs import AGGREGATE_DIR, AGGREGATE_POLLS, AGGREGATE_TIMESERIES, SWEDISHPOLLS_POLLS

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/pollofpolls"
PROCESSED = ROOT / "data/processed"


def offline_acquisition(raw_dir: Path, **kwargs):
    """Stand-in for the network: the committed SwedishPolls raw files."""

    assert tuple(s.key for s in kwargs["sources"]) == ("swedishpolls", "swedishpolls_sources")
    assert kwargs["manifest_filename"] == RF.MANIFEST_FILENAME
    for name in ("swedishpolls_polls.csv", "swedishpolls_sources.csv"):
        shutil.copyfile(RAW / name, raw_dir / name)
    return {"acquisition_diagnostics": []}, ["swedishpolls: unchanged"]


class RefreshTests(unittest.TestCase):
    def stage(self, tmp: Path) -> tuple[Path, Path]:
        raw, processed = tmp / "data/raw/pollofpolls", tmp / "data/processed/pollofpolls"
        raw.mkdir(parents=True)
        shutil.copytree(PROCESSED / AGGREGATE_DIR, tmp / "data/processed" / AGGREGATE_DIR)
        processed.mkdir(parents=True)
        return raw, processed

    def test_committed_inputs_reproduce_the_committed_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            raw, processed = self.stage(tmp)

            def no_refit(*args, **kwargs):
                raise AssertionError("stored segments must not be refitted")

            from scripts.poll_aggregate import outputs
            real_build = outputs.build_aggregate

            def build_without_refit(*args, **kwargs):
                return real_build(*args, fit=no_refit, **kwargs)

            with patch.object(RF, "acquire_all", side_effect=offline_acquisition), \
                    patch.object(outputs, "build_aggregate", side_effect=build_without_refit):
                result = RF.refresh_snapshot(raw, processed)
            self.assertTrue(filecmp.cmp(processed / SWEDISHPOLLS_POLLS.name, PROCESSED / SWEDISHPOLLS_POLLS,
                                        shallow=False))
            for relative in (AGGREGATE_TIMESERIES, AGGREGATE_POLLS, AGGREGATE_DIR / "swedishpolls_aggregate_metadata.json"):
                with self.subTest(relative=relative):
                    self.assertTrue(filecmp.cmp(tmp / "data/processed" / relative, PROCESSED / relative, shallow=False))
            self.assertIn("aggregate: SwedishPollsAggregate-v0.2", result["messages"][-1])

    def test_invalid_swedishpolls_is_a_validation_error_and_installs_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            raw, processed = self.stage(tmp)
            with patch.object(RF, "acquire_all", side_effect=offline_acquisition), \
                    patch.object(RF, "validate_swedishpolls",
                                 return_value=[{"severity": "error", "code": "x", "message": "bad"}]):
                with self.assertRaises(PollingValidationError):
                    RF.refresh_snapshot(raw, processed)
            self.assertFalse((processed / SWEDISHPOLLS_POLLS.name).exists())

    def test_the_aggregate_is_the_sibling_of_the_staged_table(self) -> None:
        self.assertEqual(RF.aggregate_dir_for(Path("/x/data/processed/pollofpolls")),
                         Path("/x/data/processed") / AGGREGATE_DIR)


if __name__ == "__main__":
    unittest.main()
