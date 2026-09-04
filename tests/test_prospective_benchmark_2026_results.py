"""Final certified result provenance and denominator tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prospective_benchmark_2026.results import OfficialResultError, PARTY_ORDER, load_official_result


class TestOfficialResult(unittest.TestCase):
    def _write(self, root: Path, *, status: str = "FINAL_CERTIFIED", bad_hash: bool = False) -> Path:
        raw = root / "valmyndigheten-final.json"
        raw.write_bytes(b"official raw fixture\n")
        denominator = 1_000_000
        counts = {party: 100_000 - index * 5_000 for index, party in enumerate(PARTY_ORDER)}
        manifest = {
            "schema_version": "1.0",
            "authority": "Valmyndigheten",
            "certification_status": status,
            "election_date": "2026-09-13",
            "official_source_url": "https://resultat.val.se/val2026/slutligt",
            "retrieved_at_utc": "2026-09-30T12:00:00Z",
            "raw_path": raw.name,
            "raw_sha256": "0" * 64 if bad_hash else hashlib.sha256(raw.read_bytes()).hexdigest(),
            "valid_national_votes": denominator,
            "parties": {
                party: {
                    "votes": count,
                    "vote_share_percentage_points": 100.0 * count / denominator,
                    "seats": 10 + index,
                }
                for index, (party, count) in enumerate(counts.items())
            },
        }
        path = root / "result-manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_requires_final_certification_and_raw_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(OfficialResultError):
                load_official_result(self._write(root, status="PRELIMINARY"))
            with self.assertRaises(OfficialResultError):
                load_official_result(self._write(root, bad_hash=True))

    def test_preserves_official_denominator_without_eight_party_renormalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = load_official_result(self._write(Path(tmp)))
            self.assertEqual(len(result.manifest_sha256), 64)
            self.assertLess(sum(result.vote_shares.values()), 100.0)
            self.assertEqual(result.vote_shares["M"], 10.0)

    def test_share_must_match_votes_over_valid_national_votes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._write(root)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["parties"]["L"]["vote_share_percentage_points"] = 99.0
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(OfficialResultError):
                load_official_result(path)


if __name__ == "__main__":
    unittest.main()
