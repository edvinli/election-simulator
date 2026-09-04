"""End-to-end post-election report tests using only local immutable fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prospective_benchmark_2026.archive import ModelCapture, append_capture
from scripts.prospective_benchmark_2026.report import build_report, render_markdown
from scripts.prospective_benchmark_2026.time_rules import capture_id_for_date, classify_capture_time


PARTIES = ("M", "L", "C", "KD", "S", "V", "MP", "SD")


class TestPostElectionReport(unittest.TestCase):
    def _archive(self, root: Path) -> Path:
        archive = root / "archive"
        archive.mkdir()
        protocol = b'{"frozen":true}\n'
        digest = hashlib.sha256(protocol).hexdigest()
        (archive / "protocol.json").write_bytes(protocol)
        (archive / "protocol.sha256").write_text(f"{digest}  protocol.json\n", encoding="utf-8")
        (archive / "index.json").write_text(json.dumps({
            "schema_version": "1.0",
            "protocol_path": "protocol.json",
            "protocol_sha256": digest,
            "captures": [],
        }) + "\n", encoding="utf-8")
        return archive

    def _forecast(self, system: str, *, offset: float) -> dict:
        actual = {party: value for party, value in zip(PARTIES, (20, 4, 7, 6, 30, 8, 5, 18))}
        central = {party: value + offset for party, value in actual.items()}
        quantiles = {
            party: {"0.05": value - 2 + offset, "0.50": value + offset, "0.95": value + 2 + offset}
            for party, value in actual.items()
        }
        return {
            "schema_version": "1.0",
            "system": system,
            "available": True,
            "election_date": "2026-09-13",
            "party_order": list(PARTIES),
            "vote_share_unit": "percentage_points",
            "vote_share_denominator": "official_national_valid_votes",
            "published_central_prediction": {"kind": "published", "values": central},
            "published_quantiles": quantiles,
            "threshold_probabilities_4pct": {"L": 0.5, "C": 0.9, "KD": 0.8, "MP": 0.7},
            "draws": {"verified_predictive_vote_draws": False, "path": None},
        }

    def _append(self, archive: Path, day: str) -> None:
        append_capture(
            root=archive,
            capture_id=capture_id_for_date(day),
            timing=classify_capture_time(day, f"{day}T21:31:00Z", durable=True).to_dict(),
            models={
                "election_simulator": ModelCapture(
                    status="AVAILABLE",
                    forecast=self._forecast("election_simulator", offset=0.5),
                    provenance={"generation_id": f"es-{day}"},
                ),
                "botten_ada": ModelCapture(
                    status="PARITY_UNVERIFIED",
                    forecast=self._forecast("botten_ada", offset=0.2),
                    provenance={"source": "official fixture", "draws": "PARITY_UNVERIFIED"},
                ),
            },
        )

    def _result(self, root: Path, *, status: str = "FINAL_CERTIFIED") -> Path:
        raw = root / "official-raw.json"
        raw.write_bytes(b"final certified fixture\n")
        denominator = 1_000_000
        shares = dict(zip(PARTIES, (20, 4, 7, 6, 30, 8, 5, 18)))
        manifest = {
            "schema_version": "1.0",
            "authority": "Valmyndigheten",
            "certification_status": status,
            "election_date": "2026-09-13",
            "official_source_url": "https://resultat.val.se/final",
            "retrieved_at_utc": "2026-09-30T10:00:00Z",
            "raw_path": raw.name,
            "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "valid_national_votes": denominator,
            "parties": {
                party: {
                    "votes": int(share * denominator / 100),
                    "vote_share_percentage_points": share,
                    "seats": 20,
                }
                for party, share in shares.items()
            },
        }
        path = root / "result.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_wis_fallback_report_and_equal_date_weighting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = self._archive(root)
            self._append(archive, "2026-09-04")
            self._append(archive, "2026-09-12")
            report = build_report(archive_root=archive, result_manifest=self._result(root))
            self.assertEqual(report["final_forecast"]["primary_tier"], "compatible_published_quantiles_wis")
            self.assertEqual(report["final_probabilistic_winner"], "botten_ada")
            self.assertEqual(report["final_point_winner"], "botten_ada")
            self.assertEqual(report["campaign"]["scorable_capture_count"], 2)
            self.assertEqual(report["campaign"]["by_metric"]["mean_wis"]["capture_count"], 2)
            self.assertEqual(len(report["official_result"]["normalized_manifest_sha256"]), 64)
            self.assertEqual(report["final_forecast"]["threshold_4pct"]["L"]["outcome_share_gte_4pct"], True)
            markdown = render_markdown(report)
            self.assertIn("Probabilistic winner: `botten_ada`", markdown)
            self.assertIn("small score differences", markdown.lower())

    def test_no_final_capture_means_no_final_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = self._archive(root)
            self._append(archive, "2026-09-04")
            report = build_report(archive_root=archive, result_manifest=self._result(root))
            self.assertIsNone(report["final_forecast"])
            self.assertIsNone(report["final_probabilistic_winner"])


if __name__ == "__main__":
    unittest.main()
