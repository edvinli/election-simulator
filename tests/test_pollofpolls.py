from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from scripts.pollofpolls.acquire import USER_AGENT, _read_url
from scripts.pollofpolls.normalize import (
    enrich_with_swedishpolls,
    merge_homepage_polls,
    normalize_party,
    parse_date,
    parse_homepage_polls,
    parse_party_chart_payload,
    parse_percentage,
    parse_swedishpolls_payloads,
    parse_timeseries_payload,
    polls_to_long_rows,
    reconstruct_chart_polls,
)
from scripts.pollofpolls.validate import (
    INDIVIDUAL_FIELDS,
    SWEDISHPOLLS_FIELDS,
    TIMESERIES_FIELDS,
    validate_individual_polls,
    validate_swedishpolls,
    validate_timeseries,
)


FIXTURES = Path(__file__).parent / "fixtures"


class ParsingTests(unittest.TestCase):
    def test_party_name_normalization_preserves_historical_liberal_aliases(self) -> None:
        for source_label in ("FP", "Folkpartiet", "Folkpartiet Liberalerna", "Liberalerna", "L"):
            self.assertEqual(normalize_party(source_label), "L")
        self.assertEqual(normalize_party("Övriga"), "other")
        self.assertEqual(normalize_party("Piratpartiet"), "Piratpartiet")

    def test_date_parsing(self) -> None:
        self.assertEqual(parse_date("2014-09-15"), date(2014, 9, 15))
        with self.assertRaises(ValueError):
            parse_date("15/09")

    def test_percentage_and_missing_value_parsing(self) -> None:
        self.assertEqual(parse_percentage("28,4 %"), 28.4)
        self.assertEqual(parse_percentage("0.0"), 0.0)
        for missing in ("", "NaN", "null", None):
            self.assertIsNone(parse_percentage(missing))

    def test_html_timeseries_source_and_representative_observation(self) -> None:
        rows, labels = parse_timeseries_payload(
            (FIXTURES / "pollofpolls_timeseries.html").read_bytes()
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["date"], "2014-09-15")
        self.assertEqual(rows[0]["M"], 23.3)
        self.assertEqual(rows[0]["L"], 5.8)
        self.assertEqual(rows[0]["FI"], 3.0)
        self.assertIsNone(rows[0]["other"])
        self.assertIsNone(rows[1]["FI"])
        self.assertEqual(labels["L"], "FP")

    def test_homepage_period_and_original_pollster(self) -> None:
        polls = parse_homepage_polls(
            (FIXTURES / "homepage.html").read_bytes(), date(2026, 1, 10)
        )
        self.assertEqual(polls[0]["pollster"], "Sifo")
        self.assertEqual(polls[0]["pollster_original"], "Kantar-Sifo")
        self.assertEqual(polls[0]["interview_start"], date(2025, 12, 28))
        self.assertEqual(polls[0]["interview_end"], date(2026, 1, 5))
        self.assertEqual(polls[0]["values"]["other"], 2.4)


class ReconstructionAndValidationTests(unittest.TestCase):
    def _chart_polls(self):
        payloads = {
            "M": parse_party_chart_payload((FIXTURES / "party_M.csv").read_bytes(), "M"),
            "L": parse_party_chart_payload((FIXTURES / "party_L.csv").read_bytes(), "L"),
        }
        return reconstruct_chart_polls(payloads)

    def test_chart_runs_reconstruct_interview_spans(self) -> None:
        polls = self._chart_polls()
        demoskop = next(poll for poll in polls if poll["pollster"] == "Demoskop")
        ipsos = next(poll for poll in polls if poll["pollster"] == "Ipsos")
        self.assertEqual(demoskop["interview_start"], date(2014, 9, 1))
        self.assertEqual(demoskop["interview_end"], date(2014, 9, 3))
        self.assertEqual(demoskop["values"], {"M": 22.1, "L": 5.5})
        self.assertEqual(ipsos["interview_start"], date(2014, 9, 2))
        self.assertEqual(ipsos["interview_end"], date(2014, 9, 4))

    def test_ambiguous_fi_zero_is_not_a_real_zero(self) -> None:
        homepage = parse_homepage_polls(
            (FIXTURES / "homepage.html").read_bytes(), date(2026, 1, 10)
        )
        polls = merge_homepage_polls([], homepage)
        rows = polls_to_long_rows(polls, {"http://pollofpolls.se/": "2026-01-10T00:00:00+00:00"})
        fi = next(row for row in rows if row["party"] == "FI")
        self.assertIsNone(fi["support"])
        self.assertEqual(fi["source_value"], 0.0)
        self.assertEqual(fi["support_status"], "ambiguous_zero_or_included_in_other")

    def test_duplicate_timeseries_dates_are_reported(self) -> None:
        base = {field: None for field in TIMESERIES_FIELDS}
        base.update({"date": "2014-09-15", "M": 23.3, "source_url": "fixture", "retrieved_at": "fixture"})
        issues = validate_timeseries([dict(base), dict(base)])
        codes = {issue["code"] for issue in issues}
        self.assertIn("duplicate_timeseries_dates", codes)

    def test_schema_validation_for_long_rows(self) -> None:
        poll = {
            "poll_id": "fixture",
            "pollster": "Demoskop",
            "pollster_original": "Demoskop",
            "interview_start": date(2014, 9, 1),
            "interview_end": date(2014, 9, 3),
            "values": {"M": 22.1, "L": 5.5},
            "value_sources": {},
        }
        rows = polls_to_long_rows([poll], {"http://pollofpolls.se/": "fixture"})
        self.assertEqual(set(rows[0]), set(INDIVIDUAL_FIELDS))
        errors = [issue for issue in validate_individual_polls(rows) if issue["severity"] == "error"]
        self.assertEqual(errors, [])

    def test_swedishpolls_parser_preserves_missing_dates_and_provenance(self) -> None:
        wide, rows = parse_swedishpolls_payloads(
            (FIXTURES / "swedishpolls.csv").read_bytes(),
            (FIXTURES / "swedishpolls_sources.csv").read_bytes(),
            retrieved_at="2026-08-26T00:00:00+00:00",
        )
        self.assertEqual(len(wide), 2)
        self.assertEqual(len(rows), 20)
        self.assertEqual(set(rows[0]), set(SWEDISHPOLLS_FIELDS))
        ipsos_m = next(row for row in rows if row["pollster"] == "Ipsos" and row["party"] == "M")
        self.assertEqual(ipsos_m["support"], 17.9)
        self.assertEqual(ipsos_m["sample_size"], 1661)
        self.assertIn("https://example.test/ipsos", ipsos_m["row_source_references_json"])
        sifo = next(poll for poll in wide if poll["pollster"] == "Sifo")
        self.assertIsNone(sifo["interview_start"])
        self.assertIsNone(sifo["interview_end"])
        self.assertTrue(sifo["collection_period_approximate"])
        errors = [issue for issue in validate_swedishpolls(rows) if issue["severity"] == "error"]
        self.assertEqual(errors, [])

    def test_exact_supplementary_match_enriches_metadata_only(self) -> None:
        wide, _ = parse_swedishpolls_payloads(
            (FIXTURES / "swedishpolls.csv").read_bytes(),
            (FIXTURES / "swedishpolls_sources.csv").read_bytes(),
            retrieved_at="2026-08-26T00:00:00+00:00",
        )
        poll = {
            "poll_id": "pop-fixture",
            "pollster": "Ipsos",
            "pollster_original": "Ipsos",
            "interview_start": date(2026, 8, 11),
            "interview_end": date(2026, 8, 23),
            "values": {"M": 18.1, "L": 2.3},
            "value_sources": {},
        }
        rows = polls_to_long_rows([poll], {"http://pollofpolls.se/": "fixture"})
        crosswalk = enrich_with_swedishpolls(
            rows, wide, metadata_retrieved_at="2026-08-26T00:00:00+00:00"
        )
        m_row = next(row for row in rows if row["party"] == "M")
        self.assertEqual(m_row["support"], 18.1)
        self.assertEqual(m_row["source_value"], 18.1)
        self.assertEqual(m_row["publication_date"], "2026-08-25")
        self.assertEqual(m_row["sample_size"], 1661)
        self.assertEqual(m_row["metadata_match_status"], "exact_span_match")
        self.assertEqual(crosswalk[0]["max_party_absolute_difference"], 0.2)

    def test_duplicate_supplementary_poll_is_reported_not_discarded(self) -> None:
        _, rows = parse_swedishpolls_payloads(
            (FIXTURES / "swedishpolls.csv").read_bytes(),
            (FIXTURES / "swedishpolls_sources.csv").read_bytes(),
            retrieved_at="2026-08-26T00:00:00+00:00",
        )
        duplicate = [dict(row, poll_id="swp-duplicate") for row in rows[:10]]
        issues = validate_swedishpolls(rows + duplicate)
        self.assertIn("swedishpolls_duplicate_candidates", {issue["code"] for issue in issues})


class AcquisitionIdentityTests(unittest.TestCase):
    """The ingestion identifies this repository, and nothing more."""

    def test_user_agent_names_the_current_repository(self) -> None:
        self.assertEqual(
            USER_AGENT,
            "election-simulator-pollofpolls-ingestion/1.0 "
            "(+https://github.com/edvinli/election-simulator)",
        )
        # The pre-extraction identity must not resurface.
        self.assertNotIn("edvinli.github.io", USER_AGENT)

    def test_user_agent_is_only_an_identification_header(self) -> None:
        """Changing the identity must not change what acquisition requests."""

        captured = {}

        class _Response:
            status = 200
            headers = {"content-type": "text/csv"}

            def read(self):
                return b"payload"

            def geturl(self):
                return "https://example.invalid/polls.csv"

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def _fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.headers)
            captured["timeout"] = timeout
            return _Response()

        with patch("scripts.pollofpolls.acquire.urllib.request.urlopen", _fake_urlopen):
            payload, meta = _read_url("https://example.invalid/polls.csv", 30.0)

        self.assertEqual(payload, b"payload")
        self.assertEqual(meta["http_status"], 200)
        self.assertEqual(meta["http_headers"], {"content-type": "text/csv"})
        self.assertEqual(captured["url"], "https://example.invalid/polls.csv")
        self.assertEqual(captured["timeout"], 30.0)
        # urllib title-cases header names; the identity is carried in exactly
        # one header and the request is otherwise unchanged.
        self.assertEqual(captured["headers"]["User-agent"], USER_AGENT)
        self.assertEqual(
            set(captured["headers"]), {"User-agent", "Accept", "Accept-encoding"}
        )
        self.assertEqual(captured["headers"]["Accept"], "text/csv,text/html;q=0.9,*/*;q=0.1")
        self.assertEqual(captured["headers"]["Accept-encoding"], "identity")


class RefreshStabilityTests(unittest.TestCase):
    """Processed scientific outputs must be byte-stable across semantically identical refreshes."""

    def _normalize_fixture_homepage(self, html_payload: bytes, retrieved_at: str) -> list[dict]:
        reference = parse_date(retrieved_at[:10])
        homepage_polls = parse_homepage_polls(html_payload, reference)
        chart_polls = reconstruct_chart_polls(
            {
                "M": parse_party_chart_payload((FIXTURES / "party_M.csv").read_bytes(), "M"),
                "L": parse_party_chart_payload((FIXTURES / "party_L.csv").read_bytes(), "L"),
            }
        )
        polls = merge_homepage_polls(chart_polls, homepage_polls)
        return polls_to_long_rows(polls)

    def test_homepage_table_id_and_retrieval_time_differ_produces_byte_identical_rows(self) -> None:
        raw_text = (FIXTURES / "homepage.html").read_text(encoding="utf-8")
        payload_a = raw_text.replace(
            'table id="csvtohtml_id-6a90a39fbab3e"',
            'table id="csvtohtml_id-aaaaaaaaaaaaa"',
        ).encode("utf-8")
        payload_b = raw_text.replace(
            'table id="csvtohtml_id-6a90a39fbab3e"',
            'table id="csvtohtml_id-bbbbbbbbbbbbb"',
        ).encode("utf-8")

        rows_a = self._normalize_fixture_homepage(payload_a, "2026-08-27T20:52:47+00:00")
        rows_b = self._normalize_fixture_homepage(payload_b, "2026-08-28T21:06:02+00:00")

        self.assertEqual(rows_a, rows_b)
        # Format both to CSV bytes to verify byte-level stability
        import io, csv
        buf_a, buf_b = io.StringIO(), io.StringIO()
        writer_a = csv.DictWriter(buf_a, fieldnames=list(INDIVIDUAL_FIELDS))
        writer_b = csv.DictWriter(buf_b, fieldnames=list(INDIVIDUAL_FIELDS))
        writer_a.writeheader(); writer_a.writerows(rows_a)
        writer_b.writeheader(); writer_b.writerows(rows_b)
        self.assertEqual(buf_a.getvalue().encode("utf-8"), buf_b.getvalue().encode("utf-8"))

    def test_actual_poll_value_change_changes_processed_output(self) -> None:
        raw_text = (FIXTURES / "homepage.html").read_text(encoding="utf-8")
        payload_a = raw_text.encode("utf-8")
        # Modify Kantar-Sifo support value in the first row from 18,2 to 19,5
        payload_c = raw_text.replace("18,2", "19,5", 1).encode("utf-8")
        self.assertNotEqual(payload_a, payload_c)

        rows_a = self._normalize_fixture_homepage(payload_a, "2026-08-27T20:52:47+00:00")
        rows_c = self._normalize_fixture_homepage(payload_c, "2026-08-27T20:52:47+00:00")

        self.assertNotEqual(rows_a, rows_c)
        changed_rows = [
            (ra, rc) for ra, rc in zip(rows_a, rows_c) if ra != rc
        ]
        self.assertEqual(len(changed_rows), 10)
        m_a = next(ra for ra, _ in changed_rows if ra["party"] == "M")
        m_c = next(rc for _, rc in changed_rows if rc["party"] == "M")
        self.assertEqual(m_a["pollster"], "Sifo")
        self.assertEqual(m_a["support"], 18.2)
        self.assertEqual(m_c["support"], 19.5)
        self.assertNotEqual(m_a["poll_id"], m_c["poll_id"])


if __name__ == "__main__":
    unittest.main()

