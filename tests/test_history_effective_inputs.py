"""Reuse is certified by historical model inputs, not acquisition row IDs."""
from __future__ import annotations

from copy import deepcopy
import csv
from datetime import date
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.forecast_history.contract import deterministic_history_sha256, validate_history_contract
from scripts.forecast_history.effective_inputs import DATA_FILES, EffectiveInputs, ROOT, previous_fingerprints
from scripts.forecast_history.generate import build_history, first_changed_poll_date, update_history_with_production_result
from scripts.pollofpolls.normalize import parse_swedishpolls_payloads
from tests import test_forecast_history
from types import SimpleNamespace


class EffectiveInputReuseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name)
        for relative in DATA_FILES:
            target = self.data / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / 'data/processed' / relative, target)
        self.calls = []
        votes, seats = test_forecast_history.ForecastHistoryTests._matrices()

        def runner(**kwargs):
            self.calls.append(kwargs['as_of'])
            return SimpleNamespace(vote_shares_matrix=votes, seats_matrix=seats,
                                   manifest={'source_git_commit': 'a' * 40})

        self.kwargs = dict(dates=['2026-05-23', '2026-05-24', '2026-05-25'],
                           samples=4, production_latest_samples=4, seed=12345,
                           model_commit='a' * 40, archive_dir=None, simulation_runner=runner,
                           model_data_dir=self.data,
                           poll_file=self.data / DATA_FILES[2], timeseries_file=self.data / DATA_FILES[1])

    def rewrite(self, relative, transform):
        path = self.data / relative
        with path.open() as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        rows = transform(rows)
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def rebuild(self, payload):
        self.calls.clear()
        return build_history(**self.kwargs, existing_payload=payload)

    def test_row_reordering_and_legacy_id_replacement_reuse_certified_points(self):
        first = build_history(**self.kwargs)
        for relative in (DATA_FILES[0], DATA_FILES[2]):
            def reorder(rows):
                for row in rows:
                    row['poll_id'] = 'new-transport-' + row['poll_id']
                return list(reversed(rows))
            self.rewrite(relative, reorder)
        rebuilt = self.rebuild(first)
        self.assertEqual(self.calls, [])
        self.assertEqual(rebuilt['series'], first['series'])
        self.assertEqual(rebuilt['reconstruction_inputs'], first['reconstruction_inputs'])

    def test_future_poll_insertion_and_future_timeseries_do_not_invalidate(self):
        first = build_history(**self.kwargs)
        def insert(rows):
            poll = rows[0]['poll_id']
            new = [dict(row) for row in rows if row['poll_id'] == poll]
            for row in new:
                row.update(poll_id='future-insert', publication_date='2026-06-01',
                           interview_start='2026-05-26', interview_end='2026-05-30')
            return new + list(reversed(rows))
        for relative in (DATA_FILES[0], DATA_FILES[2]):
            self.rewrite(relative, insert)
        def future_ts(rows):
            for row in rows:
                if row['date'] > '2026-05-25':
                    row['M'] = str(float(row['M']) + 0.01)
                    row['S'] = str(float(row['S']) - 0.01)
            return list(reversed(rows))
        self.rewrite(DATA_FILES[1], future_ts)
        self.assertEqual(self.rebuild(first)['series'], first['series'])
        self.assertEqual(self.calls, [])

    def test_real_historical_opinion_change_invalidates_only_affected_dates(self):
        first = build_history(**self.kwargs)
        def revise(rows):
            for row in rows:
                if row['date'] == '2026-05-24':
                    row['M'] = str(float(row['M']) + 0.5)
                    row['S'] = str(float(row['S']) - 0.5)
            return rows
        self.rewrite(DATA_FILES[1], revise)
        rebuilt = self.rebuild(first)
        self.assertEqual(self.calls, ['2026-05-24'])
        self.assertEqual(rebuilt['series'][0], first['series'][0])
        # The official point is an immutable publication, not a reconstructed cache entry.
        self.assertEqual(rebuilt['series'][-1], first['series'][-1])

    def test_historical_individual_poll_value_change_invalidates(self):
        first = build_history(**self.kwargs)
        def revise(rows):
            eligible = [r for r in rows if r['publication_date'] and r['publication_date'] < '2026-05-23'
                        and r['interview_end'] and r['interview_end'] <= '2026-05-23'
                        and r['support'] and r['party'] == 'M']
            target = max(eligible, key=lambda r: r['publication_date'])['poll_id']
            for row in rows:
                if row['poll_id'] == target and row['party'] == 'M':
                    row['support'] = str(float(row['support']) + 0.1)
            return rows
        self.rewrite(DATA_FILES[0], revise)
        self.rebuild(first)
        self.assertEqual(self.calls, ['2026-05-23', '2026-05-24'])

    def test_training_input_change_invalidates_even_before_chart_start(self):
        first = build_history(**self.kwargs)
        def revise(rows):
            for row in rows:
                if '2022-09-01' <= row['publication_date'] <= '2022-09-11' and row['support']:
                    if row['party'] == 'M':
                        row['support'] = str(float(row['support']) + 0.1)
                    elif row['party'] == 'S':
                        row['support'] = str(float(row['support']) - 0.1)
            return rows
        self.rewrite(DATA_FILES[2], revise)
        self.rebuild(first)
        self.assertEqual(self.calls, ['2026-05-23', '2026-05-24'])

    def test_publication_and_fieldwork_eligibility_are_preserved(self):
        first = build_history(**self.kwargs)
        def insert(rows):
            poll = rows[0]['poll_id']
            copies = []
            for published, end, suffix in [('2026-06-01', '2026-05-20', 'future-publication'),
                                           ('2026-05-20', '2026-06-01', 'future-fieldwork'),
                                           ('', '2026-05-20', 'undated')]:
                for original in rows:
                    if original['poll_id'] == poll:
                        row = dict(original)
                        row.update(poll_id=suffix, publication_date=published,
                                   interview_start='2026-05-19', interview_end=end)
                        copies.append(row)
            return copies + rows
        self.rewrite(DATA_FILES[0], insert)
        self.assertEqual(self.rebuild(first)['series'], first['series'])
        self.assertEqual(self.calls, [])

    def test_missing_legacy_source_fails_closed(self):
        first = build_history(**self.kwargs)
        first.pop('reconstruction_inputs')
        first['deterministic_content_sha256'] = deterministic_history_sha256(first)
        self.rebuild(first)
        self.assertEqual(self.calls, ['2026-05-23', '2026-05-24'])

    def test_legacy_source_must_match_recorded_hashes(self):
        payload = json.loads((ROOT / 'files/election-simulator/history/coalition-timeseries.json').read_text())
        payload['source_hashes']['timeseries_source_sha256'] = '0' * 64
        self.assertEqual(previous_fingerprints(payload), {})

    def test_dirty_legacy_provenance_cannot_be_recovered_from_git(self):
        payload = json.loads((ROOT / 'files/election-simulator/history/coalition-timeseries.json').read_text())
        payload['source_worktree_clean'] = False
        self.assertEqual(previous_fingerprints(payload), {})

    def test_roll_in_preserves_original_fingerprints_after_input_revision(self):
        first = build_history(**self.kwargs)
        def revise(rows):
            for row in rows:
                if row['date'] == '2026-05-24':
                    row['M'] = str(float(row['M']) + 0.1)
                    row['S'] = str(float(row['S']) - 0.1)
            return rows
        self.rewrite(DATA_FILES[1], revise)
        votes, seats = test_forecast_history.ForecastHistoryTests._matrices()
        result = SimpleNamespace(
            vote_shares_matrix=votes, seats_matrix=seats,
            summary=SimpleNamespace(as_of='2026-05-26', total_samples=4),
            manifest={'source_git_commit': 'a' * 40, 'base_seed': 12345},
        )
        rolled = update_history_with_production_result(
            first, result, poll_file=self.kwargs['poll_file'],
            timeseries_file=self.kwargs['timeseries_file'], archive_dir=None,
        )
        self.assertEqual(rolled['reconstruction_inputs'], first['reconstruction_inputs'])
        self.assertEqual(rolled['series'][:2], first['series'][:2])

    def test_seed_change_is_not_input_equivalence(self):
        before = EffectiveInputs(self.data, election_date=date(2026, 9, 13), seed=12345)
        after = EffectiveInputs(self.data, election_date=date(2026, 9, 13), seed=12346)
        day = date(2026, 5, 24)
        self.assertNotEqual(before.fingerprint(day), after.fingerprint(day))

    def test_input_metadata_is_validated_and_covered_by_self_hash(self):
        first = build_history(**self.kwargs)
        bad = deepcopy(first)
        bad['reconstruction_inputs']['dates']['2026-05-23'] = 'bad'
        bad['deterministic_content_sha256'] = deterministic_history_sha256(bad)
        with self.assertRaisesRegex(ValueError, 'effective input fingerprint'):
            validate_history_contract(bad)
        bad = deepcopy(first)
        bad['reconstruction_inputs']['dates']['2026-05-23'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'does not match payload'):
            validate_history_contract(bad)


class CanonicalPollIdentityTests(unittest.TestCase):
    def test_new_ids_do_not_depend_on_source_row_and_retain_row_provenance(self):
        fixture = ROOT / 'tests/fixtures/swedishpolls.csv'
        reader = csv.DictReader(io.StringIO(fixture.read_text()))
        fields, rows = reader.fieldnames, list(reader)
        def parse(values):
            stream = io.StringIO()
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)
            return parse_swedishpolls_payloads(stream.getvalue().encode(), b'Company;PublYearMonth;PublDate;source\n')[0]
        old = parse(rows)
        new_poll = dict(rows[0], PublDate='2026-09-07', PublYearMonth='2026-sep')
        new = parse([new_poll, *reversed(rows)])
        self.assertEqual({p['poll_id'] for p in old}, {p['poll_id'] for p in new[1:]})
        self.assertNotEqual([p['source_row'] for p in old], [p['source_row'] for p in new[1:]])

    def test_chart_comparison_interprets_legacy_ids_and_preserves_multiplicity(self):
        poll = dict(poll_id='old-row-id', company='House', publication_date='2026-05-24', parties={'M': 20})
        changed_id = dict(poll, poll_id='stable-id')
        self.assertIsNone(first_changed_poll_date([poll], [changed_id]))
        self.assertEqual(first_changed_poll_date([poll, poll], [changed_id]), date(2026, 5, 24))
