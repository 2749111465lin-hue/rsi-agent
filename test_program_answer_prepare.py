"""Synthetic paired-root preparation: no API, keys or archived-code execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from code_rsi import live_evolution as live, paired_evolution as paired
from code_rsi import program_answer_prepare as prepare, prepare_musique as pm
from code_rsi.budget import digest, save
from code_rsi.v3 import execution
from code_rsi.v3.infrastructure import DeepSeekTransport, UnknownProviderOutcome
from code_rsi.v3.rag import RagEngine
from test_prepare_answer_probe import RecordedExecutor
from test_v3_musique_calibration import ScriptedTransport
from test_v3_musique_runtime_input import fixture
import test_v3_live_phases as phase_fixture

REVISION = 'a' * 40


class NoChangeTransport:
    def __init__(self):
        self.qa = ScriptedTransport()
        self.calls = 0

    def send(self, body, timeout):
        self.calls += 1
        payload = json.loads(body['messages'][-1]['content'])
        if 'source_files' not in payload:
            return self.qa(body)
        response = {'parent_source_sha256': payload['parent_source_sha256'],
            'change_status': 'no_change', 'edits': [], 'mechanism': 'synthetic no change',
            'intended_target_module': payload['decision']['target_module']}
        return {'model': 'synthetic-frozen-model', 'usage': {'prompt_tokens': 1, 'completion_tokens': 1},
                'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(response)}}]}


class ProgramAnswerPrepareTests(unittest.TestCase):
    def setUp(self):
        f = phase_fixture.LivePhaseTests(methodName='runTest')
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.base = f.base
        template = f.staged(fit_only=True)
        tasks, refs = [], {}
        for index in (1, 0):
            row = fixture('root-prepare-' + str(index), 2)
            row['answer'] = 'Synthetic Port'
            task, reference = pm._adapt(row)
            tasks.append(task); refs[task['question_id']] = reference
        self.tasks = tasks
        template['panels']['D_fit'] = {
            'tasks_file': f.binding(self.base/'public.json', tasks),
            'references_file': f.binding(self.base/'private.json', refs)}
        template.update(schema=live.SCHEMA5, metric='f1', expansions=1, max_calls=29,
            edit_policy={'schema': 'rag-rsi-edit-policy-1', 'mode': 'program'},
            proposal_protocol={'schema': 'rag-rsi-proposal-protocol-1', 'format': 'exact_edits',
                               'max_edits': 8, 'max_edit_chars': 12000})
        template['controls']['case_schedule'] = [{'question_id': t['question_id'], 'repeat': 0} for t in tasks]
        groups = {qid: str(ref.get('pair_group_id', ref.get('source_question_id'))) for qid, ref in refs.items()}
        template['reference_groups_file'] = f.binding(self.base/'groups.json',
            {'schema': live.GROUPS_SCHEMA, 'groups': {'D_fit': groups}})
        self.out = self.base/'paired-source'
        template['output_dir'] = str(self.out/'template_validation_only')
        self.plan = {'schema': paired.SCHEMA2, 'purpose': 'paired_development_smoke',
            'output_dir': str(self.out), 'search_template': template, 'blocks': 1,
            'schedule_policy': paired.SCHEDULE_POLICY, 'max_calls': 44, 'hard_cny': 40,
            'entry_sha256': live._hash(paired.__file__)}
        self.transport = NoChangeTransport()
        with patch.object(execution, 'execute', side_effect=RecordedExecutor()):
            self.completed = paired.run(self.plan, approved_plan_hash=digest(self.plan), execute=True,
                                        transport_factory=lambda: self.transport)
        self.source_patch = patch.object(prepare, '_historical_source', return_value={
            'source_revision': REVISION, 'fixture_note': 'Git source validation independently tested'})
        self.source_patch.start(); self.addCleanup(self.source_patch.stop)

    def packet(self, value=None):
        return prepare.build_paired_root_cases(self.plan if value is None else value, source_revision=REVISION)

    def read(self, path):
        return json.loads(path.read_bytes())

    def test_complete_order_deterministic_and_references_never_opened(self):
        private = Path(self.plan['search_template']['panels']['D_fit']['references_file']['path'])
        private.unlink()  # Source answers are unnecessary after the source run has frozen.
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.out.rglob('*') if p.is_file()}
        calls = self.transport.calls
        with patch.object(DeepSeekTransport, '__init__', side_effect=AssertionError('transport forbidden')):
            a = self.packet()
            b = self.packet(self.out/'paired_plan.json')
        self.assertEqual(a, b)
        self.assertEqual(a['schema'], prepare.PACKET_SCHEMA)
        self.assertEqual([x['case_id'] for x in a['cases']], ['Q01', 'Q02'])
        self.assertEqual([x['task']['question_id'] for x in a['cases']], [x['question_id'] for x in self.tasks])
        self.assertEqual(self.transport.calls, calls)
        self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.out.rglob('*') if p.is_file()})
        for case in a['cases']:
            source = case['source_binding']
            self.assertFalse(source['reference_files_opened'])
            self.assertTrue(source['reconstruction']['all_requests_identical'])
            self.assertEqual(source['reconstruction']['replay']['new_model_calls'], 0)
            self.assertEqual(len(source['model_requests']), 3)
            for request in source['model_requests']:
                self.assertEqual(request['bank'], 'paired/0/root/D_fit/shared-root/0/' + case['task']['question_id'] + '/0')
                value = self.read(Path(request['file']['path']))
                self.assertEqual(request['request_key'], digest({'body': value['body'], 'bank': request['bank']}))
            self.assertNotIn('score', source)

    def test_no_implicit_block_repeat_or_heldout_selection(self):
        for transform in (
            lambda p: p.update(blocks=2),
            lambda p: p['search_template'].update(repeats=2),
            lambda p: p['search_template'].update(phase_order=['search', 'select', 'report']),
            lambda p: p['search_template']['panels'].update(D_report={})):
            p = deepcopy(self.plan); transform(p)
            with self.assertRaises(ValueError): self.packet(p)

    def test_incomplete_run_is_rejected(self):
        path = self.out/'paired_search_complete.json'
        completed = self.read(path); completed['completed_opportunities'] -= 1
        save(path, completed)
        with self.assertRaisesRegex(ValueError, 'completely frozen'): self.packet()

    def test_pending_request_is_not_cleared(self):
        path = next((self.out/'requests').glob('*.json'))
        record = self.read(path); record['state'] = 'pending'; save(path, record)
        before = path.read_bytes()
        with self.assertRaises(UnknownProviderOutcome): self.packet()
        self.assertEqual(path.read_bytes(), before)

    def test_orphan_settlement_or_duplicate_reserve_is_rejected(self):
        path = self.out/'ledger.jsonl'
        original = path.read_bytes()
        for first in (json.loads(original.decode().splitlines()[0]),
                      {'event': 'settle', 'id': 'absent', 'actual': {}, 'reservation_exceeded': False}):
            path.write_bytes(original + json.dumps(first).encode() + b'\n')
            with self.assertRaises(ValueError): self.packet()
        path.write_bytes(original)

    def test_wrong_exact_bank_is_not_replaced_by_same_body(self):
        ledger = self.out/'ledger.jsonl'
        events = [json.loads(line) for line in ledger.read_text().splitlines()]
        event = next(e for e in events if e['event'] == 'reserve' and e['metadata']['paired_condition'] == 'root')
        old = event['metadata']['request_key']
        record_path = self.out/'requests'/(old+'.json')
        record = self.read(record_path)
        event['metadata']['bank'] += '/other'
        key = digest({'body': record['body'], 'bank': event['metadata']['bank']})
        event['metadata']['request_key'] = record['key'] = key
        save(record_path.with_name(key+'.json'), record); record_path.unlink()
        ledger.write_text('\n'.join(json.dumps(e) for e in events)+'\n', encoding='utf-8')
        with self.assertRaises(execution.HostError) as caught: self.packet()
        self.assertIn('exact source bank/body', str(caught.exception.__cause__))

    def test_modified_sealed_cell_and_archive_are_rejected(self):
        path = next((self.out/'blocks/0/root_measurements').glob('*/**/measured.json'))
        original = path.read_bytes(); path.write_text('{}', encoding='utf-8')
        with self.assertRaises(ValueError): self.packet()
        path.write_bytes(original)
        root = self.read(self.out/'blocks/0/cases/root.json')
        source = self.out/'blocks/0/cases/archive/programs'/root['program_id']/'files/rag_core.py'
        source.write_bytes(source.read_bytes()+b'\n')
        with self.assertRaises(ValueError): self.packet()

    def test_public_panel_binding_is_enforced(self):
        path = Path(self.plan['search_template']['panels']['D_fit']['tasks_file']['path'])
        values = self.read(path); values.reverse(); save(path, values)
        with self.assertRaisesRegex(ValueError, 'binding'): self.packet()

    def test_changed_maintained_root_cannot_substitute_for_archive(self):
        old = prepare.root_files
        def drift(config):
            files = old(config); files['rag_core.py'] += '\n'
            return files
        with patch.object(prepare, 'root_files', drift), self.assertRaisesRegex(ValueError, 'contract'):
            self.packet()

    def test_same_final_answer_does_not_excuse_prefix_drift(self):
        class DriftEngine(RagEngine):
            def solve(self, task):
                return super().solve({'question': task['question'] + ' changed'})
        with patch.object(prepare, 'RagEngine', DriftEngine), self.assertRaisesRegex(Exception, 'replay request'):
            self.packet()

    def test_same_answer_does_not_excuse_complete_result_drift(self):
        class DriftEngine(RagEngine):
            def solve(self, task):
                result = super().solve(task); result['stop_reason'] = 'different'
                return result
        with patch.object(prepare, 'RagEngine', DriftEngine), self.assertRaisesRegex(ValueError, 'complete original'):
            self.packet()

    def test_cached_response_tampering_is_not_accepted_as_historical_evidence(self):
        case = self.packet()['cases'][0]
        path = Path(case['source_binding']['model_requests'][0]['file']['path'])
        record = self.read(path)
        response = json.loads(record['response']['choices'][0]['message']['content'])
        response['constraints'] = ['fabricated changed constraint']
        record['response']['choices'][0]['message']['content'] = json.dumps(response)
        save(path, record)
        with self.assertRaisesRegex(ValueError, 'source response'): self.packet()

    def test_host_observation_is_recomputed_even_if_row_hashes_are_resealed(self):
        shared = self.out/'blocks/0/root_measurements'
        seal = self.read(shared/'shared_root_seal.json')
        measurement_path = shared/seal['measurement_identity']/'measurement.json'
        measurement = self.read(measurement_path)
        cell_path = shared/measurement['identity_hash']/(digest(self.tasks[0]['question_id'])[:16]+'_0')/'measured.json'
        cell = self.read(cell_path)
        cell['payload']['host_evidence_trace']['read_presentations'][0]['verified_quotes'] = []
        cell['payload_sha256'] = digest(cell['payload'])
        measurement['rows'][0] = deepcopy(cell['payload'])
        save(cell_path, cell); save(measurement_path, measurement)
        seal['result_sha256'] = digest(measurement)
        seal['artifacts'] = {n: hashlib.sha256((shared/n).read_bytes()).hexdigest() for n in seal['artifacts']}
        seal['seal_hash'] = digest({k: v for k, v in seal.items() if k != 'seal_hash'})
        save(shared/'shared_root_seal.json', seal)
        for arm in ('cases', 'trace'):
            directory = self.out/'blocks/0'/arm
            local_path = directory/'measurements/shared_root.json'
            local = self.read(local_path); local['result'] = deepcopy(measurement)
            local['seal_sha256'] = hashlib.sha256((shared/'shared_root_seal.json').read_bytes()).hexdigest()
            save(local_path, local)
            phase_path = directory/'phase_search.json'; phase = self.read(phase_path)
            phase['artifacts']['measurements/shared_root.json'] = hashlib.sha256(local_path.read_bytes()).hexdigest()
            phase['seal_hash'] = digest({k: v for k, v in phase.items() if k != 'seal_hash'})
            save(phase_path, phase)
        with self.assertRaisesRegex(ValueError, 'host observations'): self.packet()

    def test_completion_opportunity_must_match_arm_seals(self):
        path = self.out/'paired_search_complete.json'
        value = self.read(path); value['terminal_records'][0]['attempt']['status'] = 'measured'; save(path, value)
        with self.assertRaisesRegex(ValueError, 'schedule'): self.packet()


class HistoricalSourceTests(unittest.TestCase):
    def setUp(self):
        names = ['archive.py', 'budget.py', 'sandbox.py', 'sdk.py', 'candidate_runner.py', 'linux_launcher.sh', 'v3/rag.py']
        self.blobs = {name: (name+'\n').encode() for name in names}
        self.blobs.update({'paired_evolution.py': b'# paired\n', 'live_evolution.py': b'# live\n'})
        self.plan = {'entry_sha256': hashlib.sha256(self.blobs['paired_evolution.py'].replace(b'\n', b'\r\n')).hexdigest(),
            'search_template': {'entry_sha256': hashlib.sha256(self.blobs['live_evolution.py']).hexdigest(),
                'runtime_source_hashes': {n: digest(self.blobs[n].decode()) for n in names}}}
        def git(*args):
            if args[0] == 'rev-parse': return (REVISION+'\n').encode()
            if args[0] == 'ls-tree': return b'code_rsi/v3/rag.py\n'
            if args[0] == 'show': return self.blobs[args[1].split(':code_rsi/', 1)[1]]
            raise AssertionError('unexpected Git invocation')
        guard = patch.object(prepare, '_git', side_effect=git)
        guard.start(); self.addCleanup(guard.stop)

    def test_git_bytes_and_explicit_checkout_newlines_are_bound_without_execution(self):
        result = prepare._historical_source(self.plan, REVISION)
        self.assertFalse(result['historical_code_executed'])
        self.assertEqual(result['entry_files']['paired_evolution.py']['checkout_representation'], 'git_lf_to_windows_crlf')
        self.assertEqual(result['entry_files']['live_evolution.py']['checkout_representation'], 'git_blob_bytes')

    def test_revision_alias_path_or_short_hash_is_rejected(self):
        for value in ('HEAD', REVISION[:7], '../elsewhere', REVISION+'^{tree}', None):
            with self.assertRaises(ValueError): prepare._historical_source(self.plan, value)

    def test_all_historical_runtime_files_and_entry_bytes_must_match(self):
        for change in (
            lambda p: p['search_template']['runtime_source_hashes'].pop('v3/rag.py'),
            lambda p: p['search_template']['runtime_source_hashes'].update({'../evil.py': '0'*64}),
            lambda p: p['search_template']['runtime_source_hashes'].update({'budget.py': '0'*64}),
            lambda p: p.update(entry_sha256='0'*64)):
            p = deepcopy(self.plan); change(p)
            with self.assertRaises(ValueError): prepare._historical_source(p, REVISION)


if __name__ == '__main__':
    unittest.main()
