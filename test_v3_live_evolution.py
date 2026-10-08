"""Synthetic entrypoint boundary tests; no network, credentials or candidate execution."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from code_rsi import live_evolution as live
from code_rsi.budget import Ledger, digest, save
from code_rsi.v3 import execution
from code_rsi.v3.datasets import adapt_multihop
from code_rsi.v3.infrastructure import PROMPTS, StructuredModel, UnknownProviderOutcome


class FixtureTransport:
    def __init__(self, *, unknown=False):
        self.requests = []
        self.unknown = unknown

    def send(self, body, timeout):
        stage = next(k for k, v in PROMPTS.items() if v == body['messages'][0]['content'])
        payload = json.loads(body['messages'][1]['content'])
        self.requests.append((stage, payload))
        if self.unknown:
            raise TimeoutError('synthetic unknown physical result')
        if stage == 'develop':
            value = {'writes': {'rag_core.py': payload['source_files']['rag_core.py'] + '\nSYNTHETIC_EDIT = 1\n'},
                     'mechanism': 'synthetic structural fixture only',
                     'target_module': payload['decision']['target_module']}
        else:
            value = {'answer': 'candidate guess', 'citation_ids': [], 'evidence_sufficient': False}
        return {'model': 'synthetic-not-a-provider',
                'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 100}}


def fake_execute(archive, node_id, task, backend, model, directory, *, limits=None, nonce='first'):
    """Real host Measurement/runner/cache with a synthetic receipt in place of WSL."""
    broker = execution.HostBroker(task, backend, model, **(limits or {}))
    response = broker('complete', {'stage': 'answer', 'payload': {
        'question': task['question'], 'nonce': [nonce, node_id], 'evidence': []}})
    origin = broker.answer_origin_receipt(response['answer'])
    citation = broker.citation_receipt(response['answer'], [])
    node = archive.load_node(node_id)
    return {'schema': execution.EXECUTION_SCHEMA, 'node_id': node_id,
            'program_id': node['program_id'], 'question_id': task['question_id'],
            'answer': response['answer'], 'answer_usable': bool(response['answer'].strip()),
            'execution_ok': True, 'answer_origin_valid': origin['valid'],
            'answer_origin_status': origin['status'], 'host_answer_origin_validation': origin,
            'citation_source_valid': citation['valid'], 'citation_status': citation['status'],
            'host_citation_validation': citation, 'citations': [], 'failure_classes': [],
            'model_errors': broker.model_errors, 'trace': broker.events,
            'host_evidence_trace': {'read_presentations': broker.read_presentations,
                                    'final_observations': broker.final_observations},
            'candidate_reported': None, 'resource_usage': broker.counts}


class LiveEvolutionTests(unittest.TestCase):
    def setUp(self):
        runs = live.PROJECT / 'runs'
        runs.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='v3_live_entry_test_', dir=runs)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.plan = self.make_plan()

    def binding(self, path, value):
        save(path, value)
        return {'path': str(path), 'sha256': live._hash(path)}

    def make_plan(self):
        panels = {}
        for role in live.ROLES:
            task, ref = adapt_multihop({'id': role, 'query': 'Synthetic public question ' + role,
                                       'answer': 'PRIVATE_GOLD_' + role})
            task['documents'] = [{'docid': 'doc-' + role, 'text': 'Synthetic public document ' + role}]
            panels[role] = {
                'tasks_file': self.binding(self.base / (role + '-tasks.json'), [task]),
                'references_file': self.binding(self.base / (role + '-refs.json'), {task['question_id']: ref})}
        return {'schema': live.SCHEMA, 'purpose': 'development_evolution',
                'output_dir': str(self.base / 'execution'), 'panels': panels,
                'corpus': None, 'corpus_ref': None,
                'model': {'name': 'deepseek-flash', 'temperature': 0, 'thinking': 'disabled',
                          'max_input_bytes': 120000,
                          'output_limits': {'plan': 1200, 'read': 2200, 'answer': 800, 'develop': 18000},
                          'prices': {'input_hit': .04, 'input_miss': 2, 'output': 8}},
                'root_config': {'mode': 'iterative', 'max_model_calls': 7, 'max_rounds': 3},
                'limits': {'max_models': 7, 'max_searches': 8, 'max_reads': 8},
                'repeats': 1, 'expansions': 1, 'select_candidates': 2, 'metric': 'em',
                'allow_proxy_metric': True, 'synthetic': True, 'max_calls': 43, 'hard_cny': 15,
                'runtime_source_hashes': live._runtime_source_hashes(), 'entry_sha256': live._hash(live.__file__),
                'credential_source': {'kind': 'env_file', 'variable': 'DEEPSEEK_API_KEY',
                                      'path': str(self.base / 'MUST_NOT_READ.env')}}

    def rewrite_role(self, role, kind, value):
        item = self.plan['panels'][role][kind]
        self.plan['panels'][role][kind] = self.binding(Path(item['path']), value)

    def read_role(self, role, kind):
        return json.loads(Path(self.plan['panels'][role][kind]['path']).read_bytes())

    def run_fixture(self, transport, **kwargs):
        with patch.object(execution, 'execute', side_effect=fake_execute):
            return live.run(self.plan, approved_plan_hash=digest(self.plan), execute=True,
                            transport_factory=lambda: transport, **kwargs)

    def test_preflight_derived_budget_no_writes_credentials_or_transport(self):
        before = sorted(p.relative_to(self.base).as_posix() for p in self.base.rglob('*'))
        with patch.object(live, 'credential_from_plan', side_effect=AssertionError('credential access')), \
             patch.object(live, 'deepseek_transport', side_effect=AssertionError('transport access')):
            report = live.preflight(self.plan)
        self.assertEqual(report['max_calls'], 43)
        self.assertEqual(report['qa_call_ceiling'], 42)
        self.assertEqual(report['developer_call_ceiling'], 1)
        expected = ((120000 + 1024) * 43 * 2 + (42 * 2200 + 18000) * 8) / 1e6
        self.assertAlmostEqual(report['conservative_cny_upper_bound'], expected, places=6)
        self.assertFalse(report['credentials_read'])
        self.assertEqual(before, sorted(p.relative_to(self.base).as_posix() for p in self.base.rglob('*')))
        self.assertNotIn('code_rsi/live_evolution.py', self.plan['runtime_source_hashes'])
        self.assertTrue(report['private_references_read_locally'])

    def test_complete_budget_not_just_expansions(self):
        for count in (1, 42, 44):
            with self.subTest(count=count):
                plan = deepcopy(self.plan); plan['max_calls'] = count
                with self.assertRaisesRegex(ValueError, 'structural ceiling'):
                    live.preflight(plan)
        self.plan['hard_cny'] = 1
        with self.assertRaisesRegex(ValueError, 'conservative'):
            live.preflight(self.plan)

    def test_developer_output_has_its_own_cap(self):
        a = live.preflight(self.plan)
        self.plan['model']['output_limits']['develop'] += 1000
        b = live.preflight(self.plan)
        self.assertAlmostEqual(b['conservative_cny_upper_bound'] - a['conservative_cny_upper_bound'], .008)
        self.assertNotEqual(a['model_identity'], b['model_identity'])

    def test_root_rounds_and_final_fit_both_model_budgets(self):
        for key in ('host', 'rag'):
            with self.subTest(key=key):
                plan = deepcopy(self.plan)
                if key == 'host':
                    plan['limits']['max_models'] = 4
                else:
                    plan['root_config']['max_model_calls'] = 4
                with self.assertRaisesRegex(ValueError, 'rounds and final'):
                    live.preflight(plan)
        self.plan['root_config'].update(mode='single_pass', max_rounds=64, max_model_calls=2)
        self.assertEqual(live.preflight(self.plan)['max_calls'], 43)

    def test_root_retrieval_profile_fits_host(self):
        for problem in ('limit','budget'):
            plan=deepcopy(self.plan)
            if problem=='limit':plan['root_config']['search_limit']=31
            else:plan['limits']['max_searches']=1
            with self.subTest(problem=problem),self.assertRaisesRegex(ValueError,'search'):
                live.preflight(plan)

    def test_planned_single_requires_exactly_three_root_model_calls(self):
        # max_rounds does not create extra reads in planned_single mode. Its
        # planner, one read and final answer must fit both independent budgets.
        for rounds in (1, 64):
            with self.subTest(max_rounds=rounds):
                plan = deepcopy(self.plan)
                plan['root_config'].update(mode='planned_single', max_rounds=rounds, max_model_calls=3)
                plan['limits']['max_models'] = 3
                plan['max_calls'] = 19  # Six QA outcomes * three calls + one developer call.
                report = live.preflight(plan)
                self.assertEqual(report['qa_call_ceiling'], 18)
                self.assertEqual(report['max_calls'], 19)
                for budget in ('host', 'rag'):
                    insufficient = deepcopy(plan)
                    if budget == 'host':
                        insufficient['limits']['max_models'] = 2
                    else:
                        insufficient['root_config']['max_model_calls'] = 2
                    with self.subTest(budget=budget), self.assertRaises(ValueError):
                        live.preflight(insufficient)

    def test_exact_approval_and_execute_before_output_creation(self):
        for execute, approved in ((False, digest(self.plan)), (True, 'wrong')):
            with self.subTest(execute=execute, approved=approved):
                with self.assertRaisesRegex(ValueError, 'exact approved'):
                    live.run(self.plan, approved_plan_hash=approved, execute=execute,
                             transport_factory=lambda: FixtureTransport())
                self.assertFalse(Path(self.plan['output_dir']).exists())

    def test_synthetic_never_falls_through_to_real_transport(self):
        with patch.object(live, 'credential_from_plan', side_effect=AssertionError('credential access')):
            with self.assertRaisesRegex(ValueError, 'injected transport'):
                live.run(self.plan, approved_plan_hash=digest(self.plan), execute=True)
        self.plan['synthetic'] = False
        with self.assertRaisesRegex(ValueError, 'forbid injection'):
            self.run_fixture(FixtureTransport())
        self.assertFalse(Path(self.plan['output_dir']).exists())

    def test_questions_cannot_cross_roles(self):
        tasks = self.read_role('D_select', 'tasks_file')
        tasks[0]['question'] = '  SYNTHETIC PUBLIC QUESTION D_fit  '
        self.rewrite_role('D_select', 'tasks_file', tasks)
        with self.assertRaisesRegex(ValueError, 'overlap across roles'):
            live.preflight(self.plan)

    def test_private_source_groups_cannot_cross_roles(self):
        for role in ('D_fit', 'D_report'):
            refs = self.read_role(role, 'references_file')
            next(iter(refs.values()))['source_question_id'] = 'same-family'
            self.rewrite_role(role, 'references_file', refs)
        with self.assertRaisesRegex(ValueError, 'group crosses roles'):
            live.preflight(self.plan)

    def test_missing_gold_is_not_scored_as_zero(self):
        refs = self.read_role('D_report', 'references_file')
        next(iter(refs.values()))['reference_available'] = False
        self.rewrite_role('D_report', 'references_file', refs)
        with self.assertRaisesRegex(ValueError, 'available answer references'):
            live.preflight(self.plan)

    def test_proxy_acknowledgement_is_explicit(self):
        self.plan['allow_proxy_metric'] = False
        with self.assertRaisesRegex(ValueError, 'proxy acknowledgement'):
            live.preflight(self.plan)

    def test_changed_input_or_source_rejected(self):
        for kind in ('entry', 'runtime', 'task'):
            with self.subTest(kind=kind):
                plan = deepcopy(self.plan)
                if kind == 'entry': plan['entry_sha256'] = 'changed'
                elif kind == 'runtime': plan['runtime_source_hashes'] = {}
                else: plan['panels']['D_fit']['tasks_file']['sha256'] = 'changed'
                with self.assertRaises(ValueError): live.preflight(plan)

    def bound(self, allowed, check=lambda: None):
        transport = FixtureTransport()
        model = StructuredModel(self.base / 'standalone-requests',
                                Ledger(self.base / 'standalone-ledger.jsonl', {'run': {'calls': 5, 'cny': 15}}),
                                transport, bank='unit', prices=self.plan['model']['prices'])
        wrapper = live._BoundModel(model, live._model_identity(self.plan['model'], PROMPTS), allowed, check)
        return wrapper, model, transport

    def test_model_roles_are_capabilities(self):
        for allowed, forbidden in ((('plan', 'read', 'answer'), 'develop'), (('develop',), 'answer')):
            wrapper, model, transport = self.bound(allowed)
            with self.assertRaisesRegex(execution.HostError, 'role cannot'):
                wrapper.complete(forbidden, {})
            self.assertEqual(transport.requests, [])

    def test_identity_drift_cannot_dispatch(self):
        for kind in ('model', 'limits', 'prices', 'prompts', 'source'):
            with self.subTest(kind=kind):
                wrapper, model, transport = self.bound(('answer',))
                if kind == 'model': model.model = 'different-model'
                elif kind == 'limits': model.limits['answer'] += 1
                elif kind == 'prices': model.prices['output'] += 1
                elif kind == 'source': wrapper._check = lambda: (_ for _ in ()).throw(execution.HostError('source drift'))
                with patch.dict(PROMPTS, {'answer': PROMPTS['answer'] + (' changed' if kind == 'prompts' else '')}):
                    with self.assertRaises(execution.HostError): wrapper.complete('answer', {})
                self.assertEqual(transport.requests, [])

    def test_run_and_resume_shared_ledger_and_fit_only_developer(self):
        transport = FixtureTransport()
        with patch.object(live, 'credential_from_plan', side_effect=AssertionError('credential access')):
            result = self.run_fixture(transport)
            calls = len(transport.requests)
            resumed = self.run_fixture(transport)
        self.assertEqual(result, resumed)
        self.assertEqual(len(transport.requests), calls)
        self.assertEqual(result['fit_nodes'], 2)
        self.assertTrue(result['synthetic'])
        self.assertFalse(result['report_used_for_decisions'])
        develops = [p for stage, p in transport.requests if stage == 'develop']
        self.assertEqual(len(develops), 1)
        payload = json.dumps(develops[0])
        self.assertIn('Synthetic public question D_fit', payload)
        self.assertNotIn('Synthetic public question D_select', payload)
        self.assertNotIn('Synthetic public question D_report', payload)
        self.assertNotIn('PRIVATE_GOLD_', payload)
        status = json.loads((Path(self.plan['output_dir']) / 'live_status.json').read_bytes())
        self.assertEqual(status['ledger']['used']['run']['calls'], calls)
        self.assertEqual(status['ledger']['pending'], 0)
        self.assertGreater(calls, 1)
        self.assertLessEqual(calls, self.plan['max_calls'])

    def test_unknown_request_stops_resume_even_with_a_changed_payload(self):
        transport = FixtureTransport(unknown=True)
        with self.assertRaises(UnknownProviderOutcome): self.run_fixture(transport)
        out = Path(self.plan['output_dir'])
        ledger = Ledger(out / 'ledger.jsonl', {'run': {'calls': 43, 'cny': 15}})
        self.assertEqual(ledger.summary()['pending'], 0)
        self.assertEqual(ledger.summary()['used']['run']['calls'], 1)
        constructed = []
        def factory():
            constructed.append(True)
            return FixtureTransport()
        with patch.object(execution, 'execute', side_effect=lambda *a, **k: fake_execute(*a, **k, nonce='changed')):
            with self.assertRaises(UnknownProviderOutcome):
                live.run(self.plan, approved_plan_hash=digest(self.plan), execute=True, transport_factory=factory)
        self.assertEqual(constructed, [])
        self.assertEqual(len(transport.requests), 1)
        self.assertFalse(list(out.glob('measurements/**/measured.json')))

    def test_unsettled_cache_blocks_without_a_ledger(self):
        for state in ('pending', 'response_received'):
            with self.subTest(state=state):
                out = Path(self.plan['output_dir'])
                save(out / 'requests' / 'old-key.json', {'key': 'old-key', 'state': state})
                with patch.object(live, 'credential_from_plan', side_effect=AssertionError('credential access')):
                    with self.assertRaises(UnknownProviderOutcome): self.run_fixture(FixtureTransport())
                self.assertFalse((out / 'ledger.jsonl').exists())

    def test_settled_cache_without_ledger_is_not_a_fresh_budget(self):
        out = Path(self.plan['output_dir'])
        save(out / 'requests' / 'old-key.json', {'key': 'old-key', 'state': 'settled'})
        with self.assertRaisesRegex(execution.HostError, 'no complete ledger'):
            self.run_fixture(FixtureTransport())

    def test_empty_or_truncated_ledger_blocks_cached_requests(self):
        transport = FixtureTransport()
        self.run_fixture(transport)
        out = Path(self.plan['output_dir'])
        original = (out / 'ledger.jsonl').read_text(encoding='utf-8')
        for altered in ('', '\n'.join(original.splitlines()[2:]) + '\n'):
            with self.subTest(empty=not altered):
                (out / 'ledger.jsonl').write_text(altered, encoding='utf-8')
                with self.assertRaisesRegex(execution.HostError, 'complete ledger differ'):
                    self.run_fixture(transport)
        self.assertEqual(len(transport.requests), 6)

    def test_cli_has_no_transport_injection_path(self):
        with patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                live.main(['run', '--plan', 'not-opened', '--execute', '--approved-plan-hash', 'x',
                           '--transport-factory', 'arbitrary.module'])


if __name__ == '__main__':
    unittest.main()
