"""Schema-3 orchestration tests with synthetic host receipts, never candidate exec.

SyntheticProgramExecutor drives the real execute/HostBroker with a test sandbox.
It parses only the trusted root wrapper's literal CONFIG as data, simulates its
known final guidance/postprocessing, and never imports or executes archived code.
Actual WSL execution of arbitrary archived programs requires separate integration.
"""
import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from code_rsi import answer_probe as probe
from code_rsi import program_answer_artifacts as artifacts
from code_rsi.budget import digest, save, stable
from code_rsi.v3.execution import execute, root_files, HostError
from code_rsi.v3.infrastructure import UnknownProviderOutcome
from code_rsi.v3.paired_analysis import SPEC_SCHEMA, SINGLE_CONTRAST_SCHEMA
import test_answer_probe as legacy
from test_answer_probe import ScriptedTransport


class SyntheticProgramExecutor:
    def __init__(self, *, prefix_mismatch=False, final_mismatch=False):
        self.calls = []
        self.prefix_mismatch = prefix_mismatch
        self.final_mismatch = final_mismatch

    def __call__(self, archive, node_id, task, backend, model, directory, *, limits):
        node = archive.load_node(node_id)
        archived_files = archive.load_program(node['program_id'])['files']
        self.calls.append({'node_id': node_id, 'program_id': node['program_id'],
                           'files_sha256': digest(archived_files), 'capture': model.capture})
        config = None
        for statement in ast.parse(archived_files['rag.py']).body:
            if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id == 'CONFIG'):
                value = statement.value
                if (not isinstance(value, ast.Call) or len(value.args) != 1
                        or not isinstance(value.args[0], ast.Constant)
                        or not isinstance(value.args[0].value, str)):
                    raise AssertionError('fixture requires trusted literal CONFIG wrapper')
                config = json.loads(value.args[0].value)
        if config is None or archived_files['rag_core.py'] != root_files(config)['rag_core.py']:
            raise AssertionError('fixture only simulates maintained engine sources')
        case = deepcopy(model.case)
        guidance = config.get('prompts', {}).get('answer', '')
        prefix_mismatch, final_mismatch = self.prefix_mismatch, self.final_mismatch
        uppercase = "(result['answer'] or '').upper()" in archived_files['rag.py']
        class SyntheticSandbox:
            def run(self, files, corpus, question, broker, seconds):
                exported = {name: (Path(files) / name).read_text(encoding='utf-8')
                            for name in archived_files}
                if exported != archived_files:
                    raise AssertionError('executor did not receive archived program files')
                for index, event in enumerate(case['events'][:-1]):
                    request = deepcopy(event['request'])
                    if prefix_mismatch and index == 0:
                        if event['name'] == 'complete': request['payload']['unexpected'] = True
                        else: request['query'] += ' changed'
                    broker(event['name'], request)
                payload = deepcopy(case['events'][-1]['request']['payload'])
                payload['additional_guidance'] = guidance
                if final_mismatch: payload['additional_guidance'] += ' unfrozen change'
                response = broker('complete', {'stage': 'answer', 'payload': payload})
                answer = response.get('answer', '') or ''
                if uppercase: answer = answer.upper()
                chosen = response.get('citation_ids', [])
                citations = [deepcopy(row) for row in payload.get('evidence', [])
                             if row['citation_id'] in chosen]
                return {'result': {'answer': answer, 'citations': citations},
                        'runtime': {'isolation_checks': {'synthetic_fixture_only': False}}}
        return execute(archive, node_id, task, backend, model, directory,
                       limits=limits, sandbox=SyntheticSandbox())


class ProgramAnswerProbeTests(unittest.TestCase):
    def setUp(self):
        runs = Path(__file__).parent / 'runs'
        runs.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix='synthetic-program-answer-', dir=runs)
        self.root = Path(temporary.name)
        self.addCleanup(temporary.cleanup)
        guard = patch.object(probe, 'credential_from_plan', side_effect=AssertionError('credential read'))
        guard.start(); self.addCleanup(guard.stop)

    def bind(self, name):
        path = self.root / name
        return {'path': str(path), 'sha256': probe.file_hash(path)}

    def plan(self, *, questions=1, repeats=1, candidates=1, opaque=False,
             same_body=False, postprocess=False):
        plan = legacy.AnswerProbeTests.plan(self, questions=questions, repeats=repeats, opaque=opaque)
        packet = json.loads((self.root / 'cases.json').read_text())
        for case in packet['cases']:
            case['events'][-1]['response']['answer'] = 'HISTORICAL PLACEHOLDER'
            case['events'][-1]['response_sha256'] = digest(case['events'][-1]['response'])
        save(self.root / 'cases.json', packet)
        config = packet['cases'][0]['config']
        programs = [{'name': 'reference', 'files': root_files(config), 'source': None}]
        for index in range(1, candidates + 1):
            changed = deepcopy(config)
            if not same_body and not postprocess:
                changed['prompts'] = {**changed.get('prompts', {}),
                                      'answer': 'Synthetic relational check ' + str(index)}
            files = root_files(changed)
            if postprocess:
                files['rag.py'] = files['rag.py'].replace("result['answer'] or ''", "(result['answer'] or '').upper()")
            elif same_body:
                files['rag.py'] += '\nSYNTHETIC_VARIANT = ' + str(index) + '\n'
            programs.append({'name': 'candidate_' + str(index), 'files': files, 'source': None})
        bundle = {'schema': 'rag-rsi-program-answer-bundle-1', 'source_plan_hash': None,
                  'programs': programs}
        save(self.root / 'programs.json', bundle)
        capture_executor = SyntheticProgramExecutor()
        self.capture_index = getattr(self, 'capture_index', 0) + 1
        projections = artifacts.capture_programs(packet, bundle, plan['model'],
                          self.root / ('captures-' + str(self.capture_index)), executor=capture_executor)
        save(self.root / 'projections.json', projections)
        arms = [row['name'] for row in programs]
        plan.update(schema=probe.PROGRAM_SCHEMA, purpose='fixed_prefix_program_suffix',
                    source_revision=None, programs_file=self.bind('programs.json'),
                    projections_file=self.bind('projections.json'), cases_file=self.bind('cases.json'),
                    arms=[{'name': name} for name in arms],
                    max_calls=questions * repeats * len(arms),
                    helper_source_hashes=probe.helper_hashes(schema=probe.PROGRAM_SCHEMA))
        plan['analysis'].update(schema=SINGLE_CONTRAST_SCHEMA if candidates == 1 else SPEC_SCHEMA,
            comparisons=[{'name': arm + '_vs_reference', 'baseline': 'reference', 'candidate': arm}
                         for arm in arms[1:]])
        return plan

    def generate(self, plan, executor=None, transport=None):
        executor = executor or SyntheticProgramExecutor()
        transport = transport or ScriptedTransport()
        frozen = probe.generate(plan, approved_plan_hash=digest(plan), executor=executor, transport=transport)
        return frozen, executor, transport

    def reject_before_private(self, plan):
        original = probe._verified_bytes
        opened = []
        def checked(item):
            if item == plan['references_file']:
                opened.append(True)
                raise AssertionError('references parsed before complete validated panel')
            return original(item)
        with patch.object(probe, '_verified_bytes', side_effect=checked), self.assertRaises((ValueError, HostError)):
            probe.grade(plan)
        self.assertEqual(opened, [])

    def rebind(self, plan, cell, record, frozen):
        path = Path(plan['output_dir']) / cell['file']
        record['payload_hash'] = digest(record['payload'])
        save(path, record)
        cell['sha256'] = probe.file_hash(path)
        save(Path(plan['output_dir']) / 'generation_freeze.json', frozen)

    def test_two_and_three_arm_preflight_only_reads_public_program_projection_material(self):
        for candidates in (1, 2):
            with self.subTest(candidates=candidates):
                plan = self.plan(questions=2, repeats=2, candidates=candidates, opaque=True)
                original = probe._verified_bytes
                def checked(item):
                    self.assertNotEqual(item, plan['references_file'])
                    return original(item)
                with patch.object(probe, '_verified_bytes', side_effect=checked), \
                     patch.object(probe, 'deepseek_transport', side_effect=AssertionError('transport construction')):
                    check = probe.preflight(plan)
                self.assertEqual(check['max_new_calls'], 4 * (candidates + 1))
                self.assertEqual(check['new_search_calls'], 0)
                self.assertEqual(check['new_read_model_calls'], 0)
                self.assertFalse(check['reference_or_credential_access'])
                self.assertFalse(Path(plan['output_dir']).exists())

    def test_old_schemas_reject_new_program_fields_and_new_schema_rejects_legacy_fields(self):
        plan = self.plan()
        mutations = [lambda p: p.update(schema=probe.SCHEMA, purpose='fixed_evidence_final_judgment_ablation'),
                     lambda p: p.update(schema=probe.CONTEXT_SCHEMA, purpose='fixed_evidence_quote_context'),
                     lambda p: p.update(source_arm='loop'), lambda p: p.update(advance_criteria={}),
                     lambda p: p.update(source_revision='a' * 40), lambda p: p.pop('programs_file'),
                     lambda p: p.pop('projections_file'), lambda p: p['arms'].reverse(),
                     lambda p: p['arms'][1].update(name='other'),
                     lambda p: p['analysis']['comparisons'][0].update(baseline='candidate_1', candidate='reference')]
        for mutate in mutations:
            changed = deepcopy(plan); mutate(changed)
            with self.subTest(keys=sorted(changed)), self.assertRaises(ValueError): probe.preflight(changed)

    def test_plan_cannot_drop_frozen_second_candidate_by_rewriting_budget_and_analysis(self):
        plan = self.plan(candidates=2)
        plan['arms'] = plan['arms'][:2]
        plan['max_calls'] = 2
        plan['analysis']['schema'] = SINGLE_CONTRAST_SCHEMA
        plan['analysis']['comparisons'] = plan['analysis']['comparisons'][:1]
        transport, executor = ScriptedTransport(), SyntheticProgramExecutor()
        with self.assertRaises(ValueError): self.generate(plan, executor, transport)
        self.assertEqual(transport.sent, [])
        self.assertEqual(executor.calls, [])
        self.assertFalse(Path(plan['output_dir']).exists())

    def test_budget_uses_each_frozen_program_answer_body(self):
        plan = self.plan(questions=2, repeats=2, candidates=2)
        check = probe.preflight(plan)
        packet = json.loads((self.root / 'projections.json').read_text())
        shape = probe._RequestShape(plan['model'])
        total = 0
        for row in packet['projections']:
            body = shape.request_body('answer', row['final_payload'])
            size = len(stable(body).encode('utf-8'))
            self.assertEqual(check['answer_body_bytes'][row['case_id']][row['arm']], size)
            self.assertEqual(check['answer_body_hashes'][row['case_id']][row['arm']], digest(body))
            total += plan['repeats'] * ((size + 1024) * 2 + 800 * 8) / 1e6
        self.assertAlmostEqual(check['conservative_cny_upper_bound'], total)
        changed = deepcopy(plan); changed['hard_cny'] = total / 2
        with self.assertRaises(ValueError): probe.preflight(changed)

    def test_missing_projection_or_mutated_program_rejected_before_dispatch(self):
        plan = self.plan()
        for filename, mutation in [('projections.json', lambda x: x['projections'].pop()),
            ('programs.json', lambda x: x['programs'][1]['files'].update({'rag.py': x['programs'][1]['files']['rag.py'] + '\nCHANGED = True\n'}))]:
            path = self.root / filename
            original = path.read_bytes()
            value = json.loads(original); mutation(value); save(path, value)
            changed = deepcopy(plan); changed['projections_file' if filename.startswith('projections') else 'programs_file'] = self.bind(filename)
            transport, executor = ScriptedTransport(), SyntheticProgramExecutor()
            with self.assertRaises(ValueError): self.generate(changed, executor, transport)
            self.assertEqual(transport.sent, []); self.assertEqual(executor.calls, [])
            path.write_bytes(original)

    def test_prefix_and_unfrozen_final_mismatch_dispatch_zero_new_calls(self):
        for option in ('prefix_mismatch', 'final_mismatch'):
            plan = self.plan()
            plan['output_dir'] = str(self.root / ('output-' + option))
            transport = ScriptedTransport()
            with self.subTest(option=option), self.assertRaises((ValueError, HostError)):
                self.generate(plan, SyntheticProgramExecutor(**{option: True}), transport)
            self.assertEqual(transport.sent, [])
            self.assertFalse((Path(plan['output_dir']) / 'generation_freeze.json').exists())

    def test_program_files_reach_executor_and_same_body_arms_have_independent_banks(self):
        plan = self.plan(questions=2, repeats=2, candidates=2, same_body=True, opaque=True)
        frozen, executor, transport = self.generate(plan)
        self.assertEqual(len(frozen['cells']), 12)
        self.assertEqual(len(executor.calls), 12)
        self.assertEqual(len({row['files_sha256'] for row in executor.calls}), 3)
        self.assertTrue(all(row['capture'] is False for row in executor.calls))
        self.assertEqual(len(transport.sent), 12)
        self.assertLess(len({digest(body) for body in transport.sent}), 12)
        records = [json.loads(path.read_text()) for path in (Path(plan['output_dir']) / 'requests').glob('*.json')
                   if path.name != 'returned_model.json']
        self.assertEqual(len({row['key'] for row in records}), 12)
        self.assertFalse(frozen['references_parsed'])

    def test_projection_historical_answer_is_not_scored_as_a_new_measurement(self):
        plan = self.plan()
        self.reject_before_private(plan)
        self.generate(plan)
        report = probe.grade(plan)
        self.assertTrue(report['references_parsed_after_complete_generation'])
        self.assertFalse(report['independent_quality_evidence'])
        self.assertTrue(all(row['mean_em'] == 1 for row in report['summary'].values()))
        self.assertTrue(all(row['answer_sha256'] == digest('Beacon Port') for row in report['rows']))

    def test_complete_resume_and_freeze_interruption_never_repurchase_answers(self):
        plan = self.plan(opaque=True)
        original = probe.freeze
        def interrupt(path, value):
            if Path(path).name == 'generation_freeze.json': raise OSError('synthetic freeze interruption')
            return original(path, value)
        executor, transport = SyntheticProgramExecutor(), ScriptedTransport()
        with patch.object(probe, 'freeze', side_effect=interrupt), self.assertRaises(OSError):
            self.generate(plan, executor, transport)
        self.assertEqual(len(transport.sent), 2)
        self.reject_before_private(plan)
        frozen, _, _ = self.generate(plan, executor, transport)
        self.assertEqual(len(transport.sent), 2)
        self.assertEqual(len(executor.calls), 2)
        with patch.object(probe, 'deepseek_transport', side_effect=AssertionError('transport construction')):
            self.assertEqual(probe.generate(plan, approved_plan_hash=digest(plan), executor=executor), frozen)
        self.assertEqual(len(executor.calls), 2)

    def test_unknown_dispatch_stops_and_recovery_does_not_send_again(self):
        plan = self.plan()
        sent = []
        def unknown(body):
            sent.append(deepcopy(body)); raise TimeoutError('synthetic unknown provider outcome')
        for attempt in range(2):
            with self.subTest(attempt=attempt), self.assertRaises(UnknownProviderOutcome):
                self.generate(plan, transport=unknown)
        self.assertEqual(len(sent), 1)
        self.assertFalse((Path(plan['output_dir']) / 'generation_freeze.json').exists())
        self.reject_before_private(plan)

    def test_missing_duplicate_or_foreign_cells_rejected_before_private_read(self):
        plan = self.plan(candidates=2)
        frozen, _, _ = self.generate(plan)
        variants = [frozen['cells'][:-1], [frozen['cells'][0]] * 3]
        wrong = deepcopy(frozen['cells']); wrong[0]['identity']['arm'] = 'unknown'; variants.append(wrong)
        for cells in variants:
            save(Path(plan['output_dir']) / 'generation_freeze.json', {**frozen, 'cells': cells})
            self.reject_before_private(plan)

    def test_candidate_postprocessing_executes_and_invalidates_even_normalized_correct_answer(self):
        plan = self.plan(postprocess=True)
        self.generate(plan)
        report = probe.grade(plan)
        self.assertEqual(report['status'], 'protocol_invalid')
        child = next(row for row in report['rows'] if row['arm'] == 'candidate_1')
        self.assertEqual(child['answer_sha256'], digest('BEACON PORT'))
        self.assertFalse(child['answer_origin_valid'])
        self.assertIsNone(child['metrics'])
        self.assertIsNone(report['summary']['reference']['mean_f1'])
        self.assertIsNone(report['summary']['candidate_1']['mean_f1'])

    def test_rehashed_forged_answer_origin_or_final_trace_rejected_before_private_read(self):
        plan = self.plan()
        frozen, _, _ = self.generate(plan)
        cell = frozen['cells'][0]
        record = json.loads((Path(plan['output_dir']) / cell['file']).read_text())
        record['payload']['answer'] = 'forged returned value'
        record['payload']['answer_origin_valid'] = True
        self.rebind(plan, cell, record, frozen)
        self.reject_before_private(plan)

    def test_exact_approved_plan_required_before_output_or_transport(self):
        plan = self.plan()
        executor, transport = SyntheticProgramExecutor(), ScriptedTransport()
        with self.assertRaises(ValueError):
            probe.generate(plan, approved_plan_hash='wrong', executor=executor, transport=transport)
        self.assertEqual(executor.calls, []); self.assertEqual(transport.sent, [])
        self.assertFalse(Path(plan['output_dir']).exists())


if __name__ == '__main__':
    unittest.main()
