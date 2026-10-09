"""Synthetic integration counterexamples for the opt-in runtime capability card.

No external API, credential, real question, or generated-candidate execution.
Only the execution boundary is replaced; plans, gates, ledgers and recovery are real.
"""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from code_rsi import live_evolution as live, paired_evolution as paired
from code_rsi.budget import Ledger, digest, save, stable
from code_rsi.v3 import evolution, execution
from code_rsi.v3.infrastructure import (
    PROMPTS, ModelResponseError, StructuredModel, UnknownProviderOutcome,
)
from code_rsi.v3.runtime_contract import build_runtime_contract
import test_v3_proposal_protocol_entry as exact_fixture
import test_v3_paired_entry as paired_fixture


class DuplicateEditTransport(exact_fixture.ExactTransport):
    """Return a well-formed JSON object with a duplicate key that json.loads loses."""
    def __init__(self, nested=False):
        super().__init__()
        self.nested = nested

    def send(self, body, timeout):
        response = super().send(body, timeout)
        payload = json.loads(body['messages'][-1]['content'])
        if 'source_files' in payload:
            message = response['choices'][0]['message']
            text = message['content']
            if self.nested:
                text = text.replace('"new":', '"old": "SYNTHETIC_DUPLICATE_ANCHOR", "new":', 1)
            else:
                text = text[:-1] + ', "mechanism": "Synthetic repeated mechanism"}'
            message['content'] = text
        return response


class UnknownDevelopTransport(exact_fixture.ExactTransport):
    def send(self, body, timeout):
        payload = json.loads(body['messages'][-1]['content'])
        if 'source_files' in payload:
            self.bodies.append(deepcopy(body))
            self.requests.append(('develop', payload))
            raise TimeoutError('synthetic unknown developer outcome')
        return super().send(body, timeout)


class RuntimeContractEntryTests(unittest.TestCase):
    binding = paired_fixture.PairedEntryTests.binding
    make_plan = paired_fixture.PairedEntryTests.make_plan
    read_role = paired_fixture.PairedEntryTests.read_role
    rewrite_role = paired_fixture.PairedEntryTests.rewrite_role
    staged = paired_fixture.PairedEntryTests.staged
    run_fixture = paired_fixture.PairedEntryTests.run_fixture

    def setUp(self):
        exact_fixture.PairedProtocolEntryTests.setUp(self)
        self.plan['schema'] = live.SCHEMA6
        self.pair['schema'] = paired.SCHEMA3
        self.refresh_card()

    def refresh_card(self):
        self.plan['runtime_contract'] = build_runtime_contract(
            self.plan['limits'], self.plan['model'], self.plan['proposal_protocol'], PROMPTS)
        return self.plan['runtime_contract']

    def old_version(self):
        self.plan['schema'] = live.SCHEMA5
        self.plan.pop('runtime_contract')
        self.pair['schema'] = paired.SCHEMA2

    def artifact(self, relative):
        return json.loads((self.out / relative).read_text(encoding='utf-8'))

    def model(self, *, with_card=True, bank='unit', transport=None):
        config = self.plan['model']
        ledger = Ledger(self.base / 'unit-ledger.jsonl', {'run': {'calls': 20, 'cny': 20}})
        def forbidden(body):
            raise AssertionError('unexpected transport')
        return StructuredModel(self.base / 'unit-requests', ledger,
            forbidden if transport is None else transport, bank=bank,
            prices=config['prices'], model=config['name'],
            max_input_bytes=config['max_input_bytes'], limits=config['output_limits'],
            proposal_protocol=self.plan['proposal_protocol'],
            runtime_contract=self.plan.get('runtime_contract') if with_card else None)

    def test_preflight_binds_full_card_without_private_parse_writes_or_calls(self):
        before = sorted(str(p) for p in self.base.rglob('*'))
        reference = self.plan['panels']['D_fit']['references_file']
        parsed = []
        original = live._file
        def observe(item, *, parse=False):
            if item == reference and parse:
                parsed.append(item)
            return original(item, parse=parse)
        with patch.object(live, '_file', side_effect=observe), \
             patch.object(live, 'credential_from_plan', side_effect=AssertionError('credential')), \
             patch.object(live, 'deepseek_transport', side_effect=AssertionError('network')):
            check = live.preflight(self.plan, phase='search')
            report = paired.preflight(self.pair)
        self.assertEqual(check['runtime_contract'], self.plan['runtime_contract'])
        self.assertEqual(report['schema'], paired.SCHEMA3)
        self.assertEqual(report['proposal_opportunities'], 4)
        self.assertEqual(report['max_calls'], 39)
        self.assertEqual(parsed, [])
        self.assertEqual(before, sorted(str(p) for p in self.base.rglob('*')))
        check['runtime_contract']['host_limits']['max_reads'] += 1
        self.assertNotEqual(check['runtime_contract'], self.plan['runtime_contract'])

    def test_schema6_requires_card_and_old_schema_exact_fields_exclude_it(self):
        for card in (None, {}, {'schema': 'not-a-runtime-contract'}):
            changed = deepcopy(self.plan); changed['runtime_contract'] = card
            with self.subTest(card=card), self.assertRaises(ValueError):
                live.preflight(changed, phase='search')
        changed = deepcopy(self.plan); changed.pop('runtime_contract')
        with self.assertRaises(ValueError):
            live.preflight(changed, phase='search')
        for schema, fields, phase in (
            (live.SCHEMA, live.FIELDS, None),
            (live.SCHEMA2, live.FIELDS_V2, None),
            (live.SCHEMA3, live.FIELDS_V3, 'search'),
            (live.SCHEMA4, live.FIELDS_V4, 'search'),
            (live.SCHEMA5, live.FIELDS_V5, 'search'),
        ):
            changed = {k: deepcopy(v) for k, v in self.plan.items() if k in fields}
            changed['schema'] = schema
            changed['runtime_contract'] = deepcopy(self.plan['runtime_contract'])
            with self.subTest(schema=schema), self.assertRaisesRegex(ValueError, 'exact'):
                live.preflight(changed, phase=phase)
        self.assertFalse(self.out.exists())

    def test_paired3_cannot_silently_map_to_legacy_template_or_vice_versa(self):
        for outer, inner in ((paired.SCHEMA3, live.SCHEMA5),
                             (paired.SCHEMA2, live.SCHEMA6),
                             (paired.SCHEMA, live.SCHEMA6)):
            plan = deepcopy(self.pair)
            plan['schema'] = outer; plan['search_template']['schema'] = inner
            if inner == live.SCHEMA5:
                plan['search_template'].pop('runtime_contract')
            with self.subTest(outer=outer, inner=inner), self.assertRaises(ValueError):
                paired.preflight(plan)
        self.assertFalse(self.out.exists())

    def test_self_consistent_but_stale_card_cannot_misdescribe_active_dependencies(self):
        for section, key in (('limits', 'max_reads'), ('model', 'max_input_bytes'),
                             ('proposal_protocol', 'max_edits')):
            plan = deepcopy(self.plan)
            plan[section][key] += -1 if section == 'model' else 1
            with self.subTest(section=section), self.assertRaisesRegex(ValueError, 'runtime|contract'):
                live.preflight(plan, phase='search')
        changed = deepcopy(self.plan)
        changed['model']['output_limits']['answer'] += 1
        with self.assertRaisesRegex(ValueError, 'runtime|contract'):
            live.preflight(changed, phase='search')
        with patch.dict(PROMPTS, {'read': PROMPTS['read'] + ' Synthetic drift.'}):
            with self.assertRaisesRegex(ValueError, 'runtime|contract'):
                live.preflight(self.plan, phase='search')
        self.assertFalse(self.out.exists())

    def test_opt_in_changes_develop_and_identity_but_preserves_qa_wire_bytes(self):
        legacy = self.model(with_card=False, bank='legacy-shape')
        current = self.model(bank='new-shape')
        self.assertNotEqual(legacy.identity, current.identity)
        payload = {'question': 'Synthetic input only', 'evidence': []}
        for stage in ('plan', 'read', 'answer'):
            with self.subTest(stage=stage):
                self.assertEqual(stable(legacy.request_body(stage, payload)),
                                 stable(current.request_body(stage, payload)))
        old_payload = {'proposal_protocol': deepcopy(exact_fixture.EXACT)}
        new_payload = {**old_payload, 'runtime_contract': deepcopy(self.plan['runtime_contract'])}
        self.assertNotEqual(legacy.request_body('develop', old_payload),
                            current.request_body('develop', new_payload))
        with self.assertRaises(ValueError):
            current.request_body('develop', old_payload)
        with self.assertRaises(ValueError):
            legacy.request_body('develop', new_payload)
        self.assertEqual((legacy.calls, current.calls), (0, 0))

    def test_full_paired_run_freezes_identical_card_actual_bodies_and_resumes_free(self):
        transport = exact_fixture.PairedSequenceTransport()
        first, executions = self.run_fixture(transport)
        self.assertEqual(first['status'], 'complete')
        self.assertEqual(first['schema'], paired.SCHEMA3)
        self.assertEqual((executions, len(transport.requests)), (3, 7))
        self.assertEqual(first['ledger']['used']['run']['calls'], 7)
        bodies = {}
        for condition in ('cases', 'trace'):
            manifest = self.artifact('blocks/0/' + condition + '/manifest.json')
            self.assertEqual(manifest['runtime_contract'], self.plan['runtime_contract'])
            self.assertEqual(manifest['developer_configuration']['runtime_contract'], self.plan['runtime_contract'])
            self.assertEqual(manifest['model_identity'], live.preflight(self.plan, phase='search')['model_identity'])
            body = self.artifact('blocks/0/feedback/' + condition + '.json')
            payload = json.loads(body['messages'][-1]['content']); bodies[condition] = payload
            self.assertEqual(payload['runtime_contract'], self.plan['runtime_contract'])
            self.assertEqual(len(payload['feedback']['cases']), 1)
            actual = [b for b in transport.bodies
                      if 'source_files' in json.loads(b['messages'][-1]['content'])
                      and json.loads(b['messages'][-1]['content'])['feedback']['condition'] == condition]
            self.assertEqual(len(actual), 2)
            self.assertTrue(all(b == body for b in actual))
        self.assertEqual(bodies['cases']['runtime_contract'], bodies['trace']['runtime_contract'])
        self.assertIn('execution_flow', bodies['trace']['feedback']['cases'][0])
        self.assertNotIn('execution_flow', bodies['cases']['feedback']['cases'][0])
        with patch.object(live, 'credential_from_plan', side_effect=AssertionError('credential')):
            second, executions = self.run_fixture(transport_factory=lambda: (_ for _ in ()).throw(AssertionError('resume transport')))
        self.assertEqual(first, second)
        self.assertEqual((executions, len(transport.requests)), (0, 7))

    def test_card_counts_against_joint_gate_and_trace_overflow_blocks_both_proposals(self):
        # First obtain actual frozen wire sizes, then construct a separate run whose
        # budget fits cases but cannot fit complete trace plus the same full card.
        transport = exact_fixture.PairedSequenceTransport()
        self.run_fixture(transport, stop_after=1)
        sizes = {c: len(stable(self.artifact('blocks/0/feedback/' + c + '.json')).encode('utf-8'))
                 for c in ('cases', 'trace')}
        self.assertGreater(sizes['trace'] - sizes['cases'], 100)
        self.plan['model']['max_input_bytes'] = (sizes['cases'] + sizes['trace']) // 2
        self.refresh_card()
        self.out = self.base / 'joint-overflow'
        self.pair['output_dir'] = str(self.out)
        self.plan['output_dir'] = str(self.out / 'template_validation_only')
        second = exact_fixture.PairedSequenceTransport()
        with patch.object(evolution, '_fit_development_request', side_effect=AssertionError('asymmetric crop')):
            with self.assertRaises(ValueError):
                self.run_fixture(second)
        self.assertEqual([stage for stage, _ in second.requests], ['answer'])
        self.assertFalse((self.out / 'blocks/0/feedback_gate.json').exists())
        for condition in ('cases', 'trace'):
            self.assertFalse((self.out / 'blocks/0' / condition / 'steps/0/received_proposal.json').exists())

    def test_different_card_in_one_arm_rejected_before_any_develop_dispatch(self):
        original = evolution.ProgramDeveloper.prepare_request
        def changed(developer, *args, **kwargs):
            payload = original(developer, *args, **kwargs)
            if developer.feedback_condition == 'trace':
                payload['runtime_contract']['host_limits']['max_reads'] += 1
            return payload
        transport = exact_fixture.PairedSequenceTransport()
        with patch.object(evolution.ProgramDeveloper, 'prepare_request', autospec=True, side_effect=changed):
            with self.assertRaises((ValueError, execution.HostError)):
                self.run_fixture(transport)
        self.assertEqual([s for s, _ in transport.requests], ['answer'])
        self.assertFalse((self.out / 'blocks/0/feedback_gate.json').exists())

    def test_resume_rejects_changed_card_even_if_new_card_matches_new_limits(self):
        transport = exact_fixture.PairedSequenceTransport()
        self.run_fixture(transport, stop_after=1)
        before = len(transport.requests)
        self.plan['limits']['max_reads'] += 1
        self.refresh_card()
        with self.assertRaises((ValueError, RuntimeError)):
            self.run_fixture(transport_factory=lambda: (_ for _ in ()).throw(AssertionError('tamper transport')))
        self.assertEqual(len(transport.requests), before)

    def test_saved_body_card_tamper_is_detected_without_dispatch(self):
        transport = exact_fixture.PairedSequenceTransport()
        self.run_fixture(transport, stop_after=1)
        path = self.out / 'blocks/0/feedback/trace.json'
        body = json.loads(path.read_text(encoding='utf-8'))
        payload = json.loads(body['messages'][-1]['content'])
        payload['runtime_contract']['host_limits']['max_reads'] += 1
        body['messages'][-1]['content'] = stable(payload)
        save(path, body)
        before = len(transport.requests)
        with self.assertRaises((ValueError, RuntimeError)):
            self.run_fixture(transport_factory=lambda: (_ for _ in ()).throw(AssertionError('tamper transport')))
        self.assertEqual(len(transport.requests), before)

    def test_duplicate_keys_are_rejected_counted_and_not_repurchased_in_schema6(self):
        for nested in (False, True):
            with self.subTest(nested=nested):
                self.out = self.base / ('duplicates-' + str(nested))
                self.pair['output_dir'] = str(self.out)
                self.plan['output_dir'] = str(self.out / 'template_validation_only')
                transport = DuplicateEditTransport(nested=nested)
                result, executions = self.run_fixture(transport)
                self.assertEqual((executions, len(transport.requests)), (1, 5))
                self.assertEqual(len(result['terminal_records']), 4)
                self.assertTrue(all(r['attempt']['status'] == 'rejected' for r in result['terminal_records']))
                self.assertEqual(result['ledger']['used']['run']['calls'], 5)
                resumed, count = self.run_fixture(transport_factory=lambda: (_ for _ in ()).throw(AssertionError('duplicate retry')))
                self.assertEqual(resumed, result)
                self.assertEqual(count, 0)
                self.assertEqual(len(transport.requests), 5)

    def test_old_schema_duplicate_parsing_is_not_retroactively_reclassified(self):
        self.old_version()
        transport = DuplicateEditTransport()
        result, executions = self.run_fixture(transport)
        self.assertEqual((executions, len(transport.requests)), (3, 7))
        self.assertEqual(sum(r['attempt']['status'] == 'measured' for r in result['terminal_records']), 2)
        for _, payload in transport.requests:
            self.assertNotIn('runtime_contract', payload)

    def test_strict_duplicate_parser_is_develop_only_and_cache_keeps_single_charge(self):
        sent = []
        def duplicate_response(body):
            sent.append(deepcopy(body))
            return {'choices': [{'finish_reason': 'stop', 'message': {'content': '{"value":1,"value":2}'}}],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 10}}
        model = self.model(transport=duplicate_response)
        self.assertEqual(model.complete('answer', {}), {'value': 2})
        payload = {'proposal_protocol': deepcopy(exact_fixture.EXACT),
                   'runtime_contract': deepcopy(self.plan['runtime_contract'])}
        for _ in range(2):
            with self.assertRaises(ModelResponseError):
                model.complete('develop', payload)
        self.assertEqual(len(sent), 2)  # one QA and one develop, then cached rejection
        self.assertEqual(model.calls, 2)
        legacy = self.model(with_card=False, bank='legacy-duplicate', transport=duplicate_response)
        self.assertEqual(legacy.complete('develop', {'proposal_protocol': deepcopy(exact_fixture.EXACT)}), {'value': 2})
        self.assertEqual(len(sent), 3)

    def test_unknown_develop_outcome_blocks_resume_without_transport_or_free_retry(self):
        transport = UnknownDevelopTransport()
        with self.assertRaises(UnknownProviderOutcome):
            self.run_fixture(transport)
        self.assertEqual([s for s, _ in transport.requests], ['answer', 'develop'])
        with self.assertRaises(UnknownProviderOutcome):
            self.run_fixture(transport_factory=lambda: (_ for _ in ()).throw(AssertionError('unknown retry')))
        self.assertEqual(len(transport.requests), 2)


if __name__ == '__main__':
    unittest.main()
