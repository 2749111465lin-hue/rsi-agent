"""Synthetic-only integration checks for explicit live evolution controls."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from code_rsi import live_evolution as live
from code_rsi.budget import digest
import test_v3_live_evolution as fixture


class LiveControlTests(unittest.TestCase):
    # Reuse only fixture construction, not inherited test discovery.
    setUp = fixture.LiveEvolutionTests.setUp
    binding = fixture.LiveEvolutionTests.binding
    make_plan = fixture.LiveEvolutionTests.make_plan
    read_role = fixture.LiveEvolutionTests.read_role
    run_fixture = fixture.LiveEvolutionTests.run_fixture

    def schema2(self, *, feedback='rich', module_policy='experience_coverage_v1',
                memory='mechanism', parent_policy='adaptive', fixed_module=None):
        self.plan['schema'] = live.SCHEMA2
        self.plan['controls'] = {
            'parent_policy': parent_policy, 'module_policy': module_policy,
            'fixed_module': fixed_module, 'memory': memory, 'feedback': feedback,
            'case_schedule': [] if feedback == 'rich' else [{
                'question_id': self.read_role('D_fit', 'tasks_file')[0]['question_id'], 'repeat': 0}]}
        return self.plan

    def controlled(self, feedback='trace'):
        return self.schema2(feedback=feedback, parent_policy='fixed_root',
                            module_policy='fixed', fixed_module='evidence_selection', memory='none')

    def test_schema1_keeps_legacy_manifest_and_developer_cache(self):
        with patch.object(live, 'validate_controls', side_effect=AssertionError('schema1 entry changed')):
            check = live.preflight(self.plan)
        self.assertEqual(check['schema'], live.SCHEMA)
        self.assertNotIn('controls', check)
        self.assertNotIn('proposal_bank_policy', check)
        transport = fixture.FixtureTransport()
        self.run_fixture(transport)
        out = Path(self.plan['output_dir'])
        manifest = json.loads((out / 'manifest.json').read_bytes())
        self.assertNotIn('controls', manifest)
        self.assertNotIn('active_controls', manifest)
        self.assertNotIn('proposal_bank_policy', manifest)
        events = [json.loads(line) for line in (out / 'ledger.jsonl').read_text(encoding='utf-8').splitlines()]
        banks = [e['metadata']['bank'] for e in events if e['event'] == 'reserve'
                 and e['metadata'].get('stage') == 'develop']
        self.assertEqual(banks, ['develop'])

    def test_schema2_requires_exact_explicit_fields(self):
        valid = deepcopy(self.schema2())
        variants = []
        missing = deepcopy(valid); missing.pop('controls'); variants.append(missing)
        unknown = deepcopy(valid); unknown['extra'] = True; variants.append(unknown)
        legacy = deepcopy(valid); legacy['schema'] = live.SCHEMA; variants.append(legacy)
        bad_schema = deepcopy(valid); bad_schema['schema'] = 'rag-rsi-live-evolution-3'; variants.append(bad_schema)
        for plan in variants:
            with self.subTest(schema=plan['schema'], keys=sorted(plan)), self.assertRaisesRegex(ValueError, 'exact'):
                live.preflight(plan)
        self.assertFalse(Path(self.plan['output_dir']).exists())

    def test_schema2_preflight_calls_shared_validator_without_side_effects(self):
        self.controlled()
        before = sorted(p.relative_to(self.base).as_posix() for p in self.base.rglob('*'))
        with patch.object(live, 'validate_controls', wraps=live.validate_controls) as validator, \
             patch.object(live, 'credential_from_plan', side_effect=AssertionError('credential')), \
             patch.object(live, 'deepseek_transport', side_effect=AssertionError('transport')):
            check = live.preflight(self.plan)
        validator.assert_called_once_with(self.plan['controls'], self.read_role('D_fit', 'tasks_file'),
                                          self.plan['repeats'], allow_legacy=False)
        self.assertEqual(check['schema'], live.SCHEMA2)
        self.assertEqual(check['controls'], self.plan['controls'])
        self.assertEqual(check['proposal_bank_policy'], live.PROPOSAL_BANK_POLICY)
        self.assertEqual(check['new_api_calls'], 0)
        self.assertEqual(before, sorted(p.relative_to(self.base).as_posix() for p in self.base.rglob('*')))
        check['controls']['case_schedule'].clear()
        self.assertEqual(len(self.plan['controls']['case_schedule']), 1)

    def test_allowed_control_combinations(self):
        for module_policy in ('round_robin_v1', 'experience_coverage_v1', 'fixed'):
            for memory in ('none', 'mechanism'):
                with self.subTest(module_policy=module_policy, memory=memory):
                    self.schema2(module_policy=module_policy, memory=memory,
                                 fixed_module='query_rewrite' if module_policy == 'fixed' else None)
                    self.assertEqual(live.preflight(self.plan)['controls'], self.plan['controls'])
        self.schema2(parent_policy='fixed_root', module_policy='fixed', fixed_module='retrieval', memory='none')
        live.preflight(self.plan)
        for condition in ('aggregate', 'cases', 'trace'):
            with self.subTest(feedback=condition):
                self.controlled(condition)
                live.preflight(self.plan)

    def test_invalid_control_values_and_cross_factor_combinations(self):
        original = deepcopy(self.schema2())
        invalid = [
            {'module_policy': 'legacy'}, {'memory': 'legacy'}, {'parent_policy': 'random'},
            {'module_policy': 'other'}, {'memory': 'other'}, {'feedback': 'other'},
            {'fixed_module': 'retrieval'}, {'module_policy': 'fixed', 'fixed_module': None},
            {'module_policy': 'fixed', 'fixed_module': 'other'}, {'parent_policy': 'fixed_root'},
            {'case_schedule': None}, {'case_schedule': [{'question_id': 'x', 'repeat': 0}]},
        ]
        for mutation in invalid:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                plan = deepcopy(original); plan['controls'].update(mutation); live.preflight(plan)
        for key in ('parent_policy', 'module_policy', 'fixed_module', 'memory', 'feedback', 'case_schedule'):
            with self.subTest(missing=key), self.assertRaisesRegex(ValueError, 'exact'):
                plan = deepcopy(original); plan['controls'].pop(key); live.preflight(plan)
        with self.assertRaisesRegex(ValueError, 'exact'):
            plan = deepcopy(original); plan['controls']['legacy_implicit'] = True; live.preflight(plan)
        self.controlled()
        for key, value in (('parent_policy', 'adaptive'), ('memory', 'mechanism'),
                           ('module_policy', 'round_robin_v1')):
            plan = deepcopy(self.plan); plan['controls'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                live.preflight(plan)

    def test_schedule_is_fit_only_bounded_and_unambiguous(self):
        self.controlled()
        qid = self.plan['controls']['case_schedule'][0]['question_id']
        schedules = [[], [{'question_id': qid, 'repeat': 1}], [{'question_id': qid, 'repeat': -1}],
                     [{'question_id': qid, 'repeat': True}], [{'question_id': qid, 'repeat': 0}] * 2,
                     [{'question_id': self.read_role('D_select', 'tasks_file')[0]['question_id'], 'repeat': 0}],
                     [{'question_id': qid, 'repeat': 0, 'hint': 'forbidden'}],
                     [{'question_id': qid}], [{'question_id': qid, 'repeat': 0}] * 17]
        for schedule in schedules:
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                plan = deepcopy(self.plan); plan['controls']['case_schedule'] = schedule
                live.preflight(plan)

    def test_controls_change_approval_and_cannot_replace_frozen_plan(self):
        self.schema2(memory='none')
        original_hash = digest(self.plan)
        changed = deepcopy(self.plan); changed['controls']['module_policy'] = 'round_robin_v1'
        self.assertNotEqual(live.preflight(changed)['plan_hash'], original_hash)
        with self.assertRaisesRegex(ValueError, 'exact approved'):
            live.run(changed, approved_plan_hash=original_hash, execute=True,
                     transport_factory=lambda: fixture.FixtureTransport())
        transport = fixture.FixtureTransport()
        self.run_fixture(transport)
        calls = len(transport.requests)
        self.plan = changed
        with self.assertRaises(ValueError):
            self.run_fixture(transport)
        self.assertEqual(len(transport.requests), calls)

    def test_schema2_manifest_and_developer_bind_exact_controls(self):
        self.controlled('cases')
        transport = fixture.FixtureTransport()
        self.run_fixture(transport)
        out = Path(self.plan['output_dir'])
        manifest = json.loads((out / 'manifest.json').read_bytes())
        self.assertEqual(manifest['controls'], self.plan['controls'])
        self.assertEqual(manifest['active_controls'], self.plan['controls'])
        self.assertEqual(manifest['proposal_bank_policy'], live.PROPOSAL_BANK_POLICY)
        self.assertEqual(manifest['developer_configuration'], {
            'feedback_condition': 'cases', 'case_schedule': self.plan['controls']['case_schedule'],
            'proposal_model_factory_present': True})
        encoded = json.dumps([payload for stage, payload in transport.requests if stage == 'develop'])
        self.assertNotIn('PRIVATE_GOLD_', encoded)
        self.assertNotIn('Synthetic public question D_select', encoded)
        self.assertNotIn('Synthetic public question D_report', encoded)
        self.assertNotIn('proposal_slot', encoded)

    def test_same_body_different_proposals_are_independent_and_resume_cached(self):
        self.controlled('aggregate')
        self.plan.update(expansions=2, max_calls=51)
        transport = fixture.FixtureTransport()
        self.run_fixture(transport)
        develops = [payload for stage, payload in transport.requests if stage == 'develop']
        self.assertEqual(len(develops), 2)
        self.assertEqual(develops[0], develops[1])
        self.assertNotIn('proposal_slot', json.dumps(develops))
        out = Path(self.plan['output_dir'])
        events = [json.loads(line) for line in (out / 'ledger.jsonl').read_text(encoding='utf-8').splitlines()]
        proposals = [e['metadata'] for e in events if e['event'] == 'reserve'
                     and e['metadata'].get('stage') == 'develop']
        self.assertEqual({r['bank'] for r in proposals}, {'develop/proposal/0', 'develop/proposal/1'})
        self.assertEqual(len({r['request_key'] for r in proposals}), 2)
        for proposal in proposals:
            record = json.loads((out / 'requests' / (proposal['request_key'] + '.json')).read_bytes())
            self.assertEqual(record['key'], digest({'body': record['body'], 'bank': proposal['bank']}))
        before = len(transport.requests)
        self.run_fixture(transport)
        self.assertEqual(len(transport.requests), before)

    def test_request_body_capability_does_not_expose_other_stage(self):
        wrapper, model, transport = fixture.LiveEvolutionTests.bound(self, ('develop',))
        self.assertEqual(wrapper.request_body('develop', {}), model.request_body('develop', {}))
        with self.assertRaisesRegex(live.HostError, 'role cannot'):
            wrapper.request_body('answer', {})
        self.assertEqual(transport.requests, [])


if __name__ == '__main__':
    unittest.main()
