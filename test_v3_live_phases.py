"""Explicit phase boundaries use synthetic fixtures, never a provider or private corpus."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from code_rsi import live_evolution as live
from code_rsi.budget import Ledger, LimitExceeded, digest, save
from code_rsi.v3 import execution
from code_rsi.v3.infrastructure import DeepSeekTransport, UnknownProviderOutcome
import test_v3_live_evolution as fixture


class LivePhaseTests(unittest.TestCase):
    setUp = fixture.LiveEvolutionTests.setUp
    binding = fixture.LiveEvolutionTests.binding
    make_plan = fixture.LiveEvolutionTests.make_plan
    read_role = fixture.LiveEvolutionTests.read_role
    rewrite_role = fixture.LiveEvolutionTests.rewrite_role

    def staged(self, *, fit_only=False):
        self.plan['schema'] = live.SCHEMA3
        self.plan['phase_order'] = ['search'] if fit_only else ['search', 'select', 'report']
        if fit_only:
            self.plan['panels'] = {'D_fit': self.plan['panels']['D_fit']}
            self.plan['max_calls'] = 15
        self.plan['controls'] = {'parent_policy': 'fixed_root', 'module_policy': 'fixed',
            'fixed_module': 'answer_generation', 'memory': 'none', 'feedback': 'cases',
            'case_schedule': [{'question_id': self.read_role('D_fit', 'tasks_file')[0]['question_id'], 'repeat': 0}]}
        groups = {}
        for role in self.plan['panels']:
            refs = self.read_role(role, 'references_file')
            groups[role] = {qid: None if ref.get('pair_group_id', ref.get('source_question_id')) is None else
                           str(ref.get('pair_group_id', ref.get('source_question_id'))) for qid, ref in refs.items()}
        self.plan['reference_groups_file'] = self.binding(self.base/'reference-groups.json',
            {'schema': live.GROUPS_SCHEMA, 'groups': groups})
        return self.plan

    def run_phase(self, phase, transport=None, *, factory=None):
        if factory is None:
            factory = lambda: transport
        with patch.object(execution, 'execute', side_effect=fixture.fake_execute):
            return live.run(self.plan, phase=phase, approved_plan_hash=digest(self.plan),
                            execute=True, transport_factory=factory)

    def parsed_roles(self, call):
        original = live._file
        parsed = []
        bindings = {role: self.plan['panels'][role]['references_file'] for role in self.plan['panels']}
        def observe(item, *, parse=False):
            if parse:
                parsed.extend(role for role, binding in bindings.items() if item == binding)
            return original(item, parse=parse)
        with patch.object(live, '_file', side_effect=observe):
            value = call()
        return value, parsed

    def test_schema3_requires_explicit_phase_and_exact_new_fields(self):
        self.staged()
        for phase in (None, 'all', 'D_fit'):
            with self.subTest(phase=phase), self.assertRaisesRegex(ValueError, 'explicit declared phase'):
                live.preflight(self.plan, phase=phase)
        for key in ('phase_order', 'reference_groups_file', 'controls'):
            plan = deepcopy(self.plan); plan.pop(key)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'exact'):
                live.preflight(plan, phase='search')
        for order in ([], ['search', 'select'], ['report', 'select', 'search']):
            plan = deepcopy(self.plan); plan['phase_order'] = order
            with self.subTest(order=order), self.assertRaisesRegex(ValueError, 'phase_order'):
                live.preflight(plan, phase='search')
        self.assertFalse(Path(self.plan['output_dir']).exists())

    def test_legacy_still_has_no_phase_and_old_field_set(self):
        with self.assertRaisesRegex(ValueError, 'legacy'):
            live.preflight(self.plan, phase='search')
        self.assertEqual(live.preflight(self.plan)['schema'], live.SCHEMA)

    def test_preflight_hashes_all_references_without_parsing_or_io(self):
        self.staged()
        before = sorted(str(p) for p in self.base.rglob('*'))
        with patch.object(live, 'credential_from_plan', side_effect=AssertionError('credential')), \
             patch.object(live, 'deepseek_transport', side_effect=AssertionError('network')):
            check, parsed = self.parsed_roles(lambda: live.preflight(self.plan, phase='search'))
        self.assertEqual(parsed, [])
        self.assertFalse(check['private_references_read_locally'])
        self.assertEqual(check['reference_roles_parsed'], [])
        self.assertEqual(check['phase_limits']['search']['calls'], 15)
        self.assertEqual(check['phase_limits']['select']['calls'], 14)
        self.assertEqual(check['phase_limits']['report']['calls'], 14)
        self.assertEqual(sum(x['calls'] for x in check['phase_limits'].values()), 43)
        self.assertAlmostEqual(sum(x['cny'] for x in check['phase_limits'].values()), check['conservative_cny_upper_bound'], places=6)
        self.assertEqual(sorted(str(p) for p in self.base.rglob('*')), before)

    def test_fit_only_needs_no_holdout_bindings_or_placeholder_rows(self):
        self.staged(fit_only=True)
        check = live.preflight(self.plan, phase='search')
        self.assertEqual(check['question_counts'], {'D_fit': 1})
        self.assertEqual(check['max_calls'], 15)
        result, parsed = self.parsed_roles(lambda: self.run_phase('search', fixture.FixtureTransport()))
        self.assertEqual(result['status'], 'search_frozen')
        self.assertEqual(parsed, ['D_fit'])
        out = Path(self.plan['output_dir'])
        self.assertTrue((out/'phase_search.json').is_file())
        self.assertFalse((out/'delivery_lock.json').exists())
        self.assertFalse((out/'report.json').exists())
        with self.assertRaisesRegex(ValueError, 'explicit declared phase'):
            self.run_phase('select', fixture.FixtureTransport())
        plan = deepcopy(self.plan); plan['panels']['D_report'] = deepcopy(plan['panels']['D_fit'])
        with self.assertRaisesRegex(ValueError, 'role-scoped'):
            live.preflight(plan, phase='search')

    def test_three_phases_load_only_current_reference_then_resume_without_transport(self):
        self.staged()
        transport = fixture.FixtureTransport()
        calls = []
        for phase, role in live.PHASE_ROLES.items():
            with self.subTest(phase=phase):
                result, parsed = self.parsed_roles(lambda: self.run_phase(phase, transport))
                self.assertEqual(parsed, [role])
                calls.append(len(transport.requests))
                def forbidden_factory():
                    raise AssertionError('completed phase built transport')
                resumed, again = self.parsed_roles(lambda: self.run_phase(phase, factory=forbidden_factory))
                self.assertEqual(again, [])
                self.assertEqual(result, resumed)
                self.assertEqual(len(transport.requests), calls[-1])
        self.assertLess(calls[0], calls[1]); self.assertLess(calls[1], calls[2])
        out = Path(self.plan['output_dir'])
        events = [json.loads(line) for line in (out/'ledger.jsonl').read_text(encoding='utf-8').splitlines()]
        reservations = [x for x in events if x['event'] == 'reserve']
        self.assertEqual(len(reservations), len(transport.requests))
        for event in reservations:
            phase = event['metadata']['execution_phase']
            self.assertEqual(event['scopes'], ['run', 'phase:'+phase])
            self.assertEqual(live._phase_for_bank(event['metadata']['bank']), phase)
        status = json.loads((out/'live_status.json').read_bytes())
        self.assertEqual(status['status'], 'phase_complete')
        self.assertEqual(status['ledger']['used']['run']['calls'], len(transport.requests))
        self.assertEqual(sum(v['calls'] for k,v in status['ledger']['used'].items() if k.startswith('phase:')),len(transport.requests))

    def test_skipping_predecessor_does_not_parse_refs_or_construct_transport(self):
        self.staged()
        for phase in ('select', 'report'):
            def call():
                return self.run_phase(phase, factory=lambda: (_ for _ in ()).throw(AssertionError('transport')))
            original = live._file
            def no_reference_parse(item, *, parse=False):
                if parse and item in [b['references_file'] for b in self.plan['panels'].values()]:
                    raise AssertionError('reference parsed before predecessor')
                return original(item, parse=parse)
            with self.subTest(phase=phase), patch.object(live, '_file', side_effect=no_reference_parse), \
                 self.assertRaisesRegex(ValueError, 'phase receipt'):
                call()

    def test_malformed_heldout_json_not_parsed_until_its_phase(self):
        self.staged()
        item = self.plan['panels']['D_select']['references_file']
        Path(item['path']).write_text('not-json PRIVATE_SENTINEL', encoding='utf-8')
        item['sha256'] = live._hash(item['path'])
        transport = fixture.FixtureTransport()
        self.run_phase('search', transport)
        prior = len(transport.requests)
        with self.assertRaises(ValueError):
            self.run_phase('select', factory=lambda: (_ for _ in ()).throw(AssertionError('transport')))
        self.assertEqual(len(transport.requests), prior)

    def test_group_metadata_must_be_exact_answer_free_and_frozen(self):
        self.staged()
        original = json.loads(Path(self.plan['reference_groups_file']['path']).read_bytes())
        variants = []
        added = deepcopy(original); added['answers'] = {'hidden':'PRIVATE'}; variants.append(added)
        missing = deepcopy(original); missing['groups']['D_fit'] = {}; variants.append(missing)
        wrong = deepcopy(original); next(iter(wrong['groups'].values()))[next(iter(wrong['groups']['D_fit']))] = []; variants.append(wrong)
        for value in variants:
            with self.subTest(value=value):
                self.plan['reference_groups_file'] = self.binding(self.base/'reference-groups.json', value)
                with self.assertRaisesRegex(ValueError, 'metadata'):
                    live.preflight(self.plan, phase='search')
        self.plan['reference_groups_file'] = self.binding(self.base/'reference-groups.json', original)
        Path(self.plan['reference_groups_file']['path']).write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            live.preflight(self.plan, phase='search')

    def test_private_group_mismatch_rejected_before_first_call(self):
        self.staged()
        refs = self.read_role('D_fit', 'references_file')
        next(iter(refs.values()))['source_question_id'] = 'unrecorded-group'
        self.rewrite_role('D_fit', 'references_file', refs)
        with self.assertRaisesRegex(ValueError, 'group differs'):
            self.run_phase('search', factory=lambda: (_ for _ in ()).throw(AssertionError('transport')))

    def test_metadata_cross_role_overlap_is_detected_without_reading_gold(self):
        self.staged()
        metadata = json.loads(Path(self.plan['reference_groups_file']['path']).read_bytes())
        for role in ('D_fit','D_report'):
            metadata['groups'][role] = {qid:'same-family' for qid in metadata['groups'][role]}
        self.plan['reference_groups_file'] = self.binding(self.base/'reference-groups.json', metadata)
        with self.assertRaisesRegex(ValueError, 'group crosses roles'):
            live.preflight(self.plan, phase='search')

    def test_unknown_request_blocks_any_phase_without_new_transport(self):
        self.staged()
        transport = fixture.FixtureTransport(unknown=True)
        with self.assertRaises(UnknownProviderOutcome): self.run_phase('search', transport)
        self.assertEqual(len(transport.requests), 1)
        for phase in ('search', 'select', 'report'):
            with self.subTest(phase=phase), self.assertRaises(UnknownProviderOutcome):
                self.run_phase(phase, factory=lambda: (_ for _ in ()).throw(AssertionError('new transport')))

    def test_changed_previous_seal_blocks_before_private_unlock(self):
        self.staged()
        self.run_phase('search', fixture.FixtureTransport())
        seal = Path(self.plan['output_dir'])/'search_frozen.json'
        value = json.loads(seal.read_bytes()); value['extra'] = True; save(seal,value)
        with patch.object(live, '_reference_loader', return_value=lambda role: (_ for _ in ()).throw(AssertionError('unlock'))), \
             self.assertRaisesRegex(ValueError, 'artifact changed'):
            self.run_phase('select', factory=lambda: (_ for _ in ()).throw(AssertionError('new transport')))

    def test_phase_ledger_enforces_both_caps(self):
        limits={'run':{'calls':2,'cny':2}, 'phase:search':{'calls':1,'cny':1}, 'phase:select':{'calls':2,'cny':2}}
        ledger=Ledger(self.base/'budget.jsonl',limits)
        search=live._PhaseLedger(ledger,'search'); select=live._PhaseLedger(ledger,'select')
        rid=search.reserve(['run'],{'calls':1,'cny':.5},{'bank':'develop/0'}); search.settle(rid)
        with self.assertRaises(LimitExceeded): search.reserve(['run'],{'calls':1,'cny':.1})
        rid=select.reserve(['run'],{'calls':1,'cny':.5}); select.settle(rid)
        with self.assertRaises(LimitExceeded): select.reserve(['run'],{'calls':1,'cny':.1})
        self.assertEqual(ledger.summary()['used']['run']['calls'],2)

    def test_phase_accounting_rejects_scope_relabel(self):
        self.staged(); self.run_phase('search',fixture.FixtureTransport())
        path=Path(self.plan['output_dir'])/'ledger.jsonl'
        events=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        reserve=next(x for x in events if x['event']=='reserve'); reserve['scopes']=['run']
        path.write_text(''.join(json.dumps(x)+'\n' for x in events),encoding='utf-8')
        with self.assertRaisesRegex(execution.HostError,'phase request scope'):
            self.run_phase('search',fixture.FixtureTransport())

    def test_lazy_live_transport_preserves_provider_identity_guard(self):
        called=[]
        proxy=live._LazyLiveTransport(lambda: called.append(True) or fixture.FixtureTransport())
        self.assertIsInstance(proxy,DeepSeekTransport)
        self.assertEqual(called,[])

    def test_cli_requires_phase_only_for_schema3_and_has_no_injection_switch(self):
        self.staged()
        path=self.base/'plan.json'; save(path,self.plan)
        with self.assertRaisesRegex(ValueError,'explicit declared phase'):
            live.main(['preflight','--plan',str(path)])
        with patch('builtins.print'):
            result=live.main(['preflight','--plan',str(path),'--phase','search'])
        self.assertEqual(result['phase'],'search')
        with patch('sys.stderr'), self.assertRaises(SystemExit):
            live.main(['run','--plan',str(path),'--phase','search','--execute','--approved-plan-hash',digest(self.plan),
                       '--transport-factory','arbitrary.module'])


if __name__ == '__main__':
    unittest.main()
