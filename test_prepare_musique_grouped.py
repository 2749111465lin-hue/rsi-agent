"""Synthetic tests for explicit-history, observed atomic-disjoint MuSiQue panels."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from code_rsi import prepare_musique as pm
import test_v3_prepare_musique as legacy


class GroupedPanelsTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parent / 'runs'
        root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='test_grouped_musique_', dir=root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train = self.root / 'musique_ans_v1.0_train.jsonl'
        self.dev = self.root / 'musique_ans_v1.0_dev.jsonl'
        self.outputs = self.root / 'output'
        self.patch = patch.object(pm, 'RUNS_ROOT', self.outputs)
        self.patch.start(); self.addCleanup(self.patch.stop)

    def inputs(self, train=None, dev=None):
        self.train_rows = train or [legacy.fixture('A'), legacy.fixture('B'), legacy.fixture('C')]
        self.dev_rows = dev or [legacy.fixture('D'), legacy.fixture('E')]
        for path, rows in ((self.train, self.train_rows), (self.dev, self.dev_rows)):
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')

    def qid(self, row):
        return pm._adapt(row)[0]['question_id']

    def history(self, entries=None):
        return {'schema': pm.HISTORY_SCHEMA, 'source_files': {
            split: {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            for split, path in (('train', self.train), ('dev', self.dev))}, 'entries': entries or []}

    def prepare(self, history=None, quotas=None, seed='frozen-test-seed'):
        return pm.prepare_grouped_panels(self.train, self.dev, seed=seed,
            quotas=quotas or legacy.quotas(), history=self.history() if history is None else history)

    def chosen(self, bundle):
        return {task['question_id'] for tasks in bundle['public_panels'].values() for task in tasks}

    def test_legacy_schema2_and_new_schema3_are_explicit(self):
        self.inputs()
        old = pm.prepare_panels(self.train, self.dev, seed='fixed', quotas=legacy.quotas())
        new = self.prepare()
        self.assertEqual(old['manifest']['schema'], pm.SCHEMA)
        self.assertNotIn('groups', old)
        self.assertEqual(new['manifest']['schema'], pm.GROUPED_SCHEMA)
        self.assertEqual(new['manifest']['sampling_policy'], 'known_atomic_disjoint_observed_panel_v1')
        self.assertIn('schema2_unchanged', new['manifest']['subanswer_policy'])

    def test_complete_panels_atomic_groups_and_private_separation(self):
        self.inputs([legacy.fixture(f't{h}_{i}', h) for h in (2, 3, 4) for i in range(3)],
                    [legacy.fixture(f'd{h}_{i}', h) for h in (2, 3, 4) for i in range(2)])
        quotas = {role: {2: 1, 3: 1, 4: 1} for role in pm.ROLES}
        bundle = self.prepare(quotas=quotas)
        self.assertEqual(len(self.chosen(bundle)), 9)
        groups = bundle['groups']['question_groups']
        self.assertEqual(len(groups), len(set(groups.values())))
        self.assertEqual(set(groups), self.chosen(bundle))
        for role, tasks in bundle['public_panels'].items():
            self.assertNotIn('PRIVATE_', json.dumps(tasks))
            self.assertEqual(bundle['manifest']['audit'][role]['selected'], quotas[role])
        self.assertNotIn('PRIVATE_', json.dumps(bundle['manifest']))
        self.assertNotIn('PRIVATE_', json.dumps(bundle['groups']))
        self.assertFalse(bundle['manifest']['model_scores_used'])

    def test_history_required_closed_and_source_bound(self):
        self.inputs()
        invalid = [None, {}, {**self.history(), 'extra': True}]
        bad = self.history(); bad['source_files']['train']['sha256'] = '0' * 64; invalid.append(bad)
        bad = self.history(); bad['source_files']['train']['path'] = str(self.dev); invalid.append(bad)
        for history in invalid:
            with self.subTest(history_type=type(history).__name__):
                with self.assertRaises(pm.PanelError):
                    pm.prepare_grouped_panels(self.train, self.dev, seed='fixed', quotas=legacy.quotas(), history=history)

    def test_unknown_duplicate_wrong_role_and_status_rejected(self):
        self.inputs()
        entry = {'question_id': self.qid(self.train_rows[0]), 'role': 'D_fit', 'status': 'exposed'}
        variations = [[{**entry, 'question_id': 'unknown'}], [entry, entry], [{**entry, 'role': 'D_report'}],
                      [{**entry, 'status': 'unknown'}], [{**entry, 'status': {}}], [{**entry, 'extra': 1}]]
        for entries in variations:
            with self.assertRaises(pm.PanelError):
                self.prepare(history=self.history(entries))

    def test_unused_bridge_does_not_propagate_exposure(self):
        a, bridge, c, clean = [legacy.fixture(name) for name in ('A', 'bridge', 'C', 'clean')]
        bridge['paragraphs'][0]['paragraph_text'] = a['paragraphs'][0]['paragraph_text']
        bridge['paragraphs'][1]['paragraph_text'] = c['paragraphs'][0]['paragraph_text']
        self.inputs([a, bridge, c, clean])
        history = self.history([{'question_id': self.qid(a), 'role': 'D_fit', 'status': 'exposed'}])
        result = self.prepare(history)
        self.assertIn(self.qid(c), self.chosen(result))
        self.assertNotIn(self.qid(a), self.chosen(result))
        self.assertNotIn(self.qid(bridge), self.chosen(result))
        sensitivity = result['manifest']['full_source_family_sensitivity']
        self.assertGreater(sensitivity['history_connected_row_count'], result['manifest']['history_directly_excluded_rows'])
        self.assertFalse(sensitivity['used_for_selection'])

    def test_selected_bridge_blocks_its_actual_overlap(self):
        bridge, c, clean = [legacy.fixture(name) for name in ('bridge', 'C', 'clean')]
        bridge['paragraphs'][0]['paragraph_text'] = c['paragraphs'][0]['paragraph_text']
        self.inputs([bridge, c, clean])
        original = pm._group_hash
        def force_fixed_order(value):
            if isinstance(value, list) and value and value[0] == 'rag-rsi-musique-observed-atomic-disjoint-1':
                return '0' if value[-1] == self.qid(bridge) else '1' + original(value)
            return original(value)
        with patch.object(pm, '_group_hash', side_effect=force_fixed_order):
            result = self.prepare()
        self.assertIn(self.qid(bridge), self.chosen(result))
        self.assertNotIn(self.qid(c), self.chosen(result))
        records = [r for r in result['groups']['candidate_audit'] if r['question_id'] == self.qid(c)]
        self.assertTrue(any('selected_atomic_overlap_support_paragraph' in d['reasons'] for r in records for d in r['role_decisions'].values()))

    def test_direct_exposed_reserved_links_are_separate_from_family_links(self):
        a, b, bridge, c = [legacy.fixture(name) for name in ('A', 'B', 'bridge', 'C')]
        b['paragraphs'][0]['paragraph_text'] = a['paragraphs'][0]['paragraph_text']
        bridge['paragraphs'][0]['paragraph_text'] = a['paragraphs'][0]['paragraph_text']
        bridge['paragraphs'][1]['paragraph_text'] = c['paragraphs'][0]['paragraph_text']
        self.inputs([a, b, bridge, c, legacy.fixture('clean1'), legacy.fixture('clean2')])
        history = self.history([{'question_id': self.qid(a), 'role': 'D_fit', 'status': 'exposed'},
            {'question_id': self.qid(b), 'role': 'D_select', 'status': 'reserved'},
            {'question_id': self.qid(c), 'role': 'D_fit', 'status': 'reserved'}])
        result = self.prepare(history)
        direct = result['manifest']['reserved_history_directly_associated_with_exposure']
        family = result['manifest']['full_source_family_sensitivity']['reserved_hypothetical_family_connections_to_exposed']
        self.assertEqual({e['question_id'] for e in direct}, {self.qid(b)})
        self.assertEqual({e['question_id'] for e in family}, {self.qid(b), self.qid(c)})
        self.assertTrue(all(e['role'] == 'D_select' for e in direct))
        self.assertFalse({self.qid(a), self.qid(b), self.qid(c)} & self.chosen(result))

    def test_raw_templates_with_different_bindings_do_not_collide(self):
        a, b = legacy.fixture('A'), legacy.fixture('B')
        a['question_decomposition'][0]['answer'] = 'Alpha Entity'
        b['question_decomposition'][0]['answer'] = 'Beta Entity'
        for row in (a, b): row['question_decomposition'][1]['question'] = 'Where is #1?'
        self.inputs([a, b, legacy.fixture('clean')])
        result = self.prepare(self.history([{'question_id': self.qid(a), 'role': 'D_fit', 'status': 'exposed'}]))
        self.assertIn(self.qid(b), self.chosen(result))
        self.assertTrue(result['manifest']['raw_template_rule_rejected'])

    def test_grounded_identical_subquestions_block_despite_different_ids(self):
        a, b = legacy.fixture('A'), legacy.fixture('B')
        a['question_decomposition'][0]['answer'] = 'Alpha Entity'
        a['question_decomposition'][1]['question'] = 'Where is #1?'
        b['question_decomposition'][1]['question'] = '  WHERE is Alpha Entity?  '
        self.inputs([a, b, legacy.fixture('clean1'), legacy.fixture('clean2')])
        result = self.prepare(self.history([{'question_id': self.qid(a), 'role': 'D_fit', 'status': 'exposed'}]))
        self.assertNotIn(self.qid(b), self.chosen(result))
        audit = next(r for r in result['groups']['candidate_audit'] if r['question_id'] == self.qid(b))
        self.assertIn('normalized_subquestion', audit['direct_history_matches'])

    def test_unresolved_numeric_reference_omits_only_extra_edge(self):
        a, b = legacy.fixture('A'), legacy.fixture('B')
        for row in (a, b): row['question_decomposition'][0]['question'] = 'What is #9 called?'
        self.inputs([a, b, legacy.fixture('clean')])
        result = self.prepare(self.history([{'question_id': self.qid(a), 'role': 'D_fit', 'status': 'reserved'}]))
        self.assertIn(self.qid(b), self.chosen(result))
        self.assertEqual(result['manifest']['text_identity_unresolved_count'], 2)
        b['paragraphs'][0]['paragraph_text'] = a['paragraphs'][0]['paragraph_text']
        self.inputs([a, b, legacy.fixture('clean1'), legacy.fixture('clean2')])
        result = self.prepare(self.history([{'question_id': self.qid(a), 'role': 'D_fit', 'status': 'reserved'}]))
        self.assertNotIn(self.qid(b), self.chosen(result))

    def test_common_subanswer_alone_is_not_schema3_overlap(self):
        self.inputs()
        for row in self.train_rows + self.dev_rows: row['question_decomposition'][0]['answer'] = 'Common Answer'
        self.inputs(self.train_rows, self.dev_rows)
        self.assertEqual(self.prepare()['manifest']['status'], 'ready')

    def test_cross_split_atomic_overlap_is_blocked(self):
        train = [legacy.fixture(name) for name in ('shared', 'clean1', 'clean2')]
        dev = [legacy.fixture('report')]
        train[0]['paragraphs'][0]['paragraph_text'] = dev[0]['paragraphs'][0]['paragraph_text']
        self.inputs(train, dev)
        result = self.prepare()
        self.assertIn(self.qid(dev[0]), self.chosen(result))
        self.assertNotIn(self.qid(train[0]), self.chosen(result))

    def test_fixed_seed_is_deterministic_and_order_independent(self):
        train = [legacy.fixture(f't{i}') for i in range(9)]
        dev = [legacy.fixture(f'd{i}') for i in range(6)]
        self.inputs(train, dev)
        first = self.prepare()
        self.assertEqual(first, self.prepare())
        self.inputs(list(reversed(train)), list(reversed(dev)))
        second = self.prepare()
        self.assertEqual(first['public_panels'], second['public_panels'])
        self.assertEqual(first['groups']['question_groups'], second['groups']['question_groups'])

    def test_quota_shortfall_records_all_roles_and_no_partial_panel(self):
        self.inputs([legacy.fixture('one')], [legacy.fixture('dev')])
        with self.assertRaises(pm.PanelQuotaError) as raised: self.prepare()
        error = raised.exception
        self.assertEqual(error.manifest['status'], 'quota_shortfall')
        self.assertEqual(set(error.manifest['audit']), set(pm.ROLES))
        self.assertIsNotNone(error.groups)
        history = self.root / 'history.json'; history.write_text(json.dumps(self.history()), encoding='utf-8')
        out = self.outputs / 'failed'
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            status = pm.main(['--train', str(self.train), '--dev', str(self.dev), '--out', str(out), '--seed', 'fixed',
                '--history', str(history), '--fit-quota', '1,0,0', '--select-quota', '1,0,0', '--report-quota', '1,0,0'])
        self.assertEqual(status, 2)
        self.assertEqual({p.name for p in out.iterdir()}, {'manifest.json', 'groups.json'})

    def test_publishing_binds_group_audit_and_never_overwrites(self):
        self.inputs(); bundle = self.prepare(); out = pm.write_panels(bundle, self.outputs / 'ready')
        manifest = json.loads((out / 'manifest.json').read_bytes())
        self.assertEqual(manifest['files']['groups.json']['sha256'], hashlib.sha256((out / 'groups.json').read_bytes()).hexdigest())
        self.assertTrue(manifest['files']['groups.json']['private'])
        with self.assertRaises(pm.PanelError): pm.write_panels(bundle, out)

    def assert_publication_rejected(self, bundle):
        out = self.outputs / 'rejected'
        with self.assertRaises(pm.PanelError):
            pm.write_panels(bundle, out)
        self.assertFalse(out.exists(), 'invalid handoff created output before validation')

    def test_writer_rejects_truncated_panel_even_with_ready_label(self):
        self.inputs(); original = self.prepare()
        for role in pm.ROLES:
            bundle = deepcopy(original)
            bundle['public_panels'][role] = []
            with self.subTest(role=role):
                self.assert_publication_rejected(bundle)
        bundle = deepcopy(original)
        bundle['public_panels']['D_report'] = []
        bundle['private_references']['D_report'] = {}
        self.assert_publication_rejected(bundle)

    def test_writer_rejects_changed_public_id_and_cross_role_duplicate(self):
        self.inputs(); original = self.prepare()
        bundle = deepcopy(original)
        bundle['public_panels']['D_fit'][0]['question_id'] += '-unfrozen'
        self.assert_publication_rejected(bundle)
        bundle = deepcopy(original)
        duplicate = deepcopy(bundle['public_panels']['D_fit'][0])
        bundle['public_panels']['D_select'] = [duplicate]
        bundle['private_references']['D_select'] = {
            duplicate['question_id']: deepcopy(bundle['private_references']['D_fit'][duplicate['question_id']])}
        bundle['manifest']['audit']['D_select']['selected_question_ids'] = [duplicate['question_id']]
        self.assert_publication_rejected(bundle)

    def test_writer_rejects_missing_extra_wrong_identity_or_unscorable_references(self):
        self.inputs(); original = self.prepare()
        qid = original['public_panels']['D_fit'][0]['question_id']
        for mutation in ('missing', 'extra', 'identity', 'empty_answers'):
            bundle = deepcopy(original)
            refs = bundle['private_references']['D_fit']
            if mutation == 'missing':
                refs.pop(qid)
            elif mutation == 'extra':
                refs['unknown'] = deepcopy(refs[qid])
            elif mutation == 'identity':
                refs[qid]['question_id'] = 'wrong-reference-identity'
            else:
                refs[qid]['answers'] = []
            with self.subTest(mutation=mutation):
                self.assert_publication_rejected(bundle)

    def test_writer_rejects_missing_unknown_or_shared_group_mapping(self):
        self.inputs(); original = self.prepare()
        qids = list(original['groups']['question_groups'])
        for mutation in ('whole_audit', 'missing_question', 'unknown_group', 'shared_group', 'missing_record'):
            bundle = deepcopy(original)
            mapping = bundle['groups']['question_groups']
            if mutation == 'whole_audit':
                bundle.pop('groups')
            elif mutation == 'missing_question':
                mapping.pop(qids[0])
            elif mutation == 'unknown_group':
                mapping[qids[0]] = 'unrecorded-group'
            elif mutation == 'shared_group':
                mapping[qids[0]] = mapping[qids[1]]
            else:
                bundle['groups']['groups'].pop(mapping[qids[0]])
            with self.subTest(mutation=mutation):
                self.assert_publication_rejected(bundle)

    def test_writer_rejects_wrong_member_role_or_duplicate_group_member(self):
        self.inputs(); original = self.prepare()
        qid = original['public_panels']['D_report'][0]['question_id']
        gid = original['groups']['question_groups'][qid]
        for mutation in ('role', 'hop', 'duplicate', 'false_selected_flag', 'history_status'):
            bundle = deepcopy(original)
            group = bundle['groups']['groups'][gid]
            if mutation == 'role':
                group['members'][0]['role'] = 'D_fit'
            elif mutation == 'hop':
                group['members'][0]['hop'] = 3
            elif mutation == 'duplicate':
                group['members'].append(deepcopy(group['members'][0]))
                group['row_count'] = 2
            elif mutation == 'false_selected_flag':
                group['contains_selected_question'] = False
            else:
                group['members'][0]['history_status'] = 'exposed'
            with self.subTest(mutation=mutation):
                self.assert_publication_rejected(bundle)

    def test_writer_rejects_quota_audit_and_candidate_handoff_mismatch(self):
        self.inputs(); original = self.prepare()
        qid = original['public_panels']['D_fit'][0]['question_id']
        for mutation in ('quota', 'audit_count', 'audit_id', 'missing_candidate', 'false_overlap'):
            bundle = deepcopy(original)
            if mutation == 'quota':
                bundle['manifest']['quotas']['D_fit'][2] = 2
            elif mutation == 'audit_count':
                bundle['manifest']['audit']['D_fit']['selected'][2] = 2
            elif mutation == 'audit_id':
                bundle['manifest']['audit']['D_fit']['selected_question_ids'] = ['foreign']
            elif mutation == 'missing_candidate':
                bundle['groups']['candidate_audit'] = [r for r in bundle['groups']['candidate_audit'] if r['question_id'] != qid]
            else:
                row = next(r for r in bundle['groups']['candidate_audit'] if r['question_id'] == qid)
                row['role_decisions']['D_fit']['selected_overlap_question_ids'] = {'question_id': ['historical']}
            with self.subTest(mutation=mutation):
                self.assert_publication_rejected(bundle)

    def test_writer_preserves_historical_shared_group_and_selected_identities(self):
        a, b = legacy.fixture('A'), legacy.fixture('B')
        b['paragraphs'][0]['paragraph_text'] = a['paragraphs'][0]['paragraph_text']
        self.inputs([a, b, legacy.fixture('clean1'), legacy.fixture('clean2')])
        history = self.history([{'question_id': self.qid(a), 'role': 'D_fit', 'status': 'exposed'},
                                {'question_id': self.qid(b), 'role': 'D_select', 'status': 'reserved'}])
        bundle = self.prepare(history)
        before = deepcopy(bundle)
        out = pm.write_panels(bundle, self.outputs / 'history-preserved')
        self.assertEqual(bundle, before)
        self.assertTrue(any(g['row_count'] == 2 and not g['contains_selected_question']
                            for g in bundle['groups']['groups'].values()))
        for role in pm.ROLES:
            self.assertEqual(json.loads((out / role / 'public_tasks.json').read_bytes()), bundle['public_panels'][role])

    def test_history_file_change_during_materialization_fails(self):
        self.inputs(); history = self.root / 'history.json'; history.write_text(json.dumps(self.history()), encoding='utf-8')
        original = pm._materialize
        def mutate(*args, **kwargs):
            history.write_text('{}', encoding='utf-8')
            return original(*args, **kwargs)
        with patch.object(pm, '_materialize', side_effect=mutate):
            with self.assertRaisesRegex(pm.PanelError, 'history file changed'):
                self.prepare(history)


if __name__ == '__main__': unittest.main()
