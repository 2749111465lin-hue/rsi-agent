"""Pure synthetic host-flow regressions; no model, benchmark or credential access."""
from copy import deepcopy
import hashlib
import json
import unittest

from code_rsi.v3.diagnostics import FLOW_BOUNDS, compact_feedback, execution_flow
from test_v3_diagnostics import measurement, tasks
from test_v3_fixture_origin import bind_synthetic_origin


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def window(docid, text, start=0, sid='s1'):
    return {'docid': docid, 'start': start, 'end': start + len(text), 'text': text, 'source_id': sid}


def fingerprint(source):
    return {k: source[k] for k in ('docid', 'start', 'end')} | {'text_sha256': sha(source['text'])}


def quote(source, text=None):
    value = text if text is not None else source['text'][:4]
    start = source['start'] + source['text'].index(value)
    return {'docid': source['docid'], 'start': start, 'end': start + len(value), 'quote': value}


def receipt(rounds, *, legacy=False, final_quotes=None, qid='q1', score=0, repeat=0):
    events, presentations, verified = [], [], []
    for n, spec in enumerate(rounds):
        sources = spec['sources']; returned = spec.get('returned', sources)
        for query in spec.get('queries', ['synthetic query ' + str(n)]):
            event = {'name': 'search', 'request': {'query': query, 'limit': 5}, 'response_hash': sha('opaque response')}
            if not legacy:
                event.update(observed_windows=[fingerprint(s) for s in returned[:30]],
                             observed_window_count=len(returned), observed_windows_truncated=len(returned) > 30)
            events.append(event)
        quotes = spec.get('quotes', [quote(s) for s in sources])
        presentations.append({'sources': deepcopy(sources), 'verified_quotes': quotes, **({} if legacy else {'event_index': len(events)})})
        events.append({'name': 'complete', 'request': {'stage': 'read', 'payload': {'sources': deepcopy(sources)}},
                       'response_hash': sha('completed model response')})
        verified.extend(quotes)
    selected = final_quotes if final_quotes is not None else verified
    evidence = {'e' + str(i): q for i, q in enumerate(selected)}
    final = {'evidence': evidence, 'response': {'answer': 'synthetic answer', 'citation_ids': list(evidence), 'evidence_sufficient': True},
             **({} if legacy else {'event_index': len(events)})}
    events.append({'name': 'complete', 'request': {'stage': 'answer', 'payload': {'evidence': list(evidence.values())}},
                   'response_hash': sha('completed answer')})
    result = {'schema': 'rag-rsi-v3-execution-2', 'role': 'D_fit', 'question_id': qid, 'node_id': 'child',
            'score': score, 'repeat': repeat, 'answer': 'synthetic answer', 'answer_usable': True,
            'execution_ok': True, 'citation_source_valid': True, 'failure_classes': [], 'model_errors': [],
            'trace': events, 'host_evidence_trace': {'read_presentations': presentations, 'final_observations': [final]},
            'host_citation_validation': {'valid': True, 'status': 'source_and_presentation_verified',
                                         'raw_citation_ids': list(evidence), 'presented_citation_ids': list(evidence)},
            'candidate_reported': {}}
    return result if legacy else bind_synthetic_origin(result)


class ExecutionFlowTests(unittest.TestCase):
    def test_full_first_window_then_stale_second_read_exposes_blockage(self):
        full = [window('old-' + str(i), chr(97+i) * 6000, sid='s'+str(i)) for i in range(4)]
        novel = window('key', 'new decisive evidence')
        row = receipt([{'sources': full}, {'sources': full, 'returned': [novel], 'queries': ['bridge entity birthplace']}])
        flow = execution_flow(row)
        first, second = flow['reads']
        self.assertEqual(first['source_chars'], 24000)
        self.assertEqual(second['source_chars'], 24000)
        self.assertEqual(second['new_source_count'], 0)
        self.assertEqual(second['reused_source_count'], 4)
        self.assertEqual(second['returned_not_presented_count'], 1)
        self.assertEqual(second['queries'][0]['query_excerpt'], 'bridge entity birthplace')
        self.assertEqual(second['drop_cause'], 'not_observed')

    def test_rolling_new_sources_and_old_verified_quote_retention(self):
        old, new = window('old', 'old bridge evidence'), window('new', 'new answer evidence')
        row = receipt([{'sources': [old]}, {'sources': [new]}])
        flow = execution_flow(row)
        self.assertEqual(flow['reads'][1]['new_source_count'], 1)
        self.assertEqual(flow['reads'][1]['reused_source_count'], 0)
        self.assertEqual(flow['reads'][1]['returned_not_presented_count'], 0)
        self.assertEqual(flow['final']['verified_quotes_retained_count'], 2)
        self.assertEqual(flow['final']['verified_quotes_cited_count'], 2)
        self.assertEqual(flow['final']['verified_quotes_not_retained_count'], 0)
        self.assertEqual(flow['semantic_support'], 'not_host_verified')

    def test_legacy_hash_never_invents_return_count_or_dropped_window(self):
        a, b = window('a', 'abcd'), window('b', 'efgh')
        flow = execution_flow(receipt([{'sources': [a]}, {'sources': [b]}], legacy=True))
        self.assertEqual(flow['counts']['searches_with_unknown_return_count'], 2)
        self.assertIsNone(flow['reads'][0]['queries'][0]['returned_count'])
        self.assertIsNone(flow['reads'][1]['returned_not_presented_count'])
        self.assertEqual(flow['reads'][1]['new_source_count'], 1)

    def test_partial_return_metadata_retains_count_but_not_set_difference(self):
        sources = [window('s'+str(i), 'abcd') for i in range(31)]
        flow = execution_flow(receipt([{'sources': sources[:1], 'returned': sources}]))
        read = flow['reads'][0]
        self.assertEqual(read['queries'][0]['returned_count'], 31)
        self.assertEqual(read['queries'][0]['window_coverage'], 'partial')
        self.assertIsNone(read['returned_not_presented_count'])
        self.assertIsNone(read['returned_exactly_presented_count'])

    def test_subspan_overlap_is_not_falsely_called_dropped(self):
        full = window('doc', 'abcdefghij')
        partial = window('doc', 'cdef', 2)
        read = execution_flow(receipt([{'sources': [partial], 'returned': [full]}]))['reads'][0]
        self.assertEqual(read['returned_partial_overlap_count'], 1)
        self.assertEqual(read['returned_not_presented_count'], 0)
        self.assertEqual(read['returned_exactly_presented_count'], 0)

    def test_window_identity_ignores_id_alias_but_uses_offsets_and_text(self):
        original = window('doc', 'abcd', sid='s1')
        renamed = {**original, 'source_id': 's999'}
        offset = window('doc', 'abcd', start=10)
        changed = window('doc', 'wxyz', start=10)
        flow = execution_flow(receipt([{'sources': [x]} for x in (original, renamed, offset, changed)]))
        self.assertEqual([r['new_source_count'] for r in flow['reads']], [1, 0, 1, 1])
        self.assertEqual(flow['counts']['unique_presented_windows'], 3)

    def test_candidate_reported_and_record_trace_payload_cannot_forge_flow(self):
        row = receipt([{'sources': [window('real', 'abcd')]}])
        expected = execution_flow(row)
        fake = {'trace': [{'name': 'search', 'observed_windows': [], 'observed_window_count': 999}],
                'state': {'sources': [{'source_id': 'forged'}]}, 'host_evidence_trace': {'read_presentations': []},
                'answer': 'PRIVATE_REFERENCE_SENTINEL', 'usage': {'source_chars': 999999}}
        row['candidate_reported'] = fake
        row['trace'].append({'name': 'record_trace', 'request': {'result': fake}})
        self.assertEqual(execution_flow(row), expected)
        self.assertNotIn('PRIVATE_REFERENCE_SENTINEL', json.dumps(execution_flow(row)))

    def test_legacy_answer_observation_is_selected_by_actual_returned_answer(self):
        old, new = window('old', 'abcd'), window('new', 'efgh')
        row = receipt([{'sources': [old, new]}], final_quotes=[quote(old)], legacy=True)
        response = row['host_evidence_trace']['final_observations'][0]['response']
        row['trace'][-1]['response_hash'] = hashlib.sha256(json.dumps(response, sort_keys=True,
            ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        row['host_evidence_trace']['final_observations'].append({
            'evidence': {'e2': quote(new)}, 'response': {'answer': 'abandoned answer', 'citation_ids': ['e2']}})
        final = execution_flow(row)['final']
        self.assertEqual(final['answer_excerpt'], 'synthetic answer')
        self.assertEqual(final['verified_quotes_retained_count'], 1)
        self.assertEqual(final['verified_quotes_not_retained_count'], 1)
        self.assertEqual(final['verified_quotes_cited_count'], 1)

    def test_failed_read_does_not_shift_successful_quote_association(self):
        a, b = window('a', 'abcd'), window('b', 'efgh')
        row = receipt([{'sources': [a]}, {'sources': [b]}])
        row['host_evidence_trace']['read_presentations'].pop(0)
        row['trace'][1]['response_hash'] = sha('unimportant failed hash')
        row['trace'][1]['model_completed'] = False
        flow = execution_flow(row)
        self.assertIsNone(flow['reads'][0]['verified_quote_count'])
        self.assertEqual(flow['reads'][1]['verified_quote_count'], 1)
        self.assertEqual(flow['reads'][1]['quotes'][0]['docid'], 'b')

    def test_complete_counts_and_explicit_omissions_obey_real_utf8_budget(self):
        rounds = []
        for i in range(12):
            sources = [window('DOC'+str(i)+'x'*200, '😀'*300, start=j*1000, sid=str(j)) for j in range(3)]
            rounds.append({'sources': sources, 'queries': ['查询😀\\\"\n'*200 + str(k) for k in range(3)],
                           'quotes': [quote(s, s['text']) for s in sources]})
        row = receipt(rounds)
        before = deepcopy(row)
        flow = execution_flow(row)
        self.assertEqual(row, before)
        self.assertEqual(flow['counts']['search_calls'], 36)
        self.assertEqual(flow['counts']['read_model_calls'], 12)
        self.assertEqual(flow['counts']['presented_source_occurrences'], 36)
        self.assertEqual(flow['counts']['presented_source_chars'], 10800)
        queries = [q for r in flow['reads'] for q in r['queries']] + flow['trailing_queries']
        self.assertEqual(flow['omitted']['queries'], 36-len(queries))
        self.assertEqual(flow['omitted']['reads'], 12-len(flow['reads']))
        quotes = [q for r in flow['reads'] for q in r['quotes']]
        self.assertEqual(flow['omitted']['quote_samples'], 36-len(quotes))
        self.assertEqual(flow['omitted']['quote_chars'], 10800-sum(len(q['quote_excerpt']) for q in quotes))
        self.assertTrue(flow['truncated'])
        self.assertLessEqual(len(json.dumps(flow, ensure_ascii=False, separators=(',', ':')).encode()), FLOW_BOUNDS['json_bytes'])
        self.assertLessEqual(FLOW_BOUNDS['json_bytes'], 6000)
        self.assertTrue(all(len(q['query_excerpt']) <= 320 for q in queries))

    def test_reads_after_selected_final_do_not_inflate_lost_quote_count(self):
        row = receipt([{'sources': [window('before', 'abcd')]}, {'sources': [window('after', 'efgh')]}],
                      final_quotes=[quote(window('before', 'abcd'))])
        # Host event order is read-before, selected answer, then read-after.
        trace = row['trace']
        row['trace'] = trace[:2] + [trace[-1]] + trace[2:-1]
        row['host_evidence_trace']['final_observations'][0]['event_index'] = 2
        row['host_evidence_trace']['read_presentations'][1]['event_index'] = 4
        row = bind_synthetic_origin(row)
        final = execution_flow(row)['final']
        self.assertEqual(final['retention_time_alignment'], 'event_index')
        self.assertEqual(final['verified_read_quotes_before_final_count'], 1)
        self.assertEqual(final['verified_quotes_retained_count'], 1)
        self.assertEqual(final['verified_quotes_not_retained_count'], 0)
        self.assertEqual(final['reads_after_final_count'], 1)

    def test_legacy_retention_requires_matching_final_response_hash(self):
        row = receipt([{'sources': [window('old', 'abcd')]}], legacy=True)
        self.assertIsNone(execution_flow(row)['final']['verified_quotes_retained_count'])
        response = row['host_evidence_trace']['final_observations'][0]['response']
        row['trace'][-1]['response_hash'] = hashlib.sha256(json.dumps(response, sort_keys=True,
            ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        final = execution_flow(row)['final']
        self.assertEqual(final['retention_time_alignment'], 'legacy_response_hash')
        self.assertEqual(final['verified_quotes_retained_count'], 1)

    def test_repeated_returns_are_counted_as_occurrences_not_documents(self):
        source = window('same', 'abcd')
        flow = execution_flow(receipt([{'sources': [], 'returned': [source], 'queries': ['first', 'second']}]))
        self.assertEqual(flow['counts']['search_returned_count'], 2)
        self.assertEqual(flow['reads'][0]['returned_not_presented_count'], 2)
        self.assertEqual(flow['count_units']['returned_counts'], 'backend_window_occurrences_not_unique_documents')

    def test_nonfit_and_unknown_outcomes_cannot_supply_flow(self):
        row = receipt([{'sources': []}])
        for role in ('D_select', 'D_report'):
            with self.subTest(role=role):
                with self.assertRaises(ValueError): execution_flow({**row, 'role': role})
        with self.assertRaises(ValueError): execution_flow({**row, 'provider_outcome': 'unknown'})
        self.assertEqual(execution_flow({'trace': [], 'candidate_reported': row})['status'], 'host_trace_unavailable')


class CompactFlowTests(unittest.TestCase):
    def test_host_metadata_does_not_change_existing_case_selection_or_reward(self):
        rows = [receipt([{'sources': [window('s'+str(i), 'abcd')]}], qid='q'+str(i), score=i/3) for i in range(4)]
        parent = measurement([dict(r, score=1-r['score']) for r in rows], node='parent')
        current = measurement(rows, parent_measurement=parent)
        original = deepcopy(current)
        for group in (current, current['parent_measurement']):
            for r in group['rows']:
                for event in r['trace']:
                    for key in ('observed_windows', 'observed_window_count', 'observed_windows_truncated'):
                        event.pop(key, None)
        old = compact_feedback(current, tasks(rows), max_cases=3)
        new = compact_feedback(original, tasks(rows), max_cases=3)
        self.assertEqual(new['schema'], 'rag-rsi-v3-feedback-3')
        self.assertEqual(new['paired_summary']['min_signed_gain'], -1)
        self.assertEqual(new['cases'][0]['signed_delta'], -1)
        self.assertNotEqual(new['cases'][0]['execution_flow'], old['cases'][0]['execution_flow'])
        for feedback in (old, new):
            for case in feedback['cases']:
                case.pop('execution_flow'); case.pop('parent_execution_flow'); case.pop('parent_flow_pairing')
        self.assertEqual(old, new)

    def test_parent_flow_prefers_same_repeat_then_labels_unpaired(self):
        child = receipt([{'sources': [window('child', 'abcd')]}], score=0, repeat=1)
        parent_rows = [receipt([{'sources': [window('parent-'+str(i), 'abcd')]}], score=1, repeat=i) for i in (0, 1)]
        value = measurement([child], parent_measurement=measurement(parent_rows, node='parent'))
        feedback = compact_feedback(value, tasks([child]))
        case = feedback['cases'][0]
        self.assertEqual(case['parent_flow_pairing']['kind'], 'same_repeat')
        self.assertEqual(case['parent_execution_flow']['reads'][0]['sources'][0]['docid'], 'parent-1')
        value['rows'][0]['repeat'] = 9
        case = compact_feedback(value, tasks([child]))['cases'][0]
        self.assertEqual(case['parent_flow_pairing']['kind'], 'unpaired_representative')
        self.assertEqual(case['parent_flow_pairing']['child_repeat'], 9)

    def test_identity_nonfit_and_unknown_stay_out_of_learning(self):
        row = receipt([{'sources': [window('s', 'abcd')]}])
        current = measurement([row])
        for role in ('D_select', 'D_report'):
            with self.assertRaises(ValueError): compact_feedback({**current, 'role': role}, tasks([row]))
        current['rows'][0]['panel_hash'] = 'wrong-panel'
        with self.assertRaises(ValueError): compact_feedback(current, tasks([row]))
        current['rows'][0].pop('panel_hash')
        current['rows'][0]['error_type'] = 'UnknownProviderOutcome'
        result = compact_feedback(current, tasks([row]))
        self.assertIsNone(result['score'])
        self.assertEqual(result['cases'], [])

    def test_gold_and_arbitrary_task_fields_are_not_sent(self):
        row = receipt([{'sources': [window('s', 'abcd')]}])
        task = {**tasks([row])[0], 'answers': ['PRIVATE_REFERENCE_SENTINEL'], 'reference': 'PRIVATE_REFERENCE_SENTINEL'}
        feedback = compact_feedback(measurement([row]), [task])
        self.assertNotIn('PRIVATE_REFERENCE_SENTINEL', json.dumps(feedback))
        self.assertTrue(feedback['raw_reference_objects_not_sent'])
        self.assertTrue(feedback['fit_feedback_can_reveal_accepted_answers'])
        self.assertNotIn('reference_not_sent', feedback)


if __name__ == '__main__':
    unittest.main()
