"""Synthetic host boundaries for exact read-window context; zero network calls."""
from copy import deepcopy
import hashlib
import json
import unittest

from code_rsi.budget import digest
from code_rsi.v3 import execution
from code_rsi.v3.rag import RagEngine, quote_context
import test_v3_quote_grounding as fixture


def context_item(source, *, quote=fixture.QUOTE, start=None):
    item = fixture.evidence(source, quote, start)
    citation = {key: item[key] for key in ('source_id', 'start', 'end', 'quote')}
    item['context'] = quote_context(citation, source)
    return item


def receipt(broker, citations):
    answer = 'Northport'
    origin = broker.answer_origin_receipt(answer)
    cited = broker.citation_receipt(answer, citations)
    return {'schema': execution.EXECUTION_SCHEMA, 'execution_ok': True, 'answer': answer,
            'answer_origin_valid': origin['valid'], 'answer_origin_status': origin['status'],
            'host_answer_origin_validation': origin, 'citation_source_valid': cited['valid'],
            'citations': deepcopy(citations), 'trace': deepcopy(broker.events),
            'host_evidence_trace': {'read_presentations': deepcopy(broker.read_presentations),
                                    'final_observations': deepcopy(broker.final_observations)}}


def replace_final_evidence(record, evidence):
    """Recompute all easy checksums so the provenance check must catch the fault."""
    observed = record['host_evidence_trace']['final_observations'][-1]
    event = record['trace'][observed['event_index']]
    event['request']['payload']['evidence'] = deepcopy(evidence)
    event['payload_sha256'] = digest(event['request']['payload'])
    observed['payload_sha256'] = event['payload_sha256']
    observed['evidence'] = {row['citation_id']: deepcopy(row) for row in evidence}
    record['citations'] = deepcopy(evidence)
    record['host_answer_origin_validation'] = execution._answer_origin_receipt(
        record['answer'], record['host_evidence_trace']['final_observations'],
        record['trace'], execution_ok=True)


class QuoteContextHostTests(unittest.TestCase):
    def ready(self, source_window=None):
        broker, source = fixture.broker_with_quote({'source_id': 's1', 'quote': fixture.QUOTE},
                                                   source_window=source_window)
        return broker, source, context_item(source)

    def finish(self):
        broker, source, item = self.ready()
        broker('complete', {'stage': 'answer', 'payload': {'evidence': [item]}})
        return broker, source, item, receipt(broker, [item])

    def test_valid_context_keeps_exact_quote_and_host_only_source_status(self):
        broker, source, item, record = self.finish()
        self.assertEqual(item['quote'], fixture.QUOTE)
        self.assertEqual(item['context']['source_sha256'], hashlib.sha256(source['text'].encode()).hexdigest())
        self.assertEqual(broker.read_presentations[0]['model_response'],
                         fixture.read_result([{'source_id': 's1', 'quote': fixture.QUOTE}]))
        self.assertTrue(broker.citation_receipt('Northport', [item])['valid'])
        self.assertEqual(broker.citation_receipt('Northport', [item])['semantic_support'], 'model_assessed_only')
        self.assertTrue(execution.validate_answer_origin(json.loads(json.dumps(record)))['valid'])

    def test_context_closed_fields_radius_bounds_and_text_are_exact(self):
        broker, source, item = self.ready()
        original = deepcopy(item['context'])
        variants = [None, {}, {**original, 'extra': 'unchecked'}, {**original, 'radius_chars': 257},
                    {**original, 'radius_chars': 256.0}, {**original, 'start': original['start']-1},
                    {**original, 'end': original['end']+1}, {**original, 'source_start': 0},
                    {**original, 'source_end': original['source_end']+1},
                    {**original, 'text': original['text'].replace('Northport', 'Southport')},
                    {**original, 'source_sha256': '0'*64}]
        for context in variants:
            with self.subTest(context=context), self.assertRaises(execution.CandidateEvidenceError):
                broker('complete', {'stage': 'answer', 'payload': {'evidence': [{**item, 'context': context}]}})
        self.assertEqual(broker.counts['model_calls'], 1)

    def test_only_searching_a_larger_window_cannot_expand_a_shorter_read(self):
        model = fixture.model_with_read(lambda p: fixture.read_result([{'source_id': 's1', 'quote': fixture.QUOTE}]))
        broker = execution.HostBroker(fixture.task(), fixture.Backend(), model)
        full = {**broker('search', {'query': 'Mira', 'limit': 5})[0], 'source_id': 's1'}
        lo = full['text'].index(fixture.QUOTE)
        short = {**full, 'start': full['start']+lo, 'end': full['start']+lo+len(fixture.QUOTE)+1,
                 'text': fixture.QUOTE+' '}
        broker('complete', {'stage': 'read', 'payload': {'sources': [short]}})
        with self.assertRaisesRegex(execution.CandidateEvidenceError, 'read window'):
            broker('complete', {'stage': 'answer', 'payload': {'evidence': [context_item(full)]}})
        good = context_item(short)
        broker('complete', {'stage': 'answer', 'payload': {'evidence': [good]}})
        self.assertTrue(broker.citation_receipt('Northport', [good])['valid'])

    def test_relabeling_source_id_cannot_borrow_another_binding(self):
        broker, source, item = self.ready()
        forged = {**item, 'source_id': 'unread-source'}
        with self.assertRaises(execution.CandidateEvidenceError):
            broker('complete', {'stage': 'answer', 'payload': {'evidence': [forged]}})
        self.assertEqual(broker.counts['model_calls'], 1)

    def test_host_hash_ignores_untrusted_source_hash_metadata(self):
        source_window = {**fixture.window(), 'text_sha256': 'candidate-lie'}
        broker, source, item = self.ready(source_window)
        self.assertNotEqual(item['context']['source_sha256'], source['text_sha256'])
        broker('complete', {'stage': 'answer', 'payload': {'evidence': [item]}})
        self.assertTrue(execution.validate_answer_origin(receipt(broker, [item]))['valid'])

    def test_returned_context_cannot_be_omitted_rewritten_or_reassigned(self):
        broker, source, item, record = self.finish()
        missing = deepcopy(item); missing.pop('context')
        changed = deepcopy(item); changed['context']['text'] += '!'
        reassigned = {**item, 'source_id': 'other'}
        for candidate in (missing, changed, reassigned):
            with self.subTest(candidate=candidate):
                check = broker.citation_receipt('Northport', [candidate])
                self.assertFalse(check['valid'])
                self.assertEqual(check['status'], 'candidate_citation_mismatch')

    def test_context_cannot_be_added_only_to_returned_citation(self):
        broker, source, item = self.ready()
        plain = deepcopy(item); plain.pop('context')
        broker('complete', {'stage': 'answer', 'payload': {'evidence': [plain]}})
        self.assertFalse(broker.citation_receipt('Northport', [item])['valid'])
        self.assertTrue(broker.citation_receipt('Northport', [plain])['valid'])

    def test_cache_requires_recorded_actual_read_response(self):
        broker, source, item, record = self.finish()
        for mutation in ('missing', 'changed', 'failed', 'duplicate_event'):
            cached = deepcopy(record)
            presentation = cached['host_evidence_trace']['read_presentations'][0]
            if mutation == 'missing': presentation.pop('model_response')
            elif mutation == 'changed': presentation['model_response']['claims'] = []
            elif mutation == 'failed': cached['trace'][presentation['event_index']]['model_completed'] = False
            else: cached['host_evidence_trace']['read_presentations'].append(deepcopy(presentation))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                execution.validate_answer_origin(cached)

    def test_cache_regrounds_response_instead_of_trusting_verified_quote_list(self):
        broker, source, item, record = self.finish()
        presentation = record['host_evidence_trace']['read_presentations'][0]
        presentation['model_response']['claims'] = []
        record['trace'][presentation['event_index']]['response_hash'] = digest(presentation['model_response'])
        # verified_quotes and the complete final body are untouched, but the reader
        # no longer returned the quote. Recomputed outer hashes cannot restore it.
        with self.assertRaisesRegex(ValueError, 'earlier verified read-window'):
            execution.validate_answer_origin(record)

    def test_cache_source_must_equal_actual_read_payload(self):
        broker, source, item, record = self.finish()
        record['host_evidence_trace']['read_presentations'][0]['sources'][0]['text'] = 'forged'
        with self.assertRaisesRegex(ValueError, 'actual read request'):
            execution.validate_answer_origin(record)

    def test_context_tamper_fails_even_after_final_hashes_are_recomputed(self):
        broker, source, item, record = self.finish()
        forged = deepcopy(item); forged['context']['text'] = forged['context']['text'].replace('Northport', 'Southport')
        replace_final_evidence(record, [forged])
        with self.assertRaisesRegex(ValueError, 'earlier verified read-window'):
            execution.validate_answer_origin(record)

    def test_context_from_a_later_read_cannot_rewrite_an_earlier_answer(self):
        model = fixture.model_with_read(lambda p: fixture.read_result([{'source_id': 's1', 'quote': fixture.QUOTE}]))
        broker = execution.HostBroker(fixture.task(), fixture.Backend(), model)
        full = {**broker('search', {'query': 'Mira', 'limit': 5})[0], 'source_id': 's1'}
        lo = full['text'].index(fixture.QUOTE)
        short = {**full, 'start': full['start']+lo, 'end': full['start']+lo+len(fixture.QUOTE), 'text': fixture.QUOTE}
        broker('complete', {'stage': 'read', 'payload': {'sources': [short]}})
        plain = fixture.evidence(short)
        broker('complete', {'stage': 'answer', 'payload': {'evidence': [plain]}})
        broker('complete', {'stage': 'read', 'payload': {'sources': [full]}})
        cached = receipt(broker, [plain])
        replace_final_evidence(cached, [context_item(full)])
        with self.assertRaisesRegex(ValueError, 'earlier verified read-window'):
            execution.validate_answer_origin(cached)

    def test_claimed_valid_cached_returned_citations_must_match_context(self):
        broker, source, item, original = self.finish()
        for mutation in ('missing_context', 'empty', 'duplicate'):
            cached = deepcopy(original)
            if mutation == 'missing_context': cached['citations'][0].pop('context')
            elif mutation == 'empty': cached['citations'] = []
            else: cached['citations'].append(deepcopy(cached['citations'][0]))
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'returned quote context'):
                execution.validate_answer_origin(cached)

    def test_no_context_legacy_receipt_needs_no_new_response_field(self):
        broker, source, item = self.ready()
        plain = deepcopy(item); plain.pop('context')
        broker('complete', {'stage': 'answer', 'payload': {'evidence': [plain]}})
        cached = receipt(broker, [plain])
        cached['host_evidence_trace']['read_presentations'][0].pop('model_response')
        self.assertTrue(execution.validate_answer_origin(cached)['valid'])

    def test_root_broker_integration_adds_no_new_model_or_retrieval_calls(self):
        captured = {}
        for radius in (0, 256):
            model = fixture.model_with_read(lambda p: fixture.read_result([{'source_id': 's1', 'quote': fixture.QUOTE}]))
            broker = execution.HostBroker(fixture.task(), fixture.Backend(), model)
            class Backend:
                def search(self, query, limit):
                    return broker('search', {'query': query, 'limit': limit})
            class Model:
                def complete(self, stage, payload):
                    return broker('complete', {'stage': stage, 'payload': payload})
            result = RagEngine(Backend(), Model(), config={'final_context_radius': radius}).solve({'question': fixture.QUESTION})
            returned = [c for c in result['state']['citations'] if c['citation_id'] in result['citation_ids']]
            self.assertTrue(broker.citation_receipt(result['answer'], returned)['valid'])
            self.assertTrue(execution.validate_answer_origin(receipt(broker, returned))['valid'])
            self.assertEqual(broker.counts, {'model_calls': 3, 'search_calls': 1, 'read_calls': 0})
            captured[radius] = [p for stage, p in model.calls if stage == 'answer'][0]
        expected = deepcopy(captured[256]); expected['evidence'][0].pop('context')
        self.assertEqual(captured[0], expected)

    def repeated_window(self, *, later_sid='s1'):
        model = fixture.model_with_read(lambda p: fixture.read_result([
            {'source_id': p['sources'][0]['source_id'], 'quote': fixture.QUOTE}]))
        text = 'L'*350 + fixture.QUOTE + 'R'*350
        broker = execution.HostBroker(fixture.task(), fixture.Backend([fixture.window(text, start=0)]), model)
        full = {**broker('search', {'query': 'Mira', 'limit': 5})[0], 'source_id': 's1'}
        short = {**full, 'start': 340, 'end': 360+len(fixture.QUOTE),
                 'text': full['text'][340:360+len(fixture.QUOTE)]}
        broker('complete', {'stage': 'read', 'payload': {'sources': [short]}})
        later = {**full, 'source_id': later_sid}
        broker('complete', {'stage': 'read', 'payload': {'sources': [later]}})
        return broker, short, later

    def test_repeated_quote_keeps_first_window_even_if_source_id_is_reused_or_changed(self):
        for sid in ('s1', 's2'):
            with self.subTest(sid=sid):
                broker, first, later = self.repeated_window(later_sid=sid)
                with self.assertRaisesRegex(execution.CandidateEvidenceError, 'first verified read window'):
                    broker('complete', {'stage': 'answer', 'payload': {'evidence': [context_item(later)]}})
                item = context_item(first)
                broker('complete', {'stage': 'answer', 'payload': {'evidence': [item]}})
                self.assertTrue(execution.validate_answer_origin(receipt(broker, [item]))['valid'])

    def test_cached_first_window_is_chosen_by_event_order_not_presentation_list_order(self):
        broker, first, later = self.repeated_window()
        item = context_item(first)
        broker('complete', {'stage': 'answer', 'payload': {'evidence': [item]}})
        original = receipt(broker, [item])
        original['host_evidence_trace']['read_presentations'].reverse()
        self.assertTrue(execution.validate_answer_origin(original)['valid'])
        forged = deepcopy(original)
        replace_final_evidence(forged, [context_item(later)])
        with self.assertRaisesRegex(ValueError, 'earlier verified read-window'):
            execution.validate_answer_origin(forged)

    def test_cache_cannot_delete_the_first_read_to_relabel_a_later_window(self):
        broker, first, later = self.repeated_window()
        item = context_item(first)
        broker('complete', {'stage': 'answer', 'payload': {'evidence': [item]}})
        cached = receipt(broker, [item])
        cached['host_evidence_trace']['read_presentations'].pop(0)
        replace_final_evidence(cached, [context_item(later)])
        with self.assertRaisesRegex(ValueError, 'cover completed read events'):
            execution.validate_answer_origin(cached)

    def test_measurement_epoch_explicitly_versions_context_contract(self):
        measure = execution.Measurement(None, 'unused', lambda bank: None)
        self.assertIn('quote-context-1', measure.epoch)


if __name__ == '__main__':
    unittest.main()
