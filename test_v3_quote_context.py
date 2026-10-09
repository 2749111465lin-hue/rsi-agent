"""Bounded source-neighbor tests; synthetic models prove contracts, not quality."""
import copy
import hashlib
import json
import unittest

from code_rsi.v3.rag import RagContractError, RagEngine, ground_quote, quote_context
from code_rsi.v3.execution import root_files
from code_rsi.v3.edit_scope import observe_edit_scope
from test_v3_quote_grounding import Backend, Model, QUESTION, read_result, window, model_with_read

TEXT = '标题：Synthetic Atlas\n' + 'L' * 300 + 'The visitor arrived.' + 'R' * 300
QUOTE = 'The visitor arrived.'


def run(radius=None, cap=64000, claim='The visit is model-assessed.'):
    def reader(payload):
        result = read_result([{'source_id': payload['sources'][0]['source_id'], 'quote': QUOTE}])
        result['claims'][0]['text'] = claim
        result['gaps'] = ['Unknown date']
        result['conflicts'] = ['Unresolved fictional contradiction']
        return result
    model = model_with_read(reader)
    config = {'max_payload_chars': cap}
    if radius is not None:
        config['final_context_radius'] = radius
    engine = RagEngine(Backend([window(TEXT, 100)]), model, config=config)
    result = engine.solve({'question': QUESTION})
    return result, model.calls


class QuoteContextTests(unittest.TestCase):
    def setUp(self):
        self.source = {**window(TEXT, 100), 'source_id': 's1'}
        self.quote = ground_quote({'source_id': 's1', 'quote': QUOTE}, self.source)

    def test_exact_unicode_bounds_and_hash_are_recomputed(self):
        context = quote_context(self.quote, {**self.source, 'text_sha256': 'forged'})
        self.assertEqual(context['start'], self.quote['start'] - 256)
        self.assertEqual(context['end'], self.quote['end'] + 256)
        self.assertEqual(context['text'], self.source['text'][context['start']-100:context['end']-100])
        self.assertEqual(context['source_sha256'], hashlib.sha256(TEXT.encode('utf-8')).hexdigest())
        self.assertEqual(self.source['text'], TEXT)

    def test_source_boundaries_clip_without_fetching_hidden_text(self):
        source = {**window('题名 ' + QUOTE + ' End', 120), 'source_id': 's1'}
        context = quote_context({'source_id': 's1', 'quote': QUOTE}, source)
        self.assertEqual((context['start'], context['end'], context['text']),
                         (source['start'], source['end'], source['text']))

    def test_no_extra_text_and_invalid_quotes_have_no_context(self):
        exact = {**window(QUOTE), 'source_id': 's1'}
        self.assertIsNone(quote_context({'source_id': 's1', 'quote': QUOTE}, exact))
        for patch in ({'quote': QUOTE.lower()}, {'start': 1}, {'source_id': 'other'}):
            self.assertIsNone(quote_context({**self.quote, **patch}, self.source))
        for radius in (True, 0, -1, 255, 257, '256', 256.0):
            self.assertIsNone(quote_context(self.quote, self.source, radius))

    def test_default_and_explicit_zero_preserve_every_model_input(self):
        baseline, calls = run()
        zero, zero_calls = run(0)
        self.assertEqual(calls, zero_calls)
        self.assertEqual(baseline, zero)
        self.assertNotIn('context', calls[-1][1]['evidence'][0])
        self.assertFalse(any(event['stage'] == 'final_quote_context' for event in baseline['trace']))

    def test_only_final_evidence_context_changes(self):
        old, old_calls = run(0)
        new, new_calls = run(256)
        self.assertEqual(old_calls[:-1], new_calls[:-1])
        stripped = copy.deepcopy(new_calls[-1][1])
        for item in stripped['evidence']:
            self.assertTrue(item.pop('context'))
        self.assertEqual(stripped, old_calls[-1][1])
        self.assertEqual(old['usage'], new['usage'])
        self.assertEqual(new['correctness'], 'unknown')
        self.assertEqual(new['state']['claims'], old['state']['claims'])
        self.assertEqual(new['state']['conflicts'], old['state']['conflicts'])
        self.assertEqual(new['state']['citations'], new_calls[-1][1]['evidence'])

    def test_context_cannot_displace_existing_evidence_or_judgments(self):
        _, unlimited = run(0, claim='a' * 1400)
        base_size = len(json.dumps(unlimited[-1][1], ensure_ascii=False))
        self.assertGreater(base_size, max(len(json.dumps(p, ensure_ascii=False)) for _, p in unlimited[:-1]))
        baseline, old_calls = run(0, cap=base_size, claim='a' * 1400)
        bounded, new_calls = run(256, cap=base_size, claim='a' * 1400)
        self.assertEqual(old_calls, new_calls)
        self.assertEqual(baseline['state']['citations'], bounded['state']['citations'])
        event = next(e for e in bounded['trace'] if e['stage'] == 'final_quote_context')
        self.assertEqual(event['included_citation_ids'], [])
        self.assertEqual(event['budget_skipped_citation_ids'], ['e1'])
        self.assertNotIn('payload_budget', bounded['failure_types'])

    def test_first_source_context_survives_window_eviction(self):
        first = window('Original title ' + QUOTE + ' First neighbor.', 100, 'first')
        second = window('Second source says it departed.', 900, 'second')
        backend = Backend(by_query={'first': [first], 'follow': [second]})
        def reader(payload):
            source = payload['sources'][0]
            quote = QUOTE if payload['round'] == 1 else 'it departed.'
            return read_result([{'source_id': source['source_id'], 'quote': quote}],
                               queries=['follow'] if payload['round'] == 1 else [],
                               ready=payload['round'] == 2)
        model = model_with_read(reader, initial_query='first')
        result = RagEngine(backend, model, config={'final_context_radius': 256}).solve({'question': QUESTION})
        self.assertEqual([s['docid'] for s in result['state']['sources']], ['second'])
        items = model.calls[-1][1]['evidence']
        self.assertEqual(items[0]['context']['text'], first['text'])
        self.assertEqual(items[1]['context']['text'], second['text'])
        self.assertEqual(len(backend.queries), 2)

    def test_configuration_rejects_implicit_or_arbitrary_radii(self):
        for value in (True, False, 1, 255, 257, '256', 256.0, None):
            with self.subTest(value=value), self.assertRaises(RagContractError):
                run(value if value is not None else -1)

    def test_config_edit_is_evidence_selection_not_declared_answer_module(self):
        receipt = observe_edit_scope(root_files({'final_context_radius': 0}),
                                     root_files({'final_context_radius': 256}), 'answer_generation')
        self.assertEqual(receipt['attribution'], 'single_module')
        self.assertEqual(receipt['associated_module'], 'evidence_selection')
        self.assertTrue(receipt['intent_mismatch'])


if __name__ == '__main__':
    unittest.main()
