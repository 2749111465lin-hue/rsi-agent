"""Synthetic boundaries for privileged MuSiQue support-material diagnostics."""
from copy import deepcopy
import hashlib
import json
import unittest

from code_rsi.budget import digest
from code_rsi.v3.datasets import adapt_musique
from code_rsi.v3.infrastructure import LocalCorpus
from code_rsi.v3.musique_calibration import (
    ARMS, SCHEMA, MATERIALS_SCHEMA, MaterialError, SupportCorpus,
    build_backends, build_support_materials, validate_support_materials, evidence_stages,
)


def fixture(key="one", count=6, supports=(1, 3)):
    row = {"id": "2hop_" + key, "question": "A public question " + key,
           "answer": "PRIVATE_ANSWER", "answer_aliases": ["PRIVATE_ALIAS"],
           "question_decomposition": [{"answer": "PRIVATE_HOP", "question": "PRIVATE_SUBQUESTION"}],
           "paragraphs": [{"idx": i, "title": "Public title " + key + str(i),
                           "paragraph_text": "Unique public paragraph " + key + str(i),
                           "is_supporting": i in supports} for i in range(count)]}
    task, ref = adapt_musique(row)
    ref["support_annotation_available"] = True
    return task, ref


class SupportMaterialsTests(unittest.TestCase):
    def panel(self):
        pairs = [fixture("one"), fixture("two", supports=(0, 2, 4, 5))]
        return [p[0] for p in pairs], {p[0]["question_id"]: p[1] for p in pairs}

    def test_projection_contains_only_ids_binding_and_preserves_original_order(self):
        tasks, refs = self.panel()
        original = deepcopy((tasks, refs))
        for ref in refs.values():
            ref["supporting_docids"].reverse()
        value = build_support_materials(tasks, refs)
        self.assertEqual(set(value), {"schema", "tasks_sha256", "rows"})
        self.assertEqual(value["schema"], MATERIALS_SCHEMA)
        self.assertEqual(value["tasks_sha256"], digest(tasks))
        self.assertNotIn("PRIVATE_", json.dumps(value))
        docs = validate_support_materials(value, tasks)
        for task, row in zip(tasks, value["rows"]):
            expected = [d for d in task["documents"] if d["docid"] in refs[task["question_id"]]["supporting_docids"]]
            self.assertEqual(row, {"question_id": task["question_id"], "docids": [d["docid"] for d in expected]})
            self.assertEqual(docs[task["question_id"]], expected)
        self.assertEqual(tasks, original[0])
        docs[tasks[0]["question_id"]][0]["text"] = "mutated copy"
        self.assertNotIn("mutated copy", json.dumps(tasks))

    def test_builder_never_accesses_answers_or_decomposition(self):
        tasks, refs = self.panel()
        class Guarded(dict):
            def __getitem__(self, key):
                if key in {"answers", "answer", "answer_aliases", "question_decomposition", "answerable"}:
                    raise AssertionError("private answer field accessed")
                return super().__getitem__(key)
            def get(self, key, default=None):
                if key in {"answers", "answer", "answer_aliases", "question_decomposition", "answerable"}:
                    raise AssertionError("private answer field accessed")
                return super().get(key, default)
        guarded = {qid: Guarded(ref) for qid, ref in refs.items()}
        self.assertEqual(build_support_materials(tasks, guarded), build_support_materials(tasks, refs))

    def test_annotation_answer_changes_do_not_change_materials(self):
        tasks, refs = self.panel()
        other = deepcopy(refs)
        for ref in other.values():
            ref.update(answers=["DIFFERENT_PRIVATE"], question_decomposition=[{"answer": "DIFFERENT"}])
        self.assertEqual(build_support_materials(tasks, refs), build_support_materials(tasks, other))

    def test_reference_coverage_identity_and_explicit_unavailability_rejected(self):
        tasks, refs = self.panel()
        qid = tasks[0]["question_id"]
        variants = []
        value = deepcopy(refs); value.pop(qid); variants.append(value)
        value = deepcopy(refs); value["extra"] = value[qid]; variants.append(value)
        for key, changed in [("question_id", "other"), ("dataset", "browsecomp-plus"),
                             ("support_annotation_available", False), ("support_annotation_available", None)]:
            value = deepcopy(refs); value[qid][key] = changed; variants.append(value)
        for value in variants:
            with self.subTest(keys=list(value)), self.assertRaises(MaterialError):
                build_support_materials(tasks, value)

    def test_material_root_and_row_shapes_are_allowlisted(self):
        tasks, refs = self.panel(); value = build_support_materials(tasks, refs)
        variants = []
        bad = deepcopy(value); bad["answer"] = "PRIVATE"; variants.append(bad)
        bad = deepcopy(value); bad["schema"] = "other"; variants.append(bad)
        bad = deepcopy(value); bad["rows"][0]["text"] = "not allowed"; variants.append(bad)
        bad = deepcopy(value); bad["rows"] = {}; variants.append(bad)
        bad = deepcopy(value); bad["rows"] = bad["rows"][:1]; variants.append(bad)
        bad = deepcopy(value); bad["rows"][1] = deepcopy(bad["rows"][0]); variants.append(bad)
        for bad in variants:
            with self.subTest(keys=list(bad)), self.assertRaises(MaterialError):
                validate_support_materials(bad, tasks)

    def test_support_count_duplicates_unknown_foreign_and_order_rejected(self):
        tasks, refs = self.panel(); value = build_support_materials(tasks, refs)
        qid = tasks[0]["question_id"]; ids = [d["docid"] for d in tasks[0]["documents"]]
        candidates = [[], ids[:1], ids[:5], [ids[0], ids[0]], [ids[0], "missing"],
                      [ids[0], tasks[1]["documents"][0]["docid"]], [ids[3], ids[1]], [True, ids[1]]]
        for selected in candidates:
            bad = deepcopy(value); bad["rows"][0]["docids"] = selected
            with self.subTest(selected=selected), self.assertRaises(MaterialError):
                validate_support_materials(bad, tasks)
        bad = deepcopy(value); bad["rows"][0]["question_id"] = "foreign"
        with self.assertRaises(MaterialError): validate_support_materials(bad, tasks)

    def test_stale_public_task_content_or_order_binding_rejected(self):
        tasks, refs = self.panel(); value = build_support_materials(tasks, refs)
        for field in ["question", "documents"]:
            changed = deepcopy(tasks)
            if field == "question": changed[0][field] += " changed"
            else: changed[0][field][0]["text"] += " changed"
            with self.subTest(field=field), self.assertRaises(MaterialError):
                validate_support_materials(value, changed)
        with self.assertRaises(MaterialError): validate_support_materials(value, list(reversed(tasks)))

    def test_excluded_and_non_musique_contexts_rejected(self):
        tasks, refs = self.panel(); value = build_support_materials(tasks, refs)
        changed = deepcopy(tasks)
        changed[0]["excluded_docids"] = [value["rows"][0]["docids"][0]]
        with self.assertRaises(MaterialError): validate_support_materials(value, changed)
        changed = deepcopy(tasks); changed[0]["dataset"] = "multihop-rag"
        with self.assertRaises(MaterialError): build_support_materials(changed, refs)
        with self.assertRaises(MaterialError): build_support_materials([], {})

    def test_support_corpus_returns_complete_text_query_independently(self):
        task, ref = fixture()
        task["documents"][1]["text"] = " 星\n" + "long public text " * 700 + "end"
        value = build_support_materials([task], {task["question_id"]: ref})
        docs = validate_support_materials(value, [task])[task["question_id"]]
        backend = SupportCorpus(docs, scope=task["question_id"])
        first = backend.search("unrelated query", limit=2)
        self.assertEqual(first, backend.search("different words", limit=30))
        for row, doc in zip(first, docs):
            self.assertEqual(row["docid"], doc["docid"])
            self.assertEqual(row["text"], doc["text"])
            self.assertEqual((row["start"], row["end"]), (0, len(doc["text"])))
            self.assertEqual(row["document_hash"], hashlib.sha256(doc["text"].encode()).hexdigest())
            self.assertEqual(backend.read(row["docid"], 0, row["end"])["text"], doc["text"])
            self.assertNotIn("is_supporting", row)
        self.assertGreater(len(first[0]["text"]), 5000)
        first[0]["text"] = "external mutation"
        self.assertNotEqual(backend.search("again")[0]["text"], "external mutation")
        docs[0]["text"] = "input mutation"
        self.assertNotEqual(backend.search("again")[0]["text"], "input mutation")

    def test_support_corpus_rejects_partial_delivery_and_invalid_limits(self):
        task, ref = fixture(supports=(0, 1, 2, 3))
        value = build_support_materials([task], {task["question_id"]: ref})
        corpus = SupportCorpus(validate_support_materials(value, [task])[task["question_id"]], scope=task["question_id"])
        for limit in [0, 1, 3, 31, True, 4.0]:
            with self.subTest(limit=limit), self.assertRaises(MaterialError): corpus.search("query", limit)
        for query in ["", "   ", None, 4]:
            with self.subTest(query=query), self.assertRaises(MaterialError): corpus.search(query)
        foreign, _ = fixture("foreign")
        with self.assertRaises(ValueError): corpus.read(foreign["documents"][0]["docid"], 0, 1)
        with self.assertRaises(ValueError): corpus.read(task["documents"][5]["docid"], 0, 1)

    def test_support_backend_identity_is_versioned_and_distinct_from_normal(self):
        tasks, refs = self.panel(); materials = build_support_materials(tasks, refs)
        docs = validate_support_materials(materials, tasks)[tasks[0]["question_id"]]
        one = SupportCorpus(docs, scope=tasks[0]["question_id"])
        self.assertNotEqual(one.identity, LocalCorpus(docs, scope=tasks[0]["question_id"]).identity)
        self.assertEqual(one.identity, SupportCorpus(deepcopy(docs), scope=tasks[0]["question_id"]).identity)
        self.assertNotEqual(one.identity, SupportCorpus(list(reversed(docs)), scope=tasks[0]["question_id"]).identity)
        changed = deepcopy(docs); changed[0]["text"] += " changed"
        self.assertNotEqual(one.identity, SupportCorpus(changed, scope=tasks[0]["question_id"]).identity)

    def test_support_constructor_rejects_foreign_private_and_duplicate_records(self):
        task, ref = fixture(); materials = build_support_materials([task], {task["question_id"]: ref})
        docs = validate_support_materials(materials, [task])[task["question_id"]]
        variants = [[], docs[:1], [docs[0], docs[0]]]
        bad = deepcopy(docs); bad[0]["answer"] = "PRIVATE"; variants.append(bad)
        bad = deepcopy(docs); bad[0]["docid"] = "foreign/p/0"; variants.append(bad)
        bad = deepcopy(docs); bad[0]["text"] = ""; variants.append(bad)
        for bad in variants:
            with self.subTest(count=len(bad)), self.assertRaises(MaterialError):
                SupportCorpus(bad, scope=task["question_id"])

    def test_normal_backends_share_full_context_and_diagnostic_is_separate(self):
        tasks, refs = self.panel(); materials = build_support_materials(tasks, refs)
        backends, panel = build_backends(tasks, materials)
        self.assertEqual(ARMS, {"planned": "planned_single", "loop": "iterative", "support": "planned_single"})
        self.assertEqual(len(backends), 6)
        for task in tasks:
            qid = task["question_id"]
            self.assertIs(backends[(qid, "planned")], backends[(qid, "loop")])
            self.assertEqual(set(backends[(qid, "planned")].docs), {d["docid"] for d in task["documents"]})
            self.assertEqual(set(backends[(qid, "support")].docs), set(refs[qid]["supporting_docids"]))
            self.assertNotEqual(backends[(qid, "planned")].identity, backends[(qid, "support")].identity)
        self.assertEqual(panel, build_backends(tasks, materials)[1])
        changed = deepcopy(materials)
        changed["rows"][0]["docids"] = [d["docid"] for d in tasks[0]["documents"][:2]]
        other, other_panel = build_backends(tasks, changed)
        qid = tasks[0]["question_id"]
        self.assertEqual(other[(qid, "planned")].identity, backends[(qid, "planned")].identity)
        self.assertNotEqual(other[(qid, "support")].identity, backends[(qid, "support")].identity)
        self.assertNotEqual(panel, other_panel)
        expected = [{"question_id": q, "arm": arm, "backend": b.identity} for (q, arm), b in backends.items()]
        expected.sort(key=lambda item: (item["question_id"], item["arm"]))
        self.assertEqual(panel, digest({"schema": SCHEMA, "backend_identities": expected}))

class EvidenceStagesTests(unittest.TestCase):
    def receipt(self):
        task, ref = fixture()
        docs = {row['docid']: row for row in task['documents']}
        first, second = ref['supporting_docids']
        def source(docid, complete):
            text = docs[docid]['text']
            text = text if complete else text[:5]
            return {'docid': docid, 'start': 0, 'end': len(text), 'text': text,
                    'text_sha256': hashlib.sha256(text.encode()).hexdigest()}
        sources = [source(first, True), source(second, False)]
        quote = {'docid': first, 'start': 0, 'end': 5, 'quote': docs[first]['text'][:5]}
        windows = [{'docid': docid} for docid in [first, second, task['documents'][0]['docid']]]
        receipt = {'question_id': task['question_id'], 'execution_ok': True,
                   'resource_usage': {'search_calls': 1, 'model_calls': 2},
                   'trace': [{'name': 'search', 'observed_windows': windows,
                              'observed_window_count': 3, 'observed_windows_truncated': False},
                             {'name': 'complete', 'request': {'stage': 'read', 'payload': {}}, 'model_completed': True},
                             {'name': 'complete', 'request': {'stage': 'answer', 'payload': {}}, 'model_completed': True}],
                   'host_evidence_trace': {'read_presentations': [
                       {'event_index': 1, 'sources': sources, 'verified_quotes': [quote, deepcopy(quote)]}],
                       'final_observations': [{'event_index': 2, 'evidence': {'E1': quote}}]}}
        return receipt, task, ref

    def test_stage_sets_deduplicate_and_full_text_is_separate_from_document_hit(self):
        receipt, task, ref = self.receipt()
        before = deepcopy((receipt, task, ref))
        value = evidence_stages(receipt, task, ref)
        self.assertTrue(value['document_membership_only'])
        self.assertFalse(value['semantic_support_verified'])
        expected = {'retrieval': (3, 2, 1.0), 'presented': (2, 2, 1.0),
                    'quoted': (1, 1, .5), 'final_seen': (1, 1, .5),
                    'full_support_presented': (1, 1, .5)}
        for name, (count, hits, recall) in expected.items():
            self.assertEqual(value['stages'][name], {'complete': True, 'document_count': count,
                             'matched_support_count': hits, 'support_recall': recall})
        self.assertEqual((receipt, task, ref), before)
        output = json.dumps(value)
        self.assertNotIn('PRIVATE', output)
        self.assertNotIn(task['question_id'], output)
        self.assertNotIn('Public title', output)

    def test_failed_execution_is_unknown_at_every_stage_not_prefix_zero(self):
        receipt, task, ref = self.receipt()
        receipt['execution_ok'] = False
        result = evidence_stages(receipt, task, ref)
        self.assertFalse(result['execution_complete'])
        for row in result['stages'].values():
            self.assertEqual(row, {'complete': False, 'document_count': None,
                                  'matched_support_count': None, 'support_recall': None})

    def test_search_missing_truncated_count_mismatch_and_invalid_ids_are_unknown(self):
        receipt, task, ref = self.receipt()
        variants = []
        bad = deepcopy(receipt); bad['trace'][0].pop('observed_windows'); variants.append(bad)
        bad = deepcopy(receipt); bad['trace'][0]['observed_windows_truncated'] = True; variants.append(bad)
        bad = deepcopy(receipt); bad['trace'][0]['observed_window_count'] = 4; variants.append(bad)
        bad = deepcopy(receipt); bad['resource_usage']['search_calls'] = 2; variants.append(bad)
        bad = deepcopy(receipt); bad['trace'][0]['observed_windows'][0]['docid'] = []; variants.append(bad)
        for bad in variants:
            value = evidence_stages(bad, task, ref)['stages']
            self.assertIsNone(value['retrieval']['support_recall'])
            self.assertTrue(value['presented']['complete'])

    def test_completed_empty_search_read_and_quote_are_zero_but_no_answer_is_unknown(self):
        receipt, task, ref = self.receipt()
        receipt.update(trace=[], resource_usage={'search_calls': 0, 'model_calls': 0},
                       host_evidence_trace={'read_presentations': [], 'final_observations': []})
        value = evidence_stages(receipt, task, ref)['stages']
        for name in ('retrieval', 'presented', 'quoted', 'full_support_presented'):
            self.assertEqual(value[name]['support_recall'], 0.0)
            self.assertTrue(value[name]['complete'])
        self.assertIsNone(value['final_seen']['support_recall'])

    def test_missing_read_or_model_logs_cannot_look_like_zero_quotes(self):
        receipt, task, ref = self.receipt()
        for mutate in ('missing_reads', 'missing_quote_list', 'wrong_event_index', 'missing_call'):
            bad = deepcopy(receipt)
            if mutate == 'missing_reads': bad['host_evidence_trace'].pop('read_presentations')
            elif mutate == 'missing_quote_list': bad['host_evidence_trace']['read_presentations'][0].pop('verified_quotes')
            elif mutate == 'wrong_event_index': bad['host_evidence_trace']['read_presentations'][0]['event_index'] = 0
            else: bad['resource_usage']['model_calls'] = 3
            value = evidence_stages(bad, task, ref)['stages']
            self.assertIsNone(value['quoted']['support_recall'])
            self.assertTrue(value['retrieval']['complete'])

    def test_truncated_read_response_has_no_verified_quotes_and_unknown_presentation(self):
        receipt, task, ref = self.receipt()
        receipt['trace'][1]['model_completed'] = False
        receipt['host_evidence_trace']['read_presentations'] = []
        value = evidence_stages(receipt, task, ref)['stages']
        self.assertIsNone(value['presented']['support_recall'])
        self.assertIsNone(value['full_support_presented']['support_recall'])
        self.assertEqual(value['quoted']['support_recall'], 0.0)

    def test_final_seen_uses_last_successful_answer_input_not_output_citations(self):
        receipt, task, ref = self.receipt()
        receipt['citations'] = [{'docid': ref['supporting_docids'][1]}]
        receipt['trace'].append({'name': 'complete', 'request': {'stage': 'answer', 'payload': {}}, 'model_completed': True})
        receipt['resource_usage']['model_calls'] = 3
        receipt['host_evidence_trace']['final_observations'].append({'event_index': 3, 'evidence': {}})
        value = evidence_stages(receipt, task, ref)['stages']
        self.assertEqual(value['quoted']['support_recall'], .5)
        self.assertEqual(value['final_seen']['support_recall'], 0.0)
        receipt['host_evidence_trace']['final_observations'].pop()
        self.assertIsNone(evidence_stages(receipt, task, ref)['stages']['final_seen']['support_recall'])

    def test_full_material_requires_one_exact_window_not_hash_only_or_fragment_union(self):
        receipt, task, ref = self.receipt()
        original = deepcopy(receipt['host_evidence_trace']['read_presentations'][0]['sources'][0])
        for field, value in [('text', original['text'] + ' changed'), ('start', True),
                             ('end', original['end'] - 1), ('text_sha256', '0' * 64)]:
            bad = deepcopy(receipt)
            bad['host_evidence_trace']['read_presentations'][0]['sources'][0][field] = value
            self.assertEqual(evidence_stages(bad, task, ref)['stages']['full_support_presented']['support_recall'], 0.0)
        cut = len(original['text']) // 2
        fragments = []
        for lo, hi in [(0, cut), (cut, len(original['text']))]:
            fragment = deepcopy(original); fragment.update(start=lo, end=hi, text=original['text'][lo:hi])
            fragment['text_sha256'] = hashlib.sha256(fragment['text'].encode()).hexdigest()
            fragments.append(fragment)
        receipt['host_evidence_trace']['read_presentations'][0]['sources'] = fragments
        self.assertEqual(evidence_stages(receipt, task, ref)['stages']['full_support_presented']['support_recall'], 0.0)

    def test_private_answers_never_accessed_and_wrong_task_reference_rejected(self):
        receipt, task, ref = self.receipt()
        class Guarded(dict):
            def get(self, key, default=None):
                if key in {'answers', 'answer', 'answer_aliases', 'question_decomposition', 'answerable'}:
                    raise AssertionError('answer field accessed')
                return super().get(key, default)
            def __getitem__(self, key):
                if key in {'answers', 'answer', 'answer_aliases', 'question_decomposition', 'answerable'}:
                    raise AssertionError('answer field accessed')
                return super().__getitem__(key)
        self.assertEqual(evidence_stages(receipt, task, Guarded(ref)), evidence_stages(receipt, task, ref))
        wrong = deepcopy(ref); wrong['question_id'] = 'foreign'
        with self.assertRaises(MaterialError): evidence_stages(receipt, task, wrong)
        receipt['question_id'] = 'foreign'
        with self.assertRaises(MaterialError): evidence_stages(receipt, task, ref)


if __name__ == "__main__":
    unittest.main()
