"""Synthetic-only adapter tests: no benchmark files, network, or model calls."""
import copy
import json
import unittest

from code_rsi.v3.datasets import (
    DatasetFormatError, adapt_bright, adapt_browsecomp, adapt_multihop,
    adapt_musique, evaluate_answer, filter_documents, normalize_answer,
    validate_public_task, validate_task_collection,
)


def musique_row(qid="2hop_fixture", *, answerable=True):
    return {
        "id": qid, "question": "Which museum holds the blue vase?",
        "answer": "PRIVATE_FINAL", "answer_aliases": ["PRIVATE_ALIAS"],
        "answerable": answerable,
        "question_decomposition": [{"question": "PRIVATE_SUBQUESTION", "answer": "PRIVATE_STEP"}],
        "metadata": {"answer": "PRIVATE_NESTED"},
        "paragraphs": [
            {"idx": idx, "title": f"Public title {idx}", "paragraph_text": f"Public passage {idx}.",
             "is_supporting": idx < 2, "answer": "PRIVATE_PARAGRAPH", "reasoning": "PRIVATE_REASONING"}
            for idx in range(20)
        ],
    }


class DatasetAdapterTests(unittest.TestCase):
    def test_musique_allowlist_removes_all_annotation_paths(self):
        public, private = adapt_musique(musique_row())
        self.assertNotIn("PRIVATE_", json.dumps(public))
        self.assertEqual(private["answers"], ["PRIVATE_FINAL", "PRIVATE_ALIAS"])
        self.assertEqual(len(private["supporting_docids"]), 2)
        self.assertEqual(public["corpus_scope"], "question_local")
        self.assertIsNone(public["corpus_ref"])
        self.assertEqual(len(public["documents"]), 20)

    def test_public_identity_does_not_depend_on_answers_or_labels(self):
        original = musique_row()
        changed = copy.deepcopy(original)
        changed.update(answer="DIFFERENT_SECRET", answerable=False,
                       question_decomposition=[{"answer": "OTHER"}])
        changed["paragraphs"][0]["is_supporting"] = False
        self.assertEqual(adapt_musique(original)[0], adapt_musique(changed)[0])

    def test_full_pair_keeps_source_group_but_has_distinct_context_ids(self):
        positive = musique_row()
        negative = musique_row(answerable=False)
        negative["paragraphs"][0]["paragraph_text"] = "Different public distractor."
        first, first_ref = adapt_musique(positive)
        second, second_ref = adapt_musique(negative)
        self.assertNotEqual(first["question_id"], second["question_id"])
        self.assertEqual(first_ref["pair_group_id"], second_ref["pair_group_id"])
        self.assertFalse(set(d["docid"] for d in first["documents"]) &
                         set(d["docid"] for d in second["documents"]))
        validate_task_collection([first, second])

    def test_local_corpus_cannot_be_pooled_or_replaced(self):
        first, _ = adapt_musique(musique_row("one"))
        second, _ = adapt_musique(musique_row("two"))
        with self.assertRaises(DatasetFormatError):
            filter_documents(first, first["documents"] + second["documents"])
        changed = copy.deepcopy(first["documents"][:1])
        changed[0]["text"] = "Unapproved replacement"
        with self.assertRaises(DatasetFormatError):
            filter_documents(first, changed)
        pooled = copy.deepcopy(first)
        pooled["corpus_ref"] = "shared-pool"
        with self.assertRaises(DatasetFormatError):
            validate_public_task(pooled)

    def test_local_filter_allows_original_subset(self):
        public, _ = adapt_musique(musique_row())
        self.assertEqual(filter_documents(public, public["documents"][:2]), public["documents"][:2])

    def test_exact_twenty_and_unique_paragraph_ids_required(self):
        short = musique_row()
        short["paragraphs"].pop()
        with self.assertRaises(DatasetFormatError):
            adapt_musique(short)
        duplicate = musique_row()
        duplicate["paragraphs"][1]["idx"] = 0
        with self.assertRaises(DatasetFormatError):
            adapt_musique(duplicate)

    def test_private_and_public_outputs_are_detached_from_input(self):
        row = musique_row()
        public, private = adapt_musique(row)
        row["question_decomposition"][0]["answer"] = "CHANGED"
        row["paragraphs"][0]["paragraph_text"] = "CHANGED"
        self.assertEqual(private["question_decomposition"][0]["answer"], "PRIVATE_STEP")
        self.assertEqual(public["documents"][0]["text"], "Public passage 0.")

    def test_browsecomp_preserves_fixed_reference_but_not_labels(self):
        row = {"query_id": "bc-one", "query": "A public question?", "answer": "PRIVATE_ANSWER",
               "gold_docids": ["PRIVATE_GOLD"], "evidence_docids": ["PRIVATE_EVIDENCE"],
               "decomposition": ["PRIVATE_SUBQUERY"], "reasoning": "PRIVATE_REASONING"}
        public, private = adapt_browsecomp(row, "frozen-corpus@sha256:fixture")
        self.assertEqual(public["documents"], [])
        self.assertEqual(public["corpus_ref"], "frozen-corpus@sha256:fixture")
        self.assertNotIn("PRIVATE_", json.dumps(public))
        self.assertEqual(private["official_metric"], "llm_judge")
        self.assertFalse(private["rule_metrics_are_official"])
        self.assertEqual(evaluate_answer("private_answer", private, "em"), 1.0)
        with self.assertRaises(DatasetFormatError):
            evaluate_answer("PRIVATE_ANSWER", private, "official")

    def test_reference_free_public_runtime(self):
        public, private = adapt_browsecomp({"query_id": "unlabelled", "query": "Question?"}, "fixed-corpus")
        validate_public_task(public)
        self.assertEqual(filter_documents(public, [{"docid": "d", "text": "Public document"}]),
                         [{"docid": "d", "text": "Public document"}])
        self.assertFalse(private["reference_available"])
        with self.assertRaises(DatasetFormatError):
            evaluate_answer("guess", private, "em")

    def test_multihop_never_promotes_gold_evidence_to_corpus(self):
        row = {"query": "A question?", "answer": "PRIVATE_ANSWER", "question_type": "null_query",
               "evidence_list": [{"title": "PRIVATE_TITLE", "fact": "PRIVATE_FACT"}]}
        public, private = adapt_multihop(row)
        self.assertEqual(public["documents"], [])
        self.assertEqual(public["corpus_ref"], "multihop-rag:corpus")
        self.assertNotIn("PRIVATE_", json.dumps(public))
        self.assertNotIn("question_type", public)
        self.assertEqual(private["evidence_list"], row["evidence_list"])
        self.assertEqual(adapt_multihop(dict(row, answer="OTHER"))[0], public)

    def test_multihop_explicit_full_documents_are_allowlisted(self):
        row = {"id": "m1", "question": "Question?", "documents": [
            {"docid": "d1", "text": "Corpus material", "is_supporting": True,
             "metadata": {"gold": "PRIVATE"}, "title": "Title"}]}
        public, _ = adapt_multihop(row)
        self.assertEqual(public["documents"], [{"docid": "d1", "text": "Corpus material", "title": "Title"}])
        self.assertIsNone(public["corpus_ref"])

    def test_bright_excluded_ids_are_enforced_and_gold_reasoning_hidden(self):
        row = {"id": "br1", "query": "Find a principle", "excluded_ids": ["excluded"],
               "gold_ids": ["gold"], "gold_answer": "PRIVATE_ANSWER", "reasoning": "PRIVATE_REASONING"}
        public, private = adapt_bright(row, "bright:biology@fixed")
        self.assertEqual(public["task_type"], "retrieval")
        self.assertEqual(public["excluded_docids"], ["excluded"])
        self.assertNotIn("PRIVATE_", json.dumps(public))
        self.assertNotIn("gold_ids", public)
        result = filter_documents(public, [{"docid": "excluded", "text": "Forbidden"},
                                          {"docid": "allowed", "text": "Allowed"}])
        self.assertEqual(result, [{"docid": "allowed", "text": "Allowed"}])
        with self.assertRaises(DatasetFormatError):
            evaluate_answer("any", private, "em")

    def test_excluded_documents_cannot_be_in_public_task(self):
        public, _ = adapt_bright({"id": "b", "query": "Q", "excluded_ids": ["x"]}, "corpus")
        public["documents"] = [{"docid": "x", "text": "excluded"}]
        with self.assertRaises(DatasetFormatError):
            validate_public_task(public)

    def test_question_and_document_collisions_fail(self):
        public, _ = adapt_multihop({"id": "q", "query": "Q"})
        with self.assertRaisesRegex(DatasetFormatError, "duplicate question_id"):
            validate_task_collection([public, copy.deepcopy(public)])
        changed = copy.deepcopy(public)
        changed["question"] = "Other"
        with self.assertRaisesRegex(DatasetFormatError, "conflicting question_id"):
            validate_task_collection([public, changed])
        with self.assertRaises(DatasetFormatError):
            adapt_multihop({"query": "Q", "documents": [{"docid": "d", "text": "one"},
                                                       {"docid": "d", "text": "two"}]})

    def test_same_shared_corpus_rejects_conflicting_document_ids(self):
        one, _ = adapt_multihop({"id": "one", "query": "Q1", "corpus_ref": "same",
                                "documents": [{"docid": "d", "text": "original"}]})
        two, _ = adapt_multihop({"id": "two", "query": "Q2", "corpus_ref": "same",
                                "documents": [{"docid": "d", "text": "changed"}]})
        with self.assertRaisesRegex(DatasetFormatError, "conflicting shared document"):
            validate_task_collection([one, two])
        two["documents"][0]["text"] = "original"
        validate_task_collection([one, two])

    def test_extra_public_fields_fail_closed(self):
        public, _ = adapt_multihop({"query": "Q"})
        public["metadata"] = {"answer": "hidden"}
        with self.assertRaises(DatasetFormatError):
            validate_public_task(public)
        public, _ = adapt_musique(musique_row())
        public["documents"][0]["is_supporting"] = True
        with self.assertRaises(DatasetFormatError):
            validate_public_task(public)

    def test_unknown_or_ambiguous_input_formats_fail(self):
        calls = [lambda: adapt_musique({"prompt": "Q"}),
                 lambda: adapt_browsecomp({"query_id": "q", "prompt": "Q"}, "corpus"),
                 lambda: adapt_multihop({"prompt": "Q"}),
                 lambda: adapt_multihop({"query": "Q", "question": "Different"}),
                 lambda: adapt_bright({"id": "b", "query": "Q"}, "corpus"),
                 lambda: adapt_bright({"id": "b", "query": "Q", "excluded_ids": "x"}, "corpus"),
                 lambda: adapt_multihop({"query": "Q", "documents": "wrong"})]
        for call in calls:
            with self.subTest(call=call), self.assertRaises(DatasetFormatError):
                call()

    def test_rule_em_and_token_f1_normalize_and_use_aliases(self):
        _, ref = adapt_musique(musique_row())
        ref["answers"] = ["The Blue Vase", "azure jar"]
        self.assertEqual(evaluate_answer(" blue vase! ", ref, "em"), 1.0)
        self.assertEqual(evaluate_answer("azure jar", ref, "f1"), 1.0)
        self.assertAlmostEqual(evaluate_answer("blue", ref, "f1"), 2 / 3)
        self.assertEqual(evaluate_answer("", ref, "f1"), 0.0)
        self.assertEqual(normalize_answer(" A, Blue  Vase! "), "blue vase")

    def test_full_unanswerability_is_not_ordinary_answer_em(self):
        _, ref = adapt_musique(musique_row(answerable=False))
        with self.assertRaisesRegex(DatasetFormatError, "paired sufficiency"):
            evaluate_answer("PRIVATE_FINAL", ref, "em")

    def test_invalid_boolean_is_not_coerced(self):
        row = musique_row()
        row["answerable"] = "false"
        with self.assertRaises(DatasetFormatError):
            adapt_musique(row)
        row = musique_row()
        row["paragraphs"][0]["idx"] = True
        with self.assertRaises(DatasetFormatError):
            adapt_musique(row)


if __name__ == "__main__":
    unittest.main()