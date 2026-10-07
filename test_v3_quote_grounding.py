"""Offline exact quote grounding and bounded zero-hit recovery checks.

All names, texts and model decisions below are synthetic fixtures. These tests
measure interface behavior only, not real model or benchmark improvement.
"""
import copy
import json
import unittest

from code_rsi.v3.datasets import adapt_multihop
from code_rsi.v3.execution import CandidateEvidenceError, HostBroker
from code_rsi.v3.rag import RagContractError, RagEngine, ground_quote


QUESTION = "Where was Mira Vale born?"
TEXT = "前缀：Mira Vale was born in Northport. 后缀"
QUOTE = "Mira Vale was born in Northport."


def window(text=TEXT, start=120, docid="synthetic-bio"):
    return {"docid": docid, "text": text, "start": start, "end": start + len(text)}


def read_result(citations=None, *, queries=None, ready=True):
    return {"claims": [{"text": "Birthplace claim", "citations": citations}] if citations else [],
            "bridge_entities": [], "gaps": [], "conflicts": [],
            "queries": queries or [], "ready": ready}


class Backend:
    def __init__(self, rows=None, by_query=None):
        self.rows = [window()] if rows is None else rows
        self.by_query, self.queries = by_query, []

    def search(self, query, limit=5):
        self.queries.append(query)
        return copy.deepcopy((self.by_query.get(query, []) if self.by_query is not None
                              else self.rows)[:limit])


class Model:
    def __init__(self, callback):
        self.callback, self.calls = callback, []

    def complete(self, stage, payload):
        self.calls.append((stage, copy.deepcopy(payload)))
        return self.callback(stage, payload)


def model_with_read(reader, *, initial_query="initial query", answer_ids=None):
    def callback(stage, payload):
        if stage == "plan":
            return {"constraints": [], "queries": [initial_query]}
        if stage == "read":
            return reader(payload)
        ids = ([item["citation_id"] for item in payload.get("evidence", [])]
               if answer_ids is None else answer_ids)
        return {"answer": "Northport" if ids else "Insufficient information",
                "citation_ids": ids, "evidence_sufficient": bool(ids)}
    return Model(callback)


def task():
    public, _ = adapt_multihop({"id": "synthetic-quote-question", "query": QUESTION,
                                "answer": "Northport", "documents": []})
    return public


def broker_with_quote(citation, *, source_window=None):
    model = model_with_read(lambda payload: read_result([copy.deepcopy(citation)]))
    broker = HostBroker(task(), Backend([source_window or window()]), model)
    source = {**broker("search", {"query": "Mira", "limit": 5})[0], "source_id": "s1"}
    broker("complete", {"stage": "read", "payload": {"sources": [source]}})
    return broker, source


def evidence(source, quote=QUOTE, start=None):
    start = source["start"] + source["text"].index(quote) if start is None else start
    return {"citation_id": "e1", "source_id": source["source_id"], "docid": source["docid"],
            "start": start, "end": start + len(quote), "quote": quote}


class ExactQuoteGroundingTests(unittest.TestCase):
    def setUp(self):
        self.source = {**window(), "source_id": "s1"}
        self.raw = {"source_id": "s1", "quote": QUOTE}

    def test_unique_quote_gets_absolute_unicode_character_offsets(self):
        original = copy.deepcopy(self.raw)
        self.assertEqual(ground_quote(self.raw, self.source),
                         {**self.raw, "start": 123, "end": 123 + len(QUOTE)})
        self.assertEqual(self.raw, original)

    def test_original_explicit_offset_contract_remains_valid(self):
        cited = {**self.raw, "start": 123, "end": 123 + len(QUOTE)}
        self.assertEqual(ground_quote(cited, self.source), cited)

    def test_wrong_explicit_offsets_never_fall_back_to_unique_quote(self):
        for change in ({"start": 120, "end": 120 + len(QUOTE)},
                       {"start": 123}, {"end": 123 + len(QUOTE)},
                       {"start": None, "end": None}, {"start": True, "end": 154}):
            with self.subTest(change=change):
                self.assertIsNone(ground_quote({**self.raw, **change}, self.source))

    def test_ambiguous_and_overlapping_quotes_require_explicit_disambiguation(self):
        for text, quote in (("Town Town", "Town"), ("aaa", "aa")):
            source = {**window(text), "source_id": "s1"}
            raw = {"source_id": "s1", "quote": quote}
            with self.subTest(text=text):
                self.assertIsNone(ground_quote(raw, source))
                explicit = {**raw, "start": 120, "end": 120 + len(quote)}
                self.assertEqual(ground_quote(explicit, source), explicit)

    def test_no_fuzzy_case_or_whitespace_normalization(self):
        for quote in (QUOTE.lower(), QUOTE.replace(" ", "  "), "North Port", ""):
            with self.subTest(quote=quote):
                self.assertIsNone(ground_quote({**self.raw, "quote": quote}, self.source))

    def test_bad_source_identity_and_untrusted_shapes_rejected(self):
        for raw in ({**self.raw, "source_id": "other"}, {**self.raw, "source_id": []},
                    {**self.raw, "docid": "forged"}, [], None):
            with self.subTest(raw=raw):
                self.assertIsNone(ground_quote(raw, self.source))
        self.assertIsNone(ground_quote(self.raw, {**self.source, "end": 900}))

    def test_core_answer_uses_canonical_quote_first_evidence(self):
        model = model_with_read(lambda payload: read_result([self.raw]))
        result = RagEngine(Backend(), model).solve({"question": QUESTION})
        self.assertEqual(result["answer"], "Northport")
        self.assertTrue(result["citations_valid"])
        self.assertEqual(result["state"]["citations"][0]["start"], 123)
        self.assertEqual(result["correctness"], "unknown")
        self.assertEqual(result["state"]["claims"][0]["support_status"], "model_assessed")
        json.dumps(result, allow_nan=False)

    def test_bad_quote_does_not_erase_usable_final_answer(self):
        raw = {**self.raw, "start": 120, "end": 120 + len(QUOTE)}
        model = model_with_read(lambda payload: read_result([raw]), answer_ids=["e1"])
        result = RagEngine(Backend(), model).solve({"question": QUESTION})
        self.assertEqual(result["answer"], "Northport")
        self.assertTrue(result["answer_usable"])
        self.assertFalse(result["citations_valid"])
        self.assertIn("invalid_quote", result["failure_types"])
        self.assertEqual(result["state"]["citations"], [])


class HostQuoteBoundaryTests(unittest.TestCase):
    def test_quote_first_host_final_chain_is_verified(self):
        broker, source = broker_with_quote({"source_id": "s1", "quote": QUOTE})
        item = evidence(source)
        self.assertEqual(broker.verified_read_quotes, {(source["docid"], 123, 123 + len(QUOTE), QUOTE)})
        broker("complete", {"stage": "answer", "payload": {"evidence": [item]}})
        receipt = broker.citation_receipt("Northport", [item])
        self.assertTrue(receipt["valid"])
        self.assertEqual(receipt["semantic_support"], "model_assessed_only")

    def test_core_and_trusted_broker_agree_end_to_end(self):
        model = model_with_read(lambda payload: read_result([{"source_id": "s1", "quote": QUOTE}]))
        broker = HostBroker(task(), Backend(), model)

        class BrokerBackend:
            def search(self, query, limit):
                return broker("search", {"query": query, "limit": limit})

        class BrokerModel:
            def complete(self, stage, payload):
                return broker("complete", {"stage": stage, "payload": payload})

        result = RagEngine(BrokerBackend(), BrokerModel()).solve({"question": QUESTION})
        self.assertTrue(result["citations_valid"])
        receipt = broker.citation_receipt(result["answer"], result["state"]["citations"])
        self.assertTrue(receipt["valid"])
        self.assertEqual(broker.counts["model_calls"], 3)

    def test_host_rejects_bad_explicit_or_ambiguous_quote_even_if_candidate_repairs_it(self):
        for source_window, raw, quote in (
                (window(), {"source_id": "s1", "quote": QUOTE, "start": 120, "end": 120 + len(QUOTE)}, QUOTE),
                (window("Town Town"), {"source_id": "s1", "quote": "Town"}, "Town"),
                (window("aaa"), {"source_id": "s1", "quote": "aa"}, "aa")):
            with self.subTest(raw=raw):
                broker, source = broker_with_quote(raw, source_window=source_window)
                self.assertEqual(broker.verified_read_quotes, set())
                with self.assertRaises(CandidateEvidenceError):
                    broker("complete", {"stage": "answer", "payload": {"evidence": [evidence(source, quote)]}})
                self.assertEqual(broker.counts["model_calls"], 1)

    def test_host_does_not_accept_quote_outside_current_presented_window(self):
        model = model_with_read(lambda payload: read_result([{"source_id": "s1", "quote": QUOTE}]))
        broker = HostBroker(task(), Backend(), model)
        full = {**broker("search", {"query": "Mira", "limit": 5})[0], "source_id": "s1"}
        partial = {**full, "text": full["text"][:3], "end": full["start"] + 3}
        broker("complete", {"stage": "read", "payload": {"sources": [partial]}})
        self.assertEqual(broker.verified_read_quotes, set())
        with self.assertRaises(CandidateEvidenceError):
            broker("complete", {"stage": "answer", "payload": {"evidence": [evidence(full)]}})

    def test_uniqueness_is_checked_only_in_this_presented_window(self):
        model = model_with_read(lambda payload: read_result([{"source_id": "s1", "quote": "Town"}]))
        broker = HostBroker(task(), Backend([window("Town Town")]), model)
        full = broker("search", {"query": "Town", "limit": 5})[0]
        partial = {**full, "source_id": "s1", "text": "Town", "start": 125, "end": 129}
        broker("complete", {"stage": "read", "payload": {"sources": [partial]}})
        self.assertEqual(broker.verified_read_quotes, {(full["docid"], 125, 129, "Town")})

    def test_host_allows_correct_explicit_disambiguation(self):
        broker, source = broker_with_quote({"source_id": "s1", "quote": "Town", "start": 125, "end": 129},
                                           source_window=window("Town Town"))
        item = evidence(source, "Town", 125)
        broker("complete", {"stage": "answer", "payload": {"evidence": [item]}})
        self.assertTrue(broker.citation_receipt("Northport", [item])["valid"])

    def test_final_candidate_cannot_forge_offsets_or_omit_canonical_span(self):
        broker, source = broker_with_quote({"source_id": "s1", "quote": QUOTE})
        item = evidence(source)
        for forged in ({**item, "start": 120, "end": 120 + len(QUOTE)},
                       {k: v for k, v in item.items() if k not in {"start", "end"}}):
            with self.subTest(forged=forged), self.assertRaises(CandidateEvidenceError):
                broker("complete", {"stage": "answer", "payload": {"evidence": [forged]}})
        broker("complete", {"stage": "answer", "payload": {"evidence": [item]}})
        self.assertFalse(broker.citation_receipt("Northport", [{**item, "start": 120}])["valid"])


class StagnationWindowTests(unittest.TestCase):
    def recovery(self, config):
        backend = Backend(by_query={"rewritten birthplace query": [window()]})

        def reader(payload):
            if not payload["sources"]:
                return read_result(queries=["rewritten birthplace query"], ready=False)
            return read_result([{"source_id": payload["sources"][0]["source_id"], "quote": QUOTE}])

        model = model_with_read(reader, initial_query="no hits query")
        return RagEngine(backend, model, config=config).solve({"question": QUESTION}), backend, model

    def test_initial_zero_hits_can_recover_after_model_query_rewrite(self):
        result, backend, model = self.recovery({"max_stagnant_rounds": 2})
        self.assertEqual(backend.queries, ["no hits query", "rewritten birthplace query"])
        self.assertEqual(result["answer"], "Northport")
        self.assertTrue(result["citations_valid"])
        reads = [p for stage, p in model.calls if stage == "read"]
        self.assertEqual(reads[0]["sources"], [])
        self.assertEqual(reads[1]["stagnation"], {"consecutive_rounds": 1, "stop_after": 2})
        self.assertEqual(result["state"]["consecutive_stagnant_rounds"], 0)
        self.assertEqual(result["usage"]["final_calls"], 1)

    def test_default_preserves_immediate_stagnation_protocol(self):
        result, backend, _ = self.recovery({})
        self.assertEqual(backend.queries, ["no hits query"])
        self.assertEqual(result["stop_reason"], "no_evidence_progress")
        self.assertEqual(result["answer"], "Insufficient information")

    def test_single_pass_stays_single_pass_with_larger_window(self):
        result, backend, _ = self.recovery({"mode": "single_pass", "max_stagnant_rounds": 2})
        self.assertEqual(len(backend.queries), 1)
        self.assertEqual(result["stop_reason"], "single_pass")

    def test_novel_empty_queries_stop_at_bounded_stagnation_window(self):
        model = model_with_read(lambda p: read_result(queries=["rewrite " + str(p["round"])], ready=False))
        result = RagEngine(Backend([]), model,
                           config={"max_stagnant_rounds": 2, "max_rounds": 20}).solve({"question": QUESTION})
        self.assertEqual(result["stop_reason"], "no_evidence_progress")
        self.assertEqual(result["usage"]["search_calls"], 2)
        self.assertEqual(result["usage"]["model_calls"], 4)
        self.assertEqual(result["usage"]["final_calls"], 1)

    def test_real_new_evidence_resets_consecutive_stagnation(self):
        backend = Backend(by_query={"q2": [window()], "q4": [window("A second grounded fact.", docid="second")]})

        def reader(payload):
            current = payload["round"]
            claims = []
            if current in {2, 4}:
                source = payload["sources"][-1]
                claims = [{"source_id": source["source_id"], "quote": source["text"]}]
            return read_result(claims, queries=["q" + str(current + 1)], ready=current == 4)

        model = model_with_read(reader, initial_query="q1")
        result = RagEngine(backend, model,
                           config={"max_stagnant_rounds": 2, "max_rounds": 4}).solve({"question": QUESTION})
        self.assertEqual(backend.queries, ["q1", "q2", "q3", "q4"])
        self.assertEqual(result["state"]["consecutive_stagnant_rounds"], 0)
        self.assertEqual(len(result["state"]["citations"]), 2)

    def test_repeated_query_and_final_reservation_override_stagnation_window(self):
        model = model_with_read(lambda p: read_result(queries=["initial query"], ready=False))
        result = RagEngine(Backend([]), model, config={"max_stagnant_rounds": 20}).solve({"question": QUESTION})
        self.assertEqual(result["stop_reason"], "repeated_queries")
        self.assertEqual(result["usage"]["search_calls"], 1)
        model = model_with_read(lambda p: read_result(queries=["rewrite " + str(p["round"])], ready=False))
        result = RagEngine(Backend([]), model,
                           config={"max_stagnant_rounds": 20, "max_model_calls": 3}).solve({"question": QUESTION})
        self.assertEqual(result["stop_reason"], "final_reserved")
        self.assertEqual([stage for stage, _ in model.calls], ["plan", "read", "answer"])
        self.assertEqual(result["usage"]["final_calls"], 1)

    def test_stagnation_window_requires_positive_integer(self):
        for value in (0, -1, True, 1.5, None):
            with self.subTest(value=value), self.assertRaises(RagContractError):
                RagEngine(Backend(), model_with_read(lambda p: read_result()),
                          config={"max_stagnant_rounds": value})


if __name__ == "__main__":
    unittest.main()
