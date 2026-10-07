"""Offline mechanism and safety tests; no provider or external data is used."""
import copy
import json
import unittest

from code_rsi.v3.rag import RagContractError, RagEngine


QUESTION = "Where was the discoverer of the Aster comet born?"


def read_result(claims=None, **updates):
    value = {"claims": claims or [], "bridge_entities": [], "gaps": [],
             "conflicts": [], "queries": [], "ready": False}
    value.update(updates)
    return value


def quote_claim(source, text=None):
    return {"text": text or source["text"], "citations": [{
        "source_id": source["source_id"], "start": source["start"],
        "end": source["end"], "quote": source["text"]}]}


class TwoHopBackend:
    def __init__(self):
        self.queries = []

    def search(self, query, limit):
        self.queries.append(query)
        if "mira vale" in query.casefold():
            text, docid, start = "Mira Vale was born in Northport.", "biography", 100
        else:
            text, docid, start = "The Aster comet was discovered by Mira Vale.", "discovery", 50
        return [{"docid": docid, "text": text, "start": start,
                 "end": start + len(text), "score": 1}][:limit]


class TwoHopModel:
    """Deterministic fixture for orchestration, never evidence of LLM quality."""
    def __init__(self):
        self.calls = []

    def complete(self, stage, payload):
        self.calls.append((stage, copy.deepcopy(payload)))
        if stage == "plan":
            return {"constraints": ["Identify discoverer and then birthplace"],
                    "queries": ["Aster comet discoverer"]}
        if stage == "read":
            sources = payload["sources"]
            found = any("born in Northport" in item["text"] for item in sources)
            return read_result([quote_claim(item) for item in sources],
                               bridge_entities=["Mira Vale"],
                               gaps=[] if found else ["Birthplace of Mira Vale"],
                               queries=[] if found else ["Mira Vale birthplace"], ready=found)
        for item in payload["evidence"]:
            if "born in Northport" in item["quote"]:
                return {"answer": "Northport", "citation_ids": [item["citation_id"]],
                        "evidence_sufficient": True}
        return {"answer": "Insufficient information", "citation_ids": [],
                "evidence_sufficient": False}


class ScriptedModel:
    def __init__(self, callback):
        self.callback, self.calls = callback, []

    def complete(self, stage, payload):
        self.calls.append((stage, copy.deepcopy(payload)))
        return self.callback(stage, payload)


def model_with_read(read_callback, final_callback=None):
    def callback(stage, payload):
        if stage == "plan":
            return {"constraints": [], "queries": [QUESTION]}
        if stage == "read":
            return read_callback(payload)
        if final_callback:
            return final_callback(payload)
        return {"answer": "Insufficient information", "citation_ids": [],
                "evidence_sufficient": False}
    return ScriptedModel(callback)


def fixture_comparison():
    task = {"task_id": "synthetic-two-hop", "question": QUESTION}
    return {mode: RagEngine(TwoHopBackend(), TwoHopModel(), config={"mode": mode}).solve(task)
            for mode in ("single_pass", "iterative")}


class RagEngineTests(unittest.TestCase):
    def test_second_hop_is_driven_by_read_entity(self):
        backend, model = TwoHopBackend(), TwoHopModel()
        result = RagEngine(backend, model).solve({"question": QUESTION})
        self.assertEqual(backend.queries, ["Aster comet discoverer", "Mira Vale birthplace"])
        self.assertEqual(result["answer"], "Northport")
        self.assertTrue(result["answer_usable"])
        self.assertTrue(result["citations_valid"])
        self.assertTrue(result["model_claims_evidence"])
        self.assertEqual(result["correctness"], "unknown")
        self.assertEqual(result["usage"]["model_calls"], 4)
        self.assertEqual(result["state"]["citations"][1]["start"], 100)
        self.assertEqual(result["failure_types"], [])
        json.dumps(result, allow_nan=False)

    def test_single_pass_and_loop_on_identical_question(self):
        results = fixture_comparison()
        self.assertEqual(results["single_pass"]["answer"], "Insufficient information")
        self.assertEqual(results["iterative"]["answer"], "Northport")
        self.assertEqual(results["single_pass"]["usage"]["search_calls"], 1)
        self.assertEqual(results["iterative"]["usage"]["search_calls"], 2)

    def test_reference_fields_are_rejected_before_backend_or_model(self):
        backend, model = TwoHopBackend(), TwoHopModel()
        for key in ("gold", "reference_answer", "answers", "metadata", "reference"):
            with self.subTest(key=key), self.assertRaises(RagContractError):
                RagEngine(backend, model).solve({"question": QUESTION, key: "secret"})
        self.assertEqual(backend.queries, [])
        self.assertEqual(model.calls, [])

    def test_task_id_never_enters_model_payload(self):
        model = TwoHopModel()
        result = RagEngine(TwoHopBackend(), model).solve({"question": QUESTION, "task_id": "local-only"})
        self.assertEqual(result["task_id"], "local-only")
        self.assertNotIn("local-only", json.dumps(model.calls))

    def test_quote_forgery_never_becomes_evidence(self):
        def read(payload):
            claim = quote_claim(payload["sources"][0])
            claim["citations"][0]["quote"] = "A completely fabricated birthplace."
            return read_result([claim], ready=True)
        result = RagEngine(TwoHopBackend(), model_with_read(read)).solve({"question": QUESTION})
        self.assertIn("invalid_quote", result["failure_types"])
        self.assertEqual(result["state"]["citations"], [])
        self.assertEqual(result["state"]["claims"], [])

    def test_wrong_offsets_and_unknown_source_are_rejected(self):
        for change in ({"start": 0}, {"source_id": "s999"}, {"source_id": []}, {"end": True}):
            def read(payload):
                claim = quote_claim(payload["sources"][0])
                claim["citations"][0].update(change)
                return read_result([claim], ready=True)
            with self.subTest(change=change):
                result = RagEngine(TwoHopBackend(), model_with_read(read)).solve({"question": QUESTION})
                self.assertEqual(result["state"]["citations"], [])
                self.assertIn("invalid_quote", result["failure_types"])

    def test_answer_survives_invalid_citation(self):
        model = model_with_read(lambda p: read_result([quote_claim(p["sources"][0])], ready=True),
                                lambda p: {"answer": "Northport", "citation_ids": ["e999"],
                                           "evidence_sufficient": True})
        result = RagEngine(TwoHopBackend(), model).solve({"question": QUESTION})
        self.assertEqual(result["answer"], "Northport")
        self.assertTrue(result["answer_usable"])
        self.assertFalse(result["citations_valid"])
        self.assertIn("invalid_answer_citation", result["failure_types"])
        self.assertEqual(result["correctness"], "unknown")

    def test_answer_survives_malformed_citation_list(self):
        model = model_with_read(lambda p: read_result(ready=True),
                                lambda p: {"answer": "Northport", "citation_ids": {"x": "e1"},
                                           "evidence_sufficient": True})
        result = RagEngine(TwoHopBackend(), model).solve({"question": QUESTION})
        self.assertEqual(result["answer"], "Northport")
        self.assertTrue(result["answer_usable"])
        self.assertFalse(result["citations_valid"])

    def test_repeated_query_stops_without_duplicate_search(self):
        backend = TwoHopBackend()
        model = model_with_read(lambda p: read_result([quote_claim(p["sources"][0])],
                                                      queries=["  " + QUESTION.upper() + "  "]))
        result = RagEngine(backend, model).solve({"question": QUESTION})
        self.assertEqual(result["stop_reason"], "repeated_queries")
        self.assertEqual(len(backend.queries), 1)
        self.assertEqual(result["usage"]["final_calls"], 1)

    def test_query_churn_without_evidence_progress_stops(self):
        model = model_with_read(lambda p: read_result(queries=["different words"] ))
        result = RagEngine(TwoHopBackend(), model).solve({"question": QUESTION})
        self.assertEqual(result["stop_reason"], "no_evidence_progress")
        self.assertEqual(result["usage"]["search_calls"], 1)

    def test_ungrounded_bridge_is_rejected(self):
        model = model_with_read(lambda p: read_result(bridge_entities=["Invented Person"], queries=["Invented Person birthplace"]))
        result = RagEngine(TwoHopBackend(), model).solve({"question": QUESTION})
        self.assertEqual(result["state"]["bridge_entities"], [])
        self.assertIn("ungrounded_bridge_entity", result["failure_types"])
        self.assertEqual(result["stop_reason"], "no_evidence_progress")

    def test_final_call_is_reserved_at_tight_budget(self):
        model = TwoHopModel()
        result = RagEngine(TwoHopBackend(), model, config={"max_model_calls": 2}).solve({"question": QUESTION})
        self.assertEqual([stage for stage, _ in model.calls], ["plan", "answer"])
        self.assertEqual(result["stop_reason"], "final_reserved")
        self.assertEqual(result["usage"]["model_calls"], 2)
        self.assertEqual(result["usage"]["final_calls"], 1)

    def test_read_schema_failure_still_finishes(self):
        model = model_with_read(lambda p: {"claims": "bad"})
        result = RagEngine(TwoHopBackend(), model).solve({"question": QUESTION})
        self.assertIn("model_or_schema_failure", result["failure_types"])
        self.assertEqual(result["usage"]["final_calls"], 1)
        self.assertEqual(result["stop_reason"], "read_failure")

    def test_deep_read_schema_failure_is_atomic(self):
        def read(payload):
            first = quote_claim(payload["sources"][0])
            return read_result([first, {"text": "bad", "citations": "not-list"}])
        result = RagEngine(TwoHopBackend(), model_with_read(read)).solve({"question": QUESTION})
        self.assertIn("read_schema_failure", result["failure_types"])
        self.assertEqual(result["state"]["citations"], [])

    def test_truncated_read_is_not_used(self):
        def read(payload):
            value = read_result([quote_claim(payload["sources"][0])], ready=True)
            value["_meta"] = {"finish_reason": "length"}
            return value
        result = RagEngine(TwoHopBackend(), model_with_read(read)).solve({"question": QUESTION})
        self.assertIn("truncated_response", result["failure_types"])
        self.assertEqual(result["state"]["citations"], [])
        self.assertEqual(result["usage"]["final_calls"], 1)

    def test_final_truncation_has_no_usable_answer(self):
        model = model_with_read(lambda p: read_result(ready=True),
                                lambda p: {"answer": "Northport", "citation_ids": [],
                                           "evidence_sufficient": True, "_meta": {"truncated": True}})
        result = RagEngine(TwoHopBackend(), model).solve({"question": QUESTION})
        self.assertFalse(result["answer_usable"])
        self.assertIsNone(result["answer"])
        self.assertIn("truncated_response", result["failure_types"])

    def test_conflict_is_preserved_and_not_verified_away(self):
        model = model_with_read(lambda p: read_result([quote_claim(p["sources"][0])],
                                                      conflicts=["Sources disagree about birthplace"], ready=True),
                                lambda p: {"answer": "Northport", "citation_ids": ["e1"],
                                           "evidence_sufficient": True})
        result = RagEngine(TwoHopBackend(), model).solve({"question": QUESTION})
        self.assertTrue(result["answer_usable"])
        self.assertTrue(result["citations_valid"])
        self.assertEqual(result["evidence_status"], "model_assessed_conflicted")
        self.assertIn("model_claims_support_despite_conflict", result["failure_types"])
        final = model.calls[-1][1]
        self.assertEqual(final["conflicts"], ["Sources disagree about birthplace"])
        self.assertEqual(result["correctness"], "unknown")

    def test_source_clipping_changes_presented_offset_boundary(self):
        def read(payload):
            source = payload["sources"][0]
            self.assertTrue(source["source_truncated"])
            self.assertEqual(source["end"], source["start"] + 12)
            return read_result([quote_claim(source)], ready=True)
        result = RagEngine(TwoHopBackend(), model_with_read(read),
                           config={"max_source_chars": 12}).solve({"question": QUESTION})
        self.assertEqual(result["state"]["citations"][0]["quote"], "The Aster co")

    def test_stage_prompt_override_cannot_replace_schema(self):
        model = TwoHopModel()
        result = RagEngine(TwoHopBackend(), model,
                           config={"prompts": {"read": "Prefer explicit dates."}}).solve({"question": QUESTION})
        read_payload = next(payload for stage, payload in model.calls if stage == "read")
        self.assertEqual(read_payload["additional_guidance"], "Prefer explicit dates.")
        self.assertIn("citations", read_payload["output_schema"]["claims"][0])
        self.assertEqual(result["answer"], "Northport")

    def test_backend_failure_is_contained(self):
        class BrokenBackend:
            def search(self, query, limit):
                raise RuntimeError("do not retain this private transport detail")
        result = RagEngine(BrokenBackend(), TwoHopModel()).solve({"question": QUESTION})
        self.assertIn("backend_failure", result["failure_types"])
        self.assertEqual(result["usage"]["final_calls"], 1)
        self.assertNotIn("private transport", json.dumps(result))

    def test_optional_backend_read(self):
        class ReadBackend:
            def search(self, query, limit):
                return [{"docid": "document", "start": 10, "end": 15}]
            def read(self, docid, start, end):
                return "hello"
        model = model_with_read(lambda p: read_result([quote_claim(p["sources"][0])], ready=True))
        result = RagEngine(ReadBackend(), model).solve({"question": QUESTION})
        self.assertEqual(result["usage"]["read_calls"], 1)
        self.assertEqual(result["state"]["citations"][0]["quote"], "hello")

    def test_model_cannot_mutate_host_sources_or_schema(self):
        def read(payload):
            source = payload["sources"][0]
            source["text"] = "Forged source text"
            source["end"] = source["start"] + len(source["text"])
            payload["output_schema"].clear()
            return read_result([quote_claim(source)], ready=True)
        result = RagEngine(TwoHopBackend(), model_with_read(read)).solve({"question": QUESTION})
        self.assertIn("invalid_quote", result["failure_types"])
        self.assertEqual(result["state"]["sources"][0]["text"], "The Aster comet was discovered by Mira Vale.")
        self.assertEqual(result["state"]["citations"], [])
        self.assertEqual(RagEngine(TwoHopBackend(), TwoHopModel()).solve({"question": QUESTION})["answer"], "Northport")

    def test_final_context_reservation_compacts_whole_claims(self):
        def read(payload):
            claims = [quote_claim(payload["sources"][0], str(index) + " " + "x" * 1200)
                      for index in range(10)]
            return read_result(claims, ready=True)
        model = model_with_read(read, lambda p: {"answer": "Mira Vale", "citation_ids": ["e1"],
                                                "evidence_sufficient": True})
        result = RagEngine(TwoHopBackend(), model, config={"max_payload_chars": 5000}).solve({"question": QUESTION})
        self.assertEqual(result["usage"]["final_calls"], 1)
        self.assertTrue(result["answer_usable"])
        self.assertTrue(result["citations_valid"])
        event = next(item for item in result["trace"] if item["stage"] == "final_context_compaction")
        self.assertGreater(event["omitted"]["claims"], 0)
        final = model.calls[-1][1]
        self.assertLessEqual(len(json.dumps(final, ensure_ascii=False)), 5000)
        self.assertEqual(final["evidence"][0]["quote"], "The Aster comet was discovered by Mira Vale.")

    def test_rephrased_claims_without_new_source_evidence_are_not_progress(self):
        def read(payload):
            return read_result([quote_claim(payload["sources"][0], "paraphrase " + str(payload["round"]))],
                               queries=["new query number " + str(payload["round"])])
        result = RagEngine(TwoHopBackend(), model_with_read(read)).solve({"question": QUESTION})
        self.assertEqual(result["stop_reason"], "no_evidence_progress")
        self.assertEqual(result["state"]["rounds"], 2)
        self.assertEqual(len(result["state"]["sources"]), 1)
    def test_configuration_rejects_one_call_and_unknown_mode(self):
        for config in ({"max_model_calls": 1}, {"mode": "anything"}, {"gold": "leak"}):
            with self.subTest(config=config), self.assertRaises(RagContractError):
                RagEngine(TwoHopBackend(), TwoHopModel(), config=config)


if __name__ == "__main__":
    unittest.main()