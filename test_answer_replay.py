"""Synthetic answer-only replay checks; no provider, private data or code exec."""
import ast
from copy import deepcopy
import json
import unittest

from code_rsi.answer_replay import (AnswerReplayRouter, project_final_payload,
                                   replay_files)
from code_rsi.budget import digest
from code_rsi.v3.execution import HostBroker, root_files
from code_rsi.v3.infrastructure import ModelResponseError, UnknownProviderOutcome
from code_rsi.v3.rag import RagEngine
from code_rsi.v3.reader_replay import ReplayContractError, ReplayMismatch
from test_v3_reader_replay import dispatch, synthetic_case


HEADERS = {"question", "instructions", "stage_instructions", "additional_guidance", "output_schema"}
DERIVED = {"constraints", "claims", "gaps", "conflicts", "stop_reason", "unresolved_conflict_count"}


class FreshModel:
    def __init__(self):
        self.calls = []
        self.timeout_seconds = 150

    def complete(self, stage, payload):
        self.calls.append((stage, deepcopy(payload)))
        return {"answer": "Northport", "citation_ids": ["e1"], "evidence_sufficient": True}


def finish(router, case, arm):
    for event in case["events"][:-1]:
        dispatch(router, event)
    request = deepcopy(case["events"][-1]["request"])
    request["payload"] = project_final_payload(request["payload"], arm)
    return router.complete(**request)


class AnswerProjectionTests(unittest.TestCase):
    def setUp(self):
        self.case = synthetic_case()
        self.payload = self.case["events"][-1]["request"]["payload"]

    def test_full_state_has_exact_frozen_fields_and_is_an_isolated_copy(self):
        self.assertEqual(set(self.payload), HEADERS | DERIVED | {"evidence"})
        original = deepcopy(self.payload)
        projected = project_final_payload(self.payload, "full_state")
        self.assertEqual(projected, original)
        projected["constraints"].append("extra")
        projected["evidence"][0]["quote"] = "changed"
        projected["output_schema"]["answer"] = "changed"
        self.assertEqual(self.payload, original)

    def test_evidence_only_removes_entire_derived_block_and_preserves_headers(self):
        self.payload.update(constraints=["additional restriction"], claims=[{"text": "claim"}],
                            gaps=["missing"], conflicts=["conflict"], stop_reason="conflict",
                            unresolved_conflict_count=1)
        projected = project_final_payload(self.payload, "evidence_only")
        self.assertEqual(set(projected), HEADERS | {"evidence"})
        for key in HEADERS:
            self.assertEqual(projected[key], self.payload[key])
        self.assertTrue(DERIVED.isdisjoint(projected))

    def test_evidence_order_offsets_and_all_metadata_are_preserved_without_aliases(self):
        original = deepcopy(self.payload)
        projected = project_final_payload(self.payload, "evidence_only")
        allowed = {"citation_id", "source_id", "docid", "start", "end", "quote", "source_verified"}
        self.assertTrue(all(set(row) == allowed for row in projected["evidence"]))
        self.assertEqual(projected["evidence"], original["evidence"])
        self.assertEqual(digest(projected["evidence"]), digest(original["evidence"]))
        projected["evidence"][0]["quote"] = "mutated"
        projected["evidence"].reverse()
        self.assertEqual(self.payload, original)

    def test_evidence_metadata_cannot_reintroduce_removed_judgments(self):
        for arm in ("full_state", "evidence_only"):
            for key in ("gaps", "claims", "semantic_support", "diagnostic_metadata"):
                with self.subTest(arm=arm, hidden=key):
                    payload = deepcopy(self.payload)
                    payload["evidence"][0][key] = {"status": "insufficient evidence"}
                    with self.assertRaises(ReplayContractError):
                        project_final_payload(payload, arm)
            payload = deepcopy(self.payload)
            del payload["evidence"][0]["source_verified"]
            with self.assertRaises(ReplayContractError):
                project_final_payload(payload, arm)

    def test_unknown_and_missing_fields_are_rejected_in_both_conditions(self):
        for arm in ("full_state", "evidence_only"):
            for key in ("future_metadata", "unresolved_constraints", "answer", "reference_answer"):
                with self.subTest(arm=arm, added=key):
                    payload = deepcopy(self.payload)
                    payload[key] = "a hidden derived assertion"
                    with self.assertRaises(ReplayContractError):
                        project_final_payload(payload, arm)
            for key in self.payload:
                with self.subTest(arm=arm, missing=key):
                    payload = deepcopy(self.payload)
                    del payload[key]
                    with self.assertRaises(ReplayContractError):
                        project_final_payload(payload, arm)

    def test_invalid_arm_is_rejected(self):
        for arm in ("", "reader_only", None):
            with self.subTest(arm=arm):
                with self.assertRaises(ReplayContractError):
                    project_final_payload(self.payload, arm)

    def test_generated_files_preserve_original_config_and_engine_without_case_answers(self):
        for arm in ("full_state", "evidence_only"):
            with self.subTest(arm=arm):
                files = replay_files(self.case, arm)
                self.assertEqual(set(files), {"rag.py", "rag_core.py"})
                self.assertEqual(files["rag_core.py"], root_files(self.case["config"])["rag_core.py"])
                parsed = ast.parse(files["rag.py"])
                ast.parse(files["rag_core.py"])
                config_node = next(node for node in parsed.body if isinstance(node, ast.Assign) and
                                   any(isinstance(target, ast.Name) and target.id == "CONFIG" for target in node.targets))
                self.assertEqual(json.loads(config_node.value.args[0].value), self.case["config"])
                self.assertNotIn(self.case["task"]["question"], files["rag.py"])
                self.assertNotIn(self.case["events"][-1]["response"]["answer"], files["rag.py"])
                self.assertNotIn("read_ordinal", files["rag.py"])


class AnswerReplayTests(unittest.TestCase):
    def setUp(self):
        self.case = synthetic_case()

    def test_offline_full_state_reproduces_original_engine_without_capping_rounds(self):
        router = AnswerReplayRouter(self.case)
        result = RagEngine(router, router, config=self.case["config"]).solve({"question": self.case["task"]["question"]})
        self.assertEqual(result["answer"], self.case["events"][-1]["response"]["answer"])
        self.assertEqual(result["failure_types"], [])
        summary = router.assert_complete()
        self.assertEqual(summary["new_model_calls"], 0)
        self.assertEqual(summary["replayed_model_calls"], 4)
        self.assertEqual(summary["replayed_search_calls"], 2)
        self.assertEqual(router.replayed_calls, len(self.case["events"]))

    def test_evidence_only_requires_live_model_and_invalid_models_fail_early(self):
        with self.assertRaises(ReplayContractError):
            AnswerReplayRouter(self.case, arm="evidence_only")
        with self.assertRaises(ReplayContractError):
            AnswerReplayRouter(self.case, object())
        with self.assertRaises(ReplayContractError):
            AnswerReplayRouter(self.case, FreshModel(), arm="invalid")

    def test_both_conditions_replay_all_reads_and_purchase_only_one_fresh_final(self):
        for arm in ("full_state", "evidence_only"):
            with self.subTest(arm=arm):
                frozen = deepcopy(self.case)
                live = FreshModel()
                router = AnswerReplayRouter(self.case, live, arm=arm)
                router.timeout_seconds = 11
                response = finish(router, self.case, arm)
                self.assertEqual(response["answer"], "Northport")
                self.assertNotEqual(response, self.case["events"][-1]["response"])
                self.assertEqual([stage for stage, _ in live.calls], ["answer"])
                expected = project_final_payload(self.case["events"][-1]["request"]["payload"], arm)
                self.assertEqual(live.calls[0][1], expected)
                self.assertEqual(live.timeout_seconds, 11)
                summary = router.assert_complete()
                self.assertEqual(summary["new_model_calls"], 1)
                self.assertEqual(summary["replayed_model_calls"], 3)
                self.assertEqual(summary["replayed_search_calls"], 2)
                self.assertEqual(router.replayed_calls, len(self.case["events"]) - 1)
                self.assertEqual(self.case, frozen)

    def test_condition_identity_differs_and_case_mutation_does_not_change_router(self):
        full = AnswerReplayRouter(self.case, FreshModel(), arm="full_state")
        minimal = AnswerReplayRouter(self.case, FreshModel(), arm="evidence_only")
        self.assertNotEqual(full.identity, minimal.identity)
        self.case["events"][0]["response"]["queries"].append("later mutation")
        self.assertEqual(full.case["events"][0]["response"]["queries"], ["Mira"])
        self.assertEqual(minimal.case["events"][0]["response"]["queries"], ["Mira"])

    def test_prefix_mismatch_is_sticky_and_never_dispatches(self):
        live = FreshModel()
        router = AnswerReplayRouter(self.case, live)
        with self.assertRaises(ReplayMismatch) as first:
            router.search("unexpected", 5)
        with self.assertRaises(ReplayMismatch) as later:
            dispatch(router, self.case["events"][0])
        self.assertIs(first.exception, later.exception)
        self.assertEqual(live.calls, [])
        self.assertEqual(router.new_calls, 0)
        with self.assertRaises(ReplayMismatch):
            router.assert_complete()

    def test_last_reader_must_match_exactly_and_is_never_bought_again(self):
        for key, value in (("gaps", ["different"]), ("constraints", ["different"]),
                           ("additional_guidance", "change reader")):
            with self.subTest(key=key):
                live = FreshModel()
                router = AnswerReplayRouter(self.case, live)
                for event in self.case["events"][:-2]:
                    dispatch(router, event)
                request = deepcopy(self.case["events"][-2]["request"])
                request["payload"][key] = value
                with self.assertRaises(ReplayMismatch):
                    router.complete(**request)
                self.assertEqual(live.calls, [])
                self.assertEqual(router.new_calls, 0)

    def test_final_header_evidence_and_derived_side_channels_are_strictly_frozen(self):
        def evidence_text(payload): payload["evidence"][0]["quote"] = "fabricated"
        def evidence_order(payload): payload["evidence"].reverse()
        def evidence_offset(payload): payload["evidence"][0]["start"] += 1
        changes = [("question", lambda p: p.update(question="different")),
                   ("instructions", lambda p: p.update(instructions="different")),
                   ("additional_guidance", lambda p: p.update(additional_guidance="be confident")),
                   ("stage_instructions", lambda p: p.update(stage_instructions="different")),
                   ("output_schema", lambda p: p.update(output_schema={})),
                   ("evidence_text", evidence_text), ("evidence_order", evidence_order),
                   ("evidence_offset", evidence_offset),
                   ("derived", lambda p: p.update(gaps=["changed or reintroduced"])),
                   ("sidecar", lambda p: p.update(metadata={"conflicts": ["hidden"]}))]
        for arm in ("full_state", "evidence_only"):
            for label, change in changes:
                with self.subTest(arm=arm, change=label):
                    live = FreshModel()
                    router = AnswerReplayRouter(self.case, live, arm=arm)
                    for event in self.case["events"][:-1]:
                        dispatch(router, event)
                    payload = project_final_payload(self.case["events"][-1]["request"]["payload"], arm)
                    change(payload)
                    with self.assertRaises(ReplayMismatch):
                        router.complete("answer", payload)
                    self.assertEqual(live.calls, [])
                    self.assertEqual(router.new_calls, 0)

    def test_rehashed_historical_unknown_final_field_cannot_bypass_projection_contract(self):
        case = deepcopy(self.case)
        case["events"][-1]["request"]["payload"]["hidden_state"] = {"answer": "forbidden"}
        case["original_final_payload_sha256"] = digest(case["events"][-1]["request"]["payload"])
        with self.assertRaises(ReplayContractError):
            AnswerReplayRouter(case, FreshModel())

    def test_completed_format_error_consumes_final_request_without_repurchase(self):
        calls = []
        class Broken:
            def complete(self, stage, payload):
                calls.append(stage)
                raise ModelResponseError("completed invalid final response")
        router = AnswerReplayRouter(self.case, Broken())
        with self.assertRaises(ModelResponseError):
            finish(router, self.case, "full_state")
        self.assertTrue(router.assert_complete()["complete"])
        self.assertEqual(calls, ["answer"])
        with self.assertRaises(ReplayMismatch):
            dispatch(router, self.case["events"][-1])
        self.assertEqual(calls, ["answer"])

    def test_unknown_physical_outcome_remains_fatal_and_cannot_retry(self):
        error = UnknownProviderOutcome("synthetic unknown")
        calls = []
        class Unknown:
            def complete(self, stage, payload):
                calls.append(stage)
                raise error
        router = AnswerReplayRouter(self.case, Unknown())
        with self.assertRaises(UnknownProviderOutcome) as first:
            finish(router, self.case, "full_state")
        with self.assertRaises(UnknownProviderOutcome) as later:
            dispatch(router, self.case["events"][-1])
        self.assertIs(first.exception, error)
        self.assertIs(later.exception, error)
        self.assertEqual(calls, ["answer"])
        with self.assertRaises(UnknownProviderOutcome):
            router.assert_complete()

    def test_broker_validates_fresh_answer_and_quotes_after_evidence_only_projection(self):
        for arm in ("full_state", "evidence_only"):
            with self.subTest(arm=arm):
                live = FreshModel()
                router = AnswerReplayRouter(self.case, live, arm=arm)
                broker = HostBroker(self.case["task"], router, router)
                class Services:
                    def search(self, query, limit=5):
                        return broker("search", {"query": query, "limit": limit})
                    def complete(self, stage, payload):
                        if stage == "answer":
                            payload = project_final_payload(payload, arm)
                        return broker("complete", {"stage": stage, "payload": payload})
                services = Services()
                result = RagEngine(services, services, config=self.case["config"]).solve({"question": self.case["task"]["question"]})
                self.assertEqual(result["answer"], "Northport")
                self.assertTrue(broker.answer_origin_receipt(result["answer"])["valid"])
                self.assertFalse(broker.answer_origin_receipt(self.case["events"][-1]["response"]["answer"])["valid"])
                citations = [row for row in result["state"]["citations"] if row["citation_id"] in result["citation_ids"]]
                self.assertTrue(broker.citation_receipt(result["answer"], citations)["valid"])
                self.assertEqual([stage for stage, _ in live.calls], ["answer"])
                self.assertEqual(router.assert_complete()["new_model_calls"], 1)
                self.assertEqual(len(broker.read_presentations), 2)
                self.assertEqual(len(broker.final_observations), 1)

    def test_outputs_are_isolated_and_extra_calls_fail_after_completion(self):
        router = AnswerReplayRouter(self.case)
        response = dispatch(router, self.case["events"][0])
        response["queries"].append("mutated")
        self.assertEqual(router.case["events"][0]["response"]["queries"], ["Mira"])
        for event in self.case["events"][1:]:
            dispatch(router, event)
        router.assert_complete()
        with self.assertRaises(ReplayMismatch):
            router.search("new", 1)
        self.assertFalse(router.summary()["complete"])


if __name__ == "__main__":
    unittest.main()
