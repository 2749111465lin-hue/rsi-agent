"""Archived-program final replay contracts; synthetic data, no candidate exec."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from code_rsi.archive import ProgramArchive
from code_rsi.answer_replay import AnswerReplayRouter, ProgramAnswerReplayRouter
from code_rsi.budget import digest
from code_rsi.v3.execution import HostBroker, execute, root_files
from code_rsi.v3.infrastructure import ModelResponseError, UnknownProviderOutcome
from code_rsi.v3.reader_replay import ReplayContractError, ReplayMismatch
from test_v3_reader_replay import dispatch, synthetic_case


class RecordingAnswer:
    def __init__(self, error=None):
        self.calls = []
        self.error = error
        self.timeout_seconds = 150
        self.response = {"answer": "Northport", "citation_ids": ["e1"], "evidence_sufficient": True}

    def complete(self, stage, payload):
        self.calls.append((stage, deepcopy(payload)))
        if self.error is not None:
            raise self.error
        return self.response


def prefix(router, case):
    for event in case["events"][:-1]:
        dispatch(router, event)


class ProgramAnswerReplayTests(unittest.TestCase):
    def setUp(self):
        self.case = synthetic_case()
        self.payload = deepcopy(self.case["events"][-1]["request"]["payload"])
        self.payload["stage_instructions"] = "Check each relation before choosing the final entity."
        self.payload["bridge_entities"] = ["Mira"]

    def test_constructor_separates_capture_from_measurement_and_requires_explicit_payload(self):
        for kwargs in ({}, {"final_payload": self.payload}, {"capture": 1},
                       {"capture": True, "live_model": RecordingAnswer()},
                       {"capture": True, "final_payload": self.payload},
                       {"live_model": RecordingAnswer()}, {"live_model": object(), "final_payload": self.payload}):
            with self.subTest(kwargs=tuple(kwargs)), self.assertRaises(ReplayContractError):
                ProgramAnswerReplayRouter(self.case, **kwargs)
        self.assertFalse(ProgramAnswerReplayRouter(self.case, capture=True).measurement_eligible)

    def test_frozen_final_payload_requires_original_question_and_finite_json(self):
        for payload in (None, [], {}, {"question": "different"},
                        {**self.payload, "bad": float("nan")},
                        {**self.payload, "bad": float("inf")}, {**self.payload, "bad": {1, 2}}):
            with self.subTest(payload_type=type(payload).__name__), self.assertRaises(ReplayContractError):
                ProgramAnswerReplayRouter(self.case, RecordingAnswer(), final_payload=payload)

    def test_capture_records_actual_changed_payload_without_host_engine_or_provider(self):
        with patch("code_rsi.answer_replay.RagEngine", side_effect=AssertionError("host engine execution")):
            router = ProgramAnswerReplayRouter(self.case, capture=True)
            prefix(router, self.case)
            response = router.complete("answer", self.payload)
        self.assertEqual(router.captured_payload, self.payload)
        self.assertEqual(response, self.case["events"][-1]["response"])
        self.assertTrue(router.final_response_replayed)
        self.assertFalse(router.measurement_eligible)
        self.assertEqual(router.new_calls, 0)
        self.assertEqual(router.replayed_calls, len(self.case["events"]))
        self.assertEqual(router.assert_complete()["replayed_model_calls"], 4)

    def test_payload_case_capture_and_response_snapshots_do_not_alias(self):
        router = ProgramAnswerReplayRouter(self.case, capture=True)
        prefix(router, self.case)
        response = router.complete("answer", self.payload)
        frozen = deepcopy(self.payload)
        self.payload["bridge_entities"].append("later")
        router.captured_payload["bridge_entities"].append("other")
        response["answer"] = "changed"
        self.assertEqual(router.captured_payload, frozen)
        self.assertNotEqual(router.case["events"][-1]["response"]["answer"], "changed")
        live = RecordingAnswer()
        measured = ProgramAnswerReplayRouter(self.case, live, final_payload=frozen)
        frozen["bridge_entities"].append("mutated")
        measured.final_payload["bridge_entities"].append("mutated again")
        prefix(measured, self.case)
        actual = measured.complete("answer", measured.final_payload)
        actual["answer"] = "changed"
        self.assertEqual(measured.final_payload["bridge_entities"], ["Mira"])
        self.assertEqual(live.response["answer"], "Northport")

    def test_identity_binds_capture_mode_case_and_every_final_payload_field(self):
        live = RecordingAnswer()
        left = ProgramAnswerReplayRouter(self.case, live, final_payload=self.payload)
        same = ProgramAnswerReplayRouter(self.case, RecordingAnswer(), final_payload=deepcopy(self.payload))
        other = ProgramAnswerReplayRouter(self.case, live, final_payload={**self.payload, "bridge_entities": []})
        capture = ProgramAnswerReplayRouter(self.case, capture=True)
        self.assertEqual(left.identity, same.identity)
        self.assertEqual(len({left.identity, other.identity, capture.identity}), 3)
        altered = deepcopy(self.case)
        altered["case_id"] += "_other"
        self.assertNotEqual(left.identity, ProgramAnswerReplayRouter(altered, live, final_payload=self.payload).identity)

    def test_live_replays_all_upstream_calls_and_dispatches_only_one_fresh_answer(self):
        live = RecordingAnswer()
        router = ProgramAnswerReplayRouter(self.case, live, final_payload=self.payload)
        router.timeout_seconds = 11
        prefix(router, self.case)
        self.assertEqual(live.calls, [])
        response = router.complete("answer", deepcopy(self.payload))
        self.assertNotEqual(response, self.case["events"][-1]["response"])
        self.assertEqual(live.calls, [("answer", self.payload)])
        self.assertEqual(live.timeout_seconds, 11)
        self.assertFalse(router.final_response_replayed)
        self.assertIsNone(router.measurement_eligible)
        self.assertEqual(router.assert_complete(), {"replayed_model_calls": 3, "new_model_calls": 1,
                         "replayed_search_calls": 2, "replayed_reads": 0, "complete": True})

    def test_every_upstream_request_is_exact_and_mismatch_is_sticky_before_dispatch(self):
        for index, event in enumerate(self.case["events"][:-1]):
            with self.subTest(index=index):
                live = RecordingAnswer()
                router = ProgramAnswerReplayRouter(self.case, live, final_payload=self.payload)
                for preceding in self.case["events"][:index]:
                    dispatch(router, preceding)
                wrong = deepcopy(event)
                if wrong["name"] == "complete":
                    wrong["request"]["payload"]["unexpected"] = True
                else:
                    wrong["request"]["query"] += " changed"
                with self.assertRaises(ReplayMismatch) as first:
                    dispatch(router, wrong)
                with self.assertRaises(ReplayMismatch) as later:
                    dispatch(router, event)
                self.assertIs(first.exception, later.exception)
                with self.assertRaises(ReplayMismatch): router.assert_complete()
                self.assertEqual(live.calls, [])

    def test_source_read_rpc_reuses_exact_span_and_returns_an_isolated_response(self):
        case = deepcopy(self.case)
        window = deepcopy(case["events"][1]["response"][0])
        event = {"name": "read", "request": {key: window[key] for key in ("docid", "start", "end")},
                 "response": window, "response_sha256": digest(window)}
        case["events"].insert(2, event)
        live = RecordingAnswer()
        router = ProgramAnswerReplayRouter(case, live, final_payload=self.payload)
        for preceding in case["events"][:2]: dispatch(router, preceding)
        returned = dispatch(router, event)
        returned["text"] = "changed outside router"
        self.assertEqual(router.case["events"][2]["response"], window)
        for following in case["events"][3:-1]: dispatch(router, following)
        router.complete("answer", self.payload)
        self.assertEqual(router.assert_complete()["replayed_reads"], 1)
        self.assertEqual(len(live.calls), 1)
        denied = RecordingAnswer()
        other = ProgramAnswerReplayRouter(case, denied, final_payload=self.payload)
        for preceding in case["events"][:2]: dispatch(other, preceding)
        with self.assertRaises(ReplayMismatch):
            other.read(window["docid"], window["start"], window["end"] + 1)
        self.assertEqual(denied.calls, [])

    def test_capture_also_rejects_changed_prefix_and_early_answer(self):
        for capture in (False, True):
            with self.subTest(capture=capture):
                live = None if capture else RecordingAnswer()
                router = ProgramAnswerReplayRouter(self.case, live, capture=capture,
                                                  final_payload=None if capture else self.payload)
                with self.assertRaises(ReplayMismatch):
                    router.complete("answer", self.payload)
                self.assertIsNone(router.captured_payload)
                self.assertFalse(router.final_response_replayed)
                if live is not None: self.assertEqual(live.calls, [])

    def test_final_body_changes_reject_before_live_dispatch_including_new_fields(self):
        alterations = [lambda p: p.update(question="changed"),
                       lambda p: p.update(bridge_entities=["other"]),
                       lambda p: p.update(stage_instructions="other"),
                       lambda p: p["evidence"].reverse(),
                       lambda p: p.update(new_metadata="not frozen"),
                       lambda p: p.update(bad=float("nan"))]
        for index, alter in enumerate(alterations):
            with self.subTest(index=index):
                live = RecordingAnswer()
                router = ProgramAnswerReplayRouter(self.case, live, final_payload=self.payload)
                prefix(router, self.case)
                payload = deepcopy(self.payload); alter(payload)
                with self.assertRaises(ReplayMismatch): router.complete("answer", payload)
                self.assertEqual(live.calls, [])

    def test_capture_rejects_invalid_final_question_json_or_wrong_service(self):
        for request in ({"question": "different"}, {**self.payload, "bad": float("inf")}, []):
            with self.subTest(request_type=type(request).__name__):
                router = ProgramAnswerReplayRouter(self.case, capture=True)
                prefix(router, self.case)
                with self.assertRaises(ReplayMismatch): router.complete("answer", request)
                self.assertIsNone(router.captured_payload)
        for name in ("read", "answer_retry"):
            router = ProgramAnswerReplayRouter(self.case, capture=True)
            prefix(router, self.case)
            with self.assertRaises(ReplayMismatch): router.complete(name, self.payload)

    def test_extra_tool_or_second_answer_is_fatal_after_capture_or_live_completion(self):
        for capture in (False, True):
            for extra in ("search", "read", "answer"):
                with self.subTest(capture=capture, extra=extra):
                    live = None if capture else RecordingAnswer()
                    router = ProgramAnswerReplayRouter(self.case, live, capture=capture,
                                                      final_payload=None if capture else self.payload)
                    prefix(router, self.case); router.complete("answer", self.payload)
                    router.assert_complete()
                    with self.assertRaises(ReplayMismatch):
                        if extra == "search": router.search("unexpected", 5)
                        elif extra == "read": router.read("other", 0, 10)
                        else: router.complete("answer", self.payload)
                    with self.assertRaises(ReplayMismatch): router.assert_complete()
                    if live is not None: self.assertEqual(len(live.calls), 1)

    def test_completed_billed_error_consumes_final_without_retry_or_fake_response(self):
        live = RecordingAnswer(ModelResponseError("synthetic completed bad JSON"))
        router = ProgramAnswerReplayRouter(self.case, live, final_payload=self.payload)
        prefix(router, self.case)
        with self.assertRaises(ModelResponseError): router.complete("answer", self.payload)
        self.assertTrue(router.assert_complete()["complete"])
        self.assertFalse(router.final_response_replayed)
        with self.assertRaises(ReplayMismatch): router.complete("answer", self.payload)
        self.assertEqual(len(live.calls), 1)

    def test_unknown_or_other_dispatch_exception_stays_fatal_without_retry(self):
        for error in (UnknownProviderOutcome("synthetic unknown"), RuntimeError("synthetic failure")):
            with self.subTest(error=type(error).__name__):
                live = RecordingAnswer(error)
                router = ProgramAnswerReplayRouter(self.case, live, final_payload=self.payload)
                prefix(router, self.case)
                for attempt in range(2):
                    with self.assertRaises(type(error)) as caught:
                        router.complete("answer", self.payload)
                    self.assertIs(caught.exception, error)
                with self.assertRaises(type(error)): router.assert_complete()
                self.assertEqual(len(live.calls), 1)
                self.assertFalse(router.summary()["complete"])

    def test_incomplete_prefix_cannot_be_declared_complete(self):
        router = ProgramAnswerReplayRouter(self.case, capture=True)
        prefix(router, self.case)
        with self.assertRaises(ReplayMismatch): router.assert_complete()
        self.assertIsNone(router.captured_payload)
        self.assertEqual(router.new_calls, 0)

    def test_legacy_answer_router_contract_is_not_relaxed(self):
        with self.assertRaises(ReplayContractError):
            AnswerReplayRouter(self.case, arm="evidence_only")
        original = AnswerReplayRouter(self.case)
        for event in self.case["events"]: dispatch(original, event)
        self.assertEqual(original.assert_complete()["new_model_calls"], 0)
        old = AnswerReplayRouter(self.case, RecordingAnswer())
        prefix(old, self.case)
        with self.assertRaises(ReplayMismatch): old.complete("answer", self.payload)

    def test_host_evidence_guard_still_rejects_forged_quotes_before_dispatch(self):
        payload = deepcopy(self.payload)
        payload["evidence"][0]["quote"] = "invented quote"
        live = RecordingAnswer()
        router = ProgramAnswerReplayRouter(self.case, live, final_payload=payload)
        broker = HostBroker(self.case["task"], router, router)
        for event in self.case["events"][:-1]: broker(event["name"], event["request"])
        with self.assertRaises(ValueError): broker("complete", {"stage": "answer", "payload": payload})
        self.assertEqual(live.calls, [])

    def test_execute_keeps_returned_answer_origin_check_after_exact_replay(self):
        runs = Path(__file__).parent / "runs"
        with tempfile.TemporaryDirectory(prefix="program-replay-", dir=runs) as temporary:
            root = Path(temporary)
            archive = ProgramArchive(root / "archive")
            node = archive.record(root_files(self.case["config"]), {}, session_id="synthetic", attempt=0)
            case, payload = deepcopy(self.case), deepcopy(self.payload)
            class SyntheticSandbox:
                # Exercise execute/HostBroker without importing archived sources.
                def __init__(self, forged): self.forged = forged
                def run(self, files, corpus, question, broker, seconds):
                    for event in case["events"][:-1]: broker(event["name"], event["request"])
                    response = broker("complete", {"stage": "answer", "payload": payload})
                    return {"result": {"answer": "forged returned answer" if self.forged else response["answer"],
                                       "citations": [deepcopy(payload["evidence"][0])]},
                            "runtime": {"isolation_checks": {"synthetic_fixture": True}}}
            for forged in (False, True):
                with self.subTest(forged=forged):
                    live = RecordingAnswer()
                    router = ProgramAnswerReplayRouter(case, live, final_payload=payload)
                    receipt = execute(archive, node["node_id"], case["task"], router, router,
                                      root / str(forged), sandbox=SyntheticSandbox(forged))
                    router.assert_complete()
                    self.assertIs(receipt["answer_origin_valid"], not forged)
                    self.assertEqual(len(live.calls), 1)
                    if forged: self.assertIn("invalid_answer_origin", receipt["failure_classes"])


if __name__ == "__main__":
    unittest.main()
