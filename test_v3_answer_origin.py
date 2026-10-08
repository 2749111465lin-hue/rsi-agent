"""Synthetic answer-origin regressions; opt-in real WSL uses no API or secrets."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from code_rsi.archive import ProgramArchive
from code_rsi.budget import digest, save
from code_rsi.sandbox import SandboxExecutionError
from code_rsi.v3 import execution
from code_rsi.v3.datasets import adapt_multihop
from code_rsi.v3.infrastructure import LocalCorpus, ModelResponseError, UnknownProviderOutcome


EXPECTED = "Synthetic Northport"
WRONG = "Synthetic Southport"


def fixture():
    return adapt_multihop({"id": "synthetic-answer-origin", "query": "Which invented port?",
                          "answer": EXPECTED, "documents": [{"docid": "synthetic",
                          "text": "The synthetic record is assigned to Synthetic Northport."}]})


def final(answer=EXPECTED, **extra):
    return {"answer": answer, "citation_ids": [], "evidence_sufficient": False, **extra}


class ScriptedModel:
    def __init__(self, responses=()):
        self.responses = list(responses)
        self.calls = []
        self.identity = "synthetic-answer-origin-" + digest(
            [type(r).__name__ if isinstance(r, Exception) else r for r in self.responses])

    def complete(self, stage, payload):
        self.calls.append((stage, copy.deepcopy(payload)))
        response = self.responses[len(self.calls)-1]
        if isinstance(response, Exception):
            raise response
        return copy.deepcopy(response)


class ReturnSandbox:
    """Exercise host RPC only, without importing or executing candidate code."""
    def __init__(self, *, calls=0, returned=EXPECTED, report=None, fail=False):
        self.answer_calls, self.returned, self.report, self.fail = calls, returned, report, fail
        self.runs = 0

    def run(self, program_dir, corpus_file, question, broker, **kwargs):
        self.runs += 1
        try:
            for _ in range(self.answer_calls):
                broker("complete", {"stage": "answer", "payload": {"evidence": []}})
            if self.report is not None:
                broker("record_trace", {"result": self.report})
        except Exception as exc:
            raise SandboxExecutionError("trusted_broker_rejected", {"kind": "broker_error"}) from exc
        if self.fail:
            raise SandboxExecutionError("candidate_failed", {"kind": "candidate_error"})
        return {"result": {"answer": self.returned, "citations": [], "abstention_reason": None},
                "runtime": {"isolation_checks": {"synthetic_rpc": True}}}


class AnswerOriginTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parent / "runs"
        root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="answer_origin_", dir=root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = ProgramArchive(self.root / "archive")
        self.node = self.archive.record(execution.root_files(), {}, session_id="origin", attempt=0)
        self.task, self.reference = fixture()

    def run_measurement(self, sandbox, responses=()):
        model = ScriptedModel(responses)
        measure = execution.Measurement(self.archive, self.root / "measurements", lambda bank: model)
        with patch.object(execution, "Sandbox", return_value=sandbox):
            result = measure.run(self.node, [self.task], {self.task["question_id"]: self.reference},
                                 role="D_fit", bank="synthetic")
        return result, measure, model

    def resume(self, measure, sandbox):
        with patch.object(execution, "Sandbox", return_value=sandbox):
            return measure.run(self.node, [self.task], {self.task["question_id"]: self.reference},
                               role="D_fit", bank="synthetic")

    def test_zero_call_correct_constant_is_diagnostic_only(self):
        result, _, model = self.run_measurement(ReturnSandbox())
        row = result["rows"][0]
        self.assertEqual(result["score"], 1.)
        self.assertFalse(result["valid_program"])
        self.assertTrue(row["execution_ok"])
        self.assertTrue(row["answer_usable"])
        self.assertFalse(row["answer_origin_valid"])
        self.assertEqual(row["answer_origin_status"], "no_observed_final_answer")
        self.assertIn("invalid_answer_origin", row["failure_classes"])
        self.assertEqual(model.calls, [])
        self.assertFalse(execution.validate_answer_origin(row)["valid"])

    def test_overridden_model_answer_is_not_eligible_despite_correct_text(self):
        report = {"answer_origin_valid": True, "answer": EXPECTED,
                  "host_answer_origin_validation": {"valid": True}}
        result, _, _ = self.run_measurement(ReturnSandbox(calls=1, report=report), [final(WRONG)])
        row = result["rows"][0]
        self.assertEqual(row["score"], 1.)
        self.assertFalse(result["valid_program"])
        self.assertEqual(row["answer_origin_status"], "candidate_answer_mismatch")
        self.assertTrue(row["candidate_reported"]["answer_origin_valid"])
        self.assertEqual(row["host_answer_origin_validation"]["event_index"], 0)

    def test_legitimate_model_answer_without_citations_has_valid_origin(self):
        result, _, _ = self.run_measurement(ReturnSandbox(calls=1), [final()])
        row = result["rows"][0]
        self.assertTrue(result["valid_program"])
        self.assertTrue(row["answer_origin_valid"])
        self.assertFalse(row["citation_source_valid"])
        self.assertEqual(row["citation_status"], "missing_citations")
        origin = row["host_answer_origin_validation"]
        observed = row["host_evidence_trace"]["final_observations"][-1]
        event = row["trace"][origin["event_index"]]
        self.assertEqual(origin["response_sha256"], digest(observed["response"]))
        self.assertEqual(origin["response_sha256"], event["response_hash"])
        self.assertEqual(origin["payload_sha256"], digest(event["request"]["payload"]))
        self.assertEqual(event["request"]["payload"]["question"], self.task["question"])

    def test_abstention_can_strip_outer_whitespace_without_citations(self):
        result, _, _ = self.run_measurement(
            ReturnSandbox(calls=1, returned="Insufficient information"),
            [final(" \nInsufficient information\t ")])
        row = result["rows"][0]
        self.assertTrue(result["valid_program"])
        self.assertTrue(row["answer_origin_valid"])
        self.assertNotIn("invalid_answer_origin", row["failure_classes"])
        self.assertFalse(row["citation_source_valid"])

    def test_blank_model_answer_keeps_origin_fact_but_is_not_deliverable(self):
        result, _, _ = self.run_measurement(ReturnSandbox(calls=1, returned=""), [final("  ")])
        row = result["rows"][0]
        self.assertTrue(row["answer_origin_valid"])
        self.assertFalse(row["answer_usable"])
        self.assertFalse(result["valid_program"])
        self.assertIn("answer_empty", row["failure_classes"])
        self.assertNotIn("invalid_answer_origin", row["failure_classes"])
        self.assertEqual(result["score"], 0.)

    def test_blank_answer_cache_cannot_forge_usability(self):
        _, measure, _ = self.run_measurement(ReturnSandbox(calls=1, returned=""), [final("  ")])
        path = next((self.root / "measurements").rglob("measured.json"))
        record = json.loads(path.read_text(encoding="utf-8"))
        record["payload"]["answer_usable"] = True
        record["payload_sha256"] = digest(record["payload"])
        save(path, record)
        with self.assertRaises(ValueError): self.resume(measure, ReturnSandbox(calls=1, returned=""))

    def test_internal_whitespace_and_case_cannot_be_rewritten(self):
        for model_answer in ("Synthetic  Northport", "synthetic Northport", "Synthetic\nNorthport"):
            with self.subTest(model_answer=model_answer):
                broker = execution.HostBroker(self.task, LocalCorpus([]), ScriptedModel([final(model_answer)]))
                broker("complete", {"stage": "answer", "payload": {"evidence": []}})
                self.assertFalse(broker.answer_origin_receipt(EXPECTED)["valid"])

    def test_earlier_successful_answer_cannot_replace_latest_success(self):
        result, _, _ = self.run_measurement(ReturnSandbox(calls=2), [final(), final(WRONG)])
        row = result["rows"][0]
        self.assertFalse(result["valid_program"])
        self.assertEqual(row["host_answer_origin_validation"]["event_index"], 1)
        self.assertEqual(row["answer_origin_status"], "candidate_answer_mismatch")
        self.assertEqual(row["citation_status"], "no_observed_final_answer")

    def test_failed_completed_response_does_not_replace_last_success(self):
        result, _, _ = self.run_measurement(ReturnSandbox(calls=2),
                                           [final(), ModelResponseError("completed malformed response")])
        row = result["rows"][0]
        self.assertTrue(result["valid_program"])
        self.assertEqual(row["host_answer_origin_validation"]["event_index"], 0)
        self.assertFalse(row["trace"][1]["model_completed"])

    def test_answerless_success_cannot_reuse_an_earlier_answer(self):
        result, _, _ = self.run_measurement(ReturnSandbox(calls=2), [final(), {"citation_ids": []}])
        self.assertFalse(result["valid_program"])
        self.assertEqual(result["rows"][0]["answer_origin_status"], "invalid_model_answer")

    def test_failed_execution_has_explicit_invalid_origin(self):
        result, _, _ = self.run_measurement(ReturnSandbox(calls=1, fail=True), [final()])
        row = result["rows"][0]
        self.assertFalse(row["execution_ok"])
        self.assertFalse(result["valid_program"])
        self.assertEqual(row["answer_origin_status"], "execution_failed")
        self.assertFalse(execution.validate_answer_origin(row)["valid"])

    def test_plan_stage_answer_does_not_count_as_final_origin(self):
        broker = execution.HostBroker(self.task, LocalCorpus([]), ScriptedModel([final()]))
        broker("complete", {"stage": "plan", "payload": {}})
        self.assertEqual(broker.answer_origin_receipt(EXPECTED)["status"], "no_observed_final_answer")

    def test_valid_and_invalid_cells_resume_without_new_execution(self):
        for calls in (0, 1):
            with self.subTest(calls=calls):
                sandbox = ReturnSandbox(calls=calls)
                result, measure, _ = self.run_measurement(sandbox, [final()] if calls else [])
                self.assertEqual(result, self.resume(measure, sandbox))
                self.assertEqual(sandbox.runs, 1)
                self.assertEqual(result["valid_program"], bool(calls))
                self.assertIn("v5-answer-origin-1", result["evaluator_epoch"])

    def test_unknown_provider_result_remains_fatal_and_uncached(self):
        with self.assertRaises(UnknownProviderOutcome):
            self.run_measurement(ReturnSandbox(calls=1), [UnknownProviderOutcome("synthetic unknown")])
        self.assertEqual(list((self.root / "measurements").rglob("measured.json")), [])
        self.assertEqual(list((self.root / "measurements").rglob("measurement.json")), [])
        self.assertEqual(list((self.root / "measurements").rglob("execution.json")), [])

    def test_cache_rejects_legacy_schema_and_missing_origin_even_when_rehashed(self):
        result, measure, _ = self.run_measurement(ReturnSandbox(calls=1), [final()])
        path = next((self.root / "measurements").rglob("measured.json"))
        original = json.loads(path.read_text(encoding="utf-8"))
        for problem in ("cell_schema", "execution_schema", "origin_flag", "origin_status", "origin_receipt",
                        "events", "observations"):
            with self.subTest(problem=problem):
                record = copy.deepcopy(original)
                row = record["payload"]
                if problem == "cell_schema": record["schema"] = "rag-rsi-v3-measured-cell-2"
                elif problem == "execution_schema": row["schema"] = "rag-rsi-v3-execution-2"
                elif problem == "origin_flag": row.pop("answer_origin_valid")
                elif problem == "origin_status": row.pop("answer_origin_status")
                elif problem == "origin_receipt": row.pop("host_answer_origin_validation")
                elif problem == "events": row.pop("trace")
                else: row["host_evidence_trace"].pop("final_observations")
                record["payload_sha256"] = digest(row)
                save(path, record)
                with self.assertRaises(ValueError): self.resume(measure, ReturnSandbox(calls=1))
        save(path, original)

    def test_cache_rejects_forged_origin_and_inconsistent_event_bindings(self):
        _, measure, _ = self.run_measurement(ReturnSandbox(calls=2), [final(WRONG), final()])
        path = next((self.root / "measurements").rglob("measured.json"))
        original = json.loads(path.read_text(encoding="utf-8"))
        for problem in ("returned_answer", "flag", "status", "origin_index", "observation_answer",
                        "observation_hash", "event_hash", "event_success", "payload_hash",
                        "payload_text", "evidence", "drop_last", "reverse", "missing_success",
                        "numeric_flag", "boolean_index", "invalid_stage"):
            with self.subTest(problem=problem):
                record = copy.deepcopy(original); row = record["payload"]
                origin = row["host_answer_origin_validation"]
                observed = row["host_evidence_trace"]["final_observations"]
                event = row["trace"][-1]
                if problem == "returned_answer": row["answer"] = WRONG
                elif problem == "flag": row["answer_origin_valid"] = False
                elif problem == "status": row["answer_origin_status"] = "candidate_answer_mismatch"
                elif problem == "origin_index": origin["event_index"] = 0
                elif problem == "observation_answer": observed[-1]["response"]["answer"] = WRONG
                elif problem == "observation_hash": observed[-1]["response_sha256"] = digest("forged")
                elif problem == "event_hash": event["response_hash"] = digest("forged")
                elif problem == "event_success": event["model_completed"] = False
                elif problem == "payload_hash": event["payload_sha256"] = digest({})
                elif problem == "payload_text": event["request"]["payload"]["question"] = "forged"
                elif problem == "evidence": observed[-1]["evidence"] = {"e1": {"citation_id": "e1"}}
                elif problem == "drop_last": observed.pop()
                elif problem == "reverse": observed.reverse()
                elif problem == "missing_success": event.pop("model_completed")
                elif problem == "numeric_flag": origin["valid"] = 1
                elif problem == "boolean_index": origin["event_index"] = True
                else: event["request"]["stage"] = []
                record["payload_sha256"] = digest(row); save(path, record)
                with self.assertRaises(ValueError): self.resume(measure, ReturnSandbox(calls=2))
        save(path, original)

    def test_no_call_cache_cannot_claim_valid_origin_by_boolean_forgery(self):
        _, measure, _ = self.run_measurement(ReturnSandbox())
        path = next((self.root / "measurements").rglob("measured.json"))
        record = json.loads(path.read_text(encoding="utf-8")); row = record["payload"]
        row["answer_origin_valid"] = True
        row["answer_origin_status"] = "last_successful_answer_verified"
        row["host_answer_origin_validation"].update(valid=True, status=row["answer_origin_status"])
        record["payload_sha256"] = digest(row); save(path, record)
        with self.assertRaises(ValueError): self.resume(measure, ReturnSandbox())


@unittest.skipUnless(os.environ.get("RUN_V3_ANSWER_ORIGIN_WSL") == "1", "explicit real-WSL regression")
class RealWSLAnswerOriginTests(unittest.TestCase):
    def test_real_candidate_output_is_bound_to_model_response(self):
        root = Path(__file__).resolve().parent / "runs"
        root.mkdir(exist_ok=True)
        task, reference = fixture()
        summary = {"synthetic": True, "runtime": "actual WSL isolation", "new_api_calls": 0, "cases": {}}
        with tempfile.TemporaryDirectory(prefix="origin_wsl_", dir=root) as temporary:
            folder = Path(temporary); archive = ProgramArchive(folder / "archive")
            cases = {
                "zero_call": ([], "    answer = " + repr(EXPECTED), False),
                "override": ([final(WRONG)], "    services.call('complete', {'stage':'answer','payload':{'evidence':[]}})\n    answer = " + repr(EXPECTED), False),
                "legitimate": ([final()], "    answer = services.call('complete', {'stage':'answer','payload':{'evidence':[]}})['answer']", True),
                "earlier_answer": ([final(), final(WRONG)], "    answer = services.call('complete', {'stage':'answer','payload':{'evidence':[]}})['answer']\n    services.call('complete', {'stage':'answer','payload':{'evidence':[]}})", False),
                "trim_abstention": ([final(" \nInsufficient information\t")], "    answer = services.call('complete', {'stage':'answer','payload':{'evidence':[]}})['answer'].strip()", True),
            }
            for index, (name, (responses, body, valid)) in enumerate(cases.items()):
                with self.subTest(case=name):
                    candidate = "def solve(question, services):\n" + body + "\n    return {'answer':answer,'citations':[],'abstention_reason':None}\n"
                    files = {**execution.root_files(), "rag.py": candidate}
                    execution.validate_sources(files)
                    node = archive.record(files, {}, session_id="origin-wsl", attempt=index)
                    model = ScriptedModel(responses)
                    measure = execution.Measurement(archive, folder / name, lambda bank: model)
                    result = measure.run(node, [task], {task["question_id"]: reference}, role="D_fit", bank="synthetic")
                    row = result["rows"][0]
                    self.assertEqual(result["valid_program"], valid)
                    self.assertEqual(row["answer_origin_valid"], valid)
                    self.assertTrue(row["isolation_verified"])
                    self.assertEqual(len(model.calls), len(responses))
                    if name != "trim_abstention": self.assertEqual(result["score"], 1.)
                    summary["cases"][name] = {key: row[key] for key in
                        ("score", "answer_origin_valid", "answer_origin_status", "execution_ok", "isolation_verified", "resource_usage")}
                    summary["cases"][name]["valid_program"] = result["valid_program"]
            summary["source_sha256"] = digest(Path(execution.__file__).read_text(encoding="utf-8"))
        save(root / "answer_origin_regression" / "summary.json", summary)


if __name__ == "__main__":
    unittest.main()
