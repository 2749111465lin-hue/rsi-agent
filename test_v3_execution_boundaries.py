"""Synthetic host-provenance, fatal-failure and resume-integrity checks."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from code_rsi.archive import ProgramArchive
from code_rsi.budget import LimitExceeded, digest, save
from code_rsi.sandbox import SandboxExecutionError
from code_rsi.v3.datasets import adapt_multihop
from code_rsi.v3.execution import (CandidateEvidenceError, HostBroker, HostError,
                                    Measurement, execute, root_files)
from code_rsi.v3.infrastructure import LocalCorpus, ModelResponseError, UnknownProviderOutcome


TEXT = "Mira Vale was born in Northport."


def task_and_reference():
    return adapt_multihop({"id": "synthetic-q", "query": "Where was Mira Vale born?",
                          "answer": "Northport", "documents": [{"docid": "bio", "text": TEXT}]})


def evidence(source):
    return {"citation_id": "e1", "source_id": source["source_id"], "docid": source["docid"],
            "start": source["start"], "end": source["end"], "quote": source["text"]}


class Model:
    identity = "synthetic-model-v1"

    def __init__(self, raw_ids=None, error=None, mutate=False):
        self.raw_ids = ["e1"] if raw_ids is None else raw_ids
        self.error, self.mutate, self.calls = error, mutate, []

    def complete(self, stage, payload):
        self.calls.append((stage, copy.deepcopy(payload)))
        if self.error is not None:
            raise self.error
        if stage == "read":
            source = payload["sources"][0]
            if self.mutate:
                source["text"] = "fabricated source"
                source["end"] = source["start"] + len(source["text"])
            return {"claims": [{"text": "birthplace claim", "citations": [{
                "source_id": source["source_id"], "start": source["start"],
                "end": source["end"], "quote": source["text"]}]}],
                "bridge_entities": [], "gaps": [], "conflicts": [], "queries": [], "ready": True}
        return {"answer": "Northport", "citation_ids": self.raw_ids, "evidence_sufficient": True}


class BrokerSandbox:
    """Simulates only the sandbox RPC contract; does not import candidate code."""
    def __init__(self, *, wrong_quote=False, forge_report=False):
        self.wrong_quote, self.forge_report, self.calls = wrong_quote, forge_report, 0

    def run(self, program_dir, corpus_file, question, broker, **kwargs):
        self.calls += 1
        try:
            rows = broker("search", {"query": "Mira", "limit": 5})
            source = {**rows[0], "source_id": "s1"}
            broker("complete", {"stage": "read", "payload": {"sources": [source]}})
            item = evidence(source)
            final = broker("complete", {"stage": "answer", "payload": {"evidence": [item]}})
            citations = [item] if "e1" in final["citation_ids"] else []
            if self.wrong_quote and citations:
                citations[0] = {**item, "quote": "forged"}
            if self.forge_report:
                broker("record_trace", {"result": {"citations_valid": True, "correctness": "correct"}})
            return {"ok": True, "result": {"answer": final["answer"], "citations": citations,
                    "abstention_reason": None}, "runtime": {"isolation_checks": {"synthetic_rpc": True}}}
        except Exception as exc:
            raise SandboxExecutionError("trusted_broker_rejected", {"kind": "broker_error"}) from exc


class ExecutionBoundaryTests(unittest.TestCase):
    def temporary(self):
        return tempfile.TemporaryDirectory(dir=Path(__file__).parent / "runs")

    def archive(self, root):
        archive = ProgramArchive(Path(root) / "archive")
        node = archive.record(root_files(), {}, session_id="boundary-fixture", attempt=0)
        return archive, node

    def broker(self, model=None):
        task, _ = task_and_reference()
        return HostBroker(task, LocalCorpus(task["documents"]), model or Model())

    def presented_read(self, broker):
        rows = broker("search", {"query": "Mira", "limit": 5})
        source = {**rows[0], "source_id": "s1"}
        broker("complete", {"stage": "read", "payload": {"sources": [source]}})
        return source

    def test_valid_host_chain_and_raw_ids(self):
        broker = self.broker()
        source = self.presented_read(broker)
        item = evidence(source)
        broker("complete", {"stage": "answer", "payload": {"evidence": [item]}})
        receipt = broker.citation_receipt("Northport", [item])
        self.assertTrue(receipt["valid"])
        self.assertEqual(receipt["raw_citation_ids"], ["e1"])
        self.assertEqual(receipt["semantic_support"], "model_assessed_only")

    def test_read_payload_cannot_introduce_fabricated_source(self):
        for source in ({"source_id": "s1", "docid": "missing", "start": 0, "end": 4, "text": "fake"},
                       {"source_id": "s1", "docid": "bio", "start": 0, "end": 4, "text": "fake"}):
            broker = self.broker()
            broker("search", {"query": "Mira", "limit": 5})
            with self.subTest(source=source), self.assertRaises(CandidateEvidenceError):
                broker("complete", {"stage": "read", "payload": {"sources": [source]}})
            self.assertEqual(broker.counts["model_calls"], 0)

    def test_retrieved_but_never_model_read_quote_cannot_enter_answer(self):
        broker = self.broker()
        source = {**broker("search", {"query": "Mira", "limit": 5})[0], "source_id": "s1"}
        with self.assertRaises(CandidateEvidenceError):
            broker("complete", {"stage": "answer", "payload": {"evidence": [evidence(source)]}})
        self.assertEqual(broker.counts["model_calls"], 0)

    def test_answer_payload_quote_forgery_rejected_before_model(self):
        broker = self.broker()
        item = evidence(self.presented_read(broker))
        item["quote"] = "fake"
        with self.assertRaises(CandidateEvidenceError):
            broker("complete", {"stage": "answer", "payload": {"evidence": [item]}})
        self.assertEqual(broker.counts["model_calls"], 1)

    def test_model_mutation_cannot_rewrite_host_presented_source(self):
        broker = self.broker(Model(mutate=True))
        source = self.presented_read(broker)
        self.assertEqual(broker.source_windows[0]["text"], TEXT)
        self.assertEqual(broker.verified_read_quotes, set())
        with self.assertRaises(CandidateEvidenceError):
            broker("complete", {"stage": "answer", "payload": {"evidence": [evidence(source)]}})

    def test_unknown_raw_citation_is_not_hidden_by_wrapper_filtering(self):
        with self.temporary() as tmp:
            archive, node = self.archive(tmp)
            task, _ = task_and_reference()
            result = execute(archive, node["node_id"], task, LocalCorpus(task["documents"]),
                             Model(raw_ids=["e999"]), Path(tmp)/"cell", sandbox=BrokerSandbox(forge_report=True))
            self.assertEqual(result["answer"], "Northport")
            self.assertTrue(result["answer_usable"])
            self.assertTrue(result["execution_ok"])
            self.assertFalse(result["citation_source_valid"])
            self.assertEqual(result["citation_status"], "invalid_model_citation_ids")
            self.assertEqual(result["host_citation_validation"]["raw_citation_ids"], ["e999"])
            self.assertIn("invalid_answer_citation", result["failure_classes"])
            self.assertTrue(result["candidate_reported"]["citations_valid"])

    def test_missing_citations_are_never_true(self):
        broker = self.broker(Model(raw_ids=[]))
        item = evidence(self.presented_read(broker))
        broker("complete", {"stage": "answer", "payload": {"evidence": [item]}})
        receipt = broker.citation_receipt("Northport", [])
        self.assertFalse(receipt["valid"])
        self.assertEqual(receipt["status"], "missing_citations")

    def test_final_only_can_cite_actual_final_presented_ids(self):
        broker = self.broker()
        self.presented_read(broker)
        broker("complete", {"stage": "answer", "payload": {"evidence": []}})
        self.assertFalse(broker.citation_receipt("Northport", [])["valid"])
        self.assertEqual(broker.citation_receipt("Northport", [])["status"], "invalid_model_citation_ids")

    def test_candidate_cannot_change_final_source_quote(self):
        broker = self.broker()
        item = evidence(self.presented_read(broker))
        broker("complete", {"stage": "answer", "payload": {"evidence": [item]}})
        item["quote"] = "invented"
        receipt = broker.citation_receipt("Northport", [item])
        self.assertFalse(receipt["valid"])
        self.assertEqual(receipt["status"], "candidate_citation_mismatch")

    def test_generic_model_exception_is_fatal_host_error(self):
        broker = self.broker(Model(error=RuntimeError("sensitive adapter internals")))
        with self.assertRaises(HostError) as caught:
            broker("complete", {"stage": "plan", "payload": {}})
        self.assertNotIn("sensitive", str(caught.exception))
        self.assertIs(broker.fatal, caught.exception)
        with self.assertRaises(HostError):
            broker("complete", {"stage": "answer", "payload": {}})
        self.assertEqual(len(broker.model.calls), 1)

    def test_unknown_and_limit_failures_stay_fatal(self):
        for error in (UnknownProviderOutcome("unknown"), LimitExceeded("cap")):
            broker = self.broker(Model(error=error))
            with self.subTest(error=type(error).__name__), self.assertRaises(type(error)):
                broker("complete", {"stage": "plan", "payload": {}})
            self.assertIs(broker.fatal, error)

    def test_completed_bad_response_leaves_reserved_final(self):
        model = Model(error=ModelResponseError("truncated completed response"))
        broker = self.broker(model)
        result = broker("complete", {"stage": "plan", "payload": {}})
        self.assertTrue(result["_meta"]["truncated"])
        self.assertIsNone(broker.fatal)
        model.error = None
        broker("complete", {"stage": "answer", "payload": {}})
        self.assertEqual(broker.counts["model_calls"], 2)

    def test_raw_truncation_metadata_cannot_mint_verified_quotes(self):
        class TruncatedModel(Model):
            def complete(self, stage, payload):
                result = super().complete(stage, payload)
                result["_meta"] = {"finish_reason": "length"}
                return result
        broker = self.broker(TruncatedModel())
        self.presented_read(broker)
        self.assertEqual(broker.verified_read_quotes, set())
        self.assertEqual(broker.read_presentations, [])
        self.assertEqual(broker.model_errors, ["ModelResponseError"])
        self.assertIsNone(broker.fatal)
    def test_host_failure_does_not_write_complete_cell(self):
        with self.temporary() as tmp:
            archive, node = self.archive(tmp)
            task, ref = task_and_reference()
            measure = Measurement(archive, Path(tmp)/"measurements", lambda bank: Model(error=RuntimeError("broken")))
            with patch("code_rsi.v3.execution.Sandbox", return_value=BrokerSandbox()):
                with self.assertRaises(HostError):
                    measure.run(node, [task], {task["question_id"]: ref}, role="D_fit", bank="same")
            self.assertEqual(list((Path(tmp)/"measurements").rglob("measured.json")), [])
            self.assertEqual(list((Path(tmp)/"measurements").rglob("measurement.json")), [])

    def test_bad_citation_does_not_erase_external_answer_score(self):
        with self.temporary() as tmp:
            archive, node = self.archive(tmp)
            task, ref = task_and_reference()
            measure = Measurement(archive, Path(tmp)/"measurements", lambda bank: Model(raw_ids=["e999"]))
            with patch("code_rsi.v3.execution.Sandbox", return_value=BrokerSandbox()):
                result = measure.run(node, [task], {task["question_id"]: ref}, role="D_fit", bank="same")
            self.assertEqual(result["score"], 1.0)
            self.assertFalse(result["rows"][0]["citation_source_valid"])
            self.assertTrue(result["rows"][0]["answer_usable"])

    def test_intact_resume_reuses_cell_without_sandbox(self):
        with self.temporary() as tmp:
            archive, node = self.archive(tmp)
            task, ref = task_and_reference()
            measure = Measurement(archive, Path(tmp)/"measurements", lambda bank: Model())
            sandbox = BrokerSandbox()
            with patch("code_rsi.v3.execution.Sandbox", return_value=sandbox):
                first = measure.run(node, [task], {task["question_id"]: ref}, role="D_fit", bank="same", repeats=2)
                second = measure.run(node, [task], {task["question_id"]: ref}, role="D_fit", bank="same", repeats=2)
            self.assertEqual(first, second)
            self.assertEqual(sandbox.calls, 2)
            cell = next((Path(tmp)/"measurements").rglob("measured.json"))
            record = json.loads(cell.read_text(encoding="utf-8"))
            self.assertEqual(record["identity_sha256"], digest(record["identity"]))
            self.assertEqual(record["payload_sha256"], digest(record["payload"]))

    def test_resume_rejects_content_and_identity_corruption(self):
        for kind in ("payload", "identity", "payload_rehashed"):
            with self.subTest(kind=kind), self.temporary() as tmp:
                archive, node = self.archive(tmp)
                task, ref = task_and_reference()
                measure = Measurement(archive, Path(tmp)/"measurements", lambda bank: Model())
                sandbox = BrokerSandbox()
                with patch("code_rsi.v3.execution.Sandbox", return_value=sandbox):
                    measure.run(node, [task], {task["question_id"]: ref}, role="D_fit", bank="same")
                cell = next((Path(tmp)/"measurements").rglob("measured.json"))
                record = json.loads(cell.read_text(encoding="utf-8"))
                if kind == "payload":
                    record["payload"]["score"] = 0
                elif kind == "identity":
                    record["identity"]["role"] = "D_report"
                    record["identity_sha256"] = digest(record["identity"])
                else:
                    record["payload"]["question_id"] = "another-question"
                    record["payload_sha256"] = digest(record["payload"])
                save(cell, record)
                with patch("code_rsi.v3.execution.Sandbox", return_value=sandbox), self.assertRaises(ValueError):
                    measure.run(node, [task], {task["question_id"]: ref}, role="D_fit", bank="same")
                self.assertEqual(sandbox.calls, 1)

    def test_changed_model_identity_cannot_reuse_old_measurement(self):
        with self.temporary() as tmp:
            archive, node = self.archive(tmp)
            task, ref = task_and_reference()
            model = Model()
            measure = Measurement(archive, Path(tmp)/"measurements", lambda bank: model)
            sandbox = BrokerSandbox()
            with patch("code_rsi.v3.execution.Sandbox", return_value=sandbox):
                first = measure.run(node, [task], {task["question_id"]: ref}, role="D_fit", bank="same")
                model.identity = "different-frozen-model"
                second = measure.run(node, [task], {task["question_id"]: ref}, role="D_fit", bank="same")
            self.assertNotEqual(first["identity_hash"], second["identity_hash"])
            self.assertEqual(sandbox.calls, 2)


if __name__ == "__main__":
    unittest.main()