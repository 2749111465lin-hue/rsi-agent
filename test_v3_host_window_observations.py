"""Zero-API host window observations; synthetic RPC fixtures, no WSL."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from code_rsi.archive import ProgramArchive
from code_rsi.budget import digest
from code_rsi.v3.datasets import adapt_multihop, evaluate_answer
from code_rsi.v3.execution import HostBroker, HostError, execute, root_files
from code_rsi.v3.infrastructure import LocalCorpus, ModelResponseError, UnknownProviderOutcome
from test_v3_execution_boundaries import BrokerSandbox, Model, task_and_reference


TEXT = "前缀🙂 e\u0301 café\n尾"


def window(docid="window-doc", text=TEXT, start=37, **metadata):
    return {"docid": docid, "start": start, "end": start + len(text), "text": text, **metadata}


class Backend:
    def __init__(self, rows=None, read_result=None, error=None):
        self.rows = [window()] if rows is None else rows
        self.read_result = window() if read_result is None else read_result
        self.error = error
        self.calls = []

    def search(self, query, limit):
        self.calls.append(("search", query, limit))
        if self.error is not None:
            raise self.error
        return self.rows

    def read(self, **payload):
        self.calls.append(("read", deepcopy(payload)))
        if self.error is not None:
            raise self.error
        return self.read_result


class CompletedModel:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def complete(self, stage, payload):
        self.calls.append((stage, deepcopy(payload)))
        if self.error is not None:
            raise self.error
        if stage == "read":
            return {"claims": [{"text": "synthetic claim", "citations": [
                {"source_id": source["source_id"], "quote": source["text"]}]} for source in payload["sources"]],
                "bridge_entities": [], "gaps": [], "conflicts": [], "queries": [], "ready": True}
        if stage == "answer":
            return {"answer": "Synthetic answer", "citation_ids": list(c["citation_id"] for c in payload["evidence"]),
                    "evidence_sufficient": bool(payload["evidence"])}
        return {"queries": ["synthetic query"], "constraints": []}


def broker(backend=None, model=None, **limits):
    task, _ = adapt_multihop({"id": "window-test", "query": "Synthetic public question?",
                              "answer": "Synthetic answer", "documents": []})
    return HostBroker(task, backend or Backend(), model or CompletedModel(), **limits)


def evidence(source):
    return {"citation_id": "e1", "docid": source["docid"], "source_id": source["source_id"],
            "start": source["start"], "end": source["end"], "quote": source["text"]}


class HostWindowObservationTests(unittest.TestCase):
    def test_search_allowlist_recomputes_hash_and_preserves_original_rpc(self):
        row = window(docid=7, text_sha256="forged", gold="untrusted metadata",
                     observed_windows=[{"docid": "forged"}],
                     observed_window_count=999, observed_windows_truncated=True,
                     event_index=999, nested={"answer": "backend-controlled"})
        backend = Backend([row])
        host = broker(backend)
        returned = host("search", {"query": "synthetic", "limit": 5})
        self.assertEqual(returned, [row])
        event = host.events[0]
        self.assertEqual(event["response_hash"], digest([row]))
        self.assertEqual(event["observed_window_count"], 1)
        self.assertFalse(event["observed_windows_truncated"])
        self.assertEqual(event["observed_windows"], [{"docid": "7", "start": 37,
                          "end": 37 + len(TEXT), "text_sha256": hashlib.sha256(TEXT.encode("utf-8")).hexdigest()}])
        self.assertEqual(set(event["observed_windows"][0]), {"docid", "start", "end", "text_sha256"})
        self.assertEqual(len(backend.calls), 1)
        self.assertNotIn("backend-controlled", json.dumps(event["observed_windows"]))
        returned[0]["text"] = "candidate mutation"
        row["text"] = "backend mutation"
        self.assertEqual(host.source_windows[0]["text"], TEXT)
        self.assertEqual(event["observed_windows"][0]["text_sha256"], hashlib.sha256(TEXT.encode("utf-8")).hexdigest())

    def test_document_read_uses_character_offsets_and_actual_utf8_bytes(self):
        row = window(text_sha256="do not trust", private_reasoning="do not copy")
        host = broker(Backend(read_result=row))
        request = {"docid": row["docid"], "start": row["start"], "end": row["end"]}
        self.assertEqual(host("read", request), row)
        self.assertNotEqual(len(TEXT), len(TEXT.encode("utf-8")))
        event = host.events[0]
        self.assertEqual(event["request"], request)
        self.assertEqual(event["response_hash"], digest(row))
        self.assertEqual(event["observed_window_count"], 1)
        self.assertFalse(event["observed_windows_truncated"])
        observed = event["observed_windows"][0]
        self.assertEqual(observed["end"] - observed["start"], len(TEXT))
        self.assertEqual(observed["text_sha256"], hashlib.sha256(TEXT.encode("utf-8")).hexdigest())
        self.assertNotIn("text", observed)
        self.assertNotIn("private_reasoning", observed)

    def test_hash_preserves_unicode_composition_and_doc_id_alias(self):
        first = window(text="e\u0301")
        second = window(text="é")
        second["doc_id"] = second.pop("docid")
        host = broker(Backend([first, second]))
        returned = host("search", {"query": "synthetic", "limit": 2})
        self.assertEqual(returned, [first, second])
        observed = host.events[0]["observed_windows"]
        self.assertNotEqual(observed[0]["text_sha256"], observed[1]["text_sha256"])
        self.assertEqual(observed[1]["docid"], "window-doc")
        self.assertNotIn("doc_id", observed[1])

    def test_empty_search_has_known_empty_observation(self):
        host = broker(Backend([]))
        self.assertEqual(host("search", {"query": "no matches", "limit": 5}), [])
        self.assertEqual(host.events[0]["observed_windows"], [])
        self.assertEqual(host.events[0]["observed_window_count"], 0)
        self.assertFalse(host.events[0]["observed_windows_truncated"])

    def test_excluded_rows_are_not_observed_as_returned_sources(self):
        host = broker(Backend([window(docid="excluded"), window(docid="allowed")]))
        host.task["excluded_docids"] = ["excluded"]
        returned = host("search", {"query": "synthetic", "limit": 5})
        self.assertEqual([r["docid"] for r in returned], ["allowed"])
        self.assertEqual([r["docid"] for r in host.events[0]["observed_windows"]], ["allowed"])
        self.assertEqual(host.events[0]["observed_window_count"], 1)

    def test_oversized_backend_bounds_observation_only_not_rpc(self):
        rows = [window(docid="doc" + str(i)) for i in range(35)]
        host = broker(Backend(rows))
        returned = host("search", {"query": "synthetic", "limit": 5})
        self.assertEqual(returned, rows)
        event = host.events[0]
        self.assertEqual(event["response_hash"], digest(rows))
        self.assertEqual(event["observed_window_count"], 35)
        self.assertTrue(event["observed_windows_truncated"])
        self.assertEqual(len(event["observed_windows"]), 30)
        self.assertEqual([w["docid"] for w in event["observed_windows"]], [r["docid"] for r in rows[:30]])
        self.assertEqual(len(host.source_windows), 35)

    def test_existing_search_budget_bounds_normal_observation_volume(self):
        backend = Backend([window(docid="doc" + str(i)) for i in range(30)])
        host = broker(backend)
        for _ in range(8):
            host("search", {"query": "synthetic", "limit": 30})
        with self.assertRaises(ValueError):
            host("search", {"query": "one more", "limit": 30})
        self.assertEqual(len(backend.calls), 8)
        self.assertEqual(sum(len(e["observed_windows"]) for e in host.events), 240)

    def test_failed_backend_never_creates_completed_observation(self):
        malformed = window()
        malformed["end"] += 1
        cases = [(Backend([window(), malformed]), "search", {"query": "synthetic", "limit": 5}),
                 (Backend(read_result=malformed), "read", {"docid": "window-doc", "start": 37, "end": 40}),
                 (Backend(error=RuntimeError("backend failure")), "search", {"query": "synthetic", "limit": 5}),
                 (Backend(error=RuntimeError("backend failure")), "read", {"docid": "window-doc", "start": 37, "end": 40})]
        for backend, name, request in cases:
            with self.subTest(name=name, rows=backend.rows):
                host = broker(backend)
                with self.assertRaises(HostError):
                    host(name, request)
                self.assertEqual(host.events, [])
                self.assertEqual(host.source_windows, [])
                self.assertEqual(host.read_presentations, [])
                self.assertEqual(host.final_observations, [])
                self.assertEqual(len(backend.calls), 1)

    def test_read_and_answer_indices_link_exact_events_without_payload_changes(self):
        model = CompletedModel()
        host = broker(model=model)
        host("complete", {"stage": "plan", "payload": {"constraints": []}})
        source = {**host("search", {"query": "synthetic", "limit": 5})[0], "source_id": "s1"}
        payload = {"sources": [source], "question": "candidate cannot replace host question"}
        read_response = host("complete", {"stage": "read", "payload": payload})
        self.assertEqual(host.read_presentations[0]["event_index"], 2)
        self.assertEqual(model.calls[1], ("read", {**payload, "question": host.task["question"]}))
        self.assertNotIn("event_index", model.calls[1][1])
        self.assertNotIn("observed_windows", model.calls[1][1])
        self.assertEqual(host.events[2]["response_hash"], digest(read_response))
        self.assertNotIn("observed_windows", host.events[2])
        host("read", {"docid": source["docid"], "start": source["start"], "end": source["end"]})
        item = evidence(source)
        answer_payload = {"evidence": [item]}
        answer_response = host("complete", {"stage": "answer", "payload": answer_payload})
        self.assertEqual(host.final_observations[0]["event_index"], 4)
        self.assertEqual(model.calls[2], ("answer", {**answer_payload, "question": host.task["question"]}))
        self.assertEqual(answer_response, host.final_observations[0]["response"])
        self.assertTrue(host.citation_receipt(answer_response["answer"], [item])["valid"])
        for observation in host.read_presentations + host.final_observations:
            self.assertEqual(host.events[observation["event_index"]]["name"], "complete")

    def test_candidate_report_cannot_forge_host_observations_or_indices(self):
        host = broker()
        forged = {"observed_windows": [window(docid="fabricated")], "event_index": 999}
        host("record_trace", {"result": forged})
        host("search", {"query": "synthetic", "limit": 5})
        host("complete", {"stage": "read", "payload": {"sources": []}})
        self.assertEqual(host.reported, forged)
        self.assertNotIn("observed_windows", host.events[0])
        self.assertEqual(host.events[1]["observed_windows"][0]["docid"], "window-doc")
        self.assertEqual(host.read_presentations[0]["event_index"], 2)

    def test_failed_model_response_creates_no_successful_read_or_final_observation(self):
        for stage in ("read", "answer"):
            for error in (ModelResponseError("truncated response"), UnknownProviderOutcome("unknown"), RuntimeError("service error")):
                with self.subTest(stage=stage, error=type(error).__name__):
                    host = broker(model=CompletedModel(error))
                    payload = {"stage": stage, "payload": {"sources": []} if stage == "read" else {"evidence": []}}
                    if isinstance(error, ModelResponseError):
                        response = host("complete", payload)
                        self.assertEqual(response["_meta"]["finish_reason"], "error")
                        # Existing completed-failure event is retained, without a
                        # successful presentation/observation or new window fields.
                        self.assertEqual(len(host.events), 1)
                        self.assertNotIn("observed_windows", host.events[0])
                    else:
                        with self.assertRaises((HostError, UnknownProviderOutcome)):
                            host("complete", payload)
                        self.assertEqual(host.events, [])
                    self.assertEqual(host.read_presentations, [])
                    self.assertEqual(host.final_observations, [])

    def test_execution_receipt_keeps_schema_and_answer_score(self):
        runs = Path(__file__).resolve().parent / "runs"
        runs.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="host_window_observations_", dir=runs) as tmp:
            archive = ProgramArchive(Path(tmp) / "archive")
            node = archive.record(root_files(), {}, session_id="window-observation", attempt=0)
            task, reference = task_and_reference()
            receipt = execute(archive, node["node_id"], task, LocalCorpus(task["documents"]), Model(),
                              Path(tmp) / "cell", sandbox=BrokerSandbox())
            self.assertEqual(receipt["schema"], "rag-rsi-v3-execution-2")
            self.assertNotIn("observed_windows", receipt)
            self.assertNotIn("event_index", receipt)
            self.assertTrue(receipt["citation_source_valid"])
            self.assertEqual(evaluate_answer(receipt["answer"], reference, "f1"), 1.0)
            self.assertEqual(receipt["trace"][0]["observed_window_count"], 1)
            self.assertEqual(receipt["host_evidence_trace"]["read_presentations"][0]["event_index"], 1)
            self.assertEqual(receipt["host_evidence_trace"]["final_observations"][0]["event_index"], 2)


if __name__ == "__main__":
    unittest.main()
