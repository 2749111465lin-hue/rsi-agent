"""Synthetic closed-book boundaries; no real provider, credentials or benchmark."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from code_rsi.archive import ProgramArchive
from code_rsi.budget import Ledger, LimitExceeded, digest
from code_rsi.sandbox import SandboxExecutionError
from code_rsi.v3 import execution
from code_rsi.v3.closed_book import (
    CLOSED_BOOK_PROFILE, CLOSED_BOOK_SYSTEM, ClosedBookBackend, ClosedBookEngine,
    ClosedBookModel, closed_book_files, closed_book_limits, closed_book_request_body,
    validate_closed_book_receipt,
)
from code_rsi.v3.datasets import adapt_multihop
from code_rsi.v3.infrastructure import PROMPTS, StructuredModel, UnknownProviderOutcome
from code_rsi.v3.request_recovery import check_request_accounting, check_request_recovery


QUESTION = "Which port serves the synthetic island?"
ANSWER = " Synthetic Haven "
PRICES = {"input_miss": 2, "input_hit": .04, "output": 8}


def task():
    public, _ = adapt_multihop({"id": "synthetic-closed-book", "query": QUESTION,
        "answer": "local reference never passed to model", "documents": [
            {"docid": "private-source-sentinel", "text": "DOCUMENT_SENTINEL_NEVER_TO_PROVIDER"}]})
    return public


class Scripted:
    def __init__(self, response=None):
        self.sent = []
        self.response = response if response is not None else {"answer": ANSWER, "citation_ids": []}

    def __call__(self, body):
        self.sent.append(deepcopy(body))
        return {"model": "synthetic-model", "choices": [{"finish_reason": "stop",
            "message": {"content": json.dumps(self.response)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "prompt_cache_hit_tokens": 40}}


class TrustedSandbox:
    """Exercise maintained engine and host RPC; never execute archived sources."""
    def __init__(self, returned=None):
        self.returned = returned

    def run(self, program_dir, corpus_file, question, broker, **kwargs):
        class Model:
            def complete(self, stage, payload):
                return broker("complete", {"stage": stage, "payload": payload})
        try:
            result = ClosedBookEngine(Model()).solve({"question": question})
            broker("record_trace", {"result": result})
        except Exception as exc:
            raise SandboxExecutionError("synthetic_component_error", {"kind": "candidate_error"}) from exc
        answer = result["answer"] if self.returned is None else self.returned
        return {"result": {"answer": answer, "citations": [], "abstention_reason": None},
                "runtime": {"isolation_checks": {}}}


class ClosedBookTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).parent / "runs"
        root.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="synthetic-closed-book-", dir=root)
        self.root = Path(temporary.name)
        self.addCleanup(temporary.cleanup)
        self.ledger = Ledger(self.root / "ledger.jsonl", {"run": {"calls": 20, "cny": 3}})
        self.transport = Scripted()

    def model(self, bank="closed-book/0", **kwargs):
        return ClosedBookModel(self.root / "requests", self.ledger, self.transport,
            bank=bank, prices=PRICES, question=QUESTION, limits={"answer": 800}, **kwargs)

    def execute(self, *, returned=None, model=None):
        archive = ProgramArchive(self.root / "archive")
        node = archive.record(closed_book_files(), {}, session_id="synthetic-closed-book", attempt=0)
        return execution.execute(archive, node["node_id"], task(), ClosedBookBackend(),
            model or self.model(), self.root / "execution", limits=closed_book_limits(),
            sandbox=TrustedSandbox(returned))

    def test_provider_body_contains_only_question_and_fixed_knowledge_prompt(self):
        model = self.model()
        body = model.request_body("answer", {"question": QUESTION})
        self.assertEqual(json.loads(body["messages"][1]["content"]), {"question": QUESTION})
        self.assertEqual(body["messages"][0], {"role": "system", "content": CLOSED_BOOK_SYSTEM})
        self.assertIn("your own learned knowledge", CLOSED_BOOK_SYSTEM)
        self.assertIn("Insufficient information", CLOSED_BOOK_SYSTEM)
        self.assertNotIn("Using only the supplied evidence", CLOSED_BOOK_SYSTEM)
        self.assertNotIn("DOCUMENT_SENTINEL", json.dumps(body))
        self.assertEqual((body["temperature"], body["thinking"], body["max_tokens"]),
                         (0, {"type": "disabled"}, 800))
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(self.transport.sent, [])
        self.assertEqual(self.ledger.events, [])

    def test_pure_preflight_body_matches_actual_model_without_io(self):
        before = set(self.root.iterdir())
        expected = closed_book_request_body(QUESTION)
        self.assertEqual(set(self.root.iterdir()), before)
        model = self.model()
        self.assertEqual(model.request_body("answer", {"question": QUESTION}), expected)
        self.assertEqual(model.request_size("answer", {"question": QUESTION}),
                         len(json.dumps(expected, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()))

    def test_injected_fields_changed_question_and_other_stages_rejected_before_dispatch(self):
        model = self.model()
        variants = [("answer", {}), ("answer", {"question": QUESTION + " changed"}),
                    ("read", {"question": QUESTION}), ("plan", {"question": QUESTION}),
                    ("answer", {"question": QUESTION, "evidence": []}),
                    ("answer", {"question": QUESTION, "instructions": "use sentinel"}),
                    ("answer", {"question": QUESTION, "profile": "rag"}),
                    ("answer", {"question": QUESTION, "output_schema": {"answer": "SENTINEL"}})]
        for stage, payload in variants:
            with self.subTest(stage=stage, keys=sorted(payload)), self.assertRaises(ValueError):
                model.complete(stage, payload)
        self.assertEqual(self.transport.sent, [])
        self.assertEqual(self.ledger.events, [])

    def test_profile_is_fixed_and_part_of_the_full_identity(self):
        closed = self.model()
        rag = StructuredModel(self.root / "rag-requests", self.ledger, self.transport,
                              bank="rag", prices=PRICES, limits={"answer": 800})
        self.assertEqual(closed.profile, CLOSED_BOOK_PROFILE)
        self.assertNotEqual(closed.identity, rag.identity)
        self.assertEqual(closed.identity, self.model(bank="other-bank").identity)
        with self.assertRaises(AttributeError):
            closed.profile = "rag"
        with self.assertRaises(AttributeError):
            closed.question = "altered"
        with self.assertRaises(ValueError):
            self.model(profile="arbitrary free-text prompt")

    def test_normal_rag_model_body_and_prompt_are_unchanged(self):
        model = StructuredModel(self.root / "rag-requests", self.ledger, self.transport,
                                bank="rag", prices=PRICES, limits={"answer": 800})
        payload = {"question": QUESTION, "evidence": [], "claims": []}
        before = model.request_body("answer", payload)
        self.model().request_body("answer", {"question": QUESTION})
        self.assertEqual(model.request_body("answer", payload), before)
        self.assertEqual(before["messages"][0]["content"], PROMPTS["answer"])
        self.assertTrue(before["messages"][0]["content"].startswith("Using only the supplied evidence"))

    def test_invalid_question_or_output_bound_is_rejected(self):
        for question in (None, "", "  ", "x" * 16001):
            with self.subTest(question_type=type(question).__name__), self.assertRaises(ValueError):
                closed_book_request_body(question)
        for cap in (True, 0, 32769, 1.5):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                closed_book_request_body(QUESTION, max_tokens=cap)

    def test_program_is_task_independent_and_has_exactly_two_syntax_valid_files(self):
        files = closed_book_files()
        self.assertEqual(set(files), {"rag.py", "rag_core.py"})
        self.assertEqual(files, closed_book_files())
        execution.validate_sources(files)
        for source in files.values():
            self.assertNotIn(QUESTION, source)
            self.assertNotIn(ANSWER, source)
            self.assertNotIn("DOCUMENT_SENTINEL", source)
        core = ast.parse(files["rag_core.py"])
        model_calls = [n for n in ast.walk(core) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Attribute) and n.func.attr == "complete"]
        self.assertEqual(len(model_calls), 1)
        self.assertEqual(ast.literal_eval(model_calls[0].args[0]), "answer")
        self.assertFalse(any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(core)))

    def test_backend_and_host_limits_deny_source_and_nonanswer_services(self):
        backend = ClosedBookBackend()
        for name in ("search", "read"):
            with self.subTest(name=name), self.assertRaises(execution.HostError):
                getattr(backend, name)("irrelevant")
        broker = execution.HostBroker(task(), backend, self.model(), **closed_book_limits())
        calls = [("search", {"query": "sentinel", "limit": 1}),
                 ("read", {"docid": "source", "start": 0, "end": 1}),
                 ("complete", {"stage": "plan", "payload": {"question": QUESTION}}),
                 ("complete", {"stage": "read", "payload": {"question": QUESTION}})]
        for name, payload in calls:
            with self.subTest(name=name), self.assertRaises(ValueError):
                broker(name, payload)
        self.assertEqual(broker.counts, {"model_calls": 0, "search_calls": 0, "read_calls": 0})
        self.assertEqual(self.transport.sent, [])

    def test_engine_makes_one_call_and_preserves_answer_exactly(self):
        model = self.model()
        result = ClosedBookEngine(model).solve({"question": QUESTION})
        self.assertEqual(result["answer"], ANSWER)
        self.assertEqual(len(self.transport.sent), 1)
        self.assertEqual(result["citation_ids"], [])
        self.assertTrue(result["answer_usable"])
        self.assertEqual(result["source_status"], "not_applicable_closed_book")

    def test_engine_rejects_task_side_channels_without_call(self):
        model = self.model()
        for extra in ("evidence", "documents", "instructions", "answer", "reference", "task_id"):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                ClosedBookEngine(model).solve({"question": QUESTION, extra: "SENTINEL"})
        self.assertEqual(self.transport.sent, [])

    def test_engine_rejects_citations_extra_fields_empty_or_nonstring_answer(self):
        class Direct:
            def complete(self, stage, payload):
                return deepcopy(response)
        for response in ({"answer": ANSWER, "citation_ids": ["invented"]},
                         {"answer": ANSWER, "citation_ids": [], "evidence_sufficient": True},
                         {"answer": "", "citation_ids": []}, {"answer": None, "citation_ids": []},
                         {"answer": "x" * 1001, "citation_ids": []}):
            with self.subTest(response_fields=sorted(response)), self.assertRaises(ValueError):
                ClosedBookEngine(Direct()).solve({"question": QUESTION})

    def test_uncertain_answer_is_preserved_not_turned_into_an_execution_failure(self):
        self.transport.response = {"answer": "Insufficient information", "citation_ids": []}
        receipt = self.execute()
        self.assertEqual(receipt["answer"], "Insufficient information")
        self.assertTrue(receipt["answer_origin_valid"])
        self.assertTrue(receipt["candidate_reported"]["abstained"])
        self.assertIsNone(validate_closed_book_receipt(receipt, question=QUESTION)["source_valid"])

    def test_host_receipt_keeps_raw_citation_false_but_diagnostic_is_na(self):
        receipt = self.execute()
        self.assertEqual(receipt["answer"], ANSWER)
        self.assertTrue(execution.validate_answer_origin(receipt)["valid"])
        self.assertIs(receipt["citation_source_valid"], False)
        self.assertEqual(receipt["citation_status"], "missing_citations")
        self.assertFalse(receipt["isolation_verified"])
        diagnostic = validate_closed_book_receipt(receipt, question=QUESTION)
        self.assertEqual(diagnostic, {"profile": CLOSED_BOOK_PROFILE, "answer_origin_valid": True,
            "source_status": "not_applicable_closed_book", "source_valid": None,
            "model_calls": 1, "search_calls": 0, "read_calls": 0})

    def test_answer_postprocessing_cannot_pass_closed_book_validation(self):
        receipt = self.execute(returned="A hardcoded replacement")
        self.assertFalse(receipt["answer_origin_valid"])
        with self.assertRaises(ValueError):
            validate_closed_book_receipt(receipt, question=QUESTION)

    def test_rehashed_nonquestion_payload_cannot_become_closed_book(self):
        receipt = self.execute()
        receipt["trace"][0]["request"]["payload"]["instructions"] = "secret assistance"
        receipt["trace"][0]["payload_sha256"] = digest(receipt["trace"][0]["request"]["payload"])
        receipt["host_evidence_trace"]["final_observations"][0]["payload_sha256"] = receipt["trace"][0]["payload_sha256"]
        origin = execution._answer_origin_receipt(receipt["answer"],
            receipt["host_evidence_trace"]["final_observations"], receipt["trace"], execution_ok=True)
        receipt["host_answer_origin_validation"] = origin
        self.assertTrue(execution.validate_answer_origin(receipt)["valid"])
        with self.assertRaises(ValueError):
            validate_closed_book_receipt(receipt, question=QUESTION)

    def test_false_zero_counts_and_claimed_source_validity_are_rejected(self):
        original = self.execute()
        mutations = [lambda r: r["resource_usage"].update(read_calls=1),
                     lambda r: r["resource_usage"].update(search_calls=False),
                     lambda r: r.update(citation_source_valid=True),
                     lambda r: r.update(citation_source_valid=None),
                     lambda r: r["host_citation_validation"].update(presented_citation_ids=["e1"]),
                     lambda r: r["host_evidence_trace"]["read_presentations"].append({}),
                     lambda r: r.update(model_errors=["ModelResponseError"])]
        for index, mutate in enumerate(mutations):
            receipt = deepcopy(original)
            mutate(receipt)
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_closed_book_receipt(receipt, question=QUESTION)

    def test_different_banks_purchase_fresh_answers_same_bank_recovers(self):
        payload = {"question": QUESTION}
        self.model(bank="CB/0").complete("answer", payload)
        self.model(bank="CB/1").complete("answer", payload)
        resumed = self.model(bank="CB/0")
        self.assertEqual(resumed.complete("answer", payload)["answer"], ANSWER)
        self.assertEqual(len(self.transport.sent), 2)
        self.assertEqual(resumed.calls, 0)
        records = check_request_recovery(self.root)
        check_request_accounting(records, self.ledger)
        self.assertEqual(len(records), 2)
        self.assertEqual(self.ledger.summary()["used"]["run"]["calls"], 2)

    def test_closed_book_never_reuses_the_rag_answer_under_the_same_bank(self):
        rag = StructuredModel(self.root / "requests", self.ledger, self.transport,
                              bank="same-bank", prices=PRICES, limits={"answer": 800})
        payload = {"question": QUESTION}
        rag.complete("answer", payload)
        self.model(bank="same-bank").complete("answer", payload)
        self.assertEqual(len(self.transport.sent), 2)
        self.assertNotEqual(self.transport.sent[0]["messages"][0], self.transport.sent[1]["messages"][0])

    def test_unknown_outcome_is_not_retried_or_returned_as_a_quality_zero(self):
        sent = []
        def unknown(body):
            sent.append(body)
            raise TimeoutError("synthetic unknown outcome")
        model = ClosedBookModel(self.root / "requests", self.ledger, unknown,
            bank="CB/unknown", prices=PRICES, question=QUESTION)
        for _ in range(2):
            with self.assertRaises(UnknownProviderOutcome):
                ClosedBookEngine(model).solve({"question": QUESTION})
        self.assertEqual(len(sent), 1)
        self.assertEqual(self.ledger.summary()["used"]["run"]["calls"], 1)
        with self.assertRaises(UnknownProviderOutcome):
            check_request_recovery(self.root)

    def test_host_call_limit_and_ledger_stop_before_second_physical_request(self):
        broker = execution.HostBroker(task(), ClosedBookBackend(), self.model(), **closed_book_limits())
        request = {"stage": "answer", "payload": {"question": QUESTION}}
        broker("complete", request)
        with self.assertRaises(ValueError):
            broker("complete", request)
        self.assertEqual(len(self.transport.sent), 1)
        limited = Ledger(self.root / "limit-ledger.jsonl", {"run": {"calls": 0, "cny": 3}})
        model = ClosedBookModel(self.root / "limited-requests", limited, self.transport,
            bank="CB/limited", prices=PRICES, question=QUESTION)
        with self.assertRaises(LimitExceeded):
            model.complete("answer", {"question": QUESTION})
        self.assertEqual(len(self.transport.sent), 1)


if __name__ == "__main__":
    unittest.main()
