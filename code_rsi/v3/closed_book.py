"""Question-only baseline through the existing model, ledger and WSL boundaries.

This component does not relax the RAG model or host defaults. The runner must
freeze the returned source files and explicitly supply closed_book_limits().
Citation provenance is inapplicable; answer provenance remains mandatory.
"""
from __future__ import annotations

import ast
from pathlib import Path

from ..budget import digest
from .execution import HostError, validate_answer_origin
from .infrastructure import StructuredModel


CLOSED_BOOK_PROFILE = "rag-rsi-closed-book-1"
CLOSED_BOOK_SYSTEM = (
    "Answer the original question using your own learned knowledge. No documents "
    "or retrieval tools are available. Give the shortest complete entity, title, "
    "date or phrase that answers the question. Check its conditions, relation "
    "direction, dates, negation and attribution. If you do not know the answer, "
    "return 'Insufficient information'. Do not invent citations or provide "
    "reasoning. Return JSON only with exactly {answer:string,citation_ids:[]}; "
    "citation_ids must always be an empty array."
)


def _question(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 16000:
        raise ValueError("bounded nonempty closed-book question required")
    return value


def closed_book_request_body(question, *, model="deepseek-flash", max_tokens=800):
    """Build the exact provider body without ledger, cache or credential access."""
    question = _question(question)
    if not isinstance(model, str) or not model:
        raise ValueError("closed-book model identity required")
    # Reuse the maintained provider envelope; constructing a shape has no I/O.
    shape = object.__new__(StructuredModel)
    shape.model, shape.limits = model, {"answer": max_tokens}
    body = StructuredModel.request_body(shape, "answer", {"question": question})
    body["messages"][0]["content"] = CLOSED_BOOK_SYSTEM
    return body


class ClosedBookModel(StructuredModel):
    """Host-fixed question and profile; candidates cannot select another prompt."""

    def __init__(self, *args, question, profile=CLOSED_BOOK_PROFILE, **kwargs):
        if profile != CLOSED_BOOK_PROFILE:
            raise ValueError("unknown closed-book profile")
        self._question = _question(question)
        super().__init__(*args, **kwargs)
        self.identity = digest({"model_identity": self.identity,
            "answer_profile": {"name": CLOSED_BOOK_PROFILE, "system": CLOSED_BOOK_SYSTEM,
                               "response_fields": ["answer", "citation_ids"], "citation_ids": []},
            "question_sha256": digest(self._question)})

    @property
    def question(self):
        return self._question

    @property
    def profile(self):
        return CLOSED_BOOK_PROFILE

    def request_body(self, stage, payload):
        if (stage != "answer" or not isinstance(payload, dict)
                or set(payload) != {"question"} or payload["question"] != self._question):
            raise ValueError("closed-book accepts only its frozen question and one answer stage")
        return closed_book_request_body(self._question, model=self.model,
                                        max_tokens=self.limits["answer"])


class ClosedBookBackend:
    """An explicit unavailable source capability, including for direct callers."""
    identity = digest({"kind": "closed_book", "profile": CLOSED_BOOK_PROFILE,
                       "retrieval_available": False})

    def search(self, *args, **kwargs):
        raise HostError("closed-book retrieval is unavailable")

    def read(self, *args, **kwargs):
        raise HostError("closed-book document reading is unavailable")


class ClosedBookEngine:
    """Pure maintained program: one question-only call and unchanged answer."""

    def __init__(self, model):
        if not callable(getattr(model, "complete", None)):
            raise ValueError("closed-book model service required")
        self.model = model

    def solve(self, task):
        if (not isinstance(task, dict) or set(task) != {"question"}
                or not isinstance(task["question"], str) or not task["question"].strip()
                or len(task["question"]) > 16000):
            raise ValueError("closed-book task must contain only its original question")
        result = self.model.complete("answer", {"question": task["question"]})
        if (not isinstance(result, dict) or set(result) != {"answer", "citation_ids"}
                or not isinstance(result["answer"], str) or not result["answer"].strip()
                or len(result["answer"]) > 1000 or result["citation_ids"] != []):
            raise ValueError("invalid closed-book answer response")
        answer = result["answer"]
        return {"answer": answer, "answer_usable": True, "citation_ids": [],
                "abstained": answer.strip().casefold() in
                    {"insufficient information", "unknown", "i don't know"},
                "source_status": "not_applicable_closed_book"}


def closed_book_files():
    """Fixed source only; questions, references and model outputs never enter it."""
    source = Path(__file__).read_text(encoding="utf-8")
    classes = [node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == "ClosedBookEngine"]
    if len(classes) != 1:
        raise ValueError("maintained closed-book engine source changed")
    core = ast.get_source_segment(source, classes[0]) + "\n"
    wrapper = """from rag_core import ClosedBookEngine
class Model:
    def __init__(self, services): self.services = services
    def complete(self, stage, payload):
        return self.services.call('complete', {'stage':stage,'payload':payload})
def solve(question, services):
    result = ClosedBookEngine(Model(services)).solve({'question':question})
    services.call('record_trace', {'result':result})
    return {'answer':result['answer'], 'citations':[],
            'abstention_reason':'closed_book_uncertain' if result['abstained'] else None}
"""
    return {"rag.py": wrapper, "rag_core.py": core}


def closed_book_limits():
    return {"max_models": 1, "max_searches": 0, "max_reads": 0}


def validate_closed_book_receipt(receipt, *, question):
    """Validate host observations, returning N/A only for a true closed-book cell.

    Source/program identity and WSL isolation must additionally be checked by
    the enclosing runner. This function never trusts candidate diagnostic flags.
    """
    question = _question(question)
    origin = validate_answer_origin(receipt)
    if (origin["valid"] is not True or receipt.get("execution_ok") is not True
            or receipt.get("answer_usable") is not True or receipt.get("model_errors") != []
            or receipt.get("failure_classes") != []):
        raise ValueError("closed-book measurement needs a completed observed model answer")
    usage = receipt.get("resource_usage")
    expected = {"model_calls": 1, "search_calls": 0, "read_calls": 0}
    if (not isinstance(usage, dict) or set(usage) != set(expected)
            or any(type(usage[key]) is not int or usage[key] != value for key, value in expected.items())):
        raise ValueError("closed-book must make exactly one model call and no source calls")
    trace = receipt["trace"]
    if (len(trace) not in (1, 2) or trace[0].get("name") != "complete"
            or trace[0].get("request") != {"stage": "answer", "payload": {"question": question}}
            or trace[0].get("model_completed") is not True
            or (len(trace) == 2 and trace[1].get("name") != "record_trace")):
        raise ValueError("closed-book trace must contain one question-only answer")
    host = receipt["host_evidence_trace"]
    if (set(host) != {"read_presentations", "final_observations"}
            or host["read_presentations"] != [] or len(host["final_observations"]) != 1):
        raise ValueError("closed-book cannot contain reader or source observations")
    observed = host["final_observations"][0]
    response = observed["response"]
    if (observed["evidence"] != {} or set(response) != {"answer", "citation_ids"}
            or response["citation_ids"] != [] or response["answer"] != receipt["answer"]
            or not isinstance(response["answer"], str) or not response["answer"].strip()
            or len(response["answer"]) > 1000 or receipt.get("citations") != []):
        raise ValueError("closed-book answer or empty-source contract differs")
    cited = receipt.get("host_citation_validation")
    if (receipt.get("citation_source_valid") is not False
            or receipt.get("citation_status") != "missing_citations"
            or not isinstance(cited, dict) or cited.get("valid") is not False
            or cited.get("status") != "missing_citations"
            or cited.get("raw_citation_ids") != [] or cited.get("presented_citation_ids") != []
            or cited.get("model_claims_evidence") is not None):
        raise ValueError("closed-book source provenance must not be claimed valid")
    return {"profile": CLOSED_BOOK_PROFILE, "answer_origin_valid": True,
            "source_status": "not_applicable_closed_book", "source_valid": None,
            "model_calls": 1, "search_calls": 0, "read_calls": 0}
