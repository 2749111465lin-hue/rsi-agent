"""Final-answer input ablation over a strictly replayed RAG prefix.

Both conditions preserve question, instructions and grounded evidence exactly.
No new reader/search calls are allowed; only the last answer may be purchased.
"""
from copy import deepcopy

from .budget import digest
from .v3.execution import root_files
from .v3.rag import RagEngine
from .v3.infrastructure import ModelResponseError
from .v3.reader_replay import (
    ReplayContractError, ReplayRouter, validate_case,
)

ARMS = ("full_state", "evidence_only")
CONTEXT_ARMS = ("full_state", "quote_context")
HEADERS = ("question", "instructions", "stage_instructions",
           "additional_guidance", "output_schema")
JUDGMENTS = ("constraints", "claims", "gaps", "conflicts",
             "stop_reason", "unresolved_conflict_count")
FULL_FIELDS = set(HEADERS) | set(JUDGMENTS) | {"evidence"}
EVIDENCE_FIELDS = set(HEADERS) | {"evidence"}


def project_final_payload(payload, arm):
    """A closed field projection, never a semantic rewrite or quote cleanup."""
    if arm not in ARMS:
        raise ReplayContractError("unknown final-answer condition")
    if not isinstance(payload, dict) or set(payload) != FULL_FIELDS:
        raise ReplayContractError("frozen final payload fields differ")
    if not isinstance(payload["evidence"], list):
        raise ReplayContractError("frozen final evidence must be a list")
    quote_fields = {"citation_id", "source_id", "docid", "start", "end", "quote", "source_verified"}
    if any(not isinstance(row, dict) or set(row) != quote_fields for row in payload["evidence"]):
        raise ReplayContractError("unexpected metadata in frozen evidence")
    digest(payload)  # Reject non-JSON/non-finite values.
    keep = FULL_FIELDS if arm == "full_state" else EVIDENCE_FIELDS
    return {key: deepcopy(value) for key, value in payload.items() if key in keep}


def project_case_payload(case, arm):
    """Render the maintained context option over an exact historical prefix.

    This uses the same engine as the isolated program, without buying a model
    response or embedding task-specific text in source. The historical answer
    is only a placeholder to finish rendering, never a new-arm measurement.
    """
    case = validate_case(case)
    original = project_final_payload(case["events"][-1]["request"]["payload"], "full_state")
    if arm != "quote_context":
        return project_final_payload(original, arm)
    if case["config"].get("final_context_radius", 0) != 0:
        raise ReplayContractError("quote-context baseline must have radius zero")

    class Projector(ReplayRouter):
        projected = None
        def _call(self, name, request):
            if self._cursor != len(self.case["events"]) - 1:
                return super()._call(name, request)
            if name != "complete" or request.get("stage") != "answer":
                self._reject("context projection must end with one answer")
            self.projected = deepcopy(request["payload"])
            self._cursor += 1
            return deepcopy(self.case["events"][-1]["response"])

    router = Projector(case)
    RagEngine(router, router, config={**case["config"], "final_context_radius": 256}).solve(
        {"question": case["task"]["question"]})
    router.assert_complete()
    if router.projected is None:
        raise ReplayContractError("context projection did not produce a final payload")
    stripped = deepcopy(router.projected)
    for item in stripped.get("evidence", []):
        item.pop("context", None)
    if digest(stripped) != digest(original):
        raise ReplayContractError("context option changed fields beyond source neighborhoods")
    return router.projected


def replay_files(case, arm="full_state"):
    case = validate_case(case)
    project_case_payload(case, arm)
    if arm == "quote_context":
        return root_files({**case["config"], "final_context_radius": 256})
    files = root_files(case["config"])
    original = """class Model:
    def __init__(self, services): self.services = services
    def complete(self, stage, payload):
        return self.services.call('complete', {'stage':stage,'payload':payload})
"""
    if files["rag.py"].count(original) != 1:
        raise ReplayContractError("trusted root wrapper changed")
    if arm == "full_state":
        return files
    replacement = """class Model:
    def __init__(self, services): self.services = services
    def complete(self, stage, payload):
        if stage == 'answer':
            if set(payload) != set(%r):
                raise ValueError('frozen final payload fields differ')
            payload = {key: value for key, value in payload.items() if key in %r}
        return self.services.call('complete', {'stage':stage,'payload':payload})
""" % (tuple(sorted(FULL_FIELDS)), tuple(sorted(EVIDENCE_FIELDS)))
    files["rag.py"] = files["rag.py"].replace(original, replacement)
    return files


class AnswerReplayRouter(ReplayRouter):
    """Reuse transport/error plumbing; enforce an exact one-answer suffix."""
    def __init__(self, case, live_model=None, arm="full_state"):
        super().__init__(case, live_model=live_model)
        self.arm = arm
        self.projected_payload = project_case_payload(self.case, arm)
        if live_model is None and arm != "full_state":
            raise ReplayContractError("changed answer input requires a fresh response")
        self.identity = digest({"schema": "rag-rsi-answer-replay-1",
                                "case": digest(self.case), "arm": arm})

    def _call(self, name, request):
        if self._fatal is not None:
            raise self._fatal
        if self._cursor >= len(self.case["events"]):
            self._reject("extra call after replay completion")
        event = self.case["events"][self._cursor]
        final = self._cursor == len(self.case["events"]) - 1
        expected = ({"stage": "answer", "payload": self.projected_payload}
                    if final else event["request"])
        if name != event["name"] or digest(request) != digest(expected):
            self._reject("request differs from exact frozen prefix or final projection")
        if final and self.live_model is not None:
            self.new_calls += 1
            self._counts["new_model_calls"] += 1
            try:
                response = self.live_model.complete("answer", deepcopy(self.projected_payload))
            except ModelResponseError:
                self._cursor += 1
                raise
            except BaseException as error:
                self._fatal = error
                raise
            self._cursor += 1
            return deepcopy(response)
        self._cursor += 1
        self.replayed_calls += 1
        key = {"complete": "replayed_model_calls", "search": "replayed_search_calls",
               "read": "replayed_reads"}[name]
        self._counts[key] += 1
        return deepcopy(event["response"])
