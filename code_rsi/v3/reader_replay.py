"""Strict frozen-prefix replay for a final-reader intervention.

The original RAG engine owns evidence merging and final context construction.
This module does not retrieve new documents, score answers, read references or
create provider transports. Generated wrappers execute through the existing WSL
path; frozen candidate code is never dynamically executed by this module.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import re

from ..budget import digest, stable
from .datasets import validate_public_task
from .execution import HostError, _matches_source, _source_identity, root_files
from .infrastructure import ModelResponseError
from .rag import DEFAULTS, INSTRUCTIONS, SCHEMAS, RagEngine

SCHEMA = "rag-rsi-reader-replay-case-1"
_FIELDS = {"schema", "case_id", "task", "config", "target_read", "events",
           "engine_sha256", "original_final_payload_sha256", "source_binding"}
_HEADERS = ("question", "instructions", "stage_instructions", "additional_guidance", "output_schema")


class ReplayContractError(ValueError):
    """The frozen case cannot be replayed under the declared original engine."""


class ReplayMismatch(HostError):
    """A call diverged from the host-frozen prefix; it must never dispatch live."""


def _require(condition, message):
    if not condition:
        raise ReplayContractError(message)


def _snapshot(value):
    # Reject non-JSON/NaN values before producing isolated working copies.
    stable(value)
    return deepcopy(value)


def _guidance(value):
    if value is not None and (not isinstance(value, str) or len(value) > 8000):
        raise ReplayContractError("guidance must be a string of at most 8000 characters or None")
    return value


def _window(row):
    _require(_source_identity(row) is not None, "invalid frozen source window")
    if "text_sha256" in row:
        _require(row["text_sha256"] == hashlib.sha256(row["text"].encode()).hexdigest(),
                 "frozen window text hash differs")


def validate_case(case):
    """Return an isolated validated case; does no I/O except reading trusted source.

    Semantic evidence sufficiency is not asserted here. Exact quote/answer
    provenance is still independently enforced by HostBroker during execution.
    """
    _require(isinstance(case, dict) and set(case) == _FIELDS, "unexpected reader replay case fields")
    case = _snapshot(case)
    _require(case["schema"] == SCHEMA, "unsupported reader replay case schema")
    _require(isinstance(case["case_id"], str) and re.fullmatch(r"[A-Za-z0-9_-]{1,96}", case["case_id"]),
             "case_id must be a bounded anonymous identifier")
    validate_public_task(case["task"])
    _require(case["task"]["task_type"] == "qa", "reader replay requires a public QA task")
    _require(isinstance(case["config"], dict), "frozen configuration required")
    class Services:
        def search(self, *args): raise AssertionError("validation must not search")
        def complete(self, *args): raise AssertionError("validation must not generate")
    RagEngine(Services(), Services(), config=case["config"])
    config = {**DEFAULTS, **case["config"]}
    core = root_files(case["config"])["rag_core.py"]
    _require(case["engine_sha256"] == hashlib.sha256(core.encode("utf-8")).hexdigest(),
             "reader engine differs from frozen case")
    _require(type(case["target_read"]) is int and case["target_read"] >= 1,
             "target_read must identify the last positive reader ordinal")
    _require(isinstance(case["source_binding"], dict) and bool(case["source_binding"]),
             "source binding required")
    events = case["events"]
    _require(isinstance(events, list) and bool(events), "frozen events required")
    windows, read_count, target_index, answer_count = [], 0, None, 0
    for index, event in enumerate(events):
        _require(isinstance(event, dict) and set(event) == {"name", "request", "response", "response_sha256"},
                 "unexpected frozen event fields")
        name, request, response = event["name"], event["request"], event["response"]
        _require(name in {"search", "read", "complete"} and isinstance(request, dict),
                 "invalid replay event")
        _require(event["response_sha256"] == digest(response), "frozen response hash differs")
        if name == "search":
            _require(set(request) == {"query", "limit"} and isinstance(request["query"], str)
                     and bool(request["query"].strip()) and len(request["query"]) <= 16000
                     and type(request["limit"]) is int and 1 <= request["limit"] <= 30,
                     "invalid frozen search request")
            _require(isinstance(response, list) and len(response) <= request["limit"],
                     "invalid frozen search response")
            for row in response:
                _window(row); windows.append(row)
        elif name == "read":
            _require(set(request) == {"docid", "start", "end"}
                     and isinstance(request["docid"], (str, int)) and not isinstance(request["docid"], bool)
                     and type(request["start"]) is int and type(request["end"]) is int
                     and 0 <= request["start"] < request["end"], "invalid frozen source-read request")
            _window(response)
            _require((str(request["docid"]), request["start"], request["end"]) ==
                     (str(response["docid"]), response["start"], response["end"]),
                     "source-read response differs from requested span")
            windows.append(response)
        else:
            _require(set(request) == {"stage", "payload"} and request["stage"] in SCHEMAS
                     and isinstance(request["payload"], dict), "invalid frozen model request")
            stage, payload = request["stage"], request["payload"]
            _require(payload.get("question") == case["task"]["question"], "frozen question differs")
            _require(payload.get("instructions") == "" and payload.get("stage_instructions") == INSTRUCTIONS[stage]
                     and payload.get("additional_guidance") == config["prompts"].get(stage, "")
                     and payload.get("output_schema") == SCHEMAS[stage], "frozen engine prompt contract differs")
            _require(isinstance(response, dict) and set(response) - {"_meta"} == set(SCHEMAS[stage]),
                     "frozen model response schema differs")
            if "_meta" in response:
                _require(isinstance(response["_meta"], dict) and not response["_meta"].get("truncated")
                         and response["_meta"].get("finish_reason") not in {"length", "max_tokens", "error"},
                         "cannot replay incomplete historical model response")
            if stage == "read":
                read_count += 1
                _require(payload.get("round") == read_count, "reader ordinals differ")
                sources = payload.get("sources")
                _require(isinstance(sources, list), "frozen reader sources required")
                seen = set()
                for source in sources:
                    _window(source)
                    sid = source.get("source_id")
                    _require(isinstance(sid, str) and bool(sid) and sid not in seen
                             and _matches_source(source, windows), "reader source was not returned by frozen backend")
                    _require(str(source["docid"]) not in case["task"]["excluded_docids"], "excluded source in reader")
                    seen.add(sid)
                if read_count == case["target_read"]:
                    target_index = index
            elif stage == "answer":
                answer_count += 1
                _require(index == len(events) - 1, "only the last event may answer")
                _require(digest(payload) == case["original_final_payload_sha256"],
                         "original final payload hash differs")
    _require(read_count == case["target_read"] and answer_count == 1 and target_index == len(events) - 2,
             "target must be the last reader followed immediately by one answer")
    rounds = config["max_rounds"] if config["mode"] == "iterative" else 1
    _require(read_count <= rounds, "reader count exceeds frozen workflow")
    return case


def replay_files(case, guidance=None):
    """Use the current trusted root wrapper and unchanged original RagEngine.

    Only the target reader's additional_guidance is replaced. Capping rounds at
    its existing ordinal prevents a new reader response from adding searches.
    """
    case = validate_case(case); guidance = _guidance(guidance)
    config = deepcopy(case["config"])
    config["max_rounds"] = min(config.get("max_rounds", DEFAULTS["max_rounds"]), case["target_read"])
    files = root_files(config)
    original = """class Model:
    def __init__(self, services): self.services = services
    def complete(self, stage, payload):
        return self.services.call('complete', {'stage':stage,'payload':payload})
"""
    replacement = """class Model:
    def __init__(self, services):
        self.services = services
        self.read_ordinal = 0
    def complete(self, stage, payload):
        if stage == 'read':
            self.read_ordinal += 1
            if self.read_ordinal == %d and %r is not None:
                payload = dict(payload)
                payload['additional_guidance'] = %r
        return self.services.call('complete', {'stage':stage,'payload':payload})
""" % (case["target_read"], guidance, guidance)
    if files["rag.py"].count(original) != 1:
        raise ReplayContractError("trusted root model wrapper changed; explicit adapter update required")
    files["rag.py"] = files["rag.py"].replace(original, replacement)
    return files


class ReplayRouter:
    """Host-side strict event router; live requests are only final read + answer."""
    def __init__(self, case, live_model=None, guidance=None):
        self.case = validate_case(case)
        self.guidance = _guidance(guidance)
        if live_model is not None and not callable(getattr(live_model, "complete", None)):
            raise ReplayContractError("live_model must support complete")
        self.live_model = live_model
        self.backend = self.model = self
        self.identity = digest({"schema": SCHEMA, "case": digest(self.case), "guidance": self.guidance})
        self.replayed_calls = self.new_calls = 0
        self._counts = {"replayed_model_calls": 0, "new_model_calls": 0,
                        "replayed_search_calls": 0, "replayed_reads": 0}
        self._cursor = 0
        self._fatal = None
        self._timeout_seconds = 150
        self._target_index = len(self.case["events"]) - 2

    @property
    def timeout_seconds(self):
        return self._timeout_seconds

    @timeout_seconds.setter
    def timeout_seconds(self, value):
        self._timeout_seconds = value
        if self.live_model is not None and hasattr(self.live_model, "timeout_seconds"):
            self.live_model.timeout_seconds = value

    def _reject(self, message):
        self._fatal = ReplayMismatch(message)
        raise self._fatal

    def _call(self, name, request):
        if self._fatal is not None:
            raise self._fatal
        if self._cursor >= len(self.case["events"]):
            self._reject("extra call after replay completion")
        event = self.case["events"][self._cursor]
        expected = deepcopy(event["request"])
        target = self._cursor == self._target_index
        final_live = self.live_model is not None and self._cursor == len(self.case["events"]) - 1
        if target and self.guidance is not None:
            expected["payload"]["additional_guidance"] = self.guidance
        if name != event["name"]:
            self._reject("replay service order differs")
        if final_live:
            if (set(request) != {"stage", "payload"} or request.get("stage") != "answer"
                    or not isinstance(request.get("payload"), dict)
                    or set(request["payload"]) != set(expected["payload"])
                    or any(request["payload"].get(key) != expected["payload"].get(key) for key in _HEADERS)):
                self._reject("live final engine contract differs")
        elif digest(request) != digest(expected):
            self._reject("replay request differs from frozen prefix")
        if self.live_model is not None and (target or final_live):
            self.new_calls += 1
            self._counts["new_model_calls"] += 1
            try:
                response = self.live_model.complete(request["stage"], _snapshot(request["payload"]))
            except ModelResponseError:
                # This completed billed request is not retried. HostBroker owns
                # its model-error handling, after which the original engine can
                # still reserve a final answer from its retained prefix.
                self._cursor += 1
                raise
            except BaseException as exc:
                self._fatal = exc
                raise
            self._cursor += 1
            return _snapshot(response)
        self._cursor += 1
        self.replayed_calls += 1
        key = {"complete": "replayed_model_calls", "search": "replayed_search_calls", "read": "replayed_reads"}[name]
        self._counts[key] += 1
        return deepcopy(event["response"])

    def search(self, query, limit=5):
        return self._call("search", {"query": query, "limit": limit})

    def read(self, docid, start, end):
        return self._call("read", {"docid": docid, "start": start, "end": end})

    def complete(self, stage, payload):
        return self._call("complete", {"stage": stage, "payload": payload})

    def summary(self):
        return {**self._counts, "complete": self._cursor == len(self.case["events"]) and self._fatal is None}

    def assert_complete(self):
        if self._fatal is not None:
            raise self._fatal
        if self._cursor != len(self.case["events"]):
            raise ReplayMismatch("replay stopped before all frozen events were consumed")
        return self.summary()
