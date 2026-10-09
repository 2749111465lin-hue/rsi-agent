"""A frozen, explicit description of capabilities available to RAG candidates.

This is an instruction contract, not an additional executor or quality gate. The
caller must rebuild it from the active host/model settings when freezing a run.
The enclosing plan, model identity and request body bind its exact contents.
"""
from __future__ import annotations

from copy import deepcopy

from ..budget import stable
from .proposal_protocol import validate_proposal_protocol

SCHEMA = "rag-rsi-candidate-runtime-contract-1"
QA_STAGES = ("plan", "read", "answer")
HOST_FIELDS = {"max_models", "max_reads", "max_searches"}


def _integer(value, lower, upper, name):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError("invalid runtime contract " + name)
    return value


def _construct(host_limits, model_config, proposal_protocol, qa_prompts):
    if not isinstance(host_limits, dict) or set(host_limits) != HOST_FIELDS:
        raise ValueError("runtime contract requires exact host limits")
    host = {key: _integer(host_limits[key], 1 if key == "max_models" else 0, 64, key)
            for key in sorted(HOST_FIELDS)}
    if not isinstance(model_config, dict):
        raise ValueError("runtime contract requires model configuration")
    maximum = _integer(model_config.get("max_input_bytes"), 1, 120000, "max_input_bytes")
    limits = model_config.get("output_limits")
    if not isinstance(limits, dict) or not set(QA_STAGES) <= set(limits):
        raise ValueError("runtime contract requires QA output limits")
    output = {stage: _integer(limits[stage], 1, 32768, stage + " output limit")
              for stage in QA_STAGES}
    protocol = validate_proposal_protocol(proposal_protocol)
    if protocol is None or protocol["format"] != "exact_edits":
        raise ValueError("runtime contract requires exact_edits")
    if (not isinstance(qa_prompts, dict) or not set(QA_STAGES) <= set(qa_prompts)
            or any(not isinstance(qa_prompts[stage], str) or not qa_prompts[stage].strip()
                   for stage in QA_STAGES)):
        raise ValueError("runtime contract requires immutable QA prompts")
    prompts = {stage: qa_prompts[stage] for stage in QA_STAGES}
    card = {
        "schema": SCHEMA,
        "entrypoint": "solve(question, services) in rag.py; rag_core.py is the other editable file",
        "host_limits": host,
        "model_request_limits": {"max_input_bytes": maximum, "output_limits": output},
        "immutable_qa_system_prompts": prompts,
        "capabilities": {
            "complete": {
                "request": {"stage": "plan | read | answer", "payload": "JSON object"},
                "stages_are_exhaustive": True,
                "unsupported_examples": ["answer_retry", "verify", "develop"],
                "response_schemas": {
                    "plan": {"constraints": ["constraint"], "queries": ["search query"]},
                    "read": {"claims": [{"text": "claim", "citations": [
                        {"source_id": "s1", "quote": "exact unique quote"}]}],
                        "bridge_entities": ["entity literally present in question or source"],
                        "gaps": ["missing fact"], "conflicts": ["unresolved conflict"],
                        "queries": ["next query using read evidence"], "ready": False},
                    "answer": {"answer": "short answer", "citation_ids": ["e1"],
                               "evidence_sufficient": True}},
            },
            "search": {"request": {"query": "nonempty string, at most 16000 characters", "limit": "integer 1..30"},
                       "response": "source windows with host source_id and absolute offsets"},
            "read": {"request": {"docid": "existing nonexcluded document id",
                                 "start": "absolute character offset",
                                 "end": "exclusive absolute character offset"},
                     "response": "host source window; bounded by the fixed backend"},
            "record_trace": {"request": {"result": "JSON object"}, "maximum_calls": 1},
        },
        "call_budget_rule": "plan/read may consume at most max_models-1 calls; answer may use the reserved final call. Another answer is allowed only if total budget remains; use the supported answer stage, not an invented stage.",
        "answer_origin_rule": "Return the answer string from the last successful answer call; modifying its content later makes origin invalid. Citation identity/source checking is separate from semantic correctness.",
        "evidence_rule": "Read-stage quotations must match presented source windows. Answer-stage evidence must have prior read-stage provenance. Bridge entity occurrence is not proof of the relationship required by the question.",
        "optimization_freedom": "Use LLMs for semantic reasoning. You may change reusable candidate logic and prompts within the declared edit policy; host capabilities, scoring, data roles and total budgets remain fixed.",
        "edit_output": {
            "top_level_keys": ["parent_source_sha256", "change_status", "edits", "mechanism",
                               "intended_target_module"],
            "edit_keys": ["file", "old", "new"],
            "extra_keys_allowed": False,
            "max_edits": protocol["max_edits"],
            "max_edit_chars": protocol["max_edit_chars"],
            "anchor_rule": "Every old fragment must be nonempty, occur exactly once in the original parent and differ from new; edit intervals cannot overlap.",
            "character_budget_rule": "sum(len(edit.old)+len(edit.new)) over all edits <= max_edit_chars; lengths count Unicode characters, not UTF-8 bytes.",
            "json_rule": "Emit a single complete JSON object. Do not repeat JSON keys, add helper fields such as old_occurrences, or append repeated replacement text.",
            "no_change_rule": "change_status=no_change requires edits=[]; no-op edits are not a no_change response.",
            "minimal_example": {
                "parent_source_sha256": "<copy supplied identity>", "change_status": "modified",
                "edits": [{"file": "rag_core.py", "old": "<unique original source fragment>",
                           "new": "<changed replacement fragment>"}],
                "mechanism": "<reusable mechanism; state which returned behavior actually changes>",
                "intended_target_module": "answer_generation"},
        },
    }
    return card


def build_runtime_contract(host_limits, model_config, proposal_protocol, qa_prompts):
    """Project real settings into a detached card without files, keys or examples."""
    return _construct(host_limits, model_config, proposal_protocol, qa_prompts)


def validate_runtime_contract(value):
    """Strictly validate a closed card, returning None for the legacy interface."""
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("explicit runtime contract schema required")
    try:
        fields = deepcopy(value)
        output = fields["edit_output"]
        expected = _construct(fields["host_limits"], fields["model_request_limits"],
                              {"schema": "rag-rsi-proposal-protocol-1", "format": "exact_edits",
                               "max_edits": output["max_edits"], "max_edit_chars": output["max_edit_chars"]},
                              fields["immutable_qa_system_prompts"])
        # Canonical JSON equality preserves type distinctions such as 1 vs True.
        if stable(value) != stable(expected):
            raise ValueError("runtime contract structure or fixed capabilities changed")
    except (KeyError, TypeError, UnicodeError) as error:
        raise ValueError("invalid runtime contract structure") from error
    return deepcopy(value)
