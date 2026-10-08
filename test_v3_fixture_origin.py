"""Synthetic test-fixture event construction, never a legacy-log migration path.

Call this only while building or deliberately editing synthetic test inputs.
Every result is checked by the production answer-origin validator; the helper
provides no bypass flag and never reads an execution receipt from disk.
"""
from copy import deepcopy

from code_rsi.budget import digest
from code_rsi.v3.execution import EXECUTION_SCHEMA, _answer_origin_receipt, validate_answer_origin


def bind_synthetic_origin(receipt, *, synthesize_final=False):
    """Return an independently validated synthetic execution-3 fixture copy.

    Existing final observations must have a corresponding answer event. Minimal
    test rows may explicitly request a newly constructed synthetic answer event;
    this option does not repair partially supplied answer events/observations.
    Changes to an answer, final response, evidence or event order require a fresh
    call during fixture construction. Score/repeat/node changes do not.
    """
    row = deepcopy(receipt)
    trace = row.setdefault("trace", [])
    evidence_trace = row.setdefault("host_evidence_trace", {})
    presentations = evidence_trace.setdefault("read_presentations", [])
    observations = evidence_trace.setdefault("final_observations", [])
    answer_indices = [i for i, event in enumerate(trace)
                      if event.get("name") == "complete"
                      and event.get("request", {}).get("stage") == "answer"
                      and event.get("model_completed", True)]
    if synthesize_final and not observations:
        if answer_indices:
            raise ValueError("synthetic answer event lacks its explicit observation")
        answer_indices.append(len(trace))
        trace.append({"name": "complete", "request": {"stage": "answer", "payload": {"evidence": []}}})
        observations.append({"evidence": {}, "response": {"answer": row["answer"],
                              "citation_ids": [], "evidence_sufficient": False}})
    if len(answer_indices) != len(observations):
        raise ValueError("synthetic answer observations must match answer events")

    read_indices = [i for i, event in enumerate(trace)
                    if event.get("name") == "complete" and event.get("request", {}).get("stage") == "read"]
    if len(read_indices) == len(presentations):
        for index, presentation in zip(read_indices, presentations):
            presentation["event_index"] = index
            trace[index]["request"].setdefault("payload", {}).setdefault(
                "sources", deepcopy(presentation.get("sources", [])))
    for index, observed in zip(answer_indices, observations):
        supplied = {key: {**deepcopy(value), "citation_id": key}
                    for key, value in observed["evidence"].items()}
        observed["evidence"] = supplied
        event = trace[index]
        event["request"].setdefault("payload", {})["evidence"] = list(deepcopy(supplied).values())
        observed["response"].setdefault("citation_ids", list(supplied))
        observed["event_index"] = index
        observed["response_sha256"] = digest(observed["response"])
        event["response_hash"] = observed["response_sha256"]
    for event in trace:
        if event.get("name") == "complete":
            event["request"].setdefault("payload", {})
            event["payload_sha256"] = digest(event["request"]["payload"])
            event.setdefault("model_completed", True)
    for index, observed in zip(answer_indices, observations):
        observed["payload_sha256"] = trace[index]["payload_sha256"]

    origin = _answer_origin_receipt(row["answer"], observations, trace, execution_ok=row["execution_ok"])
    row.update(schema=EXECUTION_SCHEMA, answer_origin_valid=origin["valid"],
               answer_origin_status=origin["status"], host_answer_origin_validation=origin)
    validate_answer_origin(row)
    return row
