"""Reconstruct fixed MuSiQue answer states from an authenticated prior run.

Only host-owned, frozen D_fit material is read. No credentials, references,
provider dispatch, grader output, or arbitrary archived code are used. The
trusted current RagEngine is replayed with the exact original configuration.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .budget import digest
from .v3 import calibration as cal, musique_calibration as musique
from .v3.evolution import freeze
from .v3.execution import HostBroker, root_files, validate_answer_origin
from .v3.infrastructure import StructuredModel
from .v3.rag import RagEngine
from .v3.reader_replay import SCHEMA, ReplayRouter, validate_case

PACKET_SCHEMA = "rag-rsi-reader-replay-cases-1"
SELECTION_RULE = "all_source_question_ids_in_frozen_order__support__repeat_0__without_scores"


def _snapshot_file(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def _unchanged(binding):
    if hashlib.sha256(Path(binding["path"]).read_bytes()).hexdigest() != binding["sha256"]:
        raise ValueError("source artifact changed during answer preparation")


class _RequestShape(StructuredModel):
    """Reuse the exact provider body constructor without I/O initialization."""
    def __init__(self, spec):
        self.model = spec["name"]
        self.limits = {"plan": 1200, "read": 2200, "answer": 800, "develop": 18000}
        self.limits.update(spec["output_limits"])
        self.max_input_bytes = spec["max_input_bytes"]


def build_cases(source_plan):
    """Return every support/repeat-0 state, selected without opening scores.

    Accept the original plan dictionary or its JSON path. Membership, archive,
    execution, accounting and source runtime are checked through calibration's
    existing complete-generation validator before any state is reconstructed.
    """
    if isinstance(source_plan, (str, Path)):
        plan, supplied_binding = _snapshot_file(source_plan)
    elif isinstance(source_plan, dict):
        plan, supplied_binding = deepcopy(source_plan), None
    else:
        raise ValueError("source plan must be a frozen plan object or path")
    if plan.get("schema") != musique.SCHEMA or plan.get("data_role") != "D_fit":
        raise ValueError("only the original MuSiQue D_fit calibration is eligible")
    out = Path(plan["output_dir"]).resolve(strict=True)
    stored, plan_binding = _snapshot_file(out / "plan.json")
    if stored != plan:
        raise ValueError("source plan differs from its run")
    frozen, freeze_binding = _snapshot_file(out / "generation_freeze.json")
    cells = cal._validated_generation(plan, frozen)
    tasks = cal._task_snapshot(plan)
    byqid = {task["question_id"]: task for task in tasks}
    backends, _ = cal._musique_environment(plan, tasks)
    selected = {item["identity"]["question_id"]: item for item in cells
                if item["identity"]["arm"] == "support" and item["identity"]["repeat"] == 0}
    if set(selected) != set(plan["question_ids"]):
        raise ValueError("source support repeat-0 panel incomplete")
    config = next(arm["config"] for arm in plan["arms"] if arm["name"] == "support")
    shape = _RequestShape(plan["model"])
    cases = []
    bindings = [plan_binding, freeze_binding] + ([supplied_binding] if supplied_binding else [])
    for ordinal, qid in enumerate(plan["question_ids"], 1):
        item, task = selected[qid], byqid[qid]
        receipt = item["payload"]
        validate_answer_origin(receipt)
        if (not receipt["execution_ok"] or not receipt["answer_origin_valid"]
                or not receipt["answer_usable"] or receipt["model_errors"]):
            raise ValueError("selected source state lacks a complete usable answer")
        raw_cell, cell_binding = _snapshot_file(cal._cell_path(out, item["file"]))
        if raw_cell["identity"] != item["identity"] or raw_cell["payload"] != receipt:
            raise ValueError("source cell changed during reconstruction")
        bindings.append(cell_binding)
        bank = f"calibration/{qid}/0"
        model_bindings = []

        class CachedModel:
            def complete(self, stage, payload):
                body = shape.request_body(stage, payload)
                key = digest({"body": body, "bank": bank})
                record, file_binding = _snapshot_file(out / "requests" / (key + ".json"))
                if (record.get("key") != key or record.get("state") != "settled"
                        or record.get("body") != body):
                    raise ValueError("source model request differs from its exact bank and body")
                choices = record.get("response", {}).get("choices", [])
                if not choices or choices[0].get("finish_reason") != "stop":
                    raise ValueError("source model result incomplete")
                result = json.loads(choices[0].get("message", {}).get("content", ""))
                if not isinstance(result, dict):
                    raise ValueError("source model result is not an object")
                model_bindings.append({"event_index": len(broker.events), "stage": stage,
                    "bank": bank, "request_key": key, "body_sha256": digest(body),
                    "response_sha256": digest(result), "file": file_binding})
                bindings.append(file_binding)
                return deepcopy(result)

        broker = HostBroker(task, backends[(qid, "support")], CachedModel(), **plan["limits"])
        events = []
        for index, event in enumerate(receipt["trace"]):
            name, request = event["name"], event["request"]
            if name not in {"complete", "search", "read", "record_trace"}:
                raise ValueError("unsupported source event")
            if name == "record_trace" and index != len(receipt["trace"]) - 1:
                raise ValueError("source diagnostic must follow the final answer")
            if name == "complete" and event.get("model_completed") is not True:
                raise ValueError("cannot replay incomplete source model events")
            response = broker(name, request)
            if digest(response) != event["response_hash"]:
                raise ValueError("source event response differs from reconstruction")
            if name != "record_trace":
                events.append({"name": name, "request": deepcopy(request),
                               "response": response, "response_sha256": digest(response)})
        if (broker.events != receipt["trace"] or broker.reported != receipt["candidate_reported"]
                or broker.reported is None or broker.model_errors != receipt["model_errors"]
                or broker.counts != receipt["resource_usage"]
                or {"read_presentations": broker.read_presentations,
                    "final_observations": broker.final_observations} != receipt["host_evidence_trace"]
                or broker.answer_origin_receipt(receipt["answer"]) != receipt["host_answer_origin_validation"]
                or broker.citation_receipt(receipt["answer"], receipt["citations"]) != receipt["host_citation_validation"]):
            raise ValueError("reconstructed host observations differ from source receipt")
        case = {"schema": SCHEMA, "case_id": f"Q{ordinal:02d}", "task": deepcopy(task),
            "config": deepcopy(config),
            "target_read": sum(e["name"] == "complete" and e["request"]["stage"] == "read" for e in events),
            "events": events,
            "engine_sha256": hashlib.sha256(root_files(config)["rag_core.py"].encode()).hexdigest(),
            "original_final_payload_sha256": digest(events[-1]["request"]["payload"]),
            "source_binding": {"kind": "frozen_musique_calibration_support_repeat_0",
                "selection_rule": SELECTION_RULE, "source_plan": plan_binding,
                "source_plan_hash": digest(plan), "source_generation_freeze": freeze_binding,
                "source_cell": cell_binding, "source_identity": deepcopy(item["identity"]),
                "model_requests": model_bindings}}
        case = validate_case(case)
        router = ReplayRouter(case)
        # Original config is essential: capping rounds can alter stop_reason.
        reconstructed = RagEngine(router, router, config=case["config"]).solve({"question": task["question"]})
        router.assert_complete()
        if reconstructed != receipt["candidate_reported"]:
            raise ValueError("original engine no longer reproduces the frozen state")
        cases.append(case)
    for binding in bindings:
        _unchanged(binding)
    return {"schema": PACKET_SCHEMA, "cases": cases}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    target = Path(args.output).resolve()
    runs = Path(__file__).resolve().parents[1] / "runs"
    if runs.resolve() not in target.parents:
        raise ValueError("answer preparation output must remain under project runs")
    packet = build_cases(args.source_plan)
    freeze(target, packet)
    print(json.dumps({"cases": len(packet["cases"]), "packet_hash": digest(packet),
        "file_sha256": cal.file_hash(target), "source_selection": SELECTION_RULE,
        "new_api_calls": 0, "references_parsed": False, "credentials_read": False}))
    return packet


if __name__ == "__main__":
    main()
