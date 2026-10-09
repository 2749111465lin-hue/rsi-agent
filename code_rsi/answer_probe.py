"""Bounded final-answer input experiment; no live retrieval or reader purchase.

Shared archive, sandbox, accounting, recovery and metrics come from v3.
Only the experiment contract and its one-answer suffix are implemented here.
"""
from copy import deepcopy
import argparse
import hashlib
import json
import math
from pathlib import Path

from .archive import ProgramArchive
from .budget import Ledger, digest, save
from .answer_replay import ARMS, CONTEXT_ARMS, AnswerReplayRouter, project_case_payload, replay_files
from .v3.calibration import (run_lock, _verified_bytes, _saved_execution, _execution_facts,
                            _cell_path, credential_from_plan, file_hash)
from .v3.evolution import freeze, read, recoverable_record, _runtime_source_hashes
from .v3.execution import execute, HostBroker, HostError
from .v3.infrastructure import StructuredModel, deepseek_transport
from .v3.reader_probe import (_file_contract, _RequestShape, _order, _relative,
                              _limits, _identity, _accounting)
from .v3.reader_replay import validate_case
from .v3.paired_analysis import validate_analysis, paired_analysis, SINGLE_CONTRAST_SCHEMA
from .v3.task_metrics import score_task

SCHEMA = "rag-rsi-answer-probe-1"
CONTEXT_SCHEMA = "rag-rsi-answer-probe-2"
CONTEXT_ADVANCE = {"mean_f1_gain_gt": 0, "mean_em_gain_gte": 0,
                   "nonabstaining_f1_zero_increase_lte": 0}


def _contract(plan):
    if plan.get("schema") == SCHEMA and plan.get("purpose") == "fixed_evidence_final_judgment_ablation":
        return {"arms": ARMS, "contrast": "remove_derived_judgments", "source_arm": "support"}
    if (plan.get("schema") == CONTEXT_SCHEMA and plan.get("purpose") == "fixed_evidence_quote_context"
            and plan.get("source_arm") == "loop"
            and digest(plan.get("advance_criteria")) == digest(CONTEXT_ADVANCE)):
        return {"arms": CONTEXT_ARMS, "contrast": "restore_source_context", "source_arm": "loop"}
    raise ValueError("explicit versioned final-answer diagnostic contract required")


def _arms(plan):
    return _contract(plan)["arms"]

HELPERS = ("answer_replay.py", "answer_probe.py", "prepare_answer_probe.py")


def helper_hashes():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in HELPERS}


def _sources(plan):
    if (plan["runtime_source_hashes"] != _runtime_source_hashes()
            or plan["helper_source_hashes"] != helper_hashes()):
        raise ValueError("runtime or answer-probe source changed after freeze")


def _snapshot(plan):
    _file_contract(plan["cases_file"])
    packet = json.loads(_verified_bytes(plan["cases_file"]))
    if (not isinstance(packet, dict) or set(packet) != {"schema", "cases"}
            or packet["schema"] != "rag-rsi-reader-replay-cases-1"
            or not isinstance(packet["cases"], list) or not 1 <= len(packet["cases"]) <= 64):
        raise ValueError("bounded complete replay packet required")
    cases = [validate_case(case) for case in packet["cases"]]
    if (len({c["case_id"] for c in cases}) != len(cases)
            or len({c["task"]["question_id"] for c in cases}) != len(cases)
            or any(c["task"]["dataset"] != "musique" for c in cases)):
        raise ValueError("one fixed state per MuSiQue question required")
    if plan["question_use"] == "used_diagnostic":
        _file_contract(plan["source_plan_file"])
        source = json.loads(_verified_bytes(plan["source_plan_file"]))
        from .prepare_answer_probe import build_cases
        kwargs = ({"source_arm": "loop", "source_checkout": plan["source_checkout"]}
                  if plan["schema"] == CONTEXT_SCHEMA else {})
        if build_cases(source, **kwargs) != packet:
            raise ValueError("case packet is not the complete declared source/repeat-0 panel")
        if plan["references_file"] != source["references_file"]:
            raise ValueError("reference binding differs from source panel")
        if plan["analysis"]["question_groups"] != source["analysis"]["question_groups"]:
            raise ValueError("dependence groups differ from source panel")
        if any(plan["model"][key] != source["model"][key] for key in
               ("name", "temperature", "thinking", "max_input_bytes", "prices")):
            raise ValueError("model parameters differ from source measurement")
        if source["model"]["output_limits"]["answer"] != plan["model"]["output_limits"]["answer"]:
            raise ValueError("answer output limit differs")
    elif (plan["source_plan_file"] is not None
          or (plan["schema"] == CONTEXT_SCHEMA and plan["source_checkout"] is not None)):
        raise ValueError("synthetic probe cannot claim a real source plan or checkout")
    return cases


def preflight(plan):
    fields = {"schema", "purpose", "question_use", "source_plan_file", "cases_file",
              "references_file", "output_dir", "arms", "repeats", "schedule_seed", "analysis",
              "model", "hard_cny", "max_calls", "runtime_source_hashes",
              "helper_source_hashes", "credential_source"}
    if not isinstance(plan, dict):
        raise ValueError("explicit final-answer diagnostic contract required")
    contract = _contract(plan)
    if plan["schema"] == CONTEXT_SCHEMA:
        fields |= {"source_arm", "source_checkout", "advance_criteria"}
    if (set(plan) != fields or plan["question_use"] not in {"used_diagnostic", "synthetic"}):
        raise ValueError("explicit final-answer diagnostic contract required")
    _sources(plan)
    _file_contract(plan["references_file"])  # Metadata only until complete generation.
    if plan["arms"] != [{"name": name} for name in _arms(plan)]:
        raise ValueError("exact protocol-specific two-arm contrast required")
    if (type(plan["repeats"]) is not int or not 1 <= plan["repeats"] <= 8
            or type(plan["schedule_seed"]) is not int or plan["schedule_seed"] < 0):
        raise ValueError("invalid repeat/schedule specification")
    model = plan["model"]
    if (set(model) != {"name", "temperature", "thinking", "max_input_bytes", "output_limits", "prices"}
            or model["name"] != "deepseek-flash" or model["temperature"] != 0
            or model["thinking"] != "disabled" or type(model["max_input_bytes"]) is not int
            or not 1000 <= model["max_input_bytes"] <= 120000
            or model["output_limits"] != {"answer": 800}
            or set(model["prices"]) != {"input_hit", "input_miss", "output"}
            or any(type(v) not in (float, int) or not math.isfinite(v) or v < 0
                   for v in model["prices"].values())
            or model["prices"]["input_hit"] > model["prices"]["input_miss"]):
        raise ValueError("frozen model and valid price envelope required")
    cases = _snapshot(plan)
    validate_analysis(plan["analysis"], question_ids=[c["task"]["question_id"] for c in cases],
                      arm_names=_arms(plan))
    if (plan["analysis"]["schema"] != SINGLE_CONTRAST_SCHEMA
            or plan["analysis"]["primary_metric"] != "answer_f1"
            or plan["analysis"]["comparisons"] != [{"name": contract["contrast"],
                "baseline": "full_state", "candidate": contract["arms"][1]}]):
        raise ValueError("single predeclared F1 contrast required")
    credential = plan["credential_source"]
    if (set(credential) != {"kind", "path", "variable"} or credential["kind"] != "env_file"
            or credential["variable"] != "DEEPSEEK_API_KEY"
            or not isinstance(credential["path"], str) or not credential["path"]):
        raise ValueError("credential source metadata required")
    out = Path(plan["output_dir"]).resolve()
    if (Path(__file__).resolve().parents[1]/"runs").resolve() not in out.parents:
        raise ValueError("output must remain within existing project runs")
    shape = _RequestShape(model)
    sizes, fingerprints, contexts = {}, {}, {}
    bound = 0.0
    for case in cases:
        original = case["events"][-1]["request"]["payload"]
        if original["additional_guidance"] != "" or original["instructions"] != "":
            raise ValueError("side-channel prompt guidance is not part of this intervention")
        sizes[case["case_id"]] = {}
        fingerprints[case["case_id"]] = {}
        for arm in _arms(plan):
            replay_files(case, arm)
            payload = project_case_payload(case, arm)
            size = shape.request_size("answer", payload)
            if size > model["max_input_bytes"]:
                raise ValueError("frozen complete answer body exceeds limit")
            if arm == "quote_context":
                baseline = project_case_payload(case, "full_state")
                additions = [item for item in payload["evidence"] if "context" in item]
                contexts[case["case_id"]] = {
                    "original_citations": len(baseline["evidence"]),
                    "context_citations": len(additions),
                    "added_source_chars": sum(len(item["context"]["text"])-len(item["quote"]) for item in additions),
                    "final_payload_chars": len(json.dumps(payload, ensure_ascii=False)),
                    "baseline_payload_chars": len(json.dumps(baseline, ensure_ascii=False)),
                    "all_noncontext_fields_equal": True}
            sizes[case["case_id"]][arm] = size
            fingerprints[case["case_id"]][arm] = digest(shape.request_body("answer", payload))
            bound += plan["repeats"]*((size+1024)*model["prices"]["input_miss"]
                                     +800*model["prices"]["output"])/1e6
    calls = len(cases)*2*plan["repeats"]
    if (type(plan["max_calls"]) is not int or plan["max_calls"] != calls
            or type(plan["hard_cny"]) not in (int, float) or not math.isfinite(plan["hard_cny"])
            or plan["hard_cny"] < bound):
        raise ValueError("hard caps do not cover exact frozen answer envelope")
    return {"schema": plan["schema"]+"-preflight", "status": "ready", "plan_hash": digest(plan),
            "states": len(cases), "outcomes": calls, "max_new_calls": calls,
            "conservative_cny_upper_bound": bound, "hard_cny": plan["hard_cny"],
            "answer_body_bytes": sizes, "answer_body_hashes": fingerprints,
            "new_search_calls": 0, "new_read_model_calls": 0, "new_api_calls": 0,
            "reference_or_credential_access": False, "all_source_questions_retained": True,
            "scope": "used " + contract["source_arm"] + "-state mechanism diagnosis, not end-to-end or independent improvement",
            **({"context_delivery": contexts, "radius_chars": 256,
                "advance_criteria": deepcopy(CONTEXT_ADVANCE)} if plan["schema"] == CONTEXT_SCHEMA else {})}


def _bank(plan, case_id, arm, repeat):
    return "answer-probe/"+digest([digest(plan), case_id, arm, repeat])


def _model(plan, case, arm, repeat, ledger, transport):
    spec = plan["model"]
    return StructuredModel(Path(plan["output_dir"])/"requests", ledger, transport,
        bank=_bank(plan, case["case_id"], arm["name"], repeat), prices=spec["prices"],
        model=spec["name"], scope="run", max_input_bytes=spec["max_input_bytes"],
        limits=spec["output_limits"])


def _verify_observations(plan, case, arm, repeat, receipt, ledger, records):
    class NoDispatch:
        def send(self, *args):
            raise HostError("verification cannot dispatch")
    model = _model(plan, case, arm, repeat, ledger, NoDispatch())
    class CacheOnly:
        def complete(self, stage, payload):
            key = digest({"body": model.request_body(stage, payload), "bank": model.bank})
            if key not in records or read(model.directory/(key+".json")) != records[key]:
                raise HostError("missing or changed settled answer")
            return model.complete(stage, payload)
    router = AnswerReplayRouter(case, CacheOnly(), arm["name"])
    broker = HostBroker(case["task"], router, router, **_limits(case))
    for event in receipt["trace"]:
        broker(event["name"], event["request"])
    router.assert_complete()
    observed = {"read_presentations": broker.read_presentations,
                "final_observations": broker.final_observations}
    if broker.events != receipt["trace"] or observed != receipt["host_evidence_trace"] or model.calls != 0:
        raise ValueError("frozen observations differ from verified request/response replay")


def _validate_frozen(plan, frozen, *, cases=None):
    _sources(plan)
    cases = _snapshot(plan) if cases is None else cases
    bycase = {c["case_id"]: c for c in cases}
    out = Path(plan["output_dir"])
    if (not isinstance(frozen, dict)
            or set(frozen) != {"schema", "plan_hash", "cells", "ledger", "references_parsed"}
            or frozen["schema"] != plan["schema"]+"-generation" or frozen["plan_hash"] != digest(plan)
            or frozen["references_parsed"] is not False or read(out/"plan.json") != plan):
        raise ValueError("complete matching final-answer generation required")
    ledger, records = _accounting(out, plan)
    if ledger.summary() != frozen["ledger"]:
        raise ValueError("ledger changed since generation")
    expected = {(c, a, r) for c in bycase for a in _arms(plan) for r in range(plan["repeats"])}
    if not isinstance(frozen["cells"], list) or len(frozen["cells"]) != len(expected):
        raise ValueError("incomplete final-answer panel")
    archive = ProgramArchive(out/"archive")
    seen, result = set(), []
    for cell in frozen["cells"]:
        if not isinstance(cell, dict) or set(cell) != {"file", "sha256", "identity"}:
            raise ValueError("invalid frozen cell")
        identity = cell["identity"]
        if (not isinstance(identity, dict) or type(identity.get("repeat")) is not int
                or not isinstance(identity.get("case_id"), str) or not isinstance(identity.get("arm"), str)):
            raise ValueError("invalid cell identity")
        key = (identity["case_id"], identity["arm"], identity["repeat"])
        if key not in expected or key in seen:
            raise ValueError("foreign or duplicate cell")
        seen.add(key)
        case, arm = bycase[key[0]], {"name": key[1]}
        node = archive.load_node(identity["node_id"])
        if (identity != _identity(plan, case, arm, key[2], node)
                or archive.load_program(node["program_id"])["files"] != replay_files(case, key[1])
                or cell["file"] != _relative(*key)):
            raise ValueError("frozen source/identity/path differs")
        receipt = _saved_execution(_cell_path(out, cell["file"]), identity, sha256=cell["sha256"])
        if receipt is None:
            raise ValueError("missing generated cell")
        _verify_observations(plan, case, arm, key[2], receipt, ledger, records)
        result.append({"identity": identity, "payload": receipt})
    return result


def generate(plan, *, approved_plan_hash, transport=None, executor=execute):
    plan = deepcopy(plan)
    check = preflight(plan)
    if approved_plan_hash != check["plan_hash"]:
        raise ValueError("execution requires the exact approved plan")
    cases = _snapshot(plan)
    out = Path(plan["output_dir"])
    with run_lock(out):
        freeze(out/"plan.json", plan)
        ledger, _ = _accounting(out, plan)
        existing = read(out/"generation_freeze.json")
        if existing is not None:
            _validate_frozen(plan, existing, cases=cases)
            return existing
        archive = ProgramArchive(out/"archive")
        nodes = {}
        for index, (case, arm) in enumerate((c,a) for c in cases for a in plan["arms"]):
            nodes[(case["case_id"], arm["name"])] = recoverable_record(
                archive, replay_files(case, arm["name"]), {}, session_id="answer-probe", attempt=index)
        order = _order(plan, cases)
        freeze(out/"schedule.json", [{"case_id": c["case_id"], "arm": a["name"], "repeat": r}
                                    for c,a,r in order])
        cells = []
        try:
            for case, arm, repeat in order:
                _sources(plan)
                _verified_bytes(plan["cases_file"])
                node = nodes[(case["case_id"], arm["name"])]
                identity = _identity(plan, case, arm, repeat, node)
                relative = _relative(case["case_id"], arm["name"], repeat)
                target = _cell_path(out, relative)
                receipt = _saved_execution(target, identity)
                if receipt is None:
                    if transport is None:
                        transport = deepseek_transport(credential_from_plan(plan))
                    model = _model(plan, case, arm, repeat, ledger, transport)
                    router = AnswerReplayRouter(case, model, arm["name"])
                    receipt = executor(archive, node["node_id"], case["task"], router, router,
                                       target.parent, limits=_limits(case))
                    router.assert_complete()
                    _execution_facts(receipt, identity)
                    freeze(target, {"identity": identity, "payload": receipt, "payload_hash": digest(receipt)})
                cells.append({"file": relative, "sha256": file_hash(target), "identity": identity})
                save(out/"progress.json", {"status": "running", "completed": len(cells),
                                          "total": len(order), "ledger": ledger.summary()})
            _accounting(out, plan)
            frozen = {"schema": plan["schema"]+"-generation", "plan_hash": digest(plan), "cells": cells,
                      "ledger": ledger.summary(), "references_parsed": False}
            _validate_frozen(plan, frozen, cases=cases)
            freeze(out/"generation_freeze.json", frozen)
            save(out/"progress.json", {"status": "generation_complete", "completed": len(cells),
                                      "total": len(order), "ledger": ledger.summary()})
            return frozen
        except BaseException as error:
            save(out/"progress.json", {"status": "stopped", "reason_type": type(error).__name__,
                "completed": len(cells), "total": len(order), "ledger": ledger.summary()})
            raise


def grade(plan):
    plan = deepcopy(plan)
    check = preflight(plan)
    out = Path(plan["output_dir"])
    with run_lock(out):
        frozen = read(out/"generation_freeze.json")
        cases = _snapshot(plan)
        cells = _validate_frozen(plan, frozen, cases=cases)
        # Answer references are parsed only beyond the full-generation gate.
        references = json.loads(_verified_bytes(plan["references_file"]))
        byqid = {c["task"]["question_id"]: c["task"] for c in cases}
        if not isinstance(references, dict) or set(references) != set(byqid):
            raise ValueError("references must cover the complete fixed panel")
        rows = []
        for item in cells:
            identity, receipt = item["identity"], item["payload"]
            qid = identity["question_id"]
            reference = references[qid]
            if (reference.get("question_id") != qid or reference.get("dataset") != "musique"
                    or reference.get("answerable") is not True or reference.get("reference_available") is not True
                    or not isinstance(reference.get("answers"), list) or not reference["answers"]
                    or any(not isinstance(x, str) or not x.strip() for x in reference["answers"])):
                raise ValueError("complete answerable MuSiQue reference required")
            eligible = receipt["execution_ok"] and receipt["answer_origin_valid"] and receipt["answer_usable"]
            rows.append({"question_id": qid, "case_id": identity["case_id"], "arm": identity["arm"],
                "repeat": identity["repeat"], "answer_sha256": digest(receipt["answer"]),
                "program_eligible": eligible,
                "answer_origin_valid": receipt["answer_origin_valid"], "source_valid": receipt["citation_source_valid"],
                "abstained": receipt["answer"].strip().casefold() in
                    {"insufficient information", "unknown", "i don't know"},
                "metrics": score_task(receipt, reference, task=byqid[qid]) if eligible else None,
                "model_errors": receipt["model_errors"]})
        valid = all(row["program_eligible"] for row in rows)
        paired = (paired_analysis(rows, plan["analysis"], arm_names=_arms(plan), expected_repeats=plan["repeats"])
                  if valid else {"status": "protocol_invalid", "quality_comparison_valid": False,
                                 "qualified": None, "successful_subset_analysis": False,
                                 "ineligible_cells": [
                                     {k: row[k] for k in ("case_id", "arm", "repeat")}
                                     for row in rows if not row["program_eligible"]]})
        summary = {}
        for arm in _arms(plan):
            group = [r for r in rows if r["arm"] == arm]
            summary[arm] = {"outcomes": len(group), "abstentions": sum(r["abstained"] for r in group),
                "nonabstaining_f1_zero": sum(not r["abstained"] and r["metrics"]["answer_f1"] == 0 for r in group) if valid else None,
                "nonabstaining_em_wrong": sum(not r["abstained"] and r["metrics"]["answer_em"] == 0 for r in group) if valid else None,
                "mean_em": sum(r["metrics"]["answer_em"] for r in group)/len(group) if valid else None,
                "mean_f1": sum(r["metrics"]["answer_f1"] for r in group)/len(group) if valid else None,
                "source_valid": sum(r["source_valid"] for r in group),
                "repeat_string_consistent_questions": (
                    sum(len({r["answer_sha256"] for r in group if r["question_id"] == q}) == 1
                        for q in byqid) if plan["repeats"] > 1 else None)}
        ledger, records = _accounting(out, plan)
        bank_to_arm = {_bank(plan, c["case_id"], a["name"], r): a["name"]
                       for c, a, r in _order(plan, cases)}
        reservations = {event["id"]: event for event in ledger.events if event["event"] == "reserve"}
        costs = {arm: {"calls": 0, "input": 0, "output": 0, "cny": 0.0} for arm in _arms(plan)}
        for event in ledger.events:
            if event["event"] != "settle":
                continue
            arm = bank_to_arm[reservations[event["id"]]["metadata"]["bank"]]
            for key, value in event["actual"].items():
                costs[arm][key] = costs[arm].get(key, 0) + value
        observed_models = sorted({str(record["response"].get("model", "unavailable"))
                                  for record in records.values()})
        fingerprints = sorted({str(record["response"].get("system_fingerprint", "unavailable"))
                               for record in records.values()})
        report = {"schema": plan["schema"]+"-report", "plan_hash": digest(plan),
            "status": "local_diagnostic_complete" if paired["quality_comparison_valid"] else "protocol_invalid",
            "summary": summary, "rows": rows, "paired_analysis": paired, "ledger": frozen["ledger"],
            "per_arm_actual_cost": costs, "returned_models": observed_models,
            "provider_fingerprints": fingerprints,
            "references_parsed_after_complete_generation": True, "new_search_calls": 0,
            "new_reader_calls": 0, "independent_quality_evidence": False,
            "claim": "Fixed historical " + _contract(plan)["source_arm"] + "-state final-input diagnosis; not RSI or end-to-end improvement."}
        if plan["schema"] == CONTEXT_SCHEMA:
            baseline, candidate = summary["full_state"], summary["quote_context"]
            checks = {"all_cells_eligible": valid,
                      "f1_increased": valid and candidate["mean_f1"] > baseline["mean_f1"],
                      "em_not_decreased": valid and candidate["mean_em"] >= baseline["mean_em"],
                      "nonabstaining_f1_zero_not_increased": valid and candidate["nonabstaining_f1_zero"] <= baseline["nonabstaining_f1_zero"]}
            report["advance_decision"] = {"criteria": deepcopy(CONTEXT_ADVANCE), "checks": checks,
                "passed": all(checks.values()), "independent_confirmation_required": True,
                "automatic_deployment": False, "rsi_benefit_verified": False}
            report["source_arm"] = "loop"
            report["context_delivery"] = check["context_delivery"]
            for arm in _arms(plan):
                group = [row for row in rows if row["arm"] == arm]
                summary[arm]["abstaining_source_invalid"] = sum(row["abstained"] and not row["source_valid"] for row in group)
                summary[arm]["nonabstaining_source_invalid"] = sum(not row["abstained"] and not row["source_valid"] for row in group)
        freeze(out/"report.json", report)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "generate", "grade"])
    parser.add_argument("--plan", required=True)
    parser.add_argument("--approved-plan-hash")
    args = parser.parse_args(argv)
    plan = read(args.plan)
    if args.command == "preflight":
        result = preflight(plan)
    elif args.command == "generate":
        result = generate(plan, approved_plan_hash=args.approved_plan_hash)
    else:
        result = grade(plan)
    public = result if args.command == "preflight" else {
        "status": result.get("status", "generation_complete"), "plan_hash": result["plan_hash"], "ledger": result["ledger"]}
    print(json.dumps(public, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
