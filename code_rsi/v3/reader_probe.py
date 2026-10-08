"""Fixed last-read -> answer intervention, using the existing WSL RAG runtime.

Historical prefix responses are replayed; only the final read and answer are new
model requests. This is a local mechanism probe, not a new full search experiment.
No credential/reference is read on import or during preflight.
"""
from __future__ import annotations
from copy import deepcopy
import argparse
import hashlib
import json
import math
from pathlib import Path
import random

from ..archive import ProgramArchive
from ..budget import Ledger, digest, save
from .calibration import (run_lock, _verified_bytes, _saved_execution, _execution_facts,
                          _cell_path, credential_from_plan, file_hash)
from .evolution import freeze, read, recoverable_record, _runtime_source_hashes
from .execution import execute, validate_answer_origin, HostBroker, HostError
from .infrastructure import StructuredModel, deepseek_transport
from .reader_replay import validate_case, replay_files, ReplayRouter
from .request_recovery import check_request_recovery, check_request_accounting
from .task_metrics import score_task

SCHEMA = "rag-rsi-reader-probe-1"
GUIDANCE = (
    "Prioritize the relation explicitly requested by the original question and "
    "the currently unresolved gaps. Re-examine the current source text even when "
    "known_claims says that information is missing; known_claims are model "
    "assessments, not established facts. Extract factual propositions only when "
    "the cited passages support them. A failure to find information in the "
    "supplied windows belongs in gaps, scoped to those windows. Do not use an "
    "unrelated passage or metadata to justify a claim that information does not "
    "exist. Distinguish evidence identifying a candidate answer from evidence "
    "verifying the other question constraints. Preserve genuine conflicting "
    "evidence. Do not declare the question fully supported merely because a "
    "candidate answer or one matching constraint was found."
)


class _RequestShape(StructuredModel):
    """Use the provider's existing body builder without constructing I/O state."""
    def __init__(self, spec):
        self.model = spec["name"]
        self.limits = {"plan": 1200, "read": 2200, "answer": 800, "develop": 18000}
        self.limits.update(spec["output_limits"])
        self.max_input_bytes = spec["max_input_bytes"]


def _file_contract(item):
    if (not isinstance(item, dict) or set(item) != {"path", "sha256"}
            or not isinstance(item["path"], str) or not item["path"]
            or not isinstance(item["sha256"], str) or len(item["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in item["sha256"])):
        raise ValueError("frozen path and SHA256 required")


def _snapshot(plan):
    _file_contract(plan["cases_file"])
    packet = json.loads(_verified_bytes(plan["cases_file"]))
    if (not isinstance(packet, dict) or set(packet) != {"schema", "cases"}
            or packet["schema"] != "rag-rsi-reader-replay-cases-1"
            or not isinstance(packet["cases"], list) or not 1 <= len(packet["cases"]) <= 16):
        raise ValueError("bounded replay case packet required")
    for case in packet["cases"]:
        validate_case(case)
    cases = packet["cases"]
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("duplicate frozen state")
    byqid = {}
    for case in cases:
        qid = case["task"]["question_id"]
        if qid in byqid and byqid[qid] != case["task"]:
            raise ValueError("same question identity has differing public inputs")
        byqid[qid] = case["task"]
    return cases


def preflight(plan):
    plan = deepcopy(plan)
    fields = {"schema", "purpose", "question_use", "cases_file", "references_file",
              "annotations_file", "output_dir", "arms", "repeats", "schedule_seed",
              "model", "hard_cny", "max_calls", "runtime_source_hashes", "credential_source"}
    if (set(plan) != fields or plan.get("schema") != SCHEMA
            or plan.get("purpose") != "fixed_last_read_to_answer"
            or plan.get("question_use") not in {"used_diagnostic", "synthetic"}):
        raise ValueError("explicit local reader diagnostic contract required")
    cases = _snapshot(plan)
    # Deliberately validate metadata only. These files are opened after generation.
    for key in ("references_file", "annotations_file"):
        _file_contract(plan[key])
    if plan["arms"] != [{"name": "A", "guidance": None}, {"name": "B", "guidance": GUIDANCE}]:
        raise ValueError("only the frozen last-read guidance intervention is allowed")
    if (type(plan["repeats"]) is not int or not 1 <= plan["repeats"] <= 8
            or type(plan["schedule_seed"]) is not int or plan["schedule_seed"] < 0):
        raise ValueError("invalid repeat or schedule contract")
    model = plan["model"]
    if (set(model) != {"name", "temperature", "thinking", "max_input_bytes", "output_limits", "prices"}
            or model["name"] != "deepseek-flash" or model["temperature"] != 0
            or model["thinking"] != "disabled" or type(model["max_input_bytes"]) is not int
            or not 1000 <= model["max_input_bytes"] <= 120000
            or model["output_limits"] != {"read": 2200, "answer": 800}
            or set(model["prices"]) != {"input_miss", "input_hit", "output"}
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in model["prices"].values())
            or model["prices"]["input_hit"] > model["prices"]["input_miss"]):
        raise ValueError("frozen model, response limits and valid prices required")
    units = len(cases)*2*plan["repeats"]
    calls = units*2
    bound = (calls*(model["max_input_bytes"]+1024)*model["prices"]["input_miss"]
             + units*3000*model["prices"]["output"])/1e6
    if (type(plan["max_calls"]) is not int or plan["max_calls"] != calls
            or type(plan["hard_cny"]) not in (int, float) or not math.isfinite(plan["hard_cny"])
            or plan["hard_cny"] < bound):
        raise ValueError("budget cannot cover the frozen worst-case call envelope")
    if plan["runtime_source_hashes"] != _runtime_source_hashes():
        raise ValueError("runtime changed since reader plan freeze")
    credential = plan["credential_source"]
    if (set(credential) != {"kind", "path", "variable"} or credential["kind"] != "env_file"
            or credential["variable"] != "DEEPSEEK_API_KEY" or not isinstance(credential["path"], str)):
        raise ValueError("explicit credential source metadata required")
    out = Path(plan["output_dir"]).resolve()
    runs = Path(__file__).resolve().parents[2]/"runs"
    if runs.resolve() not in out.parents:
        raise ValueError("reader output must remain under project runs")
    # Build trusted source and exact complete read bodies without any I/O model.
    shapes = _RequestShape(model)
    read_bytes = {}
    for case in cases:
        read_bytes[case["case_id"]] = {}
        for arm in plan["arms"]:
            replay_files(case, guidance=arm["guidance"])
            payload = deepcopy(case["events"][-2]["request"]["payload"])
            if arm["guidance"] is not None:
                payload["additional_guidance"] = arm["guidance"]
            size = shapes.request_size("read", payload)
            if size > model["max_input_bytes"]:
                raise ValueError("frozen final read exceeds complete provider input limit")
            read_bytes[case["case_id"]][arm["name"]] = size
    return {"schema": SCHEMA+"-preflight", "plan_hash": digest(plan), "status": "ready",
            "states": len(cases), "questions": len({c["task"]["question_id"] for c in cases}),
            "outcomes": units, "max_new_calls": calls, "conservative_cny_upper_bound": bound,
            "hard_cny": plan["hard_cny"], "read_request_bytes": read_bytes, "old_responses_only_in_prefix": True,
            "independent_bank_per_arm_state_repeat": True, "new_search_calls": 0,
            "reference_or_credential_access": False, "new_api_calls": 0,
            "scope": "fixed-state local mechanism diagnosis; not full-loop or independent quality evidence"}


def _order(plan, cases):
    rng = random.Random(plan["schedule_seed"])
    order = []
    for repeat in range(plan["repeats"]):
        group = list(cases); rng.shuffle(group)
        for case in group:
            arms = list(plan["arms"]); rng.shuffle(arms)
            order.extend((case, arm, repeat) for arm in arms)
    return order


def _relative(case_id, arm, repeat):
    return "cells/"+digest([case_id, arm, repeat])[:24]+"/generation.json"


def _limits(case):
    return {"max_models": case["config"].get("max_model_calls", 7),
            "max_searches": max(1, sum(e["name"] == "search" for e in case["events"])),
            "max_reads": max(1, sum(e["name"] == "read" for e in case["events"]))}


def _identity(plan, case, arm, repeat, node):
    return {"plan_hash": digest(plan), "case_id": case["case_id"], "arm": arm["name"],
            "repeat": repeat, "case_hash": digest(case), "question_id": case["task"]["question_id"],
            "node_id": node["node_id"], "program_id": node["program_id"]}


def _accounting(out, plan):
    records = check_request_recovery(out, status_filename="progress.json")
    ledger = Ledger(out/"ledger.jsonl", {"run": {"cny": plan["hard_cny"], "calls": plan["max_calls"]}})
    check_request_accounting(records, ledger)
    return ledger, records


def _bank(plan, case_id, arm, repeat):
    return "reader-probe/"+digest([digest(plan),case_id,arm,repeat])


def _verify_observations(plan, case, arm, repeat, receipt, ledger, records):
    """Rebuild host observations from bound responses, not asserted quote flags."""
    out = Path(plan["output_dir"]); spec = plan["model"]
    class NoDispatch:
        def send(self, *args):
            raise HostError("frozen verification must not dispatch")
    cached = StructuredModel(out/"requests",ledger,NoDispatch(),
        bank=_bank(plan,case["case_id"],arm["name"],repeat),prices=spec["prices"],
        model=spec["name"],scope="run",max_input_bytes=spec["max_input_bytes"],limits=spec["output_limits"])
    class CacheOnly:
        def complete(self, stage, payload):
            key = digest({"body":cached.request_body(stage,payload),"bank":cached.bank})
            if key not in records:
                raise HostError("frozen suffix lacks its bound settled response")
            # Use the exact checked snapshot; never reserve or send on a missing key.
            path = cached.directory/(key+".json")
            if read(path) != records[key]:
                raise HostError("settled response changed during verification")
            return cached.complete(stage,payload)
    router = ReplayRouter(case,live_model=CacheOnly(),guidance=arm["guidance"])
    broker = HostBroker(case["task"],router.backend,router.model,**_limits(case))
    for event in receipt["trace"]:
        broker(event["name"],event["request"])
    router.assert_complete()
    expected = {"read_presentations":broker.read_presentations,"final_observations":broker.final_observations}
    if broker.events != receipt["trace"] or expected != receipt["host_evidence_trace"]:
        raise ValueError("reader observations differ from bound requests and responses")
    if cached.calls != 0:
        raise HostError("verification unexpectedly purchased a response")


def _validate_frozen(plan, frozen):
    preflight(plan)
    cases = {c["case_id"]: c for c in _snapshot(plan)}
    out = Path(plan["output_dir"])
    if (not isinstance(frozen, dict) or set(frozen) != {"schema", "plan_hash", "cells", "ledger", "references_parsed"}
            or frozen["schema"] != SCHEMA+"-generation" or frozen["plan_hash"] != digest(plan)
            or frozen["references_parsed"] is not False or read(out/"plan.json") != plan):
        raise ValueError("complete matching reader generation freeze required")
    ledger, records = _accounting(out, plan)
    if ledger.summary() != frozen["ledger"]:
        raise ValueError("reader ledger changed since freeze")
    expected = {(c, a["name"], r) for c in cases for a in plan["arms"] for r in range(plan["repeats"])}
    if not isinstance(frozen["cells"], list) or len(frozen["cells"]) != len(expected):
        raise ValueError("incomplete reader panel")
    archive = ProgramArchive(out/"archive")
    arms = {a["name"]: a for a in plan["arms"]}
    result = []; seen = set()
    for cell in frozen["cells"]:
        if not isinstance(cell, dict) or set(cell) != {"file", "sha256", "identity"}:
            raise ValueError("invalid frozen reader cell")
        identity = cell["identity"]
        if (not isinstance(identity,dict) or type(identity.get("repeat")) is not int
                or not isinstance(identity.get("case_id"),str) or not isinstance(identity.get("arm"),str)):
            raise ValueError("invalid reader identity types")
        key = (identity.get("case_id"), identity.get("arm"), identity.get("repeat"))
        if key not in expected or key in seen:
            raise ValueError("duplicate or foreign reader cell")
        seen.add(key)
        case, arm = cases[key[0]], arms[key[1]]
        node = archive.load_node(identity["node_id"])
        if identity != _identity(plan, case, arm, key[2], node):
            raise ValueError("reader identity differs")
        if archive.load_program(node["program_id"])["files"] != replay_files(case, guidance=arm["guidance"]):
            raise ValueError("reader archived program differs")
        relative = _relative(*key)
        if cell["file"] != relative:
            raise ValueError("reader cell path differs")
        receipt = _saved_execution(_cell_path(out, relative), identity, sha256=cell["sha256"])
        if receipt is None:
            raise ValueError("missing reader generation")
        _verify_observations(plan,case,arm,key[2],receipt,ledger,records)
        result.append({"identity": identity, "payload": receipt})
    return result


def generate(plan, *, approved_plan_hash, transport=None, executor=execute):
    plan = deepcopy(plan); check = preflight(plan)
    if approved_plan_hash != check["plan_hash"]:
        raise ValueError("reader execution requires exact reviewed plan hash")
    cases = _snapshot(plan); out = Path(plan["output_dir"])
    with run_lock(out):
        freeze(out/"plan.json", plan)
        ledger, _ = _accounting(out, plan)
        existing = read(out/"generation_freeze.json")
        if existing is not None:
            _validate_frozen(plan, existing)
            return existing
        archive = ProgramArchive(out/"archive"); nodes = {}
        for index, (case, arm) in enumerate((c,a) for c in cases for a in plan["arms"]):
            nodes[(case["case_id"],arm["name"])] = recoverable_record(
                archive, replay_files(case,guidance=arm["guidance"]), {}, session_id="reader-probe", attempt=index)
        order = _order(plan, cases)
        freeze(out/"schedule.json", [{"case_id":c["case_id"],"arm":a["name"],"repeat":r} for c,a,r in order])
        cells = []
        try:
            for case, arm, repeat in order:
                if _runtime_source_hashes() != plan["runtime_source_hashes"]:
                    raise ValueError("runtime changed during reader generation")
                node = nodes[(case["case_id"],arm["name"])]
                identity = _identity(plan,case,arm,repeat,node)
                relative = _relative(case["case_id"],arm["name"],repeat)
                target = _cell_path(out,relative)
                receipt = _saved_execution(target,identity)
                if receipt is None:
                    # Lazy credential access only after plan/recovery/cell checks.
                    if transport is None:
                        transport = deepseek_transport(credential_from_plan(plan))
                    spec = plan["model"]
                    bank = _bank(plan,case["case_id"],arm["name"],repeat)
                    model = StructuredModel(out/"requests",ledger,transport,bank=bank,prices=spec["prices"],
                        model=spec["name"],scope="run",max_input_bytes=spec["max_input_bytes"],limits=spec["output_limits"])
                    router = ReplayRouter(case,live_model=model,guidance=arm["guidance"])
                    receipt = executor(archive,node["node_id"],case["task"],router.backend,router.model,
                                       target.parent,limits=_limits(case))
                    router.assert_complete()
                    _execution_facts(receipt,identity)
                    freeze(target,{"identity":identity,"payload":receipt,"payload_hash":digest(receipt)})
                cells.append({"file":relative,"sha256":file_hash(target),"identity":identity})
                save(out/"progress.json",{"status":"running","completed":len(cells),"total":len(order),"ledger":ledger.summary()})
            records = check_request_recovery(out,status_filename="progress.json")
            check_request_accounting(records,ledger)
            frozen = {"schema":SCHEMA+"-generation","plan_hash":digest(plan),"cells":cells,
                      "ledger":ledger.summary(),"references_parsed":False}
            # Validate every receipt/program/account before admitting private references.
            _validate_frozen(plan,frozen)
            freeze(out/"generation_freeze.json",frozen)
            save(out/"progress.json",{"status":"generation_complete","completed":len(cells),"total":len(order),"ledger":ledger.summary()})
            return frozen
        except Exception as error:
            save(out/"progress.json",{"status":"stopped","reason_type":type(error).__name__,"completed":len(cells),"total":len(order),"ledger":ledger.summary()})
            raise


def _covers(quote, anchor):
    return (str(quote.get("docid")) == str(anchor["docid"])
            and type(quote.get("start")) is int and type(quote.get("end")) is int
            and quote["start"] <= anchor["start"] < anchor["end"] <= quote["end"])


def _mechanics(case, receipt, anchors):
    # Anchors are evaluator-only offsets/hash, never sent to the model or wrapper.
    reads = receipt["host_evidence_trace"]["read_presentations"]
    finals = receipt["host_evidence_trace"]["final_observations"]
    sources = [e["request"]["payload"]["sources"] for e in case["events"]
               if e["name"] == "complete" and e["request"]["stage"] == "read"][-1]
    for anchor in anchors:
        if (set(anchor) != {"docid","start","end","quote_sha256"}
                or type(anchor["start"]) is not int or type(anchor["end"]) is not int
                or not 0 <= anchor["start"] < anchor["end"]):
            raise ValueError("invalid diagnostic anchor")
        containing = [s for s in sources if str(s["docid"]) == str(anchor["docid"])
                      and s["start"] <= anchor["start"] < anchor["end"] <= s["end"]]
        if not containing or any(hashlib.sha256(s["text"][anchor["start"]-s["start"]:anchor["end"]-s["start"]].encode()).hexdigest() != anchor["quote_sha256"] for s in containing):
            raise ValueError("anchor is not bound to a complete frozen presented span")
    last_event = [i for i,e in enumerate(receipt["trace"]) if e["name"] == "complete" and e["request"]["stage"] == "read"][-1]
    current = [x for x in reads if x["event_index"] == last_event]
    quotes = current[0]["verified_quotes"] if current else []
    prefix = [q for x in reads if x["event_index"] < last_event for q in x["verified_quotes"]]
    presented = list(finals[-1]["evidence"].values()) if finals else []
    return {"anchor_count":len(anchors),"last_read_observed":bool(current),
            "anchor_in_prefix":[any(_covers(q,a) for q in prefix) for a in anchors],
            "anchor_in_new_read":[any(_covers(q,a) for q in quotes) if current else None for a in anchors],
            "anchor_in_answer_input":[any(_covers(q,a) for q in presented) for a in anchors],
            "new_verified_quote_count":len(quotes) if current else None,"semantic_support":"not_independently_adjudicated",
            "anchor_match_proves_all_constraints":False}


def grade(plan):
    plan = deepcopy(plan); out = Path(plan["output_dir"])
    with run_lock(out):
        frozen = read(out/"generation_freeze.json")
        cells = _validate_frozen(plan,frozen)
        cases = {c["case_id"]:c for c in _snapshot(plan)}
        # First parsing of references/anchors occurs beyond the full-generation gate.
        reference_record = json.loads(_verified_bytes(plan["references_file"]))
        references = reference_record.get("rows") if isinstance(reference_record,dict) else reference_record
        if not isinstance(references,list):
            raise ValueError("reference rows required")
        byid = {str(r["query_id"]):r for r in references}
        if len(byid) != len(references):
            raise ValueError("duplicate references")
        annotation = json.loads(_verified_bytes(plan["annotations_file"]))
        if (set(annotation) != {"schema","cases"} or annotation["schema"] != "rag-rsi-reader-anchors-1"
                or set(annotation["cases"]) != set(cases)):
            raise ValueError("annotations must cover the exact state panel")
        rows = []
        for item in cells:
            identity, receipt = item["identity"],item["payload"]
            case = cases[identity["case_id"]]; task = case["task"]; ref = byid[task["question_id"]]
            if ref["question"] != task["question"] or not isinstance(ref["reference_answer"],str) or not ref["reference_answer"].strip():
                raise ValueError("reference differs from frozen public question")
            private = {"question_id":task["question_id"],"dataset":task["dataset"],"answers":[ref["reference_answer"]],"reference_available":True,"official_metric":"llm_judge"}
            eligible = receipt["execution_ok"] and receipt["answer_origin_valid"] and receipt["answer_usable"]
            rows.append({"case_id":case["case_id"],"question_id":task["question_id"],"arm":identity["arm"],"repeat":identity["repeat"],
                "program_eligible":eligible,"metrics":score_task(receipt,private,task=task,allow_proxy_metrics=True) if eligible else None,
                "mechanics":_mechanics(case,receipt,annotation["cases"][case["case_id"]]),
                "citation_status":receipt["citation_status"],"answer_origin_valid":receipt["answer_origin_valid"],
                "abstained":receipt["answer"].strip().casefold() in {"insufficient information","unknown","i don't know"},
                "observed_failure_classes":receipt["failure_classes"],"model_errors":receipt["model_errors"]})
        records = check_request_recovery(out,status_filename="progress.json")
        bank_to_arm = {"reader-probe/"+digest([digest(plan),c["case_id"],a["name"],r]):a["name"]
                       for c,a,r in _order(plan,list(cases.values()))}
        ledger,_ = _accounting(out,plan)
        reservations = {e["id"]:e for e in ledger.events if e["event"] == "reserve"}
        cost = {a["name"]:{"calls":0,"input":0,"output":0,"cny":0} for a in plan["arms"]}
        for event in ledger.events:
            if event["event"] != "settle":continue
            arm = bank_to_arm[reservations[event["id"]]["metadata"]["bank"]]
            for key,value in event["actual"].items():
                cost[arm][key] = cost[arm].get(key,0)+value
        hit_counts = {a["name"]:[] for a in plan["arms"]}
        for record in records.values():
            reservation = reservations[record["reservation"]]
            arm = bank_to_arm[reservation["metadata"]["bank"]]
            usage = record["response"].get("usage",{})
            value = usage.get("prompt_cache_hit_tokens")
            hit_counts[arm].append(value if type(value) is int and value >= 0 else None)
        for arm, hits in hit_counts.items():
            cost[arm]["input_hit"] = sum(hits) if all(x is not None for x in hits) else None
            cost[arm]["input_hit_status"] = "provider_reported" if all(x is not None for x in hits) else "unavailable_for_some_calls"
        report = {"schema":SCHEMA+"-report","plan_hash":digest(plan),"states":len(cases),
                  "questions":len({c["task"]["question_id"] for c in cases.values()}),"outcomes":len(rows),
                  "all_states_retained":True,"rows":rows,"ledger":frozen["ledger"],"per_arm_actual_cost":cost,
                  "status":"local_diagnostic_complete" if all(x["program_eligible"] for x in rows) else "protocol_invalid",
                  "official_browsecomp_score":False,"independent_quality_evidence":False,"new_search_calls":0,
                  "references_parsed_after_complete_generation":True,
                  "claim":"Fixed-state suffix mechanism probe. No full-search quality or RSI evolution claim."}
        freeze(out/"report.json",report)
        return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["preflight","generate","grade"])
    parser.add_argument("--plan",required=True)
    parser.add_argument("--approved-plan-hash")
    args=parser.parse_args(argv);plan=read(args.plan)
    if args.command == "preflight": result=preflight(plan)
    elif args.command == "generate": result=generate(plan,approved_plan_hash=args.approved_plan_hash)
    else: result=grade(plan)
    # Never print raw private replay cells or reference-bearing outputs.
    public = result if args.command == "preflight" else {"status":result.get("status","generation_complete"),"plan_hash":result["plan_hash"],"ledger":result["ledger"]}
    print(json.dumps(public,ensure_ascii=False,indent=2))
    return result


if __name__ == "__main__": main()
