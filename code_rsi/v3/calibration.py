"""Frozen RAG workflow calibration; generation finishes before references are parsed.

The CLI preflight is free of API/credential access. Execution requires the exact
reviewed plan hash. It shares the real program archive, WSL, ledger and models
with evolution; no experiment-specific provider or runner shell is introduced.
"""
from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
from ..archive import ProgramArchive
from ..budget import Ledger, digest, save
from .datasets import validate_task_collection
from .diagnostics import diagnose_execution
from .evolution import freeze, read, recoverable_record, _runtime_source_hashes
from .execution import execute, root_files, validate_sources, EXECUTION_SCHEMA, validate_answer_origin
from .infrastructure import BrowseCompCorpus, StructuredModel, deepseek_transport
from .rag import RagEngine, DEFAULTS
from .paired_analysis import validate_analysis, paired_analysis
from .request_recovery import check_request_recovery, check_request_accounting


SCHEMA="rag-rsi-v3-calibration-1"
THREE_ARM_SCHEMA="rag-rsi-v3-calibration-2"
REQUEST_COUPLING="shared_exact_request_per_question_repeat"


def file_hash(path):
    hasher=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            hasher.update(block)
    return hasher.hexdigest()


def _verified_file(item):
    if not isinstance(item,dict) or set(item)!={"path","sha256"}:
        raise ValueError("file path and SHA256 required")
    path=Path(item["path"]).resolve(strict=True)
    if file_hash(path)!=item["sha256"]:
        raise ValueError("frozen file changed: "+path.name)
    return path


def _verified_bytes(item):
    """Hash and parse callers' data from one immutable in-memory byte snapshot."""
    if not isinstance(item,dict) or set(item)!={"path","sha256"}:
        raise ValueError("file path and SHA256 required")
    raw=Path(item["path"]).resolve(strict=True).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=item["sha256"]:
        raise ValueError("frozen file changed")
    return raw


def _task_snapshot(plan):
    tasks=json.loads(_verified_bytes(plan["tasks_file"]))
    validate_task_collection(tasks)
    if not tasks or any(t["task_type"]!="qa" for t in tasks):
        raise ValueError("nonempty QA task panel required")
    if sorted(t["question_id"] for t in tasks)!=sorted(plan["question_ids"]):
        raise ValueError("declared question scope differs")
    for task in tasks:
        if task["dataset"]!="browsecomp-plus" or task["corpus_ref"]!=plan["corpus_ref"]:
            raise ValueError("this initial live calibration binds the fixed BrowseComp corpus")
    return tasks


class _PreflightServices:
    """Callable dependency stubs; configuration validation must never use I/O."""
    def search(self, query, limit=5):
        raise AssertionError("preflight must not search")

    def complete(self, stage, payload):
        raise AssertionError("preflight must not call a model")


def load_plan(path):
    plan=read(path)
    if not isinstance(plan,dict) or plan.get("schema") not in {SCHEMA,THREE_ARM_SCHEMA}:
        raise ValueError("unknown calibration plan")
    return plan


def preflight(plan, *, verify_corpus=True):
    """Public zero-call report; task values remain local to the validated snapshot."""
    return _preflight(deepcopy(plan),verify_corpus=verify_corpus)[0]


def _preflight(plan, *, verify_corpus=True):
    three_arm=plan.get("schema")==THREE_ARM_SCHEMA
    if plan.get("schema") not in {SCHEMA,THREE_ARM_SCHEMA} or plan.get("purpose") != (
            "workflow_decomposition_calibration" if three_arm else "used_development_calibration"):
        raise ValueError("calibration purpose must be explicit")
    if three_arm:
        if plan.get("request_coupling")!=REQUEST_COUPLING:
            raise ValueError("three-arm exact-request coupling must be frozen")
        if plan.get("question_use") not in {"used_development","unused_declared","synthetic"}:
            raise ValueError("question use must be declared; unused is not independently verified")
        if type(plan.get("schedule_seed")) is not int or plan["schedule_seed"]<0:
            raise ValueError("nonnegative integer schedule seed required")
    tasks=_task_snapshot(plan)
    # Hashing freezes bytes; no reference values are parsed before generation.
    _verified_file(plan["references_file"])
    if verify_corpus:
        _verified_file(plan["corpus"])
    arms=plan["arms"]
    arm_count=3 if three_arm else 2
    if (not isinstance(arms,list) or len(arms)!=arm_count
            or any(not isinstance(a,dict) or not isinstance(a.get("name"),str)
                   or not a["name"] or not isinstance(a.get("config"),dict) for a in arms)
            or len({a["name"] for a in arms})!=arm_count):
        raise ValueError("exactly "+str(arm_count)+" named frozen arms required")
    modes=[]; calls=0; output=0
    model=plan["model"]
    if model["name"]!="deepseek-flash" or model.get("thinking")!="disabled" or model.get("temperature")!=0:
        raise ValueError("explicit frozen DeepSeek generation settings required")
    if type(model["max_input_bytes"]) is not int or not 1000<=model["max_input_bytes"]<=120000:
        raise ValueError("input byte bound outside profile")
    prices=model["prices"]
    if set(prices)!={"input_hit","input_miss","output"} or any(type(x) not in (float,int) or not math.isfinite(x) or x<0 for x in prices.values()):
        raise ValueError("invalid CNY price envelope")
    if prices["input_hit"]>prices["input_miss"]:
        raise ValueError("cache price cannot exceed reservation price")
    for stage in ("plan","read","answer"):
        cap=model["output_limits"][stage]
        if type(cap) is not int or not 1<=cap<=32768:
            raise ValueError("output limit required")
    limits=plan.get("limits")
    if (not isinstance(limits,dict) or set(limits)!={"max_models","max_searches","max_reads"}
            or any(type(v) is not int or not 1<=v<=1000000 for v in limits.values())):
        raise ValueError("positive integer host model/search/read limits required")
    repeats=plan["repeats"]
    if type(repeats) is not int or not 1<=repeats<=5:
        raise ValueError("repeats outside calibration profile")
    for arm in arms:
        config=arm["config"]
        services=_PreflightServices()
        RagEngine(services,services,config=config)
        cfg={**DEFAULTS,**config}
        if cfg["search_limit"]>30:
            raise ValueError("search_limit exceeds host maximum of 30")
        modes.append(cfg["mode"])
        rounds=cfg["max_rounds"] if cfg["mode"]=="iterative" else 1
        plan_calls=int(cfg["mode"] in {"planned_single","iterative"})
        maximum=plan_calls+rounds+1
        if plan["limits"]["max_models"]<maximum or cfg["max_model_calls"]<maximum:
            raise ValueError("profile cannot reserve all declared rounds and final")
        searches=1 if cfg["mode"]=="single_pass" else rounds*min(cfg["max_queries_per_round"],24)
        if limits["max_searches"]<searches:
            raise ValueError("host search budget cannot cover declared workflow queries")
        calls += maximum
        output += plan_calls*model["output_limits"]["plan"]+rounds*model["output_limits"]["read"]+model["output_limits"]["answer"]
    expected_modes={"single_pass","planned_single","iterative"} if three_arm else {"single_pass","iterative"}
    if set(modes)!=expected_modes:
        raise ValueError("comparison requires exactly the declared workflow modes")
    if three_arm:
        if any("mode" not in arm["config"] for arm in arms):
            raise ValueError("three-arm modes must be explicit")
        mode_to_name={arm["config"]["mode"]:arm["name"] for arm in arms}
        if next({**DEFAULTS,**a["config"]}["max_rounds"] for a in arms if a["config"]["mode"]=="iterative")<2:
            raise ValueError("iteration comparison must permit at least two reads")
        analysis=validate_analysis(plan.get("analysis"),question_ids=plan["question_ids"],arm_names=[a["name"] for a in arms])
        expected_comparisons={
            ("planning",mode_to_name["single_pass"],mode_to_name["planned_single"]),
            ("iteration",mode_to_name["planned_single"],mode_to_name["iterative"])}
        if {(c["name"],c["baseline"],c["candidate"]) for c in analysis["comparisons"]}!=expected_comparisons:
            raise ValueError("freeze planning A-to-B and iteration B-to-C comparisons")
    # Same per-question envelope; realized expenditure is reported separately.
    base={k:v for k,v in arms[0]["config"].items() if k!="mode"}
    if any(base!={k:v for k,v in arm["config"].items() if k!="mode"} for arm in arms[1:]):
        raise ValueError("only the iterative workflow switch may differ between arms")
    calls*=len(tasks)*repeats; output*=len(tasks)*repeats
    if type(plan["max_calls"]) is not int or plan["max_calls"]!=calls:
        raise ValueError("call ceiling differs from derived exact structural bound")
    worst=((model["max_input_bytes"]+1024)*calls*prices["input_miss"]+output*prices["output"])/1e6
    if type(plan["hard_cny"]) not in (float,int) or not math.isfinite(plan["hard_cny"]) or not worst<=plan["hard_cny"]<=100:
        raise ValueError("hard cap cannot cover conservative complete-plan envelope")
    expected=_runtime_source_hashes()
    if plan.get("runtime_source_hashes")!=expected:
        raise ValueError("runtime changed since plan freeze")
    return {"status":"ready_for_explicit_execution","plan_hash":digest(plan),"question_count":len(tasks),
        "repeats":repeats,"answer_outcomes":len(tasks)*repeats*arm_count,"arm_count":arm_count,"max_calls":calls,
        "conservative_cny_upper_bound":round(worst,6),"hard_cny":plan["hard_cny"],
        "new_api_calls":0,"credentials_read":False,"references_parsed":False,
        "same_resource_ceiling":True,"same_realized_cost":False,
        "primary_claim":"workflow decomposition; declared newness is not independent confirmation" if three_arm else "development calibration; no independent benchmark generalization",
        "request_coupling":REQUEST_COUPLING,"prefix_coupling_includes_identical_final_requests":True,
        "question_use":plan.get("question_use","used_development"),
        "independent_groups":len(set(plan["analysis"]["question_groups"].values())) if three_arm else len(tasks)},tasks


@contextmanager
def run_lock(directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    with (directory/"run.lock").open("a+b") as stream:
        if stream.tell()==0: stream.write(b"0"); stream.flush()
        stream.seek(0)
        try:
            if os.name=="nt":
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("another process currently owns this run") from exc
        try: yield
        finally:
            stream.seek(0)
            if os.name=="nt": msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def credential_from_plan(plan):
    source=plan["credential_source"]
    if source.get("variable")!="DEEPSEEK_API_KEY" or source.get("kind")!="env_file":
        raise ValueError("explicit authorized credential source required")
    # Read only the named variable; never log file contents or another value.
    with Path(source["path"]).open(encoding="utf-8-sig") as stream:
        for line in stream:
            name,sep,value=line.strip().partition("=")
            if sep and name.strip()=="DEEPSEEK_API_KEY":
                value=value.strip()
                if value[:1] in {"'",'"'} and value[-1:]==value[:1]: value=value[1:-1]
                if value and "${" not in value: return value
                break
    raise ValueError("authorized credential variable missing or interpolated")


def _execution_facts(receipt, identity):
    """Accept only completed host execution receipts, never pending quality zeros."""
    allowed={"schema","node_id","program_id","question_id","resource_usage","model_errors",
             "trace","host_evidence_trace","candidate_reported","diagnostic_trust","answer",
             "answer_usable","execution_ok","citation_source_valid","citation_status",
             "host_citation_validation","citations","isolation_verified","failure_classes","error","role",
             "answer_origin_valid","answer_origin_status","host_answer_origin_validation"}
    if (not isinstance(receipt,dict) or receipt.get("schema")!=EXECUTION_SCHEMA
            or set(receipt)-allowed):
        raise ValueError("generation requires the current completed host execution schema")
    for name in ("node_id","program_id","question_id"):
        if receipt.get(name)!=identity[name]:
            raise ValueError("generation payload identity mismatch")
    if (not isinstance(receipt.get("answer"),str)
            or any(type(receipt.get(name)) is not bool for name in
                   ("answer_usable","execution_ok","citation_source_valid"))
            or not isinstance(receipt.get("resource_usage"),dict)
            or any(type(receipt["resource_usage"].get(name)) is not int
                   or receipt["resource_usage"][name]<0 for name in
                   ("model_calls","search_calls","read_calls"))
            or any(not isinstance(receipt.get(name),list)
                   or any(not isinstance(code,str) for code in receipt[name])
                   for name in ("failure_classes","model_errors"))
            or not isinstance(receipt.get("trace"),list)
            or not isinstance(receipt.get("host_evidence_trace"),dict)
            or any(not isinstance(receipt["host_evidence_trace"].get(name),list)
                   for name in ("read_presentations","final_observations"))):
        raise ValueError("generation payload lacks trusted execution facts")
    if receipt['answer_usable'] is not (receipt['execution_ok'] and bool(receipt['answer'].strip())):
        raise ValueError("generation answer usability differs from observed answer")
    validate_answer_origin(receipt)
    if diagnose_execution(receipt)["measurement_status"]!="observed":
        raise ValueError("unavailable execution cannot become a frozen quality outcome")
    return receipt


def _saved_execution(path, identity, *, sha256=None):
    try:
        raw=Path(path).read_bytes()
    except FileNotFoundError:
        if sha256 is not None:
            raise ValueError("frozen generation checkpoint missing")
        return None
    if sha256 is not None and hashlib.sha256(raw).hexdigest()!=sha256:
        raise ValueError("frozen generation cell changed")
    record=json.loads(raw)
    if (not isinstance(record,dict) or set(record)!={"identity","payload","payload_hash"}
            or record["identity"]!=identity or not isinstance(record["payload"],dict)
            or digest(record["payload"])!=record["payload_hash"]):
        raise ValueError("generation checkpoint integrity mismatch")
    return _execution_facts(record["payload"],identity)


def _cell_relative(qid, arm, repeat):
    return "cells/"+digest({"q":qid,"a":arm,"r":repeat})[:24]+"/generation.json"


def _cell_path(out, relative):
    # Callers first compare the relative string with the canonical cell name.
    # Resolving also prevents an existing cell-directory link escaping the run.
    out=Path(out).resolve()
    path=(out/relative).resolve()
    if not path.is_relative_to(out):
        raise ValueError("generation cell path escapes output directory")
    return path


def _validated_generation(plan, frozen, *, expected_backend=None):
    """Validate all completion evidence before resume or private-reference access.

    The returned receipts are the same snapshots whose hashes were checked;
    grading never reopens cell payloads after crossing the reference boundary.
    """
    if (not isinstance(frozen,dict)
            or frozen.get("schema")!="rag-rsi-v3-generation-freeze-1"
            or frozen.get("plan_hash")!=digest(plan)
            or frozen.get("references_parsed_by_runner") is not False
            or not isinstance(frozen.get("corpus_identity"),str)
            or not frozen["corpus_identity"]
            or not isinstance(frozen.get("ledger"),dict)
            or frozen["ledger"].get("pending")!=0):
        raise ValueError("complete matching generation freeze required before reference access")
    if _runtime_source_hashes()!=plan["runtime_source_hashes"]:
        raise ValueError("runtime or scorer changed since generation plan freeze")
    if expected_backend is not None and frozen["corpus_identity"]!=expected_backend:
        raise ValueError("frozen backend identity differs")
    expected={(qid,arm["name"],rep) for qid in plan["question_ids"]
              for arm in plan["arms"] for rep in range(plan["repeats"])}
    cells=frozen.get("cells")
    if not isinstance(cells,list) or len(cells)!=len(expected) or not expected:
        raise ValueError("generation freeze does not cover the complete question/arm/repeat panel")
    out=Path(plan["output_dir"])
    records=check_request_recovery(out,status_filename="progress.json")
    ledger=Ledger(out/"ledger.jsonl",{"run":{"cny":plan["hard_cny"],"calls":plan["max_calls"]}})
    check_request_accounting(records,ledger)
    if frozen["ledger"]!=ledger.summary():
        raise ValueError("frozen generation accounting differs from settled run")
    if read(out/"plan.json")!=plan:
        raise ValueError("stored generation plan differs")
    if not (out/"archive").is_dir():
        raise ValueError("frozen program archive missing")
    archive=ProgramArchive(out/"archive")
    arm_specs={arm["name"]:(index,root_files(arm["config"]))
               for index,arm in enumerate(plan["arms"])}
    seen=set(); seen_paths=set(); nodes={}; result=[]
    identity_fields={"plan_hash","node_id","program_id","question_id","arm","repeat","backend"}
    for cell in cells:
        if not isinstance(cell,dict) or set(cell)!={"file","sha256","identity"}:
            raise ValueError("invalid frozen generation cell")
        identity=cell["identity"]
        if (not isinstance(identity,dict) or set(identity)!=identity_fields
                or not isinstance(identity["question_id"],str) or not isinstance(identity["arm"],str)
                or type(identity["repeat"]) is not int
                or identity["plan_hash"]!=digest(plan)
                or identity["backend"]!=frozen["corpus_identity"]):
            raise ValueError("frozen generation identity mismatch")
        key=(identity["question_id"],identity["arm"],identity["repeat"])
        if key not in expected or key in seen:
            raise ValueError("duplicate or out-of-panel generation cell")
        seen.add(key)
        relative=_cell_relative(*key)
        if cell["file"]!=relative:
            raise ValueError("generation cell path differs from canonical identity")
        path=_cell_path(out,relative)
        if path in seen_paths:
            raise ValueError("generation cell paths alias each other")
        seen_paths.add(path)
        arm=identity["arm"]
        if arm not in nodes:
            node=archive.load_node(identity["node_id"])
            index,files=arm_specs[arm]
            program=archive.load_program(node["program_id"])
            if (node["session_id"]!="calibration-arms" or node["attempt"]!=index
                    or node["parent_node_id"] is not None or program["files"]!=files
                    or program["metadata"]!={name:{} for name in
                                             ("config","prompts","dependency_lock","index_builder")}):
                raise ValueError("frozen arm does not match its declared root program")
            nodes[arm]=node
        node=nodes[arm]
        if (identity["node_id"],identity["program_id"])!=(node["node_id"],node["program_id"]):
            raise ValueError("generation node/program differs from frozen arm")
        receipt=_saved_execution(path,identity,sha256=cell["sha256"])
        result.append({"file":relative,"identity":deepcopy(identity),"payload":receipt})
    if seen!=expected:
        raise ValueError("incomplete generation panel")
    return result


def generate(plan, *, approved_plan_hash, transport=None, backend=None, executor=execute):
    plan=deepcopy(plan)
    check,tasks=_preflight(plan,verify_corpus=backend is None)
    if approved_plan_hash!=check["plan_hash"]:
        raise ValueError("execution hash differs from reviewed plan")
    out=Path(plan["output_dir"])
    with run_lock(out):
        freeze(out/"plan.json",plan)
        records=check_request_recovery(out,status_filename="progress.json")
        ledger=Ledger(out/"ledger.jsonl",{"run":{"cny":plan["hard_cny"],"calls":plan["max_calls"]}})
        check_request_accounting(records,ledger)
        frozen=read(out/"generation_freeze.json")
        if frozen is not None:
            _validated_generation(plan,frozen,expected_backend=backend.identity if backend is not None else None)
            return frozen
        if transport is None: transport=deepseek_transport(credential_from_plan(plan))
        if backend is None: backend=BrowseCompCorpus(plan["corpus"]["path"],corpus_hash=plan["corpus"]["sha256"])
        archive=ProgramArchive(out/"archive")
        nodes={}
        for i,arm in enumerate(plan["arms"]):
            files=root_files(arm["config"]); validate_sources(files)
            nodes[arm["name"]]=recoverable_record(archive,files,{},session_id="calibration-arms",attempt=i)
        order=[]
        rng=random.Random(plan["schedule_seed"])
        for rep in range(plan["repeats"]):
            qorder=list(tasks); rng.shuffle(qorder)
            for task in qorder:
                aorder=list(plan["arms"]); rng.shuffle(aorder)
                order.extend((task,arm,rep) for arm in aorder)
        freeze(out/"schedule.json",[{"question_id":t["question_id"],"arm":a["name"],"repeat":r} for t,a,r in order])
        cells=[]
        try:
            for index,(task,arm,rep) in enumerate(order):
                if _runtime_source_hashes()!=plan["runtime_source_hashes"]:
                    raise ValueError("runtime changed during generation")
                node=nodes[arm["name"]]
                relative=_cell_relative(task["question_id"],arm["name"],rep)
                done=_cell_path(out,relative)
                cell=done.parent
                identity={"plan_hash":approved_plan_hash,"node_id":node["node_id"],"program_id":node["program_id"],
                    "question_id":task["question_id"],"arm":arm["name"],"repeat":rep,"backend":backend.identity}
                receipt=_saved_execution(done,identity)
                if receipt is None:
                    model=StructuredModel(out/"requests",ledger,transport,bank=f"calibration/{task['question_id']}/{rep}",
                       prices=plan["model"]["prices"],model=plan["model"]["name"],max_input_bytes=plan["model"]["max_input_bytes"],limits=plan["model"]["output_limits"])
                    receipt=executor(archive,node["node_id"],deepcopy(task),backend,model,cell,limits=deepcopy(plan["limits"]))
                    _execution_facts(receipt,identity)
                    save(done,{"identity":identity,"payload":receipt,"payload_hash":digest(receipt)})
                cells.append({"file":relative,"sha256":file_hash(done),"identity":identity})
                save(out/"progress.json",{"status":"running","pid":os.getpid(),"completed":index+1,"total":len(order),"ledger":ledger.summary()})
        except BaseException as exc:
            save(out/"progress.json",{"status":"stopped","completed":len(cells),"total":len(order),"reason_type":type(exc).__name__,"ledger":ledger.summary()})
            raise
        check_request_accounting(check_request_recovery(out,status_filename="progress.json"),ledger)
        # Corpus verification after live execution detects an externally changed index.
        if isinstance(backend,BrowseCompCorpus): _verified_file(plan["corpus"])
        frozen={"schema":"rag-rsi-v3-generation-freeze-1","plan_hash":approved_plan_hash,"cells":cells,
                "references_parsed_by_runner":False,"ledger":ledger.summary(),"corpus_identity":backend.identity}
        _validated_generation(plan,frozen,expected_backend=backend.identity)
        freeze(out/"generation_freeze.json",frozen)
        save(out/"progress.json",{"status":"generation_complete","completed":len(cells),"total":len(order),"ledger":ledger.summary()})
        return frozen


def _shared_prefix_diagnostic(cells, plan):
    """Observe matched successful plan/read requests; missing evidence stays unknown."""
    names={a["config"]["mode"]:a["name"] for a in plan["arms"]}
    grouped={(c["identity"]["question_id"],c["identity"]["repeat"],c["identity"]["arm"]):c["payload"] for c in cells}
    def first(receipt,stage):
        return next((e for e in receipt["trace"] if e.get("name")=="complete"
                     and e.get("request",{}).get("stage")==stage),None)
    pairs=[]
    for qid in sorted(plan["question_ids"]):
        for repeat in range(plan["repeats"]):
            pair={"question_id":qid,"repeat":repeat}
            for stage in ("plan","read"):
                left=first(grouped[(qid,repeat,names["planned_single"])],stage)
                right=first(grouped[(qid,repeat,names["iterative"])],stage)
                observed=all(e and e.get("model_completed") is True and isinstance(e.get("payload_sha256"),str)
                             and isinstance(e.get("response_hash"),str) for e in (left,right))
                pair[stage+"_payload_equal"]=left["payload_sha256"]==right["payload_sha256"] if observed else None
                pair[stage+"_response_equal"]=left["response_hash"]==right["response_hash"] if observed else None
            values=[pair[k] for k in ("plan_payload_equal","plan_response_equal","read_payload_equal","read_response_equal")]
            pair["prefix_verified"]=False if False in values else True if all(v is True for v in values) else None
            pairs.append(pair)
    counts={"verified_pairs":sum(p["prefix_verified"] is True for p in pairs),
            "unknown_pairs":sum(p["prefix_verified"] is None for p in pairs),
            "mismatched_pairs":sum(p["prefix_verified"] is False for p in pairs)}
    valid=False if counts["mismatched_pairs"] else None if counts["unknown_pairs"] else True
    return {"pairs":pairs,"pair_count":len(pairs),**counts,
            "status":"verified" if valid is True else "mismatch" if valid is False else "unknown",
            "all_successful_prefixes_verified":valid,
            "claim":"request/response identity only; missing/failed prefix is unknown, not a quality zero"}


def grade(plan):
    from .task_metrics import score_task
    plan=deepcopy(plan)
    out=Path(plan["output_dir"])
    if plan.get("schema")==THREE_ARM_SCHEMA:
        _,task_list=_preflight(plan,verify_corpus=False)
    else:
        task_list=_task_snapshot(plan)
    tasks={t["question_id"]:t for t in task_list}
    frozen=read(out/"generation_freeze.json")
    cells=_validated_generation(plan,frozen)
    # This is the first parsing of private references, after full generation proof.
    refs=[json.loads(line) for line in _verified_bytes(plan["references_file"]).decode("utf-8").splitlines() if line.strip()]
    byid={str(r["query_id"]):r for r in refs}
    if len(byid)!=len(refs): raise ValueError("duplicate reference question")
    rows=[]; blind=[]; mapping=[]
    for item in cells:
        identity=item["identity"]; receipt=item["payload"]
        qid=identity["question_id"]; task=tasks[qid]; reference=byid[qid]
        if reference["question"]!=task["question"]: raise ValueError("reference question differs")
        if not isinstance(reference.get("reference_answer"),str) or not reference["reference_answer"].strip():
            raise ValueError("calibration reference answer unavailable")
        private={"question_id":qid,"dataset":"browsecomp-plus","answers":[reference["reference_answer"]],
                 "reference_available":True,"official_metric":"llm_judge"}
        metrics=score_task(receipt,private,task=task,allow_proxy_metrics=True)
        for name in ("answer_em","answer_f1"):
            value=metrics.get(name)
            if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:
                raise ValueError("calibration requires an available finite unit score: "+name)
        row={"question_id":qid,"arm":identity["arm"],"repeat":identity["repeat"],"answer_usable":receipt["answer_usable"],
             "source_valid":receipt["citation_source_valid"],
             "answer_origin_valid":receipt["answer_origin_valid"],
             "program_eligible":receipt["execution_ok"] and receipt["answer_origin_valid"] and receipt["answer_usable"],
             "metrics":metrics,"logical_usage":receipt["resource_usage"],
             "diagnostics":diagnose_execution(receipt)}
        rows.append(row)
        blind_id=digest({"plan":digest(plan),"cell":item["file"],"blind":True})[:16]
        blind.append({"blind_id":blind_id,"question":task["question"],"reference_answer":reference["reference_answer"],
                      "answer":receipt["answer"],"answer_usable":receipt["answer_usable"],
                      "answer_origin_valid":receipt["answer_origin_valid"]})
        mapping.append({"blind_id":blind_id,**identity})
    summary={}
    for arm in plan["arms"]:
        group=[r for r in rows if r["arm"]==arm["name"]]
        summary[arm["name"]]={"outcomes":len(group),"delivery_rate":sum(r["answer_usable"] for r in group)/len(group),
          "source_valid_rate":sum(r["source_valid"] for r in group)/len(group),
          "answer_origin_valid_rate":sum(r["answer_origin_valid"] for r in group)/len(group),
          "eligible_outcomes":sum(r["program_eligible"] for r in group),
          "raw_scores_are_diagnostic_if_ineligible":True,
          "proxy_answer_em":sum(r["metrics"]["answer_em"] for r in group)/len(group),
          "proxy_answer_f1":sum(r["metrics"]["answer_f1"] for r in group)/len(group),
          "logical_model_calls":sum(r["logical_usage"]["model_calls"] for r in group),
          "structural_failure_counts":dict(sorted(Counter(code for r in group
              for code in set(r["diagnostics"]["host_observed"])).items())),
          "model_reported_counts":dict(sorted(Counter(code for r in group
              for code in set(r["diagnostics"]["model_reported"])).items())),
          "diagnostic_count_unit":"answer outcome; repeats are not independent observations",
          "model_reports_are_verified_truth":False}
    quality_valid=all(r["program_eligible"] for r in rows)
    report={"schema":"rag-rsi-v3-calibration-report-3" if plan["schema"]==THREE_ARM_SCHEMA else "rag-rsi-v3-calibration-report-2",
      "status":"local_proxy_scored" if quality_valid else "protocol_invalid",
      "quality_comparison_valid":quality_valid,"arms":summary,
      "official_browsecomp_score":False,"semantic_judging":"not performed",
      "question_count":len(tasks),"independent_units":len(tasks),
      "all_outcomes_retained":True,"ledger":frozen["ledger"],
      "claim":"workflow calibration; proxy scores are not official quality evidence",
      "cost_accounting":{
          "request_coupling":REQUEST_COUPLING,
          "physical_calls":frozen["ledger"].get("used",{}).get("run",{}).get("calls",0),
          "physical_cost_cny":frozen["ledger"].get("used",{}).get("run",{}).get("cny",0),
          "logical_calls_by_arm":{name:data["logical_model_calls"] for name,data in summary.items()},
          "per_arm_physical_cost_attribution":"not_identifiable_with_shared_cache",
          "same_realized_cost":False,"repeat_is_independent_unit":False}}
    if plan["schema"]==THREE_ARM_SCHEMA:
        report["analysis"]=paired_analysis(rows,plan["analysis"],arm_names=[a["name"] for a in plan["arms"]],expected_repeats=plan["repeats"])
        report["independent_units"]=len(set(plan["analysis"]["question_groups"].values()))
        report["question_use"]=plan["question_use"]
        report["question_newness_independently_verified"]=False
        prefix=_shared_prefix_diagnostic(cells,plan)
        report["shared_prefix_diagnostic"]=prefix
        report["execution_comparison_valid"]=quality_valid
        report["mechanism_comparison_valid"]=prefix["all_successful_prefixes_verified"] if quality_valid else False
        report["mechanism_claim"]=("shared_prefix_verified; effectiveness not established" if report["mechanism_comparison_valid"] is True
             else "prefix_unverified_or_protocol_invalid; no isolated_iteration_effect_claim")
        if prefix["mismatched_pairs"]:
            # Keep every raw outcome, but do not infer the pre-registered contrasts
            # after an observed violation of their shared-prefix protocol.
            report["quality_comparison_valid"]=False
            report["status"]="protocol_invalid"
            report["analysis"].update(status="protocol_invalid",quality_comparison_valid=False,qualified=None,
                                      protocol_failure="observed_shared_prefix_mismatch")
    else:
        raw_deltas={qid:sum(r["metrics"]["answer_f1"]*(1 if r["arm"]==plan["arms"][1]["name"] else -1)
                           for r in rows if r["question_id"]==qid)/plan["repeats"] for qid in tasks}
        report.update(paired_order=plan["arms"][1]["name"]+" minus "+plan["arms"][0]["name"],
          paired_question_f1_deltas=raw_deltas if quality_valid else None,
          raw_paired_question_f1_deltas=raw_deltas)
    freeze(out/"grading/rows.json",rows); freeze(out/"grading/blind_packet.json",sorted(blind,key=lambda r:r["blind_id"]))
    freeze(out/"grading/private_map.json",mapping); freeze(out/"report.json",report)
    return report
