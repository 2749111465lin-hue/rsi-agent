"""Task-neutral candidate execution through the existing isolated WSL runtime.

Only trusted host code may score outputs. A candidate's semantic diagnostics are
labelled as reported, not trusted measurement; host resource/answer facts are
recorded independently. No old stage runner is imported.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
from ..archive import ProgramArchive
from ..budget import LimitExceeded, digest, save, stable
from ..sandbox import Sandbox, SandboxExecutionError
from .datasets import validate_public_task, evaluate_answer, filter_documents
from .infrastructure import LocalCorpus, UnknownProviderOutcome, ModelResponseError
from .rag import ground_quote
from .task_metrics import score_task


def root_files(config=None):
    core = Path(__file__).with_name("rag.py").read_text(encoding="utf-8")
    wrapper = """import json
from rag_core import RagEngine
CONFIG = json.loads(%r)
class Backend:
    def __init__(self, services): self.services = services
    def search(self, query, limit=5):
        return self.services.call('search', {'query':query,'limit':limit})
    def read(self, docid, start, end):
        return self.services.call('read', {'docid':docid,'start':start,'end':end})
class Model:
    def __init__(self, services): self.services = services
    def complete(self, stage, payload):
        return self.services.call('complete', {'stage':stage,'payload':payload})
def solve(question, services):
    result = RagEngine(Backend(services),Model(services),config=CONFIG).solve({'question':question})
    services.call('record_trace', {'result':result})
    citations = [c for c in result['state']['citations'] if c['citation_id'] in result['citation_ids']]
    return {'answer':result['answer'] or '', 'citations':citations,
            'abstention_reason':'insufficient evidence' if result['abstained'] else None}
""" % stable(config or {})
    return {"rag.py":wrapper,"rag_core.py":core}


def validate_sources(files):
    if not isinstance(files,dict) or set(files) != {"rag.py","rag_core.py"}:
        raise ValueError("candidate must contain exactly rag.py and rag_core.py")
    for name,source in files.items():
        if not isinstance(source,str) or len(source.encode()) > 220000:
            raise ValueError("source size limit")
        ast.parse(source,filename=name)
    tree=ast.parse(files["rag.py"])
    if not any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=="solve" for n in tree.body):
        raise ValueError("candidate solve entry missing")
    # Syntax is only a cheap preflight. Capability isolation is enforced by WSL,
    # not by this AST check, and quality is never inferred from syntax success.


class HostError(RuntimeError):
    """A trusted service failure; never a measured candidate-quality zero."""


class CandidateEvidenceError(ValueError):
    """The candidate supplied evidence without the required host provenance."""


def _snapshot(value):
    return json.loads(stable(value))


def _source_identity(item, *, quote=False):
    if not isinstance(item, dict):
        return None
    docid = item.get("docid", item.get("doc_id"))
    lo, hi = item.get("start"), item.get("end")
    text = item.get("quote" if quote else "text")
    if (not isinstance(docid, (str, int)) or isinstance(docid, bool) or not str(docid)
            or type(lo) is not int or type(hi) is not int or lo < 0 or hi <= lo
            or not isinstance(text, str) or len(text) != hi-lo):
        return None
    return str(docid), lo, hi, text


def _matches_source(item, windows, *, quote=False):
    identity = _source_identity(item, quote=quote)
    if identity is None:
        return False
    docid, lo, hi, text = identity
    return any(docid == str(window["docid"])
               and window["start"] <= lo < hi <= window["end"]
               and window["text"][lo-window["start"]:hi-window["start"]] == text
               for window in windows)


class HostBroker:
    def __init__(self, task, backend, model, *, max_models=7, max_searches=8, max_reads=8):
        validate_public_task(task)
        if task["task_type"] != "qa":
            raise ValueError("retrieval-only benchmark requires retrieval evaluation")
        self.task,self.backend,self.model=_snapshot(task),backend,model
        self.max_models,self.max_searches,self.max_reads=max_models,max_searches,max_reads
        self.counts={"model_calls":0,"search_calls":0,"read_calls":0}
        self.events=[]; self.reported=None; self.source_windows=[]; self.model_errors=[]; self.fatal=None
        self.read_presentations=[]
        self.verified_read_quotes=set()
        self.final_observations=[]

    def _record_windows(self, value, *, many):
        rows = value if many else [value]
        if not isinstance(rows, list) or any(_source_identity(row) is None for row in rows):
            raise HostError("backend returned an invalid source window")
        rows = _snapshot(rows)
        self.source_windows.extend(rows)
        return rows if many else rows[0]

    def _presented_sources(self, material):
        sources = material.get("sources", [])
        if not isinstance(sources, list):
            raise CandidateEvidenceError("read sources must be a list")
        by_id = {}
        for source in sources:
            sid = source.get("source_id") if isinstance(source, dict) else None
            if not isinstance(sid, str) or not sid or sid in by_id:
                raise CandidateEvidenceError("read source IDs must be unique nonempty strings")
            if not _matches_source(source, self.source_windows):
                raise CandidateEvidenceError("read source was not returned by the host backend")
            by_id[sid] = _snapshot(source)
        return by_id

    def _presented_evidence(self, material):
        evidence = material.get("evidence", [])
        if not isinstance(evidence, list):
            raise CandidateEvidenceError("answer evidence must be a list")
        by_id = {}
        for item in evidence:
            cid = item.get("citation_id") if isinstance(item, dict) else None
            if not isinstance(cid, str) or not cid or cid in by_id:
                raise CandidateEvidenceError("answer citation IDs must be unique nonempty strings")
            identity = _source_identity(item, quote=True)
            if (identity not in self.verified_read_quotes
                    or not _matches_source(item, self.source_windows, quote=True)):
                raise CandidateEvidenceError("answer quote lacks verified read-stage provenance")
            by_id[cid] = _snapshot(item)
        return by_id

    def _record_read(self, sources, result):
        verified = []
        claims = result.get("claims", []) if isinstance(result, dict) else []
        # Malformed completed responses remain model failures, not host truth.
        if not isinstance(claims, list):
            claims = []
        for claim in claims:
            citations = claim.get("citations", []) if isinstance(claim, dict) else []
            if not isinstance(citations, list):
                continue
            for citation in citations:
                if not isinstance(citation, dict) or not isinstance(citation.get("source_id"), str):
                    continue
                source = sources.get(citation["source_id"])
                # This helper is imported from trusted host code, never from the
                # candidate's mutable rag_core.py. Only this call's window counts.
                grounded = ground_quote(citation, source)
                if grounded is None:
                    continue
                item = {**grounded, "docid": source["docid"]}
                if _matches_source(item, [source], quote=True):
                    identity = _source_identity(item, quote=True)
                    self.verified_read_quotes.add(identity)
                    verified.append({"docid": identity[0], "start": identity[1],
                                     "end": identity[2], "quote": identity[3]})
        self.read_presentations.append({"event_index": len(self.events),
                                        "sources": list(sources.values()),
                                        "verified_quotes": verified,
                                        "semantic_support": "model_assessed_only"})

    def citation_receipt(self, answer, citations):
        """Host reconstruction; no field from record_trace is used as evidence."""
        default = {"valid": False, "status": "no_observed_final_answer", "raw_citation_ids": [],
                   "presented_citation_ids": [], "model_claims_evidence": None,
                   "semantic_support": "model_assessed_only"}
        if not isinstance(answer, str):
            return default
        observations = [item for item in self.final_observations
                        if isinstance(item["response"].get("answer"), str)
                        and item["response"]["answer"].strip() == answer.strip()]
        if not observations:
            return default
        observed = observations[-1]
        raw = observed["response"].get("citation_ids")
        evidence = observed["evidence"]
        assessment = observed["response"].get("evidence_sufficient")
        record = {**default, "status": "missing_citations",
                  "presented_citation_ids": list(evidence),
                  "model_claims_evidence": assessment if type(assessment) is bool else None}
        if (not isinstance(raw, list) or not all(isinstance(cid, str) and cid for cid in raw)
                or len(raw) != len(set(raw))):
            return {**record, "status": "invalid_model_citation_ids"}
        record["raw_citation_ids"] = list(raw)
        if not raw:
            return record
        if any(cid not in evidence for cid in raw):
            return {**record, "status": "invalid_model_citation_ids"}
        if not isinstance(citations, list) or len(citations) != len(raw):
            return {**record, "status": "candidate_citation_mismatch"}
        supplied = {}
        for citation in citations:
            cid = citation.get("citation_id") if isinstance(citation, dict) else None
            if not isinstance(cid, str) or cid in supplied or cid not in evidence:
                return {**record, "status": "candidate_citation_mismatch"}
            if (_source_identity(citation, quote=True) != _source_identity(evidence[cid], quote=True)
                    or not _matches_source(citation, self.source_windows, quote=True)):
                return {**record, "status": "candidate_citation_mismatch"}
            supplied[cid] = citation
        if set(supplied) != set(raw):
            return {**record, "status": "candidate_citation_mismatch"}
        return {**record, "valid": True, "status": "source_and_presentation_verified"}

    def __call__(self, name, payload, remaining=180):
        if self.fatal is not None:
            raise self.fatal
        if remaining<=0 or not isinstance(payload,dict) or len(stable(payload).encode())>400000:
            raise ValueError("RPC request over budget")
        payload = _snapshot(payload)
        if name == "search":
            if (set(payload)!={"query","limit"} or self.counts["search_calls"]>=self.max_searches
                    or not isinstance(payload["query"],str) or not payload["query"].strip()
                    or len(payload["query"])>16000 or type(payload["limit"]) is not int
                    or not 1<=payload["limit"]<=30):
                raise ValueError("search limit or schema")
            self.counts["search_calls"]+=1
            try:
                result=self.backend.search(payload["query"],payload["limit"])
                if not isinstance(result,list) or any(not isinstance(row,dict) for row in result):
                    raise HostError("backend search returned an invalid result")
                result=[row for row in result if str(row.get("docid")) not in self.task["excluded_docids"]]
                result=self._record_windows(result,many=True)
            except Exception as exc:
                self.fatal=exc if isinstance(exc,HostError) else HostError("backend search failed: "+type(exc).__name__)
                raise self.fatal from exc
        elif name == "read":
            if (set(payload)!={"docid","start","end"} or self.counts["read_calls"]>=self.max_reads
                    or not isinstance(payload["docid"],(str,int)) or isinstance(payload["docid"],bool)
                    or type(payload["start"]) is not int or type(payload["end"]) is not int
                    or not 0<=payload["start"]<payload["end"]):
                raise ValueError("read limit or schema")
            self.counts["read_calls"]+=1
            if str(payload["docid"]) in self.task["excluded_docids"]:
                raise ValueError("excluded source")
            try:
                result=self._record_windows(self.backend.read(**payload),many=False)
            except Exception as exc:
                self.fatal=exc if isinstance(exc,HostError) else HostError("backend read failed: "+type(exc).__name__)
                raise self.fatal from exc
        elif name == "complete":
            if set(payload)!={"stage","payload"} or payload["stage"] not in {"plan","read","answer"} or not isinstance(payload["payload"],dict):
                raise ValueError("model action schema")
            stage=payload["stage"]
            cap=self.max_models if stage=="answer" else self.max_models-1
            if self.counts["model_calls"]>=cap:
                raise ValueError("reserved final answer budget")
            visible={**payload["payload"],"question":self.task["question"]}
            sources=self._presented_sources(visible) if stage=="read" else {}
            evidence=self._presented_evidence(visible) if stage=="answer" else {}
            self.counts["model_calls"]+=1
            completed=False
            # Reference objects are never available to this broker or sandbox.
            try:
                if hasattr(self.model,"timeout_seconds"):
                    self.model.timeout_seconds=max(.1,min(150,remaining-4))
                result=self.model.complete(stage,_snapshot(visible))
                try:
                    result=_snapshot(result)
                    if not isinstance(result,dict):
                        raise ValueError("non-object model result")
                    meta=result.get("_meta",{})
                    if not isinstance(meta,dict):
                        raise ValueError("invalid model result metadata")
                    if meta.get("truncated") or meta.get("finish_reason") in ("length","max_tokens","error"):
                        raise ModelResponseError("truncated or failed completed response")
                except ModelResponseError:
                    raise
                except (ValueError,TypeError) as exc:
                    raise ModelResponseError("completed response is not a JSON object") from exc
                completed=True
            except (LimitExceeded,UnknownProviderOutcome) as exc:
                self.fatal=exc
                raise
            except ModelResponseError as exc:
                self.model_errors.append(type(exc).__name__)
                result={"_meta":{"truncated":"truncat" in str(exc).lower(),"finish_reason":"error"}}
            except Exception as exc:
                self.fatal=HostError("model service failed: "+type(exc).__name__)
                raise self.fatal from exc
            if completed and stage=="read":
                self._record_read(sources,result)
            if completed and stage=="answer":
                self.final_observations.append({"event_index":len(self.events),
                                                "evidence":evidence,"response":_snapshot(result),
                                                "payload_sha256":digest(visible)})
        elif name == "record_trace":
            if set(payload)!={"result"} or not isinstance(payload["result"],dict) or self.reported is not None:
                raise ValueError("one candidate diagnostic object allowed")
            self.reported=_snapshot(payload["result"]); result={"recorded":True}
        else:
            raise ValueError("unavailable service")
        event={"name":name,"request":payload,"response_hash":digest(result)}
        if name in {"search","read"}:
            # Both branches have completed _record_windows validation. Never use
            # backend metadata or the candidate's reported trace as observations.
            windows=result if name=="search" else [result]
            observed=[]
            for row in windows[:30]:
                docid,lo,hi,text=_source_identity(row)
                observed.append({"docid":docid,"start":lo,"end":hi,
                                 "text_sha256":hashlib.sha256(text.encode("utf-8")).hexdigest()})
            # A nonconforming backend can exceed its requested limit. Bound only
            # the new log field; preserve the existing RPC result and its hash.
            event.update(observed_windows=observed,observed_window_count=len(windows),
                         observed_windows_truncated=len(windows)>30)
        self.events.append(event)
        return _snapshot(result)


def execute(archive, node_id, task, backend, model, directory, *, limits=None, sandbox=None):
    validate_public_task(task)
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    node=archive.load_node(node_id)
    export=directory/"export"
    if not export.exists():
        archive.export_node(node_id,export)
    else:
        program=archive.load_program(node["program_id"])
        if {p.name for p in (export/"files").iterdir() if p.is_file()} != set(program["files"]):
            raise ValueError("candidate export file set changed")
        for name,source in program["files"].items():
            if (export/"files"/name).read_text(encoding="utf-8")!=source:
                raise ValueError("candidate export changed")
    stub=directory/"corpus_stub.json"; stub.write_text("[]",encoding="utf-8")
    broker=HostBroker(task,backend,model,**(limits or {}))
    try:
        result=(sandbox or Sandbox()).run(export/"files",stub,task["question"],broker,seconds=180)
        if broker.fatal is not None:
            raise broker.fatal
        answer=result["result"]["answer"]
        usable=isinstance(answer,str) and bool(answer.strip())
        citations=result["result"]["citations"]
        citation_check=broker.citation_receipt(answer,citations)
        checks=result["runtime"].get("isolation_checks") or {}
        failures=[] if usable else ["answer_empty"]
        if citation_check["status"] in {"invalid_model_citation_ids","candidate_citation_mismatch"}:
            failures.append("invalid_answer_citation")
        receipt={"answer":answer,"answer_usable":usable,"execution_ok":True,
            "citation_source_valid":citation_check["valid"],"citation_status":citation_check["status"],
            "host_citation_validation":citation_check,"citations":citations,
            "isolation_verified":bool(checks) and all(checks.values()),
            "failure_classes":failures}
    except SandboxExecutionError as exc:
        if broker.fatal is not None:
            raise broker.fatal
        receipt={"answer":"","answer_usable":False,"execution_ok":False,
            "citation_source_valid":False,"citation_status":"execution_failed","citations":[],
            "isolation_verified":False,
            "failure_classes":[exc.details.get("kind","execution_error")],"error":str(exc)}
    receipt.update({"schema":"rag-rsi-v3-execution-2","node_id":node_id,"program_id":node["program_id"],
         "question_id":task["question_id"],"resource_usage":broker.counts,
         "model_errors":broker.model_errors,"trace":broker.events,
         "host_evidence_trace":{"read_presentations":broker.read_presentations,
                                "final_observations":broker.final_observations},
         "candidate_reported":broker.reported,"diagnostic_trust":"candidate_reported_not_correctness"})
    save(directory/"execution.json",receipt)
    return receipt


def _cell_record(identity, payload):
    return {"schema":"rag-rsi-v3-measured-cell-2","identity":identity,
            "identity_sha256":digest(identity),"payload":payload,"payload_sha256":digest(payload)}


def _verified_cell(path, expected):
    record=json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(record,dict) or set(record)!={"schema","identity","identity_sha256","payload","payload_sha256"}
            or record["schema"]!="rag-rsi-v3-measured-cell-2"
            or record["identity"]!=expected or record["identity_sha256"]!=digest(expected)
            or not isinstance(record["payload"],dict)
            or record["payload_sha256"]!=digest(record["payload"])):
        raise ValueError("measured cell identity or content integrity failure")
    row=record["payload"]
    for field in ("node_id","program_id","question_id","role","repeat"):
        if row.get(field)!=expected[field]:
            raise ValueError("measured cell field differs from identity: "+field)
    if type(row.get("score")) not in (int,float) or not 0<=row["score"]<=1:
        raise ValueError("cached score outside unit interval")
    if type(row.get("execution_ok")) is not bool or type(row.get("answer_usable")) is not bool:
        raise ValueError("cached cell lacks execution validity")
    metrics=row.get("task_metrics")
    names=("answer_em","answer_f1","support_em","support_f1","answerability")
    if (not isinstance(metrics,dict) or not set(names)<=set(metrics)
            or metrics.get("question_id")!=expected["question_id"]
            or metrics.get("dataset")!=expected["dataset"]
            or not isinstance(metrics.get("protocol"),str) or not metrics["protocol"]
            or not isinstance(metrics.get("metric_status"),dict)):
        raise ValueError("cached cell lacks matching task metrics")
    for name in names:
        value=metrics[name]
        if value is not None and (type(value) not in (int,float) or not 0<=value<=1):
            raise ValueError("cached task metric outside unit interval")
        if not isinstance(metrics["metric_status"].get(name),str) or not metrics["metric_status"][name]:
            raise ValueError("cached task metric lacks availability status")
    return row


class Measurement:
    def __init__(self, archive, directory, model_factory, backend_factory=None, *, metric="em", scorer=None, limits=None,
                 allow_proxy_metrics=False):
        if type(allow_proxy_metrics) is not bool:
            raise ValueError("allow_proxy_metrics must be explicitly boolean")
        self.archive,self.directory=archive,Path(directory)
        self.model_factory,self.backend_factory=model_factory,backend_factory
        self.metric,self.scorer,self.limits=metric,scorer,limits
        self.allow_proxy_metrics=allow_proxy_metrics
        primary="task-rule-"+metric if scorer is None else "external-"+metric
        self.epoch=primary+"-v4-task-metrics-1-proxy-"+str(int(allow_proxy_metrics))

    def backend(self, task):
        if task["corpus_scope"]=="question_local" or task["documents"]:
            return LocalCorpus(filter_documents(task,task["documents"]),scope=task["question_id"],excluded=task["excluded_docids"])
        if self.backend_factory is None:
            raise ValueError("shared corpus adapter required")
        return self.backend_factory(task)

    def run(self,node,public_tasks,references,*,role,bank,repeats=1):
        if role not in {"D_fit","D_select","D_report"} or type(repeats) is not int or not 1<=repeats<=8:
            raise ValueError("measurement contract")
        if not public_tasks:
            raise ValueError("empty panel")
        if len({task["question_id"] for task in public_tasks})!=len(public_tasks):
            raise ValueError("duplicate question in measurement panel")
        for task in public_tasks:
            validate_public_task(task)
            ref=references[task["question_id"]]
            if ref["question_id"]!=task["question_id"] or ref["dataset"]!=task["dataset"]:
                raise ValueError("reference identity mismatch")
            if task["dataset"]=="musique" and ref.get("answerable") is False:
                raise ValueError("MuSiQue-Full requires explicit paired sufficiency evaluation; scalar Measurement cannot score Full")
            if task["dataset"]=="browsecomp-plus" and self.scorer is None and not self.allow_proxy_metrics:
                raise ValueError("BCP default rule score requires explicit allow_proxy_metrics; it is not an official judge")
        stored=self.archive.load_node(node["node_id"])
        if stored["program_id"]!=node["program_id"]:
            raise ValueError("node and program identities differ")
        panel=digest(public_tasks)
        environments={}; backends={}; models={}
        for task in public_tasks:
            qid=task["question_id"]
            backend=self.backend(task)
            if not getattr(backend,"identity",None):
                raise ValueError("backend must expose a frozen identity")
            backends[qid]=backend
            environments[qid]={"backend":backend.identity,"models":{}}
            for rep in range(repeats):
                model=self.model_factory(f"{role}/{bank}/{qid}/{rep}")
                if not getattr(model,"identity",None):
                    raise ValueError("model must expose a frozen identity")
                models[(qid,rep)]=model
                environments[qid]["models"][str(rep)]=model.identity
        identity=digest({"node":node["node_id"],"program":node["program_id"],"panel":panel,
            "reference_hash":digest(references),"epoch":self.epoch,"role":role,"bank":bank,
            "repeats":repeats,"limits":self.limits,"environments":environments,
            "allow_proxy_metrics":self.allow_proxy_metrics,
            "cell_schema":"rag-rsi-v3-measured-cell-2"})
        folder=self.directory/identity
        rows=[]
        for task in public_tasks:
            qid=task["question_id"]
            for rep in range(repeats):
                cell=folder/(digest(qid)[:16]+"_"+str(rep))
                done=cell/"measured.json"
                expected={"measurement_identity_hash":identity,"node_id":node["node_id"],
                    "program_id":node["program_id"],"question_id":qid,"dataset":task["dataset"],
                    "role":role,"bank":bank,"repeat":rep,
                    "task_hash":digest(task),"reference_hash":digest(references[qid]),
                    "evaluator_epoch":self.epoch,"backend_identity":environments[qid]["backend"],
                    "model_identity":environments[qid]["models"][str(rep)]}
                if done.exists():
                    row=_verified_cell(done,expected)
                else:
                    backend,model=backends[qid],models[(qid,rep)]
                    if backend.identity!=expected["backend_identity"] or model.identity!=expected["model_identity"]:
                        raise HostError("measurement environment changed before execution")
                    receipt=execute(self.archive,node["node_id"],task,backend,model,cell,limits=self.limits)
                    ref=references[qid]
                    metrics=score_task(receipt,ref,task=task,allow_proxy_metrics=self.allow_proxy_metrics)
                    score=(self.scorer(receipt["answer"],ref) if self.scorer else
                           evaluate_answer(receipt["answer"],ref,self.metric)) if receipt["answer_usable"] else 0.0
                    if type(score) not in (int,float) or not 0<=score<=1:
                        raise ValueError("scorer must return unit score")
                    row={**receipt,"score":score,"task_metrics":metrics,"repeat":rep,"role":role}
                    save(done,_cell_record(expected,row))
                rows.append(row)
        per_question={task["question_id"]:sum(x["score"] for x in rows if x["question_id"]==task["question_id"])/repeats for task in public_tasks}
        result={"node_id":node["node_id"],"program_id":node["program_id"],"role":role,
           "panel_hash":panel,"identity_hash":identity,"evaluator_epoch":self.epoch,
           "score":sum(per_question.values())/len(per_question),"per_question":per_question,
           "complete":True,"valid_program":all(r["execution_ok"] for r in rows),"rows":rows,
           "resource_usage":{"calls":sum(r["resource_usage"]["model_calls"] for r in rows)},
           "metric":self.metric,"cost_note":"logical calls; billed cost is authoritative ledger only"}
        save(folder/"measurement.json",result)
        return result