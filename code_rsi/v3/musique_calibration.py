"""Pure MuSiQue calibration materials and task-local backends.

Support labels are used only to construct an explicitly privileged diagnostic
condition. Answer/decomposition values are never accessed or returned here.
No file, network, reference acquisition, or model call occurs in this module.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib

from ..budget import digest
from .datasets import filter_documents, validate_task_collection
from .infrastructure import LocalCorpus

SCHEMA = "rag-rsi-v3-musique-calibration-1"
ARMS = {"planned": "planned_single", "loop": "iterative", "support": "planned_single"}
MATERIALS_SCHEMA = "rag-rsi-musique-support-materials-1"
SUPPORT_BACKEND_VERSION = "musique-complete-support-context-1"


class MaterialError(ValueError):
    """Invalid frozen materials; messages never expose task or reference values."""


def _tasks(tasks):
    if not isinstance(tasks, list) or not tasks:
        raise MaterialError("nonempty MuSiQue task list required")
    try:
        validate_task_collection(tasks)
    except (ValueError, TypeError):
        raise MaterialError("invalid public task panel") from None
    if any(task["dataset"] != "musique" or task["corpus_scope"] != "question_local"
           or task["corpus_ref"] is not None or task["task_type"] != "qa" for task in tasks):
        raise MaterialError("only original question-local MuSiQue QA tasks allowed")
    return {task["question_id"]: task for task in tasks}


def _support_ids(value, task):
    if (not isinstance(value, list) or not 2 <= len(value) <= 4
            or any(not isinstance(x, str) or not x for x in value)
            or len(set(value)) != len(value)):
        raise MaterialError("two to four distinct support document IDs required")
    ids = set(value)
    allowed = {doc["docid"] for doc in task["documents"]}
    if not ids <= allowed or ids & set(task["excluded_docids"]):
        raise MaterialError("support document outside original allowed task context")
    return ids


def build_support_materials(tasks, refs):
    """Project only support IDs and their public-task binding from private refs.

    Reference identity metadata and explicit annotation availability are checked;
    answer strings, aliases, answerability and decomposition are not accessed.
    The original task/question/document identities remain unchanged.
    """
    byid = _tasks(tasks)
    if not isinstance(refs, Mapping) or set(refs) != set(byid):
        raise MaterialError("references must cover exactly the public task panel")
    rows = []
    for qid, task in byid.items():
        ref = refs[qid]
        if (not isinstance(ref, Mapping) or ref.get("question_id") != qid
                or ref.get("dataset") != "musique"):
            raise MaterialError("support reference identity differs from public task")
        if "support_annotation_available" in ref and ref["support_annotation_available"] is not True:
            raise MaterialError("support annotation is not explicitly available")
        ids = _support_ids(ref.get("supporting_docids"), task)
        rows.append({"question_id": qid,
                     "docids": [d["docid"] for d in task["documents"] if d["docid"] in ids]})
    result = {"schema": MATERIALS_SCHEMA, "tasks_sha256": digest(tasks), "rows": rows}
    validate_support_materials(result, tasks)
    return result


def validate_support_materials(value, tasks):
    """Return detached original full documents, never labels or rewritten tasks.

    This validates the frozen projection against public inputs. Its agreement
    with private support annotations is checked by the post-generation grader.
    """
    byid = _tasks(tasks)
    if (not isinstance(value, dict) or set(value) != {"schema", "tasks_sha256", "rows"}
            or value["schema"] != MATERIALS_SCHEMA or value["tasks_sha256"] != digest(tasks)
            or not isinstance(value["rows"], list) or len(value["rows"]) != len(byid)):
        raise MaterialError("support materials structure or task binding differs")
    result = {}
    for row in value["rows"]:
        if (not isinstance(row, dict) or set(row) != {"question_id", "docids"}
                or not isinstance(row["question_id"], str) or row["question_id"] not in byid
                or row["question_id"] in result):
            raise MaterialError("support materials need one row per original task")
        qid = row["question_id"]
        task = byid[qid]
        ids = _support_ids(row["docids"], task)
        docs = [doc for doc in task["documents"] if doc["docid"] in ids]
        if row["docids"] != [doc["docid"] for doc in docs]:
            raise MaterialError("support IDs must retain original public document order")
        result[qid] = deepcopy(docs)
    if set(result) != set(byid):
        raise MaterialError("support materials omit an original task")
    return {qid: result[qid] for qid in byid}


class SupportCorpus(LocalCorpus):
    """Diagnostic backend returning every complete support document in order.

    Query-independent delivery removes document selection, not reader/generator
    errors. The caller must separately ensure all complete text fits its read
    input budget. It must not identify this as normal retrieval performance.
    """
    def __init__(self, documents, *, scope):
        if (not isinstance(scope, str) or not scope or not isinstance(documents, list)
                or not 2 <= len(documents) <= 4):
            raise MaterialError("bounded support documents and original task scope required")
        for row in documents:
            if (not isinstance(row, dict) or not {"docid", "text"} <= set(row)
                    or set(row) - {"docid", "text", "title", "url"}
                    or not isinstance(row["docid"], str) or not row["docid"].startswith(scope + "/p/")
                    or not isinstance(row["text"], str) or not row["text"].strip()
                    or any(not isinstance(row[k], str) for k in ("title", "url") if k in row)):
                raise MaterialError("support backend accepts only original public document records")
        docs = deepcopy(documents)
        try:
            super().__init__(docs, scope=scope)
        except (ValueError, TypeError):
            raise MaterialError("invalid or repeated support document identity") from None
        self._ordered_docids = tuple(row["docid"] for row in docs)
        self.identity = digest({"version": SUPPORT_BACKEND_VERSION, "scope": scope, "documents": docs})

    def search(self, query, limit=5):
        if (not isinstance(query, str) or not query.strip() or type(limit) is not int
                or not 1 <= limit <= 30):
            raise MaterialError("valid query and bounded integer limit required")
        if limit < len(self._ordered_docids):
            raise MaterialError("support delivery limit would omit an annotated document")
        return [{"docid": docid, "text": self.docs[docid], "start": 0,
                 "end": len(self.docs[docid]),
                 "document_hash": hashlib.sha256(self.docs[docid].encode("utf-8")).hexdigest(),
                 "score": 0.0} for docid in self._ordered_docids]


def build_backends(tasks, materials):
    """Create shared normal backends plus isolated support-diagnostic backends."""
    byid = _tasks(tasks)
    supports = validate_support_materials(materials, tasks)
    backends = {}
    identities = []
    for qid, task in byid.items():
        normal = LocalCorpus(filter_documents(task, task["documents"]),
                             scope=qid, excluded=task["excluded_docids"])
        diagnostic = SupportCorpus(supports[qid], scope=qid)
        for arm, backend in (("planned", normal), ("loop", normal), ("support", diagnostic)):
            backends[(qid, arm)] = backend
            identities.append({"question_id": qid, "arm": arm, "backend": backend.identity})
    identities.sort(key=lambda item: (item["question_id"], item["arm"]))
    panel_identity = digest({"schema": SCHEMA, "backend_identities": identities})
    return backends, panel_identity


def evidence_stages(receipt, task, reference):
    """Measure support-document membership along observed host evidence stages.

    Only support IDs are read from the reference. This does not measure semantic
    sufficiency or recall of answer-bearing sentences. Incomplete executions
    and missing stage logs yield null metrics, never zero. The final stage uses
    input evidence of the last successful answer call, not its output citations.
    """
    materials = build_support_materials([task], {task["question_id"]: reference})
    gold = set(materials["rows"][0]["docids"])
    docs = {row["docid"]: row["text"] for row in task["documents"]}
    if not isinstance(receipt, Mapping) or receipt.get("question_id") != task["question_id"]:
        raise MaterialError("evidence receipt differs from original task identity")

    def metric(ids=None):
        return {"complete": ids is not None,
                "document_count": len(ids) if ids is not None else None,
                "matched_support_count": len(ids & gold) if ids is not None else None,
                "support_recall": len(ids & gold) / len(gold) if ids is not None else None}

    def docids(rows):
        if (not isinstance(rows, list)
                or any(not isinstance(row, Mapping) or not isinstance(row.get("docid"), str)
                       or row["docid"] not in docs for row in rows)):
            return None
        return {row["docid"] for row in rows}

    names = ("retrieval", "presented", "quoted", "final_seen", "full_support_presented")
    result = {"schema": "rag-rsi-musique-evidence-stages-1",
              "support_document_count": len(gold),
              "execution_complete": receipt.get("execution_ok") is True,
              "document_membership_only": True, "semantic_support_verified": False,
              "stages": {name: metric() for name in names}}
    if not result["execution_complete"]:
        return result
    trace = receipt.get("trace")
    usage = receipt.get("resource_usage")
    if (not isinstance(trace, list) or any(not isinstance(event, Mapping) for event in trace)
            or not isinstance(usage, Mapping)):
        return result

    searches = [event for event in trace if event.get("name") == "search"]
    retrieved = set()
    search_complete = (type(usage.get("search_calls")) is int
                       and usage["search_calls"] == len(searches))
    for event in searches:
        rows = event.get("observed_windows")
        ids = docids(rows)
        if (ids is None or event.get("observed_windows_truncated") is not False
                or type(event.get("observed_window_count")) is not int
                or event["observed_window_count"] != len(rows)):
            search_complete = False
        elif search_complete:
            retrieved.update(ids)
    if search_complete:
        result["stages"]["retrieval"] = metric(retrieved)

    calls = [(index, event) for index, event in enumerate(trace) if event.get("name") == "complete"]
    if (type(usage.get("model_calls")) is not int or usage["model_calls"] != len(calls)
            or any(not isinstance(event.get("request"), Mapping)
                   or event["request"].get("stage") not in ("plan", "read", "answer")
                   or not isinstance(event["request"].get("payload"), Mapping)
                   or type(event.get("model_completed")) is not bool for _, event in calls)):
        return result
    host = receipt.get("host_evidence_trace")
    if not isinstance(host, Mapping):
        return result
    read_calls = [(index, event) for index, event in calls if event["request"]["stage"] == "read"]
    successful_reads = [index for index, event in read_calls if event["model_completed"]]
    reads = host.get("read_presentations")
    reads_complete = (isinstance(reads, list)
                      and all(isinstance(row, Mapping) and type(row.get("event_index")) is int for row in reads)
                      and [row["event_index"] for row in reads] == successful_reads)
    if reads_complete:
        presented, quoted, full = set(), set(), set()
        sources_complete = quotes_complete = True
        for observed in reads:
            sources = observed.get("sources")
            source_ids = docids(sources)
            quote_ids = docids(observed.get("verified_quotes"))
            if source_ids is None:
                sources_complete = False
            else:
                presented.update(source_ids)
                for source in sources:
                    text = docs[source["docid"]]
                    if (type(source.get("start")) is int and source["start"] == 0
                            and type(source.get("end")) is int and source["end"] == len(text)
                            and source.get("text") == text
                            and source.get("text_sha256") == hashlib.sha256(text.encode("utf-8")).hexdigest()):
                        full.add(source["docid"])
            if quote_ids is None or source_ids is None or not quote_ids <= source_ids:
                quotes_complete = False
            else:
                quoted.update(quote_ids)
        # Failed model read calls have no read_presentations record; do not
        # mistake its absent sources for a known empty reader input.
        if sources_complete and len(successful_reads) == len(read_calls):
            result["stages"]["presented"] = metric(presented)
            result["stages"]["full_support_presented"] = metric(full)
        # A failed completed read response cannot create host-verified quotes.
        if quotes_complete:
            result["stages"]["quoted"] = metric(quoted)

    finals = host.get("final_observations")
    successful_answers = [index for index, event in calls
                          if event["request"]["stage"] == "answer" and event["model_completed"]]
    if (isinstance(finals, list) and finals
            and all(isinstance(row, Mapping) and type(row.get("event_index")) is int for row in finals)
            and [row["event_index"] for row in finals] == successful_answers
            and isinstance(finals[-1].get("evidence"), Mapping)):
        ids = docids(list(finals[-1]["evidence"].values()))
        if ids is not None:
            result["stages"]["final_seen"] = metric(ids)
    return result
