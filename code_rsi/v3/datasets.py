"""Offline dataset adapters with allowlisted public inputs and private references.

No files, network, models, or scoring references are consulted by public runtime
helpers. Gold evidence is never used to construct a retrieval corpus.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import re
import string
from typing import Any, Mapping, TypedDict


class DatasetFormatError(ValueError):
    """An unsupported, ambiguous, or leaking dataset/task representation."""


class PublicTask(TypedDict):
    id: str
    question_id: str
    dataset: str
    question: str
    task_type: str
    documents: list[dict[str, str]]
    corpus_ref: str | None
    corpus_scope: str
    excluded_docids: list[str]


_PUBLIC_KEYS = set(PublicTask.__annotations__)
_DOC_KEYS = {"docid", "text", "title", "url"}
_DATASETS = {"musique", "browsecomp-plus", "multihop-rag", "bright"}
MUSIQUE_DOCUMENT_RENDERING = "title_lf_paragraph_v1"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetFormatError(f"{name} must be an object")
    return value


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise DatasetFormatError(f"{name} must be a {'possibly empty ' if empty else 'nonempty '}string")
    return value


def _identifier(value: Any, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise DatasetFormatError(f"{name} must be a string or integer id")
    return _text(str(value), name)


def _alias(row: Mapping[str, Any], keys: tuple[str, ...], *, required: bool = True) -> Any:
    found = [row[key] for key in keys if key in row]
    if not found:
        if required:
            raise DatasetFormatError(f"missing field: {'/'.join(keys)}")
        return None
    if any(value != found[0] for value in found[1:]):
        raise DatasetFormatError(f"conflicting aliases: {'/'.join(keys)}")
    return found[0]


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _strings(value: Any, name: str) -> list[str]:
    if not isinstance(value, list):
        raise DatasetFormatError(f"{name} must be a list")
    return [_text(item, name) for item in value]


def _documents(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise DatasetFormatError("documents must be a list")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in value:
        row = _mapping(item, "document")
        docid = _identifier(_alias(row, ("docid", "id")), "document id")
        if docid in seen:
            raise DatasetFormatError(f"duplicate or conflicting document id: {docid}")
        seen.add(docid)
        document = {"docid": docid, "text": _text(_alias(row, ("text", "content")), "document text")}
        for name in ("title", "url"):
            if name in row:
                document[name] = _text(row[name], name, empty=True)
        result.append(document)
    return result


def _task(dataset: str, qid: str, question: str, documents: list[dict[str, str]],
          corpus_ref: str | None, *, local: bool = False, excluded: list[str] | None = None) -> PublicTask:
    task: PublicTask = {
        "id": qid, "question_id": qid, "dataset": dataset, "question": question,
        "task_type": "retrieval" if dataset == "bright" else "qa",
        "documents": documents, "corpus_ref": corpus_ref,
        "corpus_scope": "question_local" if local else "shared",
        "excluded_docids": list(excluded or []),
    }
    validate_public_task(task)
    return task


def _reference(row: Mapping[str, Any], task: PublicTask, official_metric: str,
               answer_keys: tuple[str, ...] = ("answer",)) -> dict[str, Any]:
    answer = _alias(row, answer_keys, required=False)
    answers = [] if answer is None else [_text(answer, "answer", empty=True)]
    if "answer_aliases" in row:
        if answer is None:
            raise DatasetFormatError("answer_aliases without answer")
        answers.extend(_strings(row["answer_aliases"], "answer_aliases"))
    return {
        "question_id": task["question_id"], "dataset": task["dataset"],
        "answers": list(dict.fromkeys(answers)), "official_metric": official_metric,
        "rule_metrics_are_official": task["dataset"] == "musique",
        "reference_available": answer is not None,
    }


def adapt_musique(row: Mapping[str, Any]) -> tuple[PublicTask, dict[str, Any]]:
    """Adapt complete official per-question contexts (up to 20), including Full pairs.

    Public IDs include a hash of only the question and public context. Full pairs
    with the same source ID therefore cannot overwrite one another. Never pool
    these contexts: doing so can restore deliberately missing Full evidence.
    """
    row = _mapping(row, "MuSiQue row")
    source_id = _identifier(_alias(row, ("id", "question_id")), "question id")
    question = _text(row.get("question"), "question")
    paragraphs = row.get("paragraphs")
    if not isinstance(paragraphs, list) or not 1 <= len(paragraphs) <= 20:
        raise DatasetFormatError("MuSiQue requires 1 to 20 original local paragraphs")
    clean = []
    seen: set[int] = set()
    support: list[int] = []
    for item in paragraphs:
        paragraph = _mapping(item, "paragraph")
        idx = paragraph.get("idx")
        if type(idx) is not int or idx < 0 or idx in seen:
            raise DatasetFormatError("paragraph idx must be a unique nonnegative integer")
        seen.add(idx)
        title = _text(paragraph.get("title"), "title", empty=True)
        body = _text(paragraph.get("paragraph_text"), "paragraph_text")
        # Public title is part of the readable source, not a hidden annotation.
        # All search/read/quote offsets refer to this canonical text. No padding
        # or filtering: some official v1.0 examples have fewer than 20 paragraphs.
        clean.append({"idx": idx, "title": title,
                      "text": title + "\n" + body if title else body})
        if "is_supporting" in paragraph:
            if type(paragraph["is_supporting"]) is not bool:
                raise DatasetFormatError("is_supporting must be boolean")
            if paragraph["is_supporting"]:
                support.append(idx)
    if seen != set(range(len(paragraphs))):
        raise DatasetFormatError("MuSiQue paragraph indices must cover the original context")
    qid = f"musique:{source_id}:{_digest([question, clean])[:24]}"
    docs = [{"docid": f"{qid}/p/{p['idx']}", "text": p["text"], "title": p["title"]} for p in clean]
    task = _task("musique", qid, question, docs, None, local=True)
    reference = _reference(row, task, "answer_em_f1_support_f1_and_full_group_sufficiency")
    reference.update(source_question_id=source_id, pair_group_id=source_id,
                     supporting_docids=[f"{qid}/p/{idx}" for idx in support])
    if "answerable" in row:
        if type(row["answerable"]) is not bool:
            raise DatasetFormatError("answerable must be boolean")
        reference["answerable"] = row["answerable"]
    # Diagnostic annotations remain in the private scoring process only.
    if "question_decomposition" in row:
        if not isinstance(row["question_decomposition"], list):
            raise DatasetFormatError("question_decomposition must be a list")
        reference["question_decomposition"] = deepcopy(row["question_decomposition"])
    return task, reference


def adapt_browsecomp(row: Mapping[str, Any], corpus_ref: str) -> tuple[PublicTask, dict[str, Any]]:
    """Accept already-decoded query rows; never decrypt or load benchmark files."""
    row = _mapping(row, "BrowseComp-Plus row")
    qid = _identifier(_alias(row, ("query_id", "question_id", "id")), "query id")
    question = _text(_alias(row, ("query", "question")), "query")
    task = _task("browsecomp-plus", qid, question, [], _text(corpus_ref, "corpus_ref"))
    reference = _reference(row, task, "llm_judge")
    for key in ("evidence", "gold", "evidence_docids", "gold_docids", "evidence_docs", "gold_docs"):
        if key in row:
            reference[key] = deepcopy(row[key])
    return task, reference


def adapt_multihop(row: Mapping[str, Any]) -> tuple[PublicTask, dict[str, Any]]:
    """Use an explicit full-corpus reference or documents, never evidence_list.

    Original rows have no stable ID or full corpus. In that case the ID is a
    public-query hash and the caller must resolve 'multihop-rag:corpus'.
    """
    row = _mapping(row, "MultiHop-RAG row")
    question = _text(_alias(row, ("query", "question")), "query")
    raw_id = _alias(row, ("qid", "question_id", "id"), required=False)
    qid = _identifier(raw_id, "question id") if raw_id is not None else f"multihop:{_digest(question)}"
    docs = _documents(row.get("documents", []))
    corpus_ref = row.get("corpus_ref", None if docs else "multihop-rag:corpus")
    if corpus_ref is not None:
        corpus_ref = _text(corpus_ref, "corpus_ref")
    task = _task("multihop-rag", qid, question, docs, corpus_ref)
    reference = _reference(row, task, "upstream_qa_script_not_normalized_em_f1", ("answer", "gold"))
    if "evidence_list" in row:
        if not isinstance(row["evidence_list"], list):
            raise DatasetFormatError("evidence_list must be a list")
        reference["evidence_list"] = deepcopy(row["evidence_list"])
    if "question_type" in row:
        reference["question_type"] = _text(row["question_type"], "question_type")
    return task, reference


def adapt_bright(row: Mapping[str, Any], corpus_ref: str) -> tuple[PublicTask, dict[str, Any]]:
    """BRIGHT is retrieval-only here; excluded IDs are enforced public filters."""
    row = _mapping(row, "BRIGHT row")
    qid = _identifier(_alias(row, ("id", "question_id")), "query id")
    question = _text(row.get("query"), "query")
    excluded = _strings(row.get("excluded_ids"), "excluded_ids")
    if len(set(excluded)) != len(excluded):
        raise DatasetFormatError("duplicate excluded_ids")
    task = _task("bright", qid, question, [], _text(corpus_ref, "corpus_ref"), excluded=excluded)
    reference = _reference(row, task, "retrieval_ndcg_at_10", ("gold_answer",))
    for key in ("gold_ids", "gold_ids_long"):
        if key in row:
            reference[key] = _strings(row[key], key)
    if "reasoning" in row:
        reference["reasoning"] = _text(row["reasoning"], "reasoning", empty=True)
    return task, reference


def validate_public_task(task: Mapping[str, Any]) -> None:
    """Reject extra fields instead of trusting a name-based private-key denylist."""
    task = _mapping(task, "public task")
    if set(task) != _PUBLIC_KEYS:
        raise DatasetFormatError(f"public task keys differ: {sorted(set(task) ^ _PUBLIC_KEYS)}")
    if not isinstance(task["dataset"], str) or task["dataset"] not in _DATASETS:
        raise DatasetFormatError("unknown dataset")
    qid = _text(task["question_id"], "question_id")
    if task["id"] != qid:
        raise DatasetFormatError("id and question_id conflict")
    _text(task["question"], "question")
    expected_type = "retrieval" if task["dataset"] == "bright" else "qa"
    if task["task_type"] != expected_type:
        raise DatasetFormatError("invalid task_type for dataset")
    docs = _documents(task["documents"])
    for doc in task["documents"]:
        if set(doc) - _DOC_KEYS:
            raise DatasetFormatError("private or unknown document metadata in public task")
    excluded = _strings(task["excluded_docids"], "excluded_docids")
    if len(set(excluded)) != len(excluded):
        raise DatasetFormatError("duplicate excluded_docids")
    if set(excluded) & {doc["docid"] for doc in docs}:
        raise DatasetFormatError("excluded documents present in public task")
    ref = task["corpus_ref"]
    if ref is not None:
        _text(ref, "corpus_ref")
    if task["dataset"] == "musique":
        if task["corpus_scope"] != "question_local" or ref is not None or not 1 <= len(docs) <= 20:
            raise DatasetFormatError("MuSiQue must retain its original 1 to 20 question-local documents")
        if any(not doc["docid"].startswith(f"{qid}/p/") for doc in docs):
            raise DatasetFormatError("MuSiQue document belongs to another question scope")
    elif task["corpus_scope"] != "shared" or (not docs and ref is None):
        raise DatasetFormatError("shared task requires documents or a corpus reference")


def validate_task_collection(tasks: list[Mapping[str, Any]]) -> None:
    """Run before evaluation; duplicate IDs cannot silently overwrite a result."""
    if not isinstance(tasks, list):
        raise DatasetFormatError("tasks must be a list")
    seen: dict[str, str] = {}
    shared_docs: dict[tuple[str, str, str], str] = {}
    for task in tasks:
        validate_public_task(task)
        qid = task["question_id"]
        digest = _digest(task)
        if qid in seen:
            kind = "duplicate" if seen[qid] == digest else "conflicting"
            raise DatasetFormatError(f"{kind} question_id: {qid}")
        seen[qid] = digest
        if task["corpus_scope"] == "shared" and task["corpus_ref"] is not None:
            for document in task["documents"]:
                key = (task["dataset"], task["corpus_ref"], document["docid"])
                doc_digest = _digest(document)
                if key in shared_docs and shared_docs[key] != doc_digest:
                    raise DatasetFormatError(f"conflicting shared document id: {document['docid']}")
                shared_docs[key] = doc_digest


def filter_documents(task: Mapping[str, Any], documents: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Reference-free guard to apply before ranking (and defensively on results).

    For MuSiQue, foreign/mutated paragraphs are rejected, not silently pooled.
    For BRIGHT, excluded documents are removed before exposing them to a model.
    """
    validate_public_task(task)
    docs = _documents(documents)
    if task["corpus_scope"] == "question_local":
        allowed = {doc["docid"]: doc for doc in task["documents"]}
        for doc in docs:
            if allowed.get(doc["docid"]) != doc:
                raise DatasetFormatError("foreign or changed document in question-local corpus")
    excluded = set(task["excluded_docids"])
    return [doc for doc in docs if doc["docid"] not in excluded]


def normalize_answer(answer: str) -> str:
    """SQuAD/MuSiQue-style lowercase, ASCII punctuation, articles, whitespace."""
    answer = _text(answer, "answer", empty=True).lower()
    answer = "".join(char for char in answer if char not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", answer).split())


def evaluate_answer(prediction: str, reference: Mapping[str, Any], metric: str = "em") -> float:
    """Return local rule EM/F1; this is NOT the BrowseComp official judge.

    BRIGHT needs retrieval metrics. MuSiQue-Full unanswerable rows need paired
    sufficiency scoring; neither is silently mapped onto ordinary QA accuracy.
    The controller, not this helper, must count incomplete delivery as failure.
    """
    if metric not in {"em", "f1"}:
        raise DatasetFormatError("supported rule metrics are em and f1; no official judge is implemented")
    prediction = _text(prediction, "prediction", empty=True)
    reference = _mapping(reference, "private reference")
    if not isinstance(reference.get("dataset"), str) or reference["dataset"] not in _DATASETS:
        raise DatasetFormatError("unknown reference dataset")
    if reference["dataset"] == "bright":
        raise DatasetFormatError("BRIGHT is a retrieval task, not main QA")
    if reference.get("answerable") is False:
        raise DatasetFormatError("unanswerable MuSiQue-Full rows require paired sufficiency evaluation")
    answers = reference.get("answers")
    if not isinstance(answers, list) or not answers:
        raise DatasetFormatError("no answer reference available")
    golds = [normalize_answer(_text(answer, "reference answer", empty=True)) for answer in answers]
    predicted = normalize_answer(prediction)
    if metric == "em":
        return float(any(predicted == gold for gold in golds))
    pred_tokens = predicted.split()
    scores = []
    for gold in golds:
        gold_tokens = gold.split()
        if not pred_tokens or not gold_tokens:
            scores.append(float(pred_tokens == gold_tokens))
            continue
        overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
        scores.append(2.0 * overlap / (len(pred_tokens) + len(gold_tokens)))
    return max(scores)