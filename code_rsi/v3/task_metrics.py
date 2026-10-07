"""Pure, host-only task metrics; no network, file loading, or reference exposure.

MuSiQue rules follow the upstream v1.0 evaluator at revision 922ac98f19a20199.
See docs/v3_task_metrics.md for missing-data and Full-pair semantics.
"""
from __future__ import annotations

from collections import defaultdict
from math import isfinite, log2
from statistics import fmean
from typing import Any, Mapping

from .datasets import evaluate_answer, validate_public_task

MUSIQUE_REVISION = "922ac98f19a201998dbdae6d7f2887a5258dbdeb"
MUSIQUE_PROTOCOL = "musique-v1.0-rules-" + MUSIQUE_REVISION[:12]
_FIELDS = ("answer_em", "answer_f1", "support_em", "support_f1", "answerability")


class TaskMetricError(ValueError):
    """An invalid metric contract; never silently interpret it as a wrong answer."""


def _object(value, name):
    if not isinstance(value, Mapping):
        raise TaskMetricError(name + " must be an object")
    return value


def _ids(value, name):
    if not isinstance(value, (list, tuple)) or any(not isinstance(x, str) or not x for x in value):
        raise TaskMetricError(name + " must be a sequence of nonempty string IDs")
    return list(value)


def support_set_metrics(predicted, gold):
    """Paragraph-set EM/F1 after ID mapping, including upstream empty/empty=1.

    Duplicates have no weight. This helper needs explicit sets, not absent fields.
    The caller must establish annotation availability before using it.
    """
    pred, expected = set(_ids(predicted, "predicted support")), set(_ids(gold, "gold support"))
    denominator = len(pred) + len(expected)
    return {"support_em": float(pred == expected),
            "support_f1": 2 * len(pred & expected) / denominator if denominator else 1.0}


def _prediction_support(prediction, task):
    forms = []
    allowed = {d["docid"] for d in task["documents"]} if task is not None else None
    if "support_docids" in prediction:
        forms.append(set(_ids(prediction["support_docids"], "support_docids")))
    if "predicted_support_idxs" in prediction:
        values = prediction["predicted_support_idxs"]
        if not isinstance(values, list) or any(type(x) is not int for x in values):
            raise TaskMetricError("predicted_support_idxs must be a list of integers")
        if task is None:
            return None, "unavailable_task_mapping"
        forms.append({task["question_id"] + "/p/" + str(i) for i in values})
    if "citations" in prediction:
        citations = prediction["citations"]
        if not isinstance(citations, list):
            raise TaskMetricError("citations must be a list")
        if citations and task is None:
            return None, "unavailable_task_mapping"
        if any(not isinstance(c, Mapping) or not isinstance(c.get("docid"), str) or not c["docid"]
               for c in citations):
            return None, "unavailable_citation_mapping"
        forms.append({c["docid"] for c in citations})
    if not forms:
        return [], "missing_prediction"
    # Explicit official indices can identify an incorrect paragraph; they remain
    # false positives under the official set metric. Unmapped citation/doc IDs
    # cannot be silently dropped or assigned to the right paragraph.
    direct = set(prediction.get("support_docids", []))
    cited = {c["docid"] for c in prediction.get("citations", [])}
    if allowed is not None and (direct | cited) - allowed:
        return None, "unavailable_citation_mapping"
    if any(form != forms[0] for form in forms[1:]):
        raise TaskMetricError("conflicting support prediction representations")
    return sorted(forms[0]), "ok"


def score_task(prediction, reference, *, task=None, allow_proxy_metrics=False):
    """Score one completed prediction against a private adapted reference.

    Returns answer_em/answer_f1/support_em/support_f1/answerability plus explicit
    status/protocol metadata. None means unavailable/not applicable, never 1.
    Full aggregation is separate; no answerability is inferred from answer text,
    citation presence, or abstention. Unknown provider outcomes must not be sent
    here: reconcile them in the host before entering the scoring phase.
    """
    reference = _object(reference, "reference")
    prediction = {"answer": prediction} if isinstance(prediction, str) else _object(prediction, "prediction")
    if type(allow_proxy_metrics) is not bool:
        raise TaskMetricError("allow_proxy_metrics must be boolean")
    dataset = reference.get("dataset")
    if dataset not in {"musique", "browsecomp-plus", "multihop-rag", "bright"}:
        raise TaskMetricError("unsupported reference dataset")
    qid = reference.get("question_id")
    if not isinstance(qid, str) or not qid:
        raise TaskMetricError("reference question_id required")
    if task is not None:
        validate_public_task(task)
        if (task["question_id"], task["dataset"]) != (qid, dataset):
            raise TaskMetricError("task/reference identity mismatch")
    if "question_id" in prediction and prediction["question_id"] != qid:
        raise TaskMetricError("prediction/reference identity mismatch")
    statuses = {name: "not_applicable" for name in _FIELDS}
    result = {name: None for name in _FIELDS}
    result.update(question_id=qid, dataset=dataset, metric_status=statuses,
                  protocol=MUSIQUE_PROTOCOL if dataset == "musique" else "local-rule-diagnostics-v1",
                  proxy_metrics=dataset in {"browsecomp-plus", "multihop-rag"}, official_judge=False,
                  pair_group_id=reference.get("pair_group_id", reference.get("source_question_id")),
                  gold_answerable=None, predicted_answerable=None)
    if dataset == "bright":
        statuses.update({name: "not_applicable_retrieval_task" for name in _FIELDS})
        return result
    if dataset == "browsecomp-plus" and not allow_proxy_metrics:
        statuses.update(answer_em="unavailable_official_judge", answer_f1="unavailable_official_judge")
        result["protocol"] = "browsecomp-plus-official-judge-required"
        return result

    gold_answerable = reference.get("answerable") if dataset == "musique" else None
    if gold_answerable is not None and type(gold_answerable) is not bool:
        raise TaskMetricError("reference answerable must be boolean")
    if dataset == "musique":
        result["gold_answerable"] = gold_answerable
        predicted_answerable = prediction.get("predicted_answerable")
        if predicted_answerable is not None and type(predicted_answerable) is not bool:
            raise TaskMetricError("predicted_answerable must be boolean")
        result["predicted_answerable"] = predicted_answerable
        if gold_answerable is None:
            statuses["answerability"] = "unavailable_reference"
        elif predicted_answerable is None:
            statuses["answerability"] = "unavailable_prediction"
        else:
            result["answerability"] = float(predicted_answerable == gold_answerable)
            statuses["answerability"] = "ok"
        if gold_answerable is False:
            for name in _FIELDS[:4]:
                statuses[name] = "not_applicable_unanswerable_branch"
            return result

    answers = reference.get("answers")
    if answers is not None and (not isinstance(answers, list) or any(not isinstance(a, str) for a in answers)):
        raise TaskMetricError("reference answers must be a string list")
    answer_keys = [k for k in ("answer", "predicted_answer") if k in prediction]
    if any(not isinstance(prediction[k], str) for k in answer_keys):
        raise TaskMetricError("predicted answer must be text")
    if len(answer_keys) == 2 and prediction[answer_keys[0]] != prediction[answer_keys[1]]:
        raise TaskMetricError("conflicting answer prediction representations")
    if "answer_usable" in prediction and type(prediction["answer_usable"]) is not bool:
        raise TaskMetricError("answer_usable must be boolean")
    if not answers or reference.get("reference_available") is False:
        statuses.update(answer_em="unavailable_reference", answer_f1="unavailable_reference")
    elif not answer_keys or prediction.get("answer_usable") is False:
        result.update(answer_em=0.0, answer_f1=0.0)
        reason = "missing_prediction" if not answer_keys else "host_delivery_failure"
        statuses.update(answer_em=reason, answer_f1=reason)
    else:
        for metric in ("em", "f1"):
            result["answer_" + metric] = evaluate_answer(prediction[answer_keys[0]], reference, metric)
            statuses["answer_" + metric] = "ok" if dataset == "musique" else "proxy_rule_only"

    if dataset == "musique":
        gold = reference.get("supporting_docids")
        annotated = reference.get("support_annotation_available")
        if annotated is not None and type(annotated) is not bool:
            raise TaskMetricError("support_annotation_available must be boolean")
        reason = None
        if gold is None or annotated is False:
            reason = "unavailable_reference"
        else:
            gold = _ids(gold, "supporting_docids")
            if not gold and annotated is not True:
                reason = "unavailable_empty_annotation_ambiguous"
        if reason is not None:
            statuses.update(support_em=reason, support_f1=reason)
        else:
            if task is not None and set(gold) - {d["docid"] for d in task["documents"]}:
                raise TaskMetricError("gold support lies outside the question-local task")
            support, status = _prediction_support(prediction, task)
            if support is None:
                statuses.update(support_em=status, support_f1=status)
            elif status == "missing_prediction":
                # Missing output is not an explicitly predicted empty support set.
                result.update(support_em=0.0, support_f1=0.0)
                statuses.update(support_em=status, support_f1=status)
            else:
                result.update(support_set_metrics(support, gold))
                statuses.update(support_em="ok", support_f1="ok")
    return result


def _unit_score(value, name):
    if value is None:
        return None
    if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1:
        raise TaskMetricError(name + " must be None or a finite unit score")
    return float(value)


def aggregate_musique_full(results):
    """Official two-instance pair gating; do not pool paragraph contexts.

    Every group needs distinct public IDs and exactly one true/false gold label.
    Missing metrics propagate to None, without dropping rows from denominators.
    Returned primary metrics are rounded to 3 decimals like evaluate_v1.0.py.
    support_em and answerability/group EM are additional diagnostics.
    """
    if not isinstance(results, (list, tuple)) or not results:
        raise TaskMetricError("nonempty Full results required")
    groups = defaultdict(list)
    seen = set()
    for row in results:
        _object(row, "Full result")
        if row.get("dataset") != "musique" or row.get("protocol") != MUSIQUE_PROTOCOL:
            raise TaskMetricError("Full aggregation requires MuSiQue scoring results")
        qid, group = row.get("question_id"), row.get("pair_group_id")
        if not isinstance(qid, str) or not qid or qid in seen:
            raise TaskMetricError("Full public question IDs must be unique")
        if not isinstance(group, str) or not group:
            raise TaskMetricError("private pair_group_id required")
        if type(row.get("gold_answerable")) is not bool:
            raise TaskMetricError("Full requires explicit gold answerable")
        if row.get("predicted_answerable") is not None and type(row["predicted_answerable"]) is not bool:
            raise TaskMetricError("Full predicted_answerable must be boolean or absent")
        for name in _FIELDS:
            _unit_score(row.get(name), name)
        expected = (float(row["predicted_answerable"] == row["gold_answerable"])
                    if row.get("predicted_answerable") is not None else None)
        if row.get("answerability") != expected:
            raise TaskMetricError("answerability score conflicts with labels")
        groups[group].append(row)
        seen.add(qid)
    pairs = []
    for group, rows in groups.items():
        if len(rows) != 2 or sorted(r["gold_answerable"] for r in rows) != [False, True]:
            raise TaskMetricError("official Full requires exactly one answerable/unanswerable pair per group")
        answerable = next(r for r in rows if r["gold_answerable"])
        gate = (float(all(r["answerability"] == 1.0 for r in rows))
                if all(r["answerability"] is not None for r in rows) else None)
        pairs.append((answerable, gate))
    def mean_complete(values):
        return round(fmean(values), 3) if all(v is not None for v in values) else None
    out = {"dataset": "musique-full", "protocol": MUSIQUE_PROTOCOL, "n_instances": len(results),
           "n_groups": len(pairs), "n_answerable": len(pairs), "independent_unit": "question_pair",
           "answerability": mean_complete([r["answerability"] for r in results]),
           "group_sufficiency": mean_complete([gate for _, gate in pairs]), "metric_status": {}}
    for name in _FIELDS[:4]:
        out[name] = mean_complete([r[name] for r, _ in pairs])
        group_name = "group_" + name.split("_")[0] + "_sufficiency_" + name.split("_")[1]
        out[group_name] = mean_complete([r[name] * gate if r[name] is not None and gate is not None else None
                                         for r, gate in pairs])
    for name, value in out.items():
        if name in _FIELDS or name.startswith("group_"):
            out["metric_status"][name] = "ok" if value is not None else "unavailable_incomplete_metrics"
    return out


def score_bright(ranked_docids, reference, *, excluded_docids, long_context=False):
    """Binary nDCG@10 for an explicitly ordered, already filtered retrieval list.

    Not a QA metric or a replacement for the full upstream pytrec_eval report.
    Ties must have been resolved by the caller's frozen ranking protocol.
    """
    reference = _object(reference, "reference")
    if reference.get("dataset") != "bright" or type(long_context) is not bool:
        raise TaskMetricError("BRIGHT reference and boolean long_context required")
    ranked = _ids(ranked_docids, "ranked_docids")
    excluded = set(_ids(excluded_docids, "excluded_docids"))
    key = "gold_ids_long" if long_context else "gold_ids"
    if key not in reference:
        return {"ndcg_at_10": None, "status": "unavailable_reference", "protocol": "bright-binary-ndcg10-v1"}
    gold = set(_ids(reference[key], key))
    if len(ranked) != len(set(ranked)):
        raise TaskMetricError("ranking must not contain duplicate document IDs")
    if excluded & (set(ranked) | gold):
        raise TaskMetricError("excluded IDs must be absent from ranking and gold")
    if not gold:
        return {"ndcg_at_10": None, "status": "unavailable_empty_qrels", "protocol": "bright-binary-ndcg10-v1"}
    gain = sum(1 / log2(rank + 2) for rank, docid in enumerate(ranked[:10]) if docid in gold)
    ideal = sum(1 / log2(rank + 2) for rank in range(min(10, len(gold))))
    return {"ndcg_at_10": gain / ideal, "status": "ok", "protocol": "bright-binary-ndcg10-v1"}
