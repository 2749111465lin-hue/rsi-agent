"""Offline document-set recall from ID-only labels and trusted observations.

No files, credentials, reference text, model calls, or network are accessed.
The caller must adapt host receipts: ``verified_quotes`` means mechanically
source-validated quotes, not semantically sufficient evidence. Qrels may be
incomplete, so documents outside them are not judged wrong; precision, F1 and
answer correctness are intentionally absent.

Each layer is the union over the WHOLE trajectory. A single query's top five is
not the same as this across-query union, and neither permits inventing recall@10
from only five observed results. No ranking, cutoff or success-subset strategy
aggregation is produced. Missing/incomplete observations stay unknown, not zero.
"""
from __future__ import annotations

from collections.abc import Mapping

SCHEMA = "rag-rsi-document-recall-audit-1"
LAYERS = ("retrieved", "presented", "verified_quotes")
LABELS = ("evidence", "gold")


class RetrievalAuditError(ValueError):
    """Invalid identity or observation contract; never an incorrect answer."""


def _object(value, name):
    if not isinstance(value, Mapping):
        raise RetrievalAuditError(name + " must be an object")
    return value


def _id(value, name):
    if not isinstance(value, str) or not value.strip():
        raise RetrievalAuditError(name + " must be a nonempty string")
    return value


def _docids(value, name):
    if value is None:
        return None
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise RetrievalAuditError(name + " must be a collection of string IDs or null")
    return {_id(item, name) for item in value}


def _complete(value, name):
    if value is not None and type(value) is not bool:
        raise RetrievalAuditError(name + " must be boolean or null")
    return value


def audit_document_recall(question_id, annotations, trajectories):
    """Audit a specified question without selecting or averaging eligible rows.

    ``annotations`` binds ``question_id`` and optional ``evidence_docids`` and
    ``gold_docids``. Missing/null/empty annotations have no recall denominator.
    Each trajectory binds ``trajectory_id``, ``question_id``, ``complete`` and
    ``layers``. Each retrieved/presented/verified_quotes layer contains
    ``complete`` and ``docids``. Completeness must be explicitly True both for
    the whole trajectory and for that layer; omitted/null flags are unknown.

    Complete empty observations against nonempty labels yield real zero recall.
    Incomplete observations may retain a known observed count but never acquire
    a full-trajectory hit count, recall, any_hit or all_hit. Different label sets
    are assessed independently. Counts describe documents, not facts or claims.
    """
    _id(question_id, "question_id")
    labels = _object(annotations, "annotations")
    if labels.get("question_id") != question_id:
        raise RetrievalAuditError("annotation question_id mismatch")
    if not isinstance(trajectories, (list, tuple)):
        raise RetrievalAuditError("trajectories must be a sequence")
    targets, label_status = {}, {}
    for name in LABELS:
        ids = _docids(labels.get(name + "_docids"), name + "_docids")
        reason = "missing_annotation" if ids is None else "empty_annotation" if not ids else None
        targets[name] = ids
        label_status[name] = {"status": "unknown" if reason else "known", "reason": reason,
                              "document_count": len(ids) if ids is not None else None}
    rows, seen = [], set()
    for raw in trajectories:
        row = _object(raw, "trajectory")
        tid = _id(row.get("trajectory_id"), "trajectory_id")
        if tid in seen:
            raise RetrievalAuditError("duplicate trajectory_id")
        seen.add(tid)
        if row.get("question_id") != question_id:
            raise RetrievalAuditError("trajectory question_id mismatch")
        whole = _complete(row.get("complete"), "trajectory complete")
        raw_layers = _object(row.get("layers", {}), "layers")
        out = {"trajectory_id": tid, "question_id": question_id,
               "trajectory_complete": whole, "layers": {}}
        for layer_name in LAYERS:
            layer = _object(raw_layers.get(layer_name, {}), layer_name)
            complete = _complete(layer.get("complete"), layer_name + " complete")
            observed = _docids(layer.get("docids"), layer_name + " docids")
            reasons = []
            if whole is not True:
                reasons.append("trajectory_incomplete" if whole is False else "trajectory_completeness_unknown")
            if complete is not True:
                reasons.append("layer_incomplete" if complete is False else "layer_completeness_unknown")
            if observed is None:
                reasons.append("missing_layer_docids")
            measurement = {"observable": not reasons, "status": "unknown" if reasons else "observed",
                           "reasons": reasons, "observed_doc_count": len(observed) if observed is not None else None}
            for name in LABELS:
                target = targets[name]
                unavailable = reasons + ([label_status[name]["reason"]] if label_status[name]["reason"] else [])
                eligible = not unavailable
                hits = len(observed & target) if eligible else None
                measurement[name] = {"eligible": eligible, "status": "ok" if eligible else "unknown",
                                     "reasons": unavailable, "hit_count": hits,
                                     "document_recall": hits / len(target) if eligible else None,
                                     "any_hit": hits > 0 if eligible else None,
                                     "all_hit": hits == len(target) if eligible else None}
            out["layers"][layer_name] = measurement
        rows.append(out)
    return {"schema": SCHEMA, "question_id": question_id, "scope": "whole_trajectory_document_union",
            "annotation_status": label_status, "rows": rows}
