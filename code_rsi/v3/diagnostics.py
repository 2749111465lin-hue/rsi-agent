"""Deterministic RAG diagnostics from existing host receipts; no model calls.

These are structural observations and explicitly labelled model reports, not a
semantic verifier or reward. Only complete, identity-matched D_fit measurements
may enter compact_feedback. See docs/v3_feedback_contract.md.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
import hashlib
import json
import math

MODULES = ("query_rewrite", "retrieval", "evidence_selection", "answer_generation")
# Design priors in [0, 1], NOT learned probabilities, causal labels, or rewards.
FAILURE_MODULE_PRIORS = {
    "empty_retrieval": {"query_rewrite": 1.0, "retrieval": 0.8},
    "no_retrieval": {"retrieval": 1.0, "query_rewrite": 0.5},
    "no_evidence_read": {"evidence_selection": 1.0},
    "no_verified_read_quotes": {"evidence_selection": 0.9, "query_rewrite": 0.3},
    "no_final_evidence": {"evidence_selection": 1.0, "answer_generation": 0.8},
    "no_observed_final_answer": {"answer_generation": 1.0},
    "answer_empty": {"answer_generation": 1.0},
    "invalid_answer_citation": {"answer_generation": 1.0, "evidence_selection": 0.5},
    "model_parse_failure": {"evidence_selection": 0.6, "answer_generation": 0.6,
                            "query_rewrite": 0.3},
    "repeated_query": {"query_rewrite": 1.0, "retrieval": 0.3},
}
MODEL_FAILURE_MODULE_PRIORS = {
    "evidence_gap": {"query_rewrite": 1.0, "retrieval": 0.8, "evidence_selection": 0.5},
    "evidence_conflict": {"evidence_selection": 1.0, "query_rewrite": 0.7,
                          "answer_generation": 0.6},
    "evidence_insufficient": {"query_rewrite": 0.8, "evidence_selection": 0.8},
    "plan_schema_failure": {"query_rewrite": 1.0},
    "read_schema_failure": {"evidence_selection": 1.0},
    "answer_schema_failure": {"answer_generation": 1.0},
    "invalid_quote": {"evidence_selection": 1.0},
    "ungrounded_bridge_entity": {"query_rewrite": 1.0},
    "no_evidence_progress": {"query_rewrite": 1.0, "retrieval": 0.6},
    "no_followup_queries": {"query_rewrite": 1.0},
    "repeated_queries": {"query_rewrite": 1.0},
    "model_claims_support_despite_conflict": {"evidence_selection": 1.0,
                                             "answer_generation": 0.8},
}
_UNAVAILABLE = frozenset({
    "unknownprovideroutcome", "unknownproviderresult", "unknownapioutcome",
    "unknownoutcome", "pending", "incomplete", "hosterror", "hostfailure",
    "budgetblocked", "limitexceeded", "providererror", "measurementunavailable",
})


def _map(value):
    return value if isinstance(value, Mapping) else {}


def _items(value):
    return list(value) if isinstance(value, (list, tuple)) else []


def _text(value, limit=240):
    return value[:limit] if isinstance(value, str) else ""


def _number(value):
    return float(value) if (type(value) in (int, float) and math.isfinite(value)) else None


def _digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                     allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _codes(value):
    if isinstance(value, Mapping):
        return sorted(k for k, v in value.items() if isinstance(k, str) and v)
    return sorted(set(x for x in _items(value) if isinstance(x, str)))


def _role_ok(obj, *, required=False):
    role = obj.get("role")
    if (required or role is not None) and role not in ("D_fit", "fit"):
        raise ValueError("learning feedback requires D_fit")
    if "split" in obj and obj["split"] not in ("D_fit", "fit"):
        raise ValueError("learning feedback split conflicts with D_fit")


def _unavailable(obj):
    """Only explicit host outcome fields; never infer an API result from silence."""
    for key in ("status", "measurement_status", "error_type", "provider_outcome"):
        value = obj.get(key)
        token = "".join(c for c in str(value).lower() if c.isalnum())
        if token in _UNAVAILABLE or (key == "provider_outcome" and token == "unknown"):
            return "/" + key
    for key in ("failure_classes", "model_errors"):
        if any("".join(c for c in v.lower() if c.isalnum()) in _UNAVAILABLE
               for v in _codes(obj.get(key))):
            return "/" + key
    if obj.get("complete") is False:
        return "/complete"
    return None


def _rank(priors):
    return sorted((m for m in MODULES if priors.get(m, 0) > 0),
                  key=lambda m: (-priors[m], MODULES.index(m)))


def diagnose_execution(receipt):
    """Return host facts, model/candidate reports, bounded module priors and refs.

    Input must be a host-owned execution receipt. A bare RagEngine result belongs
    in candidate_reported, never in host fields. Missing observations stay unknown.
    Explicit select/report roles are rejected, including for standalone use.
    """
    if not isinstance(receipt, Mapping):
        raise TypeError("execution receipt must be a mapping")
    _role_ok(receipt)
    unavailable = _unavailable(receipt)
    if unavailable:
        return {"host_observed": ["measurement_unavailable"], "model_reported": [],
                "suggested_modules": [], "module_priors": {}, "model_details": {},
                "evidence_refs": {"host:measurement_unavailable": [unavailable]},
                "measurement_status": "unavailable", "observations": {}}

    host, model, refs, details, priors = set(), set(), defaultdict(list), {}, {}

    def add(code, path, *, reported=False):
        (model if reported else host).add(code)
        key = ("model:" if reported else "host:") + code
        if path not in refs[key] and len(refs[key]) < 4:
            refs[key].append(path)
        weights = (MODEL_FAILURE_MODULE_PRIORS if reported else
                   FAILURE_MODULE_PRIORS).get(code, {})
        for module, weight in weights.items():
            priors[module] = max(priors.get(module, 0), weight * (0.35 if reported else 1))

    trace = _items(receipt.get("trace"))
    evidence = _map(receipt.get("host_evidence_trace"))
    reads = _items(evidence.get("read_presentations"))
    finals = _items(evidence.get("final_observations"))
    schema = receipt.get("schema", "")
    trace_complete = (isinstance(schema, str) and schema.startswith("rag-rsi-v3-execution-")
                      and isinstance(receipt.get("trace"), list)
                      and isinstance(receipt.get("host_evidence_trace"), Mapping)
                      and type(receipt.get("execution_ok")) is bool)
    searches, read_calls, answer_calls, empty_searches, seen_queries = 0, 0, 0, 0, set()
    error_hashes = {_digest({"_meta": {"truncated": t, "finish_reason": "error"}})
                    for t in (True, False)}
    for i, item in enumerate(trace):
        event = _map(item)
        request = _map(event.get("request"))
        path = "/trace/" + str(i)
        if event.get("name") == "search":
            searches += 1
            if event.get("response_hash") == _digest([]):
                empty_searches += 1
                add("empty_retrieval", path + "/response_hash")
            query = request.get("query")
            if isinstance(query, str) and query.strip():
                key = " ".join(query.casefold().split())
                if key in seen_queries:
                    add("repeated_query", path + "/request/query")
                seen_queries.add(key)
        elif event.get("name") == "complete":
            stage = request.get("stage")
            read_calls += int(stage == "read")
            answer_calls += int(stage == "answer")
            if event.get("response_hash") in error_hashes:
                add("model_parse_failure", path + "/response_hash")
                module = {"plan": "query_rewrite", "read": "evidence_selection",
                          "answer": "answer_generation"}.get(stage)
                if module:
                    priors[module] = 1.0

    if receipt.get("execution_ok") is False:
        add("execution_error", "/execution_ok")
    if receipt.get("answer_usable") is False and receipt.get("execution_ok") is True:
        add("answer_empty", "/answer_usable")
    for code in _codes(receipt.get("failure_classes")):
        # Only the host's own failure field: candidate_reported is handled below.
        if code in FAILURE_MODULE_PRIORS or code in (
                "execution_error", "wall_timeout", "sandbox_failure", "isolation_failure",
                "CandidateEvidenceError"):
            add(code, "/failure_classes")
    parse_errors = {"ModelResponseError", "JSONDecodeError", "ResponseSchemaError", "OutputParserError"}
    if set(_codes(receipt.get("model_errors"))) & parse_errors:
        add("model_parse_failure", "/model_errors")
    if trace_complete and not searches:
        add("no_retrieval", "/trace")
    if trace_complete and not reads:
        add("no_evidence_read", "/host_evidence_trace/read_presentations")
    valid_read_quotes = sum(len(_items(_map(item).get("verified_quotes"))) for item in reads)
    if reads and not valid_read_quotes:
        add("no_verified_read_quotes", "/host_evidence_trace/read_presentations")

    citation_check = _map(receipt.get("host_citation_validation"))
    status = citation_check.get("status", receipt.get("citation_status"))
    if status == "no_observed_final_answer":
        add("no_observed_final_answer", "/host_citation_validation/status")
    elif status in ("missing_citations", "invalid_model_citation_ids", "candidate_citation_mismatch"):
        add("invalid_answer_citation", "/host_citation_validation/status")
    if trace_complete and not answer_calls and not finals:
        add("no_observed_final_answer", "/trace")

    # Use the final observation whose answer the host matched, not an abandoned call.
    selected = None
    for i, item in enumerate(finals):
        item = _map(item)
        response = _map(item.get("response"))
        answer, returned = response.get("answer"), receipt.get("answer")
        if isinstance(answer, str) and isinstance(returned, str) and answer.strip() == returned.strip():
            selected = (i, item)
    final_count = None
    if selected is not None:
        i, item = selected
        presented = item.get("evidence")
        if isinstance(presented, (Mapping, list, tuple)):
            final_count = len(presented)
            if not final_count:
                add("no_final_evidence", "/host_evidence_trace/final_observations/" + str(i) + "/evidence")
        if _map(item.get("response")).get("evidence_sufficient") is False:
            add("evidence_insufficient",
                "/host_evidence_trace/final_observations/" + str(i) + "/response/evidence_sufficient",
                reported=True)
    elif (status not in (None, "no_observed_final_answer", "execution_failed")
          and isinstance(citation_check.get("presented_citation_ids"), list)):
        final_count = len(citation_check["presented_citation_ids"])
        if not final_count:
            add("no_final_evidence", "/host_citation_validation/presented_citation_ids")
    if citation_check.get("model_claims_evidence") is False:
        add("evidence_insufficient", "/host_citation_validation/model_claims_evidence", reported=True)

    reported = _map(receipt.get("candidate_reported"))
    state = _map(reported.get("state"))
    for field, code in (("gaps", "evidence_gap"), ("conflicts", "evidence_conflict")):
        values = sorted(set(_text(x) for x in _items(state.get(field))
                            if isinstance(x, str) and x.strip()))
        if values:
            add(code, "/candidate_reported/state/" + field, reported=True)
            details[field] = values[:3]
    for code in _codes(reported.get("failure_types")):
        if code in MODEL_FAILURE_MODULE_PRIORS:
            add(code, "/candidate_reported/failure_types", reported=True)
    if reported.get("stop_reason") in MODEL_FAILURE_MODULE_PRIORS:
        add(reported["stop_reason"], "/candidate_reported/stop_reason", reported=True)
    if reported.get("model_claims_evidence") is False:
        add("evidence_insufficient", "/candidate_reported/model_claims_evidence", reported=True)

    return {"host_observed": sorted(host), "model_reported": sorted(model),
            "suggested_modules": _rank(priors),
            "module_priors": {m: round(priors[m], 4) for m in _rank(priors)},
            "evidence_refs": dict(sorted(refs.items())), "model_details": details,
            "measurement_status": "observed",
            "observations": {"completed_search_calls": searches, "empty_search_calls": empty_searches,
                             "read_model_calls": read_calls, "completed_read_presentations": len(reads),
                             "verified_read_quote_count": valid_read_quotes,
                             "answer_model_calls": answer_calls, "final_evidence_count": final_count,
                             "trace_complete": trace_complete,
                             "semantic_support": "not_host_verified"}}


def _identity(measurement):
    _role_ok(measurement, required=True)
    for key in ("panel_hash", "evaluator_epoch"):
        if not isinstance(measurement.get(key), str) or not measurement[key]:
            raise ValueError("feedback requires " + key)
    return {key: measurement[key] for key in ("panel_hash", "evaluator_epoch")}


def _validated_rows(measurement):
    identity = _identity(measurement)
    rows = measurement.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("feedback requires nonempty host measurement rows")
    grouped = defaultdict(list)
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("invalid host measurement row")
        _role_ok(row)
        for key, value in identity.items():
            if key in row and row[key] != value:
                raise ValueError("row identity differs from measurement")
        qid = row.get("question_id")
        if not isinstance(qid, str) or not qid:
            raise ValueError("missing question_id")
        grouped[qid].append(row)
    return grouped


def _scored_groups(measurement, grouped):
    if measurement.get("complete") is not True:
        raise ValueError("learning feedback requires complete measurement")
    per_question = _map(measurement.get("per_question"))
    if set(per_question) != set(grouped):
        raise ValueError("per_question must match measured rows")
    result = {}
    for qid, rows in sorted(grouped.items()):
        scores = [_number(row.get("score")) for row in rows]
        if any(s is None for s in scores):
            raise ValueError("completed row requires a finite host score")
        score = _number(per_question.get(qid))
        mean = sum(scores) / len(scores)
        if score is None or not math.isclose(score, mean, abs_tol=1e-9, rel_tol=1e-9):
            raise ValueError("per_question score differs from host rows")
        result[qid] = score
    panel_mean = sum(result.values()) / len(result)
    if _number(measurement.get("score")) is None or not math.isclose(
            float(measurement["score"]), panel_mean, abs_tol=1e-9, rel_tol=1e-9):
        raise ValueError("panel score differs from host rows")
    return result


def _case_info(case):
    diagnostic = case["diagnostics"]
    hard = set(diagnostic["host_observed"])
    informative = hard - {"answer_empty", "execution_error"}
    return (5 * ("execution_error" in hard) + 2 * len(informative) + len(hard)
            + 0.5 * len(diagnostic["model_reported"])
            + 2 * abs(case.get("signed_delta") or 0))


def _witnesses(row):
    """Two bounded source-provenance witnesses, never a semantic support verdict."""
    quotes = []
    for read in _items(_map(row.get("host_evidence_trace")).get("read_presentations")):
        for quote in _items(_map(read).get("verified_quotes")):
            quote = _map(quote)
            text = quote.get("quote")
            if not isinstance(text, str) or not text:
                continue
            item = {"docid": _text(str(quote.get("docid", "")), 100),
                    "start": quote.get("start") if type(quote.get("start")) is int else None,
                    "end": quote.get("end") if type(quote.get("end")) is int else None,
                    "quote_excerpt": text[:240], "excerpt_truncated": len(text) > 240,
                    "full_quote_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "semantic_support": "not_host_verified"}
            if item not in quotes:
                quotes.append(item)
    return sorted(quotes, key=lambda x: (x["docid"], x["start"] or 0, x["full_quote_sha256"]))[:2]


def compact_feedback(measurement, tasks, max_cases=4):
    """Bounded developer feedback; optional parent_measurement supplies signed pairs.

    Reject non-fit/mismatched roles and identities before looking at content.
    Unknown API/host outcomes return unavailable with no scores or learning cases.
    Only question text is copied from tasks; reference/answer fields are ignored.
    """
    if not isinstance(measurement, Mapping):
        raise TypeError("measurement must be a mapping")
    if type(max_cases) is not int or not 0 <= max_cases <= 16:
        raise ValueError("max_cases must be an integer from 0 through 16")
    identity = _identity(measurement)
    base = {"schema": "rag-rsi-v3-feedback-1", "role": "D_fit", **identity,
            "node_id": _text(measurement.get("node_id"), 160),
            "reference_not_sent": True, "module_priors_are_design_heuristics": True,
            "semantic_support": "not_host_verified"}

    def unavailable(reason):
        return {**base, "measurement_status": "unavailable", "score": None,
                "cases": [], "summary": {"reason": reason},
                "module_priors": {}, "suggested_modules": [], "paired_summary": None}

    reason = _unavailable(measurement)
    if reason:
        return unavailable(reason)
    groups = _validated_rows(measurement)
    for rows in groups.values():
        for row in rows:
            if _unavailable(row):
                return unavailable("host row outcome unavailable")
    scores = _scored_groups(measurement, groups)

    task_values = list(tasks.values()) if isinstance(tasks, Mapping) else list(tasks)
    questions = {}
    for task in task_values:
        if not isinstance(task, Mapping):
            raise ValueError("public tasks must be mappings")
        _role_ok(task)
        qid = task.get("question_id")
        if qid in questions:
            raise ValueError("duplicate public question_id")
        if not isinstance(qid, str):
            raise ValueError("public task requires question_id")
        questions[qid] = _text(task.get("question"), 800)
    if set(questions) != set(groups):
        raise ValueError("public tasks must exactly match the fit panel")

    parent = measurement.get("parent_measurement")
    parent_scores, parent_groups = {}, {}
    if parent is not None:
        if not isinstance(parent, Mapping) or _identity(parent) != identity:
            raise ValueError("parent measurement identity differs")
        if parent.get("metric") != measurement.get("metric"):
            raise ValueError("parent metric differs")
        parent_groups = _validated_rows(parent)
        if _unavailable(parent) or any(_unavailable(r) for rows in parent_groups.values() for r in rows):
            raise ValueError("unavailable parent cannot provide signed learning pairs")
        parent_scores = _scored_groups(parent, parent_groups)
        if set(parent_scores) != set(scores):
            raise ValueError("parent and child question panels differ")

    target = _number(measurement.get("score_max"))
    if target is None and measurement.get("metric") in ("em", "f1"):
        target = 1.0
    cases, host_counts, model_counts, prior_totals = [], Counter(), Counter(), Counter()
    for qid, rows in sorted(groups.items()):
        options = []
        diagnostics = [diagnose_execution(row) for row in rows]
        # One question, one vote: repeated calls/failure logs cannot inflate priors.
        host_counts.update(set(code for d in diagnostics for code in d["host_observed"]))
        model_counts.update(set(code for d in diagnostics for code in d["model_reported"]))
        for module in MODULES:
            prior_totals[module] += max((d["module_priors"].get(module, 0) for d in diagnostics), default=0)
        for row, diagnosis in zip(rows, diagnostics):
            case = {"question_id": qid, "question": questions[qid],
                    "prediction": _text(row.get("answer"), 400),
                    "host_score": scores[qid], "sampled_repeat_score": float(row["score"]),
                    "repeat": row.get("repeat", 0), "repeat_count": len(rows),
                    "diagnostics": diagnosis, "evidence_witnesses": _witnesses(row),
                    "signed_delta": scores[qid] - parent_scores[qid] if parent_scores else None,
                    "parent_host_score": parent_scores.get(qid)}
            options.append(case)
        # Deterministic representative; no dependence on input file/list order.
        representative = min(options, key=lambda c: (-_case_info(c), c["sampled_repeat_score"],
                                                      _digest(c)))
        if parent_scores:
            pd = [diagnose_execution(row) for row in parent_groups[qid]]
            representative["paired_diagnostics"] = {
                "kind": "same_question_mean_scores",
                "parent_node_id": _text(parent.get("node_id"), 160),
                "parent_repeat_count": len(parent_groups[qid]),
                "parent_host_observed": sorted(set(x for d in pd for x in d["host_observed"])),
                "parent_model_reported": sorted(set(x for d in pd for x in d["model_reported"])),
                "causal_attribution": "not_established"}
        cases.append(representative)

    selected = []

    def choose(pool, key):
        if len(selected) < max_cases and pool:
            choice = min(pool, key=key)
            if choice not in selected:
                selected.append(choice)

    regressions = [c for c in cases if c["signed_delta"] is not None and c["signed_delta"] < 0]
    improvements = [c for c in cases if c["signed_delta"] is not None and c["signed_delta"] > 0]
    if regressions:
        choose(regressions, lambda c: (c["signed_delta"], -_case_info(c), c["question_id"]))
    else:
        failures = [c for c in cases if c["diagnostics"]["host_observed"]
                    or c["diagnostics"]["model_reported"]
                    or (target is not None and c["host_score"] < target)]
        choose(failures, lambda c: (-_case_info(c), c["host_score"], c["question_id"]))
    if improvements:
        choose(improvements, lambda c: (-c["signed_delta"], -_case_info(c), c["question_id"]))
    elif selected and target is not None:
        choose([c for c in cases if c not in selected and c["host_score"] >= target],
               lambda c: (-c["host_score"], -_case_info(c), c["question_id"]))
    while len(selected) < min(max_cases, len(cases)):
        covered = {("host", code) for c in selected for code in c["diagnostics"]["host_observed"]}
        covered |= {("model", code) for c in selected for code in c["diagnostics"]["model_reported"]}

        def diverse(c):
            categories = {("host", code) for code in c["diagnostics"]["host_observed"]}
            categories |= {("model", code) for code in c["diagnostics"]["model_reported"]}
            return (-len(categories - covered), -_case_info(c), c["host_score"], c["question_id"])

        choose([c for c in cases if c not in selected], diverse)

    priors = {m: round(prior_totals[m] / len(groups), 4) for m in MODULES if prior_totals[m] > 0}
    deltas = [c["signed_delta"] for c in cases if c["signed_delta"] is not None]
    paired_summary = None if not deltas else {
        "paired_questions": len(deltas), "improved": sum(d > 0 for d in deltas),
        "regressed": sum(d < 0 for d in deltas), "unchanged": sum(d == 0 for d in deltas),
        "mean_signed_gain": sum(deltas) / len(deltas), "min_signed_gain": min(deltas),
        "max_signed_gain": max(deltas), "clipped_negative_gains": False}
    return {**base, "measurement_status": "complete", "score": float(measurement["score"]),
            "metric": measurement.get("metric"), "cases": selected,
            "summary": {"question_count": len(groups), "selected_cases": len(selected),
                        "host_observed": dict(sorted(host_counts.items())),
                        "model_reported": dict(sorted(model_counts.items())),
                        "score_shortfall_questions": (sum(s < target for s in scores.values())
                                                     if target is not None else None),
                        "selection": "signed contrasts then distinct informative failures"},
            "module_priors": {m: priors[m] for m in _rank(priors)},
            "suggested_modules": _rank(priors), "paired_summary": paired_summary}
