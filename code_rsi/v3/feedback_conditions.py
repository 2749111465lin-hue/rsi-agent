"""Explicit information controls for a frozen D_fit developer-feedback contrast.

Aggregate, cases and trace are nested information sets, not different case
selectors. References, experience, priors and free-form summaries are never
copied into these views. The caller must bind the schedule before measurement
and reject an oversized complete provider body without asymmetric cropping.
"""
from collections.abc import Mapping
from copy import deepcopy

from .diagnostics import compact_feedback, diagnose_execution, execution_flow

SCHEMA = "rag-rsi-v3-controlled-feedback-1"
CONDITIONS = ("aggregate", "cases", "trace")
RESOURCE_FIELDS = ("model_calls", "search_calls", "read_calls")
DELIVERY_FIELDS = ("execution_ok", "answer_usable", "answer_origin_valid", "citation_source_valid")


def _panel(measurement, task_values, schedule):
    """Validate all rows even when the condition will disclose no cases."""
    checked = compact_feedback(measurement, task_values, max_cases=0)
    if checked.get("measurement_status") != "complete":
        raise ValueError("controlled feedback requires a complete available D_fit measurement")
    if checked.get("metric") not in {"em", "f1"}:
        raise ValueError("controlled feedback requires an explicit EM or F1 metric")
    if (not isinstance(schedule, list) or not 1 <= len(schedule) <= 16
            or any(not isinstance(item, Mapping) or set(item) != {"question_id", "repeat"}
                   or not isinstance(item["question_id"], str) or not item["question_id"]
                   or type(item["repeat"]) is not int or item["repeat"] < 0 for item in schedule)):
        raise ValueError("case_schedule requires one to sixteen explicit question/repeat pairs")
    if len({item["question_id"] for item in schedule}) != len(schedule):
        raise ValueError("each scheduled question may appear at most once")
    questions = {}
    for task in task_values:
        question = task.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("public question must be nonempty text")
        questions[task["question_id"]] = question
    rows, bykey, repeats = measurement["rows"], {}, {}
    for row in rows:
        repeat = row.get("repeat")
        if type(repeat) is not int or repeat < 0:
            raise ValueError("every measured row requires an explicit nonnegative integer repeat")
        key = row["question_id"], repeat
        if key in bykey:
            raise ValueError("duplicate measured question/repeat")
        if not 0 <= row["score"] <= 1:
            raise ValueError("host score must remain within the unit interval")
        if not isinstance(row.get("answer"), str):
            raise ValueError("every measured answer must be text")
        bykey[key] = row
        repeats.setdefault(key[0], set()).add(repeat)
    repeat_sets = list(repeats.values())
    if (any(values != repeat_sets[0] for values in repeat_sets)
            or repeat_sets[0] != set(range(len(repeat_sets[0])))):
        raise ValueError("complete fit panel requires identical contiguous repeats for every question")
    if any((item["question_id"], item["repeat"]) not in bykey for item in schedule):
        raise ValueError("scheduled case is absent from the complete measured panel")
    return checked, questions, rows, bykey, repeats


def _delivery(rows):
    result = {}
    for field in DELIVERY_FIELDS:
        values = [row.get(field) for row in rows]
        if any(value is not None and type(value) is not bool for value in values):
            raise ValueError("host delivery flags must be booleans or explicitly unavailable")
        result[field] = None if any(value is None for value in values) else sum(values)
    return result


def _resources(rows):
    values = {field: [] for field in RESOURCE_FIELDS}
    for row in rows:
        usage = row.get("resource_usage")
        if usage is not None and not isinstance(usage, Mapping):
            raise ValueError("host resource usage must be a mapping when available")
        for field in RESOURCE_FIELDS:
            value = usage.get(field) if usage is not None else None
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("host resource counters must be nonnegative integers")
            values[field].append(value)
    return {field: None if any(value is None for value in items) else sum(items)
            for field, items in values.items()}


def _diagnostics(row):
    observed = diagnose_execution(row)
    # This is a new whitelist object, never a redacted copy of a raw receipt or
    # the richer diagnostics object (which also contains module priors).
    return {"host_observed": list(observed["host_observed"]),
            "model_reported": list(observed["model_reported"]),
            "model_details": {field: list(observed["model_details"].get(field, []))
                              for field in ("gaps", "conflicts")},
            "semantic_support": "not_host_verified", "model_reports_are_unverified": True}


def controlled_feedback(measurement, tasks, *, condition, case_schedule):
    """Return one strict feedback view without reading reference objects.

    Full question/prediction strings are shared exactly by cases and trace.
    A predeclared schedule chooses the same repeat regardless of its score,
    failure type or apparent informativeness. Program-ineligible raw scores
    never become supervision, including in the per-question view.
    """
    if condition not in CONDITIONS:
        raise ValueError("unknown controlled-feedback condition")
    task_values = list(tasks.values()) if isinstance(tasks, Mapping) else list(tasks)
    checked, questions, rows, bykey, repeats = _panel(measurement, task_values, case_schedule)
    eligible = checked["program_eligible"]
    feedback = {"schema": SCHEMA, "condition": condition, "role": "D_fit",
                "score": checked["score"], "metric": checked["metric"],
                "question_count": len(questions), "measured_rows": len(rows),
                "program_eligible": eligible, "delivery": _delivery(rows),
                "resource_usage": _resources(rows),
                "raw_reference_objects_not_sent": True,
                "fit_feedback_can_reveal_accepted_answers": condition != "aggregate"}
    if condition == "aggregate":
        return feedback
    cases = []
    for item in case_schedule:
        qid, repeat = item["question_id"], item["repeat"]
        row = bykey[(qid, repeat)]
        case = {"question_id": qid, "repeat": repeat, "question": questions[qid],
                "prediction": row["answer"],
                "sampled_repeat_score": float(row["score"]) if eligible is True else None,
                "host_score": float(measurement["per_question"][qid]) if eligible is True else None,
                "repeat_count": len(repeats[qid])}
        if condition == "trace":
            case["execution_flow"] = execution_flow(row)
            case["diagnostics"] = _diagnostics(row)
        cases.append(case)
    feedback["cases"] = cases
    return deepcopy(feedback)
