"""Pre-registered descriptive paired analysis of already host-graded rows.

Repeats are averaged within a question. Questions receive equal weight; whole
predeclared dependence groups are resampled jointly across comparisons. The
percentile/Bonferroni intervals and normal design sensitivity are approximations,
not finite-sample coverage or power guarantees. No p values are calculated.
This module checks a complete panel; it does not authenticate external receipts.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import math
import random
from statistics import NormalDist, fmean, stdev

SPEC_SCHEMA = "rag-rsi-paired-analysis-1"
SINGLE_CONTRAST_SCHEMA = "rag-rsi-paired-analysis-single-1"
METRICS = ("answer_em", "answer_f1")
FIELDS = {"schema", "primary_metric", "comparisons", "question_groups", "confidence_level",
          "bootstrap_samples", "bootstrap_seed", "target_effect", "power"}


def _names(values, label):
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError(label + " must be a collection of distinct names")
    try:
        result = list(values)
    except TypeError as error:
        raise ValueError(label + " must be a collection of names") from error
    if (not result or any(not isinstance(x, str) or not x.strip() or x != x.strip() for x in result)
            or len(result) != len(set(result))):
        raise ValueError(label + " must contain distinct nonblank strings")
    return result


def _finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def validate_analysis(spec, *, question_ids, arm_names):
    """Validate all nine pre-registered fields and return an independent value."""
    questions = _names(question_ids, "question_ids")
    arms = _names(arm_names, "arm_names")
    if len(arms) < 2 or not isinstance(spec, Mapping) or set(spec) != FIELDS:
        raise ValueError("analysis requires its exact nine fields and at least two arms")
    if spec["schema"] not in (SPEC_SCHEMA, SINGLE_CONTRAST_SCHEMA) or spec["primary_metric"] not in METRICS:
        raise ValueError("unsupported analysis schema or primary metric")
    groups = spec["question_groups"]
    if (not isinstance(groups, Mapping) or set(groups) != set(questions)
            or any(not isinstance(g, str) or not g.strip() or g != g.strip() for g in groups.values())):
        raise ValueError("question_groups must cover every question exactly with nonblank group IDs")
    comparisons = spec["comparisons"]
    # A new contract permits one primary contrast without weakening older plans.
    count = 1 if spec["schema"] == SINGLE_CONTRAST_SCHEMA else 2
    if not isinstance(comparisons, list) or len(comparisons) != count:
        word = "one" if count == 1 else "two"
        raise ValueError("exactly " + word + " pre-registered comparisons are required")
    names, pairs = set(), set()
    for item in comparisons:
        if not isinstance(item, Mapping) or set(item) != {"name", "baseline", "candidate"}:
            raise ValueError("each comparison requires name, baseline and candidate")
        name, baseline, candidate = item["name"], item["baseline"], item["candidate"]
        if (not isinstance(name, str) or not name.strip() or name != name.strip() or name in names
                or not isinstance(baseline, str) or not isinstance(candidate, str)
                or baseline not in arms or candidate not in arms or baseline == candidate):
            raise ValueError("comparison identities must be unique and refer to distinct declared arms")
        pair = frozenset((baseline, candidate))
        if pair in pairs:
            raise ValueError("reversed or duplicated versions of the same comparison are not independent declarations")
        names.add(name); pairs.add(pair)
    bounds = {"confidence_level": lambda x: .8 <= x < 1,
              "target_effect": lambda x: 0 < x <= 1, "power": lambda x: .5 < x < 1}
    for field, valid in bounds.items():
        if not _finite(spec[field]) or not valid(spec[field]):
            raise ValueError("invalid " + field)
    if type(spec["bootstrap_samples"]) is not int or not 1000 <= spec["bootstrap_samples"] <= 100000:
        raise ValueError("bootstrap_samples must be an integer from 1000 to 100000")
    if type(spec["bootstrap_seed"]) is not int or spec["bootstrap_seed"] < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    return {"schema": spec["schema"], "primary_metric": spec["primary_metric"],
            "comparisons": [dict(x) for x in comparisons],
            "question_groups": {q: groups[q] for q in sorted(questions)},
            "confidence_level": float(spec["confidence_level"]),
            "bootstrap_samples": spec["bootstrap_samples"], "bootstrap_seed": spec["bootstrap_seed"],
            "target_effect": float(spec["target_effect"]), "power": float(spec["power"])}


def _signs(values):
    return {"positive": sum(x > 0 for x in values), "zero": sum(x == 0 for x in values),
            "negative": sum(x < 0 for x in values)}


def _constant(values):
    return max(values) - min(values) <= 1e-12


def _quantile(ordered, probability):
    position = (len(ordered) - 1) * probability
    low = math.floor(position); high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _design(comparison, sizes, spec):
    count = len(sizes)
    result = {"status": "pilot_variance_approximation", "linearized_cluster_se": None,
              "mde": None, "required_independent_groups": None, "additional_independent_groups": None,
              "target_effect": spec["target_effect"], "power": spec["power"],
              "power_is_guaranteed": False,
              "assumptions": "independent groups; same cluster-size and residual distribution; normal approximation"}
    if count < 2:
        return {**result, "availability": "insufficient_clusters"}
    mean = comparison["mean_delta"]
    residuals = [comparison["per_group"][g]["sum_delta"] - mean * n for g, n in sizes.items()]
    se = math.sqrt(count / (count - 1) * math.fsum(r * r for r in residuals)) / sum(sizes.values())
    if se <= 1e-12:
        return {**result, "availability": "insufficient_variation"}
    tail = (1 - spec["confidence_level"]) / (2 * len(spec["comparisons"]))
    critical = -NormalDist().inv_cdf(tail)
    mde = (critical + NormalDist().inv_cdf(spec["power"])) * se
    log_required = math.log(count) + 2 * (math.log(mde) - math.log(spec["target_effect"]))
    needed = max(2, math.ceil(math.exp(log_required))) if log_required < 709 else None
    return {**result, "availability": "available" if needed is not None else "group_count_exceeds_numeric_range",
            "linearized_cluster_se": se, "mde": mde, "required_independent_groups": needed,
            "additional_independent_groups": max(0, needed - count) if needed is not None else None,
            "mde_exceeds_unit_effect_range": mde > 1, "bonferroni_normal_critical_value": critical}


def _arm_noise(cells, questions, arm, repeats, primary):
    details = {}
    for q in questions:
        scores = [cells[(q, arm, r)][primary] for r in range(repeats)]
        exact = [cells[(q, arm, r)]["answer_em"] == 1 for r in range(repeats)]
        details[q] = {"mean_score": fmean(scores), "repeat_scores": scores,
                      "repeat_sample_stddev": stdev(scores) if repeats > 1 else None,
                      "repeat_range": max(scores) - min(scores) if repeats > 1 else None,
                      "adjacent_score_changes": sum(a != b for a, b in zip(scores, scores[1:])) if repeats > 1 else None,
                      "adjacent_exact_match_flips": sum(a != b for a, b in zip(exact, exact[1:])) if repeats > 1 else None}
    result = {"mean_score": fmean(x["mean_score"] for x in details.values()), "per_question": details,
              "noise_metric": primary, "exact_match_state": "answer_em == 1",
              "repeat_count_per_question": repeats, "repeats_are_independent_questions": False,
              "mean_repeat_sample_stddev": None, "mean_repeat_range": None,
              "questions_with_repeat_variation": None, "adjacent_exact_match_flip_count": None,
              "adjacent_exact_match_flip_rate": None}
    if repeats > 1:
        flips = sum(x["adjacent_exact_match_flips"] for x in details.values())
        result.update(mean_repeat_sample_stddev=fmean(x["repeat_sample_stddev"] for x in details.values()),
                      mean_repeat_range=fmean(x["repeat_range"] for x in details.values()),
                      questions_with_repeat_variation=sum(x["repeat_range"] > 0 for x in details.values()),
                      adjacent_exact_match_flip_count=flips,
                      adjacent_exact_match_flip_rate=flips / (len(questions) * (repeats - 1)))
    return result


def paired_analysis(rows, spec, *, arm_names, expected_repeats):
    """Analyze a complete question x arm x repeat panel; never select successful rows."""
    if type(expected_repeats) is not int or expected_repeats < 1:
        raise ValueError("expected_repeats must be a positive integer")
    if not isinstance(spec, Mapping) or not isinstance(spec.get("question_groups"), Mapping):
        raise ValueError("pre-registered question_groups are required")
    arms = _names(arm_names, "arm_names")
    spec = validate_analysis(spec, question_ids=list(spec["question_groups"]), arm_names=arms)
    questions = list(spec["question_groups"])
    cells, ineligible = {}, []
    try:
        iterator = iter(rows)
    except TypeError as error:
        raise ValueError("graded rows must be an iterable panel") from error
    for row in iterator:
        if not isinstance(row, Mapping):
            raise ValueError("graded rows must be mappings")
        q, arm, repeat = row.get("question_id"), row.get("arm"), row.get("repeat")
        if (not isinstance(q, str) or q not in spec["question_groups"] or not isinstance(arm, str) or arm not in arms
                or type(repeat) is not int or not 0 <= repeat < expected_repeats
                or type(row.get("program_eligible")) is not bool):
            raise ValueError("invalid graded row identity or eligibility")
        key = (q, arm, repeat)
        if key in cells:
            raise ValueError("duplicate question/arm/repeat row")
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping) or any(not _finite(metrics.get(m)) or not 0 <= metrics[m] <= 1 for m in METRICS):
            raise ValueError("both answer metrics must be finite scores in [0, 1]")
        cells[key] = {m: float(metrics[m]) for m in METRICS}
        if row["program_eligible"] is not True:
            ineligible.append({"question_id": q, "arm": arm, "repeat": repeat})
    if len(cells) != len(questions) * len(arms) * expected_repeats:
        raise ValueError("missing rows: the full pre-registered panel is required")
    primary = spec["primary_metric"]
    arm_stats = {a: _arm_noise(cells, questions, a, expected_repeats, primary) for a in arms}
    groups = {g: [q for q in questions if spec["question_groups"][q] == g]
              for g in sorted(set(spec["question_groups"].values()))}
    sizes = {g: len(qs) for g, qs in groups.items()}
    comparisons = {}
    for item in spec["comparisons"]:
        per_question = {q: arm_stats[item["candidate"]]["per_question"][q]["mean_score"] -
                           arm_stats[item["baseline"]]["per_question"][q]["mean_score"] for q in questions}
        per_group = {g: {"question_ids": qs, "sample_size": len(qs),
                         "sum_delta": math.fsum(per_question[q] for q in qs),
                         "mean_delta": fmean(per_question[q] for q in qs)} for g, qs in groups.items()}
        comparisons[item["name"]] = {**item, "mean_delta": fmean(per_question.values()),
            "sample_size": len(questions), "cluster_count": len(groups), "per_question": per_question,
            "per_group": per_group, "question_sign_counts": _signs(per_question.values()),
            "group_sign_counts": _signs([x["mean_delta"] for x in per_group.values()])}
    result = {"schema": "rag-rsi-paired-result-1", "analysis": spec,
              "status": "protocol_invalid" if ineligible else "qualified",
              "quality_comparison_valid": not ineligible, "primary_metric": primary,
              "sample_size": len(questions), "cluster_count": len(groups), "expected_repeats": expected_repeats,
              "cell_count": len(cells), "estimand": "equal-question mean of repeat-averaged candidate-minus-baseline scores",
              "resampling_unit": "pre-registered dependence group", "hypothesis_tests_performed": False,
              "qualified": None,
              "raw_diagnostics": {"comparisons": deepcopy(comparisons), "arms": deepcopy(arm_stats),
                  "ineligible_cells": sorted(ineligible, key=lambda x: (x["question_id"], x["arm"], x["repeat"])),
                  "successful_subset_analysis": False, "diagnostic_only": True}}
    if ineligible:
        return result
    count = len(groups)
    rng = random.Random(spec["bootstrap_seed"])
    schedule = hashlib.sha256()
    samples = {name: [] for name in comparisons}
    group_ids = list(groups)
    if count > 1:
        for _ in range(spec["bootstrap_samples"]):
            multiplicities = [0] * count
            for _ in range(count):
                multiplicities[rng.randrange(count)] += 1
            schedule.update((repr(multiplicities) + ";").encode("ascii"))
            denominator = sum(n * sizes[g] for n, g in zip(multiplicities, group_ids))
            for name, comparison in comparisons.items():
                samples[name].append(math.fsum(n * comparison["per_group"][g]["sum_delta"]
                    for n, g in zip(multiplicities, group_ids)) / denominator)
    tail = (1 - spec["confidence_level"]) / (2 * len(comparisons))
    for name, comparison in comparisons.items():
        values = sorted(samples[name])
        status = ("insufficient_clusters" if count < 2 else
                  "insufficient_variation" if _constant(list(comparison["per_question"].values())) else
                  "degenerate" if _constant(values) else "estimated")
        comparison["confidence_interval"] = {"status": status,
            "lower": _quantile(values, tail) if status == "estimated" else None,
            "upper": _quantile(values, 1 - tail) if status == "estimated" else None,
            "method": "cluster percentile bootstrap with Bonferroni adjustment",
            "family_confidence_level": spec["confidence_level"],
            "per_comparison_confidence_level": 1 - 2 * tail,
            "coverage": "approximate_familywise_not_finite_sample_guarantee",
            "expected_bootstrap_tail_count": spec["bootstrap_samples"] * tail,
            "tail_resolution_limited": spec["bootstrap_samples"] * tail < 10}
        comparison["design_sensitivity"] = _design(comparison, sizes, spec)
    result["qualified"] = {"comparisons": comparisons, "arms": arm_stats,
        "resampling": {"method": "shared_whole_group_ratio_bootstrap", "same_draws_for_all_comparisons": True,
            "numerator": "sum of sampled group difference totals", "denominator": "sum of sampled group question counts",
            "bootstrap_samples": spec["bootstrap_samples"] if count > 1 else 0,
            "bootstrap_seed": spec["bootstrap_seed"], "schedule_sha256": schedule.hexdigest() if count > 1 else None}}
    return result
