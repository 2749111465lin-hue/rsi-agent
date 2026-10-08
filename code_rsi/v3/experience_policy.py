"""Deterministic, fit-only action selection from host-observed experience.

This is a project design inferred from OpenMLE, GEPA and RAG evidence work,
not a reproduction or a statistically calibrated confidence-bound algorithm.
A host must construct these cards; field checks cannot authenticate their origin.
Answer quality determines the incumbent. Module utility uses signed, identity-
matched parent/child answer gains, a descriptive small-sample/dispersion penalty,
observed failure frequency and same-unit measured cost. Retrieval proxies never
enter quality or gain. Missing costs are conservatively imputed, never free.
Every fourth step explores an unmeasured module; every fifth nonzero step may
expand an underexplored alternative parent. An empty history rotates by step.
No random state, model calls, disk access or sealed split access is used.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
import re
from statistics import fmean, median, stdev
from typing import Any, Iterable, Mapping

from .edit_scope import validated_scope

DEFAULT_MODULES = (
    "query_rewrite", "retrieval", "evidence_selection", "answer_generation",
)
FIT_ROLES = frozenset(("D_fit", "fit"))
FAILURE_STATUSES = frozenset(("failed", "error", "invalid", "execution_failed"))
COST_KEYS = ("cny", "usd", "seconds", "calls", "model_invocations", "tokens")
POLICY_VERSION = "rag-rsi-v3-experience-3"


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _order(card: Mapping[str, Any]) -> tuple:
    step = next((_number(card.get(k)) for k in ("step", "step_id", "generation_id")
                 if _number(card.get(k)) is not None), None)
    if step is None:
        match = re.search(r"(\d+)$", str(card.get("node_id", "")))
        step = float(match.group(1)) if match else -1.0
    return (step, str(card.get("created_at", "")), str(card["node_id"]),
            str(card.get("evaluation_id", "")))


def _parents(card: Mapping[str, Any]) -> list[str]:
    value = card.get("parent_node_ids", [])
    if not value and isinstance(card.get("parent_node_id"), str):
        value = [card["parent_node_id"]]
    return [x for x in value if isinstance(x, str)] if isinstance(value, (list, tuple)) else []


def _score(card: Mapping[str, Any]) -> float | None:
    if (card.get("complete") is not True or card.get("valid_program") is not True
            or card.get("program_eligible") is False):
        return None
    # score is the established host answer-quality field, not a retrieval score.
    if card.get("score_kind") in ("retrieval", "retrieval_proxy", "coverage", "evidence_coverage"):
        return None
    return _number(card.get("answer_score", card.get("score")))


def _failures(card: Mapping[str, Any]) -> tuple[str, ...]:
    labels = set()
    if isinstance(card.get("failure_class"), str) and card["failure_class"]:
        labels.add(card["failure_class"])
    classes = card.get("failure_classes") or {}
    if isinstance(classes, Mapping):
        labels.update(str(k) for k, v in classes.items()
                      if _number(v) is not None and v > 0)
    elif isinstance(classes, (list, tuple)):
        labels.update(x for x in classes if isinstance(x, str) and x)
    for receipt in card.get("failure_receipts") or []:
        if isinstance(receipt, Mapping) and isinstance(receipt.get("failure_class"), str):
            labels.add(receipt["failure_class"])
    failure = card.get("failure")
    if isinstance(failure, Mapping):
        labels.add(str(failure.get("failure_class", "execution_failure")))
    elif failure or card.get("status") in FAILURE_STATUSES:
        labels.add("execution_failure")
    return tuple(sorted(x for x in labels if x))


def _admit(cards: Iterable[Mapping[str, Any]], panel_hash: str,
           evaluator_epoch: str) -> tuple[list[dict], dict]:
    rejected = Counter()
    identities: dict[str, list[dict]] = {}
    for card in cards:
        if not isinstance(card, Mapping) or not isinstance(card.get("node_id"), str) or not card["node_id"]:
            rejected["invalid_card"] += 1
            continue
        roles = [card[k] for k in ("role", "split") if k in card]
        if not roles or any(role not in FIT_ROLES for role in roles):
            rejected["non_fit_role"] += 1
            continue
        if card.get("panel_hash") != panel_hash or card.get("evaluator_epoch") != evaluator_epoch:
            rejected["identity_mismatch"] += 1
            continue
        if card.get("source") not in (None, "host", "host_observed", "host_measured_D_fit"):
            rejected["non_host_source"] += 1
            continue
        if _score(card) is None and not _failures(card):
            rejected["unmeasured_without_failure"] += 1
            continue
        identities.setdefault(card["node_id"], []).append(dict(card))
    admitted = []
    for node_id in sorted(identities):
        group = identities[node_id]
        serial = {json.dumps(x, sort_keys=True, default=str, ensure_ascii=False) for x in group}
        if len(serial) != 1:
            rejected["conflicting_node_identity"] += len(group)
            continue
        admitted.append(group[0])
        rejected["duplicate_replay"] += len(group) - 1
    return sorted(admitted, key=_order), {k: v for k, v in sorted(rejected.items()) if v}


def _module(card: Mapping[str, Any]) -> str | None:
    # A declared intention (including legacy module/target_module) is never a
    # module gain label. This is syntactic scope association, not causal evidence.
    scope = validated_scope(card.get("actual_edit_scope"))
    if scope is None or scope["attribution"] != "single_module":
        return None
    if card.get("intended_target_module") != scope["intended_target_module"]:
        return None
    return scope["associated_module"]


def _cost(card: Mapping[str, Any], key: str) -> float | None:
    usage = card.get("resource_usage") or {}
    value = _number(usage.get(key)) if isinstance(usage, Mapping) else None
    return value if value is not None and value >= 0 else None


def _failure_context(card: Mapping[str, Any], by_id: Mapping[str, dict]) -> set[str]:
    explicit = card.get("context_failure_classes")
    if isinstance(explicit, (list, tuple)) and explicit:
        return {str(x) for x in explicit}
    parents = [by_id[p] for p in _parents(card) if p in by_id]
    labels = set().union(*(_failures(p) for p in parents)) if parents else set()
    return labels or {"answer_quality"}


def _gain(card: Mapping[str, Any], by_id: Mapping[str, dict]) -> float | None:
    value = _score(card)
    parent_ids = _parents(card)
    if value is None or not parent_ids or any(p not in by_id for p in parent_ids):
        return None
    parent_scores = [_score(by_id[p]) for p in parent_ids]
    if any(s is None for s in parent_scores):
        return None
    return value - max(parent_scores)


def _modules(allowed_modules: Iterable[str] | None) -> tuple[str, ...]:
    if allowed_modules is None:
        return DEFAULT_MODULES
    if isinstance(allowed_modules, str):
        raise ValueError("allowed_modules must be a sequence, not a string")
    values = tuple(dict.fromkeys(allowed_modules))
    if not values or any(not isinstance(x, str) or not x for x in values):
        raise ValueError("allowed_modules must contain non-empty module names")
    return values


def _debug_hint(card: Mapping[str, Any], modules: tuple[str, ...]) -> str | None:
    if _module(card) in modules:
        return _module(card)
    text = " ".join(_failures(card)).lower()
    aliases = (
        (("retriev", "search"), ("retrieval", "retriever")),
        (("query", "entity", "rewrite"), ("query_rewrite", "query")),
        (("context", "evidence", "citation", "pack"), ("evidence_selection", "evidence")),
        (("answer", "generat", "format", "schema"), ("answer_generation", "answer")),
    )
    for markers, names in aliases:
        if any(marker in text for marker in markers):
            for name in names:
                if name in modules:
                    return name
    return None


def choose_next(cards: Iterable[Mapping[str, Any]], *, step: int,
                panel_hash: str, evaluator_epoch: str,
                allowed_modules: Iterable[str] | None = None) -> dict:
    """Choose Draft, Debug or Improve, then a module, using legal fit evidence.

    Cards need role, panel_hash, evaluator_epoch, node_id, host complete/valid flags
    and answer score, or an explicit failure. step/step_id gives chronology.
    For module learning, include host actual_edit_scope and parent_node_ids.
    Legacy declarations may select parents but never supply module gain samples.
    Old cards missing evaluator_epoch are rejected rather than silently reused.
    """
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    if not isinstance(panel_hash, str) or not panel_hash or not isinstance(evaluator_epoch, str) or not evaluator_epoch:
        raise ValueError("non-empty panel_hash and evaluator_epoch are required")
    modules = _modules(allowed_modules)
    legal, rejected = _admit(cards, panel_hash, evaluator_epoch)
    by_id = {c["node_id"]: c for c in legal}
    measured = [c for c in legal if _score(c) is not None]
    # Successful descendants resolve failed ancestors, including chains of repairs.
    resolved = set()
    for card in measured:
        todo = list(_parents(card))
        seen = set()
        while todo:
            node_id = todo.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            resolved.add(node_id)
            if node_id in by_id:
                todo.extend(_parents(by_id[node_id]))
    expansion_counts = Counter(p for c in legal for p in _parents(c) if p in by_id)
    incumbent = min(measured, key=lambda c: (-_score(c), _order(c))) if measured else None
    pending = [c for c in legal if _score(c) is None and _failures(c)
               and c["node_id"] not in resolved]
    if pending:
        parent = max(pending, key=_order)
        operator, parent_reason = "Debug", "latest_unresolved_fit_failure"
    elif measured:
        parent = incumbent
        operator, parent_reason = "Improve", "best_measured_answer_quality"
        if step > 0 and step % 5 == 0:
            def behavior_key(c):
                behavior = c.get("behavior") or {}
                if isinstance(behavior, Mapping) and behavior.get("group_hash"):
                    return ("observed", str(behavior["group_hash"]))
                return ("program", str(c.get("program_id", c["node_id"])))
            alternatives = [c for c in measured if c["node_id"] != incumbent["node_id"]
                            and (expansion_counts[c["node_id"]] < expansion_counts[incumbent["node_id"]]
                                 or behavior_key(c) != behavior_key(incumbent))]
            if alternatives:
                least = min(expansion_counts[c["node_id"]] for c in alternatives)
                pool = sorted((c for c in alternatives if expansion_counts[c["node_id"]] == least),
                              key=_order)
                parent = pool[(step // 5 - 1) % len(pool)]
                parent_reason = "deterministic_underexpanded_parent_exploration"
    else:
        parent = None
        operator, parent_reason = "Draft", "no_legal_fit_parent"
    context = set(_failures(parent)) if parent else set()
    context = context or {"answer_quality"}
    trials = [c for c in legal if _module(c) in modules and _parents(c)
              and all(p in by_id for p in _parents(c))
              and _failure_context(c, by_id) & context]
    cost_key = next((k for k in COST_KEYS if any(_cost(c, k) is not None for c in trials)), None)
    known_costs = [_cost(c, cost_key) for c in trials
                   if cost_key and _cost(c, cost_key) is not None]
    cost_scale = median([x for x in known_costs if x > 0]) if any(x > 0 for x in known_costs) else 1.0
    missing_cost = 2 * max(known_costs + [cost_scale])
    stats = {}
    for module in modules:
        relevant = [c for c in trials if _module(c) == module]
        gains = [g for c in relevant if (g := _gain(c, by_id)) is not None]
        failure_count = sum(_score(c) is None and bool(_failures(c)) for c in relevant)
        n = len(gains)
        mean_gain = fmean(gains) if gains else None
        uncertainty = (0.05 / math.sqrt(n + 1) +
                       (stdev(gains) / math.sqrt(n) if n > 1 else 0.0))
        imputed_count = sum(_cost(c, cost_key) is None for c in relevant) if cost_key else len(relevant)
        effective = (fmean([_cost(c, cost_key) if _cost(c, cost_key) is not None else missing_cost
                           for c in relevant]) if relevant and cost_key else None)
        failure_penalty = 0.05 * failure_count / max(1, len(relevant))
        cost_penalty = 0.03 * math.log1p(effective / cost_scale) if effective is not None else 0.0
        utility = mean_gain - uncertainty - failure_penalty - cost_penalty if mean_gain is not None else None
        stats[module] = {
            "trials": len(relevant), "gain_samples": n, "signed_gains": gains,
            "mean_signed_gain": mean_gain, "uncertainty_penalty": uncertainty,
            "failure_count": failure_count, "failure_penalty": failure_penalty,
            "cost_unit": cost_key, "mean_effective_cost": effective,
            "missing_cost_count": imputed_count, "cost_penalty": cost_penalty,
            "utility": utility, "experience_ids": [c["node_id"] for c in relevant],
        }
    known = [m for m in modules if stats[m]["utility"] is not None]
    unknown = [m for m in modules if stats[m]["utility"] is None]
    raw_priors=(parent.get("diagnostics") or {}).get("module_priors",{}) if parent else {}
    priors={m:v for m in modules if (v:=_number(raw_priors.get(m))) is not None and 0<v<=1}
    def diagnostic_choice(pool, rotation):
        ranked=[m for m in pool if m in priors]
        return max(ranked,key=lambda m:(priors[m],-modules.index(m))) if ranked else pool[rotation%len(pool)]
    debug_hint = _debug_hint(parent, modules) if parent and operator == "Debug" else None
    if debug_hint:
        target, module_reason = debug_hint, "failed_module_requires_repair"
    elif not known:
        target = diagnostic_choice(modules,step)
        module_reason = "observed_failure_cold_start_prior" if priors else "deterministic_cold_start_rotation"
    elif unknown and (step % 4 == 0 or max(stats[m]["mean_signed_gain"] for m in known) <= 0):
        target = diagnostic_choice(unknown,step//4)
        module_reason = "diagnostic_unmeasured_module_exploration" if any(m in priors for m in unknown) else "deterministic_unmeasured_module_exploration"
    else:
        target = max(known, key=lambda m: (stats[m]["utility"], -modules.index(m)))
        module_reason = "matched_failure_signed_gain_uncertainty_and_cost"
    decision = {
        "parent_node_id": parent["node_id"] if parent else None,
        "operator": operator, "target_module": target, "intended_target_module": target,
        "reason": f"{parent_reason}; {module_reason}",
        "experience_ids": [],
        "panel_hash": panel_hash, "evaluator_epoch": evaluator_epoch, "role": "D_fit",
        "diagnostics": {
            "policy_version": POLICY_VERSION, "step": step, "accepted_cards": len(legal),
            "rejected_cards": rejected, "failure_context": sorted(context),
            "parent_answer_score": _score(parent) if parent else None,
            "incumbent_node_id": incumbent["node_id"] if incumbent else None,
            "parent_expansion_counts": {c["node_id"]: expansion_counts[c["node_id"]] for c in measured},
            "failure_context_is_diagnostic_not_reward": True,
            "module_statistics": stats, "allowed_modules": list(modules),
            "module_statistics_basis": "host_ast_scope_association_only",
            "module_association_is_causal": False,
            "unattributed_experience_ids": [c["node_id"] for c in legal if _parents(c) and _module(c) is None],
            "cold_start_module_priors":priors,"priors_added_to_answer_reward":False,
            "cost_unit": cost_key, "missing_cost_imputation": missing_cost if cost_key else None,
            "quality_basis": "host_answer_score_only",
            "uncertainty_is_calibrated_confidence_bound": False,
        },
    }
    decision["experience_ids"] = [c["node_id"] for c in memory_for_action(legal, decision)]
    return decision


def memory_for_action(cards: Iterable[Mapping[str, Any]], decision: Mapping[str, Any],
                      limit: int = 4) -> list[dict]:
    """Return newest legal fit records relevant to this exact module/action.

    Selected-parent failures and repairs of the same failure family are included.
    Scores remain None for invalid measurements. Full question text is not copied.
    Decision identity is required; report/select cards are rejected again here.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    if not limit:
        return []
    panel, epoch = decision.get("panel_hash"), decision.get("evaluator_epoch")
    if not isinstance(panel, str) or not panel or not isinstance(epoch, str) or not epoch:
        raise ValueError("decision must carry panel_hash and evaluator_epoch")
    legal, _ = _admit(cards, panel, epoch)
    by_id = {c["node_id"]: c for c in legal}
    target, parent_id = decision.get("target_module"), decision.get("parent_node_id")
    context = set(decision.get("diagnostics", {}).get("failure_context") or ["answer_quality"])
    relevant = []
    for card in legal:
        selected_parent = card["node_id"] == parent_id
        module_match = _module(card) == target
        related = bool(_failure_context(card, by_id) & context or set(_failures(card)) & context)
        if selected_parent or (module_match and related):
            relevant.append(card)
    output = []
    for card in sorted(relevant, key=_order, reverse=True)[:limit]:
        output.append(deepcopy({
            "node_id": card["node_id"], "program_id": card.get("program_id"),
            "evaluation_id": card.get("evaluation_id"), "role": card.get("role", card.get("split")),
            "panel_hash": panel, "evaluator_epoch": epoch, "operator": card.get("operator"),
            "target_module": card.get("intended_target_module", card.get("target_module", card.get("module"))),
            "intended_target_module": card.get("intended_target_module", card.get("target_module", card.get("module"))),
            "associated_module": _module(card),
            "actual_edit_scope": validated_scope(card.get("actual_edit_scope")),
            "module_association_is_causal": False, "parent_node_ids": _parents(card),
            "complete": card.get("complete") is True, "valid_program": card.get("valid_program") is True,
            "score": _score(card), "signed_delta_vs_best_parent": _gain(card, by_id),
            "program_eligible": _score(card) is not None,
            "paired_comparison_eligible": _gain(card, by_id) is not None,
            "raw_diagnostics": card.get("raw_diagnostics"),
            "failure_classes": list(_failures(card)),
            "failure_assessment_source": card.get("failure_assessment_source", card.get("failure_provenance")),
            "failure_receipts": (card.get("failure_receipts") or [])[-4:],
            "diagnostics": card.get("diagnostics"),
            "resource_usage": card.get("resource_usage"),
        }))
    return output
