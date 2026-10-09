"""Build frozen MuSiQue-Ans panels locally; no downloads or model calls.

Official data/overlap definition:
https://github.com/StonyBrookNLP/musique/tree/922ac98f19a201998dbdae6d7f2887a5258dbdeb
https://aclanthology.org/2022.tacl-1.31.pdf (Table 2 and section 5, S5)
Only a trusted local preparation process sees annotations. Public runtime inputs
are produced by the existing allowlist adapter, without combining local corpora.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

from .v3.datasets import (MUSIQUE_DOCUMENT_RENDERING, adapt_musique,
                          normalize_answer, validate_task_collection)

SCHEMA = "rag-rsi-musique-ans-panels-2"
GROUPED_SCHEMA = "rag-rsi-musique-ans-panels-3"
HISTORY_SCHEMA = "rag-rsi-musique-exposure-history-1"
GROUP_RELATIONS = ("question_id", "source_question_pair", "normalized_question",
                   "singlehop_id", "normalized_subquestion", "support_paragraph")
REVISION = "922ac98f19a201998dbdae6d7f2887a5258dbdeb"
RUNS_ROOT = Path(__file__).resolve().parents[1] / "runs"
ROLES = ("D_fit", "D_select", "D_report")
SELECTION_ORDER = ("D_report", "D_fit", "D_select")
FEATURES = ("question_id", "source_question_pair", "normalized_question",
            "singlehop_id", "support_paragraph", "subanswer")


class PanelError(ValueError):
    """Unsafe input or output contract; messages never contain example text."""


class PanelQuotaError(PanelError):
    def __init__(self, manifest, groups=None):
        super().__init__("requested panel quotas are unmet under the frozen sampling policy")
        self.manifest = manifest
        self.groups = groups


@dataclass(frozen=True)
class Candidate:
    line: int
    question_id: str
    source_id: str
    hop: int
    features: dict[str, frozenset[str]]
    order_key: str
    unresolved_text: tuple[str, ...] = ()


def _hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _id(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise PanelError("missing or invalid source identity")
    return str(value)


def _object_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise PanelError("duplicate JSON object field")
        value[key] = item
    return value


def _json(raw):
    try:
        return json.loads(raw, object_pairs_hook=_object_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(PanelError("nonfinite JSON value")))
    except (ValueError, UnicodeError) as error:
        raise PanelError("invalid JSON object") from error


def _adapt(row):
    if not isinstance(row, dict) or row.get("answerable") is not True:
        raise PanelError("only explicitly answerable official MuSiQue-Ans rows are accepted")
    if not isinstance(row.get("id"), str) or not row["id"].strip():
        raise PanelError("official id must be a nonempty string")
    if not isinstance(row.get("answer"), str) or not row["answer"].strip():
        raise PanelError("answer annotation is required")
    if not isinstance(row.get("answer_aliases"), list):
        raise PanelError("official answer_aliases annotation is required")
    paragraphs = row.get("paragraphs")
    if (not isinstance(paragraphs, list) or not 1 <= len(paragraphs) <= 20
            or any(not isinstance(p, dict) or type(p.get("idx")) is not int
                   or type(p.get("is_supporting")) is not bool for p in paragraphs)
            or {p["idx"] for p in paragraphs} != set(range(len(paragraphs)))):
        raise PanelError("one to twenty original indexed paragraphs with complete support annotations are required")
    decomposition = row.get("question_decomposition")
    if not isinstance(decomposition, list) or len(decomposition) not in (2, 3, 4):
        raise PanelError("official two-to-four-hop decomposition annotation is required")
    supporting = {p["idx"] for p in paragraphs if p["is_supporting"]}
    for item in decomposition:
        if not isinstance(item, dict):
            raise PanelError("invalid single-hop annotation")
        _id(item.get("id"))
        if (not isinstance(item.get("question"), str) or not item["question"].strip()
                or not isinstance(item.get("answer"), str)
                or type(item.get("paragraph_support_idx")) is not int
                or item["paragraph_support_idx"] not in supporting):
            raise PanelError("incomplete or inconsistent single-hop support annotation")
    try:
        task, reference = adapt_musique(row)
    except (ValueError, TypeError) as error:
        raise PanelError("row does not match the official MuSiQue schema") from error
    # Annotation completeness was established above, never inferred from absence.
    reference["support_annotation_available"] = True
    return task, reference


def _grounded_subquestions(decomposition):
    """Identity hashes only: annotations never become runtime source text.

    Natural-language #N can be literal (the upstream formatter has exceptions).
    Ambiguous/non-prior references therefore omit only this extra text edge;
    identity, single-hop and support edges and the original row remain intact.
    """
    texts, unresolved = [], []
    for position, item in enumerate(decomposition):
        question = item["question"]
        matches = list(re.finditer(r"#(\d+)\b", question))
        indices = [int(match.group(1)) for match in matches]
        if any(index < 1 or index > position for index in indices):
            unresolved.append("nonprior_or_literal_number_reference")
            continue
        if any(not decomposition[index - 1]["answer"].strip() for index in indices):
            unresolved.append("empty_bound_answer")
            continue
        texts.append(re.sub(r"#(\d+)\b", lambda match: decomposition[int(match.group(1)) - 1]["answer"], question))
    return texts, tuple(unresolved)


def _candidate(row, task, reference, line, split, seed, *, grouped=False):
    tokens = {kind: set() for kind in FEATURES}
    tokens["question_id"].add(task["question_id"])
    tokens["source_question_pair"].add(reference["pair_group_id"])
    tokens["normalized_question"].add(_hash(_normalized(task["question"])))
    for item in row["question_decomposition"]:
        tokens["singlehop_id"].add(_hash(_id(item["id"])))
        answer = normalize_answer(unicodedata.normalize("NFKC", item["answer"]))
        if answer:
            tokens["subanswer"].add(_hash(answer))
    for paragraph in row["paragraphs"]:
        if paragraph["is_supporting"]:
            tokens["support_paragraph"].add(_hash(_normalized(paragraph["paragraph_text"])))
    unresolved = ()
    if grouped:
        texts, unresolved = _grounded_subquestions(row["question_decomposition"])
        tokens["normalized_subquestion"] = {_hash(_normalized(text)) for text in texts}
        tokens["raw_normalized_subquestion"] = {_hash(_normalized(item["question"])) for item in row["question_decomposition"]}
    order = _hash(json.dumps([GROUPED_SCHEMA if grouped else SCHEMA, seed, split, reference["pair_group_id"], task["question_id"]],
                             ensure_ascii=False, separators=(",", ":")))
    return Candidate(line, task["question_id"], reference["pair_group_id"],
                     len(row["question_decomposition"]),
                     {kind: frozenset(values) for kind, values in tokens.items()}, order, unresolved)


def _scan(path, split, seed, *, grouped=False):
    path = Path(path).resolve(strict=True)
    if path.name != "musique_ans_v1.0_" + split + ".jsonl" or not path.is_file():
        raise PanelError("use the explicitly specified official v1.0 train/dev JSONL filenames")
    digest = hashlib.sha256()
    candidates = []
    identities = {}
    with path.open("rb") as stream:
        for line, raw in enumerate(stream, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                row = _json(raw)
                task, reference = _adapt(row)
                fingerprint = _hash(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                prior = identities.get(row["id"])
                if prior is not None and prior != fingerprint:
                    raise PanelError("conflicting rows share one official answerable source id")
                identities[row["id"]] = fingerprint
                candidates.append(_candidate(row, task, reference, line, split, seed, grouped=grouped))
            except PanelError as error:
                raise PanelError(f"invalid {split} row at line {line}: {error}") from error
    if not candidates:
        raise PanelError("input split is empty")
    candidates.sort(key=lambda c: (c.order_key, c.question_id, c.line))
    return candidates, {"path": str(path), "sha256": digest.hexdigest(),
                        "rows": len(candidates), "hop_counts": dict(sorted(Counter(c.hop for c in candidates).items()))}


def _quotas(quotas):
    if not isinstance(quotas, dict) or set(quotas) != set(ROLES):
        raise PanelError("explicit quotas for all three roles are required")
    clean = {}
    for role in ROLES:
        q = quotas[role]
        if (not isinstance(q, dict) or set(q) != {2, 3, 4}
                or any(type(n) is not int or n < 0 for n in q.values()) or sum(q.values()) < 1):
            raise PanelError("each role needs nonnegative integer 2/3/4-hop quotas and a positive total")
        clean[role] = dict(q)
    return clean


def _materialize(source, selected):
    wanted = {candidate.line: (role, candidate) for role, candidates in selected.items()
              for candidate in candidates}
    material = {}
    digest = hashlib.sha256()
    with Path(source["path"]).open("rb") as stream:
        for line, raw in enumerate(stream, 1):
            digest.update(raw)
            if line not in wanted:
                continue
            role, candidate = wanted[line]
            task, reference = _adapt(_json(raw))
            if task["question_id"] != candidate.question_id:
                raise PanelError("input identity changed during preparation")
            material[(role, line)] = (task, reference)
    if digest.hexdigest() != source["sha256"] or len(material) != len(wanted):
        raise PanelError("input bytes changed during preparation")
    return material


def prepare_panels(train_path, dev_path, *, seed, quotas):
    """Read user-specified local inputs and return an unpublished complete bundle.

    Report sampling is independent of train selection. The builder never checks
    model predictions or scores. Private collision tokens never enter manifests.
    """
    if not isinstance(seed, str) or not seed.strip():
        raise PanelError("an explicit nonempty string seed is required")
    quotas = _quotas(quotas)
    train, train_info = _scan(train_path, "train", seed)
    dev, dev_info = _scan(dev_path, "dev", seed)
    sources = {"train": train_info, "dev": dev_info}
    selected = {role: [] for role in ROLES}
    indices = {role: {kind: {} for kind in FEATURES} for role in ROLES}
    audits = {}
    manifest = {"schema": SCHEMA, "status": "preparing", "dataset": "MuSiQue-Ans",
                "data_version": "v1.0", "upstream_code_revision": REVISION,
                "source_authenticity": "user_supplied_files; schema_checked; not_upstream_checksum_verified",
                "seed": seed, "quotas": quotas, "sources": sources,
                "role_sources": {"D_fit": "train", "D_select": "train", "D_report": "dev"},
                "selection_order": list(SELECTION_ORDER),
                "ordering": "sha256(JSON([schema,seed,split,source_id,public_question_id])); ascending",
                "normalization": {"question_and_support_text": "Unicode-NFKC + casefold + whitespace-collapse",
                                  "subanswer": "Unicode-NFKC + official MuSiQue answer normalization; skip normalized empty",
                                  "singlehop_id": "string identity; integer and matching string compare equal"},
                "filters": list(FEATURES), "audit": audits,
                "private_fields_used_only_locally": True, "model_scores_used": False,
                "local_corpus_paragraph_limit": 20,
                "original_paragraph_count_preserved": True,
                "document_rendering": MUSIQUE_DOCUMENT_RENDERING,
                "panel_claim": "frozen stratified pilot subset; not the full official benchmark"}
    for role in SELECTION_ORDER:
        candidates = dev if role == "D_report" else train
        counts = {2: 0, 3: 0, 4: 0}
        hits, quota_skips = Counter(), Counter()
        rejected = []
        visited = 0
        for candidate in candidates:
            if counts == quotas[role]:
                break
            visited += 1
            if counts[candidate.hop] >= quotas[role][candidate.hop]:
                quota_skips[candidate.hop] += 1
                continue
            conflicts = []
            for other in SELECTION_ORDER:
                for kind in FEATURES:
                    # Shared subquestions within a training role are permitted;
                    # exact duplicate questions/source identities are not.
                    if other == role and kind not in FEATURES[:3]:
                        continue
                    matching = sorted({indices[other][kind][token] for token in candidate.features[kind]
                                       if token in indices[other][kind]})
                    if matching:
                        name = ("within_role_" if other == role else "cross_role_") + kind
                        hits[name] += 1
                        conflicts.append({"reason": name, "other_role": other,
                                          "matched_question_ids": matching})
            if conflicts:
                rejected.append({"source_line": candidate.line, "question_id": candidate.question_id,
                                 "source_question_id": candidate.source_id, "hop": candidate.hop,
                                 "conflicts": conflicts})
                continue
            selected[role].append(candidate)
            counts[candidate.hop] += 1
            for kind, tokens in candidate.features.items():
                for token in tokens:
                    indices[role][kind].setdefault(token, candidate.question_id)
        audits[role] = {"requested": quotas[role], "selected": counts,
                        "selected_question_ids": [c.question_id for c in selected[role]],
                        "visited": visited, "unvisited": len(candidates) - visited,
                        "quota_skips": dict(sorted(quota_skips.items())),
                        "filter_hits": dict(sorted(hits.items())),
                        "rejected_count": len(rejected), "rejected_candidates": rejected}
        if counts != quotas[role]:
            manifest.update(status="quota_shortfall", failed_role=role)
            raise PanelQuotaError(manifest)
    material = {}
    material.update(_materialize(train_info, {r: selected[r] for r in ("D_fit", "D_select")}))
    material.update(_materialize(dev_info, {"D_report": selected["D_report"]}))
    panels, references = {}, {}
    for role in ROLES:
        pairs = [material[(role, c.line)] for c in selected[role]]
        panels[role] = [task for task, _ in pairs]
        validate_task_collection(panels[role])
        references[role] = {task["question_id"]: ref for task, ref in pairs}
    manifest["paragraph_count_distribution"] = {
        role: dict(sorted(Counter(len(task["documents"]) for task in panels[role]).items()))
        for role in ROLES}
    manifest["status"] = "ready"
    return {"public_panels": panels, "private_references": references, "manifest": manifest}


def _group_hash(value):
    return _hash(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _components(rows, relations=GROUP_RELATIONS):
    parents = list(range(len(rows)))
    owners = {}
    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    for index, (_, candidate) in enumerate(rows):
        for kind in relations:
            for token in candidate.features[kind]:
                key = (kind, token)
                if key in owners:
                    parents[root(index)] = root(owners[key])
                else:
                    owners[key] = index
    result = {}
    for index, row in enumerate(rows):
        result.setdefault(root(index), []).append(row)
    return list(result.values())


def _component_summary(components):
    sizes = sorted((len(component) for component in components), reverse=True)
    return {"component_count": len(components), "row_count": sum(sizes),
            "size_distribution": dict(sorted(Counter(sizes).items())),
            "largest_sizes": sizes[:20], "maximum_size": max(sizes, default=0),
            "cross_split_components": sum(len({split for split, _ in component}) > 1 for component in components),
            "known_relations_exhaust_all_dependencies": False}


def _history(value, sources, rows):
    value = deepcopy(value)
    binding = None
    if isinstance(value, (str, Path)):
        path = Path(value).resolve(strict=True)
        raw = path.read_bytes()
        value = _json(raw)
        binding = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    if (not isinstance(value, dict) or set(value) != {"schema", "source_files", "entries"}
            or value["schema"] != HISTORY_SCHEMA or not isinstance(value["source_files"], dict)
            or set(value["source_files"]) != {"train", "dev"} or not isinstance(value["entries"], list)):
        raise PanelError("explicit, closed exposure-history contract required")
    for split in ("train", "dev"):
        item = value["source_files"][split]
        if (not isinstance(item, dict) or set(item) != {"path", "sha256"}
                or not isinstance(item["path"], str) or not isinstance(item["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
                or Path(item["path"]).resolve(strict=True) != Path(sources[split]["path"])
                or item["sha256"] != sources[split]["sha256"]):
            raise PanelError("history source path or SHA256 differs from scanned source")
    locations = {}
    for split, candidate in rows:
        locations.setdefault(candidate.question_id, set()).add(split)
    entries = {}
    for entry in value["entries"]:
        if (not isinstance(entry, dict) or set(entry) != {"question_id", "role", "status"}
                or not isinstance(entry["question_id"], str) or not entry["question_id"]
                or not isinstance(entry["role"], str) or entry["role"] not in ROLES
                or not isinstance(entry["status"], str) or entry["status"] not in {"exposed", "reserved"}):
            raise PanelError("invalid history question, role or exposure status")
        qid = entry["question_id"]
        split = "dev" if entry["role"] == "D_report" else "train"
        if qid not in locations or split not in locations[qid]:
            raise PanelError("history question is absent or incompatible with its preserved role")
        if qid in entries:
            raise PanelError("duplicate or conflicting history identity")
        entries[qid] = dict(entry)
    return entries, {"schema": HISTORY_SCHEMA, "contract_sha256": _group_hash(value),
                     "file": binding, "entry_count": len(entries),
                     "coverage": "explicit_history_only; completeness_not_independently_established"}


def prepare_grouped_panels(train_path, dev_path, *, seed, quotas, history):
    """Schema 3 samples the observed atomic-disjoint panel, never its scores.

    Unselected bridge questions do not propagate exposure. All direct atomic
    overlaps with explicit history and between selected questions are blocked.
    Whole-corpus closure remains a conservative sensitivity diagnostic only.
    """
    if not isinstance(seed, str) or not seed.strip():
        raise PanelError("an explicit nonempty string seed is required")
    quotas = _quotas(quotas)
    train, train_info = _scan(train_path, "train", seed, grouped=True)
    dev, dev_info = _scan(dev_path, "dev", seed, grouped=True)
    sources = {"train": train_info, "dev": dev_info}
    rows = [("train", candidate) for candidate in train] + [("dev", candidate) for candidate in dev]
    history_entries, history_contract = _history(history, sources, rows)
    by_question = {}
    for split, candidate in rows:
        by_question.setdefault((split, candidate.question_id), candidate)
    history_rows = [("dev" if entry["role"] == "D_report" else "train",
                     by_question[("dev" if entry["role"] == "D_report" else "train", qid)])
                    for qid, entry in sorted(history_entries.items())]
    history_atoms = {kind: {} for kind in GROUP_RELATIONS}
    for _, candidate in history_rows:
        for kind in GROUP_RELATIONS:
            for token in candidate.features[kind]:
                history_atoms[kind].setdefault(token, set()).add(candidate.question_id)
    def matches(candidate, atoms):
        return {kind: sorted({qid for token in candidate.features[kind] for qid in atoms[kind].get(token, ())})
                for kind in GROUP_RELATIONS if any(token in atoms[kind] for token in candidate.features[kind])}
    direct_history = {(split, candidate.line): matches(candidate, history_atoms) for split, candidate in rows}
    direct_reserved = []
    for _, candidate in history_rows:
        entry = history_entries[candidate.question_id]
        if entry["status"] != "reserved":
            continue
        overlap = matches(candidate, history_atoms)
        exposed = sorted({qid for qids in overlap.values() for qid in qids if history_entries[qid]["status"] == "exposed"})
        if exposed:
            direct_reserved.append({**entry, "direct_exposed_question_ids": exposed,
                "relations": sorted(kind for kind, qids in overlap.items() if set(qids) & set(exposed))})
    # This statistic intentionally includes unused bridge rows. It is not the
    # exposure decision and must not be described as proven information leakage.
    components = _components(rows)
    base_relations = tuple(kind for kind in GROUP_RELATIONS if kind != "normalized_subquestion")
    raw_components = _components(rows, base_relations + ("raw_normalized_subquestion",))
    family_links = []
    excluded_families = []
    for component in components:
        hits = sorted({candidate.question_id for _, candidate in component} & set(history_entries))
        entries = [history_entries[qid] for qid in hits]
        if entries:
            excluded_families.append(component)
        if any(entry["status"] == "exposed" for entry in entries):
            family_links.extend(dict(entry) for entry in entries if entry["status"] == "reserved")
    sensitivity = {"scope": "full_source_family_closure_including_unobserved_bridge_questions",
        "used_for_selection": False, "is_evidence_of_actual_leakage": False,
        "summary": _component_summary(components),
        "history_connected_component_count": len(excluded_families),
        "history_connected_row_count": sum(map(len, excluded_families)),
        "reserved_hypothetical_family_connections_to_exposed": family_links}
    available = [(split, candidate) for split, candidate in rows if not direct_history[(split, candidate.line)]]
    audit_rows = {(split, candidate.line): {"source_split": split, "source_line": candidate.line,
        "question_id": candidate.question_id, "hop": candidate.hop,
        "direct_history_matches": direct_history[(split, candidate.line)], "role_decisions": {}}
        for split, candidate in rows}
    ordering_version = "rag-rsi-musique-observed-atomic-disjoint-1"
    ordered = sorted(rows, key=lambda item: (_group_hash([ordering_version, seed, item[0], item[1].source_id, item[1].question_id]), item[1].question_id, item[1].line))
    chosen_atoms = {kind: {} for kind in GROUP_RELATIONS}
    selected = {role: [] for role in ROLES}
    assignments = {}
    audits = {}
    for role in SELECTION_ORDER:
        split = "dev" if role == "D_report" else "train"
        counts = {2: 0, 3: 0, 4: 0}
        decisions = Counter()
        for source, candidate in ordered:
            record = audit_rows[(source, candidate.line)]
            overlap = matches(candidate, chosen_atoms)
            reasons = []
            if source != split:
                reasons.append("source_split_unavailable")
            if record["direct_history_matches"]:
                reasons.extend("direct_history_" + kind for kind in record["direct_history_matches"])
            if overlap:
                reasons.extend("selected_atomic_overlap_" + kind for kind in overlap)
            if counts[candidate.hop] >= quotas[role][candidate.hop]:
                reasons.append("hop_quota_filled")
            if not reasons:
                reasons = ["selected"]
                assignments[candidate.question_id] = role
                selected[role].append(candidate)
                counts[candidate.hop] += 1
                for kind in GROUP_RELATIONS:
                    for token in candidate.features[kind]:
                        chosen_atoms[kind].setdefault(token, set()).add(candidate.question_id)
            record["role_decisions"][role] = {"reasons": reasons, "selected_overlap_question_ids": overlap}
            decisions.update(reasons)
        audits[role] = {"requested": quotas[role], "selected": counts,
            "selected_question_ids": [candidate.question_id for candidate in selected[role]],
            "decision_counts": dict(sorted(decisions.items()))}
    # Closure is checked on actual history and selected samples. Historical
    # relatedness can remain, but every new sample must form its own component.
    observed = history_rows + [("dev" if role == "D_report" else "train", candidate)
                               for role in ROLES for candidate in selected[role]]
    groups, question_groups = {}, {}
    for component in _components(observed):
        identities = sorted((split, candidate.question_id) for split, candidate in component)
        group_id = "musique_dependency_" + _group_hash(identities)
        new_ids = [candidate.question_id for _, candidate in component if candidate.question_id in assignments]
        if new_ids and (len(component) != 1 or len(new_ids) != 1):
            raise PanelError("observed panel still shares an atomic dependency")
        question_groups.update({qid: group_id for qid in new_ids})
        groups[group_id] = {"members": [{"source_split": split, "question_id": candidate.question_id,
            "source_line": candidate.line, "hop": candidate.hop,
            "role": assignments.get(candidate.question_id, history_entries.get(candidate.question_id, {}).get("role")),
            "history_status": history_entries.get(candidate.question_id, {}).get("status")}
            for split, candidate in component], "row_count": len(component),
            "contains_selected_question": bool(new_ids)}
    unresolved = Counter(reason for _, candidate in rows for reason in candidate.unresolved_text)
    manifest = {"schema": GROUPED_SCHEMA, "status": "ready", "dataset": "MuSiQue-Ans", "data_version": "v1.0",
        "upstream_code_revision": REVISION, "source_authenticity": "user_supplied_files; schema_checked; not_upstream_checksum_verified",
        "sampling_policy": "known_atomic_disjoint_observed_panel_v1", "seed": seed, "quotas": quotas,
        "quota_unit": "one_question_per_observed_known_dependence_component",
        "sources": sources, "role_sources": {"D_fit": "train", "D_select": "train", "D_report": "dev"},
        "selection_order": list(SELECTION_ORDER), "history": history_contract,
        "dependency_relations": list(GROUP_RELATIONS),
        "normalization": "Unicode-NFKC + casefold + whitespace-collapse; #N replaced by prior annotated answer only when all numeric references resolve",
        "ambiguous_number_reference_policy": "omit_only_extra_subquestion_text_edge; retain_row_and_all_other_edges",
        "text_identity_unresolved_count": sum(unresolved.values()), "text_identity_unresolved_reasons": dict(unresolved),
        "raw_template_rule_rejected": True, "raw_template_sensitivity_only": _component_summary(raw_components),
        "subanswer_policy": "not_a_dependency_edge_or_extra_cross_role_filter_in_schema3; schema2_unchanged",
        "component_rule": "direct_history_atomic_exclusion_then_pairwise_disjoint_selected_samples; observed_set_transitive_closure_verified",
        "candidate_order": "sha256(JSON([ordering_namespace,seed,split,source_id,question_id])); ascending",
        "ordering_namespace": ordering_version, "known_relations_exhaust_all_dependencies": False,
        "full_source_family_sensitivity": sensitivity,
        "reserved_history_directly_associated_with_exposure": direct_reserved,
        "association_is_not_proof_of_actual_answer_leakage": True,
        "history_directly_excluded_rows": len(rows) - len(available),
        "after_direct_history_exclusion_hop_counts": {split: dict(sorted(Counter(candidate.hop for source, candidate in available if source == split).items())) for split in ("train", "dev")},
        "audit": audits, "private_fields_used_only_locally": True, "model_scores_used": False,
        "document_rendering": MUSIQUE_DOCUMENT_RENDERING, "original_paragraph_count_preserved": True,
        "panel_claim": "known-atomic-disjoint observed sample relative to explicit history; not full potential-family isolation or guaranteed independence"}
    group_record = {"schema": "rag-rsi-musique-dependency-groups-1", "panel_schema": GROUPED_SCHEMA,
        "sampling_policy": manifest["sampling_policy"], "history_contract_sha256": history_contract["contract_sha256"],
        "groups": groups, "question_groups": question_groups,
        "candidate_audit": [audit_rows[(split, candidate.line)] for split, candidate in rows],
        "contains_annotation_text": False}
    if history_contract["file"]:
        binding = history_contract["file"]
        if hashlib.sha256(Path(binding["path"]).read_bytes()).hexdigest() != binding["sha256"]:
            raise PanelError("history file changed during preparation")
    shortfalls = [role for role in ROLES if audits[role]["selected"] != quotas[role]]
    if shortfalls:
        manifest.update(status="quota_shortfall", failed_role=shortfalls[0], failed_roles=shortfalls)
        raise PanelQuotaError(manifest, group_record)
    # Freeze all role assignments before materializing selected annotations.
    material = _materialize(train_info, {role: selected[role] for role in ("D_fit", "D_select")})
    material.update(_materialize(dev_info, {"D_report": selected["D_report"]}))
    panels, references = {}, {}
    for role in ROLES:
        pairs = [material[(role, candidate.line)] for candidate in selected[role]]
        panels[role] = [task for task, _ in pairs]
        validate_task_collection(panels[role])
        references[role] = {task["question_id"]: ref for task, ref in pairs}
    if history_contract["file"]:
        binding = history_contract["file"]
        if hashlib.sha256(Path(binding["path"]).read_bytes()).hexdigest() != binding["sha256"]:
            raise PanelError("history file changed during preparation")
    manifest["paragraph_count_distribution"] = {role: dict(sorted(Counter(len(task["documents"]) for task in panels[role]).items())) for role in ROLES}
    return {"public_panels": panels, "private_references": references, "manifest": manifest, "groups": group_record}


def _encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _output_path(output_dir):
    output = Path(output_dir).resolve()
    root = RUNS_ROOT.resolve()
    if root not in output.parents:
        raise PanelError("output must be a new child directory under this project's runs directory")
    if output.exists():
        raise PanelError("output already exists; frozen panels are never overwritten")
    return output


def _validate_grouped_publication(bundle):
    """Recheck the schema3 publication boundary before creating any files.

    Preparation and publication are separate callable steps. A ready label must
    not hide a truncated role or inconsistent task/reference/group handoff.
    This validates the existing preparation contract, without resampling.
    """
    def require(condition, reason):
        if not condition:
            raise PanelError("schema3 publication mismatch: " + reason)

    require(isinstance(bundle, dict) and set(bundle) ==
            {"public_panels", "private_references", "manifest", "groups"}, "bundle fields")
    manifest, groups = bundle["manifest"], bundle["groups"]
    quotas = _quotas(manifest.get("quotas"))
    role_sources = {"D_fit": "train", "D_select": "train", "D_report": "dev"}
    require(manifest.get("role_sources") == role_sources, "role sources")
    require(isinstance(manifest.get("audit"), dict) and set(manifest["audit"]) == set(ROLES), "role audit")
    for field in ("public_panels", "private_references"):
        require(isinstance(bundle[field], dict) and set(bundle[field]) == set(ROLES), "complete role set")
    selected = {}
    for role in ROLES:
        tasks, references = bundle["public_panels"][role], bundle["private_references"][role]
        try:
            validate_task_collection(tasks)
        except (ValueError, TypeError) as error:
            raise PanelError("schema3 publication mismatch: public task contract") from error
        identities = [task["question_id"] for task in tasks]
        require(isinstance(references, dict) and set(references) == set(identities), "reference coverage")
        counts = {2: 0, 3: 0, 4: 0}
        for task in tasks:
            qid, reference = task["question_id"], references[task["question_id"]]
            require(qid not in selected, "question appears in more than one role")
            require(task["dataset"] == "musique" and task["task_type"] == "qa"
                    and task["corpus_scope"] == "question_local", "public task kind")
            require(isinstance(reference, dict) and reference.get("question_id") == qid
                    and reference.get("dataset") == "musique"
                    and reference.get("reference_available") is True
                    and reference.get("answerable") is True
                    and reference.get("support_annotation_available") is True, "reference identity")
            answers = reference.get("answers")
            require(isinstance(answers, list) and bool(answers)
                    and all(isinstance(answer, str) and answer.strip() for answer in answers), "reference answers missing")
            decomposition = reference.get("question_decomposition")
            require(isinstance(decomposition, list) and len(decomposition) in (2, 3, 4), "reference hop annotation")
            counts[len(decomposition)] += 1
            selected[qid] = {"role": role, "source_split": role_sources[role], "hop": len(decomposition)}
        audit = manifest["audit"][role]
        require(isinstance(audit, dict) and audit.get("requested") == quotas[role]
                and audit.get("selected") == quotas[role] and counts == quotas[role]
                and audit.get("selected_question_ids") == identities, "quota or selected question order")
    group_fields = {"schema", "panel_schema", "sampling_policy", "history_contract_sha256",
                    "groups", "question_groups", "candidate_audit", "contains_annotation_text"}
    require(isinstance(groups, dict) and set(groups) == group_fields
            and groups["schema"] == "rag-rsi-musique-dependency-groups-1"
            and groups["panel_schema"] == GROUPED_SCHEMA
            and groups["sampling_policy"] == manifest.get("sampling_policy")
            and isinstance(manifest.get("history"), dict)
            and groups["history_contract_sha256"] == manifest["history"].get("contract_sha256")
            and groups["contains_annotation_text"] is False, "dependency audit identity")
    mapping, records = groups["question_groups"], groups["groups"]
    require(isinstance(mapping, dict) and set(mapping) == set(selected)
            and all(isinstance(value, str) and value for value in mapping.values()), "question group coverage")
    require(len(set(mapping.values())) == len(mapping), "selected questions share a group")
    require(isinstance(records, dict) and set(mapping.values()) <= set(records), "selected groups missing")
    observed, chosen_members = set(), {}
    member_fields = {"source_split", "question_id", "source_line", "hop", "role", "history_status"}
    for gid, record in records.items():
        require(isinstance(record, dict) and set(record) ==
                {"members", "row_count", "contains_selected_question"}, "group fields")
        members = record["members"]
        require(isinstance(members, list) and bool(members) and type(record["row_count"]) is int
                and record["row_count"] == len(members), "group row count")
        new = []
        for member in members:
            require(isinstance(member, dict) and set(member) == member_fields
                    and isinstance(member["question_id"], str) and bool(member["question_id"])
                    and member["role"] in ROLES and member["source_split"] == role_sources[member["role"]]
                    and type(member["source_line"]) is int and member["source_line"] > 0
                    and type(member["hop"]) is int and member["hop"] in (2, 3, 4), "group member")
            qid = member["question_id"]
            require(qid not in observed, "repeated group member")
            observed.add(qid)
            if qid in selected:
                require(mapping[qid] == gid and member["history_status"] is None
                        and all(member[key] == value for key, value in selected[qid].items()), "selected group member")
                new.append(qid)
                chosen_members[qid] = member
            else:
                require(member["history_status"] in ("exposed", "reserved"), "history member status")
        expected_gid = "musique_dependency_" + _group_hash(sorted(
            (member["source_split"], member["question_id"]) for member in members))
        require(gid == expected_gid and record["contains_selected_question"] is bool(new), "group binding")
        require(not new or len(members) == len(new) == 1, "selected group is not atomic-disjoint")
    require(set(chosen_members) == set(selected), "selected member coverage")
    audits = groups["candidate_audit"]
    require(isinstance(audits, list), "candidate audit missing")
    selected_rows, positions, split_counts = {}, set(), Counter()
    for row in audits:
        require(isinstance(row, dict) and set(row) == {"source_split", "source_line", "question_id",
                "hop", "direct_history_matches", "role_decisions"}, "candidate audit fields")
        split, line = row["source_split"], row["source_line"]
        require(split in ("train", "dev") and type(line) is int and line > 0
                and (split, line) not in positions, "candidate audit position")
        positions.add((split, line)); split_counts[split] += 1
        decisions = row["role_decisions"]
        require(isinstance(decisions, dict) and set(decisions) == set(ROLES), "candidate role decisions")
        for role, decision in decisions.items():
            require(isinstance(decision, dict) and set(decision) ==
                    {"reasons", "selected_overlap_question_ids"}
                    and isinstance(decision["reasons"], list), "candidate decision fields")
            if "selected" not in decision["reasons"]:
                continue
            qid = row["question_id"]
            require(isinstance(qid, str) and qid in selected and qid not in selected_rows
                    and selected[qid]["role"] == role and decision["reasons"] == ["selected"]
                    and decision["selected_overlap_question_ids"] == {} and row["direct_history_matches"] == {}
                    and all(row[key] == chosen_members[qid][key] for key in
                            ("source_split", "source_line", "question_id", "hop")), "selected candidate audit")
            selected_rows[qid] = role
    require(set(selected_rows) == set(selected), "selected audit coverage")
    sources = manifest.get("sources")
    require(isinstance(sources, dict) and set(sources) == {"train", "dev"}
            and all(isinstance(sources[split], dict) and type(sources[split].get("rows")) is int
                    and sources[split]["rows"] == split_counts[split] for split in ("train", "dev")), "full candidate audit coverage")


def write_panels(bundle, output_dir):
    """Publish only a complete bundle; the completion manifest is written last."""
    output = _output_path(output_dir)
    manifest = dict(bundle["manifest"])
    if manifest.get("status") != "ready":
        raise PanelError("cannot publish an incomplete panel bundle")
    if manifest.get("schema") == GROUPED_SCHEMA:
        _validate_grouped_publication(bundle)
    files = {}
    for role in ROLES:
        validate_task_collection(bundle["public_panels"][role])
        for name, value in (("public_tasks", bundle["public_panels"][role]),
                            ("private_references", bundle["private_references"][role])):
            relative = role + "/" + name + ".json"
            files[relative] = _encoded(value)
    if manifest.get("schema") == GROUPED_SCHEMA:
        groups = bundle.get("groups")
        if not isinstance(groups, dict) or groups.get("schema") != "rag-rsi-musique-dependency-groups-1":
            raise PanelError("schema3 requires its dependency-group audit")
        files["groups.json"] = _encoded(groups)
    manifest["files"] = {name: {"sha256": hashlib.sha256(raw).hexdigest(),
                                "private": name.endswith("private_references.json") or name == "groups.json"}
                         for name, raw in files.items()}
    # Serialize everything before creating the directory; quota/input failures
    # never leave public task files. A missing manifest means an incomplete write.
    manifest_bytes = _encoded(manifest)
    output.mkdir(parents=True, exist_ok=False)
    for relative, raw in files.items():
        target = output / relative
        target.parent.mkdir(exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
    with (output / "manifest.json").open("xb") as stream:
        stream.write(manifest_bytes)
    return output


def _parse_quota(value):
    try:
        values = [int(part) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("quota must be three integers: 2hop,3hop,4hop") from error
    if len(values) != 3 or any(n < 0 for n in values) or not sum(values):
        raise argparse.ArgumentTypeError("quota needs three nonnegative integers with positive total")
    return dict(zip((2, 3, 4), values))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--history", help="Explicit schema3 exposure-history JSON; omission preserves schema2")
    parser.add_argument("--fit-quota", type=_parse_quota, default=_parse_quota("16,8,8"))
    parser.add_argument("--select-quota", type=_parse_quota, default=_parse_quota("8,4,4"))
    parser.add_argument("--report-quota", type=_parse_quota, default=_parse_quota("16,8,8"))
    args = parser.parse_args(argv)
    try:
        output = _output_path(args.out)
        q = {"D_fit": args.fit_quota, "D_select": args.select_quota, "D_report": args.report_quota}
        bundle = (prepare_grouped_panels(args.train, args.dev, seed=args.seed, quotas=q, history=args.history)
                  if args.history else prepare_panels(args.train, args.dev, seed=args.seed, quotas=q))
        write_panels(bundle, output)
    except PanelQuotaError as error:
        # Preserve the rejection audit, but publish no runnable panel on failure.
        output.mkdir(parents=True, exist_ok=False)
        if error.groups is not None:
            raw = _encoded(error.groups)
            with (output / "groups.json").open("xb") as stream:
                stream.write(raw)
            error.manifest["files"] = {"groups.json": {"sha256": hashlib.sha256(raw).hexdigest(), "private": True}}
        with (output / "manifest.json").open("xb") as stream:
            stream.write(_encoded(error.manifest))
        print(json.dumps({"status": "quota_shortfall", "failed_role": error.manifest["failed_role"],
                          "audit_manifest": str(output / "manifest.json")}), file=sys.stderr)
        return 2
    except (PanelError, OSError) as error:
        # Avoid OS paths/messages accidentally including example contents.
        message = str(error) if isinstance(error, PanelError) else "local input/output failure"
        print(json.dumps({"status": "error", "reason": message}), file=sys.stderr)
        return 2
    print(json.dumps({"status": "ready", "manifest": str(output / "manifest.json"),
                      "counts": {role: len(bundle["public_panels"][role]) for role in ROLES}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
