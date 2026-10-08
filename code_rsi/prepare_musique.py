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
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

from .v3.datasets import adapt_musique, normalize_answer, validate_task_collection

SCHEMA = "rag-rsi-musique-ans-panels-1"
REVISION = "922ac98f19a201998dbdae6d7f2887a5258dbdeb"
RUNS_ROOT = Path(__file__).resolve().parents[1] / "runs"
ROLES = ("D_fit", "D_select", "D_report")
SELECTION_ORDER = ("D_report", "D_fit", "D_select")
FEATURES = ("question_id", "source_question_pair", "normalized_question",
            "singlehop_id", "support_paragraph", "subanswer")


class PanelError(ValueError):
    """Unsafe input or output contract; messages never contain example text."""


class PanelQuotaError(PanelError):
    def __init__(self, manifest):
        super().__init__("requested panel quotas are unmet under the frozen sampling policy")
        self.manifest = manifest


@dataclass(frozen=True)
class Candidate:
    line: int
    question_id: str
    source_id: str
    hop: int
    features: dict[str, frozenset[str]]
    order_key: str


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
    if (not isinstance(paragraphs, list) or len(paragraphs) != 20
            or any(not isinstance(p, dict) or type(p.get("idx")) is not int
                   or type(p.get("is_supporting")) is not bool for p in paragraphs)
            or {p["idx"] for p in paragraphs} != set(range(20))):
        raise PanelError("twenty indexed paragraphs with complete support annotations are required")
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


def _candidate(row, task, reference, line, split, seed):
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
    order = _hash(json.dumps([SCHEMA, seed, split, reference["pair_group_id"], task["question_id"]],
                             ensure_ascii=False, separators=(",", ":")))
    return Candidate(line, task["question_id"], reference["pair_group_id"],
                     len(row["question_decomposition"]),
                     {kind: frozenset(values) for kind, values in tokens.items()}, order)


def _scan(path, split, seed):
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
                candidates.append(_candidate(row, task, reference, line, split, seed))
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
                "local_corpus_paragraphs_per_task": 20,
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
    manifest["status"] = "ready"
    return {"public_panels": panels, "private_references": references, "manifest": manifest}


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


def write_panels(bundle, output_dir):
    """Publish only a complete bundle; the completion manifest is written last."""
    output = _output_path(output_dir)
    manifest = dict(bundle["manifest"])
    if manifest.get("status") != "ready":
        raise PanelError("cannot publish an incomplete panel bundle")
    files = {}
    for role in ROLES:
        validate_task_collection(bundle["public_panels"][role])
        for name, value in (("public_tasks", bundle["public_panels"][role]),
                            ("private_references", bundle["private_references"][role])):
            relative = role + "/" + name + ".json"
            files[relative] = _encoded(value)
    manifest["files"] = {name: {"sha256": hashlib.sha256(raw).hexdigest(),
                                "private": name.endswith("private_references.json")}
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
    parser.add_argument("--fit-quota", type=_parse_quota, default=_parse_quota("16,8,8"))
    parser.add_argument("--select-quota", type=_parse_quota, default=_parse_quota("8,4,4"))
    parser.add_argument("--report-quota", type=_parse_quota, default=_parse_quota("16,8,8"))
    args = parser.parse_args(argv)
    try:
        output = _output_path(args.out)
        bundle = prepare_panels(args.train, args.dev, seed=args.seed,
                                quotas={"D_fit": args.fit_quota, "D_select": args.select_quota,
                                        "D_report": args.report_quota})
        write_panels(bundle, output)
    except PanelQuotaError as error:
        # Preserve the rejection audit, but publish no runnable panel on failure.
        output.mkdir(parents=True, exist_ok=False)
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
