"""Conservative static audit against D_fit text already visible to development.

No code is executed and no references are read. A pass is not a proof against
hardcoding: computed/encoded text, aliases, semantic hints and short strings can
escape these deliberately narrow rules. This is not a replacement for isolation
or held-out evaluation. Call only with the actual exposed fit feedback payload.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
from collections.abc import Mapping
import hashlib
import json
import re
import unicodedata

SCHEMA = "rag-rsi-fit-literal-audit-1"
RULE_VERSION = "fit-literals-1"
RULES = {"full_question_min_alnum": 12, "question_fragment_chars": 64,
         "prediction_fragment_chars": 48, "evidence_fragment_chars": 96,
         "specific_min_non_generic_words": 3, "specific_min_cjk_chars": 20,
         "long_id_min_chars": 12, "long_id_min_distinct_alnum": 6,
         "normalization": "NFKC, casefold, remove Unicode format characters, collapse whitespace; retain punctuation"}
ID_FIELDS = {"question_id", "task_id", "query_id", "qid"}
INPUT_NAMES = {"task", "question", "request", "payload", "input", "public_task"}
SCHEMA_WORDS = ID_FIELDS | {"id", "answer", "answers", "question", "instructions", "citation_ids",
                           "evidence", "sources", "score", "role", "stage", "type"}
GENERIC_WORDS = set(("a an and are as at be because been before by can cannot could determine do does "
    "each evidence find for from give has have how if in information insufficient into is it its may "
    "missing no not of on only or output please provide provided question questions read response "
    "return search should source sources support that the their them there these they this to unknown "
    "use using was were what when where which who why will with without would answer answers "
    "available based cite concise exact json format complete relevant following identify determine "
    "need needed enough unable tell know data text details additional result results found").split())


def _hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize(value):
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join("".join(c for c in value if unicodedata.category(c) != "Cf").split())


def _specific(value):
    words = {w for w in re.findall(r"[^\W\d_]+", value) if len(w) >= 3 and w not in GENERIC_WORDS}
    cjk = sum("\u3400" <= c <= "\u9fff" for c in value)
    return len(words) >= RULES["specific_min_non_generic_words"] or cjk >= RULES["specific_min_cjk_chars"]


def _long_id(value):
    letters = {c for c in value if c.isalnum()}
    return (len(value) >= RULES["long_id_min_chars"] and len(letters) >= RULES["long_id_min_distinct_alnum"]
            and (any(c.isdigit() for c in value) or any(c in "-_:/" for c in value)))


def _fit_role(value):
    if any(value[k] not in ("D_fit", "fit") for k in ("role", "split") if k in value):
        raise ValueError("literal audit accepts D_fit inputs only")


def _targets(public_tasks, exposed_feedback):
    if isinstance(public_tasks, (str, bytes, Mapping)):
        raise ValueError("public_tasks must be a sequence of public task mappings")
    tasks = list(public_tasks)
    if not tasks or any(not isinstance(t, Mapping) for t in tasks):
        raise ValueError("nonempty public task mappings required")
    # Check all roles before accessing any question or prediction text.
    for task in tasks:
        _fit_role(task)
    questions, ids = [], []
    for task in tasks:
        qid, question = task.get("question_id"), task.get("question")
        if (not isinstance(qid, str) or not qid.strip() or not isinstance(question, str) or not question.strip()
                or len(question) > 16000 or qid in ids):
            raise ValueError("invalid or duplicate public fit question identity")
        ids.append(qid); questions.append(_normalize(question))
    predictions, evidence = [], []
    if exposed_feedback is not None:
        if not isinstance(exposed_feedback, Mapping) or exposed_feedback.get("role") not in ("D_fit", "fit"):
            raise ValueError("exposed feedback must explicitly identify D_fit")
        _fit_role(exposed_feedback)
        cases = exposed_feedback.get("cases", [])
        if not isinstance(cases, list) or any(not isinstance(c, Mapping) for c in cases):
            raise ValueError("invalid exposed fit cases")
        for case in cases:
            _fit_role(case)
            if case.get("question_id") not in ids:
                raise ValueError("exposed feedback case is outside the fit question set")
        def append_text(destination, value):
            if value is None:
                return
            if not isinstance(value, str) or len(value) > 16000:
                raise ValueError("invalid exposed fit text")
            if value.strip():
                destination.append(_normalize(value))
        for case in cases:
            append_text(predictions, case.get("prediction"))
            witnesses = case.get("evidence_witnesses", [])
            if not isinstance(witnesses, list):
                raise ValueError("invalid exposed witness list")
            for witness in witnesses:
                if not isinstance(witness, Mapping):
                    raise ValueError("invalid exposed witness")
                _fit_role(witness)
                append_text(evidence, witness.get("quote_excerpt"))
            for key in ("execution_flow", "parent_execution_flow"):
                flow = case.get(key)
                if flow is None:
                    continue
                if not isinstance(flow, Mapping):
                    raise ValueError("invalid exposed execution flow")
                _fit_role(flow)
                final = flow.get("final", {})
                if not isinstance(final, Mapping):
                    raise ValueError("invalid exposed final excerpt")
                append_text(predictions, final.get("answer_excerpt"))
                reads = flow.get("reads", [])
                if not isinstance(reads, list):
                    raise ValueError("invalid exposed read excerpts")
                for read in reads:
                    if not isinstance(read, Mapping) or not isinstance(read.get("quotes", []), list):
                        raise ValueError("invalid exposed read quotes")
                    for quote in read.get("quotes", []):
                        if not isinstance(quote, Mapping):
                            raise ValueError("invalid exposed quote")
                        append_text(evidence, quote.get("quote_excerpt"))
    return questions, {_normalize(q) for q in ids}, sorted(set(predictions)), sorted(set(evidence))


def _static_string(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _static_string(node.left), _static_string(node.right)
        return left + right if left is not None and right is not None else None
    if isinstance(node, ast.FormattedValue) and node.conversion in (-1, ord("s")) and node.format_spec is None:
        return _static_string(node.value)
    if isinstance(node, ast.JoinedStr):
        parts = [_static_string(x) for x in node.values]
        return "".join(parts) if all(x is not None for x in parts) else None
    if (isinstance(node, ast.Call) and not node.keywords and len(node.args) == 1
            and isinstance(node.func, ast.Attribute) and node.func.attr == "join"
            and isinstance(node.args[0], (ast.List, ast.Tuple))):
        separator = _static_string(node.func.value)
        parts = [_static_string(x) for x in node.args[0].elts]
        if separator is not None and all(x is not None for x in parts):
            return separator.join(parts)
    return None


def _inventory(source):
    tree = ast.parse(source)
    scoped, id_arguments = [], set()
    def scan(node, scope=()):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope += (node.name,)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(arg.arg == "id" for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs):
                    id_arguments.add(scope)
        scoped.append((node, scope))
        for child in ast.iter_child_nodes(node):
            scan(child, scope)
    scan(tree)

    def id_reference(node, scope):
        for item in ast.walk(node):
            if isinstance(item, ast.Name) and (item.id in ID_FIELDS or item.id == "id" and scope in id_arguments):
                return True
            if isinstance(item, ast.Attribute) and (item.attr in ID_FIELDS or item.attr == "id"
                    and isinstance(item.value, ast.Name) and item.value.id in INPUT_NAMES):
                return True
            if isinstance(item, ast.Subscript):
                key, obj = _static_string(item.slice), item.value
            elif (isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
                  and item.func.attr == "get" and item.args):
                key, obj = _static_string(item.args[0]), item.func.value
            else:
                continue
            if key in ID_FIELDS or key == "id" and isinstance(obj, ast.Name) and obj.id in INPUT_NAMES:
                return True
        return False

    tables = defaultdict(set)
    for node, scope in scoped:
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and id_reference(node.slice, scope):
            tables[scope].add(node.value.id)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name) and node.args and id_reference(node.args[0], scope)):
            tables[scope].add(node.func.value.id)
    found = []
    def emit(node, text, route, linked):
        normalized = _normalize(text)
        if normalized:
            found.append({"text": normalized, "line": getattr(node, "lineno", 1),
                          "column": getattr(node, "col_offset", 0), "route": route, "linked": linked})
    def walk(node, scope=(), route=False, linked=False):
        static = _static_string(node)
        if static is not None:
            emit(node, static, route, linked); return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope += (node.name,)
        if isinstance(node, ast.JoinedStr):
            fragments, first = [], node
            for value in node.values:
                fragment = _static_string(value)
                if fragment is not None:
                    if not fragments: first = value
                    fragments.append(fragment)
                else:
                    if fragments: emit(first, "".join(fragments), route, linked)
                    fragments = []; walk(value, scope)
            if fragments: emit(first, "".join(fragments), route, linked)
            return
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            linked = any(id_reference(x, scope) for x in operands)
            for value in operands: walk(value, scope, True, linked)
            return
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if key is not None: walk(key, scope, True, linked)
                walk(value, scope)
            return
        if isinstance(node, ast.Assign):
            bound = any(isinstance(t, ast.Name) and t.id in tables[scope] for t in node.targets)
            for target in node.targets: walk(target, scope)
            walk(node.value, scope, False, bound); return
        if isinstance(node, ast.Subscript):
            bound = isinstance(node.value, ast.Name) and node.value.id in tables[scope]
            walk(node.value, scope, False, id_reference(node.slice, scope))
            walk(node.slice, scope, True, linked or bound or id_reference(node.value, scope)); return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" and node.args:
            bound = isinstance(node.func.value, ast.Name) and node.func.value.id in tables[scope]
            walk(node.func.value, scope, False, id_reference(node.args[0], scope))
            walk(node.args[0], scope, True, linked or bound)
            for value in node.args[1:]: walk(value, scope)
            for value in node.keywords: walk(value, scope)
            return
        if isinstance(node, ast.Match):
            walk(node.subject, scope)
            for case in node.cases:
                walk(case.pattern, scope, True, id_reference(node.subject, scope))
                if case.guard: walk(case.guard, scope)
                for value in case.body: walk(value, scope)
            return
        for value in ast.iter_child_nodes(node):
            walk(value, scope, route, linked)
    walk(tree)
    return found


def _fragment_index(texts, width):
    result = set()
    for text in texts:
        for start in range(max(0, len(text) - width + 1)):
            piece = text[start:start + width]
            if _specific(piece):
                result.add(piece)
    return result


def audit_fit_literals(parent_files, child_files, public_tasks, *, exposed_feedback=None):
    """Return pass/reject evidence for explicit new literal matches, without text disclosure."""
    names = {"rag.py", "rag_core.py"}
    if (not isinstance(parent_files, Mapping) or not isinstance(child_files, Mapping)
            or set(parent_files) != names or set(child_files) != names
            or any(not isinstance(s, str) or len(s.encode("utf-8")) > 220000
                   for files in (parent_files, child_files) for s in files.values())):
        raise ValueError("literal audit requires complete bounded candidate source pairs")
    questions, ids, predictions, evidence = _targets(public_tasks, exposed_feedback)
    full_questions = sorted({q for q in questions if sum(c.isalnum() for c in q) >= RULES["full_question_min_alnum"]})
    indexes = [("question_fragment", RULES["question_fragment_chars"],
                _fragment_index(questions, RULES["question_fragment_chars"])),
               ("exposed_prediction", RULES["prediction_fragment_chars"],
                _fragment_index(predictions, RULES["prediction_fragment_chars"])),
               ("exposed_evidence", RULES["evidence_fragment_chars"],
                _fragment_index(evidence, RULES["evidence_fragment_chars"]))]
    findings, child_count, exempted = [], 0, 0
    for file in sorted(names):
        try:
            before, after = _inventory(parent_files[file]), _inventory(child_files[file])
        except (SyntaxError, RecursionError, ValueError) as error:
            raise ValueError("unable to parse candidate source for literal audit: " + file) from error
        child_count += len(after)
        signature = lambda item: (item["text"], item["route"], item["linked"])
        untouched = defaultdict(list)
        for index, item in enumerate(before): untouched[signature(item)].append(index)
        remaining = set(range(len(before))); changed = []
        for item in after:
            same = untouched[signature(item)]
            if same:
                remaining.discard(same.pop())
            else:
                changed.append(item)
        old = [before[index] for index in sorted(remaining)]
        prior_capacity, consumed = {}, Counter()
        for item in changed:
            value = item["text"]
            matches = [("full_question", q) for q in full_questions if q in value]
            if item["route"] and value in ids and value not in SCHEMA_WORDS and (item["linked"] or _long_id(value)):
                matches.append(("question_id_route", value))
            for kind, width, index in indexes:
                if kind == "question_fragment" and any(k == "full_question" for k, _ in matches):
                    continue
                for start in range(max(0, len(value) - width + 1)):
                    piece = value[start:start + width]
                    if piece in index:
                        matches.append((kind, piece)); break
            for kind, matched in matches:
                fingerprint = _hash(kind + "\0" + matched)
                if fingerprint not in prior_capacity:
                    prior_capacity[fingerprint] = sum(matched in previous["text"] and
                        (kind != "question_id_route" or previous["route"] and
                         (previous["linked"] or _long_id(previous["text"]))) for previous in old)
                if consumed[fingerprint] < prior_capacity[fingerprint]:
                    consumed[fingerprint] += 1; exempted += 1; continue
                findings.append({"file": file, "line": item["line"], "column": item["column"],
                                 "kind": kind, "fingerprint": fingerprint, "matched_chars": len(matched)})
    receipt = {"schema": SCHEMA, "rule_version": RULE_VERSION, "status": "reject" if findings else "pass",
        "findings": findings, "rules": dict(RULES),
        "parent_source_sha256": {f: _hash(parent_files[f]) for f in sorted(names)},
        "child_source_sha256": {f: _hash(child_files[f]) for f in sorted(names)},
        "scope": {"role": "D_fit", "public_question_count": len(questions),
                  "exposed_prediction_count": len(predictions), "exposed_evidence_count": len(evidence),
                  "references_read": False},
        "coverage": {"child_string_expressions": child_count, "parent_matches_exempted": exempted,
                     "new_matches": len(findings)},
        "limitations": ["Conservative static literal check; pass is not proof of absence of hardcoding.",
            "No execution: computed/encoded strings, alias data flow and semantic hints can evade detection.",
            "Short/generic feedback and unlinked short IDs are intentionally outside hard-reject rules.",
            "Named short-ID lookup inference is limited to one lexical scope; existing fit literals are grandfathered within each file.",
            "Only public fit questions and whitelisted already-exposed prediction/quote excerpts are inspected; no private references or held-out panels."]}
    receipt["receipt_sha256"] = _hash(json.dumps(receipt, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    return receipt
