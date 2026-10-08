"""Host-observed AST edit locations, never semantic or causal attribution.

The fixed map names scopes in the maintained RAG implementation, not arbitrary
candidate function names. New helpers, shared logic and orchestration remain
unknown. A receipt binds both full sources; the runner recomputes it on replay.
Integrity validation is not authentication of an externally supplied card.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from typing import Mapping

from ..budget import digest

SCHEMA = "rag-rsi-v3-edit-scope-1"
MODULES = ("query_rewrite", "retrieval", "evidence_selection", "answer_generation")
FUNCTION_MODULES = {
    ("rag_core.py", "_query_key"): ("query_rewrite", "answer_generation"),
    ("rag_core.py", "ground_quote"): ("evidence_selection",),
    ("rag_core.py", "RagEngine.solve.search"): ("retrieval", "evidence_selection"),
    ("rag_core.py", "RagEngine.solve.consume_read"): ("evidence_selection",),
    ("rag_core.py", "RagEngine.solve.final_size"): ("answer_generation",),
    ("rag.py", "Backend.search"): ("retrieval",),
    ("rag.py", "Backend.read"): ("retrieval",),
}
CONFIG_MODULES = {
    "search_limit": ("retrieval",),
    "max_queries_per_round": ("retrieval",),
    "max_source_chars": ("evidence_selection",),
    "max_context_chars": ("evidence_selection",),
    "max_evidence_items": ("evidence_selection",),
    "max_answer_chars": ("answer_generation",),
}
STAGE_MODULES = {
    "plan": ("query_rewrite",),
    "read": ("query_rewrite", "evidence_selection"),
    "answer": ("answer_generation",),
}


def _dump(node):
    return ast.dump(node, include_attributes=False)


def _json_object(value):
    # JSON-loaded CONFIG is the exact wrapper representation used by root_files.
    if (isinstance(value, ast.Call) and not value.keywords and len(value.args) == 1
            and isinstance(value.func, ast.Attribute) and value.func.attr == "loads"
            and isinstance(value.func.value, ast.Name) and value.func.value.id == "json"
            and isinstance(value.args[0], ast.Constant) and isinstance(value.args[0].value, str)):
        def pairs(items):
            result = {}
            for key, val in items:
                if key in result:
                    raise ValueError("duplicate config key")
                result[key] = val
            return result
        try:
            parsed = json.loads(value.args[0].value, object_pairs_hook=pairs)
            if isinstance(parsed, dict):
                return ast.parse(repr(parsed), mode="eval").body
        except (ValueError, SyntaxError):
            pass
    return value


def _inventory(source):
    records = {}
    def put(kind, path, value):
        key = (kind, path)
        # Duplicate bindings must never collapse to one apparently safe edit.
        if key in records:
            records[key] += "\nDUPLICATE_BINDING\n" + value
        else:
            records[key] = value

    def constants(path, node):
        if isinstance(node, ast.Dict) and all(isinstance(k, ast.Constant) and isinstance(k.value, str)
                                              for k in node.keys):
            keys = [k.value for k in node.keys]
            if len(keys) != len(set(keys)):
                put("constant", path, _dump(node)); return
            put("constant_order", path, json.dumps(keys))
            if not keys and path not in {"CONFIG", "CONFIG.prompts", "DEFAULTS", "DEFAULTS.prompts", "INSTRUCTIONS", "SCHEMAS"}:
                put("constant", path, _dump(node))
            # Empty prompt dictionaries have no changed value when a known stage
            # is added; their container is represented by constant_order.
            for key, value in zip(keys, node.values):
                suffix = "." + key if key.isidentifier() else "[" + repr(key) + "]"
                constants(path + suffix, value)
        else:
            try:
                ast.literal_eval(node)
                literal = True
            except (ValueError, TypeError):
                literal = False
            put("constant", path, ("" if literal else "DYNAMIC_CONSTANT\n") + _dump(node))

    class Extract(ast.NodeTransformer):
        def __init__(self, path=""):
            self.path = path

        def visit_FunctionDef(self, node):
            path = self.path + "." + node.name if self.path else node.name
            header = deepcopy(node); header.body = []
            put("function_signature", path, _dump(header))
            body = ast.Module(body=node.body, type_ignores=[])
            put("function", path, _dump(Extract(path).visit(body)))
            return ast.Expr(value=ast.Constant(value="function:" + node.name))

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node):
            path = self.path + "." + node.name if self.path else node.name
            header = deepcopy(node); header.body = []
            put("class_signature", path, _dump(header))
            body = ast.Module(body=node.body, type_ignores=[])
            put("class", path, _dump(Extract(path).visit(body)))
            return ast.Expr(value=ast.Constant(value="class:" + node.name))

        def visit_Assign(self, node):
            if not self.path and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                constants(name, _json_object(node.value) if name == "CONFIG" else node.value)
                return ast.Expr(value=ast.Constant(value="constant:" + name))
            return self.generic_visit(node)

    put("module", "<module>", _dump(Extract().visit(ast.parse(source))))
    return records


def _scope_modules(file, kind, path, operation):
    if kind == "function" and operation == "modified":
        return sorted(FUNCTION_MODULES.get((file, path), ()))
    if kind != "constant":
        return []
    parts = path.split(".")
    if ((file == "rag.py" and parts[0] == "CONFIG") or
            (file == "rag_core.py" and parts[0] == "DEFAULTS")):
        if len(parts) == 2:
            return sorted(CONFIG_MODULES.get(parts[1], ()))
        if len(parts) == 3 and parts[1] == "prompts":
            return sorted(STAGE_MODULES.get(parts[2], ()))
    if file == "rag_core.py" and parts[0] in {"INSTRUCTIONS", "SCHEMAS"} and len(parts) >= 2:
        return sorted(STAGE_MODULES.get(parts[1], ()))
    return []


def _classification(changes, intended):
    modules = sorted({m for row in changes for m in row["modules"]})
    unknown = any(not row["modules"] for row in changes)
    status = ("unknown" if unknown else "mixed" if len(modules) > 1 else
              "single_module" if len(modules) == 1 else "none")
    associated = modules[0] if status == "single_module" else None
    return {"attribution": status, "associated_module": associated,
            "affected_modules": modules, "has_unknown_scope": unknown,
            "intent_mismatch": None if status in {"unknown", "none"} else associated != intended,
            "association_is_causal": False}


def observe_edit_scope(parent_files, child_files, intended_target_module):
    """Compare complete source pairs without executing candidate code."""
    if intended_target_module not in MODULES:
        raise ValueError("unknown intended target module")
    if set(parent_files) != {"rag.py", "rag_core.py"} or set(child_files) != set(parent_files):
        raise ValueError("edit scope requires both complete source pairs")
    changes = []
    for file in sorted(parent_files):
        before, after = _inventory(parent_files[file]), _inventory(child_files[file])
        for kind, path in sorted(set(before) | set(after)):
            key = (kind, path)
            if before.get(key) == after.get(key):
                continue
            if kind == "constant_order":
                # Membership changes are represented by the individual leaves.
                # Preserve an unknown receipt for order changes among shared keys.
                try:
                    old = json.loads(before.get(key, "[]")); new = json.loads(after.get(key, "[]"))
                except ValueError:  # duplicate declarations remain unknown
                    old = new = None
                if old is not None and [x for x in old if x in new] == [x for x in new if x in old]:
                    continue
            operation = "added" if key not in before else "removed" if key not in after else "modified"
            mapped = _scope_modules(file, kind, path, operation)
            if any(marker in before.get(key, "") + after.get(key, "")
                   for marker in ("DUPLICATE_BINDING", "DYNAMIC_CONSTANT")):
                mapped = []
            changes.append({"file": file, "kind": kind, "path": path,
                            "operation": operation, "modules": mapped})
        if (_dump(ast.parse(parent_files[file])) != _dump(ast.parse(child_files[file]))
                and not any(row["file"] == file for row in changes)):
            changes.append({"file": file, "kind": "module", "path": "<unresolved_ast_change>",
                            "operation": "modified", "modules": []})
    hashes = lambda files: {name: hashlib.sha256(text.encode("utf-8")).hexdigest()
                            for name, text in sorted(files.items())}
    receipt = {"schema": SCHEMA, "source": "host_ast_diff",
               "intended_target_module": intended_target_module,
               "parent_source_sha256": hashes(parent_files), "child_source_sha256": hashes(child_files),
               "changes": changes, **_classification(changes, intended_target_module)}
    receipt["receipt_sha256"] = digest(receipt)
    return receipt


def validated_scope(value):
    """Reject missing/legacy/malformed receipts; this is integrity, not origin authentication."""
    if not isinstance(value, Mapping):
        return None
    scope = deepcopy(dict(value))
    checksum = scope.pop("receipt_sha256", None)
    try:
        correct_digest = digest(scope)
    except (TypeError, ValueError):
        return None
    if (scope.get("schema") != SCHEMA or scope.get("source") != "host_ast_diff"
            or scope.get("intended_target_module") not in MODULES or checksum != correct_digest):
        return None
    for field in ("parent_source_sha256", "child_source_sha256"):
        hashes = scope.get(field)
        if (not isinstance(hashes, dict) or set(hashes) != {"rag.py", "rag_core.py"}
                or any(not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
                       for v in hashes.values())):
            return None
    changes = scope.get("changes")
    if not isinstance(changes, list):
        return None
    for row in changes:
        if (not isinstance(row, dict) or set(row) != {"file", "kind", "path", "operation", "modules"}
                or not isinstance(row["file"], str) or row["file"] not in {"rag.py", "rag_core.py"}
                or not isinstance(row["path"], str) or not isinstance(row["kind"], str)
                or not isinstance(row["operation"], str) or row["operation"] not in {"added", "removed", "modified"}
                or not isinstance(row["modules"], list)):
            return None
        # Host may demote a duplicate binding to unknown, never promote an unknown scope.
        if row["modules"] and row["modules"] != _scope_modules(row["file"], row["kind"], row["path"], row["operation"]):
            return None
    if any(scope.get(k) != v for k, v in _classification(changes, scope["intended_target_module"]).items()):
        return None
    return {**scope, "receipt_sha256": checksum}
