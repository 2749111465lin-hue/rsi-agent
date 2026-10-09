"""Static prompt-edit boundary; candidate source is parsed, never executed.

This is an edit constraint, not a replacement for the execution sandbox. Receipt
hashes bind local inputs and checks; they are not external authentication.
"""
from __future__ import annotations

import ast
import codecs
from io import BytesIO
from copy import deepcopy
import hashlib
import json
import math
import tokenize
from collections.abc import Mapping

from ..budget import digest

SCHEMA = "rag-rsi-edit-policy-1"
RECEIPT_SCHEMA = "rag-rsi-edit-policy-receipt-1"
STAGES = ("plan", "read", "answer")
FILES = {"rag.py", "rag_core.py"}


def validate_edit_policy(value):
    """Preserve None for legacy snapshots; explicit policies have exact fields."""
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("exact edit policy required")
    mode = value.get("mode")
    fields = {"schema", "mode"}
    if mode == "prompt_only":
        fields.add("allowed_prompt_stages")
    elif mode != "program":
        raise ValueError("unknown edit policy mode")
    if set(value) != fields:
        raise ValueError("unexpected edit policy fields")
    if mode == "program":
        return deepcopy(value)
    stages = value["allowed_prompt_stages"]
    if (not isinstance(stages, list) or not stages
            or any(not isinstance(s, str) or s not in STAGES for s in stages)
            or len(stages) != len(set(stages))):
        raise ValueError("distinct known allowed prompt stages required")
    return {"schema": SCHEMA, "mode": mode,
            "allowed_prompt_stages": [s for s in STAGES if s in stages]}


class _Invalid(ValueError):
    pass


def _pairs(items):
    result = {}
    for key, value in items:
        if not isinstance(key, str) or key in result:
            raise _Invalid("duplicate_or_nonstring_config_key")
        result[key] = value
    return result


def _json_constant(value):
    raise _Invalid("nonfinite_config_value")


def _finite(value):
    if type(value) is float and not math.isfinite(value):
        raise _Invalid("nonfinite_config_value")
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)
    return value


def _literal(node):
    # A small literal grammar avoids evaluating calls, comprehensions, names,
    # unpacking, or overloaded operators. Dict keys are checked before collapse.
    if isinstance(node, ast.Constant) and type(node.value) in (str, int, float, bool, type(None)):
        return _finite(node.value)
    if isinstance(node, ast.Dict):
        items = []
        for key, value in zip(node.keys, node.values):
            if not isinstance(key, ast.Constant) or type(key.value) is not str:
                raise _Invalid("duplicate_or_nonstring_config_key")
            items.append((key.value, _literal(value)))
        return _pairs(items)
    if isinstance(node, ast.List):
        return [_literal(item) for item in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        if not isinstance(node.operand, ast.Constant) or type(node.operand.value) not in (int, float):
            raise _Invalid("dynamic_config")
        return _finite(-node.operand.value if isinstance(node.op, ast.USub) else node.operand.value)
    raise _Invalid("dynamic_config")


def _root_name(node):
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _bindings(tree, assignment, json_call):
    target = assignment.targets[0]
    imports = []
    for node in ast.walk(tree):
        dynamic = {"eval", "exec", "compile", "globals", "locals", "vars", "setattr", "delattr", "getattr", "__import__", "__builtins__"}
        if ((isinstance(node, ast.Name) and node.id in dynamic)
                or (isinstance(node, ast.Attribute) and node.attr in dynamic)):
            raise _Invalid("dynamic_wrapper_binding")
        if isinstance(node, (ast.MatchAs, ast.MatchStar, ast.MatchMapping)):
            names = [getattr(node, "name", None), getattr(node, "rest", None)]
            if any(name in {"CONFIG", "json"} for name in names):
                raise _Invalid("ambiguous_pattern_binding")
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                if name == "CONFIG":
                    raise _Invalid("ambiguous_config_binding")
                if name == "json":
                    if alias.name != "json" or alias.asname not in (None, "json") or node not in tree.body:
                        raise _Invalid("shadowed_json_binding")
                    imports.append(node)
        if isinstance(node, ast.ImportFrom):
            if any(a.name == "*" or (a.asname or a.name) in {"json", "CONFIG"} for a in node.names):
                raise _Invalid("ambiguous_import_binding")
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            if node.id == "CONFIG" and node is not target:
                raise _Invalid("ambiguous_config_binding")
            if node.id == "json":
                raise _Invalid("shadowed_json_binding")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.arg, ast.ExceptHandler)):
            name = node.arg if isinstance(node, ast.arg) else node.name
            if name in {"CONFIG", "json"}:
                raise _Invalid("ambiguous_config_binding" if name == "CONFIG" else "shadowed_json_binding")
        if isinstance(node, (ast.Global, ast.Nonlocal)) and set(node.names) & {"CONFIG", "json"}:
            raise _Invalid("ambiguous_config_binding")
        if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
            if _root_name(node) in {"CONFIG", "json"}:
                raise _Invalid("config_or_json_mutation")
        if isinstance(node, ast.Call) and _root_name(node.func) in {"CONFIG", "json"} and node is not json_call:
            raise _Invalid("config_or_json_mutation")
    if len(imports) > 1 or (json_call is not None and len(imports) != 1):
        raise _Invalid("shadowed_json_binding")
    if json_call is not None and tree.body.index(imports[0]) >= tree.body.index(assignment):
        raise _Invalid("shadowed_json_binding")


def _view(source):
    # ast.parse(str) ignores cookies; Python loads candidate files from bytes.
    encoding, _ = tokenize.detect_encoding(BytesIO(source.encode("utf-8")).readline)
    if codecs.lookup(encoding).name not in {"utf-8", "utf-8-sig"}:
        raise _Invalid("non_utf8_source_encoding")
    tree = ast.parse(source)
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in node.targets)]
    if (len(assignments) != 1 or len(assignments[0].targets) != 1
            or not isinstance(assignments[0].targets[0], ast.Name)):
        raise _Invalid("ambiguous_config_assignment")
    assignment = assignments[0]
    value, json_call = assignment.value, None
    if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name) and value.func.value.id == "json"
            and value.func.attr == "loads" and len(value.args) == 1 and not value.keywords
            and isinstance(value.args[0], ast.Constant) and type(value.args[0].value) is str):
        json_call = value
        config = _finite(json.loads(value.args[0].value, object_pairs_hook=_pairs,
                                   parse_constant=_json_constant))
    else:
        config = _literal(value)
    _bindings(tree, assignment, json_call)
    if not isinstance(config, dict):
        raise _Invalid("config_not_dictionary")
    prompts = config.get("prompts", {})
    if (not isinstance(prompts, dict) or set(prompts) - set(STAGES)
            or any(type(v) is not str or len(v) > 8000 for v in prompts.values())):
        raise _Invalid("invalid_prompt_dictionary")
    nonprompt = {key: val for key, val in config.items() if key != "prompts"}
    # JSON spelling distinguishes bool/int/float and signed zero. Preserve map
    # order as well: changing unrelated config structure is not a prompt edit.
    nonprompt_identity = json.dumps(nonprompt, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    assignment.value = ast.Constant(value="<HOST_CONFIG_PLACEHOLDER>")
    return prompts, nonprompt_identity, ast.dump(tree, include_attributes=False)


def _hashes(files):
    if (not isinstance(files, Mapping) or set(files) != FILES
            or any(not isinstance(value, str) for value in files.values())):
        return None
    try:
        return {name: hashlib.sha256(files[name].encode("utf-8")).hexdigest() for name in sorted(FILES)}
    except UnicodeError:
        return None


def check_edit_policy(parent_files, child_files, policy):
    """Return a deterministic frozen receipt; only malformed policy raises.

    Unrestricted mode does not claim to discover prompt differences. Source
    syntax, candidate safety, literal leakage and quality remain separate gates.
    """
    policy = validate_edit_policy(policy)
    parent_hash, child_hash = _hashes(parent_files), _hashes(child_files)
    reasons, changed = [], None
    if parent_hash is None:
        reasons.append("invalid_parent_sources")
    if child_hash is None:
        reasons.append("invalid_child_sources")
    if policy is not None and policy["mode"] == "prompt_only" and not reasons:
        changed = []
        if parent_files["rag_core.py"] != child_files["rag_core.py"]:
            reasons.append("core_source_changed")
        views = []
        for label, files in (("parent", parent_files), ("child", child_files)):
            try:
                views.append(_view(files["rag.py"]))
            except _Invalid as error:
                reasons.append(label + "_" + str(error))
                views.append(None)
            except (SyntaxError, ValueError, TypeError, RecursionError, OverflowError, LookupError):
                reasons.append(label + "_invalid_static_config_or_source")
                views.append(None)
        before, after = views
        if before is not None and after is not None:
            changed = [stage for stage in STAGES if (stage in before[0]) != (stage in after[0])
                       or before[0].get(stage) != after[0].get(stage)]
            if before[1] != after[1]:
                reasons.append("nonprompt_config_changed")
            if before[2] != after[2]:
                reasons.append("wrapper_ast_changed")
            if set(changed) - set(policy["allowed_prompt_stages"]):
                reasons.append("prompt_stage_not_allowed")
            if not any(before[0].get(stage, "") != after[0].get(stage, "") for stage in STAGES):
                reasons.append("no_prompt_change")
    receipt = {"schema": RECEIPT_SCHEMA, "source": "host_static_edit_policy",
               "parent_source_sha256": parent_hash, "child_source_sha256": child_hash,
               "policy": policy, "changed_prompt_stages": changed,
               "allowed": not reasons, "reason_codes": reasons}
    receipt["receipt_sha256"] = digest(receipt)
    return receipt
