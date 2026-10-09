"""Materialize literal edits against a frozen parent without executing source.

Source identities use budget.digest on the complete source-text mapping: this is
canonical JSON hashing, not hashing on-disk file bytes. Newlines are preserved.
Exact application is a proposal transport contract, not a safety, AST-change,
module-attribution or semantic-improvement judgment; those remain host gates.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib

from ..budget import digest

SCHEMA = "rag-rsi-proposal-protocol-1"
RECEIPT_SCHEMA = "rag-rsi-materialized-edits-1"
FILES = ("rag.py", "rag_core.py")


def validate_proposal_protocol(value):
    """Return a detached exact protocol, preserving None for legacy manifests."""
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("exact proposal protocol required")
    format_name = value.get("format")
    fields = {"schema", "format"}
    if format_name == "exact_edits":
        fields |= {"max_edits", "max_edit_chars"}
    elif format_name != "whole_files":
        raise ValueError("unknown proposal format")
    if set(value) != fields:
        raise ValueError("unexpected proposal protocol fields")
    if format_name == "exact_edits":
        for field, maximum in (("max_edits", 64), ("max_edit_chars", 200000)):
            if type(value[field]) is not int or not 1 <= value[field] <= maximum:
                raise ValueError("invalid " + field)
    return deepcopy(value)


def _sources(files):
    if (not isinstance(files, Mapping) or set(files) != set(FILES)
            or any(not isinstance(files[name], str) for name in FILES)):
        raise ValueError("exact rag.py and rag_core.py source-text mapping required")
    result = {name: files[name] for name in FILES}
    try:
        for value in result.values():
            value.encode("utf-8")
    except UnicodeError as error:
        raise ValueError("source text must be UTF-8 encodable") from error
    return result


def source_identity(files):
    """Canonical JSON identity of both exact source strings, not file-byte SHA."""
    return digest(_sources(files))


def _text_hash(value):
    try:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
    except UnicodeError as error:
        raise ValueError("edit text must be UTF-8 encodable") from error


def materialize_edits(parent_files, proposal, protocol):
    """Return full candidate files and a deterministic, excerpt-free receipt.

    All anchors locate once in the ORIGINAL parent, counting overlapping string
    occurrences. Adjacent edit ranges are legal; overlapping ranges are not.
    Applying right to left means edit ordering cannot change anchor resolution.
    An explicit no_change response returns unchanged sources and a receipt; the
    host records rejection/attempt accounting without executing that candidate.
    source_changed is exact text inequality, never a behavioral guarantee.
    """
    protocol = validate_proposal_protocol(protocol)
    if protocol is None or protocol["format"] != "exact_edits":
        raise ValueError("materialize_edits requires explicit exact_edits protocol")
    parent = _sources(parent_files)
    parent_hash = source_identity(parent)
    fields = {"parent_source_sha256", "edits", "mechanism", "intended_target_module", "change_status"}
    if not isinstance(proposal, dict) or set(proposal) != fields:
        raise ValueError("exact edit response fields required")
    if not isinstance(proposal["parent_source_sha256"], str) or proposal["parent_source_sha256"] != parent_hash:
        raise ValueError("parent source identity mismatch")
    for field in ("mechanism", "intended_target_module"):
        if not isinstance(proposal[field], str) or not proposal[field].strip():
            raise ValueError("nonempty " + field + " required")
    declared_status = proposal["change_status"]
    if not isinstance(declared_status, str) or declared_status not in ("modified", "no_change"):
        raise ValueError("change_status must be modified or no_change")
    edits = proposal["edits"]
    if not isinstance(edits, list):
        raise ValueError("edits must be a list")
    if declared_status == "no_change":
        if edits:
            raise ValueError("no_change must have an empty edits list")
    elif not 1 <= len(edits) <= protocol["max_edits"]:
        raise ValueError("edit count outside frozen protocol")
    located, total_chars = [], 0
    for index, edit in enumerate(edits):
        if not isinstance(edit, dict) or set(edit) != {"file", "old", "new"}:
            raise ValueError("exact file/old/new edit fields required")
        file, old, new = edit["file"], edit["old"], edit["new"]
        if not isinstance(file, str) or file not in parent:
            raise ValueError("edit file outside candidate sources")
        if not isinstance(old, str) or not old or not isinstance(new, str) or old == new:
            raise ValueError("edit requires a nonempty changed literal anchor")
        total_chars += len(old) + len(new)
        if total_chars > protocol["max_edit_chars"]:
            raise ValueError("edit characters exceed frozen protocol")
        old_hash, new_hash = _text_hash(old), _text_hash(new)
        start = parent[file].find(old)
        if start < 0:
            raise ValueError("literal edit anchor absent from original parent")
        if parent[file].find(old, start + 1) >= 0:
            raise ValueError("literal edit anchor is not unique in original parent")
        located.append({"edit_index": index, "file": file, "start": start,
                        "end": start + len(old), "old_sha256": old_hash,
                        "new_sha256": new_hash, "old_chars": len(old), "new_chars": len(new)})
    by_file = {file: sorted((row for row in located if row["file"] == file),
                           key=lambda row: row["start"]) for file in FILES}
    for rows in by_file.values():
        for left, right in zip(rows, rows[1:]):
            if left["end"] > right["start"]:
                raise ValueError("edit intervals overlap in original parent")
    candidate = dict(parent)
    for file, rows in by_file.items():
        for row in reversed(rows):
            replacement = edits[row["edit_index"]]["new"]
            candidate[file] = candidate[file][:row["start"]] + replacement + candidate[file][row["end"]:]
    receipt = {"schema": RECEIPT_SCHEMA, "protocol": protocol,
               "declared_change_status": declared_status, "source_changed": candidate != parent,
               "parent_source_sha256": parent_hash,
               "child_source_sha256": source_identity(candidate),
               "proposal_sha256": digest(proposal), "edits": located,
               "source_hash_kind": "budget.digest_complete_source_text_mapping",
               "edit_hash_kind": "sha256_exact_utf8_text",
               "offset_kind": "unicode_character_offsets_in_original_parent"}
    receipt["receipt_sha256"] = digest(receipt)
    return candidate, receipt
