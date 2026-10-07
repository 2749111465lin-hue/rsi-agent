"""Candidate-facing RPC SDK; no credentials, scorer, or filesystem capabilities."""
import json
import re
import sys

MAX_FRAME = 2 * 1024 * 1024


def read_frame(stream):
    line = stream.readline(MAX_FRAME + 1)
    if not line or len(line.encode("utf-8")) > MAX_FRAME or not line.endswith("\n"):
        raise RuntimeError("missing_or_oversized_rpc_frame")
    obj = json.loads(line)
    if not isinstance(obj, dict):
        raise ValueError("rpc_object_required")
    return obj


def emit(stream, obj):
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if len(line.encode("utf-8")) + 1 > MAX_FRAME:
        raise ValueError("rpc_frame_too_large")
    stream.write(line + "\n")
    stream.flush()


class Services:
    def __init__(self, budget, reader=None, writer=None):
        self.budget = dict(budget)
        self._reader = reader or sys.stdin
        self._writer = writer or sys.stdout
        self._sequence = 0

    def call(self, name, payload):
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name):
            raise ValueError("invalid_service_name")
        if not isinstance(payload, dict):
            raise ValueError("service_payload_must_be_object")
        self._sequence += 1
        if self._sequence > self.budget.get("max_rpc_calls", 64):
            raise RuntimeError("rpc_call_limit")
        emit(self._writer, {"kind": "call", "id": self._sequence, "name": name, "payload": payload})
        result = read_frame(self._reader)
        if result.get("id") != self._sequence:
            raise RuntimeError("rpc_response_id_mismatch")
        if not result.get("ok"):
            raise RuntimeError(str(result.get("error", "service_rejected"))[:2000])
        return result.get("result")
