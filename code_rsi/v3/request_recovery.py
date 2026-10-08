"""Run-wide request recovery gates shared by live evolution and calibration.

Call recovery before creating the ledger or transport, then verify accounting
before dispatch. A changed body or bank never clears an unresolved old request.
"""
from __future__ import annotations
import json
from pathlib import Path

from ..budget import digest
from .execution import HostError
from .infrastructure import UnknownProviderOutcome


def check_request_recovery(directory, *, status_filename="live_status.json"):
    """Unknown physical outcomes stop the whole run, even if the next key changes."""
    directory = Path(directory)
    requests = directory / "requests"
    caches = list(requests.glob("*.json"))
    records = {}
    for path in caches:
        if path.name == "returned_model.json":
            continue
        try:
            record = json.loads(path.read_bytes())
        except (ValueError, OSError) as exc:
            raise HostError("unreadable request cache; reconcile before resume") from exc
        if not isinstance(record, dict) or record.get("state") not in {"pending", "response_received", "settled"}:
            raise HostError("invalid request cache; reconcile before resume")
        if record["state"] != "settled":
            raise UnknownProviderOutcome("unresolved request anywhere in this run; reconcile before resume")
        records[path.stem] = record
    if caches and not (directory / "ledger.jsonl").is_file():
        raise HostError("request cache has no complete ledger; reconcile before resume")
    status = directory / status_filename
    if status.exists():
        previous = json.loads(status.read_bytes())
        if previous.get("reason_type") == "UnknownProviderOutcome":
            raise UnknownProviderOutcome("previous run stopped with unknown provider outcome; reconcile before resume")
    return records


def check_request_accounting(records, ledger):
    reservations = {e["id"]: e for e in ledger.events if e["event"] == "reserve"}
    settled = {e["id"] for e in ledger.events if e["event"] == "settle"}
    if len(records) != len(reservations) or set(reservations) != settled:
        raise HostError("request cache and complete ledger differ; reconcile before resume")
    for key, record in records.items():
        reserve = reservations.get(record.get("reservation"))
        metadata = reserve.get("metadata", {}) if reserve else {}
        if (record.get("key") != key or metadata.get("request_key") != key
                or "body" not in record or "response" not in record
                or digest({"body": record["body"], "bank": metadata.get("bank")}) != key):
            raise HostError("request identity differs from its ledger reservation; reconcile before resume")

