"""Reference-free reconstruction of a completed paired smoke's root trajectories.

Git objects are read only as bytes. No historical or candidate code is imported,
no reference file is opened, and no model/transport is created. Internal hashes
establish local provenance consistency, not independent external authentication.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from .archive import ProgramArchive, Limits, _safe
from .budget import Ledger, digest
from .prepare_answer_probe import PACKET_SCHEMA, _RequestShape, _snapshot_file, _unchanged
from .v3.datasets import filter_documents, validate_task_collection
from .v3.evolution import EvolutionRunner
from .v3.execution import CELL_SCHEMA, HostBroker, Measurement, root_files, validate_answer_origin
from .v3.infrastructure import LocalCorpus, proposal_model_prompts
from .v3.rag import RagEngine
from .v3.reader_replay import SCHEMA, ReplayRouter, validate_case
from .v3.request_recovery import check_request_accounting, check_request_recovery

SELECTION_RULE = "all_frozen_D_fit_tasks_in_file_order__shared_root_block_0__repeat_0__without_scores"
_PROJECT = Path(__file__).resolve().parents[1]


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _git(*args):
    # Arguments are separate argv elements; no shell, imports or archive execution.
    return subprocess.check_output(["git", "-C", str(_PROJECT), *args], stderr=subprocess.PIPE)


def _historical_source(plan, revision):
    _require(type(revision) is str and re.fullmatch(r"[0-9a-f]{40}", revision),
             "source_revision must be a full lowercase local Git commit")
    _require(_git("rev-parse", "--verify", revision + "^{commit}").decode().strip() == revision,
             "source_revision does not identify the exact local commit")
    template = plan["search_template"]
    names = _git("ls-tree", "-r", "--name-only", revision, "--", "code_rsi/v3").decode().splitlines()
    expected = {n.removeprefix("code_rsi/") for n in names
                if n.startswith("code_rsi/v3/") and n.count("/") == 2 and n.endswith(".py")}
    expected.update(("archive.py", "budget.py", "sandbox.py", "sdk.py", "candidate_runner.py", "linux_launcher.sh"))
    original = template["runtime_source_hashes"]
    normalized = {k.replace("\\", "/"): v for k, v in original.items()}
    _require(len(normalized) == len(original) and set(normalized) == expected,
             "historical runtime inventory differs from its Git snapshot")
    files = {}
    for name in sorted(expected):
        raw = _git("show", revision + ":code_rsi/" + name)
        # Match the old runtime's read_text universal-newline source digest.
        text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        _require(digest(text) == normalized[name], "historical runtime source hash differs: " + name)
        files[name] = {"git_blob_raw_sha256": _sha(raw), "source_text_digest": digest(text)}
    entries = {}
    for name, wanted in (("paired_evolution.py", plan["entry_sha256"]),
                         ("live_evolution.py", template["entry_sha256"])):
        raw = _git("show", revision + ":code_rsi/" + name)
        # Git may normalize a Windows checkout's CRLF bytes to LF. Admit only
        # this explicit reversible checkout representation, never arbitrary edits.
        choices = {"git_blob_bytes": raw}
        if b"\r" not in raw:
            choices["git_lf_to_windows_crlf"] = raw.replace(b"\n", b"\r\n")
        matched = [label for label, body in choices.items() if _sha(body) == wanted]
        _require(bool(matched), "historical entry byte hash differs: " + name)
        entries[name] = {"git_blob_raw_sha256": _sha(raw), "frozen_entry_sha256": wanted,
                         "checkout_representation": matched[0]}
    return {"source_revision": revision, "runtime_source_hashes": deepcopy(original),
            "git_runtime_files": files, "entry_files": entries,
            "historical_code_executed": False}


def _bound_json(binding, bindings):
    value, actual = _snapshot_file(binding["path"])
    _require(actual == binding, "source file binding differs")
    bindings.append(actual)
    return value


def _read(path, bindings):
    value, binding = _snapshot_file(_safe(path))
    bindings.append(binding)
    return value, binding


def _read_only_runner(directory, manifest):
    # Construct only the state consumed by the existing read-only seal methods;
    # EvolutionRunner.__init__ would initialize/resolve references and outputs.
    runner = object.__new__(EvolutionRunner)
    runner.directory = _safe(directory)
    runner.manifest = runner._frozen_manifest = deepcopy(manifest)
    runner.lifecycle = deepcopy(manifest["lifecycle"])
    runner.shared_root = deepcopy(manifest["shared_root"])
    runner.archive = object.__new__(ProgramArchive)
    runner.archive.root, runner.archive.limits = _safe(directory / "archive"), Limits()
    return runner


def _settled_accounting(out, plan, completion, bindings):
    records = check_request_recovery(out)
    path = _safe(out / "ledger.jsonl")
    raw = path.read_bytes()
    binding = {"path": str(path), "sha256": _sha(raw)}
    bindings.append(binding)
    ledger = object.__new__(Ledger)
    ledger.used, ledger.pending, ledger.events, ledger.stopped = {}, {}, [], False
    seen, settled = set(), set()
    for line in raw.decode("utf-8").splitlines():
        event = json.loads(line)
        kind, rid = event.get("event"), event.get("id")
        _require(type(rid) is str and bool(rid), "invalid source ledger identity")
        if kind == "reserve":
            _require(rid not in seen, "duplicate source ledger reservation")
            seen.add(rid)
            meta = event.get("metadata", {})
            arm = meta.get("paired_condition")
            _require(meta.get("paired_block") == 0 and arm in {"root", "cases", "trace"}
                     and event.get("scopes") == ["run", "block:0", "root:0" if arm == "root" else "arm:0:" + arm]
                     and meta.get("bank", "").startswith("paired/0/" + arm + "/")
                     and not (arm == "root" and meta.get("stage") == "develop"),
                     "source bank and ledger scopes differ")
        elif kind == "settle":
            _require(rid in ledger.pending and rid not in settled and event.get("reservation_exceeded") is False,
                     "unresolved, repeated or overrun source settlement")
            settled.add(rid)
        else:
            raise ValueError("unknown source ledger event")
        amounts = event["amount"] if kind == "reserve" else event["actual"]
        _require(isinstance(amounts, dict) and all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                 for v in amounts.values()) and amounts.get("calls") == 1,
                 "invalid source accounting amounts")
        Ledger._apply(ledger, event)
    _require(not ledger.pending and not ledger.stopped, "source ledger has unresolved outcome")
    check_request_accounting(records, ledger)
    used = ledger.used.get("run", {})
    _require(used.get("calls", 0) == len(records) and used.get("calls", 0) <= plan["max_calls"]
             and used.get("cny", 0) <= plan["hard_cny"] + 1e-9
             and completion.get("ledger") == {"used": ledger.used, "pending": 0, "alerts": []},
             "completed source accounting differs from its ledger")
    reserves = {e["id"]: e for e in ledger.events if e["event"] == "reserve"}
    for key in records:
        _, bound = _read(out / "requests" / (key + ".json"), bindings)
    return records, reserves, binding


def _reconstruct_case(task, config, receipt, identity, out, shape, records, reserves, bindings, source, ordinal):
    validate_answer_origin(receipt)
    _require(receipt.get("execution_ok") is True and receipt.get("answer_origin_valid") is True
             and receipt.get("answer_usable") is True and receipt.get("model_errors") == [],
             "source root contains a technically incomplete cell; no subset is allowed")
    backend = LocalCorpus(filter_documents(task, task["documents"]), scope=task["question_id"],
                          excluded=task["excluded_docids"])
    _require(identity["backend_identity"] == backend.identity, "source local corpus identity differs")
    bank = "paired/0/root/D_fit/shared-root/0/" + task["question_id"] + "/0"
    requests = []

    class CachedModel:
        def complete(self, stage, payload):
            body = shape.request_body(stage, payload)
            key = digest({"body": body, "bank": bank})
            _require(key in records, "missing exact source bank/body request")
            record, bound = _read(out / "requests" / (key + ".json"), bindings)
            _require(record == records[key] and record.get("body") == body and record.get("state") == "settled"
                     and reserves[record["reservation"]]["metadata"].get("stage") == stage,
                     "source request or settlement identity differs")
            choices = record.get("response", {}).get("choices", [])
            _require(len(choices) == 1 and choices[0].get("finish_reason") == "stop", "incomplete source response")
            result = json.loads(choices[0].get("message", {}).get("content", ""))
            _require(isinstance(result, dict), "source model response is not an object")
            requests.append({"event_index": len(broker.events), "stage": stage, "bank": bank,
                             "request_key": key, "body_sha256": digest(body), "response_sha256": digest(result),
                             "reservation": record["reservation"], "file": bound})
            return deepcopy(result)

    broker = HostBroker(task, backend, CachedModel(), **source["limits"])
    events = []
    for index, event in enumerate(receipt["trace"]):
        name, request = event["name"], event["request"]
        _require(name in {"complete", "search", "read", "record_trace"}, "unsupported source event")
        _require(name != "record_trace" or index == len(receipt["trace"]) - 1, "source record_trace is not last")
        _require(name != "complete" or event.get("model_completed") is True, "incomplete source model event")
        response = broker(name, deepcopy(request))
        _require(digest(response) == event["response_hash"], "source response differs from host reconstruction")
        if name != "record_trace":
            events.append({"name": name, "request": deepcopy(request), "response": response,
                           "response_sha256": digest(response)})
    _require(broker.events == receipt["trace"] and broker.reported == receipt["candidate_reported"]
             and broker.reported is not None and broker.model_errors == receipt["model_errors"]
             and broker.counts == receipt["resource_usage"]
             and {"read_presentations": broker.read_presentations, "final_observations": broker.final_observations}
                 == receipt["host_evidence_trace"]
             and broker.answer_origin_receipt(receipt["answer"]) == receipt["host_answer_origin_validation"]
             and broker.citation_receipt(receipt["answer"], receipt["citations"]) == receipt["host_citation_validation"],
             "reconstructed host observations differ from source receipt")
    case = {"schema": SCHEMA, "case_id": f"Q{ordinal:02d}", "task": deepcopy(task), "config": deepcopy(config),
            "target_read": sum(e["name"] == "complete" and e["request"]["stage"] == "read" for e in events),
            "events": events, "engine_sha256": _sha(root_files(config)["rag_core.py"].encode()),
            "original_final_payload_sha256": digest(events[-1]["request"]["payload"]),
            "source_binding": {**deepcopy(source), "source_identity": deepcopy(identity), "model_requests": requests}}
    case = validate_case(case)
    router = ReplayRouter(case)
    reconstructed = RagEngine(router, router, config=config).solve({"question": task["question"]})
    replay = router.assert_complete()
    _require(reconstructed == receipt["candidate_reported"], "maintained root cannot reproduce the complete original result")
    case["source_binding"]["reconstruction"] = {
        "all_requests_identical": True, "replay": replay, "events_sha256": digest(events),
        "original_candidate_reported_sha256": digest(receipt["candidate_reported"]),
        "current_candidate_reported_sha256": digest(reconstructed)}
    return case


def build_paired_root_cases(plan_or_path, *, source_revision):
    """Return the complete root/repeat-0 panel from one settled schema-2 block.

    Only public task files, existing execution records, source archives and request
    receipts are opened. Reference bindings are compared as opaque metadata;
    reference answers are neither loaded nor required to exist for reconstruction.
    """
    bindings = []
    if isinstance(plan_or_path, (str, Path)):
        plan, supplied = _read(plan_or_path, bindings)
    else:
        _require(isinstance(plan_or_path, dict), "source plan must be an object or JSON path")
        plan = deepcopy(plan_or_path)
    _require(plan.get("schema") == "rag-rsi-paired-development-2" and type(plan.get("blocks")) is int
             and plan["blocks"] == 1 and plan.get("purpose") == "paired_development_smoke"
             and plan.get("schedule_policy") == "alternate_block_and_slot_v1", "one completed paired schema-2 block required")
    template = plan["search_template"]
    _require(template.get("schema") == "rag-rsi-live-evolution-5" and template.get("phase_order") == ["search"]
             and set(template.get("panels", {})) == {"D_fit"} and type(template.get("repeats")) is int
             and template["repeats"] == 1 and template.get("corpus") is None and template.get("corpus_ref") is None,
             "source must be search-only D_fit with one repeat and local documents")
    history = _historical_source(plan, source_revision)
    out = _safe(Path(plan["output_dir"]).resolve(strict=True))
    stored, plan_binding = _read(out / "paired_plan.json", bindings)
    _require(stored == plan, "source plan differs from frozen run")
    completion, complete_binding = _read(out / "paired_search_complete.json", bindings)
    count = 2 * template["expansions"]
    _require(completion.get("schema") == plan["schema"] and completion.get("status") == "complete"
             and completion.get("plan_hash") == digest(plan) and completion.get("completed_opportunities") == count
             and completion.get("planned_opportunities") == count and completion.get("prepared_blocks") == [0]
             and completion.get("heldout_roles_used") is False and len(completion.get("terminal_records", [])) == count,
             "paired source is not completely frozen")
    tasks_binding = template["panels"]["D_fit"]["tasks_file"]
    tasks = _bound_json(tasks_binding, bindings)
    validate_task_collection(tasks)
    _require(bool(tasks) and all(t["dataset"] == "musique" and t["corpus_scope"] == "question_local" for t in tasks),
             "source requires the complete public MuSiQue local task panel")
    qids = [t["question_id"] for t in tasks]
    controls = template["controls"]
    _require(controls.get("parent_policy") == "fixed_root" and controls.get("module_policy") == "fixed"
             and controls.get("memory") == "none" and controls.get("feedback") == "cases"
             and controls.get("case_schedule") == [{"question_id": q, "repeat": 0} for q in qids],
             "source control schedule does not cover the full frozen panel")
    records, reserves, ledger_binding = _settled_accounting(out, plan, completion, bindings)
    config = template["root_config"]
    files = root_files(config)
    shape = _RequestShape(template["model"])
    shape.proposal_protocol = template.get("proposal_protocol")
    model_identity = digest({"model": shape.model, "prompts": proposal_model_prompts(shape.proposal_protocol),
        "limits": shape.limits, "max_input_bytes": shape.max_input_bytes, "temperature": 0,
        "thinking": "disabled", "response_format": "json_object", "prices": template["model"]["prices"],
        **({"proposal_protocol": shape.proposal_protocol} if shape.proposal_protocol is not None else {})})
    shared = out / "blocks/0/root_measurements"
    contract, contract_binding = _read(shared / "contract.json", bindings)
    epoch = Measurement(None, shared, None, metric=template["metric"], limits=template["limits"],
                        allow_proxy_metrics=template["allow_proxy_metric"]).epoch
    _require(contract == {"schema": "rag-rsi-shared-root-contract-1", "bank": "shared-root/0", "block_id": "0",
        "root_files_sha256": {k: _sha(v.encode()) for k, v in files.items()}, "panel_hash": digest(tasks),
        "reference_file_sha256": template["panels"]["D_fit"]["references_file"]["sha256"],
        "evaluator_epoch": epoch, "metric": template["metric"], "limits": template["limits"],
        "repeats": 1, "model_identity": model_identity, "runtime_source_hashes": template["runtime_source_hashes"]},
        "source root contract differs from plan or maintained root")
    seal, seal_binding = _read(shared / "shared_root_seal.json", bindings)
    source_arms = {}
    validators = []
    roots = []
    attempts = {}
    for arm in ("cases", "trace"):
        directory = out / "blocks/0" / arm
        manifest, manifest_binding = _read(directory / "manifest.json", bindings)
        expected_controls = {**deepcopy(controls), "feedback": arm}
        _require(manifest.get("paired_plan_hash") == digest(plan) and manifest.get("paired_block") == 0
                 and manifest.get("condition") == arm and manifest.get("controls") == expected_controls
                 and manifest.get("root_config") == config and manifest.get("runtime_source_hashes") == template["runtime_source_hashes"]
                 and manifest.get("public_panel_hashes") == {"D_fit": digest(tasks)}
                 and manifest.get("model_identity") == model_identity and manifest.get("limits") == template["limits"]
                 and manifest.get("repeats") == 1 and manifest.get("metric") == template["metric"]
                 and manifest.get("lifecycle", {}).get("phase_order") == ["search"]
                 and manifest["lifecycle"].get("reference_bindings") == {"D_fit": template["panels"]["D_fit"]["references_file"]},
                 "source arm manifest differs from source plan")
        expected_shared = {"schema": "rag-rsi-shared-root-1", "directory": str(shared), "block_id": "0", "bank": "shared-root/0"}
        _require(manifest.get("shared_root") == expected_shared, "source shared root directory identity differs")
        runner = _read_only_runner(directory, manifest)
        runner._shared_contract = contract
        result = runner._verify_phase("search")
        reference = runner._verify_shared_root()
        validators.append(runner)
        _require(reference is not None, "missing sealed source root")
        root, root_binding = _read(directory / "root.json", bindings)
        program = runner.archive.load_program(root["program_id"])
        _require(program["files"] == files and root["parent_node_id"] is None, "root archive differs from maintained root")
        roots.append(reference["result"])
        attempts[arm] = result["search"]["terminal_attempts"]
        _require(len(attempts[arm]) == template["expansions"], "source arm opportunities incomplete")
        _, phase_binding = _read(directory / "phase_search.json", bindings)
        _, local_binding = _read(directory / "measurements/shared_root.json", bindings)
        archive_bindings = {}
        for name in ("manifest.json", "files/rag.py", "files/rag_core.py"):
            path = runner.archive.root / "programs" / root["program_id"] / name
            bound = {"path": str(path), "sha256": _sha(path.read_bytes())}
            bindings.append(bound); archive_bindings[name] = bound
        source_arms[arm] = {"manifest": manifest_binding, "phase_seal": phase_binding, "root": root_binding,
                            "local_shared_reference": local_binding, "root_archive": archive_bindings}
    _require(roots[0] == roots[1], "two arms do not reference the exact same root measurement")
    scheduled = []
    for slot in range(template["expansions"]):
        for arm in (("cases", "trace") if slot % 2 == 0 else ("trace", "cases")):
            scheduled.append({"schedule_index": len(scheduled), "block": 0, "condition": arm,
                              "slot": slot, "attempt": attempts[arm][slot]})
    _require(completion["terminal_records"] == scheduled, "completion schedule differs from sealed attempts")
    measurement = roots[0]
    _, measurement_binding = _read(shared / measurement["identity_hash"] / "measurement.json", bindings)
    rows = measurement["rows"]
    _require([r["question_id"] for r in rows] == qids and all(r.get("repeat") == 0 for r in rows),
             "source root panel incomplete, duplicated or reordered")
    source = {"kind": "frozen_paired_root_repeat_0", "selection_rule": SELECTION_RULE, "limits": template["limits"],
        "source_plan": plan_binding, "source_plan_hash": digest(plan), "source_completion": complete_binding,
        "source_tasks": deepcopy(tasks_binding), "source_ledger": ledger_binding,
        "shared_root_contract": contract_binding, "shared_root_seal": seal_binding,
        "source_measurement": measurement_binding, "source_arms": source_arms,
        "historical_runtime": history, "reference_files_opened": False,
        "original_root_source_sha256": digest(files), "source_engine_sha256": _sha(files["rag_core.py"].encode()),
        "current_engine_sha256": _sha(files["rag_core.py"].encode())}
    cases = []
    for ordinal, (task, row) in enumerate(zip(tasks, rows), 1):
        cell_path = shared / measurement["identity_hash"] / (digest(task["question_id"])[:16] + "_0") / "measured.json"
        cell, binding = _read(cell_path, bindings)
        ident = cell["identity"]
        _require(cell.get("schema") == CELL_SCHEMA and cell.get("identity_sha256") == digest(ident)
                 and cell.get("payload") == row and cell.get("payload_sha256") == digest(row)
                 and ident.get("measurement_identity_hash") == measurement["identity_hash"]
                 and ident.get("node_id") == measurement["node_id"] and ident.get("program_id") == measurement["program_id"]
                 and ident.get("question_id") == task["question_id"] and ident.get("dataset") == "musique"
                 and ident.get("role") == "D_fit" and ident.get("bank") == "shared-root/0" and ident.get("repeat") == 0
                 and ident.get("task_hash") == digest(task) and ident.get("model_identity") == model_identity
                 and ident.get("evaluator_epoch") == epoch, "source measurement cell identity differs")
        cases.append(_reconstruct_case(task, config, row, ident, out, shape, records, reserves, bindings,
                                      {**source, "source_cell": binding}, ordinal))
    # Recheck complete artifact inventories after reconstruction, including
    # files not read into the case, so concurrent edits cannot slip through.
    for runner in validators:
        runner._verify_phase("search")
        runner._verify_shared_root()
    for binding in bindings:
        _unchanged(binding)
    return {"schema": PACKET_SCHEMA, "cases": cases}
