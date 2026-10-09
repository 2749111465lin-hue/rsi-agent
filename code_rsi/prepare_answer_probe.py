"""Reconstruct fixed MuSiQue answer states from a frozen, host-validated prior run.

Only host-owned, frozen D_fit material is read. No credentials, references,
provider dispatch, grader output, or arbitrary archived code are used. The
trusted maintained source and current RagEngine are replayed with the exact
original configuration; hashes establish local record consistency, not external
authentication.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .budget import digest
from .v3 import calibration as cal, musique_calibration as musique
from .v3.evolution import freeze
from .v3.execution import HostBroker, root_files, validate_answer_origin
from .v3.infrastructure import StructuredModel
from .v3.rag import RagEngine
from .v3.reader_replay import SCHEMA, ReplayRouter, validate_case

PACKET_SCHEMA = "rag-rsi-reader-replay-cases-1"
SELECTION_RULE = "all_source_question_ids_in_frozen_order__support__repeat_0__without_scores"


def _snapshot_file(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def _unchanged(binding):
    if hashlib.sha256(Path(binding["path"]).read_bytes()).hexdigest() != binding["sha256"]:
        raise ValueError("source artifact changed during answer preparation")


class _RequestShape(StructuredModel):
    """Reuse the exact provider body constructor without I/O initialization."""
    def __init__(self, spec):
        self.model = spec["name"]
        self.limits = {"plan": 1200, "read": 2200, "answer": 800, "develop": 18000}
        self.limits.update(spec["output_limits"])
        self.max_input_bytes = spec["max_input_bytes"]


def _build_current_cases(source_plan, *, source_arm="support"):
    """Return every selected-arm/repeat-0 state without opening scores.

    Accept the original plan dictionary or its JSON path. Membership, archive,
    execution, accounting and source runtime are checked through calibration's
    existing complete-generation validator before any state is reconstructed.
    """
    _selection_rule(source_arm)
    if isinstance(source_plan, (str, Path)):
        plan, supplied_binding = _snapshot_file(source_plan)
    elif isinstance(source_plan, dict):
        plan, supplied_binding = deepcopy(source_plan), None
    else:
        raise ValueError("source plan must be a frozen plan object or path")
    if plan.get("schema") != musique.SCHEMA or plan.get("data_role") != "D_fit":
        raise ValueError("only the original MuSiQue D_fit calibration is eligible")
    out = Path(plan["output_dir"]).resolve(strict=True)
    stored, plan_binding = _snapshot_file(out / "plan.json")
    if stored != plan:
        raise ValueError("source plan differs from its run")
    frozen, freeze_binding = _snapshot_file(out / "generation_freeze.json")
    cells = cal._validated_generation(plan, frozen)
    tasks = cal._task_snapshot(plan)
    byqid = {task["question_id"]: task for task in tasks}
    backends, _ = cal._musique_environment(plan, tasks)
    selected = {item["identity"]["question_id"]: item for item in cells
                if item["identity"]["arm"] == source_arm and item["identity"]["repeat"] == 0}
    if set(selected) != set(plan["question_ids"]):
        raise ValueError("source selected-arm repeat-0 panel incomplete")
    config = next(arm["config"] for arm in plan["arms"] if arm["name"] == source_arm)
    shape = _RequestShape(plan["model"])
    cases = []
    bindings = [plan_binding, freeze_binding] + ([supplied_binding] if supplied_binding else [])
    for ordinal, qid in enumerate(plan["question_ids"], 1):
        item, task = selected[qid], byqid[qid]
        receipt = item["payload"]
        validate_answer_origin(receipt)
        if (not receipt["execution_ok"] or not receipt["answer_origin_valid"]
                or not receipt["answer_usable"] or receipt["model_errors"]):
            raise ValueError("selected source state lacks a complete usable answer")
        raw_cell, cell_binding = _snapshot_file(cal._cell_path(out, item["file"]))
        if raw_cell["identity"] != item["identity"] or raw_cell["payload"] != receipt:
            raise ValueError("source cell changed during reconstruction")
        bindings.append(cell_binding)
        bank = f"calibration/{qid}/0"
        model_bindings = []

        class CachedModel:
            def complete(self, stage, payload):
                body = shape.request_body(stage, payload)
                key = digest({"body": body, "bank": bank})
                record, file_binding = _snapshot_file(out / "requests" / (key + ".json"))
                if (record.get("key") != key or record.get("state") != "settled"
                        or record.get("body") != body):
                    raise ValueError("source model request differs from its exact bank and body")
                choices = record.get("response", {}).get("choices", [])
                if not choices or choices[0].get("finish_reason") != "stop":
                    raise ValueError("source model result incomplete")
                result = json.loads(choices[0].get("message", {}).get("content", ""))
                if not isinstance(result, dict):
                    raise ValueError("source model result is not an object")
                model_bindings.append({"event_index": len(broker.events), "stage": stage,
                    "bank": bank, "request_key": key, "body_sha256": digest(body),
                    "response_sha256": digest(result), "file": file_binding})
                bindings.append(file_binding)
                return deepcopy(result)

        broker = HostBroker(task, backends[(qid, source_arm)], CachedModel(), **plan["limits"])
        events = []
        for index, event in enumerate(receipt["trace"]):
            name, request = event["name"], event["request"]
            if name not in {"complete", "search", "read", "record_trace"}:
                raise ValueError("unsupported source event")
            if name == "record_trace" and index != len(receipt["trace"]) - 1:
                raise ValueError("source diagnostic must follow the final answer")
            if name == "complete" and event.get("model_completed") is not True:
                raise ValueError("cannot replay incomplete source model events")
            response = broker(name, request)
            if digest(response) != event["response_hash"]:
                raise ValueError("source event response differs from reconstruction")
            if name != "record_trace":
                events.append({"name": name, "request": deepcopy(request),
                               "response": response, "response_sha256": digest(response)})
        if (broker.events != receipt["trace"] or broker.reported != receipt["candidate_reported"]
                or broker.reported is None or broker.model_errors != receipt["model_errors"]
                or broker.counts != receipt["resource_usage"]
                or {"read_presentations": broker.read_presentations,
                    "final_observations": broker.final_observations} != receipt["host_evidence_trace"]
                or broker.answer_origin_receipt(receipt["answer"]) != receipt["host_answer_origin_validation"]
                or broker.citation_receipt(receipt["answer"], receipt["citations"]) != receipt["host_citation_validation"]):
            raise ValueError("reconstructed host observations differ from source receipt")
        case = {"schema": SCHEMA, "case_id": f"Q{ordinal:02d}", "task": deepcopy(task),
            "config": deepcopy(config),
            "target_read": sum(e["name"] == "complete" and e["request"]["stage"] == "read" for e in events),
            "events": events,
            "engine_sha256": hashlib.sha256(root_files(config)["rag_core.py"].encode()).hexdigest(),
            "original_final_payload_sha256": digest(events[-1]["request"]["payload"]),
            "source_binding": {"kind": f"frozen_musique_calibration_{source_arm}_repeat_0",
                "selection_rule": _selection_rule(source_arm), "source_plan": plan_binding,
                "source_plan_hash": digest(plan), "source_generation_freeze": freeze_binding,
                "source_cell": cell_binding, "source_identity": deepcopy(item["identity"]),
                "model_requests": model_bindings}}
        case = validate_case(case)
        router = ReplayRouter(case)
        # Original config is essential: capping rounds can alter stop_reason.
        reconstructed = RagEngine(router, router, config=case["config"]).solve({"question": task["question"]})
        router.assert_complete()
        if reconstructed != receipt["candidate_reported"]:
            raise ValueError("original engine no longer reproduces the frozen state")
        cases.append(case)
    for binding in bindings:
        _unchanged(binding)
    return {"schema": PACKET_SCHEMA, "cases": cases}



# Only this maintained checkout may supply an earlier trusted runtime. Neither
# caller-supplied archive paths nor source code contained in a case is imported.
TRUSTED_SOURCE_CHECKOUT = Path("D:/Codex/Projects/rsi-agent-feedback")
TRUSTED_PACKAGE_INIT_SHA256 = "0f627a65febe0c7ddda592fccf3e45149e6950ac7852cbb148fad9463c37d7a5"
_SOURCE_RUNTIME_NAMES = frozenset((
    "archive.py", "budget.py", "candidate_runner.py", "linux_launcher.sh", "sandbox.py", "sdk.py",
    "v3/__init__.py", "v3/__main__.py", "v3/browsecomp_data.py", "v3/calibration.py",
    "v3/datasets.py", "v3/diagnostics.py", "v3/edit_scope.py", "v3/evolution.py",
    "v3/execution.py", "v3/experience_policy.py", "v3/fit_literal_audit.py", "v3/infrastructure.py",
    "v3/musique_calibration.py", "v3/offline_demo.py", "v3/paired_analysis.py", "v3/rag.py",
    "v3/reader_probe.py", "v3/reader_replay.py", "v3/request_recovery.py", "v3/task_metrics.py"))


def _selection_rule(source_arm):
    if source_arm not in {"support", "loop"}:
        raise ValueError("source_arm must be support or loop")
    return f"all_source_question_ids_in_frozen_order__{source_arm}__repeat_0__without_scores"


def _trusted_source_binding(plan, source_checkout):
    root = Path(source_checkout).resolve(strict=True)
    if root != TRUSTED_SOURCE_CHECKOUT.resolve(strict=True):
        raise ValueError("source checkout is not the allowlisted maintained runtime")
    declared = plan.get("runtime_source_hashes")
    if (not isinstance(declared, dict)
            or {name.replace("\\", "/") for name in declared} != _SOURCE_RUNTIME_NAMES):
        raise ValueError("source runtime manifest does not cover the trusted import closure")
    files = {}
    for name, expected in declared.items():
        normalized = name.replace("\\", "/")
        path = (root / "code_rsi" / normalized).resolve(strict=True)
        if not path.is_relative_to(root / "code_rsi"):
            raise ValueError("source runtime file escapes maintained checkout")
        raw = path.read_bytes()
        # Preserve the original protocol's JSON/text digest, including universal
        # newline handling; raw SHA below additionally binds physical bytes.
        if digest(raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")) != expected:
            raise ValueError("source runtime changed since the original plan")
        files[normalized] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    init = (root / "code_rsi" / "__init__.py").resolve(strict=True)
    if not init.is_relative_to(root / "code_rsi"):
        raise ValueError("source package initializer escapes maintained checkout")
    init_hash = hashlib.sha256(init.read_bytes()).hexdigest()
    if init_hash != TRUSTED_PACKAGE_INIT_SHA256:
        raise ValueError("unreviewed source package initializer")
    files["__init__.py"] = {"path": str(init), "sha256": init_hash}
    return {"checkout": str(root), "runtime_source_hashes": deepcopy(declared), "files": files}


# A stdlib-only child starts with no code_rsi imports. Its loader compiles only
# preverified maintained source snapshots, avoiding stale .pyc, arbitrary path
# imports and all archived candidate execution. Request caches provide outputs;
# the source builder never constructs a provider transport or reads references.
_SOURCE_CHILD = r"""
import hashlib, importlib.abc, importlib.util, json, sys
from pathlib import Path
job = json.load(sys.stdin)
root = Path(job['source']['checkout']).resolve(strict=True)
if root != Path('D:/Codex/Projects/rsi-agent-feedback').resolve(strict=True):
    raise ValueError('unapproved maintained checkout')
modules = {}
for name, binding in job['source']['files'].items():
    path = Path(binding['path']).resolve(strict=True)
    if path != (root/'code_rsi'/name).resolve(strict=True) or not path.is_relative_to(root/'code_rsi'):
        raise ValueError('source import path differs')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding['sha256']:
        raise ValueError('source changed before child import')
    if name.endswith('.py'):
        bits = ['code_rsi'] + name[:-3].split('/')
        package = bits[-1] == '__init__'
        if package: bits.pop()
        modules['.'.join(bits)] = (path, raw, package)
helper = job['helper']
path = Path(helper['path']).resolve(strict=True)
raw = path.read_bytes()
if hashlib.sha256(raw).hexdigest() != helper['sha256']:
    raise ValueError('preparation helper changed before child import')
modules['code_rsi._answer_source_builder'] = (path, raw, False)
class VerifiedModules(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'code_rsi' or fullname.startswith('code_rsi.'):
            if fullname not in modules:
                raise ImportError('unbound source runtime import: '+fullname)
            return importlib.util.spec_from_loader(fullname, self, is_package=modules[fullname][2])
    def create_module(self, spec): return None
    def exec_module(self, module):
        path, raw, package = modules[module.__name__]
        module.__file__ = str(path)
        if package: module.__path__ = [str(path.parent)]
        exec(compile(raw, str(path), 'exec'), module.__dict__)
sys.meta_path.insert(0, VerifiedModules())
from code_rsi import _answer_source_builder as builder
if builder._trusted_source_binding(job['plan'], root) != job['source']:
    raise ValueError('source manifest does not match frozen protocol')
packet = builder._build_current_cases(job['plan'], source_arm=job['source_arm'])
for binding in list(job['source']['files'].values()) + [helper]:
    builder._unchanged(binding)
print(json.dumps({'packet': packet, 'source': job['source'], 'helper': helper}, ensure_ascii=True))
"""


def _source_packet(plan, source_arm, source_checkout):
    source = _trusted_source_binding(plan, source_checkout)
    helper = {"path": str(Path(__file__).resolve()),
              "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    job = {"plan": plan, "source_arm": source_arm, "source": source, "helper": helper}
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", _SOURCE_CHILD],
                               input=json.dumps(job, ensure_ascii=True), capture_output=True,
                               text=True, encoding="utf-8", timeout=180, check=False)
    if completed.returncode:
        # Do not echo private input/model payloads from child process failures.
        raise ValueError("trusted source reconstruction failed; no case was accepted")
    result = json.loads(completed.stdout)
    if (not isinstance(result, dict) or set(result) != {"packet", "source", "helper"}
            or result["source"] != source or result["helper"] != helper):
        raise ValueError("source reconstruction identity differs")
    for binding in list(source["files"].values()) + [helper]:
        _unchanged(binding)
    return result


def _migrate_packet(plan, source_arm, reconstructed):
    packet = deepcopy(reconstructed["packet"])
    if (not isinstance(packet, dict) or set(packet) != {"schema", "cases"}
            or packet["schema"] != PACKET_SCHEMA
            or not isinstance(packet["cases"], list)
            or len(packet["cases"]) != len(plan["question_ids"])):
        raise ValueError("complete source packet required")
    bindings = list(reconstructed["source"]["files"].values()) + [reconstructed["helper"]]
    for ordinal, (qid, case) in enumerate(zip(plan["question_ids"], packet["cases"]), 1):
        source = case["source_binding"]
        identity = source["source_identity"]
        if (case["case_id"] != f"Q{ordinal:02d}" or case["task"]["question_id"] != qid
                or identity["question_id"] != qid or identity["arm"] != source_arm
                or identity["repeat"] != 0 or source["source_plan_hash"] != digest(plan)
                or source["selection_rule"] != _selection_rule(source_arm)):
            raise ValueError("source migration selection differs")
        old_engine = case["engine_sha256"]
        raw_cell, raw_binding = _snapshot_file(source["source_cell"]["path"])
        if raw_binding != source["source_cell"] or raw_cell["identity"] != identity:
            raise ValueError("source cell changed before migration")
        # The old validator already checked this entire frozen receipt. Its
        # candidate state is compared, never used to produce new source code.
        old_result = raw_cell["payload"]["candidate_reported"]
        case["engine_sha256"] = hashlib.sha256(root_files(case["config"])["rag_core.py"].encode()).hexdigest()
        case = validate_case(case)
        router = ReplayRouter(case)
        current_result = RagEngine(router, router, config=case["config"]).solve({"question": case["task"]["question"]})
        replay = router.assert_complete()
        if current_result != old_result:
            raise ValueError("current engine differs from the complete original frozen state")
        source = case["source_binding"]
        source["runtime_migration"] = {
            "schema": "rag-rsi-trusted-source-migration-1",
            "source_arm": source_arm, "source_runtime": deepcopy(reconstructed["source"]),
            "preparation_helper": deepcopy(reconstructed["helper"]),
            "source_engine_sha256": old_engine, "current_engine_sha256": case["engine_sha256"],
            "original_candidate_reported_sha256": digest(old_result),
            "current_candidate_reported_sha256": digest(current_result),
            "all_requests_identical": True, "events_sha256": digest(case["events"]),
            "replay": replay}
        bindings.extend([source["source_plan"], source["source_generation_freeze"], source["source_cell"]])
        bindings.extend(request["file"] for request in source["model_requests"])
        packet["cases"][ordinal - 1] = case
    for binding in bindings:
        _unchanged(binding)
    return packet


def build_cases(source_plan, *, source_arm="support", source_checkout=None):
    """Mechanically select a complete repeat-0 arm; never select using scores.

    Default same-runtime reconstruction stays backward compatible. Explicit
    cross-runtime reconstruction checks the frozen source under its own
    trusted maintained code, then proves every current request and full result
    identical before rebinding the case to the current engine.
    """
    _selection_rule(source_arm)
    if source_checkout is None:
        return _build_current_cases(source_plan, source_arm=source_arm)
    if isinstance(source_plan, (str, Path)):
        plan, supplied_binding = _snapshot_file(source_plan)
    elif isinstance(source_plan, dict):
        plan, supplied_binding = deepcopy(source_plan), None
    else:
        raise ValueError("source plan must be a frozen plan object or path")
    if plan.get("schema") != musique.SCHEMA or plan.get("data_role") != "D_fit":
        raise ValueError("only the original MuSiQue D_fit calibration is eligible")
    result = _migrate_packet(plan, source_arm, _source_packet(plan, source_arm, source_checkout))
    if supplied_binding is not None:
        _unchanged(supplied_binding)
    return result


def prepare_program_inputs(source_plan, *, source_revision, output_dir):
    """Freeze all measured archived programs against the same paired-root prefix.

    This is preparation only: candidate execution stays in WSL, old answers are
    placeholders for capturing requests, and neither API credentials nor answer
    references are read. The resulting artifacts do not authorize a paid run.
    """
    from .archive import ProgramArchive
    from .program_answer_prepare import build_paired_root_cases
    from .program_answer_artifacts import (PROGRAMS_SCHEMA, accepted_program_sources,
                                           capture_programs)
    target = Path(output_dir).resolve()
    runs = Path(__file__).resolve().parents[1] / "runs"
    if runs.resolve() not in target.parents:
        raise ValueError("program preparation must remain under project runs")
    source, source_binding = _snapshot_file(source_plan)
    packet = build_paired_root_cases(source_plan, source_revision=source_revision)
    programs = []
    for item in accepted_program_sources(source):
        origin = item["source"]
        archive = ProgramArchive(origin["archive_dir"])
        loaded = archive.load_program(origin["program_id"])
        programs.append({"name": item["name"], "source": deepcopy(origin), "files": loaded["files"]})
    bundle = {"schema": PROGRAMS_SCHEMA, "source_plan_hash": digest(source), "programs": programs}
    model = deepcopy(source["search_template"]["model"])
    model["output_limits"] = {"answer": model["output_limits"]["answer"]}
    freeze(target / "cases.json", packet)
    freeze(target / "programs.json", bundle)
    projections = capture_programs(packet, bundle, model, target / "captures")
    freeze(target / "projections.json", projections)
    _unchanged(source_binding)
    def binding(name):
        path = target / name
        return {"path": str(path), "sha256": cal.file_hash(path)}
    result = {"schema": "rag-rsi-program-answer-preparation-1", "source_plan_file": source_binding,
        "source_revision": source_revision, "cases_file": binding("cases.json"),
        "programs_file": binding("programs.json"), "projections_file": binding("projections.json"),
        "model": model, "cases": len(packet["cases"]), "programs": len(programs),
        "captured_requests": len(projections["projections"]),
        "new_api_calls": 0, "references_parsed": False, "credentials_read": False,
        "old_answer_is_measurement": False, "paid_execution_authorized": False}
    freeze(target / "preparation.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-arm", choices=("support", "loop"), default="support")
    parser.add_argument("--source-checkout")
    parser.add_argument("--paired-programs", action="store_true")
    parser.add_argument("--source-revision")
    args = parser.parse_args(argv)
    if args.paired_programs:
        if args.source_checkout is not None or args.source_arm != "support":
            raise ValueError("paired program preparation cannot mix calibration source flags")
        result = prepare_program_inputs(args.source_plan, source_revision=args.source_revision,
                                        output_dir=args.output)
        print(json.dumps(result, ensure_ascii=True))
        return result
    if args.source_revision is not None:
        raise ValueError("source_revision requires paired-program preparation")
    target = Path(args.output).resolve()
    runs = Path(__file__).resolve().parents[1] / "runs"
    if runs.resolve() not in target.parents:
        raise ValueError("answer preparation output must remain under project runs")
    packet = build_cases(args.source_plan, source_arm=args.source_arm, source_checkout=args.source_checkout)
    freeze(target, packet)
    print(json.dumps({"cases": len(packet["cases"]), "packet_hash": digest(packet),
        "file_sha256": cal.file_hash(target), "source_selection": _selection_rule(args.source_arm),
        "new_api_calls": 0, "references_parsed": False, "credentials_read": False}))
    return packet


if __name__ == "__main__":
    main()
