"""Frozen archived programs and WSL-captured final requests; no provider access."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .archive import ProgramArchive
from .answer_replay import ProgramAnswerReplayRouter
from .budget import digest
from .v3.calibration import _execution_facts, _verified_bytes, file_hash
from .v3.evolution import freeze, read, recoverable_record
from .v3.execution import HostBroker, execute, root_files
from .v3.reader_probe import _RequestShape, _file_contract, _limits
from .v3.reader_replay import validate_case

PROGRAMS_SCHEMA = "rag-rsi-program-answer-bundle-1"
PROJECTIONS_SCHEMA = "rag-rsi-program-answer-projections-1"
CAPTURE_SCHEMA = "rag-rsi-program-answer-capture-1"
PACKET_SCHEMA = "rag-rsi-reader-replay-cases-1"


def _hash_string(value):
    return (isinstance(value, str) and len(value) == 64
            and all(ch in "0123456789abcdef" for ch in value))


def _bound_json(binding):
    _file_contract(binding)
    return json.loads(_verified_bytes(binding))


def _archive(directory):
    path = Path(directory)
    if (not path.is_absolute() or not path.is_dir()
            or not (path / "programs").is_dir() or not (path / "nodes").is_dir()):
        raise ValueError("existing absolute program archive required")
    return ProgramArchive(path)


def _packet(packet):
    if (not isinstance(packet, dict) or set(packet) != {"schema", "cases"}
            or packet["schema"] != PACKET_SCHEMA or not isinstance(packet["cases"], list)
            or not 1 <= len(packet["cases"]) <= 64):
        raise ValueError("complete bounded prefix packet required")
    cases = [validate_case(case) for case in packet["cases"]]
    if (len({case["case_id"] for case in cases}) != len(cases)
            or len({case["task"]["question_id"] for case in cases}) != len(cases)):
        raise ValueError("one fixed prefix per distinct question required")
    return cases


def accepted_program_sources(source_plan):
    """Select every measured child in terminal schedule order, without scores.

    This reads frozen paired/search metadata and immutable archives only. It
    deliberately does not invoke a live preflight, reference loader or scorer.
    """
    if (not isinstance(source_plan, dict)
            or source_plan.get("schema") not in {"rag-rsi-paired-development-1", "rag-rsi-paired-development-2"}):
        raise ValueError("frozen paired source plan required")
    out = Path(source_plan["output_dir"])
    if not out.is_absolute() or read(out / "paired_plan.json") != source_plan:
        raise ValueError("paired source plan differs from its completed run")
    result = read(out / "paired_search_complete.json")
    blocks = source_plan.get("blocks")
    expansions = source_plan.get("search_template", {}).get("expansions")
    if type(blocks) is not int or blocks < 1 or type(expansions) is not int or expansions < 1:
        raise ValueError("invalid paired source opportunity count")
    count = blocks * 2 * expansions
    if (not isinstance(result, dict) or result.get("schema") != source_plan["schema"]
            or result.get("status") != "complete" or result.get("plan_hash") != digest(source_plan)
            or result.get("heldout_roles_used") is not False
            or result.get("completed_opportunities") != count or result.get("planned_opportunities") != count
            or not isinstance(result.get("terminal_records"), list)
            or len(result["terminal_records"]) != count):
        raise ValueError("complete matching paired terminal panel required")
    reference_archive = _archive(str(out / "blocks/0/cases/archive"))
    root = read(out / "blocks/0/cases/root.json")
    if not isinstance(root, dict) or "node_id" not in root:
        raise ValueError("paired reference root missing")
    root = reference_archive.load_node(root["node_id"])
    if root["parent_node_id"] is not None:
        raise ValueError("paired reference must be a root node")
    selected = [{"name": "reference", "source": {"archive_dir": str(reference_archive.root),
                 "node_id": root["node_id"], "program_id": root["program_id"]}}]
    seen = set()
    for index, record in enumerate(result["terminal_records"]):
        if (not isinstance(record, dict) or set(record) != {"schedule_index", "block", "condition", "slot", "attempt"}
                or record["schedule_index"] != index or read(out / "schedule" / f"{index}.json") != record
                or type(record["block"]) is not int or not 0 <= record["block"] < blocks
                or record["condition"] not in {"cases", "trace"}
                or type(record["slot"]) is not int or not 0 <= record["slot"] < expansions):
            raise ValueError("paired terminal schedule binding differs")
        identity = (record["block"], record["condition"], record["slot"])
        if identity in seen:
            raise ValueError("duplicate paired terminal opportunity")
        seen.add(identity)
        directory = out / "blocks" / str(record["block"]) / record["condition"]
        attempt = record["attempt"]
        if (not isinstance(attempt, dict) or read(directory / "steps" / str(record["slot"]) / "attempt.json") != attempt
                or attempt.get("role") != "D_fit" or attempt.get("step") != record["slot"]
                or attempt.get("status") not in {"measured", "rejected"}):
            raise ValueError("terminal attempt differs from its source step")
        if attempt["status"] != "measured":
            if attempt.get("node_id") is not None:
                raise ValueError("rejected opportunity cannot supply a candidate")
            continue
        archive = _archive(str(directory / "archive"))
        child = archive.load_node(attempt["node_id"])
        if (read(directory / "steps" / str(record["slot"]) / "child.json") != child
                or child["parent_node_id"] != attempt.get("parent_node_id")
                or child["parent_node_id"] is None
                or archive.load_node(child["parent_node_id"])["program_id"] != root["program_id"]):
            raise ValueError("measured candidate is not a child of the shared reference program")
        selected.append({"name": "candidate_" + str(len(selected)), "source": {
            "archive_dir": str(archive.root), "node_id": child["node_id"], "program_id": child["program_id"]}})
    if len(selected) not in (2, 3):
        raise ValueError("this diagnostic requires all one or two measured candidates")
    return selected


def _programs(bundle, cases):
    if (not isinstance(bundle, dict) or set(bundle) != {"schema", "source_plan_hash", "programs"}
            or bundle["schema"] != PROGRAMS_SCHEMA or not isinstance(bundle["programs"], list)
            or len(bundle["programs"]) not in (2, 3)):
        raise ValueError("versioned complete program bundle required")
    names = ["reference", "candidate_1", "candidate_2"][:len(bundle["programs"])]
    programs, metadata = deepcopy(bundle["programs"]), {}
    for name, program in zip(names, programs):
        if (not isinstance(program, dict) or set(program) != {"name", "files", "source"}
                or program["name"] != name or not isinstance(program["files"], dict)
                or set(program["files"]) != {"rag.py", "rag_core.py"}
                or any(not isinstance(text, str) or not text for text in program["files"].values())):
            raise ValueError("program names and exact two-file sources required")
        digest(program["files"])
        if name != "reference" and program["files"] == programs[0]["files"]:
            raise ValueError("candidate must actually differ from the reference source")
        source = program["source"]
        if source is None:
            metadata[name] = {}
            continue
        if (not isinstance(source, dict) or set(source) != {"archive_dir", "node_id", "program_id"}
                or not _hash_string(source["node_id"]) or not _hash_string(source["program_id"])
                or not isinstance(source["archive_dir"], str)):
            raise ValueError("immutable source archive binding required")
        archive = _archive(source["archive_dir"])
        node = archive.load_node(source["node_id"])
        loaded = archive.load_program(source["program_id"])
        if node["program_id"] != source["program_id"] or loaded["files"] != program["files"]:
            raise ValueError("program files differ from source archive")
        metadata[name] = loaded["metadata"]
    real = [program["source"] is not None for program in programs]
    if any(real) != all(real):
        raise ValueError("synthetic and real source bindings cannot be mixed")
    if all(real):
        if not _hash_string(bundle["source_plan_hash"]):
            raise ValueError("real bundle requires a source plan hash")
        root_source = programs[0]["source"]
        root = _archive(root_source["archive_dir"]).load_node(root_source["node_id"])
        if root["parent_node_id"] is not None:
            raise ValueError("reference binding is not a root")
        for program in programs[1:]:
            source = program["source"]; archive = _archive(source["archive_dir"])
            child = archive.load_node(source["node_id"])
            if (child["parent_node_id"] is None
                    or archive.load_node(child["parent_node_id"])["program_id"] != root["program_id"]):
                raise ValueError("candidate parent program differs from reference")
    elif bundle["source_plan_hash"] is not None:
        raise ValueError("synthetic bundle cannot claim a real source plan hash")
    if any(programs[0]["files"] != root_files(case["config"]) for case in cases):
        raise ValueError("reference program differs from the frozen case configurations")
    return programs, metadata, all(real)


def _capture_name(case, arm):
    return digest([case["case_id"], arm])[:24]


def _validate_capture(record, case, program, path, *, real):
    fields = {"schema", "case_id", "arm", "program_id", "case_sha256", "files_sha256",
              "final_payload", "receipt", "measurement_eligible", "final_response_replayed"}
    if (not isinstance(record, dict) or set(record) != fields or record["schema"] != CAPTURE_SCHEMA
            or record["case_id"] != case["case_id"] or record["arm"] != program["name"]
            or record["case_sha256"] != digest(case) or record["files_sha256"] != digest(program["files"])
            or record["measurement_eligible"] is not False or record["final_response_replayed"] is not True):
        raise ValueError("capture identity or non-measurement declaration differs")
    path = Path(path)
    if (not path.is_absolute() or path.name != "capture.json" or path.parent.name != _capture_name(case, program["name"])
            or path.parent.parent.name != "captures"):
        raise ValueError("capture path differs from its canonical identity")
    receipt = record["receipt"]
    if not isinstance(receipt, dict):
        raise ValueError("capture requires a completed execution receipt")
    archive = _archive(str(path.parent.parent.parent / "archive"))
    node = archive.load_node(receipt.get("node_id"))
    if (record["program_id"] != node["program_id"]
            or archive.load_program(node["program_id"])["files"] != program["files"]
            or (real and record["program_id"] != program["source"]["program_id"])):
        raise ValueError("captured program differs from the archived source")
    _execution_facts(receipt, {"node_id": node["node_id"], "program_id": node["program_id"],
                              "question_id": case["task"]["question_id"]})
    if (not receipt["execution_ok"] or not receipt["answer_origin_valid"] or not receipt["answer_usable"]
            or receipt["model_errors"] or (real and receipt.get("isolation_verified") is not True)):
        raise ValueError("capture requires complete origin-verified isolated execution")
    router = ProgramAnswerReplayRouter(case, capture=True)
    broker = HostBroker(case["task"], router, router, **_limits(case))
    for event in receipt["trace"]:
        broker(event["name"], event["request"])
    router.assert_complete()
    observed = {"read_presentations": broker.read_presentations, "final_observations": broker.final_observations}
    if (router.captured_payload != record["final_payload"] or not router.final_response_replayed
            or router.measurement_eligible is not False or router.new_calls != 0
            or broker.events != receipt["trace"] or observed != receipt["host_evidence_trace"]
            or broker.counts != receipt["resource_usage"] or broker.model_errors != receipt["model_errors"]
            or broker.reported != receipt["candidate_reported"]
            or broker.answer_origin_receipt(receipt["answer"]) != receipt["host_answer_origin_validation"]
            or broker.citation_receipt(receipt["answer"], receipt["citations"]) != receipt["host_citation_validation"]):
        raise ValueError("capture observations differ from exact historical-response replay")
    return deepcopy(record["final_payload"])


def capture_programs(packet, bundle, model, output_dir, *, executor=execute):
    """Capture every program/case via the existing executor, without API access."""
    cases = _packet(packet)
    programs, metadata, real = _programs(bundle, cases)
    shape = _RequestShape(model)
    digest(model)
    out = Path(output_dir).resolve()
    runs = Path(__file__).resolve().parents[1] / "runs"
    if runs.resolve() not in out.parents:
        raise ValueError("capture output must remain inside existing project runs")
    archive = ProgramArchive(out / "archive")
    nodes = {program["name"]: recoverable_record(archive, program["files"], metadata[program["name"]],
             session_id="program-answer-capture", attempt=index) for index, program in enumerate(programs)}
    projections = []
    for case in cases:
        for program in programs:
            arm = program["name"]; node = nodes[arm]
            path = out / "captures" / _capture_name(case, arm) / "capture.json"
            record = read(path)
            if record is None:
                router = ProgramAnswerReplayRouter(case, capture=True)
                receipt = executor(archive, node["node_id"], case["task"], router, router, path.parent, limits=_limits(case))
                router.assert_complete()
                record = {"schema": CAPTURE_SCHEMA, "case_id": case["case_id"], "arm": arm,
                    "program_id": node["program_id"], "case_sha256": digest(case), "files_sha256": digest(program["files"]),
                    "final_payload": router.captured_payload, "receipt": receipt,
                    "measurement_eligible": router.measurement_eligible, "final_response_replayed": router.final_response_replayed}
                _validate_capture(record, case, program, path, real=real)
                freeze(path, record)
            payload = _validate_capture(record, case, program, path, real=real)
            projections.append({"case_id": case["case_id"], "arm": arm, "final_payload": payload,
                "request_body_sha256": digest(shape.request_body("answer", payload)),
                "capture_file": {"path": str(path), "sha256": file_hash(path)}})
    return {"schema": PROJECTIONS_SCHEMA, "cases_sha256": digest(packet), "programs_sha256": digest(bundle),
            "model_sha256": digest(model), "projections": projections}


def validate_program_artifacts(plan, cases):
    """Verify immutable programs and captures without executing source or refs."""
    packet = {"schema": PACKET_SCHEMA, "cases": deepcopy(cases)}
    cases = _packet(packet)
    bundle = _bound_json(plan["programs_file"])
    programs, _, real = _programs(bundle, cases)
    if plan.get("question_use") == "used_diagnostic":
        if not real:
            raise ValueError("real diagnostic requires source archive provenance")
        source_plan = _bound_json(plan["source_plan_file"])
        if (bundle["source_plan_hash"] != digest(source_plan)
                or [{"name": p["name"], "source": p["source"]} for p in programs] != accepted_program_sources(source_plan)):
            raise ValueError("bundle omits or changes the complete accepted source panel")
    elif plan.get("question_use") != "synthetic" or real or plan.get("source_plan_file") is not None:
        raise ValueError("explicit synthetic or exposed-development source contract required")
    frozen = _bound_json(plan["projections_file"])
    if (not isinstance(frozen, dict) or set(frozen) != {"schema", "cases_sha256", "programs_sha256", "model_sha256", "projections"}
            or frozen["schema"] != PROJECTIONS_SCHEMA or frozen["cases_sha256"] != digest(packet)
            or frozen["programs_sha256"] != digest(bundle) or frozen["model_sha256"] != digest(plan["model"])
            or not isinstance(frozen["projections"], list)):
        raise ValueError("projection bundle differs from frozen cases, programs or model")
    expected = [(case["case_id"], program["name"]) for case in cases for program in programs]
    if len(frozen["projections"]) != len(expected):
        raise ValueError("complete program-by-case projection panel required")
    case_by_id = {case["case_id"]: case for case in cases}
    program_by_name = {program["name"]: program for program in programs}
    shape = _RequestShape(plan["model"]); payloads = {}
    for item, key in zip(frozen["projections"], expected):
        if (not isinstance(item, dict) or set(item) != {"case_id", "arm", "final_payload", "request_body_sha256", "capture_file"}
                or (item["case_id"], item["arm"]) != key):
            raise ValueError("projection order or identity differs")
        record = _bound_json(item["capture_file"])
        payload = _validate_capture(record, case_by_id[key[0]], program_by_name[key[1]],
                                    item["capture_file"]["path"], real=real)
        if (payload != item["final_payload"]
                or item["request_body_sha256"] != digest(shape.request_body("answer", payload))):
            raise ValueError("captured final request differs from frozen provider body")
        payloads[key] = payload
    return {"files": {p["name"]: deepcopy(p["files"]) for p in programs},
            "payloads": payloads, "programs": programs}
