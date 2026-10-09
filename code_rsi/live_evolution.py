"""Frozen live entrypoint for the existing v3 evolution runner.

Preflight reads local public tasks/private references for host validation only.
Schema3/4/5 preflight only hashes references; phases unlock their own reference map.
Schema4 freezes an edit policy; schema5 also freezes the proposal representation.
References never enter the ProgramDeveloper model payload. No API or credential
access occurs on import/preflight. This file is outside v3's frozen source set.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .budget import Ledger, digest, save
from .v3.calibration import credential_from_plan, run_lock
from .v3.datasets import validate_task_collection
from .v3.evolution import (EvolutionRunner, ProgramDeveloper, _runtime_source_hashes,
                           freeze, validate_controls)
from .v3.execution import HostError
from .v3.infrastructure import (BrowseCompCorpus, StructuredModel, UnknownProviderOutcome,
                                deepseek_transport, DeepSeekTransport, PROMPTS, proposal_model_prompts)
from .v3.rag import RagEngine
from .v3.edit_policy import validate_edit_policy
from .v3.proposal_protocol import validate_proposal_protocol
from .v3.request_recovery import (check_request_recovery as _check_request_recovery,
                                   check_request_accounting as _check_request_accounting)

SCHEMA = "rag-rsi-live-evolution-1"
SCHEMA2 = "rag-rsi-live-evolution-2"
SCHEMA3 = "rag-rsi-live-evolution-3"
SCHEMA4 = "rag-rsi-live-evolution-4"
SCHEMA5 = "rag-rsi-live-evolution-5"
GROUPS_SCHEMA = "rag-rsi-role-reference-groups-1"
PHASE_ROLES = {"search": "D_fit", "select": "D_select", "report": "D_report"}
PROPOSAL_BANK_POLICY = "host_proposal_slot_v1"
ROLES = ("D_fit", "D_select", "D_report")
STAGES = ("plan", "read", "answer", "develop")
PROJECT = Path(__file__).resolve().parent.parent
FIELDS = {"schema", "purpose", "output_dir", "panels", "corpus", "corpus_ref", "model",
          "root_config", "limits", "repeats", "expansions", "select_candidates", "metric",
          "allow_proxy_metric", "synthetic", "max_calls", "hard_cny", "runtime_source_hashes",
          "entry_sha256", "credential_source"}
FIELDS_V2 = FIELDS | {"controls"}
FIELDS_V3 = FIELDS_V2 | {"phase_order", "reference_groups_file"}
FIELDS_V4 = FIELDS_V3 | {"edit_policy"}
FIELDS_V5 = FIELDS_V4 | {"proposal_protocol"}


def _hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _file(item, *, parse=False):
    if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
        raise ValueError("frozen file requires path and sha256")
    path = Path(item["path"]).resolve(strict=True)
    if parse:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("frozen input bytes changed")
        return json.loads(raw)
    if _hash(path) != item["sha256"]:
        raise ValueError("frozen corpus changed")
    return path


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("invalid " + name)
    return value


class _NoIO:
    def search(self, *args):
        raise AssertionError("preflight cannot search")

    def complete(self, *args):
        raise AssertionError("preflight cannot call a model")


def _model_identity(model, prompts, proposal_protocol=None):
    protocol = validate_proposal_protocol(proposal_protocol)
    return digest({"model": model["name"], "prompts": proposal_model_prompts(protocol, prompts), "limits": model["output_limits"],
                   "max_input_bytes": model["max_input_bytes"], "temperature": 0,
                   "thinking": "disabled", "response_format": "json_object", "prices": model["prices"],
                   **({"proposal_protocol":protocol} if protocol is not None else {})})


def _prepare(plan, *, phase=None):
    if not isinstance(plan, dict):
        raise ValueError("exact live-evolution plan schema required")
    schema = plan.get("schema")
    fields = (FIELDS if schema == SCHEMA else FIELDS_V2 if schema == SCHEMA2 else
              FIELDS_V3 if schema == SCHEMA3 else FIELDS_V4 if schema == SCHEMA4 else
              FIELDS_V5 if schema == SCHEMA5 else None)
    if fields is None or set(plan) != fields:
        raise ValueError("exact live-evolution plan schema required")
    staged = schema in (SCHEMA3, SCHEMA4, SCHEMA5)
    edit_policy = validate_edit_policy(plan["edit_policy"]) if schema in (SCHEMA4, SCHEMA5) else None
    proposal_protocol = validate_proposal_protocol(plan["proposal_protocol"]) if schema == SCHEMA5 else None
    if schema == SCHEMA5 and proposal_protocol is None:
        raise ValueError("schema5 requires an explicit proposal protocol")
    if schema in (SCHEMA4, SCHEMA5) and edit_policy is None:
        raise ValueError("schema4 requires an explicit edit policy")
    if staged:
        if plan["phase_order"] not in (["search"], ["search", "select", "report"]):
            raise ValueError("exact supported phase_order required")
        if phase not in plan["phase_order"]:
            raise ValueError("schema3 requires an explicit declared phase")
        roles = tuple(PHASE_ROLES[name] for name in plan["phase_order"])
    else:
        if phase is not None:
            raise ValueError("legacy plans cannot select a phase")
        roles = ROLES
    if plan["purpose"] != "development_evolution":
        raise ValueError("explicit development_evolution purpose required")
    out = Path(plan["output_dir"]).resolve()
    if (PROJECT / "runs").resolve() not in out.parents:
        raise ValueError("output must be a child of this project's runs directory")
    if plan["runtime_source_hashes"] != _runtime_source_hashes() or plan["entry_sha256"] != _hash(__file__):
        raise ValueError("runtime or live entry differs from the approved source identity")
    if type(plan["synthetic"]) is not bool or type(plan["allow_proxy_metric"]) is not bool:
        raise ValueError("synthetic and allow_proxy_metric must be explicit booleans")
    if plan["metric"] not in {"em", "f1"}:
        raise ValueError("live entry supports frozen local em/f1 only; no external judge")
    source = plan["credential_source"]
    if (not isinstance(source, dict) or set(source) != {"kind", "variable", "path"}
            or source["kind"] != "env_file" or source["variable"] != "DEEPSEEK_API_KEY"
            or not isinstance(source["path"], str) or not Path(source["path"]).is_absolute()):
        raise ValueError("explicit absolute credential source required; preflight does not open it")
    if not isinstance(plan["panels"], dict) or set(plan["panels"]) != set(roles):
        raise ValueError("exact declared role-scoped input file bindings required")
    group_metadata = None
    if staged:
        value = _file(plan["reference_groups_file"], parse=True)
        if (not isinstance(value, dict) or set(value) != {"schema", "groups"}
                or value["schema"] != GROUPS_SCHEMA or not isinstance(value["groups"], dict)
                or set(value["groups"]) != set(roles)):
            raise ValueError("exact role reference-groups metadata required")
        group_metadata = value["groups"]
    panels, references, seen, groups = {}, {}, {}, {}
    needs_corpus = False
    for role in roles:
        files = plan["panels"][role]
        if not isinstance(files, dict) or set(files) != {"tasks_file", "references_file"}:
            raise ValueError("each role requires tasks_file and references_file")
        tasks = _file(files["tasks_file"], parse=True)
        refs = None if staged else _file(files["references_file"], parse=True)
        if staged:
            _file(files["references_file"])  # Hash bytes only; no JSON parsing here.
        validate_task_collection(tasks)
        if not tasks or (not staged and (not isinstance(refs, dict) or set(refs) != {t["question_id"] for t in tasks})):
            raise ValueError("nonempty tasks and exact role-scoped reference map required")
        if staged and (not isinstance(group_metadata[role], dict)
                or set(group_metadata[role]) != {t["question_id"] for t in tasks}
                or any(v is not None and (not isinstance(v, str) or not v.strip())
                       for v in group_metadata[role].values())):
            raise ValueError("reference group metadata must exactly cover each role")
        for task in tasks:
            qid = task["question_id"]
            ref = None if staged else refs[qid]
            if task["task_type"] != "qa" or task["dataset"] not in {"musique", "browsecomp-plus", "multihop-rag"}:
                raise ValueError("only answerable QA panels are supported")
            if not staged and (not isinstance(ref, dict) or ref.get("question_id") != qid
                    or ref.get("dataset") != task["dataset"] or ref.get("reference_available") is not True
                    or not isinstance(ref.get("answers"), list) or not ref["answers"]
                    or any(not isinstance(a, str) or not a.strip() for a in ref["answers"])
                    or ref.get("answerable") is False):
                raise ValueError("complete available answer references required; MuSiQue-Full is unsupported")
            if task["dataset"] in {"browsecomp-plus", "multihop-rag"} and not plan["allow_proxy_metric"]:
                raise ValueError("BCP/MultiHop local em/f1 requires explicit proxy acknowledgement")
            for key in ("id:" + qid, "question:" + digest(task["question"].strip().casefold())):
                if key in seen and seen[key] != role:
                    raise ValueError("question overlap across roles")
                seen[key] = role
            group = group_metadata[role][qid] if staged else ref.get("pair_group_id", ref.get("source_question_id"))
            if group is not None:
                key = (task["dataset"], str(group))
                if key in groups and groups[key] != role:
                    raise ValueError("source question group crosses roles")
                groups[key] = role
            if not task["documents"]:
                needs_corpus = True
                if task["dataset"] != "browsecomp-plus" or task["corpus_ref"] != plan["corpus_ref"]:
                    raise ValueError("shared live corpus must be the bound BrowseComp index")
        panels[role] = tasks
        if not staged:
            references[role] = refs
    if needs_corpus:
        if not isinstance(plan["corpus_ref"], str) or not plan["corpus_ref"]:
            raise ValueError("shared corpus reference required")
        _file(plan["corpus"])
    elif plan["corpus"] is not None or plan["corpus_ref"] is not None:
        raise ValueError("question-local runs use null corpus bindings")
    stub = _NoIO()
    config = RagEngine(stub, stub, config=plan["root_config"]).config
    if not isinstance(plan["limits"], dict) or set(plan["limits"]) != {"max_models", "max_searches", "max_reads"}:
        raise ValueError("complete host limits required")
    for name, value in plan["limits"].items():
        _integer(value, 1, 64, name)
    rounds = config["max_rounds"] if config["mode"] == "iterative" else 1
    if config["search_limit"]>30:
        raise ValueError("search_limit exceeds host maximum of 30")
    searches=1 if config["mode"]=="single_pass" else rounds*min(config["max_queries_per_round"],24)
    if plan["limits"]["max_searches"]<searches:
        raise ValueError("root search budget cannot cover declared workflow queries")
    root_calls = int(config["mode"] in {"iterative", "planned_single"}) + rounds + 1
    if min(config["max_model_calls"], plan["limits"]["max_models"]) < root_calls:
        raise ValueError("root profile cannot reserve all declared rounds and final")
    repeats = _integer(plan["repeats"], 1, 8, "repeats")
    controls = (validate_controls(plan["controls"], panels["D_fit"], repeats, allow_legacy=False)
                if schema in (SCHEMA2, SCHEMA3, SCHEMA4, SCHEMA5) else None)
    expansions = _integer(plan["expansions"], 0, 16, "expansions")
    candidates = _integer(plan["select_candidates"], 1, 17, "select_candidates")
    model = plan["model"]
    if (not isinstance(model, dict) or set(model) != {"name", "thinking", "temperature", "max_input_bytes", "output_limits", "prices"}
            or model["name"] != "deepseek-flash" or model["thinking"] != "disabled"
            or type(model["temperature"]) not in (int, float) or model["temperature"] != 0):
        raise ValueError("explicit frozen DeepSeek generation settings required")
    _integer(model["max_input_bytes"], 1000, 120000, "max_input_bytes")
    if not isinstance(model["output_limits"], dict) or set(model["output_limits"]) != set(STAGES):
        raise ValueError("plan/read/answer/develop output caps required")
    for stage in STAGES:
        _integer(model["output_limits"][stage], 1, 32768, "output cap")
    prices = model["prices"]
    if (not isinstance(prices, dict) or set(prices) != {"input_hit", "input_miss", "output"}
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in prices.values())
            or prices["input_hit"] > prices["input_miss"]):
        raise ValueError("invalid CNY price envelope")
    nodes = expansions + 1
    outcome_counts = {"search": repeats * nodes * len(panels["D_fit"])}
    if "D_select" in panels:
        outcome_counts["select"] = repeats * min(candidates, nodes) * len(panels["D_select"])
        outcome_counts["report"] = repeats * 2 * len(panels["D_report"])
    outcomes = sum(outcome_counts.values())
    qa_calls = outcomes * plan["limits"]["max_models"]
    calls = qa_calls + expansions
    _integer(plan["max_calls"], 1, 1000000, "max_calls")
    if plan["max_calls"] != calls:
        raise ValueError("max_calls differs from complete evolution structural ceiling")
    output = qa_calls * max(model["output_limits"][s] for s in ("plan", "read", "answer"))
    output += expansions * model["output_limits"]["develop"]
    worst = ((model["max_input_bytes"] + 1024) * calls * prices["input_miss"] + output * prices["output"]) / 1e6
    hard = plan["hard_cny"]
    if type(hard) not in (int, float) or not math.isfinite(hard) or hard <= 0 or worst > hard:
        raise ValueError("hard_cny cannot cover the conservative complete-run envelope")
    report = {"schema": schema, "status": "ready_for_explicit_execution", "plan_hash": digest(plan),
              "question_counts": {r: len(panels[r]) for r in roles}, "max_calls": calls,
              "qa_call_ceiling": qa_calls, "developer_call_ceiling": expansions,
              "conservative_cny_upper_bound": round(worst, 6), "hard_cny": hard,
              "model_identity": _model_identity(model, PROMPTS, proposal_protocol), "new_api_calls": 0,
              "credentials_read": False, "private_references_read_locally": not staged,
              "developer_receives_gold": False, "synthetic": plan["synthetic"]}
    if proposal_protocol is not None:
        report["proposal_protocol"] = deepcopy(proposal_protocol)
    if edit_policy is not None:
        report["edit_policy"] = deepcopy(edit_policy)
    if controls is not None:
        report.update(controls=controls, proposal_bank_policy=PROPOSAL_BANK_POLICY)
    if staged:
        phase_limits = {}
        for name, count in outcome_counts.items():
            qa = count * plan["limits"]["max_models"]
            develop = expansions if name == "search" else 0
            output_bound = qa * max(model["output_limits"][s] for s in ("plan", "read", "answer")) + develop * model["output_limits"]["develop"]
            phase_limits[name] = {"calls": qa + develop,
                "cny": ((model["max_input_bytes"] + 1024) * (qa + develop) * prices["input_miss"] + output_bound * prices["output"]) / 1e6}
        report.update(phase=phase, phase_order=deepcopy(plan["phase_order"]),
            phase_limits=phase_limits, phase_prerequisites_checked=False,
            reference_roles_parsed=[], lifecycle={"schema": "rag-rsi-evolution-phases-1",
                "phase_order": deepcopy(plan["phase_order"]),
                "reference_bindings": {role: deepcopy(plan["panels"][role]["references_file"]) for role in roles},
                "reference_groups": deepcopy(group_metadata)})
    return report, panels, references


def preflight(plan, *, phase=None):
    return _prepare(deepcopy(plan), phase=phase)[0]


def _reference_loader(plan, panels, groups):
    """Called by the core only after its preceding-phase seal check."""
    def load(role):
        if role not in panels:
            raise ValueError("reference role is outside the declared lifecycle")
        _file(plan["reference_groups_file"])
        refs = _file(plan["panels"][role]["references_file"], parse=True)
        tasks = panels[role]
        if not isinstance(refs, dict) or set(refs) != {task["question_id"] for task in tasks}:
            raise ValueError("exact role-scoped reference map required")
        for task in tasks:
            qid, dataset = task["question_id"], task["dataset"]
            ref = refs[qid]
            if (not isinstance(ref, dict) or ref.get("question_id") != qid
                    or ref.get("dataset") != dataset or ref.get("reference_available") is not True
                    or not isinstance(ref.get("answers"), list) or not ref["answers"]
                    or any(not isinstance(a, str) or not a.strip() for a in ref["answers"])
                    or ref.get("answerable") is False):
                raise ValueError("complete available answer references required")
            group = ref.get("pair_group_id", ref.get("source_question_id"))
            if group is not None and (not isinstance(group, (str, int)) or isinstance(group, bool)):
                raise ValueError("invalid private source group")
            if (None if group is None else str(group)) != groups[role][qid]:
                raise ValueError("reference group differs from frozen preparation metadata")
        return refs
    return load


class _PhaseLedger:
    """Use one append-only ledger; each reservation charges two fixed scopes."""
    def __init__(self, ledger, phase):
        self.ledger, self.phase = ledger, phase

    def reserve(self, scopes, amount, metadata=None):
        if scopes != ["run"]:
            raise ValueError("phase requests must charge the shared run scope")
        return self.ledger.reserve(["run", "phase:" + self.phase], amount,
            {**(metadata or {}), "execution_phase": self.phase})

    def settle(self, *args, **kwargs):
        return self.ledger.settle(*args, **kwargs)


class _LazyTransport:
    def __init__(self, factory):
        self.factory, self.transport = factory, None

    def send(self, body, timeout):
        if self.transport is None:
            self.transport = self.factory()
        target = self.transport
        return target.send(body, timeout) if hasattr(target, "send") else target(body)


class _LazyLiveTransport(_LazyTransport, DeepSeekTransport):
    # Preserve StructuredModel's live-response model-identity requirement.
    pass


def _phase_for_bank(bank):
    if bank == "develop" or bank.startswith("develop/"):
        return "search"
    return next((phase for phase, role in PHASE_ROLES.items() if bank.startswith(role + "/")), None)


def _phase_accounting(ledger, phase_order):
    for event in ledger.events:
        if event["event"] != "reserve":
            continue
        phase = event.get("metadata", {}).get("execution_phase")
        bank = event.get("metadata", {}).get("bank", "")
        if (phase not in phase_order or event.get("scopes") != ["run", "phase:" + phase]
                or not isinstance(bank, str) or _phase_for_bank(bank) != phase):
            raise HostError("phase request scope or bank differs from its ledger")



class _BoundModel:
    """Stage capability and identity checks happen before every possible dispatch."""
    def __init__(self, model, expected, allowed, check, dispatch_check=None):
        self._model, self._expected, self._allowed, self._check = model, expected, frozenset(allowed), check
        self._dispatch_check = dispatch_check

    def _verify(self):
        self._check()
        current = {"name": self._model.model, "max_input_bytes": self._model.max_input_bytes,
                   "output_limits": self._model.limits, "prices": self._model.prices}
        if self._model.identity != self._expected or _model_identity(current, PROMPTS, self._model.proposal_protocol) != self._expected:
            raise HostError("live model identity drift; no dispatch allowed")

    @property
    def identity(self):
        self._verify()
        return self._expected

    @property
    def max_input_bytes(self):
        return self._model.max_input_bytes

    def request_body(self, stage, payload):
        if stage not in self._allowed:
            raise HostError("model role cannot inspect this stage")
        self._verify()
        return self._model.request_body(stage, payload)

    def request_size(self, stage, payload):
        if stage not in self._allowed:
            raise HostError("model role cannot size this stage")
        self._verify()
        return self._model.request_size(stage, payload)

    @property
    def timeout_seconds(self):
        return self._model.timeout_seconds

    @timeout_seconds.setter
    def timeout_seconds(self, value):
        self._model.timeout_seconds = value

    def complete(self, stage, payload):
        if self._dispatch_check is not None:
            self._dispatch_check()
        if stage not in self._allowed:
            raise HostError("model role cannot call this stage")
        self._verify()
        return self._model.complete(stage, payload)


def run(plan, *, approved_plan_hash, execute=False, transport_factory=None, phase=None):
    plan = deepcopy(plan)
    check, panels, references = _prepare(plan, phase=phase)
    if execute is not True or approved_plan_hash != check["plan_hash"]:
        raise ValueError("execute and the exact approved plan hash are required")
    if plan["synthetic"] != (transport_factory is not None):
        raise ValueError("synthetic runs require an injected transport; live runs forbid injection")
    staged = plan["schema"] in (SCHEMA3, SCHEMA4, SCHEMA5)
    out = Path(plan["output_dir"]).resolve()
    with run_lock(out):
        freeze(out / "live_plan.json", plan)
        records = _check_request_recovery(out)
        ledger_limits = {"run": {"calls": plan["max_calls"], "cny": plan["hard_cny"]}}
        if staged:
            ledger_limits.update({"phase:" + key: value for key, value in check["phase_limits"].items()})
        ledger = Ledger(out / "ledger.jsonl", ledger_limits)
        _check_request_accounting(records, ledger)
        if staged:
            _phase_accounting(ledger, plan["phase_order"])
        corpus_stamp = None
        if plan["corpus"] is not None:
            stat = Path(plan["corpus"]["path"]).stat()
            corpus_stamp = (stat.st_size, stat.st_mtime_ns)

        def check_binding():
            if _runtime_source_hashes() != plan["runtime_source_hashes"] or _hash(__file__) != plan["entry_sha256"]:
                raise HostError("frozen runtime identity drift; no dispatch allowed")
            if corpus_stamp is not None:
                stat = Path(plan["corpus"]["path"]).stat()
                if (stat.st_size, stat.st_mtime_ns) != corpus_stamp:
                    raise HostError("frozen corpus changed; no dispatch allowed")

        if staged:
            transport = (_LazyTransport(transport_factory) if transport_factory is not None else
                         _LazyLiveTransport(lambda: deepseek_transport(credential_from_plan(plan))))
        else:
            transport = transport_factory() if transport_factory is not None else deepseek_transport(credential_from_plan(plan))
        model_config = plan["model"]

        def bound(bank, stages):
            check_binding()
            scoped_ledger = _PhaseLedger(ledger, phase) if staged else ledger
            model = StructuredModel(out / "requests", scoped_ledger, transport, bank=bank,
                                    prices=model_config["prices"], model=model_config["name"],
                                    max_input_bytes=model_config["max_input_bytes"], limits=model_config["output_limits"], proposal_protocol=check.get("proposal_protocol"))
            def check_phase_bank():
                if _phase_for_bank(bank) != phase:
                    raise HostError("model bank is outside the requested execution phase")
            wrapped = _BoundModel(model, check["model_identity"], stages, check_binding,
                                  dispatch_check=check_phase_bank if staged else None)
            wrapped.identity
            return wrapped

        corpora = {}
        def backend_factory(task):
            check_binding()
            excluded = tuple(sorted(task["excluded_docids"]))
            if excluded not in corpora:
                corpora[excluded] = BrowseCompCorpus(plan["corpus"]["path"], corpus_hash=plan["corpus"]["sha256"], excluded=excluded)
            return corpora[excluded]

        manifest = {"schema": "rag-rsi-v3-run-1", "live_plan_hash": check["plan_hash"],
                    "model_identity": check["model_identity"], "synthetic": plan["synthetic"],
                    **{key: deepcopy(plan[key]) for key in ("metric", "expansions", "repeats", "root_config",
                                                          "select_candidates", "limits", "allow_proxy_metric")}}
        if plan["schema"] in (SCHEMA4, SCHEMA5):
            manifest["edit_policy"] = deepcopy(check["edit_policy"])
        if plan["schema"] == SCHEMA5:
            manifest["proposal_protocol"] = deepcopy(check["proposal_protocol"])
        if plan["schema"] in (SCHEMA2, SCHEMA3, SCHEMA4, SCHEMA5):
            controls = deepcopy(check["controls"])
            manifest.update(controls=controls, proposal_bank_policy=PROPOSAL_BANK_POLICY)
            developer = ProgramDeveloper(bound("develop", ("develop",)),
                feedback_condition=controls["feedback"], case_schedule=controls["case_schedule"],
                proposal_model_factory=lambda slot: bound(f"develop/proposal/{slot}", ("develop",)),
                edit_policy=check.get("edit_policy"), proposal_protocol=check.get("proposal_protocol"))
        else:
            # Retain the original manifest and proposal cache identity for v1 resumes.
            developer = ProgramDeveloper(bound("develop", ("develop",)))
        if staged:
            manifest.update(lifecycle=deepcopy(check["lifecycle"]),
                            reference_groups_file=deepcopy(plan["reference_groups_file"]))
        runner_kwargs = ({"reference_loader": _reference_loader(plan, panels, check["lifecycle"]["reference_groups"])}
                         if staged else {})
        runner = EvolutionRunner(out, manifest, panels, references,
                                 lambda bank: bound(bank, ("plan", "read", "answer")),
                                 developer, backend_factory=backend_factory, **runner_kwargs)
        try:
            if staged:
                runner.check_phase(phase)
                result = runner.run_phase(phase)
            else:
                result = runner.run()
            check_binding()
            if plan["corpus"] is not None:
                _file(plan["corpus"])
        except BaseException as exc:
            save(out / "live_status.json", {"status": "stopped", "reason_type": type(exc).__name__, "ledger": ledger.summary()})
            raise
        status = {"status": "phase_complete" if staged else "complete",
                  "plan_hash": check["plan_hash"], "ledger": ledger.summary()}
        if staged:
            _check_request_accounting(_check_request_recovery(out), ledger)
            _phase_accounting(ledger, plan["phase_order"])
            status["phase"] = phase
        save(out / "live_status.json", status)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Frozen live entry for the existing RAG RSI evolution runner")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run"):
        command = sub.add_parser(name)
        command.add_argument("--plan", required=True)
        command.add_argument("--phase", choices=tuple(PHASE_ROLES))
        if name == "run":
            command.add_argument("--execute", action="store_true", required=True)
            command.add_argument("--approved-plan-hash", required=True)
    args = parser.parse_args(argv)
    plan = json.loads(Path(args.plan).read_bytes())
    result = preflight(plan, phase=args.phase) if args.command == "preflight" else run(
        plan, approved_plan_hash=args.approved_plan_hash, execute=args.execute, phase=args.phase)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
