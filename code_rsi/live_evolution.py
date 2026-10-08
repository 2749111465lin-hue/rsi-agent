"""Frozen live entrypoint for the existing v3 evolution runner.

Preflight reads local public tasks/private references for host validation only.
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
from .v3.evolution import EvolutionRunner, ProgramDeveloper, _runtime_source_hashes, freeze
from .v3.execution import HostError
from .v3.infrastructure import (BrowseCompCorpus, StructuredModel, UnknownProviderOutcome,
                                deepseek_transport, PROMPTS)
from .v3.rag import RagEngine

SCHEMA = "rag-rsi-live-evolution-1"
ROLES = ("D_fit", "D_select", "D_report")
STAGES = ("plan", "read", "answer", "develop")
PROJECT = Path(__file__).resolve().parent.parent
FIELDS = {"schema", "purpose", "output_dir", "panels", "corpus", "corpus_ref", "model",
          "root_config", "limits", "repeats", "expansions", "select_candidates", "metric",
          "allow_proxy_metric", "synthetic", "max_calls", "hard_cny", "runtime_source_hashes",
          "entry_sha256", "credential_source"}


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


def _model_identity(model, prompts):
    return digest({"model": model["name"], "prompts": prompts, "limits": model["output_limits"],
                   "max_input_bytes": model["max_input_bytes"], "temperature": 0,
                   "thinking": "disabled", "response_format": "json_object", "prices": model["prices"]})


def _prepare(plan):
    if not isinstance(plan, dict) or set(plan) != FIELDS or plan["schema"] != SCHEMA:
        raise ValueError("exact live-evolution plan schema required")
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
    if not isinstance(plan["panels"], dict) or set(plan["panels"]) != set(ROLES):
        raise ValueError("three role-scoped input file bindings required")
    panels, references, seen, groups = {}, {}, {}, {}
    needs_corpus = False
    for role in ROLES:
        files = plan["panels"][role]
        if not isinstance(files, dict) or set(files) != {"tasks_file", "references_file"}:
            raise ValueError("each role requires tasks_file and references_file")
        tasks, refs = _file(files["tasks_file"], parse=True), _file(files["references_file"], parse=True)
        validate_task_collection(tasks)
        if not tasks or not isinstance(refs, dict) or set(refs) != {t["question_id"] for t in tasks}:
            raise ValueError("nonempty tasks and exact role-scoped reference map required")
        for task in tasks:
            qid = task["question_id"]
            ref = refs[qid]
            if task["task_type"] != "qa" or task["dataset"] not in {"musique", "browsecomp-plus", "multihop-rag"}:
                raise ValueError("only answerable QA panels are supported")
            if (not isinstance(ref, dict) or ref.get("question_id") != qid
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
            group = ref.get("pair_group_id", ref.get("source_question_id"))
            if group is not None:
                key = (ref["dataset"], str(group))
                if key in groups and groups[key] != role:
                    raise ValueError("source question group crosses roles")
                groups[key] = role
            if not task["documents"]:
                needs_corpus = True
                if task["dataset"] != "browsecomp-plus" or task["corpus_ref"] != plan["corpus_ref"]:
                    raise ValueError("shared live corpus must be the bound BrowseComp index")
        panels[role], references[role] = tasks, refs
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
    rounds = 1 if config["mode"] == "single_pass" else config["max_rounds"]
    root_calls = int(config["mode"] == "iterative") + rounds + 1
    if min(config["max_model_calls"], plan["limits"]["max_models"]) < root_calls:
        raise ValueError("root profile cannot reserve all declared rounds and final")
    repeats = _integer(plan["repeats"], 1, 8, "repeats")
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
    outcomes = repeats * (nodes * len(panels["D_fit"]) + min(candidates, nodes) * len(panels["D_select"])
                          + 2 * len(panels["D_report"]))
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
    report = {"schema": SCHEMA, "status": "ready_for_explicit_execution", "plan_hash": digest(plan),
              "question_counts": {r: len(panels[r]) for r in ROLES}, "max_calls": calls,
              "qa_call_ceiling": qa_calls, "developer_call_ceiling": expansions,
              "conservative_cny_upper_bound": round(worst, 6), "hard_cny": hard,
              "model_identity": _model_identity(model, PROMPTS), "new_api_calls": 0,
              "credentials_read": False, "private_references_read_locally": True,
              "developer_receives_gold": False, "synthetic": plan["synthetic"]}
    return report, panels, references


def preflight(plan):
    return _prepare(deepcopy(plan))[0]


class _BoundModel:
    """Stage capability and identity checks happen before every possible dispatch."""
    def __init__(self, model, expected, allowed, check):
        self._model, self._expected, self._allowed, self._check = model, expected, frozenset(allowed), check

    def _verify(self):
        self._check()
        current = {"name": self._model.model, "max_input_bytes": self._model.max_input_bytes,
                   "output_limits": self._model.limits, "prices": self._model.prices}
        if self._model.identity != self._expected or _model_identity(current, PROMPTS) != self._expected:
            raise HostError("live model identity drift; no dispatch allowed")

    @property
    def identity(self):
        self._verify()
        return self._expected

    @property
    def max_input_bytes(self):
        return self._model.max_input_bytes

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
        if stage not in self._allowed:
            raise HostError("model role cannot call this stage")
        self._verify()
        return self._model.complete(stage, payload)


def _check_request_recovery(directory):
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
    status = directory / "live_status.json"
    if status.exists():
        previous = json.loads(status.read_bytes())
        if previous.get("reason_type") == "UnknownProviderOutcome":
            raise UnknownProviderOutcome("previous run stopped with unknown provider outcome; reconcile before resume")
    return records


def _check_request_accounting(records, ledger):
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


def run(plan, *, approved_plan_hash, execute=False, transport_factory=None):
    plan = deepcopy(plan)
    check, panels, references = _prepare(plan)
    if execute is not True or approved_plan_hash != check["plan_hash"]:
        raise ValueError("execute and the exact approved plan hash are required")
    if plan["synthetic"] != (transport_factory is not None):
        raise ValueError("synthetic runs require an injected transport; live runs forbid injection")
    out = Path(plan["output_dir"]).resolve()
    with run_lock(out):
        freeze(out / "live_plan.json", plan)
        records = _check_request_recovery(out)
        ledger = Ledger(out / "ledger.jsonl", {"run": {"calls": plan["max_calls"], "cny": plan["hard_cny"]}})
        _check_request_accounting(records, ledger)
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

        transport = transport_factory() if transport_factory is not None else deepseek_transport(credential_from_plan(plan))
        model_config = plan["model"]

        def bound(bank, stages):
            check_binding()
            model = StructuredModel(out / "requests", ledger, transport, bank=bank,
                                    prices=model_config["prices"], model=model_config["name"],
                                    max_input_bytes=model_config["max_input_bytes"], limits=model_config["output_limits"])
            wrapped = _BoundModel(model, check["model_identity"], stages, check_binding)
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
        runner = EvolutionRunner(out, manifest, panels, references,
                                 lambda bank: bound(bank, ("plan", "read", "answer")),
                                 ProgramDeveloper(bound("develop", ("develop",))), backend_factory=backend_factory)
        try:
            result = runner.run()
            check_binding()
            if plan["corpus"] is not None:
                _file(plan["corpus"])
        except BaseException as exc:
            save(out / "live_status.json", {"status": "stopped", "reason_type": type(exc).__name__, "ledger": ledger.summary()})
            raise
        save(out / "live_status.json", {"status": "complete", "plan_hash": check["plan_hash"], "ledger": ledger.summary()})
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Frozen live entry for the existing RAG RSI evolution runner")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run"):
        command = sub.add_parser(name)
        command.add_argument("--plan", required=True)
        if name == "run":
            command.add_argument("--execute", action="store_true", required=True)
            command.add_argument("--approved-plan-hash", required=True)
    args = parser.parse_args(argv)
    plan = json.loads(Path(args.plan).read_bytes())
    result = preflight(plan) if args.command == "preflight" else run(
        plan, approved_plan_hash=args.approved_plan_hash, execute=args.execute)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
