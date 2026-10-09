"""Paired, D_fit-only modifier smoke using the existing evolution runner.

This adapter owns scheduling and a single paid-request ledger, not a second
search algorithm. Two feedback conditions share one root measurement per block.
All proposal opportunities, including rejections, follow a frozen schedule.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

from . import live_evolution as live
from .budget import Ledger, digest, save
from .v3.evolution import EvolutionRunner, ProgramDeveloper, freeze, read
from .v3.execution import HostError
from .v3.infrastructure import StructuredModel, BrowseCompCorpus, PROMPTS

SCHEMA = "rag-rsi-paired-development-1"
FIELDS = {"schema", "purpose", "output_dir", "search_template", "blocks", "schedule_policy",
          "max_calls", "hard_cny", "entry_sha256"}
CONDITIONS = ("cases", "trace")
SCHEDULE_POLICY = "alternate_block_and_slot_v1"


def _cost(template, qa_calls, developer_calls):
    model = template["model"]
    calls = qa_calls + developer_calls
    output = qa_calls * max(model["output_limits"][s] for s in ("plan", "read", "answer"))
    output += developer_calls * model["output_limits"]["develop"]
    return ((model["max_input_bytes"] + 1024) * calls * model["prices"]["input_miss"]
            + output * model["prices"]["output"]) / 1e6


def _prepare(plan):
    if (not isinstance(plan, dict) or set(plan) != FIELDS or plan["schema"] != SCHEMA
            or plan["purpose"] != "paired_development_smoke"
            or plan["schedule_policy"] != SCHEDULE_POLICY):
        raise ValueError("exact paired development plan required")
    if plan["entry_sha256"] != live._hash(__file__):
        raise ValueError("paired entry differs from frozen source")
    blocks = live._integer(plan["blocks"], 1, 8, "paired blocks")
    out = Path(plan["output_dir"]).resolve()
    if (live.PROJECT / "runs").resolve() not in out.parents:
        raise ValueError("paired output must be a project runs child")
    template = plan["search_template"]
    if (not isinstance(template, dict) or template.get("schema") != live.SCHEMA3
            or template.get("phase_order") != ["search"]
            or Path(template["output_dir"]).resolve() != out / "template_validation_only"):
        raise ValueError("paired smoke requires a search-only schema3 template")
    checked, panels, refs = live._prepare(template, phase="search")
    controls = checked["controls"]
    if (controls["parent_policy"] != "fixed_root" or controls["module_policy"] != "fixed"
            or controls["memory"] != "none" or controls["feedback"] != "cases"):
        raise ValueError("paired template requires cases, fixed root/module and no memory")
    expansions = live._integer(template["expansions"], 1, 16, "paired proposal opportunities")
    units = len(panels["D_fit"]) * template["repeats"] * template["limits"]["max_models"]
    root = {"calls": units, "cny": _cost(template, units, 0)}
    arm = {"calls": expansions * units + expansions,
           "cny": _cost(template, expansions * units, expansions)}
    block = {key: root[key] + 2 * arm[key] for key in root}
    total_calls = blocks * block["calls"]
    worst = blocks * block["cny"]
    live._integer(plan["max_calls"], 1, 1000000, "paired max_calls")
    hard = plan["hard_cny"]
    if plan["max_calls"] != total_calls:
        raise ValueError("paired call ceiling must count one root per block")
    if type(hard) not in (int, float) or not math.isfinite(hard) or hard <= 0 or hard < worst:
        raise ValueError("paired hard cap cannot cover complete conservative envelope")
    limits = {"run": {"calls": total_calls, "cny": hard}}
    schedule = []
    for b in range(blocks):
        limits[f"block:{b}"] = deepcopy(block)
        limits[f"root:{b}"] = deepcopy(root)
        for condition in CONDITIONS:
            limits[f"arm:{b}:{condition}"] = deepcopy(arm)
        for slot in range(expansions):
            order = CONDITIONS if (b + slot) % 2 == 0 else CONDITIONS[::-1]
            schedule.extend({"block": b, "condition": condition, "slot": slot} for condition in order)
    report = {"schema": SCHEMA, "status": "static_preflight_only", "plan_hash": digest(plan),
              "blocks": blocks, "conditions": list(CONDITIONS), "question_count": len(panels["D_fit"]),
              "proposal_opportunities": len(schedule), "schedule": schedule, "max_calls": total_calls,
              "feedback_case_count": len(controls["case_schedule"]),
              "feedback_covers_all_fit_questions": {x["question_id"] for x in controls["case_schedule"]} == {t["question_id"] for t in panels["D_fit"]},
              "conservative_cny_upper_bound": worst, "hard_cny": hard, "ledger_limits": limits,
              "reference_roles_parsed": [], "root_measurements_shared_within_block": True,
              "cache_reused_across_blocks": False, "full_feedback_body_gate": "after_root_before_either_proposal",
              "new_external_api_calls": 0, "credentials_read": False,
              "real_llm_improvement_verified": False}
    return report, checked, panels


def preflight(plan):
    return _prepare(deepcopy(plan))[0]


class _ScopedLedger:
    def __init__(self, ledger, block, condition):
        self.ledger, self.block, self.condition = ledger, block, condition

    @property
    def scopes(self):
        specific = f"root:{self.block}" if self.condition == "root" else f"arm:{self.block}:{self.condition}"
        return ["run", f"block:{self.block}", specific]

    def reserve(self, scopes, amount, metadata=None):
        if scopes != ["run"]:
            raise ValueError("paired requests require common run scope")
        return self.ledger.reserve(self.scopes, amount,
            {**(metadata or {}), "paired_block": self.block, "paired_condition": self.condition})

    def settle(self, *args, **kwargs):
        return self.ledger.settle(*args, **kwargs)


def _accounting(records, ledger, blocks):
    live._check_request_accounting(records, ledger)
    for event in ledger.events:
        if event["event"] != "reserve":
            continue
        metadata = event.get("metadata", {})
        block, condition = metadata.get("paired_block"), metadata.get("paired_condition")
        bank = metadata.get("bank", "")
        if (type(block) is not int or not 0 <= block < blocks or condition not in ("root", *CONDITIONS)
                or event["scopes"] != _ScopedLedger(ledger, block, condition).scopes
                or not isinstance(bank, str) or not bank.startswith(f"paired/{block}/{condition}/")):
            raise HostError("paired bank and ledger scopes differ")
        if condition == "root" and metadata.get("stage") == "develop":
            raise HostError("shared root cannot charge a proposal")


def _common_payload(payload):
    result = deepcopy(payload)
    result["feedback"]["condition"] = "cases"
    for case in result["feedback"]["cases"]:
        case.pop("execution_flow", None)
        case.pop("diagnostics", None)
    return result


class _PairedModel(live._BoundModel):
    def __init__(self, *args, proposal_gate=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.proposal_gate = proposal_gate

    def complete(self, stage, payload):
        if stage == "develop":
            expected = self.proposal_gate() if self.proposal_gate else None
            if expected is None or digest(self.request_body(stage, payload)) != expected:
                raise HostError("proposal body differs from the paired full-request gate")
        return super().complete(stage, payload)


def run(plan, *, approved_plan_hash, execute=False, transport_factory=None, stop_after=None):
    plan = deepcopy(plan)
    checked, template_check, panels = _prepare(plan)
    if execute is not True or approved_plan_hash != checked["plan_hash"]:
        raise ValueError("execute and exact paired plan hash required")
    template = plan["search_template"]
    if template["synthetic"] != (transport_factory is not None):
        raise ValueError("synthetic paired run requires injection; live run forbids it")
    total = len(checked["schedule"])
    stop_after = total if stop_after is None else live._integer(stop_after, 0, total, "stop_after")
    out = Path(plan["output_dir"]).resolve()
    with live.run_lock(out):
        freeze(out / "paired_plan.json", plan)
        records = live._check_request_recovery(out)
        ledger = Ledger(out / "ledger.jsonl", checked["ledger_limits"])
        _accounting(records, ledger, plan["blocks"])
        stored_indices = sorted(int(p.stem) for p in (out / "schedule").glob("*.json") if p.stem.isdigit())
        if stored_indices != list(range(len(stored_indices))) or len(stored_indices) > total:
            raise HostError("paired schedule has missing or extra terminal slots")
        # A request for a shorter prefix must still verify previously completed work.
        verify_until = max(stop_after, len(stored_indices))
        transport = (live._LazyTransport(transport_factory) if transport_factory is not None else
                     live._LazyLiveTransport(lambda: live.deepseek_transport(live.credential_from_plan(template))))
        model_config = template["model"]
        gates = {}
        corpus_stamp = None
        if template["corpus"] is not None:
            stat = Path(template["corpus"]["path"]).stat()
            corpus_stamp = (stat.st_size, stat.st_mtime_ns)

        def check_binding():
            if (live._hash(__file__) != plan["entry_sha256"]
                    or live._hash(live.__file__) != template["entry_sha256"]
                    or live._runtime_source_hashes() != template["runtime_source_hashes"]):
                raise HostError("paired frozen runtime changed")
            if corpus_stamp is not None:
                stat = Path(template["corpus"]["path"]).stat()
                if (stat.st_size, stat.st_mtime_ns) != corpus_stamp:
                    raise HostError("paired corpus changed")

        def bound(block, condition, bank, stages):
            check_binding()
            model = StructuredModel(out / "requests", _ScopedLedger(ledger, block, condition), transport,
                bank=f"paired/{block}/{condition}/{bank}", prices=model_config["prices"],
                model=model_config["name"], max_input_bytes=model_config["max_input_bytes"],
                limits=model_config["output_limits"])
            return _PairedModel(model, template_check["model_identity"], stages, check_binding,
                                proposal_gate=lambda: gates.get((block, condition)))

        corpora = {}
        def backend_factory(task):
            check_binding()
            excluded = tuple(sorted(task["excluded_docids"]))
            if excluded not in corpora:
                corpora[excluded] = BrowseCompCorpus(template["corpus"]["path"],
                    corpus_hash=template["corpus"]["sha256"], excluded=excluded)
            return corpora[excluded]

        def make_runner(block, condition):
            controls = deepcopy(template["controls"])
            controls["feedback"] = condition
            manifest = {"schema": "rag-rsi-v3-paired-search-1", "paired_plan_hash": checked["plan_hash"],
                        "paired_block": block, "condition": condition,
                        "model_identity": template_check["model_identity"], "synthetic": template["synthetic"],
                        **{key: deepcopy(template[key]) for key in ("metric", "expansions", "repeats", "root_config",
                            "select_candidates", "limits", "allow_proxy_metric")},
                        "controls": controls, "proposal_bank_policy": live.PROPOSAL_BANK_POLICY,
                        "lifecycle": deepcopy(template_check["lifecycle"]),
                        "reference_groups_file": deepcopy(template["reference_groups_file"]),
                        "shared_root": {"schema": "rag-rsi-shared-root-1", "directory": str(out / "blocks" / str(block) / "root_measurements"),
                                        "block_id": str(block), "bank": f"shared-root/{block}"}}
            developer = ProgramDeveloper(bound(block, condition, "develop", ("develop",)),
                feedback_condition=condition, case_schedule=controls["case_schedule"],
                proposal_model_factory=lambda slot: bound(block, condition, f"develop/proposal/{slot}", ("develop",)))
            return EvolutionRunner(out / "blocks" / str(block) / condition, manifest, panels, {},
                lambda bank: bound(block, condition, bank, ("plan", "read", "answer")), developer,
                backend_factory=backend_factory,
                reference_loader=live._reference_loader(template, panels, template_check["lifecycle"]["reference_groups"]),
                root_model_factory=lambda bank: bound(block, "root", bank, ("plan", "read", "answer")))

        def prepare_block(block):
            runners, roots, payloads, records = {}, {}, {}, {}
            for condition in CONDITIONS:
                runner = make_runner(block, condition)
                progress = runner.run_search_until(0)
                root = runner.archive.load_node(progress["search"]["order"][0])
                program = runner.archive.load_program(root["program_id"])
                receipt = read(runner.directory / "measurements" / "shared_root.json")
                measurement = receipt["result"]
                cards = progress["search"]["cards"][:1]
                decision = runner._decision(cards, root, measurement, 0, [])
                payload = runner.developer.prepare_request(program, decision, [], measurement, panels["D_fit"])
                body = runner.developer.model.request_body("develop", payload)
                size = runner.developer.model.request_size("develop", payload)
                if size > model_config["max_input_bytes"]:
                    raise ValueError("paired complete request is oversized; neither arm may crop cases")
                runners[condition], roots[condition], payloads[condition] = runner, measurement, payload
                records[condition] = {"request_bytes": size, "request_body_hash": digest(body)}
                freeze(out / "blocks" / str(block) / "feedback" / (condition + ".json"), body)
            if roots["cases"] != roots["trace"] or _common_payload(payloads["trace"]) != payloads["cases"]:
                raise HostError("paired arms do not share the same root measurement and common feedback")
            record = {"schema": "rag-rsi-paired-body-gate-1", "block": block,
                      "root_measurement_hash": digest(roots["cases"]), "common_payload_equal": True,
                      "scheduled_cases_not_cropped": True, "conditions": records}
            freeze(out / "blocks" / str(block) / "feedback_gate.json", record)
            for condition in CONDITIONS:
                gates[(block, condition)] = records[condition]["request_body_hash"]
            return runners

        runners_by_block = {}
        try:
            for index, item in enumerate(checked["schedule"][:verify_until]):
                block, condition, slot = item["block"], item["condition"], item["slot"]
                if block not in runners_by_block:
                    runners_by_block[block] = prepare_block(block)
                result = runners_by_block[block][condition].run_search_until(slot + 1)
                attempt = next((a for a in result["search"]["terminal_attempts"] if a["step"] == slot), None)
                if attempt is None:
                    raise HostError("scheduled proposal did not produce a terminal opportunity receipt")
                freeze(out / "schedule" / f"{index}.json", {"schedule_index": index, **item, "attempt": attempt})
            _accounting(live._check_request_recovery(out), ledger, plan["blocks"])
            check_binding()
            if template["corpus"] is not None:
                live._file(template["corpus"])
            completed = []
            for index, item in enumerate(checked["schedule"]):
                record = read(out / "schedule" / f"{index}.json")
                if record is None:
                    break
                if {k: record.get(k) for k in ("block", "condition", "slot")} != item:
                    raise HostError("paired terminal schedule differs from plan")
                completed.append(record)
            result = {"schema": SCHEMA, "status": "complete" if len(completed) == total else "in_progress",
                      "plan_hash": checked["plan_hash"], "completed_opportunities": len(completed),
                      "planned_opportunities": total, "terminal_records": completed,
                      "planned_root_measurements_per_block": 1, "prepared_blocks": sorted(runners_by_block),
                      "root_cost_paid_once": True,
                      "heldout_roles_used": False, "quality_claim": "engineering_fixture_only" if template["synthetic"]
                          else "exposed_development_smoke_not_independent_quality_evidence",
                      "ledger": ledger.summary()}
            if len(completed) == total:
                freeze(out / "paired_search_complete.json", result)
            save(out / "live_status.json", {"status": result["status"], "plan_hash": checked["plan_hash"], "ledger": ledger.summary()})
            return result
        except BaseException as exc:
            save(out / "live_status.json", {"status": "stopped", "reason_type": type(exc).__name__, "ledger": ledger.summary()})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Paired development-only modifier trial")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--plan", required=True)
        if name == "run":
            cmd.add_argument("--execute", action="store_true", required=True)
            cmd.add_argument("--approved-plan-hash", required=True)
            cmd.add_argument("--stop-after", type=int)
    args = parser.parse_args(argv)
    plan = json.loads(Path(args.plan).read_bytes())
    result = preflight(plan) if args.command == "preflight" else run(plan,
        approved_plan_hash=args.approved_plan_hash, execute=args.execute, stop_after=args.stop_after)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
