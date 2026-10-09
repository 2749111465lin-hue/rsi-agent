"""One data-driven loop: edit -> execute -> measure -> experience -> select -> report.

The same runtime serves every task adapter. Old experiments remain immutable;
this module does not import any v25* or P4/P5 continuation script.
"""
from __future__ import annotations
import ast
from copy import deepcopy
import json
from pathlib import Path
from ..archive import ProgramArchive, Conflict
from ..budget import digest, save, stable
from .datasets import validate_task_collection
from .execution import Measurement, root_files, validate_sources
from .experience_policy import DEFAULT_MODULES, choose_next, memory_for_action
from .diagnostics import compact_feedback
from .edit_scope import observe_edit_scope, validated_scope
from .fit_literal_audit import audit_fit_literals


def freeze(path, value):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))!=value:
            raise ValueError("frozen run artifact differs: "+path.name)
    else:
        save(path,value)


def read(path):
    path=Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _runtime_source_hashes():
    base=Path(__file__).parent.parent
    paths=list(Path(__file__).parent.glob("*.py"))
    paths += [base/name for name in
              ("archive.py","budget.py","sandbox.py","sdk.py","candidate_runner.py","linux_launcher.sh")]
    return {str(p.relative_to(base)):digest(p.read_text(encoding="utf-8")) for p in sorted(paths)}


def _validate_receipt(archive, receipt, files, metadata, *, session_id, attempt, parent_node_id=None):
    if not isinstance(receipt,dict) or not isinstance(receipt.get("node_id"),str):
        raise ValueError("invalid archive receipt")
    node=archive.load_node(receipt["node_id"])
    if receipt!=node:
        raise ValueError("archive receipt differs from archived node")
    if (node["session_id"],node["attempt"],node["parent_node_id"])!=(session_id,attempt,parent_node_id):
        raise ValueError("archive receipt identity differs from expected attempt")
    program=archive.load_program(node["program_id"])
    # The archive fills missing metadata sections with empty objects.
    if not isinstance(metadata,dict) or not set(metadata)<=set(program["metadata"]):
        raise ValueError("invalid expected archive metadata")
    canonical_metadata={key:metadata.get(key,{}) for key in program["metadata"]}
    if program["files"]!=files or program["metadata"]!=canonical_metadata:
        raise ValueError("archive receipt source differs from expected proposal")
    return node


def recoverable_record(archive, files, metadata, *, session_id, attempt, parent_node_id=None):
    expected={"session_id":session_id,"attempt":attempt,"parent_node_id":parent_node_id}
    def existing_attempt():
        matches=[]
        for path in (archive.root/"nodes").glob("*.json"):
            raw=read(path)
            if not isinstance(raw,dict):
                raise ValueError("invalid archived attempt")
            if raw.get("session_id")==session_id and raw.get("attempt")==attempt:
                matches.append(archive.load_node(path.stem))
        if len(matches)>1:
            raise ValueError("ambiguous archive recovery")
        return _validate_receipt(archive,matches[0],files,metadata,**expected) if matches else None
    node=existing_attempt()
    if node is not None:
        return node
    try:
        return archive.record(files,metadata,**expected)
    except Conflict:
        # Reconcile a completed write if its caller's receipt was not persisted.
        node=existing_attempt()
        if node is None:
            raise
        return node


def program_change(parent, writes):
    if not isinstance(writes,dict) or not writes or not set(writes)<={"rag.py","rag_core.py"}:
        raise ValueError("unsupported candidate patch")
    files={**parent["files"],**writes}
    validate_sources(files)
    if all(ast.dump(ast.parse(files[name]),include_attributes=False)==
           ast.dump(ast.parse(parent["files"][name]),include_attributes=False) for name in files):
        raise ValueError("patch changes no executable behavior")
    return files


def _proposal_files(program, decision, proposal):
    if (not isinstance(proposal,dict) or set(proposal) not in
            ({"writes","mechanism","target_module"}, {"writes","mechanism","intended_target_module"})):
        raise ValueError("invalid development proposal")
    if not isinstance(proposal["mechanism"],str) or not proposal["mechanism"].strip():
        raise ValueError("invalid development proposal")
    intended=proposal.get("intended_target_module",proposal.get("target_module"))
    if intended!=decision.get("intended_target_module",decision["target_module"]):
        raise ValueError("proposal intent differs from declared action")
    return program_change(program,proposal["writes"])


def _fit_development_request(model, payload):
    """Reduce optional excerpts against the actual double-encoded request size.

    Retain selected cases, questions, signed scores, diagnostics codes and flow
    totals. Never truncate executable source or silently drop a learning case.
    The provider's existing byte limit remains the final dispatch boundary.
    """
    size = getattr(model, "request_size", None)
    limit = getattr(model, "max_input_bytes", None)
    if not callable(size) or type(limit) is not int:
        return payload  # Scripted/non-provider models have no paid request body.
    original = size("develop", payload)
    if original <= limit:
        return payload
    result = deepcopy(payload)
    feedback = result["feedback"]
    budget = {"limit_bytes": limit, "original_request_bytes": original,
              "reductions": [], "cases_and_scores_preserved": True, "source_preserved": True}
    feedback["request_budget"] = budget

    def shorten(obj, field, maximum, flag):
        text = obj.get(field)
        if not isinstance(text, str) or len(text) <= maximum:
            return 0
        removed = len(text) - maximum
        obj[field] = text[:maximum]; obj[flag] = True
        return removed

    for tier in (1, 2):
        budget["reductions"].append("shorter_excerpts" if tier == 1 else "omit_optional_details")
        for case in feedback["cases"]:
            shorten(case, "prediction", 120 if tier == 1 else 0, "prediction_truncated_for_request_budget")
            details = case.get("diagnostics", {}).get("model_details", {})
            for field, values in list(details.items()):
                if isinstance(values, list):
                    smaller = [v[:80] if isinstance(v, str) else v for v in values] if tier == 1 else []
                    if smaller != values:
                        details[field] = smaller
                        case["diagnostics"]["model_details_truncated_for_request_budget"] = True
            for witness in case.get("evidence_witnesses", []):
                shorten(witness, "quote_excerpt", 80 if tier == 1 else 0, "excerpt_truncated")
            for name in ("execution_flow", "parent_execution_flow"):
                flow = case.get(name)
                if not isinstance(flow, dict) or flow.get("status") != "observed":
                    continue
                omitted = flow["omitted"]
                samples = [q for r in flow["reads"] for q in r["queries"]] + flow["trailing_queries"]
                for query in samples:
                    omitted["query_chars"] += shorten(query, "query_excerpt", 80 if tier == 1 else 0, "query_truncated")
                for read in flow["reads"]:
                    for quote in read["quotes"]:
                        omitted["quote_chars"] += shorten(quote, "quote_excerpt", 80 if tier == 1 else 0, "quote_truncated")
                    if tier == 2:
                        omitted["queries"] += len(read["queries"])
                        omitted["source_samples"] += len(read["sources"])
                        omitted["quote_samples"] += len(read["quotes"])
                        read["queries"] = []; read["sources"] = []; read["quotes"] = []
                omitted["answer_chars"] += shorten(flow["final"], "answer_excerpt", 120 if tier == 1 else 0, "answer_truncated")
                if tier == 2:
                    omitted["queries"] += len(flow["trailing_queries"])
                    flow["trailing_queries"] = []
                flow["truncated"] = True
                flow["details_reduced_for_request_budget"] = True
        if size("develop", result) <= limit:
            return result
    raise ValueError("full source and required development feedback exceed the complete request budget")


def _controlled_developer_decision(decision):
    from .diagnostics import MODULES
    if (not isinstance(decision, dict) or decision.get("operator") not in ("Draft", "Improve", "Debug")
            or decision.get("target_module") not in MODULES):
        raise ValueError("controlled development requires a declared operator and target module")
    target = decision["target_module"]
    intended = decision.get("intended_target_module", target)
    if intended != target:
        raise ValueError("controlled development target and intended target differ")
    return {"operator": decision["operator"], "target_module": target,
            "intended_target_module": intended}


def _strict_developer_size(model, payload, *, required=False):
    size = getattr(model, "request_size", None)
    limit = getattr(model, "max_input_bytes", None)
    identity = getattr(model, "identity", None)
    if size is None and limit is None and identity is None and not required:
        return None  # Pure scripted preparation; never a paid-dispatch exemption.
    if (not callable(size) or type(limit) is not int or limit <= 0
            or (required and (not isinstance(identity, str) or not identity))):
        raise ValueError("controlled development needs an exact model request-size contract")
    measured = size("develop", payload)
    if type(measured) is not int or measured < 0:
        raise ValueError("invalid complete development request size")
    if measured > limit:
        raise ValueError("complete controlled development request exceeds budget; no cases are cropped")
    return measured


def _match_developer_models(projected, actual, payload):
    projected_size = _strict_developer_size(projected, payload, required=True)
    actual_size = _strict_developer_size(actual, payload, required=True)
    if (projected.identity != actual.identity
            or projected.max_input_bytes != actual.max_input_bytes
            or projected_size != actual_size):
        raise ValueError("proposal model differs from the frozen projection contract")
    left, right = getattr(projected, "request_body", None), getattr(actual, "request_body", None)
    if callable(left) and callable(right) and digest(left("develop", payload)) != digest(right("develop", payload)):
        raise ValueError("proposal provider body differs from its projection")
    if not callable(getattr(actual, "complete", None)):
        raise ValueError("proposal model must implement complete")


class ProgramDeveloper:
    def __init__(self, model, *, feedback_condition="rich", case_schedule=None,
                 proposal_model_factory=None):
        from .feedback_conditions import CONDITIONS
        if feedback_condition not in ("rich", *CONDITIONS):
            raise ValueError("unknown developer feedback condition")
        if proposal_model_factory is not None and not callable(proposal_model_factory):
            raise ValueError("proposal_model_factory must be callable")
        if feedback_condition == "rich":
            if case_schedule is not None and case_schedule != []:
                raise ValueError("rich feedback uses its existing case selection")
        elif (not isinstance(case_schedule, list) or not 1 <= len(case_schedule) <= 16
              or any(not isinstance(item, dict) or set(item) != {"question_id", "repeat"}
                     or not isinstance(item["question_id"], str) or not item["question_id"]
                     or type(item["repeat"]) is not int or item["repeat"] < 0 for item in case_schedule)
              or len({item["question_id"] for item in case_schedule}) != len(case_schedule)):
            raise ValueError("controlled developer requires a fixed distinct question/repeat schedule")
        self.model = model
        self.feedback_condition = feedback_condition
        self.case_schedule = [] if feedback_condition == "rich" else deepcopy(case_schedule)
        self.proposal_model_factory = proposal_model_factory
        self._proposal_models = {}
        self._proposal_payloads = {}

    def configuration_snapshot(self):
        return {"feedback_condition": self.feedback_condition,
                "case_schedule": deepcopy(self.case_schedule),
                "proposal_model_factory_present": self.proposal_model_factory is not None}

    def prepare_request(self, program, decision, experience, result, tasks):
        """Exact outbound request; private references are deliberately not an argument."""
        if result["role"]!="D_fit":
            raise ValueError("developer receives D_fit only")
        if self.feedback_condition == "rich":
            feedback = compact_feedback(result,tasks,max_cases=4)
            outbound_decision, outbound_experience = decision, experience
            if "proposal_slot" in decision:
                outbound_decision = {key: value for key, value in decision.items() if key != "proposal_slot"}
        else:
            from .feedback_conditions import controlled_feedback
            feedback = controlled_feedback(result, tasks, condition=self.feedback_condition,
                                           case_schedule=self.case_schedule)
            outbound_decision, outbound_experience = _controlled_developer_decision(decision), []
        payload={"source_files":program["files"],"decision":outbound_decision,
             "experience":outbound_experience,"feedback":feedback,
             "edit_boundary":("Change reusable behavior; do not embed examples/answers. Return complete changed files. "
                "intended_target_module (legacy target_module) declares intent only. General refactors are allowed. "
                "The host separately records actual AST scopes and intent mismatches; mixed or unknown edits "
                "retain whole-program scores but cannot supply a single-module gain. Scope associations are not causal. "
                "The host rejects newly embedded development-question literals and sufficiently specific strings "
                "from feedback already shown; keep fixes reusable rather than task-specific lookup code.")}
        if self.feedback_condition == "rich":
            return _fit_development_request(self.model,payload)
        _strict_developer_size(self.model, payload)
        return payload

    def propose(self, program, decision, experience, result, tasks, references):
        payload=self.prepare_request(program,decision,experience,result,tasks)
        model = self.model
        if self.proposal_model_factory is not None:
            slot = decision.get("proposal_slot")
            if type(slot) is not int or slot < 0:
                raise ValueError("proposal factory requires a nonnegative integer proposal_slot")
            # The slot identifies a host cache bank, never experimental feedback.
            payload = deepcopy(payload)
            payload["decision"].pop("proposal_slot", None)
            payload_hash = digest(payload)
            if slot in self._proposal_payloads and self._proposal_payloads[slot] != payload_hash:
                raise ValueError("a proposal slot cannot resume with a different request")
            if slot not in self._proposal_models:
                self._proposal_models[slot] = self.proposal_model_factory(slot)
            model = self._proposal_models[slot]
            _match_developer_models(self.model, model, payload)
            self._proposal_payloads[slot] = payload_hash
        elif self.feedback_condition != "rich":
            raise ValueError("controlled proposals require distinct slot model factories")
        output=model.complete("develop",payload)
        _proposal_files(program,decision,output)
        return output


def experience_card(result, parent, *, operator, module, step, mechanism, edit_scope=None):
    scope=validated_scope(edit_scope) if edit_scope is not None else None
    if edit_scope is not None and (scope is None or scope["intended_target_module"]!=module):
        raise ValueError("invalid host edit scope receipt or intent")
    per=result["per_question"]
    feedback=compact_feedback({**result,**({"parent_measurement":parent} if parent else {})},
              [{"question_id":qid,"question":""} for qid in per],max_cases=0)
    if feedback["measurement_status"]!="complete":
        raise ValueError("unavailable outcome cannot enter experience")
    raw_paired={qid:score-parent["per_question"][qid] for qid,score in per.items()} if parent else {}
    # Feedback revalidates current host answer-origin receipts. Historical flags
    # alone cannot promote legacy rows into current learning evidence.
    eligible=feedback.get("program_eligible") is True
    pair_eligible=feedback.get("paired_comparison_eligible") is True
    raw_delta=result["score"]-parent["score"] if parent else None
    paired=raw_paired if pair_eligible else None if parent else {}
    failures=sorted(set(f for row in result["rows"] for f in row.get("failure_classes",[]))
                    | set(feedback["summary"]["host_observed"]))
    diagnostic={"host_observed":sorted(feedback["summary"]["host_observed"]),
                "model_reported":sorted(feedback["summary"]["model_reported"]),
                "module_priors":feedback["module_priors"],"priors_are_design_heuristics":True,
                "semantic_support":"not_host_verified","paired_summary":feedback["paired_summary"]}
    if any(x<1 for x in per.values()): failures.append("answer_quality")
    return {"node_id":result["node_id"],"program_id":result["program_id"],"role":"D_fit",
      "source":"host_measured_D_fit", "intended_target_module":module,
      "actual_edit_scope":scope,"associated_module":scope["associated_module"] if scope else None,
      "module_attribution":scope["attribution"] if scope else "unknown",
      "module_association_is_causal":False,
      "panel_hash":result["panel_hash"],"evaluator_epoch":result["evaluator_epoch"],
      "complete":True,"valid_program":eligible,"program_eligible":eligible,
      "paired_comparison_eligible":pair_eligible,"score":result["score"] if eligible else None,
      "signed_delta_vs_best_parent":raw_delta if pair_eligible else None,
      "signed_deltas":({parent["node_id"]:raw_delta} if pair_eligible else None) if parent else {},
      "raw_diagnostics":{"answer_score":result["score"],"signed_delta_vs_best_parent":raw_delta,
          "signed_deltas":{parent["node_id"]:raw_delta} if parent else {},"paired_deltas":raw_paired,
          "paired_signed_gain":sum(raw_paired.values())/len(raw_paired) if raw_paired else None,
          "reported_valid_program":result.get("valid_program"),
          "host_program_eligibility":feedback.get("program_eligible"),
          "diagnostic_only":True,"not_quality_reward":True},
      "parent_node_ids":[parent["node_id"]] if parent else [],"operator":operator,"target_module":module,
      "step":step,"hypothesis":mechanism,"failure_classes":failures,"diagnostics":diagnostic,
      "failure_assessment_source":"host_execution_and_fit_answer_scores",
      "paired_deltas":paired,"resource_usage":result["resource_usage"],
      "behavior":{"group_hash":digest([r["answer"] for r in result["rows"]])},
      "reward":{"answer_quality":result["score"] if eligible else None,
           "paired_signed_gain":sum(paired.values())/len(paired) if paired else None,
           "delivery_rate":sum(r["answer_usable"] for r in result["rows"])/len(result["rows"]),
           "source_valid_rate":sum(r["citation_source_valid"] for r in result["rows"])/len(result["rows"]),
           "proxy_added_to_terminal_quality":False}}


CONTROL_FIELDS = {"parent_policy", "module_policy", "fixed_module", "memory", "feedback", "case_schedule"}
LEGACY_CONTROLS = {"parent_policy": "adaptive", "module_policy": "legacy", "fixed_module": None,
                   "memory": "legacy", "feedback": "rich", "case_schedule": []}


def validate_controls(controls, tasks, repeats, *, allow_legacy=False):
    """Freeze distinct experimental factors; controlled feedback uses one fixed parent."""
    if controls is None and allow_legacy:
        return deepcopy(LEGACY_CONTROLS)
    if not isinstance(controls, dict) or set(controls) != CONTROL_FIELDS:
        raise ValueError("exact evolution controls required")
    c = deepcopy(controls)
    if (c["parent_policy"] not in ("adaptive", "fixed_root")
            or c["module_policy"] not in ("legacy", "round_robin_v1", "experience_coverage_v1", "fixed")
            or c["memory"] not in ("legacy", "none", "mechanism")
            or c["feedback"] not in ("rich", "aggregate", "cases", "trace")):
        raise ValueError("unsupported evolution control")
    if not allow_legacy and (c["module_policy"] == "legacy" or c["memory"] == "legacy"):
        raise ValueError("new controls cannot silently select legacy behavior")
    if c["module_policy"] == "fixed":
        if c["fixed_module"] not in DEFAULT_MODULES:
            raise ValueError("fixed module must name an available RAG module")
    elif c["fixed_module"] is not None:
        raise ValueError("fixed_module requires fixed module policy")
    if c["parent_policy"] == "fixed_root" and c["module_policy"] != "fixed":
        raise ValueError("fixed_root currently requires fixed module policy")
    schedule = c["case_schedule"]
    if not isinstance(schedule, list):
        raise ValueError("case schedule must be a frozen list")
    if c["feedback"] == "rich":
        if schedule:
            raise ValueError("rich feedback uses its existing case selection, not case_schedule")
    else:
        if (c["parent_policy"], c["module_policy"], c["memory"]) != ("fixed_root", "fixed", "none"):
            raise ValueError("controlled feedback requires fixed root, fixed module and no memory")
        if type(repeats) is not int or repeats < 1 or not 1 <= len(schedule) <= 16:
            raise ValueError("bounded case schedule and repeat count required")
        qids = {t["question_id"] for t in tasks}
        seen = set()
        for item in schedule:
            if (not isinstance(item, dict) or set(item) != {"question_id", "repeat"}
                    or not isinstance(item["question_id"], str) or item["question_id"] not in qids
                    or item["question_id"] in seen or type(item["repeat"]) is not int
                    or not 0 <= item["repeat"] < repeats):
                raise ValueError("case schedule must contain distinct D_fit questions and available repeats")
            seen.add(item["question_id"])
    return c


def _literal_rejection(audit):
    return {"reason":"new development-specific code literals", "reason_code":"fit_literal_match",
            "audit_sha256":digest(audit),"findings":deepcopy(audit["findings"][:8]),
            "finding_count":len(audit["findings"]),"next_step_allowed":True}


class EvolutionRunner:
    def __init__(self,directory,manifest,panels,references,model_factory,developer,backend_factory=None,scorer=None):
        self.directory=Path(directory)
        manifest,panels,references=deepcopy((manifest,panels,references))
        self.manifest=manifest; self.panels=panels; self.references=references
        self.developer=developer
        if set(panels)!={"D_fit","D_select","D_report"} or any(not x for x in panels.values()):
            raise ValueError("three nonempty independent panels required")
        for role,tasks in panels.items():
            validate_task_collection(tasks)
            if set(references[role])!={t["question_id"] for t in tasks}:
                raise ValueError("exact role-scoped reference set required")
        seen={}
        for role,tasks in panels.items():
            for task in tasks:
                for key in (task["question_id"],digest(task["question"].strip().casefold())):
                    if key in seen and seen[key]!=role:
                        raise ValueError("question overlap across roles")
                    seen[key]=role
        groups={}
        for role,refs in references.items():
            for reference in refs.values():
                group=reference.get("pair_group_id",reference.get("source_question_id"))
                if group is not None:
                    key=(reference["dataset"],str(group))
                    if key in groups and groups[key]!=role:
                        raise ValueError("source question group crosses roles")
                    groups[key]=role
        if type(manifest.get("expansions")) is not int or not 0<=manifest["expansions"]<=16:
            raise ValueError("bounded expansion count required")
        if type(manifest.get("select_candidates",3)) is not int or not 1<=manifest.get("select_candidates",3)<=17:
            raise ValueError("select_candidates must be an integer from 1 to 17")
        if manifest.get("metric") not in {"em","f1"} and scorer is None:
            raise ValueError("explicit supported metric or external scorer required")
        if any(t["dataset"]=="browsecomp-plus" for ts in panels.values() for t in ts) and scorer is None and not manifest.get("allow_proxy_metric"):
            raise ValueError("BCP official judge required, or explicitly label proxy metric")
        self.controlled_run = "controls" in manifest
        self.controls = validate_controls(manifest.get("controls"), panels["D_fit"],
            manifest.get("repeats", 1), allow_legacy=not self.controlled_run)
        if self.controlled_run and isinstance(developer, ProgramDeveloper):
            configuration = developer.configuration_snapshot()
            if (configuration["feedback_condition"] != self.controls["feedback"]
                    or configuration["case_schedule"] != self.controls["case_schedule"]
                    or not configuration["proposal_model_factory_present"]):
                raise ValueError("developer configuration differs from frozen controls or independent proposal banks")
        elif self.controlled_run and self.controls["feedback"] != "rich":
            raise ValueError("controlled feedback requires the host ProgramDeveloper")
        self._frozen_manifest=deepcopy(self._snapshot())
        freeze(self.directory/"manifest.json",self._frozen_manifest)
        self.archive=ProgramArchive(self.directory/"archive")
        self.measure=Measurement(self.archive,self.directory/"measurements",model_factory,backend_factory,
             metric=manifest["metric"],scorer=scorer,limits=manifest.get("limits"),
             allow_proxy_metrics=manifest.get("allow_proxy_metric",False))

    def _snapshot(self):
        return {**deepcopy(self.manifest),
                "public_panel_hashes":{r:digest(ts) for r,ts in self.panels.items()},
                "private_reference_hashes":{r:digest(ref) for r,ref in self.references.items()},
                "runtime_source_hashes":_runtime_source_hashes(),
                **({"active_controls": deepcopy(self.controls),
                    "developer_configuration": self.developer.configuration_snapshot()
                        if isinstance(self.developer, ProgramDeveloper) else None}
                   if self.controlled_run else {})}

    def _assert_frozen(self):
        current=self._snapshot()
        if current!=self._frozen_manifest:
            raise ValueError("frozen run inputs or runtime source changed")
        freeze(self.directory/"manifest.json",current)

    def _measure(self,node,role):
        self._assert_frozen()
        result=self.measure.run(node,deepcopy(self.panels[role]),deepcopy(self.references[role]),
                    role=role,bank="common",repeats=self.manifest.get("repeats",1))
        self._assert_frozen()
        return result

    def _memory(self, cards, decision):
        if self.controls["memory"] == "none":
            return []
        return memory_for_action(cards, decision, include_mechanism=self.controls["memory"] == "mechanism")

    def _decision(self, cards, root, baseline, step, attempts):
        c = self.controls
        if c["parent_policy"] == "fixed_root":
            if not cards[0]["valid_program"]:
                raise ValueError("fixed-parent feedback experiment requires an eligible root")
            d = {"parent_node_id": root["node_id"], "operator": "Improve",
                 "target_module": c["fixed_module"], "intended_target_module": c["fixed_module"],
                 "reason": "pre_registered_fixed_parent_and_module", "experience_ids": [],
                 "panel_hash": baseline["panel_hash"], "evaluator_epoch": self.measure.epoch,
                 "role": "D_fit", "diagnostics": {"module_policy": "fixed", "causal_claim": False}}
        else:
            d = choose_next(cards, step=step, panel_hash=baseline["panel_hash"],
                evaluator_epoch=self.measure.epoch,
                module_policy="legacy" if c["module_policy"] == "fixed" else c["module_policy"],
                attempt_history=attempts if self.controlled_run else None)
            if d["parent_node_id"] is None:
                d["parent_node_id"] = root["node_id"]
            if c["module_policy"] == "fixed":
                d.update(target_module=c["fixed_module"], intended_target_module=c["fixed_module"])
                d["reason"] = d["reason"].split(";")[0] + "; pre_registered_fixed_module"
                d["diagnostics"]["module_policy"] = "fixed"
        if self.controlled_run:
            d["proposal_slot"] = step
            d["experience_ids"] = [row["node_id"] for row in self._memory(cards, d)]
        return d

    def _terminal_attempt(self, folder, decision, step, status, attempts, node_id=None):
        if not self.controlled_run:
            return
        record = {"attempt_id": digest({"panel": decision["panel_hash"],
                    "epoch": decision["evaluator_epoch"], "step": step}),
                  "step": step, "role": "D_fit", "panel_hash": decision["panel_hash"],
                  "evaluator_epoch": decision["evaluator_epoch"],
                  "parent_node_id": decision["parent_node_id"],
                  "intended_target_module": decision["target_module"],
                  "status": status, "node_id": node_id}
        freeze(folder/"attempt.json", record)
        attempts.append(record)

    def _audit_literals(self, folder, program, proposed, decision, cards, development_result):
        public=[{"question_id":t["question_id"],"question":t["question"]} for t in self.panels["D_fit"]]
        feedback=None
        if isinstance(self.developer,ProgramDeveloper):
            payload=self.developer.prepare_request(deepcopy(program),deepcopy(decision),
                self._memory(cards,decision),deepcopy(development_result),deepcopy(self.panels["D_fit"]))
            feedback=payload["feedback"]
        # Local receipt only; never includes private answers or select/report questions.
        context={"schema":"rag-rsi-v3-literal-context-1","public_tasks":public,"exposed_feedback":feedback}
        freeze(folder/"literal_context.json",context)
        audit=audit_fit_literals(program["files"],proposed,public,exposed_feedback=feedback)
        freeze(folder/"literal_audit.json",audit)
        return audit

    def run(self):
        self._assert_frozen()
        files=root_files(self.manifest.get("root_config")); validate_sources(files)
        self._assert_frozen()
        root=read(self.directory/"root.json")
        if root is None:
            root=recoverable_record(self.archive,files,{},session_id="v3-root",attempt=0)
            save(self.directory/"root.json",root)
        else:
            root=_validate_receipt(self.archive,root,files,{},session_id="v3-root",attempt=0)
        results={}; cards=[]; attempts=[]
        baseline=self._measure(root,"D_fit"); results[root["node_id"]]=baseline
        cards.append(experience_card(baseline,None,operator="Draft",module="retrieval",step=0,mechanism="frozen root"))
        for step in range(self.manifest["expansions"]):
            folder=self.directory/"steps"/str(step)
            decision=self._decision(cards,root,baseline,step,attempts)
            decision["recent_rejections"]=[r for i in range(max(0,step-4),step)
                      if (r:=read(self.directory/"steps"/str(i)/"rejected.json"))]
            decision["recent_edit_scopes"]=[{"node_id":c["node_id"],
                "intended_target_module":c["intended_target_module"],"actual_edit_scope":c["actual_edit_scope"]}
                for c in cards[-4:] if c.get("actual_edit_scope")]
            freeze(folder/"decision.json",decision)
            parent=self.archive.load_node(decision["parent_node_id"])
            program=self.archive.load_program(parent["program_id"])
            development_result=deepcopy(results[parent["node_id"]])
            if parent.get("parent_node_id") in results:
                development_result["parent_measurement"]=deepcopy(results[parent["parent_node_id"]])
            proposal=read(folder/"proposal.json")
            received=read(folder/"received_proposal.json")
            child=read(folder/"child.json")
            rejected=read(folder/"rejected.json")
            if rejected:
                if proposal is not None or child is not None:
                    raise ValueError("rejected attempt also has a proposal or child receipt")
                if rejected.get("reason_code")=="fit_literal_match" or (folder/"literal_audit.json").exists():
                    if received is None:
                        raise ValueError("literal rejection has no received proposal")
                    proposed=_proposal_files(program,decision,received)
                    audit=self._audit_literals(folder,program,proposed,decision,cards,development_result)
                    if audit["status"]=="reject":
                        if rejected!=_literal_rejection(audit):
                            raise ValueError("saved literal rejection differs from observed source audit")
                    elif (rejected!={"reason":"previously visited program","next_step_allowed":True}
                          or not any(self.archive.load_program(c["program_id"])["files"]==proposed for c in cards)):
                        raise ValueError("saved rejection is not supported by literal audit or visited source")
                self._terminal_attempt(folder,decision,step,"rejected",attempts)
                continue
            if proposal is None:
                if child is not None:
                    raise ValueError("child receipt has no proposal")
                try:
                    if received is None:
                        received=deepcopy(self.developer.propose(deepcopy(program),deepcopy(decision),
                          self._memory(cards,decision),deepcopy(development_result),
                          deepcopy(self.panels["D_fit"]),deepcopy(self.references["D_fit"])))
                        self._assert_frozen()
                        freeze(folder/"received_proposal.json",received)
                    proposal=received
                    proposed=_proposal_files(program,decision,proposal)
                except (ValueError,SyntaxError) as exc:
                    self._assert_frozen()
                    save(folder/"rejected.json",{"reason":str(exc),"next_step_allowed":True})
                    self._terminal_attempt(folder,decision,step,"rejected",attempts)
                    continue
                audit=self._audit_literals(folder,program,proposed,decision,cards,development_result)
                if audit["status"]=="reject":
                    freeze(folder/"rejected.json",_literal_rejection(audit))
                    self._terminal_attempt(folder,decision,step,"rejected",attempts)
                    continue
                if any(self.archive.load_program(c["program_id"])["files"]==proposed for c in cards):
                    save(folder/"rejected.json",{"reason":"previously visited program","next_step_allowed":True})
                    self._terminal_attempt(folder,decision,step,"rejected",attempts)
                    continue
                proposal={"writes":proposal["writes"],"mechanism":proposal["mechanism"],
                          "intended_target_module":decision["target_module"]}
                save(folder/"proposal.json",proposal)
            else:
                # A saved proposal is an input to recovery, not an exemption from validation.
                proposed=_proposal_files(program,decision,proposal)
                if received is None:
                    raise ValueError("approved proposal has no received proposal checkpoint")
                if (_proposal_files(program,decision,received)!=proposed or received["mechanism"]!=proposal["mechanism"]):
                    raise ValueError("approved proposal receipt source differs from received proposal")
                audit=self._audit_literals(folder,program,proposed,decision,cards,development_result)
                if audit["status"]=="reject":
                    raise ValueError("saved accepted proposal fails current literal audit")
                if any(self.archive.load_program(c["program_id"])["files"]==proposed for c in cards):
                    raise ValueError("replayed proposal repeats a previously visited program")
            expected={"session_id":"v3-evolution","attempt":step,"parent_node_id":parent["node_id"]}
            if child is None:
                child=recoverable_record(self.archive,proposed,program["metadata"],**expected)
                save(folder/"child.json",child)
            else:
                child=_validate_receipt(self.archive,child,proposed,program["metadata"],**expected)
            scope=observe_edit_scope(program["files"],proposed,decision["target_module"])
            freeze(folder/"edit_scope.json",scope)
            measured=self._measure(child,"D_fit"); results[child["node_id"]]=measured
            card=experience_card(measured,results[parent["node_id"]],operator=decision["operator"],module=decision["target_module"],step=step+1,mechanism=proposal["mechanism"],edit_scope=scope)
            cards.append(card); freeze(folder/"experience.json",card)
            self._terminal_attempt(folder,decision,step,"measured",attempts,child["node_id"])
        # One immutable search freeze, one selection, one report. No shadow judging.
        freeze(self.directory/"search_frozen.json",{"cards":cards,"order":[c["node_id"] for c in cards],
            **({"terminal_attempts": attempts, "controls": self.controls} if self.controlled_run else {})})
        valid=[c for c in cards if c["valid_program"]]
        candidates=[root["node_id"]]
        for card in sorted(valid,key=lambda c:-c["score"]):
            if len(candidates)>=self.manifest.get("select_candidates",3): break
            if card["node_id"] not in candidates:
                candidates.append(card["node_id"])
        selected=[self._measure(self.archive.load_node(n),"D_select") for n in candidates]
        eligible=[(i,r) for i,r in enumerate(selected) if r["valid_program"]]
        if not eligible: raise ValueError("no technically valid selection candidate")
        creation_order={c["node_id"]:i for i,c in enumerate(cards)}
        winner=max(eligible,key=lambda ir:(ir[1]["score"],-creation_order[ir[1]["node_id"]]))[1]
        lock={"node_id":winner["node_id"],"program_id":winner["program_id"],
              "selection_score":winner["score"],"candidate_ids":candidates,"tie_rule":"reference then creation order"}
        freeze(self.directory/"delivery_lock.json",lock)
        delivered=self._measure(self.archive.load_node(winner["node_id"]),"D_report")
        reference=self._measure(root,"D_report")
        delivered_eligible=delivered.get("complete") is True and delivered.get("valid_program") is True
        reference_eligible=reference.get("complete") is True and reference.get("valid_program") is True
        quality_valid=delivered_eligible and reference_eligible
        raw_gain=delivered["score"]-reference["score"]
        raw_deltas={q:score-reference["per_question"][q] for q,score in delivered["per_question"].items()}
        report={"schema":"rag-rsi-v3-report-2","status":"complete" if quality_valid else "protocol_invalid",
            "metric":self.manifest["metric"],"quality_comparison_valid":quality_valid,
            "report_program_eligible":delivered_eligible,"reference_program_eligible":reference_eligible,
            "proxy_metric":bool(self.manifest.get("allow_proxy_metric")),"fit_nodes":len(cards),
            "delivery_lock":lock,"report_score":delivered["score"] if delivered_eligible else None,
            "reference_score":reference["score"] if reference_eligible else None,
            "paired_report_gain":raw_gain if quality_valid else None,
            "paired_question_deltas":raw_deltas if quality_valid else None,
            "raw_diagnostics":{"report_score":delivered["score"],"reference_score":reference["score"],
                "paired_report_gain":raw_gain,"paired_question_deltas":raw_deltas,
                "scores_are_diagnostic_if_ineligible":True},
            "independent_unit":"question, not repeat","report_used_for_decisions":False,
            "synthetic":bool(self.manifest.get("synthetic",False)),
            "quality_claim":("invalid_protocol_no_quality_claim" if not quality_valid else
                "engineering_fixture_only" if self.manifest.get("synthetic") else "estimate_on_frozen_report_panel")}
        if self.controlled_run:
            report.update(search_controls=deepcopy(self.controls),
                terminal_proposals=len(attempts), rejected_proposals=sum(a["status"] == "rejected" for a in attempts))
        save(self.directory/"report.json",report)
        return report
