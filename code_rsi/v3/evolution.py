"""One data-driven loop: edit -> execute -> measure -> experience -> select -> report.

The same runtime serves every task adapter. Old experiments remain immutable;
this module does not import any v25* or P4/P5 continuation script.
"""
from __future__ import annotations
import ast
from copy import deepcopy
import json
import hashlib
from pathlib import Path
from ..archive import ProgramArchive, Conflict, _safe as _safe_archive_path
from ..budget import digest, save, stable
from .datasets import validate_task_collection
from .execution import Measurement, root_files, validate_sources
from .experience_policy import DEFAULT_MODULES, choose_next, memory_for_action
from .diagnostics import compact_feedback
from .edit_scope import observe_edit_scope, validated_scope
from .edit_policy import validate_edit_policy, check_edit_policy
from .proposal_protocol import validate_proposal_protocol, source_identity, materialize_edits
from .runtime_contract import validate_runtime_contract
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


def _proposal_files(program, decision, proposal, proposal_protocol=None):
    protocol = validate_proposal_protocol(proposal_protocol)
    if protocol is not None and protocol["format"] == "exact_edits":
        files, _ = materialize_edits(program["files"], proposal, protocol)
        if proposal["intended_target_module"] != decision.get("intended_target_module", decision["target_module"]):
            raise ValueError("proposal intent differs from declared action")
        if proposal["change_status"] == "no_change":
            raise ValueError("model declared no change")
        return program_change(program, files)
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
                 proposal_model_factory=None, edit_policy=None, proposal_protocol=None, runtime_contract=None):
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
        self.edit_policy = validate_edit_policy(edit_policy)
        self.proposal_protocol = validate_proposal_protocol(proposal_protocol)
        self.runtime_contract = validate_runtime_contract(runtime_contract)
        if self.runtime_contract is not None:
            if self.proposal_protocol is None or self.proposal_protocol["format"] != "exact_edits":
                raise ValueError("runtime contract requires exact edits")
            if any(self.runtime_contract["edit_output"][key] != self.proposal_protocol[key]
                   for key in ("max_edits", "max_edit_chars")):
                raise ValueError("runtime contract edit bounds differ from the proposal protocol")
        self.feedback_condition = feedback_condition
        self.case_schedule = [] if feedback_condition == "rich" else deepcopy(case_schedule)
        self.proposal_model_factory = proposal_model_factory
        self._proposal_models = {}
        self._proposal_payloads = {}

    def configuration_snapshot(self):
        return {"feedback_condition": self.feedback_condition,
                "case_schedule": deepcopy(self.case_schedule),
                "proposal_model_factory_present": self.proposal_model_factory is not None,
                **({"edit_policy": deepcopy(self.edit_policy)} if self.edit_policy is not None else {}),
                **({"proposal_protocol": deepcopy(self.proposal_protocol)} if self.proposal_protocol is not None else {}),
                **({"runtime_contract": deepcopy(self.runtime_contract)} if self.runtime_contract is not None else {})}

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
        if self.edit_policy is not None:
            payload["edit_policy"] = deepcopy(self.edit_policy)
            if self.edit_policy["mode"] == "prompt_only":
                payload["edit_boundary"] = (
                    "Only change literal strings at rag.py CONFIG.prompts for the stages listed in edit_policy. "
                    "Return complete changed files. Keep rag_core.py byte-identical; keep every other configuration "
                    "value and wrapper code unchanged. Do not add functions, imports, calls, data, or hardcoded "
                    "examples/answers. The host rejects out-of-scope changes before candidate execution; "
                    "rejections still consume a proposal opportunity. Module intent and observed scope do not "
                    "override this frozen edit policy. Existing development-literal checks remain in force.")
        if self.proposal_protocol is not None:
            payload["proposal_protocol"] = deepcopy(self.proposal_protocol)
            if self.proposal_protocol["format"] == "exact_edits":
                payload["parent_source_sha256"] = source_identity(program["files"])
                payload["edit_boundary"] = payload["edit_boundary"].replace(
                    "Return complete changed files.",
                    "Return literal edits against the supplied read-only parent. The host constructs a separate candidate. "
                    "Declare change_status='modified' with concrete edits or 'no_change' with edits=[]. "
                    "Model claims, source text changes and AST changes are recorded separately.")
        if self.runtime_contract is not None:
            payload["runtime_contract"] = deepcopy(self.runtime_contract)
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
        # Exact edits are saved raw by the host before validation/materialization.
        if self.proposal_protocol is None or self.proposal_protocol["format"] == "whole_files":
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


def _edit_policy_rejection(receipt):
    return {"reason": "proposal violates frozen edit policy", "reason_code": "edit_policy_violation",
            "edit_policy_receipt_sha256": receipt["receipt_sha256"],
            "violations": deepcopy(receipt["reason_codes"]), "next_step_allowed": True}


PHASE_SCHEMA = "rag-rsi-evolution-phases-1"
PHASE_ROLES = {"search": "D_fit", "select": "D_select", "report": "D_report"}


def _byte_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _reference_group(ref):
    group = ref.get("pair_group_id", ref.get("source_question_id"))
    return None if group is None else str(group)


class EvolutionRunner:
    def __init__(self,directory,manifest,panels,references,model_factory,developer,backend_factory=None,scorer=None,*,reference_loader=None,root_model_factory=None):
        self.directory=Path(directory)
        manifest,panels,references=deepcopy((manifest,panels,references))
        self.manifest=manifest; self.panels=panels; self.references=references
        self.developer=developer
        self.proposal_protocol = validate_proposal_protocol(manifest.get("proposal_protocol"))
        if "proposal_protocol" in manifest and self.proposal_protocol is None:
            raise ValueError("explicit proposal_protocol cannot be null")
        if isinstance(developer, ProgramDeveloper) and developer.proposal_protocol != self.proposal_protocol:
            raise ValueError("developer proposal protocol differs from frozen manifest")
        self.runtime_contract = validate_runtime_contract(manifest.get("runtime_contract"))
        if "runtime_contract" in manifest and self.runtime_contract is None:
            raise ValueError("explicit runtime_contract cannot be null")
        if isinstance(developer, ProgramDeveloper) and developer.runtime_contract != self.runtime_contract:
            raise ValueError("developer runtime contract differs from frozen manifest")
        if self.runtime_contract is not None and self.runtime_contract["host_limits"] != manifest.get("limits"):
            raise ValueError("runtime contract host limits differ from frozen manifest")
        self.edit_policy = validate_edit_policy(manifest.get("edit_policy"))
        if "edit_policy" in manifest and self.edit_policy is None:
            raise ValueError("explicit edit_policy cannot be null")
        if isinstance(developer, ProgramDeveloper) and developer.edit_policy != self.edit_policy:
            raise ValueError("developer edit policy differs from frozen manifest")
        self.lifecycle = manifest.get("lifecycle")
        self.reference_loader = reference_loader
        self._loaded_reference_hashes = {}
        self._active_phase = None
        self.shared_root = deepcopy(manifest.get("shared_root"))
        self.root_measure = None
        if self.lifecycle is not None:
            lc = self.lifecycle
            if (not isinstance(lc, dict) or set(lc) != {"schema", "phase_order", "reference_bindings", "reference_groups"}
                    or lc["schema"] != PHASE_SCHEMA
                    or lc["phase_order"] not in (["search"], ["search", "select", "report"])):
                raise ValueError("invalid explicit evolution lifecycle")
            roles = {PHASE_ROLES[phase] for phase in lc["phase_order"]}
            if (set(panels) != roles or references != {} or not callable(reference_loader)
                    or not isinstance(lc["reference_bindings"], dict) or set(lc["reference_bindings"]) != roles
                    or not isinstance(lc["reference_groups"], dict) or set(lc["reference_groups"]) != roles):
                raise ValueError("staged runs require exact panels, unloaded references and a reference loader")
            groups = {}
            for role, tasks in panels.items():
                binding = lc["reference_bindings"][role]
                if (not isinstance(binding, dict) or set(binding) != {"path", "sha256"}
                        or not isinstance(binding["path"], str) or not Path(binding["path"]).is_absolute()
                        or not isinstance(binding["sha256"], str) or len(binding["sha256"]) != 64
                        or any(ch not in "0123456789abcdef" for ch in binding["sha256"])):
                    raise ValueError("references require absolute file and SHA256 bindings")
                mapping = lc["reference_groups"][role]
                if not isinstance(mapping, dict) or set(mapping) != {t["question_id"] for t in tasks}:
                    raise ValueError("exact answer-free reference group metadata required")
                for task in tasks:
                    group = mapping[task["question_id"]]
                    if group is not None:
                        if not isinstance(group, str) or not group:
                            raise ValueError("invalid reference group identity")
                        key = (task["dataset"], group)
                        if key in groups and groups[key] != role:
                            raise ValueError("source question group crosses roles")
                        groups[key] = role
        elif reference_loader is not None or set(panels) != set(PHASE_ROLES.values()):
            raise ValueError("three nonempty independent panels required")
        if any(not x for x in panels.values()):
            raise ValueError("nonempty independent panels required")
        for role,tasks in panels.items():
            validate_task_collection(tasks)
            if self.lifecycle is None and set(references[role])!={t["question_id"] for t in tasks}:
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
        if self.shared_root is not None:
            shared = self.shared_root
            if (self.lifecycle is None or not isinstance(shared, dict)
                    or set(shared) != {"schema", "directory", "block_id", "bank"}
                    or shared["schema"] != "rag-rsi-shared-root-1"
                    or not isinstance(shared["directory"], str) or not Path(shared["directory"]).is_absolute()
                    or not isinstance(shared["block_id"], str) or not shared["block_id"].strip()
                    or shared["bank"] != "shared-root/" + shared["block_id"]
                    or self.controls["parent_policy"] != "fixed_root" or self.controls["memory"] != "none"
                    or self.controls["feedback"] not in ("cases", "trace") or not callable(root_model_factory)
                    or not isinstance(manifest.get("model_identity"), str) or not manifest["model_identity"]):
                raise ValueError("shared root requires a bounded staged fixed-root feedback experiment")
            shared_dir = _safe_archive_path(shared["directory"])
            own_dir = self.directory.resolve()
            if shared_dir == own_dir or shared_dir in own_dir.parents or own_dir in shared_dir.parents:
                raise ValueError("shared root and condition directories must not contain one another")
        elif root_model_factory is not None:
            raise ValueError("root_model_factory requires a frozen shared-root binding")
        self._frozen_manifest=deepcopy(self._snapshot())
        if self.lifecycle is not None and self._frozen_manifest["private_reference_file_hashes"] != {
                r: b["sha256"] for r, b in self.lifecycle["reference_bindings"].items()}:
            raise ValueError("frozen reference file bytes changed")
        freeze(self.directory/"manifest.json",self._frozen_manifest)
        self.archive=ProgramArchive(self.directory/"archive")
        self.measure=Measurement(self.archive,self.directory/"measurements",model_factory,backend_factory,
             metric=manifest["metric"],scorer=scorer,limits=manifest.get("limits"),
             allow_proxy_metrics=manifest.get("allow_proxy_metric",False))
        if self.shared_root is not None:
            def checked_root_model(bank):
                model = root_model_factory(bank)
                if getattr(model, "identity", None) != self.manifest["model_identity"]:
                    raise ValueError("shared root model identity differs from its frozen contract")
                return model
            self.root_measure = Measurement(self.archive, shared_dir, checked_root_model, backend_factory,
                metric=manifest["metric"], scorer=scorer, limits=manifest.get("limits"),
                allow_proxy_metrics=manifest.get("allow_proxy_metric", False))
            root_source = root_files(manifest.get("root_config"))
            self._shared_contract = {"schema": "rag-rsi-shared-root-contract-1",
                "root_files_sha256": {name: hashlib.sha256(value.encode("utf-8")).hexdigest()
                                       for name, value in root_source.items()},
                "panel_hash": digest(self.panels["D_fit"]),
                "reference_file_sha256": self.lifecycle["reference_bindings"]["D_fit"]["sha256"],
                "metric": manifest["metric"], "evaluator_epoch": self.root_measure.epoch,
                "limits": deepcopy(manifest.get("limits")), "repeats": manifest.get("repeats", 1),
                "model_identity": manifest["model_identity"], "runtime_source_hashes": _runtime_source_hashes(),
                "block_id": shared["block_id"], "bank": shared["bank"]}
            contract_path = shared_dir/"contract.json"
            if not contract_path.exists() and any(shared_dir.rglob("*.json")):
                raise ValueError("shared root artifacts exist without their contract")
            freeze(contract_path, self._shared_contract)
            self._verify_shared_root()

    def _shared_json_inventory(self):
        directory = _safe_archive_path(self.shared_root["directory"])
        result = []
        for path in directory.rglob("*.json"):
            _safe_archive_path(path)
            relative = path.relative_to(directory).as_posix()
            if relative != "shared_root_seal.json":
                result.append(relative)
        return sorted(result)

    def _verify_shared_root(self):
        if self.shared_root is None:
            return None
        directory = _safe_archive_path(self.shared_root["directory"])
        if read(directory/"contract.json") != self._shared_contract:
            raise ValueError("shared root contract changed")
        local = self.directory/"measurements"/"shared_root.json"
        seal_path = directory/"shared_root_seal.json"
        if not seal_path.exists():
            if local.exists():
                raise ValueError("shared root reference has no frozen external seal")
            return None
        seal = read(seal_path)
        fields = {"schema", "contract_sha256", "measurement_identity", "result_sha256", "artifacts", "seal_hash"}
        if (not isinstance(seal, dict) or set(seal) != fields
                or seal["schema"] != "rag-rsi-shared-root-seal-1"
                or seal["contract_sha256"] != digest(self._shared_contract)
                or seal["seal_hash"] != digest({k: v for k, v in seal.items() if k != "seal_hash"})
                or not isinstance(seal["measurement_identity"], str) or len(seal["measurement_identity"]) != 64
                or any(c not in "0123456789abcdef" for c in seal["measurement_identity"])
                or not isinstance(seal["artifacts"], dict)
                or sorted(seal["artifacts"]) != self._shared_json_inventory()):
            raise ValueError("invalid or incomplete shared root seal")
        identity = seal["measurement_identity"]
        required = {"contract.json", identity + "/measurement.json"}
        measured = [name for name in seal["artifacts"] if name.endswith("/measured.json")]
        if (not required <= set(seal["artifacts"]) or not measured
                or any(name != "contract.json" and not name.startswith(identity + "/") for name in seal["artifacts"])):
            raise ValueError("shared root seal does not cover one complete measurement")
        for name, expected in seal["artifacts"].items():
            path = _safe_archive_path(directory/name)
            if Path(name).is_absolute() or ".." in Path(name).parts or _byte_hash(path) != expected:
                raise ValueError("shared root artifact changed")
        result = read(directory/identity/"measurement.json")
        if (not isinstance(result, dict) or result.get("identity_hash") != identity
                or result.get("role") != "D_fit" or result.get("panel_hash") != self._shared_contract["panel_hash"]
                or result.get("evaluator_epoch") != self._shared_contract["evaluator_epoch"]
                or result.get("metric") != self._shared_contract["metric"]
                or result.get("complete") is not True or digest(result) != seal["result_sha256"]):
            raise ValueError("shared root result differs from its sealed measurement")
        reference = {"schema": "rag-rsi-shared-root-reference-1", "shared_root": deepcopy(self.shared_root),
            "contract_sha256": digest(self._shared_contract), "seal_sha256": _byte_hash(seal_path),
            "result": result}
        if local.exists() and read(local) != reference:
            raise ValueError("local shared root reference differs from the frozen external result")
        return reference

    def _measure_shared_root(self, node):
        _validate_receipt(self.archive, node, root_files(self.manifest.get("root_config")), {},
                          session_id="v3-root", attempt=0)
        existing = self._verify_shared_root()
        result = self.root_measure.run(node, deepcopy(self.panels["D_fit"]), deepcopy(self.references["D_fit"]),
            role="D_fit", bank=self.shared_root["bank"], repeats=self.manifest.get("repeats", 1),
            cached_only=existing is not None)
        if existing is not None and result != existing["result"]:
            raise ValueError("shared root cache reconstruction changed its sealed result")
        directory = Path(self.shared_root["directory"])
        if existing is None:
            seal = {"schema": "rag-rsi-shared-root-seal-1", "contract_sha256": digest(self._shared_contract),
                "measurement_identity": result["identity_hash"], "result_sha256": digest(result),
                "artifacts": {name: _byte_hash(directory/name) for name in self._shared_json_inventory()}}
            seal["seal_hash"] = digest(seal)
            freeze(directory/"shared_root_seal.json", seal)
        reference = self._verify_shared_root()
        freeze(self.directory/"measurements"/"shared_root.json", reference)
        return result

    def _snapshot(self):
        return {**deepcopy(self.manifest),
                "public_panel_hashes":{r:digest(ts) for r,ts in self.panels.items()},
                **({"private_reference_file_hashes": {
                    role: _byte_hash(binding["path"])
                    for role, binding in self.lifecycle["reference_bindings"].items()}}
                   if self.lifecycle is not None else
                   {"private_reference_hashes":{r:digest(ref) for r,ref in self.references.items()}}),
                "runtime_source_hashes":_runtime_source_hashes(),
                **({"active_controls": deepcopy(self.controls),
                    "developer_configuration": self.developer.configuration_snapshot()
                        if isinstance(self.developer, ProgramDeveloper) else None}
                   if self.controlled_run or self.edit_policy is not None or self.proposal_protocol is not None or self.runtime_contract is not None else {}),
                **({"active_edit_policy": deepcopy(self.edit_policy)} if self.edit_policy is not None else {}),
                **({"active_proposal_protocol": deepcopy(self.proposal_protocol)} if self.proposal_protocol is not None else {}),
                **({"active_runtime_contract": deepcopy(self.runtime_contract)} if self.runtime_contract is not None else {})}

    def _assert_frozen(self):
        if self.runtime_contract != validate_runtime_contract(self.manifest.get("runtime_contract")):
            raise ValueError("frozen runtime contract changed")
        if self.proposal_protocol != validate_proposal_protocol(self.manifest.get("proposal_protocol")):
            raise ValueError("frozen proposal protocol changed")
        if self.edit_policy != validate_edit_policy(self.manifest.get("edit_policy")):
            raise ValueError("frozen edit policy changed")
        if self.shared_root != self.manifest.get("shared_root"):
            raise ValueError("shared root binding changed")
        if self.shared_root is not None:
            self._verify_shared_root()
        if self.lifecycle != self.manifest.get("lifecycle"):
            raise ValueError("frozen lifecycle changed")
        if self.lifecycle is not None and {r: digest(ref) for r, ref in self.references.items()} != self._loaded_reference_hashes:
            raise ValueError("loaded reference content changed outside phase loader")
        current=self._snapshot()
        if current!=self._frozen_manifest:
            raise ValueError("frozen run inputs or runtime source changed")
        freeze(self.directory/"manifest.json",current)

    @property
    def exact_edits(self):
        return self.proposal_protocol is not None and self.proposal_protocol["format"] == "exact_edits"

    def _proposal_rejection(self, error, received):
        record = {"reason": str(error), "next_step_allowed": True}
        if self.exact_edits and received is not None:
            record["received_proposal_sha256"] = digest(received)
        return record

    def _materialize_received(self, folder, program, decision, received, *, required=False):
        if not self.exact_edits:
            return _proposal_files(program, decision, received)
        proposed, receipt = materialize_edits(program["files"], received, self.proposal_protocol)
        observation = {"schema": "rag-rsi-source-change-1",
            "parent_source_sha256": source_identity(program["files"]),
            "child_source_sha256": source_identity(proposed),
            "declared_change_status": received["change_status"],
            "source_changed": proposed != program["files"], "source_valid": True}
        try:
            validate_sources(proposed)
            observation["ast_changed"] = any(
                ast.dump(ast.parse(proposed[name]), include_attributes=False) !=
                ast.dump(ast.parse(program["files"][name]), include_attributes=False) for name in proposed)
        except (ValueError, SyntaxError):
            observation.update(source_valid=False, ast_changed=None)
        observation["passes_source_change_gate"] = (
            received["change_status"] == "modified" and observation["source_valid"]
            and observation["ast_changed"] is True)
        observation["receipt_sha256"] = digest(observation)
        for name, value in (("materialization.json", receipt), ("change_receipt.json", observation)):
            path = folder/name
            if required and not path.is_file():
                raise RuntimeError("saved proposal lacks its " + name)
            try:
                freeze(path, value)
            except ValueError as error:
                raise RuntimeError("proposal checkpoint mismatch: " + name) from error
        return _proposal_files(program, decision, received, self.proposal_protocol)

    def _verify_rejected_materialization(self, folder, program, decision, received, rejected):
        """Return True only for a revalidated structural/no-change rejection."""
        if not self.exact_edits:
            return False
        checkpoints = [folder/"materialization.json", folder/"change_receipt.json"]
        if received is None:
            if any(p.exists() for p in checkpoints) or "received_proposal_sha256" in rejected:
                raise ValueError("rejected proposal lost its raw input")
            return False  # Provider-level JSON failure is kept in the request cache.
        if "received_proposal_sha256" in rejected and rejected["received_proposal_sha256"] != digest(received):
            raise ValueError("rejected raw proposal changed")
        try:
            materialize_edits(program["files"], received, self.proposal_protocol)
        except ValueError as error:
            if any(p.exists() for p in checkpoints) or rejected != self._proposal_rejection(error, received):
                raise ValueError("saved patch rejection is unsupported") from error
            return True
        try:
            self._materialize_received(folder, program, decision, received, required=True)
        except (ValueError, SyntaxError) as error:
            if rejected != self._proposal_rejection(error, received):
                raise ValueError("saved source-change rejection is unsupported") from error
            return True
        if "received_proposal_sha256" in rejected:
            raise ValueError("saved structural rejection no longer matches source")
        return False  # Existing policy/literal/visited checks must still run.

    def _audit_edit_policy(self, folder, program, proposed, *, required=False):
        if self.edit_policy is None:
            return None
        path = folder / "edit_policy.json"
        if required and not path.is_file():
            raise ValueError("saved proposal lacks its edit policy receipt")
        receipt = check_edit_policy(program["files"], proposed, self.edit_policy)
        freeze(path, receipt)
        return receipt

    def _measure(self,node,role):
        self._assert_frozen()
        if self.lifecycle is not None and (self._active_phase is None or PHASE_ROLES[self._active_phase] != role):
            raise ValueError("measurement role differs from active phase")
        if self.shared_root is not None and role == "D_fit" and node.get("parent_node_id") is None:
            result = self._measure_shared_root(node)
        else:
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

    def _run_search(self, stop_after=None):
        target = self.manifest["expansions"] if stop_after is None else stop_after
        if type(target) is not int or not 0 <= target <= self.manifest["expansions"]:
            raise ValueError("partial search count must be within frozen expansion count")
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
        for step in range(target):
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
                if self._verify_rejected_materialization(folder, program, decision, received, rejected):
                    self._terminal_attempt(folder, decision, step, "rejected", attempts)
                    continue
                if self.edit_policy is not None:
                    if received is None and (rejected.get("reason_code") == "edit_policy_violation"
                                             or (folder/"edit_policy.json").exists()):
                        raise ValueError("policy rejection has no received proposal")
                    if received is not None:
                        try:
                            proposed = _proposal_files(program, decision, received, self.proposal_protocol)
                        except (ValueError, SyntaxError):
                            if rejected.get("reason_code") == "edit_policy_violation" or (folder/"edit_policy.json").exists():
                                raise ValueError("policy receipt cannot cover a malformed proposal")
                        else:
                            receipt = self._audit_edit_policy(folder, program, proposed, required=True)
                            if not receipt["allowed"]:
                                if rejected != _edit_policy_rejection(receipt):
                                    raise ValueError("saved policy rejection differs from observed source")
                                self._terminal_attempt(folder, decision, step, "rejected", attempts)
                                continue
                            if rejected.get("reason_code") == "edit_policy_violation":
                                raise ValueError("saved policy rejection is not supported")
                if rejected.get("reason_code")=="fit_literal_match" or (folder/"literal_audit.json").exists():
                    if received is None:
                        raise ValueError("literal rejection has no received proposal")
                    proposed=_proposal_files(program,decision,received,self.proposal_protocol)
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
                    proposed=self._materialize_received(folder,program,decision,proposal)
                except (ValueError,SyntaxError) as exc:
                    self._assert_frozen()
                    save(folder/"rejected.json",self._proposal_rejection(exc,received))
                    self._terminal_attempt(folder,decision,step,"rejected",attempts)
                    continue
                policy_receipt = self._audit_edit_policy(folder, program, proposed)
                if policy_receipt is not None and not policy_receipt["allowed"]:
                    freeze(folder/"rejected.json", _edit_policy_rejection(policy_receipt))
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
                writes = ({name: source for name, source in proposed.items() if source != program["files"][name]}
                          if self.exact_edits else proposal["writes"])
                proposal={"writes":writes,"mechanism":proposal["mechanism"],
                          "intended_target_module":decision["target_module"]}
                save(folder/"proposal.json",proposal)
            else:
                # A saved proposal is an input to recovery, not an exemption from validation.
                proposed=_proposal_files(program,decision,proposal)
                if received is None:
                    raise ValueError("approved proposal has no received proposal checkpoint")
                if (self._materialize_received(folder,program,decision,received,required=True)!=proposed or received["mechanism"]!=proposal["mechanism"]):
                    raise ValueError("approved proposal receipt source differs from received proposal")
                policy_receipt = self._audit_edit_policy(folder, program, proposed, required=True)
                if policy_receipt is not None and not policy_receipt["allowed"]:
                    raise ValueError("saved accepted proposal violates frozen edit policy")
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
        if target == self.manifest["expansions"]:
            freeze(self.directory/"search_frozen.json",{"cards":cards,"order":[c["node_id"] for c in cards],
                **({"terminal_attempts": attempts, "controls": self.controls} if self.controlled_run else {})})
        return root, cards, attempts

    def _run_select(self, root, cards):
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
        return lock

    def _run_report(self, root, cards, attempts, lock):
        delivered=self._measure(self.archive.load_node(lock["node_id"]),"D_report")
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
        if self.runtime_contract is not None:
            report["runtime_contract_sha256"] = digest(self.runtime_contract)
        if self.proposal_protocol is not None:
            report["proposal_protocol"] = deepcopy(self.proposal_protocol)
        if self.edit_policy is not None:
            report["edit_policy"] = deepcopy(self.edit_policy)
            report["edit_policy_rejections"] = sum(
                (read(self.directory/"steps"/str(i)/"rejected.json") or {}).get("reason_code") == "edit_policy_violation"
                for i in range(self.manifest["expansions"]))
        save(self.directory/"report.json",report)
        return report

    def run(self):
        """Legacy complete lifecycle, preserving v1/v2 receipts and cache banks."""
        if self.lifecycle is not None:
            raise ValueError("staged lifecycle requires explicit run_phase; automatic full run is forbidden")
        root, cards, attempts = self._run_search()
        lock = self._run_select(root, cards)
        return self._run_report(root, cards, attempts, lock)

    def _load_phase_reference(self, phase):
        role = PHASE_ROLES[phase]
        self._assert_frozen()
        if role in self.references:
            return
        binding = self.lifecycle["reference_bindings"][role]
        raw = Path(binding["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
            raise ValueError("frozen reference bytes changed before unlock")
        refs = deepcopy(self.reference_loader(role))
        if refs != json.loads(raw):
            raise ValueError("reference loader differs from frozen file")
        tasks = self.panels[role]
        if not isinstance(refs, dict) or set(refs) != {t["question_id"] for t in tasks}:
            raise ValueError("exact role-scoped reference set required")
        for task in tasks:
            qid = task["question_id"]
            ref = refs[qid]
            if (not isinstance(ref, dict) or ref.get("question_id") != qid
                    or ref.get("dataset") != task["dataset"] or ref.get("reference_available") is not True
                    or not isinstance(ref.get("answers"), list) or not ref["answers"]
                    or any(not isinstance(a, str) or not a.strip() for a in ref["answers"])
                    or ref.get("answerable") is False):
                raise ValueError("complete available answer references required")
            if _reference_group(ref) != self.lifecycle["reference_groups"][role][qid]:
                raise ValueError("reference group differs from frozen answer-free metadata")
        self.references[role] = refs
        self._loaded_reference_hashes[role] = digest(refs)
        self._assert_frozen()

    def _tree_inventory(self, name):
        return sorted(p.relative_to(self.directory).as_posix()
                      for p in (self.directory/name).rglob("*.json") if p.is_file())

    def _seal_phase(self, phase, result):
        paths = {"manifest.json", "root.json", "search_frozen.json"}
        for name in ("archive", "steps", "measurements", "search_progress"):
            paths.update(self._tree_inventory(name))
        for previous in self.lifecycle["phase_order"][:self.lifecycle["phase_order"].index(phase)]:
            paths.add(f"phase_{previous}.json")
        if phase in ("select", "report"):
            paths.add("delivery_lock.json")
        if phase == "report":
            paths.add("report.json")
        seal = {"schema": PHASE_SCHEMA, "phase": phase,
                "manifest_hash": digest(self._frozen_manifest), "result": deepcopy(result),
                "measurement_files": self._tree_inventory("measurements"),
                "artifacts": {p: _byte_hash(self.directory/p) for p in sorted(paths)},
                "immutable_trees": {n: self._tree_inventory(n) for n in ("archive", "steps")}}
        seal["seal_hash"] = digest(seal)
        freeze(self.directory/f"phase_{phase}.json", seal)
        return result

    def _verify_phase(self, phase):
        seal = read(self.directory/f"phase_{phase}.json")
        if (not isinstance(seal, dict) or set(seal) != {"schema", "phase", "manifest_hash", "result", "artifacts", "immutable_trees", "measurement_files", "seal_hash"}
                or seal["seal_hash"] != digest({k: v for k, v in seal.items() if k != "seal_hash"})
                or seal["schema"] != PHASE_SCHEMA or seal["phase"] != phase
                or seal["manifest_hash"] != digest(self._frozen_manifest)
                or not isinstance(seal["artifacts"], dict)
                or seal["immutable_trees"] != {n: self._tree_inventory(n) for n in ("archive", "steps")}):
            raise ValueError("missing or invalid frozen phase receipt: " + phase)
        required = {"manifest.json", "root.json", "search_frozen.json"}
        if phase in ("select", "report"):
            required.add("delivery_lock.json")
        if phase == "report":
            required.add("report.json")
        required.update(p for paths in seal["immutable_trees"].values() for p in paths)
        if (not isinstance(seal["measurement_files"], list) or not seal["measurement_files"]
                or any(not isinstance(p, str) or not p.startswith("measurements/")
                       or not p.endswith(".json") for p in seal["measurement_files"])):
            raise ValueError("frozen phase requires its measurement inventory")
        required.update(seal["measurement_files"])
        required.update(f"phase_{previous}.json" for previous in
                        self.lifecycle["phase_order"][:self.lifecycle["phase_order"].index(phase)])
        if not required <= set(seal["artifacts"]):
            raise ValueError("frozen phase receipt omits required artifacts")
        for name, expected in seal["artifacts"].items():
            path = self.directory/name
            if (not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts
                    or not path.is_file() or _byte_hash(path) != expected):
                raise ValueError("frozen phase artifact changed: " + str(name))
        search = read(self.directory/"search_frozen.json")
        root = read(self.directory/"root.json")
        if (not isinstance(search, dict) or not search.get("cards")
                or search.get("order") != [c["node_id"] for c in search["cards"]]
                or search["order"][0] != root["node_id"]):
            raise ValueError("invalid frozen search order")
        _validate_receipt(self.archive, root, root_files(self.manifest.get("root_config")), {},
                          session_id="v3-root", attempt=0)
        for card in search["cards"]:
            node = self.archive.load_node(card["node_id"])
            self.archive.load_program(node["program_id"])
            if node["program_id"] != card["program_id"]:
                raise ValueError("frozen card differs from archived program")
        if phase == "search":
            expected_result = {"schema": "rag-rsi-v3-search-phase-1", "status": "search_frozen", "search": search}
        elif phase == "select":
            lock = read(self.directory/"delivery_lock.json")
            if lock["node_id"] not in search["order"] or lock["node_id"] not in lock["candidate_ids"]:
                raise ValueError("delivery lock outside frozen candidates")
            expected_result = {"schema": "rag-rsi-v3-select-phase-1", "status": "delivery_locked", "delivery_lock": lock}
        else:
            expected_result = read(self.directory/"report.json")
            if expected_result.get("delivery_lock") != read(self.directory/"delivery_lock.json"):
                raise ValueError("report differs from frozen delivery lock")
        if seal["result"] != expected_result:
            raise ValueError("frozen phase result differs from its artifacts")
        return deepcopy(seal["result"])

    def _progress_files(self):
        files = {}
        for path in (self.directory/"search_progress").glob("*.json"):
            if not path.stem.isdigit() or str(int(path.stem)) != path.stem:
                raise ValueError("invalid partial search checkpoint identity")
            n = int(path.stem)
            if not 0 <= n < self.manifest["expansions"]:
                raise ValueError("partial search checkpoint exceeds frozen expansions")
            files[n] = path
        return dict(sorted(files.items()))

    def _verify_search_progress(self, n):
        path = self.directory/"search_progress"/(str(n)+".json")
        checkpoint = read(path)
        fields = {"schema", "manifest_hash", "completed_opportunities", "result", "artifacts", "measurement_files", "checkpoint_hash"}
        if (not isinstance(checkpoint, dict) or set(checkpoint) != fields
                or checkpoint["schema"] != "rag-rsi-search-progress-checkpoint-1"
                or checkpoint["manifest_hash"] != digest(self._frozen_manifest)
                or checkpoint["completed_opportunities"] != n
                or checkpoint["checkpoint_hash"] != digest({k: v for k, v in checkpoint.items() if k != "checkpoint_hash"})
                or not isinstance(checkpoint["artifacts"], dict)
                or not isinstance(checkpoint["measurement_files"], list) or not checkpoint["measurement_files"]):
            raise ValueError("invalid partial search checkpoint")
        required = {"manifest.json", "root.json", *checkpoint["measurement_files"]}
        if not required <= set(checkpoint["artifacts"]):
            raise ValueError("partial search checkpoint omits required artifacts")
        for name, expected in checkpoint["artifacts"].items():
            if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("unsafe partial search artifact path")
            target = self.directory/name
            if not target.is_file() or _byte_hash(target) != expected:
                raise ValueError("partial search artifact changed")
        result = checkpoint["result"]
        if (not isinstance(result, dict) or set(result) != {"schema", "status", "completed_opportunities", "search"}
                or result["schema"] != "rag-rsi-v3-search-progress-1" or result["status"] != "search_in_progress"
                or result["completed_opportunities"] != n or not isinstance(result["search"], dict)):
            raise ValueError("partial search result differs")
        search = result["search"]
        root = read(self.directory/"root.json")
        if (not isinstance(search.get("cards"), list) or not search["cards"]
                or search.get("order") != [c["node_id"] for c in search["cards"]]
                or search["order"][0] != root["node_id"]
                or (self.controlled_run and len(search.get("terminal_attempts", [])) != n)):
            raise ValueError("partial search result has incomplete opportunities")
        return deepcopy(result)

    def _freeze_search_progress(self, n, result):
        paths = {"manifest.json", "root.json"}
        for name in ("archive", "steps", "measurements", "search_progress"):
            paths.update(self._tree_inventory(name))
        checkpoint = {"schema": "rag-rsi-search-progress-checkpoint-1",
            "manifest_hash": digest(self._frozen_manifest), "completed_opportunities": n,
            "result": deepcopy(result), "measurement_files": self._tree_inventory("measurements"),
            "artifacts": {name: _byte_hash(self.directory/name) for name in sorted(paths)}}
        checkpoint["checkpoint_hash"] = digest(checkpoint)
        freeze(self.directory/"search_progress"/(str(n)+".json"), checkpoint)
        return result

    def run_search_until(self, n):
        if type(n) is not int or not 0 <= n <= self.manifest["expansions"]:
            raise ValueError("partial search count must be within frozen expansion count")
        complete = self.check_phase("search")
        if complete is not None:
            return complete
        progress = self._progress_files()
        if n in progress:
            return self._verify_search_progress(n)
        if progress and n < max(progress):
            raise ValueError("cannot reconstruct an unrecorded earlier partial search prefix")
        self._load_phase_reference("search")
        self.check_phase("search")
        self._active_phase = "search"
        try:
            root, cards, attempts = self._run_search(stop_after=n)
            search = {"cards": cards, "order": [c["node_id"] for c in cards],
                **({"terminal_attempts": attempts, "controls": self.controls} if self.controlled_run else {})}
            self._assert_frozen()
            if n == self.manifest["expansions"]:
                return self._seal_phase("search", {"schema": "rag-rsi-v3-search-phase-1",
                    "status": "search_frozen", "search": read(self.directory/"search_frozen.json")})
            result = {"schema": "rag-rsi-v3-search-progress-1", "status": "search_in_progress",
                "completed_opportunities": n, "search": search}
            return self._freeze_search_progress(n, result)
        finally:
            self._active_phase = None

    def check_phase(self, phase):
        """Validate predecessors without parsing any private answer references."""
        self._assert_frozen()
        for n in self._progress_files():
            self._verify_search_progress(n)
        if self.lifecycle is None or phase not in self.lifecycle["phase_order"]:
            raise ValueError("phase is outside frozen lifecycle")
        for previous in self.lifecycle["phase_order"][:self.lifecycle["phase_order"].index(phase)]:
            self._verify_phase(previous)
        if (self.directory/f"phase_{phase}.json").exists():
            return self._verify_phase(phase)
        return None

    def run_phase(self, phase):
        complete = self.check_phase(phase)
        if complete is not None:
            return complete
        self._load_phase_reference(phase)
        # A loader cannot alter the predecessor checkpoint and then dispatch.
        self.check_phase(phase)
        self._active_phase = phase
        try:
            if phase == "search":
                self._run_search()
                result = {"schema": "rag-rsi-v3-search-phase-1", "status": "search_frozen",
                          "search": read(self.directory/"search_frozen.json")}
            else:
                root = read(self.directory/"root.json")
                search = read(self.directory/"search_frozen.json")
                if phase == "select":
                    result = {"schema": "rag-rsi-v3-select-phase-1", "status": "delivery_locked",
                              "delivery_lock": self._run_select(root, search["cards"])}
                else:
                    result = self._run_report(root, search["cards"], search.get("terminal_attempts", []),
                                              read(self.directory/"delivery_lock.json"))
            self._assert_frozen()
            return self._seal_phase(phase, result)
        finally:
            self._active_phase = None
