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
from .experience_policy import choose_next, memory_for_action


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
    if not isinstance(proposal,dict) or set(proposal)!={"writes","mechanism","target_module"}:
        raise ValueError("invalid development proposal")
    if not isinstance(proposal["mechanism"],str) or not proposal["mechanism"].strip():
        raise ValueError("invalid development proposal")
    if proposal["target_module"]!=decision["target_module"]:
        raise ValueError("proposal module differs from declared action")
    return program_change(program,proposal["writes"])


class ProgramDeveloper:
    def __init__(self, model):
        self.model=model

    def propose(self, program, decision, experience, result, tasks, references):
        if result["role"]!="D_fit":
            raise ValueError("developer receives D_fit only")
        by_id={t["question_id"]:t for t in tasks}
        failures=[r for r in result["rows"] if r["score"]<1 or not r["execution_ok"]]
        examples=[]
        for row in failures[:4]:
            qid=row["question_id"]
            reported=row.get("candidate_reported") or {}
            examples.append({"question":by_id[qid]["question"],"reference_not_sent":True,
                "prediction":row["answer"],"host_score":row["score"],
                "host_failures":row["failure_classes"],"model_errors":row["model_errors"],
                "evidence_state":reported.get("state"),"failure_reported":reported.get("failure_types"),
                "failure_reported_is_not_verified":True})
        output=self.model.complete("develop",{"source_files":program["files"],"decision":decision,
             "experience":experience,"feedback":{"role":"D_fit","score":result["score"],"examples":examples},
             "edit_boundary":"Change reusable module behavior, do not embed examples/answers. Return complete changed files."})
        _proposal_files(program,decision,output)
        return output


def experience_card(result, parent, *, operator, module, step, mechanism):
    per=result["per_question"]
    paired={qid:score-parent["per_question"][qid] for qid,score in per.items()} if parent else {}
    failures=sorted(set(f for row in result["rows"] for f in row["failure_classes"]))
    if any(x<1 for x in per.values()): failures.append("answer_quality")
    return {"node_id":result["node_id"],"program_id":result["program_id"],"role":"D_fit",
      "panel_hash":result["panel_hash"],"evaluator_epoch":result["evaluator_epoch"],
      "complete":True,"valid_program":result["valid_program"],"score":result["score"],
      "signed_delta_vs_best_parent":result["score"]-parent["score"] if parent else None,
      "signed_deltas":{parent["node_id"]:result["score"]-parent["score"]} if parent else {},
      "parent_node_ids":[parent["node_id"]] if parent else [],"operator":operator,"target_module":module,
      "step":step,"hypothesis":mechanism,"failure_classes":failures,
      "failure_assessment_source":"host_execution_and_fit_answer_scores",
      "paired_deltas":paired,"resource_usage":result["resource_usage"],
      "behavior":{"group_hash":digest([r["answer"] for r in result["rows"]])},
      "reward":{"answer_quality":result["score"],"paired_signed_gain":sum(paired.values())/len(paired) if paired else None,
           "delivery_rate":sum(r["answer_usable"] for r in result["rows"])/len(result["rows"]),
           "source_valid_rate":sum(r["citation_source_valid"] for r in result["rows"])/len(result["rows"]),
           "proxy_added_to_terminal_quality":False}}


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
        self._frozen_manifest=deepcopy(self._snapshot())
        freeze(self.directory/"manifest.json",self._frozen_manifest)
        self.archive=ProgramArchive(self.directory/"archive")
        self.measure=Measurement(self.archive,self.directory/"measurements",model_factory,backend_factory,
             metric=manifest["metric"],scorer=scorer,limits=manifest.get("limits"))

    def _snapshot(self):
        return {**deepcopy(self.manifest),
                "public_panel_hashes":{r:digest(ts) for r,ts in self.panels.items()},
                "private_reference_hashes":{r:digest(ref) for r,ref in self.references.items()},
                "runtime_source_hashes":_runtime_source_hashes()}

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
        results={}; cards=[]
        baseline=self._measure(root,"D_fit"); results[root["node_id"]]=baseline
        cards.append(experience_card(baseline,None,operator="Draft",module="retrieval",step=0,mechanism="frozen root"))
        for step in range(self.manifest["expansions"]):
            folder=self.directory/"steps"/str(step)
            decision=choose_next(cards,step=step,panel_hash=baseline["panel_hash"],evaluator_epoch=self.measure.epoch)
            if decision["parent_node_id"] is None:
                decision["parent_node_id"]=root["node_id"]
            decision["recent_rejections"]=[r for i in range(max(0,step-4),step)
                      if (r:=read(self.directory/"steps"/str(i)/"rejected.json"))]
            freeze(folder/"decision.json",decision)
            parent=self.archive.load_node(decision["parent_node_id"])
            program=self.archive.load_program(parent["program_id"])
            proposal=read(folder/"proposal.json")
            child=read(folder/"child.json")
            rejected=read(folder/"rejected.json")
            if rejected:
                if proposal is not None or child is not None:
                    raise ValueError("rejected attempt also has a proposal or child receipt")
                continue
            if proposal is None:
                if child is not None:
                    raise ValueError("child receipt has no proposal")
                try:
                    proposal=deepcopy(self.developer.propose(deepcopy(program),deepcopy(decision),
                      memory_for_action(cards,decision),deepcopy(results[parent["node_id"]]),
                      deepcopy(self.panels["D_fit"]),deepcopy(self.references["D_fit"])))
                    self._assert_frozen()
                    proposed=_proposal_files(program,decision,proposal)
                except (ValueError,SyntaxError) as exc:
                    self._assert_frozen()
                    save(folder/"rejected.json",{"reason":str(exc),"next_step_allowed":True})
                    continue
                if any(self.archive.load_program(c["program_id"])["files"]==proposed for c in cards):
                    save(folder/"rejected.json",{"reason":"previously visited program","next_step_allowed":True})
                    continue
                save(folder/"proposal.json",proposal)
            else:
                # A saved proposal is an input to recovery, not an exemption from validation.
                proposed=_proposal_files(program,decision,proposal)
                if any(self.archive.load_program(c["program_id"])["files"]==proposed for c in cards):
                    raise ValueError("replayed proposal repeats a previously visited program")
            expected={"session_id":"v3-evolution","attempt":step,"parent_node_id":parent["node_id"]}
            if child is None:
                child=recoverable_record(self.archive,proposed,program["metadata"],**expected)
                save(folder/"child.json",child)
            else:
                child=_validate_receipt(self.archive,child,proposed,program["metadata"],**expected)
            measured=self._measure(child,"D_fit"); results[child["node_id"]]=measured
            card=experience_card(measured,results[parent["node_id"]],operator=decision["operator"],module=decision["target_module"],step=step+1,mechanism=proposal["mechanism"])
            cards.append(card); save(folder/"experience.json",card)
        # One immutable search freeze, one selection, one report. No shadow judging.
        freeze(self.directory/"search_frozen.json",{"cards":cards,"order":[c["node_id"] for c in cards]})
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
        report={"schema":"rag-rsi-v3-report-1","status":"complete","metric":self.manifest["metric"],
            "proxy_metric":bool(self.manifest.get("allow_proxy_metric")),"fit_nodes":len(cards),
            "delivery_lock":lock,"report_score":delivered["score"],"reference_score":reference["score"],
            "paired_report_gain":delivered["score"]-reference["score"],
            "paired_question_deltas":{q:score-reference["per_question"][q] for q,score in delivered["per_question"].items()},
            "independent_unit":"question, not repeat","report_used_for_decisions":False,
            "synthetic":bool(self.manifest.get("synthetic",False)),
            "quality_claim":"engineering_fixture_only" if self.manifest.get("synthetic") else "estimate_on_frozen_report_panel"}
        save(self.directory/"report.json",report)
        return report
