"""Synthetic capture artifacts and provenance checks; no provider or source exec."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from code_rsi import program_answer_artifacts as artifacts
from code_rsi.archive import ProgramArchive
from code_rsi.budget import digest, save
from code_rsi.v3.execution import EXECUTION_SCHEMA, HostBroker, HostError, root_files
from test_v3_reader_replay import synthetic_case


class SyntheticProgramExecutor:
    """Read only a trusted CONFIG literal, then issue scripted host RPC calls."""
    def __init__(self, *, isolation_verified=False, changed_prefix=False, forged_answer=False):
        self.calls = []
        self.isolation_verified = isolation_verified
        self.changed_prefix = changed_prefix
        self.forged_answer = forged_answer

    def __call__(self, archive, node_id, task, backend, model, directory, *, limits):
        node = archive.load_node(node_id)
        files = archive.load_program(node["program_id"])["files"]
        module = ast.parse(files["rag.py"])
        assignment = next(n for n in module.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in n.targets))
        config = json.loads(assignment.value.args[0].value)
        self.calls.append((model.case["case_id"], node["program_id"]))
        broker = HostBroker(task, backend, model, **limits)
        for index, event in enumerate(model.case["events"][:-1]):
            request = deepcopy(event["request"])
            if self.changed_prefix and index == 0:
                request["payload"]["unexpected"] = True
            broker(event["name"], request)
        payload = deepcopy(model.case["events"][-1]["request"]["payload"])
        payload["additional_guidance"] = config.get("prompts", {}).get("answer", "")
        response = broker("complete", {"stage": "answer", "payload": payload})
        answer = "forged postprocessing" if self.forged_answer else response["answer"]
        returned = [deepcopy(e) for e in payload["evidence"] if e["citation_id"] in response["citation_ids"]]
        reported = {"synthetic_only": True}
        broker("record_trace", {"result": reported})
        origin, cited = broker.answer_origin_receipt(answer), broker.citation_receipt(answer, returned)
        return {"schema": EXECUTION_SCHEMA, "node_id": node_id, "program_id": node["program_id"],
                "question_id": task["question_id"], "answer": answer, "answer_usable": bool(answer.strip()),
                "execution_ok": True, "citation_source_valid": cited["valid"], "citation_status": cited["status"],
                "host_citation_validation": cited, "citations": returned,
                "answer_origin_valid": origin["valid"], "answer_origin_status": origin["status"],
                "host_answer_origin_validation": origin, "isolation_verified": self.isolation_verified,
                "failure_classes": [] if origin["valid"] else ["invalid_answer_origin"],
                "model_errors": broker.model_errors, "trace": broker.events,
                "host_evidence_trace": {"read_presentations": broker.read_presentations,
                                        "final_observations": broker.final_observations},
                "candidate_reported": reported, "resource_usage": broker.counts}


class ProgramArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="program-artifacts-", dir=Path(__file__).parent / "runs")
        self.root = Path(temporary.name)
        self.addCleanup(temporary.cleanup)
        self.case = synthetic_case()
        self.packet = {"schema": artifacts.PACKET_SCHEMA, "cases": [self.case]}
        self.model = {"name": "deepseek-flash", "temperature": 0, "thinking": "disabled",
                      "max_input_bytes": 120000, "output_limits": {"answer": 800},
                      "prices": {"input_miss": 2, "input_hit": .04, "output": 8}}
        self.bundle = {"schema": artifacts.PROGRAMS_SCHEMA, "source_plan_hash": None, "programs": [
            {"name": "reference", "files": root_files(self.case["config"]), "source": None},
            {"name": "candidate_1", "files": root_files({**self.case["config"],
                "prompts": {"answer": "Check the requested relation before answering."}}), "source": None}]}

    def bind(self, path):
        return {"path": str(path), "sha256": artifacts.file_hash(path)}

    def capture(self, *, executor=None, bundle=None, packet=None, suffix="captures_run"):
        return artifacts.capture_programs(packet or self.packet, bundle or self.bundle, self.model,
            self.root / suffix, executor=executor or SyntheticProgramExecutor())

    def plan(self, projections=None, *, bundle=None):
        save(self.root / "programs.json", bundle or self.bundle)
        save(self.root / "projections.json", self.capture() if projections is None else projections)
        return {"programs_file": self.bind(self.root / "programs.json"),
                "projections_file": self.bind(self.root / "projections.json"), "model": deepcopy(self.model),
                "question_use": "synthetic", "source_plan_file": None}

    def change_capture(self, plan, transform):
        projections = artifacts._bound_json(plan["projections_file"])
        item = projections["projections"][0]
        path = Path(item["capture_file"]["path"])
        record = json.loads(path.read_text(encoding="utf-8")); transform(record)
        save(path, record); item["capture_file"] = self.bind(path)
        save(self.root / "projections.json", projections)
        plan["projections_file"] = self.bind(self.root / "projections.json")

    def make_source(self):
        out = self.root / "paired"
        plan = {"schema": "rag-rsi-paired-development-2", "output_dir": str(out), "blocks": 1,
                "search_template": {"expansions": 2}}
        save(out / "paired_plan.json", plan)
        archives, roots, records = {}, {}, []
        for condition in ("cases", "trace"):
            directory = out / "blocks/0" / condition
            archive = ProgramArchive(directory / "archive"); archives[condition] = archive
            roots[condition] = archive.record(self.bundle["programs"][0]["files"], {}, session_id="root", attempt=0)
            save(directory / "root.json", roots[condition])
        schedule = [("cases", 0), ("trace", 0), ("trace", 1), ("cases", 1)]
        for index, (condition, slot) in enumerate(schedule):
            root = roots[condition]; directory = out / "blocks/0" / condition
            child = None
            if condition == "cases":
                files = root_files({**self.case["config"], "prompts": {"answer": f"Synthetic guidance {slot}."}})
                child = archives[condition].record(files, {}, session_id="children", attempt=slot,
                                                   parent_node_id=root["node_id"])
                save(directory / "steps" / str(slot) / "child.json", child)
            attempt = {"node_id": None if child is None else child["node_id"], "parent_node_id": root["node_id"],
                       "role": "D_fit", "step": slot, "status": "rejected" if child is None else "measured"}
            record = {"schedule_index": index, "block": 0, "condition": condition, "slot": slot, "attempt": attempt}
            save(directory / "steps" / str(slot) / "attempt.json", attempt)
            save(out / "schedule" / f"{index}.json", record); records.append(record)
        complete = {"schema": plan["schema"], "status": "complete", "plan_hash": digest(plan),
                    "heldout_roles_used": False, "completed_opportunities": 4, "planned_opportunities": 4,
                    "terminal_records": records}
        save(out / "paired_search_complete.json", complete)
        selected = artifacts.accepted_program_sources(plan)
        bundle = {"schema": artifacts.PROGRAMS_SCHEMA, "source_plan_hash": digest(plan), "programs": []}
        for item in selected:
            program = artifacts._archive(item["source"]["archive_dir"]).load_program(item["source"]["program_id"])
            bundle["programs"].append({**item, "files": program["files"]})
        return plan, bundle

    def test_capture_freezes_actual_payloads_and_resume_does_not_execute_again(self):
        executor = SyntheticProgramExecutor()
        projections = self.capture(executor=executor)
        self.assertEqual(len(executor.calls), 2)
        self.assertEqual(projections["cases_sha256"], digest(self.packet))
        self.assertEqual(projections["programs_sha256"], digest(self.bundle))
        self.assertEqual(projections["model_sha256"], digest(self.model))
        self.assertNotEqual(projections["projections"][0]["request_body_sha256"],
                            projections["projections"][1]["request_body_sha256"])
        for item in projections["projections"]:
            record = artifacts._bound_json(item["capture_file"])
            self.assertFalse(record["measurement_eligible"])
            self.assertTrue(record["final_response_replayed"])
            self.assertEqual(record["receipt"]["answer"], self.case["events"][-1]["response"]["answer"])
        def forbidden(*args, **kwargs): raise AssertionError("completed capture executed again")
        self.assertEqual(self.capture(executor=forbidden), projections)

    def test_validation_is_read_only_no_candidate_executor_or_references(self):
        plan = self.plan()
        plan["references_file"] = {"path": str(self.root / "must_not_read"), "sha256": "0"*64}
        with patch.object(artifacts, "execute", side_effect=AssertionError("execute")), \
             patch.object(artifacts, "capture_programs", side_effect=AssertionError("capture")):
            result = artifacts.validate_program_artifacts(plan, self.packet["cases"])
        self.assertEqual(list(result["files"]), ["reference", "candidate_1"])
        self.assertEqual(result["programs"], self.bundle["programs"])
        self.assertEqual(set(result["payloads"]), {(self.case["case_id"], name) for name in result["files"]})

    def test_two_cases_three_programs_retain_complete_order(self):
        packet = deepcopy(self.packet)
        other = deepcopy(self.case); other["case_id"] += "_two"
        other["task"]["question_id"] += "_two"
        other["task"]["id"] = other["task"]["question_id"]
        packet["cases"].append(other)
        bundle = deepcopy(self.bundle)
        bundle["programs"].append({"name": "candidate_2", "source": None,
            "files": root_files({**self.case["config"], "prompts": {"answer": "Second independent guidance."}})})
        projection = self.capture(packet=packet, bundle=bundle)
        self.assertEqual([(p["case_id"], p["arm"]) for p in projection["projections"]],
                         [(c["case_id"], p["name"]) for c in packet["cases"] for p in bundle["programs"]])

    def test_invalid_bundle_or_reference_is_rejected_before_output_creation(self):
        transforms = [lambda b: b["programs"].reverse(), lambda b: b["programs"].pop(),
                      lambda b: b["programs"][1].update(name="other"),
                      lambda b: b["programs"][0]["files"].update(extra="unexpected"),
                      lambda b: b["programs"][0]["files"].update({"rag.py": "changed"}),
                      lambda b: b["programs"][1].update(files=deepcopy(b["programs"][0]["files"])),
                      lambda b: b.update(source_plan_hash="0"*64)]
        for index, change in enumerate(transforms):
            bundle = deepcopy(self.bundle); change(bundle)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.capture(bundle=bundle, suffix=f"invalid_{index}")
            self.assertFalse((self.root / f"invalid_{index}").exists())

    def test_prefix_or_returned_answer_failure_cannot_publish_a_capture(self):
        for name, executor in (("prefix", SyntheticProgramExecutor(changed_prefix=True)),
                               ("origin", SyntheticProgramExecutor(forged_answer=True))):
            with self.subTest(name=name), self.assertRaises((ValueError, HostError)):
                self.capture(executor=executor, suffix=name)
            self.assertEqual(list((self.root / name).rglob("capture.json")), [])

    def test_projection_manifest_detects_model_case_or_bundle_change(self):
        plan = self.plan()
        for name in ("cases_sha256", "programs_sha256", "model_sha256"):
            original = artifacts._bound_json(plan["projections_file"])
            changed = deepcopy(original); changed[name] = "0"*64
            save(self.root / "changed.json", changed)
            candidate = {**plan, "projections_file": self.bind(self.root / "changed.json")}
            with self.subTest(name=name), self.assertRaises(ValueError):
                artifacts.validate_program_artifacts(candidate, self.packet["cases"])

    def test_projection_panel_cannot_drop_repeat_or_reorder_cells(self):
        plan = self.plan(); original = artifacts._bound_json(plan["projections_file"])
        for cells in (original["projections"][:1], list(reversed(original["projections"])),
                      [original["projections"][0]]*2):
            changed = {**original, "projections": cells}
            save(self.root / "changed.json", changed)
            candidate = {**plan, "projections_file": self.bind(self.root / "changed.json")}
            with self.assertRaises(ValueError): artifacts.validate_program_artifacts(candidate, self.packet["cases"])

    def test_capture_file_byte_tampering_is_rejected(self):
        plan = self.plan()
        path = Path(artifacts._bound_json(plan["projections_file"])["projections"][0]["capture_file"]["path"])
        path.write_text(path.read_text(encoding="utf-8")+" ", encoding="utf-8")
        with self.assertRaises(ValueError): artifacts.validate_program_artifacts(plan, self.packet["cases"])

    def test_rehashed_capture_cannot_claim_measurement_or_different_final_payload(self):
        transforms = [lambda r: r.update(measurement_eligible=True),
                      lambda r: r.update(final_response_replayed=False),
                      lambda r: r.update(files_sha256="0"*64),
                      lambda r: r.update(program_id="0"*64),
                      lambda r: r["final_payload"].update(additional_guidance="altered")]
        for index, change in enumerate(transforms):
            plan = self.plan(self.capture(suffix=f"source_{index}"))
            self.change_capture(plan, change)
            with self.subTest(index=index), self.assertRaises(ValueError):
                artifacts.validate_program_artifacts(plan, self.packet["cases"])

    def test_rehashed_receipt_upstream_event_or_resource_count_is_not_trusted(self):
        changes = [lambda r: r["receipt"]["trace"][0]["request"]["payload"].update(unexpected=True),
                   lambda r: r["receipt"]["resource_usage"].update(model_calls=99),
                   lambda r: r["receipt"]["host_evidence_trace"]["read_presentations"].pop()]
        for index, change in enumerate(changes):
            plan = self.plan(self.capture(suffix=f"trace_{index}")); self.change_capture(plan, change)
            with self.subTest(index=index), self.assertRaises(ValueError):
                artifacts.validate_program_artifacts(plan, self.packet["cases"])

    def test_projection_payload_and_body_hash_must_both_match_capture(self):
        plan = self.plan(); original = artifacts._bound_json(plan["projections_file"])
        for key in ("final_payload", "request_body_sha256"):
            changed = deepcopy(original)
            if key == "final_payload": changed["projections"][0][key]["additional_guidance"] = "changed"
            else: changed["projections"][0][key] = "0"*64
            save(self.root / "changed.json", changed)
            candidate = {**plan, "projections_file": self.bind(self.root / "changed.json")}
            with self.subTest(key=key), self.assertRaises(ValueError):
                artifacts.validate_program_artifacts(candidate, self.packet["cases"])

    def test_real_source_selects_all_measured_children_without_scores(self):
        source, bundle = self.make_source()
        selected = artifacts.accepted_program_sources(source)
        self.assertEqual([p["name"] for p in selected], ["reference", "candidate_1", "candidate_2"])
        self.assertEqual(selected, [{"name": p["name"], "source": p["source"]} for p in bundle["programs"]])
        self.assertTrue(all("score" not in repr(p) for p in selected))

    def test_real_bundle_requires_isolation_and_cannot_be_synthetic(self):
        source, bundle = self.make_source()
        with self.assertRaises(ValueError): self.capture(bundle=bundle, suffix="not_isolated")
        projections = self.capture(bundle=bundle, executor=SyntheticProgramExecutor(isolation_verified=True), suffix="isolated")
        plan = self.plan(projections, bundle=bundle)
        with self.assertRaises(ValueError): artifacts.validate_program_artifacts(plan, self.packet["cases"])
        save(self.root / "source_plan.json", source)
        plan.update(question_use="used_diagnostic", source_plan_file=self.bind(self.root / "source_plan.json"))
        self.assertEqual(len(artifacts.validate_program_artifacts(plan, self.packet["cases"])["programs"]), 3)

    def test_source_bundle_cannot_omit_an_accepted_measured_candidate(self):
        source, bundle = self.make_source(); bundle["programs"].pop()
        projections = self.capture(bundle=bundle, executor=SyntheticProgramExecutor(isolation_verified=True))
        plan = self.plan(projections, bundle=bundle)
        save(self.root / "source_plan.json", source)
        plan.update(question_use="used_diagnostic", source_plan_file=self.bind(self.root / "source_plan.json"))
        with self.assertRaises(ValueError): artifacts.validate_program_artifacts(plan, self.packet["cases"])

    def test_complete_report_must_match_source_step_and_frozen_schedule(self):
        source, _ = self.make_source(); out = Path(source["output_dir"])
        step = out / "blocks/0/cases/steps/0/attempt.json"
        original = json.loads(step.read_text(encoding="utf-8"))
        save(step, {**original, "status": "rejected"})
        with self.assertRaises(ValueError): artifacts.accepted_program_sources(source)
        save(step, original)
        report = json.loads((out / "paired_search_complete.json").read_text(encoding="utf-8"))
        report["terminal_records"].reverse(); save(out / "paired_search_complete.json", report)
        with self.assertRaises(ValueError): artifacts.accepted_program_sources(source)


if __name__ == "__main__":
    unittest.main()
