"""Synthetic MuSiQue calibration integration, with no network or candidate exec.

Trusted RagEngine/HostBroker exercise actual read/answer receipts. The scripted
transport proves accounting and coupling mechanics, not LLM quality or WSL.
"""
import ast
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from code_rsi.budget import digest, save
from code_rsi.v3 import calibration as cal, musique_calibration as musique
from code_rsi.v3.execution import EXECUTION_SCHEMA, HostBroker, root_files
from code_rsi.v3.paired_analysis import SINGLE_CONTRAST_SCHEMA
from code_rsi.v3.rag import RagEngine
from code_rsi import prepare_musique as pm
import test_v3_calibration as base
from test_v3_musique_runtime_input import fixture


class ScriptedTransport:
    def __init__(self):
        self.calls = []

    def __call__(self, body):
        payload = json.loads(body["messages"][-1]["content"])
        self.calls.append(deepcopy(payload))
        if "sources" in payload:
            sources = payload["sources"]
            result = {"claims": [{"text": "A synthetic public fact.",
                       "citations": [{"source_id": sources[0]["source_id"], "quote": sources[0]["text"]}]}] if sources else [],
                      "bridge_entities": [], "gaps": [], "conflicts": [], "queries": [], "ready": True}
        elif "evidence" in payload:
            result = {"answer": "Synthetic Port", "citation_ids": [e["citation_id"] for e in payload["evidence"]],
                      "evidence_sufficient": bool(payload["evidence"])}
        else:
            result = {"constraints": [], "queries": ["Public"]}
        return {"model": "synthetic-frozen-model", "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}]}


class TrustedEngineExecutor:
    """Interpret only root CONFIG as data; never import or execute archived code."""
    def __init__(self, *, partial_support=False):
        self.calls, self.partial_support = [], partial_support

    def __call__(self, archive, node_id, task, backend, model, directory, **kwargs):
        self.calls.append({"question_id": task["question_id"], "scope": backend.scope,
                           "docids": list(backend.docs), "support": isinstance(backend, musique.SupportCorpus)})
        node = archive.load_node(node_id)
        files = archive.load_program(node["program_id"])["files"]
        assignment = next(n for n in ast.parse(files["rag.py"]).body
                          if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in n.targets))
        config = json.loads(ast.literal_eval(assignment.value.args[0]))
        if files != root_files(config):
            raise AssertionError("this test executes only the imported trusted root engine")
        supplied = backend
        if self.partial_support and isinstance(backend, musique.SupportCorpus):
            class TruncatedMaterial:
                def search(self, query, limit):
                    rows = deepcopy(supplied.search(query, limit))
                    rows[0]["text"] = rows[0]["text"][:-1]
                    rows[0]["end"] -= 1
                    return rows
            backend = TruncatedMaterial()
        broker = HostBroker(task, backend, model, **kwargs.get("limits", {}))

        class Search:
            def search(self, query, limit):
                return broker("search", {"query": query, "limit": limit})

        class Model:
            def complete(self, stage, payload):
                return broker("complete", {"stage": stage, "payload": payload})

        result = RagEngine(Search(), Model(), config=config).solve({"question": task["question"]})
        answer = result["answer"] or ""
        citations = [c for c in result["state"]["citations"] if c["citation_id"] in result["citation_ids"]]
        origin = broker.answer_origin_receipt(answer)
        cited = broker.citation_receipt(answer, citations)
        return {"schema": EXECUTION_SCHEMA, "node_id": node_id, "program_id": node["program_id"],
                "question_id": task["question_id"], "answer": answer, "answer_usable": bool(answer.strip()),
                "execution_ok": True, "citation_source_valid": cited["valid"], "citation_status": cited["status"],
                "host_citation_validation": cited, "citations": citations,
                "answer_origin_valid": origin["valid"], "answer_origin_status": origin["status"],
                "host_answer_origin_validation": origin, "failure_classes": result["failure_types"],
                "model_errors": broker.model_errors, "trace": broker.events, "resource_usage": broker.counts,
                "host_evidence_trace": {"read_presentations": broker.read_presentations,
                                        "final_observations": broker.final_observations},
                "candidate_reported": result}


class MuSiQueCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.base = base.CalibrationTests(methodName="runTest")
        self.addCleanup(self.base.doCleanups)
        self.base.setUp()

    def bind(self, name):
        path = self.base.root / name
        return {"path": str(path), "sha256": cal.file_hash(path)}

    def plan(self, *, questions=2, repeats=2, opaque_references=False):
        plan = self.base.plan(questions=questions, repeats=repeats)
        tasks, refs = [], {}
        for i in range(questions):
            row = fixture("calibration" + str(i), 3 + i)
            row["answer"] = "Synthetic Port"
            task, reference = pm._adapt(row)
            tasks.append(task)
            refs[task["question_id"]] = reference
        save(self.base.root / "musique_tasks.json", tasks)
        save(self.base.root / "musique_refs.json", refs)
        save(self.base.root / "musique_support.json", musique.build_support_materials(tasks, refs))
        if opaque_references:
            (self.base.root / "musique_refs.json").write_text("OPAQUE_NOT_JSON", encoding="utf-8")
        common = {k: v for k, v in plan["arms"][0]["config"].items() if k != "mode"}
        plan.update(schema=musique.SCHEMA, purpose="musique_mechanism_calibration", data_role="D_fit",
                    tasks_file=self.bind("musique_tasks.json"), references_file=self.bind("musique_refs.json"),
                    support_materials_file=self.bind("musique_support.json"), corpus=None, corpus_ref=None,
                    question_ids=[t["question_id"] for t in tasks], question_use="synthetic",
                    request_coupling=cal.REQUEST_COUPLING,
                    arms=[{"name": name, "config": {**common, "mode": mode}} for name, mode in musique.ARMS.items()],
                    max_calls=questions * repeats * 11, hard_cny=50,
                    analysis={"schema": SINGLE_CONTRAST_SCHEMA, "primary_metric": "answer_f1",
                              "comparisons": [{"name": "iteration", "baseline": "planned", "candidate": "loop"}],
                              "question_groups": {t["question_id"]: "shared" if i < 2 else "other" for i, t in enumerate(tasks)},
                              "confidence_level": .95, "bootstrap_samples": 1000, "bootstrap_seed": 83,
                              "target_effect": .1, "power": .8})
        return plan

    def generate(self, plan, executor=None, transport=base.no_transport):
        executor = base.FakeExecutor() if executor is None else executor
        return cal.generate(plan, approved_plan_hash=digest(plan), transport=transport, executor=executor), executor

    def test_preflight_and_generation_do_not_parse_private_answers(self):
        plan = self.plan(opaque_references=True)
        original = cal._verified_bytes
        def public_only(item):
            if item == plan["references_file"]:
                self.fail("private answer map parsed before generation freeze")
            return original(item)
        with patch.object(cal, "_verified_bytes", side_effect=public_only):
            preflight = cal.preflight(plan)
            frozen, executor = self.generate(plan)
        self.assertEqual(preflight["max_calls"], 44)
        self.assertEqual(preflight["answer_outcomes"], 12)
        self.assertEqual(len(executor.calls), 12)
        self.assertFalse(preflight["references_parsed"])
        self.assertFalse(preflight["credentials_read"])
        self.assertTrue(preflight["support_labels_used_for_diagnostic"])
        self.assertFalse(frozen["references_parsed_by_runner"])

    def test_true_engine_scoped_backends_prefix_and_full_support_diagnostic(self):
        plan = self.plan(questions=3)
        executor, transport = TrustedEngineExecutor(), ScriptedTransport()
        frozen, _ = self.generate(plan, executor, transport)
        report = cal.grade(plan)
        self.assertEqual(report["quality_comparison_scope"], "planned_and_loop_only")
        self.assertTrue(report["quality_comparison_valid"])
        self.assertTrue(report["mechanism_comparison_valid"])
        self.assertEqual(report["shared_prefix_diagnostic"]["verified_pairs"], 6)
        self.assertEqual(report["analysis"]["cell_count"], 12)
        self.assertEqual(report["analysis"]["cluster_count"], 2)
        self.assertEqual(set(report["analysis"]["qualified"]["arms"]), {"planned", "loop"})
        self.assertEqual(set(report["analysis"]["qualified"]["comparisons"]), {"iteration"})
        self.assertEqual(len(frozen["cells"]), 18)
        self.assertEqual(report["support_diagnostic"]["cell_count"], 6)
        self.assertTrue(report["support_diagnostic"]["qualified_material_diagnostic"])
        self.assertFalse(report["support_diagnostic"]["used_for_selection"])
        self.assertFalse(report["support_diagnostic"]["semantic_sufficiency_verified"])
        summary = report["arms"]["support"]["evidence_stages"]["full_support_presented"]
        self.assertEqual((summary["known_cells"], summary["total_cells"], summary["mean_support_document_recall"]), (6, 6, 1.))
        graded = json.loads((Path(plan["output_dir"]) / "grading/rows.json").read_text())
        self.assertEqual({r["evidence_stages"]["stages"]["full_support_presented"]["support_recall"]
                          for r in graded if r["arm"] == "support"}, {1.})
        for call in executor.calls:
            self.assertEqual(call["scope"], call["question_id"])
            self.assertTrue(all(docid.startswith(call["question_id"] + "/p/") for docid in call["docids"]))
            self.assertEqual(len(call["docids"]), 2 if call["support"] else 3 + plan["question_ids"].index(call["question_id"]))
        self.assertNotIn("PRIVATE_", json.dumps(transport.calls))
        calls = (len(executor.calls), len(transport.calls))
        resumed, _ = self.generate(plan, executor, transport)
        self.assertEqual(resumed, frozen)
        self.assertEqual((len(executor.calls), len(transport.calls)), calls)
        self.assertEqual(cal.grade(plan), report)

    def test_fake_executor_missing_prefix_stays_unknown_and_does_not_certify_oracle(self):
        plan = self.plan()
        fake, incomplete_nodes = base.FakeExecutor(), set()
        def executor(archive, node_id, task, backend, model, directory, **kwargs):
            receipt = fake(archive, node_id, task, backend, model, directory, **kwargs)
            if node_id not in incomplete_nodes:
                # One missing search log in each arm; other cells observed zero
                # searches. The unknown cell must not be dropped or filled zero.
                incomplete_nodes.add(node_id)
                receipt["resource_usage"]["search_calls"] = 1
            return receipt
        self.generate(plan, executor)
        report = cal.grade(plan)
        self.assertIsNot(report["mechanism_comparison_valid"], True)
        self.assertFalse(report["quality_comparison_valid"])
        self.assertIsNone(report["analysis"]["qualified"])
        self.assertEqual(report["status"], "protocol_invalid")
        self.assertEqual(report["shared_prefix_diagnostic"]["unknown_pairs"], 4)
        self.assertFalse(report["support_diagnostic"]["qualified_material_diagnostic"])
        self.assertEqual(report["support_diagnostic"]["cell_count"], 4)
        self.assertTrue(report["support_diagnostic"]["all_cells_retained"])
        self.assertEqual(report["analysis"]["cell_count"], 8)
        for arm in report["arms"].values():
            stage = arm["evidence_stages"]["retrieval"]
            self.assertEqual((stage["known_cells"], stage["total_cells"]), (3, 4))
            self.assertIsNone(stage["mean_support_document_recall"])
            self.assertFalse(stage["all_cells_complete"] or stage["successful_subset_analysis"])

    def test_truncated_oracle_material_invalidates_only_diagnostic_and_keeps_all_rows(self):
        plan = self.plan()
        self.generate(plan, TrustedEngineExecutor(partial_support=True), ScriptedTransport())
        report = cal.grade(plan)
        self.assertTrue(report["quality_comparison_valid"])
        self.assertTrue(report["mechanism_comparison_valid"])
        oracle = report["support_diagnostic"]
        self.assertFalse(oracle["qualified_material_diagnostic"])
        self.assertEqual(oracle["cell_count"], 4)
        self.assertTrue(all(c["fully_presented_documents"] == 1 for c in oracle["cells"]))
        self.assertFalse(oracle["successful_subset_analysis"])
        graded = json.loads((Path(plan["output_dir"]) / "grading/rows.json").read_text())
        self.assertEqual(len(graded), 12)
        summary = report["arms"]["support"]["evidence_stages"]["full_support_presented"]
        self.assertEqual((summary["known_cells"], summary["total_cells"], summary["mean_support_document_recall"]), (4, 4, .5))
        self.assertEqual({r["evidence_stages"]["stages"]["full_support_presented"]["support_recall"]
                          for r in graded if r["arm"] == "support"}, {.5})

    def test_missing_duplicate_or_foreign_backend_freeze_cannot_read_answers(self):
        plan = self.plan()
        frozen, executor = self.generate(plan)
        path = Path(plan["output_dir"]) / "generation_freeze.json"
        mutations = []
        missing = deepcopy(frozen); missing["cells"].pop(); mutations.append(missing)
        duplicate = deepcopy(frozen); duplicate["cells"][-1] = duplicate["cells"][0]; mutations.append(duplicate)
        foreign = deepcopy(frozen); foreign["cells"][0]["identity"]["backend"] = "foreign-backend"; mutations.append(foreign)
        for altered in mutations:
            save(path, altered)
            self.base.reject_grade_before_references(plan)
            with self.assertRaises(ValueError):
                self.generate(plan, executor)
        self.assertEqual(len(executor.calls), 12)

    def test_bound_reference_bytes_changed_after_generation_cannot_grade(self):
        plan = self.plan()
        self.generate(plan)
        Path(plan["references_file"]["path"]).write_text("CHANGED_ANSWERS", encoding="utf-8")
        self.base.reject_grade_before_references(plan)

    def test_material_task_binding_or_doc_identity_mismatch_blocks_preflight(self):
        plan = self.plan()
        path = Path(plan["support_materials_file"]["path"])
        original = json.loads(path.read_text())
        altered = deepcopy(original); altered["tasks_sha256"] = "wrong"
        foreign = deepcopy(original); foreign["rows"][0]["docids"][0] = original["rows"][1]["docids"][0]
        extra = deepcopy(original); extra["rows"][0]["answer"] = "PRIVATE_FORBIDDEN"
        for value in (altered, foreign, extra):
            save(path, value)
            plan["support_materials_file"] = self.bind("musique_support.json")
            with self.assertRaises(ValueError):
                cal.preflight(plan)

    def test_locally_valid_but_wrong_support_projection_rejected_after_generation(self):
        plan = self.plan()
        tasks = json.loads(Path(plan["tasks_file"]["path"]).read_text())
        materials = json.loads(Path(plan["support_materials_file"]["path"]).read_text())
        materials["rows"][0]["docids"] = [d["docid"] for d in tasks[0]["documents"][1:3]]
        save(self.base.root / "musique_support.json", materials)
        plan["support_materials_file"] = self.bind("musique_support.json")
        self.generate(plan)
        with self.assertRaisesRegex(ValueError, "support material differs"):
            cal.grade(plan)

    def test_no_old_schema_or_nonread_resource_changes_or_oracle_primary_comparison(self):
        original = self.plan()
        mutations = [lambda p: p.update(schema=cal.SCHEMA, purpose="used_development_calibration"),
                     lambda p: p.update(data_role="D_report"),
                     lambda p: p.update(corpus_ref="pooled-corpus"),
                     lambda p: p["arms"][2]["config"].update(max_queries_per_round=1),
                     lambda p: p["analysis"]["comparisons"][0].update(candidate="support"),
                     lambda p: p.update(max_calls=p["max_calls"] - 1)]
        for change in mutations:
            plan = deepcopy(original); change(plan)
            with self.assertRaises(ValueError):
                cal.preflight(plan)
        with self.assertRaisesRegex(ValueError, "per-question"):
            cal.generate(original, approved_plan_hash=digest(original), backend=base.FakeBackend(),
                         transport=base.no_transport, executor=base.FakeExecutor())


if __name__ == "__main__":
    unittest.main()
