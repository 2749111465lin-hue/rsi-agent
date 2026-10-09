"""Synthetic final-answer probe orchestration, recovery and reference gates."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from code_rsi import answer_probe as probe
from code_rsi.answer_replay import ARMS, project_final_payload
from code_rsi.budget import digest, save, stable
from code_rsi.v3.datasets import adapt_musique
from code_rsi.v3.evolution import _runtime_source_hashes
from code_rsi.v3.execution import HostBroker, EXECUTION_SCHEMA, HostError
from code_rsi.v3.infrastructure import UnknownProviderOutcome
from code_rsi.v3.paired_analysis import SINGLE_CONTRAST_SCHEMA
from code_rsi.v3.rag import RagEngine
from test_v3_reader_probe import fixture_case as reader_fixture, ScriptedTransport, TEXT


def fixture_case(index=0):
    case = reader_fixture("state" + str(index))
    task, reference = adapt_musique({"id": "synthetic-answer-" + str(index),
        "question": case["task"]["question"], "answer": "Beacon Port", "answerable": True,
        "paragraphs": [{"idx": 0, "title": "", "paragraph_text": TEXT, "is_supporting": True}]})
    case["task"] = task
    def remap(value):
        if isinstance(value, dict):
            return {key: task["documents"][0]["docid"] if key == "docid" and item == "doc1"
                    else remap(item) for key, item in value.items()}
        if isinstance(value, list):
            return [remap(item) for item in value]
        return value
    case["events"] = remap(case["events"])
    for event in case["events"]:
        event["response_sha256"] = digest(event["response"])
    case["original_final_payload_sha256"] = digest(case["events"][-1]["request"]["payload"])
    return case, reference


class TrustedAnswerExecutor:
    """Use the trusted engine plus broker; never execute archived candidate code."""
    def __init__(self, *, mismatch_answer=False):
        self.calls = []
        self.mismatch_answer = mismatch_answer

    def __call__(self, archive, node_id, task, backend, model, directory, *, limits):
        self.calls.append((node_id, task["question_id"]))
        broker = HostBroker(task, backend, model, **limits)
        class Backend:
            def search(self, query, limit=5):
                return broker("search", {"query": query, "limit": limit})
        class Model:
            def complete(self, stage, payload):
                if stage == "answer":
                    payload = project_final_payload(payload, model.arm)
                return broker("complete", {"stage": stage, "payload": payload})
        result = RagEngine(Backend(), Model(), config=model.case["config"]).solve(
            {"question": task["question"], "task_id": task["question_id"]})
        if broker.fatal is not None:
            raise broker.fatal
        answer = "forged postprocessing" if self.mismatch_answer else result["answer"] or ""
        origin = broker.answer_origin_receipt(answer)
        node = archive.load_node(node_id)
        return {"schema": EXECUTION_SCHEMA, "node_id": node_id, "program_id": node["program_id"],
                "question_id": task["question_id"], "answer": answer,
                "answer_usable": bool(answer.strip()), "execution_ok": True,
                "citation_source_valid": False, "citation_status": "fixture-no-answer-citations",
                "answer_origin_valid": origin["valid"], "answer_origin_status": origin["status"],
                "host_answer_origin_validation": origin,
                "failure_classes": [] if origin["valid"] else ["invalid_answer_origin"],
                "model_errors": broker.model_errors, "trace": broker.events,
                "host_evidence_trace": {"read_presentations": broker.read_presentations,
                                        "final_observations": broker.final_observations},
                "candidate_reported": result, "resource_usage": broker.counts}


class AnswerProbeTests(unittest.TestCase):
    def setUp(self):
        runs = Path(__file__).parent / "runs"
        runs.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="synthetic-answer-probe-", dir=runs)
        self.root = Path(temporary.name)
        self.addCleanup(temporary.cleanup)
        guard = patch.object(probe, "credential_from_plan", side_effect=AssertionError("credential read"))
        guard.start()
        self.addCleanup(guard.stop)

    def plan(self, *, questions=1, repeats=1, opaque=False):
        pairs = [fixture_case(index) for index in range(questions)]
        cases = [case for case, _ in pairs]
        save(self.root / "cases.json", {"schema": "rag-rsi-reader-replay-cases-1", "cases": cases})
        if opaque:
            (self.root / "refs.json").write_bytes(b"opaque references must not be parsed")
        else:
            save(self.root / "refs.json", {ref["question_id"]: ref for _, ref in pairs})
        def bind(name):
            path = self.root / name
            return {"path": str(path), "sha256": probe.file_hash(path)}
        return {"schema": probe.SCHEMA, "purpose": "fixed_evidence_final_judgment_ablation",
                "question_use": "synthetic", "source_plan_file": None,
                "cases_file": bind("cases.json"), "references_file": bind("refs.json"),
                "output_dir": str(self.root / "output"), "arms": [{"name": name} for name in ARMS],
                "repeats": repeats, "schedule_seed": 42,
                "analysis": {"schema": SINGLE_CONTRAST_SCHEMA, "primary_metric": "answer_f1",
                    "comparisons": [{"name": "remove_derived_judgments", "baseline": "full_state", "candidate": "evidence_only"}],
                    "question_groups": {c["task"]["question_id"]: "g" + str(i) for i, c in enumerate(cases)},
                    "confidence_level": .95, "bootstrap_samples": 1000, "bootstrap_seed": 83,
                    "target_effect": .1, "power": .8},
                "model": {"name": "deepseek-flash", "temperature": 0, "thinking": "disabled",
                          "max_input_bytes": 120000, "output_limits": {"answer": 800},
                          "prices": {"input_miss": 2, "input_hit": .04, "output": 8}},
                "hard_cny": 50, "max_calls": questions * repeats * 2,
                "runtime_source_hashes": _runtime_source_hashes(), "helper_source_hashes": probe.helper_hashes(),
                "credential_source": {"kind": "env_file", "path": str(self.root / "absent-credentials"),
                                      "variable": "DEEPSEEK_API_KEY"}}

    def generate(self, plan, executor=None, transport=None):
        executor = executor or TrustedAnswerExecutor()
        transport = transport or ScriptedTransport()
        frozen = probe.generate(plan, approved_plan_hash=digest(plan), executor=executor, transport=transport)
        return frozen, executor, transport

    def reject_before_private(self, plan):
        original = probe._verified_bytes
        opened = []
        def guarded(item):
            if item == plan["references_file"]:
                opened.append(item)
                raise AssertionError("private file reached before complete validated freeze")
            return original(item)
        with patch.object(probe, "_verified_bytes", side_effect=guarded), self.assertRaises((ValueError, HostError)):
            probe.grade(plan)
        self.assertEqual(opened, [])

    def rebind(self, plan, cell, record, frozen):
        path = Path(plan["output_dir"]) / cell["file"]
        record["payload_hash"] = digest(record["payload"])
        save(path, record)
        cell["sha256"] = probe.file_hash(path)
        save(Path(plan["output_dir"]) / "generation_freeze.json", frozen)

    def test_preflight_does_not_parse_references_credentials_or_construct_transport(self):
        plan = self.plan(questions=2, repeats=2, opaque=True)
        original = probe._verified_bytes
        def checked(item):
            self.assertEqual(item, plan["cases_file"])
            return original(item)
        with patch.object(probe, "_verified_bytes", side_effect=checked), \
                patch.object(probe, "deepseek_transport", side_effect=AssertionError("transport construction")):
            check = probe.preflight(plan)
        self.assertEqual((check["states"], check["outcomes"], check["max_new_calls"]), (2, 8, 8))
        self.assertEqual(check["new_search_calls"], 0)
        self.assertEqual(check["new_read_model_calls"], 0)
        self.assertFalse(check["reference_or_credential_access"])
        self.assertFalse(Path(plan["output_dir"]).exists())

    def test_budget_binds_exact_complete_request_body_and_output_envelope(self):
        plan = self.plan(repeats=2)
        check = probe.preflight(plan)
        case = probe._snapshot(plan)[0]
        original = case["events"][-1]["request"]["payload"]
        shape = probe._RequestShape(plan["model"])
        bound = 0
        for arm in ARMS:
            body = shape.request_body("answer", project_final_payload(original, arm))
            size = len(stable(body).encode("utf-8"))
            self.assertEqual(check["answer_body_bytes"][case["case_id"]][arm], size)
            self.assertEqual(check["answer_body_hashes"][case["case_id"]][arm], digest(body))
            bound += 2 * ((size + 1024) * 2 + 800 * 8) / 1e6
        self.assertAlmostEqual(check["conservative_cny_upper_bound"], bound)
        plan["hard_cny"] = bound / 2
        with self.assertRaises(ValueError):
            probe.preflight(plan)

    def test_preflight_rejects_changed_model_helpers_arms_limits_and_analysis(self):
        base = self.plan()
        mutations = [lambda p: p["model"].update(temperature=.2), lambda p: p["model"].update(thinking="enabled"),
                     lambda p: p["model"]["output_limits"].update(answer=801),
                     lambda p: p["arms"][1].update(guidance="unreviewed"), lambda p: p.update(repeats=True),
                     lambda p: p.update(max_calls=3), lambda p: p.update(hard_cny=float("nan")),
                     lambda p: p.update(runtime_source_hashes={}), lambda p: p.update(helper_source_hashes={}),
                     lambda p: p["analysis"].update(primary_metric="answer_em"),
                     lambda p: p.update(source_plan_file=p["cases_file"]),
                     lambda p: p.update(output_dir=str(self.root.parent.parent / "outside-runs"))]
        for i, mutate in enumerate(mutations):
            changed = deepcopy(base)
            mutate(changed)
            with self.subTest(i=i), self.assertRaises(ValueError):
                probe.preflight(changed)

    def test_changed_case_bytes_and_duplicate_question_are_rejected(self):
        plan = self.plan()
        path = Path(plan["cases_file"]["path"])
        packet = json.loads(path.read_text())
        other = deepcopy(packet["cases"][0])
        other["case_id"] = "other-state-same-question"
        packet["cases"].append(other)
        save(path, packet)
        with self.assertRaises(ValueError):
            probe.preflight(plan)
        plan["cases_file"]["sha256"] = probe.file_hash(path)
        with self.assertRaises(ValueError):
            probe.preflight(plan)

    def test_approval_binds_full_plan_before_output_or_transport(self):
        plan = self.plan()
        executor, transport = TrustedAnswerExecutor(), ScriptedTransport()
        with self.assertRaises(ValueError):
            probe.generate(plan, approved_plan_hash="different", executor=executor, transport=transport)
        self.assertEqual(executor.calls, [])
        self.assertEqual(transport.sent, [])
        self.assertFalse(Path(plan["output_dir"]).exists())

    def test_all_prefixes_are_replayed_and_every_arm_repeat_buys_one_fresh_answer(self):
        plan = self.plan(questions=2, repeats=2, opaque=True)
        frozen, executor, transport = self.generate(plan)
        self.assertEqual(len(frozen["cells"]), 8)
        self.assertEqual(len(executor.calls), 8)
        self.assertEqual(len(transport.sent), 8)
        self.assertFalse(frozen["references_parsed"])
        self.assertTrue(all(body["max_tokens"] == 800 for body in transport.sent))
        self.assertTrue(all("sources" not in json.loads(body["messages"][-1]["content"]) for body in transport.sent))
        records = [json.loads(path.read_text()) for path in (Path(plan["output_dir"]) / "requests").glob("*.json")
                   if path.name != "returned_model.json"]
        self.assertEqual(len({record["key"] for record in records}), 8)
        self.assertLess(len({digest(record["body"]) for record in records}), 8)
        for cell in frozen["cells"]:
            receipt = json.loads((Path(plan["output_dir"]) / cell["file"]).read_text())["payload"]
            self.assertEqual(receipt["answer"], "Beacon Port")
            self.assertEqual(len(receipt["host_evidence_trace"]["read_presentations"]), 2)

    def test_complete_resume_neither_dispatches_nor_reads_credentials(self):
        plan = self.plan(opaque=True, repeats=2)
        frozen, executor, transport = self.generate(plan)
        with patch.object(probe, "deepseek_transport", side_effect=AssertionError("transport construction")):
            second = probe.generate(plan, approved_plan_hash=digest(plan), executor=executor)
        self.assertEqual(second, frozen)
        self.assertEqual(len(executor.calls), 4)
        self.assertEqual(len(transport.sent), 4)

    def test_unknown_physical_result_stops_without_freeze_zero_score_or_retry(self):
        plan = self.plan()
        sent = []
        def unknown(body):
            sent.append(body)
            raise TimeoutError("unknown physical result")
        with self.assertRaises(UnknownProviderOutcome):
            self.generate(plan, transport=unknown)
        with patch.object(probe, "deepseek_transport", side_effect=AssertionError("transport construction")), \
                self.assertRaises(UnknownProviderOutcome):
            probe.generate(plan, approved_plan_hash=digest(plan), executor=TrustedAnswerExecutor())
        self.assertEqual(len(sent), 1)
        self.assertFalse((Path(plan["output_dir"]) / "generation_freeze.json").exists())
        self.assertFalse((Path(plan["output_dir"]) / "report.json").exists())

    def test_interruption_after_cells_resumes_without_repurchasing(self):
        plan = self.plan()
        executor, transport = TrustedAnswerExecutor(), ScriptedTransport()
        original = probe.freeze
        def interrupt(path, value):
            if Path(path).name == "generation_freeze.json":
                raise OSError("synthetic freeze interruption")
            return original(path, value)
        with patch.object(probe, "freeze", side_effect=interrupt), self.assertRaises(OSError):
            self.generate(plan, executor, transport)
        self.assertEqual(len(transport.sent), 2)
        self.reject_before_private(plan)
        frozen, _, _ = self.generate(plan, executor, transport)
        self.assertEqual(len(frozen["cells"]), 2)
        self.assertEqual(len(transport.sent), 2)
        self.assertEqual(len(executor.calls), 2)

    def test_no_generation_freeze_means_no_reference_read(self):
        self.reject_before_private(self.plan())

    def test_grade_missing_duplicate_foreign_and_boolean_identity_precede_reference_access(self):
        plan = self.plan()
        frozen, _, _ = self.generate(plan)
        variants = [[], frozen["cells"][:-1], [frozen["cells"][0], frozen["cells"][0]]]
        foreign = deepcopy(frozen["cells"])
        foreign[0]["identity"]["arm"] = "foreign"
        variants.append(foreign)
        boolean = deepcopy(frozen["cells"])
        boolean[0]["identity"]["repeat"] = False
        variants.append(boolean)
        for cells in variants:
            save(Path(plan["output_dir"]) / "generation_freeze.json", {**frozen, "cells": cells})
            self.reject_before_private(plan)

    def test_rehashed_read_trace_tampering_is_rejected_before_references(self):
        plan = self.plan()
        frozen, _, _ = self.generate(plan)
        cell = frozen["cells"][0]
        record = json.loads((Path(plan["output_dir"]) / cell["file"]).read_text())
        record["payload"]["host_evidence_trace"]["read_presentations"][-1]["verified_quotes"] = []
        self.rebind(plan, cell, record, frozen)
        self.reject_before_private(plan)

    def test_rehashed_final_input_tampering_is_rejected_before_references(self):
        plan = self.plan()
        frozen, _, _ = self.generate(plan)
        cell = frozen["cells"][0]
        record = json.loads((Path(plan["output_dir"]) / cell["file"]).read_text())
        event = next(row for row in reversed(record["payload"]["trace"]) if row["name"] == "complete")
        event["request"]["payload"]["additional_guidance"] = "unapproved assertion"
        event["payload_sha256"] = digest(event["request"]["payload"])
        self.rebind(plan, cell, record, frozen)
        self.reject_before_private(plan)

    def test_source_drift_blocks_completed_report_before_reference_access(self):
        plan = self.plan()
        self.generate(plan)
        with patch.object(probe, "helper_hashes", return_value={}):
            self.reject_before_private(plan)
        with patch.object(probe, "_runtime_source_hashes", return_value={}):
            self.reject_before_private(plan)

    def test_grade_uses_local_official_metrics_only_after_complete_freeze(self):
        plan = self.plan(questions=2, repeats=2)
        self.generate(plan)
        original = probe._verified_bytes
        seen = []
        def checked(item):
            if item == plan["references_file"]:
                self.assertTrue((Path(plan["output_dir"]) / "generation_freeze.json").exists())
                seen.append(True)
            return original(item)
        with patch.object(probe, "_verified_bytes", side_effect=checked):
            report = probe.grade(plan)
        self.assertEqual(seen, [True])
        self.assertEqual(report["status"], "local_diagnostic_complete")
        self.assertEqual(len(report["rows"]), 8)
        self.assertTrue(report["references_parsed_after_complete_generation"])
        self.assertFalse(report["independent_quality_evidence"])
        for arm in ARMS:
            self.assertEqual(report["summary"][arm]["outcomes"], 4)
            self.assertEqual(report["summary"][arm]["mean_em"], 1)
            self.assertEqual(report["summary"][arm]["mean_f1"], 1)
        self.assertEqual(probe.grade(plan), report)

    def test_reference_panel_may_not_drop_question_after_freeze(self):
        plan = self.plan(questions=2)
        path = Path(plan["references_file"]["path"])
        refs = json.loads(path.read_text())
        del refs[next(iter(refs))]
        save(path, refs)
        plan["references_file"]["sha256"] = probe.file_hash(path)
        self.generate(plan)
        with self.assertRaises(ValueError):
            probe.grade(plan)
        self.assertFalse((Path(plan["output_dir"]) / "report.json").exists())

    def test_ineligible_origin_has_unknown_quality_and_no_successful_subset_means(self):
        plan = self.plan()
        self.generate(plan, executor=TrustedAnswerExecutor(mismatch_answer=True))
        report = probe.grade(plan)
        self.assertEqual(report["status"], "protocol_invalid")
        self.assertFalse(report["paired_analysis"]["quality_comparison_valid"])
        self.assertIsNone(report["paired_analysis"]["qualified"])
        self.assertEqual(len(report["rows"]), 2)
        for row in report["rows"]:
            self.assertFalse(row["program_eligible"])
            self.assertIsNone(row["metrics"])
        for arm in ARMS:
            self.assertIsNone(report["summary"][arm]["mean_em"])
            self.assertIsNone(report["summary"][arm]["mean_f1"])

    def test_completed_truncation_is_retained_without_repurchase_or_zero_imputation(self):
        plan = self.plan()
        scripted = ScriptedTransport()
        def truncate(body):
            response = scripted(body)
            response["choices"][0]["finish_reason"] = "length"
            return response
        self.generate(plan, transport=truncate)
        report = probe.grade(plan)
        self.assertEqual(len(scripted.sent), 2)
        self.assertEqual(report["status"], "protocol_invalid")
        for row in report["rows"]:
            self.assertIn("ModelResponseError", row["model_errors"])
            self.assertIsNone(row["metrics"])
        probe.generate(plan, approved_plan_hash=digest(plan), executor=TrustedAnswerExecutor())
        self.assertEqual(len(scripted.sent), 2)


if __name__ == "__main__":
    unittest.main()
