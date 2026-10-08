"""Synthetic calibration boundaries only: no credentials, API or real questions."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from code_rsi.archive import ProgramArchive
from code_rsi.budget import digest, save
from code_rsi.v3 import calibration as cal
from code_rsi.v3.evolution import _runtime_source_hashes
from code_rsi.v3.infrastructure import UnknownProviderOutcome, ModelResponseError
from code_rsi.v3.execution import HostBroker, EXECUTION_SCHEMA


class FakeBackend:
    identity = "synthetic-corpus-identity"

    def search(self, query, limit=5):
        raise AssertionError("fake executor must not search")


class FakeExecutor:
    def __init__(self, action=None, *, model_answer="Synthetic Port", returned_answer=None, model_error=None):
        self.calls, self.action = [], action
        self.model_answer, self.returned_answer, self.model_error = model_answer, returned_answer, model_error

    def __call__(self, archive, node_id, task, backend, model, directory, **kwargs):
        self.calls.append(copy.deepcopy(task))
        if self.action:
            self.action(task, model)
        node = archive.load_node(node_id)
        error, answer = self.model_error, self.model_answer
        class ScriptedAnswer:
            def complete(self, stage, payload):
                if error is not None:
                    raise error
                return {"answer":answer,"citation_ids":[],"evidence_sufficient":False}
        broker=HostBroker(task,backend,ScriptedAnswer())
        broker('complete',{'stage':'answer','payload':{'evidence':[]}})
        returned = answer if self.returned_answer is None else self.returned_answer
        origin=broker.answer_origin_receipt(returned)
        failures=[] if returned.strip() else ["answer_empty"]
        if not origin['valid']: failures.append('invalid_answer_origin')
        return {"schema":EXECUTION_SCHEMA,"node_id":node_id,"program_id":node["program_id"],
                "question_id":task["question_id"],"answer":returned,
                "answer_usable":bool(returned.strip()),"execution_ok":True,"citation_source_valid":False,
                "answer_origin_valid":origin['valid'],"answer_origin_status":origin['status'],
                "host_answer_origin_validation":origin,
                "failure_classes":failures,"model_errors":broker.model_errors,"trace":broker.events,
                "host_evidence_trace":{"read_presentations":[],"final_observations":broker.final_observations},
                "candidate_reported":{"state":{"gaps":["synthetic missing relation","another missing relation"]}},
                "resource_usage":broker.counts}


def no_transport(body):
    raise AssertionError("no physical request expected in this fixture")


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        folder = Path(__file__).parent / "runs"
        folder.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-calibration-", dir=folder)
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.credential_guard = patch.object(cal, "credential_from_plan", side_effect=AssertionError("credential read forbidden"))
        self.credential_guard.start()
        self.addCleanup(self.credential_guard.stop)

    def plan(self, *, questions=2, repeats=1, opaque_references=False):
        tasks = [{"id": "synthetic-q" + str(i), "question_id": "synthetic-q" + str(i),
                  "question": "Which fictional port serves synthetic island " + str(i) + "?",
                  "dataset": "browsecomp-plus", "task_type": "qa", "documents": [],
                  "corpus_ref": "synthetic-fixed-corpus", "corpus_scope": "shared", "excluded_docids": []}
                 for i in range(questions)]
        save(self.root / "tasks.json", tasks)
        refs = [{"query_id": t["question_id"], "question": t["question"], "reference_answer": "Synthetic Port"}
                for t in tasks]
        (self.root / "refs.jsonl").write_text("not JSON: opaque synthetic bytes" if opaque_references else
                                               "\n".join(json.dumps(r) for r in refs), encoding="utf-8")
        (self.root / "corpus.fixture").write_bytes(b"synthetic corpus marker; never opened as a database")

        def bind(name):
            path = self.root / name
            return {"path": str(path), "sha256": cal.file_hash(path)}

        common = {"max_rounds": 3, "max_model_calls": 7, "max_stagnant_rounds": 2}
        return {"schema": cal.SCHEMA, "purpose": "used_development_calibration",
                "tasks_file": bind("tasks.json"), "references_file": bind("refs.jsonl"),
                "corpus": bind("corpus.fixture"), "corpus_ref": "synthetic-fixed-corpus",
                "question_ids": [t["question_id"] for t in tasks], "repeats": repeats,
                "arms": [{"name": "single", "config": {**common, "mode": "single_pass"}},
                         {"name": "loop", "config": {**common, "mode": "iterative"}}],
                "model": {"name": "deepseek-flash", "thinking": "disabled", "temperature": 0,
                          "max_input_bytes": 120000, "prices": {"input_hit": .04, "input_miss": 2, "output": 8},
                          "output_limits": {"plan": 1200, "read": 2200, "answer": 800}},
                "limits": {"max_models": 7, "max_searches": 8, "max_reads": 8},
                "max_calls": questions * repeats * 7, "hard_cny": 30,
                "runtime_source_hashes": _runtime_source_hashes(), "schedule_seed": 27,
                "credential_source": {"kind": "env_file", "variable": "DEEPSEEK_API_KEY",
                                      "path": str(self.root / "nonexistent-credentials")},
                "output_dir": str(self.root / "output")}

    def generate(self, plan, executor=None, transport=no_transport):
        executor = executor or FakeExecutor()
        frozen = cal.generate(plan, approved_plan_hash=digest(plan), transport=transport,
                              backend=FakeBackend(), executor=executor)
        return frozen, executor

    def reject_grade_before_references(self, plan):
        original = cal._verified_bytes
        reference_reads = []

        def guarded(item):
            if item == plan["references_file"]:
                reference_reads.append(item)
                raise AssertionError("invalid generation reached private-reference access")
            return original(item)

        with patch.object(cal, "_verified_bytes", side_effect=guarded), self.assertRaises(ValueError):
            cal.grade(plan)
        self.assertEqual(reference_reads, [])

    def test_preflight_derives_112_calls_and_conservative_30_cny_envelope(self):
        plan = self.plan(questions=8, repeats=2, opaque_references=True)
        with patch.object(cal._PreflightServices, "search", side_effect=AssertionError("search")), \
                patch.object(cal._PreflightServices, "complete", side_effect=AssertionError("model")):
            result = cal.preflight(plan)
        self.assertEqual(result["max_calls"], 112)
        self.assertEqual(result["answer_outcomes"], 32)
        self.assertAlmostEqual(result["conservative_cny_upper_bound"], 28.594176)
        self.assertEqual(result["hard_cny"], 30)
        self.assertFalse(result["credentials_read"])
        self.assertFalse(result["references_parsed"])

    def test_only_mode_may_differ_between_arms(self):
        plan = self.plan()
        plan["arms"][1]["config"]["search_limit"] = 6
        with self.assertRaisesRegex(ValueError, "only the iterative"):
            cal.preflight(plan)

    def test_rejects_changed_call_ceiling_or_insufficient_cny_and_final_budget(self):
        plan = self.plan(questions=8, repeats=2)
        for changed in ({"max_calls": 111}, {"max_calls": 113}, {"hard_cny": 28},
                        {"limits": {"max_models": 4}}):
            candidate = {**plan, **changed}
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                cal.preflight(candidate)

    def test_approval_hash_mismatch_prevents_executor_and_transport(self):
        plan = self.plan()
        executor = FakeExecutor()
        with self.assertRaisesRegex(ValueError, "reviewed plan"):
            cal.generate(plan, approved_plan_hash="not-approved", transport=no_transport,
                         backend=FakeBackend(), executor=executor)
        self.assertEqual(executor.calls, [])
        self.assertFalse(Path(plan["output_dir"]).exists())

    def test_generation_does_not_parse_references_and_complete_resume_reuses_all_cells(self):
        plan = self.plan(opaque_references=True)
        first, executor = self.generate(plan)
        self.assertEqual(len(executor.calls), 4)
        second, _ = self.generate(plan, executor)
        self.assertEqual(first, second)
        self.assertEqual(len(executor.calls), 4)
        self.assertFalse(first["references_parsed_by_runner"])
        self.assertEqual(first["ledger"]["used"], {})

    def test_recovery_after_all_cells_written_before_freeze_does_not_execute_again(self):
        plan = self.plan()
        original_freeze = cal.freeze

        def interrupted(path, value):
            if Path(path).name == "generation_freeze.json":
                raise OSError("synthetic interruption before freeze")
            return original_freeze(path, value)

        executor = FakeExecutor()
        with patch.object(cal, "freeze", side_effect=interrupted), self.assertRaises(OSError):
            self.generate(plan, executor)
        self.assertEqual(len(executor.calls), 4)
        frozen, _ = self.generate(plan, executor)
        self.assertEqual(len(frozen["cells"]), 4)
        self.assertEqual(len(executor.calls), 4)

    def test_grade_only_after_complete_freeze_and_builds_blind_packet(self):
        plan = self.plan()
        self.generate(plan)
        result = cal.grade(plan)
        self.assertFalse(result["official_browsecomp_score"])
        self.assertEqual(result["independent_units"], 2)
        self.assertTrue(result["all_outcomes_retained"])
        self.assertEqual(result["arms"]["single"]["outcomes"], 2)
        self.assertEqual(result["arms"]["loop"]["proxy_answer_f1"], 1)
        for arm in result["arms"].values():
            self.assertEqual(arm["structural_failure_counts"]["no_retrieval"], 2)
            self.assertEqual(arm["model_reported_counts"], {"evidence_gap": 2, "evidence_insufficient": 2})
            self.assertFalse(arm["model_reports_are_verified_truth"])
            self.assertIn("not independent", arm["diagnostic_count_unit"])
        rows = json.loads((Path(plan["output_dir"]) / "grading/rows.json").read_text(encoding="utf-8"))
        self.assertTrue(all("diagnostics" in row and "repeat" in row and "question_id" in row for row in rows))
        packet = json.loads((Path(plan["output_dir"]) / "grading/blind_packet.json").read_text(encoding="utf-8"))
        self.assertEqual(len(packet), 4)
        self.assertTrue(all("arm" not in row and "repeat" not in row and "reference_answer" in row for row in packet))

    def test_missing_and_duplicate_freezes_rejected_before_reference_access_or_resume(self):
        plan = self.plan()
        complete, executor = self.generate(plan)
        path = Path(plan["output_dir"]) / "generation_freeze.json"
        for rows in ([], complete["cells"][:-1], complete["cells"][:-1] + [complete["cells"][0]]):
            with self.subTest(count=len(rows)):
                save(path, {**complete, "cells": rows})
                self.reject_grade_before_references(plan)
                with self.assertRaises(ValueError):
                    self.generate(plan, executor)
        self.assertEqual(len(executor.calls), 4)

    def test_out_of_panel_identity_rejected_before_reference_access(self):
        plan = self.plan()
        frozen, _ = self.generate(plan)
        for field, value in (("question_id", "foreign-q"), ("arm", "foreign-arm"), ("repeat", 9),
                             ("repeat", False), ("plan_hash", "wrong"), ("backend", "different")):
            with self.subTest(field=field, value=value):
                changed = copy.deepcopy(frozen)
                changed["cells"][0]["identity"][field] = value
                save(Path(plan["output_dir"]) / "generation_freeze.json", changed)
                self.reject_grade_before_references(plan)

    def test_cell_path_cannot_escape_or_redirect_identity(self):
        plan = self.plan()
        frozen, _ = self.generate(plan)
        for value in ("../refs.jsonl", str(self.root / "refs.jsonl"), "cells/other/generation.json",
                      frozen["cells"][0]["file"].replace("/", "\\")):
            with self.subTest(path=value):
                changed = copy.deepcopy(frozen)
                changed["cells"][0]["file"] = value
                save(Path(plan["output_dir"]) / "generation_freeze.json", changed)
                self.reject_grade_before_references(plan)

    def test_resolved_path_guard_rejects_link_escape(self):
        out = self.root / "output"
        relative = cal._cell_relative("q", "arm", 0)
        original = Path.resolve

        def resolve(path, *args, **kwargs):
            if str(path).endswith("generation.json"):
                return self.root / "escaped-generation.json"
            return original(path, *args, **kwargs)

        with patch.object(Path, "resolve", resolve), self.assertRaisesRegex(ValueError, "escapes"):
            cal._cell_path(out, relative)

    def test_cell_hash_and_rehashed_payload_identity_are_validated(self):
        plan = self.plan()
        frozen, _ = self.generate(plan)
        item = frozen["cells"][0]
        path = Path(plan["output_dir"]) / item["file"]
        record = json.loads(path.read_text(encoding="utf-8"))
        record["payload"]["question_id"] = "changed-question"
        save(path, record)
        self.reject_grade_before_references(plan)
        record["payload_hash"] = digest(record["payload"])
        save(path, record)
        item["sha256"] = cal.file_hash(path)
        save(Path(plan["output_dir"]) / "generation_freeze.json", frozen)
        self.reject_grade_before_references(plan)

    def test_archived_program_must_match_fixed_arm_not_merely_consistent_ids(self):
        plan = self.plan()
        frozen, _ = self.generate(plan)
        out = Path(plan["output_dir"])
        item = frozen["cells"][0]
        identity = item["identity"]
        arm_index = next(i for i, arm in enumerate(plan["arms"]) if arm["name"] == identity["arm"])
        files = cal.root_files(plan["arms"][arm_index]["config"])
        files["rag_core.py"] += "\n# synthetic altered candidate, never executed\n"
        archive = ProgramArchive(out / "archive")
        node = archive.record(files, {}, session_id="calibration-arms", attempt=arm_index)
        identity.update(node_id=node["node_id"], program_id=node["program_id"])
        path = out / item["file"]
        record = json.loads(path.read_text(encoding="utf-8"))
        record["identity"] = copy.deepcopy(identity)
        record["payload"].update(node_id=node["node_id"], program_id=node["program_id"])
        record["payload_hash"] = digest(record["payload"])
        save(path, record)
        item["sha256"] = cal.file_hash(path)
        save(out / "generation_freeze.json", frozen)
        self.reject_grade_before_references(plan)

    def test_task_snapshot_is_read_and_hashed_once_and_not_reopened_after_execution_starts(self):
        plan = self.plan()
        old_tasks = json.loads(Path(plan["tasks_file"]["path"]).read_text(encoding="utf-8"))
        reads = []
        original = cal._verified_bytes

        def reader(item):
            if item == plan["tasks_file"]:
                reads.append(item)
            return original(item)

        def action(task, model):
            changed = copy.deepcopy(old_tasks)
            changed[0]["question"] = "Question changed after validation"
            save(plan["tasks_file"]["path"], changed)

        executor = FakeExecutor(action)
        with patch.object(cal, "_verified_bytes", side_effect=reader):
            self.generate(plan, executor)
        self.assertEqual(len(reads), 1)
        expected = {t["question_id"]: t["question"] for t in old_tasks}
        self.assertTrue(all(t["question"] == expected[t["question_id"]] for t in executor.calls))

    def test_plan_is_deepcopied_before_calling_injected_dependencies(self):
        plan = self.plan()
        approved = digest(plan)

        def action(task, model):
            plan["question_ids"].append("external-mutation")
            plan["arms"][0]["config"]["max_rounds"] = 100

        frozen, executor = self.generate(plan, FakeExecutor(action))
        self.assertEqual(frozen["plan_hash"], approved)
        self.assertEqual(len(executor.calls), 4)
        stored = json.loads((Path(plan["output_dir"]) / "plan.json").read_text(encoding="utf-8"))
        self.assertEqual(digest(stored), approved)

    def test_unknown_physical_outcome_stops_and_resume_does_not_repurchase(self):
        plan = self.plan(opaque_references=True)
        sent = []

        def transport(body):
            sent.append(body)
            raise TimeoutError("synthetic unknown physical outcome")

        executor = FakeExecutor(lambda task, model: model.complete("answer", {"question": task["question"]}))
        for attempt in range(2):
            with self.subTest(attempt=attempt), self.assertRaises(UnknownProviderOutcome):
                self.generate(plan, executor, transport)
        self.assertEqual(len(sent), 1)
        self.assertFalse((Path(plan["output_dir"]) / "generation_freeze.json").exists())
        progress = json.loads((Path(plan["output_dir"]) / "progress.json").read_text(encoding="utf-8"))
        self.assertEqual(progress["status"], "stopped")
        self.assertEqual(progress["completed"], 0)
        self.assertEqual(progress["ledger"]["used"]["run"]["calls"], 1)

    def test_unavailable_or_unknown_schema_receipt_stops_before_checkpoint(self):
        mutations = ({"schema": "rag-rsi-v3-execution-unknown"}, {"schema":"rag-rsi-v3-execution-2"},
                     {"provider_outcome": "unknown"}, {"measurement_status": "unavailable"},
                     {"failure_classes": ["UnknownProviderOutcome"]},
                     {"model_errors": ["HostError"]})
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=mutation):
                plan = self.plan()
                plan["output_dir"] = str(self.root / ("unknown-output-" + str(index)))
                base = FakeExecutor()

                def executor(*args, **kwargs):
                    return {**base(*args, **kwargs), **mutation}

                with self.assertRaises(ValueError):
                    self.generate(plan, executor)
                out = Path(plan["output_dir"])
                self.assertEqual(len(base.calls), 1)
                self.assertEqual(list(out.glob("cells/*/generation.json")), [])
                self.assertFalse((out / "generation_freeze.json").exists())
                self.reject_grade_before_references(plan)

    def test_completed_bad_model_response_remains_an_observed_outcome(self):
        plan = self.plan()
        executor=FakeExecutor(returned_answer="",model_error=ModelResponseError("synthetic malformed response"))

        self.generate(plan, executor)
        result = cal.grade(plan)
        self.assertEqual(result["arms"]["loop"]["proxy_answer_f1"], 0)
        self.assertEqual(result["arms"]["loop"]["structural_failure_counts"]["model_parse_failure"], 2)

    def test_invalid_answer_origin_retains_raw_scores_without_quality_claim(self):
        plan=self.plan()
        self.generate(plan,FakeExecutor(model_answer="Other fictional port",returned_answer="Synthetic Port"))
        report=cal.grade(plan)
        self.assertEqual(report["status"],"protocol_invalid")
        self.assertFalse(report["quality_comparison_valid"])
        self.assertIsNone(report["paired_question_f1_deltas"])
        self.assertTrue(report["all_outcomes_retained"])
        for arm in report["arms"].values():
            self.assertEqual(arm["proxy_answer_em"],1)
            self.assertEqual(arm["answer_origin_valid_rate"],0)
            self.assertEqual(arm["eligible_outcomes"],0)

    def test_blank_model_answer_cannot_forge_usable_and_become_eligible(self):
        plan=self.plan()
        base=FakeExecutor(model_answer='  ',returned_answer='')
        def executor(*args,**kwargs):
            receipt=base(*args,**kwargs)
            self.assertTrue(receipt['answer_origin_valid'])
            receipt['answer_usable']=True
            return receipt
        with self.assertRaisesRegex(ValueError,'usability'):
            self.generate(plan,executor)
        self.reject_grade_before_references(plan)

    def test_forged_origin_and_missing_origin_stop_before_freeze(self):
        for missing in (False,True):
            plan=self.plan()
            plan["output_dir"]=str(self.root/('origin-'+str(missing)))
            base=FakeExecutor()
            def executor(*args,**kwargs):
                receipt=base(*args,**kwargs)
                if missing: receipt.pop('host_answer_origin_validation')
                else: receipt['answer']='Candidate override'
                return receipt
            with self.assertRaises(ValueError): self.generate(plan,executor)
            self.assertFalse((Path(plan['output_dir'])/'generation_freeze.json').exists())
            self.reject_grade_before_references(plan)

    def test_unavailable_or_nonunit_metrics_cannot_become_zero_in_report(self):
        plan = self.plan()
        self.generate(plan)
        for name in ("answer_em", "answer_f1"):
            for value in (None, float("nan"), float("inf"), -0.1, 1.1, True):
                metrics = {"answer_em": 1.0, "answer_f1": 1.0, name: value}
                with self.subTest(name=name, value=value), \
                        patch("code_rsi.v3.task_metrics.score_task", return_value=metrics), \
                        self.assertRaisesRegex(ValueError, "available finite unit score"):
                    cal.grade(plan)
                self.assertFalse((Path(plan["output_dir"]) / "report.json").exists())

    def test_empty_reference_cannot_be_reported_as_wrong_answer(self):
        plan = self.plan()
        path = Path(plan["references_file"]["path"])
        refs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        refs[0]["reference_answer"] = "   "
        path.write_text("\n".join(json.dumps(r) for r in refs), encoding="utf-8")
        plan["references_file"]["sha256"] = cal.file_hash(path)
        self.generate(plan)
        with self.assertRaisesRegex(ValueError, "reference answer unavailable"):
            cal.grade(plan)
        self.assertFalse((Path(plan["output_dir"]) / "report.json").exists())

    def test_runtime_or_scorer_change_blocks_grade_before_reference_access(self):
        plan = self.plan()
        self.generate(plan)
        with patch.object(cal, "_runtime_source_hashes", return_value={"modified-scorer": "different"}):
            self.reject_grade_before_references(plan)

    def test_changed_tasks_bytes_are_rejected_before_executor(self):
        plan = self.plan()
        Path(plan["tasks_file"]["path"]).write_text("[]", encoding="utf-8")
        executor = FakeExecutor()
        with self.assertRaisesRegex(ValueError, "frozen file changed"):
            self.generate(plan, executor)
        self.assertEqual(executor.calls, [])


if __name__ == "__main__":
    unittest.main()
