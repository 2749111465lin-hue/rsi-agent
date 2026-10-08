"""Independent reader-probe contract/recovery tests; synthetic data and zero API."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from code_rsi.budget import digest, save
from code_rsi.v3 import reader_probe as probe
from code_rsi.v3 import reader_replay as replay
from code_rsi.v3.evolution import _runtime_source_hashes
from code_rsi.v3.execution import HostBroker, EXECUTION_SCHEMA, root_files
from code_rsi.v3.infrastructure import UnknownProviderOutcome
from code_rsi.v3.rag import RagEngine


TEXT = "Aster Port serves Blue Isle. Beacon Port serves Red Isle."
FIRST = "Aster Port serves Blue Isle."
SECOND = "Beacon Port serves Red Isle."


def reading(payload, phrase=FIRST):
    return {"claims": [{"text": phrase, "citations": [
        {"source_id": payload["sources"][0]["source_id"], "quote": phrase}]}],
        "bridge_entities": [], "gaps": [], "conflicts": [],
        "queries": ["next fictional query"], "ready": False}


def fixture_case(case_id="state0", qid="fictional-q"):
    task = {"id": qid, "question_id": qid, "question": "Which port serves Red Isle?",
            "dataset": "browsecomp-plus", "task_type": "qa", "documents": [],
            "corpus_ref": "fictional-corpus", "corpus_scope": "shared", "excluded_docids": []}
    config = {"mode": "iterative", "max_rounds": 2, "max_stagnant_rounds": 2}
    events = []
    class Services:
        def record(self, name, request, response):
            events.append({"name": name, "request": deepcopy(request),
                           "response": deepcopy(response), "response_sha256": digest(response)})
            return deepcopy(response)
        def search(self, query, limit=5):
            return self.record("search", {"query": query, "limit": limit}, [{
                "docid": "doc1", "text": TEXT, "start": 0, "end": len(TEXT),
                "document_hash": hashlib.sha256(TEXT.encode()).hexdigest()}])
        def complete(self, stage, payload):
            if stage == "plan":
                response = {"constraints": ["port relation"], "queries": ["fictional query"]}
            elif stage == "read":
                response = reading(payload)
            else:
                response = {"answer": "Insufficient information", "citation_ids": [],
                            "evidence_sufficient": False}
            return self.record("complete", {"stage": stage, "payload": payload}, response)
    services = Services()
    RagEngine(services, services, config=config).solve({"question": task["question"], "task_id": qid})
    return {"schema": replay.SCHEMA, "case_id": case_id, "task": task, "config": config,
            "target_read": 2, "events": events,
            "engine_sha256": hashlib.sha256(root_files(config)["rag_core.py"].encode()).hexdigest(),
            "original_final_payload_sha256": digest(events[-1]["request"]["payload"]),
            "source_binding": {"kind": "fictional-test-trace"}}


class ScriptedTransport:
    """A local fake provider records every physical request; never performs I/O."""
    def __init__(self, *, missing_answer=False):
        self.sent = []
        self.missing_answer = missing_answer
    def __call__(self, body):
        self.sent.append(deepcopy(body))
        payload = json.loads(body["messages"][-1]["content"])
        stage = "read" if "sources" in payload else "answer"
        if stage == "read":
            value = reading(payload, SECOND)
        else:
            value = {"answer": "" if self.missing_answer else "Beacon Port",
                     "citation_ids": [], "evidence_sufficient": False}
        return {"model": "synthetic-model", "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(value)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                      "prompt_cache_hit_tokens": 40}}


class TrustedFixtureExecutor:
    """Execute the trusted RagEngine in-process, not arbitrary archived code.

    The real HostBroker produces receipts and origin bindings. WSL execution is
    covered separately; this fixture tests orchestration without claiming sandbox proof.
    """
    def __init__(self, *, mismatch_answer=False):
        self.calls = []
        self.mismatch_answer = mismatch_answer
    def __call__(self, archive, node_id, task, backend, model, directory, *, limits):
        self.calls.append((node_id, task["question_id"]))
        broker = HostBroker(task, backend, model, **limits)
        case = model.case
        config = deepcopy(case["config"])
        config["max_rounds"] = min(config.get("max_rounds", 3), case["target_read"])
        class Backend:
            def search(self, query, limit=5):
                return broker("search", {"query": query, "limit": limit})
        class Model:
            ordinal = 0
            def complete(self, stage, payload):
                if stage == "read":
                    self.ordinal += 1
                    if self.ordinal == case["target_read"] and model.guidance is not None:
                        payload = deepcopy(payload)
                        payload["additional_guidance"] = model.guidance
                return broker("complete", {"stage": stage, "payload": payload})
        result = RagEngine(Backend(), Model(), config=config).solve(
            {"question": task["question"], "task_id": task["question_id"]})
        if broker.fatal is not None:
            raise broker.fatal
        answer = result["answer"] or ""
        if self.mismatch_answer:
            answer = "forged postprocessing"
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


class ReaderProbeTests(unittest.TestCase):
    def setUp(self):
        runs = Path(__file__).parent / "runs"
        runs.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="synthetic-reader-probe-", dir=runs)
        self.root = Path(temporary.name)
        self.addCleanup(temporary.cleanup)
        guard = patch.object(probe, "credential_from_plan", side_effect=AssertionError("credential read"))
        guard.start(); self.addCleanup(guard.stop)

    def plan(self, *, states=1, repeats=1, opaque=False):
        cases = [fixture_case("state" + str(i)) for i in range(states)]
        save(self.root / "cases.json", {"schema": "rag-rsi-reader-replay-cases-1", "cases": cases})
        if opaque:
            (self.root / "refs.json").write_bytes(b"opaque reference bytes; preflight must not parse")
            (self.root / "anchors.json").write_bytes(b"opaque anchor bytes; preflight must not parse")
        else:
            save(self.root / "refs.json", [{"query_id": "fictional-q", "question": cases[0]["task"]["question"],
                                           "reference_answer": "Beacon Port"}])
            anchors = [{"docid": "doc1", "start": TEXT.index(s), "end": TEXT.index(s)+len(s),
                        "quote_sha256": hashlib.sha256(s.encode()).hexdigest()} for s in (FIRST, SECOND)]
            save(self.root / "anchors.json", {"schema": "rag-rsi-reader-anchors-1",
                                              "cases": {c["case_id"]: anchors for c in cases}})
        def bind(name):
            path = self.root / name
            return {"path": str(path), "sha256": probe.file_hash(path)}
        return {"schema": probe.SCHEMA, "purpose": "fixed_last_read_to_answer", "question_use": "synthetic",
                "cases_file": bind("cases.json"), "references_file": bind("refs.json"),
                "annotations_file": bind("anchors.json"), "output_dir": str(self.root / "output"),
                "arms": [{"name": "A", "guidance": None}, {"name": "B", "guidance": probe.GUIDANCE}],
                "repeats": repeats, "schedule_seed": 42,
                "model": {"name": "deepseek-flash", "temperature": 0, "thinking": "disabled",
                          "max_input_bytes": 120000, "output_limits": {"read": 2200, "answer": 800},
                          "prices": {"input_miss": 2, "input_hit": .04, "output": 8}},
                "hard_cny": 50, "max_calls": states*repeats*4,
                "runtime_source_hashes": _runtime_source_hashes(),
                "credential_source": {"kind": "env_file", "path": str(self.root / "absent-credentials"),
                                      "variable": "DEEPSEEK_API_KEY"}}

    def generate(self, plan, executor=None, transport=None):
        executor = executor or TrustedFixtureExecutor()
        transport = transport or ScriptedTransport()
        frozen = probe.generate(plan, approved_plan_hash=digest(plan), executor=executor, transport=transport)
        return frozen, executor, transport

    def reject_before_private(self, plan):
        original = probe._verified_bytes
        opened = []
        def guarded(item):
            if item in (plan["references_file"], plan["annotations_file"]):
                opened.append(item)
                raise AssertionError("private file reached before complete valid freeze")
            return original(item)
        with patch.object(probe, "_verified_bytes", side_effect=guarded), self.assertRaises(ValueError):
            probe.grade(plan)
        self.assertEqual(opened, [])

    def test_preflight_counts_states_separately_from_questions_without_private_access(self):
        plan = self.plan(states=4, repeats=2, opaque=True)
        original = probe._verified_bytes
        def checked(item):
            self.assertEqual(item, plan["cases_file"])
            return original(item)
        with patch.object(probe, "_verified_bytes", side_effect=checked), \
                patch.object(probe, "deepseek_transport", side_effect=AssertionError("transport construction")):
            result = probe.preflight(plan)
        self.assertEqual((result["states"], result["questions"], result["outcomes"]), (4, 1, 16))
        self.assertEqual(result["max_new_calls"], 32)
        self.assertFalse(result["reference_or_credential_access"])
        self.assertFalse(Path(plan["output_dir"]).exists())

    def test_preflight_rejects_other_variables_and_invalid_numeric_limits(self):
        base = self.plan()
        mutations = [lambda p: p["arms"][1].update(config={"temperature": 1}),
                     lambda p: p["arms"][1].update(guidance="different unreviewed guidance"),
                     lambda p: p["model"].update(temperature=.2),
                     lambda p: p["model"].update(thinking="enabled"),
                     lambda p: p["model"]["output_limits"].update(answer=801),
                     lambda p: p.update(repeats=True), lambda p: p.update(schedule_seed=True),
                     lambda p: p.update(max_calls=3), lambda p: p.update(max_calls=5),
                     lambda p: p.update(hard_cny=.001), lambda p: p.update(hard_cny=float("nan")),
                     lambda p: p["model"]["prices"].update(input_miss=-1),
                     lambda p: p["model"]["prices"].update(input_hit=3),
                     lambda p: p.update(runtime_source_hashes={}),
                     lambda p: p.update(output_dir=str(self.root.parent.parent / "outside-runs"))]
        for i, mutate in enumerate(mutations):
            plan = deepcopy(base); mutate(plan)
            with self.subTest(i=i), self.assertRaises(ValueError): probe.preflight(plan)

    def test_changed_case_bytes_and_duplicate_state_fail_preflight(self):
        plan = self.plan()
        path = Path(plan["cases_file"]["path"])
        packet = json.loads(path.read_text()); packet["cases"].append(deepcopy(packet["cases"][0]))
        save(path, packet)
        with self.assertRaises(ValueError): probe.preflight(plan)
        plan["cases_file"]["sha256"] = probe.file_hash(path)
        with self.assertRaisesRegex(ValueError, "duplicate"): probe.preflight(plan)

    def test_approval_must_bind_full_plan_before_transport_or_output(self):
        plan = self.plan(); executor = TrustedFixtureExecutor(); transport = ScriptedTransport()
        with self.assertRaisesRegex(ValueError, "reviewed plan"):
            probe.generate(plan, approved_plan_hash="not approved", executor=executor, transport=transport)
        self.assertEqual(executor.calls, []); self.assertEqual(transport.sent, [])
        self.assertFalse(Path(plan["output_dir"]).exists())

    def test_complete_resume_neither_dispatches_nor_reads_credentials(self):
        plan = self.plan(opaque=True, repeats=2)
        frozen, executor, transport = self.generate(plan)
        self.assertEqual(len(transport.sent), 8)
        self.assertEqual(len(frozen["cells"]), 4)
        with patch.object(probe, "deepseek_transport", side_effect=AssertionError("transport construction")):
            second = probe.generate(plan, approved_plan_hash=digest(plan), executor=executor)
        self.assertEqual(second, frozen)
        self.assertEqual(len(executor.calls), 4)
        self.assertEqual(len(transport.sent), 8)

    def test_same_body_is_fresh_in_every_arm_repeat_bank(self):
        plan = self.plan(repeats=2)
        _, _, transport = self.generate(plan)
        self.assertEqual(len(transport.sent), 8)
        records = [json.loads(p.read_text()) for p in (Path(plan["output_dir"])/"requests").glob("*.json")
                   if p.name != "returned_model.json"]
        self.assertEqual(len(records), 8)
        self.assertEqual(len({r["key"] for r in records}), 8)

    def test_unknown_request_recovery_precedes_credentials_and_changed_transport(self):
        plan = self.plan(); sent = []
        def unknown(body):
            sent.append(body); raise TimeoutError("unknown physical result")
        with self.assertRaises(UnknownProviderOutcome): self.generate(plan, transport=unknown)
        with patch.object(probe, "deepseek_transport", side_effect=AssertionError("transport construction")), \
                self.assertRaises(UnknownProviderOutcome):
            probe.generate(plan, approved_plan_hash=digest(plan), executor=TrustedFixtureExecutor())
        self.assertEqual(len(sent), 1)
        self.assertFalse((Path(plan["output_dir"])/"generation_freeze.json").exists())

    def test_completed_cells_survive_interruption_before_freeze_without_repurchase(self):
        plan = self.plan(); executor = TrustedFixtureExecutor(); transport = ScriptedTransport()
        original = probe.freeze
        def fail_last(path, value):
            if Path(path).name == "generation_freeze.json": raise OSError("before freeze")
            return original(path, value)
        with patch.object(probe, "freeze", side_effect=fail_last), self.assertRaises(OSError):
            self.generate(plan, executor, transport)
        self.assertEqual(len(transport.sent), 4)
        self.reject_before_private(plan)
        frozen, _, _ = self.generate(plan, executor, transport)
        self.assertEqual(len(frozen["cells"]), 2)
        self.assertEqual(len(transport.sent), 4); self.assertEqual(len(executor.calls), 2)

    def test_grade_missing_duplicate_and_foreign_cells_before_private_files(self):
        plan = self.plan(); frozen, _, _ = self.generate(plan)
        path = Path(plan["output_dir"])/"generation_freeze.json"
        variants = [[], frozen["cells"][:-1], [frozen["cells"][0], frozen["cells"][0]]]
        bad = deepcopy(frozen["cells"]); bad[0]["identity"]["arm"] = "foreign"; variants.append(bad)
        for cells in variants:
            save(path, {**frozen, "cells": cells})
            self.reject_before_private(plan)

    def test_grade_tampered_cells_and_source_drift_before_private_files(self):
        plan = self.plan(); frozen, _, _ = self.generate(plan)
        cell = frozen["cells"][0]; path = Path(plan["output_dir"])/cell["file"]
        record = json.loads(path.read_text()); record["payload"]["question_id"] = "foreign"
        save(path, record); self.reject_before_private(plan)
        record["payload_hash"] = digest(record["payload"]); save(path, record)
        cell["sha256"] = probe.file_hash(path)
        save(Path(plan["output_dir"])/"generation_freeze.json", frozen)
        self.reject_before_private(plan)
        with patch.object(probe, "_runtime_source_hashes", return_value={}):
            self.reject_before_private(plan)

    def test_grade_reports_prefix_new_and_final_anchor_coverage_separately(self):
        plan = self.plan(states=2, repeats=2)
        frozen, _, transport = self.generate(plan)
        report = probe.grade(plan)
        self.assertEqual((report["states"], report["questions"], report["outcomes"]), (2, 1, 8))
        for row in report["rows"]:
            m = row["mechanics"]
            self.assertEqual(m["anchor_in_prefix"], [True, False])
            self.assertEqual(m["anchor_in_new_read"], [False, True])
            self.assertEqual(m["anchor_in_answer_input"], [True, True])
            self.assertFalse(m["anchor_match_proves_all_constraints"])
            self.assertEqual(m["semantic_support"], "not_independently_adjudicated")
            self.assertEqual(row["metrics"]["answer_em"], 1)
        self.assertEqual(len(transport.sent), 16)
        self.assertFalse(report["official_browsecomp_score"])
        self.assertFalse(report["independent_quality_evidence"])
        self.assertEqual(probe.grade(plan), report)

    def test_ineligible_origin_is_retained_and_quality_unknown_not_zero(self):
        plan = self.plan()
        self.generate(plan, executor=TrustedFixtureExecutor(mismatch_answer=True))
        report = probe.grade(plan)
        self.assertEqual(report["status"], "protocol_invalid")
        self.assertEqual(len(report["rows"]), 2)
        for row in report["rows"]:
            self.assertFalse(row["program_eligible"])
            self.assertIsNone(row["metrics"])
            self.assertEqual(row["mechanics"]["anchor_in_new_read"], [False, True])

    def test_rehashed_read_observation_tampering_is_rejected_before_private(self):
        plan = self.plan(); frozen, _, _ = self.generate(plan)
        cell = frozen["cells"][0]; path = Path(plan["output_dir"])/cell["file"]
        record = json.loads(path.read_text())
        # Retain authentic final-answer provenance while falsifying the proposed
        # mechanical endpoint. Outer self-hashes must not make this trustworthy.
        record["payload"]["host_evidence_trace"]["read_presentations"][-1]["verified_quotes"] = []
        record["payload_hash"] = digest(record["payload"]); save(path, record)
        cell["sha256"] = probe.file_hash(path)
        save(Path(plan["output_dir"])/"generation_freeze.json", frozen)
        self.reject_before_private(plan)

    def test_boolean_repeat_cannot_alias_integer_after_cell_rebinding(self):
        plan = self.plan(); frozen, _, _ = self.generate(plan)
        cell = frozen["cells"][0]
        record = json.loads((Path(plan["output_dir"])/cell["file"]).read_text())
        cell["identity"]["repeat"] = False
        record["identity"] = deepcopy(cell["identity"])
        cell["file"] = probe._relative(cell["identity"]["case_id"], cell["identity"]["arm"], False)
        path = Path(plan["output_dir"])/cell["file"]
        save(path, record); cell["sha256"] = probe.file_hash(path)
        save(Path(plan["output_dir"])/"generation_freeze.json", frozen)
        self.reject_before_private(plan)

    def test_actual_input_cache_hits_are_not_reported_as_zero(self):
        plan = self.plan(); self.generate(plan)
        report = probe.grade(plan)
        for arm in ("A", "B"):
            self.assertEqual(report["per_arm_actual_cost"][arm]["calls"], 2)
            self.assertEqual(report["per_arm_actual_cost"][arm]["input"], 200)
            self.assertEqual(report["per_arm_actual_cost"][arm]["input_hit"], 80)

    def test_missing_cache_hit_measurement_remains_unknown(self):
        plan = self.plan(); scripted = ScriptedTransport()
        def absent_hit(body):
            response = scripted(body)
            response["usage"].pop("prompt_cache_hit_tokens")
            return response
        self.generate(plan, transport=absent_hit)
        report = probe.grade(plan)
        self.assertTrue(all(c["input_hit"] is None for c in report["per_arm_actual_cost"].values()))

    def test_anchor_offset_without_matching_source_hash_cannot_be_scored(self):
        plan = self.plan()
        annotations = json.loads(Path(plan["annotations_file"]["path"]).read_text())
        annotations["cases"]["state0"][0]["quote_sha256"] = "0"*64
        save(Path(plan["annotations_file"]["path"]), annotations)
        plan["annotations_file"]["sha256"] = probe.file_hash(plan["annotations_file"]["path"])
        self.generate(plan)
        with self.assertRaisesRegex(ValueError, "anchor"):
            probe.grade(plan)
        self.assertFalse((Path(plan["output_dir"])/"report.json").exists())

    def test_annotation_panel_cannot_silently_drop_a_frozen_state(self):
        plan = self.plan(states=2)
        annotations = json.loads(Path(plan["annotations_file"]["path"]).read_text())
        annotations["cases"].pop("state1")
        save(Path(plan["annotations_file"]["path"]), annotations)
        plan["annotations_file"]["sha256"] = probe.file_hash(plan["annotations_file"]["path"])
        self.generate(plan)
        with self.assertRaisesRegex(ValueError, "exact state panel"):
            probe.grade(plan)

    def test_grade_wrong_reference_question_cannot_write_report(self):
        plan = self.plan()
        refs = json.loads(Path(plan["references_file"]["path"]).read_text())
        refs[0]["question"] = "A different question with the same identifier?"
        save(Path(plan["references_file"]["path"]), refs)
        plan["references_file"]["sha256"] = probe.file_hash(plan["references_file"]["path"])
        self.generate(plan)
        with self.assertRaisesRegex(ValueError, "reference differs"):
            probe.grade(plan)
        self.assertFalse((Path(plan["output_dir"])/"report.json").exists())

    def test_boundary_read_truncation_preserves_prefix_and_unknown_new_measurement(self):
        plan = self.plan(); scripted = ScriptedTransport()
        def truncate_read(body):
            response = scripted(body)
            if body["max_tokens"] == 2200:
                response["choices"][0]["finish_reason"] = "length"
            return response
        self.generate(plan, transport=truncate_read)
        report = probe.grade(plan)
        self.assertEqual(len(scripted.sent), 4)
        for row in report["rows"]:
            self.assertIn("ModelResponseError", row["model_errors"])
            self.assertFalse(row["mechanics"]["last_read_observed"])
            self.assertEqual(row["mechanics"]["anchor_in_prefix"], [True, False])
            self.assertEqual(row["mechanics"]["anchor_in_new_read"], [None, None])
            self.assertIsNone(row["mechanics"]["new_verified_quote_count"])
            self.assertEqual(row["mechanics"]["anchor_in_answer_input"], [True, False])

    def test_no_freeze_means_no_private_reference_or_anchor_read(self):
        self.reject_before_private(self.plan())


if __name__ == "__main__":
    unittest.main()
