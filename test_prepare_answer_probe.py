"""Synthetic answer-state reconstruction tests: no live models or private tasks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from code_rsi.budget import digest, save
from code_rsi import prepare_answer_probe as prepare
from code_rsi.v3 import calibration as cal
from code_rsi.v3.reader_replay import ReplayRouter
from code_rsi.v3.rag import RagEngine
import test_v3_musique_calibration as fixtures


class RecordedExecutor(fixtures.TrustedEngineExecutor):
    """Add the original wrapper's host-owned record_trace call to the fixture."""
    def __call__(self, *args, **kwargs):
        receipt = super().__call__(*args, **kwargs)
        receipt["trace"].append({"name": "record_trace",
            "request": {"result": deepcopy(receipt["candidate_reported"])},
            "response_hash": digest({"recorded": True})})
        return receipt


class PrepareAnswerProbeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.MuSiQueCalibrationTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.plan = self.fixture.plan(opaque_references=True)
        # A deliberately non-sorted frozen order must remain the selection order.
        self.plan["question_ids"] = list(reversed(self.plan["question_ids"]))
        self.transport = fixtures.ScriptedTransport()
        self.frozen, _ = self.fixture.generate(self.plan, RecordedExecutor(), self.transport)
        self.out = Path(self.plan["output_dir"])

    def packet(self):
        return prepare.build_cases(self.plan)

    def selected(self):
        qid = self.plan["question_ids"][0]
        cell = next(c for c in self.frozen["cells"] if
                    (c["identity"]["question_id"], c["identity"]["arm"], c["identity"]["repeat"]) ==
                    (qid, "support", 0))
        return cell, self.out / cell["file"]

    def mutate_receipt(self, change):
        cell, path = self.selected()
        record = json.loads(path.read_bytes())
        change(record["payload"])
        record["payload_hash"] = digest(record["payload"])
        save(path, record)
        cell["sha256"] = cal.file_hash(path)
        save(self.out / "generation_freeze.json", self.frozen)

    def test_mechanical_selection_is_complete_in_declared_order_without_scores(self):
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.out.rglob("*") if p.is_file()}
        sent = len(self.transport.calls)
        protected = {Path(self.plan["references_file"]["path"]).resolve(), self.out / "report.json"}
        raw = Path.read_bytes
        def guard(path):
            if path.resolve() in protected or "grading" in path.parts:
                self.fail("preparation opened private references or scores")
            return raw(path)
        with patch.object(Path, "read_bytes", guard):
            packet = self.packet()
        self.assertEqual(packet["schema"], prepare.PACKET_SCHEMA)
        self.assertEqual([c["case_id"] for c in packet["cases"]], ["Q01", "Q02"])
        self.assertEqual([c["task"]["question_id"] for c in packet["cases"]], self.plan["question_ids"])
        self.assertEqual(len(self.transport.calls), sent)
        after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in self.out.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_exact_model_bank_body_and_original_engine_are_bound(self):
        for case in self.packet()["cases"]:
            binding = case["source_binding"]
            self.assertEqual(binding["selection_rule"], prepare.SELECTION_RULE)
            self.assertEqual(binding["source_identity"]["repeat"], 0)
            self.assertEqual(binding["source_identity"]["arm"], "support")
            self.assertEqual(len(binding["model_requests"]), 3)
            for request in binding["model_requests"]:
                self.assertEqual(request["bank"], "calibration/" + case["task"]["question_id"] + "/0")
                record = json.loads(Path(request["file"]["path"]).read_bytes())
                self.assertEqual(request["request_key"], digest({"body": record["body"], "bank": request["bank"]}))
            router = ReplayRouter(case)
            RagEngine(router, router, config=case["config"]).solve({"question": case["task"]["question"]})
            self.assertEqual(router.assert_complete()["new_model_calls"], 0)

    def test_json_path_is_supported_and_deterministic(self):
        self.assertEqual(self.packet(), prepare.build_cases(self.out / "plan.json"))

    def test_other_dataset_or_role_is_rejected(self):
        for key, value in (("schema", "other"), ("data_role", "D_select")):
            plan = deepcopy(self.plan)
            plan[key] = value
            with self.assertRaisesRegex(ValueError, "MuSiQue D_fit"):
                prepare.build_cases(plan)

    def test_changed_cell_cannot_enter_packet(self):
        _, path = self.selected()
        path.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.packet()

    def test_rehashed_false_read_observation_is_rejected(self):
        def change(receipt):
            receipt["host_evidence_trace"]["read_presentations"][0]["verified_quotes"] = []
        self.mutate_receipt(change)
        with self.assertRaisesRegex(ValueError, "host observations"):
            self.packet()

    def test_rehashed_candidate_state_is_not_ground_truth(self):
        def change(receipt):
            receipt["candidate_reported"]["stop_reason"] = "fabricated"
            receipt["trace"][-1]["request"]["result"] = deepcopy(receipt["candidate_reported"])
        self.mutate_receipt(change)
        with self.assertRaisesRegex(ValueError, "original engine"):
            self.packet()

    def test_missing_record_trace_is_rejected(self):
        self.mutate_receipt(lambda receipt: receipt["trace"].pop())
        with self.assertRaisesRegex(ValueError, "host observations"):
            self.packet()

    def test_a_missing_exact_cache_is_not_replaced_by_another_repeat(self):
        packet = self.packet()
        path = Path(packet["cases"][0]["source_binding"]["model_requests"][0]["file"]["path"])
        path.unlink()
        with self.assertRaises(Exception):
            self.packet()

    def test_cli_freezes_idempotently_without_printing_payloads(self):
        from contextlib import redirect_stdout
        import io
        target = self.fixture.base.root / "answer_cases.json"
        output = io.StringIO()
        with redirect_stdout(output):
            first = prepare.main(["--source-plan", str(self.out / "plan.json"), "--output", str(target)])
            second = prepare.main(["--source-plan", str(self.out / "plan.json"), "--output", str(target)])
        self.assertEqual(first, second)
        lines = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertTrue(all(row["cases"] == 2 and row["new_api_calls"] == 0 for row in lines))
        self.assertNotIn("Synthetic Port", output.getvalue())
        self.assertNotIn("question", lines[0])


if __name__ == "__main__":
    unittest.main()
