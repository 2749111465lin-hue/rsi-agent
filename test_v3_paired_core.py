"""Paired search core with real host measurement/cache and a scripted transport.

The sole execution substitute constructs host-origin receipts; no generated
candidate is executed here, and no external API or credential is accessed.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from code_rsi.budget import Ledger, digest, save
from code_rsi.v3 import evolution, execution
from code_rsi.v3.datasets import adapt_multihop
from code_rsi.v3.infrastructure import PROMPTS, StructuredModel
from test_v3_live_evolution import FixtureTransport, fake_execute


class InjectedPairedCrash(RuntimeError):
    pass


class RejectTransport(FixtureTransport):
    def send(self, body, timeout):
        response = super().send(body, timeout)
        if body["messages"][0]["content"] == PROMPTS["develop"]:
            value = json.loads(response["choices"][0]["message"]["content"])
            value["writes"] = {}
            response["choices"][0]["message"]["content"] = json.dumps(value)
        return response


class PairedCoreTests(unittest.TestCase):
    def setUp(self):
        runs = Path(__file__).parent / "runs"
        runs.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="v3_paired_core_", dir=runs)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.transport = FixtureTransport()
        self.ledger = Ledger(self.base / "ledger.jsonl", {"run": {"calls": 1000, "cny": 10000}})
        self.prices = {"input_hit": .04, "input_miss": 2, "output": 8}
        self.executed = []
        self.loaded = []
        self.models = []
        self.task, ref = adapt_multihop({
            "id": "paired-fit", "query": "Synthetic paired development question",
            "answer": "candidate guess"})
        self.task["documents"] = [{"docid": "synthetic-doc", "text": "Synthetic source text."}]
        self.references = {self.task["question_id"]: ref}
        self.ref_path = self.base / "fit-references.json"
        save(self.ref_path, self.references)
        self.execution_patch = patch.object(execution, "execute", side_effect=self.execute_fixture)
        self.execution_patch.start()
        self.addCleanup(self.execution_patch.stop)

    def model(self, bank, *, name="deepseek-flash"):
        model = StructuredModel(self.base / "requests", self.ledger, self.transport,
                                bank=bank, prices=self.prices, model=name)
        self.models.append(model)
        return model

    def execute_fixture(self, archive, node_id, task, backend, model, directory, *, limits=None):
        self.executed.append({"node_id": node_id, "question_id": task["question_id"],
                              "bank": model.bank, "directory": str(directory)})
        return fake_execute(archive, node_id, task, backend, model, directory, limits=limits)

    def manifest(self, block="block-0", arm="cases", expansions=3):
        identity = self.model("identity-only").identity
        return {
            "expansions": expansions, "metric": "em", "select_candidates": 2,
            "repeats": 1, "root_config": {"mode": "iterative"},
            "limits": {"max_models": 7, "max_searches": 8, "max_reads": 8},
            "synthetic": True, "allow_proxy_metric": True, "model_identity": identity,
            "controls": {"parent_policy": "fixed_root", "module_policy": "fixed",
                         "fixed_module": "answer_generation", "memory": "none",
                         "feedback": arm, "case_schedule": [{
                             "question_id": self.task["question_id"], "repeat": 0}]},
            "lifecycle": {"schema": "rag-rsi-evolution-phases-1", "phase_order": ["search"],
                          "reference_bindings": {"D_fit": {
                              "path": str(self.ref_path.resolve()),
                              "sha256": hashlib.sha256(self.ref_path.read_bytes()).hexdigest()}},
                          "reference_groups": {"D_fit": {self.task["question_id"]: None}}},
            "shared_root": {"schema": "rag-rsi-shared-root-1",
                            "directory": str((self.base / "shared" / block).resolve()),
                            "block_id": block, "bank": "shared-root/" + block}}

    def out(self, block="block-0", arm="cases"):
        return self.base / "arms" / block / arm

    def loader(self, role):
        if role != "D_fit":
            raise AssertionError("paired pilot must not load held-out answers")
        self.loaded.append(role)
        return json.loads(self.ref_path.read_bytes())

    def runner(self, block="block-0", arm="cases", *, expansions=3,
               manifest=None, root_name="deepseek-flash"):
        manifest = self.manifest(block, arm, expansions) if manifest is None else deepcopy(manifest)
        prefix = "block/" + block + "/arm/" + arm + "/"
        projection = self.model(prefix + "develop")
        developer = evolution.ProgramDeveloper(
            projection, feedback_condition=arm, case_schedule=manifest["controls"]["case_schedule"],
            proposal_model_factory=lambda slot: self.model(prefix + "develop/proposal/" + str(slot)))
        return evolution.EvolutionRunner(
            self.out(block, arm), manifest, {"D_fit": [self.task]}, {},
            model_factory=lambda bank: self.model(prefix + bank), developer=developer,
            reference_loader=self.loader,
            root_model_factory=lambda bank: self.model("block/" + block + "/root/" + bank, name=root_name))

    def run_until(self, n, block="block-0", arm="cases", **kwargs):
        return self.runner(block, arm, **kwargs).run_search_until(n)

    def counts(self):
        return len(self.executed), len(self.transport.requests), self.ledger.summary()["used"].get("run", {}).get("calls", 0)

    def root_receipt(self, block="block-0", arm="cases"):
        return evolution.read(self.out(block, arm) / "measurements/shared_root.json")

    def shared_dir(self, block="block-0"):
        return self.base / "shared" / block

    def develop_payloads(self):
        return [payload for stage, payload in self.transport.requests if stage == "develop"]

    def assert_partial(self, result, n, block="block-0", arm="cases"):
        self.assertEqual(result["schema"], "rag-rsi-v3-search-progress-1")
        self.assertEqual(result["status"], "search_in_progress")
        self.assertEqual(result["completed_opportunities"], n)
        self.assertEqual(len(result["search"]["terminal_attempts"]), n)
        out = self.out(block, arm)
        for name in ("search_frozen.json", "phase_search.json", "phase_select.json",
                     "delivery_lock.json", "phase_report.json", "report.json"):
            self.assertFalse((out / name).exists(), name)

    def test_second_arm_root_reuses_verified_measurement_without_execute_or_request(self):
        first = self.run_until(0)
        self.assert_partial(first, 0)
        self.assertEqual(self.counts(), (1, 1, 1))
        second = self.run_until(0, arm="trace")
        self.assert_partial(second, 0, arm="trace")
        self.assertEqual(self.counts(), (1, 1, 1))
        left, right = self.root_receipt(), self.root_receipt(arm="trace")
        self.assertEqual(left["result"], right["result"])
        self.assertEqual(left["seal_sha256"], right["seal_sha256"])
        self.assertEqual(left["contract_sha256"], right["contract_sha256"])
        self.assertEqual(first["search"]["cards"][0], second["search"]["cards"][0])
        self.assertEqual(self.develop_payloads(), [])

    def test_shared_root_feedback_common_fields_are_exactly_equal(self):
        self.run_until(1)
        self.run_until(1, arm="trace")
        payloads = self.develop_payloads()
        self.assertEqual(len(payloads), 2)
        cases, trace = payloads
        self.assertEqual(cases["feedback"]["condition"], "cases")
        self.assertEqual(trace["feedback"]["condition"], "trace")
        self.assertEqual(cases["source_files"], trace["source_files"])
        self.assertEqual(cases["decision"], trace["decision"])
        self.assertEqual(cases["experience"], [])
        self.assertEqual(trace["experience"], [])
        left, right = deepcopy(cases["feedback"]), deepcopy(trace["feedback"])
        left.pop("condition"); right.pop("condition")
        for row in right["cases"]:
            row.pop("execution_flow"); row.pop("diagnostics")
        self.assertEqual(left, right)

    def test_different_block_repeats_root_measurement_and_provider_request(self):
        self.run_until(0, block="block-0")
        self.run_until(0, block="block-0", arm="trace")
        self.run_until(0, block="block-1")
        self.run_until(0, block="block-1", arm="trace")
        self.assertEqual(self.counts(), (2, 2, 2))
        self.assertEqual(self.transport.requests[0], self.transport.requests[1])
        self.assertNotEqual(self.executed[0]["bank"], self.executed[1]["bank"])
        a = self.root_receipt("block-0")["result"]
        b = self.root_receipt("block-1")["result"]
        self.assertNotEqual(a["identity_hash"], b["identity_hash"])
        self.assertNotEqual(self.shared_dir("block-0"), self.shared_dir("block-1"))

    def test_same_candidate_in_two_arms_has_distinct_measurement_and_request_banks(self):
        self.run_until(1)
        self.run_until(1, arm="trace")
        self.assertEqual(self.counts(), (3, 5, 5))  # shared root + two developers + two children
        root = evolution.read(self.out() / "root.json")["node_id"]
        children = [row for row in self.executed if row["node_id"] != root]
        self.assertEqual(len(children), 2)
        self.assertEqual(children[0]["node_id"], children[1]["node_id"])
        self.assertNotEqual(children[0]["bank"], children[1]["bank"])
        answers = [p for stage, p in self.transport.requests if stage == "answer"]
        self.assertEqual(answers[1], answers[2])

    def test_slot_interleaving_resume_and_duplicate_rejections_count_fixed_opportunities(self):
        expected = [(0, "cases"), (0, "trace"), (1, "cases"), (1, "trace"),
                    (2, "trace"), (2, "cases"), (3, "cases"), (3, "trace")]
        for n, arm in expected:
            with self.subTest(n=n, arm=arm):
                result = self.run_until(n, arm=arm)
                if n < 3:
                    self.assert_partial(result, n, arm=arm)
                else:
                    self.assertEqual(result["status"], "search_frozen")
                before = self.counts()
                self.assertEqual(self.run_until(n, arm=arm), result)
                self.assertEqual(self.counts(), before)
        self.assertEqual(self.counts(), (3, 9, 9))
        self.assertEqual(len(self.develop_payloads()), 6)
        for arm in ("cases", "trace"):
            search = evolution.read(self.out(arm=arm) / "search_frozen.json")
            self.assertEqual(len(search["cards"]), 2)
            self.assertEqual([a["step"] for a in search["terminal_attempts"]], [0, 1, 2])
            self.assertEqual([a["status"] for a in search["terminal_attempts"]],
                             ["measured", "rejected", "rejected"])
            self.assertTrue((self.out(arm=arm) / "phase_search.json").exists())
        self.assertEqual(set(self.loaded), {"D_fit"})

    def test_all_rejected_slots_finish_budget_without_creating_children(self):
        self.transport = RejectTransport()
        self.run_until(1)
        self.assertEqual(self.counts(), (1, 2, 2))
        result = self.run_until(3)
        attempts = result["search"]["terminal_attempts"]
        self.assertEqual(len(attempts), 3)
        self.assertTrue(all(a["status"] == "rejected" and a["node_id"] is None for a in attempts))
        self.assertEqual(len(result["search"]["cards"]), 1)
        self.assertEqual(self.counts(), (1, 4, 4))

    def test_partial_search_cannot_enter_selection_or_report(self):
        self.run_until(1)
        before = self.counts(), list(self.loaded)
        for phase in ("select", "report"):
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                self.runner().run_phase(phase)
        self.assertEqual((self.counts(), self.loaded), before)
        self.assertFalse((self.out() / "phase_search.json").exists())

    def test_invalid_slot_limit_does_no_new_work(self):
        for n in (True, -1, 4, 1.5, "1"):
            with self.subTest(n=n), self.assertRaises(ValueError):
                self.run_until(n)
        self.assertEqual(self.counts(), (0, 0, 0))
        self.assertEqual(self.loaded, [])

    def test_completed_search_can_resume_via_both_interfaces_without_new_work(self):
        result = self.run_until(3)
        before = self.counts(), len(self.loaded)
        self.assertEqual(self.runner().run_phase("search"), result)
        self.assertEqual(self.run_until(3), result)
        self.assertEqual((self.counts(), len(self.loaded)), before)

    def test_root_seal_survives_crash_before_local_receipt_without_repurchase(self):
        real_freeze = evolution.freeze
        def interrupted(path, value):
            if Path(path) == self.out() / "measurements/shared_root.json":
                raise InjectedPairedCrash("shared root frozen, local receipt absent")
            return real_freeze(path, value)
        with patch.object(evolution, "freeze", side_effect=interrupted):
            with self.assertRaises(InjectedPairedCrash):
                self.run_until(0)
        self.assertEqual(self.counts(), (1, 1, 1))
        self.assertTrue((self.shared_dir() / "shared_root_seal.json").exists())
        self.run_until(0)
        self.run_until(0, arm="trace")
        self.assertEqual(self.counts(), (1, 1, 1))

    def test_child_archive_crash_recovers_without_rebuying_developer(self):
        real_save = evolution.save
        def interrupted(path, value):
            if Path(path).name == "child.json":
                raise InjectedPairedCrash("archive complete, child receipt absent")
            return real_save(path, value)
        with patch.object(evolution, "save", side_effect=interrupted):
            with self.assertRaises(InjectedPairedCrash):
                self.run_until(1)
        self.assertEqual(self.counts(), (1, 2, 2))
        self.assertEqual(len(self.develop_payloads()), 1)
        result = self.run_until(1)
        self.assert_partial(result, 1)
        self.assertEqual(self.counts(), (2, 3, 3))
        self.assertEqual(len(self.develop_payloads()), 1)

    def test_sealed_root_cell_deletion_cannot_trigger_remeasurement(self):
        self.run_until(0)
        cell = next(self.shared_dir().rglob("measured.json"))
        cell.unlink()
        before = self.counts()
        with self.assertRaises(ValueError):
            self.run_until(0, arm="trace")
        self.assertEqual(self.counts(), before)

    def test_sealed_root_cell_mutation_cannot_trigger_remeasurement(self):
        self.run_until(0)
        cell = next(self.shared_dir().rglob("measured.json"))
        record = evolution.read(cell)
        record["payload"]["score"] = .123
        record["payload_sha256"] = digest(record["payload"])
        save(cell, record)
        before = self.counts()
        with self.assertRaises(ValueError):
            self.run_until(0, arm="trace")
        self.assertEqual(self.counts(), before)

    def test_sealed_root_model_identity_drift_rejected_before_execute(self):
        self.run_until(0)
        before = self.counts()
        with self.assertRaises(ValueError):
            self.run_until(0, arm="trace", root_name="different-synthetic-model")
        self.assertEqual(self.counts(), before)

    def test_shared_directory_cannot_be_relabelled_as_another_block(self):
        self.run_until(0)
        changed = self.manifest("block-1", "trace")
        changed["shared_root"]["directory"] = str(self.shared_dir().resolve())
        before = self.counts()
        with self.assertRaises(ValueError):
            self.run_until(0, "block-1", "trace", manifest=changed)
        self.assertEqual(self.counts(), before)

    def test_local_shared_receipt_mutation_rejected_before_next_slot(self):
        self.run_until(1)
        path = self.out() / "measurements/shared_root.json"
        receipt = evolution.read(path)
        receipt["result"]["score"] = .123
        save(path, receipt)
        before = self.counts()
        with self.assertRaises(ValueError):
            self.run_until(2)
        self.assertEqual(self.counts(), before)

    def test_partial_attempt_tampering_does_not_buy_next_proposal(self):
        self.run_until(1)
        path = self.out() / "steps/0/attempt.json"
        value = evolution.read(path)
        value["step"] = 2
        save(path, value)
        before = self.counts()
        with self.assertRaises(ValueError):
            self.run_until(2)
        self.assertEqual(self.counts(), before)

    def test_missing_measured_child_in_frozen_partial_cannot_be_rebought(self):
        self.run_until(1)
        cells = list((self.out() / "measurements").rglob("measured.json"))
        self.assertEqual(len(cells), 1)
        cells[0].unlink()
        before = self.counts()
        with self.assertRaises(ValueError):
            self.run_until(2)
        self.assertEqual(self.counts(), before)
        self.assertFalse((self.out() / "steps/1/decision.json").exists())

    def test_prior_saved_prefixes_remain_read_only_after_later_progress(self):
        expected = {n: self.run_until(n) for n in (0, 1, 2)}
        progress = self.out() / "search_progress"
        before_bytes = {p.name: p.read_bytes() for p in progress.glob("*.json")}
        self.assertEqual(set(before_bytes), {"0.json", "1.json", "2.json"})
        before = self.counts(), len(self.loaded)
        for n in (0, 1):
            with self.subTest(prefix=n):
                result = self.run_until(n)
                self.assertEqual(result, expected[n])
                self.assert_partial(result, n)
        self.assertEqual((self.counts(), len(self.loaded)), before)
        self.assertEqual({p.name: p.read_bytes() for p in progress.glob("*.json")}, before_bytes)


if __name__ == "__main__":
    unittest.main()
