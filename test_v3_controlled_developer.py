"""Developer information controls and proposal-bank isolation; no external API."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from code_rsi.budget import Ledger, stable
from code_rsi.v3.evolution import ProgramDeveloper
from code_rsi.v3.execution import root_files
from code_rsi.v3.infrastructure import StructuredModel, UnknownProviderOutcome
from test_v3_feedback_conditions import panel
from test_v3_fixture_origin import bind_synthetic_origin
from test_v3_feedback_integration import RecordingModel


class ControlledDeveloperTests(unittest.TestCase):
    def setUp(self):
        runs = Path(__file__).parent / "runs"
        runs.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="controlled-developer-", dir=runs)
        self.root = Path(temporary.name)
        self.addCleanup(temporary.cleanup)
        self.ledger = Ledger(self.root / "ledger.jsonl", {"run": {"calls": 20, "cny": 10}})
        self.sent, self.factory_slots, self.instances = [], [], {}
        self.result, self.tasks, self.schedule = panel()
        self.program = {"files": root_files({"max_rounds": 1})}
        self.decision = {"operator": "Improve", "target_module": "query_rewrite",
                         "intended_target_module": "query_rewrite", "proposal_slot": 0,
                         "step": 19, "reason": "REASON_SENTINEL", "parent_node_id": "PARENT_ID_SENTINEL",
                         "history": ["HISTORY_SENTINEL"], "priors": {"secret": "PRIOR_SENTINEL"},
                         "diagnostics": {"secret": "DECISION_DIAGNOSTIC_SENTINEL"},
                         "rejected_proposals": ["REJECTION_SENTINEL"]}
        self.experience = [{"mechanism": "EXPERIENCE_SENTINEL", "hypothesis": "HYPOTHESIS_SENTINEL"}]
        self.projection = self.model("projection")

    def transport(self, body):
        self.sent.append(deepcopy(body))
        payload = json.loads(body["messages"][-1]["content"])
        response = {"writes": {"rag.py": root_files({"max_rounds": 2})["rag.py"]},
                    "target_module": payload["decision"]["target_module"], "mechanism": "Synthetic reusable edit"}
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(response)}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 100}}

    def model(self, bank):
        return StructuredModel(self.root / "requests", self.ledger, self.transport, bank=bank,
                               prices={"input_miss": 2, "input_hit": .04, "output": 8})

    def factory(self, slot):
        self.factory_slots.append(slot)
        model = self.model("proposal-slot-" + str(slot))
        self.instances[slot] = model
        return model

    def developer(self, condition="aggregate", *, factory=True, projection=None):
        return ProgramDeveloper(self.projection if projection is None else projection,
                                feedback_condition=condition, case_schedule=self.schedule,
                                proposal_model_factory=self.factory if factory else None)

    def prepare(self, developer, decision=None):
        return developer.prepare_request(self.program, self.decision if decision is None else decision,
                                         self.experience, self.result, self.tasks)

    def propose(self, developer, decision=None):
        return developer.propose(self.program, self.decision if decision is None else decision,
                                 self.experience, self.result, self.tasks,
                                 {"reference": "PRIVATE_REFERENCE_SENTINEL"})

    def test_actual_aggregate_provider_body_has_no_feedback_side_channels(self):
        self.propose(self.developer())
        self.assertEqual(len(self.sent), 1)
        payload = json.loads(self.sent[0]["messages"][-1]["content"])
        self.assertEqual(set(payload), {"source_files", "decision", "experience", "feedback", "edit_boundary"})
        self.assertEqual(payload["source_files"], self.program["files"])
        self.assertEqual(payload["decision"], {"operator": "Improve", "target_module": "query_rewrite", "intended_target_module": "query_rewrite"})
        self.assertEqual(payload["experience"], [])
        self.assertNotIn("cases", payload["feedback"])
        self.assertNotIn("trace", payload["feedback"])
        serialized = stable(payload)
        for marker in ("REASON_SENTINEL", "PARENT_ID_SENTINEL", "HISTORY_SENTINEL", "PRIOR_SENTINEL",
                       "DECISION_DIAGNOSTIC_SENTINEL", "REJECTION_SENTINEL", "EXPERIENCE_SENTINEL",
                       "HYPOTHESIS_SENTINEL", "PRIVATE_REFERENCE_SENTINEL"):
            self.assertNotIn(marker, serialized)
        self.assertNotIn("proposal_slot", payload["decision"])
        self.assertEqual(self.projection.calls, 0)

    def test_cases_and_trace_common_payload_fields_and_selected_repeats_match(self):
        left, right = self.prepare(self.developer("cases")), self.prepare(self.developer("trace"))
        for key in ("source_files", "decision", "experience", "edit_boundary"):
            self.assertEqual(left[key], right[key])
        for old, new in zip(left["feedback"]["cases"], right["feedback"]["cases"]):
            self.assertEqual(old, {key: new[key] for key in old})
        self.assertEqual([(c["question_id"], c["repeat"]) for c in left["feedback"]["cases"]], [("q2", 1), ("q1", 0)])
        self.assertEqual(self.factory_slots, [])
        self.assertEqual(self.sent, [])

    def test_prepare_and_snapshot_never_call_factory_or_dispatch(self):
        developer = self.developer("cases")
        self.prepare(developer)
        snapshot = developer.configuration_snapshot()
        self.assertEqual(snapshot, {"feedback_condition": "cases", "case_schedule": self.schedule, "proposal_model_factory_present": True})
        stable(snapshot)
        snapshot["case_schedule"][0]["repeat"] = 7
        self.schedule[0]["repeat"] = 6
        self.assertEqual(developer.configuration_snapshot()["case_schedule"][0]["repeat"], 1)
        self.assertEqual(self.factory_slots, [])
        self.assertEqual(self.sent, [])
        self.assertEqual(self.ledger.events, [])

    def test_controlled_propose_without_factory_is_rejected_before_dispatch(self):
        developer = self.developer(factory=False)
        self.prepare(developer)
        with self.assertRaisesRegex(ValueError, "slot model factories"):
            self.propose(developer)
        self.assertEqual(self.sent, [])
        self.assertEqual(self.ledger.events, [])

    def test_slot_is_host_only_and_factory_called_once_for_same_slot(self):
        developer = self.developer()
        first = self.propose(developer)
        second = self.propose(developer)
        self.assertEqual(first, second)
        self.assertEqual(self.factory_slots, [0])
        self.assertEqual(len(self.sent), 1)
        self.assertNotIn("proposal_slot", self.prepare(developer)["decision"])
        self.assertNotIn("proposal_slot", json.loads(self.sent[0]["messages"][-1]["content"])["decision"])

    def test_different_slots_same_body_have_distinct_physical_banks(self):
        developer = self.developer()
        self.propose(developer)
        self.propose(developer, {**self.decision, "proposal_slot": 1})
        self.assertEqual(self.factory_slots, [0, 1])
        self.assertEqual(len(self.sent), 2)
        self.assertEqual(self.sent[0], self.sent[1])
        self.assertNotEqual(self.instances[0].bank, self.instances[1].bank)
        recovered = self.developer()
        self.propose(recovered)
        self.assertEqual(self.factory_slots, [0, 1, 0])
        self.assertEqual(len(self.sent), 2)

    def test_slot_cannot_resume_with_a_changed_model_request(self):
        developer = self.developer()
        self.propose(developer)
        with self.assertRaisesRegex(ValueError, "different request"):
            self.propose(developer, {**self.decision, "operator": "Debug"})
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.factory_slots, [0])

    def test_invalid_slot_never_reaches_factory(self):
        developer = self.developer()
        for value in (True, -1, "1", None):
            with self.subTest(slot=value), self.assertRaises(ValueError):
                self.propose(developer, {**self.decision, "proposal_slot": value})
        self.assertEqual(self.factory_slots, [])
        self.assertEqual(self.sent, [])

    def test_targets_must_agree_and_controlled_decision_fields_are_enumerated(self):
        developer = self.developer()
        for updates in ({"intended_target_module": "retrieval"}, {"target_module": "arbitrary"}, {"operator": "arbitrary free text"}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                self.prepare(developer, {**self.decision, **updates})
        decision = {key: value for key, value in self.decision.items() if key != "intended_target_module"}
        self.assertEqual(self.prepare(developer, decision)["decision"]["intended_target_module"], "query_rewrite")

    def test_controlled_overflow_rejects_without_legacy_crop_or_input_changes(self):
        row = self.result["rows"][-1]
        row["answer"] = "汉" * 50000
        row["host_evidence_trace"]["final_observations"][-1]["response"]["answer"] = row["answer"]
        self.result["rows"][-1] = bind_synthetic_origin(row)
        before = deepcopy(self.result)
        for condition in ("cases", "trace"):
            with patch("code_rsi.v3.evolution._fit_development_request", side_effect=AssertionError("asymmetric crop")), \
                    self.assertRaisesRegex(ValueError, "no cases are cropped"):
                self.propose(self.developer(condition))
        self.assertEqual(self.result, before)
        self.assertEqual(self.sent, [])
        self.assertEqual(self.factory_slots, [])

    def test_scripted_preparation_without_sizer_is_allowed_but_propose_is_not(self):
        projection = RecordingModel("query_rewrite")
        developer = self.developer("cases", projection=projection)
        self.prepare(developer)
        with self.assertRaisesRegex(ValueError, "request-size contract"):
            self.propose(developer)
        self.assertEqual(projection.calls, [])
        self.assertEqual(self.sent, [])

    def test_partial_or_invalid_sizing_contract_is_rejected_during_prepare(self):
        class Partial:
            max_input_bytes = 100000
        class BadSize:
            max_input_bytes = 100000
            identity = "synthetic"
            def request_size(self, stage, payload): return True
        for model in (Partial(), BadSize()):
            with self.subTest(model=type(model).__name__), self.assertRaises(ValueError):
                self.prepare(self.developer(projection=model))

    def test_factory_model_identity_cap_and_size_must_match_projection(self):
        for mismatch in ("identity", "limit", "size"):
            actual = self.model("mismatch-" + mismatch)
            if mismatch == "identity": actual.identity = "different"
            elif mismatch == "limit": actual.max_input_bytes -= 1
            else:
                original = actual.request_size
                actual.request_size = lambda stage, payload, original=original: original(stage, payload) + 1
            developer = ProgramDeveloper(self.projection, feedback_condition="aggregate", case_schedule=self.schedule,
                                         proposal_model_factory=lambda slot, actual=actual: actual)
            with self.subTest(mismatch=mismatch), self.assertRaises(ValueError):
                self.propose(developer)
        self.assertEqual(self.sent, [])

    def test_equal_sized_but_changed_wire_body_is_rejected(self):
        actual = self.model("same-size-other-body")
        original = actual.request_body
        def altered(stage, payload):
            body = original(stage, payload)
            body["temperature"] = 1
            return body
        actual.request_body = altered
        developer = ProgramDeveloper(self.projection, feedback_condition="aggregate", case_schedule=self.schedule,
                                     proposal_model_factory=lambda slot: actual)
        with self.assertRaisesRegex(ValueError, "provider body"):
            self.propose(developer)
        self.assertEqual(self.sent, [])

    def test_unknown_physical_outcome_is_not_retried_by_slot_factory(self):
        physical = []
        def unknown(body):
            physical.append(body)
            raise TimeoutError("unknown")
        actual = self.model("unknown-slot")
        actual.transport = unknown
        developer = ProgramDeveloper(self.projection, feedback_condition="aggregate", case_schedule=self.schedule,
                                     proposal_model_factory=lambda slot: actual)
        for _ in range(2):
            with self.assertRaises(UnknownProviderOutcome):
                self.propose(developer)
        self.assertEqual(len(physical), 1)

    def test_rich_defaults_preserve_legacy_information_and_need_no_factory(self):
        model = RecordingModel("query_rewrite")
        developer = ProgramDeveloper(model)
        self.propose(developer)
        payload = model.calls[0][1]
        self.assertEqual(payload["experience"], self.experience)
        self.assertEqual(payload["decision"]["reason"], "REASON_SENTINEL")
        self.assertIn("module_priors", payload["feedback"])
        self.assertNotIn("proposal_slot", payload["decision"])
        self.assertEqual(developer.configuration_snapshot(), {"feedback_condition": "rich", "case_schedule": [], "proposal_model_factory_present": False})
        self.assertEqual(ProgramDeveloper(model, case_schedule=[]).configuration_snapshot(), developer.configuration_snapshot())

    def test_rich_can_use_slot_factory_without_disclosing_slot(self):
        developer = ProgramDeveloper(self.projection, proposal_model_factory=self.factory)
        self.assertNotIn("proposal_slot", self.prepare(developer)["decision"])
        self.propose(developer)
        self.assertEqual(self.factory_slots, [0])
        payload = json.loads(self.sent[0]["messages"][-1]["content"])
        self.assertNotIn("proposal_slot", payload["decision"])
        self.assertEqual(payload["experience"], self.experience)

    def test_invalid_constructor_controls_fail_before_any_calls(self):
        for kwargs in ({"feedback_condition": "unsupported"}, {"feedback_condition": "aggregate"},
                       {"feedback_condition": "aggregate", "case_schedule": []},
                       {"case_schedule": self.schedule}, {"proposal_model_factory": object()}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ProgramDeveloper(self.projection, **kwargs)
        self.assertEqual(self.sent, [])


if __name__ == "__main__":
    unittest.main()
