"""Real host measurement with synthetic RPC, no providers, secrets or private data."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from code_rsi.v3 import execution
from code_rsi.v3.edit_scope import observe_edit_scope
from code_rsi.v3.evolution import ProgramDeveloper, experience_card
from code_rsi.v3.experience_policy import choose_next, memory_for_action
from test_v3_feedback_integration import fit_result, RecordingModel
from test_v3_experience_policy import scope_for_module
import test_v3_answer_origin as origin


class ExperienceEligibilityTests(unittest.TestCase):
    setUp = origin.AnswerOriginTests.setUp

    def pair(self):
        measure = execution.Measurement(self.archive, self.root / "measurements",
                  lambda bank: origin.ScriptedModel([origin.final(origin.WRONG)]))
        def run(node, answer):
            with patch.object(execution, "Sandbox", return_value=origin.ReturnSandbox(calls=1, returned=answer)):
                return measure.run(node, [self.task], {self.task["question_id"]: self.reference},
                                   role="D_fit", bank="synthetic")
        parent = run(self.node, origin.WRONG)
        files = execution.root_files({"search_limit": 4})
        child_node = self.archive.record(files, {}, session_id="eligibility", attempt=1,
                                         parent_node_id=self.node["node_id"])
        invalid = run(child_node, origin.EXPECTED)
        scope = observe_edit_scope(execution.root_files(), files, "retrieval")
        return parent, invalid, scope, child_node, run

    def make_card(self, result, parent=None, scope=None):
        return experience_card(result, parent, operator="Improve" if parent else "Draft", module="retrieval",
                               step=1 if parent else 0, mechanism="Synthetic source-scope change", edit_scope=scope)

    def test_high_raw_score_with_invalid_origin_is_not_a_learning_reward_or_gain(self):
        parent, invalid, scope, child_node, _ = self.pair()
        self.assertTrue(parent["valid_program"])
        self.assertEqual(parent["score"], 0.)
        self.assertEqual(invalid["score"], 1.)
        self.assertFalse(invalid["valid_program"])
        self.assertEqual(invalid["rows"][0]["answer_origin_status"], "candidate_answer_mismatch")
        base = self.make_card(parent)
        child = self.make_card(invalid, parent, scope)
        self.assertFalse(child["program_eligible"])
        self.assertFalse(child["paired_comparison_eligible"])
        for key in ("score", "signed_delta_vs_best_parent", "signed_deltas", "paired_deltas"):
            self.assertIsNone(child[key])
        self.assertIsNone(child["reward"]["answer_quality"])
        self.assertIsNone(child["reward"]["paired_signed_gain"])
        self.assertIsNone(child["diagnostics"]["paired_summary"])
        raw = child["raw_diagnostics"]
        self.assertEqual(raw["answer_score"], 1.)
        self.assertEqual(raw["signed_delta_vs_best_parent"], 1.)
        self.assertEqual(raw["paired_deltas"], {self.task["question_id"]: 1.})
        self.assertTrue(raw["diagnostic_only"] and raw["not_quality_reward"])
        decision = choose_next([base, child], step=1, panel_hash=parent["panel_hash"], evaluator_epoch=parent["evaluator_epoch"])
        self.assertEqual(decision["operator"], "Debug")
        self.assertEqual(decision["diagnostics"]["incumbent_node_id"], base["node_id"])
        stats = decision["diagnostics"]["module_statistics"]["retrieval"]
        self.assertEqual(stats["gain_samples"], 0)
        self.assertEqual(stats["failure_count"], 1)
        memory = memory_for_action([base, child], decision)
        failed = next(c for c in memory if c["node_id"] == child["node_id"])
        self.assertIsNone(failed["score"])
        self.assertIsNone(failed["signed_delta_vs_best_parent"])
        self.assertEqual(failed["raw_diagnostics"]["signed_delta_vs_best_parent"], 1.)
        model = RecordingModel(decision["target_module"])
        source = self.archive.load_program(child_node["program_id"])
        ProgramDeveloper(model).propose(source, decision, memory,
            {**invalid, "parent_measurement": parent}, [self.task], {})
        sent = next(c for c in model.calls[0][1]["experience"] if c["node_id"] == child["node_id"])
        self.assertIsNone(sent["score"])
        self.assertIsNone(sent["signed_delta_vs_best_parent"])
        self.assertFalse(sent["program_eligible"])
        self.assertEqual(sent["raw_diagnostics"]["answer_score"], 1.)

    def test_invalid_parent_also_blocks_gain_without_hiding_negative_raw_delta(self):
        _, invalid, _, child_node, run = self.pair()
        repair_node = self.archive.record(execution.root_files(), {}, session_id="eligibility", attempt=2,
                                          parent_node_id=child_node["node_id"])
        repair = run(repair_node, origin.WRONG)
        scope = observe_edit_scope(execution.root_files({"search_limit": 4}), execution.root_files(), "retrieval")
        card = self.make_card(repair, invalid, scope)
        self.assertTrue(card["program_eligible"])
        self.assertEqual(card["score"], 0.)
        self.assertEqual(card["reward"]["answer_quality"], 0.)
        self.assertFalse(card["paired_comparison_eligible"])
        self.assertIsNone(card["signed_delta_vs_best_parent"])
        self.assertIsNone(card["reward"]["paired_signed_gain"])
        self.assertIsNone(card["paired_deltas"])
        self.assertEqual(card["raw_diagnostics"]["signed_delta_vs_best_parent"], -1.)
        self.assertEqual(card["raw_diagnostics"]["paired_deltas"], {self.task["question_id"]: -1.})

    def test_legacy_schema_and_reported_valid_flag_cannot_promote_raw_gain(self):
        parent, invalid, scope, _, _ = self.pair()
        legacy = deepcopy(invalid)
        legacy["valid_program"] = True
        for row in legacy["rows"]:
            row["schema"] = "rag-rsi-v3-execution-2"
            for field in ("answer_origin_valid", "answer_origin_status", "host_answer_origin_validation"):
                row.pop(field, None)
        card = self.make_card(legacy, parent, scope)
        self.assertFalse(card["program_eligible"])
        self.assertFalse(card["valid_program"])
        self.assertFalse(card["paired_comparison_eligible"])
        self.assertIsNone(card["score"])
        self.assertIsNone(card["reward"]["answer_quality"])
        self.assertIsNone(card["reward"]["paired_signed_gain"])
        self.assertIsNone(card["signed_delta_vs_best_parent"])
        self.assertEqual(card["raw_diagnostics"]["answer_score"], 1.)
        self.assertEqual(card["raw_diagnostics"]["signed_delta_vs_best_parent"], 1.)
        self.assertTrue(card["raw_diagnostics"]["reported_valid_program"])
        self.assertIsNone(card["raw_diagnostics"]["host_program_eligibility"])
        decision = choose_next([self.make_card(parent), card], step=1,
                               panel_hash=parent["panel_hash"], evaluator_epoch=parent["evaluator_epoch"])
        self.assertEqual(decision["diagnostics"]["incumbent_node_id"], parent["node_id"])
        self.assertEqual(decision["diagnostics"]["module_statistics"]["retrieval"]["gain_samples"], 0)

    def test_valid_negative_gain_remains_a_signed_learning_sample(self):
        parent = fit_result({"synthetic-q": .8}, node="base")
        result = fit_result({"synthetic-q": .1}, node="regression")
        base = self.make_card(parent)
        child = self.make_card(result, parent, scope_for_module("retrieval"))
        self.assertTrue(child["program_eligible"] and child["paired_comparison_eligible"])
        self.assertAlmostEqual(child["signed_delta_vs_best_parent"], -.7)
        self.assertAlmostEqual(child["reward"]["paired_signed_gain"], -.7)
        self.assertAlmostEqual(child["paired_deltas"]["synthetic-q"], -.7)
        decision = choose_next([base, child], step=1, panel_hash=parent["panel_hash"], evaluator_epoch=parent["evaluator_epoch"])
        self.assertAlmostEqual(decision["diagnostics"]["module_statistics"]["retrieval"]["signed_gains"][0], -.7)


if __name__ == "__main__":
    unittest.main()
