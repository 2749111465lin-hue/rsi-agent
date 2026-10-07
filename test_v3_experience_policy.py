"""Offline tests for the v3 host-only deterministic experience policy."""
import copy
import math
import unittest

from code_rsi.v3.experience_policy import choose_next, memory_for_action


def card(node, score=0.5, *, step=0, module=None, parents=None, **overrides):
    value = {
        "node_id": node, "program_id": "program-" + node, "step": step,
        "role": "D_fit", "panel_hash": "panel-A", "evaluator_epoch": "eval-A",
        "complete": True, "valid_program": True, "score": score,
        "operator": "Improve" if parents else "Draft",
        "parent_node_ids": parents or [], "target_module": module,
    }
    value.update(overrides)
    return value


def choose(cards, step=1, modules=("retrieval", "answer_generation")):
    return choose_next(cards, step=step, panel_hash="panel-A",
                       evaluator_epoch="eval-A", allowed_modules=modules)


class ExperiencePolicyTests(unittest.TestCase):
    def test_empty_history_drafts_and_rotates(self):
        a, b = choose([], 0), choose([], 1)
        self.assertEqual(a["operator"], "Draft")
        self.assertIsNone(a["parent_node_id"])
        self.assertEqual(a["target_module"], "retrieval")
        self.assertEqual(b["target_module"], "answer_generation")

    def test_best_answer_parent_ignores_retrieval_proxy(self):
        cards = [card("n1", 0.7, retrieval_score=0.1),
                 card("n2", 0.5, retrieval_score=1.0)]
        self.assertEqual(choose(cards)["parent_node_id"], "n1")

    def test_explicit_proxy_score_cannot_be_parent(self):
        cards = [card("n1", 0.6), card("n2", 1.0, score_kind="retrieval_proxy")]
        self.assertEqual(choose(cards)["parent_node_id"], "n1")

    def test_negative_gain_is_retained(self):
        cards = [card("p", 0.8), card("c", 0.5, step=1, module="retrieval", parents=["p"])]
        result = choose(cards)
        stats = result["diagnostics"]["module_statistics"]["retrieval"]
        self.assertAlmostEqual(stats["mean_signed_gain"], -0.3)
        self.assertAlmostEqual(stats["signed_gains"][0], -0.3)
        self.assertEqual(result["target_module"], "answer_generation")

    def test_signed_gain_is_recomputed_from_legal_parent(self):
        cards = [card("p", 0.8),
                 card("c", 0.5, module="retrieval", parents=["p"], signed_delta_vs_best_parent=0.9)]
        stats = choose(cards)["diagnostics"]["module_statistics"]["retrieval"]
        self.assertAlmostEqual(stats["mean_signed_gain"], -0.3)

    def test_report_select_epoch_and_panel_are_isolated(self):
        cards = [card("p", 0.4),
                 card("report", 1, role="D_report"),
                 card("select", 1, role="D_select"),
                 card("old", 1, evaluator_epoch="eval-old"),
                 card("foreign", 1, panel_hash="panel-B")]
        result = choose(cards)
        self.assertEqual(result["parent_node_id"], "p")
        self.assertEqual(result["diagnostics"]["accepted_cards"], 1)
        self.assertEqual(result["diagnostics"]["rejected_cards"]["non_fit_role"], 2)
        self.assertEqual(result["diagnostics"]["rejected_cards"]["identity_mismatch"], 2)

    def test_conflicting_role_fields_rejected(self):
        self.assertEqual(choose([card("bad", split="D_report")])["operator"], "Draft")

    def test_missing_evaluator_epoch_rejected(self):
        candidate = card("old")
        candidate.pop("evaluator_epoch")
        self.assertEqual(choose([candidate])["operator"], "Draft")

    def test_non_host_source_rejected(self):
        self.assertEqual(choose([card("bad", source="candidate")])["operator"], "Draft")

    def test_conflicting_node_identity_rejected(self):
        result = choose([card("n", 0.3), card("n", 0.8)])
        self.assertEqual(result["operator"], "Draft")
        self.assertEqual(result["diagnostics"]["rejected_cards"]["conflicting_node_identity"], 2)

    def test_duplicate_replay_is_not_extra_evidence(self):
        p = card("p", 0.5)
        c = card("c", 0.6, parents=["p"], module="retrieval")
        result = choose([p, c, copy.deepcopy(c)])
        self.assertEqual(result["diagnostics"]["module_statistics"]["retrieval"]["gain_samples"], 1)

    def test_deterministic_under_input_permutation(self):
        cards = [card("p", 0.5), card("c", 0.65, step=1, module="retrieval", parents=["p"])]
        self.assertEqual(choose(cards), choose(list(reversed(cards))))
        self.assertEqual(choose(cards), choose(cards))

    def test_cheap_equal_effect_module_is_preferred(self):
        cards = [card("p", 0.4),
                 card("a", 0.6, module="retrieval", parents=["p"], resource_usage={"seconds": 1}),
                 card("b", 0.6, module="answer_generation", parents=["p"], resource_usage={"seconds": 100})]
        self.assertEqual(choose(cards)["target_module"], "retrieval")

    def test_missing_cost_is_not_free(self):
        cards = [card("p", 0.4),
                 card("a", 0.6, module="retrieval", parents=["p"], resource_usage={"seconds": 10}),
                 card("b", 0.6, module="answer_generation", parents=["p"])]
        result = choose(cards)
        self.assertEqual(result["target_module"], "retrieval")
        stats = result["diagnostics"]["module_statistics"]["answer_generation"]
        self.assertEqual(stats["missing_cost_count"], 1)
        self.assertEqual(stats["mean_effective_cost"], 20)

    def test_more_cost_cannot_make_negative_gain_better(self):
        cards = [card("p", 0.8),
                 card("a", 0.6, module="retrieval", parents=["p"], resource_usage={"seconds": 1}),
                 card("b", 0.6, module="answer_generation", parents=["p"], resource_usage={"seconds": 100})]
        result = choose(cards)
        stats = result["diagnostics"]["module_statistics"]
        self.assertGreater(stats["retrieval"]["utility"], stats["answer_generation"]["utility"])
        self.assertEqual(result["target_module"], "retrieval")

    def test_latest_explicit_failure_triggers_debug(self):
        failure = card("bad", None, step=3, module="retrieval",
                       complete=False, valid_program=False, failure_classes={"retrieve_error": 1})
        result = choose([card("p", 0.9), failure])
        self.assertEqual(result["parent_node_id"], "bad")
        self.assertEqual(result["operator"], "Debug")
        self.assertEqual(result["target_module"], "retrieval")
        self.assertIsNone(result["diagnostics"]["parent_answer_score"])

    def test_list_and_tuple_failure_classes_trigger_debug_for_invalid_measurement(self):
        for labels in (["retrieve_error"], ("retrieve_error",)):
            with self.subTest(labels=labels):
                failed = card("failed", None, complete=False, valid_program=False,
                              failure_classes=labels, module="retrieval")
                result = choose([failed])
                self.assertEqual(result["operator"], "Debug")
                self.assertEqual(result["diagnostics"]["failure_context"], ["retrieve_error"])

    def test_list_semantic_failure_classes_keep_valid_answer_in_improve(self):
        valid = card("p", 0.7, failure_classes=["missing_evidence"],
                     failure_assessment_source="model_assessed")
        result = choose([valid])
        self.assertEqual(result["operator"], "Improve")
        self.assertEqual(result["diagnostics"]["failure_context"], ["missing_evidence"])

    def test_semantic_diagnostics_do_not_turn_valid_answer_into_debug(self):
        valid = card("p", 0.7, failure_classes={"missing_evidence": 1},
                     failure_assessment_source="model_assessed")
        self.assertEqual(choose([valid])["operator"], "Improve")

    def test_negative_gain_alternative_parent_is_retained_for_exploration(self):
        cards = [card("p", 0.8, step=0, behavior={"group_hash": "base"}),
                 card("worse", 0.6, step=1, parents=["p"], module="retrieval",
                      behavior={"group_hash": "different"})]
        self.assertEqual(choose(cards, 1)["parent_node_id"], "p")
        result = choose(cards, 5)
        self.assertEqual(result["parent_node_id"], "worse")
        self.assertEqual(result["diagnostics"]["incumbent_node_id"], "p")
        self.assertIn("underexpanded_parent_exploration", result["reason"])
        self.assertAlmostEqual(result["diagnostics"]["module_statistics"]["retrieval"]["mean_signed_gain"], -0.2)

    def test_unmeasured_without_failure_is_not_debugged(self):
        result = choose([card("pending", None, complete=False, valid_program=False)])
        self.assertEqual(result["operator"], "Draft")

    def test_successful_repair_resolves_ancestor_failure(self):
        bad = card("bad", None, complete=False, valid_program=False, failure="exception")
        repaired = card("repair", 0.7, step=2, parents=["bad"], module="retrieval", operator="Debug")
        result = choose([bad, repaired])
        self.assertEqual(result["operator"], "Improve")
        self.assertEqual(result["parent_node_id"], "repair")
        self.assertEqual(result["diagnostics"]["module_statistics"]["retrieval"]["gain_samples"], 0)

    def test_same_failure_family_controls_action_statistics(self):
        cards = [
            card("p", 0.4, failure_classes={"missing_bridge": 1}),
            card("r", 0.6, step=1, parents=["p"], module="retrieval", failure_classes={"missing_bridge": 1}),
            card("other", 0.1, failure_classes={"format_failure": 1}),
            card("noise", 0.5, step=2, parents=["other"], module="answer_generation"),
        ]
        result = choose(cards)
        self.assertEqual(result["parent_node_id"], "r")
        self.assertEqual(result["target_module"], "retrieval")
        self.assertEqual(result["diagnostics"]["module_statistics"]["answer_generation"]["gain_samples"], 0)

    def test_no_cross_identity_parent_gain(self):
        cards = [card("p", 0.1, role="D_report"),
                 card("c", 0.9, parents=["p"], module="retrieval")]
        stats = choose(cards)["diagnostics"]["module_statistics"]["retrieval"]
        self.assertEqual(stats["gain_samples"], 0)

    def test_deterministic_exploration_even_when_known_gain_is_positive(self):
        cards = [card("p", 0.4),
                 card("r", 0.6, parents=["p"], module="retrieval")]
        self.assertEqual(choose(cards, 4)["target_module"], "answer_generation")
        self.assertEqual(choose(cards, 1)["target_module"], "retrieval")

    def test_memory_is_action_specific_newest_first_and_fit_only(self):
        cards = [
            card("parent", 0.7),
            card("r1", 0.6, step=1, parents=["parent"], module="retrieval"),
            card("a2", 0.6, step=2, parents=["parent"], module="answer_generation"),
            card("r3", 0.65, step=3, parents=["parent"], module="retrieval"),
            card("select", 1, step=100, role="D_select", module="retrieval"),
        ]
        decision = choose(cards, modules=("retrieval",))
        memory = memory_for_action(cards, decision)
        self.assertEqual([c["node_id"] for c in memory], ["r3", "r1", "parent"])
        self.assertEqual(decision["experience_ids"], ["r3", "r1", "parent"])
        self.assertTrue(all(c["role"] == "D_fit" for c in memory))

    def test_debug_memory_includes_latest_failure_and_previous_repairs(self):
        cards = [
            card("p", 0.6),
            card("old", None, step=1, module="retrieval", parents=["p"],
                 complete=False, valid_program=False, failure_classes={"retrieve_error": 1}),
            card("repair", 0.65, step=2, module="retrieval", parents=["old"], operator="Debug"),
            card("new", None, step=3, module="retrieval", parents=["repair"],
                 complete=False, valid_program=False, failure_classes={"retrieve_error": 1}),
        ]
        decision = choose(cards)
        memory = memory_for_action(cards, decision, 2)
        self.assertEqual([c["node_id"] for c in memory], ["new", "repair"])
        self.assertEqual(decision["operator"], "Debug")

    def test_memory_does_not_mutate_inputs(self):
        original = card("p", 0.6, resource_usage={"seconds": 4})
        cards = [original]
        memory = memory_for_action(cards, choose(cards))
        memory[0]["resource_usage"]["seconds"] = 99
        self.assertEqual(original["resource_usage"]["seconds"], 4)

    def test_invalid_numeric_quality_and_input_validation(self):
        self.assertEqual(choose([card("bad", math.nan)])["operator"], "Draft")
        with self.assertRaises(ValueError):
            choose([], -1)
        with self.assertRaises(ValueError):
            choose([], modules=[])
        with self.assertRaises(ValueError):
            memory_for_action([], {}, 1)


if __name__ == "__main__":
    unittest.main()
