"""Terminal-opportunity coverage and optional mechanism memory; zero API."""
from copy import deepcopy
import unittest

from code_rsi.v3.experience_policy import choose_next, memory_for_action, DEFAULT_MODULES
from code_rsi.v3.execution import root_files
from code_rsi.v3.edit_scope import observe_edit_scope
from test_v3_experience_policy import card


def decide(cards, step, attempts=None, policy="experience_coverage_v1"):
    return choose_next(cards, step=step, panel_hash="panel-A", evaluator_epoch="eval-A",
                       module_policy=policy, attempt_history=attempts)


def attempt(i, target, parent="p", node=None, **updates):
    return {"attempt_id": "attempt-" + str(i), "step": i, "role": "D_fit",
            "panel_hash": "panel-A", "evaluator_epoch": "eval-A", "parent_node_id": parent,
            "intended_target_module": target, "status": "measured" if node else "rejected",
            "node_id": node, **updates}


def root():
    return card("p", .8, source="host", failure_classes=["answer_quality"],
                diagnostics={"module_priors": {"evidence_selection": .9, "query_rewrite": .2}})


class ExperienceCoverageTests(unittest.TestCase):
    def test_rejections_cover_all_modules_and_do_not_become_nodes_or_gains(self):
        parent, attempts, choices = root(), [], []
        for step in range(8):
            d = decide([parent], step, attempts)
            choices.append(d["target_module"])
            self.assertEqual(d["parent_node_id"], "p")
            self.assertEqual(d["diagnostics"]["accepted_cards"], 1)
            self.assertEqual(sum(s["gain_samples"] for s in d["diagnostics"]["module_statistics"].values()), 0)
            attempts.append(attempt(step, d["target_module"]))
        self.assertEqual(set(choices[:4]), set(DEFAULT_MODULES))
        self.assertEqual({m: choices.count(m) for m in DEFAULT_MODULES}, {m: 2 for m in DEFAULT_MODULES})

    def test_growing_mixed_history_counts_opportunities_but_not_module_gains(self):
        cards, attempts, choices = [root()], [], []
        for step in range(8):
            d = decide(cards, step, attempts)
            target = d["target_module"]
            choices.append(target)
            scope = observe_edit_scope(root_files(), root_files({"prompts": {"read": "Synthetic " + str(step)}}), target)
            self.assertEqual(scope["attribution"], "mixed")
            node = "child" + str(step)
            cards.append(card(node, .4, step=step + 1, parents=[d["parent_node_id"]],
                source="host", intended_target_module=target, actual_edit_scope=scope,
                failure_classes=["answer_quality"]))
            attempts.append(attempt(step, target, parent=d["parent_node_id"], node=node))
        self.assertEqual(set(choices[:4]), set(DEFAULT_MODULES))
        self.assertEqual({m: choices.count(m) for m in DEFAULT_MODULES}, {m: 2 for m in DEFAULT_MODULES})
        self.assertEqual(sum(s["gain_samples"] for s in d["diagnostics"]["module_statistics"].values()), 0)

    def test_declared_opportunity_does_not_relabel_actual_module_gain(self):
        target = "query_rewrite"
        scope = observe_edit_scope(root_files(), root_files({"prompts": {"answer": "Synthetic answer prompt"}}), target)
        child = card("c", .9, step=1, parents=["p"], source="host",
                     intended_target_module=target, actual_edit_scope=scope)
        d = decide([root(), child], 1, [attempt(0, target, node="c")])
        stats = d["diagnostics"]["module_statistics"]
        self.assertEqual(stats["answer_generation"]["gain_samples"], 1)
        self.assertEqual(stats["query_rewrite"]["gain_samples"], 0)
        self.assertEqual(d["diagnostics"]["module_opportunity_counts"]["query_rewrite"], 1)
        self.assertEqual(d["diagnostics"]["module_opportunity_counts"]["answer_generation"], 0)
        self.assertAlmostEqual(stats["answer_generation"]["signed_gains"][0], .1)

    def test_duplicate_receipt_does_not_add_an_opportunity(self):
        record = attempt(0, "retrieval")
        d = decide([root()], 1, [record, deepcopy(record)])
        self.assertEqual(d["diagnostics"]["module_opportunity_counts"]["retrieval"], 1)
        self.assertEqual(d["diagnostics"]["attempt_history"]["duplicate_replays"], 1)

    def test_conflicting_id_or_step_is_rejected(self):
        record = attempt(0, "retrieval")
        for changed in ({**record, "intended_target_module": "answer_generation"},
                        {**record, "attempt_id": "another-id"}):
            with self.assertRaises(ValueError):
                decide([root()], 1, [record, changed])

    def test_foreign_future_nonterminal_or_fabricated_receipts_rejected(self):
        for updates in ({"role": "D_report"}, {"panel_hash": "another"},
                        {"evaluator_epoch": "old"}, {"step": 1}, {"step": True},
                        {"status": "unknown"}, {"parent_node_id": "nonexistent"},
                        {"status": "measured", "node_id": "missing"},
                        {"node_id": "p"}, {"extra_field": True}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                decide([root()], 1, [attempt(0, "retrieval", **updates)])

    def test_measured_attempt_must_match_its_real_parent_and_intent(self):
        c = card("c", .7, step=1, module="retrieval", parents=["p"])
        for row in (attempt(0, "answer_generation", node="c"),
                    attempt(0, "retrieval", parent="other", node="c"),
                    attempt(1, "retrieval", node="c")):
            with self.assertRaises(ValueError):
                decide([root(), card("other", .3), c], 2, [row])

    def test_none_history_derives_measured_child_opportunities(self):
        c = card("c", .7, step=1, module="retrieval", parents=["p"])
        derived = decide([root(), c], 1)
        explicit = decide([root(), c], 1, [attempt(0, "retrieval", node="c")])
        self.assertEqual(derived["target_module"], explicit["target_module"])
        self.assertEqual(derived["diagnostics"]["module_opportunity_counts"], explicit["diagnostics"]["module_opportunity_counts"])
        self.assertEqual(derived["diagnostics"]["attempt_history"]["source"], "derived_measured_children")

    def test_round_robin_ignores_priors_and_uses_declared_step(self):
        for step in range(8):
            d = decide([root()], step, [], "round_robin_v1")
            self.assertEqual(d["target_module"], DEFAULT_MODULES[step % len(DEFAULT_MODULES)])

    def test_debug_exception_is_shared_and_reported_without_changing_parent(self):
        failed = card("broken", None, step=1, parents=["p"], module="answer_generation",
                      valid_program=False, failure_classes=["answer_schema_failure"])
        for policy in ("legacy", "round_robin_v1", "experience_coverage_v1"):
            d = decide([root(), failed], 1, [attempt(0, "answer_generation", node="broken")], policy)
            self.assertEqual((d["parent_node_id"], d["operator"], d["target_module"]),
                             ("broken", "Debug", "answer_generation"))
            self.assertEqual(d["diagnostics"]["coverage_exception"], "debug_repair")

    def test_same_cards_keep_same_parent_policy_across_module_policies(self):
        cards = [root(), card("c", .9, step=1, module="retrieval", parents=["p"])]
        history = [attempt(0, "retrieval", node="c")]
        for step in (1, 5, 6, 10):
            decisions = [decide(cards, step, history, policy)
                         for policy in ("legacy", "round_robin_v1", "experience_coverage_v1")]
            self.assertEqual(len({(d["parent_node_id"], d["operator"]) for d in decisions}), 1)

    def test_unknown_policy_or_invalid_history_rejected(self):
        for kwargs in ({"policy": "random"}, {"attempts": {}}, {"attempts": "bad"}):
            with self.assertRaises(ValueError):
                decide([root()], 1, **kwargs)

    def test_opt_in_memory_exposes_bounded_hypothesis_and_mixed_negative_gain(self):
        p = root()
        scope = observe_edit_scope(root_files(), root_files({"prompts": {"read": "Mixed"}}), "evidence_selection")
        c = card("mixed", .4, step=1, parents=["p"], source="host",
                 intended_target_module="evidence_selection", actual_edit_scope=scope,
                 hypothesis="H" * 1700, failure_classes=["answer_quality"])
        d = decide([p, c], 1, [attempt(0, "evidence_selection", node="mixed")])
        old = memory_for_action([p, c], d)
        self.assertFalse(any("hypothesis" in row for row in old))
        self.assertNotIn("mixed", {row["node_id"] for row in old})
        new = memory_for_action([p, c], d, limit=1, include_mechanism=True)
        self.assertEqual(len(new), 1)
        self.assertEqual(new[0]["node_id"], "mixed")
        self.assertEqual(len(new[0]["hypothesis"]), 1600)
        self.assertTrue(new[0]["hypothesis_truncated"])
        self.assertEqual(new[0]["hypothesis_status"], "developer_claim_not_causal")
        self.assertEqual(new[0]["mechanism_scope"], "whole_program_unattributed")
        self.assertIsNone(new[0]["associated_module"])
        self.assertAlmostEqual(new[0]["signed_delta_vs_best_parent"], -.4)

    def test_mechanism_memory_keeps_role_epoch_filter_and_fixed_capacity(self):
        p = root()
        c = card("c", .7, step=1, module="retrieval", parents=["p"], hypothesis="Synthetic")
        d = decide([p, c], 1, [attempt(0, "retrieval", node="c")])
        foreign = card("foreign", 1, step=2, module="retrieval", parents=["p"], role="D_report")
        rows = memory_for_action([p, c, foreign], d, limit=1, include_mechanism=True)
        self.assertLessEqual(len(rows), 1)
        self.assertNotIn("foreign", {row["node_id"] for row in rows})
        self.assertEqual(memory_for_action([p], d, limit=0, include_mechanism=True), [])
        with self.assertRaises(ValueError):
            memory_for_action([p], d, include_mechanism="yes")


if __name__ == "__main__":
    unittest.main()
