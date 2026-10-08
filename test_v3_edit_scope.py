"""Synthetic source/receipt tests; no candidate execution, APIs or private data."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from code_rsi.budget import digest, save
from code_rsi.v3 import evolution
from code_rsi.v3.edit_scope import observe_edit_scope, validated_scope
from code_rsi.v3.execution import root_files
from code_rsi.v3.experience_policy import choose_next, memory_for_action
from test_v3_experience_policy import card
import test_v3_evolution_recovery as recovery


class EditScopeTests(unittest.TestCase):
    def scope(self, config, intended="query_rewrite"):
        return observe_edit_scope(root_files(), root_files(config), intended)

    def decision(self, scope, *, score=.8):
        base = card("base", .3)
        child = card("child", score, parents=["base"],
                     module=scope["intended_target_module"], actual_edit_scope=scope)
        decision = choose_next([base, child], step=1, panel_hash="panel-A", evaluator_epoch="eval-A")
        return decision, [base, child]

    def test_wrong_intent_label_does_not_own_actual_module_gain(self):
        scope = self.scope({"search_limit": 4})
        self.assertEqual(scope["attribution"], "single_module")
        self.assertEqual(scope["associated_module"], "retrieval")
        self.assertTrue(scope["intent_mismatch"])
        decision, cards = self.decision(scope)
        stats = decision["diagnostics"]["module_statistics"]
        self.assertEqual(stats["query_rewrite"]["gain_samples"], 0)
        self.assertEqual(stats["retrieval"]["gain_samples"], 1)
        self.assertAlmostEqual(stats["retrieval"]["mean_signed_gain"], .5)
        memory = next(c for c in memory_for_action(cards, decision) if c["node_id"] == "child")
        self.assertEqual(memory["intended_target_module"], "query_rewrite")
        self.assertEqual(memory["associated_module"], "retrieval")
        self.assertTrue(memory["actual_edit_scope"]["intent_mismatch"])
        self.assertFalse(memory["module_association_is_causal"])

    def test_cross_module_edit_keeps_whole_program_gain_and_parent_eligibility(self):
        scope = self.scope({"search_limit": 4, "max_answer_chars": 900})
        self.assertEqual(scope["attribution"], "mixed")
        self.assertEqual(scope["affected_modules"], ["answer_generation", "retrieval"])
        decision, cards = self.decision(scope)
        self.assertEqual(decision["parent_node_id"], "child")
        self.assertTrue(all(x["gain_samples"] == 0 for x in decision["diagnostics"]["module_statistics"].values()))
        memory = memory_for_action(cards, decision)
        self.assertAlmostEqual(memory[0]["signed_delta_vs_best_parent"], .5)
        self.assertIsNone(memory[0]["associated_module"])

    def test_orchestration_and_unknown_helpers_are_unknown(self):
        self.assertEqual(self.scope({"max_rounds": 2})["attribution"], "unknown")
        before = root_files()
        for name in ("query_rewrite", "answer_generation", "novel_helper"):
            after = {**before, "rag_core.py": before["rag_core.py"] + "\ndef " + name + "():\n    return 1\n"}
            scope = observe_edit_scope(before, after, "query_rewrite")
            self.assertEqual(scope["attribution"], "unknown")
            self.assertIsNone(scope["associated_module"])
            self.assertTrue(any(x["path"] == name for x in scope["changes"]))

    def test_nested_function_body_is_not_misreported_as_whole_solve(self):
        before = root_files()
        after = {**before, "rag_core.py": before["rag_core.py"].replace("            invalid = 0", "            invalid = 1")}
        scope = observe_edit_scope(before, after, "retrieval")
        self.assertEqual(scope["associated_module"], "evidence_selection")
        self.assertEqual([x["path"] for x in scope["changes"]], ["RagEngine.solve.consume_read"])
        self.assertTrue(scope["intent_mismatch"])

    def test_shared_search_body_and_read_prompt_are_mixed(self):
        before = root_files()
        after = {**before, "rag_core.py": before["rag_core.py"].replace("room // (len(active) - index)", "room // (len(active) - index + 1)")}
        scope = observe_edit_scope(before, after, "retrieval")
        self.assertEqual(scope["attribution"], "mixed")
        self.assertEqual([x["path"] for x in scope["changes"]], ["RagEngine.solve.search"])
        self.assertEqual(self.scope({"prompts": {"read": "Synthetic read guidance"}})["attribution"], "mixed")

    def test_known_constants_are_granular_and_literal(self):
        before = root_files()
        for old, new, path in (("shortest complete entity", "concise complete entity", "INSTRUCTIONS.answer"),
                               ('"max_answer_chars": 1000', '"max_answer_chars": 900', "DEFAULTS.max_answer_chars")):
            after = {**before, "rag_core.py": before["rag_core.py"].replace(old, new)}
            scope = observe_edit_scope(before, after, "answer_generation")
            self.assertEqual(scope["associated_module"], "answer_generation")
            self.assertEqual([x["path"] for x in scope["changes"]], [path])
        dynamic = {**before, "rag.py": before["rag.py"].replace("CONFIG = json.loads('{}')", "CONFIG = {'search_limit': unknown_function()}")}
        self.assertEqual(observe_edit_scope(before, dynamic, "retrieval")["attribution"], "unknown")
        self.assertEqual(self.scope({"prompts.plan": "misleading key"})["attribution"], "unknown")

    def test_signature_or_helper_change_prevents_single_module_claim(self):
        before = root_files()
        for changed in (before["rag_core.py"].replace("def consume_read(value):", "def consume_read(value, extra=None):"),
                        before["rag_core.py"].replace("            invalid = 0", "            invalid = new_helper()") + "\ndef new_helper():\n    return 1\n"):
            scope = observe_edit_scope(before, {**before, "rag_core.py": changed}, "evidence_selection")
            self.assertEqual(scope["attribution"], "unknown")
            self.assertIsNone(scope["associated_module"])

    def test_added_unknown_constant_is_a_named_scope(self):
        before = root_files()
        after = {**before, "rag_core.py": before["rag_core.py"] + "\nUNKNOWN_CONSTANT = 1\n"}
        scope = observe_edit_scope(before, after, "retrieval")
        self.assertEqual(scope["attribution"], "unknown")
        self.assertTrue(any(x["path"] == "UNKNOWN_CONSTANT" and x["kind"] == "constant" for x in scope["changes"]))

    def test_unknown_empty_container_cannot_hide_beside_a_known_edit(self):
        scope = self.scope({"search_limit": 4, "unknown_container": {}})
        self.assertEqual(scope["attribution"], "unknown")
        self.assertTrue(any(x["path"] == "CONFIG.unknown_container" and not x["modules"] for x in scope["changes"]))

    def test_legacy_card_keeps_parent_score_but_loses_module_claim(self):
        base = card("base", .3)
        legacy = card("legacy", .9, parents=["base"], module="retrieval")
        legacy.pop("actual_edit_scope"); legacy.pop("intended_target_module")
        for field in ("target_module", "module"):
            with self.subTest(field=field):
                item = deepcopy(legacy)
                if field == "module":
                    item["module"] = item.pop("target_module")
                decision = choose_next([base, item], step=1, panel_hash="panel-A", evaluator_epoch="eval-A")
                self.assertEqual(decision["parent_node_id"], "legacy")
                self.assertEqual(decision["diagnostics"]["module_statistics"]["retrieval"]["gain_samples"], 0)
                self.assertIn("legacy", decision["diagnostics"]["unattributed_experience_ids"])

    def test_source_and_integrity_fail_closed_for_module_learning(self):
        clean = self.scope({"search_limit": 4})
        self.assertEqual(validated_scope(clean), clean)
        for change in ("source", "hash", "classification", "promote_unknown"):
            scope = deepcopy(clean)
            if change == "source":
                scope["source"] = "candidate_reported"
            elif change == "hash":
                scope["child_source_sha256"]["rag.py"] = "0" * 64
            elif change == "classification":
                scope["associated_module"] = "query_rewrite"
            else:
                scope["changes"][0]["path"] = "CONFIG.unknown"
            if change != "hash":
                scope["receipt_sha256"] = digest({k: v for k, v in scope.items() if k != "receipt_sha256"})
            self.assertIsNone(validated_scope(scope))
            decision, _ = self.decision(scope)
            self.assertEqual(decision["parent_node_id"], "child")
            self.assertTrue(all(x["gain_samples"] == 0 for x in decision["diagnostics"]["module_statistics"].values()))

    def test_proposal_accepts_general_refactoring_but_never_model_scope_fields(self):
        program = {"files": root_files()}
        decision = {"target_module": "query_rewrite"}
        proposal = {"writes": {"rag.py": root_files({"search_limit": 4, "max_answer_chars": 900})["rag.py"]},
                    "mechanism": "Synthetic refactor", "intended_target_module": "query_rewrite"}
        changed = evolution._proposal_files(program, decision, proposal)
        self.assertEqual(observe_edit_scope(program["files"], changed, "query_rewrite")["attribution"], "mixed")
        proposal["actual_edit_scope"] = self.scope({"search_limit": 4})
        with self.assertRaisesRegex(ValueError, "invalid development proposal"):
            evolution._proposal_files(program, decision, proposal)


class ScopeRecoveryTests(unittest.TestCase):
    temporary = recovery.EvolutionRecoveryTests.temporary
    inputs = recovery.EvolutionRecoveryTests.inputs
    runner = recovery.EvolutionRecoveryTests.runner

    def test_receipt_is_frozen_recomputed_and_supplied_to_next_development(self):
        with self.temporary() as tmp:
            developer = recovery.Developer(); audit = recovery.Audit()
            self.runner(tmp, developer, audit, expansions=2).run()
            scope_path = Path(tmp) / "steps/0/edit_scope.json"
            scope = evolution.read(scope_path)
            self.assertEqual(scope["source"], "host_ast_diff")
            self.assertEqual(scope["attribution"], "unknown")
            self.assertEqual(developer.calls[1]["decision"]["recent_edit_scopes"][0]["actual_edit_scope"], scope)
            experience = evolution.read(Path(tmp) / "steps/0/experience.json")
            self.assertEqual(experience["actual_edit_scope"], scope)
            self.assertEqual(experience["signed_delta_vs_best_parent"], .5)
            self.assertIsNone(experience["associated_module"])
            self.assertIn("intended_target_module", evolution.read(Path(tmp) / "steps/0/proposal.json"))
            original = scope_path.read_bytes()
            before = len(audit.computed)
            self.runner(tmp, developer, audit, expansions=2).run()
            self.assertEqual(scope_path.read_bytes(), original)
            self.assertEqual(len(audit.computed), before)
            self.assertEqual(len(developer.calls), 2)

    def test_forged_saved_scope_fails_before_child_measurement(self):
        with self.temporary() as tmp:
            audit = recovery.Audit(); self.runner(tmp, audit=audit).run()
            path = Path(tmp) / "steps/0/edit_scope.json"
            scope = evolution.read(path)
            scope["source"] = "candidate_reported"
            save(path, scope)
            before = len(audit.invocations)
            with self.assertRaisesRegex(ValueError, "frozen run artifact differs: edit_scope.json"):
                self.runner(tmp, audit=audit).run()
            self.assertEqual(len(audit.invocations), before + 1)

    def test_missing_scope_receipt_recovers_without_new_development_or_measurement(self):
        with self.temporary() as tmp:
            developer = recovery.Developer(); audit = recovery.Audit()
            runner = self.runner(tmp, developer, audit)
            real_freeze = evolution.freeze
            def crash(path, value):
                if Path(path).name == "edit_scope.json":
                    raise recovery.InjectedCrash("after archive, before edit scope receipt")
                return real_freeze(path, value)
            with patch.object(evolution, "freeze", side_effect=crash):
                with self.assertRaises(recovery.InjectedCrash):
                    runner.run()
            self.assertEqual(len(developer.calls), 1)
            self.assertFalse((Path(tmp) / "steps/0/edit_scope.json").exists())
            self.runner(tmp, developer, audit).run()
            self.assertEqual(len(developer.calls), 1)
            self.assertEqual(len(list((Path(tmp) / "archive/nodes").glob("*.json"))), 2)
            self.assertEqual(evolution.read(Path(tmp) / "steps/0/edit_scope.json")["source"], "host_ast_diff")

    def test_invalid_report_origin_cannot_claim_even_a_positive_raw_gain(self):
        for invalid_child in (True, False):
            with self.subTest(invalid_child=invalid_child), self.temporary() as tmp:
                audit = recovery.Audit(); audit.report_scores = {False: .1, True: 1.0}
                original_run = recovery.FakeMeasurement.run
                def run(measurement, node, *args, **kwargs):
                    result = original_run(measurement, node, *args, **kwargs)
                    if kwargs["role"] == "D_report" and (node["parent_node_id"] is not None) == invalid_child:
                        result = deepcopy(result)
                        result["valid_program"] = False
                    return result
                with patch.object(recovery.FakeMeasurement, "run", run):
                    report = self.runner(tmp, audit=audit).run()
                self.assertEqual(report["schema"], "rag-rsi-v3-report-2")
                self.assertEqual(report["status"], "protocol_invalid")
                self.assertFalse(report["quality_comparison_valid"])
                self.assertEqual(report["quality_claim"], "invalid_protocol_no_quality_claim")
                self.assertIsNone(report["paired_report_gain"])
                self.assertIsNone(report["paired_question_deltas"])
                self.assertAlmostEqual(report["raw_diagnostics"]["paired_report_gain"], .9)
                self.assertEqual(report["delivery_lock"], evolution.read(Path(tmp) / "delivery_lock.json"))
                self.assertNotEqual(report["delivery_lock"]["node_id"], evolution.read(Path(tmp) / "root.json")["node_id"])

    def test_tampered_experience_is_not_silently_overwritten_on_recovery(self):
        with self.temporary() as tmp:
            self.runner(tmp).run()
            path = Path(tmp) / "steps/0/experience.json"
            experience = evolution.read(path); experience["associated_module"] = "retrieval"
            save(path, experience)
            with self.assertRaisesRegex(ValueError, "frozen run artifact differs: experience.json"):
                self.runner(tmp).run()


if __name__ == "__main__":
    unittest.main()
