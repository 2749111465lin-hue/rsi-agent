"""Integration of host feedback, developer payload and action selection.

All questions, predictions and measurements are synthetic. The recording model
does not call a provider; source validation remains the real ProgramDeveloper
path. No candidate program, external API or benchmark data is executed.
"""
from copy import deepcopy
import unittest

from code_rsi.budget import digest
from code_rsi.v3.diagnostics import compact_feedback
from code_rsi.v3.evolution import ProgramDeveloper, experience_card
from code_rsi.v3.execution import root_files
from code_rsi.v3.experience_policy import choose_next, memory_for_action


def fit_result(scores, *, node="base", empty_search=False, no_read=False):
    rows = []
    quote = {"docid": "synthetic-doc", "start": 0, "end": 4, "quote": "abcd"}
    for qid, score in scores.items():
        rows.append({
            "schema": "rag-rsi-v3-execution-2", "question_id": qid, "repeat": 0,
            "role": "D_fit", "score": score, "execution_ok": True,
            "answer": "synthetic prediction " + qid, "answer_usable": True,
            "citation_source_valid": True, "failure_classes": [], "model_errors": [],
            "trace": [
                {"name": "search", "request": {"query": "synthetic"},
                 "response_hash": digest([] if empty_search else [1])},
                {"name": "complete", "request": {"stage": "read"}, "response_hash": digest({})},
                {"name": "complete", "request": {"stage": "answer"}, "response_hash": digest({})}],
            "host_citation_validation": {
                "valid": True, "status": "source_and_presentation_verified",
                "presented_citation_ids": ["e1"], "model_claims_evidence": True},
            "host_evidence_trace": {
                "read_presentations": [] if no_read else [{"verified_quotes": [quote], "sources": []}],
                "final_observations": [{"evidence": {"e1": quote},
                    "response": {"answer": "synthetic prediction " + qid, "evidence_sufficient": True}}]},
            "candidate_reported": {"state": {"gaps": [], "conflicts": [],
                                              "sources": ["DO_NOT_SEND_FULL_STATE"]}},
        })
    return {"node_id": node, "program_id": "program-" + node, "role": "D_fit",
            "panel_hash": "synthetic-panel", "evaluator_epoch": "synthetic-epoch",
            "metric": "f1", "complete": True, "valid_program": True,
            "score": sum(scores.values()) / len(scores), "per_question": dict(scores),
            "resource_usage": {"calls": 2 * len(scores)}, "rows": rows}


def public_tasks(result):
    return [{"question_id": qid, "question": "Synthetic public question " + qid}
            for qid in sorted(result["per_question"])]


def make_card(result, parent=None, module="retrieval", step=0):
    return experience_card(result, parent, operator="Improve" if parent else "Draft",
                           module=module, step=step, mechanism="synthetic reusable change")


def choose(cards, step=1):
    return choose_next(cards, step=step, panel_hash="synthetic-panel",
                       evaluator_epoch="synthetic-epoch")


class RecordingModel:
    def __init__(self, module):
        self.calls = []
        self.output = {"writes": {"rag.py": root_files({"max_rounds": 2})["rag.py"]},
                       "mechanism": "Change reusable round budget", "target_module": module}

    def complete(self, stage, payload):
        self.calls.append((stage, deepcopy(payload)))
        return deepcopy(self.output)


class DeveloperFeedbackIntegrationTests(unittest.TestCase):
    def propose(self, result, refs=None):
        module = "query_rewrite"
        model = RecordingModel(module)
        developer = ProgramDeveloper(model)
        decision = {"parent_node_id": result["node_id"], "operator": "Improve",
                    "target_module": module}
        output = developer.propose(
            {"files": root_files({"max_rounds": 1})}, decision, [], result,
            public_tasks(result), refs or {"q": {"answers": ["PRIVATE_REFERENCE_SENTINEL"]}})
        return model, output

    def test_actual_developer_sends_signed_pairs_without_reference_or_long_state(self):
        result = fit_result({"loss": 0.0, "gain": 1.0})
        result["parent_measurement"] = fit_result({"loss": 1.0, "gain": 0.0}, node="parent")
        references = {q: {"answers": ["PRIVATE_REFERENCE_SENTINEL_" + q]}
                      for q in result["per_question"]}
        model, _ = self.propose(result, references)
        self.assertEqual(len(model.calls), 1)
        stage, payload = model.calls[0]
        self.assertEqual(stage, "develop")
        feedback = payload["feedback"]
        self.assertEqual({c["question_id"]: c["signed_delta"] for c in feedback["cases"]},
                         {"loss": -1.0, "gain": 1.0})
        self.assertTrue(feedback["reference_not_sent"])
        self.assertEqual(feedback["paired_summary"]["regressed"], 1)
        self.assertEqual(feedback["paired_summary"]["improved"], 1)
        serialized = repr(payload)
        self.assertNotIn("PRIVATE_REFERENCE_SENTINEL", serialized)
        self.assertNotIn("DO_NOT_SEND_FULL_STATE", serialized)
        self.assertNotIn("parent_measurement", payload["feedback"])

    def test_nonfit_developer_result_is_rejected_before_call(self):
        result = fit_result({"q": 0.5})
        result["role"] = "D_report"
        model = RecordingModel("query_rewrite")
        with self.assertRaises(ValueError):
            ProgramDeveloper(model).propose(
                {"files": root_files()}, {"target_module": "query_rewrite"}, [], result,
                public_tasks(result), {})
        self.assertEqual(model.calls, [])

    def test_foreign_parent_is_rejected_before_development_call(self):
        result = fit_result({"q": 0.5})
        parent = fit_result({"q": 1.0}, node="parent")
        parent["panel_hash"] = "other-panel"
        result["parent_measurement"] = parent
        model = RecordingModel("query_rewrite")
        with self.assertRaises(ValueError):
            ProgramDeveloper(model).propose(
                {"files": root_files()}, {"target_module": "query_rewrite"}, [], result,
                public_tasks(result), {})
        self.assertEqual(model.calls, [])


class ExperienceFeedbackIntegrationTests(unittest.TestCase):
    def test_card_keeps_actual_edited_module_despite_multiple_diagnostic_priors(self):
        result = fit_result({"q": 0.5}, empty_search=True)
        result["rows"][0]["candidate_reported"]["state"]["conflicts"] = ["synthetic conflict"]
        card = make_card(result, module="answer_generation")
        self.assertGreater(len(card["diagnostics"]["module_priors"]), 1)
        self.assertEqual(card["target_module"], "answer_generation")

    def test_empty_retrieval_cold_start_changes_rotation_to_query_module(self):
        card = make_card(fit_result({"q": 0.4}, empty_search=True), module="answer_generation")
        decision = choose([card], step=2)  # Old rotation alone would choose evidence_selection.
        self.assertEqual(decision["operator"], "Improve")
        self.assertEqual(decision["target_module"], "query_rewrite")
        self.assertIn("observed_failure_cold_start_prior", decision["reason"])

    def test_no_read_cold_start_selects_evidence_module(self):
        card = make_card(fit_result({"q": 0.4}, no_read=True), module="retrieval")
        decision = choose([card], step=0)
        self.assertEqual(decision["target_module"], "evidence_selection")

    def test_diagnostic_prior_does_not_change_host_reward(self):
        parent = fit_result({"q": 0.3}, node="parent")
        clean = fit_result({"q": 0.6}, node="clean")
        diagnosed = deepcopy(clean)
        diagnosed["rows"][0]["trace"][0]["response_hash"] = digest([])
        diagnosed["rows"][0]["candidate_reported"]["state"]["gaps"] = ["synthetic missing bridge"]
        plain_card = make_card(clean, parent, module="answer_generation", step=1)
        diagnosed_card = make_card(diagnosed, parent, module="answer_generation", step=1)
        self.assertNotEqual(plain_card["diagnostics"], diagnosed_card["diagnostics"])
        self.assertEqual(plain_card["reward"], diagnosed_card["reward"])
        self.assertEqual(diagnosed_card["score"], 0.6)
        self.assertAlmostEqual(diagnosed_card["reward"]["paired_signed_gain"], 0.3)
        self.assertFalse(diagnosed_card["reward"]["proxy_added_to_terminal_quality"])

    def test_positive_historical_gain_beats_conflicting_diagnostic_prior(self):
        baseline = fit_result({"q": 0.3}, node="base", empty_search=True)
        worse_query = fit_result({"q": 0.2}, node="query-child", empty_search=True)
        better_answer = fit_result({"q": 0.7}, node="answer-child", empty_search=True)
        cards = [
            make_card(baseline),
            make_card(worse_query, baseline, module="query_rewrite", step=1),
            make_card(better_answer, baseline, module="answer_generation", step=2)]
        decision = choose(cards, step=3)
        self.assertEqual(decision["parent_node_id"], "answer-child")
        self.assertEqual(decision["diagnostics"]["cold_start_module_priors"]["query_rewrite"], 1.0)
        self.assertEqual(decision["target_module"], "answer_generation")
        self.assertIn("matched_failure_signed_gain_uncertainty_and_cost", decision["reason"])
        stats = decision["diagnostics"]["module_statistics"]
        self.assertAlmostEqual(stats["query_rewrite"]["mean_signed_gain"], -0.1)
        self.assertAlmostEqual(stats["answer_generation"]["mean_signed_gain"], 0.4)
        changed = deepcopy(cards)
        for card in changed:
            card["diagnostics"]["module_priors"] = {"retrieval": 1.0}
        changed_decision = choose(changed, step=3)
        self.assertEqual(changed_decision["target_module"], "answer_generation")
        self.assertEqual(changed_decision["diagnostics"]["module_statistics"], stats)
        self.assertFalse(decision["diagnostics"]["priors_added_to_answer_reward"])

    def test_model_report_alone_does_not_force_debug(self):
        result = fit_result({"q": 1.0})
        result["rows"][0]["candidate_reported"]["state"]["conflicts"] = ["synthetic conflict"]
        card = make_card(result)
        decision = choose([card])
        self.assertTrue(card["valid_program"])
        self.assertEqual(card["reward"]["answer_quality"], 1.0)
        self.assertNotIn("evidence_conflict", card["failure_classes"])
        self.assertIn("evidence_conflict", card["diagnostics"]["model_reported"])
        self.assertEqual(decision["operator"], "Improve")

    def test_priors_are_per_question_and_match_compact_feedback(self):
        result = fit_result({"empty": 0.2, "hit": 0.8})
        result["rows"][0]["trace"][0]["response_hash"] = digest([])
        for repeat in range(1, 5):
            extra = deepcopy(result["rows"][0])
            extra["repeat"] = repeat
            result["rows"].append(extra)
        card = make_card(result)
        feedback = compact_feedback(result, public_tasks(result))
        self.assertEqual(card["diagnostics"]["module_priors"], feedback["module_priors"])
        self.assertEqual(card["diagnostics"]["module_priors"]["query_rewrite"], 0.5)

    def test_action_memory_preserves_diagnostics_and_signed_gain_without_raw_state(self):
        baseline = fit_result({"q": 0.3}, node="base", empty_search=True)
        child = fit_result({"q": 0.7}, node="child", empty_search=True)
        cards = [make_card(baseline),
                 make_card(child, baseline, module="answer_generation", step=1)]
        decision = choose(cards, step=1)
        memories = memory_for_action(cards, decision)
        child_memory = next(x for x in memories if x["node_id"] == "child")
        self.assertAlmostEqual(child_memory["signed_delta_vs_best_parent"], 0.4)
        self.assertIn("empty_retrieval", child_memory["diagnostics"]["host_observed"])
        self.assertNotIn("DO_NOT_SEND_FULL_STATE", repr(memories))

    def test_card_cannot_launder_nonfit_top_role_with_legacy_rows(self):
        for role in ("D_select", "D_report"):
            with self.subTest(role=role):
                result = fit_result({"q": 0.0})
                result["role"] = role
                result["rows"][0].pop("role")
                with self.assertRaises(ValueError):
                    make_card(result)

    def test_card_rejects_unknown_provider_zero_instead_of_learning_it(self):
        result = fit_result({"q": 0.0})
        result["rows"][0]["error_type"] = "UnknownProviderOutcome"
        with self.assertRaises(ValueError):
            make_card(result)

    def test_card_rejects_incomplete_measurement(self):
        result = fit_result({"q": 0.0})
        result["complete"] = False
        with self.assertRaises(ValueError):
            make_card(result)

    def test_card_rejects_foreign_parent_pairing(self):
        result = fit_result({"q": 0.8})
        parent = fit_result({"q": 0.2}, node="parent")
        parent["panel_hash"] = "foreign-panel"
        with self.assertRaises(ValueError):
            make_card(result, parent, module="answer_generation", step=1)


if __name__ == "__main__":
    unittest.main()
