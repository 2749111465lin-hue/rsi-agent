"""Synthetic/offline contracts; no datasets, provider calls or reference files."""
from copy import deepcopy
import hashlib
import json
import unittest

from code_rsi.v3.diagnostics import compact_feedback, diagnose_execution


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def row(qid="q1", score=1.0):
    quote = {"docid": "synthetic-source", "start": 0, "end": 4, "quote": "abcd"}
    return {
        "schema": "rag-rsi-v3-execution-2", "question_id": qid, "node_id": "child",
        "role": "D_fit", "score": score, "repeat": 0,
        "answer": "synthetic prediction", "answer_usable": True, "execution_ok": True,
        "failure_classes": [], "model_errors": [], "citation_source_valid": True,
        "resource_usage": {"search_calls": 1, "model_calls": 2, "read_calls": 0},
        "trace": [
            {"name": "search", "request": {"query": "synthetic query"}, "response_hash": digest([{"hit": 1}])},
            {"name": "complete", "request": {"stage": "read"}, "response_hash": digest({"claims": []})},
            {"name": "complete", "request": {"stage": "answer"}, "response_hash": digest({"answer": "x"})}],
        "host_citation_validation": {
            "valid": True, "status": "source_and_presentation_verified",
            "presented_citation_ids": ["e1"], "raw_citation_ids": ["e1"],
            "model_claims_evidence": True},
        "host_evidence_trace": {
            "read_presentations": [{"sources": [], "verified_quotes": [quote]}],
            "final_observations": [{"evidence": {"e1": quote},
                                    "response": {"answer": "synthetic prediction",
                                                 "evidence_sufficient": True}}]},
        "candidate_reported": {"state": {"gaps": [], "conflicts": []}, "failure_types": []}}


def measurement(rows, node="child", **kwargs):
    grouped = {}
    for item in rows:
        grouped.setdefault(item["question_id"], []).append(item["score"])
    per = {q: sum(s)/len(s) for q, s in grouped.items()}
    return {"node_id": node, "role": "D_fit", "panel_hash": "synthetic-panel",
            "evaluator_epoch": "synthetic-epoch", "metric": "f1", "complete": True,
            "score": sum(per.values())/len(per), "per_question": per, "rows": rows, **kwargs}


def tasks(rows):
    return [{"question_id": q, "question": "Synthetic public question for " + q}
            for q in sorted({r["question_id"] for r in rows})]


class ExecutionDiagnosticsTests(unittest.TestCase):
    def test_declared_abstention_without_citations_is_not_invalid_citation(self):
        receipt=row()
        receipt["host_citation_validation"].update(status="missing_citations",model_claims_evidence=False,raw_citation_ids=[])
        receipt["host_evidence_trace"]["final_observations"][0]["response"]["evidence_sufficient"]=False
        result=diagnose_execution(receipt)
        self.assertIn("missing_answer_citations",result["host_observed"])
        self.assertNotIn("invalid_answer_citation",result["host_observed"])
        self.assertNotIn("uncited_supported_claim",result["host_observed"])
        self.assertLessEqual(max(result["module_priors"].values()),0.35)

    def test_claimed_supported_answer_without_citations_needs_format_repair(self):
        receipt=row()
        receipt["host_citation_validation"].update(status="missing_citations",model_claims_evidence=True,raw_citation_ids=[])
        result=diagnose_execution(receipt)
        self.assertIn("uncited_supported_claim",result["host_observed"])
        self.assertNotIn("invalid_answer_citation",result["host_observed"])
        self.assertEqual(result["module_priors"]["answer_generation"],1)

    def test_valid_quote_is_not_a_semantic_verdict(self):
        result = diagnose_execution(row())
        self.assertEqual(result["host_observed"], [])
        self.assertEqual(result["suggested_modules"], [])
        self.assertEqual(result["observations"]["semantic_support"], "not_host_verified")

    def test_host_quoted_answer_can_still_report_a_gap(self):
        receipt = row()
        receipt["candidate_reported"]["state"]["gaps"] = ["missing date"]
        receipt["host_citation_validation"]["model_claims_evidence"] = False
        result = diagnose_execution(receipt)
        self.assertEqual(result["host_observed"], [])
        self.assertEqual(result["model_reported"], ["evidence_gap", "evidence_insufficient"])
        self.assertLessEqual(max(result["module_priors"].values()), 0.35)

    def test_empty_search_is_observed_from_host_response_hash(self):
        receipt = row()
        receipt["trace"][0]["response_hash"] = digest([])
        result = diagnose_execution(receipt)
        self.assertIn("empty_retrieval", result["host_observed"])
        self.assertEqual(result["suggested_modules"][0], "query_rewrite")
        self.assertEqual(result["evidence_refs"]["host:empty_retrieval"], ["/trace/0/response_hash"])

    def test_candidate_empty_sources_or_dedup_is_not_host_empty_retrieval(self):
        receipt = row()
        receipt["candidate_reported"].update(
            {"trace": [{"stage": "search", "source_ids": []}],
             "failure_types": ["empty_retrieval"], "state": {"sources": []}})
        result = diagnose_execution(receipt)
        self.assertNotIn("empty_retrieval", result["host_observed"])

    def test_backend_read_counter_is_not_model_evidence_read(self):
        self.assertEqual(row()["resource_usage"]["read_calls"], 0)
        self.assertNotIn("no_evidence_read", diagnose_execution(row())["host_observed"])

    def test_no_read_and_no_final_evidence_are_distinct(self):
        receipt = row()
        receipt["host_evidence_trace"]["read_presentations"] = []
        receipt["host_evidence_trace"]["final_observations"][0]["evidence"] = {}
        result = diagnose_execution(receipt)
        self.assertIn("no_evidence_read", result["host_observed"])
        self.assertIn("no_final_evidence", result["host_observed"])
        self.assertEqual(result["suggested_modules"][0], "evidence_selection")

    def test_missing_trace_does_not_prove_no_read_or_search(self):
        receipt = {"execution_ok": True, "answer_usable": True}
        result = diagnose_execution(receipt)
        self.assertEqual(result["host_observed"], [])
        self.assertFalse(result["observations"]["trace_complete"])

    def test_model_parse_stage_localizes_priority(self):
        receipt = row()
        receipt["trace"][2]["response_hash"] = digest({"_meta": {"truncated": True, "finish_reason": "error"}})
        receipt["model_errors"] = ["ModelResponseError"]
        result = diagnose_execution(receipt)
        self.assertIn("model_parse_failure", result["host_observed"])
        self.assertEqual(result["suggested_modules"][0], "answer_generation")

    def test_unknown_provider_outcome_is_not_parse_or_answer_error(self):
        for field, value in (("error_type", "UnknownProviderOutcome"), ("provider_outcome", "unknown"),
                             ("status", "host_error"), ("complete", False)):
            with self.subTest(field=field):
                receipt = row()
                receipt[field] = value
                receipt["score"] = 0
                result = diagnose_execution(receipt)
                self.assertEqual(result["measurement_status"], "unavailable")
                self.assertEqual(result["host_observed"], ["measurement_unavailable"])
                self.assertEqual(result["module_priors"], {})

    def test_unclassified_model_error_is_not_assumed_to_be_parse_failure(self):
        receipt = row()
        receipt["model_errors"] = ["UnclassifiedTransportFailure"]
        self.assertNotIn("model_parse_failure", diagnose_execution(receipt)["host_observed"])

    def test_model_claim_is_bounded_and_does_not_invalidate_execution(self):
        receipt = row()
        receipt["candidate_reported"]["state"].update(
            {"gaps": ["missing bridge"] * 20, "conflicts": ["date conflict"] * 30})
        receipt["candidate_reported"]["failure_types"] = ["answer_schema_failure"]
        result = diagnose_execution(receipt)
        self.assertEqual(result["host_observed"], [])
        self.assertIn("evidence_conflict", result["model_reported"])
        self.assertEqual(result["model_details"]["gaps"], ["missing bridge"])
        self.assertTrue(all(p <= 0.35 for p in result["module_priors"].values()))

    def test_host_failure_classes_list_and_mapping_are_supported(self):
        for value in (["invalid_answer_citation"], {"invalid_answer_citation": 2}):
            receipt = row()
            receipt["failure_classes"] = value
            self.assertIn("invalid_answer_citation", diagnose_execution(receipt)["host_observed"])

    def test_abandoned_final_is_not_selected_final_evidence(self):
        receipt = row()
        receipt["host_evidence_trace"]["final_observations"].append(
            {"evidence": {}, "response": {"answer": "abandoned", "evidence_sufficient": False}})
        result = diagnose_execution(receipt)
        self.assertNotIn("no_final_evidence", result["host_observed"])
        self.assertNotIn("evidence_insufficient", result["model_reported"])

    def test_repeated_host_queries_are_observable(self):
        receipt = row()
        receipt["trace"].append({"name": "search", "request": {"query": "  Synthetic   QUERY "},
                                 "response_hash": digest([1])})
        self.assertIn("repeated_query", diagnose_execution(receipt)["host_observed"])

    def test_nonfit_execution_is_rejected(self):
        for role in ("D_report", "D_select", "report", "select"):
            receipt = row()
            receipt["role"] = role
            with self.assertRaises(ValueError):
                diagnose_execution(receipt)


class CompactFeedbackTests(unittest.TestCase):
    def test_paired_regression_and_improvement_are_retained(self):
        rows = [row("loss", 0.0), row("gain", 1.0), row("flat", 0.5)]
        parent_rows = [row("loss", 1.0), row("gain", 0.0), row("flat", 0.5)]
        current = measurement(rows, parent_measurement=measurement(parent_rows, node="parent"))
        result = compact_feedback(current, tasks(rows), max_cases=2)
        self.assertEqual([c["question_id"] for c in result["cases"]], ["loss", "gain"])
        self.assertEqual([c["signed_delta"] for c in result["cases"]], [-1.0, 1.0])
        self.assertEqual(result["paired_summary"]["mean_signed_gain"], 0.0)
        self.assertEqual(result["paired_summary"]["regressed"], 1)
        self.assertFalse(result["paired_summary"]["clipped_negative_gains"])

    def test_negative_panel_gain_is_not_clipped(self):
        current = measurement([row(score=0.2)],
                              parent_measurement=measurement([row(score=0.8)], node="parent"))
        result = compact_feedback(current, tasks(current["rows"]))
        self.assertLess(result["paired_summary"]["mean_signed_gain"], 0)

    def test_high_information_failure_wins_over_first_rows(self):
        rows = [row("generic" + str(i), 0.0) for i in range(5)]
        special = row("late-empty", 0.0)
        special["trace"][0]["response_hash"] = digest([])
        rows.append(special)
        result = compact_feedback(measurement(rows), tasks(rows), max_cases=1)
        self.assertEqual(result["cases"][0]["question_id"], "late-empty")

    def test_success_contrast_retained_without_parent(self):
        rows = [row("fail", 0), row("ok", 1), row("other", 0)]
        rows[0]["failure_classes"] = ["invalid_answer_citation"]
        result = compact_feedback(measurement(rows), tasks(rows), max_cases=2)
        self.assertEqual({c["question_id"] for c in result["cases"]}, {"fail", "ok"})

    def test_model_report_is_not_host_failure_or_reward(self):
        receipt = row()
        receipt["candidate_reported"]["state"]["conflicts"] = ["synthetic conflict"]
        current = measurement([receipt])
        result = compact_feedback(current, tasks([receipt]))
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["summary"]["host_observed"], {})
        self.assertEqual(result["summary"]["model_reported"]["evidence_conflict"], 1)

    def test_nonfit_and_mixed_role_rows_are_rejected(self):
        for where in ("top", "row", "parent"):
            current = measurement([row()])
            if where == "top":
                current["role"] = "D_report"
            elif where == "row":
                current["rows"][0]["role"] = "D_select"
            else:
                current["parent_measurement"] = measurement([row()], role="D_select")
            with self.assertRaises(ValueError):
                compact_feedback(current, tasks(current["rows"]))

    def test_identity_isolation_in_rows_and_parents(self):
        for where, key in (("row", "panel_hash"), ("row", "evaluator_epoch"),
                           ("parent", "panel_hash"), ("parent", "evaluator_epoch"), ("parent", "metric")):
            with self.subTest(where=where, key=key):
                current = measurement([row()])
                current["parent_measurement"] = measurement([row()], node="parent")
                target = current["rows"][0] if where == "row" else current["parent_measurement"]
                target[key] = "different"
                with self.assertRaises(ValueError):
                    compact_feedback(current, tasks(current["rows"]))

    def test_unknown_row_cannot_become_zero_score_learning(self):
        receipt = row(score=0)
        receipt["error_type"] = "UnknownProviderOutcome"
        current = measurement([receipt])
        result = compact_feedback(current, tasks([receipt]))
        self.assertIsNone(result["score"])
        self.assertEqual(result["cases"], [])
        self.assertEqual(result["module_priors"], {})

    def test_incomplete_panel_has_no_learning(self):
        current = measurement([row()], complete=False)
        result = compact_feedback(current, tasks(current["rows"]))
        self.assertEqual(result["measurement_status"], "unavailable")
        self.assertIsNone(result["score"])

    def test_whitelist_and_bounded_text(self):
        receipt = row()
        receipt["candidate_reported"]["state"]["sources"] = ["RAW_STATE_SECRET" * 1000]
        receipt["candidate_reported"]["state"]["gaps"] = ["g" * 5000] * 20
        current = measurement([receipt])
        public = tasks([receipt])
        public[0].update({"answer": "GOLD_SECRET", "references": "REFERENCE_SECRET",
                          "documents": ["FULL_CORPUS_SECRET" * 10000]})
        result = compact_feedback(current, public)
        serialized = json.dumps(result)
        self.assertNotIn("GOLD_SECRET", serialized)
        self.assertNotIn("REFERENCE_SECRET", serialized)
        self.assertNotIn("RAW_STATE_SECRET", serialized)
        self.assertNotIn("FULL_CORPUS_SECRET", serialized)
        self.assertLess(len(serialized), 7000)

    def test_determinism_and_no_mutation_with_repeats(self):
        rows = [row("b", 0.2), row("a", 0.4), row("a", 0.8)]
        rows[2]["repeat"] = 1
        current = measurement(rows)
        before = deepcopy(current)
        public = tasks(rows)
        first = compact_feedback(current, public)
        shuffled = deepcopy(current)
        shuffled["rows"].reverse()
        second = compact_feedback(shuffled, list(reversed(public)))
        self.assertEqual(first, second)
        self.assertEqual(current, before)

    def test_repeats_do_not_multiply_prior_counts(self):
        receipt = row()
        receipt["trace"][0]["response_hash"] = digest([])
        rows = [deepcopy(receipt) for _ in range(5)]
        for i, item in enumerate(rows):
            item["repeat"] = i
        result = compact_feedback(measurement(rows), tasks(rows))
        self.assertEqual(result["summary"]["host_observed"]["empty_retrieval"], 1)
        self.assertEqual(result["module_priors"]["query_rewrite"], 1.0)

    def test_zero_case_budget_preserves_signed_summary(self):
        current = measurement([row(score=0.0)],
                              parent_measurement=measurement([row(score=1.0)], node="parent"))
        result = compact_feedback(current, tasks(current["rows"]), max_cases=0)
        self.assertEqual(result["cases"], [])
        self.assertEqual(result["paired_summary"]["mean_signed_gain"], -1.0)

    def test_bad_scores_or_aggregate_are_rejected(self):
        for bad in (float("nan"), float("inf"), True, "0.5"):
            current = measurement([row()])
            current["rows"][0]["score"] = bad
            with self.assertRaises(ValueError):
                compact_feedback(current, tasks(current["rows"]))
        current = measurement([row()])
        current["score"] = 0
        with self.assertRaises(ValueError):
            compact_feedback(current, tasks(current["rows"]))

    def test_custom_metric_does_not_assume_target_one(self):
        current = measurement([row(score=4.0)], metric="custom_answer_metric")
        result = compact_feedback(current, tasks(current["rows"]))
        self.assertIsNone(result["summary"]["score_shortfall_questions"])

    def test_unknown_parent_cannot_supply_learning_pairs(self):
        current = measurement([row()], parent_measurement=measurement([row()], node="parent"))
        current["parent_measurement"]["rows"][0]["provider_outcome"] = "unknown"
        with self.assertRaises(ValueError):
            compact_feedback(current, tasks(current["rows"]))

    def test_public_question_panel_must_match(self):
        current = measurement([row()])
        with self.assertRaises(ValueError):
            compact_feedback(current, [{"question_id": "other", "question": "synthetic"}])

    def test_invalid_case_budgets_are_rejected(self):
        current = measurement([row()])
        for value in (-1, 17, 1.5, True):
            with self.assertRaises(ValueError):
                compact_feedback(current, tasks(current["rows"]), max_cases=value)


if __name__ == "__main__":
    unittest.main()
