"""Synthetic information-boundary checks for aggregate/cases/trace feedback."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from code_rsi.v3 import feedback_conditions as controlled
from code_rsi.v3.diagnostics import FLOW_BOUNDS
from test_v3_diagnostics import row, measurement, tasks
from test_v3_fixture_origin import bind_synthetic_origin
from test_v3_answer_origin_feedback import receipt

COMMON_FIELDS = {"schema", "condition", "role", "score", "metric", "question_count", "measured_rows",
                 "program_eligible", "delivery", "resource_usage", "raw_reference_objects_not_sent",
                 "fit_feedback_can_reveal_accepted_answers"}
CASE_FIELDS = {"question_id", "repeat", "question", "prediction", "sampled_repeat_score", "host_score", "repeat_count"}
DIAGNOSTIC_FIELDS = {"host_observed", "model_reported", "model_details", "semantic_support", "model_reports_are_unverified"}


def panel():
    rows = []
    for qid, scores in (("q1", (1., .5)), ("q2", (0., .75))):
        for repeat, score in enumerate(scores):
            item = row(qid, score)
            item["repeat"] = repeat
            item["answer"] = "Prediction for " + qid + " repeat " + str(repeat)
            item["host_evidence_trace"]["final_observations"][-1]["response"]["answer"] = item["answer"]
            rows.append(bind_synthetic_origin(item))
    return measurement(rows), tasks(rows), [{"question_id": "q2", "repeat": 1}, {"question_id": "q1", "repeat": 0}]


class ControlledFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.measurement, self.tasks, self.schedule = panel()

    def view(self, condition, measurement_value=None, task_values=None, schedule=None):
        return controlled.controlled_feedback(
            self.measurement if measurement_value is None else measurement_value,
            self.tasks if task_values is None else task_values, condition=condition,
            case_schedule=self.schedule if schedule is None else schedule)

    def test_aggregate_has_exact_whitelist_and_only_numeric_measurement_details(self):
        result = self.view("aggregate")
        self.assertEqual(set(result), COMMON_FIELDS)
        self.assertEqual(result["score"], .5625)
        self.assertEqual(result["metric"], "f1")
        self.assertEqual(result["question_count"], 2)
        self.assertEqual(result["measured_rows"], 4)
        self.assertTrue(result["program_eligible"])
        self.assertEqual(result["delivery"], {key: 4 for key in controlled.DELIVERY_FIELDS})
        self.assertEqual(result["resource_usage"], {"model_calls": 8, "search_calls": 4, "read_calls": 0})
        self.assertFalse(result["fit_feedback_can_reveal_accepted_answers"])
        self.assertTrue(result["raw_reference_objects_not_sent"])
        for forbidden in ("cases", "question_id", "panel_hash", "node_id", "summary", "module_priors", "decision", "experience"):
            self.assertNotIn(forbidden, result)

    def test_whitelist_blocks_all_unrelated_measurement_row_and_task_sentinels(self):
        markers = []
        fields = ("priors", "module_priors", "summary", "model_details", "history", "decision", "experience", "reference", "references", "gold_answer")
        for owner, prefix in ((self.measurement, "MEASUREMENT"), (self.measurement["rows"][0], "ROW"), (self.tasks[0], "TASK")):
            for field in fields:
                marker = prefix + "_" + field + "_SECRET_SENTINEL"
                owner[field] = {"text": marker}
                markers.append(marker)
        self.tasks[0]["answer"] = "TASK_ANSWER_SECRET_SENTINEL"
        self.tasks[0]["reference_answer"] = "TASK_REF_SECRET_SENTINEL"
        markers.extend((self.tasks[0]["answer"], self.tasks[0]["reference_answer"]))
        for condition in controlled.CONDITIONS:
            with self.subTest(condition=condition):
                serialized = json.dumps(self.view(condition))
                for marker in markers:
                    self.assertNotIn(marker, serialized)

    def test_reference_objects_are_never_accessed_or_copied(self):
        class ForbiddenReference:
            def __deepcopy__(self, memo): raise AssertionError("reference copied")
            def __str__(self): raise AssertionError("reference rendered")
            def __iter__(self): raise AssertionError("reference iterated")
        self.measurement["references"] = ForbiddenReference()
        for task in self.tasks:
            task["reference"] = ForbiddenReference()
            task["answer"] = ForbiddenReference()
        for condition in controlled.CONDITIONS:
            self.assertTrue(self.view(condition)["raw_reference_objects_not_sent"])

    def test_cases_follow_declared_question_and_repeat_not_failure_severity(self):
        self.measurement["rows"][0]["candidate_reported"]["state"]["gaps"] = ["many apparent gaps"]
        result = self.view("cases")
        self.assertEqual(set(result), COMMON_FIELDS | {"cases"})
        self.assertEqual([(c["question_id"], c["repeat"]) for c in result["cases"]], [("q2", 1), ("q1", 0)])
        self.assertEqual([c["prediction"] for c in result["cases"]], ["Prediction for q2 repeat 1", "Prediction for q1 repeat 0"])
        self.assertEqual([c["sampled_repeat_score"] for c in result["cases"]], [.75, 1.])
        self.assertEqual([c["host_score"] for c in result["cases"]], [.375, .75])
        self.assertEqual([c["repeat_count"] for c in result["cases"]], [2, 2])
        self.assertTrue(all(set(c) == CASE_FIELDS for c in result["cases"]))
        self.assertTrue(result["fit_feedback_can_reveal_accepted_answers"])

    def test_cases_and_trace_share_every_common_field_exactly(self):
        cases, trace = self.view("cases"), self.view("trace")
        self.assertEqual({key: value for key, value in cases.items() if key not in {"condition", "cases"}},
                         {key: value for key, value in trace.items() if key not in {"condition", "cases"}})
        for plain, enriched in zip(cases["cases"], trace["cases"]):
            self.assertEqual({key: enriched[key] for key in CASE_FIELDS}, plain)
            self.assertEqual(set(enriched), CASE_FIELDS | {"diagnostics", "execution_flow"})
            self.assertEqual(set(enriched["diagnostics"]), DIAGNOSTIC_FIELDS)
            self.assertEqual(set(enriched["diagnostics"]["model_details"]), {"gaps", "conflicts"})

    def test_trace_feedback_is_present_only_in_trace_condition(self):
        chosen = next(r for r in self.measurement["rows"] if r["question_id"] == "q2" and r["repeat"] == 1)
        chosen["trace"][0]["request"]["query"] = "OBSERVED_QUERY_SENTINEL"
        chosen["candidate_reported"]["state"]["gaps"] = ["MODEL_GAP_SENTINEL"]
        for condition in ("aggregate", "cases"):
            serialized = json.dumps(self.view(condition))
            self.assertNotIn("OBSERVED_QUERY_SENTINEL", serialized)
            self.assertNotIn("MODEL_GAP_SENTINEL", serialized)
        result = self.view("trace")
        first = result["cases"][0]
        self.assertIn("OBSERVED_QUERY_SENTINEL", json.dumps(first["execution_flow"]))
        self.assertEqual(first["diagnostics"]["model_details"]["gaps"], ["MODEL_GAP_SENTINEL"])
        self.assertIn("evidence_gap", first["diagnostics"]["model_reported"])
        self.assertNotIn("evidence_gap", first["diagnostics"]["host_observed"])
        self.assertEqual(first["diagnostics"]["semantic_support"], "not_host_verified")
        self.assertTrue(first["diagnostics"]["model_reports_are_unverified"])

    def test_question_and_prediction_are_not_asymmetrically_cropped(self):
        selected = next(r for r in self.measurement["rows"] if r["question_id"] == "q2" and r["repeat"] == 1)
        selected["answer"] = "Long prediction " * 150
        selected["host_evidence_trace"]["final_observations"][-1]["response"]["answer"] = selected["answer"]
        index = self.measurement["rows"].index(selected)
        self.measurement["rows"][index] = bind_synthetic_origin(selected)
        next(task for task in self.tasks if task["question_id"] == "q2")["question"] = "Long original question " * 100
        for condition in ("cases", "trace"):
            result = self.view(condition)["cases"][0]
            self.assertEqual(result["prediction"], selected["answer"])
            self.assertEqual(result["question"], "Long original question " * 100)

    def test_model_details_and_execution_flow_are_bounded_separately(self):
        selected = self.measurement["rows"][-1]
        selected["candidate_reported"]["state"] = {field: [str(i) + "x" * 1000 for i in range(40)] for field in ("gaps", "conflicts")}
        result = self.view("trace")["cases"][0]
        for values in result["diagnostics"]["model_details"].values():
            self.assertLessEqual(len(values), 3)
            self.assertTrue(all(len(text) <= 240 for text in values))
        self.assertLessEqual(len(json.dumps(result["execution_flow"], ensure_ascii=False, separators=(",", ":")).encode()), FLOW_BOUNDS["json_bytes"])

    def test_input_order_and_return_mutation_do_not_change_selected_rows(self):
        before = deepcopy((self.measurement, self.tasks, self.schedule))
        first = self.view("trace")
        reordered = deepcopy(self.measurement)
        reordered["rows"].reverse()
        second = self.view("trace", reordered, list(reversed(self.tasks)))
        self.assertEqual(first, second)
        first["cases"][0]["diagnostics"]["model_reported"].append("mutated")
        first["delivery"]["execution_ok"] = -1
        self.assertEqual((self.measurement, self.tasks, self.schedule), before)

    def test_all_conditions_including_aggregate_validate_the_same_schedule(self):
        variants = [[], self.schedule * 9, [{"question_id": "q1", "repeat": 0}, {"question_id": "q1", "repeat": 1}],
                    [{"question_id": "q1", "repeat": True}], [{"question_id": "q1", "repeat": -1}],
                    [{"question_id": "q1", "repeat": 9}], [{"question_id": "foreign", "repeat": 0}],
                    [{"question_id": "q1", "repeat": 0, "selection_reason": "failure"}]]
        for condition in controlled.CONDITIONS:
            for schedule in variants:
                with self.subTest(condition=condition, schedule=schedule), self.assertRaises(ValueError):
                    self.view(condition, schedule=schedule)

    def test_duplicate_missing_or_invalid_repeat_rows_are_rejected(self):
        variants = []
        rows = deepcopy(self.measurement["rows"])
        rows.append(deepcopy(rows[0]))
        variants.append(rows)
        rows = deepcopy(self.measurement["rows"])
        rows.pop()
        variants.append(rows)
        for invalid in (True, -1, "0", None):
            rows = deepcopy(self.measurement["rows"])
            rows[0]["repeat"] = invalid
            variants.append(rows)
        for rows in variants:
            with self.subTest(repeats=[r.get("repeat") for r in rows]), self.assertRaises(ValueError):
                self.view("aggregate", measurement(rows))

    def test_incomplete_unknown_non_fit_and_wrong_task_panels_are_rejected(self):
        for changes in ({"complete": False}, {"status": "UnknownProviderOutcome"}, {"role": "D_report"}, {"role": "D_select"}, {"metric": "unreviewed_custom_metric"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.view("aggregate", {**self.measurement, **changes})
        bad = deepcopy(self.measurement)
        bad["rows"][0]["role"] = "D_report"
        with self.assertRaises(ValueError): self.view("cases", bad)
        with self.assertRaises(ValueError): self.view("aggregate", task_values=self.tasks[:1])
        with self.assertRaises(ValueError): self.view("aggregate", task_values=self.tasks + self.tasks[:1])

    def test_score_mismatch_and_out_of_range_cannot_enter_controlled_feedback(self):
        for value in (0., float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.view("aggregate", {**self.measurement, "score": value})
        rows = deepcopy(self.measurement["rows"])
        rows[0]["score"] = 1.1
        with self.assertRaises(ValueError): self.view("aggregate", measurement(rows))

    def test_invalid_origin_never_exposes_raw_supervision(self):
        bad = receipt(["Actual model answer"], "Invented candidate answer")
        fit = measurement([bad], valid_program=False)
        schedule = [{"question_id": bad["question_id"], "repeat": 0}]
        for condition in controlled.CONDITIONS:
            result = self.view(condition, fit, tasks([bad]), schedule)
            self.assertFalse(result["program_eligible"])
            self.assertIsNone(result["score"])
            self.assertNotIn("raw_score", result)
            for case in result.get("cases", []):
                self.assertIsNone(case["sampled_repeat_score"])
                self.assertIsNone(case["host_score"])
                self.assertNotIn("raw_host_score", case)

    def test_forged_origin_flag_fails_even_in_aggregate(self):
        bad = receipt(["Actual model answer"], "Invented candidate answer")
        bad["answer_origin_valid"] = True
        with self.assertRaises(ValueError):
            self.view("aggregate", measurement([bad]), tasks([bad]), [{"question_id": bad["question_id"], "repeat": 0}])

    def test_legacy_origin_and_missing_flags_remain_unknown_not_zero(self):
        legacy = row("legacy", legacy=True)
        result = self.view("aggregate", measurement([legacy]), tasks([legacy]), [{"question_id": "legacy", "repeat": 0}])
        self.assertIsNone(result["program_eligible"])
        self.assertIsNone(result["score"])
        self.assertIsNone(result["delivery"]["answer_origin_valid"])
        self.measurement["rows"][0].pop("citation_source_valid")
        self.assertIsNone(self.view("aggregate")["delivery"]["citation_source_valid"])

    def test_resource_totals_come_from_rows_and_missing_is_not_zero(self):
        self.measurement["resource_usage"] = {"model_calls": 987654, "note": "RESOURCE_SENTINEL"}
        self.measurement["rows"][0]["resource_usage"].pop("search_calls")
        result = self.view("aggregate")
        self.assertEqual(result["resource_usage"]["model_calls"], 8)
        self.assertIsNone(result["resource_usage"]["search_calls"])
        self.assertNotIn("RESOURCE_SENTINEL", json.dumps(result))

    def test_invalid_counter_or_flag_types_are_not_silently_counted_as_false(self):
        for value in (True, -1, .5, "1"):
            fit = deepcopy(self.measurement)
            fit["rows"][0]["resource_usage"]["model_calls"] = value
            with self.subTest(counter=value), self.assertRaises(ValueError): self.view("aggregate", fit)
        for field in ("citation_source_valid", "answer_usable", "execution_ok"):
            fit = deepcopy(self.measurement)
            fit["rows"][0][field] = "false"
            with self.subTest(flag=field), self.assertRaises(ValueError): self.view("aggregate", fit)

    def test_aggregate_output_does_not_change_when_only_model_reports_change(self):
        original = self.view("aggregate")
        self.measurement["rows"][0]["candidate_reported"] = {"state": {"gaps": ["subjective diagnosis"], "conflicts": ["subjective conflict"]},
                                                          "failure_types": ["no_evidence_progress"]}
        self.assertEqual(self.view("aggregate"), original)

    def test_compact_feedback_is_used_only_for_validation_with_zero_selected_cases(self):
        original = controlled.compact_feedback
        with patch.object(controlled, "compact_feedback", wraps=original) as validator:
            self.view("cases")
        self.assertEqual(validator.call_count, 1)
        self.assertEqual(validator.call_args.kwargs, {"max_cases": 0})

    def test_unknown_condition_is_rejected(self):
        for condition in (None, "score_only", "all", ""):
            with self.subTest(condition=condition), self.assertRaises(ValueError): self.view(condition)


if __name__ == "__main__":
    unittest.main()
