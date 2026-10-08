"""Synthetic single-contrast analysis boundaries; no API or benchmark data."""
from copy import deepcopy
import unittest

from code_rsi.v3.paired_analysis import (
    SINGLE_CONTRAST_SCHEMA, SPEC_SCHEMA, paired_analysis, validate_analysis,
)

ARMS = ["planned", "loop"]


def specification():
    return {"schema": SINGLE_CONTRAST_SCHEMA, "primary_metric": "answer_f1",
            "comparisons": [{"name": "iteration", "baseline": "planned", "candidate": "loop"}],
            "question_groups": {"q0": "shared", "q1": "shared", "q2": "independent2", "q3": "independent3"},
            "confidence_level": .95, "bootstrap_samples": 1000, "bootstrap_seed": 17,
            "target_effect": .1, "power": .8}


def rows():
    scores = {"q0": [1., .5], "q1": [.5, 1.], "q2": [0., 0.], "q3": [1., 1.]}
    return [{"question_id": q, "arm": arm, "repeat": r, "program_eligible": True,
             "metrics": {"answer_f1": value, "answer_em": float(value == 1)}}
            for q, candidate in scores.items() for arm in ARMS
            for r, value in enumerate(candidate if arm == "loop" else [.5, .5])]


def analyze(values=None, spec=None):
    return paired_analysis(rows() if values is None else values, specification() if spec is None else spec,
                           arm_names=ARMS, expected_repeats=2)


class SingleContrastTests(unittest.TestCase):
    def test_one_contrast_averages_two_repeats_and_resamples_shared_groups(self):
        result = analyze()
        self.assertEqual(result["status"], "qualified")
        self.assertEqual(result["analysis"]["schema"], SINGLE_CONTRAST_SCHEMA)
        self.assertEqual((result["sample_size"], result["cluster_count"], result["cell_count"]), (4, 3, 16))
        main = result["qualified"]["comparisons"]["iteration"]
        self.assertEqual(main["per_question"], {"q0": .25, "q1": .25, "q2": -.5, "q3": .5})
        self.assertEqual(main["mean_delta"], .125)
        self.assertNotAlmostEqual(main["mean_delta"], (.25 - .5 + .5) / 3)
        self.assertEqual(main["per_group"]["shared"]["sample_size"], 2)
        self.assertEqual(main["per_group"]["shared"]["sum_delta"], .5)
        interval = main["confidence_interval"]
        self.assertEqual(interval["status"], "estimated")
        self.assertEqual(interval["per_comparison_confidence_level"], .95)
        self.assertEqual((interval["lower"], interval["upper"]), (-.5, .5))
        noise = result["qualified"]["arms"]["loop"]
        self.assertEqual(noise["questions_with_repeat_variation"], 2)
        self.assertEqual(noise["adjacent_exact_match_flip_rate"], .5)
        self.assertFalse(noise["repeats_are_independent_questions"])

    def test_old_schema_still_requires_two_comparisons(self):
        spec = specification()
        spec["schema"] = SPEC_SCHEMA
        with self.assertRaisesRegex(ValueError, "exactly two"):
            validate_analysis(spec, question_ids=list(spec["question_groups"]), arm_names=ARMS)

    def test_single_schema_cannot_accept_multiple_or_zero_comparisons(self):
        for comparisons in ([], specification()["comparisons"] +
                            [{"name": "oracle", "baseline": "planned", "candidate": "oracle"}]):
            spec = specification()
            spec["comparisons"] = comparisons
            with self.assertRaisesRegex(ValueError, "exactly one"):
                validate_analysis(spec, question_ids=list(spec["question_groups"]), arm_names=ARMS + ["oracle"])

    def test_missing_duplicate_or_oracle_rows_are_not_silently_filtered(self):
        values = rows()
        oracle = deepcopy(values[0])
        oracle["arm"] = "oracle"
        for invalid in (values[:-1], values + [values[0]], values + [oracle]):
            with self.assertRaises(ValueError):
                analyze(invalid)

    def test_ineligible_row_keeps_all_questions_but_blocks_quality_comparison(self):
        values = rows()
        broken = next(x for x in values if x["question_id"] == "q2" and x["arm"] == "loop")
        broken["program_eligible"] = False
        result = analyze(values)
        self.assertEqual(result["status"], "protocol_invalid")
        self.assertFalse(result["quality_comparison_valid"])
        self.assertIsNone(result["qualified"])
        self.assertEqual((result["sample_size"], result["cluster_count"], result["cell_count"]), (4, 3, 16))
        raw = result["raw_diagnostics"]
        self.assertEqual(raw["comparisons"]["iteration"]["mean_delta"], .125)
        self.assertEqual(set(raw["comparisons"]["iteration"]["per_question"]), set(specification()["question_groups"]))
        self.assertEqual(len(raw["ineligible_cells"]), 1)
        self.assertFalse(raw["successful_subset_analysis"])
        self.assertNotIn("confidence_interval", raw["comparisons"]["iteration"])

    def test_single_connected_group_does_not_create_independent_questions(self):
        spec = specification()
        spec["question_groups"] = {q: "one-connected-component" for q in spec["question_groups"]}
        result = analyze(spec=spec)
        main = result["qualified"]["comparisons"]["iteration"]
        self.assertEqual(result["cluster_count"], 1)
        self.assertEqual(main["confidence_interval"]["status"], "insufficient_clusters")
        self.assertIsNone(main["confidence_interval"]["lower"])
        self.assertIsNone(main["design_sensitivity"]["mde"])

    def test_normalization_preserves_new_schema_without_mutating_input(self):
        spec = specification()
        original = deepcopy(spec)
        normalized = validate_analysis(spec, question_ids=list(spec["question_groups"]), arm_names=ARMS)
        self.assertEqual(normalized["schema"], SINGLE_CONTRAST_SCHEMA)
        normalized["comparisons"][0]["name"] = "changed-copy"
        normalized["question_groups"]["q0"] = "another-copy"
        self.assertEqual(spec, original)
        self.assertEqual(analyze(list(reversed(rows()))), analyze())


if __name__ == "__main__":
    unittest.main()
