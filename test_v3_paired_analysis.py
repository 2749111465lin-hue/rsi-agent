"""Synthetic statistical checks: no API, filesystem output, numpy or scipy."""
from copy import deepcopy
import json
import math
import random
import unittest

from code_rsi.v3.paired_analysis import paired_analysis, validate_analysis

ARMS = ["A", "B", "C"]


def specification(groups=None, **changes):
    value = {"schema": "rag-rsi-paired-analysis-1", "primary_metric": "answer_f1",
        "comparisons": [{"name": "planning", "baseline": "A", "candidate": "B"},
                        {"name": "iteration", "baseline": "B", "candidate": "C"}],
        "question_groups": groups or {"q0": "g0", "q1": "g1", "q2": "g2", "q3": "g3"},
        "confidence_level": .95, "bootstrap_samples": 1000, "bootstrap_seed": 17,
        "target_effect": .10, "power": .8}
    value.update(changes)
    return value


def rows_for(spec, values=None, repeats=2):
    values = values or {}
    rows = []
    for i, q in enumerate(spec["question_groups"]):
        for j, arm in enumerate(ARMS):
            scores = values.get((q, arm), [.1 * ((i + 2 * j) % 9)] * repeats)
            assert len(scores) == repeats
            for repeat, score in enumerate(scores):
                rows.append({"question_id": q, "arm": arm, "repeat": repeat,
                    "metrics": {"answer_em": float(score == 1), "answer_f1": score}, "program_eligible": True})
    return rows


def analyze(rows, spec, repeats=2):
    return paired_analysis(rows, spec, arm_names=ARMS, expected_repeats=repeats)


class PairedAnalysisTests(unittest.TestCase):
    def test_unequal_groups_use_question_weights_in_estimate_bootstrap_and_se(self):
        sizes = [1, 2, 3, 4]
        effects = [.5, 0., -.5, .25]
        groups = {}; values = {}
        for g, n in enumerate(sizes):
            for index in range(n):
                q = f"q{g}-{index}"
                groups[q] = f"g{g}"
                values[q, "A"] = [.5, .5]
                values[q, "B"] = [.5 + effects[g]] * 2
                values[q, "C"] = [.6, .6]
        spec = specification(groups)
        result = analyze(rows_for(spec, values), spec)
        comparison = result["qualified"]["comparisons"]["planning"]
        self.assertAlmostEqual(comparison["mean_delta"], 0.)
        self.assertNotAlmostEqual(comparison["mean_delta"], sum(effects) / 4)
        self.assertEqual(result["sample_size"], 10)
        self.assertEqual(result["cluster_count"], 4)
        self.assertEqual(comparison["question_sign_counts"], {"positive": 5, "zero": 2, "negative": 3})
        self.assertEqual(comparison["group_sign_counts"], {"positive": 2, "zero": 1, "negative": 1})
        self.assertEqual(comparison["per_group"]["g3"]["sample_size"], 4)
        rng = random.Random(spec["bootstrap_seed"])
        reference = []
        for _ in range(spec["bootstrap_samples"]):
            draw = [rng.randrange(4) for _ in range(4)]
            reference.append(sum(sizes[g] * effects[g] for g in draw) / sum(sizes[g] for g in draw))
        reference.sort()
        def quantile(p):
            position = (len(reference)-1) * p
            lo, hi = math.floor(position), math.ceil(position)
            return reference[lo] + (reference[hi]-reference[lo]) * (position-lo)
        interval = comparison["confidence_interval"]
        self.assertEqual(interval["status"], "estimated")
        self.assertAlmostEqual(interval["lower"], quantile(.0125))
        self.assertAlmostEqual(interval["upper"], quantile(.9875))
        self.assertAlmostEqual(interval["per_comparison_confidence_level"], .975)
        se = math.sqrt(4/3 * sum((n*d)**2 for n, d in zip(sizes, effects))) / 10
        design = comparison["design_sensitivity"]
        self.assertAlmostEqual(design["linearized_cluster_se"], se)
        self.assertEqual(design["status"], "pilot_variance_approximation")
        self.assertFalse(design["power_is_guaranteed"])
        self.assertGreater(design["mde"], 0)
        self.assertGreaterEqual(design["required_independent_groups"], 2)

    def test_repeats_do_not_inflate_sample_size_or_cluster_count(self):
        spec = specification()
        one = analyze(rows_for(spec, repeats=1), spec, 1)
        five = analyze(rows_for(spec, repeats=5), spec, 5)
        self.assertEqual(one["sample_size"], five["sample_size"])
        self.assertEqual(one["cluster_count"], five["cluster_count"])
        self.assertEqual(five["cell_count"], one["cell_count"] * 5)
        self.assertEqual(one["qualified"]["comparisons"], five["qualified"]["comparisons"])
        self.assertIsNone(one["qualified"]["arms"]["A"]["mean_repeat_sample_stddev"])
        self.assertIsNone(one["qualified"]["arms"]["A"]["mean_repeat_range"])
        self.assertIsNone(one["qualified"]["arms"]["A"]["adjacent_exact_match_flip_rate"])
        self.assertEqual(five["qualified"]["arms"]["A"]["mean_repeat_range"], 0.)

    def test_repeat_noise_is_observed_with_explicit_flip_denominator(self):
        spec = specification()
        rows = rows_for(spec, {("q0", "A"): [0., 1., 0.]}, repeats=3)
        arm = analyze(rows, spec, 3)["qualified"]["arms"]["A"]
        question = arm["per_question"]["q0"]
        self.assertAlmostEqual(question["repeat_sample_stddev"], math.sqrt(1/3))
        self.assertEqual(question["repeat_range"], 1.)
        self.assertEqual(question["adjacent_score_changes"], 2)
        self.assertEqual(question["adjacent_exact_match_flips"], 2)
        self.assertEqual(arm["adjacent_exact_match_flip_count"], 2)
        self.assertEqual(arm["adjacent_exact_match_flip_rate"], 2 / (4 * 2))
        self.assertFalse(arm["repeats_are_independent_questions"])

    def test_seed_and_canonical_question_order_are_reproducible(self):
        spec = specification()
        rows = rows_for(spec)
        before = deepcopy((rows, spec))
        first = analyze(rows, spec)
        self.assertEqual(first, analyze(list(reversed(rows)), spec))
        shuffled = deepcopy(spec)
        shuffled["question_groups"] = dict(reversed(list(shuffled["question_groups"].items())))
        self.assertEqual(first, analyze(rows, shuffled))
        changed = analyze(rows, {**spec, "bootstrap_seed": 18})
        self.assertNotEqual(first["qualified"]["resampling"]["schedule_sha256"],
                            changed["qualified"]["resampling"]["schedule_sha256"])
        self.assertEqual((rows, spec), before)
        self.assertTrue(first["qualified"]["resampling"]["same_draws_for_all_comparisons"])
        self.assertFalse(first["hypothesis_tests_performed"])

    def test_reversing_a_comparison_reverses_point_and_interval(self):
        spec = specification()
        rows = rows_for(spec)
        forward = analyze(rows, spec)["qualified"]["comparisons"]["iteration"]
        reverse = deepcopy(spec)
        reverse["comparisons"][1].update(baseline="C", candidate="B")
        backward = analyze(rows, reverse)["qualified"]["comparisons"]["iteration"]
        self.assertAlmostEqual(backward["mean_delta"], -forward["mean_delta"])
        for q, delta in forward["per_question"].items():
            self.assertAlmostEqual(backward["per_question"][q], -delta)
        if forward["confidence_interval"]["status"] == "estimated":
            self.assertAlmostEqual(backward["confidence_interval"]["lower"], -forward["confidence_interval"]["upper"])
            self.assertAlmostEqual(backward["confidence_interval"]["upper"], -forward["confidence_interval"]["lower"])
        self.assertAlmostEqual(backward["design_sensitivity"]["mde"], forward["design_sensitivity"]["mde"])

    def test_any_ineligible_cell_blocks_all_qualified_inference_and_keeps_raw(self):
        spec = specification()
        values = {("q0", "A"): [0., 0.], ("q0", "B"): [1., 1.]}
        rows = rows_for(spec, values)
        invalid = next(r for r in rows if r["question_id"] == "q0" and r["arm"] == "B")
        invalid["program_eligible"] = False
        result = analyze(rows, spec)
        self.assertEqual(result["status"], "protocol_invalid")
        self.assertFalse(result["quality_comparison_valid"])
        self.assertIsNone(result["qualified"])
        self.assertEqual(result["raw_diagnostics"]["comparisons"]["planning"]["per_question"]["q0"], 1.)
        self.assertEqual(result["sample_size"], 4)
        self.assertEqual(len(result["raw_diagnostics"]["ineligible_cells"]), 1)
        self.assertFalse(result["raw_diagnostics"]["successful_subset_analysis"])
        self.assertNotIn("confidence_interval", result["raw_diagnostics"]["comparisons"]["planning"])

    def test_single_group_has_no_interval_or_design_claim(self):
        spec = specification({f"q{i}": "one-cluster" for i in range(4)})
        for item in analyze(rows_for(spec), spec)["qualified"]["comparisons"].values():
            self.assertEqual(item["confidence_interval"]["status"], "insufficient_clusters")
            self.assertIsNone(item["confidence_interval"]["lower"])
            self.assertIsNone(item["confidence_interval"]["upper"])
            self.assertIsNone(item["design_sensitivity"]["mde"])
            self.assertIsNone(item["design_sensitivity"]["required_independent_groups"])

    def test_constant_observations_never_create_a_zero_width_evidence_interval(self):
        spec = specification()
        values = {(q, arm): [.5, .5] for q in spec["question_groups"] for arm in ARMS}
        for item in analyze(rows_for(spec, values), spec)["qualified"]["comparisons"].values():
            self.assertEqual(item["mean_delta"], 0.)
            self.assertEqual(item["confidence_interval"]["status"], "insufficient_variation")
            self.assertIsNone(item["confidence_interval"]["lower"])
            self.assertIsNone(item["confidence_interval"]["upper"])
            self.assertIsNone(item["design_sensitivity"]["linearized_cluster_se"])
            self.assertIsNone(item["design_sensitivity"]["mde"])

    def test_equal_cluster_means_can_be_degenerate_despite_question_variation(self):
        spec = specification({"q0": "g0", "q1": "g0", "q2": "g1", "q3": "g1"})
        values = {(q, "A"): [.5, .5] for q in spec["question_groups"]}
        values.update({(q, "B"): [i % 2, i % 2] for i, q in enumerate(spec["question_groups"])})
        comparison = analyze(rows_for(spec, values), spec)["qualified"]["comparisons"]["planning"]
        self.assertEqual(comparison["confidence_interval"]["status"], "degenerate")
        self.assertIsNone(comparison["confidence_interval"]["lower"])
        self.assertIsNone(comparison["design_sensitivity"]["mde"])

    def test_metric_choice_and_extreme_config_remain_finite(self):
        spec = specification(primary_metric="answer_em", confidence_level=math.nextafter(1., 0.),
                             target_effect=math.nextafter(0., 1.))
        values = {("q0", "B"): [1., 1.], ("q1", "C"): [1., 1.]}
        result = analyze(rows_for(spec, values), spec)
        item = result["qualified"]["comparisons"]["planning"]
        self.assertEqual(item["mean_delta"], .25)
        self.assertEqual(item["design_sensitivity"]["availability"], "group_count_exceeds_numeric_range")
        self.assertIsNone(item["design_sensitivity"]["required_independent_groups"])
        json.dumps(result, allow_nan=False)

    def test_missing_duplicate_foreign_and_malformed_rows_are_rejected(self):
        spec = specification(); base = rows_for(spec)
        invalid_sets = [base[:-1], base + [deepcopy(base[0])], []]
        for field, value in (("question_id", "foreign"), ("arm", "foreign"), ("repeat", True),
                             ("repeat", 2), ("program_eligible", 1), ("question_id", ["q0"])):
            rows = deepcopy(base); rows[0][field] = value; invalid_sets.append(rows)
        for rows in invalid_sets:
            with self.subTest(row_count=len(rows)):
                with self.assertRaises(ValueError): analyze(rows, spec)
        with self.assertRaises(ValueError): analyze(None, spec)
        for invalid in (math.nan, math.inf, -math.inf, -.01, 1.01, True, "0.5", None, 10**1000):
            for metric in ("answer_em", "answer_f1"):
                rows = deepcopy(base); rows[0]["metrics"][metric] = invalid
                with self.subTest(metric=metric, value=invalid):
                    with self.assertRaises(ValueError): analyze(rows, spec)
        for repeats in (0, True, 1.5):
            with self.assertRaises(ValueError): analyze(base, spec, repeats)

    def test_strict_preregistered_schema_and_bounds(self):
        spec = specification()
        for field in spec:
            broken = deepcopy(spec); broken.pop(field)
            with self.subTest(missing=field), self.assertRaises(ValueError):
                validate_analysis(broken, question_ids=list(spec["question_groups"]), arm_names=ARMS)
        invalids = [dict(spec, extra="posthoc"), dict(spec, primary_metric="retrieval"),
            dict(spec, schema="other"), dict(spec, comparisons=spec["comparisons"][:1]),
            dict(spec, question_groups={"q0": "g0"})]
        for field, values in {"confidence_level": [.79, 1., math.nan, True],
            "bootstrap_samples": [999, 100001, 1000., True], "bootstrap_seed": [-1, True, 1.5],
            "target_effect": [0., -1., 1.01, True], "power": [.5, 1., math.inf]}.items():
            invalids += [dict(spec, **{field: value}) for value in values]
        same = deepcopy(spec); same["comparisons"][1].update(baseline="B", candidate="A")
        invalids.append(same)
        for field, value in (("name", "planning"), ("baseline", "C"), ("candidate", "foreign")):
            broken = deepcopy(spec); broken["comparisons"][1][field] = value; invalids.append(broken)
        for field in ("baseline", "candidate"):
            for value in (None, True, 1, [], ["A"], {"arm": "A"}):
                broken = deepcopy(spec); broken["comparisons"][0][field] = value; invalids.append(broken)
        invalids.append(dict(spec, confidence_level=10**1000))
        for broken in invalids:
            with self.subTest(spec=broken), self.assertRaises(ValueError):
                validate_analysis(broken, question_ids=list(spec["question_groups"]), arm_names=ARMS)
        normalized = validate_analysis(spec, question_ids=reversed(list(spec["question_groups"])), arm_names=ARMS)
        normalized["comparisons"][0]["name"] = "independent-copy"
        self.assertEqual(spec["comparisons"][0]["name"], "planning")
        for changes in ({"confidence_level": .8, "target_effect": 1., "bootstrap_samples": 100000},
                        {"confidence_level": .99, "power": .9, "bootstrap_seed": 0}):
            validate_analysis({**spec, **changes}, question_ids=list(spec["question_groups"]), arm_names=ARMS)


if __name__ == "__main__":
    unittest.main()
