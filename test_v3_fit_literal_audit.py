"""Synthetic fit-literal controls; no API, execution, references or held-out data."""
from copy import deepcopy
import json
import unittest

from code_rsi.v3.fit_literal_audit import audit_fit_literals, RULES

QUESTION = ("Which northern observatory maintains the Larchmere spectral navigation archive "
            "and records the cobalt lighthouse calibration expedition during the winter solstice?")
QID = "synthetic-question-017"
PREDICTION = "The Northern Cartographic Institute at Larchmere Harbor maintains the spectral navigation archive."
EVIDENCE = ("The Larchmere expedition ledger identifies the Northern Cartographic Institute as the custodian "
            "of the spectral navigation archive, with cobalt lighthouse observations preserved in the winter calibration register.")


def files(code="pass\n"):
    return {"rag.py": "def solve(question, services):\n    return {'answer': ''}\n", "rag_core.py": code}


def tasks(question=QUESTION, qid=QID):
    return [{"question_id": qid, "question": question}]


def feedback(prediction=PREDICTION, evidence=EVIDENCE):
    return {"role": "D_fit", "cases": [{"question_id": QID, "prediction": prediction,
            "evidence_witnesses": [{"quote_excerpt": evidence}]}]}


def audit(code, *, parent=None, public=None, exposed=None):
    return audit_fit_literals(parent or files(), files(code), public or tasks(), exposed_feedback=exposed)


class FitLiteralAuditTests(unittest.TestCase):
    def assert_rejected(self, code, kind, **kwargs):
        result = audit(code, **kwargs)
        self.assertEqual(result["status"], "reject")
        self.assertIn(kind, {x["kind"] for x in result["findings"]})
        self.assertTrue(all(x["line"] >= 1 and len(x["fingerprint"]) == 64 for x in result["findings"]))
        return result

    def test_full_question_in_branches_lookup_lists_and_system_guidance(self):
        examples = ["if question == " + repr(QUESTION) + ":\n    answer = 'yes'\n",
                    "LOOKUP = {" + repr(QUESTION) + ": 'yes'}\n",
                    "PROMPT = " + repr("System guidance: answer this exact task: " + QUESTION) + "\n",
                    "MATERIAL = {'cases': [" + repr(QUESTION) + "]}\n"]
        for code in examples:
            with self.subTest(code_kind=code.split('=')[0]):
                self.assert_rejected(code, "full_question")

    def test_static_add_join_and_fstrings_are_examined_without_execution(self):
        first, last = QUESTION[:70], QUESTION[70:]
        examples = ["PROMPT = " + repr(first) + " + " + repr(last) + "\n",
                    "PROMPT = ''.join([" + repr(first) + ", " + repr(last) + "])\n",
                    "PROMPT = f" + repr(QUESTION + " {question}") + "\n",
                    "PROMPT = f" + repr(QUESTION + "{''}") + "\n"]
        for code in examples:
            with self.subTest(code=code[:30]): self.assert_rejected(code, "full_question")
        # An unknown interpolation is not silently assumed to be an empty string.
        short_question = "Which port owns the blue lantern archive?"
        code = "PROMPT = f'Which port owns {unknown()}the blue lantern archive?'\n"
        self.assertEqual(audit(code, public=tasks(short_question))["status"], "pass")

    def test_unicode_case_and_whitespace_are_normalized(self):
        spaced = QUESTION.upper().replace(" ", "\t\n  ")
        wide = "".join(chr(ord(c) + 0xFEE0) if 'A' <= c <= 'Z' else c for c in spaced)
        wide = wide.replace("ＯＢＳＥＲＶＡＴＯＲＹ", "ＯＢＳＥＲ\u200bＶＡＴＯＲＹ")
        self.assert_rejected("PROMPT = " + repr(wide) + "\n", "full_question")
        chinese = "请指出保存蓝色灯塔冬至校准记录和光谱导航档案的北方观测站正式名称是什么？"
        self.assert_rejected("TEXT = " + repr(chinese) + "\n", "full_question", public=tasks(chinese))

    def test_long_contiguous_question_fragment_is_rejected(self):
        fragment = QUESTION[15:100]
        result = self.assert_rejected("GUIDANCE = " + repr("Use the recorded context: " + fragment) + "\n", "question_fragment")
        finding = next(x for x in result["findings"] if x["kind"] == "question_fragment")
        self.assertEqual(finding["matched_chars"], RULES["question_fragment_chars"])

    def test_long_exposed_prediction_and_evidence_are_rejected(self):
        self.assert_rejected("FALLBACK = " + repr(PREDICTION) + "\n", "exposed_prediction", exposed=feedback())
        self.assert_rejected("EVIDENCE = " + repr(EVIDENCE) + "\n", "exposed_evidence", exposed=feedback())
        # Only text actually exposed is searched; hidden reference fields are ignored.
        no_exposure = {"role": "D_fit", "cases": [{"question_id": QID}], "reference_answer": PREDICTION}
        self.assertEqual(audit("FALLBACK = " + repr(PREDICTION) + "\n", exposed=no_exposure)["status"], "pass")

    def test_already_displayed_flow_answer_and_quote_excerpts_are_whitelisted(self):
        exposed = {"role": "D_fit", "cases": [{"question_id": QID,
            "parent_execution_flow": {"final": {"answer_excerpt": PREDICTION},
                                      "reads": [{"quotes": [{"quote_excerpt": EVIDENCE}]}]}}]}
        result = self.assert_rejected("A = " + repr(PREDICTION) + "\nB = " + repr(EVIDENCE) + "\n", "exposed_prediction", exposed=exposed)
        self.assertIn("exposed_evidence", {x["kind"] for x in result["findings"]})

    def test_short_answers_common_schema_and_reusable_prompts_are_not_blanket_rejected(self):
        generic = "I cannot determine the answer from the provided sources because there is insufficient information."
        code = ("CONFIG = {'1': 'yes', 'year': '2024', 'city': 'Paris', 'question_id': 'id'}\n"
                "PROMPT = 'Answer the question using only provided sources. Return JSON with answer and citation_ids.'\n"
                "FALLBACK = " + repr(generic) + "\n")
        for short in ("yes", "no", "2024", "Paris", generic):
            with self.subTest(prediction=short):
                self.assertEqual(audit(code, exposed=feedback(short, "Short quote"))["status"], "pass")
        self.assertEqual(audit("PROMPT = 'Why?'\n", public=tasks("Why?"))["status"], "pass")

    def test_specific_long_ids_require_a_route_position(self):
        self.assertEqual(audit("LABEL = " + repr(QID) + "\n")["status"], "pass")
        self.assert_rejected("if value == " + repr(QID) + ":\n    pass\n", "question_id_route")
        self.assert_rejected("LOOKUP = {" + repr(QID) + ": 'yes'}\n", "question_id_route")
        self.assert_rejected("value = lookup.get(" + repr(QID) + ")\n", "question_id_route")

    def test_short_numeric_ids_require_explicit_input_linkage(self):
        public = tasks(qid="1")
        benign = "CONFIG = {'1': 3}\nif year == '1':\n    pass\nx = CONFIG['1']\n"
        self.assertEqual(audit(benign, public=public)["status"], "pass")
        routes = ["if question_id == '1':\n    pass\n",
                  "if task.get('id') == '1':\n    pass\n",
                  "value = {'1': 'yes'}[task['question_id']]\n",
                  "def route(task):\n    table = {'1': 'yes'}\n    return table[task['id']]\n",
                  "def route(id):\n    return id == '1'\n",
                  "match task['id']:\n    case '1':\n        pass\n"]
        for code in routes:
            with self.subTest(code=code): self.assert_rejected(code, "question_id_route", public=public)

    def test_new_routing_of_an_old_nonrouting_id_is_not_grandfathered(self):
        parent = files("LABEL = '1'\n")
        self.assert_rejected("if qid == '1':\n    pass\n", "question_id_route", parent=parent, public=tasks(qid="1"))

    def test_parent_examples_are_not_new_injections_and_new_duplicate_instances_are(self):
        parent = files("EXAMPLE = " + repr(QUESTION) + "\n")
        same = parent["rag_core.py"] + "LIMIT = 5\n"
        self.assertEqual(audit(same, parent=parent)["status"], "pass")
        edited = "EXAMPLE = " + repr("Existing example: " + QUESTION + " Keep prior context.") + "\n"
        result = audit(edited, parent=parent)
        self.assertEqual(result["status"], "pass")
        self.assertGreater(result["coverage"]["parent_matches_exempted"], 0)
        duplicate = parent["rag_core.py"] + "NEW_COPY = " + repr(QUESTION) + "\n"
        result = self.assert_rejected(duplicate, "full_question", parent=parent)
        self.assertEqual(result["findings"][0]["line"], 2)

    def test_new_static_assembly_of_preexisting_short_parts_is_detected(self):
        short_question = "Which port owns the blue lantern archive?"
        left, right = short_question[:20], short_question[20:]
        parent = files("A = " + repr(left) + "\nB = " + repr(right) + "\n")
        child = "FULL = " + repr(left) + " + " + repr(right) + "\n"
        self.assert_rejected(child, "full_question", parent=parent, public=tasks(short_question))

    def test_input_immutability_determinism_and_no_text_in_receipt(self):
        public = tasks(); exposed = feedback(); parent = files(); child = files("TEXT = " + repr(QUESTION + PREDICTION + EVIDENCE) + "\n")
        before = deepcopy((public, exposed, parent, child))
        result = audit_fit_literals(parent, child, public, exposed_feedback=exposed)
        self.assertEqual((public, exposed, parent, child), before)
        self.assertEqual(result, audit_fit_literals(parent, child, public, exposed_feedback=exposed))
        serialized = json.dumps(result, ensure_ascii=False)
        for text in (QUESTION, QUESTION.casefold(), PREDICTION, EVIDENCE, QID):
            self.assertNotIn(text, serialized)
        self.assertFalse(result["scope"]["references_read"])
        self.assertEqual(result["scope"]["role"], "D_fit")
        self.assertEqual(len(result["receipt_sha256"]), 64)
        self.assertTrue(any("not proof" in x for x in result["limitations"]))

    def test_private_reference_objects_are_never_read_or_used_as_an_oracle(self):
        class NeverInspect:
            def __str__(self): raise AssertionError("private object inspected")
            def __repr__(self): raise AssertionError("private object inspected")
            def __iter__(self): raise AssertionError("private object inspected")
        public = tasks(); public[0]["references"] = NeverInspect()
        exposed = {"role": "D_fit", "cases": [{"question_id": QID}], "references": NeverInspect(),
                   "answers": NeverInspect(), "reference_answer": "PRIVATE_REFERENCE_SENTINEL_WITH_LONG_DISTINCT_CONTENT"}
        code = "S = 'PRIVATE_REFERENCE_SENTINEL_WITH_LONG_DISTINCT_CONTENT'\n"
        first = audit(code, public=public, exposed=exposed)
        exposed["reference_answer"] = "DIFFERENT_PRIVATE_REFERENCE"
        second = audit(code, public=public, exposed=exposed)
        self.assertEqual(first["status"], "pass")
        self.assertEqual(first, second)
        self.assertNotIn("PRIVATE_REFERENCE", json.dumps(first))

    def test_nonfit_and_foreign_cases_fail_before_text_is_used(self):
        for role in ("D_select", "D_report"):
            public = tasks(); public[0].update(role=role, question=object())
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "D_fit"):
                audit("pass\n", public=public)
            with self.assertRaisesRegex(ValueError, "D_fit"):
                audit("pass\n", exposed={"role": role, "cases": object()})
        with self.assertRaisesRegex(ValueError, "outside the fit"):
            audit("pass\n", exposed={"role": "D_fit", "cases": [{"question_id": "foreign", "prediction": object()}]})

    def test_comments_are_outside_static_literal_scope(self):
        self.assertEqual(audit("# " + QUESTION + "\npass\n")["status"], "pass")
        result = audit("TEXT = ''.join(chr(x) for x in [81, 50])\n")
        self.assertEqual(result["status"], "pass")
        self.assertTrue(any("computed/encoded" in x for x in result["limitations"]))


if __name__ == "__main__":
    unittest.main()
