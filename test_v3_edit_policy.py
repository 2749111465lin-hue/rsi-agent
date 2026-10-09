"""Prompt scope tests use synthetic strings only, never executing candidates."""
from copy import deepcopy
import hashlib
import json
import unittest
from unittest.mock import patch

from code_rsi.budget import digest
from code_rsi.v3.edit_policy import SCHEMA, check_edit_policy, validate_edit_policy
from code_rsi.v3.execution import root_files


def policy(*stages):
    return {"schema": SCHEMA, "mode": "prompt_only", "allowed_prompt_stages": list(stages or ("plan", "read", "answer"))}


def replace_config(files, expression):
    lines = files["rag.py"].splitlines(keepends=True)
    lines = ["CONFIG = " + expression + "\n" if line.startswith("CONFIG = ") else line for line in lines]
    return {**files, "rag.py": "".join(lines)}


class EditPolicyTests(unittest.TestCase):
    def test_policy_legacy_program_and_canonical_stage_order(self):
        self.assertIsNone(validate_edit_policy(None))
        explicit = {"schema": SCHEMA, "mode": "program"}
        self.assertEqual(validate_edit_policy(explicit), explicit)
        source = policy("answer", "plan")
        clean = validate_edit_policy(source)
        self.assertEqual(clean["allowed_prompt_stages"], ["plan", "answer"])
        self.assertEqual(source["allowed_prompt_stages"], ["answer", "plan"])

    def test_malformed_policy_raises_before_sources(self):
        invalid = [False, "program", {}, {"schema": "other", "mode": "program"},
            {"schema": SCHEMA, "mode": "other"}, {"schema": SCHEMA, "mode": "program", "extra": 1},
            {"schema": SCHEMA, "mode": "program", "allowed_prompt_stages": ["plan"]},
            {"schema": SCHEMA, "mode": "prompt_only"},
            {**policy(), "allowed_prompt_stages": []},
            {**policy(), "allowed_prompt_stages": ["plan", "plan"]},
            {**policy(), "allowed_prompt_stages": [True]},
            {**policy(), "allowed_prompt_stages": ["develop"]},
            {**policy(), "allowed_prompt_stages": ("plan",)}, {**policy(), "extra": 1}]
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ValueError):
                check_edit_policy(None, None, item)

    def test_program_mode_does_not_change_legacy_source_constraints(self):
        before = root_files()
        after = {**before, "rag_core.py": "not syntactically valid !"}
        for setting in (None, {"schema": SCHEMA, "mode": "program"}):
            receipt = check_edit_policy(before, after, setting)
            self.assertTrue(receipt["allowed"])
            self.assertIsNone(receipt["changed_prompt_stages"])

    def test_add_modify_and_remove_declared_prompts(self):
        configurations = [({}, {"prompts": {"answer": "Answer briefly."}}),
            ({"prompts": {"answer": "Old"}}, {"prompts": {"answer": "New"}}),
            ({"prompts": {"answer": "Old"}}, {"prompts": {}})]
        for before, after in configurations:
            with self.subTest(before=before):
                result = check_edit_policy(root_files(before), root_files(after), policy("answer"))
                self.assertTrue(result["allowed"], result)
                self.assertEqual(result["changed_prompt_stages"], ["answer"])

    def test_static_dict_and_root_json_forms(self):
        before = root_files({"max_rounds": 3})
        for original in (before, replace_config(before, "{'max_rounds': 3}")):
            after = replace_config(original, "{'max_rounds': 3, 'prompts': {'read': 'Exact quotes.'}}")
            result = check_edit_policy(original, after, policy("read"))
            self.assertTrue(result["allowed"], result)

    def test_string_that_looks_like_code_is_only_prompt_text(self):
        text = "exec(__import__('os').system('never run')); CONFIG = {}; json = fake"
        with patch("builtins.eval", side_effect=AssertionError("eval must not run")):
            result = check_edit_policy(root_files(), root_files({"prompts": {"plan": text}}), policy("plan"))
        self.assertTrue(result["allowed"], result)

    def test_unknown_or_unallowed_stage_rejected(self):
        result = check_edit_policy(root_files(), root_files({"prompts": {"read": "x"}}), policy("answer"))
        self.assertIn("prompt_stage_not_allowed", result["reason_codes"])
        unknown = check_edit_policy(root_files(), root_files({"prompts": {"develop": "x"}}), policy())
        self.assertFalse(unknown["allowed"])

    def test_empty_or_presentation_only_edit_is_not_real_guidance_change(self):
        before = root_files()
        for after in (before, {**before, "rag.py": "# comment\n" + before["rag.py"]},
                      root_files({"prompts": {}}), root_files({"prompts": {"answer": ""}})):
            with self.subTest(after=after["rag.py"][:30]):
                receipt = check_edit_policy(before, after, policy())
                self.assertIn("no_prompt_change", receipt["reason_codes"])
        receipt = check_edit_policy(root_files({"prompts": {"answer": ""}}), before, policy())
        self.assertIn("no_prompt_change", receipt["reason_codes"])

    def test_char_limit_matches_engine_not_utf8_bytes(self):
        self.assertTrue(check_edit_policy(root_files(), root_files({"prompts": {"answer": "文" * 8000}}), policy())["allowed"])
        for value in ("x" * 8001, 1, False, None, ["text"], {"text": "x"}):
            with self.subTest(type=type(value)):
                self.assertFalse(check_edit_policy(root_files(), root_files({"prompts": {"answer": value}}), policy())["allowed"])

    def test_nonprompt_budget_bool_int_float_and_signed_zero_changes_rejected(self):
        for old, new in ((3, 4), (1, True), (1, 1.0), (0.0, -0.0)):
            before = root_files({"max_rounds": old})
            after = root_files({"max_rounds": new, "prompts": {"answer": "Valid text"}})
            result = check_edit_policy(before, after, policy())
            self.assertIn("nonprompt_config_changed", result["reason_codes"])
        result = check_edit_policy(root_files({"mode": "iterative"}), root_files({"mode": "single_pass", "prompts": {"answer": "x"}}), policy())
        self.assertIn("nonprompt_config_changed", result["reason_codes"])

    def test_other_wrapper_ast_and_core_bytes_cannot_change(self):
        before, child = root_files(), root_files({"prompts": {"answer": "New"}})
        variants = [child["rag.py"] + "\nSCHEMAS = {}\n",
            child["rag.py"] + "\nimport math\n",
            child["rag.py"].replace("limit=5", "limit=4"),
            child["rag.py"].replace("config=CONFIG", "config={}"),
            child["rag.py"] + "\nCONFIG['max_rounds'] = 99\n",
            child["rag.py"] + "\nCONFIG.update({'max_rounds': 99})\n",
            child["rag.py"] + "\nalias = CONFIG\nalias['max_rounds'] = 99\n"]
        for source in variants:
            self.assertFalse(check_edit_policy(before, {**child, "rag.py": source}, policy())["allowed"])
        for core in (child["rag_core.py"] + "\n# even a comment\n", child["rag_core.py"].replace("SCHEMAS = {", "SCHEMAS = dict(")):
            self.assertIn("core_source_changed", check_edit_policy(before, {**child, "rag_core.py": core}, policy())["reason_codes"])

    def test_json_duplicate_keys_at_any_depth_rejected(self):
        for text in ('{"prompts": {}, "prompts": {"answer": "new"}}',
                     '{"prompts": {"answer": "one", "answer": "two"}}',
                     '{"extra": {"x": 1, "x": 2}, "prompts": {"answer": "new"}}'):
            child = replace_config(root_files(), "json.loads(" + repr(text) + ")")
            self.assertIn("child_duplicate_or_nonstring_config_key", check_edit_policy(root_files(), child, policy())["reason_codes"])

    def test_ast_duplicate_nonstring_and_unpacked_keys_rejected(self):
        for expression in ("{'prompts': {}, 'prompts': {'answer': 'new'}}",
            "{'prompts': {'answer': 'one', 'answer': 'two'}}",
            "{'extra': {'x': 1, 'x': 2}, 'prompts': {'answer': 'new'}}",
            "{1: 1, True: 2, 'prompts': {'answer': 'new'}}", "{**{}, 'prompts': {'answer': 'new'}}"):
            child = replace_config(root_files(), expression)
            self.assertFalse(check_edit_policy(root_files(), child, policy())["allowed"])

    def test_nonfinite_json_and_ast_numbers_rejected(self):
        for token in ("NaN", "Infinity", "-Infinity", "1e999"):
            text = '{"extra": ' + token + ', "prompts": {"answer": "new"}}'
            child = replace_config(root_files(), "json.loads(" + repr(text) + ")")
            self.assertFalse(check_edit_policy(root_files(), child, policy())["allowed"])
        child = replace_config(root_files(), "{'extra': 1e999, 'prompts': {'answer': 'new'}}")
        self.assertFalse(check_edit_policy(root_files(), child, policy())["allowed"])

    def test_dynamic_config_is_not_executed(self):
        for expression in ("__import__('os').system('SHOULD_NOT_RUN')", "factory()", "dict(prompts={'answer':'x'})",
                           "{'prompts': {'answer': 'x' + 'y'}}", "{'prompts': {'answer': f'{side_effect()}'}}",
                           "json.loads(fetch())", "json.loads('{}', object_hook=side_effect)"):
            child = replace_config(root_files(), expression)
            with patch("os.system", side_effect=AssertionError("candidate executed")):
                self.assertFalse(check_edit_policy(root_files(), child, policy())["allowed"])

    def test_shadowed_json_binding_and_config_rebinding_rejected(self):
        before = root_files()
        changed = root_files({"prompts": {"answer": "new"}})
        fragments = ["json = fake\n", "def json(): pass\n", "class json: pass\n",
            "from other import json\n", "import other as json\n", "import json\n",
            "json.loads = replacement\n", "CONFIG = {}\n", "CONFIG: dict = {}\n",
            "CONFIG = alias = {}\n", "def f(json): pass\n",
            "match {}:\n    case json: pass\n", "del json\n"]
        for fragment in fragments:
            for source in (fragment + changed["rag.py"], changed["rag.py"] + fragment):
                with self.subTest(fragment=fragment):
                    self.assertFalse(check_edit_policy(before, {**changed, "rag.py": source}, policy())["allowed"])

    def test_parent_dynamic_or_shadowed_behavior_is_also_rejected(self):
        for prefix, suffix in (("json = replacement\n", ""), ("", "\nexec(CONFIG['prompts']['answer'])\n"),
                              ("", "\nsetattr(json, 'loads', replacement)\n"),
                              ("", "\nrunner = eval\n"), ("", "\nx = globals()\n")):
            before = root_files({"prompts": {"answer": "before"}})
            after = root_files({"prompts": {"answer": "after"}})
            for files in (before, after): files["rag.py"] = prefix + files["rag.py"] + suffix
            result = check_edit_policy(before, after, policy("answer"))
            self.assertFalse(result["allowed"])
            self.assertTrue(any(x.startswith("parent_") for x in result["reason_codes"]))

    def test_missing_json_import_or_import_after_config_rejected(self):
        before = root_files()
        after = root_files({"prompts": {"answer": "new"}})
        for source in (after["rag.py"].replace("import json\n", ""),
                       after["rag.py"].replace("import json\n", "") + "\nimport json\n"):
            self.assertFalse(check_edit_policy(before, {**after, "rag.py": source}, policy())["allowed"])

    def test_encoding_cookie_cannot_change_python_file_loading_semantics(self):
        before = root_files()
        after = root_files({"prompts": {"answer": "new"}})
        for cookie in ("# coding: latin1\n", "# coding: unicode_escape\n", "#!/usr/bin/env python\n# coding: iso-8859-1\n"):
            result = check_edit_policy(before, {**after, "rag.py": cookie + after["rag.py"]}, policy())
            self.assertIn("child_non_utf8_source_encoding", result["reason_codes"])
        for cookie in ("# coding: utf-8\n", "# coding: utf-8-sig\n"):
            self.assertTrue(check_edit_policy(before, {**after, "rag.py": cookie + after["rag.py"]}, policy())["allowed"])

    def test_malformed_sources_return_bounded_rejection_receipts(self):
        before = root_files()
        for child in (None, {}, {"rag.py": "bad"}, {**before, "extra.py": "x"},
                      {**before, "rag.py": b"bytes"}, {**before, "rag.py": "if bad syntax !!!"},
                      replace_config(before, "[]"), replace_config(before, "{'prompts': None}")):
            result = check_edit_policy(before, child, policy())
            self.assertFalse(result["allowed"])
            self.assertTrue(result["reason_codes"])

    def test_receipt_hash_binds_both_sources_policy_and_decision(self):
        before = root_files()
        after = root_files({"prompts": {"read": "test"}})
        before_snapshot, after_snapshot = deepcopy(before), deepcopy(after)
        result = check_edit_policy(before, after, policy("read"))
        self.assertTrue(result["allowed"])
        unsigned = {key: value for key, value in result.items() if key != "receipt_sha256"}
        self.assertEqual(result["receipt_sha256"], digest(unsigned))
        self.assertEqual(result["parent_source_sha256"]["rag.py"], hashlib.sha256(before["rag.py"].encode()).hexdigest())
        self.assertEqual(result, check_edit_policy(before, after, policy("read")))
        self.assertEqual((before, after), (before_snapshot, after_snapshot))
        self.assertNotEqual(result["receipt_sha256"], check_edit_policy(before, after, policy("read", "answer"))["receipt_sha256"])


if __name__ == "__main__":
    unittest.main()
