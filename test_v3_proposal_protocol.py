"""Pure synthetic edit protocol tests; no model, files or candidate execution."""
from copy import deepcopy
import hashlib
from types import MappingProxyType
import unittest
from unittest.mock import patch

from code_rsi.budget import digest, stable
from code_rsi.v3.proposal_protocol import (SCHEMA, materialize_edits, source_identity,
                                          validate_proposal_protocol)


def protocol(**overrides):
    return {"schema": SCHEMA, "format": "exact_edits", "max_edits": 8,
            "max_edit_chars": 2000, **overrides}


def parent(wrapper="alpha beta gamma", core="first second third"):
    return {"rag.py": wrapper, "rag_core.py": core}


def proposal(files, edits, **overrides):
    return {"parent_source_sha256": source_identity(files), "edits": edits,
            "mechanism": "Synthetic structural test", "intended_target_module": "not_a_host_enum",
            "change_status": "modified",
            **overrides}


def edit(old="beta", new="NEW", file="rag.py"):
    return {"file": file, "old": old, "new": new}


class ProposalProtocolTests(unittest.TestCase):
    def test_none_and_whole_files_preserve_legacy(self):
        self.assertIsNone(validate_proposal_protocol(None))
        whole = {"schema": SCHEMA, "format": "whole_files"}
        self.assertEqual(validate_proposal_protocol(whole), whole)
        for value in (None, whole):
            with self.assertRaisesRegex(ValueError, "explicit exact_edits"):
                materialize_edits(parent(), proposal(parent(), [edit()]), value)

    def test_exact_policy_detached_and_at_limits(self):
        original = protocol(max_edits=64, max_edit_chars=200000)
        checked = validate_proposal_protocol(original)
        checked["max_edits"] = 1
        self.assertEqual(original["max_edits"], 64)
        self.assertEqual(validate_proposal_protocol(protocol(max_edits=1, max_edit_chars=1))["max_edits"], 1)

    def test_protocol_exact_fields_and_types(self):
        bad = [False, "exact_edits", {}, {**protocol(), "schema": "wrong"},
               {**protocol(), "format": "regex"}, {**protocol(), "extra": 1},
               {"schema": SCHEMA, "format": "exact_edits"},
               {"schema": SCHEMA, "format": "whole_files", "max_edits": 1}]
        for field, maximum in (("max_edits", 64), ("max_edit_chars", 200000)):
            bad.extend(protocol(**{field: value}) for value in (True, False, 0, -1, maximum+1, 1.0, "1", None))
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_proposal_protocol(value)

    def test_source_identity_is_complete_canonical_mapping_not_file_bytes(self):
        files = parent()
        self.assertEqual(source_identity(files), digest(files))
        self.assertEqual(source_identity(dict(reversed(list(files.items())))), source_identity(files))
        self.assertEqual(source_identity(MappingProxyType(files)), source_identity(files))
        self.assertNotEqual(source_identity(files), hashlib.sha256(files["rag.py"].encode()).hexdigest())
        self.assertNotEqual(source_identity(files), source_identity({**files, "rag_core.py": files["rag_core.py"]+"\n"}))

    def test_invalid_parent_mapping_rejected(self):
        for files in (None, [], {}, {"rag.py": "x"}, {**parent(), "extra.py": "x"},
                      {**parent(), "rag.py": b"bytes"}, {**parent(), "rag.py": None},
                      {**parent(), "rag.py": "\ud800"}):
            with self.subTest(type=type(files)), self.assertRaises(ValueError):source_identity(files)

    def test_parent_identity_covers_both_files_and_whitespace(self):
        files = parent();p = proposal(files, [edit()])
        for changed in ({**files,"rag.py":files["rag.py"]+" "}, {**files,"rag_core.py":"changed"}):
            with self.assertRaisesRegex(ValueError,"identity mismatch"):materialize_edits(changed,p,protocol())
        for identity in (None, 1, False, "", "0"*64):
            with self.assertRaises(ValueError):materialize_edits(files,{**p,"parent_source_sha256":identity},protocol())

    def test_simple_edit_complete_files_and_receipt_hash(self):
        files=parent();p=proposal(files,[edit()]);result,receipt=materialize_edits(files,p,protocol())
        self.assertEqual(result,parent("alpha NEW gamma"))
        self.assertEqual(receipt["parent_source_sha256"],digest(files))
        self.assertEqual(receipt["child_source_sha256"],digest(result))
        self.assertEqual(receipt["proposal_sha256"],digest(p))
        self.assertEqual(receipt["receipt_sha256"],digest({k:v for k,v in receipt.items() if k!="receipt_sha256"}))
        row=receipt["edits"][0]
        self.assertEqual((row["start"],row["end"],row["edit_index"]),(6,10,0))
        self.assertEqual(row["old_sha256"],hashlib.sha256(b"beta").hexdigest())
        self.assertEqual(row["new_sha256"],hashlib.sha256(b"NEW").hexdigest())
        self.assertNotIn("alpha beta gamma",stable(receipt))
        self.assertNotIn(p["mechanism"],stable(receipt))
        self.assertNotIn('"old":',stable(receipt));self.assertNotIn('"new":',stable(receipt))

    def test_multi_file_replacement_insertion_and_deletion(self):
        files=parent();p=proposal(files,[edit("alpha","prefix alpha"),edit("gamma",""),edit("second","SECOND", "rag_core.py")])
        result,receipt=materialize_edits(files,p,protocol())
        self.assertEqual(result,parent("prefix alpha beta ","first SECOND third"))
        self.assertEqual(len(receipt["edits"]),3)
        self.assertEqual(set(result),set(files))

    def test_anchor_must_appear_in_original_parent(self):
        files=parent("abc")
        p=proposal(files,[edit("a","z"),edit("z","Z")])
        with self.assertRaisesRegex(ValueError,"absent from original"):materialize_edits(files,p,protocol())

    def test_original_positions_not_shifted_by_earlier_replacements(self):
        files=parent("abc def ghi")
        operations=[edit("abc","A very long start"),edit("ghi","end")]
        left,receipt=materialize_edits(files,proposal(files,operations),protocol())
        right,other=materialize_edits(files,proposal(files,list(reversed(operations))),protocol())
        self.assertEqual(left,right);self.assertEqual(left["rag.py"],"A very long start def end")
        self.assertNotEqual(receipt["proposal_sha256"],other["proposal_sha256"])
        self.assertEqual([r["start"] for r in receipt["edits"]],[0,8])

    def test_new_text_does_not_introduce_anchor_ambiguity(self):
        files=parent("abc def")
        p=proposal(files,[edit("abc","def def"),edit("def","FINAL")])
        result,_=materialize_edits(files,p,protocol())
        self.assertEqual(result["rag.py"],"def def FINAL")

    def test_adjacent_ranges_allowed(self):
        files=parent("abcd");p=proposal(files,[edit("ab","X"),edit("cd","Y")])
        result,_=materialize_edits(files,p,protocol());self.assertEqual(result["rag.py"],"XY")

    def test_overlapping_intervals_rejected_even_with_unique_anchors(self):
        for edits in ([edit("abc","X"),edit("bcd","Y")], [edit("abcd","X"),edit("bc","Y")],
                      [edit("abc","X"),edit("abc","Y")]):
            files=parent("abcd")
            with self.assertRaisesRegex(ValueError,"intervals overlap"):materialize_edits(files,proposal(files,edits),protocol())

    def test_repeated_anchor_counts_overlapping_occurrences(self):
        for source,anchor in (("foo foo","foo"),("aaaa","aa"),("ababa","aba")):
            files=parent(source)
            with self.assertRaisesRegex(ValueError,"not unique"):materialize_edits(files,proposal(files,[edit(anchor,"X")]),protocol())

    def test_duplicate_anchor_in_other_file_is_not_ambiguous(self):
        files=parent("same","same")
        result,_=materialize_edits(files,proposal(files,[edit("same","A"),edit("same","B","rag_core.py")]),protocol())
        self.assertEqual(result,parent("A","B"))

    def test_exact_not_fuzzy_regex_or_normalized(self):
        files=parent("Alpha\r\nBeta [x].*")
        for anchor in ("alpha", "Alpha\nBeta", "[a-z]+", "Beta  [x]"):
            with self.assertRaisesRegex(ValueError,"absent"):materialize_edits(files,proposal(files,[edit(anchor,"x")]),protocol())
        result,_=materialize_edits(files,proposal(files,[edit("[x].*","literal")]),protocol())
        self.assertEqual(result["rag.py"],"Alpha\r\nBeta literal")

    def test_unicode_offsets_and_crlf_are_preserved(self):
        files=parent("文甲\r\n正文\r\n尾")
        result,receipt=materialize_edits(files,proposal(files,[edit("正文","新正文")]),protocol())
        self.assertEqual(result["rag.py"],"文甲\r\n新正文\r\n尾")
        self.assertEqual((receipt["edits"][0]["start"],receipt["edits"][0]["end"]),(4,6))

    def test_edit_schema_and_path_escape_rejected(self):
        files=parent()
        invalid=[None,False,"x",{}, {**edit(),"line":1},{"file":"rag.py","old":"beta"}]
        invalid.extend(edit(file=x) for x in ("../rag.py","D:/rag.py","./rag.py","rag.py/../rag_core.py","RAG.py","other.py",True,None))
        invalid.extend(edit(old=x) for x in ("",None,1,False))
        invalid.extend(edit(new=x) for x in (None,1,False,b"x"))
        invalid.extend([edit(new="beta"),edit(new="\ud800")])
        for item in invalid:
            with self.subTest(item=item),self.assertRaises(ValueError):materialize_edits(files,proposal(files,[item]),protocol())

    def test_response_exact_fields_and_metadata_only_structure(self):
        files=parent();base=proposal(files,[edit()])
        for p in (None,False,[],{**base,"writes":{}},{k:v for k,v in base.items() if k!="mechanism"}):
            with self.assertRaises(ValueError):materialize_edits(files,p,protocol())
        for field in ("mechanism","intended_target_module"):
            for value in ("", "  \t",None,True,1,[]):
                with self.assertRaises(ValueError):materialize_edits(files,{**base,field:value},protocol())
        self.assertEqual(materialize_edits(files,base,protocol())[0]["rag.py"],"alpha NEW gamma")

    def test_count_cap_and_total_character_cap_inclusive(self):
        files=parent();one=proposal(files,[edit()])
        self.assertEqual(materialize_edits(files,one,protocol(max_edits=1,max_edit_chars=7))[0]["rag.py"],"alpha NEW gamma")
        with self.assertRaisesRegex(ValueError,"characters"):materialize_edits(files,one,protocol(max_edit_chars=6))
        multiple=proposal(files,[edit(),edit("first","F","rag_core.py")])
        with self.assertRaisesRegex(ValueError,"characters"):materialize_edits(files,multiple,protocol(max_edit_chars=12))
        with self.assertRaisesRegex(ValueError,"edit count"):materialize_edits(files,multiple,protocol(max_edits=1))
        for values in ([],None,{},(edit(),)):
            with self.assertRaises(ValueError):materialize_edits(files,proposal(files,values),protocol())

    def test_character_cap_is_characters_not_utf8_bytes(self):
        files=parent("甲")
        self.assertEqual(materialize_edits(files,proposal(files,[edit("甲","乙")]),protocol(max_edit_chars=2))[0]["rag.py"],"乙")

    def test_no_candidate_execution_or_ast_validation_in_materializer(self):
        files=parent();new="__import__('os').system('NEVER_EXECUTE')"
        with patch("os.system",side_effect=AssertionError("candidate executed")), patch("builtins.eval",side_effect=AssertionError("eval")):
            result,_=materialize_edits(files,proposal(files,[edit(new=new)]),protocol())
        self.assertIn(new,result["rag.py"])
        # Syntax/AST/sandbox are later host gates, never silently repaired here.
        result,_=materialize_edits(files,proposal(files,[edit(new="def broken(")]),protocol())
        self.assertIn("def broken(",result["rag.py"])

    def test_original_inputs_unchanged_on_success_and_failure(self):
        files=parent();p=proposal(files,[edit()]);settings=protocol();snapshot=deepcopy((files,p,settings))
        result,receipt=materialize_edits(files,p,settings)
        self.assertEqual((files,p,settings),snapshot)
        result["rag.py"]="different";receipt["protocol"]["max_edits"]=1
        self.assertEqual((files,p,settings),snapshot)
        invalid=proposal(files,[edit(),edit("missing","x")]);snapshot=deepcopy((files,invalid,settings))
        with self.assertRaises(ValueError):materialize_edits(files,invalid,settings)
        self.assertEqual((files,invalid,settings),snapshot)

    def test_explicit_no_change_returns_detached_unchanged_sources_and_receipt(self):
        files = parent(); p = proposal(files, [], change_status="no_change")
        snapshot = deepcopy((files, p))
        result, receipt = materialize_edits(files, p, protocol())
        self.assertEqual(result, files)
        self.assertIsNot(result, files)
        self.assertEqual(receipt["declared_change_status"], "no_change")
        self.assertIs(receipt["source_changed"], False)
        self.assertEqual(receipt["edits"], [])
        self.assertEqual(receipt["child_source_sha256"], receipt["parent_source_sha256"])
        self.assertEqual(receipt["proposal_sha256"], digest(p))
        result["rag.py"] = "detached"
        self.assertEqual((files, p), snapshot)
        self.assertEqual(p["mechanism"], "Synthetic structural test")

    def test_change_status_is_required_exact_and_consistent_with_edits(self):
        files = parent(); base = proposal(files, [edit()])
        invalid = [{k:v for k,v in base.items() if k != "change_status"},
                   {**base, "change_status": "no_change"},
                   proposal(files, [], change_status="modified")]
        invalid.extend({**base, "change_status": value} for value in
                       (True, None, 1, [], {}, "Modified", "", "no-change"))
        invalid.extend(proposal(files, value, change_status="no_change")
                       for value in (None, {}, (), ""))
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ValueError):
                materialize_edits(files, item, protocol())
        for key, value in (("mechanism", ""), ("intended_target_module", " "),
                           ("parent_source_sha256", "bad")):
            bad = proposal(files, [], change_status="no_change", **{key:value})
            with self.assertRaises(ValueError):materialize_edits(files,bad,protocol())

    def test_host_source_change_flag_is_computed_not_taken_from_declaration(self):
        files = parent("ab")
        # Each edit changes its anchor, yet their combined text equals the parent.
        p = proposal(files, [edit("a", ""), edit("b", "ab")])
        result, receipt = materialize_edits(files, p, protocol())
        self.assertEqual(result, files)
        self.assertEqual(receipt["declared_change_status"], "modified")
        self.assertIs(receipt["source_changed"], False)
        result, receipt = materialize_edits(files, proposal(files, [edit("a", "A")]), protocol())
        self.assertIs(receipt["source_changed"], True)

    def test_deterministic_and_full_file_deletion(self):
        files=parent("all")
        p=proposal(files,[edit("all","")]);first=materialize_edits(files,p,protocol())
        self.assertEqual(first,materialize_edits(files,p,protocol()))
        self.assertEqual(first[0]["rag.py"],"")


if __name__ == "__main__":
    unittest.main()
