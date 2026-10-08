"""Deferred official references: freeze all generations before any answer access."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from code_rsi.budget import digest, save
from code_rsi.v3 import calibration as cal
import test_v3_calibration as fixtures
from test_v3_three_arm_calibration import three_arm_plan


class DeferredReferenceTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.CalibrationTests(methodName="runTest")
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()

    def plan(self):
        plan=three_arm_plan(self.fixture.plan())
        plan.pop("references_file")
        tasks=cal._task_snapshot(plan)
        plan["reference_acquisition"]={
            "schema":cal.REFERENCE_ACQUISITION_SCHEMA,
            "source":{"repo":"Tevatron/browsecomp-plus","revision":"b"*40,
                      "files":[{"path":f"data/test-{i:05d}-of-00006.parquet","size":1000,"sha256":"a"*64}
                               for i in range(6)]},
            "decoder_version":cal.DECODER_VERSION,
            "question_hashes":{t["question_id"]:hashlib.sha256(t["question"].encode("utf-8")).hexdigest() for t in tasks}}
        return plan

    def acquisition(self,plan):
        tasks=cal._task_snapshot(plan)
        rows=[{"query_id":t["question_id"],"question":t["question"],"reference_answer":"Synthetic Port"} for t in tasks]
        source=copy.deepcopy(plan["reference_acquisition"]["source"])
        receipt={"schema":"rag-rsi-bcp-column-acquisition-1","source":source,"source_sha256":digest(source),
                 "decoder_version":cal.DECODER_VERSION,"columns":["query_id","query","answer"],
                 "question_ids":sorted(plan["question_ids"]),"decoded_question_count":len(tasks),
                 "decoded_answer_count":len(tasks),"observed_unique_source_ids":len(tasks),
                 "files":[{"path":item["path"],"ranges":[],"range_bytes":0} for item in source["files"]],
                 "full_file_sha256_verified":False,"semantic_column_projection_only":True,
                 "scope":"Synthetic fixture; no real source bytes fetched."}
        return rows,receipt

    def record_path(self,plan):
        return Path(plan["output_dir"])/"references/acquired.json"

    def test_preflight_and_generation_do_not_acquire_or_open_reference_values(self):
        plan=self.plan()
        # A materialized-reference file is deliberately absent, and not required.
        with patch.object(cal,"acquire_references",side_effect=AssertionError("premature answer acquisition")) as acquire:
            result=cal.preflight(plan)
            frozen,executor=self.fixture.generate(plan)
            again,_=self.fixture.generate(plan,executor)
        self.assertEqual(acquire.call_count,0)
        self.assertEqual(frozen,again)
        self.assertEqual(len(executor.calls),6)
        self.assertFalse(self.record_path(plan).exists())
        self.assertFalse(result["references_parsed"])
        self.assertEqual(result["reference_source_verified_before_generation"],"metadata_only")

    def test_deferred_contract_cannot_mix_bindings_or_change_public_questions(self):
        original=self.plan()
        mutations=[lambda p:p.update(references_file={"path":"unused","sha256":"a"*64}),
                   lambda p:p.pop("reference_acquisition"),
                   lambda p:p["reference_acquisition"].update(extra=True),
                   lambda p:p["reference_acquisition"].update(schema="unknown"),
                   lambda p:p["reference_acquisition"].update(decoder_version="unknown"),
                   lambda p:p["reference_acquisition"].update(question_hashes={}),
                   lambda p:p["reference_acquisition"]["source"].update(repo="another/dataset")]
        for index,mutate in enumerate(mutations):
            plan=copy.deepcopy(original);mutate(plan)
            with self.subTest(index=index),patch.object(cal,"acquire_references") as acquire,self.assertRaises(ValueError):
                cal.preflight(plan)
            acquire.assert_not_called()

    def test_deferred_panel_outside_acquisition_range_is_rejected_before_execution(self):
        plan=self.plan();tasks=cal._task_snapshot(plan)
        for bad_tasks in (tasks*65,[{**tasks[0],"question_id":"x"*129}]):
            with self.subTest(size=len(bad_tasks)),patch.object(cal,"acquire_references") as acquire,self.assertRaisesRegex(ValueError,"bounded question IDs"):
                cal._reference_contract(plan,bad_tasks)
            acquire.assert_not_called()

    def test_old_schema_cannot_enable_deferred_acquisition(self):
        plan=self.fixture.plan();contract=self.plan()["reference_acquisition"]
        plan.pop("references_file");plan["reference_acquisition"]=contract
        with self.assertRaisesRegex(ValueError,"three-arm"):
            cal.preflight(plan)

    def test_no_complete_generation_means_no_acquisition_or_cached_answer_read(self):
        plan=self.plan()
        # Even a pre-existing artifact must remain unread until the freeze validates.
        save(self.record_path(plan),{"not":"a valid acquired record"})
        original_read=cal.read; reads=[]
        def guarded(path):
            if Path(path)==self.record_path(plan):
                reads.append(str(path));raise AssertionError("cached answers read before generation completion")
            return original_read(path)
        with patch.object(cal,"read",side_effect=guarded),patch.object(cal,"acquire_references") as acquire, self.assertRaises(ValueError):
            cal.grade(plan)
        self.assertEqual(reads,[]);acquire.assert_not_called()

    def test_missing_cell_or_changed_plan_prevents_acquisition(self):
        plan=self.plan();frozen,_=self.fixture.generate(plan)
        save(Path(plan["output_dir"])/"generation_freeze.json",{**frozen,"cells":frozen["cells"][:-1]})
        with patch.object(cal,"acquire_references") as acquire,self.assertRaises(ValueError):
            cal.grade(plan)
        acquire.assert_not_called()
        save(Path(plan["output_dir"])/"generation_freeze.json",frozen)
        plan["reference_acquisition"]["source"]["revision"]="c"*40
        with patch.object(cal,"acquire_references") as acquire,self.assertRaises(ValueError):
            cal.grade(plan)
        acquire.assert_not_called()

    def test_full_generation_unlocks_references_once_and_report_has_no_answers(self):
        plan=self.plan();frozen,executor=self.fixture.generate(plan)
        rows,receipt=self.acquisition(plan)
        def acquire(source,question_ids):
            self.assertEqual(source,plan["reference_acquisition"]["source"])
            self.assertEqual(question_ids,sorted(plan["question_ids"]))
            self.assertEqual(cal.read(Path(plan["output_dir"])/"generation_freeze.json"),frozen)
            return list(reversed(rows)),receipt
        with patch.object(cal,"acquire_references",side_effect=acquire) as fetched:
            first=cal.grade(plan);second=cal.grade(plan)
            self.fixture.generate(plan,executor)
        self.assertEqual(fetched.call_count,1)
        self.assertEqual(first,second)
        self.assertEqual(len(executor.calls),6)
        record=cal.read(self.record_path(plan))
        self.assertEqual(record["generation_freeze_hash"],digest(frozen))
        self.assertEqual(record["contract_hash"],digest(plan["reference_acquisition"]))
        self.assertEqual(record["rows_sha256"],digest(record["rows"]))
        self.assertEqual(record["rows"],rows)
        provenance=first["reference_provenance"]
        self.assertTrue(provenance["acquired_only_after_complete_generation"])
        self.assertFalse(provenance["references_returned_to_generation"])
        self.assertNotIn("Synthetic Port",json.dumps(first))
        self.assertNotIn("full_file_sha256_verified",provenance)
        self.assertIn("not a full-file",provenance["source_verification"])

    def test_incomplete_duplicate_foreign_empty_and_mismatched_reference_rows_fail(self):
        original=self.plan();self.fixture.generate(original)
        rows,receipt=self.acquisition(original)
        variants=[rows[:-1],rows+[rows[0]],[rows[0],rows[0]],
                  [{**rows[0],"query_id":"foreign"},rows[1]],
                  [{**rows[0],"question":"another question"},rows[1]],
                  [{**rows[0],"reference_answer":""},rows[1]],
                  [{**rows[0],"followup":"not allowed"},rows[1]]]
        for index,changed in enumerate(variants):
            with self.subTest(index=index),patch.object(cal,"acquire_references",return_value=(changed,receipt)),self.assertRaises(ValueError):
                cal.grade(original)
            self.assertFalse(self.record_path(original).exists())

    @staticmethod
    def mismatched_receipts(original):
        variants=[]
        for key,value in (("schema","unknown"),("source_sha256","bad"),("decoder_version","unknown"),
                          ("columns",["query_id","query"]),("question_ids",list(reversed(original["question_ids"]))),
                          ("decoded_question_count",True),("decoded_answer_count",0),
                          ("full_file_sha256_verified",True),("semantic_column_projection_only",False)):
            receipt=copy.deepcopy(original);receipt[key]=value;variants.append(receipt)
        # This wrong source is internally consistent; its own SHA is not a contract check.
        receipt=copy.deepcopy(original);receipt["source"]["revision"]="c"*40
        receipt["source_sha256"]=digest(receipt["source"]);variants.append(receipt)
        return variants

    def test_first_acquisition_receipt_must_match_frozen_source_and_scope(self):
        plan=self.plan();self.fixture.generate(plan);rows,receipt=self.acquisition(plan)
        for index,changed in enumerate(self.mismatched_receipts(receipt)):
            with self.subTest(index=index),patch.object(cal,"acquire_references",return_value=(rows,changed)),self.assertRaises(ValueError):
                cal.grade(plan)
            self.assertFalse(self.record_path(plan).exists())

    def test_rehashed_cached_receipt_cannot_change_frozen_source_or_scope(self):
        plan=self.plan();self.fixture.generate(plan)
        with patch.object(cal,"acquire_references",return_value=self.acquisition(plan)):
            cal.grade(plan)
        original=cal.read(self.record_path(plan))
        for index,receipt in enumerate(self.mismatched_receipts(original["source_receipt"])):
            record=copy.deepcopy(original);record["source_receipt"]=receipt
            record["source_receipt_sha256"]=digest(receipt)
            save(self.record_path(plan),record)
            with self.subTest(index=index),patch.object(cal,"acquire_references") as fetched,self.assertRaises(ValueError):
                cal.grade(plan)
            fetched.assert_not_called()
            self.assertEqual(cal.read(self.record_path(plan)),record)

    def test_acquisition_failure_preserves_generation_for_local_retry(self):
        plan=self.plan();frozen,executor=self.fixture.generate(plan)
        with patch.object(cal,"acquire_references",side_effect=OSError("synthetic retrieval failure")),self.assertRaises(OSError):
            cal.grade(plan)
        self.assertFalse(self.record_path(plan).exists())
        self.assertEqual(cal.read(Path(plan["output_dir"])/"generation_freeze.json"),frozen)
        with patch.object(cal,"acquire_references",return_value=self.acquisition(plan)) as fetched:
            cal.grade(plan)
        self.assertEqual(fetched.call_count,1)
        self.assertEqual(len(executor.calls),6)

    def test_crash_after_atomic_acquisition_before_grading_reuses_receipt(self):
        plan=self.plan();self.fixture.generate(plan)
        original_freeze=cal.freeze
        def interrupted(path,value):
            original_freeze(path,value)
            if Path(path)==self.record_path(plan):
                raise OSError("synthetic crash after committed acquisition")
        with patch.object(cal,"acquire_references",return_value=self.acquisition(plan)) as fetched:
            with patch.object(cal,"freeze",side_effect=interrupted),self.assertRaises(OSError):
                cal.grade(plan)
            self.assertTrue(self.record_path(plan).exists())
            cal.grade(plan)
        self.assertEqual(fetched.call_count,1)

    def test_interrupted_temporary_write_is_recovered_without_new_generation(self):
        plan=self.plan();self.fixture.generate(plan)
        path=self.record_path(plan);path.parent.mkdir(parents=True)
        path.with_suffix(".json.tmp").write_text("partial synthetic bytes",encoding="utf-8")
        with patch.object(cal,"acquire_references",return_value=self.acquisition(plan)) as fetched:
            cal.grade(plan)
        self.assertEqual(fetched.call_count,1)
        self.assertTrue(path.exists());self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_tampered_frozen_acquisition_is_not_overwritten_or_refetched(self):
        plan=self.plan();self.fixture.generate(plan)
        with patch.object(cal,"acquire_references",return_value=self.acquisition(plan)):
            cal.grade(plan)
        original=cal.read(self.record_path(plan))
        mutations=[lambda r:r.update(plan_hash="bad"),lambda r:r.update(contract_hash="bad"),
                   lambda r:r.update(generation_freeze_hash="bad"),
                   lambda r:r["rows"][0].update(reference_answer="tampered"),
                   lambda r:r["source_receipt"].update(extra=True),lambda r:r.update(extra=True)]
        for index,mutate in enumerate(mutations):
            changed=copy.deepcopy(original);mutate(changed);save(self.record_path(plan),changed)
            with self.subTest(index=index),patch.object(cal,"acquire_references") as fetched,self.assertRaises(ValueError):
                cal.grade(plan)
            fetched.assert_not_called()
            self.assertEqual(cal.read(self.record_path(plan)),changed)

    def test_existing_local_reference_contract_remains_compatible(self):
        plan=three_arm_plan(self.fixture.plan());self.fixture.generate(plan)
        with patch.object(cal,"acquire_references",side_effect=AssertionError("legacy contract must not fetch")):
            report=cal.grade(plan)
        self.assertEqual(report["reference_provenance"]["binding"],"frozen_local_file")
        self.assertTrue(report["reference_provenance"]["parsed_only_after_complete_generation"])
        self.assertFalse(self.record_path(plan).exists())


if __name__=="__main__":unittest.main()
