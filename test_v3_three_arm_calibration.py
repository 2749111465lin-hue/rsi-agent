"""Three-workflow pre-registration and report integration, synthetic and zero API."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from code_rsi.budget import digest, save
from code_rsi.v3 import calibration as cal
from code_rsi.v3.infrastructure import UnknownProviderOutcome
from code_rsi.v3.execution import HostBroker
import test_v3_calibration as fixtures


def three_arm_plan(base):
    plan=copy.deepcopy(base)
    common={k:v for k,v in plan["arms"][0]["config"].items() if k!="mode"}
    plan.update(schema=cal.THREE_ARM_SCHEMA,purpose="workflow_decomposition_calibration",
        question_use="synthetic",request_coupling=cal.REQUEST_COUPLING,
        arms=[{"name":name,"config":{**common,"mode":mode}} for name,mode in
              (("raw","single_pass"),("planned","planned_single"),("loop","iterative"))],
        max_calls=len(plan["question_ids"])*plan["repeats"]*10,hard_cny=50,
        analysis={"schema":"rag-rsi-paired-analysis-1","primary_metric":"answer_f1",
          "comparisons":[{"name":"planning","baseline":"raw","candidate":"planned"},
                         {"name":"iteration","baseline":"planned","candidate":"loop"}],
          "question_groups":{qid:qid for qid in plan["question_ids"]},
          "confidence_level":.95,"bootstrap_samples":1000,"bootstrap_seed":321,
          "target_effect":.1,"power":.8})
    return plan


class ThreeArmCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.CalibrationTests(methodName="runTest")
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()

    def plan(self,**kwargs):
        return three_arm_plan(self.fixture.plan(**kwargs))

    def test_three_arm_preflight_derives_160_not_old_112(self):
        p=self.plan(questions=8,repeats=2,opaque_references=True)
        result=cal.preflight(p)
        self.assertEqual(result["max_calls"],160)
        self.assertEqual(result["answer_outcomes"],48)
        self.assertEqual(result["arm_count"],3)
        self.assertFalse(result["credentials_read"])
        self.assertFalse(result["references_parsed"])
        self.assertTrue(result["prefix_coupling_includes_identical_final_requests"])
        self.assertLess(result["conservative_cny_upper_bound"],50)
        self.assertGreater(result["conservative_cny_upper_bound"],30)

    def test_new_schema_load_and_old_schema_stays_two_arm(self):
        p=self.plan();path=self.fixture.root/"three.json";save(path,p)
        self.assertEqual(cal.load_plan(path),p)
        p["schema"]=cal.SCHEMA;p["purpose"]="used_development_calibration"
        with self.assertRaisesRegex(ValueError,"exactly 2"):
            cal.preflight(p)

    def test_nonmode_changes_and_fake_decomposition_rejected(self):
        original=self.plan()
        mutations=[lambda p:p["arms"][1]["config"].update(search_limit=6),
          lambda p:p["arms"][1]["config"].update(mode="single_pass"),
          lambda p:p.update(max_calls=14),lambda p:p.update(hard_cny=1),
          lambda p:p.update(request_coupling="independent"),
          lambda p:p.update(limits={"max_models":5,"max_searches":0,"max_reads":0}),
          lambda p:p.update(limits={"max_models":5,"max_searches":1,"max_reads":8}),
          lambda p:p.update(limits={"max_models":5,"max_searches":6,"max_reads":True}),
          lambda p:p.update(schedule_seed=True),lambda p:p.update(question_use="confirmed_unused"),
          lambda p:p["analysis"]["comparisons"][1].update(baseline="raw"),
          lambda p:p["analysis"]["question_groups"].pop(p["question_ids"][0]),
          lambda p:[a["config"].update(max_rounds=1) for a in p["arms"]],
          lambda p:[a["config"].update(search_limit=31) for a in p["arms"]],
          lambda p:[a["config"].update(max_model_calls=2) for a in p["arms"]]]
        for mutate in mutations:
            p=copy.deepcopy(original);mutate(p)
            with self.subTest(mutation=mutations.index(mutate)),self.assertRaises(ValueError):
                cal.preflight(p)

    def test_three_arm_generation_and_grouped_statistics_keep_all_outcomes(self):
        p=self.plan(questions=3,repeats=2)
        q=p["question_ids"];p["analysis"]["question_groups"]={q[0]:"shared",q[1]:"shared",q[2]:"other"}
        frozen,executor=self.fixture.generate(p)
        self.assertEqual(len(frozen["cells"]),18)
        report=cal.grade(p)
        self.assertEqual(report["schema"],"rag-rsi-v3-calibration-report-3")
        self.assertEqual(report["independent_units"],2)
        self.assertEqual(report["question_count"],3)
        self.assertEqual(report["analysis"]["sample_size"],3)
        self.assertEqual(report["analysis"]["cluster_count"],2)
        self.assertTrue(report["analysis"]["quality_comparison_valid"])
        self.assertNotIn("paired_question_f1_deltas",report)
        self.assertEqual(report["cost_accounting"]["physical_calls"],0)
        self.assertEqual(set(report["cost_accounting"]["logical_calls_by_arm"]),{"raw","planned","loop"})
        # This fake executor deliberately has no plan/read prefix. Do not call it verified.
        self.assertEqual(report["shared_prefix_diagnostic"]["verified_pairs"],0)
        self.assertEqual(report["shared_prefix_diagnostic"]["unknown_pairs"],6)
        self.assertIsNone(report["mechanism_comparison_valid"])
        self.assertIsNone(report["shared_prefix_diagnostic"]["pairs"][0]["plan_payload_equal"])
        resumed,_=self.fixture.generate(p,executor)
        self.assertEqual(resumed,frozen);self.assertEqual(len(executor.calls),18)
        self.assertEqual(cal.grade(p),report)

    def test_observed_prefix_mismatch_blocks_decomposition_without_dropping_rows(self):
        p=self.plan();base=fixtures.FakeExecutor()
        def executor(archive,node_id,task,backend,model,directory,**kwargs):
            receipt=base(archive,node_id,task,backend,model,directory,**kwargs)
            source=archive.load_program(receipt["program_id"])["files"]["rag.py"]
            # Deliberately violate the fixed workflow prefix using actual host events.
            label="different planned prefix" if 'planned_single' in source else "loop prefix"
            class Scripted:
                def complete(self,stage,payload):
                    if stage=="answer":return {"answer":"Synthetic Port","citation_ids":[],"evidence_sufficient":False}
                    return {"stage":stage}
            broker=HostBroker(task,backend,Scripted())
            broker("complete",{"stage":"plan","payload":{"test_prefix":label}})
            broker("complete",{"stage":"read","payload":{"sources":[]}})
            broker("complete",{"stage":"answer","payload":{"evidence":[]}})
            origin=broker.answer_origin_receipt(receipt["answer"])
            receipt.update(trace=broker.events,resource_usage=broker.counts,
                host_evidence_trace={"read_presentations":broker.read_presentations,"final_observations":broker.final_observations},
                answer_origin_valid=origin["valid"],answer_origin_status=origin["status"],host_answer_origin_validation=origin)
            return receipt
        self.fixture.generate(p,executor)
        report=cal.grade(p)
        self.assertTrue(report["execution_comparison_valid"])
        self.assertFalse(report["quality_comparison_valid"])
        self.assertFalse(report["mechanism_comparison_valid"])
        self.assertEqual(report["shared_prefix_diagnostic"]["mismatched_pairs"],2)
        self.assertIsNone(report["analysis"]["qualified"])
        self.assertEqual(report["analysis"]["cell_count"],6)
        self.assertTrue(report["analysis"]["raw_diagnostics"]["comparisons"])

    def test_failed_protocol_preserves_raw_but_has_no_qualified_analysis(self):
        p=self.plan()
        self.fixture.generate(p,fixtures.FakeExecutor(model_answer="Wrong",returned_answer="Synthetic Port"))
        report=cal.grade(p)
        self.assertFalse(report["quality_comparison_valid"])
        self.assertFalse(report["analysis"]["quality_comparison_valid"])
        self.assertIsNone(report["analysis"]["qualified"])
        self.assertTrue(report["analysis"]["raw_diagnostics"])
        self.assertTrue(report["all_outcomes_retained"])

    def test_changed_analysis_does_not_reach_private_references(self):
        p=self.plan();self.fixture.generate(p)
        p["analysis"]["bootstrap_seed"]+=1
        self.fixture.reject_grade_before_references(p)

    def test_missing_third_arm_cell_blocks_private_references(self):
        p=self.plan();frozen,_=self.fixture.generate(p)
        frozen["cells"]=[c for c in frozen["cells"] if c["identity"]["arm"]!="planned"]
        save(Path(p["output_dir"])/"generation_freeze.json",frozen)
        self.fixture.reject_grade_before_references(p)

    def test_unknown_request_blocks_changed_payload_on_resume_before_dispatch(self):
        p=self.plan(opaque_references=True);sent=[]
        def transport(body):
            sent.append(body);raise TimeoutError("synthetic unknown physical outcome")
        def action(task,model):model.complete("answer",{"question":task["question"]})
        with self.assertRaises(UnknownProviderOutcome):
            self.fixture.generate(p,fixtures.FakeExecutor(action),transport)
        def changed(task,model):model.complete("plan",{"question":"different payload"})
        with self.assertRaises(UnknownProviderOutcome):
            self.fixture.generate(p,fixtures.FakeExecutor(changed),transport)
        self.assertEqual(len(sent),1)
        self.assertFalse((Path(p["output_dir"])/"generation_freeze.json").exists())

    def test_new_grade_checks_analysis_before_opening_reference_values(self):
        p=self.plan();self.fixture.generate(p)
        p["analysis"]["question_groups"]={}
        self.fixture.reject_grade_before_references(p)


if __name__=="__main__":unittest.main()
