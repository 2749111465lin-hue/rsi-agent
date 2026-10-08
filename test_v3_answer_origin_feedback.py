"""Host provenance reaches diagnostics without a network call or benchmark data."""
from copy import deepcopy
import unittest
from code_rsi.v3.datasets import adapt_multihop
from code_rsi.v3.infrastructure import LocalCorpus
from code_rsi.v3.execution import HostBroker, EXECUTION_SCHEMA
from code_rsi.v3.diagnostics import diagnose_execution, execution_flow, compact_feedback
from test_v3_diagnostics import measurement, tasks


def receipt(answers, returned):
    task,_=adapt_multihop({'id':'origin-flow','query':'Which synthetic port?','answer':'Synthetic Port'})
    class Model:
        def __init__(self): self.answers=iter(answers)
        def complete(self,stage,payload):
            return {'answer':next(self.answers),'citation_ids':[],'evidence_sufficient':False}
    broker=HostBroker(task,LocalCorpus([]),Model())
    for _ in answers: broker('complete',{'stage':'answer','payload':{'evidence':[]}})
    origin=broker.answer_origin_receipt(returned)
    return {'schema':EXECUTION_SCHEMA,'question_id':task['question_id'],'node_id':'child',
            'answer':returned,'answer_usable':bool(returned.strip()),'execution_ok':True,
            'answer_origin_valid':origin['valid'],'answer_origin_status':origin['status'],
            'host_answer_origin_validation':origin,'trace':broker.events,
            'host_evidence_trace':{'read_presentations':[],'final_observations':broker.final_observations},
            'citation_source_valid':False,'failure_classes':[],'model_errors':[],
            'resource_usage':broker.counts,'candidate_reported':{},'role':'D_fit','repeat':0,'score':1.}


class AnswerOriginFeedbackTests(unittest.TestCase):
    def test_overwritten_final_is_diagnosed_using_actual_last_model_answer(self):
        row=receipt(['First model answer','Last model answer'],'First model answer')
        diagnosis=diagnose_execution(row)
        self.assertIn('invalid_answer_origin',diagnosis['host_observed'])
        flow=execution_flow(row)
        self.assertFalse(flow['final']['answer_origin_valid'])
        self.assertEqual(flow['final']['answer_excerpt'],'Last model answer')
        self.assertEqual(flow['final']['answer_origin_status'],'candidate_answer_mismatch')

    def test_normal_abstention_remains_valid_origin_with_insufficient_evidence(self):
        row=receipt(['Insufficient information'],'  Insufficient information\n')
        result=diagnose_execution(row)
        self.assertNotIn('invalid_answer_origin',result['host_observed'])
        self.assertIn('evidence_insufficient',result['model_reported'])
        self.assertTrue(execution_flow(row)['final']['answer_origin_valid'])

    def test_no_model_call_does_not_invent_an_observation(self):
        row=receipt([],'Synthetic Port')
        self.assertIn('invalid_answer_origin',diagnose_execution(row)['host_observed'])
        self.assertFalse(execution_flow(row)['final']['observation_found'])

    def test_forged_host_flag_is_not_accepted_as_feedback_truth(self):
        row=receipt(['Other Port'],'Synthetic Port')
        row['answer_origin_valid']=True
        for diagnostic in (diagnose_execution,execution_flow):
            with self.assertRaises(ValueError): diagnostic(row)

    def test_developer_sees_ineligible_origin_and_honest_fit_information_disclosure(self):
        row=receipt(['Other Port'],'Synthetic Port')
        result=measurement([row],valid_program=False)
        feedback=compact_feedback(result,tasks([row]))
        self.assertEqual(feedback['schema'],'rag-rsi-v3-feedback-3')
        self.assertTrue(feedback['raw_reference_objects_not_sent'])
        self.assertTrue(feedback['fit_feedback_can_reveal_accepted_answers'])
        self.assertNotIn('reference_not_sent',feedback)
        self.assertIn('invalid_answer_origin',feedback['cases'][0]['diagnostics']['host_observed'])
        self.assertIsNone(feedback['cases'][0]['host_score'])
        self.assertEqual(feedback['cases'][0]['raw_host_score'],1)
        self.assertFalse(feedback['program_eligible'])
        self.assertFalse(feedback['cases'][0]['execution_flow']['final']['answer_origin_valid'])

    def test_positive_and_negative_raw_differences_remain_diagnostic_when_either_program_invalid(self):
        for child_bad in (True,False):
            child=receipt(['Other Port'] if child_bad else ['Synthetic Port'],'Synthetic Port')
            parent=receipt(['Other Port'] if not child_bad else ['Synthetic Port'],'Synthetic Port')
            child['score'],parent['score']=(1.,0.) if child_bad else (0.,1.)
            parent['node_id']='parent'
            p=measurement([parent],node='parent',valid_program=parent['answer_origin_valid'])
            c=measurement([child],valid_program=child['answer_origin_valid'],parent_measurement=p)
            feedback=compact_feedback(c,tasks([child]))
            expected=1 if child_bad else -1
            self.assertIsNone(feedback['paired_summary'])
            self.assertEqual(feedback['raw_paired_summary']['mean_signed_gain'],expected)
            self.assertIsNone(feedback['cases'][0]['signed_delta'])
            self.assertEqual(feedback['cases'][0]['raw_signed_delta'],expected)
            self.assertFalse(feedback['paired_comparison_eligible'])

    def test_missing_program_eligibility_does_not_become_assumed_success(self):
        row=receipt(['Synthetic Port'],'Synthetic Port')
        m=measurement([row]);m.pop('valid_program')
        feedback=compact_feedback(m,tasks([row]))
        self.assertIsNone(feedback['program_eligible'])
        self.assertIsNone(feedback['score'])
        self.assertEqual(feedback['raw_score'],1)

    def test_legacy_validity_never_becomes_current_eligibility(self):
        row=receipt(['Synthetic Port'],'Synthetic Port')
        row['schema']='rag-rsi-v3-execution-2'
        for key in ('answer_origin_valid','answer_origin_status','host_answer_origin_validation'):
            row.pop(key)
        parent=deepcopy(row);parent['node_id']='parent';parent['score']=0.
        p=measurement([parent],node='parent',valid_program=True)
        c=measurement([row],valid_program=True,parent_measurement=p)
        feedback=compact_feedback(c,tasks([row]))
        self.assertIsNone(feedback['program_eligible'])
        self.assertIsNone(feedback['score'])
        self.assertIsNone(feedback['paired_summary'])
        self.assertEqual(feedback['raw_paired_summary']['mean_signed_gain'],1)
        self.assertEqual(feedback['cases'][0]['raw_signed_delta'],1)

    def test_actual_blank_answer_overrides_untrusted_usable_flag(self):
        row=receipt(['  '],'')
        self.assertTrue(row['answer_origin_valid'])
        row['answer_usable']=True
        feedback=compact_feedback(measurement([row],valid_program=True),tasks([row]))
        self.assertFalse(feedback['program_eligible'])
        self.assertIsNone(feedback['score'])
        self.assertEqual(feedback['raw_score'],1)

if __name__=='__main__': unittest.main()
