"""Zero-API Measurement wiring tests using synthetic host execution receipts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from code_rsi.archive import ProgramArchive
from code_rsi.budget import digest, save
from code_rsi.v3 import execution
from code_rsi.v3.datasets import adapt_multihop, adapt_musique, adapt_browsecomp
from code_rsi.v3.infrastructure import LocalCorpus, UnknownProviderOutcome


class NoAPIModel:
    identity='synthetic-metrics-no-api-v1'

    def complete(self,*args,**kwargs):
        raise AssertionError('These host-metric tests never call a model')


class ReceiptExecutor:
    """Simulate only the completed host receipt, never execute candidate code."""
    def __init__(self,answer='Blue Harbor',citations=(),usable=None,error=None):
        self.answer=answer
        self.citations=list(citations)
        self.usable=bool(answer.strip()) if usable is None else usable
        self.error=error
        self.tasks=[]

    def __call__(self,archive,node_id,task,backend,model,directory,**kwargs):
        self.tasks.append(deepcopy(task))
        if self.error is not None:
            raise self.error
        node=archive.load_node(node_id)
        return {'schema':'rag-rsi-v3-execution-2','node_id':node_id,'program_id':node['program_id'],
                'question_id':task['question_id'],'answer':self.answer,'answer_usable':self.usable,
                'execution_ok':True,'citations':deepcopy(self.citations),
                'citation_source_valid':False,'citation_status':'synthetic_not_source_validated',
                'failure_classes':[] if self.usable else ['answer_empty'],
                'resource_usage':{'model_calls':0,'search_calls':0,'read_calls':0},
                'model_errors':[],'candidate_reported':None}


class MeasurementMetricsTests(unittest.TestCase):
    def temporary(self):
        root=Path(__file__).parent/'runs'
        root.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(prefix='measurement_metrics_',dir=root)

    def archive(self,directory):
        archive=ProgramArchive(Path(directory)/'archive')
        node=archive.record(execution.root_files(),{},session_id='metric-test',attempt=0)
        return archive,node

    def multihop(self):
        return adapt_multihop({'id':'metric-multi','query':'Synthetic question?',
                              'answer':'Blue Harbor','documents':[{'docid':'d','text':'Synthetic corpus'}]})

    def bcp(self):
        return adapt_browsecomp({'query_id':'metric-bcp','question':'Synthetic BCP question?',
                                'answer':'Blue Harbor'},'synthetic-fixed-corpus')

    def musique(self,answerable=True):
        return adapt_musique({'id':'metric-musique','question':'Synthetic paired question?',
            'answer':'Blue Harbor','answer_aliases':[],'answerable':answerable,
            'paragraphs':[{'idx':i,'title':'Synthetic '+str(i),
                           'paragraph_text':str(answerable)+' synthetic paragraph '+str(i),
                           'is_supporting':i in (1,7)} for i in range(20)]})

    def measure(self,directory,archive,**kwargs):
        return execution.Measurement(archive,Path(directory)/'measurements',lambda bank:NoAPIModel(),
            backend_factory=lambda task:LocalCorpus([],scope='synthetic-fixed-corpus'),**kwargs)

    def run_one(self,measure,node,task,reference,executor):
        with patch.object(execution,'execute',side_effect=executor):
            return measure.run(node,[task],{task['question_id']:reference},role='D_fit',bank='common')

    def test_primary_metric_stays_scalar_and_support_is_separate(self):
        for metric,expected in (('em',0.),('f1',2/3)):
            with self.subTest(metric=metric),self.temporary() as tmp:
                archive,node=self.archive(tmp); task,ref=self.musique()
                measure=self.measure(tmp,archive,metric=metric)
                result=self.run_one(measure,node,task,ref,ReceiptExecutor(answer='Blue'))
                row=result['rows'][0]
                self.assertAlmostEqual(result['score'],expected)
                self.assertAlmostEqual(row['task_metrics']['answer_f1'],2/3)
                self.assertEqual(row['task_metrics']['support_f1'],0.)
                self.assertEqual(row['task_metrics']['support_em'],0.)
                self.assertIsNone(row['task_metrics']['answerability'])

    def test_correct_answer_without_evidence_keeps_answer_score(self):
        with self.temporary() as tmp:
            archive,node=self.archive(tmp); task,ref=self.musique()
            result=self.run_one(self.measure(tmp,archive,metric='f1'),node,task,ref,ReceiptExecutor())
            self.assertEqual(result['score'],1.)
            self.assertEqual(result['rows'][0]['task_metrics']['support_f1'],0.)
            self.assertFalse(result['rows'][0]['citation_source_valid'])

    def test_support_success_does_not_increase_wrong_answer_score(self):
        with self.temporary() as tmp:
            archive,node=self.archive(tmp); task,ref=self.musique()
            citations=[{'docid':docid} for docid in ref['supporting_docids']]
            result=self.run_one(self.measure(tmp,archive,metric='f1'),node,task,ref,
                                ReceiptExecutor(answer='Wrong place',citations=citations))
            self.assertEqual(result['score'],0.)
            self.assertEqual(result['rows'][0]['task_metrics']['support_f1'],1.)

    def test_unknown_citation_mapping_is_unavailable_not_zero_or_success(self):
        with self.temporary() as tmp:
            archive,node=self.archive(tmp); task,ref=self.musique()
            result=self.run_one(self.measure(tmp,archive),node,task,ref,
                                ReceiptExecutor(citations=[{'citation_id':'unmapped'}]))
            metrics=result['rows'][0]['task_metrics']
            self.assertEqual(result['score'],1.)
            self.assertIsNone(metrics['support_f1'])
            self.assertEqual(metrics['metric_status']['support_f1'],'unavailable_citation_mapping')

    def test_bcp_proxy_is_explicit_and_never_official(self):
        with self.temporary() as tmp:
            archive,node=self.archive(tmp); task,ref=self.bcp(); executor=ReceiptExecutor()
            measure=self.measure(tmp,archive,allow_proxy_metrics=True)
            result=self.run_one(measure,node,task,ref,executor)
            metrics=result['rows'][0]['task_metrics']
            self.assertEqual(result['score'],1.)
            self.assertEqual(metrics['answer_f1'],1.)
            self.assertTrue(metrics['proxy_metrics'])
            self.assertFalse(metrics['official_judge'])
            self.assertEqual(metrics['metric_status']['answer_f1'],'proxy_rule_only')
            self.assertIsNone(metrics['support_f1'])
            self.assertNotIn('answers',executor.tasks[0])

    def test_bcp_without_proxy_permission_fails_before_model_factory(self):
        with self.temporary() as tmp:
            archive,node=self.archive(tmp); task,ref=self.bcp()
            def forbidden_factory(bank):
                self.fail('BCP guard must precede model construction')
            measure=execution.Measurement(archive,Path(tmp)/'measurements',forbidden_factory)
            with self.assertRaisesRegex(ValueError,'explicit allow_proxy_metrics'):
                measure.run(node,[task],{task['question_id']:ref},role='D_fit',bank='common')
            self.assertEqual(list((Path(tmp)/'measurements').rglob('measured.json')),[])

    def test_external_bcp_score_keeps_contract_without_fabricating_judge_metrics(self):
        with self.temporary() as tmp:
            archive,node=self.archive(tmp); task,ref=self.bcp(); calls=[]
            def scorer(answer,reference):
                calls.append((answer,deepcopy(reference)))
                return .375
            result=self.run_one(self.measure(tmp,archive,metric='external-test',scorer=scorer),
                                node,task,ref,ReceiptExecutor())
            self.assertEqual(result['score'],.375)
            self.assertEqual(calls,[('Blue Harbor',ref)])
            metrics=result['rows'][0]['task_metrics']
            self.assertIsNone(metrics['answer_f1'])
            self.assertEqual(metrics['metric_status']['answer_f1'],'unavailable_official_judge')

    def test_empty_or_unusable_answers_stay_zero_in_main_and_diagnostics(self):
        for answer in ('','   ','Blue Harbor'):
            with self.subTest(answer=answer),self.temporary() as tmp:
                archive,node=self.archive(tmp); task,ref=self.multihop()
                measure=self.measure(tmp,archive,scorer=lambda *a:self.fail('Unusable answer must not call scorer'))
                result=self.run_one(measure,node,task,ref,ReceiptExecutor(answer=answer,usable=False))
                self.assertEqual(result['score'],0.)
                metrics=result['rows'][0]['task_metrics']
                self.assertEqual(metrics['answer_em'],0.)
                self.assertEqual(metrics['answer_f1'],0.)
                self.assertEqual(metrics['metric_status']['answer_f1'],'host_delivery_failure')

    def test_existing_synthetic_multihop_reference_needs_no_new_fields(self):
        with self.temporary() as tmp:
            archive,node=self.archive(tmp); task,ref=self.multihop()
            result=self.run_one(self.measure(tmp,archive),node,task,ref,ReceiptExecutor())
            self.assertEqual(result['score'],1.)
            self.assertEqual(result['rows'][0]['task_metrics']['answer_em'],1.)
            self.assertEqual(result['resource_usage']['calls'],0)

    def test_default_full_rejected_before_any_execution_or_model_creation(self):
        for scorer in (None,lambda answer,reference:.5):
            with self.subTest(external=scorer is not None),self.temporary() as tmp:
                archive,node=self.archive(tmp)
                task1,ref1=self.musique(True); task2,ref2=self.musique(False)
                def forbidden_factory(bank):
                    self.fail('Full must be rejected before model creation')
                measure=execution.Measurement(archive,Path(tmp)/'measurements',forbidden_factory,scorer=scorer)
                with self.assertRaisesRegex(ValueError,'paired sufficiency'):
                    measure.run(node,[task1,task2],{task1['question_id']:ref1,task2['question_id']:ref2},
                                role='D_fit',bank='common')
                self.assertEqual(list((Path(tmp)/'measurements').rglob('measured.json')),[])

    def test_cached_task_metrics_replay_without_execution(self):
        with self.temporary() as tmp:
            archive,node=self.archive(tmp); task,ref=self.multihop(); executor=ReceiptExecutor()
            measure=self.measure(tmp,archive)
            first=self.run_one(measure,node,task,ref,executor)
            second=self.run_one(self.measure(tmp,archive),node,task,ref,executor)
            self.assertEqual(first,second)
            self.assertEqual(len(executor.tasks),1)
            self.assertNotEqual(measure.epoch,'task-rule-em-v3')
            cell=next((Path(tmp)/'measurements').rglob('measured.json'))
            record=json.loads(cell.read_text(encoding='utf-8'))
            self.assertEqual(record['schema'],'rag-rsi-v3-measured-cell-2')
            self.assertEqual(record['identity']['dataset'],'multihop-rag')

    def test_proxy_configuration_changes_cache_identity(self):
        with self.temporary() as tmp:
            archive,node=self.archive(tmp); task,ref=self.multihop(); executor=ReceiptExecutor()
            first=self.run_one(self.measure(tmp,archive),node,task,ref,executor)
            second=self.run_one(self.measure(tmp,archive,allow_proxy_metrics=True),node,task,ref,executor)
            self.assertNotEqual(first['identity_hash'],second['identity_hash'])
            self.assertEqual(len(executor.tasks),2)

    def test_old_schema_or_missing_task_metrics_cannot_be_reused(self):
        for problem in ('old-schema','missing-metrics','wrong-dataset','invalid-score'):
            with self.subTest(problem=problem),self.temporary() as tmp:
                archive,node=self.archive(tmp); task,ref=self.multihop(); executor=ReceiptExecutor()
                measure=self.measure(tmp,archive)
                self.run_one(measure,node,task,ref,executor)
                cell=next((Path(tmp)/'measurements').rglob('measured.json'))
                record=json.loads(cell.read_text(encoding='utf-8'))
                if problem=='old-schema': record['schema']='rag-rsi-v3-measured-cell-1'
                elif problem=='missing-metrics': record['payload'].pop('task_metrics')
                elif problem=='wrong-dataset': record['payload']['task_metrics']['dataset']='musique'
                else: record['payload']['task_metrics']['answer_f1']=1.5
                record['payload_sha256']=digest(record['payload'])
                save(cell,record)
                with self.assertRaises(ValueError): self.run_one(measure,node,task,ref,executor)
                self.assertEqual(len(executor.tasks),1)

    def test_unknown_provider_outcome_is_never_scored_or_cached_as_failure(self):
        with self.temporary() as tmp:
            archive,node=self.archive(tmp); task,ref=self.multihop()
            executor=ReceiptExecutor(error=UnknownProviderOutcome('synthetic unknown outcome'))
            with patch.object(execution,'score_task') as score:
                with self.assertRaises(UnknownProviderOutcome):
                    self.run_one(self.measure(tmp,archive),node,task,ref,executor)
                score.assert_not_called()
            self.assertEqual(list((Path(tmp)/'measurements').rglob('measured.json')),[])

    def test_proxy_switch_rejects_truthy_non_boolean_values(self):
        for value in (None,1,'true'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                execution.Measurement(None,'.',None,allow_proxy_metrics=value)


if __name__=='__main__':
    unittest.main()
