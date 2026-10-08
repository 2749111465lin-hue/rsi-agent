"""Exact request-byte regressions with synthetic data and a recording transport."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from code_rsi.budget import Ledger, stable
from code_rsi.live_evolution import _BoundModel, _model_identity
from code_rsi.v3.diagnostics import compact_feedback
from code_rsi.v3.evolution import ProgramDeveloper, _fit_development_request, experience_card
from code_rsi.v3.execution import root_files
from code_rsi.v3.infrastructure import StructuredModel, PROMPTS
from test_v3_feedback_flow import receipt, window, quote
from test_v3_diagnostics import measurement


def stress(char):
    rounds=[]
    for i in range(3):
        sources=[window('doc-'+str(i)+'-'+str(j),char*1500,sid='s'+str(j)) for j in range(2)]
        rounds.append({'sources':sources,'queries':[char*900+str(i),char*850],
                       'quotes':[quote(s,char*300) for s in sources]})
    rows=[]
    for i,score in enumerate((0.,.25,.75,1.)):
        row=receipt(rounds,qid='q'+str(i),score=score)
        row['answer']=char*400
        row['host_evidence_trace']['final_observations'][0]['response']['answer']=row['answer']
        row['candidate_reported']={'state':{'gaps':[str(j)+char*239 for j in range(3)],
                                          'conflicts':[str(j)+char*239 for j in range(3)]}}
        rows.append(row)
    parent_rows=deepcopy(rows)
    for row in parent_rows: row['score']=1-row['score']
    parent=measurement(parent_rows,node='parent',program_id='parent-program',valid_program=True,resource_usage={'calls':16})
    current=measurement(rows,node='child',parent_measurement=parent,program_id='child-program',valid_program=True,resource_usage={'calls':16})
    tasks=[{'question_id':r['question_id'],'question':char*800,'answer':'GOLD_SECRET'} for r in rows]
    experiences=[experience_card(current,parent,operator='Improve',module='query_rewrite',step=i,
                                mechanism='synthetic change') for i in range(4)]
    payload={'source_files':root_files(), 'decision':{'target_module':'query_rewrite','parent_node_id':'child'},
             'experience':experiences,'feedback':compact_feedback(current,tasks),
             'edit_boundary':'Change reusable module behavior, do not embed examples/answers. Return complete changed files.'}
    return current,tasks,payload


class DeveloperRequestBudgetTests(unittest.TestCase):
    def setUp(self):
        base=Path(__file__).parent/'runs';base.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='developer_bytes_',dir=base)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.sent=[]
        self.ledger=Ledger(self.root/'ledger.jsonl',{'run':{'calls':8,'cny':5}})
        def send(body):
            self.sent.append(deepcopy(body));p=json.loads(body['messages'][1]['content'])
            value={'writes':{'rag.py':p['source_files']['rag.py']+'\nSYNTHETIC_EDIT=1\n'},
                   'target_module':p['decision']['target_module'],'mechanism':'synthetic budget fixture'}
            return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(value)}}],
                    'usage':{'prompt_tokens':100,'completion_tokens':100}}
        self.model=StructuredModel(self.root/'requests',self.ledger,send,bank='synthetic',
                                  prices={'input_hit':.04,'input_miss':2,'output':8})

    def test_request_size_matches_exact_existing_wire_body_and_does_not_dispatch(self):
        payload={'question':'quoted "line"\n中文','evidence':[]}
        for stage in PROMPTS:
            expected={'model':'deepseek-flash','stream':False,'thinking':{'type':'disabled'},
                      'temperature':0,'max_tokens':self.model.limits[stage],'response_format':{'type':'json_object'},
                      'messages':[{'role':'system','content':PROMPTS[stage]},
                                  {'role':'user','content':stable(payload)}]}
            self.assertEqual(self.model.request_body(stage,payload),expected)
            self.assertEqual(self.model.request_size(stage,payload),len(stable(expected).encode()))
        self.assertEqual(self.sent,[]);self.assertEqual(self.ledger.events,[])

    def test_small_payload_preserves_identical_request(self):
        p={'source_files':root_files(),'decision':{},'experience':[],'feedback':{'cases':[]}}
        self.assertIs(_fit_development_request(self.model,p),p)

    def test_unicode_and_json_escaping_fit_without_changing_scores_or_source(self):
        for char in ('汉','🧭','"'):
            with self.subTest(char=char):
                _,_,payload=stress(char);before=deepcopy(payload)
                self.assertGreater(self.model.request_size('develop',payload),self.model.max_input_bytes)
                packed=_fit_development_request(self.model,payload)
                self.assertEqual(payload,before)
                self.assertLessEqual(self.model.request_size('develop',packed),self.model.max_input_bytes)
                self.assertEqual(packed['source_files'],payload['source_files'])
                self.assertEqual(packed['decision'],payload['decision'])
                self.assertEqual(packed['experience'],payload['experience'])
                self.assertEqual(len(packed['feedback']['cases']),4)
                for old,new in zip(payload['feedback']['cases'],packed['feedback']['cases']):
                    for key in ('question_id','question','host_score','sampled_repeat_score','signed_delta','parent_host_score'):
                        self.assertEqual(old[key],new[key])
                    for key in ('execution_flow','parent_execution_flow'):
                        self.assertEqual(old[key]['counts'],new[key]['counts'])
                        self.assertEqual([r['returned_not_presented_count'] for r in old[key]['reads']],
                                         [r['returned_not_presented_count'] for r in new[key]['reads']])
                self.assertTrue(packed['feedback']['request_budget']['source_preserved'])
                self.assertNotIn('GOLD_SECRET',stable(packed))
        self.assertEqual(self.sent,[])

    def test_second_tier_preserves_counts_scores_and_read_order(self):
        _,_,payload=stress('汉')
        self.model.max_input_bytes=100000
        before=deepcopy(payload)
        packed=_fit_development_request(self.model,payload)
        self.assertEqual(payload,before)
        self.assertLessEqual(self.model.request_size('develop',packed),100000)
        self.assertEqual(packed['source_files'],payload['source_files'])
        self.assertEqual(packed['experience'],payload['experience'])
        self.assertEqual(len(packed['feedback']['cases']),4)
        self.assertEqual(len(packed['feedback']['request_budget']['reductions']),2)
        for old,new in zip(payload['feedback']['cases'],packed['feedback']['cases']):
            for key in ('question_id','question','host_score','sampled_repeat_score','signed_delta','parent_host_score'):
                self.assertEqual(old[key],new[key])
            for key in ('execution_flow','parent_execution_flow'):
                self.assertEqual(old[key]['counts'],new[key]['counts'])
                self.assertEqual(len(old[key]['reads']),len(new[key]['reads']))
                for old_read,new_read in zip(old[key]['reads'],new[key]['reads']):
                    self.assertEqual(old_read['returned_not_presented_count'],new_read['returned_not_presented_count'])
                    for field in ('queries','sources','quotes'):
                        self.assertEqual(new_read[field],[])
        self.assertEqual(self.sent,[]);self.assertEqual(self.ledger.events,[])

    def test_actual_program_developer_uses_packed_payload(self):
        current,tasks,payload=stress('汉')
        ProgramDeveloper(self.model).propose({'files':payload['source_files']},payload['decision'],
                                            payload['experience'],current,tasks,{'answers':['GOLD_SECRET']})
        self.assertEqual(len(self.sent),1)
        self.assertLessEqual(len(stable(self.sent[0]).encode()),120000)
        delivered=json.loads(self.sent[0]['messages'][1]['content'])
        self.assertEqual(len(delivered['feedback']['cases']),4)
        self.assertIn('request_budget',delivered['feedback'])
        self.assertNotIn('GOLD_SECRET',stable(delivered))

    def test_unfittable_source_stops_before_paid_dispatch_without_truncation(self):
        _,_,payload=stress('汉')
        payload['source_files']['rag_core.py'] += '\n#'+('x'*180000)
        before=deepcopy(payload)
        with self.assertRaisesRegex(ValueError,'full source'):
            _fit_development_request(self.model,payload)
        self.assertEqual(payload,before)
        self.assertEqual(self.sent,[]);self.assertEqual(self.ledger.events,[])

    def test_live_bound_model_exposes_same_byte_limit_and_role_checked_sizer(self):
        config={'name':self.model.model,'output_limits':self.model.limits,
                'max_input_bytes':self.model.max_input_bytes,'prices':self.model.prices}
        bound=_BoundModel(self.model,_model_identity(config,PROMPTS),{'develop'},lambda:None)
        self.assertEqual(bound.max_input_bytes,120000)
        self.assertEqual(bound.request_size('develop',{}),self.model.request_size('develop',{}))
        with self.assertRaisesRegex(RuntimeError,'role'):
            bound.request_size('answer',{})
        self.assertEqual(self.sent,[])

if __name__=='__main__':unittest.main()
