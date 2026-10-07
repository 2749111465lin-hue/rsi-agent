"""Host boundaries and durable accounting tests, all synthetic and zero API."""
import copy
import json
import tempfile
import unittest
import time
from unittest.mock import patch
from pathlib import Path
from code_rsi.budget import Ledger,LimitExceeded
from code_rsi.v3.infrastructure import StructuredModel,LocalCorpus,UnknownProviderOutcome,DeepSeekTransport
from code_rsi.v3.datasets import adapt_multihop
from code_rsi.v3.execution import HostBroker,root_files,validate_sources
from code_rsi.v3.evolution import freeze,program_change,experience_card,EvolutionRunner

def delayed_fixture_worker(connection, body, key, timeout):
    time.sleep(10)


class IntegrationTests(unittest.TestCase):
    def test_live_transport_worker_has_enforced_wall_deadline_without_network(self):
        started=time.monotonic()
        with patch("code_rsi.v3.infrastructure._network_worker",delayed_fixture_worker):
            with self.assertRaises(UnknownProviderOutcome):
                DeepSeekTransport("synthetic-not-a-key").send({},.05)
        self.assertLess(time.monotonic()-started,4)

    def temporary(self):
        folder=Path(__file__).parent/"runs"
        folder.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(dir=folder)

    def test_identical_body_same_bank_reuses_one_physical_call_after_restart(self):
        with self.temporary() as tmp:
            calls=[]
            def send(body):
                calls.append(body)
                return {"choices":[{"finish_reason":"stop","message":{"content":'{"answer":"x"}'}}],"usage":{"prompt_tokens":50,"completion_tokens":8}}
            ledger=Ledger(Path(tmp)/'ledger.jsonl',{'run':{'cny':1,'calls':5}})
            args=dict(bank='fit/q/0',prices={'input_miss':2,'input_hit':.04,'output':8})
            a=StructuredModel(Path(tmp)/'cache',ledger,send,**args)
            self.assertEqual(a.complete('answer',{'question':'q'}),{'answer':'x'})
            b=StructuredModel(Path(tmp)/'cache',ledger,send,**args)
            self.assertEqual(b.complete('answer',{'question':'q'}),{'answer':'x'})
            self.assertEqual(len(calls),1)
            c=StructuredModel(Path(tmp)/'cache',ledger,send,**{**args,'bank':'fit/q/1'})
            c.complete('answer',{'question':'q'})
            self.assertEqual(len(calls),2)

    def test_unknown_transport_is_not_silently_repurchased(self):
        with self.temporary() as tmp:
            def send(body): raise TimeoutError('unknown outcome')
            ledger=Ledger(Path(tmp)/'ledger.jsonl',{'run':{'cny':1,'calls':5}})
            model=StructuredModel(Path(tmp)/'cache',ledger,send,bank='q',prices={'input_miss':2,'input_hit':.04,'output':8})
            with self.assertRaises(UnknownProviderOutcome): model.complete('plan',{'question':'q'})
            with self.assertRaises(UnknownProviderOutcome): model.complete('plan',{'question':'q'})
            self.assertEqual(ledger.summary()['used']['run']['calls'],1)

    def test_truncation_frozen_and_charged_once(self):
        with self.temporary() as tmp:
            def send(body): return {'choices':[{'finish_reason':'length','message':{'content':'{}'}}]}
            ledger=Ledger(Path(tmp)/'ledger.jsonl',{'run':{'cny':1,'calls':5}})
            model=StructuredModel(Path(tmp)/'cache',ledger,send,bank='q',prices={'input_miss':2,'input_hit':.04,'output':8})
            for _ in range(2):
                with self.assertRaisesRegex(ValueError,'truncated'): model.complete('answer',{'question':'q'})
            self.assertEqual(ledger.summary()['used']['run']['calls'],1)

    def test_budget_prevents_dispatch(self):
        with self.temporary() as tmp:
            ledger=Ledger(Path(tmp)/'ledger.jsonl',{'run':{'cny':0,'calls':0}})
            model=StructuredModel(Path(tmp)/'cache',ledger,lambda b:self.fail('sent'),bank='q',prices={'input_miss':2,'input_hit':.04,'output':8})
            with self.assertRaises(LimitExceeded): model.complete('answer',{'question':'q'})

    def test_broker_overwrites_question_and_reserves_final(self):
        task,_=adapt_multihop({'id':'q','query':'public original','answer':'secret'})
        class Model:
            calls=[]
            def complete(self,stage,payload): self.calls.append((stage,payload)); return {}
        model=Model(); broker=HostBroker(task,LocalCorpus([]),model,max_models=2)
        broker('complete',{'stage':'read','payload':{'question':'changed'}})
        self.assertEqual(model.calls[0][1]['question'],'public original')
        self.assertNotIn('secret',json.dumps(model.calls))
        with self.assertRaises(ValueError): broker('complete',{'stage':'read','payload':{}})
        broker('complete',{'stage':'answer','payload':{}})
        self.assertEqual(len(model.calls),2)

    def test_local_corpus_exclusion_and_source_exactness(self):
        backend=LocalCorpus([{'docid':'a','text':'alpha beta'},{'docid':'b','text':'alpha secret'}],excluded=['b'])
        self.assertEqual([r['docid'] for r in backend.search('alpha')],['a'])
        self.assertEqual(backend.read('a',0,5)['text'],'alpha')
        with self.assertRaises(ValueError): backend.read('b',0,5)

    def test_root_is_real_independent_source(self):
        files=root_files({'mode':'iterative'})
        validate_sources(files)
        self.assertIn('from rag_core import RagEngine',files['rag.py'])
        self.assertNotIn('from code_rsi',files['rag_core.py'])
        with self.assertRaises(ValueError): program_change({'files':files},{'rag.py':files['rag.py']+'\n# only comment\n'})

    def test_lock_cannot_change(self):
        with self.temporary() as tmp:
            path=Path(tmp)/'delivery.json'; freeze(path,{'program':'a'})
            freeze(path,{'program':'a'})
            with self.assertRaises(ValueError): freeze(path,{'program':'b'})

    def test_signed_negative_reward_survives(self):
        common={'node_id':'child','program_id':'p','panel_hash':'panel','evaluator_epoch':'em','valid_program':True,
                'role':'D_fit','complete':True,'metric':'em','resource_usage':{'calls':2},'rows':[{'question_id':'q','score':0,'failure_classes':[],'answer_usable':True,'citation_source_valid':False,'answer':'wrong'}]}
        result={**common,'score':0,'per_question':{'q':0}}
        parent={**common,'node_id':'parent','score':1,'per_question':{'q':1},'rows':[{**common['rows'][0],'score':1}]}
        card=experience_card(result,parent,operator='Improve',module='answer_generation',step=1,mechanism='test')
        self.assertEqual(card['signed_delta_vs_best_parent'],-1)
        self.assertFalse(card['reward']['proxy_added_to_terminal_quality'])

if __name__=='__main__': unittest.main()
