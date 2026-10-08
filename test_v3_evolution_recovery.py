"""Synthetic host-state recovery tests: no WSL, network, or model-quality claims."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from code_rsi.budget import digest, save
from code_rsi.v3 import evolution
from code_rsi.v3.datasets import adapt_multihop
from test_v3_fixture_origin import bind_synthetic_origin


class InjectedCrash(RuntimeError):
    pass


class Audit:
    def __init__(self):
        self.invocations=[]
        self.computed=[]
        self.seen_inputs=[]
        self.crash_report_after_save=False
        self.report_scores={False:.8,True:.1}


class FakeMeasurement:
    """Persist deterministic fixture scores so replay tests can count new work."""
    epoch='synthetic-host-state-only-v1'

    def __init__(self,archive,directory,audit):
        self.archive=archive
        self.directory=Path(directory)
        self.audit=audit

    def run(self,node,tasks,references,*,role,bank,repeats):
        key=(node['node_id'],role)
        self.audit.invocations.append(key)
        self.audit.seen_inputs.append((role,deepcopy(tasks),deepcopy(references)))
        panel_hash=digest(tasks)
        path=self.directory/(digest([key,panel_hash,digest(references),bank,repeats])+'.json')
        cached=evolution.read(path)
        if cached is not None:
            return cached
        self.audit.computed.append(key)
        child=node['parent_node_id'] is not None
        score=({False:.25,True:.75} if role=='D_fit' else
               {False:.2,True:.9} if role=='D_select' else self.audit.report_scores)[child]
        rows=[{'question_id':t['question_id'],'role':role,'repeat':0,'answer':'synthetic fixture answer',
               'score':score,'execution_ok':True,'answer_usable':True,
               'citation_source_valid':True,'failure_classes':[],'model_errors':[]}
              for t in tasks]
        rows=[bind_synthetic_origin(row,synthesize_final=True) for row in rows]
        result={'node_id':node['node_id'],'program_id':node['program_id'],
                'role':role,'panel_hash':panel_hash,'evaluator_epoch':self.epoch,
                'score':score,'per_question':{t['question_id']:score for t in tasks},
                'complete':True,'metric':'em','valid_program':True,'resource_usage':{'calls':0},'rows':rows}
        save(path,result)
        if role=='D_report' and self.audit.crash_report_after_save:
            self.audit.crash_report_after_save=False
            raise InjectedCrash('after report measurement, before runner receipt')
        return result


class Developer:
    def __init__(self,reject_first=False,repeat_root=False):
        self.calls=[]
        self.reject_first=reject_first
        self.repeat_root=repeat_root
        self.root_core=None

    def propose(self,program,decision,experience,result,tasks,references):
        self.calls.append(deepcopy({'decision':decision,'experience':experience,
                                   'result_role':result['role'],'tasks':tasks,'references':references}))
        if self.root_core is None:
            self.root_core=program['files']['rag_core.py']
        if self.reject_first and len(self.calls)==1:
            raise ValueError('synthetic preflight rejection')
        source=program['files']['rag_core.py']+'\nSYNTHETIC_HOST_TEST_EDIT = '+str(len(self.calls))+'\n'
        if self.repeat_root and len(self.calls)>1:
            source=self.root_core
        return {'writes':{'rag_core.py':source},'mechanism':'synthetic state-machine fixture',
                'target_module':decision['target_module']}


class EvolutionRecoveryTests(unittest.TestCase):
    def temporary(self):
        # Keep all fixture artifacts in the explicitly authorized D: project.
        directory=Path(__file__).parent/'runs'
        directory.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(prefix='v3_recovery_test_',dir=directory)

    def inputs(self,**overrides):
        manifest={'expansions':1,'metric':'em','select_candidates':3,'repeats':1,
                  'root_config':{'mode':'iterative'},'limits':{'max_models':7},'synthetic':True}
        manifest.update(overrides)
        panels={}; references={}
        for role in ('D_fit','D_select','D_report'):
            task,reference=adapt_multihop({'id':role,'query':'Synthetic question '+role,
                                         'answer':'private fixture '+role})
            panels[role]=[task]
            references[role]={task['question_id']:reference}
        return manifest,panels,references

    def runner(self,directory,developer=None,audit=None,inputs=None,**overrides):
        developer=developer if developer is not None else Developer()
        audit=audit if audit is not None else Audit()
        args=inputs if inputs is not None else self.inputs(**overrides)
        def factory(archive,path,*unused,**kwargs):
            return FakeMeasurement(archive,path,audit)
        with patch.object(evolution,'Measurement',side_effect=factory):
            return evolution.EvolutionRunner(directory,*args,model_factory=None,developer=developer)

    def test_external_mutation_cannot_change_frozen_inputs(self):
        with self.temporary() as tmp:
            args=self.inputs()
            original=deepcopy(args)
            audit=Audit()
            runner=self.runner(tmp,inputs=args,audit=audit)
            args[0]['root_config']['mode']='changed'
            args[0]['limits']['max_models']=99
            args[1]['D_fit'][0]['question']='replaced after freeze'
            args[1]['D_fit'][0]['documents'].append({'docid':'new','text':'late document'})
            qid=original[1]['D_fit'][0]['question_id']
            args[2]['D_fit'][qid]['answers'].append('late answer')
            self.assertEqual((runner.manifest,runner.panels,runner.references),original)
            runner.run()
            fit=next(row for row in audit.seen_inputs if row[0]=='D_fit')
            self.assertEqual(fit[1:],(original[1]['D_fit'],original[2]['D_fit']))

    def test_direct_input_mutation_fails_before_measurement(self):
        for target in ('manifest','panels','references'):
            with self.subTest(target=target),self.temporary() as tmp:
                audit=Audit(); runner=self.runner(tmp,audit=audit)
                if target=='manifest':
                    runner.manifest['root_config']['mode']='changed'
                elif target=='panels':
                    runner.panels['D_fit'][0]['question']='changed'
                else:
                    qid=runner.panels['D_fit'][0]['question_id']
                    runner.references['D_fit'][qid]['answers'].append('changed')
                with self.assertRaisesRegex(ValueError,'frozen run inputs'):
                    runner.run()
                self.assertEqual(audit.invocations,[])
                self.assertFalse((Path(tmp)/'root.json').exists())

    def test_runtime_source_change_fails_before_measurement(self):
        with self.temporary() as tmp:
            audit=Audit(); runner=self.runner(tmp,audit=audit)
            changed={**evolution._runtime_source_hashes(),'synthetic_source_change':'changed'}
            with patch.object(evolution,'_runtime_source_hashes',return_value=changed):
                with self.assertRaisesRegex(ValueError,'runtime source changed'):
                    runner.run()
            self.assertEqual(audit.invocations,[])

    def test_root_and_child_archive_write_survive_missing_receipt(self):
        for receipt in ('root.json','child.json'):
            with self.subTest(receipt=receipt),self.temporary() as tmp:
                developer=Developer(); audit=Audit()
                runner=self.runner(tmp,developer,audit)
                real_save=evolution.save
                def crashing_save(path,value):
                    if Path(path).name==receipt:
                        raise InjectedCrash('archive complete, receipt absent')
                    real_save(path,value)
                with patch.object(evolution,'save',side_effect=crashing_save):
                    with self.assertRaises(InjectedCrash):
                        runner.run()
                node_paths=list((Path(tmp)/'archive'/'nodes').glob('*.json'))
                self.assertEqual(len(node_paths),1 if receipt=='root.json' else 2)
                missing=Path(tmp)/('root.json' if receipt=='root.json' else 'steps/0/child.json')
                self.assertFalse(missing.exists())
                before={p.stem:p.read_bytes() for p in node_paths}
                resumed=self.runner(tmp,developer,audit)
                report=resumed.run()
                self.assertTrue(missing.exists())
                self.assertEqual(report['status'],'complete')
                self.assertEqual(len(developer.calls),1)
                self.assertEqual(len(list((Path(tmp)/'archive'/'nodes').glob('*.json'))),2)
                for node_id,contents in before.items():
                    self.assertEqual((Path(tmp)/'archive'/'nodes'/(node_id+'.json')).read_bytes(),contents)

    def test_existing_attempt_cannot_create_different_program(self):
        with self.temporary() as tmp:
            runner=self.runner(tmp)
            files=evolution.root_files()
            node=evolution.recoverable_record(runner.archive,files,{},session_id='attempt',attempt=0)
            replay=evolution.recoverable_record(runner.archive,files,{},session_id='attempt',attempt=0)
            self.assertEqual(node,replay)
            changed={**files,'rag_core.py':files['rag_core.py']+'\nALTERED = True\n'}
            with self.assertRaisesRegex(ValueError,'receipt source'):
                evolution.recoverable_record(runner.archive,changed,{},session_id='attempt',attempt=0)
            self.assertEqual(len(list((Path(tmp)/'archive'/'nodes').glob('*.json'))),1)

    def test_wrong_child_receipt_rejected_before_child_measurement(self):
        with self.temporary() as tmp:
            developer=Developer(); audit=Audit()
            self.runner(tmp,developer,audit).run()
            save(Path(tmp)/'steps/0/child.json',evolution.read(Path(tmp)/'root.json'))
            before=len(audit.invocations)
            with self.assertRaisesRegex(ValueError,'receipt identity'):
                self.runner(tmp,developer,audit).run()
            self.assertEqual(len(audit.invocations),before+1)
            self.assertEqual(len(developer.calls),1)

    def test_changed_proposal_rejected_even_with_existing_child(self):
        with self.temporary() as tmp:
            audit=Audit(); self.runner(tmp,audit=audit).run()
            path=Path(tmp)/'steps/0/proposal.json'
            proposal=evolution.read(path)
            proposal['writes']['rag_core.py']+='\nUNRELATED_RECOVERY_EDIT = True\n'
            save(path,proposal)
            before=len(audit.invocations)
            with self.assertRaisesRegex(ValueError,'receipt source'):
                self.runner(tmp,audit=audit).run()
            self.assertEqual(len(audit.invocations),before+1)

    def test_tampered_root_receipt_rejected_before_measurement(self):
        with self.temporary() as tmp:
            audit=Audit(); self.runner(tmp,audit=audit).run()
            path=Path(tmp)/'root.json'; root=evolution.read(path)
            root['program_id']='0'*64; save(path,root)
            before=len(audit.invocations)
            with self.assertRaisesRegex(ValueError,'receipt differs'):
                self.runner(tmp,audit=audit).run()
            self.assertEqual(len(audit.invocations),before)

    def test_select_candidates_one_never_adds_better_fit_child(self):
        with self.temporary() as tmp:
            audit=Audit(); report=self.runner(tmp,audit=audit,select_candidates=1).run()
            root=evolution.read(Path(tmp)/'root.json')
            self.assertEqual(report['fit_nodes'],2)
            self.assertEqual(report['delivery_lock']['candidate_ids'],[root['node_id']])
            self.assertEqual([n for n,r in audit.computed if r=='D_select'],[root['node_id']])

    def test_select_candidate_limit_is_positive_bounded_integer(self):
        for value in (0,-1,True,1.5,'1',18):
            with self.subTest(value=value),self.temporary() as tmp:
                with self.assertRaisesRegex(ValueError,'select_candidates'):
                    self.runner(tmp,select_candidates=value)

    def test_preflight_rejection_reaches_next_development_and_replays(self):
        with self.temporary() as tmp:
            developer=Developer(reject_first=True); audit=Audit()
            report=self.runner(tmp,developer,audit,expansions=2).run()
            self.assertEqual(report['fit_nodes'],2)
            self.assertEqual(len(developer.calls),2)
            recent=developer.calls[1]['decision']['recent_rejections']
            self.assertEqual(recent[0]['reason'],'synthetic preflight rejection')
            self.assertTrue(all(c['result_role']=='D_fit' for c in developer.calls))
            before=len(audit.computed)
            self.runner(tmp,developer,audit,expansions=2).run()
            self.assertEqual(len(developer.calls),2)
            self.assertEqual(len(audit.computed),before)

    def test_duplicate_program_rejected_and_rejection_reaches_next_step(self):
        with self.temporary() as tmp:
            developer=Developer(repeat_root=True)
            report=self.runner(tmp,developer,expansions=3).run()
            self.assertEqual(report['fit_nodes'],2)
            self.assertEqual(len(list((Path(tmp)/'archive'/'nodes').glob('*.json'))),2)
            reasons=[r['reason'] for r in developer.calls[2]['decision']['recent_rejections']]
            self.assertIn('previously visited program',reasons)

    def test_report_crash_keeps_lock_and_does_not_restart_development(self):
        with self.temporary() as tmp:
            developer=Developer(); audit=Audit(); audit.crash_report_after_save=True
            with self.assertRaises(InjectedCrash):
                self.runner(tmp,developer,audit).run()
            lock_path=Path(tmp)/'delivery_lock.json'; locked_bytes=lock_path.read_bytes()
            completed_before=list(audit.computed)
            self.assertEqual(len(developer.calls),1)
            report=self.runner(tmp,developer,audit).run()
            self.assertEqual(lock_path.read_bytes(),locked_bytes)
            self.assertEqual(len(developer.calls),1)
            self.assertEqual(audit.computed[:len(completed_before)],completed_before)
            self.assertEqual([role for _,role in audit.computed[len(completed_before):]],['D_report'])
            self.assertFalse(report['report_used_for_decisions'])
            self.assertLess(report['paired_report_gain'],0)
            self.assertNotEqual(report['delivery_lock']['node_id'],evolution.read(Path(tmp)/'root.json')['node_id'])

    def test_report_scores_cannot_change_delivery_lock(self):
        locks=[]
        for scores in ({False:1.,True:0.},{False:0.,True:1.}):
            with self.subTest(scores=scores),self.temporary() as tmp:
                audit=Audit(); audit.report_scores=scores
                self.runner(tmp,audit=audit).run()
                locks.append((Path(tmp)/'delivery_lock.json').read_bytes())
        self.assertEqual(locks[0],locks[1])


if __name__=='__main__':
    unittest.main()
