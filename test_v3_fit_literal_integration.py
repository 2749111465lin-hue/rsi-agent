"""Development-literal rejection and recovery: synthetic, no network or WSL."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from code_rsi.budget import save
from code_rsi.v3 import evolution
import test_v3_evolution_recovery as recovery


class LiteralDeveloper(recovery.Developer):
    def __init__(self, reject_first=True):
        super().__init__()
        self.literal_first=reject_first

    def propose(self, program, decision, experience, result, tasks, references):
        value=super().propose(program,decision,experience,result,tasks,references)
        if self.literal_first and len(self.calls)==1:
            value['writes']['rag_core.py']+='\nTASK_SPECIFIC_TEXT = '+repr(tasks[0]['question'])+'\n'
        return value


class CaptureModel:
    def __init__(self): self.payloads=[]
    def complete(self, stage, payload):
        assert stage=='develop'
        self.payloads.append(deepcopy(payload))
        return {'writes':{'rag_core.py':payload['source_files']['rag_core.py']+'\nGENERIC_EDIT = True\n'},
                'mechanism':'synthetic generic change',
                'intended_target_module':payload['decision']['target_module']}


class LiteralIntegrationTests(unittest.TestCase):
    def setUp(self): self.helper=recovery.EvolutionRecoveryTests()

    def test_rejected_before_child_measurement_then_feedback_and_resume(self):
        with self.helper.temporary() as tmp:
            dev=LiteralDeveloper(); audit=recovery.Audit()
            report=self.helper.runner(tmp,dev,audit,expansions=2).run()
            folder=Path(tmp)/'steps/0'
            rejection=evolution.read(folder/'rejected.json')
            self.assertEqual(rejection['reason_code'],'fit_literal_match')
            self.assertEqual(evolution.read(folder/'literal_audit.json')['status'],'reject')
            self.assertTrue((folder/'received_proposal.json').exists())
            self.assertFalse((folder/'proposal.json').exists())
            self.assertFalse((folder/'child.json').exists())
            self.assertEqual(report['fit_nodes'],2)
            self.assertEqual(len(list((Path(tmp)/'archive/nodes').glob('*.json'))),2)
            self.assertEqual(dev.calls[1]['decision']['recent_rejections'],[rejection])
            self.assertNotIn('Synthetic question D_fit',str(rejection))
            self.assertEqual(len([x for x in audit.computed if x[1]=='D_fit']),2)
            before=len(audit.computed)
            self.assertEqual(self.helper.runner(tmp,dev,audit,expansions=2).run(),report)
            self.assertEqual(len(dev.calls),2)
            self.assertEqual(len(audit.computed),before)

    def test_received_checkpoint_prevents_second_development_after_audit_crash(self):
        for literal in (True,False):
            with self.subTest(literal=literal), self.helper.temporary() as tmp:
                dev=LiteralDeveloper(reject_first=literal); audit=recovery.Audit()
                runner=self.helper.runner(tmp,dev,audit)
                with patch.object(runner,'_audit_literals',side_effect=recovery.InjectedCrash('audit not yet saved')):
                    with self.assertRaises(recovery.InjectedCrash): runner.run()
                self.assertTrue((Path(tmp)/'steps/0/received_proposal.json').exists())
                self.assertFalse((Path(tmp)/'steps/0/child.json').exists())
                report=self.helper.runner(tmp,dev,audit).run()
                self.assertEqual(len(dev.calls),1)
                self.assertEqual(report['fit_nodes'],1 if literal else 2)

    def test_tampered_audit_stops_before_measurement_or_new_proposal(self):
        with self.helper.temporary() as tmp:
            dev=LiteralDeveloper(); audit=recovery.Audit()
            self.helper.runner(tmp,dev,audit).run()
            path=Path(tmp)/'steps/0/literal_audit.json'
            receipt=evolution.read(path); receipt['status']='pass'; save(path,receipt)
            before=len(audit.computed)
            with self.assertRaisesRegex(ValueError,'frozen run artifact differs: literal_audit'):
                self.helper.runner(tmp,dev,audit).run()
            self.assertEqual(len(dev.calls),1)
            self.assertEqual(len(audit.computed),before)

    def test_tampered_rejection_cannot_skip_source_revalidation(self):
        with self.helper.temporary() as tmp:
            dev=LiteralDeveloper(); self.helper.runner(tmp,dev).run()
            path=Path(tmp)/'steps/0/rejected.json'
            value=evolution.read(path); value['audit_sha256']='forged'; save(path,value)
            with self.assertRaisesRegex(ValueError,'saved literal rejection differs'):
                self.helper.runner(tmp,dev).run()
            self.assertEqual(len(dev.calls),1)

    def test_tampered_received_candidate_cannot_replace_accepted_source(self):
        with self.helper.temporary() as tmp:
            dev=LiteralDeveloper(reject_first=False)
            self.helper.runner(tmp,dev).run()
            path=Path(tmp)/'steps/0/received_proposal.json'
            value=evolution.read(path); value['writes']['rag_core.py']+='\nCHANGED = 2\n'; save(path,value)
            with self.assertRaisesRegex(ValueError,'receipt source differs from received'):
                self.helper.runner(tmp,dev).run()
            self.assertEqual(len(dev.calls),1)

    def test_relabeling_literal_rejection_does_not_bypass_saved_audit(self):
        with self.helper.temporary() as tmp:
            dev=LiteralDeveloper(); self.helper.runner(tmp,dev).run()
            path=Path(tmp)/'steps/0/rejected.json'
            save(path,{'reason':'previously visited program','next_step_allowed':True})
            with self.assertRaisesRegex(ValueError,'saved literal rejection differs'):
                self.helper.runner(tmp,dev).run()
            self.assertEqual(len(dev.calls),1)

    def test_approved_proposal_requires_received_checkpoint(self):
        with self.helper.temporary() as tmp:
            dev=LiteralDeveloper(reject_first=False); self.helper.runner(tmp,dev).run()
            (Path(tmp)/'steps/0/received_proposal.json').unlink()
            with self.assertRaisesRegex(ValueError,'no received proposal checkpoint'):
                self.helper.runner(tmp,dev).run()
            self.assertEqual(len(dev.calls),1)

    def test_exact_sent_feedback_used_without_private_or_other_roles(self):
        with self.helper.temporary() as tmp:
            model=CaptureModel(); dev=evolution.ProgramDeveloper(model)
            self.helper.runner(tmp,dev).run()
            context=evolution.read(Path(tmp)/'steps/0/literal_context.json')
            receipt=evolution.read(Path(tmp)/'steps/0/literal_audit.json')
            self.assertEqual(context['exposed_feedback'],model.payloads[0]['feedback'])
            self.assertEqual(context['public_tasks'],[{'question_id':'D_fit','question':'Synthetic question D_fit'}])
            for text in ('private fixture','Synthetic question D_select','Synthetic question D_report'):
                self.assertNotIn(text,str(context))
            self.assertEqual(receipt['status'],'pass')
            self.assertFalse(receipt['scope']['references_read'])

    def test_private_reference_argument_does_not_change_developer_request(self):
        helper=self.helper
        manifest,panels,refs=helper.inputs()
        with helper.temporary() as tmp:
            model=CaptureModel(); dev=evolution.ProgramDeveloper(model)
            runner=helper.runner(tmp,dev,inputs=(manifest,panels,refs)); runner.run()
            program=runner.archive.load_program(evolution.read(Path(tmp)/'root.json')['program_id'])
            decision=evolution.read(Path(tmp)/'steps/0/decision.json')
            result=runner._measure(evolution.read(Path(tmp)/'root.json'),'D_fit')
            dev.propose(program,decision,[],result,panels['D_fit'],{'private':'answer one'})
            dev.propose(program,decision,[],result,panels['D_fit'],{'private':'answer two'})
            self.assertEqual(model.payloads[-1],model.payloads[-2])


if __name__=='__main__': unittest.main()
