"""Paired scheduling uses synthetic model responses and real host caches/ledgers."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from code_rsi import live_evolution as live, paired_evolution as paired
from code_rsi.budget import digest, save
from code_rsi.v3 import execution
from code_rsi.v3.evolution import ProgramDeveloper, freeze
from code_rsi.v3.infrastructure import UnknownProviderOutcome
from test_v3_live_evolution import FixtureTransport, fake_execute
import test_v3_live_phases as phase_fixture


class Crash(RuntimeError):
    pass


class PairedEntryTests(unittest.TestCase):
    binding = phase_fixture.LivePhaseTests.binding
    make_plan = phase_fixture.LivePhaseTests.make_plan
    read_role = phase_fixture.LivePhaseTests.read_role
    rewrite_role = phase_fixture.LivePhaseTests.rewrite_role
    staged = phase_fixture.LivePhaseTests.staged

    def setUp(self):
        phase_fixture.LivePhaseTests.setUp(self)
        template = self.staged(fit_only=True)
        template['expansions'] = 2
        template['max_calls'] = 23
        self.out = self.base/'paired'
        template['output_dir'] = str(self.out/'template_validation_only')
        self.pair = {'schema':paired.SCHEMA,'purpose':'paired_development_smoke',
            'output_dir':str(self.out),'search_template':template,'blocks':2,
            'schedule_policy':paired.SCHEDULE_POLICY,'max_calls':78,'hard_cny':30,
            'entry_sha256':live._hash(paired.__file__)}

    def run_fixture(self, transport=None, **kwargs):
        factory=kwargs.pop('transport_factory',lambda:transport)
        with patch.object(execution,'execute',side_effect=fake_execute) as calls:
            result=paired.run(self.pair,approved_plan_hash=digest(self.pair),execute=True,
                              transport_factory=factory,**kwargs)
        return result,calls.call_count

    def test_static_preflight_counts_shared_root_without_dispatch_or_references(self):
        original=live._file
        parsed=[]
        ref=self.plan['panels']['D_fit']['references_file']
        def observe(item,parse=False):
            if item==ref and parse: parsed.append(item)
            return original(item,parse=parse)
        with patch.object(live,'_file',side_effect=observe), \
             patch.object(live,'credential_from_plan',side_effect=AssertionError('credential')):
            report=paired.preflight(self.pair)
        self.assertEqual(report['max_calls'],78)
        self.assertEqual(report['proposal_opportunities'],8)
        self.assertEqual(report['ledger_limits']['root:0']['calls'],7)
        self.assertEqual(report['ledger_limits']['arm:0:cases']['calls'],16)
        self.assertEqual(parsed,[])
        self.assertFalse(self.out.exists())
        self.assertAlmostEqual(report['conservative_cny_upper_bound'],
                               2*(paired._cost(self.plan,7,0)+2*paired._cost(self.plan,14,2)))

    def test_frozen_alternating_schedule(self):
        schedule=paired.preflight(self.pair)['schedule']
        self.assertEqual([(x['block'],x['condition'],x['slot']) for x in schedule],
            [(0,'cases',0),(0,'trace',0),(0,'trace',1),(0,'cases',1),
             (1,'trace',0),(1,'cases',0),(1,'cases',1),(1,'trace',1)])

    def test_rejects_wrong_scope_budget_or_runtime(self):
        for key,value in [('max_calls',77),('hard_cny',.01),('blocks',True),
                          ('entry_sha256','bad'),('schedule_policy','adaptive')]:
            p=deepcopy(self.pair);p[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):paired.preflight(p)
        for key,value in [('memory','mechanism'),('feedback','trace'),('parent_policy','adaptive')]:
            p=deepcopy(self.pair);p['search_template']['controls'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):paired.preflight(p)

    def test_exact_authorization_before_writes(self):
        with self.assertRaisesRegex(ValueError,'exact paired'):
            paired.run(self.pair,approved_plan_hash='bad',execute=True,transport_factory=FixtureTransport)
        self.assertFalse(self.out.exists())

    def test_zero_prefix_spends_nothing_and_has_no_roots(self):
        def forbidden():raise AssertionError('transport')
        result,count=self.run_fixture(transport_factory=forbidden,stop_after=0)
        self.assertEqual((result['completed_opportunities'],count),(0,0))
        self.assertEqual(result['prepared_blocks'],[])
        self.assertFalse((self.out/'blocks').exists())

    def test_complete_shared_roots_distinct_blocks_and_arms(self):
        transport=FixtureTransport()
        result,count=self.run_fixture(transport)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['completed_opportunities'],8)
        # One root plus two first valid candidates per block. Second same-source
        # proposal is rejected; successful proposals are never topped up.
        self.assertEqual(count,6)
        self.assertEqual(len(transport.requests),14)
        self.assertEqual(sum(s=='develop' for s,_ in transport.requests),8)
        for b in range(2):
            def root(c):return json.loads((self.out/'blocks'/str(b)/c/'measurements/shared_root.json').read_bytes())['result']
            self.assertEqual(root('cases'),root('trace'))
            self.assertTrue((self.out/'blocks'/str(b)/'feedback_gate.json').exists())
            for c in ('cases','trace'):
                out=self.out/'blocks'/str(b)/c
                self.assertTrue((out/'phase_search.json').exists())
                self.assertFalse((out/'delivery_lock.json').exists())
                self.assertFalse((out/'report.json').exists())
        usage=result['ledger']['used']
        self.assertEqual(usage['run']['calls'],14)
        self.assertEqual(usage['root:0']['calls'],1)
        self.assertEqual(usage['root:1']['calls'],1)
        self.assertEqual(usage['arm:0:cases']['calls'],3)
        self.assertEqual(usage['arm:0:trace']['calls'],3)
        self.assertEqual(sum(x['attempt']['status']=='rejected' for x in result['terminal_records']),4)
        self.assertNotEqual(json.loads((self.out/'blocks/0/cases/measurements/shared_root.json').read_bytes())['result']['identity_hash'],
                            json.loads((self.out/'blocks/1/cases/measurements/shared_root.json').read_bytes())['result']['identity_hash'])

    def test_completion_resume_never_creates_transport_or_executes(self):
        first,_=self.run_fixture(FixtureTransport())
        def forbidden():raise AssertionError('resume transport')
        second,count=self.run_fixture(transport_factory=forbidden)
        self.assertEqual(first,second)
        self.assertEqual(count,0)

    def test_partial_resume_and_shorter_request_do_not_rewind(self):
        transport=FixtureTransport()
        first,count=self.run_fixture(transport,stop_after=1)
        self.assertEqual((first['completed_opportunities'],count),(1,2))
        short,count=self.run_fixture(transport_factory=lambda:(_ for _ in ()).throw(AssertionError('transport')),stop_after=0)
        self.assertEqual(short['completed_opportunities'],1)
        self.assertEqual(count,0)
        full,count=self.run_fixture(transport)
        self.assertEqual(full['completed_opportunities'],8)
        self.assertEqual(count,4)
        self.assertEqual(len(transport.requests),14)

    def test_gate_is_joint_and_full_trace_size_failure_prevents_both_developers(self):
        original=ProgramDeveloper.prepare_request
        def oversized(developer,*args,**kwargs):
            payload=original(developer,*args,**kwargs)
            if developer.feedback_condition=='trace':payload['oversized_fixture']='x'*130000
            return payload
        transport=FixtureTransport()
        with patch.object(ProgramDeveloper,'prepare_request',side_effect=None,autospec=True) as mock:
            mock.side_effect=oversized
            with self.assertRaisesRegex(ValueError,'oversized'):
                self.run_fixture(transport)
        self.assertEqual([s for s,_ in transport.requests],['answer'])
        self.assertFalse((self.out/'blocks/0/feedback_gate.json').exists())

    def test_trace_and_cases_common_payload_mismatch_stops_before_proposals(self):
        original=ProgramDeveloper.prepare_request
        def wrong(developer,*args,**kwargs):
            payload=original(developer,*args,**kwargs)
            if developer.feedback_condition=='trace':payload['feedback']['cases'][0]['prediction']='unpaired'
            return payload
        transport=FixtureTransport()
        with patch.object(ProgramDeveloper,'prepare_request',autospec=True,side_effect=wrong):
            with self.assertRaisesRegex(HostError,'same root'):
                self.run_fixture(transport)
        self.assertEqual([s for s,_ in transport.requests],['answer'])

    def test_unknown_outcome_blocks_resume_before_new_transport(self):
        with self.assertRaises(UnknownProviderOutcome):self.run_fixture(FixtureTransport(unknown=True))
        def forbidden():raise AssertionError('new transport after unknown')
        with self.assertRaises(UnknownProviderOutcome):self.run_fixture(transport_factory=forbidden)

    def test_schedule_receipt_tamper_rejected_without_new_calls(self):
        transport=FixtureTransport();self.run_fixture(transport,stop_after=1)
        path=self.out/'schedule/0.json';value=json.loads(path.read_bytes());value['attempt']['status']='changed';save(path,value)
        before=len(transport.requests)
        with self.assertRaises(ValueError):self.run_fixture(transport,stop_after=1)
        self.assertEqual(len(transport.requests),before)

    def test_body_gate_tamper_rejected_without_new_calls(self):
        transport=FixtureTransport();self.run_fixture(transport,stop_after=1)
        path=self.out/'blocks/0/feedback_gate.json';value=json.loads(path.read_bytes());value['common_payload_equal']=False;save(path,value)
        before=len(transport.requests)
        with self.assertRaises(ValueError):self.run_fixture(transport)
        self.assertEqual(len(transport.requests),before)

    def test_slot_crash_after_core_completion_resumes_without_rebuy(self):
        original=paired.freeze
        def fail(path,value):
            if Path(path).parent.name=='schedule':raise Crash('before schedule receipt')
            return original(path,value)
        transport=FixtureTransport()
        with patch.object(paired,'freeze',side_effect=fail),self.assertRaises(Crash):self.run_fixture(transport,stop_after=1)
        before=len(transport.requests)
        result,count=self.run_fixture(transport_factory=lambda:(_ for _ in ()).throw(AssertionError('transport')),stop_after=1)
        self.assertEqual(result['completed_opportunities'],1)
        self.assertEqual(count,0)
        self.assertEqual(len(transport.requests),before)

    def test_ledger_scope_cannot_be_relabelled_between_arms(self):
        self.run_fixture(FixtureTransport(),stop_after=1)
        path=self.out/'ledger.jsonl';events=[json.loads(s) for s in path.read_text().splitlines()]
        first=next(e for e in events if e['event']=='reserve');first['metadata']['paired_condition']='trace'
        path.write_text('\n'.join(json.dumps(e) for e in events)+'\n',encoding='utf-8')
        with self.assertRaisesRegex(HostError,'ledger scopes'):
            self.run_fixture(transport_factory=lambda:(_ for _ in ()).throw(AssertionError('transport')))


from code_rsi.v3.execution import HostError
if __name__=='__main__':unittest.main()
