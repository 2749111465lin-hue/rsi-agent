"""Synthetic shared-cache/recovery contracts; no provider or credential access."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from code_rsi.budget import Ledger, digest, save
from code_rsi.v3.datasets import adapt_multihop
from code_rsi.v3.execution import HostBroker, HostError
from code_rsi.v3.infrastructure import LocalCorpus, StructuredModel, UnknownProviderOutcome
from code_rsi.v3.request_recovery import check_request_accounting, check_request_recovery


LIMITS = {'run': {'calls': 20, 'cny': 10}}
PRICES = {'input_hit': .04, 'input_miss': 2, 'output': 8}


class Transport:
    def __init__(self, error=None):
        self.error, self.calls = error, []

    def send(self, body, timeout):
        self.calls.append(deepcopy(body))
        if self.error:
            raise self.error
        return {'model': 'synthetic-recovery-fixture', 'usage': {'prompt_tokens': 20, 'completion_tokens': 8},
                'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({
                    'answer': 'Synthetic Port', 'citation_ids': [], 'evidence_sufficient': False})}}]}


class RequestRecoveryTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parent / 'runs'
        root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='request_recovery_', dir=root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def model(self, transport, *, bank='calibration/synthetic-q/0', ledger=None):
        return StructuredModel(self.root / 'requests', ledger or Ledger(self.root / 'ledger.jsonl', LIMITS),
                               transport, bank=bank, prices=PRICES)

    def resume(self, transport, payload, *, status_filename='live_status.json', bank='calibration/synthetic-q/0'):
        records = check_request_recovery(self.root, status_filename=status_filename)
        ledger = Ledger(self.root / 'ledger.jsonl', LIMITS)
        check_request_accounting(records, ledger)
        return self.model(transport, bank=bank, ledger=ledger).complete('answer', payload)

    def settled(self):
        transport = Transport()
        model = self.model(transport)
        model.complete('answer', {'question': 'Synthetic question', 'evidence': []})
        return transport, model

    def test_live_aliases_preserve_existing_entrypoint_contract(self):
        from code_rsi import live_evolution
        self.assertIs(live_evolution._check_request_recovery, check_request_recovery)
        self.assertIs(live_evolution._check_request_accounting, check_request_accounting)

    def test_fresh_directory_is_safe_and_read_only(self):
        self.assertEqual(check_request_recovery(self.root), {})
        self.assertEqual(list(self.root.iterdir()), [])

    def test_unknown_request_cannot_be_rebought_by_changing_body_or_bank(self):
        transport = Transport(TimeoutError('synthetic physical result unknown'))
        model = self.model(transport)
        with self.assertRaises(UnknownProviderOutcome):
            model.complete('answer', {'question': 'first body'})
        # The conservative reservation is settled, so ledger.pending alone is insufficient.
        ledger = Ledger(self.root / 'ledger.jsonl', LIMITS)
        self.assertEqual(ledger.summary()['pending'], 0)
        self.assertEqual(ledger.summary()['used']['run']['calls'], 1)
        resumed_transport = Transport()
        for bank in ('calibration/synthetic-q/0', 'changed-bank'):
            with self.subTest(bank=bank), self.assertRaises(UnknownProviderOutcome):
                self.resume(resumed_transport, {'question': 'changed body'}, bank=bank)
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(resumed_transport.calls, [])

    def test_response_received_is_unresolved_before_ledger_construction(self):
        for state in ('pending', 'response_received'):
            with self.subTest(state=state):
                save(self.root / 'requests' / 'old.json', {'key': 'old', 'state': state})
                with self.assertRaises(UnknownProviderOutcome):
                    check_request_recovery(self.root)
                self.assertFalse((self.root / 'ledger.jsonl').exists())

    def test_settled_cache_without_ledger_cannot_be_a_fresh_budget(self):
        save(self.root / 'requests' / 'old.json', {'key': 'old', 'state': 'settled'})
        with self.assertRaisesRegex(HostError, 'no complete ledger'):
            check_request_recovery(self.root)

    def test_shared_exact_body_has_two_logical_calls_but_one_purchase(self):
        task, _ = adapt_multihop({'id': 'synthetic-q', 'query': 'Synthetic question?', 'answer': 'Synthetic Port'})
        transport = Transport(); ledger = Ledger(self.root / 'ledger.jsonl', LIMITS)
        hosts=[]; models=[]
        for _ in range(2):
            model=self.model(transport, ledger=ledger)
            host=HostBroker(task, LocalCorpus([]), model)
            host('complete', {'stage': 'answer', 'payload': {'evidence': []}})
            hosts.append(host); models.append(model)
        self.assertEqual(sum(h.counts['model_calls'] for h in hosts), 2)
        self.assertEqual([m.calls for m in models], [1, 0])
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(ledger.summary()['used']['run']['calls'], 1)
        records=check_request_recovery(self.root)
        check_request_accounting(records, ledger)
        self.assertEqual(len(records), 1)

    def test_equal_answers_with_different_bodies_are_not_cache_equivalent(self):
        transport, _ = self.settled()
        result = self.resume(transport, {'question': 'Different synthetic question', 'evidence': []})
        self.assertEqual(result['answer'], 'Synthetic Port')
        self.assertEqual(len(transport.calls), 2)
        ledger=Ledger(self.root / 'ledger.jsonl', LIMITS)
        check_request_accounting(check_request_recovery(self.root), ledger)
        self.assertEqual(ledger.summary()['used']['run']['calls'], 2)

    def test_repeat_bank_prevents_sharing_across_repeats(self):
        transport, _ = self.settled()
        self.resume(transport, {'question': 'Synthetic question', 'evidence': []}, bank='calibration/synthetic-q/1')
        self.assertEqual(len(transport.calls), 2)

    def test_unknown_run_status_blocks_even_with_all_requests_settled(self):
        _, model = self.settled()
        save(self.root / 'live_status.json', {'status': 'stopped', 'reason_type': 'UnknownProviderOutcome'})
        with self.assertRaises(UnknownProviderOutcome): check_request_recovery(self.root)
        # Calibration uses its own status file, without inheriting live naming.
        self.assertEqual(len(check_request_recovery(self.root, status_filename='progress.json')), 1)
        save(self.root / 'progress.json', {'status': 'stopped', 'reason_type': 'UnknownProviderOutcome'})
        with self.assertRaises(UnknownProviderOutcome):
            check_request_recovery(self.root, status_filename='progress.json')
        self.assertEqual(model.ledger.summary()['pending'], 0)

    def test_cache_body_key_and_reservation_must_match_ledger(self):
        _, model = self.settled()
        records = check_request_recovery(self.root)
        key=next(iter(records))
        for problem in ('key', 'body', 'reservation', 'missing_response', 'extra_cache', 'missing_cache'):
            with self.subTest(problem=problem):
                changed=deepcopy(records)
                if problem=='key': changed[key]['key']='another-key'
                elif problem=='body': changed[key]['body']['model']='another-model'
                elif problem=='reservation': changed[key]['reservation']='another-reservation'
                elif problem=='missing_response': changed[key].pop('response')
                elif problem=='extra_cache': changed['extra']=deepcopy(changed[key])
                else: changed={}
                with self.assertRaises(HostError): check_request_accounting(changed,model.ledger)

    def test_truncated_ledger_cannot_turn_existing_cache_into_free_requests(self):
        self.settled()
        records=check_request_recovery(self.root)
        (self.root / 'ledger.jsonl').write_text('', encoding='utf-8')
        ledger=Ledger(self.root / 'ledger.jsonl', LIMITS)
        with self.assertRaisesRegex(HostError, 'complete ledger differ'):
            check_request_accounting(records,ledger)

    def test_unreadable_or_unknown_cache_state_is_host_failure(self):
        path=self.root / 'requests' / 'bad.json'
        for value in ('{', json.dumps({'state': 'unrecognized'})):
            with self.subTest(value=value):
                path.parent.mkdir(exist_ok=True)
                path.write_text(value,encoding='utf-8')
                with self.assertRaises(HostError): check_request_recovery(self.root)

    def test_helper_is_in_frozen_runtime_source_set(self):
        from code_rsi.v3.evolution import _runtime_source_hashes
        self.assertIn(str(Path('v3') / 'request_recovery.py'), _runtime_source_hashes())


if __name__ == '__main__':
    unittest.main()
