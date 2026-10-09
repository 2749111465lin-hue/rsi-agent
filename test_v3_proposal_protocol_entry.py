"""Exact-edit provider/host integration with synthetic responses and execution."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from code_rsi import live_evolution as live, paired_evolution as paired
from code_rsi.budget import Ledger, digest, save, stable
from code_rsi.v3 import evolution, execution
from code_rsi.v3.infrastructure import PROMPTS, StructuredModel
from code_rsi.v3.proposal_protocol import source_identity
from test_v3_evolution_recovery import InjectedCrash
import test_v3_controlled_developer as developer_fixture
import test_v3_edit_policy_entry as policy_fixture
import test_v3_live_evolution as live_fixture
import test_v3_live_phases as phase_fixture
import test_v3_paired_entry as paired_fixture


EXACT = {"schema": "rag-rsi-proposal-protocol-1", "format": "exact_edits",
         "max_edits": 8, "max_edit_chars": 12000}
WHOLE = {"schema": "rag-rsi-proposal-protocol-1", "format": "whole_files"}


def exact_response(payload, mutation="allowed"):
    files = payload["source_files"]
    old = next(line for line in files["rag.py"].splitlines() if line.startswith("CONFIG = "))
    def change(config):
        config.setdefault("prompts", {})["answer"] = "Verify relation direction before composing the supported short response."
    changed = policy_fixture.edited_wrapper(files["rag.py"], change)
    new = next(line for line in changed.splitlines() if line.startswith("CONFIG = "))
    result = {"parent_source_sha256": source_identity(files),
              "change_status": "modified",
              "edits": [{"file": "rag.py", "old": old, "new": new}],
              "mechanism": "Synthetic local prompt improvement",
              "intended_target_module": payload["decision"]["intended_target_module"]}
    if mutation == "wrong_parent":
        result["parent_source_sha256"] = "0" * 64
    elif mutation == "missing_anchor":
        result["edits"][0]["old"] = "SYNTHETIC_ANCHOR_NOT_PRESENT_IN_PARENT"
    elif mutation == "ambiguous_anchor":
        result["edits"][0]["old"] = "self.services"
    elif mutation == "overlap":
        result["edits"].append({"file": "rag.py", "old": "CONFIG = ", "new": "CONFIG  = "})
    elif mutation == "no_change":
        result["edits"][0]["new"] = old
    elif mutation == "cancelled_edits":
        result["edits"] = [{"file":"rag.py", "old":"CONFIG =", "new":"CONFIG "},
                           {"file":"rag.py", "old":" json.loads(", "new":"= json.loads("}]
    elif mutation == "declared_no_change":
        result["change_status"] = "no_change"
        result["edits"] = []
    elif mutation == "no_change_with_edits":
        result["change_status"] = "no_change"
    elif mutation == "cosmetic":
        result["edits"][0]["new"] = old + "  "
    elif mutation == "policy_violation":
        anchor = "def solve(question, services):"
        result["edits"] = [{"file": "rag.py", "old": anchor,
                            "new": "SYNTHETIC_SCOPE_CHANGE = 1\n" + anchor}]
    elif mutation == "unknown_file":
        result["edits"][0]["file"] = "host.py"
    elif mutation == "wrong_format":
        return policy_fixture.proposal_for(payload, "allowed")
    return result


class ExactTransport(live_fixture.FixtureTransport):
    def __init__(self, mutation="allowed"):
        super().__init__()
        self.mutation = mutation
        self.bodies = []

    def send(self, body, timeout):
        self.bodies.append(deepcopy(body))
        payload = json.loads(body["messages"][-1]["content"])
        if "source_files" not in payload:
            return super().send(body, timeout)
        self.requests.append(("develop", payload))
        result = exact_response(payload, self.mutation)
        return {"model": "synthetic-not-a-provider",
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 100}}


class ProposalProtocolEntryTests(unittest.TestCase):
    setUp = live_fixture.LiveEvolutionTests.setUp
    binding = live_fixture.LiveEvolutionTests.binding
    make_plan = live_fixture.LiveEvolutionTests.make_plan
    read_role = live_fixture.LiveEvolutionTests.read_role
    rewrite_role = live_fixture.LiveEvolutionTests.rewrite_role
    staged = phase_fixture.LivePhaseTests.staged
    parsed_roles = phase_fixture.LivePhaseTests.parsed_roles
    execute_counted = policy_fixture.EditPolicyEntryTests.execute_counted
    run_search = policy_fixture.EditPolicyEntryTests.run_search
    out = policy_fixture.EditPolicyEntryTests.out
    artifact = policy_fixture.EditPolicyEntryTests.artifact
    crash_after_accepted_proposal = policy_fixture.EditPolicyEntryTests.crash_after_accepted_proposal

    def configured(self, *, protocol=None, policy=None, expansions=1):
        policy_fixture.EditPolicyEntryTests.configured(self, policy=policy, expansions=expansions)
        self.plan["schema"] = live.SCHEMA5
        self.plan["proposal_protocol"] = deepcopy(EXACT if protocol is None else protocol)
        return self.plan

    def crash_after_rejection(self, transport):
        with patch.object(evolution.EvolutionRunner, "_terminal_attempt",
                          side_effect=InjectedCrash("rejection durable before terminal receipt")):
            with self.assertRaises(InjectedCrash):
                self.run_search(transport)
        self.assertTrue((self.out / "steps/0/received_proposal.json").is_file())
        self.assertTrue((self.out / "steps/0/rejected.json").is_file())
        self.assertFalse((self.out / "phase_search.json").exists())

    def test_old_schema_field_sets_cannot_silently_accept_protocol(self):
        legacy = deepcopy(self.plan)
        variants = [(legacy, None)]
        second = deepcopy(legacy); second["schema"] = live.SCHEMA2
        second["controls"] = {"parent_policy": "adaptive", "module_policy": "round_robin_v1",
                              "fixed_module": None, "memory": "none", "feedback": "rich", "case_schedule": []}
        variants.append((second, None))
        third = deepcopy(self.staged(fit_only=True)); variants.append((third, "search"))
        fourth = deepcopy(third); fourth["schema"] = live.SCHEMA4
        fourth["edit_policy"] = deepcopy(policy_fixture.PROMPT_POLICY); variants.append((fourth, "search"))
        for plan, phase in variants:
            plan["proposal_protocol"] = deepcopy(EXACT)
            with self.subTest(schema=plan["schema"]), self.assertRaisesRegex(ValueError, "exact"):
                live.preflight(plan, phase=phase)
        self.assertFalse(Path(legacy["output_dir"]).exists())

    def test_schema5_requires_edit_policy_protocol_and_explicit_phase(self):
        self.configured()
        for key in ("edit_policy", "proposal_protocol"):
            value = deepcopy(self.plan); value.pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                live.preflight(value, phase="search")
        for protocol in (None, {}, {**WHOLE, "max_edits": 8}, {**EXACT, "extra": True},
                         {**EXACT, "max_edits": True}, {**EXACT, "max_edits": 0},
                         {**EXACT, "max_edit_chars": 0}):
            value = deepcopy(self.plan); value["proposal_protocol"] = protocol
            with self.subTest(protocol=protocol), self.assertRaises(ValueError):
                live.preflight(value, phase="search")
        with self.assertRaisesRegex(ValueError, "explicit declared phase"):
            live.preflight(self.plan)
        self.assertFalse(self.out.exists())

    def test_preflight_binds_contract_without_reference_unlock_or_dispatch(self):
        self.configured()
        before = sorted(str(p) for p in self.base.rglob("*"))
        with patch.object(live, "credential_from_plan", side_effect=AssertionError("credential")), \
             patch.object(live, "deepseek_transport", side_effect=AssertionError("transport")):
            result, parsed = self.parsed_roles(lambda: live.preflight(self.plan, phase="search"))
        self.assertEqual(parsed, [])
        self.assertEqual(result["proposal_protocol"], EXACT)
        self.assertEqual(result["new_api_calls"], 0)
        self.assertEqual(before, sorted(str(p) for p in self.base.rglob("*")))
        result["proposal_protocol"]["max_edits"] = 1
        self.assertEqual(self.plan["proposal_protocol"], EXACT)

    def test_exact_provider_prompt_parent_binding_materialization_then_one_child(self):
        self.configured()
        transport = ExactTransport()
        result = self.run_search(transport)
        self.assertEqual(result["status"], "search_frozen")
        self.assertEqual(len(self.executed), 2)
        self.assertEqual([s for s, _ in transport.requests], ["answer", "develop", "answer"])
        payload = next(p for s, p in transport.requests if s == "develop")
        self.assertEqual(payload["proposal_protocol"], EXACT)
        self.assertEqual(payload["parent_source_sha256"], source_identity(payload["source_files"]))
        body = next(b for b in transport.bodies if "source_files" in json.loads(b["messages"][-1]["content"]))
        self.assertNotEqual(body["messages"][0]["content"], PROMPTS["develop"])
        self.assertIn("parent_source_sha256", body["messages"][0]["content"])
        self.assertIn("edits", body["messages"][0]["content"])
        self.assertNotIn("Return complete changed files", payload["edit_boundary"])
        raw = self.artifact("steps/0/received_proposal.json")
        self.assertIn("edits", raw); self.assertNotIn("writes", raw)
        normalized = self.artifact("steps/0/proposal.json")
        self.assertIn("writes", normalized); self.assertNotIn("edits", normalized)
        expected = deepcopy(payload["source_files"])
        for edit in raw["edits"]:
            expected[edit["file"]] = expected[edit["file"]].replace(edit["old"], edit["new"], 1)
        self.assertEqual({**payload["source_files"], **normalized["writes"]}, expected)
        materialized = self.artifact("steps/0/materialization.json")
        self.assertIsInstance(materialized, dict)
        self.assertNotIn(raw["edits"][0]["old"], stable(materialized))
        self.assertNotIn(raw["edits"][0]["new"], stable(materialized))
        self.assertTrue(self.artifact("steps/0/edit_policy.json")["allowed"])
        change = self.artifact("steps/0/change_receipt.json")
        self.assertEqual(change["declared_change_status"], "modified")
        self.assertTrue(change["source_changed"] and change["source_valid"] and change["ast_changed"])
        self.assertTrue(change["passes_source_change_gate"])
        archive = evolution.ProgramArchive(self.out / "archive")
        parent_node = self.artifact("root.json")
        child_node = self.artifact("steps/0/child.json")
        self.assertNotEqual(parent_node["program_id"], child_node["program_id"])
        self.assertEqual(archive.load_program(parent_node["program_id"])["files"], payload["source_files"])
        self.assertEqual(archive.load_program(child_node["program_id"])["files"], expected)
        self.assertEqual(change["parent_source_sha256"], source_identity(payload["source_files"]))
        self.assertEqual(change["child_source_sha256"], source_identity(expected))
        manifest = self.artifact("manifest.json")
        self.assertEqual(manifest["proposal_protocol"], EXACT)
        self.assertEqual(manifest["developer_configuration"]["proposal_protocol"], EXACT)
        self.assertFalse((self.out / "delivery_lock.json").exists())

    def test_bad_exact_edits_are_preserved_rejected_and_counted_without_candidate_execution(self):
        for mutation in ("wrong_parent", "missing_anchor", "ambiguous_anchor", "overlap", "unknown_file", "wrong_format", "no_change_with_edits", "no_change"):
            with self.subTest(mutation=mutation):
                self.configured(expansions=2)
                self.plan["output_dir"] = str(self.base / mutation)
                transport = ExactTransport(mutation)
                result = self.run_search(transport)
                self.assertEqual(len(self.executed), 1)
                self.assertEqual(len(transport.requests), 3)
                self.assertEqual(len(result["search"]["terminal_attempts"]), 2)
                self.assertTrue(all(a["status"] == "rejected" for a in result["search"]["terminal_attempts"]))
                for slot in range(2):
                    raw = self.artifact(f"steps/{slot}/received_proposal.json")
                    rejected = self.artifact(f"steps/{slot}/rejected.json")
                    self.assertEqual(rejected["received_proposal_sha256"], digest(raw))
                    self.assertFalse((self.out / f"steps/{slot}/materialization.json").exists())
                    self.assertFalse((self.out / f"steps/{slot}/child.json").exists())

    def test_materialized_no_ast_change_still_rejects_without_child(self):
        for mutation in ("cancelled_edits", "cosmetic"):
            with self.subTest(mutation=mutation):
                self.configured(); self.plan["output_dir"] = str(self.base / mutation)
                transport = ExactTransport(mutation)
                result = self.run_search(transport)
                self.assertEqual(len(self.executed), 1)
                self.assertEqual(len(transport.requests), 2)
                self.assertTrue((self.out / "steps/0/materialization.json").exists())
                change = self.artifact("steps/0/change_receipt.json")
                self.assertEqual(change["declared_change_status"], "modified")
                self.assertFalse(change["ast_changed"])
                self.assertFalse(change["passes_source_change_gate"])
                self.assertEqual(change["source_changed"], mutation == "cosmetic")
                self.assertIn("no executable behavior", self.artifact("steps/0/rejected.json")["reason"])
                self.assertEqual(result["search"]["terminal_attempts"][0]["status"], "rejected")

    def test_explicit_no_change_preserves_parent_and_consumes_one_rejection_opportunity(self):
        self.configured()
        transport = ExactTransport("declared_no_change")
        result = self.run_search(transport)
        self.assertEqual(len(self.executed), 1)
        self.assertEqual(len(transport.requests), 2)
        self.assertEqual(len(result["search"]["cards"]), 1)
        self.assertEqual(self.artifact("steps/0/rejected.json")["reason"], "model declared no change")
        materialized = self.artifact("steps/0/materialization.json")
        self.assertEqual(materialized["declared_change_status"], "no_change")
        self.assertFalse(materialized["source_changed"])
        change = self.artifact("steps/0/change_receipt.json")
        self.assertFalse(change["source_changed"] or change["ast_changed"] or change["passes_source_change_gate"])
        self.assertEqual(change["parent_source_sha256"], change["child_source_sha256"])
        self.assertFalse((self.out / "steps/0/child.json").exists())
        self.assertEqual(result["search"]["terminal_attempts"][0]["status"], "rejected")

    def test_materialized_scope_violation_still_hits_independent_edit_policy(self):
        self.configured()
        transport = ExactTransport("policy_violation")
        result = self.run_search(transport)
        self.assertEqual(len(self.executed), 1)
        self.assertTrue((self.out / "steps/0/materialization.json").exists())
        self.assertFalse(self.artifact("steps/0/edit_policy.json")["allowed"])
        self.assertEqual(self.artifact("steps/0/rejected.json")["reason_code"], "edit_policy_violation")
        self.assertEqual(result["search"]["terminal_attempts"][0]["status"], "rejected")

    def test_complete_resume_has_no_new_call_execution_or_reference_unlock(self):
        self.configured()
        transport = ExactTransport()
        first = self.run_search(transport)
        before = (len(self.executed), len(transport.requests))
        def forbidden(): raise AssertionError("completed exact run built transport")
        second, parsed = self.parsed_roles(lambda: self.run_search(factory=forbidden))
        self.assertEqual(first, second)
        self.assertEqual(parsed, [])
        self.assertEqual(before, (len(self.executed), len(transport.requests)))

    def test_accepted_checkpoint_reuses_develop_and_executes_only_unfinished_child(self):
        self.configured()
        transport = ExactTransport()
        self.crash_after_accepted_proposal(transport)
        self.assertTrue((self.out / "steps/0/materialization.json").exists())
        self.assertEqual(len(self.executed), 1)
        self.run_search(transport)
        self.assertEqual(len(self.executed), 2)
        self.assertEqual([s for s, _ in transport.requests], ["answer", "develop", "answer"])

    def test_accepted_checkpoint_missing_materialization_fails_without_dispatch(self):
        self.configured()
        transport = ExactTransport()
        self.crash_after_accepted_proposal(transport)
        (self.out / "steps/0/materialization.json").unlink()
        before = (len(self.executed), len(transport.requests))
        with self.assertRaises(RuntimeError):
            self.run_search(transport)
        self.assertEqual(before, (len(self.executed), len(transport.requests)))

    def test_accepted_checkpoint_tampered_materialization_fails_without_dispatch(self):
        self.configured()
        transport = ExactTransport()
        self.crash_after_accepted_proposal(transport)
        receipt = self.artifact("steps/0/materialization.json"); receipt["tampered"] = True
        save(self.out / "steps/0/materialization.json", receipt)
        before = (len(self.executed), len(transport.requests))
        with self.assertRaises(RuntimeError):
            self.run_search(transport)
        self.assertEqual(before, (len(self.executed), len(transport.requests)))

    def test_received_patch_change_cannot_reuse_accepted_whole_proposal(self):
        self.configured()
        transport = ExactTransport()
        self.crash_after_accepted_proposal(transport)
        raw = self.artifact("steps/0/received_proposal.json")
        raw["edits"][0]["new"] += "\nSYNTHETIC_LATE_EDIT = 1\n"
        save(self.out / "steps/0/received_proposal.json", raw)
        before = (len(self.executed), len(transport.requests))
        with self.assertRaises(RuntimeError):
            self.run_search(transport)
        self.assertEqual(before, (len(self.executed), len(transport.requests)))

    def test_accepted_checkpoint_cannot_omit_source_change_receipt(self):
        self.configured()
        transport = ExactTransport()
        self.crash_after_accepted_proposal(transport)
        (self.out / "steps/0/change_receipt.json").unlink()
        before = (len(self.executed), len(transport.requests))
        with self.assertRaises(RuntimeError):
            self.run_search(transport)
        self.assertEqual(before, (len(self.executed), len(transport.requests)))

    def test_rejected_change_gate_receipt_cannot_be_flipped_on_resume(self):
        self.configured()
        transport = ExactTransport("cosmetic")
        self.crash_after_rejection(transport)
        value = self.artifact("steps/0/change_receipt.json")
        value["ast_changed"] = value["passes_source_change_gate"] = True
        save(self.out / "steps/0/change_receipt.json", value)
        before = (len(self.executed), len(transport.requests))
        with self.assertRaises(RuntimeError):
            self.run_search(transport)
        self.assertEqual(before, (len(self.executed), len(transport.requests)))

    def test_bad_patch_rejection_resume_requires_same_raw_hash_and_never_rebuys(self):
        self.configured()
        transport = ExactTransport("wrong_parent")
        self.crash_after_rejection(transport)
        before = (len(self.executed), len(transport.requests))
        result = self.run_search(transport)
        self.assertEqual(result["search"]["terminal_attempts"][0]["status"], "rejected")
        self.assertEqual(before, (len(self.executed), len(transport.requests)))

    def test_changed_raw_rejection_cannot_be_accepted_on_resume(self):
        self.configured()
        transport = ExactTransport("wrong_parent")
        self.crash_after_rejection(transport)
        payload = next(p for s, p in transport.requests if s == "develop")
        save(self.out / "steps/0/received_proposal.json", exact_response(payload))
        before = (len(self.executed), len(transport.requests))
        with self.assertRaises(ValueError):
            self.run_search(transport)
        self.assertEqual(before, (len(self.executed), len(transport.requests)))

    def test_ast_rejection_cannot_resume_without_existing_materialization(self):
        self.configured()
        transport = ExactTransport("cosmetic")
        self.crash_after_rejection(transport)
        (self.out / "steps/0/materialization.json").unlink()
        before = (len(self.executed), len(transport.requests))
        with self.assertRaises(RuntimeError):
            self.run_search(transport)
        self.assertEqual(before, (len(self.executed), len(transport.requests)))

    def test_protocol_change_rejects_old_approval_and_existing_run(self):
        self.configured()
        transport = ExactTransport()
        approved = digest(self.plan)
        self.run_search(transport)
        self.plan["proposal_protocol"]["max_edits"] = 1
        with self.assertRaisesRegex(ValueError, "exact approved"):
            live.run(self.plan, approved_plan_hash=approved, execute=True,
                     phase="search", transport_factory=lambda: transport)
        before = (len(self.executed), len(transport.requests))
        with self.assertRaises(ValueError):
            self.run_search(transport)
        self.assertEqual(before, (len(self.executed), len(transport.requests)))

    def test_explicit_whole_files_keeps_old_prompt_and_schema4_keeps_old_fields(self):
        self.configured(protocol=WHOLE)
        transport = policy_fixture.PolicyTransport()
        self.run_search(transport)
        body = next(b for b in transport.bodies if "source_files" in json.loads(b["messages"][-1]["content"]))
        self.assertEqual(body["messages"][0]["content"], PROMPTS["develop"])
        self.assertEqual(len(self.executed), 2)
        self.plan["schema"] = live.SCHEMA4; self.plan.pop("proposal_protocol")
        self.plan["output_dir"] = str(self.base / "legacy-schema4")
        legacy = policy_fixture.PolicyTransport()
        self.run_search(legacy)
        self.assertNotIn("proposal_protocol", self.artifact("manifest.json"))
        self.assertNotIn("proposal_protocol", next(p for s, p in legacy.requests if s == "develop"))
        self.assertFalse((self.out / "steps/0/materialization.json").exists())


class DeveloperProtocolPayloadTests(unittest.TestCase):
    setUp = developer_fixture.ControlledDeveloperTests.setUp
    prepare = developer_fixture.ControlledDeveloperTests.prepare
    propose = developer_fixture.ControlledDeveloperTests.propose
    factory = developer_fixture.ControlledDeveloperTests.factory

    def model(self, bank, *, protocol=EXACT):
        return StructuredModel(self.root / 'requests', self.ledger, self.transport, bank=bank,
                               prices={'input_miss':2,'input_hit':.04,'output':8},
                               proposal_protocol=deepcopy(protocol))

    def transport(self, body):
        self.sent.append(deepcopy(body))
        payload = json.loads(body['messages'][-1]['content'])
        result = exact_response(payload, getattr(self, 'mutation', 'allowed'))
        return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(result)}}],
                'usage':{'prompt_tokens':100,'completion_tokens':100}}

    def developer(self):
        return evolution.ProgramDeveloper(self.projection, feedback_condition='cases',
            case_schedule=self.schedule, proposal_model_factory=self.factory,
            edit_policy=deepcopy(policy_fixture.PROMPT_POLICY), proposal_protocol=deepcopy(EXACT))

    def test_exact_developer_includes_complete_source_identity_and_protocol_snapshot(self):
        developer = self.developer()
        payload = self.prepare(developer)
        self.assertEqual(payload['source_files'], self.program['files'])
        self.assertEqual(payload['parent_source_sha256'], source_identity(self.program['files']))
        self.assertEqual(payload['proposal_protocol'], EXACT)
        snapshot = developer.configuration_snapshot()
        self.assertEqual(snapshot['proposal_protocol'], EXACT)
        snapshot['proposal_protocol']['max_edits'] = 1
        self.assertEqual(developer.configuration_snapshot()['proposal_protocol'], EXACT)
        self.assertEqual(self.sent, []); self.assertEqual(self.factory_slots, [])

    def test_bad_parent_hash_response_reaches_host_without_developer_pre_rejection(self):
        self.mutation = 'wrong_parent'
        raw = self.propose(self.developer())
        self.assertEqual(raw['parent_source_sha256'], '0' * 64)
        self.assertEqual(len(self.sent), 1)

    def test_exact_model_contract_changes_develop_only_and_whole_keeps_default_prompt(self):
        old = self.model('old', protocol=None)
        whole = self.model('whole', protocol=WHOLE)
        exact = self.model('exact')
        self.assertEqual(old.request_body('develop', {})['messages'][0]['content'], PROMPTS['develop'])
        self.assertEqual(whole.request_body('develop', {'proposal_protocol':WHOLE})['messages'][0]['content'], PROMPTS['develop'])
        self.assertNotEqual(exact.request_body('develop', {'proposal_protocol':EXACT})['messages'][0]['content'], PROMPTS['develop'])
        with self.assertRaises(ValueError):
            exact.request_body('develop', {})
        for stage in ('plan','read','answer'):
            self.assertEqual(old.request_body(stage, {}), exact.request_body(stage, {}))
        self.assertNotEqual(old.identity, exact.identity)

    def test_legacy_request_shape_without_protocol_keeps_exact_provider_bytes(self):
        legacy = self.model('legacy-shape-comparison', protocol=None)
        shape = object.__new__(StructuredModel)
        shape.model = legacy.model
        shape.limits = deepcopy(legacy.limits)
        self.assertFalse(hasattr(shape, 'proposal_protocol'))
        payload = {'question': 'Synthetic request only.', 'note': '\u4e00\u81f4\u6027'}
        for stage in ('plan', 'read', 'answer', 'develop'):
            with self.subTest(stage=stage):
                expected = {
                    'model': legacy.model, 'stream': False,
                    'thinking': {'type': 'disabled'}, 'temperature': 0,
                    'max_tokens': legacy.limits[stage],
                    'response_format': {'type': 'json_object'},
                    'messages': [
                        {'role': 'system', 'content': PROMPTS[stage]},
                        {'role': 'user', 'content': stable(payload)}],
                }
                expected_bytes = stable(expected).encode('utf-8')
                self.assertEqual(stable(shape.request_body(stage, payload)).encode('utf-8'), expected_bytes)
                self.assertEqual(stable(legacy.request_body(stage, payload)).encode('utf-8'), expected_bytes)
                self.assertEqual(shape.request_size(stage, payload), len(expected_bytes))
        self.assertFalse(hasattr(shape, 'proposal_protocol'))
        self.assertEqual(self.sent, [])
        self.assertEqual(self.ledger.events, [])

    def test_projection_and_actual_factory_cannot_disagree_on_protocol(self):
        developer = self.developer()
        developer.proposal_model_factory = lambda slot: self.model('wrong-protocol', protocol=None)
        with self.assertRaises(ValueError):
            self.propose(developer)
        self.assertEqual(self.sent, [])

    def test_full_protocol_body_remains_inside_strict_request_budget(self):
        developer = self.developer()
        payload = self.prepare(developer)
        self.projection.max_input_bytes = self.projection.request_size('develop', payload) - 1
        with patch.object(evolution, '_fit_development_request', side_effect=AssertionError('crop')), \
             self.assertRaises(ValueError):
            self.propose(developer)
        self.assertEqual(self.sent, []); self.assertEqual(self.factory_slots, [])


class PairedSequenceTransport(ExactTransport):
    """One accepted local edit followed by an explicit no_change in each arm."""
    def __init__(self):
        super().__init__()
        self.arm_counts = {}

    def send(self, body, timeout):
        payload = json.loads(body['messages'][-1]['content'])
        if 'source_files' in payload:
            condition = payload['feedback']['condition']
            count = self.arm_counts.get(condition, 0)
            self.mutation = 'allowed' if count == 0 else 'declared_no_change'
            self.arm_counts[condition] = count + 1
        return super().send(body, timeout)


class PairedProtocolEntryTests(unittest.TestCase):
    binding = paired_fixture.PairedEntryTests.binding
    make_plan = paired_fixture.PairedEntryTests.make_plan
    read_role = paired_fixture.PairedEntryTests.read_role
    rewrite_role = paired_fixture.PairedEntryTests.rewrite_role
    staged = paired_fixture.PairedEntryTests.staged
    run_fixture = paired_fixture.PairedEntryTests.run_fixture

    def setUp(self):
        paired_fixture.PairedEntryTests.setUp(self)
        self.pair['schema'] = paired.SCHEMA2
        self.pair['blocks'] = 1
        self.pair['max_calls'] = 39
        self.plan['schema'] = live.SCHEMA5
        self.plan['edit_policy'] = deepcopy(policy_fixture.PROMPT_POLICY)
        self.plan['proposal_protocol'] = deepcopy(EXACT)

    def test_paired_versions_require_matching_template_and_static_preflight(self):
        original = live._file
        parsed = []
        binding = self.plan['panels']['D_fit']['references_file']
        def observe(item, *, parse=False):
            if item == binding and parse:
                parsed.append(True)
            return original(item, parse=parse)
        with patch.object(live, '_file', side_effect=observe), \
             patch.object(live, 'credential_from_plan', side_effect=AssertionError('credential')):
            report = paired.preflight(self.pair)
        self.assertEqual(report['schema'], paired.SCHEMA2)
        self.assertEqual(report['proposal_opportunities'], 4)
        self.assertEqual(report['max_calls'], 39)
        self.assertEqual(parsed, [])
        for outer, inner in ((paired.SCHEMA, live.SCHEMA5), (paired.SCHEMA2, live.SCHEMA3),
                             (paired.SCHEMA2, live.SCHEMA4)):
            candidate = deepcopy(self.pair)
            candidate['schema'] = outer; candidate['search_template']['schema'] = inner
            with self.subTest(outer=outer, inner=inner), self.assertRaises(ValueError):
                paired.preflight(candidate)
        self.assertFalse(self.out.exists())

    def test_paired_exact_gates_shared_root_and_mixed_outcomes_resume_without_rebuy(self):
        transport = PairedSequenceTransport()
        result, execute_count = self.run_fixture(transport)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['schema'], paired.SCHEMA2)
        self.assertEqual(result['completed_opportunities'], 4)
        self.assertEqual(execute_count, 3)  # One shared root, one independent child per arm.
        self.assertEqual(len(transport.requests), 7)
        self.assertEqual(result['ledger']['used']['run']['calls'], 7)
        self.assertEqual(result['ledger']['used']['root:0']['calls'], 1)
        statuses = [r['attempt']['status'] for r in result['terminal_records']]
        self.assertEqual(statuses.count('measured'), 2)
        self.assertEqual(statuses.count('rejected'), 2)
        payloads, roots = {}, {}
        for condition in ('cases', 'trace'):
            directory = self.out / 'blocks/0' / condition
            roots[condition] = json.loads((directory / 'measurements/shared_root.json').read_bytes())['result']
            body = json.loads((self.out / 'blocks/0/feedback' / (condition + '.json')).read_bytes())
            payloads[condition] = json.loads(body['messages'][-1]['content'])
            self.assertEqual(payloads[condition]['proposal_protocol'], EXACT)
            self.assertEqual(payloads[condition]['parent_source_sha256'], source_identity(payloads[condition]['source_files']))
            for actual in transport.bodies:
                contents = json.loads(actual['messages'][-1]['content'])
                if 'source_files' in contents and contents['feedback']['condition'] == condition:
                    self.assertEqual(actual, body)
            rejection = json.loads((directory / 'steps/1/rejected.json').read_bytes())
            self.assertEqual(rejection['reason'], 'model declared no change')
            self.assertTrue((directory / 'steps/0/child.json').exists())
            self.assertFalse((directory / 'steps/1/child.json').exists())
            self.assertTrue((directory / 'phase_search.json').exists())
            self.assertFalse((directory / 'delivery_lock.json').exists())
            self.assertFalse((directory / 'report.json').exists())
        self.assertEqual(roots['cases'], roots['trace'])
        normalized = deepcopy(payloads['trace'])
        normalized['feedback']['condition'] = 'cases'
        for case in normalized['feedback']['cases']:
            case.pop('execution_flow', None); case.pop('diagnostics', None)
        self.assertEqual(normalized, payloads['cases'])
        def forbidden(): raise AssertionError('paired completion rebuilt transport')
        resumed, execute_count = self.run_fixture(transport_factory=forbidden)
        self.assertEqual(resumed, result)
        self.assertEqual(execute_count, 0)
        self.assertEqual(len(transport.requests), 7)

    def test_paired_exact_partial_resume_keeps_independent_slots_and_fixed_count(self):
        transport = PairedSequenceTransport()
        first, count = self.run_fixture(transport, stop_after=1)
        self.assertEqual((first['completed_opportunities'], count), (1, 2))
        self.assertEqual(len(transport.requests), 3)
        def forbidden(): raise AssertionError('saved paired prefix rebuilt transport')
        saved, count = self.run_fixture(transport_factory=forbidden, stop_after=0)
        self.assertEqual(saved['completed_opportunities'], 1)
        self.assertEqual(count, 0)
        complete, count = self.run_fixture(transport)
        self.assertEqual(complete['completed_opportunities'], 4)
        self.assertEqual(count, 1)
        self.assertEqual(len(transport.requests), 7)
        self.assertEqual(transport.arm_counts, {'cases':2, 'trace':2})


if __name__ == '__main__':
    unittest.main()
