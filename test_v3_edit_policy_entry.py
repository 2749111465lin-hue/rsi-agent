"""Synthetic edit-policy entry/recovery checks; no API or candidate execution."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from code_rsi import live_evolution as live
from code_rsi.budget import digest, save, stable
from code_rsi.v3 import evolution, execution
from code_rsi.v3.infrastructure import PROMPTS
from test_v3_evolution_recovery import InjectedCrash
import test_v3_controlled_developer as developer_fixture
import test_v3_evolution_recovery as recovery_fixture
import test_v3_live_evolution as live_fixture
import test_v3_live_phases as phase_fixture


PROMPT_POLICY = {"schema": "rag-rsi-edit-policy-1", "mode": "prompt_only",
                 "allowed_prompt_stages": ["answer"]}
PROGRAM_POLICY = {"schema": "rag-rsi-edit-policy-1", "mode": "program"}


def edited_wrapper(source, change):
    """Parse a trusted fixture wrapper; never execute proposed Python."""
    assignments = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in n.targets)]
    if len(assignments) != 1:
        raise AssertionError("fixture requires exactly one CONFIG")
    assignment = assignments[0]
    config = json.loads(ast.literal_eval(assignment.value.args[0]))
    change(config)
    lines = source.splitlines(keepends=True)
    replacement = "CONFIG = json.loads(" + repr(stable(config)) + ")\n"
    return "".join(lines[:assignment.lineno - 1]) + replacement + "".join(lines[assignment.end_lineno:])


def proposal_for(payload, mutation):
    source = payload["source_files"]
    if mutation == "core":
        writes = {"rag_core.py": source["rag_core.py"] + "\nSYNTHETIC_POLICY_EDIT = 1\n"}
    elif mutation == "wrapper":
        writes = {"rag.py": source["rag.py"] + "\nSYNTHETIC_WRAPPER_EDIT = 1\n"}
    else:
        def mutate(config):
            if mutation == "budget":
                config["max_rounds"] = config.get("max_rounds", 3) + 1
            else:
                stage = "read" if mutation == "other_prompt" else "answer"
                prompt = (payload["feedback"]["cases"][0]["question"] if mutation == "fit_literal" else
                          "Check each relationship before composing a concise supported response.")
                config.setdefault("prompts", {})[stage] = prompt
        writes = {"rag.py": edited_wrapper(source["rag.py"], mutate)}
    return {"writes": writes, "mechanism": "Synthetic reusable edit for boundary testing",
            "target_module": payload["decision"]["target_module"]}


class PolicyTransport(live_fixture.FixtureTransport):
    def __init__(self, mutation="allowed"):
        super().__init__()
        self.mutation = mutation
        self.bodies = []

    def send(self, body, timeout):
        self.bodies.append(deepcopy(body))
        stage = next(k for k, v in PROMPTS.items() if v == body["messages"][0]["content"])
        if stage != "develop":
            return super().send(body, timeout)
        payload = json.loads(body["messages"][1]["content"])
        self.requests.append((stage, payload))
        value = proposal_for(payload, self.mutation)
        return {"model": "synthetic-not-a-provider",
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 100}}


class EditPolicyEntryTests(unittest.TestCase):
    setUp = live_fixture.LiveEvolutionTests.setUp
    binding = live_fixture.LiveEvolutionTests.binding
    make_plan = live_fixture.LiveEvolutionTests.make_plan
    read_role = live_fixture.LiveEvolutionTests.read_role
    rewrite_role = live_fixture.LiveEvolutionTests.rewrite_role
    staged = phase_fixture.LivePhaseTests.staged
    parsed_roles = phase_fixture.LivePhaseTests.parsed_roles

    def configured(self, *, policy=None, expansions=1):
        self.staged(fit_only=True)
        self.plan["schema"] = live.SCHEMA4
        self.plan["edit_policy"] = deepcopy(PROMPT_POLICY if policy is None else policy)
        self.plan["expansions"] = expansions
        self.plan["max_calls"] = 7 * (expansions + 1) + expansions
        self.executed = []
        return self.plan

    def execute_counted(self, *args, **kwargs):
        self.executed.append(args[1])
        return live_fixture.fake_execute(*args, **kwargs)

    def run_search(self, transport=None, *, factory=None, phase="search"):
        if factory is None:
            factory = lambda: transport
        with patch.object(execution, "execute", side_effect=self.execute_counted):
            return live.run(self.plan, phase=phase, approved_plan_hash=digest(self.plan),
                            execute=True, transport_factory=factory)

    @property
    def out(self):
        return Path(self.plan["output_dir"])

    def artifact(self, relative):
        return json.loads((self.out / relative).read_bytes())

    def crash_after_rejection(self, transport):
        with patch.object(evolution.EvolutionRunner, "_terminal_attempt",
                          side_effect=InjectedCrash("after policy decision, before terminal receipt")):
            with self.assertRaises(InjectedCrash):
                self.run_search(transport)
        self.assertTrue((self.out / "steps/0/received_proposal.json").is_file())
        self.assertTrue((self.out / "steps/0/edit_policy.json").is_file())
        self.assertTrue((self.out / "steps/0/rejected.json").is_file())
        self.assertFalse((self.out / "phase_search.json").exists())

    def crash_after_accepted_proposal(self, transport):
        original = evolution.save
        def save_then_crash(path, value):
            original(path, value)
            if Path(path).name == "proposal.json":
                raise InjectedCrash("accepted proposal durable, before child execution")
        with patch.object(evolution, "save", side_effect=save_then_crash):
            with self.assertRaises(InjectedCrash):
                self.run_search(transport)
        self.assertTrue((self.out / "steps/0/proposal.json").is_file())
        self.assertFalse((self.out / "steps/0/child.json").exists())

    def test_old_schemas_cannot_silently_accept_policy_fields(self):
        legacy = deepcopy(self.plan)
        variants = [(legacy, None)]
        second = deepcopy(legacy); second["schema"] = live.SCHEMA2
        second["controls"] = {"parent_policy": "adaptive", "module_policy": "round_robin_v1",
                              "fixed_module": None, "memory": "none", "feedback": "rich", "case_schedule": []}
        variants.append((second, None))
        third = deepcopy(self.staged(fit_only=True)); variants.append((third, "search"))
        for plan, phase in variants:
            plan["edit_policy"] = deepcopy(PROMPT_POLICY)
            with self.subTest(schema=plan["schema"]), self.assertRaisesRegex(ValueError, "exact"):
                live.preflight(plan, phase=phase)
        self.assertFalse(Path(legacy["output_dir"]).exists())

    def test_schema4_requires_exact_policy_and_explicit_phase_before_io(self):
        self.configured()
        bad = [None, {}, {**PROGRAM_POLICY, "allowed_prompt_stages": ["answer"]},
               {**PROMPT_POLICY, "allowed_prompt_stages": []},
               {**PROMPT_POLICY, "allowed_prompt_stages": ["develop"]},
               {**PROMPT_POLICY, "allowed_prompt_stages": ["answer", "answer"]},
               {**PROMPT_POLICY, "extra": True}]
        for policy in bad:
            plan = deepcopy(self.plan); plan["edit_policy"] = policy
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                live.preflight(plan, phase="search")
        missing = deepcopy(self.plan); missing.pop("edit_policy")
        with self.assertRaisesRegex(ValueError, "exact"):
            live.preflight(missing, phase="search")
        with self.assertRaisesRegex(ValueError, "explicit declared phase"):
            live.preflight(self.plan)
        self.assertFalse(self.out.exists())

    def test_preflight_binds_policy_without_parsing_references_or_touching_transport(self):
        self.configured()
        before = sorted(str(p) for p in self.base.rglob("*"))
        with patch.object(live, "credential_from_plan", side_effect=AssertionError("credential")), \
             patch.object(live, "deepseek_transport", side_effect=AssertionError("transport")):
            report, parsed = self.parsed_roles(lambda: live.preflight(self.plan, phase="search"))
        self.assertEqual(parsed, [])
        self.assertEqual(report["edit_policy"], PROMPT_POLICY)
        self.assertEqual(report["new_api_calls"], 0)
        self.assertFalse(report["credentials_read"])
        self.assertEqual(before, sorted(str(p) for p in self.base.rglob("*")))
        report["edit_policy"]["allowed_prompt_stages"].append("read")
        self.assertEqual(self.plan["edit_policy"], PROMPT_POLICY)

    def test_allowed_prompt_reaches_one_child_with_frozen_policy_receipt(self):
        self.configured()
        transport = PolicyTransport()
        result = self.run_search(transport)
        self.assertEqual(result["status"], "search_frozen")
        self.assertEqual(len(self.executed), 2)
        self.assertEqual([s for s, _ in transport.requests], ["answer", "develop", "answer"])
        self.assertEqual(result["search"]["terminal_attempts"][0]["status"], "measured")
        receipt = self.artifact("steps/0/edit_policy.json")
        self.assertTrue(receipt["allowed"])
        self.assertIn("receipt_sha256", receipt)
        manifest = self.artifact("manifest.json")
        self.assertEqual(manifest["edit_policy"], PROMPT_POLICY)
        self.assertEqual(manifest["active_edit_policy"], PROMPT_POLICY)
        self.assertEqual(manifest["developer_configuration"]["edit_policy"], PROMPT_POLICY)
        develop = next(p for s, p in transport.requests if s == "develop")
        self.assertEqual(develop["edit_policy"], PROMPT_POLICY)
        self.assertNotIn("General refactors are allowed", develop["edit_boundary"])
        self.assertNotIn("PRIVATE_GOLD", stable(develop))
        self.assertFalse((self.out / "delivery_lock.json").exists())
        self.assertFalse((self.out / "report.json").exists())

    def test_out_of_scope_edits_are_rejected_before_child_execute_and_count_opportunities(self):
        for mutation in ("core", "wrapper", "budget", "other_prompt"):
            with self.subTest(mutation=mutation):
                self.configured(expansions=2)
                self.plan["output_dir"] = str(self.base / mutation)
                transport = PolicyTransport(mutation)
                result = self.run_search(transport)
                self.assertEqual(len(self.executed), 1)
                self.assertEqual(len(transport.requests), 3)  # One root, two independent proposals.
                self.assertEqual(len(result["search"]["cards"]), 1)
                attempts = result["search"]["terminal_attempts"]
                self.assertEqual(len(attempts), 2)
                self.assertTrue(all(a["status"] == "rejected" and a["node_id"] is None for a in attempts))
                for step in range(2):
                    rejection = self.artifact(f"steps/{step}/rejected.json")
                    receipt = self.artifact(f"steps/{step}/edit_policy.json")
                    self.assertEqual(rejection["reason_code"], "edit_policy_violation")
                    self.assertFalse(receipt["allowed"])
                    self.assertEqual(rejection["edit_policy_receipt_sha256"], receipt["receipt_sha256"])
                    self.assertTrue((self.out / f"steps/{step}/received_proposal.json").exists())
                    self.assertFalse((self.out / f"steps/{step}/child.json").exists())
                self.assertEqual(len(list((self.out / "archive/nodes").glob("*.json"))), 1)

    def test_program_policy_preserves_full_program_edit_and_legacy_schema_has_no_policy(self):
        self.configured(policy=PROGRAM_POLICY)
        transport = PolicyTransport("core")
        self.run_search(transport)
        self.assertEqual(len(self.executed), 2)
        self.assertTrue(self.artifact("steps/0/edit_policy.json")["allowed"])
        self.plan["output_dir"] = str(self.base / "legacy-staged")
        self.plan["schema"] = live.SCHEMA3; self.plan.pop("edit_policy")
        legacy = PolicyTransport("core")
        self.run_search(legacy)
        self.assertNotIn("edit_policy", self.artifact("manifest.json"))
        self.assertNotIn("active_edit_policy", self.artifact("manifest.json"))
        self.assertFalse((self.out / "steps/0/edit_policy.json").exists())
        self.assertNotIn("edit_policy", next(p for s, p in legacy.requests if s == "develop"))

    def test_complete_resume_has_no_execute_transport_or_private_unlock(self):
        self.configured()
        transport = PolicyTransport()
        first = self.run_search(transport)
        before = (len(self.executed), len(transport.requests))
        def forbidden(): raise AssertionError("completed phase constructed transport")
        second, parsed = self.parsed_roles(lambda: self.run_search(factory=forbidden))
        self.assertEqual(first, second)
        self.assertEqual(parsed, [])
        self.assertEqual(before, (len(self.executed), len(transport.requests)))

    def test_rejected_checkpoint_resume_rechecks_without_rebuying(self):
        self.configured()
        transport = PolicyTransport("core")
        self.crash_after_rejection(transport)
        before = (len(self.executed), len(transport.requests))
        with patch.object(evolution, "check_edit_policy", wraps=evolution.check_edit_policy) as check:
            result = self.run_search(transport)
        self.assertGreaterEqual(check.call_count, 1)
        self.assertEqual(before, (len(self.executed), len(transport.requests)))
        self.assertEqual(result["search"]["terminal_attempts"][0]["status"], "rejected")

    def test_missing_rejected_policy_receipt_fails_before_new_work(self):
        self.configured()
        transport = PolicyTransport("core")
        self.crash_after_rejection(transport)
        (self.out / "steps/0/edit_policy.json").unlink()
        before = (len(self.executed), len(transport.requests))
        with self.assertRaisesRegex(ValueError, "policy receipt"):
            self.run_search(transport)
        self.assertEqual(before, (len(self.executed), len(transport.requests)))

    def test_changed_rejected_policy_receipt_fails_before_new_work(self):
        self.configured()
        transport = PolicyTransport("core")
        self.crash_after_rejection(transport)
        value = self.artifact("steps/0/edit_policy.json"); value["allowed"] = True
        save(self.out / "steps/0/edit_policy.json", value)
        before = (len(self.executed), len(transport.requests))
        with self.assertRaisesRegex(ValueError, "frozen run artifact differs"):
            self.run_search(transport)
        self.assertEqual(before, (len(self.executed), len(transport.requests)))

    def test_changed_received_source_cannot_reuse_rejection_receipt(self):
        self.configured()
        transport = PolicyTransport("core")
        self.crash_after_rejection(transport)
        payload = next(p for s, p in transport.requests if s == "develop")
        save(self.out / "steps/0/received_proposal.json", proposal_for(payload, "allowed"))
        before = (len(self.executed), len(transport.requests))
        with self.assertRaises(ValueError):
            self.run_search(transport)
        self.assertEqual(before, (len(self.executed), len(transport.requests)))

    def test_accepted_checkpoint_rechecks_before_only_unfinished_child_execution(self):
        self.configured()
        transport = PolicyTransport()
        self.crash_after_accepted_proposal(transport)
        self.assertEqual(len(self.executed), 1)
        with patch.object(evolution, "check_edit_policy", wraps=evolution.check_edit_policy) as check:
            self.run_search(transport)
        self.assertGreaterEqual(check.call_count, 1)
        self.assertEqual(len(self.executed), 2)
        self.assertEqual([s for s, _ in transport.requests], ["answer", "develop", "answer"])

    def test_accepted_checkpoint_without_policy_receipt_cannot_execute(self):
        self.configured()
        transport = PolicyTransport()
        self.crash_after_accepted_proposal(transport)
        (self.out / "steps/0/edit_policy.json").unlink()
        before = (len(self.executed), len(transport.requests))
        with self.assertRaisesRegex(ValueError, "policy receipt"):
            self.run_search(transport)
        self.assertEqual(before, (len(self.executed), len(transport.requests)))

    def test_matching_received_and_accepted_mutation_cannot_bypass_old_policy_receipt(self):
        self.configured()
        transport = PolicyTransport()
        self.crash_after_accepted_proposal(transport)
        payload = next(p for s, p in transport.requests if s == "develop")
        received = proposal_for(payload, "core")
        accepted = deepcopy(received); accepted["intended_target_module"] = accepted.pop("target_module")
        save(self.out / "steps/0/received_proposal.json", received)
        save(self.out / "steps/0/proposal.json", accepted)
        before = (len(self.executed), len(transport.requests))
        with self.assertRaisesRegex(ValueError, "frozen run artifact differs|violates"):
            self.run_search(transport)
        self.assertEqual(before, (len(self.executed), len(transport.requests)))

    def test_policy_change_cannot_resume_same_plan_or_skip_approval_binding(self):
        self.configured()
        approved = digest(self.plan)
        transport = PolicyTransport("core")
        self.run_search(transport)
        self.plan["edit_policy"] = deepcopy(PROGRAM_POLICY)
        with self.assertRaisesRegex(ValueError, "exact approved"):
            live.run(self.plan, approved_plan_hash=approved, execute=True,
                     phase="search", transport_factory=lambda: transport)
        before = (len(self.executed), len(transport.requests))
        with self.assertRaisesRegex(ValueError, "frozen run artifact differs"):
            self.run_search(transport)
        self.assertEqual(before, (len(self.executed), len(transport.requests)))

    def test_runtime_hash_drift_rejects_schema4_before_output(self):
        self.configured()
        self.plan["runtime_source_hashes"]["v3/edit_policy.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "runtime or live entry"):
            live.preflight(self.plan, phase="search")
        self.assertFalse(self.out.exists())

    def test_allowed_prompt_location_does_not_bypass_development_literal_guard(self):
        self.configured()
        transport = PolicyTransport("fit_literal")
        result = self.run_search(transport)
        self.assertTrue(self.artifact("steps/0/edit_policy.json")["allowed"])
        self.assertEqual(self.artifact("steps/0/rejected.json")["reason_code"], "fit_literal_match")
        self.assertEqual(self.artifact("steps/0/literal_audit.json")["status"], "reject")
        self.assertEqual(len(self.executed), 1)
        self.assertEqual(len(transport.requests), 2)
        self.assertEqual(result["search"]["terminal_attempts"][0]["status"], "rejected")

    def test_final_report_distinguishes_policy_rejections_and_preserves_legacy_fields(self):
        self.staged(fit_only=False)
        self.plan["schema"] = live.SCHEMA4
        self.plan["edit_policy"] = deepcopy(PROMPT_POLICY)
        self.executed = []
        transport = PolicyTransport("core")
        for phase in ("search", "select", "report"):
            result = self.run_search(transport, phase=phase)
        self.assertEqual(result["edit_policy"], PROMPT_POLICY)
        self.assertEqual(result["edit_policy_rejections"], 1)
        self.assertEqual(result["terminal_proposals"], 1)
        self.assertEqual(result["quality_claim"], "engineering_fixture_only")
        self.plan["schema"] = live.SCHEMA3
        self.plan.pop("edit_policy")
        self.plan["output_dir"] = str(self.base / "legacy-full")
        legacy = PolicyTransport("core")
        for phase in ("search", "select", "report"):
            old = self.run_search(legacy, phase=phase)
        self.assertNotIn("edit_policy", old)
        self.assertNotIn("edit_policy_rejections", old)

    def test_core_rejects_manifest_developer_policy_mismatch_before_measurement(self):
        fixture = recovery_fixture.EvolutionRecoveryTests()
        audit = recovery_fixture.Audit()
        inputs = fixture.inputs(edit_policy=deepcopy(PROMPT_POLICY))
        developer = evolution.ProgramDeveloper(object(), edit_policy=PROGRAM_POLICY)
        with self.assertRaisesRegex(ValueError, "policy"):
            fixture.runner(self.base / "mismatch", developer, audit, inputs=inputs)
        self.assertEqual(audit.invocations, [])

    def test_core_active_policy_and_developer_snapshot_drift_stop_before_measurement(self):
        fixture = recovery_fixture.EvolutionRecoveryTests()
        for target in ("manifest", "active", "developer"):
            with self.subTest(target=target):
                audit = recovery_fixture.Audit()
                inputs = fixture.inputs(edit_policy=deepcopy(PROMPT_POLICY))
                developer = evolution.ProgramDeveloper(object(), edit_policy=deepcopy(PROMPT_POLICY))
                runner = fixture.runner(self.base / ("drift-" + target), developer, audit, inputs=inputs)
                if target == "manifest":
                    runner.manifest["edit_policy"] = deepcopy(PROGRAM_POLICY)
                elif target == "active":
                    runner.edit_policy["allowed_prompt_stages"].append("read")
                else:
                    developer.edit_policy["allowed_prompt_stages"].append("read")
                with self.assertRaisesRegex(ValueError, "frozen"):
                    runner.run()
                self.assertEqual(audit.invocations, [])
                self.assertFalse((runner.directory / "root.json").exists())


class DeveloperPolicyPayloadTests(unittest.TestCase):
    setUp = developer_fixture.ControlledDeveloperTests.setUp
    model = developer_fixture.ControlledDeveloperTests.model
    factory = developer_fixture.ControlledDeveloperTests.factory
    transport = developer_fixture.ControlledDeveloperTests.transport
    prepare = developer_fixture.ControlledDeveloperTests.prepare
    propose = developer_fixture.ControlledDeveloperTests.propose

    def developer(self, *, policy=None, condition="cases"):
        return evolution.ProgramDeveloper(self.projection, feedback_condition=condition,
            case_schedule=self.schedule, proposal_model_factory=self.factory,
            edit_policy=deepcopy(PROMPT_POLICY if policy is None else policy))

    def test_policy_is_separate_complete_body_field_and_snapshot_is_copy(self):
        developer = self.developer()
        payload = self.prepare(developer)
        self.assertEqual(payload["edit_policy"], PROMPT_POLICY)
        self.assertNotIn("edit_policy", payload["decision"])
        self.assertEqual(payload["experience"], [])
        self.assertNotIn("General refactors are allowed", payload["edit_boundary"])
        body = self.projection.request_body("develop", payload)
        self.assertEqual(json.loads(body["messages"][-1]["content"])["edit_policy"], PROMPT_POLICY)
        snapshot = developer.configuration_snapshot()
        snapshot["edit_policy"]["allowed_prompt_stages"].append("read")
        payload["edit_policy"]["allowed_prompt_stages"].append("read")
        self.assertEqual(developer.edit_policy, PROMPT_POLICY)
        self.assertEqual(self.factory_slots, [])
        self.assertEqual(self.sent, [])

    def test_policy_changes_provider_identity_without_rewriting_case_feedback(self):
        prompt = self.prepare(self.developer())
        program = self.prepare(self.developer(policy=PROGRAM_POLICY))
        self.assertEqual(prompt["feedback"], program["feedback"])
        self.assertEqual(prompt["source_files"], program["source_files"])
        self.assertNotEqual(digest(self.projection.request_body("develop", prompt)),
                            digest(self.projection.request_body("develop", program)))
        for condition in ("cases", "trace"):
            value = self.prepare(self.developer(condition=condition))
            self.assertEqual(value["edit_policy"], PROMPT_POLICY)
            self.assertEqual([c["question_id"] for c in value["feedback"]["cases"]], ["q2", "q1"])

    def test_policy_bytes_are_inside_exact_budget_no_cropping(self):
        without = evolution.ProgramDeveloper(self.projection, feedback_condition="cases",
            case_schedule=self.schedule, proposal_model_factory=self.factory)
        limit = self.projection.request_size("develop", self.prepare(without))
        self.projection.max_input_bytes = limit
        oversized = self.developer(policy={**PROMPT_POLICY, "allowed_prompt_stages": ["plan", "read", "answer"]})
        with patch.object(evolution, "_fit_development_request", side_effect=AssertionError("legacy crop")), \
             self.assertRaisesRegex(ValueError, "no cases are cropped"):
            self.prepare(oversized)
        self.assertEqual(self.sent, [])
        self.assertEqual(self.factory_slots, [])

    def test_developer_returns_valid_out_of_policy_source_for_host_receipt(self):
        # Fixture proposes max_rounds, which violates prompt_only. It must still
        # reach the runner's durable received-proposal checkpoint before rejection.
        result = self.propose(self.developer())
        self.assertIn("rag.py", result["writes"])
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(json.loads(self.sent[0]["messages"][-1]["content"])["edit_policy"], PROMPT_POLICY)


if __name__ == "__main__":
    unittest.main()
