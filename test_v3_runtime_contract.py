"""Candidate capability cards and opt-in strict developer JSON; no network."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from code_rsi.budget import Ledger, digest, stable
from code_rsi.v3.infrastructure import (PROMPTS, ModelResponseError, StructuredModel,
                                       UnknownProviderOutcome, proposal_model_prompts)
from code_rsi.v3.runtime_contract import (SCHEMA, build_runtime_contract,
                                        validate_runtime_contract)


EXACT = {"schema": "rag-rsi-proposal-protocol-1", "format": "exact_edits",
         "max_edits": 8, "max_edit_chars": 12000}
HOST = {"max_models": 7, "max_reads": 8, "max_searches": 8}
CONFIG = {"name": "deepseek-flash", "thinking": "disabled", "temperature": 0,
          "max_input_bytes": 120000,
          "output_limits": {"plan": 1200, "read": 2200, "answer": 800, "develop": 18000},
          "prices": {"input_hit": .04, "input_miss": 2, "output": 8}}


def card():
    return build_runtime_contract(HOST, CONFIG, EXACT, PROMPTS)


class RuntimeContractTests(unittest.TestCase):
    def test_real_dynamic_limits_are_projected_without_private_model_metadata(self):
        config = deepcopy(CONFIG)
        config.update(secret="DO_NOT_SEND", path="PRIVATE_PATH")
        config["max_input_bytes"] = 54000
        config["output_limits"].update(plan=913, read=1501, answer=455, develop=23000)
        protocol = {**EXACT, "max_edits": 3, "max_edit_chars": 4567}
        result = build_runtime_contract({"max_models": 5, "max_reads": 3, "max_searches": 4},
                                        config, protocol, PROMPTS)
        self.assertEqual(result["schema"], SCHEMA)
        self.assertNotIn("status", result)
        self.assertEqual(result["host_limits"], {"max_models": 5, "max_reads": 3, "max_searches": 4})
        self.assertEqual(result["model_request_limits"], {"max_input_bytes": 54000,
                         "output_limits": {"plan": 913, "read": 1501, "answer": 455}})
        self.assertEqual(result["edit_output"]["max_edits"], 3)
        self.assertEqual(result["edit_output"]["max_edit_chars"], 4567)
        self.assertEqual(result["immutable_qa_system_prompts"], {s: PROMPTS[s] for s in ("plan", "read", "answer")})
        for forbidden in ("DO_NOT_SEND", "PRIVATE_PATH", "prices", "contract_sha256"):
            self.assertNotIn(forbidden, stable(result))
        self.assertEqual(validate_runtime_contract(result), result)

    def test_none_is_legacy_and_returns_deep_independent_values(self):
        self.assertIsNone(validate_runtime_contract(None))
        config, host, prompts = deepcopy(CONFIG), deepcopy(HOST), deepcopy(PROMPTS)
        result = build_runtime_contract(host, config, EXACT, prompts)
        copy = validate_runtime_contract(result)
        host["max_models"] = 30
        prompts["answer"] = "mutated"
        config["output_limits"]["answer"] = 6
        copy["capabilities"]["complete"]["unsupported_examples"].append("mutated")
        self.assertEqual(result["host_limits"]["max_models"], 7)
        self.assertEqual(result["immutable_qa_system_prompts"]["answer"], PROMPTS["answer"])
        self.assertNotIn("mutated", stable(result))

    def test_only_exact_edit_protocol_is_supported(self):
        for protocol in (None, {"schema": EXACT["schema"], "format": "whole_files"}):
            with self.subTest(protocol=protocol), self.assertRaisesRegex(ValueError, "exact_edits"):
                build_runtime_contract(HOST, CONFIG, protocol, PROMPTS)

    def test_closed_card_rejects_unknown_fields_at_any_level(self):
        for path in ((), ("host_limits",), ("model_request_limits",),
                     ("immutable_qa_system_prompts",), ("capabilities", "complete"),
                     ("edit_output", "minimal_example", "edits", 0)):
            value = card()
            target = value
            for key in path:
                target = target[key]
            target["undocumented"] = "sentinel"
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_runtime_contract(value)

    def test_fixed_text_schema_and_bool_type_cannot_be_changed(self):
        mutations = [lambda c: c.update(schema="rag-rsi-candidate-runtime-card-draft-2"),
                     lambda c: c.update(answer_origin_rule="Any answer is acceptable"),
                     lambda c: c["capabilities"]["complete"].update(stages_are_exhaustive=1),
                     lambda c: c["capabilities"]["complete"]["unsupported_examples"].clear(),
                     lambda c: c["edit_output"].update(extra_keys_allowed=True),
                     lambda c: c["edit_output"].update(json_rule="Last duplicate key wins")]
        for mutate in mutations:
            value = card()
            mutate(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_runtime_contract(value)

    def test_dynamic_limits_reject_bool_fraction_and_out_of_range(self):
        for key, value in (("max_models", True), ("max_models", 0), ("max_reads", -1),
                           ("max_searches", 3.0), ("max_models", 65)):
            invalid = card()
            invalid["host_limits"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_runtime_contract(invalid)
        for field, value in (("max_input_bytes", float("nan")), ("max_input_bytes", 120001)):
            invalid = card()
            invalid["model_request_limits"][field] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_runtime_contract(invalid)
        for value in (0, True, 32769):
            invalid = card()
            invalid["model_request_limits"]["output_limits"]["read"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_runtime_contract(invalid)

    def test_missing_or_mistyped_card_sections_are_rejected(self):
        for name in card():
            value = card()
            del value[name]
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_runtime_contract(value)
        for value in ([], "card", 1, {"schema": SCHEMA, "edit_output": []}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_runtime_contract(value)


class RuntimeStructuredModelTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).parent / "runs"
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="runtime_contract_", dir=base)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = Ledger(self.root / "ledger.jsonl", {"run": {"calls": 20, "cny": 10}})
        self.sent = []
        self.content = '{"change_status":"no_change","edits":[]}'
        self.finish_reason = "stop"

    def send(self, body):
        self.sent.append(deepcopy(body))
        return {"choices": [{"finish_reason": self.finish_reason,
                             "message": {"content": self.content}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20}}

    def model(self, *, enabled=True, protocol=EXACT, **kwargs):
        return StructuredModel(self.root / ("new" if enabled else "legacy"), self.ledger, self.send,
                               bank="fixture", prices=CONFIG["prices"], proposal_protocol=protocol,
                               runtime_contract=card() if enabled else None, **kwargs)

    def payload(self):
        return {"proposal_protocol": deepcopy(EXACT), "runtime_contract": card(), "source_files": {}}

    def test_actual_develop_body_and_identity_bind_full_card(self):
        model = self.model()
        payload = self.payload()
        body = model.request_body("develop", payload)
        self.assertEqual(json.loads(body["messages"][1]["content"])["runtime_contract"], card())
        self.assertEqual(model.request_size("develop", payload), len(stable(body).encode()))
        expected = {"model": "deepseek-flash", "prompts": proposal_model_prompts(EXACT),
                    "limits": CONFIG["output_limits"], "max_input_bytes": 120000,
                    "temperature": 0, "thinking": "disabled", "response_format": "json_object",
                    "prices": CONFIG["prices"], "proposal_protocol": EXACT, "runtime_contract": card()}
        self.assertEqual(model.identity, digest(expected))
        self.assertNotEqual(model.identity, self.model(enabled=False).identity)
        self.assertEqual(self.sent, [])
        self.assertEqual(self.ledger.events, [])

    def test_missing_or_changed_payload_card_fails_before_dispatch(self):
        model = self.model()
        for value in (None, {**card(), "host_limits": {**HOST, "max_models": 6}}):
            payload = self.payload()
            payload["runtime_contract"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "runtime contract"):
                model.complete("develop", payload)
        self.assertEqual(self.sent, [])
        self.assertEqual(self.ledger.events, [])

    def test_legacy_cannot_silently_accept_new_card(self):
        with self.assertRaisesRegex(ValueError, "runtime contract"):
            self.model(enabled=False).complete("develop", self.payload())
        self.assertEqual(self.sent, [])

    def test_model_settings_or_immutable_prompts_cannot_disagree_with_card(self):
        for args in ({"max_input_bytes": 100000}, {"limits": {"read": 1000}},
                     {"protocol": {**EXACT, "max_edit_chars": 11000}}, {"protocol": None}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.model(**args)
        changed = card()
        changed["immutable_qa_system_prompts"]["answer"] = "invent answers"
        with self.assertRaisesRegex(ValueError, "actual model"):
            StructuredModel(self.root / "wrong", self.ledger, self.send, bank="fixture",
                            prices=CONFIG["prices"], proposal_protocol=EXACT, runtime_contract=changed)

    def test_mutation_after_construction_is_rejected_not_dispatched(self):
        for mutate in (lambda m: m.runtime_contract["host_limits"].update(max_models=6),
                       lambda m: m.limits.update(read=2100),
                       lambda m: setattr(m, "max_input_bytes", 110000)):
            model = self.model()
            mutate(model)
            payload = self.payload()
            payload["runtime_contract"] = deepcopy(model.runtime_contract)
            with self.assertRaisesRegex(ValueError, "changed"):
                model.complete("develop", payload)
        self.assertEqual(self.sent, [])

    def test_root_nested_and_escaped_duplicate_keys_are_billed_once_and_never_repaired(self):
        values = ['{"edits":[],"edits":[]}',
                  '{"edits":[{"old":"a","new":"b","new":"c"}]}',
                  '{"mechanism":"one","mechani\\u0073m":"two"}']
        for i, value in enumerate(values):
            self.content = value
            model = self.model()
            payload = {**self.payload(), "slot": i}
            before = len(self.sent)
            for attempt in range(2):
                with self.assertRaisesRegex(ModelResponseError, "unique-key JSON"):
                    model.complete("develop", payload)
            self.assertEqual(len(self.sent) - before, 1)
            self.assertEqual(model.calls, 1)
        self.assertEqual(self.ledger.summary()["pending"], 0)
        self.assertEqual(self.ledger.used["run"]["calls"], 3)
        for file in (self.root / "new").glob("*.json"):
            self.assertEqual(json.loads(file.read_text(encoding="utf-8"))["state"], "settled")

    def test_new_developer_rejects_nonfinite_json_with_settled_usage(self):
        self.content = '{"x":NaN}'
        with self.assertRaises(ModelResponseError):
            self.model().complete("develop", self.payload())
        self.assertEqual(self.ledger.used["run"]["calls"], 1)
        self.assertEqual(self.ledger.summary()["pending"], 0)

    def test_unique_nested_objects_remain_valid(self):
        self.content = '{"a":{"new":"x"},"b":{"new":"y"},"array":[{"old":"z"}]}'
        self.assertEqual(self.model().complete("develop", self.payload()), json.loads(self.content))

    def test_legacy_develop_duplicates_keep_last_key_behavior(self):
        self.content = '{"a":1,"a":2}'
        self.assertEqual(self.model(enabled=False).complete("develop", {"proposal_protocol": EXACT}), {"a": 2})

    def test_qa_requests_and_duplicate_parsing_unchanged_even_with_contract(self):
        model, legacy = self.model(), self.model(enabled=False)
        self.content = '{"answer":"first","answer":"last"}'
        for stage in ("plan", "read", "answer"):
            payload = {"question": "Synthetic question"}
            self.assertEqual(model.request_body(stage, payload), legacy.request_body(stage, payload))
            self.assertEqual(model.complete(stage, payload), {"answer": "last"})

    def test_legacy_identity_and_attribute_free_shape_preserve_exact_bytes(self):
        legacy = self.model(enabled=False, protocol=None)
        expected = {"model": "deepseek-flash", "prompts": PROMPTS, "limits": CONFIG["output_limits"],
                    "max_input_bytes": 120000, "temperature": 0, "thinking": "disabled",
                    "response_format": "json_object", "prices": CONFIG["prices"]}
        self.assertEqual(legacy.identity, digest(expected))
        shape = object.__new__(StructuredModel)
        shape.model, shape.limits = legacy.model, deepcopy(legacy.limits)
        for stage in PROMPTS:
            self.assertEqual(stable(shape.request_body(stage, {})), stable(legacy.request_body(stage, {})))

    def test_unknown_physical_result_is_not_a_json_error_and_never_retried(self):
        model = self.model()
        def fail(body):
            self.sent.append(deepcopy(body))
            raise TimeoutError("synthetic unknown outcome")
        model.transport = fail
        for attempt in range(2):
            with self.assertRaises(UnknownProviderOutcome):
                model.complete("develop", self.payload())
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.ledger.used["run"]["calls"], 1)
        caches = list((self.root / "new").glob("*.json"))
        self.assertEqual(len(caches), 1)
        self.assertEqual(json.loads(caches[0].read_text(encoding="utf-8"))["state"], "pending")

    def test_truncated_completed_response_is_billed_without_repair(self):
        self.finish_reason = "length"
        model = self.model()
        for attempt in range(2):
            with self.assertRaisesRegex(ModelResponseError, "truncated"):
                model.complete("develop", self.payload())
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.ledger.summary()["pending"], 0)


if __name__ == "__main__":
    unittest.main()
