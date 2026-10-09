"""Synthetic context-only replay/probe contracts; no provider or private data."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from code_rsi import answer_probe as probe
from code_rsi.answer_replay import (ARMS, CONTEXT_ARMS, AnswerReplayRouter,
                                   project_case_payload, project_final_payload, replay_files)
from code_rsi.budget import digest, save, stable
from code_rsi.v3.datasets import adapt_musique
from code_rsi.v3.evolution import _runtime_source_hashes
from code_rsi.v3.execution import EXECUTION_SCHEMA, HostBroker, HostError, root_files
from code_rsi.v3.infrastructure import UnknownProviderOutcome
from code_rsi.v3.paired_analysis import SINGLE_CONTRAST_SCHEMA
from code_rsi.v3.rag import RagEngine
from code_rsi.v3.reader_replay import SCHEMA as CASE_SCHEMA, ReplayContractError, ReplayMismatch


QUOTE = "Beacon Port serves the synthetic island."
CRITERIA = {"mean_f1_gain_gt": 0, "mean_em_gain_gte": 0,
            "nonabstaining_f1_zero_increase_lte": 0}


def synthetic_case(index=0, *, kind="normal", cap=64000, long_claim=False):
    """Record trusted maintenance execution; never execute archived model code."""
    text = QUOTE if kind == "exact" else "Synthetic title\n" + "L" * 280 + QUOTE + "R" * 280
    task, reference = adapt_musique({"id": "context-panel-" + str(index),
        "question": "Which synthetic port serves panel %s?" % index,
        "answer": "Beacon Port", "answerable": True,
        "paragraphs": [{"idx": 0, "title": "", "paragraph_text": text, "is_supporting": True}]})
    config = {"mode": "iterative", "max_rounds": 3, "max_stagnant_rounds": 2,
              "max_payload_chars": cap}
    events = []
    class Recording:
        def record(self, name, request, response):
            events.append({"name": name, "request": deepcopy(request), "response": deepcopy(response),
                           "response_sha256": digest(response)})
            return deepcopy(response)

        def search(self, query, limit=5):
            lo, hi = 0, len(text)
            if kind == "repeated" and query == "first synthetic query":
                lo, hi = text.index(QUOTE) - 5, text.index(QUOTE) + len(QUOTE) + 5
            return self.record("search", {"query": query, "limit": limit}, [
                {"docid": task["documents"][0]["docid"], "start": lo, "end": hi,
                 "text": text[lo:hi], "score": 1.0}])

        def complete(self, stage, payload):
            if stage == "plan":
                response = {"constraints": ["synthetic port relation"], "queries": ["first synthetic query"]}
            elif stage == "read":
                source = payload["sources"][0]
                more = kind == "repeated" and payload["round"] == 1
                response = {"claims": [] if kind == "empty" else [{
                    "text": "x" * 1450 if long_claim else "A synthetic relation, model assessed only.",
                    "citations": [{"source_id": source["source_id"], "quote": QUOTE}]}],
                    "bridge_entities": [], "gaps": ["Unverified synthetic gap"], "conflicts": [],
                    "queries": ["second synthetic query"] if more else [], "ready": not more}
            else:
                response = {"answer": "Archived placeholder", "citation_ids": ["e1"] if payload["evidence"] else [],
                            "evidence_sufficient": bool(payload["evidence"])}
            return self.record("complete", {"stage": stage, "payload": payload}, response)
    services = Recording()
    result = RagEngine(services, services, config=config).solve({"question": task["question"]})
    assert result["answer_usable"] and result["answer"] == "Archived placeholder"
    return {"schema": CASE_SCHEMA, "case_id": "C%02d" % index, "task": task, "config": config,
            "target_read": sum(e["name"] == "complete" and e["request"]["stage"] == "read" for e in events),
            "events": events, "engine_sha256": hashlib.sha256(root_files()["rag_core.py"].encode()).hexdigest(),
            "original_final_payload_sha256": digest(events[-1]["request"]["payload"]),
            "source_binding": {"kind": "synthetic-context-loop-repeat-zero"}}, reference


def dispatch(router, event):
    return getattr(router, "complete" if event["name"] == "complete" else event["name"])(
        **deepcopy(event["request"]))


class FreshAnswer:
    def __init__(self):
        self.calls = []

    def complete(self, stage, payload):
        self.calls.append((stage, deepcopy(payload)))
        return {"answer": "Fresh synthetic answer", "citation_ids": [], "evidence_sufficient": False}


class ContextReplayTests(unittest.TestCase):
    def test_legacy_projection_semantics_are_not_extended_by_new_arm(self):
        case, _ = synthetic_case()
        original = case["events"][-1]["request"]["payload"]
        self.assertEqual(ARMS, ("full_state", "evidence_only"))
        self.assertEqual(CONTEXT_ARMS, ("full_state", "quote_context"))
        self.assertEqual(project_case_payload(case, "full_state"), original)
        self.assertEqual(project_case_payload(case, "evidence_only"), project_final_payload(original, "evidence_only"))
        with self.assertRaises(ReplayContractError):
            project_final_payload(original, "quote_context")

    def test_render_preserves_all_noncontext_fields_and_exact_bounded_text(self):
        case, _ = synthetic_case()
        snapshot = deepcopy(case)
        projected = project_case_payload(case, "quote_context")
        item = projected["evidence"][0]
        source = next(e["request"]["payload"]["sources"][0] for e in case["events"]
                      if e["name"] == "complete" and e["request"]["stage"] == "read")
        context = item.pop("context")
        self.assertEqual(context["radius_chars"], 256)
        self.assertEqual(context["start"], max(source["start"], item["start"] - 256))
        self.assertEqual(context["end"], min(source["end"], item["end"] + 256))
        self.assertEqual(context["text"], source["text"][context["start"]-source["start"]:context["end"]-source["start"]])
        self.assertEqual(context["source_sha256"], hashlib.sha256(source["text"].encode()).hexdigest())
        self.assertEqual(projected, case["events"][-1]["request"]["payload"])
        projected["claims"][0]["text"] = "mutated outside frozen case"
        self.assertEqual(case, snapshot)

    def test_first_grounding_window_survives_same_quote_in_later_larger_window(self):
        case, _ = synthetic_case(kind="repeated")
        reads = [e["request"]["payload"] for e in case["events"]
                 if e["name"] == "complete" and e["request"]["stage"] == "read"]
        self.assertEqual(len(reads), 2)
        projected = project_case_payload(case, "quote_context")
        self.assertEqual(len(projected["evidence"]), 1)
        item = projected["evidence"][0]
        self.assertEqual(item["source_id"], reads[0]["sources"][0]["source_id"])
        self.assertEqual(item["context"]["text"], reads[0]["sources"][0]["text"])
        self.assertNotEqual(item["context"]["text"], reads[1]["sources"][0]["text"])

    def test_no_quotes_or_no_neighbors_preserve_the_case_and_original_body(self):
        for kind in ("empty", "exact"):
            with self.subTest(kind=kind):
                case, _ = synthetic_case(kind=kind)
                projected = project_case_payload(case, "quote_context")
                self.assertEqual(projected, case["events"][-1]["request"]["payload"])
                self.assertFalse(any("context" in item for item in projected["evidence"]))

    def test_payload_budget_drops_neighbors_without_dropping_original_material(self):
        case, _ = synthetic_case(long_claim=True)
        original = case["events"][-1]["request"]["payload"]
        cap = len(json.dumps(original, ensure_ascii=False))
        self.assertGreater(cap, max(len(json.dumps(e["request"]["payload"], ensure_ascii=False))
                                   for e in case["events"][:-1] if e["name"] == "complete"))
        tight, _ = synthetic_case(long_claim=True, cap=cap)
        self.assertEqual(project_case_payload(tight, "quote_context"), original)
        self.assertEqual(tight["events"][-1]["request"]["payload"], original)

    def test_files_change_only_trusted_config_without_embedding_task_text(self):
        case, _ = synthetic_case()
        files = replay_files(case, "quote_context")
        self.assertEqual(files, root_files({**case["config"], "final_context_radius": 256}))
        self.assertNotIn(case["task"]["question"], files["rag.py"])
        self.assertNotIn("Archived placeholder", files["rag.py"])
        tree = ast.parse(files["rag.py"])
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in n.targets))
        self.assertEqual(json.loads(assignment.value.args[0].value),
                         {**case["config"], "final_context_radius": 256})

    def test_context_router_replays_prefix_and_only_purchases_one_new_answer(self):
        for kind in ("normal", "empty", "exact", "repeated"):
            with self.subTest(kind=kind):
                case, _ = synthetic_case(kind=kind)
                live = FreshAnswer()
                router = AnswerReplayRouter(case, live, "quote_context")
                result = RagEngine(router, router, config={**case["config"], "final_context_radius": 256}).solve(
                    {"question": case["task"]["question"]})
                self.assertEqual(result["answer"], "Fresh synthetic answer")
                self.assertEqual([s for s, _ in live.calls], ["answer"])
                self.assertEqual(live.calls[0][1], project_case_payload(case, "quote_context"))
                self.assertEqual(router.assert_complete()["new_model_calls"], 1)
                self.assertEqual(router.replayed_calls, len(case["events"]) - 1)

    def test_old_answer_cannot_be_reused_even_when_context_adds_nothing(self):
        for kind in ("empty", "exact"):
            case, _ = synthetic_case(kind=kind)
            with self.assertRaises(ReplayContractError):
                AnswerReplayRouter(case, arm="quote_context")

    def test_prefix_or_final_neighbor_tampering_never_dispatches(self):
        case, _ = synthetic_case()
        for final in (False, True):
            live = FreshAnswer()
            router = AnswerReplayRouter(case, live, "quote_context")
            if final:
                for event in case["events"][:-1]:
                    dispatch(router, event)
                payload = project_case_payload(case, "quote_context")
                payload["evidence"][0]["context"]["text"] += "not from source"
                with self.assertRaises(ReplayMismatch):
                    router.complete("answer", payload)
            else:
                request = deepcopy(case["events"][0]["request"])
                request["payload"]["constraints"] = ["unfrozen"]
                with self.assertRaises(ReplayMismatch):
                    router.complete(**request)
            self.assertEqual(live.calls, [])
            with self.assertRaises(ReplayMismatch):
                router.assert_complete()


class TrustedContextExecutor:
    """Trusted maintenance engine and host RPC only; no candidate-code execution."""
    def __init__(self, *, mismatch_answer=False):
        self.calls = []
        self.mismatch_answer = mismatch_answer

    def __call__(self, archive, node_id, task, backend, model, directory, *, limits):
        self.calls.append((model.case["case_id"], model.arm))
        broker = HostBroker(task, backend, model, **limits)
        class Backend:
            def search(self, query, limit=5):
                return broker("search", {"query": query, "limit": limit})
        class Model:
            def complete(self, stage, payload):
                return broker("complete", {"stage": stage, "payload": payload})
        cfg = {**model.case["config"], "final_context_radius": 256 if model.arm == "quote_context" else 0}
        result = RagEngine(Backend(), Model(), config=cfg).solve({"question": task["question"]})
        if broker.fatal is not None:
            raise broker.fatal
        answer = "invalid postprocessing" if self.mismatch_answer else result["answer"] or ""
        returned = [c for c in result["state"]["citations"] if c["citation_id"] in result["citation_ids"]]
        origin = broker.answer_origin_receipt(answer)
        cited = broker.citation_receipt(answer, returned)
        node = archive.load_node(node_id)
        return {"schema": EXECUTION_SCHEMA, "node_id": node_id, "program_id": node["program_id"],
                "question_id": task["question_id"], "answer": answer, "answer_usable": bool(answer.strip()),
                "execution_ok": True, "citation_source_valid": cited["valid"], "citation_status": cited["status"],
                "host_citation_validation": cited, "citations": returned,
                "answer_origin_valid": origin["valid"], "answer_origin_status": origin["status"],
                "host_answer_origin_validation": origin,
                "failure_classes": [] if origin["valid"] else ["invalid_answer_origin"],
                "model_errors": broker.model_errors, "trace": broker.events,
                "host_evidence_trace": {"read_presentations": broker.read_presentations,
                                        "final_observations": broker.final_observations},
                "candidate_reported": result, "resource_usage": broker.counts}


class ScriptedAnswers:
    def __init__(self, choose=None):
        self.sent = []
        self.choose = choose

    def __call__(self, body):
        self.sent.append(deepcopy(body))
        payload = json.loads(body["messages"][-1]["content"])
        assert "evidence" in payload and "sources" not in payload
        answer = self.choose(payload) if self.choose else "Beacon Port"
        response = {"answer": answer, "citation_ids": [e["citation_id"] for e in payload["evidence"]],
                    "evidence_sufficient": answer != "Insufficient information"}
        return {"model": "synthetic-model", "choices": [{"finish_reason": "stop",
                "message": {"content": json.dumps(response)}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "prompt_cache_hit_tokens": 40}}


class ContextProbeTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).parent / "runs"
        root.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="synthetic-context-probe-", dir=root)
        self.root = Path(temporary.name)
        self.addCleanup(temporary.cleanup)
        guard = patch.object(probe, "credential_from_plan", side_effect=AssertionError("credential read"))
        guard.start()
        self.addCleanup(guard.stop)

    def plan(self, *, kinds=("normal",), repeats=1, opaque=False):
        pairs = [synthetic_case(i, kind=kind) for i, kind in enumerate(kinds)]
        cases = [case for case, _ in pairs]
        save(self.root / "cases.json", {"schema": "rag-rsi-reader-replay-cases-1", "cases": cases})
        if opaque:
            (self.root / "refs.json").write_bytes(b"opaque reference; forbidden before generation")
        else:
            save(self.root / "refs.json", {r["question_id"]: r for _, r in pairs})
        def bind(name):
            path = self.root / name
            return {"path": str(path), "sha256": probe.file_hash(path)}
        return {"schema": "rag-rsi-answer-probe-2", "purpose": "fixed_evidence_quote_context",
                "question_use": "synthetic", "source_plan_file": None, "source_arm": "loop", "source_checkout": None,
                "advance_criteria": deepcopy(CRITERIA),
                "cases_file": bind("cases.json"), "references_file": bind("refs.json"),
                "output_dir": str(self.root / "output"), "arms": [{"name": name} for name in CONTEXT_ARMS],
                "repeats": repeats, "schedule_seed": 31,
                "analysis": {"schema": SINGLE_CONTRAST_SCHEMA, "primary_metric": "answer_f1",
                    "comparisons": [{"name": "restore_source_context", "baseline": "full_state", "candidate": "quote_context"}],
                    "question_groups": {c["task"]["question_id"]: "g" + str(i) for i, c in enumerate(cases)},
                    "confidence_level": .95, "bootstrap_samples": 1000, "bootstrap_seed": 83,
                    "target_effect": .1, "power": .8},
                "model": {"name": "deepseek-flash", "temperature": 0, "thinking": "disabled",
                          "max_input_bytes": 120000, "output_limits": {"answer": 800},
                          "prices": {"input_miss": 2, "input_hit": .04, "output": 8}},
                "hard_cny": 50, "max_calls": len(kinds) * repeats * 2,
                "runtime_source_hashes": _runtime_source_hashes(), "helper_source_hashes": probe.helper_hashes(),
                "credential_source": {"kind": "env_file", "path": str(self.root / "absent-credentials"),
                                      "variable": "DEEPSEEK_API_KEY"}}

    def generate(self, plan, *, executor=None, transport=None):
        executor = executor or TrustedContextExecutor()
        transport = transport or ScriptedAnswers()
        frozen = probe.generate(plan, approved_plan_hash=digest(plan), executor=executor, transport=transport)
        return frozen, executor, transport

    def reject_before_reference(self, plan):
        original = probe._verified_bytes
        def guarded(binding):
            if binding == plan["references_file"]:
                raise AssertionError("reference read before full valid freeze")
            return original(binding)
        with patch.object(probe, "_verified_bytes", side_effect=guarded), self.assertRaises((ValueError, HostError)):
            probe.grade(plan)

    def test_preflight_is_zero_call_and_keeps_zero_quote_and_no_neighbor_cases(self):
        plan = self.plan(kinds=("normal", "empty", "exact"), repeats=2, opaque=True)
        original = probe._verified_bytes
        def checked(binding):
            self.assertEqual(binding, plan["cases_file"])
            return original(binding)
        with patch.object(probe, "_verified_bytes", side_effect=checked), \
                patch.object(probe, "deepseek_transport", side_effect=AssertionError("transport construction")):
            result = probe.preflight(plan)
        self.assertEqual((result["states"], result["outcomes"], result["max_new_calls"]), (3, 12, 12))
        self.assertEqual(result["new_api_calls"], 0)
        self.assertFalse(result["reference_or_credential_access"])
        self.assertFalse(Path(plan["output_dir"]).exists())
        self.assertEqual(result["advance_criteria"], CRITERIA)
        self.assertEqual(result["context_delivery"]["C00"]["context_citations"], 1)
        self.assertEqual(result["context_delivery"]["C01"]["original_citations"], 0)
        self.assertEqual(result["context_delivery"]["C02"]["context_citations"], 0)
        self.assertTrue(all(v["all_noncontext_fields_equal"] for v in result["context_delivery"].values()))

    def test_legacy_schema_still_preflights_the_original_ablation(self):
        plan = self.plan()
        for key in ("source_arm", "source_checkout", "advance_criteria"):
            del plan[key]
        plan.update(schema=probe.SCHEMA, purpose="fixed_evidence_final_judgment_ablation",
                    arms=[{"name": name} for name in ARMS])
        plan["analysis"]["comparisons"] = [{"name": "remove_derived_judgments",
            "baseline": "full_state", "candidate": "evidence_only"}]
        result = probe.preflight(plan)
        self.assertEqual(result["max_new_calls"], 2)
        self.assertNotIn("context_delivery", result)
        self.assertNotIn("advance_criteria", result)

    def test_context_budget_uses_the_actual_complete_provider_body(self):
        plan = self.plan(repeats=2)
        check = probe.preflight(plan)
        case = probe._snapshot(plan)[0]
        shape = probe._RequestShape(plan["model"])
        bound = 0
        for arm in CONTEXT_ARMS:
            body = shape.request_body("answer", project_case_payload(case, arm))
            size = len(stable(body).encode())
            self.assertEqual(check["answer_body_bytes"][case["case_id"]][arm], size)
            self.assertEqual(check["answer_body_hashes"][case["case_id"]][arm], digest(body))
            bound += 2 * ((size + 1024) * 2 + 800 * 8) / 1e6
        self.assertAlmostEqual(check["conservative_cny_upper_bound"], bound)
        sizes = check["answer_body_bytes"][case["case_id"]]
        self.assertGreater(sizes["quote_context"], sizes["full_state"])
        plan["hard_cny"] = bound / 2
        with self.assertRaises(ValueError):
            probe.preflight(plan)

    def test_schema_purpose_arms_source_and_criteria_cannot_cross_protocols(self):
        original = self.plan()
        mutations = [lambda p: p.update(schema=probe.SCHEMA),
                     lambda p: p.update(purpose="fixed_evidence_final_judgment_ablation"),
                     lambda p: p.update(arms=[{"name": n} for n in ARMS]),
                     lambda p: p.update(source_arm="support"),
                     lambda p: p.update(source_checkout=str(self.root)),
                     lambda p: p["analysis"]["comparisons"][0].update(name="remove_derived_judgments"),
                     lambda p: p["advance_criteria"].update(mean_f1_gain_gt=-.01),
                     lambda p: p["advance_criteria"].update(nonabstaining_f1_zero_increase_lte=1),
                     lambda p: p["advance_criteria"].update(mean_em_gain_gte=False),
                     lambda p: p["advance_criteria"].update(extra="unfrozen"),
                     lambda p: p.pop("advance_criteria")]
        for index, mutate in enumerate(mutations):
            plan = deepcopy(original)
            mutate(plan)
            with self.subTest(index=index), self.assertRaises(ValueError):
                probe.preflight(plan)

    def test_generation_buys_one_new_answer_per_cell_and_resume_buys_none(self):
        plan = self.plan(kinds=("normal", "empty", "exact"), repeats=2)
        frozen, executor, transport = self.generate(plan)
        self.assertEqual(len(frozen["cells"]), 12)
        self.assertEqual(len(transport.sent), 12)
        self.assertEqual(len(executor.calls), 12)
        ledger = [json.loads(line) for line in (Path(plan["output_dir"]) / "ledger.jsonl").read_text().splitlines()]
        reserves = [x for x in ledger if x["event"] == "reserve"]
        self.assertEqual(len({x["metadata"]["bank"] for x in reserves}), 12)
        with patch.object(probe, "deepseek_transport", side_effect=AssertionError("transport construction")):
            resumed = probe.generate(plan, approved_plan_hash=digest(plan), executor=executor)
        self.assertEqual(resumed, frozen)
        self.assertEqual(len(transport.sent), 12)
        self.assertEqual(len(executor.calls), 12)

    def test_reference_read_waits_for_complete_validated_generation(self):
        plan = self.plan()
        self.reject_before_reference(plan)
        self.generate(plan)
        original = probe._verified_bytes
        seen = []
        def checked(binding):
            if binding == plan["references_file"]:
                self.assertTrue((Path(plan["output_dir"]) / "generation_freeze.json").exists())
                seen.append(True)
            return original(binding)
        with patch.object(probe, "_verified_bytes", side_effect=checked):
            report = probe.grade(plan)
        self.assertEqual(seen, [True])
        self.assertFalse(report["independent_quality_evidence"])
        self.assertEqual(set(report["summary"]), set(CONTEXT_ARMS))
        self.assertEqual(report, probe.grade(plan))

    def test_unknown_physical_result_has_no_score_freeze_or_retry(self):
        plan = self.plan()
        sent = []
        def unknown(body):
            sent.append(body)
            raise TimeoutError("synthetic unknown physical outcome")
        with self.assertRaises(UnknownProviderOutcome):
            self.generate(plan, transport=unknown)
        with self.assertRaises(UnknownProviderOutcome):
            probe.generate(plan, approved_plan_hash=digest(plan), executor=TrustedContextExecutor())
        self.assertEqual(len(sent), 1)
        self.assertFalse((Path(plan["output_dir"]) / "generation_freeze.json").exists())
        self.assertFalse((Path(plan["output_dir"]) / "report.json").exists())

    def test_interrupted_freeze_resumes_without_buying_new_answers(self):
        plan = self.plan()
        executor, transport = TrustedContextExecutor(), ScriptedAnswers()
        original = probe.freeze
        def interrupt(path, value):
            if Path(path).name == "generation_freeze.json":
                raise OSError("synthetic freeze interruption")
            return original(path, value)
        with patch.object(probe, "freeze", side_effect=interrupt), self.assertRaises(OSError):
            self.generate(plan, executor=executor, transport=transport)
        self.assertEqual(len(transport.sent), 2)
        self.reject_before_reference(plan)
        frozen, _, _ = self.generate(plan, executor=executor, transport=transport)
        self.assertEqual(len(frozen["cells"]), 2)
        self.assertEqual(len(transport.sent), 2)
        self.assertEqual(len(executor.calls), 2)

    def test_invalid_origin_does_not_create_successful_subset_quality(self):
        plan = self.plan()
        self.generate(plan, executor=TrustedContextExecutor(mismatch_answer=True))
        report = probe.grade(plan)
        self.assertEqual(report["status"], "protocol_invalid")
        self.assertFalse(report["paired_analysis"]["quality_comparison_valid"])
        self.assertTrue(all(r["metrics"] is None for r in report["rows"]))
        self.assertTrue(all(s["mean_f1"] is None for s in report["summary"].values()))
        self.assertFalse(report["advance_decision"]["passed"])
        self.assertFalse(any(report["advance_decision"]["checks"].values()))

    def test_single_invalid_cell_does_not_rank_the_remaining_successes(self):
        plan = self.plan(kinds=("normal", "normal"))
        trusted = TrustedContextExecutor()
        def one_invalid(archive, node_id, task, backend, model, directory, *, limits):
            trusted.mismatch_answer = model.arm == "quote_context" and model.case["case_id"] == "C01"
            return trusted(archive, node_id, task, backend, model, directory, limits=limits)
        self.generate(plan, executor=one_invalid)
        report = probe.grade(plan)
        self.assertEqual(sum(r["program_eligible"] for r in report["rows"]), 3)
        self.assertEqual(sum(r["metrics"] is None for r in report["rows"]), 1)
        self.assertTrue(all(s["mean_em"] is None and s["mean_f1"] is None
                            for s in report["summary"].values()))
        self.assertFalse(report["paired_analysis"]["successful_subset_analysis"])
        self.assertFalse(report["advance_decision"]["passed"])

    def test_rehashed_context_tampering_still_fails_before_reference_access(self):
        plan = self.plan()
        frozen, _, transport = self.generate(plan)
        cell = next(c for c in frozen["cells"] if c["identity"]["arm"] == "quote_context")
        path = Path(plan["output_dir"]) / cell["file"]
        record = json.loads(path.read_text())
        event = next(e for e in reversed(record["payload"]["trace"]) if e["name"] == "complete")
        event["request"]["payload"]["evidence"][0]["context"]["text"] += "forged neighborhood"
        event["payload_sha256"] = digest(event["request"]["payload"])
        record["payload_hash"] = digest(record["payload"])
        save(path, record)
        cell["sha256"] = probe.file_hash(path)
        save(Path(plan["output_dir"]) / "generation_freeze.json", frozen)
        self.reject_before_reference(plan)
        self.assertEqual(len(transport.sent), 2)

    def test_missing_or_duplicated_context_arm_cannot_be_dropped_after_generation(self):
        plan = self.plan()
        frozen, _, _ = self.generate(plan)
        baseline = next(c for c in frozen["cells"] if c["identity"]["arm"] == "full_state")
        for cells in ([baseline], [baseline, baseline]):
            with self.subTest(count=len(cells)):
                save(Path(plan["output_dir"]) / "generation_freeze.json", {**frozen, "cells": cells})
                self.reject_before_reference(plan)

    def test_extra_wrong_answer_blocks_advancement_even_when_em_and_f1_improve(self):
        plan = self.plan(kinds=("normal", "normal"))
        def choose(payload):
            contextual = any("context" in item for item in payload["evidence"])
            second = payload["question"].endswith("panel 1?")
            if second:
                return "Wrong Dock" if contextual else "Insufficient information"
            return "Beacon Port" if contextual else "Beacon"
        self.generate(plan, transport=ScriptedAnswers(choose))
        report = probe.grade(plan)
        baseline, candidate = report["summary"]["full_state"], report["summary"]["quote_context"]
        self.assertGreater(candidate["mean_f1"], baseline["mean_f1"])
        self.assertGreater(candidate["mean_em"], baseline["mean_em"])
        self.assertEqual((baseline["nonabstaining_f1_zero"], candidate["nonabstaining_f1_zero"]), (0, 1))
        gate = report["advance_decision"]
        self.assertEqual(gate["criteria"], CRITERIA)
        self.assertEqual(gate["checks"], {"all_cells_eligible": True, "f1_increased": True,
            "em_not_decreased": True, "nonabstaining_f1_zero_not_increased": False})
        self.assertFalse(gate["passed"])

    def test_passed_diagnostic_is_not_independent_improvement_or_automatic_deployment(self):
        plan = self.plan(kinds=("normal", "normal"))
        def choose(payload):
            return "Beacon Port" if any("context" in item for item in payload["evidence"]) else "Insufficient information"
        self.generate(plan, transport=ScriptedAnswers(choose))
        report = probe.grade(plan)
        gate = report["advance_decision"]
        self.assertTrue(gate["passed"])
        self.assertTrue(all(gate["checks"].values()))
        self.assertTrue(gate["independent_confirmation_required"])
        self.assertFalse(gate["automatic_deployment"])
        self.assertFalse(gate["rsi_benefit_verified"])
        self.assertFalse(report["independent_quality_evidence"])
        self.assertEqual(report["source_arm"], "loop")
        self.assertEqual(report["context_delivery"], probe.preflight(plan)["context_delivery"])

    def test_equal_scores_do_not_satisfy_strict_positive_f1_advancement(self):
        plan = self.plan()
        self.generate(plan)
        report = probe.grade(plan)
        self.assertEqual(report["summary"]["quote_context"]["mean_f1"],
                         report["summary"]["full_state"]["mean_f1"])
        self.assertFalse(report["advance_decision"]["checks"]["f1_increased"])
        self.assertFalse(report["advance_decision"]["passed"])


if __name__ == "__main__":
    unittest.main()
