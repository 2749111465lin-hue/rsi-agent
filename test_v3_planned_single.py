"""Synthetic planned-single comparisons; no provider, references or external data."""
from copy import deepcopy
import unittest

from code_rsi.v3.rag import RagContractError, RagEngine


QUESTION = "Where was the discoverer of the fictional Aster comet born?"
CONSTRAINTS = ["Identify the discoverer", "Find that person's birthplace"]
INITIAL_QUERIES = ["Aster comet discoverer", "Aster comet archive"]
FOLLOWUP = "Mira Vale birthplace"


class ChainBackend:
    def __init__(self):
        self.calls = []

    def search(self, query, limit):
        if query == FOLLOWUP:
            docid, text = "birthplace", "Mira Vale was born in Northport."
        elif query == INITIAL_QUERIES[1]:
            docid, text = "archive", "The archive credits Mira Vale with the Aster discovery."
        else:
            docid, text = "discovery", "The Aster comet was discovered by Mira Vale."
        rows = [{"docid": docid, "start": 0, "end": len(text), "text": text}][:limit]
        self.calls.append({"query": query, "limit": limit, "rows": deepcopy(rows)})
        return rows


class ChainModel:
    def __init__(self, *, planner=None, reader=None, ready_first=False):
        self.calls = []
        self.planner, self.reader, self.ready_first = planner, reader, ready_first

    def complete(self, stage, payload):
        self.calls.append((stage, deepcopy(payload)))
        override = self.planner if stage == "plan" else self.reader if stage == "read" else None
        if isinstance(override, Exception):
            raise override
        if override is not None:
            return deepcopy(override)
        if stage == "plan":
            return {"constraints": list(CONSTRAINTS), "queries": list(INITIAL_QUERIES)}
        if stage == "read":
            found = any(source["docid"] == "birthplace" for source in payload["sources"])
            return {"claims": [{"text": source["text"], "citations": [
                        {"source_id": source["source_id"], "quote": source["text"]}]
                    } for source in payload["sources"]],
                    "bridge_entities": ["Mira Vale"], "gaps": [] if found else ["Birthplace missing"],
                    "conflicts": [], "queries": [] if found else [FOLLOWUP],
                    "ready": found or self.ready_first}
        for item in payload["evidence"]:
            if item["docid"] == "birthplace":
                return {"answer": "Northport", "citation_ids": [item["citation_id"]],
                        "evidence_sufficient": True}
        return {"answer": "Insufficient information", "citation_ids": [], "evidence_sufficient": False}


def run(mode, *, calls=5, rounds=3, model=None):
    backend, model = ChainBackend(), model or ChainModel()
    config = {"mode": mode, "max_model_calls": calls, "max_rounds": rounds,
              "max_stagnant_rounds": 2, "search_limit": 4,
              "prompts": {"plan": "Synthetic shared planning guidance", "read": "Synthetic shared reading guidance"}}
    result = RagEngine(backend, model, config=config).solve({"question": QUESTION})
    return result, backend, model


class PlannedSingleTests(unittest.TestCase):
    def test_first_plan_retrieval_and_read_are_identical_to_iterative(self):
        planned, single_backend, single_model = run("planned_single")
        loop, loop_backend, loop_model = run("iterative")
        self.assertEqual(single_model.calls[:2], loop_model.calls[:2])
        self.assertEqual(single_backend.calls, loop_backend.calls[:2])
        self.assertEqual([call["query"] for call in single_backend.calls], INITIAL_QUERIES)
        first_read = single_model.calls[1][1]
        self.assertEqual(first_read["constraints"], CONSTRAINTS)
        self.assertEqual(first_read["round"], 1)
        self.assertEqual({source["docid"] for source in first_read["sources"]}, {"discovery", "archive"})
        self.assertEqual([stage for stage, _ in single_model.calls], ["plan", "read", "answer"])
        self.assertEqual([stage for stage, _ in loop_model.calls], ["plan", "read", "read", "answer"])
        self.assertEqual([call["query"] for call in loop_backend.calls], INITIAL_QUERIES + [FOLLOWUP])
        self.assertEqual(planned["answer"], "Insufficient information")
        self.assertEqual(loop["answer"], "Northport")
        self.assertNotIn("birthplace", {item["docid"] for item in single_model.calls[-1][1]["evidence"]})
        self.assertIn("birthplace", {item["docid"] for item in loop_model.calls[-1][1]["evidence"]})
        self.assertEqual(planned["state"]["rounds"], 1)
        self.assertEqual(loop["state"]["rounds"], 2)

    def test_two_call_planned_mode_is_rejected_before_any_service(self):
        backend, model = ChainBackend(), ChainModel()
        with self.assertRaisesRegex(RagContractError, "planned_single"):
            RagEngine(backend, model, config={"mode": "planned_single", "max_model_calls": 2})
        self.assertEqual(backend.calls, [])
        self.assertEqual(model.calls, [])

    def test_three_or_five_call_cap_always_finishes_after_first_read(self):
        for cap in (3, 5):
            for rounds in (1, 9):
                with self.subTest(cap=cap, rounds=rounds):
                    result, backend, model = run("planned_single", calls=cap, rounds=rounds)
                    self.assertEqual([stage for stage, _ in model.calls], ["plan", "read", "answer"])
                    self.assertEqual(result["usage"]["model_calls"], 3)
                    self.assertEqual(result["usage"]["final_calls"], 1)
                    self.assertEqual(result["state"]["rounds"], 1)
                    self.assertEqual(result["stop_reason"], "planned_single")
                    self.assertEqual(model.calls[-1][1]["stop_reason"], "planned_single")
                    self.assertNotIn(FOLLOWUP, [call["query"] for call in backend.calls])

    def test_original_single_pass_still_skips_planning_at_all_caps(self):
        for cap in (2, 3, 5):
            with self.subTest(cap=cap):
                result, backend, model = run("single_pass", calls=cap)
                self.assertEqual([stage for stage, _ in model.calls], ["read", "answer"])
                self.assertEqual([call["query"] for call in backend.calls], [QUESTION])
                self.assertEqual(model.calls[0][1]["constraints"], [])
                self.assertEqual(result["stop_reason"], "single_pass")
                self.assertEqual(result["usage"]["model_calls"], 2)

    def test_iterative_keeps_existing_two_three_and_five_call_behavior(self):
        expected = {2: ["plan", "answer"], 3: ["plan", "read", "answer"],
                    5: ["plan", "read", "read", "answer"]}
        for cap, stages in expected.items():
            with self.subTest(cap=cap):
                result, backend, model = run("iterative", calls=cap)
                self.assertEqual([stage for stage, _ in model.calls], stages)
                self.assertLessEqual(result["usage"]["model_calls"], cap)
                self.assertEqual(result["usage"]["final_calls"], 1)
                self.assertEqual(result["answer"], "Northport" if cap == 5 else "Insufficient information")
                if cap < 5:
                    self.assertEqual(result["stop_reason"], "final_reserved")

    def test_failed_planners_fall_back_to_question_then_read_and_answer(self):
        failures = [RuntimeError("synthetic planner failure"), {"queries": []},
                    {"constraints": [], "queries": "not a list"},
                    {"constraints": [], "queries": [], "_meta": {"truncated": True}}]
        for failure in failures:
            with self.subTest(failure=failure):
                planned, backend, model = run("planned_single", calls=3, model=ChainModel(planner=failure))
                loop, loop_backend, loop_model = run("iterative", calls=3, model=ChainModel(planner=failure))
                self.assertEqual(model.calls[:2], loop_model.calls[:2])
                self.assertEqual(backend.calls[0], loop_backend.calls[0])
                self.assertEqual([call["query"] for call in backend.calls], [QUESTION])
                self.assertEqual([stage for stage, _ in model.calls], ["plan", "read", "answer"])
                self.assertIn("plan_schema_failure", planned["failure_types"])
                self.assertTrue(planned["answer_usable"])
                self.assertEqual(planned["usage"]["model_calls"], 3)

    def test_empty_plan_queries_preserve_constraints_and_question_fallback(self):
        plan = {"constraints": list(CONSTRAINTS), "queries": []}
        _, backend, model = run("planned_single", calls=3, model=ChainModel(planner=plan))
        _, loop_backend, loop_model = run("iterative", calls=3, model=ChainModel(planner=plan))
        self.assertEqual(model.calls[:2], loop_model.calls[:2])
        self.assertEqual(backend.calls[0], loop_backend.calls[0])
        self.assertEqual(backend.calls[0]["query"], QUESTION)
        self.assertEqual(model.calls[1][1]["constraints"], CONSTRAINTS)

    def test_model_ready_precedes_mode_stop_reason(self):
        for mode in ("planned_single", "iterative"):
            with self.subTest(mode=mode):
                result, backend, model = run(mode, model=ChainModel(ready_first=True))
                self.assertEqual(result["stop_reason"], "model_ready")
                self.assertEqual([stage for stage, _ in model.calls], ["plan", "read", "answer"])
                self.assertEqual(result["state"]["rounds"], 1)
                self.assertNotIn(FOLLOWUP, [call["query"] for call in backend.calls])

    def test_read_failures_keep_original_reason_and_reserved_final(self):
        invalid_read = {"claims": "bad", "bridge_entities": [], "gaps": [],
                        "conflicts": [], "queries": [], "ready": False}
        cases = [(RuntimeError("synthetic read failure"), "read_failure"),
                 ({"claims": []}, "read_failure"), (invalid_read, "read_schema_failure"),
                 ({**invalid_read, "claims": [], "_meta": {"truncated": True}}, "read_failure")]
        for response, expected_stop in cases:
            for mode in ("planned_single", "iterative"):
                with self.subTest(response=response, mode=mode):
                    result, backend, model = run(mode, calls=3, model=ChainModel(reader=response))
                    self.assertEqual(result["stop_reason"], expected_stop)
                    self.assertEqual([stage for stage, _ in model.calls], ["plan", "read", "answer"])
                    self.assertEqual(result["usage"]["final_calls"], 1)
                    self.assertNotIn(FOLLOWUP, [call["query"] for call in backend.calls])


if __name__ == "__main__":
    unittest.main()
