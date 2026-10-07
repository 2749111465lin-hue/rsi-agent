"""Working-window regressions from a live ingress bug; all fixtures synthetic."""
import copy
import json
import unittest

from code_rsi.v3 import rag as prototype

QUESTION = "Where was the captain of the fictional Aurora expedition born?"
BRIDGE = "The captain of the Aurora expedition was Mira Vale."
ANSWER = "Mira Vale was born in Northport."


def source(docid, text, size=None, start=0):
    if size is not None:
        text = text + "x" * (size - len(text))
    return {"docid": docid, "text": text, "start": start, "end": start + len(text)}


class Backend:
    def __init__(self, results):
        self.results, self.calls = results, []

    def search(self, query, limit=5):
        self.calls.append((query, limit))
        return copy.deepcopy(self.results.get(query, [])[:limit])


class Model:
    def __init__(self, initial, reader, final=None):
        self.initial, self.reader, self.final = initial, reader, final
        self.calls = []

    def complete(self, stage, payload):
        self.calls.append((stage, copy.deepcopy(payload)))
        if stage == "plan":
            return {"constraints": ["identify captain then birthplace"], "queries": self.initial}
        if stage == "read":
            return self.reader(payload)
        if self.final:
            return self.final(payload)
        return {"answer": "Insufficient information", "citation_ids": [], "evidence_sufficient": False}


def read_reply(payload, *, quote=None, queries=None, ready=False, conflicts=None, bridges=None):
    claims = []
    if quote is not None:
        found = next((s for s in payload["sources"] if quote in s["text"]), None)
        if found:
            claims = [{"text": quote, "citations": [{"source_id": found["source_id"], "quote": quote}]}]
    return {"claims": claims, "bridge_entities": bridges or [], "gaps": [],
            "conflicts": conflicts or [], "queries": queries or [], "ready": ready}


def two_hop(engine=prototype.RagEngine, *, extra_old_citation=False, conflict=False):
    initial = [source("bridge", BRIDGE, 6000, 120)]
    initial += [source("distractor-" + str(i), "Unrelated synthetic filler " + str(i), 6000) for i in range(4)]
    backend = Backend({"captain search": initial, QUESTION: initial,
                       "Mira Vale birthplace": [source("birthplace", ANSWER, 6000, 400)]})

    def reader(payload):
        if payload["round"] == 1:
            result = read_reply(payload, quote=BRIDGE, queries=["Mira Vale birthplace"], bridges=["Mira Vale"],
                                conflicts=["Unresolved synthetic date conflict"] if conflict else [])
            result["gaps"] = ["birthplace missing"]
            return result
        result = read_reply(payload, quote=ANSWER, ready=True, bridges=["Mira Vale"])
        if extra_old_citation:
            result["claims"].append({"text": "stale source claim", "citations": [{"source_id": "s1", "quote": BRIDGE}]})
        return result

    def final(payload):
        evidence = {item["quote"]: item["citation_id"] for item in payload["evidence"]}
        sufficient = BRIDGE in evidence and ANSWER in evidence
        return {"answer": "Northport" if sufficient else "Insufficient information",
                "citation_ids": [evidence[q] for q in (BRIDGE, ANSWER)] if sufficient else [],
                "evidence_sufficient": sufficient}

    model = Model(["captain search"], reader, final)
    result = engine(backend, model).solve({"question": QUESTION})
    return result, backend, model


class WorkingWindowTests(unittest.TestCase):
    def test_first_full_window_then_second_hop_answer_and_old_quote_survive(self):
        result, backend, model = two_hop()
        reads = [p for stage, p in model.calls if stage == "read"]
        self.assertEqual(sum(len(s["text"]) for s in reads[0]["sources"]), 24000)
        self.assertEqual([s["docid"] for s in reads[1]["sources"]], ["birthplace"])
        self.assertEqual(reads[1]["known_claims"][0]["text"], BRIDGE)
        self.assertEqual(reads[1]["bridge_entities"], ["Mira Vale"])
        self.assertEqual(reads[1]["gaps"], ["birthplace missing"])
        self.assertEqual(result["answer"], "Northport")
        self.assertTrue(result["citations_valid"])
        final = model.calls[-1][1]
        self.assertEqual([c["quote"] for c in final["evidence"]], [BRIDGE, ANSWER])
        self.assertEqual([c["start"] for c in final["evidence"]], [120, 400])
        self.assertEqual(result["citation_ids"], ["e1", "e2"])
        self.assertEqual([s["docid"] for s in result["state"]["sources"]], ["birthplace"])
        self.assertEqual(result["correctness"], "unknown")
        json.dumps(result, allow_nan=False)


    def test_source_ids_never_reused_across_windows(self):
        result, _, model = two_hop()
        ids = [s["source_id"] for stage, p in model.calls if stage == "read" for s in p["sources"]]
        self.assertEqual(ids, ["s1", "s2", "s3", "s4", "s5"])
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual([c["source_id"] for c in result["state"]["citations"]], ["s1", "s5"])

    def test_each_source_window_and_entire_payload_keep_original_caps(self):
        result, _, model = two_hop()
        reads = [p for stage, p in model.calls if stage == "read"]
        for payload in reads:
            self.assertLessEqual(sum(len(s["text"]) for s in payload["sources"]), 24000)
            self.assertTrue(all(len(s["text"]) <= 6000 for s in payload["sources"]))
        for _, payload in model.calls:
            self.assertLessEqual(len(json.dumps(payload, ensure_ascii=False)), 64000)
        self.assertEqual(result["usage"]["source_chars"], 30000)
        self.assertEqual(result["usage"]["context_source_chars"], 6000)
        self.assertEqual(result["usage"]["peak_context_source_chars"], 24000)

    def test_two_queries_interleave_instead_of_first_query_consuming_window(self):
        backend = Backend({q: [source(q + str(i), q + str(i), 6000) for i in range(5)] for q in ("left", "right")})
        model = Model(["left", "right"], lambda p: read_reply(p, ready=True))
        result = prototype.RagEngine(backend, model).solve({"question": QUESTION})
        payload = next(p for stage, p in model.calls if stage == "read")
        self.assertEqual([s["docid"] for s in payload["sources"]], ["left0", "right0", "left1", "right1"])
        self.assertEqual(sum(len(s["text"]) for s in payload["sources"]), 24000)
        self.assertEqual(result["usage"]["search_calls"], 2)

    def test_large_first_result_still_reserves_a_turn_for_second_query(self):
        backend = Backend({q: [source(q, q, 24000)] for q in ("left", "right")})
        model = Model(["left", "right"], lambda p: read_reply(p, ready=True))
        prototype.RagEngine(backend, model, config={"max_source_chars": 24000}).solve({"question": QUESTION})
        payload = next(p for stage, p in model.calls if stage == "read")
        self.assertEqual([(s["docid"], len(s["text"])) for s in payload["sources"]], [("left", 12000), ("right", 12000)])

    def recovery_fixture(self, first, second, cap):
        backend = Backend({"first": first, "next": second})
        def reader(payload):
            quote = payload["sources"][0]["text"][:100] if payload["sources"] else None
            return read_reply(payload, quote=quote, queries=["next"], ready=payload["round"] == 2)
        model = Model(["first"], reader)
        result = prototype.RagEngine(backend, model, config={"max_context_chars": cap, "max_source_chars": cap}).solve({"question": QUESTION})
        return result, [p for stage, p in model.calls if stage == "read"]

    def test_unpresented_candidate_is_not_permanently_deduplicated(self):
        first = source("first", "FIRST WINDOW", 12)
        late = source("late", "LATER WINDOW", 12)
        result, reads = self.recovery_fixture([first, late], [late], 12)
        self.assertEqual([s["docid"] for s in reads[0]["sources"]], ["first"])
        self.assertEqual([s["docid"] for s in reads[1]["sources"]], ["late"])
        self.assertEqual(result["usage"]["source_chars"], 24)

    def test_truncated_raw_result_can_return_with_its_previously_unread_tail(self):
        raw = source("partial", "UNREADTAIL!", 10)
        _, reads = self.recovery_fixture([source("filler", "F" * 10), raw], [raw], 12)
        self.assertEqual(reads[0]["sources"][1]["text"], "UN")
        self.assertEqual(reads[1]["sources"][0]["text"], "UNREADTAIL!")
        self.assertEqual(reads[1]["sources"][0]["source_id"], "s3")

    def test_already_presented_identical_span_is_deduplicated_for_new_material(self):
        seen = source("seen", "SEEN", 12)
        fresh = source("fresh", "FRESH", 12)
        _, reads = self.recovery_fixture([seen], [seen, fresh], 12)
        self.assertEqual([s["docid"] for s in reads[1]["sources"]], ["fresh"])

    def test_empty_new_round_does_not_represent_evicted_old_full_text(self):
        result, reads = self.recovery_fixture([source("first", "FIRST", 12)], [], 12)
        self.assertEqual(reads[1]["sources"], [])
        self.assertEqual(len(reads[1]["known_claims"]), 1)
        self.assertEqual(len(result["state"]["citations"]), 1)

    def test_old_source_id_cannot_ground_new_read_but_existing_quote_is_retained(self):
        result, _, model = two_hop(extra_old_citation=True)
        self.assertIn("invalid_quote", result["failure_types"])
        self.assertNotIn("stale source claim", [c["text"] for c in result["state"]["claims"]])
        self.assertEqual([c["quote"] for c in model.calls[-1][1]["evidence"]], [BRIDGE, ANSWER])

    def test_open_conflicts_and_grounded_bridges_survive_window_rotation(self):
        result, _, model = two_hop(conflict=True)
        self.assertEqual(model.calls[-1][1]["conflicts"], ["Unresolved synthetic date conflict"])
        self.assertEqual(result["state"]["bridge_entities"], ["Mira Vale"])
        self.assertNotIn("ungrounded_bridge_entity", result["failure_types"])

    def test_same_returned_window_can_be_reread_for_previously_unextracted_fact(self):
        row = source("both-facts", BRIDGE + " " + ANSWER, 6000, 100)
        backend = Backend({"initial": [row], "recheck birthplace": [row]})
        def reader(payload):
            return read_reply(payload, quote=BRIDGE if payload["round"] == 1 else ANSWER,
                              queries=["recheck birthplace"], ready=payload["round"] == 2)
        def final(payload):
            quotes = {c["quote"]: c["citation_id"] for c in payload["evidence"]}
            sufficient = BRIDGE in quotes and ANSWER in quotes
            return {"answer": "Northport" if sufficient else "Insufficient information",
                    "citation_ids": list(quotes.values()), "evidence_sufficient": sufficient}
        model = Model(["initial"], reader, final)
        result = prototype.RagEngine(backend, model).solve({"question": QUESTION})
        reads = [p for stage, p in model.calls if stage == "read"]
        self.assertEqual(reads[0]["sources"], reads[1]["sources"])
        self.assertEqual(result["answer"], "Northport")
        self.assertEqual([c["quote"] for c in result["state"]["citations"]], [BRIDGE, ANSWER])
        windows = [e for e in result["trace"] if e["stage"] == "context_window"]
        self.assertEqual(windows[1]["novel_source_ids"], [])
        self.assertEqual(windows[1]["reused_source_ids"], ["s1"])
        self.assertEqual(result["usage"]["model_calls"], 4)

    def test_fallback_alone_does_not_count_as_evidence_progress(self):
        row = source("same", "SAME FACT", 12)
        backend = Backend({"initial": [row], "recheck": [row], "unused": [row]})
        def reader(payload):
            return read_reply(payload, quote="SAME FACT", queries=["recheck" if payload["round"] == 1 else "unused"])
        model = Model(["initial"], reader)
        result = prototype.RagEngine(backend, model, config={"max_context_chars": 12}).solve({"question": QUESTION})
        self.assertEqual(result["stop_reason"], "no_evidence_progress")
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(len(result["state"]["citations"]), 1)
        self.assertEqual(result["state"]["consecutive_stagnant_rounds"], 1)
        second = [e for e in result["trace"] if e["stage"] == "context_window"][1]
        self.assertEqual(second["novel_source_ids"], [])
        self.assertEqual(second["reused_source_ids"], ["s1"])

    def test_fresh_window_wins_over_seen_fallback(self):
        seen = source("seen", "OLD FACT", 12)
        fresh = source("fresh", "NEW FACT", 12)
        result, reads = self.recovery_fixture([seen], [seen, fresh], 12)
        self.assertEqual([s["docid"] for s in reads[1]["sources"]], ["fresh"])
        second = [e for e in result["trace"] if e["stage"] == "context_window"][1]
        self.assertEqual(second["reused_source_ids"], [])
        self.assertEqual(second["novel_source_ids"], ["s2"])

    def test_empty_retrieval_never_activates_seen_fallback(self):
        result, reads = self.recovery_fixture([source("seen", "OLD FACT", 12)], [], 12)
        self.assertEqual(reads[1]["sources"], [])
        second = [e for e in result["trace"] if e["stage"] == "context_window"][1]
        self.assertEqual(second["reused_source_ids"], [])
        self.assertEqual(second["novel_source_ids"], [])

    def test_fallback_full_window_stays_within_cap_and_reuses_only_identical_ids(self):
        rows = [source("d" + str(i), "FACT " + str(i), 6000) for i in range(4)]
        result, reads = self.recovery_fixture(rows, rows, 24000)
        self.assertEqual(reads[0]["sources"], reads[1]["sources"])
        self.assertEqual(sum(len(s["text"]) for s in reads[1]["sources"]), 24000)
        second = [e for e in result["trace"] if e["stage"] == "context_window"][1]
        self.assertEqual(second["reused_source_ids"], ["s1", "s2", "s3", "s4"])
        self.assertEqual(second["novel_source_ids"], [])

    def test_changed_span_never_reuses_old_source_id(self):
        result, reads = self.recovery_fixture([source("doc", "OLD", 12)], [source("doc", "NEW", 12)], 12)
        self.assertEqual(reads[0]["sources"][0]["source_id"], "s1")
        self.assertEqual(reads[1]["sources"][0]["source_id"], "s2")
        second = [e for e in result["trace"] if e["stage"] == "context_window"][1]
        self.assertEqual(second["reused_source_ids"], [])



if __name__ == "__main__":
    unittest.main(verbosity=2)
