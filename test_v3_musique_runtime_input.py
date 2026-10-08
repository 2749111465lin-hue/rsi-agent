"""Synthetic, zero-API MuSiQue public-input and title-to-answer checks.

The trusted engine runs in process with a scripted model. These tests establish
transport, local scope and exact offsets, not real model quality or isolation.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from code_rsi import prepare_musique as pm
from code_rsi.v3.datasets import (
    DatasetFormatError, MUSIQUE_DOCUMENT_RENDERING, adapt_musique,
    filter_documents, validate_public_task,
)
from code_rsi.v3.execution import HostBroker, Measurement
from code_rsi.v3.infrastructure import LocalCorpus
from code_rsi.v3.rag import RagEngine


def fixture(key="runtime", count=3):
    return {"id": "2hop__" + key, "question": "Who directs the observatory in example " + key + "?",
            "answer": "PRIVATE_FINAL_" + key, "answer_aliases": ["PRIVATE_ALIAS_" + key],
            "answerable": True,
            "paragraphs": [{"idx": i, "title": "Public title " + key + " " + str(i),
                            "paragraph_text": "Public unrelated passage " + key + " " + str(i) + ".",
                            "is_supporting": i < 2} for i in range(count)],
            "question_decomposition": [
                {"id": key + "_hop_" + str(i), "question": "PRIVATE_HOP_" + key + str(i),
                 "answer": "PRIVATE_SUBANSWER_" + key + str(i),
                 "paragraph_support_idx": min(i, max(0, count - 1))} for i in range(2)]}


class MuSiQueRuntimeInputTests(unittest.TestCase):
    def backend(self, task):
        def forbidden(*args):
            self.fail("question-local data must not consult any external model or corpus factory")
        measurement = Measurement(None, Path(__file__).parent / "runs", forbidden,
                                  backend_factory=forbidden)
        backend = measurement.backend(task)
        self.assertIsInstance(backend, LocalCorpus)
        self.assertEqual(backend.scope, task["question_id"])
        return backend

    def run_scripted(self, row, query, quote):
        task, _ = adapt_musique(row)
        backend = self.backend(task)
        observed = []
        case = self

        class ScriptedModel:
            def complete(self, stage, payload):
                observed.append((stage, deepcopy(payload)))
                if stage == "plan":
                    return {"constraints": [], "queries": [query]}
                if stage == "read":
                    matching = [s for s in payload["sources"] if quote in s["text"]]
                    case.assertEqual(len(matching), 1)
                    return {"claims": [{"text": "The public source identifies the director.",
                                        "citations": [{"source_id": matching[0]["source_id"], "quote": quote}]}],
                            "bridge_entities": [], "gaps": [], "conflicts": [], "queries": [], "ready": True}
                case.assertEqual(stage, "answer")
                evidence = payload["evidence"]
                case.assertEqual([e["quote"] for e in evidence], [quote])
                return {"answer": "Nera Quill", "citation_ids": [e["citation_id"] for e in evidence],
                        "evidence_sufficient": True}

        broker = HostBroker(task, backend, ScriptedModel())

        class BrokerBackend:
            def search(self, query, limit):
                return broker("search", {"query": query, "limit": limit})

        class BrokerModel:
            def complete(self, stage, payload):
                return broker("complete", {"stage": stage, "payload": payload})

        result = RagEngine(BrokerBackend(), BrokerModel()).solve({"question": task["question"]})
        self.assertEqual([stage for stage, _ in observed], ["plan", "read", "answer"])
        self.assertEqual(result["answer"], "Nera Quill")
        self.assertTrue(result["citations_valid"])
        self.assertTrue(broker.answer_origin_receipt(result["answer"])["valid"])
        receipt = broker.citation_receipt(result["answer"], result["state"]["citations"])
        self.assertTrue(receipt["valid"])
        self.assertEqual(receipt["semantic_support"], "model_assessed_only")
        self.assertNotIn("PRIVATE_", json.dumps(observed))
        return task, backend, broker, result, observed

    def test_title_only_query_reaches_read_quote_and_final_answer(self):
        row = fixture()
        title = "星 Sable Observatory"
        body = "This facility is directed by Nera Quill."
        row["paragraphs"][0].update(title=title, paragraph_text=body)
        self.assertTrue(all("Sable" not in p["paragraph_text"] for p in row["paragraphs"]))
        quote = title + "\n" + body
        task, backend, broker, result, observed = self.run_scripted(row, "Sable", quote)
        doc = task["documents"][0]
        self.assertEqual(doc["text"], quote)
        self.assertEqual([s["docid"] for s in backend.search("Sable")], [doc["docid"]])
        self.assertEqual(observed[1][1]["sources"][0]["text"], quote)
        citation = result["state"]["citations"][0]
        self.assertEqual((citation["start"], citation["end"]), (0, len(quote)))
        self.assertIn((doc["docid"], 0, len(quote), quote), broker.verified_read_quotes)
        self.assertEqual(backend.read(doc["docid"], 0, len(quote))["text"], quote)

    def test_body_quote_offsets_include_unmodified_unicode_title_and_lf(self):
        row = fixture("offsets")
        title, body, quote = " 星馆  ", "  Director: Nera Quill.", "Nera Quill"
        row["paragraphs"][0].update(title=title, paragraph_text=body)
        task, backend, _, result, _ = self.run_scripted(row, "Nera", quote)
        rendered = title + "\n" + body
        self.assertEqual(task["documents"][0]["text"], rendered)
        citation = result["state"]["citations"][0]
        start = len(title) + 1 + body.index(quote)
        self.assertEqual((citation["start"], citation["end"]), (start, start + len(quote)))
        self.assertEqual(backend.read(citation["docid"], start, start + len(quote))["text"], quote)

    def test_empty_title_adds_no_newline_and_keeps_body_offsets(self):
        row = fixture("emptytitle", 1)
        body, quote = "  星 Director: Nera Quill.", "Nera Quill"
        row["paragraphs"][0].update(title="", paragraph_text=body)
        task, _, _, result, _ = self.run_scripted(row, "Nera", quote)
        self.assertEqual(task["documents"][0]["text"], body)
        self.assertEqual(result["state"]["citations"][0]["start"], body.index(quote))

    def test_all_original_counts_stay_local_without_padding_or_foreign_documents(self):
        foreign, _ = adapt_musique(fixture("foreign", 1))
        for count in range(1, 21):
            with self.subTest(count=count):
                row = fixture("count" + str(count), count)
                task, _ = adapt_musique(row)
                validate_public_task(task)
                backend = self.backend(task)
                self.assertEqual(len(task["documents"]), count)
                self.assertEqual(set(backend.docs), {task["question_id"] + "/p/" + str(i) for i in range(count)})
                self.assertEqual(len(backend.search("Public", limit=30)), count)
                self.assertEqual(backend.search("foreign"), [])
                with self.assertRaises(ValueError):
                    backend.read(foreign["documents"][0]["docid"], 0, 1)
                with self.assertRaises(DatasetFormatError):
                    filter_documents(task, task["documents"] + foreign["documents"])
                prepared, _ = pm._adapt(row)
                self.assertEqual(prepared, task)

    def test_annotation_changes_cannot_change_public_input_or_corpus_identity(self):
        row = fixture("private")
        changed = deepcopy(row)
        changed.update(answer="DIFFERENT_PRIVATE_ANSWER", answer_aliases=["DIFFERENT_PRIVATE_ALIAS"])
        for paragraph in changed["paragraphs"]:
            paragraph["is_supporting"] = paragraph["idx"] == 2
        for step in changed["question_decomposition"]:
            step.update(answer="DIFFERENT_PRIVATE_HOP", question="DIFFERENT_PRIVATE_QUESTION",
                        paragraph_support_idx=2)
        first, first_ref = adapt_musique(row)
        second, second_ref = adapt_musique(changed)
        self.assertEqual(first, second)
        self.assertNotEqual(first_ref, second_ref)
        self.assertEqual(self.backend(first).identity, self.backend(second).identity)
        self.assertEqual(pm._adapt(row)[0], pm._adapt(changed)[0])
        self.assertNotIn("PRIVATE_", json.dumps(first))
        changed["answerable"] = False
        self.assertEqual(adapt_musique(changed)[0], first)

    def test_public_title_or_body_change_updates_question_and_source_identity(self):
        row = fixture("identity")
        original, _ = adapt_musique(row)
        for field in ("title", "paragraph_text"):
            changed = deepcopy(row)
            changed["paragraphs"][0][field] += " Changed public content."
            new, _ = adapt_musique(changed)
            self.assertNotEqual(original["question_id"], new["question_id"])
            self.assertNotEqual(self.backend(original).identity, self.backend(new).identity)
            self.assertEqual(pm._adapt(changed)[0], new)

    def test_invalid_count_or_noncontiguous_indices_rejected_by_both_entrypoints(self):
        rows = [fixture("zero", 0), fixture("over", 21)]
        for idx in (True, 1, 3):
            row = fixture("badidx", 3)
            row["paragraphs"][0]["idx"] = idx
            rows.append(row)
        for row in rows:
            with self.subTest(row=row["id"]), self.assertRaises(DatasetFormatError):
                adapt_musique(row)
            with self.subTest(row=row["id"]), self.assertRaises(pm.PanelError):
                pm._adapt(row)

    def test_public_panel_preparation_retains_actual_inputs_and_count_distribution(self):
        originals = [fixture("trainone", 1), fixture("trainnineteen", 19), fixture("devtwenty", 20)]
        expected = {adapt_musique(row)[0]["question_id"]: adapt_musique(row)[0] for row in originals}
        root = Path(__file__).resolve().parent / "runs"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="musique_runtime_inputs_", dir=root) as directory:
            train, dev = (Path(directory) / ("musique_ans_v1.0_" + split + ".jsonl") for split in ("train", "dev"))
            train.write_text("".join(json.dumps(row) + "\n" for row in originals[:2]), encoding="utf-8")
            dev.write_text(json.dumps(originals[2]) + "\n", encoding="utf-8")
            bundle = pm.prepare_panels(train, dev, seed="runtime-input-contract",
                                       quotas={role: {2: 1, 3: 0, 4: 0} for role in pm.ROLES})
        actual = [task for tasks in bundle["public_panels"].values() for task in tasks]
        self.assertEqual({task["question_id"]: task for task in actual}, expected)
        self.assertEqual(sorted(len(task["documents"]) for task in actual), [1, 19, 20])
        self.assertEqual(bundle["manifest"]["document_rendering"], MUSIQUE_DOCUMENT_RENDERING)
        self.assertTrue(bundle["manifest"]["original_paragraph_count_preserved"])
        for role, tasks in bundle["public_panels"].items():
            self.assertEqual(bundle["manifest"]["paragraph_count_distribution"][role], {len(tasks[0]["documents"]): 1})


if __name__ == "__main__":
    unittest.main()
