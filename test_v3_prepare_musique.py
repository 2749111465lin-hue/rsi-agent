"""Synthetic, zero-API diagnostics for the offline MuSiQue panel builder."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from code_rsi import prepare_musique as pm


def fixture(key, hop=2):
    return {"id": f"{hop}hop__{key}", "question": f"Which object belongs to example {key}?",
            "answer": "PRIVATE_FINAL_" + key, "answer_aliases": ["PRIVATE_ALIAS_" + key],
            "answerable": True,
            "paragraphs": [{"idx": i, "title": f"Title {key} {i}",
                            "paragraph_text": f"Public passage for {key} position {i}.",
                            "is_supporting": i < hop} for i in range(20)],
            "question_decomposition": [{"id": f"{key}:single:{i}",
                                       "question": f"PRIVATE_SUBQUESTION_{key}_{i}",
                                       "answer": f"PRIVATE_SUBANSWER_{key}_{i}",
                                       "paragraph_support_idx": i} for i in range(hop)]}


def quotas(n=1):
    return {role: {2: n, 3: 0, 4: 0} for role in pm.ROLES}


class PrepareMuSiQueTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parent / "runs"
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="test_musique_panels_", dir=base)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train = self.root / "musique_ans_v1.0_train.jsonl"
        self.dev = self.root / "musique_ans_v1.0_dev.jsonl"
        self.run_root = self.root / "runs"
        self.path_patch = patch.object(pm, "RUNS_ROOT", self.run_root)
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)

    def inputs(self, train=None, dev=None):
        for path, rows in ((self.train, train if train is not None else [fixture("trainA"), fixture("trainB")]),
                           (self.dev, dev if dev is not None else [fixture("devA")])):
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def prepare(self, q=None, seed="diagnostic-seed"):
        return pm.prepare_panels(self.train, self.dev, seed=seed, quotas=q or quotas())

    def test_balanced_roles_private_separation_and_local_corpora(self):
        train = [fixture(f"train{hop}_{i}", hop) for hop in (2, 3, 4) for i in range(2)]
        dev = [fixture(f"dev{hop}", hop) for hop in (2, 3, 4)]
        self.inputs(train, dev)
        q = {role: {2: 1, 3: 1, 4: 1} for role in pm.ROLES}
        bundle = self.prepare(q)
        all_ids = []
        for role in pm.ROLES:
            tasks = bundle["public_panels"][role]
            self.assertEqual(len(tasks), 3)
            self.assertEqual(set(bundle["private_references"][role]), {t["question_id"] for t in tasks})
            for task in tasks:
                all_ids.append(task["question_id"])
                self.assertEqual(task["corpus_scope"], "question_local")
                self.assertIsNone(task["corpus_ref"])
                self.assertEqual(len(task["documents"]), 20)
                self.assertTrue(all(d["docid"].startswith(task["question_id"] + "/p/") for d in task["documents"]))
                self.assertNotIn("PRIVATE_", json.dumps(task))
                ref = bundle["private_references"][role][task["question_id"]]
                self.assertTrue(ref["support_annotation_available"])
                self.assertIn("question_decomposition", ref)
                self.assertEqual(ref["pair_group_id"], ref["source_question_id"])
            self.assertEqual(bundle["manifest"]["audit"][role]["selected"], q[role])
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertNotIn("PRIVATE_", json.dumps(bundle["manifest"]))
        self.assertEqual(bundle["manifest"]["selection_order"], ["D_report", "D_fit", "D_select"])

    def test_same_seed_reproducible_and_input_order_not_a_sampling_signal(self):
        train = [fixture("train" + str(i)) for i in range(12)]
        dev = [fixture("dev" + str(i)) for i in range(6)]
        self.inputs(train, dev)
        first = self.prepare()
        self.assertEqual(first, self.prepare())
        self.inputs(list(reversed(train)), list(reversed(dev)))
        second = self.prepare()
        self.assertEqual(first["public_panels"], second["public_panels"])
        self.assertEqual(first["private_references"], second["private_references"])
        self.assertNotEqual(first["manifest"]["sources"]["train"]["sha256"], second["manifest"]["sources"]["train"]["sha256"])
        self.assertNotEqual(first["public_panels"], self.prepare(seed="other-seed")["public_panels"])

    def test_report_selection_does_not_depend_on_train_candidates(self):
        dev = [fixture("dev" + str(i)) for i in range(6)]
        self.inputs([fixture("trainA"), fixture("trainB")], dev)
        first = self.prepare()["public_panels"]["D_report"]
        self.inputs([fixture("newA"), fixture("newB")], dev)
        self.assertEqual(first, self.prepare()["public_panels"]["D_report"])

    def assert_cross_role_filter(self, kind, transform):
        report, blocked = fixture("devA"), fixture("blocked")
        transform(report, blocked)
        self.inputs([blocked, fixture("only_clean")], [report])
        with self.assertRaises(pm.PanelQuotaError) as raised:
            self.prepare()
        audits = raised.exception.manifest["audit"].values()
        conflicts = [c for audit in audits for row in audit["rejected_candidates"] for c in row["conflicts"]]
        matching = [c for c in conflicts if c["reason"] == "cross_role_" + kind]
        self.assertTrue(matching, kind)
        self.assertTrue(any(c["other_role"] == "D_report" for c in matching))
        self.assertNotIn("PRIVATE_", json.dumps(raised.exception.manifest))
        for audit in audits:
            self.assertEqual(audit["rejected_count"], len(audit["rejected_candidates"]))

    def test_question_identity_cross_role(self):
        self.assert_cross_role_filter("question_id", lambda report, blocked: (blocked.clear(), blocked.update(deepcopy(report))))

    def test_source_pair_cross_role_even_when_public_question_ids_differ(self):
        self.assert_cross_role_filter("source_question_pair", lambda report, blocked: blocked.update(id=report["id"]))

    def test_normalized_question_cross_role(self):
        def edit(report, blocked):
            report["question"] = "Who owns the \u212a artifact?"
            blocked["question"] = "  WHO  OWNS THE K artifact?  "
        self.assert_cross_role_filter("normalized_question", edit)

    def test_singlehop_integer_string_identity_cross_role(self):
        def edit(report, blocked):
            report["question_decomposition"][0]["id"] = 17
            blocked["question_decomposition"][0]["id"] = "17"
        self.assert_cross_role_filter("singlehop_id", edit)

    def test_normalized_supporting_paragraph_cross_role(self):
        def edit(report, blocked):
            report["paragraphs"][0]["paragraph_text"] = "A \uff21 document."
            blocked["paragraphs"][0]["paragraph_text"] = "  a  A   DOCUMENT. "
        self.assert_cross_role_filter("support_paragraph", edit)

    def test_normalized_nonempty_subanswer_cross_role(self):
        def edit(report, blocked):
            report["question_decomposition"][0]["answer"] = "The OPAL!!!"
            blocked["question_decomposition"][0]["answer"] = "opal"
        self.assert_cross_role_filter("subanswer", edit)

    def test_train_fit_select_singlehop_overlap_is_blocked(self):
        train = [fixture("train" + str(i)) for i in range(3)]
        for row in train:
            row["question_decomposition"][0]["id"] = "shared-training-component"
        self.inputs(train)
        with self.assertRaises(pm.PanelQuotaError) as raised:
            self.prepare()
        manifest = raised.exception.manifest
        self.assertEqual(manifest["failed_role"], "D_select")
        self.assertEqual(manifest["audit"]["D_select"]["selected"][2], 0)
        self.assertEqual(manifest["audit"]["D_select"]["filter_hits"]["cross_role_singlehop_id"], 3)

    def test_prepared_references_work_with_existing_official_rule_metrics(self):
        from code_rsi.v3.task_metrics import score_task
        self.inputs()
        bundle = self.prepare()
        for role in pm.ROLES:
            task = bundle["public_panels"][role][0]
            ref = bundle["private_references"][role][task["question_id"]]
            prediction = {"question_id": task["question_id"], "answer": ref["answers"][0],
                          "citations": [{"docid": docid} for docid in ref["supporting_docids"]]}
            result = score_task(prediction, ref, task=task)
            self.assertEqual(result["answer_f1"], 1.0)
            self.assertEqual(result["support_f1"], 1.0)
            self.assertEqual(result["metric_status"]["answer_f1"], "ok")
            self.assertFalse(result["proxy_metrics"])

    def test_normalized_empty_subanswers_do_not_create_false_collision(self):
        train, dev = [fixture("trainA"), fixture("trainB")], [fixture("devA")]
        for row in train + dev:
            row["question_decomposition"][0]["answer"] = "The!!!"
        self.inputs(train, dev)
        self.assertEqual(self.prepare()["manifest"]["status"], "ready")

    def test_shared_distractors_are_not_gold_support_collisions(self):
        train, dev = [fixture("trainA"), fixture("trainB")], [fixture("devA")]
        for row in train + dev:
            row["paragraphs"][19]["paragraph_text"] = "Common distracting material."
        self.inputs(train, dev)
        self.assertEqual(self.prepare()["manifest"]["status"], "ready")

    def test_unknown_full_and_incomplete_annotations_fail_closed(self):
        edits = [lambda row: row.update(answerable=False),
                 lambda row: row.pop("question_decomposition"),
                 lambda row: row["paragraphs"][3].pop("is_supporting"),
                 lambda row: row["question_decomposition"][0].update(paragraph_support_idx=19),
                 lambda row: row.update(answer_aliases=None)]
        for edit in edits:
            with self.subTest(edit=edit):
                bad = fixture("bad")
                edit(bad)
                self.inputs([bad, fixture("clean")])
                with self.assertRaises(pm.PanelError) as raised:
                    self.prepare()
                self.assertNotIn("PRIVATE_", str(raised.exception))

    def test_conflicting_official_source_id_rejected_before_selection(self):
        a, b = fixture("same"), fixture("other")
        b["id"] = a["id"]
        self.inputs([a, b])
        with self.assertRaisesRegex(pm.PanelError, "conflicting rows"):
            self.prepare()

    def test_exact_duplicate_rows_do_not_satisfy_multiple_quota_slots(self):
        a = fixture("same")
        self.inputs([a, deepcopy(a)])
        with self.assertRaises(pm.PanelQuotaError):
            self.prepare()

    def test_file_changes_between_scan_and_materialization_fail(self):
        self.inputs()
        original = pm._materialize
        def changed(source, selected):
            with self.train.open("a", encoding="utf-8") as stream:
                stream.write("\n")
            return original(source, selected)
        with patch.object(pm, "_materialize", side_effect=changed):
            with self.assertRaisesRegex(pm.PanelError, "input bytes changed"):
                self.prepare()

    def test_input_filename_and_quota_contracts(self):
        self.inputs()
        wrong = self.root / "arbitrary.jsonl"
        wrong.write_bytes(self.train.read_bytes())
        with self.assertRaises(pm.PanelError):
            pm.prepare_panels(wrong, self.dev, seed="x", quotas=quotas())
        for bad in ({}, {**quotas(), "D_fit": {2: True, 3: 0, 4: 0}},
                    {**quotas(), "D_fit": {2: 0, 3: 0, 4: 0}}):
            with self.assertRaises(pm.PanelError):
                self.prepare(q=bad if bad else {"unexpected": {}})
        with self.assertRaises(pm.PanelError):
            self.prepare(seed="")

    def test_duplicate_json_fields_are_rejected(self):
        self.inputs()
        self.train.write_text('{"id":"one","id":"two"}\n', encoding="utf-8")
        with self.assertRaises(pm.PanelError):
            self.prepare()

    def test_write_hashes_new_directory_and_no_overwrite(self):
        self.inputs()
        bundle = self.prepare()
        out = pm.write_panels(bundle, self.run_root / "panel")
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["files"]), 6)
        for relative, info in manifest["files"].items():
            raw = (out / relative).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), info["sha256"])
            self.assertEqual(info["private"], relative.endswith("private_references.json"))
        before = (out / "manifest.json").read_bytes()
        with self.assertRaises(pm.PanelError):
            pm.write_panels(bundle, out)
        self.assertEqual(before, (out / "manifest.json").read_bytes())
        for unsafe in (self.root / "outside", self.run_root, self.run_root / ".." / "escaped"):
            with self.assertRaises(pm.PanelError):
                pm.write_panels(bundle, unsafe)
        corrupted = deepcopy(bundle)
        corrupted["public_panels"]["D_fit"][0]["answer"] = "private"
        with self.assertRaises(ValueError):
            pm.write_panels(corrupted, self.run_root / "corrupted")
        self.assertFalse((self.run_root / "corrupted").exists())

    def cli(self, out, quota="1,0,0"):
        return pm.main(["--train", str(self.train), "--dev", str(self.dev), "--out", str(out),
                        "--seed", "explicit-seed", "--fit-quota", quota,
                        "--select-quota", quota, "--report-quota", quota])

    def test_cli_stdout_only_summary_and_private_data_remains_local(self):
        self.inputs()
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = self.cli(self.run_root / "success")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["counts"], {role: 1 for role in pm.ROLES})
        self.assertNotIn("PRIVATE_", stdout.getvalue() + stderr.getvalue())
        self.assertNotIn("Which object", stdout.getvalue())

    def test_cli_shortfall_saves_only_failure_audit_and_never_lowers_quota(self):
        self.inputs()
        out = self.run_root / "shortfall"
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = self.cli(out, quota="2,0,0")
        self.assertEqual(code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual([p.name for p in out.iterdir()], ["manifest.json"])
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "quota_shortfall")
        self.assertEqual(manifest["failed_role"], "D_report")
        self.assertEqual(manifest["audit"]["D_report"]["requested"]["2"], 2)
        self.assertEqual(manifest["audit"]["D_report"]["selected"]["2"], 1)
        self.assertNotIn("PRIVATE_", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
