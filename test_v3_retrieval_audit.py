"""ID-only synthetic boundaries for the offline document recall audit."""
from copy import deepcopy
import json
import unittest

from code_rsi.retrieval_audit import RetrievalAuditError, audit_document_recall


class RetrievalAuditTests(unittest.TestCase):
    def labels(self):
        return {"question_id": "q", "evidence_docids": ["a", "b"], "gold_docids": ["a"]}

    def row(self, **kwargs):
        value = {"question_id": "q", "trajectory_id": "t", "complete": True,
                 "layers": {"retrieved": {"complete": True, "docids": ["a", "b", "unjudged"]},
                            "presented": {"complete": True, "docids": ["b"]},
                            "verified_quotes": {"complete": True, "docids": []}}}
        value.update(kwargs)
        return value

    def audit(self, row=None, labels=None):
        return audit_document_recall("q", self.labels() if labels is None else labels,
                                     [self.row() if row is None else row])

    def test_each_layer_and_label_denominator_remain_distinct(self):
        layers = self.audit()["rows"][0]["layers"]
        self.assertEqual(layers["retrieved"]["evidence"]["document_recall"], 1.)
        self.assertTrue(layers["retrieved"]["evidence"]["all_hit"])
        self.assertEqual(layers["presented"]["evidence"]["document_recall"], .5)
        self.assertTrue(layers["presented"]["evidence"]["any_hit"])
        self.assertFalse(layers["presented"]["evidence"]["all_hit"])
        self.assertEqual(layers["presented"]["gold"]["document_recall"], 0.)
        self.assertTrue(layers["verified_quotes"]["evidence"]["eligible"])
        self.assertFalse(layers["verified_quotes"]["evidence"]["any_hit"])

    def test_duplicates_do_not_weight_recall_and_unjudged_is_not_wrong(self):
        row = self.row()
        row["layers"]["retrieved"]["docids"] *= 3
        labels = self.labels()
        labels["evidence_docids"] *= 2
        report = self.audit(row, labels)
        layer = report["rows"][0]["layers"]["retrieved"]
        self.assertEqual(layer["observed_doc_count"], 3)
        self.assertEqual(layer["evidence"]["hit_count"], 2)
        self.assertEqual(layer["evidence"]["document_recall"], 1.)
        for field in ("precision", "false_positive", "f1", "recall_at_10"):
            self.assertNotIn(field, json.dumps(report))

    def test_missing_or_empty_annotation_is_unknown_not_vacuous_success(self):
        for ids in (None, []):
            labels = self.labels()
            labels["evidence_docids"] = ids
            report = self.audit(labels=labels)["rows"][0]["layers"]["retrieved"]
            self.assertFalse(report["evidence"]["eligible"])
            self.assertIsNone(report["evidence"]["all_hit"])
            self.assertIsNone(report["evidence"]["document_recall"])
            self.assertEqual(report["gold"]["document_recall"], 1.)
        labels = self.labels()
        del labels["evidence_docids"]
        self.assertEqual(self.audit(labels=labels)["annotation_status"]["evidence"]["reason"], "missing_annotation")

    def test_incomplete_trajectory_invalidates_all_full_union_metrics(self):
        for complete in (False, None):
            report = self.audit(self.row(complete=complete))
            for layer in report["rows"][0]["layers"].values():
                self.assertFalse(layer["observable"])
                for name in ("evidence", "gold"):
                    for field in ("hit_count", "document_recall", "any_hit", "all_hit"):
                        self.assertIsNone(layer[name][field])
            self.assertEqual(report["rows"][0]["layers"]["retrieved"]["observed_doc_count"], 3)

    def test_missing_layer_does_not_become_empty_success_or_poison_other_layers(self):
        row = self.row()
        del row["layers"]["verified_quotes"]
        result = self.audit(row)["rows"][0]["layers"]
        self.assertTrue(result["retrieved"]["observable"])
        self.assertFalse(result["verified_quotes"]["observable"])
        self.assertIsNone(result["verified_quotes"]["evidence"]["document_recall"])

    def test_layer_requires_explicit_complete_and_present_docids(self):
        for replacement in ({"docids": ["a"]}, {"complete": False, "docids": ["a"]}, {"complete": True}):
            row = self.row()
            row["layers"]["presented"] = replacement
            result = self.audit(row)["rows"][0]["layers"]["presented"]
            self.assertFalse(result["observable"])
            self.assertIsNone(result["gold"]["document_recall"])

    def test_wrong_question_identity_and_duplicate_trajectory_are_rejected(self):
        for labels, rows in ((dict(self.labels(), question_id="other"), [self.row()]),
                             (self.labels(), [self.row(question_id="other")]),
                             (self.labels(), [self.row(), self.row()])):
            with self.assertRaises(RetrievalAuditError):
                audit_document_recall("q", labels, rows)

    def test_truthy_completeness_and_malformed_ids_cannot_be_certified(self):
        for invalid in (1, "true", []):
            with self.assertRaises(RetrievalAuditError):
                self.audit(self.row(complete=invalid))
            row = self.row()
            row["layers"]["retrieved"]["complete"] = invalid
            with self.assertRaises(RetrievalAuditError):
                self.audit(row)
        for ids in ("abc", {"a": 1}, [1], [""], [" "]):
            row = self.row()
            row["layers"]["retrieved"]["docids"] = ids
            with self.assertRaises(RetrievalAuditError):
                self.audit(row)

    def test_no_success_subset_aggregation_and_no_reference_text_echo(self):
        labels = self.labels()
        labels["reference_text"] = "PRIVATE_SYNTHETIC_SENTINEL"
        rows = [self.row(), self.row(trajectory_id="failed", complete=False)]
        before = deepcopy((labels, rows))
        result = audit_document_recall("q", labels, rows)
        self.assertEqual((labels, rows), before)
        self.assertEqual(len(result["rows"]), 2)
        self.assertNotIn("PRIVATE_SYNTHETIC_SENTINEL", json.dumps(result))
        self.assertEqual(set(result), {"schema", "question_id", "scope", "annotation_status", "rows"})
        self.assertEqual(audit_document_recall("q", labels, [])["rows"], [])


if __name__ == "__main__":
    unittest.main()
