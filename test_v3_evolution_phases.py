"""Synthetic phase-boundary and recovery tests; no API or candidate execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from code_rsi.budget import digest, save
from code_rsi.v3 import evolution
from code_rsi.v3.datasets import adapt_multihop
from test_v3_evolution_recovery import Audit, Developer, FakeMeasurement, InjectedCrash


ROLES = ("D_fit", "D_select", "D_report")
PHASES = ("search", "select", "report")


class PhaseMeasurement(FakeMeasurement):
    """Reuse persistent fixture cells and crash only after new work is durable."""
    def run(self, node, tasks, references, *, role, bank, repeats):
        before = len(self.audit.computed)
        result = super().run(node, tasks, references, role=role, bank=bank, repeats=repeats)
        if (getattr(self.audit, "crash_after_role", None) == role
                and len(self.audit.computed) > before):
            self.audit.crash_after_role = None
            raise InjectedCrash("after durable " + role + " fixture measurement")
        return result


class EvolutionPhaseTests(unittest.TestCase):
    def setUp(self):
        runs = Path(__file__).parent / "runs"
        runs.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="v3_phases_test_", dir=runs)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.out = self.base / "execution"
        self.audit = Audit()
        self.developer = Developer()
        self.loaded = []
        self.allowed_roles = set(ROLES)
        self.manifest, self.panels, self.references = self.inputs()

    def inputs(self, *, search_only=False):
        roles = ROLES[:1] if search_only else ROLES
        panels, references, bindings, groups = {}, {}, {}, {}
        for role in roles:
            task, reference = adapt_multihop({
                "id": role, "query": "Synthetic phase question " + role,
                "answer": "PRIVATE_PHASE_FIXTURE_" + role})
            panels[role] = [task]
            references[role] = {task["question_id"]: reference}
            path = self.base / (role + "-references.json")
            save(path, references[role])
            bindings[role] = {"path": str(path.resolve()),
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            groups[role] = {task["question_id"]: None}
        manifest = {
            "expansions": 1, "metric": "em", "select_candidates": 3, "repeats": 1,
            "root_config": {"mode": "iterative"}, "limits": {"max_models": 7},
            "synthetic": True,
            "lifecycle": {"schema": "rag-rsi-evolution-phases-1",
                          "phase_order": list(PHASES[:1] if search_only else PHASES),
                          "reference_bindings": bindings, "reference_groups": groups}}
        return manifest, panels, references

    def loader(self, role):
        if role not in self.allowed_roles:
            raise AssertionError("forbidden private-reference access: " + role)
        self.loaded.append(role)
        path = self.manifest["lifecycle"]["reference_bindings"][role]["path"]
        return json.loads(Path(path).read_bytes())

    def runner(self, *, loader=None, supplied_references=None, legacy=False):
        manifest = deepcopy(self.manifest)
        if legacy:
            manifest.pop("lifecycle")
        references = deepcopy(self.references) if legacy else (
            {} if supplied_references is None else supplied_references)
        def factory(archive, path, *unused, **kwargs):
            return PhaseMeasurement(archive, path, self.audit)
        kwargs = {} if legacy else {"reference_loader": self.loader if loader is None else loader}
        with patch.object(evolution, "Measurement", side_effect=factory):
            return evolution.EvolutionRunner(
                self.out, manifest, self.panels, references,
                model_factory=None, developer=self.developer, **kwargs)

    def complete(self, through="report"):
        result = None
        for phase in PHASES[:PHASES.index(through) + 1]:
            result = self.runner().run_phase(phase)
        return result

    def seal_bytes(self, phase):
        return (self.out / ("phase_" + phase + ".json")).read_bytes()

    def assert_no_new_work(self, before):
        self.assertEqual((len(self.audit.invocations), len(self.audit.computed),
                          len(self.developer.calls), len(self.loaded)), before)

    def counters(self):
        return (len(self.audit.invocations), len(self.audit.computed),
                len(self.developer.calls), len(self.loaded))

    def test_constructor_and_search_check_do_not_read_private_references(self):
        self.allowed_roles = set()
        runner = self.runner()
        runner.check_phase("search")
        self.assertEqual(self.loaded, [])
        self.assertEqual(self.audit.invocations, [])
        self.assertEqual(self.developer.calls, [])

    def test_search_stops_before_select_and_report_and_only_loads_fit(self):
        self.allowed_roles = {"D_fit"}
        result = self.runner().run_phase("search")
        self.assertEqual(result["schema"], "rag-rsi-v3-search-phase-1")
        self.assertEqual(result["status"], "search_frozen")
        self.assertEqual(result["search"], evolution.read(self.out / "search_frozen.json"))
        self.assertEqual({role for _, role in self.audit.invocations}, {"D_fit"})
        self.assertEqual(set(self.loaded), {"D_fit"})
        self.assertEqual(len(self.developer.calls), 1)
        self.assertTrue((self.out / "phase_search.json").exists())
        for name in ("phase_select.json", "delivery_lock.json", "phase_report.json", "report.json"):
            self.assertFalse((self.out / name).exists(), name)
        sent = json.dumps(self.developer.calls)
        self.assertNotIn("PRIVATE_PHASE_FIXTURE_D_select", sent)
        self.assertNotIn("PRIVATE_PHASE_FIXTURE_D_report", sent)

    def test_search_only_manifest_cannot_expand_its_phase_scope(self):
        self.manifest, self.panels, self.references = self.inputs(search_only=True)
        self.allowed_roles = {"D_fit"}
        self.runner().run_phase("search")
        before = self.counters()
        for phase in ("select", "report"):
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                self.runner().check_phase(phase)
            with self.subTest(run=phase), self.assertRaises(ValueError):
                self.runner().run_phase(phase)
        self.assert_no_new_work(before)

    def test_staged_run_forbids_implicit_full_lifecycle(self):
        with self.assertRaises(ValueError):
            self.runner().run()
        self.assert_no_new_work((0, 0, 0, 0))

    def test_later_phases_require_predecessor_seal_before_reference_access(self):
        self.allowed_roles = set()
        for phase in ("select", "report"):
            with self.subTest(phase=phase):
                runner = self.runner()
                with self.assertRaises(ValueError):
                    runner.check_phase(phase)
                with self.assertRaises(ValueError):
                    runner.run_phase(phase)
        self.assert_no_new_work((0, 0, 0, 0))

    def test_select_only_loads_select_and_does_not_restart_search(self):
        self.complete("search")
        self.allowed_roles = {"D_select"}
        before = self.counters()
        self.runner().check_phase("select")
        self.assert_no_new_work(before)
        result = self.runner().run_phase("select")
        self.assertEqual(result["status"], "delivery_locked")
        self.assertEqual(result["delivery_lock"], evolution.read(self.out / "delivery_lock.json"))
        self.assertEqual({role for _, role in self.audit.invocations[before[0]:]}, {"D_select"})
        self.assertEqual(len(self.developer.calls), before[2])
        self.assertTrue((self.out / "phase_select.json").exists())
        self.assertFalse((self.out / "report.json").exists())

    def test_report_only_loads_report_and_honors_worse_locked_child(self):
        self.complete("select")
        locked = (self.out / "delivery_lock.json").read_bytes()
        self.allowed_roles = {"D_report"}
        before = self.counters()
        report = self.runner().run_phase("report")
        self.assertEqual({role for _, role in self.audit.invocations[before[0]:]}, {"D_report"})
        self.assertEqual(len(self.developer.calls), before[2])
        self.assertEqual((self.out / "delivery_lock.json").read_bytes(), locked)
        self.assertLess(report["paired_report_gain"], 0)
        self.assertFalse(report["report_used_for_decisions"])
        self.assertNotEqual(report["delivery_lock"]["node_id"], evolution.read(self.out / "root.json")["node_id"])

    def test_completed_phases_resume_without_load_measure_or_develop(self):
        expected = {}
        seals = {}
        for phase in PHASES:
            expected[phase] = self.runner().run_phase(phase)
            seals[phase] = self.seal_bytes(phase)
        self.allowed_roles = set()
        before = self.counters()
        for phase in PHASES:
            with self.subTest(phase=phase):
                resumed = self.runner()
                resumed.check_phase(phase)
                self.assertEqual(resumed.run_phase(phase), expected[phase])
                self.assertEqual(self.seal_bytes(phase), seals[phase])
        self.assert_no_new_work(before)

    def test_search_artifact_tampering_blocks_select_before_loader(self):
        self.complete("search")
        path = self.out / "search_frozen.json"
        original = path.read_bytes()
        mutations = (lambda x: x["order"].reverse(),
                     lambda x: x["cards"][0].update(score=.99),
                     lambda x: x["cards"].pop())
        self.allowed_roles = set()
        before = self.counters()
        for mutate in mutations:
            value = json.loads(original)
            mutate(value)
            save(path, value)
            with self.subTest(mutation=mutate), self.assertRaises((ValueError, RuntimeError)):
                self.runner().run_phase("select")
            self.assert_no_new_work(before)
            path.write_bytes(original)

    def test_archived_program_tampering_blocks_select_before_loader(self):
        self.complete("search")
        child = evolution.read(self.out / "steps/0/child.json")
        path = self.out / "archive/programs" / child["program_id"] / "files/rag_core.py"
        path.write_bytes(path.read_bytes() + b"\nUNDECLARED_CHANGE = 1\n")
        self.allowed_roles = set()
        before = self.counters()
        with self.assertRaises((ValueError, RuntimeError)):
            self.runner().run_phase("select")
        self.assert_no_new_work(before)

    def test_fit_measurement_tampering_blocks_select_before_loader(self):
        self.complete("search")
        path = next((self.out / "measurements").glob("*.json"))
        value = evolution.read(path)
        value["score"] = .999
        save(path, value)
        self.allowed_roles = set()
        before = self.counters()
        with self.assertRaises((ValueError, RuntimeError)):
            self.runner().run_phase("select")
        self.assert_no_new_work(before)

    def test_search_seal_mutation_blocks_next_phase(self):
        self.complete("search")
        path = self.out / "phase_search.json"
        value = evolution.read(path)
        value["manifest_hash"] = "0" * 64
        save(path, value)
        self.allowed_roles = set()
        before = self.counters()
        with self.assertRaises(ValueError):
            self.runner().run_phase("select")
        self.assert_no_new_work(before)

    def test_omitting_a_measurement_commitment_cannot_hide_changed_fit_cell(self):
        self.complete("search")
        path = next((self.out / "measurements").glob("*.json"))
        seal_path = self.out / "phase_search.json"
        seal = evolution.read(seal_path)
        relative = path.relative_to(self.out).as_posix()
        self.assertIn(relative, seal["artifacts"])
        del seal["artifacts"][relative]
        seal["seal_hash"] = digest({k: v for k, v in seal.items() if k != "seal_hash"})
        save(seal_path, seal)
        value = evolution.read(path)
        value["score"] = .999
        save(path, value)
        self.allowed_roles = set()
        before = self.counters()
        with self.assertRaises(ValueError):
            self.runner().check_phase("select")
        self.assert_no_new_work(before)

    def test_omitting_predecessor_seal_commitment_is_rejected(self):
        self.complete("select")
        path = self.out / "phase_select.json"
        seal = evolution.read(path)
        self.assertIn("phase_search.json", seal["artifacts"])
        del seal["artifacts"]["phase_search.json"]
        # Test the required inventory, independently of the self-checksum.
        seal["seal_hash"] = digest({k: v for k, v in seal.items() if k != "seal_hash"})
        save(path, seal)
        self.allowed_roles = set()
        before = self.counters()
        with self.assertRaises(ValueError):
            self.runner().check_phase("report")
        self.assert_no_new_work(before)

    def test_seal_result_must_match_committed_lock_even_with_new_checksum(self):
        self.complete("select")
        path = self.out / "phase_select.json"
        seal = evolution.read(path)
        seal["result"]["delivery_lock"]["selection_score"] += .01
        seal["seal_hash"] = digest({k: v for k, v in seal.items() if k != "seal_hash"})
        save(path, seal)
        self.allowed_roles = set()
        before = self.counters()
        with self.assertRaises(ValueError):
            self.runner().check_phase("report")
        self.assert_no_new_work(before)

    def test_lock_tampering_or_removal_blocks_report_before_loader(self):
        self.complete("select")
        path = self.out / "delivery_lock.json"
        original = path.read_bytes()
        root = evolution.read(self.out / "root.json")
        self.allowed_roles = set()
        before = self.counters()
        for mutation in ("winner", "score", "missing"):
            with self.subTest(mutation=mutation):
                value = json.loads(original)
                if mutation == "winner":
                    value.update(node_id=root["node_id"], program_id=root["program_id"])
                    save(path, value)
                elif mutation == "score":
                    value["selection_score"] += .01
                    save(path, value)
                else:
                    path.unlink()
                with self.assertRaises((ValueError, RuntimeError)):
                    self.runner().run_phase("report")
                self.assert_no_new_work(before)
                path.write_bytes(original)

    def test_missing_select_seal_cannot_be_replaced_by_delivery_lock(self):
        self.complete("select")
        (self.out / "phase_select.json").unlink()
        self.allowed_roles = set()
        before = self.counters()
        with self.assertRaises(ValueError):
            self.runner().run_phase("report")
        self.assert_no_new_work(before)

    def test_crash_after_each_phase_measurement_reuses_durable_cells(self):
        for phase, role in zip(PHASES, ROLES):
            with self.subTest(phase=phase):
                self.out = self.base / ("crash-" + phase)
                self.audit = Audit()
                self.developer = Developer()
                self.loaded = []
                self.allowed_roles = set(ROLES)
                if phase != "search":
                    self.complete(PHASES[PHASES.index(phase) - 1])
                self.audit.crash_after_role = role
                with self.assertRaises(InjectedCrash):
                    self.runner().run_phase(phase)
                self.assertFalse((self.out / ("phase_" + phase + ".json")).exists())
                computed = list(self.audit.computed)
                self.runner().run_phase(phase)
                self.assertEqual(len(set(self.audit.computed)), len(self.audit.computed))
                self.assertEqual(self.audit.computed[:len(computed)], computed)
                self.assertEqual(sum(r == role for _, r in self.audit.computed), 2)
                self.assertEqual(len(self.developer.calls), 1)

    def test_crash_before_phase_seal_does_not_repurchase_finished_work(self):
        for phase in PHASES:
            with self.subTest(phase=phase):
                self.out = self.base / ("seal-crash-" + phase)
                self.audit = Audit()
                self.developer = Developer()
                self.loaded = []
                self.allowed_roles = set(ROLES)
                if phase != "search":
                    self.complete(PHASES[PHASES.index(phase) - 1])
                real_freeze = evolution.freeze
                def interrupted(path, value):
                    if Path(path).name == "phase_" + phase + ".json":
                        raise InjectedCrash("before phase seal")
                    return real_freeze(path, value)
                with patch.object(evolution, "freeze", side_effect=interrupted):
                    with self.assertRaises(InjectedCrash):
                        self.runner().run_phase(phase)
                before = len(self.audit.computed), len(self.developer.calls)
                self.runner().run_phase(phase)
                self.assertEqual((len(self.audit.computed), len(self.developer.calls)), before)
                self.assertTrue((self.out / ("phase_" + phase + ".json")).exists())

    def test_staged_constructor_rejects_eager_reference_payload(self):
        with self.assertRaises(ValueError):
            self.runner(supplied_references=self.references)
        self.assert_no_new_work((0, 0, 0, 0))

    def test_group_overlap_rejected_without_loading_reference_answers(self):
        for role in ("D_fit", "D_report"):
            qid = self.panels[role][0]["question_id"]
            self.manifest["lifecycle"]["reference_groups"][role][qid] = "same-private-family"
        self.allowed_roles = set()
        with self.assertRaises(ValueError):
            self.runner()
        self.assert_no_new_work((0, 0, 0, 0))

    def test_loader_group_mismatch_rejected_before_measurement(self):
        def wrong_group(role):
            values = self.loader(role)
            next(iter(values.values()))["pair_group_id"] = "undeclared-group"
            return values
        with self.assertRaises(ValueError):
            self.runner(loader=wrong_group).run_phase("search")
        self.assertEqual(self.audit.invocations, [])
        self.assertEqual(self.developer.calls, [])

    def test_bound_reference_file_change_rejected_before_measurement(self):
        runner = self.runner()
        path = Path(self.manifest["lifecycle"]["reference_bindings"]["D_fit"]["path"])
        value = json.loads(path.read_bytes())
        next(iter(value.values()))["answers"].append("changed private fixture")
        save(path, value)
        with self.assertRaises(ValueError):
            runner.run_phase("search")
        self.assertEqual(self.audit.invocations, [])
        self.assertEqual(self.developer.calls, [])

    def test_legacy_run_preserves_full_pipeline_and_resume(self):
        result = self.runner(legacy=True).run()
        self.assertEqual(result["status"], "complete")
        self.assertEqual({role for _, role in self.audit.computed}, set(ROLES))
        self.assertEqual(len(self.audit.computed), 6)
        self.assertEqual(len(self.developer.calls), 1)
        before = len(self.audit.computed), len(self.developer.calls)
        self.assertEqual(self.runner(legacy=True).run(), result)
        self.assertEqual((len(self.audit.computed), len(self.developer.calls)), before)
        for phase in PHASES:
            self.assertFalse((self.out / ("phase_" + phase + ".json")).exists())


if __name__ == "__main__":
    unittest.main()
