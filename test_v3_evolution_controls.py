"""Offline controls/terminal opportunities: synthetic host data, no network."""
from copy import deepcopy
from pathlib import Path
import unittest
from code_rsi.budget import save
from code_rsi.v3 import evolution
import test_v3_evolution_recovery as recovery_fixtures
from test_v3_evolution_recovery import Developer, Audit


def controls(**changes):
    result = {"parent_policy": "adaptive", "module_policy": "experience_coverage_v1",
              "fixed_module": None, "memory": "none", "feedback": "rich", "case_schedule": []}
    result.update(changes)
    return result


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.fixture = recovery_fixtures.EvolutionRecoveryTests()

    def test_all_rejections_are_opportunities_but_not_nodes(self):
        class Reject(Developer):
            def propose(self, *args):
                self.calls.append(deepcopy(args[1]))
                raise ValueError("fixture rejected proposal")
        with self.fixture.temporary() as tmp:
            d=Reject()
            r=self.fixture.runner(tmp,d,expansions=8,controls=controls())
            report=r.run()
            self.assertEqual(report["terminal_proposals"],8)
            self.assertEqual(report["rejected_proposals"],8)
            self.assertEqual(report["fit_nodes"],1)
            targets=[v["target_module"] for v in d.calls]
            self.assertEqual({m:targets.count(m) for m in evolution.DEFAULT_MODULES},
                             {m:2 for m in evolution.DEFAULT_MODULES})
            rows=evolution.read(Path(tmp)/"search_frozen.json")["terminal_attempts"]
            self.assertTrue(all(x["status"]=="rejected" and x["node_id"] is None for x in rows))
            self.fixture.runner(tmp,d,expansions=8,controls=controls()).run()
            self.assertEqual(len(d.calls),8)

    def test_mixed_children_and_rejections_count_once_on_resume(self):
        with self.fixture.temporary() as tmp:
            d=Developer(reject_first=True)
            report=self.fixture.runner(tmp,d,expansions=4,controls=controls()).run()
            self.assertEqual(report["terminal_proposals"],4)
            self.assertEqual(report["rejected_proposals"],1)
            self.assertEqual(report["fit_nodes"],4)
            self.assertEqual(len({v["decision"]["target_module"] for v in d.calls}),4)
            search=evolution.read(Path(tmp)/"search_frozen.json")
            self.assertEqual(len(search["terminal_attempts"]),4)
            for a in search["terminal_attempts"][1:]:
                card=next(c for c in search["cards"] if c["node_id"]==a["node_id"])
                self.assertEqual(card["step"],a["step"]+1)
                self.assertEqual(card["parent_node_ids"],[a["parent_node_id"]])
            self.fixture.runner(tmp,d,expansions=4,controls=controls()).run()
            self.assertEqual(len(d.calls),4)

    def test_attempt_tampering_fails_closed_without_extra_proposals(self):
        with self.fixture.temporary() as tmp:
            d=Developer(reject_first=True)
            self.fixture.runner(tmp,d,expansions=2,controls=controls()).run()
            path=Path(tmp)/"steps/0/attempt.json"
            value=evolution.read(path); value["intended_target_module"]="invalid"
            save(path,value)
            with self.assertRaisesRegex(ValueError,"frozen run artifact differs"):
                self.fixture.runner(tmp,d,expansions=2,controls=controls()).run()
            self.assertEqual(len(d.calls),2)

    def test_fixed_parent_module_and_empty_memory_stay_fixed(self):
        c=controls(parent_policy="fixed_root",module_policy="fixed",fixed_module="answer_generation")
        with self.fixture.temporary() as tmp:
            d=Developer()
            self.fixture.runner(tmp,d,expansions=3,controls=c).run()
            root=evolution.read(Path(tmp)/"root.json")["node_id"]
            self.assertEqual([v["decision"]["parent_node_id"] for v in d.calls],[root]*3)
            self.assertEqual([v["decision"]["proposal_slot"] for v in d.calls],[0,1,2])
            self.assertTrue(all(v["experience"]==[] and v["decision"]["experience_ids"]==[] for v in d.calls))
            self.assertTrue(all(v["decision"]["target_module"]=="answer_generation" for v in d.calls))

    def test_direct_active_control_mutation_is_detected(self):
        with self.fixture.temporary() as tmp:
            audit=Audit(); runner=self.fixture.runner(tmp,audit=audit,controls=controls())
            runner.controls["memory"]="mechanism"
            with self.assertRaisesRegex(ValueError,"frozen run inputs"):
                runner.run()
            self.assertEqual(audit.invocations,[])

    def test_controlled_feedback_rejects_custom_developer(self):
        c=controls(parent_policy="fixed_root",module_policy="fixed",fixed_module="answer_generation",
                   feedback="aggregate",case_schedule=[{"question_id":"D_fit","repeat":0}])
        _,panels,_=self.fixture.inputs()
        c["case_schedule"][0]["question_id"]=panels["D_fit"][0]["question_id"]
        with self.fixture.temporary() as tmp:
            with self.assertRaisesRegex(ValueError,"host ProgramDeveloper"):
                self.fixture.runner(tmp,controls=c)

    def test_case_schedule_identity_and_combinations_are_frozen(self):
        tasks=[{"question_id":"q"}]
        valid=controls(parent_policy="fixed_root",module_policy="fixed",fixed_module="retrieval",
                       feedback="trace",case_schedule=[{"question_id":"q","repeat":1}])
        self.assertEqual(evolution.validate_controls(valid,tasks,2),valid)
        variants=[{"memory":"mechanism"},{"parent_policy":"adaptive"},{"module_policy":"round_robin_v1"},
                  {"case_schedule":[{"question_id":"report-q","repeat":0}]},
                  {"case_schedule":[{"question_id":"q","repeat":2}]},
                  {"case_schedule":[{"question_id":"q","repeat":True}]},
                  {"case_schedule":[{"question_id":"q","repeat":0}]*2}]
        for changes in variants:
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                evolution.validate_controls({**valid,**changes},tasks,2)

    def test_legacy_is_explicitly_reserved_for_old_runs(self):
        self.assertEqual(evolution.validate_controls(None,[],1,allow_legacy=True),evolution.LEGACY_CONTROLS)
        with self.assertRaises(ValueError):
            evolution.validate_controls(evolution.LEGACY_CONTROLS,[],1)


if __name__ == "__main__":
    unittest.main()
