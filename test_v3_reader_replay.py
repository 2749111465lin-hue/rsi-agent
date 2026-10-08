"""Synthetic final-reader replay tests. No provider, references or cached code exec."""
import ast
from copy import deepcopy
import hashlib
import json
import unittest

from code_rsi.budget import digest
from code_rsi.v3.datasets import adapt_multihop
from code_rsi.v3.execution import HostBroker, HostError, root_files
from code_rsi.v3.infrastructure import ModelResponseError, UnknownProviderOutcome
from code_rsi.v3.rag import RagEngine
from code_rsi.v3.reader_replay import (SCHEMA, ReplayContractError, ReplayMismatch,
                                      ReplayRouter, replay_files, validate_case)


def synthetic_case():
    task,_=adapt_multihop({"id":"synthetic-reader", "query":"Where and when was Mira born?",
                          "answer":"Northport, 1980"})
    config={"mode":"iterative", "max_rounds":3, "max_stagnant_rounds":2}
    events=[]
    class Recording:
        def search(self,query,limit=5):
            text="Mira was born in Northport." if query=="Mira" else "Mira was born in 1980."
            response=[{"docid":"city" if query=="Mira" else "date", "text":text,
                       "start":0,"end":len(text),"score":1.0}]
            events.append({"name":"search","request":{"query":query,"limit":limit},
                           "response":deepcopy(response),"response_sha256":digest(response)})
            return response
        def complete(self,stage,payload):
            if stage=="plan":
                response={"constraints":["birth city","birth year"],"queries":["Mira"]}
            elif stage=="read":
                source=payload["sources"][0];last=payload["round"]==2
                response={"claims":[{"text":"birth fact", "citations":[{
                    "source_id":source["source_id"],"quote":source["text"]}]}],
                    "bridge_entities":[],"gaps":[] if last else ["birth year missing"],
                    "conflicts":[],"queries":[] if last else ["Mira registry"],"ready":last}
            else:
                response={"answer":"Northport, 1980","citation_ids":["e1","e2"],"evidence_sufficient":True}
            events.append({"name":"complete","request":{"stage":stage,"payload":deepcopy(payload)},
                           "response":deepcopy(response),"response_sha256":digest(response)})
            return response
    recording=Recording()
    result=RagEngine(recording,recording,config=config).solve({"question":task["question"]})
    assert result["answer_usable"] and result["usage"]["read_calls"]==0
    return {"schema":SCHEMA,"case_id":"synthetic_case_01","task":task,"config":config,
            "target_read":2,"events":events,
            "engine_sha256":hashlib.sha256(root_files()["rag_core.py"].encode()).hexdigest(),
            "original_final_payload_sha256":digest(events[-1]["request"]["payload"]),
            "source_binding":{"kind":"synthetic"}}


def dispatch(router,event,*,guidance=None,target=False):
    request=deepcopy(event["request"])
    if event["name"]=="complete":
        if target and guidance is not None:
            request["payload"]["additional_guidance"]=guidance
        return router.complete(**request)
    return getattr(router,event["name"])(**request)


class ReaderReplayTests(unittest.TestCase):
    def setUp(self):
        self.case=synthetic_case()

    def test_case_validation_is_pure_and_returns_an_isolated_copy(self):
        frozen=deepcopy(self.case);validated=validate_case(self.case)
        self.assertEqual(validated,self.case)
        validated["events"][0]["response"]["queries"].append("changed")
        self.assertEqual(self.case,frozen)

    def test_original_engine_offline_reproduces_entire_prefix_and_final_payload(self):
        router=ReplayRouter(self.case)
        config={**self.case["config"],"max_rounds":self.case["target_read"]}
        result=RagEngine(router.backend,router.model,config=config).solve({"question":self.case["task"]["question"]})
        self.assertEqual(result["answer"],"Northport, 1980")
        self.assertEqual(result["failure_types"],[])
        self.assertEqual(router.assert_complete(),{"replayed_model_calls":4,"new_model_calls":0,
                         "replayed_search_calls":2,"replayed_reads":0,"complete":True})
        self.assertEqual(router.replayed_calls,len(self.case["events"]))
        self.assertEqual(router.new_calls,0)

    def test_offline_router_rejects_final_payload_drift_even_with_rehashed_case(self):
        case=deepcopy(self.case)
        case["events"][-1]["request"]["payload"]["gaps"]=["injected history"]
        case["original_final_payload_sha256"]=digest(case["events"][-1]["request"]["payload"])
        router=ReplayRouter(case)
        RagEngine(router,router,config={**case["config"],"max_rounds":2}).solve({"question":case["task"]["question"]})
        with self.assertRaises(ReplayMismatch):router.assert_complete()

    def test_files_reuse_current_core_and_only_cap_rounds_and_wrap_target_guidance(self):
        files=replay_files(self.case,"Extract each required fact.")
        self.assertEqual(files["rag_core.py"],root_files()["rag_core.py"])
        parsed=ast.parse(files["rag.py"])
        config_node=next(node for node in parsed.body if isinstance(node,ast.Assign) and
                         any(isinstance(target,ast.Name) and target.id=="CONFIG" for target in node.targets))
        config=json.loads(config_node.value.args[0].value)
        self.assertEqual(config,{**self.case["config"],"max_rounds":2})
        self.assertIn("self.read_ordinal == 2",files["rag.py"])
        self.assertIn("payload['additional_guidance']",files["rag.py"])
        self.assertEqual(self.case["config"]["max_rounds"],3)
        for files in (files,replay_files(self.case,None)):
            ast.parse(files["rag.py"]);ast.parse(files["rag_core.py"])

    def test_schema_hash_engine_question_and_source_corruption_are_rejected(self):
        def added(case):case["reference_answer"]="must not be here"
        def response(case):case["events"][0]["response"]["queries"]=["changed"]
        def engine(case):case["engine_sha256"]="0"*64
        def question(case):case["events"][0]["request"]["payload"]["question"]="different"
        def source(case):
            source=case["events"][2]["request"]["payload"]["sources"][0]
            source["text"]="fabricated";source["end"]=source["start"]+len(source["text"])
            source["text_sha256"]=hashlib.sha256(source["text"].encode()).hexdigest()
        for change in (added,response,engine,question,source):
            with self.subTest(change=change.__name__):
                case=deepcopy(self.case);change(case)
                with self.assertRaises((ReplayContractError,ValueError)):validate_case(case)

    def test_target_is_last_reader_and_no_backend_event_can_follow_it(self):
        case=deepcopy(self.case);case["target_read"]=1
        with self.assertRaises(ReplayContractError):validate_case(case)
        case=deepcopy(self.case);case["events"].insert(-1,deepcopy(case["events"][1]))
        with self.assertRaises(ReplayContractError):validate_case(case)
        for bad in (0,True,3):
            case=deepcopy(self.case);case["target_read"]=bad
            with self.assertRaises(ReplayContractError):validate_case(case)

    def test_prefix_mismatch_is_sticky_and_never_reaches_live_model(self):
        class Live:
            def complete(self,*args):raise AssertionError("no live prefix dispatch")
        router=ReplayRouter(self.case,Live())
        with self.assertRaises(ReplayMismatch) as error:router.search("unexpected",5)
        with self.assertRaises(ReplayMismatch) as later:dispatch(router,self.case["events"][0])
        self.assertIs(error.exception,later.exception)
        self.assertEqual(router.new_calls,0)
        self.assertFalse(router.summary()["complete"])

    def test_live_model_gets_only_target_reader_and_answer_not_history(self):
        calls=[];case=self.case
        class Live:
            timeout_seconds=150
            def complete(self,stage,payload):
                calls.append((stage,deepcopy(payload)))
                return deepcopy(case["events"][-2 if stage=="read" else -1]["response"])
        live=Live();router=ReplayRouter(case,live,"new reader guidance")
        router.timeout_seconds=11
        for i,event in enumerate(case["events"]):
            dispatch(router,event,guidance="new reader guidance",target=i==len(case["events"])-2)
        self.assertEqual([stage for stage,_ in calls],["read","answer"])
        self.assertEqual(calls[0][1]["additional_guidance"],"new reader guidance")
        self.assertEqual(calls[1][1]["additional_guidance"],"")
        self.assertEqual(live.timeout_seconds,11)
        self.assertEqual(router.assert_complete(),{"replayed_model_calls":2,"new_model_calls":2,
                         "replayed_search_calls":2,"replayed_reads":0,"complete":True})
        self.assertEqual(router.new_calls,2)

    def test_new_reader_queries_do_not_execute_and_merge_stays_in_original_engine(self):
        calls=[];case=self.case
        class Live:
            def complete(self,stage,payload):
                calls.append((stage,deepcopy(payload)))
                if stage=="read":
                    response=deepcopy(case["events"][-2]["response"])
                    response.update(queries=["must never execute"],ready=False)
                    return response
                return deepcopy(case["events"][-1]["response"])
        router=ReplayRouter(case,Live())
        result=RagEngine(router,router,config={**case["config"],"max_rounds":2}).solve({"question":case["task"]["question"]})
        router.assert_complete()
        self.assertEqual(result["usage"]["search_calls"],2)
        self.assertEqual(result["stop_reason"],"round_limit")
        self.assertEqual(len(calls[-1][1]["evidence"]),2)
        self.assertEqual(calls[-1][1]["stop_reason"],"round_limit")

    def test_target_guidance_is_only_allowed_live_reader_difference(self):
        class Live:
            def complete(self,*args):raise AssertionError("mismatch must not be sent")
        for key,value in (("question","changed"),("constraints",["changed"]),("additional_guidance","wrong")):
            with self.subTest(key=key):
                router=ReplayRouter(self.case,Live(),"expected")
                for event in self.case["events"][:-2]:dispatch(router,event)
                request=deepcopy(self.case["events"][-2]["request"])
                request["payload"]["additional_guidance"]="expected"
                request["payload"][key]=value
                with self.assertRaises(ReplayMismatch):router.complete(**request)
                self.assertEqual(router.new_calls,0)

    def test_final_dynamic_merge_allowed_but_answer_prompt_cannot_change(self):
        calls=[];case=self.case
        class Live:
            def complete(self,stage,payload):
                calls.append(stage);return deepcopy(case["events"][-2 if stage=="read" else -1]["response"])
        router=ReplayRouter(case,Live())
        for event in case["events"][:-1]:dispatch(router,event)
        request=deepcopy(case["events"][-1]["request"])
        request["payload"]["stage_instructions"]="different final prompt"
        with self.assertRaises(ReplayMismatch):router.complete(**request)
        self.assertEqual(calls,["read"])

    def test_completed_reader_format_error_advances_without_repurchase(self):
        calls=[];case=self.case
        class Live:
            def complete(self,stage,payload):
                calls.append(stage)
                if stage=="read":raise ModelResponseError("completed malformed reader")
                return deepcopy(case["events"][-1]["response"])
        router=ReplayRouter(case,Live())
        for event in case["events"][:-2]:dispatch(router,event)
        with self.assertRaises(ModelResponseError):dispatch(router,case["events"][-2])
        dispatch(router,case["events"][-1])
        self.assertTrue(router.assert_complete()["complete"])
        self.assertEqual(calls,["read","answer"])

    def test_host_format_failure_keeps_prefix_and_original_engine_still_answers_once(self):
        calls=[]
        class Live:
            def complete(self,stage,payload):
                calls.append(stage)
                if stage=="read":raise ModelResponseError("completed malformed reader")
                return {"answer":"Northport","citation_ids":["e1"],"evidence_sufficient":True}
        router=ReplayRouter(self.case,Live())
        broker=HostBroker(self.case["task"],router,router)
        class Services:
            def search(self,query,limit=5):return broker("search",{"query":query,"limit":limit})
            def complete(self,stage,payload):return broker("complete",{"stage":stage,"payload":payload})
        services=Services()
        result=RagEngine(services,services,config={**self.case["config"],"max_rounds":2}).solve({"question":self.case["task"]["question"]})
        self.assertEqual(result["answer"],"Northport")
        self.assertEqual(result["stop_reason"],"read_failure")
        self.assertEqual(len(result["state"]["citations"]),1)
        self.assertTrue(broker.answer_origin_receipt(result["answer"])["valid"])
        self.assertIn("ModelResponseError",broker.model_errors)
        self.assertEqual(calls,["read","answer"])
        self.assertEqual(router.assert_complete()["new_model_calls"],2)

    def test_unknown_physical_outcome_is_not_swallowed_or_retried(self):
        error=UnknownProviderOutcome("synthetic unknown")
        class Live:
            def complete(self,*args):raise error
        router=ReplayRouter(self.case,Live())
        for event in self.case["events"][:-2]:dispatch(router,event)
        with self.assertRaises(UnknownProviderOutcome) as first:dispatch(router,self.case["events"][-2])
        with self.assertRaises(UnknownProviderOutcome) as later:dispatch(router,self.case["events"][-1])
        self.assertIs(first.exception,error);self.assertIs(later.exception,error)
        self.assertEqual(router.new_calls,1)
        with self.assertRaises(UnknownProviderOutcome):router.assert_complete()

    def test_outputs_do_not_alias_frozen_events_and_extra_calls_fail(self):
        router=ReplayRouter(self.case)
        first=dispatch(router,self.case["events"][0]);first["queries"].append("mutated")
        self.assertEqual(router.case["events"][0]["response"]["queries"],["Mira"])
        for event in self.case["events"][1:]:dispatch(router,event)
        router.assert_complete()
        with self.assertRaises(ReplayMismatch):router.search("new",1)
        self.assertFalse(router.summary()["complete"])

    def test_host_broker_preserves_unknown_outcome_as_fatal_before_new_answer(self):
        error=UnknownProviderOutcome("synthetic unknown")
        class Live:
            def complete(self,*args):raise error
        router=ReplayRouter(self.case,Live())
        broker=HostBroker(self.case["task"],router,router)
        for event in self.case["events"][:-2]:broker(event["name"],deepcopy(event["request"]))
        with self.assertRaises(UnknownProviderOutcome):
            broker("complete",deepcopy(self.case["events"][-2]["request"]))
        with self.assertRaises(UnknownProviderOutcome):
            broker("complete",deepcopy(self.case["events"][-1]["request"]))
        self.assertIs(broker.fatal,error)
        self.assertEqual(router.new_calls,1)


if __name__=="__main__":
    unittest.main()
