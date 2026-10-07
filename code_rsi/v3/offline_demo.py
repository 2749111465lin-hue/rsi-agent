"""Deterministic synthetic double-hop fixture through the actual WSL execution.

No real model, benchmark or API is used. The known edit and scripted model test
plumbing only; the resulting gain must never be reported as empirical quality.
"""
import copy
from .datasets import adapt_multihop
from .evolution import EvolutionRunner, ProgramDeveloper
from ..budget import digest


STORIES=[("Aster","Mira Vale","Northport"),("Boreal","Ivo Reed","Southport"),("Cygnus","Tara Moss","Westport")]


class FixtureBackend:
    def __init__(self,task):
        self.story=next(s for s in STORIES if s[0] in task["question"])
        self.identity=digest({"fixture":"twohop-v1","story":self.story})
    def search(self,query,limit=5):
        comet,person,city=self.story
        if person.casefold() in query.casefold():
            text,docid=f"{person} was born in {city}.","bio"
        else:
            text,docid=f"The {comet} comet was discovered by {person}.","discovery"
        return [{"docid":docid,"text":text,"start":0,"end":len(text)}]


class FixtureModel:
    identity="synthetic-scripted-model-v1-not-LLM"
    def complete(self,stage,payload):
        if stage=="develop":
            source=payload["source_files"]["rag.py"]
            return {"writes":{"rag.py":source.replace("single_pass","iterative")},
              "target_module":payload["decision"]["target_module"],
              "mechanism":"SYNTHETIC KNOWN EDIT: permit a second query using the observed bridge entity"}
        comet,person,city=next(s for s in STORIES if s[0] in payload["question"])
        if stage=="plan":
            return {"constraints":["discoverer and birthplace"],"queries":[f"{comet} comet discovery"]}
        if stage=="read":
            found=any("was born" in s["text"] for s in payload["sources"])
            return {"claims":[{"text":s["text"],"citations":[{"source_id":s["source_id"],"start":s["start"],"end":s["end"],"quote":s["text"]}]} for s in payload["sources"]],
                "bridge_entities":[person],"gaps":[] if found else ["birthplace"],"conflicts":[],
                "queries":[] if found else [person+" birthplace"],"ready":found}
        cited=[e["citation_id"] for e in payload["evidence"] if "was born" in e["quote"]]
        return {"answer":city if cited else "Insufficient information","citation_ids":cited,"evidence_sufficient":bool(cited)}


def demo(directory):
    panels={}; references={}
    for role,(comet,person,city) in zip(("D_fit","D_select","D_report"),STORIES):
        task,ref=adapt_multihop({"id":"synthetic-"+comet,
            "query":f"Where was the discoverer of the {comet} comet born?","answer":city,
            "corpus_ref":"synthetic-twohop-not-benchmark"})
        panels[role]=[task]; references[role]={task["question_id"]:ref}
    manifest={"schema":"rag-rsi-v3-run-1","synthetic":True,"metric":"em","expansions":1,
      "repeats":1,"root_config":{"mode":"single_pass"},"select_candidates":2,
      "limits":{"max_models":7,"max_searches":8,"max_reads":8}}
    runner=EvolutionRunner(directory,manifest,panels,references,lambda bank:FixtureModel(),
                           ProgramDeveloper(FixtureModel()),backend_factory=FixtureBackend)
    result=runner.run()
    if result["report_score"]!=1 or result["reference_score"]!=0 or result["fit_nodes"]!=2:
        raise AssertionError("synthetic end-to-end mechanism failed")
    return result
