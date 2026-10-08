"""Real SQLite + synthetic RPC checks. No provider or benchmark data is used."""
from contextlib import closing
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from code_rsi.v3 import infrastructure as infra
from code_rsi.v3.datasets import adapt_multihop
from code_rsi.v3.execution import HostBroker, HostError


class TrackingConnection(sqlite3.Connection):
    """Observe actual VM progress and connection cleanup, without fake SQL."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.closed_observed = False
        self.progress_calls = 0
        self.progress_hook = None
        self.statements = []
        self.set_trace_callback(self.statements.append)

    def set_progress_handler(self, callback, instructions):
        if callback is None:
            return super().set_progress_handler(None, instructions)
        def observed():
            self.progress_calls += 1
            if self.progress_hook:
                self.progress_hook()
            return callback()
        return super().set_progress_handler(observed, instructions)

    def close(self):
        self.closed_observed = True
        return super().close()


class RetrievalDeadlineTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).parent / "runs"
        root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-retrieval-deadline-", dir=root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def corpus(self, *, excluded=(), large=False):
        path = self.root / "corpus.sqlite"
        docs = [("z-tie", "alpha harbor"), ("a-tie", "alpha harbor"),
                ("middle", "harbor beta gamma"), ("long", "filler " * 1200 + "alpha " * 20),
                ("empty", "unrelated"), ("unicode", "\u5317\u4eac alpha harbor")]
        if large:
            docs.extend(("extra-%05d" % i, "alpha " * (i % 9 + 1) + "beta harbor") for i in range(2000))
        with closing(sqlite3.connect(path)) as con:
            con.executescript("""CREATE TABLE docs (docid TEXT PRIMARY KEY, text TEXT NOT NULL);
                CREATE VIRTUAL TABLE search USING fts5(text, content='docs',
                    content_rowid='rowid', tokenize='porter unicode61');""")
            con.executemany("INSERT INTO docs (docid,text) VALUES (?,?)", docs)
            con.execute("INSERT INTO search(search) VALUES ('rebuild')")
            con.commit()
        return infra.BrowseCompCorpus(path, corpus_hash=hashlib.sha256(path.read_bytes()).hexdigest(), excluded=excluded)

    def tracking(self, backend):
        con = sqlite3.connect(backend.path.as_uri()+"?mode=ro", uri=True, factory=TrackingConnection)
        con.execute("PRAGMA query_only=ON")
        return con

    def original(self, backend, query, limit):
        tokens = sorted(infra.terms(query))
        if not tokens:
            return []
        expression = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)
        with closing(backend._open()) as con:
            rows = con.execute("SELECT docs.docid,docs.text,bm25(search) FROM search JOIN docs "
                "ON docs.rowid=search.rowid WHERE search MATCH ? ORDER BY bm25(search),docs.docid LIMIT ?",
                (expression,limit+len(backend.excluded))).fetchall()
        return [dict(infra.window(docid,text,query),score=score) for docid,text,score in rows
                if str(docid) not in backend.excluded][:limit]

    def test_exact_original_results_including_windows_scores_and_ties(self):
        backend = self.corpus()
        for query in ("alpha harbor", "beta", "unfindable", '"alpha"', "\u5317\u4eac", ""):
            for limit in (1,3,5,30):
                with self.subTest(query=query,limit=limit):
                    self.assertEqual(backend.search(query,limit),self.original(backend,query,limit))

    def test_excluded_overfetch_keeps_exact_original_order(self):
        backend = self.corpus(excluded=("a-tie","middle","missing"))
        for limit in (1,2,4):
            self.assertEqual(backend.search("alpha harbor",limit),self.original(backend,"alpha harbor",limit))
        self.assertNotIn("a-tie",[row["docid"] for row in backend.search("alpha",5)])

    def test_bounded_and_unbounded_search_and_read_agree(self):
        backend = self.corpus()
        self.assertEqual(backend.search_with_timeout("alpha",3,timeout_seconds=5),backend.search("alpha",3))
        self.assertEqual(backend.read_with_timeout("middle",0,6,timeout_seconds=5),backend.read("middle",0,6))
        self.assertEqual(backend.search_with_timeout("nomatch",3,timeout_seconds=5),[])

    def test_ranking_and_body_load_are_in_one_read_transaction(self):
        backend = self.corpus();con = self.tracking(backend)
        with patch.object(backend,"_open",return_value=con):
            self.assertTrue(backend.search_with_timeout("alpha harbor",3,timeout_seconds=5))
        statements = [s.strip().upper() for s in con.statements]
        match_index = next(i for i,s in enumerate(statements) if "MATCH" in s)
        self.assertTrue(any(s.startswith("BEGIN") for s in statements[:match_index]))
        self.assertNotIn("DOCS.TEXT",statements[match_index])
        self.assertNotIn("DOCS.*",statements[match_index])
        self.assertTrue(any("SELECT" in s and "TEXT" in s for s in statements[match_index+1:]))
        self.assertTrue(con.closed_observed)

    def test_invalid_time_budgets_do_not_issue_searches(self):
        backend = self.corpus()
        for budget in (0,-1,True,float("nan"),float("inf"),"5",None):
            with self.subTest(budget=budget), patch.object(backend,"_open",side_effect=AssertionError("opened")):
                with self.assertRaises((ValueError,TypeError)):
                    backend.search_with_timeout("alpha",timeout_seconds=budget)

    def test_actual_sql_progress_expiry_is_timeout_and_closes_connection(self):
        backend = self.corpus(large=True);con=self.tracking(backend);clock=[100.0]
        con.progress_hook=lambda:clock.__setitem__(0,10000.0)
        with patch.object(backend,"_open",return_value=con),patch.object(infra.time,"monotonic",side_effect=lambda:clock[0]):
            with self.assertRaises(infra.RetrievalTimeoutError):
                backend.search_with_timeout("alpha harbor",5,timeout_seconds=1)
        self.assertGreater(con.progress_calls,0)
        self.assertTrue(con.closed_observed)
        with self.assertRaises(sqlite3.ProgrammingError):
            con.execute("SELECT 1")

    def test_window_work_checks_deadline_and_never_returns_partial_results(self):
        backend=self.corpus();clock=[100.0];original=infra.terms
        def slow_terms(text):
            result=original(text)
            if len(text)>1000:
                clock[0]=10000.0
            return result
        with patch.object(infra.time,"monotonic",side_effect=lambda:clock[0]),patch.object(infra,"terms",side_effect=slow_terms):
            with self.assertRaises(infra.RetrievalTimeoutError):
                backend.search_with_timeout("alpha harbor",5,timeout_seconds=1)

    def test_read_expiry_after_sql_is_timeout_and_closes_connection(self):
        backend=self.corpus();con=self.tracking(backend);clock=[100.0]
        def advanced_after_dispatch(sql):
            if sql.lstrip().upper().startswith("SELECT"):
                clock[0]=10000.0
        con.set_trace_callback(advanced_after_dispatch)
        with patch.object(backend,"_open",return_value=con),patch.object(infra.time,"monotonic",side_effect=lambda:clock[0]):
            with self.assertRaises(infra.RetrievalTimeoutError):
                backend.read_with_timeout("middle",0,6,timeout_seconds=1)
        self.assertTrue(con.closed_observed)

    def test_database_error_is_not_mislabeled_as_deadline(self):
        backend=self.corpus();con=self.tracking(backend)
        con.set_authorizer(lambda *args: sqlite3.SQLITE_DENY)
        with patch.object(backend,"_open",return_value=con):
            with self.assertRaises(sqlite3.DatabaseError):
                backend.search_with_timeout("alpha",5,timeout_seconds=5)
        self.assertTrue(con.closed_observed)


class HostRetrievalDeadlineTests(unittest.TestCase):
    def broker(self, fail=None, *, bounded=True):
        task,_=adapt_multihop({"id":"synthetic-deadline", "query":"Where is the synthetic harbor?", "answer":"Northport"})
        class Backend:
            def __init__(self):self.calls=[]
            def search(self,query,limit=5):
                if bounded:raise AssertionError("unbounded search was called")
                self.calls.append(("legacy_search",query,limit));return []
            def read(self,docid,start,end):
                if bounded:raise AssertionError("unbounded read was called")
                return {"docid":docid,"start":start,"end":end,"text":"text"[start:end]}
            def search_with_timeout(self,query,limit=5,*,timeout_seconds):
                self.calls.append(("search",query,limit,timeout_seconds))
                if fail=="search":raise infra.RetrievalTimeoutError("synthetic deadline")
                return []
            def read_with_timeout(self,docid,start,end,*,timeout_seconds):
                self.calls.append(("read",docid,start,end,timeout_seconds))
                if fail=="read":raise infra.RetrievalTimeoutError("synthetic deadline")
                return {"docid":docid,"start":start,"end":end,"text":"text"[start:end]}
        class Model:
            def __init__(self):self.calls=0
            def complete(self,*args):self.calls+=1;return {"answer":"unused"}
        backend=Backend();model=Model()
        if not bounded:
            backend.search_with_timeout=None;backend.read_with_timeout=None
        return HostBroker(task,backend,model),backend,model

    def test_host_passes_remaining_budget_with_margin_for_search_and_read(self):
        broker,backend,_=self.broker()
        self.assertEqual(broker("search",{"query":"harbor","limit":3},remaining=4.5),[])
        broker("read",{"docid":"source","start":0,"end":4},remaining=8)
        self.assertEqual(backend.calls,[("search","harbor",3,3.5),("read","source",0,4,7)])

    def test_near_expired_host_call_still_passes_positive_bounded_time(self):
        for remaining in (.5,.0001):
            with self.subTest(remaining=remaining):
                broker,backend,_=self.broker()
                broker("search",{"query":"harbor","limit":1},remaining=remaining)
                self.assertGreater(backend.calls[0][-1],0)
                self.assertLessEqual(backend.calls[0][-1],remaining)

    def test_backend_timeout_is_fatal_not_empty_and_blocks_later_model(self):
        for method,payload in (("search",{"query":"harbor","limit":1}),("read",{"docid":"source","start":0,"end":4})):
            with self.subTest(method=method):
                broker,backend,model=self.broker(fail=method)
                with self.assertRaises(HostError) as first:
                    broker(method,payload,remaining=5)
                self.assertIs(broker.fatal,first.exception)
                self.assertIsInstance(first.exception.__cause__,infra.RetrievalTimeoutError)
                with self.assertRaises(HostError) as later:
                    broker("complete",{"stage":"answer","payload":{}},remaining=5)
                self.assertIs(first.exception,later.exception)
                self.assertEqual(model.calls,0)
                self.assertEqual(broker.events,[])
                self.assertEqual(broker.counts["model_calls"],0)

    def test_legacy_local_backends_keep_their_supported_interface(self):
        broker,backend,_=self.broker(bounded=False)
        self.assertEqual(broker("search",{"query":"harbor","limit":1},remaining=5),[])
        self.assertEqual(backend.calls,[("legacy_search","harbor",1)])


if __name__=="__main__":
    unittest.main()
