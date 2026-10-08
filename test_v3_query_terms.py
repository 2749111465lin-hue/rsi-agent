"""Offline SQLite FTS5 retrieval regressions; no provider or private corpus used."""
from contextlib import closing
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from code_rsi.budget import digest
from code_rsi.v3.infrastructure import BrowseCompCorpus, terms


class BrowseCompQueryTermsTests(unittest.TestCase):
    def setUp(self):
        folder = Path(__file__).parent / "runs"
        folder.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-query-terms-", dir=folder)
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def corpus(self, documents, *, excluded=()):
        path = self.root / "corpus.sqlite"
        with closing(sqlite3.connect(path)) as con:
            con.executescript("""
                CREATE TABLE docs (docid TEXT PRIMARY KEY, text TEXT NOT NULL);
                CREATE VIRTUAL TABLE search USING fts5(
                    text, content='docs', content_rowid='rowid',
                    tokenize='porter unicode61');
            """)
            con.executemany("INSERT INTO docs (docid, text) VALUES (?, ?)", documents)
            con.execute("INSERT INTO search(search) VALUES ('rebuild')")
            con.commit()
        self.corpus_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        return BrowseCompCorpus(path, corpus_hash=self.corpus_hash, excluded=excluded)

    def test_proper_name_after_more_than_65_terms_is_searchable(self):
        backend = self.corpus([
            ("target", "Zyzzyvan discovered the fictional island."),
            ("distractor", "alpha000 alpha001 alpha002"),
        ])
        filler = ["alpha" + str(i).zfill(3) for i in range(70)]
        for words in (filler + ["Zyzzyvan"], ["Zyzzyvan"] + filler):
            with self.subTest(proper_name_first=words[0] == "Zyzzyvan"):
                query = " ".join(words)
                self.assertEqual(len(terms(query)), 71)
                self.assertIn("target", [r["docid"] for r in backend.search(query)])

    def test_duplicate_words_and_case_do_not_change_scores(self):
        backend = self.corpus([
            ("a", "Northport northport"),
            ("b", "Northport harbor"),
            ("c", "Elsewhere"),
        ])
        self.assertEqual(backend.search("Northport"),
                         backend.search("northport NORTHPORT Northport northport"))

    def test_quotes_and_sql_punctuation_are_literal_query_input(self):
        backend = self.corpus([("port", 'Northport is called "home".')])
        for query in ('"Northport"', "'Northport'", 'Northport"; DROP TABLE docs; --'):
            with self.subTest(query=query):
                self.assertEqual([r["docid"] for r in backend.search(query)], ["port"])
        self.assertEqual(backend.read("port", 0, 9)["text"], "Northport")

    def test_chinese_terms_are_retained(self):
        backend = self.corpus([("beijing", "北京 港口"), ("suzhou", "苏州 园林")])
        self.assertEqual([r["docid"] for r in backend.search('"苏州"')], ["suzhou"])
        self.assertEqual({r["docid"] for r in backend.search("北京，苏州")},
                         {"beijing", "suzhou"})

    def test_empty_or_tokenless_query_has_no_hits(self):
        backend = self.corpus([("port", "Northport")])
        for query in ("", " \n\t", '"\'() - : *'):
            with self.subTest(query=query):
                self.assertEqual(backend.search(query), [])

    def test_exclusions_preserve_limit_and_document_id_tie_break(self):
        backend = self.corpus([(docid, "Northport") for docid in ("a", "b", "c", "d")],
                              excluded=("a", "b", "not-present"))
        self.assertEqual([r["docid"] for r in backend.search("Northport", 2)], ["c", "d"])
        with self.assertRaisesRegex(ValueError, "excluded"):
            backend.read("a", 0, 9)

    def test_bm25_ranking_precedes_document_id_order(self):
        backend = self.corpus([
            ("a-long", "alpha " + "filler " * 50),
            ("c-short", "alpha"),
            ("b-short", "alpha"),
            ("z-unrelated", "elsewhere"),
        ])
        rows = backend.search("alpha")
        self.assertEqual([r["docid"] for r in rows], ["b-short", "c-short", "a-long"])
        self.assertEqual(rows[0]["score"], rows[1]["score"])
        self.assertLess(rows[1]["score"], rows[2]["score"])

    def test_full_host_query_bound_with_8000_distinct_terms(self):
        # One-character CJK terms stress the OR expression's clause count.
        # The final two-character term uses all 16,000 input characters.
        words = [chr(0x4E00 + i) for i in range(8000)]
        words[-1] += "a"
        query = " ".join(words)
        self.assertEqual(len(query), 16000)
        self.assertEqual(len(terms(query)), 8000)
        self.assertEqual(sum(len(t) + 2 for t in words) + 4 * (len(words) - 1), 55997)
        backend = self.corpus([
            ("a-first", words[0]), ("b-middle", words[4000]), ("z-last", words[-1]),
        ])
        self.assertEqual([r["docid"] for r in backend.search(query)],
                         ["a-first", "b-middle", "z-last"])

    def test_casefold_expansion_at_full_host_query_bound(self):
        # U+1FB7 folds to alpha + combining mark + iota. The mark creates
        # a regex term boundary, so input_chars // 2 is not a safe term cap.
        letters = [chr(0x4E00 + i) for i in range(8000)]
        query = "".join(letter + "\u1fb7" for letter in letters)
        words = terms(query)
        self.assertEqual(len(query), 16000)
        self.assertEqual(len(query.casefold()), 32000)
        self.assertEqual(len(words), 8001)
        self.assertEqual(sum(len(t) + 2 for t in words) + 4 * (len(words) - 1), 72002)
        backend = self.corpus([("last", "\u03b9" + letters[-1] + "\u03b1")])
        self.assertEqual([r["docid"] for r in backend.search(query)], ["last"])

    def test_full_host_query_bound_with_one_long_term(self):
        word = "x" * 16000
        backend = self.corpus([("long", word)])
        self.assertEqual([r["docid"] for r in backend.search(word)], ["long"])

    def test_read_and_search_close_connections_on_success(self):
        backend = self.corpus([("port", "Northport")])
        for method, args in ((backend.search, ("Northport",)), (backend.read, ("port", 0, 9))):
            with self.subTest(method=method.__name__):
                con = backend._open()
                with patch.object(backend, "_open", return_value=con):
                    method(*args)
                with self.assertRaisesRegex(sqlite3.ProgrammingError, "closed"):
                    con.execute("SELECT 1")

    def test_read_and_search_close_connections_on_sql_error(self):
        backend = self.corpus([("port", "Northport")])
        for method, args in ((backend.search, ("Northport",)), (backend.read, ("port", 0, 9))):
            with self.subTest(method=method.__name__):
                con = backend._open()
                con.set_authorizer(lambda *unused: sqlite3.SQLITE_DENY)
                with patch.object(backend, "_open", return_value=con):
                    with self.assertRaises(sqlite3.DatabaseError):
                        method(*args)
                with self.assertRaisesRegex(sqlite3.ProgrammingError, "closed"):
                    con.execute("SELECT 1")

    def test_backend_identity_invalidates_truncated_search_protocol(self):
        backend = self.corpus([("port", "Northport")])
        previous = digest({"corpus_sha256": self.corpus_hash, "excluded": [],
                           "backend": "fts5-porter-v3"})
        self.assertNotEqual(backend.identity, previous)
        self.assertEqual(backend.identity, BrowseCompCorpus(
            backend.path, corpus_hash=self.corpus_hash).identity)
        self.assertNotEqual(backend.identity, BrowseCompCorpus(
            backend.path, corpus_hash=self.corpus_hash, excluded=("port",)).identity)


if __name__ == "__main__":
    unittest.main()
