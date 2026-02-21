import os
import sys
import types
import unittest
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _fake_modules():
    fake_faiss = types.ModuleType("faiss")
    fake_faiss.IndexFlatL2 = object
    fake_faiss.read_index = lambda *a, **k: None
    fake_faiss.write_index = lambda *a, **k: None

    fake_st = types.ModuleType("sentence_transformers")

    class FakeST:
        def __init__(self, name):
            self.name = name
        def encode(self, texts, show_progress_bar=False):
            if isinstance(texts, str):
                texts = [texts]
            return [[0.0] * 8 for _ in texts]

    fake_st.SentenceTransformer = FakeST
    fake_pypdf = types.ModuleType("pypdf")
    class _PdfReader:
        def __init__(self, *a, **k):
            self.pages = []
    fake_pypdf.PdfReader = _PdfReader
    return fake_faiss, fake_st, fake_pypdf


class CreateMasterDatasetUtilsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_faiss, fake_st, fake_pypdf = _fake_modules()
        cls._patcher = patch.dict(
            sys.modules,
            {
                "faiss": fake_faiss,
                "sentence_transformers": fake_st,
                "pypdf": fake_pypdf,
            },
            clear=False,
        )
        cls._patcher.start()
        global cmd
        import masterdbcreation.create_master_dataset as cmd  # noqa: F401

    @classmethod
    def tearDownClass(cls):
        cls._patcher.stop()
        if "masterdbcreation.create_master_dataset" in sys.modules:
            del sys.modules["masterdbcreation.create_master_dataset"]

    def test_normalize_ws(self):
        from masterdbcreation.create_master_dataset import _normalize_ws
        self.assertEqual(_normalize_ws("  a   b\n c  "), "a b c")

    def test_split_into_chunks(self):
        from masterdbcreation.create_master_dataset import _split_into_chunks
        text = " ".join(["word"] * 50)
        chunks = _split_into_chunks(text, chunk_size=30, overlap=5)
        self.assertTrue(len(chunks) >= 2)
        self.assertTrue(all(isinstance(c, str) and c for c in chunks))

    def test_dedupe_by_question(self):
        from masterdbcreation.create_master_dataset import _dedupe_by_question
        qa = [("Q1?", "A1"), ("Q1?  ", "A1dup"), ("Q2?", "A2")]
        out = _dedupe_by_question(qa)
        qs = [q for q, _ in out]
        self.assertIn("Q1?  ", qs)  # last duplicate remains
        self.assertIn("Q2?", qs)
        self.assertEqual(len(out), 2)

    def test_dedupe_answers(self):
        from masterdbcreation.create_master_dataset import _dedupe_answers
        qa = [("Q1?", "A1   "), ("Q2?", "A1"), ("Q3?", "A3")] 
        out = _dedupe_answers(qa)
        self.assertEqual(len(out), 2)

    def test_make_question(self):
        from masterdbcreation.create_master_dataset import _make_question
        q = _make_question("src", "This is some content about settings and options.", 0)
        self.assertTrue("Dell Data Protection Advisor" in q)


if __name__ == "__main__":
    unittest.main(verbosity=2)
