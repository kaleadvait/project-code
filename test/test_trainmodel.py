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
    return fake_faiss, fake_st


class TrainModelUtilsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_faiss, fake_st = _fake_modules()
        cls._patcher = patch.dict(
            sys.modules,
            {
                "faiss": fake_faiss,
                "sentence_transformers": fake_st,
            },
            clear=False,
        )
        cls._patcher.start()
        global tm
        import trainmasterdb.trainmodel as tm  # noqa: F401

    @classmethod
    def tearDownClass(cls):
        cls._patcher.stop()
        if "trainmasterdb.trainmodel" in sys.modules:
            del sys.modules["trainmasterdb.trainmodel"]

    def test_normalize_ws(self):
        from trainmasterdb.trainmodel import _normalize_ws
        self.assertEqual(_normalize_ws("  a   b\n c  "), "a b c")

    def test_split_indices(self):
        from trainmasterdb.trainmodel import _split_indices
        tr, te = _split_indices(10, seed=123, test_ratio=0.2)
        self.assertEqual(sorted(set(tr + te)), list(range(10)))
        self.assertTrue(len(te) >= 1)

    def test_dedupe_by_question(self):
        from trainmasterdb.trainmodel import _dedupe_by_question
        ds = _dedupe_by_question(["Q1", "Q1 ", "Q2"], ["A1", "A1dup", "A2"])
        self.assertEqual(len(ds.questions), 2)
        self.assertIn("Q2", ds.questions)


if __name__ == "__main__":
    unittest.main(verbosity=2)
