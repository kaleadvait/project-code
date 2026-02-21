import os
import sys
import types
import importlib
import io
import unittest
from unittest.mock import patch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _make_fake_modules():
    fake_faiss = types.ModuleType("faiss")

    class FakeIndex:
        def __init__(self, dim):
            self.dim = dim
            self._items = []

        def add(self, embeddings):
            try:
                self._items.extend(list(embeddings))
            except Exception:
                self._items.append(embeddings)

        @property
        def ntotal(self):
            return len(self._items)

        def search(self, q_embs, k):
            try:
                n = len(q_embs)
            except Exception:
                n = 1
            distances = [[0.0 for _ in range(k)] for _ in range(n)]
            indices = [[0 for _ in range(k)] for _ in range(n)]
            return distances, indices

    def read_index(_path):
        raise RuntimeError("force rebuild in tests")

    def write_index(_index, _path):
        return None

    fake_faiss.IndexFlatL2 = FakeIndex
    fake_faiss.read_index = read_index
    fake_faiss.write_index = write_index

    fake_st = types.ModuleType("sentence_transformers")

    class FakeEmb:
        def __init__(self, n, d):
            self._n = n
            self._d = d

        def __len__(self):
            return self._n

        def __iter__(self):
            for _ in range(self._n):
                yield [0.0] * self._d

        @property
        def shape(self):
            return (self._n, self._d)

    class FakeST:
        def __init__(self, name):
            self.name = name

        def encode(self, texts, show_progress_bar=False):
            if isinstance(texts, str):
                texts = [texts]
            return FakeEmb(len(texts), 8)

    fake_st.SentenceTransformer = FakeST

    # Stub for pypdf to satisfy `from pypdf import PdfReader` at import time
    fake_pypdf = types.ModuleType("pypdf")
    class _PdfReader:
        def __init__(self, *a, **k):
            self.pages = []
    fake_pypdf.PdfReader = _PdfReader

    return fake_faiss, fake_st, fake_pypdf


class ServerApiTests(unittest.TestCase):
    def setUp(self):
        self.fake_faiss, self.fake_st, self.fake_pypdf = _make_fake_modules()
        # Patch heavy modules
        self._patcher = patch.dict(
            sys.modules,
            {
                "faiss": self.fake_faiss,
                "sentence_transformers": self.fake_st,
                "pypdf": self.fake_pypdf,
            },
            clear=False,
        )
        self._patcher.start()

        # Provide tiny in-memory client_data.pkl and avoid touching disk
        self._pickle_patcher = patch(
            "pickle.load",
            return_value={"questions": ["q0", "q1"], "answers": ["a0", "a1"]},
        )
        import builtins as _builtins
        self._real_open = _builtins.open

        def _open_side_effect(path, mode="r", *a, **k):
            if str(path).endswith("client_data.pkl") and "b" in mode:
                return io.BytesIO(b"\x80\x04.")
            return self._real_open(path, mode, *a, **k)

        self._open_patcher = patch("builtins.open", side_effect=_open_side_effect)
        self._pickle_patcher.start()
        self._open_patcher.start()
        # Import server with fakes
        self.server = importlib.import_module("server.server")

    def tearDown(self):
        # Ensure module is cleaned up for other tests if re-imported
        if "server.server" in sys.modules:
            del sys.modules["server.server"]
        self._open_patcher.stop()
        self._pickle_patcher.stop()
        self._patcher.stop()

    def test_api_invalid_and_answered(self):
        client = self.server.app.test_client()

        r = client.post("/api", json={"text": "hi"})
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertEqual(data.get("status"), "invalid")

        r2 = client.post("/api", json={"text": "test question"})
        self.assertEqual(r2.status_code, 200)
        d2 = r2.get_json()
        self.assertEqual(d2.get("status"), "answered")
        self.assertIn("answer", d2)

    def test_metrics_and_feedback(self):
        client = self.server.app.test_client()

        m = client.get("/metrics")
        self.assertEqual(m.status_code, 200)
        md = m.get_json()
        self.assertIn("api_requests", md)
        self.assertIn("faiss", md)

        # Avoid disk writes by stubbing training-related functions
        with patch.object(self.server, "add_training_pair", return_value=None), \
             patch.object(self.server, "handle_negative_feedback", return_value=None), \
             patch.object(self.server, "retrain_model", return_value=None):
            fb = client.post(
                "/feedback",
                json={"query": "q1", "answer": "a1", "feedback": "yes"},
            )
        self.assertEqual(fb.status_code, 200)
        self.assertEqual(fb.get_json().get("status"), "feedback_received")


if __name__ == "__main__":
    unittest.main(verbosity=2)
