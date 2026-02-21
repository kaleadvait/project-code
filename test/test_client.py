import os
import sys
import unittest
from unittest.mock import patch, MagicMock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text="OK"):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text

    def json(self):
        return self._json


class ClientTests(unittest.TestCase):
    def test_troubleshoot_cancel(self):
        import client.client as client_mod
        with patch("builtins.input", return_value="n"), patch("requests.post") as mock_post:
            client_mod.troubleshot()
            mock_post.assert_not_called()

    def test_chat_answered_and_feedback(self):
        import client.client as client_mod

        inputs = [
            "hello",  # first query
            "yes",    # feedback to answered
            "q",      # quit
        ]

        def post_side_effect(url, json=None, verify=None, timeout=None):
            if url.endswith("/api"):
                return FakeResponse(200, {"status": "answered", "answer": "42"})
            if url.endswith("/feedback"):
                return FakeResponse(200, {"status": "ok"})
            if url.endswith("/troubleshoot"):
                return FakeResponse(200, {"status": "done", "continue": False})
            return FakeResponse(404, {}, text="not found")

        with patch("builtins.input", side_effect=inputs), patch("requests.post", side_effect=post_side_effect) as mock_post:
            client_mod.chat()
            called_urls = [call.args[0] for call in mock_post.mock_calls if call.args]
            self.assertTrue(any(u.endswith("/api") for u in called_urls))
            self.assertTrue(any(u.endswith("/feedback") for u in called_urls))


if __name__ == "__main__":
    unittest.main(verbosity=2)
