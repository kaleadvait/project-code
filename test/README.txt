Project tests for server, client, master dataset creation, and training utilities.

How to run
----------
- From this folder (test):
  1) Ensure you have Python 3 installed and can run `python` from the terminal.
  2) Run:  make test

- Alternatively, from anywhere in the repo without GNU make:
  - Run:  python -m unittest discover -s test -p "test_*.py" -v

Notes
-----
- Heavy dependencies (faiss, sentence_transformers, pypdf) are mocked in tests to avoid downloads and GPU/CPU requirements.
- Tests focus on utility functions and API behavior with fakes; they do not hit the network.
- If you prefer pytest, you can run:  pytest -q  (pytest is not required for these tests).
