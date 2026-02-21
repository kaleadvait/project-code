Final Project - Unified Guide (Server, Client, Training)

This guide explains how to set up the environment and run:
- The HTTPS server (server/server.py)
- The interactive client (client/client.ipynb)
- The training script (trainmasterdb/trainmodel.py)

Tested on Windows. Adjust commands for other OSes as needed.

====================================================================
PREREQUISITES
====================================================================
- Python 3.9–3.11 recommended
- pip (Python package manager)
- Internet access to download model weights for SentenceTransformers

Recommended Python packages:
  flask
  sentence-transformers
  faiss-cpu
  pypdf
  numpy
  scikit-learn
  requests
  jupyter

Create and activate a virtual environment (Windows PowerShell):
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1

Install dependencies:
  python -m pip install --upgrade pip
  pip install flask sentence-transformers faiss-cpu pypdf numpy scikit-learn requests jupyter

Notes:
- faiss-cpu is the FAISS package for CPU on Windows. If you already use a different FAISS build, keep it consistent.
- sentence-transformers will install PyTorch automatically (CPU build by default).

====================================================================
PROJECT LAYOUT (key folders/files)
====================================================================
- server/
    server.py
    client_index.faiss
    client_data.pkl
    input_data/  (PDF files used for fallback answers)
    certificate/ (server.crt, server.key)  [Expected by default; see SSL note below]
- client/
    client.ipynb  (interactive terminal-like client)
- masterdbcreation/
    master_data.pkl
    master_index.faiss
- trainmasterdb/
    trainmodel.py
    usertrained_modelfolder/
        client_data.pkl
        client_index.faiss
        training_data.pkl

====================================================================
USING MAKEFILES (optional)
====================================================================
Makefiles are provided in each module folder to streamline installs and runs:
- client/Makefile (run target starts client.py interactive chat)
- server/Makefile
- masterdbcreation/Makefile
- trainmasterdb/Makefile

Requirements:
- Windows users need a 'make' tool (e.g., via Git for Windows, MSYS2, or Chocolatey).
  If you don't have make, run the raw Python commands shown in the sections below.

Examples (from the project root):
- Client
    make -C client install
    make -C client run
- Server
    make -C server install
    make -C server run
- Master dataset creation
    make -C masterdbcreation install
    make -C masterdbcreation run
- Training
    make -C trainmasterdb install
    make -C trainmasterdb run

Tip: You can choose a specific Python by passing PY, e.g.:
    make -C server run PY=python3

====================================================================
1) RUN THE SERVER (HTTPS Flask API)
====================================================================
The server expects these files in server/ (or adjust paths in server.py):
- client_index.faiss
- client_data.pkl
- input_data/ (optional; put PDFs here for PDF-based fallback answers)

SSL certificates (required; HTTPS only by default):
- server/server.py loads certificate/server.crt and certificate/server.key relative to server/.
- If you already have certs, place them at: server/certificate/server.crt and server/certificate/server.key
- If your cert/key are elsewhere or have different names, update the two paths in server.py (at the bottom where ssl_context.load_cert_chain is called).
- The bundled client disables TLS verification (verify=False), so self-signed certs are fine for local dev.

Start the server:
  (from project root, with venv active)
  python server\server.py

Expected output:
  Server started:
  * Running on https://0.0.0.0:5000/ (Press CTRL+C to quit)

HTTP endpoints:
- POST https://127.0.0.1:5000/api        (body: {"text": "your question"})
- POST https://127.0.0.1:5000/feedback   (body: {"question","answer","feedback":"yes|no"})
- GET  https://127.0.0.1:5000/metrics
- POST https://127.0.0.1:5000/troubleshoot (body: {"consent":"yes|no|troubleshoot|Fix_Error|skip"})

====================================================================
2) RUN THE CLIENT (Jupyter Notebook)
====================================================================
The client notebook communicates with the server over HTTPS.

Steps:
1. Ensure the server is running on https://127.0.0.1:5000.
2. Start Jupyter:
     jupyter notebook
3. Open: client/client.ipynb
4. Run all cells. The notebook defines a terminal-like chat. Type your query at the prompt.
   - To trigger log troubleshooting, type: troubleshoot
   - To exit the chat, type: q or quit

Configuration:
- SERVER_URL is set to https://127.0.0.1:5000 in the notebook. Change it if your server runs elsewhere.
- The client sets verify=False for requests, which allows self-signed certs in local dev.

Optional quick API test from PowerShell (no notebook):
  $body = @{ text = "hello" } | ConvertTo-Json
  Invoke-RestMethod -Method Post -Uri https://127.0.0.1:5000/api -Body $body -ContentType 'application/json' -SkipCertificateCheck

====================================================================
3) TRAIN/RETRAIN MASTER DATASET (trainmodel.py)
====================================================================
The training script merges the master dataset with one or many client datasets, deduplicates, builds a new FAISS index, evaluates retrieval metrics, and writes outputs.

Defaults (no arguments):
- --org_dataset -> masterdbcreation\ (expects master_data.pkl, master_index.faiss)
- --usertrained_modelfolder -> trainmasterdb\usertrained_modelfolder\ (expects one or more *_data.pkl files; client_data.pkl is fine)
- --output_dir -> trainmasterdb\retrained_master\ (will be created if missing)
- --model_name -> all-MiniLM-L6-v2
- --test_ratio -> 0.2 (20% test split for metrics)
- --k -> 1,3,5 (top-k metrics)

Run with defaults:
  python trainmasterdb\trainmodel.py

Run with explicit arguments (example):
  python trainmasterdb\trainmodel.py ^
    --org_dataset "masterdbcreation" ^
    --usertrained_modelfolder "trainmasterdb\usertrained_modelfolder" ^
    --output_dir "trainmasterdb\retrained_master" ^
    --model_name "all-MiniLM-L6-v2" ^
    --test_ratio 0.2 ^
    --k "1,3,5" ^
    --latency_samples 50

Outputs (printed and saved under --output_dir):
- master_data.pkl
- master_index.faiss
- Console metrics including MRR, top-k accuracy, and latency stats

Using retrained outputs with the server:
Option A (rename/copy):
  Copy the files produced by retraining to the server folder and rename:
    retrained_master\master_data.pkl  -> server\client_data.pkl
    retrained_master\master_index.faiss -> server\client_index.faiss
  Then restart the server.

Option B (change server constants):
  In server/server.py, update:
    INDEX_FILE = "client_index.faiss"  -> set to your path/filename
    DATA_FILE  = "client_data.pkl"     -> set to your path/filename
  Then restart the server.

====================================================================
COMMON ISSUES & TROUBLESHOOTING
====================================================================
- ImportError: No module named 'faiss' or 'faiss-cpu'
  -> pip install faiss-cpu

- SSL/certificate errors from other clients
  -> The provided notebook disables cert verification. For other tools, either trust the self-signed cert or disable verification only in dev.

- FileNotFoundError (master_data.pkl / client_data.pkl / FAISS index)
  -> Ensure files exist at the expected paths. See layout and options above.

- Large model download time or slow first run
  -> SentenceTransformer downloads model weights on first use; allow time or pre-download on stable internet.

- FAISS index count mismatch warning on server start
  -> The server auto-rebuilds an index from questions if the .faiss is missing/invalid, then writes a new index to server/.

====================================================================
QUICK REFERENCE
====================================================================
Start venv (Windows):
  .\.venv\Scripts\Activate.ps1

Install packages:
  pip install flask sentence-transformers faiss-cpu pypdf numpy scikit-learn requests jupyter

Run server:
  python server\server.py

Launch notebook client:
  jupyter notebook (then open client/client.ipynb)

Run training:
  python trainmasterdb\trainmodel.py

====================================================================
SUPPORTING ENDPOINTS SUMMARY (Server)
====================================================================
- /api          (POST JSON: {"text": "..."}) -> returns answer/status/confidence
- /feedback     (POST JSON: {"question","answer","feedback":"yes|no"})
- /metrics      (GET) -> operational stats and data/index info
- /troubleshoot (POST JSON: {"consent":"yes|no|troubleshoot|Fix_Error|skip"}) -> guided log review

End of readme.
