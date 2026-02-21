
import os
import pickle
import ssl
import time
import warnings
from datetime import datetime
from collections import deque

import faiss
from flask import Flask, request
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

INDEX_FILE = "client_index.faiss"
DATA_FILE = "client_data.pkl"
UNANSWERED_FILE = "unanswered_questions.txt"
PDF_FOLDER = "input_data"
SIMILARITY_THRESHOLD = 0.6
PDF_SIMILARITY_THRESHOLD = 0.45
TRAINING_DATA_FILE = "training_data.pkl"
FEEDBACK_LOG_FILE = "feedback_log.txt"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def _looks_like_pickle(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            header = f.read(2)
        return header == b"\x80\x04"
    except Exception:
        return False


def load_faiss_index(index_path: str):
    resolved = _resolve_path(index_path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"FAISS index file not found: {resolved}")
    if _looks_like_pickle(resolved):
        raise RuntimeError(
            f"File '{resolved}' looks like a Python pickle, not a FAISS index. "
            "Make sure INDEX_FILE points to a real .faiss file produced by faiss.write_index()."
        )
    return faiss.read_index(resolved)


app = Flask(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")

_METRICS = {
    "started_at": time.time(),
    "api_requests": 0,
    "api_errors": 0,
    "api_answered": 0,
    "api_unanswered": 0,
    "api_invalid": 0,
    "latencies_ms": deque(maxlen=1000),
}


def build_faiss_index_from_questions(all_questions):
    embeddings = model.encode(all_questions)
    new_index = faiss.IndexFlatL2(embeddings.shape[1])
    new_index.add(embeddings)
    faiss.write_index(new_index, _resolve_path(INDEX_FILE))
    return new_index


try:
    from sklearn.base import InconsistentVersionWarning  # type: ignore

    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass


with open(_resolve_path(DATA_FILE), "rb") as f:
    _data = pickle.load(f)

questions = _data["questions"]
answers = _data["answers"]

try:
    index = load_faiss_index(INDEX_FILE)
    if getattr(index, "ntotal", None) is not None and index.ntotal != len(questions):
        raise RuntimeError(
            f"FAISS index count ({index.ntotal}) does not match questions count ({len(questions)})."
        )
except Exception as e:
    print(f"WARN: Using rebuilt FAISS index because '{_resolve_path(INDEX_FILE)}' is invalid: {e}")
    index = build_faiss_index_from_questions(questions)


def load_pdf_chunks():
    chunks = []
    pdf_folder = _resolve_path(PDF_FOLDER)
    if not os.path.exists(pdf_folder):
        return chunks

    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(pdf_folder, file))
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    chunks.append(text.strip())

    return chunks


def search_pdfs(query: str):
    pdf_folder = _resolve_path(PDF_FOLDER)
    if not os.path.exists(pdf_folder):
        return None

    for file in os.listdir(pdf_folder):
        if not file.endswith(".pdf"):
            continue

        reader = PdfReader(os.path.join(pdf_folder, file))
        for page in reader.pages:
            text = page.extract_text()
            if not text:
                continue

            if query.lower() in text.lower():
                return text[:500]

    return None


@app.post("/api")
def chat():
    t0 = time.perf_counter()
    try:
        payload = request.get_json(force=True) or {}
        query = str(payload.get("text", "")).strip()

        if len(query) < 3:
            _METRICS["api_requests"] += 1
            _METRICS["api_invalid"] += 1
            _METRICS["latencies_ms"].append((time.perf_counter() - t0) * 1000.0)
            return {"status": "invalid", "answer": "Please ask a more specific question.", "confidence": 0.0}

        embedding = model.encode([query])
        distances, indices = index.search(embedding, 1)
        similarity = 1 / (1 + distances[0][0])

        if similarity >= SIMILARITY_THRESHOLD:
            top_idx = int(indices[0][0])
            answer = answers[top_idx]
            _METRICS["api_requests"] += 1
            _METRICS["api_answered"] += 1
            _METRICS["latencies_ms"].append((time.perf_counter() - t0) * 1000.0)
            return {"status": "answered", "answer": answer, "confidence": float(similarity)}

        pdf_answer = search_pdfs(query)
        if pdf_answer:
            _METRICS["api_requests"] += 1
            _METRICS["api_answered"] += 1
            _METRICS["latencies_ms"].append((time.perf_counter() - t0) * 1000.0)
            return {"status": "pdf_answer", "answer": pdf_answer, "confidence": 0.4}

        unanswered_path = _resolve_path(UNANSWERED_FILE)
        existing_questions = set()
        if os.path.exists(unanswered_path):
            with open(unanswered_path, "r", encoding="utf-8") as f:
                existing_questions = {line.rstrip("\n") for line in f}

        if query not in existing_questions:
            with open(unanswered_path, "a", encoding="utf-8") as f:
                f.write(query + "\n")

        _METRICS["api_requests"] += 1
        _METRICS["api_unanswered"] += 1
        _METRICS["latencies_ms"].append((time.perf_counter() - t0) * 1000.0)
        return {
            "status": "unanswered",
            "answer": "Sorry, I do not have an answer for this question.",
            "confidence": float(similarity),
        }

    except Exception as e:
        _METRICS["api_requests"] += 1
        _METRICS["api_errors"] += 1
        _METRICS["latencies_ms"].append((time.perf_counter() - t0) * 1000.0)
        return {"status": "error", "answer": "Internal server error occurred.", "details": str(e)}


@app.get("/metrics")
def metrics():
    lat = list(_METRICS["latencies_ms"])
    lat_sorted = sorted(lat)

    def _pct(p: float):
        if not lat_sorted:
            return None
        k = int(round((p / 100.0) * (len(lat_sorted) - 1)))
        k = max(0, min(len(lat_sorted) - 1, k))
        return float(lat_sorted[k])

    return {
        "uptime_seconds": float(time.time() - _METRICS["started_at"]),
        "api_requests": int(_METRICS["api_requests"]),
        "api_answered": int(_METRICS["api_answered"]),
        "api_unanswered": int(_METRICS["api_unanswered"]),
        "api_invalid": int(_METRICS["api_invalid"]),
        "api_errors": int(_METRICS["api_errors"]),
        "latency_ms": {
            "count": int(len(lat_sorted)),
            "p50": _pct(50),
            "p90": _pct(90),
            "p95": _pct(95),
            "p99": _pct(99),
        },
        "data": {
            "questions": int(len(questions)),
            "answers": int(len(answers)),
        },
        "faiss": {
            "index_ntotal": int(getattr(index, "ntotal", -1)),
            "index_type": type(index).__name__,
        },
        "config": {
            "similarity_threshold": float(SIMILARITY_THRESHOLD),
            "pdf_similarity_threshold": float(PDF_SIMILARITY_THRESHOLD),
        },
    }


def load_training_data():
    training_path = _resolve_path(TRAINING_DATA_FILE)
    if os.path.exists(training_path):
        with open(training_path, "rb") as f:
            return pickle.load(f)
    return {"positive_examples": [], "negative_examples": []}


def save_training_data(training_data):
    with open(_resolve_path(TRAINING_DATA_FILE), "wb") as f:
        pickle.dump(training_data, f)


def add_training_pair(question, answer):
    training_data = load_training_data()
    training_data["positive_examples"].append(
        {"question": question, "answer": answer, "timestamp": str(datetime.now())}
    )
    save_training_data(training_data)
    with open(_resolve_path(FEEDBACK_LOG_FILE), "a", encoding="utf-8") as f:
        f.write(f"POSITIVE: {question} -> {answer}\n")


def handle_negative_feedback(question, answer):
    training_data = load_training_data()
    training_data["negative_examples"].append(
        {"question": question, "answer": answer, "timestamp": str(datetime.now())}
    )
    save_training_data(training_data)
    with open(_resolve_path(FEEDBACK_LOG_FILE), "a", encoding="utf-8") as f:
        f.write(f"NEGATIVE: {question} -> {answer}\n")


def retrain_model():
    global index, questions, answers
    try:
        training_data = load_training_data()
        positive_examples = training_data["positive_examples"]
        if len(positive_examples) < 5:
            return

        new_questions = questions.copy()
        new_answers = answers.copy()
        for example in positive_examples[-10:]:
            new_questions.append(example["question"])
            new_answers.append(example["answer"])

        index = build_faiss_index_from_questions(new_questions)
        questions = new_questions
        answers = new_answers

        with open(_resolve_path(DATA_FILE), "wb") as f:
            pickle.dump({"questions": new_questions, "answers": new_answers}, f)

        training_data["positive_examples"] = training_data["positive_examples"][-20:]
        save_training_data(training_data)
    except Exception as e:
        print(f"Error during model retraining: {e}")


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json(silent=True) or {}
    question = data.get("query")
    answer = data.get("answer")
    fb = data.get("feedback")

    if fb == "yes":
        add_training_pair(question, answer)
    elif fb == "no":
        handle_negative_feedback(question, answer)

    retrain_model()
    return {"status": "feedback_received"}


@app.route("/troubleshoot", methods=["POST"])
def troubleshoot():
    data = request.get_json(silent=True) or {}
    consent = str(data.get("consent", "")).strip().lower()

    if not hasattr(troubleshoot, "_log_lines"):
        troubleshoot._log_lines = None
        troubleshoot._position = None
        troubleshoot._log_file = "app.log"

    if troubleshoot._log_lines is None:
        if not os.path.exists(troubleshoot._log_file) or not os.path.isfile(troubleshoot._log_file):
            return {
                "status": "error",
                "message": f"Log file '{troubleshoot._log_file}' not found.",
                "continue": False,
            }
        with open(troubleshoot._log_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        troubleshoot._log_lines = list(reversed(lines))
        troubleshoot._position = 0

    if consent not in ("y", "yes"):
        troubleshoot._log_lines = None
        troubleshoot._position = None
        return {"status": "done", "message": "Troubleshooting stopped by user.", "continue": False}

    lines = troubleshoot._log_lines
    pos = troubleshoot._position or 0
    while pos < len(lines):
        line = lines[pos].rstrip("\n")
        pos += 1
        stripped = line.lstrip()
        if stripped.startswith("WARN") or stripped.startswith("ERR"):
            troubleshoot._position = pos
            return {"status": "ok", "message": "Next log entry:", "log": line, "continue": True}

    troubleshoot._log_lines = None
    troubleshoot._position = None
    return {"status": "done", "message": "No more WARN/ERR entries in log.", "continue": False}


if __name__ == "__main__":
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
    ssl_context.maximum_version = ssl.TLSVersion.TLSv1_2
    ssl_context.load_cert_chain("certificate/server.crt", "certificate/server.key")
    print("Server started:")
    app.run(host="0.0.0.0", port=5000, debug=True, ssl_context=ssl_context)
    print("Server stop:")
