
import os
import re
import pickle
import hashlib
from typing import List, Tuple

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


PDF_INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "input_files")
OUTPUT_DIR = os.path.dirname(__file__)

MASTER_DATA_FILE = os.path.join(OUTPUT_DIR, "master_data.pkl")
MASTER_INDEX_FILE = os.path.join(OUTPUT_DIR, "master_index.faiss")

QUESTIONS_TXT_FILE = os.path.join(OUTPUT_DIR, "questions.txt")
ANSWERS_TXT_FILE = os.path.join(OUTPUT_DIR, "answers.txt")
QA_PAIRS_TXT_FILE = os.path.join(OUTPUT_DIR, "qa_pairs.txt")

MODEL_NAME = "all-MiniLM-L6-v2"
MIN_UNIQUE_QUESTIONS = 200


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_into_chunks(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    text = _normalize_ws(text)
    if not text:
        return []
    if chunk_size <= overlap:
        overlap = 0

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def _extract_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    parts: List[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        t = _normalize_ws(t)
        if t:
            parts.append(t)
    return "\n".join(parts)


def _make_question(source_name: str, chunk: str, variant: int) -> str:
    base = chunk
    base = re.sub(r"\[[^\]]+\]", " ", base)
    base = _normalize_ws(base)
    words = base.split()
    phrase = " ".join(words[:14])
    if not phrase:
        phrase = source_name
    if variant == 0:
        return f"In Dell Data Protection Advisor (DPA), what does the documentation say about: {phrase}?"
    if variant == 1:
        return f"How is the following described in Dell Data Protection Advisor (DPA): {phrase}?"
    return f"What guidance does Dell Data Protection Advisor (DPA) provide regarding: {phrase}?"


def _dedupe_by_question(qa: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = {}
    for q, a in qa:
        k = _normalize_ws(q).lower()
        seen[k] = (q, a)
    return list(seen.values())


def _dedupe_answers(qa: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    out: List[Tuple[str, str]] = []
    for q, a in qa:
        key = hashlib.sha1(_normalize_ws(a).lower().encode("utf-8", errors="ignore")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append((q, a))
    return out


def build_master_from_pdfs() -> Tuple[List[str], List[str]]:
    if not os.path.isdir(PDF_INPUT_DIR):
        raise FileNotFoundError(f"PDF folder not found: {PDF_INPUT_DIR}")

    pdf_files = [
        os.path.join(PDF_INPUT_DIR, f)
        for f in os.listdir(PDF_INPUT_DIR)
        if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {PDF_INPUT_DIR}")

    qa: List[Tuple[str, str]] = []
    for pdf_path in sorted(pdf_files, key=lambda p: p.lower()):
        source_name = os.path.splitext(os.path.basename(pdf_path))[0]
        text = _extract_pdf_text(pdf_path)
        chunks = _split_into_chunks(text)

        for i, chunk in enumerate(chunks):
            chunk = _normalize_ws(chunk)
            if len(chunk) < 120:
                continue

            h = hashlib.md5(chunk.encode("utf-8", errors="ignore")).hexdigest()[:8]
            context = f"[{source_name}:{i}:{h}]"
            answer = f"{context} {chunk}"

            for variant in (0, 1, 2):
                q = _make_question(source_name, chunk, variant)
                qa.append((q, answer))

    qa = _dedupe_by_question(qa)
    qa = _dedupe_answers(qa)

    if len(qa) < MIN_UNIQUE_QUESTIONS:
        raise ValueError(
            f"Only generated {len(qa)} unique Q/A pairs. "
            f"Add more DPA PDFs or lower MIN_UNIQUE_QUESTIONS."
        )

    qa = qa[:MIN_UNIQUE_QUESTIONS]
    questions = [q for q, _ in qa]
    answers = [a for _, a in qa]
    return questions, answers


def _write_txt_files(questions: List[str], answers: List[str]) -> None:
    with open(QUESTIONS_TXT_FILE, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(_normalize_ws(q) + "\n")

    with open(ANSWERS_TXT_FILE, "w", encoding="utf-8") as f:
        for a in answers:
            f.write(_normalize_ws(a) + "\n")

    with open(QA_PAIRS_TXT_FILE, "w", encoding="utf-8") as f:
        for i, (q, a) in enumerate(zip(questions, answers), start=1):
            f.write(f"Q{i}: {_normalize_ws(q)}\n")
            f.write(f"A{i}: {_normalize_ws(a)}\n")
            f.write("\n")


def _build_and_save_faiss(questions: List[str]) -> faiss.Index:
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(questions, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def main() -> None:
    questions, answers = build_master_from_pdfs()
    _write_txt_files(questions, answers)

    index = _build_and_save_faiss(questions)
    with open(MASTER_DATA_FILE, "wb") as f:
        pickle.dump({"questions": questions, "answers": answers}, f)
    faiss.write_index(index, MASTER_INDEX_FILE)


if __name__ == "__main__":
    main()

