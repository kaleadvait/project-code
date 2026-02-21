
import argparse
import os
import pickle
import random
import statistics
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class QaDataset:
    questions: List[str]
    answers: List[str]


def _resolve(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def _normalize_ws(text: str) -> str:
    return " ".join((text or "").split())


def _load_pickle_qa(path: str) -> QaDataset:
    with open(path, "rb") as f:
        data = pickle.load(f)
    questions = list(data.get("questions", []))
    answers = list(data.get("answers", []))
    if len(questions) != len(answers):
        raise ValueError(f"Invalid QA pickle (len mismatch) at: {path}")
    return QaDataset(questions=questions, answers=answers)


def _iter_client_data_files(folder: str) -> Iterable[str]:
    if not os.path.isdir(folder):
        return
    for root, _dirs, files in os.walk(folder):
        for name in files:
            lname = name.lower()
            if not lname.endswith(".pkl"):
                continue
            if lname in {"client_data.pkl", "master_data.pkl"} or lname.endswith("_data.pkl"):
                yield os.path.join(root, name)


def _dedupe_by_question(questions: Sequence[str], answers: Sequence[str]) -> QaDataset:
    seen: Dict[str, Tuple[str, str]] = {}
    for q, a in zip(questions, answers):
        qn = _normalize_ws(q).lower()
        if not qn:
            continue
        seen[qn] = (_normalize_ws(q), _normalize_ws(a))
    out_q = [v[0] for v in seen.values()]
    out_a = [v[1] for v in seen.values()]
    return QaDataset(out_q, out_a)


def _build_index(model: SentenceTransformer, questions: Sequence[str]) -> Tuple[faiss.Index, np.ndarray]:
    embeddings = model.encode(list(questions), show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings


def _split_indices(n: int, seed: int, test_ratio: float) -> Tuple[List[int], List[int]]:
    idx = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    test_n = max(1, int(round(n * test_ratio)))
    test_idx = idx[:test_n]
    train_idx = idx[test_n:]
    if not train_idx:
        train_idx = test_idx
    return train_idx, test_idx


def _evaluate_retrieval(
    model: SentenceTransformer,
    train_questions: Sequence[str],
    train_answers: Sequence[str],
    test_questions: Sequence[str],
    test_answers: Sequence[str],
    k_values: Sequence[int],
    n_latency_samples: int,
) -> Dict[str, float]:
    index, _train_emb = _build_index(model, train_questions)

    k_max = max(k_values)
    q_emb = model.encode(list(test_questions), show_progress_bar=False)
    q_emb = np.asarray(q_emb, dtype="float32")

    t0 = time.perf_counter()
    dists, nn = index.search(q_emb, k_max)
    search_ms = (time.perf_counter() - t0) * 1000.0

    topk_correct = {k: 0 for k in k_values}
    mrr_sum = 0.0
    for i in range(len(test_questions)):
        true_a = _normalize_ws(test_answers[i])
        ranks = []
        for rank, j in enumerate(nn[i].tolist(), start=1):
            if j < 0 or j >= len(train_answers):
                continue
            if _normalize_ws(train_answers[j]) == true_a:
                ranks.append(rank)
        best_rank = min(ranks) if ranks else None
        if best_rank is not None:
            mrr_sum += 1.0 / float(best_rank)
            for k in k_values:
                if best_rank <= k:
                    topk_correct[k] += 1

    n_test = max(1, len(test_questions))
    metrics: Dict[str, float] = {
        "test_samples": float(len(test_questions)),
        "search_total_ms": float(search_ms),
        "search_avg_ms_per_query": float(search_ms) / float(n_test),
        "mrr": float(mrr_sum) / float(n_test),
    }
    for k in k_values:
        metrics[f"top_{k}_accuracy"] = float(topk_correct[k]) / float(n_test)

    sample_n = min(n_latency_samples, len(test_questions))
    if sample_n > 0:
        sample_idx = list(range(len(test_questions)))
        random.shuffle(sample_idx)
        sample_idx = sample_idx[:sample_n]
        encode_lat = []
        search_lat = []
        for i in sample_idx:
            t1 = time.perf_counter()
            one_emb = model.encode([test_questions[i]], show_progress_bar=False)
            one_emb = np.asarray(one_emb, dtype="float32")
            encode_lat.append((time.perf_counter() - t1) * 1000.0)

            t2 = time.perf_counter()
            index.search(one_emb, k_max)
            search_lat.append((time.perf_counter() - t2) * 1000.0)

        metrics["encode_p50_ms"] = float(statistics.median(encode_lat))
        metrics["encode_p95_ms"] = float(np.percentile(np.asarray(encode_lat), 95))
        metrics["search_p50_ms"] = float(statistics.median(search_lat))
        metrics["search_p95_ms"] = float(np.percentile(np.asarray(search_lat), 95))

    _ = dists
    return metrics


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--org_dataset",
        default=_resolve(base_dir, "..\\masterdbcreation"),
        help="Folder containing master_data.pkl and master_index.faiss",
    )
    parser.add_argument(
        "--usertrained_modelfolder",
        default=_resolve(base_dir, "usertrained_modelfolder"),
        help="Folder containing one or many client *_data.pkl datasets",
    )
    parser.add_argument(
        "--output_dir",
        default=_resolve(base_dir, "retrained_master"),
        help="Where to write the retrained master_data.pkl and master_index.faiss",
    )
    parser.add_argument("--model_name", default="all-MiniLM-L6-v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--k", default="1,3,5")
    parser.add_argument("--latency_samples", type=int, default=50)
    args = parser.parse_args()

    org_dir = os.path.abspath(args.org_dataset)
    client_dir = os.path.abspath(args.usertrained_modelfolder)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    master_data_path = os.path.join(org_dir, "master_data.pkl")
    if not os.path.exists(master_data_path):
        raise FileNotFoundError(f"master_data.pkl not found in org_dataset: {master_data_path}")

    master = _load_pickle_qa(master_data_path)

    client_files = sorted(set(_iter_client_data_files(client_dir)))
    client_datasets: List[QaDataset] = []
    for p in client_files:
        try:
            ds = _load_pickle_qa(p)
            if ds.questions:
                client_datasets.append(ds)
        except Exception:
            continue

    merged_q = list(master.questions)
    merged_a = list(master.answers)
    for ds in client_datasets:
        merged_q.extend(ds.questions)
        merged_a.extend(ds.answers)

    merged = _dedupe_by_question(merged_q, merged_a)

    k_values = [int(x.strip()) for x in str(args.k).split(",") if x.strip()]
    k_values = sorted(set([k for k in k_values if k > 0])) or [1, 3, 5]

    model = SentenceTransformer(args.model_name)

    train_idx, test_idx = _split_indices(len(merged.questions), seed=int(args.seed), test_ratio=float(args.test_ratio))
    train_questions = [merged.questions[i] for i in train_idx]
    train_answers = [merged.answers[i] for i in train_idx]
    test_questions = [merged.questions[i] for i in test_idx]
    test_answers = [merged.answers[i] for i in test_idx]

    t_build0 = time.perf_counter()
    index, _ = _build_index(model, merged.questions)
    build_ms = (time.perf_counter() - t_build0) * 1000.0

    metrics = _evaluate_retrieval(
        model=model,
        train_questions=train_questions,
        train_answers=train_answers,
        test_questions=test_questions,
        test_answers=test_answers,
        k_values=k_values,
        n_latency_samples=int(args.latency_samples),
    )
    metrics["index_build_total_ms"] = float(build_ms)
    metrics["total_questions_after_merge"] = float(len(merged.questions))
    metrics["original_master_questions"] = float(len(master.questions))
    metrics["client_dataset_files_used"] = float(len(client_datasets))

    out_data = os.path.join(out_dir, "master_data.pkl")
    out_index = os.path.join(out_dir, "master_index.faiss")
    with open(out_data, "wb") as f:
        pickle.dump({"questions": merged.questions, "answers": merged.answers}, f)
    faiss.write_index(index, out_index)

    print("RETRAINING_RESULTS")
    for key in sorted(metrics.keys()):
        val = metrics[key]
        if key.endswith("_ms"):
            print(f"{key}: {val:.3f}")
        elif "accuracy" in key or key in {"mrr"}:
            print(f"{key}: {val:.4f}")
        else:
            if float(val).is_integer():
                print(f"{key}: {int(val)}")
            else:
                print(f"{key}: {val:.6f}")

    print(f"output_master_data: {out_data}")
    print(f"output_master_index: {out_index}")


if __name__ == "__main__":
    main()
