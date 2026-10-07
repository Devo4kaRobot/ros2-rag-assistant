"""Оценка поиска по конфигу. Каждая строка таблицы абляций = один запуск.

    python src/rag/eval_retrieval.py configs/bm25_structural.yaml [--split dev|test|all]

Результат дописывается в results/retrieval.csv, ответы по вопросам — в results/runs/<name>.jsonl.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rag.chunking import chunk_pages  # noqa: E402
from rag.metrics import mrr_at_k, recall_at_budget, recall_at_k  # noqa: E402
from rag.retrievers import build_retriever  # noqa: E402


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


BUDGET_K = 20  # для метрики по объёму контекста коротких фрагментов нужно больше 10


def _gpu_peak_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**20)
    except ImportError:
        pass
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    name = cfg["name"]

    pages = load_jsonl(ROOT / "data/processed/pages.jsonl")
    chunks = [c.to_dict() for c in chunk_pages(pages, cfg["chunking"])]
    qs = [q for q in load_jsonl(ROOT / "eval/questions.jsonl")
          if q["answerable"] and (args.split == "all" or q["split"] == args.split)]

    t0 = time.time()
    retriever = build_retriever(cfg["retriever"], chunks)
    build_s = time.time() - t0
    retriever.search("warm-up query", k=10)  # первый запрос (загрузка CUDA-ядер) не учитываем в задержке

    rows, lat = [], []
    field = cfg.get("query_field", "question")  # можно подставить переведённый вопрос
    for q in qs:
        t = time.time()
        idx = retriever.search(q.get(field) or q["question"], k=BUDGET_K)
        lat.append((time.time() - t) * 1000)
        ranked = [chunks[i] for i in idx]
        rows.append({
            "id": q["id"], "type": q["type"],
            "r5": recall_at_k(ranked, q["gold"], 5), "r10": recall_at_k(ranked, q["gold"], 10),
            "mrr": mrr_at_k(ranked, q["gold"], 10),
            "r1500w": recall_at_budget(ranked, q["gold"], 1500),
            "top": [c["chunk_id"] for c in ranked[:5]],
        })

    r5 = np.array([r["r5"] for r in rows]); r10 = np.array([r["r10"] for r in rows])
    mrr = np.array([r["mrr"] for r in rows])
    rb = np.array([r["r1500w"] for r in rows])
    lo, hi = bootstrap_ci(r5)
    summary = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "name": name, "split": args.split,
        "n_questions": len(rows), "chunking": cfg["chunking"], "n_chunks": len(chunks),
        "recall@5": round(r5.mean(), 3), "recall@5_ci95": f"[{lo:.2f}; {hi:.2f}]",
        "recall@10": round(r10.mean(), 3), "mrr@10": round(mrr.mean(), 3),
        "recall@1500w": round(rb.mean(), 3),
        "latency_ms_median": round(float(np.median(lat)), 1), "index_build_s": round(build_s, 1),
        "gpu_peak_mb": _gpu_peak_mb(),
    }
    by_type = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r["r5"])
    summary["recall@5_by_type"] = " ".join(f"{t}={np.mean(v):.2f}" for t, v in sorted(by_type.items()))
    print(json.dumps(summary, ensure_ascii=False, indent=1))

    (ROOT / "results/runs").mkdir(parents=True, exist_ok=True)
    with open(ROOT / f"results/runs/{name}_{args.split}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    append_row(ROOT / "results/retrieval.csv", summary)


def append_row(path: Path, row: dict) -> None:
    """Дописывает строку; если набор колонок изменился — переписывает файл с объединённым заголовком."""
    old_rows, fields = [], list(row)
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            old_rows = list(reader)
            old_fields = reader.fieldnames or []
        fields = old_fields + [k for k in row if k not in old_fields]
        if fields == old_fields:
            with path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(old_rows + [row])


if __name__ == "__main__":
    main()
