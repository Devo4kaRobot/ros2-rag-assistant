"""Диагностика скорости реранкера: сколько секунд занимает переоценка N кандидатов.

    python src/rag/bench_reranker.py BAAI/bge-reranker-v2-m3
    python src/rag/bench_reranker.py cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

Сравнивает GPU fp16, GPU fp32 и CPU на реальных фрагментах корпуса.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rag.chunking import chunk_pages  # noqa: E402


def bench(model_name: str, device: str, half: bool, pairs, max_length: int, batch_size: int, reps: int = 3):
    import torch
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(model_name, max_length=max_length, device=device)
    if half:
        getattr(ce, "model", ce).half()
    ce.predict(pairs[:2], batch_size=batch_size)  # прогрев
    times = []
    for _ in range(reps):
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t)
    peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0
    del ce
    if device == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return statistics.median(times), peak


def main():
    import torch
    model = sys.argv[1] if len(sys.argv) > 1 else "BAAI/bge-reranker-v2-m3"
    pages = [json.loads(l) for l in open(ROOT / "data/processed/pages.jsonl", encoding="utf-8")]
    chunks = chunk_pages(pages, "recursive")[:20]
    pairs = [("как создать пакет на python в ros2?", c.text) for c in chunks]
    print(f"модель: {model}, пар: {len(pairs)}, torch {torch.__version__}, cuda: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    setups = []
    if torch.cuda.is_available():
        setups += [("cuda", True, 512, 8), ("cuda", True, 384, 8), ("cuda", False, 512, 8), ("cuda", True, 512, 2)]
    setups += [("cpu", False, 512, 8)]
    for dev, half, ml, bs in setups:
        try:
            t, peak = bench(model, dev, half, pairs, ml, bs, reps=1 if dev == "cpu" else 3)
            print(f"{dev:4} {'fp16' if half else 'fp32'} max_len={ml} batch={bs}: {t:6.2f} с на {len(pairs)} пар"
                  f"{f', видеопамять {peak:.0f} МБ' if peak else ''}")
        except Exception as e:  # noqa: BLE001
            print(f"{dev} {'fp16' if half else 'fp32'} max_len={ml} batch={bs}: ошибка {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
