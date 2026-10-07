"""Поисковики с единым интерфейсом: search(query, k) -> список индексов фрагментов.

Конфиг (YAML, ключ retriever):
  {type: bm25}
  {type: dense, model: intfloat/multilingual-e5-base}
  {type: hybrid, rrf_k: 60, candidates: 50, components: [{type: bm25}, {type: dense, model: ...}]}
  {type: rerank, model: BAAI/bge-reranker-v2-m3, top_n: 20, base: {...любой из вышеперечисленных...}}
"""
from __future__ import annotations

import numpy as np

from rag.bm25 import BM25


class BM25Retriever:
    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.bm25 = BM25([c["text"] for c in chunks], k1=k1, b=b)

    def search(self, query: str, k: int = 10) -> list[int]:
        return [i for i, _ in self.bm25.search(query, k)]


class DenseRetriever:
    def __init__(self, chunks: list[dict], model: str, batch_size: int = 16, use_prefix: bool = True,
                 fp16: bool = False, embedder=None):
        from rag.embeddings import Embedder
        self.embedder = embedder or Embedder(model, batch_size=batch_size, use_prefix=use_prefix, fp16=fp16)
        self.matrix = self.embedder.encode_passages(chunks)

    def search(self, query: str, k: int = 10) -> list[int]:
        q = self.embedder.encode_queries([query])[0]
        scores = self.matrix @ q
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        return [int(i) for i in top[np.argsort(-scores[top])]]


def rrf_fuse(rankings: list[list[int]], k: int = 60, weights: list[float] | None = None) -> list[int]:
    """Reciprocal Rank Fusion: score(d) = Σ w_i / (k + rank_i). Шкалы BM25 и косинуса несравнимы,
    поэтому складываем не скоры, а ранги. Веса позволяют не давать слабому поиску равный голос."""
    weights = weights or [1.0] * len(rankings)
    score: dict[int, float] = {}
    for w, ranking in zip(weights, rankings):
        for r, doc in enumerate(ranking, 1):
            score[doc] = score.get(doc, 0.0) + w / (k + r)
    return [d for d, _ in sorted(score.items(), key=lambda x: (-x[1], x[0]))]


class HybridRetriever:
    def __init__(self, chunks: list[dict], components: list[dict], rrf_k: int = 60, candidates: int = 50,
                 weights: list[float] | None = None):
        self.parts = [build_retriever(c, chunks) for c in components]
        self.rrf_k, self.candidates, self.weights = rrf_k, candidates, weights

    def search(self, query: str, k: int = 10) -> list[int]:
        rankings = [p.search(query, self.candidates) for p in self.parts]
        return rrf_fuse(rankings, self.rrf_k, self.weights)[:k]


class RerankRetriever:
    """Кросс-энкодер переоценивает top_n кандидатов базового поиска."""

    def __init__(self, chunks: list[dict], base: dict, model: str, top_n: int = 20,
                 max_length: int = 512, batch_size: int = 8, fp16: bool = False, score_fn=None):
        self.chunks = chunks
        self.base = build_retriever(base, chunks)
        self.top_n, self.batch_size = top_n, batch_size
        self.score_fn = score_fn
        if score_fn is None:
            from sentence_transformers import CrossEncoder
            from rag.embeddings import device
            dev = device()
            self.ce = CrossEncoder(model, max_length=max_length, device=dev)
            if dev == "cuda" and fp16:  # на GTX 16xx fp16 в ~4 раза медленнее fp32 (см. bench_reranker.py)
                getattr(self.ce, "model", self.ce).half()

    def search(self, query: str, k: int = 10) -> list[int]:
        cand = self.base.search(query, self.top_n)
        if not cand:
            return []
        pairs = [(query, self.chunks[i]["text"]) for i in cand]
        if self.score_fn is not None:
            scores = np.asarray(self.score_fn(pairs))
        else:
            scores = np.asarray(self.ce.predict(pairs, batch_size=self.batch_size, show_progress_bar=False))
        order = np.argsort(-scores, kind="stable")
        return [cand[i] for i in order[:k]]


def build_retriever(cfg: dict, chunks: list[dict]):
    cfg = dict(cfg)
    kind = cfg.pop("type")
    if kind == "bm25":
        return BM25Retriever(chunks, **cfg)
    if kind == "dense":
        return DenseRetriever(chunks, **cfg)
    if kind == "hybrid":
        return HybridRetriever(chunks, **cfg)
    if kind == "rerank":
        return RerankRetriever(chunks, **cfg)
    raise ValueError(f"Неизвестный retriever: {kind}")
