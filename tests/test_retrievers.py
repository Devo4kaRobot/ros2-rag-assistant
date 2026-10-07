"""Тесты поиска на фиктивном энкодере (без скачивания моделей)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.embeddings import Embedder  # noqa: E402
from rag.retrievers import DenseRetriever, RerankRetriever, rrf_fuse  # noqa: E402

VOCAB = ["topic", "service", "colcon", "build", "qos", "launch"]


def fake_encode(texts):
    """Мешок слов по маленькому словарю + запоминаем, что пришло на вход."""
    fake_encode.seen.extend(texts)
    return np.array([[t.lower().count(w) + 1e-3 for w in VOCAB] for t in texts], dtype=float)


fake_encode.seen = []

CHUNKS = [
    {"chunk_id": "a", "text": "colcon build builds the workspace"},
    {"chunk_id": "b", "text": "a topic carries messages; topic list shows topics"},
    {"chunk_id": "c", "text": "qos settings of a topic"},
]


def test_e5_prefixes_are_added():
    fake_encode.seen.clear()
    e = Embedder("intfloat/multilingual-e5-base", encode_fn=fake_encode)
    e.encode_queries(["q"])
    e.encode_passages(CHUNKS)
    assert fake_encode.seen[0] == "query: q"
    assert all(s.startswith("passage: ") for s in fake_encode.seen[1:])


def test_no_prefix_for_other_models_and_ablation_flag():
    fake_encode.seen.clear()
    Embedder("BAAI/bge-m3", encode_fn=fake_encode).encode_queries(["q"])
    Embedder("intfloat/multilingual-e5-base", use_prefix=False, encode_fn=fake_encode).encode_queries(["q"])
    assert fake_encode.seen == ["q", "q"]


def test_passages_keep_original_order():
    # энкодер сортирует по длине для скорости — порядок векторов должен вернуться к исходному
    e = Embedder("BAAI/bge-m3", encode_fn=fake_encode)
    m = e.encode_passages(CHUNKS)
    direct = np.array(fake_encode([c["text"] for c in CHUNKS]))
    direct /= np.linalg.norm(direct, axis=1, keepdims=True)
    assert np.allclose(m, direct)


def test_dense_retriever_ranks_by_cosine():
    r = DenseRetriever(CHUNKS, model="x", embedder=Embedder("x", encode_fn=fake_encode))
    assert r.search("how to colcon build", k=1) == [0]
    assert r.search("list topic", k=3)[0] == 1


def test_rrf_fuse_prefers_consensus():
    # документ 2 на втором месте в обоих списках обгоняет лидеров каждого списка по отдельности
    assert rrf_fuse([[1, 2, 3], [4, 2, 5]], k=1)[0] == 2
    assert set(rrf_fuse([[1], [2]])) == {1, 2}


def test_rerank_reorders_candidates():
    class Base:
        def search(self, q, k):
            return [0, 1, 2][:k]

    r = RerankRetriever.__new__(RerankRetriever)
    r.chunks, r.base, r.top_n, r.batch_size = CHUNKS, Base(), 3, 8
    r.score_fn = lambda pairs: [len(t) for _, t in pairs]  # «длиннее = релевантнее»
    assert r.search("q", k=2) == [1, 0]  # длины: b=49 > a=33 > c=23


def test_weighted_rrf():
    # с весом 0 второй список не влияет
    assert rrf_fuse([[1, 2], [2, 1]], weights=[1.0, 0.0])[:2] == [1, 2]


def test_recall_at_budget():
    from rag.metrics import recall_at_budget
    gold = [{"page_id": "P", "anchor": "a"}]
    long = {"page_id": "Q", "anchors": ["x"], "text": "w " * 1400}
    hit = {"page_id": "P", "anchors": ["a"], "text": "w " * 200}
    assert recall_at_budget([long, hit], gold, 1500) == 0.0   # 1400 + 200 > 1500 — не влезло
    assert recall_at_budget([hit, long], gold, 1500) == 1.0
    assert recall_at_budget([{**hit, "text": "w " * 3000}], gold, 1500) == 1.0  # первый берётся всегда
