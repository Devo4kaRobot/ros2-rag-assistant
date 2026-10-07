"""Эмбеддинги фрагментов с кэшем на диске.

- multilingual-e5 ТРЕБУЕТ префиксы "query: " / "passage: " — без них качество заметно падает
  (частая молчаливая ошибка), поэтому префиксы ставятся здесь автоматически по имени модели.
- По умолчанию fp32. Замер на GTX 1650 показал, что fp16 там в ~4 раза МЕДЛЕННЕЕ (у GTX 16xx
  слабая половинная точность) и почти не экономит память. fp16 включается флагом fp16: true
  для карт с тензорными ядрами (RTX).
- Эмбеддинги корпуса считаются один раз и кэшируются в data/index/; ключ кэша включает хэш текстов,
  поэтому при изменении нарезки кэш пересчитается сам.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT / "data" / "index"

_MODELS: dict[str, object] = {}


def _prefixes(model_name: str) -> tuple[str, str]:
    if "e5" in model_name.lower():
        return "query: ", "passage: "
    return "", ""


def device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_model(model_name: str, max_seq_length: int = 512, fp16: bool = False):
    """SentenceTransformer, один экземпляр на процесс."""
    key = f"{model_name}|{fp16}"
    if key not in _MODELS:
        from sentence_transformers import SentenceTransformer
        dev = device()
        m = SentenceTransformer(model_name, device=dev)
        m.max_seq_length = max_seq_length
        if dev == "cuda" and fp16:
            m.half()
        _MODELS[key] = m
    return _MODELS[key]


def unload_all() -> None:
    """Освободить видеопамять перед загрузкой следующей модели."""
    _MODELS.clear()
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _cache_path(model_name: str, chunks: list[dict], tag: str = "") -> Path:
    h = hashlib.md5()
    for c in chunks:
        h.update(c["chunk_id"].encode()); h.update(c["text"].encode())
    slug = model_name.replace("/", "__")
    return INDEX_DIR / f"{slug}{tag}__{h.hexdigest()[:12]}.npy"


class Embedder:
    """Обёртка: encode_queries / encode_passages. Для тестов можно подменить encode_fn."""

    def __init__(self, model_name: str, batch_size: int = 16, use_prefix: bool = True, fp16: bool = False,
                 encode_fn=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.fp16 = fp16
        # use_prefix=False — только для абляции «что будет, если забыть префиксы e5»
        self.q_prefix, self.p_prefix = _prefixes(model_name) if use_prefix else ("", "")
        self.tag = ("" if use_prefix else "__noprefix") + ("__fp16" if fp16 else "__fp32")
        self._encode_fn = encode_fn

    def _encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        if self._encode_fn is not None:
            v = np.asarray(self._encode_fn(texts), dtype=np.float32)
        else:
            m = load_model(self.model_name, fp16=self.fp16)
            v = m.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=show_progress).astype(np.float32)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.clip(n, 1e-12, None)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self._encode([self.q_prefix + q for q in queries])

    def encode_passages(self, chunks: list[dict], use_cache: bool = True) -> np.ndarray:
        path = _cache_path(self.model_name, chunks, self.tag)
        if use_cache and self._encode_fn is None and path.exists():
            return np.load(path)
        # длинные фрагменты первыми: батчи одинаковой длины — меньше паддинга и пиков видеопамяти
        order = sorted(range(len(chunks)), key=lambda i: -len(chunks[i]["text"]))
        vecs = self._encode([self.p_prefix + chunks[i]["text"] for i in order], show_progress=True)
        out = np.empty_like(vecs)
        out[order] = vecs
        if use_cache and self._encode_fn is None:
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            np.save(path, out)
        return out
