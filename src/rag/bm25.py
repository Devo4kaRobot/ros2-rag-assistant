"""BM25 (Okapi) на чистом Python + numpy — без внешних зависимостей."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

TOKEN_RE = re.compile(r"[a-zа-яё0-9_]+", re.I)
# минимальный стоп-лист: частые служебные слова, которые только шумят
STOP = set("""a an the and or of to in on for is are be it this that with as by at from you your can
how what which do does if not use using will we i
и в во не на что как с со по к ко у из за для это то а но или ли же бы мне мой как""".split())


def tokenize(text: str) -> list[str]:
    toks = [t.lower() for t in TOKEN_RE.findall(text)]
    out = []
    for t in toks:
        if t in STOP or len(t) < 2:
            continue
        out.append(t)
        if "_" in t:  # ros_domain_id → ещё и ros, domain, id
            out += [p for p in t.split("_") if len(p) > 1 and p not in STOP]
    return out


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs_tf = [Counter(tokenize(d)) for d in docs]
        self.len = np.array([sum(tf.values()) for tf in self.docs_tf], dtype=float)
        self.avgdl = self.len.mean()
        df = Counter()
        for tf in self.docs_tf:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        # инвертированный индекс: термин → [(doc, tf)]
        self.inv: dict[str, list[tuple[int, int]]] = {}
        for i, tf in enumerate(self.docs_tf):
            for t, c in tf.items():
                self.inv.setdefault(t, []).append((i, c))

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.docs_tf))
        norm = self.k1 * (1 - self.b + self.b * self.len / self.avgdl)
        for t in set(tokenize(query)):
            if t not in self.inv:
                continue
            idx, tf = zip(*self.inv[t])
            idx, tf = np.array(idx), np.array(tf, dtype=float)
            s[idx] += self.idf[t] * tf * (self.k1 + 1) / (tf + norm[idx])
        return s

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        s = self.scores(query)
        top = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in top if s[i] > 0]
