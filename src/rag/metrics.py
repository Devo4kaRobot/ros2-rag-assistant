"""Метрики поиска. Фрагмент релевантен, если он с правильной страницы и задевает
хотя бы один правильный раздел (gold = список {page_id, anchor})."""
from __future__ import annotations


def is_relevant(chunk: dict, gold: list[dict]) -> bool:
    anchors = set(chunk["anchors"])
    return any(g["page_id"] == chunk["page_id"] and g["anchor"] in anchors for g in gold)


def recall_at_k(ranked: list[dict], gold: list[dict], k: int) -> float:
    """Hit rate: 1, если среди top-k есть хотя бы один релевантный фрагмент."""
    return float(any(is_relevant(c, gold) for c in ranked[:k]))


def mrr_at_k(ranked: list[dict], gold: list[dict], k: int = 10) -> float:
    for i, c in enumerate(ranked[:k], 1):
        if is_relevant(c, gold):
            return 1.0 / i
    return 0.0


def recall_at_budget(ranked: list[dict], gold: list[dict], budget_words: int = 1500) -> float:
    """Есть ли релевантный фрагмент среди первых фрагментов, суммарно укладывающихся в budget_words слов
    (первый фрагмент берётся всегда). Честное сравнение нарезок разной длины: столько текста реально
    уйдёт в контекст LLM."""
    used = 0
    for c in ranked:
        n = len(c["text"].split())
        if used > 0 and used + n > budget_words:
            break
        used += n
        if is_relevant(c, gold):
            return 1.0
    return 0.0
