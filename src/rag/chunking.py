"""Три стратегии нарезки страниц на фрагменты (для таблицы абляций).

Размер — в словах (≈ 350 слов ≈ 450–500 токенов для английского текста), чтобы не
зависеть от токенайзера конкретной модели.

У каждого фрагмента есть список anchors — разделы страницы, с которыми он пересекается.
По (page_id, anchors) фрагмент сопоставляется с разметкой тестового набора, поэтому
разметку не надо переделывать при смене стратегии.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

MAX_WORDS = 350
FIXED_OVERLAP = 70
MIN_SECTION_WORDS = 50


@dataclass
class Chunk:
    chunk_id: str
    page_id: str
    title: str          # заголовок страницы
    section: str        # путь заголовков раздела, где начинается фрагмент
    url: str            # ссылка с якорем на раздел
    anchors: list[str]  # все разделы, которые задевает фрагмент
    text: str           # текст для индексации и для LLM

    def to_dict(self):
        return asdict(self)


def _page_stream(page: dict) -> tuple[str, list[tuple[int, int, dict]]]:
    """Склеивает страницу в один текст с заголовками; возвращает текст и [(start, end, section)]."""
    parts, spans, pos = [], [], 0
    for s in page["sections"]:
        block = f"## {s['titles'][-1]}\n\n{s['text']}".strip() + "\n\n"
        spans.append((pos, pos + len(block), s))
        parts.append(block)
        pos += len(block)
    return "".join(parts), spans


def _anchors_for(spans, start: int, end: int) -> list[dict]:
    return [s for a, b, s in spans if a < end and b > start]


def _url(page: dict, sec: dict | None) -> str:
    a = (sec or {}).get("html_anchor", (sec or {}).get("anchor", ""))
    return page["url"] + (f"#{a}" if a else "")


def _units(text: str) -> list[str]:
    """Абзацы; блок кода ``` ... ``` — одна неделимая единица (если не гигантский)."""
    units, buf, in_code = [], [], False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
            buf.append(line)
            if not in_code:
                units.append("\n".join(buf)); buf = []
            continue
        if in_code:
            buf.append(line); continue
        if not line.strip():
            if buf:
                units.append("\n".join(buf)); buf = []
        else:
            buf.append(line)
    if buf:
        units.append("\n".join(buf))
    return [u for u in units if u.strip()]


def _split_long(unit: str, max_words: int) -> list[str]:
    """Режет слишком длинный абзац или блок кода по строкам, затем по предложениям."""
    if len(unit.split()) <= max_words:
        return [unit]
    pieces = unit.split("\n") if "\n" in unit else re.split(r"(?<=[.!?])\s+", unit)
    out, cur = [], []
    for p in pieces:
        if cur and len(" ".join(cur + [p]).split()) > max_words:
            out.append("\n".join(cur)); cur = []
        if len(p.split()) > max_words:  # одна гигантская строка — режем по словам
            w = p.split()
            out += [" ".join(w[i:i + max_words]) for i in range(0, len(w), max_words)]
        else:
            cur.append(p)
    if cur:
        out.append("\n".join(cur))
    return out


def _pack(units: list[str], max_words: int) -> list[str]:
    out, cur, n = [], [], 0
    for u in units:
        for piece in _split_long(u, max_words):
            w = len(piece.split())
            if cur and n + w > max_words:
                out.append("\n\n".join(cur)); cur, n = [], 0
            cur.append(piece); n += w
    if cur:
        out.append("\n\n".join(cur))
    return out


def _make(page, idx, strategy, text, secs) -> Chunk:
    first = secs[0] if secs else None
    return Chunk(
        chunk_id=f"{strategy}:{page['page_id']}:{idx}",
        page_id=page["page_id"],
        title=page["title"],
        section=" > ".join(first["titles"]) if first else page["title"],
        url=_url(page, first),
        anchors=sorted({s["anchor"] for s in secs}),
        text=text.strip(),
    )


def chunk_fixed(page: dict, max_words: int = MAX_WORDS, overlap: int = FIXED_OVERLAP) -> list[Chunk]:
    text, spans = _page_stream(page)
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    out, step = [], max_words - overlap
    for i in range(0, max(len(words) - overlap, 1), step):
        w = words[i:i + max_words]
        if not w:
            break
        a, b = w[0][0], w[-1][1]
        out.append(_make(page, len(out), "fixed", text[a:b], _anchors_for(spans, a, b)))
    return out


def chunk_recursive(page: dict, max_words: int = MAX_WORDS) -> list[Chunk]:
    text, spans = _page_stream(page)
    out, cursor = [], 0
    for piece in _pack(_units(text), max_words):
        # находим позицию куска в исходном тексте, чтобы определить разделы
        probe = piece[:60]
        a = text.find(probe, cursor)
        a = cursor if a < 0 else a
        b = a + len(piece)
        cursor = a + 1
        out.append(_make(page, len(out), "recursive", piece, _anchors_for(spans, a, b)))
    return out


def chunk_structural(page: dict, max_words: int = MAX_WORDS, min_words: int = MIN_SECTION_WORDS) -> list[Chunk]:
    # 1) группы разделов: пустые и короткие разделы приклеиваем к следующему
    groups, cur, cur_words = [], [], 0
    for s in page["sections"]:
        w = len(s["text"].split())
        cur.append(s); cur_words += w
        if cur_words >= min_words:
            groups.append(cur); cur, cur_words = [], 0
    if cur:
        if groups and cur_words + sum(len(x["text"].split()) for x in groups[-1]) <= max_words:
            groups[-1] += cur
        else:
            groups.append(cur)
    # 2) каждую группу — в один или несколько фрагментов с префиксом пути заголовков
    out = []
    for g in groups:
        head = " > ".join(g[-1]["titles"] if not g[0]["text"].strip() else g[0]["titles"])
        body_parts = []
        for k, s in enumerate(g):
            if k > 0 or not s["text"].strip():
                body_parts.append(f"## {s['titles'][-1]}")
            if s["text"].strip():
                body_parts.append(s["text"])
        body = "\n\n".join(body_parts)
        pieces = _pack(_units(body), max_words - len(head.split()) - 2)
        for piece in pieces:
            out.append(_make(page, len(out), "structural", f"{head}\n\n{piece}", g))
    return out


STRATEGIES = {"fixed": chunk_fixed, "recursive": chunk_recursive, "structural": chunk_structural}


def chunk_pages(pages: list[dict], strategy: str) -> list[Chunk]:
    fn = STRATEGIES[strategy]
    return [c for p in pages for c in fn(p)]
