"""RAG-пайплайн: вопрос → поиск (bge-m3) → контекст в пределах бюджета → GigaChat → ответ со ссылками.

Решения:
- контекст набирается по бюджету слов (1500), а не фиксированным k: короткие фрагменты «по разделам»
  иначе дают слишком мало текста (см. Recall@1500 слов в шаге 2);
- LLM ставит номера фрагментов [n]; мы проверяем, что такие номера существуют, и превращаем их
  в ссылки docs.ros.org (невалидные номера считаем — это метрика качества);
- отказ — точная фраза REFUSAL; её легко распознать и посчитать долю правильных отказов.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field

from rag import settings

REFUSAL = "В документации ROS 2 я этого не нашёл."

PROMPT_V1 = f"""Ты — помощник по ROS 2 (дистрибутив Lyrical Luth). Отвечай на русском языке, опираясь ТОЛЬКО на фрагменты документации, которые даны в сообщении.

Правила:
1. Если во фрагментах нет ответа на вопрос, ответь ровно одной фразой: «{REFUSAL}» — и ничего не добавляй.
2. Команды, код, имена пакетов, параметров и переменных окружения копируй из фрагментов дословно, не переводи и не изменяй.
3. После каждого утверждения ставь номер фрагмента, на котором оно основано, в квадратных скобках, например [2] или [1][3].
4. Отвечай по делу: 2–6 предложений или короткий нумерованный список шагов. Код оформляй в обратных кавычках.
5. Не добавляй флаги, команды, версии и факты, которых нет во фрагментах. Если фрагменты отвечают на вопрос лишь частично, ответь на эту часть и прямо скажи, чего в документации нет.
6. Термины ROS не переводи дословно: пиши «топик», «нода», «сервис», «action», «launch-файл», «пакет», «воркспейс», «QoS» (не «тема», «узел», «услуга»)."""

PROMPT_V2 = f"""Ты — помощник по ROS 2, дистрибутив Lyrical Luth (Ubuntu 26.04 Resolute). Отвечай по-русски, опираясь ТОЛЬКО на фрагменты документации из сообщения. Своих знаний не используй, даже если уверен в ответе.

Как отвечать:
1. Найди во фрагментах ответ на главный вопрос пользователя.
   — Если ответа нет, напиши ровно одну фразу: «{REFUSAL}» — без пояснений, советов и ссылок.
   — Если есть только часть ответа, дай эту часть и закончи фразой «Остального в документации ROS 2 нет.»
2. Если во фрагменте пошаговая инструкция, перечисли все её шаги по порядку, ничего не пропуская (включая установку нужных пакетов и обновление списков пакетов).
3. Команды, флаги, имена пакетов, параметров и переменных окружения копируй из фрагментов дословно. Не придумывай флаги и не подставляй похожие. Числа и диапазоны бери только из фрагментов.
4. Если пример во фрагменте написан для другого дистрибутива ROS или другой версии Ubuntu, прямо скажи об этом, а не выдавай его за инструкцию для Lyrical.
5. После каждого утверждения ставь номер фрагмента в квадратных скобках, например [2]. Ставь номер только того фрагмента, в котором это утверждение действительно есть.
6. Термины ROS не переводи: «топик», «нода», «сервис», «action», «launch-файл», «пакет», «воркспейс», «QoS».
7. Пиши кратко: ответ по сути или нумерованный список шагов; команды и код — в обратных кавычках."""

# v1 → v2: исправления по итогам оценки ответов на dev (results/generation/grades_*):
# конфликт правил «отказ» и «частичный ответ», пропуск шагов, выдуманные флаги, примеры под другую Ubuntu,
# ссылки [n] на фрагменты без нужного утверждения.
PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}
DEFAULT_PROMPT = "v2"
SYSTEM_PROMPT = PROMPTS[DEFAULT_PROMPT]


@dataclass
class Source:
    n: int
    title: str
    section: str
    url: str
    chunk_id: str


@dataclass
class Answer:
    question: str
    answer: str
    refused: bool
    sources: list[Source] = field(default_factory=list)       # процитированные фрагменты
    context: list[Source] = field(default_factory=list)       # все фрагменты, ушедшие в LLM
    invalid_citations: list[int] = field(default_factory=list)
    retrieval_s: float = 0.0
    generation_s: float = 0.0
    model: str = ""
    prompt: str = ""
    refusal: str = ""          # "strict" | "soft" | ""

    def to_dict(self) -> dict:
        return asdict(self)


def select_by_budget(ranked: list[dict], budget_words: int, max_chunks: int) -> list[dict]:
    """Первые фрагменты, укладывающиеся в бюджет слов (первый берётся всегда)."""
    out, used = [], 0
    for c in ranked:
        n = len(c["text"].split())
        if out and (used + n > budget_words or len(out) >= max_chunks):
            break
        out.append(c)
        used += n
    return out


def build_user_prompt(question: str, chunks: list[dict]) -> str:
    parts = ["Фрагменты документации ROS 2:\n"]
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] (страница: {c['title']}; раздел: {c['section']})\n{c['text']}\n")
    parts.append(f"Вопрос: {question}")
    return "\n".join(parts)


CITE_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


def parse_citations(text: str, n_chunks: int) -> tuple[list[int], list[int]]:
    """Номера [n] из ответа: (валидные по порядку первого упоминания, невалидные)."""
    valid, invalid = [], []
    for m in CITE_RE.finditer(text):
        for num in re.split(r"\s*[,;]\s*", m.group(1)):
            k = int(num)
            target = valid if 1 <= k <= n_chunks else invalid
            if k not in target:
                target.append(k)
    return valid, invalid


SOFT_REFUSAL_MARKERS = ("нет информации", "не нашел", "не найдено", "не описан", "не указан", "отсутствует информация",
                        "не содержат информации", "не содержится")


def refusal_kind(text: str) -> str:
    """"strict" — точная фраза; "soft" — короткий ответ, который по сути отказ («нет информации о …»);
    "" — содержательный ответ. Частичный ответ с фразой «Остального в документации нет» — не отказ."""
    t = " ".join(text.strip().lower().replace("ё", "е").split())
    if t.startswith("в документации ros 2 я этого не нашел"):
        return "strict"
    first = re.split(r"(?<=[.!?])\s", t, maxsplit=1)[0]
    if len(t) < 300 and any(m in first for m in SOFT_REFUSAL_MARKERS):
        return "soft"
    return ""


def is_refusal(text: str) -> bool:
    return bool(refusal_kind(text))


def _source(n: int, c: dict) -> Source:
    return Source(n=n, title=c["title"], section=c["section"], url=c["url"], chunk_id=c["chunk_id"])


class RAGPipeline:
    def __init__(self, llm, retriever=None, chunks: list[dict] | None = None,
                 budget_words: int = settings.CONTEXT_BUDGET_WORDS, max_chunks: int = settings.MAX_CONTEXT_CHUNKS,
                 prompt: str = DEFAULT_PROMPT):
        self.llm = llm
        self.prompt_version, self.system_prompt = prompt, PROMPTS[prompt]
        self.budget_words, self.max_chunks = budget_words, max_chunks
        if retriever is None:
            settings.load_env()
            from rag.chunking import chunk_pages
            from rag.retrievers import build_retriever
            pages = [json.loads(l) for l in (settings.ROOT / "data/processed/pages.jsonl").open(encoding="utf-8")]
            chunks = [c.to_dict() for c in chunk_pages(pages, settings.CHUNKING)]
            retriever = build_retriever(settings.RETRIEVER, chunks)
        self.retriever, self.chunks = retriever, chunks
        self.retriever.search("warm-up", k=1)  # первый запрос к GPU — 10–30 с; делаем его при запуске, а не на пользователе

    def retrieve(self, question: str) -> list[dict]:
        ranked = [self.chunks[i] for i in self.retriever.search(question, k=settings.SEARCH_K)]
        return select_by_budget(ranked, self.budget_words, self.max_chunks)

    def answer(self, question: str) -> Answer:
        t0 = time.time()
        ctx = self.retrieve(question)
        t1 = time.time()
        res = self.llm.generate(self.system_prompt, build_user_prompt(question, ctx))
        text = res.text.strip()
        kind = refusal_kind(text)
        refused = bool(kind)
        valid, invalid = parse_citations(text, len(ctx))
        sources = [] if refused else [_source(k, ctx[k - 1]) for k in valid]
        return Answer(question=question, answer=text, refused=refused, sources=sources,
                      context=[_source(i, c) for i, c in enumerate(ctx, 1)], invalid_citations=invalid,
                      retrieval_s=round(t1 - t0, 3), generation_s=round(res.latency_s, 2), model=res.model,
                      prompt=self.prompt_version, refusal=kind)


def format_sources(ans: Answer) -> str:
    """Список источников для консоли: одна ссылка на URL, номера через запятую."""
    by_url: dict[str, list] = {}
    for s in ans.sources:
        by_url.setdefault(s.url, [s.section, []])[1].append(s.n)
    return "\n".join(f"  [{', '.join(map(str, ns))}] {sec}\n      {url}" for url, (sec, ns) in by_url.items())
