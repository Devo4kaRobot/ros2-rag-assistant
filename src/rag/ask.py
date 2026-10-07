"""Задать вопрос из консоли.

    python src/rag/ask.py "как посмотреть список топиков?"
    python src/rag/ask.py            # интерактивный режим, выход — пустая строка
    python src/rag/ask.py --context "вопрос"   # показать, какие фрагменты ушли в LLM
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rag.llm import GigaChatLLM  # noqa: E402
from rag.pipeline import RAGPipeline, format_sources  # noqa: E402


def show(ans, with_context: bool):
    print("\n" + ans.answer)
    if ans.sources:
        print("\nИсточники:\n" + format_sources(ans))
    if ans.invalid_citations:
        print(f"\n(модель сослалась на несуществующие фрагменты: {ans.invalid_citations})")
    if with_context:
        print("\nКонтекст:")
        for s in ans.context:
            print(f"  [{s.n}] {s.section}  {s.url}")
    print(f"\n(поиск {ans.retrieval_s:.2f} с, ответ {ans.generation_s:.1f} с, модель {ans.model})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--context", action="store_true")
    args = ap.parse_args()
    print("Загружаю модель поиска и индекс…", flush=True)
    rag = RAGPipeline(GigaChatLLM())
    if args.question:
        show(rag.answer(" ".join(args.question)), args.context)
        return
    while True:
        q = input("\nВопрос (пусто — выход): ").strip()
        if not q:
            break
        show(rag.answer(q), args.context)


if __name__ == "__main__":
    main()
