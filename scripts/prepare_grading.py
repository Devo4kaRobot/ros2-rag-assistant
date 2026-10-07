"""Готовит пакеты для оценки ответов: вопрос, эталон, ответ бота и полный текст фрагментов контекста.

    python scripts/prepare_grading.py results/generation/answers_dev_GigaChat-3-Ultra.jsonl

Пишет <файл>.grading.md — его читает оценщик (человек или независимая LLM) по eval/grading_rubric.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rag.chunking import chunk_pages  # noqa: E402
from rag.settings import CHUNKING  # noqa: E402


def main(path: str):
    pages = [json.loads(l) for l in (ROOT / "data/processed/pages.jsonl").open(encoding="utf-8")]
    chunks = {c.chunk_id: c for c in chunk_pages(pages, CHUNKING)}
    rows = [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]
    out = [f"# Пакет для оценки: {Path(path).name}\n\nКритерии: eval/grading_rubric.md\n"]
    for r in rows:
        out.append(f"\n---\n\n## {r['id']} ({r['type']}, {'отвечаемый' if r['answerable'] else 'НЕОТВЕЧАЕМЫЙ'})\n")
        out.append(f"**Вопрос:** {r['question']}\n\n**Эталон:** {r['reference_answer']}\n")
        out.append(f"**Ответ бота:**\n\n{r['answer']}\n")
        out.append(f"**Отказ:** {r['refused']} · **ссылки на несуществующие фрагменты:** {r['invalid_citations']}\n")
        out.append("**Контекст, который получил бот:**\n")
        for i, cid in enumerate(r.get("context_ids", []), 1):
            c = chunks.get(cid)
            text = c.text if c else "(фрагмент не найден — корпус изменился?)"
            out.append(f"\n[{i}] `{cid}`\n```\n{text}\n```\n")
    dst = Path(path).with_suffix(".grading.md")
    dst.write_text("\n".join(out), encoding="utf-8")
    print(f"{len(rows)} ответов → {dst}")


if __name__ == "__main__":
    main(sys.argv[1])
