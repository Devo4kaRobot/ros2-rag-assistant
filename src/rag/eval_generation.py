"""Оценка ответов на тестовом наборе.

    python src/rag/eval_generation.py                 # dev: 34 отвечаемых + 6 неотвечаемых
    python src/rag/eval_generation.py --limit 3       # быстрая проверка на 3 вопросах
    python src/rag/eval_generation.py --fresh         # начать заново (иначе продолжает с места остановки)
    python src/rag/eval_generation.py --model GigaChat-2   # другая модель — отдельные файлы результатов

Автоматические метрики (results/generation/summary_<split>.json):
- correct_refusal  — доля неотвечаемых вопросов, где бот честно отказался;
- false_refusal    — доля отвечаемых, где бот отказался, хотя ответ был в контексте или нет;
- context_hit      — правильный раздел попал в контекст (качество поиска на этих же вопросах);
- citation_hit     — бот сослался хотя бы на один правильный раздел;
- citation_valid   — доля ответов без ссылок на несуществующие фрагменты.
Ручная оценка: results/generation/manual_<split>.csv (открывается в Excel) — корректность и
обоснованность 0/1/2 по каждому ответу.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rag.metrics import is_relevant  # noqa: E402

OUT = ROOT / "results" / "generation"


def evaluate_row(q: dict, ans: dict, chunks_by_id: dict) -> dict:
    ctx = [chunks_by_id[s["chunk_id"]] for s in ans["context"]]
    cited = [chunks_by_id[s["chunk_id"]] for s in ans["sources"]]
    row = {
        "id": q["id"], "type": q["type"], "answerable": q["answerable"], "question": q["question"],
        "reference_answer": q["reference_answer"], "answer": ans["answer"], "refused": ans["refused"],
        "cited_urls": [s["url"] for s in ans["sources"]], "invalid_citations": ans["invalid_citations"],
        "context_ids": [s["chunk_id"] for s in ans["context"]],   # нужно оценщику обоснованности
        "cited_ids": [s["chunk_id"] for s in ans["sources"]],
        "context_hit": any(is_relevant(c, q["gold"]) for c in ctx) if q["answerable"] else None,
        "citation_hit": any(is_relevant(c, q["gold"]) for c in cited) if q["answerable"] else None,
        "retrieval_s": ans["retrieval_s"], "generation_s": ans["generation_s"], "model": ans["model"],
        "prompt": ans.get("prompt", ""), "refusal_kind": ans.get("refusal", ""),
    }
    return row


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    neg = [r for r in rows if not r["answerable"]]
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None  # noqa: E731
    return {
        "n_answerable": len(ans), "n_negative": len(neg),
        "correct_refusal": mean([r["refused"] for r in neg]),
        "correct_refusal_strict": mean([r.get("refusal_kind", "strict" if r["refused"] else "") == "strict" for r in neg]),
        "false_refusal": mean([r["refused"] for r in ans]),
        "false_refusal_with_context_hit": mean([r["refused"] for r in ans if r["context_hit"]]),
        "context_hit": mean([r["context_hit"] for r in ans]),
        "citation_hit": mean([r["citation_hit"] for r in ans if not r["refused"]]),
        "citation_valid": mean([not r["invalid_citations"] for r in rows]),
        "answers_without_citations": mean([not r["cited_urls"] for r in ans if not r["refused"]]),
        "generation_s_median": sorted(r["generation_s"] for r in rows)[len(rows) // 2] if rows else None,
        "model": rows[0]["model"] if rows else None,
    }


def write_manual_csv(rows: list[dict], path: Path):
    """utf-8-sig — чтобы Excel открыл кириллицу без «кракозябр»; разделитель ; — для русской локали Excel."""
    old = {}
    if path.exists():  # не затираем уже выставленные оценки
        with path.open(encoding="utf-8-sig", newline="") as f:
            old = {r["id"]: r for r in csv.DictReader(f, delimiter=";")}
    fields = ["id", "type", "question", "reference_answer", "answer", "refused",
              "correctness_0_2", "groundedness_0_2", "comment"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        w.writeheader()
        for r in rows:
            prev = old.get(r["id"], {})
            w.writerow({**{k: r[k] for k in fields[:6]},
                        **{k: prev.get(k, "") for k in fields[6:]}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--model", default=None, help="модель GigaChat; по умолчанию GIGACHAT_MODEL из .env")
    ap.add_argument("--prompt", default=None, help="версия системного промпта (v1, v2); по умолчанию текущая")
    args = ap.parse_args()

    from rag.llm import GigaChatLLM
    from rag.pipeline import DEFAULT_PROMPT, RAGPipeline

    qs = [json.loads(l) for l in (ROOT / "eval/questions.jsonl").open(encoding="utf-8") if l.strip()]
    qs = [q for q in qs if args.split == "all" or q["split"] == args.split][: args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    llm = GigaChatLLM(model=args.model)
    prompt = args.prompt or DEFAULT_PROMPT
    # у каждой модели и версии промпта свои файлы — так их можно сравнить (v1 — без суффикса, как исторически)
    tag = f"{args.split}_{llm.model}" + ("" if prompt == "v1" else f"_{prompt}")
    raw_path = OUT / f"answers_{tag}.jsonl"
    done = {}
    if raw_path.exists() and not args.fresh:
        done = {r["id"]: r for r in map(json.loads, raw_path.open(encoding="utf-8"))}
    todo = [q for q in qs if q["id"] not in done]
    print(f"Вопросов: {len(qs)}, уже есть ответы: {len(qs) - len(todo)}, осталось: {len(todo)}")

    rag = RAGPipeline(llm, prompt=prompt)
    chunks_by_id = {c["chunk_id"]: c for c in rag.chunks}
    mode = "w" if args.fresh else "a"
    with raw_path.open(mode, encoding="utf-8") as f:
        for i, q in enumerate(todo, 1):
            t = time.time()
            try:
                ans = rag.answer(q["question"]).to_dict()
            except Exception as e:  # noqa: BLE001 — один упавший вопрос не должен ронять весь прогон
                print(f"  {q['id']}: ошибка {type(e).__name__}: {str(e)[:200]}")
                continue
            row = evaluate_row(q, ans, chunks_by_id)
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            done[q["id"]] = row
            mark = "отказ" if row["refused"] else f"{len(row['cited_urls'])} ссыл."
            print(f"  [{i}/{len(todo)}] {q['id']} ({q['type']}): {mark}, {time.time() - t:.1f} с")

    rows = [done[q["id"]] for q in qs if q["id"] in done]
    summary = summarize(rows)
    (OUT / f"summary_{tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    write_manual_csv(rows, OUT / f"manual_{tag}.csv")
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"\nОтветы: {raw_path}\nТаблица для ручной оценки: {OUT / f'manual_{tag}.csv'}")


if __name__ == "__main__":
    main()
