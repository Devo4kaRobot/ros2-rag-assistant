"""Собирает таблицу качества ответов из автоматических метрик и оценок → results/generation_report.md

    python scripts/make_report.py

Ожидает в results/generation/: answers_<split>_<model>[_<prompt>].jsonl и (если есть)
grades_<split>_<model>_<prompt>.json — оценки корректности и обоснованности по eval/grading_rubric.md.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "results" / "generation"


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def run_key(name: str):
    m = re.match(r"answers_(dev|test|all)_(.+?)(?:_(v\d+))?\.jsonl$", name)
    return (m.group(1), m.group(2), m.group(3) or "v1") if m else None


def bootstrap_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"),) * 2
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def main():
    rows = []
    for p in sorted(GEN.glob("answers_*.jsonl")):
        key = run_key(p.name)
        if not key:
            continue
        split, model, prompt = key
        ans = load_jsonl(p)
        A = [r for r in ans if r["answerable"]]
        N = [r for r in ans if not r["answerable"]]
        g_path = GEN / f"grades_{split}_{model}_{prompt}.json"
        grades = json.loads(g_path.read_text(encoding="utf-8")) if g_path.exists() else {}
        row = {"split": split, "model": model, "prompt": prompt, "n": f"{len(A)}+{len(N)}",
               "refusal_ok": f"{sum(r['refused'] for r in N)}/{len(N)}",
               "false_refusal": f"{sum(r['refused'] for r in A)}/{len(A)}",
               "citation_hit": np.mean([r["citation_hit"] for r in A if not r["refused"]] or [np.nan]),
               "gen_s": float(np.median([r["generation_s"] for r in ans]))}
        if grades:
            c = [grades[r["id"]]["correctness"] for r in A if r["id"] in grades]
            gr = [grades[r["id"]]["groundedness"] for r in A if r["id"] in grades]
            full = [x == 2 for x in c]
            lo, hi = bootstrap_ci(full)
            row.update(correct2=np.mean(full), correct2_ci=f"[{lo:.0%}; {hi:.0%}]", wrong0=np.mean([x == 0 for x in c]),
                       correctness=np.mean(c), groundedness=np.mean(gr),
                       neg_ok=f"{sum(grades[r['id']]['correctness'] == 2 for r in N if r['id'] in grades)}/{len(N)}")
        rows.append(row)
    if not rows:
        print("нет результатов"); return
    head = ("| Выборка | Модель | Промпт | Вопросов (отв.+неотв.) | Полностью верно | 95% ДИ | Ошибочно | "
            "Корректность 0–2 | Обоснованность 0–2 | Честный отказ (оценщик) | Отказ (детектор) | Ложный отказ | "
            "Ссылка на нужный раздел | Ответ, с |\n|" + "---|" * 14 + "\n")
    f = lambda v, pct=False: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.0%}" if pct else f"{v:.2f}")  # noqa: E731
    lines = [f"| {r['split']} | {r['model']} | {r['prompt']} | {r['n']} | {f(r.get('correct2'), True)} | "
             f"{r.get('correct2_ci', '—')} | {f(r.get('wrong0'), True)} | {f(r.get('correctness'))} | "
             f"{f(r.get('groundedness'))} | {r.get('neg_ok', '—')} | {r['refusal_ok']} | {r['false_refusal']} | "
             f"{f(r['citation_hit'], True)} | {r['gen_s']:.1f} |" for r in rows]
    text = "# Качество ответов\n\n" + head + "\n".join(lines) + "\n"
    (ROOT / "results" / "generation_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
